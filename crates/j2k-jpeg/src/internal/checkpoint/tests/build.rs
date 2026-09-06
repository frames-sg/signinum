// SPDX-License-Identifier: MIT OR Apache-2.0

use super::fixtures::grayscale_jpeg;
use crate::decoder::Decoder;
use crate::entropy::block::CoefficientBlock;
use crate::error::{JpegError, MarkerKind};
use crate::internal::bit_reader::{BitReader, BitReaderSnapshot};
use crate::internal::checkpoint::{
    build_checkpoint_plan, build_checkpoint_plan_with_cap, checkpoint_count_summary,
    decode_one_mcu, skip_one_mcu, snapshot_checkpoint, terminated_scan_bytes, total_mcus,
    DeviceCheckpoint,
};

#[test]
fn non_restart_checkpoints_resume_cleanly() {
    let bytes = grayscale_jpeg(24, 24);
    let decoder = Decoder::new(&bytes).expect("decoder");
    let plan = &decoder.plan;
    let scan_bytes = &decoder.bytes[plan.scan_offset..];
    let checkpoints = build_checkpoint_plan(plan, scan_bytes, 1).expect("checkpoints");
    let reader_bytes = terminated_scan_bytes(scan_bytes).expect("terminated scan bytes");

    for pair in checkpoints.windows(2) {
        let mut prev_dc = pair[0].prev_dc;
        let mut coeff = CoefficientBlock::default();
        let mut br = BitReader::from_snapshot(
            reader_bytes.as_ref(),
            BitReaderSnapshot {
                pos: pair[0].scan_offset,
                acc: pair[0].bit_accumulator,
                bits: pair[0].bits_buffered,
            },
        );

        decode_one_mcu(plan, &mut br, &mut coeff, &mut prev_dc).expect("decode one mcu");
        let resumed = snapshot_checkpoint(pair[1].mcu_index, &br, prev_dc, pair[0].expected_rst);
        assert_eq!(resumed.scan_offset, pair[1].scan_offset);
        assert_eq!(resumed.bit_accumulator, pair[1].bit_accumulator);
        assert_eq!(resumed.bits_buffered, pair[1].bits_buffered);
        assert_eq!(resumed.prev_dc, pair[1].prev_dc);
        assert_eq!(resumed.expected_rst, pair[1].expected_rst);
    }
}

#[test]
fn checkpoint_entropy_skip_matches_full_block_decode_state() {
    for bytes in [
        grayscale_jpeg(24, 24),
        j2k_test_support::minimal_baseline_420_jpeg(),
    ] {
        let decoder = Decoder::new(&bytes).expect("decoder");
        let plan = &decoder.plan;
        let scan_bytes = &decoder.bytes[plan.scan_offset..];
        let reader_bytes = terminated_scan_bytes(scan_bytes).expect("terminated scan bytes");
        let total_mcus = total_mcus(plan);
        let mut decoded_reader = BitReader::new(reader_bytes.as_ref());
        let mut skipped_reader = BitReader::new(reader_bytes.as_ref());
        let mut decoded_prev_dc = [0i32; 4];
        let mut skipped_prev_dc = [0i32; 4];
        let mut coeff = CoefficientBlock::default();

        for _ in 0..total_mcus {
            decode_one_mcu(plan, &mut decoded_reader, &mut coeff, &mut decoded_prev_dc)
                .expect("full block decode");
            skip_one_mcu(plan, &mut skipped_reader, &mut skipped_prev_dc)
                .expect("checkpoint entropy skip");

            assert_eq!(skipped_reader.snapshot(), decoded_reader.snapshot());
            assert_eq!(skipped_prev_dc, decoded_prev_dc);
        }
    }
}

#[test]
fn checkpoint_entropy_skip_preserves_full_decode_errors() {
    let bytes = grayscale_jpeg(8, 8);
    let decoder = Decoder::new(&bytes).expect("decoder");
    let plan = &decoder.plan;

    for first_entropy_byte in u8::MIN..=u8::MAX {
        let scan = [first_entropy_byte, 0xff, 0xd9];
        let mut decoded_reader = BitReader::new(&scan);
        let mut skipped_reader = BitReader::new(&scan);
        let mut decoded_prev_dc = [0i32; 4];
        let mut skipped_prev_dc = [0i32; 4];
        let mut coeff = CoefficientBlock::default();

        let decoded = decode_one_mcu(plan, &mut decoded_reader, &mut coeff, &mut decoded_prev_dc);
        let skipped = skip_one_mcu(plan, &mut skipped_reader, &mut skipped_prev_dc);

        assert_eq!(skipped, decoded, "entropy byte {first_entropy_byte:#04x}");
        assert_eq!(
            skipped_reader.snapshot(),
            decoded_reader.snapshot(),
            "entropy byte {first_entropy_byte:#04x}"
        );
        assert_eq!(
            skipped_prev_dc, decoded_prev_dc,
            "entropy byte {first_entropy_byte:#04x}"
        );
    }
}

#[test]
fn checkpoint_count_tracks_the_actual_checkpoint_interval() {
    let total_mcus = u32::MAX;
    let cadence_mcus = total_mcus.div_ceil(2_048);
    let non_restart = checkpoint_count_summary(total_mcus, cadence_mcus, None);

    assert_eq!(
        non_restart,
        usize::try_from(total_mcus.div_ceil(cadence_mcus)).expect("count fits usize")
    );
    assert!(non_restart <= 2_048);
    assert_eq!(
        checkpoint_count_summary(total_mcus, cadence_mcus, Some(1)),
        usize::try_from(total_mcus).expect("u32 fits usize on supported targets")
    );
    assert_eq!(checkpoint_count_summary(10, 9, Some(3)), 4);
    assert_eq!(checkpoint_count_summary(12, 9, Some(3)), 4);
    assert_eq!(checkpoint_count_summary(1, 0, None), 1);
    assert_eq!(checkpoint_count_summary(0, 0, None), 1);
}

#[test]
fn checkpoint_plan_applies_the_combined_cap_to_a_missing_eoi_copy() {
    let bytes = grayscale_jpeg(8, 8);
    let decoder = Decoder::new(&bytes).expect("decoder");
    let plan = &decoder.plan;
    let terminated_scan = &decoder.bytes[plan.scan_offset..];
    let scan_without_eoi = &terminated_scan[..terminated_scan.len() - 2];
    let requested = core::mem::size_of::<DeviceCheckpoint>() + scan_without_eoi.len() + 2;

    let error = build_checkpoint_plan_with_cap(plan, scan_without_eoi, 1, requested - 1)
        .expect_err("D+T must be checked together before either allocation");
    assert_eq!(
        error,
        JpegError::MemoryCapExceeded {
            requested,
            cap: requested - 1,
        }
    );

    let checkpoints = build_checkpoint_plan_with_cap(plan, scan_without_eoi, 1, requested)
        .expect("D+T exact boundary must decode");
    assert_eq!(checkpoints.len(), 1);
}

#[test]
fn checkpoint_plan_rejects_non_eoi_terminal_marker() {
    let mut bytes = grayscale_jpeg(24, 24);
    let tail = bytes.len() - 1;
    bytes[tail] = 0xe0;

    let decoder = Decoder::new(&bytes).expect("decoder");
    let plan = &decoder.plan;
    let scan_bytes = &decoder.bytes[plan.scan_offset..];
    let error = build_checkpoint_plan(plan, scan_bytes, 1).expect_err("terminal APPn must fail");
    assert!(matches!(
        error,
        JpegError::UnexpectedMarker {
            expected: MarkerKind::Eoi,
            found: 0xe0,
            ..
        }
    ));
}
