// SPDX-License-Identifier: Apache-2.0

use crate::entropy::block::{decode_block_with_activity, CoefficientBlock};
use crate::entropy::sequential::PreparedDecodePlan;
use crate::error::{JpegError, MarkerKind};
use crate::internal::bit_reader::BitReader;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCheckpoint {
    pub mcu_index: u32,
    pub scan_offset: usize,
    pub bit_accumulator: u64,
    pub bits_buffered: u8,
    pub prev_dc: [i32; 4],
    pub expected_rst: u8,
}

pub(crate) fn build_checkpoint_plan(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    cadence_mcus: u32,
) -> Result<Vec<DeviceCheckpoint>, JpegError> {
    let total_mcus = total_mcus(plan);
    let cadence_mcus = cadence_mcus.max(1);
    let restart_interval = plan.restart_interval.filter(|&interval| interval > 0).map(u32::from);
    validate_scan_bytes(scan_bytes, restart_interval.is_some())?;

    let reader_bytes = terminated_scan_bytes(scan_bytes);

    let mut checkpoints = Vec::with_capacity(total_mcus as usize);
    let mut br = BitReader::new(&reader_bytes);
    let mut coeff = CoefficientBlock::default();
    let mut prev_dc = [0i32; 4];
    let mut expected_rst = 0u8;
    let mut mcus_since_restart = 0u32;

    checkpoints.push(snapshot_checkpoint(0, &br, prev_dc, expected_rst));

    for mcu_index in 0..total_mcus {
        if mcu_index > 0 {
            if let Some(restart) = restart_interval {
                if mcus_since_restart == restart {
                    let _ = br.ensure_bits(1);
                    let marker = br.take_marker().ok_or(JpegError::UnexpectedEoi {
                        mcu_at: mcu_index,
                        mcu_total: total_mcus,
                    })?;
                    let expected = 0xD0 | expected_rst;
                    if marker != expected {
                        return Err(JpegError::RestartMismatch {
                            offset: br.position(),
                            expected: expected_rst,
                            found: marker,
                        });
                    }
                    expected_rst = (expected_rst + 1) & 0x07;
                    br.reset_at_restart();
                    prev_dc.fill(0);
                    mcus_since_restart = 0;
                    checkpoints.push(snapshot_checkpoint(
                        mcu_index,
                        &br,
                        prev_dc,
                        expected_rst,
                    ));
                }
            } else if mcu_index.is_multiple_of(cadence_mcus) {
                checkpoints.push(snapshot_checkpoint(
                    mcu_index,
                    &br,
                    prev_dc,
                    expected_rst,
                ));
            }
        }

        decode_one_mcu(plan, &mut br, &mut coeff, &mut prev_dc)?;
        mcus_since_restart += 1;
    }

    match br.take_marker() {
        Some(0xd9) | None => {}
        Some(found) => {
            return Err(JpegError::UnexpectedMarker {
                offset: br.position().saturating_sub(2),
                expected: MarkerKind::Eoi,
                found,
            })
        }
    }

    Ok(checkpoints)
}

fn terminated_scan_bytes(scan_bytes: &[u8]) -> Vec<u8> {
    let mut reader_bytes = Vec::with_capacity(scan_bytes.len() + 2);
    reader_bytes.extend_from_slice(scan_bytes);
    if !reader_bytes.ends_with(&[0xff, 0xd9]) {
        if reader_bytes.last() == Some(&0xff) {
            reader_bytes.push(0xd9);
        } else {
            reader_bytes.extend_from_slice(&[0xff, 0xd9]);
        }
    }
    reader_bytes
}

fn validate_scan_bytes(scan_bytes: &[u8], allow_restart_markers: bool) -> Result<(), JpegError> {
    let mut index = 0usize;
    while index < scan_bytes.len() {
        if scan_bytes[index] != 0xff {
            index += 1;
            continue;
        }

        let marker_start = index;
        let next = index + 1;
        if next >= scan_bytes.len() {
            return Ok(());
        }

        match scan_bytes[next] {
            0x00 => index = next + 1,
            0xd0..=0xd7 if allow_restart_markers => index = next + 1,
            0xd9 => return Ok(()),
            found => {
                return Err(JpegError::UnexpectedMarker {
                    offset: marker_start,
                    expected: MarkerKind::Eoi,
                    found,
                })
            }
        }
    }

    Ok(())
}

fn snapshot_checkpoint(
    mcu_index: u32,
    br: &BitReader<'_>,
    prev_dc: [i32; 4],
    expected_rst: u8,
) -> DeviceCheckpoint {
    let snapshot = br.snapshot();
    DeviceCheckpoint {
        mcu_index,
        scan_offset: snapshot.pos,
        bit_accumulator: snapshot.acc,
        bits_buffered: snapshot.bits,
        prev_dc,
        expected_rst,
    }
}

fn decode_one_mcu(
    plan: &PreparedDecodePlan,
    br: &mut BitReader<'_>,
    coeff: &mut CoefficientBlock,
    prev_dc: &mut [i32; 4],
) -> Result<(), JpegError> {
    for component in &plan.components {
        let plane_index = component.output_index;
        for _ in 0..u32::from(component.h) * u32::from(component.v) {
            let _ = decode_block_with_activity(
                br,
                &component.dc_table,
                &component.ac_table,
                &mut prev_dc[plane_index],
                &component.quant,
                coeff,
            )?;
        }
    }
    Ok(())
}

fn total_mcus(plan: &PreparedDecodePlan) -> u32 {
    let mcu_width = u32::from(plan.sampling.max_h) * 8;
    let mcu_height = u32::from(plan.sampling.max_v) * 8;
    let mcus_per_row = plan.dimensions.0.div_ceil(mcu_width);
    let mcu_rows = plan.dimensions.1.div_ceil(mcu_height);
    mcus_per_row.saturating_mul(mcu_rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::Decoder;
    use crate::internal::bit_reader::{BitReader, BitReaderSnapshot};

    #[test]
    fn non_restart_checkpoints_resume_cleanly() {
        let bytes = grayscale_jpeg(24, 24);
        let decoder = Decoder::new(&bytes).expect("decoder");
        let plan = &decoder.plan;
        let scan_bytes = &decoder.bytes[plan.scan_offset..];
        let checkpoints = build_checkpoint_plan(plan, scan_bytes, 1).expect("checkpoints");
        let reader_bytes = terminated_scan_bytes(scan_bytes);

        for pair in checkpoints.windows(2) {
            let mut prev_dc = pair[0].prev_dc;
            let mut coeff = CoefficientBlock::default();
            let mut br = BitReader::from_snapshot(
                &reader_bytes,
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
    fn checkpoint_plan_rejects_non_eoi_terminal_marker() {
        let mut bytes = grayscale_jpeg(24, 24);
        let tail = bytes.len() - 1;
        bytes[tail] = 0xe0;

        let decoder = Decoder::new(&bytes).expect("decoder");
        let plan = &decoder.plan;
        let scan_bytes = &decoder.bytes[plan.scan_offset..];
        let err = build_checkpoint_plan(plan, scan_bytes, 1).expect_err("terminal APPn must fail");

        assert!(matches!(
            err,
            JpegError::UnexpectedMarker {
                expected: MarkerKind::Eoi,
                found: 0xe0,
                ..
            }
        ));
    }

    fn grayscale_jpeg(width: u16, height: u16) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&[0xff, 0xd8]);
        bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
        bytes.extend(std::iter::repeat_n(16u8, 64));
        bytes.extend_from_slice(&[
            0xff,
            0xc0,
            0x00,
            11,
            8,
            (height >> 8) as u8,
            height as u8,
            (width >> 8) as u8,
            width as u8,
            1,
            1,
            0x11,
            0,
        ]);
        bytes.extend_from_slice(&[
            0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        bytes.extend_from_slice(&[
            0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, 0, 63, 0]);

        let mcu_cols = u32::from(width).div_ceil(8);
        let mcu_rows = u32::from(height).div_ceil(8);
        let mcu_count = (mcu_cols * mcu_rows) as usize;
        for _ in 0..mcu_count {
            bytes.push(0x00);
        }

        bytes.extend_from_slice(&[0xff, 0xd9]);
        bytes
    }
}
