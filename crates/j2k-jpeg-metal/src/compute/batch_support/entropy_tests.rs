// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn empty_checkpoint() -> JpegEntropyCheckpointV1 {
    JpegEntropyCheckpointV1 {
        mcu_index: 0,
        entropy_pos: 0,
        bit_acc: 0,
        bit_count: 0,
        y_prev_dc: 0,
        cb_prev_dc: 0,
        cr_prev_dc: 0,
        reserved: 0,
    }
}

#[test]
fn batch_entropy_shape_mismatch_fails_before_owner_growth() {
    let first_entropy = [1_u8];
    let second_entropy = [2_u8];
    let first_checkpoints = [empty_checkpoint()];
    let second_checkpoints = [empty_checkpoint()];
    let result = batch_entropy_metadata(
        [&first_entropy[..], &second_entropy[..]].into_iter(),
        [&first_checkpoints[..], &second_checkpoints[..]].into_iter(),
        2,
        2,
        0,
        BatchEntropyLabels {
            offset: "test offset",
            len: "test length",
        },
    );
    let Err(error) = result else {
        panic!("checkpoint count mismatch must fail");
    };

    assert!(matches!(
        error,
        Error::MetalKernel { message }
            if message == "JPEG Metal batch entropy metadata shape mismatch"
    ));
}

#[test]
fn batch_entropy_metadata_preserves_payload_layout_and_checkpoints() {
    let first_entropy = [1_u8, 2];
    let second_entropy = [3_u8, 4, 5];
    let mut first_checkpoint = empty_checkpoint();
    first_checkpoint.entropy_pos = 1;
    let mut second_checkpoint = empty_checkpoint();
    second_checkpoint.mcu_index = 4;
    let first_checkpoints = [first_checkpoint];
    let second_checkpoints = [second_checkpoint];

    let metadata = batch_entropy_metadata(
        [&first_entropy[..], &second_entropy[..]].into_iter(),
        [&first_checkpoints[..], &second_checkpoints[..]].into_iter(),
        2,
        1,
        0,
        BatchEntropyLabels {
            offset: "test offset",
            len: "test length",
        },
    )
    .expect("entropy metadata")
    .expect("non-empty entropy");

    assert_eq!(metadata.payload_len, 5);
    assert_eq!(metadata.offsets, [0, 2]);
    assert_eq!(metadata.lens, [2, 3]);
    assert_eq!(metadata.checkpoints.len(), 2);
    assert_eq!(metadata.checkpoints[0].entropy_pos, 1);
    assert_eq!(metadata.checkpoints[1].mcu_index, 4);
}

#[test]
fn batch_entropy_metadata_accepts_empty_payload_without_host_allocation() {
    let checkpoint = [empty_checkpoint()];

    assert!(batch_entropy_metadata(
        std::iter::once(&[][..]),
        std::iter::once(checkpoint.as_slice()),
        1,
        1,
        0,
        BatchEntropyLabels {
            offset: "test offset",
            len: "test length",
        },
    )
    .expect("empty entropy")
    .is_none());
}

#[test]
fn batch_entropy_metadata_keeps_typed_collective_host_limit() {
    let entropy = [1_u8];
    let checkpoint = [empty_checkpoint()];
    let metadata_bytes =
        2 * core::mem::size_of::<u32>() + core::mem::size_of::<JpegEntropyCheckpointHost>();
    let exact_external_live = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES - metadata_bytes;
    let make_metadata = |external_live_bytes| {
        batch_entropy_metadata(
            std::iter::once(entropy.as_slice()),
            std::iter::once(checkpoint.as_slice()),
            1,
            1,
            external_live_bytes,
            BatchEntropyLabels {
                offset: "test offset",
                len: "test length",
            },
        )
    };

    make_metadata(exact_external_live).expect("exact metadata host limit");
    assert!(matches!(
        make_metadata(exact_external_live + 1),
        Err(Error::BatchInfrastructure(
            j2k_core::BatchInfrastructureError::AllocationTooLarge {
                what: "JPEG Metal batch entropy host data",
                ..
            }
        ))
    ));
}

#[test]
fn retained_batch_entropy_updates_layout_and_checkpoint_contents() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    crate::compute::runtime::with_runtime(|runtime| {
        let mut scratch = MetalBatchScratch::default();
        let keys = BatchEntropyBufferKeys {
            payload: "test payload",
            offsets: "test offsets",
            lens: "test lengths",
            checkpoints: "test checkpoints",
        };
        for (index, reverse) in [false, false, true].into_iter().enumerate() {
            let payloads: [&[u8]; 2] = if reverse {
                [b"cde", b"ab"]
            } else {
                [b"ab", b"cde"]
            };
            let mut checkpoint = empty_checkpoint();
            checkpoint.entropy_pos = if reverse { 2 } else { 1 };
            let checkpoints = [checkpoint];
            let metadata = batch_entropy_metadata(
                payloads.into_iter(),
                [checkpoints.as_slice(); 2].into_iter(),
                2,
                1,
                0,
                BatchEntropyLabels {
                    offset: "offset",
                    len: "length",
                },
            )?
            .unwrap();
            crate::buffers::take_jpeg_scratch_upload_bytes_for_test();
            let buffers = batch_entropy_buffers_from_metadata(
                runtime,
                &mut scratch,
                keys,
                payloads.into_iter(),
                &metadata,
            )?;
            let uploaded = crate::buffers::take_jpeg_scratch_upload_bytes_for_test();
            if index == 1 {
                assert_eq!(uploaded, 0, "repeat batch must reuse all four uploads");
            } else {
                assert!(uploaded >= 5, "changed entropy must be uploaded");
            }
            assert_eq!(
                crate::buffers::checked_buffer_slice::<u8>(&buffers.payload, 5, "payload")?,
                if reverse { b"cdeab" } else { b"abcde" }
            );
            assert_eq!(
                crate::buffers::checked_buffer_slice::<u32>(&buffers.offsets, 2, "offsets")?,
                metadata.offsets
            );
            assert_eq!(
                crate::buffers::checked_buffer_slice::<u32>(&buffers.lens, 2, "lengths")?,
                metadata.lens
            );
            let staged = crate::buffers::checked_buffer_slice::<JpegEntropyCheckpointHost>(
                &buffers.checkpoints,
                2,
                "checkpoints",
            )?;
            assert_eq!(staged[0].entropy_pos, checkpoint.entropy_pos);
        }
        Ok(())
    })
    .unwrap();
}
