// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    buffers::MetalBatchScratch,
    metal_types::{Buffer, DeviceRef},
    Error,
};
use j2k_core::accelerator::GpuAbi;
use j2k_jpeg::adapter::JpegEntropyCheckpointV1;

use super::super::{JpegDecodeStatus, JpegEntropyCheckpointHost};

pub(in crate::compute) struct ScratchPacketBuffers {
    pub(in crate::compute) entropy: Buffer,
    pub(in crate::compute) restart_offsets: Buffer,
    pub(in crate::compute) checkpoints: Buffer,
}

pub(in crate::compute) fn status_buffer(
    scratch: &mut MetalBatchScratch,
    device: &DeviceRef,
    count: u32,
) -> Result<Buffer, Error> {
    let bytes = crate::batch_allocation::checked_count_product(
        count as usize,
        core::mem::size_of::<JpegDecodeStatus>(),
        "JPEG Metal single decode status bytes",
    )?;
    scratch.shared_zeroed_buffer(device, "single_decode_status", bytes)
}

pub(in crate::compute) fn packet_buffers(
    scratch: &mut MetalBatchScratch,
    device: &DeviceRef,
    entropy: &[u8],
    restart_offsets: &[u32],
    checkpoints: &[JpegEntropyCheckpointV1],
) -> Result<ScratchPacketBuffers, Error> {
    if restart_offsets.is_empty() {
        return Err(Error::MetalKernel {
            message: "JPEG Metal restart offsets must contain at least one entry".to_string(),
        });
    }
    if checkpoints.is_empty() {
        return Err(Error::MetalKernel {
            message: "JPEG Metal entropy checkpoints must contain at least one entry".to_string(),
        });
    }
    let checkpoint_bytes = core::mem::size_of::<JpegEntropyCheckpointHost>();
    let total_checkpoint_bytes = crate::batch_allocation::checked_count_product(
        checkpoints.len(),
        checkpoint_bytes,
        "JPEG Metal entropy checkpoint upload bytes",
    )?;
    let checkpoint_buffer = scratch.shared_immutable_buffer_with_byte_slices(
        device,
        "single_decode_checkpoints",
        total_checkpoint_bytes,
        checkpoints
            .iter()
            .copied()
            .map(|checkpoint| CheckpointBytes(checkpoint.into())),
    )?;
    Ok(ScratchPacketBuffers {
        entropy: scratch.shared_immutable_buffer_with_byte_slices(
            device,
            "single_decode_entropy",
            entropy.len(),
            [entropy],
        )?,
        restart_offsets: scratch.shared_immutable_buffer_with_slice(
            device,
            "single_decode_restart_offsets",
            restart_offsets,
        )?,
        checkpoints: checkpoint_buffer,
    })
}

// Convert each ABI checkpoint on the stack while replaying the upload iterator.
// This avoids a temporary checkpoint vector on both cache hits and misses.
#[derive(Clone)]
struct CheckpointBytes(JpegEntropyCheckpointHost);

impl AsRef<[u8]> for CheckpointBytes {
    fn as_ref(&self) -> &[u8] {
        JpegEntropyCheckpointHost::as_bytes(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_packet_retains_uploads_and_replaces_changed_checkpoints() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        let device = j2k_metal_support::system_default_device().expect("Metal device");
        let mut scratch = MetalBatchScratch::default();
        let mut checkpoint = JpegEntropyCheckpointV1 {
            mcu_index: 0,
            entropy_pos: 0,
            bit_acc: 0,
            bit_count: 0,
            y_prev_dc: 0,
            cb_prev_dc: 0,
            cr_prev_dc: 0,
            reserved: 0,
        };
        for index in 0..3 {
            if index == 2 {
                checkpoint.y_prev_dc = -7;
            }
            crate::buffers::take_jpeg_scratch_upload_bytes_for_test();
            let buffers =
                packet_buffers(&mut scratch, &device, b"entropy", &[0], &[checkpoint]).unwrap();
            let written = crate::buffers::take_jpeg_scratch_upload_bytes_for_test();
            assert_eq!(
                written,
                match index {
                    0 => 7 + 4 + core::mem::size_of::<JpegEntropyCheckpointHost>(),
                    1 => 0,
                    _ => core::mem::size_of::<JpegEntropyCheckpointHost>(),
                }
            );
            let staged = crate::buffers::checked_buffer_read::<JpegEntropyCheckpointHost>(
                &buffers.checkpoints,
                "checkpoint",
            )
            .unwrap();
            assert_eq!(staged.y_prev_dc, checkpoint.y_prev_dc);
        }
    }
}
