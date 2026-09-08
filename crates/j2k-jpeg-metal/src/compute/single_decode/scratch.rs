// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    buffers::{checked_copy_bytes_to_buffer_at, new_decode_plane_buffer, MetalBatchScratch},
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

pub(in crate::compute) struct ScratchDecodeBuffers {
    pub(in crate::compute) plane0: Buffer,
    pub(in crate::compute) plane1: Buffer,
    pub(in crate::compute) plane2: Buffer,
    pub(in crate::compute) status: Buffer,
    pub(in crate::compute) packet: ScratchPacketBuffers,
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
    let checkpoint_buffer =
        scratch.shared_buffer(device, "single_decode_checkpoints", total_checkpoint_bytes)?;
    for (index, checkpoint) in checkpoints.iter().copied().enumerate() {
        let checkpoint = JpegEntropyCheckpointHost::from(checkpoint);
        checked_copy_bytes_to_buffer_at(
            &checkpoint_buffer,
            index * checkpoint_bytes,
            JpegEntropyCheckpointHost::as_bytes(&checkpoint),
            "upload JPEG Metal entropy checkpoint",
        )?;
    }
    Ok(ScratchPacketBuffers {
        entropy: scratch.shared_buffer_with_bytes(device, "single_decode_entropy", entropy)?,
        restart_offsets: scratch.shared_buffer_with_slice(
            device,
            "single_decode_restart_offsets",
            restart_offsets,
        )?,
        checkpoints: checkpoint_buffer,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "the scratch preparation mirrors the three plane and packet inputs bound together"
)]
pub(in crate::compute) fn decode_buffers(
    scratch: &mut MetalBatchScratch,
    device: &DeviceRef,
    plane_len: usize,
    plane0_returned: bool,
    decode_threads: u32,
    entropy: &[u8],
    restart_offsets: &[u32],
    checkpoints: &[JpegEntropyCheckpointV1],
) -> Result<ScratchDecodeBuffers, Error> {
    // The fast decode kernels cover the active plane rectangles through the
    // bounded deposit helpers in shaders_decode_helpers.metal. Callers reject
    // any failed status before publishing output, so successful reuse does not
    // require preserving or clearing earlier plane contents.
    let plane0 = if plane0_returned {
        new_decode_plane_buffer(device, plane_len, true)?
    } else {
        scratch.private_buffer(device, "single_decode_y", plane_len)?
    };
    let plane1 = scratch.private_buffer(device, "single_decode_cb", plane_len)?;
    let plane2 = scratch.private_buffer(device, "single_decode_cr", plane_len)?;
    let packet = packet_buffers(scratch, device, entropy, restart_offsets, checkpoints)?;
    let status = status_buffer(scratch, device, decode_threads)?;
    Ok(ScratchDecodeBuffers {
        plane0,
        plane1,
        plane2,
        status,
        packet,
    })
}
