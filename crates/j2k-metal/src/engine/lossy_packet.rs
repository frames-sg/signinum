// SPDX-License-Identifier: MIT OR Apache-2.0

//! Packetization of HT output on the producing command buffer.
//! Ordered headers generate descriptors for the existing parallel payload copier.

use super::abi::{
    J2kHtEncodeStatus, J2kPacketBlock, J2kPacketEncodeStatus, J2kPacketPayloadCopyJob,
    J2kResidentPacketBlockParams, J2K_ENCODE_STATUS_OK,
};
use super::resident_packet_plan::{
    build_resident_packet_plan, ResidentBatchPacketPlan, ResidentBatchPacketPlanParams,
    ResidentPacketTile,
};
use super::{
    checked_buffer_read, checked_buffer_slice, copied_recyclable_shared_slice_buffer,
    new_compute_command_encoder, take_recyclable_private_buffer, zeroed_recyclable_shared_buffer,
    Buffer, CommandBufferRef, Error, J2kResidentPacketizationEncodeJob, MetalRuntime,
};
use crate::buffer_pool::PooledBuffer;
use crate::metal_types::prelude::*;

#[derive(Clone, Copy)]
pub(super) struct HtPacketInput<'a> {
    pub(super) jobs: &'a Buffer,
    pub(super) payload: &'a Buffer,
    pub(super) status: &'a Buffer,
    pub(super) count: usize,
    pub(super) capacity: usize,
}

pub(super) struct PacketOutput {
    buffer: Buffer,
    status: Buffer,
    capacity: usize,
}

impl PacketOutput {
    fn allocate(
        runtime: &MetalRuntime,
        capacity: usize,
        shared: &mut Vec<PooledBuffer>,
    ) -> Result<Self, Error> {
        let buffer =
            super::direct_scratch::take_recyclable_shared_buffer(runtime, capacity, shared)?;
        let status =
            zeroed_recyclable_shared_buffer(runtime, size_of::<J2kPacketEncodeStatus>(), shared)?;
        Ok(Self {
            buffer,
            status,
            capacity,
        })
    }

    /// Called only after the complete HT/header/copy command buffer has finished.
    pub(super) fn read(&self) -> Result<Vec<u8>, Error> {
        let status =
            checked_buffer_read::<J2kPacketEncodeStatus>(&self.status, "lossy packet status")?;
        if status.code != J2K_ENCODE_STATUS_OK {
            return Err(super::tier1_encode::encode_status_error(
                "lossy packetization",
                status.code,
                status.detail,
            ));
        }
        let len = status.data_len as usize;
        if len > self.capacity {
            return Err(packet_error());
        }
        checked_buffer_slice(&self.buffer, len, "lossy packet payload")
    }
}

fn packet_error() -> Error {
    Error::MetalKernel {
        message: "Metal lossy packet allocation or output capacity exceeded".into(),
    }
}

fn plan_packets(
    input: HtPacketInput<'_>,
    job: J2kResidentPacketizationEncodeJob<'_>,
) -> Result<ResidentBatchPacketPlan, Error> {
    if input.count != job.code_block_count as usize || job.num_layers != 1 {
        return Err(Error::MetalStateInvariant {
            state: "lossy HT packet topology",
            reason: "HT block count must match the single-layer packet job",
        });
    }
    let tile = ResidentPacketTile {
        resolution_count: job.resolution_count,
        num_layers: job.num_layers,
        component_count: job.component_count,
        code_block_count: job.code_block_count,
        available_code_blocks: input.count,
        packet_descriptors: job.packet_descriptors,
        resolutions: job.resolutions,
        codestream: None,
    };
    build_resident_packet_plan(
        &[tile],
        &[0],
        ResidentBatchPacketPlanParams {
            family_name: "lossy HTJ2K",
            block_coding_mode: 1,
            high_throughput: 1,
            code_block_style: 0x40,
        },
        |_, tile, header| {
            header
                .checked_mul(tile.packet_descriptors.len().max(tile.resolutions.len()))
                .and_then(|bytes| bytes.checked_add(input.capacity))
                .ok_or_else(packet_error)
        },
    )
}

pub(super) fn encode(
    runtime: &MetalRuntime,
    command: &CommandBufferRef,
    input: HtPacketInput<'_>,
    job: J2kResidentPacketizationEncodeJob<'_>,
    private: &mut Vec<PooledBuffer>,
    shared: &mut Vec<PooledBuffer>,
) -> Result<PacketOutput, Error> {
    let plan = plan_packets(input, job)?;
    let resolutions =
        copied_recyclable_shared_slice_buffer(runtime, &plan.packet_resolutions, shared)?;
    let subbands = copied_recyclable_shared_slice_buffer(runtime, &plan.packet_subbands, shared)?;
    let resident_blocks =
        copied_recyclable_shared_slice_buffer(runtime, &plan.resident_blocks, shared)?;
    let descriptors =
        copied_recyclable_shared_slice_buffer(runtime, &plan.packet_descriptors, shared)?;
    let packet_states = copied_recyclable_shared_slice_buffer(runtime, &plan.state_blocks, shared)?;
    let jobs = copied_recyclable_shared_slice_buffer(runtime, &plan.packet_jobs, shared)?;
    let block_bytes = plan
        .resident_blocks
        .len()
        .checked_mul(size_of::<J2kPacketBlock>())
        .ok_or_else(packet_error)?;
    let blocks = take_recyclable_private_buffer(runtime, block_bytes, private)?;
    let copies = take_recyclable_private_buffer(
        runtime,
        plan.packet_payload_copy_job_capacity_total
            .checked_mul(size_of::<J2kPacketPayloadCopyJob>())
            .ok_or_else(packet_error)?,
        private,
    )?;
    let header = take_recyclable_private_buffer(runtime, plan.header_capacity_total, private)?;
    let scratch = take_recyclable_private_buffer(
        runtime,
        plan.scratch_words_total
            .checked_mul(size_of::<u32>())
            .ok_or_else(packet_error)?,
        private,
    )?;
    let packet_output = PacketOutput::allocate(runtime, plan.packet_output_capacity_total, shared)?;
    let output = &packet_output.buffer;
    let status = &packet_output.status;
    let kernels = runtime.encode()?;
    let encoder = new_compute_command_encoder(command)?;
    encoder.setComputePipelineState(&kernels.packet_block_prepare_resident_ht);
    for (index, buffer) in [&resident_blocks, input.jobs, input.status, &blocks]
        .into_iter()
        .enumerate()
    {
        encoder.set_buffer(index as u64, Some(buffer), 0);
    }
    encoder.set_bytes(
        4,
        &J2kResidentPacketBlockParams {
            block_count: job.code_block_count,
            tier1_job_count: job.code_block_count,
        },
    );
    j2k_metal_support::dispatch_1d_pipeline(
        &encoder,
        &kernels.packet_block_prepare_resident_ht,
        u64::from(job.code_block_count),
    );
    encoder.endEncoding();

    let encoder = new_compute_command_encoder(command)?;
    encoder.setComputePipelineState(&kernels.packet_encode_batched);
    for (index, buffer) in [
        &resolutions,
        &subbands,
        &blocks,
        input.payload,
        output,
        &header,
        &scratch,
        &jobs,
        status,
        &descriptors,
        &packet_states,
        &copies,
    ]
    .into_iter()
    .enumerate()
    {
        encoder.set_buffer(index as u64, Some(buffer), 0);
    }
    j2k_metal_support::dispatch_single_thread(&encoder);
    encoder.endEncoding();
    super::lossless_prepare::dispatch_batched_packet_payload_copy(
        runtime,
        command,
        super::resident_tier1::J2kBatchedPacketPayloadCopyDispatch {
            payload_buffer: input.payload,
            packet_output_buffer: output,
            packet_job_buffer: &jobs,
            packet_status_buffer: status,
            packet_payload_copy_job_buffer: &copies,
            tile_count: 1,
            max_payload_copy_jobs_per_tile: plan.max_payload_copy_jobs_per_tile as u64,
            label: "lossy HT packet payload copy",
            signpost_name:
                crate::profile_env::SIGNPOST_ENCODE_HYBRID_HT_PAYLOAD_COPY_COMMAND_ENCODE,
        },
    )?;
    Ok(packet_output)
}

pub(super) fn magnitude_bound(
    statuses: &[J2kHtEncodeStatus],
    capacities: &[super::abi::J2kHtEncodeBatchJob],
    levels: &[u8],
) -> Result<u8, Error> {
    let mut bound = 8;
    for ((status, job), level) in statuses.iter().copied().zip(capacities).zip(levels) {
        super::tier1_encode::validate_ht_encoded_status(status, job.output_capacity as usize)?;
        bound = bound.max(j2k_native::htj2k_required_magnitude_bound(
            u64::from(status.detail),
            false,
            *level,
        ));
    }
    Ok(bound.min(74))
}
