// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::metal_types::{Buffer, CommandBuffer};
#[cfg(target_os = "macos")]
use j2k_native::J2kPacketizationPacketDescriptor;

use super::abi::{
    split_u64_words, J2kBatchedCodestreamAssemblyJob, J2kBatchedPacketEncodeJob,
    J2kPacketDescriptor, J2kPacketResolution, J2kPacketStateBlock, J2kPacketSubband,
    J2kResidentPacketBlock,
};
use super::{
    encode_capacity::{
        codestream_progression_order_code, lossless_codestream_assembly_capacity,
        lossless_codestream_payload_offset, packet_tree_node_count,
    },
    J2kLosslessCodestreamAssemblyJob, J2kLosslessDeviceCodeBlock,
    J2kPreparedLosslessDeviceCodeBlocks, J2kResidentBatchEncodeItem,
    J2kResidentPacketizationResolution,
};
use crate::Error;

pub(super) struct PreparedLosslessBatchTile {
    pub(super) coefficient_buffer: Buffer,
    pub(super) coefficient_byte_offset: usize,
    pub(super) coefficient_byte_len: usize,
    pub(super) coefficient_buffer_is_batch_shared: bool,
    pub(super) code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
    pub(super) recyclable_private_buffers: Vec<crate::buffer_pool::PooledBuffer>,
    pub(super) prepare_command_buffer: CommandBuffer,
    pub(super) prepare_deinterleave_rct_command_buffer: Option<CommandBuffer>,
    pub(super) prepare_dwt53_command_buffer: Option<CommandBuffer>,
    pub(super) prepare_dwt53_vertical_command_buffers: Vec<CommandBuffer>,
    pub(super) prepare_dwt53_horizontal_command_buffers: Vec<CommandBuffer>,
    pub(super) prepare_coefficient_extract_command_buffer: Option<CommandBuffer>,
    pub(super) deinterleave_status_buffer: Buffer,
    pub(super) plane_buffers: Vec<Buffer>,
    pub(super) scratch_buffers: Vec<Buffer>,
    pub(super) coefficient_job_buffer: Buffer,
    pub(super) resolution_count: u32,
    pub(super) num_layers: u8,
    pub(super) component_count: u8,
    pub(super) code_block_count: u32,
    pub(super) packet_descriptors: Vec<J2kPacketizationPacketDescriptor>,
    pub(super) resolutions: Vec<J2kResidentPacketizationResolution>,
    pub(super) codestream: J2kLosslessCodestreamAssemblyJob,
}

/// Moves resident batch encode items into the family-neutral per-tile form
/// shared by the classic and HT batch drivers.
pub(super) fn prepared_lossless_batch_tiles(
    items: Vec<J2kResidentBatchEncodeItem>,
) -> Result<Vec<PreparedLosslessBatchTile>, Error> {
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal resident packet batch plan");
    let mut prepared_tiles =
        budget.try_vec(items.len(), "J2K Metal resident packet prepared tiles")?;
    for item in items {
        let J2kPreparedLosslessDeviceCodeBlocks {
            coefficient_buffer,
            coefficient_byte_offset,
            coefficient_byte_len,
            coefficient_buffer_is_batch_shared,
            code_blocks,
            recyclable_private_buffers,
            _prepare_command_buffer: prepare_command_buffer,
            _prepare_deinterleave_rct_command_buffer: prepare_deinterleave_rct_command_buffer,
            _prepare_dwt53_command_buffer: prepare_dwt53_command_buffer,
            _prepare_dwt53_vertical_command_buffers: prepare_dwt53_vertical_command_buffers,
            _prepare_dwt53_horizontal_command_buffers: prepare_dwt53_horizontal_command_buffers,
            _prepare_coefficient_extract_command_buffer: prepare_coefficient_extract_command_buffer,
            _deinterleave_status_buffer: deinterleave_status_buffer,
            _plane_buffers: plane_buffers,
            _scratch_buffers: scratch_buffers,
            _coefficient_job_buffer: coefficient_job_buffer,
        } = item.prepared;
        prepared_tiles.push(PreparedLosslessBatchTile {
            coefficient_buffer,
            coefficient_byte_offset,
            coefficient_byte_len,
            coefficient_buffer_is_batch_shared,
            code_blocks,
            recyclable_private_buffers,
            prepare_command_buffer,
            prepare_deinterleave_rct_command_buffer,
            prepare_dwt53_command_buffer,
            prepare_dwt53_vertical_command_buffers,
            prepare_dwt53_horizontal_command_buffers,
            prepare_coefficient_extract_command_buffer,
            deinterleave_status_buffer,
            plane_buffers,
            scratch_buffers,
            coefficient_job_buffer,
            resolution_count: item.resolution_count,
            num_layers: item.num_layers,
            component_count: item.component_count,
            code_block_count: item.code_block_count,
            packet_descriptors: item.packet_descriptors,
            resolutions: item.resolutions,
            codestream: item.codestream,
        });
    }
    Ok(prepared_tiles)
}

/// Per-family constants for the shared resident batch packet planner; values
/// reproduce each family's original literals so diagnostics and GPU job
/// fields stay byte-identical.
#[derive(Clone, Copy)]
pub(super) struct ResidentBatchPacketPlanParams {
    pub(super) family_name: &'static str,
    pub(super) block_coding_mode: u32,
    pub(super) high_throughput: u32,
    pub(super) code_block_style: u32,
}

pub(super) struct ResidentBatchPacketPlan {
    pub(super) packet_resolutions: Vec<J2kPacketResolution>,
    pub(super) packet_subbands: Vec<J2kPacketSubband>,
    pub(super) resident_blocks: Vec<J2kResidentPacketBlock>,
    pub(super) packet_descriptors: Vec<J2kPacketDescriptor>,
    pub(super) state_blocks: Vec<J2kPacketStateBlock>,
    pub(super) packet_jobs: Vec<J2kBatchedPacketEncodeJob>,
    pub(super) assembly_jobs: Vec<J2kBatchedCodestreamAssemblyJob>,
    pub(super) packet_output_capacity_total: usize,
    pub(super) packet_payload_copy_job_capacity_total: usize,
    pub(super) max_payload_copy_jobs_per_tile: usize,
    pub(super) header_capacity_total: usize,
    pub(super) scratch_words_total: usize,
    pub(super) codestream_capacity_total: usize,
    pub(super) codestream_offsets: Vec<usize>,
    pub(super) codestream_capacities: Vec<usize>,
}

/// Packet topology independent of transform buffers and command ownership.
/// `codestream=None` emits packet bytes for native assembly (including lossy headers).
pub(super) struct ResidentPacketTile<'a> {
    pub(super) resolution_count: u32,
    pub(super) num_layers: u8,
    pub(super) component_count: u8,
    pub(super) code_block_count: u32,
    // Bound descriptor ranges by the actual retained Tier-1 table, not the declared count.
    pub(super) available_code_blocks: usize,
    pub(super) packet_descriptors: &'a [J2kPacketizationPacketDescriptor],
    pub(super) resolutions: &'a [J2kResidentPacketizationResolution],
    pub(super) codestream: Option<J2kLosslessCodestreamAssemblyJob>,
}

/// Preserve the lossless batch driver's ownership while borrowing its packet topology.
pub(super) fn build_resident_batch_packet_plan(
    prepared_tiles: &[PreparedLosslessBatchTile],
    tile_tier1_job_bases: &[usize],
    params: ResidentBatchPacketPlanParams,
    tile_packet_output_capacity: impl Fn(
        usize,
        &PreparedLosslessBatchTile,
        usize,
    ) -> Result<usize, Error>,
) -> Result<ResidentBatchPacketPlan, Error> {
    let mut tiles =
        crate::batch_allocation::try_vec(prepared_tiles.len(), "resident packet topology views")?;
    tiles.extend(prepared_tiles.iter().map(|tile| ResidentPacketTile {
        resolution_count: tile.resolution_count,
        num_layers: tile.num_layers,
        component_count: tile.component_count,
        code_block_count: tile.code_block_count,
        available_code_blocks: tile.code_blocks.len(),
        packet_descriptors: &tile.packet_descriptors,
        resolutions: &tile.resolutions,
        codestream: Some(tile.codestream),
    }));
    build_resident_packet_plan(&tiles, tile_tier1_job_bases, params, |index, _, header| {
        tile_packet_output_capacity(index, &prepared_tiles[index], header)
    })
}

/// Build packet metadata for resident Classic/HT payloads, with optional GPU assembly.
#[expect(
    clippy::too_many_lines,
    reason = "single-pass packet planning keeps descriptor offsets and capacities consistent"
)]
pub(super) fn build_resident_packet_plan(
    prepared_tiles: &[ResidentPacketTile<'_>],
    tile_tier1_job_bases: &[usize],
    params: ResidentBatchPacketPlanParams,
    tile_packet_output_capacity: impl Fn(usize, &ResidentPacketTile<'_>, usize) -> Result<usize, Error>,
) -> Result<ResidentBatchPacketPlan, Error> {
    let batch_err = |suffix: &str| Error::MetalKernel {
        message: format!("{} Metal batch {}", params.family_name, suffix),
    };
    if prepared_tiles.len() != tile_tier1_job_bases.len() {
        return Err(batch_err("Tier-1 job-base count mismatch"));
    }
    let resolution_count = crate::batch_allocation::checked_count_sum(
        prepared_tiles.iter().map(|tile| tile.resolutions.len()),
        "J2K Metal resident packet resolutions",
    )?;
    let subband_count = crate::batch_allocation::checked_count_sum(
        prepared_tiles.iter().flat_map(|tile| {
            tile.resolutions
                .iter()
                .map(|resolution| resolution.subbands.len())
        }),
        "J2K Metal resident packet subbands",
    )?;
    let block_count = crate::batch_allocation::checked_count_sum(
        prepared_tiles.iter().map(|tile| tile.available_code_blocks),
        "J2K Metal resident packet blocks",
    )?;
    let descriptor_count = crate::batch_allocation::checked_count_sum(
        prepared_tiles
            .iter()
            .map(|tile| tile.packet_descriptors.len()),
        "J2K Metal resident packet descriptors",
    )?;
    let mut state_block_count = 0usize;
    for tile in prepared_tiles {
        for (descriptor_index, descriptor) in tile.packet_descriptors.iter().enumerate() {
            if tile.packet_descriptors[..descriptor_index]
                .iter()
                .any(|existing| existing.state_index == descriptor.state_index)
            {
                continue;
            }
            let resolution_index = usize::try_from(descriptor.packet_index)
                .map_err(|_| batch_err("descriptor packet index exceeds usize"))?;
            let resolution = tile
                .resolutions
                .get(resolution_index)
                .ok_or_else(|| batch_err("descriptor packet index out of range"))?;
            let state_blocks_for_descriptor =
                resolution
                    .subbands
                    .iter()
                    .try_fold(0usize, |total, subband| {
                        total
                            .checked_add(
                                usize::try_from(subband.code_block_count).map_err(|_| {
                                    batch_err("descriptor block count exceeds usize")
                                })?,
                            )
                            .ok_or_else(|| batch_err("state block count overflow"))
                    })?;
            state_block_count = state_block_count
                .checked_add(state_blocks_for_descriptor)
                .ok_or_else(|| batch_err("state block count overflow"))?;
        }
    }
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal resident packet batch plan");
    budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<J2kPacketResolution>(resolution_count),
        crate::batch_allocation::BatchMetadataRequest::of::<J2kPacketSubband>(subband_count),
        crate::batch_allocation::BatchMetadataRequest::of::<J2kResidentPacketBlock>(block_count),
        crate::batch_allocation::BatchMetadataRequest::of::<J2kPacketDescriptor>(descriptor_count),
        crate::batch_allocation::BatchMetadataRequest::of::<(u32, u32, usize)>(descriptor_count),
        crate::batch_allocation::BatchMetadataRequest::of::<J2kPacketStateBlock>(state_block_count),
        crate::batch_allocation::BatchMetadataRequest::of::<J2kBatchedPacketEncodeJob>(
            prepared_tiles.len(),
        ),
        crate::batch_allocation::BatchMetadataRequest::of::<J2kBatchedCodestreamAssemblyJob>(
            prepared_tiles.len(),
        ),
        crate::batch_allocation::BatchMetadataRequest::of::<usize>(prepared_tiles.len()),
        crate::batch_allocation::BatchMetadataRequest::of::<usize>(prepared_tiles.len()),
    ])?;
    let mut packet_resolutions =
        budget.try_vec(resolution_count, "J2K Metal resident packet resolutions")?;
    let mut packet_subbands =
        budget.try_vec(subband_count, "J2K Metal resident packet subbands")?;
    let mut resident_blocks = budget.try_vec(block_count, "J2K Metal resident packet blocks")?;
    let mut packet_descriptors =
        budget.try_vec(descriptor_count, "J2K Metal resident packet descriptors")?;
    let mut state_blocks =
        budget.try_vec(state_block_count, "J2K Metal resident packet state blocks")?;
    let mut packet_jobs = budget.try_vec(
        prepared_tiles.len(),
        "J2K Metal resident packet encode jobs",
    )?;
    let mut assembly_jobs = budget.try_vec(
        prepared_tiles.len(),
        "J2K Metal resident codestream assembly jobs",
    )?;
    let mut packet_output_capacity_total = 0usize;
    let mut packet_payload_copy_job_capacity_total = 0usize;
    let mut max_payload_copy_jobs_per_tile = 0usize;
    let mut header_capacity_total = 0usize;
    let mut scratch_words_total = 0usize;
    let mut codestream_capacity_total = 0usize;
    let mut codestream_offsets = budget.try_vec(
        prepared_tiles.len(),
        "J2K Metal resident codestream offsets",
    )?;
    let mut codestream_capacities = budget.try_vec(
        prepared_tiles.len(),
        "J2K Metal resident codestream capacities",
    )?;

    for (tile_index, (tile, &tier1_job_base)) in
        prepared_tiles.iter().zip(tile_tier1_job_bases).enumerate()
    {
        let local_resolution_offset = packet_resolutions.len();
        let local_subband_offset = packet_subbands.len();
        let local_block_offset = resident_blocks.len();
        let local_descriptor_offset = packet_descriptors.len();
        let local_state_block_offset = state_blocks.len();
        let mut max_tree_nodes = 1usize;
        let mut local_subband_count = 0usize;
        let mut local_resident_block_count = 0usize;
        let mut local_payload_copy_job_capacity = 0usize;

        for resolution in tile.resolutions {
            let subband_offset = u32::try_from(local_subband_count)
                .map_err(|_| batch_err("packet subband offset exceeds u32"))?;
            for subband in &resolution.subbands {
                let block_offset = u32::try_from(local_resident_block_count)
                    .map_err(|_| batch_err("packet block offset exceeds u32"))?;
                max_tree_nodes = max_tree_nodes.max(packet_tree_node_count(
                    subband.num_cbs_x,
                    subband.num_cbs_y,
                )?);
                let code_block_start = usize::try_from(subband.code_block_start)
                    .map_err(|_| batch_err("packet code-block offset exceeds usize"))?;
                let code_block_count = usize::try_from(subband.code_block_count)
                    .map_err(|_| batch_err("packet code-block count exceeds usize"))?;
                let code_block_end = code_block_start
                    .checked_add(code_block_count)
                    .ok_or_else(|| batch_err("packet code-block range overflow"))?;
                if code_block_end > tile.available_code_blocks {
                    return Err(batch_err("packet code-block range out of bounds"));
                }
                for tier1_job_index in code_block_start..code_block_end {
                    resident_blocks.push(J2kResidentPacketBlock {
                        tier1_job_index: u32::try_from(
                            tier1_job_base
                                .checked_add(tier1_job_index)
                                .ok_or_else(|| batch_err("Tier-1 index overflow"))?,
                        )
                        .map_err(|_| batch_err("Tier-1 index exceeds u32"))?,
                        previously_included: 0,
                        l_block: 3,
                        block_coding_mode: params.block_coding_mode,
                    });
                }
                packet_subbands.push(J2kPacketSubband {
                    block_offset,
                    block_count: subband.code_block_count,
                    num_cbs_x: subband.num_cbs_x,
                    num_cbs_y: subband.num_cbs_y,
                });
                local_subband_count = local_subband_count
                    .checked_add(1)
                    .ok_or_else(|| batch_err("subband count overflow"))?;
                local_resident_block_count = local_resident_block_count
                    .checked_add(code_block_count)
                    .ok_or_else(|| batch_err("resident block count overflow"))?;
            }
            packet_resolutions.push(J2kPacketResolution {
                subband_offset,
                subband_count: u32::try_from(resolution.subbands.len())
                    .map_err(|_| batch_err("resolution subband count exceeds u32"))?,
            });
        }

        if tile.resolutions.len()
            != usize::try_from(tile.resolution_count)
                .map_err(|_| batch_err("resolution count exceeds usize"))?
        {
            return Err(batch_err("resolution count mismatch"));
        }
        if local_resident_block_count
            != usize::try_from(tile.code_block_count)
                .map_err(|_| batch_err("code-block count exceeds usize"))?
        {
            return Err(batch_err("code-block count mismatch"));
        }

        let mut state_block_offsets = budget.try_vec(
            tile.packet_descriptors.len(),
            "J2K Metal resident packet state index map",
        )?;
        for descriptor in tile.packet_descriptors {
            let packet_index = usize::try_from(descriptor.packet_index)
                .map_err(|_| batch_err("descriptor packet index exceeds usize"))?;
            let resolution = packet_resolutions
                .get(local_resolution_offset + packet_index)
                .ok_or_else(|| batch_err("descriptor packet index out of range"))?;
            let subband_start = usize::try_from(resolution.subband_offset)
                .map_err(|_| batch_err("descriptor subband offset exceeds usize"))?;
            let subband_count = usize::try_from(resolution.subband_count)
                .map_err(|_| batch_err("descriptor subband count exceeds usize"))?;
            let mut packet_block_count = 0usize;
            for subband in &packet_subbands[local_subband_offset + subband_start
                ..local_subband_offset + subband_start + subband_count]
            {
                let subband_block_count = usize::try_from(subband.block_count)
                    .map_err(|_| batch_err("descriptor block count exceeds usize"))?;
                packet_block_count = packet_block_count
                    .checked_add(subband_block_count)
                    .ok_or_else(|| batch_err("descriptor block count overflow"))?;
            }
            let (state_block_offset, existing_count) = if let Some((offset, count)) =
                state_block_offsets
                    .iter()
                    .find(|(state_index, _, _)| *state_index == descriptor.state_index)
                    .map(|(_, offset, count)| (offset, count))
            {
                (*offset, *count)
            } else {
                let offset = u32::try_from(state_blocks.len() - local_state_block_offset)
                    .map_err(|_| batch_err("state block offset exceeds u32"))?;
                for subband in &packet_subbands[local_subband_offset + subband_start
                    ..local_subband_offset + subband_start + subband_count]
                {
                    for _ in 0..subband.block_count {
                        state_blocks.push(J2kPacketStateBlock {
                            previously_included: 0,
                            l_block: 3,
                        });
                    }
                }
                state_block_offsets.push((descriptor.state_index, offset, packet_block_count));
                (offset, packet_block_count)
            };
            if existing_count != packet_block_count {
                return Err(batch_err("descriptor state layout mismatch"));
            }
            local_payload_copy_job_capacity = local_payload_copy_job_capacity
                .checked_add(packet_block_count)
                .ok_or_else(|| batch_err("packet payload-copy job count overflow"))?;
            let (precinct_lo, precinct_hi) = split_u64_words(descriptor.precinct);
            packet_descriptors.push(J2kPacketDescriptor {
                packet_index: descriptor.packet_index,
                state_index: descriptor.state_index,
                layer: u32::from(descriptor.layer),
                resolution: descriptor.resolution,
                component: u32::from(descriptor.component),
                precinct_lo,
                precinct_hi,
                state_block_offset,
            });
        }

        let header_capacity = local_resident_block_count
            .checked_mul(256)
            .and_then(|bytes| bytes.checked_add(4096))
            .map(|bytes| bytes.max(4096))
            .ok_or_else(|| batch_err("packet header capacity overflow"))?;
        let packet_output_capacity =
            tile_packet_output_capacity(tile_index, tile, header_capacity)?;
        let (codestream_capacity, codestream_payload_offset) = match tile.codestream {
            Some(job) => (
                lossless_codestream_assembly_capacity(packet_output_capacity, job)?,
                lossless_codestream_payload_offset(job)?,
            ),
            None => (packet_output_capacity, 0),
        };
        let scratch_words = max_tree_nodes
            .checked_mul(6)
            .ok_or_else(|| batch_err("scratch size overflow"))?;

        let header_offset = header_capacity_total;
        let scratch_offset = scratch_words_total;
        if tile.packet_descriptors.is_empty() {
            local_payload_copy_job_capacity = local_resident_block_count;
        }
        let payload_copy_offset = packet_payload_copy_job_capacity_total;
        let codestream_offset = codestream_capacity_total;
        let packet_output_offset = codestream_offset
            .checked_add(codestream_payload_offset)
            .ok_or_else(|| batch_err("direct packet output offset overflow"))?;
        packet_jobs.push(J2kBatchedPacketEncodeJob {
            resolution_offset: u32::try_from(local_resolution_offset)
                .map_err(|_| batch_err("resolution offset exceeds u32"))?,
            subband_offset: u32::try_from(local_subband_offset)
                .map_err(|_| batch_err("subband offset exceeds u32"))?,
            block_offset: u32::try_from(local_block_offset)
                .map_err(|_| batch_err("block offset exceeds u32"))?,
            descriptor_offset: u32::try_from(local_descriptor_offset)
                .map_err(|_| batch_err("descriptor offset exceeds u32"))?,
            state_block_offset: u32::try_from(local_state_block_offset)
                .map_err(|_| batch_err("state block offset exceeds u32"))?,
            output_offset: u32::try_from(packet_output_offset)
                .map_err(|_| batch_err("packet output offset exceeds u32"))?,
            header_offset: u32::try_from(header_offset)
                .map_err(|_| batch_err("header offset exceeds u32"))?,
            scratch_offset: u32::try_from(scratch_offset)
                .map_err(|_| batch_err("scratch offset exceeds u32"))?,
            payload_copy_offset: u32::try_from(payload_copy_offset)
                .map_err(|_| batch_err("packet payload-copy offset exceeds u32"))?,
            payload_copy_capacity: u32::try_from(local_payload_copy_job_capacity)
                .map_err(|_| batch_err("packet payload-copy capacity exceeds u32"))?,
            resolution_count: tile.resolution_count,
            num_layers: u32::from(tile.num_layers),
            num_components: u32::from(tile.component_count),
            code_block_count: tile.code_block_count,
            subband_count: u32::try_from(local_subband_count)
                .map_err(|_| batch_err("local subband count exceeds u32"))?,
            descriptor_count: u32::try_from(tile.packet_descriptors.len())
                .map_err(|_| batch_err("descriptor count exceeds u32"))?,
            output_capacity: u32::try_from(packet_output_capacity)
                .map_err(|_| batch_err("packet output capacity exceeds u32"))?,
            header_capacity: u32::try_from(header_capacity)
                .map_err(|_| batch_err("header capacity exceeds u32"))?,
            scratch_node_capacity: u32::try_from(max_tree_nodes)
                .map_err(|_| batch_err("scratch node capacity exceeds u32"))?,
        });
        if let Some(codestream) = tile.codestream {
            assembly_jobs.push(J2kBatchedCodestreamAssemblyJob {
                tile_data_offset: u32::try_from(packet_output_offset)
                    .map_err(|_| batch_err("assembly packet offset exceeds u32"))?,
                codestream_offset: u32::try_from(codestream_offset)
                    .map_err(|_| batch_err("codestream offset exceeds u32"))?,
                width: codestream.width,
                height: codestream.height,
                num_components: u32::from(codestream.component_count),
                bit_depth: u32::from(codestream.bit_depth),
                signed_samples: u32::from(codestream.signed),
                num_decomposition_levels: u32::from(codestream.num_decomposition_levels),
                use_mct: u32::from(codestream.use_mct),
                guard_bits: u32::from(codestream.guard_bits),
                progression_order: codestream_progression_order_code(codestream.progression_order),
                write_tlm: u32::from(codestream.write_tlm),
                high_throughput: params.high_throughput,
                code_block_style: params.code_block_style,
                code_block_width_exp: u32::from(codestream.code_block_width_exp),
                code_block_height_exp: u32::from(codestream.code_block_height_exp),
                output_capacity: u32::try_from(codestream_capacity)
                    .map_err(|_| batch_err("codestream capacity exceeds u32"))?,
            });
        }
        codestream_offsets.push(codestream_offset);
        codestream_capacities.push(codestream_capacity);
        packet_output_capacity_total = packet_output_capacity_total
            .checked_add(packet_output_capacity)
            .ok_or_else(|| batch_err("packet output total overflow"))?;
        packet_payload_copy_job_capacity_total = packet_payload_copy_job_capacity_total
            .checked_add(local_payload_copy_job_capacity)
            .ok_or_else(|| batch_err("packet payload-copy job total overflow"))?;
        max_payload_copy_jobs_per_tile =
            max_payload_copy_jobs_per_tile.max(local_payload_copy_job_capacity);
        header_capacity_total = header_capacity_total
            .checked_add(header_capacity)
            .ok_or_else(|| batch_err("header total overflow"))?;
        scratch_words_total = scratch_words_total
            .checked_add(scratch_words)
            .ok_or_else(|| batch_err("scratch total overflow"))?;
        codestream_capacity_total = codestream_capacity_total
            .checked_add(codestream_capacity)
            .ok_or_else(|| batch_err("codestream total overflow"))?;
    }

    Ok(ResidentBatchPacketPlan {
        packet_resolutions,
        packet_subbands,
        resident_blocks,
        packet_descriptors,
        state_blocks,
        packet_jobs,
        assembly_jobs,
        packet_output_capacity_total,
        packet_payload_copy_job_capacity_total,
        max_payload_copy_jobs_per_tile,
        header_capacity_total,
        scratch_words_total,
        codestream_capacity_total,
        codestream_offsets,
        codestream_capacities,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        build_resident_batch_packet_plan, PreparedLosslessBatchTile, ResidentBatchPacketPlanParams,
    };
    use crate::Error;

    #[test]
    fn packet_ranges_are_bounded_by_actual_tier1_jobs() {
        let resolutions = [super::J2kResidentPacketizationResolution {
            subbands: vec![super::super::J2kResidentPacketizationSubband {
                code_block_start: 1,
                code_block_count: 1,
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        }];
        let tile = super::ResidentPacketTile {
            resolution_count: 1,
            num_layers: 1,
            component_count: 1,
            code_block_count: 2,
            available_code_blocks: 1,
            packet_descriptors: &[],
            resolutions: &resolutions,
            codestream: None,
        };
        let error = super::build_resident_packet_plan(
            &[tile],
            &[0],
            ResidentBatchPacketPlanParams {
                family_name: "test",
                block_coding_mode: 1,
                high_throughput: 1,
                code_block_style: 0x40,
            },
            |_, _, _| Ok(4096),
        )
        .err()
        .expect("descriptor must not index beyond the actual HT table");
        assert!(error.to_string().contains("code-block range out of bounds"));
    }

    #[test]
    fn resident_packet_plan_rejects_mismatched_tier1_job_bases() {
        let prepared_tiles: &[PreparedLosslessBatchTile] = &[];
        let error = build_resident_batch_packet_plan(
            prepared_tiles,
            &[0],
            ResidentBatchPacketPlanParams {
                family_name: "test",
                block_coding_mode: 0,
                high_throughput: 0,
                code_block_style: 0,
            },
            |_, _, _| Ok(0),
        )
        .err()
        .expect("mismatched parallel inputs must fail");

        assert!(matches!(error, Error::MetalKernel { .. }));
        assert!(error.to_string().contains("Tier-1 job-base count mismatch"));
    }
}
