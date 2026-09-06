// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::metal_types::prelude::*;

use super::super::{
    batch, batch_entropy_buffers, batch_output_buffer_or_new, bind_three_plane_pack, checked_u32,
    commit_and_wait_jpeg, copy_grouped_surfaces_to_output, dispatch_3d_pipeline,
    fast444_scaled_region_params, fast_subsampled_region_scaled_batch_groups,
    fast_subsampled_region_scaled_batch_plan, new_command_buffer, new_compute_command_encoder,
    region_scaled_batch_error_results, surface_batch_success_results, BatchEntropyBufferKeys,
    BatchEntropyBufferPlan, BatchedFastPacket, Error, FastRegionScaledMetal, JpegDecodeStatus,
    JpegFast420PacketV1, JpegFast422PacketV1, JpegFast444PacketV1, JpegWindowedPackBatchParams,
    MetalRuntime, PixelFormat, PlaneMode, Rect, RegionScaledBatchPlan, Surface,
};
use super::common::{
    decode_region_scaled_packet_surface, encode_subsampled_region_rgb_decode,
    first_region_scaled_op, region_plane_buffers, subsampled_region_rgb_batch_shape,
    subsampled_region_rgb_packets,
};

#[cfg(target_os = "macos")]
fn copy_restart_region_scaled_results_to_output<P: FastRegionScaledMetal>(
    runtime: &MetalRuntime,
    output: &crate::MetalBatchOutputBuffer,
    plan: RegionScaledBatchPlan,
    group_results: Vec<Result<Surface, Error>>,
    budget: &mut crate::batch_allocation::BatchMetadataBudget,
) -> Result<Vec<Result<Surface, Error>>, Error> {
    let request_count = group_results.len();
    let mut group_indices = budget.try_vec(
        request_count,
        "JPEG Metal restart region-scaled output indices",
    )?;
    group_indices.extend(0..request_count);
    let copied = copy_grouped_surfaces_to_output(
        runtime,
        output,
        plan.out_dims,
        plan.out_tile_len,
        &group_indices,
        group_results,
        budget.live_bytes(),
    )?;
    let mut merged = budget.try_filled(
        request_count,
        None,
        "JPEG Metal restart region-scaled merged result slots",
    )?;
    for (index, result) in copied {
        merged[index] = Some(result);
    }

    let mut ordered = budget.try_vec(
        request_count,
        "JPEG Metal ordered restart region-scaled results",
    )?;
    for (index, result) in merged.into_iter().enumerate() {
        ordered.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal restart {} region scaled buffer result for tile {index} was missing",
                P::FAMILY_NAME
            ),
        })?);
    }
    Ok(ordered)
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast444_region_scaled_rgb_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<JpegFast444PacketV1>(
        runtime, requests, packets, None,
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast444_region_scaled_rgb_batch_to_surfaces_into_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchOutputBuffer,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<JpegFast444PacketV1>(
        runtime,
        requests,
        packets,
        Some(output),
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn fast444_region_scaled_rgb_output_shape(
    packet: &JpegFast444PacketV1,
    roi: Rect,
    scale: j2k_core::Downscale,
) -> Option<((u32, u32), usize, usize)> {
    let scaled = roi.scaled_covering(scale);
    let scaled_roi = j2k_jpeg::Rect {
        x: scaled.x,
        y: scaled.y,
        w: scaled.w,
        h: scaled.h,
    };
    let params = fast444_scaled_region_params(packet, scale, scaled_roi)?;
    let out_dims = (params.scaled_width, params.scaled_height);
    let out_stride = out_dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let out_tile_len = out_stride * out_dims.1 as usize;
    Some((out_dims, out_stride, out_tile_len))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast420_region_scaled_rgb_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast420_region_scaled_rgb_batch_to_surfaces_with_output(
        runtime, requests, packets, None,
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast420_region_scaled_rgb_batch_to_surfaces_into_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchOutputBuffer,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast420_region_scaled_rgb_batch_to_surfaces_with_output(
        runtime,
        requests,
        packets,
        Some(output),
    )
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_restart_region_scaled_rgb_batch_to_surfaces_with_output<
    P: FastRegionScaledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[(&P, PlaneMode)],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if !family_packets
        .iter()
        .any(|(packet, _)| packet.restart_interval_mcus() != 0)
    {
        return Ok(None);
    }
    if family_packets.iter().any(|(packet, _)| {
        packet.entropy_bytes().is_empty() || packet.entropy_checkpoints().is_empty()
    }) {
        return Ok(None);
    }

    let mut first_plan = None;
    if output.is_some() {
        for (request, (packet, mode)) in requests.iter().zip(family_packets.iter().copied()) {
            let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
                return Ok(None);
            };
            let segment_count_u32 = checked_u32(
                packet.entropy_checkpoints().len(),
                &format!(
                    "{} restart region scaled buffer segment count",
                    P::FAMILY_NAME
                ),
            )?;
            let Some(plan) = fast_subsampled_region_scaled_batch_plan(
                packet,
                roi,
                scale,
                1,
                segment_count_u32,
                mode,
            ) else {
                return Ok(None);
            };
            batch_output_buffer_or_new(
                runtime,
                output,
                plan.out_dims,
                requests.len(),
                plan.pack_params.out_stride as usize,
                plan.out_tile_len,
            )?;
            first_plan.get_or_insert(plan);
        }
    }

    let mut result_budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal restart region-scaled buffer results",
        requests,
    )?;
    let mut results = result_budget.try_vec(
        requests.len(),
        "JPEG Metal restart region-scaled surface results",
    )?;
    for (request, (packet, mode)) in requests.iter().zip(family_packets.iter().copied()) {
        let batched_packet = packet.to_region_scaled_batched(mode);
        results.push(decode_region_scaled_packet_surface(
            runtime,
            request,
            &batched_packet,
        ));
    }

    let Some(output) = output else {
        return Ok(Some(results));
    };
    let Some(plan) = first_plan else {
        return Ok(Some(results));
    };
    Ok(Some(copy_restart_region_scaled_results_to_output::<P>(
        runtime,
        output,
        plan,
        results,
        &mut result_budget,
    )?))
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "the region-scaled GPU path keeps batch eligibility, scratch allocation, command encoding, and output copies in dispatch order"
)]
#[expect(
    clippy::similar_names,
    reason = "Cb and Cr are normative JPEG component names"
)]
pub(in crate::compute) fn try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output<
    P: FastRegionScaledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let Some(family_packets) = subsampled_region_rgb_packets::<P>(requests, packets)? else {
        return Ok(None);
    };

    let Some((first, first_mode)) = family_packets.first().copied() else {
        return Ok(None);
    };
    let Some((first_roi, first_scale)) = first_region_scaled_op(requests) else {
        return Ok(None);
    };
    if family_packets
        .iter()
        .any(|(packet, _)| packet.restart_interval_mcus() != 0)
    {
        return try_decode_fast_subsampled_restart_region_scaled_rgb_batch_to_surfaces_with_output(
            runtime,
            requests,
            &family_packets,
            output,
        );
    }
    if first.restart_interval_mcus() != 0 || first.entropy_checkpoints().is_empty() {
        return Ok(None);
    }

    let Some(groups) = fast_subsampled_region_scaled_batch_groups(requests, &family_packets)?
    else {
        return Ok(None);
    };
    if groups.len() > 1 {
        return try_decode_grouped_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output(
            runtime,
            requests,
            &family_packets,
            output,
            groups,
        );
    }

    let Some(shape) = subsampled_region_rgb_batch_shape::<P>(
        requests,
        &family_packets,
        first,
        first_mode,
        first_roi,
        first_scale,
    )?
    else {
        return Ok(None);
    };

    let mut batch_scratch = runtime.batch_scratch()?;
    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        requests,
        &mut batch_scratch,
        BatchEntropyBufferPlan {
            keys: BatchEntropyBufferKeys {
                payload: P::REGION_SCALED_KEYS.entropy,
                offsets: P::REGION_SCALED_KEYS.entropy_offsets,
                lens: P::REGION_SCALED_KEYS.entropy_lens,
                checkpoints: P::REGION_SCALED_KEYS.entropy_checkpoints,
            },
            tile_count: shape.tile_count,
            segment_count: shape.segment_count,
        },
        family_packets
            .iter()
            .map(|(packet, _)| packet.entropy_bytes()),
        family_packets
            .iter()
            .map(|(packet, _)| packet.entropy_checkpoints()),
    )?
    else {
        return Ok(None);
    };

    let (y_plane, cb_plane, cr_plane) = region_plane_buffers(
        &mut batch_scratch,
        &runtime.device,
        &P::REGION_SCALED_KEYS,
        shape.plan,
        shape.tile_count,
    )?;
    let out_buffer = batch_output_buffer_or_new(
        runtime,
        output,
        shape.plan.out_dims,
        shape.tile_count,
        shape.plan.pack_params.out_stride as usize,
        shape.plan.out_tile_len,
    )?;
    let mut status_budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal region-scaled RGB batch statuses",
        requests,
    )?;
    let statuses = status_budget.try_filled(
        shape.total_decode_threads as usize,
        JpegDecodeStatus::default(),
        "JPEG Metal region-scaled RGB decode statuses",
    )?;
    let status_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        P::REGION_SCALED_KEYS.status,
        &statuses,
    )?;

    let command_buffer = new_command_buffer(&runtime.queue)?;
    encode_subsampled_region_rgb_decode::<P>(
        runtime,
        &command_buffer,
        first,
        &entropy_buffers,
        &status_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        shape,
    )?;

    let pack_encoder = new_compute_command_encoder(&command_buffer)?;
    pack_encoder.setComputePipelineState(P::pack_windowed_rgb_batch_pipeline(runtime));
    bind_three_plane_pack::<JpegWindowedPackBatchParams>(
        &pack_encoder,
        [Some(&y_plane), Some(&cb_plane), Some(&cr_plane)],
        &out_buffer,
        &shape.plan.pack_params,
    );
    dispatch_3d_pipeline(
        &pack_encoder,
        P::pack_windowed_rgb_batch_pipeline(runtime),
        (
            shape.plan.out_dims.0,
            shape.plan.out_dims.1,
            shape.tile_count_u32,
        ),
    );
    pack_encoder.endEncoding();

    commit_and_wait_jpeg(&command_buffer)?;
    // Keep scratch leased until the CPU has consumed the GPU status below.

    if let Some(results) =
        region_scaled_batch_error_results(requests, &status_buffer, shape.total_decode_threads)?
    {
        return Ok(Some(results));
    }

    Ok(Some(surface_batch_success_results(
        requests,
        &out_buffer,
        shape.plan.out_dims,
        PixelFormat::Rgb8,
        requests.len(),
        shape.plan.out_tile_len,
        output,
    )?))
}

fn try_decode_fast420_region_scaled_rgb_batch_to_surfaces_with_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<JpegFast420PacketV1>(
        runtime, requests, packets, output,
    )
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "grouped region-scaled dispatch keeps per-group command buffers, retained resources, and result placement in one lifetime scope"
)]
fn try_decode_grouped_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output<
    P: FastRegionScaledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[(&P, PlaneMode)],
    output: Option<&crate::MetalBatchOutputBuffer>,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if let Some(output) = output {
        for (request, (packet, mode)) in requests.iter().zip(family_packets.iter().copied()) {
            let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
                return Ok(None);
            };
            let segment_count_u32 = checked_u32(
                packet.entropy_checkpoints().len(),
                &format!(
                    "{} grouped region scaled buffer segment count",
                    P::FAMILY_NAME
                ),
            )?;
            let Some(plan) = fast_subsampled_region_scaled_batch_plan(
                packet,
                roi,
                scale,
                1,
                segment_count_u32,
                mode,
            ) else {
                return Ok(None);
            };
            batch_output_buffer_or_new(
                runtime,
                Some(output),
                plan.out_dims,
                requests.len(),
                plan.pack_params.out_stride as usize,
                plan.out_tile_len,
            )?;
        }
    }

    let mut result_budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal grouped region-scaled buffer results",
        requests,
    )?;
    let mut merged_results = result_budget.try_filled(
        requests.len(),
        None,
        "JPEG Metal grouped region-scaled buffer result slots",
    )?;
    for group_indices in groups {
        let mut group_budget = crate::plan_owner_ledger::batch_execution_budget(
            "JPEG Metal grouped region-scaled buffer sub-batch",
            requests,
        )?;
        let mut group_requests = group_budget.try_vec(
            group_indices.len(),
            "JPEG Metal grouped region-scaled buffer requests",
        )?;
        group_requests.extend(group_indices.iter().map(|&index| requests[index].clone()));
        let mut group_packets = group_budget.try_vec(
            group_indices.len(),
            "JPEG Metal grouped region-scaled buffer packets",
        )?;
        group_packets.extend(group_indices.iter().map(|&index| {
            let (packet, mode) = family_packets[index];
            packet.to_region_scaled_batched(mode)
        }));
        batch::stamp_execution_owner_baseline(&mut group_requests, 0, group_budget.live_bytes());

        let Some(group_results) =
            try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<P>(
                runtime,
                &group_requests,
                &group_packets,
                None,
            )?
        else {
            return Ok(None);
        };

        if let Some(output) = output {
            let Some(&first_group_index) = group_indices.first() else {
                continue;
            };
            let batch::BatchOp::RegionScaled { roi, scale } = requests[first_group_index].op else {
                return Ok(None);
            };
            let (packet, mode) = family_packets[first_group_index];
            let segment_count_u32 = checked_u32(
                packet.entropy_checkpoints().len(),
                &format!(
                    "{} grouped region scaled buffer segment count",
                    P::FAMILY_NAME
                ),
            )?;
            let Some(plan) = fast_subsampled_region_scaled_batch_plan(
                packet,
                roi,
                scale,
                1,
                segment_count_u32,
                mode,
            ) else {
                return Ok(None);
            };
            for (original_index, result) in copy_grouped_surfaces_to_output(
                runtime,
                output,
                plan.out_dims,
                plan.out_tile_len,
                &group_indices,
                group_results,
                result_budget.live_bytes(),
            )? {
                merged_results[original_index] = Some(result);
            }
        } else {
            if group_results.len() != group_indices.len() {
                return Err(Error::MetalKernel {
                    message: format!(
                        "JPEG Metal grouped {} region scaled buffer result count mismatch",
                        P::FAMILY_NAME
                    ),
                });
            }
            for (original_index, result) in group_indices.into_iter().zip(group_results) {
                merged_results[original_index] = Some(result);
            }
        }
    }

    let mut results = result_budget.try_vec(
        requests.len(),
        "JPEG Metal ordered grouped region-scaled buffer results",
    )?;
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal grouped {} region scaled buffer result for tile {index} was missing",
                P::FAMILY_NAME
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast422_region_scaled_rgb_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast422_region_scaled_rgb_batch_to_surfaces_with_output(
        runtime, requests, packets, None,
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast422_region_scaled_rgb_batch_to_surfaces_into_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchOutputBuffer,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast422_region_scaled_rgb_batch_to_surfaces_with_output(
        runtime,
        requests,
        packets,
        Some(output),
    )
}

fn try_decode_fast422_region_scaled_rgb_batch_to_surfaces_with_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<JpegFast422PacketV1>(
        runtime, requests, packets, output,
    )
}
