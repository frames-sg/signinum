// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::{
    batch, batch_entropy_buffers, checked_u32, commit_and_wait_jpeg,
    copy_rgb8_surfaces_to_rgba_textures, dispatch_windowed_rgba_texture_pack,
    fast_subsampled_region_scaled_batch_groups, fast_subsampled_region_scaled_batch_plan,
    new_command_buffer, texture_batch_error_results, texture_batch_success_results,
    validate_rgba_texture_batch_output, windowed_texture_pack_params, BatchEntropyBufferKeys,
    BatchEntropyBufferPlan, BatchedFastPacket, Error, FastRegionScaledMetal, FastSubsampledMetal,
    JpegDecodeStatus, JpegFast420PacketV1, JpegFast422PacketV1, MetalRuntime, PixelFormat,
    PlaneMode,
};
use super::super::common::{
    decode_region_scaled_packet_surface, encode_subsampled_region_texture_decode,
    first_region_scaled_op, region_plane_buffers, subsampled_region_texture_batch_shape,
    subsampled_region_texture_packets,
};

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_restart_region_scaled_rgba_batch_to_textures<
    P: FastSubsampledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[&P],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    if !family_packets
        .iter()
        .any(|packet| packet.restart_interval_mcus() != 0)
    {
        return Ok(None);
    }
    if family_packets
        .iter()
        .any(|packet| packet.entropy_bytes().is_empty() || packet.entropy_checkpoints().is_empty())
    {
        return Ok(None);
    }

    let mut first_plan = None;
    for (request, packet) in requests.iter().zip(family_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        let segment_count_u32 = checked_u32(
            packet.entropy_checkpoints().len(),
            &format!(
                "{} restart region scaled texture segment count",
                P::FAMILY_NAME
            ),
        )?;
        let Some(plan) = fast_subsampled_region_scaled_batch_plan(
            packet,
            roi,
            scale,
            1,
            segment_count_u32,
            PlaneMode::YCbCr,
        ) else {
            return Ok(None);
        };
        let out_tile_len = plan.out_dims.0 as usize
            * plan.out_dims.1 as usize
            * PixelFormat::Rgba8.bytes_per_pixel();
        validate_rgba_texture_batch_output(output, plan.out_dims, requests.len(), out_tile_len)?;
        first_plan.get_or_insert(plan);
    }

    let Some(plan) = first_plan else {
        return Ok(Some(Vec::new()));
    };
    let mut result_budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal restart subsampled texture results",
        requests,
    )?;
    let mut surfaces = result_budget.try_vec(
        requests.len(),
        "JPEG Metal restart subsampled texture source surfaces",
    )?;
    for (request, packet) in requests.iter().zip(family_packets.iter().copied()) {
        let batched_packet = packet.to_batched();
        surfaces.push(decode_region_scaled_packet_surface(
            runtime,
            request,
            &batched_packet,
        ));
    }

    let mut group_indices = result_budget.try_vec(
        requests.len(),
        "JPEG Metal restart subsampled texture indices",
    )?;
    group_indices.extend(0..requests.len());
    let copied = copy_rgb8_surfaces_to_rgba_textures(
        runtime,
        output,
        plan.out_dims,
        requests.len(),
        &group_indices,
        surfaces,
        result_budget.live_bytes(),
    )?;
    let mut merged_results = result_budget.try_filled(
        requests.len(),
        None,
        "JPEG Metal restart subsampled texture merged results",
    )?;
    for (index, result) in copied {
        merged_results[index] = Some(result);
    }

    let mut results = result_budget.try_vec(
        requests.len(),
        "JPEG Metal ordered restart subsampled texture results",
    )?;
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal restart {} region scaled texture result for tile {index} was missing",
                P::FAMILY_NAME
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "the subsampled texture path keeps batch shape validation, retained scratch buffers, command encoding, and tile assembly in order"
)]
#[expect(
    clippy::similar_names,
    reason = "Cb and Cr are normative JPEG component names"
)]
fn try_decode_fast_subsampled_region_scaled_rgba_batch_to_textures<
    P: FastSubsampledMetal + FastRegionScaledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    let Some(family_packets) = subsampled_region_texture_packets::<P>(requests, packets)? else {
        return Ok(None);
    };

    let Some(first) = family_packets.first().copied() else {
        return Ok(None);
    };
    let Some((first_roi, first_scale)) = first_region_scaled_op(requests) else {
        return Ok(None);
    };
    if family_packets
        .iter()
        .any(|packet| packet.restart_interval_mcus() != 0)
    {
        return try_decode_fast_subsampled_restart_region_scaled_rgba_batch_to_textures(
            runtime,
            requests,
            &family_packets,
            output,
        );
    }
    if first.restart_interval_mcus() != 0 || first.entropy_checkpoints().is_empty() {
        return Ok(None);
    }

    let mut grouped_budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal grouped subsampled texture packet modes",
        requests,
    )?;
    let mut grouped_family_packets = grouped_budget.try_vec(
        family_packets.len(),
        "JPEG Metal grouped subsampled texture packet-mode pairs",
    )?;
    grouped_family_packets.extend(
        family_packets
            .iter()
            .copied()
            .map(|packet| (packet, PlaneMode::YCbCr)),
    );
    let Some(groups) =
        fast_subsampled_region_scaled_batch_groups(requests, &grouped_family_packets)?
    else {
        return Ok(None);
    };
    if groups.len() > 1 {
        return try_decode_grouped_fast_subsampled_region_scaled_rgba_batch_to_textures(
            runtime,
            requests,
            &family_packets,
            output,
            groups,
        );
    }

    let Some(shape) = subsampled_region_texture_batch_shape::<P>(
        requests,
        &family_packets,
        first,
        first_roi,
        first_scale,
    )?
    else {
        return Ok(None);
    };
    let out_tile_len = shape.plan.out_dims.0 as usize
        * shape.plan.out_dims.1 as usize
        * PixelFormat::Rgba8.bytes_per_pixel();
    validate_rgba_texture_batch_output(
        output,
        shape.plan.out_dims,
        shape.tile_count,
        out_tile_len,
    )?;

    let mut batch_scratch = runtime.batch_scratch()?;
    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        requests,
        &mut batch_scratch,
        BatchEntropyBufferPlan {
            keys: BatchEntropyBufferKeys {
                payload: P::REGION_SCALED_TEXTURE_KEYS.entropy,
                offsets: P::REGION_SCALED_TEXTURE_KEYS.entropy_offsets,
                lens: P::REGION_SCALED_TEXTURE_KEYS.entropy_lens,
                checkpoints: P::REGION_SCALED_TEXTURE_KEYS.entropy_checkpoints,
            },
            tile_count: shape.tile_count,
            segment_count: shape.segment_count,
        },
        family_packets.iter().map(|packet| packet.entropy_bytes()),
        family_packets
            .iter()
            .map(|packet| packet.entropy_checkpoints()),
    )?
    else {
        return Ok(None);
    };

    let (y_plane, cb_plane, cr_plane) = region_plane_buffers(
        &mut batch_scratch,
        &runtime.device,
        &P::REGION_SCALED_TEXTURE_KEYS,
        shape.plan,
        shape.tile_count,
    )?;
    let mut status_budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal subsampled region texture statuses",
        requests,
    )?;
    let statuses = status_budget.try_filled(
        shape.total_decode_threads as usize,
        JpegDecodeStatus::default(),
        "JPEG Metal subsampled region texture decode statuses",
    )?;
    let status_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        P::REGION_SCALED_TEXTURE_KEYS.status,
        &statuses,
    )?;

    let command_buffer = new_command_buffer(&runtime.queue)?;
    encode_subsampled_region_texture_decode::<P>(
        runtime,
        &command_buffer,
        first,
        &entropy_buffers,
        &status_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        shape,
    )?;

    dispatch_windowed_rgba_texture_pack(
        &command_buffer,
        P::pack_windowed_rgba_texture_pipeline(runtime),
        (&y_plane, &cb_plane, &cr_plane),
        output,
        windowed_texture_pack_params(shape.plan),
        shape.tile_count,
        shape.plan.out_dims,
    )?;

    commit_and_wait_jpeg(&command_buffer)?;
    // Keep scratch leased until the CPU has consumed the GPU status below.

    if let Some(results) =
        texture_batch_error_results(requests, &status_buffer, shape.total_decode_threads)?
    {
        return Ok(Some(results));
    }

    Ok(Some(texture_batch_success_results(
        requests,
        output,
        shape.plan.out_dims,
        requests.len(),
    )?))
}

pub(in crate::compute) fn try_decode_fast420_region_scaled_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    try_decode_fast_subsampled_region_scaled_rgba_batch_to_textures::<JpegFast420PacketV1>(
        runtime, requests, packets, output,
    )
}

#[cfg(target_os = "macos")]
fn try_decode_grouped_fast_subsampled_region_scaled_rgba_batch_to_textures<
    P: FastSubsampledMetal + FastRegionScaledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[&P],
    output: &crate::MetalBatchTextureOutput,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    for (request, packet) in requests.iter().zip(family_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        let segment_count_u32 = checked_u32(
            packet.entropy_checkpoints().len(),
            &format!(
                "{} grouped region scaled texture batch segment count",
                P::FAMILY_NAME
            ),
        )?;
        let Some(plan) = fast_subsampled_region_scaled_batch_plan(
            packet,
            roi,
            scale,
            1,
            segment_count_u32,
            PlaneMode::YCbCr,
        ) else {
            return Ok(None);
        };
        let out_tile_len = plan.out_dims.0 as usize
            * plan.out_dims.1 as usize
            * PixelFormat::Rgba8.bytes_per_pixel();
        validate_rgba_texture_batch_output(output, plan.out_dims, requests.len(), out_tile_len)?;
    }

    let mut result_budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal grouped subsampled texture results",
        requests,
    )?;
    let mut merged_results = result_budget.try_filled(
        requests.len(),
        None,
        "JPEG Metal grouped subsampled texture result slots",
    )?;
    for group_indices in groups {
        let group_output = output.clone_slots(&group_indices)?;
        let mut group_budget = crate::plan_owner_ledger::batch_execution_budget(
            "JPEG Metal grouped subsampled texture sub-batch",
            requests,
        )?;
        let mut group_requests = group_budget.try_vec(
            group_indices.len(),
            "JPEG Metal grouped subsampled texture requests",
        )?;
        group_requests.extend(group_indices.iter().map(|&index| requests[index].clone()));
        let mut group_packets = group_budget.try_vec(
            group_indices.len(),
            "JPEG Metal grouped subsampled texture packets",
        )?;
        group_packets.extend(
            group_indices
                .iter()
                .map(|&index| family_packets[index].to_batched()),
        );
        batch::stamp_execution_owner_baseline(&mut group_requests, 0, group_budget.live_bytes());

        let Some(group_results) = try_decode_fast_subsampled_region_scaled_rgba_batch_to_textures::<
            P,
        >(
            runtime, &group_requests, &group_packets, &group_output
        )?
        else {
            return Ok(None);
        };
        if group_results.len() != group_indices.len() {
            return Err(Error::MetalKernel {
                message: format!(
                    "JPEG Metal grouped {} region scaled texture result count mismatch",
                    P::FAMILY_NAME
                ),
            });
        }
        for (original_index, result) in group_indices.into_iter().zip(group_results) {
            merged_results[original_index] = Some(result);
        }
    }

    let mut results = result_budget.try_vec(
        requests.len(),
        "JPEG Metal ordered grouped subsampled texture results",
    )?;
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal grouped {} region scaled texture result for tile {index} was missing",
                P::FAMILY_NAME
            ),
        })?);
    }
    Ok(Some(results))
}

pub(in crate::compute) fn try_decode_fast422_region_scaled_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    try_decode_fast_subsampled_region_scaled_rgba_batch_to_textures::<JpegFast422PacketV1>(
        runtime, requests, packets, output,
    )
}
