// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::metal_types::prelude::*;

use super::super::super::{
    batch, batch_entropy_buffers, bind_fast_decode_entropy_inputs, checked_u32,
    commit_and_wait_jpeg, copy_rgb8_surfaces_to_rgba_textures, dispatch_1d_pipeline,
    dispatch_rgba_texture_pack, fast444_packets_share_region_scaled_batch_shape,
    fast444_region_scaled_batch_groups, fast444_scaled_region_params, fast_packet_huffman_tables,
    new_command_buffer, new_compute_command_encoder, plane_mode_to_u32,
    texture_batch_error_results, texture_batch_success_results, validate_rgba_texture_batch_output,
    BatchEntropyBufferKeys, BatchEntropyBufferPlan, BatchEntropyBuffers, BatchedFastPacket, Buffer,
    CommandBufferRef, Error, FastDecodeEntropyInputs, JpegDecodeStatus, JpegFast444PacketV1,
    JpegFastRegionScaledBatchParams, JpegTexturePackBatchParams, MetalRuntime, PixelFormat,
    PlaneMode, Rect,
};
use super::super::common::{
    decode_region_scaled_packet_surface, fast444_region_packets, first_region_scaled_op,
};
use super::super::rgb::fast444_region_scaled_rgb_output_shape;

#[cfg(target_os = "macos")]
fn try_decode_fast444_restart_region_scaled_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    fast444_packets: &[(&JpegFast444PacketV1, PlaneMode)],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    if !fast444_packets
        .iter()
        .any(|(packet, _)| packet.restart_interval_mcus != 0)
    {
        return Ok(None);
    }
    if fast444_packets
        .iter()
        .any(|(packet, _)| packet.entropy_bytes.is_empty() || packet.entropy_checkpoints.is_empty())
    {
        return Ok(None);
    }

    let mut first_shape = None;
    for (request, (packet, _)) in requests.iter().zip(fast444_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        let Some((out_dims, _, _)) = fast444_region_scaled_rgb_output_shape(packet, roi, scale)
        else {
            return Ok(None);
        };
        let out_tile_len =
            out_dims.0 as usize * out_dims.1 as usize * PixelFormat::Rgba8.bytes_per_pixel();
        validate_rgba_texture_batch_output(output, out_dims, requests.len(), out_tile_len)?;
        first_shape.get_or_insert(out_dims);
    }

    let Some(out_dims) = first_shape else {
        return Ok(Some(Vec::new()));
    };
    let mut result_budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal restart fast444 texture results",
        requests,
    )?;
    let mut surfaces = result_budget.try_vec(
        requests.len(),
        "JPEG Metal restart fast444 texture source surfaces",
    )?;
    for (request, (packet, mode)) in requests.iter().zip(fast444_packets.iter().copied()) {
        let batched_packet = BatchedFastPacket::Fast444(packet, mode);
        surfaces.push(decode_region_scaled_packet_surface(
            runtime,
            request,
            &batched_packet,
        ));
    }

    let mut group_indices =
        result_budget.try_vec(requests.len(), "JPEG Metal restart fast444 texture indices")?;
    group_indices.extend(0..requests.len());
    let copied = copy_rgb8_surfaces_to_rgba_textures(
        runtime,
        output,
        out_dims,
        requests.len(),
        &group_indices,
        surfaces,
        result_budget.live_bytes(),
    )?;
    let mut merged_results = result_budget.try_filled(
        requests.len(),
        None,
        "JPEG Metal restart fast444 texture merged results",
    )?;
    for (index, result) in copied {
        merged_results[index] = Some(result);
    }

    let mut results = result_budget.try_vec(
        requests.len(),
        "JPEG Metal ordered restart fast444 texture results",
    )?;
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal restart fast444 region scaled texture result for tile {index} was missing"
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct Fast444RegionTextureShape {
    tile_count: usize,
    segment_count: usize,
    total_decode_threads: u32,
    out_dims: (u32, u32),
    out_tile_len: usize,
    plane_len: usize,
    decode_params: JpegFastRegionScaledBatchParams,
    pack_params: JpegTexturePackBatchParams,
}

#[cfg(target_os = "macos")]
fn fast444_region_texture_shape(
    requests: &[batch::QueuedRequest],
    fast444_packets: &[(&JpegFast444PacketV1, PlaneMode)],
    first: &JpegFast444PacketV1,
    first_mode: PlaneMode,
    first_roi: Rect,
    first_scale: j2k_core::Downscale,
) -> Result<Option<Fast444RegionTextureShape>, Error> {
    let first_scaled = first_roi.scaled_covering(first_scale);
    let first_scaled_roi = j2k_jpeg::Rect {
        x: first_scaled.x,
        y: first_scaled.y,
        w: first_scaled.w,
        h: first_scaled.h,
    };
    let Some(first_decode_params) =
        fast444_scaled_region_params(first, first_scale, first_scaled_roi)
    else {
        return Ok(None);
    };
    let segment_count = first.entropy_checkpoints.len();
    let tile_count = fast444_packets.len();
    let tile_count_u32 = checked_u32(tile_count, "region scaled texture batch tile count")?;
    let segment_count_u32 =
        checked_u32(segment_count, "region scaled texture batch segment count")?;
    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal region scaled texture batch decode thread count overflowed"
                    .to_string(),
            })?,
        "region scaled texture batch decode thread count",
    )?;

    for (request, (packet, mode)) in requests.iter().zip(fast444_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        if scale != first_scale
            || mode != first_mode
            || !fast444_packets_share_region_scaled_batch_shape(first, packet, segment_count)
        {
            return Ok(None);
        }
        let scaled = roi.scaled_covering(scale);
        let scaled_roi = j2k_jpeg::Rect {
            x: scaled.x,
            y: scaled.y,
            w: scaled.w,
            h: scaled.h,
        };
        if fast444_scaled_region_params(packet, scale, scaled_roi) != Some(first_decode_params) {
            return Ok(None);
        }
    }

    let out_dims = (
        first_decode_params.scaled_width,
        first_decode_params.scaled_height,
    );
    let out_tile_len =
        out_dims.0 as usize * out_dims.1 as usize * PixelFormat::Rgba8.bytes_per_pixel();
    Ok(Some(Fast444RegionTextureShape {
        tile_count,
        segment_count,
        total_decode_threads,
        out_dims,
        out_tile_len,
        plane_len: first_decode_params.scaled_width as usize
            * first_decode_params.scaled_height as usize,
        decode_params: JpegFastRegionScaledBatchParams {
            scaled_width: first_decode_params.scaled_width,
            scaled_height: first_decode_params.scaled_height,
            chroma_width: first_decode_params.scaled_width,
            chroma_height: first_decode_params.scaled_height,
            mcus_per_row: first_decode_params.mcus_per_row,
            mcu_rows: first_decode_params.mcu_rows,
            segment_count: segment_count_u32,
            tile_count: tile_count_u32,
            scale_shift: first_decode_params.scale_shift,
            origin_x: first_decode_params.origin_x,
            origin_y: first_decode_params.origin_y,
        },
        pack_params: JpegTexturePackBatchParams {
            width: out_dims.0,
            height: out_dims.1,
            chroma_width: out_dims.0,
            chroma_height: out_dims.1,
            tile_index: 0,
            alpha: u32::from(u8::MAX),
            mode: plane_mode_to_u32(first_mode),
        },
    }))
}

#[cfg(target_os = "macos")]
fn encode_fast444_region_texture_decode(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    first: &JpegFast444PacketV1,
    entropy_buffers: &BatchEntropyBuffers,
    status_buffer: &Buffer,
    planes: [&Buffer; 3],
    shape: Fast444RegionTextureShape,
) -> Result<(), Error> {
    let (dc_tables, ac_tables) = fast_packet_huffman_tables(first);
    let decoder_encoder = new_compute_command_encoder(command_buffer)?;
    decoder_encoder.setComputePipelineState(&runtime.pipelines.fast444_scaled_region_batch_decode);
    bind_fast_decode_entropy_inputs::<JpegFastRegionScaledBatchParams>(
        &decoder_encoder,
        &FastDecodeEntropyInputs {
            entropy_buffer: &entropy_buffers.payload,
            planes,
            params: &shape.decode_params,
            quants: [&first.y_quant, &first.cb_quant, &first.cr_quant],
            dc_tables: &dc_tables,
            ac_tables: &ac_tables,
            slot14_buffer: &entropy_buffers.offsets,
            slot15_buffer: &entropy_buffers.lens,
            slot16_buffer: status_buffer,
        },
    );
    decoder_encoder.bind_buffer(17, Some(&entropy_buffers.checkpoints), 0);
    dispatch_1d_pipeline(
        &decoder_encoder,
        &runtime.pipelines.fast444_scaled_region_batch_decode,
        shape.total_decode_threads,
    );
    decoder_encoder.endEncoding();
    Ok(())
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "the fast444 texture path keeps validation, Metal resource ownership, command encoding, and completion checks in submission order"
)]
#[expect(
    clippy::similar_names,
    reason = "Cb and Cr are normative JPEG component names"
)]
pub(in crate::compute) fn try_decode_fast444_region_scaled_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    let Some(fast444_packets) = fast444_region_packets(requests, packets)? else {
        return Ok(None);
    };

    let Some((first, first_mode)) = fast444_packets.first().copied() else {
        return Ok(None);
    };
    let Some((first_roi, first_scale)) = first_region_scaled_op(requests) else {
        return Ok(None);
    };
    if fast444_packets
        .iter()
        .any(|(packet, _)| packet.restart_interval_mcus != 0)
    {
        return try_decode_fast444_restart_region_scaled_rgba_batch_to_textures(
            runtime,
            requests,
            &fast444_packets,
            output,
        );
    }
    if first.restart_interval_mcus != 0 || first.entropy_checkpoints.is_empty() {
        return Ok(None);
    }

    let Some(groups) = fast444_region_scaled_batch_groups(requests, &fast444_packets)? else {
        return Ok(None);
    };
    if groups.len() > 1 {
        return try_decode_grouped_fast444_region_scaled_rgba_batch_to_textures(
            runtime,
            requests,
            &fast444_packets,
            output,
            groups,
        );
    }

    let Some(shape) = fast444_region_texture_shape(
        requests,
        &fast444_packets,
        first,
        first_mode,
        first_roi,
        first_scale,
    )?
    else {
        return Ok(None);
    };
    validate_rgba_texture_batch_output(
        output,
        shape.out_dims,
        shape.tile_count,
        shape.out_tile_len,
    )?;

    let mut batch_scratch = runtime.batch_scratch()?;
    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        requests,
        &mut batch_scratch,
        BatchEntropyBufferPlan {
            keys: BatchEntropyBufferKeys {
                payload: "fast444_region_scaled_texture_entropy",
                offsets: "fast444_region_scaled_texture_entropy_offsets",
                lens: "fast444_region_scaled_texture_entropy_lens",
                checkpoints: "fast444_region_scaled_texture_entropy_checkpoints",
            },
            tile_count: shape.tile_count,
            segment_count: shape.segment_count,
        },
        fast444_packets
            .iter()
            .map(|(packet, _)| packet.entropy_bytes.as_slice()),
        fast444_packets
            .iter()
            .map(|(packet, _)| packet.entropy_checkpoints.as_slice()),
    )?
    else {
        return Ok(None);
    };

    let y_plane = batch_scratch.private_buffer(
        &runtime.device,
        "fast444_region_scaled_texture_y",
        shape.plane_len * shape.tile_count,
    )?;
    let cb_plane = batch_scratch.private_buffer(
        &runtime.device,
        "fast444_region_scaled_texture_cb",
        shape.plane_len * shape.tile_count,
    )?;
    let cr_plane = batch_scratch.private_buffer(
        &runtime.device,
        "fast444_region_scaled_texture_cr",
        shape.plane_len * shape.tile_count,
    )?;
    let mut status_budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal fast444 region texture statuses",
        requests,
    )?;
    let statuses = status_budget.try_filled(
        shape.total_decode_threads as usize,
        JpegDecodeStatus::default(),
        "JPEG Metal fast444 region texture decode statuses",
    )?;
    let status_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        "fast444_region_scaled_texture_status",
        &statuses,
    )?;

    let command_buffer = new_command_buffer(&runtime.queue)?;
    encode_fast444_region_texture_decode(
        runtime,
        &command_buffer,
        first,
        &entropy_buffers,
        &status_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        shape,
    )?;
    dispatch_rgba_texture_pack(
        &command_buffer,
        &runtime.pipelines.pack_444_rgba_texture,
        (&y_plane, &cb_plane, &cr_plane),
        output,
        shape.pack_params,
        shape.tile_count,
        shape.out_dims,
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
        shape.out_dims,
        requests.len(),
    )?))
}

#[cfg(target_os = "macos")]
fn try_decode_grouped_fast444_region_scaled_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    fast444_packets: &[(&JpegFast444PacketV1, PlaneMode)],
    output: &crate::MetalBatchTextureOutput,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    for (request, (packet, _)) in requests.iter().zip(fast444_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        let scaled = roi.scaled_covering(scale);
        let scaled_roi = j2k_jpeg::Rect {
            x: scaled.x,
            y: scaled.y,
            w: scaled.w,
            h: scaled.h,
        };
        let Some(params) = fast444_scaled_region_params(packet, scale, scaled_roi) else {
            return Ok(None);
        };
        let out_dims = (params.scaled_width, params.scaled_height);
        let out_tile_len =
            out_dims.0 as usize * out_dims.1 as usize * PixelFormat::Rgba8.bytes_per_pixel();
        validate_rgba_texture_batch_output(output, out_dims, requests.len(), out_tile_len)?;
    }

    let mut result_budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal grouped fast444 texture results",
        requests,
    )?;
    let mut merged_results = result_budget.try_filled(
        requests.len(),
        None,
        "JPEG Metal grouped fast444 texture result slots",
    )?;
    for group_indices in groups {
        let group_output = output.clone_slots(&group_indices)?;
        let mut group_budget = crate::plan_owner_ledger::batch_execution_budget(
            "JPEG Metal grouped fast444 texture sub-batch",
            requests,
        )?;
        let mut group_requests = group_budget.try_vec(
            group_indices.len(),
            "JPEG Metal grouped fast444 texture requests",
        )?;
        group_requests.extend(group_indices.iter().map(|&index| requests[index].clone()));
        let mut group_packets = group_budget.try_vec(
            group_indices.len(),
            "JPEG Metal grouped fast444 texture packets",
        )?;
        group_packets.extend(group_indices.iter().map(|&index| {
            let (packet, mode) = fast444_packets[index];
            BatchedFastPacket::Fast444(packet, mode)
        }));
        batch::stamp_execution_owner_baseline(&mut group_requests, 0, group_budget.live_bytes());

        let Some(group_results) = try_decode_fast444_region_scaled_rgba_batch_to_textures(
            runtime,
            &group_requests,
            &group_packets,
            &group_output,
        )?
        else {
            return Ok(None);
        };
        if group_results.len() != group_indices.len() {
            return Err(Error::MetalKernel {
                message: "JPEG Metal grouped fast444 region scaled texture result count mismatch"
                    .to_string(),
            });
        }
        for (original_index, result) in group_indices.into_iter().zip(group_results) {
            merged_results[original_index] = Some(result);
        }
    }

    let mut results = result_budget.try_vec(
        requests.len(),
        "JPEG Metal ordered grouped fast444 texture results",
    )?;
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal grouped fast444 region scaled texture result for tile {index} was missing"
            ),
        })?);
    }
    Ok(Some(results))
}
