// SPDX-License-Identifier: MIT OR Apache-2.0

use super::scratch::decode_buffers as scratch_decode_buffers;
use crate::metal_types::prelude::*;

use super::super::{
    bind_fast_decode_entropy_inputs, checked_entropy_segment_count, commit_and_wait_jpeg,
    decode_status_buffer, dispatch_1d_pipeline, entropy_checkpoints_buffer,
    entropy_decode_thread_count, fast444_params, fast444_region_params, fast444_scaled_params,
    fast444_scaled_region_params, fast_decode_status_error, fast_packet_huffman_tables,
    first_decode_error_status, mcu_range_for_rect, new_command_buffer, new_compute_command_encoder,
    new_decode_plane_buffer, new_private_buffer, new_shared_buffer_with_data,
    pixel_format_to_out_format, restart_offsets_buffer, restart_work_for_mcu_range, CpuDecoder,
    Error, FastDecodeEntropyInputs, JpegColorSpace, JpegDecodeStatus, JpegFast444PacketV1,
    JpegFast444Params, JpegFast444ScaledParams, MetalRuntime, PixelFormat, PlaneMode, PlaneStage,
    Surface,
};

#[cfg(target_os = "macos")]
pub(in crate::compute) fn fast444_plane_mode(decoder: &CpuDecoder<'_>) -> PlaneMode {
    match decoder.info().color_space {
        JpegColorSpace::Rgb => PlaneMode::Rgb,
        _ => PlaneMode::YCbCr,
    }
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast444_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegFast444PacketV1>,
    fmt: PixelFormat,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };

    let params = fast444_params(packet)?;
    let mode = fast444_plane_mode(decoder);
    let plane_len = params.width as usize * params.height as usize;
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let mut scratch = runtime.batch_scratch()?;
    let buffers = scratch_decode_buffers(
        &mut scratch,
        &runtime.device,
        plane_len,
        fmt == PixelFormat::Gray8 && mode != PlaneMode::Rgb,
        decode_threads,
        &packet.entropy_bytes,
        &packet.restart_offsets,
        &packet.entropy_checkpoints,
    )?;
    let y_plane = buffers.plane0;
    let chroma_blue_plane = buffers.plane1;
    let chroma_red_plane = buffers.plane2;
    let status_buffer = buffers.status;
    let entropy_buffer = buffers.packet.entropy;
    let restart_offsets_buffer = buffers.packet.restart_offsets;
    let entropy_checkpoints_buffer = buffers.packet.checkpoints;

    let (dc_tables, ac_tables) = fast_packet_huffman_tables(packet);

    let command_buffer = new_command_buffer(&runtime.queue)?;
    let decoder_encoder = new_compute_command_encoder(&command_buffer)?;
    decoder_encoder.setComputePipelineState(&runtime.pipelines.fast444_decode);
    bind_fast_decode_entropy_inputs::<JpegFast444Params>(
        &decoder_encoder,
        &FastDecodeEntropyInputs {
            entropy_buffer: &entropy_buffer,
            planes: [&y_plane, &chroma_blue_plane, &chroma_red_plane],
            params: &params,
            quants: [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
            dc_tables: &dc_tables,
            ac_tables: &ac_tables,
            slot14_buffer: &restart_offsets_buffer,
            slot15_buffer: &status_buffer,
            slot16_buffer: &entropy_checkpoints_buffer,
        },
    );
    dispatch_1d_pipeline(
        &decoder_encoder,
        &runtime.pipelines.fast444_decode,
        decode_threads,
    );
    decoder_encoder.endEncoding();
    commit_and_wait_jpeg(&command_buffer)?;

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads)? {
        return Err(fast_decode_status_error(status));
    }

    PlaneStage {
        dims: packet.dimensions,
        mode,
        plane0: y_plane,
        plane1: Some(chroma_blue_plane),
        plane2: Some(chroma_red_plane),
        cache_lease: None,
    }
    .finish_resident_with_runtime(runtime, fmt)
    .map(Some)
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast444_to_private_rgb8_tile(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegFast444PacketV1>,
) -> Result<Option<crate::ResidentPrivateJpegTile>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };

    let params = fast444_params(packet)?;
    let mode = fast444_plane_mode(decoder);
    let plane_len = params.width as usize * params.height as usize;
    let y_plane = new_private_buffer(&runtime.device, plane_len)?;
    let chroma_blue_plane = new_private_buffer(&runtime.device, plane_len)?;
    let chroma_red_plane = new_private_buffer(&runtime.device, plane_len)?;
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads)?;
    let entropy_buffer = new_shared_buffer_with_data(&runtime.device, &packet.entropy_bytes)?;
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let (dc_tables, ac_tables) = fast_packet_huffman_tables(packet);

    let command_buffer = new_command_buffer(&runtime.queue)?;
    let decoder_encoder = new_compute_command_encoder(&command_buffer)?;
    decoder_encoder.setComputePipelineState(&runtime.pipelines.fast444_decode);
    bind_fast_decode_entropy_inputs::<JpegFast444Params>(
        &decoder_encoder,
        &FastDecodeEntropyInputs {
            entropy_buffer: &entropy_buffer,
            planes: [&y_plane, &chroma_blue_plane, &chroma_red_plane],
            params: &params,
            quants: [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
            dc_tables: &dc_tables,
            ac_tables: &ac_tables,
            slot14_buffer: &restart_offsets_buffer,
            slot15_buffer: &status_buffer,
            slot16_buffer: &entropy_checkpoints_buffer,
        },
    );
    dispatch_1d_pipeline(
        &decoder_encoder,
        &runtime.pipelines.fast444_decode,
        decode_threads,
    );
    decoder_encoder.endEncoding();
    commit_and_wait_jpeg(&command_buffer)?;

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads)? {
        return Err(fast_decode_status_error(status));
    }

    Ok(Some(
        PlaneStage {
            dims: packet.dimensions,
            mode,
            plane0: y_plane,
            plane1: Some(chroma_blue_plane),
            plane2: Some(chroma_red_plane),
            cache_lease: None,
        }
        .dispatch_private_rgb8_with_runtime(runtime, status_buffer)?,
    ))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast444_region_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegFast444PacketV1>,
    fmt: PixelFormat,
    roi: j2k_jpeg::Rect,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };

    let mut params = fast444_region_params(packet, roi)?;
    let (first_mcu, end_mcu) = mcu_range_for_rect(roi, packet.mcus_per_row, packet.mcu_rows, 8, 8);
    let total_mcus = packet.mcus_per_row * packet.mcu_rows;
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        &packet.restart_offsets,
        packet.restart_interval_mcus,
        total_mcus,
        first_mcu,
        end_mcu,
    );
    params.restart_start_mcu = restart_start_mcu;
    params.restart_offset_count = checked_entropy_segment_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    )?;
    let mode = fast444_plane_mode(decoder);
    let plane_len = params.width as usize * params.height as usize;
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let mut scratch = runtime.batch_scratch()?;
    let buffers = scratch_decode_buffers(
        &mut scratch,
        &runtime.device,
        plane_len,
        fmt == PixelFormat::Gray8 && mode != PlaneMode::Rgb,
        decode_threads,
        &packet.entropy_bytes,
        restart_offsets,
        &packet.entropy_checkpoints,
    )?;
    let y_plane = buffers.plane0;
    let chroma_blue_plane = buffers.plane1;
    let chroma_red_plane = buffers.plane2;
    let status_buffer = buffers.status;
    let entropy_buffer = buffers.packet.entropy;
    let restart_offsets_buffer = buffers.packet.restart_offsets;
    let entropy_checkpoints_buffer = buffers.packet.checkpoints;

    let (dc_tables, ac_tables) = fast_packet_huffman_tables(packet);

    let command_buffer = new_command_buffer(&runtime.queue)?;
    let decoder_encoder = new_compute_command_encoder(&command_buffer)?;
    decoder_encoder.setComputePipelineState(&runtime.pipelines.fast444_region_decode);
    bind_fast_decode_entropy_inputs::<JpegFast444Params>(
        &decoder_encoder,
        &FastDecodeEntropyInputs {
            entropy_buffer: &entropy_buffer,
            planes: [&y_plane, &chroma_blue_plane, &chroma_red_plane],
            params: &params,
            quants: [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
            dc_tables: &dc_tables,
            ac_tables: &ac_tables,
            slot14_buffer: &restart_offsets_buffer,
            slot15_buffer: &status_buffer,
            slot16_buffer: &entropy_checkpoints_buffer,
        },
    );
    dispatch_1d_pipeline(
        &decoder_encoder,
        &runtime.pipelines.fast444_region_decode,
        decode_threads,
    );
    decoder_encoder.endEncoding();
    commit_and_wait_jpeg(&command_buffer)?;

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads)? {
        return Err(fast_decode_status_error(status));
    }

    PlaneStage {
        dims: (roi.w, roi.h),
        mode,
        plane0: y_plane,
        plane1: Some(chroma_blue_plane),
        plane2: Some(chroma_red_plane),
        cache_lease: None,
    }
    .finish_resident_with_runtime(runtime, fmt)
    .map(Some)
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast444_scaled_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegFast444PacketV1>,
    fmt: PixelFormat,
    scale: j2k_core::Downscale,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };
    let Some(params) = fast444_scaled_params(packet, scale) else {
        return Ok(None);
    };

    let mode = fast444_plane_mode(decoder);
    let plane_len = params.scaled_width as usize * params.scaled_height as usize;
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let mut scratch = runtime.batch_scratch()?;
    let buffers = scratch_decode_buffers(
        &mut scratch,
        &runtime.device,
        plane_len,
        fmt == PixelFormat::Gray8 && mode != PlaneMode::Rgb,
        decode_threads,
        &packet.entropy_bytes,
        &packet.restart_offsets,
        &packet.entropy_checkpoints,
    )?;
    let y_plane = buffers.plane0;
    let chroma_blue_plane = buffers.plane1;
    let chroma_red_plane = buffers.plane2;
    let status_buffer = buffers.status;
    let entropy_buffer = buffers.packet.entropy;
    let restart_offsets_buffer = buffers.packet.restart_offsets;
    let entropy_checkpoints_buffer = buffers.packet.checkpoints;

    let (dc_tables, ac_tables) = fast_packet_huffman_tables(packet);

    let command_buffer = new_command_buffer(&runtime.queue)?;
    let decoder_encoder = new_compute_command_encoder(&command_buffer)?;
    decoder_encoder.setComputePipelineState(&runtime.pipelines.fast444_scaled_decode);
    bind_fast_decode_entropy_inputs::<JpegFast444ScaledParams>(
        &decoder_encoder,
        &FastDecodeEntropyInputs {
            entropy_buffer: &entropy_buffer,
            planes: [&y_plane, &chroma_blue_plane, &chroma_red_plane],
            params: &params,
            quants: [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
            dc_tables: &dc_tables,
            ac_tables: &ac_tables,
            slot14_buffer: &restart_offsets_buffer,
            slot15_buffer: &status_buffer,
            slot16_buffer: &entropy_checkpoints_buffer,
        },
    );
    dispatch_1d_pipeline(
        &decoder_encoder,
        &runtime.pipelines.fast444_scaled_decode,
        decode_threads,
    );
    decoder_encoder.endEncoding();
    commit_and_wait_jpeg(&command_buffer)?;

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads)? {
        return Err(fast_decode_status_error(status));
    }

    PlaneStage {
        dims: (params.scaled_width, params.scaled_height),
        mode,
        plane0: y_plane,
        plane1: Some(chroma_blue_plane),
        plane2: Some(chroma_red_plane),
        cache_lease: None,
    }
    .finish_resident_with_runtime(runtime, fmt)
    .map(Some)
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast444_scaled_region_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegFast444PacketV1>,
    fmt: PixelFormat,
    scaled_roi: j2k_jpeg::Rect,
    scale: j2k_core::Downscale,
) -> Result<Option<Surface>, Error> {
    let mode = fast444_plane_mode(decoder);
    try_decode_fast444_scaled_region_to_surface_with_mode_and_status(
        runtime,
        packet,
        fmt,
        scaled_roi,
        scale,
        mode,
        |status| Ok(fast_decode_status_error(status)),
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast444_scaled_region_to_surface_with_mode_and_status(
    runtime: &MetalRuntime,
    packet: Option<&JpegFast444PacketV1>,
    fmt: PixelFormat,
    scaled_roi: j2k_jpeg::Rect,
    scale: j2k_core::Downscale,
    mode: PlaneMode,
    map_status: impl FnOnce(JpegDecodeStatus) -> Result<Error, Error>,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };
    let Some(mut params) = fast444_scaled_region_params(packet, scale, scaled_roi) else {
        return Ok(None);
    };
    let mcu_size = 8u32 >> params.scale_shift;
    let (first_mcu, end_mcu) = mcu_range_for_rect(
        scaled_roi,
        packet.mcus_per_row,
        packet.mcu_rows,
        mcu_size,
        mcu_size,
    );
    let total_mcus = packet.mcus_per_row * packet.mcu_rows;
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        &packet.restart_offsets,
        packet.restart_interval_mcus,
        total_mcus,
        first_mcu,
        end_mcu,
    );
    params.restart_start_mcu = restart_start_mcu;
    params.restart_offset_count = checked_entropy_segment_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    )?;

    let plane_len = params.scaled_width as usize * params.scaled_height as usize;
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let mut scratch = runtime.batch_scratch()?;
    let buffers = scratch_decode_buffers(
        &mut scratch,
        &runtime.device,
        plane_len,
        fmt == PixelFormat::Gray8 && mode != PlaneMode::Rgb,
        decode_threads,
        &packet.entropy_bytes,
        restart_offsets,
        &packet.entropy_checkpoints,
    )?;
    let y_plane = buffers.plane0;
    let chroma_blue_plane = buffers.plane1;
    let chroma_red_plane = buffers.plane2;
    let status_buffer = buffers.status;
    let entropy_buffer = buffers.packet.entropy;
    let restart_offsets_buffer = buffers.packet.restart_offsets;
    let entropy_checkpoints_buffer = buffers.packet.checkpoints;

    let (dc_tables, ac_tables) = fast_packet_huffman_tables(packet);

    let command_buffer = new_command_buffer(&runtime.queue)?;
    let decoder_encoder = new_compute_command_encoder(&command_buffer)?;
    decoder_encoder.setComputePipelineState(&runtime.pipelines.fast444_scaled_region_decode);
    bind_fast_decode_entropy_inputs::<JpegFast444ScaledParams>(
        &decoder_encoder,
        &FastDecodeEntropyInputs {
            entropy_buffer: &entropy_buffer,
            planes: [&y_plane, &chroma_blue_plane, &chroma_red_plane],
            params: &params,
            quants: [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
            dc_tables: &dc_tables,
            ac_tables: &ac_tables,
            slot14_buffer: &restart_offsets_buffer,
            slot15_buffer: &status_buffer,
            slot16_buffer: &entropy_checkpoints_buffer,
        },
    );
    dispatch_1d_pipeline(
        &decoder_encoder,
        &runtime.pipelines.fast444_scaled_region_decode,
        decode_threads,
    );
    decoder_encoder.endEncoding();
    commit_and_wait_jpeg(&command_buffer)?;

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads)? {
        return Err(map_status(status)?);
    }

    PlaneStage {
        dims: (scaled_roi.w, scaled_roi.h),
        mode,
        plane0: y_plane,
        plane1: Some(chroma_blue_plane),
        plane2: Some(chroma_red_plane),
        cache_lease: None,
    }
    .finish_resident_with_runtime(runtime, fmt)
    .map(Some)
}
