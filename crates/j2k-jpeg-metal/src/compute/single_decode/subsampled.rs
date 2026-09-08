// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::metal_types::prelude::*;

use super::scratch::{
    packet_buffers as scratch_packet_buffers, status_buffer as scratch_status_buffer,
};
use crate::buffers::new_shared_buffer;

use super::super::{
    bind_fast_decode_entropy_inputs, bind_three_plane_pack, checked_entropy_segment_count,
    commit_and_wait_jpeg, decode_status_buffer, dispatch_1d_pipeline, dispatch_2d_pipeline,
    encode_fast_subsampled_region_batch_item, encode_fast_subsampled_scaled_batch_item,
    entropy_checkpoints_buffer, entropy_decode_thread_count, fast422_status_error,
    fast_decode_status_error, fast_packet_huffman_tables, fast_subsampled_full_mcu_scaled_window,
    fast_subsampled_params, fast_subsampled_scaled_params, fast_subsampled_scaled_region_params,
    fast_subsampled_windowed_pack_params_for_dims, first_decode_error_status, mcu_range_for_rect,
    new_command_buffer, new_compute_command_encoder, new_decode_plane_buffer, new_private_buffer,
    new_shared_buffer_with_data, pixel_format_to_out_format, restart_offsets_buffer,
    restart_work_for_mcu_range, CpuDecoder, Error, FastDecodeEntropyInputs, FastRgbDecodeBuffer,
    FastSubsampledMetal, JpegDecodeStatus, JpegFast420PacketV1, JpegFast420Params,
    JpegFast420ScaledParams, JpegFast420WindowedPackParams, JpegFast422PacketV1,
    MTLResourceOptions, MetalRuntime, PixelFormat, Rect, Surface,
};

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast422_to_surface(
    runtime: &MetalRuntime,
    packet: Option<&JpegFast422PacketV1>,
    fmt: PixelFormat,
) -> Result<Option<Surface>, Error> {
    try_decode_fast_subsampled_to_surface(runtime, packet, fmt, fast422_status_error)
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn decode_fast422_to_rgb_buffer(
    runtime: &MetalRuntime,
    packet: Option<&JpegFast422PacketV1>,
    fmt: PixelFormat,
    output_storage: MTLResourceOptions,
) -> Result<Option<FastRgbDecodeBuffer>, Error> {
    decode_fast_subsampled_to_rgb_buffer(
        runtime,
        packet,
        fmt,
        output_storage,
        fast422_status_error,
        false,
    )
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_to_surface<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    packet: Option<&P>,
    fmt: PixelFormat,
    map_status: impl Fn(JpegDecodeStatus) -> Error,
) -> Result<Option<Surface>, Error> {
    let Some(decoded) = decode_fast_subsampled_to_rgb_buffer(
        runtime,
        packet,
        fmt,
        MTLResourceOptions::StorageModeShared,
        map_status,
        true,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(Surface::from_metal_buffer(
        decoded.buffer,
        decoded.dimensions,
        fmt,
    )?))
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::similar_names,
    reason = "Cb and Cr are normative JPEG component names"
)]
fn decode_fast_subsampled_to_rgb_buffer<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    packet: Option<&P>,
    fmt: PixelFormat,
    output_storage: MTLResourceOptions,
    map_status: impl Fn(JpegDecodeStatus) -> Error,
    reuse_temporaries: bool,
) -> Result<Option<FastRgbDecodeBuffer>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_out_format) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };

    let params = fast_subsampled_params(packet, fmt)?;
    let y_len = params.width as usize * params.height as usize;
    let chroma_len = params.chroma_width as usize * params.chroma_height as usize;
    let mut scratch = reuse_temporaries
        .then(|| runtime.batch_scratch())
        .transpose()?;
    let y_plane = if fmt == PixelFormat::Gray8 {
        new_decode_plane_buffer(&runtime.device, y_len, true)?
    } else if let Some(scratch) = scratch.as_deref_mut() {
        scratch.private_buffer(&runtime.device, "single_decode_y", y_len)?
    } else {
        new_private_buffer(&runtime.device, y_len)?
    };
    let cb_plane = if let Some(scratch) = scratch.as_deref_mut() {
        scratch.private_buffer(&runtime.device, "single_decode_cb", chroma_len)?
    } else {
        new_private_buffer(&runtime.device, chroma_len)?
    };
    let cr_plane = if let Some(scratch) = scratch.as_deref_mut() {
        scratch.private_buffer(&runtime.device, "single_decode_cr", chroma_len)?
    } else {
        new_private_buffer(&runtime.device, chroma_len)?
    };
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus(),
        packet.restart_offsets().len(),
        packet.entropy_checkpoints().len(),
    );
    let (status_buffer, entropy_buffer, restart_offsets_buffer, entropy_checkpoints_buffer) =
        if let Some(scratch) = scratch.as_deref_mut() {
            let packet_buffers = scratch_packet_buffers(
                scratch,
                &runtime.device,
                packet.entropy_bytes(),
                packet.restart_offsets(),
                packet.entropy_checkpoints(),
            )?;
            (
                scratch_status_buffer(scratch, &runtime.device, decode_threads)?,
                packet_buffers.entropy,
                packet_buffers.restart_offsets,
                packet_buffers.checkpoints,
            )
        } else {
            (
                decode_status_buffer(&runtime.device, decode_threads)?,
                new_shared_buffer_with_data(&runtime.device, packet.entropy_bytes())?,
                restart_offsets_buffer(&runtime.device, packet.restart_offsets())?,
                entropy_checkpoints_buffer(&runtime.device, packet.entropy_checkpoints())?,
            )
        };

    let (dc_tables, ac_tables) = fast_packet_huffman_tables(packet);

    let out_buffer = if fmt == PixelFormat::Gray8 {
        None
    } else {
        let out_len = crate::batch_allocation::checked_count_product(
            params.out_stride as usize,
            params.height as usize,
            "JPEG Metal subsampled output bytes",
        )?;
        Some(
            if output_storage == MTLResourceOptions::StorageModePrivate {
                new_private_buffer(&runtime.device, out_len)?
            } else {
                new_shared_buffer(&runtime.device, out_len)?
            },
        )
    };

    let decode_pipeline = P::decode_pipeline(runtime);
    let command_buffer = new_command_buffer(&runtime.queue)?;
    let decoder_encoder = new_compute_command_encoder(&command_buffer)?;
    decoder_encoder.setComputePipelineState(decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast420Params>(
        &decoder_encoder,
        &FastDecodeEntropyInputs {
            entropy_buffer: &entropy_buffer,
            planes: [&y_plane, &cb_plane, &cr_plane],
            params: &params,
            quants: [packet.y_quant(), packet.cb_quant(), packet.cr_quant()],
            dc_tables: &dc_tables,
            ac_tables: &ac_tables,
            slot14_buffer: &restart_offsets_buffer,
            slot15_buffer: &status_buffer,
            slot16_buffer: &entropy_checkpoints_buffer,
        },
    );
    dispatch_1d_pipeline(&decoder_encoder, decode_pipeline, decode_threads);
    decoder_encoder.endEncoding();

    if let Some(out_buffer) = out_buffer.as_ref() {
        let Some(pack_pipeline) = P::pack_pipeline_for_format(runtime, fmt) else {
            return Ok(None);
        };
        let pack_encoder = new_compute_command_encoder(&command_buffer)?;
        pack_encoder.setComputePipelineState(pack_pipeline);
        pack_encoder.bind_buffer(0, Some(&y_plane), 0);
        pack_encoder.bind_buffer(1, Some(&cb_plane), 0);
        pack_encoder.bind_buffer(2, Some(&cr_plane), 0);
        pack_encoder.bind_buffer(3, Some(out_buffer), 0);
        pack_encoder.bind_bytes::<JpegFast420Params>(4, &params);
        dispatch_2d_pipeline(&pack_encoder, pack_pipeline, packet.dimensions());
        pack_encoder.endEncoding();
    }

    commit_and_wait_jpeg(&command_buffer)?;
    let command_buffer = command_buffer.clone();

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads)? {
        return Err(map_status(status));
    }

    Ok(Some(FastRgbDecodeBuffer {
        buffer: out_buffer.unwrap_or(y_plane),
        dimensions: packet.dimensions(),
        status_buffer,
        command_buffer,
    }))
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_region_to_surface<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    packet: Option<&P>,
    fmt: PixelFormat,
    roi: j2k_jpeg::Rect,
    map_status: impl Fn(JpegDecodeStatus) -> Error,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };

    let command_buffer = new_command_buffer(&runtime.queue)?;
    let mut scratch = runtime.batch_scratch()?;
    let item = encode_fast_subsampled_region_batch_item(
        runtime,
        &command_buffer,
        packet,
        fmt,
        Rect {
            x: roi.x,
            y: roi.y,
            w: roi.w,
            h: roi.h,
        },
        Some(&mut scratch),
    )?;
    commit_and_wait_jpeg(&command_buffer)?;

    if let Some(status) = first_decode_error_status(&item.status_buffer, item.decode_threads)? {
        return Err(map_status(status));
    }

    Ok(Some(item.surface))
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_scaled_to_surface<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    packet: Option<&P>,
    fmt: PixelFormat,
    scale: j2k_core::Downscale,
    map_status: impl Fn(JpegDecodeStatus) -> Error,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };
    if fast_subsampled_scaled_params(packet, scale).is_none() {
        return Ok(None);
    }

    let command_buffer = new_command_buffer(&runtime.queue)?;
    let mut scratch = runtime.batch_scratch()?;
    let item = encode_fast_subsampled_scaled_batch_item(
        runtime,
        &command_buffer,
        packet,
        fmt,
        scale,
        Some(&mut scratch),
    )?;
    commit_and_wait_jpeg(&command_buffer)?;

    if let Some(status) = first_decode_error_status(&item.status_buffer, item.decode_threads)? {
        return Err(map_status(status));
    }

    Ok(Some(item.surface))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast422_region_to_surface(
    runtime: &MetalRuntime,
    packet: Option<&JpegFast422PacketV1>,
    fmt: PixelFormat,
    roi: j2k_jpeg::Rect,
) -> Result<Option<Surface>, Error> {
    try_decode_fast_subsampled_region_to_surface(runtime, packet, fmt, roi, fast422_status_error)
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast422_scaled_to_surface(
    runtime: &MetalRuntime,
    packet: Option<&JpegFast422PacketV1>,
    fmt: PixelFormat,
    scale: j2k_core::Downscale,
) -> Result<Option<Surface>, Error> {
    try_decode_fast_subsampled_scaled_to_surface(runtime, packet, fmt, scale, fast422_status_error)
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast422_scaled_region_to_surface(
    runtime: &MetalRuntime,
    packet: Option<&JpegFast422PacketV1>,
    fmt: PixelFormat,
    scaled_roi: j2k_jpeg::Rect,
    scale: j2k_core::Downscale,
) -> Result<Option<Surface>, Error> {
    try_decode_fast_subsampled_scaled_region_to_surface(
        runtime,
        packet,
        fmt,
        scaled_roi,
        scale,
        |status| Ok(fast422_status_error(status)),
    )
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "the single-image path keeps ROI validation, Metal buffer lifetimes, command encoding, and status mapping in submission order"
)]
#[expect(
    clippy::similar_names,
    reason = "Cb and Cr are normative JPEG component names"
)]
fn try_decode_fast_subsampled_scaled_region_to_surface<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    packet: Option<&P>,
    fmt: PixelFormat,
    scaled_roi: j2k_jpeg::Rect,
    scale: j2k_core::Downscale,
    map_status: impl FnOnce(JpegDecodeStatus) -> Result<Error, Error>,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };
    let Some(full_params) = fast_subsampled_scaled_params(packet, scale) else {
        return Ok(None);
    };
    let source_window = fast_subsampled_full_mcu_scaled_window::<P>(
        (full_params.scaled_width, full_params.scaled_height),
        scaled_roi,
        full_params.scale_shift,
    );
    let Some(mut decode_params) =
        fast_subsampled_scaled_region_params(packet, scale, source_window)
    else {
        return Ok(None);
    };
    let mcu_width = P::MCU_WIDTH >> decode_params.scale_shift;
    let mcu_height = P::MCU_HEIGHT >> decode_params.scale_shift;
    let (first_mcu, end_mcu) = mcu_range_for_rect(
        source_window,
        packet.mcus_per_row(),
        packet.mcu_rows(),
        mcu_width,
        mcu_height,
    );
    let total_mcus = packet.mcus_per_row() * packet.mcu_rows();
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        packet.restart_offsets(),
        packet.restart_interval_mcus(),
        total_mcus,
        first_mcu,
        end_mcu,
    );
    decode_params.restart_start_mcu = restart_start_mcu;
    decode_params.restart_offset_count = checked_entropy_segment_count(
        packet.restart_interval_mcus(),
        restart_offsets.len(),
        packet.entropy_checkpoints().len(),
    )?;
    let local_roi = j2k_jpeg::Rect {
        x: scaled_roi.x - source_window.x,
        y: scaled_roi.y - source_window.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let pack_params = fast_subsampled_windowed_pack_params_for_dims::<P>(
        (source_window.w, source_window.h),
        fmt,
        local_roi,
    )?;
    let y_len = source_window.w as usize * source_window.h as usize;
    let chroma_len =
        source_window.w.div_ceil(2) as usize * P::chroma_height(source_window.h) as usize;
    let mut scratch = runtime.batch_scratch()?;
    let y_plane = scratch.private_buffer(&runtime.device, "single_decode_y", y_len)?;
    let cb_plane = scratch.private_buffer(&runtime.device, "single_decode_cb", chroma_len)?;
    let cr_plane = scratch.private_buffer(&runtime.device, "single_decode_cr", chroma_len)?;
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus(),
        restart_offsets.len(),
        packet.entropy_checkpoints().len(),
    );
    let packet_buffers = scratch_packet_buffers(
        &mut scratch,
        &runtime.device,
        packet.entropy_bytes(),
        restart_offsets,
        packet.entropy_checkpoints(),
    )?;
    let status_buffer = scratch_status_buffer(&mut scratch, &runtime.device, decode_threads)?;
    let entropy_buffer = packet_buffers.entropy;
    let restart_offsets_buffer = packet_buffers.restart_offsets;
    let entropy_checkpoints_buffer = packet_buffers.checkpoints;

    let (dc_tables, ac_tables) = fast_packet_huffman_tables(packet);

    let out_len = crate::batch_allocation::checked_count_product(
        pack_params.out_stride as usize,
        scaled_roi.h as usize,
        "JPEG Metal subsampled region output bytes",
    )?;
    let out_buffer = new_shared_buffer(&runtime.device, out_len)?;

    let decode_pipeline = P::scaled_region_decode_pipeline(runtime);
    let command_buffer = new_command_buffer(&runtime.queue)?;
    let decoder_encoder = new_compute_command_encoder(&command_buffer)?;
    decoder_encoder.setComputePipelineState(decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast420ScaledParams>(
        &decoder_encoder,
        &FastDecodeEntropyInputs {
            entropy_buffer: &entropy_buffer,
            planes: [&y_plane, &cb_plane, &cr_plane],
            params: &decode_params,
            quants: [packet.y_quant(), packet.cb_quant(), packet.cr_quant()],
            dc_tables: &dc_tables,
            ac_tables: &ac_tables,
            slot14_buffer: &restart_offsets_buffer,
            slot15_buffer: &status_buffer,
            slot16_buffer: &entropy_checkpoints_buffer,
        },
    );
    dispatch_1d_pipeline(&decoder_encoder, decode_pipeline, decode_threads);
    decoder_encoder.endEncoding();

    let pack_encoder = new_compute_command_encoder(&command_buffer)?;
    let pack_pipeline = P::pack_windowed_pipeline_for_format(runtime, fmt);
    pack_encoder.setComputePipelineState(pack_pipeline);
    bind_three_plane_pack::<JpegFast420WindowedPackParams>(
        &pack_encoder,
        [Some(&y_plane), Some(&cb_plane), Some(&cr_plane)],
        &out_buffer,
        &pack_params,
    );
    dispatch_2d_pipeline(&pack_encoder, pack_pipeline, (scaled_roi.w, scaled_roi.h));
    pack_encoder.endEncoding();

    commit_and_wait_jpeg(&command_buffer)?;

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads)? {
        return Err(map_status(status)?);
    }

    Ok(Some(Surface::from_metal_buffer(
        out_buffer,
        (scaled_roi.w, scaled_roi.h),
        fmt,
    )?))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast420_to_surface(
    runtime: &MetalRuntime,
    _decoder: &CpuDecoder<'_>,
    packet: Option<&JpegFast420PacketV1>,
    fmt: PixelFormat,
) -> Result<Option<Surface>, Error> {
    try_decode_fast_subsampled_to_surface(runtime, packet, fmt, fast_decode_status_error)
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn decode_fast420_to_rgb_buffer(
    runtime: &MetalRuntime,
    _decoder: &CpuDecoder<'_>,
    packet: Option<&JpegFast420PacketV1>,
    fmt: PixelFormat,
    output_storage: MTLResourceOptions,
) -> Result<Option<FastRgbDecodeBuffer>, Error> {
    decode_fast_subsampled_to_rgb_buffer(
        runtime,
        packet,
        fmt,
        output_storage,
        fast_decode_status_error,
        false,
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast420_region_to_surface(
    runtime: &MetalRuntime,
    _decoder: &CpuDecoder<'_>,
    packet: Option<&JpegFast420PacketV1>,
    fmt: PixelFormat,
    roi: j2k_jpeg::Rect,
) -> Result<Option<Surface>, Error> {
    try_decode_fast_subsampled_region_to_surface(
        runtime,
        packet,
        fmt,
        roi,
        fast_decode_status_error,
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast420_scaled_to_surface(
    runtime: &MetalRuntime,
    _decoder: &CpuDecoder<'_>,
    packet: Option<&JpegFast420PacketV1>,
    fmt: PixelFormat,
    scale: j2k_core::Downscale,
) -> Result<Option<Surface>, Error> {
    try_decode_fast_subsampled_scaled_to_surface(
        runtime,
        packet,
        fmt,
        scale,
        fast_decode_status_error,
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast420_scaled_region_to_surface(
    runtime: &MetalRuntime,
    _decoder: &CpuDecoder<'_>,
    packet: Option<&JpegFast420PacketV1>,
    fmt: PixelFormat,
    scaled_roi: j2k_jpeg::Rect,
    scale: j2k_core::Downscale,
) -> Result<Option<Surface>, Error> {
    try_decode_fast420_scaled_region_to_surface_with_status(
        runtime,
        packet,
        fmt,
        scaled_roi,
        scale,
        |status| Ok(fast_decode_status_error(status)),
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast420_scaled_region_to_surface_with_status(
    runtime: &MetalRuntime,
    packet: Option<&JpegFast420PacketV1>,
    fmt: PixelFormat,
    scaled_roi: j2k_jpeg::Rect,
    scale: j2k_core::Downscale,
    map_status: impl FnOnce(JpegDecodeStatus) -> Result<Error, Error>,
) -> Result<Option<Surface>, Error> {
    try_decode_fast_subsampled_scaled_region_to_surface(
        runtime, packet, fmt, scaled_roi, scale, map_status,
    )
}
