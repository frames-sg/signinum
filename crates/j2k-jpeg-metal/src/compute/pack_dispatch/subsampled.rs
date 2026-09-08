// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::metal_types::prelude::*;

use crate::buffers::new_shared_buffer;
use crate::buffers::MetalBatchScratch;
use crate::compute::single_decode::scratch::{
    packet_buffers as scratch_packet_buffers, status_buffer as scratch_status_buffer,
};

use super::super::{
    batch, bind_fast_decode_entropy_inputs, bind_three_plane_pack, checked_entropy_segment_count,
    core_rect_to_jpeg, decode_status_buffer, dispatch_1d_pipeline, dispatch_2d_pipeline,
    entropy_checkpoints_buffer, entropy_decode_thread_count, fast_packet_huffman_tables,
    fast_subsampled_full_mcu_scaled_window, fast_subsampled_full_mcu_window,
    fast_subsampled_params, fast_subsampled_region_params, fast_subsampled_scaled_params,
    fast_subsampled_scaled_region_params, fast_subsampled_windowed_pack_params_for_dims,
    mcu_range_for_rect, new_compute_command_encoder, new_decode_plane_buffer, new_private_buffer,
    new_shared_buffer_with_data, pixel_format_to_out_format, restart_offsets_buffer,
    restart_work_for_mcu_range, BatchedDecodeItem, CommandBufferRef, Error,
    FastDecodeEntropyInputs, FastSubsampledMetal, JpegFast420Params, JpegFast420ScaledParams,
    JpegFast420WindowedPackParams, MetalRuntime, PixelFormat, Rect, Surface,
};
use super::conversion::checked_u32;
use super::requests::{
    FastSubsampledOpBatchItemRequest, FastSubsampledScaledRegionBatchItemRequest,
};

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "the region encoder binds one ordered Metal command sequence and retains all buffers required until completion"
)]
#[expect(
    clippy::similar_names,
    reason = "Cb and Cr are normative JPEG component names"
)]
pub(in crate::compute) fn encode_fast_subsampled_region_batch_item<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    packet: &P,
    fmt: PixelFormat,
    roi: Rect,
    mut scratch: Option<&mut MetalBatchScratch>,
) -> Result<BatchedDecodeItem, Error> {
    let roi = core_rect_to_jpeg(roi);
    let source_window = fast_subsampled_full_mcu_window::<P>(packet.dimensions(), roi);
    let mut params = fast_subsampled_region_params(packet, fmt, source_window)?;
    let (first_mcu, end_mcu) = mcu_range_for_rect(
        source_window,
        packet.mcus_per_row(),
        packet.mcu_rows(),
        P::MCU_WIDTH,
        P::MCU_HEIGHT,
    );
    let total_mcus = packet.mcus_per_row() * packet.mcu_rows();
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        packet.restart_offsets(),
        packet.restart_interval_mcus(),
        total_mcus,
        first_mcu,
        end_mcu,
    );
    params.restart_start_mcu = restart_start_mcu;
    params.restart_offset_count = checked_entropy_segment_count(
        packet.restart_interval_mcus(),
        restart_offsets.len(),
        packet.entropy_checkpoints().len(),
    )?;

    let local_roi = j2k_jpeg::Rect {
        x: roi.x - source_window.x,
        y: roi.y - source_window.y,
        w: roi.w,
        h: roi.h,
    };
    let pack_params = fast_subsampled_windowed_pack_params_for_dims::<P>(
        (source_window.w, source_window.h),
        fmt,
        local_roi,
    )?;
    let y_len = source_window.w as usize * source_window.h as usize;
    let chroma_len =
        source_window.w.div_ceil(2) as usize * P::chroma_height(source_window.h) as usize;
    let y_plane = if let Some(scratch) = scratch.as_deref_mut() {
        scratch.private_buffer(&runtime.device, "single_decode_y", y_len)?
    } else {
        new_decode_plane_buffer(&runtime.device, y_len, false)?
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
        restart_offsets.len(),
        packet.entropy_checkpoints().len(),
    );
    let (status_buffer, entropy_buffer, restart_offsets_buffer, entropy_checkpoints_buffer) =
        if let Some(scratch) = scratch {
            let packet_buffers = scratch_packet_buffers(
                scratch,
                &runtime.device,
                packet.entropy_bytes(),
                restart_offsets,
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
                restart_offsets_buffer(&runtime.device, restart_offsets)?,
                entropy_checkpoints_buffer(&runtime.device, packet.entropy_checkpoints())?,
            )
        };

    let (dc_tables, ac_tables) = fast_packet_huffman_tables(packet);

    let decode_pipeline = P::region_decode_pipeline(runtime);
    let decoder_encoder = new_compute_command_encoder(command_buffer)?;
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

    let out_len = crate::batch_allocation::checked_count_product(
        pack_params.out_stride as usize,
        roi.h as usize,
        "JPEG Metal packed region output bytes",
    )?;
    let out_buffer = new_shared_buffer(&runtime.device, out_len)?;
    let pack_encoder = new_compute_command_encoder(command_buffer)?;
    let pack_pipeline = P::pack_windowed_pipeline_for_format(runtime, fmt);
    pack_encoder.setComputePipelineState(pack_pipeline);
    bind_three_plane_pack::<JpegFast420WindowedPackParams>(
        &pack_encoder,
        [Some(&y_plane), Some(&cb_plane), Some(&cr_plane)],
        &out_buffer,
        &pack_params,
    );
    dispatch_2d_pipeline(&pack_encoder, pack_pipeline, (roi.w, roi.h));
    pack_encoder.endEncoding();

    Ok(BatchedDecodeItem {
        surface: Surface::from_metal_buffer(out_buffer, (roi.w, roi.h), fmt)?,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: [
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "the scaled encoder binds one ordered Metal command sequence and retains all buffers required until completion"
)]
#[expect(
    clippy::similar_names,
    reason = "Cb and Cr are normative JPEG component names"
)]
pub(in crate::compute) fn encode_fast_subsampled_scaled_batch_item<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    packet: &P,
    fmt: PixelFormat,
    scale: j2k_core::Downscale,
    mut scratch: Option<&mut MetalBatchScratch>,
) -> Result<BatchedDecodeItem, Error> {
    let Some(params) = fast_subsampled_scaled_params(packet, scale) else {
        return Err(Error::MetalKernel {
            message: format!("unsupported JPEG Metal {} scale {scale:?}", P::FAMILY_NAME),
        });
    };

    let y_len = params.scaled_width as usize * params.scaled_height as usize;
    let chroma_len = params.chroma_width as usize * params.chroma_height as usize;
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
        if let Some(scratch) = scratch {
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

    let decode_pipeline = P::scaled_decode_pipeline(runtime);
    let decoder_encoder = new_compute_command_encoder(command_buffer)?;
    decoder_encoder.setComputePipelineState(decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast420ScaledParams>(
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

    let out_buffer = if fmt == PixelFormat::Gray8 {
        None
    } else {
        let row_bytes = crate::batch_allocation::checked_count_product(
            params.scaled_width as usize,
            fmt.bytes_per_pixel(),
            "JPEG Metal packed scaled row bytes",
        )?;
        let out_len = crate::batch_allocation::checked_count_product(
            row_bytes,
            params.scaled_height as usize,
            "JPEG Metal packed scaled output bytes",
        )?;
        Some(new_shared_buffer(&runtime.device, out_len)?)
    };

    if let Some(out_buffer) = out_buffer.as_ref() {
        let pack_params = JpegFast420Params {
            width: params.scaled_width,
            height: params.scaled_height,
            chroma_width: params.chroma_width,
            chroma_height: params.chroma_height,
            mcus_per_row: params.mcus_per_row,
            mcu_rows: params.mcu_rows,
            restart_interval_mcus: params.restart_interval_mcus,
            restart_offset_count: params.restart_offset_count,
            restart_start_mcu: params.restart_start_mcu,
            entropy_len: params.entropy_len,
            out_stride: checked_u32(
                params.scaled_width as usize * fmt.bytes_per_pixel(),
                "scaled output stride",
            )?,
            alpha: u32::from(u8::MAX),
            out_format: pixel_format_to_out_format(fmt).ok_or_else(|| Error::MetalKernel {
                message: format!("unsupported JPEG Metal pixel format {fmt:?}"),
            })?,
            origin_x: 0,
            origin_y: 0,
        };
        let Some(pack_pipeline) = P::pack_pipeline_for_format(runtime, fmt) else {
            return Err(Error::MetalKernel {
                message: format!(
                    "unsupported JPEG Metal {} pixel format {fmt:?}",
                    P::FAMILY_NAME
                ),
            });
        };
        let pack_encoder = new_compute_command_encoder(command_buffer)?;
        pack_encoder.setComputePipelineState(pack_pipeline);
        pack_encoder.bind_buffer(0, Some(&y_plane), 0);
        pack_encoder.bind_buffer(1, Some(&cb_plane), 0);
        pack_encoder.bind_buffer(2, Some(&cr_plane), 0);
        pack_encoder.bind_buffer(3, Some(out_buffer), 0);
        pack_encoder.bind_bytes::<JpegFast420Params>(4, &pack_params);
        dispatch_2d_pipeline(
            &pack_encoder,
            pack_pipeline,
            (params.scaled_width, params.scaled_height),
        );
        pack_encoder.endEncoding();
    }

    let surface = match out_buffer {
        Some(out_buffer) => {
            Surface::from_metal_buffer(out_buffer, (params.scaled_width, params.scaled_height), fmt)
        }
        None => Surface::from_metal_buffer(
            y_plane.clone(),
            (params.scaled_width, params.scaled_height),
            fmt,
        ),
    }?;

    Ok(BatchedDecodeItem {
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: [
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "the scaled-region encoder binds one ordered Metal command sequence and returns every resource retained through completion"
)]
#[expect(
    clippy::similar_names,
    reason = "Cb and Cr are normative JPEG component names"
)]
pub(in crate::compute) fn encode_fast_subsampled_scaled_region_batch_item<
    P: FastSubsampledMetal,
>(
    request: FastSubsampledScaledRegionBatchItemRequest<'_, P>,
) -> Result<BatchedDecodeItem, Error> {
    let FastSubsampledScaledRegionBatchItemRequest {
        runtime,
        command_buffer,
        device_buffer_cache,
        packet,
        fmt,
        roi,
        scale,
    } = request;
    let Some(full_params) = fast_subsampled_scaled_params(packet, scale) else {
        return Err(Error::MetalKernel {
            message: format!("unsupported JPEG Metal {} scale {scale:?}", P::FAMILY_NAME),
        });
    };
    let scaled_roi = roi.scaled_covering(scale);
    let scaled_roi = j2k_jpeg::Rect {
        x: scaled_roi.x,
        y: scaled_roi.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let source_window = fast_subsampled_full_mcu_scaled_window::<P>(
        (full_params.scaled_width, full_params.scaled_height),
        scaled_roi,
        full_params.scale_shift,
    );
    let Some(mut decode_params) =
        fast_subsampled_scaled_region_params(packet, scale, source_window)
    else {
        return Err(Error::MetalKernel {
            message: format!(
                "unsupported JPEG Metal {} scaled region {scale:?}",
                P::FAMILY_NAME
            ),
        });
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
    let y_plane = new_decode_plane_buffer(&runtime.device, y_len, false)?;
    let cb_plane = new_private_buffer(&runtime.device, chroma_len)?;
    let cr_plane = new_private_buffer(&runtime.device, chroma_len)?;
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus(),
        restart_offsets.len(),
        packet.entropy_checkpoints().len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads)?;
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let (entropy_buffer, entropy_checkpoints_buffer) = device_buffer_cache.packet_buffers(
        runtime,
        packet.entropy_bytes(),
        packet.entropy_checkpoints(),
    )?;

    let (dc_tables, ac_tables) = fast_packet_huffman_tables(packet);

    let decode_pipeline = P::scaled_region_decode_pipeline(runtime);
    let decoder_encoder = new_compute_command_encoder(command_buffer)?;
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

    let out_len = crate::batch_allocation::checked_count_product(
        pack_params.out_stride as usize,
        scaled_roi.h as usize,
        "JPEG Metal packed scaled-region output bytes",
    )?;
    let out_buffer = new_shared_buffer(&runtime.device, out_len)?;
    let pack_encoder = new_compute_command_encoder(command_buffer)?;
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

    Ok(BatchedDecodeItem {
        surface: Surface::from_metal_buffer(out_buffer, (scaled_roi.w, scaled_roi.h), fmt)?,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: [
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::similar_names,
    reason = "Cb and Cr are normative JPEG component names"
)]
pub(in crate::compute) fn encode_fast_subsampled_batch_item<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    packet: &P,
    fmt: PixelFormat,
) -> Result<BatchedDecodeItem, Error> {
    let params = fast_subsampled_params(packet, fmt)?;
    let y_len = params.width as usize * params.height as usize;
    let chroma_len = params.chroma_width as usize * params.chroma_height as usize;
    let y_plane = new_decode_plane_buffer(&runtime.device, y_len, fmt == PixelFormat::Gray8)?;
    let cb_plane = new_private_buffer(&runtime.device, chroma_len)?;
    let cr_plane = new_private_buffer(&runtime.device, chroma_len)?;
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus(),
        packet.restart_offsets().len(),
        packet.entropy_checkpoints().len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads)?;
    let entropy_buffer = new_shared_buffer_with_data(&runtime.device, packet.entropy_bytes())?;
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, packet.restart_offsets())?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, packet.entropy_checkpoints())?;

    let (dc_tables, ac_tables) = fast_packet_huffman_tables(packet);

    let decode_pipeline = P::decode_pipeline(runtime);
    let decoder_encoder = new_compute_command_encoder(command_buffer)?;
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

    let surface = if fmt == PixelFormat::Gray8 {
        Surface::from_metal_buffer(y_plane.clone(), packet.dimensions(), fmt)
    } else {
        let Some(pack_pipeline) = P::pack_pipeline_for_format(runtime, fmt) else {
            return Err(Error::MetalKernel {
                message: format!(
                    "unsupported JPEG Metal {} pixel format {fmt:?}",
                    P::FAMILY_NAME
                ),
            });
        };
        let out_len = crate::batch_allocation::checked_count_product(
            params.out_stride as usize,
            params.height as usize,
            "JPEG Metal packed full output bytes",
        )?;
        let out_buffer = new_shared_buffer(&runtime.device, out_len)?;
        let pack_encoder = new_compute_command_encoder(command_buffer)?;
        pack_encoder.setComputePipelineState(pack_pipeline);
        bind_three_plane_pack::<JpegFast420Params>(
            &pack_encoder,
            [Some(&y_plane), Some(&cb_plane), Some(&cr_plane)],
            &out_buffer,
            &params,
        );
        dispatch_2d_pipeline(&pack_encoder, pack_pipeline, packet.dimensions());
        pack_encoder.endEncoding();
        Surface::from_metal_buffer(out_buffer, packet.dimensions(), fmt)
    }?;

    Ok(BatchedDecodeItem {
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: [
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

/// Route one batch request to the family's encode item for its op.
#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_fast_subsampled_op_batch_item<P: FastSubsampledMetal>(
    request: FastSubsampledOpBatchItemRequest<'_, P>,
) -> Result<BatchedDecodeItem, Error> {
    let FastSubsampledOpBatchItemRequest {
        runtime,
        command_buffer,
        device_buffer_cache,
        packet,
        fmt,
        op,
    } = request;
    match op {
        batch::BatchOp::Full => {
            encode_fast_subsampled_batch_item(runtime, command_buffer, packet, fmt)
        }
        batch::BatchOp::Region(roi) => encode_fast_subsampled_region_batch_item(
            runtime,
            command_buffer,
            packet,
            fmt,
            roi,
            None,
        ),
        batch::BatchOp::Scaled(scale) => encode_fast_subsampled_scaled_batch_item(
            runtime,
            command_buffer,
            packet,
            fmt,
            scale,
            None,
        ),
        batch::BatchOp::RegionScaled { roi, scale } => {
            encode_fast_subsampled_scaled_region_batch_item(
                FastSubsampledScaledRegionBatchItemRequest {
                    runtime,
                    command_buffer,
                    device_buffer_cache,
                    packet,
                    fmt,
                    roi,
                    scale,
                },
            )
        }
    }
}
