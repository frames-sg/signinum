// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::metal_types::prelude::*;

use super::super::scratch_pool::BatchScratchLease;

use super::super::{
    batch, batch_entropy_buffers, bind_fast_decode_entropy_inputs, checked_u32,
    commit_and_wait_jpeg, dispatch_1d_pipeline, dispatch_rgba_texture_pack,
    fast_packet_huffman_tables, fast_subsampled_full_rgb_batch_groups, new_command_buffer,
    new_compute_command_encoder, packed_pair_extent, plane_mode_to_u32,
    texture_batch_error_results, texture_batch_success_results, validate_rgba_texture_batch_output,
    BatchEntropyBufferKeys, BatchEntropyBufferPlan, BatchEntropyBuffers, BatchedFastPacket, Buffer,
    CommandBufferRef, Error, FastBatchDecodeMode, FastDecodeEntropyInputs, FastSubsampledMetal,
    FastTextureRepairCtx, JpegDecodeStatus, JpegFast420BatchParams, JpegFast420TextureBatchParams,
    JpegFast444TextureBatchParams, JpegTexturePackBatchParams, MetalBatchScratch, MetalRuntime,
    PixelFormat, PlaneMode, PreparedHuffmanHost, MODE_YCBCR,
};
#[cfg(target_os = "macos")]
mod staged;
#[cfg(target_os = "macos")]
use self::staged::decode_fast_subsampled_full_rgba_staged_texture_batch;
use super::texture_grouped::try_decode_grouped_fast_subsampled_full_rgba_batch_to_textures;

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "ordered Metal texture command and resource lifetime"
)]
pub(in crate::compute) fn try_decode_fast_subsampled_full_rgba_batch_to_textures<
    P: FastSubsampledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
    decode_mode: FastBatchDecodeMode,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    if requests.is_empty()
        || requests
            .iter()
            .any(|request| request.op != batch::BatchOp::Full || request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }
    let mut packet_budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal full texture family packet plan",
        requests,
    )?;
    packet_budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<&P>(packets.len()),
        crate::batch_allocation::BatchMetadataRequest::of::<PlaneMode>(packets.len()),
    ])?;
    let mut family_packets = packet_budget.try_vec(
        packets.len(),
        "JPEG Metal full texture family packet references",
    )?;
    let mut family_modes =
        packet_budget.try_vec(packets.len(), "JPEG Metal full texture family modes")?;
    let mut family_mode = None;
    for packet in packets {
        let Some(packet_mode) = P::texture_plane_mode_from_batched(packet) else {
            return Ok(None);
        };
        if let Some(previous_mode) = family_mode.replace(packet_mode) {
            if previous_mode != packet_mode {
                return Ok(None);
            }
        }
        let Some(packet) = P::from_batched(packet) else {
            return Ok(None);
        };
        family_packets.push(packet);
        family_modes.push(packet_mode);
    }
    let Some(first) = family_packets.first().copied() else {
        return Ok(None);
    };
    if (!P::FULL_RGB_BATCH_SUPPORTS_RESTART && first.restart_interval_mcus() != 0)
        || first.entropy_checkpoints().is_empty()
    {
        return Ok(None);
    }

    let Some(groups) = fast_subsampled_full_rgb_batch_groups(requests, &family_packets)? else {
        return Ok(None);
    };
    if groups.len() > 1 {
        return try_decode_grouped_fast_subsampled_full_rgba_batch_to_textures::<P>(
            runtime,
            requests,
            &family_packets,
            &family_modes,
            output,
            decode_mode,
            groups,
        );
    }

    let segment_count = first.entropy_checkpoints().len();
    let tile_count = family_packets.len();
    let shape = full_rgba_texture_batch_shape::<P>(
        first,
        tile_count,
        segment_count,
        family_mode.unwrap_or(PlaneMode::YCbCr),
    )?;
    validate_rgba_texture_batch_output(output, first.dimensions(), tile_count, shape.out_tile_len)?;

    #[cfg(test)]
    let total_blocks = full_rgba_texture_total_blocks::<P>(shape.total_mcus, shape.tile_count)?;

    let mut batch_scratch = runtime.batch_scratch()?;
    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        requests,
        &mut batch_scratch,
        BatchEntropyBufferPlan {
            keys: BatchEntropyBufferKeys {
                payload: P::TEXTURE_KEYS.entropy,
                offsets: P::TEXTURE_KEYS.entropy_offsets,
                lens: P::TEXTURE_KEYS.entropy_lens,
                checkpoints: P::TEXTURE_KEYS.entropy_checkpoints,
            },
            tile_count,
            segment_count,
        },
        family_packets.iter().map(|packet| packet.entropy_bytes()),
        family_packets
            .iter()
            .map(|packet| packet.entropy_checkpoints()),
    )?
    else {
        return Ok(None);
    };

    // Subsampled batches distribute entropy/IDCT across all tiles, then pack
    // private component planes. This avoids the direct shader's per-tile
    // decode and chroma-repair passes. The 4:4:4 path has no chroma repair and
    // uses a different plane-decode ABI, so it retains its direct kernel.
    let direct_texture = P::USE_FAST444_TEXTURE_PARAMS;
    #[cfg(test)]
    let direct_texture = super::super::texture_tuning::component_planes()
        .map_or(direct_texture, |planes| {
            !planes || P::USE_FAST444_TEXTURE_PARAMS
        });
    if decode_mode == FastBatchDecodeMode::Fused && direct_texture {
        return Ok(Some(
            decode_fast_subsampled_full_rgba_fused_texture_batch::<P>(FullRgbaTextureBatchCtx {
                runtime,
                requests,
                first,
                output,
                batch_scratch,
                entropy_buffers: &entropy_buffers,
                shape,
            })?,
        ));
    }

    Ok(Some(
        decode_fast_subsampled_full_rgba_staged_texture_batch::<P>(
            FullRgbaTextureBatchCtx {
                runtime,
                requests,
                first,
                output,
                batch_scratch,
                entropy_buffers: &entropy_buffers,
                shape,
            },
            decode_mode,
            #[cfg(test)]
            total_blocks,
        )?,
    ))
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct FullRgbaTextureBatchShape {
    width: u32,
    height: u32,
    chroma_width: u32,
    chroma_height: u32,
    y_len: usize,
    chroma_len: usize,
    out_tile_len: usize,
    total_mcus: usize,
    mcu_threads: Option<u32>,
    tile_count: usize,
    #[cfg(test)]
    tile_count_u32: u32,
    segment_count_u32: u32,
    total_decode_threads: u32,
    params: JpegFast420BatchParams,
    mode: PlaneMode,
}

#[cfg(target_os = "macos")]
struct FullRgbaTextureBatchCtx<'a, 'scratch, P> {
    runtime: &'a MetalRuntime,
    requests: &'a [batch::QueuedRequest],
    first: &'a P,
    output: &'a crate::MetalBatchTextureOutput,
    batch_scratch: BatchScratchLease<'scratch>,
    entropy_buffers: &'a BatchEntropyBuffers,
    shape: FullRgbaTextureBatchShape,
}

#[cfg(target_os = "macos")]
struct FullRgbaTextureRepairBuffers {
    status: Buffer,
    boundary_meta: Buffer,
    boundary_samples: Buffer,
    vertical: Option<(Buffer, Buffer)>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct FullRgbaTextureDecodeTiles<'a, P> {
    runtime: &'a MetalRuntime,
    command_buffer: &'a CommandBufferRef,
    output: &'a crate::MetalBatchTextureOutput,
    first: &'a P,
    entropy_buffers: &'a BatchEntropyBuffers,
    status_buffer: &'a Buffer,
    boundary_buffers: (&'a Buffer, &'a Buffer),
    vertical_buffers: Option<&'a (Buffer, Buffer)>,
    shape: FullRgbaTextureBatchShape,
    tile_index_ctx: &'a str,
}

#[cfg(target_os = "macos")]
fn full_rgba_texture_status_buffer<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    batch_scratch: &mut MetalBatchScratch,
    total_decode_threads: u32,
) -> Result<Buffer, Error> {
    let mut budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal full texture batch statuses",
        requests,
    )?;
    let statuses = budget.try_filled(
        total_decode_threads as usize,
        JpegDecodeStatus::default(),
        "JPEG Metal full texture decode statuses",
    )?;
    batch_scratch.shared_buffer_with_slice(&runtime.device, P::TEXTURE_KEYS.status, &statuses)
}

#[cfg(target_os = "macos")]
fn full_rgba_texture_repair_buffers<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    batch_scratch: &mut MetalBatchScratch,
    shape: FullRgbaTextureBatchShape,
) -> Result<FullRgbaTextureRepairBuffers, Error> {
    let mut budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal full texture batch statuses",
        requests,
    )?;
    let statuses = budget.try_filled(
        shape.total_decode_threads as usize,
        JpegDecodeStatus::default(),
        "JPEG Metal full texture decode statuses",
    )?;
    let total_records = P::texture_repair_record_count(
        shape.tile_count,
        shape.total_mcus,
        shape.total_decode_threads,
    )?;
    let metadata_count = total_records
        .checked_mul(P::TEXTURE_BOUNDARY_META_WORDS)
        .ok_or(j2k_core::BatchInfrastructureError::AllocationTooLarge {
            what: "JPEG Metal texture boundary metadata",
            requested: usize::MAX,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })?;
    let sample_count = total_records
        .checked_mul(P::TEXTURE_BOUNDARY_SAMPLE_BYTES)
        .ok_or(j2k_core::BatchInfrastructureError::AllocationTooLarge {
            what: "JPEG Metal texture boundary samples",
            requested: usize::MAX,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })?;
    let metadata =
        budget.try_filled(metadata_count, 0u32, "JPEG Metal texture boundary metadata")?;
    let samples = budget.try_filled(sample_count, 0u8, "JPEG Metal texture boundary samples")?;
    Ok(FullRgbaTextureRepairBuffers {
        status: batch_scratch.shared_buffer_with_slice(
            &runtime.device,
            P::TEXTURE_KEYS.status,
            &statuses,
        )?,
        boundary_meta: batch_scratch.shared_buffer_with_slice(
            &runtime.device,
            P::TEXTURE_BOUNDARY_META_KEY,
            &metadata,
        )?,
        boundary_samples: batch_scratch.shared_buffer_with_bytes(
            &runtime.device,
            P::TEXTURE_BOUNDARY_SAMPLES_KEY,
            &samples,
        )?,
        vertical: fast_subsampled_full_texture_vertical_buffers::<P>(
            runtime,
            batch_scratch,
            total_records,
            &mut budget,
        )?,
    })
}

#[cfg(target_os = "macos")]
fn full_rgba_texture_batch_shape<P: FastSubsampledMetal>(
    first: &P,
    tile_count: usize,
    segment_count: usize,
    mode: PlaneMode,
) -> Result<FullRgbaTextureBatchShape, Error> {
    let tile_count_u32 = checked_u32(
        tile_count,
        &format!("{} texture batch tile count", P::FAMILY_NAME),
    )?;
    let segment_count_u32 = checked_u32(
        segment_count,
        &format!("{} texture batch segment count", P::FAMILY_NAME),
    )?;
    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: format!(
                    "JPEG Metal {} texture batch decode thread count overflowed",
                    P::FAMILY_NAME
                ),
            })?,
        &format!("{} texture batch decode thread count", P::FAMILY_NAME),
    )?;
    let width = first.dimensions().0;
    let height = first.dimensions().1;
    let chroma_width = width.div_ceil(2);
    let chroma_height = P::chroma_height(height);
    let y_len = width as usize * height as usize;
    let chroma_len = chroma_width as usize * chroma_height as usize;
    let out_stride = width as usize * PixelFormat::Rgba8.bytes_per_pixel();
    let out_tile_len = out_stride * height as usize;
    let total_mcus = first.mcus_per_row() as usize * first.mcu_rows() as usize;
    let mcu_threads = P::texture_mcu_dispatch_threads(total_mcus)?;
    Ok(FullRgbaTextureBatchShape {
        width,
        height,
        chroma_width,
        chroma_height,
        y_len,
        chroma_len,
        out_tile_len,
        total_mcus,
        mcu_threads,
        tile_count,
        #[cfg(test)]
        tile_count_u32,
        segment_count_u32,
        total_decode_threads,
        params: JpegFast420BatchParams {
            width,
            height,
            chroma_width,
            chroma_height,
            mcus_per_row: first.mcus_per_row(),
            mcu_rows: first.mcu_rows(),
            segment_count: segment_count_u32,
            tile_count: tile_count_u32,
            out_stride: checked_u32(
                out_stride,
                &format!("{} texture batch output stride", P::FAMILY_NAME),
            )?,
            alpha: u32::from(u8::MAX),
        },
        mode,
    })
}

#[cfg(all(target_os = "macos", test))]
fn full_rgba_texture_total_blocks<P: FastSubsampledMetal>(
    total_mcus: usize,
    tile_count: usize,
) -> Result<Option<usize>, Error> {
    let Some(blocks_per_mcu) = P::FULL_RGB_BATCH_BLOCKS_PER_MCU else {
        return Ok(None);
    };
    let blocks_per_tile =
        total_mcus
            .checked_mul(blocks_per_mcu)
            .ok_or_else(|| Error::MetalKernel {
                message: format!(
                    "JPEG Metal {} texture batch block count overflowed",
                    P::FAMILY_NAME
                ),
            })?;
    blocks_per_tile
        .checked_mul(tile_count)
        .map(Some)
        .ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal {} texture batch total block count overflowed",
                P::FAMILY_NAME
            ),
        })
}

#[cfg(target_os = "macos")]
fn full_rgba_texture_params_for_tile<P: FastSubsampledMetal>(
    first: &P,
    shape: FullRgbaTextureBatchShape,
    tile_index: usize,
    tile_index_ctx: &str,
) -> Result<JpegFast420TextureBatchParams, Error> {
    Ok(JpegFast420TextureBatchParams {
        width: shape.width,
        height: shape.height,
        chroma_width: shape.chroma_width,
        chroma_height: shape.chroma_height,
        mcus_per_row: first.mcus_per_row(),
        mcu_rows: first.mcu_rows(),
        segment_count: shape.segment_count_u32,
        tile_index: checked_u32(tile_index, tile_index_ctx)?,
        alpha: u32::from(u8::MAX),
    })
}

#[cfg(target_os = "macos")]
fn bind_full_rgba_texture_params_for_tile<P: FastSubsampledMetal>(
    encoder: &crate::metal_types::ComputeCommandEncoderRef,
    buffer_index: u64,
    first: &P,
    shape: FullRgbaTextureBatchShape,
    tile_index: usize,
    tile_index_ctx: &str,
) -> Result<(), Error> {
    if P::USE_FAST444_TEXTURE_PARAMS {
        let decode_params = JpegFast444TextureBatchParams {
            width: shape.width,
            height: shape.height,
            mcus_per_row: first.mcus_per_row(),
            mcu_rows: first.mcu_rows(),
            segment_count: shape.segment_count_u32,
            tile_index: checked_u32(tile_index, tile_index_ctx)?,
            alpha: u32::from(u8::MAX),
            mode: plane_mode_to_u32(shape.mode),
        };
        encoder.bind_bytes::<JpegFast444TextureBatchParams>(buffer_index, &decode_params);
    } else {
        let decode_params =
            full_rgba_texture_params_for_tile::<P>(first, shape, tile_index, tile_index_ctx)?;
        encoder.bind_bytes::<JpegFast420TextureBatchParams>(buffer_index, &decode_params);
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn fast_subsampled_full_texture_vertical_buffers<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    batch_scratch: &mut MetalBatchScratch,
    total_repair_records: usize,
    budget: &mut crate::batch_allocation::BatchMetadataBudget,
) -> Result<Option<(Buffer, Buffer)>, Error> {
    let Some(spec) = P::TEXTURE_VERTICAL_REPAIR else {
        return Ok(None);
    };
    let meta_count = total_repair_records.checked_mul(spec.meta_words).ok_or(
        j2k_core::BatchInfrastructureError::AllocationTooLarge {
            what: "JPEG Metal vertical texture repair metadata",
            requested: usize::MAX,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        },
    )?;
    let sample_count = total_repair_records.checked_mul(spec.sample_bytes).ok_or(
        j2k_core::BatchInfrastructureError::AllocationTooLarge {
            what: "JPEG Metal vertical texture repair samples",
            requested: usize::MAX,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        },
    )?;
    budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<u32>(meta_count),
        crate::batch_allocation::BatchMetadataRequest::of::<u8>(sample_count),
    ])?;
    let vertical_meta = budget.try_filled(
        meta_count,
        0u32,
        "JPEG Metal vertical texture repair metadata",
    )?;
    let vertical_samples = budget.try_filled(
        sample_count,
        0u8,
        "JPEG Metal vertical texture repair samples",
    )?;
    Ok(Some((
        batch_scratch.shared_buffer_with_slice(&runtime.device, spec.meta_key, &vertical_meta)?,
        batch_scratch.shared_buffer_with_bytes(
            &runtime.device,
            spec.samples_key,
            &vertical_samples,
        )?,
    )))
}

#[cfg(target_os = "macos")]
fn encode_fast_subsampled_full_rgba_texture_decode_tiles<P: FastSubsampledMetal>(
    pass: &FullRgbaTextureDecodeTiles<'_, P>,
) -> Result<(), Error> {
    let (dc_tables, ac_tables) = fast_packet_huffman_tables(pass.first);
    let texture_decode_pipeline = P::rgba_texture_batch_decode_pipeline(pass.runtime);
    for index in 0..pass.shape.tile_count {
        let texture = pass
            .output
            .texture_trusted(index)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal batch texture output slot was missing".to_string(),
            })?;
        let decoder_encoder = new_compute_command_encoder(pass.command_buffer)?;
        decoder_encoder.setComputePipelineState(texture_decode_pipeline);
        decoder_encoder.bind_buffer(0, Some(&pass.entropy_buffers.payload), 0);
        bind_full_rgba_texture_params_for_tile::<P>(
            &decoder_encoder,
            4,
            pass.first,
            pass.shape,
            index,
            pass.tile_index_ctx,
        )?;
        decoder_encoder.bind_bytes::<[u16; 64]>(5, pass.first.y_quant());
        decoder_encoder.bind_bytes::<[u16; 64]>(6, pass.first.cb_quant());
        decoder_encoder.bind_bytes::<[u16; 64]>(7, pass.first.cr_quant());
        decoder_encoder.bind_bytes::<PreparedHuffmanHost>(8, &dc_tables[0]);
        decoder_encoder.bind_bytes::<PreparedHuffmanHost>(9, &ac_tables[0]);
        decoder_encoder.bind_bytes::<PreparedHuffmanHost>(10, &dc_tables[1]);
        decoder_encoder.bind_bytes::<PreparedHuffmanHost>(11, &ac_tables[1]);
        decoder_encoder.bind_bytes::<PreparedHuffmanHost>(12, &dc_tables[2]);
        decoder_encoder.bind_bytes::<PreparedHuffmanHost>(13, &ac_tables[2]);
        decoder_encoder.bind_buffer(14, Some(&pass.entropy_buffers.offsets), 0);
        decoder_encoder.bind_buffer(15, Some(&pass.entropy_buffers.lens), 0);
        decoder_encoder.bind_buffer(16, Some(pass.status_buffer), 0);
        decoder_encoder.bind_buffer(17, Some(&pass.entropy_buffers.checkpoints), 0);
        decoder_encoder.bind_buffer(18, Some(pass.boundary_buffers.0), 0);
        decoder_encoder.bind_buffer(19, Some(pass.boundary_buffers.1), 0);
        if let Some((vertical_meta_buffer, vertical_samples_buffer)) = pass.vertical_buffers {
            decoder_encoder.bind_buffer(20, Some(vertical_meta_buffer), 0);
            decoder_encoder.bind_buffer(21, Some(vertical_samples_buffer), 0);
        }
        decoder_encoder.bind_texture(0, Some(texture));
        dispatch_1d_pipeline(
            &decoder_encoder,
            texture_decode_pipeline,
            pass.shape.segment_count_u32,
        );
        decoder_encoder.endEncoding();
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn encode_fast_subsampled_full_rgba_texture_boundary_passes<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    output: &crate::MetalBatchTextureOutput,
    first: &P,
    boundary_buffers: (&Buffer, &Buffer),
    shape: FullRgbaTextureBatchShape,
    tile_index_ctx: &str,
) -> Result<(), Error> {
    let Some(repair_threads) =
        P::horizontal_repair_threads(first, shape.segment_count_u32, shape.mcu_threads)
    else {
        return Ok(());
    };
    let boundary_pipeline = P::rgba_texture_boundary_pipeline(runtime);
    for index in 0..shape.tile_count {
        let texture = output
            .texture_trusted(index)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal batch texture output slot was missing".to_string(),
            })?;
        let decode_params =
            full_rgba_texture_params_for_tile::<P>(first, shape, index, tile_index_ctx)?;
        let boundary_encoder = new_compute_command_encoder(command_buffer)?;
        boundary_encoder.setComputePipelineState(boundary_pipeline);
        boundary_encoder.bind_buffer(0, Some(boundary_buffers.0), 0);
        boundary_encoder.bind_buffer(1, Some(boundary_buffers.1), 0);
        boundary_encoder.bind_bytes::<JpegFast420TextureBatchParams>(2, &decode_params);
        boundary_encoder.bind_texture(0, Some(texture));
        dispatch_1d_pipeline(&boundary_encoder, boundary_pipeline, repair_threads);
        boundary_encoder.endEncoding();
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn decode_fast_subsampled_full_rgba_fused_texture_batch<P: FastSubsampledMetal>(
    ctx: FullRgbaTextureBatchCtx<'_, '_, P>,
) -> Result<Vec<Result<crate::MetalTextureTile, Error>>, Error> {
    let FullRgbaTextureBatchCtx {
        runtime,
        requests,
        first,
        output,
        mut batch_scratch,
        entropy_buffers,
        shape,
    } = ctx;
    // Chroma reconstruction needs neighboring samples at MCU boundaries. The
    // fused path carries same-segment boundaries in-thread, then resolves
    // cross-segment boundaries from compact shared records.
    let repair =
        full_rgba_texture_repair_buffers::<P>(runtime, requests, &mut batch_scratch, shape)?;
    let tile_index_ctx = format!("{} texture batch tile index", P::FAMILY_NAME);
    let command_buffer = new_command_buffer(&runtime.queue)?;
    let decode_tiles = FullRgbaTextureDecodeTiles {
        runtime,
        command_buffer: &command_buffer,
        output,
        first,
        entropy_buffers,
        status_buffer: &repair.status,
        boundary_buffers: (&repair.boundary_meta, &repair.boundary_samples),
        vertical_buffers: repair.vertical.as_ref(),
        shape,
        tile_index_ctx: &tile_index_ctx,
    };
    encode_fast_subsampled_full_rgba_texture_decode_tiles::<P>(&decode_tiles)?;
    encode_fast_subsampled_full_rgba_texture_boundary_passes::<P>(
        runtime,
        &command_buffer,
        output,
        first,
        (&repair.boundary_meta, &repair.boundary_samples),
        shape,
        &tile_index_ctx,
    )?;
    P::encode_extra_texture_repair_passes(
        runtime,
        &FastTextureRepairCtx {
            command_buffer: &command_buffer,
            output,
            boundary_meta_buffer: &repair.boundary_meta,
            vertical_buffers: repair.vertical.as_ref(),
            decode_params: JpegFast420TextureBatchParams {
                width: shape.width,
                height: shape.height,
                chroma_width: shape.chroma_width,
                chroma_height: shape.chroma_height,
                mcus_per_row: first.mcus_per_row(),
                mcu_rows: first.mcu_rows(),
                segment_count: shape.segment_count_u32,
                tile_index: 0,
                alpha: u32::from(u8::MAX),
            },
            tile_count: shape.tile_count,
            mcu_threads: shape.mcu_threads,
            tile_index_ctx: &tile_index_ctx,
        },
    )?;

    commit_and_wait_jpeg(&command_buffer)?;
    // Keep scratch leased until the CPU has consumed the GPU status below.
    if let Some(results) =
        texture_batch_error_results(requests, &repair.status, shape.total_decode_threads)?
    {
        return Ok(results);
    }
    texture_batch_success_results(requests, output, first.dimensions(), requests.len())
}
