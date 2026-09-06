// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::metal_types::prelude::*;

#[cfg(test)]
use super::MetalBatchScratch;
use super::{
    bind_fast_decode_entropy_inputs, commit_and_wait_jpeg, dispatch_1d_pipeline,
    dispatch_rgba_texture_pack, fast_packet_huffman_tables, full_rgba_texture_status_buffer,
    new_command_buffer, new_compute_command_encoder, packed_pair_extent,
    texture_batch_error_results, texture_batch_success_results, BatchEntropyBuffers, Buffer,
    CommandBufferRef, Error, FastBatchDecodeMode, FastDecodeEntropyInputs, FastSubsampledMetal,
    FullRgbaTextureBatchCtx, FullRgbaTextureBatchShape, JpegFast420BatchParams,
    JpegTexturePackBatchParams, PreparedHuffmanHost, MODE_YCBCR,
};
#[cfg(test)]
use crate::compute::{encode_split_coeff_idct_passes, SplitCoeffIdctPasses};
#[cfg(test)]
use std::mem::size_of;

struct FullRgbaStagedDecodePass<'a, P> {
    runtime: &'a super::MetalRuntime,
    command_buffer: &'a CommandBufferRef,
    first: &'a P,
    #[cfg(test)]
    batch_scratch: &'a mut MetalBatchScratch,
    entropy_buffers: &'a BatchEntropyBuffers,
    status_buffer: &'a Buffer,
    planes: [&'a Buffer; 3],
    shape: FullRgbaTextureBatchShape,
    #[cfg(test)]
    total_blocks: Option<usize>,
    huffman_tables: (&'a [PreparedHuffmanHost; 3], &'a [PreparedHuffmanHost; 3]),
}

#[expect(
    clippy::similar_names,
    reason = "Cb and Cr are normative JPEG component names"
)]
pub(super) fn decode_fast_subsampled_full_rgba_staged_texture_batch<P: FastSubsampledMetal>(
    ctx: FullRgbaTextureBatchCtx<'_, '_, P>,
    decode_mode: FastBatchDecodeMode,
    #[cfg(test)] total_blocks: Option<usize>,
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
    let y_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::TEXTURE_KEYS.y,
        shape.y_len * shape.tile_count,
    )?;
    let cb_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::TEXTURE_KEYS.cb,
        shape.chroma_len * shape.tile_count,
    )?;
    let cr_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::TEXTURE_KEYS.cr,
        shape.chroma_len * shape.tile_count,
    )?;
    let status_buffer = full_rgba_texture_status_buffer::<P>(
        runtime,
        requests,
        &mut batch_scratch,
        shape.total_decode_threads,
    )?;
    let (dc_tables, ac_tables) = fast_packet_huffman_tables(first);
    let command_buffer = new_command_buffer(&runtime.queue)?;
    let mut decode_pass = FullRgbaStagedDecodePass {
        runtime,
        command_buffer: &command_buffer,
        first,
        #[cfg(test)]
        batch_scratch: &mut batch_scratch,
        entropy_buffers,
        status_buffer: &status_buffer,
        planes: [&y_plane, &cb_plane, &cr_plane],
        shape,
        #[cfg(test)]
        total_blocks,
        huffman_tables: (&dc_tables, &ac_tables),
    };
    encode_fast_subsampled_full_rgba_staged_decode::<P>(&mut decode_pass, decode_mode)?;

    let pack_params = JpegTexturePackBatchParams {
        width: shape.width,
        height: shape.height,
        chroma_width: shape.chroma_width,
        chroma_height: shape.chroma_height,
        tile_index: 0,
        alpha: u32::from(u8::MAX),
        mode: MODE_YCBCR,
    };
    dispatch_rgba_texture_pack(
        &command_buffer,
        P::pack_rgba_texture_pipeline(runtime),
        (&y_plane, &cb_plane, &cr_plane),
        output,
        pack_params,
        shape.tile_count,
        (
            packed_pair_extent(shape.width),
            P::packed_height_extent(shape.height),
        ),
    )?;

    commit_and_wait_jpeg(&command_buffer)?;
    // Keep scratch leased until the CPU has consumed the GPU status below.
    if let Some(results) =
        texture_batch_error_results(requests, &status_buffer, shape.total_decode_threads)?
    {
        return Ok(results);
    }
    texture_batch_success_results(requests, output, first.dimensions(), requests.len())
}

fn encode_fast_subsampled_full_rgba_staged_decode<P: FastSubsampledMetal>(
    pass: &mut FullRgbaStagedDecodePass<'_, P>,
    decode_mode: FastBatchDecodeMode,
) -> Result<(), Error> {
    match decode_mode {
        FastBatchDecodeMode::Fused => {
            let decode_pipeline = P::full_rgb_batch_decode_pipeline(pass.runtime);
            let decoder_encoder = new_compute_command_encoder(pass.command_buffer)?;
            decoder_encoder.setComputePipelineState(decode_pipeline);
            bind_fast_decode_entropy_inputs::<JpegFast420BatchParams>(
                &decoder_encoder,
                &FastDecodeEntropyInputs {
                    entropy_buffer: &pass.entropy_buffers.payload,
                    planes: pass.planes,
                    params: &pass.shape.params,
                    quants: [
                        pass.first.y_quant(),
                        pass.first.cb_quant(),
                        pass.first.cr_quant(),
                    ],
                    dc_tables: pass.huffman_tables.0,
                    ac_tables: pass.huffman_tables.1,
                    slot14_buffer: &pass.entropy_buffers.offsets,
                    slot15_buffer: &pass.entropy_buffers.lens,
                    slot16_buffer: pass.status_buffer,
                },
            );
            decoder_encoder.bind_buffer(17, Some(&pass.entropy_buffers.checkpoints), 0);
            dispatch_1d_pipeline(
                &decoder_encoder,
                decode_pipeline,
                pass.shape.total_decode_threads,
            );
            decoder_encoder.endEncoding();
        }
        #[cfg(test)]
        FastBatchDecodeMode::SplitCoeffIdct => {
            encode_fast_subsampled_full_rgba_split_decode::<P>(pass)?;
        }
    }
    Ok(())
}

#[cfg(test)]
fn encode_fast_subsampled_full_rgba_split_decode<P: FastSubsampledMetal>(
    pass: &mut FullRgbaStagedDecodePass<'_, P>,
) -> Result<(), Error> {
    let Some((split, total_blocks)) =
        P::split_coeff_idct_pipelines(pass.runtime).zip(pass.total_blocks)
    else {
        return Err(Error::MetalKernel {
            message: format!(
                "JPEG Metal {} texture batch split coeff/IDCT decode mode is unsupported",
                P::FAMILY_NAME
            ),
        });
    };
    let coeff_bytes = total_blocks
        .checked_mul(64)
        .and_then(|bytes| bytes.checked_mul(size_of::<i16>()))
        .ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal {} texture batch coefficient scratch overflowed",
                P::FAMILY_NAME
            ),
        })?;
    let idct_component_depth =
        pass.shape
            .tile_count_u32
            .checked_mul(6)
            .ok_or_else(|| Error::MetalKernel {
                message: format!(
                    "JPEG Metal {} texture batch IDCT dispatch overflowed",
                    P::FAMILY_NAME
                ),
            })?;
    let coeff_blocks = pass.batch_scratch.private_buffer(
        &pass.runtime.device,
        P::SPLIT_TEXTURE_SCRATCH_KEYS.0,
        coeff_bytes,
    )?;
    let dc_only_flags = pass.batch_scratch.private_buffer(
        &pass.runtime.device,
        P::SPLIT_TEXTURE_SCRATCH_KEYS.1,
        total_blocks,
    )?;

    encode_split_coeff_idct_passes(SplitCoeffIdctPasses {
        command_buffer: pass.command_buffer,
        pipelines: split,
        params: &pass.shape.params,
        quants: [
            pass.first.y_quant(),
            pass.first.cb_quant(),
            pass.first.cr_quant(),
        ],
        dc_tables: pass.huffman_tables.0,
        ac_tables: pass.huffman_tables.1,
        entropy: (
            &pass.entropy_buffers.payload,
            &pass.entropy_buffers.offsets,
            &pass.entropy_buffers.lens,
            &pass.entropy_buffers.checkpoints,
        ),
        status_buffer: pass.status_buffer,
        planes: pass.planes,
        scratch: (&coeff_blocks, &dc_only_flags),
        total_decode_threads: pass.shape.total_decode_threads,
        idct_grid: (
            pass.first.mcus_per_row(),
            pass.first.mcu_rows(),
            idct_component_depth,
        ),
    })?;
    Ok(())
}
