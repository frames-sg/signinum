// SPDX-License-Identifier: Apache-2.0

use core::convert::Infallible;
use std::num::NonZeroUsize;

pub use signinum_core::TileBatchOptions;
use signinum_core::{
    collect_indexed_batch_results, tile_batch_worker_count, CompressedTransferSyntax,
    DecodeOutcome, DecoderContext, Downscale, IndexedBatchResult, PixelFormat, Rect,
    TileBatchDecode,
};
use signinum_j2k_native::{
    execute_direct_color_plan_rgb8_into, execute_direct_color_plan_rgba8_into,
    DecodeError as NativeDecodeError, DecodingError as NativeDecodingError, J2kDirectColorPlan,
    J2kDirectCpuScratch, J2kRect,
};

use crate::backend::{self, DecodeSettings};
use crate::decode::{validate_buffer, validate_region};
use crate::parse::parse_image_info;
use crate::{CpuDecodeParallelism, J2kCodec, J2kContext, J2kError, J2kScratchPool};

/// One full-tile decode request for [`decode_tiles_into`].
pub struct TileDecodeJob<'i, 'o> {
    /// Compressed J2K/HTJ2K tile bytes.
    pub input: &'i [u8],
    /// Caller-owned output buffer for this tile.
    pub out: &'o mut [u8],
    /// Distance in bytes between output rows.
    pub stride: usize,
}

/// One ROI+scaled tile decode request for [`decode_tiles_region_scaled_into`].
pub struct TileRegionScaledDecodeJob<'i, 'o> {
    /// Compressed J2K/HTJ2K tile bytes.
    pub input: &'i [u8],
    /// Caller-owned output buffer for this tile.
    pub out: &'o mut [u8],
    /// Distance in bytes between output rows.
    pub stride: usize,
    /// Region of interest in source-image coordinates.
    pub roi: Rect,
    /// Downscale factor applied to the region decode.
    pub scale: Downscale,
}

/// Error returned by J2K CPU tile batches, annotated with the first failing
/// tile index from the caller's input order.
#[derive(Debug)]
pub struct TileBatchError {
    /// Index of the first failing tile in input order.
    pub index: usize,
    /// Decode error reported for that tile.
    pub source: J2kError,
}

impl core::fmt::Display for TileBatchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "tile {} decode failed: {}", self.index, self.source)
    }
}

impl std::error::Error for TileBatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

type BatchOutcome = DecodeOutcome<Infallible>;
type J2kIndexedBatchResult = IndexedBatchResult<BatchOutcome, J2kError>;

/// One-shot parse-plus-decode of an independent J2K/HTJ2K tile into the
/// caller's buffer, reusing both caller-owned [`DecoderContext`] and
/// caller-owned [`J2kScratchPool`].
pub fn decode_tile_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext<J2kContext>,
    pool: &mut J2kScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<BatchOutcome, J2kError> {
    <J2kCodec as TileBatchDecode>::decode_tile(ctx, pool, bytes, out, stride, fmt)
}

/// One-shot parse-plus-ROI-scaled-decode of an independent J2K/HTJ2K tile
/// into the caller's buffer, reusing both caller-owned [`DecoderContext`] and
/// caller-owned [`J2kScratchPool`].
#[allow(clippy::too_many_arguments)]
pub fn decode_tile_region_scaled_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext<J2kContext>,
    pool: &mut J2kScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
) -> Result<BatchOutcome, J2kError> {
    <J2kCodec as TileBatchDecode>::decode_tile_region_scaled(
        ctx, pool, bytes, out, stride, fmt, roi, scale,
    )
}

/// Decode independent J2K/HTJ2K tiles into caller-owned output buffers using
/// a scoped CPU worker pool.
///
/// Each worker owns one [`DecoderContext`] and one [`J2kScratchPool`]. Returned
/// outcomes preserve caller input order.
pub fn decode_tiles_into(
    jobs: &mut [TileDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<BatchOutcome>, TileBatchError> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }

    let job_count = jobs.len();
    let worker_count = tile_batch_worker_count(job_count, options, available_tile_batch_workers());
    let chunk_size = job_count.div_ceil(worker_count);
    let results =
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(worker_count);
            for (chunk_index, chunk) in jobs.chunks_mut(chunk_size).enumerate() {
                let start_index = chunk_index * chunk_size;
                let inner_parallelism = inner_parallelism_for_batch(job_count);
                handles.push(scope.spawn(move || {
                    decode_tile_job_chunk(start_index, chunk, fmt, inner_parallelism)
                }));
            }

            let mut results = Vec::with_capacity(job_count);
            for handle in handles {
                match handle.join() {
                    Ok(chunk_results) => results.extend(chunk_results),
                    Err(payload) => std::panic::resume_unwind(payload),
                }
            }
            results
        });

    collect_indexed_batch_results(job_count, results, |index, source| TileBatchError {
        index,
        source,
    })
}

/// Decode independent J2K/HTJ2K tile regions at reduced resolution into
/// caller-owned output buffers using a scoped CPU worker pool.
pub fn decode_tiles_region_scaled_into(
    jobs: &mut [TileRegionScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<BatchOutcome>, TileBatchError> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }

    let job_count = jobs.len();
    let worker_count = tile_batch_worker_count(job_count, options, available_tile_batch_workers());
    let chunk_size = job_count.div_ceil(worker_count);
    let shared_direct_plan = build_repeated_direct_color_region_plan(jobs, fmt)
        .map_err(|source| TileBatchError { index: 0, source })?;
    let results = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for (chunk_index, chunk) in jobs.chunks_mut(chunk_size).enumerate() {
            let start_index = chunk_index * chunk_size;
            let shared_direct_plan = shared_direct_plan.as_ref();
            handles.push(scope.spawn(move || {
                decode_tile_region_scaled_job_chunk(
                    start_index,
                    chunk,
                    fmt,
                    inner_parallelism_for_batch(job_count),
                    shared_direct_plan,
                )
            }));
        }

        let mut results = Vec::with_capacity(job_count);
        for handle in handles {
            match handle.join() {
                Ok(chunk_results) => results.extend(chunk_results),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        results
    });

    collect_indexed_batch_results(job_count, results, |index, source| TileBatchError {
        index,
        source,
    })
}

fn available_tile_batch_workers() -> usize {
    std::thread::available_parallelism().map_or(1, NonZeroUsize::get)
}

fn inner_parallelism_for_batch(batch_size: usize) -> CpuDecodeParallelism {
    if batch_size > 1 {
        CpuDecodeParallelism::Serial
    } else {
        CpuDecodeParallelism::Auto
    }
}

fn decode_tile_job_chunk(
    start_index: usize,
    jobs: &mut [TileDecodeJob<'_, '_>],
    fmt: PixelFormat,
    inner_parallelism: CpuDecodeParallelism,
) -> Vec<J2kIndexedBatchResult> {
    let mut ctx = DecoderContext::<J2kContext>::new();
    ctx.codec_mut()
        .set_cpu_decode_parallelism(inner_parallelism);
    let mut pool = J2kScratchPool::new();
    let mut results = Vec::with_capacity(jobs.len());
    for (local_index, job) in jobs.iter_mut().enumerate() {
        let outcome =
            decode_tile_into_in_context(job.input, &mut ctx, &mut pool, job.out, job.stride, fmt);
        results.push((start_index + local_index, outcome));
    }
    results
}

fn decode_tile_region_scaled_job_chunk(
    start_index: usize,
    jobs: &mut [TileRegionScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    inner_parallelism: CpuDecodeParallelism,
    shared_direct_plan: Option<&DirectColorRegionCache>,
) -> Vec<J2kIndexedBatchResult> {
    let mut ctx = DecoderContext::<J2kContext>::new();
    ctx.codec_mut()
        .set_cpu_decode_parallelism(inner_parallelism);
    let mut pool = J2kScratchPool::new();
    let mut direct_scratch = J2kDirectCpuScratch::new();
    let mut direct_cache = None;
    let mut results = Vec::with_capacity(jobs.len());
    for (local_index, job) in jobs.iter_mut().enumerate() {
        let outcome = match decode_tile_region_scaled_shared_direct_color_u8_in_context(
            job,
            &mut ctx,
            fmt,
            &mut direct_scratch,
            shared_direct_plan,
        )
        .and_then(|outcome| {
            if outcome.is_some() {
                Ok(outcome)
            } else {
                decode_tile_region_scaled_direct_color_u8_in_context(
                    job,
                    &mut ctx,
                    fmt,
                    &mut direct_scratch,
                    &mut direct_cache,
                )
            }
        }) {
            Ok(Some(outcome)) => Ok(outcome),
            Ok(None) => decode_tile_region_scaled_into_in_context(
                job.input, &mut ctx, &mut pool, job.out, job.stride, fmt, job.roi, job.scale,
            ),
            Err(error) => Err(error),
        };
        results.push((start_index + local_index, outcome));
    }
    results
}

struct DirectColorRegionCache {
    input_ptr: usize,
    input_len: usize,
    roi: Rect,
    scale: Downscale,
    output_region: J2kRect,
    plan: J2kDirectColorPlan,
}

fn build_repeated_direct_color_region_plan(
    jobs: &[TileRegionScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
) -> Result<Option<DirectColorRegionCache>, J2kError> {
    if !is_direct_color_u8_format(fmt) {
        return Ok(None);
    }
    let Some(first) = jobs.first() else {
        return Ok(None);
    };
    if first.scale == Downscale::None {
        return Ok(None);
    }
    let key = DirectColorRegionKey {
        input_ptr: first.input.as_ptr() as usize,
        input_len: first.input.len(),
        roi: first.roi,
        scale: first.scale,
    };
    if !jobs.iter().all(|job| {
        job.input.as_ptr() as usize == key.input_ptr
            && job.input.len() == key.input_len
            && job.roi == key.roi
            && job.scale == key.scale
    }) {
        return Ok(None);
    }

    let Some((plan, output_region)) =
        build_direct_color_region_plan(first.input, first.roi, first.scale)?
    else {
        return Ok(None);
    };
    Ok(Some(DirectColorRegionCache {
        input_ptr: key.input_ptr,
        input_len: key.input_len,
        roi: key.roi,
        scale: key.scale,
        output_region,
        plan,
    }))
}

fn decode_tile_region_scaled_shared_direct_color_u8_in_context(
    job: &mut TileRegionScaledDecodeJob<'_, '_>,
    ctx: &mut DecoderContext<J2kContext>,
    fmt: PixelFormat,
    scratch: &mut J2kDirectCpuScratch,
    shared_direct_plan: Option<&DirectColorRegionCache>,
) -> Result<Option<BatchOutcome>, J2kError> {
    let Some(shared_direct_plan) = shared_direct_plan else {
        return Ok(None);
    };
    if !is_direct_color_u8_format(fmt)
        || !shared_direct_plan.matches(DirectColorRegionKey {
            input_ptr: job.input.as_ptr() as usize,
            input_len: job.input.len(),
            roi: job.roi,
            scale: job.scale,
        })
    {
        return Ok(None);
    }

    let decoded = job.roi.scaled_covering(job.scale);
    validate_buffer((decoded.w, decoded.h), job.out.len(), job.stride, fmt)?;
    ctx.codec_mut().record_tile_decode();
    execute_direct_color_plan_u8_into(
        &shared_direct_plan.plan,
        shared_direct_plan.output_region,
        scratch,
        job.out,
        job.stride,
        fmt,
    )?;
    Ok(Some(DecodeOutcome {
        decoded,
        warnings: Vec::new(),
    }))
}

fn decode_tile_region_scaled_direct_color_u8_in_context(
    job: &mut TileRegionScaledDecodeJob<'_, '_>,
    ctx: &mut DecoderContext<J2kContext>,
    fmt: PixelFormat,
    scratch: &mut J2kDirectCpuScratch,
    cache: &mut Option<DirectColorRegionCache>,
) -> Result<Option<BatchOutcome>, J2kError> {
    if !is_direct_color_u8_format(fmt) || job.scale == Downscale::None {
        return Ok(None);
    }

    let decoded = job.roi.scaled_covering(job.scale);
    validate_buffer((decoded.w, decoded.h), job.out.len(), job.stride, fmt)?;
    let key = DirectColorRegionKey {
        input_ptr: job.input.as_ptr() as usize,
        input_len: job.input.len(),
        roi: job.roi,
        scale: job.scale,
    };
    if !cache.as_ref().is_some_and(|cache| cache.matches(key)) {
        let Some((plan, output_region)) =
            build_direct_color_region_plan(job.input, job.roi, job.scale)?
        else {
            return Ok(None);
        };
        *cache = Some(DirectColorRegionCache {
            input_ptr: key.input_ptr,
            input_len: key.input_len,
            roi: key.roi,
            scale: key.scale,
            output_region,
            plan,
        });
    }

    let cache = cache
        .as_ref()
        .ok_or_else(|| J2kError::Backend("internal direct color plan cache missing".to_string()))?;
    ctx.codec_mut().record_tile_decode();
    execute_direct_color_plan_u8_into(
        &cache.plan,
        cache.output_region,
        scratch,
        job.out,
        job.stride,
        fmt,
    )?;
    Ok(Some(DecodeOutcome {
        decoded,
        warnings: Vec::new(),
    }))
}

#[derive(Clone, Copy)]
struct DirectColorRegionKey {
    input_ptr: usize,
    input_len: usize,
    roi: Rect,
    scale: Downscale,
}

impl DirectColorRegionCache {
    fn matches(&self, key: DirectColorRegionKey) -> bool {
        self.input_ptr == key.input_ptr
            && self.input_len == key.input_len
            && self.roi == key.roi
            && self.scale == key.scale
    }
}

fn is_direct_color_u8_format(fmt: PixelFormat) -> bool {
    matches!(fmt, PixelFormat::Rgb8 | PixelFormat::Rgba8)
}

fn execute_direct_color_plan_u8_into(
    plan: &J2kDirectColorPlan,
    output_region: J2kRect,
    scratch: &mut J2kDirectCpuScratch,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    match fmt {
        PixelFormat::Rgb8 => {
            execute_direct_color_plan_rgb8_into(plan, output_region, scratch, out, stride)
        }
        PixelFormat::Rgba8 => {
            execute_direct_color_plan_rgba8_into(plan, output_region, scratch, out, stride)
        }
        _ => unreachable!("validated direct color output format"),
    }
    .map_err(|error| J2kError::Backend(error.to_string()))
}

fn build_direct_color_region_plan(
    input: &[u8],
    roi: Rect,
    scale: Downscale,
) -> Result<Option<(J2kDirectColorPlan, J2kRect)>, J2kError> {
    if !input_declares_htj2k(input) {
        return Ok(None);
    }

    let Ok(parsed) = parse_image_info(input) else {
        return Ok(None);
    };
    if !matches!(
        parsed.transfer_syntax,
        CompressedTransferSyntax::HtJpeg2000Lossless | CompressedTransferSyntax::HtJpeg2000Lossy
    ) {
        return Ok(None);
    }

    validate_region(roi, parsed.info.dimensions)?;
    let target_dims = (
        parsed.info.dimensions.0.div_ceil(scale.denominator()),
        parsed.info.dimensions.1.div_ceil(scale.denominator()),
    );
    let output_region = roi.scaled_covering(scale);
    let image = backend::image(
        input,
        DecodeSettings {
            target_resolution: Some(target_dims),
            ..DecodeSettings::default()
        },
    )?;
    validate_region(output_region, (image.width(), image.height()))?;

    let mut native_context = signinum_j2k_native::DecoderContext::default();
    match image.build_direct_color_plan_region_with_context(
        &mut native_context,
        (
            output_region.x,
            output_region.y,
            output_region.w,
            output_region.h,
        ),
    ) {
        Ok(plan) if direct_color_plan_uses_only_htj2k(&plan) => Ok(Some((
            plan,
            J2kRect {
                x0: output_region.x,
                y0: output_region.y,
                x1: output_region.x + output_region.w,
                y1: output_region.y + output_region.h,
            },
        ))),
        Ok(_) => Ok(None),
        Err(error) if is_unsupported_direct_color_plan_error(error) => Ok(None),
        Err(error) => Err(J2kError::Backend(error.to_string())),
    }
}

fn input_declares_htj2k(input: &[u8]) -> bool {
    const JP2_SIGNATURE_PREFIX: [u8; 8] = [0, 0, 0, 12, b'j', b'P', b' ', b' '];

    if raw_codestream_declares_htj2k(input) {
        return true;
    }
    if !input.starts_with(&JP2_SIGNATURE_PREFIX) {
        return false;
    }
    jp2_codestream_declares_htj2k(input)
}

fn jp2_codestream_declares_htj2k(input: &[u8]) -> bool {
    let mut offset = 0usize;
    while offset < input.len() {
        let Some((box_type, payload_start, end)) = read_jp2_box_header(input, offset) else {
            return false;
        };
        if end > input.len() {
            return false;
        }
        if &box_type == b"jp2c" {
            return raw_codestream_declares_htj2k(&input[payload_start..end]);
        }
        offset = end;
    }
    false
}

fn read_jp2_box_header(input: &[u8], offset: usize) -> Option<([u8; 4], usize, usize)> {
    let header = input.get(offset..offset.checked_add(8)?)?;
    let lbox = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    let box_type = [header[4], header[5], header[6], header[7]];
    match lbox {
        0 => Some((box_type, offset.checked_add(8)?, input.len())),
        1 => {
            let extended = input.get(offset.checked_add(8)?..offset.checked_add(16)?)?;
            let xlbox = u64::from_be_bytes([
                extended[0],
                extended[1],
                extended[2],
                extended[3],
                extended[4],
                extended[5],
                extended[6],
                extended[7],
            ]);
            if xlbox < 16 || xlbox > usize::MAX as u64 {
                return None;
            }
            let end = offset.checked_add(xlbox as usize)?;
            Some((box_type, offset.checked_add(16)?, end))
        }
        length if length < 8 => None,
        length => {
            let end = offset.checked_add(length as usize)?;
            Some((box_type, offset.checked_add(8)?, end))
        }
    }
}

fn raw_codestream_declares_htj2k(input: &[u8]) -> bool {
    const MARKER_SOC: u8 = 0x4f;
    const MARKER_CAP: u8 = 0x50;
    const MARKER_COD: u8 = 0x52;
    const MARKER_SOT: u8 = 0x90;
    const MARKER_SOD: u8 = 0x93;
    const MARKER_EOC: u8 = 0xd9;

    if input.len() < 2 || input[0] != 0xff || input[1] != MARKER_SOC {
        return false;
    }

    let mut offset = 2usize;
    while offset + 2 <= input.len() {
        if input[offset] != 0xff {
            return false;
        }
        let marker = input[offset + 1];
        offset += 2;
        match marker {
            MARKER_SOT | MARKER_SOD | MARKER_EOC => return false,
            MARKER_CAP => return true,
            MARKER_COD => {
                let Some(payload) = read_codestream_segment_payload(input, &mut offset) else {
                    return false;
                };
                if payload.get(8).is_some_and(|style| style & 0x40 != 0) {
                    return true;
                }
            }
            _ => {
                if read_codestream_segment_payload(input, &mut offset).is_none() {
                    return false;
                }
            }
        }
    }
    false
}

fn read_codestream_segment_payload<'a>(input: &'a [u8], offset: &mut usize) -> Option<&'a [u8]> {
    let len_bytes = input.get(*offset..offset.checked_add(2)?)?;
    let len = u16::from_be_bytes([len_bytes[0], len_bytes[1]]) as usize;
    if len < 2 {
        return None;
    }
    let payload_start = offset.checked_add(2)?;
    let payload_end = offset.checked_add(len)?;
    let payload = input.get(payload_start..payload_end)?;
    *offset = payload_end;
    Some(payload)
}

fn direct_color_plan_uses_only_htj2k(plan: &J2kDirectColorPlan) -> bool {
    plan.component_plans.iter().all(|component| {
        component.steps.iter().any(|step| {
            matches!(
                step,
                signinum_j2k_native::J2kDirectGrayscaleStep::HtSubBand(sub_band)
                    if !sub_band.jobs.is_empty()
            )
        }) && component.steps.iter().all(|step| {
            !matches!(
                step,
                signinum_j2k_native::J2kDirectGrayscaleStep::ClassicSubBand(_)
            )
        })
    })
}

fn is_unsupported_direct_color_plan_error(error: NativeDecodeError) -> bool {
    matches!(
        error,
        NativeDecodeError::Decoding(NativeDecodingError::UnsupportedFeature(_))
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use signinum_j2k_native::{encode, encode_htj2k, EncodeOptions};

    fn encode_rgb_codestream(htj2k: bool) -> Vec<u8> {
        let pixels = (0..16 * 16 * 3)
            .map(|idx| ((idx * 11 + idx / 3) & 0xff) as u8)
            .collect::<Vec<_>>();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        };
        if htj2k {
            encode_htj2k(&pixels, 16, 16, 3, 8, false, &options).expect("encode HTJ2K")
        } else {
            encode(&pixels, 16, 16, 3, 8, false, &options).expect("encode J2K")
        }
    }

    fn wrap_codestream_jp2(codestream: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&[0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0D, 0x0A, 0x87, 0x0A]);
        bytes.extend_from_slice(&[
            0, 0, 0, 20, b'f', b't', b'y', b'p', b'j', b'p', b'2', b' ', 0, 0, 0, 0, b'j', b'p',
            b'2', b' ',
        ]);
        bytes.extend_from_slice(&[
            0, 0, 0, 45, b'j', b'p', b'2', b'h', 0, 0, 0, 22, b'i', b'h', b'd', b'r',
        ]);
        bytes.extend_from_slice(&16_u32.to_be_bytes());
        bytes.extend_from_slice(&16_u32.to_be_bytes());
        bytes.extend_from_slice(&3_u16.to_be_bytes());
        bytes.extend_from_slice(&[7, 7, 0, 0]);
        bytes.extend_from_slice(&[0, 0, 0, 15, b'c', b'o', b'l', b'r', 1, 0, 0]);
        bytes.extend_from_slice(&16_u32.to_be_bytes());

        let len = (8 + codestream.len()) as u32;
        bytes.extend_from_slice(&len.to_be_bytes());
        bytes.extend_from_slice(b"jp2c");
        bytes.extend_from_slice(codestream);
        bytes
    }

    #[test]
    fn htj2k_eligibility_accepts_raw_and_jp2_wrapped_inputs() {
        let raw_htj2k = encode_rgb_codestream(true);
        let jp2_htj2k = wrap_codestream_jp2(&raw_htj2k);
        let raw_classic = encode_rgb_codestream(false);
        let jp2_classic = wrap_codestream_jp2(&raw_classic);

        assert!(input_declares_htj2k(&raw_htj2k));
        assert!(input_declares_htj2k(&jp2_htj2k));
        assert!(!input_declares_htj2k(&raw_classic));
        assert!(!input_declares_htj2k(&jp2_classic));
    }
}
