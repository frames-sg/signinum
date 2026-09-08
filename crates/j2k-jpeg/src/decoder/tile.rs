// SPDX-License-Identifier: MIT OR Apache-2.0

//! One-shot tile and batch facade functions.

use super::warning_ownership::merged_warning_capacity_bytes;
use super::{
    additional_decode_scratch_bytes, output_format_from_parts, scaled_dimensions,
    scaled_rect_covering, DecodeOptions, DecodeOutcome, DecodedTile, Decoder, DecoderContext,
    Downscale, JpegError, JpegView, PixelFormat, PreparedJpeg, PreparedJpegTileJob, Rect,
    ScratchPool, TileBatchError, TileBatchOptions, TileDecodeJob, TileDecodeOutput,
    TileRegionScaledDecodeJob, TileScaledDecodeJob, Vec, CPU_ROI_CHECKPOINT_CADENCE_MCUS,
    DEFAULT_CONTEXT, DEFAULT_MAX_DECODE_BYTES,
};
use crate::allocation::checked_add_allocation_bytes;
use crate::internal::checkpoint::DeviceCheckpoint;

/// One-shot parse-plus-decode of an independent JPEG tile into the caller's
/// buffer, reusing a pre-allocated [`ScratchPool`]. This is the primitive
/// WSI tile-batch readers want: one function call per tile, with all
/// heap state external.
///
/// Parallelism is the caller's responsibility for this primitive. For
/// production batch decode, use [`decode_tiles_into`].
///
/// # Example
///
/// ```no_run
/// use j2k_jpeg::{decode_tile_into, PixelFormat, ScratchPool};
///
/// let bytes: &[u8] = &[];
/// let mut out = vec![0u8; 256 * 256 * 3];
/// let mut pool = ScratchPool::new();
/// decode_tile_into(bytes, &mut pool, &mut out, 256 * 3, PixelFormat::Rgb8)?;
/// # Ok::<(), j2k_jpeg::JpegError>(())
/// ```
///
/// # Errors
/// Forwarded from [`Decoder::new`] (parse) and
/// [`Decoder::decode_into_with_scratch`] (decode).
pub fn decode_tile_into(
    bytes: &[u8],
    pool: &mut ScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<DecodeOutcome, JpegError> {
    DEFAULT_CONTEXT.with(|ctx| {
        decode_tile_into_in_context(bytes, &mut ctx.borrow_mut(), pool, out, stride, fmt)
    })
}

/// One-shot parse-plus-decode of an independent JPEG tile into the caller's
/// buffer, reusing both caller-owned [`DecoderContext`] and caller-owned
/// [`ScratchPool`].
#[doc(hidden)]
pub fn decode_tile_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<DecodeOutcome, JpegError> {
    decode_tile_into_in_context_with_options(
        bytes,
        ctx,
        pool,
        out,
        stride,
        fmt,
        DecodeOptions::default(),
    )
}

/// One-shot parse-plus-decode of an independent JPEG tile into the caller's
/// buffer, reusing both caller-owned [`DecoderContext`] and caller-owned
/// [`ScratchPool`], with explicit JPEG decode options.
#[doc(hidden)]
pub fn decode_tile_into_in_context_with_options(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    options: DecodeOptions,
) -> Result<DecodeOutcome, JpegError> {
    let view = JpegView::parse_with_options(bytes, options)?;
    Decoder::with_view_in_context(view, ctx, |decoder| {
        decoder.decode_into_with_scratch(pool, out, stride, fmt)
    })
}

pub(crate) fn decode_prepared_jpeg_tile_rgb8_in_context(
    input: &PreparedJpeg<'_>,
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    out: &mut [u8],
    stride: usize,
    options: DecodeOptions,
) -> Result<DecodedTile, JpegError> {
    let view = JpegView::parse_with_options(input.as_bytes(), options)?;
    let dimensions = view.info().dimensions;
    Decoder::with_view_in_context(view, ctx, |decoder| {
        let outcome = decoder.decode_into_with_scratch(pool, out, stride, PixelFormat::Rgb8)?;
        Ok(DecodedTile {
            dimensions,
            decoded: outcome.decoded,
            warnings: outcome.warnings,
        })
    })
}

/// Plan the conservative codec-owned live set for one batch tile without
/// retaining a decoder. The claim covers the decoder's parsed/prepared
/// metadata and context reservation, reusable or direct decode scratch, and
/// the largest possible ROI entropy checkpoint cache for this image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PlannedJpegTileDecode {
    pub(crate) worker_live_bytes: usize,
    pub(crate) retained_result_bytes: usize,
}

pub(crate) fn planned_jpeg_tile_decode_live_bytes(
    input: &[u8],
    ctx: &mut DecoderContext,
    fmt: PixelFormat,
    roi: Option<Rect>,
    scale: Downscale,
    options: DecodeOptions,
) -> Result<PlannedJpegTileDecode, JpegError> {
    let view = JpegView::parse_with_options(input, options)?;
    Decoder::with_view_in_context(view, ctx, |decoder| {
        let source_rect = roi.unwrap_or_else(|| Rect::full(decoder.info.dimensions));
        if !source_rect.is_within(decoder.info.dimensions) {
            return Err(JpegError::RectOutOfBounds {
                rect: source_rect,
                width: decoder.info.dimensions.0,
                height: decoder.info.dimensions.1,
            });
        }

        let output_format = output_format_from_parts(decoder.info.sof_kind, fmt, scale)?;
        let downscale = output_format.downscale();
        let output_rect = if source_rect == Rect::full(decoder.info.dimensions) {
            Rect::full(scaled_dimensions(decoder.info.dimensions, downscale))
        } else {
            scaled_rect_covering(source_rect, downscale)?
        };
        let additional_scratch = additional_decode_scratch_bytes(
            decoder.info.sof_kind,
            decoder.info.dimensions,
            output_format,
            source_rect,
            output_rect,
            downscale,
        )?;
        let workspace_cap = decoder.decode_workspace_cap()?;
        let base_scratch = decoder.decode_scratch_bytes(workspace_cap)?;
        let scratch = checked_add_allocation_bytes(base_scratch, additional_scratch)?;
        let checkpoints = planned_roi_checkpoint_bytes(decoder, source_rect)?;
        let transient = checked_add_allocation_bytes(scratch, checkpoints)?;
        let retained = DEFAULT_MAX_DECODE_BYTES.checked_sub(workspace_cap).ok_or(
            JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: DEFAULT_MAX_DECODE_BYTES,
            },
        )?;
        let worker_live_bytes = checked_add_allocation_bytes(retained, transient)?;
        // The merged public result can retain all parsed warnings plus the bounded
        // scan-warning tail after the decoder itself has been dropped. Batch
        // planning counts that payload once per completed tile.
        let retained_result_bytes = merged_warning_capacity_bytes(decoder.warnings.capacity())?;
        Ok(PlannedJpegTileDecode {
            worker_live_bytes,
            retained_result_bytes,
        })
    })
}

fn planned_roi_checkpoint_bytes(decoder: &Decoder<'_>, roi: Rect) -> Result<usize, JpegError> {
    if roi == Rect::full(decoder.info.dimensions) || decoder.plan.restart_interval.is_some() {
        return Ok(0);
    }
    let mcu_width = u32::from(decoder.plan.sampling.max_h) * 8;
    let mcu_height = u32::from(decoder.plan.sampling.max_v) * 8;
    let total_mcus = decoder
        .info
        .dimensions
        .0
        .div_ceil(mcu_width)
        .checked_mul(decoder.info.dimensions.1.div_ceil(mcu_height))
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: DEFAULT_MAX_DECODE_BYTES,
        })?;
    let checkpoint_count = usize::try_from(total_mcus.div_ceil(CPU_ROI_CHECKPOINT_CADENCE_MCUS))
        .ok()
        .and_then(|count| count.checked_add(1))
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: DEFAULT_MAX_DECODE_BYTES,
        })?;
    let checkpoint_bytes = checkpoint_count
        .checked_mul(core::mem::size_of::<DeviceCheckpoint>())
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: DEFAULT_MAX_DECODE_BYTES,
        })?;
    checked_add_allocation_bytes(0, checkpoint_bytes)
}

/// Decode prepared TIFF/WSI JPEG tiles into caller-owned RGB8 output buffers.
///
/// Returned results preserve the caller's input order. Each job carries its
/// own [`DecodeOptions`], allowing container metadata to resolve RGB/YCbCr
/// interpretation independently per tile.
///
/// # Errors
///
/// Returns [`crate::PreparedTileBatchError`] when batch planning, allocation,
/// scheduling, or ordered collection fails. Individual JPEG decode failures
/// remain in the ordered per-tile result vector.
pub fn decode_prepared_jpeg_tiles_rgb8(
    jobs: &mut [PreparedJpegTileJob<'_, '_>],
) -> Result<Vec<Result<DecodedTile, JpegError>>, crate::PreparedTileBatchError> {
    crate::JpegBatchSession::new_one_shot(TileBatchOptions::default())
        .decode_prepared_jpeg_tiles_rgb8(jobs)
}

/// Decode independent JPEG tiles into caller-owned output buffers using a
/// scoped CPU worker pool.
///
/// Each worker owns one [`DecoderContext`] and one [`ScratchPool`], so repeated
/// tiles reuse parsed table state and heap scratch within that worker without
/// sharing mutable decoder state across threads. Returned outcomes preserve
/// the caller's input order.
///
/// # Errors
/// Returns [`TileBatchError`] with the first codec failure in input order, or
/// a typed infrastructure failure when no tile index applies.
pub fn decode_tiles_into(
    jobs: &mut [TileDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    decode_tiles_into_with_options(jobs, fmt, DecodeOptions::default(), options)
}

/// Decode independent JPEG tiles into caller-owned output buffers using a
/// scoped CPU worker pool and explicit JPEG decode options.
///
/// Use this variant when container metadata has already resolved ambiguous
/// three-component JPEG data to RGB or YCbCr via [`DecodeOptions`].
///
/// # Errors
/// Returns [`TileBatchError`] with the first codec failure in input order, or
/// a typed infrastructure failure when no tile index applies.
#[doc(hidden)]
pub fn decode_tiles_into_with_options(
    jobs: &mut [TileDecodeJob<'_, '_>],
    fmt: PixelFormat,
    decode_options: DecodeOptions,
    options: TileBatchOptions,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    crate::JpegBatchSession::new_one_shot(options).decode_tiles_into_with_options(
        jobs,
        fmt,
        decode_options,
    )
}

/// Decode independent JPEG tiles at reduced resolution into caller-owned
/// output buffers using a scoped CPU worker pool.
///
/// Each worker owns one [`DecoderContext`] and one [`ScratchPool`], so repeated
/// tiles reuse parsed table state and heap scratch within that worker without
/// sharing mutable decoder state across threads. Returned outcomes preserve
/// the caller's input order.
///
/// # Errors
/// Returns [`TileBatchError`] with the first codec failure in input order, or
/// a typed infrastructure failure when no tile index applies.
pub fn decode_tiles_scaled_into(
    jobs: &mut [TileScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    decode_tiles_scaled_into_with_options(jobs, fmt, DecodeOptions::default(), options)
}

/// Decode independent JPEG tiles at reduced resolution into caller-owned
/// output buffers using a scoped CPU worker pool and explicit JPEG decode
/// options.
///
/// # Errors
/// Returns [`TileBatchError`] with the first codec failure in input order, or
/// a typed infrastructure failure when no tile index applies.
#[doc(hidden)]
pub fn decode_tiles_scaled_into_with_options(
    jobs: &mut [TileScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    decode_options: DecodeOptions,
    options: TileBatchOptions,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    crate::JpegBatchSession::new_one_shot(options).decode_tiles_scaled_into_with_options(
        jobs,
        fmt,
        decode_options,
    )
}

/// Decode independent JPEG tile regions at reduced resolution into
/// caller-owned output buffers using a scoped CPU worker pool.
///
/// Each worker owns one [`DecoderContext`] and one [`ScratchPool`], so repeated
/// tiles reuse parsed table state and heap scratch within that worker without
/// sharing mutable decoder state across threads. Returned outcomes preserve
/// the caller's input order.
///
/// # Errors
/// Returns [`TileBatchError`] with the first codec failure in input order, or
/// a typed infrastructure failure when no tile index applies.
pub fn decode_tiles_region_scaled_into(
    jobs: &mut [TileRegionScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    decode_tiles_region_scaled_into_with_options(jobs, fmt, DecodeOptions::default(), options)
}

/// Decode independent JPEG tile regions at reduced resolution into
/// caller-owned output buffers using a scoped CPU worker pool and explicit JPEG
/// decode options.
///
/// # Errors
/// Returns [`TileBatchError`] with the first codec failure in input order, or
/// a typed infrastructure failure when no tile index applies.
#[doc(hidden)]
pub fn decode_tiles_region_scaled_into_with_options(
    jobs: &mut [TileRegionScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    decode_options: DecodeOptions,
    options: TileBatchOptions,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    crate::JpegBatchSession::new_one_shot(options).decode_tiles_region_scaled_into_with_options(
        jobs,
        fmt,
        decode_options,
    )
}

/// One-shot parse-plus-region-decode of an independent JPEG tile into the
/// caller's buffer, reusing both caller-owned [`DecoderContext`] and
/// caller-owned [`ScratchPool`].
#[doc(hidden)]
pub fn decode_tile_region_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    output: TileDecodeOutput<'_>,
    roi: Rect,
) -> Result<DecodeOutcome, JpegError> {
    decode_tile_region_into_in_context_with_options(
        bytes,
        ctx,
        pool,
        output,
        roi,
        DecodeOptions::default(),
    )
}

/// One-shot parse-plus-region-decode of an independent JPEG tile into the
/// caller's buffer, reusing caller-owned state and explicit JPEG decode
/// options.
#[doc(hidden)]
pub fn decode_tile_region_into_in_context_with_options(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    output: TileDecodeOutput<'_>,
    roi: Rect,
    options: DecodeOptions,
) -> Result<DecodeOutcome, JpegError> {
    let TileDecodeOutput { out, stride, fmt } = output;
    let view = JpegView::parse_with_options(bytes, options)?;
    Decoder::with_view_in_context(view, ctx, |decoder| {
        decoder.decode_region_into_with_scratch(pool, out, stride, fmt, roi)
    })
}

/// One-shot parse-plus-scaled-decode of an independent JPEG tile into the
/// caller's buffer, reusing both caller-owned [`DecoderContext`] and
/// caller-owned [`ScratchPool`].
#[doc(hidden)]
pub fn decode_tile_scaled_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    output: TileDecodeOutput<'_>,
    scale: Downscale,
) -> Result<DecodeOutcome, JpegError> {
    decode_tile_scaled_into_in_context_with_options(
        bytes,
        ctx,
        pool,
        output,
        scale,
        DecodeOptions::default(),
    )
}

/// One-shot parse-plus-scaled-decode of an independent JPEG tile into the
/// caller's buffer, reusing caller-owned state and explicit JPEG decode
/// options.
#[doc(hidden)]
pub fn decode_tile_scaled_into_in_context_with_options(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    output: TileDecodeOutput<'_>,
    scale: Downscale,
    options: DecodeOptions,
) -> Result<DecodeOutcome, JpegError> {
    let TileDecodeOutput { out, stride, fmt } = output;
    let view = JpegView::parse_with_options(bytes, options)?;
    Decoder::with_view_in_context(view, ctx, |decoder| {
        decoder.decode_scaled_into_with_scratch(pool, out, stride, fmt, scale)
    })
}

/// One-shot parse-plus-region-scaled-decode of an independent JPEG tile into
/// the caller's buffer, reusing both caller-owned [`DecoderContext`] and
/// caller-owned [`ScratchPool`].
#[doc(hidden)]
pub fn decode_tile_region_scaled_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    output: TileDecodeOutput<'_>,
    roi: Rect,
    scale: Downscale,
) -> Result<DecodeOutcome, JpegError> {
    decode_tile_region_scaled_into_in_context_with_options(
        bytes,
        ctx,
        pool,
        output,
        roi,
        scale,
        DecodeOptions::default(),
    )
}

/// One-shot parse-plus-region-scaled-decode of an independent JPEG tile into
/// the caller's buffer, reusing caller-owned state and explicit JPEG decode
/// options.
#[doc(hidden)]
pub fn decode_tile_region_scaled_into_in_context_with_options(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    output: TileDecodeOutput<'_>,
    roi: Rect,
    scale: Downscale,
    options: DecodeOptions,
) -> Result<DecodeOutcome, JpegError> {
    let TileDecodeOutput { out, stride, fmt } = output;
    let view = JpegView::parse_with_options(bytes, options)?;
    Decoder::with_view_in_context(view, ctx, |decoder| {
        decoder.decode_region_scaled_into_with_scratch(pool, out, stride, fmt, roi, scale)
    })
}

#[cfg(test)]
mod tests;
