// SPDX-License-Identifier: Apache-2.0

use core::convert::Infallible;
use std::num::NonZeroUsize;

use signinum_core::{DecodeOutcome, DecoderContext, Downscale, PixelFormat, Rect, TileBatchDecode};

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

/// Worker configuration for J2K CPU tile batches.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TileBatchOptions {
    /// Worker count. `None` uses [`std::thread::available_parallelism`].
    pub workers: Option<NonZeroUsize>,
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
type IndexedBatchResult = (usize, Result<BatchOutcome, J2kError>);

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
    let worker_count = tile_batch_worker_count(job_count, options);
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

    collect_tile_batch_results(job_count, results)
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
    let worker_count = tile_batch_worker_count(job_count, options);
    let chunk_size = job_count.div_ceil(worker_count);
    let results = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for (chunk_index, chunk) in jobs.chunks_mut(chunk_size).enumerate() {
            let start_index = chunk_index * chunk_size;
            handles.push(scope.spawn(move || {
                decode_tile_region_scaled_job_chunk(
                    start_index,
                    chunk,
                    fmt,
                    inner_parallelism_for_batch(job_count),
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

    collect_tile_batch_results(job_count, results)
}

fn tile_batch_worker_count(batch_size: usize, options: TileBatchOptions) -> usize {
    if batch_size <= 1 {
        return 1;
    }
    let workers = options.workers.map_or_else(
        || std::thread::available_parallelism().map_or(1, NonZeroUsize::get),
        NonZeroUsize::get,
    );
    workers.max(1).min(batch_size)
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
) -> Vec<IndexedBatchResult> {
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
) -> Vec<IndexedBatchResult> {
    let mut ctx = DecoderContext::<J2kContext>::new();
    ctx.codec_mut()
        .set_cpu_decode_parallelism(inner_parallelism);
    let mut pool = J2kScratchPool::new();
    let mut results = Vec::with_capacity(jobs.len());
    for (local_index, job) in jobs.iter_mut().enumerate() {
        let outcome = decode_tile_region_scaled_into_in_context(
            job.input, &mut ctx, &mut pool, job.out, job.stride, fmt, job.roi, job.scale,
        );
        results.push((start_index + local_index, outcome));
    }
    results
}

fn collect_tile_batch_results(
    job_count: usize,
    results: Vec<IndexedBatchResult>,
) -> Result<Vec<BatchOutcome>, TileBatchError> {
    let mut outcomes = Vec::with_capacity(job_count);
    outcomes.resize_with(job_count, || None);
    let mut first_error = None::<TileBatchError>;
    for (index, result) in results {
        match result {
            Ok(outcome) => outcomes[index] = Some(outcome),
            Err(source) => {
                if first_error
                    .as_ref()
                    .is_none_or(|current| index < current.index)
                {
                    first_error = Some(TileBatchError { index, source });
                }
            }
        }
    }

    if let Some(err) = first_error {
        return Err(err);
    }

    Ok(outcomes
        .into_iter()
        .map(|outcome| outcome.expect("successful batch stores one outcome per tile"))
        .collect())
}
