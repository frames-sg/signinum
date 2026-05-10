// SPDX-License-Identifier: Apache-2.0

//! Batch decode via [`Decoder::decode_tile`]: sequential output must match
//! parallel output byte-for-byte across a worker pool. Validates the
//! Phase 5 tile primitive under `std::thread::scope`.

use signinum_jpeg::{
    decode_tile_into_in_context, decode_tile_region_scaled_into_in_context, decode_tiles_into,
    decode_tiles_into_with_options, decode_tiles_region_scaled_into, decode_tiles_scaled_into,
    decode_tiles_scaled_into_with_options, ColorTransform, DecodeOptions, Decoder, DecoderContext,
    Downscale, PixelFormat, Rect, RowSink, ScratchPool, TileBatchOptions, TileDecodeJob,
    TileRegionScaledDecodeJob, TileScaledDecodeJob,
};
mod fixtures;
use fixtures::progressive_8x8_jpeg;
use std::num::NonZeroUsize;
use std::thread;

const BASELINE_420: &[u8] = include_bytes!("../fixtures/conformance/baseline_420_16x16.jpg");

const BATCH_SIZE: usize = 100;

#[derive(Default)]
struct CollectRows {
    rows: Vec<u8>,
}

impl RowSink<u8> for CollectRows {
    type Error = signinum_jpeg::JpegError;

    fn write_row(&mut self, _y: u32, row: &[u8]) -> Result<(), signinum_jpeg::JpegError> {
        self.rows.extend_from_slice(row);
        Ok(())
    }
}

fn decode_tile_bytes(bytes: &[u8], ctx: &mut DecoderContext, pool: &mut ScratchPool) -> Vec<u8> {
    let mut sink = CollectRows::default();
    Decoder::decode_tile(bytes, ctx, pool, &mut sink).expect("Decoder::decode_tile");
    sink.rows
}

fn decode_tile_rgb8_reference(bytes: &[u8]) -> (Vec<u8>, usize) {
    let dec = Decoder::new(bytes).expect("fixture decoder");
    let (width, height) = dec.info().dimensions;
    let stride = width as usize * 3;
    let mut out = vec![0u8; stride * height as usize];
    dec.decode_into(&mut out, stride, PixelFormat::Rgb8)
        .expect("fixture decode_into");
    (out, stride)
}

#[test]
fn production_batch_decode_empty_input_succeeds() {
    let mut jobs: Vec<TileDecodeJob<'_, '_>> = Vec::new();

    let outcomes = decode_tiles_into(&mut jobs, PixelFormat::Rgb8, TileBatchOptions::default())
        .expect("empty batch succeeds");

    assert!(outcomes.is_empty());
}

#[test]
fn production_batch_decode_worker_one_matches_single_tile_decode() {
    let (expected, stride) = decode_tile_rgb8_reference(BASELINE_420);
    let mut actual = vec![0u8; expected.len()];
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(1),
    };

    let outcomes = {
        let mut jobs = vec![TileDecodeJob {
            input: BASELINE_420,
            out: actual.as_mut_slice(),
            stride,
        }];
        decode_tiles_into(&mut jobs, PixelFormat::Rgb8, options).expect("batch decode")
    };

    assert_eq!(outcomes.len(), 1);
    assert_eq!(actual, expected);
}

#[test]
fn production_batch_decode_progressive8_matches_single_tile_decode() {
    let bytes = progressive_8x8_jpeg();
    let (expected, stride) = decode_tile_rgb8_reference(&bytes);
    let mut actual = vec![0u8; expected.len()];

    let outcomes = {
        let mut jobs = vec![TileDecodeJob {
            input: &bytes,
            out: actual.as_mut_slice(),
            stride,
        }];
        decode_tiles_into(
            &mut jobs,
            PixelFormat::Rgb8,
            TileBatchOptions {
                workers: NonZeroUsize::new(1),
            },
        )
        .expect("progressive batch decode")
    };

    assert_eq!(outcomes.len(), 1);
    assert_eq!(actual, expected);
}

#[test]
fn production_batch_decode_parallel_preserves_order_and_output() {
    const JOBS: usize = 32;
    let (expected, stride) = decode_tile_rgb8_reference(BASELINE_420);
    let mut outputs = (0..JOBS)
        .map(|_| vec![0u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(4),
    };

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileDecodeJob {
                input: BASELINE_420,
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        decode_tiles_into(&mut jobs, PixelFormat::Rgb8, options).expect("batch decode")
    };

    assert_eq!(outcomes.len(), JOBS);
    for (index, out) in outputs.iter().enumerate() {
        assert_eq!(out, &expected, "tile {index} output diverged");
    }
}

#[test]
fn production_batch_decode_with_options_preserves_forced_color_transform() {
    const JOBS: usize = 8;
    let decode_options = DecodeOptions::default().with_color_transform(ColorTransform::ForceRgb);
    let dec = Decoder::new_with_options(BASELINE_420, decode_options).expect("fixture decoder");
    let (width, height) = dec.info().dimensions;
    let stride = width as usize * 3;
    let mut expected = vec![0u8; stride * height as usize];
    dec.decode_into(&mut expected, stride, PixelFormat::Rgb8)
        .expect("reference forced-RGB decode");
    let mut outputs = (0..JOBS)
        .map(|_| vec![0u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(2),
    };

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileDecodeJob {
                input: BASELINE_420,
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        decode_tiles_into_with_options(&mut jobs, PixelFormat::Rgb8, decode_options, options)
            .expect("batch decode with options")
    };

    assert_eq!(outcomes.len(), JOBS);
    for (index, out) in outputs.iter().enumerate() {
        assert_eq!(out, &expected, "tile {index} output diverged");
    }
}

#[test]
fn production_batch_decode_reports_first_failing_tile_index() {
    let (expected, stride) = decode_tile_rgb8_reference(BASELINE_420);
    let mut outputs = (0..3)
        .map(|_| vec![0u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(2),
    };

    let err = {
        let inputs: [&[u8]; 3] = [BASELINE_420, b"not a jpeg", BASELINE_420];
        let mut jobs = inputs
            .into_iter()
            .zip(outputs.iter_mut())
            .map(|(input, out)| TileDecodeJob {
                input,
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        decode_tiles_into(&mut jobs, PixelFormat::Rgb8, options).expect_err("bad tile fails")
    };

    assert_eq!(err.index, 1);
}

#[test]
fn sequential_and_parallel_batch_produce_identical_output() {
    let tiles: Vec<&[u8]> = (0..BATCH_SIZE).map(|_| BASELINE_420).collect();

    let sequential: Vec<Vec<u8>> = {
        let mut pool = ScratchPool::new();
        let mut ctx = DecoderContext::new();
        tiles
            .iter()
            .map(|bytes| decode_tile_bytes(bytes, &mut ctx, &mut pool))
            .collect()
    };

    let parallel: Vec<Vec<u8>> = thread::scope(|scope| {
        const WORKERS: usize = 4;
        let chunk_size = tiles.len().div_ceil(WORKERS);
        let handles: Vec<_> = tiles
            .chunks(chunk_size)
            .map(|chunk| {
                scope.spawn(|| {
                    let mut pool = ScratchPool::new();
                    let mut ctx = DecoderContext::new();
                    chunk
                        .iter()
                        .map(|bytes| decode_tile_bytes(bytes, &mut ctx, &mut pool))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("worker panicked"))
            .collect()
    });

    assert_eq!(sequential.len(), parallel.len());
    for (i, (seq, par)) in sequential.iter().zip(parallel.iter()).enumerate() {
        assert_eq!(
            seq, par,
            "tile {i} diverged between sequential and parallel"
        );
    }
}

#[test]
fn pool_reuse_across_batch_matches_fresh_pool() {
    let mut reused_pool = ScratchPool::new();
    let mut reused_ctx = DecoderContext::new();
    let reused_outputs: Vec<Vec<u8>> = (0..BATCH_SIZE)
        .map(|_| decode_tile_bytes(BASELINE_420, &mut reused_ctx, &mut reused_pool))
        .collect();

    let fresh_outputs: Vec<Vec<u8>> = (0..BATCH_SIZE)
        .map(|_| {
            let mut pool = ScratchPool::new();
            let mut ctx = DecoderContext::new();
            decode_tile_bytes(BASELINE_420, &mut ctx, &mut pool)
        })
        .collect();

    for (i, (reused, fresh)) in reused_outputs.iter().zip(fresh_outputs.iter()).enumerate() {
        assert_eq!(reused, fresh, "iter {i} reused-pool output diverged");
    }
}

#[test]
fn tile_buffer_decode_matches_decoder_decode_into() {
    let dec = Decoder::new(BASELINE_420).expect("fixture decoder");
    let (width, height) = dec.info().dimensions;
    let stride = width as usize * 3;
    let mut expected = vec![0u8; stride * height as usize];
    let mut actual = vec![0u8; expected.len()];
    dec.decode_into(&mut expected, stride, PixelFormat::Rgb8)
        .expect("baseline decode_into");

    let mut ctx = DecoderContext::new();
    let mut pool = ScratchPool::new();
    decode_tile_into_in_context(
        BASELINE_420,
        &mut ctx,
        &mut pool,
        &mut actual,
        stride,
        PixelFormat::Rgb8,
    )
    .expect("tile decode_into_in_context");

    assert_eq!(actual, expected);
}

#[test]
fn tile_region_scaled_decode_matches_decoder_region_decode() {
    let dec = Decoder::new(BASELINE_420).expect("fixture decoder");
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let denom = 4;
    let scaled_w = (roi.x + roi.w).div_ceil(denom) - roi.x / denom;
    let scaled_h = (roi.y + roi.h).div_ceil(denom) - roi.y / denom;
    let stride = scaled_w as usize * 3;
    let mut expected = vec![0u8; stride * scaled_h as usize];
    let mut actual = vec![0u8; expected.len()];
    dec.decode_region_scaled_into(
        &mut expected,
        stride,
        PixelFormat::Rgb8,
        roi,
        Downscale::Quarter,
    )
    .expect("core region-scaled decode");

    let mut ctx = DecoderContext::new();
    let mut pool = ScratchPool::new();
    decode_tile_region_scaled_into_in_context(
        BASELINE_420,
        &mut ctx,
        &mut pool,
        &mut actual,
        stride,
        PixelFormat::Rgb8,
        roi,
        Downscale::Quarter,
    )
    .expect("tile region decode_into_in_context");

    assert_eq!(actual, expected);
}

#[test]
fn production_batch_region_scaled_decode_parallel_preserves_order_and_output() {
    const JOBS: usize = 32;
    let dec = Decoder::new(BASELINE_420).expect("fixture decoder");
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let denom = 4;
    let scaled_w = (roi.x + roi.w).div_ceil(denom) - roi.x / denom;
    let scaled_h = (roi.y + roi.h).div_ceil(denom) - roi.y / denom;
    let stride = scaled_w as usize * 3;
    let mut expected = vec![0u8; stride * scaled_h as usize];
    dec.decode_region_scaled_into(
        &mut expected,
        stride,
        PixelFormat::Rgb8,
        roi,
        Downscale::Quarter,
    )
    .expect("reference region-scaled decode");
    let mut outputs = (0..JOBS)
        .map(|_| vec![0u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(4),
    };

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileRegionScaledDecodeJob {
                input: BASELINE_420,
                out: out.as_mut_slice(),
                stride,
                roi,
                scale: Downscale::Quarter,
            })
            .collect::<Vec<_>>();
        decode_tiles_region_scaled_into(&mut jobs, PixelFormat::Rgb8, options)
            .expect("batch region-scaled decode")
    };

    assert_eq!(outcomes.len(), JOBS);
    for (index, out) in outputs.iter().enumerate() {
        assert_eq!(out, &expected, "tile {index} output diverged");
    }
}

#[test]
fn production_batch_scaled_decode_parallel_preserves_order_and_output() {
    const JOBS: usize = 32;
    let dec = Decoder::new(BASELINE_420).expect("fixture decoder");
    let scale = Downscale::Quarter;
    let denom = 4;
    let (width, height) = dec.info().dimensions;
    let scaled_w = width.div_ceil(denom);
    let scaled_h = height.div_ceil(denom);
    let stride = scaled_w as usize * 3;
    let mut expected = vec![0u8; stride * scaled_h as usize];
    dec.decode_scaled_into(&mut expected, stride, PixelFormat::Rgb8, scale)
        .expect("reference scaled decode");
    let mut outputs = (0..JOBS)
        .map(|_| vec![0u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(4),
    };

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileScaledDecodeJob {
                input: BASELINE_420,
                out: out.as_mut_slice(),
                stride,
                scale,
            })
            .collect::<Vec<_>>();
        decode_tiles_scaled_into(&mut jobs, PixelFormat::Rgb8, options)
            .expect("batch scaled decode")
    };

    assert_eq!(outcomes.len(), JOBS);
    for (index, out) in outputs.iter().enumerate() {
        assert_eq!(out, &expected, "tile {index} output diverged");
    }
}

#[test]
fn production_batch_scaled_decode_with_options_preserves_forced_color_transform() {
    const JOBS: usize = 8;
    let decode_options = DecodeOptions::default().with_color_transform(ColorTransform::ForceRgb);
    let dec = Decoder::new_with_options(BASELINE_420, decode_options).expect("fixture decoder");
    let scale = Downscale::Quarter;
    let denom = 4;
    let (width, height) = dec.info().dimensions;
    let scaled_w = width.div_ceil(denom);
    let scaled_h = height.div_ceil(denom);
    let stride = scaled_w as usize * 3;
    let mut expected = vec![0u8; stride * scaled_h as usize];
    dec.decode_scaled_into(&mut expected, stride, PixelFormat::Rgb8, scale)
        .expect("reference scaled forced-RGB decode");
    let mut outputs = (0..JOBS)
        .map(|_| vec![0u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(2),
    };

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileScaledDecodeJob {
                input: BASELINE_420,
                out: out.as_mut_slice(),
                stride,
                scale,
            })
            .collect::<Vec<_>>();
        decode_tiles_scaled_into_with_options(&mut jobs, PixelFormat::Rgb8, decode_options, options)
            .expect("batch scaled decode with options")
    };

    assert_eq!(outcomes.len(), JOBS);
    for (index, out) in outputs.iter().enumerate() {
        assert_eq!(out, &expected, "tile {index} output diverged");
    }
}
