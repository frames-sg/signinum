// SPDX-License-Identifier: Apache-2.0

//! Batch decode via [`Decoder::decode_tile`]: sequential output must match
//! parallel output byte-for-byte across a worker pool. Validates the
//! Phase 5 tile primitive under `std::thread::scope` (no `rayon`
//! dependency).

use slidecodec_jpeg::{
    decode_tile_into_in_context, decode_tile_region_scaled_into_in_context, Decoder,
    DecoderContext, Downscale, PixelFormat, Rect, RowSink, ScratchPool,
};
use std::thread;

const BASELINE_420: &[u8] = include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");

const BATCH_SIZE: usize = 100;

#[derive(Default)]
struct CollectRows {
    rows: Vec<u8>,
}

impl RowSink<u8> for CollectRows {
    type Error = slidecodec_jpeg::JpegError;

    fn write_row(&mut self, _y: u32, row: &[u8]) -> Result<(), slidecodec_jpeg::JpegError> {
        self.rows.extend_from_slice(row);
        Ok(())
    }
}

fn decode_tile_bytes(bytes: &[u8], ctx: &mut DecoderContext, pool: &mut ScratchPool) -> Vec<u8> {
    let mut sink = CollectRows::default();
    Decoder::decode_tile(bytes, ctx, pool, &mut sink).expect("Decoder::decode_tile");
    sink.rows
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
