// SPDX-License-Identifier: Apache-2.0

//! Reusing a `ScratchPool` across many decodes must produce byte-identical
//! output on every iteration. Regression guard for Phase 3.

use slidecodec_jpeg::{Decoder, OutputFormat, ScratchPool};

const BASELINE_420: &[u8] = include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");
const GRAYSCALE_8X8: &[u8] = include_bytes!("../../../corpus/conformance/grayscale_8x8.jpg");

#[test]
fn rgb8_decode_is_byte_stable_across_pool_reuse() {
    let dec = Decoder::new(BASELINE_420).unwrap();
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * 3;
    let len = stride * h as usize;

    let mut pool = ScratchPool::new();
    let mut out = vec![0u8; len];
    dec.decode_into_with_scratch(&mut pool, &mut out, stride, OutputFormat::Rgb8)
        .unwrap();
    let reference = out.clone();

    for i in 0..50 {
        out.fill(0);
        dec.decode_into_with_scratch(&mut pool, &mut out, stride, OutputFormat::Rgb8)
            .unwrap();
        assert_eq!(out, reference, "iteration {i} diverged from reference");
    }
}

#[test]
fn gray8_decode_is_byte_stable_across_pool_reuse() {
    let dec = Decoder::new(GRAYSCALE_8X8).unwrap();
    let (w, h) = dec.info().dimensions;
    let stride = w as usize;
    let len = stride * h as usize;

    let mut pool = ScratchPool::new();
    let mut out = vec![0u8; len];
    dec.decode_into_with_scratch(&mut pool, &mut out, stride, OutputFormat::Gray8)
        .unwrap();
    let reference = out.clone();

    for i in 0..50 {
        out.fill(0xFF);
        dec.decode_into_with_scratch(&mut pool, &mut out, stride, OutputFormat::Gray8)
            .unwrap();
        assert_eq!(out, reference, "iteration {i} diverged from reference");
    }
}

#[test]
fn shared_pool_matches_fresh_pool_output() {
    let dec = Decoder::new(BASELINE_420).unwrap();
    let (w, h) = dec.info().dimensions;
    let stride = w as usize * 3;
    let len = stride * h as usize;

    let mut out_fresh = vec![0u8; len];
    dec.decode_into(&mut out_fresh, stride, OutputFormat::Rgb8)
        .unwrap();

    let mut pool = ScratchPool::new();
    let mut out_pooled = vec![0u8; len];
    dec.decode_into_with_scratch(&mut pool, &mut out_pooled, stride, OutputFormat::Rgb8)
        .unwrap();

    assert_eq!(
        out_fresh, out_pooled,
        "decode_into vs decode_into_with_scratch diverged on RGB"
    );
}

#[test]
fn shared_pool_matches_fresh_pool_after_reuse() {
    // Guarantee the pool's stale buffers (from a previous decode) don't leak
    // into a subsequent decode of a different-shape image.
    let dec_gray = Decoder::new(GRAYSCALE_8X8).unwrap();
    let (gw, gh) = dec_gray.info().dimensions;
    let g_len = (gw as usize) * (gh as usize);

    let dec_rgb = Decoder::new(BASELINE_420).unwrap();
    let (rw, rh) = dec_rgb.info().dimensions;
    let r_stride = (rw as usize) * 3;
    let r_len = r_stride * (rh as usize);

    let mut pool = ScratchPool::new();
    let mut g_out = vec![0u8; g_len];
    let mut r_out = vec![0u8; r_len];

    // Decode gray, then rgb, alternating — asserts the pool resizes safely.
    for _ in 0..10 {
        dec_gray
            .decode_into_with_scratch(&mut pool, &mut g_out, gw as usize, OutputFormat::Gray8)
            .unwrap();
        dec_rgb
            .decode_into_with_scratch(&mut pool, &mut r_out, r_stride, OutputFormat::Rgb8)
            .unwrap();
    }

    let mut g_ref = vec![0u8; g_len];
    dec_gray
        .decode_into(&mut g_ref, gw as usize, OutputFormat::Gray8)
        .unwrap();
    assert_eq!(g_out, g_ref);

    let mut r_ref = vec![0u8; r_len];
    dec_rgb
        .decode_into(&mut r_ref, r_stride, OutputFormat::Rgb8)
        .unwrap();
    assert_eq!(r_out, r_ref);
}
