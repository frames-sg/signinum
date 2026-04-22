// SPDX-License-Identifier: Apache-2.0

//! Reusing a `ScratchPool` across many decodes must produce byte-identical
//! output on every iteration. Regression guard for Phase 3.

use slidecodec_jpeg::{Decoder, Downscale, PixelFormat, Rect, ScratchPool};

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
    dec.decode_scaled_into_with_scratch(
        &mut pool,
        &mut out,
        stride,
        PixelFormat::Rgb8,
        Downscale::None,
    )
    .unwrap();
    let reference = out.clone();

    for i in 0..50 {
        out.fill(0);
        dec.decode_scaled_into_with_scratch(
            &mut pool,
            &mut out,
            stride,
            PixelFormat::Rgb8,
            Downscale::None,
        )
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
    dec.decode_scaled_into_with_scratch(
        &mut pool,
        &mut out,
        stride,
        PixelFormat::Gray8,
        Downscale::None,
    )
    .unwrap();
    let reference = out.clone();

    for i in 0..50 {
        out.fill(0xFF);
        dec.decode_scaled_into_with_scratch(
            &mut pool,
            &mut out,
            stride,
            PixelFormat::Gray8,
            Downscale::None,
        )
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
    dec.decode_scaled_into(&mut out_fresh, stride, PixelFormat::Rgb8, Downscale::None)
        .unwrap();

    let mut pool = ScratchPool::new();
    let mut out_pooled = vec![0u8; len];
    dec.decode_scaled_into_with_scratch(
        &mut pool,
        &mut out_pooled,
        stride,
        PixelFormat::Rgb8,
        Downscale::None,
    )
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
            .decode_scaled_into_with_scratch(
                &mut pool,
                &mut g_out,
                gw as usize,
                PixelFormat::Gray8,
                Downscale::None,
            )
            .unwrap();
        dec_rgb
            .decode_scaled_into_with_scratch(
                &mut pool,
                &mut r_out,
                r_stride,
                PixelFormat::Rgb8,
                Downscale::None,
            )
            .unwrap();
    }

    let mut g_ref = vec![0u8; g_len];
    dec_gray
        .decode_scaled_into(&mut g_ref, gw as usize, PixelFormat::Gray8, Downscale::None)
        .unwrap();
    assert_eq!(g_out, g_ref);

    let mut r_ref = vec![0u8; r_len];
    dec_rgb
        .decode_scaled_into(&mut r_ref, r_stride, PixelFormat::Rgb8, Downscale::None)
        .unwrap();
    assert_eq!(r_out, r_ref);
}

#[test]
fn region_scaled_rgb8_is_byte_stable_across_pool_reuse() {
    let dec = Decoder::new(BASELINE_420).unwrap();
    let roi = Rect {
        x: 5,
        y: 5,
        w: 9,
        h: 9,
    };
    let x0 = roi.x / 4;
    let y0 = roi.y / 4;
    let x1 = (roi.x + roi.w).div_ceil(4);
    let y1 = (roi.y + roi.h).div_ceil(4);
    let (w, h) = (x1 - x0, y1 - y0);
    let stride = (w as usize) * 3;
    let mut out = vec![0u8; stride * h as usize];
    let mut fresh = vec![0u8; stride * h as usize];
    let mut pool = ScratchPool::new();

    dec.decode_region_scaled_into(
        &mut fresh,
        stride,
        PixelFormat::Rgb8,
        roi,
        Downscale::Quarter,
    )
    .unwrap();

    dec.decode_region_scaled_into_with_scratch(
        &mut pool,
        &mut out,
        stride,
        PixelFormat::Rgb8,
        roi,
        Downscale::Quarter,
    )
    .unwrap();
    assert_eq!(out, fresh, "first pooled decode must match fresh decode");

    for i in 0..50 {
        out.fill(0xA5);
        dec.decode_region_scaled_into_with_scratch(
            &mut pool,
            &mut out,
            stride,
            PixelFormat::Rgb8,
            roi,
            Downscale::Quarter,
        )
        .unwrap();
        assert_eq!(out, fresh, "iteration {i} diverged");
    }
}
