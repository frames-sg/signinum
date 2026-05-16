// SPDX-License-Identifier: Apache-2.0

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use signinum_j2k::{
    encode_j2k_lossless, EncodeBackendPreference, J2kDecoder, J2kEncodeValidation,
    J2kLosslessEncodeOptions, J2kLosslessSamples, J2kScratchPool, PixelFormat, Rect,
};
use signinum_test_support::{patterned_gray8, patterned_rgb8};

const TILE_SIDE: u32 = 128;
const ROI_SIDE: u32 = 64;

fn bench_encode_options() -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions {
        backend: EncodeBackendPreference::CpuOnly,
        validation: J2kEncodeValidation::External,
        ..J2kLosslessEncodeOptions::default()
    }
}

fn encode_gray8_codestream(width: u32, height: u32) -> Vec<u8> {
    let pixels = patterned_gray8(width, height);
    let samples =
        J2kLosslessSamples::new(&pixels, width, height, 1, 8, false).expect("valid gray8 samples");
    encode_j2k_lossless(samples, &bench_encode_options())
        .expect("encode gray8 codestream")
        .codestream
}

fn encode_rgb8_codestream(width: u32, height: u32) -> Vec<u8> {
    let pixels = patterned_rgb8(width, height);
    let samples =
        J2kLosslessSamples::new(&pixels, width, height, 3, 8, false).expect("valid rgb8 samples");
    encode_j2k_lossless(samples, &bench_encode_options())
        .expect("encode rgb8 codestream")
        .codestream
}

fn bench_lossless_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("j2k_public_lossless_encode");

    let gray = patterned_gray8(TILE_SIDE, TILE_SIDE);
    let options = bench_encode_options();
    group.bench_function("gray8_128x128", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                black_box(gray.as_slice()),
                TILE_SIDE,
                TILE_SIDE,
                1,
                8,
                false,
            )
            .expect("valid gray8 samples");
            let encoded = encode_j2k_lossless(samples, &options).expect("encode gray8");
            black_box(encoded.codestream.len());
        });
    });

    let rgb = patterned_rgb8(TILE_SIDE, TILE_SIDE);
    group.bench_function("rgb8_128x128", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                black_box(rgb.as_slice()),
                TILE_SIDE,
                TILE_SIDE,
                3,
                8,
                false,
            )
            .expect("valid rgb8 samples");
            let encoded = encode_j2k_lossless(samples, &options).expect("encode rgb8");
            black_box(encoded.codestream.len());
        });
    });

    group.finish();
}

fn bench_inspect(c: &mut Criterion) {
    let codestream = encode_rgb8_codestream(TILE_SIDE, TILE_SIDE);

    let mut group = c.benchmark_group("j2k_public_inspect");
    group.bench_function("rgb8_128x128", |b| {
        b.iter(|| {
            let info = J2kDecoder::inspect(black_box(codestream.as_slice())).expect("inspect");
            black_box(info);
        });
    });
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let codestream = encode_rgb8_codestream(TILE_SIDE, TILE_SIDE);
    let mut group = c.benchmark_group("j2k_public_decode");

    let full_stride = TILE_SIDE as usize * 3;
    let mut full = vec![0u8; full_stride * TILE_SIDE as usize];
    group.bench_function("rgb8_full_128x128", |b| {
        b.iter(|| {
            let mut decoder =
                J2kDecoder::new(black_box(codestream.as_slice())).expect("rgb8 decoder");
            decoder
                .decode_into(&mut full, full_stride, PixelFormat::Rgb8)
                .expect("decode full rgb8");
            black_box(&full);
        });
    });

    let roi = Rect {
        x: 32,
        y: 32,
        w: ROI_SIDE,
        h: ROI_SIDE,
    };
    let roi_stride = ROI_SIDE as usize * 3;
    let mut roi_out = vec![0u8; roi_stride * ROI_SIDE as usize];
    let mut pool = J2kScratchPool::new();
    group.bench_function("rgb8_roi_64x64", |b| {
        b.iter(|| {
            let mut decoder =
                J2kDecoder::new(black_box(codestream.as_slice())).expect("rgb8 decoder");
            decoder
                .decode_region_into(&mut pool, &mut roi_out, roi_stride, PixelFormat::Rgb8, roi)
                .expect("decode rgb8 roi");
            black_box(&roi_out);
        });
    });

    group.finish();
}

fn bench_decode_gray_setup(c: &mut Criterion) {
    let codestream = encode_gray8_codestream(TILE_SIDE, TILE_SIDE);
    let stride = TILE_SIDE as usize;
    let mut out = vec![0u8; stride * TILE_SIDE as usize];

    let mut group = c.benchmark_group("j2k_public_decode_gray");
    group.bench_function("gray8_full_128x128", |b| {
        b.iter(|| {
            let mut decoder =
                J2kDecoder::new(black_box(codestream.as_slice())).expect("gray8 decoder");
            decoder
                .decode_into(&mut out, stride, PixelFormat::Gray8)
                .expect("decode full gray8");
            black_box(&out);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_lossless_encode,
    bench_inspect,
    bench_decode,
    bench_decode_gray_setup
);
criterion_main!(benches);
