// SPDX-License-Identifier: Apache-2.0

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use signinum_j2k::{
    encode_j2k_lossless, DecoderContext, Downscale, EncodeBackendPreference, ImageDecodeRows,
    J2kBlockCodingMode, J2kCodec, J2kContext, J2kDecoder, J2kEncodeValidation,
    J2kLosslessEncodeOptions, J2kLosslessSamples, J2kScratchPool, PixelFormat, Rect, RowSink,
    TileBatchDecode,
};
use signinum_test_support::{patterned_gray8, patterned_rgb8};

const TILE_SIDE: u32 = 128;
const ROI_SIDE: u32 = 64;
const HT_TILE_SIDE: u32 = 512;
const BATCH_SIZE: usize = 16;

fn bench_encode_options() -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions {
        backend: EncodeBackendPreference::CpuOnly,
        validation: J2kEncodeValidation::External,
        ..J2kLosslessEncodeOptions::default()
    }
}

fn encode_gray8_codestream(width: u32, height: u32) -> Vec<u8> {
    let pixels = patterned_gray8(width, height);
    encode_gray8_codestream_from_pixels(width, height, &pixels, bench_encode_options())
}

fn encode_ht_gray8_codestream(width: u32, height: u32) -> Vec<u8> {
    let pixels = patterned_gray8(width, height);
    encode_gray8_codestream_from_pixels(
        width,
        height,
        &pixels,
        J2kLosslessEncodeOptions {
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            ..bench_encode_options()
        },
    )
}

fn encode_gray8_codestream_from_pixels(
    width: u32,
    height: u32,
    pixels: &[u8],
    options: J2kLosslessEncodeOptions,
) -> Vec<u8> {
    let samples =
        J2kLosslessSamples::new(pixels, width, height, 1, 8, false).expect("valid gray8 samples");
    encode_j2k_lossless(samples, &options)
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
    let ht_codestream = encode_ht_gray8_codestream(HT_TILE_SIDE, HT_TILE_SIDE);
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

    let ht_stride = HT_TILE_SIDE as usize;
    let mut ht_out = vec![0u8; ht_stride * HT_TILE_SIDE as usize];
    group.bench_function("htj2k_gray8_full_512x512", |b| {
        b.iter(|| {
            let mut decoder =
                J2kDecoder::new(black_box(ht_codestream.as_slice())).expect("htj2k decoder");
            decoder
                .decode_into(&mut ht_out, ht_stride, PixelFormat::Gray8)
                .expect("decode full htj2k gray8");
            black_box(&ht_out);
        });
    });

    group.finish();
}

fn bench_region_scaled(c: &mut Criterion) {
    let codestream = encode_rgb8_codestream(TILE_SIDE, TILE_SIDE);
    let roi = Rect {
        x: 32,
        y: 32,
        w: ROI_SIDE,
        h: ROI_SIDE,
    };
    let out_side = ROI_SIDE.div_ceil(Downscale::Quarter.denominator());
    let stride = out_side as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut out = vec![0u8; stride * out_side as usize];
    let mut pool = J2kScratchPool::new();

    let mut group = c.benchmark_group("j2k_public_decode_region_scaled");
    group.bench_function("rgb8_region_scaled_64x64_q4", |b| {
        b.iter(|| {
            let mut decoder =
                J2kDecoder::new(black_box(codestream.as_slice())).expect("rgb8 decoder");
            decoder
                .decode_region_scaled_into(
                    &mut pool,
                    &mut out,
                    stride,
                    PixelFormat::Rgb8,
                    roi,
                    Downscale::Quarter,
                )
                .expect("decode rgb8 region scaled");
            black_box(&out);
        });
    });
    group.finish();
}

fn bench_rows(c: &mut Criterion) {
    let codestream = encode_gray8_codestream(TILE_SIDE, TILE_SIDE);
    let mut group = c.benchmark_group("j2k_public_decode_rows");
    group.bench_function("gray8_rows_128x128", |b| {
        b.iter(|| {
            let mut decoder =
                J2kDecoder::new(black_box(codestream.as_slice())).expect("gray8 decoder");
            let mut sink = VecRowSink::new(TILE_SIDE, TILE_SIDE);
            decoder.decode_rows(&mut sink).expect("decode gray8 rows");
            black_box(sink.rows);
        });
    });
    group.finish();
}

fn bench_tile_batch(c: &mut Criterion) {
    let repeated = encode_gray8_codestream(TILE_SIDE, TILE_SIDE);
    let mut distinct = Vec::with_capacity(BATCH_SIZE);
    for idx in 0..BATCH_SIZE {
        let mut pixels = patterned_gray8(TILE_SIDE, TILE_SIDE);
        pixels[0] = pixels[0].wrapping_add(idx as u8);
        distinct.push(encode_gray8_codestream_from_pixels(
            TILE_SIDE,
            TILE_SIDE,
            &pixels,
            bench_encode_options(),
        ));
    }

    let stride = TILE_SIDE as usize;
    let mut out = vec![0u8; stride * TILE_SIDE as usize];
    let mut group = c.benchmark_group("j2k_public_tile_batch");

    group.bench_function("gray8_repeated_batch_16", |b| {
        b.iter(|| {
            let mut ctx = DecoderContext::<J2kContext>::default();
            let mut pool = J2kScratchPool::new();
            for _ in 0..BATCH_SIZE {
                <J2kCodec as TileBatchDecode>::decode_tile(
                    &mut ctx,
                    &mut pool,
                    black_box(repeated.as_slice()),
                    &mut out,
                    stride,
                    PixelFormat::Gray8,
                )
                .expect("decode repeated gray8 tile");
            }
            black_box(&out);
        });
    });

    group.bench_function("gray8_distinct_batch_16", |b| {
        b.iter(|| {
            let mut ctx = DecoderContext::<J2kContext>::default();
            let mut pool = J2kScratchPool::new();
            for codestream in &distinct {
                <J2kCodec as TileBatchDecode>::decode_tile(
                    &mut ctx,
                    &mut pool,
                    black_box(codestream.as_slice()),
                    &mut out,
                    stride,
                    PixelFormat::Gray8,
                )
                .expect("decode distinct gray8 tile");
            }
            black_box(&out);
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
    bench_region_scaled,
    bench_rows,
    bench_tile_batch,
    bench_decode_gray_setup
);
criterion_main!(benches);

struct VecRowSink {
    rows: Vec<u8>,
    width: usize,
}

impl VecRowSink {
    fn new(width: u32, height: u32) -> Self {
        Self {
            rows: vec![0; width as usize * height as usize],
            width: width as usize,
        }
    }
}

impl RowSink<u8> for VecRowSink {
    type Error = std::convert::Infallible;

    fn write_row(&mut self, y: u32, row: &[u8]) -> Result<(), Self::Error> {
        let start = y as usize * self.width;
        let end = start + row.len();
        self.rows[start..end].copy_from_slice(row);
        Ok(())
    }
}
