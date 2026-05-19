// SPDX-License-Identifier: Apache-2.0

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use signinum_j2k::{
    decode_tiles_region_scaled_into, encode_j2k_lossless, CpuDecodeParallelism, DecoderContext,
    Downscale, EncodeBackendPreference, ImageDecodeRows, J2kBlockCodingMode, J2kCodec, J2kContext,
    J2kDecoder, J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples, J2kScratchPool,
    PixelFormat, Rect, RowSink, TileBatchDecode, TileBatchOptions, TileRegionScaledDecodeJob,
};
use signinum_test_support::{patterned_gray8, patterned_rgb8};

const TILE_SIDE: u32 = 128;
const ROI_SIDE: u32 = 64;
const HT_TILE_SIDE: u32 = 512;
const CPU_MATRIX_SIDE: u32 = 512;
const BATCH_SIZE: usize = 16;

fn bench_encode_options() -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions {
        backend: EncodeBackendPreference::CpuOnly,
        validation: J2kEncodeValidation::External,
        ..J2kLosslessEncodeOptions::default()
    }
}

fn ht_encode_options() -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions {
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        ..bench_encode_options()
    }
}

fn cpu_matrix_encode_options(
    block_coding_mode: J2kBlockCodingMode,
    validation: J2kEncodeValidation,
) -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions {
        backend: EncodeBackendPreference::CpuOnly,
        validation,
        block_coding_mode,
        ..J2kLosslessEncodeOptions::default()
    }
}

fn encode_gray8_codestream(width: u32, height: u32) -> Vec<u8> {
    let pixels = patterned_gray8(width, height);
    encode_gray8_codestream_from_pixels(width, height, &pixels, bench_encode_options())
}

fn encode_ht_gray8_codestream(width: u32, height: u32) -> Vec<u8> {
    let pixels = patterned_gray8(width, height);
    encode_gray8_codestream_from_pixels(width, height, &pixels, ht_encode_options())
}

fn encode_ht_rgb8_codestream(width: u32, height: u32) -> Vec<u8> {
    let pixels = patterned_rgb8(width, height);
    encode_rgb8_codestream_from_pixels(width, height, &pixels, ht_encode_options())
}

fn wrap_codestream_jp2(
    codestream: &[u8],
    width: u32,
    height: u32,
    components: u16,
    bit_depth: u8,
    colorspace_enum: u32,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0D, 0x0A, 0x87, 0x0A]);
    bytes.extend_from_slice(&[
        0, 0, 0, 20, b'f', b't', b'y', b'p', b'j', b'p', b'2', b' ', 0, 0, 0, 0, b'j', b'p', b'2',
        b' ',
    ]);

    let bpc = bit_depth.saturating_sub(1);
    bytes.extend_from_slice(&[
        0, 0, 0, 45, b'j', b'p', b'2', b'h', 0, 0, 0, 22, b'i', b'h', b'd', b'r',
    ]);
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&components.to_be_bytes());
    bytes.extend_from_slice(&[bpc, 7, 0, 0]);
    bytes.extend_from_slice(&[0, 0, 0, 15, b'c', b'o', b'l', b'r', 1, 0, 0]);
    bytes.extend_from_slice(&colorspace_enum.to_be_bytes());

    let len = (8 + codestream.len()) as u32;
    bytes.extend_from_slice(&len.to_be_bytes());
    bytes.extend_from_slice(b"jp2c");
    bytes.extend_from_slice(codestream);
    bytes
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
    encode_rgb8_codestream_from_pixels(width, height, &pixels, bench_encode_options())
}

fn encode_rgb8_codestream_from_pixels(
    width: u32,
    height: u32,
    pixels: &[u8],
    options: J2kLosslessEncodeOptions,
) -> Vec<u8> {
    let samples =
        J2kLosslessSamples::new(pixels, width, height, 3, 8, false).expect("valid rgb8 samples");
    encode_j2k_lossless(samples, &options)
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
    let ht_repeated = encode_ht_gray8_codestream(TILE_SIDE, TILE_SIDE);
    let mut distinct = Vec::with_capacity(BATCH_SIZE);
    let mut ht_distinct = Vec::with_capacity(BATCH_SIZE);
    for idx in 0..BATCH_SIZE {
        let mut pixels = patterned_gray8(TILE_SIDE, TILE_SIDE);
        pixels[0] = pixels[0].wrapping_add(idx as u8);
        distinct.push(encode_gray8_codestream_from_pixels(
            TILE_SIDE,
            TILE_SIDE,
            &pixels,
            bench_encode_options(),
        ));
        ht_distinct.push(encode_gray8_codestream_from_pixels(
            TILE_SIDE,
            TILE_SIDE,
            &pixels,
            ht_encode_options(),
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

    group.bench_function("htj2k_gray8_repeated_batch_16", |b| {
        b.iter(|| {
            let mut ctx = DecoderContext::<J2kContext>::default();
            let mut pool = J2kScratchPool::new();
            for _ in 0..BATCH_SIZE {
                <J2kCodec as TileBatchDecode>::decode_tile(
                    &mut ctx,
                    &mut pool,
                    black_box(ht_repeated.as_slice()),
                    &mut out,
                    stride,
                    PixelFormat::Gray8,
                )
                .expect("decode repeated htj2k gray8 tile");
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

    group.bench_function("htj2k_gray8_distinct_batch_16", |b| {
        b.iter(|| {
            let mut ctx = DecoderContext::<J2kContext>::default();
            let mut pool = J2kScratchPool::new();
            for codestream in &ht_distinct {
                <J2kCodec as TileBatchDecode>::decode_tile(
                    &mut ctx,
                    &mut pool,
                    black_box(codestream.as_slice()),
                    &mut out,
                    stride,
                    PixelFormat::Gray8,
                )
                .expect("decode distinct htj2k gray8 tile");
            }
            black_box(&out);
        });
    });

    group.finish();
}

fn bench_tile_batch_region_scaled_rgb(c: &mut Criterion) {
    let repeated_classic = encode_rgb8_codestream(CPU_MATRIX_SIDE, CPU_MATRIX_SIDE);
    let repeated_htj2k = encode_ht_rgb8_codestream(CPU_MATRIX_SIDE, CPU_MATRIX_SIDE);
    let repeated_htj2k_jp2 =
        wrap_codestream_jp2(&repeated_htj2k, CPU_MATRIX_SIDE, CPU_MATRIX_SIDE, 3, 8, 16);
    let repeated_htj2k_256 = encode_ht_rgb8_codestream(256, 256);
    let repeated_htj2k_256_jp2 = wrap_codestream_jp2(&repeated_htj2k_256, 256, 256, 3, 8, 16);
    let mut distinct_classic = Vec::with_capacity(BATCH_SIZE);
    let mut distinct_htj2k = Vec::with_capacity(BATCH_SIZE);
    for idx in 0..BATCH_SIZE {
        let mut pixels = patterned_rgb8(CPU_MATRIX_SIDE, CPU_MATRIX_SIDE);
        pixels[0] = pixels[0].wrapping_add(idx as u8);
        distinct_classic.push(encode_rgb8_codestream_from_pixels(
            CPU_MATRIX_SIDE,
            CPU_MATRIX_SIDE,
            &pixels,
            bench_encode_options(),
        ));
        distinct_htj2k.push(encode_rgb8_codestream_from_pixels(
            CPU_MATRIX_SIDE,
            CPU_MATRIX_SIDE,
            &pixels,
            ht_encode_options(),
        ));
    }

    let roi = Rect {
        x: 128,
        y: 128,
        w: 256,
        h: 256,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let stride = scaled.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let output_len = stride * scaled.h as usize;
    let rgba_stride = scaled.w as usize * PixelFormat::Rgba8.bytes_per_pixel();
    let rgba_output_len = rgba_stride * scaled.h as usize;

    let roi_256 = Rect {
        x: 64,
        y: 64,
        w: 128,
        h: 128,
    };
    let scaled_256 = roi_256.scaled_covering(scale);
    let stride_256 = scaled_256.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let output_len_256 = stride_256 * scaled_256.h as usize;

    let mut group = c.benchmark_group("j2k_public_tile_batch_region_scaled_rgb_q4");
    group.bench_function("classic_repeated_512_roi256_batch16", |b| {
        b.iter(|| {
            let mut outputs = vec![vec![0_u8; output_len]; BATCH_SIZE];
            let mut jobs = outputs
                .iter_mut()
                .map(|out| TileRegionScaledDecodeJob {
                    input: black_box(repeated_classic.as_slice()),
                    out,
                    stride,
                    roi,
                    scale,
                })
                .collect::<Vec<_>>();
            let outcomes = decode_tiles_region_scaled_into(
                &mut jobs,
                PixelFormat::Rgb8,
                TileBatchOptions::default(),
            )
            .expect("decode repeated classic RGB ROI+scale batch");
            black_box((outputs, outcomes));
        });
    });
    group.bench_function("classic_distinct_512_roi256_batch16", |b| {
        b.iter(|| {
            let mut outputs = vec![vec![0_u8; output_len]; BATCH_SIZE];
            let mut jobs = outputs
                .iter_mut()
                .zip(distinct_classic.iter())
                .map(|(out, input)| TileRegionScaledDecodeJob {
                    input: black_box(input.as_slice()),
                    out,
                    stride,
                    roi,
                    scale,
                })
                .collect::<Vec<_>>();
            let outcomes = decode_tiles_region_scaled_into(
                &mut jobs,
                PixelFormat::Rgb8,
                TileBatchOptions::default(),
            )
            .expect("decode distinct classic RGB ROI+scale batch");
            black_box((outputs, outcomes));
        });
    });
    group.bench_function("htj2k_repeated_512_roi256_batch16", |b| {
        b.iter(|| {
            let mut outputs = vec![vec![0_u8; output_len]; BATCH_SIZE];
            let mut jobs = outputs
                .iter_mut()
                .map(|out| TileRegionScaledDecodeJob {
                    input: black_box(repeated_htj2k.as_slice()),
                    out,
                    stride,
                    roi,
                    scale,
                })
                .collect::<Vec<_>>();
            let outcomes = decode_tiles_region_scaled_into(
                &mut jobs,
                PixelFormat::Rgb8,
                TileBatchOptions::default(),
            )
            .expect("decode repeated HTJ2K RGB ROI+scale batch");
            black_box((outputs, outcomes));
        });
    });
    group.bench_function("htj2k_jp2_rgb8_repeated_512_roi256_batch16", |b| {
        b.iter(|| {
            let mut outputs = vec![vec![0_u8; output_len]; BATCH_SIZE];
            let mut jobs = outputs
                .iter_mut()
                .map(|out| TileRegionScaledDecodeJob {
                    input: black_box(repeated_htj2k_jp2.as_slice()),
                    out,
                    stride,
                    roi,
                    scale,
                })
                .collect::<Vec<_>>();
            let outcomes = decode_tiles_region_scaled_into(
                &mut jobs,
                PixelFormat::Rgb8,
                TileBatchOptions::default(),
            )
            .expect("decode repeated HTJ2K JP2 RGB ROI+scale batch");
            black_box((outputs, outcomes));
        });
    });
    group.bench_function("htj2k_jp2_rgba8_repeated_512_roi256_batch16", |b| {
        b.iter(|| {
            let mut outputs = vec![vec![0_u8; rgba_output_len]; BATCH_SIZE];
            let mut jobs = outputs
                .iter_mut()
                .map(|out| TileRegionScaledDecodeJob {
                    input: black_box(repeated_htj2k_jp2.as_slice()),
                    out,
                    stride: rgba_stride,
                    roi,
                    scale,
                })
                .collect::<Vec<_>>();
            let outcomes = decode_tiles_region_scaled_into(
                &mut jobs,
                PixelFormat::Rgba8,
                TileBatchOptions::default(),
            )
            .expect("decode repeated HTJ2K JP2 RGBA ROI+scale batch");
            black_box((outputs, outcomes));
        });
    });
    group.bench_function("htj2k_jp2_rgb8_repeated_256_roi128_batch16", |b| {
        b.iter(|| {
            let mut outputs = vec![vec![0_u8; output_len_256]; BATCH_SIZE];
            let mut jobs = outputs
                .iter_mut()
                .map(|out| TileRegionScaledDecodeJob {
                    input: black_box(repeated_htj2k_256_jp2.as_slice()),
                    out,
                    stride: stride_256,
                    roi: roi_256,
                    scale,
                })
                .collect::<Vec<_>>();
            let outcomes = decode_tiles_region_scaled_into(
                &mut jobs,
                PixelFormat::Rgb8,
                TileBatchOptions::default(),
            )
            .expect("decode repeated 256 HTJ2K JP2 RGB ROI+scale batch");
            black_box((outputs, outcomes));
        });
    });
    group.bench_function("htj2k_distinct_512_roi256_batch16", |b| {
        b.iter(|| {
            let mut outputs = vec![vec![0_u8; output_len]; BATCH_SIZE];
            let mut jobs = outputs
                .iter_mut()
                .zip(distinct_htj2k.iter())
                .map(|(out, input)| TileRegionScaledDecodeJob {
                    input: black_box(input.as_slice()),
                    out,
                    stride,
                    roi,
                    scale,
                })
                .collect::<Vec<_>>();
            let outcomes = decode_tiles_region_scaled_into(
                &mut jobs,
                PixelFormat::Rgb8,
                TileBatchOptions::default(),
            )
            .expect("decode distinct HTJ2K RGB ROI+scale batch");
            black_box((outputs, outcomes));
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

fn bench_cpu_encode_matrix(c: &mut Criterion) {
    let pixels = patterned_rgb8(CPU_MATRIX_SIDE, CPU_MATRIX_SIDE);
    let classic_external =
        cpu_matrix_encode_options(J2kBlockCodingMode::Classic, J2kEncodeValidation::External);
    let htj2k_external = cpu_matrix_encode_options(
        J2kBlockCodingMode::HighThroughput,
        J2kEncodeValidation::External,
    );
    let classic_roundtrip = cpu_matrix_encode_options(
        J2kBlockCodingMode::Classic,
        J2kEncodeValidation::CpuRoundTrip,
    );

    let mut group = c.benchmark_group("j2k_public_cpu_encode_matrix");
    group.bench_function("rgb8_512_classic_external", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                black_box(pixels.as_slice()),
                CPU_MATRIX_SIDE,
                CPU_MATRIX_SIDE,
                3,
                8,
                false,
            )
            .expect("valid rgb8 samples");
            let encoded =
                encode_j2k_lossless(samples, &classic_external).expect("classic CPU encode");
            black_box(encoded.codestream.len());
        });
    });

    group.bench_function("rgb8_512_htj2k_external", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                black_box(pixels.as_slice()),
                CPU_MATRIX_SIDE,
                CPU_MATRIX_SIDE,
                3,
                8,
                false,
            )
            .expect("valid rgb8 samples");
            let encoded = encode_j2k_lossless(samples, &htj2k_external).expect("HTJ2K CPU encode");
            black_box(encoded.codestream.len());
        });
    });

    group.bench_function("rgb8_512_classic_roundtrip", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                black_box(pixels.as_slice()),
                CPU_MATRIX_SIDE,
                CPU_MATRIX_SIDE,
                3,
                8,
                false,
            )
            .expect("valid rgb8 samples");
            let encoded =
                encode_j2k_lossless(samples, &classic_roundtrip).expect("classic CPU encode");
            black_box(encoded.codestream.len());
        });
    });
    group.finish();
}

fn bench_cpu_decode_matrix(c: &mut Criterion) {
    let pixels = patterned_gray8(CPU_MATRIX_SIDE, CPU_MATRIX_SIDE);
    let classic_codestream = encode_gray8_codestream_from_pixels(
        CPU_MATRIX_SIDE,
        CPU_MATRIX_SIDE,
        &pixels,
        cpu_matrix_encode_options(J2kBlockCodingMode::Classic, J2kEncodeValidation::External),
    );
    let htj2k_codestream = encode_gray8_codestream_from_pixels(
        CPU_MATRIX_SIDE,
        CPU_MATRIX_SIDE,
        &pixels,
        cpu_matrix_encode_options(
            J2kBlockCodingMode::HighThroughput,
            J2kEncodeValidation::External,
        ),
    );
    let rgb_classic_codestream = encode_rgb8_codestream(CPU_MATRIX_SIDE, CPU_MATRIX_SIDE);
    let rgb_htj2k_codestream = encode_ht_rgb8_codestream(CPU_MATRIX_SIDE, CPU_MATRIX_SIDE);

    let stride = CPU_MATRIX_SIDE as usize;
    let mut classic_out = vec![0u8; stride * CPU_MATRIX_SIDE as usize];
    let mut htj2k_out = vec![0u8; stride * CPU_MATRIX_SIDE as usize];
    let rgb_stride = CPU_MATRIX_SIDE as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut rgb_classic_out = vec![0u8; rgb_stride * CPU_MATRIX_SIDE as usize];
    let mut rgb_htj2k_out = vec![0u8; rgb_stride * CPU_MATRIX_SIDE as usize];

    let mut group = c.benchmark_group("j2k_public_cpu_decode_matrix");
    group.bench_function("gray8_512_classic_decode", |b| {
        b.iter(|| {
            let mut decoder =
                J2kDecoder::new(black_box(classic_codestream.as_slice())).expect("J2K decoder");
            decoder
                .decode_into(&mut classic_out, stride, PixelFormat::Gray8)
                .expect("decode classic gray8");
            black_box(&classic_out);
        });
    });

    group.bench_function("gray8_512_htj2k_decode", |b| {
        b.iter(|| {
            let mut decoder =
                J2kDecoder::new(black_box(htj2k_codestream.as_slice())).expect("HTJ2K decoder");
            decoder
                .decode_into(&mut htj2k_out, stride, PixelFormat::Gray8)
                .expect("decode htj2k gray8");
            black_box(&htj2k_out);
        });
    });

    group.bench_function("rgb8_512_classic_decode", |b| {
        b.iter(|| {
            let mut decoder =
                J2kDecoder::new(black_box(rgb_classic_codestream.as_slice())).expect("J2K decoder");
            decoder
                .decode_into(&mut rgb_classic_out, rgb_stride, PixelFormat::Rgb8)
                .expect("decode classic rgb8");
            black_box(&rgb_classic_out);
        });
    });

    group.bench_function("rgb8_512_classic_decode_serial", |b| {
        b.iter(|| {
            let mut decoder =
                J2kDecoder::new(black_box(rgb_classic_codestream.as_slice())).expect("J2K decoder");
            decoder.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
            decoder
                .decode_into(&mut rgb_classic_out, rgb_stride, PixelFormat::Rgb8)
                .expect("decode serial classic rgb8");
            black_box(&rgb_classic_out);
        });
    });

    group.bench_function("rgb8_512_htj2k_decode", |b| {
        b.iter(|| {
            let mut decoder =
                J2kDecoder::new(black_box(rgb_htj2k_codestream.as_slice())).expect("HTJ2K decoder");
            decoder
                .decode_into(&mut rgb_htj2k_out, rgb_stride, PixelFormat::Rgb8)
                .expect("decode htj2k rgb8");
            black_box(&rgb_htj2k_out);
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
    bench_tile_batch_region_scaled_rgb,
    bench_decode_gray_setup,
    bench_cpu_encode_matrix,
    bench_cpu_decode_matrix
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
