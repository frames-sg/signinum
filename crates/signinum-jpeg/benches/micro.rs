// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};
use signinum_jpeg::bench_support::{
    bench_idct_reference_block, BenchColorRowScratch, BenchHuffmanState, BenchRgb420RowPairScratch,
    BenchUpsampleH2V2Scratch,
};
use signinum_jpeg::Decoder;

fn bench_micro(c: &mut Criterion) {
    let small = include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");

    c.bench_function("micro/inspect_small", |b| {
        b.iter(|| {
            let info = Decoder::inspect(small).expect("signinum inspect");
            std::hint::black_box(info);
        });
    });

    let huffman = BenchHuffmanState::luma_dc_zeros(2048);
    c.bench_function("micro/huffman_luma_dc_zero_stream", |b| {
        b.iter(|| {
            let sum = huffman.decode_all().expect("huffman decode");
            std::hint::black_box(sum);
        });
    });

    c.bench_function("micro/idct_reference_block", |b| {
        b.iter(|| {
            let out = bench_idct_reference_block();
            std::hint::black_box(out);
        });
    });

    // Scalar-vs-SIMD one-block parity workload on a mid-complexity coefficient
    // block, tracking the Phase 1 speedup ratio precisely.
    let mut coeffs = [0i16; 64];
    coeffs[0] = 480;
    coeffs[1] = -120;
    coeffs[2] = 75;
    coeffs[8] = 92;
    coeffs[9] = -38;
    coeffs[10] = 17;
    coeffs[16] = -22;
    coeffs[17] = 9;
    coeffs[24] = 11;

    let mut bottom_half_zero = [0i16; 64];
    bottom_half_zero[0] = 480;
    bottom_half_zero[1] = -120;
    bottom_half_zero[2] = 75;
    bottom_half_zero[8] = 92;
    bottom_half_zero[9] = -38;
    bottom_half_zero[10] = 17;
    bottom_half_zero[16] = -22;
    bottom_half_zero[17] = 9;
    bottom_half_zero[24] = 11;

    {
        use signinum_jpeg::bench_support::bench_idct_reference_block_with;
        c.bench_function("micro/idct_islow_scalar_block", |b| {
            let mut out = [0u8; 64];
            b.iter(|| {
                bench_idct_reference_block_with(std::hint::black_box(&coeffs), &mut out);
                std::hint::black_box(&out);
            });
        });

        c.bench_function("micro/idct_islow_scalar_bottom_half_zero_block", |b| {
            let mut out = [0u8; 64];
            b.iter(|| {
                bench_idct_reference_block_with(std::hint::black_box(&bottom_half_zero), &mut out);
                std::hint::black_box(&out);
            });
        });
    }

    {
        use signinum_jpeg::bench_support::bench_idct_reduced_2x2_block_with;
        c.bench_function("micro/idct_islow_2x2_scalar_block", |b| {
            let mut out = [0u8; 4];
            b.iter(|| {
                bench_idct_reduced_2x2_block_with(std::hint::black_box(&coeffs), &mut out);
                std::hint::black_box(&out);
            });
        });
    }

    #[cfg(target_arch = "aarch64")]
    {
        use signinum_jpeg::bench_support::bench_idct_neon_block;
        c.bench_function("micro/idct_islow_neon_block", |b| {
            let mut out = [0u8; 64];
            b.iter(|| {
                bench_idct_neon_block(std::hint::black_box(&coeffs), &mut out);
                std::hint::black_box(&out);
            });
        });

        c.bench_function("micro/idct_islow_neon_bottom_half_zero_block", |b| {
            let mut out = [0u8; 64];
            b.iter(|| {
                bench_idct_neon_block(std::hint::black_box(&bottom_half_zero), &mut out);
                std::hint::black_box(&out);
            });
        });
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            use signinum_jpeg::bench_support::bench_idct_avx2_block;
            c.bench_function("micro/idct_islow_avx2_block", |b| {
                let mut out = [0u8; 64];
                b.iter(|| {
                    bench_idct_avx2_block(std::hint::black_box(&coeffs), &mut out);
                    std::hint::black_box(&out);
                });
            });
        }
    }

    // Chroma fancy upsample over two output rows. 128 chroma samples ⇒ 256
    // luma columns per row — typical of a 256-wide WSI tile's 4:2:0 chroma.
    let mut upsample = BenchUpsampleH2V2Scratch::new(128);
    c.bench_function("micro/upsample_h2v2_fancy_rows_128", |b| {
        b.iter(|| {
            upsample.run();
            std::hint::black_box(&upsample);
        });
    });

    // Odd-width 4:2:0 row-pair work item that forces the narrow chroma tail
    // handling exercised by the NEON hot-path parity test.
    let mut row_pair = BenchRgb420RowPairScratch::new(255);
    c.bench_function("micro/rgb_420_row_pair_255", |b| {
        b.iter(|| {
            row_pair.run();
            std::hint::black_box(&row_pair);
        });
    });

    // Scalar YCbCr→RGB conversion across a 256-pixel row — the path every
    // Phase 2 SIMD variant has to beat.
    let mut color = BenchColorRowScratch::new(256);
    c.bench_function("micro/ycbcr_to_rgb_row_scalar_256", |b| {
        b.iter(|| {
            color.run_scalar();
            std::hint::black_box(&color);
        });
    });

    let mut backend_color = BenchColorRowScratch::new(256);
    c.bench_function("micro/ycbcr_to_rgb_row_backend_256", |b| {
        b.iter(|| {
            backend_color.run_backend();
            std::hint::black_box(&backend_color);
        });
    });

    let mut backend_color_tail = BenchColorRowScratch::new(255);
    c.bench_function("micro/ycbcr_to_rgb_row_backend_255", |b| {
        b.iter(|| {
            backend_color_tail.run_backend();
            std::hint::black_box(&backend_color_tail);
        });
    });
}

criterion_group!(micro_benches, bench_micro);
criterion_main!(micro_benches);
