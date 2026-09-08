// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use j2k_jpeg::bench_support::{
    bench_idct_reference_block, BenchColorRowScratch, BenchGrayRowScratch, BenchHuffmanState,
    BenchRgb420RowPairScratch, BenchRgbRowScratch, BenchUpsampleH2V2Scratch,
};
use j2k_jpeg::Decoder;
use j2k_test_support::JPEG_BASELINE_420_16X16;

#[expect(
    clippy::too_many_lines,
    reason = "this benchmark registry keeps related microbenchmarks in one ordered Criterion group"
)]
fn bench_micro(c: &mut Criterion) {
    let small = JPEG_BASELINE_420_16X16;

    c.bench_function("micro/inspect_small", |b| {
        b.iter(|| {
            let info = Decoder::inspect(small).expect("j2k inspect");
            std::hint::black_box(info);
        });
    });

    let huffman = BenchHuffmanState::luma_dc_zeros(2048);
    assert_eq!(huffman.decode_all().expect("validate zero stream"), 0);
    c.bench_function("micro/huffman_luma_dc_zero_stream", |b| {
        b.iter(|| {
            let sum = huffman.decode_all().expect("huffman decode");
            std::hint::black_box(sum);
        });
    });

    // Includes standard DC table compilation and the constructor's fixed
    // eight-byte padded stream allocation, with no entropy symbols to generate.
    c.bench_function("micro/huffman_luma_dc_state_build", |b| {
        b.iter(|| {
            std::hint::black_box(BenchHuffmanState::luma_dc_zeros(std::hint::black_box(0)));
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
        use j2k_jpeg::bench_support::{
            bench_idct_dc_only_block_with, bench_idct_reference_block_with,
        };
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

        c.bench_function("micro/idct_islow_dc_only_block", |b| {
            let mut out = [0u8; 64];
            b.iter(|| {
                bench_idct_dc_only_block_with(std::hint::black_box(coeffs[0]), &mut out);
                std::hint::black_box(&out);
            });
        });
    }

    {
        use j2k_jpeg::bench_support::bench_idct_reduced_2x2_block_with;
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
        use j2k_jpeg::bench_support::BenchNeonIdct;
        let neon = BenchNeonIdct::new();
        c.bench_function("micro/idct_islow_neon_block", |b| {
            let mut out = [0u8; 64];
            b.iter(|| {
                neon.run(std::hint::black_box(&coeffs), &mut out);
                std::hint::black_box(&out);
            });
        });

        c.bench_function("micro/idct_islow_neon_bottom_half_zero_block", |b| {
            let mut out = [0u8; 64];
            b.iter(|| {
                neon.run_bottom_half_zero(std::hint::black_box(&bottom_half_zero), &mut out);
                std::hint::black_box(&out);
            });
        });
    }

    #[cfg(target_arch = "x86_64")]
    {
        use j2k_jpeg::bench_support::BenchAvx2Idct;
        if let Some(avx2) = BenchAvx2Idct::try_new() {
            c.bench_function("micro/idct_islow_avx2_block", |b| {
                let mut out = [0u8; 64];
                b.iter(|| {
                    avx2.run(std::hint::black_box(&coeffs), &mut out);
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
    assert!(row_pair.backend_matches_reference());
    c.bench_function("micro/rgb_420_row_pair_255", |b| {
        b.iter(|| {
            row_pair.run();
            std::hint::black_box(&row_pair);
        });
    });

    let mut row_pair_even = BenchRgb420RowPairScratch::new(256);
    assert!(row_pair_even.backend_matches_reference());
    c.bench_function("micro/rgb_420_row_pair_256", |b| {
        b.iter(|| {
            row_pair_even.run();
            std::hint::black_box(&row_pair_even);
        });
    });

    let mut row_pair_cropped = BenchRgb420RowPairScratch::new(257);
    assert!(row_pair_cropped.cropped_backend_matches_reference(3, 249));
    c.bench_function("micro/rgb_420_row_pair_cropped_3_249", |b| {
        b.iter(|| {
            row_pair_cropped.run_cropped(3, 249);
            std::hint::black_box(&row_pair_cropped);
        });
    });

    let mut gray = BenchGrayRowScratch::new(256);
    assert!(gray.backend_matches_scalar());
    c.bench_function("micro/gray_to_rgb_row_backend_256", |b| {
        b.iter(|| {
            gray.run_backend();
            std::hint::black_box(&gray);
        });
    });

    let mut rgb = BenchRgbRowScratch::new(256);
    assert!(rgb.backend_matches_scalar());
    c.bench_function("micro/planar_rgb_to_rgb_row_backend_256", |b| {
        b.iter(|| {
            rgb.run_backend();
            std::hint::black_box(&rgb);
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
    assert!(backend_color.backend_matches_scalar());
    c.bench_function("micro/ycbcr_to_rgb_row_backend_256", |b| {
        b.iter(|| {
            backend_color.run_backend();
            std::hint::black_box(&backend_color);
        });
    });

    let mut backend_color_tail = BenchColorRowScratch::new(255);
    assert!(backend_color_tail.backend_matches_scalar());
    c.bench_function("micro/ycbcr_to_rgb_row_backend_255", |b| {
        b.iter(|| {
            backend_color_tail.run_backend();
            std::hint::black_box(&backend_color_tail);
        });
    });

    let mut backend_color_unaligned = BenchColorRowScratch::new_unaligned(256);
    assert!(backend_color_unaligned.backend_matches_scalar());
    c.bench_function("micro/ycbcr_to_rgb_row_backend_unaligned_256", |b| {
        b.iter(|| {
            backend_color_unaligned.run_backend();
            std::hint::black_box(&backend_color_unaligned);
        });
    });
}

criterion_group! {
    name = micro_benches;
    config = Criterion::default()
        .confidence_level(0.95)
        .sample_size(50)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(10));
    targets = bench_micro
}
criterion_main!(micro_benches);
