// SPDX-License-Identifier: Apache-2.0

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use signinum::j2k::{
    encode_j2k_lossless as facade_encode_j2k_lossless, EncodeBackendPreference, J2kBlockCodingMode,
    J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples,
};
#[cfg(feature = "metal")]
use signinum::j2k::{encode_j2k_lossless_with_accelerator, BackendKind};
use signinum_test_support::patterned_rgb8;

const TILE_SIDE: u32 = 128;
const MATRIX_SIDE: u32 = 512;

fn bench_encode_options() -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions {
        backend: EncodeBackendPreference::CpuOnly,
        validation: J2kEncodeValidation::External,
        ..J2kLosslessEncodeOptions::default()
    }
}

fn matrix_encode_options(
    backend: EncodeBackendPreference,
    block_coding_mode: J2kBlockCodingMode,
) -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions {
        backend,
        validation: J2kEncodeValidation::External,
        block_coding_mode,
        ..J2kLosslessEncodeOptions::default()
    }
}

fn bench_facade_j2k_encode(c: &mut Criterion) {
    let pixels = patterned_rgb8(TILE_SIDE, TILE_SIDE);
    let options = bench_encode_options();

    let mut group = c.benchmark_group("facade_j2k_lossless_encode");
    group.bench_function("facade_cpu_only_rgb8_128x128", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                black_box(pixels.as_slice()),
                TILE_SIDE,
                TILE_SIDE,
                3,
                8,
                false,
            )
            .expect("valid rgb8 samples");
            let encoded =
                facade_encode_j2k_lossless(samples, &options).expect("facade cpu-only encode");
            black_box(encoded.codestream.len());
        });
    });

    group.bench_function("direct_cpu_only_rgb8_128x128", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                black_box(pixels.as_slice()),
                TILE_SIDE,
                TILE_SIDE,
                3,
                8,
                false,
            )
            .expect("valid rgb8 samples");
            let encoded =
                signinum_j2k::encode_j2k_lossless(samples, &options).expect("direct cpu encode");
            black_box(encoded.codestream.len());
        });
    });
    group.finish();
}

fn bench_facade_cpu_matrix(c: &mut Criterion) {
    let pixels = patterned_rgb8(MATRIX_SIDE, MATRIX_SIDE);
    let classic_options = matrix_encode_options(
        EncodeBackendPreference::CpuOnly,
        J2kBlockCodingMode::Classic,
    );
    let htj2k_options = matrix_encode_options(
        EncodeBackendPreference::CpuOnly,
        J2kBlockCodingMode::HighThroughput,
    );

    let mut group = c.benchmark_group("facade_j2k_lossless_encode_cpu_matrix");
    group.bench_function("cpu_only_rgb8_512_classic_external", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                black_box(pixels.as_slice()),
                MATRIX_SIDE,
                MATRIX_SIDE,
                3,
                8,
                false,
            )
            .expect("valid rgb8 samples");
            let encoded =
                facade_encode_j2k_lossless(samples, &classic_options).expect("classic CPU encode");
            black_box(encoded.codestream.len());
        });
    });

    group.bench_function("cpu_only_rgb8_512_htj2k_external", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                black_box(pixels.as_slice()),
                MATRIX_SIDE,
                MATRIX_SIDE,
                3,
                8,
                false,
            )
            .expect("valid rgb8 samples");
            let encoded =
                facade_encode_j2k_lossless(samples, &htj2k_options).expect("HTJ2K CPU encode");
            black_box(encoded.codestream.len());
        });
    });
    group.finish();
}

fn bench_facade_hybrid_matrix(c: &mut Criterion) {
    let pixels = patterned_rgb8(MATRIX_SIDE, MATRIX_SIDE);
    let auto_classic =
        matrix_encode_options(EncodeBackendPreference::Auto, J2kBlockCodingMode::Classic);
    let auto_htj2k = matrix_encode_options(
        EncodeBackendPreference::Auto,
        J2kBlockCodingMode::HighThroughput,
    );

    let mut group = c.benchmark_group("facade_j2k_lossless_encode_hybrid_matrix");
    group.bench_function("facade_auto_rgb8_512_classic_external", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                black_box(pixels.as_slice()),
                MATRIX_SIDE,
                MATRIX_SIDE,
                3,
                8,
                false,
            )
            .expect("valid rgb8 samples");
            let encoded =
                facade_encode_j2k_lossless(samples, &auto_classic).expect("facade Auto encode");
            black_box((encoded.backend, encoded.codestream.len()));
        });
    });

    group.bench_function("facade_auto_rgb8_512_htj2k_external", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                black_box(pixels.as_slice()),
                MATRIX_SIDE,
                MATRIX_SIDE,
                3,
                8,
                false,
            )
            .expect("valid rgb8 samples");
            let encoded =
                facade_encode_j2k_lossless(samples, &auto_htj2k).expect("facade Auto HTJ2K encode");
            black_box((encoded.backend, encoded.codestream.len()));
        });
    });

    #[cfg(feature = "metal")]
    {
        group.bench_function("direct_metal_auto_stage_rgb8_512_classic_external", |b| {
            b.iter(|| {
                let samples = J2kLosslessSamples::new(
                    black_box(pixels.as_slice()),
                    MATRIX_SIDE,
                    MATRIX_SIDE,
                    3,
                    8,
                    false,
                )
                .expect("valid rgb8 samples");
                let mut accelerator =
                    signinum::j2k::metal::MetalEncodeStageAccelerator::for_auto_host_output();
                let encoded = encode_j2k_lossless_with_accelerator(
                    samples,
                    &auto_classic,
                    BackendKind::Metal,
                    &mut accelerator,
                )
                .expect("direct Metal-stage classic encode");
                black_box((encoded.backend, encoded.codestream.len()));
            });
        });

        group.bench_function("direct_metal_cpu_rct_stage_rgb8_512_htj2k_external", |b| {
            b.iter(|| {
                let samples = J2kLosslessSamples::new(
                    black_box(pixels.as_slice()),
                    MATRIX_SIDE,
                    MATRIX_SIDE,
                    3,
                    8,
                    false,
                )
                .expect("valid rgb8 samples");
                let mut accelerator =
                    signinum::j2k::metal::MetalEncodeStageAccelerator::with_cpu_forward_rct();
                let encoded = encode_j2k_lossless_with_accelerator(
                    samples,
                    &auto_htj2k,
                    BackendKind::Metal,
                    &mut accelerator,
                )
                .expect("direct Metal-stage HTJ2K encode");
                black_box((encoded.backend, encoded.codestream.len()));
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_facade_j2k_encode,
    bench_facade_cpu_matrix,
    bench_facade_hybrid_matrix
);
criterion_main!(benches);
