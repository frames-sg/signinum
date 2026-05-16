// SPDX-License-Identifier: Apache-2.0

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use signinum::j2k::{
    encode_j2k_lossless as facade_encode_j2k_lossless, EncodeBackendPreference,
    J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples,
};
use signinum_test_support::patterned_rgb8;

const TILE_SIDE: u32 = 128;

fn bench_encode_options() -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions {
        backend: EncodeBackendPreference::CpuOnly,
        validation: J2kEncodeValidation::External,
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

criterion_group!(benches, bench_facade_j2k_encode);
criterion_main!(benches);
