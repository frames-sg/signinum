// SPDX-License-Identifier: Apache-2.0

use ashlar_core::{BackendRequest, ImageDecodeDevice, PixelFormat};
use ashlar_jpeg::Decoder as CpuDecoder;
use ashlar_jpeg_metal::Decoder as MetalDecoder;
use criterion::{criterion_group, criterion_main, Criterion};

const BASELINE_420: &[u8] = include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");

fn bench_device_upload(c: &mut Criterion) {
    let mut group = c.benchmark_group("jpeg_metal_device");

    group.bench_function("cpu_decode_rgb8", |b| {
        let decoder = CpuDecoder::new(BASELINE_420).expect("cpu decoder");
        b.iter(|| decoder.decode(PixelFormat::Rgb8).expect("cpu decode"));
    });

    group.bench_function("metal_surface_rgb8", |b| {
        let mut decoder = MetalDecoder::new(BASELINE_420).expect("metal decoder");
        b.iter(|| {
            decoder
                .decode_to_device(PixelFormat::Rgb8, BackendRequest::Metal)
                .expect("device decode")
        });
    });

    group.finish();
}

criterion_group!(benches, bench_device_upload);
criterion_main!(benches);
