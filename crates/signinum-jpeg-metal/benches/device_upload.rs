// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};
use jpeg_encoder::{ColorType, Encoder};
use signinum_core::{BackendRequest, ImageDecodeDevice, PixelFormat};
use signinum_jpeg::Decoder as CpuDecoder;
use signinum_jpeg_metal::Decoder as MetalDecoder;

const BASELINE_420: &[u8] = include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg");
const DEFAULT_GENERATED_DIM: u16 = 2048;

fn bench_device_upload(c: &mut Criterion) {
    let input = bench_input();
    let mut group = c.benchmark_group("jpeg_metal_device");

    group.bench_function("cpu_decode_rgb8", |b| {
        let decoder = CpuDecoder::new(&input).expect("cpu decoder");
        b.iter(|| decoder.decode(PixelFormat::Rgb8).expect("cpu decode"));
    });

    group.bench_function("metal_surface_rgb8", |b| {
        b.iter(|| {
            let mut decoder = MetalDecoder::new(&input).expect("metal decoder");
            decoder
                .decode_to_device(PixelFormat::Rgb8, BackendRequest::Metal)
                .expect("device decode")
        });
    });

    group.finish();
}

fn bench_input() -> Vec<u8> {
    match std::env::var_os("SIGNINUM_GPU_BENCH_JPEG") {
        Some(path) => std::fs::read(&path).unwrap_or_else(|error| {
            panic!(
                "failed to read SIGNINUM_GPU_BENCH_JPEG={}: {error}",
                path.to_string_lossy()
            )
        }),
        None if std::env::var_os("SIGNINUM_GPU_BENCH_SMALL_FIXTURE").is_some() => {
            BASELINE_420.to_vec()
        }
        None => generated_jpeg(),
    }
}

fn generated_jpeg() -> Vec<u8> {
    let dim = generated_dim();
    let mut rgb = Vec::with_capacity(dim as usize * dim as usize * 3);
    for y in 0..dim {
        for x in 0..dim {
            let xf = u32::from(x);
            let yf = u32::from(y);
            rgb.push(((xf * 13 + yf * 3) & 0xff) as u8);
            rgb.push(((xf * 5 + yf * 11 + (xf ^ yf)) & 0xff) as u8);
            rgb.push(((xf * 7 + yf * 17 + (xf.wrapping_mul(yf) >> 5)) & 0xff) as u8);
        }
    }

    let mut jpeg = Vec::new();
    Encoder::new(&mut jpeg, 90)
        .encode(&rgb, dim, dim, ColorType::Rgb)
        .expect("encode generated benchmark JPEG");
    jpeg
}

fn generated_dim() -> u16 {
    let Some(value) = std::env::var_os("SIGNINUM_GPU_BENCH_DIM") else {
        return DEFAULT_GENERATED_DIM;
    };
    let value = value
        .to_string_lossy()
        .parse::<u16>()
        .expect("SIGNINUM_GPU_BENCH_DIM must be a u16");
    assert!(
        (256..=8192).contains(&value),
        "SIGNINUM_GPU_BENCH_DIM must be between 256 and 8192"
    );
    value
}

criterion_group!(benches, bench_device_upload);
criterion_main!(benches);
