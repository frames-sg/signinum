// SPDX-License-Identifier: Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names
)]

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use signinum_core::PixelFormat;
use signinum_jpeg::{
    encode_jpeg_baseline, JpegBackend, JpegEncodeOptions, JpegSamples, JpegSubsampling,
};
use signinum_jpeg_metal::{
    encode_jpeg_baseline_batch_from_metal_buffers, encode_jpeg_baseline_from_metal_buffer,
    JpegBaselineMetalEncodeTile, MetalBackendSession,
};
use std::time::Duration;

const DEFAULT_DIM: u32 = 512;
const DEFAULT_BATCH_SIZE: usize = 8;
const DEFAULT_QUALITY: u8 = 90;

fn bench_encode_baseline(c: &mut Criterion) {
    let dim = bench_dim();
    let batch_size = bench_batch_size();
    let tile_bytes = dim as usize * dim as usize * 3;
    let rgb = generated_rgb_tiles(dim, batch_size);
    let cpu_options = options(JpegBackend::Cpu);
    let metal_options = options(JpegBackend::Metal);

    let mut single = c.benchmark_group("jpeg_baseline_encode_single");
    single.sample_size(10);
    single.warm_up_time(Duration::from_secs(1));
    single.measurement_time(Duration::from_secs(3));
    single.throughput(Throughput::Bytes(tile_bytes as u64));
    single.bench_function(format!("cpu_rgb8_422_{dim}x{dim}"), |b| {
        let tile = &rgb[..tile_bytes];
        b.iter(|| {
            encode_jpeg_baseline(
                JpegSamples::Rgb8 {
                    data: std::hint::black_box(tile),
                    width: dim,
                    height: dim,
                },
                cpu_options,
            )
            .expect("CPU JPEG Baseline encode")
        });
    });

    #[cfg(target_os = "macos")]
    if let Ok(session) = MetalBackendSession::system_default() {
        let buffer = session.device().new_buffer_with_data(
            rgb.as_ptr().cast(),
            tile_bytes as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let tile = JpegBaselineMetalEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: dim,
            height: dim,
            pitch_bytes: dim as usize * 3,
            output_width: dim,
            output_height: dim,
            format: PixelFormat::Rgb8,
        };
        single.bench_function(format!("metal_rgb8_422_{dim}x{dim}"), |b| {
            b.iter(|| {
                encode_jpeg_baseline_from_metal_buffer(
                    std::hint::black_box(tile),
                    metal_options,
                    &session,
                )
                .expect("Metal JPEG Baseline encode")
            });
        });
    }
    single.finish();

    let mut batch = c.benchmark_group("jpeg_baseline_encode_batch");
    batch.sample_size(10);
    batch.warm_up_time(Duration::from_secs(1));
    batch.measurement_time(Duration::from_secs(3));
    batch.throughput(Throughput::Bytes((tile_bytes * batch_size) as u64));
    batch.bench_function(format!("cpu_rgb8_422_{dim}x{dim}_batch{batch_size}"), |b| {
        b.iter(|| {
            let mut total = 0usize;
            for tile in rgb.chunks_exact(tile_bytes).take(batch_size) {
                let encoded = encode_jpeg_baseline(
                    JpegSamples::Rgb8 {
                        data: std::hint::black_box(tile),
                        width: dim,
                        height: dim,
                    },
                    cpu_options,
                )
                .expect("CPU JPEG Baseline batch encode");
                total = total.saturating_add(encoded.data.len());
                std::hint::black_box(encoded);
            }
            total
        });
    });

    #[cfg(target_os = "macos")]
    if let Ok(session) = MetalBackendSession::system_default() {
        let buffer = session.device().new_buffer_with_data(
            rgb.as_ptr().cast(),
            rgb.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let tiles = (0..batch_size)
            .map(|tile| JpegBaselineMetalEncodeTile {
                buffer: &buffer,
                byte_offset: tile * tile_bytes,
                width: dim,
                height: dim,
                pitch_bytes: dim as usize * 3,
                output_width: dim,
                output_height: dim,
                format: PixelFormat::Rgb8,
            })
            .collect::<Vec<_>>();
        batch.bench_function(
            format!("metal_rgb8_422_{dim}x{dim}_batch{batch_size}"),
            |b| {
                b.iter(|| {
                    encode_jpeg_baseline_batch_from_metal_buffers(
                        std::hint::black_box(&tiles),
                        metal_options,
                        &session,
                    )
                    .expect("Metal JPEG Baseline batch encode")
                });
            },
        );
    }
    batch.finish();
}

fn options(backend: JpegBackend) -> JpegEncodeOptions {
    JpegEncodeOptions {
        quality: bench_quality(),
        subsampling: JpegSubsampling::Ybr422,
        restart_interval: None,
        backend,
    }
}

fn generated_rgb_tiles(dim: u32, tile_count: usize) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(dim as usize * dim as usize * 3 * tile_count);
    for tile in 0..tile_count as u32 {
        for y in 0..dim {
            for x in 0..dim {
                rgb.push(((x * 13 + y * 3 + tile * 29) & 0xff) as u8);
                rgb.push(((x * 5 + y * 11 + (x ^ y) + tile * 17) & 0xff) as u8);
                rgb.push(((x * 7 + y * 17 + (x.wrapping_mul(y) >> 5) + tile * 23) & 0xff) as u8);
            }
        }
    }
    rgb
}

fn bench_dim() -> u32 {
    let Some(value) = std::env::var_os("SIGNINUM_JPEG_ENCODE_BENCH_DIM") else {
        return DEFAULT_DIM;
    };
    let value = value
        .to_string_lossy()
        .parse::<u32>()
        .expect("SIGNINUM_JPEG_ENCODE_BENCH_DIM must be a u32");
    assert!(
        (64..=4096).contains(&value),
        "SIGNINUM_JPEG_ENCODE_BENCH_DIM must be between 64 and 4096"
    );
    value
}

fn bench_batch_size() -> usize {
    let Some(value) = std::env::var_os("SIGNINUM_JPEG_ENCODE_BENCH_BATCH") else {
        return DEFAULT_BATCH_SIZE;
    };
    let value = value
        .to_string_lossy()
        .parse::<usize>()
        .expect("SIGNINUM_JPEG_ENCODE_BENCH_BATCH must be a usize");
    assert!(
        (1..=128).contains(&value),
        "SIGNINUM_JPEG_ENCODE_BENCH_BATCH must be between 1 and 128"
    );
    value
}

fn bench_quality() -> u8 {
    let Some(value) = std::env::var_os("SIGNINUM_JPEG_ENCODE_BENCH_QUALITY") else {
        return DEFAULT_QUALITY;
    };
    let value = value
        .to_string_lossy()
        .parse::<u8>()
        .expect("SIGNINUM_JPEG_ENCODE_BENCH_QUALITY must be a u8");
    assert!(
        (1..=100).contains(&value),
        "SIGNINUM_JPEG_ENCODE_BENCH_QUALITY must be between 1 and 100"
    );
    value
}

criterion_group!(benches, bench_encode_baseline);
criterion_main!(benches);
