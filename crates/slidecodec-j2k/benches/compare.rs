// SPDX-License-Identifier: Apache-2.0

mod common;

use common::{
    bench_inputs, distinct_rgb_tile_batch_inputs, metal_available, slidecodec_adaptive_decode,
    slidecodec_adaptive_decode_region, slidecodec_adaptive_decode_scaled,
    slidecodec_adaptive_decode_tile_batch, slidecodec_decode, slidecodec_decode_region,
    slidecodec_decode_scaled, slidecodec_decode_tile_batch, slidecodec_decode_tile_batch_distinct,
    slidecodec_inspect, slidecodec_metal_decode, slidecodec_metal_decode_region,
    slidecodec_metal_decode_scaled, slidecodec_metal_decode_tile_batch,
    slidecodec_metal_decode_tile_batch_distinct, slidecodec_metal_supports_decode,
    slidecodec_metal_supports_region, slidecodec_metal_supports_scaled,
    slidecodec_metal_supports_tile_batch, slidecodec_metal_supports_tile_batch_distinct,
    DecodeMode,
};
use criterion::{criterion_group, criterion_main, Criterion};
use slidecodec_j2k::Downscale;

fn bench_compare(c: &mut Criterion) {
    let inputs = bench_inputs();

    let mut inspect = c.benchmark_group("inspect");
    for input in &inputs {
        inspect.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_inspect(&input.bytes));
        });
    }
    inspect.finish();

    let mut decode_gray = c.benchmark_group("decode_gray");
    for input in inputs
        .iter()
        .filter(|input| input.mode == DecodeMode::Gray8)
    {
        decode_gray.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_decode(&input.bytes, input.mode));
        });
        decode_gray.bench_function(format!("slidecodec-adaptive/{}", input.name), |b| {
            b.iter(|| slidecodec_adaptive_decode(&input.bytes, input.mode));
        });
        if metal_available() && slidecodec_metal_supports_decode(&input.bytes, input.mode) {
            decode_gray.bench_function(format!("slidecodec-metal/{}", input.name), |b| {
                b.iter(|| slidecodec_metal_decode(&input.bytes, input.mode));
            });
        }
    }
    decode_gray.finish();

    let mut decode_rgb = c.benchmark_group("decode_rgb");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb8) {
        decode_rgb.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_decode(&input.bytes, input.mode));
        });
        if metal_available() && slidecodec_metal_supports_decode(&input.bytes, input.mode) {
            decode_rgb.bench_function(format!("slidecodec-metal/{}", input.name), |b| {
                b.iter(|| slidecodec_metal_decode(&input.bytes, input.mode));
            });
        }
    }
    decode_rgb.finish();

    let mut wsi_region = c.benchmark_group("wsi_region_gray");
    for input in inputs
        .iter()
        .filter(|input| input.mode == DecodeMode::Gray8)
    {
        wsi_region.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_decode_region(&input.bytes, input.mode, 256));
        });
        wsi_region.bench_function(format!("slidecodec-adaptive/{}", input.name), |b| {
            b.iter(|| slidecodec_adaptive_decode_region(&input.bytes, input.mode, 256));
        });
        if metal_available() && slidecodec_metal_supports_region(&input.bytes, input.mode, 256) {
            wsi_region.bench_function(format!("slidecodec-metal/{}", input.name), |b| {
                b.iter(|| slidecodec_metal_decode_region(&input.bytes, input.mode, 256));
            });
        }
    }
    wsi_region.finish();

    let mut wsi_scaled = c.benchmark_group("wsi_scaled_gray_q4");
    for input in inputs
        .iter()
        .filter(|input| input.mode == DecodeMode::Gray8)
    {
        wsi_scaled.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_decode_scaled(&input.bytes, input.mode, Downscale::Quarter));
        });
        wsi_scaled.bench_function(format!("slidecodec-adaptive/{}", input.name), |b| {
            b.iter(|| {
                slidecodec_adaptive_decode_scaled(&input.bytes, input.mode, Downscale::Quarter);
            });
        });
        if metal_available()
            && slidecodec_metal_supports_scaled(&input.bytes, input.mode, Downscale::Quarter)
        {
            wsi_scaled.bench_function(format!("slidecodec-metal/{}", input.name), |b| {
                b.iter(|| {
                    slidecodec_metal_decode_scaled(&input.bytes, input.mode, Downscale::Quarter);
                });
            });
        }
    }
    wsi_scaled.finish();

    let mut wsi_tile_batch = c.benchmark_group("wsi_tile_batch_gray");
    for input in inputs
        .iter()
        .filter(|input| input.mode == DecodeMode::Gray8)
    {
        wsi_tile_batch.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_decode_tile_batch(&input.bytes, input.mode, 16));
        });
        wsi_tile_batch.bench_function(format!("slidecodec-adaptive/{}", input.name), |b| {
            b.iter(|| slidecodec_adaptive_decode_tile_batch(input, 16));
        });
        if metal_available() && slidecodec_metal_supports_tile_batch(&input.bytes, input.mode) {
            wsi_tile_batch.bench_function(format!("slidecodec-metal/{}", input.name), |b| {
                b.iter(|| slidecodec_metal_decode_tile_batch(&input.bytes, input.mode, 16));
            });
        }
    }
    wsi_tile_batch.finish();

    let mut wsi_tile_batch_32 = c.benchmark_group("wsi_tile_batch_gray_32");
    for input in inputs
        .iter()
        .filter(|input| input.mode == DecodeMode::Gray8)
    {
        wsi_tile_batch_32.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_decode_tile_batch(&input.bytes, input.mode, 32));
        });
        wsi_tile_batch_32.bench_function(format!("slidecodec-adaptive/{}", input.name), |b| {
            b.iter(|| slidecodec_adaptive_decode_tile_batch(input, 32));
        });
        if metal_available() && slidecodec_metal_supports_tile_batch(&input.bytes, input.mode) {
            wsi_tile_batch_32.bench_function(format!("slidecodec-metal/{}", input.name), |b| {
                b.iter(|| slidecodec_metal_decode_tile_batch(&input.bytes, input.mode, 32));
            });
        }
    }
    wsi_tile_batch_32.finish();

    let mut wsi_tile_batch_64 = c.benchmark_group("wsi_tile_batch_gray_64");
    for input in inputs
        .iter()
        .filter(|input| input.mode == DecodeMode::Gray8)
    {
        wsi_tile_batch_64.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_decode_tile_batch(&input.bytes, input.mode, 64));
        });
        wsi_tile_batch_64.bench_function(format!("slidecodec-adaptive/{}", input.name), |b| {
            b.iter(|| slidecodec_adaptive_decode_tile_batch(input, 64));
        });
        if metal_available() && slidecodec_metal_supports_tile_batch(&input.bytes, input.mode) {
            wsi_tile_batch_64.bench_function(format!("slidecodec-metal/{}", input.name), |b| {
                b.iter(|| slidecodec_metal_decode_tile_batch(&input.bytes, input.mode, 64));
            });
        }
    }
    wsi_tile_batch_64.finish();

    let mut wsi_tile_batch_rgb = c.benchmark_group("wsi_tile_batch_rgb");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb8) {
        wsi_tile_batch_rgb.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_decode_tile_batch(&input.bytes, input.mode, 16));
        });
        if metal_available() && slidecodec_metal_supports_tile_batch(&input.bytes, input.mode) {
            wsi_tile_batch_rgb.bench_function(format!("slidecodec-metal/{}", input.name), |b| {
                b.iter(|| slidecodec_metal_decode_tile_batch(&input.bytes, input.mode, 16));
            });
        }
    }
    wsi_tile_batch_rgb.finish();

    let mut wsi_tile_batch_rgb_distinct = c.benchmark_group("wsi_tile_batch_rgb_distinct");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb8) {
        let distinct_inputs = distinct_rgb_tile_batch_inputs(input, 16);
        wsi_tile_batch_rgb_distinct.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_decode_tile_batch_distinct(&distinct_inputs, input.mode));
        });
        if metal_available()
            && slidecodec_metal_supports_tile_batch_distinct(&distinct_inputs, input.mode)
        {
            wsi_tile_batch_rgb_distinct.bench_function(
                format!("slidecodec-metal/{}", input.name),
                |b| {
                    b.iter(|| {
                        slidecodec_metal_decode_tile_batch_distinct(&distinct_inputs, input.mode);
                    });
                },
            );
        }
    }
    wsi_tile_batch_rgb_distinct.finish();
}

criterion_group!(benches, bench_compare);
criterion_main!(benches);
