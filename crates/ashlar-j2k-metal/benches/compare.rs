// SPDX-License-Identifier: Apache-2.0

mod common;

use ashlar_j2k::Downscale;
use common::{
    ashlar_adaptive_decode, ashlar_adaptive_decode_region, ashlar_adaptive_decode_region_scaled,
    ashlar_adaptive_decode_scaled, ashlar_adaptive_decode_tile_batch,
    ashlar_adaptive_decode_tile_batch_region_scaled, ashlar_decode, ashlar_decode_region,
    ashlar_decode_region_scaled, ashlar_decode_scaled, ashlar_decode_tile_batch,
    ashlar_decode_tile_batch_distinct, ashlar_decode_tile_batch_region_scaled, ashlar_inspect,
    ashlar_metal_decode, ashlar_metal_decode_region, ashlar_metal_decode_region_scaled,
    ashlar_metal_decode_scaled, ashlar_metal_decode_tile_batch,
    ashlar_metal_decode_tile_batch_distinct, ashlar_metal_decode_tile_batch_region_scaled,
    ashlar_metal_supports_decode, ashlar_metal_supports_region,
    ashlar_metal_supports_region_scaled, ashlar_metal_supports_scaled,
    ashlar_metal_supports_tile_batch, ashlar_metal_supports_tile_batch_distinct,
    ashlar_metal_supports_tile_batch_region_scaled, bench_inputs, distinct_rgb_tile_batch_inputs,
    metal_available, DecodeMode,
};
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_compare(c: &mut Criterion) {
    let inputs = bench_inputs();

    let mut inspect = c.benchmark_group("inspect");
    for input in &inputs {
        inspect.bench_function(format!("ashlar/{}", input.name), |b| {
            b.iter(|| ashlar_inspect(&input.bytes));
        });
    }
    inspect.finish();

    let mut decode_gray = c.benchmark_group("decode_gray");
    for input in inputs
        .iter()
        .filter(|input| input.mode == DecodeMode::Gray8)
    {
        decode_gray.bench_function(format!("ashlar/{}", input.name), |b| {
            b.iter(|| ashlar_decode(&input.bytes, input.mode));
        });
        decode_gray.bench_function(format!("ashlar-adaptive/{}", input.name), |b| {
            b.iter(|| ashlar_adaptive_decode(&input.bytes, input.mode));
        });
        if metal_available() && ashlar_metal_supports_decode(&input.bytes, input.mode) {
            decode_gray.bench_function(format!("ashlar-metal/{}", input.name), |b| {
                b.iter(|| ashlar_metal_decode(&input.bytes, input.mode));
            });
        }
    }
    decode_gray.finish();

    let mut decode_rgb = c.benchmark_group("decode_rgb");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb8) {
        decode_rgb.bench_function(format!("ashlar/{}", input.name), |b| {
            b.iter(|| ashlar_decode(&input.bytes, input.mode));
        });
        if metal_available() && ashlar_metal_supports_decode(&input.bytes, input.mode) {
            decode_rgb.bench_function(format!("ashlar-metal/{}", input.name), |b| {
                b.iter(|| ashlar_metal_decode(&input.bytes, input.mode));
            });
        }
    }
    decode_rgb.finish();

    let mut wsi_region = c.benchmark_group("wsi_region_gray");
    for input in inputs
        .iter()
        .filter(|input| input.mode == DecodeMode::Gray8)
    {
        wsi_region.bench_function(format!("ashlar/{}", input.name), |b| {
            b.iter(|| ashlar_decode_region(&input.bytes, input.mode, 256));
        });
        wsi_region.bench_function(format!("ashlar-adaptive/{}", input.name), |b| {
            b.iter(|| ashlar_adaptive_decode_region(&input.bytes, input.mode, 256));
        });
        if metal_available() && ashlar_metal_supports_region(&input.bytes, input.mode, 256) {
            wsi_region.bench_function(format!("ashlar-metal/{}", input.name), |b| {
                b.iter(|| ashlar_metal_decode_region(&input.bytes, input.mode, 256));
            });
        }
    }
    wsi_region.finish();

    let mut wsi_scaled = c.benchmark_group("wsi_scaled_gray_q4");
    for input in inputs
        .iter()
        .filter(|input| input.mode == DecodeMode::Gray8)
    {
        wsi_scaled.bench_function(format!("ashlar/{}", input.name), |b| {
            b.iter(|| ashlar_decode_scaled(&input.bytes, input.mode, Downscale::Quarter));
        });
        wsi_scaled.bench_function(format!("ashlar-adaptive/{}", input.name), |b| {
            b.iter(|| {
                ashlar_adaptive_decode_scaled(&input.bytes, input.mode, Downscale::Quarter);
            });
        });
        if metal_available()
            && ashlar_metal_supports_scaled(&input.bytes, input.mode, Downscale::Quarter)
        {
            wsi_scaled.bench_function(format!("ashlar-metal/{}", input.name), |b| {
                b.iter(|| {
                    ashlar_metal_decode_scaled(&input.bytes, input.mode, Downscale::Quarter);
                });
            });
        }
    }
    wsi_scaled.finish();

    let mut wsi_region_scaled = c.benchmark_group("wsi_region_scaled_gray_q4");
    for input in inputs
        .iter()
        .filter(|input| input.mode == DecodeMode::Gray8)
    {
        wsi_region_scaled.bench_function(format!("ashlar/{}", input.name), |b| {
            b.iter(|| {
                ashlar_decode_region_scaled(&input.bytes, input.mode, 256, Downscale::Quarter);
            });
        });
        wsi_region_scaled.bench_function(format!("ashlar-adaptive/{}", input.name), |b| {
            b.iter(|| {
                ashlar_adaptive_decode_region_scaled(
                    &input.bytes,
                    input.mode,
                    256,
                    Downscale::Quarter,
                );
            });
        });
        if metal_available()
            && ashlar_metal_supports_region_scaled(
                &input.bytes,
                input.mode,
                256,
                Downscale::Quarter,
            )
        {
            wsi_region_scaled.bench_function(format!("ashlar-metal/{}", input.name), |b| {
                b.iter(|| {
                    ashlar_metal_decode_region_scaled(
                        &input.bytes,
                        input.mode,
                        256,
                        Downscale::Quarter,
                    );
                });
            });
        }
    }
    wsi_region_scaled.finish();

    let mut wsi_tile_batch = c.benchmark_group("wsi_tile_batch_gray");
    for input in inputs
        .iter()
        .filter(|input| input.mode == DecodeMode::Gray8)
    {
        wsi_tile_batch.bench_function(format!("ashlar/{}", input.name), |b| {
            b.iter(|| ashlar_decode_tile_batch(&input.bytes, input.mode, 16));
        });
        wsi_tile_batch.bench_function(format!("ashlar-adaptive/{}", input.name), |b| {
            b.iter(|| ashlar_adaptive_decode_tile_batch(input, 16));
        });
        if metal_available() && ashlar_metal_supports_tile_batch(&input.bytes, input.mode) {
            wsi_tile_batch.bench_function(format!("ashlar-metal/{}", input.name), |b| {
                b.iter(|| ashlar_metal_decode_tile_batch(&input.bytes, input.mode, 16));
            });
        }
    }
    wsi_tile_batch.finish();

    let mut wsi_tile_batch_region_scaled =
        c.benchmark_group("wsi_tile_batch_region_scaled_gray_q4");
    for input in inputs
        .iter()
        .filter(|input| input.mode == DecodeMode::Gray8)
    {
        wsi_tile_batch_region_scaled.bench_function(format!("ashlar/{}", input.name), |b| {
            b.iter(|| {
                ashlar_decode_tile_batch_region_scaled(
                    &input.bytes,
                    input.mode,
                    256,
                    Downscale::Quarter,
                    16,
                );
            });
        });
        wsi_tile_batch_region_scaled.bench_function(
            format!("ashlar-adaptive/{}", input.name),
            |b| {
                b.iter(|| {
                    ashlar_adaptive_decode_tile_batch_region_scaled(
                        input,
                        256,
                        Downscale::Quarter,
                        16,
                    );
                });
            },
        );
        if metal_available()
            && ashlar_metal_supports_tile_batch_region_scaled(
                &input.bytes,
                input.mode,
                256,
                Downscale::Quarter,
            )
        {
            wsi_tile_batch_region_scaled.bench_function(
                format!("ashlar-metal/{}", input.name),
                |b| {
                    b.iter(|| {
                        ashlar_metal_decode_tile_batch_region_scaled(
                            &input.bytes,
                            input.mode,
                            256,
                            Downscale::Quarter,
                            16,
                        );
                    });
                },
            );
        }
    }
    wsi_tile_batch_region_scaled.finish();

    let mut wsi_tile_batch_32 = c.benchmark_group("wsi_tile_batch_gray_32");
    for input in inputs
        .iter()
        .filter(|input| input.mode == DecodeMode::Gray8)
    {
        wsi_tile_batch_32.bench_function(format!("ashlar/{}", input.name), |b| {
            b.iter(|| ashlar_decode_tile_batch(&input.bytes, input.mode, 32));
        });
        wsi_tile_batch_32.bench_function(format!("ashlar-adaptive/{}", input.name), |b| {
            b.iter(|| ashlar_adaptive_decode_tile_batch(input, 32));
        });
        if metal_available() && ashlar_metal_supports_tile_batch(&input.bytes, input.mode) {
            wsi_tile_batch_32.bench_function(format!("ashlar-metal/{}", input.name), |b| {
                b.iter(|| ashlar_metal_decode_tile_batch(&input.bytes, input.mode, 32));
            });
        }
    }
    wsi_tile_batch_32.finish();

    let mut wsi_tile_batch_64 = c.benchmark_group("wsi_tile_batch_gray_64");
    for input in inputs
        .iter()
        .filter(|input| input.mode == DecodeMode::Gray8)
    {
        wsi_tile_batch_64.bench_function(format!("ashlar/{}", input.name), |b| {
            b.iter(|| ashlar_decode_tile_batch(&input.bytes, input.mode, 64));
        });
        wsi_tile_batch_64.bench_function(format!("ashlar-adaptive/{}", input.name), |b| {
            b.iter(|| ashlar_adaptive_decode_tile_batch(input, 64));
        });
        if metal_available() && ashlar_metal_supports_tile_batch(&input.bytes, input.mode) {
            wsi_tile_batch_64.bench_function(format!("ashlar-metal/{}", input.name), |b| {
                b.iter(|| ashlar_metal_decode_tile_batch(&input.bytes, input.mode, 64));
            });
        }
    }
    wsi_tile_batch_64.finish();

    let mut wsi_tile_batch_rgb = c.benchmark_group("wsi_tile_batch_rgb");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb8) {
        wsi_tile_batch_rgb.bench_function(format!("ashlar/{}", input.name), |b| {
            b.iter(|| ashlar_decode_tile_batch(&input.bytes, input.mode, 16));
        });
        if metal_available() && ashlar_metal_supports_tile_batch(&input.bytes, input.mode) {
            wsi_tile_batch_rgb.bench_function(format!("ashlar-metal/{}", input.name), |b| {
                b.iter(|| ashlar_metal_decode_tile_batch(&input.bytes, input.mode, 16));
            });
        }
    }
    wsi_tile_batch_rgb.finish();

    let mut wsi_tile_batch_rgb_distinct = c.benchmark_group("wsi_tile_batch_rgb_distinct");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb8) {
        let distinct_inputs = distinct_rgb_tile_batch_inputs(input, 16);
        wsi_tile_batch_rgb_distinct.bench_function(format!("ashlar/{}", input.name), |b| {
            b.iter(|| ashlar_decode_tile_batch_distinct(&distinct_inputs, input.mode));
        });
        if metal_available()
            && ashlar_metal_supports_tile_batch_distinct(&distinct_inputs, input.mode)
        {
            wsi_tile_batch_rgb_distinct.bench_function(
                format!("ashlar-metal/{}", input.name),
                |b| {
                    b.iter(|| {
                        ashlar_metal_decode_tile_batch_distinct(&distinct_inputs, input.mode);
                    });
                },
            );
        }
    }
    wsi_tile_batch_rgb_distinct.finish();
}

criterion_group!(benches, bench_compare);
criterion_main!(benches);
