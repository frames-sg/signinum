// SPDX-License-Identifier: Apache-2.0

mod common;

use common::{
    bench_inputs, centered_roi, openjpeg_available, openjpeg_decode, slidecodec_decode,
    slidecodec_decode_region, slidecodec_decode_scaled, slidecodec_decode_tile_batch,
    slidecodec_inspect, DecodeMode,
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
        if openjpeg_available() && !input.is_ht {
            decode_gray.bench_function(format!("openjpeg/{}", input.name), |b| {
                b.iter(|| openjpeg_decode(input, None, None, 1));
            });
        }
    }
    decode_gray.finish();

    let mut decode_rgb = c.benchmark_group("decode_rgb");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb8) {
        decode_rgb.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_decode(&input.bytes, input.mode));
        });
        if openjpeg_available() && !input.is_ht {
            decode_rgb.bench_function(format!("openjpeg/{}", input.name), |b| {
                b.iter(|| openjpeg_decode(input, None, None, 1));
            });
        }
    }
    decode_rgb.finish();

    let mut wsi_region = c.benchmark_group("wsi_region_gray");
    for input in inputs
        .iter()
        .filter(|input| input.mode == DecodeMode::Gray8)
    {
        let roi = centered_roi(input.dimensions, 256);
        wsi_region.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_decode_region(&input.bytes, input.mode, 256));
        });
        if openjpeg_available() && !input.is_ht {
            wsi_region.bench_function(format!("openjpeg/{}", input.name), |b| {
                b.iter(|| openjpeg_decode(input, None, Some(roi), 1));
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
        if openjpeg_available() && !input.is_ht {
            wsi_scaled.bench_function(format!("openjpeg/{}", input.name), |b| {
                b.iter(|| openjpeg_decode(input, Some(2), None, 1));
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
        if openjpeg_available() && !input.is_ht {
            wsi_tile_batch.bench_function(format!("openjpeg/{}", input.name), |b| {
                b.iter(|| openjpeg_decode(input, None, None, 16));
            });
        }
    }
    wsi_tile_batch.finish();
}

criterion_group!(benches, bench_compare);
criterion_main!(benches);
