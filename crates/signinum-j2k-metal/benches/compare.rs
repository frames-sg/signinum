// SPDX-License-Identifier: Apache-2.0

mod common;

use common::{
    bench_inputs, distinct_gray_tile_batch_inputs, distinct_rgb_tile_batch_inputs,
    external_wsi_tile_batches, grok_decode_external_tile_batch_region_scaled,
    grok_supports_external_tile_batch_region_scaled, j2k_tile_batch_sizes, metal_available,
    openjpeg_decode_external_tile_batch_region_scaled,
    openjpeg_supports_external_tile_batch_region_scaled, signinum_adaptive_decode,
    signinum_adaptive_decode_external_tile_batch_region_scaled, signinum_adaptive_decode_region,
    signinum_adaptive_decode_region_scaled, signinum_adaptive_decode_scaled,
    signinum_adaptive_decode_tile_batch, signinum_adaptive_decode_tile_batch_region_scaled,
    signinum_adaptive_decode_tile_batch_region_scaled_distinct, signinum_decode,
    signinum_decode_external_tile_batch_region_scaled, signinum_decode_region,
    signinum_decode_region_scaled, signinum_decode_scaled, signinum_decode_tile_batch,
    signinum_decode_tile_batch_distinct, signinum_decode_tile_batch_region_scaled,
    signinum_decode_tile_batch_region_scaled_distinct, signinum_inspect, signinum_metal_decode,
    signinum_metal_decode_external_tile_batch_region_scaled, signinum_metal_decode_region,
    signinum_metal_decode_region_scaled, signinum_metal_decode_scaled,
    signinum_metal_decode_tile_batch, signinum_metal_decode_tile_batch_distinct,
    signinum_metal_decode_tile_batch_region_scaled,
    signinum_metal_decode_tile_batch_region_scaled_distinct, signinum_metal_supports_decode,
    signinum_metal_supports_external_tile_batch_region_scaled, signinum_metal_supports_region,
    signinum_metal_supports_region_scaled, signinum_metal_supports_scaled,
    signinum_metal_supports_tile_batch, signinum_metal_supports_tile_batch_distinct,
    signinum_metal_supports_tile_batch_region_scaled,
    signinum_metal_supports_tile_batch_region_scaled_distinct, DecodeMode,
};
use criterion::{criterion_group, criterion_main, Criterion};
use signinum_core::Rect;
use signinum_j2k::Downscale;
use signinum_j2k_compare::{grok, openjpeg};

fn bench_compare(c: &mut Criterion) {
    let inputs = bench_inputs();
    let batch_sizes = j2k_tile_batch_sizes();
    let max_batch_size = batch_sizes.iter().copied().max().unwrap_or(16);

    let mut inspect = c.benchmark_group("inspect");
    for input in &inputs {
        inspect.bench_function(format!("signinum/{}", input.name), |b| {
            b.iter(|| signinum_inspect(&input.bytes));
        });
    }
    inspect.finish();

    let mut decode_gray = c.benchmark_group("decode_gray");
    for input in inputs
        .iter()
        .filter(|input| input.mode == DecodeMode::Gray8)
    {
        decode_gray.bench_function(format!("signinum/{}", input.name), |b| {
            b.iter(|| signinum_decode(&input.bytes, input.mode));
        });
        decode_gray.bench_function(format!("signinum-adaptive/{}", input.name), |b| {
            b.iter(|| signinum_adaptive_decode(&input.bytes, input.mode));
        });
        if !input.is_ht && openjpeg::is_available() {
            decode_gray.bench_function(format!("openjpeg/{}", input.name), |b| {
                b.iter(|| openjpeg::decode_gray(&input.bytes).expect("OpenJPEG decode"));
            });
        }
        if grok::is_available() {
            decode_gray.bench_function(format!("grok/{}", input.name), |b| {
                b.iter(|| grok::decode_gray(&input.bytes).expect("Grok decode"));
            });
        }
        if metal_available() && signinum_metal_supports_decode(&input.bytes, input.mode) {
            decode_gray.bench_function(format!("signinum-metal/{}", input.name), |b| {
                b.iter(|| signinum_metal_decode(&input.bytes, input.mode));
            });
        }
    }
    decode_gray.finish();

    let mut decode_rgb = c.benchmark_group("decode_rgb");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb8) {
        decode_rgb.bench_function(format!("signinum/{}", input.name), |b| {
            b.iter(|| signinum_decode(&input.bytes, input.mode));
        });
        if !input.is_ht && openjpeg::is_available() {
            decode_rgb.bench_function(format!("openjpeg/{}", input.name), |b| {
                b.iter(|| openjpeg::decode_rgb(&input.bytes).expect("OpenJPEG decode"));
            });
        }
        if grok::is_available() {
            decode_rgb.bench_function(format!("grok/{}", input.name), |b| {
                b.iter(|| grok::decode_rgb(&input.bytes).expect("Grok decode"));
            });
        }
        if metal_available() && signinum_metal_supports_decode(&input.bytes, input.mode) {
            decode_rgb.bench_function(format!("signinum-metal/{}", input.name), |b| {
                b.iter(|| signinum_metal_decode(&input.bytes, input.mode));
            });
        }
    }
    decode_rgb.finish();

    let mut wsi_region = c.benchmark_group("wsi_region_gray");
    for input in inputs
        .iter()
        .filter(|input| input.mode == DecodeMode::Gray8)
    {
        wsi_region.bench_function(format!("signinum/{}", input.name), |b| {
            b.iter(|| signinum_decode_region(&input.bytes, input.mode, 256));
        });
        wsi_region.bench_function(format!("signinum-adaptive/{}", input.name), |b| {
            b.iter(|| signinum_adaptive_decode_region(&input.bytes, input.mode, 256));
        });
        if !input.is_ht && openjpeg::is_available() {
            wsi_region.bench_function(format!("openjpeg/{}", input.name), |b| {
                let roi = compare_roi(input.dimensions, 256);
                b.iter(|| {
                    openjpeg::decode_gray_region(&input.bytes, roi).expect("OpenJPEG region decode")
                });
            });
        }
        if grok::is_available() {
            wsi_region.bench_function(format!("grok/{}", input.name), |b| {
                let roi = compare_roi(input.dimensions, 256);
                b.iter(|| grok::decode_gray_region(&input.bytes, roi).expect("Grok region decode"));
            });
        }
        if metal_available() && signinum_metal_supports_region(&input.bytes, input.mode, 256) {
            wsi_region.bench_function(format!("signinum-metal/{}", input.name), |b| {
                b.iter(|| signinum_metal_decode_region(&input.bytes, input.mode, 256));
            });
        }
    }
    wsi_region.finish();

    let mut wsi_scaled = c.benchmark_group("wsi_scaled_gray_q4");
    for input in inputs
        .iter()
        .filter(|input| input.mode == DecodeMode::Gray8)
    {
        wsi_scaled.bench_function(format!("signinum/{}", input.name), |b| {
            b.iter(|| signinum_decode_scaled(&input.bytes, input.mode, Downscale::Quarter));
        });
        wsi_scaled.bench_function(format!("signinum-adaptive/{}", input.name), |b| {
            b.iter(|| {
                signinum_adaptive_decode_scaled(&input.bytes, input.mode, Downscale::Quarter);
            });
        });
        if !input.is_ht && openjpeg::is_available() {
            wsi_scaled.bench_function(format!("openjpeg/{}", input.name), |b| {
                b.iter(|| {
                    openjpeg::decode_gray_scaled(&input.bytes, 2).expect("OpenJPEG scaled decode")
                });
            });
        }
        if grok::is_available() {
            wsi_scaled.bench_function(format!("grok/{}", input.name), |b| {
                b.iter(|| grok::decode_gray_scaled(&input.bytes, 2).expect("Grok scaled decode"));
            });
        }
        if metal_available()
            && signinum_metal_supports_scaled(&input.bytes, input.mode, Downscale::Quarter)
        {
            wsi_scaled.bench_function(format!("signinum-metal/{}", input.name), |b| {
                b.iter(|| {
                    signinum_metal_decode_scaled(&input.bytes, input.mode, Downscale::Quarter);
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
        wsi_region_scaled.bench_function(format!("signinum/{}", input.name), |b| {
            b.iter(|| {
                signinum_decode_region_scaled(&input.bytes, input.mode, 256, Downscale::Quarter);
            });
        });
        wsi_region_scaled.bench_function(format!("signinum-adaptive/{}", input.name), |b| {
            b.iter(|| {
                signinum_adaptive_decode_region_scaled(
                    &input.bytes,
                    input.mode,
                    256,
                    Downscale::Quarter,
                );
            });
        });
        if metal_available()
            && signinum_metal_supports_region_scaled(
                &input.bytes,
                input.mode,
                256,
                Downscale::Quarter,
            )
        {
            wsi_region_scaled.bench_function(format!("signinum-metal/{}", input.name), |b| {
                b.iter(|| {
                    signinum_metal_decode_region_scaled(
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
        for &count in &batch_sizes {
            wsi_tile_batch.bench_function(format!("signinum/{}/batch_{count}", input.name), |b| {
                b.iter(|| signinum_decode_tile_batch(&input.bytes, input.mode, count));
            });
            wsi_tile_batch.bench_function(
                format!("signinum-adaptive/{}/batch_{count}", input.name),
                |b| {
                    b.iter(|| signinum_adaptive_decode_tile_batch(input, count));
                },
            );
            if metal_available() && signinum_metal_supports_tile_batch(&input.bytes, input.mode) {
                wsi_tile_batch.bench_function(
                    format!("signinum-metal/{}/batch_{count}", input.name),
                    |b| {
                        b.iter(|| {
                            signinum_metal_decode_tile_batch(&input.bytes, input.mode, count);
                        });
                    },
                );
            }
        }
    }
    wsi_tile_batch.finish();

    let mut wsi_tile_batch_region_scaled =
        c.benchmark_group("wsi_tile_batch_region_scaled_gray_q4");
    for input in inputs
        .iter()
        .filter(|input| input.mode == DecodeMode::Gray8)
    {
        for &count in &batch_sizes {
            wsi_tile_batch_region_scaled.bench_function(
                format!("signinum/{}/batch_{count}", input.name),
                |b| {
                    b.iter(|| {
                        signinum_decode_tile_batch_region_scaled(
                            &input.bytes,
                            input.mode,
                            256,
                            Downscale::Quarter,
                            count,
                        );
                    });
                },
            );
            wsi_tile_batch_region_scaled.bench_function(
                format!("signinum-adaptive/{}/batch_{count}", input.name),
                |b| {
                    b.iter(|| {
                        signinum_adaptive_decode_tile_batch_region_scaled(
                            input,
                            256,
                            Downscale::Quarter,
                            count,
                        );
                    });
                },
            );
            if metal_available()
                && signinum_metal_supports_tile_batch_region_scaled(
                    &input.bytes,
                    input.mode,
                    256,
                    Downscale::Quarter,
                )
            {
                wsi_tile_batch_region_scaled.bench_function(
                    format!("signinum-metal/{}/batch_{count}", input.name),
                    |b| {
                        b.iter(|| {
                            signinum_metal_decode_tile_batch_region_scaled(
                                &input.bytes,
                                input.mode,
                                256,
                                Downscale::Quarter,
                                count,
                            );
                        });
                    },
                );
            }
        }
    }
    wsi_tile_batch_region_scaled.finish();

    let mut wsi_tile_batch_region_scaled_distinct =
        c.benchmark_group("wsi_tile_batch_region_scaled_gray_distinct_q4");
    for input in inputs
        .iter()
        .filter(|input| input.mode == DecodeMode::Gray8)
    {
        for &count in &batch_sizes {
            let distinct_inputs = distinct_gray_tile_batch_inputs(input, count);
            wsi_tile_batch_region_scaled_distinct.bench_function(
                format!("signinum/{}/batch_{count}", input.name),
                |b| {
                    b.iter(|| {
                        signinum_decode_tile_batch_region_scaled_distinct(
                            &distinct_inputs,
                            input.mode,
                            256,
                            Downscale::Quarter,
                        );
                    });
                },
            );
            wsi_tile_batch_region_scaled_distinct.bench_function(
                format!("signinum-adaptive/{}/batch_{count}", input.name),
                |b| {
                    b.iter(|| {
                        signinum_adaptive_decode_tile_batch_region_scaled_distinct(
                            &distinct_inputs,
                            input.mode,
                            256,
                            Downscale::Quarter,
                        );
                    });
                },
            );
            if metal_available()
                && signinum_metal_supports_tile_batch_region_scaled_distinct(
                    &distinct_inputs,
                    input.mode,
                    256,
                    Downscale::Quarter,
                )
            {
                wsi_tile_batch_region_scaled_distinct.bench_function(
                    format!("signinum-metal/{}/batch_{count}", input.name),
                    |b| {
                        b.iter(|| {
                            signinum_metal_decode_tile_batch_region_scaled_distinct(
                                &distinct_inputs,
                                input.mode,
                                256,
                                Downscale::Quarter,
                            );
                        });
                    },
                );
            }
        }
    }
    wsi_tile_batch_region_scaled_distinct.finish();

    let external_batches = external_wsi_tile_batches(max_batch_size);
    if !external_batches.is_empty() {
        let mut external_wsi_region_scaled =
            c.benchmark_group("external_wsi_tile_batch_region_scaled_q4");
        for batch in &external_batches {
            for &count in &batch_sizes {
                if batch.inputs.len() < count {
                    continue;
                }
                external_wsi_region_scaled.bench_function(
                    format!("signinum/{}/batch_{count}", batch.name),
                    |b| {
                        b.iter(|| {
                            signinum_decode_external_tile_batch_region_scaled(
                                batch,
                                count,
                                256,
                                Downscale::Quarter,
                            );
                        });
                    },
                );
                external_wsi_region_scaled.bench_function(
                    format!("signinum-adaptive/{}/batch_{count}", batch.name),
                    |b| {
                        b.iter(|| {
                            signinum_adaptive_decode_external_tile_batch_region_scaled(
                                batch,
                                count,
                                256,
                                Downscale::Quarter,
                            );
                        });
                    },
                );
                if openjpeg::is_available()
                    && openjpeg_supports_external_tile_batch_region_scaled(batch, count)
                {
                    external_wsi_region_scaled.bench_function(
                        format!("openjpeg/{}/batch_{count}", batch.name),
                        |b| {
                            b.iter(|| {
                                openjpeg_decode_external_tile_batch_region_scaled(
                                    batch,
                                    count,
                                    256,
                                    Downscale::Quarter,
                                );
                            });
                        },
                    );
                }
                if grok_supports_external_tile_batch_region_scaled(batch, count) {
                    external_wsi_region_scaled.bench_function(
                        format!("grok/{}/batch_{count}", batch.name),
                        |b| {
                            b.iter(|| {
                                grok_decode_external_tile_batch_region_scaled(
                                    batch,
                                    count,
                                    256,
                                    Downscale::Quarter,
                                );
                            });
                        },
                    );
                }
                if metal_available()
                    && signinum_metal_supports_external_tile_batch_region_scaled(
                        batch,
                        count,
                        256,
                        Downscale::Quarter,
                    )
                {
                    external_wsi_region_scaled.bench_function(
                        format!("signinum-metal/{}/batch_{count}", batch.name),
                        |b| {
                            b.iter(|| {
                                signinum_metal_decode_external_tile_batch_region_scaled(
                                    batch,
                                    count,
                                    256,
                                    Downscale::Quarter,
                                );
                            });
                        },
                    );
                }
            }
        }
        external_wsi_region_scaled.finish();
    }

    let mut wsi_tile_batch_rgb = c.benchmark_group("wsi_tile_batch_rgb");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb8) {
        for &count in &batch_sizes {
            wsi_tile_batch_rgb.bench_function(
                format!("signinum/{}/batch_{count}", input.name),
                |b| {
                    b.iter(|| signinum_decode_tile_batch(&input.bytes, input.mode, count));
                },
            );
            if metal_available() && signinum_metal_supports_tile_batch(&input.bytes, input.mode) {
                wsi_tile_batch_rgb.bench_function(
                    format!("signinum-metal/{}/batch_{count}", input.name),
                    |b| {
                        b.iter(|| {
                            signinum_metal_decode_tile_batch(&input.bytes, input.mode, count);
                        });
                    },
                );
            }
        }
    }
    wsi_tile_batch_rgb.finish();

    let mut wsi_tile_batch_rgb_distinct = c.benchmark_group("wsi_tile_batch_rgb_distinct");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb8) {
        for &count in &batch_sizes {
            let distinct_inputs = distinct_rgb_tile_batch_inputs(input, count);
            wsi_tile_batch_rgb_distinct.bench_function(
                format!("signinum/{}/batch_{count}", input.name),
                |b| {
                    b.iter(|| signinum_decode_tile_batch_distinct(&distinct_inputs, input.mode));
                },
            );
            if metal_available()
                && signinum_metal_supports_tile_batch_distinct(&distinct_inputs, input.mode)
            {
                wsi_tile_batch_rgb_distinct.bench_function(
                    format!("signinum-metal/{}/batch_{count}", input.name),
                    |b| {
                        b.iter(|| {
                            signinum_metal_decode_tile_batch_distinct(&distinct_inputs, input.mode);
                        });
                    },
                );
            }
        }
    }
    wsi_tile_batch_rgb_distinct.finish();
}

fn compare_roi(dimensions: (u32, u32), extent: u32) -> Rect {
    Rect {
        x: dimensions.0.saturating_sub(extent) / 2,
        y: dimensions.1.saturating_sub(extent) / 2,
        w: dimensions.0.min(extent),
        h: dimensions.1.min(extent),
    }
}

criterion_group!(benches, bench_compare);
criterion_main!(benches);
