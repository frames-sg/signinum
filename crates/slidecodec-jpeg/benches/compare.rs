// SPDX-License-Identifier: Apache-2.0

mod common;

use common::{
    centered_roi,
    classification::{should_bench_decode_rows_rgb, should_compare_full_frame, CorpusInputClass},
    jpeg_decoder_decode, jpeg_decoder_decode_batch_region_scaled, jpeg_decoder_decode_batch_scaled,
    jpeg_decoder_decode_region, jpeg_decoder_decode_region_scaled, jpeg_decoder_decode_scaled,
    jpeg_decoder_inspect, libjpeg_turbo_available, libjpeg_turbo_decode,
    libjpeg_turbo_decode_batch, libjpeg_turbo_decode_batch_region_scaled,
    libjpeg_turbo_decode_batch_scaled, libjpeg_turbo_decode_region,
    libjpeg_turbo_decode_region_scaled, libjpeg_turbo_decode_scaled, libjpeg_turbo_inspect,
    load_bench_inputs, output_geometry, slidecodec_decode, slidecodec_decode_region,
    slidecodec_decode_region_scaled, slidecodec_decode_reused, slidecodec_decode_rows,
    slidecodec_decode_scaled, slidecodec_decode_tile_batch,
    slidecodec_decode_tile_batch_region_scaled, slidecodec_decode_tile_batch_scaled,
    slidecodec_decode_with_scratch, slidecodec_inspect, zune_decode,
    zune_decode_batch_region_scaled, zune_decode_batch_scaled, zune_decode_region,
    zune_decode_region_scaled, zune_decode_scaled, zune_inspect, DecodeMode, TurboJpegDecoder,
};
use criterion::{criterion_group, criterion_main, Criterion};
use slidecodec_jpeg::{Decoder, Downscale, ScratchPool};

fn bench_compare(c: &mut Criterion) {
    let inputs = load_bench_inputs();

    let mut inspect = c.benchmark_group("inspect");
    for input in &inputs {
        inspect.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_inspect(&input.bytes));
        });
        inspect.bench_function(format!("jpeg-decoder/{}", input.name), |b| {
            b.iter(|| jpeg_decoder_inspect(&input.bytes));
        });
        inspect.bench_function(format!("zune-jpeg/{}", input.name), |b| {
            b.iter(|| zune_inspect(&input.bytes));
        });
        if libjpeg_turbo_available() {
            let mut turbo = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
            inspect.bench_function(format!("libjpeg-turbo/{}", input.name), move |b| {
                b.iter(|| libjpeg_turbo_inspect(&mut turbo, &input.bytes));
            });
        }
    }
    inspect.finish();

    let mut decode_rgb = c.benchmark_group("decode_rgb");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && should_compare_full_frame(input.mode, input.input_class)
    }) {
        decode_rgb.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_decode(&input.bytes, DecodeMode::Rgb));
        });
        decode_rgb.bench_function(format!("jpeg-decoder/{}", input.name), |b| {
            b.iter(|| jpeg_decoder_decode(&input.bytes));
        });
        decode_rgb.bench_function(format!("zune-jpeg/{}", input.name), |b| {
            b.iter(|| zune_decode(&input.bytes, DecodeMode::Rgb));
        });
        if libjpeg_turbo_available() {
            let mut turbo = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
            decode_rgb.bench_function(format!("libjpeg-turbo/{}", input.name), move |b| {
                b.iter(|| libjpeg_turbo_decode(&mut turbo, &input.bytes, DecodeMode::Rgb));
            });
        }
    }
    decode_rgb.finish();

    let mut decode_gray = c.benchmark_group("decode_gray");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Gray && should_compare_full_frame(input.mode, input.input_class)
    }) {
        decode_gray.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_decode(&input.bytes, DecodeMode::Gray));
        });
        decode_gray.bench_function(format!("jpeg-decoder/{}", input.name), |b| {
            b.iter(|| jpeg_decoder_decode(&input.bytes));
        });
        decode_gray.bench_function(format!("zune-jpeg/{}", input.name), |b| {
            b.iter(|| zune_decode(&input.bytes, DecodeMode::Gray));
        });
        if libjpeg_turbo_available() {
            let mut turbo = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
            decode_gray.bench_function(format!("libjpeg-turbo/{}", input.name), move |b| {
                b.iter(|| libjpeg_turbo_decode(&mut turbo, &input.bytes, DecodeMode::Gray));
            });
        }
    }
    decode_gray.finish();

    let mut decode_reused_rgb = c.benchmark_group("decode_reused_rgb");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        let dec = Decoder::new(&input.bytes).expect("slidecodec decoder (reused-setup)");
        let (fmt, stride, len) = output_geometry(&dec, DecodeMode::Rgb);
        let mut out = vec![0u8; len];
        decode_reused_rgb.bench_function(format!("slidecodec_reused/{}", input.name), |b| {
            b.iter(|| slidecodec_decode_reused(&dec, &mut out, stride, fmt));
        });
    }
    decode_reused_rgb.finish();

    let mut decode_reused_gray = c.benchmark_group("decode_reused_gray");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Gray && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        let dec = Decoder::new(&input.bytes).expect("slidecodec decoder (reused-setup)");
        let (fmt, stride, len) = output_geometry(&dec, DecodeMode::Gray);
        let mut out = vec![0u8; len];
        decode_reused_gray.bench_function(format!("slidecodec_reused/{}", input.name), |b| {
            b.iter(|| slidecodec_decode_reused(&dec, &mut out, stride, fmt));
        });
    }
    decode_reused_gray.finish();

    let mut decode_scratch_rgb = c.benchmark_group("decode_scratch_rgb");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        let dec = Decoder::new(&input.bytes).expect("slidecodec decoder (scratch-setup)");
        let (fmt, stride, len) = output_geometry(&dec, DecodeMode::Rgb);
        let mut out = vec![0u8; len];
        let mut pool = ScratchPool::new();
        // Warm the pool once so iteration 1 pays zero allocation cost.
        slidecodec_decode_with_scratch(&dec, &mut pool, &mut out, stride, fmt);
        decode_scratch_rgb.bench_function(format!("slidecodec_scratch/{}", input.name), |b| {
            b.iter(|| slidecodec_decode_with_scratch(&dec, &mut pool, &mut out, stride, fmt));
        });
    }
    decode_scratch_rgb.finish();

    let mut decode_scratch_gray = c.benchmark_group("decode_scratch_gray");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Gray && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        let dec = Decoder::new(&input.bytes).expect("slidecodec decoder (scratch-setup)");
        let (fmt, stride, len) = output_geometry(&dec, DecodeMode::Gray);
        let mut out = vec![0u8; len];
        let mut pool = ScratchPool::new();
        slidecodec_decode_with_scratch(&dec, &mut pool, &mut out, stride, fmt);
        decode_scratch_gray.bench_function(format!("slidecodec_scratch/{}", input.name), |b| {
            b.iter(|| slidecodec_decode_with_scratch(&dec, &mut pool, &mut out, stride, fmt));
        });
    }
    decode_scratch_gray.finish();

    let mut decode_rows_rgb = c.benchmark_group("decode_rows_rgb");
    for input in inputs
        .iter()
        .filter(|input| should_bench_decode_rows_rgb(input.mode, input.input_class))
    {
        decode_rows_rgb.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_decode_rows(&input.bytes));
        });
    }
    decode_rows_rgb.finish();

    let mut wsi_tile_batch_rgb = c.benchmark_group("wsi_tile_batch_rgb");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb) {
        wsi_tile_batch_rgb.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_decode_tile_batch(&input.bytes, 64));
        });
        if libjpeg_turbo_available() {
            let mut turbo = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
            wsi_tile_batch_rgb.bench_function(format!("libjpeg-turbo/{}", input.name), move |b| {
                b.iter(|| libjpeg_turbo_decode_batch(&mut turbo, &input.bytes, 64));
            });
        }
    }
    wsi_tile_batch_rgb.finish();

    let mut wsi_region_rgb = c.benchmark_group("wsi_region_rgb");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        let roi = centered_roi(input.dimensions, 256);
        wsi_region_rgb.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_decode_region(&input.bytes, 256));
        });
        wsi_region_rgb.bench_function(format!("jpeg-decoder/{}", input.name), |b| {
            b.iter(|| jpeg_decoder_decode_region(&input.bytes, 256));
        });
        wsi_region_rgb.bench_function(format!("zune-jpeg/{}", input.name), |b| {
            b.iter(|| zune_decode_region(&input.bytes, 256));
        });
        if libjpeg_turbo_available() {
            let mut turbo = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
            wsi_region_rgb.bench_function(format!("libjpeg-turbo/{}", input.name), move |b| {
                b.iter(|| libjpeg_turbo_decode_region(&mut turbo, &input.bytes, roi));
            });
        }
    }
    wsi_region_rgb.finish();

    let mut wsi_scaled_rgb_q4 = c.benchmark_group("wsi_scaled_rgb_q4");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        wsi_scaled_rgb_q4.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_decode_scaled(&input.bytes, Downscale::Quarter));
        });
        wsi_scaled_rgb_q4.bench_function(format!("jpeg-decoder/{}", input.name), |b| {
            b.iter(|| jpeg_decoder_decode_scaled(&input.bytes, Downscale::Quarter));
        });
        wsi_scaled_rgb_q4.bench_function(format!("zune-jpeg/{}", input.name), |b| {
            b.iter(|| zune_decode_scaled(&input.bytes, Downscale::Quarter));
        });
        if libjpeg_turbo_available() {
            let mut turbo = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
            wsi_scaled_rgb_q4.bench_function(format!("libjpeg-turbo/{}", input.name), move |b| {
                b.iter(|| {
                    libjpeg_turbo_decode_scaled(&mut turbo, &input.bytes, Downscale::Quarter);
                });
            });
        }
    }
    wsi_scaled_rgb_q4.finish();

    let mut wsi_scaled_rgb_q8 = c.benchmark_group("wsi_scaled_rgb_q8");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        wsi_scaled_rgb_q8.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_decode_scaled(&input.bytes, Downscale::Eighth));
        });
        wsi_scaled_rgb_q8.bench_function(format!("jpeg-decoder/{}", input.name), |b| {
            b.iter(|| jpeg_decoder_decode_scaled(&input.bytes, Downscale::Eighth));
        });
        wsi_scaled_rgb_q8.bench_function(format!("zune-jpeg/{}", input.name), |b| {
            b.iter(|| zune_decode_scaled(&input.bytes, Downscale::Eighth));
        });
        if libjpeg_turbo_available() {
            let mut turbo = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
            wsi_scaled_rgb_q8.bench_function(format!("libjpeg-turbo/{}", input.name), move |b| {
                b.iter(|| libjpeg_turbo_decode_scaled(&mut turbo, &input.bytes, Downscale::Eighth));
            });
        }
    }
    wsi_scaled_rgb_q8.finish();

    let mut wsi_region_scaled_rgb_q4 = c.benchmark_group("wsi_region_scaled_rgb_q4");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        let roi = centered_roi(input.dimensions, 256);
        wsi_region_scaled_rgb_q4.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_decode_region_scaled(&input.bytes, 256, Downscale::Quarter));
        });
        wsi_region_scaled_rgb_q4.bench_function(format!("jpeg-decoder/{}", input.name), |b| {
            b.iter(|| {
                jpeg_decoder_decode_region_scaled(&input.bytes, 256, Downscale::Quarter);
            });
        });
        wsi_region_scaled_rgb_q4.bench_function(format!("zune-jpeg/{}", input.name), |b| {
            b.iter(|| zune_decode_region_scaled(&input.bytes, 256, Downscale::Quarter));
        });
        if libjpeg_turbo_available() {
            let mut turbo = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
            wsi_region_scaled_rgb_q4.bench_function(
                format!("libjpeg-turbo/{}", input.name),
                move |b| {
                    b.iter(|| {
                        libjpeg_turbo_decode_region_scaled(
                            &mut turbo,
                            &input.bytes,
                            roi,
                            Downscale::Quarter,
                        );
                    });
                },
            );
        }
    }
    wsi_region_scaled_rgb_q4.finish();

    let mut wsi_region_scaled_rgb_q8 = c.benchmark_group("wsi_region_scaled_rgb_q8");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        let roi = centered_roi(input.dimensions, 256);
        wsi_region_scaled_rgb_q8.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| slidecodec_decode_region_scaled(&input.bytes, 256, Downscale::Eighth));
        });
        wsi_region_scaled_rgb_q8.bench_function(format!("jpeg-decoder/{}", input.name), |b| {
            b.iter(|| {
                jpeg_decoder_decode_region_scaled(&input.bytes, 256, Downscale::Eighth);
            });
        });
        wsi_region_scaled_rgb_q8.bench_function(format!("zune-jpeg/{}", input.name), |b| {
            b.iter(|| zune_decode_region_scaled(&input.bytes, 256, Downscale::Eighth));
        });
        if libjpeg_turbo_available() {
            let mut turbo = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
            wsi_region_scaled_rgb_q8.bench_function(
                format!("libjpeg-turbo/{}", input.name),
                move |b| {
                    b.iter(|| {
                        libjpeg_turbo_decode_region_scaled(
                            &mut turbo,
                            &input.bytes,
                            roi,
                            Downscale::Eighth,
                        );
                    });
                },
            );
        }
    }
    wsi_region_scaled_rgb_q8.finish();

    let mut wsi_tile_batch_scaled_rgb_q4 = c.benchmark_group("wsi_tile_batch_scaled_rgb_q4");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb) {
        wsi_tile_batch_scaled_rgb_q4.bench_function(format!("slidecodec/{}", input.name), |b| {
            b.iter(|| {
                slidecodec_decode_tile_batch_scaled(&input.bytes, 64, Downscale::Quarter);
            });
        });
        wsi_tile_batch_scaled_rgb_q4.bench_function(format!("jpeg-decoder/{}", input.name), |b| {
            b.iter(|| jpeg_decoder_decode_batch_scaled(&input.bytes, 64, Downscale::Quarter));
        });
        wsi_tile_batch_scaled_rgb_q4.bench_function(format!("zune-jpeg/{}", input.name), |b| {
            b.iter(|| zune_decode_batch_scaled(&input.bytes, 64, Downscale::Quarter));
        });
        if libjpeg_turbo_available() {
            let mut turbo = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
            wsi_tile_batch_scaled_rgb_q4.bench_function(
                format!("libjpeg-turbo/{}", input.name),
                move |b| {
                    b.iter(|| {
                        libjpeg_turbo_decode_batch_scaled(
                            &mut turbo,
                            &input.bytes,
                            64,
                            Downscale::Quarter,
                        );
                    });
                },
            );
        }
    }
    wsi_tile_batch_scaled_rgb_q4.finish();

    let mut wsi_tile_batch_region_scaled_rgb_q4 =
        c.benchmark_group("wsi_tile_batch_region_scaled_rgb_q4");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb) {
        let roi = centered_roi(input.dimensions, 256);
        wsi_tile_batch_region_scaled_rgb_q4.bench_function(
            format!("slidecodec/{}", input.name),
            |b| {
                b.iter(|| {
                    slidecodec_decode_tile_batch_region_scaled(
                        &input.bytes,
                        64,
                        256,
                        Downscale::Quarter,
                    );
                });
            },
        );
        wsi_tile_batch_region_scaled_rgb_q4.bench_function(
            format!("jpeg-decoder/{}", input.name),
            |b| {
                b.iter(|| {
                    jpeg_decoder_decode_batch_region_scaled(
                        &input.bytes,
                        64,
                        256,
                        Downscale::Quarter,
                    );
                });
            },
        );
        wsi_tile_batch_region_scaled_rgb_q4.bench_function(
            format!("zune-jpeg/{}", input.name),
            |b| {
                b.iter(|| {
                    zune_decode_batch_region_scaled(&input.bytes, 64, 256, Downscale::Quarter);
                });
            },
        );
        if libjpeg_turbo_available() {
            let mut turbo = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
            wsi_tile_batch_region_scaled_rgb_q4.bench_function(
                format!("libjpeg-turbo/{}", input.name),
                move |b| {
                    b.iter(|| {
                        libjpeg_turbo_decode_batch_region_scaled(
                            &mut turbo,
                            &input.bytes,
                            64,
                            roi,
                            Downscale::Quarter,
                        );
                    });
                },
            );
        }
    }
    wsi_tile_batch_region_scaled_rgb_q4.finish();
}

criterion_group!(compare_benches, bench_compare);
criterion_main!(compare_benches);
