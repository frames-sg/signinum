// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};
use signinum_core::{BackendRequest, DeviceSurface, ImageDecodeDevice, PixelFormat};
use signinum_jpeg::Decoder as CpuDecoder;
use signinum_jpeg_cuda::Decoder as CudaDecoder;

const DEFAULT_JPEG: &[u8] = include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg");

fn bench_device_decode(c: &mut Criterion) {
    let input = bench_input();
    let mut group = c.benchmark_group("jpeg_cuda_device_decode");

    group.bench_function("cpu_decode_rgb8", |b| {
        b.iter(|| {
            let decoder = CpuDecoder::new(&input).expect("cpu decoder");
            decoder.decode(PixelFormat::Rgb8).expect("cpu decode")
        });
    });

    match cuda_probe(&input) {
        Some(probe) => {
            let label = if probe.used_hardware_decode {
                "cuda_nvjpeg_rgb8_surface"
            } else {
                "cuda_upload_fallback_rgb8_surface"
            };
            group.bench_function(label, |b| {
                b.iter(|| {
                    let mut decoder = CudaDecoder::new(&input).expect("cuda decoder");
                    decoder
                        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda)
                        .expect("cuda decode")
                });
            });

            let label = if probe.used_hardware_decode {
                "cuda_nvjpeg_rgb8_download"
            } else {
                "cuda_upload_fallback_rgb8_download"
            };
            group.bench_function(label, |b| {
                b.iter(|| {
                    let mut decoder = CudaDecoder::new(&input).expect("cuda decoder");
                    let surface = decoder
                        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda)
                        .expect("cuda decode");
                    let mut out = vec![0u8; surface.byte_len()];
                    surface
                        .download_into(&mut out, surface.pitch_bytes())
                        .expect("cuda download");
                    out
                });
            });
        }
        None if std::env::var_os("SIGNINUM_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("SIGNINUM_REQUIRE_CUDA_BENCH is set but CUDA decode is unavailable")
        }
        None => {
            eprintln!("skipping CUDA decode benches: CUDA runtime is unavailable");
        }
    }

    group.finish();
}

fn bench_input() -> Vec<u8> {
    match std::env::var_os("SIGNINUM_CUDA_BENCH_JPEG") {
        Some(path) => std::fs::read(&path).unwrap_or_else(|error| {
            panic!(
                "failed to read SIGNINUM_CUDA_BENCH_JPEG={}: {error}",
                path.to_string_lossy()
            )
        }),
        None => DEFAULT_JPEG.to_vec(),
    }
}

struct CudaProbe {
    used_hardware_decode: bool,
}

fn cuda_probe(input: &[u8]) -> Option<CudaProbe> {
    let mut decoder = CudaDecoder::new(input).expect("cuda decoder");
    let surface = match decoder.decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda) {
        Ok(surface) => surface,
        Err(error) => {
            eprintln!("skipping CUDA decode benches: {error}");
            return None;
        }
    };
    let stats = surface.cuda_surface().expect("cuda surface").stats();
    if std::env::var_os("SIGNINUM_REQUIRE_CUDA_JPEG_HARDWARE_DECODE").is_some()
        && !stats.used_hardware_decode()
    {
        panic!("SIGNINUM_REQUIRE_CUDA_JPEG_HARDWARE_DECODE is set but nvJPEG was not used");
    }
    Some(CudaProbe {
        used_hardware_decode: stats.used_hardware_decode(),
    })
}

criterion_group!(benches, bench_device_decode);
criterion_main!(benches);
