// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use signinum_j2k_cuda::CudaEncodeStageAccelerator;
use signinum_j2k_native::{J2kEncodeStageAccelerator, J2kForwardDwt53Job, J2kForwardRctJob};

const BENCH_DIMS: &[u32] = &[512, 1024, 2048];

fn bench_encode_stages(c: &mut Criterion) {
    let cuda_available = cuda_encode_available();

    let mut rct = c.benchmark_group("j2k_cuda_forward_rct");
    for &dim in BENCH_DIMS {
        let pixels = generate_rgb_planes(dim, dim);
        rct.bench_with_input(BenchmarkId::new("cpu", dim), &pixels, |b, planes| {
            b.iter(|| {
                let (mut plane0, mut plane1, mut plane2) = clone_planes(planes);
                cpu_forward_rct(&mut plane0, &mut plane1, &mut plane2);
                (plane0, plane1, plane2)
            });
        });

        if cuda_available {
            rct.bench_with_input(BenchmarkId::new("cuda", dim), &pixels, |b, planes| {
                b.iter(|| {
                    let (mut plane0, mut plane1, mut plane2) = clone_planes(planes);
                    let mut accelerator = CudaEncodeStageAccelerator::default();
                    let dispatched = accelerator
                        .encode_forward_rct(J2kForwardRctJob {
                            plane0: &mut plane0,
                            plane1: &mut plane1,
                            plane2: &mut plane2,
                        })
                        .expect("CUDA forward RCT");
                    assert!(dispatched, "CUDA forward RCT did not dispatch");
                    (plane0, plane1, plane2)
                });
            });
        }
    }
    rct.finish();

    let mut dwt = c.benchmark_group("j2k_cuda_forward_dwt53");
    for &dim in BENCH_DIMS {
        let samples = generate_gray_plane(dim, dim);
        dwt.bench_with_input(BenchmarkId::new("cpu", dim), &samples, |b, samples| {
            b.iter(|| cpu_forward_dwt53(samples, dim, dim, 1));
        });

        if cuda_available {
            dwt.bench_with_input(BenchmarkId::new("cuda", dim), &samples, |b, samples| {
                b.iter(|| {
                    let mut accelerator = CudaEncodeStageAccelerator::default();
                    let output = accelerator
                        .encode_forward_dwt53(J2kForwardDwt53Job {
                            samples,
                            width: dim,
                            height: dim,
                            num_levels: 1,
                        })
                        .expect("CUDA forward DWT 5/3")
                        .expect("CUDA forward DWT 5/3 dispatch");
                    assert_eq!(output.ll_width, dim / 2);
                    output
                });
            });
        }
    }
    dwt.finish();
}

fn cuda_encode_available() -> bool {
    let mut plane0 = vec![0.0; 64];
    let mut plane1 = vec![1.0; 64];
    let mut plane2 = vec![2.0; 64];
    let mut accelerator = CudaEncodeStageAccelerator::default();
    match accelerator.encode_forward_rct(J2kForwardRctJob {
        plane0: &mut plane0,
        plane1: &mut plane1,
        plane2: &mut plane2,
    }) {
        Ok(true) => true,
        Ok(false) if std::env::var_os("SIGNINUM_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("SIGNINUM_REQUIRE_CUDA_BENCH is set but CUDA encode did not dispatch")
        }
        Ok(false) => {
            eprintln!("skipping CUDA encode benches: CUDA runtime feature is disabled");
            false
        }
        Err(error) if std::env::var_os("SIGNINUM_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("SIGNINUM_REQUIRE_CUDA_BENCH is set but CUDA encode failed: {error}")
        }
        Err(error) => {
            eprintln!("skipping CUDA encode benches: {error}");
            false
        }
    }
}

fn generate_rgb_planes(width: u32, height: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = width as usize * height as usize;
    let mut plane0 = Vec::with_capacity(len);
    let mut plane1 = Vec::with_capacity(len);
    let mut plane2 = Vec::with_capacity(len);
    for y in 0..height {
        for x in 0..width {
            plane0.push(centered_sample(x * 13 + y * 3));
            plane1.push(centered_sample(x * 5 + y * 11 + (x ^ y)));
            plane2.push(centered_sample(x * 7 + y * 17 + x.wrapping_mul(y) / 31));
        }
    }
    (plane0, plane1, plane2)
}

fn generate_gray_plane(width: u32, height: u32) -> Vec<f32> {
    let len = width as usize * height as usize;
    let mut samples = Vec::with_capacity(len);
    for y in 0..height {
        for x in 0..width {
            samples.push(centered_sample(x * 9 + y * 15 + x.wrapping_mul(y) / 17));
        }
    }
    samples
}

fn centered_sample(value: u32) -> f32 {
    f32::from(u8::try_from(value & 0xff).expect("masked sample fits in u8")) - 128.0
}

fn clone_planes(planes: &(Vec<f32>, Vec<f32>, Vec<f32>)) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (planes.0.clone(), planes.1.clone(), planes.2.clone())
}

fn cpu_forward_rct(plane0: &mut [f32], plane1: &mut [f32], plane2: &mut [f32]) {
    for ((r, g), b) in plane0
        .iter_mut()
        .zip(plane1.iter_mut())
        .zip(plane2.iter_mut())
    {
        let original_r = *r;
        let original_g = *g;
        let original_b = *b;
        *r = ((original_r + 2.0 * original_g + original_b) * 0.25).floor();
        *g = original_b - original_g;
        *b = original_r - original_g;
    }
}

fn cpu_forward_dwt53(samples: &[f32], width: u32, height: u32, num_levels: u8) -> Vec<f32> {
    let full_width = width as usize;
    let mut buffer = samples.to_vec();
    let mut current_width = width as usize;
    let mut current_height = height as usize;

    for _ in 0..num_levels {
        if current_width < 2 && current_height < 2 {
            break;
        }
        if current_width >= 2 {
            let mut row = vec![0.0; current_width];
            for y in 0..current_height {
                let start = y * full_width;
                row.copy_from_slice(&buffer[start..start + current_width]);
                forward_lift_53(&mut row);
                let low_width = current_width.div_ceil(2);
                for i in 0..low_width {
                    buffer[start + i] = row[i * 2];
                }
                for i in 0..(current_width / 2) {
                    buffer[start + low_width + i] = row[i * 2 + 1];
                }
            }
        }
        if current_height >= 2 {
            let mut col = vec![0.0; current_height];
            for x in 0..current_width {
                for y in 0..current_height {
                    col[y] = buffer[y * full_width + x];
                }
                forward_lift_53(&mut col);
                let low_height = current_height.div_ceil(2);
                for i in 0..low_height {
                    buffer[i * full_width + x] = col[i * 2];
                }
                for i in 0..(current_height / 2) {
                    buffer[(low_height + i) * full_width + x] = col[i * 2 + 1];
                }
            }
        }
        current_width = current_width.div_ceil(2);
        current_height = current_height.div_ceil(2);
    }

    buffer
}

fn forward_lift_53(data: &mut [f32]) {
    let n = data.len();
    if n < 2 {
        return;
    }

    let last_even = if n.is_multiple_of(2) { n - 2 } else { n - 1 };
    for i in (1..n).step_by(2) {
        let left = data[i - 1];
        let right = if i + 1 < n {
            data[i + 1]
        } else {
            data[last_even]
        };
        data[i] -= ((left + right) * 0.5).floor();
    }

    for i in (0..n).step_by(2) {
        let left = if i > 0 { data[i - 1] } else { data[1] };
        let right = if i + 1 < n { data[i + 1] } else { left };
        data[i] += ((left + right) * 0.25 + 0.5).floor();
    }
}

criterion_group!(benches, bench_encode_stages);
criterion_main!(benches);
