// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use signinum_core::PixelFormat;
use signinum_j2k::{
    encode_j2k_lossless, EncodeBackendPreference, J2kEncodeValidation, J2kLosslessEncodeOptions,
    J2kLosslessSamples,
};
use signinum_j2k_metal::{
    encode_lossless_from_padded_metal_buffer_with_report, MetalBackendSession,
    MetalEncodeStageAccelerator, MetalLosslessEncodeTile,
};
use signinum_j2k_native::{J2kEncodeStageAccelerator, J2kForwardDwt53Job, J2kForwardRctJob};

const BENCH_DIMS: &[u32] = &[512, 1024, 2048];
const ENCODE_BENCH_DIMS: &[u32] = &[512, 1024];

fn bench_encode_stages(c: &mut Criterion) {
    let mut rct = c.benchmark_group("j2k_metal_forward_rct");
    for &dim in BENCH_DIMS {
        let pixels = generate_rgb_planes(dim, dim);
        rct.bench_with_input(BenchmarkId::new("cpu", dim), &pixels, |b, planes| {
            b.iter(|| {
                let (mut plane0, mut plane1, mut plane2) = clone_planes(planes);
                cpu_forward_rct(&mut plane0, &mut plane1, &mut plane2);
                (plane0, plane1, plane2)
            });
        });

        if metal_encode_available() {
            rct.bench_with_input(BenchmarkId::new("metal", dim), &pixels, |b, planes| {
                let mut accelerator = MetalEncodeStageAccelerator::default();
                b.iter(|| {
                    let (mut plane0, mut plane1, mut plane2) = clone_planes(planes);
                    let dispatched = accelerator
                        .encode_forward_rct(J2kForwardRctJob {
                            plane0: &mut plane0,
                            plane1: &mut plane1,
                            plane2: &mut plane2,
                        })
                        .expect("Metal forward RCT");
                    assert!(dispatched, "Metal forward RCT did not dispatch");
                    (plane0, plane1, plane2)
                });
            });
        }
    }
    rct.finish();

    let mut dwt = c.benchmark_group("j2k_metal_forward_dwt53");
    for &dim in BENCH_DIMS {
        let samples = generate_gray_plane(dim, dim);
        dwt.bench_with_input(BenchmarkId::new("cpu", dim), &samples, |b, samples| {
            b.iter(|| cpu_forward_dwt53(samples, dim, dim, 1));
        });

        if metal_encode_available() {
            dwt.bench_with_input(BenchmarkId::new("metal", dim), &samples, |b, samples| {
                let mut accelerator = MetalEncodeStageAccelerator::default();
                b.iter(|| {
                    let output = accelerator
                        .encode_forward_dwt53(J2kForwardDwt53Job {
                            samples,
                            width: dim,
                            height: dim,
                            num_levels: 1,
                        })
                        .expect("Metal forward DWT 5/3")
                        .expect("Metal forward DWT 5/3 dispatch");
                    assert_eq!(output.ll_width, dim / 2);
                    output
                });
            });
        }
    }
    dwt.finish();

    let mut encode = c.benchmark_group("j2k_metal_lossless_rgb8_encode");
    for &dim in ENCODE_BENCH_DIMS {
        let pixels = generate_rgb8_pixels(dim, dim);
        let cpu_options = J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::CpuOnly,
            validation: J2kEncodeValidation::External,
            ..J2kLosslessEncodeOptions::default()
        };
        encode.bench_with_input(BenchmarkId::new("cpu", dim), &pixels, |b, pixels| {
            b.iter(|| {
                let samples = J2kLosslessSamples::new(pixels, dim, dim, 3, 8, false)
                    .expect("valid RGB8 samples");
                encode_j2k_lossless(samples, &cpu_options).expect("CPU J2K lossless encode")
            });
        });

        #[cfg(target_os = "macos")]
        if metal_encode_available() {
            let session = MetalBackendSession::system_default().expect("Metal session");
            let buffer = private_buffer_with_bytes(&session, &pixels);
            let metal_options = J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::RequireDevice,
                validation: J2kEncodeValidation::External,
                ..J2kLosslessEncodeOptions::default()
            };
            encode.bench_with_input(BenchmarkId::new("resident_metal", dim), &pixels, |b, _| {
                b.iter(|| {
                    let encoded = encode_lossless_from_padded_metal_buffer_with_report(
                        MetalLosslessEncodeTile {
                            buffer: &buffer,
                            byte_offset: 0,
                            width: dim,
                            height: dim,
                            pitch_bytes: dim as usize * 3,
                            output_width: dim,
                            output_height: dim,
                            format: PixelFormat::Rgb8,
                        },
                        &metal_options,
                        &session,
                    )
                    .expect("resident Metal J2K lossless encode");
                    assert!(encoded.resident.coefficient_prep_used);
                    assert!(encoded.resident.packetization_used);
                    assert!(encoded.resident.codestream_assembly_used);
                    encoded.encoded
                });
            });
        }
    }
    encode.finish();
}

fn metal_encode_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        metal::Device::system_default().is_some()
    }
    #[cfg(not(target_os = "macos"))]
    {
        assert!(
            std::env::var_os("SIGNINUM_REQUIRE_METAL_BENCH").is_none(),
            "SIGNINUM_REQUIRE_METAL_BENCH is set but this is not a Metal host"
        );
        false
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

fn generate_rgb8_pixels(width: u32, height: u32) -> Vec<u8> {
    let len = width as usize * height as usize * 3;
    let mut pixels = Vec::with_capacity(len);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 3 + y * 5) & 0xff) as u8);
            pixels.push(((x * 7 + y * 11 + (x ^ y)) & 0xff) as u8);
            pixels.push(((x * 13 + y * 17 + x.wrapping_mul(y) / 31) & 0xff) as u8);
        }
    }
    pixels
}

#[cfg(target_os = "macos")]
fn private_buffer_with_bytes(session: &MetalBackendSession, bytes: &[u8]) -> metal::Buffer {
    let upload = session.device().new_buffer_with_data(
        bytes.as_ptr().cast(),
        bytes.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let private = session.device().new_buffer(
        bytes.len() as u64,
        metal::MTLResourceOptions::StorageModePrivate,
    );
    let queue = session.device().new_command_queue();
    let command_buffer = queue.new_command_buffer();
    let blit = command_buffer.new_blit_command_encoder();
    blit.copy_from_buffer(&upload, 0, &private, 0, bytes.len() as u64);
    blit.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    private
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
