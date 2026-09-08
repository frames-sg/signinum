// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use jpeg_encoder::SamplingFactor::{F_1_1, F_2_1, F_2_2};

#[test]
#[ignore = "allocation diagnostic; run explicitly with --ignored --nocapture"]
fn metal_single_session_buffer_allocation_report() {
    single_session_buffer_allocations(false);
}

#[test]
fn warm_single_session_reuses_temporary_buffers() {
    single_session_buffer_allocations(true);
}

fn single_session_buffer_allocations(require_reuse: bool) {
    if !should_run_metal_runtime() {
        return;
    }
    for (sampling, input) in [
        ("420", BASELINE_420),
        ("420-restart", BASELINE_420_RESTART),
        ("422", BASELINE_422),
        ("444", BASELINE_444),
    ] {
        for fmt in [PixelFormat::Gray8, PixelFormat::Rgb8, PixelFormat::Rgba8] {
            let session = MetalBackendSession::system_default().expect("session");
            // Exclude one-time runtime resources from per-decode allocation counts.
            session.runtime_result().as_ref().expect("runtime");
            let mut decoder = Decoder::new(input).expect("decoder");
            let (expected, _) = CpuDecoder::new(input)
                .expect("native decoder")
                .decode_request(DecodeRequest::full(fmt))
                .expect("native pixels");
            for iteration in 0..3 {
                compute::reset_jpeg_private_buffer_allocations_for_test();
                compute::reset_jpeg_shared_buffer_allocations_for_test();
                let surface = decoder
                    .decode_to_device_with_session(fmt, &session)
                    .expect("single decode");
                let private = compute::jpeg_private_buffer_allocations_for_test();
                let shared = compute::jpeg_shared_buffer_allocations_for_test();
                assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
                if require_reuse && iteration > 0 {
                    assert_eq!(
                        private, 0,
                        "warm {sampling}/{fmt:?} temporary private buffers"
                    );
                    assert_eq!(
                        shared, 1,
                        "warm {sampling}/{fmt:?} retains only fresh shared output allocation"
                    );
                }
                let mut actual = vec![0; expected.len()];
                let stride = surface.dimensions().0 as usize * fmt.bytes_per_pixel();
                surface.download_into(&mut actual, stride).expect("pixels");
                assert_eq!(actual, expected);
                println!("metal_single_session_allocations sampling={sampling} format={fmt:?} iteration={iteration} private_buffers={private} shared_buffers={shared} returned_surfaces=1");
            }
        }
    }
}

fn distinct_jpeg(side: u16, variant: u8, sampling: jpeg_encoder::SamplingFactor) -> Vec<u8> {
    let mut pixels = j2k_test_support::gpu_bench_rgb8(u32::from(side), u32::from(side));
    for sample in &mut pixels {
        *sample = sample.wrapping_add(variant.wrapping_mul(37));
    }
    let mut bytes = Vec::new();
    let mut encoder = jpeg_encoder::Encoder::new(&mut bytes, 90);
    encoder.set_sampling_factor(sampling);
    encoder
        .encode(&pixels, side, side, jpeg_encoder::ColorType::Rgb)
        .unwrap();
    bytes
}

fn check_retained_outputs(retained: &[(Surface, Vec<u8>)]) {
    for (surface, expected) in retained {
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        // First readback is deliberately after all subsequent decodes. Cached
        // host bytes must not hide an overwritten device allocation.
        let actual = surface.as_bytes().expect("retained surface pixels");
        assert!(
            actual.as_ref() == expected.as_slice(),
            "retained device output changed"
        );
    }
}

#[test]
fn single_surfaces_survive_distinct_size_and_sampling_reuse() {
    if !should_run_metal_runtime() {
        return;
    }
    let session = MetalBackendSession::system_default().unwrap();
    let mut retained = Vec::new();
    for (side, variant, sampling) in [
        (64, 0, F_2_2),
        (64, 1, F_2_2),
        (32, 2, F_2_1),
        (48, 3, F_1_1),
        (16, 4, F_2_2),
        (64, 5, F_2_2),
    ] {
        let bytes = distinct_jpeg(side, variant, sampling);
        let mut decoder = Decoder::new(&bytes).unwrap();
        for fmt in [PixelFormat::Gray8, PixelFormat::Rgb8, PixelFormat::Rgba8] {
            let (expected, _) = CpuDecoder::new(&bytes)
                .unwrap()
                .decode_request(DecodeRequest::full(fmt))
                .unwrap();
            let surface = decoder
                .decode_to_device_with_session(fmt, &session)
                .unwrap();
            retained.push((surface, expected));
        }
    }
    check_retained_outputs(&retained);
}

#[test]
fn partial_single_surfaces_survive_operation_and_size_reuse() {
    if !should_run_metal_runtime() {
        return;
    }
    let mut retained = Vec::new();
    for (side, variant) in [(64, 1), (32, 2), (48, 3), (64, 4)] {
        let bytes = distinct_jpeg(side, variant, jpeg_encoder::SamplingFactor::F_1_1);
        let mut decoder = Decoder::new(&bytes).unwrap();
        let roi = Rect {
            x: 8,
            y: 8,
            w: u32::from(side) - 16,
            h: u32::from(side) - 16,
        };
        let native_roi = j2k_jpeg::Rect {
            x: roi.x,
            y: roi.y,
            w: roi.w,
            h: roi.h,
        };
        for fmt in [PixelFormat::Gray8, PixelFormat::Rgb8, PixelFormat::Rgba8] {
            for (request, native) in [
                (
                    MetalDecodeRequest::full(fmt, BackendRequest::Metal),
                    DecodeRequest::full(fmt),
                ),
                (
                    MetalDecodeRequest::scaled(fmt, Downscale::Half, BackendRequest::Metal),
                    DecodeRequest::scaled(fmt, Downscale::Half),
                ),
                (
                    MetalDecodeRequest::region(fmt, roi, BackendRequest::Metal),
                    DecodeRequest::region(fmt, native_roi),
                ),
                (
                    MetalDecodeRequest::region_scaled(
                        fmt,
                        roi,
                        Downscale::Quarter,
                        BackendRequest::Metal,
                    ),
                    DecodeRequest::region_scaled(fmt, native_roi, Downscale::Quarter),
                ),
            ] {
                let (expected, _) = CpuDecoder::new(&bytes)
                    .unwrap()
                    .decode_request(native)
                    .unwrap();
                let surface = decoder.decode_request_to_device(request).unwrap();
                retained.push((surface, expected));
            }
        }
    }
    check_retained_outputs(&retained);
}

#[test]
fn simultaneous_single_calls_keep_independent_surfaces() {
    if !should_run_metal_runtime() {
        return;
    }
    let session = MetalBackendSession::system_default().unwrap();
    session.runtime_result().as_ref().expect("runtime");
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        for variant in [1, 2] {
            let session = &session;
            let barrier = &barrier;
            scope.spawn(move || {
                let bytes = distinct_jpeg(64, variant, jpeg_encoder::SamplingFactor::F_2_2);
                let mut decoder = Decoder::new(&bytes).unwrap();
                let mut retained = Vec::new();
                barrier.wait();
                for fmt in [PixelFormat::Rgb8, PixelFormat::Gray8, PixelFormat::Rgba8] {
                    let (expected, _) = CpuDecoder::new(&bytes)
                        .unwrap()
                        .decode_request(DecodeRequest::full(fmt))
                        .unwrap();
                    let surface = decoder.decode_to_device_with_session(fmt, session).unwrap();
                    retained.push((surface, expected));
                }
                check_retained_outputs(&retained);
            });
        }
    });
}
