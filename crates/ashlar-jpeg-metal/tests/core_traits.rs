#![cfg(target_os = "macos")]

use ashlar_core::{
    BackendKind, BackendRequest, DeviceSubmission, DeviceSurface, Downscale, ImageDecode,
    ImageDecodeDevice, ImageDecodeSubmit, PixelFormat, Rect,
};
use ashlar_jpeg_metal::{Decoder, MetalBackendSession, MetalSession, ScratchPool};

const BASELINE_420: &[u8] = include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");
const BASELINE_422: &[u8] = include_bytes!("../../../corpus/conformance/baseline_422_16x8.jpg");

#[test]
fn decode_to_metal_matches_cpu_decode_bytes() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut expected = <Decoder<'_> as ImageDecode<'_>>::from_view(
        <Decoder<'_> as ImageDecode<'_>>::parse(BASELINE_420).expect("view"),
    )
    .expect("decoder from view");
    let dims = expected.inner().info().dimensions;
    let stride = dims.0 as usize * 3;
    let mut host = vec![0u8; stride * dims.1 as usize];
    expected
        .decode_into(&mut host, stride, PixelFormat::Rgb8)
        .expect("cpu decode");

    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Metal)
        .expect("device decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), dims);
    assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
    assert_eq!(surface.byte_len(), host.len());
    assert_eq!(surface.as_bytes(), host.as_slice());
}

#[test]
fn cpu_device_request_stays_host_backed() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Cpu)
        .expect("cpu surface");
    assert_eq!(surface.backend_kind(), BackendKind::Cpu);
    assert_eq!(surface.pixel_format(), PixelFormat::Gray8);
}

#[test]
fn metal_surface_exposes_buffer_for_on_device_consumers() {
    let mut metal_decoder = Decoder::new(BASELINE_420).expect("metal decoder");
    let metal_surface = metal_decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Metal)
        .expect("metal surface");
    let (buffer, byte_offset) = metal_surface.metal_buffer().expect("metal buffer");
    assert_eq!(byte_offset, 0);
    let buffer_len = usize::try_from(buffer.length()).expect("metal buffer length fits usize");
    assert!(buffer_len >= metal_surface.byte_len());

    let mut cpu_decoder = Decoder::new(BASELINE_420).expect("cpu decoder");
    let cpu_surface = cpu_decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cpu)
        .expect("cpu surface");
    assert!(cpu_surface.metal_buffer().is_none());
}

#[cfg(target_os = "macos")]
#[test]
fn decode_to_device_with_session_uses_session_device() {
    use metal::foreign_types::ForeignTypeRef;

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut decoder = Decoder::new(BASELINE_420).expect("metal decoder");

    let surface = decoder
        .decode_to_device_with_session(PixelFormat::Rgb8, &session)
        .expect("session decode");

    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    let (buffer, _) = surface.metal_buffer().expect("metal buffer");
    assert_eq!(buffer.device().as_ptr(), session.device().as_ptr());
}

#[test]
fn fast422_decode_to_metal_matches_cpu_decode_bytes() {
    let mut decoder = Decoder::new(BASELINE_422).expect("decoder");
    let mut expected = <Decoder<'_> as ImageDecode<'_>>::from_view(
        <Decoder<'_> as ImageDecode<'_>>::parse(BASELINE_422).expect("view"),
    )
    .expect("decoder from view");
    let dims = expected.inner().info().dimensions;
    let stride = dims.0 as usize * 3;
    let mut host = vec![0u8; stride * dims.1 as usize];
    expected
        .decode_into(&mut host, stride, PixelFormat::Rgb8)
        .expect("cpu decode");

    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Metal)
        .expect("device decode");

    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), dims);
    assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
    assert_eq!(surface.as_bytes(), host.as_slice());
}

#[test]
fn region_and_scaled_metal_bytes_match_cpu_decode() {
    let roi = ashlar_core::Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };

    let mut metal_decoder = Decoder::new(BASELINE_420).expect("metal decoder");
    let region_surface = metal_decoder
        .decode_region_to_device(PixelFormat::Rgb8, roi, BackendRequest::Metal)
        .expect("region surface");

    let mut cpu_decoder = Decoder::new(BASELINE_420).expect("cpu decoder");
    let mut region_host = vec![0u8; roi.w as usize * roi.h as usize * 3];
    cpu_decoder
        .decode_region_into(
            &mut ScratchPool::new(),
            &mut region_host,
            roi.w as usize * 3,
            PixelFormat::Rgb8,
            roi,
        )
        .expect("cpu region");
    assert_eq!(region_surface.as_bytes(), region_host.as_slice());

    let scaled_surface = metal_decoder
        .decode_scaled_to_device(
            PixelFormat::Rgb8,
            ashlar_core::Downscale::Quarter,
            BackendRequest::Metal,
        )
        .expect("scaled surface");
    let mut scaled_host = vec![0u8; 4 * 4 * 3];
    cpu_decoder
        .decode_scaled_into(
            &mut ScratchPool::new(),
            &mut scaled_host,
            4 * 3,
            PixelFormat::Rgb8,
            ashlar_core::Downscale::Quarter,
        )
        .expect("cpu scaled");
    assert_eq!(scaled_surface.as_bytes(), scaled_host.as_slice());
}

#[test]
fn region_scaled_metal_bytes_match_cpu_decode() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let scale = Downscale::Quarter;

    let mut metal_decoder = Decoder::new(BASELINE_420).expect("metal decoder");
    let surface = metal_decoder
        .decode_region_scaled_to_device(PixelFormat::Rgb8, roi, scale, BackendRequest::Metal)
        .expect("region scaled surface");

    let cpu_decoder = Decoder::new(BASELINE_420).expect("cpu decoder");
    let denom = scale.denominator();
    let scaled = Rect {
        x: roi.x / denom,
        y: roi.y / denom,
        w: (roi.x + roi.w).div_ceil(denom) - roi.x / denom,
        h: (roi.y + roi.h).div_ceil(denom) - roi.y / denom,
    };
    let mut host = vec![0u8; scaled.w as usize * scaled.h as usize * 3];
    cpu_decoder
        .inner()
        .decode_region_scaled_into(
            &mut host,
            scaled.w as usize * 3,
            PixelFormat::Rgb8,
            ashlar_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        )
        .expect("cpu region scaled");

    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
    assert_eq!(surface.as_bytes(), host.as_slice());
}

#[test]
fn region_scaled_submit_trait_returns_metal_surface() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let scale = Downscale::Quarter;
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut session = MetalSession::default();

    let surface = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_region_scaled_to_device(
        &mut decoder,
        &mut session,
        PixelFormat::Rgb8,
        roi,
        scale,
        BackendRequest::Metal,
    )
    .expect("submission")
    .wait()
    .expect("surface");

    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (3, 3));
    assert!(session.submissions() >= 1);
}

#[test]
fn submit_to_device_returns_surface_and_updates_session() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut session = MetalSession::default();
    let submission = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut decoder,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Metal,
    )
    .expect("submission");
    let surface = submission.wait().expect("surface");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert!(session.submissions() >= 1);
}

#[test]
fn multiple_submits_share_one_session_flush() {
    let mut session = MetalSession::default();
    let mut a = Decoder::new(BASELINE_420).expect("decoder a");
    let mut b = Decoder::new(BASELINE_420).expect("decoder b");

    let first = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut a,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Metal,
    )
    .expect("submit a");
    let second = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut b,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Metal,
    )
    .expect("submit b");

    let second_surface = second.wait().expect("wait b");
    let first_surface = first.wait().expect("wait a");

    assert_eq!(second_surface.backend_kind(), BackendKind::Metal);
    assert_eq!(first_surface.backend_kind(), BackendKind::Metal);
    assert_eq!(session.submissions(), 1);
}
