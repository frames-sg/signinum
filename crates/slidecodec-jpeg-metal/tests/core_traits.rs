use slidecodec_core::{
    BackendKind, BackendRequest, DeviceSubmission, DeviceSurface, ImageDecode, ImageDecodeDevice,
    ImageDecodeSubmit, PixelFormat,
};
use slidecodec_jpeg_metal::{Decoder, MetalSession, ScratchPool};

const BASELINE_420: &[u8] = include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");

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
fn region_and_scaled_metal_bytes_match_cpu_decode() {
    let roi = slidecodec_core::Rect {
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
            slidecodec_core::Downscale::Quarter,
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
            slidecodec_core::Downscale::Quarter,
        )
        .expect("cpu scaled");
    assert_eq!(scaled_surface.as_bytes(), scaled_host.as_slice());
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
