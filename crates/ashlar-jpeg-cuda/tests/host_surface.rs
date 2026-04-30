use ashlar_core::{
    BackendRequest, CodecError, DeviceSubmission, DeviceSurface, ImageDecodeDevice,
    ImageDecodeSubmit, PixelFormat, Rect,
};
use ashlar_jpeg_cuda::{CudaSession, Decoder, Error};

const BASELINE_420: &[u8] = include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");

#[test]
fn auto_falls_back_to_cpu_surface() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), ashlar_core::BackendKind::Cpu);
    assert!(surface.as_host_bytes().is_some());
}

#[test]
fn explicit_cuda_request_reports_unavailable() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let error = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda)
        .expect_err("cuda unavailable");
    assert!(error.is_unsupported());
}

#[test]
fn explicit_cuda_request_short_circuits_before_decode_validation() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");

    let error = decoder
        .decode_to_device(PixelFormat::Rgba16, BackendRequest::Cuda)
        .expect_err("cuda unavailable");
    assert!(matches!(error, Error::CudaUnavailable));

    let error = decoder
        .decode_region_to_device(
            PixelFormat::Rgb8,
            Rect {
                x: 1000,
                y: 1000,
                w: 1000,
                h: 1000,
            },
            BackendRequest::Cuda,
        )
        .expect_err("cuda unavailable");
    assert!(matches!(error, Error::CudaUnavailable));
}

#[test]
fn submit_to_device_auto_falls_back_to_cpu_surface() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut session = CudaSession::default();
    let surface = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut decoder,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Auto,
    )
    .expect("submission")
    .wait()
    .expect("surface");
    assert_eq!(surface.backend_kind(), ashlar_core::BackendKind::Cpu);
    assert!(surface.as_host_bytes().is_some());
    assert!(session.submissions() >= 1);
}
