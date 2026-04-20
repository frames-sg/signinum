use slidecodec_core::{
    BackendRequest, CodecError, DeviceSubmission, DeviceSurface, ImageDecodeDevice,
    ImageDecodeSubmit, PixelFormat,
};
use slidecodec_jpeg_cuda::{CudaSession, Decoder};

const BASELINE_420: &[u8] = include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");

#[test]
fn auto_falls_back_to_cpu_surface() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), slidecodec_core::BackendKind::Cpu);
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
    assert_eq!(surface.backend_kind(), slidecodec_core::BackendKind::Cpu);
    assert!(surface.as_host_bytes().is_some());
    assert!(session.submissions() >= 1);
}
