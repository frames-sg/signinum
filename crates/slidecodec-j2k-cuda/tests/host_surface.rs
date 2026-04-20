use slidecodec_core::{
    BackendRequest, CodecError, DeviceSubmission, DeviceSurface, ImageDecodeDevice,
    ImageDecodeSubmit, PixelFormat,
};
use slidecodec_j2k_cuda::{CudaSession, J2kDecoder};
use slidecodec_j2k_native::{encode, EncodeOptions};

fn fixture() -> Vec<u8> {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 2, 3, 8, false, &options).expect("encode")
}

#[test]
fn auto_falls_back_to_cpu_surface() {
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), slidecodec_core::BackendKind::Cpu);
    assert!(surface.as_host_bytes().is_some());
}

#[test]
fn explicit_cuda_request_reports_unavailable() {
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let error = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda)
        .expect_err("cuda unavailable");
    assert!(error.is_unsupported());
}

#[test]
fn submit_to_device_auto_falls_back_to_cpu_surface() {
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut session = CudaSession::default();
    let surface = <J2kDecoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
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
