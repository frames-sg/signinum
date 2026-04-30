use ashlar_core::{
    BackendRequest, CodecError, DeviceSubmission, DeviceSurface, Downscale, ImageDecode,
    ImageDecodeDevice, ImageDecodeSubmit, PixelFormat, Rect,
};
use ashlar_j2k_cuda::{CudaSession, Error, J2kDecoder};
use ashlar_j2k_native::{encode, encode_htj2k, EncodeOptions};

fn fixture() -> Vec<u8> {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 2, 3, 8, false, &options).expect("encode")
}

fn fixture_ht_gray8() -> Vec<u8> {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, 4, 4, 1, 8, false, &options).expect("encode ht gray8")
}

#[test]
fn auto_falls_back_to_cpu_surface() {
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), ashlar_core::BackendKind::Cpu);
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
fn explicit_cuda_request_short_circuits_before_decode_validation() {
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");

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

    let error = decoder
        .decode_region_scaled_to_device(
            PixelFormat::Rgb8,
            Rect {
                x: 1000,
                y: 1000,
                w: 1000,
                h: 1000,
            },
            Downscale::Half,
            BackendRequest::Cuda,
        )
        .expect_err("cuda unavailable");
    assert!(matches!(error, Error::CudaUnavailable));
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
    assert_eq!(surface.backend_kind(), ashlar_core::BackendKind::Cpu);
    assert!(surface.as_host_bytes().is_some());
    assert!(session.submissions() >= 1);
}

#[test]
fn auto_classic_full_frame_surface_matches_host_decode() {
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), ashlar_core::BackendKind::Cpu);

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 12];
    host_decoder
        .decode_into(&mut host, 6, PixelFormat::Rgb8)
        .expect("host decode");
    assert_eq!(surface.as_host_bytes(), Some(host.as_slice()));
}

#[test]
fn auto_htj2k_full_frame_surface_matches_host_decode() {
    let bytes = fixture_ht_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), ashlar_core::BackendKind::Cpu);

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 16];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray8)
        .expect("host decode");
    assert_eq!(surface.as_host_bytes(), Some(host.as_slice()));
}

#[test]
fn auto_region_scaled_surface_matches_host_decode() {
    let bytes = fixture_ht_gray8();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_region_scaled_to_device(PixelFormat::Gray8, roi, scale, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), ashlar_core::BackendKind::Cpu);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = vec![0u8; scaled.w as usize * scaled.h as usize];
    host_decoder
        .decode_region_scaled_into(
            &mut ashlar_j2k_cuda::J2kScratchPool::new(),
            &mut host,
            scaled.w as usize,
            PixelFormat::Gray8,
            roi,
            scale,
        )
        .expect("host decode");
    assert_eq!(surface.as_host_bytes(), Some(host.as_slice()));
}
