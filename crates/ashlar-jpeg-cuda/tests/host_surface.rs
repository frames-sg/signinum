use ashlar_core::{
    BackendRequest, CodecError, DecoderContext, DeviceSubmission, DeviceSurface, Downscale,
    ImageDecode, ImageDecodeDevice, ImageDecodeSubmit, PixelFormat, Rect, TileBatchDecodeDevice,
};
use ashlar_jpeg_cuda::{Codec, CudaSession, Decoder, Error};

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

#[test]
fn auto_region_scaled_surface_matches_host_decode() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);

    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let surface = decoder
        .decode_region_scaled_to_device(PixelFormat::Rgb8, roi, scale, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), ashlar_core::BackendKind::Cpu);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));

    let mut host_decoder = Decoder::new(BASELINE_420).expect("host decoder");
    let mut host = vec![0u8; scaled.w as usize * scaled.h as usize * 3];
    host_decoder
        .decode_region_scaled_into(
            &mut ashlar_jpeg::ScratchPool::new(),
            &mut host,
            scaled.w as usize * 3,
            PixelFormat::Rgb8,
            roi,
            scale,
        )
        .expect("host decode");
    assert_eq!(surface.as_host_bytes(), Some(host.as_slice()));
}

#[test]
fn tile_batch_region_scaled_auto_surface_matches_host_decode() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let mut ctx = DecoderContext::<ashlar_jpeg::DecoderContext>::new();
    let mut pool = ashlar_jpeg::ScratchPool::new();
    let surface = Codec::decode_tile_region_scaled_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        roi,
        scale,
        BackendRequest::Auto,
    )
    .expect("surface");
    assert_eq!(surface.backend_kind(), ashlar_core::BackendKind::Cpu);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));

    let (expected, _) = ashlar_jpeg::Decoder::new(BASELINE_420)
        .expect("host decoder")
        .decode_region_scaled(
            PixelFormat::Rgb8,
            ashlar_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        )
        .expect("host decode");
    assert_eq!(surface.as_host_bytes(), Some(expected.as_slice()));
}
