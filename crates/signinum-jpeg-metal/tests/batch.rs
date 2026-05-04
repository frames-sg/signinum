use signinum_core::{
    BackendKind, BackendRequest, CodecError, DecoderContext, DeviceSubmission, DeviceSurface,
    Downscale, PixelFormat, Rect, TileBatchDecodeDevice, TileBatchDecodeSubmit,
};
use signinum_jpeg::{Decoder as CpuDecoder, DecoderContext as JpegDecoderContext};
use signinum_jpeg_metal::{Codec, MetalSession, ScratchPool};

const BASELINE_420: &[u8] = include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg");
const BASELINE_420_RESTART: &[u8] =
    include_bytes!("../fixtures/jpeg/baseline_420_restart_32x16.jpg");
const GRAYSCALE: &[u8] = include_bytes!("../fixtures/jpeg/grayscale_8x8.jpg");

#[cfg(target_os = "macos")]
#[test]
fn tile_device_decode_matches_host_tile_decode() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let surface = Codec::decode_tile_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        BackendRequest::Metal,
    )
    .expect("tile device decode");

    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download");
    assert_eq!(downloaded, surface.as_bytes());
}

#[cfg(target_os = "macos")]
#[test]
fn tile_scaled_device_decode_has_expected_dimensions() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let surface = Codec::decode_tile_scaled_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        Downscale::Quarter,
        BackendRequest::Metal,
    )
    .expect("tile scaled device decode");
    assert_eq!(surface.dimensions(), (4, 4));
}

#[cfg(target_os = "macos")]
#[test]
fn tile_region_device_decode_has_expected_dimensions() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let surface = Codec::decode_tile_region_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        roi,
        BackendRequest::Metal,
    )
    .expect("tile region device decode");
    assert_eq!(surface.dimensions(), (8, 8));
}

#[cfg(target_os = "macos")]
#[test]
fn compatible_tile_submits_flush_once() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");

    let submissions = (0..4)
        .map(|_| {
            <Codec as TileBatchDecodeSubmit>::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                BASELINE_420,
                PixelFormat::Rgb8,
                BackendRequest::Metal,
            )
            .expect("submit")
        })
        .collect::<Vec<_>>();

    for submission in submissions {
        let surface = submission.wait().expect("surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    assert_eq!(session.submissions(), 1);
}

#[test]
fn auto_small_restart_tile_batch_stays_cpu_surface() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();
    let (expected, _) = CpuDecoder::new(BASELINE_420_RESTART)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");

    let submissions = (0..7)
        .map(|_| {
            <Codec as TileBatchDecodeSubmit>::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                BASELINE_420_RESTART,
                PixelFormat::Rgb8,
                BackendRequest::Auto,
            )
            .expect("submit")
        })
        .collect::<Vec<_>>();

    for submission in submissions {
        let surface = submission.wait().expect("surface");
        assert_eq!(surface.backend_kind(), BackendKind::Cpu);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    assert_eq!(session.submissions(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn auto_restart_wsi_tile_batch_uses_metal_at_threshold() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();
    let (expected, _) = CpuDecoder::new(BASELINE_420_RESTART)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");

    let submissions = (0..8)
        .map(|_| {
            <Codec as TileBatchDecodeSubmit>::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                BASELINE_420_RESTART,
                PixelFormat::Rgb8,
                BackendRequest::Auto,
            )
            .expect("submit")
        })
        .collect::<Vec<_>>();

    for submission in submissions {
        let surface = submission.wait().expect("surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    assert_eq!(session.submissions(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn compatible_region_scaled_tile_submits_flush_once() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let scale = Downscale::Quarter;
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_region_scaled(
            PixelFormat::Rgb8,
            signinum_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        )
        .expect("cpu region scaled");

    let submissions = (0..4)
        .map(|_| {
            Codec::submit_tile_region_scaled_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                BASELINE_420,
                PixelFormat::Rgb8,
                roi,
                scale,
                BackendRequest::Metal,
            )
            .expect("submit")
        })
        .collect::<Vec<_>>();

    for submission in submissions {
        let surface = submission.wait().expect("surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.dimensions(), (2, 2));
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    assert_eq!(session.submissions(), 1);
}

#[test]
fn auto_tile_region_scaled_unsupported_metal_shape_returns_cpu_surface() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };

    let surface = Codec::decode_tile_region_scaled_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        roi,
        Downscale::Quarter,
        BackendRequest::Auto,
    )
    .expect("auto tile region scaled surface");

    assert_eq!(surface.backend_kind(), BackendKind::Cpu);
    assert_eq!(surface.dimensions(), (2, 2));
    #[cfg(target_os = "macos")]
    assert!(surface.metal_buffer().is_none());
}

#[test]
fn explicit_metal_tile_unsupported_shape_is_rejected() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let result = Codec::decode_tile_to_device(
        &mut ctx,
        &mut pool,
        GRAYSCALE,
        PixelFormat::Gray8,
        BackendRequest::Metal,
    );

    match result {
        Err(signinum_jpeg_metal::Error::UnsupportedMetalRequest { reason }) => {
            assert!(reason.contains("JPEG Metal"));
        }
        Err(other) => panic!("unexpected explicit Metal tile error: {other:?}"),
        Ok(surface) => panic!(
            "explicit Metal tile request must not fall back; got {:?}",
            surface.backend_kind()
        ),
    }
}

#[test]
fn explicit_metal_tile_unsupported_error_is_codec_unsupported() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let err = match Codec::decode_tile_to_device(
        &mut ctx,
        &mut pool,
        GRAYSCALE,
        PixelFormat::Gray8,
        BackendRequest::Metal,
    ) {
        Err(err) => err,
        Ok(surface) => panic!(
            "explicit Metal tile request must not fall back; got {:?}",
            surface.backend_kind()
        ),
    };

    assert!(err.is_unsupported());
}

#[test]
fn cuda_tile_request_remains_unsupported_backend() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let result = Codec::decode_tile_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        BackendRequest::Cuda,
    );

    match result {
        Err(signinum_jpeg_metal::Error::UnsupportedBackend {
            request: BackendRequest::Cuda,
        }) => {}
        Err(signinum_jpeg_metal::Error::UnsupportedMetalRequest { reason }) => {
            panic!("CUDA must not be reported as a Metal request: {reason}")
        }
        Err(other) => panic!("unexpected CUDA tile error: {other:?}"),
        Ok(surface) => panic!(
            "CUDA tile request unexpectedly returned {:?}",
            surface.backend_kind()
        ),
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn non_macos_explicit_metal_tile_decode_is_unavailable() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let result = Codec::decode_tile_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        BackendRequest::Metal,
    );

    assert!(matches!(
        result,
        Err(signinum_jpeg_metal::Error::MetalUnavailable)
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn incompatible_shapes_split_batches() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();

    let full = <Codec as TileBatchDecodeSubmit>::submit_tile_to_device(
        &mut ctx,
        &mut session,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        BackendRequest::Metal,
    )
    .expect("full");
    let scaled = <Codec as TileBatchDecodeSubmit>::submit_tile_scaled_to_device(
        &mut ctx,
        &mut session,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        Downscale::Quarter,
        BackendRequest::Metal,
    )
    .expect("scaled");

    let _ = full.wait().expect("full wait");
    let _ = scaled.wait().expect("scaled wait");

    assert_eq!(session.submissions(), 2);
}
