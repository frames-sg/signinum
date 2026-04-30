use ashlar_core::{
    BackendKind, BackendRequest, DecoderContext, DeviceSubmission, DeviceSurface, Downscale,
    PixelFormat, Rect, TileBatchDecodeDevice, TileBatchDecodeSubmit,
};
use ashlar_jpeg::{Decoder as CpuDecoder, DecoderContext as JpegDecoderContext};
use ashlar_jpeg_metal::{Codec, MetalSession, ScratchPool};

const BASELINE_420: &[u8] = include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");

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
            ashlar_jpeg::Rect {
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
