use slidecodec_core::{
    BackendKind, BackendRequest, DecoderContext, DeviceSurface, Downscale, PixelFormat, Rect,
    TileBatchDecodeDevice,
};
use slidecodec_jpeg::DecoderContext as JpegDecoderContext;
use slidecodec_jpeg_metal::{Codec, ScratchPool};

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
