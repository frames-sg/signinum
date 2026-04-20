use slidecodec_core::{
    BackendKind, BackendRequest, DeviceSurface, ImageDecode, ImageDecodeDevice, PixelFormat, Rect,
    TileBatchDecodeDevice,
};
use slidecodec_j2k::J2kContext;
use slidecodec_j2k_metal::{Codec, Error, J2kDecoder, J2kScratchPool};
use slidecodec_j2k_native::{encode, encode_htj2k, EncodeOptions};

fn fixture() -> Vec<u8> {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 2, 3, 8, false, &options).expect("encode")
}

fn fixture_gray12() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(8);
    for sample in [0u16, 257, 1023, 4095] {
        pixels.extend_from_slice(&sample.to_le_bytes());
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 2, 1, 12, false, &options).expect("encode gray12")
}

fn fixture_rgb12() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(12);
    for sample in [0u16, 1023, 2047, 3071, 4095, 17] {
        pixels.extend_from_slice(&sample.to_le_bytes());
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 1, 3, 12, false, &options).expect("encode rgb12")
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
fn full_decode_to_metal_matches_host_decode() {
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 12];
    host_decoder
        .decode_into(&mut host, 6, PixelFormat::Rgb8)
        .expect("host decode");

    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Metal)
        .expect("device decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (2, 2));
    assert_eq!(surface.as_bytes(), host.as_slice());
}

#[test]
fn region_and_tile_device_paths_report_expected_dimensions() {
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let roi = Rect {
        x: 0,
        y: 0,
        w: 1,
        h: 1,
    };
    let region = decoder
        .decode_region_to_device(PixelFormat::Rgb8, roi, BackendRequest::Metal)
        .expect("region surface");
    assert_eq!(region.dimensions(), (1, 1));

    let mut ctx = slidecodec_core::DecoderContext::<J2kContext>::new();
    let mut pool = J2kScratchPool::new();
    let tile = Codec::decode_tile_to_device(
        &mut ctx,
        &mut pool,
        &bytes,
        PixelFormat::Rgb8,
        BackendRequest::Metal,
    )
    .expect("tile surface");
    assert_eq!(tile.dimensions(), (2, 2));
}

#[test]
fn region_and_scaled_metal_bytes_match_host_decode() {
    let bytes = fixture();
    let roi = Rect {
        x: 0,
        y: 0,
        w: 1,
        h: 1,
    };

    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let region_surface = decoder
        .decode_region_to_device(PixelFormat::Rgb8, roi, BackendRequest::Metal)
        .expect("region surface");

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut region_host = [0u8; 3];
    host_decoder
        .decode_region_into(
            &mut J2kScratchPool::new(),
            &mut region_host,
            3,
            PixelFormat::Rgb8,
            roi,
        )
        .expect("host region");
    assert_eq!(region_surface.as_bytes(), region_host.as_slice());

    let scaled_surface = decoder
        .decode_scaled_to_device(
            PixelFormat::Rgb8,
            slidecodec_core::Downscale::Half,
            BackendRequest::Metal,
        )
        .expect("scaled surface");
    let mut scaled_host = [0u8; 3];
    host_decoder
        .decode_scaled_into(
            &mut J2kScratchPool::new(),
            &mut scaled_host,
            3,
            PixelFormat::Rgb8,
            slidecodec_core::Downscale::Half,
        )
        .expect("host scaled");
    assert_eq!(scaled_surface.as_bytes(), scaled_host.as_slice());
}

#[test]
fn metal_gray16_matches_host_decode_for_12bit_source() {
    let bytes = fixture_gray12();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 8];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray16)
        .expect("host decode");

    let surface = decoder
        .decode_to_device(PixelFormat::Gray16, BackendRequest::Metal)
        .expect("device decode");
    assert_eq!(surface.as_bytes(), host.as_slice());
}

#[test]
fn metal_rgb16_matches_host_decode_for_12bit_source() {
    let bytes = fixture_rgb12();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 12];
    host_decoder
        .decode_into(&mut host, 12, PixelFormat::Rgb16)
        .expect("host decode");

    let surface = decoder
        .decode_to_device(PixelFormat::Rgb16, BackendRequest::Metal)
        .expect("device decode");
    assert_eq!(surface.as_bytes(), host.as_slice());
}

#[test]
fn invalid_region_reports_error_instead_of_panicking() {
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    match decoder.decode_region_to_device(PixelFormat::Rgb8, roi, BackendRequest::Metal) {
        Err(Error::Decode(slidecodec_j2k::J2kError::InvalidRegion { .. })) => {}
        Err(other) => panic!("unexpected error for invalid ROI: {other:?}"),
        Ok(_) => panic!("invalid ROI should fail"),
    }
}

#[test]
fn metal_scaled_htj2k_matches_host_fallback_decode() {
    let bytes = fixture_ht_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");

    let surface = decoder
        .decode_scaled_to_device(
            PixelFormat::Gray8,
            slidecodec_core::Downscale::Half,
            BackendRequest::Metal,
        )
        .expect("metal scaled decode");

    let mut host = [0u8; 4];
    host_decoder
        .decode_scaled_into(
            &mut J2kScratchPool::new(),
            &mut host,
            2,
            PixelFormat::Gray8,
            slidecodec_core::Downscale::Half,
        )
        .expect("host scaled decode");
    assert_eq!(surface.as_bytes(), host.as_slice());
}
