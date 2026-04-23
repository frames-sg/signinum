use slidecodec_core::{
    BackendKind, BackendRequest, CodecError, DeviceSurface, Downscale, ImageDecode,
    ImageDecodeDevice, PixelFormat, Rect, TileBatchDecodeDevice,
};
use slidecodec_j2k::J2kContext;
use slidecodec_j2k_metal::{Codec, Error, J2kDecoder, J2kScratchPool};
use slidecodec_j2k_native::{encode, encode_htj2k, EncodeOptions};

fn fixture_rgb8() -> Vec<u8> {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 2, 3, 8, false, &options).expect("encode rgb8")
}

fn fixture_gray8() -> Vec<u8> {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode gray8")
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

fn fixture_gray8_irreversible() -> Vec<u8> {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: false,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode gray8 irreversible")
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
fn full_classic_grayscale_decode_to_metal_matches_host_decode() {
    let bytes = fixture_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 16];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray8)
        .expect("host decode");

    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Metal)
        .expect("device decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (4, 4));
    assert_eq!(surface.as_bytes(), host.as_slice());
}

#[test]
fn full_htj2k_decode_to_metal_matches_host_decode() {
    let bytes = fixture_ht_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 16];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray8)
        .expect("host decode");

    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Metal)
        .expect("device decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (4, 4));
    assert_eq!(surface.as_bytes(), host.as_slice());
}

#[test]
fn full_irreversible_j2k_decode_to_metal_matches_host_decode() {
    let bytes = fixture_gray8_irreversible();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 16];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray8)
        .expect("host decode");

    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Metal)
        .expect("device decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (4, 4));
    assert_eq!(surface.as_bytes(), host.as_slice());
}

#[test]
fn auto_full_grayscale_prefers_cpu_for_small_classic_fixture() {
    let bytes = fixture_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Auto)
        .expect("auto decode");
    assert_eq!(surface.backend_kind(), BackendKind::Cpu);
}

#[test]
fn auto_full_htj2k_prefers_cpu_for_small_fixture() {
    let bytes = fixture_ht_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Auto)
        .expect("auto decode");
    assert_eq!(surface.backend_kind(), BackendKind::Cpu);
}

#[test]
fn tile_full_grayscale_device_path_uses_metal_direct() {
    let bytes = fixture_gray8();
    let mut ctx = slidecodec_core::DecoderContext::<J2kContext>::new();
    let mut pool = J2kScratchPool::new();
    let surface = Codec::decode_tile_to_device(
        &mut ctx,
        &mut pool,
        &bytes,
        PixelFormat::Gray8,
        BackendRequest::Metal,
    )
    .expect("tile surface");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (4, 4));
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
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.as_bytes(), host.as_slice());
}

#[test]
fn explicit_metal_rejects_non_grayscale_formats() {
    let rgb8 = fixture_rgb8();
    let mut rgb8_decoder = J2kDecoder::new(&rgb8).expect("rgb8 decoder");
    let Err(rgb8_error) = rgb8_decoder.decode_to_device(PixelFormat::Rgb8, BackendRequest::Metal)
    else {
        panic!("rgb8 should be unsupported on explicit MetalDirect");
    };
    assert!(
        rgb8_error.is_unsupported(),
        "rgb8 error must be unsupported"
    );

    let rgb12 = fixture_rgb12();
    let mut rgb16_decoder = J2kDecoder::new(&rgb12).expect("rgb12 decoder");
    let Err(rgb16_error) =
        rgb16_decoder.decode_to_device(PixelFormat::Rgb16, BackendRequest::Metal)
    else {
        panic!("rgb16 should be unsupported on explicit MetalDirect");
    };
    assert!(
        rgb16_error.is_unsupported(),
        "rgb16 error must be unsupported"
    );
}

#[test]
fn explicit_metal_rejects_region_and_scaled_requests() {
    let bytes = fixture_gray8();
    let roi = Rect {
        x: 0,
        y: 0,
        w: 2,
        h: 2,
    };

    let mut region_decoder = J2kDecoder::new(&bytes).expect("decoder");
    let Err(region_error) =
        region_decoder.decode_region_to_device(PixelFormat::Gray8, roi, BackendRequest::Metal)
    else {
        panic!("explicit Metal region decode must be unsupported in the first cut");
    };
    assert!(region_error.is_unsupported());

    let mut scaled_decoder = J2kDecoder::new(&bytes).expect("decoder");
    let Err(scaled_error) = scaled_decoder.decode_scaled_to_device(
        PixelFormat::Gray8,
        Downscale::Half,
        BackendRequest::Metal,
    ) else {
        panic!("explicit Metal scaled decode must be unsupported in the first cut");
    };
    assert!(scaled_error.is_unsupported());
}

#[test]
fn auto_region_and_scaled_fallback_to_cpu_surface_and_match_host_decode() {
    let bytes = fixture_rgb8();
    let roi = Rect {
        x: 0,
        y: 0,
        w: 1,
        h: 1,
    };

    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let region_surface = decoder
        .decode_region_to_device(PixelFormat::Rgb8, roi, BackendRequest::Auto)
        .expect("region surface");
    assert_eq!(region_surface.backend_kind(), BackendKind::Cpu);

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
        .decode_scaled_to_device(PixelFormat::Rgb8, Downscale::Half, BackendRequest::Auto)
        .expect("scaled surface");
    assert_eq!(scaled_surface.backend_kind(), BackendKind::Cpu);

    let mut scaled_host = [0u8; 3];
    host_decoder
        .decode_scaled_into(
            &mut J2kScratchPool::new(),
            &mut scaled_host,
            3,
            PixelFormat::Rgb8,
            Downscale::Half,
        )
        .expect("host scaled");
    assert_eq!(scaled_surface.as_bytes(), scaled_host.as_slice());
}

#[test]
fn invalid_region_reports_error_instead_of_panicking() {
    let bytes = fixture_rgb8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    match decoder.decode_region_to_device(PixelFormat::Rgb8, roi, BackendRequest::Auto) {
        Err(Error::Decode(slidecodec_j2k::J2kError::InvalidRegion { .. })) => {}
        Err(other) => panic!("unexpected error for invalid ROI: {other:?}"),
        Ok(_) => panic!("invalid ROI should fail"),
    }
}
