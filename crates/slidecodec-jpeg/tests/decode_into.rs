// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `Decoder::decode_into`.

use slidecodec_jpeg::{Decoder, Downscale, JpegError, PixelFormat, Rect};

mod fixtures;
use fixtures::{
    grayscale_8x8_jpeg, minimal_baseline_420_jpeg, rgb_app14_8x8_jpeg, rgb_app14_8x8_rgb,
};

fn minimal_cmyk_baseline_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(1u8, 64));
    bytes.extend_from_slice(&[
        0xff, 0xc0, 0x00, 20, 8, 0, 8, 0, 8, 4, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0, 4, 0x11, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xaa,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xbb,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xda, 0x00, 0x0e, 4, 1, 0x00, 2, 0x00, 3, 0x00, 4, 0x00, 0, 63, 0, 0x00, 0xff, 0xd9,
    ]);
    bytes
}

#[test]
fn decode_into_rgb8_returns_decoded_rect_full_image() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).expect("baseline 4:2:0 must construct");
    let (w, h) = dec.info().dimensions;
    let mut buf = vec![0u8; (w * h * 3) as usize];
    let outcome = dec
        .decode_into(&mut buf, (w * 3) as usize, PixelFormat::Rgb8)
        .expect("baseline 4:2:0 decode must succeed");
    assert_eq!(outcome.decoded.w, w);
    assert_eq!(outcome.decoded.h, h);
}

#[test]
fn decode_owned_rgb8_matches_decode_into() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).expect("baseline 4:2:0 must construct");
    let (w, h) = dec.info().dimensions;
    let mut expected = vec![0u8; (w * h * 3) as usize];
    let expected_outcome = dec
        .decode_into(&mut expected, (w * 3) as usize, PixelFormat::Rgb8)
        .expect("baseline 4:2:0 decode must succeed");

    let (owned, outcome) = dec.decode(PixelFormat::Rgb8).unwrap();
    assert_eq!(owned, expected);
    assert_eq!(outcome, expected_outcome);
}

#[test]
fn decode_into_rgba8_writes_alpha_byte() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let (w, h) = dec.info().dimensions;
    let mut buf = vec![0u8; (w * h * 4) as usize];
    dec.decode_rgba8_into_with_alpha(&mut buf, (w * 4) as usize, 200)
        .unwrap();
    for y in 0..h as usize {
        for x in 0..w as usize {
            let idx = (y * w as usize + x) * 4;
            assert_eq!(buf[idx + 3], 200, "pixel ({x},{y}) alpha");
        }
    }
}

#[test]
fn decode_into_rgba8_defaults_alpha_to_255() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let (w, h) = dec.info().dimensions;
    let mut buf = vec![0u8; (w * h * 4) as usize];
    dec.decode_into(&mut buf, (w * 4) as usize, PixelFormat::Rgba8)
        .unwrap();
    for y in 0..h as usize {
        for x in 0..w as usize {
            let idx = (y * w as usize + x) * 4;
            assert_eq!(buf[idx + 3], 255, "pixel ({x},{y}) alpha");
        }
    }
}

#[test]
fn decode_owned_region_scaled_matches_decode_region_into() {
    let bytes = rgb_app14_8x8_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let roi = Rect {
        x: 2,
        y: 2,
        w: 4,
        h: 4,
    };
    let mut expected = vec![0u8; 2 * 2 * 3];
    let expected_outcome = dec
        .decode_region_scaled_into(
            &mut expected,
            2 * 3,
            PixelFormat::Rgb8,
            roi,
            Downscale::Half,
        )
        .unwrap();

    let (owned, outcome) = dec
        .decode_region_scaled(PixelFormat::Rgb8, roi, Downscale::Half)
        .unwrap();
    assert_eq!(owned, expected);
    assert_eq!(outcome, expected_outcome);
}

#[test]
fn decode_owned_scaled_matches_decode_scaled_into() {
    let bytes = rgb_app14_8x8_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let mut expected = vec![0u8; 4 * 4 * 3];
    let expected_outcome = dec
        .decode_scaled_into(&mut expected, 4 * 3, PixelFormat::Rgb8, Downscale::Half)
        .unwrap();

    let (owned, outcome) = dec
        .decode_scaled(PixelFormat::Rgb8, Downscale::Half)
        .unwrap();
    assert_eq!(owned, expected);
    assert_eq!(outcome, expected_outcome);
}

#[test]
fn decode_into_gray8_produces_single_byte_per_pixel() {
    let bytes = grayscale_8x8_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let (w, h) = dec.info().dimensions;
    assert_eq!((w, h), (8, 8));
    let mut buf = vec![0u8; (w * h) as usize];
    let outcome = dec
        .decode_into(&mut buf, w as usize, PixelFormat::Gray8)
        .unwrap();
    assert_eq!(outcome.decoded.w, 8);
    assert!(buf.iter().any(|&b| b != 0), "expected non-zero pixels");
}

#[test]
fn decode_into_rejects_undersized_buffer_with_api_misuse_error() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let mut buf = vec![0u8; 4];
    let err = dec
        .decode_into(&mut buf, 48, PixelFormat::Rgb8)
        .unwrap_err();
    assert!(err.is_api_misuse());
    assert!(matches!(err, JpegError::OutputBufferTooSmall { .. }));
}

#[test]
fn decode_into_rejects_stride_narrower_than_row_width() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let mut buf = vec![0u8; 16 * 16 * 3];
    let err = dec
        .decode_into(&mut buf, 10, PixelFormat::Rgb8)
        .unwrap_err();
    assert!(err.is_api_misuse());
    assert!(matches!(err, JpegError::InvalidStride { .. }));
}

#[test]
fn decode_into_tolerates_padded_stride() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let (w, h) = dec.info().dimensions;
    let padded_stride = (w as usize * 3) + 32;
    let mut buf = vec![0xAAu8; padded_stride * h as usize];
    dec.decode_into(&mut buf, padded_stride, PixelFormat::Rgb8)
        .unwrap();
    let last_row_start = (h as usize - 1) * padded_stride;
    let last_row_end = last_row_start + w as usize * 3;
    assert_eq!(
        &buf[last_row_end..last_row_end + 16],
        &[0xAA; 16],
        "stride padding must not be overwritten"
    );
}

#[test]
fn decode_into_rgb8_preserves_app14_rgb_pixels() {
    let bytes = rgb_app14_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("APP14 RGB fixture must construct");
    let (w, h) = dec.info().dimensions;
    assert_eq!((w, h), (8, 8));
    let mut buf = vec![0u8; (w * h * 3) as usize];
    dec.decode_into(&mut buf, (w * 3) as usize, PixelFormat::Rgb8)
        .expect("APP14 RGB decode must succeed");
    assert_eq!(buf, rgb_app14_8x8_rgb());
}

#[test]
fn decode_into_rgb8_scaled_preserves_constant_app14_rgb_pixels() {
    let bytes = rgb_app14_8x8_jpeg();
    let dec = Decoder::new(&bytes).unwrap();

    for (factor, dims) in [
        (Downscale::Half, (4u32, 4u32)),
        (Downscale::Quarter, (2u32, 2u32)),
        (Downscale::Eighth, (1u32, 1u32)),
    ] {
        let mut buf = vec![0u8; dims.0 as usize * dims.1 as usize * 3];
        dec.decode_scaled_into(&mut buf, dims.0 as usize * 3, PixelFormat::Rgb8, factor)
            .unwrap();
        let mut expected = Vec::with_capacity(buf.len());
        for _ in 0..(dims.0 * dims.1) {
            expected.extend_from_slice(&[200, 20, 10]);
        }
        assert_eq!(buf, expected, "factor={factor:?}");
    }
}

#[test]
fn decode_into_gray8_scaled_projects_constant_app14_rgb_pixels() {
    let bytes = rgb_app14_8x8_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let expected = ((77 * 200 + 150 * 20 + 29 * 10 + 128) >> 8) as u8;

    for (factor, dims) in [
        (Downscale::Half, (4u32, 4u32)),
        (Downscale::Quarter, (2u32, 2u32)),
        (Downscale::Eighth, (1u32, 1u32)),
    ] {
        let mut buf = vec![0u8; dims.0 as usize * dims.1 as usize];
        dec.decode_scaled_into(&mut buf, dims.0 as usize, PixelFormat::Gray8, factor)
            .unwrap();
        assert!(buf.iter().all(|&px| px == expected), "factor={factor:?}");
    }
}

#[test]
fn decode_region_into_rgb8_crops_constant_app14_rgb_pixels() {
    let bytes = rgb_app14_8x8_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let roi = Rect {
        x: 2,
        y: 1,
        w: 3,
        h: 4,
    };
    let mut buf = vec![0u8; roi.w as usize * roi.h as usize * 3];
    let outcome = dec
        .decode_region_into(&mut buf, roi.w as usize * 3, PixelFormat::Rgb8, roi)
        .unwrap();
    assert_eq!(outcome.decoded, roi);
    let mut expected = Vec::with_capacity(buf.len());
    for _ in 0..(roi.w * roi.h) {
        expected.extend_from_slice(&[200, 20, 10]);
    }
    assert_eq!(buf, expected);
}

#[test]
fn decode_region_into_rgb8_scaled_crops_constant_app14_rgb_pixels() {
    let bytes = rgb_app14_8x8_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let roi = Rect {
        x: 2,
        y: 2,
        w: 4,
        h: 4,
    };
    let mut buf = vec![0u8; 2 * 2 * 3];
    let outcome = dec
        .decode_region_scaled_into(&mut buf, 2 * 3, PixelFormat::Rgb8, roi, Downscale::Half)
        .unwrap();
    assert_eq!(outcome.decoded, roi);
    let mut expected = Vec::with_capacity(buf.len());
    for _ in 0..4 {
        expected.extend_from_slice(&[200, 20, 10]);
    }
    assert_eq!(buf, expected);
}

#[test]
fn decoder_new_rejects_cmyk_baseline_as_unsupported() {
    let bytes = minimal_cmyk_baseline_jpeg();
    let err = Decoder::new(&bytes).expect_err("CMYK should not reach scalar decoder");
    assert!(matches!(err, JpegError::UnsupportedColorSpace { .. }));
    assert!(err.is_unsupported());
}

#[test]
fn decoder_new_rejects_invalid_sequential_scan_parameters() {
    let mut bytes = minimal_baseline_420_jpeg();
    let sos = bytes
        .windows(2)
        .position(|w| w == [0xff, 0xda])
        .expect("fixture SOS");
    bytes[sos + 2 + 2 + 1 + 3 * 2] = 1;

    let err = Decoder::new(&bytes).expect_err("baseline Ss=1 must be rejected");
    assert!(matches!(
        err,
        JpegError::InvalidScanParameters {
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
            ..
        }
    ));
}
