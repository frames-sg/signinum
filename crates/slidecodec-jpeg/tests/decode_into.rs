// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `Decoder::decode_into`.

use slidecodec_jpeg::{Decoder, JpegError, OutputFormat};

mod fixtures;
use fixtures::{grayscale_8x8_jpeg, minimal_baseline_420_jpeg};

#[test]
#[ignore = "requires Task 17 fixture — see corpus/conformance/"]
fn decode_into_rgb8_returns_decoded_rect_full_image() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).expect("baseline 4:2:0 must construct");
    let (w, h) = dec.info().dimensions;
    let mut buf = vec![0u8; (w * h * 3) as usize];
    let outcome = dec
        .decode_into(&mut buf, (w * 3) as usize, OutputFormat::Rgb8)
        .expect("baseline 4:2:0 decode must succeed");
    assert_eq!(outcome.decoded.w, w);
    assert_eq!(outcome.decoded.h, h);
}

#[test]
#[ignore = "requires Task 17 fixture — see corpus/conformance/"]
fn decode_into_rgba8_writes_alpha_byte() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let (w, h) = dec.info().dimensions;
    let mut buf = vec![0u8; (w * h * 4) as usize];
    dec.decode_into(&mut buf, (w * 4) as usize, OutputFormat::Rgba8 { alpha: 200 })
        .unwrap();
    for y in 0..h as usize {
        for x in 0..w as usize {
            let idx = (y * w as usize + x) * 4;
            assert_eq!(buf[idx + 3], 200, "pixel ({x},{y}) alpha");
        }
    }
}

#[test]
#[ignore = "requires Task 17 fixture — see corpus/conformance/"]
fn decode_into_gray8_produces_single_byte_per_pixel() {
    let bytes = grayscale_8x8_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let (w, h) = dec.info().dimensions;
    assert_eq!((w, h), (8, 8));
    let mut buf = vec![0u8; (w * h) as usize];
    let outcome = dec
        .decode_into(&mut buf, w as usize, OutputFormat::Gray8)
        .unwrap();
    assert_eq!(outcome.decoded.w, 8);
    assert!(buf.iter().any(|&b| b != 0), "expected non-zero pixels");
}

#[test]
#[ignore = "requires Task 17 fixture — see corpus/conformance/"]
fn decode_into_rejects_undersized_buffer_with_api_misuse_error() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let mut buf = vec![0u8; 4];
    let err = dec
        .decode_into(&mut buf, 48, OutputFormat::Rgb8)
        .unwrap_err();
    assert!(err.is_api_misuse());
    assert!(matches!(err, JpegError::OutputBufferTooSmall { .. }));
}

#[test]
#[ignore = "requires Task 17 fixture — see corpus/conformance/"]
fn decode_into_rejects_stride_narrower_than_row_width() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let mut buf = vec![0u8; 16 * 16 * 3];
    let err = dec
        .decode_into(&mut buf, 10, OutputFormat::Rgb8)
        .unwrap_err();
    assert!(err.is_api_misuse());
    assert!(matches!(err, JpegError::InvalidStride { .. }));
}

#[test]
#[ignore = "requires Task 17 fixture — see corpus/conformance/"]
fn decode_into_tolerates_padded_stride() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).unwrap();
    let (w, h) = dec.info().dimensions;
    let padded_stride = (w as usize * 3) + 32;
    let mut buf = vec![0xAAu8; padded_stride * h as usize];
    dec.decode_into(&mut buf, padded_stride, OutputFormat::Rgb8)
        .unwrap();
    let last_row_start = (h as usize - 1) * padded_stride;
    let last_row_end = last_row_start + w as usize * 3;
    assert_eq!(
        &buf[last_row_end..last_row_end + 16],
        &[0xAA; 16],
        "stride padding must not be overwritten"
    );
}
