// SPDX-License-Identifier: Apache-2.0

//! Bit-exact parity against libjpeg-turbo's ISLOW path.

use signinum_jpeg::{Decoder, Downscale, PixelFormat, Rect};

const BASELINE_420_JPG: &[u8] =
    include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");
const BASELINE_420_RGB: &[u8] =
    include_bytes!("../../../corpus/conformance/baseline_420_16x16.rgb");

const GRAYSCALE_8X8_JPG: &[u8] = include_bytes!("../../../corpus/conformance/grayscale_8x8.jpg");
const GRAYSCALE_8X8_GRAY: &[u8] = include_bytes!("../../../corpus/conformance/grayscale_8x8.gray");

#[test]
fn baseline_420_16x16_matches_libjpeg_turbo_bit_exact() {
    let dec = Decoder::new(BASELINE_420_JPG).expect("fixture must parse");
    let (w, h) = dec.info().dimensions;
    assert_eq!((w, h), (16, 16));
    let mut out = vec![0u8; 16 * 16 * 3];
    let outcome = dec
        .decode_scaled_into(&mut out, 16 * 3, PixelFormat::Rgb8, Downscale::None)
        .expect("decode must succeed");
    assert_eq!(outcome.decoded.w, 16);
    assert_eq!(outcome.decoded.h, 16);

    if out != BASELINE_420_RGB {
        let first_diff = out
            .iter()
            .zip(BASELINE_420_RGB.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(usize::MAX);
        panic!(
            "parity mismatch at byte {first_diff}: got {:?} want {:?}\nfull decoded: {:?}\nreference:    {:?}",
            out.get(first_diff),
            BASELINE_420_RGB.get(first_diff),
            &out[..first_diff.min(out.len())],
            &BASELINE_420_RGB[..first_diff.min(BASELINE_420_RGB.len())],
        );
    }
}

#[test]
fn grayscale_8x8_matches_libjpeg_turbo_bit_exact() {
    let dec = Decoder::new(GRAYSCALE_8X8_JPG).expect("grayscale fixture must parse");
    let (w, h) = dec.info().dimensions;
    assert_eq!((w, h), (8, 8));
    let mut out = vec![0u8; 8 * 8];
    dec.decode_scaled_into(&mut out, 8, PixelFormat::Gray8, Downscale::None)
        .expect("grayscale decode must succeed");
    assert_eq!(
        out, GRAYSCALE_8X8_GRAY,
        "grayscale parity must be bit-exact against djpeg -grayscale"
    );
}

#[test]
fn baseline_420_wsi_shaped_region_matches_full_decode_crop() {
    let dec = Decoder::new(BASELINE_420_JPG).expect("fixture must parse");
    let roi = Rect {
        x: 3,
        y: 2,
        w: 10,
        h: 11,
    };

    let full = decode_full_rgb(&dec);
    let region = decode_region_rgb(&dec, roi);
    assert_eq!(region, crop_rgb8(&full, 16, roi));
}

#[test]
fn baseline_420_wsi_shaped_scaled_region_matches_full_decode_crop() {
    let dec = Decoder::new(BASELINE_420_JPG).expect("fixture must parse");
    let roi = Rect {
        x: 3,
        y: 2,
        w: 10,
        h: 11,
    };

    let mut full = vec![0u8; 8 * 8 * 3];
    dec.decode_scaled_into(&mut full, 8 * 3, PixelFormat::Rgb8, Downscale::Half)
        .expect("full scaled decode must succeed");

    let region = dec
        .decode_region_scaled(PixelFormat::Rgb8, roi, Downscale::Half)
        .expect("scaled region decode must succeed")
        .0;

    let scaled_roi = scaled_rect_covering_half(roi);
    assert_eq!(region, crop_rgb8(&full, 8, scaled_roi));
}

#[test]
fn restart_coded_grayscale_wsi_shaped_region_matches_full_decode_crop() {
    let bytes = restart_coded_grayscale_jpeg(24, 24);
    let dec = Decoder::new(&bytes).expect("restart-coded fixture must parse");
    let roi = Rect {
        x: 5,
        y: 6,
        w: 11,
        h: 10,
    };

    let mut full = vec![0u8; 24 * 24];
    dec.decode_scaled_into(&mut full, 24, PixelFormat::Gray8, Downscale::None)
        .expect("full grayscale decode must succeed");
    let region = dec
        .decode_region_scaled(PixelFormat::Gray8, roi, Downscale::None)
        .expect("restart-coded region decode must succeed")
        .0;

    assert_eq!(region, crop_gray8(&full, 24, roi));
}

fn decode_region_rgb(dec: &Decoder<'_>, roi: Rect) -> Vec<u8> {
    dec.decode_region_scaled(PixelFormat::Rgb8, roi, Downscale::None)
        .expect("region decode must succeed")
        .0
}

fn decode_full_rgb(dec: &Decoder<'_>) -> Vec<u8> {
    let (w, h) = dec.info().dimensions;
    let mut out = vec![0u8; (w * h * 3) as usize];
    dec.decode_scaled_into(
        &mut out,
        (w * 3) as usize,
        PixelFormat::Rgb8,
        Downscale::None,
    )
    .expect("full decode must succeed");
    out
}

fn crop_rgb8(full: &[u8], width: usize, roi: Rect) -> Vec<u8> {
    let mut out = Vec::with_capacity((roi.w * roi.h * 3) as usize);
    let row_stride = width * 3;
    let crop_stride = roi.w as usize * 3;
    for y in roi.y as usize..(roi.y + roi.h) as usize {
        let row = &full[y * row_stride..(y + 1) * row_stride];
        let x0 = roi.x as usize * 3;
        out.extend_from_slice(&row[x0..x0 + crop_stride]);
    }
    out
}

fn crop_gray8(full: &[u8], width: usize, roi: Rect) -> Vec<u8> {
    let mut out = Vec::with_capacity((roi.w * roi.h) as usize);
    for y in roi.y as usize..(roi.y + roi.h) as usize {
        let row = &full[y * width..(y + 1) * width];
        let x0 = roi.x as usize;
        out.extend_from_slice(&row[x0..x0 + roi.w as usize]);
    }
    out
}

fn scaled_rect_covering_half(roi: Rect) -> Rect {
    let x0 = roi.x / 2;
    let y0 = roi.y / 2;
    let x1 = (roi.x + roi.w).div_ceil(2);
    let y1 = (roi.y + roi.h).div_ceil(2);
    Rect {
        x: x0,
        y: y0,
        w: x1 - x0,
        h: y1 - y0,
    }
}

fn restart_coded_grayscale_jpeg(width: u16, height: u16) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff,
        0xc0,
        0x00,
        11,
        8,
        (height >> 8) as u8,
        height as u8,
        (width >> 8) as u8,
        width as u8,
        1,
        1,
        0x11,
        0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xdd, 0x00, 0x04, 0x00, 0x01]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, 0, 63, 0]);

    let mcu_cols = u32::from(width).div_ceil(8);
    let mcu_rows = u32::from(height).div_ceil(8);
    let mcu_count = (mcu_cols * mcu_rows) as usize;
    for mcu in 0..mcu_count {
        bytes.push(0x00);
        if mcu + 1 != mcu_count {
            bytes.extend_from_slice(&[0xff, 0xd0 | ((mcu as u8) & 0x07)]);
        }
    }

    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}
