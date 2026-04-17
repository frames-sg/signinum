// SPDX-License-Identifier: Apache-2.0

//! Bit-exact parity against libjpeg-turbo's ISLOW path.

use slidecodec_jpeg::{Decoder, OutputFormat};

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
        .decode_into(&mut out, 16 * 3, OutputFormat::Rgb8)
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
    dec.decode_into(&mut out, 8, OutputFormat::Gray8)
        .expect("grayscale decode must succeed");
    assert_eq!(
        out, GRAYSCALE_8X8_GRAY,
        "grayscale parity must be bit-exact against djpeg -grayscale"
    );
}
