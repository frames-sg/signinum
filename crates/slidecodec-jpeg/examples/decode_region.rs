// SPDX-License-Identifier: Apache-2.0

//! Decode a source-coordinate ROI from a JPEG tile.
//!
//! Run with:
//! `cargo run -p slidecodec-jpeg --example decode_region`

use slidecodec_jpeg::{Decoder, PixelFormat, Rect};

const TILE: &[u8] = include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let decoder = Decoder::new(TILE)?;
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let (rgb, outcome) = decoder.decode_region(PixelFormat::Rgb8, roi)?;

    println!(
        "decoded {}x{} ROI into {} RGB bytes",
        outcome.decoded.w,
        outcome.decoded.h,
        rgb.len()
    );
    Ok(())
}
