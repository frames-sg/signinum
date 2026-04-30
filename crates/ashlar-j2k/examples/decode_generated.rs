// SPDX-License-Identifier: Apache-2.0

//! Generate a tiny HTJ2K grayscale codestream and decode it through the public
//! WSI-shaped API.
//!
//! Run with:
//! `cargo run -p ashlar-j2k --example decode_generated`

use ashlar_j2k::{Downscale, J2kDecoder, J2kScratchPool, PixelFormat, Rect};
use ashlar_j2k_native::{encode_htj2k, EncodeOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let width = 16_u32;
    let height = 16_u32;
    let pixels: Vec<u8> = (0..width * height).map(|v| v as u8).collect();
    let bytes = encode_htj2k(
        &pixels,
        width,
        height,
        1,
        8,
        false,
        &EncodeOptions::default(),
    )?;

    let mut decoder = J2kDecoder::new(&bytes)?;
    let mut scratch = J2kScratchPool::new();

    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let mut region = vec![0_u8; (roi.w * roi.h) as usize];
    decoder.decode_region_into(
        &mut scratch,
        &mut region,
        roi.w as usize,
        PixelFormat::Gray8,
        roi,
    )?;

    let scaled_dims = (width / 2, height / 2);
    let mut scaled = vec![0_u8; (scaled_dims.0 * scaled_dims.1) as usize];
    decoder.decode_scaled_into(
        &mut scratch,
        &mut scaled,
        scaled_dims.0 as usize,
        PixelFormat::Gray8,
        Downscale::Half,
    )?;

    println!(
        "decoded {} ROI bytes and {} half-scale bytes",
        region.len(),
        scaled.len()
    );
    Ok(())
}
