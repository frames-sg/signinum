// SPDX-License-Identifier: Apache-2.0

//! Decompress a zlib-wrapped Deflate tile payload into caller-owned output.
//!
//! Run with:
//! `cargo run -p slidecodec-tilecodec --example decompress`

use flate2::{write::ZlibEncoder, Compression};
use slidecodec_tilecodec::{DeflateCodec, DeflatePool, TileDecompress};
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source: Vec<u8> = (0..=255).cycle().take(4096).collect();
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&source)?;
    let compressed = encoder.finish()?;

    let mut pool = DeflatePool::new();
    let mut decoded = vec![0_u8; source.len()];
    let written = DeflateCodec::decompress_into(&mut pool, &compressed, &mut decoded)?;

    assert_eq!(&decoded[..written], source.as_slice());
    println!("decoded {written} bytes");
    Ok(())
}
