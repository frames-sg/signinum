// SPDX-License-Identifier: Apache-2.0

//! Hidden helpers used by Criterion benches.

use crate::color::upsample::upsample_h2v2_fancy_rows;
use crate::color::ycbcr::ycbcr_to_rgb;
use crate::entropy::huffman::HuffmanTable;
use crate::error::JpegError;
use crate::idct::idct_islow;
use crate::internal::bit_reader::BitReader;
use crate::parse::tables::{HuffmanValues, RawHuffmanTable};
use alloc::vec;
use alloc::vec::Vec;

#[doc(hidden)]
pub struct BenchHuffmanState {
    table: HuffmanTable,
    bytes: Vec<u8>,
    symbols: usize,
}

impl BenchHuffmanState {
    #[must_use]
    pub fn luma_dc_zeros(symbols: usize) -> Self {
        let table = HuffmanTable::from_raw(&RawHuffmanTable {
            bits: [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        })
        .expect("standard luma DC table must be valid");
        let bytes = vec![0u8; symbols.div_ceil(4) + 8];
        Self {
            table,
            bytes,
            symbols,
        }
    }

    pub fn decode_all(&self) -> Result<u32, JpegError> {
        let mut br = BitReader::new(&self.bytes);
        let mut sum = 0u32;
        for _ in 0..self.symbols {
            sum += u32::from(self.table.decode(&mut br)?);
        }
        Ok(sum)
    }
}

#[doc(hidden)]
#[must_use]
pub fn bench_idct_reference_block() -> [u8; 64] {
    let mut coeffs = [0i16; 64];
    coeffs[0] = 64;
    coeffs[1] = 24;
    coeffs[2] = -12;
    coeffs[8] = 18;
    coeffs[9] = -7;
    coeffs[16] = 5;

    let mut out = [0u8; 64];
    idct_islow(&coeffs, &mut out);
    out
}

/// Run the scalar ISLOW IDCT on a caller-provided block. Used by
/// `tests/idct_parity.rs` as the reference oracle.
#[doc(hidden)]
pub fn bench_idct_reference_block_with(input: &[i16; 64], output: &mut [u8; 64]) {
    idct_islow(input, output);
}

/// Run the NEON IDCT on a caller-provided block. Panics if the host CPU
/// does not support NEON — on aarch64 NEON is architecturally mandatory,
/// so the feature check is a formality. Used by `tests/idct_parity.rs`.
#[cfg(target_arch = "aarch64")]
#[doc(hidden)]
pub fn bench_idct_neon_block(input: &[i16; 64], output: &mut [u8; 64]) {
    unsafe { crate::idct::neon::idct_islow(input, output) };
}

/// Run the AVX2 IDCT on a caller-provided block. Requires runtime AVX2
/// support — call `std::is_x86_feature_detected!("avx2")` first.
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
pub fn bench_idct_avx2_block(input: &[i16; 64], output: &mut [u8; 64]) {
    unsafe { crate::idct::avx2::idct_islow(input, output) };
}

/// Pre-allocated scratch for the 4:2:0 fancy-upsample microbench. Stores
/// three chroma input rows (`prev`, `curr`, `next`) of length `chroma_width`
/// and two output rows of length `2 * chroma_width`.
#[doc(hidden)]
pub struct BenchUpsampleH2V2Scratch {
    prev: Vec<u8>,
    curr: Vec<u8>,
    next: Vec<u8>,
    top: Vec<u8>,
    bot: Vec<u8>,
}

impl BenchUpsampleH2V2Scratch {
    /// Create the scratch with a deterministic chroma pattern.
    #[must_use]
    pub fn new(chroma_width: usize) -> Self {
        let seed = |offset: usize| -> Vec<u8> {
            (0..chroma_width)
                .map(|i| ((i.wrapping_add(offset) * 131) ^ 0x5A) as u8)
                .collect()
        };
        Self {
            prev: seed(0),
            curr: seed(1),
            next: seed(2),
            top: vec![0u8; chroma_width * 2],
            bot: vec![0u8; chroma_width * 2],
        }
    }

    /// Run one iteration of `upsample_h2v2_fancy_rows` into the owned buffers.
    pub fn run(&mut self) {
        let out_width = self.top.len();
        upsample_h2v2_fancy_rows(
            &self.prev,
            &self.curr,
            &self.next,
            out_width,
            &mut self.top,
            &mut self.bot,
        );
    }
}

/// Pre-allocated scratch for the scalar YCbCr→RGB row microbench. Holds three
/// planar input rows of length `width` and one packed RGB output buffer of
/// length `3 * width`.
#[doc(hidden)]
pub struct BenchColorRowScratch {
    y: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
    rgb: Vec<u8>,
}

impl BenchColorRowScratch {
    /// Create the scratch with a deterministic luminance/chroma pattern.
    #[must_use]
    pub fn new(width: usize) -> Self {
        let seed = |offset: usize, scale: usize| -> Vec<u8> {
            (0..width)
                .map(|i| ((i.wrapping_mul(scale).wrapping_add(offset)) & 0xFF) as u8)
                .collect()
        };
        Self {
            y: seed(0, 7),
            cb: seed(64, 5),
            cr: seed(192, 3),
            rgb: vec![0u8; width * 3],
        }
    }

    /// Run one iteration of the scalar per-pixel YCbCr→RGB conversion.
    pub fn run_scalar(&mut self) {
        for (((&y, &cb), &cr), pixel) in self
            .y
            .iter()
            .zip(self.cb.iter())
            .zip(self.cr.iter())
            .zip(self.rgb.chunks_exact_mut(3))
        {
            let (r, g, b) = ycbcr_to_rgb(y, cb, cr);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
    }
}
