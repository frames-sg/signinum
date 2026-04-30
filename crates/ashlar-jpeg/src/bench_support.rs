// SPDX-License-Identifier: Apache-2.0

//! Hidden helpers used by Criterion benches.

use crate::backend::Backend;
use crate::color::upsample::upsample_h2v2_fancy_rows;
use crate::color::ycbcr::ycbcr_to_rgb;
use crate::entropy::huffman::HuffmanTable;
use crate::error::JpegError;
use crate::idct::downscale::idct_islow_2x2_scalar;
use crate::idct::idct_islow;
use crate::internal::bit_reader::BitReader;
use crate::parse::tables::{HuffmanValues, RawHuffmanTable};
use alloc::vec;
use alloc::vec::Vec;
use core::cell::Cell;
use core::ptr;

// `crate::backend::scalar` is intentionally private. Reuse the production
// source file here so bench/test helpers call the real scalar row-pair kernel
// without carrying a second handwritten copy of the algorithm.
#[allow(dead_code)]
#[allow(clippy::duplicate_mod)]
#[path = "backend/scalar.rs"]
mod bench_scalar_backend;

#[doc(hidden)]
#[derive(Default, Debug, Clone)]
pub struct Bench420DispatchStats {
    scalar_chunks: usize,
    neon_tail_chunks: usize,
}

impl Bench420DispatchStats {
    pub fn scalar_chunks(&self) -> usize {
        self.scalar_chunks
    }

    pub fn neon_tail_chunks(&self) -> usize {
        self.neon_tail_chunks
    }

    #[allow(dead_code)]
    pub(crate) fn record_scalar_chunk(&mut self) {
        self.scalar_chunks += 1;
    }

    #[allow(dead_code)]
    pub(crate) fn record_neon_tail_chunk(&mut self) {
        self.neon_tail_chunks += 1;
    }
}

thread_local! {
    static BENCH_420_DISPATCH_STATS: Cell<*mut Bench420DispatchStats> = const {
        Cell::new(ptr::null_mut())
    };
}

struct Bench420DispatchStatsGuard {
    prev: *mut Bench420DispatchStats,
}

impl Drop for Bench420DispatchStatsGuard {
    fn drop(&mut self) {
        BENCH_420_DISPATCH_STATS.with(|slot| {
            slot.set(self.prev);
        });
    }
}

#[allow(dead_code)]
pub(crate) fn record_420_dispatch_scalar_chunk() {
    BENCH_420_DISPATCH_STATS.with(|slot| {
        let stats = slot.get();
        if !stats.is_null() {
            unsafe {
                (*stats).record_scalar_chunk();
            }
        }
    });
}

#[allow(dead_code)]
pub(crate) fn record_420_dispatch_neon_tail_chunk() {
    BENCH_420_DISPATCH_STATS.with(|slot| {
        let stats = slot.get();
        if !stats.is_null() {
            unsafe {
                (*stats).record_neon_tail_chunk();
            }
        }
    });
}

fn with_420_dispatch_stats<R>(stats: &mut Bench420DispatchStats, f: impl FnOnce() -> R) -> R {
    BENCH_420_DISPATCH_STATS.with(|slot| {
        let guard = Bench420DispatchStatsGuard {
            prev: slot.replace(ptr::from_mut(stats)),
        };
        let out = f();
        drop(guard);
        out
    })
}

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

/// Run the scalar reduced 2x2 IDCT on a caller-provided block. Used by future
/// quarter-scale parity and microbench coverage.
#[doc(hidden)]
pub fn bench_idct_reduced_2x2_block_with(input: &[i16; 64], output: &mut [u8; 4]) {
    idct_islow_2x2_scalar(input, output);
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

/// Pre-allocated scratch for the 4:2:0 RGB row-pair microbench. Stores two
/// luma rows, three chroma rows per plane, and two packed RGB output rows.
#[doc(hidden)]
pub struct BenchRgb420RowPairScratch {
    y_top: Vec<u8>,
    y_bottom: Vec<u8>,
    prev_cb: Vec<u8>,
    curr_cb: Vec<u8>,
    next_cb: Vec<u8>,
    prev_cr: Vec<u8>,
    curr_cr: Vec<u8>,
    next_cr: Vec<u8>,
    top: Vec<u8>,
    bottom: Vec<u8>,
}

impl BenchRgb420RowPairScratch {
    /// Create the scratch with a deterministic odd-width-friendly pattern.
    #[must_use]
    pub fn new(width: usize) -> Self {
        let chroma_width = width.div_ceil(2);
        let seed = |len: usize, offset: usize, scale: usize| -> Vec<u8> {
            (0..len)
                .map(|i| ((i.wrapping_mul(scale).wrapping_add(offset)) & 0xFF) as u8)
                .collect()
        };
        Self {
            y_top: seed(width, 5, 37),
            y_bottom: seed(width, 211, 19),
            prev_cb: seed(chroma_width, 9, 13),
            curr_cb: seed(chroma_width, 41, 17),
            next_cb: seed(chroma_width, 73, 23),
            prev_cr: seed(chroma_width, 17, 29),
            curr_cr: seed(chroma_width, 53, 31),
            next_cr: seed(chroma_width, 89, 37),
            top: vec![0u8; width * 3],
            bottom: vec![0u8; width * 3],
        }
    }

    /// Run one iteration through the detected CPU backend.
    pub fn run(&mut self) {
        bench_rgb_row_pair_from_420(
            &self.y_top,
            Some(&self.y_bottom),
            &self.prev_cb,
            &self.curr_cb,
            &self.next_cb,
            &self.prev_cr,
            &self.curr_cr,
            &self.next_cr,
            &mut self.top,
            Some(&mut self.bottom),
        );
    }

    /// Run one iteration through the scalar reference path.
    pub fn run_reference(&mut self) {
        bench_rgb_row_pair_from_420_reference(
            &self.y_top,
            Some(&self.y_bottom),
            &self.prev_cb,
            &self.curr_cb,
            &self.next_cb,
            &self.prev_cr,
            &self.curr_cr,
            &self.next_cr,
            &mut self.top,
            Some(&mut self.bottom),
        );
    }
}

/// Run the platform's normal RGB 4:2:0 row-pair backend on caller-provided
/// inputs. On aarch64 this routes through the detected NEON path.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn bench_rgb_row_pair_from_420(
    y_top: &[u8],
    y_bottom: Option<&[u8]>,
    prev_cb: &[u8],
    curr_cb: &[u8],
    next_cb: &[u8],
    prev_cr: &[u8],
    curr_cr: &[u8],
    next_cr: &[u8],
    dst_top: &mut [u8],
    dst_bottom: Option<&mut [u8]>,
) {
    Backend::detect().fill_rgb_row_pair_from_420(
        y_top, y_bottom, prev_cb, curr_cb, next_cb, prev_cr, curr_cr, next_cr, dst_top, dst_bottom,
    );
}

/// Run the RGB 4:2:0 row-pair backend with dispatch stats.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn bench_rgb_row_pair_from_420_with_stats(
    y_top: &[u8],
    y_bottom: Option<&[u8]>,
    prev_cb: &[u8],
    curr_cb: &[u8],
    next_cb: &[u8],
    prev_cr: &[u8],
    curr_cr: &[u8],
    next_cr: &[u8],
    dst_top: &mut [u8],
    dst_bottom: Option<&mut [u8]>,
    stats: &mut Bench420DispatchStats,
) {
    with_420_dispatch_stats(stats, || {
        bench_rgb_row_pair_from_420(
            y_top, y_bottom, prev_cb, curr_cb, next_cb, prev_cr, curr_cr, next_cr, dst_top,
            dst_bottom,
        );
    });
}

/// Run the scalar RGB 4:2:0 row-pair reference on caller-provided inputs.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn bench_rgb_row_pair_from_420_reference(
    y_top: &[u8],
    y_bottom: Option<&[u8]>,
    prev_cb: &[u8],
    curr_cb: &[u8],
    next_cb: &[u8],
    prev_cr: &[u8],
    curr_cr: &[u8],
    next_cr: &[u8],
    dst_top: &mut [u8],
    dst_bottom: Option<&mut [u8]>,
) {
    bench_scalar_backend::fill_rgb_row_pair_from_420(
        y_top, y_bottom, prev_cb, curr_cb, next_cb, prev_cr, curr_cr, next_cr, dst_top, dst_bottom,
    );
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
    backend: Backend,
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
            backend: Backend::detect(),
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

    /// Run one iteration through the detected production backend.
    pub fn run_backend(&mut self) {
        self.backend
            .fill_rgb_row_from_ycbcr(&self.y, &self.cb, &self.cr, &mut self.rgb);
    }
}
