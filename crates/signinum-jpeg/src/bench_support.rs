// SPDX-License-Identifier: Apache-2.0

//! Hidden helpers used by Criterion benches.

use crate::backend::Backend;
use crate::color::upsample::upsample_h2v2_fancy_rows;
use crate::color::ycbcr::ycbcr_to_rgb;
use crate::context::DecoderContext;
use crate::decoder::{Decoder, JpegView};
use crate::entropy::huffman::HuffmanTable;
use crate::entropy::sequential::decode_scan_fast_tile_rgb_profiled;
use crate::error::JpegError;
use crate::idct::downscale::idct_islow_2x2_scalar;
use crate::idct::{idct_islow, idct_islow_dc_only};
use crate::internal::bit_reader::BitReader;
use crate::internal::scratch::{ScratchPool, SinkRows};
use crate::output::{InterleavedRgbWriter, OutputWriter};
use crate::parse::tables::{HuffmanValues, RawHuffmanTable};
use alloc::vec;
use alloc::vec::Vec;
use core::cell::Cell;
use core::ptr;
use std::time::Instant;

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

    #[cfg(target_arch = "aarch64")]
    pub(crate) fn record_scalar_chunk(&mut self) {
        self.scalar_chunks += 1;
    }

    #[cfg(target_arch = "aarch64")]
    pub(crate) fn record_neon_tail_chunk(&mut self) {
        self.neon_tail_chunks += 1;
    }
}

#[doc(hidden)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchBlockActivityCounts {
    total: usize,
    dc_only: usize,
    bottom_half_zero: usize,
    general: usize,
}

impl BenchBlockActivityCounts {
    pub fn total_blocks(self) -> usize {
        self.total
    }

    pub fn dc_only_blocks(self) -> usize {
        self.dc_only
    }

    pub fn bottom_half_zero_blocks(self) -> usize {
        self.bottom_half_zero
    }

    pub fn general_blocks(self) -> usize {
        self.general
    }

    pub(crate) fn record_dc_only(&mut self) {
        self.total += 1;
        self.dc_only += 1;
    }

    pub(crate) fn record_bottom_half_zero(&mut self) {
        self.total += 1;
        self.bottom_half_zero += 1;
    }

    pub(crate) fn record_general(&mut self) {
        self.total += 1;
        self.general += 1;
    }
}

#[doc(hidden)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchFast420Profile {
    total_ns: u128,
    parse_plan_ns: u128,
    mcu_decode_ns: u128,
    rgb_emit_ns: u128,
    finish_ns: u128,
    tile_count: usize,
    block_activity_counts: BenchBlockActivityCounts,
}

impl BenchFast420Profile {
    pub fn total_ns(self) -> u128 {
        self.total_ns
    }

    pub fn parse_plan_ns(self) -> u128 {
        self.parse_plan_ns
    }

    pub fn mcu_decode_ns(self) -> u128 {
        self.mcu_decode_ns
    }

    pub fn rgb_emit_ns(self) -> u128 {
        self.rgb_emit_ns
    }

    pub fn finish_ns(self) -> u128 {
        self.finish_ns
    }

    pub fn tile_count(self) -> usize {
        self.tile_count
    }

    pub fn block_activity_counts(self) -> BenchBlockActivityCounts {
        self.block_activity_counts
    }

    pub(crate) fn set_total_ns(&mut self, ns: u128) {
        self.total_ns = ns;
    }

    pub(crate) fn set_tile_count(&mut self, tile_count: usize) {
        self.tile_count = tile_count;
    }

    pub(crate) fn add_parse_plan_ns(&mut self, ns: u128) {
        self.parse_plan_ns += ns;
    }

    pub(crate) fn add_mcu_decode_ns(&mut self, ns: u128) {
        self.mcu_decode_ns += ns;
    }

    pub(crate) fn add_rgb_emit_ns(&mut self, ns: u128) {
        self.rgb_emit_ns += ns;
    }

    pub(crate) fn add_finish_ns(&mut self, ns: u128) {
        self.finish_ns += ns;
    }

    pub(crate) fn block_activity_counts_mut(&mut self) -> &mut BenchBlockActivityCounts {
        &mut self.block_activity_counts
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

#[cfg(target_arch = "aarch64")]
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

#[cfg(target_arch = "aarch64")]
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

struct BenchProfileSinkWriter {
    rows: SinkRows,
    backend: Backend,
}

impl BenchProfileSinkWriter {
    fn new(rows: SinkRows, backend: Backend) -> Self {
        Self { rows, backend }
    }

    fn into_rows(self) -> SinkRows {
        self.rows
    }
}

impl InterleavedRgbWriter for BenchProfileSinkWriter {
    fn with_rgb_rows<R, F>(&mut self, _y: u32, row_count: usize, fill: F) -> Result<R, JpegError>
    where
        F: FnOnce(&mut [u8], Option<&mut [u8]>) -> Result<R, JpegError>,
    {
        let result = match row_count {
            1 => fill(&mut self.rows.top_row, None),
            2 => fill(&mut self.rows.top_row, Some(&mut self.rows.bottom_row)),
            _ => unreachable!("profile sink only supports one or two rows"),
        }?;
        std::hint::black_box(&self.rows.top_row);
        if row_count == 2 {
            std::hint::black_box(&self.rows.bottom_row);
        }
        Ok(result)
    }
}

impl OutputWriter for BenchProfileSinkWriter {
    fn write_rgb_row(
        &mut self,
        _y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), JpegError> {
        self.backend
            .fill_rgb_row_from_rgb(r_row, g_row, b_row, &mut self.rows.top_row);
        std::hint::black_box(&self.rows.top_row);
        Ok(())
    }

    fn write_ycbcr_row(
        &mut self,
        _y: u32,
        y_row: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
    ) -> Result<(), JpegError> {
        self.backend
            .fill_rgb_row_from_ycbcr(y_row, cb_row, cr_row, &mut self.rows.top_row);
        std::hint::black_box(&self.rows.top_row);
        Ok(())
    }

    fn write_gray_row(&mut self, _y: u32, gray_row: &[u8]) -> Result<(), JpegError> {
        self.backend
            .fill_rgb_row_from_gray(gray_row, &mut self.rows.top_row);
        std::hint::black_box(&self.rows.top_row);
        Ok(())
    }
}

#[doc(hidden)]
pub fn bench_profile_fast420_tile_batch(
    bytes: &[u8],
    batch_size: usize,
) -> Result<Option<BenchFast420Profile>, JpegError> {
    let total_start = Instant::now();
    let mut profile = BenchFast420Profile::default();
    profile.set_tile_count(batch_size);
    let mut ctx = DecoderContext::new();
    let mut pool = ScratchPool::new();

    for _ in 0..batch_size {
        let parse_plan_start = Instant::now();
        let view = JpegView::parse(bytes)?;
        let dec = Decoder::from_view_in_context(view, &mut ctx)?;
        profile.add_parse_plan_ns(parse_plan_start.elapsed().as_nanos());

        if !dec.plan.matches_fast_tile_shape() {
            return Ok(None);
        }

        let width = dec.info.dimensions.0 as usize;
        let rows = pool.take_sink_rows(width);
        let mut writer = BenchProfileSinkWriter::new(rows, dec.backend);
        decode_scan_fast_tile_rgb_profiled(
            &dec.plan,
            dec.backend,
            &dec.bytes[dec.plan.scan_offset..],
            &mut pool,
            &mut writer,
            &mut profile,
        )?;
        pool.restore_sink_rows(writer.into_rows());
    }

    profile.set_total_ns(total_start.elapsed().as_nanos());
    Ok(Some(profile))
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

/// Run the scalar DC-only ISLOW IDCT helper on a caller-provided coefficient.
#[doc(hidden)]
pub fn bench_idct_dc_only_block_with(dc_coeff: i16, output: &mut [u8; 64]) {
    idct_islow_dc_only(dc_coeff, output);
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
