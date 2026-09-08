// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hidden helpers used by Criterion benches.

#![allow(
    clippy::disallowed_macros,
    reason = "benchmark-only buffers are outside codec runtime allocation budgets"
)]

use crate::backend::scalar;
use crate::backend::{Backend, Rgb420ChromaRows, Rgb420Crop, Rgb420CroppedRowPair, Rgb420RowPair};
use crate::color::upsample::upsample_h2v2_fancy_rows;
use crate::color::ycbcr::ycbcr_to_rgb;
use crate::context::DecoderContext;
use crate::decoder::{Decoder, JpegView, SinkWriter};
use crate::entropy::huffman::HuffmanTable;
use crate::entropy::sequential::decode_scan_fast_tile_rgb_profiled;
use crate::error::JpegError;
use crate::idct::downscale::idct_islow_2x2_scalar;
use crate::idct::{idct_islow, idct_islow_dc_only};
use crate::internal::bit_reader::BitReader;
use crate::internal::scratch::ScratchPool;
use crate::parse::tables::{HuffmanTableRole, HuffmanValues, RawHuffmanTable};
use alloc::vec;
use alloc::vec::Vec;
use core::cell::Cell;
use core::ptr;
use j2k_core::RowSink;
use std::time::Instant;

#[doc(hidden)]
#[derive(Default, Debug, Clone)]
pub struct Bench420DispatchStats {
    scalar_chunks: usize,
    neon_tail_chunks: usize,
}

impl Bench420DispatchStats {
    #[must_use]
    pub fn scalar_chunks(&self) -> usize {
        self.scalar_chunks
    }

    #[must_use]
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
    #[must_use]
    pub fn total_blocks(self) -> usize {
        self.total
    }

    #[must_use]
    pub fn dc_only_blocks(self) -> usize {
        self.dc_only
    }

    #[must_use]
    pub fn bottom_half_zero_blocks(self) -> usize {
        self.bottom_half_zero
    }

    #[must_use]
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
    #[must_use]
    pub fn total_ns(self) -> u128 {
        self.total_ns
    }

    #[must_use]
    pub fn parse_plan_ns(self) -> u128 {
        self.parse_plan_ns
    }

    #[must_use]
    pub fn mcu_decode_ns(self) -> u128 {
        self.mcu_decode_ns
    }

    #[must_use]
    pub fn rgb_emit_ns(self) -> u128 {
        self.rgb_emit_ns
    }

    #[must_use]
    pub fn finish_ns(self) -> u128 {
        self.finish_ns
    }

    #[must_use]
    pub fn tile_count(self) -> usize {
        self.tile_count
    }

    #[must_use]
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
            // SAFETY: Benchmark helpers preserve production buffer sizing and backend feature checks.
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
            // SAFETY: Benchmark helpers preserve production buffer sizing and backend feature checks.
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

struct BlackBoxRowSink;

impl RowSink<u8> for BlackBoxRowSink {
    type Error = JpegError;

    fn write_row(&mut self, _y: u32, row: &[u8]) -> Result<(), Self::Error> {
        std::hint::black_box(row);
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
        let rows = pool.take_sink_rows(width.saturating_mul(3), dec.plan.scratch_bytes)?;
        let mut sink = BlackBoxRowSink;
        let mut writer = SinkWriter::new(&mut sink, rows, dec.backend);
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
        let table = HuffmanTable::from_raw(
            &RawHuffmanTable {
                bits: [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                values: HuffmanValues::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            },
            HuffmanTableRole::Dc,
        )
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

/// Run the NEON IDCT on a caller-provided block. NEON is part of the AArch64
/// architectural baseline. Used by `tests/idct_parity.rs`.
#[cfg(target_arch = "aarch64")]
#[doc(hidden)]
pub fn bench_idct_neon_block(input: &[i16; 64], output: &mut [u8; 64]) {
    let neon = fearless_simd::Level::new()
        .as_neon()
        .expect("AArch64 benchmark host must provide NEON");
    crate::idct::neon::idct_islow(neon, input, output);
}

/// Run the NEON specialization for blocks whose bottom four coefficient rows
/// are known to be zero.
#[cfg(target_arch = "aarch64")]
#[doc(hidden)]
pub fn bench_idct_neon_bottom_half_zero_block(input: &[i16; 64], output: &mut [u8; 64]) {
    let neon = fearless_simd::Level::new()
        .as_neon()
        .expect("AArch64 benchmark host must provide NEON");
    crate::idct::neon::idct_islow_bottom_half_zero(neon, input, output);
}

/// Pre-detected NEON IDCT capability for benchmark loops.
#[cfg(target_arch = "aarch64")]
#[doc(hidden)]
pub struct BenchNeonIdct {
    neon: fearless_simd::Neon,
}

#[cfg(target_arch = "aarch64")]
impl BenchNeonIdct {
    #[must_use]
    pub fn new() -> Self {
        Self {
            neon: fearless_simd::Level::new()
                .as_neon()
                .expect("AArch64 benchmark host must provide NEON"),
        }
    }

    pub fn run(&self, input: &[i16; 64], output: &mut [u8; 64]) {
        crate::idct::neon::idct_islow(self.neon, input, output);
    }

    pub fn run_bottom_half_zero(&self, input: &[i16; 64], output: &mut [u8; 64]) {
        crate::idct::neon::idct_islow_bottom_half_zero(self.neon, input, output);
    }
}

#[cfg(target_arch = "aarch64")]
impl Default for BenchNeonIdct {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(any(target_arch = "x86_64", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchAvx2Dispatch {
    Scalar,
    Avx2,
}

#[cfg(any(target_arch = "x86_64", test))]
const fn select_bench_avx2_dispatch(avx2_available: bool) -> BenchAvx2Dispatch {
    if avx2_available {
        BenchAvx2Dispatch::Avx2
    } else {
        BenchAvx2Dispatch::Scalar
    }
}

/// Run the AVX2 IDCT when the host supports it, otherwise use the scalar
/// reference implementation. The safe wrapper performs its own runtime
/// feature detection and is valid on every supported x86-64 CPU.
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
pub fn bench_idct_avx2_block(input: &[i16; 64], output: &mut [u8; 64]) {
    let avx2 = crate::simd::x86::ExactAvx2::detect();
    match select_bench_avx2_dispatch(avx2.is_some()) {
        BenchAvx2Dispatch::Scalar => idct_islow(input, output),
        BenchAvx2Dispatch::Avx2 => crate::idct::avx2::idct_islow(
            avx2.expect("dispatch selected AVX2 only when its token exists"),
            input,
            output,
        ),
    }
}

/// Pre-detected exact-AVX2 capability for benchmark loops.
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
pub struct BenchAvx2Idct {
    avx2: crate::simd::x86::ExactAvx2,
}

#[cfg(target_arch = "x86_64")]
impl BenchAvx2Idct {
    #[must_use]
    pub fn try_new() -> Option<Self> {
        crate::simd::x86::ExactAvx2::detect().map(|avx2| Self { avx2 })
    }

    pub fn run(&self, input: &[i16; 64], output: &mut [u8; 64]) {
        crate::idct::avx2::idct_islow(self.avx2, input, output);
    }
}

#[cfg(test)]
mod avx2_dispatch_tests {
    use super::{select_bench_avx2_dispatch, BenchAvx2Dispatch};

    #[test]
    fn avx2_dispatch_selects_scalar_when_feature_is_absent() {
        assert_eq!(select_bench_avx2_dispatch(false), BenchAvx2Dispatch::Scalar);
    }

    #[test]
    fn avx2_dispatch_selects_avx2_when_feature_is_present() {
        assert_eq!(select_bench_avx2_dispatch(true), BenchAvx2Dispatch::Avx2);
    }
}

#[cfg(test)]
mod contract_tests;

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
    #[expect(
        clippy::cast_possible_truncation,
        reason = "benchmark fixture values are explicitly masked to one byte"
    )]
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
        bench_rgb_row_pair_from_420(BenchRgb420RowPair::new(
            &self.y_top,
            Some(&self.y_bottom),
            BenchRgb420ChromaRows::new(
                &self.prev_cb,
                &self.curr_cb,
                &self.next_cb,
                &self.prev_cr,
                &self.curr_cr,
                &self.next_cr,
            ),
            &mut self.top,
            Some(&mut self.bottom),
        ));
    }

    /// Run one iteration through the scalar reference path.
    pub fn run_reference(&mut self) {
        bench_rgb_row_pair_from_420_reference(BenchRgb420RowPair::new(
            &self.y_top,
            Some(&self.y_bottom),
            BenchRgb420ChromaRows::new(
                &self.prev_cb,
                &self.curr_cb,
                &self.next_cb,
                &self.prev_cr,
                &self.curr_cr,
                &self.next_cr,
            ),
            &mut self.top,
            Some(&mut self.bottom),
        ));
    }

    /// Run a cropped region through the detected CPU backend.
    pub fn run_cropped(&mut self, start: usize, width: usize) {
        let request = BenchRgb420RowPair::new(
            &self.y_top,
            Some(&self.y_bottom),
            BenchRgb420ChromaRows::new(
                &self.prev_cb,
                &self.curr_cb,
                &self.next_cb,
                &self.prev_cr,
                &self.curr_cr,
                &self.next_cr,
            ),
            &mut self.top[..width * 3],
            Some(&mut self.bottom[..width * 3]),
        );
        bench_rgb_row_pair_from_420_cropped(request, start, width);
    }

    /// Return whether the detected full-row backend produces the scalar bytes.
    #[must_use]
    pub fn backend_matches_reference(&mut self) -> bool {
        let mut expected_top = vec![0u8; self.top.len()];
        let mut expected_bottom = vec![0u8; self.bottom.len()];
        bench_rgb_row_pair_from_420_reference(BenchRgb420RowPair::new(
            &self.y_top,
            Some(&self.y_bottom),
            BenchRgb420ChromaRows::new(
                &self.prev_cb,
                &self.curr_cb,
                &self.next_cb,
                &self.prev_cr,
                &self.curr_cr,
                &self.next_cr,
            ),
            &mut expected_top,
            Some(&mut expected_bottom),
        ));
        self.run();
        self.top == expected_top && self.bottom == expected_bottom
    }

    /// Return whether a cropped backend request produces the same bytes as the
    /// scalar cropped path.
    #[must_use]
    pub fn cropped_backend_matches_reference(&mut self, start: usize, width: usize) -> bool {
        let mut expected_top = vec![0u8; width * 3];
        let mut expected_bottom = vec![0u8; width * 3];
        scalar::fill_rgb_row_pair_from_420_cropped(Rgb420CroppedRowPair::new(
            BenchRgb420RowPair::new(
                &self.y_top,
                Some(&self.y_bottom),
                BenchRgb420ChromaRows::new(
                    &self.prev_cb,
                    &self.curr_cb,
                    &self.next_cb,
                    &self.prev_cr,
                    &self.curr_cr,
                    &self.next_cr,
                ),
                &mut expected_top,
                Some(&mut expected_bottom),
            )
            .into_backend(),
            Rgb420Crop::new(start, width),
        ));
        self.run_cropped(start, width);
        self.top[..width * 3] == expected_top && self.bottom[..width * 3] == expected_bottom
    }
}

/// Borrowed chroma rows for the 4:2:0 row-pair bench helper.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct BenchRgb420ChromaRows<'a> {
    prev_cb: &'a [u8],
    curr_cb: &'a [u8],
    next_cb: &'a [u8],
    prev_cr: &'a [u8],
    curr_cr: &'a [u8],
    next_cr: &'a [u8],
}

impl<'a> BenchRgb420ChromaRows<'a> {
    #[must_use]
    pub fn new(
        prev_cb: &'a [u8],
        curr_cb: &'a [u8],
        next_cb: &'a [u8],
        prev_cr: &'a [u8],
        curr_cr: &'a [u8],
        next_cr: &'a [u8],
    ) -> Self {
        Self {
            prev_cb,
            curr_cb,
            next_cb,
            prev_cr,
            curr_cr,
            next_cr,
        }
    }
}

/// Borrowed row-pair request for the 4:2:0 row-pair bench helper.
#[doc(hidden)]
pub struct BenchRgb420RowPair<'a> {
    y_top: &'a [u8],
    y_bottom: Option<&'a [u8]>,
    chroma: BenchRgb420ChromaRows<'a>,
    dst_top: &'a mut [u8],
    dst_bottom: Option<&'a mut [u8]>,
}

impl<'a> BenchRgb420RowPair<'a> {
    #[must_use]
    pub fn new(
        y_top: &'a [u8],
        y_bottom: Option<&'a [u8]>,
        chroma: BenchRgb420ChromaRows<'a>,
        dst_top: &'a mut [u8],
        dst_bottom: Option<&'a mut [u8]>,
    ) -> Self {
        Self {
            y_top,
            y_bottom,
            chroma,
            dst_top,
            dst_bottom,
        }
    }

    fn into_backend(self) -> Rgb420RowPair<'a> {
        let BenchRgb420RowPair {
            y_top,
            y_bottom,
            chroma,
            dst_top,
            dst_bottom,
        } = self;
        Rgb420RowPair::new(
            y_top,
            y_bottom,
            Rgb420ChromaRows::new(
                chroma.prev_cb,
                chroma.curr_cb,
                chroma.next_cb,
                chroma.prev_cr,
                chroma.curr_cr,
                chroma.next_cr,
            ),
            dst_top,
            dst_bottom,
        )
    }
}

/// Run the platform's normal RGB 4:2:0 row-pair backend on caller-provided
/// inputs. On aarch64 this routes through the detected NEON path.
#[doc(hidden)]
pub fn bench_rgb_row_pair_from_420(request: BenchRgb420RowPair<'_>) {
    Backend::detect().fill_rgb_row_pair_from_420(request.into_backend());
}

/// Run a cropped RGB 4:2:0 row-pair request through the detected backend.
#[doc(hidden)]
pub fn bench_rgb_row_pair_from_420_cropped(
    request: BenchRgb420RowPair<'_>,
    start: usize,
    width: usize,
) {
    Backend::detect().fill_rgb_row_pair_from_420_cropped(Rgb420CroppedRowPair::new(
        request.into_backend(),
        Rgb420Crop::new(start, width),
    ));
}

/// Run the RGB 4:2:0 row-pair backend with dispatch stats.
#[doc(hidden)]
pub fn bench_rgb_row_pair_from_420_with_stats(
    request: BenchRgb420RowPair<'_>,
    stats: &mut Bench420DispatchStats,
) {
    with_420_dispatch_stats(stats, || {
        bench_rgb_row_pair_from_420(request);
    });
}

/// Run the scalar RGB 4:2:0 row-pair reference on caller-provided inputs.
#[doc(hidden)]
pub fn bench_rgb_row_pair_from_420_reference(request: BenchRgb420RowPair<'_>) {
    scalar::fill_rgb_row_pair_from_420(request.into_backend());
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
    #[expect(
        clippy::cast_possible_truncation,
        reason = "benchmark fixture values intentionally retain the low byte of a wrapping pattern"
    )]
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
    input_offset: usize,
    width: usize,
}

impl BenchColorRowScratch {
    /// Create the scratch with a deterministic luminance/chroma pattern.
    #[must_use]
    pub fn new(width: usize) -> Self {
        Self::with_input_offset(width, 0)
    }

    /// Create a fixture whose source rows begin one byte into their backing
    /// allocations, exercising unaligned SIMD loads.
    #[must_use]
    pub fn new_unaligned(width: usize) -> Self {
        Self::with_input_offset(width, 1)
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "benchmark fixture values are explicitly masked to one byte"
    )]
    fn with_input_offset(width: usize, input_offset: usize) -> Self {
        let seed = |offset: usize, scale: usize| -> Vec<u8> {
            (0..width + input_offset)
                .map(|i| ((i.wrapping_mul(scale).wrapping_add(offset)) & 0xFF) as u8)
                .collect()
        };
        Self {
            backend: Backend::detect(),
            y: seed(0, 7),
            cb: seed(64, 5),
            cr: seed(192, 3),
            rgb: vec![0u8; width * 3],
            input_offset,
            width,
        }
    }

    fn rows(&self) -> (&[u8], &[u8], &[u8]) {
        let range = self.input_offset..self.input_offset + self.width;
        (
            &self.y[range.clone()],
            &self.cb[range.clone()],
            &self.cr[range],
        )
    }

    /// Run one iteration of the scalar per-pixel YCbCr→RGB conversion.
    pub fn run_scalar(&mut self) {
        let input_offset = self.input_offset;
        let width = self.width;
        for (((&y, &cb), &cr), pixel) in self.y[input_offset..input_offset + width]
            .iter()
            .zip(self.cb[input_offset..input_offset + width].iter())
            .zip(self.cr[input_offset..input_offset + width].iter())
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
        let input_offset = self.input_offset;
        let width = self.width;
        self.backend.fill_rgb_row_from_ycbcr(
            &self.y[input_offset..input_offset + width],
            &self.cb[input_offset..input_offset + width],
            &self.cr[input_offset..input_offset + width],
            &mut self.rgb,
        );
    }

    /// Return whether the detected backend produces the scalar row bytes.
    #[must_use]
    pub fn backend_matches_scalar(&mut self) -> bool {
        let (y, cb, cr) = self.rows();
        let mut expected = vec![0u8; self.width * 3];
        scalar::fill_rgb_row_from_ycbcr(y, cb, cr, &mut expected);
        self.run_backend();
        self.rgb == expected
    }
}

/// Pre-allocated planar grayscale-to-RGB row benchmark input.
#[doc(hidden)]
pub struct BenchGrayRowScratch {
    backend: Backend,
    gray: Vec<u8>,
    rgb: Vec<u8>,
}

impl BenchGrayRowScratch {
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "benchmark fixture values intentionally retain the low byte"
    )]
    pub fn new(width: usize) -> Self {
        Self {
            backend: Backend::detect(),
            gray: (0..width).map(|i| i.wrapping_mul(37) as u8).collect(),
            rgb: vec![0u8; width * 3],
        }
    }

    pub fn run_backend(&mut self) {
        self.backend
            .fill_rgb_row_from_gray(&self.gray, &mut self.rgb);
    }

    #[must_use]
    pub fn backend_matches_scalar(&mut self) -> bool {
        let mut expected = vec![0u8; self.rgb.len()];
        scalar::fill_rgb_row_from_gray(&self.gray, &mut expected);
        self.run_backend();
        self.rgb == expected
    }
}

/// Pre-allocated planar RGB-to-interleaved-RGB row benchmark input.
#[doc(hidden)]
pub struct BenchRgbRowScratch {
    backend: Backend,
    r: Vec<u8>,
    g: Vec<u8>,
    b: Vec<u8>,
    rgb: Vec<u8>,
}

impl BenchRgbRowScratch {
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "benchmark fixture values intentionally retain the low byte"
    )]
    pub fn new(width: usize) -> Self {
        let seed = |offset: usize, scale: usize| -> Vec<u8> {
            (0..width)
                .map(|i| i.wrapping_mul(scale).wrapping_add(offset) as u8)
                .collect()
        };
        Self {
            backend: Backend::detect(),
            r: seed(11, 37),
            g: seed(47, 29),
            b: seed(89, 19),
            rgb: vec![0u8; width * 3],
        }
    }

    pub fn run_backend(&mut self) {
        self.backend
            .fill_rgb_row_from_rgb(&self.r, &self.g, &self.b, &mut self.rgb);
    }

    #[must_use]
    pub fn backend_matches_scalar(&mut self) -> bool {
        let mut expected = vec![0u8; self.rgb.len()];
        scalar::fill_rgb_row_from_rgb(&self.r, &self.g, &self.b, &mut expected);
        self.run_backend();
        self.rgb == expected
    }
}
