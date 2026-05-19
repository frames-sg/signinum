//! Scalar HTJ2K block decoding.

use alloc::vec::Vec;

use super::build::CodeBlock;
use super::decode::DecompositionStorage;
use super::ht_tables::{UVLC_TABLE0, UVLC_TABLE1, VLC_TABLE0, VLC_TABLE1};
use crate::error::{bail, DecodingError, Result};
use crate::profile;

#[derive(Default)]
pub(crate) struct HtBlockDecodeContext {
    coefficients: Vec<u32>,
    scratch: HtBlockDecodeScratch,
    width: u32,
    height: u32,
}

impl HtBlockDecodeContext {
    fn reset(&mut self, code_block: &CodeBlock) {
        self.width = code_block.rect.width();
        self.height = code_block.rect.height();
        self.coefficients.clear();
        self.coefficients
            .resize((self.width * self.height) as usize, 0);
    }

    pub(crate) fn coefficient_rows(&self) -> impl Iterator<Item = &[u32]> {
        self.coefficients.chunks_exact(self.width as usize)
    }
}

#[derive(Default)]
pub(crate) struct HtBlockDecodeScratch {
    cleanup: Vec<u16>,
    v_n: Vec<u32>,
    sigma: Vec<u16>,
    prev_row_sig: Vec<u16>,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HtBlockDecodeStats {
    pub(crate) blocks: u128,
    pub(crate) refinement_blocks: u128,
    pub(crate) cleanup_bytes: u128,
    pub(crate) refinement_bytes: u128,
    pub(crate) ht_cleanup_us: u128,
    pub(crate) ht_mag_sgn_us: u128,
    pub(crate) ht_sigma_us: u128,
    pub(crate) ht_sigprop_us: u128,
    pub(crate) ht_magref_us: u128,
}

impl HtBlockDecodeStats {
    fn record_block(&mut self, cleanup_bytes: usize, refinement_bytes: usize) {
        self.blocks += 1;
        self.cleanup_bytes += cleanup_bytes as u128;
        if refinement_bytes > 0 {
            self.refinement_blocks += 1;
            self.refinement_bytes += refinement_bytes as u128;
        }
    }
}

pub(crate) const PHASE_LIMIT_CLEANUP: u8 = 0;
pub(crate) const PHASE_LIMIT_SIGPROP: u8 = 1;
pub(crate) const PHASE_LIMIT_MAGREF: u8 = 2;

const SIGPROP_SPREAD_MASKS: [u32; 16] = [
    0x33, 0x76, 0xEC, 0xC8, 0x330, 0x760, 0xEC0, 0xC80, 0x3300, 0x7600, 0xEC00, 0xC800, 0x33000,
    0x76000, 0xEC000, 0xC8000,
];

trait HtDecodeObserver {
    #[inline(always)]
    fn record_block(&mut self, _cleanup_bytes: usize, _refinement_bytes: usize) {}

    #[inline(always)]
    fn phase_start(&self) -> Option<profile::ProfileInstant> {
        None
    }

    #[inline(always)]
    fn add_cleanup_us(&mut self, _start: Option<profile::ProfileInstant>) {}

    #[inline(always)]
    fn add_mag_sgn_us(&mut self, _start: Option<profile::ProfileInstant>) {}

    #[inline(always)]
    fn add_sigma_us(&mut self, _start: Option<profile::ProfileInstant>) {}

    #[inline(always)]
    fn add_sigprop_us(&mut self, _start: Option<profile::ProfileInstant>) {}

    #[inline(always)]
    fn add_magref_us(&mut self, _start: Option<profile::ProfileInstant>) {}
}

struct NoHtDecodeStats;

impl HtDecodeObserver for NoHtDecodeStats {}

struct RecordingHtDecodeStats<'a> {
    stats: &'a mut HtBlockDecodeStats,
    profile_enabled: bool,
}

impl HtDecodeObserver for RecordingHtDecodeStats<'_> {
    #[inline(always)]
    fn record_block(&mut self, cleanup_bytes: usize, refinement_bytes: usize) {
        self.stats.record_block(cleanup_bytes, refinement_bytes);
    }

    #[inline(always)]
    fn phase_start(&self) -> Option<profile::ProfileInstant> {
        if self.profile_enabled {
            profile::profile_now(true)
        } else {
            None
        }
    }

    #[inline(always)]
    fn add_cleanup_us(&mut self, start: Option<profile::ProfileInstant>) {
        if self.profile_enabled {
            self.stats.ht_cleanup_us += profile::elapsed_us(start);
        }
    }

    #[inline(always)]
    fn add_mag_sgn_us(&mut self, start: Option<profile::ProfileInstant>) {
        if self.profile_enabled {
            self.stats.ht_mag_sgn_us += profile::elapsed_us(start);
        }
    }

    #[inline(always)]
    fn add_sigma_us(&mut self, start: Option<profile::ProfileInstant>) {
        if self.profile_enabled {
            self.stats.ht_sigma_us += profile::elapsed_us(start);
        }
    }

    #[inline(always)]
    fn add_sigprop_us(&mut self, start: Option<profile::ProfileInstant>) {
        if self.profile_enabled {
            self.stats.ht_sigprop_us += profile::elapsed_us(start);
        }
    }

    #[inline(always)]
    fn add_magref_us(&mut self, start: Option<profile::ProfileInstant>) {
        if self.profile_enabled {
            self.stats.ht_magref_us += profile::elapsed_us(start);
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HtBlockDecodeScratchCapacities {
    cleanup: usize,
    v_n: usize,
    sigma: usize,
    prev_row_sig: usize,
}

#[cfg(test)]
impl HtBlockDecodeScratch {
    fn capacities_for_test(&self) -> HtBlockDecodeScratchCapacities {
        HtBlockDecodeScratchCapacities {
            cleanup: self.cleanup.capacity(),
            v_n: self.v_n.capacity(),
            sigma: self.sigma.capacity(),
            prev_row_sig: self.prev_row_sig.capacity(),
        }
    }

    fn poison_for_test(&mut self) {
        self.cleanup.fill(u16::MAX);
        self.v_n.fill(u32::MAX);
        self.sigma.fill(u16::MAX);
        self.prev_row_sig.fill(u16::MAX);
    }
}

#[inline(always)]
fn zeroed_u16_scratch(buffer: &mut Vec<u16>, len: usize) -> &mut [u16] {
    if buffer.len() < len {
        buffer.resize(len, 0);
    }
    buffer[..len].fill(0);

    &mut buffer[..len]
}

#[cfg(test)]
fn zeroed_u32_scratch(buffer: &mut Vec<u32>, len: usize) -> &mut [u32] {
    if buffer.len() < len {
        buffer.resize(len, 0);
    }
    buffer[..len].fill(0);

    &mut buffer[..len]
}

#[inline(always)]
fn resized_u16_scratch(buffer: &mut Vec<u16>, len: usize) -> &mut [u16] {
    if buffer.len() < len {
        buffer.resize(len, 0);
    }

    &mut buffer[..len]
}

#[inline(always)]
fn resized_u32_scratch(buffer: &mut Vec<u32>, len: usize) -> &mut [u32] {
    if buffer.len() < len {
        buffer.resize(len, 0);
    }

    &mut buffer[..len]
}

pub(crate) fn coefficient_to_i32(value: u32, k_max: u8) -> i32 {
    let shift = 31_u32.saturating_sub(k_max as u32);
    let magnitude = ((value & 0x7FFF_FFFF) >> shift) as i32;

    if (value & 0x8000_0000) != 0 {
        -magnitude
    } else {
        magnitude
    }
}

pub(crate) fn decode_with_stats(
    code_block: &CodeBlock,
    total_bitplanes: u8,
    stripe_causal: bool,
    ctx: &mut HtBlockDecodeContext,
    storage: &DecompositionStorage<'_>,
    strict: bool,
    stats: Option<&mut HtBlockDecodeStats>,
    profile_enabled: bool,
) -> Result<()> {
    ctx.reset(code_block);

    if total_bitplanes == 0 {
        return Ok(());
    }

    if total_bitplanes > 31 {
        bail!(DecodingError::TooManyBitplanes);
    }

    let actual_bitplanes = if strict {
        total_bitplanes
            .checked_sub(code_block.missing_bit_planes)
            .ok_or(DecodingError::InvalidBitplaneCount)?
    } else {
        total_bitplanes.saturating_sub(code_block.missing_bit_planes)
    };

    let max_coding_passes = if actual_bitplanes == 0 {
        0
    } else {
        1 + 3 * (actual_bitplanes - 1)
    };

    if code_block.number_of_coding_passes > max_coding_passes && strict {
        bail!(DecodingError::TooManyCodingPasses);
    }

    if code_block.number_of_coding_passes == 0 || actual_bitplanes == 0 {
        return Ok(());
    }

    let segments = collect_code_block_segments(code_block, storage)?;
    decode_segments_validated_with_scratch_for_phase::<PHASE_LIMIT_MAGREF>(
        &segments,
        code_block.missing_bit_planes,
        total_bitplanes,
        code_block.number_of_coding_passes,
        stripe_causal,
        strict,
        &mut ctx.coefficients,
        code_block.rect.width(),
        code_block.rect.height(),
        code_block.rect.width(),
        &mut ctx.scratch,
        stats,
        profile_enabled,
    )
}

pub(crate) struct CombinedCodeBlockData {
    pub(crate) data: Vec<u8>,
    pub(crate) cleanup_length: u32,
    pub(crate) refinement_length: u32,
}

pub(crate) struct HtCodeBlockSegments<'a> {
    pub(crate) cleanup: &'a [u8],
    pub(crate) refinement: &'a [u8],
}

impl<'a> HtCodeBlockSegments<'a> {
    pub(crate) fn from_combined_payload(
        data: &'a [u8],
        cleanup_length: u32,
        refinement_length: u32,
    ) -> Result<Self> {
        let cleanup_len = cleanup_length as usize;
        let refinement_len = refinement_length as usize;
        let total_len = cleanup_len
            .checked_add(refinement_len)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        if data.len() < total_len {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }

        Ok(Self {
            cleanup: &data[..cleanup_len],
            refinement: &data[cleanup_len..total_len],
        })
    }
}

pub(crate) struct HtSigPropBenchmarkState {
    refinement_data: Vec<u8>,
    sigma: Vec<u16>,
    prev_row_sig: Vec<u16>,
    width: u32,
    height: u32,
    stride: u32,
    mstr: usize,
    stripe_causal: bool,
    p: u32,
}

impl HtSigPropBenchmarkState {
    pub(crate) fn output_len(&self) -> usize {
        if self.height == 0 {
            0
        } else {
            (self.stride as usize * (self.height as usize - 1)) + self.width as usize
        }
    }
}

pub(crate) fn prepare_sigprop_benchmark_state(
    segments: &HtCodeBlockSegments<'_>,
    missing_bit_planes: u8,
    total_bitplanes: u8,
    number_of_coding_passes: u8,
    stripe_causal: bool,
    strict: bool,
    width: u32,
    height: u32,
    stride: u32,
) -> Result<HtSigPropBenchmarkState> {
    if !validate_combined_decode(
        missing_bit_planes,
        total_bitplanes,
        number_of_coding_passes,
        strict,
    )? {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    if number_of_coding_passes < 2 || segments.refinement.is_empty() || missing_bit_planes > 28 {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }

    let lcup = segments.cleanup.len();
    let scup = cleanup_segment_suffix_length(segments.cleanup, lcup)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let sstr = cleanup_symbol_stride(width);
    let quad_rows = height.div_ceil(2) as usize;
    let mut cleanup = vec![0u16; sstr * (quad_rows + 1)];
    decode_cleanup_symbols(
        segments.cleanup,
        lcup,
        scup,
        width,
        height,
        sstr,
        &mut cleanup,
    )
    .ok_or(DecodingError::CodeBlockDecodeFailure)?;

    let mstr = sigma_stride(width);
    let sigma_rows = height.div_ceil(4) as usize + 1;
    let mut sigma = vec![0u16; sigma_rows * mstr];
    build_sigma_from_cleanup_phase(&cleanup, &mut sigma, width, height, sstr, mstr)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;

    Ok(HtSigPropBenchmarkState {
        refinement_data: segments.refinement.to_vec(),
        sigma,
        prev_row_sig: vec![0u16; width.div_ceil(4) as usize + 8],
        width,
        height,
        stride,
        mstr,
        stripe_causal,
        p: 30 - u32::from(missing_bit_planes),
    })
}

pub(crate) fn decode_sigprop_benchmark_state(
    state: &mut HtSigPropBenchmarkState,
    decoded_data: &mut [u32],
) -> Result<()> {
    if decoded_data.len() < state.output_len() {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }

    apply_significance_propagation_phase(
        &state.refinement_data,
        &state.sigma,
        decoded_data,
        state.width,
        state.height,
        state.stride,
        state.mstr,
        state.stripe_causal,
        state.p,
        &mut state.prev_row_sig,
    )
    .ok_or(DecodingError::CodeBlockDecodeFailure.into())
}

#[cfg(test)]
impl CombinedCodeBlockData {
    pub(crate) fn segments(&self) -> Result<HtCodeBlockSegments<'_>> {
        HtCodeBlockSegments::from_combined_payload(
            &self.data,
            self.cleanup_length,
            self.refinement_length,
        )
    }
}

pub(crate) fn collect_code_block_segments<'a>(
    code_block: &CodeBlock,
    storage: &'a DecompositionStorage<'a>,
) -> Result<HtCodeBlockSegments<'a>> {
    let mut cleanup = None;
    let mut refinement = None;

    for layer in &storage.layers[code_block.layers.start..code_block.layers.end] {
        let Some(range) = layer.segments.clone() else {
            continue;
        };

        for segment in &storage.segments[range] {
            match segment.idx {
                0 if cleanup.is_none() => {
                    cleanup = Some(segment.data);
                }
                1 if refinement.is_none() => {
                    refinement = Some(segment.data);
                }
                _ => bail!(DecodingError::UnsupportedFeature(
                    "unexpected HTJ2K segment layout"
                )),
            }
        }
    }

    let Some(cleanup) = cleanup else {
        bail!(DecodingError::CodeBlockDecodeFailure);
    };

    Ok(HtCodeBlockSegments {
        cleanup,
        refinement: refinement.unwrap_or(&[]),
    })
}

pub(crate) fn collect_code_block_data<'a>(
    code_block: &CodeBlock,
    storage: &'a DecompositionStorage<'a>,
) -> Result<CombinedCodeBlockData> {
    let segments = collect_code_block_segments(code_block, storage)?;
    let cleanup_length =
        u32::try_from(segments.cleanup.len()).map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
    let refinement_length = u32::try_from(segments.refinement.len())
        .map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
    let mut data = Vec::with_capacity(segments.cleanup.len() + segments.refinement.len());
    data.extend_from_slice(segments.cleanup);
    data.extend_from_slice(segments.refinement);

    Ok(CombinedCodeBlockData {
        data,
        cleanup_length,
        refinement_length,
    })
}

#[inline(always)]
fn decode_segments_with_scratch_for_phase<const PHASE_LIMIT: u8>(
    segments: &HtCodeBlockSegments<'_>,
    missing_bit_planes: u8,
    number_of_coding_passes: u8,
    width: u32,
    height: u32,
    stride: u32,
    stripe_causal: bool,
    decoded_data: &mut [u32],
    scratch: &mut HtBlockDecodeScratch,
    stats: Option<&mut HtBlockDecodeStats>,
    profile_enabled: bool,
) -> Result<()> {
    let decoded = if let Some(stats) = stats {
        let mut observer = RecordingHtDecodeStats {
            stats,
            profile_enabled,
        };
        decode_impl::<PHASE_LIMIT, _>(
            segments.cleanup,
            segments.refinement,
            decoded_data,
            missing_bit_planes as u32,
            number_of_coding_passes as u32,
            width,
            height,
            stride,
            stripe_causal,
            scratch,
            &mut observer,
        )
    } else {
        let mut observer = NoHtDecodeStats;
        decode_impl::<PHASE_LIMIT, _>(
            segments.cleanup,
            segments.refinement,
            decoded_data,
            missing_bit_planes as u32,
            number_of_coding_passes as u32,
            width,
            height,
            stride,
            stripe_causal,
            scratch,
            &mut observer,
        )
    };

    decoded.ok_or(DecodingError::CodeBlockDecodeFailure.into())
}

#[cfg(test)]
pub(crate) fn decode_segments_validated(
    segments: &HtCodeBlockSegments<'_>,
    missing_bit_planes: u8,
    total_bitplanes: u8,
    number_of_coding_passes: u8,
    stripe_causal: bool,
    strict: bool,
    decoded_data: &mut [u32],
    width: u32,
    height: u32,
    stride: u32,
) -> Result<()> {
    decode_segments_validated_for_phase::<PHASE_LIMIT_MAGREF>(
        segments,
        missing_bit_planes,
        total_bitplanes,
        number_of_coding_passes,
        stripe_causal,
        strict,
        decoded_data,
        width,
        height,
        stride,
    )
}

#[inline(always)]
#[cfg(test)]
pub(crate) fn decode_segments_validated_for_phase<const PHASE_LIMIT: u8>(
    segments: &HtCodeBlockSegments<'_>,
    missing_bit_planes: u8,
    total_bitplanes: u8,
    number_of_coding_passes: u8,
    stripe_causal: bool,
    strict: bool,
    decoded_data: &mut [u32],
    width: u32,
    height: u32,
    stride: u32,
) -> Result<()> {
    if !validate_combined_decode(
        missing_bit_planes,
        total_bitplanes,
        number_of_coding_passes,
        strict,
    )? {
        return Ok(());
    }

    let mut scratch = HtBlockDecodeScratch::default();
    decode_segments_with_scratch_for_phase::<PHASE_LIMIT>(
        segments,
        missing_bit_planes,
        number_of_coding_passes,
        width,
        height,
        stride,
        stripe_causal,
        decoded_data,
        &mut scratch,
        None,
        false,
    )
}

#[cfg(test)]
fn decode_segments_validated_with_scratch(
    segments: &HtCodeBlockSegments<'_>,
    missing_bit_planes: u8,
    total_bitplanes: u8,
    number_of_coding_passes: u8,
    stripe_causal: bool,
    strict: bool,
    decoded_data: &mut [u32],
    width: u32,
    height: u32,
    stride: u32,
    scratch: &mut HtBlockDecodeScratch,
) -> Result<()> {
    decode_segments_validated_with_scratch_for_phase::<PHASE_LIMIT_MAGREF>(
        segments,
        missing_bit_planes,
        total_bitplanes,
        number_of_coding_passes,
        stripe_causal,
        strict,
        decoded_data,
        width,
        height,
        stride,
        scratch,
        None,
        false,
    )
}

#[inline(always)]
pub(crate) fn decode_segments_validated_with_scratch_for_phase<const PHASE_LIMIT: u8>(
    segments: &HtCodeBlockSegments<'_>,
    missing_bit_planes: u8,
    total_bitplanes: u8,
    number_of_coding_passes: u8,
    stripe_causal: bool,
    strict: bool,
    decoded_data: &mut [u32],
    width: u32,
    height: u32,
    stride: u32,
    scratch: &mut HtBlockDecodeScratch,
    stats: Option<&mut HtBlockDecodeStats>,
    profile_enabled: bool,
) -> Result<()> {
    if !validate_combined_decode(
        missing_bit_planes,
        total_bitplanes,
        number_of_coding_passes,
        strict,
    )? {
        return Ok(());
    }

    decode_segments_with_scratch_for_phase::<PHASE_LIMIT>(
        segments,
        missing_bit_planes,
        number_of_coding_passes,
        width,
        height,
        stride,
        stripe_causal,
        decoded_data,
        scratch,
        stats,
        profile_enabled,
    )
}

fn validate_combined_decode(
    missing_bit_planes: u8,
    total_bitplanes: u8,
    number_of_coding_passes: u8,
    strict: bool,
) -> Result<bool> {
    if total_bitplanes == 0 {
        return Ok(false);
    }

    if total_bitplanes > 31 {
        bail!(DecodingError::TooManyBitplanes);
    }

    let actual_bitplanes = if strict {
        total_bitplanes
            .checked_sub(missing_bit_planes)
            .ok_or(DecodingError::InvalidBitplaneCount)?
    } else {
        total_bitplanes.saturating_sub(missing_bit_planes)
    };

    let max_coding_passes = if actual_bitplanes == 0 {
        0
    } else {
        1 + 3 * (actual_bitplanes - 1)
    };

    if number_of_coding_passes > max_coding_passes && strict {
        bail!(DecodingError::TooManyCodingPasses);
    }

    Ok(number_of_coding_passes != 0 && actual_bitplanes != 0)
}

#[cfg(test)]
pub(crate) fn decode_combined_validated(
    combined: &CombinedCodeBlockData,
    missing_bit_planes: u8,
    total_bitplanes: u8,
    number_of_coding_passes: u8,
    stripe_causal: bool,
    strict: bool,
    decoded_data: &mut [u32],
    width: u32,
    height: u32,
    stride: u32,
) -> Result<()> {
    let segments = combined.segments()?;
    decode_segments_validated(
        &segments,
        missing_bit_planes,
        total_bitplanes,
        number_of_coding_passes,
        stripe_causal,
        strict,
        decoded_data,
        width,
        height,
        stride,
    )
}

#[cfg(test)]
fn decode_combined_validated_with_scratch(
    combined: &CombinedCodeBlockData,
    missing_bit_planes: u8,
    total_bitplanes: u8,
    number_of_coding_passes: u8,
    stripe_causal: bool,
    strict: bool,
    decoded_data: &mut [u32],
    width: u32,
    height: u32,
    stride: u32,
    scratch: &mut HtBlockDecodeScratch,
) -> Result<()> {
    let segments = combined.segments()?;
    decode_segments_validated_with_scratch(
        &segments,
        missing_bit_planes,
        total_bitplanes,
        number_of_coding_passes,
        stripe_causal,
        strict,
        decoded_data,
        width,
        height,
        stride,
        scratch,
    )
}

struct MelDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    remaining: usize,
    unstuff: bool,
    current_byte: u8,
    bits_left: u8,
    k: usize,
    num_runs: usize,
    runs: u64,
}

impl<'a> MelDecoder<'a> {
    fn new(data: &'a [u8], lcup: usize, scup: usize) -> Self {
        Self {
            data,
            pos: lcup - scup,
            remaining: scup - 1,
            unstuff: false,
            current_byte: 0,
            bits_left: 0,
            k: 0,
            num_runs: 0,
            runs: 0,
        }
    }

    fn read_bit(&mut self) -> Option<u32> {
        if self.bits_left == 0 {
            let mut byte = if self.remaining > 0 {
                let byte = self.data.get(self.pos).copied()?;
                self.pos += 1;
                self.remaining -= 1;
                byte
            } else {
                0xFF
            };

            if self.remaining == 0 {
                byte |= 0x0F;
            }

            self.current_byte = byte;
            self.bits_left = 8 - u8::from(self.unstuff);
            self.unstuff = byte == 0xFF;
        }

        self.bits_left -= 1;
        Some(((self.current_byte >> self.bits_left) & 1) as u32)
    }

    fn read_bits(&mut self, count: usize) -> Option<u32> {
        let mut value = 0;

        for _ in 0..count {
            value = (value << 1) | self.read_bit()?;
        }

        Some(value)
    }

    fn decode_more_runs(&mut self) -> Option<()> {
        const MEL_EXP: [usize; 13] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5];

        while self.num_runs < 8 {
            let eval = MEL_EXP[self.k];
            let first = self.read_bit()?;
            let run = if first == 1 {
                self.k = (self.k + 1).min(12);
                ((1usize << eval) - 1) << 1
            } else {
                self.k = self.k.saturating_sub(1);
                (self.read_bits(eval)? as usize) << 1 | 1
            };

            self.runs |= (run as u64) << (self.num_runs * 7);
            self.num_runs += 1;

            if eval == 5 && first == 0 && self.num_runs >= 8 {
                break;
            }
        }

        Some(())
    }

    fn get_run(&mut self) -> Option<i32> {
        if self.num_runs == 0 {
            self.decode_more_runs()?;
        }

        let run = (self.runs & 0x7F) as i32;
        self.runs >>= 7;
        self.num_runs -= 1;
        Some(run)
    }
}

struct ForwardBitReader<'a, const PAD: u8> {
    data: &'a [u8],
    pos: usize,
    tmp: u64,
    bits: u32,
    unstuff: bool,
}

impl<'a, const PAD: u8> ForwardBitReader<'a, PAD> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            tmp: 0,
            bits: 0,
            unstuff: false,
        }
    }

    fn fill(&mut self) {
        while self.bits <= 32 {
            let byte = if self.pos < self.data.len() {
                let byte = self.data[self.pos];
                self.pos += 1;
                byte
            } else {
                PAD
            };

            self.tmp |= (byte as u64) << self.bits;
            self.bits += 8 - u32::from(self.unstuff);
            self.unstuff = byte == 0xFF;
        }
    }

    fn fetch(&mut self) -> u32 {
        if self.bits < 32 {
            self.fill();
        }

        self.tmp as u32
    }

    fn advance(&mut self, count: u32) {
        debug_assert!(count <= self.bits);
        self.tmp >>= count;
        self.bits -= count;
    }
}

struct ReverseBitReader<'a> {
    data: &'a [u8],
    pos: isize,
    remaining: usize,
    tmp: u64,
    bits: u32,
    unstuff: bool,
}

impl<'a> ReverseBitReader<'a> {
    fn new_vlc(data: &'a [u8], lcup: usize, scup: usize) -> Self {
        let d = data[lcup - 2];
        let tmp = u64::from(d >> 4);
        let bits = 4 - u32::from((tmp & 0x7) == 0x7);

        Self {
            data,
            pos: lcup as isize - 3,
            remaining: scup - 2,
            tmp,
            bits,
            unstuff: (d | 0x0F) > 0x8F,
        }
    }

    fn new_mrp(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: data.len() as isize - 1,
            remaining: data.len(),
            tmp: 0,
            bits: 0,
            unstuff: true,
        }
    }

    fn fill(&mut self) {
        while self.bits <= 32 {
            let byte = if self.remaining > 0 {
                let byte = self.data[self.pos as usize];
                self.pos -= 1;
                self.remaining -= 1;
                byte
            } else {
                0
            };

            let d_bits = 8 - u32::from(self.unstuff && (byte & 0x7F) == 0x7F);
            self.tmp |= (byte as u64) << self.bits;
            self.bits += d_bits;
            self.unstuff = byte > 0x8F;
        }
    }

    fn fetch(&mut self) -> u32 {
        if self.bits < 32 {
            self.fill();
        }

        self.tmp as u32
    }

    fn advance(&mut self, count: u32) -> u32 {
        debug_assert!(count <= self.bits);
        self.tmp >>= count;
        self.bits -= count;
        self.tmp as u32
    }
}

#[inline(always)]
fn read_u32_pair(values: &[u16], index: usize) -> u32 {
    u32::from(values[index]) | (u32::from(values[index + 1]) << 16)
}

fn sample_mask(bit: u32) -> u32 {
    1 << (4 + bit)
}

fn decode_mag_sgn_sample_with_vn(
    magsgn: &mut ForwardBitReader<0xFF>,
    inf: u32,
    bit: u32,
    uq: u32,
    p: u32,
) -> (u32, u32) {
    if (inf & sample_mask(bit)) == 0 {
        return (0, 0);
    }

    let ms_val = magsgn.fetch();
    let m_n = uq - ((inf >> (12 + bit)) & 1);
    magsgn.advance(m_n);

    let mut value = ms_val << 31;
    let mask = if m_n == 0 { 0 } else { (1_u32 << m_n) - 1 };
    let mut v_n = ms_val & mask;
    v_n |= ((inf >> (8 + bit)) & 1) << m_n;
    v_n |= 1;
    value |= (v_n + 2) << (p - 1);
    (value, v_n)
}

fn cleanup_symbol_stride(width: u32) -> usize {
    ((width + 2 + 7) & !7) as usize
}

fn sigma_stride(width: u32) -> usize {
    ((width.div_ceil(4) + 2 + 7) & !7) as usize
}

fn cleanup_segment_suffix_length(coded_data: &[u8], lcup: usize) -> Option<usize> {
    if lcup < 2 || coded_data.len() < lcup {
        return None;
    }

    let scup = ((coded_data[lcup - 1] as usize) << 4) + usize::from(coded_data[lcup - 2] & 0x0F);
    if !(2..=lcup).contains(&scup) || scup > 4079 {
        return None;
    }

    Some(scup)
}

#[inline(never)]
fn decode_cleanup_symbols(
    coded_data: &[u8],
    lcup: usize,
    scup: usize,
    width: u32,
    height: u32,
    sstr: usize,
    scratch: &mut [u16],
) -> Option<()> {
    let quad_rows = height.div_ceil(2) as usize;
    if scratch.len() < sstr * (quad_rows + 1) {
        return None;
    }

    let mut mel = MelDecoder::new(coded_data, lcup, scup);
    let mut vlc = ReverseBitReader::new_vlc(coded_data, lcup, scup);
    let mut run = mel.get_run()?;
    let mut c_q = 0u32;
    let mut row_offset = 0usize;
    let mut x = 0u32;

    while x < width {
        let mut vlc_val = vlc.fetch();
        let mut t0 = u32::from(VLC_TABLE0[(c_q + (vlc_val & 0x7F)) as usize]);
        if c_q == 0 {
            run -= 2;
            t0 = if run == -1 { t0 } else { 0 };
            if run < 0 {
                run = mel.get_run()?;
            }
        }
        scratch[row_offset] = t0 as u16;
        x += 2;
        c_q = ((t0 & 0x10) << 3) | ((t0 & 0xE0) << 2);
        vlc_val = vlc.advance(t0 & 0x7);

        let mut t1 = u32::from(VLC_TABLE0[(c_q + (vlc_val & 0x7F)) as usize]);
        if c_q == 0 && x < width {
            run -= 2;
            t1 = if run == -1 { t1 } else { 0 };
            if run < 0 {
                run = mel.get_run()?;
            }
        }
        if x >= width {
            t1 = 0;
        }
        scratch[row_offset + 2] = t1 as u16;
        x += 2;
        c_q = ((t1 & 0x10) << 3) | ((t1 & 0xE0) << 2);
        vlc_val = vlc.advance(t1 & 0x7);

        let mut uvlc_mode = ((t0 & 0x8) << 3) | ((t1 & 0x8) << 4);
        if uvlc_mode == 0xC0 {
            run -= 2;
            if run == -1 {
                uvlc_mode += 0x40;
            }
            if run < 0 {
                run = mel.get_run()?;
            }
        }

        let mut uvlc_entry = u32::from(UVLC_TABLE0[(uvlc_mode + (vlc_val & 0x3F)) as usize]);
        vlc_val = vlc.advance(uvlc_entry & 0x7);
        uvlc_entry >>= 3;
        let mut len = uvlc_entry & 0xF;
        let tmp = vlc_val & ((1_u32 << len) - 1);
        vlc_val = vlc.advance(len);
        uvlc_entry >>= 4;
        len = uvlc_entry & 0x7;
        uvlc_entry >>= 3;
        scratch[row_offset + 1] = (1 + (uvlc_entry & 0x7) + (tmp & !(0xFF_u32 << len))) as u16;
        scratch[row_offset + 3] = (1 + (uvlc_entry >> 3) + (tmp >> len)) as u16;

        row_offset += 4;
    }
    scratch[row_offset] = 0;
    scratch[row_offset + 1] = 0;

    for y in (2..height).step_by(2) {
        let row_base = (y >> 1) as usize * sstr;
        let prev_base = row_base - sstr;
        let mut x = 0u32;
        let mut c_q = 0u32;
        let mut row_offset = row_base;

        while x < width {
            c_q |= (u32::from(scratch[prev_base + (row_offset - row_base)]) & 0xA0) << 2;
            c_q |= (u32::from(scratch[prev_base + (row_offset - row_base) + 2]) & 0x20) << 4;

            let mut vlc_val = vlc.fetch();
            let mut t0 = u32::from(VLC_TABLE1[(c_q + (vlc_val & 0x7F)) as usize]);
            if c_q == 0 {
                run -= 2;
                t0 = if run == -1 { t0 } else { 0 };
                if run < 0 {
                    run = mel.get_run()?;
                }
            }
            scratch[row_offset] = t0 as u16;
            x += 2;

            c_q = ((t0 & 0x40) << 2) | ((t0 & 0x80) << 1);
            c_q |= u32::from(scratch[prev_base + (row_offset - row_base)]) & 0x80;
            c_q |= (u32::from(scratch[prev_base + (row_offset - row_base) + 2]) & 0xA0) << 2;
            c_q |= (u32::from(scratch[prev_base + (row_offset - row_base) + 4]) & 0x20) << 4;
            vlc_val = vlc.advance(t0 & 0x7);

            let mut t1 = u32::from(VLC_TABLE1[(c_q + (vlc_val & 0x7F)) as usize]);
            if c_q == 0 && x < width {
                run -= 2;
                t1 = if run == -1 { t1 } else { 0 };
                if run < 0 {
                    run = mel.get_run()?;
                }
            }
            if x >= width {
                t1 = 0;
            }
            scratch[row_offset + 2] = t1 as u16;
            x += 2;

            c_q = ((t1 & 0x40) << 2) | ((t1 & 0x80) << 1);
            c_q |= u32::from(scratch[prev_base + (row_offset - row_base) + 2]) & 0x80;
            vlc_val = vlc.advance(t1 & 0x7);

            let uvlc_mode = ((t0 & 0x8) << 3) | ((t1 & 0x8) << 4);
            let mut uvlc_entry = u32::from(UVLC_TABLE1[(uvlc_mode + (vlc_val & 0x3F)) as usize]);
            vlc_val = vlc.advance(uvlc_entry & 0x7);
            uvlc_entry >>= 3;
            let mut len = uvlc_entry & 0xF;
            let tmp = vlc_val & ((1_u32 << len) - 1);
            vlc_val = vlc.advance(len);
            uvlc_entry >>= 4;
            len = uvlc_entry & 0x7;
            uvlc_entry >>= 3;
            scratch[row_offset + 1] = ((uvlc_entry & 0x7) + (tmp & !(0xFF_u32 << len))) as u16;
            scratch[row_offset + 3] = ((uvlc_entry >> 3) + (tmp >> len)) as u16;

            row_offset += 4;
        }

        scratch[row_offset] = 0;
        scratch[row_offset + 1] = 0;
    }

    Some(())
}

#[inline(always)]
fn build_sigma_from_cleanup_phase(
    cleanup: &[u16],
    sigma: &mut [u16],
    width: u32,
    height: u32,
    sstr: usize,
    mstr: usize,
) -> Option<()> {
    let sigma_rows = height.div_ceil(4) as usize + 1;
    if sigma.len() < sigma_rows * mstr {
        return None;
    }

    let mut y = 0u32;
    while y < height {
        let sp_base = (y >> 1) as usize * sstr;
        let dp_base = (y >> 2) as usize * mstr;
        let mut x = 0u32;
        let mut sp = sp_base;
        let mut dp = dp_base;
        while x < width {
            let mut t0 =
                ((u32::from(cleanup[sp]) & 0x30) >> 4) | ((u32::from(cleanup[sp]) & 0xC0) >> 2);
            t0 |= ((u32::from(cleanup[sp + 2]) & 0x30) << 4)
                | ((u32::from(cleanup[sp + 2]) & 0xC0) << 6);
            let mut t1 = ((u32::from(cleanup[sp + sstr]) & 0x30) >> 2)
                | (u32::from(cleanup[sp + sstr]) & 0xC0);
            t1 |= ((u32::from(cleanup[sp + sstr + 2]) & 0x30) << 6)
                | ((u32::from(cleanup[sp + sstr + 2]) & 0xC0) << 8);
            sigma[dp] = (t0 | t1) as u16;
            x += 4;
            sp += 4;
            dp += 1;
        }
        sigma[dp] = 0;
        y += 4;
    }

    let dp_base = (height.div_ceil(4) as usize) * mstr;
    for x in 0..=width.div_ceil(4) as usize {
        sigma[dp_base + x] = 0;
    }

    Some(())
}

#[inline(always)]
fn apply_significance_propagation_phase(
    refinement_data: &[u8],
    sigma: &[u16],
    decoded_data: &mut [u32],
    width: u32,
    height: u32,
    stride: u32,
    mstr: usize,
    stripe_causal: bool,
    p: u32,
    prev_row_sig: &mut [u16],
) -> Option<()> {
    if prev_row_sig.len() < width.div_ceil(4) as usize + 8 {
        return None;
    }

    prev_row_sig.fill(0);
    let mut sigprop = ForwardBitReader::<0>::new(refinement_data);
    let stride_us = stride as usize;

    for y in (0..height).step_by(4) {
        let mut pattern = 0xFFFFu32;
        if height - y < 4 {
            pattern = 0x7777;
            if height - y < 3 {
                pattern = 0x3333;
                if height - y < 2 {
                    pattern = 0x1111;
                }
            }
        }

        let mut prev = 0u32;
        let cur_row = (y >> 2) as usize * mstr;
        let next_row = cur_row + mstr;
        let dpp = (y * stride) as usize;

        for x in (0..width).step_by(4) {
            let mut col_pattern = pattern;
            let mut s = x as i32 + 4 - width as i32;
            s = s.max(0);
            col_pattern >>= (s * 4) as u32;

            let idx = (x >> 2) as usize;
            let ps = u32::from(prev_row_sig[idx]) | (u32::from(prev_row_sig[idx + 1]) << 16);
            let ns = read_u32_pair(sigma, next_row + idx);
            let mut u = (ps & 0x8888_8888) >> 3;
            if !stripe_causal {
                u |= (ns & 0x1111_1111) << 3;
            }

            let cs = read_u32_pair(sigma, cur_row + idx);
            let mut mbr = cs;
            mbr |= (cs & 0x7777_7777) << 1;
            mbr |= (cs & 0xEEEE_EEEE) >> 1;
            mbr |= u;
            let t = mbr;
            mbr |= t << 4;
            mbr |= t >> 4;
            mbr |= prev >> 12;
            mbr &= col_pattern;
            mbr &= !cs;

            let mut new_sig = 0u32;
            if mbr != 0 {
                let mut cwd = sigprop.fetch();
                let mut cnt = 0u32;
                let inv_sig = !cs & col_pattern;
                let mut candidates = mbr;
                let mut processed = 0u32;

                while candidates != 0 {
                    let bit = candidates.trailing_zeros();
                    let sample_mask = 1u32 << bit;
                    candidates &= !sample_mask;
                    processed |= sample_mask;

                    if (cwd & 1) != 0 {
                        new_sig |= sample_mask;
                        candidates |= SIGPROP_SPREAD_MASKS[bit as usize] & inv_sig & !processed;
                    }
                    cwd >>= 1;
                    cnt += 1;
                }

                if new_sig != 0 {
                    let value = 3u32 << (p - 2);
                    let block_base = dpp + x as usize;
                    let mut sign_bits = new_sig;

                    while sign_bits != 0 {
                        let bit = sign_bits.trailing_zeros();
                        let sample_mask = 1u32 << bit;
                        sign_bits &= !sample_mask;

                        let offset = (bit >> 2) as usize + ((bit & 3) as usize * stride_us);
                        decoded_data[block_base + offset] = ((cwd & 1) << 31) | value;
                        cwd >>= 1;
                        cnt += 1;
                    }
                }

                sigprop.advance(cnt);
            }

            let combined_sig = new_sig | cs;
            prev_row_sig[idx] = combined_sig as u16;
            if idx + 1 < prev_row_sig.len() {
                prev_row_sig[idx + 1] = (combined_sig >> 16) as u16;
            }

            let t = combined_sig;
            let mut next_prev = combined_sig;
            next_prev |= (t & 0x7777) << 1;
            next_prev |= (t & 0xEEEE) >> 1;
            prev = (next_prev | u) & 0xF000;
        }
    }

    Some(())
}

#[inline(always)]
fn apply_magnitude_refinement_phase(
    refinement_data: &[u8],
    sigma: &[u16],
    decoded_data: &mut [u32],
    width: u32,
    height: u32,
    stride: u32,
    mstr: usize,
    p: u32,
) -> Option<()> {
    if p < 2 {
        return None;
    }

    let mut magref = ReverseBitReader::new_mrp(refinement_data);
    let half = 1u32 << (p - 2);

    for y in (0..height).step_by(4) {
        let mut cur_sig_idx = (y >> 2) as usize * mstr;
        let dpp = (y * stride) as usize;

        for i in (0..width).step_by(8) {
            let cwd = magref.fetch();
            let sig = read_u32_pair(sigma, cur_sig_idx);
            cur_sig_idx += 2;
            let mut col_mask = 0xFu32;
            let mut cwd_mut = cwd;

            if sig != 0 {
                for j in 0..8 {
                    if (sig & col_mask) != 0 {
                        let mut dp = dpp + i as usize + j;
                        let mut sample_mask = 0x1111_1111u32 & col_mask;

                        for _ in 0..4 {
                            if (sig & sample_mask) != 0 {
                                let mut sym = cwd_mut & 1;
                                sym = (1 - sym) << (p - 1);
                                sym |= half;
                                decoded_data[dp] ^= sym;
                                cwd_mut >>= 1;
                            }
                            sample_mask <<= 1;
                            dp += stride as usize;
                        }
                    }
                    col_mask <<= 4;
                }
            }

            magref.advance(sig.count_ones());
        }
    }

    Some(())
}

#[inline(never)]
fn decode_magnitude_sign_phase(
    coded_data: &[u8],
    lcup: usize,
    scup: usize,
    scratch: &[u16],
    decoded_data: &mut [u32],
    missing_msbs: u32,
    width: u32,
    height: u32,
    stride: u32,
    sstr: usize,
    v_n_scratch: &mut [u32],
) -> Option<()> {
    let v_n_width = width.div_ceil(2) as usize + 2;
    if v_n_scratch.len() < v_n_width {
        return None;
    }
    v_n_scratch[..v_n_width].fill(0);

    let p = 30 - missing_msbs;
    let mmsbp2 = missing_msbs + 2;
    let mut magsgn = ForwardBitReader::<0xFF>::new(&coded_data[..lcup - scup]);
    let mut prev_v_n = 0u32;
    let mut x = 0u32;
    let mut sp = 0usize;
    let mut vp = 0usize;
    let mut dp = 0usize;
    let second_row_present = height > 1;

    while x < width {
        let inf = u32::from(scratch[sp]);
        let uq = u32::from(scratch[sp + 1]);
        if uq > mmsbp2 {
            return None;
        }

        let (val0, _) = decode_mag_sgn_sample_with_vn(&mut magsgn, inf, 0, uq, p);
        decoded_data[dp] = val0;

        let (val1, v_n1) = decode_mag_sgn_sample_with_vn(&mut magsgn, inf, 1, uq, p);
        if second_row_present {
            decoded_data[dp + stride as usize] = val1;
        }
        v_n_scratch[vp] = prev_v_n | v_n1;
        prev_v_n = 0;
        dp += 1;
        x += 1;

        if x >= width {
            vp += 1;
            break;
        }

        let (val2, _) = decode_mag_sgn_sample_with_vn(&mut magsgn, inf, 2, uq, p);
        decoded_data[dp] = val2;

        let (val3, v_n3) = decode_mag_sgn_sample_with_vn(&mut magsgn, inf, 3, uq, p);
        if second_row_present {
            decoded_data[dp + stride as usize] = val3;
        }
        prev_v_n = v_n3;
        dp += 1;
        x += 1;
        sp += 2;
        vp += 1;
    }
    v_n_scratch[vp] = prev_v_n;

    for y in (2..height).step_by(2) {
        let row_base = (y >> 1) as usize * sstr;
        let mut sp = row_base;
        let mut vp = 0usize;
        let mut dp = (y * stride) as usize;
        let mut prev_v_n = 0u32;
        let mut x = 0u32;
        let second_row_present = y + 1 < height;

        while x < width {
            let inf = u32::from(scratch[sp]);
            let u_q = u32::from(scratch[sp + 1]);
            let mut gamma = inf & 0xF0;
            gamma &= gamma.wrapping_sub(0x10);
            let mut emax = v_n_scratch[vp] | v_n_scratch[vp + 1];
            emax = 31 - (emax | 2).leading_zeros();
            let kappa = if gamma != 0 { emax } else { 1 };
            let uq = u_q + kappa;
            if uq > mmsbp2 {
                return None;
            }

            let (val0, _) = decode_mag_sgn_sample_with_vn(&mut magsgn, inf, 0, uq, p);
            decoded_data[dp] = val0;

            let (val1, v_n1) = decode_mag_sgn_sample_with_vn(&mut magsgn, inf, 1, uq, p);
            if second_row_present {
                decoded_data[dp + stride as usize] = val1;
            }
            v_n_scratch[vp] = prev_v_n | v_n1;
            prev_v_n = 0;
            dp += 1;
            x += 1;

            if x >= width {
                vp += 1;
                break;
            }

            let (val2, _) = decode_mag_sgn_sample_with_vn(&mut magsgn, inf, 2, uq, p);
            decoded_data[dp] = val2;

            let (val3, v_n3) = decode_mag_sgn_sample_with_vn(&mut magsgn, inf, 3, uq, p);
            if second_row_present {
                decoded_data[dp + stride as usize] = val3;
            }
            prev_v_n = v_n3;
            dp += 1;
            x += 1;
            sp += 2;
            vp += 1;
        }

        v_n_scratch[vp] = prev_v_n;
    }

    Some(())
}

#[inline(always)]
fn decode_impl<const PHASE_LIMIT: u8, O: HtDecodeObserver>(
    cleanup_data: &[u8],
    refinement_data: &[u8],
    decoded_data: &mut [u32],
    missing_msbs: u32,
    mut num_passes: u32,
    width: u32,
    height: u32,
    stride: u32,
    stripe_causal: bool,
    scratch_buffers: &mut HtBlockDecodeScratch,
    observer: &mut O,
) -> Option<()> {
    observer.record_block(cleanup_data.len(), refinement_data.len());

    if num_passes > 1 && refinement_data.is_empty() {
        num_passes = 1;
    }

    if num_passes > 3 || missing_msbs > 30 {
        return None;
    }

    if missing_msbs == 30 {
        return None;
    }

    if missing_msbs == 29 && num_passes > 1 {
        num_passes = 1;
    }

    let p = 30 - missing_msbs;
    let lcup = cleanup_data.len();

    if lcup < 2 {
        return None;
    }

    let scup = cleanup_segment_suffix_length(cleanup_data, lcup)?;

    let quad_rows = height.div_ceil(2) as usize;
    let sstr = cleanup_symbol_stride(width);
    let scratch = zeroed_u16_scratch(&mut scratch_buffers.cleanup, sstr * (quad_rows + 1));
    let phase_start = observer.phase_start();
    decode_cleanup_symbols(cleanup_data, lcup, scup, width, height, sstr, scratch)?;
    observer.add_cleanup_us(phase_start);

    let v_n_width = width.div_ceil(2) as usize + 2;
    let v_n_scratch = resized_u32_scratch(&mut scratch_buffers.v_n, v_n_width);
    let phase_start = observer.phase_start();
    decode_magnitude_sign_phase(
        cleanup_data,
        lcup,
        scup,
        scratch,
        decoded_data,
        missing_msbs,
        width,
        height,
        stride,
        sstr,
        v_n_scratch,
    )?;
    observer.add_mag_sgn_us(phase_start);

    if PHASE_LIMIT == PHASE_LIMIT_CLEANUP {
        return Some(());
    }

    if num_passes > 1 {
        let sigma_rows = height.div_ceil(4) as usize + 1;
        let mstr = sigma_stride(width);
        let sigma = zeroed_u16_scratch(&mut scratch_buffers.sigma, sigma_rows * mstr);
        let phase_start = observer.phase_start();
        build_sigma_from_cleanup_phase(scratch, sigma, width, height, sstr, mstr)?;
        observer.add_sigma_us(phase_start);

        let prev_row_sig = resized_u16_scratch(
            &mut scratch_buffers.prev_row_sig,
            width.div_ceil(4) as usize + 8,
        );
        let phase_start = observer.phase_start();
        apply_significance_propagation_phase(
            refinement_data,
            sigma,
            decoded_data,
            width,
            height,
            stride,
            mstr,
            stripe_causal,
            p,
            prev_row_sig,
        )?;
        observer.add_sigprop_us(phase_start);

        if PHASE_LIMIT == PHASE_LIMIT_SIGPROP {
            return Some(());
        }

        if num_passes > 2 {
            let phase_start = observer.phase_start();
            apply_magnitude_refinement_phase(
                refinement_data,
                sigma,
                decoded_data,
                width,
                height,
                stride,
                mstr,
                p,
            )?;
            observer.add_magref_us(phase_start);
        }
    }

    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::j2c::ht_block_encode::encode_code_block;

    #[test]
    fn test_coefficient_to_i32_shifted_alignment() {
        let aligned = 3u32 << (31 - 5);
        assert_eq!(coefficient_to_i32(aligned, 5), 3);
        assert_eq!(coefficient_to_i32(0x8000_0000 | aligned, 5), -3);
    }

    #[test]
    fn test_direct_ht_block_roundtrip_varied_4x4() {
        let original: Vec<i32> = (0..16).map(|i| (i * 3) - 20).collect();
        let total_bitplanes = 6u8;
        let encoded = encode_code_block(&original, 4, 4, total_bitplanes).expect("encode HT block");
        assert_eq!(encoded.num_coding_passes, 1);

        let mut decoded = vec![0u32; original.len()];
        let mut scratch = HtBlockDecodeScratch::default();
        let mut observer = NoHtDecodeStats;
        let decoded_ok = decode_impl::<PHASE_LIMIT_MAGREF, _>(
            &encoded.data,
            &[],
            &mut decoded,
            u32::from(encoded.num_zero_bitplanes),
            u32::from(encoded.num_coding_passes),
            4,
            4,
            4,
            false,
            &mut scratch,
            &mut observer,
        );
        assert!(decoded_ok.is_some(), "encoded={:02x?}", encoded.data);

        let decoded_i32: Vec<i32> = decoded
            .into_iter()
            .map(|value| coefficient_to_i32(value, total_bitplanes))
            .collect();
        assert_eq!(decoded_i32, original, "encoded={:02x?}", encoded.data);
    }

    #[test]
    fn test_direct_ht_block_roundtrip_positive_varied_4x4() {
        let original: Vec<i32> = (0..16).map(|i| i * 3).collect();
        let total_bitplanes = 6u8;
        let encoded = encode_code_block(&original, 4, 4, total_bitplanes).expect("encode HT block");
        assert_eq!(encoded.num_coding_passes, 1);

        let mut decoded = vec![0u32; original.len()];
        let mut scratch = HtBlockDecodeScratch::default();
        let mut observer = NoHtDecodeStats;
        let decoded_ok = decode_impl::<PHASE_LIMIT_MAGREF, _>(
            &encoded.data,
            &[],
            &mut decoded,
            u32::from(encoded.num_zero_bitplanes),
            u32::from(encoded.num_coding_passes),
            4,
            4,
            4,
            false,
            &mut scratch,
            &mut observer,
        );
        assert!(decoded_ok.is_some(), "encoded={:02x?}", encoded.data);

        let decoded_i32: Vec<i32> = decoded
            .into_iter()
            .map(|value| coefficient_to_i32(value, total_bitplanes))
            .collect();
        assert_eq!(decoded_i32, original, "encoded={:02x?}", encoded.data);
    }

    #[test]
    fn cleanup_and_magnitude_sign_phases_decode_odd_sized_block() {
        let width = 15u32;
        let height = 13u32;
        let original: Vec<i32> = (0..(width * height))
            .map(|i| {
                let value = (i as i32 % 61) - 30;
                if i % 7 == 0 {
                    0
                } else {
                    value
                }
            })
            .collect();
        let total_bitplanes = 6u8;
        let encoded =
            encode_code_block(&original, width, height, total_bitplanes).expect("encode HT block");
        assert_eq!(encoded.num_coding_passes, 1);

        let lcup = encoded.data.len();
        let scup = cleanup_segment_suffix_length(&encoded.data, lcup).expect("valid cleanup info");
        let sstr = cleanup_symbol_stride(width);
        let quad_rows = height.div_ceil(2) as usize;
        let mut cleanup = vec![0u16; sstr * (quad_rows + 1)];
        decode_cleanup_symbols(&encoded.data, lcup, scup, width, height, sstr, &mut cleanup)
            .expect("decode cleanup symbols");

        let mut decoded = vec![0u32; original.len()];
        let mut v_n_scratch = vec![0u32; width.div_ceil(2) as usize + 2];
        decode_magnitude_sign_phase(
            &encoded.data,
            lcup,
            scup,
            &cleanup,
            &mut decoded,
            u32::from(encoded.num_zero_bitplanes),
            width,
            height,
            width,
            sstr,
            &mut v_n_scratch,
        )
        .expect("decode magnitude/sign phase");

        let decoded_i32: Vec<i32> = decoded
            .into_iter()
            .map(|value| coefficient_to_i32(value, total_bitplanes))
            .collect();
        assert_eq!(decoded_i32, original, "encoded={:02x?}", encoded.data);
    }

    #[test]
    fn sigma_phase_builds_masks_and_zeroes_edge_sentinels() {
        let width = 7u32;
        let height = 5u32;
        let sstr = cleanup_symbol_stride(width);
        let mstr = sigma_stride(width);
        let sigma_rows = height.div_ceil(4) as usize + 1;
        let mut cleanup = vec![0u16; sstr * (height.div_ceil(2) as usize + 1)];
        cleanup[0] = 0x30;
        cleanup[2] = 0xC0;
        cleanup[sstr] = 0xF0;
        cleanup[sstr + 2] = 0x30;
        cleanup[2 * sstr] = 0xC0;
        cleanup[2 * sstr + 2] = 0xF0;
        let mut sigma = vec![0u16; sigma_rows * mstr];

        build_sigma_from_cleanup_phase(&cleanup, &mut sigma, width, height, sstr, mstr)
            .expect("build sigma");

        let expected_first = (((u32::from(cleanup[0]) & 0x30) >> 4)
            | ((u32::from(cleanup[0]) & 0xC0) >> 2)
            | ((u32::from(cleanup[2]) & 0x30) << 4)
            | ((u32::from(cleanup[2]) & 0xC0) << 6)
            | ((u32::from(cleanup[sstr]) & 0x30) >> 2)
            | (u32::from(cleanup[sstr]) & 0xC0)
            | ((u32::from(cleanup[sstr + 2]) & 0x30) << 6)
            | ((u32::from(cleanup[sstr + 2]) & 0xC0) << 8)) as u16;
        let expected_second = (((u32::from(cleanup[4]) & 0x30) >> 4)
            | ((u32::from(cleanup[4]) & 0xC0) >> 2)
            | ((u32::from(cleanup[6]) & 0x30) << 4)
            | ((u32::from(cleanup[6]) & 0xC0) << 6)
            | ((u32::from(cleanup[sstr + 4]) & 0x30) >> 2)
            | (u32::from(cleanup[sstr + 4]) & 0xC0)
            | ((u32::from(cleanup[sstr + 6]) & 0x30) << 6)
            | ((u32::from(cleanup[sstr + 6]) & 0xC0) << 8)) as u16;
        assert_eq!(sigma[0], expected_first);
        assert_eq!(sigma[1], expected_second);
        assert_eq!(sigma[2], 0);

        let bottom = height.div_ceil(4) as usize * mstr;
        for x in 0..=width.div_ceil(4) as usize {
            assert_eq!(sigma[bottom + x], 0);
        }
    }

    #[test]
    fn refinement_phases_leave_output_unchanged_for_empty_sigma() {
        let width = 7u32;
        let height = 5u32;
        let stride = width;
        let mstr = sigma_stride(width);
        let sigma = vec![0u16; (height.div_ceil(4) as usize + 1) * mstr];
        let mut prev_row_sig = vec![0u16; width.div_ceil(4) as usize + 8];
        let mut decoded = vec![0x1234_5678u32; (stride * height) as usize];
        let expected = decoded.clone();

        apply_significance_propagation_phase(
            &[],
            &sigma,
            &mut decoded,
            width,
            height,
            stride,
            mstr,
            false,
            5,
            &mut prev_row_sig,
        )
        .expect("empty sigma sigprop");
        apply_magnitude_refinement_phase(&[], &sigma, &mut decoded, width, height, stride, mstr, 5)
            .expect("empty sigma magref");

        assert_eq!(decoded, expected);
    }

    #[test]
    fn sigprop_spread_masks_follow_column_major_scan_order() {
        let row_patterns = [0x33u32, 0x76, 0xEC, 0xC8];

        for bit in 0..16 {
            let expected = row_patterns[bit & 3] << (bit & !3);
            assert_eq!(SIGPROP_SPREAD_MASKS[bit], expected, "bit={bit}");
            assert_eq!(SIGPROP_SPREAD_MASKS[bit] & ((1u32 << bit) - 1), 0);
        }
    }

    #[test]
    fn combined_data_exposes_borrowed_segment_slices() {
        let combined = CombinedCodeBlockData {
            data: vec![0x11, 0x22, 0x33, 0x44, 0x55],
            cleanup_length: 3,
            refinement_length: 2,
        };

        let segments = combined.segments().expect("split combined data");

        assert_eq!(segments.cleanup, &[0x11, 0x22, 0x33]);
        assert_eq!(segments.refinement, &[0x44, 0x55]);
    }

    #[test]
    fn borrowed_segments_decode_matches_owned_combined_decode() {
        let width = 16u32;
        let height = 16u32;
        let original: Vec<i32> = (0..(width * height))
            .map(|i| {
                let value = (i as i32 % 47) - 23;
                if i % 5 == 0 {
                    0
                } else {
                    value
                }
            })
            .collect();
        let total_bitplanes = 6u8;
        let encoded =
            encode_code_block(&original, width, height, total_bitplanes).expect("encode HT block");
        let combined = CombinedCodeBlockData {
            data: encoded.data.clone(),
            cleanup_length: encoded.data.len() as u32,
            refinement_length: 0,
        };
        let segments = HtCodeBlockSegments {
            cleanup: &encoded.data,
            refinement: &[],
        };
        let mut owned_decoded = vec![0u32; original.len()];
        let mut borrowed_decoded = vec![0u32; original.len()];
        let mut scratch = HtBlockDecodeScratch::default();

        decode_combined_validated(
            &combined,
            encoded.num_zero_bitplanes,
            total_bitplanes,
            encoded.num_coding_passes,
            false,
            true,
            &mut owned_decoded,
            width,
            height,
            width,
        )
        .expect("decode owned combined payload");
        decode_segments_validated_with_scratch(
            &segments,
            encoded.num_zero_bitplanes,
            total_bitplanes,
            encoded.num_coding_passes,
            false,
            true,
            &mut borrowed_decoded,
            width,
            height,
            width,
            &mut scratch,
        )
        .expect("decode borrowed payload segments");

        assert_eq!(borrowed_decoded, owned_decoded);
    }

    #[test]
    fn scratch_resize_zeroes_existing_values_when_growing() {
        let mut scratch = HtBlockDecodeScratch::default();

        zeroed_u16_scratch(&mut scratch.cleanup, 4).fill(7);
        assert_eq!(zeroed_u16_scratch(&mut scratch.cleanup, 8), &[0; 8]);

        zeroed_u32_scratch(&mut scratch.v_n, 4).fill(9);
        assert_eq!(zeroed_u32_scratch(&mut scratch.v_n, 8), &[0; 8]);
    }

    #[test]
    fn decode_combined_validated_with_scratch_reuses_zeroed_buffers() {
        let width = 16u32;
        let height = 16u32;
        let original: Vec<i32> = (0..(width * height))
            .map(|i| {
                let value = (i as i32 % 47) - 23;
                if i % 5 == 0 {
                    0
                } else {
                    value
                }
            })
            .collect();
        let total_bitplanes = 6u8;
        let encoded =
            encode_code_block(&original, width, height, total_bitplanes).expect("encode HT block");
        let combined = CombinedCodeBlockData {
            data: encoded.data.clone(),
            cleanup_length: encoded.data.len() as u32,
            refinement_length: 0,
        };
        let mut scratch = HtBlockDecodeScratch::default();
        let mut decoded = vec![0u32; original.len()];

        decode_combined_validated_with_scratch(
            &combined,
            encoded.num_zero_bitplanes,
            total_bitplanes,
            encoded.num_coding_passes,
            false,
            true,
            &mut decoded,
            width,
            height,
            width,
            &mut scratch,
        )
        .expect("decode HT block");

        let first_capacities = scratch.capacities_for_test();
        assert!(first_capacities.cleanup > 0);
        assert!(first_capacities.v_n > 0);

        scratch.poison_for_test();
        decoded.fill(0);

        decode_combined_validated_with_scratch(
            &combined,
            encoded.num_zero_bitplanes,
            total_bitplanes,
            encoded.num_coding_passes,
            false,
            true,
            &mut decoded,
            width,
            height,
            width,
            &mut scratch,
        )
        .expect("decode HT block after scratch poison");

        assert_eq!(scratch.capacities_for_test(), first_capacities);
        let decoded_i32: Vec<i32> = decoded
            .into_iter()
            .map(|value| coefficient_to_i32(value, total_bitplanes))
            .collect();
        assert_eq!(decoded_i32, original, "encoded={:02x?}", encoded.data);
    }
}
