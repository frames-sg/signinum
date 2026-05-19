//! Bitplane decoding, described in Annex D.
//!
//! JPEG2000 groups the samples of each component into their constituent
//! bit planes and uses a special context-modeling approach to encode the
//! bits using the arithmetic encoder. In this stage, we need to "revert" the
//! context-modeling so that we can extract the magnitudes and signs of each
//! sample.
//!
//! Some of the references are taken from the
//! "JPEG2000 Standard for Image Compression" book instead of the specification.

use alloc::vec;
use alloc::vec::Vec;

use super::arithmetic_decoder::{ArithmeticDecoder, ArithmeticDecoderContext};
use super::build::{CodeBlock, SubBandType};
use super::codestream::CodeBlockStyle;
use super::decode::{DecompositionStorage, TileDecodeContext};
use crate::error::{bail, DecodingError, Result};
use crate::reader::BitReader;
use crate::J2kCodeBlockSegment;

/// Decode the layers of the given code block into coefficients.
///
/// The result will be stored in the form of a vector of signs and magnitudes
/// in the bitplane decoder context.
pub(crate) fn decode(
    code_block: &CodeBlock,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
    style: &CodeBlockStyle,
    tile_ctx: &mut TileDecodeContext,
    storage: &DecompositionStorage<'_>,
    strict: bool,
) -> Result<()> {
    tile_ctx.bit_plane_decode_context.reset(
        code_block,
        sub_band_type,
        style,
        total_bitplanes,
        strict,
    )?;
    tile_ctx.bit_plane_decode_buffers.reset();

    decode_inner(
        code_block,
        storage,
        &mut tile_ctx.bit_plane_decode_context,
        &mut tile_ctx.bit_plane_decode_buffers,
    )
    .ok_or(DecodingError::CodeBlockDecodeFailure)?;

    Ok(())
}

pub(crate) fn decode_code_block_segments_validated(
    data: &[u8],
    segments: &[J2kCodeBlockSegment],
    width: u32,
    height: u32,
    missing_bit_planes: u8,
    number_of_coding_passes: u8,
    total_bitplanes: u8,
    sub_band_type: SubBandType,
    code_block_style: &CodeBlockStyle,
    strict: bool,
    ctx: &mut BitPlaneDecodeContext,
) -> Result<()> {
    ctx.reset_for_job(
        width,
        height,
        missing_bit_planes,
        number_of_coding_passes,
        sub_band_type,
        code_block_style,
        total_bitplanes,
        strict,
    )?;

    if number_of_coding_passes == 0 || ctx.bitplanes == 0 {
        return Ok(());
    }

    decode_code_block_segments_inner(data, segments, number_of_coding_passes, ctx)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;

    Ok(())
}

fn decode_inner(
    code_block: &CodeBlock,
    storage: &DecompositionStorage<'_>,
    ctx: &mut BitPlaneDecodeContext,
    bp_buffers: &mut BitPlaneDecodeBuffers,
) -> Option<()> {
    bp_buffers.reset();

    let mut last_segment_idx = 0;
    let mut coding_passes = 0;

    // Build a list so that we can associate coding passes with their segments
    // and data more easily.
    for layer in &storage.layers[code_block.layers.start..code_block.layers.end] {
        if let Some(range) = layer.segments.clone() {
            let layer_segments = &storage.segments[range.clone()];
            for segment in layer_segments {
                if segment.idx != last_segment_idx {
                    assert_eq!(segment.idx, last_segment_idx + 1);

                    bp_buffers
                        .segment_ranges
                        .push(bp_buffers.combined_layers.len());
                    bp_buffers.segment_coding_passes.push(coding_passes);
                    last_segment_idx += 1;
                }

                bp_buffers.combined_layers.extend(segment.data);
                coding_passes += segment.coding_pases;
            }
        }
    }

    assert_eq!(coding_passes, code_block.number_of_coding_passes);

    bp_buffers
        .segment_ranges
        .push(bp_buffers.combined_layers.len());
    bp_buffers.segment_coding_passes.push(coding_passes);

    let is_normal_mode =
        !ctx.style.selective_arithmetic_coding_bypass && !ctx.style.termination_on_each_pass;

    if is_normal_mode {
        // Only one termination per code block, so we can just decode the
        // whole range in one single go, processing all coding passes at once.
        let mut decoder = ArithmeticDecoder::new(&bp_buffers.combined_layers);
        let end = code_block
            .number_of_coding_passes
            .min(ctx.max_coding_passes);
        if ctx.uses_normal_arithmetic_neighbor_path() {
            handle_normal_arithmetic_coding_passes(0, end, ctx, &mut decoder)?;
        } else {
            handle_arithmetic_coding_passes(0, end, ctx, &mut decoder)?;
        }
    } else {
        // Otherwise, each segment introduces a termination. For "termination on
        // each pass", each segment only covers one coding pass
        // and a termination is introduced every time. Otherwise, for only
        // arithmetic coding bypass, terminations are introduced based on the
        // exact index of the covered coding passes (see Table D.9).
        for segment in 0..bp_buffers.segment_coding_passes.len() - 1 {
            let start_coding_pass = bp_buffers.segment_coding_passes[segment];
            let end_coding_pass =
                bp_buffers.segment_coding_passes[segment + 1].min(ctx.max_coding_passes);

            let data = &bp_buffers.combined_layers
                [bp_buffers.segment_ranges[segment]..bp_buffers.segment_ranges[segment + 1]];

            let use_arithmetic = if ctx.style.selective_arithmetic_coding_bypass {
                if start_coding_pass <= 9 {
                    true
                } else {
                    // Only for cleanup pass.
                    start_coding_pass.is_multiple_of(3)
                }
            } else {
                true
            };

            if use_arithmetic {
                let mut decoder = ArithmeticDecoder::new(data);
                handle_arithmetic_coding_passes(
                    start_coding_pass,
                    end_coding_pass,
                    ctx,
                    &mut decoder,
                )?;
            } else {
                let mut decoder = BypassDecoder::new(data, ctx.strict);
                handle_bypass_coding_passes(start_coding_pass, end_coding_pass, ctx, &mut decoder)?;
            }
        }
    }

    Some(())
}

fn handle_arithmetic_coding_passes(
    start: u8,
    end: u8,
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
) -> Option<()> {
    handle_arithmetic_coding_passes_with_neighbors::<false>(start, end, ctx, decoder)
}

fn handle_normal_arithmetic_coding_passes(
    start: u8,
    end: u8,
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
) -> Option<()> {
    handle_arithmetic_coding_passes_with_neighbors::<true>(start, end, ctx, decoder)
}

fn handle_arithmetic_coding_passes_with_neighbors<const NORMAL_NEIGHBORS: bool>(
    start: u8,
    end: u8,
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
) -> Option<()> {
    for coding_pass in start..end {
        let current_bitplane = coding_pass.div_ceil(3);
        ctx.current_bit_position = ctx.bitplanes - 1 - current_bitplane;

        // The first bitplane only has a cleanup pass, all other bitplanes
        // are in the order SPP -> MRR -> C.
        match coding_pass % 3 {
            0 => {
                cleanup_pass_arithmetic_with_neighbors::<NORMAL_NEIGHBORS>(ctx, decoder);

                if ctx.style.segmentation_symbols {
                    let b0 = decoder.read_bit(ctx.arithmetic_decoder_context(18));
                    let b1 = decoder.read_bit(ctx.arithmetic_decoder_context(18));
                    let b2 = decoder.read_bit(ctx.arithmetic_decoder_context(18));
                    let b3 = decoder.read_bit(ctx.arithmetic_decoder_context(18));

                    if (b0 != 1 || b1 != 0 || b2 != 1 || b3 != 0) && ctx.strict {
                        return None;
                    }
                }

                ctx.reset_for_next_bitplane();
            }
            1 => {
                significance_propagation_pass_arithmetic_with_neighbors::<NORMAL_NEIGHBORS>(
                    ctx, decoder,
                );
            }
            2 => {
                magnitude_refinement_pass_arithmetic_with_neighbors::<NORMAL_NEIGHBORS>(
                    ctx, decoder,
                );
            }
            _ => unreachable!(),
        }

        if ctx.style.reset_context_probabilities {
            ctx.reset_contexts();
        }
    }

    Some(())
}

fn handle_bypass_coding_passes(
    start: u8,
    end: u8,
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut BypassDecoder<'_>,
) -> Option<()> {
    for coding_pass in start..end {
        let current_bitplane = coding_pass.div_ceil(3);
        ctx.current_bit_position = ctx.bitplanes - 1 - current_bitplane;

        match coding_pass % 3 {
            0 => {
                SafeScalarTier1::cleanup_pass_bypass(ctx, decoder)?;

                if ctx.style.segmentation_symbols {
                    let b0 = decoder.read_bit(ctx.arithmetic_decoder_context(18))?;
                    let b1 = decoder.read_bit(ctx.arithmetic_decoder_context(18))?;
                    let b2 = decoder.read_bit(ctx.arithmetic_decoder_context(18))?;
                    let b3 = decoder.read_bit(ctx.arithmetic_decoder_context(18))?;

                    if (b0 != 1 || b1 != 0 || b2 != 1 || b3 != 0) && ctx.strict {
                        return None;
                    }
                }

                ctx.reset_for_next_bitplane();
            }
            1 => {
                SafeScalarTier1::significance_propagation_pass_bypass(ctx, decoder)?;
            }
            2 => {
                SafeScalarTier1::magnitude_refinement_pass_bypass(ctx, decoder)?;
            }
            _ => unreachable!(),
        }

        if ctx.style.reset_context_probabilities {
            ctx.reset_contexts();
        }
    }

    Some(())
}

// We only allow 31 bit planes because we need one bit for the sign.
pub(crate) const BITPLANE_BIT_SIZE: u32 = size_of::<u32>() as u32 * 8 - 1;

const SIGNIFICANCE_SHIFT: u8 = 7;
const HAS_MAGNITUDE_REFINEMENT_SHIFT: u8 = 6;
const HAS_ZERO_CODING_SHIFT: u8 = 5;
const SIGNIFICANCE_MASK: u8 = 1 << SIGNIFICANCE_SHIFT;
const HAS_MAGNITUDE_REFINEMENT_MASK: u8 = 1 << HAS_MAGNITUDE_REFINEMENT_SHIFT;
const HAS_ZERO_CODING_MASK: u8 = 1 << HAS_ZERO_CODING_SHIFT;

/// Bit-packed coefficient state (only 3 bits used):
/// - Bit 7: significance state (set when first non-zero bit is encountered)
/// - Bit 6: has had magnitude refinement pass
/// - Bit 5: zero coded in current bitplane's significance propagation pass
#[derive(Default, Copy, Clone)]
pub(crate) struct CoefficientState(u8);

impl CoefficientState {
    #[inline(always)]
    fn set_bit(&mut self, shift: u8, value: u8) {
        debug_assert!(value < 2);

        self.0 &= !(1_u8 << shift);
        self.0 |= value << shift;
    }

    #[inline(always)]
    fn set_significant(&mut self) {
        self.set_bit(SIGNIFICANCE_SHIFT, 1);
    }

    #[inline(always)]
    fn is_significant(&self) -> bool {
        self.significance() == 1
    }

    #[inline(always)]
    fn significance(&self) -> u8 {
        (self.0 >> SIGNIFICANCE_SHIFT) & 1
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct Coefficient(u32);

impl Coefficient {
    pub(crate) fn get(&self) -> i32 {
        let mut magnitude = (self.0 & !0x80000000) as i32;
        // Map sign (0 for positive, 1 for negative) to 1, -1.
        magnitude *= 1 - 2 * (self.sign() as i32);

        magnitude
    }

    fn set_sign(&mut self, sign: u8) {
        self.0 |= (sign as u32) << 31;
    }

    fn sign(&self) -> u32 {
        (self.0 >> 31) & 1
    }

    fn push_bit_at(&mut self, bit: u32, position: u8) {
        self.0 |= bit << position;
    }
}

const COEFFICIENTS_PADDING: u32 = 1;

/// Store the significances of each neighbor for a specific coefficient.
/// The order from MSB to LSB is as follows:
///
/// top-left, top, top-right, left, bottom-left, right, bottom-right, bottom.
///
/// See the `context_label_sign_coding` method for why we aren't simply using
/// row-major order.
#[derive(Default, Copy, Clone)]
struct NeighborSignificances(u8);

impl NeighborSignificances {
    fn set_top_left(&mut self) {
        self.0 |= 1 << 7;
    }

    fn set_top(&mut self) {
        self.0 |= 1 << 6;
    }

    fn set_top_right(&mut self) {
        self.0 |= 1 << 5;
    }

    fn set_left(&mut self) {
        self.0 |= 1 << 4;
    }

    fn set_bottom_left(&mut self) {
        self.0 |= 1 << 3;
    }

    fn set_right(&mut self) {
        self.0 |= 1 << 2;
    }

    fn set_bottom_right(&mut self) {
        self.0 |= 1 << 1;
    }

    fn set_bottom(&mut self) {
        self.0 |= 1;
    }

    fn all(&self) -> u8 {
        self.0
    }

    // Needed for vertically causal context.
    fn all_without_bottom(&self) -> u8 {
        self.0 & 0b11110100
    }
}

#[derive(Default)]
pub(crate) struct BitPlaneDecodeBuffers {
    combined_layers: Vec<u8>,
    segment_ranges: Vec<usize>,
    segment_coding_passes: Vec<u8>,
}

impl BitPlaneDecodeBuffers {
    fn reset(&mut self) {
        self.combined_layers.clear();
        self.segment_ranges.clear();
        self.segment_coding_passes.clear();

        // The design of these two buffers is that the ranges are stored
        // as [idx, idx + 1), so we need to store the first 0 when resetting.
        self.segment_ranges.push(0);
        self.segment_coding_passes.push(0);
    }
}

pub(crate) struct BitPlaneDecodeContext {
    /// A vector of bit-packed fields for each coefficient in the code-block.
    coefficient_states: Vec<CoefficientState>,
    /// The neighbor significances for each coefficient.
    neighbor_significances: Vec<NeighborSignificances>,
    /// The magnitude and signs of each coefficient that is successively built
    /// as we advance through the bitplanes.
    coefficients: Vec<Coefficient>,
    /// The width of the code-block we are processing.
    width: u32,
    /// The width of the code-block we are processing, with padding.
    padded_width: u32,
    /// The height of the code-block we are processing.
    height: u32,
    /// The code-block style for the current code-block.
    style: CodeBlockStyle,
    /// The number of bitplanes (minus implicitly missing bitplanes) to decode.
    bitplanes: u8,
    /// Whether strict mode is enabled.
    strict: bool,
    /// The maximum number of coding passes to process.
    max_coding_passes: u8,
    /// The type of sub-band the current code block belongs to.
    sub_band_type: SubBandType,
    /// The arithmetic decoder contexts for each context label.
    contexts: [ArithmeticDecoderContext; 19],
    /// The bit position for the current bitplane.
    current_bit_position: u8,
}

impl Default for BitPlaneDecodeContext {
    fn default() -> Self {
        Self {
            coefficient_states: vec![],
            coefficients: vec![],
            neighbor_significances: vec![],
            width: 0,
            padded_width: COEFFICIENTS_PADDING * 2,
            height: 0,
            style: CodeBlockStyle::default(),
            bitplanes: 0,
            max_coding_passes: 0,
            strict: false,
            sub_band_type: SubBandType::LowLow,
            contexts: [ArithmeticDecoderContext::default(); 19],
            current_bit_position: 0,
        }
    }
}

impl BitPlaneDecodeContext {
    fn reset_for_job(
        &mut self,
        width: u32,
        height: u32,
        missing_bit_planes: u8,
        number_of_coding_passes: u8,
        sub_band_type: SubBandType,
        code_block_style: &CodeBlockStyle,
        total_bitplanes: u8,
        strict: bool,
    ) -> Result<()> {
        let padded_width = width + COEFFICIENTS_PADDING * 2;
        let padded_height = height + COEFFICIENTS_PADDING * 2;
        let num_coefficients = padded_width as usize * padded_height as usize;

        self.coefficients.clear();
        self.coefficients
            .resize(num_coefficients, Coefficient::default());

        self.neighbor_significances.clear();
        self.neighbor_significances
            .resize(num_coefficients, NeighborSignificances::default());

        self.coefficient_states.clear();
        self.coefficient_states
            .resize(num_coefficients, CoefficientState::default());

        self.width = width;
        self.padded_width = padded_width;
        self.height = height;
        self.sub_band_type = sub_band_type;
        self.style = *code_block_style;
        self.reset_contexts();

        self.bitplanes = if strict {
            total_bitplanes
                .checked_sub(missing_bit_planes)
                .ok_or(DecodingError::InvalidBitplaneCount)?
        } else {
            total_bitplanes.saturating_sub(missing_bit_planes)
        };

        self.max_coding_passes = if self.bitplanes == 0 {
            0
        } else {
            1 + 3 * (self.bitplanes - 1)
        };

        if self.max_coding_passes < number_of_coding_passes && strict {
            bail!(DecodingError::TooManyCodingPasses);
        }

        self.strict = strict;

        Ok(())
    }

    /// Completely reset context so that it can be reused for a new code-block.
    pub(crate) fn reset(
        &mut self,
        code_block: &CodeBlock,
        sub_band_type: SubBandType,
        code_block_style: &CodeBlockStyle,
        total_bitplanes: u8,
        strict: bool,
    ) -> Result<()> {
        self.reset_for_job(
            code_block.rect.width(),
            code_block.rect.height(),
            code_block.missing_bit_planes,
            code_block.number_of_coding_passes,
            sub_band_type,
            code_block_style,
            total_bitplanes,
            strict,
        )
    }

    pub(crate) fn coefficient_rows(&self) -> impl Iterator<Item = &[Coefficient]> {
        self.coefficients
            .chunks_exact(self.padded_width as usize)
            // Exclude the padding that we added.
            .map(|row| &row[COEFFICIENTS_PADDING as usize..][..self.width as usize])
            .skip(COEFFICIENTS_PADDING as usize)
            .take(self.height as usize)
    }

    fn arithmetic_decoder_context(&mut self, ctx_label: u8) -> &mut ArithmeticDecoderContext {
        &mut self.contexts[ctx_label as usize]
    }

    /// Reset each context to the initial state defined in table D.7.
    fn reset_contexts(&mut self) {
        for context in &mut self.contexts {
            context.reset();
        }

        self.contexts[0].reset_with_index(4);
        self.contexts[17].reset_with_index(3);
        self.contexts[18].reset_with_index(46);
    }

    /// Reset state that is transient for each bitplane that is decoded.
    fn reset_for_next_bitplane(&mut self) {
        let padded_width = self.padded_width as usize;
        let width = self.width as usize;
        let row_start = COEFFICIENTS_PADDING as usize;

        for row in self
            .coefficient_states
            .chunks_exact_mut(padded_width)
            .skip(COEFFICIENTS_PADDING as usize)
            .take(self.height as usize)
        {
            for state in &mut row[row_start..row_start + width] {
                state.0 &= !HAS_ZERO_CODING_MASK;
            }
        }
    }

    #[inline(always)]
    fn set_sign_index(&mut self, idx: usize, sign: u8) {
        self.coefficients[idx].set_sign(sign);
    }

    #[inline(always)]
    fn set_significant_index(&mut self, idx: usize, padded_width: usize) {
        let is_significant = self.coefficient_states[idx].is_significant();

        if !is_significant {
            self.coefficient_states[idx].set_significant();

            // Update all neighbors so they know this coefficient is significant
            // now.
            self.neighbor_significances[idx - padded_width - 1].set_bottom_right();
            self.neighbor_significances[idx - padded_width].set_bottom();
            self.neighbor_significances[idx - padded_width + 1].set_bottom_left();
            self.neighbor_significances[idx - 1].set_right();
            self.neighbor_significances[idx + 1].set_left();
            self.neighbor_significances[idx + padded_width - 1].set_top_right();
            self.neighbor_significances[idx + padded_width].set_top();
            self.neighbor_significances[idx + padded_width + 1].set_top_left();
        }
    }

    #[inline(always)]
    fn set_significant_index_for_path<const NORMAL_NEIGHBORS: bool>(
        &mut self,
        idx: usize,
        padded_width: usize,
    ) {
        if NORMAL_NEIGHBORS {
            self.set_significant_index_normal(idx, padded_width);
        } else {
            self.set_significant_index(idx, padded_width);
        }
    }

    #[inline(always)]
    fn set_significant_index_normal(&mut self, idx: usize, padded_width: usize) {
        if self.coefficient_states[idx].is_significant() {
            return;
        }

        self.coefficient_states[idx].set_significant();

        let top_start = idx - padded_width - 1;
        let top = &mut self.neighbor_significances[top_start..top_start + 3];
        top[0].set_bottom_right();
        top[1].set_bottom();
        top[2].set_bottom_left();

        let middle_start = idx - 1;
        let middle = &mut self.neighbor_significances[middle_start..middle_start + 3];
        middle[0].set_right();
        middle[2].set_left();

        let bottom_start = idx + padded_width - 1;
        let bottom = &mut self.neighbor_significances[bottom_start..bottom_start + 3];
        bottom[0].set_top_right();
        bottom[1].set_top();
        bottom[2].set_top_left();
    }

    #[inline(always)]
    fn push_magnitude_bit_index(&mut self, idx: usize, bit: u32) {
        self.coefficients[idx].push_bit_at(bit, self.current_bit_position);
    }

    #[inline(always)]
    fn sign_index(&self, idx: usize) -> u8 {
        self.coefficients[idx].sign() as u8
    }

    #[inline(always)]
    fn neighbor_in_next_stripe_y(&self, y: usize) -> bool {
        let neighbor_y = y + 1;
        neighbor_y < self.height as usize && (neighbor_y >> 2) > (y >> 2)
    }

    #[inline(always)]
    fn neighborhood_significance_states_index(&self, idx: usize, y: usize) -> u8 {
        let neighbors = &self.neighbor_significances[idx];

        if self.style.vertically_causal_context && self.neighbor_in_next_stripe_y(y) {
            neighbors.all_without_bottom()
        } else {
            neighbors.all()
        }
    }

    #[inline(always)]
    fn normal_neighborhood_significance_states_index(&self, idx: usize) -> u8 {
        self.neighbor_significances[idx].all()
    }

    #[inline(always)]
    fn uses_normal_arithmetic_neighbor_path(&self) -> bool {
        !self.style.selective_arithmetic_coding_bypass
            && !self.style.termination_on_each_pass
            && !self.style.vertically_causal_context
    }
}

fn decode_code_block_segments_inner(
    data: &[u8],
    segments: &[J2kCodeBlockSegment],
    number_of_coding_passes: u8,
    ctx: &mut BitPlaneDecodeContext,
) -> Option<()> {
    let mut expected_start = 0u8;

    for segment in segments {
        if segment.start_coding_pass != expected_start
            || segment.start_coding_pass > segment.end_coding_pass
        {
            return None;
        }
        expected_start = segment.end_coding_pass;

        let start_coding_pass = segment.start_coding_pass;
        let end_coding_pass = segment.end_coding_pass.min(ctx.max_coding_passes);
        let data_start = usize::try_from(segment.data_offset).ok()?;
        let data_length = usize::try_from(segment.data_length).ok()?;
        let data_end = data_start.checked_add(data_length)?;
        let segment_data = data.get(data_start..data_end)?;

        if segment.use_arithmetic {
            let mut decoder = ArithmeticDecoder::new(segment_data);
            if ctx.uses_normal_arithmetic_neighbor_path() {
                handle_normal_arithmetic_coding_passes(
                    start_coding_pass,
                    end_coding_pass,
                    ctx,
                    &mut decoder,
                )?;
            } else {
                handle_arithmetic_coding_passes(
                    start_coding_pass,
                    end_coding_pass,
                    ctx,
                    &mut decoder,
                )?;
            }
        } else {
            let mut decoder = BypassDecoder::new(segment_data, ctx.strict);
            handle_bypass_coding_passes(start_coding_pass, end_coding_pass, ctx, &mut decoder)?;
        }
    }

    if expected_start != number_of_coding_passes {
        return None;
    }

    Some(())
}

struct SafeScalarTier1;

impl SafeScalarTier1 {
    fn cleanup_pass_bypass(
        ctx: &mut BitPlaneDecodeContext,
        decoder: &mut BypassDecoder<'_>,
    ) -> Option<()> {
        let width = ctx.width as usize;
        let height = ctx.height as usize;
        let padded_width = ctx.padded_width as usize;

        for base_y in (0..height).step_by(4) {
            let y_end = (base_y + 4).min(height);
            let stripe_height = y_end - base_y;

            for x in 0..width {
                let top_idx = (base_y + COEFFICIENTS_PADDING as usize) * padded_width
                    + x
                    + COEFFICIENTS_PADDING as usize;

                if stripe_height == 4
                    && cleanup_run_length_candidate(ctx, top_idx, padded_width, base_y)
                {
                    let bit = decoder.read_bit(ctx.arithmetic_decoder_context(17))?;
                    if bit == 0 {
                        continue;
                    }

                    let first_significant =
                        (decoder.read_bit(ctx.arithmetic_decoder_context(18))? << 1)
                            | decoder.read_bit(ctx.arithmetic_decoder_context(18))?;
                    let first_significant = first_significant as usize;
                    let significant_y = base_y + first_significant;
                    let significant_idx = top_idx + first_significant * padded_width;
                    ctx.push_magnitude_bit_index(significant_idx, 1);
                    decode_sign_bit_bypass(significant_idx, significant_y, ctx, decoder)?;
                    ctx.set_significant_index(significant_idx, padded_width);

                    let mut idx = significant_idx + padded_width;
                    for y in significant_y + 1..y_end {
                        cleanup_coefficient_bypass(ctx, decoder, idx, y, padded_width)?;
                        idx += padded_width;
                    }
                    continue;
                }

                let mut idx = top_idx;
                for y in base_y..y_end {
                    cleanup_coefficient_bypass(ctx, decoder, idx, y, padded_width)?;
                    idx += padded_width;
                }
            }
        }

        Some(())
    }

    fn significance_propagation_pass_bypass(
        ctx: &mut BitPlaneDecodeContext,
        decoder: &mut BypassDecoder<'_>,
    ) -> Option<()> {
        let width = ctx.width as usize;
        let height = ctx.height as usize;
        let padded_width = ctx.padded_width as usize;

        for base_y in (0..height).step_by(4) {
            let y_end = (base_y + 4).min(height);
            for x in 0..width {
                let mut idx = (base_y + COEFFICIENTS_PADDING as usize) * padded_width
                    + x
                    + COEFFICIENTS_PADDING as usize;

                for y in base_y..y_end {
                    let state = ctx.coefficient_states[idx].0;
                    let neighbors = ctx.neighborhood_significance_states_index(idx, y);

                    if state & SIGNIFICANCE_MASK == 0 && neighbors != 0 {
                        let ctx_label =
                            context_label_zero_coding_from_neighbors(neighbors, ctx.sub_band_type);
                        let bit = decoder.read_bit(ctx.arithmetic_decoder_context(ctx_label))?;
                        ctx.push_magnitude_bit_index(idx, bit);
                        ctx.coefficient_states[idx].0 |= HAS_ZERO_CODING_MASK;

                        if bit == 1 {
                            decode_sign_bit_bypass(idx, y, ctx, decoder)?;
                            ctx.set_significant_index(idx, padded_width);
                        }
                    }

                    idx += padded_width;
                }
            }
        }

        Some(())
    }

    fn magnitude_refinement_pass_bypass(
        ctx: &mut BitPlaneDecodeContext,
        decoder: &mut BypassDecoder<'_>,
    ) -> Option<()> {
        let width = ctx.width as usize;
        let height = ctx.height as usize;
        let padded_width = ctx.padded_width as usize;

        for base_y in (0..height).step_by(4) {
            let y_end = (base_y + 4).min(height);
            for x in 0..width {
                let mut idx = (base_y + COEFFICIENTS_PADDING as usize) * padded_width
                    + x
                    + COEFFICIENTS_PADDING as usize;

                for y in base_y..y_end {
                    let state = ctx.coefficient_states[idx].0;

                    if state & SIGNIFICANCE_MASK != 0 && state & HAS_ZERO_CODING_MASK == 0 {
                        let neighbors = ctx.neighborhood_significance_states_index(idx, y);
                        let ctx_label =
                            context_label_magnitude_refinement_coding_from_state(state, neighbors);
                        let bit = decoder.read_bit(ctx.arithmetic_decoder_context(ctx_label))?;
                        ctx.push_magnitude_bit_index(idx, bit);
                        ctx.coefficient_states[idx].0 |= HAS_MAGNITUDE_REFINEMENT_MASK;
                    }

                    idx += padded_width;
                }
            }
        }

        Some(())
    }
}

fn cleanup_pass_arithmetic_with_neighbors<const NORMAL_NEIGHBORS: bool>(
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
) {
    let width = ctx.width as usize;
    let height = ctx.height as usize;
    let padded_width = ctx.padded_width as usize;

    for base_y in (0..height).step_by(4) {
        let y_end = (base_y + 4).min(height);
        let stripe_height = y_end - base_y;

        for x in 0..width {
            let top_idx = (base_y + COEFFICIENTS_PADDING as usize) * padded_width
                + x
                + COEFFICIENTS_PADDING as usize;

            if stripe_height == 4
                && cleanup_run_length_candidate_with_neighbors::<NORMAL_NEIGHBORS>(
                    ctx,
                    top_idx,
                    padded_width,
                    base_y,
                )
            {
                // The four contiguous samples are all cleanup candidates
                // with zero context, so Annex D permits the RLC context.
                let bit = decoder.read_bit(ctx.arithmetic_decoder_context(17));
                if bit == 0 {
                    continue;
                }

                let first_significant = (decoder.read_bit(ctx.arithmetic_decoder_context(18)) << 1)
                    | decoder.read_bit(ctx.arithmetic_decoder_context(18));
                let first_significant = first_significant as usize;
                let significant_y = base_y + first_significant;
                let significant_idx = top_idx + first_significant * padded_width;
                ctx.push_magnitude_bit_index(significant_idx, 1);
                decode_sign_bit_arithmetic_with_neighbors::<NORMAL_NEIGHBORS>(
                    significant_idx,
                    significant_y,
                    ctx,
                    decoder,
                );
                ctx.set_significant_index_for_path::<NORMAL_NEIGHBORS>(
                    significant_idx,
                    padded_width,
                );

                let mut idx = significant_idx + padded_width;
                for y in significant_y + 1..y_end {
                    cleanup_coefficient_arithmetic_with_neighbors::<NORMAL_NEIGHBORS>(
                        ctx,
                        decoder,
                        idx,
                        y,
                        padded_width,
                    );
                    idx += padded_width;
                }
                continue;
            }

            let mut idx = top_idx;
            for y in base_y..y_end {
                cleanup_coefficient_arithmetic_with_neighbors::<NORMAL_NEIGHBORS>(
                    ctx,
                    decoder,
                    idx,
                    y,
                    padded_width,
                );
                idx += padded_width;
            }
        }
    }
}

fn significance_propagation_pass_arithmetic_with_neighbors<const NORMAL_NEIGHBORS: bool>(
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
) {
    let width = ctx.width as usize;
    let height = ctx.height as usize;
    let padded_width = ctx.padded_width as usize;

    for base_y in (0..height).step_by(4) {
        let y_end = (base_y + 4).min(height);
        for x in 0..width {
            let mut idx = (base_y + COEFFICIENTS_PADDING as usize) * padded_width
                + x
                + COEFFICIENTS_PADDING as usize;

            for y in base_y..y_end {
                let state = ctx.coefficient_states[idx].0;
                let neighbors =
                    neighborhood_significance_states_for_path::<NORMAL_NEIGHBORS>(ctx, idx, y);

                // "The significance propagation pass only includes bits of coefficients
                // that were insignificant (the significance state has yet to be set)
                // and have a non-zero context."
                if state & SIGNIFICANCE_MASK == 0 && neighbors != 0 {
                    let ctx_label =
                        context_label_zero_coding_from_neighbors(neighbors, ctx.sub_band_type);
                    let bit = decoder.read_bit(ctx.arithmetic_decoder_context(ctx_label));
                    ctx.push_magnitude_bit_index(idx, bit);
                    ctx.coefficient_states[idx].0 |= HAS_ZERO_CODING_MASK;

                    // "If the value of this bit is 1 then the significance
                    // state is set to 1 and the immediate next bit to be decoded is
                    // the sign bit for the coefficient. Otherwise, the significance
                    // state remains 0."
                    if bit == 1 {
                        decode_sign_bit_arithmetic_with_neighbors::<NORMAL_NEIGHBORS>(
                            idx, y, ctx, decoder,
                        );
                        ctx.set_significant_index_for_path::<NORMAL_NEIGHBORS>(idx, padded_width);
                    }
                }

                idx += padded_width;
            }
        }
    }
}

fn magnitude_refinement_pass_arithmetic_with_neighbors<const NORMAL_NEIGHBORS: bool>(
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
) {
    let width = ctx.width as usize;
    let height = ctx.height as usize;
    let padded_width = ctx.padded_width as usize;

    for base_y in (0..height).step_by(4) {
        let y_end = (base_y + 4).min(height);
        for x in 0..width {
            let mut idx = (base_y + COEFFICIENTS_PADDING as usize) * padded_width
                + x
                + COEFFICIENTS_PADDING as usize;

            for y in base_y..y_end {
                let state = ctx.coefficient_states[idx].0;

                if state & SIGNIFICANCE_MASK != 0 && state & HAS_ZERO_CODING_MASK == 0 {
                    let neighbors =
                        neighborhood_significance_states_for_path::<NORMAL_NEIGHBORS>(ctx, idx, y);
                    let ctx_label =
                        context_label_magnitude_refinement_coding_from_state(state, neighbors);
                    let bit = decoder.read_bit(ctx.arithmetic_decoder_context(ctx_label));
                    ctx.push_magnitude_bit_index(idx, bit);
                    ctx.coefficient_states[idx].0 |= HAS_MAGNITUDE_REFINEMENT_MASK;
                }

                idx += padded_width;
            }
        }
    }
}

#[inline(always)]
fn neighborhood_significance_states_for_path<const NORMAL_NEIGHBORS: bool>(
    ctx: &BitPlaneDecodeContext,
    idx: usize,
    y: usize,
) -> u8 {
    if NORMAL_NEIGHBORS {
        ctx.normal_neighborhood_significance_states_index(idx)
    } else {
        ctx.neighborhood_significance_states_index(idx, y)
    }
}

#[inline(always)]
fn cleanup_run_length_candidate_with_neighbors<const NORMAL_NEIGHBORS: bool>(
    ctx: &BitPlaneDecodeContext,
    top_idx: usize,
    padded_width: usize,
    base_y: usize,
) -> bool {
    let mut idx = top_idx;
    for y in base_y..base_y + 4 {
        if ctx.coefficient_states[idx].0 & (SIGNIFICANCE_MASK | HAS_ZERO_CODING_MASK) != 0
            || neighborhood_significance_states_for_path::<NORMAL_NEIGHBORS>(ctx, idx, y) != 0
        {
            return false;
        }
        idx += padded_width;
    }

    true
}

#[inline(always)]
fn cleanup_coefficient_arithmetic_with_neighbors<const NORMAL_NEIGHBORS: bool>(
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
    idx: usize,
    y: usize,
    padded_width: usize,
) {
    if ctx.coefficient_states[idx].0 & (SIGNIFICANCE_MASK | HAS_ZERO_CODING_MASK) == 0 {
        let neighbors = neighborhood_significance_states_for_path::<NORMAL_NEIGHBORS>(ctx, idx, y);
        let ctx_label = context_label_zero_coding_from_neighbors(neighbors, ctx.sub_band_type);
        let bit = decoder.read_bit(ctx.arithmetic_decoder_context(ctx_label));
        ctx.push_magnitude_bit_index(idx, bit);

        if bit == 1 {
            decode_sign_bit_arithmetic_with_neighbors::<NORMAL_NEIGHBORS>(idx, y, ctx, decoder);
            ctx.set_significant_index_for_path::<NORMAL_NEIGHBORS>(idx, padded_width);
        }
    }
}

#[inline(always)]
fn cleanup_run_length_candidate(
    ctx: &BitPlaneDecodeContext,
    top_idx: usize,
    padded_width: usize,
    base_y: usize,
) -> bool {
    cleanup_run_length_candidate_with_neighbors::<false>(ctx, top_idx, padded_width, base_y)
}

#[inline(always)]
fn cleanup_coefficient_bypass(
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut BypassDecoder<'_>,
    idx: usize,
    y: usize,
    padded_width: usize,
) -> Option<()> {
    if ctx.coefficient_states[idx].0 & (SIGNIFICANCE_MASK | HAS_ZERO_CODING_MASK) == 0 {
        let neighbors = ctx.neighborhood_significance_states_index(idx, y);
        let ctx_label = context_label_zero_coding_from_neighbors(neighbors, ctx.sub_band_type);
        let bit = decoder.read_bit(ctx.arithmetic_decoder_context(ctx_label))?;
        ctx.push_magnitude_bit_index(idx, bit);

        if bit == 1 {
            decode_sign_bit_bypass(idx, y, ctx, decoder)?;
            ctx.set_significant_index(idx, padded_width);
        }
    }

    Some(())
}

/// See `context_label_sign_coding`. This table contains all context labels
/// for each combination of the bit-packed field. (0, 0) represent
/// impossible combinations.
#[rustfmt::skip]
const SIGN_CONTEXT_LOOKUP: [(u8, u8); 256] = [
    (9,0), (10,0), (10,1), (0,0), (12,0), (13,0), (11,0), (0,0), (12,1), (11,1),
    (13,1), (0,0), (0,0), (0,0), (0,0), (0,0), (12,0), (13,0), (11,0), (0,0),
    (12,0), (13,0), (11,0), (0,0), (9,0), (10,0), (10,1), (0,0), (0,0), (0,0),
    (0,0), (0,0), (12,1), (11,1), (13,1), (0,0), (9,0), (10,0), (10,1), (0,0),
    (12,1), (11,1), (13,1), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (10,0), (10,0), (9,0), (0,0), (13,0), (13,0), (12,0),
    (0,0), (11,1), (11,1), (12,1), (0,0), (0,0), (0,0), (0,0), (0,0), (13,0),
    (13,0), (12,0), (0,0), (13,0), (13,0), (12,0), (0,0), (10,0), (10,0), (9,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (11,1), (11,1), (12,1), (0,0), (10,0),
    (10,0), (9,0), (0,0), (11,1), (11,1), (12,1), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (10,1), (9,0), (10,1), (0,0),
    (11,0), (12,0), (11,0), (0,0), (13,1), (12,1), (13,1), (0,0), (0,0), (0,0),
    (0,0), (0,0), (11,0), (12,0), (11,0), (0,0), (11,0), (12,0), (11,0), (0,0),
    (10,1), (9,0), (10,1), (0,0), (0,0), (0,0), (0,0), (0,0), (13,1), (12,1),
    (13,1), (0,0), (10,1), (9,0), (10,1), (0,0), (13,1), (12,1), (13,1), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
];

#[rustfmt::skip]
const ZERO_CTX_LL_LH_LOOKUP: [u8; 256] = [
    0, 3, 1, 3, 5, 7, 6, 7, 1, 3, 2, 3, 6, 7, 6, 7, 5, 7, 6, 7, 8, 8, 8, 8, 6,
    7, 6, 7, 8, 8, 8, 8, 1, 3, 2, 3, 6, 7, 6, 7, 2, 3, 2, 3, 6, 7, 6, 7, 6, 7,
    6, 7, 8, 8, 8, 8, 6, 7, 6, 7, 8, 8, 8, 8, 3, 4, 3, 4, 7, 7, 7, 7, 3, 4, 3,
    4, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 7, 7, 7, 7, 8, 8, 8, 8, 3, 4, 3, 4,
    7, 7, 7, 7, 3, 4, 3, 4, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 7, 7, 7, 7, 8,
    8, 8, 8, 1, 3, 2, 3, 6, 7, 6, 7, 2, 3, 2, 3, 6, 7, 6, 7, 6, 7, 6, 7, 8, 8,
    8, 8, 6, 7, 6, 7, 8, 8, 8, 8, 2, 3, 2, 3, 6, 7, 6, 7, 2, 3, 2, 3, 6, 7, 6,
    7, 6, 7, 6, 7, 8, 8, 8, 8, 6, 7, 6, 7, 8, 8, 8, 8, 3, 4, 3, 4, 7, 7, 7, 7,
    3, 4, 3, 4, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 7, 7, 7, 7, 8, 8, 8, 8, 3,
    4, 3, 4, 7, 7, 7, 7, 3, 4, 3, 4, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 7, 7,
    7, 7, 8, 8, 8, 8,
];

#[rustfmt::skip]
const ZERO_CTX_HL_LOOKUP: [u8; 256] = [
    0, 5, 1, 6, 3, 7, 3, 7, 1, 6, 2, 6, 3, 7, 3, 7, 3, 7, 3, 7, 4, 7, 4, 7, 3,
    7, 3, 7, 4, 7, 4, 7, 1, 6, 2, 6, 3, 7, 3, 7, 2, 6, 2, 6, 3, 7, 3, 7, 3, 7,
    3, 7, 4, 7, 4, 7, 3, 7, 3, 7, 4, 7, 4, 7, 5, 8, 6, 8, 7, 8, 7, 8, 6, 8, 6,
    8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 6, 8, 6, 8,
    7, 8, 7, 8, 6, 8, 6, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7,
    8, 7, 8, 1, 6, 2, 6, 3, 7, 3, 7, 2, 6, 2, 6, 3, 7, 3, 7, 3, 7, 3, 7, 4, 7,
    4, 7, 3, 7, 3, 7, 4, 7, 4, 7, 2, 6, 2, 6, 3, 7, 3, 7, 2, 6, 2, 6, 3, 7, 3,
    7, 3, 7, 3, 7, 4, 7, 4, 7, 3, 7, 3, 7, 4, 7, 4, 7, 6, 8, 6, 8, 7, 8, 7, 8,
    6, 8, 6, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 6,
    8, 6, 8, 7, 8, 7, 8, 6, 8, 6, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8,
    7, 8, 7, 8, 7, 8,
];

#[rustfmt::skip]
const ZERO_CTX_HH_LOOKUP: [u8; 256] = [
    0, 1, 3, 4, 1, 2, 4, 5, 3, 4, 6, 7, 4, 5, 7, 7, 1, 2, 4, 5, 2, 2, 5, 5, 4,
    5, 7, 7, 5, 5, 7, 7, 3, 4, 6, 7, 4, 5, 7, 7, 6, 7, 8, 8, 7, 7, 8, 8, 4, 5,
    7, 7, 5, 5, 7, 7, 7, 7, 8, 8, 7, 7, 8, 8, 1, 2, 4, 5, 2, 2, 5, 5, 4, 5, 7,
    7, 5, 5, 7, 7, 2, 2, 5, 5, 2, 2, 5, 5, 5, 5, 7, 7, 5, 5, 7, 7, 4, 5, 7, 7,
    5, 5, 7, 7, 7, 7, 8, 8, 7, 7, 8, 8, 5, 5, 7, 7, 5, 5, 7, 7, 7, 7, 8, 8, 7,
    7, 8, 8, 3, 4, 6, 7, 4, 5, 7, 7, 6, 7, 8, 8, 7, 7, 8, 8, 4, 5, 7, 7, 5, 5,
    7, 7, 7, 7, 8, 8, 7, 7, 8, 8, 6, 7, 8, 8, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 7, 7, 8, 8, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 5, 7, 7, 5, 5, 7, 7,
    7, 7, 8, 8, 7, 7, 8, 8, 5, 5, 7, 7, 5, 5, 7, 7, 7, 7, 8, 8, 7, 7, 8, 8, 7,
    7, 8, 8, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8,
];

#[inline(always)]
fn decode_sign_bit_arithmetic_with_neighbors<const NORMAL_NEIGHBORS: bool>(
    idx: usize,
    y: usize,
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
) {
    let (ctx_label, xor_bit) =
        context_label_sign_coding_index_with_neighbors::<NORMAL_NEIGHBORS>(idx, y, ctx);
    let sign_bit = decoder.read_bit(ctx.arithmetic_decoder_context(ctx_label)) ^ xor_bit as u32;
    ctx.set_sign_index(idx, sign_bit as u8);
}

/// Decode a raw bypass sign bit (Section D.3.2).
#[inline(always)]
fn decode_sign_bit_bypass(
    idx: usize,
    y: usize,
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut BypassDecoder<'_>,
) -> Option<()> {
    let (ctx_label, xor_bit) = context_label_sign_coding_index(idx, y, ctx);
    let ad_ctx = ctx.arithmetic_decoder_context(ctx_label);
    let _ = xor_bit;
    let sign_bit = decoder.read_bit(ad_ctx)?;
    ctx.set_sign_index(idx, sign_bit as u8);

    Some(())
}

/// Based on Table D.2.
#[inline(always)]
fn context_label_sign_coding_index(idx: usize, y: usize, ctx: &BitPlaneDecodeContext) -> (u8, u8) {
    // A lot of subtleties go into this path. We need the significances and
    // signs of the four cardinal neighbors and then assign a context label
    // based on the signed sum, without branching on each neighbor.
    let significances = ctx.neighborhood_significance_states_index(idx, y) & 0b0101_0101;
    let padded_width = ctx.padded_width as usize;

    let top_sign = ctx.sign_index(idx - padded_width);
    let left_sign = ctx.sign_index(idx - 1);
    let right_sign = ctx.sign_index(idx + 1);
    let bottom_sign = if ctx.style.vertically_causal_context && ctx.neighbor_in_next_stripe_y(y) {
        0
    } else {
        ctx.sign_index(idx + padded_width)
    };

    // Due to the specific layout of `NeighborSignificances`, direct neighbors
    // and diagonals are interleaved. Therefore, we create a new bit-packed
    // representation that indicates whether the top/left/right/bottom sign is
    // positive, negative, or insignificant. We need two bits for this.
    // 00 represents insignificant, 01 positive and 10 negative.
    let signs = (top_sign << 6) | (left_sign << 4) | (right_sign << 2) | bottom_sign;
    let negative_significances = significances & signs;
    let positive_significances = significances & !signs;
    let merged_significances = (negative_significances << 1) | positive_significances;

    SIGN_CONTEXT_LOOKUP[merged_significances as usize]
}

#[inline(always)]
fn context_label_sign_coding_index_with_neighbors<const NORMAL_NEIGHBORS: bool>(
    idx: usize,
    y: usize,
    ctx: &BitPlaneDecodeContext,
) -> (u8, u8) {
    if NORMAL_NEIGHBORS {
        context_label_sign_coding_index_normal(idx, ctx)
    } else {
        context_label_sign_coding_index(idx, y, ctx)
    }
}

#[inline(always)]
fn context_label_sign_coding_index_normal(idx: usize, ctx: &BitPlaneDecodeContext) -> (u8, u8) {
    let significances = ctx.normal_neighborhood_significance_states_index(idx) & 0b0101_0101;
    let padded_width = ctx.padded_width as usize;

    let top_sign = ctx.sign_index(idx - padded_width);
    let left_sign = ctx.sign_index(idx - 1);
    let right_sign = ctx.sign_index(idx + 1);
    let bottom_sign = ctx.sign_index(idx + padded_width);

    let signs = (top_sign << 6) | (left_sign << 4) | (right_sign << 2) | bottom_sign;
    let negative_significances = significances & signs;
    let positive_significances = significances & !signs;
    let merged_significances = (negative_significances << 1) | positive_significances;

    SIGN_CONTEXT_LOOKUP[merged_significances as usize]
}

/// Return the context label for zero coding (Section D.3.1).
#[inline(always)]
fn context_label_zero_coding_from_neighbors(neighbors: u8, sub_band_type: SubBandType) -> u8 {
    // Once again, the neighbors field is bit-packed, so we can just generate
    // a table for all u8 values and assign the correct context based on the
    // exact value of that field.
    match sub_band_type {
        SubBandType::LowLow | SubBandType::LowHigh => ZERO_CTX_LL_LH_LOOKUP[neighbors as usize],
        SubBandType::HighLow => ZERO_CTX_HL_LOOKUP[neighbors as usize],
        SubBandType::HighHigh => ZERO_CTX_HH_LOOKUP[neighbors as usize],
    }
}

/// Return the context label for magnitude refinement coding (Table D.4).
#[inline(always)]
fn context_label_magnitude_refinement_coding_from_state(state: u8, neighbors: u8) -> u8 {
    // If magnitude refined, then 16.
    let m1 = ((state & HAS_MAGNITUDE_REFINEMENT_MASK) >> HAS_MAGNITUDE_REFINEMENT_SHIFT) * 16;
    // Else: If at least one neighbor is significant then 15, else 14.
    let m2 = 14 + neighbors.min(1);

    u8::max(m1, m2)
}

// Bypass bit reads can fail in strict mode when the raw segment runs short.
trait BitDecoder {
    fn read_bit(&mut self, context: &mut ArithmeticDecoderContext) -> Option<u32>;
}

struct BypassDecoder<'a>(BitReader<'a>, bool);

impl<'a> BypassDecoder<'a> {
    fn new(data: &'a [u8], strict: bool) -> Self {
        Self(BitReader::new(data), strict)
    }
}

impl BitDecoder for BypassDecoder<'_> {
    fn read_bit(&mut self, _: &mut ArithmeticDecoderContext) -> Option<u32> {
        self.0.read_bits_with_stuffing(1).or({
            if !self.1 {
                // If not in strict mode, just pad with ones. Not sure if
                // zeroes would be better here, but since the arithmetic decoder
                // is also padded with 0xFF maybe 1 is the better choice?
                Some(1)
            } else {
                // We have too little data, return `None`.
                None
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::bitplane_encode;
    use super::*;

    fn seed_130_cb_coefficients() -> Vec<i32> {
        let mut coefficients = Vec::with_capacity(64 * 64);
        let mut state = 130u32 ^ 0x9e37_79b9;
        for _ in 0..64 * 64 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let _r = (state >> 24) as u8;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let g = (state >> 24) as u8;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let b = (state >> 24) as u8;
            coefficients.push(i32::from(b) - i32::from(g));
        }
        coefficients
    }

    fn generated_coefficients(width: u32, height: u32, seed: u32) -> Vec<i32> {
        let mut coefficients = Vec::with_capacity(width as usize * height as usize);
        let mut state = seed ^ 0x9e37_79b9;
        for idx in 0..width * height {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let value = ((state >> 16) & 0x01ff) as i32 - 255;
            coefficients.push(if (idx + seed).is_multiple_of(11) {
                0
            } else {
                value
            });
        }
        coefficients
    }

    fn assert_code_block_round_trip(
        style: CodeBlockStyle,
        sub_band_type: SubBandType,
        width: u32,
        height: u32,
        seed: u32,
    ) {
        let total_bitplanes = 10;
        let coefficients = generated_coefficients(width, height, seed);
        let encoded = bitplane_encode::encode_code_block_segments_with_style(
            &coefficients,
            width,
            height,
            sub_band_type,
            total_bitplanes,
            &style,
        );
        let segments = encoded
            .segments
            .iter()
            .map(|segment| J2kCodeBlockSegment {
                data_offset: segment.data_offset,
                data_length: segment.data_length,
                start_coding_pass: segment.start_coding_pass,
                end_coding_pass: segment.end_coding_pass,
                use_arithmetic: segment.use_arithmetic,
            })
            .collect::<Vec<_>>();
        let mut ctx = BitPlaneDecodeContext::default();

        decode_code_block_segments_validated(
            &encoded.data,
            &segments,
            width,
            height,
            encoded.num_zero_bitplanes,
            encoded.num_coding_passes,
            total_bitplanes,
            sub_band_type,
            &style,
            true,
            &mut ctx,
        )
        .expect("decode code block");

        let decoded = ctx
            .coefficient_rows()
            .flat_map(|row| row.iter().map(Coefficient::get))
            .collect::<Vec<_>>();
        if let Some(index) = decoded
            .iter()
            .zip(coefficients.iter())
            .position(|(actual, expected)| actual != expected)
        {
            panic!(
                "coefficient mismatch at {index}: expected {}, got {}",
                coefficients[index], decoded[index]
            );
        }
    }

    #[test]
    fn classic_bitplane_round_trips_seed_130_cb_block() {
        let coefficients = seed_130_cb_coefficients();
        let style = CodeBlockStyle::default();
        let encoded = bitplane_encode::encode_code_block_segments_with_style(
            &coefficients,
            64,
            64,
            SubBandType::LowLow,
            8,
            &style,
        );
        let segments = encoded
            .segments
            .iter()
            .map(|segment| J2kCodeBlockSegment {
                data_offset: segment.data_offset,
                data_length: segment.data_length,
                start_coding_pass: segment.start_coding_pass,
                end_coding_pass: segment.end_coding_pass,
                use_arithmetic: segment.use_arithmetic,
            })
            .collect::<Vec<_>>();
        let mut ctx = BitPlaneDecodeContext::default();

        decode_code_block_segments_validated(
            &encoded.data,
            &segments,
            64,
            64,
            encoded.num_zero_bitplanes,
            encoded.num_coding_passes,
            8,
            SubBandType::LowLow,
            &style,
            true,
            &mut ctx,
        )
        .expect("decode code block");

        let decoded = ctx
            .coefficient_rows()
            .flat_map(|row| row.iter().map(Coefficient::get))
            .collect::<Vec<_>>();
        let mismatch_count = decoded
            .iter()
            .zip(coefficients.iter())
            .filter(|(actual, expected)| actual != expected)
            .count();
        if let Some(index) = decoded
            .iter()
            .zip(coefficients.iter())
            .position(|(actual, expected)| actual != expected)
        {
            panic!(
                "{mismatch_count} coefficient mismatch(es); first at {index}: expected {}, got {}",
                coefficients[index], decoded[index]
            );
        }
    }

    #[test]
    fn normal_neighborhood_significance_fast_path_returns_unmasked_neighbors() {
        let mut ctx = BitPlaneDecodeContext {
            width: 1,
            height: 8,
            padded_width: 3,
            style: CodeBlockStyle {
                vertically_causal_context: true,
                ..CodeBlockStyle::default()
            },
            ..BitPlaneDecodeContext::default()
        };
        ctx.neighbor_significances.resize(
            ctx.padded_width as usize * 10,
            NeighborSignificances::default(),
        );

        let y = 3;
        let idx = (y + COEFFICIENTS_PADDING as usize) * ctx.padded_width as usize
            + COEFFICIENTS_PADDING as usize;
        ctx.neighbor_significances[idx].set_top();
        ctx.neighbor_significances[idx].set_bottom();

        assert_eq!(ctx.neighborhood_significance_states_index(idx, y), 1 << 6);
        assert_eq!(
            ctx.normal_neighborhood_significance_states_index(idx),
            (1 << 6) | 1
        );
    }

    #[test]
    fn normal_sign_context_matches_generic_non_vertical_context() {
        let mut ctx = BitPlaneDecodeContext {
            width: 3,
            height: 3,
            padded_width: 5,
            style: CodeBlockStyle::default(),
            ..BitPlaneDecodeContext::default()
        };
        let len = ctx.padded_width as usize * (ctx.height as usize + 2);
        ctx.coefficients.resize(len, Coefficient::default());
        ctx.neighbor_significances
            .resize(len, NeighborSignificances::default());

        let y = 1;
        let idx = (y + COEFFICIENTS_PADDING as usize) * ctx.padded_width as usize
            + COEFFICIENTS_PADDING as usize
            + 1;
        let padded_width = ctx.padded_width as usize;

        ctx.neighbor_significances[idx].set_top();
        ctx.neighbor_significances[idx].set_left();
        ctx.neighbor_significances[idx].set_right();
        ctx.neighbor_significances[idx].set_bottom();
        ctx.set_sign_index(idx - padded_width, 1);
        ctx.set_sign_index(idx - 1, 0);
        ctx.set_sign_index(idx + 1, 1);
        ctx.set_sign_index(idx + padded_width, 0);

        assert_eq!(
            context_label_sign_coding_index_normal(idx, &ctx),
            context_label_sign_coding_index(idx, y, &ctx)
        );
    }

    #[test]
    fn normal_set_significant_index_matches_generic_neighbor_updates() {
        let mut generic = BitPlaneDecodeContext {
            width: 3,
            height: 3,
            padded_width: 5,
            style: CodeBlockStyle::default(),
            ..BitPlaneDecodeContext::default()
        };
        let len = generic.padded_width as usize * (generic.height as usize + 2);
        generic
            .coefficient_states
            .resize(len, CoefficientState::default());
        generic
            .neighbor_significances
            .resize(len, NeighborSignificances::default());
        let mut normal = BitPlaneDecodeContext {
            width: generic.width,
            height: generic.height,
            padded_width: generic.padded_width,
            style: generic.style,
            coefficient_states: generic.coefficient_states.clone(),
            neighbor_significances: generic.neighbor_significances.clone(),
            ..BitPlaneDecodeContext::default()
        };

        let padded_width = generic.padded_width as usize;
        let idx =
            (1 + COEFFICIENTS_PADDING as usize) * padded_width + COEFFICIENTS_PADDING as usize + 1;

        generic.set_significant_index(idx, padded_width);
        normal.set_significant_index_normal(idx, padded_width);

        assert_eq!(
            normal.coefficient_states[idx].0,
            generic.coefficient_states[idx].0
        );
        assert_eq!(
            normal
                .neighbor_significances
                .iter()
                .map(|neighbors| neighbors.0)
                .collect::<Vec<_>>(),
            generic
                .neighbor_significances
                .iter()
                .map(|neighbors| neighbors.0)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn classic_bitplane_round_trips_subband_and_style_matrix() {
        let styles = [
            CodeBlockStyle::default(),
            CodeBlockStyle {
                selective_arithmetic_coding_bypass: true,
                ..CodeBlockStyle::default()
            },
            CodeBlockStyle {
                termination_on_each_pass: true,
                reset_context_probabilities: true,
                ..CodeBlockStyle::default()
            },
            CodeBlockStyle {
                segmentation_symbols: true,
                ..CodeBlockStyle::default()
            },
            CodeBlockStyle {
                vertically_causal_context: true,
                ..CodeBlockStyle::default()
            },
        ];
        let subbands = [
            SubBandType::LowLow,
            SubBandType::LowHigh,
            SubBandType::HighLow,
            SubBandType::HighHigh,
        ];

        for (style_idx, style) in styles.into_iter().enumerate() {
            for (subband_idx, sub_band_type) in subbands.into_iter().enumerate() {
                assert_code_block_round_trip(
                    style,
                    sub_band_type,
                    32,
                    19,
                    0x4a32_1000 + style_idx as u32 * 17 + subband_idx as u32,
                );
            }
        }
    }
}
