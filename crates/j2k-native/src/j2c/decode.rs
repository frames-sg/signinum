//! Decoding JPEG2000 code streams.
//!
//! This is the "core" module of the crate that orchestrates all
//! stages in such a way that a given codestream is decoded into its
//! component channels.

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::bitplane::{BitPlaneDecodeBuffers, BitPlaneDecodeContext};
use super::build::{CodeBlock, Decomposition, Layer, Precinct, Segment, SubBand, SubBandType};
use super::codestream::{ComponentInfo, Header, QuantizationStyle, WaveletTransform};
use super::ht_block_decode::{self, HtBlockDecodeContext};
use super::idwt::IDWTOutput;
use super::progression::{progression_iterator, ProgressionData};
use super::roi::{tile_intersects_output_region, RoiPlan};
use super::tag_tree::TagNode;
use super::tile::{ComponentTile, ParsedTiles, ResolutionTile, Tile};
use super::{bitplane, build, idwt, mct, segment, tile, ComponentData};
use crate::error::{
    bail, ColorError, DecodeError, DecodingError, DirectPlanUnsupportedReason, Result, TileError,
    ValidationError,
};
use crate::j2c::segment::MAX_BITPLANE_COUNT;
use crate::profile;
use crate::reader::BitReader;
use crate::{
    add_roi_shift_to_bitplanes, apply_roi_maxshift_inverse_f32, apply_roi_maxshift_inverse_i32,
    apply_roi_maxshift_inverse_i64, decode_j2k_code_block_scalar_with_workspace,
    decode_j2k_code_block_scalar_with_workspace_midpoint, HtCodeBlockBatchJob,
    HtCodeBlockDecodeJob, HtCodeBlockDecoder, HtOwnedCodeBlockBatchJob, HtOwnedSubBandPlan,
    HtSubBandDecodeJob, J2kCodeBlockBatchJob, J2kCodeBlockDecodeJob, J2kCodeBlockDecodeWorkspace,
    J2kCodeBlockStyle, J2kDirectBandId, J2kDirectColorPlan, J2kDirectGrayscalePlan,
    J2kDirectGrayscaleStep, J2kDirectIdwtStep, J2kDirectStoreStep, J2kOwnedCodeBlockBatchJob,
    J2kOwnedSubBandPlan, J2kRect, J2kStoreComponentJob, J2kSubBandDecodeJob, J2kSubBandType,
    J2kWaveletTransform,
};
use core::mem::size_of;
use core::ops::Range;

mod allocation;
mod direct_plan;
mod reuse;
mod store;
mod subband;
mod subband_params;
mod tier1;
mod workspace;
pub(crate) use self::allocation::DecodeAllocationBudget;
use self::direct_plan::collect_classic_code_block_data;
pub(crate) use self::direct_plan::{
    build_component_grid_color_plan, build_direct_color_plan, build_direct_grayscale_plan,
    build_referenced_classic_color_plan, build_referenced_classic_grayscale_plan,
    build_referenced_classic_rgba_plan, build_referenced_htj2k_color_plan,
    build_referenced_htj2k_grayscale_plan, build_referenced_htj2k_rgba_plan,
};
use self::store::{apply_sign_shift_after_mct, component_unsigned_level_shift, store};
use self::subband::{
    code_block_required_by_index, count_classic_code_blocks, count_ht_code_blocks,
    decode_component_tile_bit_planes,
};
#[cfg(all(test, feature = "parallel"))]
use self::subband::{
    copy_decoded_classic_blocks_to_sub_band, copy_decoded_ht_blocks_to_sub_band,
    DecodedClassicBlock, DecodedHtBlock,
};
#[cfg(test)]
pub(crate) use self::subband::{
    should_decode_classic_sub_band_in_parallel, should_decode_ht_sub_band_in_parallel,
};
use self::subband_params::{
    classic_decode_job_parameters, ht_code_block_has_decodable_passes, sub_band_decode_parameters,
    SubBandDecodeParameters,
};
pub use self::workspace::{
    CpuDecodeParallelism, DecoderContext, DecoderWorkspace, DecoderWorkspaceStats,
};

pub(crate) fn decode_with_capacity_retry<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
) -> Result<()> {
    run_decode_with_capacity_retry(ctx, |ctx| {
        decode(data, header, retained_image_bytes, ctx, ht_decoder)
    })
}

fn run_decode_with_capacity_retry<'a>(
    ctx: &mut DecoderContext<'a>,
    mut decode: impl FnMut(&mut DecoderContext<'a>) -> Result<()>,
) -> Result<()> {
    ctx.storage.release_all_allocations();
    let retained_component_bytes = ctx.tile_decode_context.retained_channel_bytes()?;
    let retained_tier1_bytes = ctx.tile_decode_context.tier1_capacity_bytes()?;
    let retained_idwt_bytes = ctx.tile_decode_context.idwt_capacity_bytes()?;
    let retained_scratch_bytes = retained_tier1_bytes
        .checked_add(retained_idwt_bytes)
        .ok_or(ValidationError::ImageTooLarge)?;
    ctx.record_decode_start(
        retained_component_bytes,
        retained_tier1_bytes,
        retained_idwt_bytes,
    );

    let first_result = decode(ctx);
    let result = ctx.retry_without_retained_scratch_on_capacity(
        retained_scratch_bytes,
        first_result,
        &mut decode,
    );

    ctx.storage.release_all_allocations();
    let retained_component_bytes = ctx.tile_decode_context.retained_channel_bytes()?;
    let retained_tier1_bytes = ctx.tile_decode_context.tier1_capacity_bytes()?;
    let retained_idwt_bytes = ctx.tile_decode_context.idwt_capacity_bytes()?;
    ctx.record_decode_complete(
        retained_component_bytes,
        retained_tier1_bytes,
        retained_idwt_bytes,
    );
    result
}

pub(crate) fn prepare_region_tiles<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
) -> Result<ParsedTiles<'a>> {
    // The parsed graph remains live for the complete region session. Start it
    // from a fresh lifetime-free workspace so its baseline is exact and later
    // stripe allocations can account the retained graph explicitly.
    ctx.release_reusable_allocations();
    let mut reader = BitReader::new(data);
    let tiles = tile::parse(&mut reader, header, retained_image_bytes)?;
    if tiles.is_empty() {
        bail!(TileError::Invalid);
    }
    Ok(tiles)
}

pub(crate) fn decode_preparsed_with_capacity_retry<'a>(
    header: &Header<'a>,
    retained_image_bytes: usize,
    tiles: &ParsedTiles<'a>,
    ctx: &mut DecoderContext<'a>,
) -> Result<()> {
    run_decode_with_capacity_retry(ctx, |ctx| {
        decode_preparsed(header, retained_image_bytes, tiles, ctx)
    })
}

fn decode_preparsed<'a>(
    header: &Header<'a>,
    retained_image_bytes: usize,
    tiles: &ParsedTiles<'a>,
    ctx: &mut DecoderContext<'a>,
) -> Result<()> {
    let reused_baseline = ctx.prepare_reused_decode_baseline(retained_image_bytes)?;
    let structural_workspace_bytes = reused_baseline
        .parser_live
        .checked_add(tiles.metadata_owner_bytes())
        .ok_or(ValidationError::ImageTooLarge)?;
    if structural_workspace_bytes > crate::DEFAULT_MAX_DECODE_BYTES {
        bail!(ValidationError::ImageTooLarge);
    }
    let profile_enabled = profile::profile_stages_enabled();
    decode_parsed_tiles(
        header,
        tiles,
        structural_workspace_bytes,
        reused_baseline,
        ctx,
        &mut None,
        profile_enabled,
        DecodeProfileTimings::default(),
        profile::profile_now(profile_enabled),
    )
}

fn decode<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
) -> Result<()> {
    let mut reused_baseline = ctx.prepare_reused_decode_baseline(retained_image_bytes)?;
    let profile_enabled = profile::profile_stages_enabled();
    let total_start = profile::profile_now(profile_enabled);
    let mut profile_timings = DecodeProfileTimings::default();
    let stage_start = profile::profile_now(profile_enabled);
    let mut reader = BitReader::new(data);
    let mut parsed_tiles = tile::parse(&mut reader, header, reused_baseline.parser_live);
    if reused_baseline.scratch_capacity == 0
        && reused_baseline.channel_capacity != 0
        && matches!(
            parsed_tiles.as_ref(),
            Err(error) if reuse::is_capacity_error(error)
        )
    {
        // Retained component buffers are an optimization, not a reason for an
        // otherwise valid decode to fail its aggregate allocation cap.
        ctx.discard_reused_channels();
        reused_baseline = reuse::ReusedDecodeBaseline {
            parser_live: retained_image_bytes,
            channel_capacity: 0,
            scratch_capacity: 0,
        };
        reader = BitReader::new(data);
        parsed_tiles = tile::parse(&mut reader, header, reused_baseline.parser_live);
    }
    let tiles = parsed_tiles?;
    profile_timings.parse_tiles_us += profile::elapsed_us(stage_start);

    decode_parsed_tiles(
        header,
        &tiles,
        tiles.structural_workspace_bytes(),
        reused_baseline,
        ctx,
        ht_decoder,
        profile_enabled,
        profile_timings,
        total_start,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "the parsed decode boundary keeps graph ownership, allocation baseline, adapter, and profiling state explicit"
)]
fn decode_parsed_tiles<'a>(
    header: &Header<'a>,
    tiles: &ParsedTiles<'a>,
    structural_workspace_bytes: usize,
    reused_baseline: reuse::ReusedDecodeBaseline,
    ctx: &mut DecoderContext<'a>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    profile_enabled: bool,
    mut profile_timings: DecodeProfileTimings,
    total_start: Option<profile::ProfileInstant>,
) -> Result<()> {
    if tiles.is_empty() {
        bail!(TileError::Invalid);
    }
    let retained_decode_baseline = ctx.reset(
        header,
        &tiles[0],
        structural_workspace_bytes,
        reused_baseline.channel_capacity,
    )?;
    let retained_decode_base_without_scratch = retained_decode_baseline
        .checked_sub(reused_baseline.scratch_capacity)
        .ok_or(ValidationError::ImageTooLarge)?;
    let cpu_decode_parallelism = ctx.cpu_decode_parallelism;
    let (tile_ctx, storage) = (&mut ctx.tile_decode_context, &mut ctx.storage);

    for tile in tiles.iter() {
        ltrace!(
            "tile {} rect [{},{} {}x{}]",
            tile.idx,
            tile.rect.x0,
            tile.rect.y0,
            tile.rect.width(),
            tile.rect.height(),
        );

        if let Some(output_region) = tile_ctx.output_region {
            if !tile_intersects_output_region(tile.rect, &header.size_data, output_region) {
                ltrace!("tile {} outside output region, skipped", tile.idx);
                continue;
            }
        }

        decode_tile(
            tile,
            header,
            progression_iterator(tile)?,
            tile_ctx,
            storage,
            ht_decoder,
            cpu_decode_parallelism,
            profile_enabled,
            &mut profile_timings,
            retained_decode_base_without_scratch,
        )?;
    }

    // Note that this assumes that either all tiles have MCT or none of them.
    // In theory, only some could have it... But hopefully no such cursed
    // images exist!
    if tiles[0].mct {
        let stage_start = profile::profile_now(profile_enabled);
        mct::apply_inverse(tile_ctx, &tiles[0].component_infos, header, ht_decoder)?;
        apply_sign_shift_after_mct(tile_ctx, &header.component_infos);
        profile_timings.mct_us += profile::elapsed_us(stage_start);
    }

    if profile_enabled {
        emit_decode_profile_row(tile_ctx, &profile_timings, total_start);
    }

    // The returned image only borrows channel data. Parsed tile and packet
    // graphs borrow the codestream and must be released. Tier-1 and IDWT
    // owners are lifetime-free and remain available to the next image.
    storage.release_all_allocations();

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OutputRegion {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl OutputRegion {
    pub(crate) fn from_tuple(region: (u32, u32, u32, u32)) -> Self {
        let (x, y, width, height) = region;
        Self {
            x,
            y,
            width,
            height,
        }
    }

    fn dimensions(self) -> (u32, u32) {
        (self.width, self.height)
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecodeDebugCounters {
    pub(crate) decoded_code_blocks: usize,
    pub(crate) skipped_code_blocks: usize,
    pub(crate) idwt_output_samples: usize,
    pub(crate) ht_phase_stats: ht_block_decode::HtBlockDecodeStats,
    #[cfg(all(test, feature = "parallel"))]
    pub(crate) ht_parallel_tasks: usize,
    #[cfg(all(test, feature = "parallel"))]
    pub(crate) ht_task_workspace_growths: usize,
    #[cfg(all(test, feature = "parallel"))]
    pub(crate) parallel_coefficients: ParallelCoefficientStats,
}

#[cfg(all(test, feature = "parallel"))]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParallelCoefficientStats {
    pub(crate) allocations: usize,
    pub(crate) allocated_bytes: usize,
    pub(crate) peak_live_bytes: usize,
    pub(crate) scatter_bytes: usize,
}

#[cfg(all(test, feature = "parallel"))]
impl ParallelCoefficientStats {
    fn record_allocation(&mut self, capacity: usize, live_bytes: &mut usize) {
        if capacity == 0 {
            return;
        }
        let bytes = capacity.saturating_mul(size_of::<f32>());
        self.allocations = self.allocations.saturating_add(1);
        self.allocated_bytes = self.allocated_bytes.saturating_add(bytes);
        *live_bytes = live_bytes.saturating_add(bytes);
        self.peak_live_bytes = self.peak_live_bytes.max(*live_bytes);
    }

    fn record_scatter(&mut self, bytes: usize) {
        self.scatter_bytes = self.scatter_bytes.saturating_add(bytes);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
fn decode_tile<'a, 'b>(
    tile: &'b Tile<'a>,
    header: &Header<'_>,
    progression_iterator: Box<dyn Iterator<Item = ProgressionData> + '_>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    cpu_decode_parallelism: CpuDecodeParallelism,
    profile_enabled: bool,
    profile_timings: &mut DecodeProfileTimings,
    retained_decode_baseline: usize,
) -> Result<()> {
    storage.reset_for_next_tile();
    let retained_scratch_bytes = tile_ctx.retained_scratch_bytes()?;
    let retained_decode_baseline = retained_decode_baseline
        .checked_add(retained_scratch_bytes)
        .ok_or(ValidationError::ImageTooLarge)?;
    if retained_decode_baseline > crate::DEFAULT_MAX_DECODE_BYTES {
        return Err(ValidationError::ImageTooLarge.into());
    }
    storage.exact_integer_decode = tile_requires_exact_integer_decode(tile);
    if storage.exact_integer_decode {
        validate_exact_integer_decode_tile(tile)?;
        if tile_ctx.output_region.is_some() {
            bail!(DecodingError::UnsupportedFeature(
                "25-38 bit region decode requires exact integer region IDWT support"
            ));
        }
    }

    // This is the method that orchestrates all steps.

    // First, we build the decompositions, including their sub-bands, precincts
    // and code blocks.
    let stage_start = profile::profile_now(profile_enabled);
    build::build(
        tile,
        storage,
        retained_decode_baseline,
        tile_ctx.output_region.is_some(),
        build::BuildWorkspace::DecodePixels {
            skipped_resolution_levels: header.skipped_resolution_levels,
        },
    )?;
    if let Some(output_region) = tile_ctx.output_region {
        storage.roi_plan = RoiPlan::build(tile, header, storage, output_region)?;
        if storage.roi_plan.is_some() {
            storage.coefficients.fill(0.0);
            if storage.exact_integer_decode {
                storage.coefficients_i64.fill(0);
            }
        } else {
            build::release_unused_roi_workspace(storage, tile.component_infos.len())?;
        }
    }
    profile_timings.build_us += profile::elapsed_us(stage_start);
    // Next, we parse the layers/segments for each code block.
    let stage_start = profile::profile_now(profile_enabled);
    segment::parse(tile, progression_iterator, header, storage)?;
    profile_timings.segment_us += profile::elapsed_us(stage_start);
    // We then decode the bitplanes of each code block, yielding the
    // (possibly dequantized) coefficients of each code block.
    let stage_start = profile::profile_now(profile_enabled);
    decode_component_tile_bit_planes_budgeted(
        tile,
        tile_ctx,
        storage,
        header,
        ht_decoder,
        cpu_decode_parallelism,
        profile_enabled,
    )?;
    profile_timings.codeblock_us += profile::elapsed_us(stage_start);

    // Unlike before, we interleave the apply_idwt and store stages
    // for each component tile so we can reuse allocations better.
    for (idx, component_info) in header.component_infos.iter().enumerate() {
        // Next, we apply the inverse discrete wavelet transform.
        let stage_start = profile::profile_now(profile_enabled);
        idwt::apply(
            storage,
            tile_ctx,
            idx,
            header,
            component_info.wavelet_transform(),
            ht_decoder,
        )?;
        profile_timings.idwt_us += profile::elapsed_us(stage_start);
        // Finally, we store the raw samples for the tile area in the correct
        // location. Note that in case we have MCT, we are not applying it yet.
        // It will be applied in the very end once all tiles have been processed.
        // The reason we do this is that applying MCT requires access to the
        // data from _all_ components. If we didn't defer this until the end
        // we would have to collect the IDWT outputs of all components before
        // applying it. By not applying MCT here, we can get away with doing
        // IDWT and store on a per-component basis. Thus, we only need to
        // store one IDWT output at a time, allowing for better reuse of
        // allocations.
        let stage_start = profile::profile_now(profile_enabled);
        store(tile, header, tile_ctx, component_info, idx, ht_decoder)?;
        profile_timings.store_us += profile::elapsed_us(stage_start);
    }

    Ok(())
}

pub(crate) fn decode_component_tile_bit_planes_budgeted<'a>(
    tile: &Tile<'a>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    header: &Header<'_>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    cpu_decode_parallelism: CpuDecodeParallelism,
    profile_enabled: bool,
) -> Result<()> {
    let tier1_workspace_bytes = tier1::prepare_tier1_workspace(tile, header, tile_ctx, storage)?;
    let decode_result = decode_component_tile_bit_planes(
        tile,
        tile_ctx,
        storage,
        header,
        ht_decoder,
        cpu_decode_parallelism,
        profile_enabled,
    );
    tier1::release_tier1_workspace(tile_ctx, storage, &tier1_workspace_bytes)?;
    decode_result
}

fn tile_requires_exact_integer_decode(tile: &Tile<'_>) -> bool {
    tile.component_infos
        .iter()
        .any(ComponentInfo::requires_exact_integer_decode)
}

fn validate_exact_integer_decode_tile(tile: &Tile<'_>) -> Result<()> {
    for component in &tile.component_infos {
        if component.size_info.precision > 38 {
            bail!(DecodingError::UnsupportedFeature(
                "JPEG 2000 Part 1 component precision is limited to 38 bits"
            ));
        }
        if component.wavelet_transform() != WaveletTransform::Reversible53 {
            bail!(DecodingError::UnsupportedFeature(
                "25-38 bit decode currently requires reversible 5/3 coding"
            ));
        }
        if component.quantization_info.quantization_style != QuantizationStyle::NoQuantization {
            bail!(DecodingError::UnsupportedFeature(
                "25-38 bit decode currently requires reversible no-quantization coding"
            ));
        }
    }
    Ok(())
}

#[derive(Default)]
#[expect(
    clippy::struct_field_names,
    reason = "the repeated _us suffix makes every profiling unit explicit"
)]
struct DecodeProfileTimings {
    parse_tiles_us: u128,
    build_us: u128,
    segment_us: u128,
    codeblock_us: u128,
    idwt_us: u128,
    store_us: u128,
    mct_us: u128,
}

#[cold]
#[inline(never)]
fn emit_decode_profile_row(
    tile_ctx: &TileDecodeContext,
    profile_timings: &DecodeProfileTimings,
    total_start: Option<profile::ProfileInstant>,
) {
    profile::emit_profile_row(
        "decode",
        "cpu",
        &[
            ("parse_tiles_us", profile_timings.parse_tiles_us),
            ("build_us", profile_timings.build_us),
            ("segment_us", profile_timings.segment_us),
            ("codeblock_us", profile_timings.codeblock_us),
            ("ht_blocks", tile_ctx.debug_counters.ht_phase_stats.blocks),
            (
                "ht_refinement_blocks",
                tile_ctx.debug_counters.ht_phase_stats.refinement_blocks,
            ),
            (
                "ht_cleanup_bytes",
                tile_ctx.debug_counters.ht_phase_stats.cleanup_bytes,
            ),
            (
                "ht_refinement_bytes",
                tile_ctx.debug_counters.ht_phase_stats.refinement_bytes,
            ),
            (
                "ht_cleanup_us",
                tile_ctx.debug_counters.ht_phase_stats.ht_cleanup_us,
            ),
            (
                "ht_mag_sgn_us",
                tile_ctx.debug_counters.ht_phase_stats.ht_mag_sgn_us,
            ),
            (
                "ht_sigma_us",
                tile_ctx.debug_counters.ht_phase_stats.ht_sigma_us,
            ),
            (
                "ht_sigprop_us",
                tile_ctx.debug_counters.ht_phase_stats.ht_sigprop_us,
            ),
            (
                "ht_magref_us",
                tile_ctx.debug_counters.ht_phase_stats.ht_magref_us,
            ),
            ("idwt_us", profile_timings.idwt_us),
            ("store_us", profile_timings.store_us),
            ("mct_us", profile_timings.mct_us),
            ("total_us", profile::elapsed_us(total_start)),
        ],
    );
}

/// All decompositions for a single tile.
#[derive(Clone)]
pub(crate) struct TileDecompositions {
    pub(crate) first_ll_sub_band: usize,
    pub(crate) decompositions: Range<usize>,
}

impl TileDecompositions {
    pub(crate) fn sub_band_iter(
        &self,
        resolution: u8,
        decompositions: &[Decomposition],
    ) -> SubBandIter {
        let indices = if resolution == 0 {
            [
                self.first_ll_sub_band,
                self.first_ll_sub_band,
                self.first_ll_sub_band,
            ]
        } else {
            decompositions[self.decompositions.clone()][resolution as usize - 1].sub_bands
        };

        SubBandIter {
            next_idx: 0,
            indices,
            resolution,
        }
    }
}

#[derive(Clone)]
pub(crate) struct SubBandIter {
    resolution: u8,
    next_idx: usize,
    indices: [usize; 3],
}

impl Iterator for SubBandIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let value = if self.resolution == 0 {
            if self.next_idx > 0 {
                None
            } else {
                Some(self.indices[0])
            }
        } else if self.next_idx >= self.indices.len() {
            None
        } else {
            Some(self.indices[self.next_idx])
        };

        self.next_idx += 1;

        value
    }
}

/// Owned decomposition workspace for one active tile.
///
/// Within an image, reset retains capacities and the next build counts them in
/// its live baseline. Crossing to a new image drops every retained allocation.
#[derive(Default)]
pub(crate) struct DecompositionStorage<'a> {
    pub(crate) segments: Vec<Segment<'a>>,
    pub(crate) layers: Vec<Layer>,
    pub(crate) code_blocks: Vec<CodeBlock>,
    pub(crate) precincts: Vec<Precinct>,
    pub(crate) tag_tree_nodes: Vec<TagNode>,
    pub(crate) coefficients: Vec<f32>,
    pub(crate) coefficients_i64: Vec<i64>,
    pub(crate) sub_bands: Vec<SubBand>,
    pub(crate) decompositions: Vec<Decomposition>,
    pub(crate) tile_decompositions: Vec<TileDecompositions>,
    pub(crate) roi_plan: Option<RoiPlan>,
    pub(crate) exact_integer_decode: bool,
    /// Planned non-segment live bytes for the active tile and future decode scratch.
    pub(crate) structural_workspace_bytes: usize,
    /// Allocation/cap failure raised from the option-based packet parser.
    pub(crate) packet_workspace_error: Option<DecodeError>,
}

impl DecompositionStorage<'_> {
    pub(crate) fn reset_for_next_tile(&mut self) {
        self.segments.clear();
        self.layers.clear();
        self.code_blocks.clear();
        self.precincts.clear();
        self.tag_tree_nodes.clear();
        self.coefficients.clear();
        self.coefficients_i64.clear();
        self.sub_bands.clear();
        self.decompositions.clear();
        self.tile_decompositions.clear();
        self.roi_plan = None;
        self.exact_integer_decode = false;
        self.structural_workspace_bytes = 0;
        self.packet_workspace_error = None;
    }

    pub(crate) fn release_all_allocations(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn retained_capacity_bytes(&self) -> Result<usize> {
        let mut bytes = 0_usize;
        include_capacity::<Segment<'_>>(&mut bytes, self.segments.capacity())?;
        include_capacity::<Layer>(&mut bytes, self.layers.capacity())?;
        include_capacity::<CodeBlock>(&mut bytes, self.code_blocks.capacity())?;
        include_capacity::<Precinct>(&mut bytes, self.precincts.capacity())?;
        include_capacity::<TagNode>(&mut bytes, self.tag_tree_nodes.capacity())?;
        include_capacity::<f32>(&mut bytes, self.coefficients.capacity())?;
        include_capacity::<i64>(&mut bytes, self.coefficients_i64.capacity())?;
        include_capacity::<SubBand>(&mut bytes, self.sub_bands.capacity())?;
        include_capacity::<Decomposition>(&mut bytes, self.decompositions.capacity())?;
        include_capacity::<TileDecompositions>(&mut bytes, self.tile_decompositions.capacity())?;
        Ok(bytes)
    }
}

fn include_capacity<T>(bytes: &mut usize, capacity: usize) -> Result<()> {
    let additional = capacity
        .checked_mul(size_of::<T>())
        .ok_or(ValidationError::ImageTooLarge)?;
    *bytes = bytes
        .checked_add(additional)
        .ok_or(ValidationError::ImageTooLarge)?;
    if *bytes > crate::DEFAULT_MAX_DECODE_BYTES {
        return Err(ValidationError::ImageTooLarge.into());
    }
    Ok(())
}

/// A reusable context used during the decoding of a single tile.
///
/// Some of the fields are temporary in nature and reset after moving on to the
/// next tile, some contain global state.
#[derive(Default)]
pub(crate) struct TileDecodeContext {
    /// A reusable buffer for the IDWT output.
    pub(crate) idwt_output: IDWTOutput,
    /// A scratch buffer used during IDWT.
    pub(crate) idwt_scratch_buffer: Vec<f32>,
    /// A scratch buffer used during exact reversible integer IDWT.
    pub(crate) idwt_scratch_buffer_i64: Vec<i64>,
    /// A reusable context for decoding code blocks.
    pub(crate) bit_plane_decode_context: BitPlaneDecodeContext,
    /// Reusable buffers for decoding bitplanes.
    pub(crate) bit_plane_decode_buffers: BitPlaneDecodeBuffers,
    /// A reusable context for decoding HTJ2K code blocks.
    pub(crate) ht_block_decode_context: HtBlockDecodeContext,
    #[cfg(feature = "parallel")]
    pub(in crate::j2c::decode) ht_task_workspaces: Vec<workspace::HtTaskWorkspace>,
    /// The raw, decoded samples for each channel.
    pub(crate) channel_data: Vec<ComponentData>,
    /// Optional output window for region-local decode storage.
    pub(crate) output_region: Option<OutputRegion>,
    /// Whether irreversible samples are being decoded for an integer output.
    pub(crate) round_irreversible_output: bool,
    /// Debug counters for tests and ROI instrumentation.
    pub(crate) debug_counters: DecodeDebugCounters,
}

impl TileDecodeContext {
    fn release_all_allocations(&mut self) {
        let output_region = self.output_region;
        let round_irreversible_output = self.round_irreversible_output;
        *self = Self::default();
        self.output_region = output_region;
        self.round_irreversible_output = round_irreversible_output;
    }

    fn release_tile_scratch_allocations(&mut self) {
        self.release_idwt_allocations();
        self.release_tier1_allocations();
    }

    fn release_tier1_allocations(&mut self) {
        self.bit_plane_decode_context = BitPlaneDecodeContext::default();
        self.bit_plane_decode_buffers = BitPlaneDecodeBuffers::default();
        self.ht_block_decode_context = HtBlockDecodeContext::default();
        #[cfg(feature = "parallel")]
        {
            self.ht_task_workspaces = Vec::new();
        }
    }

    pub(crate) fn serial_tier1_capacity_bytes(&self) -> Result<usize> {
        let classic_bytes = self.bit_plane_decode_context.allocated_bytes()?;
        let buffer_bytes = self.bit_plane_decode_buffers.allocated_bytes()?;
        let ht_bytes = self.ht_block_decode_context.allocated_bytes()?;
        classic_bytes
            .checked_add(buffer_bytes)
            .and_then(|bytes| bytes.checked_add(ht_bytes))
            .ok_or(ValidationError::ImageTooLarge.into())
    }

    pub(crate) fn tier1_capacity_bytes(&self) -> Result<usize> {
        let serial_bytes = self.serial_tier1_capacity_bytes()?;
        #[cfg(feature = "parallel")]
        {
            serial_bytes
                .checked_add(workspace::ht_task_workspace_bytes(
                    &self.ht_task_workspaces,
                )?)
                .ok_or(ValidationError::ImageTooLarge.into())
        }
        #[cfg(not(feature = "parallel"))]
        {
            Ok(serial_bytes)
        }
    }

    #[cfg(all(test, feature = "parallel"))]
    pub(crate) fn ht_parallel_workspace_count(&self) -> usize {
        self.ht_task_workspaces.len()
    }

    #[cfg(all(test, feature = "parallel"))]
    pub(crate) fn ht_parallel_workspace_bytes(&self) -> Result<usize> {
        workspace::ht_task_workspace_bytes(&self.ht_task_workspaces)
    }

    pub(crate) fn idwt_capacity_bytes(&self) -> Result<usize> {
        let mut bytes = 0_usize;
        include_capacity::<f32>(&mut bytes, self.idwt_output.coefficients.capacity())?;
        include_capacity::<i64>(&mut bytes, self.idwt_output.coefficients_i64.capacity())?;
        include_capacity::<f32>(&mut bytes, self.idwt_scratch_buffer.capacity())?;
        include_capacity::<i64>(&mut bytes, self.idwt_scratch_buffer_i64.capacity())?;
        Ok(bytes)
    }

    pub(crate) fn retained_scratch_bytes(&self) -> Result<usize> {
        self.tier1_capacity_bytes()?
            .checked_add(self.idwt_capacity_bytes()?)
            .ok_or(ValidationError::ImageTooLarge.into())
    }

    fn release_idwt_allocations(&mut self) {
        self.idwt_output = IDWTOutput::default();
        self.idwt_scratch_buffer = Vec::new();
        self.idwt_scratch_buffer_i64 = Vec::new();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        collect_classic_code_block_data, CodeBlock, DecodeAllocationBudget, DecoderContext,
        DecompositionStorage, Layer, Segment,
    };
    use crate::error::DecodingError;
    #[cfg(feature = "parallel")]
    use crate::j2c::build::{SubBand, SubBandType};
    use crate::j2c::codestream::CodeBlockStyle;
    use crate::j2c::rect::IntRect;
    use alloc::vec;

    fn classic_test_style() -> CodeBlockStyle {
        CodeBlockStyle {
            selective_arithmetic_coding_bypass: false,
            reset_context_probabilities: false,
            termination_on_each_pass: true,
            vertically_causal_context: false,
            segmentation_symbols: false,
            high_throughput_block_coding: false,
            mixed_block_coding: false,
        }
    }

    fn classic_test_code_block() -> CodeBlock {
        CodeBlock {
            rect: IntRect::from_xywh(0, 0, 1, 1),
            x_idx: 0,
            y_idx: 0,
            layers: 0..1,
            has_been_included: true,
            missing_bit_planes: 0,
            number_of_coding_passes: 3,
            ht_total_coding_passes: 0,
            ht_first_cleanup_pass: None,
            ht_selected_set: None,
            coding: Some(crate::j2c::build::CodeBlockCoding::Classic),
            l_block: 3,
            non_empty_layer_count: 1,
        }
    }

    #[test]
    fn cross_image_release_drops_reusable_decode_capacities() {
        let mut ctx = DecoderContext::default();
        ctx.storage.coefficients.reserve(64);
        ctx.storage.segments.reserve(16);
        ctx.tile_decode_context.idwt_scratch_buffer.reserve(64);
        ctx.tile_decode_context
            .bit_plane_decode_context
            .reserve_coefficients_for_test(64);

        ctx.release_reusable_allocations();

        assert_eq!(ctx.storage.coefficients.capacity(), 0);
        assert_eq!(ctx.storage.segments.capacity(), 0);
        assert_eq!(ctx.tile_decode_context.idwt_scratch_buffer.capacity(), 0);
        assert_eq!(
            ctx.tile_decode_context
                .bit_plane_decode_context
                .coefficient_capacity_for_test(),
            0
        );
    }

    #[test]
    fn next_tile_reset_preserves_storage_capacity_for_accounted_reuse() {
        let mut storage = DecompositionStorage::default();
        storage.coefficients.reserve(64);
        storage.layers.reserve(16);
        let coefficient_capacity = storage.coefficients.capacity();
        let layer_capacity = storage.layers.capacity();

        storage.reset_for_next_tile();

        assert_eq!(storage.coefficients.capacity(), coefficient_capacity);
        assert_eq!(storage.layers.capacity(), layer_capacity);
    }

    #[test]
    fn collect_classic_code_block_data_preserves_zero_length_segments() {
        let mut storage = DecompositionStorage::default();
        storage.layers.push(Layer {
            segments: Some(0..3),
        });
        storage.segments.push(Segment {
            idx: 0,
            coding_pases: 1,
            data_length: 1,
            data: &[0xAA],
        });
        storage.segments.push(Segment {
            idx: 1,
            coding_pases: 1,
            data_length: 0,
            data: &[],
        });
        storage.segments.push(Segment {
            idx: 2,
            coding_pases: 1,
            data_length: 1,
            data: &[0xBB],
        });

        let mut budget = DecodeAllocationBudget::for_storage(&storage).expect("budget baseline");
        let (combined_data, segments) = collect_classic_code_block_data(
            &classic_test_code_block(),
            &classic_test_style(),
            &storage,
            &mut budget,
        )
        .expect("collect classic segments");

        assert_eq!(combined_data, vec![0xAA, 0xBB]);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].data_offset, 0);
        assert_eq!(segments[0].data_length, 1);
        assert_eq!(segments[0].start_coding_pass, 0);
        assert_eq!(segments[0].end_coding_pass, 1);
        assert_eq!(segments[1].data_offset, 1);
        assert_eq!(segments[1].data_length, 0);
        assert_eq!(segments[1].start_coding_pass, 1);
        assert_eq!(segments[1].end_coding_pass, 2);
        assert_eq!(segments[2].data_offset, 1);
        assert_eq!(segments[2].data_length, 1);
        assert_eq!(segments[2].start_coding_pass, 2);
        assert_eq!(segments[2].end_coding_pass, 3);
    }

    #[test]
    fn collect_classic_code_block_data_rejects_non_contiguous_segment_indices() {
        let mut storage = DecompositionStorage::default();
        storage.layers.push(Layer {
            segments: Some(0..2),
        });
        storage.segments.push(Segment {
            idx: 0,
            coding_pases: 1,
            data_length: 1,
            data: &[0xAA],
        });
        storage.segments.push(Segment {
            idx: 2,
            coding_pases: 2,
            data_length: 1,
            data: &[0xBB],
        });

        let mut budget = DecodeAllocationBudget::for_storage(&storage).expect("budget baseline");
        let error = collect_classic_code_block_data(
            &classic_test_code_block(),
            &classic_test_style(),
            &storage,
            &mut budget,
        )
        .expect_err("non-contiguous segment indices must fail");

        assert_eq!(error, DecodingError::CodeBlockDecodeFailure.into());
    }

    #[cfg(feature = "parallel")]
    fn copyback_test_sub_band(width: u32, height: u32) -> (SubBand, DecompositionStorage<'static>) {
        let len = (width * height) as usize;
        let storage = DecompositionStorage {
            coefficients: vec![-1.0; len],
            ..DecompositionStorage::default()
        };
        let sub_band = SubBand {
            sub_band_type: SubBandType::LowLow,
            rect: IntRect::from_xywh(0, 0, width, height),
            precincts: 0..0,
            coefficients: 0..len,
        };
        (sub_band, storage)
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn decoded_classic_block_copyback_covers_full_block() {
        let (sub_band, mut storage) = copyback_test_sub_band(4, 3);
        let mut coefficient_stats = super::ParallelCoefficientStats::default();
        let block = super::DecodedClassicBlock {
            output_x: 0,
            output_y: 0,
            width: 4,
            height: 3,
            coefficients: (0..12)
                .map(|value| f32::from(u8::try_from(value).expect("test value fits u8")))
                .collect(),
        };

        super::copy_decoded_classic_blocks_to_sub_band(
            &[block],
            &sub_band,
            &mut storage,
            &mut coefficient_stats,
        )
        .expect("full classic block copyback");

        assert_eq!(
            storage.coefficients,
            (0..12)
                .map(|value| f32::from(u8::try_from(value).expect("test value fits u8")))
                .collect::<Vec<_>>()
        );
        assert_eq!(coefficient_stats.scatter_bytes, 12 * size_of::<f32>());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn decoded_ht_block_copyback_covers_partial_edge_block() {
        let (sub_band, mut storage) = copyback_test_sub_band(5, 3);
        let mut coefficient_stats = super::ParallelCoefficientStats::default();
        let block = super::DecodedHtBlock {
            output_x: 3,
            output_y: 1,
            width: 2,
            height: 2,
            coefficients: vec![1.0, 2.0, 3.0, 4.0],
        };

        super::copy_decoded_ht_blocks_to_sub_band(
            &[block],
            &sub_band,
            &mut storage,
            &mut coefficient_stats,
        )
        .expect("partial HT block copyback");

        assert_eq!(
            storage.coefficients,
            vec![
                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 2.0, -1.0, -1.0, -1.0, 3.0,
                4.0,
            ]
        );
        assert_eq!(coefficient_stats.scatter_bytes, 4 * size_of::<f32>());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn decoded_block_copyback_rejects_out_of_bounds_blocks() {
        let (sub_band, mut storage) = copyback_test_sub_band(5, 3);
        let mut coefficient_stats = super::ParallelCoefficientStats::default();
        let block = super::DecodedClassicBlock {
            output_x: 4,
            output_y: 1,
            width: 2,
            height: 1,
            coefficients: vec![1.0, 2.0],
        };

        let error = super::copy_decoded_classic_blocks_to_sub_band(
            &[block],
            &sub_band,
            &mut storage,
            &mut coefficient_stats,
        )
        .expect_err("out-of-bounds block must fail");

        assert_eq!(error, DecodingError::CodeBlockDecodeFailure.into());
        assert_eq!(coefficient_stats.scatter_bytes, 0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn auto_cpu_parallelism_enables_ht_sub_band_parallel_branch() {
        assert!(super::should_decode_ht_sub_band_in_parallel(
            super::CpuDecodeParallelism::Auto,
            16
        ));
    }

    #[test]
    fn serial_cpu_parallelism_disables_ht_sub_band_parallel_branch() {
        assert!(!super::should_decode_ht_sub_band_in_parallel(
            super::CpuDecodeParallelism::Serial,
            16
        ));
    }
}
