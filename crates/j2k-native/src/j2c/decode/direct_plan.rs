// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    add_roi_shift_to_bitplanes, bail, build, classic_decode_job_parameters,
    code_block_required_by_index, component_unsigned_level_shift, count_classic_code_blocks,
    count_ht_code_blocks, ht_block_decode, ht_code_block_has_decodable_passes,
    progression_iterator, segment, sub_band_decode_parameters, tile, BitReader, ColorError,
    ComponentInfo, ComponentTile, DecodeAllocationBudget, DecoderContext, DecodingError,
    DecompositionStorage, DirectPlanUnsupportedReason, Header, HtOwnedCodeBlockBatchJob,
    HtOwnedSubBandPlan, J2kDirectBandId, J2kDirectColorPlan, J2kDirectGrayscalePlan,
    J2kDirectGrayscaleStep, J2kDirectIdwtStep, J2kDirectStoreStep, J2kOwnedCodeBlockBatchJob,
    J2kOwnedSubBandPlan, J2kRect, J2kWaveletTransform, ResolutionTile, Result, RoiPlan, SubBand,
    SubBandDecodeParameters, Tile, ValidationError, Vec,
};
use crate::j2c::rect::IntRect;
use crate::j2c::roi::tile_intersects_output_region;
use crate::{
    HtCodeBlockPayloadRanges, J2kClassicCodeBlockPayload, J2kCodestreamRange, J2kDirectRgbaPlan,
    J2kReferencedClassicPlan, J2kReferencedHtj2kPlan, J2kReferencedPayloadRecordSpan,
    J2kReferencedTileGeometry, J2kReferencedTilePlan,
};

mod classic;
pub(super) use self::classic::{
    collect_classic_code_block_data, collect_referenced_classic_code_block_data,
};
mod color;
pub(crate) use self::color::{build_component_grid_color_plan, build_direct_color_plan};
mod referenced_color;
pub(crate) use self::referenced_color::{
    build_referenced_classic_color_plan, build_referenced_classic_rgba_plan,
    build_referenced_htj2k_color_plan, build_referenced_htj2k_rgba_plan,
};
mod referenced_grayscale;
pub(crate) use self::referenced_grayscale::{
    build_referenced_classic_grayscale_plan, build_referenced_htj2k_grayscale_plan,
};
mod storage;
use self::storage::build_component_plan_from_storage;
use self::storage::sub_band::{strip_classic_payload_owners, strip_grayscale_payload_owners};

#[derive(Clone, Copy)]
struct PayloadRangeOwner<'a> {
    encoded_input: &'a [u8],
    codestream: &'a [u8],
}

struct ClassicPayloadCollector<'a> {
    payloads: &'a mut Vec<J2kClassicCodeBlockPayload>,
    ranges: &'a mut Vec<J2kCodestreamRange>,
}

impl ClassicPayloadCollector<'_> {
    fn prepare(
        &mut self,
        payload_capacity: usize,
        range_capacity: usize,
        budget: &mut DecodeAllocationBudget,
    ) -> Result<()> {
        budget.reserve_new(self.payloads, payload_capacity)?;
        budget.reserve_new(self.ranges, range_capacity)
    }

    fn push_range(&mut self, range: J2kCodestreamRange) -> Result<()> {
        if self.ranges.len() == self.ranges.capacity() {
            bail!(DecodingError::HostAllocationFailed);
        }
        self.ranges.push(range);
        Ok(())
    }

    fn push_payload(&mut self, payload: J2kClassicCodeBlockPayload) -> Result<()> {
        if self.payloads.len() == self.payloads.capacity() {
            bail!(DecodingError::HostAllocationFailed);
        }
        self.payloads.push(payload);
        Ok(())
    }
}

pub(crate) fn build_direct_grayscale_plan<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
) -> Result<J2kDirectGrayscalePlan> {
    ctx.release_reusable_allocations();
    let result = build_direct_grayscale_plan_inner(
        data,
        data,
        header,
        retained_image_bytes,
        ctx,
        None,
        None,
    );
    ctx.release_reusable_allocations();
    result
}

fn build_direct_grayscale_plan_inner<'a>(
    data: &'a [u8],
    payload_range_owner: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
    ht_payloads: Option<&mut Vec<HtCodeBlockPayloadRanges>>,
    classic_payloads: Option<&mut ClassicPayloadCollector<'_>>,
) -> Result<J2kDirectGrayscalePlan> {
    let mut reader = BitReader::new(data);
    let tiles = tile::parse(&mut reader, header, retained_image_bytes)?;

    if tiles.len() != 1 {
        bail!(DecodingError::DirectPlanUnsupported(
            DirectPlanUnsupportedReason::GrayscaleSingleTileCodestream
        ));
    }

    let mut next_band_id = 0;
    let output_region = ctx.tile_decode_context.output_region;
    build_direct_grayscale_tile_plan(
        data,
        payload_range_owner,
        &tiles[0],
        header,
        tiles.structural_workspace_bytes(),
        ctx,
        &mut next_band_id,
        output_region,
        None,
        ht_payloads,
        classic_payloads,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "tile-local planning keeps the borrowed input, retained baseline, global band namespace, output region, and payload collector explicit"
)]
fn build_direct_grayscale_tile_plan<'a>(
    data: &'a [u8],
    payload_range_owner: &'a [u8],
    tile: &Tile<'a>,
    header: &Header<'a>,
    structural_workspace_bytes: usize,
    ctx: &mut DecoderContext<'a>,
    next_band_id: &mut J2kDirectBandId,
    decode_region: Option<super::OutputRegion>,
    store_region: Option<super::OutputRegion>,
    ht_payloads: Option<&mut Vec<HtCodeBlockPayloadRanges>>,
    mut classic_payloads: Option<&mut ClassicPayloadCollector<'_>>,
) -> Result<J2kDirectGrayscalePlan> {
    if tile.component_infos.len() != 1 {
        bail!(DecodingError::DirectPlanUnsupported(
            DirectPlanUnsupportedReason::GrayscaleSingleComponentCodestream
        ));
    }
    ctx.tile_decode_context.channel_data.clear();
    ctx.storage.reset_for_next_tile();

    build::build(
        tile,
        &mut ctx.storage,
        structural_workspace_bytes,
        decode_region.is_some(),
        build::BuildWorkspace::CoefficientsOnly,
    )?;
    if let Some(output_region) = decode_region {
        ctx.storage.roi_plan = RoiPlan::build(tile, header, &ctx.storage, output_region)?;
        if ctx.storage.roi_plan.is_none() {
            build::release_unused_roi_workspace(&mut ctx.storage, tile.component_infos.len())?;
        }
    }

    segment::parse(tile, progression_iterator(tile)?, header, &mut ctx.storage)?;

    let component_info = &tile.component_infos[0];
    let mut budget = DecodeAllocationBudget::for_storage(&ctx.storage)?;
    if let Some(collector) = classic_payloads.as_deref_mut() {
        collector.prepare(
            ctx.storage.code_blocks.len(),
            ctx.storage.segments.len(),
            &mut budget,
        )?;
    }
    build_component_plan_from_storage(
        PayloadRangeOwner {
            encoded_input: payload_range_owner,
            codestream: data,
        },
        tile,
        header,
        &ctx.storage,
        0,
        component_unsigned_level_shift(component_info),
        &mut budget,
        next_band_id,
        store_region,
        ht_payloads,
        classic_payloads,
        false,
    )
}

fn referenced_output_region(header: &Header<'_>, ctx: &DecoderContext<'_>) -> super::OutputRegion {
    ctx.tile_decode_context
        .output_region
        .unwrap_or(super::OutputRegion {
            x: 0,
            y: 0,
            width: header.size_data.image_width(),
            height: header.size_data.image_height(),
        })
}

fn output_region_rect(output_region: super::OutputRegion) -> J2kRect {
    J2kRect {
        x0: output_region.x,
        y0: output_region.y,
        x1: output_region.x.saturating_add(output_region.width),
        y1: output_region.y.saturating_add(output_region.height),
    }
}

fn validate_grayscale_tile(tile: &Tile<'_>) -> Result<()> {
    if tile.component_infos.len() != 1 {
        bail!(DecodingError::DirectPlanUnsupported(
            DirectPlanUnsupportedReason::GrayscaleSingleComponentCodestream
        ));
    }
    validate_unit_sampled_component(&tile.component_infos[0])
}

fn validate_color_tile<const COMPONENT_COUNT: usize>(
    tile: &Tile<'_>,
    component_count_error: DirectPlanUnsupportedReason,
) -> Result<()> {
    if tile.component_infos.len() != COMPONENT_COUNT {
        bail!(DecodingError::DirectPlanUnsupported(component_count_error));
    }
    for component_info in &tile.component_infos {
        validate_unit_sampled_component(component_info)?;
    }
    let transform = tile.component_infos[0].wavelet_transform();
    if tile.mct
        && (transform != tile.component_infos[1].wavelet_transform()
            || transform != tile.component_infos[2].wavelet_transform())
    {
        bail!(ColorError::Mct);
    }
    Ok(())
}

fn validate_unit_sampled_component(component_info: &ComponentInfo) -> Result<()> {
    if component_info.size_info.horizontal_resolution != 1
        || component_info.size_info.vertical_resolution != 1
    {
        bail!(DecodingError::DirectPlanUnsupported(
            DirectPlanUnsupportedReason::ComponentUnitSampled
        ));
    }
    Ok(())
}

fn tile_intersects_output(
    tile: &Tile<'_>,
    header: &Header<'_>,
    output_region: super::OutputRegion,
) -> Result<bool> {
    let component_info = tile
        .component_infos
        .first()
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    validate_unit_sampled_component(component_info)?;
    Ok(tile_intersects_output_region(
        tile.rect,
        &header.size_data,
        output_region,
    ))
}

fn payload_record_span(
    first_record: usize,
    record_count: usize,
) -> Result<J2kReferencedPayloadRecordSpan> {
    first_record
        .checked_add(record_count)
        .ok_or(ValidationError::ImageTooLarge)?;
    Ok(J2kReferencedPayloadRecordSpan {
        first_record,
        record_count,
    })
}

fn append_decode_elements<T>(destination: &mut Vec<T>, source: &mut Vec<T>) -> Result<()> {
    let target_len = destination
        .len()
        .checked_add(source.len())
        .ok_or(ValidationError::ImageTooLarge)?;
    crate::try_reserve_decode_elements(destination, target_len)?;
    destination.append(source);
    Ok(())
}

fn append_classic_payload_records(
    payloads: &mut Vec<J2kClassicCodeBlockPayload>,
    ranges: &mut Vec<J2kCodestreamRange>,
    tile_payloads: &mut Vec<J2kClassicCodeBlockPayload>,
    tile_ranges: &mut Vec<J2kCodestreamRange>,
) -> Result<()> {
    let range_base = ranges.len();
    for payload in tile_payloads.iter_mut() {
        payload.first_range = payload
            .first_range
            .checked_add(range_base)
            .ok_or(ValidationError::ImageTooLarge)?;
        let end_range = payload.end_range().ok_or(ValidationError::ImageTooLarge)?;
        let combined_range_len = range_base
            .checked_add(tile_ranges.len())
            .ok_or(ValidationError::ImageTooLarge)?;
        if end_range > combined_range_len {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
    }
    append_decode_elements(payloads, tile_payloads)?;
    append_decode_elements(ranges, tile_ranges)
}

fn grayscale_plan_rects(
    geometry: &J2kDirectGrayscalePlan,
    output_rect: J2kRect,
) -> Result<(J2kRect, J2kRect)> {
    let mut stores = geometry.steps.iter().filter_map(|step| match step {
        J2kDirectGrayscaleStep::Store(store) => Some(store),
        J2kDirectGrayscaleStep::ClassicSubBand(_)
        | J2kDirectGrayscaleStep::HtSubBand(_)
        | J2kDirectGrayscaleStep::Idwt(_) => None,
    });
    let store = stores.next().ok_or(DecodingError::CodeBlockDecodeFailure)?;
    if stores.next().is_some() || store.copy_width == 0 || store.copy_height == 0 {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    let destination_rect = J2kRect {
        x0: store.output_x,
        y0: store.output_y,
        x1: store.output_x.saturating_add(store.copy_width),
        y1: store.output_y.saturating_add(store.copy_height),
    };
    let decoded_rect = J2kRect {
        x0: output_rect.x0.saturating_add(destination_rect.x0),
        y0: output_rect.y0.saturating_add(destination_rect.y0),
        x1: output_rect.x0.saturating_add(destination_rect.x1),
        y1: output_rect.y0.saturating_add(destination_rect.y1),
    };
    Ok((decoded_rect, destination_rect))
}

fn color_plan_rects<const COMPONENT_COUNT: usize>(
    component_plans: &[J2kDirectGrayscalePlan],
    output_rect: J2kRect,
) -> Result<(J2kRect, J2kRect)> {
    if component_plans.len() != COMPONENT_COUNT {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    let expected = grayscale_plan_rects(&component_plans[0], output_rect)?;
    for component in &component_plans[1..] {
        if grayscale_plan_rects(component, output_rect)? != expected {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
    }
    Ok(expected)
}

fn validate_and_strip_referenced_payload_owners(
    component_plans: &mut [J2kDirectGrayscalePlan],
    payloads: &[HtCodeBlockPayloadRanges],
) -> Result<()> {
    let mut record_count = 0_usize;
    for component in component_plans {
        record_count = record_count
            .checked_add(strip_grayscale_payload_owners(
                component,
                payloads
                    .get(record_count..)
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?,
            )?)
            .ok_or(ValidationError::ImageTooLarge)?;
    }
    if record_count != payloads.len() {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    Ok(())
}

fn validate_and_strip_classic_payload_owners(
    component_plans: &mut [J2kDirectGrayscalePlan],
    payload_count: usize,
) -> Result<()> {
    let mut job_count = 0usize;
    for component in component_plans {
        job_count = job_count
            .checked_add(strip_classic_payload_owners(component)?)
            .ok_or(ValidationError::ImageTooLarge)?;
    }
    if job_count != payload_count {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    Ok(())
}
