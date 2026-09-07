// SPDX-License-Identifier: MIT OR Apache-2.0

//! Direct RGB/RGBA plan construction.

use super::{
    bail, build, build_component_plan_from_storage, component_unsigned_level_shift,
    progression_iterator, segment, tile, BitReader, ClassicPayloadCollector, ColorError,
    DecodeAllocationBudget, DecoderContext, DecodingError, DirectPlanUnsupportedReason, Header,
    HtCodeBlockPayloadRanges, J2kDirectBandId, J2kDirectColorPlan, J2kDirectGrayscalePlan,
    J2kDirectRgbaPlan, J2kWaveletTransform, PayloadRangeOwner, Result, RoiPlan, Tile, Vec,
};

pub(crate) fn build_direct_color_plan<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
) -> Result<J2kDirectColorPlan> {
    ctx.release_reusable_allocations();
    let result = build_direct_color_components_plan_inner::<3>(
        data,
        data,
        header,
        retained_image_bytes,
        ctx,
        DirectPlanUnsupportedReason::ColorThreeComponentRgbCodestream,
        None,
        None,
        false,
    )
    .map(DirectColorComponentPlans::into_rgb);
    ctx.release_reusable_allocations();
    result
}

pub(crate) fn build_component_grid_color_plan<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
) -> Result<(J2kDirectColorPlan, [(u8, u8); 3])> {
    ctx.release_reusable_allocations();
    let result = build_direct_color_components_plan_inner::<3>(
        data,
        data,
        header,
        retained_image_bytes,
        ctx,
        DirectPlanUnsupportedReason::ColorThreeComponentRgbCodestream,
        None,
        None,
        true,
    )
    .map(|plans| {
        let sampling = plans.sampling;
        (plans.into_rgb(), sampling)
    });
    ctx.release_reusable_allocations();
    result
}

pub(super) struct DirectColorComponentPlans<const COMPONENT_COUNT: usize> {
    dimensions: (u32, u32),
    bit_depths: [u8; COMPONENT_COUNT],
    mct: bool,
    transform: J2kWaveletTransform,
    sampling: [(u8, u8); COMPONENT_COUNT],
    pub(super) component_plans: Vec<J2kDirectGrayscalePlan>,
}

impl DirectColorComponentPlans<3> {
    pub(super) fn into_rgb(self) -> J2kDirectColorPlan {
        J2kDirectColorPlan {
            dimensions: self.dimensions,
            bit_depths: self.bit_depths,
            mct: self.mct,
            transform: self.transform,
            component_plans: self.component_plans,
        }
    }
}

impl DirectColorComponentPlans<4> {
    pub(super) fn into_rgba(self) -> J2kDirectRgbaPlan {
        J2kDirectRgbaPlan {
            dimensions: self.dimensions,
            bit_depths: self.bit_depths,
            mct: self.mct,
            transform: self.transform,
            component_plans: self.component_plans,
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "color planning keeps the borrowed codestream, decode context, component contract, and mutually exclusive referenced payload collectors explicit"
)]
fn build_direct_color_components_plan_inner<'a, const COMPONENT_COUNT: usize>(
    data: &'a [u8],
    payload_range_owner: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
    component_count_error: DirectPlanUnsupportedReason,
    ht_payloads: Option<&mut Vec<HtCodeBlockPayloadRanges>>,
    classic_payloads: Option<&mut ClassicPayloadCollector<'_>>,
    component_grid: bool,
) -> Result<DirectColorComponentPlans<COMPONENT_COUNT>> {
    let mut reader = BitReader::new(data);
    let tiles = tile::parse(&mut reader, header, retained_image_bytes)?;

    if tiles.len() != 1 {
        bail!(DecodingError::DirectPlanUnsupported(
            DirectPlanUnsupportedReason::ColorSingleTileCodestream
        ));
    }

    let mut next_band_id = 0;
    let output_region = ctx.tile_decode_context.output_region;
    build_direct_color_tile_components_plan::<COMPONENT_COUNT>(
        data,
        payload_range_owner,
        &tiles[0],
        header,
        tiles.structural_workspace_bytes(),
        ctx,
        component_count_error,
        &mut next_band_id,
        output_region,
        None,
        ht_payloads,
        classic_payloads,
        component_grid,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "tile-local color planning keeps the borrowed input, component contract, global band namespace, output region, and payload collector explicit"
)]
pub(super) fn build_direct_color_tile_components_plan<'a, const COMPONENT_COUNT: usize>(
    data: &'a [u8],
    payload_range_owner: &'a [u8],
    tile: &Tile<'a>,
    header: &Header<'a>,
    structural_workspace_bytes: usize,
    ctx: &mut DecoderContext<'a>,
    component_count_error: DirectPlanUnsupportedReason,
    next_band_id: &mut J2kDirectBandId,
    decode_region: Option<super::super::OutputRegion>,
    store_region: Option<super::super::OutputRegion>,
    mut ht_payloads: Option<&mut Vec<HtCodeBlockPayloadRanges>>,
    mut classic_payloads: Option<&mut ClassicPayloadCollector<'_>>,
    component_grid: bool,
) -> Result<DirectColorComponentPlans<COMPONENT_COUNT>> {
    if tile.component_infos.len() != COMPONENT_COUNT {
        bail!(DecodingError::DirectPlanUnsupported(component_count_error));
    }
    if component_grid
        && (tile.mct
            || tile.component_infos.iter().any(|c| c.size_info.signed)
            || header.skipped_resolution_levels != 0
            || header.size_data.image_area_x_offset != 0
            || header.size_data.image_area_y_offset != 0
            || decode_region.is_some()
            || store_region.is_some())
    {
        bail!(DecodingError::DirectPlanUnsupported(
            DirectPlanUnsupportedReason::ComponentGridFullImage
        ));
    }
    let transform = tile.component_infos[0].wavelet_transform();
    if tile.mct
        && (transform != tile.component_infos[1].wavelet_transform()
            || transform != tile.component_infos[2].wavelet_transform())
    {
        bail!(ColorError::Mct);
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

    let mut bit_depths = [0_u8; COMPONENT_COUNT];
    let mut sampling = [(1, 1); COMPONENT_COUNT];
    let mut budget = DecodeAllocationBudget::for_storage(&ctx.storage)?;
    if let Some(collector) = classic_payloads.as_deref_mut() {
        collector.prepare(
            ctx.storage.code_blocks.len(),
            ctx.storage.segments.len(),
            &mut budget,
        )?;
    }
    let mut component_plans = Vec::new();
    budget.reserve_new(&mut component_plans, bit_depths.len())?;
    for (component_idx, bit_depth) in bit_depths.iter_mut().enumerate() {
        let component_info = &tile.component_infos[component_idx];
        *bit_depth = component_info.size_info.precision;
        sampling[component_idx] = (
            component_info.size_info.horizontal_resolution,
            component_info.size_info.vertical_resolution,
        );
        let addend = if tile.mct && component_idx < 3 {
            0.0
        } else {
            component_unsigned_level_shift(component_info)
        };
        component_plans.push(build_component_plan_from_storage(
            PayloadRangeOwner {
                encoded_input: payload_range_owner,
                codestream: data,
            },
            tile,
            header,
            &ctx.storage,
            component_idx,
            addend,
            &mut budget,
            next_band_id,
            store_region,
            ht_payloads.as_deref_mut(),
            classic_payloads.as_deref_mut(),
            component_grid,
        )?);
    }

    let dimensions = store_region.map_or_else(
        || {
            (
                header.size_data.image_width(),
                header.size_data.image_height(),
            )
        },
        |region| (region.width, region.height),
    );
    Ok(DirectColorComponentPlans {
        dimensions,
        bit_depths,
        sampling,
        mct: tile.mct,
        transform: J2kWaveletTransform::from(transform),
        component_plans,
    })
}
