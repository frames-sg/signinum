// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained RGB/RGBA plan assembly for HTJ2K and classic codestreams.

use super::{
    append_classic_payload_records, append_decode_elements,
    color::{build_direct_color_tile_components_plan, DirectColorComponentPlans},
    color_plan_rects, output_region_rect, payload_record_span, referenced_output_region, tile,
    tile_intersects_output, validate_and_strip_classic_payload_owners,
    validate_and_strip_referenced_payload_owners, validate_color_tile, BitReader,
    ClassicPayloadCollector, DecoderContext, DirectPlanUnsupportedReason, Header,
    HtCodeBlockPayloadRanges, J2kClassicCodeBlockPayload, J2kCodestreamRange, J2kDirectBandId,
    J2kRect, J2kReferencedClassicPlan, J2kReferencedHtj2kPlan, J2kReferencedTileGeometry,
    J2kReferencedTilePlan, Result, ValidationError, Vec,
};

pub(crate) fn build_referenced_htj2k_color_plan<'a>(
    data: &'a [u8],
    payload_range_owner: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
) -> Result<J2kReferencedHtj2kPlan> {
    build_referenced_htj2k_color_components_plan::<3>(
        data,
        payload_range_owner,
        header,
        retained_image_bytes,
        ctx,
        DirectPlanUnsupportedReason::ColorThreeComponentRgbCodestream,
        |plans| J2kReferencedTileGeometry::Color(plans.into_rgb()),
        |tiles, full_dimensions, output_rect, payloads| {
            J2kReferencedHtj2kPlan::color(tiles, full_dimensions, output_rect, payloads)
        },
    )
}

pub(crate) fn build_referenced_htj2k_rgba_plan<'a>(
    data: &'a [u8],
    payload_range_owner: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
) -> Result<J2kReferencedHtj2kPlan> {
    build_referenced_htj2k_color_components_plan::<4>(
        data,
        payload_range_owner,
        header,
        retained_image_bytes,
        ctx,
        DirectPlanUnsupportedReason::RgbaFourComponentRgbCodestream,
        |plans| J2kReferencedTileGeometry::Rgba(plans.into_rgba()),
        |tiles, full_dimensions, output_rect, payloads| {
            J2kReferencedHtj2kPlan::rgba(tiles, full_dimensions, output_rect, payloads)
        },
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "referenced color planning keeps the encoded owner, retained baseline, component contract, and typed plan constructor explicit"
)]
fn build_referenced_htj2k_color_components_plan<'a, const COMPONENT_COUNT: usize>(
    data: &'a [u8],
    payload_range_owner: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
    component_count_error: DirectPlanUnsupportedReason,
    into_geometry: impl Fn(DirectColorComponentPlans<COMPONENT_COUNT>) -> J2kReferencedTileGeometry,
    into_plan: impl FnOnce(
        Vec<J2kReferencedTilePlan>,
        (u32, u32),
        J2kRect,
        Vec<HtCodeBlockPayloadRanges>,
    ) -> J2kReferencedHtj2kPlan,
) -> Result<J2kReferencedHtj2kPlan> {
    ctx.release_reusable_allocations();
    let result = (|| {
        let mut reader = BitReader::new(data);
        let parsed_tiles = tile::parse(&mut reader, header, retained_image_bytes)?;
        let output_region = referenced_output_region(header, ctx);
        let output_rect = output_region_rect(output_region);
        let mut payloads = Vec::new();
        let mut tile_plans = Vec::new();
        crate::try_reserve_decode_elements(&mut tile_plans, parsed_tiles.len())?;
        let mut next_band_id: J2kDirectBandId = 0;

        for tile in parsed_tiles.iter() {
            validate_color_tile::<COMPONENT_COUNT>(tile, component_count_error)?;
            if !tile_intersects_output(tile, header, output_region)? {
                continue;
            }
            let first_record = payloads.len();
            let mut tile_payloads = Vec::new();
            let mut tile_classic_payloads = Vec::new();
            let mut tile_classic_ranges = Vec::new();
            let mut classic_collector = ClassicPayloadCollector {
                payloads: &mut tile_classic_payloads,
                ranges: &mut tile_classic_ranges,
            };
            let mut plans = build_direct_color_tile_components_plan::<COMPONENT_COUNT>(
                data,
                payload_range_owner,
                tile,
                header,
                parsed_tiles.structural_workspace_bytes(),
                ctx,
                component_count_error,
                &mut next_band_id,
                Some(output_region),
                Some(output_region),
                Some(&mut tile_payloads),
                Some(&mut classic_collector),
                false,
            )?;
            validate_and_strip_referenced_payload_owners(
                &mut plans.component_plans,
                &tile_payloads,
            )?;
            validate_and_strip_classic_payload_owners(
                &mut plans.component_plans,
                tile_classic_payloads.len(),
            )?;
            let record_count = tile_payloads.len();
            append_decode_elements(&mut payloads, &mut tile_payloads)?;
            let payload_records = payload_record_span(first_record, record_count)?;
            let (decoded_rect, destination_rect) =
                color_plan_rects::<COMPONENT_COUNT>(&plans.component_plans, output_rect)?;
            let geometry = into_geometry(plans);
            tile_plans.push(J2kReferencedTilePlan::new(
                usize::try_from(tile.idx).map_err(|_| ValidationError::ImageTooLarge)?,
                decoded_rect,
                destination_rect,
                payload_records,
                tile_classic_payloads,
                tile_classic_ranges,
                super::J2kWaveletTransform::from(tile.component_infos[0].wavelet_transform()),
                geometry,
            ));
        }

        let full_dimensions = (
            header.size_data.image_width(),
            header.size_data.image_height(),
        );
        Ok(into_plan(
            tile_plans,
            full_dimensions,
            output_rect,
            payloads,
        ))
    })();
    ctx.release_reusable_allocations();
    result
}

pub(crate) fn build_referenced_classic_color_plan<'a>(
    data: &'a [u8],
    payload_range_owner: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
) -> Result<J2kReferencedClassicPlan> {
    build_referenced_classic_color_components_plan::<3>(
        data,
        payload_range_owner,
        header,
        retained_image_bytes,
        ctx,
        DirectPlanUnsupportedReason::ColorThreeComponentRgbCodestream,
        |plans| J2kReferencedTileGeometry::Color(plans.into_rgb()),
        |tiles, full_dimensions, output_rect, payloads, ranges| {
            J2kReferencedClassicPlan::color(tiles, full_dimensions, output_rect, payloads, ranges)
        },
    )
}

pub(crate) fn build_referenced_classic_rgba_plan<'a>(
    data: &'a [u8],
    payload_range_owner: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
) -> Result<J2kReferencedClassicPlan> {
    build_referenced_classic_color_components_plan::<4>(
        data,
        payload_range_owner,
        header,
        retained_image_bytes,
        ctx,
        DirectPlanUnsupportedReason::RgbaFourComponentRgbCodestream,
        |plans| J2kReferencedTileGeometry::Rgba(plans.into_rgba()),
        |tiles, full_dimensions, output_rect, payloads, ranges| {
            J2kReferencedClassicPlan::rgba(tiles, full_dimensions, output_rect, payloads, ranges)
        },
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "referenced classic color planning keeps retained-byte ownership and typed geometry construction explicit"
)]
fn build_referenced_classic_color_components_plan<'a, const COMPONENT_COUNT: usize>(
    data: &'a [u8],
    payload_range_owner: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
    component_count_error: DirectPlanUnsupportedReason,
    into_geometry: impl Fn(DirectColorComponentPlans<COMPONENT_COUNT>) -> J2kReferencedTileGeometry,
    into_plan: impl FnOnce(
        Vec<J2kReferencedTilePlan>,
        (u32, u32),
        J2kRect,
        Vec<J2kClassicCodeBlockPayload>,
        Vec<J2kCodestreamRange>,
    ) -> J2kReferencedClassicPlan,
) -> Result<J2kReferencedClassicPlan> {
    ctx.release_reusable_allocations();
    let result = (|| {
        let mut reader = BitReader::new(data);
        let parsed_tiles = tile::parse(&mut reader, header, retained_image_bytes)?;
        let output_region = referenced_output_region(header, ctx);
        let output_rect = output_region_rect(output_region);
        let mut payloads = Vec::new();
        let mut ranges = Vec::new();
        let mut tile_plans = Vec::new();
        crate::try_reserve_decode_elements(&mut tile_plans, parsed_tiles.len())?;
        let mut next_band_id: J2kDirectBandId = 0;

        for tile in parsed_tiles.iter() {
            validate_color_tile::<COMPONENT_COUNT>(tile, component_count_error)?;
            if !tile_intersects_output(tile, header, output_region)? {
                continue;
            }
            let first_record = payloads.len();
            let mut tile_payloads = Vec::new();
            let mut tile_ranges = Vec::new();
            let mut collector = ClassicPayloadCollector {
                payloads: &mut tile_payloads,
                ranges: &mut tile_ranges,
            };
            let mut plans = build_direct_color_tile_components_plan::<COMPONENT_COUNT>(
                data,
                payload_range_owner,
                tile,
                header,
                parsed_tiles.structural_workspace_bytes(),
                ctx,
                component_count_error,
                &mut next_band_id,
                Some(output_region),
                Some(output_region),
                None,
                Some(&mut collector),
                false,
            )?;
            validate_and_strip_classic_payload_owners(
                &mut plans.component_plans,
                tile_payloads.len(),
            )?;
            let record_count = tile_payloads.len();
            append_classic_payload_records(
                &mut payloads,
                &mut ranges,
                &mut tile_payloads,
                &mut tile_ranges,
            )?;
            let payload_records = payload_record_span(first_record, record_count)?;
            let (decoded_rect, destination_rect) =
                color_plan_rects::<COMPONENT_COUNT>(&plans.component_plans, output_rect)?;
            tile_plans.push(J2kReferencedTilePlan::new(
                usize::try_from(tile.idx).map_err(|_| ValidationError::ImageTooLarge)?,
                decoded_rect,
                destination_rect,
                payload_records,
                Vec::new(),
                Vec::new(),
                super::J2kWaveletTransform::from(tile.component_infos[0].wavelet_transform()),
                into_geometry(plans),
            ));
        }

        Ok(into_plan(
            tile_plans,
            (
                header.size_data.image_width(),
                header.size_data.image_height(),
            ),
            output_rect,
            payloads,
            ranges,
        ))
    })();
    ctx.release_reusable_allocations();
    result
}
