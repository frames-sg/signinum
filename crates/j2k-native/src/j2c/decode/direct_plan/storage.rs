// SPDX-License-Identifier: MIT OR Apache-2.0

//! Component-plan assembly from parsed decomposition storage.

use super::{
    bail, count_classic_code_blocks, count_ht_code_blocks, ClassicPayloadCollector, ComponentInfo,
    ComponentTile, DecodeAllocationBudget, DecodingError, DecompositionStorage,
    DirectPlanUnsupportedReason, Header, HtCodeBlockPayloadRanges, IntRect, J2kDirectBandId,
    J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kDirectIdwtStep, J2kDirectStoreStep, J2kRect,
    J2kWaveletTransform, PayloadRangeOwner, ResolutionTile, Result, Tile, ValidationError, Vec,
};
use crate::j2c::decode::TileDecompositions;
use crate::j2c::roi::output_region_rect;

pub(super) mod sub_band;
use self::sub_band::build_grayscale_sub_band_steps;

#[expect(
    clippy::too_many_arguments,
    reason = "component planning keeps its validated storage, shared budget, output region, and optional payload collectors explicit"
)]
pub(super) fn build_component_plan_from_storage(
    payload_range_owner: PayloadRangeOwner<'_>,
    tile: &Tile<'_>,
    header: &Header<'_>,
    storage: &DecompositionStorage<'_>,
    component_idx: usize,
    store_addend: f32,
    budget: &mut DecodeAllocationBudget,
    next_band_id: &mut J2kDirectBandId,
    output_region: Option<super::super::OutputRegion>,
    ht_payloads: Option<&mut Vec<HtCodeBlockPayloadRanges>>,
    classic_payloads: Option<&mut ClassicPayloadCollector<'_>>,
    component_grid: bool,
) -> Result<J2kDirectGrayscalePlan> {
    let component_info = component_info(tile, component_idx, component_grid)?;
    let tile_decompositions = component_decompositions(storage, component_idx)?;
    let (step_capacity, active_decomposition_count) = component_step_capacity(
        component_info,
        tile_decompositions,
        storage,
        header.skipped_resolution_levels,
    )?;
    let mut steps = Vec::new();
    budget.reserve_new(&mut steps, step_capacity)?;
    let mut sub_band_ids = Vec::new();
    budget.resize_new(&mut sub_band_ids, storage.sub_bands.len(), None)?;

    append_sub_band_steps(
        payload_range_owner,
        component_info,
        tile_decompositions,
        storage,
        header,
        budget,
        next_band_id,
        &mut steps,
        &mut sub_band_ids,
        ht_payloads,
        classic_payloads,
    )?;
    let (final_rect, final_band_id) = append_idwt_steps(
        component_info,
        tile_decompositions,
        active_decomposition_count,
        storage,
        next_band_id,
        &sub_band_ids,
        &mut steps,
    )?;
    let mut store = component_store_step(
        tile,
        component_info,
        final_rect,
        final_band_id,
        header,
        output_region,
        store_addend,
    );
    if component_grid {
        // Eligibility guarantees an origin-zero, unreduced full image. Store
        // the complete native component plane; expansion belongs to the caller.
        store.source_x = 0;
        store.source_y = 0;
        store.copy_width = final_rect.width();
        store.copy_height = final_rect.height();
        store.output_width = final_rect.width();
        store.output_height = final_rect.height();
        store.output_x = 0;
        store.output_y = 0;
    }
    let dimensions = (store.output_width, store.output_height);
    steps.push(J2kDirectGrayscaleStep::Store(store));

    let sub_band_id_capacity = sub_band_ids.capacity();
    drop(sub_band_ids);
    budget.release_elements::<Option<J2kDirectBandId>>(sub_band_id_capacity)?;

    Ok(J2kDirectGrayscalePlan {
        dimensions,
        bit_depth: component_info.size_info.precision,
        steps,
    })
}

fn component_info<'a>(
    tile: &'a Tile<'_>,
    component_idx: usize,
    component_grid: bool,
) -> Result<&'a ComponentInfo> {
    let component_info =
        tile.component_infos
            .get(component_idx)
            .ok_or(DecodingError::DirectPlanUnsupported(
                DirectPlanUnsupportedReason::ComponentIndexOutOfRange,
            ))?;
    if !component_grid
        && (component_info.size_info.horizontal_resolution != 1
            || component_info.size_info.vertical_resolution != 1)
    {
        bail!(DecodingError::DirectPlanUnsupported(
            DirectPlanUnsupportedReason::ComponentUnitSampled
        ));
    }
    Ok(component_info)
}

fn component_decompositions<'a>(
    storage: &'a DecompositionStorage<'_>,
    component_idx: usize,
) -> Result<&'a TileDecompositions> {
    storage
        .tile_decompositions
        .get(component_idx)
        .ok_or(DecodingError::DirectPlanUnsupported(
            DirectPlanUnsupportedReason::ComponentDecompositionIndexOutOfRange,
        ))
        .map_err(Into::into)
}

fn component_step_capacity(
    component_info: &ComponentInfo,
    tile_decompositions: &TileDecompositions,
    storage: &DecompositionStorage<'_>,
    skipped_resolution_levels: u8,
) -> Result<(usize, usize)> {
    let active_decomposition_count = tile_decompositions
        .decompositions
        .len()
        .saturating_sub(skipped_resolution_levels as usize);
    let allows_mixed = component_info
        .coding_style
        .parameters
        .code_block_style
        .allows_mixed_block_coding();
    let sub_band_step_count = (0..component_info.num_resolution_levels()
        - skipped_resolution_levels)
        .try_fold(0_usize, |total, resolution| {
            tile_decompositions
                .sub_band_iter(resolution, &storage.decompositions)
                .try_fold(total, |total, sub_band_idx| -> Result<usize> {
                    let sub_band = &storage.sub_bands[sub_band_idx];
                    let step_count = if allows_mixed
                        && count_classic_code_blocks(sub_band_idx, sub_band, storage)? > 0
                        && count_ht_code_blocks(sub_band_idx, sub_band, storage)? > 0
                    {
                        2
                    } else {
                        1
                    };
                    total
                        .checked_add(step_count)
                        .ok_or(ValidationError::ImageTooLarge.into())
                })
        })?;
    let step_capacity = sub_band_step_count
        .checked_add(active_decomposition_count)
        .and_then(|count| count.checked_add(1))
        .ok_or(ValidationError::ImageTooLarge)?;
    Ok((step_capacity, active_decomposition_count))
}

#[expect(
    clippy::too_many_arguments,
    reason = "sub-band assembly advances the shared band namespace and both optional payload collectors"
)]
fn append_sub_band_steps(
    payload_range_owner: PayloadRangeOwner<'_>,
    component_info: &ComponentInfo,
    tile_decompositions: &TileDecompositions,
    storage: &DecompositionStorage<'_>,
    header: &Header<'_>,
    budget: &mut DecodeAllocationBudget,
    next_band_id: &mut J2kDirectBandId,
    steps: &mut Vec<J2kDirectGrayscaleStep>,
    sub_band_ids: &mut [Option<J2kDirectBandId>],
    mut ht_payloads: Option<&mut Vec<HtCodeBlockPayloadRanges>>,
    mut classic_payloads: Option<&mut ClassicPayloadCollector<'_>>,
) -> Result<()> {
    for resolution in 0..component_info.num_resolution_levels() - header.skipped_resolution_levels {
        for sub_band_idx in tile_decompositions.sub_band_iter(resolution, &storage.decompositions) {
            let sub_band_steps = build_grayscale_sub_band_steps(
                payload_range_owner,
                &storage.sub_bands[sub_band_idx],
                sub_band_idx,
                *next_band_id,
                resolution,
                component_info,
                storage,
                header,
                budget,
                ht_payloads.as_deref_mut(),
                classic_payloads.as_deref_mut(),
            )?;
            sub_band_ids[sub_band_idx] = Some(*next_band_id);
            *next_band_id = next_band_id
                .checked_add(1)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            steps.extend(sub_band_steps.into_iter().flatten());
        }
    }
    Ok(())
}

fn append_idwt_steps(
    component_info: &ComponentInfo,
    tile_decompositions: &TileDecompositions,
    active_decomposition_count: usize,
    storage: &DecompositionStorage<'_>,
    next_band_id: &mut J2kDirectBandId,
    sub_band_ids: &[Option<J2kDirectBandId>],
    steps: &mut Vec<J2kDirectGrayscaleStep>,
) -> Result<(IntRect, J2kDirectBandId)> {
    let mut ll_rect = storage.sub_bands[tile_decompositions.first_ll_sub_band].rect;
    let mut ll_band_id = sub_band_ids[tile_decompositions.first_ll_sub_band]
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let decompositions = &storage.decompositions[tile_decompositions.decompositions.clone()];

    for decomposition in &decompositions[..active_decomposition_count] {
        let [horizontal_band, vertical_band, diagonal_band] = decomposition.sub_bands;
        let output_band_id = *next_band_id;
        *next_band_id = next_band_id
            .checked_add(1)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        steps.push(J2kDirectGrayscaleStep::Idwt(J2kDirectIdwtStep {
            output_band_id,
            rect: J2kRect::from(decomposition.rect),
            transform: J2kWaveletTransform::from(component_info.wavelet_transform()),
            ll_band_id,
            ll: J2kRect::from(ll_rect),
            hl_band_id: sub_band_ids[horizontal_band]
                .ok_or(DecodingError::CodeBlockDecodeFailure)?,
            hl: J2kRect::from(storage.sub_bands[horizontal_band].rect),
            lh_band_id: sub_band_ids[vertical_band].ok_or(DecodingError::CodeBlockDecodeFailure)?,
            lh: J2kRect::from(storage.sub_bands[vertical_band].rect),
            hh_band_id: sub_band_ids[diagonal_band].ok_or(DecodingError::CodeBlockDecodeFailure)?,
            hh: J2kRect::from(storage.sub_bands[diagonal_band].rect),
        }));
        ll_rect = decomposition.rect;
        ll_band_id = output_band_id;
    }
    Ok((ll_rect, ll_band_id))
}

fn component_store_step(
    tile: &Tile<'_>,
    component_info: &ComponentInfo,
    input_rect: IntRect,
    input_band_id: J2kDirectBandId,
    header: &Header<'_>,
    output_region: Option<super::super::OutputRegion>,
    addend: f32,
) -> J2kDirectStoreStep {
    let component_tile = ComponentTile::new(tile, component_info);
    let resolution_tile = ResolutionTile::new(
        component_tile,
        component_info.num_resolution_levels() - 1 - header.skipped_resolution_levels,
    );
    let (
        source_x,
        source_y,
        copy_width,
        copy_height,
        output_width,
        output_height,
        output_x,
        output_y,
    ) = direct_store_geometry(input_rect, resolution_tile.rect, header, output_region);
    J2kDirectStoreStep {
        input_band_id,
        input_rect: J2kRect::from(input_rect),
        source_x,
        source_y,
        copy_width,
        copy_height,
        output_width,
        output_height,
        output_x,
        output_y,
        addend,
    }
}

#[expect(
    clippy::similar_names,
    reason = "paired source, destination, image, and region coordinates mirror the JPEG 2000 store boundary"
)]
fn direct_store_geometry(
    input_rect: IntRect,
    resolution_rect: IntRect,
    header: &Header<'_>,
    output_region: Option<super::super::OutputRegion>,
) -> (u32, u32, u32, u32, u32, u32, u32, u32) {
    let output_region = output_region.unwrap_or(super::super::OutputRegion {
        x: 0,
        y: 0,
        width: header.size_data.image_width(),
        height: header.size_data.image_height(),
    });
    let region = output_region_rect(&header.size_data, output_region);
    let region_x0 = region.x0;
    let region_y0 = region.y0;
    let region_x1 = region.x1;
    let region_y1 = region.y1;
    let copy_x0 = input_rect.x0.max(resolution_rect.x0).max(region_x0);
    let copy_y0 = input_rect.y0.max(resolution_rect.y0).max(region_y0);
    let copy_x1 = input_rect.x1.min(resolution_rect.x1).min(region_x1);
    let copy_y1 = input_rect.y1.min(resolution_rect.y1).min(region_y1);

    if copy_x0 >= copy_x1 || copy_y0 >= copy_y1 {
        return (0, 0, 0, 0, output_region.width, output_region.height, 0, 0);
    }

    (
        copy_x0 - input_rect.x0,
        copy_y0 - input_rect.y0,
        copy_x1 - copy_x0,
        copy_y1 - copy_y0,
        output_region.width,
        output_region.height,
        copy_x0 - region_x0,
        copy_y0 - region_y0,
    )
}
