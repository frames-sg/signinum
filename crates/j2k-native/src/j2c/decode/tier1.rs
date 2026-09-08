// SPDX-License-Identifier: MIT OR Apache-2.0

//! One active-tile budget for serial Tier-1 contexts and copied segment scratch.

use super::subband::code_block_required_by_index;
use super::{
    CodeBlock, DecodeAllocationBudget, DecompositionStorage, Header, Result, Tile,
    TileDecodeContext,
};
use crate::error::{DecodingError, ValidationError};
use crate::j2c::bitplane::classic_decode_workspace_bytes;
use crate::j2c::build::CodeBlockCoding;
use crate::j2c::ht_block_decode::ht_decode_workspace_bytes;
use core::mem::size_of;

pub(super) struct Tier1WorkspaceAccounting {
    retained_bytes: usize,
}

#[derive(Default)]
struct Tier1Requirements {
    classic_width: u32,
    classic_height: u32,
    classic_data_bytes: usize,
    classic_boundaries: usize,
    ht_width: u32,
    ht_height: u32,
    ht_data_bytes: usize,
    has_classic: bool,
    has_ht: bool,
}

impl Tier1Requirements {
    fn observe_classic(
        &mut self,
        code_block: &CodeBlock,
        storage: &DecompositionStorage<'_>,
    ) -> Result<()> {
        self.has_classic = true;
        self.classic_width = self.classic_width.max(code_block.rect.width());
        self.classic_height = self.classic_height.max(code_block.rect.height());

        let mut data_bytes = 0usize;
        let mut segment_count = 0usize;
        let layers = storage
            .layers
            .get(code_block.layers.clone())
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        for layer in layers {
            let Some(segment_range) = layer.segments.clone() else {
                continue;
            };
            let segments = storage
                .segments
                .get(segment_range)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            for segment in segments {
                data_bytes = data_bytes
                    .checked_add(segment.data.len())
                    .ok_or(ValidationError::ImageTooLarge)?;
                segment_count = segment_count
                    .checked_add(1)
                    .ok_or(ValidationError::ImageTooLarge)?;
            }
        }
        let boundaries = segment_count
            .checked_add(2)
            .ok_or(ValidationError::ImageTooLarge)?;
        self.classic_data_bytes = self.classic_data_bytes.max(data_bytes);
        self.classic_boundaries = self.classic_boundaries.max(boundaries);
        Ok(())
    }

    fn observe_ht(
        &mut self,
        code_block: &CodeBlock,
        storage: &DecompositionStorage<'_>,
    ) -> Result<()> {
        self.has_ht = true;
        self.ht_width = self.ht_width.max(code_block.rect.width());
        self.ht_height = self.ht_height.max(code_block.rect.height());
        if code_block.number_of_coding_passes != 0 {
            let (cleanup_bytes, refinement_bytes) =
                crate::j2c::ht_block_decode::selected_code_block_segment_lengths(
                    code_block, storage,
                )?;
            self.ht_data_bytes = self.ht_data_bytes.max(
                cleanup_bytes
                    .checked_add(refinement_bytes)
                    .ok_or(ValidationError::ImageTooLarge)?,
            );
        }
        Ok(())
    }

    fn logical_bytes(&self) -> Result<usize> {
        let mut bytes = 0usize;
        if self.has_classic {
            bytes = bytes
                .checked_add(classic_decode_workspace_bytes(
                    self.classic_width,
                    self.classic_height,
                )?)
                .ok_or(ValidationError::ImageTooLarge)?;
            include_elements::<u8>(&mut bytes, self.classic_data_bytes)?;
            include_elements::<usize>(&mut bytes, self.classic_boundaries)?;
            include_elements::<u8>(&mut bytes, self.classic_boundaries)?;
        }
        if self.has_ht {
            bytes = bytes
                .checked_add(ht_decode_workspace_bytes(self.ht_width, self.ht_height)?)
                .ok_or(ValidationError::ImageTooLarge)?;
            include_elements::<u8>(&mut bytes, self.ht_data_bytes)?;
        }
        Ok(bytes)
    }
}

pub(super) fn prepare_tier1_workspace(
    tile: &Tile<'_>,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'_>,
) -> Result<Tier1WorkspaceAccounting> {
    let requirements = collect_requirements(tile, header, storage)?;
    let planned_bytes = requirements.logical_bytes()?;
    let retained_bytes = tile_ctx.serial_tier1_capacity_bytes()?;
    let mut budget = DecodeAllocationBudget::for_storage(storage)?;
    budget.include_bytes(planned_bytes)?;

    let prepared = (|| {
        if requirements.has_classic {
            tile_ctx
                .bit_plane_decode_context
                .prepare(requirements.classic_width, requirements.classic_height)?;
            tile_ctx.bit_plane_decode_buffers.prepare(
                requirements.classic_data_bytes,
                requirements.classic_boundaries,
            )?;
        }
        if requirements.has_ht {
            tile_ctx.ht_block_decode_context.prepare(
                requirements.ht_width,
                requirements.ht_height,
                requirements.ht_data_bytes,
            )?;
        }

        let actual_bytes = tile_ctx.serial_tier1_capacity_bytes()?;
        if actual_bytes > planned_bytes {
            budget.include_bytes(actual_bytes - planned_bytes)?;
        }
        storage.structural_workspace_bytes = storage
            .structural_workspace_bytes
            .checked_add(actual_bytes)
            .ok_or(ValidationError::ImageTooLarge)?;
        Ok(Tier1WorkspaceAccounting { retained_bytes })
    })();

    if prepared.is_err() {
        tile_ctx.release_tier1_allocations();
    }
    prepared
}

pub(super) fn release_tier1_workspace(
    _tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'_>,
    accounting: &Tier1WorkspaceAccounting,
) -> Result<()> {
    // The active structural baseline already included retained serial and
    // parallel Tier-1 owners. Preparation charged only the complete new serial
    // owner, so remove only its now-duplicated retained capacity.
    storage.structural_workspace_bytes = storage
        .structural_workspace_bytes
        .checked_sub(accounting.retained_bytes)
        .ok_or(ValidationError::ImageTooLarge)?;
    Ok(())
}

fn collect_requirements(
    tile: &Tile<'_>,
    header: &Header<'_>,
    storage: &DecompositionStorage<'_>,
) -> Result<Tier1Requirements> {
    let mut requirements = Tier1Requirements::default();
    for component_idx in 0..tile.component_infos.len() {
        let tile_decompositions = storage
            .tile_decompositions
            .get(component_idx)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let decompositions = storage
            .decompositions
            .get(tile_decompositions.decompositions.clone())
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let active_decompositions = decompositions
            .len()
            .saturating_sub(header.skipped_resolution_levels as usize);
        observe_sub_band(
            tile_decompositions.first_ll_sub_band,
            storage,
            &mut requirements,
        )?;
        for decomposition in decompositions.iter().take(active_decompositions) {
            for &sub_band_idx in &decomposition.sub_bands {
                observe_sub_band(sub_band_idx, storage, &mut requirements)?;
            }
        }
    }
    Ok(requirements)
}

fn observe_sub_band(
    sub_band_idx: usize,
    storage: &DecompositionStorage<'_>,
    requirements: &mut Tier1Requirements,
) -> Result<()> {
    let sub_band = storage
        .sub_bands
        .get(sub_band_idx)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let precincts = storage
        .precincts
        .get(sub_band.precincts.clone())
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    for precinct in precincts {
        let code_blocks = storage
            .code_blocks
            .get(precinct.code_blocks.clone())
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        for code_block in code_blocks {
            if !code_block_required_by_index(storage, sub_band_idx, code_block) {
                continue;
            }
            match code_block.coding {
                Some(CodeBlockCoding::Classic) => {
                    requirements.observe_classic(code_block, storage)?;
                }
                Some(CodeBlockCoding::HighThroughput) => {
                    requirements.observe_ht(code_block, storage)?;
                }
                None if code_block.number_of_coding_passes == 0 => {}
                None => return Err(DecodingError::CodeBlockDecodeFailure.into()),
            }
        }
    }
    Ok(())
}

fn include_elements<T>(bytes: &mut usize, count: usize) -> Result<()> {
    let additional = count
        .checked_mul(size_of::<T>())
        .ok_or(ValidationError::ImageTooLarge)?;
    *bytes = bytes
        .checked_add(additional)
        .ok_or(ValidationError::ImageTooLarge)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Tier1Requirements;
    use crate::j2c::bitplane::classic_decode_workspace_bytes;
    use core::mem::size_of;

    #[test]
    fn classic_tier1_plan_includes_payload_and_both_boundary_arrays() {
        let requirements = Tier1Requirements {
            classic_width: 4,
            classic_height: 3,
            classic_data_bytes: 17,
            classic_boundaries: 5,
            has_classic: true,
            ..Tier1Requirements::default()
        };
        let expected = classic_decode_workspace_bytes(4, 3).expect("workspace")
            + 17
            + 5 * size_of::<usize>()
            + 5;
        assert_eq!(
            requirements.logical_bytes().expect("logical bytes"),
            expected
        );
    }
}
