// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    add_roi_shift_to_bitplanes, apply_roi_maxshift_inverse_f32, apply_roi_maxshift_inverse_i32,
    apply_roi_maxshift_inverse_i64, bitplane, classic_decode_job_parameters,
    collect_classic_code_block_data, decode_j2k_code_block_scalar_with_workspace,
    decode_j2k_code_block_scalar_with_workspace_midpoint, ht_block_decode,
    ht_code_block_has_decodable_passes, sub_band_decode_parameters, CodeBlock, ComponentInfo,
    CpuDecodeParallelism, DecodeAllocationBudget, DecodingError, DecompositionStorage, Header,
    HtCodeBlockBatchJob, HtCodeBlockDecodeJob, HtCodeBlockDecoder, HtSubBandDecodeJob,
    J2kCodeBlockBatchJob, J2kCodeBlockDecodeJob, J2kCodeBlockDecodeWorkspace, J2kSubBandDecodeJob,
    Result, SubBand, SubBandDecodeParameters, Tile, TileDecodeContext, Vec, MAX_BITPLANE_COUNT,
};

mod classic;
mod ht;
#[cfg(feature = "parallel")]
mod parallel;
mod pending;
use self::classic::decode_sub_band_classic_blocks;
use self::ht::{decode_sub_band_ht_blocks, decode_sub_band_ht_blocks_i64};
#[cfg(all(feature = "parallel", not(test)))]
use self::parallel::{copy_decoded_classic_blocks_to_sub_band, copy_decoded_ht_blocks_to_sub_band};
#[cfg(all(test, feature = "parallel"))]
pub(super) use self::parallel::{
    copy_decoded_classic_blocks_to_sub_band, copy_decoded_ht_blocks_to_sub_band, DecodedBlock,
};
#[cfg(feature = "parallel")]
use self::parallel::{
    decode_classic_sub_band_blocks_parallel, decode_ht_sub_band_blocks_parallel,
    release_coefficient_slab, ClassicParallelParameters, HtParallelParameters,
};
use self::pending::{collect_pending_classic_blocks, collect_pending_ht_blocks};
pub(in crate::j2c::decode) use self::pending::{count_classic_code_blocks, count_ht_code_blocks};

pub(crate) fn decode_component_tile_bit_planes<'a>(
    tile: &Tile<'a>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    header: &Header<'_>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    cpu_decode_parallelism: CpuDecodeParallelism,
    profile_enabled: bool,
) -> Result<()> {
    for (tile_decompositions_idx, component_info) in tile.component_infos.iter().enumerate() {
        // Only decode the resolution levels we actually care about.
        for resolution in
            0..component_info.num_resolution_levels() - header.skipped_resolution_levels
        {
            let tile_composition = &storage.tile_decompositions[tile_decompositions_idx];
            let sub_band_iter = tile_composition.sub_band_iter(resolution, &storage.decompositions);

            for sub_band_idx in sub_band_iter {
                decode_sub_band_bitplanes(
                    sub_band_idx,
                    resolution,
                    component_info,
                    tile_ctx,
                    storage,
                    header,
                    ht_decoder,
                    cpu_decode_parallelism,
                    profile_enabled,
                )?;
            }
        }
    }

    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
fn decode_sub_band_bitplanes(
    sub_band_idx: usize,
    resolution: u8,
    component_info: &ComponentInfo,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'_>,
    header: &Header<'_>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    cpu_decode_parallelism: CpuDecodeParallelism,
    profile_enabled: bool,
) -> Result<()> {
    let sub_band = storage.sub_bands[sub_band_idx].clone();
    let SubBandDecodeParameters {
        dequantization_step,
        irreversible_midpoint,
        num_bitplanes,
    } = sub_band_decode_parameters(&sub_band, resolution, component_info)?;
    let style = component_info.coding_style.parameters.code_block_style;

    if style.allows_mixed_block_coding() {
        decode_sub_band_classic_blocks(
            sub_band_idx,
            &sub_band,
            component_info,
            tile_ctx,
            storage,
            header,
            ht_decoder,
            cpu_decode_parallelism,
            num_bitplanes,
            dequantization_step,
            irreversible_midpoint,
        )?;
        if storage.exact_integer_decode {
            return decode_sub_band_ht_blocks_i64(
                sub_band_idx,
                &sub_band,
                component_info,
                tile_ctx,
                storage,
                header,
                num_bitplanes,
                profile_enabled,
            );
        }
        return decode_sub_band_ht_blocks(
            sub_band_idx,
            &sub_band,
            component_info,
            tile_ctx,
            storage,
            header,
            ht_decoder,
            cpu_decode_parallelism,
            num_bitplanes,
            dequantization_step,
            irreversible_midpoint,
            profile_enabled,
        );
    }

    if style.uses_high_throughput_block_coding() {
        if storage.exact_integer_decode {
            decode_sub_band_ht_blocks_i64(
                sub_band_idx,
                &sub_band,
                component_info,
                tile_ctx,
                storage,
                header,
                num_bitplanes,
                profile_enabled,
            )?;
            return Ok(());
        }
        decode_sub_band_ht_blocks(
            sub_band_idx,
            &sub_band,
            component_info,
            tile_ctx,
            storage,
            header,
            ht_decoder,
            cpu_decode_parallelism,
            num_bitplanes,
            dequantization_step,
            irreversible_midpoint,
            profile_enabled,
        )?;
        return Ok(());
    }
    decode_sub_band_classic_blocks(
        sub_band_idx,
        &sub_band,
        component_info,
        tile_ctx,
        storage,
        header,
        ht_decoder,
        cpu_decode_parallelism,
        num_bitplanes,
        dequantization_step,
        irreversible_midpoint,
    )
}

pub(super) fn code_block_required_by_index(
    storage: &DecompositionStorage<'_>,
    sub_band_idx: usize,
    code_block: &CodeBlock,
) -> bool {
    storage
        .roi_plan
        .as_ref()
        .is_none_or(|plan| plan.code_block_required(sub_band_idx, code_block.rect))
}

pub(crate) fn should_decode_classic_sub_band_in_parallel(
    parallelism: CpuDecodeParallelism,
    code_block_count: usize,
) -> bool {
    cfg!(feature = "parallel") && parallelism == CpuDecodeParallelism::Auto && code_block_count >= 4
}

pub(crate) fn should_decode_ht_sub_band_in_parallel(
    parallelism: CpuDecodeParallelism,
    code_block_count: usize,
) -> bool {
    parallelism == CpuDecodeParallelism::Auto && code_block_count >= 4 && {
        // One worker cannot overlap HT entropy jobs; keep its existing serial
        // workspace instead of collecting and scattering parallel staging.
        #[cfg(feature = "parallel")]
        {
            rayon::current_num_threads() > 1
        }
        #[cfg(not(feature = "parallel"))]
        {
            false
        }
    }
}
