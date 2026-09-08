// SPDX-License-Identifier: MIT OR Apache-2.0

//! Original JPEG 2000 Tier-1 execution for one sub-band.

use super::{
    add_roi_shift_to_bitplanes, apply_roi_maxshift_inverse_i64, bitplane,
    classic_decode_job_parameters, code_block_required_by_index, collect_pending_classic_blocks,
    count_classic_code_blocks, decode_j2k_code_block_scalar_with_workspace,
    decode_j2k_code_block_scalar_with_workspace_midpoint,
    should_decode_classic_sub_band_in_parallel, ComponentInfo, CpuDecodeParallelism,
    DecodeAllocationBudget, DecodingError, DecompositionStorage, Header, HtCodeBlockDecoder,
    J2kCodeBlockBatchJob, J2kCodeBlockDecodeJob, J2kCodeBlockDecodeWorkspace, J2kSubBandDecodeJob,
    Result, SubBand, TileDecodeContext, Vec, MAX_BITPLANE_COUNT,
};
use crate::j2c::build::CodeBlockCoding;

#[cfg(feature = "parallel")]
use super::{
    copy_decoded_classic_blocks_to_sub_band, decode_classic_sub_band_blocks_parallel,
    release_coefficient_slab, ClassicParallelParameters,
};

#[expect(
    clippy::cast_precision_loss,
    reason = "the codec float domain intentionally receives bounded integer samples at this rounding boundary"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the Tier-1 boundary keeps geometry, state buffers, and validated options explicit"
)]
#[expect(
    clippy::too_many_lines,
    reason = "the ordered JPEG 2000 Tier-1 state machine stays cohesive across scalar, parallel, and adapter paths"
)]
pub(super) fn decode_sub_band_classic_blocks(
    sub_band_idx: usize,
    sub_band: &SubBand,
    component_info: &ComponentInfo,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'_>,
    header: &Header<'_>,
    decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    cpu_decode_parallelism: CpuDecodeParallelism,
    num_bitplanes: u8,
    dequantization_step: f32,
    irreversible_midpoint: bool,
) -> Result<()> {
    let coded_bitplanes =
        add_roi_shift_to_bitplanes(num_bitplanes, component_info.roi_shift, MAX_BITPLANE_COUNT)?;
    if storage.exact_integer_decode {
        return decode_sub_band_classic_blocks_i64(
            sub_band_idx,
            sub_band,
            component_info,
            tile_ctx,
            storage,
            header,
            coded_bitplanes,
        );
    }

    let (job_sub_band_type, job_style) =
        classic_decode_job_parameters(sub_band.sub_band_type, component_info);
    if let Some(decoder) = decoder.as_deref_mut() {
        let mut budget = DecodeAllocationBudget::for_storage(storage)?;
        let pending_blocks = collect_pending_classic_blocks(
            sub_band_idx,
            sub_band,
            component_info,
            storage,
            &mut budget,
        )?;
        let mut batch_jobs = Vec::new();
        budget.reserve_new(&mut batch_jobs, pending_blocks.len())?;
        for pending in &pending_blocks {
            batch_jobs.push(J2kCodeBlockBatchJob {
                output_x: pending.output_x,
                output_y: pending.output_y,
                code_block: J2kCodeBlockDecodeJob {
                    data: &pending.combined_data,
                    segments: &pending.segments,
                    width: pending.width,
                    height: pending.height,
                    output_stride: sub_band.rect.width() as usize,
                    missing_bit_planes: pending.missing_bit_planes,
                    number_of_coding_passes: pending.number_of_coding_passes,
                    total_bitplanes: num_bitplanes,
                    roi_shift: component_info.roi_shift,
                    sub_band_type: job_sub_band_type,
                    style: job_style,
                    strict: header.strict,
                    dequantization_step,
                },
            });
        }

        let base_store = &mut storage.coefficients[sub_band.coefficients.clone()];
        if decoder.decode_j2k_sub_band_with_midpoint(
            J2kSubBandDecodeJob {
                width: sub_band.rect.width(),
                height: sub_band.rect.height(),
                jobs: &batch_jobs,
            },
            base_store,
            irreversible_midpoint,
        )? {
            tile_ctx.debug_counters.decoded_code_blocks += batch_jobs.len();
            return Ok(());
        }

        let (workspace_width, workspace_height) =
            batch_jobs
                .iter()
                .fold((0_u32, 0_u32), |(width, height), job| {
                    (
                        width.max(job.code_block.width),
                        height.max(job.code_block.height),
                    )
                });
        let planned_workspace =
            bitplane::classic_decode_workspace_bytes(workspace_width, workspace_height)?;
        budget.include_bytes(planned_workspace)?;
        let mut scalar_workspace = J2kCodeBlockDecodeWorkspace::default();
        scalar_workspace.prepare(workspace_width, workspace_height)?;
        let actual_workspace = scalar_workspace.allocated_bytes()?;
        if actual_workspace > planned_workspace {
            budget.include_bytes(actual_workspace - planned_workspace)?;
        }

        let output_stride = sub_band.rect.width() as usize;
        for job in batch_jobs {
            tile_ctx.debug_counters.decoded_code_blocks += 1;
            let base_idx = (job.output_y * sub_band.rect.width()) as usize + job.output_x as usize;
            let output_len = if job.code_block.height == 0 {
                0
            } else {
                output_stride
                    .checked_mul(job.code_block.height as usize - 1)
                    .and_then(|prefix| prefix.checked_add(job.code_block.width as usize))
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?
            };
            let output_slice = &mut base_store[base_idx..base_idx + output_len];
            if decoder.decode_j2k_code_block_with_midpoint(
                job.code_block,
                output_slice,
                irreversible_midpoint,
            )? {
                continue;
            }
            let decode = if irreversible_midpoint {
                decode_j2k_code_block_scalar_with_workspace_midpoint
            } else {
                decode_j2k_code_block_scalar_with_workspace
            };
            decode(job.code_block, output_slice, &mut scalar_workspace)?;
        }
        return Ok(());
    }

    let code_block_count = count_classic_code_blocks(sub_band_idx, sub_band, storage)?;
    if should_decode_classic_sub_band_in_parallel(cpu_decode_parallelism, code_block_count) {
        #[cfg(feature = "parallel")]
        {
            let mut budget = DecodeAllocationBudget::for_storage(storage)?;
            let mut collection_budget = budget;
            let pending_result = collect_pending_classic_blocks(
                sub_band_idx,
                sub_band,
                component_info,
                storage,
                &mut collection_budget,
            );
            let pending_blocks = match pending_result {
                Ok(pending_blocks) => {
                    budget = collection_budget;
                    pending_blocks
                }
                Err(error)
                    if tile_ctx.parallel_coefficients.capacity() != 0
                        && super::super::reuse::is_capacity_error(&error) =>
                {
                    release_coefficient_slab(
                        &mut tile_ctx.parallel_coefficients,
                        &mut storage.structural_workspace_bytes,
                        #[cfg(test)]
                        &mut tile_ctx.debug_counters.parallel_coefficients,
                        &mut budget,
                    )?;
                    let mut retry_budget = budget;
                    let pending_blocks = collect_pending_classic_blocks(
                        sub_band_idx,
                        sub_band,
                        component_info,
                        storage,
                        &mut retry_budget,
                    )?;
                    budget = retry_budget;
                    pending_blocks
                }
                Err(error) => return Err(error),
            };
            let decoded_blocks = decode_classic_sub_band_blocks_parallel(
                &pending_blocks,
                sub_band,
                ClassicParallelParameters {
                    sub_band_type: job_sub_band_type,
                    style: job_style,
                    strict: header.strict,
                    total_bitplanes: num_bitplanes,
                    roi_shift: component_info.roi_shift,
                    dequantization_step,
                    irreversible_midpoint,
                },
                &mut tile_ctx.parallel_coefficients,
                &mut storage.structural_workspace_bytes,
                #[cfg(test)]
                &mut tile_ctx.debug_counters.parallel_coefficients,
                &mut budget,
            )?;
            tile_ctx.debug_counters.decoded_code_blocks += decoded_blocks.len();
            copy_decoded_classic_blocks_to_sub_band(
                &decoded_blocks,
                sub_band,
                storage,
                #[cfg(test)]
                &mut tile_ctx.debug_counters.parallel_coefficients,
            )?;
            return Ok(());
        }
    }

    for precinct in sub_band
        .precincts
        .clone()
        .map(|idx| &storage.precincts[idx])
    {
        for code_block in precinct
            .code_blocks
            .clone()
            .map(|idx| &storage.code_blocks[idx])
        {
            if code_block.coding != Some(CodeBlockCoding::Classic) {
                continue;
            }
            if !code_block_required_by_index(storage, sub_band_idx, code_block) {
                tile_ctx.debug_counters.skipped_code_blocks += 1;
                continue;
            }
            tile_ctx.debug_counters.decoded_code_blocks += 1;
            let x_offset = code_block.rect.x0 - sub_band.rect.x0;
            let y_offset = code_block.rect.y0 - sub_band.rect.y0;
            let output_stride = sub_band.rect.width() as usize;
            let mut base_idx = (y_offset * sub_band.rect.width()) as usize + x_offset as usize;

            bitplane::decode(
                code_block,
                sub_band.sub_band_type,
                coded_bitplanes,
                &component_info.coding_style.parameters.code_block_style,
                tile_ctx,
                storage,
                header.strict,
            )?;
            let base_store = &mut storage.coefficients[sub_band.coefficients.clone()];
            for coefficients in tile_ctx.bit_plane_decode_context.coefficient_rows() {
                let out_row = &mut base_store[base_idx..];
                for (output, coefficient) in out_row.iter_mut().zip(coefficients.iter().copied()) {
                    *output = if irreversible_midpoint {
                        tile_ctx
                            .bit_plane_decode_context
                            .reconstruct_irreversible_midpoint(
                                coefficient,
                                code_block.number_of_coding_passes,
                                component_info.roi_shift,
                            )
                    } else {
                        apply_roi_maxshift_inverse_i64(
                            coefficient.get_i64(),
                            component_info.roi_shift,
                        ) as f32
                    };
                    *output *= dequantization_step;
                }
                base_idx += output_stride;
            }
        }
    }
    Ok(())
}

fn decode_sub_band_classic_blocks_i64(
    sub_band_idx: usize,
    sub_band: &SubBand,
    component_info: &ComponentInfo,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'_>,
    header: &Header<'_>,
    coded_bitplanes: u8,
) -> Result<()> {
    for precinct in sub_band
        .precincts
        .clone()
        .map(|idx| &storage.precincts[idx])
    {
        for code_block in precinct
            .code_blocks
            .clone()
            .map(|idx| &storage.code_blocks[idx])
        {
            if code_block.coding != Some(CodeBlockCoding::Classic) {
                continue;
            }
            if !code_block_required_by_index(storage, sub_band_idx, code_block) {
                tile_ctx.debug_counters.skipped_code_blocks += 1;
                continue;
            }
            tile_ctx.debug_counters.decoded_code_blocks += 1;
            let x_offset = code_block.rect.x0 - sub_band.rect.x0;
            let y_offset = code_block.rect.y0 - sub_band.rect.y0;
            let output_stride = sub_band.rect.width() as usize;
            let mut base_idx = (y_offset * sub_band.rect.width()) as usize + x_offset as usize;

            bitplane::decode(
                code_block,
                sub_band.sub_band_type,
                coded_bitplanes,
                &component_info.coding_style.parameters.code_block_style,
                tile_ctx,
                storage,
                header.strict,
            )?;
            let base_store = &mut storage.coefficients_i64[sub_band.coefficients.clone()];
            for coefficients in tile_ctx.bit_plane_decode_context.coefficient_rows() {
                let out_row = &mut base_store[base_idx..];
                for (output, coefficient) in out_row.iter_mut().zip(coefficients.iter().copied()) {
                    *output = apply_roi_maxshift_inverse_i64(
                        coefficient.get_i64(),
                        component_info.roi_shift,
                    );
                }
                base_idx += output_stride;
            }
        }
    }
    Ok(())
}
