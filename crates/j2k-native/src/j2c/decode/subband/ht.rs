// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    add_roi_shift_to_bitplanes, apply_roi_maxshift_inverse_f32, apply_roi_maxshift_inverse_i32,
    apply_roi_maxshift_inverse_i64, code_block_required_by_index, collect_pending_ht_blocks,
    count_ht_code_blocks, ht_block_decode, ht_code_block_has_decodable_passes,
    should_decode_ht_sub_band_in_parallel, ComponentInfo, CpuDecodeParallelism,
    DecodeAllocationBudget, DecodingError, DecompositionStorage, Header, HtCodeBlockBatchJob,
    HtCodeBlockDecodeJob, HtCodeBlockDecoder, HtSubBandDecodeJob, Result, SubBand,
    TileDecodeContext, Vec,
};
use crate::j2c::build::CodeBlockCoding;

#[cfg(feature = "parallel")]
use super::{
    copy_decoded_ht_blocks_to_sub_band, decode_ht_sub_band_blocks_parallel,
    release_coefficient_slab, HtParallelParameters,
};

#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
pub(super) fn decode_sub_band_ht_blocks_i64(
    sub_band_idx: usize,
    sub_band: &SubBand,
    component_info: &ComponentInfo,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'_>,
    header: &Header<'_>,
    num_bitplanes: u8,
    profile_enabled: bool,
) -> Result<()> {
    let coded_bitplanes = add_roi_shift_to_bitplanes(num_bitplanes, component_info.roi_shift, 31)?;
    let stripe_causal = component_info
        .coding_style
        .parameters
        .code_block_style
        .vertically_causal_context;

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
            if code_block.coding != Some(CodeBlockCoding::HighThroughput) {
                continue;
            }
            if !code_block_required_by_index(storage, sub_band_idx, code_block) {
                tile_ctx.debug_counters.skipped_code_blocks += 1;
                continue;
            }

            if !ht_code_block_has_decodable_passes(code_block, coded_bitplanes, header.strict)? {
                continue;
            }

            tile_ctx.debug_counters.decoded_code_blocks += 1;
            ht_block_decode::decode_with_stats(
                code_block,
                coded_bitplanes,
                stripe_causal,
                &mut tile_ctx.ht_block_decode_context,
                storage,
                header.strict,
                Some(&mut tile_ctx.debug_counters.ht_phase_stats),
                profile_enabled,
            )?;

            let x_offset = code_block.rect.x0 - sub_band.rect.x0;
            let y_offset = code_block.rect.y0 - sub_band.rect.y0;
            let base_store = &mut storage.coefficients_i64[sub_band.coefficients.clone()];
            let mut base_idx = (y_offset * sub_band.rect.width()) as usize + x_offset as usize;
            let output_stride = sub_band.rect.width() as usize;

            for coefficients in tile_ctx.ht_block_decode_context.coefficient_rows() {
                let out_row = &mut base_store[base_idx..];

                for (output, coefficient) in out_row.iter_mut().zip(coefficients.iter().copied()) {
                    let coefficient = i64::from(ht_block_decode::coefficient_to_i32(
                        coefficient,
                        coded_bitplanes,
                    ));
                    *output = apply_roi_maxshift_inverse_i64(coefficient, component_info.roi_shift);
                }

                base_idx += output_stride;
            }
        }
    }

    Ok(())
}

#[expect(
    clippy::cast_precision_loss,
    reason = "the codec float domain intentionally receives bounded integer samples or metadata at this rounding boundary"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
#[expect(
    clippy::too_many_lines,
    reason = "the ordered JPEG 2000 state machine stays cohesive to preserve marker, packet, pass, and sample order"
)]
pub(super) fn decode_sub_band_ht_blocks(
    sub_band_idx: usize,
    sub_band: &SubBand,
    component_info: &ComponentInfo,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'_>,
    header: &Header<'_>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    cpu_decode_parallelism: CpuDecodeParallelism,
    num_bitplanes: u8,
    dequantization_step: f32,
    irreversible_midpoint: bool,
    profile_enabled: bool,
) -> Result<()> {
    let coded_bitplanes = add_roi_shift_to_bitplanes(num_bitplanes, component_info.roi_shift, 31)?;
    let stripe_causal = component_info
        .coding_style
        .parameters
        .code_block_style
        .vertically_causal_context;

    if let Some(ht_decoder) = ht_decoder.as_deref_mut() {
        let mut budget = DecodeAllocationBudget::for_storage(storage)?;
        let pending_blocks = collect_pending_ht_blocks(
            sub_band_idx,
            sub_band,
            storage,
            header,
            num_bitplanes,
            component_info.roi_shift,
            &mut budget,
        )?;

        let mut batch_jobs = Vec::new();
        budget.reserve_new(&mut batch_jobs, pending_blocks.len())?;
        for pending in &pending_blocks {
            batch_jobs.push(HtCodeBlockBatchJob {
                output_x: pending.output_x,
                output_y: pending.output_y,
                code_block: HtCodeBlockDecodeJob {
                    data: &pending.combined.data,
                    cleanup_length: pending.combined.cleanup_length,
                    refinement_length: pending.combined.refinement_length,
                    width: pending.width,
                    height: pending.height,
                    output_stride: sub_band.rect.width() as usize,
                    missing_bit_planes: pending.missing_bit_planes,
                    number_of_coding_passes: pending.number_of_coding_passes,
                    num_bitplanes,
                    roi_shift: component_info.roi_shift,
                    stripe_causal,
                    strict: header.strict,
                    dequantization_step,
                },
            });
        }

        let base_store = &mut storage.coefficients[sub_band.coefficients.clone()];
        if ht_decoder.decode_sub_band_with_midpoint(
            HtSubBandDecodeJob {
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
            ht_decoder.decode_code_block_with_midpoint(
                job.code_block,
                &mut base_store[base_idx..base_idx + output_len],
                irreversible_midpoint,
            )?;
        }

        return Ok(());
    }

    let code_block_count = count_ht_code_blocks(sub_band_idx, sub_band, storage)?;
    if !profile_enabled
        && should_decode_ht_sub_band_in_parallel(cpu_decode_parallelism, code_block_count)
    {
        #[cfg(feature = "parallel")]
        {
            let mut budget = DecodeAllocationBudget::for_storage(storage)?;
            let mut collection_budget = budget;
            let pending_result = collect_pending_ht_blocks(
                sub_band_idx,
                sub_band,
                storage,
                header,
                num_bitplanes,
                component_info.roi_shift,
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
                    let pending_blocks = collect_pending_ht_blocks(
                        sub_band_idx,
                        sub_band,
                        storage,
                        header,
                        num_bitplanes,
                        component_info.roi_shift,
                        &mut retry_budget,
                    )?;
                    budget = retry_budget;
                    pending_blocks
                }
                Err(error) => return Err(error),
            };
            let decoded_blocks = decode_ht_sub_band_blocks_parallel(
                &pending_blocks,
                sub_band,
                HtParallelParameters {
                    strict: header.strict,
                    num_bitplanes,
                    roi_shift: component_info.roi_shift,
                    stripe_causal,
                    dequantization_step,
                    irreversible_midpoint,
                },
                &mut tile_ctx.ht_task_workspaces,
                &mut tile_ctx.parallel_coefficients,
                &mut storage.structural_workspace_bytes,
                #[cfg(test)]
                &mut tile_ctx.debug_counters.ht_parallel_tasks,
                #[cfg(test)]
                &mut tile_ctx.debug_counters.ht_task_workspace_growths,
                #[cfg(test)]
                &mut tile_ctx.debug_counters.parallel_coefficients,
                &mut budget,
            )?;
            tile_ctx.debug_counters.decoded_code_blocks += decoded_blocks.len();
            copy_decoded_ht_blocks_to_sub_band(
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
            if code_block.coding != Some(CodeBlockCoding::HighThroughput) {
                continue;
            }
            if !code_block_required_by_index(storage, sub_band_idx, code_block) {
                tile_ctx.debug_counters.skipped_code_blocks += 1;
                continue;
            }
            tile_ctx.debug_counters.decoded_code_blocks += 1;
            ht_block_decode::decode_with_stats(
                code_block,
                coded_bitplanes,
                stripe_causal,
                &mut tile_ctx.ht_block_decode_context,
                storage,
                header.strict,
                Some(&mut tile_ctx.debug_counters.ht_phase_stats),
                profile_enabled,
            )?;

            let x_offset = code_block.rect.x0 - sub_band.rect.x0;
            let y_offset = code_block.rect.y0 - sub_band.rect.y0;
            let base_store = &mut storage.coefficients[sub_band.coefficients.clone()];
            let mut base_idx = (y_offset * sub_band.rect.width()) as usize + x_offset as usize;
            let output_stride = sub_band.rect.width() as usize;

            for coefficients in tile_ctx.ht_block_decode_context.coefficient_rows() {
                let out_row = &mut base_store[base_idx..];

                for (output, coefficient) in out_row.iter_mut().zip(coefficients.iter().copied()) {
                    *output = if irreversible_midpoint {
                        let coefficient =
                            ht_block_decode::coefficient_to_f32(coefficient, coded_bitplanes);
                        apply_roi_maxshift_inverse_f32(coefficient, component_info.roi_shift)
                    } else {
                        let coefficient =
                            ht_block_decode::coefficient_to_i32(coefficient, coded_bitplanes);
                        apply_roi_maxshift_inverse_i32(coefficient, component_info.roi_shift) as f32
                    };
                    *output *= dequantization_step;
                }

                base_idx += output_stride;
            }
        }
    }

    Ok(())
}
