// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible parallel code-block output assembly.

use super::pending::{PendingClassicBlock, PendingHtBlock};
use super::{DecodeAllocationBudget, DecompositionStorage, SubBand};
use crate::error::{bail, DecodingError, Result, ValidationError};
use crate::j2c::bitplane::classic_decode_workspace_bytes;
use crate::j2c::decode::workspace::{ht_task_workspace_bytes, HtTaskWorkspace};
use crate::j2c::ht_block_decode::ht_decode_workspace_bytes;
use crate::scalar::{
    decode_ht_code_block_scalar_with_workspace_midpoint,
    decode_j2k_code_block_scalar_with_workspace_midpoint,
};
use crate::{
    decode_ht_code_block_scalar_with_workspace, decode_j2k_code_block_scalar_with_workspace,
    try_resize_decode_elements, HtCodeBlockDecodeJob, HtCodeBlockDecodeWorkspace,
    J2kCodeBlockDecodeJob, J2kCodeBlockDecodeWorkspace, J2kCodeBlockStyle, J2kSubBandType,
};
use alloc::vec::Vec;
use rayon::prelude::*;

const PARALLEL_TASKS_PER_WORKER: usize = 2;

#[derive(Clone, Copy)]
struct PreparedTaskPlan {
    active_workspaces: usize,
    large_jobs: usize,
    large_chunk_size: usize,
    small_chunk_size: usize,
}

impl PreparedTaskPlan {
    fn chunks<'a, T>(&self, values: &'a [T]) -> impl Iterator<Item = &'a [T]> {
        let (large, small) = values.split_at(self.large_jobs);
        large
            .chunks(self.large_chunk_size)
            .chain(small.chunks(self.small_chunk_size))
    }
}

pub(crate) struct DecodedBlock<'a> {
    pub(crate) output_x: u32,
    pub(crate) output_y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) coefficients: &'a mut [f32],
}

#[derive(Clone, Copy)]
pub(super) struct ClassicParallelParameters {
    pub(super) sub_band_type: J2kSubBandType,
    pub(super) style: J2kCodeBlockStyle,
    pub(super) strict: bool,
    pub(super) total_bitplanes: u8,
    pub(super) roi_shift: u8,
    pub(super) dequantization_step: f32,
    pub(super) irreversible_midpoint: bool,
}

#[derive(Clone, Copy)]
pub(super) struct HtParallelParameters {
    pub(super) strict: bool,
    pub(super) num_bitplanes: u8,
    pub(super) roi_shift: u8,
    pub(super) stripe_causal: bool,
    pub(super) dequantization_step: f32,
    pub(super) irreversible_midpoint: bool,
}

pub(super) fn decode_classic_sub_band_blocks_parallel<'a>(
    pending_blocks: &[PendingClassicBlock],
    sub_band: &SubBand,
    parameters: ClassicParallelParameters,
    coefficient_slab: &'a mut Vec<f32>,
    structural_workspace_bytes: &mut usize,
    #[cfg(test)] coefficient_stats: &mut super::super::ParallelCoefficientStats,
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<DecodedBlock<'a>>> {
    let (mut decoded_blocks, total_coefficients) = prepare_classic_outputs(
        pending_blocks,
        sub_band,
        coefficient_slab,
        structural_workspace_bytes,
        #[cfg(test)]
        coefficient_stats,
        budget,
    )?;
    let mut workspaces = preallocate_classic_workspaces(
        pending_blocks,
        coefficient_slab,
        total_coefficients,
        structural_workspace_bytes,
        #[cfg(test)]
        coefficient_stats,
        budget,
    )?;
    install_decoded_blocks(
        &mut decoded_blocks,
        coefficient_slab,
        total_coefficients,
        pending_blocks.iter().map(|pending| {
            (
                pending.output_x,
                pending.output_y,
                pending.width,
                pending.height,
            )
        }),
    )?;
    decoded_blocks
        .par_iter_mut()
        .zip(pending_blocks.par_iter())
        .zip(workspaces.par_iter_mut())
        .try_for_each(|((decoded, pending), workspace)| -> Result<()> {
            decoded.coefficients.fill(0.0);
            let decode = if parameters.irreversible_midpoint {
                decode_j2k_code_block_scalar_with_workspace_midpoint
            } else {
                decode_j2k_code_block_scalar_with_workspace
            };
            decode(
                J2kCodeBlockDecodeJob {
                    data: &pending.combined_data,
                    segments: &pending.segments,
                    width: pending.width,
                    height: pending.height,
                    output_stride: pending.width as usize,
                    missing_bit_planes: pending.missing_bit_planes,
                    number_of_coding_passes: pending.number_of_coding_passes,
                    total_bitplanes: parameters.total_bitplanes,
                    roi_shift: parameters.roi_shift,
                    sub_band_type: parameters.sub_band_type,
                    style: parameters.style,
                    strict: parameters.strict,
                    dequantization_step: parameters.dequantization_step,
                },
                decoded.coefficients,
                workspace,
            )
        })?;
    Ok(decoded_blocks)
}

fn prepare_classic_outputs<'a>(
    pending_blocks: &[PendingClassicBlock],
    sub_band: &SubBand,
    coefficient_slab: &mut Vec<f32>,
    structural_workspace_bytes: &mut usize,
    #[cfg(test)] coefficient_stats: &mut super::super::ParallelCoefficientStats,
    budget: &mut DecodeAllocationBudget,
) -> Result<(Vec<DecodedBlock<'a>>, usize)> {
    let total_coefficients = validate_pending_blocks(
        pending_blocks.iter().map(|pending| {
            (
                pending.output_x,
                pending.output_y,
                pending.width,
                pending.height,
            )
        }),
        sub_band,
    )?;
    let decoded_blocks = reserve_decoded_block_descriptors(
        pending_blocks.len(),
        coefficient_slab,
        structural_workspace_bytes,
        #[cfg(test)]
        coefficient_stats,
        budget,
    )?;
    prepare_coefficient_slab(
        coefficient_slab,
        total_coefficients,
        structural_workspace_bytes,
        #[cfg(test)]
        coefficient_stats,
        budget,
    )?;
    Ok((decoded_blocks, total_coefficients))
}

fn preallocate_classic_workspaces(
    pending_blocks: &[PendingClassicBlock],
    coefficient_slab: &mut Vec<f32>,
    total_coefficients: usize,
    structural_workspace_bytes: &mut usize,
    #[cfg(test)] coefficient_stats: &mut super::super::ParallelCoefficientStats,
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<J2kCodeBlockDecodeWorkspace>> {
    let mut trial = *budget;
    match try_preallocate_classic_workspaces(pending_blocks, &mut trial) {
        Ok(workspaces) => {
            *budget = trial;
            Ok(workspaces)
        }
        Err(error)
            if coefficient_slab.capacity() > total_coefficients
                && super::super::reuse::is_capacity_error(&error) =>
        {
            release_coefficient_slab(
                coefficient_slab,
                structural_workspace_bytes,
                #[cfg(test)]
                coefficient_stats,
                budget,
            )?;
            prepare_coefficient_slab(
                coefficient_slab,
                total_coefficients,
                structural_workspace_bytes,
                #[cfg(test)]
                coefficient_stats,
                budget,
            )?;
            let mut retry = *budget;
            let workspaces = try_preallocate_classic_workspaces(pending_blocks, &mut retry)?;
            *budget = retry;
            Ok(workspaces)
        }
        Err(error) => Err(error),
    }
}

fn try_preallocate_classic_workspaces(
    pending_blocks: &[PendingClassicBlock],
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<J2kCodeBlockDecodeWorkspace>> {
    let planned_bytes =
        pending_blocks
            .iter()
            .try_fold(0usize, |bytes, pending| -> Result<usize> {
                bytes
                    .checked_add(classic_decode_workspace_bytes(
                        pending.width,
                        pending.height,
                    )?)
                    .ok_or_else(|| ValidationError::ImageTooLarge.into())
            })?;
    budget.include_bytes(planned_bytes)?;
    let mut workspaces = Vec::new();
    budget.reserve_new(&mut workspaces, pending_blocks.len())?;
    for pending in pending_blocks {
        let planned = classic_decode_workspace_bytes(pending.width, pending.height)?;
        let mut workspace = J2kCodeBlockDecodeWorkspace::default();
        workspace.prepare(pending.width, pending.height)?;
        let actual = workspace.allocated_bytes()?;
        if actual > planned {
            budget.include_bytes(actual - planned)?;
        }
        workspaces.push(workspace);
    }
    Ok(workspaces)
}

#[cfg_attr(
    test,
    expect(
        clippy::too_many_arguments,
        reason = "test-only counters remain separate from the disjoint production buffer borrows"
    )
)]
pub(super) fn decode_ht_sub_band_blocks_parallel<'a>(
    pending_blocks: &[PendingHtBlock],
    sub_band: &SubBand,
    parameters: HtParallelParameters,
    workspaces: &mut Vec<HtTaskWorkspace>,
    coefficient_slab: &'a mut Vec<f32>,
    structural_workspace_bytes: &mut usize,
    #[cfg(test)] maximum_tasks: &mut usize,
    #[cfg(test)] workspace_growths: &mut usize,
    #[cfg(test)] coefficient_stats: &mut super::super::ParallelCoefficientStats,
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<DecodedBlock<'a>>> {
    let (mut decoded_blocks, total_coefficients) = prepare_ht_outputs(
        pending_blocks,
        sub_band,
        coefficient_slab,
        structural_workspace_bytes,
        #[cfg(test)]
        coefficient_stats,
        budget,
    )?;
    let (plan, growths) = prepare_ht_task_workspaces_with_coefficients(
        pending_blocks,
        workspaces,
        coefficient_slab,
        total_coefficients,
        structural_workspace_bytes,
        #[cfg(test)]
        coefficient_stats,
        budget,
    )?;
    install_decoded_blocks(
        &mut decoded_blocks,
        coefficient_slab,
        total_coefficients,
        pending_blocks.iter().map(|pending| {
            (
                pending.output_x,
                pending.output_y,
                pending.width,
                pending.height,
            )
        }),
    )?;
    #[cfg(test)]
    {
        *maximum_tasks = (*maximum_tasks).max(plan.active_workspaces);
        *workspace_growths = workspace_growths.saturating_add(growths);
    }
    #[cfg(not(test))]
    let _ = growths;
    let (large_decoded, small_decoded) = decoded_blocks.split_at_mut(plan.large_jobs);
    let (large_pending, small_pending) = pending_blocks.split_at(plan.large_jobs);
    large_decoded
        .par_chunks_mut(plan.large_chunk_size)
        .zip(large_pending.par_chunks(plan.large_chunk_size))
        .chain(
            small_decoded
                .par_chunks_mut(plan.small_chunk_size)
                .zip(small_pending.par_chunks(plan.small_chunk_size)),
        )
        .zip(workspaces[..plan.active_workspaces].par_iter_mut())
        .try_for_each(|((decoded_chunk, pending_chunk), slot)| -> Result<()> {
            for (decoded, pending) in decoded_chunk.iter_mut().zip(pending_chunk) {
                decoded.coefficients.fill(0.0);
                let decode = if parameters.irreversible_midpoint {
                    decode_ht_code_block_scalar_with_workspace_midpoint
                } else {
                    decode_ht_code_block_scalar_with_workspace
                };
                decode(
                    HtCodeBlockDecodeJob {
                        data: &pending.combined.data,
                        cleanup_length: pending.combined.cleanup_length,
                        refinement_length: pending.combined.refinement_length,
                        width: pending.width,
                        height: pending.height,
                        output_stride: pending.width as usize,
                        missing_bit_planes: pending.missing_bit_planes,
                        number_of_coding_passes: pending.number_of_coding_passes,
                        num_bitplanes: parameters.num_bitplanes,
                        roi_shift: parameters.roi_shift,
                        stripe_causal: parameters.stripe_causal,
                        strict: parameters.strict,
                        dequantization_step: parameters.dequantization_step,
                    },
                    decoded.coefficients,
                    &mut slot.workspace,
                )?;
            }
            Ok(())
        })?;
    Ok(decoded_blocks)
}

fn prepare_ht_outputs<'a>(
    pending_blocks: &[PendingHtBlock],
    sub_band: &SubBand,
    coefficient_slab: &mut Vec<f32>,
    structural_workspace_bytes: &mut usize,
    #[cfg(test)] coefficient_stats: &mut super::super::ParallelCoefficientStats,
    budget: &mut DecodeAllocationBudget,
) -> Result<(Vec<DecodedBlock<'a>>, usize)> {
    let total_coefficients = validate_pending_blocks(
        pending_blocks.iter().map(|pending| {
            (
                pending.output_x,
                pending.output_y,
                pending.width,
                pending.height,
            )
        }),
        sub_band,
    )?;
    let decoded_blocks = reserve_decoded_block_descriptors(
        pending_blocks.len(),
        coefficient_slab,
        structural_workspace_bytes,
        #[cfg(test)]
        coefficient_stats,
        budget,
    )?;
    prepare_coefficient_slab(
        coefficient_slab,
        total_coefficients,
        structural_workspace_bytes,
        #[cfg(test)]
        coefficient_stats,
        budget,
    )?;
    Ok((decoded_blocks, total_coefficients))
}

fn prepare_ht_task_workspaces_with_coefficients(
    pending_blocks: &[PendingHtBlock],
    workspaces: &mut Vec<HtTaskWorkspace>,
    coefficient_slab: &mut Vec<f32>,
    total_coefficients: usize,
    structural_workspace_bytes: &mut usize,
    #[cfg(test)] coefficient_stats: &mut super::super::ParallelCoefficientStats,
    budget: &mut DecodeAllocationBudget,
) -> Result<(PreparedTaskPlan, usize)> {
    let initial_tasks = bounded_parallel_task_count(pending_blocks.len());
    let mut requested_tasks = initial_tasks;
    let mut evicted_retained_bank = false;
    let mut shrunk_coefficient_slab = false;
    loop {
        let plan = balanced_task_plan(pending_blocks.len(), requested_tasks)?;
        match try_prepare_ht_task_workspaces(
            pending_blocks,
            plan,
            workspaces,
            structural_workspace_bytes,
            budget,
        ) {
            Err(error)
                if !shrunk_coefficient_slab
                    && coefficient_slab.capacity() > total_coefficients
                    && super::super::reuse::is_capacity_error(&error) =>
            {
                release_coefficient_slab(
                    coefficient_slab,
                    structural_workspace_bytes,
                    #[cfg(test)]
                    coefficient_stats,
                    budget,
                )?;
                prepare_coefficient_slab(
                    coefficient_slab,
                    total_coefficients,
                    structural_workspace_bytes,
                    #[cfg(test)]
                    coefficient_stats,
                    budget,
                )?;
                requested_tasks = initial_tasks;
                shrunk_coefficient_slab = true;
            }
            Err(error)
                if requested_tasks > 1
                    && matches!(
                        error,
                        crate::DecodeError::Validation(ValidationError::ImageTooLarge)
                    ) =>
            {
                requested_tasks = requested_tasks.div_ceil(2);
            }
            Err(error)
                if !evicted_retained_bank
                    && matches!(
                        error,
                        crate::DecodeError::Validation(ValidationError::ImageTooLarge)
                    )
                    && !workspaces.is_empty() =>
            {
                release_ht_task_bank(workspaces, structural_workspace_bytes, budget)?;
                requested_tasks = initial_tasks;
                evicted_retained_bank = true;
            }
            result => return result.map(|growths| (plan, growths)),
        }
    }
}

#[cfg(test)]
fn prepare_ht_task_workspaces(
    pending_blocks: &[PendingHtBlock],
    workspaces: &mut Vec<HtTaskWorkspace>,
    structural_workspace_bytes: &mut usize,
    budget: &mut DecodeAllocationBudget,
) -> Result<(PreparedTaskPlan, usize)> {
    let mut coefficient_slab = Vec::new();
    let mut coefficient_stats = super::super::ParallelCoefficientStats::default();
    prepare_ht_task_workspaces_with_coefficients(
        pending_blocks,
        workspaces,
        &mut coefficient_slab,
        0,
        structural_workspace_bytes,
        &mut coefficient_stats,
        budget,
    )
}

fn bounded_parallel_task_count(job_count: usize) -> usize {
    // Two tasks per worker preserve bounded workspace ownership while giving Rayon
    // another ready chunk when code-block decode costs differ.
    job_count.min(rayon::current_num_threads().saturating_mul(PARALLEL_TASKS_PER_WORKER))
}

fn balanced_task_plan(job_count: usize, requested_tasks: usize) -> Result<PreparedTaskPlan> {
    if job_count == 0 {
        return Ok(PreparedTaskPlan {
            active_workspaces: 0,
            large_jobs: 0,
            large_chunk_size: 1,
            small_chunk_size: 1,
        });
    }
    if requested_tasks == 0 {
        return Err(DecodingError::CodeBlockDecodeFailure.into());
    }
    let active_workspaces = job_count.min(requested_tasks);
    let small_chunk_size = job_count / active_workspaces;
    let large_chunks = job_count % active_workspaces;
    let large_chunk_size = if large_chunks == 0 {
        small_chunk_size
    } else {
        small_chunk_size
            .checked_add(1)
            .ok_or(ValidationError::ImageTooLarge)?
    };
    let large_jobs = large_chunks
        .checked_mul(large_chunk_size)
        .ok_or(ValidationError::ImageTooLarge)?;
    Ok(PreparedTaskPlan {
        active_workspaces,
        large_jobs,
        large_chunk_size,
        small_chunk_size,
    })
}

fn release_ht_task_bank(
    workspaces: &mut Vec<HtTaskWorkspace>,
    structural_workspace_bytes: &mut usize,
    budget: &mut DecodeAllocationBudget,
) -> Result<()> {
    let released = ht_task_workspace_bytes(workspaces)?;
    let structural = structural_workspace_bytes
        .checked_sub(released)
        .ok_or(ValidationError::ImageTooLarge)?;
    let mut adjusted = *budget;
    adjusted.release_bytes(released)?;
    *workspaces = Vec::new();
    *structural_workspace_bytes = structural;
    *budget = adjusted;
    Ok(())
}

fn charge_actual_workspace(
    budget: &mut DecodeAllocationBudget,
    planned: usize,
    actual: usize,
) -> Result<()> {
    if actual > planned {
        budget.include_bytes(actual - planned)
    } else {
        budget.release_bytes(planned - actual)
    }
}

fn updated_structural_bytes(
    structural_workspace_bytes: usize,
    old_bank_bytes: usize,
    new_bank_bytes: usize,
) -> Result<usize> {
    structural_workspace_bytes
        .checked_sub(old_bank_bytes)
        .and_then(|bytes| bytes.checked_add(new_bank_bytes))
        .ok_or(ValidationError::ImageTooLarge.into())
}

#[expect(
    clippy::too_many_lines,
    reason = "the staged replacement transaction keeps allocation, rollback, and ledger reconciliation together"
)]
fn try_prepare_ht_task_workspaces(
    pending_blocks: &[PendingHtBlock],
    plan: PreparedTaskPlan,
    workspaces: &mut Vec<HtTaskWorkspace>,
    structural_workspace_bytes: &mut usize,
    budget: &mut DecodeAllocationBudget,
) -> Result<usize> {
    let old_bank_bytes = ht_task_workspace_bytes(workspaces)?;
    let mut trial = *budget;
    if workspaces.capacity() < plan.active_workspaces {
        let mut replacement = Vec::new();
        trial.reserve_new(&mut replacement, plan.active_workspaces)?;
        for (index, chunk) in plan.chunks(pending_blocks).enumerate() {
            let (required_width, required_height) = ht_chunk_dimensions(chunk);
            let (width, height) =
                workspaces
                    .get(index)
                    .map_or((required_width, required_height), |old| {
                        (
                            required_width.max(old.prepared_width),
                            required_height.max(old.prepared_height),
                        )
                    });
            let planned = ht_decode_workspace_bytes(width, height)?;
            trial.include_bytes(planned)?;
            let mut workspace = HtCodeBlockDecodeWorkspace::default();
            workspace.reserve(width, height)?;
            charge_actual_workspace(&mut trial, planned, workspace.allocated_bytes()?)?;
            replacement.push(HtTaskWorkspace {
                workspace,
                prepared_width: width,
                prepared_height: height,
            });
        }
        let new_bank_bytes = ht_task_workspace_bytes(&replacement)?;
        #[cfg(test)]
        let growths = {
            let mut growths = 0usize;
            for (index, slot) in replacement.iter().enumerate() {
                let grew = match workspaces.get(index) {
                    Some(old) => {
                        slot.workspace.allocated_bytes()? > old.workspace.allocated_bytes()?
                    }
                    None => true,
                };
                if grew {
                    growths = growths.saturating_add(1);
                }
            }
            growths
        };
        #[cfg(not(test))]
        let growths = 0;
        trial.release_bytes(old_bank_bytes)?;
        let new_structural =
            updated_structural_bytes(*structural_workspace_bytes, old_bank_bytes, new_bank_bytes)?;
        *workspaces = replacement;
        *structural_workspace_bytes = new_structural;
        *budget = trial;
        return Ok(growths);
    }

    let replacement_count = plan
        .chunks(pending_blocks)
        .enumerate()
        .filter(|&(index, chunk)| {
            let (width, height) = ht_chunk_dimensions(chunk);
            workspaces
                .get(index)
                .is_none_or(|slot| width > slot.prepared_width || height > slot.prepared_height)
        })
        .count();
    if replacement_count == 0 {
        return Ok(0);
    }

    let mut replacements = Vec::new();
    trial.reserve_new(&mut replacements, replacement_count)?;
    let replacement_metadata_bytes = replacements
        .capacity()
        .checked_mul(core::mem::size_of::<(usize, HtTaskWorkspace)>())
        .ok_or(ValidationError::ImageTooLarge)?;
    let mut replaced_workspace_bytes = 0usize;
    #[cfg(test)]
    let mut growths = 0usize;
    for (index, chunk) in plan.chunks(pending_blocks).enumerate() {
        let (required_width, required_height) = ht_chunk_dimensions(chunk);
        if workspaces.get(index).is_some_and(|slot| {
            required_width <= slot.prepared_width && required_height <= slot.prepared_height
        }) {
            continue;
        }
        let (width, height) =
            workspaces
                .get(index)
                .map_or((required_width, required_height), |old| {
                    (
                        required_width.max(old.prepared_width),
                        required_height.max(old.prepared_height),
                    )
                });
        let planned = ht_decode_workspace_bytes(width, height)?;
        trial.include_bytes(planned)?;
        let mut workspace = HtCodeBlockDecodeWorkspace::default();
        workspace.reserve(width, height)?;
        charge_actual_workspace(&mut trial, planned, workspace.allocated_bytes()?)?;
        if let Some(old) = workspaces.get(index) {
            #[cfg(test)]
            if workspace.allocated_bytes()? > old.workspace.allocated_bytes()? {
                growths = growths.saturating_add(1);
            }
            replaced_workspace_bytes = replaced_workspace_bytes
                .checked_add(old.workspace.allocated_bytes()?)
                .ok_or(ValidationError::ImageTooLarge)?;
        } else {
            #[cfg(test)]
            {
                growths = growths.saturating_add(1);
            }
        }
        replacements.push((
            index,
            HtTaskWorkspace {
                workspace,
                prepared_width: width,
                prepared_height: height,
            },
        ));
    }
    let added_workspace_bytes =
        replacements
            .iter()
            .try_fold(0usize, |bytes, (_, slot)| -> Result<usize> {
                bytes
                    .checked_add(slot.workspace.allocated_bytes()?)
                    .ok_or(ValidationError::ImageTooLarge.into())
            })?;
    let new_bank_bytes = old_bank_bytes
        .checked_sub(replaced_workspace_bytes)
        .and_then(|bytes| bytes.checked_add(added_workspace_bytes))
        .ok_or(ValidationError::ImageTooLarge)?;
    let new_structural =
        updated_structural_bytes(*structural_workspace_bytes, old_bank_bytes, new_bank_bytes)?;
    trial.release_bytes(replaced_workspace_bytes)?;
    trial.release_bytes(replacement_metadata_bytes)?;
    for (index, replacement) in replacements {
        if let Some(slot) = workspaces.get_mut(index) {
            *slot = replacement;
        } else {
            workspaces.push(replacement);
        }
    }
    *structural_workspace_bytes = new_structural;
    *budget = trial;
    #[cfg(test)]
    let result = growths;
    #[cfg(not(test))]
    let result = 0;
    Ok(result)
}

fn ht_chunk_dimensions(chunk: &[PendingHtBlock]) -> (u32, u32) {
    chunk.iter().fold((0, 0), |(width, height), pending| {
        (width.max(pending.width), height.max(pending.height))
    })
}

fn validate_pending_blocks(
    mut blocks: impl Iterator<Item = (u32, u32, u32, u32)>,
    sub_band: &SubBand,
) -> Result<usize> {
    blocks.try_fold(0usize, |total, (output_x, output_y, width, height)| {
        if output_x
            .checked_add(width)
            .is_none_or(|x1| x1 > sub_band.rect.width())
            || output_y
                .checked_add(height)
                .is_none_or(|y1| y1 > sub_band.rect.height())
        {
            return Err(DecodingError::CodeBlockDecodeFailure.into());
        }
        total
            .checked_add(block_coefficient_count(width, height)?)
            .ok_or(ValidationError::ImageTooLarge.into())
    })
}

fn install_decoded_blocks<'a>(
    decoded_blocks: &mut Vec<DecodedBlock<'a>>,
    coefficient_slab: &'a mut [f32],
    total_coefficients: usize,
    blocks: impl Iterator<Item = (u32, u32, u32, u32)>,
) -> Result<()> {
    let mut remaining = coefficient_slab
        .get_mut(..total_coefficients)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    for (output_x, output_y, width, height) in blocks {
        let coefficient_count = block_coefficient_count(width, height)?;
        let (coefficients, tail) = remaining.split_at_mut(coefficient_count);
        remaining = tail;
        decoded_blocks.push(DecodedBlock {
            output_x,
            output_y,
            width,
            height,
            coefficients,
        });
    }
    Ok(())
}

fn prepare_coefficient_slab(
    coefficients: &mut Vec<f32>,
    len: usize,
    structural_workspace_bytes: &mut usize,
    #[cfg(test)] coefficient_stats: &mut super::super::ParallelCoefficientStats,
    budget: &mut DecodeAllocationBudget,
) -> Result<()> {
    #[cfg(test)]
    let old_bytes = coefficient_slab_bytes(coefficients)?;
    if coefficients.capacity() >= len {
        #[cfg(test)]
        coefficient_stats.record_retained(old_bytes);
        // Keep every initialized element available for later sub-bands; each
        // owning worker clears only its active borrowed slice before entropy.
        if coefficients.len() < len {
            coefficients.resize(len, 0.0);
        }
        return Ok(());
    }

    // Previous sub-band coefficients have already been scattered, so an
    // undersized retained owner is optional scratch and can be released
    // before allocating its replacement. This keeps the live ledger exact
    // and avoids rejecting a fresh slab that fits without the old capacity.
    release_coefficient_slab(
        coefficients,
        structural_workspace_bytes,
        #[cfg(test)]
        coefficient_stats,
        budget,
    )?;

    let mut replacement_budget = *budget;
    replacement_budget.include_elements::<f32>(len)?;
    let mut replacement = Vec::new();
    try_resize_decode_elements(&mut replacement, len, 0.0)?;
    replacement_budget.include_capacity_overage::<f32>(len, replacement.capacity())?;
    let new_bytes = coefficient_slab_bytes(&replacement)?;
    let new_structural = structural_workspace_bytes
        .checked_add(new_bytes)
        .ok_or(ValidationError::ImageTooLarge)?;
    *coefficients = replacement;
    *structural_workspace_bytes = new_structural;
    *budget = replacement_budget;
    #[cfg(test)]
    coefficient_stats.record_allocation(new_bytes, old_bytes.max(new_bytes));
    Ok(())
}

fn reserve_decoded_block_descriptors<'a>(
    count: usize,
    coefficient_slab: &mut Vec<f32>,
    structural_workspace_bytes: &mut usize,
    #[cfg(test)] coefficient_stats: &mut super::super::ParallelCoefficientStats,
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<DecodedBlock<'a>>> {
    let mut descriptors = Vec::new();
    let mut trial = *budget;
    match trial.reserve_new(&mut descriptors, count) {
        Ok(()) => {
            *budget = trial;
            Ok(descriptors)
        }
        Err(error)
            if coefficient_slab.capacity() != 0
                && super::super::reuse::is_capacity_error(&error) =>
        {
            release_coefficient_slab(
                coefficient_slab,
                structural_workspace_bytes,
                #[cfg(test)]
                coefficient_stats,
                budget,
            )?;
            let mut retry = *budget;
            retry.reserve_new(&mut descriptors, count)?;
            *budget = retry;
            Ok(descriptors)
        }
        Err(error) => Err(error),
    }
}

pub(super) fn release_coefficient_slab(
    coefficients: &mut Vec<f32>,
    structural_workspace_bytes: &mut usize,
    #[cfg(test)] coefficient_stats: &mut super::super::ParallelCoefficientStats,
    budget: &mut DecodeAllocationBudget,
) -> Result<()> {
    let released = coefficient_slab_bytes(coefficients)?;
    let structural = structural_workspace_bytes
        .checked_sub(released)
        .ok_or(ValidationError::ImageTooLarge)?;
    let mut adjusted = *budget;
    adjusted.release_bytes(released)?;
    #[cfg(test)]
    coefficient_stats.record_retained(released);
    *coefficients = Vec::new();
    *structural_workspace_bytes = structural;
    *budget = adjusted;
    Ok(())
}

fn coefficient_slab_bytes(coefficients: &Vec<f32>) -> Result<usize> {
    coefficients
        .capacity()
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(ValidationError::ImageTooLarge.into())
}

fn block_coefficient_count(width: u32, height: u32) -> Result<usize> {
    (width as usize)
        .checked_mul(height as usize)
        .ok_or(ValidationError::ImageTooLarge.into())
}

pub(crate) fn copy_decoded_classic_blocks_to_sub_band(
    decoded_blocks: &[DecodedBlock<'_>],
    sub_band: &SubBand,
    storage: &mut DecompositionStorage<'_>,
    #[cfg(test)] coefficient_stats: &mut super::super::ParallelCoefficientStats,
) -> Result<()> {
    copy_decoded_blocks_to_sub_band(
        decoded_blocks,
        sub_band,
        storage,
        #[cfg(test)]
        coefficient_stats,
    )
}

pub(crate) fn copy_decoded_ht_blocks_to_sub_band(
    decoded_blocks: &[DecodedBlock<'_>],
    sub_band: &SubBand,
    storage: &mut DecompositionStorage<'_>,
    #[cfg(test)] coefficient_stats: &mut super::super::ParallelCoefficientStats,
) -> Result<()> {
    copy_decoded_blocks_to_sub_band(
        decoded_blocks,
        sub_band,
        storage,
        #[cfg(test)]
        coefficient_stats,
    )
}

fn copy_decoded_blocks_to_sub_band(
    decoded_blocks: &[DecodedBlock<'_>],
    sub_band: &SubBand,
    storage: &mut DecompositionStorage<'_>,
    #[cfg(test)] coefficient_stats: &mut super::super::ParallelCoefficientStats,
) -> Result<()> {
    let sub_band_width = sub_band.rect.width() as usize;
    let base_store = &mut storage.coefficients[sub_band.coefficients.clone()];
    #[cfg(test)]
    let mut scattered_bytes = 0usize;
    for block in decoded_blocks {
        let output_x = block.output_x;
        let output_y = block.output_y;
        let width = block.width;
        let height = block.height;
        if output_x
            .checked_add(width)
            .is_none_or(|x1| x1 > sub_band.rect.width())
            || output_y
                .checked_add(height)
                .is_none_or(|y1| y1 > sub_band.rect.height())
        {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
        let block_width = width as usize;
        for row in 0..height as usize {
            let dst_start = (output_y as usize + row)
                .checked_mul(sub_band_width)
                .and_then(|offset| offset.checked_add(output_x as usize))
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let dst_end = dst_start
                .checked_add(block_width)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let src_start = row
                .checked_mul(block_width)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let src_end = src_start
                .checked_add(block_width)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            base_store[dst_start..dst_end].copy_from_slice(&block.coefficients[src_start..src_end]);
            #[cfg(test)]
            {
                scattered_bytes = scattered_bytes
                    .saturating_add(block_width.saturating_mul(core::mem::size_of::<f32>()));
            }
        }
    }
    #[cfg(test)]
    coefficient_stats.record_scatter(scattered_bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        balanced_task_plan, decode_ht_sub_band_blocks_parallel, install_decoded_blocks,
        prepare_coefficient_slab, prepare_ht_outputs, prepare_ht_task_workspaces,
        prepare_ht_task_workspaces_with_coefficients, try_prepare_ht_task_workspaces,
        validate_pending_blocks, HtParallelParameters,
    };
    use crate::error::{DecodeError, ValidationError};
    use crate::j2c::build::{SubBand, SubBandType};
    use crate::j2c::decode::subband::pending::PendingHtBlock;
    use crate::j2c::decode::DecodeAllocationBudget;
    use crate::j2c::ht_block_decode::CombinedCodeBlockData;
    use crate::j2c::rect::IntRect;
    use crate::{try_reserve_decode_elements, try_resize_decode_elements};
    use alloc::vec::Vec;
    use core::mem::size_of;
    use rayon::ThreadPoolBuilder;

    fn ht_pending(dimensions: &[(u32, u32)]) -> Vec<PendingHtBlock> {
        dimensions
            .iter()
            .map(|&(width, height)| PendingHtBlock {
                combined: CombinedCodeBlockData {
                    data: Vec::new(),
                    cleanup_length: 0,
                    refinement_length: 0,
                },
                output_x: 0,
                output_y: 0,
                width,
                height,
                missing_bit_planes: 0,
                number_of_coding_passes: 0,
            })
            .collect()
    }

    fn test_sub_band(width: u32, height: u32) -> SubBand {
        SubBand {
            sub_band_type: SubBandType::LowLow,
            rect: IntRect::from_xywh(0, 0, width, height),
            precincts: 0..0,
            coefficients: 0..(width as usize * height as usize),
        }
    }

    #[test]
    fn balanced_task_plans_preserve_exact_task_counts_and_input_order() {
        for (job_count, requested_tasks, expected_lengths) in [
            (16, 12, &[2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1][..]),
            (13, 12, &[2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1][..]),
            (12, 4, &[3, 3, 3, 3][..]),
            (3, 12, &[1, 1, 1][..]),
            (0, 12, &[][..]),
        ] {
            let values: Vec<_> = (0..job_count).collect();
            let plan = balanced_task_plan(job_count, requested_tasks).unwrap();
            let chunks: Vec<_> = plan.chunks(&values).collect();

            assert_eq!(plan.active_workspaces, expected_lengths.len());
            assert_eq!(
                chunks.iter().map(|chunk| chunk.len()).collect::<Vec<_>>(),
                expected_lengths
            );
            assert_eq!(
                chunks.into_iter().flatten().copied().collect::<Vec<_>>(),
                values
            );
        }
    }

    #[test]
    fn ht_parallel_workspace_count_is_bounded_by_workers_and_jobs() {
        let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        pool.install(|| {
            for (job_count, expected) in [(8, 4), (2, 2)] {
                let pending = ht_pending(&vec![(32, 32); job_count]);
                let mut budget = DecodeAllocationBudget::from_live_bytes(0).unwrap();
                let mut workspaces = Vec::new();
                let mut structural = 0;
                let (plan, _) = prepare_ht_task_workspaces(
                    &pending,
                    &mut workspaces,
                    &mut structural,
                    &mut budget,
                )
                .unwrap();

                assert_eq!(plan.active_workspaces, expected);
                assert_eq!(workspaces.len(), expected);
            }
        });
    }

    #[test]
    fn ht_capacity_retry_uses_actual_capacity_and_an_unpoisoned_budget() {
        let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        pool.install(|| {
            let pending = ht_pending(&[(32, 32); 8]);
            let mut workspace = crate::HtCodeBlockDecodeWorkspace::default();
            workspace.reserve(32, 32).unwrap();
            let workspace_bytes = workspace.allocated_bytes().unwrap();
            let mut metadata = Vec::<super::HtTaskWorkspace>::new();
            try_reserve_decode_elements(&mut metadata, 1).unwrap();
            let one_task_cap = metadata.capacity() * core::mem::size_of::<super::HtTaskWorkspace>()
                + workspace_bytes;
            let mut budget =
                DecodeAllocationBudget::from_live_bytes_with_cap(0, one_task_cap).unwrap();
            let mut workspaces = Vec::new();
            let mut structural = 0;

            let (plan, growths) =
                prepare_ht_task_workspaces(&pending, &mut workspaces, &mut structural, &mut budget)
                    .expect("one HT task fits after larger task plans exceed the cap");

            assert_eq!(plan.active_workspaces, 1);
            assert_eq!(growths, 1);
            assert_eq!(workspaces.len(), 1);
            assert_eq!(
                super::ht_task_workspace_bytes(&workspaces).unwrap(),
                structural
            );
            assert_eq!(budget.live_bytes(), structural);
            assert_eq!(structural, one_task_cap);
        });
    }

    #[test]
    fn ht_workspace_pressure_shrinks_coefficients_before_reducing_task_count() {
        let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        pool.install(|| {
            let pending = ht_pending(&[(32, 32); 8]);
            let total_coefficients = 8 * 32 * 32;
            let mut fresh_coefficients = vec![0.0; total_coefficients];
            let fresh_coefficient_bytes = fresh_coefficients.capacity() * size_of::<f32>();
            let mut fresh_workspaces = Vec::new();
            let mut fresh_structural = fresh_coefficient_bytes;
            let mut fresh_stats = super::super::super::ParallelCoefficientStats::default();
            let mut fresh_budget =
                DecodeAllocationBudget::from_live_bytes(fresh_structural).unwrap();
            let (fresh_plan, _) = prepare_ht_task_workspaces_with_coefficients(
                &pending,
                &mut fresh_workspaces,
                &mut fresh_coefficients,
                total_coefficients,
                &mut fresh_structural,
                &mut fresh_stats,
                &mut fresh_budget,
            )
            .unwrap();
            assert_eq!(fresh_plan.active_workspaces, 4);
            let cap = fresh_budget.live_bytes();
            let workspace_bytes = cap - fresh_coefficient_bytes;

            let oversized_len = total_coefficients + workspace_bytes / (2 * size_of::<f32>());
            let mut coefficients = vec![7.0; oversized_len];
            let old_coefficient_bytes = coefficients.capacity() * size_of::<f32>();
            assert!(old_coefficient_bytes < cap);
            assert!(old_coefficient_bytes + workspace_bytes > cap);
            let mut workspaces = Vec::new();
            let mut structural = old_coefficient_bytes;
            let mut stats = super::super::super::ParallelCoefficientStats::default();
            let mut budget =
                DecodeAllocationBudget::from_live_bytes_with_cap(structural, cap).unwrap();

            let (plan, _) = prepare_ht_task_workspaces_with_coefficients(
                &pending,
                &mut workspaces,
                &mut coefficients,
                total_coefficients,
                &mut structural,
                &mut stats,
                &mut budget,
            )
            .expect("shrinking optional coefficients preserves the fitting initial task plan");

            assert_eq!(plan.active_workspaces, fresh_plan.active_workspaces);
            assert_eq!(workspaces.len(), fresh_workspaces.len());
            assert_eq!(coefficients.len(), total_coefficients);
            assert!(budget.live_bytes() <= cap);
        });
    }

    #[test]
    fn ht_mixed_shapes_grow_componentwise_then_reuse_warm_reservations() {
        let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        pool.install(|| {
            let mut workspaces = Vec::new();
            let mut structural = 0;
            for (dimensions, expect_growth) in [
                ([(64, 32); 4], true),
                ([(32, 64); 4], true),
                ([(64, 32); 4], false),
            ] {
                let mut budget = DecodeAllocationBudget::from_live_bytes(structural).unwrap();
                let (_, growths) = prepare_ht_task_workspaces(
                    &ht_pending(&dimensions),
                    &mut workspaces,
                    &mut structural,
                    &mut budget,
                )
                .unwrap();
                assert_eq!(growths != 0, expect_growth);
                let bytes_before_decode = super::ht_task_workspace_bytes(&workspaces).unwrap();
                for slot in &mut workspaces {
                    for &(width, height) in &dimensions {
                        slot.workspace.prepare(width, height).unwrap();
                    }
                }
                assert_eq!(
                    super::ht_task_workspace_bytes(&workspaces).unwrap(),
                    bytes_before_decode
                );
            }
            assert!(workspaces
                .iter()
                .all(|slot| { (slot.prepared_width, slot.prepared_height) == (64, 64) }));
        });
    }

    #[test]
    fn ht_failed_growth_rolls_back_then_retry_evicts_and_rebuilds_exactly() {
        let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        pool.install(|| {
            let mut workspaces = Vec::new();
            let mut structural = 0;
            let mut initial_budget = DecodeAllocationBudget::from_live_bytes(0).unwrap();
            prepare_ht_task_workspaces(
                &ht_pending(&[(64, 16); 4]),
                &mut workspaces,
                &mut structural,
                &mut initial_budget,
            )
            .unwrap();
            let old_bytes = structural;
            let old_dimensions = workspaces
                .iter()
                .map(|slot| (slot.prepared_width, slot.prepared_height))
                .collect::<Vec<_>>();
            let plan = balanced_task_plan(4, 1).unwrap();
            let mut rollback_budget =
                DecodeAllocationBudget::from_live_bytes_with_cap(old_bytes, old_bytes).unwrap();

            try_prepare_ht_task_workspaces(
                &ht_pending(&[(16, 64); 4]),
                plan,
                &mut workspaces,
                &mut structural,
                &mut rollback_budget,
            )
            .expect_err("replacement peak exceeds the retained-bank cap");
            assert_eq!(structural, old_bytes);
            assert_eq!(rollback_budget.live_bytes(), old_bytes);
            assert_eq!(
                workspaces
                    .iter()
                    .map(|slot| (slot.prepared_width, slot.prepared_height))
                    .collect::<Vec<_>>(),
                old_dimensions
            );

            let mut fresh_workspace = crate::HtCodeBlockDecodeWorkspace::default();
            fresh_workspace.reserve(16, 64).unwrap();
            let mut fresh_metadata = Vec::<super::HtTaskWorkspace>::new();
            try_reserve_decode_elements(&mut fresh_metadata, 2).unwrap();
            let fresh_cap = fresh_metadata.capacity()
                * core::mem::size_of::<super::HtTaskWorkspace>()
                + 2 * fresh_workspace.allocated_bytes().unwrap();
            assert!(old_bytes <= fresh_cap);
            let mut retry_budget =
                DecodeAllocationBudget::from_live_bytes_with_cap(old_bytes, fresh_cap).unwrap();
            let (plan, growths) = prepare_ht_task_workspaces(
                &ht_pending(&[(16, 64); 4]),
                &mut workspaces,
                &mut structural,
                &mut retry_budget,
            )
            .expect("evicting the retained bank lets the fresh shape fit");

            assert_eq!(plan.active_workspaces, 2);
            assert_eq!(growths, 2);
            assert_eq!(workspaces.len(), 2);
            assert!(workspaces
                .iter()
                .all(|slot| { (slot.prepared_width, slot.prepared_height) == (16, 64) }));
            assert_eq!(
                super::ht_task_workspace_bytes(&workspaces).unwrap(),
                fresh_cap
            );
            assert_eq!(structural, fresh_cap);
            assert_eq!(retry_budget.live_bytes(), fresh_cap);
        });
    }

    #[test]
    fn ht_entropy_error_leaves_workspace_bank_reusable_for_valid_blocks() {
        let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        pool.install(|| {
            let parameters = HtParallelParameters {
                strict: true,
                num_bitplanes: 1,
                roi_shift: 0,
                stripe_causal: false,
                dequantization_step: 1.0,
                irreversible_midpoint: false,
            };
            let mut malformed = ht_pending(&[(16, 16); 4]);
            malformed[2].combined.cleanup_length = 1;
            malformed[2].number_of_coding_passes = 1;
            let sub_band = test_sub_band(16, 16);
            let mut workspaces = Vec::new();
            let mut coefficient_slab = Vec::new();
            let mut structural = 0;
            let mut maximum_tasks = 0;
            let mut workspace_growths = 0;
            let mut coefficient_stats = super::super::super::ParallelCoefficientStats::default();
            let mut malformed_budget = DecodeAllocationBudget::from_live_bytes(0).unwrap();

            let malformed_result = decode_ht_sub_band_blocks_parallel(
                &malformed,
                &sub_band,
                parameters,
                &mut workspaces,
                &mut coefficient_slab,
                &mut structural,
                &mut maximum_tasks,
                &mut workspace_growths,
                &mut coefficient_stats,
                &mut malformed_budget,
            );
            assert!(
                malformed_result.is_err(),
                "truncated cleanup payload must fail after workspace activation"
            );
            let retained = super::ht_task_workspace_bytes(&workspaces).unwrap();
            let retained_coefficients = coefficient_slab.capacity() * size_of::<f32>();
            assert_eq!(retained + retained_coefficients, structural);
            let retained_initialized_coefficients = coefficient_slab.len();
            coefficient_slab.fill(7.0);

            let mut valid_budget = DecodeAllocationBudget::from_live_bytes(structural).unwrap();
            workspace_growths = 0;
            let decoded = decode_ht_sub_band_blocks_parallel(
                &ht_pending(&[(8, 8); 4]),
                &sub_band,
                parameters,
                &mut workspaces,
                &mut coefficient_slab,
                &mut structural,
                &mut maximum_tasks,
                &mut workspace_growths,
                &mut coefficient_stats,
                &mut valid_budget,
            )
            .expect("zero-pass blocks decode after the entropy error");

            assert_eq!(workspace_growths, 0);
            assert_eq!(
                super::ht_task_workspace_bytes(&workspaces).unwrap(),
                retained
            );
            assert!(decoded
                .iter()
                .all(|block| block.coefficients.iter().all(|&value| value == 0.0)));
            drop(decoded);
            assert_eq!(coefficient_slab.len(), retained_initialized_coefficients);
            assert!(coefficient_slab[4 * 8 * 8..]
                .iter()
                .all(|&value| value.to_bits() == 7.0_f32.to_bits()));
        });
    }

    #[test]
    fn coefficient_total_rejects_overflow_before_output_allocation() {
        let sub_band = SubBand {
            sub_band_type: SubBandType::LowLow,
            rect: IntRect::from_xywh(0, 0, u32::MAX, u32::MAX),
            precincts: 0..0,
            coefficients: 0..0,
        };
        let error = validate_pending_blocks(
            [(0, 0, u32::MAX, u32::MAX), (0, 0, u32::MAX, u32::MAX)].into_iter(),
            &sub_band,
        )
        .expect_err("aggregate coefficient count must reject overflow");
        assert!(matches!(
            error,
            DecodeError::Validation(ValidationError::ImageTooLarge)
        ));
    }

    #[test]
    fn descriptor_preflight_rejects_a_late_invalid_block_before_touching_retained_slab() {
        let mut pending = ht_pending(&[(2, 2), (2, 2)]);
        pending[1].output_x = 4;
        let sub_band = test_sub_band(5, 3);
        let mut coefficients = vec![7.0; 16];
        let retained_capacity = coefficients.capacity();
        let retained_bytes = retained_capacity * size_of::<f32>();
        let mut structural = retained_bytes;
        let mut stats = super::super::super::ParallelCoefficientStats::default();
        let mut budget = DecodeAllocationBudget::from_live_bytes(retained_bytes).unwrap();

        let result = prepare_ht_outputs(
            &pending,
            &sub_band,
            &mut coefficients,
            &mut structural,
            &mut stats,
            &mut budget,
        );

        assert!(matches!(
            result,
            Err(DecodeError::Decoding(
                crate::DecodingError::CodeBlockDecodeFailure
            ))
        ));
        assert_eq!(coefficients.capacity(), retained_capacity);
        assert!(coefficients
            .iter()
            .all(|&value| value.to_bits() == 7.0_f32.to_bits()));
        assert_eq!(structural, retained_bytes);
        assert_eq!(budget.live_bytes(), retained_bytes);
        assert_eq!(
            stats,
            super::super::super::ParallelCoefficientStats::default()
        );
    }

    #[test]
    fn descriptor_reservation_releases_an_oversized_retained_slab_under_a_tight_cap() {
        let pending = ht_pending(&[(1, 1); 4]);
        let sub_band = test_sub_band(1, 1);
        let mut coefficients = vec![7.0; 16];
        let old_bytes = coefficients.capacity() * size_of::<f32>();
        let mut descriptor_probe: Vec<super::DecodedBlock<'static>> = Vec::new();
        try_reserve_decode_elements(&mut descriptor_probe, pending.len()).unwrap();
        let descriptor_bytes = descriptor_probe.capacity() * size_of::<super::DecodedBlock<'_>>();
        let mut slab_probe = Vec::new();
        try_resize_decode_elements(&mut slab_probe, pending.len(), 0.0).unwrap();
        let fresh_slab_bytes = slab_probe.capacity() * size_of::<f32>();
        let cap = descriptor_bytes + fresh_slab_bytes;
        assert!(old_bytes + descriptor_bytes > cap);
        let mut structural = old_bytes;
        let mut stats = super::super::super::ParallelCoefficientStats::default();
        let mut budget = DecodeAllocationBudget::from_live_bytes_with_cap(old_bytes, cap).unwrap();

        let (mut decoded, _) = prepare_ht_outputs(
            &pending,
            &sub_band,
            &mut coefficients,
            &mut structural,
            &mut stats,
            &mut budget,
        )
        .expect("fresh descriptors and smaller slab fit after retained slab eviction");
        install_decoded_blocks(
            &mut decoded,
            &mut coefficients,
            pending.len(),
            pending.iter().map(|pending| {
                (
                    pending.output_x,
                    pending.output_y,
                    pending.width,
                    pending.height,
                )
            }),
        )
        .unwrap();

        assert_eq!(decoded.len(), pending.len());
        assert!(decoded
            .iter()
            .all(|block| block.coefficients.iter().all(|&value| value == 0.0)));
        drop(decoded);
        assert_eq!(structural, fresh_slab_bytes);
        assert_eq!(budget.live_bytes(), descriptor_bytes + structural);
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.peak_live_bytes, old_bytes.max(fresh_slab_bytes));
    }

    #[test]
    fn coefficient_slab_growth_releases_old_capacity_before_a_tight_cap_allocation() {
        let mut coefficients = vec![1.0; 16];
        let old_bytes = coefficients.capacity() * size_of::<f32>();
        let mut fresh = Vec::new();
        try_resize_decode_elements(&mut fresh, 64, 0.0).unwrap();
        let fresh_bytes = fresh.capacity() * size_of::<f32>();
        let mut structural = old_bytes;
        let mut stats = super::super::super::ParallelCoefficientStats::default();
        let mut budget = DecodeAllocationBudget::from_live_bytes_with_cap(old_bytes, fresh_bytes)
            .expect("retained slab fits the fresh allocation cap");

        prepare_coefficient_slab(
            &mut coefficients,
            64,
            &mut structural,
            &mut stats,
            &mut budget,
        )
        .expect("old optional scratch is released before allocating the larger slab");

        assert_eq!(structural, coefficients.capacity() * size_of::<f32>());
        assert_eq!(budget.live_bytes(), structural);
        assert!(coefficients.iter().all(|&value| value == 0.0));
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.peak_live_bytes, old_bytes.max(structural));

        let error = prepare_coefficient_slab(
            &mut coefficients,
            128,
            &mut structural,
            &mut stats,
            &mut budget,
        )
        .expect_err("a slab exceeding the cap must fail after releasing optional scratch");
        assert!(matches!(
            error,
            DecodeError::Validation(ValidationError::ImageTooLarge)
        ));
        assert_eq!(coefficients.capacity(), 0);
        assert_eq!((structural, budget.live_bytes()), (0, 0));
        assert_eq!(stats.allocations, 1);

        prepare_coefficient_slab(
            &mut coefficients,
            64,
            &mut structural,
            &mut stats,
            &mut budget,
        )
        .expect("a fitting slab can be prepared after capacity failure");
        assert_eq!(budget.live_bytes(), structural);
        assert!(coefficients.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn classic_workspace_cap_can_release_an_oversized_coefficient_slab() {
        let pending = (0..4)
            .map(|index| super::PendingClassicBlock {
                combined_data: Vec::new(),
                segments: Vec::new(),
                output_x: index * 32,
                output_y: 0,
                width: 32,
                height: 32,
                missing_bit_planes: 0,
                number_of_coding_passes: 0,
            })
            .collect::<Vec<_>>();
        let sub_band = test_sub_band(128, 32);
        let parameters = super::ClassicParallelParameters {
            sub_band_type: crate::J2kSubBandType::LowLow,
            style: crate::J2kCodeBlockStyle {
                selective_arithmetic_coding_bypass: false,
                reset_context_probabilities: false,
                termination_on_each_pass: false,
                vertically_causal_context: false,
                segmentation_symbols: false,
            },
            strict: true,
            total_bitplanes: 1,
            roi_shift: 0,
            dequantization_step: 1.0,
            irreversible_midpoint: false,
        };
        let mut fresh_slab = Vec::new();
        let mut fresh_structural = 0;
        let mut fresh_stats = super::super::super::ParallelCoefficientStats::default();
        let mut fresh_budget = DecodeAllocationBudget::from_live_bytes(0).unwrap();
        let fresh = super::decode_classic_sub_band_blocks_parallel(
            &pending,
            &sub_band,
            parameters,
            &mut fresh_slab,
            &mut fresh_structural,
            &mut fresh_stats,
            &mut fresh_budget,
        )
        .expect("fresh coefficient and entropy buffers fit");
        assert!(fresh
            .iter()
            .all(|block| block.coefficients.iter().all(|&v| v == 0.0)));
        let cap = fresh_budget.live_bytes();
        drop(fresh);

        let mut retained_slab = vec![7.0; fresh_slab.capacity() * 2];
        let mut structural = retained_slab.capacity() * size_of::<f32>();
        assert!(structural < cap);
        let mut budget = DecodeAllocationBudget::from_live_bytes_with_cap(structural, cap).unwrap();
        let mut stats = super::super::super::ParallelCoefficientStats::default();
        let decoded = super::decode_classic_sub_band_blocks_parallel(
            &pending,
            &sub_band,
            parameters,
            &mut retained_slab,
            &mut structural,
            &mut stats,
            &mut budget,
        )
        .expect("oversized optional coefficients must not crowd out fitting entropy workspaces");
        assert!(decoded
            .iter()
            .all(|block| block.coefficients.iter().all(|&v| v == 0.0)));
        assert!(budget.live_bytes() <= cap);
    }
}
