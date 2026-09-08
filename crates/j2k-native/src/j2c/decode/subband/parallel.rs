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
mod tests;
