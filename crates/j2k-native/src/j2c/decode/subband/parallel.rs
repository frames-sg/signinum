// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible parallel code-block output assembly.

use super::pending::{PendingClassicBlock, PendingHtBlock};
use super::{DecodeAllocationBudget, DecompositionStorage, SubBand};
use crate::error::{bail, DecodingError, Result, ValidationError};
use crate::j2c::bitplane::classic_decode_workspace_bytes;
use crate::j2c::decode::workspace::{
    classic_task_workspace_bytes, ht_task_workspace_bytes, ClassicTaskWorkspace, HtTaskWorkspace,
};
use crate::j2c::ht_block_decode::ht_decode_workspace_bytes;
use crate::scalar::{
    decode_ht_code_block_scalar_with_workspace_midpoint,
    decode_j2k_code_block_scalar_with_workspace_midpoint,
};
use crate::{
    decode_ht_code_block_scalar_with_workspace, decode_j2k_code_block_scalar_with_workspace,
    try_reserve_decode_elements, try_resize_decode_elements, HtCodeBlockDecodeJob,
    HtCodeBlockDecodeWorkspace, J2kCodeBlockDecodeJob, J2kCodeBlockDecodeWorkspace,
    J2kCodeBlockStyle, J2kSubBandType,
};
use alloc::vec::Vec;
use rayon::prelude::*;

#[derive(Clone, Copy)]
struct PreparedTaskPlan {
    chunk_size: usize,
    active_workspaces: usize,
}

pub(crate) struct DecodedClassicBlock {
    pub(crate) output_x: u32,
    pub(crate) output_y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) coefficients: Vec<f32>,
}

pub(crate) struct DecodedHtBlock {
    pub(crate) output_x: u32,
    pub(crate) output_y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) coefficients: Vec<f32>,
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

trait DecodedSubBandBlock {
    fn output_x(&self) -> u32;
    fn output_y(&self) -> u32;
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn coefficients(&self) -> &[f32];
}

impl DecodedSubBandBlock for DecodedClassicBlock {
    fn output_x(&self) -> u32 {
        self.output_x
    }

    fn output_y(&self) -> u32 {
        self.output_y
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }
}

impl DecodedSubBandBlock for DecodedHtBlock {
    fn output_x(&self) -> u32 {
        self.output_x
    }

    fn output_y(&self) -> u32 {
        self.output_y
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }
}

pub(super) fn decode_classic_sub_band_blocks_parallel(
    pending_blocks: &[PendingClassicBlock],
    parameters: ClassicParallelParameters,
    workspaces: &mut Vec<ClassicTaskWorkspace>,
    structural_workspace_bytes: &mut usize,
    #[cfg(test)] maximum_tasks: &mut usize,
    #[cfg(test)] workspace_growths: &mut usize,
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<DecodedClassicBlock>> {
    let mut decoded_blocks = preallocate_classic_outputs(pending_blocks, budget)?;
    let (plan, growths) = prepare_classic_task_workspaces(
        pending_blocks,
        workspaces,
        structural_workspace_bytes,
        budget,
    )?;
    #[cfg(test)]
    {
        *maximum_tasks = (*maximum_tasks).max(plan.active_workspaces);
        *workspace_growths = workspace_growths.saturating_add(growths);
    }
    #[cfg(not(test))]
    let _ = growths;
    decoded_blocks
        .par_chunks_mut(plan.chunk_size)
        .zip(pending_blocks.par_chunks(plan.chunk_size))
        .zip(workspaces[..plan.active_workspaces].par_iter_mut())
        .try_for_each(|((decoded_chunk, pending_chunk), slot)| -> Result<()> {
            for (decoded, pending) in decoded_chunk.iter_mut().zip(pending_chunk) {
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
                    &mut decoded.coefficients,
                    &mut slot.workspace,
                )?;
            }
            Ok(())
        })?;
    Ok(decoded_blocks)
}

fn preallocate_classic_outputs(
    pending_blocks: &[PendingClassicBlock],
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<DecodedClassicBlock>> {
    let total_coefficients = pending_coefficient_count(
        pending_blocks
            .iter()
            .map(|pending| (pending.width, pending.height)),
    )?;
    budget.include_elements::<f32>(total_coefficients)?;

    let mut decoded_blocks = Vec::new();
    budget.reserve_new(&mut decoded_blocks, pending_blocks.len())?;
    for pending in pending_blocks {
        let coefficient_count = block_coefficient_count(pending.width, pending.height)?;
        let mut coefficients = Vec::new();
        try_resize_decode_elements(&mut coefficients, coefficient_count, 0.0)?;
        budget.include_capacity_overage::<f32>(coefficient_count, coefficients.capacity())?;
        decoded_blocks.push(DecodedClassicBlock {
            output_x: pending.output_x,
            output_y: pending.output_y,
            width: pending.width,
            height: pending.height,
            coefficients,
        });
    }
    Ok(decoded_blocks)
}

fn prepare_classic_task_workspaces(
    pending_blocks: &[PendingClassicBlock],
    workspaces: &mut Vec<ClassicTaskWorkspace>,
    structural_workspace_bytes: &mut usize,
    budget: &mut DecodeAllocationBudget,
) -> Result<(PreparedTaskPlan, usize)> {
    let initial_tasks = pending_blocks.len().min(rayon::current_num_threads());
    let mut requested_tasks = initial_tasks;
    let mut evicted_retained_bank = false;
    loop {
        let plan = uniform_task_plan(pending_blocks.len(), requested_tasks)?;
        match try_prepare_classic_task_workspaces(
            pending_blocks,
            plan,
            workspaces,
            structural_workspace_bytes,
            budget,
        ) {
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
                release_classic_task_bank(workspaces, structural_workspace_bytes, budget)?;
                requested_tasks = initial_tasks;
                evicted_retained_bank = true;
            }
            result => return result.map(|growths| (plan, growths)),
        }
    }
}

pub(super) fn decode_ht_sub_band_blocks_parallel(
    pending_blocks: &[PendingHtBlock],
    parameters: HtParallelParameters,
    workspaces: &mut Vec<HtTaskWorkspace>,
    structural_workspace_bytes: &mut usize,
    #[cfg(test)] maximum_tasks: &mut usize,
    #[cfg(test)] workspace_growths: &mut usize,
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<DecodedHtBlock>> {
    let mut decoded_blocks = preallocate_ht_outputs(pending_blocks, budget)?;
    let (plan, growths) = prepare_ht_task_workspaces(
        pending_blocks,
        workspaces,
        structural_workspace_bytes,
        budget,
    )?;
    #[cfg(test)]
    {
        *maximum_tasks = (*maximum_tasks).max(plan.active_workspaces);
        *workspace_growths = workspace_growths.saturating_add(growths);
    }
    #[cfg(not(test))]
    let _ = growths;
    decoded_blocks
        .par_chunks_mut(plan.chunk_size)
        .zip(pending_blocks.par_chunks(plan.chunk_size))
        .zip(workspaces[..plan.active_workspaces].par_iter_mut())
        .try_for_each(|((decoded_chunk, pending_chunk), slot)| -> Result<()> {
            for (decoded, pending) in decoded_chunk.iter_mut().zip(pending_chunk) {
                initialize_reserved_coefficients(
                    &mut decoded.coefficients,
                    block_coefficient_count(pending.width, pending.height)?,
                )?;
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
                    &mut decoded.coefficients,
                    &mut slot.workspace,
                )?;
            }
            Ok(())
        })?;
    Ok(decoded_blocks)
}

fn preallocate_ht_outputs(
    pending_blocks: &[PendingHtBlock],
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<DecodedHtBlock>> {
    let total_coefficients = pending_coefficient_count(
        pending_blocks
            .iter()
            .map(|pending| (pending.width, pending.height)),
    )?;
    budget.include_elements::<f32>(total_coefficients)?;

    let mut decoded_blocks = Vec::new();
    budget.reserve_new(&mut decoded_blocks, pending_blocks.len())?;
    for pending in pending_blocks {
        let coefficient_count = block_coefficient_count(pending.width, pending.height)?;
        let mut coefficients = Vec::new();
        try_reserve_decode_elements(&mut coefficients, coefficient_count)?;
        budget.include_capacity_overage::<f32>(coefficient_count, coefficients.capacity())?;
        decoded_blocks.push(DecodedHtBlock {
            output_x: pending.output_x,
            output_y: pending.output_y,
            width: pending.width,
            height: pending.height,
            coefficients,
        });
    }
    Ok(decoded_blocks)
}

fn prepare_ht_task_workspaces(
    pending_blocks: &[PendingHtBlock],
    workspaces: &mut Vec<HtTaskWorkspace>,
    structural_workspace_bytes: &mut usize,
    budget: &mut DecodeAllocationBudget,
) -> Result<(PreparedTaskPlan, usize)> {
    let initial_tasks = pending_blocks.len().min(rayon::current_num_threads());
    let mut requested_tasks = initial_tasks;
    let mut evicted_retained_bank = false;
    loop {
        let plan = uniform_task_plan(pending_blocks.len(), requested_tasks)?;
        match try_prepare_ht_task_workspaces(
            pending_blocks,
            plan,
            workspaces,
            structural_workspace_bytes,
            budget,
        ) {
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

fn uniform_task_plan(job_count: usize, requested_tasks: usize) -> Result<PreparedTaskPlan> {
    if job_count == 0 {
        return Ok(PreparedTaskPlan {
            chunk_size: 1,
            active_workspaces: 0,
        });
    }
    if requested_tasks == 0 {
        return Err(DecodingError::CodeBlockDecodeFailure.into());
    }
    let chunk_size = job_count.div_ceil(requested_tasks);
    Ok(PreparedTaskPlan {
        chunk_size,
        active_workspaces: job_count.div_ceil(chunk_size),
    })
}

fn release_classic_task_bank(
    workspaces: &mut Vec<ClassicTaskWorkspace>,
    structural_workspace_bytes: &mut usize,
    budget: &mut DecodeAllocationBudget,
) -> Result<()> {
    let released = classic_task_workspace_bytes(workspaces)?;
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
fn try_prepare_classic_task_workspaces(
    pending_blocks: &[PendingClassicBlock],
    plan: PreparedTaskPlan,
    workspaces: &mut Vec<ClassicTaskWorkspace>,
    structural_workspace_bytes: &mut usize,
    budget: &mut DecodeAllocationBudget,
) -> Result<usize> {
    let old_bank_bytes = classic_task_workspace_bytes(workspaces)?;
    let mut trial = *budget;
    if workspaces.capacity() < plan.active_workspaces {
        let mut replacement = Vec::new();
        trial.reserve_new(&mut replacement, plan.active_workspaces)?;
        for (index, chunk) in pending_blocks.chunks(plan.chunk_size).enumerate() {
            let (required_width, required_height) = classic_chunk_dimensions(chunk);
            let (width, height) =
                workspaces
                    .get(index)
                    .map_or((required_width, required_height), |old| {
                        (
                            required_width.max(old.prepared_width),
                            required_height.max(old.prepared_height),
                        )
                    });
            let planned = classic_decode_workspace_bytes(width, height)?;
            trial.include_bytes(planned)?;
            let mut workspace = J2kCodeBlockDecodeWorkspace::default();
            workspace.prepare(width, height)?;
            charge_actual_workspace(&mut trial, planned, workspace.allocated_bytes()?)?;
            replacement.push(ClassicTaskWorkspace {
                workspace,
                prepared_width: width,
                prepared_height: height,
            });
        }
        let new_bank_bytes = classic_task_workspace_bytes(&replacement)?;
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

    let replacement_count = pending_blocks
        .chunks(plan.chunk_size)
        .enumerate()
        .filter(|&(index, chunk)| {
            let (width, height) = classic_chunk_dimensions(chunk);
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
        .checked_mul(core::mem::size_of::<(usize, ClassicTaskWorkspace)>())
        .ok_or(ValidationError::ImageTooLarge)?;
    let mut replaced_workspace_bytes = 0usize;
    #[cfg(test)]
    let mut growths = 0usize;
    for (index, chunk) in pending_blocks.chunks(plan.chunk_size).enumerate() {
        let (required_width, required_height) = classic_chunk_dimensions(chunk);
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
        let planned = classic_decode_workspace_bytes(width, height)?;
        trial.include_bytes(planned)?;
        let mut workspace = J2kCodeBlockDecodeWorkspace::default();
        workspace.prepare(width, height)?;
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
            ClassicTaskWorkspace {
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
        for (index, chunk) in pending_blocks.chunks(plan.chunk_size).enumerate() {
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

    let replacement_count = pending_blocks
        .chunks(plan.chunk_size)
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
    for (index, chunk) in pending_blocks.chunks(plan.chunk_size).enumerate() {
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

fn classic_chunk_dimensions(chunk: &[PendingClassicBlock]) -> (u32, u32) {
    chunk.iter().fold((0, 0), |(width, height), pending| {
        (width.max(pending.width), height.max(pending.height))
    })
}

fn ht_chunk_dimensions(chunk: &[PendingHtBlock]) -> (u32, u32) {
    chunk.iter().fold((0, 0), |(width, height), pending| {
        (width.max(pending.width), height.max(pending.height))
    })
}

fn initialize_reserved_coefficients(coefficients: &mut Vec<f32>, len: usize) -> Result<()> {
    if coefficients.capacity() < len {
        return Err(DecodingError::CodeBlockDecodeFailure.into());
    }
    coefficients.resize(len, 0.0);
    Ok(())
}

fn pending_coefficient_count(mut dimensions: impl Iterator<Item = (u32, u32)>) -> Result<usize> {
    dimensions.try_fold(0_usize, |total, (width, height)| {
        total
            .checked_add(block_coefficient_count(width, height)?)
            .ok_or(ValidationError::ImageTooLarge.into())
    })
}

fn block_coefficient_count(width: u32, height: u32) -> Result<usize> {
    (width as usize)
        .checked_mul(height as usize)
        .ok_or(ValidationError::ImageTooLarge.into())
}

pub(crate) fn copy_decoded_classic_blocks_to_sub_band(
    decoded_blocks: &[DecodedClassicBlock],
    sub_band: &SubBand,
    storage: &mut DecompositionStorage<'_>,
) -> Result<()> {
    copy_decoded_blocks_to_sub_band(decoded_blocks, sub_band, storage)
}

pub(crate) fn copy_decoded_ht_blocks_to_sub_band(
    decoded_blocks: &[DecodedHtBlock],
    sub_band: &SubBand,
    storage: &mut DecompositionStorage<'_>,
) -> Result<()> {
    copy_decoded_blocks_to_sub_band(decoded_blocks, sub_band, storage)
}

fn copy_decoded_blocks_to_sub_band<B: DecodedSubBandBlock>(
    decoded_blocks: &[B],
    sub_band: &SubBand,
    storage: &mut DecompositionStorage<'_>,
) -> Result<()> {
    let sub_band_width = sub_band.rect.width() as usize;
    let base_store = &mut storage.coefficients[sub_band.coefficients.clone()];
    for block in decoded_blocks {
        let output_x = block.output_x();
        let output_y = block.output_y();
        let width = block.width();
        let height = block.height();
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
            base_store[dst_start..dst_end]
                .copy_from_slice(&block.coefficients()[src_start..src_end]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        decode_classic_sub_band_blocks_parallel, initialize_reserved_coefficients,
        pending_coefficient_count, prepare_classic_task_workspaces, prepare_ht_task_workspaces,
        try_prepare_classic_task_workspaces, try_prepare_ht_task_workspaces, uniform_task_plan,
        ClassicParallelParameters,
    };
    use crate::error::{DecodeError, ValidationError};
    use crate::j2c::decode::subband::pending::{PendingClassicBlock, PendingHtBlock};
    use crate::j2c::decode::DecodeAllocationBudget;
    use crate::j2c::ht_block_decode::CombinedCodeBlockData;
    use crate::try_reserve_decode_elements;
    use crate::{J2kCodeBlockStyle, J2kSubBandType};
    use alloc::vec::Vec;
    use rayon::ThreadPoolBuilder;

    fn classic_pending(dimensions: &[(u32, u32)]) -> Vec<PendingClassicBlock> {
        dimensions
            .iter()
            .map(|&(width, height)| PendingClassicBlock {
                combined_data: Vec::new(),
                segments: Vec::new(),
                output_x: 0,
                output_y: 0,
                width,
                height,
                missing_bit_planes: 0,
                number_of_coding_passes: 0,
            })
            .collect()
    }

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

    #[test]
    fn classic_parallel_workspace_count_is_bounded_by_the_current_pool() {
        let pool = ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("two-thread test pool");
        pool.install(|| {
            let pending = classic_pending(&[(32, 32); 8]);
            let mut budget = DecodeAllocationBudget::from_live_bytes(0).expect("empty budget");
            let mut workspaces = Vec::new();
            let mut structural_workspace_bytes = 0;

            let (plan, _) = prepare_classic_task_workspaces(
                &pending,
                &mut workspaces,
                &mut structural_workspace_bytes,
                &mut budget,
            )
            .expect("prepare classic workspaces");

            assert!(workspaces.len() <= rayon::current_num_threads());
            assert_eq!(workspaces.len(), plan.active_workspaces);
        });
    }

    #[test]
    fn task_count_is_bounded_by_jobs_when_the_pool_is_larger() {
        let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        pool.install(|| {
            let pending = classic_pending(&[(32, 32); 2]);
            let mut budget = DecodeAllocationBudget::from_live_bytes(0).unwrap();
            let mut workspaces = Vec::new();
            let mut structural = 0;
            let (plan, _) = prepare_classic_task_workspaces(
                &pending,
                &mut workspaces,
                &mut structural,
                &mut budget,
            )
            .unwrap();

            assert_eq!(plan.active_workspaces, 2);
            assert_eq!(workspaces.len(), 2);
        });
    }

    #[test]
    fn empty_parallel_plans_need_no_workspace() {
        let mut classic = Vec::new();
        let mut ht = Vec::new();
        let mut classic_structural = 0;
        let mut ht_structural = 0;
        let mut classic_budget = DecodeAllocationBudget::from_live_bytes(0).unwrap();
        let mut ht_budget = DecodeAllocationBudget::from_live_bytes(0).unwrap();

        let (classic_plan, classic_growths) = prepare_classic_task_workspaces(
            &[],
            &mut classic,
            &mut classic_structural,
            &mut classic_budget,
        )
        .unwrap();
        let (ht_plan, ht_growths) =
            prepare_ht_task_workspaces(&[], &mut ht, &mut ht_structural, &mut ht_budget).unwrap();

        assert_eq!((classic_plan.active_workspaces, classic_growths), (0, 0));
        assert_eq!((ht_plan.active_workspaces, ht_growths), (0, 0));
        assert!(classic.is_empty() && ht.is_empty());
    }

    #[test]
    fn mixed_shapes_grow_componentwise_once_for_classic_and_ht() {
        let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        pool.install(|| {
            let mut classic = Vec::new();
            let mut ht = Vec::new();
            let mut classic_structural = 0;
            let mut ht_structural = 0;
            for (dimensions, expect_growth) in [
                ([(64, 32); 4], true),
                ([(32, 64); 4], true),
                ([(64, 32); 4], false),
            ] {
                let mut classic_budget =
                    DecodeAllocationBudget::from_live_bytes(classic_structural).unwrap();
                let (_, growths) = prepare_classic_task_workspaces(
                    &classic_pending(&dimensions),
                    &mut classic,
                    &mut classic_structural,
                    &mut classic_budget,
                )
                .unwrap();
                assert_eq!(growths != 0, expect_growth);
                let classic_bytes_before_worker =
                    super::classic_task_workspace_bytes(&classic).unwrap();
                for &(width, height) in &dimensions {
                    classic[0].workspace.prepare(width, height).unwrap();
                }
                assert_eq!(
                    super::classic_task_workspace_bytes(&classic).unwrap(),
                    classic_bytes_before_worker
                );

                let mut ht_budget = DecodeAllocationBudget::from_live_bytes(ht_structural).unwrap();
                let (_, growths) = prepare_ht_task_workspaces(
                    &ht_pending(&dimensions),
                    &mut ht,
                    &mut ht_structural,
                    &mut ht_budget,
                )
                .unwrap();
                assert_eq!(growths != 0, expect_growth);
                let ht_bytes_before_worker = super::ht_task_workspace_bytes(&ht).unwrap();
                for &(width, height) in &dimensions {
                    ht[0].workspace.prepare(width, height).unwrap();
                }
                assert_eq!(
                    super::ht_task_workspace_bytes(&ht).unwrap(),
                    ht_bytes_before_worker
                );
            }
            assert_eq!(
                (classic[0].prepared_width, classic[0].prepared_height),
                (64, 64)
            );
            assert_eq!((ht[0].prepared_width, ht[0].prepared_height), (64, 64));
        });
    }

    #[test]
    fn capacity_retry_uses_an_unpoisoned_trial_budget() {
        let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        pool.install(|| {
            let pending = classic_pending(&[(32, 32); 8]);
            let mut probe = crate::J2kCodeBlockDecodeWorkspace::default();
            probe.prepare(32, 32).unwrap();
            let one_task_cap = core::mem::size_of::<super::ClassicTaskWorkspace>()
                + probe.allocated_bytes().unwrap();
            let mut budget =
                DecodeAllocationBudget::from_live_bytes_with_cap(0, one_task_cap).unwrap();
            let mut workspaces = Vec::new();
            let mut structural = 0;

            let (plan, growths) = prepare_classic_task_workspaces(
                &pending,
                &mut workspaces,
                &mut structural,
                &mut budget,
            )
            .expect("one task fits after the two-task trial exceeds the cap");

            assert_eq!(plan.active_workspaces, 1);
            assert_eq!(growths, 1);
            assert_eq!(workspaces.len(), 1);
            assert_eq!(budget.live_bytes(), structural);
            assert_eq!(structural, one_task_cap);
        });
    }

    #[test]
    fn failed_staged_growth_preserves_the_retained_bank_and_ledger() {
        let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        pool.install(|| {
            let mut workspaces = Vec::new();
            let mut structural = 0;
            let mut initial_budget = DecodeAllocationBudget::from_live_bytes(0).unwrap();
            prepare_classic_task_workspaces(
                &classic_pending(&[(16, 32); 4]),
                &mut workspaces,
                &mut structural,
                &mut initial_budget,
            )
            .unwrap();
            let old_bytes = structural;
            let old_dimensions = (workspaces[0].prepared_width, workspaces[0].prepared_height);
            let mut tight_budget =
                DecodeAllocationBudget::from_live_bytes_with_cap(old_bytes, old_bytes).unwrap();
            let plan = uniform_task_plan(4, 1).unwrap();

            try_prepare_classic_task_workspaces(
                &classic_pending(&[(64, 64); 4]),
                plan,
                &mut workspaces,
                &mut structural,
                &mut tight_budget,
            )
            .expect_err("replacement peak exceeds the exact retained cap");

            assert_eq!(structural, old_bytes);
            assert_eq!(tight_budget.live_bytes(), old_bytes);
            assert_eq!(
                (workspaces[0].prepared_width, workspaces[0].prepared_height),
                old_dimensions
            );
        });
    }

    #[test]
    fn ht_failed_growth_rolls_back_then_capacity_retry_evicts_and_rebuilds_exactly() {
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
            let old_dimensions = (workspaces[0].prepared_width, workspaces[0].prepared_height);
            let plan = uniform_task_plan(4, 1).unwrap();
            let mut rollback_budget =
                DecodeAllocationBudget::from_live_bytes_with_cap(old_bytes, old_bytes).unwrap();
            try_prepare_ht_task_workspaces(
                &ht_pending(&[(16, 64); 4]),
                plan,
                &mut workspaces,
                &mut structural,
                &mut rollback_budget,
            )
            .expect_err("old plus componentwise replacement exceeds the retained cap");
            assert_eq!(structural, old_bytes);
            assert_eq!(rollback_budget.live_bytes(), old_bytes);
            assert_eq!(
                (workspaces[0].prepared_width, workspaces[0].prepared_height),
                old_dimensions
            );

            let mut fresh_probe = crate::HtCodeBlockDecodeWorkspace::default();
            fresh_probe.reserve(16, 64).unwrap();
            let fresh_cap = core::mem::size_of::<super::HtTaskWorkspace>()
                + fresh_probe.allocated_bytes().unwrap();
            assert!(old_bytes <= fresh_cap);
            let mut retry_budget =
                DecodeAllocationBudget::from_live_bytes_with_cap(old_bytes, fresh_cap).unwrap();
            let (_, growths) = prepare_ht_task_workspaces(
                &ht_pending(&[(16, 64); 4]),
                &mut workspaces,
                &mut structural,
                &mut retry_budget,
            )
            .expect("evicting the orthogonal retained shape lets the fresh bank fit");

            assert_eq!(growths, 1);
            assert_eq!(
                (workspaces[0].prepared_width, workspaces[0].prepared_height),
                (16, 64)
            );
            assert_eq!(
                super::ht_task_workspace_bytes(&workspaces).unwrap(),
                fresh_cap
            );
            assert_eq!(structural, fresh_cap);
            assert_eq!(retry_budget.live_bytes(), fresh_cap);
        });
    }

    #[test]
    fn entropy_error_leaves_workspace_bank_reusable_for_a_valid_decode() {
        let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        pool.install(|| {
            let parameters = ClassicParallelParameters {
                sub_band_type: J2kSubBandType::LowLow,
                style: J2kCodeBlockStyle {
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
            let mut bad = classic_pending(&[(16, 16); 4]);
            for pending in &mut bad {
                pending.number_of_coding_passes = 1;
            }
            let mut workspaces = Vec::new();
            let mut structural = 0;
            let mut maximum_tasks = 0;
            let mut workspace_growths = 0;
            let mut bad_budget = DecodeAllocationBudget::from_live_bytes(0).unwrap();
            let bad_result = decode_classic_sub_band_blocks_parallel(
                &bad,
                parameters,
                &mut workspaces,
                &mut structural,
                &mut maximum_tasks,
                &mut workspace_growths,
                &mut bad_budget,
            );
            assert!(
                bad_result.is_err(),
                "missing arithmetic segment must fail after workspace activation"
            );
            let retained = super::classic_task_workspace_bytes(&workspaces).unwrap();
            assert_eq!(retained, structural);

            let valid = classic_pending(&[(16, 16); 4]);
            let mut valid_budget = DecodeAllocationBudget::from_live_bytes(structural).unwrap();
            workspace_growths = 0;
            let decoded = decode_classic_sub_band_blocks_parallel(
                &valid,
                parameters,
                &mut workspaces,
                &mut structural,
                &mut maximum_tasks,
                &mut workspace_growths,
                &mut valid_budget,
            )
            .expect("zero-pass blocks reset and decode after the entropy error");

            assert_eq!(workspace_growths, 0);
            assert_eq!(
                super::classic_task_workspace_bytes(&workspaces).unwrap(),
                retained
            );
            assert!(decoded
                .iter()
                .all(|block| block.coefficients.iter().all(|&value| value == 0.0)));
        });
    }

    #[test]
    fn coefficient_total_rejects_overflow_before_output_allocation() {
        let error =
            pending_coefficient_count([(u32::MAX, u32::MAX), (u32::MAX, u32::MAX)].into_iter())
                .expect_err("aggregate coefficient count must reject overflow");
        assert!(matches!(
            error,
            DecodeError::Validation(ValidationError::ImageTooLarge)
        ));
    }

    #[test]
    fn reserved_output_initialization_does_not_grow_allocation() {
        let mut coefficients = Vec::new();
        try_reserve_decode_elements(&mut coefficients, 64 * 64)
            .expect("coefficient reservation should succeed");
        let reserved_capacity = coefficients.capacity();

        initialize_reserved_coefficients(&mut coefficients, 64 * 64)
            .expect("reserved coefficients should initialize");

        assert_eq!(coefficients.len(), 64 * 64);
        assert_eq!(coefficients.capacity(), reserved_capacity);
    }

    #[test]
    fn unreserved_output_initialization_fails_without_allocating() {
        let mut coefficients = Vec::new();

        initialize_reserved_coefficients(&mut coefficients, 64 * 64)
            .expect_err("parallel initialization must not allocate");

        assert_eq!(coefficients.capacity(), 0);
        assert!(coefficients.is_empty());
    }
}
