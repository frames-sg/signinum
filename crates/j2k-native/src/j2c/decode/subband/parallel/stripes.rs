// SPDX-License-Identifier: MIT OR Apache-2.0

//! Disjoint full-row HT output with rollback until the whole band succeeds.

use super::super::super::TileDecodeContext;
use super::{
    balanced_task_plan, bounded_parallel_task_count, decode_ht_code_block_scalar_with_workspace,
    decode_ht_code_block_scalar_with_workspace_midpoint, release_coefficient_slab,
    release_ht_task_bank, try_prepare_ht_workspace_chunks, DecodeAllocationBudget, DecodingError,
    DecompositionStorage, HtCodeBlockDecodeJob, HtCodeBlockDecodeWorkspace, HtParallelParameters,
    PendingHtBlock, PreparedTaskPlan, Result, SubBand, ValidationError,
};
use alloc::vec::Vec;
use rayon::prelude::*;

#[derive(Clone, Copy)]
struct Stripe {
    first: usize,
    end: usize,
    y: u32,
    height: u32,
}

fn visit_stripes(
    pending: &[PendingHtBlock],
    width: u32,
    height: u32,
    mut visit: impl FnMut(Stripe),
) -> Option<usize> {
    if width == 0 || height == 0 || pending.is_empty() {
        return None;
    }
    let mut index = 0;
    let mut y = 0u32;
    let mut count = 0usize;
    while index < pending.len() {
        let first = index;
        let stripe_height = pending[index].height;
        if stripe_height == 0 {
            return None;
        }
        let mut x = 0u32;
        while x < width {
            let block = pending.get(index)?;
            if block.output_x != x
                || block.output_y != y
                || block.height != stripe_height
                || block.width == 0
            {
                return None;
            }
            x = x.checked_add(block.width)?;
            if x > width {
                return None;
            }
            index += 1;
        }
        let next_y = y.checked_add(stripe_height)?;
        if next_y > height {
            return None;
        }
        visit(Stripe {
            first,
            end: index,
            y,
            height: stripe_height,
        });
        y = next_y;
        count += 1;
    }
    (y == height).then_some(count)
}

struct UnpublishedBand<'a> {
    output: &'a mut [f32],
    committed: bool,
}

impl Drop for UnpublishedBand<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.output.fill(0.0);
        }
    }
}

struct StripeTask<'a> {
    output: &'a mut [f32],
    blocks: &'a [PendingHtBlock],
    y: u32,
}

impl StripeTask<'_> {
    fn decode(
        &mut self,
        width: u32,
        parameters: HtParallelParameters,
        workspace: &mut HtCodeBlockDecodeWorkspace,
    ) -> Result<()> {
        for block in self.blocks {
            let start =
                (block.output_y - self.y) as usize * width as usize + block.output_x as usize;
            let length = (block.height as usize - 1) * width as usize + block.width as usize;
            let decode = if parameters.irreversible_midpoint {
                decode_ht_code_block_scalar_with_workspace_midpoint
            } else {
                decode_ht_code_block_scalar_with_workspace
            };
            decode(
                HtCodeBlockDecodeJob {
                    data: &block.combined.data,
                    cleanup_length: block.combined.cleanup_length,
                    refinement_length: block.combined.refinement_length,
                    width: block.width,
                    height: block.height,
                    output_stride: width as usize,
                    missing_bit_planes: block.missing_bit_planes,
                    number_of_coding_passes: block.number_of_coding_passes,
                    num_bitplanes: parameters.num_bitplanes,
                    roi_shift: parameters.roi_shift,
                    stripe_causal: parameters.stripe_causal,
                    strict: parameters.strict,
                    dequantization_step: parameters.dequantization_step,
                },
                &mut self.output[start..start + length],
                workspace,
            )?;
        }
        Ok(())
    }
}

pub(in crate::j2c::decode::subband) fn try_decode_ht_stripes(
    pending: &[PendingHtBlock],
    sub_band: &SubBand,
    parameters: HtParallelParameters,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'_>,
    budget: &mut DecodeAllocationBudget,
) -> Result<bool> {
    let initial_budget = *budget;
    let initial_structural = storage.structural_workspace_bytes;
    let result = decode_ht_stripes(pending, sub_band, parameters, tile_ctx, storage, budget);
    // Stripe/task descriptors are transient; the bank or optional slab may
    // have changed even when admission fails. Keep the original cap and charge
    // only those actual retained changes before returning to the staged path.
    let mut retained_budget = initial_budget;
    retained_budget.release_bytes(initial_structural)?;
    retained_budget.include_bytes(storage.structural_workspace_bytes)?;
    *budget = retained_budget;
    match result {
        Err(error) if super::super::super::reuse::is_capacity_error(&error) => Ok(false),
        result => result,
    }
}

fn prepare_stripe_workspaces(
    pending: &[PendingHtBlock],
    stripes: &[Stripe],
    tile_ctx: &mut TileDecodeContext,
    structural_workspace_bytes: &mut usize,
    budget: &mut DecodeAllocationBudget,
) -> Result<(PreparedTaskPlan, usize)> {
    let initial_tasks = bounded_parallel_task_count(stripes.len());
    let mut requested_tasks = initial_tasks;
    let mut released_bank = false;
    let mut released_slab = false;
    loop {
        let plan = balanced_task_plan(stripes.len(), requested_tasks)?;
        let chunks = plan
            .chunks(stripes)
            .map(|chunk| &pending[chunk[0].first..chunk[chunk.len() - 1].end]);
        match try_prepare_ht_workspace_chunks(
            chunks,
            plan.active_workspaces,
            &mut tile_ctx.ht_task_workspaces,
            structural_workspace_bytes,
            budget,
        ) {
            Err(error)
                if !released_slab
                    && tile_ctx.parallel_coefficients.capacity() != 0
                    && super::super::super::reuse::is_capacity_error(&error) =>
            {
                release_coefficient_slab(
                    &mut tile_ctx.parallel_coefficients,
                    structural_workspace_bytes,
                    #[cfg(test)]
                    &mut tile_ctx.debug_counters.parallel_coefficients,
                    budget,
                )?;
                released_slab = true;
            }
            Err(error)
                if requested_tasks > 1 && super::super::super::reuse::is_capacity_error(&error) =>
            {
                requested_tasks = requested_tasks.div_ceil(2);
            }
            Err(error)
                if !released_bank
                    && !tile_ctx.ht_task_workspaces.is_empty()
                    && super::super::super::reuse::is_capacity_error(&error) =>
            {
                release_ht_task_bank(
                    &mut tile_ctx.ht_task_workspaces,
                    structural_workspace_bytes,
                    budget,
                )?;
                requested_tasks = initial_tasks;
                released_bank = true;
            }
            result => return result.map(|growths| (plan, growths)),
        }
    }
}

fn decode_ht_stripes(
    pending: &[PendingHtBlock],
    sub_band: &SubBand,
    parameters: HtParallelParameters,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'_>,
    budget: &mut DecodeAllocationBudget,
) -> Result<bool> {
    let width = sub_band.rect.width();
    let height = sub_band.rect.height();
    let Some(stripe_count) = visit_stripes(pending, width, height, |_| {}) else {
        return Ok(false);
    };
    let count = (width as usize)
        .checked_mul(height as usize)
        .ok_or(ValidationError::ImageTooLarge)?;
    let output = storage
        .coefficients
        .get_mut(sub_band.coefficients.clone())
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    // Normal tile construction starts every band at positive zero. Check the
    // invariant rather than overwriting caller state if this boundary is reused.
    if output.len() != count || output.iter().any(|value| value.to_bits() != 0) {
        return Ok(false);
    }
    let mut stripes = Vec::new();
    budget.reserve_new(&mut stripes, stripe_count)?;
    visit_stripes(pending, width, height, |stripe| stripes.push(stripe))
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let (plan, growths) = prepare_stripe_workspaces(
        pending,
        &stripes,
        tile_ctx,
        &mut storage.structural_workspace_bytes,
        budget,
    )?;
    let mut band = UnpublishedBand {
        output,
        committed: false,
    };
    let mut tasks = Vec::new();
    budget.reserve_new(&mut tasks, plan.active_workspaces)?;
    #[cfg(test)]
    {
        tile_ctx.debug_counters.ht_parallel_tasks = tile_ctx
            .debug_counters
            .ht_parallel_tasks
            .max(plan.active_workspaces);
        tile_ctx.debug_counters.ht_task_workspace_growths = tile_ctx
            .debug_counters
            .ht_task_workspace_growths
            .saturating_add(growths);
    }
    #[cfg(not(test))]
    let _ = growths;
    let mut remaining = &mut *band.output;
    for chunk in plan.chunks(&stripes) {
        let first = chunk[0];
        let last = chunk[chunk.len() - 1];
        let rows = last.y + last.height - first.y;
        let length = rows as usize * width as usize;
        let (output, tail) = remaining.split_at_mut(length);
        remaining = tail;
        tasks.push(StripeTask {
            output,
            blocks: &pending[first.first..last.end],
            y: first.y,
        });
    }
    tasks
        .par_iter_mut()
        .zip(tile_ctx.ht_task_workspaces[..plan.active_workspaces].par_iter_mut())
        .try_for_each(|(task, slot)| task.decode(width, parameters, &mut slot.workspace))?;
    drop(tasks);
    band.committed = true;
    #[cfg(test)]
    {
        tile_ctx.debug_counters.parallel_coefficients.direct_bytes = tile_ctx
            .debug_counters
            .parallel_coefficients
            .direct_bytes
            .saturating_add(count.saturating_mul(size_of::<f32>()));
    }
    Ok(true)
}

#[cfg(test)]
mod tests;
