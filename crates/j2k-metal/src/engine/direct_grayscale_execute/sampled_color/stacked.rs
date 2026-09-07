// SPDX-License-Identifier: MIT OR Apache-2.0

//! Batch independent entropy and reconstruction graphs at each component size.

use super::{
    encode_plane_stage_to_surface_in_command_buffer, expand_plane, Buffer, CommandBufferRef,
    DirectExecutionMetadata, DirectHybridStageTimings, DirectTier1Mode, Error, MetalRuntime,
    PixelFormat, PlaneStage, SampledPlan, Surface,
};
use crate::engine::{
    direct_roi::checked_f32_span,
    direct_stacked_batch::{
        encode_stacked_direct_component_plane_batch, supports_stacked_direct_component_plane_batch,
        StackedDirectComponentPlaneBatchRequest,
    },
    DirectColorBatchCommandBuffers,
};

pub(super) fn try_encode_stacked(
    runtime: &MetalRuntime,
    command: &CommandBufferRef,
    sampled: &[SampledPlan],
    fmt: PixelFormat,
    metadata: &mut DirectExecutionMetadata,
    timings: &mut DirectHybridStageTimings,
) -> Result<Option<Vec<Surface>>, Error> {
    let first = &sampled[0];
    if sampled.len() < 2
        || !sampled
            .iter()
            .all(|p| p.plan.dimensions == first.plan.dimensions && p.sampling == first.sampling)
    {
        return Ok(None);
    }
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new("sampled stacked planes");
    let mut refs = budget.try_vec(3, "sampled component references")?;
    for component in 0..3 {
        let mut plans = budget.try_vec(sampled.len(), "sampled component batch references")?;
        plans.extend(sampled.iter().map(|p| &p.plan.component_plans[component]));
        if !supports_stacked_direct_component_plane_batch(&plans) {
            return Ok(None);
        }
        refs.push(plans);
    }
    let mut stacked = budget.try_vec(3, "sampled stacked component owners")?;
    for (component_idx, plans) in refs.iter().enumerate() {
        stacked.push(encode_stacked_direct_component_plane_batch(
            StackedDirectComponentPlaneBatchRequest {
                runtime,
                command_buffers: DirectColorBatchCommandBuffers::single(command),
                compute_encoder: None,
                plans,
                component_idx,
                flattened_cpu_tier1_cache: None,
                tier1_mode: DirectTier1Mode::Metal,
                stage_timings: timings,
                retained_buffers: &mut metadata.retained_buffers,
                status_checks: &mut metadata.status_checks,
                scratch_buffers: &mut metadata.scratch_buffers,
            },
        )?);
    }
    let mut surfaces = budget.try_vec(sampled.len(), "sampled stacked outputs")?;
    for (index, sampled) in sampled.iter().enumerate() {
        let plan = &sampled.plan;
        let mut planes: [Option<Buffer>; 4] = [None, None, None, None];
        for (component, plane) in stacked.iter().enumerate() {
            let span = checked_f32_span(
                plane.dimensions.0 as usize,
                plane.dimensions.1 as usize,
                "sampled stacked plane offset",
            )?;
            let offset = span
                .bytes
                .checked_mul(index)
                .ok_or_else(|| Error::MetalKernel {
                    message: "sampled stacked plane offset overflow".into(),
                })?;
            // The unit factor also extracts this image's slice from the stack.
            planes[component] = Some(expand_plane(
                runtime,
                command,
                &plane.buffer,
                offset,
                plan.dimensions,
                sampled.sampling[component],
            )?);
        }
        let stage = PlaneStage {
            dims: plan.dimensions,
            plane_count: 3,
            color_space: j2k_native::ColorSpace::RGB,
            has_alpha: false,
            bit_depths: [
                u32::from(plan.bit_depths[0]),
                u32::from(plan.bit_depths[1]),
                u32::from(plan.bit_depths[2]),
                0,
            ],
            planes,
        };
        surfaces.push(encode_plane_stage_to_surface_in_command_buffer(
            runtime, command, &stage, fmt,
        )?);
    }
    Ok(Some(surfaces))
}
