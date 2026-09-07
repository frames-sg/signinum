// SPDX-License-Identifier: MIT OR Apache-2.0

//! Full sampled-component reconstruction without intermediate host readback.

use super::{
    allocate_direct_color_batch_execution, allocation::DirectExecutionMetadata,
    color_batch_completion::complete_direct_color_batch_command,
    encode_prepared_direct_component_plane_in_command_buffer, DirectComponentPlaneRequest,
};
use crate::engine::{
    direct_plane_pack::{encode_plane_stage_to_surface_in_command_buffer, PlaneStage},
    new_command_buffer, new_compute_command_encoder, new_shared_buffer, prepare_direct_color_plan,
    with_runtime, with_runtime_for_session, DirectHybridStageTimings, DirectTier1Mode,
    MetalRuntime, PreparedDirectColorPlan,
};
use crate::metal_types::prelude::*;
use crate::metal_types::{Buffer, CommandBufferRef};
use crate::{Error, MetalBackendSession, MetalDirectFallbackReason, Surface};
use j2k_core::PixelFormat;
use j2k_native::{DecodeSettings, DecoderContext, Image};
use std::sync::Arc;

mod stacked;

struct SampledPlan {
    plan: Arc<PreparedDirectColorPlan>,
    sampling: [(u8, u8); 3],
}

fn unsupported() -> Error {
    Error::MetalDirectFallback {
        message: "component-grid Metal decode requires full unsigned origin-zero RGB components without MCT".into(),
        reason: MetalDirectFallbackReason::UnsupportedPlan,
    }
}

pub(crate) fn decode_component_grid_color_batch(
    inputs: &[impl AsRef<[u8]>],
    fmt: PixelFormat,
    session: Option<&MetalBackendSession>,
) -> Result<Vec<Surface>, Error> {
    if !matches!(
        fmt,
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
    ) {
        return Err(unsupported());
    }
    let execute = |runtime: &MetalRuntime| {
        let mut budget =
            crate::batch_allocation::BatchMetadataBudget::new("sampled color decode plans");
        let mut sampled = budget.try_vec(inputs.len(), "sampled color plans")?;
        let mut plans = budget.try_vec(inputs.len(), "sampled color execution plans")?;
        for input in inputs {
            let image = Image::new(input.as_ref(), &DecodeSettings::default())
                .map_err(crate::error::native_decode_error)?;
            let (plan, sampling) = image
                .build_component_grid_color_plan_with_context(&mut DecoderContext::default())
                .map_err(|error| {
                    if crate::direct::is_unsupported_direct_plan_error(&error) {
                        unsupported()
                    } else {
                        crate::error::native_decode_error(error)
                    }
                })?;
            for (component, &(sx, sy)) in plan.component_plans.iter().zip(&sampling) {
                if sx == 0
                    || sy == 0
                    || component.dimensions
                        != (
                            plan.dimensions.0.div_ceil(u32::from(sx)),
                            plan.dimensions.1.div_ceil(u32::from(sy)),
                        )
                {
                    return Err(unsupported());
                }
            }
            let plan = Arc::new(prepare_direct_color_plan(&plan)?);
            plans.push(plan.clone());
            sampled.push(SampledPlan { plan, sampling });
        }
        if sampled.is_empty() {
            return Ok(Vec::new());
        }
        let mut metadata = allocate_direct_color_batch_execution(&plans, DirectTier1Mode::Metal)?;
        let mut timings = DirectHybridStageTimings::default();
        let command = new_command_buffer(&runtime.queue)?;
        crate::profile_env::label_command_buffer(&command, "J2K sampled color batch");
        let surfaces = if let Some(surfaces) = stacked::try_encode_stacked(
            runtime,
            &command,
            &sampled,
            fmt,
            &mut metadata,
            &mut timings,
        )? {
            surfaces
        } else {
            let mut surfaces = budget.try_vec(sampled.len(), "sampled color surfaces")?;
            for plan in &sampled {
                surfaces.push(encode_sampled_plan(
                    runtime,
                    &command,
                    plan,
                    fmt,
                    &mut metadata,
                    &mut timings,
                )?);
            }
            surfaces
        };
        // Checked command buffers retain encoded resources through completion;
        // status and pooled scratch owners remain in metadata until retirement.
        command.commit();
        complete_direct_color_batch_command(runtime, &command, false, &mut timings, &mut metadata)?;
        Ok(surfaces)
    };
    match session {
        Some(session) => with_runtime_for_session(session, execute),
        None => with_runtime(execute),
    }
}

fn encode_sampled_plan(
    runtime: &MetalRuntime,
    command: &CommandBufferRef,
    sampled: &SampledPlan,
    fmt: PixelFormat,
    metadata: &mut DirectExecutionMetadata,
    timings: &mut DirectHybridStageTimings,
) -> Result<Surface, Error> {
    let plan = &sampled.plan;
    let mut planes: [Option<Buffer>; 4] = [None, None, None, None];
    for (index, component) in plan.component_plans.iter().enumerate() {
        let plane = encode_prepared_direct_component_plane_in_command_buffer(
            DirectComponentPlaneRequest {
                runtime,
                command_buffer: command,
                plan: component,
                tier1_mode: DirectTier1Mode::Metal,
                stage_timings: timings,
                retained_buffers: &mut metadata.retained_buffers,
                status_checks: &mut metadata.status_checks,
                scratch_buffers: &mut metadata.scratch_buffers,
            },
        )?;
        planes[index] = Some(if sampled.sampling[index] == (1, 1) {
            plane
        } else {
            expand_plane(
                runtime,
                command,
                &plane,
                0,
                plan.dimensions,
                sampled.sampling[index],
            )?
        });
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
    encode_plane_stage_to_surface_in_command_buffer(runtime, command, &stage, fmt)
}

fn expand_plane(
    runtime: &MetalRuntime,
    command: &CommandBufferRef,
    input: &Buffer,
    input_offset: usize,
    dims: (u32, u32),
    sampling: (u8, u8),
) -> Result<Buffer, Error> {
    let span = crate::engine::direct_roi::checked_f32_span(
        dims.0 as usize,
        dims.1 as usize,
        "sampled component expansion",
    )?;
    let output = new_shared_buffer(&runtime.device, span.bytes)?;
    let encoder = new_compute_command_encoder(command)?;
    crate::profile_env::label_compute_encoder(&encoder, "J2K sampled component expansion");
    let pipeline = &runtime.decode()?.expand_sampled_plane;
    encoder.setComputePipelineState(pipeline);
    encoder.set_buffer(0, Some(input), input_offset as u64);
    encoder.set_buffer(1, Some(&output), 0);
    encoder.set_bytes::<[u32; 4]>(
        2,
        &[dims.0, dims.1, u32::from(sampling.0), u32::from(sampling.1)],
    );
    j2k_metal_support::dispatch_2d_pipeline(&encoder, pipeline, dims);
    encoder.endEncoding();
    Ok(output)
}
