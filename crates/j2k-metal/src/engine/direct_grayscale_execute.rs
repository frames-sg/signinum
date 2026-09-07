// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use std::sync::Arc;

use crate::profile_env::{label_command_buffer, metal_profile_decode_split_commands_enabled};

use super::abi::{J2kGrayStoreParams, J2kStoreParams};
use super::decode_dispatch::{
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets,
    dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets,
    dispatch_store_component_buffer_in_encoder_with_offsets,
    encode_gray_store_to_destination_in_encoder, encode_gray_store_to_surface_in_encoder,
    encode_prepared_classic_sub_band_group_to_buffer_in_encoder,
    encode_prepared_classic_sub_band_to_buffer_in_encoder,
    encode_prepared_ht_sub_band_group_to_buffer_in_encoder,
    encode_prepared_ht_sub_band_to_buffer_in_encoder, GrayStoreDestinationRequest,
    IdwtSubBandBuffers, SingleIdwtDispatch,
};
use super::direct_roi::{
    idwt_input_windows_from_slices, prepared_idwt_output_len, prepared_idwt_params,
    BandRequiredRegion,
};
use super::{
    commit_and_wait_metal, emit_direct_hybrid_stage_timings,
    encode_gray_plane_to_surface_in_encoder, encode_prepared_direct_color_plan_in_command_buffer,
    encode_repeated_direct_grayscale_plan_in_command_buffer,
    encode_repeated_gray_plane_to_surfaces_in_command_buffer,
    encode_stacked_direct_component_plane_batch, j2k_scalar_pack_params, lookup_direct_band_slice,
    lookup_direct_band_slice_entry, metal_profile_stages_enabled, new_command_buffer,
    new_compute_command_encoder, prepared_direct_color_plan_supports_runtime,
    recycle_scratch_buffers, retire_direct_status_checks,
    supports_stacked_direct_component_plane_batch, take_f32_scratch_buffer,
    try_encode_stacked_mct_rgb8_direct_color_batch, wait_for_completion_metal, with_runtime,
    with_runtime_for_device, Buffer, CommandBuffer, CommandBufferRef,
    DecodeHybridSplitCommandBuffers, Device, DirectBandSlice, DirectColorBatchCommandBuffers,
    DirectColorPlanRequest, DirectHybridStageTimings, DirectScratchBuffer, DirectStatusCheck,
    DirectStatusRetirementMode, DirectTier1Mode, Error, J2kWaveletTransform, MetalRuntime,
    PixelFormat, PreparedDirectColorPlan, PreparedDirectGrayscalePlan, PreparedDirectGrayscaleStep,
    RepeatedDirectGrayscalePlanRequest, StackedDirectColorBatchRequest,
    StackedDirectComponentPlaneBatchRequest, Surface,
};
mod allocation;
mod color_batch_completion;
mod color_destination;
mod component_plane;
mod destination;
mod destination_index_validation;
mod grayscale_batch;
mod sampled_color;
mod single;
pub(crate) use sampled_color::decode_component_grid_color_batch;

pub(in crate::engine) use self::allocation::extend_preallocated_retained_buffers;
use self::allocation::{
    allocate_direct_execution_metadata, direct_ht_job_count, DirectExecutionMetadata,
};
use self::color_batch_completion::{
    complete_direct_color_batch_command, complete_split_direct_color_batch_command,
    retire_direct_color_batch_resources,
};
pub(crate) use self::color_destination::submit_prepared_direct_color_plan_batch_into_group;
pub(in crate::engine) use self::component_plane::{
    checked_coefficient_len, encode_prepared_direct_component_plane_in_command_buffer,
    encode_prepared_direct_component_plane_in_encoder, upload_cpu_decoded_coefficients,
    DirectComponentPlaneRequest,
};
pub(crate) use self::destination::{
    submit_prepared_direct_grayscale_plan_batch_into_group, DirectDestinationConsumerOrdering,
    SubmittedDirectDestination,
};
pub(crate) use self::grayscale_batch::{
    execute_prepared_direct_grayscale_plan_batch, execute_repeated_prepared_direct_grayscale_plan,
};
pub(in crate::engine) use self::single::encode_prepared_direct_grayscale_plan_in_command_buffer;
pub(in crate::engine) use self::single::{
    encode_prepared_direct_grayscale_plan_into_in_encoder,
    DirectGrayscaleDestinationExecutionRequest,
};

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_grayscale_plan(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let DirectExecutionMetadata {
            mut retained_buffers,
            mut status_checks,
            mut scratch_buffers,
            dispatch_report: _,
        } = allocate_direct_execution_metadata(
            plan.steps.len(),
            direct_ht_job_count([plan], "J2K Metal direct grayscale execution HT jobs")?,
            crate::batch_allocation::BatchMetadataBudget::new(
                "J2K Metal direct grayscale execution resources",
            ),
        )?;
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let surface = encode_prepared_direct_grayscale_plan_in_command_buffer(
            runtime,
            &command_buffer,
            plan,
            fmt,
            &mut retained_buffers,
            &mut status_checks,
            &mut scratch_buffers,
        )?;
        let completion = commit_and_wait_metal(&command_buffer);
        let status_retirement = retire_direct_status_checks(
            runtime,
            status_checks,
            if completion.is_ok() {
                DirectStatusRetirementMode::Validate
            } else {
                DirectStatusRetirementMode::RecycleWithoutRead
            },
        );
        drop(retained_buffers);
        let scratch_retirement = recycle_scratch_buffers(runtime, scratch_buffers);
        completion.and(status_retirement).and(scratch_retirement)?;
        Ok(surface)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_grayscale_plan_with_device(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        execute_prepared_direct_grayscale_plan(plan, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan(
    plan: Arc<PreparedDirectColorPlan>,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let plans = [plan];
    let mut surfaces = execute_prepared_direct_color_plan_batch(&plans, fmt)?;
    surfaces.pop().ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect color plan produced no surface".to_string(),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan_with_device(
    plan: Arc<PreparedDirectColorPlan>,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| execute_prepared_direct_color_plan(plan, fmt))
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan_batch(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1(plans, fmt, DirectTier1Mode::Metal)
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_hybrid_cpu_tier1_direct_color_plan(
    plan: Arc<PreparedDirectColorPlan>,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let plans = [plan];
    let mut surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(&plans, fmt)?;
    surfaces.pop().ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect hybrid color plan produced no surface".to_string(),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_hybrid_cpu_tier1_direct_color_plan_with_device(
    plan: Arc<PreparedDirectColorPlan>,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        execute_hybrid_cpu_tier1_direct_color_plan(plan, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_hybrid_cpu_tier1_direct_color_plan_batch(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1(plans, fmt, DirectTier1Mode::CpuUpload)
}

#[cfg(target_os = "macos")]
pub(super) fn execute_direct_color_plan_batch_with_tier1(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
    tier1_mode: DirectTier1Mode,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1_options(plans, fmt, tier1_mode, false)
}

#[cfg(all(target_os = "macos", test))]
pub(super) fn execute_flattened_hybrid_cpu_tier1_direct_color_plan_batch_for_test(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1_options(plans, fmt, DirectTier1Mode::CpuUpload, true)
}

#[cfg(target_os = "macos")]
fn allocate_direct_color_batch_execution(
    plans: &[Arc<PreparedDirectColorPlan>],
    tier1_mode: DirectTier1Mode,
) -> Result<DirectExecutionMetadata, Error> {
    let step_count = crate::batch_allocation::checked_count_sum(
        plans.iter().flat_map(|plan| {
            plan.component_plans
                .iter()
                .map(|component| component.steps.len())
        }),
        "J2K Metal direct color batch step metadata",
    )?;
    let ht_job_count = if tier1_mode == DirectTier1Mode::Metal {
        direct_ht_job_count(
            plans.iter().flat_map(|plan| plan.component_plans.iter()),
            "J2K Metal direct color batch HT jobs",
        )?
    } else {
        0
    };
    allocate_direct_execution_metadata(
        step_count,
        ht_job_count,
        crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal direct color batch execution resources",
        ),
    )
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct DirectColorBatchRoute<'a> {
    runtime: &'a MetalRuntime,
    plans: &'a [Arc<PreparedDirectColorPlan>],
    format: PixelFormat,
    tier1_mode: DirectTier1Mode,
    force_flattened_cpu_tier1: bool,
    profile_hybrid_stages: bool,
}

#[cfg(target_os = "macos")]
fn try_encode_split_direct_color_batch(
    route: DirectColorBatchRoute<'_>,
    stage_timings: &mut DirectHybridStageTimings,
    metadata: &mut DirectExecutionMetadata,
) -> Result<Option<Vec<Surface>>, Error> {
    let DirectColorBatchRoute {
        runtime,
        plans,
        format: fmt,
        tier1_mode,
        force_flattened_cpu_tier1,
        profile_hybrid_stages,
    } = route;
    if fmt != PixelFormat::Rgb8
        || !profile_hybrid_stages
        || !metal_profile_decode_split_commands_enabled()
    {
        return Ok(None);
    }

    let split_command_buffers = DecodeHybridSplitCommandBuffers::new(runtime)?;
    if let Some(surfaces) =
        try_encode_stacked_mct_rgb8_direct_color_batch(StackedDirectColorBatchRequest {
            runtime,
            command_buffers: split_command_buffers.refs(),
            plans,
            tier1_mode,
            force_flattened_cpu_tier1,
            stage_timings,
            retained_buffers: &mut metadata.retained_buffers,
            status_checks: &mut metadata.status_checks,
            scratch_buffers: &mut metadata.scratch_buffers,
        })?
    {
        split_command_buffers.commit_in_order();
        complete_split_direct_color_batch_command(
            runtime,
            &split_command_buffers,
            stage_timings,
            metadata,
        )?;
        emit_direct_hybrid_stage_timings(stage_timings, fmt, plans.len());
        return Ok(Some(surfaces));
    }

    drop(split_command_buffers);
    retire_direct_color_batch_resources(
        runtime,
        Ok(()),
        DirectStatusRetirementMode::RecycleWithoutRead,
        metadata,
    )?;
    *stage_timings = DirectHybridStageTimings::default();
    Ok(None)
}

#[cfg(target_os = "macos")]
fn encode_direct_color_plans_individually(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
    tier1_mode: DirectTier1Mode,
    stage_timings: &mut DirectHybridStageTimings,
    metadata: &mut DirectExecutionMetadata,
) -> Result<Vec<Surface>, Error> {
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K Metal direct color batch surface collection",
    );
    let mut surfaces = budget.try_vec(plans.len(), "J2K Metal direct color surfaces")?;
    for plan in plans {
        surfaces.push(encode_prepared_direct_color_plan_in_command_buffer(
            DirectColorPlanRequest {
                runtime,
                command_buffer,
                plan,
                fmt,
                tier1_mode,
                stage_timings,
                retained_buffers: &mut metadata.retained_buffers,
                status_checks: &mut metadata.status_checks,
                scratch_buffers: &mut metadata.scratch_buffers,
            },
        )?);
    }
    Ok(surfaces)
}

#[cfg(target_os = "macos")]
fn encode_single_command_direct_color_batch(
    route: DirectColorBatchRoute<'_>,
    stage_timings: &mut DirectHybridStageTimings,
    metadata: &mut DirectExecutionMetadata,
) -> Result<Vec<Surface>, Error> {
    let DirectColorBatchRoute {
        runtime,
        plans,
        format: fmt,
        tier1_mode,
        force_flattened_cpu_tier1,
        profile_hybrid_stages,
    } = route;
    let command_buffer = new_command_buffer(&runtime.queue)?;
    if profile_hybrid_stages {
        label_command_buffer(&command_buffer, "j2k decode hybrid direct color batch");
    }
    let stacked_surfaces = if fmt == PixelFormat::Rgb8 {
        try_encode_stacked_mct_rgb8_direct_color_batch(StackedDirectColorBatchRequest {
            runtime,
            command_buffers: DirectColorBatchCommandBuffers::single(&command_buffer),
            plans,
            tier1_mode,
            force_flattened_cpu_tier1,
            stage_timings,
            retained_buffers: &mut metadata.retained_buffers,
            status_checks: &mut metadata.status_checks,
            scratch_buffers: &mut metadata.scratch_buffers,
        })?
    } else {
        None
    };
    let surfaces = match stacked_surfaces {
        Some(surfaces) => surfaces,
        None => encode_direct_color_plans_individually(
            runtime,
            &command_buffer,
            plans,
            fmt,
            tier1_mode,
            stage_timings,
            metadata,
        )?,
    };

    command_buffer.commit();
    complete_direct_color_batch_command(
        runtime,
        &command_buffer,
        profile_hybrid_stages,
        stage_timings,
        metadata,
    )?;
    if tier1_mode == DirectTier1Mode::CpuUpload {
        emit_direct_hybrid_stage_timings(stage_timings, fmt, plans.len());
    }
    Ok(surfaces)
}

#[cfg(target_os = "macos")]
fn encode_direct_color_batch_routes(
    runtime: &MetalRuntime,
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
    tier1_mode: DirectTier1Mode,
    force_flattened_cpu_tier1: bool,
    mut metadata: DirectExecutionMetadata,
) -> Result<Vec<Surface>, Error> {
    let mut stage_timings = DirectHybridStageTimings::default();
    let profile_hybrid_stages =
        tier1_mode == DirectTier1Mode::CpuUpload && metal_profile_stages_enabled();
    let route = DirectColorBatchRoute {
        runtime,
        plans,
        format: fmt,
        tier1_mode,
        force_flattened_cpu_tier1,
        profile_hybrid_stages,
    };
    if let Some(surfaces) =
        try_encode_split_direct_color_batch(route, &mut stage_timings, &mut metadata)?
    {
        return Ok(surfaces);
    }
    encode_single_command_direct_color_batch(route, &mut stage_timings, &mut metadata)
}

#[cfg(target_os = "macos")]
pub(super) fn execute_direct_color_plan_batch_with_tier1_options(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
    tier1_mode: DirectTier1Mode,
    force_flattened_cpu_tier1: bool,
) -> Result<Vec<Surface>, Error> {
    if plans.is_empty() {
        return Ok(Vec::new());
    }
    if tier1_mode == DirectTier1Mode::Metal
        && plans
            .iter()
            .any(|plan| !prepared_direct_color_plan_supports_runtime(plan, fmt))
    {
        return Err(Error::MetalDirectFallback {
            message: "unsupported classic kernel input in direct component plan".to_string(),
            reason: crate::MetalDirectFallbackReason::UnsupportedRuntimeInput,
        });
    }

    with_runtime(|runtime| {
        let metadata = allocate_direct_color_batch_execution(plans, tier1_mode)?;
        encode_direct_color_batch_routes(
            runtime,
            plans,
            fmt,
            tier1_mode,
            force_flattened_cpu_tier1,
            metadata,
        )
    })
}
