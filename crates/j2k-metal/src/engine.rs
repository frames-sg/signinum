// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::{
    Buffer, BufferRef, CommandBuffer, CommandBufferRef, ComputeCommandEncoderRef,
    ComputePipelineState, Device,
};
#[cfg(target_os = "macos")]
use j2k_core::{PixelFormat, Rect};
#[cfg(target_os = "macos")]
use j2k_native::{
    idwt_required_input_windows, idwt_required_output_margin,
    pack_j2k_code_block_scalar_from_tier1_tokens, ColorSpace as NativeColorSpace,
    DecodedComponents as NativeDecodedComponents, EncodeProgressionOrder, EncodedHtJ2kCodeBlock,
    EncodedHtJ2kCodeBlockSet, EncodedJ2kCodeBlock, HtCodeBlockDecodeJob, HtSubBandDecodeJob,
    J2kCodeBlockDecodeJob, J2kCodeBlockSegment, J2kDeinterleaveMctToF32Job,
    J2kDeinterleaveToF32Job, J2kDirectBandId, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep,
    J2kDirectIdwtStep, J2kDirectStoreStep, J2kForwardDwt53Level, J2kForwardDwt53Output,
    J2kForwardDwt97Level, J2kForwardDwt97Output, J2kHtCodeBlockEncodeJob,
    J2kHtCodeBlockSetEncodeJob, J2kInverseMctJob, J2kPacketizationBlockCodingMode,
    J2kPacketizationEncodeJob, J2kPacketizationPacketDescriptor, J2kQuantizeSubbandJob,
    J2kRequiredBandRegion, J2kSingleDecompositionIdwtJob, J2kStoreComponentJob,
    J2kSubBandDecodeJob, J2kTier1CodeBlockEncodeJob, J2kTier1TokenSegment, J2kWaveletTransform,
};
#[cfg(target_os = "macos")]
use rayon::prelude::*;

use crate::{Error, Surface};

mod abi;
#[cfg(target_os = "macos")]
mod buffer_validation;
#[cfg(target_os = "macos")]
pub(crate) use self::buffer_validation::{
    copy_interleaved_padded_to_shared_buffer, validate_metal_buffer_matches_bytes,
    validate_metal_buffers_match, PaddedInterleavedCopy,
};
#[cfg(target_os = "macos")]
mod classic_encode_pipeline;
#[cfg(target_os = "macos")]
use self::classic_encode_pipeline::{
    classic_cod_block_style_from_flags, classic_encode_code_blocks_pipeline,
    classic_encode_code_blocks_pipeline_kind, classic_resident_style_flags_from_env,
    classic_tier1_gpu_token_pack_supported, J2kClassicEncodePipelineKind,
};
#[cfg(target_os = "macos")]
mod classic_tier1_stats;
#[cfg(target_os = "macos")]
use self::classic_tier1_stats::{
    accumulate_classic_tier1_scan_estimates, classic_tier1_pass_class_counts,
};
#[cfg(target_os = "macos")]
mod code_block_decoder;
#[cfg(target_os = "macos")]
use self::code_block_decoder::MetalCodeBlockDecoder;
mod direct_cache;
use self::direct_cache::CpuTier1CoefficientCache;
#[cfg(target_os = "macos")]
mod direct_buffers;
#[cfg(target_os = "macos")]
pub(crate) use self::direct_buffers::{
    buffer_is_cpu_visible, checked_buffer_read, checked_buffer_slice, checked_buffer_slice_at,
};
#[cfg(target_os = "macos")]
use self::direct_buffers::{
    copied_recyclable_shared_slice_buffer, copied_slice_buffer, new_private_buffer,
    new_shared_buffer, new_shared_buffer_with_slice, take_classic_coefficients_scratch_buffer,
    take_classic_states_scratch_buffer, zeroed_recyclable_shared_buffer, zeroed_shared_buffer,
};
#[cfg(target_os = "macos")]
mod direct_commands;
#[cfg(target_os = "macos")]
use self::direct_commands::{
    new_blit_command_encoder, new_command_buffer, new_compute_command_encoder,
    DecodeHybridSplitCommandBuffers, DirectColorBatchCommandBuffers, DirectIdwtCommandBuffers,
};
#[cfg(target_os = "macos")]
mod direct_cpu;
#[cfg(target_os = "macos")]
use self::direct_cpu::{
    decode_classic_inputs_on_cpu_with_plan_cache, decode_ht_inputs_on_cpu_with_plan_cache,
    decode_prepared_classic_jobs_on_cpu_with_scratch,
    decode_prepared_classic_jobs_on_cpu_with_scratch_profiled,
    decode_prepared_ht_jobs_on_cpu_with_workspace,
    decode_prepared_ht_jobs_on_cpu_with_workspace_profiled, ClassicCpuDecodeInput,
    ClassicCpuDecodeScratch, HtCpuDecodeInput,
};
#[cfg(all(target_os = "macos", test))]
use self::direct_cpu::{
    decode_prepared_classic_sub_band_on_cpu, decode_prepared_ht_sub_band_group_on_cpu_profile,
};
#[cfg(target_os = "macos")]
mod direct_flattened;
#[cfg(all(target_os = "macos", test))]
use self::direct_flattened::hybrid_cpu_decode_worker_count;
#[cfg(target_os = "macos")]
use self::direct_flattened::{
    build_flattened_cpu_tier1_cache, packed_cpu_decode_coefficients,
    packed_cpu_decode_coefficients_in, packed_cpu_decode_output_len, FlattenedCpuTier1Cache,
};
mod direct_profile;
#[cfg(target_os = "macos")]
use self::direct_profile::record_completed_decode_split_gpu_stages;
use self::direct_profile::{
    elapsed_us, emit_direct_hybrid_stage_timings, CpuTier1DecodeSubstageCounters,
    DirectHybridStageTimings,
};
mod direct_plan_validation;
use self::direct_plan_validation::{
    classic_group_shapes_match, classic_sub_band_shapes_match, direct_preflight_invariant,
    ht_group_shapes_match, ht_sub_band_shapes_match, idwt_shapes_match,
    prepared_direct_color_plan_supports_runtime, store_shapes_match,
};
mod direct_scratch;
use self::direct_scratch::{
    recycle_private_buffers, recycle_scratch_buffers, recycle_shared_buffers,
    take_f32_scratch_buffer, take_recyclable_private_buffer, DirectScratchBuffer,
};
#[cfg(target_os = "macos")]
mod direct_status;
#[cfg(all(target_os = "macos", test))]
use self::direct_status::validate_direct_status;
#[cfg(target_os = "macos")]
use self::direct_status::{
    decode_classic_status_error, decode_ht_status_error, decode_mct_status_error,
    retire_direct_status_checks, DirectStatusCheck, DirectStatusRetirementMode,
};
#[cfg(target_os = "macos")]
mod direct_tier1;
#[cfg(target_os = "macos")]
use self::direct_tier1::{
    flattened_hybrid_cpu_tier1_enabled, prepare_direct_tier1_input_buffer,
    record_flattened_hybrid_cpu_decode_batch, record_hybrid_cpu_decode_inputs,
    record_hybrid_cpu_decode_worker_init, record_hybrid_repeated_output_blit,
    record_hybrid_stacked_component_batch, record_stacked_component_batch,
    should_flatten_hybrid_cpu_tier1_color_batch, DirectTier1Mode,
    HYBRID_CPU_DECODE_MIN_INPUTS_PER_TASK,
};
mod pack_params;
use self::pack_params::{j2k_pack_scale_arrays, j2k_scalar_pack_params, j2k_u32_param};
mod encode_capacity;
use self::encode_capacity::{
    classic_encode_output_capacity, classic_encode_output_capacity_for_mode,
    classic_encode_segment_capacity, classic_packet_output_capacity,
    codestream_progression_order_code, ht_encode_output_capacity,
    ht_packet_output_capacity_for_mode, lossless_codestream_assembly_capacity,
    packet_tree_node_count,
};
pub(crate) use self::encode_capacity::{
    ht_packet_output_capacity_mode_from_env, J2kClassicEncodeOutputCapacityMode,
    J2kHtPacketOutputCapacityMode,
};
mod resident_packet_plan;
use self::resident_packet_plan::{
    build_resident_batch_packet_plan, prepared_lossless_batch_tiles, ResidentBatchPacketPlan,
    ResidentBatchPacketPlanParams,
};
mod resident_types;
pub(crate) use self::resident_types::{
    J2kPendingResidentLosslessCodestream, J2kResidentBatchEncodeItem, J2kResidentEncodeStageStats,
    J2kResidentLosslessCodestream, J2kResidentLosslessCodestreamBatchResult,
};
#[cfg(target_os = "macos")]
mod gpu_timing;
#[cfg(target_os = "macos")]
use self::gpu_timing::{
    completed_command_buffer_gpu_duration, completed_command_buffers_gpu_duration,
    completed_command_buffers_gpu_duration_and_elapsed_window,
};
#[cfg(target_os = "macos")]
mod resident_stage_timing;
#[cfg(target_os = "macos")]
use self::resident_stage_timing::{
    duration_share, finish_resident_encode_split_command_buffer,
    finish_resident_encode_split_command_buffer_timed, new_resident_encode_command_buffer,
    record_completed_resident_encode_gpu_stages, J2kResidentEncodeGpuStage,
    J2kResidentEncodeGpuStageCommandBuffer,
};
#[cfg(target_os = "macos")]
mod shader_source;
#[cfg(test)]
mod test_counters;
#[cfg(test)]
pub(crate) use self::test_counters::{
    classic_gpu_token_pack_dispatches_for_test,
    classic_split_mq_byte_gpu_token_pack_dispatches_for_test,
    direct_destination_event_bridge_for_test, direct_tier1_input_buffer_prepares_for_test,
    direct_tier1_input_buffer_runtime_for_test, flattened_hybrid_cpu_decode_batches_for_test,
    ht_batch_coefficient_copy_blits_for_test, hybrid_cpu_decode_inputs_for_test,
    hybrid_cpu_decode_worker_inits_for_test, hybrid_repeated_output_blits_for_test,
    hybrid_stacked_component_batches_for_test, idwt97_stage_sequences_for_test,
    lossless_deinterleave_rct_fused_dispatches_for_test, metal_command_buffers_for_test,
    metal_compute_encoders_for_test, reset_classic_gpu_token_pack_dispatches_for_test,
    reset_classic_split_mq_byte_gpu_token_pack_dispatches_for_test,
    reset_direct_destination_event_bridge_for_test,
    reset_direct_tier1_input_buffer_prepares_for_test,
    reset_flattened_hybrid_cpu_decode_batches_for_test,
    reset_ht_batch_coefficient_copy_blits_for_test, reset_hybrid_cpu_decode_inputs_for_test,
    reset_hybrid_cpu_decode_worker_inits_for_test, reset_hybrid_repeated_output_blits_for_test,
    reset_hybrid_stacked_component_batches_for_test, reset_idwt97_stage_sequences_for_test,
    reset_lossless_deinterleave_rct_fused_dispatches_for_test,
    reset_metal_command_buffers_for_test, reset_metal_compute_encoders_for_test,
    reset_resident_codestream_command_buffer_waits_for_test,
    reset_resident_gpu_timestamp_queries_for_test, reset_stacked_component_batches_for_test,
    reset_thread_hybrid_cpu_decode_inputs_for_test,
    resident_codestream_command_buffer_waits_for_test, resident_gpu_timestamp_queries_for_test,
    stacked_component_batches_for_test, thread_hybrid_cpu_decode_inputs_for_test,
};

#[cfg(target_os = "macos")]
pub(crate) use crate::profile_env::metal_profile_stages_enabled;

#[cfg(all(target_os = "macos", test))]
pub(crate) use crate::buffer_pool::{
    private_buffer_pool_misses_for_test, private_buffer_pool_take_probes_for_test,
    reset_private_buffer_pool_misses_for_test, reset_private_buffer_pool_take_probes_for_test,
    reset_shared_buffer_pool_misses_for_test, shared_buffer_pool_misses_for_test,
};

#[cfg(all(target_os = "macos", test))]
pub(crate) use crate::profile_env::{
    force_classic_gpu_token_pack_route_for_test, force_metal_profile_stages_for_test,
};

#[cfg(target_os = "macos")]
mod runtime;
#[cfg(all(target_os = "macos", test))]
pub(crate) use self::runtime::with_isolated_runtime_for_device_for_test;
#[cfg(target_os = "macos")]
use self::runtime::{
    commit_and_wait_metal, wait_for_completion_metal, with_runtime, with_runtime_for_device,
};
#[cfg(target_os = "macos")]
pub(crate) use self::runtime::{
    current_runtime_device_registry_id, runtime_initialization_error, with_runtime_for_session,
    MetalRuntime,
};
#[cfg(all(target_os = "macos", test))]
pub(crate) use j2k_metal_support::MetalSupportError;

#[cfg(target_os = "macos")]
mod direct_plan_types;
#[cfg(all(target_os = "macos", test))]
mod ht_forward_reader_tests;
#[cfg(all(target_os = "macos", test))]
mod ht_sigprop_context_tests;
#[cfg(target_os = "macos")]
use self::direct_plan_types::{
    PreparedClassicSubBand, PreparedClassicSubBandGroup, PreparedClassicSubBandGroupMember,
    PreparedDirectGrayscaleStep, PreparedDirectIdwt, PreparedHtExecutionOwner,
    PreparedHtPayloadSource, PreparedHtSubBand, PreparedHtSubBandGroup,
    PreparedHtSubBandGroupMember,
};
#[cfg(target_os = "macos")]
pub(crate) use self::direct_plan_types::{PreparedDirectColorPlan, PreparedDirectGrayscalePlan};
#[cfg(target_os = "macos")]
mod direct_plane_pack;
#[cfg(target_os = "macos")]
use self::direct_plane_pack::{
    encode_batched_mct_rgb8_to_surfaces_in_command_buffer,
    encode_mct_rgb8_to_surface_in_command_buffer, encode_plane_stage_to_surface_in_command_buffer,
    encode_repeated_mct_rgb8_to_surfaces_in_command_buffer,
    repeated_shared_direct_color_plan_count, PlaneStage,
};
#[cfg(target_os = "macos")]
mod direct_grayscale_execute;
#[cfg(target_os = "macos")]
pub(crate) use self::direct_grayscale_execute::{
    execute_hybrid_cpu_tier1_direct_color_plan, execute_hybrid_cpu_tier1_direct_color_plan_batch,
    execute_hybrid_cpu_tier1_direct_color_plan_with_device, execute_prepared_direct_color_plan,
    execute_prepared_direct_color_plan_batch, execute_prepared_direct_color_plan_with_device,
    execute_prepared_direct_grayscale_plan, execute_prepared_direct_grayscale_plan_batch,
    execute_prepared_direct_grayscale_plan_with_device,
    execute_repeated_prepared_direct_grayscale_plan,
    submit_prepared_direct_color_plan_batch_into_group,
    submit_prepared_direct_grayscale_plan_batch_into_group, DirectDestinationConsumerOrdering,
    SubmittedDirectDestination,
};
#[cfg(target_os = "macos")]
mod direct_prepare;
#[cfg(target_os = "macos")]
pub(crate) use self::direct_prepare::{
    prepare_direct_grayscale_plan, prepare_referenced_classic_color_plan,
    prepare_referenced_classic_grayscale_plan, prepare_referenced_classic_rgba_plan,
    prepare_referenced_htj2k_color_plan, prepare_referenced_htj2k_grayscale_plan,
    prepare_referenced_htj2k_rgba_plan,
};
#[cfg(target_os = "macos")]
mod direct_roi;
#[cfg(target_os = "macos")]
pub(crate) use self::direct_roi::crop_prepared_direct_grayscale_plan_to_output_region;
#[cfg(target_os = "macos")]
mod direct_stacked_batch;
#[cfg(target_os = "macos")]
use self::direct_stacked_batch::{
    encode_prepared_direct_color_plan_in_command_buffer,
    encode_repeated_direct_grayscale_plan_in_command_buffer,
    encode_stacked_direct_component_plane_batch, lookup_direct_band_slice,
    lookup_direct_band_slice_entry, signed_sample_bias,
    supports_stacked_direct_component_plane_batch, try_encode_stacked_mct_rgb8_direct_color_batch,
    DirectBandSlice, DirectColorPlanRequest, RepeatedDirectGrayscalePlanRequest,
    StackedDirectColorBatchRequest, StackedDirectComponentPlaneBatchRequest,
};
#[cfg(target_os = "macos")]
mod direct_surface_pack;
#[cfg(all(target_os = "macos", test))]
use self::direct_surface_pack::checked_metal_surface_len;
#[cfg(all(target_os = "macos", test))]
use self::direct_surface_pack::j2k_pack_kernel_name_for;
#[cfg(target_os = "macos")]
use self::direct_surface_pack::{
    copy_plane_samples, encode_gray_plane_to_surface_in_command_buffer_with_offset,
    encode_gray_plane_to_surface_in_encoder,
    encode_repeated_gray_plane_to_surfaces_in_command_buffer, output_shape_for,
};
#[cfg(target_os = "macos")]
mod direct_execute;
#[cfg(target_os = "macos")]
pub(crate) use self::direct_execute::{
    crop_prepared_direct_color_plan_to_output_region, prepare_direct_color_plan,
    prepare_direct_color_plan_for_cpu_upload,
};
#[cfg(all(target_os = "macos", test))]
use self::direct_execute::{
    prepared_direct_grayscale_plan_compute_encoder_count,
    prepared_repeated_direct_ht_cleanup_dispatch_count,
};
#[cfg(target_os = "macos")]
mod decode_cleanup;
#[cfg(target_os = "macos")]
pub(crate) use self::decode_cleanup::{
    decode_classic_cleanup_code_block_with_midpoint, decode_classic_cleanup_sub_band_with_midpoint,
    decode_ht_cleanup_code_block, decode_ht_cleanup_sub_band,
};
#[cfg(target_os = "macos")]
mod decode_dispatch;
#[cfg(all(target_os = "macos", test))]
pub(crate) use self::decode_dispatch::idwt::decode_irreversible97_staged_single_decomposition_idwt;
#[cfg(target_os = "macos")]
pub(crate) use self::decode_dispatch::idwt::{
    decode_irreversible97_single_decomposition_idwt,
    decode_openjpeg_irreversible97_single_decomposition_idwt,
    decode_reversible53_single_decomposition_idwt,
};
#[cfg(target_os = "macos")]
pub(crate) use self::decode_dispatch::mct::decode_inverse_mct;
#[cfg(target_os = "macos")]
pub(crate) use self::decode_dispatch::store::decode_store_component_and_capture;
#[cfg(target_os = "macos")]
mod forward_transform;
#[cfg(target_os = "macos")]
mod lossy_packet;
#[cfg(target_os = "macos")]
mod lossy_prepare;
#[cfg(target_os = "macos")]
pub(crate) use self::forward_transform::{
    encode_deinterleave_mct_to_f32, encode_deinterleave_to_f32, encode_forward_dwt53,
    encode_forward_dwt97,
};
#[cfg(target_os = "macos")]
pub(crate) use lossy_prepare::encode_resident_lossy_ht_packet;
#[cfg(target_os = "macos")]
mod lossless_prepare;
#[cfg(target_os = "macos")]
pub(crate) use self::lossless_prepare::{
    encode_forward_ict, encode_forward_rct, encode_quantize_subband,
    prepare_lossless_device_code_blocks, prepare_lossless_device_code_blocks_batch,
};
#[cfg(target_os = "macos")]
mod resident_codestream;
#[cfg(target_os = "macos")]
pub(crate) use self::resident_codestream::{
    encode_lossless_codestream_buffer_from_resident_tier1, encode_tier2_packetization,
    submit_lossless_codestream_buffers_from_prepared_classic_batch,
    submit_lossless_codestream_buffers_from_prepared_ht_batch,
};
#[cfg(target_os = "macos")]
mod resident_tier1;
#[cfg(target_os = "macos")]
pub(crate) use self::resident_tier1::{
    wait_resident_lossless_codestream_batch, J2kLosslessCodestreamAssemblyJob,
    J2kLosslessCodestreamBlockCodingMode, J2kLosslessDeviceBatchPrepareItem,
    J2kLosslessDeviceCodeBlock, J2kLosslessDevicePrepareJob,
    J2kPendingResidentLosslessCodestreamBatch, J2kPreparedLosslessDeviceCodeBlocks,
    J2kResidentPacketizationEncodeJob, J2kResidentPacketizationResolution,
    J2kResidentPacketizationSubband,
};
#[cfg(target_os = "macos")]
mod tier1_encode;
#[cfg(target_os = "macos")]
pub(crate) use self::tier1_encode::{
    encode_classic_tier1_code_block, encode_classic_tier1_code_blocks,
    encode_classic_tier1_prepared_device_code_blocks_resident, encode_ht_cleanup_code_block,
    encode_ht_cleanup_code_blocks, encode_ht_code_block_sets,
    encode_ht_prepared_device_code_blocks_resident,
    read_resident_ht_tier1_code_blocks_for_cpu_packetization,
};
#[cfg(all(target_os = "macos", test))]
pub(crate) use self::tier1_encode::{
    encode_classic_tier1_code_blocks_via_gpu_token_pack_for_test,
    encode_classic_tier1_code_blocks_via_ordered_tokens_cpu_pack_for_test,
    encode_classic_tier1_code_blocks_via_split_mq_byte_raw_tokens_gpu_pack_for_test,
    encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_cpu_pack_for_test,
    encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_gpu_pack_for_test,
};

#[cfg(target_os = "macos")]
fn required_classic_output_len(job: J2kCodeBlockDecodeJob<'_>) -> Result<usize, Error> {
    if job.height == 0 {
        return Ok(0);
    }

    job.output_stride
        .checked_mul(job.height as usize - 1)
        .and_then(|prefix| prefix.checked_add(job.width as usize))
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal output size overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
fn classic_style_flags(style: j2k_native::J2kCodeBlockStyle) -> u32 {
    let mut flags = 0u32;
    if style.reset_context_probabilities {
        flags |= abi::J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES;
    }
    if style.termination_on_each_pass {
        flags |= abi::J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS;
    }
    if style.vertically_causal_context {
        flags |= abi::J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT;
    }
    if style.segmentation_symbols {
        flags |= abi::J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS;
    }
    if style.selective_arithmetic_coding_bypass {
        flags |= abi::J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS;
    }
    flags
}

#[cfg(target_os = "macos")]
mod surface_decode;
#[cfg(all(test, target_os = "macos"))]
mod tests;
#[cfg(target_os = "macos")]
pub(crate) use self::surface_decode::{
    decode_image_to_surface, decode_image_to_surface_with_device, decode_region_scaled_to_surface,
    decode_region_scaled_to_surface_with_device,
};
