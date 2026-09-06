// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG Metal compute domains and their controlled crate-facing entry points.
#[cfg(target_os = "macos")]
use crate::metal_types::{Buffer, CommandBuffer, CommandBufferRef, ComputePipelineState, Device};
#[cfg(test)]
use j2k_core::BackendRequest;
use j2k_core::{BufferError, PixelFormat, Rect};
use j2k_jpeg::{
    adapter::{
        JpegEntropyCheckpointV1, JpegFast420PacketV1, JpegFast422PacketV1, JpegFast444PacketV1,
        JpegHuffmanTable,
    },
    ColorSpace as JpegColorSpace, Decoder as CpuDecoder,
};
#[cfg(all(target_os = "macos", test))]
use j2k_metal_support::MetalSupportError;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLPixelFormat, MTLResourceOptions};

#[cfg(target_os = "macos")]
pub(crate) use crate::abi::{
    JpegBaselineEncodeHuffmanTable, JpegBaselineEncodeParams, JpegBaselineEncodeStatus,
    JpegBaselineEntropyEncodeBatchJob, JpegBaselineEntropyEncodeJob, JpegDecodeStatus,
    JpegEntropyCheckpointHost, JpegFast420BatchParams, JpegFast420Params, JpegFast420ScaledParams,
    JpegFast420TextureBatchParams, JpegFast420WindowedPackParams, JpegFast444Params,
    JpegFast444ScaledParams, JpegFast444TextureBatchParams, JpegFastRegionScaledBatchParams,
    JpegPackParams, JpegRgb8ToRgbaTextureParams, JpegTexturePackBatchParams,
    JpegWindowedPackBatchParams, JpegWindowedTexturePackBatchParams, PreparedHuffmanHost,
    FAST420_TEXTURE_BOUNDARY_META_WORDS, FAST420_TEXTURE_BOUNDARY_SAMPLE_BYTES,
    FAST420_TEXTURE_VERTICAL_META_WORDS, FAST420_TEXTURE_VERTICAL_SAMPLE_BYTES,
    FAST422_TEXTURE_BOUNDARY_META_WORDS, FAST422_TEXTURE_BOUNDARY_SAMPLE_BYTES, MODE_GRAY,
    MODE_RGB, MODE_YCBCR, OUT_GRAY, OUT_RGB, OUT_RGBA,
};
#[cfg(target_os = "macos")]
use crate::buffers::{
    new_decode_plane_buffer, new_private_buffer, new_shared_buffer_with_data, MetalBatchScratch,
};
use crate::{batch, Error, JpegFastPackets, Surface};

#[cfg(target_os = "macos")]
pub(crate) mod batch_entry;
#[cfg(target_os = "macos")]
mod batch_full;
mod batch_plan;
#[cfg(target_os = "macos")]
mod batch_region;
#[cfg(target_os = "macos")]
mod batch_support;
#[cfg(target_os = "macos")]
mod command;
#[cfg(target_os = "macos")]
pub(crate) mod encode;
mod fast_packets;
#[cfg(target_os = "macos")]
mod kernel_helpers;
#[cfg(target_os = "macos")]
mod pack_dispatch;
#[cfg(target_os = "macos")]
mod pipeline_registry;
#[cfg(target_os = "macos")]
mod region_scaled_plan;
#[cfg(target_os = "macos")]
mod runtime;
#[cfg(target_os = "macos")]
mod scratch_pool;
#[cfg(all(target_os = "macos", test))]
use self::pipeline_registry::SHADER_SOURCE;
#[cfg(target_os = "macos")]
pub(crate) mod single_decode;
#[cfg(target_os = "macos")]
mod status;
#[cfg(all(target_os = "macos", test))]
mod texture_tuning;
mod viewport_cache;
#[cfg(target_os = "macos")]
pub(crate) mod viewport_compose;
#[cfg(all(target_os = "macos", test))]
use self::batch_full::try_decode_fast_subsampled_full_rgb_batch_to_surfaces_with_mode_and_output;
#[cfg(target_os = "macos")]
use self::batch_full::{
    try_decode_fast444_full_rgb_batch_to_surfaces,
    try_decode_fast444_full_rgb_batch_to_surfaces_into_output,
    try_decode_fast444_full_rgba_batch_to_textures,
    try_decode_fast_subsampled_full_rgb_batch_to_surfaces,
    try_decode_fast_subsampled_full_rgb_batch_to_surfaces_into_output,
    try_decode_fast_subsampled_full_rgba_batch_to_textures,
};
use self::batch_plan::{
    batched_fast_packets, core_rect_to_jpeg, BatchDeviceBufferCache, BatchedDecodeItem,
    BatchedFastPacket,
};
#[cfg(target_os = "macos")]
use self::batch_region::{
    try_decode_fast420_region_scaled_rgb_batch_to_surfaces,
    try_decode_fast420_region_scaled_rgb_batch_to_surfaces_into_output,
    try_decode_fast420_region_scaled_rgba_batch_to_textures,
    try_decode_fast422_region_scaled_rgb_batch_to_surfaces,
    try_decode_fast422_region_scaled_rgb_batch_to_surfaces_into_output,
    try_decode_fast422_region_scaled_rgba_batch_to_textures,
    try_decode_fast444_region_scaled_rgb_batch_to_surfaces,
    try_decode_fast444_region_scaled_rgb_batch_to_surfaces_into_output,
    try_decode_fast444_region_scaled_rgba_batch_to_textures,
    try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output,
    try_decode_repeated_region_scaled_batch_to_surfaces,
};
#[cfg(target_os = "macos")]
use self::batch_support::{
    batch_entropy_buffers, batch_entropy_buffers_from_metadata, batch_entropy_metadata,
    fast420_batch_timing_enabled, fast_batch_decode_mode, region_scaled_batch_error_results,
    surface_batch_error_results, surface_batch_success_results, texture_batch_error_results,
    BatchEntropyBufferKeys, BatchEntropyBufferPlan, BatchEntropyBuffers, BatchEntropyLabels,
    BatchEntropyMetadata, FastBatchDecodeMode, FastBatchTiming,
};
#[cfg(all(test, target_os = "macos"))]
use self::batch_support::{fast420_batch_timing_value_enabled, fast420_batch_timing_value_mode};
use self::fast_packets::{
    checked_entropy_segment_count, entropy_checkpoints_buffer, entropy_decode_thread_count,
    fast444_params, fast444_region_params, fast444_scaled_params, fast444_scaled_region_params,
    fast_subsampled_full_mcu_scaled_window, fast_subsampled_full_mcu_window,
    fast_subsampled_params, fast_subsampled_region_params, fast_subsampled_scaled_params,
    fast_subsampled_scaled_region_params, fast_subsampled_windowed_pack_params_for_dims,
    mcu_range_for_rect, restart_offsets_buffer, restart_work_for_mcu_range, FastRegionScaledMetal,
    FastScratchKeys, FastSubsampledMetal, FastSubsampledPacket, FastTextureRepairCtx,
};
#[cfg(all(test, target_os = "macos"))]
use self::kernel_helpers::choose_1d_threadgroup_width;
#[cfg(target_os = "macos")]
use self::kernel_helpers::{
    bind_fast_decode_entropy_inputs, bind_three_plane_pack, dispatch_1d_pipeline,
    dispatch_2d_pipeline, dispatch_3d_pipeline, fast_packet_huffman_tables, packed_pair_extent,
    pixel_format_to_out_format, plane_mode_to_u32, FastDecodeEntropyInputs,
};
#[cfg(target_os = "macos")]
use self::pack_dispatch::{
    batch_output_buffer_or_new, checked_u32, copy_grouped_surfaces_to_output,
    copy_rgb8_surfaces_to_rgba_textures, dispatch_rgba_texture_pack,
    dispatch_windowed_rgba_texture_pack, encode_fast444_batch_item,
    encode_fast444_region_batch_item, encode_fast444_scaled_batch_item,
    encode_fast444_scaled_region_batch_item, encode_fast_subsampled_op_batch_item,
    encode_fast_subsampled_region_batch_item, encode_fast_subsampled_scaled_batch_item,
    texture_batch_success_results, validate_rgba_texture_batch_output,
    Fast444ScaledRegionBatchItemRequest, FastSubsampledOpBatchItemRequest,
};
#[cfg(all(target_os = "macos", test))]
use self::pack_dispatch::{encode_split_coeff_idct_passes, SplitCoeffIdctPasses};
#[cfg(target_os = "macos")]
use self::region_scaled_plan::{
    fast444_packets_share_region_scaled_batch_shape, fast444_region_scaled_batch_groups,
    fast_subsampled_full_rgb_batch_groups, fast_subsampled_packets_share_full_rgb_batch_shape,
    fast_subsampled_region_scaled_batch_groups, fast_subsampled_region_scaled_batch_plan,
    windowed_texture_pack_params, RegionScaledBatchPlan,
};
#[cfg(all(target_os = "macos", test))]
use self::single_decode::{
    try_decode_fast420_region_to_surface, try_decode_fast420_scaled_region_to_surface,
    try_decode_fast420_scaled_to_surface, try_decode_fast422_region_to_surface,
    try_decode_fast422_scaled_to_surface, try_decode_fast422_to_surface,
    try_decode_fast444_region_to_surface, try_decode_fast444_scaled_region_to_surface,
    try_decode_fast444_scaled_to_surface, try_decode_fast444_to_surface,
};
#[cfg(target_os = "macos")]
use self::single_decode::{
    try_decode_fast420_scaled_region_to_surface_with_status,
    try_decode_fast422_scaled_region_to_surface,
    try_decode_fast444_scaled_region_to_surface_with_mode_and_status,
};
#[cfg(target_os = "macos")]
use self::status::{
    decode_status_buffer, fast422_status_error, fast_decode_status_error, first_decode_error_status,
};
use self::viewport_cache::{cached_plane_stage, PlaneMode, PlaneStage};

#[cfg(all(target_os = "macos", test))]
pub(crate) use crate::buffers::{
    jpeg_private_buffer_allocations_for_test, jpeg_shared_buffer_allocations_for_test,
    reset_jpeg_private_buffer_allocations_for_test, reset_jpeg_shared_buffer_allocations_for_test,
};

#[cfg(target_os = "macos")]
use self::command::{
    commit_and_wait_jpeg, new_blit_command_encoder, new_command_buffer,
    new_compute_command_encoder, wait_for_completion_jpeg,
};
#[cfg(target_os = "macos")]
use self::runtime::{
    private_jpeg_tile_from_fast_rgb_buffer, with_runtime, with_runtime_for_session,
    FastRgbDecodeBuffer,
};
#[cfg(target_os = "macos")]
pub(crate) use self::runtime::{runtime_initialization_error, MetalRuntime};

#[cfg(target_os = "macos")]
const REGION_SCALED_BATCH_CHUNK: usize = 8;

#[cfg(all(test, target_os = "macos"))]
mod tests;
