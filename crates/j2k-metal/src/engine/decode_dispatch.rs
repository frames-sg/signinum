// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use j2k_metal_support::{dispatch_1d_pipeline, dispatch_2d_pipeline, dispatch_3d_pipeline};

use crate::profile_env::{
    hybrid_stage_signpost, label_compute_encoder, SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE,
    SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE, SIGNPOST_DECODE_HYBRID_STORE_COMMAND_ENCODE,
};

use super::abi::{
    J2kClassicCleanupBatchJob, J2kClassicRepeatedBatchParams, J2kClassicSegment, J2kClassicStatus,
    J2kGrayStoreParams, J2kHtCleanupBatchJob, J2kHtRepeatedBatchParams, J2kHtStatus,
    J2kIdwt97StepParams, J2kIdwtSingleDecompositionParams, J2kInverseMctParams,
    J2kRepeatedGrayStoreParams, J2kRepeatedIdwtSingleDecompositionParams, J2kRepeatedStoreParams,
    J2kStoreParams, J2K_CLASSIC_MAX_HEIGHT, J2K_CLASSIC_MAX_WIDTH, J2K_CLASSIC_STATUS_OK,
};
use super::{
    checked_buffer_copy_into, checked_buffer_slice, commit_and_wait_metal, copied_slice_buffer,
    decode_classic_status_error, j2k_u32_param, new_command_buffer, new_compute_command_encoder,
    new_shared_buffer, take_classic_coefficients_scratch_buffer,
    take_classic_states_scratch_buffer, with_runtime, zeroed_shared_buffer, Buffer,
    CommandBufferRef, ComputeCommandEncoderRef, DirectIdwtCommandBuffers, DirectScratchBuffer,
    DirectStatusCheck, Error, HtCodeBlockDecodeJob, J2kInverseMctJob,
    J2kSingleDecompositionIdwtJob, J2kStoreComponentJob, J2kWaveletTransform, MetalRuntime,
    PixelFormat, PreparedClassicSubBand, PreparedClassicSubBandGroup, PreparedHtSubBand,
    PreparedHtSubBandGroup, Surface,
};

mod classic_cleanup;
mod classic_subband;
mod ht_chunks;
mod ht_distinct;
mod ht_subband;
pub(in crate::engine) mod idwt;
pub(in crate::engine) mod mct;
pub(in crate::engine) mod store;

pub(in crate::engine) use self::classic_cleanup::{
    classic_batch_is_plain_arithmetic, classic_batch_uses_plain_fast_path,
    classic_repeated_uses_plain_fast_path, dispatch_classic_cleanup_batched,
    dispatch_classic_cleanup_batched_in_encoder,
    dispatch_classic_cleanup_plain_dev_repeated_batched_in_command_buffer,
    dispatch_classic_cleanup_repeated_batched_in_command_buffer,
    dispatch_classic_store_repeated_batched_in_command_buffer,
    encode_distinct_classic_sub_band_groups_to_buffer_in_command_buffer,
    encode_distinct_classic_sub_band_groups_to_buffer_in_encoder,
    encode_distinct_classic_sub_bands_to_buffer_in_command_buffer,
    encode_distinct_classic_sub_bands_to_buffer_in_encoder, ClassicCleanupBatchDispatch,
    ClassicPlainDevRepeatedCleanupDispatch, ClassicRepeatedCleanupDispatch,
    ClassicRepeatedStoreDispatch,
};
pub(in crate::engine) use self::classic_subband::{
    encode_prepared_classic_sub_band_group_to_buffer_in_encoder,
    encode_prepared_classic_sub_band_to_buffer_in_encoder,
    encode_repeated_classic_sub_band_group_to_buffer_in_command_buffer,
    encode_repeated_classic_sub_band_to_buffer_in_command_buffer,
};
pub(in crate::engine) use self::ht_chunks::{
    default_metal_ht_chunk_limits, encode_metal_ht_batches_in_encoder,
    encode_repeated_metal_ht_batch_in_command_buffer, HtBatchInput, HtPayloadSource,
    MetalHtPipelineKind, PreparedMetalHtExecutionCache,
};
pub(in crate::engine) use self::ht_distinct::{
    encode_distinct_ht_sub_band_groups_to_buffer_in_command_buffer,
    encode_distinct_ht_sub_band_groups_to_buffer_in_encoder,
    encode_distinct_ht_sub_bands_to_buffer_in_command_buffer,
    encode_distinct_ht_sub_bands_to_buffer_in_encoder,
};
pub(in crate::engine) use self::ht_subband::{
    dispatch_zero_u32_buffer_in_encoder, encode_prepared_ht_sub_band_group_to_buffer_in_encoder,
    encode_prepared_ht_sub_band_to_buffer_in_encoder,
    encode_repeated_ht_sub_band_group_to_buffer_in_command_buffer,
    encode_repeated_ht_sub_band_to_buffer_in_command_buffer, ht_batch_output_word_count,
    ht_output_word_count, required_ht_output_len,
};
pub(in crate::engine) use self::idwt::{
    dispatch_irreversible97_repeated_buffers_in_command_buffer_with_offsets,
    dispatch_irreversible97_repeated_buffers_in_encoder_with_offsets,
    dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets,
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets,
    dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets,
    dispatch_reversible53_repeated_buffers_in_encoder_with_offsets,
    dispatch_reversible53_single_decomposition_buffers_in_command_buffer_with_offsets,
    dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets, IdwtSubBandBuffers,
    RepeatedIdwtDispatch, SingleIdwtDispatch,
};
pub(in crate::engine) use self::store::{
    dispatch_store_component_buffer_in_command_buffer_with_offsets,
    dispatch_store_component_buffer_in_encoder_with_offsets,
    dispatch_store_component_repeated_in_command_buffer,
    dispatch_store_component_repeated_in_encoder, encode_gray_store_to_destination_in_encoder,
    encode_gray_store_to_surface_in_encoder,
    encode_repeated_gray_store_to_surfaces_in_command_buffer, GrayStoreDestinationRequest,
};
