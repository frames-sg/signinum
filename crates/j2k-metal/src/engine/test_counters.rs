// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

macro_rules! test_atomic_counter {
    ($counter:ident, $reset:ident, $load:ident) => {
        static $counter: AtomicUsize = AtomicUsize::new(0);

        pub(crate) fn $reset() {
            $counter.store(0, Ordering::Relaxed);
        }

        pub(crate) fn $load() -> usize {
            $counter.load(Ordering::Relaxed)
        }
    };
}

test_atomic_counter!(
    HT_BATCH_COEFFICIENT_COPY_BLITS,
    reset_ht_batch_coefficient_copy_blits_for_test,
    ht_batch_coefficient_copy_blits_for_test
);
test_atomic_counter!(
    HYBRID_STACKED_COMPONENT_BATCHES,
    reset_hybrid_stacked_component_batches_for_test,
    hybrid_stacked_component_batches_for_test
);
test_atomic_counter!(
    HYBRID_REPEATED_OUTPUT_BLITS,
    reset_hybrid_repeated_output_blits_for_test,
    hybrid_repeated_output_blits_for_test
);
test_atomic_counter!(
    HYBRID_CPU_DECODE_WORKER_INITS,
    reset_hybrid_cpu_decode_worker_inits_for_test,
    hybrid_cpu_decode_worker_inits_for_test
);
test_atomic_counter!(
    HYBRID_CPU_DECODE_INPUTS,
    reset_hybrid_cpu_decode_inputs_for_test,
    hybrid_cpu_decode_inputs_for_test
);
test_atomic_counter!(
    FLATTENED_HYBRID_CPU_DECODE_BATCHES,
    reset_flattened_hybrid_cpu_decode_batches_for_test,
    flattened_hybrid_cpu_decode_batches_for_test
);
std::thread_local! {
    static STACKED_COMPONENT_BATCHES: Cell<usize> = const { Cell::new(0) };
    static RESIDENT_GPU_TIMESTAMP_QUERIES: Cell<usize> = const { Cell::new(0) };
    static RESIDENT_CODESTREAM_COMMAND_BUFFER_WAITS: Cell<usize> = const { Cell::new(0) };
    static DIRECT_TIER1_INPUT_BUFFER_PREPARES: Cell<usize> = const { Cell::new(0) };
    static DIRECT_TIER1_INPUT_BUFFER_RUNTIME: Cell<usize> = const { Cell::new(0) };
    static HYBRID_CPU_DECODE_INPUTS_FOR_THREAD: Cell<usize> = const { Cell::new(0) };
    static LOSSLESS_DEINTERLEAVE_RCT_FUSED_DISPATCHES: Cell<usize> = const { Cell::new(0) };
    static CLASSIC_GPU_TOKEN_PACK_DISPATCHES: Cell<usize> = const { Cell::new(0) };
    static CLASSIC_SPLIT_MQ_BYTE_GPU_TOKEN_PACK_DISPATCHES: Cell<usize> = const { Cell::new(0) };
    static HT_IMMUTABLE_PAYLOAD_UPLOADS: Cell<usize> = const { Cell::new(0) };
    static HT_IMMUTABLE_JOB_UPLOADS: Cell<usize> = const { Cell::new(0) };
    static METAL_COMMAND_BUFFERS: Cell<usize> = const { Cell::new(0) };
    static METAL_COMPUTE_ENCODERS: Cell<usize> = const { Cell::new(0) };
    static DIRECT_DESTINATION_EVENT_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static DIRECT_DESTINATION_EVENT_SIGNALS: Cell<usize> = const { Cell::new(0) };
    static DIRECT_DESTINATION_EVENT_WAITS: Cell<usize> = const { Cell::new(0) };
}

pub(crate) fn reset_direct_destination_event_bridge_for_test() {
    DIRECT_DESTINATION_EVENT_ALLOCATIONS.with(|count| count.set(0));
    DIRECT_DESTINATION_EVENT_SIGNALS.with(|count| count.set(0));
    DIRECT_DESTINATION_EVENT_WAITS.with(|count| count.set(0));
}

pub(crate) fn direct_destination_event_bridge_for_test() -> (usize, usize, usize) {
    (
        DIRECT_DESTINATION_EVENT_ALLOCATIONS.with(Cell::get),
        DIRECT_DESTINATION_EVENT_SIGNALS.with(Cell::get),
        DIRECT_DESTINATION_EVENT_WAITS.with(Cell::get),
    )
}

pub(crate) fn record_direct_destination_event_allocation() {
    DIRECT_DESTINATION_EVENT_ALLOCATIONS.with(|count| count.set(count.get().saturating_add(1)));
}

pub(crate) fn record_direct_destination_event_signal() {
    DIRECT_DESTINATION_EVENT_SIGNALS.with(|count| count.set(count.get().saturating_add(1)));
}

pub(crate) fn record_direct_destination_event_wait() {
    DIRECT_DESTINATION_EVENT_WAITS.with(|count| count.set(count.get().saturating_add(1)));
}

pub(crate) fn reset_metal_command_buffers_for_test() {
    METAL_COMMAND_BUFFERS.with(|count| count.set(0));
}

pub(crate) fn metal_command_buffers_for_test() -> usize {
    METAL_COMMAND_BUFFERS.with(Cell::get)
}

pub(crate) fn record_metal_command_buffer() {
    METAL_COMMAND_BUFFERS.with(|count| count.set(count.get().saturating_add(1)));
}

pub(crate) fn reset_metal_compute_encoders_for_test() {
    METAL_COMPUTE_ENCODERS.with(|count| count.set(0));
}

pub(crate) fn metal_compute_encoders_for_test() -> usize {
    METAL_COMPUTE_ENCODERS.with(Cell::get)
}

pub(crate) fn record_metal_compute_encoder() {
    METAL_COMPUTE_ENCODERS.with(|count| count.set(count.get().saturating_add(1)));
}

pub(crate) fn reset_ht_immutable_payload_uploads_for_test() {
    HT_IMMUTABLE_PAYLOAD_UPLOADS.with(|uploads| uploads.set(0));
}

pub(crate) fn ht_immutable_payload_uploads_for_test() -> usize {
    HT_IMMUTABLE_PAYLOAD_UPLOADS.with(Cell::get)
}

pub(crate) fn reset_ht_immutable_job_uploads_for_test() {
    HT_IMMUTABLE_JOB_UPLOADS.with(|uploads| uploads.set(0));
}

pub(crate) fn ht_immutable_job_uploads_for_test() -> usize {
    HT_IMMUTABLE_JOB_UPLOADS.with(Cell::get)
}

pub(crate) fn reset_stacked_component_batches_for_test() {
    STACKED_COMPONENT_BATCHES.with(|counter| counter.set(0));
}

pub(crate) fn stacked_component_batches_for_test() -> usize {
    STACKED_COMPONENT_BATCHES.with(Cell::get)
}

pub(crate) fn record_stacked_component_batch() {
    STACKED_COMPONENT_BATCHES.with(|counter| counter.set(counter.get().saturating_add(1)));
}

pub(crate) fn reset_resident_gpu_timestamp_queries_for_test() {
    RESIDENT_GPU_TIMESTAMP_QUERIES.with(|queries| queries.set(0));
}

pub(crate) fn resident_gpu_timestamp_queries_for_test() -> usize {
    RESIDENT_GPU_TIMESTAMP_QUERIES.with(Cell::get)
}

pub(crate) fn record_resident_gpu_timestamp_query() {
    RESIDENT_GPU_TIMESTAMP_QUERIES.with(|queries| queries.set(queries.get() + 1));
}

pub(crate) fn reset_resident_codestream_command_buffer_waits_for_test() {
    RESIDENT_CODESTREAM_COMMAND_BUFFER_WAITS.with(|waits| waits.set(0));
}

pub(crate) fn resident_codestream_command_buffer_waits_for_test() -> usize {
    RESIDENT_CODESTREAM_COMMAND_BUFFER_WAITS.with(Cell::get)
}

pub(crate) fn record_resident_codestream_command_buffer_wait() {
    RESIDENT_CODESTREAM_COMMAND_BUFFER_WAITS.with(|waits| waits.set(waits.get() + 1));
}

pub(crate) fn reset_direct_tier1_input_buffer_prepares_for_test() {
    DIRECT_TIER1_INPUT_BUFFER_PREPARES.with(|counter| counter.set(0));
    DIRECT_TIER1_INPUT_BUFFER_RUNTIME.with(|identity| identity.set(0));
}

pub(crate) fn direct_tier1_input_buffer_prepares_for_test() -> usize {
    DIRECT_TIER1_INPUT_BUFFER_PREPARES.with(Cell::get)
}

pub(crate) fn direct_tier1_input_buffer_runtime_for_test() -> usize {
    DIRECT_TIER1_INPUT_BUFFER_RUNTIME.with(Cell::get)
}

pub(crate) fn record_direct_tier1_input_buffer_prepare(_runtime: &super::MetalRuntime) {
    DIRECT_TIER1_INPUT_BUFFER_PREPARES.with(|counter| counter.set(counter.get() + 1));
}

pub(crate) fn record_direct_tier1_input_buffer_runtime(runtime: &super::MetalRuntime) {
    DIRECT_TIER1_INPUT_BUFFER_RUNTIME.with(|identity| {
        identity.set(std::ptr::from_ref(runtime).addr());
    });
}

pub(crate) fn reset_thread_hybrid_cpu_decode_inputs_for_test() {
    HYBRID_CPU_DECODE_INPUTS_FOR_THREAD.with(|counter| counter.set(0));
}

pub(crate) fn thread_hybrid_cpu_decode_inputs_for_test() -> usize {
    HYBRID_CPU_DECODE_INPUTS_FOR_THREAD.with(Cell::get)
}

pub(crate) fn reset_lossless_deinterleave_rct_fused_dispatches_for_test() {
    LOSSLESS_DEINTERLEAVE_RCT_FUSED_DISPATCHES.with(|dispatches| dispatches.set(0));
}

pub(crate) fn lossless_deinterleave_rct_fused_dispatches_for_test() -> usize {
    LOSSLESS_DEINTERLEAVE_RCT_FUSED_DISPATCHES.with(Cell::get)
}

pub(crate) fn record_lossless_deinterleave_rct_fused_dispatch() {
    LOSSLESS_DEINTERLEAVE_RCT_FUSED_DISPATCHES
        .with(|dispatches| dispatches.set(dispatches.get().saturating_add(1)));
}

pub(crate) fn reset_classic_gpu_token_pack_dispatches_for_test() {
    CLASSIC_GPU_TOKEN_PACK_DISPATCHES.with(|dispatches| dispatches.set(0));
}

pub(crate) fn classic_gpu_token_pack_dispatches_for_test() -> usize {
    CLASSIC_GPU_TOKEN_PACK_DISPATCHES.with(Cell::get)
}

pub(crate) fn record_classic_gpu_token_pack_dispatch() {
    CLASSIC_GPU_TOKEN_PACK_DISPATCHES
        .with(|dispatches| dispatches.set(dispatches.get().saturating_add(1)));
}

pub(crate) fn reset_classic_split_mq_byte_gpu_token_pack_dispatches_for_test() {
    CLASSIC_SPLIT_MQ_BYTE_GPU_TOKEN_PACK_DISPATCHES.with(|dispatches| dispatches.set(0));
}

pub(crate) fn classic_split_mq_byte_gpu_token_pack_dispatches_for_test() -> usize {
    CLASSIC_SPLIT_MQ_BYTE_GPU_TOKEN_PACK_DISPATCHES.with(Cell::get)
}

pub(crate) fn record_classic_split_mq_byte_gpu_token_pack_dispatch() {
    CLASSIC_SPLIT_MQ_BYTE_GPU_TOKEN_PACK_DISPATCHES
        .with(|dispatches| dispatches.set(dispatches.get().saturating_add(1)));
}

pub(crate) fn record_ht_batch_coefficient_copy_blit() {
    HT_BATCH_COEFFICIENT_COPY_BLITS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_hybrid_stacked_component_batch() {
    HYBRID_STACKED_COMPONENT_BATCHES.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_hybrid_repeated_output_blit() {
    HYBRID_REPEATED_OUTPUT_BLITS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_hybrid_cpu_decode_worker_init() {
    HYBRID_CPU_DECODE_WORKER_INITS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_hybrid_cpu_decode_inputs(count: usize) {
    HYBRID_CPU_DECODE_INPUTS.fetch_add(count, Ordering::Relaxed);
    HYBRID_CPU_DECODE_INPUTS_FOR_THREAD
        .with(|counter| counter.set(counter.get().saturating_add(count)));
}

pub(crate) fn record_flattened_hybrid_cpu_decode_batch() {
    FLATTENED_HYBRID_CPU_DECODE_BATCHES.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_ht_immutable_payload_upload() {
    HT_IMMUTABLE_PAYLOAD_UPLOADS.with(|uploads| uploads.set(uploads.get().saturating_add(1)));
}

pub(crate) fn record_ht_immutable_job_upload() {
    HT_IMMUTABLE_JOB_UPLOADS.with(|uploads| uploads.set(uploads.get().saturating_add(1)));
}

std::thread_local! {
    static IDWT97_STAGE_SEQUENCES: Cell<usize> = const { Cell::new(0) };
    static IDWT97_LOGICAL_REQUESTED_POSITIONS: Cell<usize> = const { Cell::new(0) };
    static IDWT97_STAGE_DISPATCHES: Cell<usize> = const { Cell::new(0) };
    static IDWT_HOST_OVERWRITTEN_OUTPUT_UPLOAD_BYTES: Cell<usize> = const { Cell::new(0) };
    static IDWT_HOST_TEMPORARY_READBACK_VEC_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static IDWT_HOST_TEMPORARY_READBACK_VEC_BYTES: Cell<usize> = const { Cell::new(0) };
}

pub(crate) fn reset_idwt_host_transfer_counters_for_test() {
    IDWT_HOST_OVERWRITTEN_OUTPUT_UPLOAD_BYTES.set(0);
    IDWT_HOST_TEMPORARY_READBACK_VEC_ALLOCATIONS.set(0);
    IDWT_HOST_TEMPORARY_READBACK_VEC_BYTES.set(0);
}

pub(crate) fn idwt_host_transfer_counters_for_test() -> (usize, usize, usize) {
    (
        IDWT_HOST_OVERWRITTEN_OUTPUT_UPLOAD_BYTES.get(),
        IDWT_HOST_TEMPORARY_READBACK_VEC_ALLOCATIONS.get(),
        IDWT_HOST_TEMPORARY_READBACK_VEC_BYTES.get(),
    )
}

pub(crate) fn record_idwt_host_overwritten_output_upload(bytes: usize) {
    IDWT_HOST_OVERWRITTEN_OUTPUT_UPLOAD_BYTES.set(
        IDWT_HOST_OVERWRITTEN_OUTPUT_UPLOAD_BYTES
            .get()
            .saturating_add(bytes),
    );
}

pub(crate) fn record_idwt_host_temporary_readback_vec(bytes: usize) {
    IDWT_HOST_TEMPORARY_READBACK_VEC_ALLOCATIONS.set(
        IDWT_HOST_TEMPORARY_READBACK_VEC_ALLOCATIONS
            .get()
            .saturating_add(1),
    );
    IDWT_HOST_TEMPORARY_READBACK_VEC_BYTES.set(
        IDWT_HOST_TEMPORARY_READBACK_VEC_BYTES
            .get()
            .saturating_add(bytes),
    );
}

pub(crate) fn reset_idwt97_stage_sequences_for_test() {
    IDWT97_STAGE_SEQUENCES.set(0);
}

pub(crate) fn idwt97_stage_sequences_for_test() -> usize {
    IDWT97_STAGE_SEQUENCES.get()
}

pub(crate) fn record_idwt97_stage_sequence() {
    IDWT97_STAGE_SEQUENCES.set(IDWT97_STAGE_SEQUENCES.get() + 1);
}

pub(crate) fn reset_idwt97_logical_dispatches_for_test() {
    IDWT97_LOGICAL_REQUESTED_POSITIONS.set(0);
    IDWT97_STAGE_DISPATCHES.set(0);
}

pub(crate) fn idwt97_logical_dispatches_for_test() -> (usize, usize) {
    (
        IDWT97_LOGICAL_REQUESTED_POSITIONS.get(),
        IDWT97_STAGE_DISPATCHES.get(),
    )
}

pub(crate) fn record_idwt97_logical_dispatch(grid: (u32, u32, u32)) {
    let positions = (grid.0 as usize)
        .saturating_mul(grid.1 as usize)
        .saturating_mul(grid.2 as usize);
    IDWT97_LOGICAL_REQUESTED_POSITIONS.set(
        IDWT97_LOGICAL_REQUESTED_POSITIONS
            .get()
            .saturating_add(positions),
    );
    IDWT97_STAGE_DISPATCHES.set(IDWT97_STAGE_DISPATCHES.get().saturating_add(1));
}
