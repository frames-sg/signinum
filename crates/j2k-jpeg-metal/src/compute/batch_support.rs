// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use std::cell::RefCell;
#[cfg(all(test, target_os = "macos"))]
use std::ffi::OsStr;
use std::time::Duration;

use crate::metal_types::Buffer;
use j2k_jpeg::adapter::JpegEntropyCheckpointV1;

use crate::buffers::MetalBatchScratch;
use crate::{batch, Error, Surface};

use super::{
    checked_u32, fast_decode_status_error, first_decode_error_status, JpegEntropyCheckpointHost,
    MetalRuntime,
};
use crate::batch_allocation::{BatchMetadataBudget, BatchMetadataRequest};

const FAST420_BATCH_TIMING_ENV: &str = "J2K_JPEG_METAL_FAST420_BATCH_TIMING";

#[cfg(target_os = "macos")]
thread_local! {
    static FAST420_BATCH_PROFILE_SUMMARY: RefCell<j2k_profile::ProfileSummary> =
        RefCell::new(new_fast420_batch_profile_summary().emit_on_drop());
}

#[cfg(target_os = "macos")]
fn new_fast420_batch_profile_summary() -> j2k_profile::ProfileSummary {
    match j2k_profile::same_summary_labels(&["mode", "dimensions"])
        .and_then(j2k_profile::ProfileSummary::new)
    {
        Ok(summary) => summary,
        Err(error) => {
            j2k_profile::emit_profile_error("jpeg_metal_fast420_summary_init", &error);
            j2k_profile::ProfileSummary::default()
        }
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FastBatchDecodeMode {
    Fused,
    #[cfg(test)]
    SplitCoeffIdct,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct FastBatchTiming {
    pub(super) accepted: Duration,
    pub(super) entropy_concat: Duration,
    pub(super) buffer_alloc: Duration,
    pub(super) encode_decode: Duration,
    pub(super) wait_decode: Duration,
    pub(super) encode_pack: Duration,
    pub(super) wait_pack: Duration,
    pub(super) total: Duration,
}

#[cfg(target_os = "macos")]
impl FastBatchTiming {
    fn micros(duration: Duration) -> u128 {
        duration.as_micros()
    }

    pub(super) fn log(
        self,
        tag: &'static str,
        label: &str,
        tile_count: usize,
        dimensions: (u32, u32),
        segment_count: usize,
    ) {
        let fields = match self.profile_fields(label, tile_count, dimensions, segment_count) {
            Ok(fields) => fields,
            Err(error) => {
                j2k_profile::emit_profile_error("jpeg_metal_fast420_fields", &error);
                return;
            }
        };
        j2k_profile::emit_profile_fields(
            fast420_batch_timing_stage_mode(),
            &FAST420_BATCH_PROFILE_SUMMARY,
            "jpeg",
            "decode",
            tag,
            &fields,
        );
    }

    fn profile_fields(
        self,
        label: &str,
        tile_count: usize,
        dimensions: (u32, u32),
        segment_count: usize,
    ) -> j2k_profile::ProfileResult<[j2k_profile::ProfileField; 12]> {
        Ok([
            j2k_profile::ProfileField::label("mode", label)?,
            j2k_profile::ProfileField::metric("tiles", tile_count)?,
            j2k_profile::ProfileField::label(
                "dimensions",
                format_args!("{}x{}", dimensions.0, dimensions.1),
            )?,
            j2k_profile::ProfileField::metric("segments", segment_count)?,
            j2k_profile::ProfileField::metric("accepted_us", Self::micros(self.accepted))?,
            j2k_profile::ProfileField::metric(
                "entropy_concat_us",
                Self::micros(self.entropy_concat),
            )?,
            j2k_profile::ProfileField::metric("buffer_alloc_us", Self::micros(self.buffer_alloc))?,
            j2k_profile::ProfileField::metric(
                "encode_decode_us",
                Self::micros(self.encode_decode),
            )?,
            j2k_profile::ProfileField::metric("wait_decode_us", Self::micros(self.wait_decode))?,
            j2k_profile::ProfileField::metric("encode_pack_us", Self::micros(self.encode_pack))?,
            j2k_profile::ProfileField::metric("wait_pack_us", Self::micros(self.wait_pack))?,
            j2k_profile::ProfileField::metric("total_us", Self::micros(self.total))?,
        ])
    }
}

#[cfg(all(test, target_os = "macos"))]
mod entropy_tests;

#[cfg(all(test, target_os = "macos"))]
mod profile_tests {
    use super::*;

    #[test]
    fn fast_batch_profile_fields_reject_oversized_labels() {
        let oversized = "x".repeat(j2k_profile::ProfileLimits::default().max_token_bytes() + 1);
        assert!(matches!(
            FastBatchTiming::default().profile_fields(&oversized, 1, (16, 8), 1),
            Err(j2k_profile::ProfileError::LimitExceeded {
                what: "field value",
                ..
            })
        ));
    }
}

#[cfg(target_os = "macos")]
pub(super) fn fast_batch_decode_mode() -> FastBatchDecodeMode {
    FastBatchDecodeMode::Fused
}

#[cfg(target_os = "macos")]
pub(super) fn fast420_batch_timing_enabled() -> bool {
    fast420_batch_timing_stage_mode() != j2k_profile::ProfileStageMode::Disabled
}

#[cfg(target_os = "macos")]
fn fast420_batch_timing_stage_mode() -> j2k_profile::ProfileStageMode {
    j2k_profile::profile_stage_mode_from_env(FAST420_BATCH_TIMING_ENV)
}

#[cfg(all(test, target_os = "macos"))]
pub(super) fn fast420_batch_timing_value_enabled(value: Option<&OsStr>) -> bool {
    fast420_batch_timing_value_mode(value) != j2k_profile::ProfileStageMode::Disabled
}

#[cfg(all(test, target_os = "macos"))]
pub(super) fn fast420_batch_timing_value_mode(
    value: Option<&OsStr>,
) -> j2k_profile::ProfileStageMode {
    j2k_profile::profile_stage_mode_from_value(value.and_then(OsStr::to_str))
}

#[cfg(target_os = "macos")]
pub(super) struct BatchEntropyBuffers {
    pub(super) payload: Buffer,
    pub(super) offsets: Buffer,
    pub(super) lens: Buffer,
    pub(super) checkpoints: Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct BatchEntropyBufferKeys {
    pub(super) payload: &'static str,
    pub(super) offsets: &'static str,
    pub(super) lens: &'static str,
    pub(super) checkpoints: &'static str,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct BatchEntropyBufferPlan {
    pub(super) keys: BatchEntropyBufferKeys,
    pub(super) tile_count: usize,
    pub(super) segment_count: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct BatchEntropyLabels {
    pub(super) offset: &'static str,
    pub(super) len: &'static str,
}

#[cfg(target_os = "macos")]
pub(super) struct BatchEntropyMetadata {
    pub(super) payload_len: usize,
    pub(super) offsets: Vec<u32>,
    pub(super) lens: Vec<u32>,
    pub(super) checkpoints: Vec<JpegEntropyCheckpointHost>,
}

#[cfg(target_os = "macos")]
pub(super) fn batch_entropy_metadata<'a>(
    entropy_bytes_iter: impl Iterator<Item = &'a [u8]> + Clone,
    entropy_checkpoints_iter: impl Iterator<Item = &'a [JpegEntropyCheckpointV1]> + Clone,
    tile_count: usize,
    segment_count: usize,
    external_live_bytes: usize,
    labels: BatchEntropyLabels,
) -> Result<Option<BatchEntropyMetadata>, Error> {
    let total_entropy_len = entropy_bytes_iter
        .clone()
        .map(<[u8]>::len)
        .try_fold(0usize, usize::checked_add)
        .ok_or(j2k_core::BatchInfrastructureError::AllocationTooLarge {
            what: "JPEG Metal batch entropy bytes",
            requested: usize::MAX,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })?;
    if total_entropy_len == 0 {
        return Ok(None);
    }

    let checkpoint_count = crate::batch_allocation::checked_count_product(
        tile_count,
        segment_count,
        "JPEG Metal batch entropy checkpoints",
    )?;
    let entropy_tile_count = crate::batch_allocation::checked_count_sum(
        entropy_bytes_iter.clone().map(|_| 1_usize),
        "JPEG Metal batch entropy tile count",
    )?;
    let checkpoint_tile_count = crate::batch_allocation::checked_count_sum(
        entropy_checkpoints_iter.clone().map(|_| 1_usize),
        "JPEG Metal batch checkpoint tile count",
    )?;
    let actual_checkpoint_count = crate::batch_allocation::checked_count_sum(
        entropy_checkpoints_iter
            .clone()
            .map(<[JpegEntropyCheckpointV1]>::len),
        "JPEG Metal batch checkpoint count",
    )?;
    if entropy_tile_count != tile_count
        || checkpoint_tile_count != tile_count
        || actual_checkpoint_count != checkpoint_count
    {
        return Err(Error::MetalKernel {
            message: "JPEG Metal batch entropy metadata shape mismatch".to_string(),
        });
    }
    let mut budget = BatchMetadataBudget::with_external_live(
        "JPEG Metal batch entropy host data",
        external_live_bytes,
    );
    // Entropy stays borrowed from the retained packet owners and is copied
    // directly into checked Metal storage. Only the newly allocated host-side
    // offset/checkpoint vectors belong in this live host metadata budget.
    budget.preflight(&[
        BatchMetadataRequest::of::<u32>(tile_count),
        BatchMetadataRequest::of::<u32>(tile_count),
        BatchMetadataRequest::of::<JpegEntropyCheckpointHost>(checkpoint_count),
    ])?;
    let mut offsets = budget.try_vec(tile_count, "JPEG Metal batch entropy offsets")?;
    let mut lens = budget.try_vec(tile_count, "JPEG Metal batch entropy lengths")?;
    let mut checkpoints =
        budget.try_vec(checkpoint_count, "JPEG Metal batch entropy checkpoints")?;
    let mut payload_offset = 0_usize;
    for (entropy_bytes, entropy_checkpoints) in entropy_bytes_iter.zip(entropy_checkpoints_iter) {
        offsets.push(checked_u32(payload_offset, labels.offset)?);
        lens.push(checked_u32(entropy_bytes.len(), labels.len)?);
        payload_offset = payload_offset.checked_add(entropy_bytes.len()).ok_or(
            j2k_core::BatchInfrastructureError::AllocationTooLarge {
                what: "JPEG Metal batch entropy bytes",
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            },
        )?;
        checkpoints.extend(
            entropy_checkpoints
                .iter()
                .copied()
                .map(JpegEntropyCheckpointHost::from),
        );
    }
    debug_assert_eq!(payload_offset, total_entropy_len);

    Ok(Some(BatchEntropyMetadata {
        payload_len: total_entropy_len,
        offsets,
        lens,
        checkpoints,
    }))
}

#[cfg(target_os = "macos")]
pub(super) fn batch_entropy_buffers<'a>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    scratch: &mut MetalBatchScratch,
    plan: BatchEntropyBufferPlan,
    entropy_bytes_iter: impl Iterator<Item = &'a [u8]> + Clone,
    entropy_checkpoints_iter: impl Iterator<Item = &'a [JpegEntropyCheckpointV1]> + Clone,
) -> Result<Option<BatchEntropyBuffers>, Error> {
    let owner_budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal batch entropy buffer owners",
        requests,
    )?;
    let Some(metadata) = batch_entropy_metadata(
        entropy_bytes_iter.clone(),
        entropy_checkpoints_iter,
        plan.tile_count,
        plan.segment_count,
        owner_budget.live_bytes(),
        BatchEntropyLabels {
            offset: "region scaled batch entropy offset",
            len: "region scaled batch entropy length",
        },
    )?
    else {
        return Ok(None);
    };

    batch_entropy_buffers_from_metadata(runtime, scratch, plan.keys, entropy_bytes_iter, &metadata)
        .map(Some)
}

#[cfg(target_os = "macos")]
pub(super) fn batch_entropy_buffers_from_metadata<'a>(
    runtime: &MetalRuntime,
    scratch: &mut MetalBatchScratch,
    keys: BatchEntropyBufferKeys,
    entropy_bytes_iter: impl Iterator<Item = &'a [u8]>,
    metadata: &BatchEntropyMetadata,
) -> Result<BatchEntropyBuffers, Error> {
    Ok(BatchEntropyBuffers {
        payload: scratch.shared_buffer_with_byte_slices(
            &runtime.device,
            keys.payload,
            metadata.payload_len,
            entropy_bytes_iter,
        )?,
        offsets: scratch.shared_buffer_with_slice(
            &runtime.device,
            keys.offsets,
            &metadata.offsets,
        )?,
        lens: scratch.shared_buffer_with_slice(&runtime.device, keys.lens, &metadata.lens)?,
        checkpoints: scratch.shared_buffer_with_slice(
            &runtime.device,
            keys.checkpoints,
            &metadata.checkpoints,
        )?,
    })
}

#[cfg(target_os = "macos")]
pub(super) fn surface_batch_error_results(
    requests: &[batch::QueuedRequest],
    status_buffer: &Buffer,
    total_decode_threads: u32,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let Some(status) = first_decode_error_status(status_buffer, total_decode_threads)? else {
        return Ok(None);
    };
    let mut budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal surface batch error results",
        requests,
    )?;
    let mut results = budget.try_vec(
        requests.len(),
        "JPEG Metal surface batch error result slots",
    )?;
    let error = fast_decode_status_error(status);
    for _ in requests {
        results.push(Err(error.clone()));
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
pub(super) fn surface_batch_success_results(
    requests: &[batch::QueuedRequest],
    out_buffer: &Buffer,
    dimensions: (u32, u32),
    pixel_format: j2k_core::PixelFormat,
    tile_count: usize,
    out_tile_len: usize,
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Vec<Result<Surface, Error>>, Error> {
    let mut budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal surface batch results",
        requests,
    )?;
    let mut results = budget.try_vec(tile_count, "JPEG Metal surface batch result slots")?;
    for index in 0..tile_count {
        let offset = index * out_tile_len;
        let surface = if let Some(output) = output {
            Surface::from_batch_output_buffer_offset(output, dimensions, pixel_format, offset)
        } else {
            Surface::from_metal_buffer_offset(out_buffer.clone(), dimensions, pixel_format, offset)?
        };
        results.push(Ok(surface));
    }
    Ok(results)
}

#[cfg(target_os = "macos")]
pub(super) fn region_scaled_batch_error_results(
    requests: &[batch::QueuedRequest],
    status_buffer: &Buffer,
    total_decode_threads: u32,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    surface_batch_error_results(requests, status_buffer, total_decode_threads)
}

#[cfg(target_os = "macos")]
pub(super) fn texture_batch_error_results(
    requests: &[batch::QueuedRequest],
    status_buffer: &Buffer,
    total_decode_threads: u32,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    let Some(status) = first_decode_error_status(status_buffer, total_decode_threads)? else {
        return Ok(None);
    };
    let mut budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal texture batch error results",
        requests,
    )?;
    let mut results = budget.try_vec(
        requests.len(),
        "JPEG Metal texture batch error result slots",
    )?;
    let error = fast_decode_status_error(status);
    for _ in requests {
        results.push(Err(error.clone()));
    }
    Ok(Some(results))
}
