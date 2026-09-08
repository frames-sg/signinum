// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persistent shared-contract Metal batch decoder.

use super::{
    BatchDecodeOptions, EncodedImage, Error, MetalBackendSession, MetalBatchDecodeResult,
    MetalBatchGroup, MetalBatchGroupError, PreparedBatch, PreparedBatchGroup, PreparedImage,
};
#[cfg(target_os = "macos")]
use super::{
    MetalImageDestination, PixelFormat, PreparedColorPlanCache, PreparedGrayPlanCache,
    SubmittedMetalPreparedBatch,
};

/// Persistent Metal batch decoder.
///
/// The decoder retains one backend session across calls, including its Metal
/// runtime, command queue, pipelines, lookup tables, scratch pools, and prepared
/// direct-plan caches. Inputs queued in one call are grouped by the existing
/// distinct/repeated HTJ2K batch scheduler.
pub struct MetalBatchDecoder {
    pub(super) backend: MetalBackendSession,
    pub(super) options: BatchDecodeOptions,
    submission_count: u64,
    #[cfg(target_os = "macos")]
    pub(super) prepared_gray_plans: PreparedGrayPlanCache,
    #[cfg(target_os = "macos")]
    pub(super) prepared_color_plans: PreparedColorPlanCache,
}

impl core::fmt::Debug for MetalBatchDecoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut debug = f.debug_struct("MetalBatchDecoder");
        debug.field("backend", &self.backend);
        debug.field("options", &self.options);
        debug.field("submissions", &self.submission_count);
        #[cfg(target_os = "macos")]
        debug.field("prepared_gray_plans", &self.prepared_gray_plans.len());
        #[cfg(target_os = "macos")]
        debug.field("prepared_color_plans", &self.prepared_color_plans.len());
        debug.finish_non_exhaustive()
    }
}

impl MetalBatchDecoder {
    /// Create a persistent decoder using the system default Metal device.
    pub fn system_default() -> Result<Self, Error> {
        Self::system_default_with_options(BatchDecodeOptions::default())
    }

    /// Create a persistent decoder with retained shared batch policy.
    pub fn system_default_with_options(options: BatchDecodeOptions) -> Result<Self, Error> {
        let backend = MetalBackendSession::system_default()?;
        #[cfg(target_os = "macos")]
        {
            Ok(Self::with_backend_session_and_options(backend, options))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = backend;
            let _ = options;
            Err(Error::MetalUnavailable)
        }
    }

    /// Create a persistent decoder from an existing backend session.
    #[cfg(target_os = "macos")]
    pub fn with_backend_session(backend: MetalBackendSession) -> Self {
        Self::with_backend_session_and_options(backend, BatchDecodeOptions::default())
    }

    /// Create a persistent decoder from an existing backend session and
    /// retain its shared batch policy.
    #[cfg(target_os = "macos")]
    pub fn with_backend_session_and_options(
        backend: MetalBackendSession,
        options: BatchDecodeOptions,
    ) -> Self {
        Self {
            backend,
            options,
            submission_count: 0,
            prepared_gray_plans: PreparedGrayPlanCache::new(
                super::plan_cache::PREPARED_BATCH_PLAN_CACHE_CAP,
            ),
            prepared_color_plans: PreparedColorPlanCache::new(
                super::plan_cache::PREPARED_BATCH_PLAN_CACHE_CAP,
            ),
        }
    }

    /// Shared preparation and output policy retained by this session.
    #[must_use]
    pub const fn options(&self) -> BatchDecodeOptions {
        self.options
    }

    /// Backend session retained by this decoder.
    #[cfg(target_os = "macos")]
    pub fn backend_session(&self) -> &MetalBackendSession {
        &self.backend
    }

    /// Number of grouped codec submissions completed by the retained session.
    pub fn submissions(&self) -> Result<u64, Error> {
        Ok(self.submission_count)
    }

    #[cfg(target_os = "macos")]
    pub(super) fn record_submission(&mut self) {
        self.submission_count = self.submission_count.saturating_add(1);
    }

    /// Parse, validate, and group shared codec inputs for repeated Metal decode.
    pub fn prepare(&self, inputs: Vec<EncodedImage>) -> Result<PreparedBatch, Error> {
        j2k::prepare_batch(inputs, self.options).map_err(Error::from)
    }

    /// Regroup caller-supplied prepared images under this session's retained
    /// settings and output policy without reparsing their encoded bytes.
    pub fn prepare_prepared_images(
        &self,
        images: Vec<PreparedImage>,
    ) -> Result<PreparedBatch, Error> {
        j2k::prepare_batch_from_images(images, self.options).map_err(Error::from)
    }

    /// Prepare and decode owned codec inputs into Metal-resident groups.
    pub fn decode_batch(
        &mut self,
        inputs: Vec<EncodedImage>,
    ) -> Result<MetalBatchDecodeResult, Error> {
        let prepared = self.prepare(inputs)?;
        self.decode_prepared(&prepared)
    }

    /// Regroup and decode caller-supplied prepared images without reparsing
    /// their encoded bytes.
    pub fn decode_prepared_images(
        &mut self,
        images: Vec<PreparedImage>,
    ) -> Result<MetalBatchDecodeResult, Error> {
        let prepared = self.prepare_prepared_images(images)?;
        self.decode_prepared(&prepared)
    }

    /// Decode a reusable shared codec batch without consuming its inputs or plans.
    /// On Metal, preparation of the next group overlaps the previous submission;
    /// at most two committed groups retain scratch before completion is checked.
    pub fn decode_prepared(
        &mut self,
        prepared: &PreparedBatch,
    ) -> Result<MetalBatchDecodeResult, Error> {
        #[cfg(target_os = "macos")]
        {
            self.decode_prepared_bounded(prepared)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
                "J2K persistent Metal prepared batch output",
            );
            let mut groups = budget.try_vec(
                prepared.groups().len(),
                "J2K persistent Metal prepared output groups",
            )?;
            let mut group_errors = budget.try_vec(
                prepared.groups().len(),
                "J2K persistent Metal prepared group execution failures",
            )?;
            let mut errors = budget.try_vec(
                prepared.errors().len(),
                "J2K persistent Metal prepared indexed errors",
            )?;
            errors.extend_from_slice(prepared.errors());
            for group in prepared.groups() {
                match self.decode_prepared_group_with_options(group, prepared.options()) {
                    Ok(decoded) => groups.push(decoded),
                    Err(source) if source.session_is_unusable() => return Err(source),
                    Err(source) => group_errors.push(MetalBatchGroupError::new(group, source)),
                }
            }
            Ok(MetalBatchDecodeResult {
                groups,
                errors,
                group_errors,
            })
        }
    }

    /// Complete a batch with at most two committed groups retaining scratch.
    #[cfg(target_os = "macos")]
    fn decode_prepared_bounded(
        &mut self,
        prepared: &PreparedBatch,
    ) -> Result<MetalBatchDecodeResult, Error> {
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K synchronous prepared Metal batch",
        );
        let mut result = MetalBatchDecodeResult {
            groups: budget.try_vec(
                prepared.groups().len(),
                "J2K synchronous prepared Metal output groups",
            )?,
            errors: budget.try_vec(
                prepared.errors().len(),
                "J2K synchronous prepared indexed errors",
            )?,
            group_errors: budget.try_vec(
                prepared.groups().len(),
                "J2K synchronous prepared group errors",
            )?,
        };
        result.errors.extend_from_slice(prepared.errors());
        let retire = |pending: super::SubmittedMetalResidentGroup,
                      result: &mut MetalBatchDecodeResult| {
            match pending.wait() {
                Ok(group) => result.groups.push(group),
                Err((_, source)) if source.session_is_unusable() => return Err(*source),
                Err((source_indices, source)) => {
                    result.group_errors.push(MetalBatchGroupError {
                        source_indices,
                        source,
                    });
                }
            }
            Ok(())
        };
        let mut pending = None;
        let mut submission_errors = 0;
        for group in prepared.groups() {
            // Prepare and submit N+1 while N retains its command and scratch.
            // Retiring N before the next iteration bounds live work at two.
            let next = match self.submit_prepared_resident_group(group, prepared.options()) {
                Ok(next) => next,
                Err(source) if source.session_is_unusable() => return Err(source),
                Err(source) => {
                    // Match submit_prepared().wait(): submission failures precede
                    // completion failures, with each category in group order.
                    result
                        .group_errors
                        .insert(submission_errors, MetalBatchGroupError::new(group, source));
                    submission_errors += 1;
                    continue;
                }
            };
            if let Some(previous) = pending.replace(next) {
                retire(previous, &mut result)?;
            }
        }
        if let Some(pending) = pending {
            retire(pending, &mut result)?;
        }
        Ok(result)
    }

    /// Commit every representable shared prepared group to codec-owned Metal
    /// output storage without waiting on the CPU.
    ///
    /// Indexed preflight failures are retained in the returned guard. A
    /// non-fatal group submission failure is retained alongside other pending
    /// groups; a session-fatal failure aborts submission after safely retiring
    /// any groups already committed.
    #[cfg(target_os = "macos")]
    pub fn submit_prepared(
        &mut self,
        prepared: &PreparedBatch,
    ) -> Result<SubmittedMetalPreparedBatch, Error> {
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K submitted shared prepared Metal batch",
        );
        let mut pending_groups = budget.try_vec(
            prepared.groups().len(),
            "J2K submitted shared prepared Metal groups",
        )?;
        let mut errors = budget.try_vec(
            prepared.errors().len(),
            "J2K submitted shared prepared indexed errors",
        )?;
        errors.extend_from_slice(prepared.errors());
        let mut group_errors = budget.try_vec(
            prepared.groups().len(),
            "J2K submitted shared prepared group errors",
        )?;

        for group in prepared.groups() {
            match self.submit_prepared_resident_group(group, prepared.options()) {
                Ok(pending) => pending_groups.push(pending),
                Err(source) if source.session_is_unusable() => return Err(source),
                Err(source) => group_errors.push(MetalBatchGroupError::new(group, source)),
            }
        }
        Ok(SubmittedMetalPreparedBatch {
            pending_groups,
            errors,
            group_errors,
        })
    }

    /// Prepare and commit one shared encoded batch to codec-owned Metal output
    /// storage without waiting on the CPU.
    #[cfg(target_os = "macos")]
    pub fn submit_batch(
        &mut self,
        inputs: Vec<EncodedImage>,
    ) -> Result<SubmittedMetalPreparedBatch, Error> {
        let prepared = self.prepare(inputs)?;
        self.submit_prepared(&prepared)
    }

    /// Decode one homogeneous shared codec group using the preparation policy captured by the group.
    pub fn decode_prepared_group(
        &mut self,
        group: &PreparedBatchGroup,
    ) -> Result<MetalBatchGroup, Error> {
        self.decode_prepared_group_with_options(group, group.options())
    }
    /// Validate a caller-owned destination before a direct final-store encode.
    ///
    /// This establishes the checked external-write handoff used by framework
    /// adapters. Validation does not submit work or expose the raw buffer.
    #[cfg(target_os = "macos")]
    pub fn validate_destination(
        &self,
        destination: &MetalImageDestination,
        dimensions: (u32, u32),
        pixel_format: PixelFormat,
    ) -> Result<(), Error> {
        destination
            .validate_device(self.backend_session().device())
            .and_then(|()| destination.validate_image(dimensions, pixel_format))
            .map_err(|source| {
                crate::error::metal_kernel_support_error(
                    "J2K Metal external decode destination validation failed",
                    source,
                )
            })
    }
}

impl j2k::BatchDecoder for MetalBatchDecoder {
    type Output = MetalBatchDecodeResult;
    type Error = Error;

    fn options(&self) -> BatchDecodeOptions {
        self.options
    }

    fn decode_prepared(&mut self, prepared: &PreparedBatch) -> Result<Self::Output, Self::Error> {
        MetalBatchDecoder::decode_prepared(self, prepared)
    }
}
