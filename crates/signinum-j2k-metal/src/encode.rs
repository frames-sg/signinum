// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use crate::compute;
#[cfg(target_os = "macos")]
use metal::Buffer;
#[cfg(target_os = "macos")]
use rayon::prelude::*;
use signinum_core::DeviceSubmission;
#[cfg(target_os = "macos")]
use signinum_core::{BackendKind, DeviceSurface, PixelFormat};
#[cfg(target_os = "macos")]
use signinum_j2k::{
    EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation, J2kProgressionOrder,
};
use signinum_j2k::{EncodedJ2k, J2kLosslessEncodeOptions, J2kLosslessSamples};
#[cfg(target_os = "macos")]
use signinum_j2k_native::{
    EncodeProgressionOrder, J2kPacketizationPacketDescriptor, J2kSubBandType,
};
use signinum_j2k_native::{
    EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, J2kEncodeDispatchReport, J2kEncodeStageAccelerator,
    J2kForwardDwt53Job, J2kForwardDwt53Output, J2kForwardRctJob, J2kHtCodeBlockEncodeJob,
    J2kPacketizationEncodeJob, J2kTier1CodeBlockEncodeJob,
};
#[cfg(all(test, target_os = "macos"))]
use std::cell::Cell;
#[cfg(target_os = "macos")]
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;
#[cfg(target_os = "macos")]
use std::time::Instant;

/// Encode-stage accelerator for JPEG 2000 Metal work.
///
/// The type is wired into the native encoder hook interface and reports
/// dispatches for each required encode stage.
#[derive(Debug, Default, Clone)]
pub struct MetalEncodeStageAccelerator {
    forward_rct_attempts: usize,
    forward_dwt53_attempts: usize,
    tier1_code_block_attempts: usize,
    ht_code_block_attempts: usize,
    packetization_attempts: usize,
    forward_rct_dispatches: usize,
    forward_dwt53_dispatches: usize,
    tier1_code_block_dispatches: usize,
    ht_code_block_dispatches: usize,
    packetization_dispatches: usize,
}

impl MetalEncodeStageAccelerator {
    pub fn forward_rct_attempts(&self) -> usize {
        self.forward_rct_attempts
    }

    pub fn forward_dwt53_attempts(&self) -> usize {
        self.forward_dwt53_attempts
    }

    pub fn tier1_code_block_attempts(&self) -> usize {
        self.tier1_code_block_attempts
    }

    pub fn ht_code_block_attempts(&self) -> usize {
        self.ht_code_block_attempts
    }

    pub fn packetization_attempts(&self) -> usize {
        self.packetization_attempts
    }

    pub fn forward_rct_dispatches(&self) -> usize {
        self.forward_rct_dispatches
    }

    pub fn forward_dwt53_dispatches(&self) -> usize {
        self.forward_dwt53_dispatches
    }

    pub fn tier1_code_block_dispatches(&self) -> usize {
        self.tier1_code_block_dispatches
    }

    pub fn ht_code_block_dispatches(&self) -> usize {
        self.ht_code_block_dispatches
    }

    pub fn packetization_dispatches(&self) -> usize {
        self.packetization_dispatches
    }
}

#[cfg(target_os = "macos")]
fn metal_dispatch_result(
    result: &Result<(), crate::Error>,
    message: &'static str,
) -> Result<bool, &'static str> {
    match result {
        Ok(()) => Ok(true),
        Err(crate::Error::MetalUnavailable) => Ok(false),
        Err(_) => Err(message),
    }
}

#[cfg(target_os = "macos")]
fn metal_dispatch_option<T>(
    result: Result<T, crate::Error>,
    message: &'static str,
) -> Result<Option<T>, &'static str> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(crate::Error::MetalUnavailable) => Ok(None),
        Err(_) => Err(message),
    }
}

impl J2kEncodeStageAccelerator for MetalEncodeStageAccelerator {
    fn dispatch_report(&self) -> J2kEncodeDispatchReport {
        J2kEncodeDispatchReport {
            forward_rct: self.forward_rct_dispatches,
            forward_dwt53: self.forward_dwt53_dispatches,
            tier1_code_block: self.tier1_code_block_dispatches,
            ht_code_block: self.ht_code_block_dispatches,
            packetization: self.packetization_dispatches,
        }
    }

    fn encode_forward_rct(
        &mut self,
        job: J2kForwardRctJob<'_>,
    ) -> core::result::Result<bool, &'static str> {
        self.forward_rct_attempts = self.forward_rct_attempts.saturating_add(1);
        #[cfg(target_os = "macos")]
        {
            let result = compute::encode_forward_rct(job.plane0, job.plane1, job.plane2);
            let dispatched =
                metal_dispatch_result(&result, "Metal forward RCT encode kernel failed")?;
            if dispatched {
                self.forward_rct_dispatches = self.forward_rct_dispatches.saturating_add(1);
            }
            Ok(dispatched)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            Ok(false)
        }
    }

    fn encode_forward_dwt53(
        &mut self,
        job: J2kForwardDwt53Job<'_>,
    ) -> core::result::Result<Option<J2kForwardDwt53Output>, &'static str> {
        self.forward_dwt53_attempts = self.forward_dwt53_attempts.saturating_add(1);
        if job.num_levels == 0 {
            return Ok(None);
        }
        #[cfg(target_os = "macos")]
        {
            let output = metal_dispatch_option(
                compute::encode_forward_dwt53(job.samples, job.width, job.height, job.num_levels),
                "Metal forward 5/3 DWT encode kernel failed",
            )?;
            if output.is_some() {
                self.forward_dwt53_dispatches = self.forward_dwt53_dispatches.saturating_add(1);
            }
            Ok(output)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            Ok(None)
        }
    }

    fn encode_tier1_code_block(
        &mut self,
        job: J2kTier1CodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<EncodedJ2kCodeBlock>, &'static str> {
        self.tier1_code_block_attempts = self.tier1_code_block_attempts.saturating_add(1);
        #[cfg(target_os = "macos")]
        {
            let encoded = metal_dispatch_option(
                compute::encode_classic_tier1_code_block(job),
                "Metal classic Tier-1 encode kernel failed",
            )?;
            if encoded.is_some() {
                self.tier1_code_block_dispatches =
                    self.tier1_code_block_dispatches.saturating_add(1);
            }
            Ok(encoded)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            Ok(None)
        }
    }

    fn encode_tier1_code_blocks(
        &mut self,
        jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
    ) -> core::result::Result<Option<Vec<EncodedJ2kCodeBlock>>, &'static str> {
        self.tier1_code_block_attempts = self.tier1_code_block_attempts.saturating_add(jobs.len());
        #[cfg(target_os = "macos")]
        {
            let encoded = metal_dispatch_option(
                compute::encode_classic_tier1_code_blocks(jobs),
                "Metal classic Tier-1 encode batch kernel failed",
            )?;
            if encoded.is_some() && !jobs.is_empty() {
                self.tier1_code_block_dispatches =
                    self.tier1_code_block_dispatches.saturating_add(1);
            }
            Ok(encoded)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = jobs;
            Ok(None)
        }
    }

    fn encode_ht_code_block(
        &mut self,
        job: J2kHtCodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<EncodedHtJ2kCodeBlock>, &'static str> {
        self.ht_code_block_attempts = self.ht_code_block_attempts.saturating_add(1);
        #[cfg(target_os = "macos")]
        {
            let encoded = metal_dispatch_option(
                compute::encode_ht_cleanup_code_block(job),
                "Metal HTJ2K code-block encode kernel failed",
            )?;
            if encoded.is_some() {
                self.ht_code_block_dispatches = self.ht_code_block_dispatches.saturating_add(1);
            }
            Ok(encoded)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            Ok(None)
        }
    }

    fn encode_ht_code_blocks(
        &mut self,
        jobs: &[J2kHtCodeBlockEncodeJob<'_>],
    ) -> core::result::Result<Option<Vec<EncodedHtJ2kCodeBlock>>, &'static str> {
        self.ht_code_block_attempts = self.ht_code_block_attempts.saturating_add(jobs.len());
        #[cfg(target_os = "macos")]
        {
            let encoded = metal_dispatch_option(
                compute::encode_ht_cleanup_code_blocks(jobs),
                "Metal HTJ2K code-block encode batch kernel failed",
            )?;
            if encoded.is_some() && !jobs.is_empty() {
                self.ht_code_block_dispatches = self.ht_code_block_dispatches.saturating_add(1);
            }
            Ok(encoded)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = jobs;
            Ok(None)
        }
    }

    fn encode_packetization(
        &mut self,
        job: J2kPacketizationEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        self.packetization_attempts = self.packetization_attempts.saturating_add(1);
        #[cfg(target_os = "macos")]
        {
            let encoded = metal_dispatch_option(
                compute::encode_tier2_packetization(job),
                "Metal Tier-2 packetization encode kernel failed",
            )?;
            if encoded.is_some() {
                self.packetization_dispatches = self.packetization_dispatches.saturating_add(1);
            }
            Ok(encoded)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            Ok(None)
        }
    }
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
pub struct MetalLosslessEncodeTile<'a> {
    pub buffer: &'a Buffer,
    pub byte_offset: usize,
    pub width: u32,
    pub height: u32,
    pub pitch_bytes: usize,
    pub output_width: u32,
    pub output_height: u32,
    pub format: PixelFormat,
}

#[cfg(not(target_os = "macos"))]
#[derive(Debug, Clone, Copy)]
pub struct MetalLosslessEncodeTile<'a> {
    _private: core::marker::PhantomData<&'a ()>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalLosslessEncodeResidency {
    pub coefficient_prep_used: bool,
    pub packetization_used: bool,
    pub codestream_assembly_used: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalLosslessEncodeOutcome {
    pub encoded: EncodedJ2k,
    pub input_copy_used: bool,
    pub resident: MetalLosslessEncodeResidency,
    pub input_copy_duration: Duration,
    pub encode_duration: Duration,
    pub gpu_duration: Option<Duration>,
    pub validation_duration: Duration,
}

#[cfg(target_os = "macos")]
/// JPEG 2000 codestream bytes owned by a Metal buffer.
///
/// The buffer is CPU-readable for the current padded resident encode API, so
/// callers can stream `codestream_bytes()` into file or network writers without
/// first materializing an owned `Vec<u8>`.
pub struct MetalEncodedJ2k {
    pub codestream_buffer: Buffer,
    pub byte_offset: usize,
    pub byte_len: usize,
    pub capacity: usize,
    pub width: u32,
    pub height: u32,
    pub components: u8,
    pub bit_depth: u8,
    pub signed: bool,
}

#[cfg(target_os = "macos")]
impl MetalEncodedJ2k {
    /// Borrow the finished codestream bytes from the backing Metal buffer.
    pub fn codestream_bytes(&self) -> Result<&[u8], crate::Error> {
        let end = self.byte_offset.checked_add(self.byte_len).ok_or_else(|| {
            crate::Error::MetalKernel {
                message: "J2K Metal codestream byte range overflow".to_string(),
            }
        })?;
        let buffer_len = usize::try_from(self.codestream_buffer.length()).map_err(|_| {
            crate::Error::MetalKernel {
                message: "J2K Metal codestream buffer length exceeds usize".to_string(),
            }
        })?;
        if end > buffer_len {
            return Err(crate::Error::MetalKernel {
                message: "J2K Metal codestream byte range exceeds buffer length".to_string(),
            });
        }
        let ptr = self.codestream_buffer.contents().cast::<u8>();
        if ptr.is_null() {
            return Err(crate::Error::MetalKernel {
                message: "J2K Metal codestream buffer is not CPU-readable".to_string(),
            });
        }
        Ok(unsafe { core::slice::from_raw_parts(ptr.add(self.byte_offset), self.byte_len) })
    }

    /// Materialize the buffer-backed codestream into the compatibility `Vec` API shape.
    pub fn to_encoded_j2k(&self) -> Result<EncodedJ2k, crate::Error> {
        Ok(EncodedJ2k {
            codestream: self.codestream_bytes()?.to_vec(),
            backend: BackendKind::Metal,
            width: self.width,
            height: self.height,
            components: self.components,
            bit_depth: self.bit_depth,
            signed: self.signed,
        })
    }
}

#[cfg(not(target_os = "macos"))]
pub struct MetalEncodedJ2k {
    _private: (),
}

/// Metal lossless encode report for buffer-backed codestream output.
pub struct MetalLosslessBufferEncodeOutcome {
    pub encoded: MetalEncodedJ2k,
    pub input_copy_used: bool,
    pub resident: MetalLosslessEncodeResidency,
    pub input_copy_duration: Duration,
    pub encode_duration: Duration,
    pub gpu_duration: Option<Duration>,
    pub validation_duration: Duration,
}

/// Tuning knobs for resident Metal lossless J2K/HTJ2K tile batch encode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MetalLosslessEncodeConfig {
    /// Requested maximum number of tiles submitted concurrently.
    ///
    /// `None` uses the crate default and still clamps by the memory budget.
    pub gpu_encode_inflight_tiles: Option<usize>,
    /// Resident encode memory budget in bytes.
    ///
    /// `None` uses `min(10 GiB, hw_memsize * 0.40)` when host memory can be
    /// discovered.
    pub gpu_encode_memory_budget_bytes: Option<usize>,
}

/// Resolved resident Metal lossless J2K/HTJ2K tile batch encode metrics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MetalLosslessEncodeBatchStats {
    pub configured_inflight_tiles: Option<usize>,
    pub effective_inflight_tiles: usize,
    pub configured_memory_budget_bytes: Option<usize>,
    pub effective_memory_budget_bytes: usize,
    pub estimated_peak_bytes_per_tile: usize,
    pub max_observed_inflight_tiles: usize,
    pub encode_wall_duration: Duration,
}

/// Resident Metal lossless J2K/HTJ2K tile batch output and batch-level metrics.
pub struct MetalLosslessBufferEncodeBatchOutcome {
    pub outcomes: Vec<MetalLosslessBufferEncodeOutcome>,
    pub stats: MetalLosslessEncodeBatchStats,
}

#[cfg(target_os = "macos")]
pub struct SubmittedJ2kLosslessMetalEncode {
    inner: SubmittedJ2kLosslessMetalEncodeBatch,
}

#[cfg(target_os = "macos")]
pub struct SubmittedJ2kLosslessMetalEncodeBatch {
    state: SubmittedJ2kLosslessMetalEncodeBatchState,
}

#[cfg(target_os = "macos")]
enum SubmittedJ2kLosslessMetalEncodeBatchState {
    Ready(Vec<EncodedJ2k>),
    Deferred {
        tiles: Vec<OwnedMetalLosslessEncodeTile>,
        options: J2kLosslessEncodeOptions,
        session: crate::MetalBackendSession,
        staging: MetalEncodeInputStaging,
        config: MetalLosslessEncodeConfig,
    },
}

#[cfg(target_os = "macos")]
struct OwnedMetalLosslessEncodeTile {
    buffer: Buffer,
    byte_offset: usize,
    width: u32,
    height: u32,
    pitch_bytes: usize,
    output_width: u32,
    output_height: u32,
    format: PixelFormat,
}

#[cfg(target_os = "macos")]
impl OwnedMetalLosslessEncodeTile {
    fn from_tile(tile: MetalLosslessEncodeTile<'_>) -> Self {
        Self {
            buffer: tile.buffer.to_owned(),
            byte_offset: tile.byte_offset,
            width: tile.width,
            height: tile.height,
            pitch_bytes: tile.pitch_bytes,
            output_width: tile.output_width,
            output_height: tile.output_height,
            format: tile.format,
        }
    }

    fn as_tile(&self) -> MetalLosslessEncodeTile<'_> {
        MetalLosslessEncodeTile {
            buffer: &self.buffer,
            byte_offset: self.byte_offset,
            width: self.width,
            height: self.height,
            pitch_bytes: self.pitch_bytes,
            output_width: self.output_width,
            output_height: self.output_height,
            format: self.format,
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub struct SubmittedJ2kLosslessMetalEncode {
    _private: (),
}

#[cfg(not(target_os = "macos"))]
pub struct SubmittedJ2kLosslessMetalEncodeBatch {
    _private: (),
}

#[cfg(target_os = "macos")]
impl DeviceSubmission for SubmittedJ2kLosslessMetalEncode {
    type Output = EncodedJ2k;
    type Error = crate::Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        let mut encoded = self.inner.wait()?;
        if encoded.len() != 1 {
            return Err(crate::Error::MetalKernel {
                message: "submitted J2K Metal single encode produced an unexpected batch length"
                    .to_string(),
            });
        }
        Ok(encoded.remove(0))
    }
}

#[cfg(target_os = "macos")]
impl DeviceSubmission for SubmittedJ2kLosslessMetalEncodeBatch {
    type Output = Vec<EncodedJ2k>;
    type Error = crate::Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        match self.state {
            SubmittedJ2kLosslessMetalEncodeBatchState::Ready(encoded) => Ok(encoded),
            SubmittedJ2kLosslessMetalEncodeBatchState::Deferred {
                tiles,
                options,
                session,
                staging,
                config,
            } => {
                encode_lossless_owned_tiles_with_report(&tiles, options, &session, staging, config)
                    .map(|outcomes| {
                        outcomes
                            .into_iter()
                            .map(|outcome| outcome.encoded)
                            .collect()
                    })
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
impl DeviceSubmission for SubmittedJ2kLosslessMetalEncode {
    type Output = EncodedJ2k;
    type Error = crate::Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        let _ = self;
        Err(crate::Error::MetalUnavailable)
    }
}

#[cfg(not(target_os = "macos"))]
impl DeviceSubmission for SubmittedJ2kLosslessMetalEncodeBatch {
    type Output = Vec<EncodedJ2k>;
    type Error = crate::Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        let _ = self;
        Err(crate::Error::MetalUnavailable)
    }
}

#[cfg(target_os = "macos")]
pub fn encode_lossless_from_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJ2k, crate::Error> {
    submit_lossless_from_metal_buffer(tile, options, session)?.wait()
}

#[cfg(target_os = "macos")]
pub fn encode_lossless_from_metal_buffer_to_metal(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalEncodedJ2k, crate::Error> {
    Ok(encode_lossless_from_metal_buffer_to_metal_with_report(tile, options, session)?.encoded)
}

#[cfg(target_os = "macos")]
pub fn submit_lossless_from_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalEncode, crate::Error> {
    let inner = submit_lossless_from_metal_buffers(&[tile], options, session)?;
    Ok(SubmittedJ2kLosslessMetalEncode { inner })
}

#[cfg(target_os = "macos")]
pub fn encode_lossless_from_metal_buffer_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessEncodeOutcome, crate::Error> {
    let mut accelerator = MetalEncodeStageAccelerator::default();
    encode_lossless_tile_with_report(
        tile,
        *options,
        session,
        MetalEncodeInputStaging::CopyAndPad,
        &mut accelerator,
    )
}

#[cfg(target_os = "macos")]
pub fn encode_lossless_from_metal_buffer_to_metal_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessBufferEncodeOutcome, crate::Error> {
    let mut outcomes =
        encode_lossless_from_metal_buffers_to_metal_with_report(&[tile], options, session)?;
    if outcomes.len() != 1 {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal single buffer encode produced an unexpected batch length"
                .to_string(),
        });
    }
    Ok(outcomes.remove(0))
}

#[cfg(target_os = "macos")]
pub fn encode_lossless_from_padded_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJ2k, crate::Error> {
    submit_lossless_from_padded_metal_buffer(tile, options, session)?.wait()
}

#[cfg(target_os = "macos")]
pub fn encode_lossless_from_padded_metal_buffer_to_metal(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalEncodedJ2k, crate::Error> {
    Ok(
        encode_lossless_from_padded_metal_buffer_to_metal_with_report(tile, options, session)?
            .encoded,
    )
}

#[cfg(target_os = "macos")]
pub fn submit_lossless_from_padded_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalEncode, crate::Error> {
    let inner = submit_lossless_from_padded_metal_buffers(&[tile], options, session)?;
    Ok(SubmittedJ2kLosslessMetalEncode { inner })
}

#[cfg(target_os = "macos")]
pub fn encode_lossless_from_padded_metal_buffer_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessEncodeOutcome, crate::Error> {
    let mut accelerator = MetalEncodeStageAccelerator::default();
    encode_lossless_tile_with_report(
        tile,
        *options,
        session,
        MetalEncodeInputStaging::AlreadyPaddedContiguous,
        &mut accelerator,
    )
}

#[cfg(target_os = "macos")]
pub fn encode_lossless_from_padded_metal_buffer_to_metal_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessBufferEncodeOutcome, crate::Error> {
    let mut outcomes =
        encode_lossless_from_padded_metal_buffers_to_metal_with_report(&[tile], options, session)?;
    if outcomes.len() != 1 {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal single buffer encode produced an unexpected batch length"
                .to_string(),
        });
    }
    Ok(outcomes.remove(0))
}

#[cfg(target_os = "macos")]
pub fn encode_lossless_from_metal_buffers(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<EncodedJ2k>, crate::Error> {
    submit_lossless_from_metal_buffers(tiles, options, session)?.wait()
}

#[cfg(target_os = "macos")]
pub fn encode_lossless_from_metal_buffers_to_metal(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalEncodedJ2k>, crate::Error> {
    Ok(
        encode_lossless_from_metal_buffers_to_metal_with_report(tiles, options, session)?
            .into_iter()
            .map(|outcome| outcome.encoded)
            .collect(),
    )
}

#[cfg(target_os = "macos")]
pub fn submit_lossless_from_metal_buffers(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    submit_lossless_from_metal_buffers_with_config(
        tiles,
        options,
        session,
        MetalLosslessEncodeConfig::default(),
    )
}

#[cfg(target_os = "macos")]
pub fn submit_lossless_from_metal_buffers_with_config(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    submit_lossless_tiles(
        tiles,
        *options,
        session,
        MetalEncodeInputStaging::CopyAndPad,
        config,
    )
}

#[cfg(target_os = "macos")]
pub fn encode_lossless_from_metal_buffers_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
    encode_lossless_tiles_with_report(
        tiles,
        *options,
        session,
        MetalEncodeInputStaging::CopyAndPad,
    )
}

#[cfg(target_os = "macos")]
pub fn encode_lossless_from_metal_buffers_to_metal_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessBufferEncodeOutcome>, crate::Error> {
    Ok(encode_lossless_from_metal_buffers_to_metal_batch(
        tiles,
        options,
        session,
        MetalLosslessEncodeConfig::default(),
    )?
    .outcomes)
}

#[cfg(target_os = "macos")]
pub fn encode_lossless_from_metal_buffers_to_metal_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<MetalLosslessBufferEncodeBatchOutcome, crate::Error> {
    encode_lossless_tiles_to_metal_buffer_batch(
        tiles,
        *options,
        session,
        MetalEncodeInputStaging::CopyAndPad,
        config,
    )
}

#[cfg(target_os = "macos")]
pub fn encode_lossless_from_padded_metal_buffers(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<EncodedJ2k>, crate::Error> {
    submit_lossless_from_padded_metal_buffers(tiles, options, session)?.wait()
}

#[cfg(target_os = "macos")]
pub fn encode_lossless_from_padded_metal_buffers_to_metal(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalEncodedJ2k>, crate::Error> {
    Ok(
        encode_lossless_from_padded_metal_buffers_to_metal_with_report(tiles, options, session)?
            .into_iter()
            .map(|outcome| outcome.encoded)
            .collect(),
    )
}

#[cfg(target_os = "macos")]
pub fn submit_lossless_from_padded_metal_buffers(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    submit_lossless_from_padded_metal_buffers_with_config(
        tiles,
        options,
        session,
        MetalLosslessEncodeConfig::default(),
    )
}

#[cfg(target_os = "macos")]
pub fn submit_lossless_from_padded_metal_buffers_with_config(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    submit_lossless_tiles(
        tiles,
        *options,
        session,
        MetalEncodeInputStaging::AlreadyPaddedContiguous,
        config,
    )
}

#[cfg(target_os = "macos")]
pub fn encode_lossless_from_padded_metal_buffers_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
    encode_lossless_tiles_with_report(
        tiles,
        *options,
        session,
        MetalEncodeInputStaging::AlreadyPaddedContiguous,
    )
}

#[cfg(target_os = "macos")]
pub fn encode_lossless_from_padded_metal_buffers_to_metal_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessBufferEncodeOutcome>, crate::Error> {
    Ok(encode_lossless_from_padded_metal_buffers_to_metal_batch(
        tiles,
        options,
        session,
        MetalLosslessEncodeConfig::default(),
    )?
    .outcomes)
}

#[cfg(target_os = "macos")]
pub fn encode_lossless_from_padded_metal_buffers_to_metal_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<MetalLosslessBufferEncodeBatchOutcome, crate::Error> {
    encode_lossless_tiles_to_metal_buffer_batch(
        tiles,
        *options,
        session,
        MetalEncodeInputStaging::AlreadyPaddedContiguous,
        config,
    )
}

#[cfg(target_os = "macos")]
fn encode_lossless_tiles_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
    if options.backend != EncodeBackendPreference::CpuOnly {
        let batch = try_encode_resident_lossless_tiles_to_metal_buffer_batch(
            tiles,
            options,
            session,
            staging,
            MetalLosslessEncodeConfig::default(),
        )?;
        if let Some(outcomes) = batch {
            return outcomes
                .outcomes
                .into_iter()
                .map(|outcome| {
                    Ok(MetalLosslessEncodeOutcome {
                        encoded: outcome.encoded.to_encoded_j2k()?,
                        input_copy_used: outcome.input_copy_used,
                        resident: outcome.resident,
                        input_copy_duration: outcome.input_copy_duration,
                        encode_duration: outcome.encode_duration,
                        gpu_duration: outcome.gpu_duration,
                        validation_duration: outcome.validation_duration,
                    })
                })
                .collect();
        }
    }

    let mut accelerator = MetalEncodeStageAccelerator::default();
    tiles
        .iter()
        .map(|&tile| {
            encode_lossless_tile_with_report(tile, options, session, staging, &mut accelerator)
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn encode_lossless_owned_tiles_with_report(
    tiles: &[OwnedMetalLosslessEncodeTile],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    config: MetalLosslessEncodeConfig,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
    let borrowed = tiles
        .iter()
        .map(OwnedMetalLosslessEncodeTile::as_tile)
        .collect::<Vec<_>>();
    if options.backend != EncodeBackendPreference::CpuOnly {
        let batch = try_encode_resident_lossless_tiles_to_metal_buffer_batch(
            &borrowed, options, session, staging, config,
        )?;
        if let Some(outcomes) = batch {
            return outcomes
                .outcomes
                .into_iter()
                .map(|outcome| {
                    Ok(MetalLosslessEncodeOutcome {
                        encoded: outcome.encoded.to_encoded_j2k()?,
                        input_copy_used: outcome.input_copy_used,
                        resident: outcome.resident,
                        input_copy_duration: outcome.input_copy_duration,
                        encode_duration: outcome.encode_duration,
                        gpu_duration: outcome.gpu_duration,
                        validation_duration: outcome.validation_duration,
                    })
                })
                .collect();
        }
    }

    let mut accelerator = MetalEncodeStageAccelerator::default();
    borrowed
        .iter()
        .map(|&tile| {
            encode_lossless_tile_with_report(tile, options, session, staging, &mut accelerator)
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn encode_lossless_tiles_to_metal_buffer_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    config: MetalLosslessEncodeConfig,
) -> Result<MetalLosslessBufferEncodeBatchOutcome, crate::Error> {
    if options.backend != EncodeBackendPreference::CpuOnly {
        if let Some(batch) = try_encode_resident_lossless_tiles_to_metal_buffer_batch(
            tiles, options, session, staging, config,
        )? {
            return Ok(batch);
        }
    }

    let mut outcomes = Vec::with_capacity(tiles.len());
    for &tile in tiles {
        outcomes.push(encode_lossless_tile_to_metal_buffer_with_report(
            tile, options, session, staging,
        )?);
    }
    Ok(MetalLosslessBufferEncodeBatchOutcome {
        outcomes,
        stats: MetalLosslessEncodeBatchStats::default(),
    })
}

#[cfg(target_os = "macos")]
fn try_encode_resident_lossless_tiles_to_metal_buffer_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    config: MetalLosslessEncodeConfig,
) -> Result<Option<MetalLosslessBufferEncodeBatchOutcome>, crate::Error> {
    if tiles.is_empty() {
        return Ok(Some(MetalLosslessBufferEncodeBatchOutcome {
            outcomes: Vec::new(),
            stats: resolve_lossless_encode_config(0, 1, config)?,
        }));
    }

    let mut planned = Vec::with_capacity(tiles.len());
    for (index, &tile) in tiles.iter().enumerate() {
        let Some(item) = plan_resident_lossless_buffer_encode(index, tile, options, staging)?
        else {
            return Ok(None);
        };
        planned.push(item);
    }
    let estimated_peak_bytes_per_tile = planned
        .iter()
        .map(PlannedResidentLosslessBufferEncode::estimated_peak_bytes)
        .max()
        .unwrap_or(1);
    let mut stats =
        resolve_lossless_encode_config(tiles.len(), estimated_peak_bytes_per_tile, config)?;
    let encode_started = Instant::now();
    let outcomes = encode_planned_resident_lossless_tiles(
        planned,
        options,
        session,
        stats.effective_inflight_tiles,
        &mut stats,
    )?;
    stats.encode_wall_duration = encode_started.elapsed();
    Ok(Some(MetalLosslessBufferEncodeBatchOutcome {
        outcomes,
        stats,
    }))
}

#[cfg(any(test, target_os = "macos"))]
const GPU_ENCODE_DEFAULT_INFLIGHT_TILES: usize = 64;
#[cfg(any(test, target_os = "macos"))]
const GPU_ENCODE_FALLBACK_HW_MEM_BYTES: usize = 8 * 1024 * 1024 * 1024;
#[cfg(any(test, target_os = "macos"))]
const GPU_ENCODE_MAX_DEFAULT_MEMORY_BUDGET_BYTES: usize = 10 * 1024 * 1024 * 1024;
#[cfg(any(test, target_os = "macos"))]
const GPU_ENCODE_MEMORY_BUDGET_PERCENT: usize = 40;

#[cfg(any(test, target_os = "macos"))]
fn default_gpu_encode_memory_budget_bytes_for_hw_mem(hw_memsize: usize) -> usize {
    hw_memsize
        .saturating_mul(GPU_ENCODE_MEMORY_BUDGET_PERCENT)
        .checked_div(100)
        .unwrap_or(0)
        .clamp(1, GPU_ENCODE_MAX_DEFAULT_MEMORY_BUDGET_BYTES)
}

#[cfg(any(test, target_os = "macos"))]
fn default_gpu_encode_memory_budget_bytes() -> usize {
    let hw_memsize = host_memory_bytes().unwrap_or(GPU_ENCODE_FALLBACK_HW_MEM_BYTES);
    default_gpu_encode_memory_budget_bytes_for_hw_mem(hw_memsize)
}

#[cfg(target_os = "macos")]
fn host_memory_bytes() -> Option<usize> {
    let mut value = 0u64;
    let mut len = core::mem::size_of::<u64>();
    let name = b"hw.memsize\0";
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr().cast(),
            (&raw mut value).cast(),
            &raw mut len,
            core::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && len == core::mem::size_of::<u64>())
        .then(|| usize::try_from(value).ok())
        .flatten()
}

#[cfg(all(test, not(target_os = "macos")))]
fn host_memory_bytes() -> Option<usize> {
    None
}

#[cfg(any(test, target_os = "macos"))]
fn resolve_lossless_encode_config(
    tile_count: usize,
    estimated_peak_bytes_per_tile: usize,
    config: MetalLosslessEncodeConfig,
) -> Result<MetalLosslessEncodeBatchStats, crate::Error> {
    if config.gpu_encode_inflight_tiles == Some(0) {
        return Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode in-flight tile cap must be greater than zero",
        });
    }
    if config.gpu_encode_memory_budget_bytes == Some(0) {
        return Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode memory budget must be greater than zero",
        });
    }

    let effective_memory_budget_bytes = config
        .gpu_encode_memory_budget_bytes
        .unwrap_or_else(default_gpu_encode_memory_budget_bytes)
        .max(1);
    let estimated_peak_bytes_per_tile = estimated_peak_bytes_per_tile.max(1);
    let memory_limited_tiles =
        (effective_memory_budget_bytes / estimated_peak_bytes_per_tile).max(1);
    let configured_or_default = config
        .gpu_encode_inflight_tiles
        .unwrap_or(GPU_ENCODE_DEFAULT_INFLIGHT_TILES);
    let effective_inflight_tiles = configured_or_default
        .min(memory_limited_tiles)
        .min(tile_count.max(1))
        .max(1);

    Ok(MetalLosslessEncodeBatchStats {
        configured_inflight_tiles: config.gpu_encode_inflight_tiles,
        effective_inflight_tiles,
        configured_memory_budget_bytes: config.gpu_encode_memory_budget_bytes,
        effective_memory_budget_bytes,
        estimated_peak_bytes_per_tile,
        max_observed_inflight_tiles: 0,
        encode_wall_duration: Duration::ZERO,
    })
}

#[cfg(test)]
fn resolve_lossless_encode_config_for_test(
    tile_count: usize,
    estimated_peak_bytes_per_tile: usize,
    config: MetalLosslessEncodeConfig,
) -> Result<MetalLosslessEncodeBatchStats, crate::Error> {
    resolve_lossless_encode_config(tile_count, estimated_peak_bytes_per_tile, config)
}

#[cfg(target_os = "macos")]
fn checked_add_bytes(lhs: usize, rhs: usize) -> usize {
    lhs.saturating_add(rhs)
}

#[cfg(target_os = "macos")]
fn checked_mul_bytes(lhs: usize, rhs: usize) -> usize {
    lhs.saturating_mul(rhs)
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct LosslessSubbandPlan {
    num_cbs_x: u32,
    num_cbs_y: u32,
    code_block_start: usize,
    code_block_count: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct LosslessResolutionPlan {
    subbands: Vec<LosslessSubbandPlan>,
}

#[cfg(target_os = "macos")]
struct LosslessDeviceEncodePlan {
    components: u8,
    bit_depth: u8,
    block_coding_mode: J2kBlockCodingMode,
    num_decomposition_levels: u8,
    use_mct: bool,
    guard_bits: u8,
    code_blocks: Vec<compute::J2kLosslessDeviceCodeBlock>,
    resolutions: Vec<LosslessResolutionPlan>,
    progression_order: EncodeProgressionOrder,
    write_tlm: bool,
}

#[cfg(target_os = "macos")]
struct ResidentLosslessBufferEncodeMetadata {
    tile: OwnedMetalLosslessEncodeTile,
    components: u8,
    bit_depth: u8,
    bytes_per_pixel: usize,
    plan: LosslessDeviceEncodePlan,
    packet_descriptors: Vec<J2kPacketizationPacketDescriptor>,
    packetization_resolutions: Vec<compute::J2kResidentPacketizationResolution>,
}

#[cfg(target_os = "macos")]
struct PreparedResidentLosslessBufferEncode {
    metadata: ResidentLosslessBufferEncodeMetadata,
    prepared: compute::J2kPreparedLosslessDeviceCodeBlocks,
}

#[cfg(target_os = "macos")]
struct PlannedResidentLosslessBufferEncode {
    index: usize,
    metadata: ResidentLosslessBufferEncodeMetadata,
    coefficient_count: usize,
    bytes_per_sample: u8,
    estimated_peak_bytes: usize,
    #[cfg(test)]
    failure_injection_index: Option<usize>,
}

#[cfg(target_os = "macos")]
impl PlannedResidentLosslessBufferEncode {
    fn estimated_peak_bytes(&self) -> usize {
        self.estimated_peak_bytes
    }
}

#[cfg(target_os = "macos")]
enum ResidentLosslessTier1 {
    Classic(compute::J2kResidentLosslessTier1CodeBlocks),
    HighThroughput(compute::J2kResidentLosslessHtCodeBlocks),
}

#[cfg(target_os = "macos")]
struct SubmittedResidentLosslessBufferEncode {
    metadata: ResidentLosslessBufferEncodeMetadata,
    pending_codestream: compute::J2kPendingResidentLosslessCodestream,
}

#[cfg(target_os = "macos")]
struct FinishedResidentLosslessBufferEncode {
    metadata: ResidentLosslessBufferEncodeMetadata,
    encoded: MetalEncodedJ2k,
    encode_duration: Duration,
    gpu_duration: Option<Duration>,
}

#[cfg(target_os = "macos")]
fn lossless_device_encode_levels(width: u32, height: u32, options: J2kLosslessEncodeOptions) -> u8 {
    const MIN_LOSSLESS_DWT_DIMENSION: u32 = 64;
    let levels = if options.progression == J2kProgressionOrder::Rpcl {
        let mut levels = 0u8;
        let mut w = width;
        let mut h = height;
        let max_levels = if width.min(height) <= 1 {
            0
        } else {
            width.min(height).ilog2() as u8
        };
        while w.min(h) > MIN_LOSSLESS_DWT_DIMENSION && levels < max_levels {
            w = w.div_ceil(2);
            h = h.div_ceil(2);
            levels = levels.saturating_add(1);
        }
        levels
    } else {
        u8::from(width.min(height) >= MIN_LOSSLESS_DWT_DIMENSION)
    };

    options
        .max_decomposition_levels
        .map_or(levels, |requested| {
            let max_levels = if width.min(height) <= 1 {
                0
            } else {
                width.min(height).ilog2() as u8
            };
            requested.min(max_levels)
        })
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct LosslessSubbandInput {
    component: u32,
    subband_x: u32,
    subband_y: u32,
    width: u32,
    height: u32,
    sub_band_type: J2kSubBandType,
    total_bitplanes: u8,
}

#[cfg(target_os = "macos")]
fn push_lossless_subband_plan(
    resolution: &mut LosslessResolutionPlan,
    code_blocks: &mut Vec<compute::J2kLosslessDeviceCodeBlock>,
    coefficient_offset: &mut u32,
    subband: LosslessSubbandInput,
) -> Result<(), crate::Error> {
    if subband.width == 0 || subband.height == 0 {
        resolution.subbands.push(LosslessSubbandPlan {
            num_cbs_x: 0,
            num_cbs_y: 0,
            code_block_start: code_blocks.len(),
            code_block_count: 0,
        });
        return Ok(());
    }
    let cb_width = 64u32;
    let cb_height = 64u32;
    let num_cbs_x = subband.width.div_ceil(cb_width);
    let num_cbs_y = subband.height.div_ceil(cb_height);
    let code_block_start = code_blocks.len();
    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let block_x = cbx * cb_width;
            let block_y = cby * cb_height;
            let block_width = (block_x + cb_width).min(subband.width) - block_x;
            let block_height = (block_y + cb_height).min(subband.height) - block_y;
            let coeff_count =
                block_width
                    .checked_mul(block_height)
                    .ok_or_else(|| crate::Error::MetalKernel {
                        message: "J2K Metal resident encode code-block size overflow".to_string(),
                    })?;
            code_blocks.push(compute::J2kLosslessDeviceCodeBlock {
                coefficient_offset: *coefficient_offset,
                component: subband.component,
                subband_x: subband.subband_x,
                subband_y: subband.subband_y,
                block_x,
                block_y,
                width: block_width,
                height: block_height,
                sub_band_type: subband.sub_band_type,
                total_bitplanes: subband.total_bitplanes,
            });
            *coefficient_offset = coefficient_offset.checked_add(coeff_count).ok_or_else(|| {
                crate::Error::MetalKernel {
                    message: "J2K Metal resident encode coefficient offset overflow".to_string(),
                }
            })?;
        }
    }
    resolution.subbands.push(LosslessSubbandPlan {
        num_cbs_x,
        num_cbs_y,
        code_block_start,
        code_block_count: code_blocks.len() - code_block_start,
    });
    Ok(())
}

#[cfg(target_os = "macos")]
fn lossless_device_encode_plan(
    width: u32,
    height: u32,
    components: u8,
    bit_depth: u8,
    options: J2kLosslessEncodeOptions,
) -> Result<Option<LosslessDeviceEncodePlan>, crate::Error> {
    if !matches!(
        options.block_coding_mode,
        J2kBlockCodingMode::Classic | J2kBlockCodingMode::HighThroughput
    ) {
        return Ok(None);
    }
    let num_decomposition_levels = lossless_device_encode_levels(width, height, options);
    if num_decomposition_levels > 1 {
        return Ok(None);
    }
    let progression_order = match options.progression {
        J2kProgressionOrder::Lrcp => EncodeProgressionOrder::Lrcp,
        J2kProgressionOrder::Rpcl => EncodeProgressionOrder::Rpcl,
    };
    let use_mct = components >= 3;
    let guard_bits: u8 = if use_mct { 2 } else { 1 };
    let mut code_blocks = Vec::new();
    let mut coefficient_offset = 0u32;
    let mut component_resolutions = Vec::<Vec<LosslessResolutionPlan>>::new();
    for component in 0..components {
        let mut component_packets = Vec::new();
        let mut base_packet = LosslessResolutionPlan {
            subbands: Vec::new(),
        };
        if num_decomposition_levels == 0 {
            push_lossless_subband_plan(
                &mut base_packet,
                &mut code_blocks,
                &mut coefficient_offset,
                LosslessSubbandInput {
                    component: u32::from(component),
                    subband_x: 0,
                    subband_y: 0,
                    width,
                    height,
                    sub_band_type: J2kSubBandType::LowLow,
                    total_bitplanes: guard_bits.saturating_add(bit_depth).saturating_sub(1),
                },
            )?;
            component_packets.push(base_packet);
        } else {
            let low_width = width.div_ceil(2);
            let low_height = height.div_ceil(2);
            let high_width = width / 2;
            let high_height = height / 2;
            push_lossless_subband_plan(
                &mut base_packet,
                &mut code_blocks,
                &mut coefficient_offset,
                LosslessSubbandInput {
                    component: u32::from(component),
                    subband_x: 0,
                    subband_y: 0,
                    width: low_width,
                    height: low_height,
                    sub_band_type: J2kSubBandType::LowLow,
                    total_bitplanes: guard_bits.saturating_add(bit_depth).saturating_sub(1),
                },
            )?;
            component_packets.push(base_packet);

            let mut detail_packet = LosslessResolutionPlan {
                subbands: Vec::new(),
            };
            push_lossless_subband_plan(
                &mut detail_packet,
                &mut code_blocks,
                &mut coefficient_offset,
                LosslessSubbandInput {
                    component: u32::from(component),
                    subband_x: low_width,
                    subband_y: 0,
                    width: high_width,
                    height: low_height,
                    sub_band_type: J2kSubBandType::HighLow,
                    total_bitplanes: guard_bits.saturating_add(bit_depth),
                },
            )?;
            push_lossless_subband_plan(
                &mut detail_packet,
                &mut code_blocks,
                &mut coefficient_offset,
                LosslessSubbandInput {
                    component: u32::from(component),
                    subband_x: 0,
                    subband_y: low_height,
                    width: low_width,
                    height: high_height,
                    sub_band_type: J2kSubBandType::LowHigh,
                    total_bitplanes: guard_bits.saturating_add(bit_depth),
                },
            )?;
            push_lossless_subband_plan(
                &mut detail_packet,
                &mut code_blocks,
                &mut coefficient_offset,
                LosslessSubbandInput {
                    component: u32::from(component),
                    subband_x: low_width,
                    subband_y: low_height,
                    width: high_width,
                    height: high_height,
                    sub_band_type: J2kSubBandType::HighHigh,
                    total_bitplanes: guard_bits.saturating_add(bit_depth).saturating_add(1),
                },
            )?;
            component_packets.push(detail_packet);
        }
        component_resolutions.push(component_packets);
    }

    let resolution_count = component_resolutions.first().map_or(0usize, Vec::len);
    let mut resolutions =
        Vec::with_capacity(resolution_count.saturating_mul(usize::from(components)));
    for resolution in 0..resolution_count {
        for component in &component_resolutions {
            resolutions.push(component[resolution].clone());
        }
    }

    Ok(Some(LosslessDeviceEncodePlan {
        components,
        bit_depth,
        block_coding_mode: options.block_coding_mode,
        num_decomposition_levels,
        use_mct,
        guard_bits,
        code_blocks,
        resolutions,
        progression_order,
        write_tlm: options.progression == J2kProgressionOrder::Rpcl,
    }))
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
enum MetalEncodeInputStaging {
    CopyAndPad,
    AlreadyPaddedContiguous,
}

#[cfg(target_os = "macos")]
fn submit_lossless_tiles(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    config: MetalLosslessEncodeConfig,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous)
        && options.backend != EncodeBackendPreference::CpuOnly
    {
        let mut ready = Vec::with_capacity(tiles.len());
        let mut all_ready = true;
        for &tile in tiles {
            validate_metal_encode_tile(tile)?;
            lossless_sample_shape(tile.format)?;
            validate_padded_contiguous_metal_encode_tile(tile, tile.format.bytes_per_pixel())?;
            if let Some(outcome) = try_encode_lossless_tile_device_resident_with_report(
                tile, options, session, staging,
            )? {
                ready.push(outcome.encoded);
            } else {
                all_ready = false;
                break;
            }
        }
        if all_ready {
            return Ok(SubmittedJ2kLosslessMetalEncodeBatch {
                state: SubmittedJ2kLosslessMetalEncodeBatchState::Ready(ready),
            });
        }
        if options.backend == EncodeBackendPreference::RequireDevice {
            return Err(crate::Error::UnsupportedMetalRequest {
                reason: "J2K Metal resident encode requires classic padded contiguous Gray/RGB lossless input with at most one DWT level",
            });
        }
    }

    let mut owned = Vec::with_capacity(tiles.len());
    for &tile in tiles {
        validate_metal_encode_tile(tile)?;
        if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
            lossless_sample_shape(tile.format)?;
            validate_padded_contiguous_metal_encode_tile(tile, tile.format.bytes_per_pixel())?;
        }
        owned.push(OwnedMetalLosslessEncodeTile::from_tile(tile));
    }
    Ok(SubmittedJ2kLosslessMetalEncodeBatch {
        state: SubmittedJ2kLosslessMetalEncodeBatchState::Deferred {
            tiles: owned,
            options,
            session: session.clone(),
            staging,
            config,
        },
    })
}

#[cfg(target_os = "macos")]
fn packet_descriptors_for_lossless_device_order(
    packet_count: usize,
    num_components: u8,
) -> Result<Vec<J2kPacketizationPacketDescriptor>, crate::Error> {
    let component_count = usize::from(num_components).max(1);
    (0..packet_count)
        .map(|packet_index| {
            Ok(J2kPacketizationPacketDescriptor {
                packet_index: u32::try_from(packet_index).map_err(|_| {
                    crate::Error::MetalKernel {
                        message: "J2K Metal resident encode packet index exceeds u32".to_string(),
                    }
                })?,
                state_index: u32::try_from(packet_index).map_err(|_| {
                    crate::Error::MetalKernel {
                        message: "J2K Metal resident encode packet state index exceeds u32"
                            .to_string(),
                    }
                })?,
                layer: 0,
                resolution: u32::try_from(packet_index / component_count).map_err(|_| {
                    crate::Error::MetalKernel {
                        message: "J2K Metal resident encode packet resolution exceeds u32"
                            .to_string(),
                    }
                })?,
                component: u8::try_from(packet_index % component_count).map_err(|_| {
                    crate::Error::MetalKernel {
                        message: "J2K Metal resident encode packet component exceeds u8"
                            .to_string(),
                    }
                })?,
                precinct: 0,
            })
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn resident_packetization_resolutions_from_lossless_device_plan(
    plan: &LosslessDeviceEncodePlan,
) -> Result<Vec<compute::J2kResidentPacketizationResolution>, crate::Error> {
    plan.resolutions
        .iter()
        .map(|resolution| {
            let subbands = resolution
                .subbands
                .iter()
                .map(|subband| {
                    let code_block_end = subband
                        .code_block_start
                        .checked_add(subband.code_block_count)
                        .ok_or_else(|| crate::Error::MetalKernel {
                            message: "J2K Metal resident encode code-block range overflow"
                                .to_string(),
                        })?;
                    if code_block_end > plan.code_blocks.len() {
                        return Err(crate::Error::MetalKernel {
                            message: "J2K Metal resident encode code-block range out of bounds"
                                .to_string(),
                        });
                    }
                    Ok(compute::J2kResidentPacketizationSubband {
                        code_block_start: u32::try_from(subband.code_block_start).map_err(
                            |_| crate::Error::MetalKernel {
                                message: "J2K Metal resident encode code-block offset exceeds u32"
                                    .to_string(),
                            },
                        )?,
                        code_block_count: u32::try_from(subband.code_block_count).map_err(
                            |_| crate::Error::MetalKernel {
                                message: "J2K Metal resident encode code-block count exceeds u32"
                                    .to_string(),
                            },
                        )?,
                        num_cbs_x: subband.num_cbs_x,
                        num_cbs_y: subband.num_cbs_y,
                    })
                })
                .collect::<Result<Vec<_>, crate::Error>>()?;
            Ok(compute::J2kResidentPacketizationResolution { subbands })
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn lossless_device_coefficient_count(
    code_blocks: &[compute::J2kLosslessDeviceCodeBlock],
) -> Result<usize, crate::Error> {
    let mut count = 0usize;
    for block in code_blocks {
        let offset =
            usize::try_from(block.coefficient_offset).map_err(|_| crate::Error::MetalKernel {
                message: "J2K Metal resident encode coefficient offset exceeds usize".to_string(),
            })?;
        let block_count = (block.width as usize)
            .checked_mul(block.height as usize)
            .ok_or_else(|| crate::Error::MetalKernel {
                message: "J2K Metal resident encode coefficient count overflow".to_string(),
            })?;
        count = count.max(offset.checked_add(block_count).ok_or_else(|| {
            crate::Error::MetalKernel {
                message: "J2K Metal resident encode coefficient count overflow".to_string(),
            }
        })?);
    }
    Ok(count)
}

#[cfg(target_os = "macos")]
fn plan_resident_lossless_buffer_encode(
    index: usize,
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    staging: MetalEncodeInputStaging,
) -> Result<Option<PlannedResidentLosslessBufferEncode>, crate::Error> {
    validate_metal_encode_tile(tile)?;
    if options.backend == EncodeBackendPreference::CpuOnly {
        return Ok(None);
    }
    let (components, bit_depth) = lossless_sample_shape(tile.format)?;
    let bytes_per_pixel = tile.format.bytes_per_pixel();
    let bytes_per_sample =
        u8::try_from(tile.format.bytes_per_sample()).map_err(|_| crate::Error::MetalKernel {
            message: "J2K Metal resident encode bytes per sample exceeds u8".to_string(),
        })?;
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
        validate_padded_contiguous_metal_encode_tile(tile, bytes_per_pixel)?;
    }
    let Some(plan) = lossless_device_encode_plan(
        tile.output_width,
        tile.output_height,
        components,
        bit_depth,
        options,
    )?
    else {
        return Ok(None);
    };
    let coefficient_count = lossless_device_coefficient_count(&plan.code_blocks)?;
    let packetization_resolutions =
        resident_packetization_resolutions_from_lossless_device_plan(&plan)?;
    let packet_descriptors =
        packet_descriptors_for_lossless_device_order(plan.resolutions.len(), plan.components)?;
    let metadata = ResidentLosslessBufferEncodeMetadata {
        tile: OwnedMetalLosslessEncodeTile::from_tile(tile),
        components,
        bit_depth,
        bytes_per_pixel,
        plan,
        packet_descriptors,
        packetization_resolutions,
    };
    let estimated_peak_bytes =
        estimate_resident_lossless_encode_peak_bytes(&metadata, coefficient_count, staging);
    Ok(Some(PlannedResidentLosslessBufferEncode {
        index,
        metadata,
        coefficient_count,
        bytes_per_sample,
        estimated_peak_bytes,
        #[cfg(test)]
        failure_injection_index: test_resident_encode_failure_index(),
    }))
}

#[cfg(target_os = "macos")]
fn estimate_resident_lossless_encode_peak_bytes(
    metadata: &ResidentLosslessBufferEncodeMetadata,
    coefficient_count: usize,
    staging: MetalEncodeInputStaging,
) -> usize {
    let pixels = checked_mul_bytes(
        metadata.tile.output_width as usize,
        metadata.tile.output_height as usize,
    )
    .max(1);
    let plane_bytes = checked_mul_bytes(pixels, core::mem::size_of::<f32>());
    let code_block_count = metadata.plan.code_blocks.len().max(1);
    let packet_count = metadata
        .packet_descriptors
        .len()
        .max(metadata.plan.resolutions.len())
        .max(1);
    let input_bytes = checked_mul_bytes(
        checked_mul_bytes(metadata.tile.width as usize, metadata.tile.height as usize),
        metadata.bytes_per_pixel,
    );
    let staged_input_bytes = if matches!(staging, MetalEncodeInputStaging::CopyAndPad) {
        checked_mul_bytes(pixels, metadata.bytes_per_pixel)
    } else {
        0
    };
    let coefficient_bytes =
        checked_mul_bytes(coefficient_count.max(1), core::mem::size_of::<i32>());
    let plane_buffers = checked_mul_bytes(3, plane_bytes);
    let scratch_buffers = checked_mul_bytes(usize::from(metadata.components), plane_bytes);
    let code_block_tables = checked_mul_bytes(code_block_count, 256);
    let tier1_output = estimated_tier1_output_bytes(&metadata.plan);
    let packet_header = checked_add_bytes(checked_mul_bytes(code_block_count, 256), 4096);
    let packet_output = checked_add_bytes(
        checked_add_bytes(tier1_output, checked_mul_bytes(packet_header, packet_count)),
        1024,
    );
    let codestream_capacity = checked_add_bytes(
        packet_output,
        checked_add_bytes(4096, checked_mul_bytes(pixels, metadata.bytes_per_pixel)),
    );
    let validation_bytes = checked_mul_bytes(pixels, metadata.bytes_per_pixel).saturating_mul(
        usize::from(metadata.plan.write_tlm || metadata.plan.use_mct || metadata.components > 0),
    );

    [
        input_bytes / 4,
        staged_input_bytes,
        plane_buffers,
        scratch_buffers,
        coefficient_bytes,
        code_block_tables,
        tier1_output,
        packet_output,
        codestream_capacity,
        validation_bytes,
        4 * 1024 * 1024,
    ]
    .into_iter()
    .fold(0usize, checked_add_bytes)
}

#[cfg(target_os = "macos")]
fn estimated_tier1_output_bytes(plan: &LosslessDeviceEncodePlan) -> usize {
    const HT_ENCODE_OUTPUT_CAPACITY_PER_BLOCK: usize =
        (16_384usize * 16).div_ceil(15) + 192 + (3072 - 192);
    plan.code_blocks
        .iter()
        .map(|block| match plan.block_coding_mode {
            J2kBlockCodingMode::HighThroughput => HT_ENCODE_OUTPUT_CAPACITY_PER_BLOCK,
            J2kBlockCodingMode::Classic => {
                let samples = checked_mul_bytes(block.width as usize, block.height as usize);
                checked_add_bytes(
                    checked_mul_bytes(
                        checked_mul_bytes(samples, usize::from(block.total_bitplanes).max(1)),
                        8,
                    ),
                    4097,
                )
                .max(4097)
            }
        })
        .fold(0usize, checked_add_bytes)
        .max(1)
}

#[cfg(target_os = "macos")]
fn encode_prepared_resident_lossless_tier1(
    prepared: PreparedResidentLosslessBufferEncode,
    session: &crate::MetalBackendSession,
) -> Result<(ResidentLosslessBufferEncodeMetadata, ResidentLosslessTier1), crate::Error> {
    let tier1 = match prepared.metadata.plan.block_coding_mode {
        J2kBlockCodingMode::Classic => ResidentLosslessTier1::Classic(
            compute::encode_classic_tier1_prepared_device_code_blocks_resident(
                session,
                prepared.prepared,
            )?,
        ),
        J2kBlockCodingMode::HighThroughput => ResidentLosslessTier1::HighThroughput(
            compute::encode_ht_prepared_device_code_blocks_resident(session, prepared.prepared)?,
        ),
    };

    Ok((prepared.metadata, tier1))
}

#[cfg(target_os = "macos")]
fn resident_packetization_job_for_metadata(
    metadata: &ResidentLosslessBufferEncodeMetadata,
) -> Result<compute::J2kResidentPacketizationEncodeJob<'_>, crate::Error> {
    Ok(compute::J2kResidentPacketizationEncodeJob {
        resolution_count: u32::try_from(metadata.plan.resolutions.len()).map_err(|_| {
            crate::Error::MetalKernel {
                message: "J2K Metal resident encode resolution count exceeds u32".to_string(),
            }
        })?,
        num_layers: 1,
        num_components: metadata.plan.components,
        code_block_count: u32::try_from(metadata.plan.code_blocks.len()).map_err(|_| {
            crate::Error::MetalKernel {
                message: "J2K Metal resident encode code-block count exceeds u32".to_string(),
            }
        })?,
        packet_descriptors: &metadata.packet_descriptors,
        resolutions: &metadata.packetization_resolutions,
    })
}

#[cfg(target_os = "macos")]
fn resident_codestream_assembly_job_for_metadata(
    metadata: &ResidentLosslessBufferEncodeMetadata,
) -> compute::J2kLosslessCodestreamAssemblyJob {
    compute::J2kLosslessCodestreamAssemblyJob {
        width: metadata.tile.output_width,
        height: metadata.tile.output_height,
        num_components: metadata.plan.components,
        bit_depth: metadata.plan.bit_depth,
        signed: false,
        num_decomposition_levels: metadata.plan.num_decomposition_levels,
        use_mct: metadata.plan.use_mct,
        guard_bits: metadata.plan.guard_bits,
        progression_order: metadata.plan.progression_order,
        write_tlm: metadata.plan.write_tlm,
        block_coding_mode: match metadata.plan.block_coding_mode {
            J2kBlockCodingMode::Classic => compute::J2kLosslessCodestreamBlockCodingMode::Classic,
            J2kBlockCodingMode::HighThroughput => {
                compute::J2kLosslessCodestreamBlockCodingMode::HighThroughput
            }
        },
    }
}

#[cfg(target_os = "macos")]
fn submit_resident_lossless_buffer_encode(
    metadata: ResidentLosslessBufferEncodeMetadata,
    tier1: &ResidentLosslessTier1,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedResidentLosslessBufferEncode, crate::Error> {
    let packetization_job = resident_packetization_job_for_metadata(&metadata)?;
    let assembly_job = resident_codestream_assembly_job_for_metadata(&metadata);
    let pending_codestream = match tier1 {
        ResidentLosslessTier1::Classic(tier1) => {
            compute::submit_lossless_codestream_buffer_from_resident_classic_tier1(
                session,
                tier1,
                packetization_job,
                assembly_job,
            )?
        }
        ResidentLosslessTier1::HighThroughput(tier1) => {
            compute::submit_lossless_codestream_buffer_from_resident_ht_tier1(
                session,
                tier1,
                packetization_job,
                assembly_job,
            )?
        }
    };
    Ok(SubmittedResidentLosslessBufferEncode {
        metadata,
        pending_codestream,
    })
}

#[cfg(target_os = "macos")]
fn wait_submitted_resident_lossless_buffer_encode(
    submitted: SubmittedResidentLosslessBufferEncode,
) -> Result<FinishedResidentLosslessBufferEncode, crate::Error> {
    let encode_started = Instant::now();
    let codestream = compute::wait_resident_lossless_codestream(submitted.pending_codestream)?;
    let encode_duration = encode_started.elapsed();
    Ok(finished_resident_lossless_buffer_encode(
        submitted.metadata,
        codestream,
        encode_duration,
    ))
}

#[cfg(target_os = "macos")]
fn finished_resident_lossless_buffer_encode(
    metadata: ResidentLosslessBufferEncodeMetadata,
    codestream: compute::J2kResidentLosslessCodestream,
    encode_duration: Duration,
) -> FinishedResidentLosslessBufferEncode {
    let encoded = MetalEncodedJ2k {
        codestream_buffer: codestream.buffer,
        byte_offset: codestream.byte_offset,
        byte_len: codestream.byte_len,
        capacity: codestream.capacity,
        width: metadata.tile.output_width,
        height: metadata.tile.output_height,
        components: metadata.components,
        bit_depth: metadata.bit_depth,
        signed: false,
    };

    FinishedResidentLosslessBufferEncode {
        metadata,
        encoded,
        encode_duration,
        gpu_duration: codestream.gpu_duration,
    }
}

#[cfg(target_os = "macos")]
fn validate_finished_resident_lossless_buffer_encode(
    finished: FinishedResidentLosslessBufferEncode,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessBufferEncodeOutcome, crate::Error> {
    let FinishedResidentLosslessBufferEncode {
        metadata,
        encoded,
        encode_duration,
        gpu_duration,
    } = finished;

    let validation_duration = if options.validation == J2kEncodeValidation::CpuRoundTrip {
        let validation_started = Instant::now();
        let tile = metadata.tile.as_tile();
        if tile.width == tile.output_width
            && tile.height == tile.output_height
            && tile.pitch_bytes == tile.output_width as usize * metadata.bytes_per_pixel
        {
            validate_lossless_roundtrip_on_metal_tile_with_session(
                tile,
                encoded.codestream_bytes()?,
                session,
            )?;
        } else {
            validate_lossless_roundtrip_on_metal_region_with_session(
                tile,
                tile.output_width,
                tile.output_height,
                metadata.bytes_per_pixel,
                encoded.codestream_bytes()?,
                session,
            )?;
        }
        validation_started.elapsed()
    } else {
        Duration::ZERO
    };

    Ok(MetalLosslessBufferEncodeOutcome {
        encoded,
        input_copy_used: false,
        resident: MetalLosslessEncodeResidency {
            coefficient_prep_used: true,
            packetization_used: true,
            codestream_assembly_used: true,
        },
        input_copy_duration: Duration::ZERO,
        encode_duration,
        gpu_duration,
        validation_duration,
    })
}

#[cfg(target_os = "macos")]
struct InflightLimitedOrderedItems<T> {
    items: Vec<T>,
    max_observed_inflight_items: usize,
}

#[cfg(target_os = "macos")]
fn collect_inflight_limited_ordered<T, O, F>(
    items: Vec<T>,
    inflight_items: usize,
    f: F,
) -> Result<InflightLimitedOrderedItems<O>, crate::Error>
where
    T: Send,
    O: Send,
    F: Fn(usize, T) -> Result<O, crate::Error> + Sync,
{
    if items.is_empty() {
        return Ok(InflightLimitedOrderedItems {
            items: Vec::new(),
            max_observed_inflight_items: 0,
        });
    }

    let active = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(AtomicUsize::new(0));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(inflight_items.max(1))
        .build()
        .map_err(|err| crate::Error::MetalKernel {
            message: format!("J2K Metal encode worker pool initialization failed: {err}"),
        })?;

    let active_for_tasks = Arc::clone(&active);
    let observed_for_tasks = Arc::clone(&observed);
    let results = pool.install(|| {
        items
            .into_par_iter()
            .enumerate()
            .map(|(index, item)| {
                let _guard = ActiveTileGuard::new(&active_for_tasks, &observed_for_tasks);
                f(index, item)
            })
            .collect::<Vec<_>>()
    });

    let max_observed_inflight_items = observed.load(Ordering::Relaxed);
    let mut ordered = Vec::with_capacity(results.len());
    let mut first_error = None;
    for result in results {
        match result {
            Ok(item) if first_error.is_none() => ordered.push(item),
            Ok(_) => {}
            Err(err) => {
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
        }
    }

    if let Some(err) = first_error {
        return Err(err);
    }

    Ok(InflightLimitedOrderedItems {
        items: ordered,
        max_observed_inflight_items,
    })
}

#[cfg(target_os = "macos")]
fn encode_planned_resident_lossless_tiles(
    planned: Vec<PlannedResidentLosslessBufferEncode>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    inflight_tiles: usize,
    stats: &mut MetalLosslessEncodeBatchStats,
) -> Result<Vec<MetalLosslessBufferEncodeOutcome>, crate::Error> {
    if planned.is_empty() {
        return Ok(Vec::new());
    }
    if planned.iter().all(|planned| {
        planned.metadata.plan.block_coding_mode == J2kBlockCodingMode::HighThroughput
    }) {
        return encode_planned_resident_ht_lossless_tiles_batch(
            planned,
            options,
            session,
            inflight_tiles,
            stats,
        );
    }
    if planned
        .iter()
        .all(|planned| planned.metadata.plan.block_coding_mode == J2kBlockCodingMode::Classic)
    {
        return encode_planned_resident_classic_lossless_tiles_batch(
            planned,
            options,
            session,
            inflight_tiles,
            stats,
        );
    }

    let encoded = collect_inflight_limited_ordered(planned, inflight_tiles, |_, planned| {
        let index = planned.index;
        encode_planned_resident_lossless_tile(planned, options, session).map_err(|err| {
            crate::Error::MetalKernel {
                message: format!("J2K Metal resident encode failed at tile {index}: {err}"),
            }
        })
    })?;
    stats.max_observed_inflight_tiles = stats
        .max_observed_inflight_tiles
        .max(encoded.max_observed_inflight_items);
    Ok(encoded.items)
}

#[cfg(target_os = "macos")]
struct PreparedResidentLosslessBatchItem {
    prepared: PreparedResidentLosslessBufferEncode,
    prepare_duration: Duration,
}

#[cfg(target_os = "macos")]
fn prepare_planned_resident_ht_lossless_tiles_batch(
    planned: Vec<PlannedResidentLosslessBufferEncode>,
    session: &crate::MetalBackendSession,
) -> Result<Vec<PreparedResidentLosslessBatchItem>, crate::Error> {
    struct HtBatchPlanInfo {
        index: usize,
        coefficient_count: usize,
        bytes_per_sample: u8,
        code_blocks: Vec<compute::J2kLosslessDeviceCodeBlock>,
    }

    let started = Instant::now();
    let mut metadatas = Vec::with_capacity(planned.len());
    let mut plan_infos = Vec::with_capacity(planned.len());
    for planned in planned {
        #[cfg(test)]
        if planned.failure_injection_index == Some(planned.index) {
            return Err(crate::Error::MetalKernel {
                message: format!(
                    "injected J2K Metal resident encode failure at tile {}",
                    planned.index
                ),
            });
        }

        plan_infos.push(HtBatchPlanInfo {
            index: planned.index,
            coefficient_count: planned.coefficient_count,
            bytes_per_sample: planned.bytes_per_sample,
            code_blocks: planned.metadata.plan.code_blocks.clone(),
        });
        metadatas.push(planned.metadata);
    }

    let mut batch_items = Vec::with_capacity(metadatas.len());
    for (metadata, plan_info) in metadatas.iter().zip(plan_infos) {
        let tile = metadata.tile.as_tile();
        batch_items.push(compute::J2kLosslessDeviceBatchPrepareItem {
            tile_index: plan_info.index,
            job: compute::J2kLosslessDevicePrepareJob {
                input: tile.buffer,
                input_byte_offset: tile.byte_offset,
                input_width: tile.width,
                input_height: tile.height,
                input_pitch_bytes: tile.pitch_bytes,
                output_width: tile.output_width,
                output_height: tile.output_height,
                components: metadata.components,
                bytes_per_sample: plan_info.bytes_per_sample,
                bit_depth: metadata.bit_depth,
                num_decomposition_levels: metadata.plan.num_decomposition_levels,
                coefficient_count: plan_info.coefficient_count,
            },
            code_blocks: plan_info.code_blocks,
        });
    }

    let prepared = compute::prepare_lossless_device_code_blocks_batch(session, batch_items)?;
    let prepare_duration = duration_share(started.elapsed(), prepared.len());
    Ok(metadatas
        .into_iter()
        .zip(prepared)
        .map(|(metadata, prepared)| PreparedResidentLosslessBatchItem {
            prepared: PreparedResidentLosslessBufferEncode { metadata, prepared },
            prepare_duration,
        })
        .collect())
}

#[cfg(target_os = "macos")]
fn encode_planned_resident_ht_lossless_tiles_batch(
    planned: Vec<PlannedResidentLosslessBufferEncode>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    inflight_tiles: usize,
    stats: &mut MetalLosslessEncodeBatchStats,
) -> Result<Vec<MetalLosslessBufferEncodeOutcome>, crate::Error> {
    let planned_len = planned.len();
    let prepared =
        prepare_planned_resident_ht_lossless_tiles_batch(planned, session).map_err(|err| {
            crate::Error::MetalKernel {
                message: format!("J2K Metal resident HT batch encode failed: {err}"),
            }
        })?;
    stats.max_observed_inflight_tiles = stats.max_observed_inflight_tiles.max(
        planned_len
            .min(inflight_tiles)
            .max(usize::from(planned_len > 0)),
    );

    let mut metadatas = Vec::with_capacity(prepared.len());
    let mut prepare_durations = Vec::with_capacity(prepared.len());
    let mut batch_items = Vec::with_capacity(prepared.len());
    for item in prepared {
        let PreparedResidentLosslessBatchItem {
            prepared,
            prepare_duration,
        } = item;
        let metadata = prepared.metadata;
        let codestream = resident_codestream_assembly_job_for_metadata(&metadata);
        batch_items.push(compute::J2kResidentHtBatchEncodeItem {
            prepared: prepared.prepared,
            resolution_count: u32::try_from(metadata.plan.resolutions.len()).map_err(|_| {
                crate::Error::MetalKernel {
                    message: "J2K Metal resident encode resolution count exceeds u32".to_string(),
                }
            })?,
            num_layers: 1,
            num_components: metadata.plan.components,
            code_block_count: u32::try_from(metadata.plan.code_blocks.len()).map_err(|_| {
                crate::Error::MetalKernel {
                    message: "J2K Metal resident encode code-block count exceeds u32".to_string(),
                }
            })?,
            packet_descriptors: metadata.packet_descriptors.clone(),
            resolutions: metadata.packetization_resolutions.clone(),
            codestream,
        });
        prepare_durations.push(prepare_duration);
        metadatas.push(metadata);
    }

    let batch_started = Instant::now();
    let pending =
        compute::submit_lossless_codestream_buffers_from_prepared_ht_batch(session, batch_items)?;
    let codestreams = compute::wait_resident_lossless_codestream_batch(pending)?;
    let batch_duration = duration_share(batch_started.elapsed(), codestreams.len());
    metadatas
        .into_iter()
        .zip(prepare_durations)
        .zip(codestreams)
        .map(|((metadata, prepare_duration), codestream)| {
            let finished = finished_resident_lossless_buffer_encode(
                metadata,
                codestream,
                prepare_duration.saturating_add(batch_duration),
            );
            validate_finished_resident_lossless_buffer_encode(finished, options, session)
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn encode_planned_resident_classic_lossless_tiles_batch(
    planned: Vec<PlannedResidentLosslessBufferEncode>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    inflight_tiles: usize,
    stats: &mut MetalLosslessEncodeBatchStats,
) -> Result<Vec<MetalLosslessBufferEncodeOutcome>, crate::Error> {
    let prepared = collect_inflight_limited_ordered(planned, inflight_tiles, |_, planned| {
        let index = planned.index;
        let started = Instant::now();
        prepare_planned_resident_lossless_tile(planned, session)
            .map(|prepared| PreparedResidentLosslessBatchItem {
                prepared,
                prepare_duration: started.elapsed(),
            })
            .map_err(|err| crate::Error::MetalKernel {
                message: format!("J2K Metal resident encode failed at tile {index}: {err}"),
            })
    })?;
    stats.max_observed_inflight_tiles = stats
        .max_observed_inflight_tiles
        .max(prepared.max_observed_inflight_items);

    let mut metadatas = Vec::with_capacity(prepared.items.len());
    let mut prepare_durations = Vec::with_capacity(prepared.items.len());
    let mut batch_items = Vec::with_capacity(prepared.items.len());
    for item in prepared.items {
        let PreparedResidentLosslessBatchItem {
            prepared,
            prepare_duration,
        } = item;
        let metadata = prepared.metadata;
        let codestream = resident_codestream_assembly_job_for_metadata(&metadata);
        batch_items.push(compute::J2kResidentClassicBatchEncodeItem {
            prepared: prepared.prepared,
            resolution_count: u32::try_from(metadata.plan.resolutions.len()).map_err(|_| {
                crate::Error::MetalKernel {
                    message: "J2K Metal resident encode resolution count exceeds u32".to_string(),
                }
            })?,
            num_layers: 1,
            num_components: metadata.plan.components,
            code_block_count: u32::try_from(metadata.plan.code_blocks.len()).map_err(|_| {
                crate::Error::MetalKernel {
                    message: "J2K Metal resident encode code-block count exceeds u32".to_string(),
                }
            })?,
            packet_descriptors: metadata.packet_descriptors.clone(),
            resolutions: metadata.packetization_resolutions.clone(),
            codestream,
        });
        prepare_durations.push(prepare_duration);
        metadatas.push(metadata);
    }

    let batch_limit = inflight_tiles.max(1);
    let mut outcomes = Vec::with_capacity(metadatas.len());
    while !batch_items.is_empty() {
        let take = batch_items.len().min(batch_limit);
        let chunk_items = batch_items.drain(..take).collect();
        let chunk_metadatas = metadatas.drain(..take).collect::<Vec<_>>();
        let chunk_prepare_durations = prepare_durations.drain(..take).collect::<Vec<_>>();
        let batch_started = Instant::now();
        let pending = compute::submit_lossless_codestream_buffers_from_prepared_classic_batch(
            session,
            chunk_items,
        )?;
        let codestreams = compute::wait_resident_lossless_codestream_batch(pending)?;
        let batch_duration = duration_share(batch_started.elapsed(), codestreams.len());
        for ((metadata, prepare_duration), codestream) in chunk_metadatas
            .into_iter()
            .zip(chunk_prepare_durations)
            .zip(codestreams)
        {
            let finished = finished_resident_lossless_buffer_encode(
                metadata,
                codestream,
                prepare_duration.saturating_add(batch_duration),
            );
            outcomes.push(validate_finished_resident_lossless_buffer_encode(
                finished, options, session,
            )?);
        }
    }
    Ok(outcomes)
}

#[cfg(target_os = "macos")]
fn prepare_planned_resident_lossless_tile(
    planned: PlannedResidentLosslessBufferEncode,
    session: &crate::MetalBackendSession,
) -> Result<PreparedResidentLosslessBufferEncode, crate::Error> {
    #[cfg(test)]
    if planned.failure_injection_index == Some(planned.index) {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "injected J2K Metal resident encode failure at tile {}",
                planned.index
            ),
        });
    }

    let tile = planned.metadata.tile.as_tile();
    let prepared = compute::prepare_lossless_device_code_blocks(
        session,
        compute::J2kLosslessDevicePrepareJob {
            input: tile.buffer,
            input_byte_offset: tile.byte_offset,
            input_width: tile.width,
            input_height: tile.height,
            input_pitch_bytes: tile.pitch_bytes,
            output_width: tile.output_width,
            output_height: tile.output_height,
            components: planned.metadata.components,
            bytes_per_sample: planned.bytes_per_sample,
            bit_depth: planned.metadata.bit_depth,
            num_decomposition_levels: planned.metadata.plan.num_decomposition_levels,
            coefficient_count: planned.coefficient_count,
        },
        planned.metadata.plan.code_blocks.clone(),
    )?;
    Ok(PreparedResidentLosslessBufferEncode {
        metadata: planned.metadata,
        prepared,
    })
}

#[cfg(target_os = "macos")]
fn duration_share(duration: Duration, count: usize) -> Duration {
    if count == 0 {
        return Duration::ZERO;
    }
    let nanos = duration.as_nanos() / count as u128;
    Duration::from_nanos(nanos.min(u128::from(u64::MAX)) as u64)
}

#[cfg(target_os = "macos")]
struct ActiveTileGuard<'a> {
    active: &'a AtomicUsize,
}

#[cfg(target_os = "macos")]
impl<'a> ActiveTileGuard<'a> {
    fn new(active: &'a AtomicUsize, observed: &AtomicUsize) -> Self {
        let now = active.fetch_add(1, Ordering::AcqRel).saturating_add(1);
        let mut current = observed.load(Ordering::Relaxed);
        while now > current {
            match observed.compare_exchange(current, now, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => break,
                Err(next) => current = next,
            }
        }
        Self { active }
    }
}

#[cfg(target_os = "macos")]
impl Drop for ActiveTileGuard<'_> {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(target_os = "macos")]
fn encode_planned_resident_lossless_tile(
    planned: PlannedResidentLosslessBufferEncode,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessBufferEncodeOutcome, crate::Error> {
    let encode_started = Instant::now();
    let prepared = prepare_planned_resident_lossless_tile(planned, session)?;
    let (metadata, tier1) = encode_prepared_resident_lossless_tier1(prepared, session)?;
    let submitted = submit_resident_lossless_buffer_encode(metadata, &tier1, session)?;
    let mut finished = wait_submitted_resident_lossless_buffer_encode(submitted)?;
    finished.encode_duration = encode_started.elapsed();
    validate_finished_resident_lossless_buffer_encode(finished, options, session)
}

#[cfg(all(test, target_os = "macos"))]
thread_local! {
    static TEST_RESIDENT_ENCODE_FAILURE_INDEX: Cell<Option<usize>> = const { Cell::new(None) };
}

#[cfg(all(test, target_os = "macos"))]
fn set_test_resident_encode_failure_index(index: Option<usize>) {
    TEST_RESIDENT_ENCODE_FAILURE_INDEX.set(index);
}

#[cfg(all(test, target_os = "macos"))]
fn test_resident_encode_failure_index() -> Option<usize> {
    TEST_RESIDENT_ENCODE_FAILURE_INDEX.get()
}

#[cfg(target_os = "macos")]
fn validate_lossless_roundtrip_on_metal_tile_with_session(
    tile: MetalLosslessEncodeTile<'_>,
    codestream: &[u8],
    session: &crate::MetalBackendSession,
) -> Result<(), crate::Error> {
    let mut decoder = crate::J2kDecoder::new(codestream)?;
    let surface = decoder.decode_to_device_with_session(tile.format, session)?;
    if surface.dimensions() != (tile.output_width, tile.output_height) {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal resident validation geometry mismatch: expected {}x{}, got {}x{}",
                tile.output_width,
                tile.output_height,
                surface.dimensions().0,
                surface.dimensions().1
            ),
        });
    }
    if surface.pixel_format() != tile.format {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal resident validation format mismatch: expected {:?}, got {:?}",
                tile.format,
                surface.pixel_format()
            ),
        });
    }
    let expected_pitch = tile.output_width as usize * tile.format.bytes_per_pixel();
    if surface.pitch_bytes() != expected_pitch || tile.pitch_bytes != expected_pitch {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal resident validation requires contiguous source and decoded rows"
                .to_string(),
        });
    }
    let byte_len = expected_pitch
        .checked_mul(tile.output_height as usize)
        .ok_or_else(|| crate::Error::MetalKernel {
            message: "J2K Metal resident validation byte length overflow".to_string(),
        })?;
    let (decoded_buffer, decoded_offset) =
        surface
            .metal_buffer()
            .ok_or(crate::Error::UnsupportedMetalRequest {
                reason: "J2K Metal resident validation decode did not return a Metal buffer",
            })?;
    compute::validate_metal_buffers_match(
        tile.buffer,
        tile.byte_offset,
        decoded_buffer,
        decoded_offset,
        byte_len,
        session,
    )
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn validate_lossless_roundtrip_on_metal_region_with_session(
    source: MetalLosslessEncodeTile<'_>,
    output_width: u32,
    output_height: u32,
    bytes_per_pixel: usize,
    codestream: &[u8],
    session: &crate::MetalBackendSession,
) -> Result<(), crate::Error> {
    let staged_buffer = compute::copy_interleaved_padded_to_shared_buffer(
        source.buffer,
        source.byte_offset,
        source.width,
        source.height,
        source.pitch_bytes,
        output_width,
        output_height,
        bytes_per_pixel,
        session,
    )?;
    let staged_tile = MetalLosslessEncodeTile {
        buffer: &staged_buffer,
        byte_offset: 0,
        width: output_width,
        height: output_height,
        pitch_bytes: output_width as usize * bytes_per_pixel,
        output_width,
        output_height,
        format: source.format,
    };
    validate_lossless_roundtrip_on_metal_tile_with_session(staged_tile, codestream, session)
}

#[cfg(target_os = "macos")]
fn try_encode_lossless_tile_device_resident_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<Option<MetalLosslessEncodeOutcome>, crate::Error> {
    let Some(outcome) = try_encode_lossless_tile_device_resident_to_metal_buffer_with_report(
        tile, options, session, staging,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(MetalLosslessEncodeOutcome {
        encoded: outcome.encoded.to_encoded_j2k()?,
        input_copy_used: outcome.input_copy_used,
        resident: outcome.resident,
        input_copy_duration: outcome.input_copy_duration,
        encode_duration: outcome.encode_duration,
        gpu_duration: outcome.gpu_duration,
        validation_duration: outcome.validation_duration,
    }))
}

#[cfg(target_os = "macos")]
fn try_encode_lossless_tile_device_resident_to_metal_buffer_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<Option<MetalLosslessBufferEncodeOutcome>, crate::Error> {
    if options.backend == EncodeBackendPreference::CpuOnly {
        return Ok(None);
    }
    let (components, bit_depth) = lossless_sample_shape(tile.format)?;
    let bytes_per_pixel = tile.format.bytes_per_pixel();
    let bytes_per_sample =
        u8::try_from(tile.format.bytes_per_sample()).map_err(|_| crate::Error::MetalKernel {
            message: "J2K Metal resident encode bytes per sample exceeds u8".to_string(),
        })?;
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
        validate_padded_contiguous_metal_encode_tile(tile, bytes_per_pixel)?;
    }
    let Some(plan) = lossless_device_encode_plan(
        tile.output_width,
        tile.output_height,
        components,
        bit_depth,
        options,
    )?
    else {
        return Ok(None);
    };

    let encode_started = Instant::now();
    let coefficient_count = lossless_device_coefficient_count(&plan.code_blocks)?;
    let prepared = compute::prepare_lossless_device_code_blocks(
        session,
        compute::J2kLosslessDevicePrepareJob {
            input: tile.buffer,
            input_byte_offset: tile.byte_offset,
            input_width: tile.width,
            input_height: tile.height,
            input_pitch_bytes: tile.pitch_bytes,
            output_width: tile.output_width,
            output_height: tile.output_height,
            components,
            bytes_per_sample,
            bit_depth,
            num_decomposition_levels: plan.num_decomposition_levels,
            coefficient_count,
        },
        plan.code_blocks.clone(),
    )?;
    let packetization_resolutions =
        resident_packetization_resolutions_from_lossless_device_plan(&plan)?;
    let packet_descriptors =
        packet_descriptors_for_lossless_device_order(plan.resolutions.len(), plan.components)?;
    let packetization_job = compute::J2kResidentPacketizationEncodeJob {
        resolution_count: u32::try_from(plan.resolutions.len()).map_err(|_| {
            crate::Error::MetalKernel {
                message: "J2K Metal resident encode resolution count exceeds u32".to_string(),
            }
        })?,
        num_layers: 1,
        num_components: plan.components,
        code_block_count: u32::try_from(plan.code_blocks.len()).map_err(|_| {
            crate::Error::MetalKernel {
                message: "J2K Metal resident encode code-block count exceeds u32".to_string(),
            }
        })?,
        packet_descriptors: &packet_descriptors,
        resolutions: &packetization_resolutions,
    };
    let assembly_job = compute::J2kLosslessCodestreamAssemblyJob {
        width: tile.output_width,
        height: tile.output_height,
        num_components: plan.components,
        bit_depth: plan.bit_depth,
        signed: false,
        num_decomposition_levels: plan.num_decomposition_levels,
        use_mct: plan.use_mct,
        guard_bits: plan.guard_bits,
        progression_order: plan.progression_order,
        write_tlm: plan.write_tlm,
        block_coding_mode: match plan.block_coding_mode {
            J2kBlockCodingMode::Classic => compute::J2kLosslessCodestreamBlockCodingMode::Classic,
            J2kBlockCodingMode::HighThroughput => {
                compute::J2kLosslessCodestreamBlockCodingMode::HighThroughput
            }
        },
    };
    let codestream = match plan.block_coding_mode {
        J2kBlockCodingMode::Classic => {
            let resident_tier1 =
                compute::encode_classic_tier1_prepared_device_code_blocks_resident(
                    session, prepared,
                )?;
            compute::encode_lossless_codestream_buffer_from_resident_classic_tier1(
                session,
                &resident_tier1,
                packetization_job,
                assembly_job,
            )?
        }
        J2kBlockCodingMode::HighThroughput => {
            let resident_tier1 =
                compute::encode_ht_prepared_device_code_blocks_resident(session, prepared)?;
            compute::encode_lossless_codestream_buffer_from_resident_ht_tier1(
                session,
                &resident_tier1,
                packetization_job,
                assembly_job,
            )?
        }
    };
    let encode_duration = encode_started.elapsed();

    let encoded = MetalEncodedJ2k {
        codestream_buffer: codestream.buffer,
        byte_offset: codestream.byte_offset,
        byte_len: codestream.byte_len,
        capacity: codestream.capacity,
        width: tile.output_width,
        height: tile.output_height,
        components,
        bit_depth,
        signed: false,
    };

    let validation_duration = if options.validation == J2kEncodeValidation::CpuRoundTrip {
        let validation_started = Instant::now();
        if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
            validate_lossless_roundtrip_on_metal_tile_with_session(
                tile,
                encoded.codestream_bytes()?,
                session,
            )?;
        } else {
            validate_lossless_roundtrip_on_metal_region_with_session(
                tile,
                tile.output_width,
                tile.output_height,
                bytes_per_pixel,
                encoded.codestream_bytes()?,
                session,
            )?;
        }
        validation_started.elapsed()
    } else {
        Duration::ZERO
    };

    Ok(Some(MetalLosslessBufferEncodeOutcome {
        encoded,
        input_copy_used: false,
        resident: MetalLosslessEncodeResidency {
            coefficient_prep_used: true,
            packetization_used: true,
            codestream_assembly_used: true,
        },
        input_copy_duration: Duration::ZERO,
        encode_duration,
        gpu_duration: codestream.gpu_duration,
        validation_duration,
    }))
}

#[cfg(target_os = "macos")]
fn encode_lossless_tile_to_metal_buffer_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<MetalLosslessBufferEncodeOutcome, crate::Error> {
    validate_metal_encode_tile(tile)?;
    lossless_sample_shape(tile.format)?;
    if options.backend == EncodeBackendPreference::CpuOnly {
        return Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal buffer output encode requires a device backend",
        });
    }
    let bytes_per_pixel = tile.format.bytes_per_pixel();
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
        validate_padded_contiguous_metal_encode_tile(tile, bytes_per_pixel)?;
    }
    if let Some(outcome) = try_encode_lossless_tile_device_resident_to_metal_buffer_with_report(
        tile, options, session, staging,
    )? {
        return Ok(outcome);
    }
    Err(crate::Error::UnsupportedMetalRequest {
        reason: "J2K Metal buffer output encode requires classic padded contiguous Gray/RGB lossless input with at most one DWT level",
    })
}

#[cfg(target_os = "macos")]
fn encode_lossless_tile_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    accelerator: &mut MetalEncodeStageAccelerator,
) -> Result<MetalLosslessEncodeOutcome, crate::Error> {
    validate_metal_encode_tile(tile)?;
    let (components, bit_depth) = lossless_sample_shape(tile.format)?;
    let bytes_per_pixel = tile.format.bytes_per_pixel();
    if let Some(outcome) =
        try_encode_lossless_tile_device_resident_with_report(tile, options, session, staging)?
    {
        return Ok(outcome);
    }
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous)
        && options.backend == EncodeBackendPreference::RequireDevice
    {
        return Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal resident encode requires classic padded contiguous Gray/RGB lossless input with at most one DWT level",
        });
    }
    let mut input_copy_used = false;
    let mut input_copy_duration = Duration::ZERO;
    let mut staged_buffer = None;
    let mut source_byte_offset = tile.byte_offset;
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
        validate_padded_contiguous_metal_encode_tile(tile, bytes_per_pixel)?;
    } else {
        let copy_started = Instant::now();
        staged_buffer = Some(compute::copy_interleaved_padded_to_shared_buffer(
            tile.buffer,
            tile.byte_offset,
            tile.width,
            tile.height,
            tile.pitch_bytes,
            tile.output_width,
            tile.output_height,
            bytes_per_pixel,
            session,
        )?);
        input_copy_duration = copy_started.elapsed();
        input_copy_used = true;
        source_byte_offset = 0;
    }
    let buffer = staged_buffer.as_ref().unwrap_or(tile.buffer);
    let len = tile.output_width as usize * tile.output_height as usize * bytes_per_pixel;
    let ptr = buffer.contents().cast::<u8>();
    if ptr.is_null() {
        return Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode input buffer is not host-visible",
        });
    }
    let data = unsafe { core::slice::from_raw_parts(ptr.add(source_byte_offset), len) };
    let samples = J2kLosslessSamples::new(
        data,
        tile.output_width,
        tile.output_height,
        components,
        bit_depth,
        false,
    )
    .map_err(crate::Error::Decode)?;

    let mut encode_options = options;
    encode_options.validation = J2kEncodeValidation::External;
    let encode_started = Instant::now();
    let encoded = signinum_j2k::encode_j2k_lossless_with_accelerator(
        samples,
        &encode_options,
        BackendKind::Metal,
        accelerator,
    )
    .map_err(crate::Error::Decode)?;
    let encode_duration = encode_started.elapsed();
    let validation_duration = if options.validation == J2kEncodeValidation::CpuRoundTrip {
        let validation_started = Instant::now();
        validate_lossless_roundtrip_on_metal_with_session(samples, &encoded.codestream, session)?;
        validation_started.elapsed()
    } else {
        Duration::ZERO
    };
    Ok(MetalLosslessEncodeOutcome {
        encoded,
        input_copy_used,
        resident: MetalLosslessEncodeResidency {
            coefficient_prep_used: false,
            packetization_used: false,
            codestream_assembly_used: false,
        },
        input_copy_duration,
        encode_duration,
        gpu_duration: None,
        validation_duration,
    })
}

#[cfg(not(target_os = "macos"))]
pub fn encode_lossless_from_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJ2k, crate::Error> {
    submit_lossless_from_metal_buffer(tile, options, session)?.wait()
}

#[cfg(not(target_os = "macos"))]
pub fn encode_lossless_from_metal_buffer_to_metal(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalEncodedJ2k, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn submit_lossless_from_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalEncode, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn encode_lossless_from_metal_buffer_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessEncodeOutcome, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn encode_lossless_from_metal_buffer_to_metal_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessBufferEncodeOutcome, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn encode_lossless_from_padded_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJ2k, crate::Error> {
    submit_lossless_from_padded_metal_buffer(tile, options, session)?.wait()
}

#[cfg(not(target_os = "macos"))]
pub fn encode_lossless_from_padded_metal_buffer_to_metal(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalEncodedJ2k, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn submit_lossless_from_padded_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalEncode, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn encode_lossless_from_padded_metal_buffer_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessEncodeOutcome, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn encode_lossless_from_padded_metal_buffer_to_metal_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessBufferEncodeOutcome, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn encode_lossless_from_metal_buffers(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<EncodedJ2k>, crate::Error> {
    submit_lossless_from_metal_buffers(tiles, options, session)?.wait()
}

#[cfg(not(target_os = "macos"))]
pub fn encode_lossless_from_metal_buffers_to_metal(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalEncodedJ2k>, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn submit_lossless_from_metal_buffers(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn submit_lossless_from_metal_buffers_with_config(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    let _ = (tiles, options, session, config);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn encode_lossless_from_metal_buffers_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn encode_lossless_from_metal_buffers_to_metal_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessBufferEncodeOutcome>, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn encode_lossless_from_metal_buffers_to_metal_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<MetalLosslessBufferEncodeBatchOutcome, crate::Error> {
    let _ = (tiles, options, session, config);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn encode_lossless_from_padded_metal_buffers(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<EncodedJ2k>, crate::Error> {
    submit_lossless_from_padded_metal_buffers(tiles, options, session)?.wait()
}

#[cfg(not(target_os = "macos"))]
pub fn encode_lossless_from_padded_metal_buffers_to_metal(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalEncodedJ2k>, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn submit_lossless_from_padded_metal_buffers(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn submit_lossless_from_padded_metal_buffers_with_config(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    let _ = (tiles, options, session, config);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn encode_lossless_from_padded_metal_buffers_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn encode_lossless_from_padded_metal_buffers_to_metal_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<MetalLosslessBufferEncodeBatchOutcome, crate::Error> {
    let _ = (tiles, options, session, config);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn encode_lossless_from_padded_metal_buffers_to_metal_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessBufferEncodeOutcome>, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(target_os = "macos")]
pub fn validate_lossless_roundtrip_on_metal(
    samples: J2kLosslessSamples<'_>,
    codestream: &[u8],
) -> Result<(), crate::Error> {
    let session = crate::MetalBackendSession::system_default()?;
    validate_lossless_roundtrip_on_metal_with_session(samples, codestream, &session)
}

#[cfg(not(target_os = "macos"))]
pub fn validate_lossless_roundtrip_on_metal(
    samples: J2kLosslessSamples<'_>,
    codestream: &[u8],
) -> Result<(), crate::Error> {
    let _ = (samples, codestream);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(target_os = "macos")]
pub fn validate_lossless_roundtrip_on_metal_with_session(
    samples: J2kLosslessSamples<'_>,
    codestream: &[u8],
    session: &crate::MetalBackendSession,
) -> Result<(), crate::Error> {
    let fmt = validation_pixel_format(samples)?;
    let mut decoder = crate::J2kDecoder::new(codestream)?;
    let surface = decoder.decode_to_device_with_session(fmt, session)?;

    if surface.dimensions() != (samples.width, samples.height) {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal validation geometry mismatch: expected {}x{}, got {}x{}",
                samples.width,
                samples.height,
                surface.dimensions().0,
                surface.dimensions().1
            ),
        });
    }
    if surface.pixel_format() != fmt {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal validation format mismatch: expected {:?}, got {:?}",
                fmt,
                surface.pixel_format()
            ),
        });
    }
    let expected_pitch = samples.width as usize * fmt.bytes_per_pixel();
    if surface.pitch_bytes() != expected_pitch {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal validation pitch mismatch: expected {expected_pitch}, got {}",
                surface.pitch_bytes()
            ),
        });
    }
    if surface.byte_len() != samples.data.len() {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal validation length mismatch: expected {} bytes, got {} bytes",
                samples.data.len(),
                surface.byte_len()
            ),
        });
    }

    let (buffer, byte_offset) =
        surface
            .metal_buffer()
            .ok_or(crate::Error::UnsupportedMetalRequest {
                reason: "J2K Metal validation decode did not return a Metal buffer",
            })?;
    compute::validate_metal_buffer_matches_bytes(samples.data, buffer, byte_offset, session)
}

#[cfg(not(target_os = "macos"))]
pub fn validate_lossless_roundtrip_on_metal_with_session(
    samples: J2kLosslessSamples<'_>,
    codestream: &[u8],
    session: &crate::MetalBackendSession,
) -> Result<(), crate::Error> {
    let _ = (samples, codestream, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(target_os = "macos")]
fn validation_pixel_format(samples: J2kLosslessSamples<'_>) -> Result<PixelFormat, crate::Error> {
    match (samples.components, samples.bit_depth) {
        (1, 1..=8) => Ok(PixelFormat::Gray8),
        (3, 1..=8) => Ok(PixelFormat::Rgb8),
        (1, 9..=16) => Ok(PixelFormat::Gray16),
        (3, 9..=16) => Ok(PixelFormat::Rgb16),
        _ => Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal validation supports only grayscale or RGB samples up to 16 bits",
        }),
    }
}

#[cfg(target_os = "macos")]
fn lossless_sample_shape(format: PixelFormat) -> Result<(u8, u8), crate::Error> {
    match format {
        PixelFormat::Gray8 => Ok((1, 8)),
        PixelFormat::Rgb8 => Ok((3, 8)),
        PixelFormat::Gray16 => Ok((1, 16)),
        PixelFormat::Rgb16 => Ok((3, 16)),
        PixelFormat::Rgba8 | PixelFormat::Rgba16 => Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode from RGBA tiles requires explicit alpha handling",
        }),
        _ => Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode received an unknown pixel format",
        }),
    }
}

#[cfg(target_os = "macos")]
fn validate_metal_encode_tile(tile: MetalLosslessEncodeTile<'_>) -> Result<(), crate::Error> {
    if tile.width == 0 || tile.height == 0 || tile.output_width == 0 || tile.output_height == 0 {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal encode tile dimensions must be nonzero".to_string(),
        });
    }
    if tile.width > tile.output_width || tile.height > tile.output_height {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal encode input tile exceeds output tile dimensions".to_string(),
        });
    }
    let row_bytes = tile
        .width
        .checked_mul(tile.format.bytes_per_pixel() as u32)
        .ok_or_else(|| crate::Error::MetalKernel {
            message: "J2K Metal encode row byte count overflow".to_string(),
        })? as usize;
    if tile.pitch_bytes < row_bytes {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal encode tile pitch is shorter than one row".to_string(),
        });
    }
    let required_end = tile
        .byte_offset
        .checked_add(
            tile.pitch_bytes
                .checked_mul(tile.height.saturating_sub(1) as usize)
                .and_then(|prefix| prefix.checked_add(row_bytes))
                .ok_or_else(|| crate::Error::MetalKernel {
                    message: "J2K Metal encode input byte range overflow".to_string(),
                })?,
        )
        .ok_or_else(|| crate::Error::MetalKernel {
            message: "J2K Metal encode input byte range overflow".to_string(),
        })?;
    let buffer_len =
        usize::try_from(tile.buffer.length()).map_err(|_| crate::Error::MetalKernel {
            message: "J2K Metal encode buffer length exceeds usize".to_string(),
        })?;
    if required_end > buffer_len {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal encode input byte range exceeds buffer length: need {required_end}, buffer has {buffer_len}"
            ),
        });
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn validate_padded_contiguous_metal_encode_tile(
    tile: MetalLosslessEncodeTile<'_>,
    bytes_per_pixel: usize,
) -> Result<(), crate::Error> {
    if tile.width != tile.output_width || tile.height != tile.output_height {
        return Err(crate::Error::MetalKernel {
            message:
                "J2K Metal no-copy encode requires input dimensions to match output dimensions"
                    .to_string(),
        });
    }
    let expected_pitch = (tile.output_width as usize)
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| crate::Error::MetalKernel {
            message: "J2K Metal no-copy encode pitch overflow".to_string(),
        })?;
    if tile.pitch_bytes != expected_pitch {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal no-copy encode requires contiguous rows: expected pitch {expected_pitch}, got {}",
                tile.pitch_bytes
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::MetalEncodeStageAccelerator;
    #[cfg(target_os = "macos")]
    use crate::compute;
    #[cfg(target_os = "macos")]
    use metal::foreign_types::ForeignType;
    #[cfg(target_os = "macos")]
    use metal::Buffer;
    use signinum_core::DeviceSubmission;
    #[cfg(target_os = "macos")]
    use signinum_core::{BackendKind, PixelFormat};
    #[cfg(target_os = "macos")]
    use signinum_j2k::{
        encode_j2k_lossless_with_accelerator, EncodeBackendPreference, J2kBlockCodingMode,
        J2kEncodeValidation, J2kLosslessSamples, J2kProgressionOrder,
    };
    use signinum_j2k::{EncodedJ2k, J2kLosslessEncodeOptions};
    use signinum_j2k_native::{encode_with_accelerator, DecodeSettings, EncodeOptions, Image};
    #[cfg(target_os = "macos")]
    use signinum_j2k_native::{J2kCodeBlockStyle, J2kEncodeStageAccelerator, J2kForwardDwt53Job};

    #[cfg(target_os = "macos")]
    fn private_buffer_with_bytes(session: &crate::MetalBackendSession, bytes: &[u8]) -> Buffer {
        let upload = session.device().new_buffer_with_data(
            bytes.as_ptr().cast(),
            bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let private = session.device().new_buffer(
            bytes.len() as u64,
            metal::MTLResourceOptions::StorageModePrivate,
        );
        let queue = session.device().new_command_queue();
        let command_buffer = queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(&upload, 0, &private, 0, bytes.len() as u64);
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        private
    }

    #[cfg(target_os = "macos")]
    fn overwrite_private_buffer_with_bytes(
        session: &crate::MetalBackendSession,
        dst: &Buffer,
        bytes: &[u8],
    ) {
        let upload = session.device().new_buffer_with_data(
            bytes.as_ptr().cast(),
            bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let queue = session.device().new_command_queue();
        let command_buffer = queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(&upload, 0, dst, 0, bytes.len() as u64);
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn inflight_limited_runner_starts_next_item_before_slow_peer_finishes() {
        use std::sync::{Arc, Condvar, Mutex};
        use std::time::Duration;

        #[derive(Default)]
        struct Probe {
            third_item_started: bool,
        }

        let probe = Arc::new((Mutex::new(Probe::default()), Condvar::new()));
        let task_probe = Arc::clone(&probe);

        let outcomes = super::collect_inflight_limited_ordered(vec![0usize, 1, 2], 2, move |_, item| {
            match item {
                0 => Ok(item),
                1 => {
                    let (lock, cvar) = &*task_probe;
                    let state = lock.lock().expect("probe mutex");
                    let (state, _timeout) = cvar
                        .wait_timeout_while(state, Duration::from_millis(250), |state| {
                            !state.third_item_started
                        })
                        .expect("probe wait");
                    if !state.third_item_started {
                        return Err(crate::Error::MetalKernel {
                            message:
                                "runner waited for the whole in-flight chunk before scheduling more work"
                                    .to_string(),
                        });
                    }
                    Ok(item)
                }
                2 => {
                    let (lock, cvar) = &*task_probe;
                    let mut state = lock.lock().expect("probe mutex");
                    state.third_item_started = true;
                    cvar.notify_all();
                    Ok(item)
                }
                _ => unreachable!("unexpected test item"),
            }
        })
        .expect("in-flight runner should slide past a slow peer");

        assert_eq!(outcomes.items, vec![0, 1, 2]);
        assert!(outcomes.max_observed_inflight_items <= 2);
        assert!(outcomes.max_observed_inflight_items > 0);
    }

    #[test]
    fn submitted_lossless_metal_encode_public_api_is_available() {
        fn assert_single_submission<
            S: DeviceSubmission<Output = EncodedJ2k, Error = crate::Error>,
        >() {
        }
        fn assert_batch_submission<
            S: DeviceSubmission<Output = Vec<EncodedJ2k>, Error = crate::Error>,
        >() {
        }
        fn assert_submit_single_fn(
            _submit: for<'tile, 'options, 'session> fn(
                super::MetalLosslessEncodeTile<'tile>,
                &'options J2kLosslessEncodeOptions,
                &'session crate::MetalBackendSession,
            ) -> Result<
                crate::SubmittedJ2kLosslessMetalEncode,
                crate::Error,
            >,
        ) {
        }
        fn assert_submit_batch_fn(
            _submit: for<'slice, 'tile, 'options, 'session> fn(
                &'slice [super::MetalLosslessEncodeTile<'tile>],
                &'options J2kLosslessEncodeOptions,
                &'session crate::MetalBackendSession,
            ) -> Result<
                crate::SubmittedJ2kLosslessMetalEncodeBatch,
                crate::Error,
            >,
        ) {
        }

        assert_single_submission::<crate::SubmittedJ2kLosslessMetalEncode>();
        assert_batch_submission::<crate::SubmittedJ2kLosslessMetalEncodeBatch>();
        assert_submit_single_fn(crate::submit_lossless_from_metal_buffer);
        assert_submit_single_fn(crate::submit_lossless_from_padded_metal_buffer);
        assert_submit_batch_fn(crate::submit_lossless_from_metal_buffers);
        assert_submit_batch_fn(crate::submit_lossless_from_padded_metal_buffers);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_dispatch_option_treats_unavailable_as_no_dispatch() {
        let result: Result<Option<u8>, &'static str> =
            super::metal_dispatch_option(Err(crate::Error::MetalUnavailable), "kernel failed");

        assert_eq!(result, Ok(None));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_dispatch_option_preserves_kernel_errors() {
        let result: Result<Option<u8>, &'static str> = super::metal_dispatch_option(
            Err(crate::Error::MetalKernel {
                message: "bad status".to_string(),
            }),
            "kernel failed",
        );

        assert_eq!(result, Err("kernel failed"));
    }

    #[test]
    fn metal_encode_stage_accelerator_preserves_cpu_codestream_validity() {
        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| (i & 0xFF) as u8).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 8, 8, 3, 8, false, &options, &mut accelerator)
                .expect("encode with metal stage accelerator");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.num_components, 3);
        assert_eq!(decoded.bit_depth, 8);
        assert_eq!(accelerator.forward_rct_attempts(), 1);
        assert_eq!(accelerator.forward_dwt53_attempts(), 3);
        assert!(accelerator.tier1_code_block_attempts() > 0);
        assert_eq!(accelerator.packetization_attempts(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_forward_rct_dispatch_round_trips_rgb8_lossless_tile() {
        let pixels: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 17) & 0xFF) as u8).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            ..EncodeOptions::default()
        };
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 7, 5, 3, 8, false, &options, &mut accelerator)
                .expect("encode with metal forward RCT");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(accelerator.forward_rct_attempts(), 1);
        assert_eq!(accelerator.forward_rct_dispatches(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_validation_decodes_and_compares_lossless_codestream_on_device() {
        let pixels: Vec<u8> = (0..16 * 16 * 3).map(|i| ((i * 29) & 0xFF) as u8).collect();
        let samples = J2kLosslessSamples::new(&pixels, 16, 16, 3, 8, false).unwrap();
        let encoded = signinum_j2k::encode_j2k_lossless(
            samples,
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::CpuOnly,
                ..J2kLosslessEncodeOptions::default()
            },
        )
        .expect("lossless encode");

        super::validate_lossless_roundtrip_on_metal(samples, &encoded.codestream)
            .expect("Metal lossless validation");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_buffer_lossless_encode_pads_edge_tile_on_device() {
        let pixels: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 19) & 0xFF) as u8).collect();
        let device = metal::Device::system_default().expect("Metal device");
        if !compute::ht_simd_prototype_available_for_device_for_test(&device)
            .expect("HTJ2K SIMD prototype availability query")
        {
            return;
        }
        let session = crate::MetalBackendSession::new(device);
        let buffer = session.device().new_buffer_with_data(
            pixels.as_ptr().cast(),
            pixels.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let encoded = super::encode_lossless_from_metal_buffer(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 7,
                height: 5,
                pitch_bytes: 7 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::RequireDevice,
                ..J2kLosslessEncodeOptions::default()
            },
            &session,
        )
        .expect("Metal buffer lossless encode");

        assert_eq!(encoded.backend, BackendKind::Metal);
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        for y in 0..8usize {
            for x in 0..8usize {
                let dst = (y * 8 + x) * 3;
                if x < 7 && y < 5 {
                    let src = (y * 7 + x) * 3;
                    assert_eq!(&decoded.data[dst..dst + 3], &pixels[src..src + 3]);
                } else {
                    assert_eq!(&decoded.data[dst..dst + 3], &[0, 0, 0]);
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn submitted_metal_buffer_lossless_encode_wait_round_trips() {
        let pixels: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 19) & 0xFF) as u8).collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = session.device().new_buffer_with_data(
            pixels.as_ptr().cast(),
            pixels.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let submitted: crate::SubmittedJ2kLosslessMetalEncode =
            crate::submit_lossless_from_metal_buffer(
                super::MetalLosslessEncodeTile {
                    buffer: &buffer,
                    byte_offset: 0,
                    width: 7,
                    height: 5,
                    pitch_bytes: 7 * 3,
                    output_width: 8,
                    output_height: 8,
                    format: PixelFormat::Rgb8,
                },
                &J2kLosslessEncodeOptions {
                    backend: EncodeBackendPreference::RequireDevice,
                    ..J2kLosslessEncodeOptions::default()
                },
                &session,
            )
            .expect("submit Metal buffer lossless encode");
        let encoded = submitted.wait().expect("wait Metal buffer lossless encode");

        assert_eq!(encoded.backend, BackendKind::Metal);
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        for y in 0..8usize {
            for x in 0..8usize {
                let dst = (y * 8 + x) * 3;
                if x < 7 && y < 5 {
                    let src = (y * 7 + x) * 3;
                    assert_eq!(&decoded.data[dst..dst + 3], &pixels[src..src + 3]);
                } else {
                    assert_eq!(&decoded.data[dst..dst + 3], &[0, 0, 0]);
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_buffer_lossless_encode_accepts_padded_contiguous_input_without_copy() {
        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 31) & 0xFF) as u8).collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = session.device().new_buffer_with_data(
            pixels.as_ptr().cast(),
            pixels.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::RequireDevice,
                ..J2kLosslessEncodeOptions::default()
            },
            &session,
        )
        .expect("Metal padded buffer lossless encode");

        assert_eq!(encoded.encoded.backend, BackendKind::Metal);
        assert!(!encoded.input_copy_used);
        assert_eq!(encoded.input_copy_duration, std::time::Duration::ZERO);
        let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_rgb8_encode_uses_resident_coefficient_prep() {
        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 31) & 0xFF) as u8).collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::RequireDevice,
                ..J2kLosslessEncodeOptions::default()
            },
            &session,
        )
        .expect("Metal private padded buffer lossless encode");

        assert_eq!(encoded.encoded.backend, BackendKind::Metal);
        assert!(!encoded.input_copy_used);
        assert!(encoded.resident.coefficient_prep_used);
        assert!(encoded.resident.packetization_used);
        assert!(encoded.resident.codestream_assembly_used);
        let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_rgb8_encode_to_metal_buffer_exposes_finished_bytes() {
        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 37) & 0xFF) as u8).collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let encoded = super::encode_lossless_from_padded_metal_buffer_to_metal_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::RequireDevice,
                ..J2kLosslessEncodeOptions::default()
            },
            &session,
        )
        .expect("Metal private padded buffer lossless encode to Metal buffer");

        assert!(!encoded.input_copy_used);
        assert!(encoded.resident.coefficient_prep_used);
        assert!(encoded.resident.packetization_used);
        assert!(encoded.resident.codestream_assembly_used);
        assert!(
            encoded.gpu_duration.is_some(),
            "resident Metal encode should report command-buffer GPU duration"
        );
        assert_eq!(encoded.encoded.byte_offset, 0);
        assert!(encoded.encoded.byte_len > 0);
        assert!(encoded.encoded.capacity >= encoded.encoded.byte_len);
        let codestream = encoded
            .encoded
            .codestream_bytes()
            .expect("Metal codestream bytes are CPU-readable");
        assert!(codestream.starts_with(&[0xFF, 0x4F]));
        let decoded = Image::new(codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_edge_private_rgb8_encode_to_metal_buffer_pads_and_stays_resident() {
        let pixels: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 41) & 0xFF) as u8).collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let encoded = super::encode_lossless_from_metal_buffer_to_metal_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 7,
                height: 5,
                pitch_bytes: 7 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::RequireDevice,
                ..J2kLosslessEncodeOptions::default()
            },
            &session,
        )
        .expect("Metal private edge buffer lossless encode to Metal buffer");

        assert!(!encoded.input_copy_used);
        assert!(encoded.resident.coefficient_prep_used);
        assert!(encoded.resident.packetization_used);
        assert!(encoded.resident.codestream_assembly_used);
        let codestream = encoded
            .encoded
            .codestream_bytes()
            .expect("Metal codestream bytes are CPU-readable");
        assert!(codestream.starts_with(&[0xFF, 0x4F]));
        let decoded = Image::new(codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        for y in 0..8usize {
            for x in 0..8usize {
                let dst = (y * 8 + x) * 3;
                if x < 7 && y < 5 {
                    let src = (y * 7 + x) * 3;
                    assert_eq!(&decoded.data[dst..dst + 3], &pixels[src..src + 3]);
                } else {
                    assert_eq!(&decoded.data[dst..dst + 3], &[0, 0, 0]);
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn submitted_private_padded_rgb8_encode_snapshots_before_wait() {
        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 31) & 0xFF) as u8).collect();
        let replacement = vec![0u8; pixels.len()];
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let submitted = super::submit_lossless_from_padded_metal_buffer(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::RequireDevice,
                ..J2kLosslessEncodeOptions::default()
            },
            &session,
        )
        .expect("submit Metal private padded RGB8 encode");
        overwrite_private_buffer_with_bytes(&session, &buffer, &replacement);

        let encoded = submitted.wait().expect("wait submitted encode");
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_gray8_dwt_encode_uses_resident_coefficient_prep() {
        let mut pixels = Vec::with_capacity(128 * 128);
        for y in 0..128u32 {
            for x in 0..128u32 {
                pixels.push(((x * 7 + y * 11 + (x ^ y)) & 0xFF) as u8);
            }
        }
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 128,
                height: 128,
                pitch_bytes: 128,
                output_width: 128,
                output_height: 128,
                format: PixelFormat::Gray8,
            },
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::RequireDevice,
                ..J2kLosslessEncodeOptions::default()
            },
            &session,
        )
        .expect("Metal private padded DWT buffer lossless encode");

        assert_eq!(encoded.encoded.backend, BackendKind::Metal);
        assert!(!encoded.input_copy_used);
        assert!(encoded.resident.coefficient_prep_used);
        assert!(encoded.resident.packetization_used);
        assert!(encoded.resident.codestream_assembly_used);
        let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_rgb8_dwt_encode_uses_resident_coefficient_prep() {
        let mut pixels = Vec::with_capacity(128 * 128 * 3);
        for y in 0..128u32 {
            for x in 0..128u32 {
                pixels.push(((x * 3 + y * 5) & 0xFF) as u8);
                pixels.push(((x * 7 + y * 11) & 0xFF) as u8);
                pixels.push(((x * 13 + y * 17) & 0xFF) as u8);
            }
        }
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 128,
                height: 128,
                pitch_bytes: 128 * 3,
                output_width: 128,
                output_height: 128,
                format: PixelFormat::Rgb8,
            },
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::RequireDevice,
                ..J2kLosslessEncodeOptions::default()
            },
            &session,
        )
        .expect("Metal private padded RGB8 DWT buffer lossless encode");

        assert_eq!(encoded.encoded.backend, BackendKind::Metal);
        assert!(encoded.resident.coefficient_prep_used);
        assert!(encoded.resident.packetization_used);
        assert!(encoded.resident.codestream_assembly_used);
        let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_gray8_rpcl_encode_uses_resident_coefficient_prep() {
        let mut pixels = Vec::with_capacity(128 * 128);
        for y in 0..128u32 {
            for x in 0..128u32 {
                pixels.push(((x * 5 + y * 9 + (x ^ y)) & 0xFF) as u8);
            }
        }
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 128,
                height: 128,
                pitch_bytes: 128,
                output_width: 128,
                output_height: 128,
                format: PixelFormat::Gray8,
            },
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::RequireDevice,
                progression: J2kProgressionOrder::Rpcl,
                ..J2kLosslessEncodeOptions::default()
            },
            &session,
        )
        .expect("Metal private padded RPCL buffer lossless encode");

        assert_eq!(encoded.encoded.backend, BackendKind::Metal);
        assert!(encoded.resident.coefficient_prep_used);
        assert!(encoded.resident.packetization_used);
        assert!(encoded.resident.codestream_assembly_used);
        let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_gray16_encode_uses_resident_coefficient_prep() {
        let mut pixels = Vec::with_capacity(8 * 8 * 2);
        for idx in 0..64u16 {
            let value = idx.wrapping_mul(997).wrapping_add(123);
            pixels.extend_from_slice(&value.to_le_bytes());
        }
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8 * 2,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Gray16,
            },
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::RequireDevice,
                ..J2kLosslessEncodeOptions::default()
            },
            &session,
        )
        .expect("Metal private padded Gray16 buffer lossless encode");

        assert_eq!(encoded.encoded.backend, BackendKind::Metal);
        assert!(!encoded.input_copy_used);
        assert!(encoded.resident.coefficient_prep_used);
        assert!(encoded.resident.packetization_used);
        assert!(encoded.resident.codestream_assembly_used);
        let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_ht_encode_to_metal_buffer_stays_resident() {
        let pixels: Vec<u8> = (0..8 * 8).map(|i| ((i * 31) & 0xFF) as u8).collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let encoded = super::encode_lossless_from_padded_metal_buffer_to_metal_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Gray8,
            },
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::RequireDevice,
                block_coding_mode: J2kBlockCodingMode::HighThroughput,
                ..J2kLosslessEncodeOptions::default()
            },
            &session,
        )
        .expect("Metal private padded HTJ2K buffer lossless encode");

        assert!(!encoded.input_copy_used);
        assert!(encoded.resident.coefficient_prep_used);
        assert!(encoded.resident.packetization_used);
        assert!(encoded.resident.codestream_assembly_used);
        let codestream = encoded
            .encoded
            .codestream_bytes()
            .expect("Metal codestream bytes are CPU-readable");
        assert!(codestream.windows(2).any(|window| window == [0xFF, 0x50]));
        let cod_marker = codestream
            .windows(2)
            .position(|window| window == [0xFF, 0x52])
            .expect("COD marker");
        assert_eq!(codestream[cod_marker + 12], 0x40);
        let decoded = Image::new(codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_buffer_lossless_batch_encodes_padded_contiguous_inputs() {
        let first: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 7) & 0xFF) as u8).collect();
        let second: Vec<u8> = (0..8 * 8 * 3)
            .map(|i| ((i * 13 + 5) & 0xFF) as u8)
            .collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let first_buffer = session.device().new_buffer_with_data(
            first.as_ptr().cast(),
            first.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let second_buffer = session.device().new_buffer_with_data(
            second.as_ptr().cast(),
            second.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let tiles = [
            super::MetalLosslessEncodeTile {
                buffer: &first_buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
            super::MetalLosslessEncodeTile {
                buffer: &second_buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
        ];

        let encoded = super::encode_lossless_from_padded_metal_buffers_with_report(
            &tiles,
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::RequireDevice,
                ..J2kLosslessEncodeOptions::default()
            },
            &session,
        )
        .expect("Metal padded buffer batch lossless encode");

        assert_eq!(encoded.len(), 2);
        for (frame, expected) in encoded.iter().zip([first, second]) {
            assert_eq!(frame.encoded.backend, BackendKind::Metal);
            assert!(!frame.input_copy_used);
            let decoded = Image::new(&frame.encoded.codestream, &DecodeSettings::default())
                .expect("codestream parses")
                .decode_native()
                .expect("codestream decodes");
            assert_eq!(decoded.data, expected);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_batch_encode_to_metal_buffers_exposes_per_frame_bytes() {
        let first: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 17) & 0xFF) as u8).collect();
        let second: Vec<u8> = (0..8 * 8 * 3)
            .map(|i| 255u8.wrapping_sub(((i * 23) & 0xFF) as u8))
            .collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let first_buffer = private_buffer_with_bytes(&session, &first);
        let second_buffer = private_buffer_with_bytes(&session, &second);
        let tiles = [
            super::MetalLosslessEncodeTile {
                buffer: &first_buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
            super::MetalLosslessEncodeTile {
                buffer: &second_buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
        ];

        let encoded = super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
            &tiles,
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::RequireDevice,
                ..J2kLosslessEncodeOptions::default()
            },
            &session,
        )
        .expect("Metal padded buffer batch lossless encode to Metal buffers");

        assert_eq!(encoded.len(), 2);
        assert_eq!(
            encoded[0].encoded.codestream_buffer.as_ptr(),
            encoded[1].encoded.codestream_buffer.as_ptr(),
            "classic J2K resident batch encode should assemble codestreams into one shared batch buffer"
        );
        assert_eq!(encoded[0].encoded.byte_offset, 0);
        assert!(
            encoded[1].encoded.byte_offset > 0,
            "second classic J2K batch codestream should be a nonzero slice into the shared batch buffer"
        );
        for (frame, expected) in encoded.iter().zip([first, second]) {
            assert!(!frame.input_copy_used);
            assert!(frame.resident.coefficient_prep_used);
            assert!(frame.resident.packetization_used);
            assert!(frame.resident.codestream_assembly_used);
            let codestream = frame
                .encoded
                .codestream_bytes()
                .expect("Metal codestream bytes are CPU-readable");
            let decoded = Image::new(codestream, &DecodeSettings::default())
                .expect("codestream parses")
                .decode_native()
                .expect("codestream decodes");
            assert_eq!(decoded.data, expected);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_edge_private_batch_encode_to_metal_buffers_stays_resident() {
        let first: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 17) & 0xFF) as u8).collect();
        let second: Vec<u8> = (0..6 * 8 * 3)
            .map(|i| 255u8.wrapping_sub(((i * 19) & 0xFF) as u8))
            .collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let first_buffer = private_buffer_with_bytes(&session, &first);
        let second_buffer = private_buffer_with_bytes(&session, &second);
        compute::reset_ht_batch_coefficient_copy_blits_for_test();
        let tiles = [
            super::MetalLosslessEncodeTile {
                buffer: &first_buffer,
                byte_offset: 0,
                width: 7,
                height: 5,
                pitch_bytes: 7 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
            super::MetalLosslessEncodeTile {
                buffer: &second_buffer,
                byte_offset: 0,
                width: 6,
                height: 8,
                pitch_bytes: 6 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
        ];

        let encoded = super::encode_lossless_from_metal_buffers_to_metal_with_report(
            &tiles,
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::RequireDevice,
                ..J2kLosslessEncodeOptions::default()
            },
            &session,
        )
        .expect("Metal edge buffer batch lossless encode to Metal buffers");

        assert_eq!(encoded.len(), 2);
        for frame in &encoded {
            assert!(!frame.input_copy_used);
            assert!(frame.resident.coefficient_prep_used);
            assert!(frame.resident.packetization_used);
            assert!(frame.resident.codestream_assembly_used);
        }

        for (frame, (expected, width, height)) in encoded
            .iter()
            .zip([(first, 7usize, 5usize), (second, 6usize, 8usize)])
        {
            let codestream = frame
                .encoded
                .codestream_bytes()
                .expect("Metal codestream bytes are CPU-readable");
            let decoded = Image::new(codestream, &DecodeSettings::default())
                .expect("codestream parses")
                .decode_native()
                .expect("codestream decodes");
            for y in 0..8usize {
                for x in 0..8usize {
                    let dst = (y * 8 + x) * 3;
                    if x < width && y < height {
                        let src = (y * width + x) * 3;
                        assert_eq!(&decoded.data[dst..dst + 3], &expected[src..src + 3]);
                    } else {
                        assert_eq!(&decoded.data[dst..dst + 3], &[0, 0, 0]);
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_ht_private_batch_encode_to_metal_buffers_stays_resident() {
        let first: Vec<u8> = (0..8 * 8).map(|i| ((i * 11) & 0xFF) as u8).collect();
        let second: Vec<u8> = (0..8 * 8)
            .map(|i| 255u8.wrapping_sub(((i * 13) & 0xFF) as u8))
            .collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let first_buffer = private_buffer_with_bytes(&session, &first);
        let second_buffer = private_buffer_with_bytes(&session, &second);
        let tiles = [
            super::MetalLosslessEncodeTile {
                buffer: &first_buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Gray8,
            },
            super::MetalLosslessEncodeTile {
                buffer: &second_buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Gray8,
            },
        ];

        let encoded = super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
            &tiles,
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::RequireDevice,
                block_coding_mode: J2kBlockCodingMode::HighThroughput,
                ..J2kLosslessEncodeOptions::default()
            },
            &session,
        )
        .expect("Metal HTJ2K batch lossless encode to Metal buffers");

        assert_eq!(encoded.len(), 2);
        assert_eq!(
            compute::ht_batch_coefficient_copy_blits_for_test(),
            0,
            "HTJ2K resident batch prep should write directly into the batch coefficient buffer"
        );
        assert_eq!(
            encoded[0].encoded.codestream_buffer.as_ptr(),
            encoded[1].encoded.codestream_buffer.as_ptr(),
            "HTJ2K resident batch encode should assemble codestreams into one shared batch buffer"
        );
        assert_eq!(encoded[0].encoded.byte_offset, 0);
        assert!(
            encoded[1].encoded.byte_offset > 0,
            "second HTJ2K batch codestream should be a nonzero slice into the shared batch buffer"
        );
        for (frame, expected) in encoded.iter().zip([first, second]) {
            assert!(!frame.input_copy_used);
            assert!(frame.resident.coefficient_prep_used);
            assert!(frame.resident.packetization_used);
            assert!(frame.resident.codestream_assembly_used);
            let codestream = frame
                .encoded
                .codestream_bytes()
                .expect("Metal codestream bytes are CPU-readable");
            assert!(codestream.windows(2).any(|window| window == [0xFF, 0x50]));
            let decoded = Image::new(codestream, &DecodeSettings::default())
                .expect("codestream parses")
                .decode_native()
                .expect("codestream decodes");
            assert_eq!(decoded.data, expected);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_ht_private_batch_encode_reuses_private_arenas_between_batches() {
        const WIDTH: usize = 37;
        const HEIGHT: usize = 41;
        let first: Vec<u8> = (0..WIDTH * HEIGHT)
            .map(|i| ((i * 7 + 3) & 0xFF) as u8)
            .collect();
        let second: Vec<u8> = (0..WIDTH * HEIGHT)
            .map(|i| 255u8.wrapping_sub(((i * 5 + 11) & 0xFF) as u8))
            .collect();
        let device = metal::Device::system_default().expect("Metal device");
        let session = crate::MetalBackendSession::new(device.clone());
        let first_buffer = private_buffer_with_bytes(&session, &first);
        let second_buffer = private_buffer_with_bytes(&session, &second);
        let tiles = [
            super::MetalLosslessEncodeTile {
                buffer: &first_buffer,
                byte_offset: 0,
                width: WIDTH as u32,
                height: HEIGHT as u32,
                pitch_bytes: WIDTH,
                output_width: WIDTH as u32,
                output_height: HEIGHT as u32,
                format: PixelFormat::Gray8,
            },
            super::MetalLosslessEncodeTile {
                buffer: &second_buffer,
                byte_offset: 0,
                width: WIDTH as u32,
                height: HEIGHT as u32,
                pitch_bytes: WIDTH,
                output_width: WIDTH as u32,
                output_height: HEIGHT as u32,
                format: PixelFormat::Gray8,
            },
        ];
        let options = J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            validation: J2kEncodeValidation::External,
            ..J2kLosslessEncodeOptions::default()
        };

        compute::with_isolated_runtime_for_device_for_test(&device, || {
            compute::reset_private_buffer_pool_misses_for_test();
            super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
                &tiles, &options, &session,
            )?;
            let first_misses = compute::private_buffer_pool_misses_for_test();
            assert!(
                first_misses > 0,
                "first unique HTJ2K batch should populate reusable private arenas"
            );

            compute::reset_private_buffer_pool_misses_for_test();
            let encoded = super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
                &tiles, &options, &session,
            )?;

            assert_eq!(
                compute::private_buffer_pool_misses_for_test(),
                0,
                "second same-shape HTJ2K batch should reuse private arenas"
            );
            assert_eq!(encoded.len(), 2);
            Ok(())
        })
        .expect("isolated HTJ2K Metal runtime");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_ht_private_batch_encode_uses_simd_prototype_when_enabled() {
        const WIDTH: usize = 32;
        const HEIGHT: usize = 32;
        let first: Vec<u8> = (0..WIDTH * HEIGHT)
            .map(|i| ((i * 3 + 17) & 0xFF) as u8)
            .collect();
        let second: Vec<u8> = (0..WIDTH * HEIGHT)
            .map(|i| ((i * 11 + 5) & 0xFF) as u8)
            .collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let first_buffer = private_buffer_with_bytes(&session, &first);
        let second_buffer = private_buffer_with_bytes(&session, &second);
        let tiles = [
            super::MetalLosslessEncodeTile {
                buffer: &first_buffer,
                byte_offset: 0,
                width: WIDTH as u32,
                height: HEIGHT as u32,
                pitch_bytes: WIDTH,
                output_width: WIDTH as u32,
                output_height: HEIGHT as u32,
                format: PixelFormat::Gray8,
            },
            super::MetalLosslessEncodeTile {
                buffer: &second_buffer,
                byte_offset: 0,
                width: WIDTH as u32,
                height: HEIGHT as u32,
                pitch_bytes: WIDTH,
                output_width: WIDTH as u32,
                output_height: HEIGHT as u32,
                format: PixelFormat::Gray8,
            },
        ];
        let options = J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            validation: J2kEncodeValidation::External,
            ..J2kLosslessEncodeOptions::default()
        };

        let _route = compute::force_ht_simd_prototype_route_for_test(true);
        compute::reset_ht_simd_prototype_dispatches_for_test();
        let encoded = super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
            &tiles, &options, &session,
        )
        .expect("HTJ2K SIMD prototype batch encode");

        assert_eq!(encoded.len(), 2);
        assert!(
            compute::ht_simd_prototype_dispatches_for_test() > 0,
            "enabled HTJ2K batch encode should route through the SIMD prototype"
        );
        for (frame, expected) in encoded.iter().zip([first, second]) {
            let codestream = frame
                .encoded
                .codestream_bytes()
                .expect("Metal codestream bytes are CPU-readable");
            let decoded = Image::new(codestream, &DecodeSettings::default())
                .expect("codestream parses")
                .decode_native()
                .expect("codestream decodes");
            assert_eq!(decoded.data, expected);
        }
    }

    #[test]
    fn default_gpu_encode_memory_budget_uses_forty_percent_capped_at_ten_gib() {
        const GIB: usize = 1024 * 1024 * 1024;

        assert_eq!(
            super::default_gpu_encode_memory_budget_bytes_for_hw_mem(8 * GIB),
            8 * GIB * 40 / 100
        );
        assert_eq!(
            super::default_gpu_encode_memory_budget_bytes_for_hw_mem(16 * GIB),
            16 * GIB * 40 / 100
        );
        assert_eq!(
            super::default_gpu_encode_memory_budget_bytes_for_hw_mem(24 * GIB),
            24 * GIB * 40 / 100
        );
        assert_eq!(
            super::default_gpu_encode_memory_budget_bytes_for_hw_mem(64 * GIB),
            10 * GIB
        );
    }

    #[test]
    fn gpu_encode_inflight_resolution_clamps_requested_tiles_by_memory_budget() {
        let stats = super::resolve_lossless_encode_config_for_test(
            100,
            1_000,
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: Some(32),
                gpu_encode_memory_budget_bytes: Some(4_500),
            },
        )
        .expect("resolved config");

        assert_eq!(stats.configured_inflight_tiles, Some(32));
        assert_eq!(stats.effective_inflight_tiles, 4);
        assert_eq!(stats.configured_memory_budget_bytes, Some(4_500));
        assert_eq!(stats.effective_memory_budget_bytes, 4_500);
        assert_eq!(stats.estimated_peak_bytes_per_tile, 1_000);
    }

    #[test]
    fn gpu_encode_default_inflight_uses_sixty_four_when_memory_allows() {
        let stats = super::resolve_lossless_encode_config_for_test(
            100,
            1_000,
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: None,
                gpu_encode_memory_budget_bytes: Some(1_000_000),
            },
        )
        .expect("resolved config");

        assert_eq!(stats.configured_inflight_tiles, None);
        assert_eq!(stats.effective_inflight_tiles, 64);
    }

    #[test]
    fn gpu_encode_inflight_resolution_rejects_zero_overrides() {
        let err = super::resolve_lossless_encode_config_for_test(
            4,
            1_000,
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: Some(0),
                gpu_encode_memory_budget_bytes: Some(4_000),
            },
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("in-flight"),
            "unexpected error: {err}"
        );

        let err = super::resolve_lossless_encode_config_for_test(
            4,
            1_000,
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: Some(2),
                gpu_encode_memory_budget_bytes: Some(0),
            },
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("memory budget"),
            "unexpected error: {err}"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_ht_batch_encode_preserves_order_and_matches_inflight_one() {
        let inputs = [
            (0..8 * 8)
                .map(|i| ((i * 11 + 3) & 0xFF) as u8)
                .collect::<Vec<_>>(),
            (0..8 * 8)
                .map(|i| ((i * 13 + 5) & 0xFF) as u8)
                .collect::<Vec<_>>(),
            (0..8 * 8)
                .map(|i| ((i * 17 + 7) & 0xFF) as u8)
                .collect::<Vec<_>>(),
            (0..8 * 8)
                .map(|i| ((i * 19 + 9) & 0xFF) as u8)
                .collect::<Vec<_>>(),
        ];
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffers = inputs
            .iter()
            .map(|bytes| private_buffer_with_bytes(&session, bytes))
            .collect::<Vec<_>>();
        let tiles = buffers
            .iter()
            .map(|buffer| super::MetalLosslessEncodeTile {
                buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Gray8,
            })
            .collect::<Vec<_>>();
        let options = J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            validation: J2kEncodeValidation::External,
            ..J2kLosslessEncodeOptions::default()
        };

        let serial = super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
            &tiles,
            &options,
            &session,
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: Some(1),
                gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
            },
        )
        .expect("serial Metal HTJ2K batch");
        let parallel = super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
            &tiles,
            &options,
            &session,
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: Some(2),
                gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
            },
        )
        .expect("parallel Metal HTJ2K batch");
        let repeated_parallel = super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
            &tiles,
            &options,
            &session,
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: Some(2),
                gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
            },
        )
        .expect("repeated parallel Metal HTJ2K batch");

        assert_eq!(serial.outcomes.len(), inputs.len());
        assert_eq!(parallel.outcomes.len(), inputs.len());
        assert_eq!(parallel.stats.effective_inflight_tiles, 2);
        assert!(parallel.stats.max_observed_inflight_tiles <= 2);
        assert!(parallel.stats.max_observed_inflight_tiles > 0);
        for (((serial_outcome, parallel_outcome), repeated_outcome), expected) in serial
            .outcomes
            .iter()
            .zip(parallel.outcomes.iter())
            .zip(repeated_parallel.outcomes.iter())
            .zip(inputs.iter())
        {
            let serial_bytes = serial_outcome
                .encoded
                .codestream_bytes()
                .expect("serial codestream");
            let parallel_bytes = parallel_outcome
                .encoded
                .codestream_bytes()
                .expect("parallel codestream");
            let repeated_bytes = repeated_outcome
                .encoded
                .codestream_bytes()
                .expect("repeated parallel codestream");
            assert_eq!(parallel_bytes, serial_bytes);
            assert_eq!(repeated_bytes, serial_bytes);

            let decoded = Image::new(parallel_bytes, &DecodeSettings::default())
                .expect("codestream parses")
                .decode_native()
                .expect("codestream decodes");
            assert_eq!(&decoded.data, expected);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_parallel_batch_returns_indexed_injected_failure() {
        let first: Vec<u8> = (0..8 * 8).map(|i| ((i * 3) & 0xFF) as u8).collect();
        let second: Vec<u8> = (0..8 * 8).map(|i| ((i * 5) & 0xFF) as u8).collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let first_buffer = private_buffer_with_bytes(&session, &first);
        let second_buffer = private_buffer_with_bytes(&session, &second);
        let tiles = [
            super::MetalLosslessEncodeTile {
                buffer: &first_buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Gray8,
            },
            super::MetalLosslessEncodeTile {
                buffer: &second_buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Gray8,
            },
        ];
        let options = J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            validation: J2kEncodeValidation::External,
            ..J2kLosslessEncodeOptions::default()
        };

        super::set_test_resident_encode_failure_index(Some(1));
        let Err(err) = super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
            &tiles,
            &options,
            &session,
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: Some(2),
                gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
            },
        ) else {
            panic!("injected failure should fail the batch");
        };
        super::set_test_resident_encode_failure_index(None);

        assert!(
            err.to_string().contains("tile 1"),
            "unexpected error: {err}"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_forward_dwt53_dispatch_round_trips_gray8_lossless_tile() {
        let pixels: Vec<u8> = (0..8 * 8).map(|i| ((i * 5) & 0xFF) as u8).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 8, 8, 1, 8, false, &options, &mut accelerator)
                .expect("encode with metal forward DWT 5/3");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(accelerator.forward_dwt53_attempts(), 1);
        assert_eq!(accelerator.forward_dwt53_dispatches(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_lossless_facade_dispatches_rct_and_dwt_for_wsi_sized_rgb_tile() {
        let mut pixels = Vec::with_capacity(128 * 128 * 3);
        for y in 0..128u32 {
            for x in 0..128u32 {
                pixels.push(((x * 3 + y * 5) & 0xFF) as u8);
                pixels.push(((x * 7 + y * 11) & 0xFF) as u8);
                pixels.push(((x * 13 + y * 17) & 0xFF) as u8);
            }
        }
        let samples =
            J2kLosslessSamples::new(&pixels, 128, 128, 3, 8, false).expect("valid RGB samples");
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let encoded = encode_j2k_lossless_with_accelerator(
            samples,
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::PreferDevice,
                ..J2kLosslessEncodeOptions::default()
            },
            BackendKind::Metal,
            &mut accelerator,
        )
        .expect("Metal-accelerated lossless encode");

        assert_eq!(encoded.backend, BackendKind::Metal);
        assert_eq!(accelerator.forward_rct_dispatches(), 1);
        assert_eq!(accelerator.forward_dwt53_dispatches(), 3);
        assert!(accelerator.tier1_code_block_attempts() > 0);
        assert_eq!(accelerator.packetization_attempts(), 1);
        assert!(accelerator.tier1_code_block_dispatches() > 0);
        assert_eq!(accelerator.packetization_dispatches(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_tier1_uses_one_batched_dispatch_for_multiple_code_blocks() {
        let pixels: Vec<u8> = (0..256 * 256)
            .map(|idx| ((idx * 17 + 3) & 0xFF) as u8)
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            ..EncodeOptions::default()
        };
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 256, 256, 1, 8, false, &options, &mut accelerator)
                .expect("encode with batched Metal classic Tier-1");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert!(accelerator.tier1_code_block_attempts() > 1);
        assert_eq!(accelerator.tier1_code_block_dispatches(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_htj2k_uses_one_batched_dispatch_for_multiple_code_blocks() {
        let pixels: Vec<u8> = (0..256 * 256)
            .map(|idx| ((idx * 23 + 9) & 0xFF) as u8)
            .collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 256, 256, 1, 8, false).expect("valid gray samples");
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let encoded = encode_j2k_lossless_with_accelerator(
            samples,
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::RequireDevice,
                block_coding_mode: J2kBlockCodingMode::HighThroughput,
                ..J2kLosslessEncodeOptions::default()
            },
            BackendKind::Metal,
            &mut accelerator,
        )
        .expect("Metal-accelerated HTJ2K lossless encode");

        assert_eq!(encoded.backend, BackendKind::Metal);
        assert!(accelerator.ht_code_block_attempts() > 1);
        assert_eq!(accelerator.ht_code_block_dispatches(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_htj2k_lossless_facade_dispatches_ht_code_blocks_and_packetization() {
        let pixels: Vec<u8> = (0..64).map(|value| ((value * 13) & 0xFF) as u8).collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).expect("valid gray samples");
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let encoded = encode_j2k_lossless_with_accelerator(
            samples,
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::RequireDevice,
                block_coding_mode: J2kBlockCodingMode::HighThroughput,
                ..J2kLosslessEncodeOptions::default()
            },
            BackendKind::Metal,
            &mut accelerator,
        )
        .expect("Metal-accelerated HTJ2K lossless encode");

        assert_eq!(encoded.backend, BackendKind::Metal);
        assert!(accelerator.ht_code_block_attempts() > 0);
        assert!(accelerator.ht_code_block_dispatches() > 0);
        assert_eq!(accelerator.packetization_attempts(), 1);
        assert_eq!(accelerator.packetization_dispatches(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_tier1_kernel_matches_scalar_oracle() {
        let coeffs: Vec<i32> = (0..64)
            .map(|idx| {
                let value = ((idx * 37 + 11) & 0x1ff) - 255;
                if idx % 5 == 0 {
                    0
                } else {
                    value
                }
            })
            .collect();
        let style = J2kCodeBlockStyle {
            selective_arithmetic_coding_bypass: false,
            reset_context_probabilities: false,
            termination_on_each_pass: false,
            vertically_causal_context: false,
            segmentation_symbols: false,
        };
        let job = signinum_j2k_native::J2kTier1CodeBlockEncodeJob {
            coefficients: &coeffs,
            width: 8,
            height: 8,
            sub_band_type: signinum_j2k_native::J2kSubBandType::HighHigh,
            total_bitplanes: 9,
            style,
        };

        let gpu = compute::encode_classic_tier1_code_block(job).expect("Metal classic encode");
        let cpu = signinum_j2k_native::encode_j2k_code_block_scalar_with_style(
            &coeffs,
            8,
            8,
            signinum_j2k_native::J2kSubBandType::HighHigh,
            9,
            style,
        )
        .expect("scalar classic encode");

        assert_eq!(gpu.data, cpu.data);
        assert_eq!(gpu.segments.len(), cpu.segments.len());
        for (gpu_segment, cpu_segment) in gpu.segments.iter().zip(cpu.segments.iter()) {
            assert_eq!(gpu_segment.data_offset, cpu_segment.data_offset);
            assert_eq!(gpu_segment.data_length, cpu_segment.data_length);
            assert_eq!(gpu_segment.start_coding_pass, cpu_segment.start_coding_pass);
            assert_eq!(gpu_segment.end_coding_pass, cpu_segment.end_coding_pass);
            assert_eq!(gpu_segment.use_arithmetic, cpu_segment.use_arithmetic);
        }
        assert_eq!(gpu.number_of_coding_passes, cpu.number_of_coding_passes);
        assert_eq!(gpu.missing_bit_planes, cpu.missing_bit_planes);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_tier1_kernel_matches_scalar_for_terminated_passes() {
        let coeffs: Vec<i32> = (0..64)
            .map(|idx| {
                let value = ((idx * 43 + 5) & 0x3ff) - 511;
                if idx % 6 == 0 {
                    0
                } else {
                    value
                }
            })
            .collect();
        let style = J2kCodeBlockStyle {
            selective_arithmetic_coding_bypass: false,
            reset_context_probabilities: true,
            termination_on_each_pass: true,
            vertically_causal_context: false,
            segmentation_symbols: true,
        };
        let job = signinum_j2k_native::J2kTier1CodeBlockEncodeJob {
            coefficients: &coeffs,
            width: 8,
            height: 8,
            sub_band_type: signinum_j2k_native::J2kSubBandType::LowHigh,
            total_bitplanes: 10,
            style,
        };

        let gpu =
            compute::encode_classic_tier1_code_block(job).expect("Metal classic terminated encode");
        let cpu = signinum_j2k_native::encode_j2k_code_block_scalar_with_style(
            &coeffs,
            8,
            8,
            signinum_j2k_native::J2kSubBandType::LowHigh,
            10,
            style,
        )
        .expect("scalar classic terminated encode");

        assert_eq!(gpu.data, cpu.data);
        assert_eq!(gpu.segments.len(), cpu.segments.len());
        for (gpu_segment, cpu_segment) in gpu.segments.iter().zip(cpu.segments.iter()) {
            assert_eq!(gpu_segment.data_offset, cpu_segment.data_offset);
            assert_eq!(gpu_segment.data_length, cpu_segment.data_length);
            assert_eq!(gpu_segment.start_coding_pass, cpu_segment.start_coding_pass);
            assert_eq!(gpu_segment.end_coding_pass, cpu_segment.end_coding_pass);
            assert_eq!(gpu_segment.use_arithmetic, cpu_segment.use_arithmetic);
        }
        assert_eq!(gpu.number_of_coding_passes, cpu.number_of_coding_passes);
        assert_eq!(gpu.missing_bit_planes, cpu.missing_bit_planes);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_tier1_kernel_matches_scalar_for_selective_bypass() {
        let coeffs: Vec<i32> = (0..64)
            .map(|idx| {
                let value = ((idx * 61 + 29) & 0x7ff) - 1023;
                if idx % 4 == 0 {
                    0
                } else {
                    value
                }
            })
            .collect();
        let style = J2kCodeBlockStyle {
            selective_arithmetic_coding_bypass: true,
            reset_context_probabilities: false,
            termination_on_each_pass: false,
            vertically_causal_context: false,
            segmentation_symbols: false,
        };
        let job = signinum_j2k_native::J2kTier1CodeBlockEncodeJob {
            coefficients: &coeffs,
            width: 8,
            height: 8,
            sub_band_type: signinum_j2k_native::J2kSubBandType::HighLow,
            total_bitplanes: 11,
            style,
        };

        let gpu =
            compute::encode_classic_tier1_code_block(job).expect("Metal classic bypass encode");
        let cpu = signinum_j2k_native::encode_j2k_code_block_scalar_with_style(
            &coeffs,
            8,
            8,
            signinum_j2k_native::J2kSubBandType::HighLow,
            11,
            style,
        )
        .expect("scalar classic bypass encode");

        assert_eq!(gpu.data, cpu.data);
        assert_eq!(gpu.segments.len(), cpu.segments.len());
        for (gpu_segment, cpu_segment) in gpu.segments.iter().zip(cpu.segments.iter()) {
            assert_eq!(gpu_segment.data_offset, cpu_segment.data_offset);
            assert_eq!(gpu_segment.data_length, cpu_segment.data_length);
            assert_eq!(gpu_segment.start_coding_pass, cpu_segment.start_coding_pass);
            assert_eq!(gpu_segment.end_coding_pass, cpu_segment.end_coding_pass);
            assert_eq!(gpu_segment.use_arithmetic, cpu_segment.use_arithmetic);
        }
        assert_eq!(gpu.number_of_coding_passes, cpu.number_of_coding_passes);
        assert_eq!(gpu.missing_bit_planes, cpu.missing_bit_planes);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_htj2k_cleanup_kernel_matches_scalar_oracle() {
        let coeffs: Vec<i32> = (0..64)
            .map(|idx| {
                let value = ((idx * 19 + 7) & 0xff) - 127;
                if idx % 7 == 0 {
                    0
                } else {
                    value
                }
            })
            .collect();
        let job = signinum_j2k_native::J2kHtCodeBlockEncodeJob {
            coefficients: &coeffs,
            width: 8,
            height: 8,
            total_bitplanes: 8,
        };

        let gpu = compute::encode_ht_cleanup_code_block(job).expect("Metal HT encode");
        let cpu = signinum_j2k_native::encode_ht_code_block_scalar(&coeffs, 8, 8, 8)
            .expect("scalar HT encode");

        assert_eq!(gpu.data, cpu.data);
        assert_eq!(gpu.num_coding_passes, cpu.num_coding_passes);
        assert_eq!(gpu.num_zero_bitplanes, cpu.num_zero_bitplanes);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn ht_simd_prototype_matches_scalar_for_64x64_block() {
        if !compute::ht_simd_prototype_available_for_test()
            .expect("HTJ2K SIMD prototype availability query")
        {
            return;
        }
        let coeffs: Vec<i32> = (0..4096)
            .map(|idx| {
                let value = ((idx * 37 + idx / 11 + 13) & 0xff) - 127;
                if idx % 17 == 0 || idx % 29 == 0 {
                    0
                } else {
                    value
                }
            })
            .collect();
        let job = signinum_j2k_native::J2kHtCodeBlockEncodeJob {
            coefficients: &coeffs,
            width: 64,
            height: 64,
            total_bitplanes: 8,
        };

        let scalar = {
            let _route = compute::force_ht_simd_prototype_route_for_test(false);
            compute::encode_ht_cleanup_code_blocks(&[job])
                .expect("scalar Metal HT encode")
                .remove(0)
        };
        let simd = compute::encode_ht_cleanup_code_blocks_simd_prototype_for_test(&[job])
            .expect("SIMD prototype Metal HT encode")
            .remove(0);

        assert_eq!(simd.data, scalar.data);
        assert_eq!(simd.num_coding_passes, scalar.num_coding_passes);
        assert_eq!(simd.num_zero_bitplanes, scalar.num_zero_bitplanes);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn ht_simd_prototype_matches_scalar_for_mixed_block_batch() {
        if !compute::ht_simd_prototype_available_for_test()
            .expect("HTJ2K SIMD prototype availability query")
        {
            return;
        }
        let all_zero = vec![0; 64];
        let non_square: Vec<i32> = (0..512)
            .map(|idx| {
                let value = (idx * 23 + 9) & 0x7f;
                if idx % 5 == 0 {
                    -value
                } else {
                    value
                }
            })
            .collect();
        let bitplane_edge: Vec<i32> = (0..256)
            .map(|idx| match idx % 4 {
                0 => 255,
                1 => -255,
                2 => 1,
                _ => 0,
            })
            .collect();
        let wide: Vec<i32> = (0..4096)
            .map(|idx| {
                let value = ((idx * 41 + idx / 7 + 3) & 0xff) - 128;
                if idx % 31 == 0 {
                    0
                } else {
                    value
                }
            })
            .collect();
        let jobs = [
            signinum_j2k_native::J2kHtCodeBlockEncodeJob {
                coefficients: &all_zero,
                width: 8,
                height: 8,
                total_bitplanes: 8,
            },
            signinum_j2k_native::J2kHtCodeBlockEncodeJob {
                coefficients: &non_square,
                width: 16,
                height: 32,
                total_bitplanes: 8,
            },
            signinum_j2k_native::J2kHtCodeBlockEncodeJob {
                coefficients: &bitplane_edge,
                width: 16,
                height: 16,
                total_bitplanes: 8,
            },
            signinum_j2k_native::J2kHtCodeBlockEncodeJob {
                coefficients: &wide,
                width: 64,
                height: 64,
                total_bitplanes: 8,
            },
        ];

        let scalar = {
            let _route = compute::force_ht_simd_prototype_route_for_test(false);
            compute::encode_ht_cleanup_code_blocks(&jobs).expect("scalar Metal HT batch")
        };
        let simd = compute::encode_ht_cleanup_code_blocks_simd_prototype_for_test(&jobs)
            .expect("SIMD prototype Metal HT batch");

        assert_eq!(simd.len(), scalar.len());
        for (simd_block, scalar_block) in simd.iter().zip(scalar.iter()) {
            assert_eq!(simd_block.data, scalar_block.data);
            assert_eq!(simd_block.num_coding_passes, scalar_block.num_coding_passes);
            assert_eq!(
                simd_block.num_zero_bitplanes,
                scalar_block.num_zero_bitplanes
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn ht_simd_prototype_length_estimate_matches_scalar() {
        if !compute::ht_simd_prototype_available_for_test()
            .expect("HTJ2K SIMD prototype availability query")
        {
            return;
        }
        let all_zero = vec![0; 64];
        let patterned: Vec<i32> = (0..4096)
            .map(|idx| {
                let value = ((idx * 53 + idx / 13 + 21) & 0xff) - 127;
                if idx % 19 == 0 || idx % 37 == 0 {
                    0
                } else {
                    value
                }
            })
            .collect();
        let jobs = [
            signinum_j2k_native::J2kHtCodeBlockEncodeJob {
                coefficients: &all_zero,
                width: 8,
                height: 8,
                total_bitplanes: 8,
            },
            signinum_j2k_native::J2kHtCodeBlockEncodeJob {
                coefficients: &patterned,
                width: 64,
                height: 64,
                total_bitplanes: 8,
            },
        ];

        let scalar =
            compute::encode_ht_cleanup_code_blocks_with_segment_lengths_for_test(&jobs, false)
                .expect("scalar Metal HT segment lengths");
        let simd =
            compute::encode_ht_cleanup_code_blocks_with_segment_lengths_for_test(&jobs, true)
                .expect("SIMD prototype Metal HT segment lengths");

        assert_eq!(simd.len(), scalar.len());
        for ((simd_block, simd_lengths), (scalar_block, scalar_lengths)) in
            simd.iter().zip(scalar.iter())
        {
            assert_eq!(simd_block.data, scalar_block.data);
            assert_eq!(simd_block.num_coding_passes, scalar_block.num_coding_passes);
            assert_eq!(
                simd_block.num_zero_bitplanes,
                scalar_block.num_zero_bitplanes
            );
            assert_eq!(simd_lengths, scalar_lengths);
            assert_eq!(
                simd_lengths.magnitude_sign + simd_lengths.mel + simd_lengths.vlc,
                u32::try_from(simd_block.data.len()).expect("HT data length fits u32")
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_tier2_packetization_kernel_matches_scalar_oracle() {
        let block0 = [0x12, 0x34, 0x56, 0x78];
        let block1 = [0x9a, 0xbc];
        let code_blocks = vec![
            signinum_j2k_native::J2kPacketizationCodeBlock {
                data: &block0,
                num_coding_passes: 1,
                num_zero_bitplanes: 2,
                previously_included: false,
                l_block: 3,
                block_coding_mode: signinum_j2k_native::J2kPacketizationBlockCodingMode::Classic,
            },
            signinum_j2k_native::J2kPacketizationCodeBlock {
                data: &block1,
                num_coding_passes: 1,
                num_zero_bitplanes: 1,
                previously_included: false,
                l_block: 3,
                block_coding_mode:
                    signinum_j2k_native::J2kPacketizationBlockCodingMode::HighThroughput,
            },
        ];
        let subband = signinum_j2k_native::J2kPacketizationSubband {
            code_blocks,
            num_cbs_x: 2,
            num_cbs_y: 1,
        };
        let resolution = signinum_j2k_native::J2kPacketizationResolution {
            subbands: vec![subband],
        };
        let resolutions = [resolution];
        let packet_descriptors = [signinum_j2k_native::J2kPacketizationPacketDescriptor {
            packet_index: 0,
            state_index: 0,
            layer: 0,
            resolution: 0,
            component: 0,
            precinct: 0,
        }];
        let job = signinum_j2k_native::J2kPacketizationEncodeJob {
            resolution_count: 1,
            num_layers: 1,
            num_components: 1,
            code_block_count: 2,
            progression_order: signinum_j2k_native::J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &packet_descriptors,
            resolutions: &resolutions,
        };

        let gpu = compute::encode_tier2_packetization(job).expect("Metal packet encode");
        let cpu = signinum_j2k_native::encode_j2k_packetization_scalar(job)
            .expect("scalar packet encode");

        assert_eq!(gpu, cpu);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_tier2_packetization_reuses_descriptor_state_across_layers() {
        let block0 = vec![0x11];
        let block1 = vec![0x22];
        let first = signinum_j2k_native::J2kPacketizationResolution {
            subbands: vec![signinum_j2k_native::J2kPacketizationSubband {
                code_blocks: vec![signinum_j2k_native::J2kPacketizationCodeBlock {
                    data: &block0,
                    num_coding_passes: 1,
                    num_zero_bitplanes: 0,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode:
                        signinum_j2k_native::J2kPacketizationBlockCodingMode::Classic,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };
        let second = signinum_j2k_native::J2kPacketizationResolution {
            subbands: vec![signinum_j2k_native::J2kPacketizationSubband {
                code_blocks: vec![signinum_j2k_native::J2kPacketizationCodeBlock {
                    data: &block1,
                    num_coding_passes: 1,
                    num_zero_bitplanes: 0,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode:
                        signinum_j2k_native::J2kPacketizationBlockCodingMode::Classic,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };
        let resolutions = [first, second];
        let packet_descriptors = [
            signinum_j2k_native::J2kPacketizationPacketDescriptor {
                packet_index: 0,
                state_index: 0,
                layer: 0,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
            signinum_j2k_native::J2kPacketizationPacketDescriptor {
                packet_index: 1,
                state_index: 0,
                layer: 1,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
        ];
        let job = signinum_j2k_native::J2kPacketizationEncodeJob {
            resolution_count: 2,
            num_layers: 2,
            num_components: 1,
            code_block_count: 2,
            progression_order: signinum_j2k_native::J2kPacketizationProgressionOrder::Rpcl,
            packet_descriptors: &packet_descriptors,
            resolutions: &resolutions,
        };

        let gpu = compute::encode_tier2_packetization(job).expect("Metal packet encode");
        let cpu = signinum_j2k_native::encode_j2k_packetization_scalar(job)
            .expect("scalar packet encode");

        assert_eq!(gpu, cpu);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_tier2_packetization_honors_explicit_descriptor_order() {
        let block0 = vec![0xA0];
        let block1 = vec![0xB0];
        let first = signinum_j2k_native::J2kPacketizationResolution {
            subbands: vec![signinum_j2k_native::J2kPacketizationSubband {
                code_blocks: vec![signinum_j2k_native::J2kPacketizationCodeBlock {
                    data: &block0,
                    num_coding_passes: 1,
                    num_zero_bitplanes: 0,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode:
                        signinum_j2k_native::J2kPacketizationBlockCodingMode::Classic,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };
        let second = signinum_j2k_native::J2kPacketizationResolution {
            subbands: vec![signinum_j2k_native::J2kPacketizationSubband {
                code_blocks: vec![signinum_j2k_native::J2kPacketizationCodeBlock {
                    data: &block1,
                    num_coding_passes: 1,
                    num_zero_bitplanes: 0,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode:
                        signinum_j2k_native::J2kPacketizationBlockCodingMode::Classic,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };
        let resolutions = [first, second];
        let packet_descriptors = [
            signinum_j2k_native::J2kPacketizationPacketDescriptor {
                packet_index: 1,
                state_index: 1,
                layer: 0,
                resolution: 1,
                component: 0,
                precinct: 0,
            },
            signinum_j2k_native::J2kPacketizationPacketDescriptor {
                packet_index: 0,
                state_index: 0,
                layer: 0,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
        ];
        let job = signinum_j2k_native::J2kPacketizationEncodeJob {
            resolution_count: 2,
            num_layers: 1,
            num_components: 1,
            code_block_count: 2,
            progression_order: signinum_j2k_native::J2kPacketizationProgressionOrder::Rpcl,
            packet_descriptors: &packet_descriptors,
            resolutions: &resolutions,
        };

        let gpu = compute::encode_tier2_packetization(job).expect("Metal packet encode");
        let cpu = signinum_j2k_native::encode_j2k_packetization_scalar(job)
            .expect("scalar packet encode");

        assert_eq!(gpu, cpu);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_forward_dwt53_handles_single_sample_edge_dimensions() {
        for (width, height) in [(1, 8), (8, 1)] {
            let samples: Vec<f32> = (0..width * height)
                .map(|i| {
                    f32::from(
                        u8::try_from((i * 11 + width * 3 + height * 5) & 0xFF)
                            .expect("masked sample fits in u8"),
                    ) - 128.0
                })
                .collect();
            let mut accelerator = MetalEncodeStageAccelerator::default();

            let output = accelerator
                .encode_forward_dwt53(J2kForwardDwt53Job {
                    samples: &samples,
                    width,
                    height,
                    num_levels: 1,
                })
                .expect("metal DWT 5/3 stage")
                .expect("metal DWT 5/3 dispatch");

            assert_eq!(output.ll_width, width.div_ceil(2));
            assert_eq!(output.ll_height, height.div_ceil(2));
            assert_eq!(output.levels.len(), 1);
            assert_eq!(accelerator.forward_dwt53_attempts(), 1);
            assert_eq!(accelerator.forward_dwt53_dispatches(), 1);
        }
    }
}
