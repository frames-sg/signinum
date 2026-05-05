// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use crate::compute;
#[cfg(target_os = "macos")]
use metal::Buffer;
use signinum_core::DeviceSubmission;
#[cfg(target_os = "macos")]
use signinum_core::{BackendKind, DeviceSurface, PixelFormat};
use signinum_j2k::{
    EncodeBackendPreference, EncodedJ2k, J2kBlockCodingMode, J2kEncodeValidation,
    J2kLosslessEncodeOptions, J2kLosslessSamples, J2kProgressionOrder,
};
use signinum_j2k_native::{
    EncodeProgressionOrder, EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, J2kEncodeDispatchReport,
    J2kEncodeStageAccelerator, J2kForwardDwt53Job, J2kForwardDwt53Output, J2kForwardRctJob,
    J2kHtCodeBlockEncodeJob, J2kPacketizationEncodeJob, J2kPacketizationPacketDescriptor,
    J2kSubBandType, J2kTier1CodeBlockEncodeJob,
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
            } => {
                let mut accelerator = MetalEncodeStageAccelerator::default();
                let mut encoded = Vec::with_capacity(tiles.len());
                for tile in &tiles {
                    encoded.push(
                        encode_lossless_tile_with_report(
                            tile.as_tile(),
                            options,
                            &session,
                            staging,
                            &mut accelerator,
                        )?
                        .encoded,
                    );
                }
                Ok(encoded)
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
    submit_lossless_tiles(
        tiles,
        *options,
        session,
        MetalEncodeInputStaging::CopyAndPad,
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
    encode_lossless_tiles_to_metal_buffer_with_report(
        tiles,
        *options,
        session,
        MetalEncodeInputStaging::CopyAndPad,
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
    submit_lossless_tiles(
        tiles,
        *options,
        session,
        MetalEncodeInputStaging::AlreadyPaddedContiguous,
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
    encode_lossless_tiles_to_metal_buffer_with_report(
        tiles,
        *options,
        session,
        MetalEncodeInputStaging::AlreadyPaddedContiguous,
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
        if let Some(outcomes) = try_encode_resident_lossless_tiles_to_metal_buffer_with_report(
            tiles, options, session, staging,
        )? {
            return outcomes
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
    let mut outcomes = Vec::with_capacity(tiles.len());
    for &tile in tiles {
        outcomes.push(encode_lossless_tile_with_report(
            tile,
            options,
            session,
            staging,
            &mut accelerator,
        )?);
    }
    Ok(outcomes)
}

#[cfg(target_os = "macos")]
fn encode_lossless_tiles_to_metal_buffer_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<Vec<MetalLosslessBufferEncodeOutcome>, crate::Error> {
    if options.backend != EncodeBackendPreference::CpuOnly {
        if let Some(outcomes) = try_encode_resident_lossless_tiles_to_metal_buffer_with_report(
            tiles, options, session, staging,
        )? {
            return Ok(outcomes);
        }
    }

    let mut outcomes = Vec::with_capacity(tiles.len());
    for &tile in tiles {
        outcomes.push(encode_lossless_tile_to_metal_buffer_with_report(
            tile, options, session, staging,
        )?);
    }
    Ok(outcomes)
}

#[cfg(target_os = "macos")]
fn try_encode_resident_lossless_tiles_to_metal_buffer_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<Option<Vec<MetalLosslessBufferEncodeOutcome>>, crate::Error> {
    if tiles.is_empty() {
        return Ok(Some(Vec::new()));
    }

    let encode_started = Instant::now();
    let mut prepared = Vec::with_capacity(tiles.len());
    for &tile in tiles {
        validate_metal_encode_tile(tile)?;
        let Some(item) = prepare_resident_lossless_buffer_encode(tile, options, session, staging)?
        else {
            return Ok(None);
        };
        prepared.push(item);
    }

    let mut tier1_items = Vec::with_capacity(prepared.len());
    for item in prepared {
        tier1_items.push(encode_prepared_resident_lossless_tier1(item, session)?);
    }

    let mut submitted = Vec::with_capacity(tier1_items.len());
    for (metadata, tier1) in tier1_items {
        submitted.push(submit_resident_lossless_buffer_encode(
            metadata, &tier1, session,
        )?);
    }
    let submission_duration = encode_started.elapsed();
    let submission_share = duration_share(submission_duration, submitted.len());

    let mut finished = Vec::with_capacity(submitted.len());
    for item in submitted {
        let mut item = wait_submitted_resident_lossless_buffer_encode(item)?;
        item.encode_duration = item.encode_duration.saturating_add(submission_share);
        finished.push(item);
    }

    let mut outcomes = Vec::with_capacity(finished.len());
    for item in finished {
        outcomes.push(validate_finished_resident_lossless_buffer_encode(
            item, options, session,
        )?);
    }
    Ok(Some(outcomes))
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
        .map_or(levels, |max_levels| levels.min(max_levels))
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
fn prepare_resident_lossless_buffer_encode(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<Option<PreparedResidentLosslessBufferEncode>, crate::Error> {
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

    Ok(Some(PreparedResidentLosslessBufferEncode {
        metadata: ResidentLosslessBufferEncodeMetadata {
            tile: OwnedMetalLosslessEncodeTile::from_tile(tile),
            components,
            bit_depth,
            bytes_per_pixel,
            plan,
            packet_descriptors,
            packetization_resolutions,
        },
        prepared,
    }))
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
    let metadata = submitted.metadata;
    let encoded = MetalEncodedJ2k {
        codestream_buffer: codestream.buffer,
        byte_offset: 0,
        byte_len: codestream.byte_len,
        capacity: codestream.capacity,
        width: metadata.tile.output_width,
        height: metadata.tile.output_height,
        components: metadata.components,
        bit_depth: metadata.bit_depth,
        signed: false,
    };

    Ok(FinishedResidentLosslessBufferEncode {
        metadata,
        encoded,
        encode_duration,
        gpu_duration: codestream.gpu_duration,
    })
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
        byte_offset: 0,
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
pub fn encode_lossless_from_padded_metal_buffers_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
    let _ = (tiles, options, session);
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
    use metal::Buffer;
    use signinum_core::DeviceSubmission;
    #[cfg(target_os = "macos")]
    use signinum_core::{BackendKind, PixelFormat};
    #[cfg(target_os = "macos")]
    use signinum_j2k::{
        encode_j2k_lossless_with_accelerator, EncodeBackendPreference, J2kBlockCodingMode,
        J2kLosslessSamples, J2kProgressionOrder,
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
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
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
