// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use crate::compute;
#[cfg(target_os = "macos")]
use metal::Buffer;
#[cfg(target_os = "macos")]
use signinum_core::{BackendKind, DeviceSurface, PixelFormat};
use signinum_j2k::{EncodedJ2k, J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples};
use signinum_j2k_native::{
    EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, J2kEncodeDispatchReport, J2kEncodeStageAccelerator,
    J2kForwardDwt53Job, J2kForwardDwt53Output, J2kForwardRctJob, J2kHtCodeBlockEncodeJob,
    J2kPacketizationEncodeJob, J2kTier1CodeBlockEncodeJob,
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
            compute::encode_forward_rct(job.plane0, job.plane1, job.plane2)
                .map_err(|_| "Metal forward RCT encode kernel failed")?;
            self.forward_rct_dispatches = self.forward_rct_dispatches.saturating_add(1);
            Ok(true)
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
            let output =
                compute::encode_forward_dwt53(job.samples, job.width, job.height, job.num_levels)
                    .map_err(|_| "Metal forward 5/3 DWT encode kernel failed")?;
            self.forward_dwt53_dispatches = self.forward_dwt53_dispatches.saturating_add(1);
            Ok(Some(output))
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
            let encoded = compute::encode_classic_tier1_code_block(job)
                .map_err(|_| "Metal classic Tier-1 encode kernel failed")?;
            self.tier1_code_block_dispatches = self.tier1_code_block_dispatches.saturating_add(1);
            Ok(Some(encoded))
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
            let encoded = compute::encode_classic_tier1_code_blocks(jobs)
                .map_err(|_| "Metal classic Tier-1 encode batch kernel failed")?;
            if !jobs.is_empty() {
                self.tier1_code_block_dispatches =
                    self.tier1_code_block_dispatches.saturating_add(1);
            }
            Ok(Some(encoded))
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
            let encoded = compute::encode_ht_cleanup_code_block(job)
                .map_err(|_| "Metal HTJ2K code-block encode kernel failed")?;
            self.ht_code_block_dispatches = self.ht_code_block_dispatches.saturating_add(1);
            Ok(Some(encoded))
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
            let encoded = compute::encode_ht_cleanup_code_blocks(jobs)
                .map_err(|_| "Metal HTJ2K code-block encode batch kernel failed")?;
            if !jobs.is_empty() {
                self.ht_code_block_dispatches = self.ht_code_block_dispatches.saturating_add(1);
            }
            Ok(Some(encoded))
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
            let encoded = compute::encode_tier2_packetization(job)
                .map_err(|_| "Metal Tier-2 packetization encode kernel failed")?;
            self.packetization_dispatches = self.packetization_dispatches.saturating_add(1);
            Ok(Some(encoded))
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
pub struct MetalLosslessEncodeOutcome {
    pub encoded: EncodedJ2k,
    pub input_copy_used: bool,
    pub input_copy_duration: Duration,
    pub encode_duration: Duration,
    pub validation_duration: Duration,
}

#[cfg(target_os = "macos")]
pub fn encode_lossless_from_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJ2k, crate::Error> {
    encode_lossless_from_metal_buffer_with_report(tile, options, session)
        .map(|outcome| outcome.encoded)
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
pub fn encode_lossless_from_padded_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJ2k, crate::Error> {
    encode_lossless_from_padded_metal_buffer_with_report(tile, options, session)
        .map(|outcome| outcome.encoded)
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
pub fn encode_lossless_from_metal_buffers(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<EncodedJ2k>, crate::Error> {
    encode_lossless_from_metal_buffers_with_report(tiles, options, session).map(|outcomes| {
        outcomes
            .into_iter()
            .map(|outcome| outcome.encoded)
            .collect()
    })
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
pub fn encode_lossless_from_padded_metal_buffers(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<EncodedJ2k>, crate::Error> {
    encode_lossless_from_padded_metal_buffers_with_report(tiles, options, session).map(|outcomes| {
        outcomes
            .into_iter()
            .map(|outcome| outcome.encoded)
            .collect()
    })
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
fn encode_lossless_tiles_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
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
#[derive(Debug, Clone, Copy)]
enum MetalEncodeInputStaging {
    CopyAndPad,
    AlreadyPaddedContiguous,
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
    let validation_started = Instant::now();
    validate_lossless_roundtrip_on_metal_with_session(samples, &encoded.codestream, session)?;
    let validation_duration = validation_started.elapsed();
    Ok(MetalLosslessEncodeOutcome {
        encoded,
        input_copy_used,
        input_copy_duration,
        encode_duration,
        validation_duration,
    })
}

#[cfg(not(target_os = "macos"))]
pub fn encode_lossless_from_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJ2k, crate::Error> {
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
pub fn encode_lossless_from_padded_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJ2k, crate::Error> {
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
pub fn encode_lossless_from_metal_buffers(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<EncodedJ2k>, crate::Error> {
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
pub fn encode_lossless_from_padded_metal_buffers(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<EncodedJ2k>, crate::Error> {
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
    use signinum_core::{BackendKind, PixelFormat};
    use signinum_j2k::{
        encode_j2k_lossless_with_accelerator, EncodeBackendPreference, J2kBlockCodingMode,
        J2kLosslessEncodeOptions, J2kLosslessSamples,
    };
    use signinum_j2k_native::{
        encode_with_accelerator, DecodeSettings, EncodeOptions, Image, J2kCodeBlockStyle,
        J2kEncodeStageAccelerator, J2kForwardDwt53Job,
    };

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
