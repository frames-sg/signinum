// SPDX-License-Identifier: Apache-2.0

use alloc::vec::Vec;

use signinum_core::{BackendKind, Unsupported};
use signinum_j2k_native::{DecodeSettings, EncodeOptions, Image};

use crate::J2kError;

/// Backend preference for JPEG 2000 lossless encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EncodeBackendPreference {
    /// Pick the fastest safe backend. Currently resolves to CPU because no
    /// device encoder is exposed by signinum yet.
    #[default]
    Auto,
    /// Require the pure Rust CPU encoder.
    CpuOnly,
    /// Prefer a device encoder, but fall back to CPU when unavailable.
    PreferDevice,
    /// Require a device encoder and fail if unavailable or unsupported.
    RequireDevice,
}

/// Supported JPEG 2000 progression orders for the lossless encode facade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum J2kProgressionOrder {
    /// Layer-resolution-component-position progression.
    #[default]
    Lrcp,
}

/// Reversible transform profile for lossless JPEG 2000 output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ReversibleTransform {
    /// Reversible color transform with 5/3 wavelet transform.
    #[default]
    Rct53,
}

/// Options controlling JPEG 2000 lossless encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct J2kLosslessEncodeOptions {
    pub backend: EncodeBackendPreference,
    pub progression: J2kProgressionOrder,
    pub reversible_transform: ReversibleTransform,
}

impl Default for J2kLosslessEncodeOptions {
    fn default() -> Self {
        Self {
            backend: EncodeBackendPreference::Auto,
            progression: J2kProgressionOrder::Lrcp,
            reversible_transform: ReversibleTransform::Rct53,
        }
    }
}

/// Borrowed interleaved samples and image geometry for lossless encoding.
#[derive(Debug, Clone, Copy)]
pub struct J2kLosslessSamples<'a> {
    pub data: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub components: u8,
    pub bit_depth: u8,
    pub signed: bool,
}

impl<'a> J2kLosslessSamples<'a> {
    /// Validate and construct a sample descriptor.
    pub fn new(
        data: &'a [u8],
        width: u32,
        height: u32,
        components: u8,
        bit_depth: u8,
        signed: bool,
    ) -> Result<Self, J2kError> {
        if width == 0 || height == 0 {
            return Err(J2kError::Backend("invalid dimensions".to_string()));
        }
        if !matches!(components, 1 | 3) {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JPEG 2000 lossless encode supports only grayscale or RGB samples",
            }));
        }
        if bit_depth == 0 || bit_depth > 16 {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JPEG 2000 lossless encode supports 1-16 bits per sample",
            }));
        }
        let bytes_per_sample = if bit_depth <= 8 { 1usize } else { 2usize };
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|px| px.checked_mul(components as usize))
            .and_then(|samples| samples.checked_mul(bytes_per_sample))
            .ok_or(J2kError::DimensionOverflow { width, height })?;
        if data.len() != expected {
            return Err(J2kError::Backend(format!(
                "pixel data too short: expected {expected} bytes, got {}",
                data.len()
            )));
        }
        Ok(Self {
            data,
            width,
            height,
            components,
            bit_depth,
            signed,
        })
    }
}

/// Encoded JPEG 2000 lossless codestream and encode metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedJ2k {
    pub codestream: Vec<u8>,
    pub backend: BackendKind,
    pub width: u32,
    pub height: u32,
    pub components: u8,
    pub bit_depth: u8,
    pub signed: bool,
}

/// Encode interleaved samples into a raw JPEG 2000 lossless codestream.
pub fn encode_j2k_lossless(
    samples: J2kLosslessSamples<'_>,
    options: &J2kLosslessEncodeOptions,
) -> Result<EncodedJ2k, J2kError> {
    let backend = resolve_encode_backend(options.backend)?;
    let codestream = encode_cpu(samples)?;
    validate_lossless_roundtrip(samples, &codestream)?;
    Ok(EncodedJ2k {
        codestream,
        backend,
        width: samples.width,
        height: samples.height,
        components: samples.components,
        bit_depth: samples.bit_depth,
        signed: samples.signed,
    })
}

fn resolve_encode_backend(preference: EncodeBackendPreference) -> Result<BackendKind, J2kError> {
    match preference {
        EncodeBackendPreference::Auto
        | EncodeBackendPreference::CpuOnly
        | EncodeBackendPreference::PreferDevice => Ok(BackendKind::Cpu),
        EncodeBackendPreference::RequireDevice => Err(J2kError::Unsupported(Unsupported {
            what: "device JPEG 2000 lossless encode backend is unavailable",
        })),
    }
}

fn encode_cpu(samples: J2kLosslessSamples<'_>) -> Result<Vec<u8>, J2kError> {
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 0,
        ..EncodeOptions::default()
    };
    signinum_j2k_native::encode(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &options,
    )
    .map_err(|err| J2kError::Backend(format!("JPEG 2000 lossless encode failed: {err}")))
}

fn validate_lossless_roundtrip(
    samples: J2kLosslessSamples<'_>,
    codestream: &[u8],
) -> Result<(), J2kError> {
    let decoded = Image::new(codestream, &DecodeSettings::default())
        .map_err(|err| J2kError::Backend(format!("encoded codestream validation failed: {err}")))?
        .decode_native()
        .map_err(|err| J2kError::Backend(format!("encoded codestream validation failed: {err}")))?;

    if decoded.width != samples.width
        || decoded.height != samples.height
        || decoded.num_components != samples.components
        || decoded.bit_depth != samples.bit_depth
    {
        return Err(J2kError::Backend(
            "JPEG 2000 lossless encode failed round-trip geometry validation".to_string(),
        ));
    }
    if decoded.data != samples.data {
        let mismatch = decoded
            .data
            .iter()
            .zip(samples.data.iter())
            .position(|(actual, expected)| actual != expected);
        return Err(J2kError::Backend(match mismatch {
            Some(index) => format!(
                "JPEG 2000 lossless encode failed round-trip validation at byte {index}: expected {}, got {}",
                samples.data[index], decoded.data[index]
            ),
            None => format!(
                "JPEG 2000 lossless encode failed round-trip validation: expected {} bytes, got {} bytes",
                samples.data.len(),
                decoded.data.len()
            ),
        }));
    }
    Ok(())
}
