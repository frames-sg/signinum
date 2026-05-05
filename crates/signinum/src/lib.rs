// SPDX-License-Identifier: Apache-2.0

//! Facade crate for the `signinum` pathology image codecs.
//!
//! Runtime backend requests default to [`BackendRequest::Auto`]. The facade
//! compiles portable CPU codecs plus the Metal adapter by default, then uses
//! device backends for supported workloads when they are compiled and available.
//! CPU is the fallback for `Auto`, not the policy default.
//!
//! # Examples
//!
//! JPEG decode imports:
//!
//! ```no_run
//! use signinum::jpeg::{Decoder, PixelFormat};
//!
//! let bytes = std::fs::read("tile.jpg").unwrap();
//! let mut decoder = Decoder::new(&bytes).unwrap();
//! let info = decoder.info();
//! let stride = info.dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
//! let mut out = vec![0; stride * info.dimensions.1 as usize];
//! decoder.decode_into(&mut out, stride, PixelFormat::Rgb8).unwrap();
//! ```
//!
//! JPEG 2000 lossless encode with the runtime default:
//!
//! ```
//! use signinum::j2k::{encode_j2k_lossless, J2kLosslessEncodeOptions, J2kLosslessSamples};
//! use signinum::BackendRequest;
//!
//! assert_eq!(BackendRequest::default(), BackendRequest::Auto);
//! let pixels = [0u8; 4 * 4];
//! let samples = J2kLosslessSamples::new(&pixels, 4, 4, 1, 8, false).unwrap();
//! let encoded = encode_j2k_lossless(samples, &J2kLosslessEncodeOptions::default()).unwrap();
//! assert!(encoded.codestream.starts_with(&[0xFF, 0x4F]));
//! ```
//!
//! Tile decompression imports:
//!
//! ```
//! use signinum::tilecodec::UncompressedCodec;
//! use signinum::TileDecompress;
//!
//! let mut pool = <UncompressedCodec as TileDecompress>::Pool::default();
//! let mut out = [0u8; 3];
//! let written = UncompressedCodec::decompress_into(&mut pool, &[1, 2, 3], &mut out).unwrap();
//! assert_eq!(written, 3);
//! ```

#![warn(unreachable_pub)]

pub mod core {
    //! Shared codec contracts and backend selection types.

    pub use signinum_core::*;
}

pub mod jpeg {
    //! Baseline JPEG decode APIs.

    pub use signinum_jpeg::*;

    #[cfg(feature = "cuda")]
    pub mod cuda {
        //! CUDA JPEG adapter APIs.

        pub use signinum_jpeg_cuda::*;
    }

    #[cfg(feature = "metal")]
    pub mod metal {
        //! Metal JPEG adapter APIs.

        pub use signinum_jpeg_metal::*;
    }
}

pub mod j2k {
    //! JPEG 2000 inspect, decode, and encode APIs.

    pub use signinum_j2k::{
        adapter, context, encode_j2k_lossless as encode_j2k_lossless_cpu,
        encode_j2k_lossless_with_accelerator, error, j2k_lossless_decomposition_levels, scratch,
        view, BackendKind, BackendRequest, BufferError, CodecError, CompressedPayloadKind,
        CompressedTransferSyntax, DecodeOutcome, DecodeRowsError, DecoderContext, Downscale,
        EncodeBackendPreference, EncodedJ2k, ImageCodec, ImageDecode, ImageDecodeRows,
        J2kBlockCodingMode, J2kCodec, J2kContext, J2kDecoder, J2kEncodeDispatchReport,
        J2kEncodeStageAccelerator, J2kEncodeValidation, J2kError, J2kLosslessEncodeOptions,
        J2kLosslessSamples, J2kProgressionOrder, J2kScratchPool, J2kView, PassthroughCandidate,
        PassthroughDecision, PassthroughRejectReason, PassthroughRequirements, PixelFormat, Rect,
        ReversibleTransform, RowSink, TileBatchDecode,
    };

    #[cfg(feature = "cuda")]
    pub mod cuda {
        //! CUDA JPEG 2000 adapter APIs.

        pub use signinum_j2k_cuda::*;
    }

    #[cfg(feature = "metal")]
    pub mod metal {
        //! Metal JPEG 2000 adapter APIs.

        pub use signinum_j2k_metal::*;
    }

    /// Encode interleaved samples into a raw JPEG 2000 lossless codestream.
    ///
    /// With [`EncodeBackendPreference::Auto`] or
    /// [`EncodeBackendPreference::PreferDevice`], the facade tries compiled
    /// device encode-stage accelerators first and falls back to CPU only when
    /// no device stage dispatches. Device kernel and validation failures are
    /// returned to the caller.
    pub fn encode_j2k_lossless(
        samples: J2kLosslessSamples<'_>,
        options: &J2kLosslessEncodeOptions,
    ) -> Result<EncodedJ2k, J2kError> {
        if options.backend == EncodeBackendPreference::CpuOnly {
            return signinum_j2k::encode_j2k_lossless(samples, options);
        }

        if let Some(encoded) = try_metal_encode(samples, *options)? {
            return Ok(encoded);
        }
        if let Some(encoded) = try_cuda_encode(samples, *options)? {
            return Ok(encoded);
        }

        signinum_j2k::encode_j2k_lossless(samples, options)
    }

    #[cfg(feature = "metal")]
    fn try_metal_encode(
        samples: J2kLosslessSamples<'_>,
        options: J2kLosslessEncodeOptions,
    ) -> Result<Option<EncodedJ2k>, J2kError> {
        let mut accelerator = signinum_j2k_metal::MetalEncodeStageAccelerator::default();
        encode_with_device_accelerator(samples, options, BackendKind::Metal, &mut accelerator)
    }

    #[cfg(not(feature = "metal"))]
    #[allow(clippy::unnecessary_wraps)]
    fn try_metal_encode(
        _samples: J2kLosslessSamples<'_>,
        _options: J2kLosslessEncodeOptions,
    ) -> Result<Option<EncodedJ2k>, J2kError> {
        Ok(None)
    }

    #[cfg(feature = "cuda")]
    fn try_cuda_encode(
        samples: J2kLosslessSamples<'_>,
        options: J2kLosslessEncodeOptions,
    ) -> Result<Option<EncodedJ2k>, J2kError> {
        let mut accelerator = signinum_j2k_cuda::CudaEncodeStageAccelerator::default();
        encode_with_device_accelerator(samples, options, BackendKind::Cuda, &mut accelerator)
    }

    #[cfg(not(feature = "cuda"))]
    #[allow(clippy::unnecessary_wraps)]
    fn try_cuda_encode(
        _samples: J2kLosslessSamples<'_>,
        _options: J2kLosslessEncodeOptions,
    ) -> Result<Option<EncodedJ2k>, J2kError> {
        Ok(None)
    }

    #[cfg_attr(not(any(feature = "metal", feature = "cuda")), allow(dead_code))]
    fn encode_with_device_accelerator(
        samples: J2kLosslessSamples<'_>,
        options: J2kLosslessEncodeOptions,
        backend: BackendKind,
        accelerator: &mut impl J2kEncodeStageAccelerator,
    ) -> Result<Option<EncodedJ2k>, J2kError> {
        let device_options = J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::PreferDevice,
            ..options
        };
        let encoded = signinum_j2k::encode_j2k_lossless_with_accelerator(
            samples,
            &device_options,
            backend,
            accelerator,
        )?;

        Ok((encoded.backend == backend).then_some(encoded))
    }
}

pub mod tilecodec {
    //! Tile decompression codecs for container integrations.

    pub use signinum_tilecodec::*;
}

pub use core::{
    BackendCapabilities, BackendKind, BackendRequest, BufferError, CodecError, DecodeOutcome,
    DecodeRowsError, DecoderContext, DeviceSurface, Downscale, ImageCodec, ImageDecode,
    ImageDecodeDevice, ImageDecodeRows, PixelFormat, Rect, RowSink, TileBatchDecode,
    TileBatchDecodeManyDevice, TileDecompress,
};
pub use core::{
    CompressedPayloadKind, CompressedTransferSyntax, PassthroughCandidate, PassthroughDecision,
    PassthroughRejectReason, PassthroughRequirements,
};
pub use j2k::{
    encode_j2k_lossless, encode_j2k_lossless_with_accelerator, j2k_lossless_decomposition_levels,
    EncodeBackendPreference, EncodedJ2k, J2kBlockCodingMode, J2kCodec, J2kContext, J2kDecoder,
    J2kEncodeDispatchReport, J2kEncodeStageAccelerator, J2kEncodeValidation, J2kError,
    J2kLosslessEncodeOptions, J2kLosslessSamples, J2kProgressionOrder, ReversibleTransform,
};
pub use jpeg::{
    ColorSpace, ColorTransform, DecodeOptions, Decoder as JpegDecoder, JpegCodec, JpegError,
    JpegView,
};
pub use tilecodec::{DeflateCodec, LzwCodec, TileCodecError, UncompressedCodec, ZstdCodec};
