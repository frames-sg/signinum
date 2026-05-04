// SPDX-License-Identifier: Apache-2.0

//! JPEG decoder optimized for whole-slide images.
//!
//! See the top-level README for project positioning. The primary entry point
//! is [`Decoder`] — start with [`Decoder::inspect`] for header-only parsing.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]
// `missing_docs` remains staged crate-by-crate; see Cargo.toml for rationale.

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("signinum-jpeg currently supports only x86_64 and aarch64 targets");

extern crate alloc;

pub mod info;
pub use info::{
    ColorSpace, ColorTransform, DecodeOptions, Info, McuGeometry, Rect, RestartIndex,
    RestartSegment, SamplingFactors, SofKind,
};
pub use signinum_core::{
    CacheStats, CodecContext, DecodeRowsError, Downscale, ImageCodec, ImageDecode, ImageDecodeRows,
    PixelFormat, PixelLayout, RowSink, Sample, SampleType, TileBatchDecode, TileDecompress,
};

pub mod context;
pub use context::DecoderContext;

pub mod adapter;

pub mod error;
pub use error::{
    BuilderConflictReason, HuffmanFailure, JpegError, MarkerKind, TableKind, UnsupportedReason,
    Warning,
};

pub(crate) mod parse;

pub(crate) mod entropy;

pub(crate) mod idct;

pub(crate) mod internal;

pub(crate) mod color;

pub(crate) mod backend;

pub(crate) mod output;

pub mod encoder;
pub use encoder::{
    encode_jpeg_baseline, EncodedJpeg, JpegBackend, JpegEncodeError, JpegEncodeOptions,
    JpegSamples, JpegSubsampling,
};

pub mod decoder;
pub use decoder::{
    decode_tile_into, decode_tile_into_in_context, decode_tile_region_into_in_context,
    decode_tile_region_scaled_into_in_context, decode_tile_scaled_into_in_context,
    ComponentRowWriter, DecodeOutcome, Decoder, JpegView,
};

pub use internal::scratch::ScratchPool;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JpegCodec;

#[doc(hidden)]
pub mod bench_support;
