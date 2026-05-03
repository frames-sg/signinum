// SPDX-License-Identifier: Apache-2.0

//! JPEG 2000 inspect support for signinum.

extern crate alloc;

mod backend;
mod decode;
mod encode;

pub mod context;
pub use context::J2kContext;

pub mod error;
pub use error::J2kError;

pub mod scratch;
pub use scratch::J2kScratchPool;

pub mod adapter;

pub mod view;
pub use view::{J2kCodec, J2kDecoder, J2kView};

pub use encode::{
    encode_j2k_lossless, encode_j2k_lossless_with_accelerator, j2k_lossless_decomposition_levels,
    EncodeBackendPreference, EncodedJ2k, J2kLosslessEncodeOptions, J2kLosslessSamples,
    J2kProgressionOrder, ReversibleTransform,
};

#[doc(hidden)]
pub use signinum_j2k_native::{
    EncodedJ2kCodeBlock, J2kEncodeDispatchReport, J2kEncodeStageAccelerator, J2kForwardDwt53Job,
    J2kForwardDwt53Level, J2kForwardDwt53Output, J2kForwardRctJob, J2kPacketizationBlockCodingMode,
    J2kPacketizationCodeBlock, J2kPacketizationEncodeJob, J2kPacketizationResolution,
    J2kPacketizationSubband, J2kTier1CodeBlockEncodeJob,
};

pub use signinum_core::{
    BackendKind, BackendRequest, BufferError, CodecError, DecodeOutcome, DecodeRowsError,
    DecoderContext, Downscale, ImageCodec, ImageDecode, ImageDecodeRows, PixelFormat, Rect,
    RowSink, TileBatchDecode,
};

pub(crate) mod parse;
