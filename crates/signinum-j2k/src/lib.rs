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
    encode_j2k_lossless, EncodeBackendPreference, EncodedJ2k, J2kLosslessEncodeOptions,
    J2kLosslessSamples, J2kProgressionOrder, ReversibleTransform,
};

pub use signinum_core::{
    BufferError, CodecError, DecodeOutcome, DecodeRowsError, DecoderContext, Downscale, ImageCodec,
    ImageDecode, ImageDecodeRows, PixelFormat, Rect, RowSink, TileBatchDecode,
};

pub(crate) mod parse;
