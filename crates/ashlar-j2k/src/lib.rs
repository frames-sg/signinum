// SPDX-License-Identifier: Apache-2.0

//! JPEG 2000 inspect support for ashlar.

extern crate alloc;

mod backend;
mod decode;

pub mod context;
pub use context::J2kContext;

pub mod error;
pub use error::J2kError;

pub mod scratch;
pub use scratch::J2kScratchPool;

pub mod adapter;

pub mod view;
pub use view::{J2kCodec, J2kDecoder, J2kView};

pub use ashlar_core::{
    BufferError, CodecError, DecodeOutcome, DecodeRowsError, DecoderContext, Downscale, ImageCodec,
    ImageDecode, ImageDecodeRows, PixelFormat, Rect, RowSink, TileBatchDecode,
};

pub(crate) mod parse;
