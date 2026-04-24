// SPDX-License-Identifier: Apache-2.0

//! JPEG 2000 inspect support for slidecodec.

extern crate alloc;

mod backend;
mod decode;

pub mod context;
pub use context::J2kContext;

pub mod error;
pub use error::J2kError;

pub mod scratch;
pub use scratch::J2kScratchPool;

pub mod view;
pub use view::{J2kCodec, J2kDecoder, J2kView};

pub use slidecodec_core::{
    BufferError, CodecError, DecodeOutcome, DecodeRowsError, DecoderContext, Downscale, ImageCodec,
    ImageDecode, ImageDecodeRows, PixelFormat, Rect, RowSink, TileBatchDecode,
};

#[doc(hidden)]
pub mod __private;

pub(crate) mod parse;
