#![no_std]
#![warn(unreachable_pub)]

extern crate alloc;

pub mod backend;
pub mod context;
pub mod error;
pub mod pixel;
pub mod row_sink;
pub mod sample;
pub mod scale;
pub mod scratch;
pub mod traits;
pub mod types;

pub use backend::CpuFeatures;
pub use context::{CacheStats, CodecContext, DecoderContext};
pub use error::{BufferError, CodecError, InputError, NotImplemented, Unsupported};
pub use pixel::{PixelFormat, PixelLayout};
pub use row_sink::RowSink;
pub use sample::{Sample, SampleType};
pub use scale::Downscale;
pub use scratch::ScratchPool;
pub use traits::{
    DecodeRowsError, ImageCodec, ImageDecode, ImageDecodeRows, TileBatchDecode, TileDecompress,
};
pub use types::{Colorspace, DecodeOutcome, Info, Rect, TileLayout, WarningKind};
