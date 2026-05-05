//! Shared traits and value types for the `signinum` workspace.
//!
//! Codec crates use this crate to expose common pixel formats, decode
//! outcomes, row sinks, caller-owned scratch pools, and CPU/GPU backend
//! selection contracts without depending on each other.

#![no_std]
#![warn(unreachable_pub)]

extern crate alloc;

pub mod backend;
pub mod context;
pub mod error;
pub mod passthrough;
pub mod pixel;
pub mod row_sink;
pub mod sample;
pub mod scale;
pub mod scratch;
pub mod traits;
pub mod types;

pub use backend::{BackendCapabilities, BackendKind, BackendRequest, CpuFeatures};
pub use context::{CacheStats, CodecContext, DecoderContext};
pub use error::{BufferError, CodecError, InputError, NotImplemented, Unsupported};
pub use passthrough::{
    CompressedPayloadKind, CompressedTransferSyntax, PassthroughCandidate, PassthroughDecision,
    PassthroughRejectReason, PassthroughRequirements,
};
pub use pixel::{PixelFormat, PixelLayout};
pub use row_sink::RowSink;
pub use sample::{Sample, SampleType};
pub use scale::Downscale;
pub use scratch::ScratchPool;
pub use traits::{
    DecodeRowsError, DeviceSubmission, DeviceSurface, ImageCodec, ImageDecode, ImageDecodeDevice,
    ImageDecodeRows, ImageDecodeSubmit, ReadySubmission, TileBatchDecode, TileBatchDecodeDevice,
    TileBatchDecodeManyDevice, TileBatchDecodeSubmit, TileDecompress,
};
pub use types::{CodedUnitLayout, Colorspace, DecodeOutcome, Info, Rect, TileLayout, WarningKind};
