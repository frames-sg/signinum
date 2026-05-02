// SPDX-License-Identifier: Apache-2.0

use crate::{
    backend::{BackendKind, BackendRequest},
    context::{CodecContext, DecoderContext},
    error::CodecError,
    pixel::PixelFormat,
    row_sink::RowSink,
    sample::Sample,
    scale::Downscale,
    scratch::ScratchPool,
    types::{DecodeOutcome, Info, Rect},
};

/// Error wrapper used by row-streaming decode when either the codec or the
/// caller-provided row sink can fail.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DecodeRowsError<D, E>
where
    D: core::error::Error + 'static,
    E: core::error::Error + 'static,
{
    #[error(transparent)]
    Decode(D),
    #[error(transparent)]
    Sink(E),
}

/// Common associated types shared by image codecs.
pub trait ImageCodec {
    /// Codec-specific error type.
    type Error: CodecError;
    /// Non-fatal warning type returned in successful decode outcomes.
    type Warning: core::fmt::Debug + core::fmt::Display + Send + Sync + 'static;
    /// Caller-owned scratch pool type used to reuse allocations.
    type Pool: ScratchPool;
}

/// Decoded image data resident on a specific backend.
pub trait DeviceSurface {
    /// Backend that owns or produced the surface.
    fn backend_kind(&self) -> BackendKind;
    /// Surface dimensions in pixels.
    fn dimensions(&self) -> (u32, u32);
    /// Pixel format stored by the surface.
    fn pixel_format(&self) -> PixelFormat;
    /// Number of bytes represented by the surface.
    fn byte_len(&self) -> usize;
}

/// Submitted device decode operation that can be waited on for completion.
pub trait DeviceSubmission {
    /// Completed output type.
    type Output;
    /// Submission or decode error type.
    type Error;

    /// Wait for the submission and return its output.
    fn wait(self) -> Result<Self::Output, Self::Error>;
}

/// Already-completed submission used by synchronous fallback paths.
#[derive(Debug)]
pub struct ReadySubmission<T, E>(Result<T, E>);

impl<T, E> ReadySubmission<T, E> {
    /// Wrap an immediate result as a submission.
    pub fn from_result(result: Result<T, E>) -> Self {
        Self(result)
    }
}

impl<T, E> DeviceSubmission for ReadySubmission<T, E> {
    type Output = T;
    type Error = E;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        self.0
    }
}

/// Borrowed-image decode API for codecs that parse compressed bytes directly.
pub trait ImageDecode<'a>: ImageCodec + Sized + 'a {
    /// Borrowed parse product that can later construct a decoder.
    type View: 'a;

    /// Inspect metadata without decoding pixels.
    fn inspect(input: &'a [u8]) -> Result<Info, Self::Error>;
    /// Parse compressed bytes into a borrowed view.
    fn parse(input: &'a [u8]) -> Result<Self::View, Self::Error>;
    /// Build a decoder from a parsed view.
    fn from_view(view: Self::View) -> Result<Self, Self::Error>;

    /// Decode the full image into caller-owned output.
    fn decode_into(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    /// Decode the full image into caller-owned output with reusable scratch.
    fn decode_into_with_scratch(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    /// Decode a source-coordinate region into caller-owned output.
    fn decode_region_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    /// Decode the full image at reduced resolution into caller-owned output.
    fn decode_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    /// Decode a source-coordinate region at reduced resolution into caller-owned output.
    fn decode_region_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;
}

/// Decode API for implementations that can submit work to a device backend.
pub trait ImageDecodeSubmit<'a>: ImageDecode<'a> {
    /// Mutable session state shared across submissions.
    type Session: Default + Send;
    /// Device surface returned by completed submissions.
    type DeviceSurface: DeviceSurface;
    /// Submission handle type.
    type SubmittedSurface: DeviceSubmission<Output = Self::DeviceSurface, Error = Self::Error>;

    /// Submit full-image decode to the requested backend.
    fn submit_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;

    /// Submit region decode to the requested backend.
    fn submit_region_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;

    /// Submit reduced-resolution decode to the requested backend.
    fn submit_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;

    /// Submit region decode at reduced resolution to the requested backend.
    fn submit_region_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;
}

/// Synchronous device-output decode API.
pub trait ImageDecodeDevice<'a>: ImageDecode<'a> {
    /// Device surface returned by decode calls.
    type DeviceSurface: DeviceSurface;

    /// Decode the full image to the requested backend.
    fn decode_to_device(
        &mut self,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error>;

    /// Decode a source-coordinate region to the requested backend.
    fn decode_region_to_device(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error>;

    /// Decode the full image at reduced resolution to the requested backend.
    fn decode_scaled_to_device(
        &mut self,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error>;

    /// Decode a source-coordinate region at reduced resolution to the requested backend.
    fn decode_region_scaled_to_device(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error>;
}

/// Row-streaming decode API for large images or stripe-oriented callers.
pub trait ImageDecodeRows<'a, S: Sample>: ImageDecode<'a> {
    /// Decode rows into `sink` without requiring one contiguous output buffer.
    fn decode_rows<R: RowSink<S>>(
        &mut self,
        sink: &mut R,
    ) -> Result<DecodeOutcome<Self::Warning>, DecodeRowsError<Self::Error, R::Error>>;
}

/// Stateless tile-batch decode helpers that reuse caller-owned context.
pub trait TileBatchDecode: ImageCodec {
    /// Codec-specific context cached across tiles.
    type Context: CodecContext;

    /// Decode one tile into caller-owned output.
    fn decode_tile<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &'a [u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    /// Decode one tile region into caller-owned output.
    fn decode_tile_region<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &'a [u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    /// Decode one tile at reduced resolution into caller-owned output.
    fn decode_tile_scaled<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &'a [u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    /// Decode one tile region at reduced resolution into caller-owned output.
    #[allow(clippy::too_many_arguments)]
    fn decode_tile_region_scaled<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &'a [u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;
}

/// Tile-batch helpers that return synchronous device surfaces.
pub trait TileBatchDecodeDevice: ImageCodec {
    /// Codec-specific context cached across tiles.
    type Context: CodecContext;
    /// Device surface returned by decode calls.
    type DeviceSurface: DeviceSurface;

    /// Decode one tile to the requested backend.
    fn decode_tile_to_device<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &'a [u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error>;

    /// Decode one tile region to the requested backend.
    fn decode_tile_region_to_device<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &'a [u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error>;

    /// Decode one tile at reduced resolution to the requested backend.
    fn decode_tile_scaled_to_device<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &'a [u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error>;

    /// Decode one tile region at reduced resolution to the requested backend.
    fn decode_tile_region_scaled_to_device<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &'a [u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error>;
}

/// Tile-batch helpers that queue device submissions.
pub trait TileBatchDecodeSubmit: ImageCodec {
    /// Codec-specific context cached across tiles.
    type Context: CodecContext;
    /// Mutable session state shared across submissions.
    type Session: Default + Send;
    /// Device surface returned by completed submissions.
    type DeviceSurface: DeviceSurface;
    /// Submission handle type.
    type SubmittedSurface: DeviceSubmission<Output = Self::DeviceSurface, Error = Self::Error>;

    /// Submit one full tile to the requested backend.
    fn submit_tile_to_device<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &'a [u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;

    /// Submit one tile region to the requested backend.
    fn submit_tile_region_to_device<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &'a [u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;

    /// Submit one tile at reduced resolution to the requested backend.
    fn submit_tile_scaled_to_device<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &'a [u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;

    /// Submit one tile region at reduced resolution to the requested backend.
    #[allow(clippy::too_many_arguments)]
    fn submit_tile_region_scaled_to_device<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &'a [u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;
}

/// Tile payload decompression API for container codecs such as Deflate, Zstd,
/// LZW, and uncompressed data.
pub trait TileDecompress {
    /// Codec-specific error type.
    type Error: CodecError;
    /// Caller-owned scratch pool type.
    type Pool: ScratchPool;

    /// Return the expected decoded size when the compressed payload encodes it.
    fn expected_size(input: &[u8]) -> Result<Option<usize>, Self::Error>;

    /// Decompress `input` into `out`, returning the number of bytes written.
    fn decompress_into(
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
    ) -> Result<usize, Self::Error>;
}
