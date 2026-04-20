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

pub trait ImageCodec {
    type Error: CodecError;
    type Warning: core::fmt::Debug + core::fmt::Display + Send + Sync + 'static;
    type Pool: ScratchPool;
}

pub trait DeviceSurface {
    fn backend_kind(&self) -> BackendKind;
    fn dimensions(&self) -> (u32, u32);
    fn pixel_format(&self) -> PixelFormat;
    fn byte_len(&self) -> usize;
}

pub trait DeviceSubmission {
    type Output;
    type Error;

    fn wait(self) -> Result<Self::Output, Self::Error>;
}

#[derive(Debug)]
pub struct ReadySubmission<T, E>(Result<T, E>);

impl<T, E> ReadySubmission<T, E> {
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

pub trait ImageDecode<'a>: ImageCodec + Sized + 'a {
    type View: 'a;

    fn inspect(input: &'a [u8]) -> Result<Info, Self::Error>;
    fn parse(input: &'a [u8]) -> Result<Self::View, Self::Error>;
    fn from_view(view: Self::View) -> Result<Self, Self::Error>;

    fn decode_into(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    fn decode_into_with_scratch(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    fn decode_region_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    fn decode_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;
}

pub trait ImageDecodeSubmit<'a>: ImageDecode<'a> {
    type Session: Default + Send;
    type DeviceSurface: DeviceSurface;
    type SubmittedSurface: DeviceSubmission<Output = Self::DeviceSurface, Error = Self::Error>;

    fn submit_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;

    fn submit_region_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;

    fn submit_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;
}

pub trait ImageDecodeDevice<'a>: ImageDecode<'a> {
    type DeviceSurface: DeviceSurface;

    fn decode_to_device(
        &mut self,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error>;

    fn decode_region_to_device(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error>;

    fn decode_scaled_to_device(
        &mut self,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error>;
}

pub trait ImageDecodeRows<'a, S: Sample>: ImageDecode<'a> {
    fn decode_rows<R: RowSink<S>>(
        &mut self,
        sink: &mut R,
    ) -> Result<DecodeOutcome<Self::Warning>, DecodeRowsError<Self::Error, R::Error>>;
}

pub trait TileBatchDecode: ImageCodec {
    type Context: CodecContext;

    fn decode_tile<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &'a [u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    fn decode_tile_region<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &'a [u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    fn decode_tile_scaled<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &'a [u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;
}

pub trait TileBatchDecodeDevice: ImageCodec {
    type Context: CodecContext;
    type DeviceSurface: DeviceSurface;

    fn decode_tile_to_device<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &'a [u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error>;

    fn decode_tile_region_to_device<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &'a [u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error>;

    fn decode_tile_scaled_to_device<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &'a [u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error>;
}

pub trait TileBatchDecodeSubmit: ImageCodec {
    type Context: CodecContext;
    type Session: Default + Send;
    type DeviceSurface: DeviceSurface;
    type SubmittedSurface: DeviceSubmission<Output = Self::DeviceSurface, Error = Self::Error>;

    fn submit_tile_to_device<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &'a [u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;

    fn submit_tile_region_to_device<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &'a [u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;

    fn submit_tile_scaled_to_device<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &'a [u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;
}

pub trait TileDecompress {
    type Error: CodecError;
    type Pool: ScratchPool;

    fn expected_size(input: &[u8]) -> Result<Option<usize>, Self::Error>;

    fn decompress_into(
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
    ) -> Result<usize, Self::Error>;
}
