// SPDX-License-Identifier: Apache-2.0

//! CUDA-facing device-output adapter for `slidecodec-j2k`.
//!
//! This crate intentionally exposes the same backend-selection surface as the
//! Metal adapter, but the `0.1.0` implementation is fallback-only: CPU and
//! auto requests return host-backed surfaces, while explicit CUDA requests
//! report CUDA as unavailable.

#![warn(unreachable_pub)]

mod ht;

use core::convert::Infallible;

use slidecodec_core::{
    BackendKind, BackendRequest, BufferError, CodecError, DecodeOutcome, DeviceSubmission,
    DeviceSurface, Downscale, ImageCodec, ImageDecode, ImageDecodeDevice, ImageDecodeSubmit,
    PixelFormat, ReadySubmission, Rect, TileBatchDecode, TileBatchDecodeDevice,
    TileBatchDecodeSubmit,
};
use slidecodec_j2k::{
    J2kCodec as CpuCodec, J2kContext as CpuJ2kContext, J2kDecoder as CpuDecoder, J2kError,
    J2kScratchPool as CpuJ2kScratchPool, J2kView,
    __private::decode_image_into_with_native_context_and_ht_decoder,
    __private::device_plan::{DeviceDecodePlan, DeviceDecodeRequest},
};
use slidecodec_j2k_native::{
    DecodeSettings as NativeDecodeSettings, DecoderContext as NativeDecoderContext,
    Image as NativeImage,
};

use crate::ht::CudaHtBlockDecoder;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Decode(#[from] J2kError),
    #[error(transparent)]
    Buffer(#[from] BufferError),
    #[error("backend request {request:?} is not supported by slidecodec-j2k-cuda")]
    UnsupportedBackend { request: BackendRequest },
    #[error("CUDA is unavailable on this host")]
    CudaUnavailable,
}

impl CodecError for Error {
    fn is_truncated(&self) -> bool {
        matches!(self, Self::Decode(inner) if inner.is_truncated())
    }

    fn is_not_implemented(&self) -> bool {
        matches!(self, Self::Decode(inner) if inner.is_not_implemented())
    }

    fn is_unsupported(&self) -> bool {
        matches!(
            self,
            Self::UnsupportedBackend { .. } | Self::CudaUnavailable
        ) || matches!(self, Self::Decode(inner) if inner.is_unsupported())
    }

    fn is_buffer_error(&self) -> bool {
        matches!(self, Self::Buffer(_))
            || matches!(self, Self::Decode(inner) if inner.is_buffer_error())
    }
}

#[derive(Debug)]
pub struct Surface {
    backend: BackendKind,
    dimensions: (u32, u32),
    fmt: PixelFormat,
    pitch_bytes: usize,
    bytes: Vec<u8>,
}

impl Surface {
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }

    pub fn as_host_bytes(&self) -> Option<&[u8]> {
        Some(&self.bytes)
    }

    pub fn download_into(&self, out: &mut [u8], stride: usize) -> Result<(), Error> {
        copy_into_output(&self.bytes, self.dimensions, self.fmt, out, stride)
    }
}

impl DeviceSurface for Surface {
    fn backend_kind(&self) -> BackendKind {
        self.backend
    }

    fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    fn pixel_format(&self) -> PixelFormat {
        self.fmt
    }

    fn byte_len(&self) -> usize {
        self.bytes.len()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaSession {
    submissions: u64,
}

impl CudaSession {
    pub fn submissions(&self) -> u64 {
        self.submissions
    }

    fn record_submit(&mut self) {
        self.submissions = self.submissions.saturating_add(1);
    }
}

pub struct J2kDecoder<'a> {
    inner: CpuDecoder<'a>,
    pool: CpuJ2kScratchPool,
    native_image: Option<NativeImage<'a>>,
    native_context: NativeDecoderContext<'a>,
}

impl<'a> J2kDecoder<'a> {
    pub fn new(input: &'a [u8]) -> Result<Self, Error> {
        Ok(Self {
            inner: CpuDecoder::new(input)?,
            pool: CpuJ2kScratchPool::new(),
            native_image: None,
            native_context: NativeDecoderContext::default(),
        })
    }

    fn ensure_native_image(&mut self) -> Result<(), Error> {
        if self.native_image.is_none() {
            self.native_image = Some(
                NativeImage::new(self.inner.bytes(), &NativeDecodeSettings::default())
                    .map_err(|error| J2kError::Backend(error.to_string()))?,
            );
        }
        Ok(())
    }

    fn decode_to_host_surface_with_cuda_ht(&mut self, fmt: PixelFormat) -> Result<Surface, Error> {
        self.ensure_native_image()?;
        let dims = self.inner.info().dimensions;
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        let (Some(image), native_context) = (self.native_image.as_ref(), &mut self.native_context)
        else {
            return Err(Error::Decode(J2kError::Backend(
                "native image cache missing".to_string(),
            )));
        };
        let mut ht_decoder = CudaHtBlockDecoder::default();
        decode_image_into_with_native_context_and_ht_decoder(
            image,
            native_context,
            &mut ht_decoder,
            &mut out,
            stride,
            fmt,
        )?;
        wrap_surface(out, dims, fmt, BackendRequest::Auto)
    }

    fn decode_to_surface_impl(
        &mut self,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        let dims = self.inner.info().dimensions;
        match backend {
            BackendRequest::Cpu => {
                let stride = dims.0 as usize * fmt.bytes_per_pixel();
                let mut out = vec![0u8; stride * dims.1 as usize];
                self.inner
                    .decode_into_with_scratch(&mut self.pool, &mut out, stride, fmt)?;
                wrap_surface(out, dims, fmt, backend)
            }
            BackendRequest::Auto => match self.decode_to_host_surface_with_cuda_ht(fmt) {
                Ok(surface) => Ok(surface),
                Err(error) if error.is_unsupported() || error.is_not_implemented() => {
                    let stride = dims.0 as usize * fmt.bytes_per_pixel();
                    let mut out = vec![0u8; stride * dims.1 as usize];
                    self.inner
                        .decode_into_with_scratch(&mut self.pool, &mut out, stride, fmt)?;
                    wrap_surface(out, dims, fmt, backend)
                }
                Err(error) => Err(error),
            },
            BackendRequest::Cuda => Err(Error::CudaUnavailable),
            BackendRequest::Metal => Err(Error::UnsupportedBackend { request: backend }),
        }
    }

    fn decode_region_to_surface_impl(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        let plan = DeviceDecodePlan::for_image(
            self.inner.info().dimensions,
            DeviceDecodeRequest::Region { roi },
        )?;
        let dims = plan.output_dims();
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        self.inner
            .decode_region_into(&mut self.pool, &mut out, stride, fmt, plan.source_rect())?;
        wrap_surface(out, dims, fmt, backend)
    }

    fn decode_scaled_to_surface_impl(
        &mut self,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        let dims = DeviceDecodePlan::for_image(
            self.inner.info().dimensions,
            DeviceDecodeRequest::Scaled { scale },
        )?
        .output_dims();
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        self.inner
            .decode_scaled_into(&mut self.pool, &mut out, stride, fmt, scale)?;
        wrap_surface(out, dims, fmt, backend)
    }
}

impl ImageCodec for J2kDecoder<'_> {
    type Error = Error;
    type Warning = Infallible;
    type Pool = CpuJ2kScratchPool;
}

impl<'a> ImageDecode<'a> for J2kDecoder<'a> {
    type View = J2kView<'a>;

    fn inspect(input: &'a [u8]) -> Result<slidecodec_core::Info, Self::Error> {
        Ok(CpuDecoder::inspect(input)?)
    }

    fn parse(input: &'a [u8]) -> Result<Self::View, Self::Error> {
        Ok(J2kView::parse(input)?)
    }

    fn from_view(view: Self::View) -> Result<Self, Self::Error> {
        Ok(Self {
            inner: CpuDecoder::from_view(view)?,
            pool: CpuJ2kScratchPool::new(),
            native_image: None,
            native_context: NativeDecoderContext::default(),
        })
    }

    fn decode_into(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        Ok(self.inner.decode_into(out, stride, fmt)?)
    }

    fn decode_into_with_scratch(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        Ok(self
            .inner
            .decode_into_with_scratch(pool, out, stride, fmt)?)
    }

    fn decode_region_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        Ok(self.inner.decode_region_into(pool, out, stride, fmt, roi)?)
    }

    fn decode_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        Ok(self
            .inner
            .decode_scaled_into(pool, out, stride, fmt, scale)?)
    }
}

impl<'a> ImageDecodeDevice<'a> for J2kDecoder<'a> {
    type DeviceSurface = Surface;

    fn decode_to_device(
        &mut self,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error> {
        let mut session = CudaSession::default();
        <Self as ImageDecodeSubmit<'a>>::submit_to_device(self, &mut session, fmt, backend)?.wait()
    }

    fn decode_region_to_device(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error> {
        let mut session = CudaSession::default();
        <Self as ImageDecodeSubmit<'a>>::submit_region_to_device(
            self,
            &mut session,
            fmt,
            roi,
            backend,
        )?
        .wait()
    }

    fn decode_scaled_to_device(
        &mut self,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error> {
        let mut session = CudaSession::default();
        <Self as ImageDecodeSubmit<'a>>::submit_scaled_to_device(
            self,
            &mut session,
            fmt,
            scale,
            backend,
        )?
        .wait()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Codec;

impl ImageCodec for Codec {
    type Error = Error;
    type Warning = Infallible;
    type Pool = CpuJ2kScratchPool;
}

impl Codec {
    fn decode_tile_to_surface_impl(
        ctx: &mut slidecodec_core::DecoderContext<CpuJ2kContext>,
        pool: &mut CpuJ2kScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        let dims = CpuDecoder::inspect(input)?.dimensions;
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        CpuCodec::decode_tile(ctx, pool, input, &mut out, stride, fmt)?;
        wrap_surface(out, dims, fmt, backend)
    }

    fn decode_tile_region_to_surface_impl(
        ctx: &mut slidecodec_core::DecoderContext<CpuJ2kContext>,
        pool: &mut CpuJ2kScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        let dims = DeviceDecodePlan::for_image(
            CpuDecoder::inspect(input)?.dimensions,
            DeviceDecodeRequest::Region { roi },
        )?
        .output_dims();
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        CpuCodec::decode_tile_region(ctx, pool, input, &mut out, stride, fmt, roi)?;
        wrap_surface(out, dims, fmt, backend)
    }

    fn decode_tile_scaled_to_surface_impl(
        ctx: &mut slidecodec_core::DecoderContext<CpuJ2kContext>,
        pool: &mut CpuJ2kScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        let dims = DeviceDecodePlan::for_image(
            CpuDecoder::inspect(input)?.dimensions,
            DeviceDecodeRequest::Scaled { scale },
        )?
        .output_dims();
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        CpuCodec::decode_tile_scaled(ctx, pool, input, &mut out, stride, fmt, scale)?;
        wrap_surface(out, dims, fmt, backend)
    }
}

impl<'a> ImageDecodeSubmit<'a> for J2kDecoder<'a> {
    type Session = CudaSession;
    type DeviceSurface = Surface;
    type SubmittedSurface = ReadySubmission<Surface, Error>;

    fn submit_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        session.record_submit();
        Ok(ReadySubmission::from_result(
            self.decode_to_surface_impl(fmt, backend),
        ))
    }

    fn submit_region_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        session.record_submit();
        Ok(ReadySubmission::from_result(
            self.decode_region_to_surface_impl(fmt, roi, backend),
        ))
    }

    fn submit_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        session.record_submit();
        Ok(ReadySubmission::from_result(
            self.decode_scaled_to_surface_impl(fmt, scale, backend),
        ))
    }
}

impl TileBatchDecodeSubmit for Codec {
    type Context = CpuJ2kContext;
    type Session = CudaSession;
    type DeviceSurface = Surface;
    type SubmittedSurface = ReadySubmission<Surface, Error>;

    fn submit_tile_to_device(
        ctx: &mut slidecodec_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        session.record_submit();
        Ok(ReadySubmission::from_result(
            Self::decode_tile_to_surface_impl(ctx, pool, input, fmt, backend),
        ))
    }

    fn submit_tile_region_to_device(
        ctx: &mut slidecodec_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        session.record_submit();
        Ok(ReadySubmission::from_result(
            Self::decode_tile_region_to_surface_impl(ctx, pool, input, fmt, roi, backend),
        ))
    }

    fn submit_tile_scaled_to_device(
        ctx: &mut slidecodec_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        session.record_submit();
        Ok(ReadySubmission::from_result(
            Self::decode_tile_scaled_to_surface_impl(ctx, pool, input, fmt, scale, backend),
        ))
    }
}

impl TileBatchDecodeDevice for Codec {
    type Context = CpuJ2kContext;
    type DeviceSurface = Surface;

    fn decode_tile_to_device(
        ctx: &mut slidecodec_core::DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error> {
        let mut session = CudaSession::default();
        <Self as TileBatchDecodeSubmit>::submit_tile_to_device(
            ctx,
            &mut session,
            pool,
            input,
            fmt,
            backend,
        )?
        .wait()
    }

    fn decode_tile_region_to_device(
        ctx: &mut slidecodec_core::DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error> {
        let mut session = CudaSession::default();
        <Self as TileBatchDecodeSubmit>::submit_tile_region_to_device(
            ctx,
            &mut session,
            pool,
            input,
            fmt,
            roi,
            backend,
        )?
        .wait()
    }

    fn decode_tile_scaled_to_device(
        ctx: &mut slidecodec_core::DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error> {
        let mut session = CudaSession::default();
        <Self as TileBatchDecodeSubmit>::submit_tile_scaled_to_device(
            ctx,
            &mut session,
            pool,
            input,
            fmt,
            scale,
            backend,
        )?
        .wait()
    }
}

fn wrap_surface(
    bytes: Vec<u8>,
    dimensions: (u32, u32),
    fmt: PixelFormat,
    backend: BackendRequest,
) -> Result<Surface, Error> {
    match backend {
        BackendRequest::Cpu | BackendRequest::Auto => Ok(Surface {
            backend: BackendKind::Cpu,
            dimensions,
            fmt,
            pitch_bytes: dimensions.0 as usize * fmt.bytes_per_pixel(),
            bytes,
        }),
        BackendRequest::Cuda => Err(Error::CudaUnavailable),
        BackendRequest::Metal => Err(Error::UnsupportedBackend { request: backend }),
    }
}

fn copy_into_output(
    src: &[u8],
    dimensions: (u32, u32),
    fmt: PixelFormat,
    out: &mut [u8],
    stride: usize,
) -> Result<(), Error> {
    let row_bytes = dimensions.0 as usize * fmt.bytes_per_pixel();
    if stride < row_bytes {
        return Err(BufferError::StrideTooSmall { row_bytes, stride }.into());
    }
    let required = if dimensions.1 == 0 {
        0
    } else {
        stride * (dimensions.1 as usize - 1) + row_bytes
    };
    if out.len() < required {
        return Err(BufferError::OutputTooSmall {
            required,
            have: out.len(),
        }
        .into());
    }
    for y in 0..dimensions.1 as usize {
        let src_row = &src[y * row_bytes..(y + 1) * row_bytes];
        let dst_start = y * stride;
        out[dst_start..dst_start + row_bytes].copy_from_slice(src_row);
    }
    Ok(())
}

pub use slidecodec_j2k::{J2kContext, J2kScratchPool};
