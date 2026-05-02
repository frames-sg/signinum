// SPDX-License-Identifier: Apache-2.0

//! CUDA-facing device-output adapter for `signinum-jpeg`.
//!
//! This crate intentionally exposes the same backend-selection surface as the
//! Metal adapter. CPU and auto requests return host-backed surfaces, while
//! explicit CUDA requests upload decoded output into CUDA device memory when
//! the `cuda-runtime` feature and a CUDA driver are available.

#![warn(unreachable_pub)]

use signinum_core::{
    BackendKind, BackendRequest, BufferError, CodecError, DecodeOutcome, DeviceSubmission,
    DeviceSurface, Downscale, ImageCodec, ImageDecode, ImageDecodeDevice, ImageDecodeSubmit,
    PixelFormat, ReadySubmission, Rect, TileBatchDecodeDevice, TileBatchDecodeSubmit,
};
#[cfg(feature = "cuda-runtime")]
use signinum_cuda_runtime::{CudaContext, CudaDeviceBuffer, CudaError};
use signinum_jpeg::{
    decode_tile_into_in_context, decode_tile_region_into_in_context,
    decode_tile_region_scaled_into_in_context, decode_tile_scaled_into_in_context,
    ColorSpace as JpegColorSpace, DecodeOutcome as JpegDecodeOutcome, Decoder as CpuDecoder,
    DecoderContext as CpuDecoderContext, JpegError, JpegView, Rect as JpegRect,
    ScratchPool as CpuScratchPool, Warning as CpuWarning,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Decode(#[from] JpegError),
    #[error(transparent)]
    Buffer(#[from] BufferError),
    #[error("backend request {request:?} is not supported by signinum-jpeg-cuda")]
    UnsupportedBackend { request: BackendRequest },
    #[error("CUDA is unavailable on this host")]
    CudaUnavailable,
    #[cfg(feature = "cuda-runtime")]
    #[error("CUDA runtime error: {message}")]
    CudaRuntime { message: String },
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
enum Storage {
    Host(Vec<u8>),
    #[cfg(feature = "cuda-runtime")]
    Cuda(CudaDeviceBuffer),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaSurfaceStats {
    kernel_dispatches: usize,
}

impl CudaSurfaceStats {
    pub fn kernel_dispatches(self) -> usize {
        self.kernel_dispatches
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaSurface<'a> {
    #[cfg(feature = "cuda-runtime")]
    buffer: &'a CudaDeviceBuffer,
    #[cfg(not(feature = "cuda-runtime"))]
    _marker: core::marker::PhantomData<&'a ()>,
    stats: CudaSurfaceStats,
}

impl CudaSurface<'_> {
    pub fn device_ptr(&self) -> u64 {
        #[cfg(feature = "cuda-runtime")]
        {
            self.buffer.device_ptr()
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            unreachable!("CudaSurface cannot be constructed without cuda-runtime support")
        }
    }

    pub fn stats(&self) -> CudaSurfaceStats {
        self.stats
    }
}

#[derive(Debug)]
pub struct Surface {
    backend: BackendKind,
    dimensions: (u32, u32),
    fmt: PixelFormat,
    pitch_bytes: usize,
    stats: CudaSurfaceStats,
    storage: Storage,
}

impl Surface {
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }

    pub fn as_host_bytes(&self) -> Option<&[u8]> {
        match &self.storage {
            Storage::Host(bytes) => Some(bytes),
            #[cfg(feature = "cuda-runtime")]
            Storage::Cuda(_) => None,
        }
    }

    pub fn download_into(&self, out: &mut [u8], stride: usize) -> Result<(), Error> {
        match &self.storage {
            Storage::Host(bytes) => copy_into_output(bytes, self.dimensions, self.fmt, out, stride),
            #[cfg(feature = "cuda-runtime")]
            Storage::Cuda(buffer) => {
                let mut tight = vec![0u8; self.byte_len()];
                buffer.copy_to_host(&mut tight).map_err(cuda_error)?;
                copy_into_output(&tight, self.dimensions, self.fmt, out, stride)
            }
        }
    }

    pub fn cuda_surface(&self) -> Option<CudaSurface<'_>> {
        #[cfg(feature = "cuda-runtime")]
        match &self.storage {
            Storage::Cuda(buffer) => Some(CudaSurface {
                buffer,
                stats: self.stats,
            }),
            Storage::Host(_) => None,
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            let _ = self.stats;
            None
        }
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
        self.pitch_bytes * self.dimensions.1 as usize
    }
}

#[derive(Clone, Default)]
pub struct CudaSession {
    submissions: u64,
    #[cfg(feature = "cuda-runtime")]
    context: Option<CudaContext>,
}

impl CudaSession {
    pub fn submissions(&self) -> u64 {
        self.submissions
    }

    #[cfg(feature = "cuda-runtime")]
    pub fn is_runtime_initialized(&self) -> bool {
        self.context.is_some()
    }

    fn record_submit(&mut self) {
        self.submissions = self.submissions.saturating_add(1);
    }

    #[cfg(feature = "cuda-runtime")]
    fn cuda_context(&mut self) -> Result<CudaContext, Error> {
        if self.context.is_none() {
            self.context = Some(CudaContext::system_default().map_err(cuda_error)?);
        }
        self.context.clone().ok_or(Error::CudaUnavailable)
    }
}

impl std::fmt::Debug for CudaSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("CudaSession");
        debug.field("submissions", &self.submissions);
        #[cfg(feature = "cuda-runtime")]
        debug.field("runtime_initialized", &self.is_runtime_initialized());
        debug.finish_non_exhaustive()
    }
}

pub struct Decoder<'a> {
    inner: CpuDecoder<'a>,
}

impl<'a> Decoder<'a> {
    pub fn new(input: &'a [u8]) -> Result<Self, Error> {
        Ok(Self {
            inner: CpuDecoder::new(input)?,
        })
    }

    fn decode_to_surface_impl(
        &mut self,
        session: &mut CudaSession,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        let (bytes, _outcome) = self.inner.decode(fmt)?;
        wrap_surface(bytes, self.inner.info().dimensions, fmt, backend, session)
    }

    fn decode_region_to_surface_impl(
        &mut self,
        session: &mut CudaSession,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        let (bytes, outcome) = self.inner.decode_region(fmt, to_jpeg_rect(roi))?;
        wrap_surface(
            bytes,
            (outcome.decoded.w, outcome.decoded.h),
            fmt,
            backend,
            session,
        )
    }

    fn decode_scaled_to_surface_impl(
        &mut self,
        session: &mut CudaSession,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        let (bytes, outcome) = self.inner.decode_scaled(fmt, scale)?;
        wrap_surface(
            bytes,
            (outcome.decoded.w, outcome.decoded.h),
            fmt,
            backend,
            session,
        )
    }

    fn decode_region_scaled_to_surface_impl(
        &mut self,
        session: &mut CudaSession,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        let (bytes, outcome) = self
            .inner
            .decode_region_scaled(fmt, to_jpeg_rect(roi), scale)?;
        wrap_surface(
            bytes,
            (outcome.decoded.w, outcome.decoded.h),
            fmt,
            backend,
            session,
        )
    }
}

impl ImageCodec for Decoder<'_> {
    type Error = Error;
    type Warning = CpuWarning;
    type Pool = CpuScratchPool;
}

impl<'a> ImageDecode<'a> for Decoder<'a> {
    type View = JpegView<'a>;

    fn inspect(input: &'a [u8]) -> Result<signinum_core::Info, Self::Error> {
        Ok(convert_info(&CpuDecoder::inspect(input)?))
    }

    fn parse(input: &'a [u8]) -> Result<Self::View, Self::Error> {
        Ok(JpegView::parse(input)?)
    }

    fn from_view(view: Self::View) -> Result<Self, Self::Error> {
        Ok(Self {
            inner: CpuDecoder::from_view(view)?,
        })
    }

    fn decode_into(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        Ok(convert_outcome(self.inner.decode_into(out, stride, fmt)?))
    }

    fn decode_into_with_scratch(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        Ok(convert_outcome(
            self.inner
                .decode_into_with_scratch(pool, out, stride, fmt)?,
        ))
    }

    fn decode_region_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        Ok(convert_outcome(
            self.inner.decode_region_into_with_scratch(
                pool,
                out,
                stride,
                fmt,
                to_jpeg_rect(roi),
            )?,
        ))
    }

    fn decode_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        Ok(convert_outcome(
            self.inner
                .decode_scaled_into_with_scratch(pool, out, stride, fmt, scale)?,
        ))
    }

    fn decode_region_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        Ok(convert_outcome(
            self.inner.decode_region_scaled_into_with_scratch(
                pool,
                out,
                stride,
                fmt,
                to_jpeg_rect(roi),
                scale,
            )?,
        ))
    }
}

impl<'a> ImageDecodeDevice<'a> for Decoder<'a> {
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

    fn decode_region_scaled_to_device(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error> {
        let mut session = CudaSession::default();
        <Self as ImageDecodeSubmit<'a>>::submit_region_scaled_to_device(
            self,
            &mut session,
            fmt,
            roi,
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
    type Warning = CpuWarning;
    type Pool = CpuScratchPool;
}

impl Codec {
    fn decode_tile_to_surface_impl(
        ctx: &mut signinum_core::DecoderContext<CpuDecoderContext>,
        session: &mut CudaSession,
        pool: &mut CpuScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        let dims = CpuDecoder::inspect(input)?.dimensions;
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        decode_tile_into_in_context(input, ctx.codec_mut(), pool, &mut out, stride, fmt)?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    fn decode_tile_region_to_surface_impl(
        ctx: &mut signinum_core::DecoderContext<CpuDecoderContext>,
        session: &mut CudaSession,
        pool: &mut CpuScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        let dims = (roi.w, roi.h);
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        decode_tile_region_into_in_context(
            input,
            ctx.codec_mut(),
            pool,
            &mut out,
            stride,
            fmt,
            to_jpeg_rect(roi),
        )?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    fn decode_tile_scaled_to_surface_impl(
        ctx: &mut signinum_core::DecoderContext<CpuDecoderContext>,
        session: &mut CudaSession,
        pool: &mut CpuScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        let dims = (
            CpuDecoder::inspect(input)?
                .dimensions
                .0
                .div_ceil(scale.denominator()),
            CpuDecoder::inspect(input)?
                .dimensions
                .1
                .div_ceil(scale.denominator()),
        );
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        decode_tile_scaled_into_in_context(
            input,
            ctx.codec_mut(),
            pool,
            &mut out,
            stride,
            fmt,
            scale,
        )?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_tile_region_scaled_to_surface_impl(
        ctx: &mut signinum_core::DecoderContext<CpuDecoderContext>,
        session: &mut CudaSession,
        pool: &mut CpuScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        let dims = {
            let scaled = roi.scaled_covering(scale);
            (scaled.w, scaled.h)
        };
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        decode_tile_region_scaled_into_in_context(
            input,
            ctx.codec_mut(),
            pool,
            &mut out,
            stride,
            fmt,
            to_jpeg_rect(roi),
            scale,
        )?;
        wrap_surface(out, dims, fmt, backend, session)
    }
}

impl<'a> ImageDecodeSubmit<'a> for Decoder<'a> {
    type Session = CudaSession;
    type DeviceSurface = Surface;
    type SubmittedSurface = ReadySubmission<Surface, Error>;

    fn submit_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            self.decode_to_surface_impl(session, fmt, backend),
        ))
    }

    fn submit_region_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            self.decode_region_to_surface_impl(session, fmt, roi, backend),
        ))
    }

    fn submit_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            self.decode_scaled_to_surface_impl(session, fmt, scale, backend),
        ))
    }

    fn submit_region_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            self.decode_region_scaled_to_surface_impl(session, fmt, roi, scale, backend),
        ))
    }
}

impl TileBatchDecodeSubmit for Codec {
    type Context = CpuDecoderContext;
    type Session = CudaSession;
    type DeviceSurface = Surface;
    type SubmittedSurface = ReadySubmission<Surface, Error>;

    fn submit_tile_to_device(
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            Self::decode_tile_to_surface_impl(ctx, session, pool, input, fmt, backend),
        ))
    }

    fn submit_tile_region_to_device(
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            Self::decode_tile_region_to_surface_impl(ctx, session, pool, input, fmt, roi, backend),
        ))
    }

    fn submit_tile_scaled_to_device(
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            Self::decode_tile_scaled_to_surface_impl(
                ctx, session, pool, input, fmt, scale, backend,
            ),
        ))
    }

    fn submit_tile_region_scaled_to_device(
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            Self::decode_tile_region_scaled_to_surface_impl(
                ctx, session, pool, input, fmt, roi, scale, backend,
            ),
        ))
    }
}

impl TileBatchDecodeDevice for Codec {
    type Context = CpuDecoderContext;
    type DeviceSurface = Surface;

    fn decode_tile_to_device(
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
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
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
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
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
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

    fn decode_tile_region_scaled_to_device(
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error> {
        let mut session = CudaSession::default();
        <Self as TileBatchDecodeSubmit>::submit_tile_region_scaled_to_device(
            ctx,
            &mut session,
            pool,
            input,
            fmt,
            roi,
            scale,
            backend,
        )?
        .wait()
    }
}

fn convert_info(info: &signinum_jpeg::Info) -> signinum_core::Info {
    signinum_core::Info {
        dimensions: info.dimensions,
        components: match info.color_space {
            JpegColorSpace::Grayscale => 1,
            JpegColorSpace::YCbCr | JpegColorSpace::Rgb => 3,
            JpegColorSpace::Cmyk | JpegColorSpace::Ycck => 4,
        },
        colorspace: match info.color_space {
            JpegColorSpace::Grayscale => signinum_core::Colorspace::Grayscale,
            JpegColorSpace::YCbCr => signinum_core::Colorspace::YCbCr,
            JpegColorSpace::Rgb => signinum_core::Colorspace::Rgb,
            JpegColorSpace::Cmyk => signinum_core::Colorspace::Cmyk,
            JpegColorSpace::Ycck => signinum_core::Colorspace::Ycck,
        },
        bit_depth: info.bit_depth,
        tile_layout: None,
        coded_unit_layout: Some(signinum_core::CodedUnitLayout {
            unit_width: info.mcu_geometry.width,
            unit_height: info.mcu_geometry.height,
            units_x: info.mcu_geometry.columns,
            units_y: info.mcu_geometry.rows,
        }),
        restart_interval: info.restart_interval.map(u32::from),
        resolution_levels: 1,
    }
}

fn convert_outcome(outcome: JpegDecodeOutcome) -> DecodeOutcome<CpuWarning> {
    DecodeOutcome {
        decoded: Rect {
            x: outcome.decoded.x,
            y: outcome.decoded.y,
            w: outcome.decoded.w,
            h: outcome.decoded.h,
        },
        warnings: outcome.warnings,
    }
}

fn to_jpeg_rect(rect: Rect) -> JpegRect {
    JpegRect {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: rect.h,
    }
}

fn wrap_surface(
    bytes: Vec<u8>,
    dimensions: (u32, u32),
    fmt: PixelFormat,
    backend: BackendRequest,
    session: &mut CudaSession,
) -> Result<Surface, Error> {
    validate_surface_request(backend)?;
    let pitch_bytes = dimensions.0 as usize * fmt.bytes_per_pixel();
    match backend {
        BackendRequest::Cpu | BackendRequest::Auto => Ok(Surface {
            backend: BackendKind::Cpu,
            dimensions,
            fmt,
            pitch_bytes,
            stats: CudaSurfaceStats::default(),
            storage: Storage::Host(bytes),
        }),
        BackendRequest::Cuda => wrap_cuda_surface(&bytes, dimensions, fmt, pitch_bytes, session),
        BackendRequest::Metal => Err(Error::UnsupportedBackend { request: backend }),
    }
}

fn validate_surface_request(backend: BackendRequest) -> Result<(), Error> {
    match backend {
        BackendRequest::Cpu | BackendRequest::Auto | BackendRequest::Cuda => Ok(()),
        BackendRequest::Metal => Err(Error::UnsupportedBackend { request: backend }),
    }
}

#[cfg(feature = "cuda-runtime")]
fn wrap_cuda_surface(
    bytes: &[u8],
    dimensions: (u32, u32),
    fmt: PixelFormat,
    pitch_bytes: usize,
    session: &mut CudaSession,
) -> Result<Surface, Error> {
    let context = session.cuda_context()?;
    let output = context.copy_with_kernel(bytes).map_err(cuda_error)?;
    let (buffer, stats) = output.into_parts();
    Ok(Surface {
        backend: BackendKind::Cuda,
        dimensions,
        fmt,
        pitch_bytes,
        stats: CudaSurfaceStats {
            kernel_dispatches: stats.kernel_dispatches(),
        },
        storage: Storage::Cuda(buffer),
    })
}

#[cfg(not(feature = "cuda-runtime"))]
fn wrap_cuda_surface(
    _bytes: &[u8],
    _dimensions: (u32, u32),
    _fmt: PixelFormat,
    _pitch_bytes: usize,
    _session: &mut CudaSession,
) -> Result<Surface, Error> {
    Err(Error::CudaUnavailable)
}

#[cfg(feature = "cuda-runtime")]
fn cuda_error(error: CudaError) -> Error {
    match error {
        CudaError::Unavailable { .. } => Error::CudaUnavailable,
        other => Error::CudaRuntime {
            message: other.to_string(),
        },
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

pub use signinum_jpeg::{DecoderContext, ScratchPool};
