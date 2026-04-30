// SPDX-License-Identifier: Apache-2.0

//! CUDA-facing device-output adapter for `ashlar-jpeg`.
//!
//! This crate intentionally exposes the same backend-selection surface as the
//! Metal adapter, but the `0.1.0` implementation is fallback-only: CPU and
//! auto requests return host-backed surfaces, while explicit CUDA requests
//! report CUDA as unavailable.

#![warn(unreachable_pub)]

use ashlar_core::{
    BackendKind, BackendRequest, BufferError, CodecError, DecodeOutcome, DeviceSubmission,
    DeviceSurface, Downscale, ImageCodec, ImageDecode, ImageDecodeDevice, ImageDecodeSubmit,
    PixelFormat, ReadySubmission, Rect, TileBatchDecodeDevice, TileBatchDecodeSubmit,
};
use ashlar_jpeg::{
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
    #[error("backend request {request:?} is not supported by ashlar-jpeg-cuda")]
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
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_host_surface_request(backend)?;
        let (bytes, _outcome) = self.inner.decode(fmt)?;
        wrap_surface(bytes, self.inner.info().dimensions, fmt, backend)
    }

    fn decode_region_to_surface_impl(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_host_surface_request(backend)?;
        let (bytes, outcome) = self.inner.decode_region(fmt, to_jpeg_rect(roi))?;
        wrap_surface(bytes, (outcome.decoded.w, outcome.decoded.h), fmt, backend)
    }

    fn decode_scaled_to_surface_impl(
        &mut self,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_host_surface_request(backend)?;
        let (bytes, outcome) = self.inner.decode_scaled(fmt, scale)?;
        wrap_surface(bytes, (outcome.decoded.w, outcome.decoded.h), fmt, backend)
    }

    fn decode_region_scaled_to_surface_impl(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_host_surface_request(backend)?;
        let (bytes, outcome) = self
            .inner
            .decode_region_scaled(fmt, to_jpeg_rect(roi), scale)?;
        wrap_surface(bytes, (outcome.decoded.w, outcome.decoded.h), fmt, backend)
    }
}

impl ImageCodec for Decoder<'_> {
    type Error = Error;
    type Warning = CpuWarning;
    type Pool = CpuScratchPool;
}

impl<'a> ImageDecode<'a> for Decoder<'a> {
    type View = JpegView<'a>;

    fn inspect(input: &'a [u8]) -> Result<ashlar_core::Info, Self::Error> {
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
        ctx: &mut ashlar_core::DecoderContext<CpuDecoderContext>,
        pool: &mut CpuScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_host_surface_request(backend)?;
        let dims = CpuDecoder::inspect(input)?.dimensions;
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        decode_tile_into_in_context(input, ctx.codec_mut(), pool, &mut out, stride, fmt)?;
        wrap_surface(out, dims, fmt, backend)
    }

    fn decode_tile_region_to_surface_impl(
        ctx: &mut ashlar_core::DecoderContext<CpuDecoderContext>,
        pool: &mut CpuScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_host_surface_request(backend)?;
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
        wrap_surface(out, dims, fmt, backend)
    }

    fn decode_tile_scaled_to_surface_impl(
        ctx: &mut ashlar_core::DecoderContext<CpuDecoderContext>,
        pool: &mut CpuScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_host_surface_request(backend)?;
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
        wrap_surface(out, dims, fmt, backend)
    }

    fn decode_tile_region_scaled_to_surface_impl(
        ctx: &mut ashlar_core::DecoderContext<CpuDecoderContext>,
        pool: &mut CpuScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_host_surface_request(backend)?;
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
        wrap_surface(out, dims, fmt, backend)
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
        validate_host_surface_request(backend)?;
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
        validate_host_surface_request(backend)?;
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
        validate_host_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            self.decode_scaled_to_surface_impl(fmt, scale, backend),
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
        validate_host_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            self.decode_region_scaled_to_surface_impl(fmt, roi, scale, backend),
        ))
    }
}

impl TileBatchDecodeSubmit for Codec {
    type Context = CpuDecoderContext;
    type Session = CudaSession;
    type DeviceSurface = Surface;
    type SubmittedSurface = ReadySubmission<Surface, Error>;

    fn submit_tile_to_device(
        ctx: &mut ashlar_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_host_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            Self::decode_tile_to_surface_impl(ctx, pool, input, fmt, backend),
        ))
    }

    fn submit_tile_region_to_device(
        ctx: &mut ashlar_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_host_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            Self::decode_tile_region_to_surface_impl(ctx, pool, input, fmt, roi, backend),
        ))
    }

    fn submit_tile_scaled_to_device(
        ctx: &mut ashlar_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_host_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            Self::decode_tile_scaled_to_surface_impl(ctx, pool, input, fmt, scale, backend),
        ))
    }

    fn submit_tile_region_scaled_to_device(
        ctx: &mut ashlar_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_host_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            Self::decode_tile_region_scaled_to_surface_impl(
                ctx, pool, input, fmt, roi, scale, backend,
            ),
        ))
    }
}

impl TileBatchDecodeDevice for Codec {
    type Context = CpuDecoderContext;
    type DeviceSurface = Surface;

    fn decode_tile_to_device(
        ctx: &mut ashlar_core::DecoderContext<Self::Context>,
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
        ctx: &mut ashlar_core::DecoderContext<Self::Context>,
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
        ctx: &mut ashlar_core::DecoderContext<Self::Context>,
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
        ctx: &mut ashlar_core::DecoderContext<Self::Context>,
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

fn convert_info(info: &ashlar_jpeg::Info) -> ashlar_core::Info {
    ashlar_core::Info {
        dimensions: info.dimensions,
        components: match info.color_space {
            JpegColorSpace::Grayscale => 1,
            JpegColorSpace::YCbCr | JpegColorSpace::Rgb => 3,
            JpegColorSpace::Cmyk | JpegColorSpace::Ycck => 4,
        },
        colorspace: match info.color_space {
            JpegColorSpace::Grayscale => ashlar_core::Colorspace::Grayscale,
            JpegColorSpace::YCbCr => ashlar_core::Colorspace::YCbCr,
            JpegColorSpace::Rgb => ashlar_core::Colorspace::Rgb,
            JpegColorSpace::Cmyk => ashlar_core::Colorspace::Cmyk,
            JpegColorSpace::Ycck => ashlar_core::Colorspace::Ycck,
        },
        bit_depth: info.bit_depth,
        tile_layout: None,
        coded_unit_layout: Some(ashlar_core::CodedUnitLayout {
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
) -> Result<Surface, Error> {
    validate_host_surface_request(backend)?;
    Ok(Surface {
        backend: BackendKind::Cpu,
        dimensions,
        fmt,
        pitch_bytes: dimensions.0 as usize * fmt.bytes_per_pixel(),
        bytes,
    })
}

fn validate_host_surface_request(backend: BackendRequest) -> Result<(), Error> {
    match backend {
        BackendRequest::Cpu | BackendRequest::Auto => Ok(()),
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

pub use ashlar_jpeg::{DecoderContext, ScratchPool};
