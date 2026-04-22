// SPDX-License-Identifier: Apache-2.0

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

mod batch;
#[cfg(target_os = "macos")]
mod compute;
mod session;
pub mod viewport;

use std::sync::Arc;

use slidecodec_core::{
    BackendKind, BackendRequest, BufferError, CodecError, DecodeOutcome, DeviceSubmission,
    DeviceSurface, Downscale, ImageCodec, ImageDecode, ImageDecodeDevice, ImageDecodeSubmit,
    PixelFormat, Rect, TileBatchDecodeDevice, TileBatchDecodeSubmit,
};
use slidecodec_jpeg::{
    ColorSpace as JpegColorSpace, DecodeOutcome as JpegDecodeOutcome, Decoder as CpuDecoder,
    DecoderContext as CpuDecoderContext, JpegError, JpegView, Rect as JpegRect,
    ScratchPool as CpuScratchPool, Warning as CpuWarning,
};

#[cfg(target_os = "macos")]
use metal::{Buffer, Device, MTLResourceOptions};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Decode(#[from] JpegError),
    #[error(transparent)]
    Buffer(#[from] BufferError),
    #[error("backend request {request:?} is not supported by slidecodec-jpeg-metal")]
    UnsupportedBackend { request: BackendRequest },
    #[error("Metal is unavailable on this host")]
    MetalUnavailable,
    #[error("Metal kernel error: {message}")]
    MetalKernel { message: String },
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
            Self::UnsupportedBackend { .. } | Self::MetalUnavailable | Self::MetalKernel { .. }
        ) || matches!(self, Self::Decode(inner) if inner.is_unsupported())
    }

    fn is_buffer_error(&self) -> bool {
        matches!(self, Self::Buffer(_))
            || matches!(self, Self::Decode(inner) if inner.is_buffer_error())
    }
}

pub(crate) enum Storage {
    Host(Vec<u8>),
    #[cfg(target_os = "macos")]
    Metal(Buffer),
}

pub struct Surface {
    backend: BackendKind,
    dimensions: (u32, u32),
    fmt: PixelFormat,
    pitch_bytes: usize,
    storage: Storage,
}

impl Surface {
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }

    pub fn as_bytes(&self) -> &[u8] {
        match &self.storage {
            Storage::Host(bytes) => bytes,
            #[cfg(target_os = "macos")]
            Storage::Metal(buffer) => {
                let len = self.byte_len();
                unsafe { core::slice::from_raw_parts(buffer.contents().cast::<u8>(), len) }
            }
        }
    }

    pub fn download_into(&self, out: &mut [u8], stride: usize) -> Result<(), Error> {
        copy_into_output(self.as_bytes(), self.dimensions, self.fmt, out, stride)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_metal_buffer(
        buffer: Buffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
    ) -> Self {
        Self {
            backend: BackendKind::Metal,
            dimensions,
            fmt,
            pitch_bytes: dimensions.0 as usize * fmt.bytes_per_pixel(),
            storage: Storage::Metal(buffer),
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

#[derive(Default)]
pub struct MetalSession {
    shared: session::SharedSession,
}

impl MetalSession {
    pub fn submissions(&self) -> u64 {
        self.shared.0.lock().expect("metal session").submissions
    }
}

impl core::fmt::Debug for MetalSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalSession")
            .field("submissions", &self.submissions())
            .finish()
    }
}

pub struct Decoder<'a> {
    inner: CpuDecoder<'a>,
    source: Arc<[u8]>,
}

impl<'a> Decoder<'a> {
    pub fn new(input: &'a [u8]) -> Result<Self, Error> {
        Ok(Self {
            inner: CpuDecoder::new(input)?,
            source: Arc::<[u8]>::from(input),
        })
    }

    pub fn from_view(view: JpegView<'a>) -> Result<Self, Error> {
        let inner = CpuDecoder::from_view(view)?;
        let source = Arc::<[u8]>::from(slidecodec_jpeg::__private::decoder_bytes(&inner));
        Ok(Self { inner, source })
    }

    pub fn inner(&self) -> &CpuDecoder<'a> {
        &self.inner
    }

    pub fn into_inner(self) -> CpuDecoder<'a> {
        self.inner
    }
}

impl ImageCodec for Decoder<'_> {
    type Error = Error;
    type Warning = CpuWarning;
    type Pool = CpuScratchPool;
}

impl<'a> ImageDecode<'a> for Decoder<'a> {
    type View = JpegView<'a>;

    fn inspect(input: &'a [u8]) -> Result<slidecodec_core::Info, Self::Error> {
        Ok(convert_info(&CpuDecoder::inspect(input)?))
    }

    fn parse(input: &'a [u8]) -> Result<Self::View, Self::Error> {
        Ok(JpegView::parse(input)?)
    }

    fn from_view(view: Self::View) -> Result<Self, Self::Error> {
        Self::from_view(view)
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
}

impl<'a> ImageDecodeDevice<'a> for Decoder<'a> {
    type DeviceSurface = Surface;

    fn decode_to_device(
        &mut self,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error> {
        let mut session = MetalSession::default();
        <Self as ImageDecodeSubmit<'a>>::submit_to_device(self, &mut session, fmt, backend)?.wait()
    }

    fn decode_region_to_device(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error> {
        let mut session = MetalSession::default();
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
        let mut session = MetalSession::default();
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
    type Warning = CpuWarning;
    type Pool = CpuScratchPool;
}

impl<'a> ImageDecodeSubmit<'a> for Decoder<'a> {
    type Session = MetalSession;
    type DeviceSurface = Surface;
    type SubmittedSurface = batch::MetalSubmission;

    fn submit_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let slot = session
            .shared
            .0
            .lock()
            .expect("metal session")
            .queue_request(batch::QueuedRequest::new(
                Arc::clone(&self.source),
                fmt,
                backend,
                batch::BatchOp::Full,
            ));
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
    }

    fn submit_region_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let slot = session
            .shared
            .0
            .lock()
            .expect("metal session")
            .queue_request(batch::QueuedRequest::new(
                Arc::clone(&self.source),
                fmt,
                backend,
                batch::BatchOp::Region(roi),
            ));
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
    }

    fn submit_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let slot = session
            .shared
            .0
            .lock()
            .expect("metal session")
            .queue_request(batch::QueuedRequest::new(
                Arc::clone(&self.source),
                fmt,
                backend,
                batch::BatchOp::Scaled(scale),
            ));
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
    }
}

impl TileBatchDecodeSubmit for Codec {
    type Context = CpuDecoderContext;
    type Session = MetalSession;
    type DeviceSurface = Surface;
    type SubmittedSurface = batch::MetalSubmission;

    fn submit_tile_to_device(
        ctx: &mut slidecodec_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let _ = (ctx, pool);
        let slot = session
            .shared
            .0
            .lock()
            .expect("metal session")
            .queue_request(batch::QueuedRequest::new(
                Arc::<[u8]>::from(input),
                fmt,
                backend,
                batch::BatchOp::Full,
            ));
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
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
        let _ = (ctx, pool);
        let slot = session
            .shared
            .0
            .lock()
            .expect("metal session")
            .queue_request(batch::QueuedRequest::new(
                Arc::<[u8]>::from(input),
                fmt,
                backend,
                batch::BatchOp::Region(roi),
            ));
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
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
        let _ = (ctx, pool);
        let slot = session
            .shared
            .0
            .lock()
            .expect("metal session")
            .queue_request(batch::QueuedRequest::new(
                Arc::<[u8]>::from(input),
                fmt,
                backend,
                batch::BatchOp::Scaled(scale),
            ));
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
    }
}

impl TileBatchDecodeDevice for Codec {
    type Context = CpuDecoderContext;
    type DeviceSurface = Surface;

    fn decode_tile_to_device(
        ctx: &mut slidecodec_core::DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error> {
        let mut session = MetalSession::default();
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
        let mut session = MetalSession::default();
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
        let mut session = MetalSession::default();
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

pub(crate) fn decode_surface_from_bytes(
    input: &[u8],
    fmt: PixelFormat,
    backend: BackendRequest,
    op: batch::BatchOp,
) -> Result<Surface, Error> {
    let decoder = CpuDecoder::new(input)?;
    let mut pool = CpuScratchPool::new();
    decode_surface_from_decoder(&decoder, &mut pool, fmt, backend, op)
}

fn decode_surface_from_decoder(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    fmt: PixelFormat,
    backend: BackendRequest,
    op: batch::BatchOp,
) -> Result<Surface, Error> {
    #[cfg(not(target_os = "macos"))]
    if matches!(backend, BackendRequest::Metal) {
        return Err(Error::MetalUnavailable);
    }

    match op {
        batch::BatchOp::Full => match backend {
            BackendRequest::Cpu => {
                let dims = decoder.info().dimensions;
                let stride = dims.0 as usize * fmt.bytes_per_pixel();
                let mut out = vec![0u8; stride * dims.1 as usize];
                decoder.decode_into_with_scratch(pool, &mut out, stride, fmt)?;
                upload_surface(out, dims, fmt, backend)
            }
            BackendRequest::Auto | BackendRequest::Metal => {
                #[cfg(target_os = "macos")]
                {
                    compute::decode_to_surface(decoder, pool, fmt)
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let dims = decoder.info().dimensions;
                    let stride = dims.0 as usize * fmt.bytes_per_pixel();
                    let mut out = vec![0u8; stride * dims.1 as usize];
                    decoder.decode_into_with_scratch(pool, &mut out, stride, fmt)?;
                    upload_surface(out, dims, fmt, BackendRequest::Cpu)
                }
            }
            BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
        },
        batch::BatchOp::Region(roi) => match backend {
            BackendRequest::Cpu => {
                let dims = (roi.w, roi.h);
                let stride = dims.0 as usize * fmt.bytes_per_pixel();
                let mut out = vec![0u8; stride * dims.1 as usize];
                decoder.decode_region_into_with_scratch(
                    pool,
                    &mut out,
                    stride,
                    fmt,
                    to_jpeg_rect(roi),
                )?;
                upload_surface(out, dims, fmt, backend)
            }
            BackendRequest::Auto | BackendRequest::Metal => {
                #[cfg(target_os = "macos")]
                {
                    compute::decode_region_to_surface(decoder, pool, fmt, to_jpeg_rect(roi))
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let dims = (roi.w, roi.h);
                    let stride = dims.0 as usize * fmt.bytes_per_pixel();
                    let mut out = vec![0u8; stride * dims.1 as usize];
                    decoder.decode_region_into_with_scratch(
                        pool,
                        &mut out,
                        stride,
                        fmt,
                        to_jpeg_rect(roi),
                    )?;
                    upload_surface(out, dims, fmt, BackendRequest::Cpu)
                }
            }
            BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
        },
        batch::BatchOp::Scaled(scale) => match backend {
            BackendRequest::Cpu => {
                let dims = scaled_dims(decoder.info().dimensions, scale);
                let stride = dims.0 as usize * fmt.bytes_per_pixel();
                let mut out = vec![0u8; stride * dims.1 as usize];
                decoder.decode_scaled_into_with_scratch(pool, &mut out, stride, fmt, scale)?;
                upload_surface(out, dims, fmt, backend)
            }
            BackendRequest::Auto | BackendRequest::Metal => {
                #[cfg(target_os = "macos")]
                {
                    compute::decode_scaled_to_surface(decoder, pool, fmt, scale)
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let dims = scaled_dims(decoder.info().dimensions, scale);
                    let stride = dims.0 as usize * fmt.bytes_per_pixel();
                    let mut out = vec![0u8; stride * dims.1 as usize];
                    decoder.decode_scaled_into_with_scratch(pool, &mut out, stride, fmt, scale)?;
                    upload_surface(out, dims, fmt, BackendRequest::Cpu)
                }
            }
            BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
        },
    }
}

fn convert_info(info: &slidecodec_jpeg::Info) -> slidecodec_core::Info {
    slidecodec_core::Info {
        dimensions: info.dimensions,
        components: match info.color_space {
            JpegColorSpace::Grayscale => 1,
            JpegColorSpace::YCbCr | JpegColorSpace::Rgb => 3,
            JpegColorSpace::Cmyk | JpegColorSpace::Ycck => 4,
        },
        colorspace: match info.color_space {
            JpegColorSpace::Grayscale => slidecodec_core::Colorspace::Grayscale,
            JpegColorSpace::YCbCr => slidecodec_core::Colorspace::YCbCr,
            JpegColorSpace::Rgb => slidecodec_core::Colorspace::Rgb,
            JpegColorSpace::Cmyk => slidecodec_core::Colorspace::Cmyk,
            JpegColorSpace::Ycck => slidecodec_core::Colorspace::Ycck,
        },
        bit_depth: info.bit_depth,
        tile_layout: None,
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

fn scaled_dims(full: (u32, u32), scale: Downscale) -> (u32, u32) {
    (
        full.0.div_ceil(scale.denominator()),
        full.1.div_ceil(scale.denominator()),
    )
}

pub(crate) fn upload_surface(
    bytes: Vec<u8>,
    dimensions: (u32, u32),
    fmt: PixelFormat,
    backend: BackendRequest,
) -> Result<Surface, Error> {
    let pitch_bytes = dimensions.0 as usize * fmt.bytes_per_pixel();
    match backend {
        BackendRequest::Cpu => Ok(Surface {
            backend: BackendKind::Cpu,
            dimensions,
            fmt,
            pitch_bytes,
            storage: Storage::Host(bytes),
        }),
        BackendRequest::Auto | BackendRequest::Metal => {
            #[cfg(target_os = "macos")]
            {
                let device = Device::system_default().ok_or(Error::MetalUnavailable)?;
                let buffer = device.new_buffer_with_data(
                    bytes.as_ptr().cast(),
                    bytes.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                Ok(Surface {
                    backend: BackendKind::Metal,
                    dimensions,
                    fmt,
                    pitch_bytes,
                    storage: Storage::Metal(buffer),
                })
            }
            #[cfg(not(target_os = "macos"))]
            {
                if matches!(backend, BackendRequest::Auto) {
                    Ok(Surface {
                        backend: BackendKind::Cpu,
                        dimensions,
                        fmt,
                        pitch_bytes,
                        storage: Storage::Host(bytes),
                    })
                } else {
                    Err(Error::MetalUnavailable)
                }
            }
        }
        BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
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

pub use slidecodec_jpeg::{
    DecoderContext, Downscale as JpegDownscale, PixelFormat as JpegPixelFormat, ScratchPool,
};
pub use slidecodec_jpeg::{Info, Rect as JpegRectPublic};
