// SPDX-License-Identifier: Apache-2.0

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

#[cfg(target_os = "macos")]
mod compute;

use core::convert::Infallible;

use slidecodec_core::{
    BackendKind, BackendRequest, BufferError, CodecError, DecodeOutcome, DeviceSubmission,
    DeviceSurface, Downscale, ImageCodec, ImageDecode, ImageDecodeDevice, ImageDecodeSubmit,
    PixelFormat, ReadySubmission, Rect, TileBatchDecodeDevice, TileBatchDecodeSubmit,
};
use slidecodec_j2k::{
    J2kContext as CpuJ2kContext, J2kDecoder as CpuDecoder, J2kError,
    J2kScratchPool as CpuJ2kScratchPool, J2kView,
};
#[cfg(target_os = "macos")]
use slidecodec_j2k_native::{
    DecodeSettings as NativeDecodeSettings, DecoderContext as NativeDecoderContext,
    Image as NativeImage,
};

#[cfg(target_os = "macos")]
use metal::{Buffer, Device, MTLResourceOptions};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Decode(#[from] J2kError),
    #[error(transparent)]
    Buffer(#[from] BufferError),
    #[error("backend request {request:?} is not supported by slidecodec-j2k-metal")]
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MetalSession {
    submissions: u64,
}

impl MetalSession {
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
    #[cfg(target_os = "macos")]
    native_image: Option<NativeImage<'a>>,
    #[cfg(target_os = "macos")]
    native_context: NativeDecoderContext<'a>,
}

impl<'a> J2kDecoder<'a> {
    pub fn new(input: &'a [u8]) -> Result<Self, Error> {
        Ok(Self {
            inner: CpuDecoder::new(input)?,
            pool: CpuJ2kScratchPool::new(),
            #[cfg(target_os = "macos")]
            native_image: None,
            #[cfg(target_os = "macos")]
            native_context: NativeDecoderContext::default(),
        })
    }

    pub fn from_view(view: J2kView<'a>) -> Result<Self, Error> {
        Ok(Self {
            inner: CpuDecoder::from_view(view)?,
            pool: CpuJ2kScratchPool::new(),
            #[cfg(target_os = "macos")]
            native_image: None,
            #[cfg(target_os = "macos")]
            native_context: NativeDecoderContext::default(),
        })
    }

    pub fn inner(&self) -> &CpuDecoder<'a> {
        &self.inner
    }

    #[cfg(target_os = "macos")]
    fn ensure_native_image(&mut self) -> Result<(), Error> {
        if self.native_image.is_none() {
            self.native_image = Some(
                NativeImage::new(self.inner.bytes(), &NativeDecodeSettings::default())
                    .map_err(|error| J2kError::Backend(error.to_string()))?,
            );
        }
        Ok(())
    }

    fn decode_to_surface_impl(
        &mut self,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        #[cfg(not(target_os = "macos"))]
        if matches!(backend, BackendRequest::Metal) {
            return Err(Error::MetalUnavailable);
        }
        match backend {
            BackendRequest::Cpu => {
                let dims = self.inner.info().dimensions;
                let stride = dims.0 as usize * fmt.bytes_per_pixel();
                let mut out = vec![0u8; stride * dims.1 as usize];
                self.inner
                    .decode_into_with_scratch(&mut self.pool, &mut out, stride, fmt)?;
                upload_surface(out, dims, fmt, backend)
            }
            BackendRequest::Auto | BackendRequest::Metal => {
                #[cfg(target_os = "macos")]
                {
                    self.ensure_native_image()?;
                    let (Some(image), native_context) =
                        (self.native_image.as_ref(), &mut self.native_context)
                    else {
                        return Err(Error::Decode(J2kError::Backend(
                            "native image cache missing".to_string(),
                        )));
                    };
                    compute::decode_image_to_surface(image, native_context, fmt)
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let dims = self.inner.info().dimensions;
                    let stride = dims.0 as usize * fmt.bytes_per_pixel();
                    let mut out = vec![0u8; stride * dims.1 as usize];
                    self.inner
                        .decode_into_with_scratch(&mut self.pool, &mut out, stride, fmt)?;
                    upload_surface(out, dims, fmt, BackendRequest::Cpu)
                }
            }
            BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
        }
    }

    fn decode_region_to_surface_impl(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        let dims = self.inner.info().dimensions;
        if !roi.is_within(dims) {
            return Err(J2kError::InvalidRegion {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
                image_w: dims.0,
                image_h: dims.1,
            }
            .into());
        }
        #[cfg(not(target_os = "macos"))]
        if matches!(backend, BackendRequest::Metal) {
            return Err(Error::MetalUnavailable);
        }
        match backend {
            BackendRequest::Cpu => {
                let region_dims = (roi.w, roi.h);
                let stride = region_dims.0 as usize * fmt.bytes_per_pixel();
                let mut out = vec![0u8; stride * region_dims.1 as usize];
                self.inner
                    .decode_region_into(&mut self.pool, &mut out, stride, fmt, roi)?;
                upload_surface(out, region_dims, fmt, backend)
            }
            BackendRequest::Auto | BackendRequest::Metal => {
                #[cfg(target_os = "macos")]
                {
                    self.ensure_native_image()?;
                    let (Some(image), native_context) =
                        (self.native_image.as_ref(), &mut self.native_context)
                    else {
                        return Err(Error::Decode(J2kError::Backend(
                            "native image cache missing".to_string(),
                        )));
                    };
                    compute::decode_region_to_surface(image, native_context, fmt, roi)
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let region_dims = (roi.w, roi.h);
                    let stride = region_dims.0 as usize * fmt.bytes_per_pixel();
                    let mut out = vec![0u8; stride * region_dims.1 as usize];
                    self.inner
                        .decode_region_into(&mut self.pool, &mut out, stride, fmt, roi)?;
                    upload_surface(out, region_dims, fmt, BackendRequest::Cpu)
                }
            }
            BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
        }
    }

    fn decode_scaled_to_surface_impl(
        &mut self,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        #[cfg(not(target_os = "macos"))]
        if matches!(backend, BackendRequest::Metal) {
            return Err(Error::MetalUnavailable);
        }
        match backend {
            BackendRequest::Cpu => {
                let dims = scaled_dims(self.inner.info().dimensions, scale);
                let stride = dims.0 as usize * fmt.bytes_per_pixel();
                let mut out = vec![0u8; stride * dims.1 as usize];
                self.inner
                    .decode_scaled_into(&mut self.pool, &mut out, stride, fmt, scale)?;
                upload_surface(out, dims, fmt, backend)
            }
            BackendRequest::Auto | BackendRequest::Metal => {
                #[cfg(target_os = "macos")]
                {
                    match compute::decode_scaled_to_surface(
                        self.inner.bytes(),
                        self.inner.info().dimensions,
                        fmt,
                        scale,
                    ) {
                        Ok(surface) => Ok(surface),
                        Err(error) if scale != Downscale::None && is_htj2k_scaled_gap(&error) => {
                            let dims = scaled_dims(self.inner.info().dimensions, scale);
                            let stride = dims.0 as usize * fmt.bytes_per_pixel();
                            let mut out = vec![0u8; stride * dims.1 as usize];
                            self.inner.decode_scaled_into(
                                &mut self.pool,
                                &mut out,
                                stride,
                                fmt,
                                scale,
                            )?;
                            upload_surface(out, dims, fmt, BackendRequest::Metal)
                        }
                        Err(error) => Err(error),
                    }
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let dims = scaled_dims(self.inner.info().dimensions, scale);
                    let stride = dims.0 as usize * fmt.bytes_per_pixel();
                    let mut out = vec![0u8; stride * dims.1 as usize];
                    self.inner
                        .decode_scaled_into(&mut self.pool, &mut out, stride, fmt, scale)?;
                    upload_surface(out, dims, fmt, BackendRequest::Cpu)
                }
            }
            BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
        }
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
        Self::from_view(view)
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
    type Warning = Infallible;
    type Pool = CpuJ2kScratchPool;
}

impl<'a> ImageDecodeSubmit<'a> for J2kDecoder<'a> {
    type Session = MetalSession;
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
    type Session = MetalSession;
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
        let _ = (ctx, pool);
        let mut decoder = J2kDecoder::new(input)?;
        <J2kDecoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
            &mut decoder,
            session,
            fmt,
            backend,
        )
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
        let mut decoder = J2kDecoder::new(input)?;
        <J2kDecoder<'_> as ImageDecodeSubmit<'_>>::submit_region_to_device(
            &mut decoder,
            session,
            fmt,
            roi,
            backend,
        )
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
        let mut decoder = J2kDecoder::new(input)?;
        <J2kDecoder<'_> as ImageDecodeSubmit<'_>>::submit_scaled_to_device(
            &mut decoder,
            session,
            fmt,
            scale,
            backend,
        )
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

fn scaled_dims(full: (u32, u32), scale: Downscale) -> (u32, u32) {
    (
        full.0.div_ceil(scale.denominator()),
        full.1.div_ceil(scale.denominator()),
    )
}

fn is_htj2k_scaled_gap(error: &Error) -> bool {
    matches!(error, Error::Decode(J2kError::Backend(message)) if message.contains("HTJ2K decode"))
}

fn upload_surface(
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

pub use slidecodec_j2k::{J2kContext, J2kScratchPool};
