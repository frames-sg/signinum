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
    __private::{
        build_metal_fast420_packet, build_metal_fast420_packet_for_decoder,
        build_metal_fast422_packet, build_metal_fast422_packet_for_decoder,
        build_metal_fast444_packet, build_metal_fast444_packet_for_decoder, decoder_bytes,
        JpegMetalFast420PacketV1, JpegMetalFast422PacketV1, JpegMetalFast444PacketV1,
    },
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

#[derive(Clone)]
pub(crate) enum Storage {
    Host(Vec<u8>),
    #[cfg(target_os = "macos")]
    Metal {
        buffer: Buffer,
        offset: usize,
    },
}

#[derive(Clone)]
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
            Storage::Metal { buffer, offset } => {
                let len = self.byte_len();
                unsafe {
                    core::slice::from_raw_parts(buffer.contents().cast::<u8>().add(*offset), len)
                }
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
        Self::from_metal_buffer_offset(buffer, dimensions, fmt, 0)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_metal_buffer_offset(
        buffer: Buffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
        offset: usize,
    ) -> Self {
        Self {
            backend: BackendKind::Metal,
            dimensions,
            fmt,
            pitch_bytes: dimensions.0 as usize * fmt.bytes_per_pixel(),
            storage: Storage::Metal { buffer, offset },
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
    fast444_packet: Option<Arc<JpegMetalFast444PacketV1>>,
    fast422_packet: Option<Arc<JpegMetalFast422PacketV1>>,
    fast420_packet: Option<Arc<JpegMetalFast420PacketV1>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutoDevicePath {
    CpuUpload,
    MetalKernel,
}

impl<'a> Decoder<'a> {
    pub fn new(input: &'a [u8]) -> Result<Self, Error> {
        let inner = CpuDecoder::new(input)?;
        Ok(Self {
            fast444_packet: build_metal_fast444_packet(input).ok().map(Arc::new),
            fast422_packet: build_metal_fast422_packet(input).ok().map(Arc::new),
            fast420_packet: build_metal_fast420_packet(input).ok().map(Arc::new),
            inner,
            source: Arc::<[u8]>::from(input),
        })
    }

    pub fn from_view(view: JpegView<'a>) -> Result<Self, Error> {
        let inner = CpuDecoder::from_view(view)?;
        let source = Arc::<[u8]>::from(decoder_bytes(&inner));
        let fast444_packet = build_metal_fast444_packet_for_decoder(&inner)
            .ok()
            .map(Arc::new);
        let fast422_packet = build_metal_fast422_packet_for_decoder(&inner)
            .ok()
            .map(Arc::new);
        let fast420_packet = build_metal_fast420_packet_for_decoder(&inner)
            .ok()
            .map(Arc::new);
        Ok(Self {
            inner,
            source,
            fast444_packet,
            fast422_packet,
            fast420_packet,
        })
    }

    pub fn inner(&self) -> &CpuDecoder<'a> {
        &self.inner
    }

    pub fn into_inner(self) -> CpuDecoder<'a> {
        self.inner
    }

    pub fn decode_region_scaled_to_device(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        let mut pool = CpuScratchPool::new();
        decode_region_scaled_surface_from_decoder(
            &self.inner,
            &mut pool,
            fmt,
            roi,
            scale,
            backend,
            self.fast444_packet.as_deref(),
            self.fast422_packet.as_deref(),
            self.fast420_packet.as_deref(),
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

impl Codec {
    #[allow(clippy::too_many_arguments)]
    pub fn submit_tile_region_scaled_to_device(
        ctx: &mut slidecodec_core::DecoderContext<CpuDecoderContext>,
        session: &mut MetalSession,
        pool: &mut CpuScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<<Self as TileBatchDecodeSubmit>::SubmittedSurface, Error> {
        let _ = (ctx, pool);
        let slot = {
            let mut state = session.shared.0.lock().expect("metal session");
            let input = state.intern_input_slice(input);
            let (fast444_packet, fast422_packet, fast420_packet) =
                state.resolve_fast_packets(&input, backend);
            state.queue_request(batch::QueuedRequest::new_shared(
                input,
                fmt,
                backend,
                batch::BatchOp::RegionScaled { roi, scale },
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ))
        };
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
    }

    pub fn decode_tile_region_scaled_to_device(
        ctx: &mut slidecodec_core::DecoderContext<CpuDecoderContext>,
        pool: &mut CpuScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        let mut session = MetalSession::default();
        Self::submit_tile_region_scaled_to_device(
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
        let fast444_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast444_packet.clone()
        } else {
            None
        };
        let fast422_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast422_packet.clone()
        } else {
            None
        };
        let fast420_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast420_packet.clone()
        } else {
            None
        };
        let slot = session
            .shared
            .0
            .lock()
            .expect("metal session")
            .queue_request(batch::QueuedRequest::new_shared(
                Arc::clone(&self.source),
                fmt,
                backend,
                batch::BatchOp::Full,
                fast444_packet,
                fast422_packet,
                fast420_packet,
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
        let fast444_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast444_packet.clone()
        } else {
            None
        };
        let fast422_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast422_packet.clone()
        } else {
            None
        };
        let fast420_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast420_packet.clone()
        } else {
            None
        };
        let slot = session
            .shared
            .0
            .lock()
            .expect("metal session")
            .queue_request(batch::QueuedRequest::new_shared(
                Arc::clone(&self.source),
                fmt,
                backend,
                batch::BatchOp::Region(roi),
                fast444_packet,
                fast422_packet,
                fast420_packet,
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
        let fast444_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast444_packet.clone()
        } else {
            None
        };
        let fast422_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast422_packet.clone()
        } else {
            None
        };
        let fast420_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast420_packet.clone()
        } else {
            None
        };
        let slot = session
            .shared
            .0
            .lock()
            .expect("metal session")
            .queue_request(batch::QueuedRequest::new_shared(
                Arc::clone(&self.source),
                fmt,
                backend,
                batch::BatchOp::Scaled(scale),
                fast444_packet,
                fast422_packet,
                fast420_packet,
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
        let slot = {
            let mut state = session.shared.0.lock().expect("metal session");
            let input = state.intern_input_slice(input);
            let (fast444_packet, fast422_packet, fast420_packet) =
                state.resolve_fast_packets(&input, backend);
            state.queue_request(batch::QueuedRequest::new_shared(
                input,
                fmt,
                backend,
                batch::BatchOp::Full,
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ))
        };
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
        let slot = {
            let mut state = session.shared.0.lock().expect("metal session");
            let input = state.intern_input_slice(input);
            let (fast444_packet, fast422_packet, fast420_packet) =
                state.resolve_fast_packets(&input, backend);
            state.queue_request(batch::QueuedRequest::new_shared(
                input,
                fmt,
                backend,
                batch::BatchOp::Region(roi),
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ))
        };
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
        let slot = {
            let mut state = session.shared.0.lock().expect("metal session");
            let input = state.intern_input_slice(input);
            let (fast444_packet, fast422_packet, fast420_packet) =
                state.resolve_fast_packets(&input, backend);
            state.queue_request(batch::QueuedRequest::new_shared(
                input,
                fmt,
                backend,
                batch::BatchOp::Scaled(scale),
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ))
        };
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
    fast444_packet: Option<Arc<JpegMetalFast444PacketV1>>,
    fast422_packet: Option<Arc<JpegMetalFast422PacketV1>>,
    fast420_packet: Option<Arc<JpegMetalFast420PacketV1>>,
) -> Result<Surface, Error> {
    let decoder = CpuDecoder::new(input)?;
    let mut pool = CpuScratchPool::new();
    let build_auto_packets =
        matches!(backend, BackendRequest::Auto) && decoder.info().restart_interval.is_some();
    let build_metal_packets = matches!(backend, BackendRequest::Metal);
    let fast444_packet = if build_auto_packets || build_metal_packets {
        fast444_packet.or_else(|| {
            build_metal_fast444_packet_for_decoder(&decoder)
                .ok()
                .map(Arc::new)
        })
    } else {
        None
    };
    let fast422_packet = if build_auto_packets || build_metal_packets {
        fast422_packet.or_else(|| {
            build_metal_fast422_packet_for_decoder(&decoder)
                .ok()
                .map(Arc::new)
        })
    } else {
        None
    };
    let fast420_packet = if build_auto_packets || build_metal_packets {
        fast420_packet.or_else(|| {
            build_metal_fast420_packet_for_decoder(&decoder)
                .ok()
                .map(Arc::new)
        })
    } else {
        None
    };
    decode_surface_from_decoder(
        &decoder,
        &mut pool,
        fmt,
        backend,
        op,
        fast444_packet.as_deref(),
        fast422_packet.as_deref(),
        fast420_packet.as_deref(),
    )
}

pub(crate) fn decode_compatible_batch(
    requests: &[batch::QueuedRequest],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    #[cfg(target_os = "macos")]
    {
        compute::decode_full_batch_to_surfaces(requests)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = requests;
        Ok(None)
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_surface_from_decoder(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    fmt: PixelFormat,
    backend: BackendRequest,
    op: batch::BatchOp,
    fast444_packet: Option<&JpegMetalFast444PacketV1>,
    fast422_packet: Option<&JpegMetalFast422PacketV1>,
    fast420_packet: Option<&JpegMetalFast420PacketV1>,
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
            BackendRequest::Auto => {
                #[cfg(target_os = "macos")]
                {
                    match choose_auto_device_path(
                        decoder,
                        op,
                        fast444_packet,
                        fast422_packet,
                        fast420_packet,
                    ) {
                        AutoDevicePath::CpuUpload => {
                            let dims = decoder.info().dimensions;
                            let stride = dims.0 as usize * fmt.bytes_per_pixel();
                            let mut out = vec![0u8; stride * dims.1 as usize];
                            decoder.decode_into_with_scratch(pool, &mut out, stride, fmt)?;
                            upload_surface(out, dims, fmt, backend)
                        }
                        AutoDevicePath::MetalKernel => compute::decode_to_surface(
                            decoder,
                            pool,
                            fmt,
                            fast444_packet,
                            fast422_packet,
                            fast420_packet,
                        ),
                    }
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
            BackendRequest::Metal => {
                #[cfg(target_os = "macos")]
                {
                    compute::decode_to_surface(
                        decoder,
                        pool,
                        fmt,
                        fast444_packet,
                        fast422_packet,
                        fast420_packet,
                    )
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
            BackendRequest::Auto => {
                #[cfg(target_os = "macos")]
                {
                    match choose_auto_device_path(
                        decoder,
                        op,
                        fast444_packet,
                        fast422_packet,
                        fast420_packet,
                    ) {
                        AutoDevicePath::CpuUpload => {
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
                        AutoDevicePath::MetalKernel => compute::decode_region_to_surface(
                            decoder,
                            pool,
                            fmt,
                            to_jpeg_rect(roi),
                            fast444_packet,
                            fast422_packet,
                            fast420_packet,
                        ),
                    }
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
            BackendRequest::Metal => {
                #[cfg(target_os = "macos")]
                {
                    compute::decode_region_to_surface(
                        decoder,
                        pool,
                        fmt,
                        to_jpeg_rect(roi),
                        fast444_packet,
                        fast422_packet,
                        fast420_packet,
                    )
                }
                #[cfg(not(target_os = "macos"))]
                unreachable!("Metal region path is gated above");
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
            BackendRequest::Auto => {
                #[cfg(target_os = "macos")]
                {
                    match choose_auto_device_path(
                        decoder,
                        op,
                        fast444_packet,
                        fast422_packet,
                        fast420_packet,
                    ) {
                        AutoDevicePath::CpuUpload => {
                            let dims = scaled_dims(decoder.info().dimensions, scale);
                            let stride = dims.0 as usize * fmt.bytes_per_pixel();
                            let mut out = vec![0u8; stride * dims.1 as usize];
                            decoder.decode_scaled_into_with_scratch(
                                pool, &mut out, stride, fmt, scale,
                            )?;
                            upload_surface(out, dims, fmt, backend)
                        }
                        AutoDevicePath::MetalKernel => compute::decode_scaled_to_surface(
                            decoder,
                            pool,
                            fmt,
                            scale,
                            fast444_packet,
                            fast422_packet,
                            fast420_packet,
                        ),
                    }
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
            BackendRequest::Metal => {
                #[cfg(target_os = "macos")]
                {
                    compute::decode_scaled_to_surface(
                        decoder,
                        pool,
                        fmt,
                        scale,
                        fast444_packet,
                        fast422_packet,
                        fast420_packet,
                    )
                }
                #[cfg(not(target_os = "macos"))]
                unreachable!("Metal scaled path is gated above");
            }
            BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
        },
        batch::BatchOp::RegionScaled { roi, scale } => decode_region_scaled_surface_from_decoder(
            decoder,
            pool,
            fmt,
            roi,
            scale,
            backend,
            fast444_packet,
            fast422_packet,
            fast420_packet,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_region_scaled_surface_from_decoder(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
    backend: BackendRequest,
    fast444_packet: Option<&JpegMetalFast444PacketV1>,
    fast422_packet: Option<&JpegMetalFast422PacketV1>,
    fast420_packet: Option<&JpegMetalFast420PacketV1>,
) -> Result<Surface, Error> {
    match backend {
        BackendRequest::Cpu => {
            decode_region_scaled_cpu_upload(decoder, pool, fmt, roi, scale, backend)
        }
        BackendRequest::Auto => {
            #[cfg(target_os = "macos")]
            {
                match choose_auto_device_path(
                    decoder,
                    batch::BatchOp::RegionScaled { roi, scale },
                    fast444_packet,
                    fast422_packet,
                    fast420_packet,
                ) {
                    AutoDevicePath::CpuUpload => {
                        decode_region_scaled_cpu_upload(decoder, pool, fmt, roi, scale, backend)
                    }
                    AutoDevicePath::MetalKernel => compute::decode_region_scaled_to_surface(
                        decoder,
                        pool,
                        fmt,
                        to_jpeg_rect(roi),
                        scale,
                        fast444_packet,
                        fast422_packet,
                        fast420_packet,
                    ),
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                decode_region_scaled_cpu_upload(decoder, pool, fmt, roi, scale, backend)
            }
        }
        BackendRequest::Metal => {
            #[cfg(target_os = "macos")]
            {
                compute::decode_region_scaled_to_surface(
                    decoder,
                    pool,
                    fmt,
                    to_jpeg_rect(roi),
                    scale,
                    fast444_packet,
                    fast422_packet,
                    fast420_packet,
                )
            }
            #[cfg(not(target_os = "macos"))]
            {
                Err(Error::MetalUnavailable)
            }
        }
        BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
    }
}

fn decode_region_scaled_cpu_upload(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
    backend: BackendRequest,
) -> Result<Surface, Error> {
    let scaled = scaled_rect_covering(roi, scale);
    let dims = (scaled.w, scaled.h);
    let stride = dims.0 as usize * fmt.bytes_per_pixel();
    let mut out = vec![0u8; stride * dims.1 as usize];
    decoder.decode_region_scaled_into_with_scratch(
        pool,
        &mut out,
        stride,
        fmt,
        to_jpeg_rect(roi),
        scale,
    )?;
    upload_surface(out, dims, fmt, backend)
}

fn choose_auto_device_path(
    decoder: &CpuDecoder<'_>,
    op: batch::BatchOp,
    fast444_packet: Option<&JpegMetalFast444PacketV1>,
    fast422_packet: Option<&JpegMetalFast422PacketV1>,
    fast420_packet: Option<&JpegMetalFast420PacketV1>,
) -> AutoDevicePath {
    if matches!(op, batch::BatchOp::RegionScaled { .. }) {
        return AutoDevicePath::CpuUpload;
    }

    let direct_packet =
        fast444_packet.is_some() || fast420_packet.is_some() || fast422_packet.is_some();
    if decoder.info().restart_interval.is_some() && direct_packet {
        AutoDevicePath::MetalKernel
    } else {
        AutoDevicePath::CpuUpload
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
        coded_unit_layout: Some(slidecodec_core::CodedUnitLayout {
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

fn scaled_dims(full: (u32, u32), scale: Downscale) -> (u32, u32) {
    (
        full.0.div_ceil(scale.denominator()),
        full.1.div_ceil(scale.denominator()),
    )
}

fn scaled_rect_covering(rect: Rect, scale: Downscale) -> Rect {
    let denom = scale.denominator();
    let x_end = rect.x + rect.w;
    let y_end = rect.y + rect.h;
    let x0 = rect.x / denom;
    let y0 = rect.y / denom;
    let x1 = x_end.div_ceil(denom);
    let y1 = y_end.div_ceil(denom);
    Rect {
        x: x0,
        y: y0,
        w: x1.saturating_sub(x0),
        h: y1.saturating_sub(y0),
    }
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
                    storage: Storage::Metal { buffer, offset: 0 },
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

#[cfg(test)]
mod tests {
    use super::*;
    use slidecodec_jpeg::__private::{build_metal_fast420_packet, build_metal_fast444_packet};

    const BASELINE_420: &[u8] =
        include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");
    const BASELINE_420_RESTART: &[u8] =
        include_bytes!("../../../corpus/conformance/baseline_420_restart_32x16.jpg");
    const BASELINE_444: &[u8] = include_bytes!("../../../corpus/conformance/baseline_444_8x8.jpg");

    #[test]
    fn auto_device_path_prefers_cpu_upload_for_nonrestart_packets() {
        let decoder_420 = CpuDecoder::new(BASELINE_420).expect("420 decoder");
        let packet_420 = build_metal_fast420_packet(BASELINE_420).expect("420 packet");
        assert_eq!(
            choose_auto_device_path(
                &decoder_420,
                batch::BatchOp::Full,
                None,
                None,
                Some(&packet_420),
            ),
            AutoDevicePath::CpuUpload
        );

        let decoder_444 = CpuDecoder::new(BASELINE_444).expect("444 decoder");
        let packet_444 = build_metal_fast444_packet(BASELINE_444).expect("444 packet");
        assert_eq!(
            choose_auto_device_path(
                &decoder_444,
                batch::BatchOp::Scaled(Downscale::Quarter),
                Some(&packet_444),
                None,
                None,
            ),
            AutoDevicePath::CpuUpload
        );
    }

    #[test]
    fn auto_device_path_prefers_metal_for_restart_packets() {
        let decoder = CpuDecoder::new(BASELINE_420_RESTART).expect("restart decoder");
        let packet = build_metal_fast420_packet(BASELINE_420_RESTART).expect("restart packet");

        assert_eq!(
            choose_auto_device_path(&decoder, batch::BatchOp::Full, None, None, Some(&packet)),
            AutoDevicePath::MetalKernel
        );
        assert_eq!(
            choose_auto_device_path(
                &decoder,
                batch::BatchOp::Region(Rect {
                    x: 0,
                    y: 0,
                    w: 16,
                    h: 16,
                }),
                None,
                None,
                Some(&packet),
            ),
            AutoDevicePath::MetalKernel
        );
    }

    #[test]
    fn auto_device_path_prefers_cpu_upload_for_region_scaled_even_with_restart_packets() {
        let decoder = CpuDecoder::new(BASELINE_420_RESTART).expect("restart decoder");
        let packet = build_metal_fast420_packet(BASELINE_420_RESTART).expect("restart packet");

        assert_eq!(
            choose_auto_device_path(
                &decoder,
                batch::BatchOp::RegionScaled {
                    roi: Rect {
                        x: 0,
                        y: 0,
                        w: 16,
                        h: 16,
                    },
                    scale: Downscale::Quarter,
                },
                None,
                None,
                Some(&packet),
            ),
            AutoDevicePath::CpuUpload
        );
    }
}
