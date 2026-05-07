// SPDX-License-Identifier: Apache-2.0

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

mod batch;
#[cfg(target_os = "macos")]
mod compute;
mod encode;
mod routing;
mod session;
pub mod viewport;

use std::sync::Arc;
#[cfg(target_os = "macos")]
use std::sync::OnceLock;

use signinum_core::{
    BackendKind, BackendRequest, BufferError, CodecError, DecodeOutcome, DeviceSubmission,
    DeviceSurface, Downscale, ImageCodec, ImageDecode, ImageDecodeDevice, ImageDecodeSubmit,
    PixelFormat, Rect, TileBatchDecodeDevice, TileBatchDecodeSubmit,
};
use signinum_jpeg::{
    adapter::{
        build_metal_fast420_packet, build_metal_fast420_packet_for_decoder,
        build_metal_fast422_packet, build_metal_fast422_packet_for_decoder,
        build_metal_fast444_packet, build_metal_fast444_packet_for_decoder, decoder_bytes,
        JpegMetalFast420PacketV1, JpegMetalFast422PacketV1, JpegMetalFast444PacketV1,
    },
    ColorSpace as JpegColorSpace, DecodeOutcome as JpegDecodeOutcome, Decoder as CpuDecoder,
    DecoderContext as CpuDecoderContext, JpegError, JpegView, Rect as JpegRect,
    ScratchPool as CpuScratchPool, Warning as CpuWarning,
};

pub use encode::{
    encode_jpeg_baseline_batch_from_metal_buffers, encode_jpeg_baseline_from_metal_buffer,
    JpegBaselineMetalEncodeTile,
};

#[cfg(target_os = "macos")]
use metal::{Buffer, CommandBuffer, Device, MTLResourceOptions};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Decode(#[from] JpegError),
    #[error(transparent)]
    Encode(#[from] signinum_jpeg::JpegEncodeError),
    #[error(transparent)]
    Buffer(#[from] BufferError),
    #[error("backend request {request:?} is not supported by signinum-jpeg-metal")]
    UnsupportedBackend { request: BackendRequest },
    #[error("unsupported JPEG Metal request: {reason}")]
    UnsupportedMetalRequest { reason: &'static str },
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
            Self::UnsupportedBackend { .. }
                | Self::MetalUnavailable
                | Self::MetalKernel { .. }
                | Self::UnsupportedMetalRequest { .. }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceResidency {
    Host,
    MetalResidentDecode,
    CpuStagedMetalUpload,
}

#[derive(Clone)]
pub struct Surface {
    backend: BackendKind,
    residency: SurfaceResidency,
    dimensions: (u32, u32),
    fmt: PixelFormat,
    pitch_bytes: usize,
    storage: Storage,
}

impl Surface {
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }

    pub fn residency(&self) -> SurfaceResidency {
        self.residency
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
    pub fn metal_buffer(&self) -> Option<(&Buffer, usize)> {
        match &self.storage {
            Storage::Metal { buffer, offset } => Some((buffer, *offset)),
            Storage::Host(_) => None,
        }
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
            residency: SurfaceResidency::MetalResidentDecode,
            dimensions,
            fmt,
            pitch_bytes: dimensions.0 as usize * fmt.bytes_per_pixel(),
            storage: Storage::Metal { buffer, offset },
        }
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_cpu_staged_metal_buffer(
        buffer: Buffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
    ) -> Self {
        Self::from_cpu_staged_metal_buffer_offset(buffer, dimensions, fmt, 0)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_cpu_staged_metal_buffer_offset(
        buffer: Buffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
        offset: usize,
    ) -> Self {
        Self {
            backend: BackendKind::Metal,
            residency: SurfaceResidency::CpuStagedMetalUpload,
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

#[cfg(target_os = "macos")]
#[doc(hidden)]
#[derive(Clone)]
pub struct ResidentPrivateJpegTile {
    pub buffer: Buffer,
    pub byte_offset: usize,
    pub dimensions: (u32, u32),
    pub pixel_format: PixelFormat,
    pub pitch_bytes: usize,
    pub status_buffer: Buffer,
    pub command_buffer: CommandBuffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
pub struct MetalBackendSession {
    device: Device,
    runtime: Arc<OnceLock<Result<compute::MetalRuntime, String>>>,
}

#[cfg(target_os = "macos")]
impl MetalBackendSession {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            runtime: Arc::new(OnceLock::new()),
        }
    }

    pub fn system_default() -> Result<Self, Error> {
        Device::system_default()
            .map(Self::new)
            .ok_or(Error::MetalUnavailable)
    }

    pub fn device(&self) -> &metal::DeviceRef {
        self.device.as_ref()
    }
}

#[cfg(target_os = "macos")]
impl core::fmt::Debug for MetalBackendSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalBackendSession")
            .field("device", &self.device.name())
            .field("runtime_initialized", &self.runtime.get().is_some())
            .finish()
    }
}

#[cfg(not(target_os = "macos"))]
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalBackendSession {
    _private: (),
}

#[cfg(not(target_os = "macos"))]
impl MetalBackendSession {
    pub fn system_default() -> Result<Self, Error> {
        Err(Error::MetalUnavailable)
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

    pub fn decode_to_device_with_session(
        &mut self,
        fmt: PixelFormat,
        session: &MetalBackendSession,
    ) -> Result<Surface, Error> {
        #[cfg(target_os = "macos")]
        {
            let mut pool = CpuScratchPool::new();
            let decision = choose_route(
                &self.inner,
                BackendRequest::Metal,
                fmt,
                batch::BatchOp::Full,
                self.fast444_packet.as_deref(),
                self.fast422_packet.as_deref(),
                self.fast420_packet.as_deref(),
            );
            if let Some(err) = routing::decision_error(decision) {
                return Err(err);
            }
            match decision {
                routing::RouteDecision::MetalKernel => {
                    reject_cpu_staged_metal_upload(compute::decode_to_surface_with_session(
                        &self.inner,
                        &mut pool,
                        fmt,
                        self.fast444_packet.as_deref(),
                        self.fast422_packet.as_deref(),
                        self.fast420_packet.as_deref(),
                        session,
                    )?)
                }
                routing::RouteDecision::CpuHost
                | routing::RouteDecision::RejectExplicitMetal { .. }
                | routing::RouteDecision::RejectUnsupportedBackend { .. }
                | routing::RouteDecision::MetalUnavailable => unreachable!("handled above"),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = session;
            let decision = choose_route(
                &self.inner,
                BackendRequest::Metal,
                fmt,
                batch::BatchOp::Full,
                self.fast444_packet.as_deref(),
                self.fast422_packet.as_deref(),
                self.fast420_packet.as_deref(),
            );
            if let Some(err) = routing::decision_error(decision) {
                return Err(err);
            }
            Err(Error::MetalUnavailable)
        }
    }

    #[cfg(target_os = "macos")]
    #[doc(hidden)]
    pub fn decode_private_rgb8_tile_with_session(
        &mut self,
        session: &MetalBackendSession,
    ) -> Result<ResidentPrivateJpegTile, Error> {
        let decision = choose_route(
            &self.inner,
            BackendRequest::Metal,
            PixelFormat::Rgb8,
            batch::BatchOp::Full,
            self.fast444_packet.as_deref(),
            self.fast422_packet.as_deref(),
            self.fast420_packet.as_deref(),
        );
        if let Some(err) = routing::decision_error(decision) {
            return Err(err);
        }
        match decision {
            routing::RouteDecision::MetalKernel => compute::decode_private_rgb8_tile_with_session(
                &self.inner,
                self.fast444_packet.as_deref(),
                self.fast422_packet.as_deref(),
                self.fast420_packet.as_deref(),
                session,
            ),
            routing::RouteDecision::CpuHost
            | routing::RouteDecision::RejectExplicitMetal { .. }
            | routing::RouteDecision::RejectUnsupportedBackend { .. }
            | routing::RouteDecision::MetalUnavailable => unreachable!("handled above"),
        }
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

    fn decode_region_scaled_to_device(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error> {
        Decoder::decode_region_scaled_to_device(self, fmt, roi, scale, backend)
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
        ctx: &mut signinum_core::DecoderContext<CpuDecoderContext>,
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
        ctx: &mut signinum_core::DecoderContext<CpuDecoderContext>,
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

    fn submit_region_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
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
                batch::BatchOp::RegionScaled { roi, scale },
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
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
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
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
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
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
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
        Codec::submit_tile_region_scaled_to_device(
            ctx, session, pool, input, fmt, roi, scale, backend,
        )
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
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
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
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
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

    fn decode_tile_region_scaled_to_device(
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::DeviceSurface, Self::Error> {
        Codec::decode_tile_region_scaled_to_device(ctx, pool, input, fmt, roi, scale, backend)
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

#[allow(clippy::unnecessary_wraps)]
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
    match op {
        batch::BatchOp::Full => match backend {
            BackendRequest::Cpu => decode_full_cpu_upload(decoder, pool, fmt),
            BackendRequest::Auto | BackendRequest::Metal => {
                let decision = choose_route(
                    decoder,
                    backend,
                    fmt,
                    op,
                    fast444_packet,
                    fast422_packet,
                    fast420_packet,
                );
                if let Some(err) = routing::decision_error(decision) {
                    return Err(err);
                }
                match decision {
                    routing::RouteDecision::CpuHost => decode_full_cpu_upload(decoder, pool, fmt),
                    routing::RouteDecision::MetalKernel => {
                        #[cfg(target_os = "macos")]
                        {
                            reject_cpu_staged_metal_upload(compute::decode_to_surface(
                                decoder,
                                pool,
                                fmt,
                                fast444_packet,
                                fast422_packet,
                                fast420_packet,
                            )?)
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            let _ = (
                                decoder,
                                pool,
                                fmt,
                                fast444_packet,
                                fast422_packet,
                                fast420_packet,
                            );
                            Err(Error::MetalUnavailable)
                        }
                    }
                    routing::RouteDecision::RejectExplicitMetal { .. }
                    | routing::RouteDecision::RejectUnsupportedBackend { .. }
                    | routing::RouteDecision::MetalUnavailable => unreachable!("handled above"),
                }
            }
            BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
        },
        batch::BatchOp::Region(roi) => match backend {
            BackendRequest::Cpu => decode_region_cpu_upload(decoder, pool, fmt, roi),
            BackendRequest::Auto | BackendRequest::Metal => {
                let decision = choose_route(
                    decoder,
                    backend,
                    fmt,
                    op,
                    fast444_packet,
                    fast422_packet,
                    fast420_packet,
                );
                if let Some(err) = routing::decision_error(decision) {
                    return Err(err);
                }
                match decision {
                    routing::RouteDecision::CpuHost => {
                        decode_region_cpu_upload(decoder, pool, fmt, roi)
                    }
                    routing::RouteDecision::MetalKernel => {
                        #[cfg(target_os = "macos")]
                        {
                            reject_cpu_staged_metal_upload(compute::decode_region_to_surface(
                                decoder,
                                pool,
                                fmt,
                                to_jpeg_rect(roi),
                                fast444_packet,
                                fast422_packet,
                                fast420_packet,
                            )?)
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            let _ = (
                                decoder,
                                pool,
                                fmt,
                                roi,
                                fast444_packet,
                                fast422_packet,
                                fast420_packet,
                            );
                            Err(Error::MetalUnavailable)
                        }
                    }
                    routing::RouteDecision::RejectExplicitMetal { .. }
                    | routing::RouteDecision::RejectUnsupportedBackend { .. }
                    | routing::RouteDecision::MetalUnavailable => unreachable!("handled above"),
                }
            }
            BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
        },
        batch::BatchOp::Scaled(scale) => match backend {
            BackendRequest::Cpu => decode_scaled_cpu_upload(decoder, pool, fmt, scale),
            BackendRequest::Auto | BackendRequest::Metal => {
                let decision = choose_route(
                    decoder,
                    backend,
                    fmt,
                    op,
                    fast444_packet,
                    fast422_packet,
                    fast420_packet,
                );
                if let Some(err) = routing::decision_error(decision) {
                    return Err(err);
                }
                match decision {
                    routing::RouteDecision::CpuHost => {
                        decode_scaled_cpu_upload(decoder, pool, fmt, scale)
                    }
                    routing::RouteDecision::MetalKernel => {
                        #[cfg(target_os = "macos")]
                        {
                            reject_cpu_staged_metal_upload(compute::decode_scaled_to_surface(
                                decoder,
                                pool,
                                fmt,
                                scale,
                                fast444_packet,
                                fast422_packet,
                                fast420_packet,
                            )?)
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            let _ = (
                                decoder,
                                pool,
                                fmt,
                                scale,
                                fast444_packet,
                                fast422_packet,
                                fast420_packet,
                            );
                            Err(Error::MetalUnavailable)
                        }
                    }
                    routing::RouteDecision::RejectExplicitMetal { .. }
                    | routing::RouteDecision::RejectUnsupportedBackend { .. }
                    | routing::RouteDecision::MetalUnavailable => unreachable!("handled above"),
                }
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

fn decode_full_cpu_upload(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let dims = decoder.info().dimensions;
    let stride = dims.0 as usize * fmt.bytes_per_pixel();
    let mut out = vec![0u8; stride * dims.1 as usize];
    decoder.decode_into_with_scratch(pool, &mut out, stride, fmt)?;
    upload_surface(out, dims, fmt, BackendRequest::Cpu)
}

fn decode_region_cpu_upload(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<Surface, Error> {
    let dims = (roi.w, roi.h);
    let stride = dims.0 as usize * fmt.bytes_per_pixel();
    let mut out = vec![0u8; stride * dims.1 as usize];
    decoder.decode_region_into_with_scratch(pool, &mut out, stride, fmt, to_jpeg_rect(roi))?;
    upload_surface(out, dims, fmt, BackendRequest::Cpu)
}

fn decode_scaled_cpu_upload(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    fmt: PixelFormat,
    scale: Downscale,
) -> Result<Surface, Error> {
    let dims = scaled_dims(decoder.info().dimensions, scale);
    let stride = dims.0 as usize * fmt.bytes_per_pixel();
    let mut out = vec![0u8; stride * dims.1 as usize];
    decoder.decode_scaled_into_with_scratch(pool, &mut out, stride, fmt, scale)?;
    upload_surface(out, dims, fmt, BackendRequest::Cpu)
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
            decode_region_scaled_cpu_upload(decoder, pool, fmt, roi, scale, BackendRequest::Cpu)
        }
        BackendRequest::Auto | BackendRequest::Metal => {
            let decision = choose_route(
                decoder,
                backend,
                fmt,
                batch::BatchOp::RegionScaled { roi, scale },
                fast444_packet,
                fast422_packet,
                fast420_packet,
            );
            if let Some(err) = routing::decision_error(decision) {
                return Err(err);
            }
            match decision {
                routing::RouteDecision::CpuHost => decode_region_scaled_cpu_upload(
                    decoder,
                    pool,
                    fmt,
                    roi,
                    scale,
                    BackendRequest::Cpu,
                ),
                routing::RouteDecision::MetalKernel => {
                    #[cfg(target_os = "macos")]
                    {
                        reject_cpu_staged_metal_upload(compute::decode_region_scaled_to_surface(
                            decoder,
                            pool,
                            fmt,
                            to_jpeg_rect(roi),
                            scale,
                            fast444_packet,
                            fast422_packet,
                            fast420_packet,
                        )?)
                    }
                    #[cfg(not(target_os = "macos"))]
                    {
                        let _ = (
                            decoder,
                            pool,
                            fmt,
                            roi,
                            scale,
                            fast444_packet,
                            fast422_packet,
                            fast420_packet,
                        );
                        Err(Error::MetalUnavailable)
                    }
                }
                routing::RouteDecision::RejectExplicitMetal { .. }
                | routing::RouteDecision::RejectUnsupportedBackend { .. }
                | routing::RouteDecision::MetalUnavailable => unreachable!("handled above"),
            }
        }
        BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
    }
}

fn reject_cpu_staged_metal_upload(surface: Surface) -> Result<Surface, Error> {
    if surface.residency() == SurfaceResidency::CpuStagedMetalUpload {
        return Err(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal explicit device decode requires a direct resident Metal decode; use the CPU path for CPU-staged output",
        });
    }
    Ok(surface)
}

#[allow(clippy::too_many_arguments)]
fn choose_route(
    decoder: &CpuDecoder<'_>,
    backend: BackendRequest,
    fmt: PixelFormat,
    op: batch::BatchOp,
    fast444_packet: Option<&JpegMetalFast444PacketV1>,
    fast422_packet: Option<&JpegMetalFast422PacketV1>,
    fast420_packet: Option<&JpegMetalFast420PacketV1>,
) -> routing::RouteDecision {
    let capabilities = routing::JpegMetalCapabilities::for_request(
        decoder,
        fmt,
        op,
        fast444_packet,
        fast422_packet,
        fast420_packet,
    );
    routing::decide_route(backend, capabilities)
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
            residency: SurfaceResidency::Host,
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
                    residency: SurfaceResidency::CpuStagedMetalUpload,
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
                        residency: SurfaceResidency::Host,
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

pub use signinum_jpeg::{
    DecoderContext, Downscale as JpegDownscale, PixelFormat as JpegPixelFormat, ScratchPool,
};
pub use signinum_jpeg::{Info, Rect as JpegRectPublic};

#[cfg(test)]
mod tests {
    use super::*;
    use signinum_jpeg::adapter::{build_metal_fast420_packet, build_metal_fast444_packet};

    const BASELINE_420: &[u8] = include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg");
    const BASELINE_420_RESTART: &[u8] =
        include_bytes!("../fixtures/jpeg/baseline_420_restart_32x16.jpg");
    const BASELINE_444: &[u8] = include_bytes!("../fixtures/jpeg/baseline_444_8x8.jpg");
    #[cfg(not(target_os = "macos"))]
    const GRAYSCALE: &[u8] = include_bytes!("../fixtures/jpeg/grayscale_8x8.jpg");

    #[test]
    fn auto_route_prefers_cpu_host_for_nonrestart_packets() {
        let decoder_420 = CpuDecoder::new(BASELINE_420).expect("420 decoder");
        let packet_420 = build_metal_fast420_packet(BASELINE_420).expect("420 packet");
        assert_eq!(
            choose_route(
                &decoder_420,
                BackendRequest::Auto,
                PixelFormat::Rgb8,
                batch::BatchOp::Full,
                None,
                None,
                Some(&packet_420),
            ),
            routing::RouteDecision::CpuHost
        );

        let decoder_444 = CpuDecoder::new(BASELINE_444).expect("444 decoder");
        let packet_444 = build_metal_fast444_packet(BASELINE_444).expect("444 packet");
        assert_eq!(
            choose_route(
                &decoder_444,
                BackendRequest::Auto,
                PixelFormat::Rgb8,
                batch::BatchOp::Scaled(Downscale::Quarter),
                Some(&packet_444),
                None,
                None,
            ),
            routing::RouteDecision::CpuHost
        );
    }

    #[test]
    fn auto_route_keeps_small_single_restart_packets_on_cpu_host() {
        let decoder = CpuDecoder::new(BASELINE_420_RESTART).expect("restart decoder");
        let packet = build_metal_fast420_packet(BASELINE_420_RESTART).expect("restart packet");

        assert_eq!(
            choose_route(
                &decoder,
                BackendRequest::Auto,
                PixelFormat::Rgb8,
                batch::BatchOp::Full,
                None,
                None,
                Some(&packet)
            ),
            routing::RouteDecision::CpuHost
        );
        assert_eq!(
            choose_route(
                &decoder,
                BackendRequest::Auto,
                PixelFormat::Rgb8,
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
            routing::RouteDecision::CpuHost
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_backend_session_reuses_compiled_runtime() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        assert!(session.runtime.get().is_none());

        let mut first = Decoder::new(BASELINE_420).expect("first decoder");
        let first_surface = first
            .decode_to_device_with_session(PixelFormat::Rgb8, &session)
            .expect("first session decode");
        assert_eq!(
            first_surface.residency(),
            SurfaceResidency::MetalResidentDecode
        );
        let first_runtime = session
            .runtime
            .get()
            .and_then(|runtime| runtime.as_ref().ok())
            .map(std::ptr::from_ref::<compute::MetalRuntime>)
            .expect("session runtime after first decode");

        let mut second = Decoder::new(BASELINE_420).expect("second decoder");
        second
            .decode_to_device_with_session(PixelFormat::Rgb8, &session)
            .expect("second session decode");
        let second_runtime = session
            .runtime
            .get()
            .and_then(|runtime| runtime.as_ref().ok())
            .map(std::ptr::from_ref::<compute::MetalRuntime>)
            .expect("session runtime after second decode");

        assert_eq!(first_runtime, second_runtime);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn jpeg_device_decode_uses_private_internal_planes() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let mut decoder = Decoder::new(BASELINE_420).expect("decoder");

        compute::reset_jpeg_private_buffer_allocations_for_test();
        let surface = decoder
            .decode_to_device_with_session(PixelFormat::Rgb8, &session)
            .expect("resident JPEG Metal decode");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert!(
            compute::jpeg_private_buffer_allocations_for_test() > 0,
            "resident JPEG Metal decode should use Private internal planes"
        );
        let _ = surface.as_bytes();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn jpeg_private_rgb8_tile_uses_private_output_buffer() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let mut decoder = Decoder::new(BASELINE_420).expect("decoder");

        let tile = decoder
            .decode_private_rgb8_tile_with_session(&session)
            .expect("resident private JPEG Metal decode");

        assert_eq!(tile.dimensions, (16, 16));
        assert_eq!(tile.pixel_format, PixelFormat::Rgb8);
        assert_eq!(tile.pitch_bytes, 16 * PixelFormat::Rgb8.bytes_per_pixel());
        assert_eq!(tile.byte_offset, 0);
        assert_eq!(tile.buffer.storage_mode(), metal::MTLStorageMode::Private);
        assert!(tile.status_buffer.length() > 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn jpeg_gray_region_decode_uses_private_internal_planes() {
        let roi = Rect {
            x: 4,
            y: 4,
            w: 8,
            h: 8,
        };
        let mut expected_decoder = Decoder::new(BASELINE_420).expect("expected decoder");
        let mut expected = vec![0; roi.w as usize * roi.h as usize];
        expected_decoder
            .decode_region_into(
                &mut CpuScratchPool::new(),
                &mut expected,
                roi.w as usize,
                PixelFormat::Gray8,
                roi,
            )
            .expect("expected CPU region decode");

        let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
        compute::reset_jpeg_private_buffer_allocations_for_test();
        let surface = decoder
            .decode_region_to_device(PixelFormat::Gray8, roi, BackendRequest::Metal)
            .expect("resident JPEG Metal region decode");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert!(
            compute::jpeg_private_buffer_allocations_for_test() >= 3,
            "resident Gray8 region decode should keep decoded Y/Cb/Cr planes Private"
        );
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn uploaded_metal_surface_is_marked_cpu_staged() {
        let surface = upload_surface(
            vec![1, 2, 3],
            (1, 1),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
        )
        .expect("CPU staged Metal upload");

        assert_eq!(surface.residency(), SurfaceResidency::CpuStagedMetalUpload);
    }

    #[test]
    fn auto_route_prefers_cpu_host_for_region_scaled_even_with_restart_packets() {
        let decoder = CpuDecoder::new(BASELINE_420_RESTART).expect("restart decoder");
        let packet = build_metal_fast420_packet(BASELINE_420_RESTART).expect("restart packet");

        assert_eq!(
            choose_route(
                &decoder,
                BackendRequest::Auto,
                PixelFormat::Rgb8,
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
            routing::RouteDecision::CpuHost
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn session_decode_rejects_unsupported_shape_before_host_unavailability() {
        let mut decoder = Decoder::new(GRAYSCALE).expect("decoder");
        let session = MetalBackendSession::default();

        assert!(matches!(
            decoder.decode_to_device_with_session(PixelFormat::Gray8, &session),
            Err(Error::UnsupportedMetalRequest { .. })
        ));
    }
}
