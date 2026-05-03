// SPDX-License-Identifier: Apache-2.0

//! CUDA-facing device-output adapter for `signinum-j2k`.
//!
//! This crate intentionally exposes the same backend-selection surface as the
//! Metal adapter. CPU and auto requests return host-backed surfaces, while
//! explicit CUDA requests upload decoded output into CUDA device memory when
//! the `cuda-runtime` feature and a CUDA driver are available.

#![warn(unreachable_pub)]

use core::convert::Infallible;

use signinum_core::{
    BackendKind, BackendRequest, BufferError, CodecError, DecodeOutcome, DeviceSubmission,
    DeviceSurface, Downscale, ImageCodec, ImageDecode, ImageDecodeDevice, ImageDecodeSubmit,
    PixelFormat, ReadySubmission, Rect, TileBatchDecode, TileBatchDecodeDevice,
    TileBatchDecodeSubmit,
};
#[cfg(feature = "cuda-runtime")]
use signinum_cuda_runtime::{CudaContext, CudaDeviceBuffer, CudaDwt53Output, CudaError};
use signinum_j2k::{
    adapter::device_plan::{DeviceDecodePlan, DeviceDecodeRequest},
    J2kCodec as CpuCodec, J2kContext as CpuJ2kContext, J2kDecoder as CpuDecoder, J2kError,
    J2kScratchPool as CpuJ2kScratchPool, J2kView,
};
use signinum_j2k_native::{
    EncodedJ2kCodeBlock, J2kEncodeDispatchReport, J2kEncodeStageAccelerator, J2kForwardDwt53Job,
    J2kForwardDwt53Output, J2kForwardRctJob, J2kPacketizationEncodeJob, J2kTier1CodeBlockEncodeJob,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Decode(#[from] J2kError),
    #[error(transparent)]
    Buffer(#[from] BufferError),
    #[error("backend request {request:?} is not supported by signinum-j2k-cuda")]
    UnsupportedBackend { request: BackendRequest },
    #[error("CUDA is unavailable on this host")]
    CudaUnavailable,
    #[cfg(feature = "cuda-runtime")]
    #[error("CUDA runtime error: {message}")]
    CudaRuntime { message: String },
}

#[derive(Debug, Default, Clone)]
pub struct CudaEncodeStageAccelerator {
    #[cfg(feature = "cuda-runtime")]
    context: Option<CudaContext>,
    forward_rct_attempts: usize,
    forward_dwt53_attempts: usize,
    tier1_code_block_attempts: usize,
    packetization_attempts: usize,
    forward_rct_dispatches: usize,
    forward_dwt53_dispatches: usize,
    tier1_code_block_dispatches: usize,
    packetization_dispatches: usize,
}

impl CudaEncodeStageAccelerator {
    #[cfg(feature = "cuda-runtime")]
    fn cuda_context(&mut self) -> core::result::Result<Option<CudaContext>, &'static str> {
        if self.context.is_none() {
            match CudaContext::system_default() {
                Ok(context) => self.context = Some(context),
                Err(_) if cuda_runtime_required() => return Err("CUDA encode stage unavailable"),
                Err(_) => return Ok(None),
            }
        }
        Ok(self.context.clone())
    }

    pub fn forward_rct_attempts(&self) -> usize {
        self.forward_rct_attempts
    }

    pub fn forward_dwt53_attempts(&self) -> usize {
        self.forward_dwt53_attempts
    }

    pub fn tier1_code_block_attempts(&self) -> usize {
        self.tier1_code_block_attempts
    }

    pub fn packetization_attempts(&self) -> usize {
        self.packetization_attempts
    }

    pub fn forward_rct_dispatches(&self) -> usize {
        self.forward_rct_dispatches
    }

    pub fn forward_dwt53_dispatches(&self) -> usize {
        self.forward_dwt53_dispatches
    }

    pub fn tier1_code_block_dispatches(&self) -> usize {
        self.tier1_code_block_dispatches
    }

    pub fn packetization_dispatches(&self) -> usize {
        self.packetization_dispatches
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_runtime_required() -> bool {
    std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_some()
}

impl J2kEncodeStageAccelerator for CudaEncodeStageAccelerator {
    fn dispatch_report(&self) -> J2kEncodeDispatchReport {
        J2kEncodeDispatchReport {
            forward_rct: self.forward_rct_dispatches,
            forward_dwt53: self.forward_dwt53_dispatches,
            tier1_code_block: self.tier1_code_block_dispatches,
            packetization: self.packetization_dispatches,
        }
    }

    fn encode_forward_rct(
        &mut self,
        job: J2kForwardRctJob<'_>,
    ) -> core::result::Result<bool, &'static str> {
        self.forward_rct_attempts = self.forward_rct_attempts.saturating_add(1);
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            context
                .j2k_forward_rct(job.plane0, job.plane1, job.plane2)
                .map_err(|_| "CUDA forward RCT encode kernel failed")?;
            self.forward_rct_dispatches = self.forward_rct_dispatches.saturating_add(1);
            return Ok(true);
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        Ok(false)
    }

    fn encode_forward_dwt53(
        &mut self,
        job: J2kForwardDwt53Job<'_>,
    ) -> core::result::Result<Option<J2kForwardDwt53Output>, &'static str> {
        self.forward_dwt53_attempts = self.forward_dwt53_attempts.saturating_add(1);
        if job.num_levels == 0 {
            return Ok(None);
        }
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let output = context
                .j2k_forward_dwt53(job.samples, job.width, job.height, job.num_levels)
                .map_err(|_| "CUDA forward 5/3 DWT encode kernel failed")?;
            let dispatches = output.execution().kernel_dispatches();
            self.forward_dwt53_dispatches =
                self.forward_dwt53_dispatches.saturating_add(dispatches);
            return Ok(Some(cuda_dwt53_output_to_j2k(&output)?));
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        Ok(None)
    }

    fn encode_tier1_code_block(
        &mut self,
        _job: J2kTier1CodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<EncodedJ2kCodeBlock>, &'static str> {
        self.tier1_code_block_attempts = self.tier1_code_block_attempts.saturating_add(1);
        Ok(None)
    }

    fn encode_packetization(
        &mut self,
        _job: J2kPacketizationEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        self.packetization_attempts = self.packetization_attempts.saturating_add(1);
        Ok(None)
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_dwt53_output_to_j2k(
    output: &CudaDwt53Output,
) -> core::result::Result<J2kForwardDwt53Output, &'static str> {
    let (ll_width, ll_height) = output.ll_dimensions();
    let transformed = output.transformed();
    let full_width = output
        .levels()
        .first()
        .map_or(ll_width, |level| level.width) as usize;
    let mut ll = Vec::with_capacity((ll_width as usize) * (ll_height as usize));
    for y in 0..ll_height as usize {
        let row_start = y
            .checked_mul(full_width)
            .ok_or("CUDA DWT LL row offset overflow")?;
        ll.extend_from_slice(&transformed[row_start..row_start + ll_width as usize]);
    }

    let mut levels = Vec::with_capacity(output.levels().len());
    for shape in output.levels() {
        levels.push(signinum_j2k_native::J2kForwardDwt53Level {
            hl: extract_cuda_subband(
                transformed,
                full_width,
                shape.low_width,
                0,
                shape.high_width,
                shape.low_height,
            )?,
            lh: extract_cuda_subband(
                transformed,
                full_width,
                0,
                shape.low_height,
                shape.low_width,
                shape.high_height,
            )?,
            hh: extract_cuda_subband(
                transformed,
                full_width,
                shape.low_width,
                shape.low_height,
                shape.high_width,
                shape.high_height,
            )?,
            width: shape.width,
            height: shape.height,
            low_width: shape.low_width,
            low_height: shape.low_height,
            high_width: shape.high_width,
            high_height: shape.high_height,
        });
    }
    levels.reverse();

    Ok(J2kForwardDwt53Output {
        ll,
        ll_width,
        ll_height,
        levels,
    })
}

#[cfg(feature = "cuda-runtime")]
fn extract_cuda_subband(
    transformed: &[f32],
    full_width: usize,
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
) -> core::result::Result<Vec<f32>, &'static str> {
    let mut out = Vec::with_capacity((width as usize) * (height as usize));
    for y in 0..height as usize {
        let row_start = (y0 as usize)
            .checked_add(y)
            .and_then(|row| row.checked_mul(full_width))
            .and_then(|row| row.checked_add(x0 as usize))
            .ok_or("CUDA DWT subband offset overflow")?;
        out.extend_from_slice(&transformed[row_start..row_start + width as usize]);
    }
    Ok(out)
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

pub struct J2kDecoder<'a> {
    inner: CpuDecoder<'a>,
    pool: CpuJ2kScratchPool,
}

impl<'a> J2kDecoder<'a> {
    pub fn new(input: &'a [u8]) -> Result<Self, Error> {
        Ok(Self {
            inner: CpuDecoder::new(input)?,
            pool: CpuJ2kScratchPool::new(),
        })
    }

    fn decode_to_surface_impl(
        &mut self,
        session: &mut CudaSession,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        let dims = self.inner.info().dimensions;
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        self.inner
            .decode_into_with_scratch(&mut self.pool, &mut out, stride, fmt)?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    fn decode_region_to_surface_impl(
        &mut self,
        session: &mut CudaSession,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        let plan = DeviceDecodePlan::for_image(
            self.inner.info().dimensions,
            DeviceDecodeRequest::Region { roi },
        )?;
        let dims = plan.output_dims();
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        self.inner
            .decode_region_into(&mut self.pool, &mut out, stride, fmt, plan.source_rect())?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    fn decode_scaled_to_surface_impl(
        &mut self,
        session: &mut CudaSession,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        let dims = DeviceDecodePlan::for_image(
            self.inner.info().dimensions,
            DeviceDecodeRequest::Scaled { scale },
        )?
        .output_dims();
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        self.inner
            .decode_scaled_into(&mut self.pool, &mut out, stride, fmt, scale)?;
        wrap_surface(out, dims, fmt, backend, session)
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
        let plan = DeviceDecodePlan::for_image(
            self.inner.info().dimensions,
            DeviceDecodeRequest::RegionScaled { roi, scale },
        )?;
        let dims = plan.output_dims();
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        self.inner.decode_region_scaled_into(
            &mut self.pool,
            &mut out,
            stride,
            fmt,
            plan.source_rect(),
            scale,
        )?;
        wrap_surface(out, dims, fmt, backend, session)
    }
}

impl ImageCodec for J2kDecoder<'_> {
    type Error = Error;
    type Warning = Infallible;
    type Pool = CpuJ2kScratchPool;
}

impl<'a> ImageDecode<'a> for J2kDecoder<'a> {
    type View = J2kView<'a>;

    fn inspect(input: &'a [u8]) -> Result<signinum_core::Info, Self::Error> {
        Ok(CpuDecoder::inspect(input)?)
    }

    fn parse(input: &'a [u8]) -> Result<Self::View, Self::Error> {
        Ok(J2kView::parse(input)?)
    }

    fn from_view(view: Self::View) -> Result<Self, Self::Error> {
        Ok(Self {
            inner: CpuDecoder::from_view(view)?,
            pool: CpuJ2kScratchPool::new(),
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

    fn decode_region_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        Ok(self
            .inner
            .decode_region_scaled_into(pool, out, stride, fmt, roi, scale)?)
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
    type Warning = Infallible;
    type Pool = CpuJ2kScratchPool;
}

impl Codec {
    fn decode_tile_to_surface_impl(
        ctx: &mut signinum_core::DecoderContext<CpuJ2kContext>,
        session: &mut CudaSession,
        pool: &mut CpuJ2kScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        let dims = CpuDecoder::inspect(input)?.dimensions;
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        CpuCodec::decode_tile(ctx, pool, input, &mut out, stride, fmt)?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    fn decode_tile_region_to_surface_impl(
        ctx: &mut signinum_core::DecoderContext<CpuJ2kContext>,
        session: &mut CudaSession,
        pool: &mut CpuJ2kScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        let dims = DeviceDecodePlan::for_image(
            CpuDecoder::inspect(input)?.dimensions,
            DeviceDecodeRequest::Region { roi },
        )?
        .output_dims();
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        CpuCodec::decode_tile_region(ctx, pool, input, &mut out, stride, fmt, roi)?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    fn decode_tile_scaled_to_surface_impl(
        ctx: &mut signinum_core::DecoderContext<CpuJ2kContext>,
        session: &mut CudaSession,
        pool: &mut CpuJ2kScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        let dims = DeviceDecodePlan::for_image(
            CpuDecoder::inspect(input)?.dimensions,
            DeviceDecodeRequest::Scaled { scale },
        )?
        .output_dims();
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        CpuCodec::decode_tile_scaled(ctx, pool, input, &mut out, stride, fmt, scale)?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_tile_region_scaled_to_surface_impl(
        ctx: &mut signinum_core::DecoderContext<CpuJ2kContext>,
        session: &mut CudaSession,
        pool: &mut CpuJ2kScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        let dims = DeviceDecodePlan::for_image(
            CpuDecoder::inspect(input)?.dimensions,
            DeviceDecodeRequest::RegionScaled { roi, scale },
        )?
        .output_dims();
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        CpuCodec::decode_tile_region_scaled(ctx, pool, input, &mut out, stride, fmt, roi, scale)?;
        wrap_surface(out, dims, fmt, backend, session)
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
    type Context = CpuJ2kContext;
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
    type Context = CpuJ2kContext;
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

pub use signinum_j2k::{J2kContext, J2kScratchPool};

#[cfg(test)]
mod tests {
    use super::CudaEncodeStageAccelerator;
    use signinum_j2k_native::{encode_with_accelerator, DecodeSettings, EncodeOptions, Image};

    #[test]
    fn cuda_encode_stage_accelerator_preserves_cpu_codestream_validity() {
        let pixels: Vec<u8> = (0u8..192).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 8, 8, 3, 8, false, &options, &mut accelerator)
                .expect("encode with CUDA stage accelerator");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.num_components, 3);
        assert_eq!(decoded.bit_depth, 8);
        assert_eq!(accelerator.forward_rct_attempts(), 1);
        assert_eq!(accelerator.forward_dwt53_attempts(), 3);
        assert!(accelerator.tier1_code_block_attempts() > 0);
        assert_eq!(accelerator.packetization_attempts(), 1);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_forward_rct_dispatches_when_runtime_required() {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels: Vec<u8> = (0u16..7 * 5 * 3)
            .map(|i| u8::try_from((i * 17) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 7, 5, 3, 8, false, &options, &mut accelerator)
                .expect("encode with CUDA forward RCT");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(accelerator.forward_rct_attempts(), 1);
        assert_eq!(accelerator.forward_rct_dispatches(), 1);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_forward_dwt53_dispatches_when_runtime_required() {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels: Vec<u8> = (0u16..8 * 8)
            .map(|i| u8::try_from((i * 5) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 8, 8, 1, 8, false, &options, &mut accelerator)
                .expect("encode with CUDA forward DWT 5/3");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(accelerator.forward_dwt53_attempts(), 1);
        assert_eq!(accelerator.forward_dwt53_dispatches(), 2);
    }
}
