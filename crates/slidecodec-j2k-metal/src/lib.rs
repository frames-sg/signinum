// SPDX-License-Identifier: Apache-2.0

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

mod batch;
mod classic;
#[cfg(target_os = "macos")]
mod compute;
#[cfg(target_os = "macos")]
mod direct;
mod ht;
mod idwt;
mod mct;
mod store;

use core::convert::Infallible;
use std::sync::Arc;
#[cfg(target_os = "macos")]
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::{Mutex, OnceLock},
};

use slidecodec_core::{
    BackendKind, BackendRequest, BufferError, CodecError, DecodeOutcome, DeviceSubmission,
    DeviceSurface, Downscale, ImageCodec, ImageDecode, ImageDecodeDevice, ImageDecodeSubmit,
    PixelFormat, ReadySubmission, Rect, TileBatchDecodeDevice, TileBatchDecodeSubmit,
};
use slidecodec_j2k::{
    J2kContext as CpuJ2kContext, J2kDecoder as CpuDecoder, J2kError,
    J2kScratchPool as CpuJ2kScratchPool, J2kView,
    __private::device_plan::{DeviceDecodePlan, DeviceDecodeRequest},
};
#[cfg(target_os = "macos")]
use slidecodec_j2k_native::{
    DecodeSettings as NativeDecodeSettings, DecoderContext as NativeDecoderContext,
    Image as NativeImage, J2kDirectGrayscalePlan,
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

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct DirectGrayPlanCacheEntry {
    key: u64,
    plan: J2kDirectGrayscalePlan,
    prepared: Arc<crate::compute::PreparedDirectGrayscalePlan>,
}

#[cfg(target_os = "macos")]
static DIRECT_GRAY_PLAN_CACHE: OnceLock<Mutex<Option<DirectGrayPlanCacheEntry>>> = OnceLock::new();

pub struct Surface {
    backend: BackendKind,
    dimensions: (u32, u32),
    fmt: PixelFormat,
    pitch_bytes: usize,
    byte_offset: usize,
    storage: Storage,
}

impl Surface {
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }

    pub fn as_bytes(&self) -> &[u8] {
        match &self.storage {
            Storage::Host(bytes) => {
                let len = self.byte_len();
                &bytes[self.byte_offset..self.byte_offset + len]
            }
            #[cfg(target_os = "macos")]
            Storage::Metal(buffer) => {
                let len = self.byte_len();
                unsafe {
                    core::slice::from_raw_parts(
                        buffer.contents().cast::<u8>().add(self.byte_offset),
                        len,
                    )
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
            Storage::Metal(buffer) => Some((buffer, self.byte_offset)),
            Storage::Host(_) => None,
        }
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
            byte_offset: 0,
            storage: Storage::Metal(buffer),
        }
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_metal_buffer_with_offset(
        buffer: Buffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
        byte_offset: usize,
    ) -> Self {
        Self {
            backend: BackendKind::Metal,
            dimensions,
            fmt,
            pitch_bytes: dimensions.0 as usize * fmt.bytes_per_pixel(),
            byte_offset,
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

#[derive(Clone, Default)]
pub struct MetalSession {
    shared: batch::SharedSession,
}

impl MetalSession {
    pub fn submissions(&self) -> u64 {
        self.shared.0.lock().expect("J2K Metal session").submissions
    }

    fn record_submit(&mut self) {
        let mut session = self.shared.0.lock().expect("J2K Metal session");
        session.submissions = session.submissions.saturating_add(1);
    }
}

impl core::fmt::Debug for MetalSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalSession")
            .field("submissions", &self.submissions())
            .finish()
    }
}

/// Convenience wrapper for submitting a group of J2K/HTJ2K tiles to one
/// decoder session.
///
/// This is intentionally codec-scoped: callers own slide metadata, tile
/// coordinates, cache policy, and viewport decisions. The batch only preserves
/// submission order and lets compatible tile requests share the Metal session.
#[derive(Default)]
pub struct MetalTileBatch {
    session: MetalSession,
    submissions: Vec<batch::MetalSubmission>,
}

impl MetalTileBatch {
    /// Create an empty tile batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty tile batch with capacity for `capacity` submissions.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            submissions: Vec::with_capacity(capacity),
            ..Self::default()
        }
    }

    /// Number of queued tile requests.
    pub fn len(&self) -> usize {
        self.submissions.len()
    }

    /// Whether the batch has no queued tile requests.
    pub fn is_empty(&self) -> bool {
        self.submissions.is_empty()
    }

    /// Number of Metal session submissions already flushed.
    ///
    /// Queued requests normally do not increment this until `decode_all` waits
    /// on the first result.
    pub fn submissions(&self) -> u64 {
        self.session.submissions()
    }

    /// Queue a full-tile decode request, copying the compressed tile bytes into
    /// the batch.
    pub fn push_tile(
        &mut self,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<usize, Error> {
        self.push_shared_tile(Arc::<[u8]>::from(input), fmt, backend)
    }

    /// Queue a full-tile decode request backed by shared compressed tile bytes.
    pub fn push_shared_tile(
        &mut self,
        input: Arc<[u8]>,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<usize, Error> {
        let slot = self.submissions.len();
        let submission = batch::queue_tile_request_shared(
            &mut self.session,
            input,
            fmt,
            backend,
            batch::BatchOp::Full,
        );
        self.submissions.push(submission);
        Ok(slot)
    }

    /// Queue a region decode request, copying the compressed tile bytes into
    /// the batch.
    pub fn push_tile_region(
        &mut self,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<usize, Error> {
        self.push_shared_tile_region(Arc::<[u8]>::from(input), fmt, roi, backend)
    }

    /// Queue a region decode request backed by shared compressed tile bytes.
    pub fn push_shared_tile_region(
        &mut self,
        input: Arc<[u8]>,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<usize, Error> {
        let slot = self.submissions.len();
        let submission = batch::queue_tile_request_shared(
            &mut self.session,
            input,
            fmt,
            backend,
            batch::BatchOp::Region(roi),
        );
        self.submissions.push(submission);
        Ok(slot)
    }

    /// Queue a scaled decode request, copying the compressed tile bytes into
    /// the batch.
    pub fn push_tile_scaled(
        &mut self,
        input: &[u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<usize, Error> {
        self.push_shared_tile_scaled(Arc::<[u8]>::from(input), fmt, scale, backend)
    }

    /// Queue a scaled decode request backed by shared compressed tile bytes.
    pub fn push_shared_tile_scaled(
        &mut self,
        input: Arc<[u8]>,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<usize, Error> {
        let slot = self.submissions.len();
        let submission = batch::queue_tile_request_shared(
            &mut self.session,
            input,
            fmt,
            backend,
            batch::BatchOp::Scaled(scale),
        );
        self.submissions.push(submission);
        Ok(slot)
    }

    /// Decode all queued tile requests and return surfaces in submission order.
    pub fn decode_all(self) -> Result<Vec<Surface>, Error> {
        let mut surfaces = Vec::with_capacity(self.submissions.len());
        for submission in self.submissions {
            surfaces.push(submission.wait()?);
        }
        Ok(surfaces)
    }
}

pub struct J2kDecoder<'a> {
    inner: CpuDecoder<'a>,
    pool: CpuJ2kScratchPool,
    #[cfg(target_os = "macos")]
    native_image: Option<NativeImage<'a>>,
    #[cfg(target_os = "macos")]
    native_context: NativeDecoderContext<'a>,
    #[cfg(target_os = "macos")]
    native_direct_gray_plan: Option<J2kDirectGrayscalePlan>,
    #[cfg(target_os = "macos")]
    native_prepared_direct_gray_plan: Option<Arc<crate::compute::PreparedDirectGrayscalePlan>>,
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
            #[cfg(target_os = "macos")]
            native_direct_gray_plan: None,
            #[cfg(target_os = "macos")]
            native_prepared_direct_gray_plan: None,
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
            #[cfg(target_os = "macos")]
            native_direct_gray_plan: None,
            #[cfg(target_os = "macos")]
            native_prepared_direct_gray_plan: None,
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

    #[cfg(target_os = "macos")]
    fn ensure_prepared_direct_gray_plan(
        &mut self,
    ) -> Result<Option<Arc<crate::compute::PreparedDirectGrayscalePlan>>, Error> {
        let cache_key = direct_gray_plan_cache_key(self.inner.bytes());
        if self.native_prepared_direct_gray_plan.is_none() {
            if let Some((plan, prepared)) = cached_global_direct_gray_plan(cache_key) {
                self.native_direct_gray_plan = Some(plan);
                self.native_prepared_direct_gray_plan = Some(prepared);
            }
        }
        if self.native_prepared_direct_gray_plan.is_none() {
            self.ensure_native_image()?;
            let (Some(image), native_context) =
                (self.native_image.as_ref(), &mut self.native_context)
            else {
                return Err(Error::Decode(J2kError::Backend(
                    "native image cache missing".to_string(),
                )));
            };

            let plan = match image.build_direct_grayscale_plan_with_context(native_context) {
                Ok(plan) => plan,
                Err(error) if direct::is_unsupported_direct_plan_error(&error.to_string()) => {
                    return Ok(None);
                }
                Err(error) => {
                    return Err(Error::Decode(J2kError::Backend(format!(
                        "failed to build J2K MetalDirect grayscale plan: {error}"
                    ))));
                }
            };
            let prepared = Arc::new(crate::compute::prepare_direct_grayscale_plan(&plan)?);
            store_global_direct_gray_plan(cache_key, &plan, prepared.clone());
            self.native_direct_gray_plan = Some(plan);
            self.native_prepared_direct_gray_plan = Some(prepared);
        }

        Ok(self.native_prepared_direct_gray_plan.clone())
    }

    #[cfg(target_os = "macos")]
    fn decode_direct_to_surface(&mut self, fmt: PixelFormat) -> Result<Option<Surface>, Error> {
        let Some(plan) = self.ensure_prepared_direct_gray_plan()? else {
            return Ok(None);
        };
        Ok(Some(
            crate::compute::execute_prepared_direct_grayscale_plan(&plan, fmt)?,
        ))
    }

    #[cfg(target_os = "macos")]
    fn decode_full_to_metal_surface(&mut self, fmt: PixelFormat) -> Result<Surface, Error> {
        self.ensure_native_image()?;
        let (Some(image), native_context) = (self.native_image.as_ref(), &mut self.native_context)
        else {
            return Err(Error::Decode(J2kError::Backend(
                "native image cache missing".to_string(),
            )));
        };
        crate::compute::decode_image_to_surface(image, native_context, fmt)
    }

    #[cfg(target_os = "macos")]
    fn decode_repeated_grayscale_cpu_to_surfaces(
        &mut self,
        fmt: PixelFormat,
        count: usize,
    ) -> Result<Vec<Surface>, Error> {
        let mut surfaces = Vec::with_capacity(count);
        for _ in 0..count {
            surfaces.push(self.decode_to_cpu_surface(fmt)?);
        }
        Ok(surfaces)
    }

    #[cfg(target_os = "macos")]
    fn should_auto_use_direct_for_repeated(
        plan: &J2kDirectGrayscalePlan,
        fmt: PixelFormat,
        count: usize,
    ) -> bool {
        if !matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) || count == 0 {
            return false;
        }

        let max_dim = plan.dimensions.0.max(plan.dimensions.1);
        max_dim >= 1024 && count >= 16
    }

    #[cfg(target_os = "macos")]
    #[doc(hidden)]
    pub fn decode_repeated_grayscale_direct_to_device(
        &mut self,
        fmt: PixelFormat,
        count: usize,
    ) -> Result<Vec<Surface>, Error> {
        if count == 0 {
            return Ok(Vec::new());
        }
        if self.native_direct_gray_plan.is_none() {
            let cache_key = direct_gray_plan_cache_key(self.inner.bytes());
            if let Some((plan, prepared)) = cached_global_direct_gray_plan(cache_key) {
                self.native_direct_gray_plan = Some(plan);
                self.native_prepared_direct_gray_plan = Some(prepared);
            }
        }
        if self.native_direct_gray_plan.is_none() {
            self.ensure_native_image()?;
            let (Some(image), native_context) =
                (self.native_image.as_ref(), &mut self.native_context)
            else {
                return Err(Error::Decode(J2kError::Backend(
                    "native image cache missing".to_string(),
                )));
            };
            let cache_key = direct_gray_plan_cache_key(self.inner.bytes());
            let plan = image
                .build_direct_grayscale_plan_with_context(native_context)
                .map_err(|error| J2kError::Backend(error.to_string()))?;
            let prepared = Arc::new(crate::compute::prepare_direct_grayscale_plan(&plan)?);
            store_global_direct_gray_plan(cache_key, &plan, prepared.clone());
            self.native_direct_gray_plan = Some(plan);
            self.native_prepared_direct_gray_plan = Some(prepared);
        }
        let Some(plan) = self.native_prepared_direct_gray_plan.as_ref() else {
            return Ok(Vec::new());
        };
        crate::compute::execute_repeated_prepared_direct_grayscale_plan(plan, fmt, count)
    }

    #[cfg(target_os = "macos")]
    #[doc(hidden)]
    pub fn decode_repeated_grayscale_auto_to_device(
        &mut self,
        fmt: PixelFormat,
        count: usize,
    ) -> Result<Vec<Surface>, Error> {
        if count == 0 {
            return Ok(Vec::new());
        }
        if !matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
            return self.decode_repeated_grayscale_cpu_to_surfaces(fmt, count);
        }
        let dims = self.inner.info().dimensions;
        if dims.0.max(dims.1) < 1024 || count < 16 {
            return self.decode_repeated_grayscale_cpu_to_surfaces(fmt, count);
        }
        if self.native_direct_gray_plan.is_none() {
            let cache_key = direct_gray_plan_cache_key(self.inner.bytes());
            if let Some((plan, prepared)) = cached_global_direct_gray_plan(cache_key) {
                self.native_direct_gray_plan = Some(plan);
                self.native_prepared_direct_gray_plan = Some(prepared);
            }
        }
        if self.native_direct_gray_plan.is_none() {
            self.ensure_native_image()?;
            let (Some(image), native_context) =
                (self.native_image.as_ref(), &mut self.native_context)
            else {
                return Err(Error::Decode(J2kError::Backend(
                    "native image cache missing".to_string(),
                )));
            };
            let cache_key = direct_gray_plan_cache_key(self.inner.bytes());
            let Ok(plan) = image.build_direct_grayscale_plan_with_context(native_context) else {
                return self.decode_repeated_grayscale_cpu_to_surfaces(fmt, count);
            };
            let prepared = Arc::new(crate::compute::prepare_direct_grayscale_plan(&plan)?);
            store_global_direct_gray_plan(cache_key, &plan, prepared.clone());
            self.native_direct_gray_plan = Some(plan);
            self.native_prepared_direct_gray_plan = Some(prepared);
        }
        let Some(plan) = self.native_direct_gray_plan.as_ref() else {
            return self.decode_repeated_grayscale_cpu_to_surfaces(fmt, count);
        };
        if Self::should_auto_use_direct_for_repeated(plan, fmt, count) {
            let Some(prepared) = self.native_prepared_direct_gray_plan.as_ref() else {
                return self.decode_repeated_grayscale_cpu_to_surfaces(fmt, count);
            };
            crate::compute::execute_repeated_prepared_direct_grayscale_plan(prepared, fmt, count)
        } else {
            self.decode_repeated_grayscale_cpu_to_surfaces(fmt, count)
        }
    }

    fn decode_to_cpu_surface(&mut self, fmt: PixelFormat) -> Result<Surface, Error> {
        let dims = self.inner.info().dimensions;
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        self.inner
            .decode_into_with_scratch(&mut self.pool, &mut out, stride, fmt)?;
        upload_surface(out, dims, fmt, BackendRequest::Cpu)
    }

    fn decode_region_to_cpu_surface(
        &mut self,
        fmt: PixelFormat,
        plan: DeviceDecodePlan,
    ) -> Result<Surface, Error> {
        let dims = plan.output_dims();
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        self.inner
            .decode_region_into(&mut self.pool, &mut out, stride, fmt, plan.source_rect())?;
        upload_surface(out, dims, fmt, BackendRequest::Cpu)
    }

    fn decode_scaled_to_cpu_surface(
        &mut self,
        fmt: PixelFormat,
        scale: Downscale,
        plan: DeviceDecodePlan,
    ) -> Result<Surface, Error> {
        let dims = plan.output_dims();
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        self.inner
            .decode_scaled_into(&mut self.pool, &mut out, stride, fmt, scale)?;
        upload_surface(out, dims, fmt, BackendRequest::Cpu)
    }

    #[cfg(target_os = "macos")]
    fn decode_region_to_metal_surface(
        &mut self,
        fmt: PixelFormat,
        plan: DeviceDecodePlan,
    ) -> Result<Surface, Error> {
        self.ensure_native_image()?;
        let (Some(image), native_context) = (self.native_image.as_ref(), &mut self.native_context)
        else {
            return Err(Error::Decode(J2kError::Backend(
                "native image cache missing".to_string(),
            )));
        };
        crate::compute::decode_image_region_to_surface(
            image,
            native_context,
            fmt,
            plan.source_rect(),
        )
    }

    #[cfg(target_os = "macos")]
    fn decode_scaled_to_metal_surface(
        &mut self,
        fmt: PixelFormat,
        scale: Downscale,
        plan: DeviceDecodePlan,
    ) -> Result<Surface, Error> {
        crate::compute::decode_scaled_to_surface(self.inner.bytes(), plan.source_dims(), fmt, scale)
    }

    pub(crate) fn decode_to_surface_impl(
        &mut self,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        #[cfg(not(target_os = "macos"))]
        if matches!(backend, BackendRequest::Metal) {
            return Err(Error::MetalUnavailable);
        }
        match backend {
            BackendRequest::Cpu => self.decode_to_cpu_surface(fmt),
            BackendRequest::Auto => {
                #[cfg(target_os = "macos")]
                {
                    self.decode_to_cpu_surface(fmt)
                }
                #[cfg(not(target_os = "macos"))]
                {
                    self.decode_to_cpu_surface(fmt)
                }
            }
            BackendRequest::Metal => {
                #[cfg(target_os = "macos")]
                {
                    if let Some(surface) = self.decode_direct_to_surface(fmt)? {
                        Ok(surface)
                    } else {
                        self.decode_full_to_metal_surface(fmt)
                    }
                }
                #[cfg(not(target_os = "macos"))]
                {
                    Err(Error::MetalUnavailable)
                }
            }
            BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
        }
    }

    pub(crate) fn decode_region_to_surface_impl(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        let plan = DeviceDecodePlan::for_image(
            self.inner.info().dimensions,
            DeviceDecodeRequest::Region { roi },
        )?;
        #[cfg(not(target_os = "macos"))]
        if matches!(backend, BackendRequest::Metal) {
            return Err(Error::MetalUnavailable);
        }
        match backend {
            BackendRequest::Cpu => self.decode_region_to_cpu_surface(fmt, plan),
            BackendRequest::Auto => {
                #[cfg(target_os = "macos")]
                {
                    self.decode_region_to_cpu_surface(fmt, plan)
                }
                #[cfg(not(target_os = "macos"))]
                {
                    self.decode_region_to_cpu_surface(fmt, plan)
                }
            }
            BackendRequest::Metal => {
                #[cfg(target_os = "macos")]
                {
                    self.decode_region_to_metal_surface(fmt, plan)
                }
                #[cfg(not(target_os = "macos"))]
                unreachable!("Metal region path is gated above");
            }
            BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
        }
    }

    pub(crate) fn decode_scaled_to_surface_impl(
        &mut self,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        let plan = DeviceDecodePlan::for_image(
            self.inner.info().dimensions,
            DeviceDecodeRequest::Scaled { scale },
        )?;
        #[cfg(not(target_os = "macos"))]
        if matches!(backend, BackendRequest::Metal) {
            return Err(Error::MetalUnavailable);
        }
        match backend {
            BackendRequest::Cpu => self.decode_scaled_to_cpu_surface(fmt, scale, plan),
            BackendRequest::Auto => {
                #[cfg(target_os = "macos")]
                {
                    self.decode_scaled_to_cpu_surface(fmt, scale, plan)
                }
                #[cfg(not(target_os = "macos"))]
                {
                    self.decode_scaled_to_cpu_surface(fmt, scale, plan)
                }
            }
            BackendRequest::Metal => {
                #[cfg(target_os = "macos")]
                {
                    self.decode_scaled_to_metal_surface(fmt, scale, plan)
                }
                #[cfg(not(target_os = "macos"))]
                unreachable!("Metal scaled path is gated above");
            }
            BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
        }
    }
}

#[cfg(target_os = "macos")]
fn direct_gray_plan_cache_key(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

#[cfg(target_os = "macos")]
fn cached_global_direct_gray_plan(
    key: u64,
) -> Option<(
    J2kDirectGrayscalePlan,
    Arc<crate::compute::PreparedDirectGrayscalePlan>,
)> {
    let cache = DIRECT_GRAY_PLAN_CACHE.get_or_init(|| Mutex::new(None));
    let guard = cache.lock().ok()?;
    guard
        .as_ref()
        .and_then(|entry| (entry.key == key).then(|| (entry.plan.clone(), entry.prepared.clone())))
}

#[cfg(target_os = "macos")]
fn store_global_direct_gray_plan(
    key: u64,
    plan: &J2kDirectGrayscalePlan,
    prepared: Arc<crate::compute::PreparedDirectGrayscalePlan>,
) {
    let cache = DIRECT_GRAY_PLAN_CACHE.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = cache.lock() {
        *guard = Some(DirectGrayPlanCacheEntry {
            key,
            plan: plan.clone(),
            prepared,
        });
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_full_grayscale_batch_direct_to_device(
    inputs: &[Arc<[u8]>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    if !matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
        return Err(Error::MetalKernel {
            message: format!("J2K MetalDirect full grayscale batch does not support {fmt:?}"),
        });
    }

    let mut plans = Vec::with_capacity(inputs.len());
    for input in inputs {
        let mut decoder = J2kDecoder::new(input.as_ref())?;
        let Some(plan) = decoder.ensure_prepared_direct_gray_plan()? else {
            return Err(Error::MetalKernel {
                message: format!(
                    "explicit J2K MetalDirect batch currently supports full grayscale Gray8/Gray16 only; fmt={fmt:?}"
                ),
            });
        };
        plans.push(plan);
    }
    crate::compute::execute_prepared_direct_grayscale_plan_batch(&plans, fmt)
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
        Ok(batch::queue_tile_request(
            session,
            input,
            fmt,
            backend,
            batch::BatchOp::Full,
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
        let _ = (ctx, pool);
        Ok(batch::queue_tile_request(
            session,
            input,
            fmt,
            backend,
            batch::BatchOp::Region(roi),
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
        let _ = (ctx, pool);
        Ok(batch::queue_tile_request(
            session,
            input,
            fmt,
            backend,
            batch::BatchOp::Scaled(scale),
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
            byte_offset: 0,
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
                    byte_offset: 0,
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
                        byte_offset: 0,
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
