// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{borrow::Cow, ops::Range, sync::Arc};

#[cfg(target_os = "macos")]
use crate::engine::completed_metal_buffer_bytes;
#[cfg(target_os = "macos")]
use crate::metal_types::Buffer;
use j2k_core::{
    copy_tight_pixels_to_strided_output, BackendKind, DeviceMemoryRange, DeviceSurface,
    PixelFormat, SurfaceMetadata, SurfaceResidency,
};
#[cfg(target_os = "macos")]
use j2k_metal_support::{MetalImageLayout, ResidentMetalImage};

#[cfg(target_os = "macos")]
use crate::error::metal_kernel_support_error;
use crate::Error;

mod readback;
#[cfg(all(test, target_os = "macos"))]
#[path = "surface/tests.rs"]
mod readback_regression_tests;

pub use self::readback::download_surfaces_packed;

#[derive(Clone)]
pub(crate) enum Storage {
    Host(Arc<Vec<u8>>),
    #[cfg(target_os = "macos")]
    Metal(ResidentMetalImage),
}

impl Storage {
    pub(crate) fn from_host(bytes: Vec<u8>) -> Self {
        Self::Host(Arc::new(bytes))
    }
}

#[derive(Clone)]
/// Decoded J2K image surface returned by the Metal backend.
pub struct Surface {
    pub(crate) backend: BackendKind,
    pub(crate) residency: SurfaceResidency,
    pub(crate) dimensions: (u32, u32),
    pub(crate) fmt: PixelFormat,
    pub(crate) pitch_bytes: usize,
    pub(crate) byte_offset: usize,
    pub(crate) storage: Storage,
}

impl Surface {
    fn metadata(&self) -> SurfaceMetadata {
        SurfaceMetadata::new(
            self.backend,
            self.residency,
            self.dimensions,
            self.fmt,
            self.pitch_bytes,
        )
        .with_byte_offset(self.byte_offset)
    }

    /// Current residency for the surface bytes.
    pub fn residency(&self) -> SurfaceResidency {
        self.residency
    }

    /// Number of bytes between consecutive rows.
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }

    fn checked_storage_range(&self, storage_len: usize) -> Result<Range<usize>, Error> {
        let len = self.byte_len();
        let end = self
            .byte_offset
            .checked_add(len)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal surface byte range overflows usize".to_string(),
            })?;
        if end > storage_len {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K Metal surface byte range {start}..{end} exceeds storage length {storage_len}",
                    start = self.byte_offset
                ),
            });
        }
        Ok(self.byte_offset..end)
    }

    fn storage_bytes(&self) -> Result<Cow<'_, [u8]>, Error> {
        match &self.storage {
            Storage::Host(bytes) => {
                let range = self.checked_storage_range(bytes.len())?;
                Ok(Cow::Borrowed(&bytes[range]))
            }
            #[cfg(target_os = "macos")]
            Storage::Metal(image) => {
                // SAFETY: A returned `Surface` represents a completed decode.
                // External access to the handle is unsafe and requires callers
                // to exclude overlapping mutation during this owned readback.
                match unsafe {
                    j2k_metal_support::checked_buffer_read_vec::<u8>(
                        image.raw_buffer(),
                        image.byte_offset(),
                        self.byte_len(),
                    )
                } {
                    Ok(bytes) => {
                        #[cfg(all(test, target_os = "macos"))]
                        record_download_temporary_allocation_for_test();
                        Ok(Cow::Owned(bytes))
                    }
                    Err(
                        error @ j2k_metal_support::MetalSupportError::BufferContentsUnavailable,
                    ) => Err(metal_kernel_support_error(
                        "J2K Metal surface buffer is not host-addressable",
                        error,
                    )),
                    Err(error) => Err(metal_kernel_support_error(
                        format!("J2K Metal surface byte range invalid: {error}"),
                        error,
                    )),
                }
            }
        }
    }

    /// Return the tightly packed surface bytes.
    ///
    /// Host-backed surfaces are borrowed without copying. Metal-backed surfaces
    /// are copied into owned storage. Synchronization, readback, and validated
    /// range failures are returned through the backend's typed error contract.
    pub fn as_bytes(&self) -> Result<Cow<'_, [u8]>, Error> {
        self.storage_bytes()
    }

    /// Copy the tightly packed surface into a caller-provided strided buffer.
    /// Completed CPU-visible Metal storage is copied directly into the caller's
    /// rows without an intermediate host allocation.
    pub fn download_into(&self, out: &mut [u8], stride: usize) -> Result<(), Error> {
        match &self.storage {
            Storage::Host(_) => {
                let storage = self.storage_bytes()?;
                copy_tight_pixels_to_strided_output(
                    storage.as_ref(),
                    self.dimensions,
                    self.fmt,
                    out,
                    stride,
                )
                .map_err(Error::from)
            }
            #[cfg(target_os = "macos")]
            Storage::Metal(image) => {
                // SAFETY: A returned `Surface` represents a completed decode,
                // and safe APIs expose its resident allocation as immutable.
                let bytes = unsafe {
                    completed_metal_buffer_bytes(
                        image.raw_buffer(),
                        image.byte_offset(),
                        self.byte_len(),
                    )
                }
                .map_err(|error| match error {
                    error @ j2k_metal_support::MetalSupportError::BufferContentsUnavailable => {
                        metal_kernel_support_error(
                            "J2K Metal surface buffer is not host-addressable",
                            error,
                        )
                    }
                    error => metal_kernel_support_error(
                        format!("J2K Metal surface byte range invalid: {error}"),
                        error,
                    ),
                })?;
                copy_tight_pixels_to_strided_output(bytes, self.dimensions, self.fmt, out, stride)
                    .map_err(Error::from)
            }
        }
    }

    #[cfg(target_os = "macos")]
    /// Return the Metal buffer and byte offset when the surface is Metal-backed.
    ///
    /// # Safety
    ///
    /// All prior writers must have completed before this call. The caller must
    /// ensure that no CPU or GPU access through the returned handle (or a clone
    /// of it) mutates the surface range while this surface or any clone sharing
    /// the allocation remains alive.
    pub unsafe fn metal_buffer(
        &self,
    ) -> Option<(
        &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        usize,
    )> {
        self.metal_buffer_trusted()
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn metal_buffer_trusted(
        &self,
    ) -> Option<(
        &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        usize,
    )> {
        match &self.storage {
            Storage::Metal(image) => {
                // SAFETY: backend code binds this private handle under the
                // immutable resident-image contract.
                Some((unsafe { image.raw_buffer() }, image.byte_offset()))
            }
            Storage::Host(_) => None,
        }
    }

    #[cfg(target_os = "macos")]
    /// Borrow the immutable resident image when this surface is Metal-backed.
    pub fn resident_metal_image(&self) -> Option<&ResidentMetalImage> {
        match &self.storage {
            Storage::Metal(image) => Some(image),
            Storage::Host(_) => None,
        }
    }

    #[cfg(target_os = "macos")]
    /// Consume this surface and return its immutable resident image.
    pub fn into_resident_metal_image(self) -> Option<ResidentMetalImage> {
        match self.storage {
            Storage::Metal(image) => Some(image),
            Storage::Host(_) => None,
        }
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_metal_buffer(
        buffer: Buffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
    ) -> Result<Self, Error> {
        Self::from_metal_buffer_with_offset(buffer, dimensions, fmt, 0)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_metal_buffer_with_offset(
        buffer: Buffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
        byte_offset: usize,
    ) -> Result<Self, Error> {
        let pitch_bytes = usize::try_from(dimensions.0)
            .ok()
            .and_then(|width| width.checked_mul(fmt.bytes_per_pixel()))
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal surface pitch overflows usize".to_string(),
            })?;
        let layout = MetalImageLayout::new(byte_offset, dimensions, pitch_bytes, fmt)
            .map_err(|source| metal_kernel_support_error("J2K Metal surface layout", source))?;
        // SAFETY: surface constructors are crate-private and producer paths do
        // not expose the returned surface until their command buffer completes.
        // The producer command owns the only pending write, and later raw
        // access remains inside the audited backend.
        let image = unsafe { ResidentMetalImage::from_exclusive_pending_buffer(buffer, layout) }
            .map_err(|source| metal_kernel_support_error("J2K Metal resident surface", source))?;
        Ok(Self {
            backend: BackendKind::Metal,
            residency: SurfaceResidency::MetalResidentDecode,
            dimensions,
            fmt,
            pitch_bytes,
            byte_offset,
            storage: Storage::Metal(image),
        })
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_resident_metal_image(image: ResidentMetalImage) -> Self {
        let layout = image.layout();
        Self {
            backend: BackendKind::Metal,
            residency: SurfaceResidency::MetalResidentDecode,
            dimensions: layout.dimensions(),
            fmt: layout.pixel_format(),
            pitch_bytes: layout.pitch_bytes(),
            byte_offset: layout.byte_offset(),
            storage: Storage::Metal(image),
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
std::thread_local! {
    static DOWNLOAD_TEMPORARY_ALLOCATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(all(test, target_os = "macos"))]
fn reset_download_temporary_allocations_for_test() {
    DOWNLOAD_TEMPORARY_ALLOCATIONS.with(|count| count.set(0));
}

#[cfg(all(test, target_os = "macos"))]
fn record_download_temporary_allocation_for_test() {
    DOWNLOAD_TEMPORARY_ALLOCATIONS.with(|count| count.set(count.get().saturating_add(1)));
}

#[cfg(all(test, target_os = "macos"))]
fn download_temporary_allocations_for_test() -> usize {
    DOWNLOAD_TEMPORARY_ALLOCATIONS.with(std::cell::Cell::get)
}

#[doc(hidden)]
impl DeviceSurface for Surface {
    fn backend_kind(&self) -> BackendKind {
        self.metadata().backend
    }

    fn residency(&self) -> SurfaceResidency {
        self.metadata().residency
    }

    fn dimensions(&self) -> (u32, u32) {
        self.metadata().dimensions
    }

    fn pixel_format(&self) -> PixelFormat {
        self.metadata().pixel_format
    }

    fn byte_len(&self) -> usize {
        self.metadata().byte_len()
    }

    fn memory_range(&self) -> Option<DeviceMemoryRange> {
        match &self.storage {
            Storage::Host(_) => None,
            #[cfg(target_os = "macos")]
            Storage::Metal(image) => Some(DeviceMemoryRange::new(
                BackendKind::Metal,
                // SAFETY: reading the handle identity does not access or mutate
                // the allocation, and this surface retains the resident owner.
                u64::try_from(core::ptr::from_ref(unsafe { image.raw_buffer() }).addr()).ok()?,
                image.byte_offset(),
                self.byte_len(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Storage, Surface};
    use crate::Error;
    use j2k_core::{BackendKind, PixelFormat, SurfaceResidency};

    fn host_surface(bytes: Vec<u8>, byte_offset: usize) -> Surface {
        Surface {
            backend: BackendKind::Cpu,
            residency: SurfaceResidency::Host,
            dimensions: (2, 1),
            fmt: PixelFormat::Gray8,
            pitch_bytes: 2,
            byte_offset,
            storage: Storage::from_host(bytes),
        }
    }

    #[test]
    fn host_backed_byte_access_borrows_the_validated_range() {
        let surface = host_surface(vec![9, 1, 2, 8], 1);
        let bytes = surface.as_bytes().expect("valid host surface bytes");

        assert!(matches!(bytes, std::borrow::Cow::Borrowed(_)));
        assert_eq!(bytes.as_ref(), [1, 2]);
    }

    #[test]
    fn invalid_host_backed_range_returns_an_error_without_panicking() {
        let surface = host_surface(vec![1, 2], 1);
        let error = surface
            .as_bytes()
            .expect_err("out-of-range host surface must fail");

        assert!(matches!(error, Error::MetalKernel { .. }));
    }

    #[test]
    fn cloning_a_host_surface_shares_the_pixel_owner() {
        let surface = host_surface(vec![1, 2], 0);
        let cloned = surface.clone();

        match (&surface.storage, &cloned.storage) {
            (Storage::Host(original), Storage::Host(clone)) => {
                assert!(std::sync::Arc::ptr_eq(original, clone));
            }
            #[cfg(target_os = "macos")]
            _ => panic!("host surface clone must preserve host storage"),
        }
    }
}
