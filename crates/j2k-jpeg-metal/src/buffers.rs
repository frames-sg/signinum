// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::metal_types::{Buffer, BufferRef, DeviceRef};
use j2k_core::accelerator::GpuAbi;
use j2k_metal_support::{
    checked_buffer_fill_bytes, checked_buffer_read as support_checked_buffer_read,
    checked_buffer_read_vec, checked_buffer_write, checked_private_buffer, checked_shared_buffer,
    checked_shared_buffer_with_bytes, checked_shared_buffer_with_slice, MetalSupportError,
};
#[cfg(test)]
use std::cell::Cell;

use crate::{error::metal_kernel_support_error, Error};

#[cfg(test)]
std::thread_local! {
    static JPEG_PRIVATE_BUFFER_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static JPEG_SHARED_BUFFER_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_jpeg_private_buffer_allocations_for_test() {
    JPEG_PRIVATE_BUFFER_ALLOCATIONS.with(|allocations| allocations.set(0));
}

#[cfg(test)]
pub(crate) fn reset_jpeg_shared_buffer_allocations_for_test() {
    JPEG_SHARED_BUFFER_ALLOCATIONS.with(|allocations| allocations.set(0));
}

#[cfg(test)]
pub(crate) fn jpeg_private_buffer_allocations_for_test() -> usize {
    JPEG_PRIVATE_BUFFER_ALLOCATIONS.with(Cell::get)
}

#[cfg(test)]
pub(crate) fn jpeg_shared_buffer_allocations_for_test() -> usize {
    JPEG_SHARED_BUFFER_ALLOCATIONS.with(Cell::get)
}

pub(crate) fn new_shared_buffer(device: &DeviceRef, bytes: usize) -> Result<Buffer, Error> {
    #[cfg(test)]
    JPEG_SHARED_BUFFER_ALLOCATIONS.with(|allocations| allocations.set(allocations.get() + 1));
    checked_shared_buffer(device, bytes).map_err(buffer_allocation_error)
}

pub(crate) fn new_shared_buffer_with_data(
    device: &DeviceRef,
    bytes: &[u8],
) -> Result<Buffer, Error> {
    #[cfg(test)]
    JPEG_SHARED_BUFFER_ALLOCATIONS.with(|allocations| allocations.set(allocations.get() + 1));
    checked_shared_buffer_with_bytes(device, bytes).map_err(buffer_allocation_error)
}

pub(crate) fn new_shared_buffer_with_slice<T: GpuAbi>(
    device: &DeviceRef,
    values: &[T],
) -> Result<Buffer, Error> {
    #[cfg(test)]
    JPEG_SHARED_BUFFER_ALLOCATIONS.with(|allocations| allocations.set(allocations.get() + 1));
    checked_shared_buffer_with_slice(device, values).map_err(buffer_allocation_error)
}

pub(crate) fn new_private_buffer(device: &DeviceRef, bytes: usize) -> Result<Buffer, Error> {
    #[cfg(test)]
    JPEG_PRIVATE_BUFFER_ALLOCATIONS.with(|allocations| allocations.set(allocations.get() + 1));
    checked_private_buffer(device, bytes).map_err(buffer_allocation_error)
}

fn buffer_allocation_error(error: MetalSupportError) -> Error {
    metal_kernel_support_error(
        format!("JPEG Metal buffer allocation failed: {error}"),
        error,
    )
}

fn buffer_access_error(context: &str, error: MetalSupportError) -> Error {
    metal_kernel_support_error(
        format!("JPEG Metal {context} buffer access invalid: {error}"),
        error,
    )
}

fn buffer_readback_error(context: &str, error: MetalSupportError) -> Error {
    buffer_access_error(context, error)
}

pub(crate) fn checked_buffer_read<T: GpuAbi>(
    buffer: &BufferRef,
    context: &str,
) -> Result<T, Error> {
    // SAFETY: JPEG readback helpers are called only for CPU-initialized buffers
    // or after `commit_and_wait_jpeg` has completed the producing commands.
    unsafe { support_checked_buffer_read::<T>(buffer, 0) }
        .map_err(|error| buffer_access_error(context, error))
}

pub(crate) fn checked_buffer_slice<T: GpuAbi>(
    buffer: &BufferRef,
    len: usize,
    context: &str,
) -> Result<Vec<T>, Error> {
    checked_buffer_slice_at(buffer, 0, len, context)
}

pub(crate) fn checked_buffer_slice_at<T: GpuAbi>(
    buffer: &BufferRef,
    byte_offset: usize,
    len: usize,
    context: &str,
) -> Result<Vec<T>, Error> {
    // SAFETY: JPEG readback helpers are called only for CPU-initialized buffers
    // or after `commit_and_wait_jpeg` has completed the producing commands.
    unsafe { checked_buffer_read_vec::<T>(buffer, byte_offset, len) }
        .map_err(|error| buffer_readback_error(context, error))
}

pub(crate) fn checked_copy_bytes_to_buffer_at(
    buffer: &BufferRef,
    byte_offset: usize,
    bytes: &[u8],
    context: &str,
) -> Result<(), Error> {
    // SAFETY: Viewport-cache writes occur during CPU staging while the cached
    // buffer is not submitted to a Metal command buffer.
    unsafe { checked_buffer_write::<u8>(buffer, byte_offset, bytes) }
        .map_err(|error| buffer_access_error(context, error))
}

pub(crate) fn checked_fill_buffer_u8(
    buffer: &BufferRef,
    len: usize,
    value: u8,
    context: &str,
) -> Result<(), Error> {
    // SAFETY: Viewport-cache fills occur during CPU staging while the cached
    // buffer is not submitted to a Metal command buffer.
    unsafe { checked_buffer_fill_bytes(buffer, 0, len, value) }
        .map_err(|error| buffer_access_error(context, error))
}

pub(crate) fn new_decode_plane_buffer(
    device: &DeviceRef,
    bytes: usize,
    returned_publicly: bool,
) -> Result<Buffer, Error> {
    if returned_publicly {
        new_shared_buffer(device, bytes)
    } else {
        new_private_buffer(device, bytes)
    }
}

#[cfg(test)]
mod tests {
    use j2k_metal_support::{system_default_device, MetalSupportError};

    use super::{
        buffer_access_error, buffer_readback_error, checked_buffer_slice,
        jpeg_shared_buffer_allocations_for_test, reset_jpeg_shared_buffer_allocations_for_test,
        MetalBatchScratch,
    };
    use crate::Error;

    #[test]
    fn buffer_access_errors_keep_jpeg_context() {
        let error = buffer_access_error(
            "status readback",
            MetalSupportError::BufferAlignment {
                offset_bytes: 1,
                align: 4,
            },
        );
        assert!(matches!(
            error,
            Error::MetalSupport { message, source: MetalSupportError::BufferAlignment { .. } }
                if message.contains("JPEG Metal status readback")
                    && message.contains("not aligned")
        ));
    }

    #[test]
    fn readback_allocation_errors_keep_the_typed_element_count_without_fake_bytes() {
        let source = MetalSupportError::BufferReadbackAllocation {
            abi_name: "test status",
            element_count: usize::MAX,
        };
        let error = buffer_readback_error("status readback", source.clone());
        assert!(matches!(
            &error,
            Error::MetalSupport { source: stored, .. } if stored == &source
        ));
        assert!(error.to_string().contains("test status"));
        assert!(error.to_string().contains(&usize::MAX.to_string()));
        assert!(std::error::Error::source(&error).is_some());
    }

    #[test]
    fn shared_scratch_stages_slices_directly_and_reuses_capacity() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }

        let device = system_default_device().expect("Metal device");
        let mut scratch = MetalBatchScratch::default();
        reset_jpeg_shared_buffer_allocations_for_test();
        let first = scratch
            .shared_buffer_with_byte_slices(
                &device,
                "direct entropy staging test",
                5,
                [b"ab".as_slice(), b"cde".as_slice()],
            )
            .expect("first staging");
        assert_eq!(
            checked_buffer_slice::<u8>(&first, 5, "direct entropy staging test")
                .expect("staged bytes"),
            b"abcde"
        );
        let allocations = jpeg_shared_buffer_allocations_for_test();

        let second = scratch
            .shared_buffer_with_byte_slices(
                &device,
                "direct entropy staging test",
                4,
                [b"wxyz".as_slice()],
            )
            .expect("reused staging");

        assert!(core::ptr::eq(
            objc2::rc::Retained::as_ptr(&first),
            objc2::rc::Retained::as_ptr(&second)
        ));
        assert_eq!(jpeg_shared_buffer_allocations_for_test(), allocations);
        assert_eq!(
            checked_buffer_slice::<u8>(&second, 4, "reused entropy staging test")
                .expect("restaged bytes"),
            b"wxyz"
        );
    }

    #[test]
    fn shared_scratch_rejects_direct_staging_length_mismatch() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }

        let device = system_default_device().expect("Metal device");
        let mut scratch = MetalBatchScratch::default();
        let error = scratch
            .shared_buffer_with_byte_slices(
                &device,
                "direct entropy staging mismatch test",
                3,
                [b"ab".as_slice()],
            )
            .expect_err("short staging must fail");

        assert!(matches!(error, Error::MetalKernel { .. }));
    }

    #[test]
    fn shared_scratch_direct_staging_preserves_typed_buffer_limit() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }

        let device = system_default_device().expect("Metal device");
        let mut scratch = MetalBatchScratch::default();
        let requested = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES + 1;
        let error = scratch
            .shared_buffer_with_byte_slices(
                &device,
                "direct entropy staging limit test",
                requested,
                std::iter::empty(),
            )
            .expect_err("oversized staging must fail");

        assert!(matches!(
            error,
            Error::MetalSupport {
                source: MetalSupportError::BufferAllocationTooLarge {
                    requested: actual,
                    ..
                },
                ..
            } if actual == requested
        ));
    }
}

struct ReusablePrivateBuffer {
    key: &'static str,
    capacity: usize,
    buffer: Buffer,
}

struct ReusableSharedBuffer {
    key: &'static str,
    capacity: usize,
    buffer: Buffer,
}

#[derive(Default)]
pub(crate) struct MetalBatchScratch {
    private_buffers: Vec<ReusablePrivateBuffer>,
    shared_buffers: Vec<ReusableSharedBuffer>,
}

impl MetalBatchScratch {
    pub(crate) fn private_buffer(
        &mut self,
        device: &DeviceRef,
        key: &'static str,
        bytes: usize,
    ) -> Result<Buffer, Error> {
        let bytes = bytes.max(1);
        if let Some(entry) = self
            .private_buffers
            .iter()
            .find(|entry| entry.key == key && entry.capacity >= bytes)
        {
            return Ok(entry.buffer.clone());
        }

        let buffer = new_private_buffer(device, bytes)?;
        if let Some(entry) = self
            .private_buffers
            .iter_mut()
            .find(|entry| entry.key == key)
        {
            entry.capacity = bytes;
            entry.buffer = buffer.clone();
        } else {
            crate::batch_allocation::try_reserve_for_push(
                &mut self.private_buffers,
                "JPEG Metal private scratch metadata",
            )?;
            self.private_buffers.push(ReusablePrivateBuffer {
                key,
                capacity: bytes,
                buffer: buffer.clone(),
            });
        }
        Ok(buffer)
    }

    fn shared_buffer(
        &mut self,
        device: &DeviceRef,
        key: &'static str,
        bytes: usize,
    ) -> Result<Buffer, Error> {
        let capacity = bytes.max(1);
        let buffer = if let Some(entry) = self
            .shared_buffers
            .iter()
            .find(|entry| entry.key == key && entry.capacity >= capacity)
        {
            entry.buffer.clone()
        } else {
            let buffer = new_shared_buffer(device, capacity)?;
            if let Some(entry) = self
                .shared_buffers
                .iter_mut()
                .find(|entry| entry.key == key)
            {
                entry.capacity = capacity;
                entry.buffer = buffer.clone();
            } else {
                crate::batch_allocation::try_reserve_for_push(
                    &mut self.shared_buffers,
                    "JPEG Metal shared scratch metadata",
                )?;
                self.shared_buffers.push(ReusableSharedBuffer {
                    key,
                    capacity,
                    buffer: buffer.clone(),
                });
            }
            buffer
        };

        Ok(buffer)
    }

    pub(crate) fn shared_buffer_with_bytes(
        &mut self,
        device: &DeviceRef,
        key: &'static str,
        bytes: &[u8],
    ) -> Result<Buffer, Error> {
        let buffer = self.shared_buffer(device, key, bytes.len())?;

        if !bytes.is_empty() {
            // SAFETY: This scratch buffer is exclusively leased during CPU
            // initialization and has not yet been submitted to Metal.
            unsafe { checked_buffer_write::<u8>(&buffer, 0, bytes) }
                .map_err(|error| buffer_access_error("shared scratch upload", error))?;
        }
        Ok(buffer)
    }

    pub(crate) fn shared_buffer_with_byte_slices<'a>(
        &mut self,
        device: &DeviceRef,
        key: &'static str,
        total_bytes: usize,
        slices: impl IntoIterator<Item = &'a [u8]>,
    ) -> Result<Buffer, Error> {
        let buffer = self.shared_buffer(device, key, total_bytes)?;
        let mut offset = 0_usize;
        for bytes in slices {
            let end = offset
                .checked_add(bytes.len())
                .ok_or_else(|| Error::MetalKernel {
                    message: "JPEG Metal shared scratch staging length overflowed".to_string(),
                })?;
            if end > total_bytes {
                return Err(Error::MetalKernel {
                    message: "JPEG Metal shared scratch staging length mismatch".to_string(),
                });
            }
            if !bytes.is_empty() {
                // SAFETY: This scratch buffer is exclusively leased during CPU
                // initialization and has not yet been submitted to Metal. Each
                // checked range is disjoint and lies below `total_bytes`.
                unsafe { checked_buffer_write::<u8>(&buffer, offset, bytes) }
                    .map_err(|error| buffer_access_error("shared scratch upload", error))?;
            }
            offset = end;
        }
        if offset != total_bytes {
            return Err(Error::MetalKernel {
                message: "JPEG Metal shared scratch staging length mismatch".to_string(),
            });
        }
        Ok(buffer)
    }

    pub(crate) fn shared_buffer_with_slice<T: GpuAbi>(
        &mut self,
        device: &DeviceRef,
        key: &'static str,
        values: &[T],
    ) -> Result<Buffer, Error> {
        let bytes = T::slice_as_bytes(values);
        self.shared_buffer_with_bytes(device, key, bytes)
    }
}
