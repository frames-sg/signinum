// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use std::mem::{size_of, size_of_val};

use crate::metal_types::{Buffer, BufferRef, Device};
use j2k_core::accelerator::GpuAbi;
use j2k_metal_support::{
    checked_buffer_fill_bytes, checked_buffer_read as support_checked_buffer_read,
    checked_buffer_read_vec, checked_buffer_write, checked_private_buffer, checked_shared_buffer,
    checked_shared_buffer_with_slice, MetalSupportError,
};

use crate::{error::metal_kernel_support_error, Error};

use super::abi::J2K_CLASSIC_MAX_COEFF_COUNT;
use super::{
    direct_scratch::{take_recyclable_shared_buffer, DirectScratchBuffer},
    MetalRuntime,
};

fn buffer_access_error(context: &str, error: MetalSupportError) -> Error {
    metal_kernel_support_error(
        format!("J2K Metal {context} buffer access invalid: {error}"),
        error,
    )
}

fn buffer_allocation_error(error: MetalSupportError) -> Error {
    metal_kernel_support_error(
        format!("J2K Metal buffer allocation failed: {error}"),
        error,
    )
}

pub(super) fn new_shared_buffer(device: &Device, bytes: usize) -> Result<Buffer, Error> {
    checked_shared_buffer(device, bytes).map_err(buffer_allocation_error)
}

pub(super) fn new_private_buffer(device: &Device, bytes: usize) -> Result<Buffer, Error> {
    checked_private_buffer(device, bytes).map_err(buffer_allocation_error)
}

pub(super) fn new_shared_buffer_with_slice<T: GpuAbi>(
    device: &Device,
    data: &[T],
) -> Result<Buffer, Error> {
    checked_shared_buffer_with_slice(device, data).map_err(buffer_allocation_error)
}

pub(crate) fn checked_buffer_read<T: GpuAbi>(buffer: &Buffer, context: &str) -> Result<T, Error> {
    // SAFETY: J2K readback helpers are called only for CPU-initialized buffers
    // or after the producing Metal command buffer has completed.
    unsafe { support_checked_buffer_read::<T>(buffer, 0) }
        .map_err(|error| buffer_access_error(context, error))
}

pub(crate) fn checked_buffer_slice<T: GpuAbi>(
    buffer: &Buffer,
    len: usize,
    context: &str,
) -> Result<Vec<T>, Error> {
    checked_buffer_slice_at(buffer, 0, len, context)
}

pub(crate) fn checked_buffer_slice_at<T: GpuAbi>(
    buffer: &Buffer,
    byte_offset: usize,
    len: usize,
    context: &str,
) -> Result<Vec<T>, Error> {
    // SAFETY: J2K readback helpers are called only for CPU-initialized buffers
    // or after the producing Metal command buffer has completed.
    let result = unsafe { checked_buffer_read_vec::<T>(buffer, byte_offset, len) }
        .map_err(|error| buffer_access_error(context, error));
    #[cfg(test)]
    if let Ok(values) = &result {
        if !values.is_empty() {
            crate::engine::test_counters::record_idwt_host_temporary_readback_vec(size_of_val(
                values.as_slice(),
            ));
        }
    }
    result
}

pub(crate) fn checked_buffer_copy_into<T: GpuAbi>(
    buffer: &Buffer,
    byte_offset: usize,
    output: &mut [T],
    context: &str,
) -> Result<(), Error> {
    let result = (|| {
        let buffer_len = buffer.length();
        let element_size = size_of::<T>();
        if element_size == 0 {
            return Err(MetalSupportError::BufferZeroSizedType { abi_name: T::NAME });
        }
        let byte_len =
            output
                .len()
                .checked_mul(element_size)
                .ok_or(MetalSupportError::BufferBounds {
                    offset_bytes: byte_offset,
                    byte_len: usize::MAX,
                    buffer_len,
                })?;
        let end = byte_offset
            .checked_add(byte_len)
            .ok_or(MetalSupportError::BufferBounds {
                offset_bytes: byte_offset,
                byte_len,
                buffer_len,
            })?;
        if end > buffer_len {
            return Err(MetalSupportError::BufferBounds {
                offset_bytes: byte_offset,
                byte_len,
                buffer_len,
            });
        }
        let align = core::mem::align_of::<T>();
        if !byte_offset.is_multiple_of(align) {
            return Err(MetalSupportError::BufferAlignment {
                offset_bytes: byte_offset,
                align,
            });
        }
        if output.is_empty() {
            return Ok(());
        }
        // SAFETY: These private readback callers use completed/synchronized,
        // immutable Metal storage and a separate, exclusively borrowed host
        // destination. Reuse the audited support read for storage and actual
        // pointer alignment, then the existing checked byte-borrow boundary.
        let bytes = unsafe {
            support_checked_buffer_read::<T>(buffer, byte_offset)?;
            completed_metal_buffer_bytes(buffer, byte_offset, byte_len)?
        };
        T::slice_as_bytes_mut(output).copy_from_slice(bytes);
        Ok(())
    })();
    result.map_err(|error| buffer_access_error(context, error))
}

#[cfg(target_os = "macos")]
/// Borrow a validated byte range from a completed CPU-visible Metal buffer.
///
/// # Safety
///
/// All Metal writers must have completed, and the returned range must remain
/// immutable for the lifetime of the borrow.
pub(crate) unsafe fn completed_metal_buffer_bytes(
    buffer: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    byte_offset: usize,
    byte_len: usize,
) -> Result<&[u8], j2k_metal_support::MetalSupportError> {
    let buffer_len = buffer.length();
    if byte_offset
        .checked_add(byte_len)
        .is_none_or(|end| end > buffer_len)
    {
        return Err(j2k_metal_support::MetalSupportError::BufferBounds {
            offset_bytes: byte_offset,
            byte_len,
            buffer_len,
        });
    }
    if byte_len == 0 {
        return Ok(&[]);
    }

    // Reuse the support boundary to validate that this buffer is CPU-visible.
    // SAFETY: The caller guarantees completion and immutability for this range.
    let _ = unsafe { j2k_metal_support::checked_buffer_read::<u8>(buffer, byte_offset) }?;
    let base = buffer.contents().as_ptr().cast::<u8>();
    // SAFETY: The full range was checked above, Metal returned CPU-visible
    // storage, and the caller guarantees no overlapping writer.
    Ok(unsafe { core::slice::from_raw_parts(base.add(byte_offset), byte_len) })
}

pub(crate) fn buffer_is_cpu_visible(buffer: &BufferRef) -> bool {
    matches!(
        buffer.storageMode(),
        objc2_metal::MTLStorageMode::Shared | objc2_metal::MTLStorageMode::Managed
    )
}

/// Copy caller-owned GPU ABI input into Metal-owned shared storage.
pub(super) fn copied_slice_buffer<T: GpuAbi>(device: &Device, data: &[T]) -> Result<Buffer, Error> {
    new_shared_buffer_with_slice(device, data)
}

pub(super) fn copied_recyclable_shared_slice_buffer<T: GpuAbi>(
    runtime: &MetalRuntime,
    data: &[T],
    recyclable_shared_buffers: &mut Vec<crate::buffer_pool::PooledBuffer>,
) -> Result<Buffer, Error> {
    let size = size_of_val(data).max(1);
    let buffer = take_recyclable_shared_buffer(runtime, size, recyclable_shared_buffers)?;
    // SAFETY: The recycled buffer is exclusively held by this preparation
    // path and has not yet been submitted to Metal.
    unsafe { checked_buffer_write(&buffer, 0, data) }
        .map_err(|error| buffer_access_error("recyclable upload", error))?;
    Ok(buffer)
}

pub(super) fn zeroed_shared_buffer(device: &Device, bytes: usize) -> Result<Buffer, Error> {
    // Keep zero-byte callers on the shared helper path instead of
    // early-returning with a bespoke placeholder.
    let bytes = bytes.max(1);
    let buffer = new_shared_buffer(device, bytes)?;
    // SAFETY: The new buffer has not been submitted to Metal and has no other
    // CPU access while it is initialized.
    unsafe { checked_buffer_fill_bytes(&buffer, 0, bytes, 0) }
        .map_err(|error| buffer_access_error("new shared buffer clear", error))?;
    Ok(buffer)
}

pub(super) fn zeroed_recyclable_shared_buffer(
    runtime: &MetalRuntime,
    bytes: usize,
    recyclable_shared_buffers: &mut Vec<crate::buffer_pool::PooledBuffer>,
) -> Result<Buffer, Error> {
    let bytes = bytes.max(1);
    let buffer = take_recyclable_shared_buffer(runtime, bytes, recyclable_shared_buffers)?;
    // SAFETY: The recycled buffer is exclusively held by this preparation
    // path and has not yet been resubmitted to Metal.
    unsafe { checked_buffer_fill_bytes(&buffer, 0, bytes, 0) }
        .map_err(|error| buffer_access_error("recyclable buffer clear", error))?;
    Ok(buffer)
}

fn classic_coefficients_scratch_bytes(job_count: usize) -> Result<usize, Error> {
    job_count
        .max(1)
        .checked_mul(J2K_CLASSIC_MAX_COEFF_COUNT)
        .and_then(|count| count.checked_mul(size_of::<u32>()))
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K coefficient scratch size overflow".to_string(),
        })
}

pub(super) fn take_classic_coefficients_scratch_buffer(
    runtime: &MetalRuntime,
    job_count: usize,
) -> Result<DirectScratchBuffer, Error> {
    let bytes = classic_coefficients_scratch_bytes(job_count)?;
    Ok(DirectScratchBuffer {
        buffer: runtime.take_private_buffer(bytes)?,
    })
}

fn classic_states_scratch_bytes(job_count: usize) -> Result<usize, Error> {
    job_count
        .max(1)
        .checked_mul(J2K_CLASSIC_MAX_COEFF_COUNT)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K MetalDirect states scratch overflow".to_string(),
        })
}

pub(super) fn take_classic_states_scratch_buffer(
    runtime: &MetalRuntime,
    job_count: usize,
) -> Result<DirectScratchBuffer, Error> {
    let bytes = classic_states_scratch_bytes(job_count)?;
    Ok(DirectScratchBuffer {
        buffer: runtime.take_private_buffer(bytes)?,
    })
}

#[cfg(test)]
#[path = "direct_buffers/performance.rs"]
mod performance;

#[cfg(test)]
mod tests {
    use j2k_metal_support::MetalSupportError;

    use super::{
        buffer_access_error, checked_buffer_copy_into, new_private_buffer,
        new_shared_buffer_with_slice,
    };
    use crate::Error;

    #[test]
    fn buffer_access_errors_keep_j2k_context() {
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
                if message.contains("J2K Metal status readback")
                    && message.contains("not aligned")
        ));
    }

    #[test]
    fn checked_buffer_copy_into_preserves_destination_outside_the_selected_slice() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        crate::engine::with_runtime(|runtime| {
            let source = [7.0_f32, -1.25, 2.5, -3.75, 7.0];
            let buffer = new_shared_buffer_with_slice(&runtime.device, &source)?;
            let mut output = [11.0_f32; 5];
            checked_buffer_copy_into(
                &buffer,
                core::mem::size_of::<f32>(),
                &mut output[1..4],
                "selected copy",
            )?;
            assert_eq!(output[0].to_bits(), 11.0_f32.to_bits());
            assert_eq!(output[4].to_bits(), 11.0_f32.to_bits());
            for (actual, expected) in output[1..4].iter().zip(&source[1..4]) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
            Ok(())
        })
        .expect("copy selected shared Metal buffer range");
    }

    #[test]
    fn checked_buffer_copy_into_rejects_invalid_ranges_without_changing_output() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        crate::engine::with_runtime(|runtime| {
            let buffer = new_shared_buffer_with_slice(&runtime.device, &[1.0_f32, 2.0])?;
            let mut output = [13.0_f32; 2];
            let alignment = checked_buffer_copy_into(&buffer, 1, &mut output[..1], "alignment")
                .expect_err("misaligned typed copy must fail");
            assert!(matches!(
                alignment,
                Error::MetalSupport {
                    source: MetalSupportError::BufferAlignment { .. },
                    ..
                }
            ));
            assert!(output
                .iter()
                .all(|value| value.to_bits() == 13.0_f32.to_bits()));

            let bounds = checked_buffer_copy_into(
                &buffer,
                core::mem::size_of::<f32>(),
                &mut output,
                "bounds",
            )
            .expect_err("out-of-bounds typed copy must fail");
            assert!(matches!(
                bounds,
                Error::MetalSupport {
                    source: MetalSupportError::BufferBounds { .. },
                    ..
                }
            ));
            assert!(output
                .iter()
                .all(|value| value.to_bits() == 13.0_f32.to_bits()));

            checked_buffer_copy_into::<f32>(
                &buffer,
                2 * core::mem::size_of::<f32>(),
                &mut [],
                "empty end",
            )?;
            let empty_unaligned =
                checked_buffer_copy_into::<f32>(&buffer, 1, &mut [], "empty unaligned")
                    .expect_err("empty typed copy still validates alignment");
            assert!(matches!(
                empty_unaligned,
                Error::MetalSupport {
                    source: MetalSupportError::BufferAlignment { .. },
                    ..
                }
            ));
            let past_end = checked_buffer_copy_into::<f32>(
                &buffer,
                3 * core::mem::size_of::<f32>(),
                &mut [],
                "empty past end",
            )
            .expect_err("empty copy past the buffer must fail");
            assert!(matches!(
                past_end,
                Error::MetalSupport {
                    source: MetalSupportError::BufferBounds { .. },
                    ..
                }
            ));

            let mut zero_sized_output = [[0_u8; 0]; 1];
            let zero_sized =
                checked_buffer_copy_into(&buffer, 0, &mut zero_sized_output, "zero-sized ABI")
                    .expect_err("zero-sized GPU ABI element must fail");
            assert!(matches!(
                zero_sized,
                Error::MetalSupport {
                    source: MetalSupportError::BufferZeroSizedType { .. },
                    ..
                }
            ));

            let private = new_private_buffer(&runtime.device, core::mem::size_of::<f32>())?;
            let private_error = checked_buffer_copy_into(&private, 0, &mut output[..1], "private")
                .expect_err("private Metal storage must not be copied on the CPU");
            assert!(matches!(private_error, Error::MetalSupport { .. }));
            assert!(output
                .iter()
                .all(|value| value.to_bits() == 13.0_f32.to_bits()));
            Ok(())
        })
        .expect("validate checked Metal buffer copy failures");
    }
}
