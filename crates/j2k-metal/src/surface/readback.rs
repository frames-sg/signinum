// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

#[cfg(target_os = "macos")]
use super::completed_metal_buffer_bytes;
use super::Surface;
#[cfg(target_os = "macos")]
use crate::metal_types::Buffer;
use crate::Error;
#[cfg(target_os = "macos")]
use j2k_core::DeviceSurface as _;
#[cfg(target_os = "macos")]
use objc2_metal::MTLStorageMode;

#[cfg(all(test, target_os = "macos"))]
std::thread_local! {
    static PACKED_STAGING_COMMANDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static PACKED_STAGING_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(all(test, target_os = "macos"))]
pub(super) fn reset_packed_staging_for_test() {
    PACKED_STAGING_COMMANDS.with(|count| count.set(0));
    PACKED_STAGING_BYTES.with(|count| count.set(0));
}

#[cfg(all(test, target_os = "macos"))]
fn record_packed_staging_for_test(bytes: usize) {
    PACKED_STAGING_COMMANDS.with(|count| count.set(count.get().saturating_add(1)));
    PACKED_STAGING_BYTES.with(|count| count.set(count.get().saturating_add(bytes)));
}

#[cfg(all(test, target_os = "macos"))]
pub(super) fn packed_staging_for_test() -> (usize, usize) {
    (
        PACKED_STAGING_COMMANDS.with(std::cell::Cell::get),
        PACKED_STAGING_BYTES.with(std::cell::Cell::get),
    )
}

#[cfg(target_os = "macos")]
fn map_packed_support(error: &j2k_metal_support::MetalSupportError) -> Error {
    Error::MetalKernel {
        message: format!("J2K Metal packed surface readback failed: {error}"),
    }
}

#[cfg(target_os = "macos")]
fn is_cpu_visible(buffer: &crate::metal_types::BufferRef) -> bool {
    matches!(
        buffer.storageMode(),
        MTLStorageMode::Shared | MTLStorageMode::Managed
    )
}

#[cfg(target_os = "macos")]
fn preflight_packed_surfaces(
    session: &crate::MetalBackendSession,
    surfaces: &[&Surface],
) -> Result<(usize, usize), Error> {
    let mut total = 0usize;
    let mut private_bytes = 0usize;
    for surface in surfaces {
        let (buffer, source_offset) =
            surface
                .metal_buffer_trusted()
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal packed surface readback received a host surface"
                        .to_string(),
                })?;
        if !core::ptr::eq(buffer.device().as_ref(), session.device()) {
            return Err(Error::MetalKernel {
                message: "J2K Metal packed surface belongs to a different device".to_string(),
            });
        }
        let len = surface.byte_len();
        if source_offset
            .checked_add(len)
            .is_none_or(|end| end > buffer.length())
        {
            return Err(Error::MetalKernel {
                message: "Metal source copy range is out of bounds".to_string(),
            });
        }
        total = total.checked_add(len).ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal packed surface readback size overflow".to_string(),
        })?;
        if !is_cpu_visible(buffer) {
            private_bytes = private_bytes
                .checked_add(len)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal packed private staging size overflow".to_string(),
                })?;
        }
    }
    let allocation_cap = session
        .device()
        .maxBufferLength()
        .min(j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES);
    if total > allocation_cap {
        return Err(map_packed_support(
            &j2k_metal_support::MetalSupportError::BufferAllocationTooLarge {
                requested: total,
                cap: allocation_cap,
            },
        ));
    }
    Ok((total, private_bytes))
}

#[cfg(target_os = "macos")]
fn allocate_packed_output(total: usize) -> Result<Vec<u8>, Error> {
    let mut output = Vec::new();
    output.try_reserve_exact(total).map_err(|_| {
        map_packed_support(
            &j2k_metal_support::MetalSupportError::BufferReadbackAllocation {
                abi_name: <u8 as j2k_core::accelerator::GpuAbi>::NAME,
                element_count: total,
            },
        )
    })?;
    Ok(output)
}

#[cfg(target_os = "macos")]
fn stage_private_surfaces(
    session: &crate::MetalBackendSession,
    surfaces: &[&Surface],
    private_bytes: usize,
) -> Result<Option<Buffer>, Error> {
    use j2k_metal_support::{
        checked_blit_command_encoder, checked_command_buffer, checked_shared_buffer,
        commit_and_wait,
    };

    if private_bytes == 0 {
        return Ok(None);
    }
    #[cfg(all(test, target_os = "macos"))]
    record_packed_staging_for_test(private_bytes);
    let staging = checked_shared_buffer(session.device(), private_bytes)
        .map_err(|error| map_packed_support(&error))?;
    let runtime = session.runtime()?;
    let command = checked_command_buffer(runtime.command_queue())
        .map_err(|error| map_packed_support(&error))?;
    let blit =
        checked_blit_command_encoder(&command).map_err(|error| map_packed_support(&error))?;
    let mut staging_offset = 0usize;
    for surface in surfaces {
        let (buffer, source_offset) =
            surface
                .metal_buffer_trusted()
                .ok_or(Error::MetalStateInvariant {
                    state: "packed Metal readback",
                    reason: "preflighted surface lost its resident buffer",
                })?;
        if is_cpu_visible(buffer) {
            continue;
        }
        let len = surface.byte_len();
        if let Err(error) = blit.copy_from_buffer(
            buffer,
            source_offset as u64,
            &staging,
            staging_offset as u64,
            len as u64,
        ) {
            blit.endEncoding();
            return Err(error);
        }
        staging_offset += len;
    }
    blit.endEncoding();
    commit_and_wait(&command).map_err(|error| map_packed_support(&error))?;
    Ok(Some(staging))
}

#[cfg(target_os = "macos")]
fn copy_packed_surfaces(
    surfaces: &[&Surface],
    staging: Option<&crate::metal_types::BufferRef>,
    output: &mut Vec<u8>,
) -> Result<(), Error> {
    let mut staging_offset = 0usize;
    for surface in surfaces {
        let (buffer, source_offset) =
            surface
                .metal_buffer_trusted()
                .ok_or(Error::MetalStateInvariant {
                    state: "packed Metal readback",
                    reason: "preflighted surface lost its resident buffer",
                })?;
        let len = surface.byte_len();
        let source = if is_cpu_visible(buffer) {
            // SAFETY: Surfaces represent completed immutable decodes and the
            // source range was fully validated during preflight.
            unsafe { completed_metal_buffer_bytes(buffer, source_offset, len) }
                .map_err(|error| map_packed_support(&error))?
        } else {
            let staging = staging.ok_or(Error::MetalStateInvariant {
                state: "packed Metal readback",
                reason: "private surface has no completed staging buffer",
            })?;
            // SAFETY: The staging blit completed and no writer overlaps this read.
            let bytes = unsafe { completed_metal_buffer_bytes(staging, staging_offset, len) }
                .map_err(|error| map_packed_support(&error))?;
            staging_offset += len;
            bytes
        };
        // Preflight reserved the full result; extend initializes each byte once.
        output.extend_from_slice(source);
    }
    Ok(())
}

/// Read completed Metal-resident surfaces into one tightly packed host
/// allocation.
///
/// CPU-visible surfaces are copied directly. Private ranges share one compact
/// staging buffer submitted on the session's command queue. Surface order is
/// preserved. Every surface must belong to the supplied session's device; host
/// surfaces are rejected rather than copied.
#[cfg(target_os = "macos")]
#[doc(hidden)]
pub fn download_surfaces_packed(
    session: &crate::MetalBackendSession,
    surfaces: &[&Surface],
) -> Result<Vec<u8>, Error> {
    let (total, private_bytes) = preflight_packed_surfaces(session, surfaces)?;
    let mut output = allocate_packed_output(total)?;
    if total == 0 {
        return Ok(output);
    }
    let staging = stage_private_surfaces(session, surfaces, private_bytes)?;
    copy_packed_surfaces(surfaces, staging.as_deref(), &mut output)?;
    Ok(output)
}

/// Return `MetalUnavailable` on platforms without Metal support.
#[cfg(not(target_os = "macos"))]
#[doc(hidden)]
pub fn download_surfaces_packed(
    _session: &crate::MetalBackendSession,
    _surfaces: &[&Surface],
) -> Result<Vec<u8>, Error> {
    Err(Error::MetalUnavailable)
}
