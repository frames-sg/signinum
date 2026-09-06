// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;
#[cfg(target_os = "macos")]
use j2k_core::{BackendKind, PixelFormat, SurfaceResidency};

#[cfg(target_os = "macos")]
use super::{
    download_surfaces_packed, download_temporary_allocations_for_test,
    readback::{packed_staging_for_test, reset_packed_staging_for_test},
    reset_download_temporary_allocations_for_test, Storage, Surface,
};

#[cfg(target_os = "macos")]
fn should_run_metal_runtime() -> bool {
    j2k_test_support::metal_runtime_gate(module_path!())
}

#[cfg(target_os = "macos")]
fn completed_surface(
    buffer: crate::metal_types::Buffer,
    byte_offset: usize,
    dimensions: (u32, u32),
    format: PixelFormat,
) -> Surface {
    let pitch_bytes = dimensions.0 as usize * format.bytes_per_pixel();
    let layout =
        j2k_metal_support::MetalImageLayout::new(byte_offset, dimensions, pitch_bytes, format)
            .expect("valid surface test layout");
    // SAFETY: Test callers move a completed buffer into this immutable image
    // and retain no writable alias to it.
    let image =
        unsafe { j2k_metal_support::ResidentMetalImage::from_completed_buffer(buffer, layout) }
            .expect("completed test surface");
    Surface::from_resident_metal_image(image)
}

#[cfg(target_os = "macos")]
fn shared_surface(
    session: &crate::MetalBackendSession,
    allocation: &[u8],
    byte_offset: usize,
    dimensions: (u32, u32),
    format: PixelFormat,
) -> Surface {
    let buffer = j2k_metal_support::checked_shared_buffer_with_bytes(session.device(), allocation)
        .expect("shared test surface allocation");
    completed_surface(buffer, byte_offset, dimensions, format)
}

#[cfg(target_os = "macos")]
fn private_surface(session: &crate::MetalBackendSession, bytes: &[u8]) -> Surface {
    use j2k_metal_support::{
        checked_blit_command_encoder, checked_command_buffer, checked_command_queue,
        checked_private_buffer, checked_shared_buffer_with_bytes, commit_and_wait,
    };

    let source = checked_shared_buffer_with_bytes(session.device(), bytes)
        .expect("private test source allocation");
    let destination =
        checked_private_buffer(session.device(), bytes.len()).expect("private test allocation");
    let queue = checked_command_queue(session.device()).expect("private test command queue");
    let command = checked_command_buffer(&queue).expect("private test command buffer");
    let blit = checked_blit_command_encoder(&command).expect("private test blit encoder");
    blit.copy_from_buffer(&source, 0, &destination, 0, bytes.len() as u64)
        .expect("copy test bytes into private storage");
    blit.endEncoding();
    commit_and_wait(&command).expect("complete private test upload");
    completed_surface(
        destination,
        0,
        (u32::try_from(bytes.len()).expect("small fixture length"), 1),
        PixelFormat::Gray8,
    )
}

#[cfg(target_os = "macos")]
#[test]
fn shared_subrange_downloads_directly_into_strided_destination() {
    if !should_run_metal_runtime() {
        return;
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal test session");
    let surface = shared_surface(
        &session,
        &[0xA0, 0xA1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0xB0],
        2,
        (2, 2),
        PixelFormat::Rgb8,
    );
    let mut output = [0xEE; 16];

    reset_download_temporary_allocations_for_test();
    surface
        .download_into(&mut output, 8)
        .expect("strided shared surface download");

    assert_eq!(
        output,
        [1, 2, 3, 4, 5, 6, 0xEE, 0xEE, 7, 8, 9, 10, 11, 12, 0xEE, 0xEE]
    );
    assert_eq!(download_temporary_allocations_for_test(), 0);
}

#[cfg(target_os = "macos")]
#[test]
fn private_surface_download_into_preserves_the_host_addressability_error() {
    if !should_run_metal_runtime() {
        return;
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal test session");
    let surface = private_surface(&session, &[1, 2, 3, 4]);
    let mut output = [0u8; 4];

    let error = surface
        .download_into(&mut output, 4)
        .expect_err("private surface cannot be copied through CPU-visible storage");

    assert!(matches!(
        error,
        crate::Error::MetalSupport { message,
            source: j2k_metal_support::MetalSupportError::BufferContentsUnavailable }
            if message.contains("surface buffer is not host-addressable")
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn packed_shared_subranges_avoid_metal_staging() {
    if !should_run_metal_runtime() {
        return;
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal test session");
    let first = shared_surface(
        &session,
        &[90, 1, 2, 3, 4, 91],
        1,
        (4, 1),
        PixelFormat::Gray8,
    );
    let second = shared_surface(
        &session,
        &[80, 81, 5, 6, 7, 8, 82],
        2,
        (4, 1),
        PixelFormat::Gray8,
    );

    reset_packed_staging_for_test();
    let output = download_surfaces_packed(&session, &[&first, &second])
        .expect("packed shared surface download");

    assert_eq!(output, [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(packed_staging_for_test(), (0, 0));
}

#[cfg(target_os = "macos")]
#[test]
fn packed_mixed_storage_stages_only_private_bytes() {
    if !should_run_metal_runtime() {
        return;
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal test session");
    let shared = shared_surface(
        &session,
        &[90, 1, 2, 3, 4, 91],
        1,
        (4, 1),
        PixelFormat::Gray8,
    );
    let private = private_surface(&session, &[5, 6, 7, 8]);

    reset_packed_staging_for_test();
    let output = download_surfaces_packed(&session, &[&shared, &private])
        .expect("packed mixed surface download");

    assert_eq!(output, [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(packed_staging_for_test(), (1, 4));
}

#[cfg(target_os = "macos")]
#[test]
fn empty_packed_download_avoids_staging() {
    if !should_run_metal_runtime() {
        return;
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal test session");

    reset_packed_staging_for_test();
    let output = download_surfaces_packed(&session, &[]).expect("empty packed download");

    assert!(output.is_empty());
    assert_eq!(packed_staging_for_test(), (0, 0));
}

#[cfg(target_os = "macos")]
#[test]
fn packed_host_rejection_happens_before_staging() {
    if !should_run_metal_runtime() {
        return;
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal test session");
    let host = Surface {
        backend: BackendKind::Cpu,
        residency: SurfaceResidency::Host,
        dimensions: (4, 1),
        fmt: PixelFormat::Gray8,
        pitch_bytes: 4,
        byte_offset: 0,
        storage: Storage::from_host(vec![1, 2, 3, 4]),
    };

    reset_packed_staging_for_test();
    let error = download_surfaces_packed(&session, &[&host])
        .expect_err("packed host surface must be rejected");

    assert!(
        matches!(error, crate::Error::MetalKernel { message } if message.contains("host surface"))
    );
    assert_eq!(packed_staging_for_test(), (0, 0));
}

#[cfg(target_os = "macos")]
#[test]
fn packed_shared_total_preserves_the_metal_allocation_cap() {
    if !should_run_metal_runtime() {
        return;
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal test session");
    let dimensions = (1024, 1024);
    let surface_bytes =
        dimensions.0 as usize * dimensions.1 as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let surface = shared_surface(
        &session,
        &vec![0; surface_bytes],
        0,
        dimensions,
        PixelFormat::Rgb8,
    );
    let cap = session
        .device()
        .maxBufferLength()
        .min(j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES);
    let surface_count = cap / surface_bytes + 1;
    let surfaces = vec![&surface; surface_count];
    let requested = surface_bytes * surface_count;

    reset_packed_staging_for_test();
    let error = download_surfaces_packed(&session, &surfaces)
        .map(|bytes| bytes.len())
        .expect_err("packed output larger than the prior Metal cap must be rejected");

    assert!(matches!(
        error,
        crate::Error::MetalKernel { message }
            if message.contains(&requested.to_string()) && message.contains(&cap.to_string())
    ));
    assert_eq!(packed_staging_for_test(), (0, 0));
}
