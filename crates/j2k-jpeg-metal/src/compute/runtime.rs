// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal device/session lifecycle and mutable runtime cache ownership.

use std::{
    cell::RefCell,
    sync::{Arc, Mutex, MutexGuard},
};

use super::pipeline_registry::JpegPipelineRegistry;
use super::scratch_pool::{BatchScratchLease, BatchScratchPool};
use super::viewport_cache::{
    CachedViewportPlanes, ViewportPlaneCacheGate, ViewportPlaneCacheLease,
};
use crate::error::{metal_runtime_support_error, Error};
use crate::metal_types::{Buffer, CommandBuffer, CommandQueue, Device};
use j2k_core::PixelFormat;
use j2k_metal_support::{checked_command_queue, system_default_device, MetalSupportError};

thread_local! {
    static DEFAULT_METAL_SESSION: RefCell<Option<Result<crate::MetalBackendSession, MetalSupportError>>> = const { RefCell::new(None) };
}

pub(crate) struct MetalRuntime {
    pub(in crate::compute) device: Device,
    pub(in crate::compute) queue: CommandQueue,
    pub(in crate::compute) pipelines: Arc<JpegPipelineRegistry>,
    batch_scratch: BatchScratchPool,
    viewport_plane_cache: Mutex<Option<CachedViewportPlanes>>,
    viewport_plane_cache_gate: Arc<ViewportPlaneCacheGate>,
}

// SAFETY: Metal devices, queues, and immutable pipeline states support
// cross-thread use. All mutable host-side caches are protected by mutexes, and
// each command encoder remains confined to the submission that creates it.
unsafe impl Send for MetalRuntime {}
// SAFETY: Shared runtime operations allocate independent command buffers;
// shared scratch/cache mutation is serialized by the corresponding mutex.
unsafe impl Sync for MetalRuntime {}

impl MetalRuntime {
    #[cfg(test)]
    pub(in crate::compute) fn new() -> Result<Self, MetalSupportError> {
        let device = system_default_device()?;
        Self::new_with_device(device)
    }

    pub(crate) fn new_with_device(device: Device) -> Result<Self, MetalSupportError> {
        let pipelines = JpegPipelineRegistry::shared(&device)?;
        let queue = checked_command_queue(&device)?;
        Ok(Self {
            device,
            queue,
            pipelines,
            batch_scratch: BatchScratchPool::default(),
            viewport_plane_cache: Mutex::new(None),
            viewport_plane_cache_gate: ViewportPlaneCacheGate::new(),
        })
    }

    pub(in crate::compute) fn batch_scratch(&self) -> Result<BatchScratchLease<'_>, Error> {
        self.batch_scratch.acquire()
    }

    #[cfg(test)]
    pub(in crate::compute) fn batch_scratch_in_use_for_test(&self) -> bool {
        self.batch_scratch.in_use()
    }

    pub(in crate::compute) fn viewport_plane_cache(
        &self,
    ) -> Result<MutexGuard<'_, Option<CachedViewportPlanes>>, Error> {
        self.viewport_plane_cache
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "JPEG Metal viewport plane cache",
            })
    }

    pub(in crate::compute) fn viewport_plane_cache_lease(
        &self,
    ) -> Result<ViewportPlaneCacheLease, Error> {
        self.viewport_plane_cache_gate.acquire()
    }

    #[cfg(test)]
    pub(in crate::compute) fn viewport_plane_cache_id_for_test(
        &self,
    ) -> Result<Option<usize>, Error> {
        Ok(self
            .viewport_plane_cache()?
            .as_ref()
            .map(|cached| objc2::rc::Retained::as_ptr(&cached.plane0).cast::<()>() as usize))
    }
}

pub(in crate::compute) fn with_runtime<R>(
    operation: impl FnOnce(&MetalRuntime) -> Result<R, Error>,
) -> Result<R, Error> {
    DEFAULT_METAL_SESSION.with(|session| {
        let mut session = session.borrow_mut();
        if session.is_none() {
            *session = Some(system_default_device().map(crate::MetalBackendSession::new));
        }
        let Some(session) = session.as_ref() else {
            return Err(Error::MetalRuntime {
                message: "JPEG Metal default session was not initialized".to_string(),
            });
        };
        match session {
            Ok(session) => with_runtime_for_session(session, operation),
            Err(error) => Err(runtime_initialization_error(error)),
        }
    })
}

pub(in crate::compute) fn with_runtime_for_session<R>(
    session: &crate::MetalBackendSession,
    operation: impl FnOnce(&MetalRuntime) -> Result<R, Error>,
) -> Result<R, Error> {
    match session.runtime_result() {
        Ok(runtime) => operation(runtime),
        Err(error) => Err(runtime_initialization_error(error)),
    }
}

pub(crate) fn runtime_initialization_error(error: &MetalSupportError) -> Error {
    metal_runtime_support_error(error)
}

pub(in crate::compute) struct FastRgbDecodeBuffer {
    pub(in crate::compute) buffer: Buffer,
    pub(in crate::compute) dimensions: (u32, u32),
    pub(in crate::compute) status_buffer: Buffer,
    pub(in crate::compute) command_buffer: CommandBuffer,
}

pub(in crate::compute) fn private_jpeg_tile_from_fast_rgb_buffer(
    decoded: FastRgbDecodeBuffer,
) -> Result<crate::ResidentPrivateJpegTile, Error> {
    crate::ResidentPrivateJpegTile::new(
        decoded.buffer,
        0,
        decoded.dimensions,
        PixelFormat::Rgb8,
        decoded.dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel(),
        decoded.status_buffer,
        decoded.command_buffer,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_slots_allow_two_independent_batches() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        let runtime = Arc::new(MetalRuntime::new().expect("runtime"));
        let mut first = runtime.batch_scratch().expect("first lease");
        let first_buffer = first
            .shared_buffer_with_bytes(&runtime.device, "concurrent status", &[1, 2])
            .expect("first status");
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let other_runtime = Arc::clone(&runtime);
        let worker = std::thread::spawn(move || {
            let mut second = other_runtime.batch_scratch().expect("second lease");
            let buffer = second
                .shared_buffer_with_bytes(&other_runtime.device, "concurrent status", &[3, 4])
                .expect("second status");
            ready_tx
                .send(objc2::rc::Retained::as_ptr(&buffer).cast::<()>() as usize)
                .expect("ready");
            let _ = release_rx.recv();
        });
        let concurrent_buffer = ready_rx.recv_timeout(std::time::Duration::from_secs(1));
        let first_bytes =
            crate::buffers::checked_buffer_slice::<u8>(&first_buffer, 2, "first status")
                .expect("read own status");
        drop(first);
        let _ = release_tx.send(());
        worker.join().expect("worker");
        assert_ne!(
            concurrent_buffer.expect(
                "second batch must acquire independent scratch while the first is in flight"
            ),
            objc2::rc::Retained::as_ptr(&first_buffer).cast::<()>() as usize
        );
        assert_eq!(
            first_bytes,
            [1, 2],
            "concurrent staging must not overwrite the first batch"
        );
    }

    #[test]
    fn scratch_slots_apply_backpressure_when_both_are_leased() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        let runtime = Arc::new(MetalRuntime::new().expect("runtime"));
        let first = runtime.batch_scratch().expect("first lease");
        let second = runtime.batch_scratch().expect("second lease");
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let other = Arc::clone(&runtime);
        let worker = std::thread::spawn(move || {
            started_tx.send(()).expect("started");
            let _third = other.batch_scratch().expect("third lease");
            ready_tx.send(()).expect("ready");
        });
        started_rx.recv().expect("worker started");
        let blocked = ready_rx.recv_timeout(std::time::Duration::from_millis(100));
        drop(second);
        let progressed = ready_rx.recv_timeout(std::time::Duration::from_secs(1));
        drop(first);
        worker.join().expect("worker");
        progressed
            .expect("a waiter must reuse either released slot, even while slot zero stays busy");
        assert!(matches!(
            blocked,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        ));
    }
}
