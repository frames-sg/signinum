// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use std::{
    cell::RefCell,
    sync::{Arc, Mutex},
};

use crate::metal_types::{Buffer, CommandBufferRef, CommandQueue, Device};
use j2k_metal_support::{
    checked_command_queue, checked_shared_buffer, commit_and_wait, wait_for_completion,
    MetalSupportError,
};

use crate::{
    buffer_pool::{MetalBufferPools, PooledBuffer},
    error::{metal_kernel_support_error, metal_runtime_support_error},
    Error,
};

mod decode;
pub(in crate::engine) use decode::DecodeKernels;
mod encode;
pub(in crate::engine) use encode::EncodeKernels;
mod kernel_cache;
use kernel_cache::{shared_kernel_groups, KernelCachePoisoned, SharedKernelGroups};
mod profile;
use profile::ClassicTier1ProfileKernels;
mod buffers;
use buffers::BufferKernels;

#[cfg(test)]
use j2k_metal_support::system_default_device;

thread_local! {
    static DEFAULT_METAL_SESSION: RefCell<Option<Result<crate::MetalBackendSession, MetalSupportError>>> = const { RefCell::new(None) };
    static METAL_RUNTIME_OVERRIDE: RefCell<Option<Arc<MetalRuntime>>> = const { RefCell::new(None) };
}

pub(crate) struct MetalRuntime {
    kernels: Result<Arc<SharedKernelGroups>, KernelCachePoisoned>,

    pub(super) device: Device,
    pub(crate) queue: CommandQueue,
    pub(super) tier1_dummy_buffer: Buffer,
    pub(super) buffer_pools: MetalBufferPools,
    pub(in crate::engine) prepared_ht_execution_cache:
        Mutex<super::decode_dispatch::PreparedMetalHtExecutionCache>,
}

// SAFETY: Every retained Metal object in the runtime is documented by Metal
// as usable across threads. CPU-side mutable pool/cache state is protected by
// its own mutex. OnceLock serializes each kernel group's initialization;
// pipelines and lookup buffers are immutable after successful initialization,
// and command submission remains serialized by each Metal queue.
unsafe impl Send for MetalRuntime {}
// SAFETY: Shared references expose only immutable Metal handles or
// mutex-protected pool/cache operations; no unsynchronized CPU mutation is
// reachable through `MetalRuntime`.
unsafe impl Sync for MetalRuntime {}

impl MetalRuntime {
    #[cfg(test)]
    pub(crate) fn new() -> Result<Self, MetalSupportError> {
        let device = system_default_device()?;
        Self::new_with_device(&device)
    }

    pub(crate) fn new_with_device(device: &Device) -> Result<Self, MetalSupportError> {
        let queue = checked_command_queue(device)?;
        Self::new_with_device_and_queue(device, queue)
    }

    pub(crate) fn new_with_device_and_queue(
        device: &Device,
        queue: CommandQueue,
    ) -> Result<Self, MetalSupportError> {
        let kernels = shared_kernel_groups(device);
        Self::new_with_device_queue_and_kernels(device, queue, kernels)
    }

    fn new_with_device_queue_and_kernels(
        device: &Device,
        queue: CommandQueue,
        kernels: Result<Arc<SharedKernelGroups>, KernelCachePoisoned>,
    ) -> Result<Self, MetalSupportError> {
        Ok(Self {
            device: device.clone(),
            queue,
            kernels,
            tier1_dummy_buffer: checked_shared_buffer(device, 1)?,
            buffer_pools: MetalBufferPools::new(device),
            prepared_ht_execution_cache: Mutex::new(
                super::decode_dispatch::PreparedMetalHtExecutionCache::new(),
            ),
        })
    }

    #[cfg(test)]
    fn new_isolated() -> Result<Self, MetalSupportError> {
        let device = system_default_device()?;
        Self::new_isolated_with_device(&device)
    }

    #[cfg(test)]
    fn new_isolated_with_device(device: &Device) -> Result<Self, MetalSupportError> {
        let queue = checked_command_queue(device)?;
        Self::new_with_device_queue_and_kernels(
            device,
            queue,
            Ok(Arc::new(SharedKernelGroups::default())),
        )
    }

    fn kernels(&self) -> Result<&SharedKernelGroups, Error> {
        self.kernels
            .as_deref()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "J2K Metal shared kernel cache",
            })
    }

    #[cfg(test)]
    fn kernels_for_test(&self) -> &SharedKernelGroups {
        self.kernels
            .as_deref()
            .expect("isolated test runtime has kernel groups")
    }

    pub(in crate::engine) fn decode(&self) -> Result<&DecodeKernels, Error> {
        self.kernels()?
            .decode
            .get_or_init(|| DecodeKernels::new(&self.device))
            .as_ref()
            .map_err(runtime_initialization_error)
    }

    pub(in crate::engine) fn encode(&self) -> Result<&EncodeKernels, Error> {
        self.kernels()?
            .encode
            .get_or_init(|| EncodeKernels::new(&self.device))
            .as_ref()
            .map_err(runtime_initialization_error)
    }

    pub(in crate::engine) fn profile(&self) -> Result<&ClassicTier1ProfileKernels, Error> {
        self.kernels()?
            .profile
            .get_or_init(|| ClassicTier1ProfileKernels::new(&self.device))
            .as_ref()
            .map_err(runtime_initialization_error)
    }

    pub(in crate::engine) fn buffers(&self) -> Result<&BufferKernels, Error> {
        self.kernels()?
            .buffers
            .get_or_init(|| BufferKernels::new(&self.device))
            .as_ref()
            .map_err(runtime_initialization_error)
    }

    pub(crate) fn command_queue(&self) -> &crate::metal_types::CommandQueueRef {
        self.queue.as_ref()
    }

    pub(super) fn take_private_buffer(&self, bytes: usize) -> Result<PooledBuffer, Error> {
        self.buffer_pools.take_private(&self.device, bytes)
    }

    pub(super) fn recycle_private_buffer(&self, buffer: PooledBuffer) -> Result<(), Error> {
        self.buffer_pools.recycle_private(buffer)
    }

    pub(super) fn take_shared_buffer(&self, bytes: usize) -> Result<PooledBuffer, Error> {
        self.buffer_pools.take_shared(&self.device, bytes)
    }

    pub(super) fn recycle_shared_buffer(&self, buffer: PooledBuffer) -> Result<(), Error> {
        self.buffer_pools.recycle_shared(buffer)
    }

    pub(crate) fn buffer_pool_diagnostics(
        &self,
    ) -> Result<crate::MetalBufferPoolsDiagnostics, Error> {
        self.buffer_pools.diagnostics()
    }
}

pub(super) fn with_runtime<R>(
    f: impl FnOnce(&MetalRuntime) -> Result<R, Error>,
) -> Result<R, Error> {
    let override_runtime = METAL_RUNTIME_OVERRIDE.with(|slot| slot.borrow().clone());
    if let Some(runtime) = override_runtime {
        return f(&runtime);
    }

    DEFAULT_METAL_SESSION.with(|session| {
        let mut session = session.borrow_mut();
        if session.is_none() {
            *session = Some(
                j2k_metal_support::system_default_device().map(crate::MetalBackendSession::new),
            );
        }
        let Some(session) = session.as_ref() else {
            return Err(Error::MetalRuntime {
                message: "J2K Metal default session was not initialized".to_string(),
            });
        };
        match session {
            Ok(session) => with_runtime_for_session(session, f),
            Err(error) => Err(runtime_initialization_error(error)),
        }
    })
}

pub(crate) fn current_runtime_device_registry_id() -> Result<u64, Error> {
    with_runtime(|runtime| Ok(runtime.device.registryID()))
}

pub(crate) fn runtime_initialization_error(error: &MetalSupportError) -> Error {
    metal_runtime_support_error(error)
}

pub(super) fn commit_and_wait_metal(command_buffer: &CommandBufferRef) -> Result<(), Error> {
    commit_and_wait(command_buffer)
        .map_err(|error| metal_kernel_support_error(error.to_string(), error))
}

pub(super) fn wait_for_completion_metal(command_buffer: &CommandBufferRef) -> Result<(), Error> {
    wait_for_completion(command_buffer)
        .map_err(|error| metal_kernel_support_error(error.to_string(), error))
}

struct RuntimeOverrideGuard {
    previous: Option<Arc<MetalRuntime>>,
}

impl Drop for RuntimeOverrideGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        METAL_RUNTIME_OVERRIDE.with(|slot| {
            slot.replace(previous);
        });
    }
}

pub(crate) fn with_runtime_for_session<R>(
    session: &crate::MetalBackendSession,
    f: impl FnOnce(&MetalRuntime) -> Result<R, Error>,
) -> Result<R, Error> {
    let runtime = session.runtime()?;
    let previous = METAL_RUNTIME_OVERRIDE.with(|slot| slot.replace(Some(runtime.clone())));
    let _guard = RuntimeOverrideGuard { previous };
    f(&runtime)
}

pub(super) fn with_runtime_for_device<R>(
    device: &Device,
    f: impl FnOnce(&MetalRuntime) -> Result<R, Error>,
) -> Result<R, Error> {
    let override_runtime = METAL_RUNTIME_OVERRIDE.with(|slot| slot.borrow().clone());
    if let Some(runtime) = override_runtime {
        if objc2::rc::Retained::as_ptr(&runtime.device) == objc2::rc::Retained::as_ptr(device) {
            return f(&runtime);
        }
    }

    let session = crate::MetalBackendSession::new(device.clone());
    with_runtime_for_session(&session, f)
}

#[cfg(test)]
pub(crate) fn with_isolated_runtime_for_device_for_test<R>(
    device: &Device,
    f: impl FnOnce() -> Result<R, Error>,
) -> Result<R, Error> {
    let runtime = Arc::new(
        MetalRuntime::new_isolated_with_device(device)
            .map_err(|error| runtime_initialization_error(&error))?,
    );
    let previous = METAL_RUNTIME_OVERRIDE.with(|slot| slot.replace(Some(runtime)));
    let _guard = RuntimeOverrideGuard { previous };
    f()
}

#[cfg(test)]
mod resource_profile_tests {
    use std::{
        sync::{Arc, Barrier},
        time::{Duration, Instant},
    };

    use super::{system_default_device, MetalRuntime};
    use crate::metal_types::prelude::*;

    #[test]
    fn decode_initializes_only_its_own_group_and_reuses_it() {
        let runtime = MetalRuntime::new_isolated().expect("Metal runtime");
        assert!(runtime.kernels_for_test().decode.get().is_none());
        assert!(runtime.kernels_for_test().encode.get().is_none());
        assert!(runtime.kernels_for_test().profile.get().is_none());
        assert!(runtime.kernels_for_test().buffers.get().is_none());
        let first = runtime.decode().expect("decode kernels");
        let second = runtime.decode().expect("cached decode kernels");
        assert!(std::ptr::eq(first, second));
        assert!(runtime.kernels_for_test().encode.get().is_none());
        assert!(runtime.kernels_for_test().profile.get().is_none());
        assert!(runtime.kernels_for_test().buffers.get().is_none());
    }

    #[test]
    fn same_device_runtimes_share_kernels_but_keep_mutable_state_isolated() {
        let runtime_a = MetalRuntime::new().expect("first Metal runtime");
        let runtime_b =
            MetalRuntime::new_with_device(&runtime_a.device).expect("second Metal runtime");

        assert!(std::ptr::eq(
            runtime_a.decode().expect("first decode kernels"),
            runtime_b.decode().expect("shared decode kernels")
        ));
        assert!(std::ptr::eq(
            runtime_a.encode().expect("first encode kernels"),
            runtime_b.encode().expect("shared encode kernels")
        ));
        assert!(!std::ptr::eq(
            objc2::rc::Retained::as_ptr(&runtime_a.queue),
            objc2::rc::Retained::as_ptr(&runtime_b.queue)
        ));
        assert!(!std::ptr::eq(
            &raw const runtime_a.buffer_pools,
            &raw const runtime_b.buffer_pools
        ));
        assert!(!std::ptr::eq(
            &raw const runtime_a.prepared_ht_execution_cache,
            &raw const runtime_b.prepared_ht_execution_cache
        ));
    }

    #[test]
    fn concurrent_same_device_runtimes_share_initialized_kernels() {
        const WORKERS: usize = 4;

        let start = Arc::new(Barrier::new(WORKERS));
        let workers = (0..WORKERS)
            .map(|_| {
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    let runtime = MetalRuntime::new().expect("concurrent Metal runtime");
                    runtime.decode().expect("concurrent decode kernels");
                    runtime.encode().expect("concurrent encode kernels");
                    runtime
                })
            })
            .collect::<Vec<_>>();
        let runtimes = workers
            .into_iter()
            .map(|worker| worker.join().expect("Metal runtime worker"))
            .collect::<Vec<_>>();
        let first_decode = runtimes[0].decode().expect("first decode kernels");
        let first_encode = runtimes[0].encode().expect("first encode kernels");

        assert!(runtimes.iter().skip(1).all(|runtime| std::ptr::eq(
            first_decode,
            runtime.decode().expect("shared decode kernels")
        )));
        assert!(runtimes.iter().skip(1).all(|runtime| std::ptr::eq(
            first_encode,
            runtime.encode().expect("shared encode kernels")
        )));
    }

    #[test]
    #[ignore = "manual Metal cold-start benchmark"]
    fn benchmark_cold_and_repeated_session_kernel_initialization() {
        const PAIRED_SAMPLES: u32 = 5;

        let device = system_default_device().expect("Metal device");
        let cold_start = Instant::now();
        let anchor = MetalRuntime::new_with_device(&device).expect("cold Metal runtime");
        let anchor_decode = anchor.decode().expect("cold decode kernels");
        let anchor_encode = anchor.encode().expect("cold encode kernels");
        let cold_elapsed = cold_start.elapsed();

        let measure_uncached = || {
            let start = Instant::now();
            let runtime = MetalRuntime::new_isolated_with_device(&device)
                .expect("uncached repeated Metal runtime");
            let decode = runtime.decode().expect("uncached decode kernels");
            let encode = runtime.encode().expect("uncached encode kernels");
            let elapsed = start.elapsed();
            assert!(!std::ptr::eq(anchor_decode, decode));
            assert!(!std::ptr::eq(anchor_encode, encode));
            elapsed
        };
        let measure_shared = || {
            let start = Instant::now();
            let runtime = MetalRuntime::new_with_device(&device).expect("shared Metal runtime");
            let decode = runtime.decode().expect("shared decode kernels");
            let encode = runtime.encode().expect("shared encode kernels");
            let elapsed = start.elapsed();
            assert!(std::ptr::eq(anchor_decode, decode));
            assert!(std::ptr::eq(anchor_encode, encode));
            elapsed
        };
        let mut uncached_elapsed = Duration::ZERO;
        let mut shared_elapsed = Duration::ZERO;
        for sample in 0..PAIRED_SAMPLES {
            if sample.is_multiple_of(2) {
                uncached_elapsed += measure_uncached();
                shared_elapsed += measure_shared();
            } else {
                shared_elapsed += measure_shared();
                uncached_elapsed += measure_uncached();
            }
        }

        eprintln!(
            "METAL_SESSION_STARTUP cold_setup_us={} uncached_mean_us={} shared_mean_us={} paired_samples={PAIRED_SAMPLES}",
            cold_elapsed.as_micros(),
            (uncached_elapsed / PAIRED_SAMPLES).as_micros(),
            (shared_elapsed / PAIRED_SAMPLES).as_micros(),
        );
    }

    #[test]
    fn failed_optional_groups_do_not_block_decode_or_buffer_operations() {
        let runtime = MetalRuntime::new_isolated().expect("Metal runtime");
        let failure = j2k_metal_support::MetalSupportError::ShaderLibrary {
            message: "test optional compiler failure".into(),
        };
        assert!(runtime
            .kernels_for_test()
            .encode
            .set(Err(failure.clone()))
            .is_ok());
        assert!(runtime.kernels_for_test().profile.set(Err(failure)).is_ok());
        runtime
            .decode()
            .expect("decode independent of optional groups");
        runtime
            .buffers()
            .expect("buffer operations independent of profiling");
        for error in [runtime.encode().err(), runtime.profile().err()] {
            assert!(matches!(
                error,
                Some(crate::Error::MetalSupport {
                    source: j2k_metal_support::MetalSupportError::ShaderLibrary { .. },
                    ..
                })
            ));
        }
        // Failures remain cached; a later decode does not retry optional initialization.
        runtime.decode().expect("decode remains usable");
        assert!(runtime
            .kernels_for_test()
            .encode
            .get()
            .is_some_and(Result::is_err));
        assert!(runtime
            .kernels_for_test()
            .profile
            .get()
            .is_some_and(Result::is_err));
    }

    #[test]
    fn key_tier1_pipeline_static_resources_are_queryable() {
        let device = system_default_device().expect("P6 resource test requires Metal");
        let runtime = MetalRuntime::new_with_device(&device).expect("create Metal runtime");
        let pipelines = [
            (
                "metal-ht-cleanup-decode",
                &runtime
                    .decode()
                    .expect("decode kernels")
                    .ht_cleanup_batched_cleanup_only,
            ),
            (
                "metal-ht-encode",
                &runtime
                    .encode()
                    .expect("encode kernels")
                    .ht_encode_code_blocks,
            ),
            (
                "metal-classic-decode",
                &runtime
                    .decode()
                    .expect("decode kernels")
                    .classic_cleanup_plain_batched,
            ),
            (
                "metal-classic-encode",
                &runtime
                    .encode()
                    .expect("encode kernels")
                    .classic_encode_code_blocks_32,
            ),
            (
                "metal-classic-profile",
                &runtime.profile().expect("profiling kernels").density,
            ),
        ];

        for (workload, pipeline) in pipelines {
            let shared_bytes = pipeline.staticThreadgroupMemoryLength();
            let simd_width = pipeline.threadExecutionWidth();
            let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
            assert!(simd_width > 0, "{workload} has no reported SIMD width");
            assert!(
                max_threads >= simd_width,
                "{workload} reports fewer maximum threads than one SIMD group"
            );
            eprintln!(
                "P6_RESOURCE workload={workload} shared_bytes_per_group={shared_bytes} thread_execution_width={simd_width} max_total_threads_per_threadgroup={max_threads}"
            );
        }
    }
}
