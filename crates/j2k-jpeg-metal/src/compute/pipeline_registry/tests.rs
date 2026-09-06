// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Barrier,
};

use j2k_metal_support::{system_default_device, MetalSupportError};
use objc2_metal::MTLDevice as _;

use super::{JpegPipelineRegistry, PipelineRegistryCache};

const CONCURRENT_WORKERS: usize = 8;

fn should_run_metal_runtime() -> bool {
    j2k_test_support::metal_runtime_gate(module_path!())
}

#[test]
fn registry_cache_reuses_success_for_one_device() {
    if !should_run_metal_runtime() {
        return;
    }

    let cache = PipelineRegistryCache::default();
    let device = system_default_device().expect("Metal device");
    let registry_id = device.registryID();
    let loads = AtomicUsize::new(0);
    let first = cache
        .get_or_try_init(registry_id, || {
            loads.fetch_add(1, Ordering::Relaxed);
            JpegPipelineRegistry::load(&device)
        })
        .expect("first registry");
    let second = cache
        .get_or_try_init(registry_id, || {
            loads.fetch_add(1, Ordering::Relaxed);
            JpegPipelineRegistry::load(&device)
        })
        .expect("shared registry");

    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(loads.load(Ordering::Relaxed), 1);

    let cached = Arc::downgrade(&first);
    drop((first, second));
    let third = cache
        .get_or_try_init(registry_id, || {
            loads.fetch_add(1, Ordering::Relaxed);
            JpegPipelineRegistry::load(&device)
        })
        .expect("persistent registry");
    assert!(Arc::ptr_eq(
        &cached.upgrade().expect("cache owns successful registry"),
        &third
    ));
    assert_eq!(loads.load(Ordering::Relaxed), 1);
}

#[test]
fn registry_cache_keeps_devices_distinct() {
    if !should_run_metal_runtime() {
        return;
    }

    let cache = PipelineRegistryCache::default();
    let device = system_default_device().expect("Metal device");
    let first = cache
        .get_or_try_init(7, || JpegPipelineRegistry::load(&device))
        .expect("first registry");
    let second = cache
        .get_or_try_init(8, || JpegPipelineRegistry::load(&device))
        .expect("second registry");

    assert!(!Arc::ptr_eq(&first, &second));
}

#[test]
fn registry_cache_retries_after_initialization_failure() {
    if !should_run_metal_runtime() {
        return;
    }

    let cache = PipelineRegistryCache::default();
    let device = system_default_device().expect("Metal device");
    let loads = AtomicUsize::new(0);
    let result = cache.get_or_try_init(7, || {
        loads.fetch_add(1, Ordering::Relaxed);
        Err(MetalSupportError::MetalUnavailable)
    });
    let Err(error) = result else {
        panic!("initialization must fail");
    };
    assert_eq!(error, MetalSupportError::MetalUnavailable);
    let _registry = cache
        .get_or_try_init(7, || {
            loads.fetch_add(1, Ordering::Relaxed);
            JpegPipelineRegistry::load(&device)
        })
        .expect("retry registry");

    assert_eq!(loads.load(Ordering::Relaxed), 2);
}

#[test]
fn registry_cache_bypasses_slot_poison_after_loader_panic() {
    if !should_run_metal_runtime() {
        return;
    }

    let cache = PipelineRegistryCache::default();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = cache.get_or_try_init(7, || panic!("simulated loader panic"));
    }));
    assert!(panic.is_err());

    let poisoned_error = cache.get_or_try_init(7, || Err(MetalSupportError::MetalUnavailable));
    assert!(matches!(
        poisoned_error,
        Err(MetalSupportError::MetalUnavailable)
    ));
    let device = system_default_device().expect("Metal device");
    cache
        .get_or_try_init(7, || JpegPipelineRegistry::load(&device))
        .expect("poisoned cache must not block a later typed load");
}

#[test]
fn registry_cache_serializes_concurrent_initialization_per_device() {
    if !should_run_metal_runtime() {
        return;
    }

    let cache = Arc::new(PipelineRegistryCache::default());
    let loads = Arc::new(AtomicUsize::new(0));
    let start = Arc::new(Barrier::new(CONCURRENT_WORKERS));
    let keep_alive = Arc::new(Barrier::new(CONCURRENT_WORKERS));
    let workers = (0..CONCURRENT_WORKERS)
        .map(|_| {
            let cache = Arc::clone(&cache);
            let loads = Arc::clone(&loads);
            let start = Arc::clone(&start);
            let keep_alive = Arc::clone(&keep_alive);
            std::thread::spawn(move || {
                let device = system_default_device().expect("Metal device");
                start.wait();
                let registry = cache
                    .get_or_try_init(device.registryID(), || {
                        loads.fetch_add(1, Ordering::Relaxed);
                        JpegPipelineRegistry::load(&device)
                    })
                    .expect("registry");
                keep_alive.wait();
                registry
            })
        })
        .collect::<Vec<_>>();
    let registries = workers
        .into_iter()
        .map(|worker| worker.join().expect("worker"))
        .collect::<Vec<_>>();

    assert_eq!(loads.load(Ordering::Relaxed), 1);
    assert!(registries
        .iter()
        .all(|registry| Arc::ptr_eq(registry, &registries[0])));
}

#[test]
fn runtimes_share_pipelines_but_keep_command_queues_independent() {
    if !should_run_metal_runtime() {
        return;
    }

    let runtime_a = crate::compute::runtime::MetalRuntime::new().expect("Metal runtime");
    let runtime_b =
        crate::compute::runtime::MetalRuntime::new_with_device(runtime_a.device.clone())
            .expect("Metal runtime");

    assert!(Arc::ptr_eq(&runtime_a.pipelines, &runtime_b.pipelines));
    assert!(!core::ptr::eq(
        objc2::rc::Retained::as_ptr(&runtime_a.queue),
        objc2::rc::Retained::as_ptr(&runtime_b.queue)
    ));
}
