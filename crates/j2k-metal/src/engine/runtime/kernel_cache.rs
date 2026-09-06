// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, LazyLock, Mutex, OnceLock, Weak};

use crate::metal_types::{prelude::*, Device};
use j2k_metal_support::MetalSupportError;

use super::{BufferKernels, ClassicTier1ProfileKernels, DecodeKernels, EncodeKernels};

static KERNEL_GROUPS_BY_DEVICE: LazyLock<KernelGroupCache> =
    LazyLock::new(KernelGroupCache::default);
const MAX_TRACKED_METAL_DEVICES: usize = 8;

#[derive(Default)]
pub(super) struct SharedKernelGroups {
    pub(super) decode: OnceLock<Result<DecodeKernels, MetalSupportError>>,
    pub(super) encode: OnceLock<Result<EncodeKernels, MetalSupportError>>,
    pub(super) profile: OnceLock<Result<ClassicTier1ProfileKernels, MetalSupportError>>,
    pub(super) buffers: OnceLock<Result<BufferKernels, MetalSupportError>>,
}

// SAFETY: The retained Metal pipeline states and lookup buffers become
// immutable before each OnceLock publishes a kernel group. Metal permits
// concurrent use of those resources across queues and threads, and OnceLock
// serializes initialization and exposes only shared references afterward.
unsafe impl Send for SharedKernelGroups {}
// SAFETY: Shared references expose only immutable Metal handles. Each runtime
// continues to own its mutable queue, pools, and execution caches separately.
unsafe impl Sync for SharedKernelGroups {}

struct KernelGroupCache {
    state: Mutex<KernelGroupCacheState>,
}

struct KernelGroupCacheState {
    slots: [Option<KernelGroupCacheEntry>; MAX_TRACKED_METAL_DEVICES],
    next_replacement: usize,
}

struct KernelGroupCacheEntry {
    device: DeviceKernelCacheKey,
    groups: Weak<SharedKernelGroups>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct KernelCachePoisoned;

// The registry ID separates physical GPUs. Retaining the MTLDevice object
// address in the key also preserves this crate's exact-device ownership rule
// for the lookup buffers stored alongside immutable pipeline states.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DeviceKernelCacheKey {
    registry_id: u64,
    object_address: usize,
}

impl DeviceKernelCacheKey {
    fn new(device: &Device) -> Self {
        Self {
            registry_id: device.registryID(),
            object_address: objc2::rc::Retained::as_ptr(device).addr(),
        }
    }

    #[cfg(test)]
    const fn synthetic(registry_id: u64, object_address: usize) -> Self {
        Self {
            registry_id,
            object_address,
        }
    }
}

impl Default for KernelGroupCache {
    fn default() -> Self {
        Self {
            state: Mutex::new(KernelGroupCacheState {
                slots: std::array::from_fn(|_| None),
                next_replacement: 0,
            }),
        }
    }
}

impl KernelGroupCache {
    fn get(
        &self,
        device: DeviceKernelCacheKey,
    ) -> Result<Arc<SharedKernelGroups>, KernelCachePoisoned> {
        let mut state = self.state.lock().map_err(|_| KernelCachePoisoned)?;

        for entry in state.slots.iter().flatten() {
            if entry.device == device {
                if let Some(groups) = entry.groups.upgrade() {
                    return Ok(groups);
                }
            }
        }

        let shared = Arc::new(SharedKernelGroups::default());
        let reusable_slot = state.slots.iter().position(|entry| {
            entry
                .as_ref()
                .is_none_or(|entry| entry.groups.strong_count() == 0)
        });
        let slot = if let Some(slot) = reusable_slot {
            slot
        } else {
            let slot = state.next_replacement;
            state.next_replacement = (state.next_replacement + 1) % MAX_TRACKED_METAL_DEVICES;
            slot
        };
        state.slots[slot] = Some(KernelGroupCacheEntry {
            device,
            groups: Arc::downgrade(&shared),
        });
        Ok(shared)
    }
}

pub(super) fn shared_kernel_groups(
    device: &Device,
) -> Result<Arc<SharedKernelGroups>, KernelCachePoisoned> {
    KERNEL_GROUPS_BY_DEVICE.get(DeviceKernelCacheKey::new(device))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{DeviceKernelCacheKey, KernelGroupCache};

    fn device(registry_id: u64, object_address: usize) -> DeviceKernelCacheKey {
        DeviceKernelCacheKey::synthetic(registry_id, object_address)
    }

    #[test]
    fn cache_shares_live_groups_only_with_the_same_device() {
        let cache = KernelGroupCache::default();
        let first = cache.get(device(7, 11)).expect("first groups");
        let same_device = cache.get(device(7, 11)).expect("same-device groups");
        let other_device = cache.get(device(8, 12)).expect("other-device groups");

        assert!(Arc::ptr_eq(&first, &same_device));
        assert!(!Arc::ptr_eq(&first, &other_device));
    }

    #[test]
    fn cache_does_not_alias_distinct_device_objects_with_one_registry_id() {
        let cache = KernelGroupCache::default();
        let first = cache.get(device(7, 11)).expect("first groups");
        let distinct_object = cache.get(device(7, 12)).expect("distinct-object groups");

        assert!(!Arc::ptr_eq(&first, &distinct_object));
    }

    #[test]
    fn cache_does_not_keep_device_groups_alive() {
        let cache = KernelGroupCache::default();
        let groups = cache.get(device(7, 11)).expect("groups");
        let released = Arc::downgrade(&groups);
        drop(groups);

        assert!(released.upgrade().is_none());
        let replacement = cache.get(device(7, 11)).expect("replacement groups");
        assert_eq!(Arc::strong_count(&replacement), 1);
    }

    #[test]
    fn cache_metadata_has_a_fixed_device_bound() {
        let cache = KernelGroupCache::default();
        let live_groups = (0..=super::MAX_TRACKED_METAL_DEVICES as u64)
            .map(|registry_id| {
                cache
                    .get(device(
                        registry_id,
                        usize::try_from(registry_id).expect("small test device key") + 1,
                    ))
                    .expect("bounded groups")
            })
            .collect::<Vec<_>>();

        assert_eq!(
            cache
                .state
                .lock()
                .expect("kernel cache state")
                .slots
                .iter()
                .flatten()
                .count(),
            super::MAX_TRACKED_METAL_DEVICES
        );
        assert_eq!(live_groups.len(), super::MAX_TRACKED_METAL_DEVICES + 1);
    }

    #[test]
    fn cache_reports_poisoned_state() {
        let cache = Arc::new(KernelGroupCache::default());
        let poison = Arc::clone(&cache);
        let panic = std::thread::spawn(move || {
            let _state = poison.state.lock().expect("initial cache lock");
            panic!("poison kernel cache for test");
        });
        assert!(panic.join().is_err());

        assert!(matches!(
            cache.get(device(7, 11)),
            Err(super::KernelCachePoisoned)
        ));
    }
}
