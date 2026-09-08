// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded, independently leased scratch with notification on either slot's release.

use crate::{buffers::MetalBatchScratch, Error};
use std::{
    ops::{Deref, DerefMut},
    sync::{Condvar, Mutex, TryLockError},
};

pub(super) struct BatchScratchPool {
    slots: [Mutex<Option<MetalBatchScratch>>; 2],
    // A panic in an owned lease must remain fail-closed after moving the
    // scratch value out of its mutex. Access this flag under the same gate.
    availability: Mutex<bool>,
    released: Condvar,
}

impl Default for BatchScratchPool {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| Mutex::new(Some(MetalBatchScratch::default()))),
            availability: Mutex::new(false),
            released: Condvar::new(),
        }
    }
}

fn poisoned() -> Error {
    Error::MetalStatePoisoned {
        state: "JPEG Metal batch scratch",
    }
}

impl BatchScratchPool {
    pub(super) fn acquire(&self) -> Result<BatchScratchLease<'_>, Error> {
        let mut availability = self.availability.lock().map_err(|_| poisoned())?;
        loop {
            if *availability {
                return Err(poisoned());
            }
            for slot in &self.slots {
                match slot.try_lock() {
                    Ok(mut entry) => {
                        if let Some(scratch) = entry.take() {
                            drop(entry);
                            drop(availability);
                            return Ok(BatchScratchLease {
                                scratch: Some(scratch),
                                slot,
                                pool: self,
                                panicking_on_acquire: std::thread::panicking(),
                            });
                        }
                    }
                    Err(TryLockError::WouldBlock) => {}
                    Err(TryLockError::Poisoned(_)) => return Err(poisoned()),
                }
            }
            // The availability mutex joins this check to release notifications,
            // so neither a free slot nor a wakeup can be missed between them.
            availability = self.released.wait(availability).map_err(|_| poisoned())?;
        }
    }

    #[cfg(test)]
    pub(super) fn in_use(&self) -> bool {
        self.slots.iter().any(|slot| match slot.try_lock() {
            Ok(entry) => entry.is_none(),
            Err(TryLockError::WouldBlock) => true,
            Err(TryLockError::Poisoned(_)) => false,
        })
    }
}

pub(in crate::compute) struct BatchScratchLease<'a> {
    scratch: Option<MetalBatchScratch>,
    slot: &'a Mutex<Option<MetalBatchScratch>>,
    pool: &'a BatchScratchPool,
    panicking_on_acquire: bool,
}

impl Deref for BatchScratchLease<'_> {
    type Target = MetalBatchScratch;
    fn deref(&self) -> &Self::Target {
        self.scratch
            .as_ref()
            .expect("scratch is present until lease drop")
    }
}

impl DerefMut for BatchScratchLease<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.scratch
            .as_mut()
            .expect("scratch is present until lease drop")
    }
}

impl Drop for BatchScratchLease<'_> {
    fn drop(&mut self) {
        // Restore under the acquisition gate before notifying, without holding
        // any pool lock during GPU work. Preserve MutexGuard's panic contract.
        let mut availability = self
            .pool
            .availability
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if std::thread::panicking() && !self.panicking_on_acquire {
            *availability = true;
        }
        let mut entry = self
            .slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *entry = self.scratch.take();
        drop(entry);
        // Wake every waiter so a poisoned owner cannot leave other callers
        // asleep indefinitely after the first waiter reports that failure.
        self.pool.released.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_leases_do_not_hold_pool_mutexes() {
        let pool = BatchScratchPool::default();
        let first = pool.acquire().expect("first lease");
        let second = pool.acquire().expect("second lease");
        assert!(pool.in_use());
        assert!(
            pool.availability.try_lock().is_ok(),
            "availability gate held by lease"
        );
        assert!(
            pool.slots.iter().all(|slot| slot.try_lock().is_ok()),
            "slot mutex held by lease"
        );
        drop(first);
        drop(second);
        assert!(!pool.in_use());
    }

    #[test]
    fn scratch_slots_fail_closed_after_panicking_owner() {
        let pool = BatchScratchPool::default();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _lease = pool.acquire().expect("scratch lease");
            panic!("abandon scratch during an owner panic");
        }));
        assert!(result.is_err());
        assert!(matches!(
            pool.acquire(),
            Err(Error::MetalStatePoisoned { .. })
        ));
    }
}
