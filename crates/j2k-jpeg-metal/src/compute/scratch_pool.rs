// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded, independently leased scratch with notification on either slot's release.

use crate::{buffers::MetalBatchScratch, Error};
use std::{
    ops::{Deref, DerefMut},
    sync::{Condvar, Mutex, MutexGuard, TryLockError},
};

pub(super) struct BatchScratchPool {
    slots: [Mutex<MetalBatchScratch>; 2],
    availability: Mutex<()>,
    released: Condvar,
}

impl Default for BatchScratchPool {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| Mutex::new(MetalBatchScratch::default())),
            availability: Mutex::new(()),
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
            for slot in &self.slots {
                match slot.try_lock() {
                    Ok(scratch) => {
                        drop(availability);
                        return Ok(BatchScratchLease {
                            scratch: Some(scratch),
                            pool: self,
                        });
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
        self.slots
            .iter()
            .any(|slot| matches!(slot.try_lock(), Err(TryLockError::WouldBlock)))
    }
}

pub(in crate::compute) struct BatchScratchLease<'a> {
    scratch: Option<MutexGuard<'a, MetalBatchScratch>>,
    pool: &'a BatchScratchPool,
}

impl Deref for BatchScratchLease<'_> {
    type Target = MetalBatchScratch;
    fn deref(&self) -> &Self::Target {
        self.scratch
            .as_deref()
            .expect("scratch is present until lease drop")
    }
}

impl DerefMut for BatchScratchLease<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.scratch
            .as_deref_mut()
            .expect("scratch is present until lease drop")
    }
}

impl Drop for BatchScratchLease<'_> {
    fn drop(&mut self) {
        // Release under the same gate used by acquire before notifying. Recover
        // the gate only to finish cleanup during unwinding; future acquisition
        // still reports poison, and a panicking owner poisons its scratch slot.
        let _availability = self
            .pool
            .availability
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        drop(self.scratch.take());
        // Wake every waiter so a poisoned owner cannot leave other callers
        // asleep indefinitely after the first waiter reports that failure.
        self.pool.released.notify_all();
    }
}
