// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prepared-plan cache lifecycle for region-scaled Metal color decode.

#[cfg(test)]
use std::cell::Cell;
use std::sync::{Arc, Mutex, OnceLock};

use crate::session::direct_plan_cache::{
    cached_session_region_scaled_color_plan, prepared_plan_cache_error,
    store_session_region_scaled_color_plan,
};
use crate::session::{PreparedPlanCache, PreparedPlanCacheKey};
use crate::Error;

pub(crate) const REGION_SCALED_COLOR_PLAN_CACHE_CAP: usize = 128;

static REGION_SCALED_COLOR_PLAN_CACHE: OnceLock<
    Mutex<PreparedPlanCache<Arc<crate::engine::PreparedDirectColorPlan>>>,
> = OnceLock::new();

#[cfg(test)]
macro_rules! test_atomic_counter {
    ($counter:ident, $reset:ident, $load:ident) => {
        std::thread_local! {
            static $counter: Cell<usize> = const { Cell::new(0) };
        }

        pub(crate) fn $reset() {
            $counter.with(|counter| counter.set(0));
        }

        pub(crate) fn $load() -> usize {
            $counter.with(Cell::get)
        }
    };
}

#[cfg(test)]
test_atomic_counter!(
    REGION_SCALED_COLOR_PLAN_BUILDS,
    reset_region_scaled_color_plan_builds_for_test,
    region_scaled_color_plan_builds_for_test
);
#[cfg(test)]
static REGION_SCALED_COLOR_PLAN_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
pub(crate) fn region_scaled_color_plan_test_lock_for_test() -> std::sync::MutexGuard<'static, ()> {
    REGION_SCALED_COLOR_PLAN_TEST_LOCK
        .lock()
        .expect("region scaled color plan test lock")
}

#[cfg(test)]
pub(crate) fn reset_region_scaled_color_plan_cache_for_test() {
    if let Some(cache) = REGION_SCALED_COLOR_PLAN_CACHE.get() {
        if let Ok(mut guard) = cache.lock() {
            guard.clear();
        }
    }
}

#[cfg(test)]
pub(super) fn record_region_scaled_color_plan_build_for_test() {
    REGION_SCALED_COLOR_PLAN_BUILDS.with(|counter| counter.set(counter.get().saturating_add(1)));
}

#[derive(Clone, Copy)]
pub(super) enum RegionScaledColorPlanCache<'a> {
    Uncached,
    Global,
    Session(&'a crate::MetalBackendSession),
}

impl RegionScaledColorPlanCache<'_> {
    pub(super) fn get(
        self,
        key: PreparedPlanCacheKey<'_>,
    ) -> Result<Option<Arc<crate::engine::PreparedDirectColorPlan>>, Error> {
        match self {
            Self::Uncached => Ok(None),
            Self::Global => cached_region_scaled_direct_color_plan(key),
            Self::Session(session) => cached_session_region_scaled_color_plan(session, key),
        }
    }

    pub(super) fn store(
        self,
        key: PreparedPlanCacheKey<'_>,
        plan: Arc<crate::engine::PreparedDirectColorPlan>,
    ) -> Result<(), Error> {
        match self {
            Self::Uncached => Ok(()),
            Self::Global => {
                plan.disable_dynamic_cpu_tier1_retention()?;
                store_region_scaled_direct_color_plan(key, plan)
            }
            Self::Session(session) => {
                plan.disable_dynamic_cpu_tier1_retention()?;
                store_session_region_scaled_color_plan(session, key, plan)
            }
        }
    }
}

fn cached_region_scaled_direct_color_plan(
    key: PreparedPlanCacheKey<'_>,
) -> Result<Option<Arc<crate::engine::PreparedDirectColorPlan>>, Error> {
    let cache = REGION_SCALED_COLOR_PLAN_CACHE
        .get_or_init(|| Mutex::new(PreparedPlanCache::new(REGION_SCALED_COLOR_PLAN_CACHE_CAP)));
    let mut guard = cache.lock().map_err(|_| Error::MetalStatePoisoned {
        state: "global region-scaled color prepared-plan cache",
    })?;
    Ok(guard.get(key).cloned())
}

fn store_region_scaled_direct_color_plan(
    key: PreparedPlanCacheKey<'_>,
    plan: Arc<crate::engine::PreparedDirectColorPlan>,
) -> Result<(), Error> {
    let cache = REGION_SCALED_COLOR_PLAN_CACHE
        .get_or_init(|| Mutex::new(PreparedPlanCache::new(REGION_SCALED_COLOR_PLAN_CACHE_CAP)));
    let mut guard = cache.lock().map_err(|_| Error::MetalStatePoisoned {
        state: "global region-scaled color prepared-plan cache",
    })?;
    evict_one_region_scaled_color_plan_if_needed(&mut guard, key, plan)
}

fn evict_one_region_scaled_color_plan_if_needed<T: crate::session::PreparedPlanCacheValue>(
    cache: &mut PreparedPlanCache<T>,
    key: PreparedPlanCacheKey<'_>,
    value: T,
) -> Result<(), Error> {
    cache.insert(key, value).map(|_| ()).map_err(|error| {
        prepared_plan_cache_error(
            "Metal region-scaled prepared-plan cache update failed",
            error,
        )
    })
}

#[cfg(test)]
#[test]
fn region_scaled_build_counts_exclude_other_test_threads() {
    let _guard = region_scaled_color_plan_test_lock_for_test();
    reset_region_scaled_color_plan_builds_for_test();
    record_region_scaled_color_plan_build_for_test();
    std::thread::spawn(record_region_scaled_color_plan_build_for_test)
        .join()
        .unwrap();
    assert_eq!(
        region_scaled_color_plan_builds_for_test(),
        1,
        "cache reuse assertions must count only their own synchronous plan builds"
    );
}
