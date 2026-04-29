// SPDX-License-Identifier: Apache-2.0

use slidecodec_core::{CacheStats, CodecContext};
#[cfg(target_os = "macos")]
use slidecodec_j2k_native::J2kDirectGrayscalePlan;

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
struct DirectGrayPlanCache {
    key: u64,
    plan: J2kDirectGrayscalePlan,
}

#[derive(Debug, Default, Clone)]
pub struct J2kContext {
    hits: u64,
    misses: u64,
    #[cfg(target_os = "macos")]
    direct_gray_plan: Option<DirectGrayPlanCache>,
}

impl J2kContext {
    pub const fn new() -> Self {
        Self {
            hits: 0,
            misses: 0,
            #[cfg(target_os = "macos")]
            direct_gray_plan: None,
        }
    }

    pub(crate) fn record_tile_decode(&mut self) {
        self.misses = self.misses.saturating_add(1);
    }

    #[cfg(target_os = "macos")]
    #[doc(hidden)]
    pub fn cached_direct_gray_plan(&mut self, key: u64) -> Option<J2kDirectGrayscalePlan> {
        if let Some(cache) = &self.direct_gray_plan {
            if cache.key == key {
                self.hits = self.hits.saturating_add(1);
                return Some(cache.plan.clone());
            }
        }
        self.misses = self.misses.saturating_add(1);
        None
    }

    #[cfg(target_os = "macos")]
    #[doc(hidden)]
    pub fn store_direct_gray_plan(&mut self, key: u64, plan: J2kDirectGrayscalePlan) {
        self.direct_gray_plan = Some(DirectGrayPlanCache { key, plan });
    }
}

impl CodecContext for J2kContext {
    fn clear(&mut self) {
        self.hits = 0;
        self.misses = 0;
        #[cfg(target_os = "macos")]
        {
            self.direct_gray_plan = None;
        }
    }

    fn cache_stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
        }
    }
}
