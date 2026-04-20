// SPDX-License-Identifier: Apache-2.0

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("slidecodec-core only supports x86_64 and aarch64 targets");

use core::sync::atomic::{AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub sse41: bool,
    pub neon: bool,
}

impl CpuFeatures {
    pub fn detect() -> Self {
        static DETECTED: AtomicU8 = AtomicU8::new(0);

        let cached = DETECTED.load(Ordering::Acquire);
        if cached != 0 {
            return Self::from_cache_byte(cached);
        }

        let detected = Self::detect_uncached();
        let encoded = detected.to_cache_byte();
        let _ = DETECTED.compare_exchange(0, encoded, Ordering::AcqRel, Ordering::Acquire);
        Self::from_cache_byte(DETECTED.load(Ordering::Acquire))
    }

    fn detect_uncached() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2: std::is_x86_feature_detected!("avx2"),
                sse41: std::is_x86_feature_detected!("sse4.1"),
                neon: false,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                avx2: false,
                sse41: false,
                neon: true,
            }
        }
    }

    const fn to_cache_byte(self) -> u8 {
        let mut encoded = 1_u8;
        if self.avx2 {
            encoded |= 1 << 1;
        }
        if self.sse41 {
            encoded |= 1 << 2;
        }
        if self.neon {
            encoded |= 1 << 3;
        }
        encoded
    }

    const fn from_cache_byte(encoded: u8) -> Self {
        let bits = encoded.saturating_sub(1);
        Self {
            avx2: (bits & (1 << 1)) != 0,
            sse41: (bits & (1 << 2)) != 0,
            neon: (bits & (1 << 3)) != 0,
        }
    }
}
