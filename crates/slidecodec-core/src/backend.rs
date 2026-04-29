// SPDX-License-Identifier: Apache-2.0

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("slidecodec-core only supports x86_64 and aarch64 targets");

use core::sync::atomic::{AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BackendRequest {
    #[default]
    Auto,
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
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
                avx2: detect_x86_avx2(),
                sse41: detect_x86_sse41(),
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

#[cfg(target_arch = "x86_64")]
fn detect_x86_sse41() -> bool {
    // SAFETY: CPUID is available on x86_64 by architecture guarantee.
    let features = unsafe { core::arch::x86_64::__cpuid(1) };
    (features.ecx & (1 << 19)) != 0
}

#[cfg(target_arch = "x86_64")]
fn detect_x86_avx2() -> bool {
    // SAFETY: CPUID is available on x86_64 by architecture guarantee.
    let leaf1 = unsafe { core::arch::x86_64::__cpuid(1) };
    let osxsave = (leaf1.ecx & (1 << 27)) != 0;
    let avx = (leaf1.ecx & (1 << 28)) != 0;
    if !(osxsave && avx) {
        return false;
    }

    // SAFETY: XGETBV is only executed after CPUID reports OSXSAVE support.
    let xcr0 = unsafe { core::arch::x86_64::_xgetbv(0) };
    let xmm_enabled = (xcr0 & 0b10) != 0;
    let ymm_enabled = (xcr0 & 0b100) != 0;
    if !(xmm_enabled && ymm_enabled) {
        return false;
    }

    // SAFETY: CPUID is available on x86_64 by architecture guarantee.
    let leaf7 = unsafe { core::arch::x86_64::__cpuid_count(7, 0) };
    (leaf7.ebx & (1 << 5)) != 0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub cpu: CpuFeatures,
    pub metal: bool,
    pub cuda: bool,
}

impl BackendCapabilities {
    #[must_use]
    pub fn detect() -> Self {
        Self {
            cpu: CpuFeatures::detect(),
            metal: cfg!(target_os = "macos"),
            cuda: false,
        }
    }

    #[must_use]
    pub const fn supports(self, request: BackendRequest) -> bool {
        match request {
            BackendRequest::Auto | BackendRequest::Cpu => true,
            BackendRequest::Metal => self.metal,
            BackendRequest::Cuda => self.cuda,
        }
    }

    #[must_use]
    pub fn resolve(self, request: BackendRequest) -> Option<BackendKind> {
        match request {
            BackendRequest::Auto => {
                if self.metal {
                    Some(BackendKind::Metal)
                } else if self.cuda {
                    Some(BackendKind::Cuda)
                } else {
                    Some(BackendKind::Cpu)
                }
            }
            BackendRequest::Cpu => Some(BackendKind::Cpu),
            BackendRequest::Metal if self.metal => Some(BackendKind::Metal),
            BackendRequest::Cuda if self.cuda => Some(BackendKind::Cuda),
            BackendRequest::Metal | BackendRequest::Cuda => None,
        }
    }
}
