// SPDX-License-Identifier: Apache-2.0

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub sse41: bool,
    pub neon: bool,
}

impl CpuFeatures {
    pub fn detect() -> Self {
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

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::default()
        }
    }
}
