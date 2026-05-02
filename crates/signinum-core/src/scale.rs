// SPDX-License-Identifier: Apache-2.0

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Downscale {
    None,
    Half,
    Quarter,
    Eighth,
}

impl Downscale {
    pub const fn denominator(self) -> u32 {
        match self {
            Self::None => 1,
            Self::Half => 2,
            Self::Quarter => 4,
            Self::Eighth => 8,
        }
    }

    pub const fn output_block_size(self) -> u32 {
        match self {
            Self::None => 8,
            Self::Half => 4,
            Self::Quarter => 2,
            Self::Eighth => 1,
        }
    }
}
