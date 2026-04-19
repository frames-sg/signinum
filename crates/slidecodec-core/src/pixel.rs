// SPDX-License-Identifier: Apache-2.0

use crate::sample::SampleType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PixelLayout {
    Rgb,
    Rgba,
    Gray,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PixelFormat {
    Rgb8,
    Rgba8,
    Gray8,
    Rgb16,
    Rgba16,
    Gray16,
}

impl PixelFormat {
    pub const fn layout(self) -> PixelLayout {
        match self {
            Self::Rgb8 | Self::Rgb16 => PixelLayout::Rgb,
            Self::Rgba8 | Self::Rgba16 => PixelLayout::Rgba,
            Self::Gray8 | Self::Gray16 => PixelLayout::Gray,
        }
    }

    pub const fn sample(self) -> SampleType {
        match self {
            Self::Rgb8 | Self::Rgba8 | Self::Gray8 => SampleType::U8,
            Self::Rgb16 | Self::Rgba16 | Self::Gray16 => SampleType::U16,
        }
    }

    pub const fn channels(self) -> usize {
        match self.layout() {
            PixelLayout::Rgb => 3,
            PixelLayout::Rgba => 4,
            PixelLayout::Gray => 1,
        }
    }

    pub const fn bytes_per_sample(self) -> usize {
        match self.sample() {
            SampleType::U8 => 1,
            SampleType::U16 => 2,
        }
    }

    pub const fn bytes_per_pixel(self) -> usize {
        self.channels() * self.bytes_per_sample()
    }
}
