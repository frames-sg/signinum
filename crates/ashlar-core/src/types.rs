// SPDX-License-Identifier: Apache-2.0

use alloc::vec::Vec;

use crate::scale::Downscale;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Colorspace {
    Grayscale,
    YCbCr,
    Rgb,
    Cmyk,
    Ycck,
    SRgb,
    SGray,
    IccTagged,
    Rct,
    Ict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileLayout {
    pub tile_width: u32,
    pub tile_height: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CodedUnitLayout {
    pub unit_width: u32,
    pub unit_height: u32,
    pub units_x: u32,
    pub units_y: u32,
}

impl CodedUnitLayout {
    pub const fn unit_count(&self) -> u32 {
        self.units_x.saturating_mul(self.units_y)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    pub const fn full(dims: (u32, u32)) -> Self {
        Self {
            x: 0,
            y: 0,
            w: dims.0,
            h: dims.1,
        }
    }

    pub fn is_within(&self, dims: (u32, u32)) -> bool {
        let (w, h) = dims;
        self.x.checked_add(self.w).is_some_and(|r| r <= w)
            && self.y.checked_add(self.h).is_some_and(|b| b <= h)
    }

    #[must_use]
    pub fn scaled_covering(&self, scale: Downscale) -> Self {
        let denom = scale.denominator();
        let x_end = self.x.saturating_add(self.w);
        let y_end = self.y.saturating_add(self.h);
        let x0 = self.x / denom;
        let y0 = self.y / denom;
        let x1 = x_end.div_ceil(denom);
        let y1 = y_end.div_ceil(denom);
        Self {
            x: x0,
            y: y0,
            w: x1.saturating_sub(x0),
            h: y1.saturating_sub(y0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Info {
    pub dimensions: (u32, u32),
    pub components: u8,
    pub colorspace: Colorspace,
    pub bit_depth: u8,
    pub tile_layout: Option<TileLayout>,
    pub coded_unit_layout: Option<CodedUnitLayout>,
    pub restart_interval: Option<u32>,
    pub resolution_levels: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WarningKind {
    MinorCompliance,
    NonFatalTruncation,
    UnusualFeature,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeOutcome<W> {
    pub decoded: Rect,
    pub warnings: Vec<W>,
}
