// SPDX-License-Identifier: Apache-2.0

//! Image metadata and primitive value types. See spec Sections 2 and 4.
//!
//! `info.rs` intentionally has **no dependency on `error.rs`** — `error`
//! depends on us (for `Rect` and `SofKind`), and the reverse would create a
//! cycle. `DecodeOutcome`, which does need `Warning`, lives in `decoder.rs`
//! and is added in M1b when the decode methods are introduced.

use alloc::vec::Vec;

/// Start-of-frame variant. Determines the decode pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SofKind {
    /// SOF0: baseline sequential, 8-bit, Huffman.
    Baseline8,
    /// SOF1: extended sequential, 8-bit, Huffman.
    Extended8,
    /// SOF1: extended sequential, 12-bit, Huffman.
    Extended12,
    /// SOF2: progressive, 8-bit, Huffman.
    Progressive8,
    /// SOF2: progressive, 12-bit, Huffman.
    Progressive12,
    /// SOF3: lossless (Annex H predictor), Huffman.
    Lossless,
}

/// Declared input color space after APP14 detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    Grayscale,
    YCbCr,
    Rgb,
    Cmyk,
    Ycck,
}

/// Per-component (H, V) sampling factors, stored in declaration order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SamplingFactors {
    /// (H, V) pairs, one per component, each in `1..=4`.
    pub components: Vec<(u8, u8)>,
    /// `max(H_i)` across components — MCU width in data units.
    pub max_h: u8,
    /// `max(V_i)` across components — MCU height in data units.
    pub max_v: u8,
}

/// Inclusive axis-aligned rectangle in image coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    /// The full image rect for the given dimensions.
    pub fn full(dims: (u32, u32)) -> Self {
        Self { x: 0, y: 0, w: dims.0, h: dims.1 }
    }

    /// True if the rect is fully inside the bounding box of size `dims`.
    pub fn is_within(&self, dims: (u32, u32)) -> bool {
        let (w, h) = dims;
        self.x.checked_add(self.w).is_some_and(|r| r <= w)
            && self.y.checked_add(self.h).is_some_and(|b| b <= h)
    }
}

/// Caller-requested output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Rgb8,
    Rgba8 { alpha: u8 },
    Gray8,
    RawYCbCr8,
}

impl OutputFormat {
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb8 => 3,
            Self::Rgba8 { .. } => 4,
            Self::Gray8 => 1,
            Self::RawYCbCr8 => 3,
        }
    }
}

/// IDCT-level downscale factor; applies only to DCT-based SOFs (see spec
/// Section 4 matrix).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownscaleFactor {
    Full,
    Half,
    Quarter,
    Eighth,
}

/// Override for APP14 color-transform detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorTransform {
    Auto,
    ForceRgb,
    ForceYCbCr,
}

/// Header-derived image metadata. Populated by `Decoder::inspect` and by
/// `Decoder::new`. `scan_count` is the number of SOS markers observed in
/// the input — for sequential this is always 1; for progressive it is the
/// count of refinement passes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Info {
    pub dimensions: (u32, u32),
    pub color_space: ColorSpace,
    pub sampling: SamplingFactors,
    pub sof_kind: SofKind,
    pub bit_depth: u8,
    pub restart_interval: Option<u16>,
    pub scan_count: u16,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_full_matches_dimensions() {
        let r = Rect::full((1024, 768));
        assert_eq!(r, Rect { x: 0, y: 0, w: 1024, h: 768 });
    }

    #[test]
    fn rect_is_within_accepts_contained_rect() {
        assert!(Rect { x: 0, y: 0, w: 100, h: 100 }.is_within((100, 100)));
        assert!(Rect { x: 10, y: 20, w: 30, h: 40 }.is_within((100, 100)));
    }

    #[test]
    fn rect_is_within_rejects_overflowing_rect() {
        assert!(!Rect { x: 50, y: 50, w: 60, h: 10 }.is_within((100, 100)));
        assert!(!Rect { x: u32::MAX, y: 0, w: 1, h: 1 }.is_within((100, 100)));
    }

    #[test]
    fn output_format_bytes_per_pixel_matches_spec() {
        assert_eq!(OutputFormat::Rgb8.bytes_per_pixel(), 3);
        assert_eq!(OutputFormat::Rgba8 { alpha: 255 }.bytes_per_pixel(), 4);
        assert_eq!(OutputFormat::Gray8.bytes_per_pixel(), 1);
        assert_eq!(OutputFormat::RawYCbCr8.bytes_per_pixel(), 3);
    }
}
