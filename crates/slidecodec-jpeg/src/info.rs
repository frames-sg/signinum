// SPDX-License-Identifier: Apache-2.0

//! Image metadata and primitive value types. See spec Sections 2 and 4.
//!
//! `info.rs` intentionally has **no dependency on `error.rs`** — `error`
//! depends on us (for `Rect` and `SofKind`), and the reverse would create a
//! cycle. `DecodeOutcome`, which does need `Warning`, lives in `decoder.rs`
//! and is added in M1b when the decode methods are introduced.

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplingFactors {
    components: [(u8, u8); 4],
    component_count: u8,
    /// `max(H_i)` across components — MCU width in data units.
    pub max_h: u8,
    /// `max(V_i)` across components — MCU height in data units.
    pub max_v: u8,
}

impl SamplingFactors {
    pub fn from_components(components: &[(u8, u8)]) -> Self {
        assert!(
            components.len() <= 4,
            "sampling metadata supports at most four components"
        );
        let mut packed = [(0u8, 0u8); 4];
        let mut max_h = 0u8;
        let mut max_v = 0u8;
        for (idx, &(h, v)) in components.iter().enumerate() {
            packed[idx] = (h, v);
            max_h = max_h.max(h);
            max_v = max_v.max(v);
        }
        Self {
            components: packed,
            component_count: components.len() as u8,
            max_h,
            max_v,
        }
    }

    pub fn len(&self) -> usize {
        self.component_count as usize
    }

    pub fn is_empty(&self) -> bool {
        self.component_count == 0
    }

    pub fn component(&self, index: usize) -> Option<(u8, u8)> {
        self.components().get(index).copied()
    }

    pub fn components(&self) -> &[(u8, u8)] {
        &self.components[..self.component_count as usize]
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (u8, u8)> + '_ {
        self.components().iter().copied()
    }
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
        Self {
            x: 0,
            y: 0,
            w: dims.0,
            h: dims.1,
        }
    }

    /// True if the rect is fully inside the bounding box of size `dims`.
    pub fn is_within(&self, dims: (u32, u32)) -> bool {
        let (w, h) = dims;
        self.x.checked_add(self.w).is_some_and(|r| r <= w)
            && self.y.checked_add(self.h).is_some_and(|b| b <= h)
    }
}

/// Internal JPEG-specific output format used behind the public core
/// `PixelFormat` + `Downscale` API adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputFormat {
    Rgb8,
    Rgb8Scaled { factor: DownscaleFactor },
    Rgba8 { alpha: u8 },
    Gray8,
    Gray8Scaled { factor: DownscaleFactor },
}

impl OutputFormat {
    pub(crate) fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb8 | Self::Rgb8Scaled { .. } => 3,
            Self::Rgba8 { .. } => 4,
            Self::Gray8 | Self::Gray8Scaled { .. } => 1,
        }
    }

    pub(crate) fn downscale(self) -> DownscaleFactor {
        match self {
            Self::Rgb8 | Self::Rgba8 { .. } | Self::Gray8 => DownscaleFactor::Full,
            Self::Rgb8Scaled { factor } | Self::Gray8Scaled { factor } => factor,
        }
    }
}

/// IDCT-level downscale factor; applies only to DCT-based SOFs (see spec
/// Section 4 matrix).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DownscaleFactor {
    Full,
    Half,
    Quarter,
    Eighth,
}

impl DownscaleFactor {
    pub(crate) const fn denominator(self) -> u32 {
        match self {
            Self::Full => 1,
            Self::Half => 2,
            Self::Quarter => 4,
            Self::Eighth => 8,
        }
    }

    pub(crate) const fn output_block_size(self) -> u32 {
        match self {
            Self::Full => 8,
            Self::Half => 4,
            Self::Quarter => 2,
            Self::Eighth => 1,
        }
    }
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
        assert_eq!(
            r,
            Rect {
                x: 0,
                y: 0,
                w: 1024,
                h: 768
            }
        );
    }

    #[test]
    fn rect_is_within_accepts_contained_rect() {
        assert!(Rect {
            x: 0,
            y: 0,
            w: 100,
            h: 100
        }
        .is_within((100, 100)));
        assert!(Rect {
            x: 10,
            y: 20,
            w: 30,
            h: 40
        }
        .is_within((100, 100)));
    }

    #[test]
    fn rect_is_within_rejects_overflowing_rect() {
        assert!(!Rect {
            x: 50,
            y: 50,
            w: 60,
            h: 10
        }
        .is_within((100, 100)));
        assert!(!Rect {
            x: u32::MAX,
            y: 0,
            w: 1,
            h: 1
        }
        .is_within((100, 100)));
    }

    #[test]
    fn output_format_bytes_per_pixel_matches_spec() {
        assert_eq!(OutputFormat::Rgb8.bytes_per_pixel(), 3);
        assert_eq!(
            OutputFormat::Rgb8Scaled {
                factor: DownscaleFactor::Quarter
            }
            .bytes_per_pixel(),
            3
        );
        assert_eq!(OutputFormat::Rgba8 { alpha: 255 }.bytes_per_pixel(), 4);
        assert_eq!(OutputFormat::Gray8.bytes_per_pixel(), 1);
        assert_eq!(
            OutputFormat::Gray8Scaled {
                factor: DownscaleFactor::Half
            }
            .bytes_per_pixel(),
            1
        );
    }

    #[test]
    fn sampling_factors_store_components_without_heap_state() {
        let sampling = SamplingFactors::from_components(&[(2, 2), (1, 1), (1, 1)]);
        assert_eq!(sampling.len(), 3);
        assert_eq!(sampling.component(0), Some((2, 2)));
        assert_eq!(sampling.component(1), Some((1, 1)));
        assert_eq!(sampling.component(3), None);
        assert_eq!(sampling.components(), &[(2, 2), (1, 1), (1, 1)]);
        assert_eq!(sampling.max_h, 2);
        assert_eq!(sampling.max_v, 2);
    }
}
