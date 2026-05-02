// SPDX-License-Identifier: Apache-2.0

//! YCbCr → RGB conversion. Scalar implementation uses libjpeg-turbo's 16-bit
//! fixed-point coefficients (`jdcolor.c`), so outputs match their ISLOW path
//! byte-for-byte.
//!
//! Coefficients (all × 2^16, rounded):
//!     R = Y                        + 1.40200 * (Cr - 128)
//!     G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
//!     B = Y + 1.77200 * (Cb - 128)

const FIX_1_40200: i32 = 91_881; // (int)(1.40200 * 65536 + 0.5)
const FIX_0_34414: i32 = 22_554; // (int)(0.34414 * 65536 + 0.5)
const FIX_0_71414: i32 = 46_802; // (int)(0.71414 * 65536 + 0.5)
const FIX_1_77200: i32 = 116_130; // (int)(1.77200 * 65536 + 0.5)
const ROUND: i32 = 1 << 15; // 0.5 in 16-bit fixed point

const fn clamp_to_u8(v: i32) -> u8 {
    if v < 0 {
        0
    } else if v > 255 {
        255
    } else {
        v as u8
    }
}

/// Convert one YCbCr pixel to RGB. `y`, `cb`, `cr` are the 8-bit component
/// values as read from the decoded block after IDCT and upsample.
///
/// Returns `(R, G, B)` clamped to `[0, 255]`.
pub(crate) fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let y = y as i32;
    let cb_centered = cb as i32 - 128;
    let cr_centered = cr as i32 - 128;
    let r = y + ((FIX_1_40200 * cr_centered + ROUND) >> 16);
    let g = y - ((FIX_0_34414 * cb_centered + FIX_0_71414 * cr_centered + ROUND) >> 16);
    let b = y + ((FIX_1_77200 * cb_centered + ROUND) >> 16);

    (clamp_to_u8(r), clamp_to_u8(g), clamp_to_u8(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_gray_roundtrips_to_equal_rgb_channels() {
        let (r, g, b) = ycbcr_to_rgb(128, 128, 128);
        assert_eq!((r, g, b), (128, 128, 128));
    }

    #[test]
    fn bright_red_maps_to_high_r_low_gb() {
        // libjpeg-turbo: Y=76 Cb=85 Cr=255 ≈ pure red (255, 0, 0).
        let (r, g, b) = ycbcr_to_rgb(76, 85, 255);
        assert!(r > 240 && g < 15 && b < 15, "got ({r}, {g}, {b})");
    }

    #[test]
    fn clamps_out_of_range_arithmetic_to_0_255() {
        // Y=255, large Cr pushes R arithmetic above 255 → saturate high.
        let (r, _, _) = ycbcr_to_rgb(255, 128, 255);
        assert_eq!(r, 255, "R must saturate at 255");
        // Y=255, large Cb pushes B arithmetic above 255 → saturate high.
        let (_, _, b) = ycbcr_to_rgb(255, 255, 128);
        assert_eq!(b, 255, "B must saturate at 255");
        // Y=0, small Cr pushes R arithmetic below 0 → saturate low.
        let (r, _, _) = ycbcr_to_rgb(0, 128, 0);
        assert_eq!(r, 0, "R must saturate at 0");
    }

    #[test]
    fn matches_libjpeg_turbo_fixed_point_expectations() {
        // Sampled checks against libjpeg-turbo jdcolor.c computed values.
        // Y=100 Cb=150 Cr=200 → R=201, G=41, B=139 (16-bit fixed point).
        let (r, g, b) = ycbcr_to_rgb(100, 150, 200);
        assert!((r as i32 - 201).abs() <= 1, "R={r}, expected ≈201");
        assert!((g as i32 - 41).abs() <= 1, "G={g}, expected ≈41");
        assert!((b as i32 - 139).abs() <= 1, "B={b}, expected ≈139");
    }
}
