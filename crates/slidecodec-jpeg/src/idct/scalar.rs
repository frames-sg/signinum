// SPDX-License-Identifier: Apache-2.0

//! Integer "slow" IDCT (ISLOW) — Chen-Wang decomposition, 16-bit fixed point.
//! Bit-exact with libjpeg-turbo's `jidctint.c` algorithm on the same inputs.
//!
//! Input: 64 dequantized DCT coefficients in natural (row-major) order, each
//! already multiplied by its quantization entry.
//!
//! Output: 64 u8 pixel samples in natural order, level-shifted by +128 and
//! clamped to `[0, 255]`.

#![allow(dead_code)]

const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;

const FIX_0_298631336: i32 = 2446;
const FIX_0_390180644: i32 = 3196;
const FIX_0_541196100: i32 = 4433;
const FIX_0_765366865: i32 = 6270;
const FIX_0_899976223: i32 = 7373;
const FIX_1_175875602: i32 = 9633;
const FIX_1_501321110: i32 = 12299;
const FIX_1_847759065: i32 = 15137;
const FIX_1_961570560: i32 = 16069;
const FIX_2_053119869: i32 = 16819;
const FIX_2_562915447: i32 = 20995;
const FIX_3_072711026: i32 = 25172;

/// Inverse DCT of a single 8×8 block, with level shift and clamping.
pub(crate) fn idct_islow(input: &[i16; 64], output: &mut [u8; 64]) {
    let mut work = [0i32; 64];
    for col in 0..8 {
        idct_1d_column(input, &mut work, col);
    }
    for row in 0..8 {
        idct_1d_row(&work, output, row);
    }
}

fn idct_1d_column(input: &[i16; 64], work: &mut [i32; 64], col: usize) {
    let p0 = input[col] as i32;
    let p1 = input[col + 8] as i32;
    let p2 = input[col + 16] as i32;
    let p3 = input[col + 24] as i32;
    let p4 = input[col + 32] as i32;
    let p5 = input[col + 40] as i32;
    let p6 = input[col + 48] as i32;
    let p7 = input[col + 56] as i32;

    if p1 == 0 && p2 == 0 && p3 == 0 && p4 == 0 && p5 == 0 && p6 == 0 && p7 == 0 {
        let dc = p0 << PASS1_BITS;
        work[col] = dc;
        work[col + 8] = dc;
        work[col + 16] = dc;
        work[col + 24] = dc;
        work[col + 32] = dc;
        work[col + 40] = dc;
        work[col + 48] = dc;
        work[col + 56] = dc;
        return;
    }

    let z2 = p2;
    let z3 = p6;
    let z1 = (z2 + z3) * FIX_0_541196100;
    let tmp2 = z1 + z3 * (-FIX_1_847759065);
    let tmp3 = z1 + z2 * FIX_0_765366865;

    let z2 = p0;
    let z3 = p4;
    let tmp0 = (z2 + z3) << CONST_BITS;
    let tmp1 = (z2 - z3) << CONST_BITS;

    let tmp10 = tmp0 + tmp3;
    let tmp13 = tmp0 - tmp3;
    let tmp11 = tmp1 + tmp2;
    let tmp12 = tmp1 - tmp2;

    let tmp0 = p7;
    let tmp1 = p5;
    let tmp2 = p3;
    let tmp3 = p1;

    let z1 = tmp0 + tmp3;
    let z2 = tmp1 + tmp2;
    let z3 = tmp0 + tmp2;
    let z4 = tmp1 + tmp3;
    let z5 = (z3 + z4) * FIX_1_175875602;

    let tmp0 = tmp0 * FIX_0_298631336;
    let tmp1 = tmp1 * FIX_2_053119869;
    let tmp2 = tmp2 * FIX_3_072711026;
    let tmp3 = tmp3 * FIX_1_501321110;
    let z1 = z1 * (-FIX_0_899976223);
    let z2 = z2 * (-FIX_2_562915447);
    let z3 = z3 * (-FIX_1_961570560);
    let z4 = z4 * (-FIX_0_390180644);

    let z3 = z3 + z5;
    let z4 = z4 + z5;

    let tmp0 = tmp0 + z1 + z3;
    let tmp1 = tmp1 + z2 + z4;
    let tmp2 = tmp2 + z2 + z3;
    let tmp3 = tmp3 + z1 + z4;

    let shift = CONST_BITS - PASS1_BITS;
    let rounding = 1i32 << (shift - 1);
    work[col] = (tmp10 + tmp3 + rounding) >> shift;
    work[col + 56] = (tmp10 - tmp3 + rounding) >> shift;
    work[col + 8] = (tmp11 + tmp2 + rounding) >> shift;
    work[col + 48] = (tmp11 - tmp2 + rounding) >> shift;
    work[col + 16] = (tmp12 + tmp1 + rounding) >> shift;
    work[col + 40] = (tmp12 - tmp1 + rounding) >> shift;
    work[col + 24] = (tmp13 + tmp0 + rounding) >> shift;
    work[col + 32] = (tmp13 - tmp0 + rounding) >> shift;
}

fn idct_1d_row(work: &[i32; 64], output: &mut [u8; 64], row: usize) {
    let base = row * 8;
    let p0 = work[base];
    let p1 = work[base + 1];
    let p2 = work[base + 2];
    let p3 = work[base + 3];
    let p4 = work[base + 4];
    let p5 = work[base + 5];
    let p6 = work[base + 6];
    let p7 = work[base + 7];

    let shift = CONST_BITS + PASS1_BITS + 3;
    let rounding = 1i32 << (shift - 1);

    if p1 == 0 && p2 == 0 && p3 == 0 && p4 == 0 && p5 == 0 && p6 == 0 && p7 == 0 {
        let dc_shift = PASS1_BITS + 3;
        let rounding_dc = 1i32 << (dc_shift - 1);
        let pixel = descale_and_clamp(p0 + rounding_dc, dc_shift);
        for i in 0..8 {
            output[base + i] = pixel;
        }
        return;
    }

    let z2 = p2;
    let z3 = p6;
    let z1 = (z2 + z3) * FIX_0_541196100;
    let tmp2 = z1 + z3 * (-FIX_1_847759065);
    let tmp3 = z1 + z2 * FIX_0_765366865;

    let tmp0 = (p0 + p4) << CONST_BITS;
    let tmp1 = (p0 - p4) << CONST_BITS;

    let tmp10 = tmp0 + tmp3;
    let tmp13 = tmp0 - tmp3;
    let tmp11 = tmp1 + tmp2;
    let tmp12 = tmp1 - tmp2;

    let tmp0 = p7;
    let tmp1 = p5;
    let tmp2 = p3;
    let tmp3 = p1;

    let z1 = tmp0 + tmp3;
    let z2 = tmp1 + tmp2;
    let z3 = tmp0 + tmp2;
    let z4 = tmp1 + tmp3;
    let z5 = (z3 + z4) * FIX_1_175875602;

    let tmp0 = tmp0 * FIX_0_298631336;
    let tmp1 = tmp1 * FIX_2_053119869;
    let tmp2 = tmp2 * FIX_3_072711026;
    let tmp3 = tmp3 * FIX_1_501321110;
    let z1 = z1 * (-FIX_0_899976223);
    let z2 = z2 * (-FIX_2_562915447);
    let z3 = z3 * (-FIX_1_961570560);
    let z4 = z4 * (-FIX_0_390180644);

    let z3 = z3 + z5;
    let z4 = z4 + z5;

    let tmp0 = tmp0 + z1 + z3;
    let tmp1 = tmp1 + z2 + z4;
    let tmp2 = tmp2 + z2 + z3;
    let tmp3 = tmp3 + z1 + z4;

    output[base] = descale_and_clamp(tmp10 + tmp3 + rounding, shift);
    output[base + 7] = descale_and_clamp(tmp10 - tmp3 + rounding, shift);
    output[base + 1] = descale_and_clamp(tmp11 + tmp2 + rounding, shift);
    output[base + 6] = descale_and_clamp(tmp11 - tmp2 + rounding, shift);
    output[base + 2] = descale_and_clamp(tmp12 + tmp1 + rounding, shift);
    output[base + 5] = descale_and_clamp(tmp12 - tmp1 + rounding, shift);
    output[base + 3] = descale_and_clamp(tmp13 + tmp0 + rounding, shift);
    output[base + 4] = descale_and_clamp(tmp13 - tmp0 + rounding, shift);
}

fn descale_and_clamp(value: i32, shift: i32) -> u8 {
    let shifted = value >> shift;
    let level_shifted = shifted + 128;
    level_shifted.clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_zero_input_produces_level_shifted_gray_block() {
        let input = [0i16; 64];
        let mut output = [0u8; 64];
        idct_islow(&input, &mut output);
        for (i, &px) in output.iter().enumerate() {
            assert_eq!(px, 128, "pixel {i} = {px}, expected 128");
        }
    }

    #[test]
    fn dc_only_input_produces_uniform_block() {
        let mut input = [0i16; 64];
        input[0] = 8 * 8;
        let mut output = [0u8; 64];
        idct_islow(&input, &mut output);
        for &px in &output {
            assert!((px as i32 - 136).abs() <= 1, "got {px}");
        }
    }

    #[test]
    fn clamps_extreme_coefficients_into_0_255() {
        let mut input = [0i16; 64];
        input[0] = i16::MAX;
        let mut output = [0u8; 64];
        idct_islow(&input, &mut output);
        assert!(output.iter().all(|&px| px == 255));

        let mut input = [0i16; 64];
        input[0] = i16::MIN;
        let mut output = [0u8; 64];
        idct_islow(&input, &mut output);
        assert!(output.iter().all(|&px| px == 0));
    }

    #[test]
    fn roundtrip_identity_basis_reconstructs_8x8_impulse() {
        let mut input = [0i16; 64];
        input[1] = 400;
        let mut output = [0u8; 64];
        idct_islow(&input, &mut output);
        let left = output[0] as i32;
        let right = output[7] as i32;
        assert!(
            (left - right).abs() > 40,
            "AC[1] basis should produce horizontal variation, got L={left} R={right}"
        );
    }
}
