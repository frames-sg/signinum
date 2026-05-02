// SPDX-License-Identifier: Apache-2.0

//! Reduced-size ISLOW IDCTs derived from libjpeg-turbo's `jidctred.c`.

use core::num::Wrapping;

const CONST_BITS: usize = 13;
const PASS1_BITS: usize = 2;

const FIX_0_211164243: Wrapping<i32> = Wrapping(1730);
const FIX_0_509795579: Wrapping<i32> = Wrapping(4176);
const FIX_0_601344887: Wrapping<i32> = Wrapping(4926);
const FIX_0_720959822: Wrapping<i32> = Wrapping(5906);
const FIX_0_765366865: Wrapping<i32> = Wrapping(6270);
const FIX_0_850430095: Wrapping<i32> = Wrapping(6967);
const FIX_0_899976223: Wrapping<i32> = Wrapping(7373);
const FIX_1_061594337: Wrapping<i32> = Wrapping(8697);
const FIX_1_272758580: Wrapping<i32> = Wrapping(10426);
const FIX_1_451774981: Wrapping<i32> = Wrapping(11893);
const FIX_1_847759065: Wrapping<i32> = Wrapping(15137);
const FIX_2_172734803: Wrapping<i32> = Wrapping(17799);
const FIX_2_562915447: Wrapping<i32> = Wrapping(20995);
const FIX_3_624509785: Wrapping<i32> = Wrapping(29692);

pub(crate) fn idct_islow_4x4(input: &[i16; 64], output: &mut [u8; 16]) {
    let mut work = [Wrapping(0i32); 32];
    for col in 0..8 {
        if col == 4 {
            continue;
        }
        idct_4x4_column(input, &mut work, col);
    }
    for row in 0..4 {
        idct_4x4_row(&work, output, row);
    }
}

pub(crate) fn idct_islow_4x4_dc_only(dc_coeff: i16, output: &mut [u8; 16]) {
    output.fill(dc_only_pixel(dc_coeff));
}

pub(crate) fn idct_islow_2x2(input: &[i16; 64], output: &mut [u8; 4]) {
    idct_islow_2x2_scalar(input, output);
}

pub(crate) fn idct_islow_2x2_dc_only(dc_coeff: i16, output: &mut [u8; 4]) {
    output.fill(dc_only_pixel(dc_coeff));
}

pub(crate) fn idct_islow_2x2_scalar(input: &[i16; 64], output: &mut [u8; 4]) {
    let mut work = [Wrapping(0i32); 16];
    for col in 0..8 {
        if col == 2 || col == 4 || col == 6 {
            continue;
        }
        idct_2x2_column(input, &mut work, col);
    }
    for row in 0..2 {
        idct_2x2_row(&work, output, row);
    }
}

pub(crate) fn idct_islow_1x1(input: &[i16; 64]) -> u8 {
    descale_and_clamp(Wrapping(input[0] as i32), 3)
}

#[inline]
fn dc_only_pixel(dc_coeff: i16) -> u8 {
    descale_and_clamp(Wrapping(i32::from(dc_coeff)), 3)
}

fn idct_4x4_column(input: &[i16; 64], work: &mut [Wrapping<i32>; 32], col: usize) {
    let p0 = Wrapping(input[col] as i32);
    let p1 = Wrapping(input[col + 8] as i32);
    let p2 = Wrapping(input[col + 16] as i32);
    let p3 = Wrapping(input[col + 24] as i32);
    let p5 = Wrapping(input[col + 40] as i32);
    let p6 = Wrapping(input[col + 48] as i32);
    let p7 = Wrapping(input[col + 56] as i32);

    if p1.0 == 0 && p2.0 == 0 && p3.0 == 0 && p5.0 == 0 && p6.0 == 0 && p7.0 == 0 {
        let dc = p0 << PASS1_BITS;
        work[col] = dc;
        work[8 + col] = dc;
        work[16 + col] = dc;
        work[24 + col] = dc;
        return;
    }

    let mut tmp0 = p0 << (CONST_BITS + 1);
    let z2 = p2;
    let z3 = p6;
    let tmp2 = z2 * FIX_1_847759065 + z3 * (-FIX_0_765366865);
    let tmp10 = tmp0 + tmp2;
    let tmp12 = tmp0 - tmp2;

    let z1 = p7;
    let z2 = p5;
    let z3 = p3;
    let z4 = p1;
    tmp0 = z1 * (-FIX_0_211164243)
        + z2 * FIX_1_451774981
        + z3 * (-FIX_2_172734803)
        + z4 * FIX_1_061594337;
    let tmp2 = z1 * (-FIX_0_509795579)
        + z2 * (-FIX_0_601344887)
        + z3 * FIX_0_899976223
        + z4 * FIX_2_562915447;

    let shift = CONST_BITS - PASS1_BITS + 1;
    work[col] = descale(tmp10 + tmp2, shift);
    work[24 + col] = descale(tmp10 - tmp2, shift);
    work[8 + col] = descale(tmp12 + tmp0, shift);
    work[16 + col] = descale(tmp12 - tmp0, shift);
}

fn idct_4x4_row(work: &[Wrapping<i32>; 32], output: &mut [u8; 16], row: usize) {
    let base = row * 8;
    let p0 = work[base];
    let p1 = work[base + 1];
    let p2 = work[base + 2];
    let p3 = work[base + 3];
    let p5 = work[base + 5];
    let p6 = work[base + 6];
    let p7 = work[base + 7];

    if p1.0 == 0 && p2.0 == 0 && p3.0 == 0 && p5.0 == 0 && p6.0 == 0 && p7.0 == 0 {
        let dc = descale_and_clamp(p0, PASS1_BITS + 3);
        let out = row * 4;
        output[out..out + 4].fill(dc);
        return;
    }

    let mut tmp0 = p0 << (CONST_BITS + 1);
    let tmp2 = p2 * FIX_1_847759065 + p6 * (-FIX_0_765366865);
    let tmp10 = tmp0 + tmp2;
    let tmp12 = tmp0 - tmp2;

    tmp0 = p7 * (-FIX_0_211164243)
        + p5 * FIX_1_451774981
        + p3 * (-FIX_2_172734803)
        + p1 * FIX_1_061594337;
    let tmp2 = p7 * (-FIX_0_509795579)
        + p5 * (-FIX_0_601344887)
        + p3 * FIX_0_899976223
        + p1 * FIX_2_562915447;

    let shift = CONST_BITS + PASS1_BITS + 3 + 1;
    let out = row * 4;
    output[out] = descale_and_clamp(tmp10 + tmp2, shift);
    output[out + 3] = descale_and_clamp(tmp10 - tmp2, shift);
    output[out + 1] = descale_and_clamp(tmp12 + tmp0, shift);
    output[out + 2] = descale_and_clamp(tmp12 - tmp0, shift);
}

fn idct_2x2_column(input: &[i16; 64], work: &mut [Wrapping<i32>; 16], col: usize) {
    let p0 = Wrapping(input[col] as i32);
    let p1 = Wrapping(input[col + 8] as i32);
    let p3 = Wrapping(input[col + 24] as i32);
    let p5 = Wrapping(input[col + 40] as i32);
    let p7 = Wrapping(input[col + 56] as i32);

    if p1.0 == 0 && p3.0 == 0 && p5.0 == 0 && p7.0 == 0 {
        let dc = p0 << PASS1_BITS;
        work[col] = dc;
        work[8 + col] = dc;
        return;
    }

    let tmp10 = p0 << (CONST_BITS + 2);
    let tmp0 = p7 * (-FIX_0_720959822)
        + p5 * FIX_0_850430095
        + p3 * (-FIX_1_272758580)
        + p1 * FIX_3_624509785;

    let shift = CONST_BITS - PASS1_BITS + 2;
    work[col] = descale(tmp10 + tmp0, shift);
    work[8 + col] = descale(tmp10 - tmp0, shift);
}

fn idct_2x2_row(work: &[Wrapping<i32>; 16], output: &mut [u8; 4], row: usize) {
    let base = row * 8;
    let p0 = work[base];
    let p1 = work[base + 1];
    let p3 = work[base + 3];
    let p5 = work[base + 5];
    let p7 = work[base + 7];

    if p1.0 == 0 && p3.0 == 0 && p5.0 == 0 && p7.0 == 0 {
        let dc = descale_and_clamp(p0, PASS1_BITS + 3);
        let out = row * 2;
        output[out] = dc;
        output[out + 1] = dc;
        return;
    }

    let tmp10 = p0 << (CONST_BITS + 2);
    let tmp0 = p7 * (-FIX_0_720959822)
        + p5 * FIX_0_850430095
        + p3 * (-FIX_1_272758580)
        + p1 * FIX_3_624509785;

    let shift = CONST_BITS + PASS1_BITS + 3 + 2;
    let out = row * 2;
    output[out] = descale_and_clamp(tmp10 + tmp0, shift);
    output[out + 1] = descale_and_clamp(tmp10 - tmp0, shift);
}

fn descale(value: Wrapping<i32>, shift: usize) -> Wrapping<i32> {
    Wrapping(value.0 >> shift)
}

fn descale_and_clamp(value: Wrapping<i32>, shift: usize) -> u8 {
    let shifted = value.0 >> shift;
    let level_shifted = shifted.wrapping_add(128);
    level_shifted.clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduced_idcts_preserve_zero_block_level_shift() {
        let input = [0i16; 64];
        let mut out4 = [0u8; 16];
        let mut out2 = [0u8; 4];
        idct_islow_4x4(&input, &mut out4);
        idct_islow_2x2(&input, &mut out2);
        assert!(out4.iter().all(|&px| px == 128));
        assert!(out2.iter().all(|&px| px == 128));
        assert_eq!(idct_islow_1x1(&input), 128);
    }

    #[test]
    fn reduced_dc_only_helpers_match_full_reduced_idcts() {
        for dc in [-300, -37, 0, 37, 300] {
            let mut input = [0i16; 64];
            input[0] = dc;
            let mut expected4 = [0u8; 16];
            let mut actual4 = [0u8; 16];
            let mut expected2 = [0u8; 4];
            let mut actual2 = [0u8; 4];

            idct_islow_4x4(&input, &mut expected4);
            idct_islow_4x4_dc_only(input[0], &mut actual4);
            idct_islow_2x2(&input, &mut expected2);
            idct_islow_2x2_dc_only(input[0], &mut actual2);

            assert_eq!(actual4, expected4);
            assert_eq!(actual2, expected2);
        }
    }
}
