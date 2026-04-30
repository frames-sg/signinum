// SPDX-License-Identifier: Apache-2.0

//! NEON SIMD integer ISLOW IDCT. Bit-exact with `super::scalar::idct_islow`.
//!
//! Layout: 8 column 1D IDCTs in parallel (pass 1, shift 11) → 8×8 i32
//! transpose (as four 4×4 i32 transposes) → 8 row 1D IDCTs in parallel
//! (pass 2, shift 18) → second 8×8 transpose → level shift +128 → saturate
//! to u8 → store.
//!
//! Intermediates live in `int32x4_t` throughout — same 32-bit wrapping
//! arithmetic as the scalar's `Wrapping<i32>`, so no information is lost
//! between passes even on adversarial coefficients.

use core::arch::aarch64::{
    int16x4_t, int16x8_t, int32x4_t, vaddq_s32, vcombine_s16, vcombine_s32, vdupq_n_s32,
    vget_high_s16, vget_high_s32, vget_low_s16, vget_low_s32, vgetq_lane_u64, vld1q_s16, vmovl_s16,
    vmulq_n_s32, vorrq_u64, vqmovn_s32, vqmovun_s16, vreinterpretq_u64_s16, vshlq_n_s32,
    vshrq_n_s32, vst1_u8, vsubq_s32, vtrnq_s32,
};

const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;

const FIX_0_298631336: i32 = 2_446;
const FIX_0_390180644: i32 = 3_196;
const FIX_0_541196100: i32 = 4_433;
const FIX_0_765366865: i32 = 6_270;
const FIX_0_899976223: i32 = 7_373;
const FIX_1_175875602: i32 = 9_633;
const FIX_1_501321110: i32 = 12_299;
const FIX_1_847759065: i32 = 15_137;
const FIX_1_961570560: i32 = 16_069;
const FIX_2_053119869: i32 = 16_819;
const FIX_2_562915447: i32 = 20_995;
const FIX_3_072711026: i32 = 25_172;

/// Inverse DCT of one 8×8 block. Output is level-shifted (+128) and
/// saturated to `[0, 255]`, matching the scalar path byte-for-byte.
///
/// # Safety
/// Caller ensures the target CPU supports NEON. On aarch64 NEON is
/// architecturally mandatory, so the dispatch in `Backend::detect` picks
/// this variant unconditionally for aarch64.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn idct_islow(input: &[i16; 64], output: &mut [u8; 64]) {
    const PASS1_SHIFT: i32 = CONST_BITS - PASS1_BITS;
    const PASS2_SHIFT: i32 = CONST_BITS + PASS1_BITS + 3;
    // Load 8 rows as int16x8_t so the common bottom-half-zero shortcut can
    // reuse the tail rows instead of rescanning coefficients scalar-by-scalar.
    let src = input.as_ptr();
    let row0 = unsafe { vld1q_s16(src) };
    let row1 = unsafe { vld1q_s16(src.add(8)) };
    let row2 = unsafe { vld1q_s16(src.add(16)) };
    let row3 = unsafe { vld1q_s16(src.add(24)) };
    let row4 = unsafe { vld1q_s16(src.add(32)) };
    let row5 = unsafe { vld1q_s16(src.add(40)) };
    let row6 = unsafe { vld1q_s16(src.add(48)) };
    let row7 = unsafe { vld1q_s16(src.add(56)) };
    let bottom_half_zero = bottom_half_rows_are_zero(row4, row5, row6, row7);
    if bottom_half_zero {
        unsafe {
            idct_islow_bottom_half_zero_rows(row0, row1, row2, row3, output);
        }
        return;
    }

    let (r0l, r0h) = widen(row0);
    let (r1l, r1h) = widen(row1);
    let (r2l, r2h) = widen(row2);
    let (r3l, r3h) = widen(row3);
    let (r4l, r4h) = widen(row4);
    let (r5l, r5h) = widen(row5);
    let (r6l, r6h) = widen(row6);
    let (r7l, r7h) = widen(row7);

    // Pass 1: column IDCT. `rN*` has lane `c` holding `(row N, col c)`, so
    // passing them as `p0..p7` makes each SIMD lane process one column's 8
    // samples. Output `cw_*[k]` has lane `c` holding pass-1 result at
    // (row k, col c).
    let round1 = vdupq_n_s32(1 << (PASS1_SHIFT - 1));
    let cw_lo = idct_1d_x4::<PASS1_SHIFT>(r0l, r1l, r2l, r3l, r4l, r5l, r6l, r7l, round1);
    let cw_hi = idct_1d_x4::<PASS1_SHIFT>(r0h, r1h, r2h, r3h, r4h, r5h, r6h, r7h, round1);

    // Transpose to pass-2 input. We need `q_c[l] = (row l, col c)`, meaning
    // 8 int32x4_t pairs where lane = row. Split into four independent 4×4
    // i32 transposes because `cw_lo` covers cols 0..3 and `cw_hi` covers
    // cols 4..7, each across two row halves.
    //
    // q_c_lo: 4 low rows (0..3) of col c.
    // q_c_hi: 4 high rows (4..7) of col c.
    let [q0l, q1l, q2l, q3l] = transpose_4x4_i32(cw_lo[0], cw_lo[1], cw_lo[2], cw_lo[3]);
    let [q4l, q5l, q6l, q7l] = transpose_4x4_i32(cw_hi[0], cw_hi[1], cw_hi[2], cw_hi[3]);
    let [q0h, q1h, q2h, q3h] = transpose_4x4_i32(cw_lo[4], cw_lo[5], cw_lo[6], cw_lo[7]);
    let [q4h, q5h, q6h, q7h] = transpose_4x4_i32(cw_hi[4], cw_hi[5], cw_hi[6], cw_hi[7]);

    // Pass 2: row IDCT.
    let round2 = vdupq_n_s32(1 << (PASS2_SHIFT - 1));

    let rw_lo = idct_1d_x4::<PASS2_SHIFT>(q0l, q1l, q2l, q3l, q4l, q5l, q6l, q7l, round2);
    let rw_hi = idct_1d_x4::<PASS2_SHIFT>(q0h, q1h, q2h, q3h, q4h, q5h, q6h, q7h, round2);

    // `rw_lo[k]` lane `l` = (row l, col k) for rows 0..3; `rw_hi[k]` for
    // rows 4..7. Transpose back to row-major, applying the +128 level
    // shift, then pack to u8.
    let bias = vdupq_n_s32(128);

    // Transpose rw_lo (cols 0..7 × rows 0..3) into row-major low (cols 0..3).
    let [fll0, fll1, fll2, fll3] = transpose_4x4_i32(
        vaddq_s32(rw_lo[0], bias),
        vaddq_s32(rw_lo[1], bias),
        vaddq_s32(rw_lo[2], bias),
        vaddq_s32(rw_lo[3], bias),
    );
    let [flh0, flh1, flh2, flh3] = transpose_4x4_i32(
        vaddq_s32(rw_lo[4], bias),
        vaddq_s32(rw_lo[5], bias),
        vaddq_s32(rw_lo[6], bias),
        vaddq_s32(rw_lo[7], bias),
    );
    let [fhl0, fhl1, fhl2, fhl3] = transpose_4x4_i32(
        vaddq_s32(rw_hi[0], bias),
        vaddq_s32(rw_hi[1], bias),
        vaddq_s32(rw_hi[2], bias),
        vaddq_s32(rw_hi[3], bias),
    );
    let [fhh0, fhh1, fhh2, fhh3] = transpose_4x4_i32(
        vaddq_s32(rw_hi[4], bias),
        vaddq_s32(rw_hi[5], bias),
        vaddq_s32(rw_hi[6], bias),
        vaddq_s32(rw_hi[7], bias),
    );

    // `fll_r` = row r (0..3), cols 0..3 as i32x4.
    // `flh_r` = row r (0..3), cols 4..7.
    // `fhl_r` = row r (4..7), cols 0..3.
    // `fhh_r` = row r (4..7), cols 4..7.
    let store = output.as_mut_ptr();
    unsafe {
        store_row(store, fll0, flh0);
        store_row(store.add(8), fll1, flh1);
        store_row(store.add(16), fll2, flh2);
        store_row(store.add(24), fll3, flh3);
        store_row(store.add(32), fhl0, fhh0);
        store_row(store.add(40), fhl1, fhh1);
        store_row(store.add(48), fhl2, fhh2);
        store_row(store.add(56), fhl3, fhh3);
    }
}

/// Inverse DCT for blocks whose natural-order rows 4..7 are known to be zero.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn idct_islow_bottom_half_zero(input: &[i16; 64], output: &mut [u8; 64]) {
    let src = input.as_ptr();
    unsafe {
        idct_islow_bottom_half_zero_rows(
            vld1q_s16(src),
            vld1q_s16(src.add(8)),
            vld1q_s16(src.add(16)),
            vld1q_s16(src.add(24)),
            output,
        );
    }
}

#[target_feature(enable = "neon")]
unsafe fn idct_islow_bottom_half_zero_rows(
    row0: int16x8_t,
    row1: int16x8_t,
    row2: int16x8_t,
    row3: int16x8_t,
    output: &mut [u8; 64],
) {
    const PASS1_SHIFT: i32 = CONST_BITS - PASS1_BITS;
    const PASS2_SHIFT: i32 = CONST_BITS + PASS1_BITS + 3;
    let (r0l, r0h) = widen(row0);
    let (r1l, r1h) = widen(row1);
    let (r2l, r2h) = widen(row2);
    let (r3l, r3h) = widen(row3);

    let round1 = vdupq_n_s32(1 << (PASS1_SHIFT - 1));
    let cw_lo = idct_1d_x4_bottom_half_zero::<PASS1_SHIFT>(r0l, r1l, r2l, r3l, round1);
    let cw_hi = idct_1d_x4_bottom_half_zero::<PASS1_SHIFT>(r0h, r1h, r2h, r3h, round1);

    let [q0l, q1l, q2l, q3l] = transpose_4x4_i32(cw_lo[0], cw_lo[1], cw_lo[2], cw_lo[3]);
    let [q4l, q5l, q6l, q7l] = transpose_4x4_i32(cw_hi[0], cw_hi[1], cw_hi[2], cw_hi[3]);
    let [q0h, q1h, q2h, q3h] = transpose_4x4_i32(cw_lo[4], cw_lo[5], cw_lo[6], cw_lo[7]);
    let [q4h, q5h, q6h, q7h] = transpose_4x4_i32(cw_hi[4], cw_hi[5], cw_hi[6], cw_hi[7]);

    let round2 = vdupq_n_s32(1 << (PASS2_SHIFT - 1));
    let rw_lo = idct_1d_x4::<PASS2_SHIFT>(q0l, q1l, q2l, q3l, q4l, q5l, q6l, q7l, round2);
    let rw_hi = idct_1d_x4::<PASS2_SHIFT>(q0h, q1h, q2h, q3h, q4h, q5h, q6h, q7h, round2);

    let bias = vdupq_n_s32(128);
    let [fll0, fll1, fll2, fll3] = transpose_4x4_i32(
        vaddq_s32(rw_lo[0], bias),
        vaddq_s32(rw_lo[1], bias),
        vaddq_s32(rw_lo[2], bias),
        vaddq_s32(rw_lo[3], bias),
    );
    let [flh0, flh1, flh2, flh3] = transpose_4x4_i32(
        vaddq_s32(rw_lo[4], bias),
        vaddq_s32(rw_lo[5], bias),
        vaddq_s32(rw_lo[6], bias),
        vaddq_s32(rw_lo[7], bias),
    );
    let [fhl0, fhl1, fhl2, fhl3] = transpose_4x4_i32(
        vaddq_s32(rw_hi[0], bias),
        vaddq_s32(rw_hi[1], bias),
        vaddq_s32(rw_hi[2], bias),
        vaddq_s32(rw_hi[3], bias),
    );
    let [fhh0, fhh1, fhh2, fhh3] = transpose_4x4_i32(
        vaddq_s32(rw_hi[4], bias),
        vaddq_s32(rw_hi[5], bias),
        vaddq_s32(rw_hi[6], bias),
        vaddq_s32(rw_hi[7], bias),
    );

    let store = output.as_mut_ptr();
    unsafe {
        store_row(store, fll0, flh0);
        store_row(store.add(8), fll1, flh1);
        store_row(store.add(16), fll2, flh2);
        store_row(store.add(24), fll3, flh3);
        store_row(store.add(32), fhl0, fhh0);
        store_row(store.add(40), fhl1, fhh1);
        store_row(store.add(48), fhl2, fhh2);
        store_row(store.add(56), fhl3, fhh3);
    }
}

#[inline]
#[cfg(test)]
fn bottom_half_is_zero(input: &[i16; 64]) -> bool {
    let tail = unsafe { input.as_ptr().add(32) };
    unsafe {
        bottom_half_rows_are_zero(
            vld1q_s16(tail),
            vld1q_s16(tail.add(8)),
            vld1q_s16(tail.add(16)),
            vld1q_s16(tail.add(24)),
        )
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn bottom_half_rows_are_zero(
    row4: int16x8_t,
    row5: int16x8_t,
    row6: int16x8_t,
    row7: int16x8_t,
) -> bool {
    let bottom = vorrq_u64(
        vorrq_u64(vreinterpretq_u64_s16(row4), vreinterpretq_u64_s16(row5)),
        vorrq_u64(vreinterpretq_u64_s16(row6), vreinterpretq_u64_s16(row7)),
    );
    vgetq_lane_u64::<0>(bottom) == 0 && vgetq_lane_u64::<1>(bottom) == 0
}

/// Saturating-narrow an (i32x4, i32x4) pair to u8x8 and store at `dst`.
/// Matches the scalar's `wrapping_add(128).clamp(0, 255)` on finite
/// 32-bit values: the narrowing cascade `i32 → i16 → u8` with saturation
/// at each step produces the same u8 as the scalar's explicit clamp.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_row(dst: *mut u8, lo: int32x4_t, hi: int32x4_t) {
    let packed_i16: int16x8_t = vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi));
    unsafe {
        vst1_u8(dst, vqmovun_s16(packed_i16));
    }
}

/// One 1D IDCT pass over 4 lanes of i32. Eight inputs carrying the 4 column
/// values (pass 1) or 4 row values (pass 2) — output order matches the
/// scalar descale positions.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
fn idct_1d_x4<const SHIFT: i32>(
    p0: int32x4_t,
    p1: int32x4_t,
    p2: int32x4_t,
    p3: int32x4_t,
    p4: int32x4_t,
    p5: int32x4_t,
    p6: int32x4_t,
    p7: int32x4_t,
    rounding: int32x4_t,
) -> [int32x4_t; 8] {
    // Even half.
    let z1 = vmulq_n_s32(vaddq_s32(p2, p6), FIX_0_541196100);
    let tmp2 = vaddq_s32(z1, vmulq_n_s32(p6, -FIX_1_847759065));
    let tmp3 = vaddq_s32(z1, vmulq_n_s32(p2, FIX_0_765366865));
    let tmp0 = vshlq_n_s32::<CONST_BITS>(vaddq_s32(p0, p4));
    let tmp1 = vshlq_n_s32::<CONST_BITS>(vsubq_s32(p0, p4));
    let tmp10 = vaddq_s32(tmp0, tmp3);
    let tmp13 = vsubq_s32(tmp0, tmp3);
    let tmp11 = vaddq_s32(tmp1, tmp2);
    let tmp12 = vsubq_s32(tmp1, tmp2);

    // Odd half. Scalar aliases p7 → tmp0, p5 → tmp1, p3 → tmp2, p1 → tmp3.
    let z1o = vaddq_s32(p7, p1);
    let z2o = vaddq_s32(p5, p3);
    let z3o = vaddq_s32(p7, p3);
    let z4o = vaddq_s32(p5, p1);
    let z5 = vmulq_n_s32(vaddq_s32(z3o, z4o), FIX_1_175875602);

    let o0 = vmulq_n_s32(p7, FIX_0_298631336);
    let o1 = vmulq_n_s32(p5, FIX_2_053119869);
    let o2 = vmulq_n_s32(p3, FIX_3_072711026);
    let o3 = vmulq_n_s32(p1, FIX_1_501321110);
    let z1m = vmulq_n_s32(z1o, -FIX_0_899976223);
    let z2m = vmulq_n_s32(z2o, -FIX_2_562915447);
    let z3m = vmulq_n_s32(z3o, -FIX_1_961570560);
    let z4m = vmulq_n_s32(z4o, -FIX_0_390180644);
    let z3f = vaddq_s32(z3m, z5);
    let z4f = vaddq_s32(z4m, z5);

    let k0 = vaddq_s32(vaddq_s32(o0, z1m), z3f);
    let k1 = vaddq_s32(vaddq_s32(o1, z2m), z4f);
    let k2 = vaddq_s32(vaddq_s32(o2, z2m), z3f);
    let k3 = vaddq_s32(vaddq_s32(o3, z1m), z4f);

    let out0 = vshrq_n_s32::<SHIFT>(vaddq_s32(vaddq_s32(tmp10, k3), rounding));
    let out7 = vshrq_n_s32::<SHIFT>(vaddq_s32(vsubq_s32(tmp10, k3), rounding));
    let out1 = vshrq_n_s32::<SHIFT>(vaddq_s32(vaddq_s32(tmp11, k2), rounding));
    let out6 = vshrq_n_s32::<SHIFT>(vaddq_s32(vsubq_s32(tmp11, k2), rounding));
    let out2 = vshrq_n_s32::<SHIFT>(vaddq_s32(vaddq_s32(tmp12, k1), rounding));
    let out5 = vshrq_n_s32::<SHIFT>(vaddq_s32(vsubq_s32(tmp12, k1), rounding));
    let out3 = vshrq_n_s32::<SHIFT>(vaddq_s32(vaddq_s32(tmp13, k0), rounding));
    let out4 = vshrq_n_s32::<SHIFT>(vaddq_s32(vsubq_s32(tmp13, k0), rounding));

    [out0, out1, out2, out3, out4, out5, out6, out7]
}

#[target_feature(enable = "neon")]
fn idct_1d_x4_bottom_half_zero<const SHIFT: i32>(
    p0: int32x4_t,
    p1: int32x4_t,
    p2: int32x4_t,
    p3: int32x4_t,
    rounding: int32x4_t,
) -> [int32x4_t; 8] {
    let z1 = vmulq_n_s32(p2, FIX_0_541196100);
    let tmp2 = z1;
    let tmp3 = vaddq_s32(z1, vmulq_n_s32(p2, FIX_0_765366865));
    let tmp0 = vshlq_n_s32::<CONST_BITS>(p0);
    let tmp1 = tmp0;
    let tmp10 = vaddq_s32(tmp0, tmp3);
    let tmp13 = vsubq_s32(tmp0, tmp3);
    let tmp11 = vaddq_s32(tmp1, tmp2);
    let tmp12 = vsubq_s32(tmp1, tmp2);

    let z5 = vmulq_n_s32(vaddq_s32(p1, p3), FIX_1_175875602);
    let z1m = vmulq_n_s32(p1, -FIX_0_899976223);
    let z2m = vmulq_n_s32(p3, -FIX_2_562915447);
    let z3f = vaddq_s32(vmulq_n_s32(p3, -FIX_1_961570560), z5);
    let z4f = vaddq_s32(vmulq_n_s32(p1, -FIX_0_390180644), z5);

    let k0 = vaddq_s32(z1m, z3f);
    let k1 = vaddq_s32(z2m, z4f);
    let k2 = vaddq_s32(vaddq_s32(vmulq_n_s32(p3, FIX_3_072711026), z2m), z3f);
    let k3 = vaddq_s32(vaddq_s32(vmulq_n_s32(p1, FIX_1_501321110), z1m), z4f);

    let out0 = vshrq_n_s32::<SHIFT>(vaddq_s32(vaddq_s32(tmp10, k3), rounding));
    let out7 = vshrq_n_s32::<SHIFT>(vaddq_s32(vsubq_s32(tmp10, k3), rounding));
    let out1 = vshrq_n_s32::<SHIFT>(vaddq_s32(vaddq_s32(tmp11, k2), rounding));
    let out6 = vshrq_n_s32::<SHIFT>(vaddq_s32(vsubq_s32(tmp11, k2), rounding));
    let out2 = vshrq_n_s32::<SHIFT>(vaddq_s32(vaddq_s32(tmp12, k1), rounding));
    let out5 = vshrq_n_s32::<SHIFT>(vaddq_s32(vsubq_s32(tmp12, k1), rounding));
    let out3 = vshrq_n_s32::<SHIFT>(vaddq_s32(vaddq_s32(tmp13, k0), rounding));
    let out4 = vshrq_n_s32::<SHIFT>(vaddq_s32(vsubq_s32(tmp13, k0), rounding));

    [out0, out1, out2, out3, out4, out5, out6, out7]
}

#[inline]
#[target_feature(enable = "neon")]
fn widen(row: int16x8_t) -> (int32x4_t, int32x4_t) {
    (vmovl_s16(vget_low_s16(row)), vmovl_s16(vget_high_s16(row)))
}

/// 4×4 i32 transpose. Input rows a,b,c,d; output the 4 columns.
#[inline]
#[target_feature(enable = "neon")]
fn transpose_4x4_i32(a: int32x4_t, b: int32x4_t, c: int32x4_t, d: int32x4_t) -> [int32x4_t; 4] {
    let t01 = vtrnq_s32(a, b);
    let t23 = vtrnq_s32(c, d);
    let col0 = vcombine_s32(vget_low_s32(t01.0), vget_low_s32(t23.0));
    let col1 = vcombine_s32(vget_low_s32(t01.1), vget_low_s32(t23.1));
    let col2 = vcombine_s32(vget_high_s32(t01.0), vget_high_s32(t23.0));
    let col3 = vcombine_s32(vget_high_s32(t01.1), vget_high_s32(t23.1));
    [col0, col1, col2, col3]
}

// Silence unused-helper warnings left from the prior draft.
#[allow(dead_code)]
fn _unused_refs(_: int16x4_t) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idct::scalar::idct_islow as idct_scalar;

    fn run_both(input: &[i16; 64]) -> ([u8; 64], [u8; 64]) {
        let mut scalar_out = [0u8; 64];
        idct_scalar(input, &mut scalar_out);
        let mut neon_out = [0u8; 64];
        unsafe { idct_islow(input, &mut neon_out) };
        (scalar_out, neon_out)
    }

    #[test]
    fn neon_matches_scalar_on_all_zero_input() {
        let (s, n) = run_both(&[0; 64]);
        assert_eq!(s, n);
    }

    #[test]
    fn neon_matches_scalar_on_dc_only_input() {
        let mut input = [0i16; 64];
        input[0] = 8 * 8;
        let (s, n) = run_both(&input);
        assert_eq!(s, n);
    }

    #[test]
    fn neon_matches_scalar_on_saturation_block() {
        let mut input = [0i16; 64];
        input[0] = i16::MAX;
        let (s, n) = run_both(&input);
        assert_eq!(s, n);
    }

    #[test]
    fn neon_matches_scalar_on_saturation_block_negative() {
        let mut input = [0i16; 64];
        input[0] = i16::MIN;
        let (s, n) = run_both(&input);
        assert_eq!(s, n);
    }

    #[test]
    fn neon_matches_scalar_on_horizontal_basis() {
        let mut input = [0i16; 64];
        input[1] = 400;
        let (s, n) = run_both(&input);
        assert_eq!(s, n);
    }

    #[test]
    fn neon_matches_scalar_on_mixed_coefficients() {
        let mut input = [0i16; 64];
        input[0] = 64;
        input[1] = 24;
        input[2] = -12;
        input[8] = 18;
        input[9] = -7;
        input[16] = 5;
        let (s, n) = run_both(&input);
        assert_eq!(s, n);
    }

    #[test]
    fn neon_bottom_half_zero_specialization_matches_scalar() {
        let mut input = [0i16; 64];
        input[0] = 64;
        input[1] = 24;
        input[2] = -12;
        input[8] = 18;
        input[9] = -7;
        input[16] = 5;
        let mut scalar_out = [0u8; 64];
        idct_scalar(&input, &mut scalar_out);
        let mut neon_out = [0u8; 64];
        unsafe { idct_islow_bottom_half_zero(&input, &mut neon_out) };
        assert_eq!(scalar_out, neon_out);
    }

    #[test]
    fn bottom_half_zero_detects_zero_and_nonzero_tails() {
        let mut block = [0i16; 64];
        block[0] = 7;
        assert!(bottom_half_is_zero(&block));

        block[32] = 1;
        assert!(!bottom_half_is_zero(&block));

        block[32] = 0;
        block[63] = -1;
        assert!(!bottom_half_is_zero(&block));
    }
}
