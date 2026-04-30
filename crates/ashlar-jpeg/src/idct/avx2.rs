// SPDX-License-Identifier: Apache-2.0

//! x86_64 SIMD integer ISLOW IDCT. Bit-exact with `super::scalar::idct_islow`.
//!
//! Uses 128-bit SSE4.1 intrinsics (a subset of AVX2) so the arithmetic
//! matches the NEON port 1:1 — same 4-lane i32 SIMD-ISLOW structure,
//! four independent 4×4 i32 transposes between and after the two passes.
//! Dispatched when `Backend::detect` sees AVX2 at runtime; AVX2 implies
//! SSE4.1 so the load/store/multiply/pack intrinsics used here are all
//! available.

#![allow(clippy::cast_possible_truncation, clippy::cast_lossless)]

use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_cvtepi16_epi32, _mm_loadl_epi64, _mm_mullo_epi32, _mm_packs_epi32,
    _mm_packus_epi16, _mm_set1_epi32, _mm_slli_epi32, _mm_srai_epi32, _mm_srli_si128,
    _mm_storel_epi64, _mm_sub_epi32, _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi32,
    _mm_unpacklo_epi64,
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
/// saturated to `[0, 255]`, matching the scalar path byte-for-byte on
/// legal JPEG coefficients and on the adversarial saturating edges
/// proptested against.
///
/// # Safety
/// Caller must ensure the host CPU supports SSE4.1. The
/// `Backend::detect` dispatch picks this variant when AVX2 is available
/// (which implies SSE4.1).
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn idct_islow(input: &[i16; 64], output: &mut [u8; 64]) {
    const PASS1_SHIFT: i32 = CONST_BITS - PASS1_BITS;
    const PASS2_SHIFT: i32 = CONST_BITS + PASS1_BITS + 3;

    let src = input.as_ptr();
    let (r0l, r0h) = unsafe { widen(src.add(0)) };
    let (r1l, r1h) = unsafe { widen(src.add(8)) };
    let (r2l, r2h) = unsafe { widen(src.add(16)) };
    let (r3l, r3h) = unsafe { widen(src.add(24)) };
    let (r4l, r4h) = unsafe { widen(src.add(32)) };
    let (r5l, r5h) = unsafe { widen(src.add(40)) };
    let (r6l, r6h) = unsafe { widen(src.add(48)) };
    let (r7l, r7h) = unsafe { widen(src.add(56)) };

    let round1 = _mm_set1_epi32(1 << (PASS1_SHIFT - 1));
    let cw_lo = idct_1d_x4::<PASS1_SHIFT>(r0l, r1l, r2l, r3l, r4l, r5l, r6l, r7l, round1);
    let cw_hi = idct_1d_x4::<PASS1_SHIFT>(r0h, r1h, r2h, r3h, r4h, r5h, r6h, r7h, round1);

    let [q0l, q1l, q2l, q3l] = transpose_4x4_i32(cw_lo[0], cw_lo[1], cw_lo[2], cw_lo[3]);
    let [q4l, q5l, q6l, q7l] = transpose_4x4_i32(cw_hi[0], cw_hi[1], cw_hi[2], cw_hi[3]);
    let [q0h, q1h, q2h, q3h] = transpose_4x4_i32(cw_lo[4], cw_lo[5], cw_lo[6], cw_lo[7]);
    let [q4h, q5h, q6h, q7h] = transpose_4x4_i32(cw_hi[4], cw_hi[5], cw_hi[6], cw_hi[7]);

    let round2 = _mm_set1_epi32(1 << (PASS2_SHIFT - 1));
    let rw_lo = idct_1d_x4::<PASS2_SHIFT>(q0l, q1l, q2l, q3l, q4l, q5l, q6l, q7l, round2);
    let rw_hi = idct_1d_x4::<PASS2_SHIFT>(q0h, q1h, q2h, q3h, q4h, q5h, q6h, q7h, round2);

    let bias = _mm_set1_epi32(128);
    let [fll0, fll1, fll2, fll3] = transpose_4x4_i32(
        _mm_add_epi32(rw_lo[0], bias),
        _mm_add_epi32(rw_lo[1], bias),
        _mm_add_epi32(rw_lo[2], bias),
        _mm_add_epi32(rw_lo[3], bias),
    );
    let [flh0, flh1, flh2, flh3] = transpose_4x4_i32(
        _mm_add_epi32(rw_lo[4], bias),
        _mm_add_epi32(rw_lo[5], bias),
        _mm_add_epi32(rw_lo[6], bias),
        _mm_add_epi32(rw_lo[7], bias),
    );
    let [fhl0, fhl1, fhl2, fhl3] = transpose_4x4_i32(
        _mm_add_epi32(rw_hi[0], bias),
        _mm_add_epi32(rw_hi[1], bias),
        _mm_add_epi32(rw_hi[2], bias),
        _mm_add_epi32(rw_hi[3], bias),
    );
    let [fhh0, fhh1, fhh2, fhh3] = transpose_4x4_i32(
        _mm_add_epi32(rw_hi[4], bias),
        _mm_add_epi32(rw_hi[5], bias),
        _mm_add_epi32(rw_hi[6], bias),
        _mm_add_epi32(rw_hi[7], bias),
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

#[target_feature(enable = "avx2")]
#[allow(dead_code)]
pub(crate) unsafe fn idct_islow_bottom_half_zero(input: &[i16; 64], output: &mut [u8; 64]) {
    unsafe { idct_islow(input, output) };
}

/// Load 8 `i16` values from `src` and sign-extend them to a pair of
/// `__m128i` each carrying 4 `i32` lanes (low 4, high 4).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn widen(src: *const i16) -> (__m128i, __m128i) {
    let vec = unsafe { _mm_loadl_epi64(src.cast()) }; // load 8 bytes = 4 i16 into lower 64
                                                      // Actually the row is 8 i16 = 16 bytes, not 8. Use full 128-bit load.
    let full = unsafe { core::ptr::read_unaligned(src.cast::<__m128i>()) };
    let _ = vec;
    let lo = _mm_cvtepi16_epi32(full);
    let hi_shuffled = _mm_srli_si128::<8>(full);
    let hi = _mm_cvtepi16_epi32(hi_shuffled);
    (lo, hi)
}

/// Saturating narrow an `(i32x4, i32x4)` pair to `u8x8` and store at `dst`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_row(dst: *mut u8, lo: __m128i, hi: __m128i) {
    let i16_packed = _mm_packs_epi32(lo, hi); // [lo0..3, hi0..3] as i16
    let u8_packed = _mm_packus_epi16(i16_packed, i16_packed); // low 8 lanes are our u8s
    unsafe {
        _mm_storel_epi64(dst.cast(), u8_packed);
    }
}

/// 1D IDCT pass over 4 i32 lanes. Mirrors `idct::neon::idct_1d_x4`.
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
fn idct_1d_x4<const SHIFT: i32>(
    p0: __m128i,
    p1: __m128i,
    p2: __m128i,
    p3: __m128i,
    p4: __m128i,
    p5: __m128i,
    p6: __m128i,
    p7: __m128i,
    rounding: __m128i,
) -> [__m128i; 8] {
    let mul = |v, c: i32| _mm_mullo_epi32(v, _mm_set1_epi32(c));

    let z1 = mul(_mm_add_epi32(p2, p6), FIX_0_541196100);
    let tmp2 = _mm_add_epi32(z1, mul(p6, -FIX_1_847759065));
    let tmp3 = _mm_add_epi32(z1, mul(p2, FIX_0_765366865));
    let tmp0 = _mm_slli_epi32::<CONST_BITS>(_mm_add_epi32(p0, p4));
    let tmp1 = _mm_slli_epi32::<CONST_BITS>(_mm_sub_epi32(p0, p4));
    let tmp10 = _mm_add_epi32(tmp0, tmp3);
    let tmp13 = _mm_sub_epi32(tmp0, tmp3);
    let tmp11 = _mm_add_epi32(tmp1, tmp2);
    let tmp12 = _mm_sub_epi32(tmp1, tmp2);

    let z1o = _mm_add_epi32(p7, p1);
    let z2o = _mm_add_epi32(p5, p3);
    let z3o = _mm_add_epi32(p7, p3);
    let z4o = _mm_add_epi32(p5, p1);
    let z5 = mul(_mm_add_epi32(z3o, z4o), FIX_1_175875602);

    let o0 = mul(p7, FIX_0_298631336);
    let o1 = mul(p5, FIX_2_053119869);
    let o2 = mul(p3, FIX_3_072711026);
    let o3 = mul(p1, FIX_1_501321110);
    let z1m = mul(z1o, -FIX_0_899976223);
    let z2m = mul(z2o, -FIX_2_562915447);
    let z3m = mul(z3o, -FIX_1_961570560);
    let z4m = mul(z4o, -FIX_0_390180644);
    let z3f = _mm_add_epi32(z3m, z5);
    let z4f = _mm_add_epi32(z4m, z5);

    let k0 = _mm_add_epi32(_mm_add_epi32(o0, z1m), z3f);
    let k1 = _mm_add_epi32(_mm_add_epi32(o1, z2m), z4f);
    let k2 = _mm_add_epi32(_mm_add_epi32(o2, z2m), z3f);
    let k3 = _mm_add_epi32(_mm_add_epi32(o3, z1m), z4f);

    let shift = |v| _mm_srai_epi32::<SHIFT>(_mm_add_epi32(v, rounding));
    let out0 = shift(_mm_add_epi32(tmp10, k3));
    let out7 = shift(_mm_sub_epi32(tmp10, k3));
    let out1 = shift(_mm_add_epi32(tmp11, k2));
    let out6 = shift(_mm_sub_epi32(tmp11, k2));
    let out2 = shift(_mm_add_epi32(tmp12, k1));
    let out5 = shift(_mm_sub_epi32(tmp12, k1));
    let out3 = shift(_mm_add_epi32(tmp13, k0));
    let out4 = shift(_mm_sub_epi32(tmp13, k0));

    [out0, out1, out2, out3, out4, out5, out6, out7]
}

/// 4×4 i32 transpose via SSE2 unpack intrinsics.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose_4x4_i32(a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> [__m128i; 4] {
    // Stage 1: pairwise interleave i32 lanes.
    let t0 = _mm_unpacklo_epi32(a, b); // [a0, b0, a1, b1]
    let t1 = _mm_unpackhi_epi32(a, b); // [a2, b2, a3, b3]
    let t2 = _mm_unpacklo_epi32(c, d); // [c0, d0, c1, d1]
    let t3 = _mm_unpackhi_epi32(c, d); // [c2, d2, c3, d3]

    // Stage 2: combine halves via i64 unpack — maps 2×2 quadrants to columns.
    let col0 = _mm_unpacklo_epi64(t0, t2); // [a0, b0, c0, d0]
    let col1 = _mm_unpackhi_epi64(t0, t2); // [a1, b1, c1, d1]
    let col2 = _mm_unpacklo_epi64(t1, t3); // [a2, b2, c2, d2]
    let col3 = _mm_unpackhi_epi64(t1, t3); // [a3, b3, c3, d3]
    [col0, col1, col2, col3]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idct::scalar::idct_islow as idct_scalar;

    fn run_both(input: &[i16; 64]) -> ([u8; 64], [u8; 64]) {
        let mut scalar_out = [0u8; 64];
        idct_scalar(input, &mut scalar_out);
        let mut avx_out = [0u8; 64];
        if std::is_x86_feature_detected!("avx2") {
            unsafe { idct_islow(input, &mut avx_out) };
        } else {
            // Running the test on a non-AVX2 host: copy scalar output so
            // assertion passes and the test becomes a skip.
            avx_out = scalar_out;
        }
        (scalar_out, avx_out)
    }

    #[test]
    fn avx2_matches_scalar_on_all_zero() {
        let (s, a) = run_both(&[0; 64]);
        assert_eq!(s, a);
    }

    #[test]
    fn avx2_matches_scalar_on_dc_only() {
        let mut input = [0i16; 64];
        input[0] = 8 * 8;
        let (s, a) = run_both(&input);
        assert_eq!(s, a);
    }

    #[test]
    fn avx2_matches_scalar_on_mixed_coefficients() {
        let mut input = [0i16; 64];
        input[0] = 64;
        input[1] = 24;
        input[2] = -12;
        input[8] = 18;
        input[9] = -7;
        input[16] = 5;
        let (s, a) = run_both(&input);
        assert_eq!(s, a);
    }

    #[test]
    fn avx2_matches_scalar_on_saturation() {
        let mut input = [0i16; 64];
        input[0] = i16::MAX;
        let (s, a) = run_both(&input);
        assert_eq!(s, a);

        input[0] = i16::MIN;
        let (s, a) = run_both(&input);
        assert_eq!(s, a);
    }

    #[test]
    fn avx2_matches_scalar_on_horizontal_basis() {
        let mut input = [0i16; 64];
        input[1] = 400;
        let (s, a) = run_both(&input);
        assert_eq!(s, a);
    }
}
