// SPDX-License-Identifier: Apache-2.0
//
//! Proptest-driven bit-exact parity between the scalar ISLOW IDCT and every
//! SIMD backend variant. Any divergence here is a merge-blocker for Phase 1.
//!
//! The test only exercises the SIMD variant the host CPU can actually run —
//! we do not cross-compile to the other arch. CI runs on both aarch64 and
//! x86_64 so each backend is exercised at least once.

#![cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]

use proptest::prelude::*;

fn scalar(input: &[i16; 64]) -> [u8; 64] {
    let mut out = [0u8; 64];
    ashlar_jpeg::bench_support::bench_idct_reference_block_with(input, &mut out);
    out
}

#[cfg(target_arch = "aarch64")]
fn assert_neon_matches_scalar(input: &[i16; 64]) {
    let scalar_out = scalar(input);
    let mut neon_out = [0u8; 64];
    ashlar_jpeg::bench_support::bench_idct_neon_block(input, &mut neon_out);
    assert_eq!(
        scalar_out, neon_out,
        "NEON IDCT diverged from scalar on input {input:?}"
    );
}

#[cfg(target_arch = "x86_64")]
fn assert_avx2_matches_scalar(input: &[i16; 64]) {
    if !std::is_x86_feature_detected!("avx2") {
        return; // AVX2 unavailable on this host — skip silently.
    }
    let scalar_out = scalar(input);
    let mut avx_out = [0u8; 64];
    ashlar_jpeg::bench_support::bench_idct_avx2_block(input, &mut avx_out);
    assert_eq!(
        scalar_out, avx_out,
        "AVX2 IDCT diverged from scalar on input {input:?}"
    );
}

fn vec_into_block(v: Vec<i16>) -> [i16; 64] {
    let mut arr = [0i16; 64];
    arr.copy_from_slice(&v);
    arr
}

fn small_coefficients() -> impl Strategy<Value = [i16; 64]> {
    // Typical JPEG space after dequantization: DC up to ~1024, AC small.
    prop::collection::vec(-512i16..512, 64..=64).prop_map(vec_into_block)
}

fn large_coefficients() -> impl Strategy<Value = [i16; 64]> {
    // Full i16 range — exercises wrapping-then-clamp on saturating inputs.
    prop::collection::vec(any::<i16>(), 64..=64).prop_map(vec_into_block)
}

fn sparse_blocks() -> impl Strategy<Value = [i16; 64]> {
    any::<i16>().prop_map(|dc| {
        let mut block = [0i16; 64];
        block[0] = dc;
        block
    })
}

fn bottom_half_zero_blocks() -> impl Strategy<Value = [i16; 64]> {
    prop::collection::vec(-512i16..512, 32..=32).prop_map(|top_half| {
        let mut block = [0i16; 64];
        block[..32].copy_from_slice(&top_half);
        block
    })
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 2_000,
        .. ProptestConfig::default()
    })]

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_on_small_coefficients(block in small_coefficients()) {
        assert_neon_matches_scalar(&block);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_on_large_coefficients(block in large_coefficients()) {
        assert_neon_matches_scalar(&block);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_on_sparse_blocks(block in sparse_blocks()) {
        assert_neon_matches_scalar(&block);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_on_bottom_half_zero_blocks(block in bottom_half_zero_blocks()) {
        assert_neon_matches_scalar(&block);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_small_coefficients(block in small_coefficients()) {
        assert_avx2_matches_scalar(&block);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_large_coefficients(block in large_coefficients()) {
        assert_avx2_matches_scalar(&block);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_sparse_blocks(block in sparse_blocks()) {
        assert_avx2_matches_scalar(&block);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_bottom_half_zero_blocks(block in bottom_half_zero_blocks()) {
        assert_avx2_matches_scalar(&block);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_matches_scalar_on_hand_picked_edges() {
    assert_neon_matches_scalar(&[0; 64]);

    let mut block = [0i16; 64];
    block[0] = i16::MAX;
    assert_neon_matches_scalar(&block);
    block[0] = i16::MIN;
    assert_neon_matches_scalar(&block);

    assert_neon_matches_scalar(&[i16::MAX; 64]);
    assert_neon_matches_scalar(&[i16::MIN; 64]);
    assert_neon_matches_scalar(&[1; 64]);
    assert_neon_matches_scalar(&[-1; 64]);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_matches_scalar_on_hand_picked_edges() {
    assert_avx2_matches_scalar(&[0; 64]);

    let mut block = [0i16; 64];
    block[0] = i16::MAX;
    assert_avx2_matches_scalar(&block);
    block[0] = i16::MIN;
    assert_avx2_matches_scalar(&block);

    assert_avx2_matches_scalar(&[i16::MAX; 64]);
    assert_avx2_matches_scalar(&[i16::MIN; 64]);
    assert_avx2_matches_scalar(&[1; 64]);
    assert_avx2_matches_scalar(&[-1; 64]);
}
