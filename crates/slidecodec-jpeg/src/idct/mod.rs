// SPDX-License-Identifier: Apache-2.0

//! Inverse discrete cosine transform. M1b ships only the scalar ISLOW path,
//! which is also the ground-truth parity oracle — every future SIMD variant
//! (M4) is tested for bit-exact match against `idct_islow`.

pub(crate) mod scalar;

#[allow(unused_imports)]
pub(crate) use scalar::idct_islow;
