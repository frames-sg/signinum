// SPDX-License-Identifier: Apache-2.0

//! Entropy decoding — Huffman tables and the per-MCU block decoder.

pub(crate) mod block;
pub(crate) mod huffman;
pub(crate) mod progressive;
pub(crate) mod sequential;

/// T.81 §A.3.6 zigzag order: the 8×8 coefficient scan order from DC to
/// highest-frequency AC. Coefficient `k` in the stream lands at linear
/// position `ZIGZAG[k]` in the 8×8 block (row-major).
#[rustfmt::skip]
pub(crate) const ZIGZAG: [u8; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];
