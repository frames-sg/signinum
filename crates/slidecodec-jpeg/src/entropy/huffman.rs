// SPDX-License-Identifier: Apache-2.0

//! Huffman decoder. Two layers:
//!
//! 1. **Fast lookup** — a 256-entry table indexed by the next 8 bits of the
//!    stream. Each entry carries `(symbol, bit_length)` or `(_, 0)` if the
//!    code is longer than 8 bits.
//! 2. **Slow path** — per-length arrays (`min_code`, `max_code`, `val_offset`)
//!    implementing the T.81 §F.2.2.3 decode procedure for codes up to 16 bits.
//!
//! Built once from [`crate::parse::tables::RawHuffmanTable`]; read many times
//! by [`crate::entropy::block::decode_block`].

#![allow(dead_code)]

use crate::error::{HuffmanFailure, JpegError};
use crate::parse::tables::RawHuffmanTable;
use alloc::vec::Vec;

/// Number of fast-lookup entries. One per possible 8-bit peek value.
const FAST_BITS: u8 = 8;
const FAST_ENTRIES: usize = 1 << FAST_BITS;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HuffmanTable {
    /// Fast path: `fast[peek8] = (symbol, bit_length)`. `bit_length == 0`
    /// means "code longer than 8 bits — use slow path".
    fast: [(u8, u8); FAST_ENTRIES],
    /// Slow path, indexed by code length `l` ∈ `1..=16`:
    /// - `min_code[l]`: smallest `l`-bit code; `i32::MAX` if no `l`-bit code.
    /// - `max_code[l]`: largest `l`-bit code; `-1` if no `l`-bit code.
    /// - `val_offset[l]`: index into `values` where `l`-bit symbols begin,
    ///   adjusted so `symbol = values[code - min_code[l] + val_offset[l]]`.
    min_code: [i32; 17],
    max_code: [i32; 17],
    val_offset: [i32; 17],
    values: Vec<u8>,
}

impl HuffmanTable {
    /// Build the decode table from a raw `(bits, values)` pair parsed out of
    /// a DHT segment. Per T.81 §C.2 and Annex C.
    ///
    /// # Errors
    /// - `HuffmanDecode { CodeOverflow }` if `bits` is oversubscribed (Kraft
    ///   inequality violated — the table claims more codes of some length than
    ///   there is remaining code space).
    pub(crate) fn from_raw(raw: &RawHuffmanTable) -> Result<Self, JpegError> {
        let mut fast = [(0u8, 0u8); FAST_ENTRIES];
        let mut min_code = [i32::MAX; 17];
        let mut max_code = [-1i32; 17];
        let mut val_offset = [0i32; 17];

        let total_values: usize = raw.bits.iter().map(|&b| b as usize).sum();
        if total_values != raw.values.len() {
            return Err(JpegError::HuffmanDecode {
                mcu: 0,
                reason: HuffmanFailure::CodeOverflow,
            });
        }
        let mut huffsize = Vec::with_capacity(total_values);
        for (len_minus_1, &count) in raw.bits.iter().enumerate() {
            let len = (len_minus_1 + 1) as u8;
            for _ in 0..count {
                huffsize.push(len);
            }
        }

        let mut huffcode = Vec::with_capacity(total_values);
        let mut code: u32 = 0;
        let mut si = huffsize.first().copied().unwrap_or(0);
        for &s in &huffsize {
            while s != si {
                code <<= 1;
                si += 1;
            }
            huffcode.push(code);
            code = code.checked_add(1).ok_or(JpegError::HuffmanDecode {
                mcu: 0,
                reason: HuffmanFailure::CodeOverflow,
            })?;
        }
        if si > 0 && (code - 1) >= (1u32 << si) {
            return Err(JpegError::HuffmanDecode {
                mcu: 0,
                reason: HuffmanFailure::CodeOverflow,
            });
        }

        let mut k = 0usize;
        for len_minus_1 in 0..16 {
            let len = len_minus_1 + 1;
            let count = raw.bits[len_minus_1] as usize;
            if count == 0 {
                continue;
            }
            min_code[len] = huffcode[k] as i32;
            max_code[len] = huffcode[k + count - 1] as i32;
            val_offset[len] = k as i32 - min_code[len];
            k += count;
        }

        k = 0;
        for len_minus_1 in 0..FAST_BITS as usize {
            let len = (len_minus_1 + 1) as u8;
            let count = raw.bits[len_minus_1] as usize;
            for _ in 0..count {
                let c = huffcode[k];
                let fast_index_base = (c as usize) << (FAST_BITS - len);
                let fast_count = 1 << (FAST_BITS - len);
                for j in 0..fast_count {
                    fast[fast_index_base + j] = (raw.values[k], len);
                }
                k += 1;
            }
        }

        Ok(Self {
            fast,
            min_code,
            max_code,
            val_offset,
            values: raw.values.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::bit_reader::BitReader;

    /// Standard JPEG luminance DC table from Annex K.3 — well-known fixture.
    /// `bits[0..16]` counts per length; `values` lists the symbols in order.
    fn luma_dc_raw() -> RawHuffmanTable {
        RawHuffmanTable {
            bits: [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            values: alloc::vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        }
    }

    #[test]
    fn builds_fast_table_from_standard_luma_dc() {
        let table = HuffmanTable::from_raw(&luma_dc_raw()).unwrap();
        let (sym, len) = table.fast[0b0000_0000];
        assert_eq!((sym, len), (0, 2));
        let (sym, len) = table.fast[0b0011_1111];
        assert_eq!((sym, len), (0, 2));
        let (sym, len) = table.fast[0x40];
        assert_eq!((sym, len), (1, 3));
    }

    #[test]
    fn rejects_oversubscribed_code_table() {
        let raw = RawHuffmanTable {
            bits: [1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: alloc::vec![0, 1, 2, 3, 4],
        };
        let err = HuffmanTable::from_raw(&raw).unwrap_err();
        assert!(matches!(
            err,
            JpegError::HuffmanDecode {
                reason: HuffmanFailure::CodeOverflow,
                ..
            }
        ));
    }

    #[test]
    fn handles_empty_table_without_panic() {
        let raw = RawHuffmanTable {
            bits: [0; 16],
            values: alloc::vec![],
        };
        let table = HuffmanTable::from_raw(&raw).unwrap();
        assert!(table.fast.iter().all(|&(_, len)| len == 0));
    }

    // Fixture for Task 6 (`decodes_all_standard_luma_dc_codes`) kept as reference
    // data. It exercises every standard JPEG luma DC code — Annex K.3.
    // The actual test lands once `HuffmanTable::decode` exists.
    #[allow(dead_code)]
    fn luma_dc_code_cases() -> &'static [(u32, u8, u8)] {
        &[
            (0b00, 2, 0),
            (0b010, 3, 1),
            (0b011, 3, 2),
            (0b100, 3, 3),
            (0b101, 3, 4),
            (0b110, 3, 5),
            (0b1110, 4, 6),
            (0b1111_0, 5, 7),
            (0b1111_10, 6, 8),
            (0b1111_110, 7, 9),
            (0b1111_1110, 8, 10),
            (0b1111_1111_0, 9, 11),
        ]
    }

    #[test]
    #[ignore = "requires HuffmanTable::decode — lands in Task 6"]
    fn decodes_all_standard_luma_dc_codes() {
        // Placeholder: full body lands in Task 6 once `HuffmanTable::decode`
        // exists. Keep `BitReader` referenced so the import stays pulled in.
        let _ = BitReader::new(&[]);
        let _ = luma_dc_code_cases();
        let _ = HuffmanTable::from_raw(&luma_dc_raw()).unwrap();
    }
}
