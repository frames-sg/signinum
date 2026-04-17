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

use crate::error::{HuffmanFailure, JpegError};
use crate::internal::bit_reader::BitReader;
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
    ///   pre-adjusted by subtracting `min_code[l]` so
    ///   `symbol = values[code + val_offset[l]]`.
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

    /// Decode one symbol from the bit reader. Common case (code ≤ 8 bits) is
    /// a single array lookup; long codes fall through to a per-length scan.
    ///
    /// # Errors
    /// - `HuffmanDecode { TableExhausted }` if the stream ran out of bits.
    /// - `HuffmanDecode { CodeOverflow }` if no 1..=16-bit code matches.
    pub(crate) fn decode(&self, br: &mut BitReader<'_>) -> Result<u8, JpegError> {
        br.ensure_bits_padded(FAST_BITS)?;
        let peek = br.peek_bits(FAST_BITS) as usize;
        let (sym, len) = self.fast[peek];
        if len != 0 {
            br.consume_bits(len);
            return Ok(sym);
        }
        // Slow path: compare against `max_code[l]` for l = 9..=16.
        br.ensure_bits_padded(16)?;
        let code16 = br.peek_bits(16) as i32;
        for len in (FAST_BITS as usize + 1)..=16 {
            let l = len as u8;
            let c = code16 >> (16 - l);
            if c <= self.max_code[len] {
                br.consume_bits(l);
                let idx = (c + self.val_offset[len]) as usize;
                if idx >= self.values.len() {
                    return Err(JpegError::HuffmanDecode {
                        mcu: 0,
                        reason: HuffmanFailure::InvalidSymbol,
                    });
                }
                return Ok(self.values[idx]);
            }
        }
        Err(JpegError::HuffmanDecode {
            mcu: 0,
            reason: HuffmanFailure::CodeOverflow,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Exercises every standard JPEG luma DC code — Annex K.3.
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
    fn decodes_all_standard_luma_dc_codes() {
        let table = HuffmanTable::from_raw(&luma_dc_raw()).unwrap();
        for &(code, len, expected) in luma_dc_code_cases() {
            let mut bytes = alloc::vec![0u8; 4];
            let shift = 32 - len;
            let aligned = code << shift;
            bytes[0] = (aligned >> 24) as u8;
            bytes[1] = (aligned >> 16) as u8;
            bytes[2] = (aligned >> 8) as u8;
            bytes[3] = aligned as u8;
            let mut br = BitReader::new(&bytes);
            let sym = table.decode(&mut br).unwrap();
            assert_eq!(sym, expected, "code={code:b} len={len}");
        }
    }

    #[test]
    fn decodes_9_plus_bit_codes_via_slow_path() {
        let table = HuffmanTable::from_raw(&luma_dc_raw()).unwrap();
        // Code `111111110` (9 bits) → symbol 11. A literal 0xFF in a JPEG
        // entropy stream must be byte-stuffed as `FF 00` (T.81 §F.1.2.3) so
        // the BitReader does not mistake it for a marker prefix.
        let bytes = [0xFFu8, 0x00, 0b0100_0000];
        let mut br = BitReader::new(&bytes);
        let sym = table.decode(&mut br).unwrap();
        assert_eq!(sym, 11);
    }

    #[test]
    fn reports_huffman_failure_on_truncated_bit_stream() {
        let table = HuffmanTable::from_raw(&luma_dc_raw()).unwrap();
        let bytes = [0u8];
        let mut br = BitReader::new(&bytes);
        let _ = table.decode(&mut br).unwrap();
        let err = table.decode(&mut br).unwrap_err();
        assert!(matches!(
            err,
            JpegError::HuffmanDecode {
                reason: HuffmanFailure::TableExhausted,
                ..
            }
        ));
    }
}
