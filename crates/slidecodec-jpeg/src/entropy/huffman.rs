// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::inline_always)]

//! Huffman decoder. Two layers:
//!
//! 1. **Fast lookup** — a 4096-entry table indexed by the next 12 bits of the
//!    stream. Each entry carries `(symbol, bit_length)` or `(_, 0)` if the
//!    code is longer than 12 bits.
//! 2. **Slow path** — per-length arrays (`min_code`, `max_code`, `val_offset`)
//!    implementing the T.81 §F.2.2.3 decode procedure for codes up to 16 bits.
//!
//! Built once from [`crate::parse::tables::RawHuffmanTable`]; read many times
//! by [`crate::entropy::block::decode_block`].

use crate::error::{HuffmanFailure, JpegError};
use crate::internal::bit_reader::BitReader;
use crate::parse::tables::{HuffmanValues, RawHuffmanTable};
use alloc::boxed::Box;

/// Number of fast-lookup entries. One per possible 12-bit peek value.
const FAST_BITS: u8 = 12;
const FAST_ENTRIES: usize = 1 << FAST_BITS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AcDecoded {
    Eob,
    Zrl,
    Value { run: usize, value: i32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AcSkipDecoded {
    Eob,
    Zrl,
    Value { run: usize },
}

const AC_FAST_KIND_SHIFT: u32 = 28;
const AC_FAST_KIND_MASK: u32 = 0xF << AC_FAST_KIND_SHIFT;
const AC_FAST_VALUE: u32 = 1 << AC_FAST_KIND_SHIFT;
const AC_FAST_EOB: u32 = 2 << AC_FAST_KIND_SHIFT;
const AC_FAST_ZRL: u32 = 3 << AC_FAST_KIND_SHIFT;
const AC_FAST_LEN_MASK: u32 = 0x0F;
const AC_FAST_RUN_MASK: u32 = 0xF0;
const AC_FAST_VALUE_SHIFT: u32 = 8;

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
    values: HuffmanValues,
    fast_ac: Box<[u32; FAST_ENTRIES]>,
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
        let mut fast_ac = Box::new([0u32; FAST_ENTRIES]);

        let total_values: usize = raw.bits.iter().map(|&b| b as usize).sum();
        if total_values != raw.values.len() {
            return Err(JpegError::HuffmanDecode {
                mcu: 0,
                reason: HuffmanFailure::CodeOverflow,
            });
        }
        let mut huffsize = [0u8; 256];
        let mut huffsize_len = 0usize;
        for (len_minus_1, &count) in raw.bits.iter().enumerate() {
            let len = (len_minus_1 + 1) as u8;
            for _ in 0..count {
                huffsize[huffsize_len] = len;
                huffsize_len += 1;
            }
        }

        let mut huffcode = [0u16; 256];
        let mut code: u32 = 0;
        let mut si = huffsize.first().copied().unwrap_or(0);
        for (k, &s) in huffsize[..huffsize_len].iter().enumerate() {
            while s != si {
                code <<= 1;
                si += 1;
            }
            huffcode[k] = code as u16;
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
            min_code[len] = i32::from(huffcode[k]);
            max_code[len] = i32::from(huffcode[k + count - 1]);
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
                    fast[fast_index_base + j] = (raw.values.as_slice()[k], len);
                }
                k += 1;
            }
        }

        for (idx, &(sym, len)) in fast.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let run = usize::from((sym >> 4) & 0x0F);
            let ssss = sym & 0x0F;
            if ssss == 0 {
                fast_ac[idx] = match run {
                    0 => pack_ac_eob(len),
                    15 => pack_ac_zrl(len),
                    _ => 0,
                };
                continue;
            }
            let total_len = len + ssss;
            if total_len > FAST_BITS {
                continue;
            }

            let mag_shift = FAST_BITS - total_len;
            let mag_mask = (1u16 << ssss) - 1;
            let mag_bits = ((idx as u16) >> mag_shift) & mag_mask;
            let value = huff_extend(mag_bits as i32, ssss);
            if !(i16::MIN as i32..=i16::MAX as i32).contains(&value) {
                continue;
            }
            fast_ac[idx] = pack_ac_value(total_len, run as u8, value as i16);
        }

        Ok(Self {
            fast,
            min_code,
            max_code,
            val_offset,
            values: raw.values.clone(),
            fast_ac,
        })
    }

    /// Decode one symbol from the bit reader. Common case (code ≤ 8 bits) is
    /// a single array lookup; long codes fall through to a per-length scan.
    ///
    /// # Errors
    /// - `HuffmanDecode { TableExhausted }` if the stream ran out of bits.
    /// - `HuffmanDecode { CodeOverflow }` if no 1..=16-bit code matches.
    #[inline(always)]
    pub(crate) fn decode(&self, br: &mut BitReader<'_>) -> Result<u8, JpegError> {
        br.ensure_bits_padded(FAST_BITS)?;
        let peek = br.peek_bits(FAST_BITS) as usize;
        let (sym, len) = self.fast[peek];
        if len != 0 {
            br.consume_bits(len);
            return Ok(sym);
        }
        // Slow path: compare against `max_code[l]` for l = 13..=16.
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
                return Ok(self.values.get(idx).expect("validated huffman index"));
            }
        }
        Err(JpegError::HuffmanDecode {
            mcu: 0,
            reason: HuffmanFailure::CodeOverflow,
        })
    }

    #[inline(always)]
    pub(crate) fn decode_fast_ac(&self, br: &mut BitReader<'_>) -> Result<AcDecoded, JpegError> {
        br.ensure_bits_padded(FAST_BITS)?;
        let peek = br.peek_bits(FAST_BITS) as usize;
        let packed = self.fast_ac[peek];
        if packed != 0 {
            br.consume_bits((packed & AC_FAST_LEN_MASK) as u8);
            return Ok(match packed & AC_FAST_KIND_MASK {
                AC_FAST_VALUE => AcDecoded::Value {
                    run: ((packed & AC_FAST_RUN_MASK) >> 4) as usize,
                    value: i32::from(((packed >> AC_FAST_VALUE_SHIFT) & 0xFFFF) as u16 as i16),
                },
                AC_FAST_EOB => AcDecoded::Eob,
                AC_FAST_ZRL => AcDecoded::Zrl,
                _ => unreachable!("invalid AC fast-table tag"),
            });
        }

        let (sym, len) = self.fast[peek];
        let sym = if len != 0 {
            br.consume_bits(len);
            sym
        } else {
            self.decode(br)?
        };

        let run = (sym >> 4) as usize;
        let ssss = sym & 0x0F;
        if ssss == 0 {
            return Ok(if run == 15 {
                AcDecoded::Zrl
            } else {
                AcDecoded::Eob
            });
        }

        let value = br.receive_extend(ssss)?;
        Ok(AcDecoded::Value { run, value })
    }

    #[inline(always)]
    pub(crate) fn skip_fast_ac(&self, br: &mut BitReader<'_>) -> Result<AcSkipDecoded, JpegError> {
        br.ensure_bits_padded(FAST_BITS)?;
        let peek = br.peek_bits(FAST_BITS) as usize;
        let packed = self.fast_ac[peek];
        if packed != 0 {
            br.consume_bits((packed & AC_FAST_LEN_MASK) as u8);
            return Ok(match packed & AC_FAST_KIND_MASK {
                AC_FAST_VALUE => AcSkipDecoded::Value {
                    run: ((packed & AC_FAST_RUN_MASK) >> 4) as usize,
                },
                AC_FAST_EOB => AcSkipDecoded::Eob,
                AC_FAST_ZRL => AcSkipDecoded::Zrl,
                _ => unreachable!("invalid AC fast-table tag"),
            });
        }

        let (sym, len) = self.fast[peek];
        let sym = if len != 0 {
            br.consume_bits(len);
            sym
        } else {
            self.decode(br)?
        };

        let run = (sym >> 4) as usize;
        let ssss = sym & 0x0F;
        if ssss == 0 {
            return Ok(if run == 15 {
                AcSkipDecoded::Zrl
            } else {
                AcSkipDecoded::Eob
            });
        }

        br.ensure_bits(ssss)?;
        br.consume_bits(ssss);
        Ok(AcSkipDecoded::Value { run })
    }
}

#[inline]
fn pack_ac_value(total_len: u8, run: u8, value: i16) -> u32 {
    AC_FAST_VALUE
        | ((u32::from(value as u16)) << AC_FAST_VALUE_SHIFT)
        | (u32::from(run) << 4)
        | u32::from(total_len)
}

#[inline]
fn pack_ac_eob(total_len: u8) -> u32 {
    AC_FAST_EOB | u32::from(total_len)
}

#[inline]
fn pack_ac_zrl(total_len: u8) -> u32 {
    AC_FAST_ZRL | (15 << 4) | u32::from(total_len)
}

fn huff_extend(v: i32, ssss: u8) -> i32 {
    let threshold = 1i32 << (ssss - 1);
    if v < threshold {
        v + ((-1i32) << ssss) + 1
    } else {
        v
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
            values: HuffmanValues::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        }
    }

    #[test]
    fn builds_fast_table_from_standard_luma_dc() {
        let table = HuffmanTable::from_raw(&luma_dc_raw()).unwrap();
        let (sym, len) = table.fast[0b0000_0000_0000];
        assert_eq!((sym, len), (0, 2));
        let (sym, len) = table.fast[0b0011_1111_1111];
        assert_eq!((sym, len), (0, 2));
        let (sym, len) = table.fast[0b0100_0000_0000];
        assert_eq!((sym, len), (1, 3));
    }

    #[test]
    fn widened_fast_table_covers_9_bit_luma_dc_code() {
        let table = HuffmanTable::from_raw(&luma_dc_raw()).unwrap();
        let idx = ((0b1_1111_1110u32) << 3) as usize;
        let (sym, len) = table.fast.get(idx).copied().unwrap_or((0, 0));
        assert_eq!((sym, len), (11, 9));
    }

    #[test]
    fn rejects_oversubscribed_code_table() {
        let raw = RawHuffmanTable {
            bits: [1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0, 1, 2, 3, 4]),
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
            values: HuffmanValues::default(),
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
            (0b1_1110, 5, 7),
            (0b11_1110, 6, 8),
            (0b111_1110, 7, 9),
            (0b1111_1110, 8, 10),
            (0b1_1111_1110, 9, 11),
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
        let bytes = [];
        let mut br = BitReader::new(&bytes);
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
