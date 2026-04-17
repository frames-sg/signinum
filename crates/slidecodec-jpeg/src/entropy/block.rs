// SPDX-License-Identifier: Apache-2.0

//! Per-block entropy decode: one 8×8 DCT coefficient block.
//!
//! Steps per T.81 §F.2.1:
//! 1. Decode DC category (`T`) via the DC Huffman table; read `T` bits to get
//!    the DC difference; add to `prev_dc` to recover absolute DC.
//! 2. Loop up to 63 AC coefficients: decode a byte `rs` via the AC Huffman
//!    table; `rrrr = rs >> 4` is a run of zeros; `ssss = rs & 0x0F` is the
//!    next value's category. `rs == 0x00` means EOB (all remaining AC = 0);
//!    `rs == 0xF0` means ZRL (16 zeros, continue).
//! 3. Dequantize by multiplying each surviving coefficient with its quant
//!    table entry; write to the output block in zigzag-inverted position.
//!
//! Produces a 64-entry array in row-major (natural) order, suitable for
//! direct consumption by the IDCT.

#![allow(dead_code)]

use crate::entropy::huffman::HuffmanTable;
use crate::entropy::ZIGZAG;
use crate::error::{HuffmanFailure, JpegError};
use crate::internal::bit_reader::BitReader;

/// Decode one 8×8 DCT block from the scan.
///
/// - `prev_dc` is read and updated in place so the caller threads DC prediction
///   across blocks of the same component.
/// - `quant` is the 64-entry quant table (natural / zigzag-natural order matches
///   how DQT stored it: linear). Multiplication is a straight elementwise scale.
/// - `out` is cleared and filled with the dequantized coefficients in row-major
///   order (natural 8×8 layout), ready for the IDCT.
pub(crate) fn decode_block(
    br: &mut BitReader<'_>,
    dc_table: &HuffmanTable,
    ac_table: &HuffmanTable,
    prev_dc: &mut i32,
    quant: &[u16; 64],
    out: &mut [i16; 64],
) -> Result<(), JpegError> {
    out.fill(0);

    // DC.
    let ssss = dc_table.decode(br)?;
    if ssss > 15 {
        return Err(JpegError::HuffmanDecode {
            mcu: 0,
            reason: HuffmanFailure::InvalidSymbol,
        });
    }
    let diff = br.receive_extend(ssss)?;
    *prev_dc = prev_dc.wrapping_add(diff);
    // Dequant the DC in natural-order position 0 (zigzag index 0 → natural 0).
    let dc_dequant = (*prev_dc).wrapping_mul(quant[0] as i32);
    out[0] = clamp_i16(dc_dequant);

    // AC.
    let mut k: usize = 1;
    while k < 64 {
        let rs = ac_table.decode(br)?;
        let rrrr = (rs >> 4) as usize;
        let ssss = rs & 0x0F;
        if ssss == 0 {
            if rrrr == 15 {
                // ZRL — 16 zeros, continue.
                k += 16;
                continue;
            }
            // EOB — all remaining AC coefficients are zero.
            break;
        }
        k += rrrr;
        if k >= 64 {
            return Err(JpegError::HuffmanDecode {
                mcu: 0,
                reason: HuffmanFailure::InvalidSymbol,
            });
        }
        let value = br.receive_extend(ssss)?;
        let natural_idx = ZIGZAG[k] as usize;
        // Quant table entries are stored in zigzag order per T.81 §B.2.4.1,
        // so `quant[k]` is the matching coefficient (not `quant[natural_idx]`).
        let dequant = value.wrapping_mul(quant[k] as i32);
        out[natural_idx] = clamp_i16(dequant);
        k += 1;
    }
    Ok(())
}

fn clamp_i16(v: i32) -> i16 {
    v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::tables::RawHuffmanTable;
    use alloc::vec;

    /// DC table that decodes bit `0` → symbol `0` (DC category 0 = no diff).
    /// Single code of length 1 → symbol 0.
    fn trivial_dc_table() -> HuffmanTable {
        let raw = RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: vec![0],
        };
        HuffmanTable::from_raw(&raw).unwrap()
    }

    /// AC table that decodes bit `0` → symbol `0x00` (EOB).
    fn eob_ac_table() -> HuffmanTable {
        let raw = RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: vec![0x00],
        };
        HuffmanTable::from_raw(&raw).unwrap()
    }

    #[test]
    fn decodes_all_zero_block() {
        // DC code `0` (→ category 0, no diff bits) then AC code `0` (EOB).
        // Pad with zeros so the Huffman decoder's 8-bit peek never runs dry.
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let dc = trivial_dc_table();
        let ac = eob_ac_table();
        let quant = [1u16; 64];
        let mut prev_dc = 0i32;
        let mut out = [0i16; 64];
        decode_block(&mut br, &dc, &ac, &mut prev_dc, &quant, &mut out).unwrap();
        assert_eq!(prev_dc, 0);
        assert!(out.iter().all(|&c| c == 0));
    }

    #[test]
    fn dequantizes_dc_coefficient() {
        let raw = RawHuffmanTable {
            bits: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: vec![2],
        };
        let dc = HuffmanTable::from_raw(&raw).unwrap();
        let ac = eob_ac_table();
        // Bits: 00 (DC code → ssss=2) 11 (extend → diff=3) 0 (EOB).
        // Trailing zero bytes satisfy the decoder's 8-bit peek requirement.
        let bytes = [0b0011_0000u8, 0, 0, 0];
        let mut br = BitReader::new(&bytes);
        let quant = [7u16; 64];
        let mut prev_dc = 0i32;
        let mut out = [0i16; 64];
        decode_block(&mut br, &dc, &ac, &mut prev_dc, &quant, &mut out).unwrap();
        assert_eq!(prev_dc, 3);
        assert_eq!(out[0], 21, "DC = 3 * quant 7 = 21");
        assert!(out[1..].iter().all(|&c| c == 0));
    }

    #[test]
    fn dc_prediction_accumulates_across_blocks() {
        let raw = RawHuffmanTable {
            bits: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: vec![2],
        };
        let dc = HuffmanTable::from_raw(&raw).unwrap();
        let ac = eob_ac_table();
        // Block 1: 00 11 0 (diff=+3). Block 2: 00 11 0 (diff=+3). Pad for peek.
        let bytes = [0b0011_0001u8, 0b1000_0000u8, 0, 0];
        let mut br = BitReader::new(&bytes);
        let quant = [1u16; 64];
        let mut prev_dc = 10i32;
        let mut out = [0i16; 64];
        decode_block(&mut br, &dc, &ac, &mut prev_dc, &quant, &mut out).unwrap();
        assert_eq!(prev_dc, 13);
        decode_block(&mut br, &dc, &ac, &mut prev_dc, &quant, &mut out).unwrap();
        assert_eq!(prev_dc, 16);
    }
}
