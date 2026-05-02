// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::inline_always)]

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

use crate::entropy::huffman::{AcDecoded, AcSkipDecoded, HuffmanTable};
use crate::entropy::ZIGZAG;
use crate::error::{HuffmanFailure, JpegError};
use crate::internal::bit_reader::BitReader;

const DENSE_CLEAR_THRESHOLD: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockActivity {
    DcOnly,
    BottomHalfZero,
    General,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReducedIdctCoefficients {
    Half,
    Quarter,
}

impl ReducedIdctCoefficients {
    #[inline(always)]
    fn keeps(self, natural_idx: usize) -> bool {
        let row = natural_idx >> 3;
        let col = natural_idx & 7;
        match self {
            Self::Half => row != 4 && col != 4,
            Self::Quarter => !matches!(row, 2 | 4 | 6) && !matches!(col, 2 | 4 | 6),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClearMode {
    Sparse,
    Dense,
}

#[derive(Debug, Clone)]
pub(crate) struct CoefficientBlock {
    coeffs: [i16; 64],
    touched: [u8; 64],
    touched_len: usize,
    clear_mode: ClearMode,
}

impl Default for CoefficientBlock {
    fn default() -> Self {
        Self {
            coeffs: [0; 64],
            touched: [0; 64],
            touched_len: 0,
            clear_mode: ClearMode::Sparse,
        }
    }
}

impl CoefficientBlock {
    #[inline(always)]
    fn clear_touched(&mut self) {
        match self.clear_mode {
            ClearMode::Sparse => {
                for &idx in &self.touched[..self.touched_len] {
                    self.coeffs[idx as usize] = 0;
                }
            }
            ClearMode::Dense => self.coeffs.fill(0),
        }
        self.touched_len = 0;
        self.clear_mode = ClearMode::Sparse;
    }

    #[inline(always)]
    fn store(&mut self, idx: usize, value: i16) {
        self.coeffs[idx] = value;
        if self.clear_mode == ClearMode::Sparse {
            if self.touched_len < DENSE_CLEAR_THRESHOLD {
                self.touched[self.touched_len] = idx as u8;
                self.touched_len += 1;
            } else {
                self.clear_mode = ClearMode::Dense;
            }
        }
    }

    #[inline(always)]
    pub(crate) fn coefficients(&self) -> &[i16; 64] {
        &self.coeffs
    }

    #[inline(always)]
    pub(crate) fn dc_coeff(&self) -> i16 {
        self.coeffs[0]
    }
}

#[inline(always)]
fn extend_activity(activity: BlockActivity, natural_idx: usize) -> BlockActivity {
    if natural_idx < 32 {
        match activity {
            BlockActivity::DcOnly | BlockActivity::BottomHalfZero => BlockActivity::BottomHalfZero,
            BlockActivity::General => BlockActivity::General,
        }
    } else {
        BlockActivity::General
    }
}

/// Decode one 8×8 DCT block from the scan.
///
/// - `prev_dc` is read and updated in place so the caller threads DC prediction
///   across blocks of the same component.
/// - `quant` is the 64-entry quant table (natural / zigzag-natural order matches
///   how DQT stored it: linear). Multiplication is a straight elementwise scale.
/// - `out` is cleared and filled with the dequantized coefficients in row-major
///   order (natural 8×8 layout), ready for the IDCT.
#[cfg(test)]
pub(crate) fn decode_block(
    br: &mut BitReader<'_>,
    dc_table: &HuffmanTable,
    ac_table: &HuffmanTable,
    prev_dc: &mut i32,
    quant: &[u16; 64],
    block: &mut CoefficientBlock,
) -> Result<(), JpegError> {
    decode_block_with_activity(br, dc_table, ac_table, prev_dc, quant, block).map(|_| ())
}

#[inline(always)]
pub(crate) fn decode_block_with_activity(
    br: &mut BitReader<'_>,
    dc_table: &HuffmanTable,
    ac_table: &HuffmanTable,
    prev_dc: &mut i32,
    quant: &[u16; 64],
    block: &mut CoefficientBlock,
) -> Result<BlockActivity, JpegError> {
    block.clear_touched();

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
    block.store(0, clamp_i16(dc_dequant));

    // AC.
    let mut k: usize = 1;
    let mut activity = BlockActivity::DcOnly;
    while k < 64 {
        match ac_table.decode_fast_ac(br)? {
            AcDecoded::Eob => break,
            AcDecoded::Zrl => {
                // ZRL — 16 zeros, continue.
                k += 16;
            }
            AcDecoded::Value { run, value } => {
                k += run;
                if k >= 64 {
                    return Err(JpegError::HuffmanDecode {
                        mcu: 0,
                        reason: HuffmanFailure::InvalidSymbol,
                    });
                }
                let natural_idx = ZIGZAG[k] as usize;
                // Quant table entries are stored in zigzag order per T.81 §B.2.4.1,
                // so `quant[k]` is the matching coefficient (not `quant[natural_idx]`).
                let dequant = value.wrapping_mul(quant[k] as i32);
                block.store(natural_idx, clamp_i16(dequant));
                activity = extend_activity(activity, natural_idx);
                k += 1;
            }
        }
    }
    Ok(activity)
}

#[inline(always)]
#[cfg(test)]
pub(crate) fn decode_block_with_dc_status(
    br: &mut BitReader<'_>,
    dc_table: &HuffmanTable,
    ac_table: &HuffmanTable,
    prev_dc: &mut i32,
    quant: &[u16; 64],
    block: &mut CoefficientBlock,
) -> Result<bool, JpegError> {
    block.clear_touched();

    let ssss = dc_table.decode(br)?;
    if ssss > 15 {
        return Err(JpegError::HuffmanDecode {
            mcu: 0,
            reason: HuffmanFailure::InvalidSymbol,
        });
    }
    let diff = br.receive_extend(ssss)?;
    *prev_dc = prev_dc.wrapping_add(diff);
    let dc_dequant = (*prev_dc).wrapping_mul(quant[0] as i32);
    block.store(0, clamp_i16(dc_dequant));

    let mut k: usize = 1;
    let mut dc_only = true;
    while k < 64 {
        match ac_table.decode_fast_ac(br)? {
            AcDecoded::Eob => break,
            AcDecoded::Zrl => {
                k += 16;
            }
            AcDecoded::Value { run, value } => {
                k += run;
                if k >= 64 {
                    return Err(JpegError::HuffmanDecode {
                        mcu: 0,
                        reason: HuffmanFailure::InvalidSymbol,
                    });
                }
                let natural_idx = ZIGZAG[k] as usize;
                let dequant = value.wrapping_mul(quant[k] as i32);
                block.store(natural_idx, clamp_i16(dequant));
                dc_only = false;
                k += 1;
            }
        }
    }
    Ok(dc_only)
}

#[inline(always)]
pub(crate) fn decode_block_for_reduced_idct(
    br: &mut BitReader<'_>,
    dc_table: &HuffmanTable,
    ac_table: &HuffmanTable,
    prev_dc: &mut i32,
    quant: &[u16; 64],
    block: &mut CoefficientBlock,
    keep: ReducedIdctCoefficients,
) -> Result<bool, JpegError> {
    block.clear_touched();

    let ssss = dc_table.decode(br)?;
    if ssss > 15 {
        return Err(JpegError::HuffmanDecode {
            mcu: 0,
            reason: HuffmanFailure::InvalidSymbol,
        });
    }
    let diff = br.receive_extend(ssss)?;
    *prev_dc = prev_dc.wrapping_add(diff);
    let dc_dequant = (*prev_dc).wrapping_mul(quant[0] as i32);
    block.store(0, clamp_i16(dc_dequant));

    let mut k: usize = 1;
    let mut dc_only_for_reduced_idct = true;
    while k < 64 {
        match ac_table.decode_fast_ac(br)? {
            AcDecoded::Eob => break,
            AcDecoded::Zrl => {
                k += 16;
            }
            AcDecoded::Value { run, value } => {
                k += run;
                if k >= 64 {
                    return Err(JpegError::HuffmanDecode {
                        mcu: 0,
                        reason: HuffmanFailure::InvalidSymbol,
                    });
                }
                let natural_idx = ZIGZAG[k] as usize;
                if keep.keeps(natural_idx) {
                    let dequant = value.wrapping_mul(quant[k] as i32);
                    block.store(natural_idx, clamp_i16(dequant));
                    dc_only_for_reduced_idct = false;
                }
                k += 1;
            }
        }
    }
    Ok(dc_only_for_reduced_idct)
}

#[inline(always)]
pub(crate) fn decode_block_for_1x1_idct(
    br: &mut BitReader<'_>,
    dc_table: &HuffmanTable,
    ac_table: &HuffmanTable,
    prev_dc: &mut i32,
    quant: &[u16; 64],
    block: &mut CoefficientBlock,
) -> Result<(), JpegError> {
    block.clear_touched();

    let ssss = dc_table.decode(br)?;
    if ssss > 15 {
        return Err(JpegError::HuffmanDecode {
            mcu: 0,
            reason: HuffmanFailure::InvalidSymbol,
        });
    }
    let diff = br.receive_extend(ssss)?;
    *prev_dc = prev_dc.wrapping_add(diff);
    let dc_dequant = (*prev_dc).wrapping_mul(quant[0] as i32);
    block.store(0, clamp_i16(dc_dequant));

    let mut k: usize = 1;
    while k < 64 {
        match ac_table.skip_fast_ac(br)? {
            AcSkipDecoded::Eob => break,
            AcSkipDecoded::Zrl => {
                k += 16;
            }
            AcSkipDecoded::Value { run } => {
                k += run;
                if k >= 64 {
                    return Err(JpegError::HuffmanDecode {
                        mcu: 0,
                        reason: HuffmanFailure::InvalidSymbol,
                    });
                }
                k += 1;
            }
        }
    }
    Ok(())
}

#[inline(always)]
pub(crate) fn skip_block(
    br: &mut BitReader<'_>,
    dc_table: &HuffmanTable,
    ac_table: &HuffmanTable,
    prev_dc: &mut i32,
) -> Result<(), JpegError> {
    let ssss = dc_table.decode(br)?;
    if ssss > 15 {
        return Err(JpegError::HuffmanDecode {
            mcu: 0,
            reason: HuffmanFailure::InvalidSymbol,
        });
    }
    let diff = br.receive_extend(ssss)?;
    *prev_dc = prev_dc.wrapping_add(diff);

    let mut k: usize = 1;
    while k < 64 {
        match ac_table.skip_fast_ac(br)? {
            AcSkipDecoded::Eob => break,
            AcSkipDecoded::Zrl => {
                k += 16;
            }
            AcSkipDecoded::Value { run } => {
                k += run;
                if k >= 64 {
                    return Err(JpegError::HuffmanDecode {
                        mcu: 0,
                        reason: HuffmanFailure::InvalidSymbol,
                    });
                }
                k += 1;
            }
        }
    }
    Ok(())
}

fn clamp_i16(v: i32) -> i16 {
    v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::tables::{HuffmanValues, RawHuffmanTable};

    /// DC table that decodes bit `0` → symbol `0` (DC category 0 = no diff).
    /// Single code of length 1 → symbol 0.
    fn trivial_dc_table() -> HuffmanTable {
        let raw = RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0]),
        };
        HuffmanTable::from_raw(&raw).unwrap()
    }

    /// AC table that decodes bit `0` → symbol `0x00` (EOB).
    fn eob_ac_table() -> HuffmanTable {
        let raw = RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0x00]),
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
        let mut out = CoefficientBlock::default();
        let activity =
            decode_block_with_activity(&mut br, &dc, &ac, &mut prev_dc, &quant, &mut out).unwrap();
        assert_eq!(prev_dc, 0);
        assert_eq!(activity, BlockActivity::DcOnly);
        assert!(out.coefficients().iter().all(|&c| c == 0));
    }

    #[test]
    fn dequantizes_dc_coefficient() {
        let raw = RawHuffmanTable {
            bits: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[2]),
        };
        let dc = HuffmanTable::from_raw(&raw).unwrap();
        let ac = eob_ac_table();
        // Bits: 00 (DC code → ssss=2) 11 (extend → diff=3) 0 (EOB).
        // Trailing zero bytes satisfy the decoder's 8-bit peek requirement.
        let bytes = [0b0011_0000u8, 0, 0, 0];
        let mut br = BitReader::new(&bytes);
        let quant = [7u16; 64];
        let mut prev_dc = 0i32;
        let mut out = CoefficientBlock::default();
        let activity =
            decode_block_with_activity(&mut br, &dc, &ac, &mut prev_dc, &quant, &mut out).unwrap();
        assert_eq!(prev_dc, 3);
        assert_eq!(activity, BlockActivity::DcOnly);
        assert_eq!(out.coefficients()[0], 21, "DC = 3 * quant 7 = 21");
        assert!(out.coefficients()[1..].iter().all(|&c| c == 0));
    }

    #[test]
    fn dc_prediction_accumulates_across_blocks() {
        let raw = RawHuffmanTable {
            bits: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[2]),
        };
        let dc = HuffmanTable::from_raw(&raw).unwrap();
        let ac = eob_ac_table();
        // Block 1: 00 11 0 (diff=+3). Block 2: 00 11 0 (diff=+3). Pad for peek.
        let bytes = [0b0011_0001u8, 0b1000_0000u8, 0, 0];
        let mut br = BitReader::new(&bytes);
        let quant = [1u16; 64];
        let mut prev_dc = 10i32;
        let mut out = CoefficientBlock::default();
        decode_block(&mut br, &dc, &ac, &mut prev_dc, &quant, &mut out).unwrap();
        assert_eq!(prev_dc, 13);
        decode_block(&mut br, &dc, &ac, &mut prev_dc, &quant, &mut out).unwrap();
        assert_eq!(prev_dc, 16);
    }

    #[test]
    fn reports_general_activity_when_ac_coefficient_is_present() {
        let dc = trivial_dc_table();
        let raw = RawHuffmanTable {
            bits: [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0x01, 0x00]),
        };
        let ac = HuffmanTable::from_raw(&raw).unwrap();
        // AC symbols: `00` => 0x01 (run 0, size 1), then `010` => EOB.
        // Payload bit `1` gives AC value +1 at zigzag slot 1.
        let bytes = [0b0001_0100u8, 0, 0, 0];
        let mut br = BitReader::new(&bytes);
        let quant = [1u16; 64];
        let mut prev_dc = 0i32;
        let mut out = CoefficientBlock::default();
        let activity =
            decode_block_with_activity(&mut br, &dc, &ac, &mut prev_dc, &quant, &mut out).unwrap();
        assert_eq!(activity, BlockActivity::BottomHalfZero);
        assert_eq!(out.coefficients()[crate::entropy::ZIGZAG[1] as usize], 1);
    }

    #[test]
    fn dc_status_decoder_matches_block_coefficients_without_activity_classification() {
        let dc = trivial_dc_table();
        let raw = RawHuffmanTable {
            bits: [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0x01, 0x00]),
        };
        let ac = HuffmanTable::from_raw(&raw).unwrap();
        let bytes = [0b0001_0100u8, 0, 0, 0];
        let quant = [1u16; 64];
        let mut activity_reader = BitReader::new(&bytes);
        let mut dc_status_reader = BitReader::new(&bytes);
        let mut activity_prev_dc = 0i32;
        let mut dc_status_prev_dc = 0i32;
        let mut activity_block = CoefficientBlock::default();
        let mut dc_status_block = CoefficientBlock::default();

        let activity = decode_block_with_activity(
            &mut activity_reader,
            &dc,
            &ac,
            &mut activity_prev_dc,
            &quant,
            &mut activity_block,
        )
        .unwrap();
        let dc_only = decode_block_with_dc_status(
            &mut dc_status_reader,
            &dc,
            &ac,
            &mut dc_status_prev_dc,
            &quant,
            &mut dc_status_block,
        )
        .unwrap();

        assert_eq!(activity, BlockActivity::BottomHalfZero);
        assert!(!dc_only);
        assert_eq!(dc_status_prev_dc, activity_prev_dc);
        assert_eq!(
            dc_status_block.coefficients(),
            activity_block.coefficients()
        );
        assert_eq!(dc_status_reader.snapshot(), activity_reader.snapshot());
    }

    #[test]
    fn reduced_idct_decoder_keeps_only_coefficients_read_by_scale() {
        let dc = trivial_dc_table();
        let raw = RawHuffmanTable {
            bits: [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0x41, 0x00]),
        };
        let ac = HuffmanTable::from_raw(&raw).unwrap();
        let bytes = [0b0001_0100u8, 0, 0, 0];
        let quant = [1u16; 64];
        let ignored_by_quarter = ZIGZAG[5] as usize;
        assert_eq!(ignored_by_quarter, 2);

        let mut full_reader = BitReader::new(&bytes);
        let mut quarter_reader = BitReader::new(&bytes);
        let mut half_reader = BitReader::new(&bytes);
        let mut full_prev_dc = 0i32;
        let mut quarter_prev_dc = 0i32;
        let mut half_prev_dc = 0i32;
        let mut full_block = CoefficientBlock::default();
        let mut quarter_block = CoefficientBlock::default();
        let mut half_block = CoefficientBlock::default();

        decode_block_with_activity(
            &mut full_reader,
            &dc,
            &ac,
            &mut full_prev_dc,
            &quant,
            &mut full_block,
        )
        .unwrap();
        let quarter_dc_only = decode_block_for_reduced_idct(
            &mut quarter_reader,
            &dc,
            &ac,
            &mut quarter_prev_dc,
            &quant,
            &mut quarter_block,
            ReducedIdctCoefficients::Quarter,
        )
        .unwrap();
        let half_dc_only = decode_block_for_reduced_idct(
            &mut half_reader,
            &dc,
            &ac,
            &mut half_prev_dc,
            &quant,
            &mut half_block,
            ReducedIdctCoefficients::Half,
        )
        .unwrap();

        assert_eq!(quarter_prev_dc, full_prev_dc);
        assert_eq!(half_prev_dc, full_prev_dc);
        assert_eq!(quarter_reader.snapshot(), full_reader.snapshot());
        assert_eq!(half_reader.snapshot(), full_reader.snapshot());
        assert_eq!(full_block.coefficients()[ignored_by_quarter], 1);
        assert_eq!(quarter_block.coefficients()[ignored_by_quarter], 0);
        assert_eq!(half_block.coefficients()[ignored_by_quarter], 1);
        assert!(quarter_dc_only);
        assert!(!half_dc_only);
    }

    #[test]
    fn one_by_one_idct_decoder_keeps_dc_and_skips_ac_values() {
        let dc = trivial_dc_table();
        let raw = RawHuffmanTable {
            bits: [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0x01, 0x00]),
        };
        let ac = HuffmanTable::from_raw(&raw).unwrap();
        let bytes = [0b0001_0100u8, 0, 0, 0];
        let quant = [1u16; 64];
        let mut full_reader = BitReader::new(&bytes);
        let mut one_by_one_reader = BitReader::new(&bytes);
        let mut full_prev_dc = 0i32;
        let mut one_by_one_prev_dc = 0i32;
        let mut full_block = CoefficientBlock::default();
        let mut one_by_one_block = CoefficientBlock::default();

        decode_block_with_activity(
            &mut full_reader,
            &dc,
            &ac,
            &mut full_prev_dc,
            &quant,
            &mut full_block,
        )
        .unwrap();
        decode_block_for_1x1_idct(
            &mut one_by_one_reader,
            &dc,
            &ac,
            &mut one_by_one_prev_dc,
            &quant,
            &mut one_by_one_block,
        )
        .unwrap();

        assert_eq!(one_by_one_prev_dc, full_prev_dc);
        assert_eq!(one_by_one_reader.snapshot(), full_reader.snapshot());
        assert_eq!(one_by_one_block.dc_coeff(), full_block.dc_coeff());
        assert!(one_by_one_block.coefficients()[1..].iter().all(|&c| c == 0));
    }

    #[test]
    fn skip_block_consumes_stream_and_updates_dc_like_decode() {
        let dc_raw = RawHuffmanTable {
            bits: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[2]),
        };
        let dc = HuffmanTable::from_raw(&dc_raw).unwrap();
        let ac_raw = RawHuffmanTable {
            bits: [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0x01, 0x00]),
        };
        let ac = HuffmanTable::from_raw(&ac_raw).unwrap();
        let bytes = [0b0011_0010u8, 0b1000_0000, 0, 0];
        let quant = [1u16; 64];

        let mut decoded_reader = BitReader::new(&bytes);
        let mut skipped_reader = BitReader::new(&bytes);
        let mut decoded_prev_dc = 5i32;
        let mut skipped_prev_dc = 5i32;
        let mut out = CoefficientBlock::default();

        decode_block_with_activity(
            &mut decoded_reader,
            &dc,
            &ac,
            &mut decoded_prev_dc,
            &quant,
            &mut out,
        )
        .unwrap();
        skip_block(&mut skipped_reader, &dc, &ac, &mut skipped_prev_dc).unwrap();

        assert_eq!(skipped_prev_dc, decoded_prev_dc);
        assert_eq!(skipped_reader.snapshot(), decoded_reader.snapshot());
    }

    #[test]
    fn extend_activity_promotes_top_half_ac_without_marking_general() {
        assert_eq!(
            extend_activity(BlockActivity::DcOnly, 31),
            BlockActivity::BottomHalfZero
        );
        assert_eq!(
            extend_activity(BlockActivity::BottomHalfZero, 7),
            BlockActivity::BottomHalfZero
        );
    }

    #[test]
    fn extend_activity_marks_bottom_half_ac_as_general() {
        assert_eq!(
            extend_activity(BlockActivity::DcOnly, 32),
            BlockActivity::General
        );
        assert_eq!(
            extend_activity(BlockActivity::BottomHalfZero, 40),
            BlockActivity::General
        );
    }

    #[test]
    fn switches_to_dense_clear_after_threshold_and_zeroes_full_block() {
        let mut block = CoefficientBlock::default();
        for (i, idx) in [0usize, 1, 8, 16, 24].into_iter().enumerate() {
            block.store(idx, (i + 1) as i16);
        }

        assert_eq!(block.clear_mode, ClearMode::Dense);
        block.clear_touched();

        assert!(block.coefficients().iter().all(|&c| c == 0));
        assert_eq!(block.touched_len, 0);
        assert_eq!(block.clear_mode, ClearMode::Sparse);
    }

    #[test]
    fn stays_sparse_below_dense_clear_threshold() {
        let mut block = CoefficientBlock::default();
        for (i, idx) in [0usize, 2, 4, 6].into_iter().enumerate() {
            block.store(idx, (i + 1) as i16);
        }

        assert_eq!(block.clear_mode, ClearMode::Sparse);
        assert_eq!(block.touched_len, DENSE_CLEAR_THRESHOLD);
    }
}
