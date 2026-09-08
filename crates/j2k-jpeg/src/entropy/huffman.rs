// SPDX-License-Identifier: MIT OR Apache-2.0

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
use crate::parse::tables::{HuffmanTableRole, HuffmanValues, RawHuffmanTable};
use alloc::vec::Vec;
use core::num::NonZeroU32;

/// Number of fast-lookup entries. One per possible 12-bit peek value.
const FAST_BITS: u8 = 12;
const FAST_ENTRIES: usize = 1 << FAST_BITS;

const AC_FAST_KIND_SHIFT: u32 = 28;
pub(crate) const AC_FAST_KIND_MASK: u32 = 0xF << AC_FAST_KIND_SHIFT;
pub(crate) const AC_FAST_VALUE: u32 = 1 << AC_FAST_KIND_SHIFT;
pub(crate) const AC_FAST_EOB: u32 = 2 << AC_FAST_KIND_SHIFT;
pub(crate) const AC_FAST_ZRL: u32 = 3 << AC_FAST_KIND_SHIFT;
const AC_FAST_LEN_MASK: u32 = 0x0F;
const AC_FAST_RUN_MASK: u32 = 0xF0;
const AC_FAST_VALUE_SHIFT: u32 = 8;

const DC_FAST_LEN_MASK: u32 = 0x0F;
const DC_FAST_VALUE_SHIFT: u32 = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HuffmanTable {
    /// Fast path: `fast[peek12] = (symbol, bit_length)`. `bit_length == 0`
    /// means "code longer than 12 bits — use slow path".
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
    role: HuffmanTableRole,
    packed: [u32; FAST_ENTRIES],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DcHuffmanTable<'a>(&'a HuffmanTable);

#[derive(Clone, Copy, Debug)]
pub(crate) struct AcHuffmanTable<'a>(&'a HuffmanTable);

/// Checked handle into the one compiled-table arena retained by a decoder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PreparedHuffmanTableId(NonZeroU32);

/// Heap owner for every compiled Huffman table referenced by one decoder.
///
/// Table values are fully inline, so this vector is the only Huffman-table
/// allocation retained by prepared decode metadata.
#[derive(Debug)]
pub(crate) struct PreparedHuffmanTables {
    entries: Vec<HuffmanTable>,
}

pub(crate) type CanonicalHuffmanDerivation = j2k_codec_math::jpeg::CanonicalHuffmanDerivation;

pub(crate) fn derive_canonical_huffman(
    raw: &RawHuffmanTable,
) -> Result<CanonicalHuffmanDerivation, JpegError> {
    j2k_codec_math::jpeg::derive_canonical_huffman(&raw.bits, raw.values.len()).map_err(|_| {
        JpegError::HuffmanDecode {
            mcu: 0,
            reason: HuffmanFailure::CodeOverflow,
        }
    })
}

impl HuffmanTable {
    /// Build the decode table from a raw `(bits, values)` pair parsed out of
    /// a DHT segment. Per T.81 §C.2 and Annex C.
    ///
    /// # Errors
    /// - `HuffmanDecode { CodeOverflow }` if `bits` is oversubscribed (Kraft
    ///   inequality violated).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "canonical JPEG Huffman code lengths and symbol positions are bounded to 16 bits and 256 entries"
    )]
    pub(crate) fn from_raw(
        raw: &RawHuffmanTable,
        role: HuffmanTableRole,
    ) -> Result<Self, JpegError> {
        let canonical = derive_canonical_huffman(raw)?;
        let mut fast = [(0u8, 0u8); FAST_ENTRIES];
        let mut packed = [0u32; FAST_ENTRIES];

        let mut k = 0;
        for len_minus_1 in 0..FAST_BITS as usize {
            let len = (len_minus_1 + 1) as u8;
            let count = raw.bits[len_minus_1] as usize;
            for _ in 0..count {
                let c = canonical.huffcode[k];
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
            if role == HuffmanTableRole::Dc && sym <= 15 {
                let total_len = len + sym;
                if total_len <= FAST_BITS {
                    let diff = if sym == 0 {
                        0
                    } else {
                        let mag_shift = FAST_BITS - total_len;
                        let mag_mask = (1u16 << sym) - 1;
                        let mag_bits = ((idx as u16) >> mag_shift) & mag_mask;
                        huff_extend(i32::from(mag_bits), sym)
                    };
                    if (i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&diff) {
                        packed[idx] = pack_dc_value(total_len, diff as i16);
                    }
                }
                continue;
            }

            if role != HuffmanTableRole::Ac {
                continue;
            }
            let run = usize::from((sym >> 4) & 0x0F);
            let ssss = sym & 0x0F;
            if ssss == 0 {
                packed[idx] = match run {
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
            let value = huff_extend(i32::from(mag_bits), ssss);
            if !(i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&value) {
                continue;
            }
            packed[idx] = pack_ac_value(total_len, run as u8, value as i16);
        }

        Ok(Self {
            fast,
            min_code: canonical.min_code,
            max_code: canonical.max_code,
            val_offset: canonical.val_offset,
            values: raw.values.clone(),
            role,
            packed,
        })
    }

    pub(crate) fn dc(&self) -> Result<DcHuffmanTable<'_>, JpegError> {
        if self.role != HuffmanTableRole::Dc {
            return Err(JpegError::InternalInvariant {
                reason: "prepared AC Huffman table was resolved for a DC decode",
            });
        }
        Ok(DcHuffmanTable(self))
    }

    pub(crate) fn ac(&self) -> Result<AcHuffmanTable<'_>, JpegError> {
        if self.role != HuffmanTableRole::Ac {
            return Err(JpegError::InternalInvariant {
                reason: "prepared DC Huffman table was resolved for an AC decode",
            });
        }
        Ok(AcHuffmanTable(self))
    }

    /// Decode one symbol from the bit reader. Common case (code ≤ 12 bits) is
    /// a single array lookup; long codes fall through to a per-length scan.
    ///
    /// # Errors
    /// - `HuffmanDecode { TableExhausted }` if the stream ran out of bits.
    /// - `HuffmanDecode { CodeOverflow }` if no 1..=16-bit code matches.
    #[expect(
        clippy::inline_always,
        reason = "measured Huffman lookup hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        reason = "the Huffman slow path converts only validated 16-bit codes and non-negative table offsets"
    )]
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
                return self.values.get(idx).ok_or(JpegError::HuffmanDecode {
                    mcu: 0,
                    reason: HuffmanFailure::InvalidSymbol,
                });
            }
        }
        Err(JpegError::HuffmanDecode {
            mcu: 0,
            reason: HuffmanFailure::CodeOverflow,
        })
    }

    #[expect(
        clippy::inline_always,
        reason = "measured Huffman lookup hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    #[expect(
        clippy::cast_possible_wrap,
        reason = "packed DC values intentionally reinterpret a validated 16-bit two's-complement field"
    )]
    fn decode_fast_dc(&self, br: &mut BitReader<'_>) -> Result<i32, JpegError> {
        br.ensure_bits_padded(FAST_BITS)?;
        let peek = br.peek_bits(FAST_BITS) as usize;
        let packed = self.packed[peek];
        if packed != 0 {
            br.consume_bits((packed & DC_FAST_LEN_MASK) as u8);
            return Ok(i32::from(
                ((packed >> DC_FAST_VALUE_SHIFT) & 0xFFFF) as u16 as i16,
            ));
        }

        let ssss = self.decode(br)?;
        if ssss > 15 {
            return Err(JpegError::HuffmanDecode {
                mcu: 0,
                reason: HuffmanFailure::InvalidSymbol,
            });
        }
        br.receive_extend(ssss)
    }

    #[expect(
        clippy::inline_always,
        reason = "measured Huffman lookup hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "JPEG receive-extend values are bounded to the signed 16-bit packed AC field"
    )]
    fn decode_fast_ac(&self, br: &mut BitReader<'_>) -> Result<u32, JpegError> {
        br.ensure_bits_padded(FAST_BITS)?;
        let peek = br.peek_bits(FAST_BITS) as usize;
        let packed = self.packed[peek];
        if packed != 0 {
            br.consume_bits((packed & AC_FAST_LEN_MASK) as u8);
            return Ok(packed);
        }

        let (sym, len) = self.fast[peek];
        let sym = if len != 0 {
            br.consume_bits(len);
            sym
        } else {
            self.decode(br)?
        };

        let run = sym >> 4;
        let ssss = sym & 0x0F;
        if ssss == 0 {
            return Ok(if run == 15 {
                pack_ac_zrl(0)
            } else {
                pack_ac_eob(0)
            });
        }

        let value = br.receive_extend(ssss)?;
        Ok(pack_ac_value(0, run, value as i16))
    }

    #[expect(
        clippy::inline_always,
        reason = "measured Huffman lookup hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    fn skip_fast_ac(&self, br: &mut BitReader<'_>) -> Result<u32, JpegError> {
        br.ensure_bits_padded(FAST_BITS)?;
        let peek = br.peek_bits(FAST_BITS) as usize;
        let packed = self.packed[peek];
        if packed != 0 {
            br.consume_bits((packed & AC_FAST_LEN_MASK) as u8);
            return Ok(packed);
        }

        let (sym, len) = self.fast[peek];
        let sym = if len != 0 {
            br.consume_bits(len);
            sym
        } else {
            self.decode(br)?
        };

        let run = sym >> 4;
        let ssss = sym & 0x0F;
        if ssss == 0 {
            return Ok(if run == 15 {
                pack_ac_zrl(0)
            } else {
                pack_ac_eob(0)
            });
        }

        br.ensure_bits(ssss)?;
        br.consume_bits(ssss);
        Ok(pack_ac_value(0, run, 0))
    }
}

impl DcHuffmanTable<'_> {
    #[expect(
        clippy::inline_always,
        reason = "measured Huffman lookup hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    pub(crate) fn decode_fast_dc(self, br: &mut BitReader<'_>) -> Result<i32, JpegError> {
        self.0.decode_fast_dc(br)
    }

    pub(crate) fn decode(self, br: &mut BitReader<'_>) -> Result<u8, JpegError> {
        self.0.decode(br)
    }
}

impl AcHuffmanTable<'_> {
    #[expect(
        clippy::inline_always,
        reason = "measured Huffman lookup hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    pub(crate) fn decode_fast_ac(self, br: &mut BitReader<'_>) -> Result<u32, JpegError> {
        self.0.decode_fast_ac(br)
    }

    #[expect(
        clippy::inline_always,
        reason = "measured Huffman lookup hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    pub(crate) fn skip_fast_ac(self, br: &mut BitReader<'_>) -> Result<u32, JpegError> {
        self.0.skip_fast_ac(br)
    }

    pub(crate) fn decode(self, br: &mut BitReader<'_>) -> Result<u8, JpegError> {
        self.0.decode(br)
    }
}

impl PreparedHuffmanTableId {
    fn for_next_index(index: usize) -> Result<Self, JpegError> {
        let one_based = index
            .checked_add(1)
            .and_then(|value| u32::try_from(value).ok())
            .and_then(NonZeroU32::new)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })?;
        Ok(Self(one_based))
    }

    fn index(self) -> Option<usize> {
        usize::try_from(self.0.get())
            .ok()
            .and_then(|value| value.checked_sub(1))
    }
}

impl PreparedHuffmanTables {
    #[cfg(test)]
    pub(crate) fn try_with_capacity(capacity: usize) -> Result<Self, JpegError> {
        let mut live_bytes = 0;
        Self::try_with_capacity_and_live_budget(
            capacity,
            &mut live_bytes,
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )
    }

    pub(crate) fn try_with_capacity_and_live_budget(
        capacity: usize,
        live_bytes: &mut usize,
        cap: usize,
    ) -> Result<Self, JpegError> {
        let mut entries = Vec::new();
        crate::allocation::try_reserve_for_len_with_live_budget(
            &mut entries,
            capacity,
            live_bytes,
            cap,
        )?;
        Ok(Self { entries })
    }

    pub(crate) fn push(
        &mut self,
        table: HuffmanTable,
    ) -> Result<PreparedHuffmanTableId, JpegError> {
        let id = PreparedHuffmanTableId::for_next_index(self.entries.len())?;
        if self.entries.len() == self.entries.capacity() {
            return Err(JpegError::InternalInvariant {
                reason: "prepared Huffman arena exceeded its reserved capacity",
            });
        }
        self.entries.push(table);
        Ok(id)
    }

    pub(crate) fn get(&self, id: PreparedHuffmanTableId) -> Option<&HuffmanTable> {
        id.index().and_then(|index| self.entries.get(index))
    }

    pub(crate) fn get_dc(
        &self,
        id: PreparedHuffmanTableId,
    ) -> Result<DcHuffmanTable<'_>, JpegError> {
        self.get(id)
            .ok_or(JpegError::InternalInvariant {
                reason: "prepared component references a missing Huffman table",
            })?
            .dc()
    }

    pub(crate) fn get_ac(
        &self,
        id: PreparedHuffmanTableId,
    ) -> Result<AcHuffmanTable<'_>, JpegError> {
        self.get(id)
            .ok_or(JpegError::InternalInvariant {
                reason: "prepared component references a missing Huffman table",
            })?
            .ac()
    }

    pub(crate) fn id_at(&self, index: usize) -> Option<PreparedHuffmanTableId> {
        if index >= self.entries.len() {
            return None;
        }
        PreparedHuffmanTableId::for_next_index(index).ok()
    }

    pub(crate) fn retained_allocation_bytes(&self) -> Result<usize, JpegError> {
        crate::allocation::checked_allocation_bytes::<HuffmanTable>(self.entries.capacity())
    }

    pub(crate) fn try_clone_with_live_budget(
        &self,
        live_bytes: &mut usize,
        cap: usize,
    ) -> Result<Self, JpegError> {
        let mut entries = Vec::new();
        crate::allocation::try_reserve_for_len_with_live_budget(
            &mut entries,
            self.entries.len(),
            live_bytes,
            cap,
        )?;
        entries.extend(self.entries.iter().cloned());
        Ok(Self { entries })
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn capacity(&self) -> usize {
        self.entries.capacity()
    }
}

#[expect(
    clippy::inline_always,
    reason = "measured Huffman lookup hot path requires cross-helper inlining"
)]
#[inline(always)]
pub(crate) fn ac_decoded_run(packed: u32) -> usize {
    ((packed & AC_FAST_RUN_MASK) >> 4) as usize
}

#[expect(
    clippy::inline_always,
    reason = "measured Huffman lookup hot path requires cross-helper inlining"
)]
#[inline(always)]
#[expect(
    clippy::cast_possible_wrap,
    reason = "packed AC values intentionally reinterpret a 16-bit two's-complement field"
)]
pub(crate) fn ac_decoded_value(packed: u32) -> i32 {
    i32::from(((packed >> AC_FAST_VALUE_SHIFT) & 0xFFFF) as u16 as i16)
}

#[inline]
#[expect(
    clippy::cast_sign_loss,
    reason = "signed AC coefficients are intentionally stored as a 16-bit two's-complement bit field"
)]
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

#[inline]
#[expect(
    clippy::cast_sign_loss,
    reason = "signed DC coefficients are intentionally stored as a 16-bit two's-complement bit field"
)]
fn pack_dc_value(total_len: u8, value: i16) -> u32 {
    (u32::from(value as u16) << DC_FAST_VALUE_SHIFT) | u32::from(total_len)
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

    struct LegacyLookups {
        fast: [(u8, u8); FAST_ENTRIES],
        min_code: [i32; 17],
        max_code: [i32; 17],
        val_offset: [i32; 17],
        fast_dc: [u32; FAST_ENTRIES],
        fast_ac: [u32; FAST_ENTRIES],
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the legacy oracle reproduces conversions bounded by the 12-bit lookup and JPEG symbol fields"
    )]
    fn legacy_lookups(raw: &RawHuffmanTable) -> Result<LegacyLookups, JpegError> {
        let canonical = derive_canonical_huffman(raw)?;
        let mut fast = [(0u8, 0u8); FAST_ENTRIES];
        let mut fast_dc = [0u32; FAST_ENTRIES];
        let mut fast_ac = [0u32; FAST_ENTRIES];
        let mut k = 0;
        for len_minus_1 in 0..FAST_BITS as usize {
            let len = (len_minus_1 + 1) as u8;
            for _ in 0..raw.bits[len_minus_1] {
                let code = canonical.huffcode[k];
                let base = usize::from(code) << (FAST_BITS - len);
                let count = 1 << (FAST_BITS - len);
                fast[base..base + count].fill((raw.values.as_slice()[k], len));
                k += 1;
            }
        }
        for (idx, &(sym, len)) in fast.iter().enumerate() {
            if len == 0 {
                continue;
            }
            if sym <= 15 {
                let total_len = len + sym;
                if total_len <= FAST_BITS {
                    let diff = if sym == 0 {
                        0
                    } else {
                        let mag_shift = FAST_BITS - total_len;
                        let mag_mask = (1u16 << sym) - 1;
                        let mag_bits = ((idx as u16) >> mag_shift) & mag_mask;
                        huff_extend(i32::from(mag_bits), sym)
                    };
                    if (i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&diff) {
                        fast_dc[idx] = pack_dc_value(total_len, diff as i16);
                    }
                }
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
            let value = huff_extend(i32::from(mag_bits), ssss);
            if (i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&value) {
                fast_ac[idx] = pack_ac_value(total_len, run as u8, value as i16);
            }
        }
        Ok(LegacyLookups {
            fast,
            min_code: canonical.min_code,
            max_code: canonical.max_code,
            val_offset: canonical.val_offset,
            fast_dc,
            fast_ac,
        })
    }

    #[test]
    fn compiled_table_retains_only_one_packed_specialization() {
        assert!(core::mem::size_of::<HuffmanTable>() < 32 * 1024);
    }

    #[test]
    fn role_specializations_match_legacy_tables_for_all_fast_indices_and_code_lengths() {
        let standard_dc = luma_dc_raw();
        // Annex K luminance AC symbol order.
        let standard_ac_values = [
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51,
            0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1,
            0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18,
            0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
            0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57,
            0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
            0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92,
            0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
            0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
            0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8,
            0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2,
            0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
        ];
        let standard_ac = RawHuffmanTable {
            bits: [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D],
            values: HuffmanValues::from_slice(&standard_ac_values),
        };
        let length_values: Vec<u8> = (0..16).collect();
        let length_coverage = RawHuffmanTable {
            bits: [1; 16],
            values: HuffmanValues::from_slice(&length_values),
        };
        let empty = RawHuffmanTable {
            bits: [0; 16],
            values: HuffmanValues::default(),
        };
        let complete = RawHuffmanTable {
            bits: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0, 0xF0]),
        };

        for raw in [
            &standard_dc,
            &standard_ac,
            &length_coverage,
            &empty,
            &complete,
        ] {
            assert_matches_legacy(raw);
        }

        // Short codes exercise packed magnitudes and every run/EOB/ZRL symbol;
        // longer codes also verify that both specializations keep the slow path.
        for code_length in 1..=16 {
            for symbol in 0..=u8::MAX {
                let mut bits = [0; 16];
                bits[code_length - 1] = 1;
                assert_matches_legacy(&RawHuffmanTable {
                    bits,
                    values: HuffmanValues::from_slice(&[symbol]),
                });
            }
        }

        let oversubscribed = RawHuffmanTable {
            bits: [1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0, 1, 2, 3, 4]),
        };
        let Err(legacy_error) = legacy_lookups(&oversubscribed) else {
            panic!("legacy builder must reject an oversubscribed tree");
        };
        for role in [HuffmanTableRole::Dc, HuffmanTableRole::Ac] {
            assert_eq!(
                HuffmanTable::from_raw(&oversubscribed, role).unwrap_err(),
                legacy_error
            );
        }
    }

    fn assert_matches_legacy(raw: &RawHuffmanTable) {
        let legacy = legacy_lookups(raw).unwrap();
        for role in [HuffmanTableRole::Dc, HuffmanTableRole::Ac] {
            let table = HuffmanTable::from_raw(raw, role).unwrap();
            assert_eq!(table.fast, legacy.fast);
            assert_eq!(table.min_code, legacy.min_code);
            assert_eq!(table.max_code, legacy.max_code);
            assert_eq!(table.val_offset, legacy.val_offset);
            assert_eq!(
                table.packed,
                match role {
                    HuffmanTableRole::Dc => legacy.fast_dc,
                    HuffmanTableRole::Ac => legacy.fast_ac,
                }
            );
        }
    }

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
        let table = HuffmanTable::from_raw(&luma_dc_raw(), HuffmanTableRole::Dc).unwrap();
        let (sym, len) = table.fast[0b0000_0000_0000];
        assert_eq!((sym, len), (0, 2));
        let (sym, len) = table.fast[0b0011_1111_1111];
        assert_eq!((sym, len), (0, 2));
        let (sym, len) = table.fast[0b0100_0000_0000];
        assert_eq!((sym, len), (1, 3));
    }

    #[test]
    fn widened_fast_table_covers_9_bit_luma_dc_code() {
        let table = HuffmanTable::from_raw(&luma_dc_raw(), HuffmanTableRole::Dc).unwrap();
        let idx = 0b1_1111_1110usize << usize::from(FAST_BITS - 9);
        let (sym, len) = table.fast.get(idx).copied().unwrap_or((0, 0));
        assert_eq!((sym, len), (11, 9));
    }

    #[test]
    fn prepared_arena_ids_are_checked_and_capacity_bounded() {
        let table =
            HuffmanTable::from_raw(&luma_dc_raw(), HuffmanTableRole::Dc).expect("valid fixture");
        let mut arena = PreparedHuffmanTables::try_with_capacity(1).expect("bounded arena");
        let id = arena.push(table.clone()).expect("reserved slot");

        assert_eq!(arena.get(id), Some(&table));
        assert!(matches!(
            arena.push(table),
            Err(JpegError::InternalInvariant {
                reason: "prepared Huffman arena exceeded its reserved capacity"
            })
        ));
    }

    #[test]
    fn prepared_arena_validates_roles_and_accounts_actual_capacity() {
        let raw = luma_dc_raw();
        let mut arena = PreparedHuffmanTables::try_with_capacity(2).expect("bounded arena");
        let dc = arena
            .push(HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc).unwrap())
            .unwrap();
        let ac = arena
            .push(HuffmanTable::from_raw(&raw, HuffmanTableRole::Ac).unwrap())
            .unwrap();

        assert!(arena.get_dc(dc).is_ok());
        assert!(arena.get_ac(dc).is_err());
        assert!(arena.get_ac(ac).is_ok());
        assert!(arena.get_dc(ac).is_err());
        assert_eq!(
            arena.retained_allocation_bytes().unwrap(),
            arena.capacity() * core::mem::size_of::<HuffmanTable>()
        );
    }

    #[test]
    fn rejects_oversubscribed_code_table() {
        let raw = RawHuffmanTable {
            bits: [1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0, 1, 2, 3, 4]),
        };
        let err = HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc).unwrap_err();
        assert!(matches!(
            err,
            JpegError::HuffmanDecode {
                reason: HuffmanFailure::CodeOverflow,
                ..
            }
        ));
    }

    #[test]
    fn accepts_complete_prefix_table_for_decoder_compatibility() {
        let raw = RawHuffmanTable {
            bits: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0, 1]),
        };
        HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc)
            .expect("CPU decoder accepts a complete prefix table");
    }

    #[test]
    fn handles_empty_table_without_panic() {
        let raw = RawHuffmanTable {
            bits: [0; 16],
            values: HuffmanValues::default(),
        };
        let table = HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc).unwrap();
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
        let table = HuffmanTable::from_raw(&luma_dc_raw(), HuffmanTableRole::Dc).unwrap();
        for &(code, len, expected) in luma_dc_code_cases() {
            let shift = 32 - len;
            let aligned = code << shift;
            let bytes = aligned.to_be_bytes();
            let mut br = BitReader::new(&bytes);
            let sym = table.decode(&mut br).unwrap();
            assert_eq!(sym, expected, "code={code:b} len={len}");
        }
    }

    #[test]
    fn fast_dc_decodes_symbol_and_magnitude_in_one_lookup() {
        let table = HuffmanTable::from_raw(&luma_dc_raw(), HuffmanTableRole::Dc).unwrap();
        // Standard luma DC code `011` => category 2, followed by magnitude
        // bits `10` => diff +2. The fast DC path should consume all 5 bits.
        let bytes = [0b0111_0000u8, 0, 0, 0, 0, 0, 0, 0];
        let mut br = BitReader::new(&bytes);

        let diff = table.dc().unwrap().decode_fast_dc(&mut br).unwrap();

        assert_eq!(diff, 2);
        assert_eq!(br.snapshot().bits, 51);
        assert_eq!(br.peek_bits(3), 0);
    }

    #[test]
    fn decodes_single_bit_table_before_marker_padding() {
        let raw = RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0]),
        };
        let table = HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc).unwrap();
        let mut br = BitReader::new(&[0x7f, 0xff, 0xc4]);

        let symbol = table.decode(&mut br).unwrap();

        assert_eq!(symbol, 0);
    }

    #[test]
    fn decodes_9_plus_bit_codes_via_slow_path() {
        let table = HuffmanTable::from_raw(&luma_dc_raw(), HuffmanTableRole::Dc).unwrap();
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
        let table = HuffmanTable::from_raw(&luma_dc_raw(), HuffmanTableRole::Dc).unwrap();
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
