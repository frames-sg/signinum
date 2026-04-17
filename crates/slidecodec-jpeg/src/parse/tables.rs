// SPDX-License-Identifier: Apache-2.0

//! Parse DQT (Define Quantization Table) and DHT (Define Huffman Table)
//! segments into slot-indexed table storage.

#![allow(dead_code)] // header parser in Task 14 wires these up.

use crate::error::JpegError;

/// Up to four quant tables (8-bit precision). 16-bit precision widens each
/// entry but the slot model is identical; we decode into u16 for uniformity.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct QuantTables {
    pub(crate) entries: [Option<[u16; 64]>; 4],
}

/// Up to four Huffman tables per class (DC/AC). Raw `bits[0..16]` counts and
/// `values[...]` payload; tree building happens in the entropy module (M1b).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct HuffmanTables {
    pub(crate) dc: [Option<RawHuffmanTable>; 4],
    pub(crate) ac: [Option<RawHuffmanTable>; 4],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RawHuffmanTable {
    pub(crate) bits: [u8; 16],
    pub(crate) values: alloc::vec::Vec<u8>,
}

/// Parse a DQT payload into one or more table slots. Multiple tables may be
/// concatenated within a single DQT marker per T.81 §B.2.4.1.
pub(crate) fn parse_dqt(
    payload: &[u8],
    payload_offset: usize,
    tables: &mut QuantTables,
) -> Result<(), JpegError> {
    let mut i = 0;
    while i < payload.len() {
        if i >= payload.len() {
            return Err(JpegError::Truncated {
                offset: payload_offset + i,
                expected: 1,
            });
        }
        let pq = payload[i] >> 4;
        let tq = (payload[i] & 0x0F) as usize;
        if tq > 3 {
            return Err(JpegError::InvalidSegmentLength {
                offset: payload_offset + i,
                marker: 0xDB,
                length: (payload.len() + 2) as u16,
            });
        }
        let entry_bytes = if pq == 0 { 1 } else { 2 };
        let needed = 1 + 64 * entry_bytes;
        if i + needed > payload.len() {
            return Err(JpegError::Truncated {
                offset: payload_offset + i + needed,
                expected: (i + needed) - payload.len(),
            });
        }
        let mut entries = [0u16; 64];
        if pq == 0 {
            for k in 0..64 {
                entries[k] = u16::from(payload[i + 1 + k]);
            }
        } else if pq == 1 {
            for k in 0..64 {
                entries[k] = u16::from_be_bytes([
                    payload[i + 1 + k * 2],
                    payload[i + 1 + k * 2 + 1],
                ]);
            }
        } else {
            return Err(JpegError::UnsupportedBitDepth { depth: pq });
        }
        tables.entries[tq] = Some(entries);
        i += needed;
    }
    Ok(())
}

/// Parse a DHT payload into the given table storage.
pub(crate) fn parse_dht(
    payload: &[u8],
    payload_offset: usize,
    tables: &mut HuffmanTables,
) -> Result<(), JpegError> {
    let mut i = 0;
    while i < payload.len() {
        if i + 17 > payload.len() {
            return Err(JpegError::Truncated {
                offset: payload_offset + i + 17,
                expected: (i + 17) - payload.len(),
            });
        }
        let tc = payload[i] >> 4;
        let th = (payload[i] & 0x0F) as usize;
        if th > 3 {
            return Err(JpegError::InvalidSegmentLength {
                offset: payload_offset + i,
                marker: 0xC4,
                length: (payload.len() + 2) as u16,
            });
        }
        let mut bits = [0u8; 16];
        bits.copy_from_slice(&payload[i + 1..i + 17]);
        let total_values: usize = bits.iter().map(|&b| b as usize).sum();
        if total_values > 256 {
            return Err(JpegError::InvalidSegmentLength {
                offset: payload_offset + i + 1,
                marker: 0xC4,
                length: (payload.len() + 2) as u16,
            });
        }
        if i + 17 + total_values > payload.len() {
            return Err(JpegError::Truncated {
                offset: payload_offset + i + 17 + total_values,
                expected: (i + 17 + total_values) - payload.len(),
            });
        }
        let values = payload[i + 17..i + 17 + total_values].to_vec();
        let table = RawHuffmanTable { bits, values };
        match tc {
            0 => tables.dc[th] = Some(table),
            1 => tables.ac[th] = Some(table),
            other => {
                return Err(JpegError::InvalidSegmentLength {
                    offset: payload_offset + i,
                    marker: 0xC4,
                    length: other as u16,
                });
            }
        }
        i += 17 + total_values;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ones_64() -> [u16; 64] { [1; 64] }

    #[test]
    fn parses_single_8bit_quant_table() {
        let mut payload = alloc::vec![0u8]; // Pq=0, Tq=0
        payload.extend(core::iter::repeat(1u8).take(64));
        let mut tables = QuantTables::default();
        parse_dqt(&payload, 0, &mut tables).unwrap();
        assert_eq!(tables.entries[0].unwrap(), ones_64());
    }

    #[test]
    fn parses_multiple_8bit_quant_tables_in_one_segment() {
        let mut payload = alloc::vec![0u8]; // Pq=0, Tq=0
        payload.extend(core::iter::repeat(1u8).take(64));
        payload.push(0x01); // Pq=0, Tq=1
        payload.extend(core::iter::repeat(2u8).take(64));
        let mut tables = QuantTables::default();
        parse_dqt(&payload, 0, &mut tables).unwrap();
        assert_eq!(tables.entries[0].unwrap(), [1u16; 64]);
        assert_eq!(tables.entries[1].unwrap(), [2u16; 64]);
    }

    #[test]
    fn parses_16bit_quant_table() {
        let mut payload = alloc::vec![0x10u8]; // Pq=1, Tq=0
        for _ in 0..64 {
            payload.extend_from_slice(&0x0102u16.to_be_bytes());
        }
        let mut tables = QuantTables::default();
        parse_dqt(&payload, 0, &mut tables).unwrap();
        assert_eq!(tables.entries[0].unwrap(), [0x0102u16; 64]);
    }

    #[test]
    fn rejects_truncated_dqt() {
        let payload = alloc::vec![0u8, 1, 2, 3];
        let mut tables = QuantTables::default();
        let err = parse_dqt(&payload, 0, &mut tables).unwrap_err();
        assert!(matches!(err, JpegError::Truncated { .. }));
    }

    #[test]
    fn parses_single_dc_huffman_table() {
        // Tc=0, Th=0, bits = [0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0], 12 values (standard JPEG luma DC).
        let mut payload = alloc::vec![0u8, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
        payload.extend_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        let mut tables = HuffmanTables::default();
        parse_dht(&payload, 0, &mut tables).unwrap();
        let t = tables.dc[0].as_ref().unwrap();
        assert_eq!(t.bits, [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(t.values, alloc::vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    }

    #[test]
    fn parses_multiple_huffman_tables_in_one_segment() {
        // First table: Tc=0 Th=0 with 1 value; second: Tc=1 Th=0 with 1 value
        let payload = alloc::vec![
            0u8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xAA,
            0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xBB,
        ];
        let mut tables = HuffmanTables::default();
        parse_dht(&payload, 0, &mut tables).unwrap();
        assert_eq!(tables.dc[0].as_ref().unwrap().values, alloc::vec![0xAA]);
        assert_eq!(tables.ac[0].as_ref().unwrap().values, alloc::vec![0xBB]);
    }

    #[test]
    fn rejects_huffman_with_more_than_256_values() {
        let mut payload = alloc::vec![0u8];
        // 16 counts summing to 257 (invalid)
        for _ in 0..16 {
            payload.push(17);
        }
        payload.push(0); // first value, won't get far
        let mut tables = HuffmanTables::default();
        let err = parse_dht(&payload, 0, &mut tables).unwrap_err();
        assert!(matches!(err, JpegError::InvalidSegmentLength { .. }));
    }
}
