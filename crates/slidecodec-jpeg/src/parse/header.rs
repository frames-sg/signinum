// SPDX-License-Identifier: Apache-2.0

//! Drive the marker walker and accumulate parsed segments into a `ParsedHeader`.
//! Produces the `Info` struct returned by `Decoder::inspect`.

#![allow(dead_code)] // M1b consumes quant_table_ids/warnings/huffman_tables/…

use crate::error::{JpegError, MarkerKind, Warning};
use crate::info::{ColorSpace, Info, SamplingFactors, SofKind};
use crate::parse::adobe_app14::{parse_adobe_app14, AdobeTransform};
use crate::parse::markers::MarkerWalker;
use crate::parse::scan::{parse_scan_header, ParsedScan};
use crate::parse::sof::parse_sof;
use crate::parse::tables::{parse_dht, parse_dqt, HuffmanTables, QuantTables};
use alloc::vec::Vec;

/// Everything the header walk produces. `Info` is derivable from this by
/// calling `.info()`.
#[derive(Debug)]
pub(crate) struct ParsedHeader {
    pub(crate) sof_kind: SofKind,
    pub(crate) bit_depth: u8,
    pub(crate) dimensions: (u32, u32),
    pub(crate) sampling: SamplingFactors,
    pub(crate) quant_table_ids: Vec<u8>,
    pub(crate) quant_tables: QuantTables,
    pub(crate) huffman_tables: HuffmanTables,
    pub(crate) restart_interval: Option<u16>,
    pub(crate) adobe: Option<AdobeTransform>,
    /// Number of SOS markers observed during header parsing. `parse_header`
    /// stops at the first SOS, so this is always `1` when a scan exists and
    /// `0` when the stream ends before any SOS (which is then rejected as
    /// `MissingMarker { Sos }` before `ParsedHeader` is returned).
    ///
    /// Callers that need the full count for progressive streams obtain it
    /// from the decode path in M1b (which walks every scan).
    pub(crate) scan_count: u16,
    pub(crate) warnings: Vec<Warning>,
    /// Byte offset of the first entropy byte after the first SOS header —
    /// i.e., the cursor position *immediately after* the SOS marker and
    /// payload, not the leading 0xFF of FFDA. Consumed by M1b's entropy
    /// decoder as the start of the scan bitstream.
    pub(crate) sos_offset: Option<usize>,
    /// Parsed SOS header — which components participate in the first scan and
    /// their Huffman table selectors. `None` iff `sos_offset` is also `None`
    /// (stream ended before any SOS — rejected as `MissingMarker { Sos }`
    /// before `ParsedHeader` is returned).
    pub(crate) scan: Option<ParsedScan>,
}

impl ParsedHeader {
    pub(crate) fn color_space(&self) -> ColorSpace {
        // Per spec Section 4 matrix.
        match (self.sampling.components.len(), self.adobe) {
            (1, _) => ColorSpace::Grayscale,
            (3, Some(AdobeTransform::YCbCr)) => ColorSpace::YCbCr,
            (3, Some(AdobeTransform::Unknown)) => ColorSpace::Rgb,
            (3, None) => ColorSpace::YCbCr, // JFIF default
            (3, Some(AdobeTransform::Ycck)) => ColorSpace::YCbCr, // invalid combo, fall back
            (4, Some(AdobeTransform::Ycck)) => ColorSpace::Ycck,
            (4, _) => ColorSpace::Cmyk,
            _ => ColorSpace::YCbCr, // unreachable — SOF parser already rejected other counts
        }
    }

    pub(crate) fn info(&self) -> Info {
        Info {
            dimensions: self.dimensions,
            color_space: self.color_space(),
            sampling: self.sampling.clone(),
            sof_kind: self.sof_kind,
            bit_depth: self.bit_depth,
            restart_interval: self.restart_interval,
            scan_count: self.scan_count,
        }
    }
}

/// Walk headers from the start of the input. Stops at the first SOS.
pub(crate) fn parse_header(bytes: &[u8]) -> Result<ParsedHeader, JpegError> {
    let mut walker = MarkerWalker::new(bytes);
    walker.read_soi()?;

    let mut sof: Option<crate::parse::sof::ParsedSof> = None;
    let mut sof_seen_code: Option<u8> = None;
    let mut quant_tables = QuantTables::default();
    let mut huffman_tables = HuffmanTables::default();
    let mut restart_interval: Option<u16> = None;
    let mut adobe: Option<AdobeTransform> = None;
    let mut warnings: Vec<Warning> = Vec::new();
    let mut scan_count = 0u16;
    let mut sos_offset: Option<usize> = None;
    let mut scan: Option<ParsedScan> = None;

    loop {
        match walker.next_marker()? {
            None => {
                // EOI — header walk complete.
                break;
            }
            Some(m) => match m.code {
                // SOF (we only accept four, everything else in 0xC* is routed
                // to `parse_sof` which returns UnsupportedSof).
                0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                    if sof.is_some() {
                        return Err(JpegError::DuplicateMarker {
                            offset: m.offset,
                            marker: MarkerKind::Sof,
                        });
                    }
                    sof = Some(parse_sof(m.code, m.payload, m.offset + 4)?);
                    sof_seen_code = Some(m.code);
                }
                0xDB => parse_dqt(m.payload, m.offset + 4, &mut quant_tables)?,
                0xC4 => parse_dht(m.payload, m.offset + 4, &mut huffman_tables)?,
                0xDD => {
                    if m.payload.len() != 2 {
                        return Err(JpegError::InvalidSegmentLength {
                            offset: m.offset,
                            marker: 0xDD,
                            length: (m.payload.len() + 2) as u16,
                        });
                    }
                    restart_interval = Some(u16::from_be_bytes([m.payload[0], m.payload[1]]));
                }
                0xDA => {
                    let parsed = parse_scan_header(m.payload, m.offset + 4)?;
                    sos_offset = Some(walker.position());
                    scan = Some(parsed);
                    scan_count = 1;
                    break;
                }
                0xEE => {
                    // APP14
                    if let Some(t) = parse_adobe_app14(m.payload) {
                        adobe = Some(t);
                        if matches!(t, AdobeTransform::Unknown) && m.payload.len() >= 12 {
                            if m.payload[11] > 2 {
                                warnings.push(Warning::AdobeApp14Ambiguous {
                                    raw_transform: m.payload[11],
                                });
                            }
                        }
                    } else {
                        warnings.push(Warning::UnknownAppMarker {
                            marker: 0xEE,
                            size: m.payload.len(),
                        });
                    }
                }
                0xE0 => {
                    // APP0 JFIF — presence noted, contents not interpreted in v1.
                }
                0xE2 => {
                    warnings.push(Warning::IccProfileIgnored {
                        size: m.payload.len(),
                    });
                }
                0xE1..=0xEF => {
                    warnings.push(Warning::UnknownAppMarker {
                        marker: m.code,
                        size: m.payload.len(),
                    });
                }
                0xFE => {
                    // COM — ignored silently.
                }
                _ => {
                    return Err(JpegError::InvalidMarker {
                        offset: m.offset,
                        marker: m.code,
                    });
                }
            },
        }
    }

    let sof = sof.ok_or(JpegError::MissingMarker {
        marker: MarkerKind::Sof,
    })?;
    let _ = sof_seen_code; // reserved for future use (progressive / lossless routing)

    Ok(ParsedHeader {
        sof_kind: sof.sof_kind,
        bit_depth: sof.bit_depth,
        dimensions: (u32::from(sof.width), u32::from(sof.height)),
        sampling: sof.sampling,
        quant_table_ids: sof.quant_table_ids,
        quant_tables,
        huffman_tables,
        restart_interval,
        adobe,
        scan_count,
        warnings,
        sos_offset,
        scan,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Build a tiny but structurally valid baseline JPEG header (SOI + DQT +
    /// SOF0 4:2:0 + DHT DC0 + DHT AC0 + SOS + dummy scan byte + EOI).
    fn minimal_baseline_jpeg() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&[0xFF, 0xD8]); // SOI

        // DQT: Pq=0 Tq=0 followed by 64 bytes of 1
        v.extend_from_slice(&[0xFF, 0xDB, 0x00, 67, 0x00]);
        v.extend(core::iter::repeat(1u8).take(64));

        // SOF0: precision 8, 16x16, 3 components, Y(2x2) Cb(1x1) Cr(1x1), all using Tq=0
        v.extend_from_slice(&[
            0xFF,
            0xC0,
            0x00,
            17,
            8,
            0,
            16,
            0,
            16,
            3,
            1,
            (2 << 4) | 2,
            0,
            2,
            (1 << 4) | 1,
            0,
            3,
            (1 << 4) | 1,
            0,
        ]);

        // DHT DC0 and AC0 (minimal: 1 value each)
        // DHT length = 2 (length field) + 1 (Tc/Th) + 16 (bits[]) + 1 (value) = 20
        v.extend_from_slice(&[
            0xFF, 0xC4, 0x00, 20, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xAA,
        ]);
        v.extend_from_slice(&[
            0xFF, 0xC4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xBB,
        ]);

        // SOS: 3 components, Ss=0 Se=63 Ah=0 Al=0
        v.extend_from_slice(&[0xFF, 0xDA, 0x00, 12, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0]);

        // Minimal scan data + EOI
        v.extend_from_slice(&[0x00, 0xFF, 0xD9]);
        v
    }

    #[test]
    fn parses_minimal_baseline_jpeg() {
        let h = parse_header(&minimal_baseline_jpeg()).unwrap();
        assert_eq!(h.dimensions, (16, 16));
        assert_eq!(h.sof_kind, SofKind::Baseline8);
        assert_eq!(h.color_space(), ColorSpace::YCbCr);
        assert_eq!(h.bit_depth, 8);
        assert_eq!(h.sampling.components, vec![(2, 2), (1, 1), (1, 1)]);
        assert!(h.quant_tables.entries[0].is_some());
        assert!(h.huffman_tables.dc[0].is_some());
        assert!(h.huffman_tables.ac[0].is_some());
        assert!(h.sos_offset.is_some());
        assert_eq!(h.scan_count, 1);
    }

    #[test]
    fn rejects_missing_sof() {
        // SOI directly followed by SOS/EOI
        let bytes = vec![0xFF, 0xD8, 0xFF, 0xD9];
        let err = parse_header(&bytes).unwrap_err();
        assert!(matches!(
            err,
            JpegError::MissingMarker {
                marker: MarkerKind::Sof
            }
        ));
    }

    #[test]
    fn rejects_duplicate_sof() {
        let mut bytes = minimal_baseline_jpeg();
        // Insert a second SOF0 before SOS. Find SOS offset and splice.
        let sos_pos = bytes.windows(2).position(|w| w == [0xFF, 0xDA]).unwrap();
        let second_sof = vec![
            0xFF,
            0xC0,
            0x00,
            17,
            8,
            0,
            16,
            0,
            16,
            3,
            1,
            (2 << 4) | 2,
            0,
            2,
            (1 << 4) | 1,
            0,
            3,
            (1 << 4) | 1,
            0,
        ];
        bytes.splice(sos_pos..sos_pos, second_sof.iter().copied());
        let err = parse_header(&bytes).unwrap_err();
        assert!(matches!(
            err,
            JpegError::DuplicateMarker {
                marker: MarkerKind::Sof,
                ..
            }
        ));
    }

    #[test]
    fn info_method_produces_expected_fields() {
        let h = parse_header(&minimal_baseline_jpeg()).unwrap();
        let info = h.info();
        assert_eq!(info.dimensions, (16, 16));
        assert_eq!(info.sof_kind, SofKind::Baseline8);
        assert_eq!(info.scan_count, 1);
    }

    #[test]
    fn app14_ycbcr_overrides_default() {
        let mut bytes = minimal_baseline_jpeg();
        // Insert APP14 marker right after SOI: FF EE len=16 [Adobe..Transform=1]
        let mut app14 = vec![0xFF, 0xEE, 0x00, 14];
        app14.extend_from_slice(b"Adobe");
        app14.extend_from_slice(&[0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0x01]);
        bytes.splice(2..2, app14.iter().copied());
        let h = parse_header(&bytes).unwrap();
        assert_eq!(h.color_space(), ColorSpace::YCbCr);
    }

    #[test]
    fn app14_unknown_marks_rgb_for_3_components() {
        let mut bytes = minimal_baseline_jpeg();
        let mut app14 = vec![0xFF, 0xEE, 0x00, 14];
        app14.extend_from_slice(b"Adobe");
        app14.extend_from_slice(&[0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00]);
        bytes.splice(2..2, app14.iter().copied());
        let h = parse_header(&bytes).unwrap();
        assert_eq!(h.color_space(), ColorSpace::Rgb);
    }

    #[test]
    fn scan_count_is_one_after_header_only_parse() {
        // Even for a progressive-shaped stream with two SOSes, header-only
        // parsing stops at the first SOS. The full count is the decode
        // path's responsibility (M1b) and is not exercised here.
        let mut bytes = minimal_baseline_jpeg();
        let eoi_pos = bytes.windows(2).rposition(|w| w == [0xFF, 0xD9]).unwrap();
        let second_scan = vec![
            0xFF, 0xDA, 0x00, 12, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0, 0x00,
        ];
        bytes.splice(eoi_pos..eoi_pos, second_scan.iter().copied());
        let h = parse_header(&bytes).unwrap();
        assert_eq!(
            h.scan_count, 1,
            "header-only inspect sees only the first SOS"
        );
    }

    #[test]
    fn sos_offset_points_at_first_entropy_byte() {
        let bytes = minimal_baseline_jpeg();
        let h = parse_header(&bytes).unwrap();
        let sos_marker_pos = bytes.windows(2).position(|w| w == [0xFF, 0xDA]).unwrap();
        // SOS marker (2) + length field (2) + 10-byte payload = 14 bytes.
        let expected_first_entropy_byte = sos_marker_pos + 14;
        assert_eq!(
            h.sos_offset,
            Some(expected_first_entropy_byte),
            "sos_offset must point at the first entropy byte, not the leading FFDA"
        );
    }

    #[test]
    fn extracts_scan_component_table_selectors() {
        let h = parse_header(&minimal_baseline_jpeg()).unwrap();
        let scan = h.scan.as_ref().expect("SOS must be parsed");
        assert_eq!(scan.components.len(), 3);
        // minimal_baseline_jpeg uses Td=0 Ta=0 for every component.
        for (i, comp) in scan.components.iter().enumerate() {
            assert_eq!(comp.dc_table, 0, "component {i}");
            assert_eq!(comp.ac_table, 0, "component {i}");
        }
        assert_eq!((scan.ss, scan.se, scan.ah, scan.al), (0, 63, 0, 0));
    }

    #[test]
    fn rejects_malformed_sos_length() {
        // SOS with length claiming 12 bytes but only containing 4 payload bytes
        let mut bytes = minimal_baseline_jpeg();
        let sos_pos = bytes.windows(2).position(|w| w == [0xFF, 0xDA]).unwrap();
        // Find the end of the SOS segment (SOS header is 14 bytes: FF DA + len + 10 payload bytes)
        let sos_end = sos_pos + 2 + 12;
        // Truncate the SOS segment so its declared length (12) extends beyond the buffer.
        bytes.drain(sos_pos + 4..sos_end);
        let err = parse_header(&bytes).unwrap_err();
        assert!(matches!(
            err,
            JpegError::Truncated { .. } | JpegError::InvalidSegmentLength { .. }
        ));
    }
}
