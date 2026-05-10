// SPDX-License-Identifier: Apache-2.0

//! Drive the marker walker and accumulate parsed segments into a `ParsedHeader`.
//! Produces the `Info` struct returned by `Decoder::inspect`.

use crate::error::{JpegError, MarkerKind, Warning};
use crate::info::{ColorSpace, Info, McuGeometry, SamplingFactors, SofKind};
use crate::parse::adobe_app14::{parse_adobe_app14, AdobeTransform};
use crate::parse::markers::MarkerWalker;
use crate::parse::scan::{parse_scan_header, ParsedScan};
use crate::parse::sof::parse_sof;
use crate::parse::tables::{parse_dht, parse_dqt, HuffmanTables, QuantTables};
use alloc::vec::Vec;
use memchr::memchr;

/// One entropy-coded progressive scan plus the table state active at its SOS.
#[derive(Debug, Clone)]
pub(crate) struct ParsedProgressiveScan {
    pub(crate) scan: ParsedScan,
    pub(crate) entropy_offset: usize,
    pub(crate) entropy_end: usize,
    pub(crate) huffman_tables: HuffmanTables,
    pub(crate) quant_tables: QuantTables,
    pub(crate) restart_interval: Option<u16>,
}

/// Everything the header walk produces. `Info` is derivable from this by
/// calling `.info()`.
#[derive(Debug)]
pub(crate) struct ParsedHeader {
    pub(crate) sof_kind: SofKind,
    pub(crate) bit_depth: u8,
    pub(crate) dimensions: (u32, u32),
    pub(crate) sampling: SamplingFactors,
    pub(crate) component_ids: Vec<u8>,
    pub(crate) quant_table_ids: Vec<u8>,
    pub(crate) quant_tables: QuantTables,
    pub(crate) huffman_tables: HuffmanTables,
    pub(crate) restart_interval: Option<u16>,
    pub(crate) adobe: Option<AdobeTransform>,
    /// Number of SOS markers observed in the input. The header walk still
    /// stops at the first SOS for decode setup, but we perform a lightweight
    /// post-SOS marker scan to count later scans for progressive metadata.
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
    pub(crate) progressive_scans: Vec<ParsedProgressiveScan>,
}

impl ParsedHeader {
    pub(crate) fn color_space(&self) -> ColorSpace {
        color_space_for_components(self.sampling.len(), self.adobe)
    }

    pub(crate) fn info(&self) -> Info {
        Info {
            dimensions: self.dimensions,
            color_space: self.color_space(),
            sampling: self.sampling,
            sof_kind: self.sof_kind,
            bit_depth: self.bit_depth,
            restart_interval: self.restart_interval,
            mcu_geometry: McuGeometry::from_sampling(self.dimensions, self.sampling),
            scan_count: self.scan_count,
        }
    }
}

pub(crate) fn parse_info(bytes: &[u8]) -> Result<Info, JpegError> {
    let mut walker = MarkerWalker::new(bytes);
    walker.read_soi()?;

    let mut sof: Option<crate::parse::sof::ParsedSof> = None;
    let mut restart_interval: Option<u16> = None;
    let mut adobe: Option<AdobeTransform> = None;
    let mut scan_count = 0u16;

    loop {
        match walker.next_marker()? {
            None => break,
            Some(m) => match m.code {
                0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                    if sof.is_some() {
                        return Err(JpegError::DuplicateMarker {
                            offset: m.offset,
                            marker: MarkerKind::Sof,
                        });
                    }
                    sof = Some(parse_sof(m.code, m.payload, m.offset + 4)?);
                }
                0xDD => {
                    if m.payload.len() != 2 {
                        return Err(JpegError::InvalidSegmentLength {
                            offset: m.offset,
                            marker: 0xDD,
                            length: (m.payload.len() + 2) as u16,
                        });
                    }
                    restart_interval = normalize_restart_interval(u16::from_be_bytes([
                        m.payload[0],
                        m.payload[1],
                    ]));
                }
                0xDA => {
                    let parsed = parse_scan_header(m.payload, m.offset + 4)?;
                    if let Some(sof) = sof.as_ref() {
                        validate_scan_parameters(sof.sof_kind, &parsed, m.offset + 4)?;
                        validate_sequential_scan_components(sof, &parsed, m.offset + 4)?;
                        scan_count = match sof.sof_kind {
                            SofKind::Progressive8 | SofKind::Progressive12 => {
                                count_scan_markers(bytes, walker.position())
                            }
                            _ => 1,
                        };
                    }
                    break;
                }
                0xEE => {
                    if let Some(t) = parse_adobe_app14(m.payload) {
                        adobe = Some(t);
                    }
                }
                0xDB | 0xC4 | 0xE0 | 0xE1..=0xEF | 0xFE => {}
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
    let dimensions = (u32::from(sof.width), u32::from(sof.height));
    Ok(Info {
        dimensions,
        color_space: color_space_for_components(sof.sampling.len(), adobe),
        sampling: sof.sampling,
        sof_kind: sof.sof_kind,
        bit_depth: sof.bit_depth,
        restart_interval,
        mcu_geometry: McuGeometry::from_sampling(dimensions, sof.sampling),
        scan_count,
    })
}

/// Walk headers from the start of the input.
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
    let mut progressive_scans: Vec<ParsedProgressiveScan> = Vec::new();

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
                    restart_interval = normalize_restart_interval(u16::from_be_bytes([
                        m.payload[0],
                        m.payload[1],
                    ]));
                }
                0xDA => {
                    let parsed = parse_scan_header(m.payload, m.offset + 4)?;
                    if let Some(sof) = sof.as_ref() {
                        validate_scan_parameters(sof.sof_kind, &parsed, m.offset + 4)?;
                        validate_sequential_scan_components(sof, &parsed, m.offset + 4)?;
                        validate_progressive_scan_components(sof, &parsed, m.offset + 4)?;
                    }
                    sos_offset = Some(walker.position());
                    scan = Some(parsed.clone());
                    if matches!(
                        sof.as_ref().map(|sof| sof.sof_kind),
                        Some(SofKind::Progressive8 | SofKind::Progressive12)
                    ) {
                        progressive_scans = collect_progressive_scans(
                            bytes,
                            parsed,
                            walker.position(),
                            &mut huffman_tables,
                            &mut quant_tables,
                            &mut restart_interval,
                            sof.as_ref().expect("SOF already checked"),
                            &mut warnings,
                        )?;
                        scan_count = progressive_scans.len().min(u16::MAX as usize) as u16;
                    } else {
                        scan_count = count_scan_markers(bytes, walker.position());
                    }
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
        component_ids: sof.component_ids,
        quant_table_ids: sof.quant_table_ids,
        quant_tables,
        huffman_tables,
        restart_interval,
        adobe,
        scan_count,
        warnings,
        sos_offset,
        scan,
        progressive_scans,
    })
}

fn validate_scan_parameters(
    sof_kind: SofKind,
    scan: &ParsedScan,
    offset: usize,
) -> Result<(), JpegError> {
    if matches!(sof_kind, SofKind::Baseline8 | SofKind::Extended8)
        && (scan.ss != 0 || scan.se != 63 || scan.ah != 0 || scan.al != 0)
    {
        return Err(JpegError::InvalidScanParameters {
            offset,
            ss: scan.ss,
            se: scan.se,
            ah: scan.ah,
            al: scan.al,
        });
    }
    Ok(())
}

fn normalize_restart_interval(interval: u16) -> Option<u16> {
    (interval > 0).then_some(interval)
}

fn validate_sequential_scan_components(
    sof: &crate::parse::sof::ParsedSof,
    scan: &ParsedScan,
    offset: usize,
) -> Result<(), JpegError> {
    if !matches!(sof.sof_kind, SofKind::Baseline8 | SofKind::Extended8) {
        return Ok(());
    }

    let mut seen = Vec::with_capacity(scan.components.len());
    for (i, comp) in scan.components.iter().enumerate() {
        let component_offset = offset + 1 + i * 2;
        if !sof.component_ids.contains(&comp.id) {
            return Err(JpegError::UnknownScanComponent {
                offset: component_offset,
                component: comp.id,
            });
        }
        if seen.contains(&comp.id) {
            return Err(JpegError::DuplicateScanComponent {
                offset: component_offset,
                component: comp.id,
            });
        }
        seen.push(comp.id);
    }

    if seen.len() != sof.component_ids.len() {
        return Err(JpegError::InvalidSequentialComponentSet {
            offset,
            expected: sof.component_ids.len() as u8,
            found: seen.len() as u8,
        });
    }

    Ok(())
}

fn validate_progressive_scan_components(
    sof: &crate::parse::sof::ParsedSof,
    scan: &ParsedScan,
    offset: usize,
) -> Result<(), JpegError> {
    if !matches!(sof.sof_kind, SofKind::Progressive8 | SofKind::Progressive12) {
        return Ok(());
    }
    if scan.components.is_empty()
        || scan.ss > scan.se
        || scan.se > 63
        || scan.ah > 13
        || scan.al > 13
        || (scan.ah != 0 && scan.ah != scan.al + 1)
        || (scan.ss == 0 && scan.se != 0)
        || (scan.ss > 0 && scan.components.len() != 1)
    {
        return Err(JpegError::InvalidScanParameters {
            offset,
            ss: scan.ss,
            se: scan.se,
            ah: scan.ah,
            al: scan.al,
        });
    }

    let mut seen = Vec::with_capacity(scan.components.len());
    for (i, comp) in scan.components.iter().enumerate() {
        let component_offset = offset + 1 + i * 2;
        if !sof.component_ids.contains(&comp.id) {
            return Err(JpegError::UnknownScanComponent {
                offset: component_offset,
                component: comp.id,
            });
        }
        if seen.contains(&comp.id) {
            return Err(JpegError::DuplicateScanComponent {
                offset: component_offset,
                component: comp.id,
            });
        }
        seen.push(comp.id);
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn collect_progressive_scans(
    bytes: &[u8],
    first_scan: ParsedScan,
    first_entropy_offset: usize,
    huffman_tables: &mut HuffmanTables,
    quant_tables: &mut QuantTables,
    restart_interval: &mut Option<u16>,
    sof: &crate::parse::sof::ParsedSof,
    warnings: &mut Vec<Warning>,
) -> Result<Vec<ParsedProgressiveScan>, JpegError> {
    let mut scans = Vec::new();
    let mut pending = Some(ParsedProgressiveScan {
        scan: first_scan,
        entropy_offset: first_entropy_offset,
        entropy_end: bytes.len(),
        huffman_tables: huffman_tables.clone(),
        quant_tables: quant_tables.clone(),
        restart_interval: *restart_interval,
    });
    let mut pos = first_entropy_offset;

    while let Some((marker_offset, code)) = next_marker_after_entropy(bytes, pos) {
        if let Some(mut scan) = pending.take() {
            scan.entropy_end = marker_offset;
            scans.push(scan);
        }

        match code {
            0xD9 => return Ok(scans),
            0xDB => {
                let (payload, next) = marker_payload(bytes, marker_offset, code)?;
                parse_dqt(payload, marker_offset + 4, quant_tables)?;
                pos = next;
            }
            0xC4 => {
                let (payload, next) = marker_payload(bytes, marker_offset, code)?;
                parse_dht(payload, marker_offset + 4, huffman_tables)?;
                pos = next;
            }
            0xDD => {
                let (payload, next) = marker_payload(bytes, marker_offset, code)?;
                if payload.len() != 2 {
                    return Err(JpegError::InvalidSegmentLength {
                        offset: marker_offset,
                        marker: 0xDD,
                        length: (payload.len() + 2) as u16,
                    });
                }
                *restart_interval =
                    normalize_restart_interval(u16::from_be_bytes([payload[0], payload[1]]));
                pos = next;
            }
            0xDA => {
                let (payload, entropy_offset) = marker_payload(bytes, marker_offset, code)?;
                let parsed = parse_scan_header(payload, marker_offset + 4)?;
                validate_scan_parameters(sof.sof_kind, &parsed, marker_offset + 4)?;
                validate_progressive_scan_components(sof, &parsed, marker_offset + 4)?;
                pending = Some(ParsedProgressiveScan {
                    scan: parsed,
                    entropy_offset,
                    entropy_end: bytes.len(),
                    huffman_tables: huffman_tables.clone(),
                    quant_tables: quant_tables.clone(),
                    restart_interval: *restart_interval,
                });
                pos = entropy_offset;
            }
            0xEE => {
                let (payload, next) = marker_payload(bytes, marker_offset, code)?;
                if let Some(t) = parse_adobe_app14(payload) {
                    if matches!(t, AdobeTransform::Unknown) && payload.len() >= 12 {
                        if payload[11] > 2 {
                            warnings.push(Warning::AdobeApp14Ambiguous {
                                raw_transform: payload[11],
                            });
                        }
                    }
                } else {
                    warnings.push(Warning::UnknownAppMarker {
                        marker: 0xEE,
                        size: payload.len(),
                    });
                }
                pos = next;
            }
            0xE0 | 0xFE => {
                let (_, next) = marker_payload(bytes, marker_offset, code)?;
                pos = next;
            }
            0xE1..=0xEF => {
                let (payload, next) = marker_payload(bytes, marker_offset, code)?;
                if code == 0xE2 {
                    warnings.push(Warning::IccProfileIgnored {
                        size: payload.len(),
                    });
                } else {
                    warnings.push(Warning::UnknownAppMarker {
                        marker: code,
                        size: payload.len(),
                    });
                }
                pos = next;
            }
            0x01 | 0xD8 => {
                pos = marker_offset + 2;
            }
            _ => {
                return Err(JpegError::InvalidMarker {
                    offset: marker_offset,
                    marker: code,
                });
            }
        }
    }

    if let Some(scan) = pending {
        scans.push(scan);
    }
    Ok(scans)
}

fn next_marker_after_entropy(bytes: &[u8], mut pos: usize) -> Option<(usize, u8)> {
    while pos < bytes.len() {
        let ff_rel = memchr(0xFF, &bytes[pos..])?;
        let marker_prefix = pos + ff_rel;
        let mut code_pos = marker_prefix + 1;
        while code_pos < bytes.len() && bytes[code_pos] == 0xFF {
            code_pos += 1;
        }
        if code_pos >= bytes.len() {
            return None;
        }
        let code = bytes[code_pos];
        match code {
            0x00 | 0xD0..=0xD7 => pos = code_pos + 1,
            _ => return Some((code_pos - 1, code)),
        }
    }
    None
}

fn marker_payload(
    bytes: &[u8],
    marker_offset: usize,
    marker: u8,
) -> Result<(&[u8], usize), JpegError> {
    let len_pos = marker_offset + 2;
    if len_pos + 1 >= bytes.len() {
        return Err(JpegError::Truncated {
            offset: len_pos,
            expected: len_pos + 2 - bytes.len(),
        });
    }
    let length = usize::from(u16::from_be_bytes([bytes[len_pos], bytes[len_pos + 1]]));
    if length < 2 {
        return Err(JpegError::InvalidSegmentLength {
            offset: len_pos,
            marker,
            length: length as u16,
        });
    }
    let payload_start = len_pos + 2;
    let payload_end = len_pos
        .checked_add(length)
        .ok_or(JpegError::InvalidSegmentLength {
            offset: len_pos,
            marker,
            length: length as u16,
        })?;
    if payload_end > bytes.len() {
        return Err(JpegError::Truncated {
            offset: payload_start,
            expected: payload_end - bytes.len(),
        });
    }
    Ok((&bytes[payload_start..payload_end], payload_end))
}

fn count_scan_markers(bytes: &[u8], mut pos: usize) -> u16 {
    let mut count = 1u16;
    while pos < bytes.len() {
        let Some(ff_rel) = memchr(0xFF, &bytes[pos..]) else {
            break;
        };
        let marker_offset = pos + ff_rel;
        let mut code_pos = marker_offset + 1;
        while code_pos < bytes.len() && bytes[code_pos] == 0xFF {
            code_pos += 1;
        }
        if code_pos >= bytes.len() {
            break;
        }
        pos = code_pos + 1;
        let code = bytes[code_pos];
        match code {
            0x00 => {}
            0xD0..=0xD7 => {}
            0xD9 => break,
            0xDA => {
                count = count.saturating_add(1);
                if let Some(next) = skip_marker_segment(bytes, marker_offset) {
                    pos = next;
                } else {
                    break;
                }
            }
            0x01 | 0xD8 => {
                pos = marker_offset + 2;
            }
            _ => {
                if let Some(next) = skip_marker_segment(bytes, marker_offset) {
                    pos = next;
                } else {
                    break;
                }
            }
        }
    }
    count
}

fn skip_marker_segment(bytes: &[u8], marker_offset: usize) -> Option<usize> {
    let len_pos = marker_offset + 2;
    if len_pos + 1 >= bytes.len() {
        return None;
    }
    let length = usize::from(u16::from_be_bytes([bytes[len_pos], bytes[len_pos + 1]]));
    if length < 2 {
        return None;
    }
    let next = len_pos.checked_add(length)?;
    if next > bytes.len() {
        return None;
    }
    Some(next)
}

fn color_space_for_components(component_count: usize, adobe: Option<AdobeTransform>) -> ColorSpace {
    match (component_count, adobe) {
        (1, _) => ColorSpace::Grayscale,
        (3, Some(AdobeTransform::YCbCr)) => ColorSpace::YCbCr,
        (3, Some(AdobeTransform::Unknown)) => ColorSpace::Rgb,
        (3, None) => ColorSpace::YCbCr,
        (3, Some(AdobeTransform::Ycck)) => ColorSpace::YCbCr,
        (4, Some(AdobeTransform::Ycck)) => ColorSpace::Ycck,
        (4, _) => ColorSpace::Cmyk,
        _ => ColorSpace::YCbCr,
    }
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
        v.extend(core::iter::repeat_n(1u8, 64));

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
        assert_eq!(h.sampling.components(), &[(2, 2), (1, 1), (1, 1)]);
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
    fn scan_count_tracks_progressive_sos_markers() {
        let mut bytes = minimal_baseline_jpeg();
        let sof_pos = bytes.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        bytes[sof_pos + 1] = 0xC2;
        let first_sos_pos = bytes.windows(2).position(|w| w == [0xFF, 0xDA]).unwrap();
        bytes[first_sos_pos + 12] = 0;
        let eoi_pos = bytes.windows(2).rposition(|w| w == [0xFF, 0xD9]).unwrap();
        let second_scan = vec![
            0xFF, 0xDA, 0x00, 12, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 0, 0x10, 0x00,
        ];
        bytes.splice(eoi_pos..eoi_pos, second_scan.iter().copied());
        let h = parse_header(&bytes).unwrap();
        assert_eq!(h.sof_kind, SofKind::Progressive8);
        assert_eq!(h.scan_count, 2);
        assert_eq!(h.progressive_scans.len(), 2);
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
