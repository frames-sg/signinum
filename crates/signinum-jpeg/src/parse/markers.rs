// SPDX-License-Identifier: Apache-2.0

//! Low-level marker-walker. Handles the raw byte-stream structure:
//! SOI → sequence of `FFxx` markers with length-prefixed payloads, including
//! one or more SOS segments (each followed by entropy-coded scan data) →
//! terminating EOI.
//!
//! The walker treats EOI as the only terminator and returns SOS as a regular
//! length-prefixed marker. After a caller parses the SOS payload the entropy
//! decoder (`BitReader`) consumes the compressed scan bytes and surfaces the
//! next non-RST marker; progressive streams resume marker enumeration there.

use crate::error::{JpegError, MarkerKind};

/// One parsed marker plus a borrow of its payload bytes. For stand-alone
/// markers (SOI, EOI, RSTn, TEM) the payload slice is empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Marker<'a> {
    pub(crate) code: u8,
    pub(crate) offset: usize,
    pub(crate) payload: &'a [u8],
}

/// Cursor over a JPEG byte stream that yields marker segments.
pub(crate) struct MarkerWalker<'a> {
    bytes: &'a [u8],
    pos: usize,
    soi_seen: bool,
}

impl<'a> MarkerWalker<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            soi_seen: false,
        }
    }

    /// Read and consume the SOI marker (`FFD8`). Must be called before
    /// `next_marker`.
    pub(crate) fn read_soi(&mut self) -> Result<(), JpegError> {
        if self.bytes.len() < 2 {
            return Err(JpegError::Truncated {
                offset: self.pos,
                expected: 2 - self.bytes.len(),
            });
        }
        if self.bytes[0] != 0xFF || self.bytes[1] != 0xD8 {
            return Err(JpegError::UnexpectedMarker {
                offset: 0,
                expected: MarkerKind::Soi,
                found: self.bytes[1],
            });
        }
        self.pos = 2;
        self.soi_seen = true;
        Ok(())
    }

    /// Read the next marker. Returns `None` only when EOI is reached. SOS is
    /// returned as a normal length-prefixed marker — the entropy decoder is
    /// responsible for consuming the scan bytes that follow before the next
    /// marker can be read.
    ///
    /// Every marker must be preceded by at least one `0xFF` byte. If the next
    /// byte is not `0xFF`, the walker returns `JpegError::InvalidMarker`
    /// pointing at the unexpected byte — there is no "fall through to marker
    /// code" path, which prevents a stray `0x00` or random byte from being
    /// misinterpreted as a legal marker.
    pub(crate) fn next_marker(&mut self) -> Result<Option<Marker<'a>>, JpegError> {
        debug_assert!(self.soi_seen, "read_soi must be called before next_marker");

        // Require at least one 0xFF byte before the marker code. Consume any
        // additional 0xFF fill bytes per T.81 §B.1.1.2.
        let scan_start = self.pos;
        if self.pos >= self.bytes.len() {
            return Err(JpegError::Truncated {
                offset: self.pos,
                expected: 2,
            });
        }
        if self.bytes[self.pos] != 0xFF {
            return Err(JpegError::InvalidMarker {
                offset: self.pos,
                marker: self.bytes[self.pos],
            });
        }
        while self.pos < self.bytes.len() && self.bytes[self.pos] == 0xFF {
            self.pos += 1;
        }
        if self.pos >= self.bytes.len() {
            return Err(JpegError::Truncated {
                offset: self.pos,
                expected: 1,
            });
        }
        let _ = scan_start; // kept for future diagnostics

        let code = self.bytes[self.pos];
        let marker_offset = self.pos - 1; // safe: we consumed ≥1 0xFF
        self.pos += 1;

        match code {
            // Stand-alone markers with no payload (restart markers, TEM, and
            // the escape 0x00 which should not appear in header space).
            0x00 | 0x01 | 0xD0..=0xD7 => {
                return Ok(Some(Marker {
                    code,
                    offset: marker_offset,
                    payload: &[],
                }));
            }
            // EOI: end of image — terminates header walk.
            0xD9 => {
                return Ok(None);
            }
            _ => {}
        }

        // Length-prefixed marker (including SOS).
        if self.pos + 2 > self.bytes.len() {
            return Err(JpegError::Truncated {
                offset: self.pos,
                expected: self.pos + 2 - self.bytes.len(),
            });
        }
        let length = u16::from_be_bytes([self.bytes[self.pos], self.bytes[self.pos + 1]]);
        if length < 2 {
            return Err(JpegError::InvalidSegmentLength {
                offset: self.pos,
                marker: code,
                length,
            });
        }
        let payload_len = (length as usize) - 2;
        if self.pos + 2 + payload_len > self.bytes.len() {
            return Err(JpegError::Truncated {
                offset: self.pos + 2,
                expected: self.pos + 2 + payload_len - self.bytes.len(),
            });
        }
        let payload = &self.bytes[self.pos + 2..self.pos + 2 + payload_len];
        self.pos += 2 + payload_len;
        Ok(Some(Marker {
            code,
            offset: marker_offset,
            payload,
        }))
    }

    /// Byte offset of the leading 0xFF of the most recently returned marker.
    /// Valid after `next_marker()` returned `None` (EOI) or `Some(Marker)`.
    pub(crate) fn position(&self) -> usize {
        self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walker(bytes: &[u8]) -> MarkerWalker<'_> {
        let mut w = MarkerWalker::new(bytes);
        w.read_soi().unwrap();
        w
    }

    #[test]
    fn read_soi_accepts_valid_header() {
        let mut w = MarkerWalker::new(&[0xFF, 0xD8, 0xFF, 0xD9]);
        w.read_soi().unwrap();
    }

    #[test]
    fn read_soi_rejects_wrong_bytes() {
        let mut w = MarkerWalker::new(&[0x00, 0x00]);
        let err = w.read_soi().unwrap_err();
        assert!(matches!(
            err,
            JpegError::UnexpectedMarker {
                expected: MarkerKind::Soi,
                ..
            }
        ));
    }

    #[test]
    fn read_soi_rejects_truncated_input() {
        let mut w = MarkerWalker::new(&[0xFF]);
        let err = w.read_soi().unwrap_err();
        assert!(matches!(err, JpegError::Truncated { .. }));
    }

    #[test]
    fn next_marker_returns_none_at_eoi() {
        let mut w = walker(&[0xFF, 0xD8, 0xFF, 0xD9]);
        assert!(w.next_marker().unwrap().is_none());
    }

    #[test]
    fn next_marker_returns_sos_as_regular_marker() {
        // SOI + SOS len=12 + 10-byte payload + scan body + EOI
        let bytes = &[
            0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x0C, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0, 0x00, 0xFF,
            0xD9,
        ];
        let mut w = walker(bytes);
        let m = w.next_marker().unwrap().unwrap();
        assert_eq!(m.code, 0xDA);
        assert_eq!(m.payload.len(), 10);
    }

    #[test]
    fn next_marker_returns_payload_for_length_prefixed_marker() {
        // SOI + DQT (FFDB) len=5 with 3 payload bytes (len field counts itself) + EOI
        let bytes = &[
            0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x05, 0xAA, 0xBB, 0xCC, 0xFF, 0xD9,
        ];
        let mut w = walker(bytes);
        let m = w.next_marker().unwrap().unwrap();
        assert_eq!(m.code, 0xDB);
        assert_eq!(m.payload, &[0xAA, 0xBB, 0xCC]);
        assert!(w.next_marker().unwrap().is_none());
    }

    #[test]
    fn next_marker_rejects_length_less_than_two() {
        // SOI + DQT with length=1 (invalid)
        let bytes = &[0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x01];
        let mut w = walker(bytes);
        let err = w.next_marker().unwrap_err();
        assert!(matches!(
            err,
            JpegError::InvalidSegmentLength { length: 1, .. }
        ));
    }

    #[test]
    fn next_marker_rejects_truncated_payload() {
        // SOI + APP0 with length=100 but only 3 bytes of payload available
        let bytes = &[0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x64, 0xAA, 0xBB, 0xCC];
        let mut w = walker(bytes);
        let err = w.next_marker().unwrap_err();
        assert!(matches!(err, JpegError::Truncated { .. }));
    }

    #[test]
    fn next_marker_skips_fill_bytes() {
        // SOI + extra FFs before DQT
        let bytes = &[
            0xFF, 0xD8, 0xFF, 0xFF, 0xFF, 0xDB, 0x00, 0x03, 0xAA, 0xFF, 0xD9,
        ];
        let mut w = walker(bytes);
        let m = w.next_marker().unwrap().unwrap();
        assert_eq!(m.code, 0xDB);
        assert_eq!(m.payload, &[0xAA]);
    }

    #[test]
    fn next_marker_handles_standalone_rst() {
        let bytes = &[0xFF, 0xD8, 0xFF, 0xD0, 0xFF, 0xD9];
        let mut w = walker(bytes);
        let m = w.next_marker().unwrap().unwrap();
        assert_eq!(m.code, 0xD0);
        assert!(m.payload.is_empty());
    }

    #[test]
    fn next_marker_rejects_non_ff_byte_in_marker_position() {
        // After SOI, the parser expects 0xFF. A stray 0x00 must be an error,
        // not silently promoted to a marker code with negative offset.
        let bytes = &[0xFF, 0xD8, 0x00, 0xC0, 0xFF, 0xD9];
        let mut w = walker(bytes);
        let err = w.next_marker().unwrap_err();
        assert!(matches!(
            err,
            JpegError::InvalidMarker {
                marker: 0x00,
                offset: 2
            }
        ));
    }

    #[test]
    fn next_marker_rejects_random_byte_between_markers() {
        let bytes = &[0xFF, 0xD8, 0xAA, 0xC0, 0xFF, 0xD9];
        let mut w = walker(bytes);
        let err = w.next_marker().unwrap_err();
        assert!(matches!(err, JpegError::InvalidMarker { marker: 0xAA, .. }));
    }
}
