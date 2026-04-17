// SPDX-License-Identifier: Apache-2.0

//! Public [`Decoder`] entry points. M1a exposes [`Decoder::inspect`] only;
//! [`Decoder::new`] and the decode methods land in M1b.

use crate::error::JpegError;
use crate::error::Warning;
use crate::info::Info;
use crate::info::Rect;
use crate::info::SofKind;
use crate::parse::header::parse_header;
use crate::parse::header::ParsedHeader;
use alloc::vec::Vec;

/// Non-fatal outcome of a successful decode. See spec Section 2.
///
/// `DecodeOutcome` lives on `decoder.rs` rather than `info.rs` because it
/// carries `Warning` values from `error.rs`, and moving it into `info` would
/// create a `info → error` cycle (see `info.rs` header note).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeOutcome {
    /// The rectangle actually written to the output buffer. For `decode_into`
    /// this is always `Rect::full(info.dimensions)`; later milestones add
    /// `decode_region_into` which can return a narrower rect.
    pub decoded: Rect,
    /// Warnings emitted during parse or decode. Empty when the stream is
    /// syntactically clean and every capability was exercised without fallback.
    pub warnings: Vec<Warning>,
}

/// A borrowed view of a JPEG stream ready to decode. Constructed via
/// [`Decoder::new`]. Internally holds the parsed header snapshot plus a borrow
/// of the input bytes.
///
/// `Decoder<'a>: Send + Sync` because all state is either POD (`Info`) or
/// immutable after construction (`ParsedHeader`). No interior mutability.
#[derive(Debug)]
pub struct Decoder<'a> {
    // `bytes` and `header` are stashed for `decode_into` (Task 15); currently
    // unread by the M1b public surface which only exposes `inspect` + `new`.
    #[allow(dead_code)]
    pub(crate) bytes: &'a [u8],
    #[allow(dead_code)]
    pub(crate) header: ParsedHeader,
    pub(crate) info: Info,
}

impl<'a> Decoder<'a> {
    /// Parse the headers without decoding any pixels. Cheap — O(header size).
    ///
    /// # Errors
    /// Returns any structural, unsupported-SOF, or sanity-check error
    /// encountered before the Start-of-Scan marker. See [`JpegError`].
    pub fn inspect(input: &'a [u8]) -> Result<Info, JpegError> {
        parse_header(input).map(|h| h.info())
    }

    /// Build a decoder ready for `decode_into`. Parses the full header, builds
    /// any derived lookup structures, and validates that the stream is one of
    /// the SOFs this release implements.
    ///
    /// # Errors
    /// - Any parse error encountered before SOS (see [`Self::inspect`]).
    /// - [`JpegError::NotImplemented`] for SOFs that parse but are not yet
    ///   decodable (Extended12, Progressive, Lossless — all land in M3).
    pub fn new(input: &'a [u8]) -> Result<Self, JpegError> {
        let header = parse_header(input)?;
        let info = header.info();
        match info.sof_kind {
            SofKind::Baseline8 | SofKind::Extended8 => {}
            other => return Err(JpegError::NotImplemented { sof: other }),
        }
        Ok(Self {
            bytes: input,
            header,
            info,
        })
    }

    /// The parsed header as a public [`Info`]. Cheap to clone; safe to log.
    pub fn info(&self) -> &Info {
        &self.info
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Warning;
    use crate::info::Rect;
    use alloc::vec;

    #[test]
    fn decode_outcome_carries_decoded_rect_and_warnings() {
        let outcome = DecodeOutcome {
            decoded: Rect { x: 0, y: 0, w: 32, h: 16 },
            warnings: vec![Warning::MissingEoi],
        };
        assert_eq!(outcome.decoded.w, 32);
        assert_eq!(outcome.decoded.h, 16);
        assert_eq!(outcome.warnings.len(), 1);
    }

    #[test]
    fn decode_outcome_defaults_to_empty_warnings() {
        let outcome = DecodeOutcome {
            decoded: Rect::full((8, 8)),
            warnings: Vec::new(),
        };
        assert!(outcome.warnings.is_empty());
    }

    // Reproduces the minimal baseline JPEG fixture used in parse::header::tests.
    // Duplicated because we cannot import pub(crate) test helpers from unit tests
    // of another module. See `tests/inspect.rs` for the integration-test copy.
    fn minimal_baseline_jpeg() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&[0xFF, 0xD8]);
        v.extend_from_slice(&[0xFF, 0xDB, 0x00, 67, 0x00]);
        v.extend(core::iter::repeat(1u8).take(64));
        v.extend_from_slice(&[
            0xFF, 0xC0, 0x00, 17, 8, 0, 16, 0, 16, 3,
            1, (2 << 4) | 2, 0, 2, (1 << 4) | 1, 0, 3, (1 << 4) | 1, 0,
        ]);
        v.extend_from_slice(&[
            0xFF, 0xC4, 0x00, 20, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xAA,
        ]);
        v.extend_from_slice(&[
            0xFF, 0xC4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xBB,
        ]);
        v.extend_from_slice(&[0xFF, 0xDA, 0x00, 12, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0]);
        v.extend_from_slice(&[0x00, 0xFF, 0xD9]);
        v
    }

    #[test]
    fn decoder_new_succeeds_on_baseline_stream() {
        let bytes = minimal_baseline_jpeg();
        let dec = Decoder::new(&bytes).expect("baseline stream must construct");
        assert_eq!(dec.info().dimensions, (16, 16));
        assert_eq!(dec.info().sof_kind, crate::info::SofKind::Baseline8);
    }

    #[test]
    fn decoder_new_rejects_arithmetic_coding_as_unsupported() {
        let mut bytes = minimal_baseline_jpeg();
        let pos = bytes.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        bytes[pos + 1] = 0xC9;
        let err = Decoder::new(&bytes).unwrap_err();
        assert!(err.is_unsupported());
    }

    #[test]
    fn decoder_new_reports_not_implemented_for_progressive() {
        // Swap SOF0 → SOF2 (progressive 8-bit) — parses fine but M1b cannot decode.
        let mut bytes = minimal_baseline_jpeg();
        let pos = bytes.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        bytes[pos + 1] = 0xC2;
        let err = Decoder::new(&bytes).unwrap_err();
        assert!(err.is_not_implemented());
        assert!(
            !err.is_unsupported(),
            "Progressive support lands in M3; M1b must NOT report it as a permanent routing decision"
        );
    }

    #[test]
    fn decoder_new_reports_not_implemented_for_lossless() {
        let mut bytes = minimal_baseline_jpeg();
        let pos = bytes.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        bytes[pos + 1] = 0xC3;
        let err = Decoder::new(&bytes).unwrap_err();
        assert!(err.is_not_implemented());
    }
}
