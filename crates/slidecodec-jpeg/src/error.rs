// SPDX-License-Identifier: Apache-2.0

//! Typed error and warning taxonomy. See spec Section 6.

use crate::info::{ColorSpace, Rect, SofKind};

/// A category of JPEG marker. Carried in [`JpegError::UnexpectedMarker`] and
/// related variants so callers can branch on marker class without parsing the
/// raw byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerKind {
    /// Start of image (`FFD8`).
    Soi,
    /// Start of frame (any of `FFC0..=FFC3`).
    Sof,
    /// Define quantization table (`FFDB`).
    Dqt,
    /// Define Huffman table (`FFC4`).
    Dht,
    /// Define restart interval (`FFDD`).
    Dri,
    /// Start of scan (`FFDA`).
    Sos,
    /// End of image (`FFD9`).
    Eoi,
    /// Adobe APP14 (`FFEE`).
    App14,
    /// Any other marker, raw byte preserved.
    Other(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnsupportedReason {
    ArithmeticCoding,
    Hierarchical,
    ArithmeticAndHierarchical,
    DifferentialBaseline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HuffmanFailure {
    CodeOverflow,
    InvalidSymbol,
    TableExhausted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuilderConflictReason {
    NoInput,
    InputAndScanFragments,
    ScanFragmentsEmpty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableKind {
    Quant,
    HuffmanAc,
    HuffmanDc,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum JpegError {
    #[error("JPEG truncated at offset {offset}: expected {expected} more bytes")]
    Truncated { offset: usize, expected: usize },

    #[error("invalid marker FF{marker:02X} at offset {offset}")]
    InvalidMarker { offset: usize, marker: u8 },

    #[error("expected {expected:?}, found FF{found:02X} at offset {offset}")]
    UnexpectedMarker {
        offset: usize,
        expected: MarkerKind,
        found: u8,
    },

    #[error("missing required marker {marker:?}")]
    MissingMarker { marker: MarkerKind },

    #[error("duplicate {marker:?} at offset {offset}")]
    DuplicateMarker { offset: usize, marker: MarkerKind },

    #[error("invalid length {length} for marker FF{marker:02X} at offset {offset}")]
    InvalidSegmentLength {
        offset: usize,
        marker: u8,
        length: u16,
    },

    /// Unsupported SOF variant. Carries the raw marker byte (e.g. `0xC9` for
    /// arithmetic extended-sequential) so callers routing to a fallback
    /// decoder can distinguish FFC5 from FFC9 without relying on `reason`.
    #[error("unsupported SOF marker FF{marker:02X} ({reason:?})")]
    UnsupportedSof {
        marker: u8,
        reason: UnsupportedReason,
    },

    #[error("unsupported component count: {count}")]
    UnsupportedComponentCount { count: u8 },

    #[error("unsupported color space for decode: {color_space:?}")]
    UnsupportedColorSpace { color_space: ColorSpace },

    #[error("unsupported bit depth: {depth}")]
    UnsupportedBitDepth { depth: u8 },

    #[error("unsupported lossless predictor: {predictor}")]
    UnsupportedPredictor { predictor: u8 },

    #[error("zero dimension in SOF: {width}×{height}")]
    ZeroDimension { width: u16, height: u16 },

    #[error("dimension overflow: {width}×{height} exceeds 65500")]
    DimensionOverflow { width: u32, height: u32 },

    #[error("invalid sampling ({h}×{v}) for component {component}")]
    InvalidSampling { component: u8, h: u8, v: u8 },

    #[error("missing quantization table {table_id} for component {component}")]
    MissingQuantTable { component: u8, table_id: u8 },

    #[error("missing Huffman table class={class} id={id} for component {component}")]
    MissingHuffmanTable { component: u8, class: u8, id: u8 },

    #[error(
        "invalid sequential scan parameters at offset {offset}: Ss={ss} Se={se} Ah={ah} Al={al}"
    )]
    InvalidScanParameters {
        offset: usize,
        ss: u8,
        se: u8,
        ah: u8,
        al: u8,
    },

    #[error("Huffman decode failed at MCU {mcu}: {reason:?}")]
    HuffmanDecode { mcu: u32, reason: HuffmanFailure },

    #[error("restart mismatch at offset {offset}: expected RST{expected}, found FF{found:02X}")]
    RestartMismatch {
        offset: usize,
        expected: u8,
        found: u8,
    },

    #[error("unexpected EOI at MCU {mcu_at}/{mcu_total}")]
    UnexpectedEoi { mcu_at: u32, mcu_total: u32 },

    #[error("coefficient overflow at MCU {mcu}, component {component}")]
    CoefficientOverflow { mcu: u32, component: u8 },

    #[error("decode size {requested} bytes exceeds cap {cap} bytes")]
    MemoryCapExceeded { requested: usize, cap: usize },

    #[error("output buffer too small: need {required} bytes, got {provided}")]
    OutputBufferTooSmall { required: usize, provided: usize },

    #[error("stride {stride} smaller than row width {row}")]
    InvalidStride { stride: usize, row: usize },

    #[error("rect {rect:?} out of image bounds ({width}×{height})")]
    RectOutOfBounds { rect: Rect, width: u32, height: u32 },

    #[error("downscale not supported for {sof:?} streams")]
    DownscaleUnsupported { sof: SofKind },

    #[error("scan fragments overlap at MCU {mcu}")]
    ScanFragmentsOverlap { mcu: u32 },

    #[error("builder input configuration conflict: {reason:?}")]
    BuilderConflict { reason: BuilderConflictReason },

    /// Transient pre-1.0 gap: the SOF is parseable and will eventually be
    /// supported by the decoder, but the current release does not implement
    /// it yet. M3 removes this variant by implementing Extended12, Progressive,
    /// and Lossless. Distinct from `UnsupportedSof` because callers routing
    /// to a fallback decoder on `is_unsupported()` should NOT reroute streams
    /// that a newer version of slidecodec will decode natively.
    #[error("decode not yet implemented for {sof:?} — see CHANGELOG for milestone")]
    NotImplemented { sof: SofKind },
}

impl JpegError {
    /// True if the error is recoverable by routing to a different decoder —
    /// any `Unsupported*` variant.
    pub fn is_unsupported(&self) -> bool {
        matches!(
            self,
            Self::UnsupportedSof { .. }
                | Self::UnsupportedComponentCount { .. }
                | Self::UnsupportedColorSpace { .. }
                | Self::UnsupportedBitDepth { .. }
                | Self::UnsupportedPredictor { .. }
        )
    }

    /// True if the input was truncated — caller may retry with more bytes.
    pub fn is_truncated(&self) -> bool {
        matches!(self, Self::Truncated { .. } | Self::UnexpectedEoi { .. })
    }

    /// True if the error indicates caller misuse, not a decode failure.
    pub fn is_api_misuse(&self) -> bool {
        matches!(
            self,
            Self::OutputBufferTooSmall { .. }
                | Self::InvalidStride { .. }
                | Self::RectOutOfBounds { .. }
                | Self::DownscaleUnsupported { .. }
                | Self::ScanFragmentsOverlap { .. }
                | Self::BuilderConflict { .. }
        )
    }

    /// True if the error is a transient "not yet implemented" gap — the stream
    /// is valid and will decode on a future slidecodec release, so callers
    /// should *not* reroute to a different decoder permanently. See
    /// [`Self::is_unsupported`] for errors that are permanent routing decisions.
    pub fn is_not_implemented(&self) -> bool {
        matches!(self, Self::NotImplemented { .. })
    }

    /// Byte offset where the error was detected in the input stream, if any.
    pub fn offset(&self) -> Option<usize> {
        match self {
            Self::Truncated { offset, .. }
            | Self::InvalidMarker { offset, .. }
            | Self::UnexpectedMarker { offset, .. }
            | Self::DuplicateMarker { offset, .. }
            | Self::InvalidSegmentLength { offset, .. }
            | Self::InvalidScanParameters { offset, .. }
            | Self::RestartMismatch { offset, .. } => Some(*offset),
            _ => None,
        }
    }
}

/// Non-fatal notices emitted during decode. See spec Section 6.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Warning {
    MissingEoi,
    SofDimensionsPatched { from: (u16, u16), to: (u16, u16) },
    NonstandardTables,
    AdobeApp14Ambiguous { raw_transform: u8 },
    IccProfileIgnored { size: usize },
    UnknownAppMarker { marker: u8, size: usize },
    RestartRecovered { offset: usize },
    PrecisionClamped { from_bits: u8, to_bits: u8 },
    UnknownColorProfile,
    TableCacheMismatch { which: TableKind, id: u8 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::info::ColorSpace;

    #[test]
    fn unsupported_predicate_matches_only_unsupported_variants() {
        assert!(JpegError::UnsupportedSof {
            marker: 0xC9,
            reason: UnsupportedReason::ArithmeticCoding,
        }
        .is_unsupported());
        assert!(JpegError::UnsupportedColorSpace {
            color_space: ColorSpace::Cmyk,
        }
        .is_unsupported());
        assert!(JpegError::UnsupportedBitDepth { depth: 16 }.is_unsupported());
        assert!(!JpegError::Truncated {
            offset: 0,
            expected: 1
        }
        .is_unsupported());
    }

    #[test]
    fn truncated_predicate_covers_truncation_and_unexpected_eoi() {
        assert!(JpegError::Truncated {
            offset: 10,
            expected: 5
        }
        .is_truncated());
        assert!(JpegError::UnexpectedEoi {
            mcu_at: 3,
            mcu_total: 10
        }
        .is_truncated());
        assert!(!JpegError::InvalidMarker {
            offset: 4,
            marker: 0xFF
        }
        .is_truncated());
    }

    #[test]
    fn api_misuse_predicate_covers_caller_bugs() {
        assert!(JpegError::OutputBufferTooSmall {
            required: 100,
            provided: 64
        }
        .is_api_misuse());
        assert!(JpegError::InvalidStride { stride: 2, row: 8 }.is_api_misuse());
        assert!(JpegError::BuilderConflict {
            reason: BuilderConflictReason::NoInput
        }
        .is_api_misuse());
        assert!(!JpegError::Truncated {
            offset: 0,
            expected: 1
        }
        .is_api_misuse());
    }

    #[test]
    fn offset_returns_some_for_byte_positioned_errors() {
        assert_eq!(
            JpegError::InvalidMarker {
                offset: 42,
                marker: 0xBA
            }
            .offset(),
            Some(42),
        );
        assert_eq!(JpegError::UnsupportedBitDepth { depth: 16 }.offset(), None,);
    }

    #[test]
    fn not_implemented_predicate_distinguishes_from_unsupported() {
        let not_impl = JpegError::NotImplemented {
            sof: SofKind::Progressive8,
        };
        assert!(not_impl.is_not_implemented());
        assert!(
            !not_impl.is_unsupported(),
            "NotImplemented is a transient M1b/M2 gap — callers routing on is_unsupported() must NOT \
             reroute these streams, because M3 adds real support"
        );
        assert!(!not_impl.is_truncated());
        assert!(!not_impl.is_api_misuse());

        let unsupported = JpegError::UnsupportedSof {
            marker: 0xC9,
            reason: UnsupportedReason::ArithmeticCoding,
        };
        assert!(!unsupported.is_not_implemented());
        assert!(unsupported.is_unsupported());
    }
}
