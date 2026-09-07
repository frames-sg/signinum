//! Error types for JPEG 2000 codec operations.

use core::fmt;
use j2k_types::J2kEncodeStageError;

/// The main error type for JPEG 2000 decoding operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeError {
    /// Errors related to JP2 file format and box parsing.
    Format(FormatError),
    /// Errors related to codestream markers.
    Marker(MarkerError),
    /// Errors related to tile processing.
    Tile(TileError),
    /// Errors related to image dimensions and validation.
    Validation(ValidationError),
    /// Errors related to decoding operations.
    Decoding(DecodingError),
    /// Errors related to color space and component handling.
    Color(ColorError),
    /// Simultaneously live decode/container allocations exceed the shared cap.
    AllocationTooLarge {
        /// Allocation family or phase being checked.
        what: &'static str,
        /// Requested live bytes, saturated on arithmetic overflow.
        requested: usize,
        /// Maximum permitted live bytes.
        cap: usize,
    },
    /// The host allocator rejected a checked, cap-valid decode allocation.
    HostAllocationFailed {
        /// Allocation being attempted.
        what: &'static str,
        /// Requested allocation bytes, saturated on arithmetic overflow.
        bytes: usize,
    },
}

/// Backend-neutral classification used by codec adapters.
///
/// This preserves the small amount of structured information that adapters
/// need without requiring each adapter to match the complete native error
/// hierarchy independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeErrorClass {
    /// Input ended before a required fixed-size read.
    InputTooShort {
        /// Required byte count.
        need: usize,
        /// Available byte count.
        have: usize,
    },
    /// Input ended while reading a named segment.
    InputTruncatedAt {
        /// Byte offset where truncation was detected.
        offset: usize,
        /// Stable segment label.
        segment: &'static str,
    },
    /// The codestream or container uses an unsupported feature.
    Unsupported {
        /// Stable user-facing feature label.
        what: &'static str,
    },
    /// All other native decoder failures.
    Backend,
}

/// Error returned by native JPEG 2000 and HTJ2K encode operations.
///
/// The variants deliberately keep resource failures distinct from malformed
/// requests, accelerator failures, and generated-codestream validation. This
/// lets facade and transcode callers preserve actionable failure categories
/// without parsing display strings.
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// Caller-provided samples, geometry, metadata, or options are invalid.
    InvalidInput {
        /// Stable description of the invalid condition.
        what: &'static str,
    },
    /// The requested encode feature or shape is not implemented.
    Unsupported {
        /// Stable description of the unsupported condition.
        what: &'static str,
    },
    /// Checked arithmetic overflowed while planning an encode phase.
    ArithmeticOverflow {
        /// Name of the value or phase whose size overflowed.
        what: &'static str,
    },
    /// Simultaneously live host allocations would exceed the shared cap.
    AllocationTooLarge {
        /// Name of the encode phase being checked.
        what: &'static str,
        /// Checked requested live host bytes at the rejected allocation boundary.
        requested: usize,
        /// Maximum permitted live host bytes.
        cap: usize,
    },
    /// The allocator could not reserve a checked, cap-valid host allocation.
    HostAllocationFailed {
        /// Name of the allocation that failed.
        what: &'static str,
        /// Requested allocation bytes, saturated on element-size overflow.
        bytes: usize,
    },
    /// An optional encode-stage accelerator accepted work but failed it.
    Accelerator {
        /// Encode-stage operation that failed.
        operation: &'static str,
        /// Structured stage failure with an optional concrete backend source.
        source: J2kEncodeStageError,
    },
    /// A generated codestream failed the requested validation contract.
    CodestreamValidation {
        /// Stable validation failure detail.
        detail: &'static str,
    },
    /// Internal encode state violated an invariant.
    InternalInvariant {
        /// Stable description of the violated invariant.
        what: &'static str,
    },
}

impl DecodeError {
    /// Classify this error for a facade or accelerator adapter.
    #[must_use]
    pub const fn classify(&self) -> DecodeErrorClass {
        match *self {
            Self::Format(FormatError::TooShort { need, have }) => {
                DecodeErrorClass::InputTooShort { need, have }
            }
            Self::Format(FormatError::TruncatedAt { offset, segment }) => {
                DecodeErrorClass::InputTruncatedAt { offset, segment }
            }
            Self::Format(FormatError::Unsupported) => DecodeErrorClass::Unsupported {
                what: "JP2 image format",
            },
            Self::Marker(MarkerError::Unsupported) => DecodeErrorClass::Unsupported {
                what: "JPEG 2000 marker",
            },
            Self::Decoding(DecodingError::DirectPlanUnsupported(reason)) => {
                DecodeErrorClass::Unsupported {
                    what: direct_plan_unsupported_what(reason),
                }
            }
            Self::Decoding(DecodingError::UnsupportedFeature(what)) => {
                DecodeErrorClass::Unsupported { what }
            }
            Self::Decoding(DecodingError::UnexpectedEof) => DecodeErrorClass::InputTruncatedAt {
                offset: 0,
                segment: "JPEG 2000 entropy data",
            },
            _ => DecodeErrorClass::Backend,
        }
    }
}

/// Errors related to JP2 file format and box parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FormatError {
    /// Input ended before a required fixed-size box read.
    TooShort {
        /// Required byte count.
        need: usize,
        /// Available byte count.
        have: usize,
    },
    /// Input ended while reading a named box segment.
    TruncatedAt {
        /// Byte offset where truncation was detected.
        offset: usize,
        /// Name of the segment being read.
        segment: &'static str,
    },
    /// Invalid JP2 signature.
    InvalidSignature,
    /// Invalid JP2 file type.
    InvalidFileType,
    /// Invalid or malformed JP2 box.
    InvalidBox,
    /// Required JP2 box is absent.
    MissingRequiredBox(&'static str),
    /// Missing codestream data.
    MissingCodestream,
    /// Unsupported JP2 image format.
    Unsupported,
}

/// Errors related to codestream markers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MarkerError {
    /// Invalid marker encountered.
    Invalid,
    /// Unsupported marker encountered.
    Unsupported,
    /// Expected a specific marker.
    Expected(&'static str),
    /// Missing a required marker.
    Missing(&'static str),
    /// Failed to read or parse a marker.
    ParseFailure(&'static str),
}

/// Errors related to tile processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TileError {
    /// Invalid image tile was encountered.
    Invalid,
    /// Invalid tile index in tile-part header.
    InvalidIndex,
    /// Invalid tile or image offsets.
    InvalidOffsets,
    /// PPT marker present when PPM marker exists in main header.
    PpmPptConflict,
}

/// Errors related to image dimensions and validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ValidationError {
    /// Invalid image dimensions.
    InvalidDimensions,
    /// Image dimensions exceed supported limits.
    ImageTooLarge,
    /// Image has too many channels.
    TooManyChannels,
    /// The SIZ tile grid implies more tiles than any conforming codestream can address.
    TooManyTiles,
    /// Invalid component metadata.
    InvalidComponentMetadata,
    /// Invalid JP2 channel definition metadata.
    InvalidChannelDefinition,
    /// Invalid progression order.
    InvalidProgressionOrder,
    /// Invalid transformation type.
    InvalidTransformation,
    /// Invalid quantization style.
    InvalidQuantizationStyle,
    /// Missing exponents for precinct sizes.
    MissingPrecinctExponents,
    /// Not enough exponents provided in header.
    InsufficientExponents,
    /// Missing exponent step size.
    MissingStepSize,
    /// Invalid quantization exponents.
    InvalidExponents,
}

/// Errors related to decoding operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodingError {
    /// An error occurred while decoding a code-block.
    CodeBlockDecodeFailure,
    /// A backend-specific code-block decode failure with user-visible context.
    CodeBlockDecodeFailureWithContext(&'static str),
    /// A direct-plan builder rejected an unsupported image or codestream shape.
    DirectPlanUnsupported(DirectPlanUnsupportedReason),
    /// The codestream uses a feature that this decoder does not implement yet.
    UnsupportedFeature(&'static str),
    /// Number of bitplanes in a code-block is too large.
    TooManyBitplanes,
    /// A code-block contains too many coding passes.
    TooManyCodingPasses,
    /// Invalid number of bitplanes in a code-block.
    InvalidBitplaneCount,
    /// A precinct was invalid.
    InvalidPrecinct,
    /// A packet header or its declared body could not be parsed completely.
    PacketParseFailure(&'static str),
    /// A progression iterator ver invalid.
    InvalidProgressionIterator,
    /// Unexpected end of data.
    UnexpectedEof,
    /// Caller-provided output buffer is too small for the decoded image.
    OutputBufferTooSmall,
    /// A bounded host decode workspace could not be allocated.
    HostAllocationFailed,
}

/// Errors related to color space and component handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ColorError {
    /// Multi-component transform failed.
    Mct,
    /// Failed to resolve palette indices.
    PaletteResolutionFailed,
    /// Failed to convert from sYCC to RGB.
    SyccConversionFailed,
    /// Failed to convert from LAB to RGB.
    LabConversionFailed,
}

/// Structured reasons why a direct JPEG 2000 device plan cannot be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DirectPlanUnsupportedReason {
    /// Grayscale direct plans require grayscale images without alpha.
    GrayscaleImageWithoutAlpha,
    /// Grayscale direct plans require a single-tile codestream.
    GrayscaleSingleTileCodestream,
    /// Grayscale direct plans require a single-component codestream.
    GrayscaleSingleComponentCodestream,
    /// Color direct plans require RGB images without alpha.
    ColorRgbImageWithoutAlpha,
    /// Color direct plans require a single-tile codestream.
    ColorSingleTileCodestream,
    /// Color direct plans require three RGB components.
    ColorThreeComponentRgbCodestream,
    /// RGBA direct plans require an RGB image with explicit alpha.
    RgbaRgbImageWithAlpha,
    /// RGBA direct plans require four components in semantic R, G, B, A order.
    RgbaFourComponentRgbCodestream,
    /// A direct component plan index did not exist.
    ComponentIndexOutOfRange,
    /// Direct component plans require unit-sampled components.
    ComponentUnitSampled,
    /// Component-grid plans require a full unsigned origin-zero image without MCT.
    ComponentGridFullImage,
    /// A direct component decomposition index did not exist.
    ComponentDecompositionIndexOutOfRange,
    /// Direct device plans do not yet represent classic and HT code-blocks in one sub-band.
    MixedCodeBlockCoding,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format(e) => write!(f, "{e}"),
            Self::Marker(e) => write!(f, "{e}"),
            Self::Tile(e) => write!(f, "{e}"),
            Self::Validation(e) => write!(f, "{e}"),
            Self::Decoding(e) => write!(f, "{e}"),
            Self::Color(e) => write!(f, "{e}"),
            Self::AllocationTooLarge {
                what,
                requested,
                cap,
            } => write!(
                f,
                "{what} requires {requested} live host bytes, exceeding the {cap}-byte cap"
            ),
            Self::HostAllocationFailed { what, bytes } => {
                write!(
                    f,
                    "host allocation failed for {bytes} bytes while allocating {what}"
                )
            }
        }
    }
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { what } => write!(f, "invalid encode input: {what}"),
            Self::Unsupported { what } => write!(f, "unsupported encode request: {what}"),
            Self::ArithmeticOverflow { what } => {
                write!(f, "encode size overflow while planning {what}")
            }
            Self::AllocationTooLarge {
                what,
                requested,
                cap,
            } => write!(
                f,
                "{what} requires {requested} live host bytes, exceeding the {cap}-byte cap"
            ),
            Self::HostAllocationFailed { what, bytes } => {
                write!(
                    f,
                    "host allocation failed for {bytes} bytes while allocating {what}"
                )
            }
            Self::Accelerator { operation, source } => {
                write!(f, "encode accelerator failed during {operation}: {source}")
            }
            Self::CodestreamValidation { detail } => {
                write!(f, "generated codestream validation failed: {detail}")
            }
            Self::InternalInvariant { what } => {
                write!(f, "native encode invariant failed: {what}")
            }
        }
    }
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort { need, have } => {
                write!(f, "input too short: need {need} bytes, have {have}")
            }
            Self::TruncatedAt { offset, segment } => {
                write!(
                    f,
                    "input truncated at offset {offset} while reading {segment}"
                )
            }
            Self::InvalidSignature => write!(f, "invalid JP2 signature"),
            Self::InvalidFileType => write!(f, "invalid JP2 file type"),
            Self::InvalidBox => write!(f, "invalid JP2 box"),
            Self::MissingRequiredBox(box_type) => write!(f, "missing required JP2 box {box_type}"),
            Self::MissingCodestream => write!(f, "missing codestream data"),
            Self::Unsupported => write!(f, "unsupported JP2 image"),
        }
    }
}

impl fmt::Display for MarkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid => write!(f, "invalid marker"),
            Self::Unsupported => write!(f, "unsupported marker"),
            Self::Expected(marker) => write!(f, "expected {marker} marker"),
            Self::Missing(marker) => write!(f, "missing {marker} marker"),
            Self::ParseFailure(marker) => write!(f, "failed to parse {marker} marker"),
        }
    }
}

impl fmt::Display for TileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid => write!(f, "image contains no tiles"),
            Self::InvalidIndex => write!(f, "invalid tile index in tile-part header"),
            Self::InvalidOffsets => write!(f, "invalid tile offsets"),
            Self::PpmPptConflict => {
                write!(
                    f,
                    "PPT marker present when PPM marker exists in main header"
                )
            }
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions => write!(f, "invalid image dimensions"),
            Self::ImageTooLarge => write!(f, "image is too large"),
            Self::TooManyChannels => write!(f, "image has too many channels"),
            Self::TooManyTiles => write!(f, "image has too many tiles"),
            Self::InvalidComponentMetadata => write!(f, "invalid component metadata"),
            Self::InvalidChannelDefinition => write!(f, "invalid channel definition"),
            Self::InvalidProgressionOrder => write!(f, "invalid progression order"),
            Self::InvalidTransformation => write!(f, "invalid transformation type"),
            Self::InvalidQuantizationStyle => write!(f, "invalid quantization style"),
            Self::MissingPrecinctExponents => {
                write!(f, "missing exponents for precinct sizes")
            }
            Self::InsufficientExponents => {
                write!(f, "not enough exponents provided in header")
            }
            Self::MissingStepSize => write!(f, "missing exponent step size"),
            Self::InvalidExponents => write!(f, "invalid quantization exponents"),
        }
    }
}

impl fmt::Display for DecodingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CodeBlockDecodeFailure => write!(f, "failed to decode code-block"),
            Self::CodeBlockDecodeFailureWithContext(context) => {
                write!(f, "failed to decode code-block: {context}")
            }
            Self::DirectPlanUnsupported(reason) => {
                write!(f, "unsupported decoding feature: {reason}")
            }
            Self::UnsupportedFeature(feature) => {
                write!(f, "unsupported decoding feature: {feature}")
            }
            Self::TooManyBitplanes => write!(f, "number of bitplanes is too large"),
            Self::TooManyCodingPasses => {
                write!(f, "code-block contains too many coding passes")
            }
            Self::InvalidBitplaneCount => write!(f, "invalid number of bitplanes"),
            Self::InvalidPrecinct => write!(f, "a precinct was invalid"),
            Self::PacketParseFailure(context) => {
                write!(f, "failed to parse JPEG 2000 packet data: {context}")
            }
            Self::InvalidProgressionIterator => {
                write!(f, "a progression iterator was invalid")
            }
            Self::UnexpectedEof => write!(f, "unexpected end of data"),
            Self::OutputBufferTooSmall => write!(f, "output buffer is too small"),
            Self::HostAllocationFailed => write!(f, "host decode workspace allocation failed"),
        }
    }
}

impl fmt::Display for DirectPlanUnsupportedReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(direct_plan_unsupported_what(*self))
    }
}

const fn direct_plan_unsupported_what(reason: DirectPlanUnsupportedReason) -> &'static str {
    match reason {
        DirectPlanUnsupportedReason::GrayscaleImageWithoutAlpha => {
            "direct grayscale plan only supports grayscale images without alpha"
        }
        DirectPlanUnsupportedReason::GrayscaleSingleTileCodestream => {
            "direct grayscale plan only supports single-tile codestreams"
        }
        DirectPlanUnsupportedReason::GrayscaleSingleComponentCodestream => {
            "direct grayscale plan only supports single-component codestreams"
        }
        DirectPlanUnsupportedReason::ColorRgbImageWithoutAlpha => {
            "direct color plan only supports RGB images without alpha"
        }
        DirectPlanUnsupportedReason::ColorSingleTileCodestream => {
            "direct color plan only supports single-tile codestreams"
        }
        DirectPlanUnsupportedReason::ColorThreeComponentRgbCodestream => {
            "direct color plan only supports three-component RGB codestreams"
        }
        DirectPlanUnsupportedReason::RgbaRgbImageWithAlpha => {
            "direct RGBA plan only supports RGB images with explicit alpha"
        }
        DirectPlanUnsupportedReason::RgbaFourComponentRgbCodestream => {
            "direct RGBA plan only supports four-component RGB-alpha codestreams"
        }
        DirectPlanUnsupportedReason::ComponentIndexOutOfRange => {
            "direct component plan index is out of range"
        }
        DirectPlanUnsupportedReason::ComponentGridFullImage => {
            "component-grid plans require a full unsigned origin-zero image without MCT"
        }
        DirectPlanUnsupportedReason::ComponentUnitSampled => {
            "direct component plan only supports unit-sampled components"
        }
        DirectPlanUnsupportedReason::ComponentDecompositionIndexOutOfRange => {
            "direct component decomposition index is out of range"
        }
        DirectPlanUnsupportedReason::MixedCodeBlockCoding => {
            "direct device plan does not support mixed classic and HT code-block coding in one sub-band"
        }
    }
}

impl fmt::Display for ColorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mct => write!(f, "multi-component transform failed"),
            Self::PaletteResolutionFailed => write!(f, "failed to resolve palette indices"),
            Self::SyccConversionFailed => write!(f, "failed to convert from sYCC to RGB"),
            Self::LabConversionFailed => write!(f, "failed to convert from LAB to RGB"),
        }
    }
}

impl core::error::Error for DecodeError {}
impl core::error::Error for EncodeError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Accelerator { source, .. } => Some(source),
            _ => None,
        }
    }
}
impl core::error::Error for FormatError {}
impl core::error::Error for MarkerError {}
impl core::error::Error for TileError {}
impl core::error::Error for ValidationError {}
impl core::error::Error for DecodingError {}
impl core::error::Error for DirectPlanUnsupportedReason {}
impl core::error::Error for ColorError {}

impl From<FormatError> for DecodeError {
    fn from(e: FormatError) -> Self {
        Self::Format(e)
    }
}

impl From<MarkerError> for DecodeError {
    fn from(e: MarkerError) -> Self {
        Self::Marker(e)
    }
}

impl From<TileError> for DecodeError {
    fn from(e: TileError) -> Self {
        Self::Tile(e)
    }
}

impl From<ValidationError> for DecodeError {
    fn from(e: ValidationError) -> Self {
        Self::Validation(e)
    }
}

impl From<j2k_types::DecodePlanAllocationError> for DecodeError {
    fn from(_: j2k_types::DecodePlanAllocationError) -> Self {
        Self::Validation(ValidationError::ImageTooLarge)
    }
}

impl From<DecodingError> for DecodeError {
    fn from(e: DecodingError) -> Self {
        Self::Decoding(e)
    }
}

impl From<ColorError> for DecodeError {
    fn from(e: ColorError) -> Self {
        Self::Color(e)
    }
}

/// Result type for JPEG 2000 decoding operations.
pub type Result<T> = core::result::Result<T, DecodeError>;

/// Result type for JPEG 2000 and HTJ2K encoding operations.
pub type EncodeResult<T> = core::result::Result<T, EncodeError>;

macro_rules! bail {
    ($err:expr) => {
        return Err($err.into())
    };
}

macro_rules! err {
    ($err:expr) => {
        Err($err.into())
    };
}

pub(crate) use bail;
pub(crate) use err;

#[cfg(test)]
mod classification_tests {
    use alloc::string::ToString;

    use super::{
        DecodeError, DecodeErrorClass, DecodingError, DirectPlanUnsupportedReason, EncodeError,
        FormatError, MarkerError, ValidationError,
    };

    #[test]
    fn facade_classification_preserves_structured_input_and_support_details() {
        let cases = [
            (
                DecodeError::Format(FormatError::TooShort { need: 9, have: 3 }),
                DecodeErrorClass::InputTooShort { need: 9, have: 3 },
            ),
            (
                DecodeError::Format(FormatError::TruncatedAt {
                    offset: 17,
                    segment: "SIZ",
                }),
                DecodeErrorClass::InputTruncatedAt {
                    offset: 17,
                    segment: "SIZ",
                },
            ),
            (
                DecodeError::Format(FormatError::Unsupported),
                DecodeErrorClass::Unsupported {
                    what: "JP2 image format",
                },
            ),
            (
                DecodeError::Marker(MarkerError::Unsupported),
                DecodeErrorClass::Unsupported {
                    what: "JPEG 2000 marker",
                },
            ),
            (
                DecodeError::Decoding(DecodingError::UnsupportedFeature("packet marker")),
                DecodeErrorClass::Unsupported {
                    what: "packet marker",
                },
            ),
            (
                DecodeError::Decoding(DecodingError::UnexpectedEof),
                DecodeErrorClass::InputTruncatedAt {
                    offset: 0,
                    segment: "JPEG 2000 entropy data",
                },
            ),
            (
                DecodeError::Validation(ValidationError::InvalidDimensions),
                DecodeErrorClass::Backend,
            ),
        ];

        for (error, expected) in cases {
            assert_eq!(error.classify(), expected, "{error}");
        }
    }

    #[test]
    fn direct_plan_classification_and_display_share_the_same_label() {
        let reason = DirectPlanUnsupportedReason::ColorThreeComponentRgbCodestream;
        let error = DecodeError::Decoding(DecodingError::DirectPlanUnsupported(reason));
        let DecodeErrorClass::Unsupported { what } = error.classify() else {
            panic!("direct-plan errors must classify as unsupported");
        };

        assert_eq!(what, reason.to_string());
    }

    #[test]
    fn encode_resource_errors_keep_cap_and_allocator_failures_distinct() {
        let cap_error = EncodeError::AllocationTooLarge {
            what: "Tier-2 packet assembly",
            requested: 513,
            cap: 512,
        };
        let allocation_error = EncodeError::HostAllocationFailed {
            what: "Tier-2 packet body",
            bytes: 511,
        };

        assert!(cap_error.to_string().contains("513"));
        assert!(cap_error.to_string().contains("512-byte cap"));
        assert!(allocation_error.to_string().contains("511"));
        assert_ne!(cap_error, allocation_error);
    }
}
