// SPDX-License-Identifier: Apache-2.0

use crate::{Colorspace, Info, TileLayout};

/// Compressed syntax carried by a source frame or accepted by a destination.
///
/// The enum intentionally names codec profiles rather than container-specific
/// UIDs. Container integrations can map these variants to their local transfer
/// syntax identifiers and keep that policy outside the codec crates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CompressedTransferSyntax {
    /// Baseline 8-bit JPEG interchange format.
    JpegBaseline8,
    /// Sequential JPEG beyond the baseline profile.
    JpegExtendedSequential,
    /// Classic JPEG 2000 codestream using reversible coding.
    Jpeg2000Lossless,
    /// Classic JPEG 2000 codestream using irreversible coding.
    Jpeg2000Lossy,
    /// High-throughput JPEG 2000 codestream using reversible coding.
    HtJpeg2000Lossless,
    /// High-throughput JPEG 2000 codestream using irreversible coding.
    HtJpeg2000Lossy,
}

impl CompressedTransferSyntax {
    /// True when the syntax profile is lossless.
    #[must_use]
    pub const fn is_lossless(self) -> bool {
        matches!(self, Self::Jpeg2000Lossless | Self::HtJpeg2000Lossless)
    }

    /// True when the syntax belongs to the JPEG 2000 family.
    #[must_use]
    pub const fn is_jpeg2000_family(self) -> bool {
        matches!(
            self,
            Self::Jpeg2000Lossless
                | Self::Jpeg2000Lossy
                | Self::HtJpeg2000Lossless
                | Self::HtJpeg2000Lossy
        )
    }
}

/// Encapsulation shape of the compressed bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CompressedPayloadKind {
    /// Complete JPEG interchange byte stream.
    JpegInterchange,
    /// Raw JPEG 2000 / HTJ2K codestream bytes.
    Jpeg2000Codestream,
    /// JP2 file-format wrapper around a JPEG 2000 codestream.
    Jp2File,
}

/// A borrowed compressed frame/tile that may be copied unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassthroughCandidate<'a> {
    bytes: &'a [u8],
    transfer_syntax: CompressedTransferSyntax,
    payload_kind: CompressedPayloadKind,
    info: Info,
}

impl<'a> PassthroughCandidate<'a> {
    /// Construct a candidate from already-inspected compressed bytes.
    #[must_use]
    pub const fn new(
        bytes: &'a [u8],
        transfer_syntax: CompressedTransferSyntax,
        payload_kind: CompressedPayloadKind,
        info: Info,
    ) -> Self {
        Self {
            bytes,
            transfer_syntax,
            payload_kind,
            info,
        }
    }

    /// Original compressed bytes. A successful passthrough decision returns
    /// this exact slice.
    #[must_use]
    pub const fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Source compressed syntax.
    #[must_use]
    pub const fn transfer_syntax(&self) -> CompressedTransferSyntax {
        self.transfer_syntax
    }

    /// Source payload/container shape.
    #[must_use]
    pub const fn payload_kind(&self) -> CompressedPayloadKind {
        self.payload_kind
    }

    /// Header metadata inspected from the compressed payload.
    #[must_use]
    pub const fn info(&self) -> &Info {
        &self.info
    }

    /// Evaluate whether this candidate can be copied unchanged into a
    /// destination with the supplied requirements.
    #[must_use]
    pub fn evaluate(&self, requirements: &PassthroughRequirements) -> PassthroughDecision<'a> {
        match self.copy_bytes_if_eligible(requirements) {
            Ok(bytes) => PassthroughDecision::Copy { bytes },
            Err(reason) => PassthroughDecision::Transcode { reason },
        }
    }

    /// Return the original compressed bytes only when passthrough is legal.
    pub fn copy_bytes_if_eligible(
        &self,
        requirements: &PassthroughRequirements,
    ) -> Result<&'a [u8], PassthroughRejectReason> {
        if self.bytes.is_empty() {
            return Err(PassthroughRejectReason::EmptyPayload);
        }
        if self.transfer_syntax != requirements.transfer_syntax {
            return Err(PassthroughRejectReason::TransferSyntaxMismatch {
                source: self.transfer_syntax,
                destination: requirements.transfer_syntax,
            });
        }
        if self.payload_kind != requirements.payload_kind {
            return Err(PassthroughRejectReason::PayloadKindMismatch {
                source: self.payload_kind,
                destination: requirements.payload_kind,
            });
        }
        if let Some(destination) = requirements.dimensions {
            if self.info.dimensions != destination {
                return Err(PassthroughRejectReason::DimensionsMismatch {
                    source: self.info.dimensions,
                    destination,
                });
            }
        }
        if let Some(destination) = requirements.components {
            if self.info.components != destination {
                return Err(PassthroughRejectReason::ComponentsMismatch {
                    source: self.info.components,
                    destination,
                });
            }
        }
        if let Some(destination) = requirements.bit_depth {
            if self.info.bit_depth != destination {
                return Err(PassthroughRejectReason::BitDepthMismatch {
                    source: self.info.bit_depth,
                    destination,
                });
            }
        }
        if let Some(destination) = requirements.colorspace {
            if self.info.colorspace != destination {
                return Err(PassthroughRejectReason::ColorspaceMismatch {
                    source: self.info.colorspace,
                    destination,
                });
            }
        }
        if let Some(destination) = requirements.tile_layout {
            if self.info.tile_layout != Some(destination) {
                return Err(PassthroughRejectReason::TileLayoutMismatch {
                    source: self.info.tile_layout,
                    destination,
                });
            }
        }

        Ok(self.bytes)
    }
}

/// Destination requirements for copying compressed bytes unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PassthroughRequirements {
    pub transfer_syntax: CompressedTransferSyntax,
    pub payload_kind: CompressedPayloadKind,
    pub dimensions: Option<(u32, u32)>,
    pub components: Option<u8>,
    pub bit_depth: Option<u8>,
    pub colorspace: Option<Colorspace>,
    pub tile_layout: Option<TileLayout>,
}

impl PassthroughRequirements {
    /// Start a requirements set with the mandatory syntax and payload shape.
    #[must_use]
    pub const fn new(
        transfer_syntax: CompressedTransferSyntax,
        payload_kind: CompressedPayloadKind,
    ) -> Self {
        Self {
            transfer_syntax,
            payload_kind,
            dimensions: None,
            components: None,
            bit_depth: None,
            colorspace: None,
            tile_layout: None,
        }
    }

    /// Require exact frame/tile dimensions.
    #[must_use]
    pub const fn with_dimensions(mut self, dimensions: (u32, u32)) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Require an exact component count.
    #[must_use]
    pub const fn with_components(mut self, components: u8) -> Self {
        self.components = Some(components);
        self
    }

    /// Require an exact bit depth.
    #[must_use]
    pub const fn with_bit_depth(mut self, bit_depth: u8) -> Self {
        self.bit_depth = Some(bit_depth);
        self
    }

    /// Require an exact colorspace.
    #[must_use]
    pub const fn with_colorspace(mut self, colorspace: Colorspace) -> Self {
        self.colorspace = Some(colorspace);
        self
    }

    /// Require an exact tile layout.
    #[must_use]
    pub const fn with_tile_layout(mut self, tile_layout: TileLayout) -> Self {
        self.tile_layout = Some(tile_layout);
        self
    }
}

/// Result of a passthrough eligibility check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassthroughDecision<'a> {
    /// Copy these compressed bytes unchanged.
    Copy { bytes: &'a [u8] },
    /// Decode/transcode instead, for the stated reason.
    Transcode { reason: PassthroughRejectReason },
}

/// First reason a compressed payload was rejected for byte-preserving copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PassthroughRejectReason {
    EmptyPayload,
    TransferSyntaxMismatch {
        source: CompressedTransferSyntax,
        destination: CompressedTransferSyntax,
    },
    PayloadKindMismatch {
        source: CompressedPayloadKind,
        destination: CompressedPayloadKind,
    },
    DimensionsMismatch {
        source: (u32, u32),
        destination: (u32, u32),
    },
    ComponentsMismatch {
        source: u8,
        destination: u8,
    },
    BitDepthMismatch {
        source: u8,
        destination: u8,
    },
    ColorspaceMismatch {
        source: Colorspace,
        destination: Colorspace,
    },
    TileLayoutMismatch {
        source: Option<TileLayout>,
        destination: TileLayout,
    },
}
