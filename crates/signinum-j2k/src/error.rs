// SPDX-License-Identifier: Apache-2.0

use signinum_core::{BufferError, CodecError, InputError, NotImplemented, Unsupported};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum J2kError {
    #[error(transparent)]
    Buffer(#[from] BufferError),

    #[error(transparent)]
    Input(#[from] InputError),

    #[error(transparent)]
    NotImplemented(#[from] NotImplemented),

    #[error(transparent)]
    Unsupported(#[from] Unsupported),

    #[error("backend decode failed: {0}")]
    Backend(String),

    #[error("region ({x},{y} {w}x{h}) is outside image bounds {image_w}x{image_h}")]
    InvalidRegion {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        image_w: u32,
        image_h: u32,
    },

    #[error("invalid JP2 box at offset {offset}: {what}")]
    InvalidBox { offset: usize, what: &'static str },

    #[error("missing required JP2 box {box_type}")]
    MissingRequiredBox { box_type: &'static str },

    #[error("invalid codestream marker FF{marker:02X} at offset {offset}")]
    InvalidMarker { offset: usize, marker: u8 },

    #[error("missing required codestream marker {marker}")]
    MissingRequiredMarker { marker: &'static str },

    #[error("invalid SIZ segment: {what}")]
    InvalidSiz { what: &'static str },

    #[error("invalid COD segment: {what}")]
    InvalidCod { what: &'static str },

    #[error("dimension overflow: {width}x{height}")]
    DimensionOverflow { width: u32, height: u32 },
}

impl CodecError for J2kError {
    fn is_truncated(&self) -> bool {
        matches!(
            self,
            Self::Input(InputError::TooShort { .. } | InputError::TruncatedAt { .. })
        )
    }

    fn is_not_implemented(&self) -> bool {
        matches!(self, Self::NotImplemented(_))
    }

    fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported(_))
    }

    fn is_buffer_error(&self) -> bool {
        matches!(self, Self::Buffer(_))
    }
}
