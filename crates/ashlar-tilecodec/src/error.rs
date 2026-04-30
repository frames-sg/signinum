// SPDX-License-Identifier: Apache-2.0

use ashlar_core::{BufferError, CodecError, InputError, Unsupported};

#[derive(Debug, thiserror::Error)]
pub enum TileCodecError {
    #[error(transparent)]
    Buffer(#[from] BufferError),
    #[error(transparent)]
    Input(#[from] InputError),
    #[error(transparent)]
    Unsupported(#[from] Unsupported),
    #[error("{0}")]
    Backend(String),
}

impl CodecError for TileCodecError {
    fn is_truncated(&self) -> bool {
        matches!(self, Self::Input(_))
    }

    fn is_not_implemented(&self) -> bool {
        false
    }

    fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported(_))
    }

    fn is_buffer_error(&self) -> bool {
        matches!(self, Self::Buffer(_))
    }
}
