// SPDX-License-Identifier: Apache-2.0

use crate::{pixel::PixelFormat, sample::SampleType};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BufferError {
    #[error("output buffer too small: required {required} bytes, have {have}")]
    OutputTooSmall { required: usize, have: usize },
    #[error("input buffer too small: required {required} bytes, have {have}")]
    InputTooSmall { required: usize, have: usize },
    #[error("buffer size overflow while computing {what}")]
    SizeOverflow { what: &'static str },
    #[error("stride {stride} is smaller than row width {row_bytes}")]
    StrideTooSmall { row_bytes: usize, stride: usize },
    #[error("stride {stride} is not aligned to {align}")]
    StrideNotAligned { stride: usize, align: usize },
    #[error("pixel format {fmt:?} does not match sample type {sample_type:?}")]
    SampleTypeMismatch {
        fmt: PixelFormat,
        sample_type: SampleType,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InputError {
    #[error("input too short: need {need} bytes, have {have}")]
    TooShort { need: usize, have: usize },
    #[error("input truncated at offset {offset} while reading {segment}")]
    TruncatedAt {
        offset: usize,
        segment: &'static str,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("not yet implemented: {what}")]
pub struct NotImplemented {
    pub what: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("unsupported: {what}")]
pub struct Unsupported {
    pub what: &'static str,
}

pub trait CodecError: core::error::Error + Send + Sync + 'static {
    fn is_truncated(&self) -> bool;
    fn is_not_implemented(&self) -> bool;
    fn is_unsupported(&self) -> bool;
    fn is_buffer_error(&self) -> bool;
}
