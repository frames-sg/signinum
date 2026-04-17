// SPDX-License-Identifier: Apache-2.0

//! JPEG decoder optimized for whole-slide images.
//!
//! See the top-level README for project positioning. The primary entry point
//! is [`Decoder`] — start with [`Decoder::inspect`] for header-only parsing.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(unreachable_pub)]
// `missing_docs` is scheduled to turn on before 0.1.0; see Cargo.toml for rationale.

extern crate alloc;

pub mod info;
pub use info::{
    ColorSpace, ColorTransform, DownscaleFactor, Info, OutputFormat, Rect, SamplingFactors, SofKind,
};

pub mod error;
pub use error::{
    BuilderConflictReason, HuffmanFailure, JpegError, MarkerKind, TableKind, UnsupportedReason,
    Warning,
};

pub(crate) mod parse;

pub(crate) mod entropy;

pub(crate) mod internal;

pub mod decoder;
pub use decoder::{DecodeOutcome, Decoder};
