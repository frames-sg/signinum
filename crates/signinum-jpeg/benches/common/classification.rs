// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

use signinum_jpeg::ColorSpace;

pub(crate) const FULL_FRAME_MAX_OUTPUT_BYTES: usize = 512 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum DecodeMode {
    Gray,
    Rgb,
}

impl DecodeMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Gray => "gray",
            Self::Rgb => "rgb",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum CorpusInputClass {
    BoundedFullFrame,
    VeryLarge,
}

impl CorpusInputClass {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::BoundedFullFrame => "bounded_full_frame",
            Self::VeryLarge => "very_large",
        }
    }
}

pub(crate) fn color_space_mode(color_space: ColorSpace) -> Option<DecodeMode> {
    match color_space {
        ColorSpace::Grayscale => Some(DecodeMode::Gray),
        ColorSpace::YCbCr | ColorSpace::Rgb => Some(DecodeMode::Rgb),
        ColorSpace::Cmyk | ColorSpace::Ycck => None,
    }
}

pub(crate) fn classify_corpus_input(dimensions: (u32, u32), mode: DecodeMode) -> CorpusInputClass {
    match full_frame_output_len(dimensions, mode) {
        Some(bytes) if bytes <= FULL_FRAME_MAX_OUTPUT_BYTES => CorpusInputClass::BoundedFullFrame,
        _ => CorpusInputClass::VeryLarge,
    }
}

pub(crate) fn should_bench_decode_rows_rgb(
    mode: DecodeMode,
    input_class: CorpusInputClass,
) -> bool {
    should_bench_decode_rows_rgb_for_policy(mode, input_class, force_full_frame_compare_from_env())
}

pub(crate) fn should_bench_decode_rows_rgb_for_policy(
    mode: DecodeMode,
    input_class: CorpusInputClass,
    force_full_frame: bool,
) -> bool {
    if force_full_frame {
        return false;
    }
    mode == DecodeMode::Rgb && input_class == CorpusInputClass::VeryLarge
}

pub(crate) fn should_compare_full_frame(mode: DecodeMode, input_class: CorpusInputClass) -> bool {
    should_compare_full_frame_for_policy(mode, input_class, force_full_frame_compare_from_env())
}

pub(crate) fn should_compare_full_frame_for_policy(
    mode: DecodeMode,
    input_class: CorpusInputClass,
    force_full_frame: bool,
) -> bool {
    match input_class {
        CorpusInputClass::BoundedFullFrame => true,
        CorpusInputClass::VeryLarge => {
            force_full_frame && matches!(mode, DecodeMode::Gray | DecodeMode::Rgb)
        }
    }
}

fn force_full_frame_compare_from_env() -> bool {
    std::env::var_os("SIGNINUM_FORCE_FULL_FRAME")
        .is_some_and(|value| !matches!(value.to_str(), Some("0" | "false" | "FALSE" | "False")))
}

fn full_frame_output_len(dimensions: (u32, u32), mode: DecodeMode) -> Option<usize> {
    let bpp = match mode {
        DecodeMode::Gray => 1usize,
        DecodeMode::Rgb => 3usize,
    };
    usize::try_from(dimensions.0)
        .ok()
        .and_then(|width| {
            usize::try_from(dimensions.1)
                .ok()
                .map(|height| (width, height))
        })
        .and_then(|(width, height)| width.checked_mul(height))
        .and_then(|pixels| pixels.checked_mul(bpp))
}
