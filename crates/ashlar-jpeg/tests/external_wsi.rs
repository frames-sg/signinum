// SPDX-License-Identifier: Apache-2.0

//! Optional local-corpus regression coverage for extracted WSI JPEGs.

#[path = "../benches/common/classification.rs"]
mod classification;

use ashlar_jpeg::{ColorSpace, Decoder, Downscale, JpegError, PixelFormat, RowSink};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Default)]
struct NullSink;

impl RowSink<u8> for NullSink {
    type Error = JpegError;

    fn write_row(&mut self, _y: u32, _row: &[u8]) -> Result<(), JpegError> {
        Ok(())
    }
}

#[test]
fn bounded_full_frame_classification_uses_output_bytes_and_mode() {
    use classification::{
        classify_corpus_input, CorpusInputClass, DecodeMode, FULL_FRAME_MAX_OUTPUT_BYTES,
    };

    let rgb_threshold_height = (FULL_FRAME_MAX_OUTPUT_BYTES / 3) as u32;
    assert_eq!(
        classify_corpus_input((1, rgb_threshold_height), DecodeMode::Rgb),
        CorpusInputClass::BoundedFullFrame
    );
    assert_eq!(
        classify_corpus_input((1, rgb_threshold_height + 1), DecodeMode::Rgb),
        CorpusInputClass::VeryLarge
    );

    let gray_threshold_height = FULL_FRAME_MAX_OUTPUT_BYTES as u32;
    assert_eq!(
        classify_corpus_input((1, gray_threshold_height), DecodeMode::Gray),
        CorpusInputClass::BoundedFullFrame
    );
    assert_eq!(
        classify_corpus_input((1, gray_threshold_height + 1), DecodeMode::Gray),
        CorpusInputClass::VeryLarge
    );
}

#[test]
fn oversized_or_overflowing_inputs_are_treated_as_very_large() {
    use classification::{classify_corpus_input, CorpusInputClass, DecodeMode};

    assert_eq!(
        classify_corpus_input((u32::MAX, u32::MAX), DecodeMode::Rgb),
        CorpusInputClass::VeryLarge
    );
    assert_eq!(
        classify_corpus_input((u32::MAX, u32::MAX), DecodeMode::Gray),
        CorpusInputClass::VeryLarge
    );
}

#[test]
fn large_file_row_streaming_is_rgb_only_for_the_bench_contract() {
    use classification::{
        classify_corpus_input, color_space_mode, should_bench_decode_rows_rgb_for_policy,
        CorpusInputClass, DecodeMode,
    };

    assert_eq!(
        color_space_mode(ColorSpace::Grayscale),
        Some(DecodeMode::Gray)
    );
    assert_eq!(color_space_mode(ColorSpace::YCbCr), Some(DecodeMode::Rgb));
    assert_eq!(color_space_mode(ColorSpace::Rgb), Some(DecodeMode::Rgb));
    assert_eq!(color_space_mode(ColorSpace::Cmyk), None);
    assert_eq!(color_space_mode(ColorSpace::Ycck), None);

    assert!(should_bench_decode_rows_rgb_for_policy(
        DecodeMode::Rgb,
        CorpusInputClass::VeryLarge,
        false,
    ));
    assert!(!should_bench_decode_rows_rgb_for_policy(
        DecodeMode::Rgb,
        CorpusInputClass::BoundedFullFrame,
        false,
    ));
    assert!(!should_bench_decode_rows_rgb_for_policy(
        DecodeMode::Gray,
        CorpusInputClass::VeryLarge,
        false,
    ));

    assert_eq!(
        classify_corpus_input((1, 1), DecodeMode::Rgb),
        CorpusInputClass::BoundedFullFrame
    );
}

#[test]
fn force_full_frame_policy_disables_large_rgb_skip() {
    use classification::{
        classify_corpus_input, should_bench_decode_rows_rgb_for_policy,
        should_compare_full_frame_for_policy, CorpusInputClass, DecodeMode,
        FULL_FRAME_MAX_OUTPUT_BYTES,
    };

    let rgb_threshold_height = (FULL_FRAME_MAX_OUTPUT_BYTES / 3) as u32 + 1;
    assert_eq!(
        classify_corpus_input((1, rgb_threshold_height), DecodeMode::Rgb),
        CorpusInputClass::VeryLarge
    );
    assert!(should_compare_full_frame_for_policy(
        DecodeMode::Rgb,
        CorpusInputClass::VeryLarge,
        true,
    ));
    assert!(!should_bench_decode_rows_rgb_for_policy(
        DecodeMode::Rgb,
        CorpusInputClass::VeryLarge,
        true,
    ));
}

#[test]
fn extracted_wsi_jpegs_decode_when_local_corpus_is_available() {
    let Some(root) = std::env::var_os("ASHLAR_WSI_ROOT") else {
        return;
    };
    let mut files = Vec::new();
    collect_jpegs(Path::new(&root), &mut files);
    files.sort();

    for path in files {
        eprintln!("decoding {}", path.display());
        let bytes = fs::read(&path).expect("read external jpeg");
        let dec = Decoder::new(&bytes).unwrap_or_else(|err| {
            panic!("Decoder::new failed for {}: {err:?}", path.display());
        });
        let Some(mode) = classification::color_space_mode(dec.info().color_space) else {
            continue;
        };
        let (width, height) = dec.info().dimensions;
        if classification::classify_corpus_input((width, height), mode)
            == classification::CorpusInputClass::VeryLarge
        {
            dec.decode_rows(&mut NullSink).unwrap_or_else(|err| {
                panic!("decode_rows failed for {}: {err:?}", path.display());
            });
            continue;
        }
        let (fmt, stride, len) = match dec.info().color_space {
            ColorSpace::Grayscale => {
                let stride = width as usize;
                (PixelFormat::Gray8, stride, stride * height as usize)
            }
            ColorSpace::YCbCr | ColorSpace::Rgb => {
                let stride = (width as usize) * 3;
                (PixelFormat::Rgb8, stride, stride * height as usize)
            }
            ColorSpace::Cmyk | ColorSpace::Ycck => continue,
        };
        let mut out = vec![0u8; len];
        dec.decode_scaled_into(&mut out, stride, fmt, Downscale::None)
            .unwrap_or_else(|err| {
                panic!("decode_into failed for {}: {err:?}", path.display());
            });
    }
}

fn collect_jpegs(root: &Path, out: &mut Vec<PathBuf>) {
    if root.is_file() {
        if root
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg"))
        {
            out.push(root.to_path_buf());
        }
        return;
    }
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_jpegs(&path, out);
            continue;
        }
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg"))
        {
            out.push(path);
        }
    }
}
