// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `Decoder::inspect`.

use slidecodec_jpeg::{ColorSpace, Decoder, JpegError, SofKind};

mod fixtures;
use fixtures::progressive_8x8_jpeg;

fn minimal_baseline_jpeg() -> Vec<u8> {
    // Same construction as parse::header::tests — duplicated here because
    // integration tests cannot access pub(crate) helpers.
    let mut v = Vec::new();
    v.extend_from_slice(&[0xFF, 0xD8]);
    v.extend_from_slice(&[0xFF, 0xDB, 0x00, 67, 0x00]);
    v.extend(core::iter::repeat_n(1u8, 64));
    v.extend_from_slice(&[
        0xFF,
        0xC0,
        0x00,
        17,
        8,
        0,
        16,
        0,
        16,
        3,
        1,
        (2 << 4) | 2,
        0,
        2,
        (1 << 4) | 1,
        0,
        3,
        (1 << 4) | 1,
        0,
    ]);
    // DHT length = 2 (length field) + 1 (Tc/Th) + 16 (bits[]) + 1 (value) = 20
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

fn minimal_baseline_jpeg_with_restart_interval(interval: u16) -> Vec<u8> {
    let mut bytes = minimal_baseline_jpeg();
    let sos_pos = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xda])
        .expect("SOS marker");
    bytes.splice(
        sos_pos..sos_pos,
        [
            0xff,
            0xdd,
            0x00,
            0x04,
            (interval >> 8) as u8,
            interval as u8,
        ],
    );
    bytes
}

#[test]
fn inspect_returns_info_for_valid_baseline_jpeg() {
    let info = Decoder::inspect(&minimal_baseline_jpeg()).unwrap();
    assert_eq!(info.dimensions, (16, 16));
    assert_eq!(info.sof_kind, SofKind::Baseline8);
    assert_eq!(info.color_space, ColorSpace::YCbCr);
    assert_eq!(info.bit_depth, 8);
    assert!(info.restart_interval.is_none());
    assert_eq!(info.scan_count, 1, "single SOS → scan_count must be 1");
}

#[test]
fn inspect_returns_typed_error_for_empty_input() {
    let err = Decoder::inspect(&[]).unwrap_err();
    assert!(matches!(err, JpegError::Truncated { .. }));
}

#[test]
fn inspect_returns_typed_error_for_missing_sof() {
    // SOI + EOI, nothing between
    let bytes = &[0xFF, 0xD8, 0xFF, 0xD9];
    let err = Decoder::inspect(bytes).unwrap_err();
    assert!(matches!(err, JpegError::MissingMarker { .. }));
}

#[test]
fn inspect_returns_typed_error_for_arithmetic_coding() {
    // Swap SOF0 → SOF9 in the minimal JPEG
    let mut bytes = minimal_baseline_jpeg();
    let pos = bytes.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
    bytes[pos + 1] = 0xC9;
    let err = Decoder::inspect(&bytes).unwrap_err();
    assert!(err.is_unsupported());
}

#[test]
fn inspect_is_api_misuse_predicate_negative_for_all_parse_errors() {
    // Parse errors are never API misuse.
    let err = Decoder::inspect(&[]).unwrap_err();
    assert!(!err.is_api_misuse());
}

#[test]
fn inspect_reports_all_progressive_scans() {
    let info = Decoder::inspect(&progressive_8x8_jpeg()).unwrap();
    assert_eq!(info.sof_kind, SofKind::Progressive8);
    assert_eq!(info.scan_count, 10);
}

#[test]
fn inspect_treats_dri_zero_as_no_restart_interval() {
    let info = Decoder::inspect(&minimal_baseline_jpeg_with_restart_interval(0)).unwrap();
    assert!(info.restart_interval.is_none());
}
