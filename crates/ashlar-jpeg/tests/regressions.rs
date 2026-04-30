// SPDX-License-Identifier: Apache-2.0

//! Regression coverage for structural decode bugs and allocation guardrails.

use ashlar_jpeg::{Decoder, Downscale, JpegError, PixelFormat, RowSink};

mod fixtures;
use fixtures::minimal_baseline_420_jpeg;

const R_ID: u8 = 0x52;
const G_ID: u8 = 0x47;
const B_ID: u8 = 0x42;

#[test]
fn decoder_new_rejects_unknown_scan_component_id() {
    let mut bytes = minimal_baseline_420_jpeg();
    let sos = bytes
        .windows(2)
        .position(|w| w == [0xff, 0xda])
        .expect("fixture SOS");
    bytes[sos + 9] = 9;

    let err = Decoder::new(&bytes).expect_err("unknown Cs_i must be rejected");
    assert!(matches!(
        err,
        JpegError::UnknownScanComponent { component: 9, .. }
    ));
}

#[test]
fn decoder_new_rejects_duplicate_scan_component_id() {
    let mut bytes = minimal_baseline_420_jpeg();
    let sos = bytes
        .windows(2)
        .position(|w| w == [0xff, 0xda])
        .expect("fixture SOS");
    bytes[sos + 9] = 2;

    let err = Decoder::new(&bytes).expect_err("duplicate Cs_i must be rejected");
    assert!(matches!(
        err,
        JpegError::DuplicateScanComponent { component: 2, .. }
    ));
}

#[test]
fn decoder_new_rejects_missing_scan_component_in_sequential_scan() {
    let bytes = minimal_two_component_scan_jpeg();

    let err = Decoder::new(&bytes).expect_err("baseline Ns != Nf must be rejected");
    assert!(matches!(
        err,
        JpegError::InvalidSequentialComponentSet {
            expected: 3,
            found: 2,
            ..
        }
    ));
}

#[test]
fn decode_into_matches_when_scan_order_differs_from_sof_order() {
    let canonical = rgb_app14_constant_jpeg([R_ID, G_ID, B_ID]);
    let reordered = rgb_app14_constant_jpeg([B_ID, G_ID, R_ID]);

    let canonical_rgb = decode_rgb(&canonical);
    let reordered_rgb = decode_rgb(&reordered);

    assert_ne!(
        canonical_rgb[0], canonical_rgb[1],
        "fixture must produce distinct RGB channels"
    );
    assert_eq!(
        canonical_rgb, reordered_rgb,
        "scan-order permutation must not change logical RGB output"
    );
}

#[test]
fn decoder_new_rejects_extra_sequential_scan() {
    let mut bytes = minimal_baseline_420_jpeg();
    let eoi = bytes
        .windows(2)
        .rposition(|w| w == [0xff, 0xd9])
        .expect("fixture EOI");
    let second_scan = [
        0xff, 0xda, 0x00, 0x0c, 3, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0, 0x00,
    ];
    bytes.splice(eoi..eoi, second_scan);

    let err = Decoder::new(&bytes).expect_err("extra SOS must be rejected");
    assert!(matches!(
        err,
        JpegError::InvalidSequentialScanCount { count: 2, .. }
    ));
}

#[test]
fn decoder_new_rejects_dimension_overflow() {
    let bytes = minimal_baseline_jpeg((65_535, 65_535));

    let err = Decoder::new(&bytes).expect_err("oversized dimensions must be rejected");
    assert!(matches!(
        err,
        JpegError::DimensionOverflow {
            width: 65_535,
            height: 65_535
        }
    ));
}

#[test]
fn decode_into_large_streaming_decode_hits_output_validation() {
    let bytes = minimal_baseline_jpeg((50_000, 50_000));
    let dec = Decoder::new(&bytes).expect("header must parse before cap check");

    let err = dec
        .decode_scaled_into(&mut [], 50_000, PixelFormat::Gray8, Downscale::None)
        .expect_err("undersized caller buffer must be rejected");
    assert!(matches!(err, JpegError::OutputBufferTooSmall { .. }));
}

#[derive(Default)]
struct NullSink;

impl RowSink<u8> for NullSink {
    type Error = JpegError;

    fn write_row(&mut self, _y: u32, _row: &[u8]) -> Result<(), JpegError> {
        Ok(())
    }
}

#[test]
fn decode_rows_does_not_use_full_image_scratch_cap() {
    let bytes = minimal_grayscale_jpeg((65_000, 65_000));
    let dec = Decoder::new(&bytes).expect("header must parse before row decode");

    let err = dec
        .decode_rows(&mut NullSink)
        .expect_err("synthetic scan is intentionally truncated");
    assert!(
        !matches!(err, JpegError::MemoryCapExceeded { .. }),
        "row-streaming decode should not preflight full-image scratch"
    );
}

#[test]
fn decode_into_handles_restart_marker_after_partial_entropy_byte() {
    let bytes = grayscale_restart_jpeg();
    let dec = Decoder::new(&bytes).expect("restart fixture must parse");
    let (width, height) = dec.info().dimensions;
    let stride = width as usize;
    let mut out = vec![0u8; stride * height as usize];

    dec.decode_scaled_into(&mut out, stride, PixelFormat::Gray8, Downscale::None)
        .expect("restart-coded baseline stream must decode");
    assert!(out.iter().all(|&sample| sample == 128));
}

fn decode_rgb(bytes: &[u8]) -> Vec<u8> {
    let dec = Decoder::new(bytes).expect("fixture must construct");
    let (w, h) = dec.info().dimensions;
    let mut out = vec![0u8; (w * h * 3) as usize];
    dec.decode_scaled_into(
        &mut out,
        (w * 3) as usize,
        PixelFormat::Rgb8,
        Downscale::None,
    )
    .expect("fixture decode must succeed");
    out
}

fn minimal_two_component_scan_jpeg() -> Vec<u8> {
    let mut bytes = minimal_baseline_jpeg((16, 16));
    let sos = bytes
        .windows(2)
        .position(|w| w == [0xff, 0xda])
        .expect("synthetic SOS");
    let eoi = bytes
        .windows(2)
        .rposition(|w| w == [0xff, 0xd9])
        .expect("synthetic EOI");
    bytes.splice(
        sos..eoi,
        [0xff, 0xda, 0x00, 0x0a, 2, 1, 0x00, 2, 0x00, 0, 63, 0, 0x00],
    );
    bytes
}

fn minimal_baseline_jpeg((width, height): (u16, u16)) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff,
        0xc0,
        0x00,
        17,
        8,
        (height >> 8) as u8,
        height as u8,
        (width >> 8) as u8,
        width as u8,
        3,
        1,
        0x11,
        0,
        2,
        0x11,
        0,
        3,
        0x11,
        0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xda, 0x00, 0x0c, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0,
    ]);
    bytes.extend_from_slice(&[0x00, 0xff, 0xd9]);
    bytes
}

fn minimal_grayscale_jpeg((width, height): (u16, u16)) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff,
        0xc0,
        0x00,
        11,
        8,
        (height >> 8) as u8,
        height as u8,
        (width >> 8) as u8,
        width as u8,
        1,
        1,
        0x11,
        0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, 0, 63, 0]);
    bytes.extend_from_slice(&[0x00, 0xff, 0xd9]);
    bytes
}

fn grayscale_restart_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[0xff, 0xc0, 0x00, 11, 8, 0, 8, 0, 16, 1, 1, 0x11, 0]);
    bytes.extend_from_slice(&[0xff, 0xdd, 0x00, 0x04, 0x00, 0x01]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, 0, 63, 0]);
    bytes.extend_from_slice(&[0x00, 0xff, 0xd0, 0x00, 0xff, 0xd9]);
    bytes
}

fn rgb_app14_constant_jpeg(scan_order: [u8; 3]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[
        0xff, 0xee, 0x00, 0x0e, b'A', b'd', b'o', b'b', b'e', 0x00, 0x00, 0x64, 0x00, 0x00, 0x00,
        0x00,
    ]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff, 0xc0, 0x00, 17, 8, 0, 8, 0, 8, 3, R_ID, 0x11, 0, G_ID, 0x11, 0, B_ID, 0x11, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00,
    ]);
    bytes.extend_from_slice(&[
        0xff,
        0xda,
        0x00,
        0x0c,
        3,
        scan_order[0],
        0x00,
        scan_order[1],
        0x00,
        scan_order[2],
        0x00,
        0,
        63,
        0,
    ]);
    bytes.extend(pack_entropy_bytes(scan_order));
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn pack_entropy_bytes(scan_order: [u8; 3]) -> Vec<u8> {
    let mut bits = Vec::new();
    for component in scan_order {
        let dc = match component {
            R_ID => 288u16,
            G_ID => 79u16,
            B_ID => 39u16,
            other => panic!("unexpected component {other:02x}"),
        };
        push_bits(&mut bits, 0, 1);
        push_bits(&mut bits, u32::from(dc), 9);
        push_bits(&mut bits, 0, 1);
    }

    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bits.len() {
        let mut byte = 0u8;
        for bit in 0..8 {
            byte <<= 1;
            let set = bits.get(i + bit).copied().unwrap_or(true);
            if set {
                byte |= 1;
            }
        }
        out.push(byte);
        if byte == 0xff {
            out.push(0x00);
        }
        i += 8;
    }
    out
}

fn push_bits(bits: &mut Vec<bool>, value: u32, width: u8) {
    for shift in (0..width).rev() {
        bits.push(((value >> shift) & 1) != 0);
    }
}
