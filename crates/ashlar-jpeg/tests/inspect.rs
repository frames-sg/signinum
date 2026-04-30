// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `Decoder::inspect`.

use ashlar_jpeg::{
    ColorSpace, ColorTransform, DecodeOptions, Decoder, JpegError, JpegView, McuGeometry,
    RestartSegment, SofKind,
};

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

fn restart_coded_grayscale_jpeg(width: u16, height: u16) -> Vec<u8> {
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
    bytes.extend_from_slice(&[0xff, 0xdd, 0x00, 0x04, 0x00, 0x01]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, 0, 63, 0]);

    let mcu_cols = u32::from(width).div_ceil(8);
    let mcu_rows = u32::from(height).div_ceil(8);
    let mcu_count = (mcu_cols * mcu_rows) as usize;
    for mcu in 0..mcu_count {
        bytes.push(0x00);
        if mcu + 1 != mcu_count {
            bytes.extend_from_slice(&[0xff, 0xd0 | ((mcu as u8) & 0x07)]);
        }
    }

    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn scan_data_offset(bytes: &[u8]) -> usize {
    let sos_pos = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xda])
        .expect("SOS marker");
    let len = u16::from_be_bytes([bytes[sos_pos + 2], bytes[sos_pos + 3]]) as usize;
    sos_pos + 2 + len
}

fn restart_marker_offsets(bytes: &[u8]) -> Vec<usize> {
    bytes
        .windows(2)
        .enumerate()
        .filter_map(|(offset, window)| {
            (window[0] == 0xff && (0xd0..=0xd7).contains(&window[1])).then_some(offset)
        })
        .collect()
}

#[test]
fn inspect_returns_info_for_valid_baseline_jpeg() {
    let info = Decoder::inspect(&minimal_baseline_jpeg()).unwrap();
    assert_eq!(info.dimensions, (16, 16));
    assert_eq!(info.sof_kind, SofKind::Baseline8);
    assert_eq!(info.color_space, ColorSpace::YCbCr);
    assert_eq!(info.bit_depth, 8);
    assert!(info.restart_interval.is_none());
    assert_eq!(
        info.mcu_geometry,
        McuGeometry {
            width: 16,
            height: 16,
            columns: 1,
            rows: 1,
            count: 1,
        }
    );
    assert_eq!(info.scan_count, 1, "single SOS → scan_count must be 1");
}

#[test]
fn decode_options_color_transform_setter_round_trips() {
    let mut options = DecodeOptions::default();
    options.set_color_transform(ColorTransform::ForceRgb);
    assert!(matches!(
        options.color_transform(),
        ColorTransform::ForceRgb
    ));
}

#[test]
fn inspect_with_options_forces_three_component_color_space() {
    let bytes = minimal_baseline_jpeg();
    let auto = Decoder::inspect(&bytes).unwrap();
    assert_eq!(auto.color_space, ColorSpace::YCbCr);

    let force_rgb = Decoder::inspect_with_options(
        &bytes,
        DecodeOptions::default().with_color_transform(ColorTransform::ForceRgb),
    )
    .unwrap();
    assert_eq!(force_rgb.color_space, ColorSpace::Rgb);

    let force_ycbcr = Decoder::inspect_with_options(
        &bytes,
        DecodeOptions::default().with_color_transform(ColorTransform::ForceYCbCr),
    )
    .unwrap();
    assert_eq!(force_ycbcr.color_space, ColorSpace::YCbCr);
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

#[test]
fn inspect_reports_restart_interval_and_mcu_geometry_for_wsi_planning() {
    let info = Decoder::inspect(&fixtures::baseline_420_restart_32x16_jpeg()).unwrap();

    assert_eq!(info.dimensions, (32, 16));
    assert_eq!(info.restart_interval, Some(2));
    assert_eq!(
        info.mcu_geometry,
        McuGeometry {
            width: 16,
            height: 16,
            columns: 2,
            rows: 1,
            count: 2,
        }
    );
}

#[test]
fn jpeg_view_restart_index_reports_original_byte_offsets() {
    let bytes = restart_coded_grayscale_jpeg(24, 8);
    let view = JpegView::parse(&bytes).expect("view");
    let index = view
        .restart_index()
        .expect("restart index")
        .expect("DRI should produce an index");
    let scan_data_offset = scan_data_offset(&bytes);
    let rst_offsets = restart_marker_offsets(&bytes);

    assert_eq!(index.scan_data_offset, scan_data_offset);
    assert_eq!(index.interval_mcus, 1);
    assert_eq!(
        index.segments,
        vec![
            RestartSegment {
                start_mcu: 0,
                entropy_offset: scan_data_offset,
                marker_offset: None,
                marker: None,
            },
            RestartSegment {
                start_mcu: 1,
                entropy_offset: rst_offsets[0] + 2,
                marker_offset: Some(rst_offsets[0]),
                marker: Some(0xd0),
            },
            RestartSegment {
                start_mcu: 2,
                entropy_offset: rst_offsets[1] + 2,
                marker_offset: Some(rst_offsets[1]),
                marker: Some(0xd1),
            },
        ]
    );

    let decoder_index = Decoder::new(&bytes)
        .expect("decoder")
        .restart_index()
        .expect("decoder restart index");
    assert_eq!(decoder_index, Some(index));
}

#[test]
fn restart_index_is_none_without_dri() {
    let bytes = minimal_baseline_jpeg();
    let view = JpegView::parse(&bytes).expect("view");
    assert_eq!(view.restart_index().expect("restart index"), None);
}
