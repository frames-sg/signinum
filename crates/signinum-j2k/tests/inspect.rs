// SPDX-License-Identifier: Apache-2.0

use signinum_core::Colorspace;
use signinum_j2k::{J2kDecoder, J2kError, J2kView};
use signinum_j2k_native::{encode_htj2k, EncodeOptions};

fn minimal_codestream() -> Vec<u8> {
    let mut bytes = vec![0xFF, 0x4F];
    let mut siz = Vec::new();
    push_u16(&mut siz, 0);
    push_u32(&mut siz, 128);
    push_u32(&mut siz, 64);
    push_u32(&mut siz, 0);
    push_u32(&mut siz, 0);
    push_u32(&mut siz, 64);
    push_u32(&mut siz, 64);
    push_u32(&mut siz, 0);
    push_u32(&mut siz, 0);
    push_u16(&mut siz, 3);
    for _ in 0..3 {
        siz.extend_from_slice(&[0x07, 0x01, 0x01]);
    }
    bytes.extend_from_slice(&[0xFF, 0x51]);
    push_u16(&mut bytes, (siz.len() + 2) as u16);
    bytes.extend_from_slice(&siz);

    let cod = [0x00, 0x00, 0x00, 0x01, 0x01, 0x05, 0x04, 0x04, 0x00, 0x01];
    bytes.extend_from_slice(&[0xFF, 0x52]);
    push_u16(&mut bytes, (cod.len() + 2) as u16);
    bytes.extend_from_slice(&cod);
    bytes.extend_from_slice(&[0xFF, 0x90, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    bytes
}

fn codestream_without_siz() -> Vec<u8> {
    let mut bytes = vec![0xFF, 0x4F];
    bytes.extend_from_slice(&[
        0xFF, 0x52, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x01, 0x01, 0x05, 0x04, 0x04, 0x00, 0x01,
    ]);
    bytes.extend_from_slice(&[0xFF, 0x90, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    bytes
}

fn codestream_without_cod() -> Vec<u8> {
    let mut bytes = vec![0xFF, 0x4F];
    let mut siz = Vec::new();
    push_u16(&mut siz, 0);
    push_u32(&mut siz, 128);
    push_u32(&mut siz, 64);
    push_u32(&mut siz, 0);
    push_u32(&mut siz, 0);
    push_u32(&mut siz, 64);
    push_u32(&mut siz, 64);
    push_u32(&mut siz, 0);
    push_u32(&mut siz, 0);
    push_u16(&mut siz, 3);
    for _ in 0..3 {
        siz.extend_from_slice(&[0x07, 0x01, 0x01]);
    }
    bytes.extend_from_slice(&[0xFF, 0x51]);
    push_u16(&mut bytes, (siz.len() + 2) as u16);
    bytes.extend_from_slice(&siz);
    bytes.extend_from_slice(&[0xFF, 0x90, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    bytes
}

fn codestream_truncated_after_main_header() -> Vec<u8> {
    let mut bytes = minimal_codestream();
    bytes.truncate(bytes.len() - 10);
    bytes
}

fn minimal_jp2() -> Vec<u8> {
    let codestream = minimal_codestream();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0D, 0x0A, 0x87, 0x0A]);
    bytes.extend_from_slice(&[
        0, 0, 0, 20, b'f', b't', b'y', b'p', b'j', b'p', b'2', b' ', 0, 0, 0, 0, b'j', b'p', b'2',
        b' ',
    ]);
    bytes.extend_from_slice(&[
        0, 0, 0, 45, b'j', b'p', b'2', b'h', 0, 0, 0, 22, b'i', b'h', b'd', b'r', 0, 0, 0, 64, 0,
        0, 0, 128, 0, 3, 7, 7, 0, 0, 0, 0, 0, 15, b'c', b'o', b'l', b'r', 1, 0, 0, 0, 0, 0, 16,
    ]);
    let len = (8 + codestream.len()) as u32;
    bytes.extend_from_slice(&len.to_be_bytes());
    bytes.extend_from_slice(b"jp2c");
    bytes.extend_from_slice(&codestream);
    bytes
}

fn ht_codestream() -> Vec<u8> {
    let pixels = [10_u8, 20, 30, 40];
    encode_htj2k(&pixels, 2, 2, 1, 8, false, &EncodeOptions::default()).expect("encode ht")
}

fn ht_jp2() -> Vec<u8> {
    let codestream = ht_codestream();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0D, 0x0A, 0x87, 0x0A]);
    bytes.extend_from_slice(&[
        0, 0, 0, 20, b'f', b't', b'y', b'p', b'j', b'p', b'2', b' ', 0, 0, 0, 0, b'j', b'p', b'2',
        b' ',
    ]);
    bytes.extend_from_slice(&[
        0, 0, 0, 45, b'j', b'p', b'2', b'h', 0, 0, 0, 22, b'i', b'h', b'd', b'r', 0, 0, 0, 2, 0, 0,
        0, 2, 0, 1, 7, 7, 0, 0, 0, 0, 0, 15, b'c', b'o', b'l', b'r', 1, 0, 0, 0, 0, 0, 17,
    ]);
    let len = (8 + codestream.len()) as u32;
    bytes.extend_from_slice(&len.to_be_bytes());
    bytes.extend_from_slice(b"jp2c");
    bytes.extend_from_slice(&codestream);
    bytes
}

fn jp2_with_jp2c_before_jp2h() -> Vec<u8> {
    let codestream = minimal_codestream();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0D, 0x0A, 0x87, 0x0A]);
    bytes.extend_from_slice(&[
        0, 0, 0, 20, b'f', b't', b'y', b'p', b'j', b'p', b'2', b' ', 0, 0, 0, 0, b'j', b'p', b'2',
        b' ',
    ]);
    let len = (8 + codestream.len()) as u32;
    bytes.extend_from_slice(&len.to_be_bytes());
    bytes.extend_from_slice(b"jp2c");
    bytes.extend_from_slice(&codestream);
    bytes.extend_from_slice(&[
        0, 0, 0, 45, b'j', b'p', b'2', b'h', 0, 0, 0, 22, b'i', b'h', b'd', b'r', 0, 0, 0, 64, 0,
        0, 0, 128, 0, 3, 7, 7, 0, 0, 0, 0, 0, 15, b'c', b'o', b'l', b'r', 1, 0, 0, 0, 0, 0, 16,
    ]);
    bytes
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

#[test]
fn inspect_raw_codestream_reports_core_info() {
    let info = J2kDecoder::inspect(&minimal_codestream()).expect("codestream inspect");
    assert_eq!(info.dimensions, (128, 64));
    assert_eq!(info.components, 3);
    assert_eq!(info.bit_depth, 8);
    assert_eq!(info.colorspace, Colorspace::Rct);
    assert_eq!(info.resolution_levels, 6);
    let tiles = info.tile_layout.expect("tile layout");
    assert_eq!(tiles.tile_width, 64);
    assert_eq!(tiles.tile_height, 64);
    assert_eq!(tiles.tiles_x, 2);
    assert_eq!(tiles.tiles_y, 1);
}

#[test]
fn inspect_jp2_uses_container_colorspace() {
    let info = J2kDecoder::inspect(&minimal_jp2()).expect("jp2 inspect");
    assert_eq!(info.dimensions, (128, 64));
    assert_eq!(info.colorspace, Colorspace::SRgb);
}

#[test]
fn view_and_decoder_share_inspect_info() {
    let bytes = ht_jp2();
    let view = J2kView::parse(&bytes).expect("view");
    let dec = J2kDecoder::from_view(view).expect("decoder");
    assert_eq!(dec.info().dimensions, (2, 2));
    assert_eq!(dec.info().components, 1);
}

#[test]
fn codestream_without_siz_is_rejected() {
    let err = J2kDecoder::inspect(&codestream_without_siz()).unwrap_err();
    assert!(matches!(
        err,
        J2kError::MissingRequiredMarker { marker: "SIZ" }
    ));
}

#[test]
fn bad_jp2_signature_is_rejected() {
    let mut bad = minimal_jp2();
    bad[11] = 0x00;
    let err = J2kDecoder::inspect(&bad).unwrap_err();
    assert!(matches!(err, J2kError::InvalidBox { .. }));
}

#[test]
fn codestream_without_cod_is_rejected() {
    let err = J2kDecoder::inspect(&codestream_without_cod()).unwrap_err();
    assert!(matches!(
        err,
        J2kError::MissingRequiredMarker { marker: "COD" }
    ));
}

#[test]
fn codestream_truncated_after_main_header_is_rejected() {
    let err = J2kDecoder::inspect(&codestream_truncated_after_main_header()).unwrap_err();
    assert!(matches!(
        err,
        J2kError::Input(signinum_core::InputError::TruncatedAt { .. })
    ));
}

#[test]
fn jp2_with_codestream_before_header_is_rejected() {
    let err = J2kDecoder::inspect(&jp2_with_jp2c_before_jp2h()).unwrap_err();
    assert!(matches!(err, J2kError::InvalidBox { .. }));
}

#[test]
fn inspect_ht_codestream_reports_core_info() {
    let info = J2kDecoder::inspect(&ht_codestream()).expect("ht inspect");
    assert_eq!(info.dimensions, (2, 2));
    assert_eq!(info.components, 1);
    assert_eq!(info.bit_depth, 8);
    assert_eq!(info.colorspace, Colorspace::SGray);
}

#[test]
fn inspect_ht_jp2_reports_core_info() {
    let info = J2kDecoder::inspect(&ht_jp2()).expect("ht jp2 inspect");
    assert_eq!(info.dimensions, (2, 2));
    assert_eq!(info.components, 1);
    assert_eq!(info.colorspace, Colorspace::SGray);
}
