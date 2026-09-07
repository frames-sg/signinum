// SPDX-License-Identifier: MIT OR Apache-2.0

//! Regression scaffold for JP2 palette/component-map validation.

use j2k_native::{
    encode, encode_htj2k, ColorError, DecodeError, DecodeSettings, EncodeOptions, FormatError,
    Image,
};

fn jp2_box(box_type: [u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let payload_len = u32::try_from(payload.len()).expect("test box payload length fits u32");
    let box_len = payload_len
        .checked_add(8)
        .expect("test box length fits u32");
    out.extend_from_slice(&box_len.to_be_bytes());
    out.extend_from_slice(&box_type);
    out.extend_from_slice(payload);
    out
}

#[test]
fn empty_cmap_with_palette_returns_error() {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let codestream = encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode fixture");
    let jp2 = jp2_with_empty_cmap(&codestream);

    let result = Image::new(&jp2, &DecodeSettings::default()).and_then(|image| image.decode());

    let err = result.expect_err("empty cmap must be rejected explicitly");
    assert!(
        matches!(
            err,
            DecodeError::Format(FormatError::InvalidBox)
                | DecodeError::Color(ColorError::PaletteResolutionFailed)
        ),
        "unexpected empty cmap error: {err:?}"
    );
}

#[test]
fn paired_comparison_preserves_mixed_palette_precision_and_signedness() {
    let indices = vec![0_u8; 16];
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let codestream = encode(&indices, 4, 4, 1, 8, false, &options).expect("encode index fixture");
    let source_bytes =
        jp2_with_header_payload(&codestream, &mixed_palette_jp2h_payload(4, 4, 0x0100, true));
    let different_bytes =
        jp2_with_header_payload(&codestream, &mixed_palette_jp2h_payload(4, 4, 0x0200, true));
    let settings = DecodeSettings::default();
    let source = Image::new(&source_bytes, &settings).expect("source palette JP2 parses");
    let different = Image::new(&different_bytes, &settings).expect("different palette JP2 parses");

    let source_components = source
        .decode_native_components()
        .expect("mixed palette components decode");
    assert_eq!(source_components.planes()[0].bit_depth(), 8);
    assert!(!source_components.planes()[0].signed());
    assert_eq!(source_components.planes()[1].bit_depth(), 16);
    assert!(source_components.planes()[1].signed());
    assert!(
        source_components
            .planes()
            .iter()
            .all(|plane| plane.sampling() == (1, 1)),
        "resolved palette columns must use the display grid, not the index component sampling"
    );
    assert!(
        !source
            .decoded_samples_equal(&different)
            .expect("paired palette comparison"),
        "16-bit signed palette differences must not be truncated through the 8-bit index precision"
    );
}

#[test]
fn paired_comparison_preserves_palette_integers_above_f32_precision() {
    let indices = vec![0_u8; 16];
    let codestream = encode(
        &indices,
        4,
        4,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        },
    )
    .expect("encode index fixture");
    let source_bytes = jp2_with_header_payload(
        &codestream,
        &high_precision_palette_jp2h_payload(4, 4, 0x0100_0000),
    );
    let different_bytes = jp2_with_header_payload(
        &codestream,
        &high_precision_palette_jp2h_payload(4, 4, 0x0100_0001),
    );
    let settings = DecodeSettings::default();
    let source = Image::new(&source_bytes, &settings).expect("source 25-bit palette JP2 parses");
    let different =
        Image::new(&different_bytes, &settings).expect("different 25-bit palette JP2 parses");

    let source_components = source
        .decode_native_components()
        .expect("25-bit palette components decode");
    let different_components = different
        .decode_native_components()
        .expect("different 25-bit palette components decode");
    assert_eq!(source_components.planes()[0].bit_depth(), 25);
    assert_eq!(source_components.planes()[0].data()[..4], [0, 0, 0, 1]);
    assert_eq!(different_components.planes()[0].data()[..4], [1, 0, 0, 1]);
    assert!(
        !source
            .decoded_samples_equal(&different)
            .expect("paired high-precision palette comparison"),
        "adjacent 25-bit palette values above 2^24 must remain distinguishable"
    );
}

#[test]
fn high_precision_sycc_palette_does_not_reuse_pretransform_integer_shadow() {
    let indices = vec![0_u8; 16];
    let codestream = encode(
        &indices,
        4,
        4,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        },
    )
    .expect("encode index fixture");
    let bytes =
        jp2_with_header_payload(&codestream, &high_precision_sycc_palette_jp2h_payload(4, 4));
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("sYCC palette JP2 parses");
    let components = image
        .decode_native_components()
        .expect("sYCC palette components decode");

    assert_eq!(components.planes().len(), 3);
    assert_eq!(components.planes()[0].bit_depth(), 25);
    assert_eq!(components.planes()[0].data()[..4], [0, 0, 0, 0]);
    assert_ne!(
        components.planes()[0].data()[..4],
        [0, 0, 0, 1],
        "native output must use converted RGB samples, not the exact pre-transform Y shadow"
    );
}

#[test]
fn retained_full_jp2_parse_baseline_covers_implicit_palette_mapping() {
    let indices = vec![0_u8; 16];
    let codestream = encode(
        &indices,
        4,
        4,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        },
    )
    .expect("encode index fixture");
    let bytes = jp2_with_header_payload(
        &codestream,
        &mixed_palette_jp2h_payload(4, 4, 0x0100, false),
    );
    assert_full_parse_respects_retained_baseline(&bytes);
}

#[test]
fn retained_full_jp2_parse_baseline_covers_icc_profile_clone() {
    let pixels = vec![17_u8; 4 * 4 * 3];
    let codestream = encode(
        &pixels,
        4,
        4,
        3,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        },
    )
    .expect("encode RGB fixture");
    let bytes = jp2_with_header_payload(
        &codestream,
        &icc_jp2h_payload(4, 4, include_bytes!("../assets/ProPhoto-v2-micro.icc")),
    );
    assert_full_parse_respects_retained_baseline(&bytes);
}

fn assert_full_parse_respects_retained_baseline(bytes: &[u8]) {
    let settings = DecodeSettings::default();
    let comfortable_baseline = j2k_native::DEFAULT_MAX_DECODE_BYTES - 1024 * 1024;
    Image::new_with_retained_baseline(bytes, &settings, comfortable_baseline)
        .expect("small full JP2 parse fits within one MiB of remaining headroom");
    assert!(
        Image::new_with_retained_baseline(
            bytes,
            &settings,
            j2k_native::DEFAULT_MAX_DECODE_BYTES - 1,
        )
        .is_err(),
        "one remaining byte cannot hide full JP2 parser-owned allocation growth"
    );
}

#[test]
fn missing_ihdr_returns_invalid_box() {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let codestream = encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode fixture");
    let colr = jp2_box(*b"colr", &[1, 0, 0, 0, 0, 0, 17]);
    let jp2 = jp2_with_header_payload(&codestream, &colr);

    let Err(err) = Image::new(&jp2, &DecodeSettings::default()) else {
        panic!("missing ihdr must reject");
    };

    assert!(matches!(
        err,
        DecodeError::Format(FormatError::MissingRequiredBox("ihdr"))
    ));
}

#[test]
fn invalid_ihdr_compression_type_returns_invalid_box() {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let codestream = encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode fixture");
    let ihdr = {
        let mut payload = Vec::new();
        payload.extend_from_slice(&4_u32.to_be_bytes());
        payload.extend_from_slice(&4_u32.to_be_bytes());
        payload.extend_from_slice(&1_u16.to_be_bytes());
        payload.extend_from_slice(&[7, 0, 0, 0]);
        jp2_box(*b"ihdr", &payload)
    };
    let colr = jp2_box(*b"colr", &[1, 0, 0, 0, 0, 0, 17]);
    let mut jp2h_payload = Vec::new();
    jp2h_payload.extend_from_slice(&ihdr);
    jp2h_payload.extend_from_slice(&colr);
    let jp2 = jp2_with_header_payload(&codestream, &jp2h_payload);

    let Err(err) = Image::new(&jp2, &DecodeSettings::default()) else {
        panic!("invalid ihdr compression type must reject");
    };

    assert!(matches!(err, DecodeError::Format(FormatError::InvalidBox)));
}

#[test]
fn missing_colr_returns_invalid_box() {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let codestream = encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode fixture");
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&4_u32.to_be_bytes());
    ihdr.extend_from_slice(&4_u32.to_be_bytes());
    ihdr.extend_from_slice(&1_u16.to_be_bytes());
    ihdr.extend_from_slice(&[7, 7, 0, 0]);
    let jp2h = jp2_box(*b"ihdr", &ihdr);
    let jp2 = jp2_with_header_payload(&codestream, &jp2h);

    let Err(err) = Image::new(&jp2, &DecodeSettings::default()) else {
        panic!("missing COLR must reject");
    };

    assert!(matches!(
        err,
        DecodeError::Format(FormatError::MissingRequiredBox("colr"))
    ));
}

#[test]
fn ihdr_dimension_mismatch_returns_invalid_box() {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let codestream = encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode fixture");
    let jp2 = jp2_with_header_payload(&codestream, &basic_jp2h_payload(5, 4, 1, 8));

    let Err(err) = Image::new(&jp2, &DecodeSettings::default()) else {
        panic!("IHDR dimensions must reject");
    };

    assert!(matches!(err, DecodeError::Format(FormatError::InvalidBox)));
}

#[test]
fn ihdr_bpc_mismatch_returns_invalid_box() {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let codestream = encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode fixture");
    let jp2 = jp2_with_header_payload(&codestream, &basic_jp2h_payload(4, 4, 1, 16));

    let Err(err) = Image::new(&jp2, &DecodeSettings::default()) else {
        panic!("IHDR BPC mismatch must reject");
    };

    assert!(matches!(err, DecodeError::Format(FormatError::InvalidBox)));
}

#[test]
fn bpcc_precision_mismatch_returns_invalid_box() {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let codestream = encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode fixture");
    let jp2 = jp2_with_header_payload(&codestream, &bpcc_jp2h_payload(4, 4, 1, &[15]));

    let Err(err) = Image::new(&jp2, &DecodeSettings::default()) else {
        panic!("BPCC precision mismatch must reject");
    };

    assert!(matches!(err, DecodeError::Format(FormatError::InvalidBox)));
}

#[test]
fn jph_file_type_rejects_classic_codestream() {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let codestream = encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode fixture");
    let jp2 = jp2_with_header_payload_and_file_type(
        &codestream,
        &basic_jp2h_payload(4, 4, 1, 8),
        b"jph \0\0\0\0jph ",
    );

    let Err(err) = Image::new(&jp2, &DecodeSettings::default()) else {
        panic!("JPH file type must reject classic codestreams");
    };

    assert!(matches!(
        err,
        DecodeError::Format(FormatError::InvalidFileType)
    ));
}

#[test]
fn jp2_file_type_rejects_htj2k_codestream() {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let codestream = encode_htj2k(&pixels, 4, 4, 1, 8, false, &options).expect("encode fixture");
    let jp2 = jp2_with_header_payload(&codestream, &basic_jp2h_payload(4, 4, 1, 8));

    let Err(err) = Image::new(&jp2, &DecodeSettings::default()) else {
        panic!("JP2 file type must reject HTJ2K codestreams");
    };

    assert!(matches!(
        err,
        DecodeError::Format(FormatError::InvalidFileType)
    ));
}

#[test]
fn jph_file_type_accepts_htj2k_codestream() {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let codestream = encode_htj2k(&pixels, 4, 4, 1, 8, false, &options).expect("encode fixture");
    let jp2 = jp2_with_header_payload_and_file_type(
        &codestream,
        &basic_jp2h_payload(4, 4, 1, 8),
        b"jph \0\0\0\0jph ",
    );

    let image = Image::new(&jp2, &DecodeSettings::default()).expect("JPH parses");
    let bitmap = image.decode_native().expect("JPH decodes");

    assert_eq!(bitmap.data, pixels);
}

#[test]
fn premultiplied_opacity_cdef_sets_alpha() {
    let pixels: Vec<u8> = (0_u8..64).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let codestream = encode(&pixels, 4, 4, 4, 8, false, &options).expect("encode fixture");
    let jp2 = jp2_with_header_payload(
        &codestream,
        &cdef_jp2h_payload(
            4,
            4,
            4,
            8,
            16,
            &[(0, 0, 1), (1, 0, 2), (2, 0, 3), (3, 2, 0)],
        ),
    );

    let image = Image::new(&jp2, &DecodeSettings::default()).expect("JP2 parses");

    assert!(image.has_alpha());
    let error = image
        .build_component_grid_color_plan_with_context(&mut j2k_native::DecoderContext::default())
        .expect_err("component-grid RGB planning must not drop explicit alpha");
    assert!(matches!(
        error,
        j2k_native::DecodeError::Decoding(j2k_native::DecodingError::DirectPlanUnsupported(
            j2k_native::DirectPlanUnsupportedReason::ColorRgbImageWithoutAlpha
        ))
    ));
    let components = image
        .decode_native_components()
        .expect("premultiplied-opacity JP2 decodes to owned planes");
    assert!(components.has_alpha());
    assert_eq!(components.planes().len(), 4);
}

#[test]
fn unspecified_cdef_association_decodes() {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let codestream = encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode fixture");
    let jp2 = jp2_with_header_payload(
        &codestream,
        &cdef_jp2h_payload(4, 4, 1, 8, 17, &[(0, u16::MAX, u16::MAX)]),
    );

    let bitmap = Image::new(&jp2, &DecodeSettings::default())
        .expect("JP2 parses")
        .decode_native()
        .expect("JP2 decodes");

    assert_eq!(bitmap.data, pixels);
}

fn jp2_with_empty_cmap(codestream: &[u8]) -> Vec<u8> {
    let ihdr = {
        let mut payload = Vec::new();
        payload.extend_from_slice(&4_u32.to_be_bytes());
        payload.extend_from_slice(&4_u32.to_be_bytes());
        payload.extend_from_slice(&1_u16.to_be_bytes());
        payload.extend_from_slice(&[7, 7, 0, 0]);
        jp2_box(*b"ihdr", &payload)
    };
    let colr = jp2_box(*b"colr", &[1, 0, 0, 0, 0, 0, 17]);
    let pclr = {
        let mut payload = Vec::new();
        payload.extend_from_slice(&1_u16.to_be_bytes());
        payload.push(1);
        payload.push(7);
        payload.push(0);
        jp2_box(*b"pclr", &payload)
    };
    let cmap = jp2_box(*b"cmap", &[]);

    let mut jp2h_payload = Vec::new();
    jp2h_payload.extend_from_slice(&ihdr);
    jp2h_payload.extend_from_slice(&colr);
    jp2h_payload.extend_from_slice(&pclr);
    jp2h_payload.extend_from_slice(&cmap);

    jp2_with_header_payload(codestream, &jp2h_payload)
}

fn basic_jp2h_payload(width: u32, height: u32, components: u16, bit_depth: u8) -> Vec<u8> {
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&components.to_be_bytes());
    ihdr.extend_from_slice(&[bit_depth.saturating_sub(1), 7, 0, 0]);
    let colr = jp2_box(*b"colr", &[1, 0, 0, 0, 0, 0, 17]);

    let mut jp2h_payload = Vec::new();
    jp2h_payload.extend_from_slice(&jp2_box(*b"ihdr", &ihdr));
    jp2h_payload.extend_from_slice(&colr);
    jp2h_payload
}

fn bpcc_jp2h_payload(width: u32, height: u32, components: u16, bpcc_payload: &[u8]) -> Vec<u8> {
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&components.to_be_bytes());
    ihdr.extend_from_slice(&[0xff, 7, 0, 0]);
    let colr = jp2_box(*b"colr", &[1, 0, 0, 0, 0, 0, 17]);

    let mut jp2h_payload = Vec::new();
    jp2h_payload.extend_from_slice(&jp2_box(*b"ihdr", &ihdr));
    jp2h_payload.extend_from_slice(&jp2_box(*b"bpcc", bpcc_payload));
    jp2h_payload.extend_from_slice(&colr);
    jp2h_payload
}

fn mixed_palette_jp2h_payload(
    width: u32,
    height: u32,
    signed_16_value: u16,
    include_component_mapping: bool,
) -> Vec<u8> {
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&1_u16.to_be_bytes());
    ihdr.extend_from_slice(&[7, 7, 0, 0]);

    let mut palette = Vec::new();
    palette.extend_from_slice(&1_u16.to_be_bytes());
    palette.extend_from_slice(&[2, 7, 0x8f, 0]);
    palette.extend_from_slice(&signed_16_value.to_be_bytes());

    let mut jp2h_payload = Vec::new();
    jp2h_payload.extend_from_slice(&jp2_box(*b"ihdr", &ihdr));
    jp2h_payload.extend_from_slice(&jp2_box(*b"colr", &[1, 0, 0, 0, 0, 0, 17]));
    jp2h_payload.extend_from_slice(&jp2_box(*b"pclr", &palette));
    if include_component_mapping {
        jp2h_payload.extend_from_slice(&jp2_box(*b"cmap", &[0, 0, 1, 0, 0, 0, 1, 1]));
    }
    jp2h_payload
}

fn icc_jp2h_payload(width: u32, height: u32, profile: &[u8]) -> Vec<u8> {
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&3_u16.to_be_bytes());
    ihdr.extend_from_slice(&[7, 7, 0, 0]);

    let mut color = vec![2, 0, 0];
    color.extend_from_slice(profile);
    let mut jp2h_payload = Vec::new();
    jp2h_payload.extend_from_slice(&jp2_box(*b"ihdr", &ihdr));
    jp2h_payload.extend_from_slice(&jp2_box(*b"colr", &color));
    jp2h_payload
}

fn high_precision_palette_jp2h_payload(width: u32, height: u32, value: u32) -> Vec<u8> {
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&1_u16.to_be_bytes());
    ihdr.extend_from_slice(&[7, 7, 0, 0]);

    let mut palette = Vec::new();
    palette.extend_from_slice(&1_u16.to_be_bytes());
    palette.extend_from_slice(&[1, 24]);
    palette.extend_from_slice(&value.to_be_bytes());

    let mut jp2h_payload = Vec::new();
    jp2h_payload.extend_from_slice(&jp2_box(*b"ihdr", &ihdr));
    jp2h_payload.extend_from_slice(&jp2_box(*b"colr", &[1, 0, 0, 0, 0, 0, 17]));
    jp2h_payload.extend_from_slice(&jp2_box(*b"pclr", &palette));
    jp2h_payload.extend_from_slice(&jp2_box(*b"cmap", &[0, 0, 1, 0]));
    jp2h_payload
}

fn high_precision_sycc_palette_jp2h_payload(width: u32, height: u32) -> Vec<u8> {
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&1_u16.to_be_bytes());
    ihdr.extend_from_slice(&[7, 7, 0, 0]);

    let mut palette = Vec::new();
    palette.extend_from_slice(&1_u16.to_be_bytes());
    palette.extend_from_slice(&[3, 24, 24, 24]);
    palette.extend_from_slice(&0x0100_0000_u32.to_be_bytes());
    palette.extend_from_slice(&0_u32.to_be_bytes());
    palette.extend_from_slice(&0_u32.to_be_bytes());

    let mut jp2h_payload = Vec::new();
    jp2h_payload.extend_from_slice(&jp2_box(*b"ihdr", &ihdr));
    jp2h_payload.extend_from_slice(&jp2_box(*b"colr", &[1, 0, 0, 0, 0, 0, 18]));
    jp2h_payload.extend_from_slice(&jp2_box(*b"pclr", &palette));
    jp2h_payload.extend_from_slice(&jp2_box(*b"cmap", &[0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 2]));
    jp2h_payload
}

fn cdef_jp2h_payload(
    width: u32,
    height: u32,
    components: u16,
    bit_depth: u8,
    colorspace: u32,
    definitions: &[(u16, u16, u16)],
) -> Vec<u8> {
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&components.to_be_bytes());
    ihdr.extend_from_slice(&[bit_depth.saturating_sub(1), 7, 0, 0]);
    let mut colr = vec![1, 0, 0];
    colr.extend_from_slice(&colorspace.to_be_bytes());
    let mut cdef = Vec::new();
    let definition_count = u16::try_from(definitions.len()).expect("test CDEF count fits u16");
    cdef.extend_from_slice(&definition_count.to_be_bytes());
    for (channel, channel_type, association) in definitions {
        cdef.extend_from_slice(&channel.to_be_bytes());
        cdef.extend_from_slice(&channel_type.to_be_bytes());
        cdef.extend_from_slice(&association.to_be_bytes());
    }

    let mut jp2h_payload = Vec::new();
    jp2h_payload.extend_from_slice(&jp2_box(*b"ihdr", &ihdr));
    jp2h_payload.extend_from_slice(&jp2_box(*b"colr", &colr));
    jp2h_payload.extend_from_slice(&jp2_box(*b"cdef", &cdef));
    jp2h_payload
}

fn jp2_with_header_payload(codestream: &[u8], jp2h_payload: &[u8]) -> Vec<u8> {
    jp2_with_header_payload_and_file_type(codestream, jp2h_payload, b"jp2 \0\0\0\0jp2 ")
}

fn jp2_with_header_payload_and_file_type(
    codestream: &[u8],
    jp2h_payload: &[u8],
    ftyp_payload: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&jp2_box(*b"jP  ", &[0x0d, 0x0a, 0x87, 0x0a]));
    out.extend_from_slice(&jp2_box(*b"ftyp", ftyp_payload));
    out.extend_from_slice(&jp2_box(*b"jp2h", jp2h_payload));
    out.extend_from_slice(&jp2_box(*b"jp2c", codestream));
    out
}
