// SPDX-License-Identifier: Apache-2.0

use dicom_toolkit_jpeg2000::{encode, encode_htj2k, DecodeSettings, EncodeOptions, Image};
use slidecodec_core::{
    BufferError, DecoderContext, Downscale, ImageDecodeRows, PixelFormat, Rect, RowSink,
    TileBatchDecode,
};
use slidecodec_j2k::{J2kCodec, J2kContext, J2kDecoder, J2kError};

fn encode_codestream(
    pixels: &[u8],
    width: u32,
    height: u32,
    components: u8,
    bit_depth: u8,
    reversible: bool,
) -> Vec<u8> {
    let options = EncodeOptions {
        reversible,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(
        pixels, width, height, components, bit_depth, false, &options,
    )
    .expect("encode")
}

fn encode_ht_codestream(
    pixels: &[u8],
    width: u32,
    height: u32,
    components: u8,
    bit_depth: u8,
) -> Vec<u8> {
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode_htj2k(
        pixels, width, height, components, bit_depth, false, &options,
    )
    .expect("encode ht")
}

fn wrap_codestream_jp2(
    codestream: &[u8],
    width: u32,
    height: u32,
    components: u16,
    bit_depth: u8,
    colorspace_enum: u32,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0D, 0x0A, 0x87, 0x0A]);
    bytes.extend_from_slice(&[
        0, 0, 0, 20, b'f', b't', b'y', b'p', b'j', b'p', b'2', b' ', 0, 0, 0, 0, b'j', b'p', b'2',
        b' ',
    ]);

    let bpc = bit_depth.saturating_sub(1);
    bytes.extend_from_slice(&[
        0, 0, 0, 45, b'j', b'p', b'2', b'h', 0, 0, 0, 22, b'i', b'h', b'd', b'r',
    ]);
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&components.to_be_bytes());
    bytes.extend_from_slice(&[bpc, 7, 0, 0]);
    bytes.extend_from_slice(&[0, 0, 0, 15, b'c', b'o', b'l', b'r', 1, 0, 0]);
    bytes.extend_from_slice(&colorspace_enum.to_be_bytes());

    let len = (8 + codestream.len()) as u32;
    bytes.extend_from_slice(&len.to_be_bytes());
    bytes.extend_from_slice(b"jp2c");
    bytes.extend_from_slice(codestream);
    bytes
}

fn backend_decode_u8(bytes: &[u8]) -> Vec<u8> {
    Image::new(bytes, &DecodeSettings::default())
        .expect("backend image")
        .decode()
        .expect("backend decode")
}

fn backend_decode_u8_scaled(bytes: &[u8], target_resolution: (u32, u32)) -> Vec<u8> {
    let settings = DecodeSettings {
        target_resolution: Some(target_resolution),
        ..DecodeSettings::default()
    };
    Image::new(bytes, &settings)
        .expect("backend image")
        .decode()
        .expect("backend decode")
}

fn crop_u8(full: &[u8], full_width: usize, channels: usize, roi: Rect) -> Vec<u8> {
    let mut out = Vec::with_capacity(roi.w as usize * roi.h as usize * channels);
    let row_bytes = full_width * channels;
    let roi_row_bytes = roi.w as usize * channels;
    for y in roi.y as usize..(roi.y + roi.h) as usize {
        let start = y * row_bytes + roi.x as usize * channels;
        out.extend_from_slice(&full[start..start + roi_row_bytes]);
    }
    out
}

fn decimate_u8(
    full: &[u8],
    full_width: usize,
    full_height: usize,
    channels: usize,
    denom: usize,
) -> Vec<u8> {
    let out_width = full_width.div_ceil(denom);
    let out_height = full_height.div_ceil(denom);
    let mut out = Vec::with_capacity(out_width * out_height * channels);
    let row_bytes = full_width * channels;
    for y in (0..full_height).step_by(denom) {
        let row = &full[y * row_bytes..(y + 1) * row_bytes];
        for x in (0..full_width).step_by(denom) {
            let start = x * channels;
            out.extend_from_slice(&row[start..start + channels]);
        }
    }
    out
}

#[derive(Default)]
struct CollectRowsU8 {
    rows: Vec<u8>,
}

impl RowSink<u8> for CollectRowsU8 {
    type Error = J2kError;

    fn write_row(&mut self, _y: u32, row: &[u8]) -> Result<(), Self::Error> {
        self.rows.extend_from_slice(row);
        Ok(())
    }
}

#[derive(Default)]
struct CollectRowsU16 {
    rows: Vec<u16>,
}

impl RowSink<u16> for CollectRowsU16 {
    type Error = J2kError;

    fn write_row(&mut self, _y: u32, row: &[u16]) -> Result<(), Self::Error> {
        self.rows.extend_from_slice(row);
        Ok(())
    }
}

#[test]
fn decode_rgb8_codestream_roundtrips_reversible_pixels() {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let codestream = encode_codestream(&pixels, 2, 2, 3, 8, true);
    let expected = backend_decode_u8(&codestream);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 12];
    let outcome = decoder
        .decode_into(&mut out, 2 * 3, PixelFormat::Rgb8)
        .expect("decode");
    assert_eq!(outcome.decoded, slidecodec_core::Rect::full((2, 2)));
    assert_eq!(out, expected.as_slice());
}

#[test]
fn decode_rgba8_fills_opaque_alpha_for_rgb_source() {
    let pixels = [1, 2, 3, 4, 5, 6];
    let codestream = encode_codestream(&pixels, 2, 1, 3, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 8];
    decoder
        .decode_into(&mut out, 2 * 4, PixelFormat::Rgba8)
        .expect("decode");
    assert_eq!(out, [1, 2, 3, 255, 4, 5, 6, 255]);
}

#[test]
fn decode_gray8_jp2_roundtrips_reversible_pixels() {
    let pixels = [3, 9, 27, 81];
    let codestream = encode_codestream(&pixels, 2, 2, 1, 8, true);
    let jp2 = wrap_codestream_jp2(&codestream, 2, 2, 1, 8, 17);
    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = [0_u8; 4];
    decoder
        .decode_into(&mut out, 2, PixelFormat::Gray8)
        .expect("decode");
    assert_eq!(out, pixels);
}

#[test]
fn decode_gray16_roundtrips_native_samples() {
    let samples = [0_u16, 1024, 2048, 4095];
    let pixels: Vec<u8> = samples.into_iter().flat_map(u16::to_le_bytes).collect();
    let codestream = encode_codestream(&pixels, 2, 2, 1, 12, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 8];
    decoder
        .decode_into(&mut out, 2 * 2, PixelFormat::Gray16)
        .expect("decode");
    assert_eq!(out, pixels.as_slice());
}

#[test]
fn decode_gray16_widens_8bit_samples_to_full_u16_range() {
    let pixels = [0_u8, 64, 128, 255];
    let codestream = encode_codestream(&pixels, 2, 2, 1, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 8];
    decoder
        .decode_into(&mut out, 2 * 2, PixelFormat::Gray16)
        .expect("decode");
    let expected: Vec<u8> = [0_u16, 16448, 32896, 65535]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    assert_eq!(out, expected.as_slice());
}

#[test]
fn decode_rgb16_roundtrips_native_samples() {
    let samples = [0_u16, 1, 2, 1024, 2048, 3072];
    let pixels: Vec<u8> = samples.into_iter().flat_map(u16::to_le_bytes).collect();
    let codestream = encode_codestream(&pixels, 2, 1, 3, 12, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 12];
    decoder
        .decode_into(&mut out, 2 * 3 * 2, PixelFormat::Rgb16)
        .expect("decode");
    assert_eq!(out, pixels.as_slice());
}

#[test]
fn decode_rejects_unsupported_rgba16_output() {
    let pixels = [1, 2, 3, 4, 5, 6];
    let codestream = encode_codestream(&pixels, 2, 1, 3, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 16];
    let err = decoder
        .decode_into(&mut out, 2 * 4 * 2, PixelFormat::Rgba16)
        .unwrap_err();
    assert!(matches!(err, J2kError::Unsupported(_)));
}

#[test]
fn decode_rejects_small_output_buffer() {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let codestream = encode_codestream(&pixels, 2, 2, 3, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 11];
    let err = decoder
        .decode_into(&mut out, 6, PixelFormat::Rgb8)
        .unwrap_err();
    assert!(matches!(
        err,
        J2kError::Buffer(BufferError::OutputTooSmall { .. })
    ));
}

#[test]
fn decode_rejects_too_small_stride() {
    let pixels = [10, 20, 30, 40, 50, 60];
    let codestream = encode_codestream(&pixels, 2, 1, 3, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 6];
    let err = decoder
        .decode_into(&mut out, 5, PixelFormat::Rgb8)
        .unwrap_err();
    assert!(matches!(
        err,
        J2kError::Buffer(BufferError::StrideTooSmall { .. })
    ));
}

#[test]
fn decode_scaled_into_matches_backend_target_resolution_decode() {
    let pixels: Vec<u8> = (0_u8..48).collect();
    let codestream = encode_codestream(&pixels, 4, 4, 3, 8, true);
    let expected = backend_decode_u8_scaled(&codestream, (2, 2));
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut pool = slidecodec_j2k::J2kScratchPool::new();
    let mut out = [0_u8; 12];
    let outcome = decoder
        .decode_scaled_into(
            &mut pool,
            &mut out,
            2 * 3,
            PixelFormat::Rgb8,
            Downscale::Half,
        )
        .expect("scaled decode");
    assert_eq!(outcome.decoded, Rect::full((2, 2)));
    assert_eq!(out, expected.as_slice());
}

#[test]
fn decode_region_into_matches_cropping_full_decode() {
    let pixels = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8];
    let codestream = encode_codestream(&pixels, 3, 3, 1, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut full = [0_u8; 9];
    decoder
        .decode_into(&mut full, 3, PixelFormat::Gray8)
        .expect("full decode");

    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    let expected = crop_u8(&full, 3, 1, roi);
    let mut pool = slidecodec_j2k::J2kScratchPool::new();
    let mut out = [0_u8; 4];
    let outcome = decoder
        .decode_region_into(&mut pool, &mut out, 2, PixelFormat::Gray8, roi)
        .expect("region decode");
    assert_eq!(outcome.decoded, roi);
    assert_eq!(out, expected.as_slice());
}

#[test]
fn decode_rows_u8_matches_full_rgb8_decode() {
    let pixels = [10_u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let codestream = encode_codestream(&pixels, 2, 2, 3, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut full = [0_u8; 12];
    decoder
        .decode_into(&mut full, 2 * 3, PixelFormat::Rgb8)
        .expect("full decode");

    let mut sink = CollectRowsU8::default();
    <J2kDecoder<'_> as ImageDecodeRows<'_, u8>>::decode_rows(&mut decoder, &mut sink)
        .expect("row decode");
    assert_eq!(sink.rows, full);
}

#[test]
fn decode_rows_u16_matches_full_gray16_decode() {
    let samples = [0_u16, 1024, 2048, 4095];
    let pixels: Vec<u8> = samples.into_iter().flat_map(u16::to_le_bytes).collect();
    let codestream = encode_codestream(&pixels, 2, 2, 1, 12, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut full = [0_u8; 8];
    decoder
        .decode_into(&mut full, 2 * 2, PixelFormat::Gray16)
        .expect("full decode");

    let mut sink = CollectRowsU16::default();
    <J2kDecoder<'_> as ImageDecodeRows<'_, u16>>::decode_rows(&mut decoder, &mut sink)
        .expect("row decode");
    let collected: Vec<u8> = sink.rows.into_iter().flat_map(u16::to_le_bytes).collect();
    assert_eq!(collected, full);
}

#[test]
fn tile_batch_decode_matches_borrowed_decoder_decode() {
    let pixels = [10_u8, 20, 30, 40, 50, 60];
    let codestream = encode_codestream(&pixels, 2, 1, 3, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut expected = [0_u8; 6];
    decoder
        .decode_into(&mut expected, 2 * 3, PixelFormat::Rgb8)
        .expect("decoder decode");

    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut pool = slidecodec_j2k::J2kScratchPool::new();
    let mut out = [0_u8; 6];
    let outcome = <J2kCodec as TileBatchDecode>::decode_tile(
        &mut ctx,
        &mut pool,
        &codestream,
        &mut out,
        2 * 3,
        PixelFormat::Rgb8,
    )
    .expect("tile decode");
    assert_eq!(outcome.decoded, Rect::full((2, 1)));
    assert_eq!(out, expected);
}

#[test]
fn tile_batch_region_decode_matches_decoder_region_decode() {
    let pixels = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8];
    let codestream = encode_codestream(&pixels, 3, 3, 1, 8, true);
    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut pool = slidecodec_j2k::J2kScratchPool::new();
    let mut expected = [0_u8; 4];
    decoder
        .decode_region_into(&mut pool, &mut expected, 2, PixelFormat::Gray8, roi)
        .expect("decoder region");

    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut out = [0_u8; 4];
    <J2kCodec as TileBatchDecode>::decode_tile_region(
        &mut ctx,
        &mut pool,
        &codestream,
        &mut out,
        2,
        PixelFormat::Gray8,
        roi,
    )
    .expect("tile region");
    assert_eq!(out, expected);
}

#[test]
fn tile_batch_scaled_decode_matches_decoder_scaled_decode() {
    let pixels: Vec<u8> = (0_u8..48).collect();
    let codestream = encode_codestream(&pixels, 4, 4, 3, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut pool = slidecodec_j2k::J2kScratchPool::new();
    let mut expected = [0_u8; 12];
    decoder
        .decode_scaled_into(
            &mut pool,
            &mut expected,
            2 * 3,
            PixelFormat::Rgb8,
            Downscale::Half,
        )
        .expect("decoder scaled");

    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut out = [0_u8; 12];
    <J2kCodec as TileBatchDecode>::decode_tile_scaled(
        &mut ctx,
        &mut pool,
        &codestream,
        &mut out,
        2 * 3,
        PixelFormat::Rgb8,
        Downscale::Half,
    )
    .expect("tile scaled");
    assert_eq!(out, expected);
}

#[test]
fn decode_region_into_rejects_out_of_bounds_roi() {
    let pixels = [0_u8, 1, 2, 3];
    let codestream = encode_codestream(&pixels, 2, 2, 1, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut pool = slidecodec_j2k::J2kScratchPool::new();
    let mut out = [0_u8; 4];
    let err = decoder
        .decode_region_into(
            &mut pool,
            &mut out,
            2,
            PixelFormat::Gray8,
            Rect {
                x: 1,
                y: 1,
                w: 2,
                h: 2,
            },
        )
        .unwrap_err();
    assert!(matches!(err, J2kError::InvalidRegion { .. }));
}

#[test]
fn decode_htj2k_gray8_roundtrips_reversible_pixels() {
    let pixels = [3_u8, 9, 27, 81];
    let codestream = encode_ht_codestream(&pixels, 2, 2, 1, 8);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 4];
    decoder
        .decode_into(&mut out, 2, PixelFormat::Gray8)
        .expect("ht decode");
    assert_eq!(out, pixels);
}

#[test]
fn decode_htj2k_scaled_into_matches_full_decode_decimation() {
    let pixels: Vec<u8> = (0_u8..16).collect();
    let codestream = encode_ht_codestream(&pixels, 4, 4, 1, 8);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut full = [0_u8; 16];
    decoder
        .decode_into(&mut full, 4, PixelFormat::Gray8)
        .expect("full decode");
    let expected = decimate_u8(&full, 4, 4, 1, 2);
    let mut pool = slidecodec_j2k::J2kScratchPool::new();
    let mut out = [0_u8; 4];
    decoder
        .decode_scaled_into(&mut pool, &mut out, 2, PixelFormat::Gray8, Downscale::Half)
        .expect("scaled decode");
    assert_eq!(out, expected.as_slice());
}

#[test]
fn decode_rows_u8_matches_full_gray8_decode_for_htj2k() {
    let pixels = [2_u8, 4, 6, 8];
    let codestream = encode_ht_codestream(&pixels, 2, 2, 1, 8);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut full = [0_u8; 4];
    decoder
        .decode_into(&mut full, 2, PixelFormat::Gray8)
        .expect("full decode");

    let mut sink = CollectRowsU8::default();
    <J2kDecoder<'_> as ImageDecodeRows<'_, u8>>::decode_rows(&mut decoder, &mut sink)
        .expect("row decode");
    assert_eq!(sink.rows, full);
}

#[test]
fn tile_batch_decode_matches_borrowed_decoder_for_htj2k() {
    let pixels = [7_u8, 11, 13, 17];
    let codestream = encode_ht_codestream(&pixels, 2, 2, 1, 8);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut expected = [0_u8; 4];
    decoder
        .decode_into(&mut expected, 2, PixelFormat::Gray8)
        .expect("decoder decode");

    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut pool = slidecodec_j2k::J2kScratchPool::new();
    let mut out = [0_u8; 4];
    <J2kCodec as TileBatchDecode>::decode_tile(
        &mut ctx,
        &mut pool,
        &codestream,
        &mut out,
        2,
        PixelFormat::Gray8,
    )
    .expect("tile decode");
    assert_eq!(out, expected);
}
