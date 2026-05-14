// SPDX-License-Identifier: Apache-2.0

use signinum_core::{
    BufferError, DecoderContext, Downscale, ImageDecodeRows, PixelFormat, Rect, RowSink,
    TileBatchDecode,
};
use signinum_j2k::{J2kCodec, J2kContext, J2kDecoder, J2kError};
use signinum_j2k_native::{encode, encode_htj2k, DecodeSettings, EncodeOptions, Image};

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

fn backend_decode_u8_region(bytes: &[u8], roi: Rect) -> Vec<u8> {
    let mut context = signinum_j2k_native::DecoderContext::default();
    Image::new(bytes, &DecodeSettings::default())
        .expect("backend image")
        .decode_region_with_context((roi.x, roi.y, roi.w, roi.h), &mut context)
        .expect("backend region decode")
        .data
}

fn locally_inspectable_codestream_without_decode_headers() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xFF, 0x4F]);

    bytes.extend_from_slice(&[0xFF, 0x51]);
    bytes.extend_from_slice(&41_u16.to_be_bytes());
    bytes.extend_from_slice(&0_u16.to_be_bytes());
    bytes.extend_from_slice(&2_u32.to_be_bytes());
    bytes.extend_from_slice(&2_u32.to_be_bytes());
    bytes.extend_from_slice(&0_u32.to_be_bytes());
    bytes.extend_from_slice(&0_u32.to_be_bytes());
    bytes.extend_from_slice(&2_u32.to_be_bytes());
    bytes.extend_from_slice(&2_u32.to_be_bytes());
    bytes.extend_from_slice(&0_u32.to_be_bytes());
    bytes.extend_from_slice(&0_u32.to_be_bytes());
    bytes.extend_from_slice(&1_u16.to_be_bytes());
    bytes.extend_from_slice(&[7, 1, 1]);

    bytes.extend_from_slice(&[0xFF, 0x52]);
    bytes.extend_from_slice(&12_u16.to_be_bytes());
    bytes.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);

    bytes.extend_from_slice(&[0xFF, 0x90]);
    bytes
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

fn crop_bytes(full: &[u8], full_width: usize, bytes_per_pixel: usize, roi: Rect) -> Vec<u8> {
    let mut out = Vec::with_capacity(roi.w as usize * roi.h as usize * bytes_per_pixel);
    let row_bytes = full_width * bytes_per_pixel;
    let roi_row_bytes = roi.w as usize * bytes_per_pixel;
    for y in roi.y as usize..(roi.y + roi.h) as usize {
        let start = y * row_bytes + roi.x as usize * bytes_per_pixel;
        out.extend_from_slice(&full[start..start + roi_row_bytes]);
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
    assert_eq!(outcome.decoded, signinum_core::Rect::full((2, 2)));
    assert_eq!(out, expected.as_slice());
}

#[test]
fn decoder_new_rejects_codestream_that_only_header_inspection_accepts() {
    let malformed = locally_inspectable_codestream_without_decode_headers();

    J2kDecoder::inspect(&malformed).expect("header inspection still succeeds");
    let Err(err) = J2kDecoder::new(&malformed) else {
        panic!("decoder construction must validate backend");
    };

    assert!(
        matches!(err, J2kError::Backend(_)),
        "expected backend construction error, got {err:?}"
    );
}

#[test]
fn decoder_reuses_native_context_across_multiple_decode_calls() {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let codestream = encode_codestream(&pixels, 2, 2, 3, 8, true);
    let expected = backend_decode_u8(&codestream);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");

    let mut full = [0_u8; 12];
    decoder
        .decode_into(&mut full, 2 * 3, PixelFormat::Rgb8)
        .expect("first decode");
    assert_eq!(full, expected.as_slice());

    let mut scaled = [0_u8; 3];
    decoder
        .decode_scaled_into(
            &mut signinum_j2k::J2kScratchPool::new(),
            &mut scaled,
            3,
            PixelFormat::Rgb8,
            Downscale::Half,
        )
        .expect("scaled decode");

    let mut second = [0_u8; 12];
    decoder
        .decode_into(&mut second, 2 * 3, PixelFormat::Rgb8)
        .expect("second decode");
    assert_eq!(second, expected.as_slice());
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
    let mut pool = signinum_j2k::J2kScratchPool::new();
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
    let mut pool = signinum_j2k::J2kScratchPool::new();
    let mut out = [0_u8; 4];
    let outcome = decoder
        .decode_region_into(&mut pool, &mut out, 2, PixelFormat::Gray8, roi)
        .expect("region decode");
    assert_eq!(outcome.decoded, roi);
    assert_eq!(out, expected.as_slice());
}

#[test]
fn decode_region_scaled_into_matches_cropping_scaled_decode_for_supported_formats() {
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let scale = Downscale::Half;
    let scaled_roi = roi.scaled_covering(scale);

    let rgb8_pixels: Vec<u8> = (0_u8..48).collect();
    let rgb8_codestream = encode_codestream(&rgb8_pixels, 4, 4, 3, 8, true);
    for fmt in [PixelFormat::Rgb8, PixelFormat::Rgba8] {
        let mut scaled_decoder = J2kDecoder::new(&rgb8_codestream).expect("scaled decoder");
        let scaled_stride = 2 * fmt.bytes_per_pixel();
        let mut scaled = vec![0_u8; scaled_stride * 2];
        scaled_decoder
            .decode_scaled_into(
                &mut signinum_j2k::J2kScratchPool::new(),
                &mut scaled,
                scaled_stride,
                fmt,
                scale,
            )
            .expect("scaled decode");
        let expected = crop_bytes(&scaled, 2, fmt.bytes_per_pixel(), scaled_roi);

        let mut decoder = J2kDecoder::new(&rgb8_codestream).expect("decoder");
        let stride = scaled_roi.w as usize * fmt.bytes_per_pixel();
        let mut out = vec![0_u8; stride * scaled_roi.h as usize];
        let outcome = decoder
            .decode_region_scaled_into(
                &mut signinum_j2k::J2kScratchPool::new(),
                &mut out,
                stride,
                fmt,
                roi,
                scale,
            )
            .expect("region scaled decode");
        assert_eq!(outcome.decoded, scaled_roi);
        assert_eq!(out, expected, "format {fmt:?}");
    }

    let gray8_pixels: Vec<u8> = (0_u8..16).collect();
    let gray8_codestream = encode_codestream(&gray8_pixels, 4, 4, 1, 8, true);
    let mut gray8_scaled_decoder =
        J2kDecoder::new(&gray8_codestream).expect("gray8 scaled decoder");
    let mut gray8_scaled = vec![0_u8; 2 * 2];
    gray8_scaled_decoder
        .decode_scaled_into(
            &mut signinum_j2k::J2kScratchPool::new(),
            &mut gray8_scaled,
            2,
            PixelFormat::Gray8,
            scale,
        )
        .expect("gray8 scaled decode");
    let expected_gray8 = crop_bytes(
        &gray8_scaled,
        2,
        PixelFormat::Gray8.bytes_per_pixel(),
        scaled_roi,
    );
    let mut gray8_decoder = J2kDecoder::new(&gray8_codestream).expect("gray8 decoder");
    let mut gray8_out = vec![0_u8; scaled_roi.w as usize * scaled_roi.h as usize];
    let gray8_stride = scaled_roi.w as usize * PixelFormat::Gray8.bytes_per_pixel();
    let outcome = gray8_decoder
        .decode_region_scaled_into(
            &mut signinum_j2k::J2kScratchPool::new(),
            &mut gray8_out,
            gray8_stride,
            PixelFormat::Gray8,
            roi,
            scale,
        )
        .expect("gray8 region scaled decode");
    assert_eq!(outcome.decoded, scaled_roi);
    assert_eq!(gray8_out, expected_gray8);

    let gray16_samples = [
        0_u16, 64, 128, 192, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 3072, 4095,
    ];
    let gray16_pixels: Vec<u8> = gray16_samples
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    let gray16_codestream = encode_codestream(&gray16_pixels, 4, 4, 1, 12, true);
    let mut gray16_scaled_decoder =
        J2kDecoder::new(&gray16_codestream).expect("gray16 scaled decoder");
    let mut gray16_scaled = vec![0_u8; 2 * 2 * 2];
    gray16_scaled_decoder
        .decode_scaled_into(
            &mut signinum_j2k::J2kScratchPool::new(),
            &mut gray16_scaled,
            2 * 2,
            PixelFormat::Gray16,
            scale,
        )
        .expect("gray16 scaled decode");
    let expected_gray16 = crop_bytes(
        &gray16_scaled,
        2,
        PixelFormat::Gray16.bytes_per_pixel(),
        scaled_roi,
    );
    let mut gray16_decoder = J2kDecoder::new(&gray16_codestream).expect("gray16 decoder");
    let mut gray16_out = vec![0_u8; scaled_roi.w as usize * scaled_roi.h as usize * 2];
    let gray16_stride = scaled_roi.w as usize * PixelFormat::Gray16.bytes_per_pixel();
    let outcome = gray16_decoder
        .decode_region_scaled_into(
            &mut signinum_j2k::J2kScratchPool::new(),
            &mut gray16_out,
            gray16_stride,
            PixelFormat::Gray16,
            roi,
            scale,
        )
        .expect("gray16 region scaled decode");
    assert_eq!(outcome.decoded, scaled_roi);
    assert_eq!(gray16_out, expected_gray16);

    let rgb16_samples = [
        0_u16, 1, 2, 64, 65, 66, 128, 129, 130, 192, 193, 194, 256, 257, 258, 512, 513, 514, 768,
        769, 770, 1024, 1025, 1026, 1280, 1281, 1282, 1536, 1537, 1538, 1792, 1793, 1794, 2048,
        2049, 2050, 2304, 2305, 2306, 2560, 2561, 2562, 3072, 3073, 3074, 4093, 4094, 4095,
    ];
    let rgb16_pixels: Vec<u8> = rgb16_samples
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    let rgb16_codestream = encode_codestream(&rgb16_pixels, 4, 4, 3, 12, true);
    let mut rgb16_scaled_decoder =
        J2kDecoder::new(&rgb16_codestream).expect("rgb16 scaled decoder");
    let mut rgb16_scaled = vec![0_u8; 2 * 2 * 3 * 2];
    rgb16_scaled_decoder
        .decode_scaled_into(
            &mut signinum_j2k::J2kScratchPool::new(),
            &mut rgb16_scaled,
            2 * 3 * 2,
            PixelFormat::Rgb16,
            scale,
        )
        .expect("rgb16 scaled decode");
    let expected_rgb16 = crop_bytes(
        &rgb16_scaled,
        2,
        PixelFormat::Rgb16.bytes_per_pixel(),
        scaled_roi,
    );
    let mut rgb16_decoder = J2kDecoder::new(&rgb16_codestream).expect("rgb16 decoder");
    let mut rgb16_out = vec![0_u8; scaled_roi.w as usize * scaled_roi.h as usize * 3 * 2];
    let rgb16_stride = scaled_roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let outcome = rgb16_decoder
        .decode_region_scaled_into(
            &mut signinum_j2k::J2kScratchPool::new(),
            &mut rgb16_out,
            rgb16_stride,
            PixelFormat::Rgb16,
            roi,
            scale,
        )
        .expect("rgb16 region scaled decode");
    assert_eq!(outcome.decoded, scaled_roi);
    assert_eq!(rgb16_out, expected_rgb16);
}

#[test]
fn decode_region_scaled_htj2k_gray8_matches_cropping_scaled_decode() {
    let pixels: Vec<u8> = (0_u8..16).collect();
    let codestream = encode_ht_codestream(&pixels, 4, 4, 1, 8);
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let scale = Downscale::Half;
    let scaled_roi = roi.scaled_covering(scale);

    let mut scaled_decoder = J2kDecoder::new(&codestream).expect("scaled decoder");
    let mut scaled = vec![0_u8; 2 * 2];
    scaled_decoder
        .decode_scaled_into(
            &mut signinum_j2k::J2kScratchPool::new(),
            &mut scaled,
            2,
            PixelFormat::Gray8,
            scale,
        )
        .expect("scaled decode");
    let expected = crop_bytes(&scaled, 2, PixelFormat::Gray8.bytes_per_pixel(), scaled_roi);

    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let stride = scaled_roi.w as usize * PixelFormat::Gray8.bytes_per_pixel();
    let mut out = vec![0_u8; stride * scaled_roi.h as usize];
    let outcome = decoder
        .decode_region_scaled_into(
            &mut signinum_j2k::J2kScratchPool::new(),
            &mut out,
            stride,
            PixelFormat::Gray8,
            roi,
            scale,
        )
        .expect("region scaled decode");

    assert_eq!(outcome.decoded, scaled_roi);
    assert_eq!(out, expected);
}

#[test]
fn decode_region_scaled_none_matches_region_decode() {
    let pixels = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8];
    let codestream = encode_codestream(&pixels, 3, 3, 1, 8, true);
    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };

    let mut expected_decoder = J2kDecoder::new(&codestream).expect("expected decoder");
    let mut expected = [0_u8; 4];
    expected_decoder
        .decode_region_into(
            &mut signinum_j2k::J2kScratchPool::new(),
            &mut expected,
            2,
            PixelFormat::Gray8,
            roi,
        )
        .expect("region decode");

    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 4];
    let outcome = decoder
        .decode_region_scaled_into(
            &mut signinum_j2k::J2kScratchPool::new(),
            &mut out,
            2,
            PixelFormat::Gray8,
            roi,
            Downscale::None,
        )
        .expect("region scaled none decode");

    assert_eq!(outcome.decoded, roi);
    assert_eq!(out, expected);
}

#[test]
fn native_backend_region_decode_matches_cropping_full_decode() {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let codestream = encode_codestream(&pixels, 2, 2, 3, 8, true);
    let roi = Rect {
        x: 1,
        y: 0,
        w: 1,
        h: 2,
    };
    let expected = crop_u8(&backend_decode_u8(&codestream), 2, 3, roi);
    let actual = backend_decode_u8_region(&codestream, roi);
    assert_eq!(actual, expected);
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
    let mut pool = signinum_j2k::J2kScratchPool::new();
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
    assert_eq!(ctx.cache_stats().misses, 1);
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
    let mut pool = signinum_j2k::J2kScratchPool::new();
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
    let mut pool = signinum_j2k::J2kScratchPool::new();
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
fn tile_batch_region_scaled_decode_matches_decoder_region_scaled_decode() {
    let pixels: Vec<u8> = (0_u8..48).collect();
    let codestream = encode_codestream(&pixels, 4, 4, 3, 8, true);
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let scale = Downscale::Half;
    let scaled_roi = roi.scaled_covering(scale);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel();

    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut pool = signinum_j2k::J2kScratchPool::new();
    let mut expected = vec![0_u8; stride * scaled_roi.h as usize];
    decoder
        .decode_region_scaled_into(
            &mut pool,
            &mut expected,
            stride,
            PixelFormat::Rgb8,
            roi,
            scale,
        )
        .expect("decoder region scaled");

    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut out = vec![0_u8; stride * scaled_roi.h as usize];
    let outcome = <J2kCodec as TileBatchDecode>::decode_tile_region_scaled(
        &mut ctx,
        &mut pool,
        &codestream,
        &mut out,
        stride,
        PixelFormat::Rgb8,
        roi,
        scale,
    )
    .expect("tile region scaled");
    assert_eq!(outcome.decoded, scaled_roi);
    assert_eq!(out, expected);
}

#[test]
fn decode_region_into_rejects_out_of_bounds_roi() {
    let pixels = [0_u8, 1, 2, 3];
    let codestream = encode_codestream(&pixels, 2, 2, 1, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut pool = signinum_j2k::J2kScratchPool::new();
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
fn decode_htj2k_scaled_into_matches_native_target_resolution_decode() {
    let pixels: Vec<u8> = (0_u8..16).collect();
    let codestream = encode_ht_codestream(&pixels, 4, 4, 1, 8);
    let expected = backend_decode_u8_scaled(&codestream, (2, 2));
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut pool = signinum_j2k::J2kScratchPool::new();
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
    let mut pool = signinum_j2k::J2kScratchPool::new();
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
