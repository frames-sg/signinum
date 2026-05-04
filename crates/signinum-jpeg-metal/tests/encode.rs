#![allow(clippy::similar_names)]

#[cfg(target_os = "macos")]
fn assert_independent_decoder_accepts(
    encoded: &[u8],
    width: u32,
    height: u32,
    expected_format: jpeg_decoder::PixelFormat,
) {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(encoded));
    let decoded = decoder.decode().expect("jpeg-decoder accepts Metal JPEG");
    let info = decoder.info().expect("jpeg-decoder exposes frame info");
    assert_eq!(
        (u32::from(info.width), u32::from(info.height)),
        (width, height)
    );
    assert_eq!(info.pixel_format, expected_format);
    let expected_components = match expected_format {
        jpeg_decoder::PixelFormat::L8 => 1usize,
        jpeg_decoder::PixelFormat::RGB24 => 3usize,
        jpeg_decoder::PixelFormat::CMYK32 => 4usize,
        jpeg_decoder::PixelFormat::L16 => 2usize,
    };
    assert_eq!(
        decoded.len(),
        width as usize * height as usize * expected_components
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_baseline_encoder_round_trips_rgb_422() {
    use signinum_core::PixelFormat;
    use signinum_jpeg::{DecodeOptions, Decoder, JpegBackend, JpegEncodeOptions, JpegSubsampling};
    use signinum_jpeg_metal::{
        encode_jpeg_baseline_from_metal_buffer, JpegBaselineMetalEncodeTile, MetalBackendSession,
    };

    let width = 19u32;
    let height = 17u32;
    let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            rgb.push(((x * 17 + y * 3) & 0xff) as u8);
            rgb.push(((x * 5 + y * 11 + 40) & 0xff) as u8);
            rgb.push(((x * 13 + y * 7 + 90) & 0xff) as u8);
        }
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let buffer = session.device().new_buffer_with_data(
        rgb.as_ptr().cast(),
        rgb.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let encoded = encode_jpeg_baseline_from_metal_buffer(
        JpegBaselineMetalEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width,
            height,
            pitch_bytes: width as usize * 3,
            output_width: width,
            output_height: height,
            format: PixelFormat::Rgb8,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: None,
            backend: JpegBackend::Metal,
        },
        &session,
    )
    .expect("Metal JPEG baseline encode");

    assert_eq!(encoded.backend, JpegBackend::Metal);
    assert!(encoded.data.starts_with(&[0xff, 0xd8]));
    assert!(encoded.data.ends_with(&[0xff, 0xd9]));

    let decoder = Decoder::new_with_options(&encoded.data, DecodeOptions::default())
        .expect("parse Metal-encoded JPEG");
    let (decoded, outcome) = decoder
        .decode(PixelFormat::Rgb8)
        .expect("decode Metal-encoded JPEG");

    assert_eq!((outcome.decoded.w, outcome.decoded.h), (width, height));
    assert_eq!(decoded.len(), rgb.len());
    assert_independent_decoder_accepts(
        &encoded.data,
        width,
        height,
        jpeg_decoder::PixelFormat::RGB24,
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_baseline_encoder_round_trips_all_rgb_subsampling_modes() {
    use signinum_core::PixelFormat;
    use signinum_jpeg::{DecodeOptions, Decoder, JpegBackend, JpegEncodeOptions, JpegSubsampling};
    use signinum_jpeg_metal::{
        encode_jpeg_baseline_from_metal_buffer, JpegBaselineMetalEncodeTile, MetalBackendSession,
    };

    let width = 23u32;
    let height = 19u32;
    let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            rgb.push(((x * 29 + y * 3 + 11) & 0xff) as u8);
            rgb.push(((x * 7 + y * 17 + 40) & 0xff) as u8);
            rgb.push(((x * 13 + y * 5 + 90) & 0xff) as u8);
        }
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let buffer = session.device().new_buffer_with_data(
        rgb.as_ptr().cast(),
        rgb.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    for subsampling in [
        JpegSubsampling::Ybr444,
        JpegSubsampling::Ybr422,
        JpegSubsampling::Ybr420,
    ] {
        let encoded = encode_jpeg_baseline_from_metal_buffer(
            JpegBaselineMetalEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width,
                height,
                pitch_bytes: width as usize * 3,
                output_width: width,
                output_height: height,
                format: PixelFormat::Rgb8,
            },
            JpegEncodeOptions {
                quality: 88,
                subsampling,
                restart_interval: Some(5),
                backend: JpegBackend::Metal,
            },
            &session,
        )
        .expect("Metal JPEG baseline encode");

        assert_eq!(encoded.backend, JpegBackend::Metal);
        let decoder = Decoder::new_with_options(&encoded.data, DecodeOptions::default())
            .expect("parse Metal-encoded JPEG");
        let (decoded, outcome) = decoder
            .decode(PixelFormat::Rgb8)
            .expect("decode Metal-encoded JPEG");

        assert_eq!((outcome.decoded.w, outcome.decoded.h), (width, height));
        assert_eq!(decoded.len(), rgb.len());
        assert_independent_decoder_accepts(
            &encoded.data,
            width,
            height,
            jpeg_decoder::PixelFormat::RGB24,
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_baseline_encoder_round_trips_gray_with_padded_output() {
    use signinum_core::PixelFormat;
    use signinum_jpeg::{DecodeOptions, Decoder, JpegBackend, JpegEncodeOptions, JpegSubsampling};
    use signinum_jpeg_metal::{
        encode_jpeg_baseline_from_metal_buffer, JpegBaselineMetalEncodeTile, MetalBackendSession,
    };

    let width = 7u32;
    let height = 5u32;
    let output_width = 13u32;
    let output_height = 11u32;
    let mut gray = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            gray.push(((x * 19 + y * 23) & 0xff) as u8);
        }
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let buffer = session.device().new_buffer_with_data(
        gray.as_ptr().cast(),
        gray.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let encoded = encode_jpeg_baseline_from_metal_buffer(
        JpegBaselineMetalEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width,
            height,
            pitch_bytes: width as usize,
            output_width,
            output_height,
            format: PixelFormat::Gray8,
        },
        JpegEncodeOptions {
            quality: 85,
            subsampling: JpegSubsampling::Gray,
            restart_interval: Some(3),
            backend: JpegBackend::Metal,
        },
        &session,
    )
    .expect("Metal JPEG baseline encode");

    assert_eq!(encoded.backend, JpegBackend::Metal);
    let decoder = Decoder::new_with_options(&encoded.data, DecodeOptions::default())
        .expect("parse Metal-encoded gray JPEG");
    let (decoded, outcome) = decoder
        .decode(PixelFormat::Gray8)
        .expect("decode Metal-encoded gray JPEG");

    assert_eq!(
        (outcome.decoded.w, outcome.decoded.h),
        (output_width, output_height)
    );
    assert_eq!(
        decoded.len(),
        output_width as usize * output_height as usize
    );
    assert_independent_decoder_accepts(
        &encoded.data,
        output_width,
        output_height,
        jpeg_decoder::PixelFormat::L8,
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_baseline_batch_encoder_round_trips_multiple_rgb_tiles() {
    use signinum_core::PixelFormat;
    use signinum_jpeg::{DecodeOptions, Decoder, JpegBackend, JpegEncodeOptions, JpegSubsampling};
    use signinum_jpeg_metal::{
        encode_jpeg_baseline_batch_from_metal_buffers, JpegBaselineMetalEncodeTile,
        MetalBackendSession,
    };

    let width = 32u32;
    let height = 24u32;
    let tile_count_u32 = 3u32;
    let tile_count = tile_count_u32 as usize;
    let mut rgb = Vec::with_capacity(width as usize * height as usize * 3 * tile_count);
    for tile in 0..tile_count_u32 {
        for y in 0..height {
            for x in 0..width {
                rgb.push(((x * 11 + y * 7 + tile * 31) & 0xff) as u8);
                rgb.push(((x * 5 + y * 17 + tile * 19) & 0xff) as u8);
                rgb.push(((x * 23 + y * 3 + tile * 13) & 0xff) as u8);
            }
        }
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let buffer = session.device().new_buffer_with_data(
        rgb.as_ptr().cast(),
        rgb.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let tile_bytes = width as usize * height as usize * 3;
    let tiles: Vec<_> = (0..tile_count)
        .map(|tile| JpegBaselineMetalEncodeTile {
            buffer: &buffer,
            byte_offset: tile * tile_bytes,
            width,
            height,
            pitch_bytes: width as usize * 3,
            output_width: width,
            output_height: height,
            format: PixelFormat::Rgb8,
        })
        .collect();

    let encoded = encode_jpeg_baseline_batch_from_metal_buffers(
        &tiles,
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: Some(4),
            backend: JpegBackend::Metal,
        },
        &session,
    )
    .expect("Metal JPEG baseline batch encode");

    assert_eq!(encoded.len(), tile_count);
    for frame in encoded {
        assert_eq!(frame.backend, JpegBackend::Metal);
        let decoder = Decoder::new_with_options(&frame.data, DecodeOptions::default())
            .expect("parse Metal batch JPEG");
        let (decoded, outcome) = decoder
            .decode(PixelFormat::Rgb8)
            .expect("decode Metal batch JPEG");
        assert_eq!((outcome.decoded.w, outcome.decoded.h), (width, height));
        assert_eq!(decoded.len(), tile_bytes);
        assert_independent_decoder_accepts(
            &frame.data,
            width,
            height,
            jpeg_decoder::PixelFormat::RGB24,
        );
    }
}
