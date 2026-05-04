use signinum_jpeg::{
    encode_jpeg_baseline, DecodeOptions, Decoder, EncodedJpeg, JpegBackend, JpegEncodeOptions,
    JpegSamples, JpegSubsampling, PixelFormat,
};
use std::io::Cursor;

fn patterned_rgb(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 17 + y * 3) & 0xFF) as u8);
            pixels.push(((x * 5 + y * 11 + 40) & 0xFF) as u8);
            pixels.push(((x * 13 + y * 7 + 90) & 0xFF) as u8);
        }
    }
    pixels
}

fn patterned_gray(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 19 + y * 23) & 0xFF) as u8);
        }
    }
    pixels
}

fn encode_rgb(subsampling: JpegSubsampling) -> EncodedJpeg {
    let width = 19;
    let height = 17;
    let rgb = patterned_rgb(width, height);
    encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width,
            height,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode RGB baseline JPEG")
}

fn assert_independent_decoder_accepts(
    encoded: &[u8],
    width: u32,
    height: u32,
    expected_format: jpeg_decoder::PixelFormat,
) {
    let mut decoder = jpeg_decoder::Decoder::new(Cursor::new(encoded));
    let decoded = decoder.decode().expect("jpeg-decoder accepts encoded JPEG");
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

#[test]
fn cpu_encoder_round_trips_rgb_444_422_420() {
    for subsampling in [
        JpegSubsampling::Ybr444,
        JpegSubsampling::Ybr422,
        JpegSubsampling::Ybr420,
    ] {
        let encoded = encode_rgb(subsampling);
        assert_eq!(encoded.backend, JpegBackend::Cpu);
        assert!(encoded.data.starts_with(&[0xFF, 0xD8]));
        assert!(encoded.data.ends_with(&[0xFF, 0xD9]));

        let decoder = Decoder::new_with_options(&encoded.data, DecodeOptions::default())
            .expect("parse encoded JPEG");
        let (decoded, outcome) = decoder.decode(PixelFormat::Rgb8).expect("decode RGB JPEG");

        assert_eq!((outcome.decoded.w, outcome.decoded.h), (19, 17));
        assert_eq!(decoded.len(), 19 * 17 * 3);
        assert_independent_decoder_accepts(&encoded.data, 19, 17, jpeg_decoder::PixelFormat::RGB24);
    }
}

#[test]
fn cpu_encoder_round_trips_gray_and_writes_required_markers() {
    let width = 13;
    let height = 11;
    let gray = patterned_gray(width, height);
    let encoded = encode_jpeg_baseline(
        JpegSamples::Gray8 {
            data: &gray,
            width,
            height,
        },
        JpegEncodeOptions {
            quality: 85,
            subsampling: JpegSubsampling::Gray,
            restart_interval: Some(4),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode gray JPEG");

    for marker in [
        [0xFF, 0xDB],
        [0xFF, 0xC4],
        [0xFF, 0xC0],
        [0xFF, 0xDA],
        [0xFF, 0xDD],
    ] {
        assert!(
            encoded.data.windows(2).any(|window| window == marker),
            "missing marker {:02X}{:02X}",
            marker[0],
            marker[1]
        );
    }
    assert!(
        !encoded
            .data
            .windows(3)
            .any(|window| window[0] == 0xFF && window[1] == 0xFF && window[2] != 0x00),
        "entropy/header should not contain unstuffed fill-marker pairs"
    );

    let decoder = Decoder::new_with_options(&encoded.data, DecodeOptions::default())
        .expect("parse encoded gray JPEG");
    let (decoded, outcome) = decoder
        .decode(PixelFormat::Gray8)
        .expect("decode gray JPEG");

    assert_eq!((outcome.decoded.w, outcome.decoded.h), (width, height));
    assert_eq!(decoded.len(), width as usize * height as usize);
    assert_independent_decoder_accepts(&encoded.data, width, height, jpeg_decoder::PixelFormat::L8);
}
