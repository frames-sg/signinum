use std::sync::Arc;

use slidecodec_core::{
    BackendKind, BackendRequest, DeviceSubmission, DeviceSurface, Downscale, ImageDecode,
    ImageDecodeDevice, PixelFormat, Rect, TileBatchDecodeDevice, TileBatchDecodeSubmit,
};
use slidecodec_j2k::J2kContext;
use slidecodec_j2k_metal::{
    Codec, Error, J2kDecoder, J2kScratchPool, MetalSession, MetalTileBatch,
};
use slidecodec_j2k_native::{encode, encode_htj2k, EncodeOptions};

fn fixture_rgb8() -> Vec<u8> {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 2, 3, 8, false, &options).expect("encode rgb8")
}

fn fixture_gray8() -> Vec<u8> {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode gray8")
}

fn fixture_gray8_sized(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x + y) & 0xFF) as u8);
        }
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        guard_bits: 2,
        ..EncodeOptions::default()
    };
    encode(&pixels, width, height, 1, 8, false, &options).expect("encode sized gray8")
}

fn fixture_gray8_reversed() -> Vec<u8> {
    let pixels: Vec<u8> = (0..16).rev().collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode reversed gray8")
}

fn fixture_gray12() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(8);
    for sample in [0u16, 257, 1023, 4095] {
        pixels.extend_from_slice(&sample.to_le_bytes());
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 2, 1, 12, false, &options).expect("encode gray12")
}

fn fixture_gray8_irreversible() -> Vec<u8> {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: false,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode gray8 irreversible")
}

fn fixture_rgb12() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(12);
    for sample in [0u16, 1023, 2047, 3071, 4095, 17] {
        pixels.extend_from_slice(&sample.to_le_bytes());
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 1, 3, 12, false, &options).expect("encode rgb12")
}

fn fixture_ht_gray8() -> Vec<u8> {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, 4, 4, 1, 8, false, &options).expect("encode ht gray8")
}

fn fixture_direct_rgb8() -> Vec<u8> {
    fixture_direct_rgb8_offset(0)
}

fn fixture_direct_rgb8_offset(offset: u8) -> Vec<u8> {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let pixels = pixels.map(|sample: u8| sample.saturating_add(offset));
    let options = EncodeOptions {
        reversible: false,
        guard_bits: 4,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 2, 3, 8, false, &options).expect("encode direct rgb8")
}

fn fixture_direct_rgb8_variant(seed: u8) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(8 * 8 * 3);
    for y in 0..8u8 {
        for x in 0..8u8 {
            pixels.push(seed.wrapping_add(x.wrapping_mul(17)).wrapping_add(y));
            pixels.push(seed.wrapping_add(x).wrapping_add(y.wrapping_mul(19)));
            pixels.push(
                seed.wrapping_add(x.wrapping_mul(7))
                    .wrapping_add(y.wrapping_mul(11)),
            );
        }
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 8, 8, 3, 8, false, &options).expect("encode direct rgb8 variant")
}

#[test]
fn full_classic_grayscale_decode_to_metal_matches_host_decode() {
    let bytes = fixture_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 16];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray8)
        .expect("host decode");

    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Metal)
        .expect("device decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (4, 4));
    assert_eq!(surface.as_bytes(), host.as_slice());
}

#[test]
fn full_htj2k_decode_to_metal_matches_host_decode() {
    let bytes = fixture_ht_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 16];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray8)
        .expect("host decode");

    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Metal)
        .expect("device decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (4, 4));
    assert_eq!(surface.as_bytes(), host.as_slice());
}

#[test]
fn full_irreversible_j2k_decode_to_metal_matches_host_decode() {
    let bytes = fixture_gray8_irreversible();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 16];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray8)
        .expect("host decode");

    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Metal)
        .expect("device decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (4, 4));
    assert_eq!(surface.as_bytes(), host.as_slice());
}

#[test]
fn auto_full_grayscale_prefers_cpu_for_small_classic_fixture() {
    let bytes = fixture_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Auto)
        .expect("auto decode");
    assert_eq!(surface.backend_kind(), BackendKind::Cpu);
}

#[test]
fn auto_full_htj2k_prefers_cpu_for_small_fixture() {
    let bytes = fixture_ht_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Auto)
        .expect("auto decode");
    assert_eq!(surface.backend_kind(), BackendKind::Cpu);
}

#[test]
fn auto_repeated_grayscale_keeps_short_512_batch_on_cpu() {
    let bytes = fixture_gray8_sized(512, 512);
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surfaces = decoder
        .decode_repeated_grayscale_auto_to_device(PixelFormat::Gray8, 8)
        .expect("auto repeated decode");
    assert_eq!(surfaces.len(), 8);
    assert!(surfaces
        .iter()
        .all(|surface| surface.backend_kind() == BackendKind::Cpu));
}

#[test]
fn auto_repeated_grayscale_uses_metal_for_512_batch() {
    let bytes = fixture_gray8_sized(512, 512);
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surfaces = decoder
        .decode_repeated_grayscale_auto_to_device(PixelFormat::Gray8, 16)
        .expect("auto repeated decode");
    assert_eq!(surfaces.len(), 16);
    assert!(surfaces
        .iter()
        .all(|surface| surface.backend_kind() == BackendKind::Metal));
}

#[test]
fn tile_full_grayscale_device_path_uses_metal_direct() {
    let bytes = fixture_gray8();
    let mut ctx = slidecodec_core::DecoderContext::<J2kContext>::new();
    let mut pool = J2kScratchPool::new();
    let surface = Codec::decode_tile_to_device(
        &mut ctx,
        &mut pool,
        &bytes,
        PixelFormat::Gray8,
        BackendRequest::Metal,
    )
    .expect("tile surface");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (4, 4));
}

#[test]
fn metal_surface_exposes_buffer_for_on_device_consumers() {
    let bytes = fixture_gray8();
    let mut metal_decoder = J2kDecoder::new(&bytes).expect("metal decoder");
    let metal_surface = metal_decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Metal)
        .expect("metal surface");
    let (buffer, byte_offset) = metal_surface.metal_buffer().expect("metal buffer");
    assert_eq!(byte_offset, 0);
    let buffer_len = usize::try_from(buffer.length()).expect("metal buffer length fits usize");
    assert!(buffer_len >= metal_surface.byte_len());

    let mut cpu_decoder = J2kDecoder::new(&bytes).expect("cpu decoder");
    let cpu_surface = cpu_decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Cpu)
        .expect("cpu surface");
    assert!(cpu_surface.metal_buffer().is_none());
}

#[test]
fn submitted_full_grayscale_tiles_flush_as_one_device_batch() {
    let bytes = fixture_gray8();
    let mut ctx = slidecodec_core::DecoderContext::<J2kContext>::new();
    let mut session = MetalSession::default();
    let mut pool = J2kScratchPool::new();

    let submissions = (0..3)
        .map(|_| {
            Codec::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                &bytes,
                PixelFormat::Gray8,
                BackendRequest::Metal,
            )
            .expect("submit tile")
        })
        .collect::<Vec<_>>();

    assert_eq!(
        session.submissions(),
        0,
        "submitted tile surfaces should stay queued until a wait flushes the session"
    );

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 16];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray8)
        .expect("host decode");

    for submission in submissions {
        let surface = submission.wait().expect("surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.as_bytes(), host.as_slice());
    }
    assert_eq!(
        session.submissions(),
        1,
        "compatible queued grayscale tiles should flush through one repeated Metal batch"
    );
}

#[test]
fn submitted_auto_512_grayscale_tiles_flush_as_one_metal_batch() {
    let bytes = fixture_gray8_sized(512, 512);
    let mut ctx = slidecodec_core::DecoderContext::<J2kContext>::new();
    let mut session = MetalSession::default();
    let mut pool = J2kScratchPool::new();

    let submissions = (0..16)
        .map(|_| {
            Codec::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                &bytes,
                PixelFormat::Gray8,
                BackendRequest::Auto,
            )
            .expect("submit auto tile")
        })
        .collect::<Vec<_>>();

    assert_eq!(
        session.submissions(),
        0,
        "auto submitted tile surfaces should stay queued until a wait flushes the session"
    );

    for submission in submissions {
        let surface = submission.wait().expect("surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.dimensions(), (512, 512));
    }
    assert_eq!(
        session.submissions(),
        1,
        "compatible auto grayscale tiles should flush through one repeated Metal batch"
    );
}

#[test]
fn submitted_distinct_full_grayscale_tiles_flush_as_one_device_batch() {
    let classic_bytes = fixture_gray8();
    let reversed_bytes = fixture_gray8_reversed();
    let mut ctx = slidecodec_core::DecoderContext::<J2kContext>::new();
    let mut session = MetalSession::default();
    let mut pool = J2kScratchPool::new();

    let classic_submission = Codec::submit_tile_to_device(
        &mut ctx,
        &mut session,
        &mut pool,
        &classic_bytes,
        PixelFormat::Gray8,
        BackendRequest::Metal,
    )
    .expect("submit classic tile");
    let reversed_submission = Codec::submit_tile_to_device(
        &mut ctx,
        &mut session,
        &mut pool,
        &reversed_bytes,
        PixelFormat::Gray8,
        BackendRequest::Metal,
    )
    .expect("submit reversed tile");

    assert_eq!(
        session.submissions(),
        0,
        "distinct submitted tile surfaces should stay queued until wait"
    );

    let mut classic_host_decoder = J2kDecoder::new(&classic_bytes).expect("classic host decoder");
    let mut classic_host = [0u8; 16];
    classic_host_decoder
        .decode_into(&mut classic_host, 4, PixelFormat::Gray8)
        .expect("classic host decode");

    let mut reversed_host_decoder =
        J2kDecoder::new(&reversed_bytes).expect("reversed host decoder");
    let mut reversed_host = [0u8; 16];
    reversed_host_decoder
        .decode_into(&mut reversed_host, 4, PixelFormat::Gray8)
        .expect("reversed host decode");

    let classic_surface = classic_submission.wait().expect("classic surface");
    let reversed_surface = reversed_submission.wait().expect("reversed surface");
    assert_eq!(classic_surface.backend_kind(), BackendKind::Metal);
    assert_eq!(reversed_surface.backend_kind(), BackendKind::Metal);
    assert_eq!(classic_surface.as_bytes(), classic_host.as_slice());
    assert_eq!(reversed_surface.as_bytes(), reversed_host.as_slice());
    assert_eq!(
        session.submissions(),
        1,
        "distinct queued grayscale tiles should flush through one Metal command buffer"
    );
}

#[test]
fn submitted_full_rgb_tiles_flush_as_one_device_batch() {
    let bytes = fixture_direct_rgb8();
    let mut ctx = slidecodec_core::DecoderContext::<J2kContext>::new();
    let mut session = MetalSession::default();
    let mut pool = J2kScratchPool::new();

    let submissions = (0..3)
        .map(|_| {
            Codec::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                &bytes,
                PixelFormat::Rgb8,
                BackendRequest::Metal,
            )
            .expect("submit rgb tile")
        })
        .collect::<Vec<_>>();

    assert_eq!(
        session.submissions(),
        0,
        "submitted RGB tile surfaces should stay queued until a wait flushes the session"
    );

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 12];
    host_decoder
        .decode_into(&mut host, 6, PixelFormat::Rgb8)
        .expect("host decode");

    for submission in submissions {
        let surface = submission.wait().expect("surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.as_bytes(), host.as_slice());
    }
    assert_eq!(
        session.submissions(),
        1,
        "compatible queued RGB tiles should flush through one Metal batch"
    );
}

#[test]
fn submitted_distinct_full_rgb_tiles_flush_as_one_device_batch() {
    let rgb_tiles = [
        fixture_direct_rgb8_variant(0),
        fixture_direct_rgb8_variant(5),
        fixture_direct_rgb8_variant(11),
    ];
    assert_ne!(rgb_tiles[0], rgb_tiles[1], "RGB batch fixtures must differ");
    assert_ne!(rgb_tiles[1], rgb_tiles[2], "RGB batch fixtures must differ");
    let mut ctx = slidecodec_core::DecoderContext::<J2kContext>::new();
    let mut session = MetalSession::default();
    let mut pool = J2kScratchPool::new();

    let submissions = rgb_tiles
        .iter()
        .map(|bytes| {
            Codec::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                bytes,
                PixelFormat::Rgb8,
                BackendRequest::Metal,
            )
            .expect("submit distinct rgb tile")
        })
        .collect::<Vec<_>>();

    assert_eq!(
        session.submissions(),
        0,
        "distinct RGB tile surfaces should stay queued until a wait flushes the session"
    );

    let expected = rgb_tiles
        .iter()
        .map(|bytes| {
            let mut host_decoder = J2kDecoder::new(bytes).expect("host decoder");
            let stride = 8 * 3;
            let mut host = vec![0u8; stride * 8];
            host_decoder
                .decode_into(&mut host, stride, PixelFormat::Rgb8)
                .expect("host decode");
            host
        })
        .collect::<Vec<_>>();

    let mut surfaces = Vec::with_capacity(submissions.len());
    for (submission, host) in submissions.into_iter().zip(expected) {
        let surface = submission.wait().expect("surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.as_bytes(), host.as_slice());
        surfaces.push(surface);
    }
    assert_eq!(
        session.submissions(),
        1,
        "distinct queued RGB tiles should flush through one Metal command buffer"
    );

    let surface_bytes = surfaces[0].byte_len();
    let offsets = surfaces
        .iter()
        .map(|surface| {
            let (_buffer, offset) = surface.metal_buffer().expect("RGB batch Metal buffer");
            offset
        })
        .collect::<Vec<_>>();
    assert_eq!(
        offsets,
        (0..surfaces.len())
            .map(|index| index * surface_bytes)
            .collect::<Vec<_>>(),
        "distinct queued RGB tiles should be packed as one stacked Metal batch output"
    );
}

#[test]
fn metal_tile_batch_decodes_submitted_tiles_in_order() {
    let classic_bytes = fixture_gray8();
    let reversed_bytes = fixture_gray8_reversed();
    let mut batch = MetalTileBatch::new();

    assert!(batch.is_empty());
    assert_eq!(
        batch
            .push_tile(&classic_bytes, PixelFormat::Gray8, BackendRequest::Metal)
            .expect("push classic tile"),
        0
    );
    assert_eq!(
        batch
            .push_shared_tile(
                Arc::<[u8]>::from(reversed_bytes.as_slice()),
                PixelFormat::Gray8,
                BackendRequest::Metal,
            )
            .expect("push reversed tile"),
        1
    );
    assert_eq!(batch.len(), 2);
    assert_eq!(batch.submissions(), 0);

    let surfaces = batch.decode_all().expect("batch decode");
    assert_eq!(surfaces.len(), 2);
    assert_eq!(surfaces[0].backend_kind(), BackendKind::Metal);
    assert_eq!(surfaces[1].backend_kind(), BackendKind::Metal);

    let mut classic_host_decoder = J2kDecoder::new(&classic_bytes).expect("classic host decoder");
    let mut classic_host = [0u8; 16];
    classic_host_decoder
        .decode_into(&mut classic_host, 4, PixelFormat::Gray8)
        .expect("classic host decode");

    let mut reversed_host_decoder =
        J2kDecoder::new(&reversed_bytes).expect("reversed host decoder");
    let mut reversed_host = [0u8; 16];
    reversed_host_decoder
        .decode_into(&mut reversed_host, 4, PixelFormat::Gray8)
        .expect("reversed host decode");

    assert_eq!(surfaces[0].as_bytes(), classic_host.as_slice());
    assert_eq!(surfaces[1].as_bytes(), reversed_host.as_slice());
}

#[test]
fn metal_tile_batch_supports_region_and_scaled_requests() {
    let bytes = fixture_gray8();
    let roi = Rect {
        x: 0,
        y: 0,
        w: 2,
        h: 2,
    };
    let mut batch = MetalTileBatch::with_capacity(2);

    assert_eq!(
        batch
            .push_tile_region(&bytes, PixelFormat::Gray8, roi, BackendRequest::Metal)
            .expect("push region tile"),
        0
    );
    assert_eq!(
        batch
            .push_tile_scaled(
                &bytes,
                PixelFormat::Gray8,
                Downscale::Half,
                BackendRequest::Metal
            )
            .expect("push scaled tile"),
        1
    );

    let surfaces = batch.decode_all().expect("batch decode");
    assert_eq!(surfaces.len(), 2);
    assert_eq!(surfaces[0].dimensions(), (2, 2));
    assert_eq!(surfaces[1].dimensions(), (2, 2));
    assert_eq!(surfaces[0].backend_kind(), BackendKind::Metal);
    assert_eq!(surfaces[1].backend_kind(), BackendKind::Metal);
}

#[test]
fn repeated_classic_grayscale_direct_decode_matches_host_decode() {
    let bytes = fixture_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surfaces = decoder
        .decode_repeated_grayscale_direct_to_device(PixelFormat::Gray8, 3)
        .expect("repeated direct decode");
    assert_eq!(surfaces.len(), 3);

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 16];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray8)
        .expect("host decode");

    for surface in surfaces {
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.as_bytes(), host.as_slice());
    }
}

#[test]
fn repeated_ht_grayscale_direct_decode_matches_host_decode() {
    let bytes = fixture_ht_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surfaces = decoder
        .decode_repeated_grayscale_direct_to_device(PixelFormat::Gray8, 3)
        .expect("repeated direct decode");
    assert_eq!(surfaces.len(), 3);

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 16];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray8)
        .expect("host decode");

    for surface in surfaces {
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.as_bytes(), host.as_slice());
    }
}

#[test]
fn metal_gray16_matches_host_decode_for_12bit_source() {
    let bytes = fixture_gray12();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 8];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray16)
        .expect("host decode");

    let surface = decoder
        .decode_to_device(PixelFormat::Gray16, BackendRequest::Metal)
        .expect("device decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.as_bytes(), host.as_slice());
}

#[test]
fn explicit_metal_rgb_full_tile_matches_host_decode() {
    let rgb8 = fixture_rgb8();
    {
        let mut decoder = J2kDecoder::new(&rgb8).expect("rgb8 decoder");
        let mut host_decoder = J2kDecoder::new(&rgb8).expect("rgb8 host decoder");
        let mut host = [0u8; 12];
        host_decoder
            .decode_into(&mut host, 6, PixelFormat::Rgb8)
            .expect("host rgb8 decode");
        let surface = decoder
            .decode_to_device(PixelFormat::Rgb8, BackendRequest::Metal)
            .expect("explicit Metal rgb8 decode");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.dimensions(), (2, 2));
        assert_eq!(surface.as_bytes(), host.as_slice());
    }

    {
        let mut decoder = J2kDecoder::new(&rgb8).expect("rgba8 decoder");
        let mut host_decoder = J2kDecoder::new(&rgb8).expect("rgba8 host decoder");
        let mut host = [0u8; 16];
        host_decoder
            .decode_into(&mut host, 8, PixelFormat::Rgba8)
            .expect("host rgba8 decode");
        let surface = decoder
            .decode_to_device(PixelFormat::Rgba8, BackendRequest::Metal)
            .expect("explicit Metal rgba8 decode");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.dimensions(), (2, 2));
        assert_eq!(surface.as_bytes(), host.as_slice());
    }

    let rgb12 = fixture_rgb12();
    {
        let mut decoder = J2kDecoder::new(&rgb12).expect("rgb12 decoder");
        let mut host_decoder = J2kDecoder::new(&rgb12).expect("rgb12 host decoder");
        let mut host = [0u8; 12];
        host_decoder
            .decode_into(&mut host, 12, PixelFormat::Rgb16)
            .expect("host rgb16 decode");
        let surface = decoder
            .decode_to_device(PixelFormat::Rgb16, BackendRequest::Metal)
            .expect("explicit Metal rgb16 decode");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.dimensions(), (2, 1));
        assert_eq!(surface.as_bytes(), host.as_slice());
    }
}

#[test]
fn explicit_metal_region_and_scaled_grayscale_match_host_decode() {
    let bytes = fixture_gray8();
    let roi = Rect {
        x: 0,
        y: 0,
        w: 2,
        h: 2,
    };

    let mut host_region_decoder = J2kDecoder::new(&bytes).expect("host region decoder");
    let mut host_region = [0u8; 4];
    host_region_decoder
        .decode_region_into(
            &mut J2kScratchPool::new(),
            &mut host_region,
            2,
            PixelFormat::Gray8,
            roi,
        )
        .expect("host region decode");

    let mut region_decoder = J2kDecoder::new(&bytes).expect("decoder");
    let region_surface = region_decoder
        .decode_region_to_device(PixelFormat::Gray8, roi, BackendRequest::Metal)
        .expect("explicit Metal region decode");
    assert_eq!(region_surface.backend_kind(), BackendKind::Metal);
    assert_eq!(region_surface.dimensions(), (2, 2));
    assert_eq!(region_surface.as_bytes(), host_region.as_slice());

    let mut host_scaled_decoder = J2kDecoder::new(&bytes).expect("host scaled decoder");
    let mut host_scaled = [0u8; 4];
    host_scaled_decoder
        .decode_scaled_into(
            &mut J2kScratchPool::new(),
            &mut host_scaled,
            2,
            PixelFormat::Gray8,
            Downscale::Half,
        )
        .expect("host scaled decode");

    let mut scaled_decoder = J2kDecoder::new(&bytes).expect("decoder");
    let scaled_surface = scaled_decoder
        .decode_scaled_to_device(PixelFormat::Gray8, Downscale::Half, BackendRequest::Metal)
        .expect("explicit Metal scaled decode");
    assert_eq!(scaled_surface.backend_kind(), BackendKind::Metal);
    assert_eq!(scaled_surface.dimensions(), (2, 2));
    assert_eq!(scaled_surface.as_bytes(), host_scaled.as_slice());
}

#[test]
fn explicit_metal_region_and_scaled_htj2k_grayscale_match_host_decode() {
    let bytes = fixture_ht_gray8();
    let roi = Rect {
        x: 0,
        y: 0,
        w: 2,
        h: 2,
    };

    let mut host_region_decoder = J2kDecoder::new(&bytes).expect("host region decoder");
    let mut host_region = [0u8; 4];
    host_region_decoder
        .decode_region_into(
            &mut J2kScratchPool::new(),
            &mut host_region,
            2,
            PixelFormat::Gray8,
            roi,
        )
        .expect("host region decode");

    let mut region_decoder = J2kDecoder::new(&bytes).expect("decoder");
    let region_surface = region_decoder
        .decode_region_to_device(PixelFormat::Gray8, roi, BackendRequest::Metal)
        .expect("explicit Metal region decode");
    assert_eq!(region_surface.backend_kind(), BackendKind::Metal);
    assert_eq!(region_surface.dimensions(), (2, 2));
    assert_eq!(region_surface.as_bytes(), host_region.as_slice());

    let mut host_scaled_decoder = J2kDecoder::new(&bytes).expect("host scaled decoder");
    let mut host_scaled = [0u8; 4];
    host_scaled_decoder
        .decode_scaled_into(
            &mut J2kScratchPool::new(),
            &mut host_scaled,
            2,
            PixelFormat::Gray8,
            Downscale::Half,
        )
        .expect("host scaled decode");

    let mut scaled_decoder = J2kDecoder::new(&bytes).expect("decoder");
    let scaled_surface = scaled_decoder
        .decode_scaled_to_device(PixelFormat::Gray8, Downscale::Half, BackendRequest::Metal)
        .expect("explicit Metal scaled decode");
    assert_eq!(scaled_surface.backend_kind(), BackendKind::Metal);
    assert_eq!(scaled_surface.dimensions(), (2, 2));
    assert_eq!(scaled_surface.as_bytes(), host_scaled.as_slice());
}

#[test]
fn auto_region_and_scaled_fallback_to_cpu_surface_and_match_host_decode() {
    let bytes = fixture_rgb8();
    let roi = Rect {
        x: 0,
        y: 0,
        w: 1,
        h: 1,
    };

    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let region_surface = decoder
        .decode_region_to_device(PixelFormat::Rgb8, roi, BackendRequest::Auto)
        .expect("region surface");
    assert_eq!(region_surface.backend_kind(), BackendKind::Cpu);

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut region_host = [0u8; 3];
    host_decoder
        .decode_region_into(
            &mut J2kScratchPool::new(),
            &mut region_host,
            3,
            PixelFormat::Rgb8,
            roi,
        )
        .expect("host region");
    assert_eq!(region_surface.as_bytes(), region_host.as_slice());

    let scaled_surface = decoder
        .decode_scaled_to_device(PixelFormat::Rgb8, Downscale::Half, BackendRequest::Auto)
        .expect("scaled surface");
    assert_eq!(scaled_surface.backend_kind(), BackendKind::Cpu);

    let mut scaled_host = [0u8; 3];
    host_decoder
        .decode_scaled_into(
            &mut J2kScratchPool::new(),
            &mut scaled_host,
            3,
            PixelFormat::Rgb8,
            Downscale::Half,
        )
        .expect("host scaled");
    assert_eq!(scaled_surface.as_bytes(), scaled_host.as_slice());
}

#[test]
fn invalid_region_reports_error_instead_of_panicking() {
    let bytes = fixture_rgb8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    match decoder.decode_region_to_device(PixelFormat::Rgb8, roi, BackendRequest::Auto) {
        Err(Error::Decode(slidecodec_j2k::J2kError::InvalidRegion { .. })) => {}
        Err(other) => panic!("unexpected error for invalid ROI: {other:?}"),
        Ok(_) => panic!("invalid ROI should fail"),
    }
}
