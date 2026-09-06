// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

mod residency;

fn metal_session() -> Option<MetalBackendSession> {
    should_run_metal_runtime()
        .then(|| MetalBackendSession::system_default().expect("Metal backend session"))
}

#[test]
fn rgb8_fast444_batch_decode_can_write_into_reusable_metal_textures() {
    let Some(session) = metal_session() else {
        return;
    };
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (8, 8), 2).expect("texture output");
    let inputs = [BASELINE_444, BASELINE_444];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_444)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode into reusable textures");

    assert_eq!(tiles.len(), 2);
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(output.dimensions(), (8, 8));
    assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (8, 8));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(output.shares_access_gate_with_tile(&tile));
        // SAFETY: The decode call above waited for completion, and this test
        // submits no overlapping writer through either raw texture handle.
        let tile_texture = unsafe { tile.texture() };
        // SAFETY: Same completed decode and no-overlapping-writer invariant.
        let output_texture = unsafe { output.texture(index) }.expect("output texture");
        assert!(std::ptr::eq(tile_texture, output_texture));
        assert_eq!(
            download_rgba8_texture(&session, tile_texture, tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_resizes_reusable_metal_textures() {
    let Some(session) = metal_session() else {
        return;
    };
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = Codec::decode_rgb8_decoder_batch_into_resizable_metal_textures_with_session(
        &decoders,
        &mut output,
        &session,
    )
    .expect("decode cached decoder batch into resizable reusable textures");

    assert_eq!(output.dimensions(), (16, 16));
    assert_eq!(output.tile_capacity(), 2);
    let expected_tiles = [expected_rgba.as_slice(), expected_rgba.as_slice()];
    assert_reusable_rgba_texture_tiles(&session, &output, tiles, (16, 16), &expected_tiles);
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_can_write_into_fixed_metal_textures() {
    let Some(session) = metal_session() else {
        return;
    };
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (16, 16), 2).expect("texture output");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Decoders(&decoders),
        Rgb8MetalBatchOp::Full,
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode cached decoder batch into fixed reusable textures");

    assert_eq!(tiles.len(), 2);
    assert_eq!(output.dimensions(), (16, 16));
    assert_eq!(output.tile_capacity(), 2);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (16, 16));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_rejects_mixed_output_dimensions_without_resizing_textures() {
    let Some(session) = metal_session() else {
        return;
    };
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_444).expect("second decoder");
    let decoders = [&first, &second];

    let Err(err) = Codec::decode_rgb8_decoder_batch_into_resizable_metal_textures_with_session(
        &decoders,
        &mut output,
        &session,
    ) else {
        panic!("mixed output dimensions should be rejected");
    };

    assert!(matches!(err, Error::UnsupportedMetalRequest { .. }));
    assert_eq!(output.dimensions(), (1, 1));
    assert_eq!(output.tile_capacity(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_rejects_mixed_sampling_without_resizing_textures() {
    let Some(session) = metal_session() else {
        return;
    };
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");
    let rgb = j2k_test_support::patterned_rgb8(16, 16);
    let fast420 = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: 16,
            height: 16,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode fast420 jpeg");
    let fast444 = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: 16,
            height: 16,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode fast444 jpeg");
    let first = Decoder::new(&fast420.data).expect("first decoder");
    let second = Decoder::new(&fast444.data).expect("second decoder");
    let decoders = [&first, &second];

    let Err(err) = Codec::decode_rgb8_decoder_batch_into_resizable_metal_textures_with_session(
        &decoders,
        &mut output,
        &session,
    ) else {
        panic!("mixed sampling should be rejected");
    };

    assert!(matches!(
        err,
        Error::UnsupportedMetalRequest { reason }
            if reason.contains("same fast-packet sampling family")
    ));
    assert_eq!(output.dimensions(), (1, 1));
    assert_eq!(output.tile_capacity(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_scaled_batch_decode_can_write_into_reusable_metal_textures() {
    let Some(session) = metal_session() else {
        return;
    };
    let scale = Downscale::Quarter;
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (4, 4), 2).expect("texture output");
    let inputs = [BASELINE_420, BASELINE_420];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::scaled(PixelFormat::Rgb8, scale))
        .expect("cpu scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Scaled(scale),
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode scaled into reusable textures");

    assert_eq!(tiles.len(), 2);
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(output.dimensions(), (4, 4));
    assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (4, 4));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_scaled_batch_decode_resizes_reusable_metal_textures() {
    let Some(session) = metal_session() else {
        return;
    };
    let scale = Downscale::Quarter;
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");
    let inputs = [BASELINE_420, BASELINE_420];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::scaled(PixelFormat::Rgb8, scale))
        .expect("cpu scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Scaled(scale),
        MetalTextureBatchTarget::Resizable(&mut output),
        &session,
    )
    .expect("decode scaled into resizable reusable textures");

    assert_eq!(output.dimensions(), (4, 4));
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(tiles.len(), 2);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (4, 4));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_scaled_batch_resizes_reusable_metal_textures() {
    let Some(session) = metal_session() else {
        return;
    };
    let scale = Downscale::Quarter;
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::scaled(PixelFormat::Rgb8, scale))
        .expect("cpu scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Decoders(&decoders),
        Rgb8MetalBatchOp::Scaled(scale),
        MetalTextureBatchTarget::Resizable(&mut output),
        &session,
    )
    .expect("decode cached decoder scaled batch into resizable reusable textures");

    assert_eq!(output.dimensions(), (4, 4));
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(tiles.len(), 2);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (4, 4));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_scaled_batch_can_write_into_fixed_metal_textures() {
    let Some(session) = metal_session() else {
        return;
    };
    let scale = Downscale::Quarter;
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (4, 4), 2).expect("texture output");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::scaled(PixelFormat::Rgb8, scale))
        .expect("cpu scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Decoders(&decoders),
        Rgb8MetalBatchOp::Scaled(scale),
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode cached decoder scaled batch into fixed reusable textures");

    assert_eq!(tiles.len(), 2);
    assert_eq!(output.dimensions(), (4, 4));
    assert_eq!(output.tile_capacity(), 2);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (4, 4));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_fast422_region_scaled_batch_decode_can_write_into_reusable_metal_textures() {
    let Some(session) = metal_session() else {
        return;
    };
    let roi = Rect {
        x: 1,
        y: 1,
        w: 9,
        h: 6,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 2)
        .expect("texture output");
    let inputs = [BASELINE_422, BASELINE_422];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_422)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("cpu region scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::RegionScaled { roi, scale },
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode region scaled into reusable textures");

    assert_eq!(tiles.len(), 2);
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(output.dimensions(), (scaled.w, scaled.h));
    assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the end-to-end fast422 case verifies mixed tables, resident texture reuse, grouped dispatches, and CPU parity together"
)]
fn rgb8_table_mixed_fast422_region_scaled_texture_batch_groups_resident_dispatches() {
    let Some(session) = metal_session() else {
        return;
    };
    let dimensions = (128, 96);
    let roi = Rect {
        x: 0,
        y: 0,
        w: dimensions.0,
        h: dimensions.1,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let rgb_a = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_b = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_c = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    for (index, pixel) in rgb_b.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(41)
            .wrapping_add(29);
        pixel[0] ^= delta.rotate_left(1);
        pixel[1] = pixel[1].wrapping_add(delta);
        pixel[2] = pixel[2].wrapping_sub(delta.rotate_right(2));
    }
    for (index, pixel) in rgb_c.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index).wrapping_mul(59).wrapping_add(3);
        pixel[0] = pixel[0].wrapping_sub(delta.rotate_left(2));
        pixel[1] ^= delta.rotate_right(1);
        pixel[2] = pixel[2].wrapping_add(delta);
    }

    let jpeg_a = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_a,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode first fast422 region-scaled table group jpeg");
    let jpeg_b = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_b,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 71,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode second fast422 region-scaled table group jpeg");
    let jpeg_c = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_c,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode third fast422 region-scaled table group jpeg");
    let packet_a = build_fast422_packet(&jpeg_a.data).expect("first fast422 packet");
    let packet_b = build_fast422_packet(&jpeg_b.data).expect("second fast422 packet");
    let packet_c = build_fast422_packet(&jpeg_c.data).expect("third fast422 packet");
    assert_eq!(packet_a.y_quant, packet_c.y_quant);
    assert_eq!(packet_a.cb_quant, packet_c.cb_quant);
    assert_eq!(packet_a.cr_quant, packet_c.cr_quant);
    assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
    assert_eq!(packet_a.y_ac_table, packet_c.y_ac_table);
    assert_eq!(
        packet_a.entropy_checkpoints.len(),
        packet_c.entropy_checkpoints.len()
    );
    assert_ne!(packet_a.y_quant, packet_b.y_quant);

    let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 3)
        .expect("texture output");
    let inputs = [
        jpeg_a.data.as_slice(),
        jpeg_b.data.as_slice(),
        jpeg_c.data.as_slice(),
    ];
    let (expected_rgb_a, _) = CpuDecoder::new(&jpeg_a.data)
        .expect("first cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("first cpu region scaled decode");
    let (expected_rgb_b, _) = CpuDecoder::new(&jpeg_b.data)
        .expect("second cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("second cpu region scaled decode");
    let (expected_rgb_c, _) = CpuDecoder::new(&jpeg_c.data)
        .expect("third cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("third cpu region scaled decode");
    let expected_tiles = [
        rgb_to_rgba_opaque(&expected_rgb_a),
        rgb_to_rgba_opaque(&expected_rgb_b),
        rgb_to_rgba_opaque(&expected_rgb_c),
    ];
    assert_ne!(expected_tiles[0], expected_tiles[1]);
    assert_ne!(expected_tiles[0], expected_tiles[2]);
    assert_ne!(expected_tiles[1], expected_tiles[2]);

    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::RegionScaled { roi, scale },
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode table-mixed fast422 region-scaled tiles into reusable textures");

    assert_eq!(tiles.len(), 3);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        let actual_rgba =
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions());
        assert_eq!(actual_rgba.as_slice(), expected_tiles[index].as_slice());
    }
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the end-to-end fast444 case verifies mixed tables, resident texture reuse, grouped dispatches, and CPU parity together"
)]
fn rgb8_table_mixed_fast444_region_scaled_texture_batch_groups_resident_dispatches() {
    let Some(session) = metal_session() else {
        return;
    };
    let dimensions = (96, 96);
    let roi = Rect {
        x: 0,
        y: 0,
        w: dimensions.0,
        h: dimensions.1,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let rgb_a = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_b = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_c = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    for (index, pixel) in rgb_b.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(61)
            .wrapping_add(13);
        pixel[0] = pixel[0].wrapping_add(delta);
        pixel[1] ^= delta.rotate_left(1);
        pixel[2] = pixel[2].wrapping_sub(delta.rotate_right(2));
    }
    for (index, pixel) in rgb_c.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(67)
            .wrapping_add(31);
        pixel[0] = pixel[0].wrapping_sub(delta.rotate_left(2));
        pixel[1] = pixel[1].wrapping_add(delta.rotate_right(1));
        pixel[2] ^= delta;
    }

    let jpeg_a = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_a,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 91,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode first fast444 region-scaled table group jpeg");
    let jpeg_b = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_b,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 70,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode second fast444 region-scaled table group jpeg");
    let jpeg_c = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_c,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 91,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode third fast444 region-scaled table group jpeg");
    let packet_a = build_fast444_packet(&jpeg_a.data).expect("first fast444 packet");
    let packet_b = build_fast444_packet(&jpeg_b.data).expect("second fast444 packet");
    let packet_c = build_fast444_packet(&jpeg_c.data).expect("third fast444 packet");
    assert_eq!(packet_a.y_quant, packet_c.y_quant);
    assert_eq!(packet_a.cb_quant, packet_c.cb_quant);
    assert_eq!(packet_a.cr_quant, packet_c.cr_quant);
    assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
    assert_eq!(packet_a.y_ac_table, packet_c.y_ac_table);
    assert_eq!(
        packet_a.entropy_checkpoints.len(),
        packet_c.entropy_checkpoints.len()
    );
    assert_ne!(packet_a.y_quant, packet_b.y_quant);

    let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 3)
        .expect("texture output");
    let inputs = [
        jpeg_a.data.as_slice(),
        jpeg_b.data.as_slice(),
        jpeg_c.data.as_slice(),
    ];
    let (expected_rgb_a, _) = CpuDecoder::new(&jpeg_a.data)
        .expect("first cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("first cpu region scaled decode");
    let (expected_rgb_b, _) = CpuDecoder::new(&jpeg_b.data)
        .expect("second cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("second cpu region scaled decode");
    let (expected_rgb_c, _) = CpuDecoder::new(&jpeg_c.data)
        .expect("third cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("third cpu region scaled decode");
    let expected_tiles = [
        rgb_to_rgba_opaque(&expected_rgb_a),
        rgb_to_rgba_opaque(&expected_rgb_b),
        rgb_to_rgba_opaque(&expected_rgb_c),
    ];
    assert_ne!(expected_tiles[0], expected_tiles[1]);
    assert_ne!(expected_tiles[0], expected_tiles[2]);
    assert_ne!(expected_tiles[1], expected_tiles[2]);

    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::RegionScaled { roi, scale },
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode table-mixed fast444 region-scaled tiles into reusable textures");

    assert_eq!(tiles.len(), 3);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        let actual_rgba =
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions());
        assert_eq!(actual_rgba.as_slice(), expected_tiles[index].as_slice());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_fast420_region_scaled_batch_decode_can_write_into_reusable_metal_textures() {
    let Some(session) = metal_session() else {
        return;
    };
    let roi = Rect {
        x: 1,
        y: 2,
        w: 10,
        h: 9,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 2)
        .expect("texture output");
    let inputs = [BASELINE_420, BASELINE_420];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("cpu region scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::RegionScaled { roi, scale },
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode region scaled into reusable textures");

    assert_eq!(tiles.len(), 2);
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(output.dimensions(), (scaled.w, scaled.h));
    assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_region_scaled_batch_resizes_reusable_metal_textures() {
    let Some(session) = metal_session() else {
        return;
    };
    let roi = Rect {
        x: 1,
        y: 2,
        w: 10,
        h: 9,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("cpu region scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Decoders(&decoders),
        Rgb8MetalBatchOp::RegionScaled { roi, scale },
        MetalTextureBatchTarget::Resizable(&mut output),
        &session,
    )
    .expect("decode cached decoder batch into resizable reusable textures");

    assert_eq!(output.dimensions(), (scaled.w, scaled.h));
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(tiles.len(), 2);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_region_scaled_batch_can_write_into_fixed_metal_textures() {
    let Some(session) = metal_session() else {
        return;
    };
    let roi = Rect {
        x: 1,
        y: 2,
        w: 10,
        h: 9,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 2)
        .expect("texture output");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("cpu region scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Decoders(&decoders),
        Rgb8MetalBatchOp::RegionScaled { roi, scale },
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode cached decoder region-scaled batch into fixed reusable textures");

    assert_eq!(tiles.len(), 2);
    assert_eq!(output.dimensions(), (scaled.w, scaled.h));
    assert_eq!(output.tile_capacity(), 2);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_restart_fast420_region_scaled_batch_decode_writes_reusable_metal_textures() {
    let Some(session) = metal_session() else {
        return;
    };
    let dimensions = (128, 128);
    let roi = Rect {
        x: 9,
        y: 11,
        w: 73,
        h: 67,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let rgb = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: Some(4),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode restart-coded fast420 region-scaled texture jpeg");
    let packet = build_fast420_packet(&jpeg.data).expect("restart fast420 packet");
    assert_ne!(packet.restart_interval_mcus, 0);
    assert!(!packet.restart_offsets.is_empty());

    let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 2)
        .expect("texture output");
    let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
    let (expected_rgb, _) = CpuDecoder::new(&jpeg.data)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("cpu region-scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::RegionScaled { roi, scale },
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode restart-coded region-scaled tiles into reusable textures");

    assert_eq!(tiles.len(), 2);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
fn assert_restart_region_scaled_texture_batch_writes_reusable_metal_output(
    subsampling: JpegSubsampling,
    dimensions: (u32, u32),
) {
    let Some(session) = metal_session() else {
        return;
    };
    let roi = Rect {
        x: 0,
        y: 0,
        w: dimensions.0,
        h: dimensions.1,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let rgb = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling,
            restart_interval: Some(256),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode restart-coded region-scaled texture jpeg");
    match subsampling {
        JpegSubsampling::Ybr422 => {
            let packet = build_fast422_packet(&jpeg.data).expect("restart fast422 packet");
            assert_ne!(packet.restart_interval_mcus, 0);
            assert!(!packet.restart_offsets.is_empty());
        }
        JpegSubsampling::Ybr444 => {
            let packet = build_fast444_packet(&jpeg.data).expect("restart fast444 packet");
            assert_ne!(packet.restart_interval_mcus, 0);
            assert!(!packet.restart_offsets.is_empty());
        }
        _ => panic!("restart region-scaled texture helper expects fast422 or fast444"),
    }

    let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 2)
        .expect("texture output");
    let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
    let (expected_rgb, _) = CpuDecoder::new(&jpeg.data)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("cpu region-scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::RegionScaled { roi, scale },
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode restart-coded region-scaled tiles into reusable textures");

    assert_eq!(tiles.len(), 2);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_restart_fast422_region_scaled_batch_decode_writes_reusable_metal_textures() {
    assert_restart_region_scaled_texture_batch_writes_reusable_metal_output(
        JpegSubsampling::Ybr422,
        (128, 96),
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_restart_fast444_region_scaled_batch_decode_writes_reusable_metal_textures() {
    assert_restart_region_scaled_texture_batch_writes_reusable_metal_output(
        JpegSubsampling::Ybr444,
        (96, 96),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the end-to-end fast420 case verifies mixed tables, resident texture reuse, grouped dispatches, and CPU parity together"
)]
fn rgb8_table_mixed_fast420_region_scaled_texture_batch_groups_resident_dispatches() {
    let Some(session) = metal_session() else {
        return;
    };
    let dimensions = (128, 128);
    let roi = Rect {
        x: 9,
        y: 11,
        w: 77,
        h: 65,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let rgb_a = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_b = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_c = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    for (index, pixel) in rgb_b.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(43)
            .wrapping_add(19);
        pixel[0] = pixel[0].wrapping_add(delta.rotate_left(1));
        pixel[1] = pixel[1].wrapping_sub(delta);
        pixel[2] ^= delta.rotate_right(2);
    }
    for (index, pixel) in rgb_c.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(47)
            .wrapping_add(23);
        pixel[0] ^= delta.rotate_left(2);
        pixel[1] = pixel[1].wrapping_add(delta.rotate_right(1));
        pixel[2] = pixel[2].wrapping_sub(delta);
    }

    let jpeg_a = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_a,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode first fast420 region-scaled table group jpeg");
    let jpeg_b = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_b,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 72,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode second fast420 region-scaled table group jpeg");
    let jpeg_c = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_c,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode third fast420 region-scaled table group jpeg");
    let packet_a = build_fast420_packet(&jpeg_a.data).expect("first fast420 packet");
    let packet_b = build_fast420_packet(&jpeg_b.data).expect("second fast420 packet");
    let packet_c = build_fast420_packet(&jpeg_c.data).expect("third fast420 packet");
    assert_eq!(packet_a.y_quant, packet_c.y_quant);
    assert_eq!(packet_a.cb_quant, packet_c.cb_quant);
    assert_eq!(packet_a.cr_quant, packet_c.cr_quant);
    assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
    assert_eq!(packet_a.y_ac_table, packet_c.y_ac_table);
    assert_eq!(
        packet_a.entropy_checkpoints.len(),
        packet_c.entropy_checkpoints.len()
    );
    assert_ne!(packet_a.y_quant, packet_b.y_quant);

    let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 3)
        .expect("texture output");
    let inputs = [
        jpeg_a.data.as_slice(),
        jpeg_b.data.as_slice(),
        jpeg_c.data.as_slice(),
    ];
    let (expected_rgb_a, _) = CpuDecoder::new(&jpeg_a.data)
        .expect("first cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("first cpu region scaled decode");
    let (expected_rgb_b, _) = CpuDecoder::new(&jpeg_b.data)
        .expect("second cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("second cpu region scaled decode");
    let (expected_rgb_c, _) = CpuDecoder::new(&jpeg_c.data)
        .expect("third cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("third cpu region scaled decode");
    let expected_tiles = [
        rgb_to_rgba_opaque(&expected_rgb_a),
        rgb_to_rgba_opaque(&expected_rgb_b),
        rgb_to_rgba_opaque(&expected_rgb_c),
    ];
    assert_ne!(expected_tiles[0], expected_tiles[1]);
    assert_ne!(expected_tiles[0], expected_tiles[2]);
    assert_ne!(expected_tiles[1], expected_tiles[2]);

    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::RegionScaled { roi, scale },
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode table-mixed fast420 region-scaled tiles into reusable textures");

    assert_eq!(tiles.len(), 3);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        let actual_rgba =
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions());
        assert_eq!(actual_rgba.as_slice(), expected_tiles[index].as_slice());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_fast420_batch_decode_can_write_into_reusable_metal_textures() {
    let Some(session) = metal_session() else {
        return;
    };
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (16, 16), 2).expect("texture output");
    let inputs = [BASELINE_420, BASELINE_420];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode into reusable textures");

    assert_eq!(tiles.len(), 2);
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(output.dimensions(), (16, 16));
    assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (16, 16));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_fast422_batch_decode_can_write_into_reusable_metal_textures() {
    let Some(session) = metal_session() else {
        return;
    };
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (16, 8), 2).expect("texture output");
    let inputs = [BASELINE_422, BASELINE_422];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_422)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode into reusable textures");

    assert_eq!(tiles.len(), 2);
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(output.dimensions(), (16, 8));
    assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (16, 8));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_texture_batch_decode_avoids_private_rgba_staging_buffers() {
    if !should_run_metal_runtime() {
        return;
    }

    let cases = [
        (BASELINE_420, (16, 16), 3),
        (BASELINE_422, (16, 8), 3),
        (BASELINE_444, (8, 8), 0),
    ];

    for (input, dimensions, expected_private_allocations) in cases {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 2)
            .expect("texture output");
        let inputs = [input, input];

        compute::reset_jpeg_private_buffer_allocations_for_test();
        let tiles = decode_rgb8_texture_batch_with_session(
            Rgb8MetalBatchSource::Bytes(&inputs),
            Rgb8MetalBatchOp::Full,
            MetalTextureBatchTarget::Reusable(&output),
            &session,
        )
        .expect("decode into reusable textures");
        assert_eq!(tiles.len(), 2);
        for tile in tiles {
            assert_eq!(
                tile.expect("texture tile").pixel_format(),
                PixelFormat::Rgba8
            );
        }

        assert_eq!(
                compute::jpeg_private_buffer_allocations_for_test(),
                expected_private_allocations,
                "texture batch decode should not allocate a private RGBA staging buffer for {dimensions:?}"
            );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_fast444_texture_batch_decode_fuses_directly_into_reusable_metal_textures() {
    let Some(session) = metal_session() else {
        return;
    };
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (8, 8), 2).expect("texture output");
    let inputs = [BASELINE_444, BASELINE_444];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_444)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode into reusable textures");

    let expected_tiles = [expected_rgba.as_slice(), expected_rgba.as_slice()];
    assert_reusable_rgba_texture_tiles(&session, &output, tiles, (8, 8), &expected_tiles);
    assert_eq!(
        compute::jpeg_private_buffer_allocations_for_test(),
        0,
        "fused 4:4:4 texture batch decode should not allocate private Y/Cb/Cr staging planes"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the end-to-end fast444 case verifies mixed tables, resident texture reuse, grouped dispatches, allocation counts, and CPU parity together"
)]
fn rgb8_table_mixed_fast444_texture_batch_groups_resident_dispatches() {
    let Some(session) = metal_session() else {
        return;
    };
    let dimensions = (64, 64);
    let rgb_a = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_b = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_c = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    for (index, pixel) in rgb_b.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index).wrapping_mul(31).wrapping_add(5);
        pixel[0] = pixel[0].wrapping_sub(delta);
        pixel[1] = pixel[1].wrapping_add(delta.rotate_left(1));
        pixel[2] ^= delta.rotate_right(2);
    }
    for (index, pixel) in rgb_c.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(37)
            .wrapping_add(17);
        pixel[0] ^= delta.rotate_left(3);
        pixel[1] = pixel[1].wrapping_sub(delta.rotate_right(1));
        pixel[2] = pixel[2].wrapping_add(delta);
    }

    let jpeg_a = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_a,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 92,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode first fast444 table group jpeg");
    let jpeg_b = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_b,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 71,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode second fast444 table group jpeg");
    let jpeg_c = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_c,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 92,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode third fast444 table group jpeg");
    let packet_a = build_fast444_packet(&jpeg_a.data).expect("first fast444 packet");
    let packet_b = build_fast444_packet(&jpeg_b.data).expect("second fast444 packet");
    let packet_c = build_fast444_packet(&jpeg_c.data).expect("third fast444 packet");
    assert_eq!(packet_a.y_quant, packet_c.y_quant);
    assert_eq!(packet_a.cb_quant, packet_c.cb_quant);
    assert_eq!(packet_a.cr_quant, packet_c.cr_quant);
    assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
    assert_eq!(packet_a.y_ac_table, packet_c.y_ac_table);
    assert_eq!(
        packet_a.entropy_checkpoints.len(),
        packet_c.entropy_checkpoints.len()
    );
    assert_ne!(packet_a.y_quant, packet_b.y_quant);

    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 3).expect("texture output");
    let inputs = [
        jpeg_a.data.as_slice(),
        jpeg_b.data.as_slice(),
        jpeg_c.data.as_slice(),
    ];
    let (expected_rgb_a, _) = CpuDecoder::new(&jpeg_a.data)
        .expect("first cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("first cpu decode");
    let (expected_rgb_b, _) = CpuDecoder::new(&jpeg_b.data)
        .expect("second cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("second cpu decode");
    let (expected_rgb_c, _) = CpuDecoder::new(&jpeg_c.data)
        .expect("third cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("third cpu decode");
    let expected_tiles = [
        rgb_to_rgba_opaque(&expected_rgb_a),
        rgb_to_rgba_opaque(&expected_rgb_b),
        rgb_to_rgba_opaque(&expected_rgb_c),
    ];
    assert_ne!(expected_tiles[0], expected_tiles[1]);
    assert_ne!(expected_tiles[0], expected_tiles[2]);
    assert_ne!(expected_tiles[1], expected_tiles[2]);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode table-mixed fast444 tiles into reusable textures");

    assert_eq!(tiles.len(), 3);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), dimensions);
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        let actual_rgba =
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions());
        assert_eq!(actual_rgba.as_slice(), expected_tiles[index].as_slice());
    }
    assert_eq!(
            compute::jpeg_private_buffer_allocations_for_test(),
            0,
            "table-mixed resident 4:4:4 texture dispatches should not allocate private Y/Cb/Cr staging planes"
        );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_fast422_texture_batch_decode_uses_reusable_component_planes() {
    let Some(session) = metal_session() else {
        return;
    };
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (16, 8), 2).expect("texture output");
    let inputs = [BASELINE_422, BASELINE_422];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_422)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode into reusable textures");

    let expected_tiles = [expected_rgba.as_slice(), expected_rgba.as_slice()];
    assert_reusable_rgba_texture_tiles(&session, &output, tiles, (16, 8), &expected_tiles);
    assert_eq!(
        compute::jpeg_private_buffer_allocations_for_test(),
        3,
        "subsampled texture decode allocates only three reusable component planes"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_wide_fast422_texture_batch_decode_uses_reusable_component_planes() {
    let Some(session) = metal_session() else {
        return;
    };
    let dimensions = (48, 16);
    let rgb = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 92,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode 4:2:2 source jpeg");
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 2).expect("texture output");
    let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
    let (expected_rgb, _) = CpuDecoder::new(&jpeg.data)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode into reusable textures");

    let expected_tiles = [expected_rgba.as_slice(), expected_rgba.as_slice()];
    assert_reusable_rgba_texture_tiles(&session, &output, tiles, dimensions, &expected_tiles);
    assert_eq!(
        compute::jpeg_private_buffer_allocations_for_test(),
        3,
        "subsampled texture decode allocates only three reusable component planes"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the end-to-end fast422 case verifies mixed tables, resident texture reuse, grouped dispatches, allocation counts, and CPU parity together"
)]
fn rgb8_table_mixed_fast422_texture_batch_groups_resident_dispatches() {
    let Some(session) = metal_session() else {
        return;
    };
    let dimensions = (96, 48);
    let rgb_a = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_b = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_c = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    for (index, pixel) in rgb_b.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(23)
            .wrapping_add(11);
        pixel[0] = pixel[0].wrapping_add(delta.rotate_left(1));
        pixel[1] ^= delta;
        pixel[2] = pixel[2].wrapping_sub(delta.rotate_right(2));
    }
    for (index, pixel) in rgb_c.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(19)
            .wrapping_add(53);
        pixel[0] ^= delta.rotate_left(2);
        pixel[1] = pixel[1].wrapping_sub(delta);
        pixel[2] = pixel[2].wrapping_add(delta.rotate_right(1));
    }

    let jpeg_a = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_a,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 91,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode first fast422 table group jpeg");
    let jpeg_b = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_b,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 73,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode second fast422 table group jpeg");
    let jpeg_c = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_c,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 91,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode third fast422 table group jpeg");
    let packet_a = build_fast422_packet(&jpeg_a.data).expect("first fast422 packet");
    let packet_b = build_fast422_packet(&jpeg_b.data).expect("second fast422 packet");
    let packet_c = build_fast422_packet(&jpeg_c.data).expect("third fast422 packet");
    assert_eq!(packet_a.y_quant, packet_c.y_quant);
    assert_eq!(packet_a.cb_quant, packet_c.cb_quant);
    assert_eq!(packet_a.cr_quant, packet_c.cr_quant);
    assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
    assert_eq!(packet_a.y_ac_table, packet_c.y_ac_table);
    assert_eq!(
        packet_a.entropy_checkpoints.len(),
        packet_c.entropy_checkpoints.len()
    );
    assert_ne!(packet_a.y_quant, packet_b.y_quant);

    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 3).expect("texture output");
    let inputs = [
        jpeg_a.data.as_slice(),
        jpeg_b.data.as_slice(),
        jpeg_c.data.as_slice(),
    ];
    let (expected_rgb_a, _) = CpuDecoder::new(&jpeg_a.data)
        .expect("first cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("first cpu decode");
    let (expected_rgb_b, _) = CpuDecoder::new(&jpeg_b.data)
        .expect("second cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("second cpu decode");
    let (expected_rgb_c, _) = CpuDecoder::new(&jpeg_c.data)
        .expect("third cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("third cpu decode");
    let expected_tiles = [
        rgb_to_rgba_opaque(&expected_rgb_a),
        rgb_to_rgba_opaque(&expected_rgb_b),
        rgb_to_rgba_opaque(&expected_rgb_c),
    ];
    assert_ne!(expected_tiles[0], expected_tiles[1]);
    assert_ne!(expected_tiles[0], expected_tiles[2]);
    assert_ne!(expected_tiles[1], expected_tiles[2]);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode table-mixed fast422 tiles into reusable textures");

    assert_eq!(tiles.len(), 3);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), dimensions);
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        let actual_rgba =
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions());
        assert_eq!(actual_rgba.as_slice(), expected_tiles[index].as_slice());
    }
    assert_eq!(
        compute::jpeg_private_buffer_allocations_for_test(),
        3,
        "subsampled texture decode allocates only three reusable component planes"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_fast420_texture_batch_decode_uses_reusable_component_planes() {
    let Some(session) = metal_session() else {
        return;
    };
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (16, 16), 2).expect("texture output");
    let inputs = [BASELINE_420, BASELINE_420];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode into reusable textures");

    assert_eq!(tiles.len(), 2);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (16, 16));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions()),
            expected_rgba
        );
    }
    assert_eq!(
        compute::jpeg_private_buffer_allocations_for_test(),
        3,
        "subsampled texture decode allocates only three reusable component planes"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_wide_row_fast420_texture_batch_decode_uses_reusable_component_planes() {
    let Some(session) = metal_session() else {
        return;
    };
    let dimensions = (32, 16);
    let rgb = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 92,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode 4:2:0 source jpeg");
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 2).expect("texture output");
    let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
    let (expected_rgb, _) = CpuDecoder::new(&jpeg.data)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode into reusable textures");

    let expected_tiles = [expected_rgba.as_slice(), expected_rgba.as_slice()];
    assert_reusable_rgba_texture_tiles(&session, &output, tiles, dimensions, &expected_tiles);
    assert_eq!(
        compute::jpeg_private_buffer_allocations_for_test(),
        3,
        "subsampled texture decode allocates only three reusable component planes"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_multi_row_fast420_texture_batch_decode_uses_reusable_component_planes() {
    let Some(session) = metal_session() else {
        return;
    };
    let dimensions = (16, 32);
    let rgb = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 92,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode 4:2:0 source jpeg");
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 2).expect("texture output");
    let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
    let (expected_rgb, _) = CpuDecoder::new(&jpeg.data)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode into reusable textures");

    let expected_tiles = [expected_rgba.as_slice(), expected_rgba.as_slice()];
    assert_reusable_rgba_texture_tiles(&session, &output, tiles, dimensions, &expected_tiles);
    assert_eq!(
        compute::jpeg_private_buffer_allocations_for_test(),
        3,
        "subsampled texture decode allocates only three reusable component planes"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_multi_axis_fast420_texture_batch_decode_uses_reusable_component_planes() {
    let Some(session) = metal_session() else {
        return;
    };
    for dimensions in [(32, 32), (48, 48)] {
        let rgb = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
        let jpeg = encode_jpeg_baseline(
            JpegSamples::Rgb8 {
                data: &rgb,
                width: dimensions.0,
                height: dimensions.1,
            },
            JpegEncodeOptions {
                quality: 92,
                subsampling: JpegSubsampling::Ybr420,
                restart_interval: None,
                backend: JpegBackend::Cpu,
            },
        )
        .expect("encode 4:2:0 source jpeg");
        let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 2)
            .expect("texture output");
        let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
        let (expected_rgb, _) = CpuDecoder::new(&jpeg.data)
            .expect("cpu decoder")
            .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
            .expect("cpu decode");
        let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

        compute::reset_jpeg_private_buffer_allocations_for_test();
        let tiles = decode_rgb8_texture_batch_with_session(
            Rgb8MetalBatchSource::Bytes(&inputs),
            Rgb8MetalBatchOp::Full,
            MetalTextureBatchTarget::Reusable(&output),
            &session,
        )
        .expect("decode into reusable textures");

        let expected_tiles = [expected_rgba.as_slice(), expected_rgba.as_slice()];
        assert_reusable_rgba_texture_tiles(&session, &output, tiles, dimensions, &expected_tiles);
        assert_eq!(
            compute::jpeg_private_buffer_allocations_for_test(),
            3,
            "subsampled texture decode allocates only three reusable component planes"
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_chunked_multi_axis_fast420_texture_batch_decode_uses_reusable_component_planes() {
    let Some(session) = metal_session() else {
        return;
    };
    let dimensions = (736, 720);
    let rgb = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode chunked 4:2:0 source jpeg");
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 2).expect("texture output");
    let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
    let (expected_rgb, _) = CpuDecoder::new(&jpeg.data)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode into reusable textures");

    let expected_tiles = [expected_rgba.as_slice(), expected_rgba.as_slice()];
    assert_reusable_rgba_texture_tiles(&session, &output, tiles, dimensions, &expected_tiles);
    assert_eq!(
        compute::jpeg_private_buffer_allocations_for_test(),
        3,
        "subsampled texture decode allocates only three reusable component planes"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_restart_fast420_texture_batch_decode_uses_reusable_component_planes() {
    let Some(session) = metal_session() else {
        return;
    };
    let dimensions = (48, 48);
    let rgb = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: Some(2),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode restart 4:2:0 source jpeg");
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 2).expect("texture output");
    let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
    let (expected_rgb, _) = CpuDecoder::new(&jpeg.data)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode into reusable textures");

    let expected_tiles = [expected_rgba.as_slice(), expected_rgba.as_slice()];
    assert_reusable_rgba_texture_tiles(&session, &output, tiles, dimensions, &expected_tiles);
    assert_eq!(
        compute::jpeg_private_buffer_allocations_for_test(),
        3,
        "subsampled texture decode allocates only three reusable component planes"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_distinct_restart_fast420_texture_batch_decode_uses_reusable_component_planes() {
    let Some(session) = metal_session() else {
        return;
    };
    let dimensions = (128, 128);
    let rgb_a = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_b = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    for (index, pixel) in rgb_b.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(17)
            .wrapping_add(31);
        pixel[0] = pixel[0].wrapping_add(delta);
        pixel[1] = pixel[1].wrapping_sub(delta.rotate_left(1));
        pixel[2] ^= delta.rotate_right(1);
    }
    assert_ne!(rgb_a, rgb_b);

    let jpeg_a = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_a,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: Some(4),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode first restart 4:2:0 source jpeg");
    let jpeg_b = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_b,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: Some(4),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode second restart 4:2:0 source jpeg");
    assert_ne!(jpeg_a.data, jpeg_b.data);

    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 2).expect("texture output");
    let inputs = [jpeg_a.data.as_slice(), jpeg_b.data.as_slice()];
    let (expected_rgb_a, _) = CpuDecoder::new(&jpeg_a.data)
        .expect("first cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("first cpu decode");
    let (expected_rgb_b, _) = CpuDecoder::new(&jpeg_b.data)
        .expect("second cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("second cpu decode");
    let expected_tiles = [
        rgb_to_rgba_opaque(&expected_rgb_a),
        rgb_to_rgba_opaque(&expected_rgb_b),
    ];
    assert_ne!(expected_tiles[0], expected_tiles[1]);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode distinct restart tiles into reusable textures");

    assert_eq!(tiles.len(), 2);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), dimensions);
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        let actual_rgba =
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions());
        assert_eq!(actual_rgba.as_slice(), expected_tiles[index].as_slice());
    }
    assert_eq!(
        compute::jpeg_private_buffer_allocations_for_test(),
        3,
        "subsampled texture decode allocates only three reusable component planes"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the end-to-end restart case verifies mixed tables, resident texture reuse, grouped dispatches, allocation counts, and CPU parity together"
)]
fn rgb8_table_mixed_restart_fast420_texture_batch_groups_resident_dispatches() {
    let Some(session) = metal_session() else {
        return;
    };
    let dimensions = (128, 128);
    let rgb_a = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_b = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_c = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    for (index, pixel) in rgb_b.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index).wrapping_mul(29).wrapping_add(7);
        pixel[0] ^= delta;
        pixel[1] = pixel[1].wrapping_add(delta.rotate_left(2));
        pixel[2] = pixel[2].wrapping_sub(delta.rotate_right(2));
    }
    for (index, pixel) in rgb_c.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(13)
            .wrapping_add(41);
        pixel[0] = pixel[0].wrapping_sub(delta.rotate_left(1));
        pixel[1] ^= delta.rotate_right(3);
        pixel[2] = pixel[2].wrapping_add(delta);
    }

    let jpeg_a = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_a,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: Some(4),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode first table group jpeg");
    let jpeg_b = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_b,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 74,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: Some(4),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode second table group jpeg");
    let jpeg_c = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_c,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: Some(4),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode third table group jpeg");
    let packet_a = build_fast420_packet(&jpeg_a.data).expect("first fast420 packet");
    let packet_b = build_fast420_packet(&jpeg_b.data).expect("second fast420 packet");
    let packet_c = build_fast420_packet(&jpeg_c.data).expect("third fast420 packet");
    assert_eq!(packet_a.y_quant, packet_c.y_quant);
    assert_eq!(packet_a.cb_quant, packet_c.cb_quant);
    assert_eq!(packet_a.cr_quant, packet_c.cr_quant);
    assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
    assert_eq!(packet_a.y_ac_table, packet_c.y_ac_table);
    assert_eq!(
        packet_a.entropy_checkpoints.len(),
        packet_c.entropy_checkpoints.len()
    );
    assert_ne!(packet_a.y_quant, packet_b.y_quant);

    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 3).expect("texture output");
    let inputs = [
        jpeg_a.data.as_slice(),
        jpeg_b.data.as_slice(),
        jpeg_c.data.as_slice(),
    ];
    let (expected_rgb_a, _) = CpuDecoder::new(&jpeg_a.data)
        .expect("first cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("first cpu decode");
    let (expected_rgb_b, _) = CpuDecoder::new(&jpeg_b.data)
        .expect("second cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("second cpu decode");
    let (expected_rgb_c, _) = CpuDecoder::new(&jpeg_c.data)
        .expect("third cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("third cpu decode");
    let expected_tiles = [
        rgb_to_rgba_opaque(&expected_rgb_a),
        rgb_to_rgba_opaque(&expected_rgb_b),
        rgb_to_rgba_opaque(&expected_rgb_c),
    ];
    assert_ne!(expected_tiles[0], expected_tiles[1]);
    assert_ne!(expected_tiles[0], expected_tiles[2]);
    assert_ne!(expected_tiles[1], expected_tiles[2]);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode table-mixed restart tiles into reusable textures");

    assert_eq!(tiles.len(), 3);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), dimensions);
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        let actual_rgba =
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions());
        assert_eq!(actual_rgba.as_slice(), expected_tiles[index].as_slice());
    }
    assert_eq!(
        compute::jpeg_private_buffer_allocations_for_test(),
        3,
        "subsampled texture decode allocates only three reusable component planes"
    );
}
