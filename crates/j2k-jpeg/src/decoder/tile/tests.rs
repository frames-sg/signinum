// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    decode_prepared_jpeg_tiles_rgb8, decode_tile_into, decode_tile_into_in_context,
    decode_tile_into_in_context_with_options, decode_tile_region_into_in_context,
    decode_tile_region_into_in_context_with_options, decode_tile_region_scaled_into_in_context,
    decode_tile_region_scaled_into_in_context_with_options, decode_tile_scaled_into_in_context,
    decode_tile_scaled_into_in_context_with_options, planned_jpeg_tile_decode_live_bytes,
    planned_roi_checkpoint_bytes, DecodeOptions, Decoder, DecoderContext, Downscale, JpegError,
    JpegView, PixelFormat, PreparedJpegTileJob, Rect, ScratchPool, TileDecodeOutput,
};
use crate::{
    encode_jpeg_baseline, prepare_tiff_jpeg_tile, JpegBackend, JpegEncodeOptions, JpegSamples,
    JpegSubsampling, JpegTilePrepareOptions,
};
use alloc::vec::Vec;
use j2k_core::CodecContext;
use j2k_test_support::JPEG_BASELINE_420_16X16;

const JPEG: &[u8] = JPEG_BASELINE_420_16X16;

mod batch;

fn full_rgb8_reference() -> (Vec<u8>, usize) {
    let decoder = Decoder::new(JPEG).expect("fixture decoder");
    let (width, height) = decoder.info().dimensions;
    let stride = width as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut output = vec![0u8; stride * height as usize];
    decoder
        .decode_into(&mut output, stride, PixelFormat::Rgb8)
        .expect("reference decode");
    (output, stride)
}

fn scaled_dimensions(scale: Downscale) -> (u32, u32) {
    let denominator = scale.denominator();
    (16u32.div_ceil(denominator), 16u32.div_ceil(denominator))
}

fn scaled_rect(roi: Rect, scale: Downscale) -> Rect {
    let denominator = scale.denominator();
    Rect {
        x: roi.x / denominator,
        y: roi.y / denominator,
        w: (roi.x + roi.w).div_ceil(denominator) - roi.x / denominator,
        h: (roi.y + roi.h).div_ceil(denominator) - roi.y / denominator,
    }
}

fn crop_rgb8(input: &[u8], input_stride: usize, rect: Rect) -> Vec<u8> {
    let row_bytes = rect.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut output = Vec::with_capacity(row_bytes * rect.h as usize);
    for y in rect.y..rect.y + rect.h {
        let start =
            y as usize * input_stride + rect.x as usize * PixelFormat::Rgb8.bytes_per_pixel();
        output.extend_from_slice(&input[start..start + row_bytes]);
    }
    output
}

fn wide_nonrestart_420_jpeg() -> Vec<u8> {
    const WIDTH: u32 = 96;
    const HEIGHT: u32 = 16;
    let pixels = j2k_test_support::patterned_rgb8(WIDTH, HEIGHT);
    encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &pixels,
            width: WIDTH,
            height: HEIGHT,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode wide nonrestart fixture")
    .data
}

fn alternate_nonrestart_420_jpeg() -> Vec<u8> {
    const WIDTH: u32 = 96;
    const HEIGHT: u32 = 16;
    let mut pixels = j2k_test_support::patterned_rgb8(WIDTH, HEIGHT);
    for (index, value) in pixels.iter_mut().enumerate() {
        let offset = u8::try_from(index % 251)
            .expect("modulo bounds alternate fixture offset")
            .wrapping_mul(29)
            .wrapping_add(17);
        *value = value.wrapping_add(offset);
    }
    encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &pixels,
            width: WIDTH,
            height: HEIGHT,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode alternate nonrestart fixture")
    .data
}

#[test]
fn warmed_context_decode_does_not_clone_the_cached_plan_arena() {
    let first = wide_nonrestart_420_jpeg();
    let second = alternate_nonrestart_420_jpeg();
    assert_ne!(first, second);
    let stride = 96 * PixelFormat::Rgb8.bytes_per_pixel();
    let mut expected = vec![0u8; stride * 16];
    Decoder::new(&second)
        .expect("alternate fixture decoder")
        .decode_into(&mut expected, stride, PixelFormat::Rgb8)
        .expect("alternate fixture reference pixels");
    let mut output = vec![0u8; stride * 16];
    let mut context = DecoderContext::new();
    let mut pool = ScratchPool::new();

    decode_tile_into_in_context(
        &first,
        &mut context,
        &mut pool,
        &mut output,
        stride,
        PixelFormat::Rgb8,
    )
    .expect("warm context plan");
    let clones_after_miss = context.decode_plan_clone_count();
    let misses_after_miss = context.cache_stats().misses;
    decode_tile_into_in_context(
        &second,
        &mut context,
        &mut pool,
        &mut output,
        stride,
        PixelFormat::Rgb8,
    )
    .expect("reuse context plan");

    assert_eq!(context.decode_plan_clone_count(), clones_after_miss);
    assert_eq!(context.cache_stats().misses, misses_after_miss);
    assert_eq!(output, expected);
}

#[test]
fn warmed_full_region_and_scaled_context_routes_reuse_the_cached_plan() {
    let jpeg = wide_nonrestart_420_jpeg();
    let decoder = Decoder::new(&jpeg).expect("fixture decoder");
    let mut context = DecoderContext::new();
    let mut pool = ScratchPool::new();
    let full_stride = 96 * PixelFormat::Rgb8.bytes_per_pixel();
    let mut full = vec![0u8; full_stride * 16];
    decode_tile_into_in_context(
        &jpeg,
        &mut context,
        &mut pool,
        &mut full,
        full_stride,
        PixelFormat::Rgb8,
    )
    .expect("warm full decode");
    let clones_after_miss = context.decode_plan_clone_count();

    let roi = Rect {
        x: 8,
        y: 2,
        w: 32,
        h: 10,
    };
    let roi_stride = roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut expected_roi = vec![0u8; roi_stride * roi.h as usize];
    decoder
        .decode_region_into(&mut expected_roi, roi_stride, PixelFormat::Rgb8, roi)
        .expect("reference region decode");
    let mut region = vec![0u8; expected_roi.len()];
    decode_tile_region_into_in_context(
        &jpeg,
        &mut context,
        &mut pool,
        TileDecodeOutput {
            out: &mut region,
            stride: roi_stride,
            fmt: PixelFormat::Rgb8,
        },
        roi,
    )
    .expect("leased region decode");
    assert_eq!(region, expected_roi);

    let scale = Downscale::Half;
    let scaled_width = 96u32.div_ceil(scale.denominator());
    let scaled_height = 16u32.div_ceil(scale.denominator());
    let scaled_stride = scaled_width as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut expected_scaled = vec![0u8; scaled_stride * scaled_height as usize];
    decoder
        .decode_scaled_into(
            &mut expected_scaled,
            scaled_stride,
            PixelFormat::Rgb8,
            scale,
        )
        .expect("reference scaled decode");
    let mut scaled = vec![0u8; expected_scaled.len()];
    decode_tile_scaled_into_in_context(
        &jpeg,
        &mut context,
        &mut pool,
        TileDecodeOutput {
            out: &mut scaled,
            stride: scaled_stride,
            fmt: PixelFormat::Rgb8,
        },
        scale,
    )
    .expect("leased scaled decode");
    assert_eq!(scaled, expected_scaled);

    let output_roi = scaled_rect(roi, scale);
    let region_scaled_stride = output_roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut expected_region_scaled = vec![0u8; region_scaled_stride * output_roi.h as usize];
    decoder
        .decode_region_scaled_into(
            &mut expected_region_scaled,
            region_scaled_stride,
            PixelFormat::Rgb8,
            roi,
            scale,
        )
        .expect("reference region-scaled decode");
    let mut region_scaled = vec![0u8; expected_region_scaled.len()];
    decode_tile_region_scaled_into_in_context(
        &jpeg,
        &mut context,
        &mut pool,
        TileDecodeOutput {
            out: &mut region_scaled,
            stride: region_scaled_stride,
            fmt: PixelFormat::Rgb8,
        },
        roi,
        scale,
    )
    .expect("leased region-scaled decode");
    assert_eq!(region_scaled, expected_region_scaled);
    assert_eq!(context.decode_plan_clone_count(), clones_after_miss);
}

#[test]
fn cached_plan_is_restored_after_context_decode_error() {
    let jpeg = wide_nonrestart_420_jpeg();
    let stride = 96 * PixelFormat::Rgb8.bytes_per_pixel();
    let mut output = vec![0u8; stride * 16];
    let mut context = DecoderContext::new();
    let mut pool = ScratchPool::new();
    decode_tile_into_in_context(
        &jpeg,
        &mut context,
        &mut pool,
        &mut output,
        stride,
        PixelFormat::Rgb8,
    )
    .expect("warm context plan");
    let clones = context.decode_plan_clone_count();
    let occupied = context.cache_stats().occupied_slots;
    let retained = context.retained_allocation_bytes();

    let error = decode_tile_into_in_context(
        &jpeg,
        &mut context,
        &mut pool,
        &mut [],
        stride,
        PixelFormat::Rgb8,
    )
    .expect_err("short output must fail");
    assert!(matches!(error, JpegError::OutputBufferTooSmall { .. }));
    assert_eq!(context.decode_plan_clone_count(), clones);
    assert_eq!(context.cache_stats().occupied_slots, occupied);
    assert_eq!(context.retained_allocation_bytes(), retained);

    decode_tile_into_in_context(
        &jpeg,
        &mut context,
        &mut pool,
        &mut output,
        stride,
        PixelFormat::Rgb8,
    )
    .expect("restored cache plan remains reusable");
    assert_eq!(context.decode_plan_clone_count(), clones);
}

#[test]
fn warmed_batch_planning_reuses_the_cached_plan_without_cloning() {
    let jpeg = wide_nonrestart_420_jpeg();
    let mut context = DecoderContext::new();
    let first = planned_jpeg_tile_decode_live_bytes(
        &jpeg,
        &mut context,
        PixelFormat::Rgb8,
        None,
        Downscale::None,
        DecodeOptions::default(),
    )
    .expect("first batch plan");
    let clones = context.decode_plan_clone_count();
    let misses = context.cache_stats().misses;
    let second = planned_jpeg_tile_decode_live_bytes(
        &jpeg,
        &mut context,
        PixelFormat::Rgb8,
        None,
        Downscale::None,
        DecodeOptions::default(),
    )
    .expect("warmed batch plan");

    assert_eq!(second, first);
    assert_eq!(context.decode_plan_clone_count(), clones);
    assert_eq!(context.cache_stats().misses, misses);
}

#[test]
fn panicking_context_operation_safely_evicts_the_leased_plan() {
    let jpeg = wide_nonrestart_420_jpeg();
    let stride = 96 * PixelFormat::Rgb8.bytes_per_pixel();
    let mut output = vec![0u8; stride * 16];
    let mut context = DecoderContext::new();
    let mut pool = ScratchPool::new();
    decode_tile_into_in_context(
        &jpeg,
        &mut context,
        &mut pool,
        &mut output,
        stride,
        PixelFormat::Rgb8,
    )
    .expect("warm context plan");
    let retained = context.retained_allocation_bytes();
    let evictions = context.cache_stats().evictions;

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let view = JpegView::parse(&jpeg).expect("parse panic fixture");
        let _: Result<(), JpegError> = Decoder::with_view_in_context(view, &mut context, |_| {
            panic!("intentional leased operation panic")
        });
    }));
    assert!(panic.is_err());
    assert!(context.retained_allocation_bytes() < retained);
    assert_eq!(context.cache_stats().evictions, evictions + 1);

    decode_tile_into_in_context(
        &jpeg,
        &mut context,
        &mut pool,
        &mut output,
        stride,
        PixelFormat::Rgb8,
    )
    .expect("context remains valid after panic eviction");
}

#[test]
fn cached_plan_lease_preflight_preserves_cache_at_live_budget_boundary() {
    let jpeg = wide_nonrestart_420_jpeg();
    let mut context = DecoderContext::new();
    Decoder::from_view_in_context(
        JpegView::parse(&jpeg).expect("parse cache warm fixture"),
        &mut context,
    )
    .expect("warm cached plan");
    let view = JpegView::parse(&jpeg).expect("parse preflight fixture");
    let scan_offset = view.header.sos_offset.expect("baseline fixture SOS");
    let prefix = &jpeg[..scan_offset];
    let parsed_bytes = view
        .header
        .retained_allocation_bytes()
        .expect("parsed header bytes");
    let exact_external = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES
        .checked_sub(parsed_bytes)
        .and_then(|remaining| remaining.checked_sub(context.retained_allocation_bytes()))
        .expect("fixture leaves external live budget");
    let retained = context.retained_allocation_bytes();
    let occupied = context.cache_stats().occupied_slots;
    let plan = context
        .cached_decode_plan(prefix)
        .expect("warmed exact-key plan");

    Decoder::validate_cached_plan_lease(&view.header, &context, plan, exact_external)
        .expect("exact parsed plus context live boundary");
    assert!(matches!(
        Decoder::validate_cached_plan_lease(&view.header, &context, plan, exact_external + 1),
        Err(JpegError::MemoryCapExceeded { .. })
    ));
    assert_eq!(context.retained_allocation_bytes(), retained);
    assert_eq!(context.cache_stats().occupied_slots, occupied);
    assert_eq!(
        context
            .cached_decode_plan(prefix)
            .expect("rejected preflight preserves cached plan")
            .scan_offset,
        plan.scan_offset
    );
}

#[test]
fn one_shot_and_context_tile_routes_match_decoder_output() {
    let (expected, stride) = full_rgb8_reference();
    let mut one_shot = vec![0u8; expected.len()];
    let mut context_default = vec![0u8; expected.len()];
    let mut context_explicit = vec![0u8; expected.len()];
    let mut pool = ScratchPool::new();

    let one_shot_outcome =
        decode_tile_into(JPEG, &mut pool, &mut one_shot, stride, PixelFormat::Rgb8)
            .expect("one-shot tile decode");
    let mut context = DecoderContext::new();
    let default_outcome = decode_tile_into_in_context(
        JPEG,
        &mut context,
        &mut pool,
        &mut context_default,
        stride,
        PixelFormat::Rgb8,
    )
    .expect("context tile decode");
    let explicit_outcome = decode_tile_into_in_context_with_options(
        JPEG,
        &mut context,
        &mut pool,
        &mut context_explicit,
        stride,
        PixelFormat::Rgb8,
        DecodeOptions::default(),
    )
    .expect("explicit context tile decode");

    assert_eq!(one_shot, expected);
    assert_eq!(context_default, expected);
    assert_eq!(context_explicit, expected);
    assert_eq!(one_shot_outcome.decoded, Rect::full((16, 16)));
    assert_eq!(default_outcome, one_shot_outcome);
    assert_eq!(explicit_outcome, one_shot_outcome);
}

#[test]
fn context_region_routes_match_decoder_output() {
    let roi = Rect {
        x: 3,
        y: 4,
        w: 9,
        h: 7,
    };
    let stride = roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut expected = vec![0u8; stride * roi.h as usize];
    Decoder::new(JPEG)
        .expect("fixture decoder")
        .decode_region_into(&mut expected, stride, PixelFormat::Rgb8, roi)
        .expect("reference region decode");
    let mut default_output = vec![0u8; expected.len()];
    let mut explicit_output = vec![0u8; expected.len()];
    let mut context = DecoderContext::new();
    let mut pool = ScratchPool::new();

    let default_outcome = decode_tile_region_into_in_context(
        JPEG,
        &mut context,
        &mut pool,
        TileDecodeOutput {
            out: &mut default_output,
            stride,
            fmt: PixelFormat::Rgb8,
        },
        roi,
    )
    .expect("default region route");
    let explicit_outcome = decode_tile_region_into_in_context_with_options(
        JPEG,
        &mut context,
        &mut pool,
        TileDecodeOutput {
            out: &mut explicit_output,
            stride,
            fmt: PixelFormat::Rgb8,
        },
        roi,
        DecodeOptions::default(),
    )
    .expect("explicit region route");

    assert_eq!(default_output, expected);
    assert_eq!(explicit_output, expected);
    assert_eq!(default_outcome.decoded, roi);
    assert_eq!(explicit_outcome, default_outcome);
}

#[test]
fn context_scaled_routes_cover_all_reduced_idct_kernels() {
    let decoder = Decoder::new(JPEG).expect("fixture decoder");
    let mut context = DecoderContext::new();
    let mut pool = ScratchPool::new();

    for scale in [Downscale::Half, Downscale::Quarter, Downscale::Eighth] {
        let dimensions = scaled_dimensions(scale);
        let stride = dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
        let mut expected = vec![0u8; stride * dimensions.1 as usize];
        decoder
            .decode_scaled_into(&mut expected, stride, PixelFormat::Rgb8, scale)
            .expect("reference scaled decode");
        let mut default_output = vec![0u8; expected.len()];
        let mut explicit_output = vec![0u8; expected.len()];

        decode_tile_scaled_into_in_context(
            JPEG,
            &mut context,
            &mut pool,
            TileDecodeOutput {
                out: &mut default_output,
                stride,
                fmt: PixelFormat::Rgb8,
            },
            scale,
        )
        .expect("default scaled route");
        decode_tile_scaled_into_in_context_with_options(
            JPEG,
            &mut context,
            &mut pool,
            TileDecodeOutput {
                out: &mut explicit_output,
                stride,
                fmt: PixelFormat::Rgb8,
            },
            scale,
            DecodeOptions::default(),
        )
        .expect("explicit scaled route");

        assert_eq!(default_output, expected, "default scale {scale:?}");
        assert_eq!(explicit_output, expected, "explicit scale {scale:?}");
    }
}

#[test]
fn context_region_scaled_routes_match_decoder_output() {
    let roi = Rect {
        x: 3,
        y: 4,
        w: 9,
        h: 7,
    };
    let decoder = Decoder::new(JPEG).expect("fixture decoder");
    let mut context = DecoderContext::new();
    let mut pool = ScratchPool::new();

    for scale in [Downscale::Half, Downscale::Quarter, Downscale::Eighth] {
        let output_rect = scaled_rect(roi, scale);
        let stride = output_rect.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
        let mut expected = vec![0u8; stride * output_rect.h as usize];
        decoder
            .decode_region_scaled_into(&mut expected, stride, PixelFormat::Rgb8, roi, scale)
            .expect("reference region-scaled decode");
        let mut default_output = vec![0u8; expected.len()];
        let mut explicit_output = vec![0u8; expected.len()];

        decode_tile_region_scaled_into_in_context(
            JPEG,
            &mut context,
            &mut pool,
            TileDecodeOutput {
                out: &mut default_output,
                stride,
                fmt: PixelFormat::Rgb8,
            },
            roi,
            scale,
        )
        .expect("default region-scaled route");
        decode_tile_region_scaled_into_in_context_with_options(
            JPEG,
            &mut context,
            &mut pool,
            TileDecodeOutput {
                out: &mut explicit_output,
                stride,
                fmt: PixelFormat::Rgb8,
            },
            roi,
            scale,
            DecodeOptions::default(),
        )
        .expect("explicit region-scaled route");

        assert_eq!(default_output, expected, "default scale {scale:?}");
        assert_eq!(explicit_output, expected, "explicit scale {scale:?}");
    }
}

#[test]
fn narrow_middle_region_skips_outer_mcus_without_desynchronizing_entropy() {
    let jpeg = wide_nonrestart_420_jpeg();
    let decoder = Decoder::new(&jpeg).expect("wide nonrestart fixture decoder");
    let roi = Rect {
        x: 40,
        y: 0,
        w: 16,
        h: 16,
    };
    let stride = roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let full_stride = 96 * PixelFormat::Rgb8.bytes_per_pixel();
    let mut full = vec![0u8; full_stride * 16];
    decoder
        .decode_into(&mut full, full_stride, PixelFormat::Rgb8)
        .expect("reference full decode");
    let expected = crop_rgb8(&full, full_stride, roi);
    let mut actual = vec![0u8; expected.len()];
    let mut context = DecoderContext::new();
    let mut pool = ScratchPool::new();

    decode_tile_region_into_in_context(
        &jpeg,
        &mut context,
        &mut pool,
        TileDecodeOutput {
            out: &mut actual,
            stride,
            fmt: PixelFormat::Rgb8,
        },
        roi,
    )
    .expect("context region decode");
    assert_eq!(actual, expected);

    for scale in [Downscale::Half, Downscale::Quarter, Downscale::Eighth] {
        let output_rect = scaled_rect(roi, scale);
        let full_dimensions = (
            96u32.div_ceil(scale.denominator()),
            16u32.div_ceil(scale.denominator()),
        );
        let full_stride = full_dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
        let mut full = vec![0u8; full_stride * full_dimensions.1 as usize];
        decoder
            .decode_scaled_into(&mut full, full_stride, PixelFormat::Rgb8, scale)
            .expect("reference full scaled decode");
        let expected = crop_rgb8(&full, full_stride, output_rect);
        let stride = output_rect.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
        let mut actual = vec![0u8; expected.len()];
        decode_tile_region_scaled_into_in_context(
            &jpeg,
            &mut context,
            &mut pool,
            TileDecodeOutput {
                out: &mut actual,
                stride,
                fmt: PixelFormat::Rgb8,
            },
            roi,
            scale,
        )
        .expect("context region-scaled decode");

        assert_eq!(actual, expected, "scale {scale:?}");
    }
}

#[test]
fn prepared_batch_facade_returns_ordered_tile_metadata() {
    let prepared = prepare_tiff_jpeg_tile(JPEG, None, JpegTilePrepareOptions::default())
        .expect("prepared fixture");
    let (expected, stride) = full_rgb8_reference();
    let mut output = vec![0u8; expected.len()];
    let results = decode_prepared_jpeg_tiles_rgb8(&mut [PreparedJpegTileJob {
        input: prepared,
        out: &mut output,
        stride,
        options: DecodeOptions::default(),
    }])
    .expect("prepared batch infrastructure");

    assert_eq!(output, expected);
    assert_eq!(results.len(), 1);
    let decoded = results[0].as_ref().expect("prepared tile decode");
    assert_eq!(decoded.dimensions, (16, 16));
    assert_eq!(decoded.decoded, Rect::full((16, 16)));
}

#[test]
fn tile_planning_accounts_full_roi_and_rejects_out_of_bounds_roi() {
    let mut context = DecoderContext::new();
    let full = planned_jpeg_tile_decode_live_bytes(
        JPEG,
        &mut context,
        PixelFormat::Rgb8,
        None,
        Downscale::None,
        DecodeOptions::default(),
    )
    .expect("full tile plan");
    let roi = Rect {
        x: 2,
        y: 3,
        w: 8,
        h: 7,
    };
    let region = planned_jpeg_tile_decode_live_bytes(
        JPEG,
        &mut context,
        PixelFormat::Rgb8,
        Some(roi),
        Downscale::Quarter,
        DecodeOptions::default(),
    )
    .expect("region tile plan");
    let error = planned_jpeg_tile_decode_live_bytes(
        JPEG,
        &mut context,
        PixelFormat::Rgb8,
        Some(Rect {
            x: 15,
            y: 15,
            w: 2,
            h: 2,
        }),
        Downscale::None,
        DecodeOptions::default(),
    )
    .expect_err("out-of-bounds region must fail planning");

    assert!(full.worker_live_bytes > 0);
    assert!(full.retained_result_bytes > 0);
    assert!(region.worker_live_bytes > 0);
    assert!(matches!(error, JpegError::RectOutOfBounds { .. }));
}

#[test]
fn checkpoint_planning_only_reserves_for_partial_nonrestart_decode() {
    let full_decoder = Decoder::new(JPEG).expect("fixture decoder");
    let full = Rect::full(full_decoder.info().dimensions);
    assert_eq!(
        planned_roi_checkpoint_bytes(&full_decoder, full).expect("full checkpoint plan"),
        0
    );

    let partial = Rect {
        x: 1,
        y: 1,
        w: 8,
        h: 8,
    };
    assert!(
        planned_roi_checkpoint_bytes(&full_decoder, partial).expect("partial checkpoint plan") > 0
    );

    let restart_bytes = j2k_test_support::baseline_420_restart_32x16_jpeg();
    let restart_decoder = Decoder::new(&restart_bytes).expect("restart fixture decoder");
    assert_eq!(
        planned_roi_checkpoint_bytes(&restart_decoder, partial).expect("restart checkpoint plan"),
        0
    );
}

#[test]
fn malformed_one_shot_input_does_not_mutate_output() {
    let mut output = [0xa5; 16];
    let mut pool = ScratchPool::new();
    let error = decode_tile_into(b"not a jpeg", &mut pool, &mut output, 4, PixelFormat::Rgb8)
        .expect_err("malformed input must fail");

    assert_eq!(output, [0xa5; 16]);
    assert!(!error.is_api_misuse());
}
