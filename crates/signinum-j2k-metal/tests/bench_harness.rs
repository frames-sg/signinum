// SPDX-License-Identifier: Apache-2.0

#[test]
fn compare_bench_adaptive_region_scaled_uses_auto_device_submit() {
    let source = include_str!("../benches/common/mod.rs");

    assert!(
        source.contains("fn signinum_adaptive_decode_tile_batch_region_scaled_to_device("),
        "compare bench must expose a device Auto helper for adaptive ROI+scaled tile batches"
    );

    let device_helper = source
        .split("fn signinum_adaptive_decode_tile_batch_region_scaled_to_device(")
        .nth(1)
        .expect("adaptive ROI+scaled device helper must exist");
    let device_helper_body = device_helper
        .split("pub(crate) fn signinum_adaptive_decode_tile_batch(")
        .next()
        .expect("adaptive ROI+scaled device helper body must be delimited by the next helper");
    assert!(
        device_helper_body.contains("BackendRequest::Auto"),
        "adaptive ROI+scaled device helper must submit through BackendRequest::Auto"
    );

    let adaptive_fn = source
        .split("pub(crate) fn signinum_adaptive_decode_tile_batch_region_scaled(")
        .nth(1)
        .expect("adaptive ROI+scaled benchmark helper must exist");
    let adaptive_body = adaptive_fn
        .split("fn should_auto_use_direct_grayscale_input(")
        .next()
        .expect("adaptive ROI+scaled helper body must be delimited by the next helper");
    assert!(
        adaptive_body.contains("signinum_adaptive_decode_tile_batch_region_scaled_to_device("),
        "adaptive ROI+scaled benchmark helper must call the Auto device-submit helper"
    );
}

#[test]
fn compare_bench_distinct_and_external_region_scaled_expose_auto_variants() {
    let common = include_str!("../benches/common/mod.rs");
    let compare = include_str!("../benches/compare.rs");

    for helper in [
        "signinum_adaptive_decode_tile_batch_region_scaled_distinct",
        "signinum_adaptive_decode_external_tile_batch_region_scaled",
    ] {
        assert!(
            common.contains(&format!("pub(crate) fn {helper}(")),
            "common benchmark helpers must expose {helper}"
        );

        let helper_body = common
            .split(&format!("pub(crate) fn {helper}("))
            .nth(1)
            .expect("helper must exist")
            .split("pub(crate) fn signinum_metal_supports")
            .next()
            .expect("helper body must be delimited before support helpers");
        assert!(
            helper_body.contains("BackendRequest::Auto"),
            "{helper} must submit through BackendRequest::Auto"
        );
        assert!(
            compare.matches(helper).count() >= 2,
            "compare bench must import and register {helper}"
        );
    }
}

#[test]
fn compare_bench_batch_sizes_are_env_configurable() {
    let common = include_str!("../benches/common/mod.rs");
    let compare = include_str!("../benches/compare.rs");

    assert!(
        common.contains("SIGNINUM_J2K_TILE_BATCH_SIZES"),
        "J2K compare benches must expose env-controlled batch-size sweeps"
    );
    assert!(
        common.contains("DEFAULT_J2K_TILE_BATCH_SIZES") && common.contains("&[16, 32, 64, 128]"),
        "default J2K batch-size sweep must cover 16/32/64/128"
    );
    assert!(
        compare.contains("let batch_sizes = j2k_tile_batch_sizes();"),
        "compare bench must parse batch sizes once and reuse them across batch groups"
    );
    assert!(
        compare.contains("external_wsi_tile_batches(max_batch_size)"),
        "external WSI loader must load enough tiles for the largest requested batch"
    );
}

#[test]
fn compare_bench_keeps_multiple_htj2k_gray_sizes() {
    let common = include_str!("../benches/common/mod.rs");

    assert!(
        common.contains("inputs.extend(ht_bench_inputs()"),
        "bench_inputs must include all generated HTJ2K size candidates, not only the first success"
    );
    for fixture_name in ["\"htj2k_gray_1024\"", "\"htj2k_gray_512\""] {
        assert!(
            common.contains(fixture_name),
            "HTJ2K bench coverage must include {fixture_name}"
        );
    }
}

#[test]
fn external_wsi_bench_groups_by_codec_family() {
    let common = include_str!("../benches/common/mod.rs");

    assert!(
        common.contains("enum ExternalCodecFamily"),
        "external WSI batches must track classic J2K versus HTJ2K separately"
    );
    assert!(
        common.contains("external_codec_family("),
        "external WSI loader must classify each tile/frame by compressed transfer syntax"
    );
    assert!(
        common.contains("\"htj2k_gray8\"") && common.contains("\"j2k_gray8\""),
        "external WSI batch labels must keep HTJ2K and classic J2K timing separate"
    );
}

#[test]
fn metal_region_scaled_benches_gate_to_direct_modes() {
    let common = include_str!("../benches/common/mod.rs");

    assert!(
        common.contains("fn supports_metal_region_scaled_mode("),
        "ROI+scaled Metal benchmark support must have a cheap format gate"
    );
    assert!(
        common.contains("DecodeMode::Gray8 | DecodeMode::Gray16 | DecodeMode::Rgb8"),
        "ROI+scaled Metal benchmarks must advertise RGB once RGB is GPU-native"
    );
    for helper in [
        "signinum_metal_supports_region_scaled",
        "signinum_metal_supports_tile_batch_region_scaled_distinct",
        "signinum_metal_supports_external_tile_batch_region_scaled",
    ] {
        let helper_body = common
            .split(&format!("pub(crate) fn {helper}("))
            .nth(1)
            .expect("support helper must exist")
            .split("pub(crate) fn")
            .next()
            .expect("support helper body must be delimited");
        assert!(
            helper_body.contains("supports_metal_region_scaled_mode("),
            "{helper} must skip unsupported ROI+scaled Metal formats before probing decode"
        );
    }
}

#[test]
fn compare_bench_rgb_region_scaled_reports_resident_vs_cpu_staged_metal() {
    let common = include_str!("../benches/common/mod.rs");
    let compare = include_str!("../benches/compare.rs");

    assert!(
        common.contains("signinum_cpu_staged_metal_decode_tile_batch_region_scaled"),
        "common benchmarks must expose a CPU decode plus CPU-to-Metal upload baseline"
    );
    assert!(
        common.contains("decode_region_scaled_to_cpu_staged_metal_surface_with_session"),
        "CPU-staged baseline must use the explicit CPU-staged Metal API"
    );
    assert!(
        compare.contains("wsi_tile_batch_region_scaled_rgb_q4"),
        "compare bench must include an RGB ROI+scaled batch group"
    );
    assert!(
        compare.contains("signinum-cpu-staged-metal")
            && compare.contains("signinum-metal-resident"),
        "RGB ROI+scaled benches must report CPU-staged Metal and resident hybrid Metal separately"
    );
}

#[test]
fn compare_bench_distinct_rgb_region_scaled_reports_cpu_auto_and_resident_metal() {
    let compare = include_str!("../benches/compare.rs");

    assert!(
        compare.contains("wsi_tile_batch_region_scaled_rgb_distinct_q4"),
        "compare bench must include a distinct RGB ROI+scaled batch group"
    );
    assert!(
        compare.contains("distinct_rgb_tile_batch_inputs(input, count)"),
        "distinct RGB ROI+scaled group must use generated distinct RGB tiles"
    );
    assert!(
        compare.contains("signinum_decode_tile_batch_region_scaled_distinct")
            && compare.contains("signinum_adaptive_decode_tile_batch_region_scaled_distinct")
            && compare.contains("signinum_metal_decode_tile_batch_region_scaled_distinct"),
        "distinct RGB ROI+scaled group must report CPU, Auto, and explicit resident Metal timings"
    );
}

#[test]
fn compare_bench_exposes_htj2k_measurement_first_surfaces() {
    let common = include_str!("../benches/common/mod.rs");
    let compare = include_str!("../benches/compare.rs");

    for expected in [
        "htj2k_region_scaled_plan_build",
        "htj2k_feeder_coalesce",
        "htj2k_metal_route",
    ] {
        assert!(
            compare.contains(expected),
            "compare benchmark is missing `{expected}`"
        );
    }

    for expected in [
        "benchmark_region_scaled_direct_plan_prepare",
        "benchmark_group_region_scaled_requests",
        "\"htj2k_rgb_512\"",
    ] {
        assert!(
            common.contains(expected),
            "common benchmark helpers are missing `{expected}`"
        );
    }
}
