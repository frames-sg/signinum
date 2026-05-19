// SPDX-License-Identifier: Apache-2.0

#[test]
fn public_api_bench_exposes_cpu_decode_regression_surface() {
    let bench = include_str!("../benches/public_api.rs");

    for expected in [
        "j2k_public_cpu_encode_matrix",
        "j2k_public_cpu_decode_matrix",
        "rgb8_512_classic_external",
        "rgb8_512_htj2k_external",
        "rgb8_512_classic_roundtrip",
        "gray8_512_classic_decode",
        "gray8_512_htj2k_decode",
        "rgb8_512_classic_decode",
        "rgb8_512_classic_decode_serial",
        "rgb8_512_htj2k_decode",
        "j2k_public_decode_region_scaled",
        "j2k_public_decode_rows",
        "j2k_public_tile_batch",
        "j2k_public_tile_batch_region_scaled_rgb_q4",
        "rgb8_region_scaled_64x64_q4",
        "gray8_rows_128x128",
        "gray8_repeated_batch_16",
        "gray8_distinct_batch_16",
        "htj2k_gray8_repeated_batch_16",
        "htj2k_gray8_distinct_batch_16",
        "classic_repeated_512_roi256_batch16",
        "classic_distinct_512_roi256_batch16",
        "htj2k_repeated_512_roi256_batch16",
        "htj2k_jp2_rgb8_repeated_512_roi256_batch16",
        "htj2k_jp2_rgba8_repeated_512_roi256_batch16",
        "htj2k_jp2_rgb8_repeated_256_roi128_batch16",
        "htj2k_distinct_512_roi256_batch16",
        "htj2k_gray8_full_512x512",
    ] {
        assert!(
            bench.contains(expected),
            "public API benchmark is missing `{expected}`"
        );
    }
}
