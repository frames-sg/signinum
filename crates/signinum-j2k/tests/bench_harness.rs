// SPDX-License-Identifier: Apache-2.0

#[test]
fn public_api_bench_exposes_cpu_decode_regression_surface() {
    let bench = include_str!("../benches/public_api.rs");

    for expected in [
        "j2k_public_decode_region_scaled",
        "j2k_public_decode_rows",
        "j2k_public_tile_batch",
        "rgb8_region_scaled_64x64_q4",
        "gray8_rows_128x128",
        "gray8_repeated_batch_16",
        "gray8_distinct_batch_16",
        "htj2k_gray8_repeated_batch_16",
        "htj2k_gray8_full_512x512",
    ] {
        assert!(
            bench.contains(expected),
            "public API benchmark is missing `{expected}`"
        );
    }
}
