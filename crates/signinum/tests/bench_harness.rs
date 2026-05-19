// SPDX-License-Identifier: Apache-2.0

#[test]
fn facade_bench_exposes_cpu_and_hybrid_encode_surfaces() {
    let bench = include_str!("../benches/facade.rs");

    for expected in [
        "facade_j2k_lossless_encode_cpu_matrix",
        "cpu_only_rgb8_512_classic_external",
        "cpu_only_rgb8_512_htj2k_external",
        "facade_j2k_lossless_encode_hybrid_matrix",
        "facade_auto_rgb8_512_classic_external",
        "facade_auto_rgb8_512_htj2k_external",
        "direct_metal_auto_stage_rgb8_512_classic_external",
        "direct_metal_cpu_rct_stage_rgb8_512_htj2k_external",
    ] {
        assert!(
            bench.contains(expected),
            "facade benchmark is missing `{expected}`"
        );
    }
}
