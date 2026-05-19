// SPDX-License-Identifier: Apache-2.0

#[test]
fn native_bench_exposes_classic_and_htj2k_code_block_surface() {
    let tier1_bench = include_str!("../benches/tier1_bitplane.rs");
    let sigprop_bench = include_str!("../benches/htj2k_sigprop_phase.rs");
    let direct_cpu_bench = include_str!("../benches/direct_cpu.rs");

    for expected in [
        "tier1_bitplane_decode",
        "tier1_bitplane_encode",
        "htj2k_cleanup_decode",
        "htj2k_cleanup_encode",
        "htj2k_cleanup_encode_distribution",
        "rho_eq_uq_64x64",
        "htj2k_cleanup_encode_parallel_granularity",
        "serial_128_blocks",
        "rayon_par_iter_global_128_blocks",
        "rayon_par_iter_threads",
        "rayon_par_chunks_128_blocks",
        "htj2k_cleanup_encode_parallel_batch_size",
        "j2k_tier1_encode_parallel_batch_size",
        "serial_blocks",
        "rayon_par_iter_global_blocks",
        "htj2k_refinement_fixture_decode",
        "htj2k_refinement_block_decode",
        "ds0_ht_09_b11_full",
        "ds0_ht_09_b11_cleanup",
        "ds0_ht_09_b11_sigprop",
        "ds0_ht_09_b11_magref_full",
        "decode_64x64",
        "encode_64x64",
    ] {
        assert!(
            tier1_bench.contains(expected),
            "native benchmark is missing `{expected}`"
        );
    }

    for expected in [
        "htj2k_refinement_sigprop_phase",
        "ds0_ht_09_b11_sigprop_only",
        "htj2k_cpuupload_decode_batch",
        "ds0_ht_09_b11_scalar_batch",
    ] {
        assert!(
            sigprop_bench.contains(expected),
            "native SigProp benchmark is missing `{expected}`"
        );
    }

    for expected in [
        "j2k_native_direct_cpu_color_plan",
        "htj2k_rgb8_roi256_q4_fresh_scratch",
        "htj2k_rgb8_roi256_q4_reuse_scratch",
        "htj2k_rgba8_roi256_q4_reuse_scratch",
    ] {
        assert!(
            direct_cpu_bench.contains(expected),
            "native direct CPU benchmark is missing `{expected}`"
        );
    }
}
