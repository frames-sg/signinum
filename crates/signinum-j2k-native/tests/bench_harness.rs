// SPDX-License-Identifier: Apache-2.0

#[test]
fn native_bench_exposes_classic_and_htj2k_code_block_surface() {
    let bench = include_str!("../benches/tier1_bitplane.rs");

    for expected in [
        "tier1_bitplane_decode",
        "tier1_bitplane_encode",
        "htj2k_cleanup_decode",
        "htj2k_cleanup_encode",
        "decode_64x64",
        "encode_64x64",
    ] {
        assert!(
            bench.contains(expected),
            "native benchmark is missing `{expected}`"
        );
    }
}
