// SPDX-License-Identifier: Apache-2.0

#[test]
fn roi_batch_compare_binary_exposes_grok_wsi_surfaces() {
    let source = include_str!("../src/bin/jp2k_roi_batch_compare.rs");

    for expected in [
        "htj2k_raw_rgb8_512_roi256_q4_repeated_batch16",
        "htj2k_jp2_rgb8_512_roi256_q4_repeated_batch16",
        "htj2k_jp2_rgb8_256_roi128_q4_repeated_batch16",
        "signinum",
        "grok",
    ] {
        assert!(
            source.contains(expected),
            "ROI batch compare binary is missing `{expected}`"
        );
    }
}
