// SPDX-License-Identifier: Apache-2.0

#![cfg(target_arch = "aarch64")]

use slidecodec_jpeg::bench_support::{
    bench_rgb_row_pair_from_420, bench_rgb_row_pair_from_420_reference,
    bench_rgb_row_pair_from_420_with_stats, Bench420DispatchStats,
};

fn seeded_row(len: usize, seed: u8, step: u8) -> Vec<u8> {
    let seed = usize::from(seed);
    let step = usize::from(step);
    (0..len)
        .map(|i| {
            let mixed = i.wrapping_mul(step).wrapping_add(seed);
            (mixed ^ (mixed >> 8) ^ (mixed >> 16)) as u8
        })
        .collect()
}

fn assert_dual_row_pair_matches_reference(width: usize) {
    let chroma_width = width.div_ceil(2);
    let y_top = seeded_row(width, 11, 37);
    let y_bottom = seeded_row(width, 203, 19);
    let prev_cb = seeded_row(chroma_width, 7, 13);
    let curr_cb = seeded_row(chroma_width, 41, 17);
    let next_cb = seeded_row(chroma_width, 89, 23);
    let prev_cr = seeded_row(chroma_width, 3, 29);
    let curr_cr = seeded_row(chroma_width, 67, 31);
    let next_cr = seeded_row(chroma_width, 131, 37);

    let mut expected_top = vec![0u8; width * 3];
    let mut expected_bottom = vec![0u8; width * 3];
    bench_rgb_row_pair_from_420_reference(
        &y_top,
        Some(&y_bottom),
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        &mut expected_top,
        Some(&mut expected_bottom),
    );

    let mut actual_top = vec![0u8; width * 3];
    let mut actual_bottom = vec![0u8; width * 3];
    let mut stats = Bench420DispatchStats::default();
    bench_rgb_row_pair_from_420_with_stats(
        &y_top,
        Some(&y_bottom),
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        &mut actual_top,
        Some(&mut actual_bottom),
        &mut stats,
    );

    assert_eq!(
        actual_top, expected_top,
        "top row diverged at width {width}"
    );
    assert_eq!(
        actual_bottom, expected_bottom,
        "bottom row diverged at width {width}"
    );
}

fn assert_top_only_row_pair_matches_reference(width: usize) {
    let chroma_width = width.div_ceil(2);
    let y_top = seeded_row(width, 11, 37);
    let prev_cb = seeded_row(chroma_width, 7, 13);
    let curr_cb = seeded_row(chroma_width, 41, 17);
    let next_cb = seeded_row(chroma_width, 89, 23);
    let prev_cr = seeded_row(chroma_width, 3, 29);
    let curr_cr = seeded_row(chroma_width, 67, 31);
    let next_cr = seeded_row(chroma_width, 131, 37);

    let mut expected_top = vec![0u8; width * 3];
    bench_rgb_row_pair_from_420_reference(
        &y_top,
        None,
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        &mut expected_top,
        None,
    );

    let mut actual_top = vec![0u8; width * 3];
    bench_rgb_row_pair_from_420(
        &y_top,
        None,
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        &mut actual_top,
        None,
    );

    assert_eq!(
        actual_top, expected_top,
        "top-only row diverged at width {width}"
    );
}

#[test]
fn neon_420_row_pair_matches_reference_for_odd_widths_and_tail_cases() {
    for &width in &[3usize, 7, 15, 17, 255, 257] {
        assert_dual_row_pair_matches_reference(width);
        assert_top_only_row_pair_matches_reference(width);
    }
}

#[test]
#[cfg(not(feature = "scalar-only"))]
fn neon_420_row_pair_width_255_stays_on_neon_for_tail_dispatch() {
    let width = 255usize;
    let chroma_width = width.div_ceil(2);
    let y_top = seeded_row(width, 11, 37);
    let y_bottom = seeded_row(width, 203, 19);
    let prev_cb = seeded_row(chroma_width, 7, 13);
    let curr_cb = seeded_row(chroma_width, 41, 17);
    let next_cb = seeded_row(chroma_width, 89, 23);
    let prev_cr = seeded_row(chroma_width, 3, 29);
    let curr_cr = seeded_row(chroma_width, 67, 31);
    let next_cr = seeded_row(chroma_width, 131, 37);

    let mut stats = Bench420DispatchStats::default();
    let mut actual_top = vec![0u8; width * 3];
    let mut actual_bottom = vec![0u8; width * 3];
    bench_rgb_row_pair_from_420_with_stats(
        &y_top,
        Some(&y_bottom),
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        &mut actual_top,
        Some(&mut actual_bottom),
        &mut stats,
    );

    assert_eq!(
        stats.scalar_chunks(),
        0,
        "width 255 should not use scalar fallback"
    );
    assert!(
        stats.neon_tail_chunks() > 0,
        "width 255 should exercise a NEON tail chunk"
    );
}
