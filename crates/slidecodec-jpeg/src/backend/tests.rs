// SPDX-License-Identifier: Apache-2.0

use crate::color::upsample::upsample_h2v2_fancy_rows;
use crate::color::ycbcr::ycbcr_to_rgb;
use alloc::vec;
use alloc::vec::Vec;

use super::scalar;

#[test]
fn gray_rows_expand_to_equal_rgb_channels() {
    let gray = [10u8, 40, 90, 200];
    let mut dst = vec![0u8; gray.len() * 3];
    scalar::fill_rgb_row_from_gray(&gray, &mut dst);
    assert_eq!(dst, vec![10, 10, 10, 40, 40, 40, 90, 90, 90, 200, 200, 200]);
}

#[test]
fn ycbcr_rows_match_per_pixel_reference() {
    let y = [16u8, 40, 90, 200];
    let cb = [128u8, 100, 200, 180];
    let cr = [128u8, 220, 10, 90];
    let mut dst = vec![0u8; y.len() * 3];
    scalar::fill_rgb_row_from_ycbcr(&y, &cb, &cr, &mut dst);

    let expected: Vec<u8> = y
        .iter()
        .zip(cb.iter())
        .zip(cr.iter())
        .flat_map(|((&y, &cb), &cr)| {
            let (r, g, b) = ycbcr_to_rgb(y, cb, cr);
            [r, g, b]
        })
        .collect();

    assert_eq!(dst, expected);
}

#[test]
fn ycbcr_420_row_pair_matches_reference() {
    let y_top = [16u8, 24, 32, 40, 48, 56, 64, 72];
    let y_bot = [80u8, 88, 96, 104, 112, 120, 128, 136];
    let prev_cb = [120u8, 100, 140, 160];
    let curr_cb = [110u8, 90, 130, 170];
    let next_cb = [100u8, 80, 120, 180];
    let prev_cr = [130u8, 150, 170, 190];
    let curr_cr = [140u8, 160, 180, 200];
    let next_cr = [150u8, 170, 190, 210];
    let mut dst_top = vec![0u8; y_top.len() * 3];
    let mut dst_bot = vec![0u8; y_bot.len() * 3];

    scalar::fill_rgb_row_pair_from_420(
        &y_top,
        Some(&y_bot),
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        &mut dst_top,
        Some(&mut dst_bot),
    );

    let mut cb_top = vec![0u8; y_top.len()];
    let mut cb_bot = vec![0u8; y_top.len()];
    let mut cr_top = vec![0u8; y_top.len()];
    let mut cr_bot = vec![0u8; y_top.len()];
    upsample_h2v2_fancy_rows(
        &prev_cb,
        &curr_cb,
        &next_cb,
        y_top.len(),
        &mut cb_top,
        &mut cb_bot,
    );
    upsample_h2v2_fancy_rows(
        &prev_cr,
        &curr_cr,
        &next_cr,
        y_top.len(),
        &mut cr_top,
        &mut cr_bot,
    );

    let expected_top: Vec<u8> = y_top
        .iter()
        .zip(cb_top.iter())
        .zip(cr_top.iter())
        .flat_map(|((&y, &cb), &cr)| {
            let (r, g, b) = ycbcr_to_rgb(y, cb, cr);
            [r, g, b]
        })
        .collect();
    let expected_bot: Vec<u8> = y_bot
        .iter()
        .zip(cb_bot.iter())
        .zip(cr_bot.iter())
        .flat_map(|((&y, &cb), &cr)| {
            let (r, g, b) = ycbcr_to_rgb(y, cb, cr);
            [r, g, b]
        })
        .collect();

    assert_eq!(dst_top, expected_top);
    assert_eq!(dst_bot, expected_bot);
}

#[test]
fn backend_scalar_420_row_pair_matches_reference_for_odd_widths() {
    let backend = super::Backend {
        kind: super::BackendKind::Scalar,
    };
    let y_top = [16u8, 24, 32, 40, 48, 56, 64];
    let y_bot = [80u8, 88, 96, 104, 112, 120, 128];
    let prev_cb = [120u8, 100, 140, 160];
    let curr_cb = [110u8, 90, 130, 170];
    let next_cb = [100u8, 80, 120, 180];
    let prev_cr = [130u8, 150, 170, 190];
    let curr_cr = [140u8, 160, 180, 200];
    let next_cr = [150u8, 170, 190, 210];

    let mut expected_top = vec![0u8; y_top.len() * 3];
    let mut expected_bot = vec![0u8; y_bot.len() * 3];
    scalar::fill_rgb_row_pair_from_420(
        &y_top,
        Some(&y_bot),
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        &mut expected_top,
        Some(&mut expected_bot),
    );

    let mut actual_top = vec![0u8; y_top.len() * 3];
    let mut actual_bot = vec![0u8; y_bot.len() * 3];
    backend.fill_rgb_row_pair_from_420(
        &y_top,
        Some(&y_bot),
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        &mut actual_top,
        Some(&mut actual_bot),
    );

    assert_eq!(actual_top, expected_top);
    assert_eq!(actual_bot, expected_bot);
}

#[test]
fn backend_scalar_420_row_pair_handles_missing_bottom_row() {
    let backend = super::Backend {
        kind: super::BackendKind::Scalar,
    };
    let y_top = [16u8, 24, 32, 40, 48];
    let prev_cb = [120u8, 100, 140];
    let curr_cb = [110u8, 90, 130];
    let next_cb = [100u8, 80, 120];
    let prev_cr = [130u8, 150, 170];
    let curr_cr = [140u8, 160, 180];
    let next_cr = [150u8, 170, 190];

    let mut expected_top = vec![0u8; y_top.len() * 3];
    scalar::fill_rgb_row_pair_from_420(
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

    let mut actual_top = vec![0u8; y_top.len() * 3];
    backend.fill_rgb_row_pair_from_420(
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

    assert_eq!(actual_top, expected_top);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_ycbcr_rows_match_scalar_reference_for_tail_widths() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let y = [0u8, 16, 33, 64, 96, 127, 128, 129, 160, 192, 224, 255, 12];
    let cb = [255u8, 240, 200, 180, 160, 140, 128, 120, 96, 64, 32, 16, 0];
    let cr = [0u8, 15, 32, 64, 96, 120, 128, 136, 160, 192, 224, 240, 255];
    let mut expected = vec![0u8; y.len() * 3];
    let mut actual = vec![0u8; y.len() * 3];

    scalar::fill_rgb_row_from_ycbcr(&y, &cb, &cr, &mut expected);
    super::x86::fill_rgb_row_from_ycbcr_for_test(&y, &cb, &cr, &mut actual);

    assert_eq!(actual, expected);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_gray_rows_match_scalar_reference() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let gray = [0u8, 16, 33, 64, 96, 127, 128, 129, 160, 192, 224, 255, 12];
    let mut expected = vec![0u8; gray.len() * 3];
    let mut actual = vec![0u8; gray.len() * 3];

    scalar::fill_rgb_row_from_gray(&gray, &mut expected);
    super::x86::fill_rgb_row_from_gray_for_test(&gray, &mut actual);

    assert_eq!(actual, expected);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_rgb_rows_match_scalar_reference() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let r = [0u8, 16, 33, 64, 96, 127, 128, 129, 160, 192, 224, 255, 12];
    let g = [255u8, 240, 200, 180, 160, 140, 128, 120, 96, 64, 32, 16, 0];
    let b = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
    let mut expected = vec![0u8; r.len() * 3];
    let mut actual = vec![0u8; r.len() * 3];

    scalar::fill_rgb_row_from_rgb(&r, &g, &b, &mut expected);
    super::x86::fill_rgb_row_from_rgb_for_test(&r, &g, &b, &mut actual);

    assert_eq!(actual, expected);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_ycbcr_rows_match_scalar_reference_across_multiple_chunks() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let len = 31usize;
    let y: Vec<u8> = (0..len)
        .map(|i| ((i as u8).wrapping_mul(37)).wrapping_add(11))
        .collect();
    let cb: Vec<u8> = (0..len)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(29)))
        .collect();
    let cr: Vec<u8> = (0..len)
        .map(|i| ((i as u8).wrapping_mul(53)).wrapping_add(97))
        .collect();
    let mut expected = vec![0u8; len * 3];
    let mut actual = vec![0u8; len * 3];

    scalar::fill_rgb_row_from_ycbcr(&y, &cb, &cr, &mut expected);
    super::x86::fill_rgb_row_from_ycbcr_for_test(&y, &cb, &cr, &mut actual);

    assert_eq!(actual, expected);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_420_row_pair_matches_scalar_reference() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let backend = super::Backend {
        kind: super::BackendKind::Avx2,
    };
    let y_top = [16u8, 24, 32, 40, 48, 56, 64, 72];
    let y_bot = [80u8, 88, 96, 104, 112, 120, 128, 136];
    let prev_cb = [120u8, 100, 140, 160];
    let curr_cb = [110u8, 90, 130, 170];
    let next_cb = [100u8, 80, 120, 180];
    let prev_cr = [130u8, 150, 170, 190];
    let curr_cr = [140u8, 160, 180, 200];
    let next_cr = [150u8, 170, 190, 210];

    let mut expected_top = vec![0u8; y_top.len() * 3];
    let mut expected_bot = vec![0u8; y_bot.len() * 3];
    scalar::fill_rgb_row_pair_from_420(
        &y_top,
        Some(&y_bot),
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        &mut expected_top,
        Some(&mut expected_bot),
    );

    let mut actual_top = vec![0u8; y_top.len() * 3];
    let mut actual_bot = vec![0u8; y_bot.len() * 3];
    backend.fill_rgb_row_pair_from_420(
        &y_top,
        Some(&y_bot),
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        &mut actual_top,
        Some(&mut actual_bot),
    );

    assert_eq!(actual_top, expected_top);
    assert_eq!(actual_bot, expected_bot);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_420_row_pair_matches_scalar_reference_for_odd_widths() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let backend = super::Backend {
        kind: super::BackendKind::Avx2,
    };
    let y_top = [16u8, 24, 32, 40, 48, 56, 64];
    let y_bot = [80u8, 88, 96, 104, 112, 120, 128];
    let prev_cb = [120u8, 100, 140, 160];
    let curr_cb = [110u8, 90, 130, 170];
    let next_cb = [100u8, 80, 120, 180];
    let prev_cr = [130u8, 150, 170, 190];
    let curr_cr = [140u8, 160, 180, 200];
    let next_cr = [150u8, 170, 190, 210];

    let mut expected_top = vec![0u8; y_top.len() * 3];
    let mut expected_bot = vec![0u8; y_bot.len() * 3];
    scalar::fill_rgb_row_pair_from_420(
        &y_top,
        Some(&y_bot),
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        &mut expected_top,
        Some(&mut expected_bot),
    );

    let mut actual_top = vec![0u8; y_top.len() * 3];
    let mut actual_bot = vec![0u8; y_bot.len() * 3];
    backend.fill_rgb_row_pair_from_420(
        &y_top,
        Some(&y_bot),
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        &mut actual_top,
        Some(&mut actual_bot),
    );

    assert_eq!(actual_top, expected_top);
    assert_eq!(actual_bot, expected_bot);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_420_row_pair_handles_missing_bottom_row() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let backend = super::Backend {
        kind: super::BackendKind::Avx2,
    };
    let y_top = [16u8, 24, 32, 40, 48];
    let prev_cb = [120u8, 100, 140];
    let curr_cb = [110u8, 90, 130];
    let next_cb = [100u8, 80, 120];
    let prev_cr = [130u8, 150, 170];
    let curr_cr = [140u8, 160, 180];
    let next_cr = [150u8, 170, 190];

    let mut expected_top = vec![0u8; y_top.len() * 3];
    scalar::fill_rgb_row_pair_from_420(
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

    let mut actual_top = vec![0u8; y_top.len() * 3];
    backend.fill_rgb_row_pair_from_420(
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

    assert_eq!(actual_top, expected_top);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_ycbcr_rows_match_scalar_reference_for_tail_widths() {
    let y = [0u8, 16, 33, 64, 96, 127, 128, 129, 160, 192, 224, 255, 12];
    let cb = [255u8, 240, 200, 180, 160, 140, 128, 120, 96, 64, 32, 16, 0];
    let cr = [0u8, 15, 32, 64, 96, 120, 128, 136, 160, 192, 224, 240, 255];
    let mut expected = vec![0u8; y.len() * 3];
    let mut actual = vec![0u8; y.len() * 3];

    scalar::fill_rgb_row_from_ycbcr(&y, &cb, &cr, &mut expected);
    super::neon::fill_rgb_row_from_ycbcr_for_test(&y, &cb, &cr, &mut actual);

    assert_eq!(actual, expected);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_ycbcr_rows_match_scalar_reference_across_multiple_chunks() {
    let len = 31usize;
    let y: Vec<u8> = (0..len)
        .map(|i| ((i as u8).wrapping_mul(37)).wrapping_add(11))
        .collect();
    let cb: Vec<u8> = (0..len)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(29)))
        .collect();
    let cr: Vec<u8> = (0..len)
        .map(|i| ((i as u8).wrapping_mul(53)).wrapping_add(97))
        .collect();
    let mut expected = vec![0u8; len * 3];
    let mut actual = vec![0u8; len * 3];

    scalar::fill_rgb_row_from_ycbcr(&y, &cb, &cr, &mut expected);
    super::neon::fill_rgb_row_from_ycbcr_for_test(&y, &cb, &cr, &mut actual);

    assert_eq!(actual, expected);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_ycbcr_rows_match_scalar_reference_for_offset_subslice_and_odd_tail_width() {
    let len = 255usize;
    let y_buf: Vec<u8> = (0..len + 3)
        .map(|i| ((i as u8).wrapping_mul(37)).wrapping_add(11))
        .collect();
    let cb_buf: Vec<u8> = (0..len + 3)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(29)))
        .collect();
    let cr_buf: Vec<u8> = (0..len + 3)
        .map(|i| ((i as u8).wrapping_mul(53)).wrapping_add(97))
        .collect();
    let y = &y_buf[1..=len];
    let cb = &cb_buf[1..=len];
    let cr = &cr_buf[1..=len];
    let mut expected = vec![0u8; len * 3];
    let mut actual = vec![0u8; len * 3];

    scalar::fill_rgb_row_from_ycbcr(y, cb, cr, &mut expected);
    super::neon::fill_rgb_row_from_ycbcr_for_test(y, cb, cr, &mut actual);

    assert_eq!(actual, expected);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_420_row_pair_matches_scalar_reference_for_tail_widths() {
    let backend = super::Backend {
        kind: super::BackendKind::Neon,
    };
    let y_top = [0u8, 16, 33, 64, 96, 127, 128, 129, 160, 192, 224, 255, 12];
    let y_bot = [12u8, 255, 224, 192, 160, 129, 128, 127, 96, 64, 33, 16, 0];
    let prev_cb = [255u8, 240, 200, 180, 160, 140, 128];
    let curr_cb = [240u8, 220, 180, 160, 140, 120, 96];
    let next_cb = [220u8, 200, 160, 140, 120, 96, 64];
    let prev_cr = [0u8, 15, 32, 64, 96, 120, 128];
    let curr_cr = [16u8, 32, 64, 96, 120, 136, 160];
    let next_cr = [32u8, 64, 96, 120, 136, 160, 192];

    let mut expected_top = vec![0u8; y_top.len() * 3];
    let mut expected_bot = vec![0u8; y_bot.len() * 3];
    scalar::fill_rgb_row_pair_from_420(
        &y_top,
        Some(&y_bot),
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        &mut expected_top,
        Some(&mut expected_bot),
    );

    let mut actual_top = vec![0u8; y_top.len() * 3];
    let mut actual_bot = vec![0u8; y_bot.len() * 3];
    backend.fill_rgb_row_pair_from_420(
        &y_top,
        Some(&y_bot),
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        &mut actual_top,
        Some(&mut actual_bot),
    );

    assert_eq!(actual_top, expected_top);
    assert_eq!(actual_bot, expected_bot);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_420_row_pair_matches_scalar_reference_across_multiple_chunks() {
    let backend = super::Backend {
        kind: super::BackendKind::Neon,
    };
    let len = 31usize;
    let y_top: Vec<u8> = (0..len)
        .map(|i| ((i as u8).wrapping_mul(37)).wrapping_add(11))
        .collect();
    let y_bot: Vec<u8> = (0..len)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(19)))
        .collect();
    let chroma_len = len.div_ceil(2);
    let prev_cb: Vec<u8> = (0..chroma_len)
        .map(|i| ((i as u8).wrapping_mul(17)).wrapping_add(41))
        .collect();
    let curr_cb: Vec<u8> = (0..chroma_len)
        .map(|i| ((i as u8).wrapping_mul(29)).wrapping_add(13))
        .collect();
    let next_cb: Vec<u8> = (0..chroma_len)
        .map(|i| ((i as u8).wrapping_mul(43)).wrapping_add(7))
        .collect();
    let prev_cr: Vec<u8> = (0..chroma_len)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(11)))
        .collect();
    let curr_cr: Vec<u8> = (0..chroma_len)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(23)))
        .collect();
    let next_cr: Vec<u8> = (0..chroma_len)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(31)))
        .collect();

    let mut expected_top = vec![0u8; len * 3];
    let mut expected_bot = vec![0u8; len * 3];
    scalar::fill_rgb_row_pair_from_420(
        &y_top,
        Some(&y_bot),
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        &mut expected_top,
        Some(&mut expected_bot),
    );

    let mut actual_top = vec![0u8; len * 3];
    let mut actual_bot = vec![0u8; len * 3];
    backend.fill_rgb_row_pair_from_420(
        &y_top,
        Some(&y_bot),
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        &mut actual_top,
        Some(&mut actual_bot),
    );

    assert_eq!(actual_top, expected_top);
    assert_eq!(actual_bot, expected_bot);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_420_row_pair_handles_missing_bottom_row() {
    let backend = super::Backend {
        kind: super::BackendKind::Neon,
    };
    let y_top = [16u8, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96];
    let prev_cb = [120u8, 100, 140, 160, 180, 200];
    let curr_cb = [110u8, 90, 130, 170, 190, 210];
    let next_cb = [100u8, 80, 120, 180, 200, 220];
    let prev_cr = [130u8, 150, 170, 190, 210, 230];
    let curr_cr = [140u8, 160, 180, 200, 220, 240];
    let next_cr = [150u8, 170, 190, 210, 230, 250];

    let mut expected_top = vec![0u8; y_top.len() * 3];
    scalar::fill_rgb_row_pair_from_420(
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

    let mut actual_top = vec![0u8; y_top.len() * 3];
    backend.fill_rgb_row_pair_from_420(
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

    assert_eq!(actual_top, expected_top);
}
