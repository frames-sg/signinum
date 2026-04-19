// SPDX-License-Identifier: Apache-2.0

use crate::color::ycbcr::ycbcr_to_rgb;

pub(crate) fn fill_rgb_row_from_gray(gray_row: &[u8], dst: &mut [u8]) {
    for (&gray, pixel) in gray_row.iter().zip(dst.chunks_exact_mut(3)) {
        pixel[0] = gray;
        pixel[1] = gray;
        pixel[2] = gray;
    }
}

pub(crate) fn fill_rgb_row_from_rgb(r_row: &[u8], g_row: &[u8], b_row: &[u8], dst: &mut [u8]) {
    for (((&r, &g), &b), pixel) in r_row
        .iter()
        .zip(g_row.iter())
        .zip(b_row.iter())
        .zip(dst.chunks_exact_mut(3))
    {
        pixel[0] = r;
        pixel[1] = g;
        pixel[2] = b;
    }
}

pub(crate) fn fill_rgb_row_from_ycbcr(y_row: &[u8], cb_row: &[u8], cr_row: &[u8], dst: &mut [u8]) {
    for (((&y_sample, &cb_sample), &cr_sample), pixel) in y_row
        .iter()
        .zip(cb_row.iter())
        .zip(cr_row.iter())
        .zip(dst.chunks_exact_mut(3))
    {
        let (r, g, b) = ycbcr_to_rgb(y_sample, cb_sample, cr_sample);
        pixel[0] = r;
        pixel[1] = g;
        pixel[2] = b;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_rgb_row_pair_from_420(
    y_top: &[u8],
    y_bottom: Option<&[u8]>,
    prev_cb: &[u8],
    curr_cb: &[u8],
    next_cb: &[u8],
    prev_cr: &[u8],
    curr_cr: &[u8],
    next_cr: &[u8],
    dst_top: &mut [u8],
    dst_bottom: Option<&mut [u8]>,
) {
    let width = y_top.len();
    debug_assert_eq!(width * 3, dst_top.len());
    debug_assert!(y_bottom.is_none_or(|row| row.len() == width));
    debug_assert!(dst_bottom.as_ref().is_none_or(|row| row.len() == width * 3));

    for (x, pixel) in dst_top.chunks_exact_mut(3).enumerate() {
        let cb = h2v2_sample(prev_cb, curr_cb, x);
        let cr = h2v2_sample(prev_cr, curr_cr, x);
        let (r, g, b) = ycbcr_to_rgb(y_top[x], cb, cr);
        pixel[0] = r;
        pixel[1] = g;
        pixel[2] = b;
    }

    if let (Some(y_bottom), Some(dst_bottom)) = (y_bottom, dst_bottom) {
        for (x, pixel) in dst_bottom.chunks_exact_mut(3).enumerate() {
            let cb = h2v2_sample(next_cb, curr_cb, x);
            let cr = h2v2_sample(next_cr, curr_cr, x);
            let (r, g, b) = ycbcr_to_rgb(y_bottom[x], cb, cr);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
    }
}

fn h2v2_sample(near: &[u8], curr: &[u8], x: usize) -> u8 {
    debug_assert_eq!(near.len(), curr.len());
    let n = curr.len();
    if n == 0 {
        return 0;
    }
    let sample = (x / 2).min(n - 1);
    let colsum = |idx: usize| 3 * u32::from(curr[idx]) + u32::from(near[idx]);
    if n == 1 {
        return ((4 * colsum(0) + 8) >> 4) as u8;
    }

    let this = colsum(sample);
    match x {
        0 => ((this * 4 + 8) >> 4) as u8,
        _ if x == n * 2 - 1 => ((this * 4 + 7) >> 4) as u8,
        _ if x.is_multiple_of(2) => {
            let last = colsum(sample - 1);
            ((this * 3 + last + 8) >> 4) as u8
        }
        _ => {
            let next = colsum(sample + 1);
            ((this * 3 + next + 7) >> 4) as u8
        }
    }
}
