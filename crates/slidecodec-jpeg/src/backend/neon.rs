// SPDX-License-Identifier: Apache-2.0

use core::arch::aarch64::{
    int32x4_t, uint16x8_t, uint8x16_t, uint8x8_t, uint8x8x3_t, vaddq_s32, vaddq_u16, vcombine_u16,
    vcombine_u8, vdupq_n_s32, vdupq_n_u16, vget_high_u16, vget_high_u8, vget_low_u16, vget_low_u8,
    vld1_u8, vmovl_u16, vmovl_u8, vmulq_n_s32, vqmovn_u16, vqmovun_s32, vreinterpretq_s32_u32,
    vshrq_n_s32, vshrq_n_u16, vst1q_u8, vst3_u8, vsubq_s32, vzip_u8, vzipq_u16,
};

use super::scalar;

pub(crate) fn fill_rgb_row_from_gray(gray_row: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(dst.len(), gray_row.len() * 3);
    unsafe {
        fill_rgb_row_from_gray_neon(gray_row, dst);
    }
}

#[target_feature(enable = "neon")]
unsafe fn fill_rgb_row_from_gray_neon(gray_row: &[u8], dst: &mut [u8]) {
    let width = gray_row.len();
    let mut offset = 0;
    while offset + LANES <= width {
        let g = unsafe { vld1_u8(gray_row.as_ptr().add(offset)) };
        unsafe {
            vst3_u8(dst.as_mut_ptr().add(offset * 3), uint8x8x3_t(g, g, g));
        }
        offset += LANES;
    }
    if offset < width {
        scalar::fill_rgb_row_from_gray(&gray_row[offset..], &mut dst[offset * 3..]);
    }
}

pub(crate) fn fill_rgb_row_from_rgb(r_row: &[u8], g_row: &[u8], b_row: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(r_row.len(), g_row.len());
    debug_assert_eq!(r_row.len(), b_row.len());
    debug_assert_eq!(dst.len(), r_row.len() * 3);
    unsafe {
        fill_rgb_row_from_rgb_neon(r_row, g_row, b_row, dst);
    }
}

#[target_feature(enable = "neon")]
unsafe fn fill_rgb_row_from_rgb_neon(r_row: &[u8], g_row: &[u8], b_row: &[u8], dst: &mut [u8]) {
    let width = r_row.len();
    let mut offset = 0;
    while offset + LANES <= width {
        let r = unsafe { vld1_u8(r_row.as_ptr().add(offset)) };
        let g = unsafe { vld1_u8(g_row.as_ptr().add(offset)) };
        let b = unsafe { vld1_u8(b_row.as_ptr().add(offset)) };
        unsafe {
            vst3_u8(dst.as_mut_ptr().add(offset * 3), uint8x8x3_t(r, g, b));
        }
        offset += LANES;
    }
    if offset < width {
        scalar::fill_rgb_row_from_rgb(
            &r_row[offset..],
            &g_row[offset..],
            &b_row[offset..],
            &mut dst[offset * 3..],
        );
    }
}

const FIX_1_40200: i32 = 91_881;
const FIX_0_34414: i32 = 22_554;
const FIX_0_71414: i32 = 46_802;
const FIX_1_77200: i32 = 116_130;
const ROUND: i32 = 1 << 15;
const LANES: usize = 8;
const UPSAMPLED_LANES: usize = LANES * 2;

pub(crate) fn fill_rgb_row_from_ycbcr(y_row: &[u8], cb_row: &[u8], cr_row: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(y_row.len(), cb_row.len());
    debug_assert_eq!(y_row.len(), cr_row.len());
    debug_assert_eq!(dst.len(), y_row.len() * 3);
    unsafe {
        fill_rgb_row_from_ycbcr_neon(y_row, cb_row, cr_row, dst);
    }
}

#[cfg(test)]
pub(super) fn fill_rgb_row_from_ycbcr_for_test(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    dst: &mut [u8],
) {
    fill_rgb_row_from_ycbcr(y_row, cb_row, cr_row, dst);
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
    debug_assert_eq!(dst_top.len(), y_top.len() * 3);
    debug_assert!(y_bottom.is_none_or(|row| row.len() == y_top.len()));
    debug_assert!(dst_bottom
        .as_ref()
        .is_none_or(|row| row.len() == y_top.len() * 3));
    debug_assert_eq!(prev_cb.len(), curr_cb.len());
    debug_assert_eq!(prev_cb.len(), next_cb.len());
    debug_assert_eq!(prev_cr.len(), curr_cr.len());
    debug_assert_eq!(prev_cr.len(), next_cr.len());
    unsafe {
        fill_rgb_row_pair_from_420_neon(
            y_top, y_bottom, prev_cb, curr_cb, next_cb, prev_cr, curr_cr, next_cr, dst_top,
            dst_bottom,
        );
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_rgb_row_pair_from_420_neon(
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
    let chroma_width = curr_cb.len();
    if let (Some(y_bottom), Some(dst_bottom)) = (y_bottom, dst_bottom) {
        unsafe {
            fill_rgb_row_pair_from_420_neon_dual(
                y_top, y_bottom, prev_cb, curr_cb, next_cb, prev_cr, curr_cr, next_cr, dst_top,
                dst_bottom,
            );
        }
    } else {
        unsafe {
            fill_rgb_row_pair_from_420_neon_top_only(
                y_top,
                prev_cb,
                curr_cb,
                prev_cr,
                curr_cr,
                dst_top,
                chroma_width,
                width,
            );
        }
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_rgb_row_pair_from_420_neon_dual(
    y_top: &[u8],
    y_bottom: &[u8],
    prev_cb: &[u8],
    curr_cb: &[u8],
    next_cb: &[u8],
    prev_cr: &[u8],
    curr_cr: &[u8],
    next_cr: &[u8],
    dst_top: &mut [u8],
    dst_bottom: &mut [u8],
) {
    let width = y_top.len();
    let chroma_width = curr_cb.len();
    let mut sample = 0usize;

    while sample < chroma_width {
        let chunk_samples = (chroma_width - sample).min(LANES);
        let x = sample * 2;
        if x >= width {
            break;
        }
        let chunk_width = (width - x).min(chunk_samples * 2);

        if can_vectorize_420_chunk(chroma_width, sample, chunk_width) {
            unsafe {
                fill_rgb_row_pair_from_420_chunk16_interior_neon(
                    &y_top[x..x + UPSAMPLED_LANES],
                    &y_bottom[x..x + UPSAMPLED_LANES],
                    prev_cb,
                    curr_cb,
                    next_cb,
                    prev_cr,
                    curr_cr,
                    next_cr,
                    sample,
                    &mut dst_top[x * 3..(x + UPSAMPLED_LANES) * 3],
                    &mut dst_bottom[x * 3..(x + UPSAMPLED_LANES) * 3],
                );
            }
            sample += chunk_samples;
            continue;
        }

        let mut cb_top = [0u8; UPSAMPLED_LANES];
        let mut cb_bot = [0u8; UPSAMPLED_LANES];
        let mut cr_top = [0u8; UPSAMPLED_LANES];
        let mut cr_bot = [0u8; UPSAMPLED_LANES];

        unsafe {
            fill_upsampled_420_chunk(prev_cb, curr_cb, sample, width, &mut cb_top[..chunk_width]);
            fill_upsampled_420_chunk(next_cb, curr_cb, sample, width, &mut cb_bot[..chunk_width]);
            fill_upsampled_420_chunk(prev_cr, curr_cr, sample, width, &mut cr_top[..chunk_width]);
            fill_upsampled_420_chunk(next_cr, curr_cr, sample, width, &mut cr_bot[..chunk_width]);
            fill_rgb_row_from_ycbcr_neon(
                &y_top[x..x + chunk_width],
                &cb_top[..chunk_width],
                &cr_top[..chunk_width],
                &mut dst_top[x * 3..(x + chunk_width) * 3],
            );
            fill_rgb_row_from_ycbcr_neon(
                &y_bottom[x..x + chunk_width],
                &cb_bot[..chunk_width],
                &cr_bot[..chunk_width],
                &mut dst_bottom[x * 3..(x + chunk_width) * 3],
            );
        }

        sample += chunk_samples;
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_rgb_row_pair_from_420_neon_top_only(
    y_top: &[u8],
    prev_cb: &[u8],
    curr_cb: &[u8],
    prev_cr: &[u8],
    curr_cr: &[u8],
    dst_top: &mut [u8],
    chroma_width: usize,
    width: usize,
) {
    let mut sample = 0usize;

    while sample < chroma_width {
        let chunk_samples = (chroma_width - sample).min(LANES);
        let x = sample * 2;
        if x >= width {
            break;
        }
        let chunk_width = (width - x).min(chunk_samples * 2);

        if can_vectorize_420_chunk(chroma_width, sample, chunk_width) {
            unsafe {
                fill_rgb_row_from_420_chunk16_interior_neon(
                    &y_top[x..x + UPSAMPLED_LANES],
                    prev_cb,
                    curr_cb,
                    prev_cr,
                    curr_cr,
                    sample,
                    &mut dst_top[x * 3..(x + UPSAMPLED_LANES) * 3],
                );
            }
            sample += chunk_samples;
            continue;
        }

        let mut cb_top = [0u8; UPSAMPLED_LANES];
        let mut cr_top = [0u8; UPSAMPLED_LANES];
        unsafe {
            fill_upsampled_420_chunk(prev_cb, curr_cb, sample, width, &mut cb_top[..chunk_width]);
            fill_upsampled_420_chunk(prev_cr, curr_cr, sample, width, &mut cr_top[..chunk_width]);
            fill_rgb_row_from_ycbcr_neon(
                &y_top[x..x + chunk_width],
                &cb_top[..chunk_width],
                &cr_top[..chunk_width],
                &mut dst_top[x * 3..(x + chunk_width) * 3],
            );
        }

        sample += chunk_samples;
    }
}

#[target_feature(enable = "neon")]
unsafe fn fill_rgb_row_from_420_chunk16_interior_neon(
    y_row: &[u8],
    near_cb: &[u8],
    curr_cb: &[u8],
    near_cr: &[u8],
    curr_cr: &[u8],
    sample_offset: usize,
    dst: &mut [u8],
) {
    debug_assert_eq!(y_row.len(), UPSAMPLED_LANES);
    debug_assert_eq!(dst.len(), UPSAMPLED_LANES * 3);

    let cb = unsafe { upsampled_420_chunk16_u16(near_cb, curr_cb, sample_offset) };
    let cr = unsafe { upsampled_420_chunk16_u16(near_cr, curr_cr, sample_offset) };
    let y_lo = unsafe { load_eight(y_row, 0) };
    let y_hi = unsafe { load_eight(y_row, LANES) };
    unsafe {
        fill_chunk_from_vectors_u16(y_lo, cb.0, cr.0, &mut dst[..LANES * 3]);
        fill_chunk_from_vectors_u16(y_hi, cb.1, cr.1, &mut dst[LANES * 3..]);
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_rgb_row_pair_from_420_chunk16_interior_neon(
    y_top: &[u8],
    y_bottom: &[u8],
    prev_cb: &[u8],
    curr_cb: &[u8],
    next_cb: &[u8],
    prev_cr: &[u8],
    curr_cr: &[u8],
    next_cr: &[u8],
    sample_offset: usize,
    dst_top: &mut [u8],
    dst_bottom: &mut [u8],
) {
    debug_assert_eq!(y_top.len(), UPSAMPLED_LANES);
    debug_assert_eq!(y_bottom.len(), UPSAMPLED_LANES);
    debug_assert_eq!(dst_top.len(), UPSAMPLED_LANES * 3);
    debug_assert_eq!(dst_bottom.len(), UPSAMPLED_LANES * 3);

    let (cb_top, cb_bottom) =
        unsafe { upsampled_420_chunk16_pair_u16(prev_cb, curr_cb, next_cb, sample_offset) };
    let (cr_top, cr_bottom) =
        unsafe { upsampled_420_chunk16_pair_u16(prev_cr, curr_cr, next_cr, sample_offset) };
    let y_top_lo = unsafe { load_eight(y_top, 0) };
    let y_top_hi = unsafe { load_eight(y_top, LANES) };
    let y_bottom_lo = unsafe { load_eight(y_bottom, 0) };
    let y_bottom_hi = unsafe { load_eight(y_bottom, LANES) };

    unsafe {
        fill_chunk_from_vectors_u16(y_top_lo, cb_top.0, cr_top.0, &mut dst_top[..LANES * 3]);
        fill_chunk_from_vectors_u16(y_top_hi, cb_top.1, cr_top.1, &mut dst_top[LANES * 3..]);
        fill_chunk_from_vectors_u16(
            y_bottom_lo,
            cb_bottom.0,
            cr_bottom.0,
            &mut dst_bottom[..LANES * 3],
        );
        fill_chunk_from_vectors_u16(
            y_bottom_hi,
            cb_bottom.1,
            cr_bottom.1,
            &mut dst_bottom[LANES * 3..],
        );
    }
}

#[target_feature(enable = "neon")]
unsafe fn fill_rgb_row_from_ycbcr_neon(y_row: &[u8], cb_row: &[u8], cr_row: &[u8], dst: &mut [u8]) {
    let width = y_row.len();
    let mut offset = 0;

    while offset + UPSAMPLED_LANES <= width {
        unsafe {
            fill_rgb_row_from_ycbcr_chunk16_neon(
                &y_row[offset..offset + UPSAMPLED_LANES],
                vcombine_u8(
                    load_eight(cb_row, offset),
                    load_eight(cb_row, offset + LANES),
                ),
                vcombine_u8(
                    load_eight(cr_row, offset),
                    load_eight(cr_row, offset + LANES),
                ),
                &mut dst[offset * 3..(offset + UPSAMPLED_LANES) * 3],
            );
        }
        offset += UPSAMPLED_LANES;
    }

    while offset + LANES <= width {
        unsafe {
            fill_chunk(
                y_row,
                cb_row,
                cr_row,
                &mut dst[offset * 3..(offset + LANES) * 3],
                offset,
            );
        }
        offset += LANES;
    }

    if offset < width {
        scalar::fill_rgb_row_from_ycbcr(
            &y_row[offset..],
            &cb_row[offset..],
            &cr_row[offset..],
            &mut dst[offset * 3..],
        );
    }
}

#[target_feature(enable = "neon")]
unsafe fn fill_chunk(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    dst_chunk: &mut [u8],
    offset: usize,
) {
    debug_assert_eq!(dst_chunk.len(), LANES * 3);

    let y = unsafe { load_eight(y_row, offset) };
    let cb = unsafe { load_eight(cb_row, offset) };
    let cr = unsafe { load_eight(cr_row, offset) };
    unsafe { fill_chunk_from_vectors(y, cb, cr, dst_chunk) };
}

#[target_feature(enable = "neon")]
unsafe fn fill_chunk_from_vectors(
    y: uint8x8_t,
    cb: uint8x8_t,
    cr: uint8x8_t,
    dst_chunk: &mut [u8],
) {
    debug_assert_eq!(dst_chunk.len(), LANES * 3);
    let y16 = vmovl_u8(y);
    let cb16 = vmovl_u8(cb);
    let cr16 = vmovl_u8(cr);

    let y_lo = widen_low(y16);
    let y_hi = widen_high(y16);
    let cb_lo = subtract_bias(widen_low(cb16));
    let cb_hi = subtract_bias(widen_high(cb16));
    let cr_lo = subtract_bias(widen_low(cr16));
    let cr_hi = subtract_bias(widen_high(cr16));

    let (r_lo, g_lo, b_lo) = convert_half(y_lo, cb_lo, cr_lo);
    let (r_hi, g_hi, b_hi) = convert_half(y_hi, cb_hi, cr_hi);

    let r_bytes = pack_eight_u8(r_lo, r_hi);
    let g_bytes = pack_eight_u8(g_lo, g_hi);
    let b_bytes = pack_eight_u8(b_lo, b_hi);

    unsafe {
        vst3_u8(
            dst_chunk.as_mut_ptr(),
            uint8x8x3_t(r_bytes, g_bytes, b_bytes),
        );
    }
}

#[target_feature(enable = "neon")]
unsafe fn fill_chunk_from_vectors_u16(
    y: uint8x8_t,
    cb: uint16x8_t,
    cr: uint16x8_t,
    dst_chunk: &mut [u8],
) {
    debug_assert_eq!(dst_chunk.len(), LANES * 3);
    let y16 = vmovl_u8(y);

    let y_lo = widen_low(y16);
    let y_hi = widen_high(y16);
    let cb_lo = subtract_bias(widen_low(cb));
    let cb_hi = subtract_bias(widen_high(cb));
    let cr_lo = subtract_bias(widen_low(cr));
    let cr_hi = subtract_bias(widen_high(cr));

    let (r_lo, g_lo, b_lo) = convert_half(y_lo, cb_lo, cr_lo);
    let (r_hi, g_hi, b_hi) = convert_half(y_hi, cb_hi, cr_hi);

    let r_bytes = pack_eight_u8(r_lo, r_hi);
    let g_bytes = pack_eight_u8(g_lo, g_hi);
    let b_bytes = pack_eight_u8(b_lo, b_hi);

    unsafe {
        vst3_u8(
            dst_chunk.as_mut_ptr(),
            uint8x8x3_t(r_bytes, g_bytes, b_bytes),
        );
    }
}

#[target_feature(enable = "neon")]
unsafe fn fill_rgb_row_from_ycbcr_chunk16_neon(
    y_row: &[u8],
    cb: uint8x16_t,
    cr: uint8x16_t,
    dst: &mut [u8],
) {
    debug_assert_eq!(y_row.len(), UPSAMPLED_LANES);
    debug_assert_eq!(dst.len(), UPSAMPLED_LANES * 3);

    let y_lo = unsafe { load_eight(y_row, 0) };
    let y_hi = unsafe { load_eight(y_row, LANES) };
    unsafe {
        fill_chunk_from_vectors(
            y_lo,
            vget_low_u8(cb),
            vget_low_u8(cr),
            &mut dst[..LANES * 3],
        );
        fill_chunk_from_vectors(
            y_hi,
            vget_high_u8(cb),
            vget_high_u8(cr),
            &mut dst[LANES * 3..],
        );
    }
}

#[target_feature(enable = "neon")]
unsafe fn load_eight(src: &[u8], offset: usize) -> uint8x8_t {
    unsafe { vld1_u8(src.as_ptr().add(offset)) }
}

#[target_feature(enable = "neon")]
fn widen_low(values: core::arch::aarch64::uint16x8_t) -> int32x4_t {
    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(values)))
}

#[target_feature(enable = "neon")]
fn widen_high(values: core::arch::aarch64::uint16x8_t) -> int32x4_t {
    vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(values)))
}

#[target_feature(enable = "neon")]
fn subtract_bias(values: int32x4_t) -> int32x4_t {
    vsubq_s32(values, vdupq_n_s32(128))
}

#[target_feature(enable = "neon")]
fn fixed_mul_shift(values: int32x4_t, coefficient: i32) -> int32x4_t {
    vshrq_n_s32(
        vaddq_s32(vmulq_n_s32(values, coefficient), vdupq_n_s32(ROUND)),
        16,
    )
}

#[target_feature(enable = "neon")]
fn convert_half(y: int32x4_t, cb: int32x4_t, cr: int32x4_t) -> (int32x4_t, int32x4_t, int32x4_t) {
    let r = vaddq_s32(y, fixed_mul_shift(cr, FIX_1_40200));
    let g = vsubq_s32(
        y,
        vshrq_n_s32(
            vaddq_s32(
                vaddq_s32(vmulq_n_s32(cb, FIX_0_34414), vmulq_n_s32(cr, FIX_0_71414)),
                vdupq_n_s32(ROUND),
            ),
            16,
        ),
    );
    let b = vaddq_s32(y, fixed_mul_shift(cb, FIX_1_77200));
    (r, g, b)
}

#[target_feature(enable = "neon")]
fn pack_eight_u8(low: int32x4_t, high: int32x4_t) -> uint8x8_t {
    let words = vcombine_u16(vqmovun_s32(low), vqmovun_s32(high));
    vqmovn_u16(words)
}

#[target_feature(enable = "neon")]
unsafe fn fill_upsampled_420_chunk(
    near: &[u8],
    curr: &[u8],
    sample_offset: usize,
    output_width: usize,
    out: &mut [u8],
) {
    if can_vectorize_420_chunk(curr.len(), sample_offset, out.len()) {
        unsafe {
            fill_upsampled_420_chunk_neon(near, curr, sample_offset, out);
        }
        return;
    }
    fill_upsampled_420_chunk_scalar(near, curr, sample_offset, output_width, out);
}

fn fill_upsampled_420_chunk_scalar(
    near: &[u8],
    curr: &[u8],
    sample_offset: usize,
    output_width: usize,
    out: &mut [u8],
) {
    debug_assert_eq!(near.len(), curr.len());
    let n = curr.len();
    if out.is_empty() || n == 0 {
        return;
    }

    let colsum = |i: usize| 3 * u32::from(curr[i]) + u32::from(near[i]);
    for sample in 0..out.len().div_ceil(2) {
        let global_sample = sample_offset + sample;
        let this = colsum(global_sample);
        let x = global_sample * 2;
        let local_x = sample * 2;
        out[local_x] = if x == 0 {
            ((this * 4 + 8) >> 4) as u8
        } else {
            let last = colsum(global_sample - 1);
            ((this * 3 + last + 8) >> 4) as u8
        };
        if local_x + 1 >= out.len() {
            break;
        }
        out[local_x + 1] = if x + 1 == output_width - 1 {
            ((this * 4 + 7) >> 4) as u8
        } else {
            let next = colsum((global_sample + 1).min(n - 1));
            ((this * 3 + next + 7) >> 4) as u8
        };
    }
}

fn can_vectorize_420_chunk(chroma_width: usize, sample_offset: usize, out_len: usize) -> bool {
    out_len == UPSAMPLED_LANES && sample_offset > 0 && sample_offset + LANES < chroma_width
}

#[target_feature(enable = "neon")]
unsafe fn fill_upsampled_420_chunk_neon(
    near: &[u8],
    curr: &[u8],
    sample_offset: usize,
    out: &mut [u8],
) {
    debug_assert!(can_vectorize_420_chunk(
        curr.len(),
        sample_offset,
        out.len()
    ));
    unsafe {
        vst1q_u8(
            out.as_mut_ptr(),
            upsampled_420_chunk16(near, curr, sample_offset),
        );
    }
}

#[target_feature(enable = "neon")]
unsafe fn upsampled_420_chunk16(near: &[u8], curr: &[u8], sample_offset: usize) -> uint8x16_t {
    let lanes = unsafe { upsampled_420_chunk16_u16(near, curr, sample_offset) };
    let even8 = vqmovn_u16(lanes.0);
    let odd8 = vqmovn_u16(lanes.1);
    let zipped = vzip_u8(even8, odd8);
    vcombine_u8(zipped.0, zipped.1)
}

#[target_feature(enable = "neon")]
unsafe fn upsampled_420_chunk16_u16(
    near: &[u8],
    curr: &[u8],
    sample_offset: usize,
) -> core::arch::aarch64::uint16x8x2_t {
    let this = unsafe { colsum_eight(near, curr, sample_offset) };
    let prev = unsafe { colsum_eight(near, curr, sample_offset - 1) };
    let next = unsafe { colsum_eight(near, curr, sample_offset + 1) };
    let three_this = vaddq_u16(this, vaddq_u16(this, this));

    let even = vshrq_n_u16(vaddq_u16(vaddq_u16(three_this, prev), vdupq_n_u16(8)), 4);
    let odd = vshrq_n_u16(vaddq_u16(vaddq_u16(three_this, next), vdupq_n_u16(7)), 4);
    vzipq_u16(even, odd)
}

#[target_feature(enable = "neon")]
unsafe fn upsampled_420_chunk16_pair_u16(
    top_near: &[u8],
    curr: &[u8],
    bottom_near: &[u8],
    sample_offset: usize,
) -> (
    core::arch::aarch64::uint16x8x2_t,
    core::arch::aarch64::uint16x8x2_t,
) {
    let curr_prev = vmovl_u8(unsafe { vld1_u8(curr.as_ptr().add(sample_offset - 1)) });
    let curr_this = vmovl_u8(unsafe { vld1_u8(curr.as_ptr().add(sample_offset)) });
    let curr_next = vmovl_u8(unsafe { vld1_u8(curr.as_ptr().add(sample_offset + 1)) });

    let three_prev = vaddq_u16(curr_prev, vaddq_u16(curr_prev, curr_prev));
    let three_this = vaddq_u16(curr_this, vaddq_u16(curr_this, curr_this));
    let three_next = vaddq_u16(curr_next, vaddq_u16(curr_next, curr_next));

    let top_prev = vaddq_u16(
        three_prev,
        vmovl_u8(unsafe { vld1_u8(top_near.as_ptr().add(sample_offset - 1)) }),
    );
    let top_this = vaddq_u16(
        three_this,
        vmovl_u8(unsafe { vld1_u8(top_near.as_ptr().add(sample_offset)) }),
    );
    let top_next = vaddq_u16(
        three_next,
        vmovl_u8(unsafe { vld1_u8(top_near.as_ptr().add(sample_offset + 1)) }),
    );

    let bottom_prev = vaddq_u16(
        three_prev,
        vmovl_u8(unsafe { vld1_u8(bottom_near.as_ptr().add(sample_offset - 1)) }),
    );
    let bottom_this = vaddq_u16(
        three_this,
        vmovl_u8(unsafe { vld1_u8(bottom_near.as_ptr().add(sample_offset)) }),
    );
    let bottom_next = vaddq_u16(
        three_next,
        vmovl_u8(unsafe { vld1_u8(bottom_near.as_ptr().add(sample_offset + 1)) }),
    );

    let top_three_this = vaddq_u16(top_this, vaddq_u16(top_this, top_this));
    let top_even = vshrq_n_u16(
        vaddq_u16(vaddq_u16(top_three_this, top_prev), vdupq_n_u16(8)),
        4,
    );
    let top_odd = vshrq_n_u16(
        vaddq_u16(vaddq_u16(top_three_this, top_next), vdupq_n_u16(7)),
        4,
    );

    let bottom_three_this = vaddq_u16(bottom_this, vaddq_u16(bottom_this, bottom_this));
    let bottom_even = vshrq_n_u16(
        vaddq_u16(vaddq_u16(bottom_three_this, bottom_prev), vdupq_n_u16(8)),
        4,
    );
    let bottom_odd = vshrq_n_u16(
        vaddq_u16(vaddq_u16(bottom_three_this, bottom_next), vdupq_n_u16(7)),
        4,
    );

    (
        vzipq_u16(top_even, top_odd),
        vzipq_u16(bottom_even, bottom_odd),
    )
}

#[target_feature(enable = "neon")]
unsafe fn colsum_eight(near: &[u8], curr: &[u8], sample_offset: usize) -> uint16x8_t {
    let near16 = vmovl_u8(unsafe { vld1_u8(near.as_ptr().add(sample_offset)) });
    let curr16 = vmovl_u8(unsafe { vld1_u8(curr.as_ptr().add(sample_offset)) });
    vaddq_u16(vaddq_u16(curr16, curr16), vaddq_u16(curr16, near16))
}
