// SPDX-License-Identifier: Apache-2.0

use alloc::vec::Vec;
use core::arch::x86_64::{
    __m128i, __m256i, _mm256_add_epi32, _mm256_cvtepu8_epi32, _mm256_extracti128_si256,
    _mm256_mullo_epi32, _mm256_set1_epi32, _mm256_srai_epi32, _mm256_sub_epi32, _mm_cvtsi128_si64,
    _mm_loadl_epi64, _mm_packs_epi32, _mm_packus_epi16,
};
use core::cell::RefCell;

use crate::color::upsample::{upsample_h2v2_fancy_row, upsample_h2v2_fancy_rows};

use super::scalar;

const FIX_1_40200: i32 = 91_881;
const FIX_0_34414: i32 = 22_554;
const FIX_0_71414: i32 = 46_802;
const FIX_1_77200: i32 = 116_130;
const ROUND: i32 = 1 << 15;
const LANES: usize = 8;
const RGB_UNROLL: usize = 8;

#[derive(Default)]
struct RowPairScratch {
    cb_top: Vec<u8>,
    cb_bottom: Vec<u8>,
    cr_top: Vec<u8>,
    cr_bottom: Vec<u8>,
}

impl RowPairScratch {
    fn ensure_width(&mut self, width: usize) {
        self.cb_top.resize(width, 0);
        self.cb_bottom.resize(width, 0);
        self.cr_top.resize(width, 0);
        self.cr_bottom.resize(width, 0);
    }
}

std::thread_local! {
    static ROW_PAIR_SCRATCH: RefCell<RowPairScratch> = RefCell::new(RowPairScratch::default());
}

pub(crate) fn fill_rgb_row_from_gray(gray_row: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(dst.len(), gray_row.len() * 3);
    let mut offset = 0;
    while offset + RGB_UNROLL <= gray_row.len() {
        let chunk = &gray_row[offset..offset + RGB_UNROLL];
        let dst_chunk = &mut dst[offset * 3..(offset + RGB_UNROLL) * 3];
        for (gray, pixel) in chunk.iter().zip(dst_chunk.chunks_exact_mut(3)) {
            pixel[0] = *gray;
            pixel[1] = *gray;
            pixel[2] = *gray;
        }
        offset += RGB_UNROLL;
    }
    if offset < gray_row.len() {
        scalar::fill_rgb_row_from_gray(&gray_row[offset..], &mut dst[offset * 3..]);
    }
}

pub(crate) fn fill_rgb_row_from_rgb(r_row: &[u8], g_row: &[u8], b_row: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(r_row.len(), g_row.len());
    debug_assert_eq!(r_row.len(), b_row.len());
    debug_assert_eq!(dst.len(), r_row.len() * 3);
    let mut offset = 0;
    while offset + RGB_UNROLL <= r_row.len() {
        let r_chunk = &r_row[offset..offset + RGB_UNROLL];
        let g_chunk = &g_row[offset..offset + RGB_UNROLL];
        let b_chunk = &b_row[offset..offset + RGB_UNROLL];
        let dst_chunk = &mut dst[offset * 3..(offset + RGB_UNROLL) * 3];
        for (((&r, &g), &b), pixel) in r_chunk
            .iter()
            .zip(g_chunk.iter())
            .zip(b_chunk.iter())
            .zip(dst_chunk.chunks_exact_mut(3))
        {
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
        offset += RGB_UNROLL;
    }
    if offset < r_row.len() {
        scalar::fill_rgb_row_from_rgb(
            &r_row[offset..],
            &g_row[offset..],
            &b_row[offset..],
            &mut dst[offset * 3..],
        );
    }
}

pub(crate) fn fill_rgb_row_from_ycbcr(y_row: &[u8], cb_row: &[u8], cr_row: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(y_row.len(), cb_row.len());
    debug_assert_eq!(y_row.len(), cr_row.len());
    debug_assert_eq!(dst.len(), y_row.len() * 3);
    unsafe {
        fill_rgb_row_from_ycbcr_avx2(y_row, cb_row, cr_row, dst);
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
    debug_assert_eq!(dst_top.len(), y_top.len() * 3);
    debug_assert!(y_bottom.is_none_or(|row| row.len() == y_top.len()));
    debug_assert!(dst_bottom
        .as_ref()
        .is_none_or(|row| row.len() == y_top.len() * 3));
    debug_assert_eq!(prev_cb.len(), curr_cb.len());
    debug_assert_eq!(prev_cb.len(), next_cb.len());
    debug_assert_eq!(prev_cr.len(), curr_cr.len());
    debug_assert_eq!(prev_cr.len(), next_cr.len());

    ROW_PAIR_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.ensure_width(y_top.len());
        unsafe {
            fill_rgb_row_pair_from_420_avx2(
                y_top,
                y_bottom,
                prev_cb,
                curr_cb,
                next_cb,
                prev_cr,
                curr_cr,
                next_cr,
                dst_top,
                dst_bottom,
                &mut scratch,
            );
        }
    });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_rgb_row_pair_from_420_cropped(
    y_top: &[u8],
    y_bottom: Option<&[u8]>,
    prev_cb: &[u8],
    curr_cb: &[u8],
    next_cb: &[u8],
    prev_cr: &[u8],
    curr_cr: &[u8],
    next_cr: &[u8],
    crop_start: usize,
    crop_width: usize,
    dst_top: &mut [u8],
    dst_bottom: Option<&mut [u8]>,
) {
    scalar::fill_rgb_row_pair_from_420_cropped(
        y_top, y_bottom, prev_cb, curr_cb, next_cb, prev_cr, curr_cr, next_cr, crop_start,
        crop_width, dst_top, dst_bottom,
    );
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_rgb_row_pair_from_420_avx2(
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
    scratch: &mut RowPairScratch,
) {
    let width = y_top.len();
    let cb_top = &mut scratch.cb_top[..width];
    let cr_top = &mut scratch.cr_top[..width];
    if let (Some(y_bottom), Some(dst_bottom)) = (y_bottom, dst_bottom) {
        let cb_bottom = &mut scratch.cb_bottom[..width];
        let cr_bottom = &mut scratch.cr_bottom[..width];
        upsample_h2v2_fancy_rows(prev_cb, curr_cb, next_cb, width, cb_top, cb_bottom);
        upsample_h2v2_fancy_rows(prev_cr, curr_cr, next_cr, width, cr_top, cr_bottom);
        unsafe {
            fill_rgb_row_from_ycbcr_avx2(y_top, cb_top, cr_top, dst_top);
            fill_rgb_row_from_ycbcr_avx2(y_bottom, cb_bottom, cr_bottom, dst_bottom);
        }
    } else {
        upsample_h2v2_fancy_row(prev_cb, curr_cb, next_cb, width, false, cb_top);
        upsample_h2v2_fancy_row(prev_cr, curr_cr, next_cr, width, false, cr_top);
        unsafe {
            fill_rgb_row_from_ycbcr_avx2(y_top, cb_top, cr_top, dst_top);
        }
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

#[cfg(test)]
pub(super) fn fill_rgb_row_from_gray_for_test(gray_row: &[u8], dst: &mut [u8]) {
    fill_rgb_row_from_gray(gray_row, dst);
}

#[cfg(test)]
pub(super) fn fill_rgb_row_from_rgb_for_test(
    r_row: &[u8],
    g_row: &[u8],
    b_row: &[u8],
    dst: &mut [u8],
) {
    fill_rgb_row_from_rgb(r_row, g_row, b_row, dst);
}

#[target_feature(enable = "avx2")]
unsafe fn fill_rgb_row_from_ycbcr_avx2(y_row: &[u8], cb_row: &[u8], cr_row: &[u8], dst: &mut [u8]) {
    let width = y_row.len();
    let mut offset = 0;

    while offset + (LANES * 2) <= width {
        unsafe {
            fill_chunk(
                y_row,
                cb_row,
                cr_row,
                &mut dst[offset * 3..(offset + LANES) * 3],
                offset,
            );
            fill_chunk(
                y_row,
                cb_row,
                cr_row,
                &mut dst[(offset + LANES) * 3..(offset + LANES * 2) * 3],
                offset + LANES,
            );
        }
        offset += LANES * 2;
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

#[target_feature(enable = "avx2")]
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

    let bias = _mm256_set1_epi32(128);
    let y32 = _mm256_cvtepu8_epi32(y);
    let cb32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(cb), bias);
    let cr32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(cr), bias);

    let r = _mm256_add_epi32(y32, fixed_mul_shift(cr32, FIX_1_40200));
    let g = _mm256_sub_epi32(
        y32,
        _mm256_srai_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(cb32, _mm256_set1_epi32(FIX_0_34414)),
                    _mm256_mullo_epi32(cr32, _mm256_set1_epi32(FIX_0_71414)),
                ),
                _mm256_set1_epi32(ROUND),
            ),
            16,
        ),
    );
    let b = _mm256_add_epi32(y32, fixed_mul_shift(cb32, FIX_1_77200));

    unsafe {
        store_rgb_chunk(dst_chunk, r, g, b);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn load_eight(src: &[u8], offset: usize) -> __m128i {
    unsafe { _mm_loadl_epi64(src.as_ptr().add(offset).cast()) }
}

#[target_feature(enable = "avx2")]
fn fixed_mul_shift(values: __m256i, coefficient: i32) -> __m256i {
    _mm256_srai_epi32(
        _mm256_add_epi32(
            _mm256_mullo_epi32(values, _mm256_set1_epi32(coefficient)),
            _mm256_set1_epi32(ROUND),
        ),
        16,
    )
}

#[target_feature(enable = "avx2")]
unsafe fn store_rgb_chunk(dst_chunk: &mut [u8], r: __m256i, g: __m256i, b: __m256i) {
    let r_bytes = unsafe { pack_eight_u8(r) };
    let g_bytes = unsafe { pack_eight_u8(g) };
    let b_bytes = unsafe { pack_eight_u8(b) };

    for ((((r, g), b), pixel), _) in r_bytes
        .iter()
        .zip(g_bytes.iter())
        .zip(b_bytes.iter())
        .zip(dst_chunk.chunks_exact_mut(3))
        .zip(0..LANES)
    {
        pixel[0] = *r;
        pixel[1] = *g;
        pixel[2] = *b;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn pack_eight_u8(values: __m256i) -> [u8; LANES] {
    let words = _mm_packs_epi32(
        _mm256_extracti128_si256(values, 0),
        _mm256_extracti128_si256(values, 1),
    );
    let bytes = _mm_packus_epi16(words, words);
    _mm_cvtsi128_si64(bytes).to_ne_bytes()
}
