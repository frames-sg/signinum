// SPDX-License-Identifier: Apache-2.0

//! Hot-path backend dispatch for interleaved RGB row production and the 8×8
//! inverse DCT.

use crate::idct;
use slidecodec_core::CpuFeatures;

mod scalar;

#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendKind {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Backend {
    kind: BackendKind,
}

impl Backend {
    pub(crate) fn detect() -> Self {
        let cpu = CpuFeatures::detect();

        #[cfg(target_arch = "x86_64")]
        {
            if !cfg!(feature = "scalar-only") && cpu.avx2 {
                return Self {
                    kind: BackendKind::Avx2,
                };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if !cfg!(feature = "scalar-only") && cpu.neon {
                return Self {
                    kind: BackendKind::Neon,
                };
            }
        }

        Self {
            kind: BackendKind::Scalar,
        }
    }

    pub(crate) fn fill_rgb_row_from_gray(self, gray_row: &[u8], dst: &mut [u8]) {
        match self.kind {
            BackendKind::Scalar => scalar::fill_rgb_row_from_gray(gray_row, dst),
            #[cfg(target_arch = "x86_64")]
            BackendKind::Avx2 => x86::fill_rgb_row_from_gray(gray_row, dst),
            #[cfg(target_arch = "aarch64")]
            BackendKind::Neon => neon::fill_rgb_row_from_gray(gray_row, dst),
        }
    }

    pub(crate) fn fill_rgb_row_from_rgb(
        self,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
        dst: &mut [u8],
    ) {
        match self.kind {
            BackendKind::Scalar => scalar::fill_rgb_row_from_rgb(r_row, g_row, b_row, dst),
            #[cfg(target_arch = "x86_64")]
            BackendKind::Avx2 => x86::fill_rgb_row_from_rgb(r_row, g_row, b_row, dst),
            #[cfg(target_arch = "aarch64")]
            BackendKind::Neon => neon::fill_rgb_row_from_rgb(r_row, g_row, b_row, dst),
        }
    }

    pub(crate) fn fill_rgb_row_from_ycbcr(
        self,
        y_row: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
        dst: &mut [u8],
    ) {
        match self.kind {
            BackendKind::Scalar => scalar::fill_rgb_row_from_ycbcr(y_row, cb_row, cr_row, dst),
            #[cfg(target_arch = "x86_64")]
            BackendKind::Avx2 => x86::fill_rgb_row_from_ycbcr(y_row, cb_row, cr_row, dst),
            #[cfg(target_arch = "aarch64")]
            BackendKind::Neon => neon::fill_rgb_row_from_ycbcr(y_row, cb_row, cr_row, dst),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fill_rgb_row_pair_from_420(
        self,
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
        match self.kind {
            BackendKind::Scalar => scalar::fill_rgb_row_pair_from_420(
                y_top, y_bottom, prev_cb, curr_cb, next_cb, prev_cr, curr_cr, next_cr, dst_top,
                dst_bottom,
            ),
            #[cfg(target_arch = "x86_64")]
            BackendKind::Avx2 => x86::fill_rgb_row_pair_from_420(
                y_top, y_bottom, prev_cb, curr_cb, next_cb, prev_cr, curr_cr, next_cr, dst_top,
                dst_bottom,
            ),
            #[cfg(target_arch = "aarch64")]
            BackendKind::Neon => neon::fill_rgb_row_pair_from_420(
                y_top, y_bottom, prev_cb, curr_cb, next_cb, prev_cr, curr_cr, next_cr, dst_top,
                dst_bottom,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fill_rgb_row_pair_from_420_cropped(
        self,
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
        match self.kind {
            BackendKind::Scalar => scalar::fill_rgb_row_pair_from_420_cropped(
                y_top, y_bottom, prev_cb, curr_cb, next_cb, prev_cr, curr_cr, next_cr, crop_start,
                crop_width, dst_top, dst_bottom,
            ),
            #[cfg(target_arch = "x86_64")]
            BackendKind::Avx2 => x86::fill_rgb_row_pair_from_420_cropped(
                y_top, y_bottom, prev_cb, curr_cb, next_cb, prev_cr, curr_cr, next_cr, crop_start,
                crop_width, dst_top, dst_bottom,
            ),
            #[cfg(target_arch = "aarch64")]
            BackendKind::Neon => neon::fill_rgb_row_pair_from_420_cropped(
                y_top, y_bottom, prev_cb, curr_cb, next_cb, prev_cr, curr_cr, next_cr, crop_start,
                crop_width, dst_top, dst_bottom,
            ),
        }
    }

    pub(crate) fn prefers_cropped_420_region(self, row_width: usize, crop_width: usize) -> bool {
        if crop_width == 0 || crop_width >= row_width {
            return false;
        }
        match self.kind {
            BackendKind::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            BackendKind::Avx2 => true,
            #[cfg(target_arch = "aarch64")]
            BackendKind::Neon => true,
        }
    }

    /// 8×8 inverse DCT of a dequantized coefficient block. Output is
    /// level-shifted by +128 and clamped to `[0, 255]` — bit-exact with
    /// [`idct::scalar::idct_islow`] on every legal JPEG input.
    pub(crate) fn idct(self, input: &[i16; 64], output: &mut [u8; 64]) {
        match self.kind {
            BackendKind::Scalar => idct::scalar::idct_islow(input, output),
            #[cfg(target_arch = "x86_64")]
            BackendKind::Avx2 => unsafe { idct::avx2::idct_islow(input, output) },
            #[cfg(target_arch = "aarch64")]
            BackendKind::Neon => unsafe { idct::neon::idct_islow(input, output) },
        }
    }

    pub(crate) fn idct_bottom_half_zero(self, input: &[i16; 64], output: &mut [u8; 64]) {
        match self.kind {
            BackendKind::Scalar => idct::scalar::idct_islow_bottom_half_zero(input, output),
            #[cfg(target_arch = "x86_64")]
            BackendKind::Avx2 => unsafe { idct::avx2::idct_islow(input, output) },
            #[cfg(target_arch = "aarch64")]
            BackendKind::Neon => unsafe { idct::neon::idct_islow_bottom_half_zero(input, output) },
        }
    }
}
