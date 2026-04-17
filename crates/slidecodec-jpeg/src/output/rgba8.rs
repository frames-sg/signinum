// SPDX-License-Identifier: Apache-2.0

//! `Rgba8Writer` — 4-byte-per-pixel RGBA output. Alpha is a constant captured
//! at construction time and written verbatim into every pixel's A channel.

#![allow(dead_code)]

use crate::color::ycbcr::ycbcr_to_rgb;
use crate::output::OutputWriter;

pub(crate) struct Rgba8Writer<'o> {
    out: &'o mut [u8],
    stride: usize,
    width: u32,
    alpha: u8,
}

impl<'o> Rgba8Writer<'o> {
    pub(crate) fn new(out: &'o mut [u8], stride: usize, width: u32, alpha: u8) -> Self {
        Self {
            out,
            stride,
            width,
            alpha,
        }
    }
}

impl OutputWriter for Rgba8Writer<'_> {
    fn write_ycbcr_row(&mut self, y: u32, y_row: &[u8], cb_row: &[u8], cr_row: &[u8]) {
        let dst_start = (y as usize) * self.stride;
        let width = self.width as usize;
        let dst = &mut self.out[dst_start..dst_start + width * 4];
        let alpha = self.alpha;
        for i in 0..width {
            let (r, g, b) = ycbcr_to_rgb(y_row[i], cb_row[i], cr_row[i]);
            dst[i * 4] = r;
            dst[i * 4 + 1] = g;
            dst[i * 4 + 2] = b;
            dst[i * 4 + 3] = alpha;
        }
    }

    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) {
        let dst_start = (y as usize) * self.stride;
        let width = self.width as usize;
        let dst = &mut self.out[dst_start..dst_start + width * 4];
        let alpha = self.alpha;
        for i in 0..width {
            dst[i * 4] = gray_row[i];
            dst[i * 4 + 1] = gray_row[i];
            dst[i * 4 + 2] = gray_row[i];
            dst[i * 4 + 3] = alpha;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn writes_alpha_byte_per_pixel() {
        let mut buf = vec![0u8; 2 * 4];
        let mut w = Rgba8Writer::new(&mut buf, 8, 2, 200);
        w.write_ycbcr_row(0, &[128, 128], &[128, 128], &[128, 128]);
        assert_eq!(buf[3], 200);
        assert_eq!(buf[7], 200);
    }

    #[test]
    fn grayscale_expands_with_alpha() {
        let mut buf = vec![0u8; 3 * 4];
        let mut w = Rgba8Writer::new(&mut buf, 12, 3, 255);
        w.write_gray_row(0, &[10, 20, 30]);
        assert_eq!(
            buf,
            vec![10, 10, 10, 255, 20, 20, 20, 255, 30, 30, 30, 255]
        );
    }

    #[test]
    fn ycbcr_color_conversion_honors_alpha() {
        let mut buf = vec![0u8; 4];
        let mut w = Rgba8Writer::new(&mut buf, 4, 1, 99);
        w.write_ycbcr_row(0, &[128], &[128], &[128]);
        assert_eq!(buf, vec![128, 128, 128, 99]);
    }
}
