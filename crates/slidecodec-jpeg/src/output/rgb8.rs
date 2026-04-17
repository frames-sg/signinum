// SPDX-License-Identifier: Apache-2.0

//! `Rgb8Writer` — 3-byte-per-pixel RGB output.

use crate::color::ycbcr::ycbcr_to_rgb;
use crate::output::OutputWriter;

pub(crate) struct Rgb8Writer<'o> {
    out: &'o mut [u8],
    stride: usize,
    width: u32,
}

impl<'o> Rgb8Writer<'o> {
    pub(crate) fn new(out: &'o mut [u8], stride: usize, width: u32) -> Self {
        Self { out, stride, width }
    }
}

impl OutputWriter for Rgb8Writer<'_> {
    fn write_rgb_row(&mut self, y: u32, r_row: &[u8], g_row: &[u8], b_row: &[u8]) {
        let dst_start = (y as usize) * self.stride;
        let width = self.width as usize;
        let dst = &mut self.out[dst_start..dst_start + width * 3];
        for i in 0..width {
            dst[i * 3] = r_row[i];
            dst[i * 3 + 1] = g_row[i];
            dst[i * 3 + 2] = b_row[i];
        }
    }

    fn write_ycbcr_row(&mut self, y: u32, y_row: &[u8], cb_row: &[u8], cr_row: &[u8]) {
        let dst_start = (y as usize) * self.stride;
        let width = self.width as usize;
        let dst = &mut self.out[dst_start..dst_start + width * 3];
        for i in 0..width {
            let (r, g, b) = ycbcr_to_rgb(y_row[i], cb_row[i], cr_row[i]);
            dst[i * 3] = r;
            dst[i * 3 + 1] = g;
            dst[i * 3 + 2] = b;
        }
    }

    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) {
        let dst_start = (y as usize) * self.stride;
        let width = self.width as usize;
        let dst = &mut self.out[dst_start..dst_start + width * 3];
        for i in 0..width {
            dst[i * 3] = gray_row[i];
            dst[i * 3 + 1] = gray_row[i];
            dst[i * 3 + 2] = gray_row[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn writes_single_row_at_stride_offset_zero() {
        let mut buf = vec![0u8; 3 * 3];
        let mut w = Rgb8Writer::new(&mut buf, 9, 3);
        w.write_ycbcr_row(0, &[128, 128, 128], &[128, 128, 128], &[128, 128, 128]);
        assert_eq!(buf, vec![128, 128, 128, 128, 128, 128, 128, 128, 128]);
    }

    #[test]
    fn writes_row_at_nonzero_stride_offset() {
        let mut buf = vec![0u8; 16 * 16];
        let mut w = Rgb8Writer::new(&mut buf, 16, 4);
        w.write_ycbcr_row(5, &[128, 128, 128, 128], &[128; 4], &[128; 4]);
        for i in 0..12 {
            assert_eq!(buf[80 + i], 128);
        }
        assert!(buf[..80].iter().all(|&b| b == 0));
    }

    #[test]
    fn grayscale_row_expands_to_identical_rgb_channels() {
        let mut buf = vec![0u8; 12];
        let mut w = Rgb8Writer::new(&mut buf, 12, 4);
        w.write_gray_row(0, &[50, 100, 150, 200]);
        assert_eq!(
            buf,
            vec![50, 50, 50, 100, 100, 100, 150, 150, 150, 200, 200, 200]
        );
    }

    #[test]
    fn ycbcr_row_applies_color_conversion() {
        let mut buf = vec![0u8; 3];
        let mut w = Rgb8Writer::new(&mut buf, 3, 1);
        w.write_ycbcr_row(0, &[76], &[85], &[255]);
        assert!(buf[0] > 240);
        assert!(buf[1] < 15);
        assert!(buf[2] < 15);
    }
}
