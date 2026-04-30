// SPDX-License-Identifier: Apache-2.0

//! `Rgb8Writer` — 3-byte-per-pixel RGB output.

use crate::color::ycbcr::ycbcr_to_rgb;
use crate::error::JpegError;
use crate::output::{InterleavedRgbWriter, OutputWriter};

pub(crate) struct Rgb8Writer<'o> {
    out: &'o mut [u8],
    stride: usize,
    width: u32,
}

impl<'o> Rgb8Writer<'o> {
    pub(crate) fn new(out: &'o mut [u8], stride: usize, width: u32) -> Self {
        Self { out, stride, width }
    }

    fn row_mut(&mut self, y: u32) -> &mut [u8] {
        let dst_start = (y as usize) * self.stride;
        let width = self.width as usize;
        &mut self.out[dst_start..dst_start + width * 3]
    }
}

impl InterleavedRgbWriter for Rgb8Writer<'_> {
    fn with_rgb_rows<R, F>(&mut self, y: u32, row_count: usize, fill: F) -> Result<R, JpegError>
    where
        F: FnOnce(&mut [u8], Option<&mut [u8]>) -> Result<R, JpegError>,
    {
        let top_len = self.width as usize * 3;
        match row_count {
            1 => {
                let top = self.row_mut(y);
                debug_assert_eq!(top.len(), top_len);
                fill(top, None)
            }
            2 => {
                let top_start = (y as usize) * self.stride;
                let bottom_start = ((y + 1) as usize) * self.stride;
                let (head, tail) = self.out.split_at_mut(bottom_start);
                let top = &mut head[top_start..top_start + top_len];
                let bottom = &mut tail[..top_len];
                fill(top, Some(bottom))
            }
            _ => unreachable!("Rgb8Writer only supports one or two rows"),
        }
    }
}

impl OutputWriter for Rgb8Writer<'_> {
    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), JpegError> {
        let dst_start = (y as usize) * self.stride;
        let width = self.width as usize;
        let dst = &mut self.out[dst_start..dst_start + width * 3];
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
        Ok(())
    }

    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
    ) -> Result<(), JpegError> {
        let dst_start = (y as usize) * self.stride;
        let width = self.width as usize;
        let dst = &mut self.out[dst_start..dst_start + width * 3];
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
        Ok(())
    }

    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), JpegError> {
        let dst_start = (y as usize) * self.stride;
        let width = self.width as usize;
        let dst = &mut self.out[dst_start..dst_start + width * 3];
        for (&gray, pixel) in gray_row.iter().zip(dst.chunks_exact_mut(3)) {
            pixel[0] = gray;
            pixel[1] = gray;
            pixel[2] = gray;
        }
        Ok(())
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
        w.write_ycbcr_row(0, &[128, 128, 128], &[128, 128, 128], &[128, 128, 128])
            .unwrap();
        assert_eq!(buf, vec![128, 128, 128, 128, 128, 128, 128, 128, 128]);
    }

    #[test]
    fn writes_row_at_nonzero_stride_offset() {
        let mut buf = vec![0u8; 16 * 16];
        let mut w = Rgb8Writer::new(&mut buf, 16, 4);
        w.write_ycbcr_row(5, &[128, 128, 128, 128], &[128; 4], &[128; 4])
            .unwrap();
        for i in 0..12 {
            assert_eq!(buf[80 + i], 128);
        }
        assert!(buf[..80].iter().all(|&b| b == 0));
    }

    #[test]
    fn grayscale_row_expands_to_identical_rgb_channels() {
        let mut buf = vec![0u8; 12];
        let mut w = Rgb8Writer::new(&mut buf, 12, 4);
        w.write_gray_row(0, &[50, 100, 150, 200]).unwrap();
        assert_eq!(
            buf,
            vec![50, 50, 50, 100, 100, 100, 150, 150, 150, 200, 200, 200]
        );
    }

    #[test]
    fn ycbcr_row_applies_color_conversion() {
        let mut buf = vec![0u8; 3];
        let mut w = Rgb8Writer::new(&mut buf, 3, 1);
        w.write_ycbcr_row(0, &[76], &[85], &[255]).unwrap();
        assert!(buf[0] > 240);
        assert!(buf[1] < 15);
        assert!(buf[2] < 15);
    }
}
