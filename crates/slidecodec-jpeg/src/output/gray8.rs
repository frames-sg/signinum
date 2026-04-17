// SPDX-License-Identifier: Apache-2.0

//! `Gray8Writer` — 1-byte-per-pixel grayscale output. Multi-component YCbCr
//! inputs project to luminance via `Y` (ignoring chroma).

use crate::output::OutputWriter;

pub(crate) struct Gray8Writer<'o> {
    out: &'o mut [u8],
    stride: usize,
    width: u32,
}

impl<'o> Gray8Writer<'o> {
    pub(crate) fn new(out: &'o mut [u8], stride: usize, width: u32) -> Self {
        Self { out, stride, width }
    }
}

impl OutputWriter for Gray8Writer<'_> {
    fn write_ycbcr_row(&mut self, y: u32, y_row: &[u8], _cb_row: &[u8], _cr_row: &[u8]) {
        let dst_start = (y as usize) * self.stride;
        let width = self.width as usize;
        self.out[dst_start..dst_start + width].copy_from_slice(&y_row[..width]);
    }

    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) {
        let dst_start = (y as usize) * self.stride;
        let width = self.width as usize;
        self.out[dst_start..dst_start + width].copy_from_slice(&gray_row[..width]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn copies_gray_row_verbatim() {
        let mut buf = vec![0u8; 4];
        let mut w = Gray8Writer::new(&mut buf, 4, 4);
        w.write_gray_row(0, &[10, 20, 30, 40]);
        assert_eq!(buf, vec![10, 20, 30, 40]);
    }

    #[test]
    fn ycbcr_row_projects_to_y_channel_only() {
        let mut buf = vec![0u8; 4];
        let mut w = Gray8Writer::new(&mut buf, 4, 4);
        w.write_ycbcr_row(0, &[10, 20, 30, 40], &[250; 4], &[5; 4]);
        assert_eq!(buf, vec![10, 20, 30, 40]);
    }

    #[test]
    fn respects_stride_across_rows() {
        let mut buf = vec![0u8; 16];
        let mut w = Gray8Writer::new(&mut buf, 8, 2);
        w.write_gray_row(0, &[1, 2]);
        w.write_gray_row(1, &[3, 4]);
        assert_eq!(buf[0..2], [1, 2]);
        assert_eq!(buf[8..10], [3, 4]);
    }
}
