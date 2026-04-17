// SPDX-License-Identifier: Apache-2.0

//! `Rgba8Writer` — implemented in Task 12.

#![allow(dead_code)]

use crate::output::OutputWriter;

pub(crate) struct Rgba8Writer<'o> {
    _out: &'o mut [u8],
    _stride: usize,
    _width: u32,
    _alpha: u8,
}

impl<'o> Rgba8Writer<'o> {
    pub(crate) fn new(_out: &'o mut [u8], _stride: usize, _width: u32, _alpha: u8) -> Self {
        unimplemented!("Task 12")
    }
}

impl OutputWriter for Rgba8Writer<'_> {
    fn write_ycbcr_row(&mut self, _y: u32, _y_row: &[u8], _cb_row: &[u8], _cr_row: &[u8]) {
        unimplemented!("Task 12")
    }
    fn write_gray_row(&mut self, _y: u32, _gray_row: &[u8]) {
        unimplemented!("Task 12")
    }
}
