// SPDX-License-Identifier: Apache-2.0

//! `Gray8Writer` — implemented in Task 13.

#![allow(dead_code)]

use crate::output::OutputWriter;

pub(crate) struct Gray8Writer<'o> {
    _out: &'o mut [u8],
    _stride: usize,
    _width: u32,
}

impl<'o> Gray8Writer<'o> {
    pub(crate) fn new(_out: &'o mut [u8], _stride: usize, _width: u32) -> Self {
        unimplemented!("Task 13")
    }
}

impl OutputWriter for Gray8Writer<'_> {
    fn write_ycbcr_row(&mut self, _y: u32, _y_row: &[u8], _cb_row: &[u8], _cr_row: &[u8]) {
        unimplemented!("Task 13")
    }
    fn write_gray_row(&mut self, _y: u32, _gray_row: &[u8]) {
        unimplemented!("Task 13")
    }
}
