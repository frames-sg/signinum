// SPDX-License-Identifier: Apache-2.0

use alloc::vec::Vec;
use slidecodec_core::ScratchPool;

/// Caller-owned reusable scratch for `slidecodec-j2k`.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct J2kScratchPool {
    packed_bytes: Vec<u8>,
    row_u16: Vec<u16>,
}

impl J2kScratchPool {
    pub const fn new() -> Self {
        Self {
            packed_bytes: Vec::new(),
            row_u16: Vec::new(),
        }
    }

    pub(crate) fn packed_bytes(&mut self, len: usize) -> &mut [u8] {
        if self.packed_bytes.len() != len {
            self.packed_bytes.resize(len, 0);
        }
        &mut self.packed_bytes
    }

    pub(crate) fn packed_bytes_and_row_u16(
        &mut self,
        packed_len: usize,
        row_len: usize,
    ) -> (&mut [u8], &mut [u16]) {
        if self.packed_bytes.len() != packed_len {
            self.packed_bytes.resize(packed_len, 0);
        }
        if self.row_u16.len() != row_len {
            self.row_u16.resize(row_len, 0);
        }
        (&mut self.packed_bytes, &mut self.row_u16)
    }
}

impl ScratchPool for J2kScratchPool {
    fn bytes_allocated(&self) -> usize {
        self.packed_bytes.capacity() + self.row_u16.capacity() * core::mem::size_of::<u16>()
    }

    fn reset(&mut self) {
        self.packed_bytes.clear();
        self.row_u16.clear();
    }
}
