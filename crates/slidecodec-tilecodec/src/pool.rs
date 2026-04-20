// SPDX-License-Identifier: Apache-2.0

use slidecodec_core::ScratchPool;

#[derive(Debug, Default)]
pub struct DeflatePool {
    pub(crate) scratch: Vec<u8>,
}

#[derive(Debug, Default)]
pub struct ZstdPool {
    pub(crate) scratch: Vec<u8>,
}

#[derive(Debug, Default)]
pub struct LzwPool {
    pub(crate) scratch: Vec<u8>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoPool;

impl DeflatePool {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl ZstdPool {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl LzwPool {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl ScratchPool for DeflatePool {
    fn bytes_allocated(&self) -> usize {
        self.scratch.capacity()
    }

    fn reset(&mut self) {
        self.scratch.clear();
    }
}

impl ScratchPool for ZstdPool {
    fn bytes_allocated(&self) -> usize {
        self.scratch.capacity()
    }

    fn reset(&mut self) {
        self.scratch.clear();
    }
}

impl ScratchPool for LzwPool {
    fn bytes_allocated(&self) -> usize {
        self.scratch.capacity()
    }

    fn reset(&mut self) {
        self.scratch.clear();
    }
}

impl ScratchPool for NoPool {
    fn bytes_allocated(&self) -> usize {
        0
    }

    fn reset(&mut self) {}
}
