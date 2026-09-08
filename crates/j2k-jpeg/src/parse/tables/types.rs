// SPDX-License-Identifier: MIT OR Apache-2.0

//! Heap-free raw JPEG table payloads.

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HuffmanValues {
    bytes: [u8; 256],
    len: u16,
}

impl Default for HuffmanValues {
    fn default() -> Self {
        Self {
            bytes: [0; 256],
            len: 0,
        }
    }
}

impl HuffmanValues {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "JPEG Huffman value arrays contain at most 256 symbols"
    )]
    pub(crate) fn from_slice(values: &[u8]) -> Self {
        debug_assert!(values.len() <= 256);
        let mut out = Self::default();
        out.bytes[..values.len()].copy_from_slice(values);
        out.len = values.len() as u16;
        out
    }

    pub(crate) fn len(&self) -> usize {
        usize::from(self.len)
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.len()]
    }

    pub(crate) fn get(&self, index: usize) -> Option<u8> {
        self.as_slice().get(index).copied()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RawHuffmanTable {
    pub(crate) bits: [u8; 16],
    pub(crate) values: HuffmanValues,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HuffmanTableRole {
    Dc,
    Ac,
}
