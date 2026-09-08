// SPDX-License-Identifier: MIT OR Apache-2.0

//! Raw table-version arenas and compact active-slot snapshots.

use alloc::vec::Vec;
use core::num::NonZeroU32;

use crate::allocation::checked_allocation_bytes;
use crate::error::JpegError;
use crate::parse::allocation::ParsedMetadataBudget;

use super::{HuffmanTableRole, RawHuffmanTable};

const TABLE_SLOTS: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RawHuffmanTableId(NonZeroU32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RawQuantTableId(NonZeroU32);

impl RawHuffmanTableId {
    fn for_next_index(index: usize) -> Result<Self, JpegError> {
        Ok(Self(nonzero_id(index)?))
    }

    fn index(self) -> Option<usize> {
        usize::try_from(self.0.get())
            .ok()
            .and_then(|value| value.checked_sub(1))
    }
}

impl RawQuantTableId {
    fn for_next_index(index: usize) -> Result<Self, JpegError> {
        Ok(Self(nonzero_id(index)?))
    }

    fn index(self) -> Option<usize> {
        usize::try_from(self.0.get())
            .ok()
            .and_then(|value| value.checked_sub(1))
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProgressiveTableState {
    dc: [Option<RawHuffmanTableId>; TABLE_SLOTS],
    ac: [Option<RawHuffmanTableId>; TABLE_SLOTS],
    quant: [Option<RawQuantTableId>; TABLE_SLOTS],
}

#[derive(Debug, Default)]
pub(crate) struct QuantTables {
    pub(crate) entries: [Option<[u16; 64]>; TABLE_SLOTS],
    active: [Option<RawQuantTableId>; TABLE_SLOTS],
    pub(crate) versions: Vec<[u16; 64]>,
}

impl QuantTables {
    pub(super) fn define(
        &mut self,
        slot: usize,
        table: [u16; 64],
        budget: &mut ParsedMetadataBudget,
    ) -> Result<(), JpegError> {
        if slot >= TABLE_SLOTS {
            return Err(JpegError::InternalInvariant {
                reason: "unvalidated DQT slot reached table definition",
            });
        }
        let id = RawQuantTableId::for_next_index(self.versions.len())?;
        budget.try_push(&mut self.versions, table)?;
        self.entries[slot] = Some(table);
        self.active[slot] = Some(id);
        Ok(())
    }

    pub(crate) fn resolve(&self, state: &ProgressiveTableState, slot: u8) -> Option<&[u16; 64]> {
        state
            .quant
            .get(usize::from(slot))
            .copied()
            .flatten()
            .and_then(RawQuantTableId::index)
            .and_then(|index| self.versions.get(index))
    }

    pub(crate) fn retained_allocation_bytes(&self) -> Result<usize, JpegError> {
        checked_allocation_bytes::<[u16; 64]>(self.versions.capacity())
    }
}

#[derive(Debug, Default)]
pub(crate) struct HuffmanTables {
    pub(crate) dc: [Option<RawHuffmanTable>; TABLE_SLOTS],
    pub(crate) ac: [Option<RawHuffmanTable>; TABLE_SLOTS],
    active_dc: [Option<RawHuffmanTableId>; TABLE_SLOTS],
    active_ac: [Option<RawHuffmanTableId>; TABLE_SLOTS],
    pub(crate) versions: Vec<RawHuffmanVersion>,
}

#[derive(Debug)]
pub(crate) struct RawHuffmanVersion {
    pub(crate) role: HuffmanTableRole,
    pub(crate) raw: RawHuffmanTable,
}

impl HuffmanTables {
    pub(super) fn define(
        &mut self,
        class: u8,
        slot: usize,
        table: RawHuffmanTable,
        budget: &mut ParsedMetadataBudget,
    ) -> Result<(), JpegError> {
        if class > 1 {
            return Err(JpegError::InternalInvariant {
                reason: "unvalidated DHT class reached table definition",
            });
        }
        if slot >= TABLE_SLOTS {
            return Err(JpegError::InternalInvariant {
                reason: "unvalidated DHT slot reached table definition",
            });
        }
        let id = RawHuffmanTableId::for_next_index(self.versions.len())?;
        let role = if class == 0 {
            HuffmanTableRole::Dc
        } else {
            HuffmanTableRole::Ac
        };
        budget.try_push(
            &mut self.versions,
            RawHuffmanVersion {
                role,
                raw: table.clone(),
            },
        )?;
        if class == 0 {
            self.dc[slot] = Some(table);
            self.active_dc[slot] = Some(id);
        } else {
            self.ac[slot] = Some(table);
            self.active_ac[slot] = Some(id);
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn resolve_dc(
        &self,
        state: &ProgressiveTableState,
        slot: u8,
    ) -> Option<&RawHuffmanTable> {
        resolve_huffman(&self.versions, &state.dc, slot)
    }

    #[cfg(test)]
    pub(crate) fn resolve_ac(
        &self,
        state: &ProgressiveTableState,
        slot: u8,
    ) -> Option<&RawHuffmanTable> {
        resolve_huffman(&self.versions, &state.ac, slot)
    }

    pub(crate) fn active_dc_version_index(&self, slot: u8) -> Option<usize> {
        self.active_dc
            .get(usize::from(slot))
            .copied()
            .flatten()
            .and_then(RawHuffmanTableId::index)
    }

    pub(crate) fn active_ac_version_index(&self, slot: u8) -> Option<usize> {
        self.active_ac
            .get(usize::from(slot))
            .copied()
            .flatten()
            .and_then(RawHuffmanTableId::index)
    }

    pub(crate) fn retained_allocation_bytes(&self) -> Result<usize, JpegError> {
        checked_allocation_bytes::<RawHuffmanVersion>(self.versions.capacity())
    }
}

impl ProgressiveTableState {
    pub(crate) fn capture(huffman: &HuffmanTables, quant: &QuantTables) -> Self {
        Self {
            dc: huffman.active_dc,
            ac: huffman.active_ac,
            quant: quant.active,
        }
    }

    pub(crate) fn dc_version_index(self, slot: u8) -> Option<usize> {
        self.dc
            .get(usize::from(slot))
            .copied()
            .flatten()
            .and_then(RawHuffmanTableId::index)
    }

    pub(crate) fn ac_version_index(self, slot: u8) -> Option<usize> {
        self.ac
            .get(usize::from(slot))
            .copied()
            .flatten()
            .and_then(RawHuffmanTableId::index)
    }
}

#[cfg(test)]
fn resolve_huffman<'a>(
    versions: &'a [RawHuffmanVersion],
    slots: &[Option<RawHuffmanTableId>; TABLE_SLOTS],
    slot: u8,
) -> Option<&'a RawHuffmanTable> {
    slots
        .get(usize::from(slot))
        .copied()
        .flatten()
        .and_then(RawHuffmanTableId::index)
        .and_then(|index| versions.get(index))
        .map(|version| &version.raw)
}

fn nonzero_id(index: usize) -> Result<NonZeroU32, JpegError> {
    let one_based = index
        .checked_add(1)
        .and_then(|value| u32::try_from(value).ok())
        .and_then(NonZeroU32::new)
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })?;
    Ok(one_based)
}
