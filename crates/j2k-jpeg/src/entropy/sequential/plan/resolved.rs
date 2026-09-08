// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked prepared-table resolution outside entropy hot loops.

use super::{PreparedComponentPlan, PreparedDecodePlan};
use crate::entropy::huffman::{AcHuffmanTable, DcHuffmanTable, PreparedHuffmanTableId};
use crate::error::JpegError;

/// Borrowed component metadata with table IDs resolved once before entering an
/// entropy hot loop.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ResolvedPreparedComponentPlan<'a> {
    pub(crate) quant: &'a [u16; 64],
    pub(crate) dc_table: DcHuffmanTable<'a>,
    pub(crate) ac_table: AcHuffmanTable<'a>,
}

impl PreparedDecodePlan {
    pub(crate) fn dc_table(
        &self,
        component: &PreparedComponentPlan,
    ) -> Result<DcHuffmanTable<'_>, JpegError> {
        self.huffman_tables
            .get_dc(component.dc_table.ok_or(JpegError::InternalInvariant {
                reason: "prepared component references a missing Huffman table",
            })?)
    }

    pub(crate) fn dc_table_by_id(
        &self,
        id: PreparedHuffmanTableId,
    ) -> Result<DcHuffmanTable<'_>, JpegError> {
        self.huffman_tables.get_dc(id)
    }

    pub(crate) fn ac_table(
        &self,
        component: &PreparedComponentPlan,
    ) -> Result<AcHuffmanTable<'_>, JpegError> {
        self.huffman_tables
            .get_ac(component.ac_table.ok_or(JpegError::InternalInvariant {
                reason: "prepared component references a missing Huffman table",
            })?)
    }

    pub(crate) fn resolve_component<'a>(
        &'a self,
        component: &'a PreparedComponentPlan,
    ) -> Result<ResolvedPreparedComponentPlan<'a>, JpegError> {
        Ok(ResolvedPreparedComponentPlan {
            quant: &component.quant,
            dc_table: self.dc_table(component)?,
            ac_table: self.ac_table(component)?,
        })
    }

    pub(crate) fn resolved_component(
        &self,
        index: usize,
    ) -> Result<ResolvedPreparedComponentPlan<'_>, JpegError> {
        let component = self
            .components
            .get(index)
            .ok_or(JpegError::InternalInvariant {
                reason: "prepared component index is outside the decode plan",
            })?;
        self.resolve_component(component)
    }
}
