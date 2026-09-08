// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prepared progressive decode metadata and checked table access.

use alloc::vec::Vec;

use crate::allocation::{checked_add_allocation_bytes, checked_allocation_bytes};
use crate::entropy::huffman::{
    AcHuffmanTable, DcHuffmanTable, PreparedHuffmanTableId, PreparedHuffmanTables,
};
use crate::error::JpegError;
use crate::info::{ColorSpace, SamplingFactors};

#[derive(Debug, Clone)]
pub(crate) struct PreparedProgressiveComponentPlan {
    pub(crate) h: u8,
    pub(crate) v: u8,
    pub(crate) output_index: usize,
    pub(crate) quant: [u16; 64],
    pub(crate) block_cols: u32,
    pub(crate) block_rows: u32,
    pub(crate) sample_width: u32,
    pub(crate) sample_height: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedProgressiveScanComponent {
    pub(crate) component_index: usize,
    pub(crate) dc_table: Option<PreparedHuffmanTableId>,
    pub(crate) ac_table: Option<PreparedHuffmanTableId>,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedProgressiveScan {
    pub(crate) component_start: usize,
    pub(crate) component_len: usize,
    pub(crate) ss: u8,
    pub(crate) se: u8,
    pub(crate) ah: u8,
    pub(crate) al: u8,
    pub(crate) entropy_offset: usize,
    /// Absolute parser-recorded entropy boundary; code zero denotes EOF.
    pub(crate) terminal_offset: usize,
    pub(crate) terminal_code: u8,
    pub(crate) restart_interval: Option<u16>,
}

#[derive(Debug)]
pub(crate) struct PreparedProgressivePlan {
    pub(crate) components: Vec<PreparedProgressiveComponentPlan>,
    pub(crate) scan_components: Vec<PreparedProgressiveScanComponent>,
    pub(crate) scans: Vec<PreparedProgressiveScan>,
    pub(crate) huffman_tables: PreparedHuffmanTables,
    pub(crate) sampling: SamplingFactors,
    pub(crate) color_space: ColorSpace,
    pub(crate) dimensions: (u32, u32),
    pub(crate) mcu_cols: u32,
    pub(crate) mcu_rows: u32,
    pub(crate) scratch_bytes: usize,
}

impl PreparedProgressivePlan {
    pub(crate) fn retained_allocation_bytes(&self) -> Result<usize, JpegError> {
        let component_bytes = checked_allocation_bytes::<PreparedProgressiveComponentPlan>(
            self.components.capacity(),
        )?;
        let scan_component_bytes = checked_allocation_bytes::<PreparedProgressiveScanComponent>(
            self.scan_components.capacity(),
        )?;
        let scan_bytes =
            checked_allocation_bytes::<PreparedProgressiveScan>(self.scans.capacity())?;
        let mut total = checked_add_allocation_bytes(component_bytes, scan_component_bytes)?;
        total = checked_add_allocation_bytes(total, scan_bytes)?;
        checked_add_allocation_bytes(total, self.huffman_tables.retained_allocation_bytes()?)
    }

    pub(super) fn scan_components(
        &self,
        scan: &PreparedProgressiveScan,
    ) -> Result<&[PreparedProgressiveScanComponent], JpegError> {
        let end = scan.component_start.checked_add(scan.component_len).ok_or(
            JpegError::InternalInvariant {
                reason: "progressive scan-component range overflow",
            },
        )?;
        self.scan_components
            .get(scan.component_start..end)
            .ok_or(JpegError::InternalInvariant {
                reason: "progressive scan-component range is outside the prepared plan",
            })
    }

    pub(super) fn dc_table(
        &self,
        id: Option<PreparedHuffmanTableId>,
    ) -> Result<DcHuffmanTable<'_>, JpegError> {
        self.huffman_tables
            .get_dc(id.ok_or(JpegError::InternalInvariant {
                reason: "progressive scan references a missing prepared Huffman table",
            })?)
    }

    pub(super) fn ac_table(
        &self,
        id: Option<PreparedHuffmanTableId>,
    ) -> Result<AcHuffmanTable<'_>, JpegError> {
        self.huffman_tables
            .get_ac(id.ok_or(JpegError::InternalInvariant {
                reason: "progressive scan references a missing prepared Huffman table",
            })?)
    }
}

#[derive(Debug)]
pub(crate) struct ProgressiveDctBlocks {
    pub(crate) quantized: Vec<Vec<[i32; 64]>>,
}

impl ProgressiveDctBlocks {
    pub(crate) fn capacity_bytes(&self) -> Result<usize, JpegError> {
        super::allocation::coefficient_capacity_bytes(self.quantized.capacity(), &self.quantized)
    }
}
