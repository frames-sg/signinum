// SPDX-License-Identifier: Apache-2.0

use crate::error::{HuffmanFailure, JpegError, MarkerKind};
use crate::info::{ColorSpace, SamplingFactors, SofKind};
use crate::parse::header::parse_header;
use crate::parse::scan::ScanComponent;
use crate::parse::tables::RawHuffmanTable;
use alloc::vec::Vec;

const PLANNER_FAST_BITS: u8 = 12;
const PLANNER_FAST_ENTRIES: usize = 1 << PLANNER_FAST_BITS;
const MAX_NONRESTART_ENTROPY_CHECKPOINTS: u32 = 2048;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalFast420PacketError {
    Decode(JpegError),
    UnsupportedSof(SofKind),
    UnsupportedColorSpace(ColorSpace),
    UnsupportedSampling,
    UnsupportedComponentOrder,
    MissingScan,
    MissingQuantTable { slot: u8 },
    MissingHuffmanTable { kind: TableKind, slot: u8 },
    EntropyMarkerUnsupported { marker: u8 },
    TruncatedEntropy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableKind {
    Dc,
    Ac,
}

impl From<JpegError> for MetalFast420PacketError {
    fn from(value: JpegError) -> Self {
        Self::Decode(value)
    }
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalHuffmanTable {
    pub bits: [u8; 16],
    pub values_len: u16,
    pub values: [u8; 256],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegMetalEntropyCheckpointV1 {
    pub mcu_index: u32,
    pub entropy_pos: u32,
    pub bit_acc: u64,
    pub bit_count: u32,
    pub y_prev_dc: i32,
    pub cb_prev_dc: i32,
    pub cr_prev_dc: i32,
    pub reserved: u32,
}

impl JpegMetalEntropyCheckpointV1 {
    fn restart(mcu_index: u32, entropy_pos: u32) -> Self {
        Self {
            mcu_index,
            entropy_pos,
            bit_acc: 0,
            bit_count: 0,
            y_prev_dc: 0,
            cb_prev_dc: 0,
            cr_prev_dc: 0,
            reserved: 0,
        }
    }
}

impl MetalHuffmanTable {
    fn from_raw(raw: &RawHuffmanTable) -> Self {
        let mut values = [0u8; 256];
        let slice = raw.values.as_slice();
        values[..slice.len()].copy_from_slice(slice);
        Self {
            bits: raw.bits,
            values_len: slice.len() as u16,
            values,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegMetalFast420PacketV1 {
    pub dimensions: (u32, u32),
    pub mcus_per_row: u32,
    pub mcu_rows: u32,
    pub restart_interval_mcus: u32,
    pub restart_offsets: Vec<u32>,
    pub entropy_checkpoints: Vec<JpegMetalEntropyCheckpointV1>,
    pub y_quant: [u16; 64],
    pub cb_quant: [u16; 64],
    pub cr_quant: [u16; 64],
    pub y_dc_table: MetalHuffmanTable,
    pub y_ac_table: MetalHuffmanTable,
    pub cb_dc_table: MetalHuffmanTable,
    pub cb_ac_table: MetalHuffmanTable,
    pub cr_dc_table: MetalHuffmanTable,
    pub cr_ac_table: MetalHuffmanTable,
    pub entropy_bytes: Vec<u8>,
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegMetalFast422PacketV1 {
    pub dimensions: (u32, u32),
    pub mcus_per_row: u32,
    pub mcu_rows: u32,
    pub restart_interval_mcus: u32,
    pub restart_offsets: Vec<u32>,
    pub y_quant: [u16; 64],
    pub cb_quant: [u16; 64],
    pub cr_quant: [u16; 64],
    pub y_dc_table: MetalHuffmanTable,
    pub y_ac_table: MetalHuffmanTable,
    pub cb_dc_table: MetalHuffmanTable,
    pub cb_ac_table: MetalHuffmanTable,
    pub cr_dc_table: MetalHuffmanTable,
    pub cr_ac_table: MetalHuffmanTable,
    pub entropy_bytes: Vec<u8>,
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegMetalFast444PacketV1 {
    pub dimensions: (u32, u32),
    pub mcus_per_row: u32,
    pub mcu_rows: u32,
    pub restart_interval_mcus: u32,
    pub restart_offsets: Vec<u32>,
    pub entropy_checkpoints: Vec<JpegMetalEntropyCheckpointV1>,
    pub y_quant: [u16; 64],
    pub cb_quant: [u16; 64],
    pub cr_quant: [u16; 64],
    pub y_dc_table: MetalHuffmanTable,
    pub y_ac_table: MetalHuffmanTable,
    pub cb_dc_table: MetalHuffmanTable,
    pub cb_ac_table: MetalHuffmanTable,
    pub cr_dc_table: MetalHuffmanTable,
    pub cr_ac_table: MetalHuffmanTable,
    pub entropy_bytes: Vec<u8>,
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegMetalGrayPacketV1 {
    pub dimensions: (u32, u32),
    pub mcus_per_row: u32,
    pub mcu_rows: u32,
    pub restart_interval_mcus: u32,
    pub restart_offsets: Vec<u32>,
    pub y_quant: [u16; 64],
    pub y_dc_table: MetalHuffmanTable,
    pub y_ac_table: MetalHuffmanTable,
    pub entropy_bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EntropySegments {
    entropy_bytes: Vec<u8>,
    restart_offsets: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
enum PlannerLayout {
    Fast420,
    Fast444,
}

#[derive(Debug, Clone)]
struct PlannerHuffman {
    fast: [(u8, u8); PLANNER_FAST_ENTRIES],
    max_code: [i32; 17],
    val_offset: [i32; 17],
    values: [u8; 256],
    values_len: usize,
}

impl PlannerHuffman {
    fn from_metal(raw: &MetalHuffmanTable) -> Result<Self, MetalFast420PacketError> {
        let mut fast = [(0u8, 0u8); PLANNER_FAST_ENTRIES];
        let mut max_code = [-1i32; 17];
        let mut val_offset = [0i32; 17];
        let mut values = [0u8; 256];
        let values_len = usize::from(raw.values_len);
        values[..values_len].copy_from_slice(&raw.values[..values_len]);

        let mut huffsize = [0u8; 256];
        let mut huffsize_len = 0usize;
        for (len_minus_1, &count) in raw.bits.iter().enumerate() {
            let len = (len_minus_1 + 1) as u8;
            for _ in 0..count {
                huffsize[huffsize_len] = len;
                huffsize_len += 1;
            }
        }

        let mut huffcode = [0u16; 256];
        let mut code: u32 = 0;
        let mut si = huffsize.first().copied().unwrap_or(0);
        for (k, &s) in huffsize[..huffsize_len].iter().enumerate() {
            while s != si {
                code <<= 1;
                si += 1;
            }
            huffcode[k] = code as u16;
            code = code.checked_add(1).ok_or_else(planner_code_overflow)?;
        }
        if si > 0 && (code - 1) >= (1u32 << si) {
            return Err(planner_code_overflow());
        }

        let mut k = 0usize;
        for len_minus_1 in 0..16 {
            let len = len_minus_1 + 1;
            let count = raw.bits[len_minus_1] as usize;
            if count == 0 {
                continue;
            }
            let min_code = i32::from(huffcode[k]);
            max_code[len] = i32::from(huffcode[k + count - 1]);
            val_offset[len] = k as i32 - min_code;
            k += count;
        }

        k = 0;
        for len_minus_1 in 0..PLANNER_FAST_BITS as usize {
            let len = (len_minus_1 + 1) as u8;
            let count = raw.bits[len_minus_1] as usize;
            for _ in 0..count {
                let c = huffcode[k];
                let fast_index_base = (c as usize) << (PLANNER_FAST_BITS - len);
                let fast_count = 1 << (PLANNER_FAST_BITS - len);
                for j in 0..fast_count {
                    fast[fast_index_base + j] = (raw.values[k], len);
                }
                k += 1;
            }
        }

        Ok(Self {
            fast,
            max_code,
            val_offset,
            values,
            values_len,
        })
    }

    fn decode(&self, reader: &mut PlannerBitReader<'_>) -> Result<u8, MetalFast420PacketError> {
        reader.ensure_bits_padded(PLANNER_FAST_BITS);
        let peek = reader.peek_bits(PLANNER_FAST_BITS) as usize;
        let (sym, len) = self.fast[peek];
        if len != 0 {
            reader.consume_bits(len);
            return Ok(sym);
        }

        reader.ensure_bits_padded(16);
        let code16 = reader.peek_bits(16) as i32;
        for len in (PLANNER_FAST_BITS as usize + 1)..=16 {
            let l = len as u8;
            let c = code16 >> (16 - l);
            if c <= self.max_code[len] {
                reader.consume_bits(l);
                let idx = (c + self.val_offset[len]) as usize;
                if idx >= self.values_len {
                    return Err(planner_invalid_symbol());
                }
                return Ok(self.values[idx]);
            }
        }
        Err(planner_code_overflow())
    }
}

struct PlannerBitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    acc: u64,
    bits: u8,
}

impl<'a> PlannerBitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            acc: 0,
            bits: 0,
        }
    }

    fn checkpoint(
        &self,
        mcu_index: u32,
        y_prev_dc: i32,
        cb_prev_dc: i32,
        cr_prev_dc: i32,
    ) -> Result<JpegMetalEntropyCheckpointV1, MetalFast420PacketError> {
        Ok(JpegMetalEntropyCheckpointV1 {
            mcu_index,
            entropy_pos: u32::try_from(self.pos).map_err(|_| planner_truncated_entropy())?,
            bit_acc: self.acc,
            bit_count: u32::from(self.bits),
            y_prev_dc,
            cb_prev_dc,
            cr_prev_dc,
            reserved: 0,
        })
    }

    fn ensure_bits(&mut self, n: u8) -> Result<(), MetalFast420PacketError> {
        while self.bits < n {
            if !self.refill_one_byte() {
                return Err(planner_table_exhausted());
            }
        }
        Ok(())
    }

    fn ensure_bits_padded(&mut self, n: u8) {
        while self.bits < n {
            if !self.refill_one_byte() {
                self.acc |= 1u64 << (63 - self.bits);
                self.bits += 1;
            }
        }
    }

    fn refill_one_byte(&mut self) -> bool {
        let Some(&byte) = self.bytes.get(self.pos) else {
            return false;
        };
        let shift = 64 - 8 - self.bits;
        self.acc |= u64::from(byte) << shift;
        self.pos += 1;
        self.bits += 8;
        true
    }

    fn peek_bits(&self, n: u8) -> u32 {
        if n == 0 {
            0
        } else {
            (self.acc >> (64 - n)) as u32
        }
    }

    fn consume_bits(&mut self, n: u8) {
        self.acc <<= n;
        self.bits -= n;
    }

    fn receive_extend(&mut self, ssss: u8) -> Result<i32, MetalFast420PacketError> {
        if ssss == 0 {
            return Ok(0);
        }
        self.ensure_bits(ssss)?;
        let value = self.peek_bits(ssss) as i32;
        self.consume_bits(ssss);
        let threshold = 1i32 << (ssss - 1);
        Ok(if value < threshold {
            value + ((-1i32) << ssss) + 1
        } else {
            value
        })
    }
}

pub fn build_metal_fast420_packet(
    bytes: &[u8],
) -> Result<JpegMetalFast420PacketV1, MetalFast420PacketError> {
    let header = parse_header(bytes)?;
    if !matches!(header.sof_kind, SofKind::Baseline8 | SofKind::Extended8) {
        return Err(MetalFast420PacketError::UnsupportedSof(header.sof_kind));
    }
    if header.bit_depth != 8 {
        return Err(MetalFast420PacketError::Decode(
            JpegError::UnsupportedBitDepth {
                depth: header.bit_depth,
            },
        ));
    }
    if header.color_space() != ColorSpace::YCbCr {
        return Err(MetalFast420PacketError::UnsupportedColorSpace(
            header.color_space(),
        ));
    }
    if header.sampling != SamplingFactors::from_components(&[(2, 2), (1, 1), (1, 1)]) {
        return Err(MetalFast420PacketError::UnsupportedSampling);
    }
    let scan = header
        .scan
        .as_ref()
        .ok_or(MetalFast420PacketError::MissingScan)?;
    let [y_scan, cb_scan, cr_scan] = ordered_scan_triplet(&header.component_ids, &scan.components)?;

    let y_quant = quant_for_component(&header.quant_table_ids, &header.quant_tables.entries, 0)?;
    let cb_quant = quant_for_component(&header.quant_table_ids, &header.quant_tables.entries, 1)?;
    let cr_quant = quant_for_component(&header.quant_table_ids, &header.quant_tables.entries, 2)?;
    let y_dc_table = huffman_table(&header.huffman_tables.dc, TableKind::Dc, y_scan.dc_table)?;
    let y_ac_table = huffman_table(&header.huffman_tables.ac, TableKind::Ac, y_scan.ac_table)?;
    let cb_dc_table = huffman_table(&header.huffman_tables.dc, TableKind::Dc, cb_scan.dc_table)?;
    let cb_ac_table = huffman_table(&header.huffman_tables.ac, TableKind::Ac, cb_scan.ac_table)?;
    let cr_dc_table = huffman_table(&header.huffman_tables.dc, TableKind::Dc, cr_scan.dc_table)?;
    let cr_ac_table = huffman_table(&header.huffman_tables.ac, TableKind::Ac, cr_scan.ac_table)?;

    let entropy_offset = header
        .sos_offset
        .ok_or(MetalFast420PacketError::MissingScan)?;
    let restart_interval_mcus = u32::from(header.restart_interval.unwrap_or(0));
    let EntropySegments {
        entropy_bytes,
        restart_offsets,
    } = extract_entropy_segments(&bytes[entropy_offset..], header.restart_interval)?;
    let (width, height) = header.dimensions;
    let mcus_per_row = width.div_ceil(16);
    let mcu_rows = height.div_ceil(16);
    let entropy_checkpoints = build_triplet_entropy_checkpoints(
        PlannerLayout::Fast420,
        &entropy_bytes,
        mcus_per_row
            .checked_mul(mcu_rows)
            .expect("JPEG Metal fast420 MCU count fits in u32"),
        restart_interval_mcus,
        &restart_offsets,
        [&y_dc_table, &cb_dc_table, &cr_dc_table],
        [&y_ac_table, &cb_ac_table, &cr_ac_table],
    )?;

    Ok(JpegMetalFast420PacketV1 {
        dimensions: header.dimensions,
        mcus_per_row,
        mcu_rows,
        restart_interval_mcus,
        restart_offsets,
        entropy_checkpoints,
        y_quant,
        cb_quant,
        cr_quant,
        y_dc_table,
        y_ac_table,
        cb_dc_table,
        cb_ac_table,
        cr_dc_table,
        cr_ac_table,
        entropy_bytes,
    })
}

pub fn build_metal_fast444_packet(
    bytes: &[u8],
) -> Result<JpegMetalFast444PacketV1, MetalFast420PacketError> {
    let header = parse_header(bytes)?;
    if !matches!(header.sof_kind, SofKind::Baseline8 | SofKind::Extended8) {
        return Err(MetalFast420PacketError::UnsupportedSof(header.sof_kind));
    }
    if header.bit_depth != 8 {
        return Err(MetalFast420PacketError::Decode(
            JpegError::UnsupportedBitDepth {
                depth: header.bit_depth,
            },
        ));
    }
    if !matches!(header.color_space(), ColorSpace::YCbCr | ColorSpace::Rgb) {
        return Err(MetalFast420PacketError::UnsupportedColorSpace(
            header.color_space(),
        ));
    }
    if header.sampling != SamplingFactors::from_components(&[(1, 1), (1, 1), (1, 1)]) {
        return Err(MetalFast420PacketError::UnsupportedSampling);
    }
    let scan = header
        .scan
        .as_ref()
        .ok_or(MetalFast420PacketError::MissingScan)?;
    let [y_scan, cb_scan, cr_scan] = ordered_scan_triplet(&header.component_ids, &scan.components)?;

    let y_quant = quant_for_component(&header.quant_table_ids, &header.quant_tables.entries, 0)?;
    let cb_quant = quant_for_component(&header.quant_table_ids, &header.quant_tables.entries, 1)?;
    let cr_quant = quant_for_component(&header.quant_table_ids, &header.quant_tables.entries, 2)?;
    let y_dc_table = huffman_table(&header.huffman_tables.dc, TableKind::Dc, y_scan.dc_table)?;
    let y_ac_table = huffman_table(&header.huffman_tables.ac, TableKind::Ac, y_scan.ac_table)?;
    let cb_dc_table = huffman_table(&header.huffman_tables.dc, TableKind::Dc, cb_scan.dc_table)?;
    let cb_ac_table = huffman_table(&header.huffman_tables.ac, TableKind::Ac, cb_scan.ac_table)?;
    let cr_dc_table = huffman_table(&header.huffman_tables.dc, TableKind::Dc, cr_scan.dc_table)?;
    let cr_ac_table = huffman_table(&header.huffman_tables.ac, TableKind::Ac, cr_scan.ac_table)?;

    let entropy_offset = header
        .sos_offset
        .ok_or(MetalFast420PacketError::MissingScan)?;
    let restart_interval_mcus = u32::from(header.restart_interval.unwrap_or(0));
    let EntropySegments {
        entropy_bytes,
        restart_offsets,
    } = extract_entropy_segments(&bytes[entropy_offset..], header.restart_interval)?;
    let (width, height) = header.dimensions;
    let mcus_per_row = width.div_ceil(8);
    let mcu_rows = height.div_ceil(8);
    let entropy_checkpoints = build_triplet_entropy_checkpoints(
        PlannerLayout::Fast444,
        &entropy_bytes,
        mcus_per_row
            .checked_mul(mcu_rows)
            .expect("JPEG Metal fast444 MCU count fits in u32"),
        restart_interval_mcus,
        &restart_offsets,
        [&y_dc_table, &cb_dc_table, &cr_dc_table],
        [&y_ac_table, &cb_ac_table, &cr_ac_table],
    )?;

    Ok(JpegMetalFast444PacketV1 {
        dimensions: header.dimensions,
        mcus_per_row,
        mcu_rows,
        restart_interval_mcus,
        restart_offsets,
        entropy_checkpoints,
        y_quant,
        cb_quant,
        cr_quant,
        y_dc_table,
        y_ac_table,
        cb_dc_table,
        cb_ac_table,
        cr_dc_table,
        cr_ac_table,
        entropy_bytes,
    })
}

pub fn build_metal_fast422_packet(
    bytes: &[u8],
) -> Result<JpegMetalFast422PacketV1, MetalFast420PacketError> {
    let header = parse_header(bytes)?;
    if !matches!(header.sof_kind, SofKind::Baseline8 | SofKind::Extended8) {
        return Err(MetalFast420PacketError::UnsupportedSof(header.sof_kind));
    }
    if header.bit_depth != 8 {
        return Err(MetalFast420PacketError::Decode(
            JpegError::UnsupportedBitDepth {
                depth: header.bit_depth,
            },
        ));
    }
    if header.color_space() != ColorSpace::YCbCr {
        return Err(MetalFast420PacketError::UnsupportedColorSpace(
            header.color_space(),
        ));
    }
    if header.sampling != SamplingFactors::from_components(&[(2, 1), (1, 1), (1, 1)]) {
        return Err(MetalFast420PacketError::UnsupportedSampling);
    }
    let scan = header
        .scan
        .as_ref()
        .ok_or(MetalFast420PacketError::MissingScan)?;
    let [y_scan, cb_scan, cr_scan] = ordered_scan_triplet(&header.component_ids, &scan.components)?;

    let y_quant = quant_for_component(&header.quant_table_ids, &header.quant_tables.entries, 0)?;
    let cb_quant = quant_for_component(&header.quant_table_ids, &header.quant_tables.entries, 1)?;
    let cr_quant = quant_for_component(&header.quant_table_ids, &header.quant_tables.entries, 2)?;
    let y_dc_table = huffman_table(&header.huffman_tables.dc, TableKind::Dc, y_scan.dc_table)?;
    let y_ac_table = huffman_table(&header.huffman_tables.ac, TableKind::Ac, y_scan.ac_table)?;
    let cb_dc_table = huffman_table(&header.huffman_tables.dc, TableKind::Dc, cb_scan.dc_table)?;
    let cb_ac_table = huffman_table(&header.huffman_tables.ac, TableKind::Ac, cb_scan.ac_table)?;
    let cr_dc_table = huffman_table(&header.huffman_tables.dc, TableKind::Dc, cr_scan.dc_table)?;
    let cr_ac_table = huffman_table(&header.huffman_tables.ac, TableKind::Ac, cr_scan.ac_table)?;

    let entropy_offset = header
        .sos_offset
        .ok_or(MetalFast420PacketError::MissingScan)?;
    let restart_interval_mcus = u32::from(header.restart_interval.unwrap_or(0));
    let EntropySegments {
        entropy_bytes,
        restart_offsets,
    } = extract_entropy_segments(&bytes[entropy_offset..], header.restart_interval)?;
    let (width, height) = header.dimensions;

    Ok(JpegMetalFast422PacketV1 {
        dimensions: header.dimensions,
        mcus_per_row: width.div_ceil(16),
        mcu_rows: height.div_ceil(8),
        restart_interval_mcus,
        restart_offsets,
        y_quant,
        cb_quant,
        cr_quant,
        y_dc_table,
        y_ac_table,
        cb_dc_table,
        cb_ac_table,
        cr_dc_table,
        cr_ac_table,
        entropy_bytes,
    })
}

pub fn build_metal_gray_packet(
    bytes: &[u8],
) -> Result<JpegMetalGrayPacketV1, MetalFast420PacketError> {
    let header = parse_header(bytes)?;
    if !matches!(header.sof_kind, SofKind::Baseline8 | SofKind::Extended8) {
        return Err(MetalFast420PacketError::UnsupportedSof(header.sof_kind));
    }
    if header.bit_depth != 8 {
        return Err(MetalFast420PacketError::Decode(
            JpegError::UnsupportedBitDepth {
                depth: header.bit_depth,
            },
        ));
    }
    if header.color_space() != ColorSpace::Grayscale {
        return Err(MetalFast420PacketError::UnsupportedColorSpace(
            header.color_space(),
        ));
    }
    if header.sampling != SamplingFactors::from_components(&[(1, 1)]) {
        return Err(MetalFast420PacketError::UnsupportedSampling);
    }

    let scan = header
        .scan
        .as_ref()
        .ok_or(MetalFast420PacketError::MissingScan)?;
    if header.component_ids.len() != 1 || scan.components.len() != 1 {
        return Err(MetalFast420PacketError::UnsupportedComponentOrder);
    }
    if scan.components[0].id != header.component_ids[0] {
        return Err(MetalFast420PacketError::UnsupportedComponentOrder);
    }

    let y_quant = quant_for_component(&header.quant_table_ids, &header.quant_tables.entries, 0)?;
    let y_dc_table = huffman_table(
        &header.huffman_tables.dc,
        TableKind::Dc,
        scan.components[0].dc_table,
    )?;
    let y_ac_table = huffman_table(
        &header.huffman_tables.ac,
        TableKind::Ac,
        scan.components[0].ac_table,
    )?;

    let entropy_offset = header
        .sos_offset
        .ok_or(MetalFast420PacketError::MissingScan)?;
    let restart_interval_mcus = u32::from(header.restart_interval.unwrap_or(0));
    let EntropySegments {
        entropy_bytes,
        restart_offsets,
    } = extract_entropy_segments(&bytes[entropy_offset..], header.restart_interval)?;
    let (width, height) = header.dimensions;

    Ok(JpegMetalGrayPacketV1 {
        dimensions: header.dimensions,
        mcus_per_row: width.div_ceil(8),
        mcu_rows: height.div_ceil(8),
        restart_interval_mcus,
        restart_offsets,
        y_quant,
        y_dc_table,
        y_ac_table,
        entropy_bytes,
    })
}

pub fn build_metal_fast420_packet_for_decoder(
    decoder: &crate::decoder::Decoder<'_>,
) -> Result<JpegMetalFast420PacketV1, MetalFast420PacketError> {
    build_metal_fast420_packet(decoder.bytes)
}

pub fn build_metal_fast444_packet_for_decoder(
    decoder: &crate::decoder::Decoder<'_>,
) -> Result<JpegMetalFast444PacketV1, MetalFast420PacketError> {
    build_metal_fast444_packet(decoder.bytes)
}

pub fn build_metal_fast422_packet_for_decoder(
    decoder: &crate::decoder::Decoder<'_>,
) -> Result<JpegMetalFast422PacketV1, MetalFast420PacketError> {
    build_metal_fast422_packet(decoder.bytes)
}

pub fn build_metal_gray_packet_for_decoder(
    decoder: &crate::decoder::Decoder<'_>,
) -> Result<JpegMetalGrayPacketV1, MetalFast420PacketError> {
    build_metal_gray_packet(decoder.bytes)
}

fn quant_for_component(
    quant_table_ids: &[u8],
    tables: &[Option<[u16; 64]>; 4],
    component_idx: usize,
) -> Result<[u16; 64], MetalFast420PacketError> {
    let slot = *quant_table_ids
        .get(component_idx)
        .ok_or(MetalFast420PacketError::UnsupportedComponentOrder)?;
    tables[slot as usize].ok_or(MetalFast420PacketError::MissingQuantTable { slot })
}

fn ordered_scan_triplet(
    component_ids: &[u8],
    scan_components: &[ScanComponent],
) -> Result<[ScanComponent; 3], MetalFast420PacketError> {
    if component_ids.len() != 3 || scan_components.len() != 3 {
        return Err(MetalFast420PacketError::UnsupportedComponentOrder);
    }

    let mut ordered = [None; 3];
    for (index, &component_id) in component_ids.iter().enumerate() {
        let Some(component) = scan_components
            .iter()
            .copied()
            .find(|component| component.id == component_id)
        else {
            return Err(MetalFast420PacketError::UnsupportedComponentOrder);
        };
        ordered[index] = Some(component);
    }

    match ordered {
        [Some(first), Some(second), Some(third)] => Ok([first, second, third]),
        _ => Err(MetalFast420PacketError::UnsupportedComponentOrder),
    }
}

fn huffman_table(
    tables: &[Option<RawHuffmanTable>; 4],
    kind: TableKind,
    slot: u8,
) -> Result<MetalHuffmanTable, MetalFast420PacketError> {
    let raw = tables[slot as usize]
        .as_ref()
        .ok_or(MetalFast420PacketError::MissingHuffmanTable { kind, slot })?;
    Ok(MetalHuffmanTable::from_raw(raw))
}

fn planner_error(reason: HuffmanFailure) -> MetalFast420PacketError {
    MetalFast420PacketError::Decode(JpegError::HuffmanDecode { mcu: 0, reason })
}

fn planner_code_overflow() -> MetalFast420PacketError {
    planner_error(HuffmanFailure::CodeOverflow)
}

fn planner_invalid_symbol() -> MetalFast420PacketError {
    planner_error(HuffmanFailure::InvalidSymbol)
}

fn planner_table_exhausted() -> MetalFast420PacketError {
    planner_error(HuffmanFailure::TableExhausted)
}

fn planner_truncated_entropy() -> MetalFast420PacketError {
    MetalFast420PacketError::TruncatedEntropy
}

fn nonrestart_entropy_chunk_mcus(total_mcus: u32) -> u32 {
    total_mcus
        .div_ceil(MAX_NONRESTART_ENTROPY_CHECKPOINTS)
        .max(1)
}

fn restart_entropy_checkpoints(
    total_mcus: u32,
    restart_interval_mcus: u32,
    restart_offsets: &[u32],
) -> Vec<JpegMetalEntropyCheckpointV1> {
    restart_offsets
        .iter()
        .enumerate()
        .filter_map(|(index, &offset)| {
            let Ok(index) = u32::try_from(index) else {
                return None;
            };
            let mcu_index = index.saturating_mul(restart_interval_mcus);
            (mcu_index < total_mcus)
                .then_some(JpegMetalEntropyCheckpointV1::restart(mcu_index, offset))
        })
        .collect()
}

fn build_triplet_entropy_checkpoints(
    layout: PlannerLayout,
    entropy_bytes: &[u8],
    total_mcus: u32,
    restart_interval_mcus: u32,
    restart_offsets: &[u32],
    dc_tables: [&MetalHuffmanTable; 3],
    ac_tables: [&MetalHuffmanTable; 3],
) -> Result<Vec<JpegMetalEntropyCheckpointV1>, MetalFast420PacketError> {
    if total_mcus == 0 {
        return Ok(vec![JpegMetalEntropyCheckpointV1::restart(0, 0)]);
    }
    if restart_interval_mcus != 0 {
        let checkpoints =
            restart_entropy_checkpoints(total_mcus, restart_interval_mcus, restart_offsets);
        if !checkpoints.is_empty() {
            return Ok(checkpoints);
        }
        return Ok(vec![JpegMetalEntropyCheckpointV1::restart(0, 0)]);
    }

    let dc_tables = alloc::vec![
        PlannerHuffman::from_metal(dc_tables[0])?,
        PlannerHuffman::from_metal(dc_tables[1])?,
        PlannerHuffman::from_metal(dc_tables[2])?,
    ]
    .into_boxed_slice();
    let ac_tables = alloc::vec![
        PlannerHuffman::from_metal(ac_tables[0])?,
        PlannerHuffman::from_metal(ac_tables[1])?,
        PlannerHuffman::from_metal(ac_tables[2])?,
    ]
    .into_boxed_slice();
    let mut reader = PlannerBitReader::new(entropy_bytes);
    let mut checkpoints = Vec::new();
    let chunk_mcus = nonrestart_entropy_chunk_mcus(total_mcus);
    let mut next_checkpoint_mcu = 0u32;
    let mut y_prev_dc = 0i32;
    let mut cb_prev_dc = 0i32;
    let mut cr_prev_dc = 0i32;

    for mcu_index in 0..total_mcus {
        if mcu_index == next_checkpoint_mcu {
            checkpoints.push(reader.checkpoint(mcu_index, y_prev_dc, cb_prev_dc, cr_prev_dc)?);
            next_checkpoint_mcu = next_checkpoint_mcu.saturating_add(chunk_mcus);
        }
        skip_triplet_mcu(
            layout,
            &mut reader,
            [&dc_tables[0], &dc_tables[1], &dc_tables[2]],
            [&ac_tables[0], &ac_tables[1], &ac_tables[2]],
            &mut y_prev_dc,
            &mut cb_prev_dc,
            &mut cr_prev_dc,
        )?;
    }

    if checkpoints.is_empty() {
        checkpoints.push(JpegMetalEntropyCheckpointV1::restart(0, 0));
    }
    Ok(checkpoints)
}

fn skip_triplet_mcu(
    layout: PlannerLayout,
    reader: &mut PlannerBitReader<'_>,
    dc_tables: [&PlannerHuffman; 3],
    ac_tables: [&PlannerHuffman; 3],
    y_prev_dc: &mut i32,
    cb_prev_dc: &mut i32,
    cr_prev_dc: &mut i32,
) -> Result<(), MetalFast420PacketError> {
    match layout {
        PlannerLayout::Fast420 => {
            for _ in 0..4 {
                skip_block(reader, dc_tables[0], ac_tables[0], y_prev_dc)?;
            }
            skip_block(reader, dc_tables[1], ac_tables[1], cb_prev_dc)?;
            skip_block(reader, dc_tables[2], ac_tables[2], cr_prev_dc)?;
        }
        PlannerLayout::Fast444 => {
            skip_block(reader, dc_tables[0], ac_tables[0], y_prev_dc)?;
            skip_block(reader, dc_tables[1], ac_tables[1], cb_prev_dc)?;
            skip_block(reader, dc_tables[2], ac_tables[2], cr_prev_dc)?;
        }
    }
    Ok(())
}

fn skip_block(
    reader: &mut PlannerBitReader<'_>,
    dc_table: &PlannerHuffman,
    ac_table: &PlannerHuffman,
    prev_dc: &mut i32,
) -> Result<(), MetalFast420PacketError> {
    let ssss = dc_table.decode(reader)?;
    if ssss > 15 {
        return Err(planner_invalid_symbol());
    }
    let diff = reader.receive_extend(ssss)?;
    *prev_dc = prev_dc.wrapping_add(diff);

    let mut k = 1usize;
    while k < 64 {
        let sym = ac_table.decode(reader)?;
        let run = usize::from(sym >> 4);
        let ssss = sym & 0x0F;
        if ssss == 0 {
            if run == 15 {
                k += 16;
                continue;
            }
            break;
        }

        k += run;
        if k >= 64 {
            return Err(planner_invalid_symbol());
        }
        let _ = reader.receive_extend(ssss)?;
        k += 1;
    }
    Ok(())
}

fn extract_entropy_segments(
    bytes: &[u8],
    restart_interval: Option<u16>,
) -> Result<EntropySegments, MetalFast420PacketError> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut restart_offsets = vec![0u32];
    let mut pos = 0usize;
    let mut expected_rst = 0xD0u8;
    while pos < bytes.len() {
        let byte = bytes[pos];
        if byte != 0xFF {
            out.push(byte);
            pos += 1;
            continue;
        }
        let next = *bytes
            .get(pos + 1)
            .ok_or(MetalFast420PacketError::TruncatedEntropy)?;
        match next {
            0x00 => {
                out.push(0xFF);
                pos += 2;
            }
            0xD9 => {
                return Ok(EntropySegments {
                    entropy_bytes: out,
                    restart_offsets,
                });
            }
            0xD0..=0xD7 if restart_interval.unwrap_or(0) != 0 => {
                if next != expected_rst {
                    return Err(MetalFast420PacketError::EntropyMarkerUnsupported { marker: next });
                }
                restart_offsets.push(
                    u32::try_from(out.len())
                        .map_err(|_| MetalFast420PacketError::TruncatedEntropy)?,
                );
                expected_rst = if expected_rst == 0xD7 {
                    0xD0
                } else {
                    expected_rst + 1
                };
                pos += 2;
            }
            marker => {
                return Err(MetalFast420PacketError::EntropyMarkerUnsupported { marker });
            }
        }
    }
    Err(MetalFast420PacketError::Decode(JpegError::MissingMarker {
        marker: MarkerKind::Eoi,
    }))
}
