// SPDX-License-Identifier: Apache-2.0

//! Shared decode context for tile-oriented workloads.

use crate::entropy::huffman::HuffmanTable;
use crate::entropy::sequential::PreparedDecodePlan;
use crate::error::JpegError;
use crate::error::Warning;
use crate::info::Info;
use crate::parse::tables::RawHuffmanTable;
use alloc::sync::Arc;
use signinum_core::{CacheStats, CodecContext};

const QUANT_CACHE_SLOTS: usize = 8;
const HUFFMAN_CACHE_SLOTS: usize = 8;
const PLAN_CACHE_SLOTS: usize = 8;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

#[derive(Debug, Clone)]
struct CachedQuantTable {
    digest: u64,
    table: Arc<[u16; 64]>,
}

#[derive(Debug, Clone)]
struct CachedHuffmanTable {
    digest: u64,
    table: Arc<HuffmanTable>,
}

#[derive(Debug, Clone)]
struct CachedDecodePlan {
    digest: u64,
    header_prefix: Arc<[u8]>,
    info: Info,
    warnings: Arc<[Warning]>,
    plan: PreparedDecodePlan,
}

/// Shared decode context for WSI tile batches.
///
/// Reuse one context across many related JPEG tiles to amortize Huffman-table
/// construction and quant-table cloning when the stream family repeats the same
/// DHT/DQT definitions across tiles.
#[derive(Debug, Default)]
pub struct DecoderContext {
    quant_tables: [Option<CachedQuantTable>; QUANT_CACHE_SLOTS],
    huffman_tables: [Option<CachedHuffmanTable>; HUFFMAN_CACHE_SLOTS],
    decode_plans: [Option<CachedDecodePlan>; PLAN_CACHE_SLOTS],
}

impl DecoderContext {
    /// Create an empty decode context.
    #[must_use]
    pub fn new() -> Self {
        Self {
            quant_tables: core::array::from_fn(|_| None),
            huffman_tables: core::array::from_fn(|_| None),
            decode_plans: core::array::from_fn(|_| None),
        }
    }

    pub(crate) fn resolve_quant_table(&mut self, table: [u16; 64]) -> Arc<[u16; 64]> {
        let digest = digest_quant_table(&table);
        let start = (digest as usize) % self.quant_tables.len();
        for probe in 0..self.quant_tables.len() {
            let slot = (start + probe) % self.quant_tables.len();
            match &self.quant_tables[slot] {
                Some(cached) if cached.digest == digest => return Arc::clone(&cached.table),
                None => {
                    let table = Arc::new(table);
                    self.quant_tables[slot] = Some(CachedQuantTable {
                        digest,
                        table: Arc::clone(&table),
                    });
                    return table;
                }
                Some(_) => {}
            }
        }

        let slot = start;
        let table = Arc::new(table);
        self.quant_tables[slot] = Some(CachedQuantTable {
            digest,
            table: Arc::clone(&table),
        });
        table
    }

    pub(crate) fn resolve_huffman_table(
        &mut self,
        raw: &RawHuffmanTable,
    ) -> Result<Arc<HuffmanTable>, JpegError> {
        let digest = digest_huffman_table(raw);
        let start = (digest as usize) % self.huffman_tables.len();
        for probe in 0..self.huffman_tables.len() {
            let slot = (start + probe) % self.huffman_tables.len();
            match &self.huffman_tables[slot] {
                Some(cached) if cached.digest == digest => return Ok(Arc::clone(&cached.table)),
                None => {
                    let table = Arc::new(HuffmanTable::from_raw(raw)?);
                    self.huffman_tables[slot] = Some(CachedHuffmanTable {
                        digest,
                        table: Arc::clone(&table),
                    });
                    return Ok(table);
                }
                Some(_) => {}
            }
        }

        let slot = start;
        let table = Arc::new(HuffmanTable::from_raw(raw)?);
        self.huffman_tables[slot] = Some(CachedHuffmanTable {
            digest,
            table: Arc::clone(&table),
        });
        Ok(table)
    }

    pub(crate) fn resolve_decode_plan<F>(
        &mut self,
        header_prefix: &[u8],
        build: F,
    ) -> Result<(Info, Arc<[Warning]>, PreparedDecodePlan), JpegError>
    where
        F: FnOnce(&mut Self) -> Result<(Info, Arc<[Warning]>, PreparedDecodePlan), JpegError>,
    {
        let digest = digest_bytes(header_prefix);
        let start = (digest as usize) % self.decode_plans.len();
        let mut empty_slot = None;
        for probe in 0..self.decode_plans.len() {
            let slot = (start + probe) % self.decode_plans.len();
            match &self.decode_plans[slot] {
                Some(cached)
                    if cached.digest == digest
                        && cached.header_prefix.as_ref() == header_prefix =>
                {
                    return Ok((
                        cached.info.clone(),
                        Arc::clone(&cached.warnings),
                        cached.plan.clone(),
                    ));
                }
                None => {
                    empty_slot = Some(slot);
                    break;
                }
                Some(_) => {}
            }
        }

        let built = build(self)?;
        let slot = empty_slot.unwrap_or(start);
        self.decode_plans[slot] = Some(CachedDecodePlan {
            digest,
            header_prefix: Arc::<[u8]>::from(header_prefix),
            info: built.0.clone(),
            warnings: Arc::clone(&built.1),
            plan: built.2.clone(),
        });
        Ok(built)
    }
}

impl CodecContext for DecoderContext {
    fn clear(&mut self) {
        *self = Self::new();
    }

    fn cache_stats(&self) -> CacheStats {
        CacheStats::default()
    }
}

fn digest_bytes(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn digest_quant_table(table: &[u16; 64]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &entry in table {
        for byte in entry.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

fn digest_huffman_table(raw: &RawHuffmanTable) -> u64 {
    let mut hash = digest_bytes(&raw.bits);
    for &byte in raw.values.as_slice() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::info::{ColorSpace, SamplingFactors, SofKind};
    use alloc::vec;

    #[test]
    fn quant_table_cache_hits_return_same_arc() {
        let mut ctx = DecoderContext::new();
        let first = ctx.resolve_quant_table([7; 64]);
        let second = ctx.resolve_quant_table([7; 64]);
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn huffman_table_cache_hits_return_same_arc() {
        let raw = RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: crate::parse::tables::HuffmanValues::from_slice(&[0]),
        };
        let mut ctx = DecoderContext::new();
        let first = ctx.resolve_huffman_table(&raw).unwrap();
        let second = ctx.resolve_huffman_table(&raw).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn prepared_plan_cache_hits_skip_rebuild() {
        let mut ctx = DecoderContext::new();
        let prefix = [0xFF, 0xD8, 0xFF, 0xDA];
        let warnings = Arc::<[Warning]>::from([]);
        let mut builds = 0usize;

        let first = ctx
            .resolve_decode_plan(&prefix, |_| {
                builds += 1;
                Ok((
                    Info {
                        dimensions: (16, 16),
                        color_space: ColorSpace::YCbCr,
                        sampling: SamplingFactors::from_components(&[(2, 2), (1, 1), (1, 1)]),
                        sof_kind: SofKind::Baseline8,
                        bit_depth: 8,
                        restart_interval: None,
                        mcu_geometry: crate::info::McuGeometry {
                            width: 16,
                            height: 16,
                            columns: 1,
                            rows: 1,
                            count: 1,
                        },
                        scan_count: 1,
                    },
                    Arc::clone(&warnings),
                    PreparedDecodePlan {
                        components: vec![],
                        sampling: SamplingFactors::from_components(&[(2, 2), (1, 1), (1, 1)]),
                        color_space: ColorSpace::YCbCr,
                        restart_interval: None,
                        dimensions: (16, 16),
                        scan_offset: 42,
                        scratch_bytes: 0,
                    },
                ))
            })
            .unwrap();

        let second = ctx
            .resolve_decode_plan(&prefix, |_| {
                builds += 1;
                unreachable!("cache hit should bypass rebuild")
            })
            .unwrap();

        assert_eq!(builds, 1);
        assert_eq!(first.0, second.0);
        assert!(Arc::ptr_eq(&first.1, &second.1));
        assert_eq!(first.2.scan_offset, second.2.scan_offset);
    }
}
