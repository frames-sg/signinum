// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared decode context for tile-oriented workloads.

use crate::allocation::{checked_add_allocation_bytes, try_reserve_for_len_with_live_budget};
use crate::entropy::huffman::HuffmanTable;
use crate::entropy::sequential::PreparedDecodePlan;
use crate::error::JpegError;
use crate::parse::tables::{HuffmanTableRole, RawHuffmanTable};
use alloc::vec::Vec;
use core::mem::size_of;
use j2k_core::{CacheStats, CodecContext};

const QUANT_CACHE_SLOTS: usize = 8;
const HUFFMAN_CACHE_SLOTS: usize = 8;
const PLAN_CACHE_SLOTS: usize = 8;
const MAX_DECODE_PLAN_CACHE_BYTES: usize = 16 * 1024 * 1024;
const TABLE_CACHE_ALLOCATION_RESERVE_BYTES: usize = 1024 * 1024;

/// Conservative heap reservation used by decode workspace planning. Decode
/// plan keys/entries are hard-capped at 16 MiB; the remaining MiB covers the
/// fallibly reserved inline Huffman-cache arena and allocator bookkeeping.
pub(crate) const MAX_DECODER_CONTEXT_ALLOCATION_BYTES: usize =
    MAX_DECODE_PLAN_CACHE_BYTES + TABLE_CACHE_ALLOCATION_RESERVE_BYTES;

#[derive(Debug, Clone)]
struct CachedQuantTable {
    digest: u64,
    table: [u16; 64],
}

#[derive(Debug)]
struct CachedHuffmanTable {
    digest: u64,
    role: HuffmanTableRole,
    raw: RawHuffmanTable,
    table: HuffmanTable,
}

#[derive(Debug)]
struct CachedDecodePlan {
    digest: u64,
    header_prefix: Vec<u8>,
    plan: PreparedDecodePlan,
    allocation_bytes: usize,
}

pub(crate) struct CachedDecodePlanLease<'a> {
    // The vacant slot's plan lives in the temporary decoder and its key in this
    // guard; both stay charged until restoration or abandoned-lease Drop.
    slot: &'a mut Option<CachedDecodePlan>,
    decode_plan_cache_bytes: &'a mut usize,
    cache_evictions: &'a mut u64,
    digest: u64,
    header_prefix: Vec<u8>,
    allocation_bytes: usize,
    restored: bool,
}

impl CachedDecodePlanLease<'_> {
    pub(crate) fn restore(mut self, plan: PreparedDecodePlan) {
        let header_prefix = core::mem::take(&mut self.header_prefix);
        *self.slot = Some(CachedDecodePlan {
            digest: self.digest,
            header_prefix,
            plan,
            allocation_bytes: self.allocation_bytes,
        });
        self.restored = true;
    }
}

impl Drop for CachedDecodePlanLease<'_> {
    fn drop(&mut self) {
        if self.restored {
            return;
        }
        *self.decode_plan_cache_bytes = self
            .decode_plan_cache_bytes
            .saturating_sub(self.allocation_bytes);
        *self.cache_evictions = self.cache_evictions.saturating_add(1);
    }
}

/// Shared decode context for WSI tile batches.
///
/// Reuse one context across many related JPEG tiles to amortize Huffman-table
/// construction and quant-table cloning when the stream family repeats the same
/// DHT/DQT definitions across tiles.
#[derive(Debug, Default)]
pub struct DecoderContext {
    quant_tables: [Option<CachedQuantTable>; QUANT_CACHE_SLOTS],
    huffman_tables: Vec<Option<CachedHuffmanTable>>,
    decode_plans: [Option<CachedDecodePlan>; PLAN_CACHE_SLOTS],
    decode_plan_cache_bytes: usize,
    cache_hits: u64,
    cache_misses: u64,
    cache_evictions: u64,
    #[cfg(test)]
    decode_plan_clones: u64,
}

impl DecoderContext {
    /// Create an empty decode context.
    #[must_use]
    pub fn new() -> Self {
        Self {
            quant_tables: core::array::from_fn(|_| None),
            huffman_tables: Vec::new(),
            decode_plans: core::array::from_fn(|_| None),
            decode_plan_cache_bytes: 0,
            cache_hits: 0,
            cache_misses: 0,
            cache_evictions: 0,
            #[cfg(test)]
            decode_plan_clones: 0,
        }
    }

    pub(crate) fn resolve_quant_table(&mut self, table: [u16; 64]) -> [u16; 64] {
        let digest = digest_quant_table(&table);
        self.resolve_quant_table_with_digest(table, digest)
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the digest cast only selects a cache shard and intentionally uses the native word"
    )]
    pub(crate) fn cached_decode_plan(&self, header_prefix: &[u8]) -> Option<&PreparedDecodePlan> {
        let digest = digest_bytes(header_prefix);
        let start = (digest as usize) % self.decode_plans.len();
        (0..self.decode_plans.len())
            .map(|probe| (start + probe) % self.decode_plans.len())
            .filter_map(|slot| self.decode_plans[slot].as_ref())
            .find(|cached| {
                cached.digest == digest && cached.header_prefix.as_slice() == header_prefix
            })
            .map(|cached| &cached.plan)
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the digest cast only selects a cache shard and intentionally uses the native word"
    )]
    pub(crate) fn lease_cached_decode_plan(
        &mut self,
        header_prefix: &[u8],
    ) -> Result<Option<(CachedDecodePlanLease<'_>, PreparedDecodePlan)>, JpegError> {
        let digest = digest_bytes(header_prefix);
        let start = (digest as usize) % self.decode_plans.len();
        let slot = (0..self.decode_plans.len())
            .map(|probe| (start + probe) % self.decode_plans.len())
            .find(|&slot| {
                self.decode_plans[slot].as_ref().is_some_and(|cached| {
                    cached.digest == digest && cached.header_prefix.as_slice() == header_prefix
                })
            });
        let Some(slot) = slot else {
            return Ok(None);
        };
        self.cache_hits = self.cache_hits.saturating_add(1);
        let cached = self.decode_plans[slot]
            .take()
            .ok_or(JpegError::InternalInvariant {
                reason: "matched cached decode plan disappeared before leasing",
            })?;
        let CachedDecodePlan {
            digest,
            header_prefix,
            plan,
            allocation_bytes,
        } = cached;
        Ok(Some((
            CachedDecodePlanLease {
                slot: &mut self.decode_plans[slot],
                decode_plan_cache_bytes: &mut self.decode_plan_cache_bytes,
                cache_evictions: &mut self.cache_evictions,
                digest,
                header_prefix,
                allocation_bytes,
                restored: false,
            },
            plan,
        )))
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the digest cast only selects a cache shard and intentionally uses the native word"
    )]
    fn resolve_quant_table_with_digest(&mut self, table: [u16; 64], digest: u64) -> [u16; 64] {
        let start = (digest as usize) % self.quant_tables.len();
        for probe in 0..self.quant_tables.len() {
            let slot = (start + probe) % self.quant_tables.len();
            match &self.quant_tables[slot] {
                Some(cached) if cached.digest == digest && cached.table == table => {
                    self.cache_hits = self.cache_hits.saturating_add(1);
                    return cached.table;
                }
                None => {
                    self.quant_tables[slot] = Some(CachedQuantTable { digest, table });
                    self.cache_misses = self.cache_misses.saturating_add(1);
                    return table;
                }
                Some(_) => {}
            }
        }

        let slot = start;
        self.quant_tables[slot] = Some(CachedQuantTable { digest, table });
        self.cache_misses = self.cache_misses.saturating_add(1);
        self.cache_evictions = self.cache_evictions.saturating_add(1);
        table
    }

    pub(crate) fn resolve_huffman_table_with_live_budget(
        &mut self,
        raw: &RawHuffmanTable,
        role: HuffmanTableRole,
        live_bytes: &mut usize,
        cap: usize,
    ) -> Result<HuffmanTable, JpegError> {
        let digest = digest_huffman_table(raw, role);
        self.resolve_huffman_table_with_digest_and_live_budget(raw, role, digest, live_bytes, cap)
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the digest cast only selects a cache shard and intentionally uses the native word"
    )]
    fn resolve_huffman_table_with_digest_and_live_budget(
        &mut self,
        raw: &RawHuffmanTable,
        role: HuffmanTableRole,
        digest: u64,
        live_bytes: &mut usize,
        cap: usize,
    ) -> Result<HuffmanTable, JpegError> {
        self.ensure_huffman_cache_slots(live_bytes, cap)?;
        let start = (digest as usize) % self.huffman_tables.len();
        for probe in 0..self.huffman_tables.len() {
            let slot = (start + probe) % self.huffman_tables.len();
            match &self.huffman_tables[slot] {
                Some(cached)
                    if cached.digest == digest && cached.role == role && &cached.raw == raw =>
                {
                    self.cache_hits = self.cache_hits.saturating_add(1);
                    return Ok(cached.table.clone());
                }
                None => {
                    let table = HuffmanTable::from_raw(raw, role)?;
                    self.huffman_tables[slot] = Some(CachedHuffmanTable {
                        digest,
                        role,
                        raw: raw.clone(),
                        table: table.clone(),
                    });
                    self.cache_misses = self.cache_misses.saturating_add(1);
                    return Ok(table);
                }
                Some(_) => {}
            }
        }

        let slot = start;
        let table = HuffmanTable::from_raw(raw, role)?;
        self.huffman_tables[slot] = Some(CachedHuffmanTable {
            digest,
            role,
            raw: raw.clone(),
            table: table.clone(),
        });
        self.cache_misses = self.cache_misses.saturating_add(1);
        self.cache_evictions = self.cache_evictions.saturating_add(1);
        Ok(table)
    }

    fn ensure_huffman_cache_slots(
        &mut self,
        live_bytes: &mut usize,
        cap: usize,
    ) -> Result<(), JpegError> {
        if self.huffman_tables.len() == HUFFMAN_CACHE_SLOTS {
            return Ok(());
        }
        try_reserve_for_len_with_live_budget(
            &mut self.huffman_tables,
            HUFFMAN_CACHE_SLOTS,
            live_bytes,
            cap,
        )?;
        self.huffman_tables
            .resize_with(HUFFMAN_CACHE_SLOTS, || None);
        Ok(())
    }

    pub(crate) fn resolve_decode_plan<F>(
        &mut self,
        header_prefix: &[u8],
        retained_external_bytes: usize,
        build: F,
    ) -> Result<PreparedDecodePlan, JpegError>
    where
        F: FnOnce(&mut Self) -> Result<PreparedDecodePlan, JpegError>,
    {
        let digest = digest_bytes(header_prefix);
        self.resolve_decode_plan_with_digest(header_prefix, digest, retained_external_bytes, build)
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the digest cast only selects a cache shard and intentionally uses the native word"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "cache lookup, fallible admission, and exact retained-byte accounting form one transaction"
    )]
    fn resolve_decode_plan_with_digest<F>(
        &mut self,
        header_prefix: &[u8],
        digest: u64,
        retained_external_bytes: usize,
        build: F,
    ) -> Result<PreparedDecodePlan, JpegError>
    where
        F: FnOnce(&mut Self) -> Result<PreparedDecodePlan, JpegError>,
    {
        let start = (digest as usize) % self.decode_plans.len();
        let retained_context_bytes = self.retained_allocation_bytes();
        let initial_live_bytes =
            checked_add_allocation_bytes(retained_external_bytes, retained_context_bytes)?;
        for probe in 0..self.decode_plans.len() {
            let slot = (start + probe) % self.decode_plans.len();
            match &self.decode_plans[slot] {
                Some(cached)
                    if cached.digest == digest
                        && cached.header_prefix.as_slice() == header_prefix =>
                {
                    self.cache_hits = self.cache_hits.saturating_add(1);
                    let mut live_bytes = initial_live_bytes;
                    let cloned = try_clone_decode_plan(&cached.plan, &mut live_bytes, None)?
                        .ok_or(JpegError::InternalInvariant {
                            reason: "cached decode plan unexpectedly bypassed cloning",
                        })?;
                    #[cfg(test)]
                    {
                        self.decode_plan_clones = self.decode_plan_clones.saturating_add(1);
                    }
                    return Ok(cloned);
                }
                Some(_) | None => {}
            }
        }
        let built = build(self)?;
        self.cache_misses = self.cache_misses.saturating_add(1);
        let predicted_bytes = decode_plan_entry_bytes(header_prefix.len(), &built)?;
        if predicted_bytes > MAX_DECODE_PLAN_CACHE_BYTES {
            // The cache is an optimization. A key that cannot fit its entire
            // exact entry under the aggregate cache budget is decoded but not
            // retained.
            return Ok(built);
        }
        self.evict_decode_plans_until_fits(start, predicted_bytes);
        let slot = self.first_empty_decode_plan_slot(start).unwrap_or_else(|| {
            self.evict_decode_plan_slot(start);
            start
        });
        let mut live_bytes = checked_add_allocation_bytes(
            retained_external_bytes,
            self.retained_allocation_bytes(),
        )?;
        live_bytes = checked_add_allocation_bytes(live_bytes, built.retained_allocation_bytes()?)?;
        let mut owned_prefix = Vec::new();
        try_reserve_for_len_with_live_budget(
            &mut owned_prefix,
            header_prefix.len(),
            &mut live_bytes,
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?;
        owned_prefix.extend_from_slice(header_prefix);
        let prefix_bytes = owned_prefix.capacity();
        let Some(cached_plan) = try_clone_decode_plan(&built, &mut live_bytes, Some(prefix_bytes))?
        else {
            return Ok(built);
        };
        #[cfg(test)]
        {
            self.decode_plan_clones = self.decode_plan_clones.saturating_add(1);
        }
        let allocation_bytes = owned_prefix
            .capacity()
            .checked_add(cached_plan.retained_allocation_bytes()?)
            .ok_or_else(context_cap_error)?;
        if allocation_bytes > MAX_DECODE_PLAN_CACHE_BYTES {
            return Ok(built);
        }
        self.evict_decode_plans_until_fits(start, allocation_bytes);
        if self.decode_plans[slot].is_some() {
            return Err(JpegError::InternalInvariant {
                reason: "decode-plan cache selected an occupied insertion slot",
            });
        }
        let new_cache_bytes = self
            .decode_plan_cache_bytes
            .checked_add(allocation_bytes)
            .ok_or(JpegError::InternalInvariant {
                reason: "decode-plan cache byte accounting overflowed",
            })?;
        let huffman_bytes = self
            .huffman_tables
            .capacity()
            .checked_mul(size_of::<Option<CachedHuffmanTable>>())
            .ok_or(JpegError::InternalInvariant {
                reason: "Huffman cache byte accounting overflowed",
            })?;
        let retained_after_insert =
            new_cache_bytes
                .checked_add(huffman_bytes)
                .ok_or(JpegError::InternalInvariant {
                    reason: "decoder context byte accounting overflowed",
                })?;
        if retained_after_insert > MAX_DECODER_CONTEXT_ALLOCATION_BYTES {
            // The cache is optional. Allocator capacity rounding can make an
            // otherwise valid entry exceed the context-only retention cap;
            // decode with `built` and release the attempted cache copy.
            return Ok(built);
        }
        self.decode_plans[slot] = Some(CachedDecodePlan {
            digest,
            header_prefix: owned_prefix,
            plan: cached_plan,
            allocation_bytes,
        });
        self.decode_plan_cache_bytes = new_cache_bytes;
        Ok(built)
    }

    fn first_empty_decode_plan_slot(&self, start: usize) -> Option<usize> {
        (0..self.decode_plans.len())
            .map(|probe| (start + probe) % self.decode_plans.len())
            .find(|&slot| self.decode_plans[slot].is_none())
    }

    fn evict_decode_plans_until_fits(&mut self, start: usize, incoming_bytes: usize) {
        for probe in 0..self.decode_plans.len() {
            if self.decode_plan_cache_bytes.saturating_add(incoming_bytes)
                <= MAX_DECODE_PLAN_CACHE_BYTES
            {
                break;
            }
            let slot = (start + probe) % self.decode_plans.len();
            self.evict_decode_plan_slot(slot);
        }
    }

    fn evict_decode_plan_slot(&mut self, slot: usize) {
        if let Some(cached) = self.decode_plans[slot].take() {
            self.decode_plan_cache_bytes = self
                .decode_plan_cache_bytes
                .saturating_sub(cached.allocation_bytes);
            self.cache_evictions = self.cache_evictions.saturating_add(1);
        }
    }

    pub(crate) fn retained_allocation_bytes(&self) -> usize {
        let huffman_bytes = self
            .huffman_tables
            .capacity()
            .saturating_mul(size_of::<Option<CachedHuffmanTable>>());
        self.decode_plan_cache_bytes.saturating_add(huffman_bytes)
    }

    #[cfg(test)]
    pub(crate) const fn decode_plan_clone_count(&self) -> u64 {
        self.decode_plan_clones
    }

    fn occupied_cache_slots(&self) -> u64 {
        let occupied = self
            .quant_tables
            .iter()
            .filter(|slot| slot.is_some())
            .count()
            + self
                .huffman_tables
                .iter()
                .filter(|slot| slot.is_some())
                .count()
            + self
                .decode_plans
                .iter()
                .filter(|slot| slot.is_some())
                .count();
        occupied as u64
    }
}

#[doc(hidden)]
impl CodecContext for DecoderContext {
    fn clear(&mut self) {
        *self = Self::new();
    }

    fn cache_stats(&self) -> CacheStats {
        CacheStats::with_slots(
            self.cache_hits,
            self.cache_misses,
            self.occupied_cache_slots(),
            self.cache_evictions,
        )
    }
}

fn digest_bytes(bytes: &[u8]) -> u64 {
    j2k_core::__j2k_fnv1a64_bytes!(bytes)
}

fn digest_quant_table(table: &[u16; 64]) -> u64 {
    let mut hash = j2k_core::__j2k_fnv1a64_init!();
    for &entry in table {
        for byte in entry.to_le_bytes() {
            j2k_core::__j2k_fnv1a64_update!(hash, byte);
        }
    }
    hash
}

fn digest_huffman_table(raw: &RawHuffmanTable, role: HuffmanTableRole) -> u64 {
    let mut hash = digest_bytes(&raw.bits);
    for &byte in raw.values.as_slice() {
        j2k_core::__j2k_fnv1a64_update!(hash, byte);
    }
    j2k_core::__j2k_fnv1a64_update!(
        hash,
        match role {
            HuffmanTableRole::Dc => 0u8,
            HuffmanTableRole::Ac => 1u8,
        }
    );
    hash
}

fn decode_plan_entry_bytes(
    header_prefix_len: usize,
    plan: &PreparedDecodePlan,
) -> Result<usize, JpegError> {
    header_prefix_len
        .checked_add(plan.retained_allocation_bytes()?)
        .ok_or_else(context_cap_error)
}

fn context_cap_error() -> JpegError {
    JpegError::MemoryCapExceeded {
        requested: usize::MAX,
        cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    }
}

fn try_clone_decode_plan(
    plan: &PreparedDecodePlan,
    live_bytes: &mut usize,
    cache_prefix_bytes: Option<usize>,
) -> Result<Option<PreparedDecodePlan>, JpegError> {
    let mut components = Vec::new();
    try_reserve_for_len_with_live_budget(
        &mut components,
        plan.components.len(),
        live_bytes,
        j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    )?;
    components.extend(plan.components.iter().cloned());
    if let Some(prefix_bytes) = cache_prefix_bytes {
        let projected = prefix_bytes
            .checked_add(PreparedDecodePlan::allocation_bytes_for_counts(
                components.capacity(),
                plan.huffman_tables.len(),
            )?)
            .ok_or_else(context_cap_error)?;
        if projected > MAX_DECODE_PLAN_CACHE_BYTES {
            return Ok(None);
        }
    }
    let huffman_tables = plan
        .huffman_tables
        .try_clone_with_live_budget(live_bytes, j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES)?;
    Ok(Some(PreparedDecodePlan {
        components,
        huffman_tables,
        sampling: plan.sampling,
        color_space: plan.color_space,
        restart_interval: plan.restart_interval,
        dimensions: plan.dimensions,
        scan_offset: plan.scan_offset,
        scratch_bytes: plan.scratch_bytes,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::sequential::PreparedComponentPlan;
    use crate::info::{ColorSpace, SamplingFactors};
    use alloc::vec;

    fn empty_plan(scan_offset: usize) -> PreparedDecodePlan {
        PreparedDecodePlan {
            components: vec![],
            huffman_tables: crate::entropy::huffman::PreparedHuffmanTables::try_with_capacity(0)
                .expect("empty arena"),
            sampling: SamplingFactors::from_validated_components(&[(1, 1)]),
            color_space: ColorSpace::Grayscale,
            restart_interval: None,
            dimensions: (16, 16),
            scan_offset,
            scratch_bytes: 0,
        }
    }

    fn resolve_huffman_table(
        ctx: &mut DecoderContext,
        raw: &RawHuffmanTable,
    ) -> Result<HuffmanTable, JpegError> {
        let mut live_bytes = ctx.retained_allocation_bytes();
        ctx.resolve_huffman_table_with_live_budget(
            raw,
            HuffmanTableRole::Dc,
            &mut live_bytes,
            MAX_DECODER_CONTEXT_ALLOCATION_BYTES,
        )
    }

    fn resolve_huffman_table_with_digest(
        ctx: &mut DecoderContext,
        raw: &RawHuffmanTable,
        digest: u64,
    ) -> Result<HuffmanTable, JpegError> {
        let mut live_bytes = ctx.retained_allocation_bytes();
        ctx.resolve_huffman_table_with_digest_and_live_budget(
            raw,
            HuffmanTableRole::Dc,
            digest,
            &mut live_bytes,
            MAX_DECODER_CONTEXT_ALLOCATION_BYTES,
        )
    }

    #[test]
    fn quant_table_cache_hits_return_same_value() {
        let mut ctx = DecoderContext::new();
        let first = ctx.resolve_quant_table([7; 64]);
        let second = ctx.resolve_quant_table([7; 64]);
        assert_eq!(first, second);

        let stats = ctx.cache_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.occupied_slots, 1);
        assert_eq!(stats.evictions, 0);
    }

    #[test]
    fn huffman_table_cache_hits_return_same_value() {
        let raw = RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: crate::parse::tables::HuffmanValues::from_slice(&[0]),
        };
        let mut ctx = DecoderContext::new();
        let first = resolve_huffman_table(&mut ctx, &raw).unwrap();
        let second = resolve_huffman_table(&mut ctx, &raw).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn quant_table_digest_collision_compares_full_table_contents() {
        let mut ctx = DecoderContext::new();
        let first = ctx.resolve_quant_table_with_digest([7; 64], 0);
        let second = ctx.resolve_quant_table_with_digest([8; 64], 0);

        assert_ne!(first, second);
        assert_eq!(first, [7; 64]);
        assert_eq!(second, [8; 64]);
        assert_eq!(ctx.cache_stats().misses, 2);
    }

    #[test]
    fn huffman_table_digest_collision_compares_full_raw_table_contents() {
        let first_raw = RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: crate::parse::tables::HuffmanValues::from_slice(&[0]),
        };
        let second_raw = RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: crate::parse::tables::HuffmanValues::from_slice(&[1]),
        };
        let mut ctx = DecoderContext::new();

        let first = resolve_huffman_table_with_digest(&mut ctx, &first_raw, 0).unwrap();
        let second = resolve_huffman_table_with_digest(&mut ctx, &second_raw, 0).unwrap();

        assert_ne!(first, second);
        assert_eq!(ctx.cache_stats().misses, 2);
    }

    #[test]
    fn prepared_plan_cache_hits_skip_rebuild() {
        let mut ctx = DecoderContext::new();
        let prefix = [0xFF, 0xD8, 0xFF, 0xDA];
        let mut builds = 0usize;

        let first = ctx
            .resolve_decode_plan(&prefix, 0, |_| {
                builds += 1;
                Ok(empty_plan(42))
            })
            .unwrap();

        let second = ctx
            .resolve_decode_plan(&prefix, 0, |_| {
                builds += 1;
                unreachable!("cache hit should bypass rebuild")
            })
            .unwrap();

        assert_eq!(builds, 1);
        assert_eq!(first.scan_offset, second.scan_offset);
    }

    #[test]
    fn cache_hit_clone_shares_one_exact_external_live_budget() {
        let raw = RawHuffmanTable {
            bits: [0; 16],
            values: crate::parse::tables::HuffmanValues::default(),
        };
        let mut huffman_tables =
            crate::entropy::huffman::PreparedHuffmanTables::try_with_capacity(1)
                .expect("bounded arena");
        let table = huffman_tables
            .push(HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc).expect("empty table"))
            .expect("reserved arena");
        let mut plan = empty_plan(7);
        plan.huffman_tables = huffman_tables;
        plan.components.push(PreparedComponentPlan {
            h: 1,
            v: 1,
            output_index: 0,
            quant: [1; 64],
            dc_table: Some(table),
            ac_table: Some(table),
        });

        let mut ctx = DecoderContext::new();
        let prefix = [0xFF, 0xD8, 0xFF, 0xDA];
        ctx.resolve_decode_plan(&prefix, 0, |_| Ok(plan))
            .expect("initial cache insertion");
        let cached_plan_bytes = ctx
            .decode_plans
            .iter()
            .flatten()
            .next()
            .expect("cached plan")
            .plan
            .retained_allocation_bytes()
            .expect("cached plan bytes");
        let exact_external = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES
            .checked_sub(ctx.retained_allocation_bytes())
            .and_then(|remaining| remaining.checked_sub(cached_plan_bytes))
            .expect("fixture leaves an external budget");

        ctx.resolve_decode_plan(&prefix, exact_external, |_| {
            unreachable!("cache hit must bypass rebuild")
        })
        .expect("exact live boundary");
        assert!(matches!(
            ctx.resolve_decode_plan(&prefix, exact_external + 1, |_| {
                unreachable!("cache hit must bypass rebuild")
            }),
            Err(JpegError::MemoryCapExceeded { .. })
        ));
    }

    #[test]
    fn prepared_plan_digest_collision_compares_full_header_prefix() {
        let mut ctx = DecoderContext::new();
        let first = ctx
            .resolve_decode_plan_with_digest(b"first", 0, 0, |_| Ok(empty_plan(1)))
            .unwrap();
        let second = ctx
            .resolve_decode_plan_with_digest(b"second", 0, 0, |_| Ok(empty_plan(2)))
            .unwrap();
        let first_hit = ctx
            .resolve_decode_plan_with_digest(b"first", 0, 0, |_| {
                unreachable!("full-key cache hit must bypass rebuild")
            })
            .unwrap();

        assert_eq!(first.scan_offset, 1);
        assert_eq!(second.scan_offset, 2);
        assert_eq!(first_hit.scan_offset, 1);
        assert_eq!(ctx.cache_stats().hits, 1);
    }

    #[test]
    fn prepared_plan_lease_compares_full_header_prefix_after_digest_collision() {
        let requested = b"requested";
        let digest = digest_bytes(requested);
        let slot_count = u64::try_from(PLAN_CACHE_SLOTS).unwrap();
        let start = usize::try_from(digest % slot_count).unwrap();
        let wrong_slot = start;
        let matching_slot = (start + 1) % PLAN_CACHE_SLOTS;
        let wrong_prefix = b"collision".to_vec();
        let matching_prefix = requested.to_vec();
        let wrong_plan = empty_plan(11);
        let matching_plan = empty_plan(29);
        let wrong_bytes = decode_plan_entry_bytes(wrong_prefix.capacity(), &wrong_plan).unwrap();
        let matching_bytes =
            decode_plan_entry_bytes(matching_prefix.capacity(), &matching_plan).unwrap();
        let mut ctx = DecoderContext::new();
        ctx.decode_plans[wrong_slot] = Some(CachedDecodePlan {
            digest,
            header_prefix: wrong_prefix,
            plan: wrong_plan,
            allocation_bytes: wrong_bytes,
        });
        ctx.decode_plans[matching_slot] = Some(CachedDecodePlan {
            digest,
            header_prefix: matching_prefix,
            plan: matching_plan,
            allocation_bytes: matching_bytes,
        });
        ctx.decode_plan_cache_bytes = wrong_bytes + matching_bytes;
        let retained = ctx.retained_allocation_bytes();
        let occupied = ctx.cache_stats().occupied_slots;

        assert!(ctx.lease_cached_decode_plan(b"absent").unwrap().is_none());
        let (lease, plan) = ctx
            .lease_cached_decode_plan(requested)
            .unwrap()
            .expect("matching full prefix must lease");
        assert_eq!(plan.scan_offset, 29);
        lease.restore(plan);

        assert_eq!(
            ctx.cached_decode_plan(requested)
                .expect("leased plan restored under exact key")
                .scan_offset,
            29
        );
        assert_eq!(ctx.retained_allocation_bytes(), retained);
        assert_eq!(ctx.cache_stats().occupied_slots, occupied);
    }

    #[test]
    fn prepared_plan_cache_full_eviction_is_deterministic() {
        let mut ctx = DecoderContext::new();
        let cache_slots = u8::try_from(PLAN_CACHE_SLOTS).expect("plan cache slot count fits u8");
        for key in 0..cache_slots {
            ctx.resolve_decode_plan_with_digest(&[key], 0, 0, |_| Ok(empty_plan(usize::from(key))))
                .unwrap();
        }
        ctx.resolve_decode_plan_with_digest(&[cache_slots], 0, 0, |_| {
            Ok(empty_plan(PLAN_CACHE_SLOTS))
        })
        .unwrap();

        assert_eq!(ctx.cache_stats().evictions, 1);
        let mut rebuilt = false;
        let first = ctx
            .resolve_decode_plan_with_digest(&[0], 0, 0, |_| {
                rebuilt = true;
                Ok(empty_plan(99))
            })
            .unwrap();
        assert!(rebuilt, "the start slot must be the deterministic victim");
        assert_eq!(first.scan_offset, 99);
    }

    #[test]
    fn decode_plan_cache_entry_boundary_bypasses_oversized_keys() {
        let plan = empty_plan(0);
        assert_eq!(
            decode_plan_entry_bytes(MAX_DECODE_PLAN_CACHE_BYTES, &plan).unwrap(),
            MAX_DECODE_PLAN_CACHE_BYTES
        );
        assert!(
            decode_plan_entry_bytes(MAX_DECODE_PLAN_CACHE_BYTES + 1, &plan).unwrap()
                > MAX_DECODE_PLAN_CACHE_BYTES
        );
    }

    #[test]
    fn decode_plan_cache_entry_counts_tables_retained_after_table_cache_eviction() {
        let raw = RawHuffmanTable {
            bits: [0; 16],
            values: crate::parse::tables::HuffmanValues::default(),
        };
        let mut huffman_tables =
            crate::entropy::huffman::PreparedHuffmanTables::try_with_capacity(1)
                .expect("bounded arena");
        let table = huffman_tables
            .push(HuffmanTable::from_raw(&raw, HuffmanTableRole::Dc).expect("empty table"))
            .expect("reserved arena");
        let mut plan = empty_plan(0);
        plan.huffman_tables = huffman_tables;
        plan.components.push(PreparedComponentPlan {
            h: 1,
            v: 1,
            output_index: 0,
            quant: [1; 64],
            dc_table: Some(table),
            ac_table: Some(table),
        });

        let entry_bytes = decode_plan_entry_bytes(0, &plan).expect("bounded plan");
        assert_eq!(entry_bytes, plan.retained_allocation_bytes().unwrap());
        let logical_bytes = PreparedDecodePlan::allocation_bytes_for_counts(
            plan.components.len(),
            plan.huffman_tables.len(),
        )
        .expect("logical plan bytes");
        // `push` from an empty vector commonly retains spare component slots.
        // Cache entry accounting must use that allocator-returned capacity,
        // not the one logical component requested by this fixture.
        let component_spare_bytes = (plan.components.capacity() - plan.components.len())
            * size_of::<PreparedComponentPlan>();
        assert_eq!(entry_bytes - logical_bytes, component_spare_bytes);
        assert!(entry_bytes > size_of::<PreparedComponentPlan>());
    }

    #[test]
    fn oversized_decode_plan_key_is_not_retained() {
        let prefix = vec![0u8; MAX_DECODE_PLAN_CACHE_BYTES + 1];
        let mut ctx = DecoderContext::new();
        let mut builds = 0usize;
        for _ in 0..2 {
            ctx.resolve_decode_plan(&prefix, 0, |_| {
                builds += 1;
                Ok(empty_plan(builds))
            })
            .unwrap();
        }

        assert_eq!(builds, 2, "oversized keys must bypass the cache");
        assert_eq!(ctx.decode_plan_cache_bytes, 0);
        assert!(ctx.decode_plans.iter().all(Option::is_none));
    }

    #[test]
    fn identical_raw_huffman_tables_keep_roles_distinct_even_on_digest_collision() {
        let raw = RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: crate::parse::tables::HuffmanValues::from_slice(&[0]),
        };
        for force_collision in [false, true] {
            let mut ctx = DecoderContext::new();
            for role in [
                HuffmanTableRole::Dc,
                HuffmanTableRole::Ac,
                HuffmanTableRole::Dc,
                HuffmanTableRole::Ac,
            ] {
                let mut live_bytes = ctx.retained_allocation_bytes();
                let digest = if force_collision {
                    0
                } else {
                    digest_huffman_table(&raw, role)
                };
                let table = ctx
                    .resolve_huffman_table_with_digest_and_live_budget(
                        &raw,
                        role,
                        digest,
                        &mut live_bytes,
                        MAX_DECODER_CONTEXT_ALLOCATION_BYTES,
                    )
                    .expect("role-specific cached table");
                assert_eq!(table.dc().is_ok(), role == HuffmanTableRole::Dc);
                assert_eq!(table.ac().is_ok(), role == HuffmanTableRole::Ac);
                assert_eq!(live_bytes, ctx.retained_allocation_bytes());
            }
            let stats = ctx.cache_stats();
            assert_eq!(stats.misses, 2);
            assert_eq!(stats.hits, 2);
            assert_eq!(stats.occupied_slots, 2);
        }
    }

    #[test]
    fn role_specific_cache_slots_obey_exact_live_byte_boundary() {
        let raw = RawHuffmanTable {
            bits: [0; 16],
            values: crate::parse::tables::HuffmanValues::default(),
        };
        let external_bytes = 7;
        let cache_bytes = HUFFMAN_CACHE_SLOTS * size_of::<Option<CachedHuffmanTable>>();
        let exact_cap = external_bytes + cache_bytes;
        let mut ctx = DecoderContext::new();
        let mut live_bytes = external_bytes;
        let error = ctx
            .resolve_huffman_table_with_live_budget(
                &raw,
                HuffmanTableRole::Dc,
                &mut live_bytes,
                exact_cap - 1,
            )
            .expect_err("one byte below the slot allocation");
        assert!(
            matches!(error, JpegError::MemoryCapExceeded { requested, cap }
            if requested == exact_cap && cap == exact_cap - 1)
        );
        assert_eq!(ctx.retained_allocation_bytes(), 0);
        assert_eq!(live_bytes, external_bytes);
        assert_eq!(ctx.cache_stats().occupied_slots, 0);

        for role in [HuffmanTableRole::Dc, HuffmanTableRole::Ac] {
            ctx.resolve_huffman_table_with_live_budget(&raw, role, &mut live_bytes, exact_cap)
                .expect("both roles fit in the already charged slots");
            assert_eq!(live_bytes, exact_cap);
            assert_eq!(ctx.retained_allocation_bytes(), cache_bytes);
            assert_eq!(
                ctx.retained_allocation_bytes(),
                ctx.huffman_tables.capacity() * size_of::<Option<CachedHuffmanTable>>()
            );
        }
        assert_eq!(ctx.cache_stats().occupied_slots, 2);
    }

    #[test]
    fn context_reserve_covers_all_fixed_table_cache_allocations() {
        let maximum_table_bytes =
            HUFFMAN_CACHE_SLOTS.saturating_mul(size_of::<Option<CachedHuffmanTable>>());
        assert!(maximum_table_bytes <= TABLE_CACHE_ALLOCATION_RESERVE_BYTES);

        let ctx = DecoderContext::new();
        assert_eq!(ctx.retained_allocation_bytes(), 0);
        assert!(ctx.decode_plan_cache_bytes <= MAX_DECODE_PLAN_CACHE_BYTES);
    }
}
