// SPDX-License-Identifier: MIT OR Apache-2.0

//! Actual-capacity accounting for parsed-to-prepared decoder construction.

use alloc::vec::Vec;
#[cfg(test)]
use core::mem::size_of;

use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

use crate::allocation::try_reserve_for_len_with_live_budget;
use crate::context::DecoderContext;
use crate::entropy::huffman::{HuffmanTable, PreparedHuffmanTables};
use crate::error::JpegError;
use crate::parse::tables::{HuffmanTableRole, RawHuffmanTable};

/// One live-byte ledger for the entire parsed-to-prepared handoff.
///
/// Every retained vector reserve updates `live_bytes` with the allocator's
/// reported capacity before the next reserve starts. `context_bytes` lets
/// budgeted context-cache growth share the same total without double counting.
#[derive(Debug)]
pub(in crate::decoder) struct PreparedConstructionBudget {
    external_live_bytes: usize,
    parsed_bytes: usize,
    context_bytes: usize,
    live_bytes: usize,
    cap: usize,
}

impl PreparedConstructionBudget {
    pub(super) fn with_external_live(
        external_live_bytes: usize,
        parsed_bytes: usize,
        context_bytes: usize,
    ) -> Result<Self, JpegError> {
        Self::with_external_live_and_cap(
            external_live_bytes,
            parsed_bytes,
            context_bytes,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )
    }

    #[cfg(test)]
    fn with_cap(parsed_bytes: usize, context_bytes: usize, cap: usize) -> Result<Self, JpegError> {
        Self::with_external_live_and_cap(0, parsed_bytes, context_bytes, cap)
    }

    fn with_external_live_and_cap(
        external_live_bytes: usize,
        parsed_bytes: usize,
        context_bytes: usize,
        cap: usize,
    ) -> Result<Self, JpegError> {
        let live_bytes = checked_add_under_cap(external_live_bytes, parsed_bytes, cap)?;
        let live_bytes = checked_add_under_cap(live_bytes, context_bytes, cap)?;
        Ok(Self {
            external_live_bytes,
            parsed_bytes,
            context_bytes,
            live_bytes,
            cap,
        })
    }

    pub(super) fn try_vec<T>(&mut self, len: usize) -> Result<Vec<T>, JpegError> {
        let mut values = Vec::new();
        try_reserve_for_len_with_live_budget(&mut values, len, &mut self.live_bytes, self.cap)?;
        Ok(values)
    }

    pub(super) fn try_huffman_tables(
        &mut self,
        len: usize,
    ) -> Result<PreparedHuffmanTables, JpegError> {
        PreparedHuffmanTables::try_with_capacity_and_live_budget(
            len,
            &mut self.live_bytes,
            self.cap,
        )
    }

    pub(super) fn resolve_huffman_table(
        &mut self,
        ctx: &mut DecoderContext,
        raw: &RawHuffmanTable,
        role: HuffmanTableRole,
    ) -> Result<HuffmanTable, JpegError> {
        let non_context_bytes = self.live_bytes.checked_sub(self.context_bytes).ok_or(
            JpegError::InternalInvariant {
                reason: "prepared construction context bytes exceed its live-byte ledger",
            },
        )?;
        let table =
            ctx.resolve_huffman_table_with_live_budget(raw, role, &mut self.live_bytes, self.cap)?;
        let context_bytes = ctx.retained_allocation_bytes();
        let expected = checked_add_under_cap(non_context_bytes, context_bytes, self.cap)?;
        if self.live_bytes != expected {
            return Err(JpegError::InternalInvariant {
                reason: "budgeted context growth did not reconcile its actual capacity",
            });
        }
        self.context_bytes = context_bytes;
        Ok(table)
    }

    /// A plan-cache hit clones a plan, while a miss may retain a second clone
    /// in the context. Both paths already budget their allocations internally;
    /// rebase this ledger to the exact owners returned by the cache operation.
    pub(super) fn rebase_after_plan_cache(
        &mut self,
        context_bytes: usize,
        prepared_bytes: usize,
    ) -> Result<(), JpegError> {
        let base = checked_add_under_cap(self.external_live_bytes, self.parsed_bytes, self.cap)?;
        let base = checked_add_under_cap(base, context_bytes, self.cap)?;
        self.live_bytes = checked_add_under_cap(base, prepared_bytes, self.cap)?;
        self.context_bytes = context_bytes;
        Ok(())
    }

    pub(super) fn verify_retained(
        &self,
        context_bytes: usize,
        prepared_bytes: usize,
    ) -> Result<(), JpegError> {
        let base = checked_add_under_cap(self.external_live_bytes, self.parsed_bytes, self.cap)?;
        let base = checked_add_under_cap(base, context_bytes, self.cap)?;
        let expected = checked_add_under_cap(base, prepared_bytes, self.cap)?;
        if self.context_bytes != context_bytes || self.live_bytes != expected {
            return Err(JpegError::InternalInvariant {
                reason: "prepared construction ledger diverged from retained allocation capacity",
            });
        }
        Ok(())
    }

    #[cfg(test)]
    fn include_retained_capacity<T>(&mut self, capacity: usize) -> Result<(), JpegError> {
        let bytes = capacity
            .checked_mul(size_of::<T>())
            .ok_or_else(|| cap_overflow(self.cap))?;
        self.live_bytes = checked_add_under_cap(self.live_bytes, bytes, self.cap)?;
        Ok(())
    }
}

fn checked_add_under_cap(left: usize, right: usize, cap: usize) -> Result<usize, JpegError> {
    let requested = left.checked_add(right).ok_or_else(|| cap_overflow(cap))?;
    if requested > cap {
        return Err(JpegError::MemoryCapExceeded { requested, cap });
    }
    Ok(requested)
}

fn cap_overflow(cap: usize) -> JpegError {
    JpegError::MemoryCapExceeded {
        requested: usize::MAX,
        cap,
    }
}

#[cfg(test)]
mod tests;
