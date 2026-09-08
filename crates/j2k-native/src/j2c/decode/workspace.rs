// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reusable decoder workspace ownership, policy, and diagnostics.

use super::{DecompositionStorage, OutputRegion, TileDecodeContext};
#[cfg(feature = "parallel")]
use crate::{HtCodeBlockDecodeWorkspace, J2kCodeBlockDecodeWorkspace};
#[cfg(feature = "parallel")]
use alloc::vec::Vec;

#[cfg(feature = "parallel")]
pub(in crate::j2c::decode) struct ClassicTaskWorkspace {
    pub(in crate::j2c::decode) workspace: J2kCodeBlockDecodeWorkspace,
    // Componentwise maxima make orthogonal edge shapes reusable without growth.
    pub(in crate::j2c::decode) prepared_width: u32,
    pub(in crate::j2c::decode) prepared_height: u32,
}

#[cfg(feature = "parallel")]
pub(in crate::j2c::decode) struct HtTaskWorkspace {
    pub(in crate::j2c::decode) workspace: HtCodeBlockDecodeWorkspace,
    // Componentwise maxima make orthogonal edge shapes reusable without growth.
    pub(in crate::j2c::decode) prepared_width: u32,
    pub(in crate::j2c::decode) prepared_height: u32,
}

#[cfg(feature = "parallel")]
#[derive(Default)]
pub(in crate::j2c::decode) struct ParallelTaskWorkspaces {
    pub(in crate::j2c::decode) classic: Vec<ClassicTaskWorkspace>,
    pub(in crate::j2c::decode) ht: Vec<HtTaskWorkspace>,
}

#[cfg(feature = "parallel")]
pub(in crate::j2c::decode) fn classic_task_workspace_bytes(
    slots: &Vec<ClassicTaskWorkspace>,
) -> crate::Result<usize> {
    let mut bytes = slots
        .capacity()
        .checked_mul(core::mem::size_of::<ClassicTaskWorkspace>())
        .ok_or(crate::ValidationError::ImageTooLarge)?;
    for slot in slots {
        bytes = bytes
            .checked_add(slot.workspace.allocated_bytes()?)
            .ok_or(crate::ValidationError::ImageTooLarge)?;
    }
    Ok(bytes)
}

#[cfg(feature = "parallel")]
pub(in crate::j2c::decode) fn ht_task_workspace_bytes(
    slots: &Vec<HtTaskWorkspace>,
) -> crate::Result<usize> {
    let mut bytes = slots
        .capacity()
        .checked_mul(core::mem::size_of::<HtTaskWorkspace>())
        .ok_or(crate::ValidationError::ImageTooLarge)?;
    for slot in slots {
        bytes = bytes
            .checked_add(slot.workspace.allocated_bytes()?)
            .ok_or(crate::ValidationError::ImageTooLarge)?;
    }
    Ok(bytes)
}

/// CPU parallelism policy for native JPEG 2000 decode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CpuDecodeParallelism {
    /// Allow a single tile decode to use internal code-block parallelism.
    #[default]
    Auto,
    /// Keep code-block decode serial for callers that already parallelize tiles.
    Serial,
}

/// Observable counters and retained ownership for a reusable decoder workspace.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecoderWorkspaceStats {
    decode_calls: u64,
    component_owner_reuses: u64,
    tier1_owner_reuses: u64,
    idwt_owner_reuses: u64,
    scratch_capacity_retries: u64,
    retained_component_bytes: usize,
    retained_tier1_bytes: usize,
    retained_idwt_bytes: usize,
}

impl DecoderWorkspaceStats {
    /// Number of native image decode calls made with this workspace.
    #[must_use]
    pub const fn decode_calls(self) -> u64 {
        self.decode_calls
    }

    /// Number of calls that began with reusable decoded-component owners.
    #[must_use]
    pub const fn component_owner_reuses(self) -> u64 {
        self.component_owner_reuses
    }

    /// Number of calls that began with reusable Tier-1 allocations.
    #[must_use]
    pub const fn tier1_owner_reuses(self) -> u64 {
        self.tier1_owner_reuses
    }

    /// Number of calls that began with reusable IDWT allocations.
    #[must_use]
    pub const fn idwt_owner_reuses(self) -> u64 {
        self.idwt_owner_reuses
    }

    /// Number of retained-scratch evictions followed by a fresh retry.
    #[must_use]
    pub const fn scratch_capacity_retries(self) -> u64 {
        self.scratch_capacity_retries
    }

    /// Retained component-owner capacity after the most recent completed core decode.
    #[must_use]
    pub const fn retained_component_bytes(self) -> usize {
        self.retained_component_bytes
    }

    /// Retained classic and HT Tier-1 capacity after the most recent call.
    #[must_use]
    pub const fn retained_tier1_bytes(self) -> usize {
        self.retained_tier1_bytes
    }

    /// Retained floating-point and exact-integer IDWT capacity after the most recent call.
    #[must_use]
    pub const fn retained_idwt_bytes(self) -> usize {
        self.retained_idwt_bytes
    }

    /// Total retained lifetime-free decode scratch after the most recent call.
    #[must_use]
    pub const fn retained_scratch_bytes(self) -> usize {
        self.retained_tier1_bytes
            .saturating_add(self.retained_idwt_bytes)
    }
}

/// Lifetime-free allocation owner that can be moved between borrowing decoder contexts.
///
/// Parsed packet and tile graphs remain in [`DecoderContext`] and are always
/// released before this value is recovered. Decoded component, Tier-1, and
/// IDWT allocations are retained for reuse with unrelated encoded inputs.
#[derive(Default)]
pub struct DecoderWorkspace {
    tile_decode_context: TileDecodeContext,
    pub(super) cpu_decode_parallelism: CpuDecodeParallelism,
    stats: DecoderWorkspaceStats,
}

impl DecoderWorkspace {
    /// Return reuse counters and retained allocation sizes.
    #[must_use]
    pub const fn stats(&self) -> DecoderWorkspaceStats {
        self.stats
    }
}

/// A decoder context for decoding JPEG2000 images.
pub struct DecoderContext<'a> {
    pub(crate) tile_decode_context: TileDecodeContext,
    pub(crate) storage: DecompositionStorage<'a>,
    pub(super) cpu_decode_parallelism: CpuDecodeParallelism,
    workspace_stats: DecoderWorkspaceStats,
}

impl Default for DecoderContext<'_> {
    fn default() -> Self {
        Self {
            tile_decode_context: TileDecodeContext::default(),
            storage: DecompositionStorage::default(),
            cpu_decode_parallelism: CpuDecodeParallelism::Auto,
            workspace_stats: DecoderWorkspaceStats::default(),
        }
    }
}

impl DecoderContext<'_> {
    /// Create a borrowing decoder context from a lifetime-free reusable workspace.
    #[must_use]
    pub fn from_workspace(workspace: DecoderWorkspace) -> Self {
        Self {
            tile_decode_context: workspace.tile_decode_context,
            storage: DecompositionStorage::default(),
            cpu_decode_parallelism: workspace.cpu_decode_parallelism,
            workspace_stats: workspace.stats,
        }
    }

    /// Release input-borrowing graph owners and recover the reusable workspace.
    #[must_use]
    pub fn into_workspace(mut self) -> DecoderWorkspace {
        self.storage.release_all_allocations();
        DecoderWorkspace {
            tile_decode_context: self.tile_decode_context,
            cpu_decode_parallelism: self.cpu_decode_parallelism,
            stats: self.workspace_stats,
        }
    }

    /// Return reuse counters for this context's lifetime-free workspace state.
    #[must_use]
    pub const fn workspace_stats(&self) -> DecoderWorkspaceStats {
        self.workspace_stats
    }

    pub(super) fn record_decode_start(
        &mut self,
        retained_component_bytes: usize,
        retained_tier1_bytes: usize,
        retained_idwt_bytes: usize,
    ) {
        self.workspace_stats.decode_calls = self.workspace_stats.decode_calls.saturating_add(1);
        if retained_component_bytes != 0 {
            self.workspace_stats.component_owner_reuses = self
                .workspace_stats
                .component_owner_reuses
                .saturating_add(1);
        }
        if retained_tier1_bytes != 0 {
            self.workspace_stats.tier1_owner_reuses =
                self.workspace_stats.tier1_owner_reuses.saturating_add(1);
        }
        if retained_idwt_bytes != 0 {
            self.workspace_stats.idwt_owner_reuses =
                self.workspace_stats.idwt_owner_reuses.saturating_add(1);
        }
    }

    pub(super) fn record_scratch_capacity_retry(&mut self) {
        self.workspace_stats.scratch_capacity_retries = self
            .workspace_stats
            .scratch_capacity_retries
            .saturating_add(1);
    }

    pub(super) fn record_decode_complete(
        &mut self,
        retained_component_bytes: usize,
        retained_tier1_bytes: usize,
        retained_idwt_bytes: usize,
    ) {
        self.workspace_stats.retained_component_bytes = retained_component_bytes;
        self.workspace_stats.retained_tier1_bytes = retained_tier1_bytes;
        self.workspace_stats.retained_idwt_bytes = retained_idwt_bytes;
    }

    pub(crate) fn release_reusable_allocations(&mut self) {
        self.tile_decode_context.release_all_allocations();
        self.storage.release_all_allocations();
    }

    pub(crate) fn set_output_region(&mut self, output_region: Option<(u32, u32, u32, u32)>) {
        self.tile_decode_context.output_region = output_region.map(OutputRegion::from_tuple);
    }

    pub(crate) fn set_round_irreversible_output(&mut self, enabled: bool) {
        self.tile_decode_context.round_irreversible_output = enabled;
    }

    /// Return the native CPU decode parallelism policy.
    #[must_use]
    pub fn cpu_decode_parallelism(&self) -> CpuDecodeParallelism {
        self.cpu_decode_parallelism
    }

    /// Set the native CPU decode parallelism policy.
    pub fn set_cpu_decode_parallelism(&mut self, parallelism: CpuDecodeParallelism) {
        self.cpu_decode_parallelism = parallelism;
    }
}
