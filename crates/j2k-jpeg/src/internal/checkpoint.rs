// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG entropy checkpoint facade and reusable lazy checkpoint cache.

use alloc::vec::Vec;

use crate::entropy::sequential::PreparedDecodePlan;
use crate::error::JpegError;
#[cfg(test)]
use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

mod allocation;
mod build;
mod cache;
mod eoi;
mod planning;

pub(crate) use self::build::build_checkpoint_plan_mapped_from_validated_with_live_budget;
#[cfg(test)]
use self::build::build_checkpoint_plan_mapped_with_live_budget;
pub(crate) use self::cache::{checkpoint_before_mcu, CpuCheckpointCache};
pub(crate) use self::eoi::{validate_scan_bytes, ValidatedScanBytes};
pub(crate) use self::planning::{checkpoint_count_summary, total_mcus};

#[cfg(test)]
use self::allocation::{
    checked_actual_checkpoint_live_bytes, checked_checkpoint_workspace_bytes,
    host_allocation_error, try_checkpoint_vec,
};
#[cfg(test)]
use self::build::{decode_one_mcu, skip_one_mcu, snapshot_checkpoint};
#[cfg(test)]
use self::cache::{
    checked_checkpoint_reservation_peak, reconcile_actual_checkpoint_capacity,
    reserve_checkpoint_capacity,
};
#[cfg(test)]
use self::eoi::{terminated_scan_bytes, terminated_scan_bytes_with_cap};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
/// Entropy decoder resume point for device-side JPEG decode.
pub struct DeviceCheckpoint {
    /// MCU index where decoding can resume.
    pub mcu_index: u32,
    /// Byte offset into scan entropy data.
    pub scan_offset: usize,
    /// Buffered entropy bits at the checkpoint.
    pub bit_accumulator: u64,
    /// Number of valid bits in `bit_accumulator`.
    pub bits_buffered: u8,
    /// Previous DC predictor per component.
    pub prev_dc: [i32; 4],
    /// Next expected restart marker index.
    pub expected_rst: u8,
}

#[cfg(test)]
pub(crate) fn build_checkpoint_plan(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    cadence_mcus: u32,
) -> Result<Vec<DeviceCheckpoint>, JpegError> {
    let mut live_bytes = 0;
    build_checkpoint_plan_with_live_budget(
        plan,
        scan_bytes,
        cadence_mcus,
        &mut live_bytes,
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    )
}

#[cfg(test)]
pub(crate) fn build_checkpoint_plan_with_cap(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    cadence_mcus: u32,
    allocation_cap: usize,
) -> Result<Vec<DeviceCheckpoint>, JpegError> {
    let mut live_bytes = 0;
    build_checkpoint_plan_with_live_budget(
        plan,
        scan_bytes,
        cadence_mcus,
        &mut live_bytes,
        allocation_cap,
    )
}

#[cfg(test)]
fn build_checkpoint_plan_with_live_budget(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    cadence_mcus: u32,
    live_bytes: &mut usize,
    allocation_cap: usize,
) -> Result<Vec<DeviceCheckpoint>, JpegError> {
    build_checkpoint_plan_mapped_with_live_budget(
        plan,
        scan_bytes,
        cadence_mcus,
        live_bytes,
        allocation_cap,
        Ok,
    )
}

pub(crate) fn build_checkpoint_plan_from_validated_with_live_budget(
    plan: &PreparedDecodePlan,
    validated_scan: ValidatedScanBytes<'_>,
    cadence_mcus: u32,
    live_bytes: &mut usize,
    allocation_cap: usize,
) -> Result<Vec<DeviceCheckpoint>, JpegError> {
    build_checkpoint_plan_mapped_from_validated_with_live_budget(
        plan,
        validated_scan,
        cadence_mcus,
        live_bytes,
        allocation_cap,
        Ok,
    )
}

#[cfg(test)]
mod tests;
