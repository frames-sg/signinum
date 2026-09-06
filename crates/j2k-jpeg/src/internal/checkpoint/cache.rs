// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lazily extended entropy checkpoints used by CPU region decoding.

use alloc::vec::Vec;

use super::allocation::checkpoint_allocation_bytes;
use super::build::{skip_one_mcu, snapshot_checkpoint};
use super::{total_mcus, DeviceCheckpoint};
use crate::entropy::sequential::PreparedDecodePlan;
use crate::error::JpegError;
use crate::internal::bit_reader::{BitReader, BitReaderSnapshot};
use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

mod allocation;

pub(super) use self::allocation::reserve_checkpoint_capacity;
#[cfg(test)]
pub(super) use self::allocation::{
    checked_checkpoint_reservation_peak, reconcile_actual_checkpoint_capacity,
};

#[derive(Debug, Default)]
pub(crate) struct CpuCheckpointCache {
    pub(crate) checkpoints: Vec<DeviceCheckpoint>,
}

impl CpuCheckpointCache {
    pub(crate) fn retained_allocation_bytes(&self) -> Result<usize, JpegError> {
        checkpoint_allocation_bytes(
            self.checkpoints.capacity(),
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )
    }
}

pub(crate) fn checkpoint_before_mcu(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    cadence_mcus: u32,
    target_mcu: u32,
    retained_decoder_baseline_bytes: usize,
    cache: &mut CpuCheckpointCache,
) -> Result<Option<DeviceCheckpoint>, JpegError> {
    if plan.restart_interval.is_some() || target_mcu == 0 {
        return Ok(None);
    }

    let total_mcus = total_mcus(plan);
    let target_mcu = target_mcu.min(total_mcus);
    let cadence_mcus = cadence_mcus.max(1);
    let target_checkpoint_mcu = (target_mcu / cadence_mcus) * cadence_mcus;
    if target_checkpoint_mcu == 0 {
        return Ok(None);
    }

    let required_capacity = usize::try_from(target_checkpoint_mcu / cadence_mcus)
        .ok()
        .and_then(|checkpoint_count| checkpoint_count.checked_add(1))
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })?;
    reserve_checkpoint_capacity(
        &mut cache.checkpoints,
        required_capacity,
        retained_decoder_baseline_bytes,
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    )?;

    if cache.checkpoints.is_empty() {
        cache.checkpoints.push(snapshot_checkpoint(
            0,
            &BitReader::new(scan_bytes),
            [0; 4],
            0,
        ));
    }

    let last_mcu = cache
        .checkpoints
        .last()
        .map_or(0, |checkpoint| checkpoint.mcu_index);
    if last_mcu < target_checkpoint_mcu {
        extend_non_restart_checkpoints(
            plan,
            scan_bytes,
            cadence_mcus,
            target_checkpoint_mcu,
            cache,
        )?;
    }

    Ok(cache
        .checkpoints
        .iter()
        .rev()
        .find(|checkpoint| checkpoint.mcu_index > 0 && checkpoint.mcu_index <= target_mcu)
        .copied())
}

fn extend_non_restart_checkpoints(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    cadence_mcus: u32,
    target_checkpoint_mcu: u32,
    cache: &mut CpuCheckpointCache,
) -> Result<(), JpegError> {
    let start = cache
        .checkpoints
        .last()
        .copied()
        .unwrap_or_else(|| snapshot_checkpoint(0, &BitReader::new(scan_bytes), [0; 4], 0));
    let mut br = BitReader::from_snapshot(
        scan_bytes,
        BitReaderSnapshot {
            pos: start.scan_offset,
            acc: start.bit_accumulator,
            bits: start.bits_buffered,
        },
    );
    let mut prev_dc = start.prev_dc;
    let mut mcu_index = start.mcu_index;

    while mcu_index < target_checkpoint_mcu {
        skip_one_mcu(plan, &mut br, &mut prev_dc)?;
        mcu_index += 1;
        if mcu_index.is_multiple_of(cadence_mcus) {
            cache
                .checkpoints
                .push(snapshot_checkpoint(mcu_index, &br, prev_dc, 0));
        }
    }
    Ok(())
}
