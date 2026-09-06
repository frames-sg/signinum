// SPDX-License-Identifier: MIT OR Apache-2.0

//! Single-pass checkpoint entropy traversal into caller-selected final storage.

use alloc::vec::Vec;

use super::allocation::try_checkpoint_vec_with_live_budget;
#[cfg(test)]
use super::eoi::validate_scan_bytes;
use super::eoi::ValidatedScanBytes;
use super::planning::plan_checkpoint_build_from_validated;
use super::DeviceCheckpoint;
use crate::entropy::block::skip_block;
#[cfg(test)]
use crate::entropy::block::{decode_block_with_activity, CoefficientBlock};
use crate::entropy::sequential::PreparedDecodePlan;
use crate::error::{JpegError, MarkerKind};
use crate::internal::bit_reader::BitReader;

#[cfg(test)]
pub(crate) fn build_checkpoint_plan_mapped_with_live_budget<T, E>(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    cadence_mcus: u32,
    live_bytes: &mut usize,
    allocation_cap: usize,
    map_checkpoint: impl FnMut(DeviceCheckpoint) -> Result<T, E>,
) -> Result<Vec<T>, E>
where
    E: From<JpegError>,
{
    let restart_markers = plan.restart_interval.is_some_and(|interval| interval > 0);
    let validated_scan = validate_scan_bytes(scan_bytes, restart_markers, 0).map_err(E::from)?;
    build_checkpoint_plan_mapped_from_validated_with_live_budget(
        plan,
        validated_scan,
        cadence_mcus,
        live_bytes,
        allocation_cap,
        map_checkpoint,
    )
}

pub(crate) fn build_checkpoint_plan_mapped_from_validated_with_live_budget<T, E>(
    plan: &PreparedDecodePlan,
    validated_scan: ValidatedScanBytes<'_>,
    cadence_mcus: u32,
    live_bytes: &mut usize,
    allocation_cap: usize,
    mut map_checkpoint: impl FnMut(DeviceCheckpoint) -> Result<T, E>,
) -> Result<Vec<T>, E>
where
    E: From<JpegError>,
{
    let build = plan_checkpoint_build_from_validated::<T>(
        plan,
        validated_scan,
        cadence_mcus,
        *live_bytes,
        allocation_cap,
    )
    .map_err(E::from)?;
    let mut checkpoints = try_checkpoint_vec_with_live_budget(
        build.expected_checkpoint_count,
        live_bytes,
        allocation_cap,
    )
    .map_err(E::from)?;
    let reader_bytes = build
        .validated_scan
        .terminated_with_live_budget(*live_bytes, allocation_cap)
        .map_err(E::from)?;
    let mut br = BitReader::new(reader_bytes.as_ref());
    let mut prev_dc = [0i32; 4];
    let mut expected_rst = 0u8;
    let mut mcus_since_restart = 0u32;

    push_planned_checkpoint(
        &mut checkpoints,
        map_checkpoint(snapshot_checkpoint(0, &br, prev_dc, expected_rst))?,
        build.expected_checkpoint_count,
    )
    .map_err(E::from)?;

    for mcu_index in 0..build.total_mcus {
        if mcu_index > 0 {
            if let Some(restart) = build.restart_interval {
                if mcus_since_restart == restart {
                    expected_rst = br
                        .consume_restart_marker(expected_rst, mcu_index, build.total_mcus)
                        .map_err(E::from)?;
                    prev_dc.fill(0);
                    mcus_since_restart = 0;
                    push_planned_checkpoint(
                        &mut checkpoints,
                        map_checkpoint(snapshot_checkpoint(mcu_index, &br, prev_dc, expected_rst))?,
                        build.expected_checkpoint_count,
                    )
                    .map_err(E::from)?;
                }
            } else if mcu_index.is_multiple_of(build.cadence_mcus) {
                push_planned_checkpoint(
                    &mut checkpoints,
                    map_checkpoint(snapshot_checkpoint(mcu_index, &br, prev_dc, expected_rst))?,
                    build.expected_checkpoint_count,
                )
                .map_err(E::from)?;
            }
        }

        skip_one_mcu(plan, &mut br, &mut prev_dc).map_err(E::from)?;
        mcus_since_restart += 1;
    }

    match br.take_marker() {
        Some(0xd9) | None => {}
        Some(found) => {
            return Err(E::from(JpegError::UnexpectedMarker {
                offset: br.position().saturating_sub(2),
                expected: MarkerKind::Eoi,
                found,
            }));
        }
    }

    if checkpoints.len() != build.expected_checkpoint_count {
        return Err(E::from(JpegError::InternalInvariant {
            reason: "checkpoint materialization disagrees with its allocation plan",
        }));
    }
    Ok(checkpoints)
}

fn push_planned_checkpoint<T>(
    checkpoints: &mut Vec<T>,
    checkpoint: T,
    expected_checkpoint_count: usize,
) -> Result<(), JpegError> {
    if checkpoints.len() >= expected_checkpoint_count {
        return Err(JpegError::InternalInvariant {
            reason: "checkpoint materialization exceeded its allocation plan",
        });
    }
    checkpoints.push(checkpoint);
    Ok(())
}

pub(super) fn snapshot_checkpoint(
    mcu_index: u32,
    br: &BitReader<'_>,
    prev_dc: [i32; 4],
    expected_rst: u8,
) -> DeviceCheckpoint {
    let snapshot = br.snapshot();
    DeviceCheckpoint {
        mcu_index,
        scan_offset: snapshot.pos,
        bit_accumulator: snapshot.acc,
        bits_buffered: snapshot.bits,
        prev_dc,
        expected_rst,
    }
}

pub(super) fn skip_one_mcu(
    plan: &PreparedDecodePlan,
    br: &mut BitReader<'_>,
    prev_dc: &mut [i32; 4],
) -> Result<(), JpegError> {
    // Checkpoints retain only the entropy position and DC predictors. `skip_block`
    // performs the same Huffman validation and bit consumption without writing
    // dequantized coefficients that this traversal would immediately discard.
    for component in &plan.components {
        let plane_index = component.output_index;
        let dc_table = plan.dc_table(component)?;
        let ac_table = plan.ac_table(component)?;
        for _ in 0..u32::from(component.h) * u32::from(component.v) {
            skip_block(br, dc_table, ac_table, &mut prev_dc[plane_index])?;
        }
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn decode_one_mcu(
    plan: &PreparedDecodePlan,
    br: &mut BitReader<'_>,
    coeff: &mut CoefficientBlock,
    prev_dc: &mut [i32; 4],
) -> Result<(), JpegError> {
    for component in &plan.components {
        let plane_index = component.output_index;
        let dc_table = plan.dc_table(component)?;
        let ac_table = plan.ac_table(component)?;
        for _ in 0..u32::from(component.h) * u32::from(component.v) {
            let _ = decode_block_with_activity(
                br,
                dc_table,
                ac_table,
                &mut prev_dc[plane_index],
                &component.quant,
                coeff,
            )?;
        }
    }
    Ok(())
}
