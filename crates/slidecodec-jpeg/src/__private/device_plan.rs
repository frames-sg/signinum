// SPDX-License-Identifier: Apache-2.0

use crate::decoder::Decoder;
use crate::entropy::sequential::PreparedDecodePlan;
use crate::error::{JpegError, MarkerKind};
use crate::info::ColorSpace;
use alloc::vec::Vec;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceComponentPlan {
    pub h: u8,
    pub v: u8,
    pub output_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCheckpoint {
    pub mcu_index: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceDecodePlan {
    pub dimensions: (u32, u32),
    pub color_space: ColorSpace,
    pub restart_interval: Option<u16>,
    pub scan_bytes: Vec<u8>,
    pub components: Vec<DeviceComponentPlan>,
    pub checkpoints: Vec<DeviceCheckpoint>,
    pub matches_fast_420: bool,
    pub matches_fast_444: bool,
}

pub fn build_device_plan(
    decoder: &Decoder<'_>,
    cadence_mcus: u32,
) -> Result<DeviceDecodePlan, JpegError> {
    let plan = &decoder.plan;
    let scan_bytes = scan_payload_bytes(decoder.bytes, plan.scan_offset)?;
    let checkpoints = build_checkpoint_plan(plan, &scan_bytes, cadence_mcus);

    Ok(DeviceDecodePlan {
        dimensions: plan.dimensions,
        color_space: plan.color_space,
        restart_interval: plan.restart_interval,
        scan_bytes,
        components: plan
            .components
            .iter()
            .map(|component| DeviceComponentPlan {
                h: component.h,
                v: component.v,
                output_index: component.output_index,
            })
            .collect(),
        checkpoints,
        matches_fast_420: plan.matches_fast_tile_shape(),
        matches_fast_444: plan.matches_fast_rgb444_shape(),
    })
}

fn scan_payload_bytes(bytes: &[u8], scan_offset: usize) -> Result<Vec<u8>, JpegError> {
    let scan = &bytes[scan_offset..];
    let mut index = 0usize;
    while index < scan.len() {
        if scan[index] != 0xff {
            index += 1;
            continue;
        }

        let marker_start = index;
        let mut next = index + 1;
        while next < scan.len() && scan[next] == 0xff {
            next += 1;
        }
        if next >= scan.len() {
            return Err(JpegError::MissingMarker {
                marker: MarkerKind::Eoi,
            });
        }

        match scan[next] {
            0x00 | 0xd0..=0xd7 => {
                index = next + 1;
            }
            0xd9 => return Ok(scan[..marker_start].to_vec()),
            found => {
                return Err(JpegError::UnexpectedMarker {
                    offset: scan_offset + marker_start,
                    expected: MarkerKind::Eoi,
                    found,
                })
            }
        }
    }

    Err(JpegError::MissingMarker {
        marker: MarkerKind::Eoi,
    })
}

fn build_checkpoint_plan(
    plan: &PreparedDecodePlan,
    _scan_bytes: &[u8],
    cadence_mcus: u32,
) -> Vec<DeviceCheckpoint> {
    let cadence_mcus = cadence_mcus.max(1);
    let mcu_width = u32::from(plan.sampling.max_h) * 8;
    let mcu_height = u32::from(plan.sampling.max_v) * 8;
    let mcus_per_row = plan.dimensions.0.div_ceil(mcu_width);
    let mcu_rows = plan.dimensions.1.div_ceil(mcu_height);
    let total_mcus = mcus_per_row.saturating_mul(mcu_rows);

    let mut checkpoints = Vec::new();
    let mut mcu_index = 0u32;
    while mcu_index < total_mcus {
        checkpoints.push(DeviceCheckpoint { mcu_index });
        mcu_index = mcu_index.saturating_add(cadence_mcus);
        if mcu_index == 0 {
            break;
        }
    }

    if checkpoints.is_empty() {
        checkpoints.push(DeviceCheckpoint { mcu_index: 0 });
    }

    checkpoints
}
