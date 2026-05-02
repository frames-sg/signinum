// SPDX-License-Identifier: Apache-2.0

use crate::decoder::Decoder;
use crate::error::{JpegError, MarkerKind};
use crate::info::ColorSpace;
use crate::internal::checkpoint::{build_checkpoint_plan, DeviceCheckpoint};
use crate::Warning;
use alloc::borrow::Cow;
use alloc::vec::Vec;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceComponentPlan {
    pub h: u8,
    pub v: u8,
    pub output_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceDecodePlan<'a> {
    pub dimensions: (u32, u32),
    pub color_space: ColorSpace,
    pub restart_interval: Option<u16>,
    pub warnings: Vec<Warning>,
    pub scan_bytes: Cow<'a, [u8]>,
    pub components: Vec<DeviceComponentPlan>,
    pub checkpoints: Vec<DeviceCheckpoint>,
    pub matches_fast_420: bool,
    pub matches_fast_422: bool,
    pub matches_fast_444: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceBatchSummary {
    pub restart_interval: Option<u16>,
    pub checkpoint_count: usize,
    pub matches_fast_420: bool,
    pub matches_fast_422: bool,
    pub matches_fast_444: bool,
}

pub fn build_device_plan<'a>(
    decoder: &'a Decoder<'a>,
    cadence_mcus: u32,
) -> Result<DeviceDecodePlan<'a>, JpegError> {
    let plan = &decoder.plan;
    let restart_interval = plan.restart_interval.filter(|&interval| interval > 0);
    let (scan_bytes, missing_eoi) =
        scan_payload_bytes(decoder.bytes, plan.scan_offset, restart_interval.is_some())?;
    let checkpoints = build_checkpoint_plan(plan, scan_bytes.as_ref(), cadence_mcus)?;
    let mut warnings = decoder.warnings.to_vec();
    if missing_eoi {
        warnings.push(Warning::MissingEoi);
    }

    Ok(DeviceDecodePlan {
        dimensions: plan.dimensions,
        color_space: plan.color_space,
        restart_interval,
        warnings,
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
        matches_fast_422: plan.matches_fast_rgb422_shape(),
        matches_fast_444: plan.matches_fast_rgb444_shape(),
    })
}

pub fn summarize_device_batch(decoder: &Decoder<'_>, cadence_mcus: u32) -> DeviceBatchSummary {
    let plan = &decoder.plan;
    let restart_interval = plan.restart_interval.filter(|&interval| interval > 0);
    let total_mcus = total_mcus(plan);
    let cadence_mcus = cadence_mcus.max(1);
    let checkpoint_count = match restart_interval {
        Some(restart) => 1usize.saturating_add(
            total_mcus
                .saturating_sub(1)
                .checked_div(u32::from(restart))
                .unwrap_or(0) as usize,
        ),
        None => 1usize.saturating_add(
            total_mcus
                .saturating_sub(1)
                .checked_div(cadence_mcus)
                .unwrap_or(0) as usize,
        ),
    };

    DeviceBatchSummary {
        restart_interval,
        checkpoint_count,
        matches_fast_420: plan.matches_fast_tile_shape(),
        matches_fast_422: plan.matches_fast_rgb422_shape(),
        matches_fast_444: plan.matches_fast_rgb444_shape(),
    }
}

fn scan_payload_bytes(
    bytes: &[u8],
    scan_offset: usize,
    allow_restart_markers: bool,
) -> Result<(Cow<'_, [u8]>, bool), JpegError> {
    let scan = &bytes[scan_offset..];
    let mut index = 0usize;
    while index < scan.len() {
        if scan[index] != 0xff {
            index += 1;
            continue;
        }

        let marker_start = index;
        let next = index + 1;
        if next >= scan.len() {
            return Ok((Cow::Borrowed(scan), true));
        }

        match scan[next] {
            0x00 => {
                index = next + 1;
            }
            0xd0..=0xd7 if allow_restart_markers => {
                index = next + 1;
            }
            0xd9 => return Ok((Cow::Borrowed(&scan[..=next]), false)),
            found => {
                return Err(JpegError::UnexpectedMarker {
                    offset: scan_offset + marker_start,
                    expected: MarkerKind::Eoi,
                    found,
                })
            }
        }
    }

    Ok((Cow::Borrowed(scan), true))
}

fn total_mcus(plan: &crate::entropy::sequential::PreparedDecodePlan) -> u32 {
    let mcu_width = u32::from(plan.sampling.max_h) * 8;
    let mcu_height = u32::from(plan.sampling.max_v) * 8;
    let mcus_per_row = plan.dimensions.0.div_ceil(mcu_width);
    let mcu_rows = plan.dimensions.1.div_ceil(mcu_height);
    mcus_per_row.saturating_mul(mcu_rows)
}
