// SPDX-License-Identifier: MIT OR Apache-2.0

//! Progressive scan entropy traversal and coefficient refinement.

use alloc::vec::Vec;

use crate::allocation::try_reserve_for_len_with_live_budget;
use crate::entropy::ZIGZAG;
use crate::error::{HuffmanFailure, JpegError};
use crate::internal::bit_reader::BitReader;

use super::allocation::{
    allocate_coefficients, checked_phase_capacity, coefficient_capacity_bytes,
};
use super::model::{
    PreparedProgressiveComponentPlan, PreparedProgressivePlan, PreparedProgressiveScan,
    PreparedProgressiveScanComponent, ProgressiveDctBlocks,
};
use super::terminal::finish_progressive_scan;

struct ProgressiveBlockTarget<'a> {
    component: &'a PreparedProgressiveComponentPlan,
    scan_component: &'a PreparedProgressiveScanComponent,
    block_x: u32,
    block_y: u32,
}

pub(crate) fn decode_progressive_dct_blocks(
    plan: &PreparedProgressivePlan,
    bytes: &[u8],
    external_live_bytes: usize,
) -> Result<ProgressiveDctBlocks, JpegError> {
    let mut coeffs = allocate_coefficients(plan, external_live_bytes)?;
    let coefficient_live_bytes = checked_phase_capacity(
        external_live_bytes,
        coefficient_capacity_bytes(coeffs.capacity(), &coeffs)?,
        plan.scratch_bytes,
    )?;
    for scan in &plan.scans {
        decode_progressive_scan(plan, scan, bytes, &mut coeffs, coefficient_live_bytes)?;
    }
    Ok(ProgressiveDctBlocks { quantized: coeffs })
}

fn decode_progressive_scan(
    plan: &PreparedProgressivePlan,
    scan: &PreparedProgressiveScan,
    bytes: &[u8],
    coeffs: &mut [Vec<[i32; 64]>],
    coefficient_live_bytes: usize,
) -> Result<(), JpegError> {
    let scan_bytes = bytes
        .get(scan.entropy_offset..)
        .ok_or(JpegError::Truncated {
            offset: scan.entropy_offset,
            expected: 1,
        })?;
    let mut br = BitReader::new_with_eof_padding(scan_bytes, scan.terminal_code == 0);
    let mut live_bytes = coefficient_live_bytes;
    let mut dc_predictors = Vec::new();
    try_reserve_for_len_with_live_budget(
        &mut dc_predictors,
        plan.components.len(),
        &mut live_bytes,
        plan.scratch_bytes,
    )?;
    dc_predictors.resize(plan.components.len(), 0i32);
    let mut eob_run = 0u32;
    let restart = u32::from(scan.restart_interval.unwrap_or(0));
    let mut mcus_since_restart = 0u32;
    let mut expected_rst = 0u8;
    let total_mcus = scan_mcu_count(plan, scan)?;

    for mcu_index in 0..total_mcus {
        if restart > 0 && mcus_since_restart == restart {
            consume_restart(
                &mut br,
                mcu_index,
                total_mcus,
                &mut expected_rst,
                &mut dc_predictors,
                &mut eob_run,
            )?;
            mcus_since_restart = 0;
        }

        decode_progressive_mcu(
            plan,
            scan,
            &mut br,
            coeffs,
            &mut dc_predictors,
            &mut eob_run,
            mcu_index,
        )?;
        mcus_since_restart += 1;
    }

    finish_progressive_scan(&mut br, scan_bytes, scan, eob_run)
}

fn scan_mcu_count(
    plan: &PreparedProgressivePlan,
    scan: &PreparedProgressiveScan,
) -> Result<u32, JpegError> {
    let scan_components = plan.scan_components(scan)?;
    if scan_components.len() > 1 {
        Ok(plan.mcu_cols.saturating_mul(plan.mcu_rows))
    } else {
        let scan_component = scan_components
            .first()
            .ok_or(JpegError::InternalInvariant {
                reason: "prepared progressive scan has no components",
            })?;
        let component = plan.components.get(scan_component.component_index).ok_or(
            JpegError::InternalInvariant {
                reason: "prepared progressive scan references an unknown component",
            },
        )?;
        Ok(progressive_coded_block_cols(component)
            .saturating_mul(progressive_coded_block_rows(component)))
    }
}

fn consume_restart(
    br: &mut BitReader<'_>,
    mcu_index: u32,
    total_mcus: u32,
    expected_rst: &mut u8,
    dc_predictors: &mut [i32],
    eob_run: &mut u32,
) -> Result<(), JpegError> {
    *expected_rst = br.consume_restart_marker(*expected_rst, mcu_index, total_mcus)?;
    dc_predictors.fill(0);
    *eob_run = 0;
    Ok(())
}

fn decode_progressive_mcu(
    plan: &PreparedProgressivePlan,
    scan: &PreparedProgressiveScan,
    br: &mut BitReader<'_>,
    coeffs: &mut [Vec<[i32; 64]>],
    dc_predictors: &mut [i32],
    eob_run: &mut u32,
    mcu_index: u32,
) -> Result<(), JpegError> {
    let scan_components = plan.scan_components(scan)?;
    if scan_components.len() > 1 {
        let mcu_x = mcu_index % plan.mcu_cols;
        let mcu_y = mcu_index / plan.mcu_cols;
        for scan_component in scan_components {
            let component = plan.components.get(scan_component.component_index).ok_or(
                JpegError::InternalInvariant {
                    reason: "prepared progressive scan references an unknown component",
                },
            )?;
            for by in 0..u32::from(component.v) {
                for bx in 0..u32::from(component.h) {
                    let target = ProgressiveBlockTarget {
                        component,
                        scan_component,
                        block_x: mcu_x * u32::from(component.h) + bx,
                        block_y: mcu_y * u32::from(component.v) + by,
                    };
                    decode_progressive_block_at(
                        plan,
                        scan,
                        &target,
                        br,
                        coeffs,
                        dc_predictors,
                        eob_run,
                    )?;
                }
            }
        }
    } else {
        let scan_component = scan_components
            .first()
            .ok_or(JpegError::InternalInvariant {
                reason: "prepared progressive scan has no components",
            })?;
        let component = plan.components.get(scan_component.component_index).ok_or(
            JpegError::InternalInvariant {
                reason: "prepared progressive scan references an unknown component",
            },
        )?;
        let coded_cols = progressive_coded_block_cols(component);
        let target = ProgressiveBlockTarget {
            component,
            scan_component,
            block_x: mcu_index % coded_cols,
            block_y: mcu_index / coded_cols,
        };
        decode_progressive_block_at(plan, scan, &target, br, coeffs, dc_predictors, eob_run)?;
    }

    Ok(())
}

fn decode_progressive_block_at(
    plan: &PreparedProgressivePlan,
    scan: &PreparedProgressiveScan,
    target: &ProgressiveBlockTarget<'_>,
    br: &mut BitReader<'_>,
    coeffs: &mut [Vec<[i32; 64]>],
    dc_predictors: &mut [i32],
    eob_run: &mut u32,
) -> Result<(), JpegError> {
    let block_index = (target.block_y as usize)
        .checked_mul(target.component.block_cols as usize)
        .and_then(|base| base.checked_add(target.block_x as usize))
        .ok_or(JpegError::HuffmanDecode {
            mcu: 0,
            reason: HuffmanFailure::InvalidSymbol,
        })?;
    let block = coeffs
        .get_mut(target.scan_component.component_index)
        .and_then(|component_coeffs| component_coeffs.get_mut(block_index))
        .ok_or(JpegError::HuffmanDecode {
            mcu: 0,
            reason: HuffmanFailure::InvalidSymbol,
        })?;

    if scan.ah == 0 {
        decode_progressive_block_first(
            plan,
            scan,
            target.scan_component,
            br,
            block,
            &mut dc_predictors[target.scan_component.component_index],
            eob_run,
        )
    } else {
        decode_progressive_block_refine(plan, scan, target.scan_component, br, block, eob_run)
    }
}

fn decode_progressive_block_first(
    plan: &PreparedProgressivePlan,
    scan: &PreparedProgressiveScan,
    scan_component: &PreparedProgressiveScanComponent,
    br: &mut BitReader<'_>,
    block: &mut [i32; 64],
    dc_predictor: &mut i32,
    eob_run: &mut u32,
) -> Result<(), JpegError> {
    if scan.ss == 0 {
        let dc_table = plan.dc_table(scan_component.dc_table)?;
        let ssss = dc_table.decode(br)?;
        if ssss > 15 {
            return Err(invalid_symbol());
        }
        let diff = br.receive_extend(ssss)?;
        *dc_predictor = dc_predictor.wrapping_add(diff);
        block[0] = dc_predictor.wrapping_shl(u32::from(scan.al));
        return Ok(());
    }

    let ac_table = plan.ac_table(scan_component.ac_table)?;
    if *eob_run > 0 {
        *eob_run -= 1;
        return Ok(());
    }

    let mut k = scan.ss;
    while k <= scan.se {
        let symbol = ac_table.decode(br)?;
        let run = symbol >> 4;
        let ssss = symbol & 0x0F;
        if ssss == 0 {
            if run == 15 {
                k = k.saturating_add(16);
            } else {
                *eob_run = decode_eob_run(br, run)?;
                break;
            }
        } else {
            k = k.saturating_add(run);
            if k > scan.se {
                return Err(invalid_symbol());
            }
            let value = br.receive_extend(ssss)?.wrapping_shl(u32::from(scan.al));
            block[usize::from(ZIGZAG[k as usize])] = value;
            k += 1;
        }
    }

    Ok(())
}

fn progressive_coded_block_cols(component: &PreparedProgressiveComponentPlan) -> u32 {
    component
        .sample_width
        .div_ceil(8)
        .max(1)
        .min(component.block_cols)
}

fn progressive_coded_block_rows(component: &PreparedProgressiveComponentPlan) -> u32 {
    component
        .sample_height
        .div_ceil(8)
        .max(1)
        .min(component.block_rows)
}

fn decode_progressive_block_refine(
    plan: &PreparedProgressivePlan,
    scan: &PreparedProgressiveScan,
    scan_component: &PreparedProgressiveScanComponent,
    br: &mut BitReader<'_>,
    block: &mut [i32; 64],
    eob_run: &mut u32,
) -> Result<(), JpegError> {
    let bit = 1i32 << scan.al;
    if scan.ss == 0 {
        if br.read_bits(1)? != 0 {
            block[0] |= bit;
        }
        return Ok(());
    }

    let ac_table = plan.ac_table(scan_component.ac_table)?;
    if *eob_run > 0 {
        *eob_run -= 1;
        refine_non_zeroes(br, block, scan.ss, scan.se, 64, bit)?;
        return Ok(());
    }

    let mut k = scan.ss;
    while k <= scan.se {
        let symbol = ac_table.decode(br)?;
        let run = symbol >> 4;
        let ssss = symbol & 0x0F;
        let mut zero_run_length = usize::from(run);
        let mut value = 0i32;

        match ssss {
            0 => {
                if run == 15 {
                    zero_run_length = 15;
                } else {
                    *eob_run = decode_eob_run(br, run)?;
                    zero_run_length = 64;
                }
            }
            1 => {
                value = if br.read_bits(1)? != 0 { bit } else { -bit };
            }
            _ => return Err(invalid_symbol()),
        }

        k = refine_non_zeroes(br, block, k, scan.se, zero_run_length, bit)?;
        if value != 0 {
            if k > scan.se {
                return Err(invalid_symbol());
            }
            block[usize::from(ZIGZAG[k as usize])] = value;
        }
        k += 1;
    }

    Ok(())
}

pub(super) fn decode_eob_run(br: &mut BitReader<'_>, run_bits: u8) -> Result<u32, JpegError> {
    let mut eob_run = (1u32 << run_bits) - 1;
    if run_bits > 0 {
        eob_run += br.read_bits(run_bits)?;
    }
    Ok(eob_run)
}

pub(super) fn refine_non_zeroes(
    br: &mut BitReader<'_>,
    block: &mut [i32; 64],
    start: u8,
    end: u8,
    mut zero_run_length: usize,
    bit: i32,
) -> Result<u8, JpegError> {
    for k in start..=end {
        let idx = usize::from(ZIGZAG[k as usize]);
        let coeff = &mut block[idx];
        if *coeff == 0 {
            if zero_run_length == 0 {
                return Ok(k);
            }
            zero_run_length -= 1;
        } else if br.read_bits(1)? != 0 && (*coeff & bit) == 0 {
            if *coeff > 0 {
                *coeff = coeff.wrapping_add(bit);
            } else {
                *coeff = coeff.wrapping_sub(bit);
            }
        }
    }
    Ok(end)
}

fn invalid_symbol() -> JpegError {
    JpegError::HuffmanDecode {
        mcu: 0,
        reason: HuffmanFailure::InvalidSymbol,
    }
}
