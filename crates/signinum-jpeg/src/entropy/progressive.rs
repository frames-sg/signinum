// SPDX-License-Identifier: Apache-2.0

//! Progressive 8-bit Huffman scan decoder.

use crate::backend::Backend;
use crate::color::upsample::{upsample_h2v1_fancy_row, upsample_h2v2_fancy_row};
use crate::entropy::huffman::HuffmanTable;
use crate::entropy::ZIGZAG;
use crate::error::{HuffmanFailure, JpegError, Warning};
use crate::info::{ColorSpace, SamplingFactors};
use crate::internal::bit_reader::BitReader;
use crate::output::OutputWriter;
use alloc::sync::Arc;
use alloc::vec::Vec;

#[derive(Debug, Clone)]
pub(crate) struct PreparedProgressiveComponentPlan {
    pub(crate) h: u8,
    pub(crate) v: u8,
    pub(crate) output_index: usize,
    pub(crate) quant: Arc<[u16; 64]>,
    pub(crate) block_cols: u32,
    pub(crate) block_rows: u32,
    pub(crate) sample_width: u32,
    pub(crate) sample_height: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedProgressiveScanComponent {
    pub(crate) component_index: usize,
    pub(crate) dc_table: Option<Arc<HuffmanTable>>,
    pub(crate) ac_table: Option<Arc<HuffmanTable>>,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedProgressiveScan {
    pub(crate) components: Vec<PreparedProgressiveScanComponent>,
    pub(crate) ss: u8,
    pub(crate) se: u8,
    pub(crate) ah: u8,
    pub(crate) al: u8,
    pub(crate) entropy_offset: usize,
    pub(crate) restart_interval: Option<u16>,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedProgressivePlan {
    pub(crate) components: Vec<PreparedProgressiveComponentPlan>,
    pub(crate) scans: Vec<PreparedProgressiveScan>,
    pub(crate) sampling: SamplingFactors,
    pub(crate) color_space: ColorSpace,
    pub(crate) dimensions: (u32, u32),
    pub(crate) mcu_cols: u32,
    pub(crate) mcu_rows: u32,
    pub(crate) scratch_bytes: usize,
}

#[derive(Debug)]
struct ComponentImage {
    plane: Vec<u8>,
    stride: usize,
}

pub(crate) fn decode_progressive<W: OutputWriter>(
    plan: &PreparedProgressivePlan,
    backend: Backend,
    bytes: &[u8],
    writer: &mut W,
) -> Result<Vec<Warning>, JpegError> {
    let mut coeffs = allocate_coefficients(plan)?;
    for scan in &plan.scans {
        decode_progressive_scan(plan, scan, bytes, &mut coeffs)?;
    }
    let images = render_component_images(plan, backend, &coeffs)?;
    emit_component_images(plan, &images, writer)?;
    Ok(Vec::new())
}

fn allocate_coefficients(plan: &PreparedProgressivePlan) -> Result<Vec<Vec<[i32; 64]>>, JpegError> {
    let mut coeffs = Vec::with_capacity(plan.components.len());
    for component in &plan.components {
        let blocks = (component.block_cols as usize)
            .checked_mul(component.block_rows as usize)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: usize::MAX,
            })?;
        coeffs.push(vec![[0i32; 64]; blocks]);
    }
    Ok(coeffs)
}

fn decode_progressive_scan(
    plan: &PreparedProgressivePlan,
    scan: &PreparedProgressiveScan,
    bytes: &[u8],
    coeffs: &mut [Vec<[i32; 64]>],
) -> Result<(), JpegError> {
    let scan_bytes = bytes
        .get(scan.entropy_offset..)
        .ok_or(JpegError::Truncated {
            offset: scan.entropy_offset,
            expected: 1,
        })?;
    let mut br = BitReader::new(scan_bytes);
    let mut dc_predictors = vec![0i32; plan.components.len()];
    let mut eob_run = 0u32;
    let restart = u32::from(scan.restart_interval.unwrap_or(0));
    let mut mcus_since_restart = 0u32;
    let mut expected_rst = 0u8;
    let total_mcus = scan_mcu_count(plan, scan);

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

    Ok(())
}

fn scan_mcu_count(plan: &PreparedProgressivePlan, scan: &PreparedProgressiveScan) -> u32 {
    if scan.components.len() > 1 {
        plan.mcu_cols.saturating_mul(plan.mcu_rows)
    } else {
        let component = &plan.components[scan.components[0].component_index];
        progressive_coded_block_cols(component)
            .saturating_mul(progressive_coded_block_rows(component))
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
    let _ = br.ensure_bits(1);
    let marker = br.take_marker().ok_or(JpegError::UnexpectedEoi {
        mcu_at: mcu_index,
        mcu_total: total_mcus,
    })?;
    let expected = 0xD0 | *expected_rst;
    if marker != expected {
        return Err(JpegError::RestartMismatch {
            offset: br.position(),
            expected: *expected_rst,
            found: marker,
        });
    }
    *expected_rst = (*expected_rst + 1) & 0x07;
    br.reset_at_restart();
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
    if scan.components.len() > 1 {
        let mcu_x = mcu_index % plan.mcu_cols;
        let mcu_y = mcu_index / plan.mcu_cols;
        for scan_component in &scan.components {
            let component = &plan.components[scan_component.component_index];
            for by in 0..u32::from(component.v) {
                for bx in 0..u32::from(component.h) {
                    let block_x = mcu_x * u32::from(component.h) + bx;
                    let block_y = mcu_y * u32::from(component.v) + by;
                    decode_progressive_block_at(
                        component,
                        scan,
                        scan_component,
                        br,
                        coeffs,
                        dc_predictors,
                        eob_run,
                        block_x,
                        block_y,
                    )?;
                }
            }
        }
    } else {
        let scan_component = &scan.components[0];
        let component = &plan.components[scan_component.component_index];
        let coded_cols = progressive_coded_block_cols(component);
        let block_x = mcu_index % coded_cols;
        let block_y = mcu_index / coded_cols;
        decode_progressive_block_at(
            component,
            scan,
            scan_component,
            br,
            coeffs,
            dc_predictors,
            eob_run,
            block_x,
            block_y,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_progressive_block_at(
    component: &PreparedProgressiveComponentPlan,
    scan: &PreparedProgressiveScan,
    scan_component: &PreparedProgressiveScanComponent,
    br: &mut BitReader<'_>,
    coeffs: &mut [Vec<[i32; 64]>],
    dc_predictors: &mut [i32],
    eob_run: &mut u32,
    block_x: u32,
    block_y: u32,
) -> Result<(), JpegError> {
    let block_index = (block_y as usize)
        .checked_mul(component.block_cols as usize)
        .and_then(|base| base.checked_add(block_x as usize))
        .ok_or(JpegError::HuffmanDecode {
            mcu: 0,
            reason: HuffmanFailure::InvalidSymbol,
        })?;
    let block = coeffs
        .get_mut(scan_component.component_index)
        .and_then(|component_coeffs| component_coeffs.get_mut(block_index))
        .ok_or(JpegError::HuffmanDecode {
            mcu: 0,
            reason: HuffmanFailure::InvalidSymbol,
        })?;

    if scan.ah == 0 {
        decode_progressive_block_first(
            scan,
            scan_component,
            br,
            block,
            &mut dc_predictors[scan_component.component_index],
            eob_run,
        )
    } else {
        decode_progressive_block_refine(scan, scan_component, br, block, eob_run)
    }
}

fn decode_progressive_block_first(
    scan: &PreparedProgressiveScan,
    scan_component: &PreparedProgressiveScanComponent,
    br: &mut BitReader<'_>,
    block: &mut [i32; 64],
    dc_predictor: &mut i32,
    eob_run: &mut u32,
) -> Result<(), JpegError> {
    if scan.ss == 0 {
        let dc_table = scan_component
            .dc_table
            .as_ref()
            .ok_or_else(missing_dc_table)?;
        let ssss = dc_table.decode(br)?;
        if ssss > 15 {
            return Err(invalid_symbol());
        }
        let diff = br.receive_extend(ssss)?;
        *dc_predictor = dc_predictor.wrapping_add(diff);
        block[0] = dc_predictor.wrapping_shl(u32::from(scan.al));
        return Ok(());
    }

    let ac_table = scan_component
        .ac_table
        .as_ref()
        .ok_or_else(missing_ac_table)?;
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

    let ac_table = scan_component
        .ac_table
        .as_ref()
        .ok_or_else(missing_ac_table)?;
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

fn decode_eob_run(br: &mut BitReader<'_>, run_bits: u8) -> Result<u32, JpegError> {
    let mut eob_run = (1u32 << run_bits) - 1;
    if run_bits > 0 {
        eob_run += br.read_bits(run_bits)?;
    }
    Ok(eob_run)
}

fn refine_non_zeroes(
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

fn render_component_images(
    plan: &PreparedProgressivePlan,
    backend: Backend,
    coeffs: &[Vec<[i32; 64]>],
) -> Result<Vec<ComponentImage>, JpegError> {
    let mut images = Vec::with_capacity(plan.components.len());
    for (component, component_coeffs) in plan.components.iter().zip(coeffs.iter()) {
        let stride = component.block_cols as usize * 8;
        let rows = component.block_rows as usize * 8;
        let mut plane = vec![0u8; stride * rows];
        let mut dequant = [0i16; 64];
        let mut pixels = [0u8; 64];
        for by in 0..component.block_rows as usize {
            for bx in 0..component.block_cols as usize {
                let block_index = by * component.block_cols as usize + bx;
                dequantize_block(
                    &component_coeffs[block_index],
                    &component.quant,
                    &mut dequant,
                );
                backend.idct(&dequant, &mut pixels);
                deposit_block(&mut plane, stride, bx * 8, by * 8, &pixels);
            }
        }
        images.push(ComponentImage { plane, stride });
    }
    Ok(images)
}

fn dequantize_block(coeffs: &[i32; 64], quant: &[u16; 64], out: &mut [i16; 64]) {
    out.fill(0);
    for k in 0..64 {
        let natural_idx = usize::from(ZIGZAG[k]);
        let value = coeffs[natural_idx].wrapping_mul(i32::from(quant[k]));
        out[natural_idx] = value.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

fn deposit_block(plane: &mut [u8], stride: usize, x: usize, y: usize, block: &[u8; 64]) {
    for row in 0..8 {
        let dst = (y + row) * stride + x;
        let src = row * 8;
        plane[dst..dst + 8].copy_from_slice(&block[src..src + 8]);
    }
}

fn emit_component_images<W: OutputWriter>(
    plan: &PreparedProgressivePlan,
    images: &[ComponentImage],
    writer: &mut W,
) -> Result<(), JpegError> {
    let (width, height) = plan.dimensions;
    let width_usize = width as usize;
    if plan.components.len() == 1 {
        let mut gray = vec![0u8; width_usize];
        for y in 0..height {
            upsample_component_row(plan, 0, &images[0], y, &mut gray);
            writer.write_gray_row(y, &gray)?;
        }
        return Ok(());
    }

    let first = component_slot(plan, 0)?;
    let second = component_slot(plan, 1)?;
    let third = component_slot(plan, 2)?;
    let mut a = vec![0u8; width_usize];
    let mut b = vec![0u8; width_usize];
    let mut c = vec![0u8; width_usize];
    for y in 0..height {
        upsample_component_row(plan, first, &images[first], y, &mut a);
        upsample_component_row(plan, second, &images[second], y, &mut b);
        upsample_component_row(plan, third, &images[third], y, &mut c);
        match plan.color_space {
            ColorSpace::YCbCr => writer.write_ycbcr_row(y, &a, &b, &c)?,
            ColorSpace::Rgb => writer.write_rgb_row(y, &a, &b, &c)?,
            ColorSpace::Grayscale => writer.write_gray_row(y, &a)?,
            ColorSpace::Cmyk | ColorSpace::Ycck => {
                return Err(JpegError::UnsupportedColorSpace {
                    color_space: plan.color_space,
                });
            }
        }
    }

    Ok(())
}

fn component_slot(plan: &PreparedProgressivePlan, output_index: usize) -> Result<usize, JpegError> {
    plan.components
        .iter()
        .position(|component| component.output_index == output_index)
        .ok_or(JpegError::UnsupportedColorSpace {
            color_space: plan.color_space,
        })
}

fn upsample_component_row(
    plan: &PreparedProgressivePlan,
    component_index: usize,
    image: &ComponentImage,
    y: u32,
    out: &mut [u8],
) {
    let component = &plan.components[component_index];
    let h_ratio = plan.sampling.max_h / component.h;
    let v_ratio = plan.sampling.max_v / component.v;
    if h_ratio == 1 && v_ratio == 1 {
        let sample_y = (y as usize).min(component.sample_height.saturating_sub(1) as usize);
        let row = component_row(component, image, sample_y);
        out.copy_from_slice(&row[..out.len()]);
    } else if h_ratio == 2 && v_ratio == 1 {
        let sample_y = (y as usize).min(component.sample_height.saturating_sub(1) as usize);
        let row = component_row(component, image, sample_y);
        upsample_h2v1_fancy_row(row, out.len(), out);
    } else if h_ratio == 2 && v_ratio == 2 {
        let sample_y = ((y / 2) as usize).min(component.sample_height.saturating_sub(1) as usize);
        let prev_y = sample_y.saturating_sub(1);
        let next_y = (sample_y + 1).min(component.sample_height.saturating_sub(1) as usize);
        let prev = component_row(component, image, prev_y);
        let curr = component_row(component, image, sample_y);
        let next = component_row(component, image, next_y);
        upsample_h2v2_fancy_row(prev, curr, next, out.len(), y % 2 == 1, out);
    } else {
        upsample_nearest(plan, component, image, y, out);
    }
}

fn component_row<'a>(
    component: &PreparedProgressiveComponentPlan,
    image: &'a ComponentImage,
    y: usize,
) -> &'a [u8] {
    let width = component.sample_width as usize;
    let row_start = y * image.stride;
    &image.plane[row_start..row_start + width]
}

fn upsample_nearest(
    plan: &PreparedProgressivePlan,
    component: &PreparedProgressiveComponentPlan,
    image: &ComponentImage,
    y: u32,
    out: &mut [u8],
) {
    let sample_y = ((y as usize) * usize::from(component.v) / usize::from(plan.sampling.max_v))
        .min(component.sample_height.saturating_sub(1) as usize);
    let row = component_row(component, image, sample_y);
    for (x, dst) in out.iter_mut().enumerate() {
        let sample_x = (x * usize::from(component.h) / usize::from(plan.sampling.max_h))
            .min(row.len().saturating_sub(1));
        *dst = row[sample_x];
    }
}

fn missing_dc_table() -> JpegError {
    JpegError::MissingHuffmanTable {
        component: 0,
        class: 0,
        id: 0,
    }
}

fn missing_ac_table() -> JpegError {
    JpegError::MissingHuffmanTable {
        component: 0,
        class: 1,
        id: 0,
    }
}

fn invalid_symbol() -> JpegError {
    JpegError::HuffmanDecode {
        mcu: 0,
        reason: HuffmanFailure::InvalidSymbol,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_eob_run_combines_prefix_and_extra_bits() {
        let bytes = [0b1010_0000u8];
        let mut br = BitReader::new(&bytes);

        let run = decode_eob_run(&mut br, 3).unwrap();

        assert_eq!(run, 12);
    }

    #[test]
    fn refine_non_zeroes_updates_existing_coefficients_by_sign() {
        let mut block = [0i32; 64];
        block[usize::from(ZIGZAG[1])] = 4;
        block[usize::from(ZIGZAG[2])] = -4;
        let bytes = [0b1100_0000u8];
        let mut br = BitReader::new(&bytes);

        refine_non_zeroes(&mut br, &mut block, 1, 2, 64, 2).unwrap();

        assert_eq!(block[usize::from(ZIGZAG[1])], 6);
        assert_eq!(block[usize::from(ZIGZAG[2])], -6);
    }

    #[test]
    fn refine_non_zeroes_stops_at_requested_zero_run() {
        let mut block = [0i32; 64];
        block[usize::from(ZIGZAG[3])] = 8;
        let bytes = [0u8];
        let mut br = BitReader::new(&bytes);

        let index = refine_non_zeroes(&mut br, &mut block, 1, 4, 1, 2).unwrap();

        assert_eq!(index, 2);
    }
}
