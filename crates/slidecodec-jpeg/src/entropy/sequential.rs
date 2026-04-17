// SPDX-License-Identifier: Apache-2.0

//! Baseline sequential scan decoder. Iterates MCUs, decodes blocks, runs the
//! IDCT, and pipes rows through an [`OutputWriter`] with chroma upsample and
//! color conversion.

use crate::color::upsample::{upsample_1x1, upsample_h2v1_fancy, upsample_h2v2_fancy};
use crate::entropy::block::decode_block;
use crate::entropy::huffman::HuffmanTable;
use crate::error::{HuffmanFailure, JpegError, Warning};
use crate::idct::idct_islow;
use crate::info::{ColorSpace, SamplingFactors};
use crate::internal::bit_reader::BitReader;
use crate::output::OutputWriter;
use alloc::vec;
use alloc::vec::Vec;

/// Per-component decode context. One entry per component declared in the
/// SOF, in scan order.
pub(crate) struct ComponentCtx<'a> {
    pub(crate) h: u8,
    pub(crate) v: u8,
    pub(crate) quant: &'a [u16; 64],
    pub(crate) dc_table: &'a HuffmanTable,
    pub(crate) ac_table: &'a HuffmanTable,
}

pub(crate) struct DecodeContext<'a> {
    pub(crate) components: Vec<ComponentCtx<'a>>,
    pub(crate) sampling: &'a SamplingFactors,
    pub(crate) color_space: ColorSpace,
    pub(crate) restart_interval: Option<u16>,
    pub(crate) dimensions: (u32, u32),
}

pub(crate) fn decode_scan_baseline<W: OutputWriter>(
    ctx: &DecodeContext<'_>,
    scan_bytes: &[u8],
    writer: &mut W,
) -> Result<Vec<Warning>, JpegError> {
    let (width, height) = ctx.dimensions;
    let max_h = ctx.sampling.max_h as u32;
    let max_v = ctx.sampling.max_v as u32;
    let mcu_width_px = 8 * max_h;
    let mcu_height_px = 8 * max_v;
    let mcus_per_row = width.div_ceil(mcu_width_px);
    let mcu_rows = height.div_ceil(mcu_height_px);

    let n_comp = ctx.components.len();
    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(n_comp);
    let mut plane_strides: Vec<usize> = Vec::with_capacity(n_comp);
    for c in &ctx.components {
        let cols = (mcus_per_row as usize) * (c.h as usize) * 8;
        let rows = (mcu_rows as usize) * (c.v as usize) * 8;
        planes.push(vec![0u8; cols * rows]);
        plane_strides.push(cols);
    }

    let mut br = BitReader::new(scan_bytes);
    let mut prev_dc = vec![0i32; n_comp];
    let mut coeff = [0i16; 64];
    let mut pixels = [0u8; 64];

    let restart = ctx.restart_interval.unwrap_or(0);
    let mut mcus_since_restart = 0u32;
    let mut expected_rst = 0u8;

    for my in 0..mcu_rows {
        for mx in 0..mcus_per_row {
            if restart > 0 && mcus_since_restart == u32::from(restart) {
                let _ = br.ensure_bits(1);
                let marker = br.take_marker().ok_or(JpegError::UnexpectedEoi {
                    mcu_at: my * mcus_per_row + mx,
                    mcu_total: mcu_rows * mcus_per_row,
                })?;
                let expected = 0xD0 | expected_rst;
                if marker != expected {
                    return Err(JpegError::RestartMismatch {
                        offset: br.position(),
                        expected: expected_rst,
                        found: marker,
                    });
                }
                expected_rst = (expected_rst + 1) & 0x07;
                br.reset_at_restart();
                prev_dc.fill(0);
                mcus_since_restart = 0;
            }

            for (ci, comp) in ctx.components.iter().enumerate() {
                let mcu_x0_px = mx * (comp.h as u32) * 8;
                let mcu_y0_px = my * (comp.v as u32) * 8;
                for vy in 0..comp.v as u32 {
                    for vx in 0..comp.h as u32 {
                        decode_block(
                            &mut br,
                            comp.dc_table,
                            comp.ac_table,
                            &mut prev_dc[ci],
                            comp.quant,
                            &mut coeff,
                        )?;
                        idct_islow(&coeff, &mut pixels);
                        deposit_block(
                            &mut planes[ci],
                            plane_strides[ci],
                            mcu_x0_px + vx * 8,
                            mcu_y0_px + vy * 8,
                            &pixels,
                        );
                    }
                }
            }
            mcus_since_restart += 1;
        }

        let out_y_start = my * mcu_height_px;
        let out_y_end = (out_y_start + mcu_height_px).min(height);
        emit_rows(ctx, &planes, &plane_strides, out_y_start, out_y_end, writer)?;
    }

    let mut warnings = Vec::new();
    if br.take_marker() != Some(0xD9) {
        warnings.push(Warning::MissingEoi);
    }
    Ok(warnings)
}

fn deposit_block(plane: &mut [u8], stride: usize, x: u32, y: u32, block: &[u8; 64]) {
    let x = x as usize;
    let y = y as usize;
    for by in 0..8 {
        let dst_start = (y + by) * stride + x;
        plane[dst_start..dst_start + 8].copy_from_slice(&block[by * 8..by * 8 + 8]);
    }
}

fn emit_rows<W: OutputWriter>(
    ctx: &DecodeContext<'_>,
    planes: &[Vec<u8>],
    strides: &[usize],
    y_start: u32,
    y_end: u32,
    writer: &mut W,
) -> Result<(), JpegError> {
    let width = ctx.dimensions.0 as usize;
    match ctx.color_space {
        ColorSpace::Grayscale => {
            let y_stride = strides[0];
            for y in y_start..y_end {
                let row_start = y as usize * y_stride;
                let y_row = &planes[0][row_start..row_start + width];
                writer.write_gray_row(y, y_row);
            }
        }
        ColorSpace::YCbCr => {
            let (cb_h, cb_v) = (ctx.components[1].h as u32, ctx.components[1].v as u32);
            let (cr_h, cr_v) = (ctx.components[2].h as u32, ctx.components[2].v as u32);

            let max_h = ctx.sampling.max_h as u32;
            let max_v = ctx.sampling.max_v as u32;

            let y_stride = strides[0];
            let cb_stride = strides[1];
            let cr_stride = strides[2];

            let mut cb_up = vec![0u8; width];
            let mut cr_up = vec![0u8; width];

            for y in y_start..y_end {
                let row_start = y as usize * y_stride;
                let y_row = &planes[0][row_start..row_start + width];
                upsample_component_row(
                    &planes[1], cb_stride, cb_h, cb_v, max_h, max_v, y, width, &mut cb_up,
                );
                upsample_component_row(
                    &planes[2], cr_stride, cr_h, cr_v, max_h, max_v, y, width, &mut cr_up,
                );
                writer.write_ycbcr_row(y, y_row, &cb_up, &cr_up);
            }
        }
        ColorSpace::Rgb => {
            let (r_h, r_v) = (ctx.components[0].h as u32, ctx.components[0].v as u32);
            let (g_h, g_v) = (ctx.components[1].h as u32, ctx.components[1].v as u32);
            let (b_h, b_v) = (ctx.components[2].h as u32, ctx.components[2].v as u32);

            let max_h = ctx.sampling.max_h as u32;
            let max_v = ctx.sampling.max_v as u32;

            let r_stride = strides[0];
            let g_stride = strides[1];
            let b_stride = strides[2];

            let mut r_up = vec![0u8; width];
            let mut g_up = vec![0u8; width];
            let mut b_up = vec![0u8; width];

            for y in y_start..y_end {
                upsample_component_row(
                    &planes[0], r_stride, r_h, r_v, max_h, max_v, y, width, &mut r_up,
                );
                upsample_component_row(
                    &planes[1], g_stride, g_h, g_v, max_h, max_v, y, width, &mut g_up,
                );
                upsample_component_row(
                    &planes[2], b_stride, b_h, b_v, max_h, max_v, y, width, &mut b_up,
                );
                writer.write_rgb_row(y, &r_up, &g_up, &b_up);
            }
        }
        ColorSpace::Cmyk | ColorSpace::Ycck => {
            return Err(JpegError::HuffmanDecode {
                mcu: 0,
                reason: HuffmanFailure::InvalidSymbol,
            });
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn upsample_component_row(
    plane: &[u8],
    stride: usize,
    comp_h: u32,
    comp_v: u32,
    max_h: u32,
    max_v: u32,
    y_out: u32,
    width: usize,
    out: &mut [u8],
) {
    let v_ratio = max_v / comp_v;
    let h_ratio = max_h / comp_h;

    // INTEGER chroma-row mapping (no f32 — no_std compatibility).
    let chroma_rows = (plane.len() / stride) as u32;
    let chroma_y = (y_out / v_ratio).min(chroma_rows.saturating_sub(1));

    let prev_y = chroma_y.saturating_sub(1);
    let next_y = (chroma_y + 1).min(chroma_rows.saturating_sub(1));

    let prev_row = &plane[prev_y as usize * stride..prev_y as usize * stride + stride];
    let curr_row = &plane[chroma_y as usize * stride..chroma_y as usize * stride + stride];
    let next_row = &plane[next_y as usize * stride..next_y as usize * stride + stride];

    match (h_ratio, v_ratio) {
        (1, 1) => {
            upsample_1x1(&curr_row[..width], out);
        }
        (2, 1) => {
            let chroma_cols = width.div_ceil(2);
            let mut tmp = alloc::vec![0u8; chroma_cols * 2];
            upsample_h2v1_fancy(&curr_row[..chroma_cols], &mut tmp);
            let n = width.min(tmp.len());
            out[..n].copy_from_slice(&tmp[..n]);
        }
        (2, 2) => {
            let chroma_cols = width.div_ceil(2);
            let mut top = alloc::vec![0u8; chroma_cols * 2];
            let mut bot = alloc::vec![0u8; chroma_cols * 2];
            upsample_h2v2_fancy(
                &prev_row[..chroma_cols],
                &curr_row[..chroma_cols],
                &next_row[..chroma_cols],
                &mut top,
                &mut bot,
            );
            let n = width.min(top.len());
            if y_out.is_multiple_of(2) {
                out[..n].copy_from_slice(&top[..n]);
            } else {
                out[..n].copy_from_slice(&bot[..n]);
            }
        }
        _ => {
            for (x, slot) in out.iter_mut().enumerate().take(width) {
                let cx = ((x as u32) / h_ratio).min(stride as u32 - 1);
                *slot = curr_row[cx as usize];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // Integration coverage in tests/decode_into.rs (Task 16).
}
