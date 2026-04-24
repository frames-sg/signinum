// SPDX-License-Identifier: Apache-2.0

//! Baseline sequential scan decoder. Iterates MCUs, decodes blocks, runs the
//! IDCT, and pipes rows through an [`OutputWriter`] with chroma upsample and
//! color conversion.

use crate::backend::Backend;
use crate::color::upsample::{
    upsample_1x1, upsample_h2v1_fancy_row, upsample_h2v2_fancy_row, upsample_h2v2_fancy_rows,
};
use crate::entropy::block::{
    decode_block_for_1x1_idct, decode_block_for_reduced_idct, decode_block_with_activity,
    skip_block, BlockActivity, CoefficientBlock, ReducedIdctCoefficients,
};
use crate::entropy::huffman::HuffmanTable;
use crate::error::{HuffmanFailure, JpegError, Warning};
use crate::idct::downscale;
use crate::info::{ColorSpace, DownscaleFactor, Rect, SamplingFactors};
use crate::internal::bit_reader::BitReader;
use crate::internal::scratch::{
    RgbGenericRows, ScratchPool, SinkRows, YCbCr420Rows, YCbCrGenericRows,
};
use crate::output::{InterleavedRgbWriter, OutputWriter};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ptr;

/// Per-component decode context. One entry per component declared in the
/// SOF, in scan order.
#[derive(Debug, Clone)]
pub(crate) struct PreparedComponentPlan {
    pub(crate) h: u8,
    pub(crate) v: u8,
    pub(crate) output_index: usize,
    pub(crate) quant: Arc<[u16; 64]>,
    pub(crate) dc_table: Arc<HuffmanTable>,
    pub(crate) ac_table: Arc<HuffmanTable>,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedDecodePlan {
    pub(crate) components: Vec<PreparedComponentPlan>,
    pub(crate) sampling: SamplingFactors,
    pub(crate) color_space: ColorSpace,
    pub(crate) restart_interval: Option<u16>,
    pub(crate) dimensions: (u32, u32),
    pub(crate) scan_offset: usize,
    pub(crate) scratch_bytes: usize,
}

impl PreparedDecodePlan {
    pub(crate) fn matches_fast_tile_shape(&self) -> bool {
        self.restart_interval.is_none()
            && is_ycbcr_420(self)
            && self.components.len() == 3
            && self.components[0].output_index == 0
            && self.components[0].h == 2
            && self.components[0].v == 2
            && self.components[1].output_index == 1
            && self.components[1].h == 1
            && self.components[1].v == 1
            && self.components[2].output_index == 2
            && self.components[2].h == 1
            && self.components[2].v == 1
    }

    pub(crate) fn matches_fast_rgb444_shape(&self) -> bool {
        self.color_space == ColorSpace::YCbCr
            && self.components.len() == 3
            && self.components[0].output_index == 0
            && self.components[0].h == 1
            && self.components[0].v == 1
            && self.components[1].output_index == 1
            && self.components[1].h == 1
            && self.components[1].v == 1
            && self.components[2].output_index == 2
            && self.components[2].h == 1
            && self.components[2].v == 1
    }
}

enum OutputScratch<'a> {
    Grayscale,
    YCbCr420(&'a mut YCbCr420Rows),
    YCbCrGeneric(&'a mut YCbCrGenericRows),
    RgbGeneric(&'a mut RgbGenericRows),
}

enum RgbOutputScratch<'a> {
    None,
    YCbCr420,
    YCbCrGeneric(&'a mut YCbCrGenericRows),
    RgbGeneric(&'a mut RgbGenericRows),
}

#[derive(Debug, Default)]
pub(crate) struct StripeBuffer {
    pub(crate) planes: Vec<Vec<u8>>,
    pub(crate) plane_strides: Vec<usize>,
    pub(crate) plane_rows: Vec<usize>,
}

#[derive(Clone, Copy)]
struct StripePlane<'a> {
    data: &'a [u8],
    stride: usize,
    rows: usize,
}

impl StripeBuffer {
    /// Grow each plane's backing Vec to the size required by `plan` and
    /// `mcus_per_row`. Never shrinks the allocation — a monotonic
    /// tile-batch workload pays the allocation cost exactly once.
    pub(crate) fn resize_for(
        &mut self,
        plan: &PreparedDecodePlan,
        mcus_per_row: u32,
        block_size: u32,
    ) {
        let n = plan.sampling.len();
        self.planes.resize_with(n, Vec::new);
        self.plane_strides.resize(n, 0);
        self.plane_rows.resize(n, 0);
        for (i, (h, v)) in plan.sampling.iter().enumerate() {
            let cols = (mcus_per_row as usize) * (h as usize) * (block_size as usize);
            let rows = (v as usize) * (block_size as usize);
            let bytes = cols * rows;
            if self.planes[i].len() < bytes {
                self.planes[i].resize(bytes, 0);
            }
            self.plane_strides[i] = cols;
            self.plane_rows[i] = rows;
        }
    }

    fn row_count(&self, plane_idx: usize) -> usize {
        self.plane_rows[plane_idx]
    }

    fn row(&self, plane_idx: usize, row: usize) -> &[u8] {
        let stride = self.plane_strides[plane_idx];
        let start = row * stride;
        &self.planes[plane_idx][start..start + stride]
    }

    fn plane(&self, plane_idx: usize) -> StripePlane<'_> {
        StripePlane {
            data: &self.planes[plane_idx],
            stride: self.plane_strides[plane_idx],
            rows: self.plane_rows[plane_idx],
        }
    }
}

pub(crate) fn decode_scan_baseline<W: OutputWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    downscale: DownscaleFactor,
    output_rect: Rect,
) -> Result<Vec<Warning>, JpegError> {
    let (width, height) = scaled_dimensions(plan.dimensions, downscale);
    let max_h = plan.sampling.max_h as u32;
    let max_v = plan.sampling.max_v as u32;
    let block_size = downscale.output_block_size();
    let mcu_width_px = block_size * max_h;
    let mcu_height_px = block_size * max_v;
    let mcus_per_row = width.div_ceil(mcu_width_px);
    let mcu_rows = height.div_ceil(mcu_height_px);

    let region_layout = stripe_region_layout(plan, downscale, output_rect);

    pool.prepare_for(plan, region_layout.stripe_mcus_per_row, block_size);

    let mut br = BitReader::new(scan_bytes);
    let mut coeff = CoefficientBlock::default();
    let mut pixels = [0u8; 64];
    let ScratchPool {
        prev_dc,
        stripe_a,
        stripe_b,
        stripe_c,
        ycbcr_420_rows,
        ycbcr_generic_rows,
        rgb_generic_rows,
        ..
    } = pool;
    let mut output_scratch = match plan.color_space {
        ColorSpace::Grayscale => OutputScratch::Grayscale,
        ColorSpace::YCbCr if is_ycbcr_420(plan) => OutputScratch::YCbCr420(ycbcr_420_rows),
        ColorSpace::YCbCr => OutputScratch::YCbCrGeneric(ycbcr_generic_rows),
        ColorSpace::Rgb => OutputScratch::RgbGeneric(rgb_generic_rows),
        ColorSpace::Cmyk | ColorSpace::Ycck => OutputScratch::Grayscale,
    };
    let mut prev_stripe: &mut StripeBuffer = stripe_a;
    let mut curr_stripe: &mut StripeBuffer = stripe_b;
    let mut next_stripe: &mut StripeBuffer = stripe_c;

    let restart = plan.restart_interval.unwrap_or(0);
    let mut mcus_since_restart = 0u32;
    let mut expected_rst = 0u8;
    let expanded_rect = expanded_output_rect(output_rect, width, height);
    let full_output_rect = expanded_rect == Rect::full((width, height));
    let first_decode_mcu_row =
        first_decode_mcu_row_for_rect(full_output_rect, expanded_rect, mcu_height_px);
    let decode_mcu_row_end =
        decode_mcu_row_end_for_rect(full_output_rect, expanded_rect, mcu_height_px, mcu_rows);
    let last_output_mcu_row = last_mcu_row_for_rect(expanded_rect, mcu_height_px, mcu_rows);
    let total_mcus = mcu_rows * mcus_per_row;
    let first_decode_mcu = first_decode_mcu_row * mcus_per_row;
    let mut current_mcu = 0u32;
    if let Some(seek) = restart_seek_for_mcu(scan_bytes, restart, first_decode_mcu) {
        br = BitReader::new(&scan_bytes[seek.scan_offset..]);
        current_mcu = seek.mcu_index;
        expected_rst = seek.expected_rst;
    }
    skip_to_mcu(
        plan,
        &mut br,
        prev_dc,
        &mut current_mcu,
        first_decode_mcu,
        total_mcus,
        restart,
        &mut mcus_since_restart,
        &mut expected_rst,
    )?;

    decode_mcu_row(
        plan,
        backend,
        &mut br,
        prev_dc,
        &mut coeff,
        &mut pixels,
        downscale,
        expanded_rect,
        full_output_rect,
        first_decode_mcu_row,
        region_layout.stripe_mcu_start,
        region_layout.stripe_mcus_per_row,
        mcus_per_row,
        mcu_rows,
        curr_stripe,
        restart,
        &mut mcus_since_restart,
        &mut expected_rst,
    )?;

    let mut has_prev = false;
    for my in first_decode_mcu_row + 1..decode_mcu_row_end {
        decode_mcu_row(
            plan,
            backend,
            &mut br,
            prev_dc,
            &mut coeff,
            &mut pixels,
            downscale,
            expanded_rect,
            full_output_rect,
            my,
            region_layout.stripe_mcu_start,
            region_layout.stripe_mcus_per_row,
            mcus_per_row,
            mcu_rows,
            next_stripe,
            restart,
            &mut mcus_since_restart,
            &mut expected_rst,
        )?;
        if full_output_rect || mcu_row_intersects_rect(my - 1, mcu_height_px, expanded_rect) {
            emit_stripe(
                plan,
                has_prev.then_some(&*prev_stripe),
                curr_stripe,
                Some(&*next_stripe),
                my - 1,
                writer,
                &mut output_scratch,
                region_layout.source_width_usize(),
                downscale,
            )?;
        }
        core::mem::swap(&mut prev_stripe, &mut curr_stripe);
        core::mem::swap(&mut curr_stripe, &mut next_stripe);
        has_prev = true;
    }

    let curr_mcu_row = decode_mcu_row_end - 1;
    if curr_mcu_row <= last_output_mcu_row
        && (full_output_rect || mcu_row_intersects_rect(curr_mcu_row, mcu_height_px, expanded_rect))
    {
        emit_stripe(
            plan,
            has_prev.then_some(&*prev_stripe),
            curr_stripe,
            None,
            curr_mcu_row,
            writer,
            &mut output_scratch,
            region_layout.source_width_usize(),
            downscale,
        )?;
    }
    finish_scan(&mut br, decode_mcu_row_end == mcu_rows)
}

pub(crate) fn decode_scan_baseline_rgb<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    downscale: DownscaleFactor,
    output_rect: Rect,
) -> Result<Vec<Warning>, JpegError> {
    let (width, height) = scaled_dimensions(plan.dimensions, downscale);
    let max_h = plan.sampling.max_h as u32;
    let max_v = plan.sampling.max_v as u32;
    let block_size = downscale.output_block_size();
    let mcu_width_px = block_size * max_h;
    let mcu_height_px = block_size * max_v;
    let mcus_per_row = width.div_ceil(mcu_width_px);
    let mcu_rows = height.div_ceil(mcu_height_px);

    let region_layout = stripe_region_layout(plan, downscale, output_rect);

    pool.prepare_for(plan, region_layout.stripe_mcus_per_row, block_size);

    let mut br = BitReader::new(scan_bytes);
    let mut coeff = CoefficientBlock::default();
    let mut pixels = [0u8; 64];
    let ScratchPool {
        prev_dc,
        stripe_a,
        stripe_b,
        stripe_c,
        ycbcr_generic_rows,
        rgb_generic_rows,
        ..
    } = pool;
    let mut output_scratch = match plan.color_space {
        ColorSpace::Grayscale => RgbOutputScratch::None,
        ColorSpace::YCbCr if is_ycbcr_420(plan) => RgbOutputScratch::YCbCr420,
        ColorSpace::YCbCr => RgbOutputScratch::YCbCrGeneric(ycbcr_generic_rows),
        ColorSpace::Rgb => RgbOutputScratch::RgbGeneric(rgb_generic_rows),
        ColorSpace::Cmyk | ColorSpace::Ycck => RgbOutputScratch::None,
    };
    let mut prev_stripe: &mut StripeBuffer = stripe_a;
    let mut curr_stripe: &mut StripeBuffer = stripe_b;
    let mut next_stripe: &mut StripeBuffer = stripe_c;

    let restart = plan.restart_interval.unwrap_or(0);
    let mut mcus_since_restart = 0u32;
    let mut expected_rst = 0u8;
    let expanded_rect = expanded_output_rect(output_rect, width, height);
    let full_output_rect = expanded_rect == Rect::full((width, height));
    let use_420_context_window = !full_output_rect && is_ycbcr_420(plan);
    let emit_rect = if use_420_context_window {
        output_rect
    } else {
        expanded_rect
    };
    let first_decode_mcu_row = if use_420_context_window {
        fast420_first_decode_mcu_row(output_rect, mcu_height_px)
    } else {
        first_decode_mcu_row_for_rect(full_output_rect, expanded_rect, mcu_height_px)
    };
    let decode_mcu_row_end = if use_420_context_window {
        fast420_decode_mcu_row_end(output_rect, mcu_height_px, mcu_rows)
    } else {
        decode_mcu_row_end_for_rect(full_output_rect, expanded_rect, mcu_height_px, mcu_rows)
    };
    let last_output_mcu_row = last_mcu_row_for_rect(emit_rect, mcu_height_px, mcu_rows);
    let total_mcus = mcu_rows * mcus_per_row;
    let first_decode_mcu = first_decode_mcu_row * mcus_per_row;
    let mut current_mcu = 0u32;
    if let Some(seek) = restart_seek_for_mcu(scan_bytes, restart, first_decode_mcu) {
        br = BitReader::new(&scan_bytes[seek.scan_offset..]);
        current_mcu = seek.mcu_index;
        expected_rst = seek.expected_rst;
    }
    skip_to_mcu(
        plan,
        &mut br,
        prev_dc,
        &mut current_mcu,
        first_decode_mcu,
        total_mcus,
        restart,
        &mut mcus_since_restart,
        &mut expected_rst,
    )?;

    decode_mcu_row(
        plan,
        backend,
        &mut br,
        prev_dc,
        &mut coeff,
        &mut pixels,
        downscale,
        expanded_rect,
        full_output_rect,
        first_decode_mcu_row,
        region_layout.stripe_mcu_start,
        region_layout.stripe_mcus_per_row,
        mcus_per_row,
        mcu_rows,
        curr_stripe,
        restart,
        &mut mcus_since_restart,
        &mut expected_rst,
    )?;

    let mut has_prev = false;
    for my in first_decode_mcu_row + 1..decode_mcu_row_end {
        decode_mcu_row(
            plan,
            backend,
            &mut br,
            prev_dc,
            &mut coeff,
            &mut pixels,
            downscale,
            expanded_rect,
            full_output_rect,
            my,
            region_layout.stripe_mcu_start,
            region_layout.stripe_mcus_per_row,
            mcus_per_row,
            mcu_rows,
            next_stripe,
            restart,
            &mut mcus_since_restart,
            &mut expected_rst,
        )?;
        if full_output_rect || mcu_row_intersects_rect(my - 1, mcu_height_px, emit_rect) {
            emit_stripe_rgb(
                plan,
                backend,
                has_prev.then_some(&*prev_stripe),
                curr_stripe,
                Some(&*next_stripe),
                my - 1,
                writer,
                &mut output_scratch,
                region_layout.source_width_usize(),
                downscale,
            )?;
        }
        core::mem::swap(&mut prev_stripe, &mut curr_stripe);
        core::mem::swap(&mut curr_stripe, &mut next_stripe);
        has_prev = true;
    }

    let curr_mcu_row = decode_mcu_row_end - 1;
    if curr_mcu_row <= last_output_mcu_row
        && (full_output_rect || mcu_row_intersects_rect(curr_mcu_row, mcu_height_px, emit_rect))
    {
        emit_stripe_rgb(
            plan,
            backend,
            has_prev.then_some(&*prev_stripe),
            curr_stripe,
            None,
            curr_mcu_row,
            writer,
            &mut output_scratch,
            region_layout.source_width_usize(),
            downscale,
        )?;
    }
    finish_scan(&mut br, decode_mcu_row_end == mcu_rows)
}

pub(crate) fn decode_scan_fast_tile_rgb<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
) -> Result<Vec<Warning>, JpegError> {
    debug_assert!(plan.matches_fast_tile_shape());

    let (width, height) = plan.dimensions;
    let max_h = plan.sampling.max_h as u32;
    let max_v = plan.sampling.max_v as u32;
    let mcu_width_px = 8 * max_h;
    let mcu_height_px = 8 * max_v;
    let mcus_per_row = width.div_ceil(mcu_width_px);
    let mcu_rows = height.div_ceil(mcu_height_px);

    pool.prepare_for(
        plan,
        mcus_per_row,
        DownscaleFactor::Full.output_block_size(),
    );

    let mut br = BitReader::new(scan_bytes);
    let mut coeff = CoefficientBlock::default();
    let mut pixels = [0u8; 64];
    let (y_comp, cb_comp, cr_comp) = fast_tile_components(plan);
    let mut y_dc = 0i32;
    let mut cb_dc = 0i32;
    let mut cr_dc = 0i32;

    let ScratchPool {
        stripe_a,
        stripe_b,
        stripe_c,
        ..
    } = pool;
    let mut prev_stripe: &mut StripeBuffer = stripe_a;
    let mut curr_stripe: &mut StripeBuffer = stripe_b;
    let mut next_stripe: &mut StripeBuffer = stripe_c;
    let mut output_scratch = RgbOutputScratch::YCbCr420;

    decode_mcu_row_fast_tile_420(
        y_comp,
        cb_comp,
        cr_comp,
        backend,
        &mut br,
        &mut y_dc,
        &mut cb_dc,
        &mut cr_dc,
        &mut coeff,
        &mut pixels,
        mcus_per_row,
        0,
        mcus_per_row,
        curr_stripe,
    )?;

    let mut has_prev = false;
    for my in 1..mcu_rows {
        decode_mcu_row_fast_tile_420(
            y_comp,
            cb_comp,
            cr_comp,
            backend,
            &mut br,
            &mut y_dc,
            &mut cb_dc,
            &mut cr_dc,
            &mut coeff,
            &mut pixels,
            mcus_per_row,
            0,
            mcus_per_row,
            next_stripe,
        )?;
        emit_stripe_rgb(
            plan,
            backend,
            has_prev.then_some(&*prev_stripe),
            curr_stripe,
            Some(&*next_stripe),
            my - 1,
            writer,
            &mut output_scratch,
            width as usize,
            DownscaleFactor::Full,
        )?;
        core::mem::swap(&mut prev_stripe, &mut curr_stripe);
        core::mem::swap(&mut curr_stripe, &mut next_stripe);
        has_prev = true;
    }

    emit_stripe_rgb(
        plan,
        backend,
        has_prev.then_some(&*prev_stripe),
        curr_stripe,
        None,
        mcu_rows - 1,
        writer,
        &mut output_scratch,
        width as usize,
        DownscaleFactor::Full,
    )?;

    let mut warnings = Vec::new();
    match br.take_marker() {
        Some(0xD9) => Ok(warnings),
        Some(found) => Err(JpegError::UnexpectedMarker {
            offset: br.position().saturating_sub(2),
            expected: crate::error::MarkerKind::Eoi,
            found,
        }),
        None => {
            warnings.push(Warning::MissingEoi);
            Ok(warnings)
        }
    }
}

#[derive(Clone, Copy)]
struct RgbCropWindow {
    scratch_x0: usize,
    scratch_x1: usize,
}

impl RgbCropWindow {
    fn new(width: usize, roi: Rect) -> Self {
        let roi_x0 = roi.x as usize;
        let roi_x1 = roi_x0 + roi.w as usize;
        let chroma_width = width.div_ceil(2);
        let sample_start = (roi_x0 / 2).saturating_sub(1);
        let sample_end = (roi_x1.div_ceil(2) + 1).min(chroma_width);
        let scratch_x0 = sample_start * 2;
        let scratch_x1 = (sample_end * 2).min(width);
        Self {
            scratch_x0,
            scratch_x1,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct StripeRegionLayout {
    pub(crate) stripe_mcu_start: u32,
    pub(crate) stripe_mcus_per_row: u32,
    pub(crate) source_x0: u32,
    pub(crate) source_width: u32,
}

impl StripeRegionLayout {
    fn new(full_width: u32, mcu_width_px: u32, output_rect: Rect) -> Self {
        let source_x0 = (output_rect.x / mcu_width_px) * mcu_width_px;
        let source_x1 = output_rect
            .x
            .saturating_add(output_rect.w)
            .div_ceil(mcu_width_px)
            .saturating_mul(mcu_width_px)
            .min(full_width);
        let stripe_mcu_start = source_x0 / mcu_width_px;
        let stripe_mcu_end = source_x1.div_ceil(mcu_width_px);
        Self {
            stripe_mcu_start,
            stripe_mcus_per_row: stripe_mcu_end.saturating_sub(stripe_mcu_start),
            source_x0,
            source_width: source_x1.saturating_sub(source_x0),
        }
    }

    fn source_width_usize(self) -> usize {
        self.source_width as usize
    }
}

pub(crate) fn stripe_region_layout(
    plan: &PreparedDecodePlan,
    downscale: DownscaleFactor,
    output_rect: Rect,
) -> StripeRegionLayout {
    let (scaled_width, scaled_height) = scaled_dimensions(plan.dimensions, downscale);
    let expanded_rect = expanded_output_rect(output_rect, scaled_width, scaled_height);
    let mcu_width_px = downscale.output_block_size() * u32::from(plan.sampling.max_h);
    StripeRegionLayout::new(scaled_width, mcu_width_px, expanded_rect)
}

#[inline]
fn last_mcu_row_for_rect(rect: Rect, mcu_height_px: u32, mcu_rows: u32) -> u32 {
    let last_y = rect.y.saturating_add(rect.h).saturating_sub(1);
    (last_y / mcu_height_px).min(mcu_rows.saturating_sub(1))
}

#[inline]
fn first_mcu_row_for_rect(rect: Rect, mcu_height_px: u32) -> u32 {
    rect.y / mcu_height_px
}

#[inline]
fn first_decode_mcu_row_for_rect(full_output_rect: bool, rect: Rect, mcu_height_px: u32) -> u32 {
    if full_output_rect {
        0
    } else {
        first_mcu_row_for_rect(rect, mcu_height_px).saturating_sub(1)
    }
}

#[inline]
fn decode_mcu_row_end_for_rect(
    full_output_rect: bool,
    rect: Rect,
    mcu_height_px: u32,
    mcu_rows: u32,
) -> u32 {
    if full_output_rect {
        return mcu_rows;
    }
    let last_output_mcu_row = last_mcu_row_for_rect(rect, mcu_height_px, mcu_rows);
    if last_output_mcu_row + 1 < mcu_rows {
        last_output_mcu_row + 2
    } else {
        mcu_rows
    }
}

#[inline]
fn fast420_first_decode_mcu_row(roi: Rect, mcu_height_px: u32) -> u32 {
    let first_row = first_mcu_row_for_rect(roi, mcu_height_px);
    if roi.y.is_multiple_of(mcu_height_px) {
        first_row.saturating_sub(1)
    } else {
        first_row
    }
}

#[inline]
fn fast420_decode_mcu_row_end(roi: Rect, mcu_height_px: u32, mcu_rows: u32) -> u32 {
    let last_row = last_mcu_row_for_rect(roi, mcu_height_px, mcu_rows);
    let last_local_y = (roi.y + roi.h - 1) % mcu_height_px;
    let needs_next_row = last_local_y == mcu_height_px.saturating_sub(1);
    if needs_next_row && last_row + 1 < mcu_rows {
        last_row + 2
    } else {
        last_row + 1
    }
}

#[inline]
fn mcu_row_intersects_rect(stripe_index: u32, mcu_height_px: u32, rect: Rect) -> bool {
    let y0 = stripe_index * mcu_height_px;
    let y1 = y0 + mcu_height_px;
    let rect_y1 = rect.y + rect.h;
    y0 < rect_y1 && y1 > rect.y
}

fn finish_scan(br: &mut BitReader<'_>, validate_eoi: bool) -> Result<Vec<Warning>, JpegError> {
    if !validate_eoi {
        return Ok(Vec::new());
    }

    let mut warnings = Vec::new();
    match br.take_marker() {
        Some(0xD9) => Ok(warnings),
        Some(found) => Err(JpegError::UnexpectedMarker {
            offset: br.position().saturating_sub(2),
            expected: crate::error::MarkerKind::Eoi,
            found,
        }),
        None => {
            warnings.push(Warning::MissingEoi);
            Ok(warnings)
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RestartSeek {
    scan_offset: usize,
    mcu_index: u32,
    expected_rst: u8,
}

fn restart_seek_for_mcu(scan_bytes: &[u8], restart: u16, target_mcu: u32) -> Option<RestartSeek> {
    if restart == 0 {
        return None;
    }
    let restart = u32::from(restart);
    let restart_index = target_mcu / restart;
    if restart_index == 0 {
        return None;
    }
    let marker_ordinal = restart_index - 1;
    let mut seen = 0u32;
    let mut pos = 0usize;
    while pos + 1 < scan_bytes.len() {
        if scan_bytes[pos] != 0xff {
            pos += 1;
            continue;
        }

        let mut marker_pos = pos + 1;
        while marker_pos < scan_bytes.len() && scan_bytes[marker_pos] == 0xff {
            marker_pos += 1;
        }
        if marker_pos >= scan_bytes.len() {
            return None;
        }

        let marker = scan_bytes[marker_pos];
        match marker {
            0x00 => pos = marker_pos + 1,
            0xd0..=0xd7 => {
                if seen == marker_ordinal {
                    return Some(RestartSeek {
                        scan_offset: marker_pos + 1,
                        mcu_index: restart_index * restart,
                        expected_rst: (restart_index & 0x07) as u8,
                    });
                }
                seen += 1;
                pos = marker_pos + 1;
            }
            0xd9 => return None,
            _ => return None,
        }
    }
    None
}

fn skip_mcu(
    plan: &PreparedDecodePlan,
    br: &mut BitReader<'_>,
    prev_dc: &mut [i32],
) -> Result<(), JpegError> {
    for comp in &plan.components {
        let plane_idx = comp.output_index;
        for _ in 0..u32::from(comp.h) * u32::from(comp.v) {
            skip_block(br, &comp.dc_table, &comp.ac_table, &mut prev_dc[plane_idx])?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn skip_to_mcu(
    plan: &PreparedDecodePlan,
    br: &mut BitReader<'_>,
    prev_dc: &mut [i32],
    current_mcu: &mut u32,
    target_mcu: u32,
    total_mcus: u32,
    restart: u16,
    mcus_since_restart: &mut u32,
    expected_rst: &mut u8,
) -> Result<(), JpegError> {
    while *current_mcu < target_mcu {
        if restart > 0 && *mcus_since_restart == u32::from(restart) {
            let _ = br.ensure_bits(1);
            let marker = br.take_marker().ok_or(JpegError::UnexpectedEoi {
                mcu_at: *current_mcu,
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
            prev_dc.fill(0);
            *mcus_since_restart = 0;
        }

        skip_mcu(plan, br, prev_dc)?;
        *mcus_since_restart += 1;
        *current_mcu += 1;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Fast420RegionLayout {
    stripe_mcu_start: u32,
    stripe_mcus_per_row: u32,
    y_decode_start: usize,
    y_decode_end: usize,
    crop_start: usize,
    crop_end: usize,
}

impl Fast420RegionLayout {
    fn new(width: usize, roi: Rect) -> Self {
        Self::new_for_mcu_width(width, roi, 16)
    }

    fn new_for_mcu_width(width: usize, roi: Rect, mcu_width_px: u32) -> Self {
        let crop_window = RgbCropWindow::new(width, roi);
        let stripe = StripeRegionLayout::new(
            width as u32,
            mcu_width_px,
            Rect {
                x: crop_window.scratch_x0 as u32,
                y: 0,
                w: (crop_window.scratch_x1 - crop_window.scratch_x0) as u32,
                h: 1,
            },
        );
        let y_decode_start = stripe.source_x0 as usize;
        let y_decode_end = y_decode_start + stripe.source_width as usize;
        let crop_start = roi.x as usize - y_decode_start;
        let crop_end = crop_start + roi.w as usize;

        Self {
            stripe_mcu_start: stripe.stripe_mcu_start,
            stripe_mcus_per_row: stripe.stripe_mcus_per_row,
            y_decode_start,
            y_decode_end,
            crop_start,
            crop_end,
        }
    }

    fn row_width(self) -> usize {
        self.y_decode_end - self.y_decode_start
    }

    #[cfg(test)]
    fn chroma_width(self) -> usize {
        self.row_width().div_ceil(2)
    }
}

pub(crate) fn decode_scan_fast_tile_rgb_region<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    roi: Rect,
) -> Result<Vec<Warning>, JpegError> {
    debug_assert!(plan.matches_fast_tile_shape());

    let (width, height) = plan.dimensions;
    let max_h = plan.sampling.max_h as u32;
    let max_v = plan.sampling.max_v as u32;
    let mcu_width_px = 8 * max_h;
    let mcu_height_px = 8 * max_v;
    let mcus_per_row = width.div_ceil(mcu_width_px);
    let mcu_rows = height.div_ceil(mcu_height_px);
    let first_decode_mcu_row = fast420_first_decode_mcu_row(roi, mcu_height_px);
    let decode_mcu_row_end = fast420_decode_mcu_row_end(roi, mcu_height_px, mcu_rows);
    let last_output_mcu_row = last_mcu_row_for_rect(roi, mcu_height_px, mcu_rows);

    let region_layout = Fast420RegionLayout::new(width as usize, roi);

    pool.prepare_for(
        plan,
        region_layout.stripe_mcus_per_row,
        DownscaleFactor::Full.output_block_size(),
    );

    let mut crop_rows = pool.take_sink_rows(region_layout.row_width());
    let result = (|| {
        let mut br = BitReader::new(scan_bytes);
        let mut coeff = CoefficientBlock::default();
        let mut pixels = [0u8; 64];
        let (y_comp, cb_comp, cr_comp) = fast_tile_components(plan);
        let mut y_dc = 0i32;
        let mut cb_dc = 0i32;
        let mut cr_dc = 0i32;
        for _ in 0..first_decode_mcu_row * mcus_per_row {
            skip_mcu_fast_tile_420(
                y_comp, cb_comp, cr_comp, &mut br, &mut y_dc, &mut cb_dc, &mut cr_dc,
            )?;
        }

        let ScratchPool {
            stripe_a,
            stripe_b,
            stripe_c,
            ..
        } = pool;
        let mut prev_stripe: &mut StripeBuffer = stripe_a;
        let mut curr_stripe: &mut StripeBuffer = stripe_b;
        let mut next_stripe: &mut StripeBuffer = stripe_c;

        decode_mcu_row_fast_tile_420(
            y_comp,
            cb_comp,
            cr_comp,
            backend,
            &mut br,
            &mut y_dc,
            &mut cb_dc,
            &mut cr_dc,
            &mut coeff,
            &mut pixels,
            mcus_per_row,
            region_layout.stripe_mcu_start,
            region_layout.stripe_mcus_per_row,
            curr_stripe,
        )?;

        let mut has_prev = false;
        for my in first_decode_mcu_row + 1..decode_mcu_row_end {
            decode_mcu_row_fast_tile_420(
                y_comp,
                cb_comp,
                cr_comp,
                backend,
                &mut br,
                &mut y_dc,
                &mut cb_dc,
                &mut cr_dc,
                &mut coeff,
                &mut pixels,
                mcus_per_row,
                region_layout.stripe_mcu_start,
                region_layout.stripe_mcus_per_row,
                next_stripe,
            )?;
            if mcu_row_intersects_rect(my - 1, mcu_height_px, roi) {
                emit_stripe_rgb_420_region(
                    plan,
                    backend,
                    has_prev.then_some(&*prev_stripe),
                    curr_stripe,
                    Some(&*next_stripe),
                    my - 1,
                    writer,
                    roi,
                    region_layout,
                    &mut crop_rows,
                    DownscaleFactor::Full,
                )?;
            }
            core::mem::swap(&mut prev_stripe, &mut curr_stripe);
            core::mem::swap(&mut curr_stripe, &mut next_stripe);
            has_prev = true;
        }

        let curr_mcu_row = decode_mcu_row_end - 1;
        if curr_mcu_row <= last_output_mcu_row
            && mcu_row_intersects_rect(curr_mcu_row, mcu_height_px, roi)
        {
            emit_stripe_rgb_420_region(
                plan,
                backend,
                has_prev.then_some(&*prev_stripe),
                curr_stripe,
                None,
                curr_mcu_row,
                writer,
                roi,
                region_layout,
                &mut crop_rows,
                DownscaleFactor::Full,
            )?;
        }
        finish_scan(&mut br, decode_mcu_row_end == mcu_rows)
    })();
    pool.restore_sink_rows(crop_rows);
    result
}

pub(crate) fn decode_scan_fast_tile_rgb_region_scaled<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    roi: Rect,
    downscale: DownscaleFactor,
) -> Result<Vec<Warning>, JpegError> {
    debug_assert!(plan.matches_fast_tile_shape());
    debug_assert!(downscale != DownscaleFactor::Full);

    let (width, height) = scaled_dimensions(plan.dimensions, downscale);
    let max_h = plan.sampling.max_h as u32;
    let max_v = plan.sampling.max_v as u32;
    let block_size = downscale.output_block_size();
    let mcu_width_px = block_size * max_h;
    let mcu_height_px = block_size * max_v;
    let mcus_per_row = width.div_ceil(mcu_width_px);
    let mcu_rows = height.div_ceil(mcu_height_px);
    let first_decode_mcu_row = fast420_first_decode_mcu_row(roi, mcu_height_px);
    let decode_mcu_row_end = fast420_decode_mcu_row_end(roi, mcu_height_px, mcu_rows);
    let last_output_mcu_row = last_mcu_row_for_rect(roi, mcu_height_px, mcu_rows);

    let region_layout = Fast420RegionLayout::new_for_mcu_width(width as usize, roi, mcu_width_px);

    pool.prepare_for(plan, region_layout.stripe_mcus_per_row, block_size);

    let mut crop_rows = pool.take_sink_rows(region_layout.row_width());
    let result = (|| {
        let mut br = BitReader::new(scan_bytes);
        let mut coeff = CoefficientBlock::default();
        let ScratchPool {
            prev_dc,
            stripe_a,
            stripe_b,
            stripe_c,
            ..
        } = pool;
        let mut pixels_4x4 = [0u8; 16];
        let mut pixels_2x2 = [0u8; 4];
        let (y_dc_slice, rest_dc) = prev_dc.split_at_mut(1);
        let (cb_dc_slice, cr_dc_slice) = rest_dc.split_at_mut(1);
        let y_dc = &mut y_dc_slice[0];
        let cb_dc = &mut cb_dc_slice[0];
        let cr_dc = &mut cr_dc_slice[0];
        for _ in 0..first_decode_mcu_row * mcus_per_row {
            skip_mcu_fast_tile_420(
                &plan.components[0],
                &plan.components[1],
                &plan.components[2],
                &mut br,
                y_dc,
                cb_dc,
                cr_dc,
            )?;
        }

        let mut prev_stripe: &mut StripeBuffer = stripe_a;
        let mut curr_stripe: &mut StripeBuffer = stripe_b;
        let mut next_stripe: &mut StripeBuffer = stripe_c;

        decode_mcu_row_fast_tile_420_scaled(
            &plan.components[0],
            &plan.components[1],
            &plan.components[2],
            &mut br,
            y_dc,
            cb_dc,
            cr_dc,
            &mut coeff,
            downscale,
            &mut pixels_4x4,
            &mut pixels_2x2,
            mcus_per_row,
            region_layout.stripe_mcu_start,
            region_layout.stripe_mcus_per_row,
            curr_stripe,
        )?;

        let mut has_prev = false;
        for my in first_decode_mcu_row + 1..decode_mcu_row_end {
            decode_mcu_row_fast_tile_420_scaled(
                &plan.components[0],
                &plan.components[1],
                &plan.components[2],
                &mut br,
                y_dc,
                cb_dc,
                cr_dc,
                &mut coeff,
                downscale,
                &mut pixels_4x4,
                &mut pixels_2x2,
                mcus_per_row,
                region_layout.stripe_mcu_start,
                region_layout.stripe_mcus_per_row,
                next_stripe,
            )?;
            if mcu_row_intersects_rect(my - 1, mcu_height_px, roi) {
                emit_stripe_rgb_420_region(
                    plan,
                    backend,
                    has_prev.then_some(&*prev_stripe),
                    curr_stripe,
                    Some(&*next_stripe),
                    my - 1,
                    writer,
                    roi,
                    region_layout,
                    &mut crop_rows,
                    downscale,
                )?;
            }
            core::mem::swap(&mut prev_stripe, &mut curr_stripe);
            core::mem::swap(&mut curr_stripe, &mut next_stripe);
            has_prev = true;
        }

        let curr_mcu_row = decode_mcu_row_end - 1;
        if curr_mcu_row <= last_output_mcu_row
            && mcu_row_intersects_rect(curr_mcu_row, mcu_height_px, roi)
        {
            emit_stripe_rgb_420_region(
                plan,
                backend,
                has_prev.then_some(&*prev_stripe),
                curr_stripe,
                None,
                curr_mcu_row,
                writer,
                roi,
                region_layout,
                &mut crop_rows,
                downscale,
            )?;
        }
        finish_scan(&mut br, decode_mcu_row_end == mcu_rows)
    })();
    pool.restore_sink_rows(crop_rows);
    result
}

fn deposit_block(plane: &mut [u8], stride: usize, x: u32, y: u32, block: &[u8; 64]) {
    let dst_row_start = (y as usize) * stride + (x as usize);
    debug_assert!(x as usize + 8 <= stride);
    debug_assert!(plane.len() >= dst_row_start + stride.saturating_mul(7) + 8);
    let mut dst = unsafe { plane.as_mut_ptr().add(dst_row_start) };
    let mut src = block.as_ptr();
    for _ in 0..8 {
        unsafe {
            ptr::copy_nonoverlapping(src, dst, 8);
            dst = dst.add(stride);
            src = src.add(8);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_stripe_rgb_420_region<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    prev: Option<&StripeBuffer>,
    curr: &StripeBuffer,
    next: Option<&StripeBuffer>,
    stripe_index: u32,
    writer: &mut W,
    roi: Rect,
    region_layout: Fast420RegionLayout,
    crop_rows: &mut SinkRows,
    downscale: DownscaleFactor,
) -> Result<(), JpegError> {
    let max_v = plan.sampling.max_v as u32;
    let mcu_height_px = downscale.output_block_size() * max_v;
    let y_start = stripe_index * mcu_height_px;
    let (_, scaled_height) = scaled_dimensions(plan.dimensions, downscale);
    let y_end = (y_start + mcu_height_px).min(scaled_height);
    let stripe_rows = (y_end - y_start) as usize;

    if stripe_rows == 0 {
        return Ok(());
    }

    let row_width = region_layout.row_width();
    let chroma_width = row_width.div_ceil(2);
    let row_len = row_width * 3;
    let crop_width = region_layout.crop_end - region_layout.crop_start;
    let crop_len = crop_width * 3;
    let use_direct_crop = downscale == DownscaleFactor::Full
        && backend.prefers_cropped_420_region(row_width, crop_width);
    let mut local_y = 0usize;
    while local_y < stripe_rows {
        let next_local_y = local_y + 1;
        let global_y = y_start + local_y as u32;
        let top_in = global_y >= roi.y && global_y < roi.y + roi.h;
        let bottom_in =
            next_local_y < stripe_rows && global_y + 1 >= roi.y && global_y + 1 < roi.y + roi.h;
        if !top_in && !bottom_in {
            local_y += 2;
            continue;
        }

        let y_top = &curr.row(0, local_y)[..row_width];
        let y_bottom =
            (next_local_y < stripe_rows).then(|| &curr.row(0, next_local_y)[..row_width]);
        let chroma_y = (local_y / 2).min(curr.row_count(1).saturating_sub(1));
        let (prev_cb, curr_cb, next_cb) = component_row_triplet(
            prev.map(|stripe| stripe.plane(1)),
            curr.plane(1),
            next.map(|stripe| stripe.plane(1)),
            chroma_y,
        );
        let (prev_cr, curr_cr, next_cr) = component_row_triplet(
            prev.map(|stripe| stripe.plane(2)),
            curr.plane(2),
            next.map(|stripe| stripe.plane(2)),
            chroma_y,
        );

        if use_direct_crop {
            match (top_in, bottom_in) {
                (true, true) => {
                    writer.with_rgb_rows(global_y - roi.y, 2, |dst_top, dst_bottom| {
                        backend.fill_rgb_row_pair_from_420_cropped(
                            y_top,
                            y_bottom,
                            &prev_cb[..chroma_width],
                            &curr_cb[..chroma_width],
                            &next_cb[..chroma_width],
                            &prev_cr[..chroma_width],
                            &curr_cr[..chroma_width],
                            &next_cr[..chroma_width],
                            region_layout.crop_start,
                            crop_width,
                            dst_top,
                            dst_bottom,
                        );
                        Ok(())
                    })?;
                }
                (true, false) => {
                    writer.with_rgb_rows(global_y - roi.y, 1, |dst, _| {
                        backend.fill_rgb_row_pair_from_420_cropped(
                            y_top,
                            None,
                            &prev_cb[..chroma_width],
                            &curr_cb[..chroma_width],
                            &next_cb[..chroma_width],
                            &prev_cr[..chroma_width],
                            &curr_cr[..chroma_width],
                            &next_cr[..chroma_width],
                            region_layout.crop_start,
                            crop_width,
                            dst,
                            None,
                        );
                        Ok(())
                    })?;
                }
                (false, true) => {
                    let y_bottom = y_bottom.expect("bottom ROI row requires a decoded bottom row");
                    writer.with_rgb_rows(global_y + 1 - roi.y, 1, |dst, _| {
                        backend.fill_rgb_row_pair_from_420_cropped(
                            y_top,
                            Some(y_bottom),
                            &prev_cb[..chroma_width],
                            &curr_cb[..chroma_width],
                            &next_cb[..chroma_width],
                            &prev_cr[..chroma_width],
                            &curr_cr[..chroma_width],
                            &next_cr[..chroma_width],
                            region_layout.crop_start,
                            crop_width,
                            &mut crop_rows.top_row[..crop_len],
                            Some(dst),
                        );
                        Ok(())
                    })?;
                }
                (false, false) => unreachable!("ROI row pair must intersect at least one row"),
            }
            local_y += 2;
            continue;
        }

        backend.fill_rgb_row_pair_from_420(
            y_top,
            y_bottom,
            &prev_cb[..chroma_width],
            &curr_cb[..chroma_width],
            &next_cb[..chroma_width],
            &prev_cr[..chroma_width],
            &curr_cr[..chroma_width],
            &next_cr[..chroma_width],
            &mut crop_rows.top_row[..row_len],
            y_bottom
                .as_ref()
                .map(|_| &mut crop_rows.bottom_row[..row_len]),
        );

        let x0 = region_layout.crop_start * 3;
        let x1 = region_layout.crop_end * 3;
        match (top_in, bottom_in) {
            (true, true) => {
                writer.with_rgb_rows(global_y - roi.y, 2, |dst_top, dst_bottom| {
                    dst_top.copy_from_slice(&crop_rows.top_row[x0..x1]);
                    dst_bottom
                        .expect("row_count=2 supplies bottom row")
                        .copy_from_slice(&crop_rows.bottom_row[x0..x1]);
                    Ok(())
                })?;
            }
            (true, false) => {
                writer.with_rgb_rows(global_y - roi.y, 1, |dst, _| {
                    dst.copy_from_slice(&crop_rows.top_row[x0..x1]);
                    Ok(())
                })?;
            }
            (false, true) => {
                writer.with_rgb_rows(global_y + 1 - roi.y, 1, |dst, _| {
                    dst.copy_from_slice(&crop_rows.bottom_row[x0..x1]);
                    Ok(())
                })?;
            }
            (false, false) => unreachable!("ROI row pair must intersect at least one row"),
        }

        local_y += 2;
    }

    Ok(())
}

fn deposit_block_4x4(plane: &mut [u8], stride: usize, x: u32, y: u32, block: &[u8; 16]) {
    let x = x as usize;
    let y = y as usize;
    for by in 0..4 {
        let dst_start = (y + by) * stride + x;
        plane[dst_start..dst_start + 4].copy_from_slice(&block[by * 4..by * 4 + 4]);
    }
}

fn deposit_block_2x2(plane: &mut [u8], stride: usize, x: u32, y: u32, block: [u8; 4]) {
    let x = x as usize;
    let y = y as usize;
    let top = y * stride + x;
    let bottom = top + stride;
    plane[top] = block[0];
    plane[top + 1] = block[1];
    plane[bottom] = block[2];
    plane[bottom + 1] = block[3];
}

fn deposit_block_1x1(plane: &mut [u8], stride: usize, x: u32, y: u32, pixel: u8) {
    let dst = (y as usize) * stride + (x as usize);
    plane[dst] = pixel;
}

#[allow(clippy::too_many_arguments)]
fn decode_mcu_row(
    plan: &PreparedDecodePlan,
    backend: Backend,
    br: &mut BitReader<'_>,
    prev_dc: &mut [i32],
    coeff: &mut CoefficientBlock,
    pixels: &mut [u8; 64],
    downscale: DownscaleFactor,
    output_rect: Rect,
    full_output_rect: bool,
    mcu_y: u32,
    stripe_mcu_start: u32,
    stripe_mcus_per_row: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    stripe: &mut StripeBuffer,
    restart: u16,
    mcus_since_restart: &mut u32,
    expected_rst: &mut u8,
) -> Result<(), JpegError> {
    let stripe_mcu_end = stripe_mcu_start + stripe_mcus_per_row;
    let block_size = downscale.output_block_size();
    let mut pixels_4x4 = [0u8; 16];
    let mut pixels_2x2 = [0u8; 4];
    for mx in 0..mcus_per_row {
        if restart > 0 && *mcus_since_restart == u32::from(restart) {
            let _ = br.ensure_bits(1);
            let marker = br.take_marker().ok_or(JpegError::UnexpectedEoi {
                mcu_at: mcu_y * mcus_per_row + mx,
                mcu_total: mcu_rows * mcus_per_row,
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
            prev_dc.fill(0);
            *mcus_since_restart = 0;
        }

        for comp in &plan.components {
            let plane_idx = comp.output_index;
            let in_region = mx >= stripe_mcu_start && mx < stripe_mcu_end;
            let local_mcu_x0_px =
                mx.saturating_sub(stripe_mcu_start) * u32::from(comp.h) * block_size;
            for vy in 0..comp.v as u32 {
                for vx in 0..comp.h as u32 {
                    let should_output = in_region
                        && (full_output_rect
                            || component_block_intersects_rect(
                                plan,
                                comp,
                                downscale,
                                mx,
                                mcu_y,
                                vx,
                                vy,
                                output_rect,
                            ));
                    if !should_output {
                        skip_block(br, &comp.dc_table, &comp.ac_table, &mut prev_dc[plane_idx])?;
                        continue;
                    }

                    let activity = decode_block_with_activity(
                        br,
                        &comp.dc_table,
                        &comp.ac_table,
                        &mut prev_dc[plane_idx],
                        &comp.quant,
                        coeff,
                    )?;
                    let block_x = local_mcu_x0_px + vx * block_size;
                    let block_y = vy * block_size;
                    match downscale {
                        DownscaleFactor::Full => {
                            match activity {
                                BlockActivity::DcOnly => {
                                    crate::idct::idct_islow_dc_only(coeff.dc_coeff(), pixels);
                                }
                                BlockActivity::BottomHalfZero => {
                                    backend.idct_bottom_half_zero(coeff.coefficients(), pixels);
                                }
                                BlockActivity::General => {
                                    backend.idct(coeff.coefficients(), pixels);
                                }
                            }
                            deposit_block(
                                &mut stripe.planes[plane_idx],
                                stripe.plane_strides[plane_idx],
                                block_x,
                                block_y,
                                pixels,
                            );
                        }
                        DownscaleFactor::Half => {
                            if activity == BlockActivity::DcOnly {
                                downscale::idct_islow_4x4_dc_only(
                                    coeff.dc_coeff(),
                                    &mut pixels_4x4,
                                );
                            } else {
                                downscale::idct_islow_4x4(coeff.coefficients(), &mut pixels_4x4);
                            }
                            deposit_block_4x4(
                                &mut stripe.planes[plane_idx],
                                stripe.plane_strides[plane_idx],
                                block_x,
                                block_y,
                                &pixels_4x4,
                            );
                        }
                        DownscaleFactor::Quarter => {
                            if activity == BlockActivity::DcOnly {
                                downscale::idct_islow_2x2_dc_only(
                                    coeff.dc_coeff(),
                                    &mut pixels_2x2,
                                );
                            } else {
                                downscale::idct_islow_2x2(coeff.coefficients(), &mut pixels_2x2);
                            }
                            deposit_block_2x2(
                                &mut stripe.planes[plane_idx],
                                stripe.plane_strides[plane_idx],
                                block_x,
                                block_y,
                                pixels_2x2,
                            );
                        }
                        DownscaleFactor::Eighth => {
                            let pixel = downscale::idct_islow_1x1(coeff.coefficients());
                            deposit_block_1x1(
                                &mut stripe.planes[plane_idx],
                                stripe.plane_strides[plane_idx],
                                block_x,
                                block_y,
                                pixel,
                            );
                        }
                    }
                }
            }
        }
        *mcus_since_restart += 1;
    }

    Ok(())
}

fn fast_tile_components(
    plan: &PreparedDecodePlan,
) -> (
    &PreparedComponentPlan,
    &PreparedComponentPlan,
    &PreparedComponentPlan,
) {
    debug_assert!(plan.matches_fast_tile_shape());
    (
        &plan.components[0],
        &plan.components[1],
        &plan.components[2],
    )
}

fn fast_rgb444_components(
    plan: &PreparedDecodePlan,
) -> (
    &PreparedComponentPlan,
    &PreparedComponentPlan,
    &PreparedComponentPlan,
) {
    debug_assert!(plan.matches_fast_rgb444_shape());
    (
        &plan.components[0],
        &plan.components[1],
        &plan.components[2],
    )
}

pub(crate) fn decode_scan_fast_rgb_444<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
) -> Result<Vec<Warning>, JpegError> {
    debug_assert!(plan.matches_fast_rgb444_shape());

    let (width, height) = plan.dimensions;
    let mcus_per_row = width.div_ceil(8);
    let mcu_rows = height.div_ceil(8);

    pool.prepare_for(
        plan,
        mcus_per_row,
        DownscaleFactor::Full.output_block_size(),
    );

    let mut br = BitReader::new(scan_bytes);
    let mut coeff = CoefficientBlock::default();
    let mut pixels = [0u8; 64];
    let (y_comp, cb_comp, cr_comp) = fast_rgb444_components(plan);
    let mut y_dc = 0i32;
    let mut cb_dc = 0i32;
    let mut cr_dc = 0i32;
    let restart = plan.restart_interval.unwrap_or(0);
    let mut mcus_since_restart = 0u32;
    let mut expected_rst = 0u8;
    let stripe = &mut pool.stripe_a;

    for my in 0..mcu_rows {
        decode_mcu_row_fast_rgb_444(
            y_comp,
            cb_comp,
            cr_comp,
            backend,
            &mut br,
            &mut y_dc,
            &mut cb_dc,
            &mut cr_dc,
            &mut coeff,
            &mut pixels,
            my,
            mcus_per_row,
            mcu_rows,
            stripe,
            restart,
            &mut mcus_since_restart,
            &mut expected_rst,
        )?;
        emit_stripe_rgb_444(plan, backend, stripe, my, writer)?;
    }

    let mut warnings = Vec::new();
    match br.take_marker() {
        Some(0xD9) => Ok(warnings),
        Some(found) => Err(JpegError::UnexpectedMarker {
            offset: br.position().saturating_sub(2),
            expected: crate::error::MarkerKind::Eoi,
            found,
        }),
        None => {
            warnings.push(Warning::MissingEoi);
            Ok(warnings)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_mcu_row_fast_rgb_444(
    y_comp: &PreparedComponentPlan,
    cb_comp: &PreparedComponentPlan,
    cr_comp: &PreparedComponentPlan,
    backend: Backend,
    br: &mut BitReader<'_>,
    y_dc: &mut i32,
    cb_dc: &mut i32,
    cr_dc: &mut i32,
    coeff: &mut CoefficientBlock,
    pixels: &mut [u8; 64],
    mcu_y: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    stripe: &mut StripeBuffer,
    restart: u16,
    mcus_since_restart: &mut u32,
    expected_rst: &mut u8,
) -> Result<(), JpegError> {
    for mx in 0..mcus_per_row {
        if restart > 0 && *mcus_since_restart == u32::from(restart) {
            let _ = br.ensure_bits(1);
            let marker = br.take_marker().ok_or(JpegError::UnexpectedEoi {
                mcu_at: mcu_y * mcus_per_row + mx,
                mcu_total: mcu_rows * mcus_per_row,
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
            *y_dc = 0;
            *cb_dc = 0;
            *cr_dc = 0;
            *mcus_since_restart = 0;
        }

        let block_x = mx * 8;

        let y_activity = decode_block_with_activity(
            br,
            &y_comp.dc_table,
            &y_comp.ac_table,
            y_dc,
            y_comp.quant.as_ref(),
            coeff,
        )?;
        match y_activity {
            BlockActivity::DcOnly => crate::idct::idct_islow_dc_only(coeff.dc_coeff(), pixels),
            BlockActivity::BottomHalfZero => {
                backend.idct_bottom_half_zero(coeff.coefficients(), pixels);
            }
            BlockActivity::General => backend.idct(coeff.coefficients(), pixels),
        }
        deposit_block(
            &mut stripe.planes[0],
            stripe.plane_strides[0],
            block_x,
            0,
            pixels,
        );

        let cb_activity = decode_block_with_activity(
            br,
            &cb_comp.dc_table,
            &cb_comp.ac_table,
            cb_dc,
            cb_comp.quant.as_ref(),
            coeff,
        )?;
        match cb_activity {
            BlockActivity::DcOnly => crate::idct::idct_islow_dc_only(coeff.dc_coeff(), pixels),
            BlockActivity::BottomHalfZero => {
                backend.idct_bottom_half_zero(coeff.coefficients(), pixels);
            }
            BlockActivity::General => backend.idct(coeff.coefficients(), pixels),
        }
        deposit_block(
            &mut stripe.planes[1],
            stripe.plane_strides[1],
            block_x,
            0,
            pixels,
        );

        let cr_activity = decode_block_with_activity(
            br,
            &cr_comp.dc_table,
            &cr_comp.ac_table,
            cr_dc,
            cr_comp.quant.as_ref(),
            coeff,
        )?;
        match cr_activity {
            BlockActivity::DcOnly => crate::idct::idct_islow_dc_only(coeff.dc_coeff(), pixels),
            BlockActivity::BottomHalfZero => {
                backend.idct_bottom_half_zero(coeff.coefficients(), pixels);
            }
            BlockActivity::General => backend.idct(coeff.coefficients(), pixels),
        }
        deposit_block(
            &mut stripe.planes[2],
            stripe.plane_strides[2],
            block_x,
            0,
            pixels,
        );

        *mcus_since_restart += 1;
    }

    Ok(())
}

fn emit_stripe_rgb_444<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    stripe: &StripeBuffer,
    stripe_index: u32,
    writer: &mut W,
) -> Result<(), JpegError> {
    let (width, height) = plan.dimensions;
    let y_start = stripe_index * 8;
    let stripe_rows = (height.saturating_sub(y_start)).min(8) as usize;
    let width = width as usize;
    let y_stride = stripe.plane_strides[0];
    let cb_stride = stripe.plane_strides[1];
    let cr_stride = stripe.plane_strides[2];

    let mut local_y = 0usize;
    while local_y + 1 < stripe_rows {
        let y_top_start = local_y * y_stride;
        let y_bottom_start = y_top_start + y_stride;
        let cb_top_start = local_y * cb_stride;
        let cb_bottom_start = cb_top_start + cb_stride;
        let cr_top_start = local_y * cr_stride;
        let cr_bottom_start = cr_top_start + cr_stride;
        writer.with_rgb_rows(y_start + local_y as u32, 2, |dst_top, dst_bottom| {
            backend.fill_rgb_row_from_ycbcr(
                &stripe.planes[0][y_top_start..y_top_start + width],
                &stripe.planes[1][cb_top_start..cb_top_start + width],
                &stripe.planes[2][cr_top_start..cr_top_start + width],
                dst_top,
            );
            backend.fill_rgb_row_from_ycbcr(
                &stripe.planes[0][y_bottom_start..y_bottom_start + width],
                &stripe.planes[1][cb_bottom_start..cb_bottom_start + width],
                &stripe.planes[2][cr_bottom_start..cr_bottom_start + width],
                dst_bottom.expect("row_count=2 supplies bottom row"),
            );
            Ok(())
        })?;
        local_y += 2;
    }

    if local_y < stripe_rows {
        let y_row_start = local_y * y_stride;
        let cb_row_start = local_y * cb_stride;
        let cr_row_start = local_y * cr_stride;
        writer.with_rgb_rows(y_start + local_y as u32, 1, |dst, _| {
            backend.fill_rgb_row_from_ycbcr(
                &stripe.planes[0][y_row_start..y_row_start + width],
                &stripe.planes[1][cb_row_start..cb_row_start + width],
                &stripe.planes[2][cr_row_start..cr_row_start + width],
                dst,
            );
            Ok(())
        })?;
    }

    Ok(())
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn skip_mcu_fast_tile_420(
    y_comp: &PreparedComponentPlan,
    cb_comp: &PreparedComponentPlan,
    cr_comp: &PreparedComponentPlan,
    br: &mut BitReader<'_>,
    y_dc: &mut i32,
    cb_dc: &mut i32,
    cr_dc: &mut i32,
) -> Result<(), JpegError> {
    for _ in 0..4 {
        skip_block(br, &y_comp.dc_table, &y_comp.ac_table, y_dc)?;
    }
    skip_block(br, &cb_comp.dc_table, &cb_comp.ac_table, cb_dc)?;
    skip_block(br, &cr_comp.dc_table, &cr_comp.ac_table, cr_dc)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_scaled_block_to_plane(
    comp: &PreparedComponentPlan,
    br: &mut BitReader<'_>,
    prev_dc: &mut i32,
    coeff: &mut CoefficientBlock,
    downscale: DownscaleFactor,
    pixels_4x4: &mut [u8; 16],
    pixels_2x2: &mut [u8; 4],
    plane: &mut [u8],
    stride: usize,
    x: u32,
    y: u32,
) -> Result<(), JpegError> {
    let keep = match downscale {
        DownscaleFactor::Full => unreachable!("scaled block path excludes full-size decode"),
        DownscaleFactor::Half => ReducedIdctCoefficients::Half,
        DownscaleFactor::Quarter => ReducedIdctCoefficients::Quarter,
        DownscaleFactor::Eighth => {
            decode_block_for_1x1_idct(
                br,
                &comp.dc_table,
                &comp.ac_table,
                prev_dc,
                comp.quant.as_ref(),
                coeff,
            )?;
            let pixel = downscale::idct_islow_1x1(coeff.coefficients());
            deposit_block_1x1(plane, stride, x, y, pixel);
            return Ok(());
        }
    };
    let dc_only = decode_block_for_reduced_idct(
        br,
        &comp.dc_table,
        &comp.ac_table,
        prev_dc,
        comp.quant.as_ref(),
        coeff,
        keep,
    )?;
    match downscale {
        DownscaleFactor::Full => unreachable!("scaled block path excludes full-size decode"),
        DownscaleFactor::Half => {
            if dc_only {
                downscale::idct_islow_4x4_dc_only(coeff.dc_coeff(), pixels_4x4);
            } else {
                downscale::idct_islow_4x4(coeff.coefficients(), pixels_4x4);
            }
            deposit_block_4x4(plane, stride, x, y, pixels_4x4);
        }
        DownscaleFactor::Quarter => {
            if dc_only {
                downscale::idct_islow_2x2_dc_only(coeff.dc_coeff(), pixels_2x2);
            } else {
                downscale::idct_islow_2x2(coeff.coefficients(), pixels_2x2);
            }
            deposit_block_2x2(plane, stride, x, y, *pixels_2x2);
        }
        DownscaleFactor::Eighth => {
            let pixel = downscale::idct_islow_1x1(coeff.coefficients());
            deposit_block_1x1(plane, stride, x, y, pixel);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_quarter_block_to_plane(
    comp: &PreparedComponentPlan,
    br: &mut BitReader<'_>,
    prev_dc: &mut i32,
    coeff: &mut CoefficientBlock,
    pixels_2x2: &mut [u8; 4],
    plane: &mut [u8],
    stride: usize,
    x: u32,
    y: u32,
) -> Result<(), JpegError> {
    let dc_only = decode_block_for_reduced_idct(
        br,
        &comp.dc_table,
        &comp.ac_table,
        prev_dc,
        comp.quant.as_ref(),
        coeff,
        ReducedIdctCoefficients::Quarter,
    )?;
    if dc_only {
        downscale::idct_islow_2x2_dc_only(coeff.dc_coeff(), pixels_2x2);
    } else {
        downscale::idct_islow_2x2(coeff.coefficients(), pixels_2x2);
    }
    deposit_block_2x2(plane, stride, x, y, *pixels_2x2);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_eighth_block_to_plane(
    comp: &PreparedComponentPlan,
    br: &mut BitReader<'_>,
    prev_dc: &mut i32,
    coeff: &mut CoefficientBlock,
    plane: &mut [u8],
    stride: usize,
    x: u32,
    y: u32,
) -> Result<(), JpegError> {
    decode_block_for_1x1_idct(
        br,
        &comp.dc_table,
        &comp.ac_table,
        prev_dc,
        comp.quant.as_ref(),
        coeff,
    )?;
    let pixel = downscale::idct_islow_1x1(coeff.coefficients());
    deposit_block_1x1(plane, stride, x, y, pixel);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_mcu_row_fast_tile_420_scaled(
    y_comp: &PreparedComponentPlan,
    cb_comp: &PreparedComponentPlan,
    cr_comp: &PreparedComponentPlan,
    br: &mut BitReader<'_>,
    y_dc: &mut i32,
    cb_dc: &mut i32,
    cr_dc: &mut i32,
    coeff: &mut CoefficientBlock,
    downscale: DownscaleFactor,
    pixels_4x4: &mut [u8; 16],
    pixels_2x2: &mut [u8; 4],
    mcus_per_row: u32,
    stripe_mcu_start: u32,
    stripe_mcus_per_row: u32,
    stripe: &mut StripeBuffer,
) -> Result<(), JpegError> {
    if downscale == DownscaleFactor::Quarter {
        return decode_mcu_row_fast_tile_420_quarter(
            y_comp,
            cb_comp,
            cr_comp,
            br,
            y_dc,
            cb_dc,
            cr_dc,
            coeff,
            pixels_2x2,
            mcus_per_row,
            stripe_mcu_start,
            stripe_mcus_per_row,
            stripe,
        );
    }
    if downscale == DownscaleFactor::Eighth {
        return decode_mcu_row_fast_tile_420_eighth(
            y_comp,
            cb_comp,
            cr_comp,
            br,
            y_dc,
            cb_dc,
            cr_dc,
            coeff,
            mcus_per_row,
            stripe_mcu_start,
            stripe_mcus_per_row,
            stripe,
        );
    }

    let block_size = downscale.output_block_size();
    let stripe_mcu_end = stripe_mcu_start + stripe_mcus_per_row;
    let y_stride = stripe.plane_strides[0];
    let cb_stride = stripe.plane_strides[1];
    let cr_stride = stripe.plane_strides[2];

    for mx in 0..mcus_per_row {
        let in_region = mx >= stripe_mcu_start && mx < stripe_mcu_end;
        if !in_region {
            skip_mcu_fast_tile_420(y_comp, cb_comp, cr_comp, br, y_dc, cb_dc, cr_dc)?;
            continue;
        }

        let local_mx = mx - stripe_mcu_start;
        let y_x = local_mx * 2 * block_size;
        let c_x = local_mx * block_size;
        decode_scaled_block_to_plane(
            y_comp,
            br,
            y_dc,
            coeff,
            downscale,
            pixels_4x4,
            pixels_2x2,
            &mut stripe.planes[0],
            y_stride,
            y_x,
            0,
        )?;
        decode_scaled_block_to_plane(
            y_comp,
            br,
            y_dc,
            coeff,
            downscale,
            pixels_4x4,
            pixels_2x2,
            &mut stripe.planes[0],
            y_stride,
            y_x + block_size,
            0,
        )?;
        decode_scaled_block_to_plane(
            y_comp,
            br,
            y_dc,
            coeff,
            downscale,
            pixels_4x4,
            pixels_2x2,
            &mut stripe.planes[0],
            y_stride,
            y_x,
            block_size,
        )?;
        decode_scaled_block_to_plane(
            y_comp,
            br,
            y_dc,
            coeff,
            downscale,
            pixels_4x4,
            pixels_2x2,
            &mut stripe.planes[0],
            y_stride,
            y_x + block_size,
            block_size,
        )?;
        decode_scaled_block_to_plane(
            cb_comp,
            br,
            cb_dc,
            coeff,
            downscale,
            pixels_4x4,
            pixels_2x2,
            &mut stripe.planes[1],
            cb_stride,
            c_x,
            0,
        )?;
        decode_scaled_block_to_plane(
            cr_comp,
            br,
            cr_dc,
            coeff,
            downscale,
            pixels_4x4,
            pixels_2x2,
            &mut stripe.planes[2],
            cr_stride,
            c_x,
            0,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_mcu_row_fast_tile_420(
    y_comp: &PreparedComponentPlan,
    cb_comp: &PreparedComponentPlan,
    cr_comp: &PreparedComponentPlan,
    backend: Backend,
    br: &mut BitReader<'_>,
    y_dc: &mut i32,
    cb_dc: &mut i32,
    cr_dc: &mut i32,
    coeff: &mut CoefficientBlock,
    pixels: &mut [u8; 64],
    mcus_per_row: u32,
    stripe_mcu_start: u32,
    stripe_mcus_per_row: u32,
    stripe: &mut StripeBuffer,
) -> Result<(), JpegError> {
    let stripe_mcu_end = stripe_mcu_start + stripe_mcus_per_row;
    for mx in 0..mcus_per_row {
        let in_region = mx >= stripe_mcu_start && mx < stripe_mcu_end;
        if !in_region {
            for _ in 0..4 {
                skip_block(br, &y_comp.dc_table, &y_comp.ac_table, y_dc)?;
            }
            skip_block(br, &cb_comp.dc_table, &cb_comp.ac_table, cb_dc)?;
            skip_block(br, &cr_comp.dc_table, &cr_comp.ac_table, cr_dc)?;
            continue;
        }

        let local_mx = mx - stripe_mcu_start;
        let y_x = local_mx * 16;
        let c_x = local_mx * 8;

        let y0_activity = decode_block_with_activity(
            br,
            &y_comp.dc_table,
            &y_comp.ac_table,
            y_dc,
            y_comp.quant.as_ref(),
            coeff,
        )?;
        match y0_activity {
            BlockActivity::DcOnly => crate::idct::idct_islow_dc_only(coeff.dc_coeff(), pixels),
            BlockActivity::BottomHalfZero => {
                backend.idct_bottom_half_zero(coeff.coefficients(), pixels);
            }
            BlockActivity::General => backend.idct(coeff.coefficients(), pixels),
        }
        deposit_block(
            &mut stripe.planes[0],
            stripe.plane_strides[0],
            y_x,
            0,
            pixels,
        );

        let y1_activity = decode_block_with_activity(
            br,
            &y_comp.dc_table,
            &y_comp.ac_table,
            y_dc,
            y_comp.quant.as_ref(),
            coeff,
        )?;
        match y1_activity {
            BlockActivity::DcOnly => crate::idct::idct_islow_dc_only(coeff.dc_coeff(), pixels),
            BlockActivity::BottomHalfZero => {
                backend.idct_bottom_half_zero(coeff.coefficients(), pixels);
            }
            BlockActivity::General => backend.idct(coeff.coefficients(), pixels),
        }
        deposit_block(
            &mut stripe.planes[0],
            stripe.plane_strides[0],
            y_x + 8,
            0,
            pixels,
        );

        let y2_activity = decode_block_with_activity(
            br,
            &y_comp.dc_table,
            &y_comp.ac_table,
            y_dc,
            y_comp.quant.as_ref(),
            coeff,
        )?;
        match y2_activity {
            BlockActivity::DcOnly => crate::idct::idct_islow_dc_only(coeff.dc_coeff(), pixels),
            BlockActivity::BottomHalfZero => {
                backend.idct_bottom_half_zero(coeff.coefficients(), pixels);
            }
            BlockActivity::General => backend.idct(coeff.coefficients(), pixels),
        }
        deposit_block(
            &mut stripe.planes[0],
            stripe.plane_strides[0],
            y_x,
            8,
            pixels,
        );

        let y3_activity = decode_block_with_activity(
            br,
            &y_comp.dc_table,
            &y_comp.ac_table,
            y_dc,
            y_comp.quant.as_ref(),
            coeff,
        )?;
        match y3_activity {
            BlockActivity::DcOnly => crate::idct::idct_islow_dc_only(coeff.dc_coeff(), pixels),
            BlockActivity::BottomHalfZero => {
                backend.idct_bottom_half_zero(coeff.coefficients(), pixels);
            }
            BlockActivity::General => backend.idct(coeff.coefficients(), pixels),
        }
        deposit_block(
            &mut stripe.planes[0],
            stripe.plane_strides[0],
            y_x + 8,
            8,
            pixels,
        );

        let cb_activity = decode_block_with_activity(
            br,
            &cb_comp.dc_table,
            &cb_comp.ac_table,
            cb_dc,
            cb_comp.quant.as_ref(),
            coeff,
        )?;
        match cb_activity {
            BlockActivity::DcOnly => crate::idct::idct_islow_dc_only(coeff.dc_coeff(), pixels),
            BlockActivity::BottomHalfZero => {
                backend.idct_bottom_half_zero(coeff.coefficients(), pixels);
            }
            BlockActivity::General => backend.idct(coeff.coefficients(), pixels),
        }
        deposit_block(
            &mut stripe.planes[1],
            stripe.plane_strides[1],
            c_x,
            0,
            pixels,
        );

        let cr_activity = decode_block_with_activity(
            br,
            &cr_comp.dc_table,
            &cr_comp.ac_table,
            cr_dc,
            cr_comp.quant.as_ref(),
            coeff,
        )?;
        match cr_activity {
            BlockActivity::DcOnly => crate::idct::idct_islow_dc_only(coeff.dc_coeff(), pixels),
            BlockActivity::BottomHalfZero => {
                backend.idct_bottom_half_zero(coeff.coefficients(), pixels);
            }
            BlockActivity::General => backend.idct(coeff.coefficients(), pixels),
        }
        deposit_block(
            &mut stripe.planes[2],
            stripe.plane_strides[2],
            c_x,
            0,
            pixels,
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_mcu_row_fast_tile_420_eighth(
    y_comp: &PreparedComponentPlan,
    cb_comp: &PreparedComponentPlan,
    cr_comp: &PreparedComponentPlan,
    br: &mut BitReader<'_>,
    y_dc: &mut i32,
    cb_dc: &mut i32,
    cr_dc: &mut i32,
    coeff: &mut CoefficientBlock,
    mcus_per_row: u32,
    stripe_mcu_start: u32,
    stripe_mcus_per_row: u32,
    stripe: &mut StripeBuffer,
) -> Result<(), JpegError> {
    const BLOCK_SIZE: u32 = 1;
    let stripe_mcu_end = stripe_mcu_start + stripe_mcus_per_row;
    let y_stride = stripe.plane_strides[0];
    let cb_stride = stripe.plane_strides[1];
    let cr_stride = stripe.plane_strides[2];

    for mx in 0..mcus_per_row {
        let in_region = mx >= stripe_mcu_start && mx < stripe_mcu_end;
        if !in_region {
            skip_mcu_fast_tile_420(y_comp, cb_comp, cr_comp, br, y_dc, cb_dc, cr_dc)?;
            continue;
        }

        let local_mx = mx - stripe_mcu_start;
        let y_x = local_mx * 2 * BLOCK_SIZE;
        let c_x = local_mx * BLOCK_SIZE;
        decode_eighth_block_to_plane(
            y_comp,
            br,
            y_dc,
            coeff,
            &mut stripe.planes[0],
            y_stride,
            y_x,
            0,
        )?;
        decode_eighth_block_to_plane(
            y_comp,
            br,
            y_dc,
            coeff,
            &mut stripe.planes[0],
            y_stride,
            y_x + BLOCK_SIZE,
            0,
        )?;
        decode_eighth_block_to_plane(
            y_comp,
            br,
            y_dc,
            coeff,
            &mut stripe.planes[0],
            y_stride,
            y_x,
            BLOCK_SIZE,
        )?;
        decode_eighth_block_to_plane(
            y_comp,
            br,
            y_dc,
            coeff,
            &mut stripe.planes[0],
            y_stride,
            y_x + BLOCK_SIZE,
            BLOCK_SIZE,
        )?;
        decode_eighth_block_to_plane(
            cb_comp,
            br,
            cb_dc,
            coeff,
            &mut stripe.planes[1],
            cb_stride,
            c_x,
            0,
        )?;
        decode_eighth_block_to_plane(
            cr_comp,
            br,
            cr_dc,
            coeff,
            &mut stripe.planes[2],
            cr_stride,
            c_x,
            0,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_mcu_row_fast_tile_420_quarter(
    y_comp: &PreparedComponentPlan,
    cb_comp: &PreparedComponentPlan,
    cr_comp: &PreparedComponentPlan,
    br: &mut BitReader<'_>,
    y_dc: &mut i32,
    cb_dc: &mut i32,
    cr_dc: &mut i32,
    coeff: &mut CoefficientBlock,
    pixels_2x2: &mut [u8; 4],
    mcus_per_row: u32,
    stripe_mcu_start: u32,
    stripe_mcus_per_row: u32,
    stripe: &mut StripeBuffer,
) -> Result<(), JpegError> {
    const BLOCK_SIZE: u32 = 2;
    let stripe_mcu_end = stripe_mcu_start + stripe_mcus_per_row;
    let y_stride = stripe.plane_strides[0];
    let cb_stride = stripe.plane_strides[1];
    let cr_stride = stripe.plane_strides[2];

    for mx in 0..mcus_per_row {
        let in_region = mx >= stripe_mcu_start && mx < stripe_mcu_end;
        if !in_region {
            skip_mcu_fast_tile_420(y_comp, cb_comp, cr_comp, br, y_dc, cb_dc, cr_dc)?;
            continue;
        }

        let local_mx = mx - stripe_mcu_start;
        let y_x = local_mx * 2 * BLOCK_SIZE;
        let c_x = local_mx * BLOCK_SIZE;
        decode_quarter_block_to_plane(
            y_comp,
            br,
            y_dc,
            coeff,
            pixels_2x2,
            &mut stripe.planes[0],
            y_stride,
            y_x,
            0,
        )?;
        decode_quarter_block_to_plane(
            y_comp,
            br,
            y_dc,
            coeff,
            pixels_2x2,
            &mut stripe.planes[0],
            y_stride,
            y_x + BLOCK_SIZE,
            0,
        )?;
        decode_quarter_block_to_plane(
            y_comp,
            br,
            y_dc,
            coeff,
            pixels_2x2,
            &mut stripe.planes[0],
            y_stride,
            y_x,
            BLOCK_SIZE,
        )?;
        decode_quarter_block_to_plane(
            y_comp,
            br,
            y_dc,
            coeff,
            pixels_2x2,
            &mut stripe.planes[0],
            y_stride,
            y_x + BLOCK_SIZE,
            BLOCK_SIZE,
        )?;
        decode_quarter_block_to_plane(
            cb_comp,
            br,
            cb_dc,
            coeff,
            pixels_2x2,
            &mut stripe.planes[1],
            cb_stride,
            c_x,
            0,
        )?;
        decode_quarter_block_to_plane(
            cr_comp,
            br,
            cr_dc,
            coeff,
            pixels_2x2,
            &mut stripe.planes[2],
            cr_stride,
            c_x,
            0,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_stripe<W: OutputWriter>(
    plan: &PreparedDecodePlan,
    prev: Option<&StripeBuffer>,
    curr: &StripeBuffer,
    next: Option<&StripeBuffer>,
    stripe_index: u32,
    writer: &mut W,
    output_scratch: &mut OutputScratch<'_>,
    source_width: usize,
    downscale: DownscaleFactor,
) -> Result<(), JpegError> {
    let max_v = plan.sampling.max_v as u32;
    let mcu_height_px = downscale.output_block_size() * max_v;
    let y_start = stripe_index * mcu_height_px;
    let (_, scaled_height) = scaled_dimensions(plan.dimensions, downscale);
    let y_end = (y_start + mcu_height_px).min(scaled_height);
    let stripe_rows = (y_end - y_start) as usize;

    if stripe_rows == 0 {
        return Ok(());
    }

    let width = source_width;
    match plan.color_space {
        ColorSpace::Grayscale => {
            for local_y in 0..stripe_rows {
                let y_row = &curr.row(0, local_y)[..width];
                writer.write_gray_row(y_start + local_y as u32, y_row)?;
            }
        }
        ColorSpace::YCbCr => {
            let (cb_h, cb_v) = plan
                .sampling
                .component(1)
                .map(|(h, v)| (u32::from(h), u32::from(v)))
                .ok_or(JpegError::UnsupportedComponentCount { count: 1 })?;
            let (cr_h, cr_v) = plan
                .sampling
                .component(2)
                .map(|(h, v)| (u32::from(h), u32::from(v)))
                .ok_or(JpegError::UnsupportedComponentCount { count: 2 })?;

            let max_h = plan.sampling.max_h as u32;
            let max_v = plan.sampling.max_v as u32;

            if is_ycbcr_420(plan) {
                let OutputScratch::YCbCr420(scratch) = output_scratch else {
                    unreachable!("4:2:0 YCbCr requires dedicated scratch");
                };
                debug_assert!(
                    max_h == 2 && max_v == 2 && cb_h == 1 && cb_v == 1 && cr_h == 1 && cr_v == 1
                );

                let mut local_y = 0usize;
                while local_y < stripe_rows {
                    let y_top = &curr.row(0, local_y)[..width];
                    let next_local_y = local_y + 1;
                    let y_bottom =
                        (next_local_y < stripe_rows).then(|| &curr.row(0, next_local_y)[..width]);

                    upsample_420_pair(
                        prev,
                        curr,
                        next,
                        1,
                        local_y as u32,
                        width,
                        &mut scratch.cb_top,
                        &mut scratch.cb_bot,
                    );
                    upsample_420_pair(
                        prev,
                        curr,
                        next,
                        2,
                        local_y as u32,
                        width,
                        &mut scratch.cr_top,
                        &mut scratch.cr_bot,
                    );

                    writer.write_ycbcr_row(
                        y_start + local_y as u32,
                        y_top,
                        &scratch.cb_top,
                        &scratch.cr_top,
                    )?;
                    if let Some(y_bottom) = y_bottom {
                        writer.write_ycbcr_row(
                            y_start + next_local_y as u32,
                            y_bottom,
                            &scratch.cb_bot,
                            &scratch.cr_bot,
                        )?;
                    }
                    local_y += 2;
                }
            } else {
                let OutputScratch::YCbCrGeneric(scratch) = output_scratch else {
                    unreachable!("generic YCbCr requires reusable row scratch");
                };

                for local_y in 0..stripe_rows {
                    let y_row = &curr.row(0, local_y)[..width];
                    upsample_component_row_stripe(
                        prev,
                        curr,
                        next,
                        1,
                        cb_h,
                        cb_v,
                        max_h,
                        max_v,
                        local_y as u32,
                        width,
                        &mut scratch.cb_up,
                    );
                    upsample_component_row_stripe(
                        prev,
                        curr,
                        next,
                        2,
                        cr_h,
                        cr_v,
                        max_h,
                        max_v,
                        local_y as u32,
                        width,
                        &mut scratch.cr_up,
                    );
                    writer.write_ycbcr_row(
                        y_start + local_y as u32,
                        y_row,
                        &scratch.cb_up,
                        &scratch.cr_up,
                    )?;
                }
            }
        }
        ColorSpace::Rgb => {
            let (r_h, r_v) = plan
                .sampling
                .component(0)
                .map(|(h, v)| (u32::from(h), u32::from(v)))
                .ok_or(JpegError::UnsupportedComponentCount { count: 0 })?;
            let (g_h, g_v) = plan
                .sampling
                .component(1)
                .map(|(h, v)| (u32::from(h), u32::from(v)))
                .ok_or(JpegError::UnsupportedComponentCount { count: 1 })?;
            let (b_h, b_v) = plan
                .sampling
                .component(2)
                .map(|(h, v)| (u32::from(h), u32::from(v)))
                .ok_or(JpegError::UnsupportedComponentCount { count: 2 })?;

            let max_h = plan.sampling.max_h as u32;
            let max_v = plan.sampling.max_v as u32;

            let OutputScratch::RgbGeneric(scratch) = output_scratch else {
                unreachable!("RGB decode requires reusable row scratch");
            };

            for local_y in 0..stripe_rows {
                upsample_component_row_stripe(
                    prev,
                    curr,
                    next,
                    0,
                    r_h,
                    r_v,
                    max_h,
                    max_v,
                    local_y as u32,
                    width,
                    &mut scratch.r,
                );
                upsample_component_row_stripe(
                    prev,
                    curr,
                    next,
                    1,
                    g_h,
                    g_v,
                    max_h,
                    max_v,
                    local_y as u32,
                    width,
                    &mut scratch.g,
                );
                upsample_component_row_stripe(
                    prev,
                    curr,
                    next,
                    2,
                    b_h,
                    b_v,
                    max_h,
                    max_v,
                    local_y as u32,
                    width,
                    &mut scratch.b,
                );
                writer.write_rgb_row(
                    y_start + local_y as u32,
                    &scratch.r,
                    &scratch.g,
                    &scratch.b,
                )?;
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
fn emit_stripe_rgb<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    prev: Option<&StripeBuffer>,
    curr: &StripeBuffer,
    next: Option<&StripeBuffer>,
    stripe_index: u32,
    writer: &mut W,
    output_scratch: &mut RgbOutputScratch<'_>,
    source_width: usize,
    downscale: DownscaleFactor,
) -> Result<(), JpegError> {
    let max_v = plan.sampling.max_v as u32;
    let mcu_height_px = downscale.output_block_size() * max_v;
    let y_start = stripe_index * mcu_height_px;
    let (_, scaled_height) = scaled_dimensions(plan.dimensions, downscale);
    let y_end = (y_start + mcu_height_px).min(scaled_height);
    let stripe_rows = (y_end - y_start) as usize;

    if stripe_rows == 0 {
        return Ok(());
    }

    let width = source_width;
    match plan.color_space {
        ColorSpace::Grayscale => {
            for local_y in 0..stripe_rows {
                let y_row = &curr.row(0, local_y)[..width];
                writer.with_rgb_rows(y_start + local_y as u32, 1, |dst, _| {
                    backend.fill_rgb_row_from_gray(y_row, dst);
                    Ok(())
                })?;
            }
        }
        ColorSpace::YCbCr if is_ycbcr_420(plan) => {
            let RgbOutputScratch::YCbCr420 = output_scratch else {
                unreachable!("4:2:0 YCbCr RGB output requires dedicated scratch");
            };
            let mut local_y = 0usize;
            while local_y < stripe_rows {
                let y_top = &curr.row(0, local_y)[..width];
                let next_local_y = local_y + 1;
                let y_bottom =
                    (next_local_y < stripe_rows).then(|| &curr.row(0, next_local_y)[..width]);
                let row_count = if y_bottom.is_some() { 2 } else { 1 };
                let chroma_y = (local_y / 2).min(curr.row_count(1).saturating_sub(1));
                let chroma_cols = width.div_ceil(2);
                let (prev_cb, curr_cb, next_cb) = component_row_triplet(
                    prev.map(|stripe| stripe.plane(1)),
                    curr.plane(1),
                    next.map(|stripe| stripe.plane(1)),
                    chroma_y,
                );
                let (prev_cr, curr_cr, next_cr) = component_row_triplet(
                    prev.map(|stripe| stripe.plane(2)),
                    curr.plane(2),
                    next.map(|stripe| stripe.plane(2)),
                    chroma_y,
                );

                writer.with_rgb_rows(
                    y_start + local_y as u32,
                    row_count,
                    |dst_top, dst_bottom| {
                        backend.fill_rgb_row_pair_from_420(
                            y_top,
                            y_bottom,
                            &prev_cb[..chroma_cols],
                            &curr_cb[..chroma_cols],
                            &next_cb[..chroma_cols],
                            &prev_cr[..chroma_cols],
                            &curr_cr[..chroma_cols],
                            &next_cr[..chroma_cols],
                            dst_top,
                            dst_bottom,
                        );
                        Ok(())
                    },
                )?;
                local_y += 2;
            }
        }
        ColorSpace::YCbCr => {
            let RgbOutputScratch::YCbCrGeneric(scratch) = output_scratch else {
                unreachable!("generic YCbCr RGB output requires reusable row scratch");
            };
            let (cb_h, cb_v) = plan
                .sampling
                .component(1)
                .map(|(h, v)| (u32::from(h), u32::from(v)))
                .ok_or(JpegError::UnsupportedComponentCount { count: 1 })?;
            let (cr_h, cr_v) = plan
                .sampling
                .component(2)
                .map(|(h, v)| (u32::from(h), u32::from(v)))
                .ok_or(JpegError::UnsupportedComponentCount { count: 2 })?;

            let max_h = plan.sampling.max_h as u32;
            let max_v = plan.sampling.max_v as u32;

            if cb_h == 1 && cb_v == 1 && cr_h == 1 && cr_v == 1 && max_h == 1 && max_v == 1 {
                for local_y in 0..stripe_rows {
                    let y_row = &curr.row(0, local_y)[..width];
                    let cb_row = &curr.row(1, local_y)[..width];
                    let cr_row = &curr.row(2, local_y)[..width];
                    writer.with_rgb_rows(y_start + local_y as u32, 1, |dst, _| {
                        backend.fill_rgb_row_from_ycbcr(y_row, cb_row, cr_row, dst);
                        Ok(())
                    })?;
                }
                return Ok(());
            }

            for local_y in 0..stripe_rows {
                let y_row = &curr.row(0, local_y)[..width];
                upsample_component_row_stripe(
                    prev,
                    curr,
                    next,
                    1,
                    cb_h,
                    cb_v,
                    max_h,
                    max_v,
                    local_y as u32,
                    width,
                    &mut scratch.cb_up,
                );
                upsample_component_row_stripe(
                    prev,
                    curr,
                    next,
                    2,
                    cr_h,
                    cr_v,
                    max_h,
                    max_v,
                    local_y as u32,
                    width,
                    &mut scratch.cr_up,
                );
                writer.with_rgb_rows(y_start + local_y as u32, 1, |dst, _| {
                    backend.fill_rgb_row_from_ycbcr(y_row, &scratch.cb_up, &scratch.cr_up, dst);
                    Ok(())
                })?;
            }
        }
        ColorSpace::Rgb => {
            let RgbOutputScratch::RgbGeneric(scratch) = output_scratch else {
                unreachable!("RGB output requires reusable row scratch");
            };
            let (r_h, r_v) = plan
                .sampling
                .component(0)
                .map(|(h, v)| (u32::from(h), u32::from(v)))
                .ok_or(JpegError::UnsupportedComponentCount { count: 0 })?;
            let (g_h, g_v) = plan
                .sampling
                .component(1)
                .map(|(h, v)| (u32::from(h), u32::from(v)))
                .ok_or(JpegError::UnsupportedComponentCount { count: 1 })?;
            let (b_h, b_v) = plan
                .sampling
                .component(2)
                .map(|(h, v)| (u32::from(h), u32::from(v)))
                .ok_or(JpegError::UnsupportedComponentCount { count: 2 })?;

            let max_h = plan.sampling.max_h as u32;
            let max_v = plan.sampling.max_v as u32;

            for local_y in 0..stripe_rows {
                upsample_component_row_stripe(
                    prev,
                    curr,
                    next,
                    0,
                    r_h,
                    r_v,
                    max_h,
                    max_v,
                    local_y as u32,
                    width,
                    &mut scratch.r,
                );
                upsample_component_row_stripe(
                    prev,
                    curr,
                    next,
                    1,
                    g_h,
                    g_v,
                    max_h,
                    max_v,
                    local_y as u32,
                    width,
                    &mut scratch.g,
                );
                upsample_component_row_stripe(
                    prev,
                    curr,
                    next,
                    2,
                    b_h,
                    b_v,
                    max_h,
                    max_v,
                    local_y as u32,
                    width,
                    &mut scratch.b,
                );
                writer.with_rgb_rows(y_start + local_y as u32, 1, |dst, _| {
                    backend.fill_rgb_row_from_rgb(&scratch.r, &scratch.g, &scratch.b, dst);
                    Ok(())
                })?;
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

fn component_row_triplet<'a>(
    prev: Option<StripePlane<'a>>,
    curr: StripePlane<'a>,
    next: Option<StripePlane<'a>>,
    local_row: usize,
) -> (&'a [u8], &'a [u8], &'a [u8]) {
    fn plane_row(plane: StripePlane<'_>, row: usize) -> &[u8] {
        let start = row * plane.stride;
        &plane.data[start..start + plane.stride]
    }

    let curr_rows = curr.rows;
    let prev_row = if local_row == 0 {
        match prev {
            Some(plane) => plane_row(plane, plane.rows - 1),
            None => plane_row(curr, 0),
        }
    } else {
        plane_row(curr, local_row - 1)
    };
    let curr_row = plane_row(curr, local_row);
    let next_row = if local_row + 1 < curr_rows {
        plane_row(curr, local_row + 1)
    } else {
        match next {
            Some(plane) => plane_row(plane, 0),
            None => plane_row(curr, curr_rows - 1),
        }
    };
    (prev_row, curr_row, next_row)
}

#[allow(clippy::too_many_arguments)]
fn upsample_component_row_stripe(
    prev: Option<&StripeBuffer>,
    curr: &StripeBuffer,
    next: Option<&StripeBuffer>,
    plane_idx: usize,
    comp_h: u32,
    comp_v: u32,
    max_h: u32,
    max_v: u32,
    local_y_out: u32,
    width: usize,
    out: &mut [u8],
) {
    let v_ratio = max_v / comp_v;
    let h_ratio = max_h / comp_h;
    let curr_plane = curr.plane(plane_idx);
    let chroma_rows = curr_plane.rows as u32;
    let chroma_y = (local_y_out / v_ratio).min(chroma_rows.saturating_sub(1));
    let (prev_row, curr_row, next_row) = component_row_triplet(
        prev.map(|stripe| stripe.plane(plane_idx)),
        curr_plane,
        next.map(|stripe| stripe.plane(plane_idx)),
        chroma_y as usize,
    );

    match (h_ratio, v_ratio) {
        (1, 1) => {
            upsample_1x1(&curr_row[..width], out);
        }
        (2, 1) => {
            let chroma_cols = width.div_ceil(2);
            upsample_h2v1_fancy_row(&curr_row[..chroma_cols], width, out);
        }
        (2, 2) => {
            let chroma_cols = width.div_ceil(2);
            upsample_h2v2_fancy_row(
                &prev_row[..chroma_cols],
                &curr_row[..chroma_cols],
                &next_row[..chroma_cols],
                width,
                !local_y_out.is_multiple_of(2),
                out,
            );
        }
        _ => {
            for (x, slot) in out.iter_mut().enumerate().take(width) {
                let cx = ((x as u32) / h_ratio).min(curr_row.len() as u32 - 1);
                *slot = curr_row[cx as usize];
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn upsample_420_pair(
    prev: Option<&StripeBuffer>,
    curr: &StripeBuffer,
    next: Option<&StripeBuffer>,
    plane_idx: usize,
    local_y_out: u32,
    width: usize,
    top: &mut [u8],
    bot: &mut [u8],
) {
    let curr_plane = curr.plane(plane_idx);
    let chroma_rows = curr_plane.rows as u32;
    let chroma_y = (local_y_out / 2).min(chroma_rows.saturating_sub(1));
    let (prev_row, curr_row, next_row) = component_row_triplet(
        prev.map(|stripe| stripe.plane(plane_idx)),
        curr_plane,
        next.map(|stripe| stripe.plane(plane_idx)),
        chroma_y as usize,
    );

    upsample_h2v2_fancy_rows(prev_row, curr_row, next_row, width, top, bot);
}

fn is_ycbcr_420(plan: &PreparedDecodePlan) -> bool {
    plan.color_space == ColorSpace::YCbCr
        && plan.sampling.max_h == 2
        && plan.sampling.max_v == 2
        && plan.sampling.components() == [(2, 2), (1, 1), (1, 1)]
}

fn scaled_dimensions(dims: (u32, u32), downscale: DownscaleFactor) -> (u32, u32) {
    let denom = downscale.denominator();
    (dims.0.div_ceil(denom), dims.1.div_ceil(denom))
}

fn expanded_output_rect(rect: Rect, width: u32, height: u32) -> Rect {
    let x = rect.x.saturating_sub(2);
    let y = rect.y.saturating_sub(2);
    let x_end = rect.x.saturating_add(rect.w).saturating_add(2).min(width);
    let y_end = rect.y.saturating_add(rect.h).saturating_add(2).min(height);
    Rect {
        x,
        y,
        w: x_end.saturating_sub(x),
        h: y_end.saturating_sub(y),
    }
}

#[allow(clippy::too_many_arguments)]
fn component_block_intersects_rect(
    plan: &PreparedDecodePlan,
    comp: &PreparedComponentPlan,
    downscale: DownscaleFactor,
    mcu_x: u32,
    mcu_y: u32,
    block_x: u32,
    block_y: u32,
    rect: Rect,
) -> bool {
    let block_size = downscale.output_block_size();
    let h_ratio = u32::from(plan.sampling.max_h / comp.h);
    let v_ratio = u32::from(plan.sampling.max_v / comp.v);
    let x0 = mcu_x * u32::from(plan.sampling.max_h) * block_size + block_x * h_ratio * block_size;
    let y0 = mcu_y * u32::from(plan.sampling.max_v) * block_size + block_y * v_ratio * block_size;
    let w = h_ratio * block_size;
    let h = v_ratio * block_size;
    let x1 = x0 + w;
    let y1 = y0 + h;
    let rect_x1 = rect.x + rect.w;
    let rect_y1 = rect.y + rect.h;
    x0 < rect_x1 && x1 > rect.x && y0 < rect_y1 && y1 > rect.y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::huffman::HuffmanTable;
    use crate::output::Rgb8Writer;
    use crate::parse::tables::{HuffmanValues, RawHuffmanTable};
    use crate::Decoder;
    use alloc::sync::Arc;
    use alloc::vec;

    const BASELINE_420_JPG: &[u8] =
        include_bytes!("../../../../corpus/conformance/baseline_420_16x16.jpg");

    #[test]
    fn fast_tile_rgb_matches_generic_baseline_decode() {
        let dec = Decoder::new(BASELINE_420_JPG).expect("fixture must parse");
        assert!(dec.plan.matches_fast_tile_shape());

        let mut generic = vec![0u8; (dec.info.dimensions.0 * dec.info.dimensions.1 * 3) as usize];
        let mut fast = vec![0u8; generic.len()];
        let mut generic_writer = Rgb8Writer::new(
            &mut generic,
            dec.info.dimensions.0 as usize * 3,
            dec.info.dimensions.0,
        );
        let mut fast_writer = Rgb8Writer::new(
            &mut fast,
            dec.info.dimensions.0 as usize * 3,
            dec.info.dimensions.0,
        );
        let mut generic_pool = ScratchPool::new();
        let mut fast_pool = ScratchPool::new();
        let scan_bytes = &dec.bytes[dec.plan.scan_offset..];

        decode_scan_baseline_rgb(
            &dec.plan,
            dec.backend,
            scan_bytes,
            &mut generic_pool,
            &mut generic_writer,
            DownscaleFactor::Full,
            Rect::full(dec.info.dimensions),
        )
        .expect("generic path must decode");
        decode_scan_fast_tile_rgb(
            &dec.plan,
            dec.backend,
            scan_bytes,
            &mut fast_pool,
            &mut fast_writer,
        )
        .expect("fast path must decode");

        assert_eq!(fast, generic);
    }

    #[test]
    fn fast_tile_row_decoder_uses_stripe_local_y_offsets() {
        let dc = trivial_dc_table();
        let ac = eob_ac_table();
        let quant = Arc::new([1u16; 64]);
        let y_comp = PreparedComponentPlan {
            h: 2,
            v: 2,
            output_index: 0,
            quant: Arc::clone(&quant),
            dc_table: Arc::clone(&dc),
            ac_table: Arc::clone(&ac),
        };
        let cb_comp = PreparedComponentPlan {
            h: 1,
            v: 1,
            output_index: 1,
            quant: Arc::clone(&quant),
            dc_table: Arc::clone(&dc),
            ac_table: Arc::clone(&ac),
        };
        let cr_comp = PreparedComponentPlan {
            h: 1,
            v: 1,
            output_index: 2,
            quant,
            dc_table: dc,
            ac_table: ac,
        };
        let mut stripe = StripeBuffer {
            planes: vec![vec![0u8; 16 * 16], vec![0u8; 8 * 8], vec![0u8; 8 * 8]],
            plane_strides: vec![16, 8, 8],
            plane_rows: vec![16, 8, 8],
        };
        let mut br = BitReader::new(&[0u8; 16]);
        let mut coeff = CoefficientBlock::default();
        let mut pixels = [0u8; 64];
        let mut y_dc = 0;
        let mut cb_dc = 0;
        let mut cr_dc = 0;

        decode_mcu_row_fast_tile_420(
            &y_comp,
            &cb_comp,
            &cr_comp,
            Backend::detect(),
            &mut br,
            &mut y_dc,
            &mut cb_dc,
            &mut cr_dc,
            &mut coeff,
            &mut pixels,
            1,
            0,
            1,
            &mut stripe,
        )
        .expect("second stripe row must still decode within stripe-local buffers");

        assert_eq!(stripe.planes[0][0], 128);
        assert_eq!(stripe.planes[0][8], 128);
        assert_eq!(stripe.planes[0][16 * 8], 128);
        assert_eq!(stripe.planes[0][16 * 8 + 8], 128);
        assert_eq!(stripe.planes[1][0], 128);
        assert_eq!(stripe.planes[2][0], 128);
    }

    #[test]
    fn fast_tile_region_layout_shrinks_horizontal_stripe_span() {
        let roi = Rect {
            x: 17,
            y: 3,
            w: 9,
            h: 8,
        };

        let layout = Fast420RegionLayout::new(64, roi);

        assert_eq!(layout.stripe_mcu_start, 0);
        assert_eq!(layout.stripe_mcus_per_row, 2);
        assert_eq!(layout.row_width(), 32);
        assert_eq!(layout.chroma_width(), 16);
        assert_eq!(layout.crop_start, 17);
        assert_eq!(layout.crop_end, 26);
        assert!(layout.y_decode_start <= roi.x as usize);
        assert!(layout.y_decode_end >= (roi.x + roi.w) as usize);
    }

    #[test]
    fn fast420_vertical_context_only_keeps_neighbor_stripes_when_needed() {
        let middle_roi = Rect {
            x: 0,
            y: 76,
            w: 256,
            h: 256,
        };
        assert_eq!(fast420_first_decode_mcu_row(middle_roi, 16), 4);
        assert_eq!(fast420_decode_mcu_row_end(middle_roi, 16, 26), 21);

        let top_edge_roi = Rect {
            x: 0,
            y: 64,
            w: 32,
            h: 16,
        };
        assert_eq!(fast420_first_decode_mcu_row(top_edge_roi, 16), 3);

        let bottom_edge_roi = Rect {
            x: 0,
            y: 78,
            w: 32,
            h: 18,
        };
        assert_eq!(fast420_decode_mcu_row_end(bottom_edge_roi, 16, 26), 7);
    }

    #[test]
    fn deposit_block_writes_expected_rows_at_offset() {
        let mut plane = vec![0xA5u8; 16 * 16];
        let mut block = [0u8; 64];
        for (i, byte) in block.iter_mut().enumerate() {
            *byte = i as u8;
        }

        deposit_block(&mut plane, 16, 3, 2, &block);

        for row in 0..8usize {
            let dst_start = (2 + row) * 16 + 3;
            assert_eq!(
                &plane[dst_start..dst_start + 8],
                &block[row * 8..row * 8 + 8]
            );
            assert_eq!(plane[(2 + row) * 16 + 2], 0xA5);
            assert_eq!(plane[(2 + row) * 16 + 11], 0xA5);
        }
        assert_eq!(plane[0], 0xA5);
        assert_eq!(plane[plane.len() - 1], 0xA5);
    }

    #[test]
    fn deposit_block_writes_expected_rows_at_bottom_right_edge() {
        let mut plane = vec![0x5Au8; 16 * 16];
        let mut block = [0u8; 64];
        for (i, byte) in block.iter_mut().enumerate() {
            *byte = 255u8.wrapping_sub(i as u8);
        }

        deposit_block(&mut plane, 16, 8, 8, &block);

        for row in 0..8usize {
            let dst_start = (8 + row) * 16 + 8;
            assert_eq!(
                &plane[dst_start..dst_start + 8],
                &block[row * 8..row * 8 + 8]
            );
            assert_eq!(plane[(8 + row) * 16 + 7], 0x5A);
        }
        assert_eq!(plane[plane.len() - 1], block[63]);
    }

    #[test]
    fn component_row_triplet_uses_neighbor_stripes_and_clamps_edges() {
        let prev = StripeBuffer {
            planes: vec![vec![], vec![10, 11, 12, 13, 14, 15], vec![]],
            plane_strides: vec![0, 2, 0],
            plane_rows: vec![0, 3, 0],
        };
        let curr = StripeBuffer {
            planes: vec![vec![], vec![20, 21, 22, 23, 24, 25], vec![]],
            plane_strides: vec![0, 2, 0],
            plane_rows: vec![0, 3, 0],
        };
        let next = StripeBuffer {
            planes: vec![vec![], vec![30, 31, 32, 33, 34, 35], vec![]],
            plane_strides: vec![0, 2, 0],
            plane_rows: vec![0, 3, 0],
        };

        let prev_plane = Some(prev.plane(1));
        let curr_plane = curr.plane(1);
        let next_plane = Some(next.plane(1));

        let (top_prev, top_curr, top_next) =
            component_row_triplet(prev_plane, curr_plane, next_plane, 0);
        assert_eq!(top_prev, &[14, 15]);
        assert_eq!(top_curr, &[20, 21]);
        assert_eq!(top_next, &[22, 23]);

        let (mid_prev, mid_curr, mid_next) =
            component_row_triplet(prev_plane, curr_plane, next_plane, 1);
        assert_eq!(mid_prev, &[20, 21]);
        assert_eq!(mid_curr, &[22, 23]);
        assert_eq!(mid_next, &[24, 25]);

        let (bot_prev, bot_curr, bot_next) =
            component_row_triplet(prev_plane, curr_plane, next_plane, 2);
        assert_eq!(bot_prev, &[22, 23]);
        assert_eq!(bot_curr, &[24, 25]);
        assert_eq!(bot_next, &[30, 31]);

        let (clamp_prev, clamp_curr, clamp_next) = component_row_triplet(None, curr_plane, None, 0);
        assert_eq!(clamp_prev, &[20, 21]);
        assert_eq!(clamp_curr, &[20, 21]);
        assert_eq!(clamp_next, &[22, 23]);

        let (tail_prev, tail_curr, tail_next) = component_row_triplet(None, curr_plane, None, 2);
        assert_eq!(tail_prev, &[22, 23]);
        assert_eq!(tail_curr, &[24, 25]);
        assert_eq!(tail_next, &[24, 25]);
    }

    #[test]
    fn emit_stripe_rgb_444_matches_direct_ycbcr_conversion_with_trailing_row() {
        let width = 17usize;
        let height = 7u32;
        let mut stripe = StripeBuffer {
            planes: vec![
                vec![0u8; width * 8],
                vec![0u8; width * 8],
                vec![0u8; width * 8],
            ],
            plane_strides: vec![width, width, width],
            plane_rows: vec![8, 8, 8],
        };
        for row in 0..8usize {
            for col in 0..width {
                stripe.planes[0][row * width + col] = ((row * 31 + col * 7 + 11) & 0xFF) as u8;
                stripe.planes[1][row * width + col] = ((row * 17 + col * 13 + 97) & 0xFF) as u8;
                stripe.planes[2][row * width + col] = ((row * 23 + col * 19 + 53) & 0xFF) as u8;
            }
        }

        let plan = PreparedDecodePlan {
            components: vec![],
            sampling: SamplingFactors::from_components(&[(1, 1), (1, 1), (1, 1)]),
            color_space: ColorSpace::YCbCr,
            restart_interval: None,
            dimensions: (width as u32, height),
            scan_offset: 0,
            scratch_bytes: 0,
        };
        let mut actual = vec![0u8; width * height as usize * 3];
        let mut writer = Rgb8Writer::new(&mut actual, width * 3, width as u32);

        emit_stripe_rgb_444(&plan, Backend::detect(), &stripe, 0, &mut writer)
            .expect("emit stripe must succeed");

        let mut expected = vec![0u8; actual.len()];
        for row in 0..height as usize {
            for col in 0..width {
                let y = stripe.planes[0][row * width + col];
                let cb = stripe.planes[1][row * width + col];
                let cr = stripe.planes[2][row * width + col];
                let (r, g, b) = crate::color::ycbcr::ycbcr_to_rgb(y, cb, cr);
                let dst = (row * width + col) * 3;
                expected[dst] = r;
                expected[dst + 1] = g;
                expected[dst + 2] = b;
            }
        }

        assert_eq!(actual, expected);
    }

    fn trivial_dc_table() -> Arc<HuffmanTable> {
        Arc::new(
            HuffmanTable::from_raw(&RawHuffmanTable {
                bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                values: HuffmanValues::from_slice(&[0]),
            })
            .expect("trivial DC table must be valid"),
        )
    }

    fn eob_ac_table() -> Arc<HuffmanTable> {
        Arc::new(
            HuffmanTable::from_raw(&RawHuffmanTable {
                bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                values: HuffmanValues::from_slice(&[0x00]),
            })
            .expect("trivial AC table must be valid"),
        )
    }
}
