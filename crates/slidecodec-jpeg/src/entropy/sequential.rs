// SPDX-License-Identifier: Apache-2.0

//! Baseline sequential scan decoder. Iterates MCUs, decodes blocks, runs the
//! IDCT, and pipes rows through an [`OutputWriter`] with chroma upsample and
//! color conversion.

use crate::backend::Backend;
use crate::color::upsample::{
    upsample_1x1, upsample_h2v1_fancy_row, upsample_h2v2_fancy_row, upsample_h2v2_fancy_rows,
};
use crate::entropy::block::{decode_block_with_activity, BlockActivity, CoefficientBlock};
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
        0,
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
    for my in 1..mcu_rows {
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
        core::mem::swap(&mut prev_stripe, &mut curr_stripe);
        core::mem::swap(&mut curr_stripe, &mut next_stripe);
        has_prev = true;
    }

    emit_stripe(
        plan,
        has_prev.then_some(&*prev_stripe),
        curr_stripe,
        None,
        mcu_rows - 1,
        writer,
        &mut output_scratch,
        region_layout.source_width_usize(),
        downscale,
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
        0,
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
    for my in 1..mcu_rows {
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
        region_layout.source_width_usize(),
        downscale,
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

#[derive(Clone, Copy)]
struct Fast420RegionLayout {
    stripe_mcu_start: u32,
    stripe_mcus_per_row: u32,
    y_decode_start: usize,
    y_decode_end: usize,
    chroma_decode_start: usize,
    chroma_decode_end: usize,
    crop_start: usize,
    crop_end: usize,
}

impl Fast420RegionLayout {
    fn new(width: usize, roi: Rect) -> Self {
        let crop_window = RgbCropWindow::new(width, roi);
        let stripe = StripeRegionLayout::new(
            width as u32,
            16,
            Rect {
                x: crop_window.scratch_x0 as u32,
                y: 0,
                w: (crop_window.scratch_x1 - crop_window.scratch_x0) as u32,
                h: 1,
            },
        );
        let y_decode_start = stripe.source_x0 as usize;
        let y_decode_end = y_decode_start + stripe.source_width as usize;
        let chroma_decode_start = (stripe.source_x0 / 2) as usize;
        let chroma_decode_end = chroma_decode_start + stripe.source_width.div_ceil(2) as usize;
        let crop_start = roi.x as usize - y_decode_start;
        let crop_end = crop_start + roi.w as usize;

        Self {
            stripe_mcu_start: stripe.stripe_mcu_start,
            stripe_mcus_per_row: stripe.stripe_mcus_per_row,
            y_decode_start,
            y_decode_end,
            chroma_decode_start,
            chroma_decode_end,
            crop_start,
            crop_end,
        }
    }

    fn row_width(self) -> usize {
        self.y_decode_end - self.y_decode_start
    }

    fn chroma_width(self) -> usize {
        self.chroma_decode_end - self.chroma_decode_start
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
                region_layout.stripe_mcu_start,
                region_layout.stripe_mcus_per_row,
                next_stripe,
            )?;
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
            )?;
            core::mem::swap(&mut prev_stripe, &mut curr_stripe);
            core::mem::swap(&mut curr_stripe, &mut next_stripe);
            has_prev = true;
        }

        emit_stripe_rgb_420_region(
            plan,
            backend,
            has_prev.then_some(&*prev_stripe),
            curr_stripe,
            None,
            mcu_rows - 1,
            writer,
            roi,
            region_layout,
            &mut crop_rows,
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
    })();
    pool.restore_sink_rows(crop_rows);
    result
}

fn deposit_block(plane: &mut [u8], stride: usize, x: u32, y: u32, block: &[u8; 64]) {
    let x = x as usize;
    let y = y as usize;
    for by in 0..8 {
        let dst_start = (y + by) * stride + x;
        plane[dst_start..dst_start + 8].copy_from_slice(&block[by * 8..by * 8 + 8]);
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
) -> Result<(), JpegError> {
    let max_v = plan.sampling.max_v as u32;
    let mcu_height_px = DownscaleFactor::Full.output_block_size() * max_v;
    let y_start = stripe_index * mcu_height_px;
    let (_, scaled_height) = scaled_dimensions(plan.dimensions, DownscaleFactor::Full);
    let y_end = (y_start + mcu_height_px).min(scaled_height);
    let stripe_rows = (y_end - y_start) as usize;

    if stripe_rows == 0 {
        return Ok(());
    }

    let row_len = region_layout.row_width() * 3;
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

        let y_top = &curr.row(0, local_y)[..region_layout.row_width()];
        let y_bottom = (next_local_y < stripe_rows)
            .then(|| &curr.row(0, next_local_y)[..region_layout.row_width()]);
        let chroma_y = (local_y / 2).min(curr.row_count(1).saturating_sub(1));
        let (prev_cb, curr_cb, next_cb) = component_row_triplet(prev, curr, next, 1, chroma_y);
        let (prev_cr, curr_cr, next_cr) = component_row_triplet(prev, curr, next, 2, chroma_y);

        backend.fill_rgb_row_pair_from_420(
            y_top,
            y_bottom,
            &prev_cb[..region_layout.chroma_width()],
            &curr_cb[..region_layout.chroma_width()],
            &next_cb[..region_layout.chroma_width()],
            &prev_cr[..region_layout.chroma_width()],
            &curr_cr[..region_layout.chroma_width()],
            &next_cr[..region_layout.chroma_width()],
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
    for by in 0..2 {
        let dst_start = (y + by) * stride + x;
        plane[dst_start..dst_start + 2].copy_from_slice(&block[by * 2..by * 2 + 2]);
    }
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
                    let activity = decode_block_with_activity(
                        br,
                        &comp.dc_table,
                        &comp.ac_table,
                        &mut prev_dc[plane_idx],
                        &comp.quant,
                        coeff,
                    )?;
                    if !in_region
                        || (!full_output_rect
                            && !component_block_intersects_rect(
                                plan,
                                comp,
                                downscale,
                                mx,
                                mcu_y,
                                vx,
                                vy,
                                output_rect,
                            ))
                    {
                        continue;
                    }
                    let block_x = local_mcu_x0_px + vx * block_size;
                    let block_y = vy * block_size;
                    match downscale {
                        DownscaleFactor::Full => {
                            match activity {
                                BlockActivity::DcOnly => {
                                    crate::idct::idct_islow_dc_only(coeff.dc_coeff(), pixels);
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
                            downscale::idct_islow_4x4(coeff.coefficients(), &mut pixels_4x4);
                            deposit_block_4x4(
                                &mut stripe.planes[plane_idx],
                                stripe.plane_strides[plane_idx],
                                block_x,
                                block_y,
                                &pixels_4x4,
                            );
                        }
                        DownscaleFactor::Quarter => {
                            downscale::idct_islow_2x2(coeff.coefficients(), &mut pixels_2x2);
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

    for local_y in 0..stripe_rows {
        let y_row = &stripe.row(0, local_y)[..width];
        let cb_row = &stripe.row(1, local_y)[..width];
        let cr_row = &stripe.row(2, local_y)[..width];
        writer.with_rgb_rows(y_start + local_y as u32, 1, |dst, _| {
            backend.fill_rgb_row_from_ycbcr(y_row, cb_row, cr_row, dst);
            Ok(())
        })?;
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
        let local_mx = mx.saturating_sub(stripe_mcu_start);
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
        if in_region {
            match y0_activity {
                BlockActivity::DcOnly => crate::idct::idct_islow_dc_only(coeff.dc_coeff(), pixels),
                BlockActivity::General => backend.idct(coeff.coefficients(), pixels),
            }
            deposit_block(
                &mut stripe.planes[0],
                stripe.plane_strides[0],
                y_x,
                0,
                pixels,
            );
        }

        let y1_activity = decode_block_with_activity(
            br,
            &y_comp.dc_table,
            &y_comp.ac_table,
            y_dc,
            y_comp.quant.as_ref(),
            coeff,
        )?;
        if in_region {
            match y1_activity {
                BlockActivity::DcOnly => crate::idct::idct_islow_dc_only(coeff.dc_coeff(), pixels),
                BlockActivity::General => backend.idct(coeff.coefficients(), pixels),
            }
            deposit_block(
                &mut stripe.planes[0],
                stripe.plane_strides[0],
                y_x + 8,
                0,
                pixels,
            );
        }

        let y2_activity = decode_block_with_activity(
            br,
            &y_comp.dc_table,
            &y_comp.ac_table,
            y_dc,
            y_comp.quant.as_ref(),
            coeff,
        )?;
        if in_region {
            match y2_activity {
                BlockActivity::DcOnly => crate::idct::idct_islow_dc_only(coeff.dc_coeff(), pixels),
                BlockActivity::General => backend.idct(coeff.coefficients(), pixels),
            }
            deposit_block(
                &mut stripe.planes[0],
                stripe.plane_strides[0],
                y_x,
                8,
                pixels,
            );
        }

        let y3_activity = decode_block_with_activity(
            br,
            &y_comp.dc_table,
            &y_comp.ac_table,
            y_dc,
            y_comp.quant.as_ref(),
            coeff,
        )?;
        if in_region {
            match y3_activity {
                BlockActivity::DcOnly => crate::idct::idct_islow_dc_only(coeff.dc_coeff(), pixels),
                BlockActivity::General => backend.idct(coeff.coefficients(), pixels),
            }
            deposit_block(
                &mut stripe.planes[0],
                stripe.plane_strides[0],
                y_x + 8,
                8,
                pixels,
            );
        }

        let cb_activity = decode_block_with_activity(
            br,
            &cb_comp.dc_table,
            &cb_comp.ac_table,
            cb_dc,
            cb_comp.quant.as_ref(),
            coeff,
        )?;
        if in_region {
            match cb_activity {
                BlockActivity::DcOnly => crate::idct::idct_islow_dc_only(coeff.dc_coeff(), pixels),
                BlockActivity::General => backend.idct(coeff.coefficients(), pixels),
            }
            deposit_block(
                &mut stripe.planes[1],
                stripe.plane_strides[1],
                c_x,
                0,
                pixels,
            );
        }

        let cr_activity = decode_block_with_activity(
            br,
            &cr_comp.dc_table,
            &cr_comp.ac_table,
            cr_dc,
            cr_comp.quant.as_ref(),
            coeff,
        )?;
        if in_region {
            match cr_activity {
                BlockActivity::DcOnly => crate::idct::idct_islow_dc_only(coeff.dc_coeff(), pixels),
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
                let (prev_cb, curr_cb, next_cb) =
                    component_row_triplet(prev, curr, next, 1, chroma_y);
                let (prev_cr, curr_cr, next_cr) =
                    component_row_triplet(prev, curr, next, 2, chroma_y);

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
    prev: Option<&'a StripeBuffer>,
    curr: &'a StripeBuffer,
    next: Option<&'a StripeBuffer>,
    plane_idx: usize,
    local_row: usize,
) -> (&'a [u8], &'a [u8], &'a [u8]) {
    let curr_rows = curr.row_count(plane_idx);
    let prev_row = if local_row == 0 {
        prev.map_or_else(
            || curr.row(plane_idx, 0),
            |stripe| stripe.row(plane_idx, stripe.row_count(plane_idx) - 1),
        )
    } else {
        curr.row(plane_idx, local_row - 1)
    };
    let curr_row = curr.row(plane_idx, local_row);
    let next_row = if local_row + 1 < curr_rows {
        curr.row(plane_idx, local_row + 1)
    } else {
        next.map_or_else(
            || curr.row(plane_idx, curr_rows - 1),
            |stripe| stripe.row(plane_idx, 0),
        )
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
    let chroma_rows = curr.row_count(plane_idx) as u32;
    let chroma_y = (local_y_out / v_ratio).min(chroma_rows.saturating_sub(1));
    let (prev_row, curr_row, next_row) =
        component_row_triplet(prev, curr, next, plane_idx, chroma_y as usize);

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
    let chroma_rows = curr.row_count(plane_idx) as u32;
    let chroma_y = (local_y_out / 2).min(chroma_rows.saturating_sub(1));
    let (prev_row, curr_row, next_row) =
        component_row_triplet(prev, curr, next, plane_idx, chroma_y as usize);

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
