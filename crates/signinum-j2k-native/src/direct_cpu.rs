use alloc::vec::Vec;

use crate::error::{bail, DecodingError, Result};
use crate::j2c::idwt;
use crate::math::floor_f32;
use crate::{
    decode_ht_code_block_scalar, decode_j2k_code_block_scalar, HtCodeBlockDecodeJob,
    HtOwnedSubBandPlan, J2kCodeBlockDecodeJob, J2kDirectBandId, J2kDirectColorPlan,
    J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kDirectIdwtStep, J2kDirectStoreStep,
    J2kIdwtBand, J2kOwnedSubBandPlan, J2kRect, J2kSingleDecompositionIdwtJob, J2kWaveletTransform,
};

/// Hidden reusable scratch for executing direct J2K RGB plans on the CPU.
#[doc(hidden)]
#[derive(Debug, Default)]
pub struct J2kDirectCpuScratch {
    component_band_sets: Vec<DirectComponentBandScratch>,
    component_planes: Vec<DirectComponentPlane>,
}

impl J2kDirectCpuScratch {
    /// Create empty direct-plan CPU scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            component_band_sets: Vec::new(),
            component_planes: Vec::new(),
        }
    }

    /// Release retained scratch allocations.
    pub fn clear(&mut self) {
        self.component_band_sets.clear();
        self.component_planes.clear();
    }

    fn prepare_component_scratch(&mut self, component_count: usize) {
        while self.component_band_sets.len() < component_count {
            self.component_band_sets
                .push(DirectComponentBandScratch::default());
        }
        while self.component_planes.len() < component_count {
            self.component_planes.push(DirectComponentPlane::default());
        }
    }

    #[cfg(test)]
    fn allocation_profile_for_tests(&self) -> DirectScratchAllocationProfile {
        let band_buffers = self
            .component_band_sets
            .iter()
            .map(|component| component.bands.len())
            .sum();
        let band_sample_len = self
            .component_band_sets
            .iter()
            .flat_map(|component| component.bands.iter())
            .map(|band| band.coefficients.len())
            .sum();
        let band_sample_capacity = self
            .component_band_sets
            .iter()
            .flat_map(|component| component.bands.iter())
            .map(|band| band.coefficients.capacity())
            .sum();
        let component_sample_len = self
            .component_planes
            .iter()
            .map(|plane| plane.samples.len())
            .sum();
        let component_sample_capacity = self
            .component_planes
            .iter()
            .map(|plane| plane.samples.capacity())
            .sum();
        DirectScratchAllocationProfile {
            component_band_sets: self.component_band_sets.len(),
            component_planes: self.component_planes.len(),
            band_buffers,
            band_sample_len,
            band_sample_capacity,
            component_sample_len,
            component_sample_capacity,
        }
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DirectScratchAllocationProfile {
    component_band_sets: usize,
    component_planes: usize,
    band_buffers: usize,
    band_sample_len: usize,
    band_sample_capacity: usize,
    component_sample_len: usize,
    component_sample_capacity: usize,
}

#[derive(Debug, Default)]
struct DirectComponentBandScratch {
    bands: Vec<DirectCpuBand>,
    active_len: usize,
}

impl DirectComponentBandScratch {
    fn reset(&mut self) {
        self.active_len = 0;
    }

    fn active(&self) -> &[DirectCpuBand] {
        &self.bands[..self.active_len]
    }

    fn prepare_band(&mut self, band_id: J2kDirectBandId, rect: J2kRect, len: usize) -> usize {
        let index = self.active_len;
        if index == self.bands.len() {
            self.bands.push(DirectCpuBand::empty());
        }
        let band = &mut self.bands[index];
        band.band_id = band_id;
        band.rect = rect;
        resize_and_zero(&mut band.coefficients, len);
        self.active_len += 1;
        index
    }
}

#[derive(Debug)]
struct DirectCpuBand {
    band_id: J2kDirectBandId,
    rect: J2kRect,
    coefficients: Vec<f32>,
}

impl DirectCpuBand {
    const fn empty() -> Self {
        Self {
            band_id: 0,
            rect: J2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            coefficients: Vec::new(),
        }
    }
}

#[derive(Debug, Default)]
struct DirectComponentPlane {
    width: u32,
    height: u32,
    samples: Vec<f32>,
}

/// Execute a hidden direct RGB plan on the CPU and write an RGB8 output region.
#[doc(hidden)]
pub fn execute_direct_color_plan_rgb8_into(
    plan: &J2kDirectColorPlan,
    output_region: J2kRect,
    scratch: &mut J2kDirectCpuScratch,
    out: &mut [u8],
    stride: usize,
) -> Result<()> {
    execute_direct_color_plan_u8_into(
        plan,
        output_region,
        scratch,
        out,
        stride,
        DirectColorU8Output::Rgb8,
    )
}

/// Execute a hidden direct RGB plan on the CPU and write an RGBA8 output region.
#[doc(hidden)]
pub fn execute_direct_color_plan_rgba8_into(
    plan: &J2kDirectColorPlan,
    output_region: J2kRect,
    scratch: &mut J2kDirectCpuScratch,
    out: &mut [u8],
    stride: usize,
) -> Result<()> {
    execute_direct_color_plan_u8_into(
        plan,
        output_region,
        scratch,
        out,
        stride,
        DirectColorU8Output::Rgba8,
    )
}

#[derive(Clone, Copy)]
enum DirectColorU8Output {
    Rgb8,
    Rgba8,
}

impl DirectColorU8Output {
    const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb8 => 3,
            Self::Rgba8 => 4,
        }
    }
}

fn execute_direct_color_plan_u8_into(
    plan: &J2kDirectColorPlan,
    output_region: J2kRect,
    scratch: &mut J2kDirectCpuScratch,
    out: &mut [u8],
    stride: usize,
    output: DirectColorU8Output,
) -> Result<()> {
    if plan.component_plans.len() != 3 {
        bail!(DecodingError::UnsupportedFeature(
            "direct CPU color plan requires three components"
        ));
    }
    validate_output_region(plan, output_region, out.len(), stride, output)?;

    scratch.prepare_component_scratch(plan.component_plans.len());
    for (component_index, component_plan) in plan.component_plans.iter().enumerate() {
        let band_scratch = &mut scratch.component_band_sets[component_index];
        let plane = &mut scratch.component_planes[component_index];
        execute_component_plan(component_plan, band_scratch, plane)?;
    }

    let [plane0, plane1, plane2, ..] = scratch.component_planes.as_mut_slice() else {
        unreachable!();
    };
    if plan.mct {
        apply_inverse_mct(plan.transform, plan.bit_depths, plane0, plane1, plane2)?;
    }
    write_rgb8_region(
        [plane0, plane1, plane2],
        plan.bit_depths,
        output_region,
        out,
        stride,
        output,
    )
}

fn execute_component_plan(
    plan: &J2kDirectGrayscalePlan,
    bands: &mut DirectComponentBandScratch,
    output: &mut DirectComponentPlane,
) -> Result<()> {
    bands.reset();
    let mut output_written = false;

    for step in &plan.steps {
        match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                execute_classic_sub_band(sub_band, bands)?;
            }
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                execute_ht_sub_band(sub_band, bands)?;
            }
            J2kDirectGrayscaleStep::Idwt(step) => {
                execute_idwt_step(step, bands)?;
            }
            J2kDirectGrayscaleStep::Store(store) => {
                store_component(store, bands.active(), output, &mut output_written)?;
            }
        }
    }

    if output_written {
        Ok(())
    } else {
        Err(DecodingError::CodeBlockDecodeFailure.into())
    }
}

fn execute_classic_sub_band(
    plan: &J2kOwnedSubBandPlan,
    bands: &mut DirectComponentBandScratch,
) -> Result<()> {
    let required_len = checked_area(plan.width, plan.height)?;
    let band_index = bands.prepare_band(plan.band_id, plan.rect, required_len);
    let output = &mut bands.bands[band_index].coefficients;
    let sub_band_width =
        usize::try_from(plan.width).map_err(|_| DecodingError::CodeBlockDecodeFailure)?;

    for job in &plan.jobs {
        let base_idx = checked_block_base(job.output_x, job.output_y, sub_band_width)?;
        let block_len = checked_block_output_len(job.output_stride, job.width, job.height)?;
        let end_idx = base_idx
            .checked_add(block_len)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        if end_idx > output.len()
            || job
                .output_x
                .checked_add(job.width)
                .is_none_or(|x| x > plan.width)
            || job
                .output_y
                .checked_add(job.height)
                .is_none_or(|y| y > plan.height)
        {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }

        let code_block = J2kCodeBlockDecodeJob {
            data: &job.data,
            segments: &job.segments,
            width: job.width,
            height: job.height,
            output_stride: job.output_stride,
            missing_bit_planes: job.missing_bit_planes,
            number_of_coding_passes: job.number_of_coding_passes,
            total_bitplanes: job.total_bitplanes,
            sub_band_type: job.sub_band_type,
            style: job.style,
            strict: job.strict,
            dequantization_step: job.dequantization_step,
        };
        decode_j2k_code_block_scalar(code_block, &mut output[base_idx..end_idx])?;
    }
    Ok(())
}

fn execute_ht_sub_band(
    plan: &HtOwnedSubBandPlan,
    bands: &mut DirectComponentBandScratch,
) -> Result<()> {
    let required_len = checked_area(plan.width, plan.height)?;
    let band_index = bands.prepare_band(plan.band_id, plan.rect, required_len);
    let output = &mut bands.bands[band_index].coefficients;
    let sub_band_width =
        usize::try_from(plan.width).map_err(|_| DecodingError::CodeBlockDecodeFailure)?;

    for job in &plan.jobs {
        let base_idx = checked_block_base(job.output_x, job.output_y, sub_band_width)?;
        let block_len = checked_block_output_len(job.output_stride, job.width, job.height)?;
        let end_idx = base_idx
            .checked_add(block_len)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        if end_idx > output.len()
            || job
                .output_x
                .checked_add(job.width)
                .is_none_or(|x| x > plan.width)
            || job
                .output_y
                .checked_add(job.height)
                .is_none_or(|y| y > plan.height)
        {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }

        let code_block = HtCodeBlockDecodeJob {
            data: &job.data,
            cleanup_length: job.cleanup_length,
            refinement_length: job.refinement_length,
            width: job.width,
            height: job.height,
            output_stride: job.output_stride,
            missing_bit_planes: job.missing_bit_planes,
            number_of_coding_passes: job.number_of_coding_passes,
            num_bitplanes: job.num_bitplanes,
            stripe_causal: job.stripe_causal,
            strict: job.strict,
            dequantization_step: job.dequantization_step,
        };
        decode_ht_code_block_scalar(code_block, &mut output[base_idx..end_idx])?;
    }
    Ok(())
}

fn execute_idwt_step(
    step: &J2kDirectIdwtStep,
    bands: &mut DirectComponentBandScratch,
) -> Result<()> {
    let output_index = bands.prepare_band(step.output_band_id, step.rect, 0);
    let (input_bands, output_bands) = bands.bands.split_at_mut(output_index);
    let output = &mut output_bands[0].coefficients;
    let ll = find_idwt_band(input_bands, step.ll_band_id)?;
    let hl = find_idwt_band(input_bands, step.hl_band_id)?;
    let lh = find_idwt_band(input_bands, step.lh_band_id)?;
    let hh = find_idwt_band(input_bands, step.hh_band_id)?;
    let job = J2kSingleDecompositionIdwtJob {
        rect: step.rect,
        transform: step.transform,
        ll,
        hl,
        lh,
        hh,
    };
    idwt::apply_single_decomposition_idwt_job(job, output)
}

fn find_idwt_band(bands: &[DirectCpuBand], band_id: J2kDirectBandId) -> Result<J2kIdwtBand<'_>> {
    let band = find_band(bands, band_id)?;
    Ok(J2kIdwtBand {
        rect: band.rect,
        coefficients: &band.coefficients,
    })
}

fn store_component(
    store: &J2kDirectStoreStep,
    bands: &[DirectCpuBand],
    plane: &mut DirectComponentPlane,
    output_written: &mut bool,
) -> Result<()> {
    let input = find_band(bands, store.input_band_id)?;
    if !*output_written {
        plane.width = store.output_width;
        plane.height = store.output_height;
        let required_len = checked_area(store.output_width, store.output_height)?;
        resize_and_zero(&mut plane.samples, required_len);
        *output_written = true;
    }
    if plane.width != store.output_width
        || plane.height != store.output_height
        || plane.samples.len() != checked_area(store.output_width, store.output_height)?
    {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }

    validate_store_bounds(store, input, plane)?;
    let input_width = input.rect.width() as usize;
    let output_width = plane.width as usize;
    let copy_width = store.copy_width as usize;
    for row in 0..store.copy_height as usize {
        let src_start = (store.source_y as usize + row)
            .checked_mul(input_width)
            .and_then(|base| base.checked_add(store.source_x as usize))
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let dst_start = (store.output_y as usize + row)
            .checked_mul(output_width)
            .and_then(|base| base.checked_add(store.output_x as usize))
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let src = &input.coefficients[src_start..src_start + copy_width];
        let dst = &mut plane.samples[dst_start..dst_start + copy_width];
        for (src, dst) in src.iter().zip(dst.iter_mut()) {
            *dst = *src + store.addend;
        }
    }
    Ok(())
}

fn find_band(bands: &[DirectCpuBand], band_id: J2kDirectBandId) -> Result<&DirectCpuBand> {
    bands
        .iter()
        .find(|band| band.band_id == band_id)
        .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into())
}

fn validate_store_bounds(
    store: &J2kDirectStoreStep,
    input: &DirectCpuBand,
    output: &DirectComponentPlane,
) -> Result<()> {
    if store
        .source_x
        .checked_add(store.copy_width)
        .is_none_or(|x| x > input.rect.width())
        || store
            .source_y
            .checked_add(store.copy_height)
            .is_none_or(|y| y > input.rect.height())
        || store
            .output_x
            .checked_add(store.copy_width)
            .is_none_or(|x| x > output.width)
        || store
            .output_y
            .checked_add(store.copy_height)
            .is_none_or(|y| y > output.height)
    {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    Ok(())
}

fn apply_inverse_mct(
    transform: J2kWaveletTransform,
    bit_depths: [u8; 3],
    plane0: &mut DirectComponentPlane,
    plane1: &mut DirectComponentPlane,
    plane2: &mut DirectComponentPlane,
) -> Result<()> {
    if plane0.width != plane1.width
        || plane1.width != plane2.width
        || plane0.height != plane1.height
        || plane1.height != plane2.height
        || plane0.samples.len() != plane1.samples.len()
        || plane1.samples.len() != plane2.samples.len()
    {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }

    let addend0 = sign_addend(bit_depths[0]);
    let addend1 = sign_addend(bit_depths[1]);
    let addend2 = sign_addend(bit_depths[2]);
    for ((y0, y1), y2) in plane0
        .samples
        .iter_mut()
        .zip(plane1.samples.iter_mut())
        .zip(plane2.samples.iter_mut())
    {
        let src0 = *y0;
        let src1 = *y1;
        let src2 = *y2;
        let (out0, out1, out2) = match transform {
            J2kWaveletTransform::Irreversible97 => (
                src0 + 1.402 * src2,
                src0 - 0.34413 * src1 - 0.71414 * src2,
                src0 + 1.772 * src1,
            ),
            J2kWaveletTransform::Reversible53 => {
                let i1 = src0 - floor_f32((src2 + src1) * 0.25);
                (src2 + i1, i1, src1 + i1)
            }
        };
        *y0 = out0 + addend0;
        *y1 = out1 + addend1;
        *y2 = out2 + addend2;
    }
    Ok(())
}

fn write_rgb8_region(
    planes: [&DirectComponentPlane; 3],
    bit_depths: [u8; 3],
    output_region: J2kRect,
    out: &mut [u8],
    stride: usize,
    output: DirectColorU8Output,
) -> Result<()> {
    let width = output_region.width() as usize;
    let height = output_region.height() as usize;
    let bytes_per_pixel = output.bytes_per_pixel();
    let row_bytes = width
        .checked_mul(bytes_per_pixel)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    for plane in planes {
        if output_region.x1 > plane.width || output_region.y1 > plane.height {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
    }

    for y in 0..height {
        let src_y = output_region.y0 as usize + y;
        let dst = &mut out[y * stride..y * stride + row_bytes];
        for x in 0..width {
            let src_x = output_region.x0 as usize + x;
            let dst = &mut dst[x * bytes_per_pixel..x * bytes_per_pixel + bytes_per_pixel];
            for channel in 0..3 {
                let plane = planes[channel];
                let sample = plane.samples[src_y * plane.width as usize + src_x];
                dst[channel] = sample_as_u8(sample, bit_depths[channel]);
            }
            if matches!(output, DirectColorU8Output::Rgba8) {
                dst[3] = u8::MAX;
            }
        }
    }
    Ok(())
}

fn validate_output_region(
    plan: &J2kDirectColorPlan,
    output_region: J2kRect,
    out_len: usize,
    stride: usize,
    output: DirectColorU8Output,
) -> Result<()> {
    if output_region.x1 > plan.dimensions.0
        || output_region.y1 > plan.dimensions.1
        || output_region.x0 > output_region.x1
        || output_region.y0 > output_region.y1
    {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    let row_bytes = output_region
        .width()
        .checked_mul(output.bytes_per_pixel() as u32)
        .and_then(|len| usize::try_from(len).ok())
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    if stride < row_bytes {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    let height = usize::try_from(output_region.height())
        .map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
    let required = if height == 0 {
        0
    } else {
        stride
            .checked_mul(height - 1)
            .and_then(|prefix| prefix.checked_add(row_bytes))
            .ok_or(DecodingError::CodeBlockDecodeFailure)?
    };
    if out_len < required {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    Ok(())
}

fn checked_area(width: u32, height: u32) -> Result<usize> {
    usize::try_from(width)
        .ok()
        .and_then(|width| width.checked_mul(height as usize))
        .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into())
}

fn checked_block_base(output_x: u32, output_y: u32, stride: usize) -> Result<usize> {
    usize::try_from(output_y)
        .ok()
        .and_then(|y| y.checked_mul(stride))
        .and_then(|base| base.checked_add(output_x as usize))
        .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into())
}

fn checked_block_output_len(stride: usize, width: u32, height: u32) -> Result<usize> {
    if height == 0 {
        return Ok(0);
    }
    stride
        .checked_mul(height as usize - 1)
        .and_then(|prefix| prefix.checked_add(width as usize))
        .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into())
}

fn resize_and_zero(buffer: &mut Vec<f32>, len: usize) {
    buffer.resize(len, 0.0);
    buffer.fill(0.0);
}

fn sign_addend(bit_depth: u8) -> f32 {
    (1_u32 << (bit_depth - 1)) as f32
}

fn sample_as_u8(sample: f32, bit_depth: u8) -> u8 {
    let rounded = sample.round();
    if bit_depth == 8 {
        return rounded.clamp(0.0, f32::from(u8::MAX)) as u8;
    }
    let max_value = if bit_depth >= 16 {
        f32::from(u16::MAX)
    } else {
        f32::from(((1_u16 << bit_depth) - 1).max(1))
    };
    ((rounded.clamp(0.0, max_value) / max_value) * f32::from(u8::MAX)).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{encode_htj2k, DecodeSettings, DecoderContext, EncodeOptions, Image};

    fn direct_htj2k_rgb_plan() -> (J2kDirectColorPlan, J2kRect) {
        let pixels = (0..16 * 16 * 3)
            .map(|idx| ((idx * 13 + idx / 3) & 0xff) as u8)
            .collect::<Vec<_>>();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        };
        let bytes = encode_htj2k(&pixels, 16, 16, 3, 8, false, &options).expect("encode HTJ2K RGB");
        let image = Image::new(
            &bytes,
            &DecodeSettings {
                target_resolution: Some((4, 4)),
                ..DecodeSettings::default()
            },
        )
        .expect("scaled image");
        let output_region = J2kRect {
            x0: 1,
            y0: 1,
            x1: 3,
            y1: 3,
        };
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_color_plan_region_with_context(&mut context, (1, 1, 2, 2))
            .expect("direct color plan");
        (plan, output_region)
    }

    #[test]
    fn direct_cpu_scratch_retains_component_buffers_between_executions() {
        let (plan, output_region) = direct_htj2k_rgb_plan();
        let stride = output_region.width() as usize * 3;
        let mut out = vec![0_u8; stride * output_region.height() as usize];
        let mut scratch = J2kDirectCpuScratch::new();

        execute_direct_color_plan_rgb8_into(&plan, output_region, &mut scratch, &mut out, stride)
            .expect("first direct execute");
        let first = scratch.allocation_profile_for_tests();

        execute_direct_color_plan_rgb8_into(&plan, output_region, &mut scratch, &mut out, stride)
            .expect("second direct execute");
        let second = scratch.allocation_profile_for_tests();

        assert_eq!(first.component_band_sets, 3);
        assert_eq!(first.component_planes, 3);
        assert_eq!(second.component_band_sets, first.component_band_sets);
        assert_eq!(second.component_planes, first.component_planes);
        assert_eq!(second.band_buffers, first.band_buffers);
        assert_eq!(
            second.component_sample_capacity,
            first.component_sample_capacity
        );
        assert_eq!(second.band_sample_capacity, first.band_sample_capacity);
        assert!(second.band_sample_capacity >= second.band_sample_len);
        assert!(second.component_sample_capacity >= second.component_sample_len);
    }
}
