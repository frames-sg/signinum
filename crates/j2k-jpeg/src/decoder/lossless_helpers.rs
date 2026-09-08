// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    downscale_profile_name, emit_jpeg_profile_fields, lossless_predict, upsample_h2v1_sample_at,
    upsample_h2v2_rows_at, BitReader, ColorSpace, DownscaleFactor, Duration, Info, JpegError,
    LosslessColorSampling, LosslessSample, MarkerKind, PreparedLosslessPlan, ProfileField, Rect,
    RestartIndex, RestartSegment, SofKind,
};
use crate::allocation::{checked_allocation_bytes, try_vec_with_capacity};
use crate::entropy::huffman::DcHuffmanTable;
use crate::entropy::sequential::PreparedDecodePlan;

pub(crate) fn restart_index_allocation_bytes(
    info: &Info,
    restart_interval: Option<u16>,
) -> Result<usize, JpegError> {
    let Some(interval_mcus) = restart_interval
        .filter(|&interval| interval > 0)
        .map(u32::from)
    else {
        return Ok(0);
    };
    let capacity = restart_segment_capacity(info.mcu_geometry.count, interval_mcus)?;
    checked_allocation_bytes::<RestartSegment>(capacity)
}

pub(super) fn restart_index_for_stream(
    bytes: &[u8],
    scan_data_offset: Option<usize>,
    info: &Info,
    restart_interval: Option<u16>,
) -> Result<Option<RestartIndex>, JpegError> {
    let Some(interval_mcus) = restart_interval
        .filter(|&interval| interval > 0)
        .map(u32::from)
    else {
        return Ok(None);
    };
    let scan_data_offset = scan_data_offset.ok_or(JpegError::MissingMarker {
        marker: MarkerKind::Sos,
    })?;
    if !matches!(info.sof_kind, SofKind::Baseline8 | SofKind::Extended8) || info.scan_count != 1 {
        return Err(JpegError::NotImplemented { sof: info.sof_kind });
    }
    let total_mcus = info.mcu_geometry.count;
    let expected_restarts = total_mcus.saturating_sub(1) / interval_mcus;
    let segment_capacity = restart_segment_capacity(total_mcus, interval_mcus)?;
    let mut segments = try_vec_with_capacity(segment_capacity)?;
    segments.push(RestartSegment {
        start_mcu: 0,
        entropy_offset: scan_data_offset,
        marker_offset: None,
        marker: None,
    });

    let mut found_restarts = 0u32;
    let mut expected_rst = 0xd0u8;
    let mut pos = scan_data_offset;
    while pos < bytes.len() {
        if bytes[pos] != 0xff {
            pos += 1;
            continue;
        }

        let mut marker_code_pos = pos + 1;
        while marker_code_pos < bytes.len() && bytes[marker_code_pos] == 0xff {
            marker_code_pos += 1;
        }
        if marker_code_pos >= bytes.len() {
            return Err(JpegError::Truncated {
                offset: pos,
                expected: 1,
            });
        }

        let marker = bytes[marker_code_pos];
        let marker_offset = marker_code_pos - 1;
        match marker {
            0x00 => pos = marker_code_pos + 1,
            0xd0..=0xd7 => {
                if found_restarts >= expected_restarts {
                    return Err(JpegError::UnexpectedMarker {
                        offset: marker_offset,
                        expected: MarkerKind::Eoi,
                        found: marker,
                    });
                }
                if marker != expected_rst {
                    return Err(JpegError::RestartMismatch {
                        offset: marker_offset,
                        expected: expected_rst & 0x07,
                        found: marker,
                    });
                }
                found_restarts += 1;
                segments.push(RestartSegment {
                    start_mcu: found_restarts.saturating_mul(interval_mcus),
                    entropy_offset: marker_code_pos + 1,
                    marker_offset: Some(marker_offset),
                    marker: Some(marker),
                });
                expected_rst = if expected_rst == 0xd7 {
                    0xd0
                } else {
                    expected_rst + 1
                };
                pos = marker_code_pos + 1;
            }
            0xd9 => {
                if found_restarts != expected_restarts {
                    return Err(JpegError::UnexpectedEoi {
                        mcu_at: found_restarts
                            .saturating_add(1)
                            .saturating_mul(interval_mcus),
                        mcu_total: total_mcus,
                    });
                }
                return Ok(Some(RestartIndex {
                    scan_data_offset,
                    interval_mcus,
                    segments,
                }));
            }
            found => {
                return Err(JpegError::UnexpectedMarker {
                    offset: marker_offset,
                    expected: MarkerKind::Eoi,
                    found,
                });
            }
        }
    }

    Err(JpegError::MissingMarker {
        marker: MarkerKind::Eoi,
    })
}

fn restart_segment_capacity(total_mcus: u32, interval_mcus: u32) -> Result<usize, JpegError> {
    let expected_restarts = total_mcus.saturating_sub(1) / interval_mcus;
    let capacity = usize::try_from(expected_restarts)
        .ok()
        .and_then(|count| count.checked_add(1))
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })?;
    checked_allocation_bytes::<RestartSegment>(capacity)?;
    Ok(capacity)
}

pub(super) fn emit_decode_scan_profile(
    scan_path: &str,
    dimensions: (u32, u32),
    decoded: Rect,
    downscale: DownscaleFactor,
    elapsed: Duration,
) {
    emit_jpeg_profile_fields("jpeg_lossless_scan_fields", "decode_scan", "cpu", || {
        Ok([
            ProfileField::label("scan_path", scan_path)?,
            ProfileField::label("downscale", downscale_profile_name(downscale))?,
            ProfileField::metric_with_summary("source_width", dimensions.0, false)?,
            ProfileField::metric_with_summary("source_height", dimensions.1, false)?,
            ProfileField::metric_with_summary("decoded_x", decoded.x, false)?,
            ProfileField::metric_with_summary("decoded_y", decoded.y, false)?,
            ProfileField::metric_with_summary("decoded_w", decoded.w, false)?,
            ProfileField::metric_with_summary("decoded_h", decoded.h, false)?,
            ProfileField::metric("scan_us", elapsed.as_micros())?,
        ])
    });
}

pub(super) fn consume_lossless_restart(
    br: &mut BitReader<'_>,
    sample_index: u32,
    total_samples: u32,
    expected_rst: &mut u8,
) -> Result<(), JpegError> {
    br.reset_at_restart();
    *expected_rst = br.consume_restart_marker(*expected_rst, sample_index, total_samples)?;
    Ok(())
}

pub(super) struct LosslessRestartTracker {
    restart_interval: u32,
    total_units: u32,
    units_since_restart: u32,
    expected_rst: u8,
}

impl LosslessRestartTracker {
    pub(super) fn new(restart_interval: Option<u16>, total_units: u32) -> Self {
        Self {
            restart_interval: u32::from(restart_interval.unwrap_or(0)),
            total_units,
            units_since_restart: 0,
            expected_rst: 0,
        }
    }

    pub(super) fn begin_unit(
        &mut self,
        br: &mut BitReader<'_>,
        unit_index: u32,
    ) -> Result<bool, JpegError> {
        if self.restart_interval > 0 && self.units_since_restart == self.restart_interval {
            consume_lossless_restart(br, unit_index, self.total_units, &mut self.expected_rst)?;
            self.units_since_restart = 0;
        }
        Ok(self.restart_interval > 0 && self.units_since_restart == 0)
    }

    pub(super) fn finish_unit(&mut self) {
        self.units_since_restart += 1;
    }
}

pub(super) fn consume_extended12_restart(
    br: &mut BitReader<'_>,
    mcu_index: u32,
    total_mcus: u32,
    expected_rst: &mut u8,
) -> Result<(), JpegError> {
    *expected_rst = br.consume_restart_marker(*expected_rst, mcu_index, total_mcus)?;
    Ok(())
}

pub(super) struct Extended12RestartTracker {
    restart_interval: u32,
    total_mcus: u32,
    mcus_since_restart: u32,
    expected_rst: u8,
}

impl Extended12RestartTracker {
    pub(super) fn new(restart_interval: Option<u16>, total_mcus: u32) -> Self {
        Self {
            restart_interval: u32::from(restart_interval.unwrap_or(0)),
            total_mcus,
            mcus_since_restart: 0,
            expected_rst: 0,
        }
    }

    pub(super) fn begin_mcu(
        &mut self,
        br: &mut BitReader<'_>,
        mcu_index: u32,
    ) -> Result<bool, JpegError> {
        if self.restart_interval > 0 && self.mcus_since_restart == self.restart_interval {
            consume_extended12_restart(br, mcu_index, self.total_mcus, &mut self.expected_rst)?;
            self.mcus_since_restart = 0;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub(super) fn finish_mcu(&mut self) {
        self.mcus_since_restart += 1;
    }
}

pub(super) fn lossless_predictor_value(
    predictor: u8,
    out: &[u8],
    stride: usize,
    x: usize,
    y: usize,
) -> i32 {
    lossless_predict(predictor, 128, x, y, |sx, sy| {
        i32::from(out[sy * stride + sx])
    })
}

pub(super) fn lossless_predictor_color_into<P: LosslessSample>(
    predictor: u8,
    out: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    component: usize,
) -> i32 {
    lossless_predict(predictor, P::RESTART_PREDICTOR, x, y, |sx, sy| {
        P::read_le(&out[sy * stride + (sx * 3 + component) * P::BYTES..])
    })
}

pub(super) fn lossless_predictor_gray_rows<P: LosslessSample>(
    predictor: u8,
    curr_row: &[u8],
    prev_row: &[u8],
    x: usize,
    y: usize,
) -> i32 {
    lossless_predict(predictor, P::RESTART_PREDICTOR, x, y, |sx, sy| {
        let row = if sy == y { curr_row } else { prev_row };
        P::read_le(&row[sx * P::BYTES..])
    })
}

pub(super) fn lossless_predictor_color_rows<P: LosslessSample>(
    predictor: u8,
    curr_row: &[u8],
    prev_row: &[u8],
    x: usize,
    y: usize,
    component: usize,
) -> i32 {
    lossless_predict(predictor, P::RESTART_PREDICTOR, x, y, |sx, sy| {
        let row = if sy == y { curr_row } else { prev_row };
        P::read_le(&row[(sx * 3 + component) * P::BYTES..])
    })
}

pub(super) trait LosslessColorSampleTarget<P: LosslessSample> {
    fn predict(&self, predictor: u8, component: usize) -> i32;
    fn write(&mut self, component: usize, sample: P);
}

pub(super) struct LosslessColorIntoSample<'a> {
    pub(super) out: &'a mut [u8],
    pub(super) stride: usize,
    pub(super) x: usize,
    pub(super) y: usize,
}

impl<P: LosslessSample> LosslessColorSampleTarget<P> for LosslessColorIntoSample<'_> {
    fn predict(&self, predictor: u8, component: usize) -> i32 {
        lossless_predictor_color_into::<P>(
            predictor,
            self.out,
            self.stride,
            self.x,
            self.y,
            component,
        )
    }

    fn write(&mut self, component: usize, sample: P) {
        let offset = self.y * self.stride + (self.x * 3 + component) * P::BYTES;
        sample.write_le(&mut self.out[offset..]);
    }
}

pub(super) struct LosslessColorRowSample<'a> {
    pub(super) curr_row: &'a mut [u8],
    pub(super) prev_row: &'a [u8],
    pub(super) x: usize,
    pub(super) y: usize,
}

impl<P: LosslessSample> LosslessColorSampleTarget<P> for LosslessColorRowSample<'_> {
    fn predict(&self, predictor: u8, component: usize) -> i32 {
        lossless_predictor_color_rows::<P>(
            predictor,
            self.curr_row,
            self.prev_row,
            self.x,
            self.y,
            component,
        )
    }

    fn write(&mut self, component: usize, sample: P) {
        let offset = (self.x * 3 + component) * P::BYTES;
        sample.write_le(&mut self.curr_row[offset..]);
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "component indices are validated against the JPEG maximum component count"
)]
pub(super) fn resolve_lossless_color_components(
    decode_plan: &PreparedDecodePlan,
) -> Result<[ResolvedLosslessColorComponent<'_>; 3], JpegError> {
    if decode_plan.components.len() != 3 {
        return Err(JpegError::UnsupportedComponentCount {
            count: decode_plan.components.len() as u8,
        });
    }
    let resolve = |index: usize| {
        let component = &decode_plan.components[index];
        if component.output_index >= 3 {
            return Err(JpegError::UnsupportedComponentCount {
                count: decode_plan.components.len() as u8,
            });
        }
        Ok(ResolvedLosslessColorComponent {
            h: usize::from(component.h),
            v: usize::from(component.v),
            output_index: component.output_index,
            dc_table: decode_plan.dc_table(component)?,
        })
    };
    Ok([resolve(0)?, resolve(1)?, resolve(2)?])
}

#[derive(Clone, Copy)]
pub(super) struct ResolvedLosslessColorComponent<'a> {
    h: usize,
    v: usize,
    output_index: usize,
    dc_table: DcHuffmanTable<'a>,
}

pub(super) fn decode_lossless_color_sample<P, T>(
    br: &mut BitReader<'_>,
    components: &[ResolvedLosslessColorComponent<'_>; 3],
    predictor: u8,
    restart_first_sample: bool,
    target: &mut T,
) -> Result<(), JpegError>
where
    P: LosslessSample,
    T: LosslessColorSampleTarget<P>,
{
    for component in components {
        let predicted = if restart_first_sample {
            P::RESTART_PREDICTOR
        } else {
            target.predict(predictor, component.output_index)
        };
        let diff = component.dc_table.decode_fast_dc(br)?;
        let sample = P::from_i32(predicted + diff)?;
        target.write(component.output_index, sample);
    }
    Ok(())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "component counts are validated against the JPEG maximum before error reporting"
)]
pub(super) fn validate_lossless_color_plan<P: LosslessSample>(
    plan: &PreparedLosslessPlan,
    decode_plan: &PreparedDecodePlan,
    info: &Info,
    color_space: ColorSpace,
) -> Result<(), JpegError> {
    if plan.bit_depth != P::BIT_DEPTH {
        return Err(JpegError::UnsupportedBitDepth {
            depth: plan.bit_depth,
        });
    }
    if info.color_space != color_space {
        return Err(JpegError::NotImplemented { sof: info.sof_kind });
    }
    if decode_plan.components.len() != 3 {
        return Err(JpegError::UnsupportedComponentCount {
            count: decode_plan.components.len() as u8,
        });
    }
    if !(1..=7).contains(&plan.predictor) {
        return Err(JpegError::UnsupportedPredictor {
            predictor: plan.predictor,
        });
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) struct LosslessPlaneSample {
    pub(super) x: usize,
    pub(super) y: usize,
    pub(super) restart_first_sample: bool,
}

pub(super) fn decode_lossless_plane_sample<P: LosslessSample>(
    br: &mut BitReader<'_>,
    table: DcHuffmanTable<'_>,
    predictor: u8,
    plane: &mut [P],
    width: usize,
    sample: LosslessPlaneSample,
) -> Result<(), JpegError> {
    let predicted = if sample.restart_first_sample {
        P::RESTART_PREDICTOR
    } else {
        lossless_predictor_plane(predictor, plane, width, sample.x, sample.y)
    };
    let diff = table.decode_fast_dc(br)?;
    plane[sample.y * width + sample.x] = P::from_i32(predicted + diff)?;
    Ok(())
}

pub(super) struct LosslessSampledColorPlanesMut<'a, P> {
    pub(super) c0: &'a mut [P],
    pub(super) c1: &'a mut [P],
    pub(super) c2: &'a mut [P],
    pub(super) dimensions: (usize, usize),
    pub(super) chroma_dimensions: (usize, usize),
}

impl<P> LosslessSampledColorPlanesMut<'_, P> {
    fn component_plane(&mut self, output_index: usize) -> Option<(&mut [P], usize, usize)> {
        match output_index {
            0 => Some((&mut *self.c0, self.dimensions.0, self.dimensions.1)),
            1 => Some((
                &mut *self.c1,
                self.chroma_dimensions.0,
                self.chroma_dimensions.1,
            )),
            2 => Some((
                &mut *self.c2,
                self.chroma_dimensions.0,
                self.chroma_dimensions.1,
            )),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct LosslessSampledMcu {
    pub(super) x: usize,
    pub(super) y: usize,
    pub(super) restart_first_mcu: bool,
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "component indices are validated against the JPEG maximum component count"
)]
pub(super) fn decode_lossless_sampled_color_mcu<P>(
    br: &mut BitReader<'_>,
    components: &[ResolvedLosslessColorComponent<'_>; 3],
    predictor: u8,
    mcu: LosslessSampledMcu,
    planes: &mut LosslessSampledColorPlanesMut<'_, P>,
) -> Result<(), JpegError>
where
    P: LosslessSample,
{
    for component in components {
        let Some((plane, plane_width, plane_height)) =
            planes.component_plane(component.output_index)
        else {
            return Err(JpegError::UnsupportedComponentCount {
                count: components.len() as u8,
            });
        };
        for local_y in 0..component.v {
            for local_x in 0..component.h {
                let x = mcu.x * component.h + local_x;
                let y = mcu.y * component.v + local_y;
                if x >= plane_width || y >= plane_height {
                    continue;
                }
                decode_lossless_plane_sample(
                    br,
                    component.dc_table,
                    predictor,
                    plane,
                    plane_width,
                    LosslessPlaneSample {
                        x,
                        y,
                        restart_first_sample: mcu.restart_first_mcu && local_x == 0 && local_y == 0,
                    },
                )?;
            }
        }
    }
    Ok(())
}

pub(super) fn lossless_predictor_plane<P: LosslessSample>(
    predictor: u8,
    plane: &[P],
    width: usize,
    x: usize,
    y: usize,
) -> i32 {
    lossless_predict(predictor, P::RESTART_PREDICTOR, x, y, |sx, sy| {
        plane[sy * width + sx].into()
    })
}

pub(super) struct LosslessColorPlanes<'a, P> {
    pub(super) c0: &'a [P],
    pub(super) c1: &'a [P],
    pub(super) c2: &'a [P],
}

type H2v1Upsample<P> = fn(&[P], usize) -> P;
type H2v2Upsample<P> = fn(&[P], usize, usize, usize, usize, usize) -> P;
type LosslessColorConvert<P> = fn(ColorSpace, P, P, P) -> (P, P, P);

struct LosslessSampledOutput<'out, 'planes, P> {
    out: &'out mut [u8],
    stride: usize,
    color_space: ColorSpace,
    sampling: LosslessColorSampling,
    dimensions: (usize, usize),
    planes: LosslessColorPlanes<'planes, P>,
    upsample_h2v1: H2v1Upsample<P>,
    upsample_h2v2: H2v2Upsample<P>,
    convert: LosslessColorConvert<P>,
}

pub(super) fn write_lossless_color8_sampled_output(
    out: &mut [u8],
    stride: usize,
    color_space: ColorSpace,
    sampling: LosslessColorSampling,
    dimensions: (usize, usize),
    planes: LosslessColorPlanes<'_, u8>,
) {
    write_lossless_color_sampled_output(LosslessSampledOutput {
        out,
        stride,
        color_space,
        sampling,
        dimensions,
        planes,
        upsample_h2v1: upsample_h2v1_u8_at,
        upsample_h2v2: upsample_h2v2_u8_at,
        convert: lossless_color8_to_rgb,
    });
}

pub(super) fn write_lossless_color16_sampled_output(
    out: &mut [u8],
    stride: usize,
    color_space: ColorSpace,
    sampling: LosslessColorSampling,
    dimensions: (usize, usize),
    planes: LosslessColorPlanes<'_, u16>,
) {
    write_lossless_color_sampled_output(LosslessSampledOutput {
        out,
        stride,
        color_space,
        sampling,
        dimensions,
        planes,
        upsample_h2v1: upsample_h2v1_u16_at,
        upsample_h2v2: upsample_h2v2_u16_at,
        convert: lossless_color16_to_rgb,
    });
}

fn write_lossless_color_sampled_output<P>(request: LosslessSampledOutput<'_, '_, P>)
where
    P: LosslessSample,
{
    let LosslessSampledOutput {
        out,
        stride,
        color_space,
        sampling,
        dimensions,
        planes,
        upsample_h2v1,
        upsample_h2v2,
        convert,
    } = request;
    let (width, height) = dimensions;
    let chroma_width = width.div_ceil(2);
    let chroma_height = match sampling {
        LosslessColorSampling::S422 => height,
        LosslessColorSampling::S420 => height.div_ceil(2),
        LosslessColorSampling::S444 => unreachable!("sampled writer is not used for 4:4:4"),
    };
    for y in 0..height {
        for x in 0..width {
            let c0_sample = planes.c0[y * width + x];
            let (c1_sample, c2_sample) = match sampling {
                LosslessColorSampling::S422 => {
                    let c1_row = &planes.c1[y * chroma_width..(y + 1) * chroma_width];
                    let c2_row = &planes.c2[y * chroma_width..(y + 1) * chroma_width];
                    (upsample_h2v1(c1_row, x), upsample_h2v1(c2_row, x))
                }
                LosslessColorSampling::S420 => (
                    upsample_h2v2(planes.c1, chroma_width, chroma_height, width, x, y),
                    upsample_h2v2(planes.c2, chroma_width, chroma_height, width, x, y),
                ),
                LosslessColorSampling::S444 => unreachable!("sampled writer is not used for 4:4:4"),
            };
            let (r, g, b) = convert(color_space, c0_sample, c1_sample, c2_sample);
            let dst = y * stride + x * 3 * P::BYTES;
            r.write_le(&mut out[dst..dst + P::BYTES]);
            g.write_le(&mut out[dst + P::BYTES..dst + 2 * P::BYTES]);
            b.write_le(&mut out[dst + 2 * P::BYTES..dst + 3 * P::BYTES]);
        }
    }
}

fn lossless_color8_to_rgb(color_space: ColorSpace, c0: u8, c1: u8, c2: u8) -> (u8, u8, u8) {
    match color_space {
        ColorSpace::Rgb => (c0, c1, c2),
        ColorSpace::YCbCr => crate::color::ycbcr::ycbcr_to_rgb(c0, c1, c2),
        _ => unreachable!("lossless sampled color path only accepts RGB/YCbCr"),
    }
}

fn lossless_color16_to_rgb(color_space: ColorSpace, c0: u16, c1: u16, c2: u16) -> (u16, u16, u16) {
    match color_space {
        ColorSpace::Rgb => (c0, c1, c2),
        ColorSpace::YCbCr => crate::color::ycbcr::ycbcr16_to_rgb16(c0, c1, c2),
        _ => unreachable!("lossless sampled color path only accepts RGB/YCbCr"),
    }
}

pub(super) fn upsample_h2v2_u8_at(
    plane: &[u8],
    chroma_width: usize,
    chroma_height: usize,
    output_width: usize,
    output_x: usize,
    output_y: usize,
) -> u8 {
    debug_assert!(!plane.is_empty());
    debug_assert!(chroma_width > 0);
    debug_assert!(chroma_height > 0);
    let chroma_y = output_y / 2;
    let current = &plane[chroma_y * chroma_width..(chroma_y + 1) * chroma_width];
    let near_y = if output_y.is_multiple_of(2) {
        chroma_y.saturating_sub(1)
    } else {
        (chroma_y + 1).min(chroma_height - 1)
    };
    let near = &plane[near_y * chroma_width..(near_y + 1) * chroma_width];
    upsample_h2v2_rows_at(current, near, output_width, output_x)
}

pub(super) fn upsample_h2v1_u8_at(row: &[u8], output_x: usize) -> u8 {
    upsample_h2v1_sample_at(row, output_x)
}

pub(super) fn upsample_h2v2_u16_at(
    plane: &[u16],
    chroma_width: usize,
    chroma_height: usize,
    output_width: usize,
    output_x: usize,
    output_y: usize,
) -> u16 {
    debug_assert!(!plane.is_empty());
    debug_assert!(chroma_width > 0);
    debug_assert!(chroma_height > 0);
    let chroma_y = output_y / 2;
    let current = &plane[chroma_y * chroma_width..(chroma_y + 1) * chroma_width];
    let near_y = if output_y.is_multiple_of(2) {
        chroma_y.saturating_sub(1)
    } else {
        (chroma_y + 1).min(chroma_height - 1)
    };
    let near = &plane[near_y * chroma_width..(near_y + 1) * chroma_width];
    upsample_h2v2_rows_at(current, near, output_width, output_x)
}

pub(super) fn upsample_h2v1_u16_at(row: &[u16], output_x: usize) -> u16 {
    upsample_h2v1_sample_at(row, output_x)
}

pub(super) fn lossless_predictor_value_u16(
    predictor: u8,
    out: &[u8],
    stride: usize,
    x: usize,
    y: usize,
) -> i32 {
    lossless_predict(predictor, 32768, x, y, |sx, sy| {
        i32::from(read_gray16_sample(out, sy * stride + sx * 2))
    })
}

pub(super) fn read_gray16_sample(out: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([out[offset], out[offset + 1]])
}

#[cfg(test)]
mod tests;
