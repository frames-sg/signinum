// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{sync::Arc, time::Instant};

use metal::Device;
use rayon::prelude::*;
use signinum_core::{Downscale, PixelFormat, Rect};
use signinum_j2k::J2kError;
use signinum_j2k_native::{
    DecodeSettings as NativeDecodeSettings, DecoderContext as NativeDecoderContext,
    Image as NativeImage,
};

use crate::{direct, Error, J2kDecoder, Surface};

pub(crate) const RGB_REGION_SCALED_METAL_DIRECT_UNSUPPORTED: &str =
    "J2K Metal ROI+scaled hybrid decode currently supports single-tile RGB direct plans for Rgb8/Rgba8/Rgb16";

#[cfg(test)]
static REGION_SCALED_COLOR_PLAN_BUILDS: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
pub(crate) fn reset_region_scaled_color_plan_builds_for_test() {
    REGION_SCALED_COLOR_PLAN_BUILDS.store(0, Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn region_scaled_color_plan_builds_for_test() -> usize {
    REGION_SCALED_COLOR_PLAN_BUILDS.load(Ordering::Relaxed)
}

enum PreparedRegionScaledDirectPlan {
    Gray(crate::compute::PreparedDirectGrayscalePlan),
    Color(crate::compute::PreparedDirectColorPlan),
}

pub(crate) fn decode_region_scaled_direct_to_surface(
    input: &[u8],
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
) -> Result<Option<Surface>, Error> {
    let Some(prepared) = build_region_scaled_direct_plan(input, fmt, roi, scale)? else {
        return Ok(None);
    };
    execute_region_scaled_direct_plan(prepared, fmt)
}

pub(crate) fn decode_region_scaled_direct_to_surface_with_device(
    input: &[u8],
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
    device: &Device,
) -> Result<Option<Surface>, Error> {
    let Some(prepared) = build_region_scaled_direct_plan(input, fmt, roi, scale)? else {
        return Ok(None);
    };
    execute_region_scaled_direct_plan_with_device(prepared, fmt, device)
}

fn build_region_scaled_direct_plan(
    input: &[u8],
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
) -> Result<Option<PreparedRegionScaledDirectPlan>, Error> {
    match fmt {
        PixelFormat::Gray8 | PixelFormat::Gray16 => {
            match build_region_scaled_direct_gray_plan(input, roi, scale) {
                Ok(plan) => Ok(Some(PreparedRegionScaledDirectPlan::Gray(plan))),
                Err(error) if is_direct_region_scaled_runtime_fallback_error(&error) => Ok(None),
                Err(error) => Err(error),
            }
        }
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 => {
            Ok(Some(PreparedRegionScaledDirectPlan::Color(
                build_region_scaled_direct_color_plan(input, roi, scale)?,
            )))
        }
        _ => Ok(None),
    }
}

#[doc(hidden)]
pub(crate) fn benchmark_region_scaled_direct_plan_prepare(
    input: &[u8],
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
) -> Result<(), Error> {
    if build_region_scaled_direct_plan(input, fmt, roi, scale)?.is_some() {
        Ok(())
    } else {
        Err(Error::UnsupportedMetalRequest {
            reason: "J2K MetalDirect ROI+scaled plan preparation is unsupported for this benchmark input",
        })
    }
}

fn execute_region_scaled_direct_plan(
    plan: PreparedRegionScaledDirectPlan,
    fmt: PixelFormat,
) -> Result<Option<Surface>, Error> {
    match plan {
        PreparedRegionScaledDirectPlan::Gray(plan) => {
            match crate::compute::execute_prepared_direct_grayscale_plan(&plan, fmt) {
                Ok(surface) => Ok(Some(surface)),
                Err(error) if is_direct_region_scaled_runtime_fallback_error(&error) => Ok(None),
                Err(error) => Err(error),
            }
        }
        PreparedRegionScaledDirectPlan::Color(plan) => {
            match crate::compute::execute_hybrid_cpu_tier1_direct_color_plan(&plan, fmt) {
                Ok(surface) => Ok(Some(surface)),
                Err(error) if is_direct_region_scaled_runtime_fallback_error(&error) => {
                    Err(Error::UnsupportedMetalRequest {
                        reason: RGB_REGION_SCALED_METAL_DIRECT_UNSUPPORTED,
                    })
                }
                Err(error) => Err(error),
            }
        }
    }
}

fn execute_region_scaled_direct_plan_with_device(
    plan: PreparedRegionScaledDirectPlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Option<Surface>, Error> {
    match plan {
        PreparedRegionScaledDirectPlan::Gray(plan) => {
            match crate::compute::execute_prepared_direct_grayscale_plan_with_device(
                &plan, fmt, device,
            ) {
                Ok(surface) => Ok(Some(surface)),
                Err(error) if is_direct_region_scaled_runtime_fallback_error(&error) => Ok(None),
                Err(error) => Err(error),
            }
        }
        PreparedRegionScaledDirectPlan::Color(plan) => {
            match crate::compute::execute_hybrid_cpu_tier1_direct_color_plan_with_device(
                &plan, fmt, device,
            ) {
                Ok(surface) => Ok(Some(surface)),
                Err(error) if is_direct_region_scaled_runtime_fallback_error(&error) => {
                    Err(Error::UnsupportedMetalRequest {
                        reason: RGB_REGION_SCALED_METAL_DIRECT_UNSUPPORTED,
                    })
                }
                Err(error) => Err(error),
            }
        }
    }
}

pub(crate) fn decode_region_scaled_grayscale_batch_direct_to_device(
    requests: &[(Arc<[u8]>, Rect, Downscale)],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    if !matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K MetalDirect region-scaled grayscale batch does not support {fmt:?}"
            ),
        });
    }

    let mut plans = Vec::with_capacity(requests.len());
    for (input, roi, scale) in requests {
        let plan = build_region_scaled_direct_gray_plan(input.as_ref(), *roi, *scale)?;
        plans.push(Arc::new(plan));
    }
    crate::compute::execute_prepared_direct_grayscale_plan_batch(&plans, fmt)
}

pub(crate) fn decode_region_scaled_color_batch_direct_to_device(
    requests: &[(Arc<[u8]>, Rect, Downscale)],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    if !matches!(
        fmt,
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
    ) {
        return Err(Error::MetalKernel {
            message: format!("J2K MetalDirect region-scaled color batch does not support {fmt:?}"),
        });
    }

    if let Some((input, roi, scale)) = repeated_region_scaled_request(requests) {
        let plan = Arc::new(build_region_scaled_direct_color_plan(
            input.as_ref(),
            roi,
            scale,
        )?);
        let plans = vec![plan; requests.len()];
        return crate::compute::execute_hybrid_cpu_tier1_direct_color_plan_batch(&plans, fmt);
    }

    let plans = requests
        .par_iter()
        .map(|(input, roi, scale)| {
            build_region_scaled_direct_color_plan(input.as_ref(), *roi, *scale).map(Arc::new)
        })
        .collect::<Result<Vec<_>, _>>()?;
    crate::compute::execute_hybrid_cpu_tier1_direct_color_plan_batch(&plans, fmt)
}

fn repeated_region_scaled_request(
    requests: &[(Arc<[u8]>, Rect, Downscale)],
) -> Option<(&Arc<[u8]>, Rect, Downscale)> {
    let (first_input, first_roi, first_scale) = requests.first()?;
    requests
        .iter()
        .all(|(input, roi, scale)| {
            *roi == *first_roi
                && *scale == *first_scale
                && (Arc::ptr_eq(input, first_input) || input.as_ref() == first_input.as_ref())
        })
        .then_some((first_input, *first_roi, *first_scale))
}

fn build_region_scaled_direct_gray_plan(
    input: &[u8],
    roi: Rect,
    scale: Downscale,
) -> Result<crate::compute::PreparedDirectGrayscalePlan, Error> {
    let image = build_region_scaled_native_image(input, scale)?;
    let mut context = NativeDecoderContext::default();
    let output_region = roi.scaled_covering(scale);
    let plan = match image.build_direct_grayscale_plan_region_with_context(
        &mut context,
        (
            output_region.x,
            output_region.y,
            output_region.w,
            output_region.h,
        ),
    ) {
        Ok(plan) => plan,
        Err(error) if direct::is_unsupported_direct_plan_error(&error.to_string()) => {
            return Err(Error::MetalKernel {
                message: format!(
                    "explicit J2K MetalDirect region-scaled batch currently supports grayscale direct plans only: {error}"
                ),
            });
        }
        Err(error) => {
            return Err(Error::Decode(J2kError::Backend(format!(
                "failed to build J2K MetalDirect region-scaled grayscale plan: {error}"
            ))));
        }
    };
    let mut prepared = crate::compute::prepare_direct_grayscale_plan(&plan)?;
    crate::compute::crop_prepared_direct_grayscale_plan_to_output_region(
        &mut prepared,
        output_region,
    )?;
    Ok(prepared)
}

fn build_region_scaled_direct_color_plan(
    input: &[u8],
    roi: Rect,
    scale: Downscale,
) -> Result<crate::compute::PreparedDirectColorPlan, Error> {
    #[cfg(test)]
    REGION_SCALED_COLOR_PLAN_BUILDS.fetch_add(1, Ordering::Relaxed);

    let profile_stages = crate::compute::metal_profile_stages_enabled();
    let total_started = profile_stages.then(Instant::now);
    let native_image_started = profile_stages.then(Instant::now);
    let image = build_region_scaled_native_image(input, scale)?;
    let native_image_us = native_image_started.map(elapsed_us).unwrap_or_default();
    let direct_plan_started = profile_stages.then(Instant::now);
    let mut context = NativeDecoderContext::default();
    let output_region = roi.scaled_covering(scale);
    let plan = match image.build_direct_color_plan_region_with_context(
        &mut context,
        (
            output_region.x,
            output_region.y,
            output_region.w,
            output_region.h,
        ),
    ) {
        Ok(plan) => plan,
        Err(error) if direct::is_unsupported_direct_plan_error(&error.to_string()) => {
            return Err(Error::UnsupportedMetalRequest {
                reason: RGB_REGION_SCALED_METAL_DIRECT_UNSUPPORTED,
            });
        }
        Err(error) => {
            return Err(Error::Decode(J2kError::Backend(format!(
                "failed to build J2K MetalDirect region-scaled color plan: {error}"
            ))));
        }
    };
    let direct_plan_us = direct_plan_started.map(elapsed_us).unwrap_or_default();
    let prepare_started = profile_stages.then(Instant::now);
    let mut prepared = crate::compute::prepare_direct_color_plan_for_cpu_upload(&plan)?;
    let prepare_us = prepare_started.map(elapsed_us).unwrap_or_default();
    let crop_started = profile_stages.then(Instant::now);
    crate::compute::crop_prepared_direct_color_plan_to_output_region(&mut prepared, output_region)?;
    let crop_us = crop_started.map(elapsed_us).unwrap_or_default();
    if let Some(started) = total_started {
        emit_region_scaled_color_plan_build_timings(
            native_image_us,
            direct_plan_us,
            prepare_us,
            crop_us,
            elapsed_us(started),
        );
    }
    Ok(prepared)
}

fn elapsed_us(started: Instant) -> u128 {
    started.elapsed().as_micros()
}

fn emit_region_scaled_color_plan_build_timings(
    native_image_us: u128,
    direct_plan_us: u128,
    prepare_us: u128,
    crop_us: u128,
    total_us: u128,
) {
    if !crate::compute::metal_profile_stages_enabled() {
        return;
    }

    for (stage, elapsed_us) in [
        ("native_image", native_image_us),
        ("direct_color_plan", direct_plan_us),
        ("prepare_cpu_upload", prepare_us),
        ("crop_prepared_plan", crop_us),
        ("plan_total", total_us),
    ] {
        eprintln!(
            "signinum_profile codec=j2k op=decode path=metal_direct_hybrid_plan stage={stage} fmt=Rgb batch_count=1 elapsed_us={elapsed_us}"
        );
    }
}

fn build_region_scaled_native_image(
    input: &[u8],
    scale: Downscale,
) -> Result<NativeImage<'_>, Error> {
    let decoder = J2kDecoder::new(input)?;
    let dims = decoder.inner.info().dimensions;
    let target_dims = (
        dims.0.div_ceil(scale.denominator()),
        dims.1.div_ceil(scale.denominator()),
    );
    let settings = NativeDecodeSettings {
        target_resolution: Some(target_dims),
        ..NativeDecodeSettings::default()
    };
    let image =
        NativeImage::new(input, &settings).map_err(|error| J2kError::Backend(error.to_string()))?;
    Ok(image)
}

fn is_direct_region_scaled_runtime_fallback_error(error: &Error) -> bool {
    crate::is_direct_runtime_fallback_error(error)
}
