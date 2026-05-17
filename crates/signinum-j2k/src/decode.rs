// SPDX-License-Identifier: Apache-2.0

use crate::backend::{ColorSpace, DecodeSettings, DecodedComponents, Image, RawBitmap};
use crate::{backend, J2kError, J2kScratchPool};
use alloc::{string::ToString, vec::Vec};
use core::convert::Infallible;
use signinum_core::{
    validate_strided_output_buffer, DecodeOutcome, Downscale, PixelFormat, Rect, Unsupported,
};
pub(crate) type J2kDecodeOutcome = DecodeOutcome<Infallible>;

pub(crate) fn decode_scaled_from_info(
    bytes: &[u8],
    full_dims: (u32, u32),
    pool: &mut J2kScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    scale: Downscale,
) -> Result<J2kDecodeOutcome, J2kError> {
    validate_supported_format(fmt)?;
    let target_dims = (
        full_dims.0.div_ceil(scale.denominator()),
        full_dims.1.div_ceil(scale.denominator()),
    );
    let settings = DecodeSettings {
        target_resolution: Some(target_dims),
        ..DecodeSettings::default()
    };
    let _ = pool;
    let _ = full_dims;
    decode_with_settings(bytes, settings, out, stride, fmt)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_region_scaled_from_info(
    bytes: &[u8],
    full_dims: (u32, u32),
    pool: &mut J2kScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
) -> Result<J2kDecodeOutcome, J2kError> {
    validate_supported_format(fmt)?;
    validate_region(roi, full_dims)?;
    if scale == Downscale::None {
        return decode_region(bytes, pool, out, stride, fmt, roi);
    }

    let target_dims = (
        full_dims.0.div_ceil(scale.denominator()),
        full_dims.1.div_ceil(scale.denominator()),
    );
    let scaled_roi = roi.scaled_covering(scale);
    validate_buffer((scaled_roi.w, scaled_roi.h), out.len(), stride, fmt)?;
    let settings = DecodeSettings {
        target_resolution: Some(target_dims),
        ..DecodeSettings::default()
    };
    let image = backend::image(bytes, settings)?;
    let image_dims = (image.width(), image.height());
    validate_region(scaled_roi, image_dims)?;
    decode_image_region_into(&image, out, stride, fmt, scaled_roi)?;

    Ok(DecodeOutcome {
        decoded: scaled_roi,
        warnings: Vec::new(),
    })
}

pub(crate) fn decode_region(
    bytes: &[u8],
    _pool: &mut J2kScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<J2kDecodeOutcome, J2kError> {
    validate_supported_format(fmt)?;
    let image = backend::image(bytes, DecodeSettings::default())?;
    let dims = (image.width(), image.height());
    validate_region(roi, dims)?;
    validate_buffer((roi.w, roi.h), out.len(), stride, fmt)?;
    decode_image_region_into(&image, out, stride, fmt, roi)?;

    Ok(DecodeOutcome {
        decoded: roi,
        warnings: Vec::new(),
    })
}

pub(crate) fn decode_with_settings(
    bytes: &[u8],
    settings: DecodeSettings,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<J2kDecodeOutcome, J2kError> {
    validate_supported_format(fmt)?;
    let image = backend::image(bytes, settings)?;
    let dims = (image.width(), image.height());
    validate_buffer(dims, out.len(), stride, fmt)?;
    decode_image_into(&image, out, stride, fmt)?;
    Ok(DecodeOutcome {
        decoded: Rect::full(dims),
        warnings: Vec::new(),
    })
}

fn decode_image_into(
    image: &Image<'_>,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    let mut native_context = signinum_j2k_native::DecoderContext::default();
    decode_image_into_with_native_context(image, &mut native_context, out, stride, fmt)
}

pub(crate) fn decode_image_into_with_native_context<'a>(
    image: &Image<'a>,
    native_context: &mut signinum_j2k_native::DecoderContext<'a>,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    let dims = (image.width(), image.height());
    match fmt {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Gray8 => {
            if can_decode_u8_directly(image.color_space(), image.has_alpha(), dims, stride, fmt) {
                image
                    .decode_into(out, native_context)
                    .map_err(|err| J2kError::Backend(err.to_string()))?;
                return Ok(());
            }
            let decoded = image
                .decode_with_context(native_context)
                .map_err(|err| J2kError::Backend(err.to_string()))?;
            write_u8_output(
                image.color_space(),
                image.has_alpha(),
                dims,
                &decoded.data,
                out,
                stride,
                fmt,
            )
        }
        PixelFormat::Rgb16 | PixelFormat::Gray16 => {
            let raw = image
                .decode_native_with_context(native_context)
                .map_err(|err| J2kError::Backend(err.to_string()))?;
            write_u16_output(
                image.color_space(),
                image.has_alpha(),
                &raw,
                out,
                stride,
                fmt,
            )
        }
        PixelFormat::Rgba16 => unreachable!("validated above"),
        _ => Err(Unsupported {
            what: "pixel format is not yet supported by signinum-j2k",
        }
        .into()),
    }
}

fn can_decode_u8_directly(
    color_space: &ColorSpace,
    has_alpha: bool,
    dims: (u32, u32),
    stride: usize,
    fmt: PixelFormat,
) -> bool {
    let width = dims.0 as usize;
    match (color_space, has_alpha, fmt) {
        (ColorSpace::RGB, false, PixelFormat::Rgb8) => stride == width * 3,
        (ColorSpace::RGB, true, PixelFormat::Rgba8) => stride == width * 4,
        (ColorSpace::Gray, false, PixelFormat::Gray8) => stride == width,
        _ => false,
    }
}

fn decode_image_region_into(
    image: &Image<'_>,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<(), J2kError> {
    let mut native_context = signinum_j2k_native::DecoderContext::default();
    decode_image_region_into_with_native_context(image, &mut native_context, out, stride, fmt, roi)
}

pub(crate) fn decode_image_region_into_with_native_context<'a>(
    image: &Image<'a>,
    native_context: &mut signinum_j2k_native::DecoderContext<'a>,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<(), J2kError> {
    match fmt {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Gray8 => {
            let components = image
                .decode_region_components_with_context((roi.x, roi.y, roi.w, roi.h), native_context)
                .map_err(|err| J2kError::Backend(err.to_string()))?;
            write_components_u8_output(&components, out, stride, fmt)
        }
        PixelFormat::Rgb16 | PixelFormat::Gray16 => {
            let raw = image
                .decode_native_region_with_context((roi.x, roi.y, roi.w, roi.h), native_context)
                .map_err(|err| J2kError::Backend(err.to_string()))?;
            write_u16_output(
                image.color_space(),
                image.has_alpha(),
                &raw,
                out,
                stride,
                fmt,
            )
        }
        PixelFormat::Rgba16 => unreachable!("validated above"),
        _ => Err(Unsupported {
            what: "pixel format is not yet supported by signinum-j2k",
        }
        .into()),
    }
}

fn write_components_u8_output(
    components: &DecodedComponents<'_>,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    let (width, height) = components.dimensions();
    let width = width as usize;
    let height = height as usize;
    let planes = components.planes();
    match (
        components.color_space(),
        components.has_alpha(),
        planes.len(),
        fmt,
    ) {
        (ColorSpace::Gray, false, 1, PixelFormat::Gray8) => {
            write_component_rows_u8(&planes[0], out, stride, width, height);
            Ok(())
        }
        (ColorSpace::RGB, false, 3, PixelFormat::Rgb8)
        | (ColorSpace::RGB, true, 4, PixelFormat::Rgb8) => {
            write_rgb_component_rows_u8(planes, out, stride, width, height);
            Ok(())
        }
        (ColorSpace::RGB, false, 3, PixelFormat::Rgba8) => {
            write_rgba_component_rows_u8(planes, out, stride, width, height, true);
            Ok(())
        }
        (ColorSpace::RGB, true, 4, PixelFormat::Rgba8) => {
            write_rgba_component_rows_u8(planes, out, stride, width, height, false);
            Ok(())
        }
        _ => Err(Unsupported {
            what: "backend color space cannot be mapped to requested 8-bit pixel format",
        }
        .into()),
    }
}

fn write_component_rows_u8(
    plane: &signinum_j2k_native::ComponentPlane<'_>,
    out: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
) {
    for y in 0..height {
        let src = &plane.samples()[y * width..(y + 1) * width];
        let dst = &mut out[y * stride..y * stride + width];
        write_samples_as_u8(src, plane.bit_depth(), dst);
    }
}

fn write_rgb_component_rows_u8(
    planes: &[signinum_j2k_native::ComponentPlane<'_>],
    out: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
) {
    for y in 0..height {
        let row = y * width;
        let dst = &mut out[y * stride..y * stride + width * 3];
        for x in 0..width {
            let dst = &mut dst[x * 3..x * 3 + 3];
            for channel in 0..3 {
                dst[channel] = sample_as_u8(
                    planes[channel].samples()[row + x],
                    planes[channel].bit_depth(),
                );
            }
        }
    }
}

fn write_rgba_component_rows_u8(
    planes: &[signinum_j2k_native::ComponentPlane<'_>],
    out: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
    synthesize_alpha: bool,
) {
    for y in 0..height {
        let row = y * width;
        let dst = &mut out[y * stride..y * stride + width * 4];
        for x in 0..width {
            let dst = &mut dst[x * 4..x * 4 + 4];
            for channel in 0..3 {
                dst[channel] = sample_as_u8(
                    planes[channel].samples()[row + x],
                    planes[channel].bit_depth(),
                );
            }
            dst[3] = if synthesize_alpha {
                u8::MAX
            } else {
                sample_as_u8(planes[3].samples()[row + x], planes[3].bit_depth())
            };
        }
    }
}

fn write_samples_as_u8(src: &[f32], bit_depth: u8, dst: &mut [u8]) {
    for (sample, dst) in src.iter().zip(dst.iter_mut()) {
        *dst = sample_as_u8(*sample, bit_depth);
    }
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

pub(crate) fn validate_supported_format(fmt: PixelFormat) -> Result<(), J2kError> {
    if matches!(fmt, PixelFormat::Rgba16) {
        return Err(Unsupported {
            what: "Rgba16 output is not supported by signinum-j2k M1",
        }
        .into());
    }
    Ok(())
}

pub(crate) fn validate_buffer(
    dims: (u32, u32),
    out_len: usize,
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    validate_strided_output_buffer(dims, out_len, stride, fmt).map_err(Into::into)
}

pub(crate) fn validate_region(roi: Rect, dims: (u32, u32)) -> Result<(), J2kError> {
    if roi.is_within(dims) {
        return Ok(());
    }
    Err(J2kError::InvalidRegion {
        x: roi.x,
        y: roi.y,
        w: roi.w,
        h: roi.h,
        image_w: dims.0,
        image_h: dims.1,
    })
}

fn write_u8_output(
    color_space: &ColorSpace,
    has_alpha: bool,
    dims: (u32, u32),
    decoded: &[u8],
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    let width = dims.0 as usize;
    let height = dims.1 as usize;
    match (color_space, has_alpha, fmt) {
        (ColorSpace::RGB, false, PixelFormat::Rgb8) => {
            copy_rows_exact(decoded, out, stride, width * 3, height);
            Ok(())
        }
        (ColorSpace::RGB, true, PixelFormat::Rgb8) => {
            drop_alpha_u8(decoded, out, stride, width, height);
            Ok(())
        }
        (ColorSpace::RGB, false, PixelFormat::Rgba8) => {
            add_opaque_alpha_u8(decoded, out, stride, width, height);
            Ok(())
        }
        (ColorSpace::RGB, true, PixelFormat::Rgba8) => {
            copy_rows_exact(decoded, out, stride, width * 4, height);
            Ok(())
        }
        (ColorSpace::Gray, false, PixelFormat::Gray8) => {
            copy_rows_exact(decoded, out, stride, width, height);
            Ok(())
        }
        _ => Err(Unsupported {
            what: "backend color space cannot be mapped to requested 8-bit pixel format",
        }
        .into()),
    }
}

fn write_u16_output(
    color_space: &ColorSpace,
    has_alpha: bool,
    raw: &RawBitmap,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    let width = raw.width as usize;
    let height = raw.height as usize;
    match (color_space, has_alpha, raw.num_components, fmt) {
        (ColorSpace::RGB, false, 3, PixelFormat::Rgb16) => {
            convert_or_copy_u16(
                &raw.data,
                raw.bytes_per_sample,
                raw.bit_depth,
                3,
                out,
                stride,
                (width, height),
            );
            Ok(())
        }
        (ColorSpace::Gray, false, 1, PixelFormat::Gray16) => {
            convert_or_copy_u16(
                &raw.data,
                raw.bytes_per_sample,
                raw.bit_depth,
                1,
                out,
                stride,
                (width, height),
            );
            Ok(())
        }
        _ => Err(Unsupported {
            what: "backend color space cannot be mapped to requested 16-bit pixel format",
        }
        .into()),
    }
}

fn copy_rows_exact(src: &[u8], out: &mut [u8], stride: usize, row_bytes: usize, height: usize) {
    for (src_row, dst_row) in src
        .chunks_exact(row_bytes)
        .zip(out.chunks_exact_mut(stride))
        .take(height)
    {
        dst_row[..row_bytes].copy_from_slice(src_row);
    }
}

fn add_opaque_alpha_u8(src: &[u8], out: &mut [u8], stride: usize, width: usize, height: usize) {
    let src_row_bytes = width * 3;
    let dst_row_bytes = width * 4;
    for (src_row, dst_row) in src
        .chunks_exact(src_row_bytes)
        .zip(out.chunks_exact_mut(stride))
        .take(height)
    {
        for (rgb, rgba) in src_row
            .chunks_exact(3)
            .zip(dst_row[..dst_row_bytes].chunks_exact_mut(4))
        {
            rgba[..3].copy_from_slice(rgb);
            rgba[3] = u8::MAX;
        }
    }
}

fn drop_alpha_u8(src: &[u8], out: &mut [u8], stride: usize, width: usize, height: usize) {
    let src_row_bytes = width * 4;
    let dst_row_bytes = width * 3;
    for (src_row, dst_row) in src
        .chunks_exact(src_row_bytes)
        .zip(out.chunks_exact_mut(stride))
        .take(height)
    {
        for (rgba, rgb) in src_row
            .chunks_exact(4)
            .zip(dst_row[..dst_row_bytes].chunks_exact_mut(3))
        {
            rgb.copy_from_slice(&rgba[..3]);
        }
    }
}

fn convert_or_copy_u16(
    src: &[u8],
    bytes_per_sample: u8,
    bit_depth: u8,
    channels: usize,
    out: &mut [u8],
    stride: usize,
    dims: (usize, usize),
) {
    let (width, height) = dims;
    let dst_row_bytes = width * channels * 2;
    let src_row_bytes = width * channels * usize::from(bytes_per_sample);
    let max_value = ((1_u32 << bit_depth.min(16)) - 1).max(1);
    for (src_row, dst_row) in src
        .chunks_exact(src_row_bytes)
        .zip(out.chunks_exact_mut(stride))
        .take(height)
    {
        let dst_row = &mut dst_row[..dst_row_bytes];
        if bytes_per_sample == 2 {
            dst_row.copy_from_slice(src_row);
            continue;
        }
        for (sample, dst_sample) in src_row.iter().zip(dst_row.chunks_exact_mut(2)) {
            let widened = (u32::from(*sample) * u32::from(u16::MAX) + (max_value / 2)) / max_value;
            dst_sample.copy_from_slice(&(widened as u16).to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{can_decode_u8_directly, ColorSpace, PixelFormat};

    #[test]
    fn direct_u8_decode_accepts_exact_rgb_and_gray_layouts() {
        assert!(can_decode_u8_directly(
            &ColorSpace::RGB,
            false,
            (128, 64),
            128 * 3,
            PixelFormat::Rgb8
        ));
        assert!(can_decode_u8_directly(
            &ColorSpace::Gray,
            false,
            (128, 64),
            128,
            PixelFormat::Gray8
        ));
    }

    #[test]
    fn direct_u8_decode_rejects_format_conversion_and_padded_stride() {
        assert!(!can_decode_u8_directly(
            &ColorSpace::RGB,
            false,
            (128, 64),
            128 * 4,
            PixelFormat::Rgba8
        ));
        assert!(!can_decode_u8_directly(
            &ColorSpace::RGB,
            true,
            (128, 64),
            128 * 3,
            PixelFormat::Rgb8
        ));
        assert!(!can_decode_u8_directly(
            &ColorSpace::Gray,
            false,
            (128, 64),
            160,
            PixelFormat::Gray8
        ));
    }
}
