// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use std::{cell::RefCell, mem::size_of};

#[cfg(target_os = "macos")]
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use slidecodec_core::{PixelFormat, Rect};
use slidecodec_j2k_native::{
    ColorSpace as NativeColorSpace, DecodeSettings as NativeDecodeSettings,
    DecodedComponents as NativeDecodedComponents, DecoderContext as NativeDecoderContext,
    Image as NativeImage,
};

use crate::{Error, Surface};

#[cfg(target_os = "macos")]
const SHADER_SOURCE: &str = r"
#include <metal_stdlib>
using namespace metal;

struct J2kPackParams {
    uint width;
    uint height;
    uint out_stride;
    uint plane_count;
    uint output_channels;
    uint opaque_alpha;
    uint bit_depths[4];
};

inline uint max_value_for_bit_depth(uint bit_depth) {
    const uint clamped = min(bit_depth, 16u);
    const uint max_value = (1u << clamped) - 1u;
    return max(max_value, 1u);
}

inline uchar scale_to_u8(float sample, uint bit_depth) {
    const float max_value = float(max_value_for_bit_depth(bit_depth));
    const float clamped = clamp(sample, 0.0f, max_value);
    return uchar(min(floor((clamped * 255.0f) / max_value + 0.5f), 255.0f));
}

inline ushort pack_to_u16(float sample, uint bit_depth) {
    const float max_value = float(max_value_for_bit_depth(bit_depth));
    const float clamped = clamp(sample, 0.0f, max_value);
    if (bit_depth <= 8u) {
        return ushort(min(floor((clamped * 65535.0f) / max_value + 0.5f), 65535.0f));
    }
    return ushort(min(floor(clamped + 0.5f), 65535.0f));
}

kernel void j2k_pack_u8(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device const float *plane3 [[buffer(3)]],
    device uchar *out [[buffer(4)]],
    constant J2kPackParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    uint out_idx = gid.y * params.out_stride + gid.x * params.output_channels;

    if (params.output_channels == 1) {
        out[out_idx] = scale_to_u8(plane0[idx], params.bit_depths[0]);
        return;
    }

    out[out_idx] = scale_to_u8(plane0[idx], params.bit_depths[0]);
    out[out_idx + 1] = scale_to_u8(plane1[idx], params.bit_depths[1]);
    out[out_idx + 2] = scale_to_u8(plane2[idx], params.bit_depths[2]);

    if (params.output_channels == 4) {
        out[out_idx + 3] = params.opaque_alpha != 0
            ? uchar(255)
            : scale_to_u8(plane3[idx], params.bit_depths[3]);
    }
}

kernel void j2k_pack_u16(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device const float *plane3 [[buffer(3)]],
    device ushort *out [[buffer(4)]],
    constant J2kPackParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    uint out_idx = (gid.y * params.out_stride) / 2u + gid.x * params.output_channels;

    if (params.output_channels == 1) {
        out[out_idx] = pack_to_u16(plane0[idx], params.bit_depths[0]);
        return;
    }

    out[out_idx] = pack_to_u16(plane0[idx], params.bit_depths[0]);
    out[out_idx + 1] = pack_to_u16(plane1[idx], params.bit_depths[1]);
    out[out_idx + 2] = pack_to_u16(plane2[idx], params.bit_depths[2]);
}
";

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kPackParams {
    width: u32,
    height: u32,
    out_stride: u32,
    plane_count: u32,
    output_channels: u32,
    opaque_alpha: u32,
    bit_depths: [u32; 4],
}

#[cfg(target_os = "macos")]
thread_local! {
    static METAL_RUNTIME: RefCell<Option<Result<MetalRuntime, String>>> = const { RefCell::new(None) };
}

#[cfg(target_os = "macos")]
struct MetalRuntime {
    device: Device,
    queue: CommandQueue,
    pack_u8: ComputePipelineState,
    pack_u16: ComputePipelineState,
}

#[cfg(target_os = "macos")]
impl MetalRuntime {
    fn new() -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or_else(|| "Metal is unavailable on this host".to_string())?;
        let options = CompileOptions::new();
        let library = device.new_library_with_source(SHADER_SOURCE, &options)?;
        let pack_u8_fn = library.get_function("j2k_pack_u8", None)?;
        let pack_u16_fn = library.get_function("j2k_pack_u16", None)?;
        let pack_u8 = device.new_compute_pipeline_state_with_function(&pack_u8_fn)?;
        let pack_u16 = device.new_compute_pipeline_state_with_function(&pack_u16_fn)?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pack_u8,
            pack_u16,
        })
    }
}

#[cfg(target_os = "macos")]
fn with_runtime<R>(f: impl FnOnce(&MetalRuntime) -> Result<R, Error>) -> Result<R, Error> {
    METAL_RUNTIME.with(|runtime| {
        let mut runtime = runtime.borrow_mut();
        if runtime.is_none() {
            *runtime = Some(MetalRuntime::new());
        }
        match runtime.as_ref().expect("runtime initialized") {
            Ok(runtime) => f(runtime),
            Err(message) => Err(Error::MetalKernel {
                message: message.clone(),
            }),
        }
    })
}

#[cfg(target_os = "macos")]
struct PlaneStage {
    dims: (u32, u32),
    plane_count: usize,
    color_space: NativeColorSpace,
    has_alpha: bool,
    bit_depths: [u32; 4],
    planes: [Option<Buffer>; 4],
}

#[cfg(target_os = "macos")]
impl PlaneStage {
    fn from_planes(
        device: &Device,
        decoded: &NativeDecodedComponents<'_>,
        roi: Option<Rect>,
    ) -> Result<Self, Error> {
        let full_dims = decoded.dimensions();
        let roi = roi.unwrap_or(Rect {
            x: 0,
            y: 0,
            w: full_dims.0,
            h: full_dims.1,
        });
        let dims = (roi.w, roi.h);
        let plane_count = decoded.planes().len();
        if plane_count == 0 || plane_count > 4 {
            return Err(Error::MetalKernel {
                message: format!("unsupported J2K plane count {plane_count}"),
            });
        }

        let mut bit_depths = [0u32; 4];
        let mut planes: [Option<Buffer>; 4] = [None, None, None, None];
        for (index, plane) in decoded.planes().iter().enumerate() {
            bit_depths[index] = u32::from(plane.bit_depth());
            let len = dims.0 as usize * dims.1 as usize;
            let buffer = device.new_buffer(
                (len * size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            copy_plane_samples(&buffer, plane.samples(), full_dims.0 as usize, roi);
            planes[index] = Some(buffer);
        }

        Ok(Self {
            dims,
            plane_count,
            color_space: decoded.color_space().clone(),
            has_alpha: decoded.has_alpha(),
            bit_depths,
            planes,
        })
    }

    fn finish_with_runtime(
        self,
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> Result<Surface, Error> {
        let pitch_bytes = self.dims.0 as usize * fmt.bytes_per_pixel();
        let out_buffer = runtime.device.new_buffer(
            (pitch_bytes * self.dims.1 as usize) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let (output_channels, opaque_alpha, pipeline) = output_shape_for(
            &self.color_space,
            self.has_alpha,
            self.plane_count,
            fmt,
            runtime,
        )?;

        let params = J2kPackParams {
            width: self.dims.0,
            height: self.dims.1,
            out_stride: u32::try_from(pitch_bytes).expect("J2K Metal output stride fits in u32"),
            plane_count: u32::try_from(self.plane_count).expect("J2K plane count fits in u32"),
            output_channels,
            opaque_alpha,
            bit_depths: self.bit_depths,
        };

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(
            0,
            self.planes[0].as_ref().map(std::convert::AsRef::as_ref),
            0,
        );
        encoder.set_buffer(
            1,
            self.planes[1].as_ref().map(std::convert::AsRef::as_ref),
            0,
        );
        encoder.set_buffer(
            2,
            self.planes[2].as_ref().map(std::convert::AsRef::as_ref),
            0,
        );
        encoder.set_buffer(
            3,
            self.planes[3].as_ref().map(std::convert::AsRef::as_ref),
            0,
        );
        encoder.set_buffer(4, Some(&out_buffer), 0);
        encoder.set_bytes(
            5,
            size_of::<J2kPackParams>() as u64,
            (&raw const params).cast(),
        );

        let width = pipeline.thread_execution_width().max(1);
        let max_threads = pipeline.max_total_threads_per_threadgroup().max(width);
        let height = (max_threads / width).max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(self.dims.0),
                height: u64::from(self.dims.1),
                depth: 1,
            },
            MTLSize {
                width,
                height,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(Surface::from_metal_buffer(out_buffer, self.dims, fmt))
    }
}

#[cfg(target_os = "macos")]
fn copy_plane_samples(buffer: &Buffer, samples: &[f32], image_width: usize, roi: Rect) {
    let row_width = roi.w as usize;
    let dst = unsafe {
        core::slice::from_raw_parts_mut(buffer.contents().cast::<f32>(), row_width * roi.h as usize)
    };

    for row in 0..roi.h as usize {
        let src_start = (roi.y as usize + row) * image_width + roi.x as usize;
        let src_end = src_start + row_width;
        let dst_start = row * row_width;
        dst[dst_start..dst_start + row_width].copy_from_slice(&samples[src_start..src_end]);
    }
}

#[cfg(target_os = "macos")]
fn output_shape_for<'a>(
    color_space: &NativeColorSpace,
    has_alpha: bool,
    plane_count: usize,
    fmt: PixelFormat,
    runtime: &'a MetalRuntime,
) -> Result<(u32, u32, &'a ComputePipelineState), Error> {
    match (color_space, has_alpha, plane_count, fmt) {
        (NativeColorSpace::Gray, false, 1, PixelFormat::Gray8) => Ok((1, 0, &runtime.pack_u8)),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgb8)
        | (NativeColorSpace::RGB, true, 4, PixelFormat::Rgb8) => Ok((3, 0, &runtime.pack_u8)),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgba8) => Ok((4, 1, &runtime.pack_u8)),
        (NativeColorSpace::RGB, true, 4, PixelFormat::Rgba8) => Ok((4, 0, &runtime.pack_u8)),
        (NativeColorSpace::Gray, false, 1, PixelFormat::Gray16) => Ok((1, 0, &runtime.pack_u16)),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgb16) => Ok((3, 0, &runtime.pack_u16)),
        _ => Err(Error::MetalKernel {
            message: format!(
                "unsupported J2K Metal mapping for {color_space:?}, alpha={has_alpha}, planes={plane_count}, fmt={fmt:?}"
            ),
        }),
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::{output_shape_for, MetalRuntime};
    use slidecodec_core::PixelFormat;
    use slidecodec_j2k_native::ColorSpace as NativeColorSpace;

    #[test]
    fn rgb16_with_alpha_is_rejected() {
        let runtime = MetalRuntime::new().expect("Metal runtime");
        let result = output_shape_for(
            &NativeColorSpace::RGB,
            true,
            4,
            PixelFormat::Rgb16,
            &runtime,
        );
        assert!(result.is_err(), "RGBA input must not silently map to Rgb16");
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_image_to_surface<'a>(
    image: &NativeImage<'a>,
    context: &mut NativeDecoderContext<'a>,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let decoded = image
            .decode_components_with_context(context)
            .map_err(|error| Error::Decode(slidecodec_j2k::J2kError::Backend(error.to_string())))?;
        let stage = PlaneStage::from_planes(&runtime.device, &decoded, None)?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_to_surface<'a>(
    image: &NativeImage<'a>,
    context: &mut NativeDecoderContext<'a>,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let decoded = image
            .decode_components_with_context(context)
            .map_err(|error| Error::Decode(slidecodec_j2k::J2kError::Backend(error.to_string())))?;
        let stage = PlaneStage::from_planes(&runtime.device, &decoded, Some(roi))?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_scaled_to_surface(
    bytes: &[u8],
    dims: (u32, u32),
    fmt: PixelFormat,
    scale: slidecodec_core::Downscale,
) -> Result<Surface, Error> {
    let target_dims = (
        dims.0.div_ceil(scale.denominator()),
        dims.1.div_ceil(scale.denominator()),
    );
    let settings = NativeDecodeSettings {
        target_resolution: Some(target_dims),
        ..NativeDecodeSettings::default()
    };
    let image = NativeImage::new(bytes, &settings)
        .map_err(|error| Error::Decode(slidecodec_j2k::J2kError::Backend(error.to_string())))?;
    let mut context = NativeDecoderContext::default();
    decode_image_to_surface(&image, &mut context, fmt)
}
