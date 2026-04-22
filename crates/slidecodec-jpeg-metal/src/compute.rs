// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use std::{cell::RefCell, mem::size_of};

#[cfg(target_os = "macos")]
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use slidecodec_core::{PixelFormat, Rect};
use slidecodec_jpeg::{ColorSpace as JpegColorSpace, ComponentRowWriter, Decoder as CpuDecoder};

use crate::{Error, Surface};

#[cfg(target_os = "macos")]
const SHADER_SOURCE: &str = r"
#include <metal_stdlib>
using namespace metal;

struct JpegPackParams {
    uint width;
    uint height;
    uint out_stride;
    uint alpha;
    uint mode;
    uint out_format;
};

constant uint MODE_GRAY = 0;
constant uint MODE_YCBCR = 1;
constant uint MODE_RGB = 2;

constant uint OUT_GRAY = 0;
constant uint OUT_RGB = 1;
constant uint OUT_RGBA = 2;

inline uchar clamp_u8(int value) {
    return uchar(clamp(value, 0, 255));
}

kernel void jpeg_pack(
    device const uchar *plane0 [[buffer(0)]],
    device const uchar *plane1 [[buffer(1)]],
    device const uchar *plane2 [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegPackParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    uint out_idx = gid.y * params.out_stride;

    if (params.out_format == OUT_GRAY) {
        out_idx += gid.x;
        if (params.mode == MODE_GRAY || params.mode == MODE_YCBCR) {
            out[out_idx] = plane0[idx];
            return;
        }

        const uint r = plane0[idx];
        const uint g = plane1[idx];
        const uint b = plane2[idx];
        out[out_idx] = uchar((77u * r + 150u * g + 29u * b + 128u) >> 8);
        return;
    }

    out_idx += gid.x * (params.out_format == OUT_RGB ? 3u : 4u);

    if (params.mode == MODE_GRAY) {
        const uchar gray = plane0[idx];
        out[out_idx] = gray;
        out[out_idx + 1] = gray;
        out[out_idx + 2] = gray;
    } else if (params.mode == MODE_RGB) {
        out[out_idx] = plane0[idx];
        out[out_idx + 1] = plane1[idx];
        out[out_idx + 2] = plane2[idx];
    } else {
        const int y = int(plane0[idx]);
        const int cb = int(plane1[idx]) - 128;
        const int cr = int(plane2[idx]) - 128;
        out[out_idx] = clamp_u8(y + ((91881 * cr + (1 << 15)) >> 16));
        out[out_idx + 1] = clamp_u8(y - ((22554 * cb + 46802 * cr + (1 << 15)) >> 16));
        out[out_idx + 2] = clamp_u8(y + ((116130 * cb + (1 << 15)) >> 16));
    }

    if (params.out_format == OUT_RGBA) {
        out[out_idx + 3] = uchar(params.alpha);
    }
}

struct JpegViewportParams {
    uint tile_width;
    uint tile_height;
    uint viewport_width;
    uint viewport_height;
    uint viewport_stride;
    uint dest_x;
    uint dest_y;
    uint mode;
};

kernel void jpeg_pack_into_viewport_rgb(
    device const uchar *plane0 [[buffer(0)]],
    device const uchar *plane1 [[buffer(1)]],
    device const uchar *plane2 [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegViewportParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.tile_width || gid.y >= params.tile_height) {
        return;
    }

    const uint dst_x = params.dest_x + gid.x;
    const uint dst_y = params.dest_y + gid.y;
    if (dst_x >= params.viewport_width || dst_y >= params.viewport_height) {
        return;
    }

    const uint idx = gid.y * params.tile_width + gid.x;
    const uint out_idx = dst_y * params.viewport_stride + dst_x * 3u;

    if (params.mode == MODE_GRAY) {
        const uchar gray = plane0[idx];
        out[out_idx] = gray;
        out[out_idx + 1] = gray;
        out[out_idx + 2] = gray;
        return;
    }

    if (params.mode == MODE_RGB) {
        out[out_idx] = plane0[idx];
        out[out_idx + 1] = plane1[idx];
        out[out_idx + 2] = plane2[idx];
        return;
    }

    const int y = int(plane0[idx]);
    const int cb = int(plane1[idx]) - 128;
    const int cr = int(plane2[idx]) - 128;
    out[out_idx] = clamp_u8(y + ((91881 * cr + (1 << 15)) >> 16));
    out[out_idx + 1] = clamp_u8(y - ((22554 * cb + 46802 * cr + (1 << 15)) >> 16));
    out[out_idx + 2] = clamp_u8(y + ((116130 * cb + (1 << 15)) >> 16));
}
";

#[cfg(target_os = "macos")]
const MODE_GRAY: u32 = 0;
#[cfg(target_os = "macos")]
const MODE_YCBCR: u32 = 1;
#[cfg(target_os = "macos")]
const MODE_RGB: u32 = 2;

#[cfg(target_os = "macos")]
const OUT_GRAY: u32 = 0;
#[cfg(target_os = "macos")]
const OUT_RGB: u32 = 1;
#[cfg(target_os = "macos")]
const OUT_RGBA: u32 = 2;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct JpegPackParams {
    width: u32,
    height: u32,
    out_stride: u32,
    alpha: u32,
    mode: u32,
    out_format: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct JpegViewportParams {
    tile_width: u32,
    tile_height: u32,
    viewport_width: u32,
    viewport_height: u32,
    viewport_stride: u32,
    dest_x: u32,
    dest_y: u32,
    mode: u32,
}

#[cfg(target_os = "macos")]
thread_local! {
    static METAL_RUNTIME: RefCell<Option<Result<MetalRuntime, String>>> = const { RefCell::new(None) };
}

#[cfg(target_os = "macos")]
struct MetalRuntime {
    device: Device,
    queue: CommandQueue,
    pack_pipeline: ComputePipelineState,
    viewport_rgb_pipeline: ComputePipelineState,
}

#[cfg(target_os = "macos")]
impl MetalRuntime {
    fn new() -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or_else(|| "Metal is unavailable on this host".to_string())?;
        let options = CompileOptions::new();
        let library = device.new_library_with_source(SHADER_SOURCE, &options)?;
        let pack_function = library.get_function("jpeg_pack", None)?;
        let pack_pipeline = device.new_compute_pipeline_state_with_function(&pack_function)?;
        let viewport_function = library.get_function("jpeg_pack_into_viewport_rgb", None)?;
        let viewport_rgb_pipeline =
            device.new_compute_pipeline_state_with_function(&viewport_function)?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pack_pipeline,
            viewport_rgb_pipeline,
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
#[derive(Clone, Copy)]
enum PlaneMode {
    Gray,
    YCbCr,
    Rgb,
}

#[cfg(target_os = "macos")]
struct PlaneStage {
    dims: (u32, u32),
    mode: PlaneMode,
    plane0: Buffer,
    plane1: Option<Buffer>,
    plane2: Option<Buffer>,
}

#[cfg(target_os = "macos")]
pub(crate) struct ViewportTileStage {
    dest: Rect,
    stage: PlaneStage,
}

#[cfg(target_os = "macos")]
impl PlaneStage {
    fn new(device: &Device, color_space: JpegColorSpace, dims: (u32, u32)) -> Result<Self, Error> {
        let len = dims.0 as usize * dims.1 as usize;
        let plane0 = device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared);
        let (mode, plane1, plane2) = match color_space {
            JpegColorSpace::Grayscale => (PlaneMode::Gray, None, None),
            JpegColorSpace::YCbCr => (
                PlaneMode::YCbCr,
                Some(device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)),
                Some(device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)),
            ),
            JpegColorSpace::Rgb => (
                PlaneMode::Rgb,
                Some(device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)),
                Some(device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)),
            ),
            JpegColorSpace::Cmyk | JpegColorSpace::Ycck => {
                return Err(Error::MetalKernel {
                    message: "Metal compute path does not support CMYK/YCCK JPEG output"
                        .to_string(),
                })
            }
        };

        Ok(Self {
            dims,
            mode,
            plane0,
            plane1,
            plane2,
        })
    }

    fn finish_with_runtime(
        self,
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> Result<Surface, Error> {
        match (self.mode, fmt) {
            (PlaneMode::Gray | PlaneMode::YCbCr, PixelFormat::Gray8) => {
                Ok(Surface::from_metal_buffer(self.plane0, self.dims, fmt))
            }
            (
                PlaneMode::Gray | PlaneMode::YCbCr | PlaneMode::Rgb,
                PixelFormat::Rgb8 | PixelFormat::Rgba8,
            )
            | (PlaneMode::Rgb, PixelFormat::Gray8) => Ok(self.dispatch_with_runtime(runtime, fmt)),
            _ => Err(Error::MetalKernel {
                message: format!("unsupported JPEG Metal pixel format {fmt:?}"),
            }),
        }
    }

    fn dispatch_with_runtime(self, runtime: &MetalRuntime, fmt: PixelFormat) -> Surface {
        let pitch_bytes = self.dims.0 as usize * fmt.bytes_per_pixel();
        let out_buffer = runtime.device.new_buffer(
            (pitch_bytes * self.dims.1 as usize) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params = JpegPackParams {
            width: self.dims.0,
            height: self.dims.1,
            out_stride: u32::try_from(pitch_bytes).expect("JPEG Metal output stride fits in u32"),
            alpha: u32::from(u8::MAX),
            mode: match self.mode {
                PlaneMode::Gray => MODE_GRAY,
                PlaneMode::YCbCr => MODE_YCBCR,
                PlaneMode::Rgb => MODE_RGB,
            },
            out_format: match fmt {
                PixelFormat::Gray8 => OUT_GRAY,
                PixelFormat::Rgb8 => OUT_RGB,
                PixelFormat::Rgba8 => OUT_RGBA,
                _ => unreachable!("validated by finish"),
            },
        };

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.pack_pipeline);
        encoder.set_buffer(0, Some(&self.plane0), 0);
        encoder.set_buffer(1, self.plane1.as_ref().map(std::convert::AsRef::as_ref), 0);
        encoder.set_buffer(2, self.plane2.as_ref().map(std::convert::AsRef::as_ref), 0);
        encoder.set_buffer(3, Some(&out_buffer), 0);
        encoder.set_bytes(
            4,
            size_of::<JpegPackParams>() as u64,
            (&raw const params).cast(),
        );

        let width = runtime.pack_pipeline.thread_execution_width().max(1);
        let max_threads = runtime
            .pack_pipeline
            .max_total_threads_per_threadgroup()
            .max(width);
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

        Surface::from_metal_buffer(out_buffer, self.dims, fmt)
    }
}

#[cfg(target_os = "macos")]
impl ComponentRowWriter for PlaneStage {
    fn write_gray_row(
        &mut self,
        y: u32,
        gray_row: &[u8],
    ) -> Result<(), slidecodec_jpeg::JpegError> {
        let width = self.dims.0 as usize;
        write_row_u8(&self.plane0, y, width, gray_row);
        Ok(())
    }

    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        chroma_blue_row: &[u8],
        chroma_red_row: &[u8],
    ) -> Result<(), slidecodec_jpeg::JpegError> {
        let width = self.dims.0 as usize;
        write_row_u8(&self.plane0, y, width, y_row);
        write_row_u8(
            self.plane1.as_ref().expect("Cb plane"),
            y,
            width,
            chroma_blue_row,
        );
        write_row_u8(
            self.plane2.as_ref().expect("Cr plane"),
            y,
            width,
            chroma_red_row,
        );
        Ok(())
    }

    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), slidecodec_jpeg::JpegError> {
        let width = self.dims.0 as usize;
        write_row_u8(&self.plane0, y, width, r_row);
        write_row_u8(self.plane1.as_ref().expect("G plane"), y, width, g_row);
        write_row_u8(self.plane2.as_ref().expect("B plane"), y, width, b_row);
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn write_row_u8(buffer: &Buffer, y: u32, width: usize, src: &[u8]) {
    let row_start = y as usize * width;
    let row_end = row_start + width;
    let len = width * (y as usize + 1);
    let dst = unsafe {
        core::slice::from_raw_parts_mut(buffer.contents().cast::<u8>(), len.max(row_end))
    };
    dst[row_start..row_end].copy_from_slice(&src[..width]);
}

#[cfg(target_os = "macos")]
fn scaled_rect_covering(rect: Rect, scale: slidecodec_core::Downscale) -> Rect {
    let denom = scale.denominator();
    let x_end = rect.x + rect.w;
    let y_end = rect.y + rect.h;
    let x0 = rect.x / denom;
    let y0 = rect.y / denom;
    let x1 = x_end.div_ceil(denom);
    let y1 = y_end.div_ceil(denom);
    Rect {
        x: x0,
        y: y0,
        w: x1.saturating_sub(x0),
        h: y1.saturating_sub(y0),
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut slidecodec_jpeg::ScratchPool,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let mut stage = PlaneStage::new(
            &runtime.device,
            decoder.info().color_space,
            decoder.info().dimensions,
        )?;
        decoder.decode_component_rows_with_scratch(pool, &mut stage)?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut slidecodec_jpeg::ScratchPool,
    fmt: PixelFormat,
    roi: slidecodec_jpeg::Rect,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let dims = (roi.w, roi.h);
        let mut stage = PlaneStage::new(&runtime.device, decoder.info().color_space, dims)?;
        decoder.decode_region_component_rows_with_scratch(
            pool,
            &mut stage,
            roi,
            slidecodec_core::Downscale::None,
        )?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_scaled_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut slidecodec_jpeg::ScratchPool,
    fmt: PixelFormat,
    scale: slidecodec_core::Downscale,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let full = decoder.info().dimensions;
        let roi = slidecodec_jpeg::Rect {
            x: 0,
            y: 0,
            w: full.0,
            h: full.1,
        };
        let scaled = scaled_rect_covering(
            Rect {
                x: 0,
                y: 0,
                w: full.0,
                h: full.1,
            },
            scale,
        );
        let mut stage = PlaneStage::new(
            &runtime.device,
            decoder.info().color_space,
            (scaled.w, scaled.h),
        )?;
        decoder.decode_region_component_rows_with_scratch(pool, &mut stage, roi, scale)?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_scaled_to_viewport_stage(
    decoder: &CpuDecoder<'_>,
    pool: &mut slidecodec_jpeg::ScratchPool,
    roi: slidecodec_jpeg::Rect,
    scale: slidecodec_core::Downscale,
    dest: Rect,
) -> Result<ViewportTileStage, Error> {
    with_runtime(|runtime| {
        let dims = (
            roi.w.div_ceil(scale.denominator()),
            roi.h.div_ceil(scale.denominator()),
        );
        if dims != (dest.w, dest.h) {
            return Err(Error::MetalKernel {
                message: format!(
                    "viewport tile dims {dims:?} do not match destination rect {dest:?}"
                ),
            });
        }
        let mut stage = PlaneStage::new(&runtime.device, decoder.info().color_space, dims)?;
        decoder.decode_region_component_rows_with_scratch(pool, &mut stage, roi, scale)?;
        Ok(ViewportTileStage { dest, stage })
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn compose_rgb_viewport(
    stages: &[ViewportTileStage],
    viewport_dims: (u32, u32),
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let pitch_bytes = viewport_dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
        let out_buffer = runtime.device.new_buffer(
            (pitch_bytes * viewport_dims.1 as usize) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        for tile in stages {
            let params = JpegViewportParams {
                tile_width: tile.stage.dims.0,
                tile_height: tile.stage.dims.1,
                viewport_width: viewport_dims.0,
                viewport_height: viewport_dims.1,
                viewport_stride: u32::try_from(pitch_bytes).expect("viewport stride fits in u32"),
                dest_x: tile.dest.x,
                dest_y: tile.dest.y,
                mode: match tile.stage.mode {
                    PlaneMode::Gray => MODE_GRAY,
                    PlaneMode::YCbCr => MODE_YCBCR,
                    PlaneMode::Rgb => MODE_RGB,
                },
            };

            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&runtime.viewport_rgb_pipeline);
            encoder.set_buffer(0, Some(&tile.stage.plane0), 0);
            encoder.set_buffer(
                1,
                tile.stage.plane1.as_ref().map(std::convert::AsRef::as_ref),
                0,
            );
            encoder.set_buffer(
                2,
                tile.stage.plane2.as_ref().map(std::convert::AsRef::as_ref),
                0,
            );
            encoder.set_buffer(3, Some(&out_buffer), 0);
            encoder.set_bytes(
                4,
                size_of::<JpegViewportParams>() as u64,
                (&raw const params).cast(),
            );

            let width = runtime
                .viewport_rgb_pipeline
                .thread_execution_width()
                .max(1);
            let max_threads = runtime
                .viewport_rgb_pipeline
                .max_total_threads_per_threadgroup()
                .max(width);
            let height = (max_threads / width).max(1);
            encoder.dispatch_threads(
                MTLSize {
                    width: u64::from(tile.stage.dims.0),
                    height: u64::from(tile.stage.dims.1),
                    depth: 1,
                },
                MTLSize {
                    width,
                    height,
                    depth: 1,
                },
            );
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(Surface::from_metal_buffer(
            out_buffer,
            viewport_dims,
            PixelFormat::Rgb8,
        ))
    })
}
