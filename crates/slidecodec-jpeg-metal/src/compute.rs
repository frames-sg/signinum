// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use std::{
    cell::RefCell,
    mem::{size_of, size_of_val},
};

#[cfg(target_os = "macos")]
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use slidecodec_core::{PixelFormat, Rect};
use slidecodec_jpeg::{
    ColorSpace as JpegColorSpace, ComponentRowWriter, Decoder as CpuDecoder,
    __private::{
        JpegMetalFast420PacketV1, JpegMetalFast444PacketV1, MetalHuffmanTable as PacketHuffmanTable,
    },
};

use crate::viewport::ViewportTile;
use crate::{Error, Surface};

#[cfg(target_os = "macos")]
const SHADER_SOURCE: &str = include_str!("shaders.metal");

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
const FAST420_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
const FAST420_STATUS_TRUNCATED: u32 = 1;
#[cfg(target_os = "macos")]
const FAST420_STATUS_HUFFMAN: u32 = 2;

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
struct JpegFast420Params {
    width: u32,
    height: u32,
    chroma_width: u32,
    chroma_height: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    restart_interval_mcus: u32,
    restart_offset_count: u32,
    entropy_len: u32,
    out_stride: u32,
    alpha: u32,
    out_format: u32,
    origin_x: u32,
    origin_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct JpegFast420ScaledParams {
    scaled_width: u32,
    scaled_height: u32,
    chroma_width: u32,
    chroma_height: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    restart_interval_mcus: u32,
    restart_offset_count: u32,
    entropy_len: u32,
    scale_shift: u32,
    origin_x: u32,
    origin_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct JpegFast444Params {
    width: u32,
    height: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    restart_interval_mcus: u32,
    restart_offset_count: u32,
    entropy_len: u32,
    origin_x: u32,
    origin_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct JpegFast444ScaledParams {
    scaled_width: u32,
    scaled_height: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    restart_interval_mcus: u32,
    restart_offset_count: u32,
    entropy_len: u32,
    scale_shift: u32,
    origin_x: u32,
    origin_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct JpegFast420WindowedPackParams {
    src_width: u32,
    src_height: u32,
    chroma_width: u32,
    chroma_height: u32,
    src_x: u32,
    src_y: u32,
    width: u32,
    height: u32,
    out_stride: u32,
    alpha: u32,
    out_format: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct MetalHuffmanTableHost {
    bits: [u8; 16],
    values_len: u16,
    reserved: u16,
    values: [u8; 256],
}

#[cfg(target_os = "macos")]
impl From<&PacketHuffmanTable> for MetalHuffmanTableHost {
    fn from(value: &PacketHuffmanTable) -> Self {
        Self {
            bits: value.bits,
            values_len: value.values_len,
            reserved: 0,
            values: value.values,
        }
    }
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct JpegDecodeStatus {
    code: u32,
    detail: u32,
    position: u32,
    reserved: u32,
}

#[cfg(target_os = "macos")]
thread_local! {
    static METAL_RUNTIME: RefCell<Option<Result<MetalRuntime, String>>> = const { RefCell::new(None) };
    static VIEWPORT_PLANE_CACHE: RefCell<Option<CachedViewportPlanes>> = const { RefCell::new(None) };
}

#[cfg(target_os = "macos")]
struct MetalRuntime {
    device: Device,
    queue: CommandQueue,
    pack_pipeline: ComputePipelineState,
    pack_420_pipeline: ComputePipelineState,
    pack_420_windowed_pipeline: ComputePipelineState,
    fast420_decode_pipeline: ComputePipelineState,
    fast420_region_decode_pipeline: ComputePipelineState,
    fast420_scaled_decode_pipeline: ComputePipelineState,
    fast420_scaled_region_decode_pipeline: ComputePipelineState,
    fast444_decode_pipeline: ComputePipelineState,
    fast444_region_decode_pipeline: ComputePipelineState,
    fast444_scaled_decode_pipeline: ComputePipelineState,
    fast444_scaled_region_decode_pipeline: ComputePipelineState,
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
        let pack_420_function = library.get_function("jpeg_pack_420", None)?;
        let pack_420_pipeline =
            device.new_compute_pipeline_state_with_function(&pack_420_function)?;
        let pack_420_windowed_function = library.get_function("jpeg_pack_420_windowed", None)?;
        let pack_420_windowed_pipeline =
            device.new_compute_pipeline_state_with_function(&pack_420_windowed_function)?;
        let fast420_decode_function = library.get_function("jpeg_decode_fast420", None)?;
        let fast420_decode_pipeline =
            device.new_compute_pipeline_state_with_function(&fast420_decode_function)?;
        let fast420_region_decode_function =
            library.get_function("jpeg_decode_fast420_region", None)?;
        let fast420_region_decode_pipeline =
            device.new_compute_pipeline_state_with_function(&fast420_region_decode_function)?;
        let fast420_scaled_decode_function =
            library.get_function("jpeg_decode_fast420_scaled", None)?;
        let fast420_scaled_decode_pipeline =
            device.new_compute_pipeline_state_with_function(&fast420_scaled_decode_function)?;
        let fast420_scaled_region_decode_function =
            library.get_function("jpeg_decode_fast420_scaled_region", None)?;
        let fast420_scaled_region_decode_pipeline = device
            .new_compute_pipeline_state_with_function(&fast420_scaled_region_decode_function)?;
        let fast444_decode_function = library.get_function("jpeg_decode_fast444", None)?;
        let fast444_decode_pipeline =
            device.new_compute_pipeline_state_with_function(&fast444_decode_function)?;
        let fast444_region_decode_function =
            library.get_function("jpeg_decode_fast444_region", None)?;
        let fast444_region_decode_pipeline =
            device.new_compute_pipeline_state_with_function(&fast444_region_decode_function)?;
        let fast444_scaled_decode_function =
            library.get_function("jpeg_decode_fast444_scaled", None)?;
        let fast444_scaled_decode_pipeline =
            device.new_compute_pipeline_state_with_function(&fast444_scaled_decode_function)?;
        let fast444_scaled_region_decode_function =
            library.get_function("jpeg_decode_fast444_scaled_region", None)?;
        let fast444_scaled_region_decode_pipeline = device
            .new_compute_pipeline_state_with_function(&fast444_scaled_region_decode_function)?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pack_pipeline,
            pack_420_pipeline,
            pack_420_windowed_pipeline,
            fast420_decode_pipeline,
            fast420_region_decode_pipeline,
            fast420_scaled_decode_pipeline,
            fast420_scaled_region_decode_pipeline,
            fast444_decode_pipeline,
            fast444_region_decode_pipeline,
            fast444_scaled_decode_pipeline,
            fast444_scaled_region_decode_pipeline,
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
#[derive(Clone, Copy, PartialEq, Eq)]
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
struct ViewportPlaneWriter<'a> {
    stage: &'a mut PlaneStage,
    dest: Rect,
}

#[cfg(target_os = "macos")]
struct CachedViewportPlanes {
    dims: (u32, u32),
    mode: PlaneMode,
    plane0: Buffer,
    plane1: Option<Buffer>,
    plane2: Option<Buffer>,
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
        dispatch_2d_pipeline(encoder, &runtime.pack_pipeline, self.dims);
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
fn write_row_u8_at(buffer: &Buffer, y: u32, x: u32, full_width: usize, src: &[u8]) {
    let row_start = y as usize * full_width + x as usize;
    let row_end = row_start + src.len();
    let len = full_width * (y as usize + 1);
    let dst = unsafe {
        core::slice::from_raw_parts_mut(buffer.contents().cast::<u8>(), len.max(row_end))
    };
    dst[row_start..row_end].copy_from_slice(src);
}

#[cfg(target_os = "macos")]
fn plane_mode_for_color_space(color_space: JpegColorSpace) -> Result<PlaneMode, Error> {
    match color_space {
        JpegColorSpace::Grayscale => Ok(PlaneMode::Gray),
        JpegColorSpace::YCbCr => Ok(PlaneMode::YCbCr),
        JpegColorSpace::Rgb => Ok(PlaneMode::Rgb),
        JpegColorSpace::Cmyk | JpegColorSpace::Ycck => Err(Error::MetalKernel {
            message: "Metal compute path does not support CMYK/YCCK JPEG output".to_string(),
        }),
    }
}

#[cfg(target_os = "macos")]
fn clear_buffer(buffer: &Buffer, len: usize) {
    unsafe {
        core::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, len);
    }
}

#[cfg(target_os = "macos")]
fn cached_viewport_stage(
    device: &Device,
    color_space: JpegColorSpace,
    dims: (u32, u32),
) -> Result<PlaneStage, Error> {
    let mode = plane_mode_for_color_space(color_space)?;
    VIEWPORT_PLANE_CACHE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let len = dims.0 as usize * dims.1 as usize;
        let refresh = slot
            .as_ref()
            .is_none_or(|cached| cached.dims != dims || cached.mode != mode);
        if refresh {
            let plane0 = device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared);
            let (plane1, plane2) = match mode {
                PlaneMode::Gray => (None, None),
                PlaneMode::YCbCr | PlaneMode::Rgb => (
                    Some(device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)),
                    Some(device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)),
                ),
            };
            *slot = Some(CachedViewportPlanes {
                dims,
                mode,
                plane0,
                plane1,
                plane2,
            });
        }

        let cached = slot.as_ref().expect("viewport plane cache");
        let stage = PlaneStage {
            dims,
            mode,
            plane0: cached.plane0.clone(),
            plane1: cached.plane1.clone(),
            plane2: cached.plane2.clone(),
        };
        clear_buffer(&stage.plane0, len);
        if let Some(plane1) = &stage.plane1 {
            clear_buffer(plane1, len);
        }
        if let Some(plane2) = &stage.plane2 {
            clear_buffer(plane2, len);
        }
        Ok(stage)
    })
}

#[cfg(target_os = "macos")]
impl ComponentRowWriter for ViewportPlaneWriter<'_> {
    fn write_gray_row(
        &mut self,
        y: u32,
        gray_row: &[u8],
    ) -> Result<(), slidecodec_jpeg::JpegError> {
        write_row_u8_at(
            &self.stage.plane0,
            self.dest.y + y,
            self.dest.x,
            self.stage.dims.0 as usize,
            gray_row,
        );
        Ok(())
    }

    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        chroma_blue_row: &[u8],
        chroma_red_row: &[u8],
    ) -> Result<(), slidecodec_jpeg::JpegError> {
        let width = self.stage.dims.0 as usize;
        write_row_u8_at(
            &self.stage.plane0,
            self.dest.y + y,
            self.dest.x,
            width,
            y_row,
        );
        write_row_u8_at(
            self.stage.plane1.as_ref().expect("Cb plane"),
            self.dest.y + y,
            self.dest.x,
            width,
            chroma_blue_row,
        );
        write_row_u8_at(
            self.stage.plane2.as_ref().expect("Cr plane"),
            self.dest.y + y,
            self.dest.x,
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
        let width = self.stage.dims.0 as usize;
        write_row_u8_at(
            &self.stage.plane0,
            self.dest.y + y,
            self.dest.x,
            width,
            r_row,
        );
        write_row_u8_at(
            self.stage.plane1.as_ref().expect("G plane"),
            self.dest.y + y,
            self.dest.x,
            width,
            g_row,
        );
        write_row_u8_at(
            self.stage.plane2.as_ref().expect("B plane"),
            self.dest.y + y,
            self.dest.x,
            width,
            b_row,
        );
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn dispatch_2d_pipeline(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    dims: (u32, u32),
) {
    let width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(width);
    let height = (max_threads / width).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(dims.0),
            height: u64::from(dims.1),
            depth: 1,
        },
        MTLSize {
            width,
            height,
            depth: 1,
        },
    );
}

#[cfg(target_os = "macos")]
fn dispatch_1d_pipeline(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    threads: u32,
) {
    let threadgroup_width = pipeline
        .max_total_threads_per_threadgroup()
        .max(pipeline.thread_execution_width())
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(threads.max(1)),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threadgroup_width,
            height: 1,
            depth: 1,
        },
    );
}

#[cfg(target_os = "macos")]
fn pixel_format_to_out_format(fmt: PixelFormat) -> Option<u32> {
    match fmt {
        PixelFormat::Gray8 => Some(OUT_GRAY),
        PixelFormat::Rgb8 => Some(OUT_RGB),
        PixelFormat::Rgba8 => Some(OUT_RGBA),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
fn fast420_params(
    packet: &JpegMetalFast420PacketV1,
    fmt: PixelFormat,
) -> Result<JpegFast420Params, Error> {
    let out_format = pixel_format_to_out_format(fmt).ok_or_else(|| Error::MetalKernel {
        message: format!("unsupported JPEG Metal fast420 pixel format {fmt:?}"),
    })?;
    let out_stride = packet.dimensions.0 as usize * fmt.bytes_per_pixel();
    Ok(JpegFast420Params {
        width: packet.dimensions.0,
        height: packet.dimensions.1,
        chroma_width: packet.dimensions.0.div_ceil(2),
        chroma_height: packet.dimensions.1.div_ceil(2),
        mcus_per_row: packet.mcus_per_row,
        mcu_rows: packet.mcu_rows,
        restart_interval_mcus: packet.restart_interval_mcus,
        restart_offset_count: u32::try_from(packet.restart_offsets.len())
            .expect("JPEG Metal fast420 restart offsets fit in u32"),
        entropy_len: u32::try_from(packet.entropy_bytes.len())
            .expect("JPEG Metal entropy payload fits in u32"),
        out_stride: u32::try_from(out_stride).expect("JPEG Metal output stride fits in u32"),
        alpha: u32::from(u8::MAX),
        out_format,
        origin_x: 0,
        origin_y: 0,
    })
}

#[cfg(target_os = "macos")]
fn full_mcu_window_420(dims: (u32, u32), roi: slidecodec_jpeg::Rect) -> slidecodec_jpeg::Rect {
    let x0 = (roi.x / 16) * 16;
    let y0 = (roi.y / 16) * 16;
    let x1 = (roi.x + roi.w).div_ceil(16) * 16;
    let y1 = (roi.y + roi.h).div_ceil(16) * 16;
    slidecodec_jpeg::Rect {
        x: x0,
        y: y0,
        w: x1.min(dims.0).saturating_sub(x0),
        h: y1.min(dims.1).saturating_sub(y0),
    }
}

#[cfg(target_os = "macos")]
fn fast420_region_params(
    packet: &JpegMetalFast420PacketV1,
    fmt: PixelFormat,
    source_window: slidecodec_jpeg::Rect,
) -> Result<JpegFast420Params, Error> {
    let out_format = pixel_format_to_out_format(fmt).ok_or_else(|| Error::MetalKernel {
        message: format!("unsupported JPEG Metal fast420 pixel format {fmt:?}"),
    })?;
    let out_stride = source_window.w as usize * fmt.bytes_per_pixel();
    Ok(JpegFast420Params {
        width: source_window.w,
        height: source_window.h,
        chroma_width: source_window.w.div_ceil(2),
        chroma_height: source_window.h.div_ceil(2),
        mcus_per_row: packet.mcus_per_row,
        mcu_rows: packet.mcu_rows,
        restart_interval_mcus: packet.restart_interval_mcus,
        restart_offset_count: u32::try_from(packet.restart_offsets.len())
            .expect("JPEG Metal fast420 restart offsets fit in u32"),
        entropy_len: u32::try_from(packet.entropy_bytes.len())
            .expect("JPEG Metal entropy payload fits in u32"),
        out_stride: u32::try_from(out_stride).expect("JPEG Metal output stride fits in u32"),
        alpha: u32::from(u8::MAX),
        out_format,
        origin_x: source_window.x,
        origin_y: source_window.y,
    })
}

#[cfg(target_os = "macos")]
fn fast420_scaled_params(
    packet: &JpegMetalFast420PacketV1,
    scale: slidecodec_core::Downscale,
) -> Option<JpegFast420ScaledParams> {
    let scale_shift = match scale {
        slidecodec_core::Downscale::Half => 1,
        slidecodec_core::Downscale::Quarter => 2,
        slidecodec_core::Downscale::Eighth => 3,
        _ => return None,
    };
    let denom = 1u32 << scale_shift;
    let scaled_width = packet.dimensions.0.div_ceil(denom);
    let scaled_height = packet.dimensions.1.div_ceil(denom);
    Some(JpegFast420ScaledParams {
        scaled_width,
        scaled_height,
        chroma_width: scaled_width.div_ceil(2),
        chroma_height: scaled_height.div_ceil(2),
        mcus_per_row: packet.mcus_per_row,
        mcu_rows: packet.mcu_rows,
        restart_interval_mcus: packet.restart_interval_mcus,
        restart_offset_count: u32::try_from(packet.restart_offsets.len())
            .expect("JPEG Metal fast420 restart offsets fit in u32"),
        entropy_len: u32::try_from(packet.entropy_bytes.len())
            .expect("JPEG Metal entropy payload fits in u32"),
        scale_shift,
        origin_x: 0,
        origin_y: 0,
    })
}

#[cfg(target_os = "macos")]
fn full_mcu_scaled_window_420(
    scaled_dims: (u32, u32),
    roi: slidecodec_jpeg::Rect,
    scale_shift: u32,
) -> slidecodec_jpeg::Rect {
    let mcu_size = 16u32 >> scale_shift;
    let x0 = (roi.x / mcu_size) * mcu_size;
    let y0 = (roi.y / mcu_size) * mcu_size;
    let x1 = (roi.x + roi.w).div_ceil(mcu_size) * mcu_size;
    let y1 = (roi.y + roi.h).div_ceil(mcu_size) * mcu_size;
    slidecodec_jpeg::Rect {
        x: x0,
        y: y0,
        w: x1.min(scaled_dims.0).saturating_sub(x0),
        h: y1.min(scaled_dims.1).saturating_sub(y0),
    }
}

#[cfg(target_os = "macos")]
fn fast420_scaled_region_params(
    packet: &JpegMetalFast420PacketV1,
    scale: slidecodec_core::Downscale,
    source_window: slidecodec_jpeg::Rect,
) -> Option<JpegFast420ScaledParams> {
    let full = fast420_scaled_params(packet, scale)?;
    Some(JpegFast420ScaledParams {
        scaled_width: source_window.w,
        scaled_height: source_window.h,
        chroma_width: source_window.w.div_ceil(2),
        chroma_height: source_window.h.div_ceil(2),
        origin_x: source_window.x,
        origin_y: source_window.y,
        ..full
    })
}

#[cfg(target_os = "macos")]
fn fast444_params(packet: &JpegMetalFast444PacketV1) -> JpegFast444Params {
    JpegFast444Params {
        width: packet.dimensions.0,
        height: packet.dimensions.1,
        mcus_per_row: packet.mcus_per_row,
        mcu_rows: packet.mcu_rows,
        restart_interval_mcus: packet.restart_interval_mcus,
        restart_offset_count: u32::try_from(packet.restart_offsets.len())
            .expect("JPEG Metal fast444 restart offsets fit in u32"),
        entropy_len: u32::try_from(packet.entropy_bytes.len())
            .expect("JPEG Metal fast444 entropy payload fits in u32"),
        origin_x: 0,
        origin_y: 0,
    }
}

#[cfg(target_os = "macos")]
fn fast444_region_params(
    packet: &JpegMetalFast444PacketV1,
    roi: slidecodec_jpeg::Rect,
) -> JpegFast444Params {
    JpegFast444Params {
        width: roi.w,
        height: roi.h,
        origin_x: roi.x,
        origin_y: roi.y,
        ..fast444_params(packet)
    }
}

#[cfg(target_os = "macos")]
fn fast444_scaled_params(
    packet: &JpegMetalFast444PacketV1,
    scale: slidecodec_core::Downscale,
) -> Option<JpegFast444ScaledParams> {
    let scale_shift = match scale {
        slidecodec_core::Downscale::Half => 1,
        slidecodec_core::Downscale::Quarter => 2,
        slidecodec_core::Downscale::Eighth => 3,
        _ => return None,
    };
    let denom = 1u32 << scale_shift;
    Some(JpegFast444ScaledParams {
        scaled_width: packet.dimensions.0.div_ceil(denom),
        scaled_height: packet.dimensions.1.div_ceil(denom),
        mcus_per_row: packet.mcus_per_row,
        mcu_rows: packet.mcu_rows,
        restart_interval_mcus: packet.restart_interval_mcus,
        restart_offset_count: u32::try_from(packet.restart_offsets.len())
            .expect("JPEG Metal fast444 restart offsets fit in u32"),
        entropy_len: u32::try_from(packet.entropy_bytes.len())
            .expect("JPEG Metal fast444 entropy payload fits in u32"),
        scale_shift,
        origin_x: 0,
        origin_y: 0,
    })
}

#[cfg(target_os = "macos")]
fn fast444_scaled_region_params(
    packet: &JpegMetalFast444PacketV1,
    scale: slidecodec_core::Downscale,
    roi: slidecodec_jpeg::Rect,
) -> Option<JpegFast444ScaledParams> {
    Some(JpegFast444ScaledParams {
        scaled_width: roi.w,
        scaled_height: roi.h,
        origin_x: roi.x,
        origin_y: roi.y,
        ..fast444_scaled_params(packet, scale)?
    })
}

#[cfg(target_os = "macos")]
fn fast420_windowed_pack_params_for_dims(
    dims: (u32, u32),
    fmt: PixelFormat,
    roi: slidecodec_jpeg::Rect,
) -> JpegFast420WindowedPackParams {
    let out_format = pixel_format_to_out_format(fmt)
        .ok_or_else(|| Error::MetalKernel {
            message: format!("unsupported JPEG Metal fast420 pixel format {fmt:?}"),
        })
        .expect("validated JPEG Metal fast420 pixel format");
    let out_stride = roi.w as usize * fmt.bytes_per_pixel();
    JpegFast420WindowedPackParams {
        src_width: dims.0,
        src_height: dims.1,
        chroma_width: dims.0.div_ceil(2),
        chroma_height: dims.1.div_ceil(2),
        src_x: roi.x,
        src_y: roi.y,
        width: roi.w,
        height: roi.h,
        out_stride: u32::try_from(out_stride).expect("JPEG Metal output stride fits in u32"),
        alpha: u32::from(u8::MAX),
        out_format,
    }
}

#[cfg(target_os = "macos")]
fn decode_error_from_cpu(
    decoder: &CpuDecoder<'_>,
    fmt: PixelFormat,
    status: JpegDecodeStatus,
) -> Error {
    if let Err(err) = decoder.decode(fmt) {
        Error::Decode(err)
    } else {
        let reason = match status.code {
            FAST420_STATUS_TRUNCATED => "truncated entropy stream",
            FAST420_STATUS_HUFFMAN => "invalid Huffman stream",
            _ => "unexpected Metal fast420 failure",
        };
        Error::MetalKernel {
            message: format!("{reason} at entropy byte {}", status.position),
        }
    }
}

#[cfg(target_os = "macos")]
fn restart_offsets_buffer(device: &Device, restart_offsets: &[u32]) -> Result<Buffer, Error> {
    if restart_offsets.is_empty() {
        return Err(Error::MetalKernel {
            message: "JPEG Metal restart offsets must contain at least one entry".to_string(),
        });
    }
    Ok(device.new_buffer_with_data(
        restart_offsets.as_ptr().cast(),
        size_of_val(restart_offsets) as u64,
        MTLResourceOptions::StorageModeShared,
    ))
}

#[cfg(target_os = "macos")]
fn restart_decode_thread_count(restart_interval_mcus: u32, restart_offsets_len: usize) -> u32 {
    if restart_interval_mcus != 0 {
        u32::try_from(restart_offsets_len)
            .expect("JPEG Metal restart offset count fits in u32")
            .max(1)
    } else {
        1
    }
}

#[cfg(target_os = "macos")]
fn decode_status_buffer(device: &Device, count: u32) -> Buffer {
    let statuses = vec![JpegDecodeStatus::default(); count as usize];
    device.new_buffer_with_data(
        statuses.as_ptr().cast(),
        size_of_val(statuses.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

#[cfg(target_os = "macos")]
fn first_decode_error_status(buffer: &Buffer, count: u32) -> Option<JpegDecodeStatus> {
    let statuses = unsafe {
        core::slice::from_raw_parts(buffer.contents().cast::<JpegDecodeStatus>(), count as usize)
    };
    statuses
        .iter()
        .copied()
        .find(|status| status.code != FAST420_STATUS_OK)
}

#[cfg(target_os = "macos")]
fn try_decode_fast420_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegMetalFast420PacketV1>,
    fmt: PixelFormat,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_out_format) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };

    let params = fast420_params(packet, fmt)?;
    let y_len = params.width as usize * params.height as usize;
    let chroma_len = params.chroma_width as usize * params.chroma_height as usize;
    let y_plane = runtime
        .device
        .new_buffer(y_len as u64, MTLResourceOptions::StorageModeShared);
    let chroma_buffers = [
        runtime
            .device
            .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared),
        runtime
            .device
            .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared),
    ];
    let decode_threads =
        restart_decode_thread_count(packet.restart_interval_mcus, packet.restart_offsets.len());
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;

    let dc_tables = [
        MetalHuffmanTableHost::from(&packet.y_dc_table),
        MetalHuffmanTableHost::from(&packet.cb_dc_table),
        MetalHuffmanTableHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        MetalHuffmanTableHost::from(&packet.y_ac_table),
        MetalHuffmanTableHost::from(&packet.cb_ac_table),
        MetalHuffmanTableHost::from(&packet.cr_ac_table),
    ];

    let out_buffer = (fmt != PixelFormat::Gray8).then(|| {
        runtime.device.new_buffer(
            (params.out_stride as usize * params.height as usize) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    });

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast420_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&chroma_buffers[0]), 0);
    decoder_encoder.set_buffer(3, Some(&chroma_buffers[1]), 0);
    decoder_encoder.set_bytes(
        4,
        size_of::<JpegFast420Params>() as u64,
        (&raw const params).cast(),
    );
    decoder_encoder.set_bytes(
        5,
        size_of::<[u16; 64]>() as u64,
        packet.y_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        6,
        size_of::<[u16; 64]>() as u64,
        packet.cb_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        7,
        size_of::<[u16; 64]>() as u64,
        packet.cr_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        8,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast420_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    if let Some(out_buffer) = out_buffer.as_ref() {
        let pack_encoder = command_buffer.new_compute_command_encoder();
        pack_encoder.set_compute_pipeline_state(&runtime.pack_420_pipeline);
        pack_encoder.set_buffer(0, Some(&y_plane), 0);
        pack_encoder.set_buffer(1, Some(&chroma_buffers[0]), 0);
        pack_encoder.set_buffer(2, Some(&chroma_buffers[1]), 0);
        pack_encoder.set_buffer(3, Some(out_buffer), 0);
        pack_encoder.set_bytes(
            4,
            size_of::<JpegFast420Params>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(pack_encoder, &runtime.pack_420_pipeline, packet.dimensions);
        pack_encoder.end_encoding();
    }

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads) {
        return Err(decode_error_from_cpu(decoder, fmt, status));
    }

    Ok(Some(match out_buffer {
        Some(out_buffer) => Surface::from_metal_buffer(out_buffer, packet.dimensions, fmt),
        None => Surface::from_metal_buffer(y_plane, packet.dimensions, fmt),
    }))
}

#[cfg(target_os = "macos")]
fn try_decode_fast420_region_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegMetalFast420PacketV1>,
    fmt: PixelFormat,
    roi: slidecodec_jpeg::Rect,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };

    let source_window = full_mcu_window_420(packet.dimensions, roi);
    let decode_params = fast420_region_params(packet, fmt, source_window)?;
    let local_roi = slidecodec_jpeg::Rect {
        x: roi.x - source_window.x,
        y: roi.y - source_window.y,
        w: roi.w,
        h: roi.h,
    };
    let pack_params =
        fast420_windowed_pack_params_for_dims((source_window.w, source_window.h), fmt, local_roi);
    let y_len = source_window.w as usize * source_window.h as usize;
    let chroma_len = source_window.w.div_ceil(2) as usize * source_window.h.div_ceil(2) as usize;
    let y_plane = runtime
        .device
        .new_buffer(y_len as u64, MTLResourceOptions::StorageModeShared);
    let chroma_buffers = [
        runtime
            .device
            .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared),
        runtime
            .device
            .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared),
    ];
    let decode_threads =
        restart_decode_thread_count(packet.restart_interval_mcus, packet.restart_offsets.len());
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;

    let dc_tables = [
        MetalHuffmanTableHost::from(&packet.y_dc_table),
        MetalHuffmanTableHost::from(&packet.cb_dc_table),
        MetalHuffmanTableHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        MetalHuffmanTableHost::from(&packet.y_ac_table),
        MetalHuffmanTableHost::from(&packet.cb_ac_table),
        MetalHuffmanTableHost::from(&packet.cr_ac_table),
    ];

    let out_buffer = runtime.device.new_buffer(
        (pack_params.out_stride as usize * roi.h as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast420_region_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&chroma_buffers[0]), 0);
    decoder_encoder.set_buffer(3, Some(&chroma_buffers[1]), 0);
    decoder_encoder.set_bytes(
        4,
        size_of::<JpegFast420Params>() as u64,
        (&raw const decode_params).cast(),
    );
    decoder_encoder.set_bytes(
        5,
        size_of::<[u16; 64]>() as u64,
        packet.y_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        6,
        size_of::<[u16; 64]>() as u64,
        packet.cb_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        7,
        size_of::<[u16; 64]>() as u64,
        packet.cr_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        8,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast420_region_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(&runtime.pack_420_windowed_pipeline);
    pack_encoder.set_buffer(0, Some(&y_plane), 0);
    pack_encoder.set_buffer(1, Some(&chroma_buffers[0]), 0);
    pack_encoder.set_buffer(2, Some(&chroma_buffers[1]), 0);
    pack_encoder.set_buffer(3, Some(&out_buffer), 0);
    pack_encoder.set_bytes(
        4,
        size_of::<JpegFast420WindowedPackParams>() as u64,
        (&raw const pack_params).cast(),
    );
    dispatch_2d_pipeline(
        pack_encoder,
        &runtime.pack_420_windowed_pipeline,
        (roi.w, roi.h),
    );
    pack_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads) {
        return Err(decode_error_from_cpu(decoder, fmt, status));
    }

    Ok(Some(Surface::from_metal_buffer(
        out_buffer,
        (roi.w, roi.h),
        fmt,
    )))
}

#[cfg(target_os = "macos")]
fn try_decode_fast420_scaled_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegMetalFast420PacketV1>,
    fmt: PixelFormat,
    scale: slidecodec_core::Downscale,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_out_format) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };
    let Some(params) = fast420_scaled_params(packet, scale) else {
        return Ok(None);
    };

    let y_len = params.scaled_width as usize * params.scaled_height as usize;
    let chroma_len = params.chroma_width as usize * params.chroma_height as usize;
    let y_plane = runtime
        .device
        .new_buffer(y_len as u64, MTLResourceOptions::StorageModeShared);
    let chroma_buffers = [
        runtime
            .device
            .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared),
        runtime
            .device
            .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared),
    ];
    let decode_threads =
        restart_decode_thread_count(packet.restart_interval_mcus, packet.restart_offsets.len());
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;

    let dc_tables = [
        MetalHuffmanTableHost::from(&packet.y_dc_table),
        MetalHuffmanTableHost::from(&packet.cb_dc_table),
        MetalHuffmanTableHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        MetalHuffmanTableHost::from(&packet.y_ac_table),
        MetalHuffmanTableHost::from(&packet.cb_ac_table),
        MetalHuffmanTableHost::from(&packet.cr_ac_table),
    ];

    let out_buffer = (fmt != PixelFormat::Gray8).then(|| {
        runtime.device.new_buffer(
            (params.scaled_width as usize * fmt.bytes_per_pixel() * params.scaled_height as usize)
                as u64,
            MTLResourceOptions::StorageModeShared,
        )
    });

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast420_scaled_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&chroma_buffers[0]), 0);
    decoder_encoder.set_buffer(3, Some(&chroma_buffers[1]), 0);
    decoder_encoder.set_bytes(
        4,
        size_of::<JpegFast420ScaledParams>() as u64,
        (&raw const params).cast(),
    );
    decoder_encoder.set_bytes(
        5,
        size_of::<[u16; 64]>() as u64,
        packet.y_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        6,
        size_of::<[u16; 64]>() as u64,
        packet.cb_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        7,
        size_of::<[u16; 64]>() as u64,
        packet.cr_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        8,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast420_scaled_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    if let Some(out_buffer) = out_buffer.as_ref() {
        let pack_params = JpegFast420Params {
            width: params.scaled_width,
            height: params.scaled_height,
            chroma_width: params.chroma_width,
            chroma_height: params.chroma_height,
            mcus_per_row: params.mcus_per_row,
            mcu_rows: params.mcu_rows,
            restart_interval_mcus: params.restart_interval_mcus,
            restart_offset_count: params.restart_offset_count,
            entropy_len: params.entropy_len,
            out_stride: u32::try_from(params.scaled_width as usize * fmt.bytes_per_pixel())
                .expect("JPEG Metal output stride fits in u32"),
            alpha: u32::from(u8::MAX),
            out_format: pixel_format_to_out_format(fmt).expect("validated output format"),
            origin_x: 0,
            origin_y: 0,
        };
        let pack_encoder = command_buffer.new_compute_command_encoder();
        pack_encoder.set_compute_pipeline_state(&runtime.pack_420_pipeline);
        pack_encoder.set_buffer(0, Some(&y_plane), 0);
        pack_encoder.set_buffer(1, Some(&chroma_buffers[0]), 0);
        pack_encoder.set_buffer(2, Some(&chroma_buffers[1]), 0);
        pack_encoder.set_buffer(3, Some(out_buffer), 0);
        pack_encoder.set_bytes(
            4,
            size_of::<JpegFast420Params>() as u64,
            (&raw const pack_params).cast(),
        );
        dispatch_2d_pipeline(
            pack_encoder,
            &runtime.pack_420_pipeline,
            (params.scaled_width, params.scaled_height),
        );
        pack_encoder.end_encoding();
    }

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads) {
        return Err(decode_error_from_cpu(decoder, fmt, status));
    }

    Ok(Some(match out_buffer {
        Some(out_buffer) => {
            Surface::from_metal_buffer(out_buffer, (params.scaled_width, params.scaled_height), fmt)
        }
        None => {
            Surface::from_metal_buffer(y_plane, (params.scaled_width, params.scaled_height), fmt)
        }
    }))
}

#[cfg(target_os = "macos")]
fn try_decode_fast420_scaled_region_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegMetalFast420PacketV1>,
    fmt: PixelFormat,
    scaled_roi: slidecodec_jpeg::Rect,
    scale: slidecodec_core::Downscale,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };
    let Some(full_params) = fast420_scaled_params(packet, scale) else {
        return Ok(None);
    };
    let source_window = full_mcu_scaled_window_420(
        (full_params.scaled_width, full_params.scaled_height),
        scaled_roi,
        full_params.scale_shift,
    );
    let Some(decode_params) = fast420_scaled_region_params(packet, scale, source_window) else {
        return Ok(None);
    };
    let local_roi = slidecodec_jpeg::Rect {
        x: scaled_roi.x - source_window.x,
        y: scaled_roi.y - source_window.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let pack_params =
        fast420_windowed_pack_params_for_dims((source_window.w, source_window.h), fmt, local_roi);
    let y_len = source_window.w as usize * source_window.h as usize;
    let chroma_len = source_window.w.div_ceil(2) as usize * source_window.h.div_ceil(2) as usize;
    let y_plane = runtime
        .device
        .new_buffer(y_len as u64, MTLResourceOptions::StorageModeShared);
    let chroma_buffers = [
        runtime
            .device
            .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared),
        runtime
            .device
            .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared),
    ];
    let decode_threads =
        restart_decode_thread_count(packet.restart_interval_mcus, packet.restart_offsets.len());
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;

    let dc_tables = [
        MetalHuffmanTableHost::from(&packet.y_dc_table),
        MetalHuffmanTableHost::from(&packet.cb_dc_table),
        MetalHuffmanTableHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        MetalHuffmanTableHost::from(&packet.y_ac_table),
        MetalHuffmanTableHost::from(&packet.cb_ac_table),
        MetalHuffmanTableHost::from(&packet.cr_ac_table),
    ];

    let out_buffer = runtime.device.new_buffer(
        (pack_params.out_stride as usize * scaled_roi.h as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast420_scaled_region_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&chroma_buffers[0]), 0);
    decoder_encoder.set_buffer(3, Some(&chroma_buffers[1]), 0);
    decoder_encoder.set_bytes(
        4,
        size_of::<JpegFast420ScaledParams>() as u64,
        (&raw const decode_params).cast(),
    );
    decoder_encoder.set_bytes(
        5,
        size_of::<[u16; 64]>() as u64,
        packet.y_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        6,
        size_of::<[u16; 64]>() as u64,
        packet.cb_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        7,
        size_of::<[u16; 64]>() as u64,
        packet.cr_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        8,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast420_scaled_region_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(&runtime.pack_420_windowed_pipeline);
    pack_encoder.set_buffer(0, Some(&y_plane), 0);
    pack_encoder.set_buffer(1, Some(&chroma_buffers[0]), 0);
    pack_encoder.set_buffer(2, Some(&chroma_buffers[1]), 0);
    pack_encoder.set_buffer(3, Some(&out_buffer), 0);
    pack_encoder.set_bytes(
        4,
        size_of::<JpegFast420WindowedPackParams>() as u64,
        (&raw const pack_params).cast(),
    );
    dispatch_2d_pipeline(
        pack_encoder,
        &runtime.pack_420_windowed_pipeline,
        (scaled_roi.w, scaled_roi.h),
    );
    pack_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads) {
        return Err(decode_error_from_cpu(decoder, fmt, status));
    }

    Ok(Some(Surface::from_metal_buffer(
        out_buffer,
        (scaled_roi.w, scaled_roi.h),
        fmt,
    )))
}

#[cfg(target_os = "macos")]
fn fast444_plane_mode(decoder: &CpuDecoder<'_>) -> PlaneMode {
    match decoder.info().color_space {
        JpegColorSpace::Rgb => PlaneMode::Rgb,
        _ => PlaneMode::YCbCr,
    }
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegMetalFast444PacketV1>,
    fmt: PixelFormat,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };

    let params = fast444_params(packet);
    let plane_len = params.width as usize * params.height as usize;
    let y_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let chroma_blue_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let chroma_red_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let decode_threads =
        restart_decode_thread_count(packet.restart_interval_mcus, packet.restart_offsets.len());
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;

    let dc_tables = [
        MetalHuffmanTableHost::from(&packet.y_dc_table),
        MetalHuffmanTableHost::from(&packet.cb_dc_table),
        MetalHuffmanTableHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        MetalHuffmanTableHost::from(&packet.y_ac_table),
        MetalHuffmanTableHost::from(&packet.cb_ac_table),
        MetalHuffmanTableHost::from(&packet.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&chroma_blue_plane), 0);
    decoder_encoder.set_buffer(3, Some(&chroma_red_plane), 0);
    decoder_encoder.set_bytes(
        4,
        size_of::<JpegFast444Params>() as u64,
        (&raw const params).cast(),
    );
    decoder_encoder.set_bytes(
        5,
        size_of::<[u16; 64]>() as u64,
        packet.y_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        6,
        size_of::<[u16; 64]>() as u64,
        packet.cb_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        7,
        size_of::<[u16; 64]>() as u64,
        packet.cr_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        8,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads) {
        return Err(decode_error_from_cpu(decoder, fmt, status));
    }

    PlaneStage {
        dims: packet.dimensions,
        mode: fast444_plane_mode(decoder),
        plane0: y_plane,
        plane1: Some(chroma_blue_plane),
        plane2: Some(chroma_red_plane),
    }
    .finish_with_runtime(runtime, fmt)
    .map(Some)
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_region_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegMetalFast444PacketV1>,
    fmt: PixelFormat,
    roi: slidecodec_jpeg::Rect,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };

    let params = fast444_region_params(packet, roi);
    let plane_len = params.width as usize * params.height as usize;
    let y_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let chroma_blue_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let chroma_red_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let decode_threads =
        restart_decode_thread_count(packet.restart_interval_mcus, packet.restart_offsets.len());
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;

    let dc_tables = [
        MetalHuffmanTableHost::from(&packet.y_dc_table),
        MetalHuffmanTableHost::from(&packet.cb_dc_table),
        MetalHuffmanTableHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        MetalHuffmanTableHost::from(&packet.y_ac_table),
        MetalHuffmanTableHost::from(&packet.cb_ac_table),
        MetalHuffmanTableHost::from(&packet.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_region_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&chroma_blue_plane), 0);
    decoder_encoder.set_buffer(3, Some(&chroma_red_plane), 0);
    decoder_encoder.set_bytes(
        4,
        size_of::<JpegFast444Params>() as u64,
        (&raw const params).cast(),
    );
    decoder_encoder.set_bytes(
        5,
        size_of::<[u16; 64]>() as u64,
        packet.y_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        6,
        size_of::<[u16; 64]>() as u64,
        packet.cb_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        7,
        size_of::<[u16; 64]>() as u64,
        packet.cr_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        8,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_region_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads) {
        return Err(decode_error_from_cpu(decoder, fmt, status));
    }

    PlaneStage {
        dims: (roi.w, roi.h),
        mode: fast444_plane_mode(decoder),
        plane0: y_plane,
        plane1: Some(chroma_blue_plane),
        plane2: Some(chroma_red_plane),
    }
    .finish_with_runtime(runtime, fmt)
    .map(Some)
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_scaled_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegMetalFast444PacketV1>,
    fmt: PixelFormat,
    scale: slidecodec_core::Downscale,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };
    let Some(params) = fast444_scaled_params(packet, scale) else {
        return Ok(None);
    };

    let plane_len = params.scaled_width as usize * params.scaled_height as usize;
    let y_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let chroma_blue_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let chroma_red_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let decode_threads =
        restart_decode_thread_count(packet.restart_interval_mcus, packet.restart_offsets.len());
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;

    let dc_tables = [
        MetalHuffmanTableHost::from(&packet.y_dc_table),
        MetalHuffmanTableHost::from(&packet.cb_dc_table),
        MetalHuffmanTableHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        MetalHuffmanTableHost::from(&packet.y_ac_table),
        MetalHuffmanTableHost::from(&packet.cb_ac_table),
        MetalHuffmanTableHost::from(&packet.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_scaled_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&chroma_blue_plane), 0);
    decoder_encoder.set_buffer(3, Some(&chroma_red_plane), 0);
    decoder_encoder.set_bytes(
        4,
        size_of::<JpegFast444ScaledParams>() as u64,
        (&raw const params).cast(),
    );
    decoder_encoder.set_bytes(
        5,
        size_of::<[u16; 64]>() as u64,
        packet.y_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        6,
        size_of::<[u16; 64]>() as u64,
        packet.cb_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        7,
        size_of::<[u16; 64]>() as u64,
        packet.cr_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        8,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_scaled_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads) {
        return Err(decode_error_from_cpu(decoder, fmt, status));
    }

    PlaneStage {
        dims: (params.scaled_width, params.scaled_height),
        mode: fast444_plane_mode(decoder),
        plane0: y_plane,
        plane1: Some(chroma_blue_plane),
        plane2: Some(chroma_red_plane),
    }
    .finish_with_runtime(runtime, fmt)
    .map(Some)
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_scaled_region_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegMetalFast444PacketV1>,
    fmt: PixelFormat,
    scaled_roi: slidecodec_jpeg::Rect,
    scale: slidecodec_core::Downscale,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };
    let Some(params) = fast444_scaled_region_params(packet, scale, scaled_roi) else {
        return Ok(None);
    };

    let plane_len = params.scaled_width as usize * params.scaled_height as usize;
    let y_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let chroma_blue_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let chroma_red_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let decode_threads =
        restart_decode_thread_count(packet.restart_interval_mcus, packet.restart_offsets.len());
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;

    let dc_tables = [
        MetalHuffmanTableHost::from(&packet.y_dc_table),
        MetalHuffmanTableHost::from(&packet.cb_dc_table),
        MetalHuffmanTableHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        MetalHuffmanTableHost::from(&packet.y_ac_table),
        MetalHuffmanTableHost::from(&packet.cb_ac_table),
        MetalHuffmanTableHost::from(&packet.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_scaled_region_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&chroma_blue_plane), 0);
    decoder_encoder.set_buffer(3, Some(&chroma_red_plane), 0);
    decoder_encoder.set_bytes(
        4,
        size_of::<JpegFast444ScaledParams>() as u64,
        (&raw const params).cast(),
    );
    decoder_encoder.set_bytes(
        5,
        size_of::<[u16; 64]>() as u64,
        packet.y_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        6,
        size_of::<[u16; 64]>() as u64,
        packet.cb_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        7,
        size_of::<[u16; 64]>() as u64,
        packet.cr_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        8,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_scaled_region_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads) {
        return Err(decode_error_from_cpu(decoder, fmt, status));
    }

    PlaneStage {
        dims: (scaled_roi.w, scaled_roi.h),
        mode: fast444_plane_mode(decoder),
        plane0: y_plane,
        plane1: Some(chroma_blue_plane),
        plane2: Some(chroma_red_plane),
    }
    .finish_with_runtime(runtime, fmt)
    .map(Some)
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
    fast444_packet: Option<&JpegMetalFast444PacketV1>,
    fast420_packet: Option<&JpegMetalFast420PacketV1>,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        if let Some(surface) = try_decode_fast444_to_surface(runtime, decoder, fast444_packet, fmt)?
        {
            return Ok(surface);
        }
        if let Some(surface) = try_decode_fast420_to_surface(runtime, decoder, fast420_packet, fmt)?
        {
            return Ok(surface);
        }
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
    fast444_packet: Option<&JpegMetalFast444PacketV1>,
    fast420_packet: Option<&JpegMetalFast420PacketV1>,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        if let Some(surface) =
            try_decode_fast444_region_to_surface(runtime, decoder, fast444_packet, fmt, roi)?
        {
            return Ok(surface);
        }
        if let Some(surface) =
            try_decode_fast420_region_to_surface(runtime, decoder, fast420_packet, fmt, roi)?
        {
            return Ok(surface);
        }
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
    fast444_packet: Option<&JpegMetalFast444PacketV1>,
    fast420_packet: Option<&JpegMetalFast420PacketV1>,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        if let Some(surface) =
            try_decode_fast444_scaled_to_surface(runtime, decoder, fast444_packet, fmt, scale)?
        {
            return Ok(surface);
        }
        if let Some(surface) =
            try_decode_fast420_scaled_to_surface(runtime, decoder, fast420_packet, fmt, scale)?
        {
            return Ok(surface);
        }
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
pub(crate) fn decode_region_scaled_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut slidecodec_jpeg::ScratchPool,
    fmt: PixelFormat,
    roi: slidecodec_jpeg::Rect,
    scale: slidecodec_core::Downscale,
    fast444_packet: Option<&JpegMetalFast444PacketV1>,
    fast420_packet: Option<&JpegMetalFast420PacketV1>,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let scaled_roi = scaled_rect_covering(
            Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        );
        if let Some(surface) = try_decode_fast444_scaled_region_to_surface(
            runtime,
            decoder,
            fast444_packet,
            fmt,
            slidecodec_jpeg::Rect {
                x: scaled_roi.x,
                y: scaled_roi.y,
                w: scaled_roi.w,
                h: scaled_roi.h,
            },
            scale,
        )? {
            return Ok(surface);
        }
        if let Some(surface) = try_decode_fast420_scaled_region_to_surface(
            runtime,
            decoder,
            fast420_packet,
            fmt,
            slidecodec_jpeg::Rect {
                x: scaled_roi.x,
                y: scaled_roi.y,
                w: scaled_roi.w,
                h: scaled_roi.h,
            },
            scale,
        )? {
            return Ok(surface);
        }
        let scaled = scaled_rect_covering(
            Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
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
pub(crate) fn compose_rgb_viewport_from_regions(
    decoder: &CpuDecoder<'_>,
    pool: &mut slidecodec_jpeg::ScratchPool,
    scale: slidecodec_core::Downscale,
    viewport_dims: (u32, u32),
    tiles: &[ViewportTile],
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let mut stage =
            cached_viewport_stage(&runtime.device, decoder.info().color_space, viewport_dims)?;
        for tile in tiles {
            let dims = scaled_rect_covering(tile.source_roi, scale);
            if (dims.w, dims.h) != (tile.dest.w, tile.dest.h) {
                return Err(Error::MetalKernel {
                    message: format!(
                        "viewport tile dims {:?} do not match destination rect {:?}",
                        (dims.w, dims.h),
                        tile.dest
                    ),
                });
            }
            let mut writer = ViewportPlaneWriter {
                stage: &mut stage,
                dest: tile.dest,
            };
            decoder.decode_region_component_rows_with_scratch(
                pool,
                &mut writer,
                slidecodec_jpeg::Rect {
                    x: tile.source_roi.x,
                    y: tile.source_roi.y,
                    w: tile.source_roi.w,
                    h: tile.source_roi.h,
                },
                scale,
            )?;
        }
        stage.finish_with_runtime(runtime, PixelFormat::Rgb8)
    })
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    const BASELINE_420: &[u8] =
        include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");
    const BASELINE_444: &[u8] = include_bytes!("../../../corpus/conformance/baseline_444_8x8.jpg");

    #[test]
    fn fast420_packet_scaled_decode_matches_cpu_scaled_bytes() {
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let packet =
            slidecodec_jpeg::__private::build_metal_fast420_packet(BASELINE_420).expect("packet");
        let (expected, _) = decoder
            .decode_scaled(PixelFormat::Rgb8, slidecodec_core::Downscale::Quarter)
            .expect("cpu scaled");

        let surface = with_runtime(|runtime| {
            let surface = try_decode_fast420_scaled_to_surface(
                runtime,
                &decoder,
                Some(&packet),
                PixelFormat::Rgb8,
                slidecodec_core::Downscale::Quarter,
            )?
            .expect("fast420 scaled surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal scaled");

        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    #[test]
    fn fast420_packet_region_decode_matches_cpu_region_bytes() {
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let packet =
            slidecodec_jpeg::__private::build_metal_fast420_packet(BASELINE_420).expect("packet");
        let roi = slidecodec_jpeg::Rect {
            x: 3,
            y: 2,
            w: 9,
            h: 10,
        };
        let (expected, _) = decoder
            .decode_region(PixelFormat::Rgb8, roi)
            .expect("cpu region");

        let surface = with_runtime(|runtime| {
            let surface = try_decode_fast420_region_to_surface(
                runtime,
                &decoder,
                Some(&packet),
                PixelFormat::Rgb8,
                roi,
            )?
            .expect("fast420 region surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal region");

        assert_eq!(surface.dimensions, (roi.w, roi.h));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    #[test]
    fn fast420_packet_region_scaled_decode_matches_cpu_region_scaled_bytes() {
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let packet =
            slidecodec_jpeg::__private::build_metal_fast420_packet(BASELINE_420).expect("packet");
        let roi = slidecodec_jpeg::Rect {
            x: 3,
            y: 2,
            w: 9,
            h: 10,
        };
        let scale = slidecodec_core::Downscale::Quarter;
        let (expected, _) = decoder
            .decode_region_scaled(PixelFormat::Rgb8, roi, scale)
            .expect("cpu region scaled");
        let scaled_roi = slidecodec_jpeg::Rect {
            x: roi.x / 4,
            y: roi.y / 4,
            w: (roi.x + roi.w).div_ceil(4) - (roi.x / 4),
            h: (roi.y + roi.h).div_ceil(4) - (roi.y / 4),
        };

        let surface = with_runtime(|runtime| {
            let surface = try_decode_fast420_scaled_region_to_surface(
                runtime,
                &decoder,
                Some(&packet),
                PixelFormat::Rgb8,
                scaled_roi,
                scale,
            )?
            .expect("fast420 scaled region surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal region scaled");

        assert_eq!(surface.dimensions, (scaled_roi.w, scaled_roi.h));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    #[test]
    fn fast444_packet_full_decode_matches_cpu_bytes() {
        let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
        let packet =
            slidecodec_jpeg::__private::build_metal_fast444_packet(BASELINE_444).expect("packet");
        let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

        let surface = with_runtime(|runtime| {
            let surface =
                try_decode_fast444_to_surface(runtime, &decoder, Some(&packet), PixelFormat::Rgb8)?
                    .expect("fast444 surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal full decode");

        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    #[test]
    fn fast444_packet_scaled_decode_matches_cpu_scaled_bytes() {
        let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
        let packet =
            slidecodec_jpeg::__private::build_metal_fast444_packet(BASELINE_444).expect("packet");
        let (expected, _) = decoder
            .decode_scaled(PixelFormat::Rgb8, slidecodec_core::Downscale::Quarter)
            .expect("cpu scaled");

        let surface = with_runtime(|runtime| {
            let surface = try_decode_fast444_scaled_to_surface(
                runtime,
                &decoder,
                Some(&packet),
                PixelFormat::Rgb8,
                slidecodec_core::Downscale::Quarter,
            )?
            .expect("fast444 scaled surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal scaled");

        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    #[test]
    fn fast444_packet_region_decode_matches_cpu_region_bytes() {
        let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
        let packet =
            slidecodec_jpeg::__private::build_metal_fast444_packet(BASELINE_444).expect("packet");
        let roi = slidecodec_jpeg::Rect {
            x: 1,
            y: 2,
            w: 5,
            h: 4,
        };
        let (expected, _) = decoder
            .decode_region(PixelFormat::Rgb8, roi)
            .expect("cpu region");

        let surface = with_runtime(|runtime| {
            let surface = try_decode_fast444_region_to_surface(
                runtime,
                &decoder,
                Some(&packet),
                PixelFormat::Rgb8,
                roi,
            )?
            .expect("fast444 region surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal region");

        assert_eq!(surface.dimensions, (roi.w, roi.h));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    #[test]
    fn fast444_packet_region_scaled_decode_matches_cpu_region_scaled_bytes() {
        let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
        let packet =
            slidecodec_jpeg::__private::build_metal_fast444_packet(BASELINE_444).expect("packet");
        let roi = slidecodec_jpeg::Rect {
            x: 1,
            y: 2,
            w: 5,
            h: 4,
        };
        let scale = slidecodec_core::Downscale::Quarter;
        let (expected, _) = decoder
            .decode_region_scaled(PixelFormat::Rgb8, roi, scale)
            .expect("cpu region scaled");
        let scaled_roi = slidecodec_jpeg::Rect {
            x: roi.x / 4,
            y: roi.y / 4,
            w: (roi.x + roi.w).div_ceil(4) - (roi.x / 4),
            h: (roi.y + roi.h).div_ceil(4) - (roi.y / 4),
        };

        let surface = with_runtime(|runtime| {
            let surface = try_decode_fast444_scaled_region_to_surface(
                runtime,
                &decoder,
                Some(&packet),
                PixelFormat::Rgb8,
                scaled_roi,
                scale,
            )?
            .expect("fast444 scaled region surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal region scaled");

        assert_eq!(surface.dimensions, (scaled_roi.w, scaled_roi.h));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}
