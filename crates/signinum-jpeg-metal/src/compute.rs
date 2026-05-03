// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::similar_names)]

#[cfg(target_os = "macos")]
use std::{
    cell::RefCell,
    ffi::OsStr,
    mem::{size_of, size_of_val},
    time::{Duration, Instant},
};

#[cfg(target_os = "macos")]
use metal::{
    Buffer, CommandBufferRef, CommandQueue, CompileOptions, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize,
};
use signinum_core::{BackendRequest, PixelFormat, Rect};
use signinum_jpeg::{
    adapter::{
        JpegMetalEntropyCheckpointV1, JpegMetalFast420PacketV1, JpegMetalFast422PacketV1,
        JpegMetalFast444PacketV1, MetalHuffmanTable as PacketHuffmanTable,
    },
    ColorSpace as JpegColorSpace, ComponentRowWriter, Decoder as CpuDecoder,
};

use crate::viewport::ViewportTile;
use crate::{batch, Error, Surface};

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
const AUTO_METAL_MIN_BATCH_REQUESTS: usize = 8;
#[cfg(target_os = "macos")]
const AUTO_METAL_MIN_BATCH_EDGE: u32 = 512;
#[cfg(target_os = "macos")]
const REGION_SCALED_BATCH_CHUNK: usize = 8;
#[cfg(target_os = "macos")]
const SPLIT_FAST420_BATCH_ENV: &str = "SIGNINUM_JPEG_METAL_SPLIT_FAST420_BATCH";
#[cfg(target_os = "macos")]
const FAST420_BATCH_TIMING_ENV: &str = "SIGNINUM_JPEG_METAL_FAST420_BATCH_TIMING";

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
    restart_start_mcu: u32,
    entropy_len: u32,
    out_stride: u32,
    alpha: u32,
    out_format: u32,
    origin_x: u32,
    origin_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct JpegFast420ScaledParams {
    scaled_width: u32,
    scaled_height: u32,
    chroma_width: u32,
    chroma_height: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    restart_interval_mcus: u32,
    restart_offset_count: u32,
    restart_start_mcu: u32,
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
    restart_start_mcu: u32,
    entropy_len: u32,
    origin_x: u32,
    origin_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct JpegFast444ScaledParams {
    scaled_width: u32,
    scaled_height: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    restart_interval_mcus: u32,
    restart_offset_count: u32,
    restart_start_mcu: u32,
    entropy_len: u32,
    scale_shift: u32,
    origin_x: u32,
    origin_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
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
struct JpegFast420BatchParams {
    width: u32,
    height: u32,
    chroma_width: u32,
    chroma_height: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    segment_count: u32,
    tile_count: u32,
    out_stride: u32,
    alpha: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct JpegFastRegionScaledBatchParams {
    scaled_width: u32,
    scaled_height: u32,
    chroma_width: u32,
    chroma_height: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    segment_count: u32,
    tile_count: u32,
    scale_shift: u32,
    origin_x: u32,
    origin_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct JpegWindowedPackBatchParams {
    src_width: u32,
    src_height: u32,
    chroma_width: u32,
    chroma_height: u32,
    src_x: u32,
    src_y: u32,
    width: u32,
    height: u32,
    tile_count: u32,
    out_stride: u32,
    alpha: u32,
    mode: u32,
    out_format: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct PreparedHuffmanHost {
    min_code: [i32; 17],
    max_code: [i32; 17],
    val_offset: [i32; 17],
    values: [u8; 256],
    fast_symbol: [u8; 512],
    fast_len: [u8; 512],
    values_len: u16,
    reserved: u16,
}

#[cfg(target_os = "macos")]
impl From<&PacketHuffmanTable> for PreparedHuffmanHost {
    fn from(value: &PacketHuffmanTable) -> Self {
        let mut min_code = [i32::MAX; 17];
        let mut max_code = [-1i32; 17];
        let mut val_offset = [0i32; 17];
        let mut values = [0u8; 256];
        let mut fast_symbol = [0u8; 512];
        let mut fast_len = [0u8; 512];
        let values_len = usize::from(value.values_len);
        values[..values_len].copy_from_slice(&value.values[..values_len]);

        let mut huffsize = [0u8; 256];
        let mut huffsize_len = 0usize;
        for (len_minus_1, &count) in value.bits.iter().enumerate() {
            let len = u8::try_from(len_minus_1 + 1).expect("JPEG Huffman code length fits in u8");
            for _ in 0..count {
                huffsize[huffsize_len] = len;
                huffsize_len += 1;
            }
        }

        let mut huffcode = [0u16; 256];
        let mut code = 0u32;
        let mut si = huffsize.first().copied().unwrap_or(0);
        for (idx, &size) in huffsize[..huffsize_len].iter().enumerate() {
            while size != si {
                code <<= 1;
                si += 1;
            }
            huffcode[idx] = u16::try_from(code).expect("JPEG Huffman code fits in u16");
            code += 1;
        }

        let mut idx = 0usize;
        for (len_minus_1, &count) in value.bits.iter().enumerate() {
            let len = len_minus_1 + 1;
            let count = usize::from(count);
            if count == 0 {
                continue;
            }
            min_code[len] = i32::from(huffcode[idx]);
            max_code[len] = i32::from(huffcode[idx + count - 1]);
            val_offset[len] =
                i32::try_from(idx).expect("JPEG Huffman value index fits in i32") - min_code[len];
            idx += count;
        }

        for idx in 0..huffsize_len {
            let len = usize::from(huffsize[idx]);
            if len == 0 || len > 9 {
                continue;
            }
            let code = usize::from(huffcode[idx]);
            let prefix = code << (9 - len);
            let fill = 1usize << (9 - len);
            for suffix in 0..fill {
                fast_symbol[prefix | suffix] = values[idx];
                fast_len[prefix | suffix] = huffsize[idx];
            }
        }

        Self {
            min_code,
            max_code,
            val_offset,
            values,
            fast_symbol,
            fast_len,
            values_len: value.values_len,
            reserved: 0,
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
#[repr(C)]
#[derive(Clone, Copy)]
struct JpegEntropyCheckpointHost {
    mcu_index: u32,
    entropy_pos: u32,
    bit_acc: u64,
    bit_count: u32,
    y_prev_dc: i32,
    cb_prev_dc: i32,
    cr_prev_dc: i32,
    reserved: u32,
}

#[cfg(target_os = "macos")]
impl From<JpegMetalEntropyCheckpointV1> for JpegEntropyCheckpointHost {
    fn from(value: JpegMetalEntropyCheckpointV1) -> Self {
        Self {
            mcu_index: value.mcu_index,
            entropy_pos: value.entropy_pos,
            bit_acc: value.bit_acc,
            bit_count: value.bit_count,
            y_prev_dc: value.y_prev_dc,
            cb_prev_dc: value.cb_prev_dc,
            cr_prev_dc: value.cr_prev_dc,
            reserved: value.reserved,
        }
    }
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
    pack_420_rgb_pipeline: ComputePipelineState,
    pack_420_rgba_pipeline: ComputePipelineState,
    pack_420_rgb_batch_pipeline: ComputePipelineState,
    pack_420_windowed_rgb_batch_pipeline: ComputePipelineState,
    pack_422_rgb_pipeline: ComputePipelineState,
    pack_422_rgba_pipeline: ComputePipelineState,
    pack_422_rgb_batch_pipeline: ComputePipelineState,
    pack_422_windowed_rgb_batch_pipeline: ComputePipelineState,
    pack_444_rgb_batch_pipeline: ComputePipelineState,
    pack_422_windowed_pipeline: ComputePipelineState,
    pack_422_windowed_rgb_pipeline: ComputePipelineState,
    pack_422_windowed_rgba_pipeline: ComputePipelineState,
    pack_420_windowed_pipeline: ComputePipelineState,
    pack_420_windowed_rgb_pipeline: ComputePipelineState,
    pack_420_windowed_rgba_pipeline: ComputePipelineState,
    fast420_decode_pipeline: ComputePipelineState,
    fast420_batch_decode_pipeline: ComputePipelineState,
    fast420_batch_coeffs_decode_pipeline: ComputePipelineState,
    fast420_batch_idct_deposit_pipeline: ComputePipelineState,
    fast420_scaled_region_batch_decode_pipeline: ComputePipelineState,
    fast422_decode_pipeline: ComputePipelineState,
    fast422_batch_decode_pipeline: ComputePipelineState,
    fast422_scaled_region_batch_decode_pipeline: ComputePipelineState,
    fast422_region_decode_pipeline: ComputePipelineState,
    fast422_scaled_decode_pipeline: ComputePipelineState,
    fast422_scaled_region_decode_pipeline: ComputePipelineState,
    fast420_region_decode_pipeline: ComputePipelineState,
    fast420_scaled_decode_pipeline: ComputePipelineState,
    fast420_scaled_region_decode_pipeline: ComputePipelineState,
    fast444_decode_pipeline: ComputePipelineState,
    fast444_region_decode_pipeline: ComputePipelineState,
    fast444_scaled_decode_pipeline: ComputePipelineState,
    fast444_scaled_region_decode_pipeline: ComputePipelineState,
    fast444_scaled_region_batch_decode_pipeline: ComputePipelineState,
}

#[cfg(target_os = "macos")]
impl MetalRuntime {
    fn new() -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or_else(|| "Metal is unavailable on this host".to_string())?;
        Self::new_with_device(device)
    }

    fn new_with_device(device: Device) -> Result<Self, String> {
        let options = CompileOptions::new();
        let library = device.new_library_with_source(SHADER_SOURCE, &options)?;
        let pack_function = library.get_function("jpeg_pack", None)?;
        let pack_pipeline = device.new_compute_pipeline_state_with_function(&pack_function)?;
        let pack_420_function = library.get_function("jpeg_pack_420", None)?;
        let pack_420_pipeline =
            device.new_compute_pipeline_state_with_function(&pack_420_function)?;
        let pack_420_rgb_function = library.get_function("jpeg_pack_420_rgb", None)?;
        let pack_420_rgb_pipeline =
            device.new_compute_pipeline_state_with_function(&pack_420_rgb_function)?;
        let pack_420_rgba_function = library.get_function("jpeg_pack_420_rgba", None)?;
        let pack_420_rgba_pipeline =
            device.new_compute_pipeline_state_with_function(&pack_420_rgba_function)?;
        let pack_420_rgb_batch_function = library.get_function("jpeg_pack_420_rgb_batch", None)?;
        let pack_420_rgb_batch_pipeline =
            device.new_compute_pipeline_state_with_function(&pack_420_rgb_batch_function)?;
        let pack_420_windowed_rgb_batch_function =
            library.get_function("jpeg_pack_420_windowed_rgb_batch", None)?;
        let pack_420_windowed_rgb_batch_pipeline = device
            .new_compute_pipeline_state_with_function(&pack_420_windowed_rgb_batch_function)?;
        let pack_422_rgb_function = library.get_function("jpeg_pack_422_rgb", None)?;
        let pack_422_rgb_pipeline =
            device.new_compute_pipeline_state_with_function(&pack_422_rgb_function)?;
        let pack_422_rgba_function = library.get_function("jpeg_pack_422_rgba", None)?;
        let pack_422_rgba_pipeline =
            device.new_compute_pipeline_state_with_function(&pack_422_rgba_function)?;
        let pack_422_rgb_batch_function = library.get_function("jpeg_pack_422_rgb_batch", None)?;
        let pack_422_rgb_batch_pipeline =
            device.new_compute_pipeline_state_with_function(&pack_422_rgb_batch_function)?;
        let pack_422_windowed_rgb_batch_function =
            library.get_function("jpeg_pack_422_windowed_rgb_batch", None)?;
        let pack_422_windowed_rgb_batch_pipeline = device
            .new_compute_pipeline_state_with_function(&pack_422_windowed_rgb_batch_function)?;
        let pack_444_rgb_batch_function = library.get_function("jpeg_pack_444_rgb_batch", None)?;
        let pack_444_rgb_batch_pipeline =
            device.new_compute_pipeline_state_with_function(&pack_444_rgb_batch_function)?;
        let pack_422_windowed_function = library.get_function("jpeg_pack_422_windowed", None)?;
        let pack_422_windowed_pipeline =
            device.new_compute_pipeline_state_with_function(&pack_422_windowed_function)?;
        let pack_422_windowed_rgb_function =
            library.get_function("jpeg_pack_422_windowed_rgb", None)?;
        let pack_422_windowed_rgb_pipeline =
            device.new_compute_pipeline_state_with_function(&pack_422_windowed_rgb_function)?;
        let pack_422_windowed_rgba_function =
            library.get_function("jpeg_pack_422_windowed_rgba", None)?;
        let pack_422_windowed_rgba_pipeline =
            device.new_compute_pipeline_state_with_function(&pack_422_windowed_rgba_function)?;
        let pack_420_windowed_function = library.get_function("jpeg_pack_420_windowed", None)?;
        let pack_420_windowed_pipeline =
            device.new_compute_pipeline_state_with_function(&pack_420_windowed_function)?;
        let pack_420_windowed_rgb_function =
            library.get_function("jpeg_pack_420_windowed_rgb", None)?;
        let pack_420_windowed_rgb_pipeline =
            device.new_compute_pipeline_state_with_function(&pack_420_windowed_rgb_function)?;
        let pack_420_windowed_rgba_function =
            library.get_function("jpeg_pack_420_windowed_rgba", None)?;
        let pack_420_windowed_rgba_pipeline =
            device.new_compute_pipeline_state_with_function(&pack_420_windowed_rgba_function)?;
        let fast420_decode_function = library.get_function("jpeg_decode_fast420", None)?;
        let fast420_decode_pipeline =
            device.new_compute_pipeline_state_with_function(&fast420_decode_function)?;
        let fast420_batch_decode_function =
            library.get_function("jpeg_decode_fast420_batch", None)?;
        let fast420_batch_decode_pipeline =
            device.new_compute_pipeline_state_with_function(&fast420_batch_decode_function)?;
        let fast420_batch_coeffs_decode_function =
            library.get_function("jpeg_decode_fast420_batch_coeffs", None)?;
        let fast420_batch_coeffs_decode_pipeline = device
            .new_compute_pipeline_state_with_function(&fast420_batch_coeffs_decode_function)?;
        let fast420_batch_idct_deposit_function =
            library.get_function("jpeg_idct_deposit_fast420_batch", None)?;
        let fast420_batch_idct_deposit_pipeline = device
            .new_compute_pipeline_state_with_function(&fast420_batch_idct_deposit_function)?;
        let fast420_scaled_region_batch_decode_function =
            library.get_function("jpeg_decode_fast420_scaled_region_batch", None)?;
        let fast420_scaled_region_batch_decode_pipeline = device
            .new_compute_pipeline_state_with_function(
                &fast420_scaled_region_batch_decode_function,
            )?;
        let fast422_decode_function = library.get_function("jpeg_decode_fast422", None)?;
        let fast422_decode_pipeline =
            device.new_compute_pipeline_state_with_function(&fast422_decode_function)?;
        let fast422_batch_decode_function =
            library.get_function("jpeg_decode_fast422_batch", None)?;
        let fast422_batch_decode_pipeline =
            device.new_compute_pipeline_state_with_function(&fast422_batch_decode_function)?;
        let fast422_scaled_region_batch_decode_function =
            library.get_function("jpeg_decode_fast422_scaled_region_batch", None)?;
        let fast422_scaled_region_batch_decode_pipeline = device
            .new_compute_pipeline_state_with_function(
                &fast422_scaled_region_batch_decode_function,
            )?;
        let fast422_region_decode_function =
            library.get_function("jpeg_decode_fast422_region", None)?;
        let fast422_region_decode_pipeline =
            device.new_compute_pipeline_state_with_function(&fast422_region_decode_function)?;
        let fast422_scaled_decode_function =
            library.get_function("jpeg_decode_fast422_scaled", None)?;
        let fast422_scaled_decode_pipeline =
            device.new_compute_pipeline_state_with_function(&fast422_scaled_decode_function)?;
        let fast422_scaled_region_decode_function =
            library.get_function("jpeg_decode_fast422_scaled_region", None)?;
        let fast422_scaled_region_decode_pipeline = device
            .new_compute_pipeline_state_with_function(&fast422_scaled_region_decode_function)?;
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
        let fast444_scaled_region_batch_decode_function =
            library.get_function("jpeg_decode_fast444_scaled_region_batch", None)?;
        let fast444_scaled_region_batch_decode_pipeline = device
            .new_compute_pipeline_state_with_function(
                &fast444_scaled_region_batch_decode_function,
            )?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pack_pipeline,
            pack_420_pipeline,
            pack_420_rgb_pipeline,
            pack_420_rgba_pipeline,
            pack_420_rgb_batch_pipeline,
            pack_420_windowed_rgb_batch_pipeline,
            pack_422_rgb_pipeline,
            pack_422_rgba_pipeline,
            pack_422_rgb_batch_pipeline,
            pack_422_windowed_rgb_batch_pipeline,
            pack_444_rgb_batch_pipeline,
            pack_422_windowed_pipeline,
            pack_422_windowed_rgb_pipeline,
            pack_422_windowed_rgba_pipeline,
            pack_420_windowed_pipeline,
            pack_420_windowed_rgb_pipeline,
            pack_420_windowed_rgba_pipeline,
            fast420_decode_pipeline,
            fast420_batch_decode_pipeline,
            fast420_batch_coeffs_decode_pipeline,
            fast420_batch_idct_deposit_pipeline,
            fast420_scaled_region_batch_decode_pipeline,
            fast422_decode_pipeline,
            fast422_batch_decode_pipeline,
            fast422_scaled_region_batch_decode_pipeline,
            fast422_region_decode_pipeline,
            fast422_scaled_decode_pipeline,
            fast422_scaled_region_decode_pipeline,
            fast420_region_decode_pipeline,
            fast420_scaled_decode_pipeline,
            fast420_scaled_region_decode_pipeline,
            fast444_decode_pipeline,
            fast444_region_decode_pipeline,
            fast444_scaled_decode_pipeline,
            fast444_scaled_region_decode_pipeline,
            fast444_scaled_region_batch_decode_pipeline,
        })
    }
}

#[cfg(target_os = "macos")]
fn pack_420_pipeline_for_format(runtime: &MetalRuntime, fmt: PixelFormat) -> &ComputePipelineState {
    match fmt {
        PixelFormat::Rgb8 => &runtime.pack_420_rgb_pipeline,
        PixelFormat::Rgba8 => &runtime.pack_420_rgba_pipeline,
        _ => &runtime.pack_420_pipeline,
    }
}

#[cfg(target_os = "macos")]
fn pack_420_windowed_pipeline_for_format(
    runtime: &MetalRuntime,
    fmt: PixelFormat,
) -> &ComputePipelineState {
    match fmt {
        PixelFormat::Rgb8 => &runtime.pack_420_windowed_rgb_pipeline,
        PixelFormat::Rgba8 => &runtime.pack_420_windowed_rgba_pipeline,
        _ => &runtime.pack_420_windowed_pipeline,
    }
}

#[cfg(target_os = "macos")]
fn pack_422_pipeline_for_format(
    runtime: &MetalRuntime,
    fmt: PixelFormat,
) -> Option<&ComputePipelineState> {
    match fmt {
        PixelFormat::Rgb8 => Some(&runtime.pack_422_rgb_pipeline),
        PixelFormat::Rgba8 => Some(&runtime.pack_422_rgba_pipeline),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
fn pack_422_windowed_pipeline_for_format(
    runtime: &MetalRuntime,
    fmt: PixelFormat,
) -> &ComputePipelineState {
    match fmt {
        PixelFormat::Rgb8 => &runtime.pack_422_windowed_rgb_pipeline,
        PixelFormat::Rgba8 => &runtime.pack_422_windowed_rgba_pipeline,
        _ => &runtime.pack_422_windowed_pipeline,
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
            Err(message) => Err(runtime_initialization_error(message)),
        }
    })
}

#[cfg(target_os = "macos")]
fn with_runtime_for_device<R>(
    device: &Device,
    f: impl FnOnce(&MetalRuntime) -> Result<R, Error>,
) -> Result<R, Error> {
    let runtime = MetalRuntime::new_with_device(device.clone())
        .map_err(|message| runtime_initialization_error(&message))?;
    f(&runtime)
}

#[cfg(target_os = "macos")]
fn runtime_initialization_error(message: &str) -> Error {
    if message == "Metal is unavailable on this host" {
        Error::MetalUnavailable
    } else {
        Error::MetalKernel {
            message: message.to_string(),
        }
    }
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
    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), signinum_jpeg::JpegError> {
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
    ) -> Result<(), signinum_jpeg::JpegError> {
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
    ) -> Result<(), signinum_jpeg::JpegError> {
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
fn cached_plane_stage(
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
    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), signinum_jpeg::JpegError> {
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
    ) -> Result<(), signinum_jpeg::JpegError> {
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
    ) -> Result<(), signinum_jpeg::JpegError> {
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
fn dispatch_3d_pipeline(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    dims: (u32, u32, u32),
) {
    let width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(width);
    let height = (max_threads / width).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(dims.0),
            height: u64::from(dims.1),
            depth: u64::from(dims.2),
        },
        MTLSize {
            width,
            height,
            depth: 1,
        },
    );
}

#[cfg(target_os = "macos")]
fn packed_pair_extent(value: u32) -> u32 {
    value.div_ceil(2).max(1)
}

#[cfg(target_os = "macos")]
fn dispatch_1d_pipeline(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    threads: u32,
) {
    let threadgroup_width = choose_1d_threadgroup_width(
        pipeline.thread_execution_width(),
        pipeline.max_total_threads_per_threadgroup(),
        threads,
    );
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
fn choose_1d_threadgroup_width(simd_width: u64, max_threads: u64, threads: u32) -> u64 {
    let simd_width = simd_width.max(1);
    let max_threads = max_threads.max(simd_width);
    let requested = u64::from(threads.max(1));
    let rounded = requested.div_ceil(simd_width) * simd_width;
    rounded.clamp(simd_width, max_threads.min(256).max(simd_width))
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
fn plane_mode_to_u32(mode: PlaneMode) -> u32 {
    match mode {
        PlaneMode::Gray => MODE_GRAY,
        PlaneMode::YCbCr => MODE_YCBCR,
        PlaneMode::Rgb => MODE_RGB,
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
        restart_offset_count: entropy_segment_count(
            packet.restart_interval_mcus,
            packet.restart_offsets.len(),
            packet.entropy_checkpoints.len(),
        ),
        restart_start_mcu: 0,
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
fn fast422_params(
    packet: &JpegMetalFast422PacketV1,
    fmt: PixelFormat,
) -> Result<JpegFast420Params, Error> {
    let out_format = pixel_format_to_out_format(fmt).ok_or_else(|| Error::MetalKernel {
        message: format!("unsupported JPEG Metal fast422 pixel format {fmt:?}"),
    })?;
    let out_stride = packet.dimensions.0 as usize * fmt.bytes_per_pixel();
    Ok(JpegFast420Params {
        width: packet.dimensions.0,
        height: packet.dimensions.1,
        chroma_width: packet.dimensions.0.div_ceil(2),
        chroma_height: packet.dimensions.1,
        mcus_per_row: packet.mcus_per_row,
        mcu_rows: packet.mcu_rows,
        restart_interval_mcus: packet.restart_interval_mcus,
        restart_offset_count: entropy_segment_count(
            packet.restart_interval_mcus,
            packet.restart_offsets.len(),
            packet.entropy_checkpoints.len(),
        ),
        restart_start_mcu: 0,
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
fn full_mcu_window_422(dims: (u32, u32), roi: signinum_jpeg::Rect) -> signinum_jpeg::Rect {
    let x0 = (roi.x / 16) * 16;
    let y0 = (roi.y / 8) * 8;
    let x1 = (roi.x + roi.w).div_ceil(16) * 16;
    let y1 = (roi.y + roi.h).div_ceil(8) * 8;
    signinum_jpeg::Rect {
        x: x0,
        y: y0,
        w: x1.min(dims.0).saturating_sub(x0),
        h: y1.min(dims.1).saturating_sub(y0),
    }
}

#[cfg(target_os = "macos")]
fn fast422_region_params(
    packet: &JpegMetalFast422PacketV1,
    fmt: PixelFormat,
    source_window: signinum_jpeg::Rect,
) -> Result<JpegFast420Params, Error> {
    let out_format = pixel_format_to_out_format(fmt).ok_or_else(|| Error::MetalKernel {
        message: format!("unsupported JPEG Metal fast422 pixel format {fmt:?}"),
    })?;
    let out_stride = source_window.w as usize * fmt.bytes_per_pixel();
    Ok(JpegFast420Params {
        width: source_window.w,
        height: source_window.h,
        chroma_width: source_window.w.div_ceil(2),
        chroma_height: source_window.h,
        mcus_per_row: packet.mcus_per_row,
        mcu_rows: packet.mcu_rows,
        restart_interval_mcus: packet.restart_interval_mcus,
        restart_offset_count: entropy_segment_count(
            packet.restart_interval_mcus,
            packet.restart_offsets.len(),
            packet.entropy_checkpoints.len(),
        ),
        restart_start_mcu: 0,
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
fn fast422_scaled_params(
    packet: &JpegMetalFast422PacketV1,
    scale: signinum_core::Downscale,
) -> Option<JpegFast420ScaledParams> {
    let scale_shift = match scale {
        signinum_core::Downscale::Half => 1,
        signinum_core::Downscale::Quarter => 2,
        signinum_core::Downscale::Eighth => 3,
        _ => return None,
    };
    let denom = 1u32 << scale_shift;
    let scaled_width = packet.dimensions.0.div_ceil(denom);
    let scaled_height = packet.dimensions.1.div_ceil(denom);
    Some(JpegFast420ScaledParams {
        scaled_width,
        scaled_height,
        chroma_width: scaled_width.div_ceil(2),
        chroma_height: scaled_height,
        mcus_per_row: packet.mcus_per_row,
        mcu_rows: packet.mcu_rows,
        restart_interval_mcus: packet.restart_interval_mcus,
        restart_offset_count: entropy_segment_count(
            packet.restart_interval_mcus,
            packet.restart_offsets.len(),
            packet.entropy_checkpoints.len(),
        ),
        restart_start_mcu: 0,
        entropy_len: u32::try_from(packet.entropy_bytes.len())
            .expect("JPEG Metal entropy payload fits in u32"),
        scale_shift,
        origin_x: 0,
        origin_y: 0,
    })
}

#[cfg(target_os = "macos")]
fn full_mcu_scaled_window_422(
    scaled_dims: (u32, u32),
    roi: signinum_jpeg::Rect,
    scale_shift: u32,
) -> signinum_jpeg::Rect {
    let mcu_width = 16u32 >> scale_shift;
    let mcu_height = 8u32 >> scale_shift;
    let x0 = (roi.x / mcu_width) * mcu_width;
    let y0 = (roi.y / mcu_height) * mcu_height;
    let x1 = (roi.x + roi.w).div_ceil(mcu_width) * mcu_width;
    let y1 = (roi.y + roi.h).div_ceil(mcu_height) * mcu_height;
    signinum_jpeg::Rect {
        x: x0,
        y: y0,
        w: x1.min(scaled_dims.0).saturating_sub(x0),
        h: y1.min(scaled_dims.1).saturating_sub(y0),
    }
}

#[cfg(target_os = "macos")]
fn fast422_scaled_region_params(
    packet: &JpegMetalFast422PacketV1,
    scale: signinum_core::Downscale,
    source_window: signinum_jpeg::Rect,
) -> Option<JpegFast420ScaledParams> {
    let full = fast422_scaled_params(packet, scale)?;
    Some(JpegFast420ScaledParams {
        scaled_width: source_window.w,
        scaled_height: source_window.h,
        chroma_width: source_window.w.div_ceil(2),
        chroma_height: source_window.h,
        origin_x: source_window.x,
        origin_y: source_window.y,
        ..full
    })
}

#[cfg(target_os = "macos")]
fn full_mcu_window_420(dims: (u32, u32), roi: signinum_jpeg::Rect) -> signinum_jpeg::Rect {
    let x0 = (roi.x / 16) * 16;
    let y0 = (roi.y / 16) * 16;
    let x1 = (roi.x + roi.w).div_ceil(16) * 16;
    let y1 = (roi.y + roi.h).div_ceil(16) * 16;
    signinum_jpeg::Rect {
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
    source_window: signinum_jpeg::Rect,
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
        restart_offset_count: entropy_segment_count(
            packet.restart_interval_mcus,
            packet.restart_offsets.len(),
            packet.entropy_checkpoints.len(),
        ),
        restart_start_mcu: 0,
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
    scale: signinum_core::Downscale,
) -> Option<JpegFast420ScaledParams> {
    let scale_shift = match scale {
        signinum_core::Downscale::Half => 1,
        signinum_core::Downscale::Quarter => 2,
        signinum_core::Downscale::Eighth => 3,
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
        restart_offset_count: entropy_segment_count(
            packet.restart_interval_mcus,
            packet.restart_offsets.len(),
            packet.entropy_checkpoints.len(),
        ),
        restart_start_mcu: 0,
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
    roi: signinum_jpeg::Rect,
    scale_shift: u32,
) -> signinum_jpeg::Rect {
    let mcu_size = 16u32 >> scale_shift;
    let x0 = (roi.x / mcu_size) * mcu_size;
    let y0 = (roi.y / mcu_size) * mcu_size;
    let x1 = (roi.x + roi.w).div_ceil(mcu_size) * mcu_size;
    let y1 = (roi.y + roi.h).div_ceil(mcu_size) * mcu_size;
    signinum_jpeg::Rect {
        x: x0,
        y: y0,
        w: x1.min(scaled_dims.0).saturating_sub(x0),
        h: y1.min(scaled_dims.1).saturating_sub(y0),
    }
}

#[cfg(target_os = "macos")]
fn fast420_scaled_region_params(
    packet: &JpegMetalFast420PacketV1,
    scale: signinum_core::Downscale,
    source_window: signinum_jpeg::Rect,
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
        restart_offset_count: entropy_segment_count(
            packet.restart_interval_mcus,
            packet.restart_offsets.len(),
            packet.entropy_checkpoints.len(),
        ),
        restart_start_mcu: 0,
        entropy_len: u32::try_from(packet.entropy_bytes.len())
            .expect("JPEG Metal fast444 entropy payload fits in u32"),
        origin_x: 0,
        origin_y: 0,
    }
}

#[cfg(target_os = "macos")]
fn fast444_region_params(
    packet: &JpegMetalFast444PacketV1,
    roi: signinum_jpeg::Rect,
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
    scale: signinum_core::Downscale,
) -> Option<JpegFast444ScaledParams> {
    let scale_shift = match scale {
        signinum_core::Downscale::Half => 1,
        signinum_core::Downscale::Quarter => 2,
        signinum_core::Downscale::Eighth => 3,
        _ => return None,
    };
    let denom = 1u32 << scale_shift;
    Some(JpegFast444ScaledParams {
        scaled_width: packet.dimensions.0.div_ceil(denom),
        scaled_height: packet.dimensions.1.div_ceil(denom),
        mcus_per_row: packet.mcus_per_row,
        mcu_rows: packet.mcu_rows,
        restart_interval_mcus: packet.restart_interval_mcus,
        restart_offset_count: entropy_segment_count(
            packet.restart_interval_mcus,
            packet.restart_offsets.len(),
            packet.entropy_checkpoints.len(),
        ),
        restart_start_mcu: 0,
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
    scale: signinum_core::Downscale,
    roi: signinum_jpeg::Rect,
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
    roi: signinum_jpeg::Rect,
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
fn fast422_windowed_pack_params_for_dims(
    dims: (u32, u32),
    fmt: PixelFormat,
    roi: signinum_jpeg::Rect,
) -> JpegFast420WindowedPackParams {
    let out_format = pixel_format_to_out_format(fmt)
        .ok_or_else(|| Error::MetalKernel {
            message: format!("unsupported JPEG Metal fast422 pixel format {fmt:?}"),
        })
        .expect("validated JPEG Metal fast422 pixel format");
    let out_stride = roi.w as usize * fmt.bytes_per_pixel();
    JpegFast420WindowedPackParams {
        src_width: dims.0,
        src_height: dims.1,
        chroma_width: dims.0.div_ceil(2),
        chroma_height: dims.1,
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
fn entropy_checkpoints_buffer(
    device: &Device,
    entropy_checkpoints: &[JpegMetalEntropyCheckpointV1],
) -> Result<Buffer, Error> {
    if entropy_checkpoints.is_empty() {
        return Err(Error::MetalKernel {
            message: "JPEG Metal entropy checkpoints must contain at least one entry".to_string(),
        });
    }
    let checkpoints = entropy_checkpoints
        .iter()
        .copied()
        .map(JpegEntropyCheckpointHost::from)
        .collect::<Vec<_>>();
    Ok(device.new_buffer_with_data(
        checkpoints.as_ptr().cast(),
        size_of_val(checkpoints.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    ))
}

#[cfg(target_os = "macos")]
fn u32_buffer(device: &Device, values: &[u32], label: &str) -> Result<Buffer, Error> {
    if values.is_empty() {
        return Err(Error::MetalKernel {
            message: format!("JPEG Metal {label} buffer must contain at least one entry"),
        });
    }
    Ok(device.new_buffer_with_data(
        values.as_ptr().cast(),
        size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    ))
}

#[cfg(target_os = "macos")]
fn entropy_segment_count(
    restart_interval_mcus: u32,
    restart_offsets_len: usize,
    entropy_checkpoints_len: usize,
) -> u32 {
    let len = if restart_interval_mcus == 0 {
        entropy_checkpoints_len
    } else {
        restart_offsets_len
    };
    u32::try_from(len)
        .expect("JPEG Metal entropy segment count fits in u32")
        .max(1)
}

#[cfg(target_os = "macos")]
fn restart_work_for_mcu_range(
    restart_offsets: &[u32],
    restart_interval_mcus: u32,
    total_mcus: u32,
    first_mcu: u32,
    end_mcu: u32,
) -> (u32, &[u32]) {
    if restart_interval_mcus == 0 || restart_offsets.len() <= 1 {
        return (0, restart_offsets);
    }

    let first_mcu = first_mcu.min(total_mcus);
    let end_mcu = end_mcu.min(total_mcus).max(first_mcu + 1);
    let restart_offset_count =
        u32::try_from(restart_offsets.len()).expect("JPEG Metal restart offsets fit in u32");
    let first_segment = (first_mcu / restart_interval_mcus).min(restart_offset_count - 1);
    let end_segment = end_mcu
        .div_ceil(restart_interval_mcus)
        .min(restart_offset_count)
        .max(first_segment + 1);
    (
        first_segment * restart_interval_mcus,
        &restart_offsets[first_segment as usize..end_segment as usize],
    )
}

#[cfg(target_os = "macos")]
fn mcu_range_for_rect(
    rect: signinum_jpeg::Rect,
    mcus_per_row: u32,
    mcu_rows: u32,
    mcu_width: u32,
    mcu_height: u32,
) -> (u32, u32) {
    if rect.w == 0 || rect.h == 0 || mcus_per_row == 0 || mcu_rows == 0 {
        return (0, 0);
    }

    let max_col = mcus_per_row - 1;
    let max_row = mcu_rows - 1;
    let last_x = rect.x.saturating_add(rect.w).saturating_sub(1);
    let last_y = rect.y.saturating_add(rect.h).saturating_sub(1);
    let first_col = (rect.x / mcu_width).min(max_col);
    let last_col = (last_x / mcu_width).min(max_col);
    let first_row = (rect.y / mcu_height).min(max_row);
    let last_row = (last_y / mcu_height).min(max_row);
    let first_mcu = first_row * mcus_per_row + first_col;
    let end_mcu = last_row * mcus_per_row + last_col + 1;
    (first_mcu, end_mcu)
}

#[cfg(target_os = "macos")]
fn entropy_decode_thread_count(
    restart_interval_mcus: u32,
    restart_offsets_len: usize,
    entropy_checkpoints_len: usize,
) -> u32 {
    entropy_segment_count(
        restart_interval_mcus,
        restart_offsets_len,
        entropy_checkpoints_len,
    )
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
enum BatchedFastPacket<'a> {
    Fast420(&'a JpegMetalFast420PacketV1),
    Fast422(&'a JpegMetalFast422PacketV1),
    Fast444(&'a JpegMetalFast444PacketV1, PlaneMode),
}

#[cfg(target_os = "macos")]
struct BatchedDecodeItem {
    request_index: usize,
    surface: Surface,
    status_buffer: Buffer,
    decode_threads: u32,
    _decode_resources: Vec<Buffer>,
}

#[cfg(target_os = "macos")]
#[derive(Default)]
struct BatchDeviceBufferCache {
    packet_buffers: Vec<SharedPacketDeviceBuffers>,
}

#[cfg(target_os = "macos")]
struct SharedPacketDeviceBuffers {
    entropy_ptr: usize,
    entropy_len: usize,
    checkpoints_ptr: usize,
    checkpoints_len: usize,
    entropy_buffer: Buffer,
    entropy_checkpoints_buffer: Buffer,
}

#[cfg(target_os = "macos")]
impl BatchDeviceBufferCache {
    fn packet_buffers(
        &mut self,
        runtime: &MetalRuntime,
        entropy_bytes: &[u8],
        entropy_checkpoints: &[JpegMetalEntropyCheckpointV1],
    ) -> Result<(Buffer, Buffer), Error> {
        let entropy_ptr = entropy_bytes.as_ptr() as usize;
        let entropy_len = entropy_bytes.len();
        let checkpoints_ptr = entropy_checkpoints.as_ptr() as usize;
        let checkpoints_len = entropy_checkpoints.len();
        if let Some(entry) = self.packet_buffers.iter().find(|entry| {
            entry.entropy_ptr == entropy_ptr
                && entry.entropy_len == entropy_len
                && entry.checkpoints_ptr == checkpoints_ptr
                && entry.checkpoints_len == checkpoints_len
        }) {
            return Ok((
                entry.entropy_buffer.clone(),
                entry.entropy_checkpoints_buffer.clone(),
            ));
        }

        let entropy_buffer = runtime.device.new_buffer_with_data(
            entropy_bytes.as_ptr().cast(),
            entropy_bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let entropy_checkpoints_buffer =
            entropy_checkpoints_buffer(&runtime.device, entropy_checkpoints)?;
        self.packet_buffers.push(SharedPacketDeviceBuffers {
            entropy_ptr,
            entropy_len,
            checkpoints_ptr,
            checkpoints_len,
            entropy_buffer: entropy_buffer.clone(),
            entropy_checkpoints_buffer: entropy_checkpoints_buffer.clone(),
        });
        Ok((entropy_buffer, entropy_checkpoints_buffer))
    }
}

#[cfg(target_os = "macos")]
fn request_allows_batched_packet(
    requests: &[batch::QueuedRequest],
    request: &batch::QueuedRequest,
    restart_interval_mcus: u32,
    dimensions: (u32, u32),
) -> bool {
    match request.backend {
        BackendRequest::Metal => true,
        BackendRequest::Auto => match request.op {
            batch::BatchOp::RegionScaled { .. } => false,
            _ => {
                requests.len() >= AUTO_METAL_MIN_BATCH_REQUESTS
                    && (restart_interval_mcus != 0
                        || auto_batch_work_is_large_enough(request, dimensions))
            }
        },
        BackendRequest::Cpu | BackendRequest::Cuda => false,
    }
}

#[cfg(target_os = "macos")]
fn auto_batch_work_is_large_enough(request: &batch::QueuedRequest, dimensions: (u32, u32)) -> bool {
    let dims = match request.op {
        batch::BatchOp::Full | batch::BatchOp::Scaled(_) => dimensions,
        batch::BatchOp::Region(roi) => (roi.w, roi.h),
        batch::BatchOp::RegionScaled { .. } => return false,
    };
    dims.0 >= AUTO_METAL_MIN_BATCH_EDGE && dims.1 >= AUTO_METAL_MIN_BATCH_EDGE
}

#[cfg(target_os = "macos")]
fn batched_fast_packets(
    requests: &[batch::QueuedRequest],
) -> Result<Option<Vec<BatchedFastPacket<'_>>>, Error> {
    if requests.len() < 2 {
        return Ok(None);
    }

    let mut packets = Vec::with_capacity(requests.len());
    for request in requests {
        let batchable_op = match request.op {
            batch::BatchOp::Full
            | batch::BatchOp::Region(_)
            | batch::BatchOp::Scaled(
                signinum_core::Downscale::Half
                | signinum_core::Downscale::Quarter
                | signinum_core::Downscale::Eighth,
            )
            | batch::BatchOp::RegionScaled {
                scale:
                    signinum_core::Downscale::Half
                    | signinum_core::Downscale::Quarter
                    | signinum_core::Downscale::Eighth,
                ..
            } => true,
            batch::BatchOp::Scaled(_) | batch::BatchOp::RegionScaled { .. } => false,
        };
        if !batchable_op
            || !matches!(
                request.backend,
                BackendRequest::Auto | BackendRequest::Metal
            )
            || pixel_format_to_out_format(request.fmt).is_none()
        {
            return Ok(None);
        }

        if let Some(packet) = request.fast420_packet.as_deref() {
            if !request_allows_batched_packet(
                requests,
                request,
                packet.restart_interval_mcus,
                packet.dimensions,
            ) {
                return Ok(None);
            }
            packets.push(BatchedFastPacket::Fast420(packet));
            continue;
        }

        if let Some(packet) = request.fast422_packet.as_deref() {
            if !request_allows_batched_packet(
                requests,
                request,
                packet.restart_interval_mcus,
                packet.dimensions,
            ) {
                return Ok(None);
            }
            packets.push(BatchedFastPacket::Fast422(packet));
            continue;
        }

        if let Some(packet) = request.fast444_packet.as_deref() {
            if !request_allows_batched_packet(
                requests,
                request,
                packet.restart_interval_mcus,
                packet.dimensions,
            ) {
                return Ok(None);
            }
            let decoder = CpuDecoder::new(request.input.as_ref())?;
            packets.push(BatchedFastPacket::Fast444(
                packet,
                fast444_plane_mode(&decoder),
            ));
            continue;
        }

        return Ok(None);
    }

    Ok(Some(packets))
}

#[cfg(target_os = "macos")]
fn core_rect_to_jpeg(rect: Rect) -> signinum_jpeg::Rect {
    signinum_jpeg::Rect {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: rect.h,
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_jpeg_pack_to_surface_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plane0: &Buffer,
    plane1: Option<&Buffer>,
    plane2: Option<&Buffer>,
    dims: (u32, u32),
    mode: PlaneMode,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    match (mode, fmt) {
        (PlaneMode::Gray | PlaneMode::YCbCr, PixelFormat::Gray8) => {
            return Ok(Surface::from_metal_buffer(plane0.clone(), dims, fmt));
        }
        (
            PlaneMode::Gray | PlaneMode::YCbCr | PlaneMode::Rgb,
            PixelFormat::Rgb8 | PixelFormat::Rgba8,
        )
        | (PlaneMode::Rgb, PixelFormat::Gray8) => {}
        _ => {
            return Err(Error::MetalKernel {
                message: format!("unsupported JPEG Metal pixel format {fmt:?}"),
            });
        }
    }

    let pitch_bytes = dims.0 as usize * fmt.bytes_per_pixel();
    let out_buffer = runtime.device.new_buffer(
        (pitch_bytes * dims.1 as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let params = JpegPackParams {
        width: dims.0,
        height: dims.1,
        out_stride: u32::try_from(pitch_bytes).expect("JPEG Metal output stride fits in u32"),
        alpha: u32::from(u8::MAX),
        mode: match mode {
            PlaneMode::Gray => MODE_GRAY,
            PlaneMode::YCbCr => MODE_YCBCR,
            PlaneMode::Rgb => MODE_RGB,
        },
        out_format: match fmt {
            PixelFormat::Gray8 => OUT_GRAY,
            PixelFormat::Rgb8 => OUT_RGB,
            PixelFormat::Rgba8 => OUT_RGBA,
            _ => unreachable!("validated by caller"),
        },
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.pack_pipeline);
    encoder.set_buffer(0, Some(plane0), 0);
    encoder.set_buffer(1, plane1.map(std::convert::AsRef::as_ref), 0);
    encoder.set_buffer(2, plane2.map(std::convert::AsRef::as_ref), 0);
    encoder.set_buffer(3, Some(&out_buffer), 0);
    encoder.set_bytes(
        4,
        size_of::<JpegPackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(encoder, &runtime.pack_pipeline, dims);
    encoder.end_encoding();

    Ok(Surface::from_metal_buffer(out_buffer, dims, fmt))
}

#[cfg(target_os = "macos")]
fn encode_fast420_region_batch_item(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &JpegMetalFast420PacketV1,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<BatchedDecodeItem, Error> {
    let roi = core_rect_to_jpeg(roi);
    let source_window = full_mcu_window_420(packet.dimensions, roi);
    let mut params = fast420_region_params(packet, fmt, source_window)?;
    let (first_mcu, end_mcu) =
        mcu_range_for_rect(source_window, packet.mcus_per_row, packet.mcu_rows, 16, 16);
    let total_mcus = packet.mcus_per_row * packet.mcu_rows;
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        &packet.restart_offsets,
        packet.restart_interval_mcus,
        total_mcus,
        first_mcu,
        end_mcu,
    );
    params.restart_start_mcu = restart_start_mcu;
    params.restart_offset_count = entropy_segment_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );

    let local_roi = signinum_jpeg::Rect {
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
    let cb_plane = runtime
        .device
        .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared);
    let cr_plane = runtime
        .device
        .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast420_region_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&cb_plane), 0);
    decoder_encoder.set_buffer(3, Some(&cr_plane), 0);
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast420_region_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let out_buffer = runtime.device.new_buffer(
        (pack_params.out_stride as usize * roi.h as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let pack_encoder = command_buffer.new_compute_command_encoder();
    let pack_pipeline = pack_420_windowed_pipeline_for_format(runtime, fmt);
    pack_encoder.set_compute_pipeline_state(pack_pipeline);
    pack_encoder.set_buffer(0, Some(&y_plane), 0);
    pack_encoder.set_buffer(1, Some(&cb_plane), 0);
    pack_encoder.set_buffer(2, Some(&cr_plane), 0);
    pack_encoder.set_buffer(3, Some(&out_buffer), 0);
    pack_encoder.set_bytes(
        4,
        size_of::<JpegFast420WindowedPackParams>() as u64,
        (&raw const pack_params).cast(),
    );
    dispatch_2d_pipeline(pack_encoder, pack_pipeline, (roi.w, roi.h));
    pack_encoder.end_encoding();

    Ok(BatchedDecodeItem {
        request_index,
        surface: Surface::from_metal_buffer(out_buffer, (roi.w, roi.h), fmt),
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
fn encode_fast420_scaled_batch_item(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &JpegMetalFast420PacketV1,
    fmt: PixelFormat,
    scale: signinum_core::Downscale,
) -> Result<BatchedDecodeItem, Error> {
    let Some(params) = fast420_scaled_params(packet, scale) else {
        return Err(Error::MetalKernel {
            message: format!("unsupported JPEG Metal fast420 scale {scale:?}"),
        });
    };

    let y_len = params.scaled_width as usize * params.scaled_height as usize;
    let chroma_len = params.chroma_width as usize * params.chroma_height as usize;
    let y_plane = runtime
        .device
        .new_buffer(y_len as u64, MTLResourceOptions::StorageModeShared);
    let cb_plane = runtime
        .device
        .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared);
    let cr_plane = runtime
        .device
        .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast420_scaled_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&cb_plane), 0);
    decoder_encoder.set_buffer(3, Some(&cr_plane), 0);
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast420_scaled_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let out_buffer = (fmt != PixelFormat::Gray8).then(|| {
        runtime.device.new_buffer(
            (params.scaled_width as usize * fmt.bytes_per_pixel() * params.scaled_height as usize)
                as u64,
            MTLResourceOptions::StorageModeShared,
        )
    });

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
            restart_start_mcu: params.restart_start_mcu,
            entropy_len: params.entropy_len,
            out_stride: u32::try_from(params.scaled_width as usize * fmt.bytes_per_pixel())
                .expect("JPEG Metal output stride fits in u32"),
            alpha: u32::from(u8::MAX),
            out_format: pixel_format_to_out_format(fmt).expect("validated output format"),
            origin_x: 0,
            origin_y: 0,
        };
        let pack_encoder = command_buffer.new_compute_command_encoder();
        let pack_pipeline = pack_420_pipeline_for_format(runtime, fmt);
        pack_encoder.set_compute_pipeline_state(pack_pipeline);
        pack_encoder.set_buffer(0, Some(&y_plane), 0);
        pack_encoder.set_buffer(1, Some(&cb_plane), 0);
        pack_encoder.set_buffer(2, Some(&cr_plane), 0);
        pack_encoder.set_buffer(3, Some(out_buffer), 0);
        pack_encoder.set_bytes(
            4,
            size_of::<JpegFast420Params>() as u64,
            (&raw const pack_params).cast(),
        );
        dispatch_2d_pipeline(
            pack_encoder,
            pack_pipeline,
            (params.scaled_width, params.scaled_height),
        );
        pack_encoder.end_encoding();
    }

    let surface = match out_buffer {
        Some(out_buffer) => {
            Surface::from_metal_buffer(out_buffer, (params.scaled_width, params.scaled_height), fmt)
        }
        None => Surface::from_metal_buffer(
            y_plane.clone(),
            (params.scaled_width, params.scaled_height),
            fmt,
        ),
    };

    Ok(BatchedDecodeItem {
        request_index,
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_fast420_scaled_region_batch_item(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    device_buffer_cache: &mut BatchDeviceBufferCache,
    request_index: usize,
    packet: &JpegMetalFast420PacketV1,
    fmt: PixelFormat,
    roi: Rect,
    scale: signinum_core::Downscale,
) -> Result<BatchedDecodeItem, Error> {
    let Some(full_params) = fast420_scaled_params(packet, scale) else {
        return Err(Error::MetalKernel {
            message: format!("unsupported JPEG Metal fast420 scale {scale:?}"),
        });
    };
    let scaled_roi = scaled_rect_covering(roi, scale);
    let scaled_roi = signinum_jpeg::Rect {
        x: scaled_roi.x,
        y: scaled_roi.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let source_window = full_mcu_scaled_window_420(
        (full_params.scaled_width, full_params.scaled_height),
        scaled_roi,
        full_params.scale_shift,
    );
    let Some(mut decode_params) = fast420_scaled_region_params(packet, scale, source_window) else {
        return Err(Error::MetalKernel {
            message: format!("unsupported JPEG Metal fast420 scaled region {scale:?}"),
        });
    };
    let mcu_size = 16u32 >> decode_params.scale_shift;
    let (first_mcu, end_mcu) = mcu_range_for_rect(
        source_window,
        packet.mcus_per_row,
        packet.mcu_rows,
        mcu_size,
        mcu_size,
    );
    let total_mcus = packet.mcus_per_row * packet.mcu_rows;
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        &packet.restart_offsets,
        packet.restart_interval_mcus,
        total_mcus,
        first_mcu,
        end_mcu,
    );
    decode_params.restart_start_mcu = restart_start_mcu;
    decode_params.restart_offset_count = entropy_segment_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let local_roi = signinum_jpeg::Rect {
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
    let cb_plane = runtime
        .device
        .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared);
    let cr_plane = runtime
        .device
        .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let (entropy_buffer, entropy_checkpoints_buffer) = device_buffer_cache.packet_buffers(
        runtime,
        &packet.entropy_bytes,
        &packet.entropy_checkpoints,
    )?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast420_scaled_region_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&cb_plane), 0);
    decoder_encoder.set_buffer(3, Some(&cr_plane), 0);
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast420_scaled_region_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let out_buffer = runtime.device.new_buffer(
        (pack_params.out_stride as usize * scaled_roi.h as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let pack_encoder = command_buffer.new_compute_command_encoder();
    let pack_pipeline = pack_420_windowed_pipeline_for_format(runtime, fmt);
    pack_encoder.set_compute_pipeline_state(pack_pipeline);
    pack_encoder.set_buffer(0, Some(&y_plane), 0);
    pack_encoder.set_buffer(1, Some(&cb_plane), 0);
    pack_encoder.set_buffer(2, Some(&cr_plane), 0);
    pack_encoder.set_buffer(3, Some(&out_buffer), 0);
    pack_encoder.set_bytes(
        4,
        size_of::<JpegFast420WindowedPackParams>() as u64,
        (&raw const pack_params).cast(),
    );
    dispatch_2d_pipeline(pack_encoder, pack_pipeline, (scaled_roi.w, scaled_roi.h));
    pack_encoder.end_encoding();

    Ok(BatchedDecodeItem {
        request_index,
        surface: Surface::from_metal_buffer(out_buffer, (scaled_roi.w, scaled_roi.h), fmt),
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
fn encode_fast420_batch_item(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &JpegMetalFast420PacketV1,
    fmt: PixelFormat,
) -> Result<BatchedDecodeItem, Error> {
    let params = fast420_params(packet, fmt)?;
    let y_len = params.width as usize * params.height as usize;
    let chroma_len = params.chroma_width as usize * params.chroma_height as usize;
    let y_plane = runtime
        .device
        .new_buffer(y_len as u64, MTLResourceOptions::StorageModeShared);
    let cb_plane = runtime
        .device
        .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared);
    let cr_plane = runtime
        .device
        .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast420_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&cb_plane), 0);
    decoder_encoder.set_buffer(3, Some(&cr_plane), 0);
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast420_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let surface = if fmt == PixelFormat::Gray8 {
        Surface::from_metal_buffer(y_plane.clone(), packet.dimensions, fmt)
    } else {
        let out_buffer = runtime.device.new_buffer(
            (params.out_stride as usize * params.height as usize) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let pack_encoder = command_buffer.new_compute_command_encoder();
        let pack_pipeline = pack_420_pipeline_for_format(runtime, fmt);
        pack_encoder.set_compute_pipeline_state(pack_pipeline);
        pack_encoder.set_buffer(0, Some(&y_plane), 0);
        pack_encoder.set_buffer(1, Some(&cb_plane), 0);
        pack_encoder.set_buffer(2, Some(&cr_plane), 0);
        pack_encoder.set_buffer(3, Some(&out_buffer), 0);
        pack_encoder.set_bytes(
            4,
            size_of::<JpegFast420Params>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(pack_encoder, pack_pipeline, packet.dimensions);
        pack_encoder.end_encoding();
        Surface::from_metal_buffer(out_buffer, packet.dimensions, fmt)
    };

    Ok(BatchedDecodeItem {
        request_index,
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
fn encode_fast422_batch_item(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &JpegMetalFast422PacketV1,
    fmt: PixelFormat,
) -> Result<BatchedDecodeItem, Error> {
    let params = fast422_params(packet, fmt)?;
    let y_len = params.width as usize * params.height as usize;
    let chroma_len = params.chroma_width as usize * params.chroma_height as usize;
    let y_plane = runtime
        .device
        .new_buffer(y_len as u64, MTLResourceOptions::StorageModeShared);
    let cb_plane = runtime
        .device
        .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared);
    let cr_plane = runtime
        .device
        .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast422_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&cb_plane), 0);
    decoder_encoder.set_buffer(3, Some(&cr_plane), 0);
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast422_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let surface = if fmt == PixelFormat::Gray8 {
        Surface::from_metal_buffer(y_plane.clone(), packet.dimensions, fmt)
    } else {
        let Some(pack_pipeline) = pack_422_pipeline_for_format(runtime, fmt) else {
            return Err(Error::MetalKernel {
                message: format!("unsupported JPEG Metal fast422 pixel format {fmt:?}"),
            });
        };
        let out_buffer = runtime.device.new_buffer(
            (params.out_stride as usize * params.height as usize) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let pack_encoder = command_buffer.new_compute_command_encoder();
        pack_encoder.set_compute_pipeline_state(pack_pipeline);
        pack_encoder.set_buffer(0, Some(&y_plane), 0);
        pack_encoder.set_buffer(1, Some(&cb_plane), 0);
        pack_encoder.set_buffer(2, Some(&cr_plane), 0);
        pack_encoder.set_buffer(3, Some(&out_buffer), 0);
        pack_encoder.set_bytes(
            4,
            size_of::<JpegFast420Params>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(pack_encoder, pack_pipeline, packet.dimensions);
        pack_encoder.end_encoding();
        Surface::from_metal_buffer(out_buffer, packet.dimensions, fmt)
    };

    Ok(BatchedDecodeItem {
        request_index,
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
fn encode_fast422_region_batch_item(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &JpegMetalFast422PacketV1,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<BatchedDecodeItem, Error> {
    let roi = core_rect_to_jpeg(roi);
    let source_window = full_mcu_window_422(packet.dimensions, roi);
    let mut params = fast422_region_params(packet, fmt, source_window)?;
    let (first_mcu, end_mcu) =
        mcu_range_for_rect(source_window, packet.mcus_per_row, packet.mcu_rows, 16, 8);
    let total_mcus = packet.mcus_per_row * packet.mcu_rows;
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        &packet.restart_offsets,
        packet.restart_interval_mcus,
        total_mcus,
        first_mcu,
        end_mcu,
    );
    params.restart_start_mcu = restart_start_mcu;
    params.restart_offset_count = entropy_segment_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );

    let local_roi = signinum_jpeg::Rect {
        x: roi.x - source_window.x,
        y: roi.y - source_window.y,
        w: roi.w,
        h: roi.h,
    };
    let pack_params =
        fast422_windowed_pack_params_for_dims((source_window.w, source_window.h), fmt, local_roi);
    let y_len = source_window.w as usize * source_window.h as usize;
    let chroma_len = source_window.w.div_ceil(2) as usize * source_window.h as usize;
    let y_plane = runtime
        .device
        .new_buffer(y_len as u64, MTLResourceOptions::StorageModeShared);
    let cb_plane = runtime
        .device
        .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared);
    let cr_plane = runtime
        .device
        .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast422_region_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&cb_plane), 0);
    decoder_encoder.set_buffer(3, Some(&cr_plane), 0);
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast422_region_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let out_buffer = runtime.device.new_buffer(
        (pack_params.out_stride as usize * roi.h as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let pack_encoder = command_buffer.new_compute_command_encoder();
    let pack_pipeline = pack_422_windowed_pipeline_for_format(runtime, fmt);
    pack_encoder.set_compute_pipeline_state(pack_pipeline);
    pack_encoder.set_buffer(0, Some(&y_plane), 0);
    pack_encoder.set_buffer(1, Some(&cb_plane), 0);
    pack_encoder.set_buffer(2, Some(&cr_plane), 0);
    pack_encoder.set_buffer(3, Some(&out_buffer), 0);
    pack_encoder.set_bytes(
        4,
        size_of::<JpegFast420WindowedPackParams>() as u64,
        (&raw const pack_params).cast(),
    );
    dispatch_2d_pipeline(pack_encoder, pack_pipeline, (roi.w, roi.h));
    pack_encoder.end_encoding();

    Ok(BatchedDecodeItem {
        request_index,
        surface: Surface::from_metal_buffer(out_buffer, (roi.w, roi.h), fmt),
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
fn encode_fast422_scaled_batch_item(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &JpegMetalFast422PacketV1,
    fmt: PixelFormat,
    scale: signinum_core::Downscale,
) -> Result<BatchedDecodeItem, Error> {
    let Some(params) = fast422_scaled_params(packet, scale) else {
        return Err(Error::MetalKernel {
            message: format!("unsupported JPEG Metal fast422 scale {scale:?}"),
        });
    };

    let y_len = params.scaled_width as usize * params.scaled_height as usize;
    let chroma_len = params.chroma_width as usize * params.chroma_height as usize;
    let y_plane = runtime
        .device
        .new_buffer(y_len as u64, MTLResourceOptions::StorageModeShared);
    let cb_plane = runtime
        .device
        .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared);
    let cr_plane = runtime
        .device
        .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast422_scaled_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&cb_plane), 0);
    decoder_encoder.set_buffer(3, Some(&cr_plane), 0);
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast422_scaled_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let out_buffer = (fmt != PixelFormat::Gray8).then(|| {
        runtime.device.new_buffer(
            (params.scaled_width as usize * fmt.bytes_per_pixel() * params.scaled_height as usize)
                as u64,
            MTLResourceOptions::StorageModeShared,
        )
    });

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
            restart_start_mcu: params.restart_start_mcu,
            entropy_len: params.entropy_len,
            out_stride: u32::try_from(params.scaled_width as usize * fmt.bytes_per_pixel())
                .expect("JPEG Metal output stride fits in u32"),
            alpha: u32::from(u8::MAX),
            out_format: pixel_format_to_out_format(fmt).expect("validated output format"),
            origin_x: 0,
            origin_y: 0,
        };
        let pack_encoder = command_buffer.new_compute_command_encoder();
        let Some(pack_pipeline) = pack_422_pipeline_for_format(runtime, fmt) else {
            return Err(Error::MetalKernel {
                message: format!("unsupported JPEG Metal fast422 pixel format {fmt:?}"),
            });
        };
        pack_encoder.set_compute_pipeline_state(pack_pipeline);
        pack_encoder.set_buffer(0, Some(&y_plane), 0);
        pack_encoder.set_buffer(1, Some(&cb_plane), 0);
        pack_encoder.set_buffer(2, Some(&cr_plane), 0);
        pack_encoder.set_buffer(3, Some(out_buffer), 0);
        pack_encoder.set_bytes(
            4,
            size_of::<JpegFast420Params>() as u64,
            (&raw const pack_params).cast(),
        );
        dispatch_2d_pipeline(
            pack_encoder,
            pack_pipeline,
            (params.scaled_width, params.scaled_height),
        );
        pack_encoder.end_encoding();
    }

    let surface = match out_buffer {
        Some(out_buffer) => {
            Surface::from_metal_buffer(out_buffer, (params.scaled_width, params.scaled_height), fmt)
        }
        None => Surface::from_metal_buffer(
            y_plane.clone(),
            (params.scaled_width, params.scaled_height),
            fmt,
        ),
    };

    Ok(BatchedDecodeItem {
        request_index,
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_fast422_scaled_region_batch_item(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    device_buffer_cache: &mut BatchDeviceBufferCache,
    request_index: usize,
    packet: &JpegMetalFast422PacketV1,
    fmt: PixelFormat,
    roi: Rect,
    scale: signinum_core::Downscale,
) -> Result<BatchedDecodeItem, Error> {
    let Some(full_params) = fast422_scaled_params(packet, scale) else {
        return Err(Error::MetalKernel {
            message: format!("unsupported JPEG Metal fast422 scale {scale:?}"),
        });
    };
    let scaled_roi = scaled_rect_covering(roi, scale);
    let scaled_roi = signinum_jpeg::Rect {
        x: scaled_roi.x,
        y: scaled_roi.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let source_window = full_mcu_scaled_window_422(
        (full_params.scaled_width, full_params.scaled_height),
        scaled_roi,
        full_params.scale_shift,
    );
    let Some(mut decode_params) = fast422_scaled_region_params(packet, scale, source_window) else {
        return Err(Error::MetalKernel {
            message: format!("unsupported JPEG Metal fast422 scaled region {scale:?}"),
        });
    };
    let mcu_width = 16u32 >> decode_params.scale_shift;
    let mcu_height = 8u32 >> decode_params.scale_shift;
    let (first_mcu, end_mcu) = mcu_range_for_rect(
        source_window,
        packet.mcus_per_row,
        packet.mcu_rows,
        mcu_width,
        mcu_height,
    );
    let total_mcus = packet.mcus_per_row * packet.mcu_rows;
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        &packet.restart_offsets,
        packet.restart_interval_mcus,
        total_mcus,
        first_mcu,
        end_mcu,
    );
    decode_params.restart_start_mcu = restart_start_mcu;
    decode_params.restart_offset_count = entropy_segment_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let local_roi = signinum_jpeg::Rect {
        x: scaled_roi.x - source_window.x,
        y: scaled_roi.y - source_window.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let pack_params =
        fast422_windowed_pack_params_for_dims((source_window.w, source_window.h), fmt, local_roi);
    let y_len = source_window.w as usize * source_window.h as usize;
    let chroma_len = source_window.w.div_ceil(2) as usize * source_window.h as usize;
    let y_plane = runtime
        .device
        .new_buffer(y_len as u64, MTLResourceOptions::StorageModeShared);
    let cb_plane = runtime
        .device
        .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared);
    let cr_plane = runtime
        .device
        .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let (entropy_buffer, entropy_checkpoints_buffer) = device_buffer_cache.packet_buffers(
        runtime,
        &packet.entropy_bytes,
        &packet.entropy_checkpoints,
    )?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast422_scaled_region_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&cb_plane), 0);
    decoder_encoder.set_buffer(3, Some(&cr_plane), 0);
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast422_scaled_region_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let out_buffer = runtime.device.new_buffer(
        (pack_params.out_stride as usize * scaled_roi.h as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let pack_encoder = command_buffer.new_compute_command_encoder();
    let pack_pipeline = pack_422_windowed_pipeline_for_format(runtime, fmt);
    pack_encoder.set_compute_pipeline_state(pack_pipeline);
    pack_encoder.set_buffer(0, Some(&y_plane), 0);
    pack_encoder.set_buffer(1, Some(&cb_plane), 0);
    pack_encoder.set_buffer(2, Some(&cr_plane), 0);
    pack_encoder.set_buffer(3, Some(&out_buffer), 0);
    pack_encoder.set_bytes(
        4,
        size_of::<JpegFast420WindowedPackParams>() as u64,
        (&raw const pack_params).cast(),
    );
    dispatch_2d_pipeline(pack_encoder, pack_pipeline, (scaled_roi.w, scaled_roi.h));
    pack_encoder.end_encoding();

    Ok(BatchedDecodeItem {
        request_index,
        surface: Surface::from_metal_buffer(out_buffer, (scaled_roi.w, scaled_roi.h), fmt),
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
fn encode_fast444_region_batch_item(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &JpegMetalFast444PacketV1,
    mode: PlaneMode,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<BatchedDecodeItem, Error> {
    let roi = core_rect_to_jpeg(roi);
    let mut params = fast444_region_params(packet, roi);
    let (first_mcu, end_mcu) = mcu_range_for_rect(roi, packet.mcus_per_row, packet.mcu_rows, 8, 8);
    let total_mcus = packet.mcus_per_row * packet.mcu_rows;
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        &packet.restart_offsets,
        packet.restart_interval_mcus,
        total_mcus,
        first_mcu,
        end_mcu,
    );
    params.restart_start_mcu = restart_start_mcu;
    params.restart_offset_count = entropy_segment_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );

    let plane_len = params.width as usize * params.height as usize;
    let y_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let cb_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let cr_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_region_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&cb_plane), 0);
    decoder_encoder.set_buffer(3, Some(&cr_plane), 0);
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_region_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let surface = encode_jpeg_pack_to_surface_in_command_buffer(
        runtime,
        command_buffer,
        &y_plane,
        Some(&cb_plane),
        Some(&cr_plane),
        (roi.w, roi.h),
        mode,
        fmt,
    )?;

    Ok(BatchedDecodeItem {
        request_index,
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
fn encode_fast444_scaled_batch_item(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &JpegMetalFast444PacketV1,
    mode: PlaneMode,
    fmt: PixelFormat,
    scale: signinum_core::Downscale,
) -> Result<BatchedDecodeItem, Error> {
    let Some(params) = fast444_scaled_params(packet, scale) else {
        return Err(Error::MetalKernel {
            message: format!("unsupported JPEG Metal fast444 scale {scale:?}"),
        });
    };

    let plane_len = params.scaled_width as usize * params.scaled_height as usize;
    let y_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let cb_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let cr_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_scaled_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&cb_plane), 0);
    decoder_encoder.set_buffer(3, Some(&cr_plane), 0);
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_scaled_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let surface = encode_jpeg_pack_to_surface_in_command_buffer(
        runtime,
        command_buffer,
        &y_plane,
        Some(&cb_plane),
        Some(&cr_plane),
        (params.scaled_width, params.scaled_height),
        mode,
        fmt,
    )?;

    Ok(BatchedDecodeItem {
        request_index,
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_fast444_scaled_region_batch_item(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    device_buffer_cache: &mut BatchDeviceBufferCache,
    request_index: usize,
    packet: &JpegMetalFast444PacketV1,
    mode: PlaneMode,
    fmt: PixelFormat,
    roi: Rect,
    scale: signinum_core::Downscale,
) -> Result<BatchedDecodeItem, Error> {
    let scaled_roi = scaled_rect_covering(roi, scale);
    let scaled_roi = signinum_jpeg::Rect {
        x: scaled_roi.x,
        y: scaled_roi.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let Some(mut params) = fast444_scaled_region_params(packet, scale, scaled_roi) else {
        return Err(Error::MetalKernel {
            message: format!("unsupported JPEG Metal fast444 scaled region {scale:?}"),
        });
    };
    let mcu_size = 8u32 >> params.scale_shift;
    let (first_mcu, end_mcu) = mcu_range_for_rect(
        scaled_roi,
        packet.mcus_per_row,
        packet.mcu_rows,
        mcu_size,
        mcu_size,
    );
    let total_mcus = packet.mcus_per_row * packet.mcu_rows;
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        &packet.restart_offsets,
        packet.restart_interval_mcus,
        total_mcus,
        first_mcu,
        end_mcu,
    );
    params.restart_start_mcu = restart_start_mcu;
    params.restart_offset_count = entropy_segment_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );

    let plane_len = params.scaled_width as usize * params.scaled_height as usize;
    let y_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let cb_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let cr_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let (entropy_buffer, entropy_checkpoints_buffer) = device_buffer_cache.packet_buffers(
        runtime,
        &packet.entropy_bytes,
        &packet.entropy_checkpoints,
    )?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_scaled_region_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&cb_plane), 0);
    decoder_encoder.set_buffer(3, Some(&cr_plane), 0);
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_scaled_region_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let surface = encode_jpeg_pack_to_surface_in_command_buffer(
        runtime,
        command_buffer,
        &y_plane,
        Some(&cb_plane),
        Some(&cr_plane),
        (scaled_roi.w, scaled_roi.h),
        mode,
        fmt,
    )?;

    Ok(BatchedDecodeItem {
        request_index,
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
fn encode_fast444_batch_item(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &JpegMetalFast444PacketV1,
    mode: PlaneMode,
    fmt: PixelFormat,
) -> Result<BatchedDecodeItem, Error> {
    let params = fast444_params(packet);
    let plane_len = params.width as usize * params.height as usize;
    let y_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let cb_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let cr_plane = runtime
        .device
        .new_buffer(plane_len as u64, MTLResourceOptions::StorageModeShared);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&cb_plane), 0);
    decoder_encoder.set_buffer(3, Some(&cr_plane), 0);
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let surface = encode_jpeg_pack_to_surface_in_command_buffer(
        runtime,
        command_buffer,
        &y_plane,
        Some(&cb_plane),
        Some(&cr_plane),
        packet.dimensions,
        mode,
        fmt,
    )?;

    Ok(BatchedDecodeItem {
        request_index,
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
fn fast420_packets_share_full_rgb_batch_shape(
    first: &JpegMetalFast420PacketV1,
    packet: &JpegMetalFast420PacketV1,
    segment_count: usize,
) -> bool {
    packet.restart_interval_mcus == 0
        && packet.dimensions == first.dimensions
        && packet.mcus_per_row == first.mcus_per_row
        && packet.mcu_rows == first.mcu_rows
        && packet.entropy_checkpoints.len() == segment_count
        && packet.y_quant == first.y_quant
        && packet.cb_quant == first.cb_quant
        && packet.cr_quant == first.cr_quant
        && packet.y_dc_table == first.y_dc_table
        && packet.y_ac_table == first.y_ac_table
        && packet.cb_dc_table == first.cb_dc_table
        && packet.cb_ac_table == first.cb_ac_table
        && packet.cr_dc_table == first.cr_dc_table
        && packet.cr_ac_table == first.cr_ac_table
}

#[cfg(target_os = "macos")]
fn checked_u32(value: usize, label: &str) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| Error::MetalKernel {
        message: format!("JPEG Metal {label} does not fit in u32"),
    })
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fast420BatchDecodeMode {
    Fused,
    SplitCoeffIdct,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Default)]
struct Fast420BatchTiming {
    accepted: Duration,
    entropy_concat: Duration,
    buffer_alloc: Duration,
    encode_decode: Duration,
    wait_decode: Duration,
    encode_pack: Duration,
    wait_pack: Duration,
    total: Duration,
}

#[cfg(target_os = "macos")]
impl Fast420BatchTiming {
    fn micros(duration: Duration) -> u128 {
        duration.as_micros()
    }

    fn log(self, label: &str, tile_count: usize, dimensions: (u32, u32), segment_count: usize) {
        eprintln!(
            concat!(
                "JPEG Metal fast420 batch timing ",
                "mode={} tiles={} dims={}x{} segments={} ",
                "accepted_us={} entropy_concat_us={} buffer_alloc_us={} ",
                "encode_decode_us={} wait_decode_us={} ",
                "encode_pack_us={} wait_pack_us={} total_us={}"
            ),
            label,
            tile_count,
            dimensions.0,
            dimensions.1,
            segment_count,
            Self::micros(self.accepted),
            Self::micros(self.entropy_concat),
            Self::micros(self.buffer_alloc),
            Self::micros(self.encode_decode),
            Self::micros(self.wait_decode),
            Self::micros(self.encode_pack),
            Self::micros(self.wait_pack),
            Self::micros(self.total)
        );
    }
}

#[cfg(target_os = "macos")]
fn fast420_batch_decode_mode() -> Fast420BatchDecodeMode {
    if split_fast420_batch_enabled() {
        Fast420BatchDecodeMode::SplitCoeffIdct
    } else {
        Fast420BatchDecodeMode::Fused
    }
}

#[cfg(target_os = "macos")]
struct BatchEntropyBuffers {
    payload: Buffer,
    offsets: Buffer,
    lens: Buffer,
    checkpoints: Buffer,
}

#[cfg(target_os = "macos")]
fn batch_entropy_buffers<'a>(
    runtime: &MetalRuntime,
    entropy_bytes_iter: impl Iterator<Item = &'a [u8]> + Clone,
    entropy_checkpoints_iter: impl Iterator<Item = &'a [JpegMetalEntropyCheckpointV1]> + Clone,
    tile_count: usize,
    segment_count: usize,
) -> Result<Option<BatchEntropyBuffers>, Error> {
    let total_entropy_len = entropy_bytes_iter
        .clone()
        .map(<[u8]>::len)
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal region scaled batch entropy length overflowed".to_string(),
        })?;
    if total_entropy_len == 0 {
        return Ok(None);
    }

    let mut entropy_bytes = Vec::with_capacity(total_entropy_len);
    let mut entropy_offsets = Vec::with_capacity(tile_count);
    let mut entropy_lens = Vec::with_capacity(tile_count);
    let mut entropy_checkpoints = Vec::with_capacity(tile_count * segment_count);
    for (bytes, checkpoints) in entropy_bytes_iter.zip(entropy_checkpoints_iter) {
        entropy_offsets.push(checked_u32(
            entropy_bytes.len(),
            "region scaled batch entropy offset",
        )?);
        entropy_lens.push(checked_u32(
            bytes.len(),
            "region scaled batch entropy length",
        )?);
        entropy_bytes.extend_from_slice(bytes);
        entropy_checkpoints.extend(checkpoints.iter().copied());
    }

    let entropy_buffer = runtime.device.new_buffer_with_data(
        entropy_bytes.as_ptr().cast(),
        entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    Ok(Some(BatchEntropyBuffers {
        payload: entropy_buffer,
        offsets: u32_buffer(
            &runtime.device,
            &entropy_offsets,
            "region scaled batch entropy offsets",
        )?,
        lens: u32_buffer(
            &runtime.device,
            &entropy_lens,
            "region scaled batch entropy lengths",
        )?,
        checkpoints: entropy_checkpoints_buffer(&runtime.device, &entropy_checkpoints)?,
    }))
}

#[cfg(target_os = "macos")]
fn region_scaled_batch_error_results(
    requests: &[batch::QueuedRequest],
    status_buffer: &Buffer,
    total_decode_threads: u32,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let Some(status) = first_decode_error_status(status_buffer, total_decode_threads) else {
        return Ok(None);
    };
    let mut results = Vec::with_capacity(requests.len());
    for request in requests {
        let decoder = CpuDecoder::new(request.input.as_ref())?;
        results.push(Err(decode_error_from_cpu(&decoder, request.fmt, status)));
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, PartialEq, Eq)]
struct RegionScaledBatchPlan {
    decode_params: JpegFastRegionScaledBatchParams,
    pack_params: JpegWindowedPackBatchParams,
    y_len: usize,
    chroma_len: usize,
    out_tile_len: usize,
    out_dims: (u32, u32),
}

#[cfg(target_os = "macos")]
fn fast420_region_scaled_batch_plan(
    packet: &JpegMetalFast420PacketV1,
    roi: Rect,
    scale: signinum_core::Downscale,
    tile_count: u32,
    segment_count: u32,
) -> Option<RegionScaledBatchPlan> {
    let full_params = fast420_scaled_params(packet, scale)?;
    let scaled_roi = scaled_rect_covering(roi, scale);
    let scaled_roi = signinum_jpeg::Rect {
        x: scaled_roi.x,
        y: scaled_roi.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let source_window = full_mcu_scaled_window_420(
        (full_params.scaled_width, full_params.scaled_height),
        scaled_roi,
        full_params.scale_shift,
    );
    let decode_params = fast420_scaled_region_params(packet, scale, source_window)?;
    let local_roi = signinum_jpeg::Rect {
        x: scaled_roi.x - source_window.x,
        y: scaled_roi.y - source_window.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let pack_params = fast420_windowed_pack_params_for_dims(
        (source_window.w, source_window.h),
        PixelFormat::Rgb8,
        local_roi,
    );
    let out_stride = scaled_roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    Some(RegionScaledBatchPlan {
        decode_params: JpegFastRegionScaledBatchParams {
            scaled_width: decode_params.scaled_width,
            scaled_height: decode_params.scaled_height,
            chroma_width: decode_params.chroma_width,
            chroma_height: decode_params.chroma_height,
            mcus_per_row: decode_params.mcus_per_row,
            mcu_rows: decode_params.mcu_rows,
            segment_count,
            tile_count,
            scale_shift: decode_params.scale_shift,
            origin_x: decode_params.origin_x,
            origin_y: decode_params.origin_y,
        },
        pack_params: JpegWindowedPackBatchParams {
            src_width: pack_params.src_width,
            src_height: pack_params.src_height,
            chroma_width: pack_params.chroma_width,
            chroma_height: pack_params.chroma_height,
            src_x: pack_params.src_x,
            src_y: pack_params.src_y,
            width: pack_params.width,
            height: pack_params.height,
            tile_count,
            out_stride: u32::try_from(out_stride).expect("JPEG Metal output stride fits in u32"),
            alpha: u32::from(u8::MAX),
            mode: MODE_YCBCR,
            out_format: OUT_RGB,
        },
        y_len: source_window.w as usize * source_window.h as usize,
        chroma_len: source_window.w.div_ceil(2) as usize * source_window.h.div_ceil(2) as usize,
        out_tile_len: out_stride * scaled_roi.h as usize,
        out_dims: (scaled_roi.w, scaled_roi.h),
    })
}

#[cfg(target_os = "macos")]
fn fast422_region_scaled_batch_plan(
    packet: &JpegMetalFast422PacketV1,
    roi: Rect,
    scale: signinum_core::Downscale,
    tile_count: u32,
    segment_count: u32,
) -> Option<RegionScaledBatchPlan> {
    let full_params = fast422_scaled_params(packet, scale)?;
    let scaled_roi = scaled_rect_covering(roi, scale);
    let scaled_roi = signinum_jpeg::Rect {
        x: scaled_roi.x,
        y: scaled_roi.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let source_window = full_mcu_scaled_window_422(
        (full_params.scaled_width, full_params.scaled_height),
        scaled_roi,
        full_params.scale_shift,
    );
    let decode_params = fast422_scaled_region_params(packet, scale, source_window)?;
    let local_roi = signinum_jpeg::Rect {
        x: scaled_roi.x - source_window.x,
        y: scaled_roi.y - source_window.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let pack_params = fast422_windowed_pack_params_for_dims(
        (source_window.w, source_window.h),
        PixelFormat::Rgb8,
        local_roi,
    );
    let out_stride = scaled_roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    Some(RegionScaledBatchPlan {
        decode_params: JpegFastRegionScaledBatchParams {
            scaled_width: decode_params.scaled_width,
            scaled_height: decode_params.scaled_height,
            chroma_width: decode_params.chroma_width,
            chroma_height: decode_params.chroma_height,
            mcus_per_row: decode_params.mcus_per_row,
            mcu_rows: decode_params.mcu_rows,
            segment_count,
            tile_count,
            scale_shift: decode_params.scale_shift,
            origin_x: decode_params.origin_x,
            origin_y: decode_params.origin_y,
        },
        pack_params: JpegWindowedPackBatchParams {
            src_width: pack_params.src_width,
            src_height: pack_params.src_height,
            chroma_width: pack_params.chroma_width,
            chroma_height: pack_params.chroma_height,
            src_x: pack_params.src_x,
            src_y: pack_params.src_y,
            width: pack_params.width,
            height: pack_params.height,
            tile_count,
            out_stride: u32::try_from(out_stride).expect("JPEG Metal output stride fits in u32"),
            alpha: u32::from(u8::MAX),
            mode: MODE_YCBCR,
            out_format: OUT_RGB,
        },
        y_len: source_window.w as usize * source_window.h as usize,
        chroma_len: source_window.w.div_ceil(2) as usize * source_window.h as usize,
        out_tile_len: out_stride * scaled_roi.h as usize,
        out_dims: (scaled_roi.w, scaled_roi.h),
    })
}

#[cfg(target_os = "macos")]
fn try_decode_fast420_full_rgb_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast420_full_rgb_batch_to_surfaces_with_mode(
        runtime,
        requests,
        packets,
        fast420_batch_decode_mode(),
    )
}

#[cfg(target_os = "macos")]
fn try_decode_fast420_full_rgb_batch_to_surfaces_with_mode(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    decode_mode: Fast420BatchDecodeMode,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let timing_enabled =
        decode_mode == Fast420BatchDecodeMode::Fused && fast420_batch_timing_enabled();
    let timing_total_start = timing_enabled.then(Instant::now);
    let mut timing = Fast420BatchTiming::default();

    if requests.len() < 2
        || requests
            .iter()
            .any(|request| request.op != batch::BatchOp::Full || request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }

    let mut fast420_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let BatchedFastPacket::Fast420(packet) = packet else {
            return Ok(None);
        };
        fast420_packets.push(*packet);
    }

    let Some(first) = fast420_packets.first().copied() else {
        return Ok(None);
    };
    if first.restart_interval_mcus != 0 || first.entropy_checkpoints.is_empty() {
        return Ok(None);
    }

    let segment_count = first.entropy_checkpoints.len();
    if !fast420_packets
        .iter()
        .all(|packet| fast420_packets_share_full_rgb_batch_shape(first, packet, segment_count))
    {
        return Ok(None);
    }

    let tile_count = fast420_packets.len();
    let tile_count_u32 = checked_u32(tile_count, "batch tile count")?;
    let segment_count_u32 = checked_u32(segment_count, "batch segment count")?;
    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal batch decode thread count overflowed".to_string(),
            })?,
        "batch decode thread count",
    )?;

    let width = first.dimensions.0;
    let height = first.dimensions.1;
    let chroma_width = width.div_ceil(2);
    let chroma_height = height.div_ceil(2);
    let y_len = width as usize * height as usize;
    let chroma_len = chroma_width as usize * chroma_height as usize;
    let out_stride = width as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let out_tile_len = out_stride * height as usize;
    let total_mcus = first.mcus_per_row as usize * first.mcu_rows as usize;
    let blocks_per_tile = total_mcus
        .checked_mul(6)
        .ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal fast420 batch block count overflowed".to_string(),
        })?;
    let total_blocks =
        blocks_per_tile
            .checked_mul(tile_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal fast420 batch total block count overflowed".to_string(),
            })?;
    let _total_blocks_u32 = checked_u32(total_blocks, "fast420 batch block count")?;

    let params = JpegFast420BatchParams {
        width,
        height,
        chroma_width,
        chroma_height,
        mcus_per_row: first.mcus_per_row,
        mcu_rows: first.mcu_rows,
        segment_count: segment_count_u32,
        tile_count: tile_count_u32,
        out_stride: checked_u32(out_stride, "batch output stride")?,
        alpha: u32::from(u8::MAX),
    };
    if timing_enabled {
        timing.accepted = timing_total_start
            .expect("timing start is set when timing is enabled")
            .elapsed();
    }

    let timing_entropy_start = timing_enabled.then(Instant::now);
    let total_entropy_len = fast420_packets
        .iter()
        .map(|packet| packet.entropy_bytes.len())
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal batch entropy length overflowed".to_string(),
        })?;
    if total_entropy_len == 0 {
        return Ok(None);
    }

    let mut entropy_bytes = Vec::with_capacity(total_entropy_len);
    let mut entropy_offsets = Vec::with_capacity(tile_count);
    let mut entropy_lens = Vec::with_capacity(tile_count);
    let mut entropy_checkpoints = Vec::with_capacity(tile_count * segment_count);
    for packet in &fast420_packets {
        entropy_offsets.push(checked_u32(entropy_bytes.len(), "batch entropy offset")?);
        entropy_lens.push(checked_u32(
            packet.entropy_bytes.len(),
            "batch entropy length",
        )?);
        entropy_bytes.extend_from_slice(&packet.entropy_bytes);
        entropy_checkpoints.extend(packet.entropy_checkpoints.iter().copied());
    }
    if timing_enabled {
        timing.entropy_concat = timing_entropy_start
            .expect("timing start is set when timing is enabled")
            .elapsed();
    }

    let timing_buffer_start = timing_enabled.then(Instant::now);
    let y_plane = runtime.device.new_buffer(
        (y_len * tile_count) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cb_plane = runtime.device.new_buffer(
        (chroma_len * tile_count) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cr_plane = runtime.device.new_buffer(
        (chroma_len * tile_count) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buffer = runtime.device.new_buffer(
        (out_tile_len * tile_count) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let status_buffer = decode_status_buffer(&runtime.device, total_decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        entropy_bytes.as_ptr().cast(),
        entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let entropy_offsets_buffer =
        u32_buffer(&runtime.device, &entropy_offsets, "batch entropy offsets")?;
    let entropy_lens_buffer = u32_buffer(&runtime.device, &entropy_lens, "batch entropy lengths")?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &entropy_checkpoints)?;
    if timing_enabled {
        timing.buffer_alloc = timing_buffer_start
            .expect("timing start is set when timing is enabled")
            .elapsed();
    }

    let dc_tables = [
        PreparedHuffmanHost::from(&first.y_dc_table),
        PreparedHuffmanHost::from(&first.cb_dc_table),
        PreparedHuffmanHost::from(&first.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&first.y_ac_table),
        PreparedHuffmanHost::from(&first.cb_ac_table),
        PreparedHuffmanHost::from(&first.cr_ac_table),
    ];

    let mut command_buffer = runtime.queue.new_command_buffer();
    let mut _split_scratch: Option<(Buffer, Buffer)> = None;
    match decode_mode {
        Fast420BatchDecodeMode::Fused => {
            let timing_encode_start = timing_enabled.then(Instant::now);
            let decoder_encoder = command_buffer.new_compute_command_encoder();
            decoder_encoder.set_compute_pipeline_state(&runtime.fast420_batch_decode_pipeline);
            decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
            decoder_encoder.set_buffer(1, Some(&y_plane), 0);
            decoder_encoder.set_buffer(2, Some(&cb_plane), 0);
            decoder_encoder.set_buffer(3, Some(&cr_plane), 0);
            decoder_encoder.set_bytes(
                4,
                size_of::<JpegFast420BatchParams>() as u64,
                (&raw const params).cast(),
            );
            decoder_encoder.set_bytes(
                5,
                size_of::<[u16; 64]>() as u64,
                first.y_quant.as_ptr().cast(),
            );
            decoder_encoder.set_bytes(
                6,
                size_of::<[u16; 64]>() as u64,
                first.cb_quant.as_ptr().cast(),
            );
            decoder_encoder.set_bytes(
                7,
                size_of::<[u16; 64]>() as u64,
                first.cr_quant.as_ptr().cast(),
            );
            decoder_encoder.set_bytes(
                8,
                size_of::<PreparedHuffmanHost>() as u64,
                (&raw const dc_tables[0]).cast(),
            );
            decoder_encoder.set_bytes(
                9,
                size_of::<PreparedHuffmanHost>() as u64,
                (&raw const ac_tables[0]).cast(),
            );
            decoder_encoder.set_bytes(
                10,
                size_of::<PreparedHuffmanHost>() as u64,
                (&raw const dc_tables[1]).cast(),
            );
            decoder_encoder.set_bytes(
                11,
                size_of::<PreparedHuffmanHost>() as u64,
                (&raw const ac_tables[1]).cast(),
            );
            decoder_encoder.set_bytes(
                12,
                size_of::<PreparedHuffmanHost>() as u64,
                (&raw const dc_tables[2]).cast(),
            );
            decoder_encoder.set_bytes(
                13,
                size_of::<PreparedHuffmanHost>() as u64,
                (&raw const ac_tables[2]).cast(),
            );
            decoder_encoder.set_buffer(14, Some(&entropy_offsets_buffer), 0);
            decoder_encoder.set_buffer(15, Some(&entropy_lens_buffer), 0);
            decoder_encoder.set_buffer(16, Some(&status_buffer), 0);
            decoder_encoder.set_buffer(17, Some(&entropy_checkpoints_buffer), 0);
            dispatch_1d_pipeline(
                decoder_encoder,
                &runtime.fast420_batch_decode_pipeline,
                total_decode_threads,
            );
            decoder_encoder.end_encoding();
            if timing_enabled {
                timing.encode_decode = timing_encode_start
                    .expect("timing start is set when timing is enabled")
                    .elapsed();
                command_buffer.commit();
                let timing_wait_start = Instant::now();
                command_buffer.wait_until_completed();
                timing.wait_decode = timing_wait_start.elapsed();
                command_buffer = runtime.queue.new_command_buffer();
            }
        }
        Fast420BatchDecodeMode::SplitCoeffIdct => {
            let coeff_bytes = total_blocks
                .checked_mul(64)
                .and_then(|bytes| bytes.checked_mul(size_of::<i16>()))
                .ok_or_else(|| Error::MetalKernel {
                    message: "JPEG Metal fast420 batch coefficient scratch overflowed".to_string(),
                })?;
            let idct_component_depth =
                tile_count_u32
                    .checked_mul(6)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "JPEG Metal fast420 batch IDCT dispatch overflowed".to_string(),
                    })?;
            let coeff_blocks = runtime
                .device
                .new_buffer(coeff_bytes as u64, MTLResourceOptions::StorageModePrivate);
            let dc_only_flags = runtime
                .device
                .new_buffer(total_blocks as u64, MTLResourceOptions::StorageModePrivate);

            let coeff_encoder = command_buffer.new_compute_command_encoder();
            coeff_encoder.set_compute_pipeline_state(&runtime.fast420_batch_coeffs_decode_pipeline);
            coeff_encoder.set_buffer(0, Some(&entropy_buffer), 0);
            coeff_encoder.set_buffer(1, Some(&coeff_blocks), 0);
            coeff_encoder.set_buffer(2, Some(&dc_only_flags), 0);
            coeff_encoder.set_bytes(
                4,
                size_of::<JpegFast420BatchParams>() as u64,
                (&raw const params).cast(),
            );
            coeff_encoder.set_bytes(
                5,
                size_of::<[u16; 64]>() as u64,
                first.y_quant.as_ptr().cast(),
            );
            coeff_encoder.set_bytes(
                6,
                size_of::<[u16; 64]>() as u64,
                first.cb_quant.as_ptr().cast(),
            );
            coeff_encoder.set_bytes(
                7,
                size_of::<[u16; 64]>() as u64,
                first.cr_quant.as_ptr().cast(),
            );
            coeff_encoder.set_bytes(
                8,
                size_of::<PreparedHuffmanHost>() as u64,
                (&raw const dc_tables[0]).cast(),
            );
            coeff_encoder.set_bytes(
                9,
                size_of::<PreparedHuffmanHost>() as u64,
                (&raw const ac_tables[0]).cast(),
            );
            coeff_encoder.set_bytes(
                10,
                size_of::<PreparedHuffmanHost>() as u64,
                (&raw const dc_tables[1]).cast(),
            );
            coeff_encoder.set_bytes(
                11,
                size_of::<PreparedHuffmanHost>() as u64,
                (&raw const ac_tables[1]).cast(),
            );
            coeff_encoder.set_bytes(
                12,
                size_of::<PreparedHuffmanHost>() as u64,
                (&raw const dc_tables[2]).cast(),
            );
            coeff_encoder.set_bytes(
                13,
                size_of::<PreparedHuffmanHost>() as u64,
                (&raw const ac_tables[2]).cast(),
            );
            coeff_encoder.set_buffer(14, Some(&entropy_offsets_buffer), 0);
            coeff_encoder.set_buffer(15, Some(&entropy_lens_buffer), 0);
            coeff_encoder.set_buffer(16, Some(&status_buffer), 0);
            coeff_encoder.set_buffer(17, Some(&entropy_checkpoints_buffer), 0);
            dispatch_1d_pipeline(
                coeff_encoder,
                &runtime.fast420_batch_coeffs_decode_pipeline,
                total_decode_threads,
            );
            coeff_encoder.end_encoding();

            let idct_encoder = command_buffer.new_compute_command_encoder();
            idct_encoder.set_compute_pipeline_state(&runtime.fast420_batch_idct_deposit_pipeline);
            idct_encoder.set_buffer(0, Some(&coeff_blocks), 0);
            idct_encoder.set_buffer(1, Some(&dc_only_flags), 0);
            idct_encoder.set_buffer(2, Some(&y_plane), 0);
            idct_encoder.set_buffer(3, Some(&cb_plane), 0);
            idct_encoder.set_buffer(4, Some(&cr_plane), 0);
            idct_encoder.set_bytes(
                5,
                size_of::<JpegFast420BatchParams>() as u64,
                (&raw const params).cast(),
            );
            dispatch_3d_pipeline(
                idct_encoder,
                &runtime.fast420_batch_idct_deposit_pipeline,
                (first.mcus_per_row, first.mcu_rows, idct_component_depth),
            );
            idct_encoder.end_encoding();

            _split_scratch = Some((coeff_blocks, dc_only_flags));
        }
    }

    let timing_pack_encode_start = timing_enabled.then(Instant::now);
    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(&runtime.pack_420_rgb_batch_pipeline);
    pack_encoder.set_buffer(0, Some(&y_plane), 0);
    pack_encoder.set_buffer(1, Some(&cb_plane), 0);
    pack_encoder.set_buffer(2, Some(&cr_plane), 0);
    pack_encoder.set_buffer(3, Some(&out_buffer), 0);
    pack_encoder.set_bytes(
        4,
        size_of::<JpegFast420BatchParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_3d_pipeline(
        pack_encoder,
        &runtime.pack_420_rgb_batch_pipeline,
        (
            packed_pair_extent(width),
            packed_pair_extent(height),
            tile_count_u32,
        ),
    );
    pack_encoder.end_encoding();
    if timing_enabled {
        timing.encode_pack = timing_pack_encode_start
            .expect("timing start is set when timing is enabled")
            .elapsed();
    }

    command_buffer.commit();
    if timing_enabled {
        let timing_wait_start = Instant::now();
        command_buffer.wait_until_completed();
        timing.wait_pack = timing_wait_start.elapsed();
        timing.total = timing_total_start
            .expect("timing start is set when timing is enabled")
            .elapsed();
        timing.log("fused-stages", tile_count, first.dimensions, segment_count);
    } else {
        command_buffer.wait_until_completed();
    }

    if let Some(status) = first_decode_error_status(&status_buffer, total_decode_threads) {
        let mut results = Vec::with_capacity(requests.len());
        for request in requests {
            let decoder = CpuDecoder::new(request.input.as_ref())?;
            results.push(Err(decode_error_from_cpu(&decoder, request.fmt, status)));
        }
        return Ok(Some(results));
    }

    let mut results = Vec::with_capacity(requests.len());
    for index in 0..requests.len() {
        results.push(Ok(Surface::from_metal_buffer_offset(
            out_buffer.clone(),
            first.dimensions,
            PixelFormat::Rgb8,
            index * out_tile_len,
        )));
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn split_fast420_batch_enabled() -> bool {
    split_fast420_batch_value_enabled(std::env::var_os(SPLIT_FAST420_BATCH_ENV).as_deref())
}

#[cfg(target_os = "macos")]
fn split_fast420_batch_value_enabled(value: Option<&OsStr>) -> bool {
    value.is_some_and(|value| value == OsStr::new("1"))
}

#[cfg(target_os = "macos")]
fn fast420_batch_timing_enabled() -> bool {
    fast420_batch_timing_value_enabled(std::env::var_os(FAST420_BATCH_TIMING_ENV).as_deref())
}

#[cfg(target_os = "macos")]
fn fast420_batch_timing_value_enabled(value: Option<&OsStr>) -> bool {
    value.is_some_and(|value| value == OsStr::new("1"))
}

#[cfg(target_os = "macos")]
fn fast422_packets_share_full_rgb_batch_shape(
    first: &JpegMetalFast422PacketV1,
    packet: &JpegMetalFast422PacketV1,
    segment_count: usize,
) -> bool {
    packet.restart_interval_mcus == 0
        && packet.dimensions == first.dimensions
        && packet.mcus_per_row == first.mcus_per_row
        && packet.mcu_rows == first.mcu_rows
        && packet.entropy_checkpoints.len() == segment_count
        && packet.y_quant == first.y_quant
        && packet.cb_quant == first.cb_quant
        && packet.cr_quant == first.cr_quant
        && packet.y_dc_table == first.y_dc_table
        && packet.y_ac_table == first.y_ac_table
        && packet.cb_dc_table == first.cb_dc_table
        && packet.cb_ac_table == first.cb_ac_table
        && packet.cr_dc_table == first.cr_dc_table
        && packet.cr_ac_table == first.cr_ac_table
}

#[cfg(target_os = "macos")]
fn try_decode_fast422_full_rgb_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if requests.len() < 2
        || requests
            .iter()
            .any(|request| request.op != batch::BatchOp::Full || request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }

    let mut fast422_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let BatchedFastPacket::Fast422(packet) = packet else {
            return Ok(None);
        };
        fast422_packets.push(*packet);
    }

    let Some(first) = fast422_packets.first().copied() else {
        return Ok(None);
    };
    if first.restart_interval_mcus != 0 || first.entropy_checkpoints.is_empty() {
        return Ok(None);
    }

    let segment_count = first.entropy_checkpoints.len();
    if !fast422_packets
        .iter()
        .all(|packet| fast422_packets_share_full_rgb_batch_shape(first, packet, segment_count))
    {
        return Ok(None);
    }

    let tile_count = fast422_packets.len();
    let tile_count_u32 = checked_u32(tile_count, "batch tile count")?;
    let segment_count_u32 = checked_u32(segment_count, "batch segment count")?;
    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal batch decode thread count overflowed".to_string(),
            })?,
        "batch decode thread count",
    )?;

    let width = first.dimensions.0;
    let height = first.dimensions.1;
    let chroma_width = width.div_ceil(2);
    let chroma_height = height;
    let y_len = width as usize * height as usize;
    let chroma_len = chroma_width as usize * chroma_height as usize;
    let out_stride = width as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let out_tile_len = out_stride * height as usize;

    let params = JpegFast420BatchParams {
        width,
        height,
        chroma_width,
        chroma_height,
        mcus_per_row: first.mcus_per_row,
        mcu_rows: first.mcu_rows,
        segment_count: segment_count_u32,
        tile_count: tile_count_u32,
        out_stride: checked_u32(out_stride, "batch output stride")?,
        alpha: u32::from(u8::MAX),
    };

    let total_entropy_len = fast422_packets
        .iter()
        .map(|packet| packet.entropy_bytes.len())
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal batch entropy length overflowed".to_string(),
        })?;
    if total_entropy_len == 0 {
        return Ok(None);
    }

    let mut entropy_bytes = Vec::with_capacity(total_entropy_len);
    let mut entropy_offsets = Vec::with_capacity(tile_count);
    let mut entropy_lens = Vec::with_capacity(tile_count);
    let mut entropy_checkpoints = Vec::with_capacity(tile_count * segment_count);
    for packet in &fast422_packets {
        entropy_offsets.push(checked_u32(entropy_bytes.len(), "batch entropy offset")?);
        entropy_lens.push(checked_u32(
            packet.entropy_bytes.len(),
            "batch entropy length",
        )?);
        entropy_bytes.extend_from_slice(&packet.entropy_bytes);
        entropy_checkpoints.extend(packet.entropy_checkpoints.iter().copied());
    }

    let y_plane = runtime.device.new_buffer(
        (y_len * tile_count) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cb_plane = runtime.device.new_buffer(
        (chroma_len * tile_count) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cr_plane = runtime.device.new_buffer(
        (chroma_len * tile_count) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buffer = runtime.device.new_buffer(
        (out_tile_len * tile_count) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let status_buffer = decode_status_buffer(&runtime.device, total_decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        entropy_bytes.as_ptr().cast(),
        entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let entropy_offsets_buffer =
        u32_buffer(&runtime.device, &entropy_offsets, "batch entropy offsets")?;
    let entropy_lens_buffer = u32_buffer(&runtime.device, &entropy_lens, "batch entropy lengths")?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&first.y_dc_table),
        PreparedHuffmanHost::from(&first.cb_dc_table),
        PreparedHuffmanHost::from(&first.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&first.y_ac_table),
        PreparedHuffmanHost::from(&first.cb_ac_table),
        PreparedHuffmanHost::from(&first.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast422_batch_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&cb_plane), 0);
    decoder_encoder.set_buffer(3, Some(&cr_plane), 0);
    decoder_encoder.set_bytes(
        4,
        size_of::<JpegFast420BatchParams>() as u64,
        (&raw const params).cast(),
    );
    decoder_encoder.set_bytes(
        5,
        size_of::<[u16; 64]>() as u64,
        first.y_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        6,
        size_of::<[u16; 64]>() as u64,
        first.cb_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        7,
        size_of::<[u16; 64]>() as u64,
        first.cr_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        8,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&entropy_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&entropy_lens_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(17, Some(&entropy_checkpoints_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast422_batch_decode_pipeline,
        total_decode_threads,
    );
    decoder_encoder.end_encoding();

    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(&runtime.pack_422_rgb_batch_pipeline);
    pack_encoder.set_buffer(0, Some(&y_plane), 0);
    pack_encoder.set_buffer(1, Some(&cb_plane), 0);
    pack_encoder.set_buffer(2, Some(&cr_plane), 0);
    pack_encoder.set_buffer(3, Some(&out_buffer), 0);
    pack_encoder.set_bytes(
        4,
        size_of::<JpegFast420BatchParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_3d_pipeline(
        pack_encoder,
        &runtime.pack_422_rgb_batch_pipeline,
        (packed_pair_extent(width), height, tile_count_u32),
    );
    pack_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&status_buffer, total_decode_threads) {
        let mut results = Vec::with_capacity(requests.len());
        for request in requests {
            let decoder = CpuDecoder::new(request.input.as_ref())?;
            results.push(Err(decode_error_from_cpu(&decoder, request.fmt, status)));
        }
        return Ok(Some(results));
    }

    let mut results = Vec::with_capacity(requests.len());
    for index in 0..requests.len() {
        results.push(Ok(Surface::from_metal_buffer_offset(
            out_buffer.clone(),
            first.dimensions,
            PixelFormat::Rgb8,
            index * out_tile_len,
        )));
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn fast444_packets_share_region_scaled_batch_shape(
    first: &JpegMetalFast444PacketV1,
    packet: &JpegMetalFast444PacketV1,
    segment_count: usize,
) -> bool {
    packet.restart_interval_mcus == 0
        && packet.dimensions == first.dimensions
        && packet.mcus_per_row == first.mcus_per_row
        && packet.mcu_rows == first.mcu_rows
        && packet.entropy_checkpoints.len() == segment_count
        && packet.y_quant == first.y_quant
        && packet.cb_quant == first.cb_quant
        && packet.cr_quant == first.cr_quant
        && packet.y_dc_table == first.y_dc_table
        && packet.y_ac_table == first.y_ac_table
        && packet.cb_dc_table == first.cb_dc_table
        && packet.cb_ac_table == first.cb_ac_table
        && packet.cr_dc_table == first.cr_dc_table
        && packet.cr_ac_table == first.cr_ac_table
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_region_scaled_rgb_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if requests.len() < 2
        || requests
            .iter()
            .any(|request| request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }

    let mut fast444_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let BatchedFastPacket::Fast444(packet, mode) = packet else {
            return Ok(None);
        };
        fast444_packets.push((*packet, *mode));
    }

    let Some((first, first_mode)) = fast444_packets.first().copied() else {
        return Ok(None);
    };
    let batch::BatchOp::RegionScaled {
        roi: first_roi,
        scale: first_scale,
    } = requests[0].op
    else {
        return Ok(None);
    };
    if first.restart_interval_mcus != 0 || first.entropy_checkpoints.is_empty() {
        return Ok(None);
    }

    let first_scaled = scaled_rect_covering(first_roi, first_scale);
    let first_scaled_roi = signinum_jpeg::Rect {
        x: first_scaled.x,
        y: first_scaled.y,
        w: first_scaled.w,
        h: first_scaled.h,
    };
    let Some(first_decode_params) =
        fast444_scaled_region_params(first, first_scale, first_scaled_roi)
    else {
        return Ok(None);
    };

    let segment_count = first.entropy_checkpoints.len();
    let tile_count = fast444_packets.len();
    let tile_count_u32 = checked_u32(tile_count, "region scaled batch tile count")?;
    let segment_count_u32 = checked_u32(segment_count, "region scaled batch segment count")?;
    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal region scaled batch decode thread count overflowed"
                    .to_string(),
            })?,
        "region scaled batch decode thread count",
    )?;

    for (request, (packet, mode)) in requests.iter().zip(fast444_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        if scale != first_scale
            || mode != first_mode
            || !fast444_packets_share_region_scaled_batch_shape(first, packet, segment_count)
        {
            return Ok(None);
        }
        let scaled = scaled_rect_covering(roi, scale);
        let scaled_roi = signinum_jpeg::Rect {
            x: scaled.x,
            y: scaled.y,
            w: scaled.w,
            h: scaled.h,
        };
        if fast444_scaled_region_params(packet, scale, scaled_roi) != Some(first_decode_params) {
            return Ok(None);
        }
    }

    let out_stride =
        first_decode_params.scaled_width as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let out_tile_len = out_stride * first_decode_params.scaled_height as usize;

    let plane_len =
        first_decode_params.scaled_width as usize * first_decode_params.scaled_height as usize;
    let decode_params = JpegFastRegionScaledBatchParams {
        scaled_width: first_decode_params.scaled_width,
        scaled_height: first_decode_params.scaled_height,
        chroma_width: first_decode_params.scaled_width,
        chroma_height: first_decode_params.scaled_height,
        mcus_per_row: first_decode_params.mcus_per_row,
        mcu_rows: first_decode_params.mcu_rows,
        segment_count: segment_count_u32,
        tile_count: tile_count_u32,
        scale_shift: first_decode_params.scale_shift,
        origin_x: first_decode_params.origin_x,
        origin_y: first_decode_params.origin_y,
    };
    let pack_params = JpegWindowedPackBatchParams {
        src_width: first_decode_params.scaled_width,
        src_height: first_decode_params.scaled_height,
        chroma_width: first_decode_params.scaled_width,
        chroma_height: first_decode_params.scaled_height,
        src_x: 0,
        src_y: 0,
        width: first_decode_params.scaled_width,
        height: first_decode_params.scaled_height,
        tile_count: tile_count_u32,
        out_stride: checked_u32(out_stride, "region scaled batch output stride")?,
        alpha: u32::from(u8::MAX),
        mode: plane_mode_to_u32(first_mode),
        out_format: OUT_RGB,
    };

    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        fast444_packets
            .iter()
            .map(|(packet, _)| packet.entropy_bytes.as_slice()),
        fast444_packets
            .iter()
            .map(|(packet, _)| packet.entropy_checkpoints.as_slice()),
        tile_count,
        segment_count,
    )?
    else {
        return Ok(None);
    };

    let y_plane = runtime.device.new_buffer(
        (plane_len * tile_count) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cb_plane = runtime.device.new_buffer(
        (plane_len * tile_count) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cr_plane = runtime.device.new_buffer(
        (plane_len * tile_count) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buffer = runtime.device.new_buffer(
        (out_tile_len * tile_count) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let status_buffer = decode_status_buffer(&runtime.device, total_decode_threads);
    let dc_tables = [
        PreparedHuffmanHost::from(&first.y_dc_table),
        PreparedHuffmanHost::from(&first.cb_dc_table),
        PreparedHuffmanHost::from(&first.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&first.y_ac_table),
        PreparedHuffmanHost::from(&first.cb_ac_table),
        PreparedHuffmanHost::from(&first.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder
        .set_compute_pipeline_state(&runtime.fast444_scaled_region_batch_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffers.payload), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&cb_plane), 0);
    decoder_encoder.set_buffer(3, Some(&cr_plane), 0);
    decoder_encoder.set_bytes(
        4,
        size_of::<JpegFastRegionScaledBatchParams>() as u64,
        (&raw const decode_params).cast(),
    );
    decoder_encoder.set_bytes(
        5,
        size_of::<[u16; 64]>() as u64,
        first.y_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        6,
        size_of::<[u16; 64]>() as u64,
        first.cb_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        7,
        size_of::<[u16; 64]>() as u64,
        first.cr_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        8,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&entropy_buffers.offsets), 0);
    decoder_encoder.set_buffer(15, Some(&entropy_buffers.lens), 0);
    decoder_encoder.set_buffer(16, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_scaled_region_batch_decode_pipeline,
        total_decode_threads,
    );
    decoder_encoder.end_encoding();

    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(&runtime.pack_444_rgb_batch_pipeline);
    pack_encoder.set_buffer(0, Some(&y_plane), 0);
    pack_encoder.set_buffer(1, Some(&cb_plane), 0);
    pack_encoder.set_buffer(2, Some(&cr_plane), 0);
    pack_encoder.set_buffer(3, Some(&out_buffer), 0);
    pack_encoder.set_bytes(
        4,
        size_of::<JpegWindowedPackBatchParams>() as u64,
        (&raw const pack_params).cast(),
    );
    dispatch_3d_pipeline(
        pack_encoder,
        &runtime.pack_444_rgb_batch_pipeline,
        (
            first_decode_params.scaled_width,
            first_decode_params.scaled_height,
            tile_count_u32,
        ),
    );
    pack_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(results) =
        region_scaled_batch_error_results(requests, &status_buffer, total_decode_threads)?
    {
        return Ok(Some(results));
    }

    let mut results = Vec::with_capacity(requests.len());
    for index in 0..requests.len() {
        results.push(Ok(Surface::from_metal_buffer_offset(
            out_buffer.clone(),
            (
                first_decode_params.scaled_width,
                first_decode_params.scaled_height,
            ),
            PixelFormat::Rgb8,
            index * out_tile_len,
        )));
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast420_region_scaled_rgb_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if requests.len() < 2
        || requests
            .iter()
            .any(|request| request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }

    let mut fast420_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let BatchedFastPacket::Fast420(packet) = packet else {
            return Ok(None);
        };
        fast420_packets.push(*packet);
    }

    let Some(first) = fast420_packets.first().copied() else {
        return Ok(None);
    };
    let batch::BatchOp::RegionScaled {
        roi: first_roi,
        scale: first_scale,
    } = requests[0].op
    else {
        return Ok(None);
    };
    if first.restart_interval_mcus != 0 || first.entropy_checkpoints.is_empty() {
        return Ok(None);
    }

    let segment_count = first.entropy_checkpoints.len();
    let tile_count = fast420_packets.len();
    let tile_count_u32 = checked_u32(tile_count, "region scaled batch tile count")?;
    let segment_count_u32 = checked_u32(segment_count, "region scaled batch segment count")?;
    let Some(first_plan) = fast420_region_scaled_batch_plan(
        first,
        first_roi,
        first_scale,
        tile_count_u32,
        segment_count_u32,
    ) else {
        return Ok(None);
    };

    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal fast420 region scaled batch decode thread count overflowed"
                    .to_string(),
            })?,
        "fast420 region scaled batch decode thread count",
    )?;

    for (request, packet) in requests.iter().zip(fast420_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        if scale != first_scale
            || !fast420_packets_share_full_rgb_batch_shape(first, packet, segment_count)
            || fast420_region_scaled_batch_plan(
                packet,
                roi,
                scale,
                tile_count_u32,
                segment_count_u32,
            ) != Some(first_plan)
        {
            return Ok(None);
        }
    }

    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        fast420_packets
            .iter()
            .map(|packet| packet.entropy_bytes.as_slice()),
        fast420_packets
            .iter()
            .map(|packet| packet.entropy_checkpoints.as_slice()),
        tile_count,
        segment_count,
    )?
    else {
        return Ok(None);
    };

    let y_plane = runtime.device.new_buffer(
        (first_plan.y_len * tile_count) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cb_plane = runtime.device.new_buffer(
        (first_plan.chroma_len * tile_count) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cr_plane = runtime.device.new_buffer(
        (first_plan.chroma_len * tile_count) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buffer = runtime.device.new_buffer(
        (first_plan.out_tile_len * tile_count) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let status_buffer = decode_status_buffer(&runtime.device, total_decode_threads);
    let dc_tables = [
        PreparedHuffmanHost::from(&first.y_dc_table),
        PreparedHuffmanHost::from(&first.cb_dc_table),
        PreparedHuffmanHost::from(&first.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&first.y_ac_table),
        PreparedHuffmanHost::from(&first.cb_ac_table),
        PreparedHuffmanHost::from(&first.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder
        .set_compute_pipeline_state(&runtime.fast420_scaled_region_batch_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffers.payload), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&cb_plane), 0);
    decoder_encoder.set_buffer(3, Some(&cr_plane), 0);
    decoder_encoder.set_bytes(
        4,
        size_of::<JpegFastRegionScaledBatchParams>() as u64,
        (&raw const first_plan.decode_params).cast(),
    );
    decoder_encoder.set_bytes(
        5,
        size_of::<[u16; 64]>() as u64,
        first.y_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        6,
        size_of::<[u16; 64]>() as u64,
        first.cb_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        7,
        size_of::<[u16; 64]>() as u64,
        first.cr_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        8,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&entropy_buffers.offsets), 0);
    decoder_encoder.set_buffer(15, Some(&entropy_buffers.lens), 0);
    decoder_encoder.set_buffer(16, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast420_scaled_region_batch_decode_pipeline,
        total_decode_threads,
    );
    decoder_encoder.end_encoding();

    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(&runtime.pack_420_windowed_rgb_batch_pipeline);
    pack_encoder.set_buffer(0, Some(&y_plane), 0);
    pack_encoder.set_buffer(1, Some(&cb_plane), 0);
    pack_encoder.set_buffer(2, Some(&cr_plane), 0);
    pack_encoder.set_buffer(3, Some(&out_buffer), 0);
    pack_encoder.set_bytes(
        4,
        size_of::<JpegWindowedPackBatchParams>() as u64,
        (&raw const first_plan.pack_params).cast(),
    );
    dispatch_3d_pipeline(
        pack_encoder,
        &runtime.pack_420_windowed_rgb_batch_pipeline,
        (first_plan.out_dims.0, first_plan.out_dims.1, tile_count_u32),
    );
    pack_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(results) =
        region_scaled_batch_error_results(requests, &status_buffer, total_decode_threads)?
    {
        return Ok(Some(results));
    }

    let mut results = Vec::with_capacity(requests.len());
    for index in 0..requests.len() {
        results.push(Ok(Surface::from_metal_buffer_offset(
            out_buffer.clone(),
            first_plan.out_dims,
            PixelFormat::Rgb8,
            index * first_plan.out_tile_len,
        )));
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast422_region_scaled_rgb_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if requests.len() < 2
        || requests
            .iter()
            .any(|request| request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }

    let mut fast422_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let BatchedFastPacket::Fast422(packet) = packet else {
            return Ok(None);
        };
        fast422_packets.push(*packet);
    }

    let Some(first) = fast422_packets.first().copied() else {
        return Ok(None);
    };
    let batch::BatchOp::RegionScaled {
        roi: first_roi,
        scale: first_scale,
    } = requests[0].op
    else {
        return Ok(None);
    };
    if first.restart_interval_mcus != 0 || first.entropy_checkpoints.is_empty() {
        return Ok(None);
    }

    let segment_count = first.entropy_checkpoints.len();
    let tile_count = fast422_packets.len();
    let tile_count_u32 = checked_u32(tile_count, "region scaled batch tile count")?;
    let segment_count_u32 = checked_u32(segment_count, "region scaled batch segment count")?;
    let Some(first_plan) = fast422_region_scaled_batch_plan(
        first,
        first_roi,
        first_scale,
        tile_count_u32,
        segment_count_u32,
    ) else {
        return Ok(None);
    };

    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal fast422 region scaled batch decode thread count overflowed"
                    .to_string(),
            })?,
        "fast422 region scaled batch decode thread count",
    )?;

    for (request, packet) in requests.iter().zip(fast422_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        if scale != first_scale
            || !fast422_packets_share_full_rgb_batch_shape(first, packet, segment_count)
            || fast422_region_scaled_batch_plan(
                packet,
                roi,
                scale,
                tile_count_u32,
                segment_count_u32,
            ) != Some(first_plan)
        {
            return Ok(None);
        }
    }

    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        fast422_packets
            .iter()
            .map(|packet| packet.entropy_bytes.as_slice()),
        fast422_packets
            .iter()
            .map(|packet| packet.entropy_checkpoints.as_slice()),
        tile_count,
        segment_count,
    )?
    else {
        return Ok(None);
    };

    let y_plane = runtime.device.new_buffer(
        (first_plan.y_len * tile_count) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cb_plane = runtime.device.new_buffer(
        (first_plan.chroma_len * tile_count) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cr_plane = runtime.device.new_buffer(
        (first_plan.chroma_len * tile_count) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buffer = runtime.device.new_buffer(
        (first_plan.out_tile_len * tile_count) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let status_buffer = decode_status_buffer(&runtime.device, total_decode_threads);
    let dc_tables = [
        PreparedHuffmanHost::from(&first.y_dc_table),
        PreparedHuffmanHost::from(&first.cb_dc_table),
        PreparedHuffmanHost::from(&first.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&first.y_ac_table),
        PreparedHuffmanHost::from(&first.cb_ac_table),
        PreparedHuffmanHost::from(&first.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder
        .set_compute_pipeline_state(&runtime.fast422_scaled_region_batch_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffers.payload), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&cb_plane), 0);
    decoder_encoder.set_buffer(3, Some(&cr_plane), 0);
    decoder_encoder.set_bytes(
        4,
        size_of::<JpegFastRegionScaledBatchParams>() as u64,
        (&raw const first_plan.decode_params).cast(),
    );
    decoder_encoder.set_bytes(
        5,
        size_of::<[u16; 64]>() as u64,
        first.y_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        6,
        size_of::<[u16; 64]>() as u64,
        first.cb_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        7,
        size_of::<[u16; 64]>() as u64,
        first.cr_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        8,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&entropy_buffers.offsets), 0);
    decoder_encoder.set_buffer(15, Some(&entropy_buffers.lens), 0);
    decoder_encoder.set_buffer(16, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast422_scaled_region_batch_decode_pipeline,
        total_decode_threads,
    );
    decoder_encoder.end_encoding();

    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(&runtime.pack_422_windowed_rgb_batch_pipeline);
    pack_encoder.set_buffer(0, Some(&y_plane), 0);
    pack_encoder.set_buffer(1, Some(&cb_plane), 0);
    pack_encoder.set_buffer(2, Some(&cr_plane), 0);
    pack_encoder.set_buffer(3, Some(&out_buffer), 0);
    pack_encoder.set_bytes(
        4,
        size_of::<JpegWindowedPackBatchParams>() as u64,
        (&raw const first_plan.pack_params).cast(),
    );
    dispatch_3d_pipeline(
        pack_encoder,
        &runtime.pack_422_windowed_rgb_batch_pipeline,
        (first_plan.out_dims.0, first_plan.out_dims.1, tile_count_u32),
    );
    pack_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(results) =
        region_scaled_batch_error_results(requests, &status_buffer, total_decode_threads)?
    {
        return Ok(Some(results));
    }

    let mut results = Vec::with_capacity(requests.len());
    for index in 0..requests.len() {
        results.push(Ok(Surface::from_metal_buffer_offset(
            out_buffer.clone(),
            first_plan.out_dims,
            PixelFormat::Rgb8,
            index * first_plan.out_tile_len,
        )));
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn requests_share_one_input(requests: &[batch::QueuedRequest]) -> bool {
    let Some(first) = requests.first() else {
        return false;
    };
    requests.iter().all(|request| {
        request.input.as_ptr() == first.input.as_ptr() && request.input.len() == first.input.len()
    })
}

#[cfg(target_os = "macos")]
fn requests_share_one_region_scaled_work(requests: &[batch::QueuedRequest]) -> bool {
    let Some(first) = requests.first() else {
        return false;
    };
    requests_share_one_input(requests)
        && requests.iter().all(|request| {
            request.fmt == first.fmt && request.backend == first.backend && request.op == first.op
        })
}

#[cfg(target_os = "macos")]
fn decode_region_scaled_packet_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    request: &batch::QueuedRequest,
    packet: &BatchedFastPacket<'_>,
) -> Result<Surface, Error> {
    let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
        return Err(Error::MetalKernel {
            message: "JPEG Metal expected a region scaled batch request".to_string(),
        });
    };
    let scaled = scaled_rect_covering(roi, scale);
    let scaled_roi = signinum_jpeg::Rect {
        x: scaled.x,
        y: scaled.y,
        w: scaled.w,
        h: scaled.h,
    };
    match packet {
        BatchedFastPacket::Fast420(packet) => try_decode_fast420_scaled_region_to_surface(
            runtime,
            decoder,
            Some(packet),
            request.fmt,
            scaled_roi,
            scale,
        ),
        BatchedFastPacket::Fast422(packet) => try_decode_fast422_scaled_region_to_surface(
            runtime,
            Some(packet),
            request.fmt,
            scaled_roi,
            scale,
        ),
        BatchedFastPacket::Fast444(packet, _) => try_decode_fast444_scaled_region_to_surface(
            runtime,
            decoder,
            Some(packet),
            request.fmt,
            scaled_roi,
            scale,
        ),
    }
    .and_then(|surface| {
        surface.ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal repeated region scaled batch was not packet-decodable".to_string(),
        })
    })
}

#[cfg(target_os = "macos")]
fn try_decode_repeated_region_scaled_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if requests.len() <= REGION_SCALED_BATCH_CHUNK
        || !requests_share_one_input(requests)
        || !requests
            .iter()
            .all(|request| matches!(request.op, batch::BatchOp::RegionScaled { .. }))
    {
        return Ok(None);
    }

    let decoder = CpuDecoder::new(requests[0].input.as_ref())?;
    if requests_share_one_region_scaled_work(requests) {
        let surface =
            decode_region_scaled_packet_surface(runtime, &decoder, &requests[0], &packets[0])?;
        return Ok(Some(
            (0..requests.len())
                .map(|_| Ok(surface.clone()))
                .collect::<Vec<_>>(),
        ));
    }

    let mut results = Vec::with_capacity(requests.len());
    for (request, packet) in requests.iter().zip(packets.iter()) {
        results.push(decode_region_scaled_packet_surface(
            runtime, &decoder, request, packet,
        ));
    }

    Ok(Some(results))
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_full_batch_to_surfaces(
    requests: &[batch::QueuedRequest],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let Some(packets) = batched_fast_packets(requests)? else {
        return Ok(None);
    };

    with_runtime(|runtime| {
        if let Some(results) =
            try_decode_fast420_full_rgb_batch_to_surfaces(runtime, requests, &packets)?
        {
            return Ok(Some(results));
        }
        if let Some(results) =
            try_decode_fast422_full_rgb_batch_to_surfaces(runtime, requests, &packets)?
        {
            return Ok(Some(results));
        }
        if let Some(results) =
            try_decode_repeated_region_scaled_batch_to_surfaces(runtime, requests, &packets)?
        {
            return Ok(Some(results));
        }
        if let Some(results) =
            try_decode_fast444_region_scaled_rgb_batch_to_surfaces(runtime, requests, &packets)?
        {
            return Ok(Some(results));
        }
        if let Some(results) =
            try_decode_fast420_region_scaled_rgb_batch_to_surfaces(runtime, requests, &packets)?
        {
            return Ok(Some(results));
        }
        if let Some(results) =
            try_decode_fast422_region_scaled_rgb_batch_to_surfaces(runtime, requests, &packets)?
        {
            return Ok(Some(results));
        }

        let mut results = Vec::with_capacity(requests.len());
        let has_region_scaled = requests
            .iter()
            .any(|request| matches!(request.op, batch::BatchOp::RegionScaled { .. }));
        let chunk_size = if has_region_scaled {
            REGION_SCALED_BATCH_CHUNK
        } else {
            requests.len().max(1)
        };
        for chunk_start in (0..requests.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(requests.len());
            let command_buffer = runtime.queue.new_command_buffer();
            let mut encoded = Vec::with_capacity(chunk_end - chunk_start);
            let mut device_buffer_cache = BatchDeviceBufferCache::default();
            for index in chunk_start..chunk_end {
                let request = &requests[index];
                let packet = &packets[index];
                let item = match packet {
                    BatchedFastPacket::Fast420(packet) => match request.op {
                        batch::BatchOp::Full => encode_fast420_batch_item(
                            runtime,
                            command_buffer,
                            index,
                            packet,
                            request.fmt,
                        )?,
                        batch::BatchOp::Region(roi) => encode_fast420_region_batch_item(
                            runtime,
                            command_buffer,
                            index,
                            packet,
                            request.fmt,
                            roi,
                        )?,
                        batch::BatchOp::Scaled(scale) => encode_fast420_scaled_batch_item(
                            runtime,
                            command_buffer,
                            index,
                            packet,
                            request.fmt,
                            scale,
                        )?,
                        batch::BatchOp::RegionScaled { roi, scale } => {
                            encode_fast420_scaled_region_batch_item(
                                runtime,
                                command_buffer,
                                &mut device_buffer_cache,
                                index,
                                packet,
                                request.fmt,
                                roi,
                                scale,
                            )?
                        }
                    },
                    BatchedFastPacket::Fast422(packet) => match request.op {
                        batch::BatchOp::Full => encode_fast422_batch_item(
                            runtime,
                            command_buffer,
                            index,
                            packet,
                            request.fmt,
                        )?,
                        batch::BatchOp::Region(roi) => encode_fast422_region_batch_item(
                            runtime,
                            command_buffer,
                            index,
                            packet,
                            request.fmt,
                            roi,
                        )?,
                        batch::BatchOp::Scaled(scale) => encode_fast422_scaled_batch_item(
                            runtime,
                            command_buffer,
                            index,
                            packet,
                            request.fmt,
                            scale,
                        )?,
                        batch::BatchOp::RegionScaled { roi, scale } => {
                            encode_fast422_scaled_region_batch_item(
                                runtime,
                                command_buffer,
                                &mut device_buffer_cache,
                                index,
                                packet,
                                request.fmt,
                                roi,
                                scale,
                            )?
                        }
                    },
                    BatchedFastPacket::Fast444(packet, mode) => match request.op {
                        batch::BatchOp::Full => encode_fast444_batch_item(
                            runtime,
                            command_buffer,
                            index,
                            packet,
                            *mode,
                            request.fmt,
                        )?,
                        batch::BatchOp::Region(roi) => encode_fast444_region_batch_item(
                            runtime,
                            command_buffer,
                            index,
                            packet,
                            *mode,
                            request.fmt,
                            roi,
                        )?,
                        batch::BatchOp::Scaled(scale) => encode_fast444_scaled_batch_item(
                            runtime,
                            command_buffer,
                            index,
                            packet,
                            *mode,
                            request.fmt,
                            scale,
                        )?,
                        batch::BatchOp::RegionScaled { roi, scale } => {
                            encode_fast444_scaled_region_batch_item(
                                runtime,
                                command_buffer,
                                &mut device_buffer_cache,
                                index,
                                packet,
                                *mode,
                                request.fmt,
                                roi,
                                scale,
                            )?
                        }
                    },
                };
                encoded.push(item);
            }

            command_buffer.commit();
            command_buffer.wait_until_completed();

            for item in encoded {
                if let Some(status) =
                    first_decode_error_status(&item.status_buffer, item.decode_threads)
                {
                    let request = &requests[item.request_index];
                    let decoder = CpuDecoder::new(request.input.as_ref())?;
                    results.push(Err(decode_error_from_cpu(&decoder, request.fmt, status)));
                } else {
                    results.push(Ok(item.surface));
                }
            }
        }
        Ok(Some(results))
    })
}

#[cfg(target_os = "macos")]
fn try_decode_fast422_to_surface(
    runtime: &MetalRuntime,
    packet: Option<&JpegMetalFast422PacketV1>,
    fmt: PixelFormat,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_out_format) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };

    let params = fast422_params(packet, fmt)?;
    let y_len = params.width as usize * params.height as usize;
    let chroma_len = params.chroma_width as usize * params.chroma_height as usize;
    let y_plane = runtime
        .device
        .new_buffer(y_len as u64, MTLResourceOptions::StorageModeShared);
    let cb_plane = runtime
        .device
        .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared);
    let cr_plane = runtime
        .device
        .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let out_buffer = (fmt != PixelFormat::Gray8).then(|| {
        runtime.device.new_buffer(
            (params.out_stride as usize * params.height as usize) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    });

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast422_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&cb_plane), 0);
    decoder_encoder.set_buffer(3, Some(&cr_plane), 0);
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast422_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    if let Some(out_buffer) = out_buffer.as_ref() {
        let Some(pack_pipeline) = pack_422_pipeline_for_format(runtime, fmt) else {
            return Ok(None);
        };
        let pack_encoder = command_buffer.new_compute_command_encoder();
        pack_encoder.set_compute_pipeline_state(pack_pipeline);
        pack_encoder.set_buffer(0, Some(&y_plane), 0);
        pack_encoder.set_buffer(1, Some(&cb_plane), 0);
        pack_encoder.set_buffer(2, Some(&cr_plane), 0);
        pack_encoder.set_buffer(3, Some(out_buffer), 0);
        pack_encoder.set_bytes(
            4,
            size_of::<JpegFast420Params>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(pack_encoder, pack_pipeline, packet.dimensions);
        pack_encoder.end_encoding();
    }

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads) {
        return Err(Error::MetalKernel {
            message: format!(
                "unexpected Metal fast422 failure at entropy byte {}",
                status.position
            ),
        });
    }

    Ok(Some(match out_buffer {
        Some(out_buffer) => Surface::from_metal_buffer(out_buffer, packet.dimensions, fmt),
        None => Surface::from_metal_buffer(y_plane, packet.dimensions, fmt),
    }))
}

#[cfg(target_os = "macos")]
fn fast422_status_error(status: JpegDecodeStatus) -> Error {
    Error::MetalKernel {
        message: format!(
            "unexpected Metal fast422 failure at entropy byte {}",
            status.position
        ),
    }
}

#[cfg(target_os = "macos")]
fn try_decode_fast422_region_to_surface(
    runtime: &MetalRuntime,
    packet: Option<&JpegMetalFast422PacketV1>,
    fmt: PixelFormat,
    roi: signinum_jpeg::Rect,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };

    let command_buffer = runtime.queue.new_command_buffer();
    let item = encode_fast422_region_batch_item(
        runtime,
        command_buffer,
        0,
        packet,
        fmt,
        Rect {
            x: roi.x,
            y: roi.y,
            w: roi.w,
            h: roi.h,
        },
    )?;
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&item.status_buffer, item.decode_threads) {
        return Err(fast422_status_error(status));
    }

    Ok(Some(item.surface))
}

#[cfg(target_os = "macos")]
fn try_decode_fast422_scaled_to_surface(
    runtime: &MetalRuntime,
    packet: Option<&JpegMetalFast422PacketV1>,
    fmt: PixelFormat,
    scale: signinum_core::Downscale,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };
    if fast422_scaled_params(packet, scale).is_none() {
        return Ok(None);
    }

    let command_buffer = runtime.queue.new_command_buffer();
    let item = encode_fast422_scaled_batch_item(runtime, command_buffer, 0, packet, fmt, scale)?;
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&item.status_buffer, item.decode_threads) {
        return Err(fast422_status_error(status));
    }

    Ok(Some(item.surface))
}

#[cfg(target_os = "macos")]
fn try_decode_fast422_scaled_region_to_surface(
    runtime: &MetalRuntime,
    packet: Option<&JpegMetalFast422PacketV1>,
    fmt: PixelFormat,
    scaled_roi: signinum_jpeg::Rect,
    scale: signinum_core::Downscale,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };
    let Some(full_params) = fast422_scaled_params(packet, scale) else {
        return Ok(None);
    };
    let source_window = full_mcu_scaled_window_422(
        (full_params.scaled_width, full_params.scaled_height),
        scaled_roi,
        full_params.scale_shift,
    );
    let Some(mut decode_params) = fast422_scaled_region_params(packet, scale, source_window) else {
        return Ok(None);
    };
    let mcu_width = 16u32 >> decode_params.scale_shift;
    let mcu_height = 8u32 >> decode_params.scale_shift;
    let (first_mcu, end_mcu) = mcu_range_for_rect(
        source_window,
        packet.mcus_per_row,
        packet.mcu_rows,
        mcu_width,
        mcu_height,
    );
    let total_mcus = packet.mcus_per_row * packet.mcu_rows;
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        &packet.restart_offsets,
        packet.restart_interval_mcus,
        total_mcus,
        first_mcu,
        end_mcu,
    );
    decode_params.restart_start_mcu = restart_start_mcu;
    decode_params.restart_offset_count = entropy_segment_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let local_roi = signinum_jpeg::Rect {
        x: scaled_roi.x - source_window.x,
        y: scaled_roi.y - source_window.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let pack_params =
        fast422_windowed_pack_params_for_dims((source_window.w, source_window.h), fmt, local_roi);
    let y_len = source_window.w as usize * source_window.h as usize;
    let chroma_len = source_window.w.div_ceil(2) as usize * source_window.h as usize;
    let y_plane = runtime
        .device
        .new_buffer(y_len as u64, MTLResourceOptions::StorageModeShared);
    let cb_plane = runtime
        .device
        .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared);
    let cr_plane = runtime
        .device
        .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let out_buffer = runtime.device.new_buffer(
        (pack_params.out_stride as usize * scaled_roi.h as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast422_scaled_region_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&cb_plane), 0);
    decoder_encoder.set_buffer(3, Some(&cr_plane), 0);
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast422_scaled_region_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let pack_encoder = command_buffer.new_compute_command_encoder();
    let pack_pipeline = pack_422_windowed_pipeline_for_format(runtime, fmt);
    pack_encoder.set_compute_pipeline_state(pack_pipeline);
    pack_encoder.set_buffer(0, Some(&y_plane), 0);
    pack_encoder.set_buffer(1, Some(&cb_plane), 0);
    pack_encoder.set_buffer(2, Some(&cr_plane), 0);
    pack_encoder.set_buffer(3, Some(&out_buffer), 0);
    pack_encoder.set_bytes(
        4,
        size_of::<JpegFast420WindowedPackParams>() as u64,
        (&raw const pack_params).cast(),
    );
    dispatch_2d_pipeline(pack_encoder, pack_pipeline, (scaled_roi.w, scaled_roi.h));
    pack_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads) {
        return Err(fast422_status_error(status));
    }

    Ok(Some(Surface::from_metal_buffer(
        out_buffer,
        (scaled_roi.w, scaled_roi.h),
        fmt,
    )))
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
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast420_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    if let Some(out_buffer) = out_buffer.as_ref() {
        let pack_encoder = command_buffer.new_compute_command_encoder();
        let pack_pipeline = pack_420_pipeline_for_format(runtime, fmt);
        pack_encoder.set_compute_pipeline_state(pack_pipeline);
        pack_encoder.set_buffer(0, Some(&y_plane), 0);
        pack_encoder.set_buffer(1, Some(&chroma_buffers[0]), 0);
        pack_encoder.set_buffer(2, Some(&chroma_buffers[1]), 0);
        pack_encoder.set_buffer(3, Some(out_buffer), 0);
        pack_encoder.set_bytes(
            4,
            size_of::<JpegFast420Params>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(pack_encoder, pack_pipeline, packet.dimensions);
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
    roi: signinum_jpeg::Rect,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };

    let source_window = full_mcu_window_420(packet.dimensions, roi);
    let mut decode_params = fast420_region_params(packet, fmt, source_window)?;
    let (first_mcu, end_mcu) =
        mcu_range_for_rect(source_window, packet.mcus_per_row, packet.mcu_rows, 16, 16);
    let total_mcus = packet.mcus_per_row * packet.mcu_rows;
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        &packet.restart_offsets,
        packet.restart_interval_mcus,
        total_mcus,
        first_mcu,
        end_mcu,
    );
    decode_params.restart_start_mcu = restart_start_mcu;
    decode_params.restart_offset_count = entropy_segment_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let local_roi = signinum_jpeg::Rect {
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
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast420_region_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let pack_encoder = command_buffer.new_compute_command_encoder();
    let pack_pipeline = pack_420_windowed_pipeline_for_format(runtime, fmt);
    pack_encoder.set_compute_pipeline_state(pack_pipeline);
    pack_encoder.set_buffer(0, Some(&y_plane), 0);
    pack_encoder.set_buffer(1, Some(&chroma_buffers[0]), 0);
    pack_encoder.set_buffer(2, Some(&chroma_buffers[1]), 0);
    pack_encoder.set_buffer(3, Some(&out_buffer), 0);
    pack_encoder.set_bytes(
        4,
        size_of::<JpegFast420WindowedPackParams>() as u64,
        (&raw const pack_params).cast(),
    );
    dispatch_2d_pipeline(pack_encoder, pack_pipeline, (roi.w, roi.h));
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
    scale: signinum_core::Downscale,
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
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
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
            restart_start_mcu: params.restart_start_mcu,
            entropy_len: params.entropy_len,
            out_stride: u32::try_from(params.scaled_width as usize * fmt.bytes_per_pixel())
                .expect("JPEG Metal output stride fits in u32"),
            alpha: u32::from(u8::MAX),
            out_format: pixel_format_to_out_format(fmt).expect("validated output format"),
            origin_x: 0,
            origin_y: 0,
        };
        let pack_encoder = command_buffer.new_compute_command_encoder();
        let pack_pipeline = pack_420_pipeline_for_format(runtime, fmt);
        pack_encoder.set_compute_pipeline_state(pack_pipeline);
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
            pack_pipeline,
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
    scaled_roi: signinum_jpeg::Rect,
    scale: signinum_core::Downscale,
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
    let Some(mut decode_params) = fast420_scaled_region_params(packet, scale, source_window) else {
        return Ok(None);
    };
    let mcu_size = 16u32 >> decode_params.scale_shift;
    let (first_mcu, end_mcu) = mcu_range_for_rect(
        source_window,
        packet.mcus_per_row,
        packet.mcu_rows,
        mcu_size,
        mcu_size,
    );
    let total_mcus = packet.mcus_per_row * packet.mcu_rows;
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        &packet.restart_offsets,
        packet.restart_interval_mcus,
        total_mcus,
        first_mcu,
        end_mcu,
    );
    decode_params.restart_start_mcu = restart_start_mcu;
    decode_params.restart_offset_count = entropy_segment_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let local_roi = signinum_jpeg::Rect {
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
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast420_scaled_region_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let pack_encoder = command_buffer.new_compute_command_encoder();
    let pack_pipeline = pack_420_windowed_pipeline_for_format(runtime, fmt);
    pack_encoder.set_compute_pipeline_state(pack_pipeline);
    pack_encoder.set_buffer(0, Some(&y_plane), 0);
    pack_encoder.set_buffer(1, Some(&chroma_buffers[0]), 0);
    pack_encoder.set_buffer(2, Some(&chroma_buffers[1]), 0);
    pack_encoder.set_buffer(3, Some(&out_buffer), 0);
    pack_encoder.set_bytes(
        4,
        size_of::<JpegFast420WindowedPackParams>() as u64,
        (&raw const pack_params).cast(),
    );
    dispatch_2d_pipeline(pack_encoder, pack_pipeline, (scaled_roi.w, scaled_roi.h));
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
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
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
    roi: signinum_jpeg::Rect,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };

    let mut params = fast444_region_params(packet, roi);
    let (first_mcu, end_mcu) = mcu_range_for_rect(roi, packet.mcus_per_row, packet.mcu_rows, 8, 8);
    let total_mcus = packet.mcus_per_row * packet.mcu_rows;
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        &packet.restart_offsets,
        packet.restart_interval_mcus,
        total_mcus,
        first_mcu,
        end_mcu,
    );
    params.restart_start_mcu = restart_start_mcu;
    params.restart_offset_count = entropy_segment_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
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
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
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
    scale: signinum_core::Downscale,
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
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
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
    scaled_roi: signinum_jpeg::Rect,
    scale: signinum_core::Downscale,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };
    let Some(mut params) = fast444_scaled_region_params(packet, scale, scaled_roi) else {
        return Ok(None);
    };
    let mcu_size = 8u32 >> params.scale_shift;
    let (first_mcu, end_mcu) = mcu_range_for_rect(
        scaled_roi,
        packet.mcus_per_row,
        packet.mcu_rows,
        mcu_size,
        mcu_size,
    );
    let total_mcus = packet.mcus_per_row * packet.mcu_rows;
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        &packet.restart_offsets,
        packet.restart_interval_mcus,
        total_mcus,
        first_mcu,
        end_mcu,
    );
    params.restart_start_mcu = restart_start_mcu;
    params.restart_offset_count = entropy_segment_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );

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
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
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
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&restart_offsets_buffer), 0);
    decoder_encoder.set_buffer(15, Some(&status_buffer), 0);
    decoder_encoder.set_buffer(16, Some(&entropy_checkpoints_buffer), 0);
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
fn scaled_rect_covering(rect: Rect, scale: signinum_core::Downscale) -> Rect {
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
    pool: &mut signinum_jpeg::ScratchPool,
    fmt: PixelFormat,
    fast444_packet: Option<&JpegMetalFast444PacketV1>,
    fast422_packet: Option<&JpegMetalFast422PacketV1>,
    fast420_packet: Option<&JpegMetalFast420PacketV1>,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        decode_to_surface_with_runtime(
            runtime,
            decoder,
            pool,
            fmt,
            fast444_packet,
            fast422_packet,
            fast420_packet,
        )
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_to_surface_with_device(
    decoder: &CpuDecoder<'_>,
    pool: &mut signinum_jpeg::ScratchPool,
    fmt: PixelFormat,
    fast444_packet: Option<&JpegMetalFast444PacketV1>,
    fast422_packet: Option<&JpegMetalFast422PacketV1>,
    fast420_packet: Option<&JpegMetalFast420PacketV1>,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |runtime| {
        decode_to_surface_with_runtime(
            runtime,
            decoder,
            pool,
            fmt,
            fast444_packet,
            fast422_packet,
            fast420_packet,
        )
    })
}

#[cfg(target_os = "macos")]
fn decode_to_surface_with_runtime(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    pool: &mut signinum_jpeg::ScratchPool,
    fmt: PixelFormat,
    fast444_packet: Option<&JpegMetalFast444PacketV1>,
    fast422_packet: Option<&JpegMetalFast422PacketV1>,
    fast420_packet: Option<&JpegMetalFast420PacketV1>,
) -> Result<Surface, Error> {
    if let Some(surface) = try_decode_fast444_to_surface(runtime, decoder, fast444_packet, fmt)? {
        return Ok(surface);
    }
    if let Some(surface) = try_decode_fast422_to_surface(runtime, fast422_packet, fmt)? {
        return Ok(surface);
    }
    if let Some(surface) = try_decode_fast420_to_surface(runtime, decoder, fast420_packet, fmt)? {
        return Ok(surface);
    }
    let mut stage = PlaneStage::new(
        &runtime.device,
        decoder.info().color_space,
        decoder.info().dimensions,
    )?;
    decoder.decode_component_rows_with_scratch(pool, &mut stage)?;
    stage.finish_with_runtime(runtime, fmt)
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut signinum_jpeg::ScratchPool,
    fmt: PixelFormat,
    roi: signinum_jpeg::Rect,
    fast444_packet: Option<&JpegMetalFast444PacketV1>,
    fast422_packet: Option<&JpegMetalFast422PacketV1>,
    fast420_packet: Option<&JpegMetalFast420PacketV1>,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        if let Some(surface) =
            try_decode_fast444_region_to_surface(runtime, decoder, fast444_packet, fmt, roi)?
        {
            return Ok(surface);
        }
        if let Some(surface) =
            try_decode_fast422_region_to_surface(runtime, fast422_packet, fmt, roi)?
        {
            return Ok(surface);
        }
        if let Some(surface) =
            try_decode_fast420_region_to_surface(runtime, decoder, fast420_packet, fmt, roi)?
        {
            return Ok(surface);
        }
        let dims = (roi.w, roi.h);
        let mut stage = cached_plane_stage(&runtime.device, decoder.info().color_space, dims)?;
        decoder.decode_region_component_rows_with_scratch(
            pool,
            &mut stage,
            roi,
            signinum_core::Downscale::None,
        )?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_scaled_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut signinum_jpeg::ScratchPool,
    fmt: PixelFormat,
    scale: signinum_core::Downscale,
    fast444_packet: Option<&JpegMetalFast444PacketV1>,
    fast422_packet: Option<&JpegMetalFast422PacketV1>,
    fast420_packet: Option<&JpegMetalFast420PacketV1>,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        if let Some(surface) =
            try_decode_fast444_scaled_to_surface(runtime, decoder, fast444_packet, fmt, scale)?
        {
            return Ok(surface);
        }
        if let Some(surface) =
            try_decode_fast422_scaled_to_surface(runtime, fast422_packet, fmt, scale)?
        {
            return Ok(surface);
        }
        if let Some(surface) =
            try_decode_fast420_scaled_to_surface(runtime, decoder, fast420_packet, fmt, scale)?
        {
            return Ok(surface);
        }
        let full = decoder.info().dimensions;
        let roi = signinum_jpeg::Rect {
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
        let mut stage = cached_plane_stage(
            &runtime.device,
            decoder.info().color_space,
            (scaled.w, scaled.h),
        )?;
        decoder.decode_region_component_rows_with_scratch(pool, &mut stage, roi, scale)?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_region_scaled_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut signinum_jpeg::ScratchPool,
    fmt: PixelFormat,
    roi: signinum_jpeg::Rect,
    scale: signinum_core::Downscale,
    fast444_packet: Option<&JpegMetalFast444PacketV1>,
    fast422_packet: Option<&JpegMetalFast422PacketV1>,
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
            signinum_jpeg::Rect {
                x: scaled_roi.x,
                y: scaled_roi.y,
                w: scaled_roi.w,
                h: scaled_roi.h,
            },
            scale,
        )? {
            return Ok(surface);
        }
        if let Some(surface) = try_decode_fast422_scaled_region_to_surface(
            runtime,
            fast422_packet,
            fmt,
            signinum_jpeg::Rect {
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
            signinum_jpeg::Rect {
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
        let mut stage = cached_plane_stage(
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
    pool: &mut signinum_jpeg::ScratchPool,
    scale: signinum_core::Downscale,
    viewport_dims: (u32, u32),
    tiles: &[ViewportTile],
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let mut stage =
            cached_plane_stage(&runtime.device, decoder.info().color_space, viewport_dims)?;
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
                signinum_jpeg::Rect {
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
    use crate::Storage;
    use std::sync::Arc;

    const BASELINE_420: &[u8] = include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg");
    const BASELINE_420_RESTART: &[u8] =
        include_bytes!("../fixtures/jpeg/baseline_420_restart_32x16.jpg");
    const BASELINE_422: &[u8] = include_bytes!("../fixtures/jpeg/baseline_422_16x8.jpg");
    const BASELINE_444: &[u8] = include_bytes!("../fixtures/jpeg/baseline_444_8x8.jpg");

    #[test]
    fn mcu_range_for_rect_covers_only_touched_rows_and_columns() {
        let roi = signinum_jpeg::Rect {
            x: 19,
            y: 35,
            w: 11,
            h: 34,
        };

        assert_eq!(mcu_range_for_rect(roi, 8, 6, 16, 16), (17, 34));
    }

    #[test]
    fn restart_work_for_mcu_range_slices_to_overlapping_restart_segments() {
        let restart_offsets = [0, 10, 20, 30, 40, 50];

        let (restart_start_mcu, offsets) =
            restart_work_for_mcu_range(&restart_offsets, 4, 24, 9, 15);

        assert_eq!(restart_start_mcu, 8);
        assert_eq!(offsets, &[20, 30]);
    }

    #[test]
    fn runtime_initialization_error_classifies_device_unavailable() {
        assert!(matches!(
            runtime_initialization_error("Metal is unavailable on this host"),
            Error::MetalUnavailable
        ));
        assert!(matches!(
            runtime_initialization_error("failed to compile Metal library"),
            Error::MetalKernel { .. }
        ));
    }

    #[test]
    fn shader_decode_block_clears_coefficients_with_vector_stores() {
        assert!(
            SHADER_SOURCE.contains("thread short4 *coeff_chunks"),
            "decode_block should clear coeffs with packed short4 stores"
        );
        assert!(
            SHADER_SOURCE.contains("coeff_chunks[i] = short4(0);"),
            "decode_block should zero each packed coefficient chunk"
        );
    }

    #[test]
    fn shader_source_keeps_entropy_fast_paths() {
        assert!(SHADER_SOURCE.contains("inline bool refill_four_bytes("));
        assert!(
            SHADER_SOURCE.contains("return refill_four_bytes(br, bytes, len) || refill_one_byte")
        );
        assert!(SHADER_SOURCE.contains("ensure_bits_padded(br, bytes, len, 9)"));
        assert!(SHADER_SOURCE.contains("table.fast_len[fast_index]"));
        assert!(SHADER_SOURCE.contains("inline bool decode_block_skip("));
        assert!(SHADER_SOURCE.contains("skip_receive_extend(br, bytes, len, ssss, status)"));
        assert!(SHADER_SOURCE.contains("inline bool configure_batch_entropy_thread("));
    }

    #[test]
    fn shader_kernels_use_incremental_mx_my() {
        assert!(
            SHADER_SOURCE.contains("inline void init_mcu_cursor("),
            "fast decode kernels should seed mx/my via init_mcu_cursor instead of dividing per MCU"
        );
        assert!(
            SHADER_SOURCE.contains("inline void advance_mcu_cursor("),
            "fast decode kernels should carry mx/my via advance_mcu_cursor instead of dividing per MCU"
        );
        assert!(
            !SHADER_SOURCE.contains("mcu_index / params.mcus_per_row"),
            "no fast kernel should still divide mcu_index by mcus_per_row inside the MCU loop"
        );
        assert!(
            !SHADER_SOURCE.contains("mcu_index % params.mcus_per_row"),
            "no fast kernel should still modulo mcu_index by mcus_per_row inside the MCU loop"
        );
    }

    #[test]
    fn split_fast420_batch_env_requires_explicit_one() {
        assert!(split_fast420_batch_value_enabled(Some(
            std::ffi::OsStr::new("1")
        )));
        assert!(!split_fast420_batch_value_enabled(Some(
            std::ffi::OsStr::new("true")
        )));
        assert!(!split_fast420_batch_value_enabled(None));
    }

    #[test]
    fn fast420_batch_timing_env_requires_explicit_one() {
        assert!(fast420_batch_timing_value_enabled(Some(
            std::ffi::OsStr::new("1")
        )));
        assert!(!fast420_batch_timing_value_enabled(Some(
            std::ffi::OsStr::new("true")
        )));
        assert!(!fast420_batch_timing_value_enabled(None));
    }

    #[test]
    fn one_dimensional_dispatch_width_tracks_work_without_full_threadgroup_waste() {
        assert_eq!(choose_1d_threadgroup_width(32, 1024, 1), 32);
        assert_eq!(choose_1d_threadgroup_width(32, 1024, 33), 64);
        assert_eq!(choose_1d_threadgroup_width(32, 1024, 256), 256);
        assert_eq!(choose_1d_threadgroup_width(32, 1024, 257), 256);
    }

    #[test]
    fn auto_batched_packets_skip_distinct_region_scaled_requests() {
        let packet = signinum_jpeg::adapter::build_metal_fast420_packet(BASELINE_420_RESTART)
            .expect("packet");
        let roi = Rect {
            x: 0,
            y: 0,
            w: 16,
            h: 16,
        };
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::<[u8]>::from(BASELINE_420_RESTART),
                PixelFormat::Rgb8,
                BackendRequest::Auto,
                batch::BatchOp::RegionScaled {
                    roi,
                    scale: signinum_core::Downscale::Quarter,
                },
                None,
                None,
                Some(packet.clone()),
            ),
            batch::QueuedRequest::new(
                Arc::<[u8]>::from(BASELINE_420_RESTART),
                PixelFormat::Rgb8,
                BackendRequest::Auto,
                batch::BatchOp::RegionScaled {
                    roi,
                    scale: signinum_core::Downscale::Quarter,
                },
                None,
                None,
                Some(packet),
            ),
        ];

        assert!(batched_fast_packets(&requests)
            .expect("packet lookup")
            .is_none());
    }

    #[test]
    fn auto_batched_packets_keep_large_repeated_region_scaled_requests_off_metal() {
        let input = Arc::<[u8]>::from(BASELINE_420);
        let packet =
            signinum_jpeg::adapter::build_metal_fast420_packet(BASELINE_420).expect("packet");
        let roi = Rect {
            x: 0,
            y: 0,
            w: 16,
            h: 16,
        };
        let requests = (0..=REGION_SCALED_BATCH_CHUNK)
            .map(|_| {
                batch::QueuedRequest::new(
                    Arc::clone(&input),
                    PixelFormat::Rgb8,
                    BackendRequest::Auto,
                    batch::BatchOp::RegionScaled {
                        roi,
                        scale: signinum_core::Downscale::Quarter,
                    },
                    None,
                    None,
                    Some(packet.clone()),
                )
            })
            .collect::<Vec<_>>();

        assert!(batched_fast_packets(&requests)
            .expect("packet lookup")
            .is_none());
    }

    #[test]
    fn auto_batched_packets_require_wsi_batch_threshold() {
        let input = Arc::<[u8]>::from(BASELINE_420_RESTART);
        let packet = signinum_jpeg::adapter::build_metal_fast420_packet(BASELINE_420_RESTART)
            .expect("packet");
        let requests = (0..7)
            .map(|_| {
                batch::QueuedRequest::new(
                    Arc::clone(&input),
                    PixelFormat::Rgb8,
                    BackendRequest::Auto,
                    batch::BatchOp::Full,
                    None,
                    None,
                    Some(packet.clone()),
                )
            })
            .collect::<Vec<_>>();

        assert!(batched_fast_packets(&requests)
            .expect("packet lookup")
            .is_none());
    }

    #[test]
    fn auto_batched_packets_accept_restart_wsi_batch_at_threshold() {
        let input = Arc::<[u8]>::from(BASELINE_420_RESTART);
        let packet = signinum_jpeg::adapter::build_metal_fast420_packet(BASELINE_420_RESTART)
            .expect("packet");
        let requests = (0..8)
            .map(|_| {
                batch::QueuedRequest::new(
                    Arc::clone(&input),
                    PixelFormat::Rgb8,
                    BackendRequest::Auto,
                    batch::BatchOp::Full,
                    None,
                    None,
                    Some(packet.clone()),
                )
            })
            .collect::<Vec<_>>();

        assert!(batched_fast_packets(&requests)
            .expect("packet lookup")
            .is_some());
    }

    #[test]
    fn auto_batched_packets_accept_large_nonrestart_wsi_batch_at_threshold() {
        let input = Arc::<[u8]>::from(generated_rgb_jpeg(512));
        let fast444_packet =
            signinum_jpeg::adapter::build_metal_fast444_packet(input.as_ref()).ok();
        let fast422_packet =
            signinum_jpeg::adapter::build_metal_fast422_packet(input.as_ref()).ok();
        let fast420_packet =
            signinum_jpeg::adapter::build_metal_fast420_packet(input.as_ref()).ok();
        assert!(
            fast444_packet.is_some() || fast422_packet.is_some() || fast420_packet.is_some(),
            "generated JPEG must be packet-decodable"
        );
        let requests = (0..8)
            .map(|_| {
                batch::QueuedRequest::new(
                    Arc::clone(&input),
                    PixelFormat::Rgb8,
                    BackendRequest::Auto,
                    batch::BatchOp::Full,
                    fast444_packet.clone(),
                    fast422_packet.clone(),
                    fast420_packet.clone(),
                )
            })
            .collect::<Vec<_>>();

        assert!(batched_fast_packets(&requests)
            .expect("packet lookup")
            .is_some());
    }

    fn generated_rgb_jpeg(dim: u16) -> Vec<u8> {
        let mut rgb = Vec::with_capacity(dim as usize * dim as usize * 3);
        for y in 0..dim {
            for x in 0..dim {
                let xf = u32::from(x);
                let yf = u32::from(y);
                rgb.push(((xf * 13 + yf * 3) & 0xff) as u8);
                rgb.push(((xf * 5 + yf * 11 + (xf ^ yf)) & 0xff) as u8);
                rgb.push(((xf * 7 + yf * 17 + (xf.wrapping_mul(yf) >> 5)) & 0xff) as u8);
            }
        }

        let mut jpeg = Vec::new();
        let mut encoder = jpeg_encoder::Encoder::new(&mut jpeg, 90);
        encoder.set_sampling_factor(jpeg_encoder::SamplingFactor::F_2_2);
        encoder
            .encode(&rgb, dim, dim, jpeg_encoder::ColorType::Rgb)
            .expect("encode generated JPEG");
        jpeg
    }

    #[test]
    fn fast420_packet_scaled_decode_matches_cpu_scaled_bytes() {
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let packet =
            signinum_jpeg::adapter::build_metal_fast420_packet(BASELINE_420).expect("packet");
        for scale in [
            signinum_core::Downscale::Half,
            signinum_core::Downscale::Quarter,
        ] {
            let (expected, _) = decoder
                .decode_scaled(PixelFormat::Rgb8, scale)
                .expect("cpu scaled");

            let surface = with_runtime(|runtime| {
                let surface = try_decode_fast420_scaled_to_surface(
                    runtime,
                    &decoder,
                    Some(&packet),
                    PixelFormat::Rgb8,
                    scale,
                )?
                .expect("fast420 scaled surface");
                Ok::<_, Error>(surface)
            })
            .expect("metal scaled");

            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast420_packet_region_decode_matches_cpu_region_bytes() {
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let packet =
            signinum_jpeg::adapter::build_metal_fast420_packet(BASELINE_420).expect("packet");
        let roi = signinum_jpeg::Rect {
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
    fn fast420_region_batch_decode_matches_cpu_region_bytes() {
        let input = Arc::<[u8]>::from(BASELINE_420);
        let packet =
            signinum_jpeg::adapter::build_metal_fast420_packet(BASELINE_420).expect("packet");
        let roi = Rect {
            x: 4,
            y: 4,
            w: 8,
            h: 8,
        };
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Region(roi),
                None,
                None,
                Some(packet.clone()),
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Region(roi),
                None,
                None,
                Some(packet),
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let (expected, _) = decoder
            .decode_region(
                PixelFormat::Rgb8,
                signinum_jpeg::Rect {
                    x: roi.x,
                    y: roi.y,
                    w: roi.w,
                    h: roi.h,
                },
            )
            .expect("cpu region");

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("region batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (roi.w, roi.h));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast420_full_batch_decode_uses_shared_surface_offsets() {
        let input = Arc::<[u8]>::from(BASELINE_420);
        let packet =
            signinum_jpeg::adapter::build_metal_fast420_packet(BASELINE_420).expect("packet");
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                None,
                Some(packet.clone()),
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                None,
                Some(packet),
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("fast420 full batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for (index, result) in results.iter().enumerate() {
            let surface = result.as_ref().expect("surface");
            assert_eq!(surface.dimensions, (16, 16));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
            let Storage::Metal { offset, .. } = &surface.storage else {
                panic!("expected Metal storage");
            };
            assert_eq!(*offset, index * expected.len());
        }
    }

    #[test]
    fn fast420_split_full_batch_decode_matches_cpu_bytes() {
        let jpeg = generated_rgb_jpeg(32);
        let input = Arc::<[u8]>::from(jpeg.into_boxed_slice());
        let packet =
            signinum_jpeg::adapter::build_metal_fast420_packet(input.as_ref()).expect("packet");
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                None,
                Some(packet.clone()),
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                None,
                Some(packet),
            ),
        ];
        let packets = batched_fast_packets(&requests)
            .expect("packet lookup")
            .expect("packets");
        let decoder = CpuDecoder::new(input.as_ref()).expect("decoder");
        let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

        let results = with_runtime(|runtime| {
            try_decode_fast420_full_rgb_batch_to_surfaces_with_mode(
                runtime,
                &requests,
                &packets,
                Fast420BatchDecodeMode::SplitCoeffIdct,
            )
        })
        .expect("batch result")
        .expect("split fast420 full batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (32, 32));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast420_batch_clears_high_ac_before_dc_only_blocks() {
        let input = Arc::<[u8]>::from(fast420_high_ac_then_dc_only_jpeg(1));
        let packet =
            signinum_jpeg::adapter::build_metal_fast420_packet(input.as_ref()).expect("packet");
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                None,
                Some(packet.clone()),
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                None,
                Some(packet),
            ),
        ];
        let decoder = CpuDecoder::new(input.as_ref()).expect("decoder");
        let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("fast420 full batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (16, 16));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast420_batch_matches_cpu_for_high_ac_overflow_coefficients() {
        let input = Arc::<[u8]>::from(fast420_high_ac_then_dc_only_jpeg(u8::MAX));
        let packet =
            signinum_jpeg::adapter::build_metal_fast420_packet(input.as_ref()).expect("packet");
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                None,
                Some(packet.clone()),
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                None,
                Some(packet),
            ),
        ];
        let decoder = CpuDecoder::new(input.as_ref()).expect("decoder");
        let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("fast420 full batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (16, 16));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast420_packet_region_scaled_decode_matches_cpu_region_scaled_bytes() {
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let packet =
            signinum_jpeg::adapter::build_metal_fast420_packet(BASELINE_420).expect("packet");
        let roi = signinum_jpeg::Rect {
            x: 3,
            y: 2,
            w: 9,
            h: 10,
        };
        let scale = signinum_core::Downscale::Quarter;
        let (expected, _) = decoder
            .decode_region_scaled(PixelFormat::Rgb8, roi, scale)
            .expect("cpu region scaled");
        let scaled_roi = signinum_jpeg::Rect {
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
    fn fast420_region_scaled_batch_decode_matches_cpu_region_scaled_bytes() {
        let input = Arc::<[u8]>::from(BASELINE_420);
        let packet =
            signinum_jpeg::adapter::build_metal_fast420_packet(BASELINE_420).expect("packet");
        let roi = Rect {
            x: 3,
            y: 2,
            w: 9,
            h: 10,
        };
        let scale = signinum_core::Downscale::Quarter;
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::RegionScaled { roi, scale },
                None,
                None,
                Some(packet.clone()),
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::RegionScaled { roi, scale },
                None,
                None,
                Some(packet),
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let (expected, _) = decoder
            .decode_region_scaled(
                PixelFormat::Rgb8,
                signinum_jpeg::Rect {
                    x: roi.x,
                    y: roi.y,
                    w: roi.w,
                    h: roi.h,
                },
                scale,
            )
            .expect("cpu region scaled");
        let scaled = scaled_rect_covering(roi, scale);

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("region scaled batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (scaled.w, scaled.h));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast420_scaled_batch_decode_matches_cpu_scaled_bytes() {
        let input = Arc::<[u8]>::from(BASELINE_420);
        let packet =
            signinum_jpeg::adapter::build_metal_fast420_packet(BASELINE_420).expect("packet");
        let scale = signinum_core::Downscale::Quarter;
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Scaled(scale),
                None,
                None,
                Some(packet.clone()),
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Scaled(scale),
                None,
                None,
                Some(packet),
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let (expected, _) = decoder
            .decode_scaled(PixelFormat::Rgb8, scale)
            .expect("cpu scaled");

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("scaled batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (4, 4));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast422_packet_full_decode_matches_cpu_bytes() {
        let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
        let packet =
            signinum_jpeg::adapter::build_metal_fast422_packet(BASELINE_422).expect("packet");
        let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

        let surface = with_runtime(|runtime| {
            let surface = try_decode_fast422_to_surface(runtime, Some(&packet), PixelFormat::Rgb8)?
                .expect("fast422 surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal full decode");

        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    #[test]
    fn fast422_full_batch_decode_matches_cpu_bytes() {
        let input = Arc::<[u8]>::from(BASELINE_422);
        let packet =
            signinum_jpeg::adapter::build_metal_fast422_packet(BASELINE_422).expect("packet");
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                Some(packet.clone()),
                None,
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                Some(packet),
                None,
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
        let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("fast422 batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for (index, result) in results.iter().enumerate() {
            let surface = result.as_ref().expect("surface");
            assert_eq!(surface.dimensions, (16, 8));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
            let Storage::Metal { offset, .. } = &surface.storage else {
                panic!("expected Metal storage");
            };
            assert_eq!(*offset, index * expected.len());
        }
    }

    #[test]
    fn fast422_packet_region_decode_matches_cpu_region_bytes() {
        let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
        let packet =
            signinum_jpeg::adapter::build_metal_fast422_packet(BASELINE_422).expect("packet");
        let roi = signinum_jpeg::Rect {
            x: 3,
            y: 1,
            w: 9,
            h: 5,
        };
        let (expected, _) = decoder
            .decode_region(PixelFormat::Rgb8, roi)
            .expect("cpu region");

        let surface = with_runtime(|runtime| {
            let surface = try_decode_fast422_region_to_surface(
                runtime,
                Some(&packet),
                PixelFormat::Rgb8,
                roi,
            )?
            .expect("fast422 region surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal region");

        assert_eq!(surface.dimensions, (roi.w, roi.h));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    #[test]
    fn fast422_region_batch_decode_matches_cpu_region_bytes() {
        let input = Arc::<[u8]>::from(BASELINE_422);
        let packet =
            signinum_jpeg::adapter::build_metal_fast422_packet(BASELINE_422).expect("packet");
        let roi = Rect {
            x: 3,
            y: 1,
            w: 9,
            h: 5,
        };
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Region(roi),
                None,
                Some(packet.clone()),
                None,
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Region(roi),
                None,
                Some(packet),
                None,
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
        let (expected, _) = decoder
            .decode_region(
                PixelFormat::Rgb8,
                signinum_jpeg::Rect {
                    x: roi.x,
                    y: roi.y,
                    w: roi.w,
                    h: roi.h,
                },
            )
            .expect("cpu region");

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("fast422 region batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (roi.w, roi.h));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast422_packet_scaled_decode_matches_cpu_scaled_bytes() {
        let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
        let packet =
            signinum_jpeg::adapter::build_metal_fast422_packet(BASELINE_422).expect("packet");
        for (scale, dims) in [
            (signinum_core::Downscale::Half, (8, 4)),
            (signinum_core::Downscale::Quarter, (4, 2)),
        ] {
            let (expected, _) = decoder
                .decode_scaled(PixelFormat::Rgb8, scale)
                .expect("cpu scaled");

            let surface = with_runtime(|runtime| {
                let surface = try_decode_fast422_scaled_to_surface(
                    runtime,
                    Some(&packet),
                    PixelFormat::Rgb8,
                    scale,
                )?
                .expect("fast422 scaled surface");
                Ok::<_, Error>(surface)
            })
            .expect("metal scaled");

            assert_eq!(surface.dimensions, dims);
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast422_scaled_batch_decode_matches_cpu_scaled_bytes() {
        let input = Arc::<[u8]>::from(BASELINE_422);
        let packet =
            signinum_jpeg::adapter::build_metal_fast422_packet(BASELINE_422).expect("packet");
        let scale = signinum_core::Downscale::Quarter;
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Scaled(scale),
                None,
                Some(packet.clone()),
                None,
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Scaled(scale),
                None,
                Some(packet),
                None,
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
        let (expected, _) = decoder
            .decode_scaled(PixelFormat::Rgb8, scale)
            .expect("cpu scaled");

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("fast422 scaled batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (4, 2));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast422_packet_region_scaled_decode_matches_cpu_region_scaled_bytes() {
        let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
        let packet =
            signinum_jpeg::adapter::build_metal_fast422_packet(BASELINE_422).expect("packet");
        let roi = signinum_jpeg::Rect {
            x: 3,
            y: 1,
            w: 9,
            h: 5,
        };
        let scale = signinum_core::Downscale::Half;
        let (expected, _) = decoder
            .decode_region_scaled(PixelFormat::Rgb8, roi, scale)
            .expect("cpu region scaled");
        let scaled_roi = signinum_jpeg::Rect {
            x: roi.x / 2,
            y: roi.y / 2,
            w: (roi.x + roi.w).div_ceil(2) - (roi.x / 2),
            h: (roi.y + roi.h).div_ceil(2) - (roi.y / 2),
        };

        let surface = with_runtime(|runtime| {
            let surface = try_decode_fast422_scaled_region_to_surface(
                runtime,
                Some(&packet),
                PixelFormat::Rgb8,
                scaled_roi,
                scale,
            )?
            .expect("fast422 scaled region surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal region scaled");

        assert_eq!(surface.dimensions, (scaled_roi.w, scaled_roi.h));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    #[test]
    fn fast422_region_scaled_batch_decode_matches_cpu_region_scaled_bytes() {
        let input = Arc::<[u8]>::from(BASELINE_422);
        let packet =
            signinum_jpeg::adapter::build_metal_fast422_packet(BASELINE_422).expect("packet");
        let roi = Rect {
            x: 3,
            y: 1,
            w: 9,
            h: 5,
        };
        let scale = signinum_core::Downscale::Half;
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::RegionScaled { roi, scale },
                None,
                Some(packet.clone()),
                None,
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::RegionScaled { roi, scale },
                None,
                Some(packet),
                None,
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
        let (expected, _) = decoder
            .decode_region_scaled(
                PixelFormat::Rgb8,
                signinum_jpeg::Rect {
                    x: roi.x,
                    y: roi.y,
                    w: roi.w,
                    h: roi.h,
                },
                scale,
            )
            .expect("cpu region scaled");
        let scaled = scaled_rect_covering(roi, scale);

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("fast422 region scaled batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (scaled.w, scaled.h));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast444_packet_full_decode_matches_cpu_bytes() {
        let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
        let packet =
            signinum_jpeg::adapter::build_metal_fast444_packet(BASELINE_444).expect("packet");
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
            signinum_jpeg::adapter::build_metal_fast444_packet(BASELINE_444).expect("packet");
        for scale in [
            signinum_core::Downscale::Half,
            signinum_core::Downscale::Quarter,
        ] {
            let (expected, _) = decoder
                .decode_scaled(PixelFormat::Rgb8, scale)
                .expect("cpu scaled");

            let surface = with_runtime(|runtime| {
                let surface = try_decode_fast444_scaled_to_surface(
                    runtime,
                    &decoder,
                    Some(&packet),
                    PixelFormat::Rgb8,
                    scale,
                )?
                .expect("fast444 scaled surface");
                Ok::<_, Error>(surface)
            })
            .expect("metal scaled");

            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast444_packet_region_decode_matches_cpu_region_bytes() {
        let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
        let packet =
            signinum_jpeg::adapter::build_metal_fast444_packet(BASELINE_444).expect("packet");
        let roi = signinum_jpeg::Rect {
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
    fn fast444_region_batch_decode_matches_cpu_region_bytes() {
        let input = Arc::<[u8]>::from(BASELINE_444);
        let packet =
            signinum_jpeg::adapter::build_metal_fast444_packet(BASELINE_444).expect("packet");
        let roi = Rect {
            x: 1,
            y: 2,
            w: 5,
            h: 4,
        };
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Region(roi),
                Some(packet.clone()),
                None,
                None,
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Region(roi),
                Some(packet),
                None,
                None,
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
        let (expected, _) = decoder
            .decode_region(
                PixelFormat::Rgb8,
                signinum_jpeg::Rect {
                    x: roi.x,
                    y: roi.y,
                    w: roi.w,
                    h: roi.h,
                },
            )
            .expect("cpu region");

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("region batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (roi.w, roi.h));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast444_packet_region_scaled_decode_matches_cpu_region_scaled_bytes() {
        let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
        let packet =
            signinum_jpeg::adapter::build_metal_fast444_packet(BASELINE_444).expect("packet");
        let roi = signinum_jpeg::Rect {
            x: 1,
            y: 2,
            w: 5,
            h: 4,
        };
        let scale = signinum_core::Downscale::Quarter;
        let (expected, _) = decoder
            .decode_region_scaled(PixelFormat::Rgb8, roi, scale)
            .expect("cpu region scaled");
        let scaled_roi = signinum_jpeg::Rect {
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

    #[test]
    fn fast444_region_scaled_batch_decode_matches_cpu_region_scaled_bytes() {
        let input = Arc::<[u8]>::from(BASELINE_444);
        let packet =
            signinum_jpeg::adapter::build_metal_fast444_packet(BASELINE_444).expect("packet");
        let roi = Rect {
            x: 1,
            y: 2,
            w: 5,
            h: 4,
        };
        let scale = signinum_core::Downscale::Quarter;
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::RegionScaled { roi, scale },
                Some(packet.clone()),
                None,
                None,
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::RegionScaled { roi, scale },
                Some(packet),
                None,
                None,
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
        let (expected, _) = decoder
            .decode_region_scaled(
                PixelFormat::Rgb8,
                signinum_jpeg::Rect {
                    x: roi.x,
                    y: roi.y,
                    w: roi.w,
                    h: roi.h,
                },
                scale,
            )
            .expect("cpu region scaled");
        let scaled = scaled_rect_covering(roi, scale);

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("region scaled batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (scaled.w, scaled.h));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast444_scaled_batch_decode_matches_cpu_scaled_bytes() {
        let input = Arc::<[u8]>::from(BASELINE_444);
        let packet =
            signinum_jpeg::adapter::build_metal_fast444_packet(BASELINE_444).expect("packet");
        let scale = signinum_core::Downscale::Quarter;
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Scaled(scale),
                Some(packet.clone()),
                None,
                None,
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Scaled(scale),
                Some(packet),
                None,
                None,
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
        let (expected, _) = decoder
            .decode_scaled(PixelFormat::Rgb8, scale)
            .expect("cpu scaled");

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("scaled batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (2, 2));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    fn fast420_high_ac_then_dc_only_jpeg(ac_quant: u8) -> Vec<u8> {
        assert!(ac_quant > 0, "JPEG quant entries must be nonzero");

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&[0xff, 0xd8]);

        let mut quant = [1u8; 64];
        quant[63] = ac_quant;
        let mut dqt = Vec::with_capacity(65);
        dqt.push(0x00);
        dqt.extend_from_slice(&quant);
        append_jpeg_segment(&mut bytes, 0xdb, &dqt);

        append_jpeg_segment(
            &mut bytes,
            0xc0,
            &[
                8,
                0,
                16,
                0,
                16,
                3,
                1,
                (2 << 4) | 2,
                0,
                2,
                (1 << 4) | 1,
                0,
                3,
                (1 << 4) | 1,
                0,
            ],
        );

        let mut dc_bits = [0u8; 16];
        dc_bits[0] = 1;
        let mut dht_dc = Vec::with_capacity(18);
        dht_dc.push(0x00);
        dht_dc.extend_from_slice(&dc_bits);
        dht_dc.push(0x00);
        append_jpeg_segment(&mut bytes, 0xc4, &dht_dc);

        let mut ac_bits = [0u8; 16];
        ac_bits[1] = 3;
        let mut dht_ac = Vec::with_capacity(20);
        dht_ac.push(0x10);
        dht_ac.extend_from_slice(&ac_bits);
        dht_ac.extend_from_slice(&[0x00, 0xf0, 0xea]);
        append_jpeg_segment(&mut bytes, 0xc4, &dht_ac);

        append_jpeg_segment(&mut bytes, 0xda, &[3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0]);

        bytes.extend_from_slice(&fast420_high_ac_entropy());
        bytes.extend_from_slice(&[0xff, 0xd9]);
        bytes
    }

    fn append_jpeg_segment(bytes: &mut Vec<u8>, marker: u8, payload: &[u8]) {
        bytes.extend_from_slice(&[0xff, marker]);
        let len = u16::try_from(payload.len() + 2).expect("JPEG segment length fits in u16");
        bytes.extend_from_slice(&len.to_be_bytes());
        bytes.extend_from_slice(payload);
    }

    fn fast420_high_ac_entropy() -> Vec<u8> {
        let mut writer = EntropyBitWriter::default();
        emit_high_ac_block(&mut writer);
        for _ in 0..5 {
            emit_dc_only_block(&mut writer);
        }
        writer.finish()
    }

    fn emit_high_ac_block(writer: &mut EntropyBitWriter) {
        writer.push_bits(0, 1);
        for _ in 0..3 {
            writer.push_bits(0b01, 2);
        }
        writer.push_bits(0b10, 2);
        writer.push_bits(0b11_1111_1111, 10);
    }

    fn emit_dc_only_block(writer: &mut EntropyBitWriter) {
        writer.push_bits(0, 1);
        writer.push_bits(0b00, 2);
    }

    #[derive(Default)]
    struct EntropyBitWriter {
        bytes: Vec<u8>,
        current: u8,
        bit_count: u8,
    }

    impl EntropyBitWriter {
        fn push_bits(&mut self, bits: u16, len: u8) {
            for shift in (0..len).rev() {
                let bit = u8::from(((bits >> shift) & 1) != 0);
                self.current = (self.current << 1) | bit;
                self.bit_count += 1;
                if self.bit_count == 8 {
                    self.push_current_byte();
                }
            }
        }

        fn finish(mut self) -> Vec<u8> {
            if self.bit_count != 0 {
                let pad_bits = 8 - self.bit_count;
                self.current = (self.current << pad_bits) | ((1u8 << pad_bits) - 1);
                self.push_current_byte();
            }
            self.bytes
        }

        fn push_current_byte(&mut self) {
            self.bytes.push(self.current);
            if self.current == 0xff {
                self.bytes.push(0x00);
            }
            self.current = 0;
            self.bit_count = 0;
        }
    }
}
