// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use std::{
    cell::RefCell,
    collections::HashMap,
    mem::{size_of, size_of_val},
    sync::{Arc, Mutex, OnceLock},
};

#[cfg(target_os = "macos")]
use metal::{
    foreign_types::ForeignType,
    objc::{runtime::Sel, Message},
    Buffer, CommandBufferRef, CommandQueue, CompileOptions, ComputeCommandEncoderRef,
    ComputePipelineState, Device, MTLCommandQueue, MTLResourceOptions, MTLSize,
};
use signinum_core::{PixelFormat, Rect};
#[cfg(target_os = "macos")]
use signinum_j2k_native::HtCodeBlockDecoder;
use signinum_j2k_native::{
    ht_uvlc_encode_table, ht_uvlc_table0, ht_uvlc_table1, ht_vlc_encode_table0,
    ht_vlc_encode_table1, ht_vlc_table0, ht_vlc_table1, ColorSpace as NativeColorSpace,
    DecodedComponents as NativeDecodedComponents, EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock,
    HtCodeBlockDecodeJob, HtSubBandDecodeJob, J2kCodeBlockDecodeJob, J2kCodeBlockSegment,
    J2kDirectBandId, J2kDirectColorPlan, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep,
    J2kDirectIdwtStep, J2kDirectStoreStep, J2kForwardDwt53Level, J2kForwardDwt53Output,
    J2kHtCodeBlockEncodeJob, J2kInverseMctJob, J2kPacketizationBlockCodingMode,
    J2kPacketizationEncodeJob, J2kSingleDecompositionIdwtJob, J2kStoreComponentJob,
    J2kSubBandDecodeJob, J2kTier1CodeBlockEncodeJob, J2kWaveletTransform,
};
#[cfg(target_os = "macos")]
use signinum_j2k_native::{
    DecodeSettings as NativeDecodeSettings, DecoderContext as NativeDecoderContext,
    Image as NativeImage,
};

#[cfg(target_os = "macos")]
use crate::{
    classic::MetalClassicBlockDecoder, ht::MetalHtBlockDecoder, idwt::MetalIdwtDecoder,
    mct::MetalMctDecoder, store::MetalStoreDecoder,
};
use crate::{Error, Surface};

#[cfg(target_os = "macos")]
#[derive(Default)]
struct MetalCodeBlockDecoder {
    classic: MetalClassicBlockDecoder,
    ht: MetalHtBlockDecoder,
    idwt: MetalIdwtDecoder,
    mct: MetalMctDecoder,
    store: MetalStoreDecoder,
}

#[cfg(target_os = "macos")]
impl HtCodeBlockDecoder for MetalCodeBlockDecoder {
    fn decode_j2k_sub_band(
        &mut self,
        job: J2kSubBandDecodeJob<'_>,
        output: &mut [f32],
    ) -> signinum_j2k_native::Result<bool> {
        self.classic.decode_j2k_sub_band(job, output)
    }

    fn decode_j2k_code_block(
        &mut self,
        job: signinum_j2k_native::J2kCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> signinum_j2k_native::Result<bool> {
        self.classic.decode_j2k_code_block(job, output)
    }

    fn decode_sub_band(
        &mut self,
        job: HtSubBandDecodeJob<'_>,
        output: &mut [f32],
    ) -> signinum_j2k_native::Result<bool> {
        self.ht.decode_sub_band(job, output)
    }

    fn decode_code_block(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> signinum_j2k_native::Result<()> {
        self.ht.decode_code_block(job, output)
    }

    fn decode_single_decomposition_idwt(
        &mut self,
        job: J2kSingleDecompositionIdwtJob<'_>,
        output: &mut [f32],
    ) -> signinum_j2k_native::Result<bool> {
        self.idwt.decode_single_decomposition_idwt(job, output)
    }

    fn decode_inverse_mct(
        &mut self,
        job: J2kInverseMctJob<'_>,
    ) -> signinum_j2k_native::Result<bool> {
        self.mct.decode_inverse_mct(job)
    }

    fn decode_store_component(
        &mut self,
        job: J2kStoreComponentJob<'_>,
    ) -> signinum_j2k_native::Result<bool> {
        self.store.decode_store_component(job)
    }
}

#[cfg(target_os = "macos")]
const SHADER_SOURCE: &str = concat!(
    r#"
#include <metal_stdlib>
using namespace metal;

kernel void j2k_zero_u32_buffer(
    device uint *buffer [[buffer(0)]],
    constant uint &word_count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= word_count) {
        return;
    }

    buffer[gid] = 0u;
}

struct J2kValidateBytesParams {
    uint byte_len;
};

struct J2kValidateBytesStatus {
    uint code;
    uint index;
    uint expected;
    uint actual;
};

kernel void j2k_validate_bytes_equal(
    device const uchar *actual [[buffer(0)]],
    device const uchar *expected [[buffer(1)]],
    device J2kValidateBytesStatus *status [[buffer(2)]],
    constant J2kValidateBytesParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }

    status[0].code = 0u;
    status[0].index = 0u;
    status[0].expected = 0u;
    status[0].actual = 0u;

    for (uint i = 0u; i < params.byte_len; ++i) {
        const uchar actual_byte = actual[i];
        const uchar expected_byte = expected[i];
        if (actual_byte != expected_byte) {
            status[0].code = 1u;
            status[0].index = i;
            status[0].expected = uint(expected_byte);
            status[0].actual = uint(actual_byte);
            return;
        }
    }
}

struct J2kCopyInterleavedParams {
    uint src_width;
    uint src_height;
    uint src_stride;
    uint dst_width;
    uint dst_height;
    uint dst_stride;
    uint bytes_per_pixel;
};

kernel void j2k_copy_interleaved_padded(
    device const uchar *src [[buffer(0)]],
    device uchar *dst [[buffer(1)]],
    constant J2kCopyInterleavedParams &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
        return;
    }

    const uint dst_idx = gid.y * params.dst_stride + gid.x * params.bytes_per_pixel;
    const bool inside_src = gid.x < params.src_width && gid.y < params.src_height;
    const uint src_idx = gid.y * params.src_stride + gid.x * params.bytes_per_pixel;
    for (uint byte_idx = 0u; byte_idx < params.bytes_per_pixel; ++byte_idx) {
        dst[dst_idx + byte_idx] = inside_src ? src[src_idx + byte_idx] : uchar(0);
    }
}

struct J2kPackParams {
    uint width;
    uint height;
    uint out_stride;
    uint output_channels;
    uint opaque_alpha;
    float max_values[4];
    float u8_scales[4];
    float u16_scales[4];
};

struct J2kMctRgb8PackParams {
    uint width;
    uint height;
    uint out_stride;
    uint transform;
    float addends[3];
    float max_values[3];
    float u8_scales[3];
};

struct J2kBatchedMctRgb8PackParams {
    uint width;
    uint height;
    uint out_stride;
    uint transform;
    uint batch_count;
    uint plane_stride;
    uint output_stride;
    float addends[3];
    float max_values[3];
    float u8_scales[3];
};

inline uchar scale_to_u8(float sample, float max_value, float scale) {
    const float clamped = clamp(sample, 0.0f, max_value);
    return uchar(min(floor(clamped * scale + 0.5f), 255.0f));
}

inline ushort pack_to_u16(float sample, float max_value, float scale) {
    const float clamped = clamp(sample, 0.0f, max_value);
    return ushort(min(floor(clamped * scale + 0.5f), 65535.0f));
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
        out[out_idx] = scale_to_u8(plane0[idx], params.max_values[0], params.u8_scales[0]);
        return;
    }

    out[out_idx] = scale_to_u8(plane0[idx], params.max_values[0], params.u8_scales[0]);
    out[out_idx + 1] = scale_to_u8(plane1[idx], params.max_values[1], params.u8_scales[1]);
    out[out_idx + 2] = scale_to_u8(plane2[idx], params.max_values[2], params.u8_scales[2]);

    if (params.output_channels == 4) {
        out[out_idx + 3] = params.opaque_alpha != 0
            ? uchar(255)
            : scale_to_u8(plane3[idx], params.max_values[3], params.u8_scales[3]);
    }
}

kernel void j2k_pack_gray8(
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
    const uint out_idx = gid.y * params.out_stride + gid.x;
    out[out_idx] = scale_to_u8(plane0[idx], params.max_values[0], params.u8_scales[0]);
}

kernel void j2k_pack_rgb8(
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
    const uint out_idx = gid.y * params.out_stride + gid.x * 3u;
    out[out_idx] = scale_to_u8(plane0[idx], params.max_values[0], params.u8_scales[0]);
    out[out_idx + 1] = scale_to_u8(plane1[idx], params.max_values[1], params.u8_scales[1]);
    out[out_idx + 2] = scale_to_u8(plane2[idx], params.max_values[2], params.u8_scales[2]);
}

kernel void j2k_pack_mct_rgb8(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant J2kMctRgb8PackParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    const float y0 = plane0[idx];
    const float y1 = plane1[idx];
    const float y2 = plane2[idx];
    float rgb0;
    float rgb1;
    float rgb2;

    if (params.transform == 0u) {
        const float i1 = y0 - floor((y2 + y1) * 0.25f);
        rgb0 = y2 + i1 + params.addends[0];
        rgb1 = i1 + params.addends[1];
        rgb2 = y1 + i1 + params.addends[2];
    } else {
        rgb0 = y2 * 1.402f + y0 + params.addends[0];
        rgb1 = y2 * -0.71414f + y1 * -0.34413f + y0 + params.addends[1];
        rgb2 = y1 * 1.772f + y0 + params.addends[2];
    }

    const uint out_idx = gid.y * params.out_stride + gid.x * 3u;
    out[out_idx] = scale_to_u8(rgb0, params.max_values[0], params.u8_scales[0]);
    out[out_idx + 1] = scale_to_u8(rgb1, params.max_values[1], params.u8_scales[1]);
    out[out_idx + 2] = scale_to_u8(rgb2, params.max_values[2], params.u8_scales[2]);
}

kernel void j2k_pack_mct_rgb8_batched(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant J2kBatchedMctRgb8PackParams &params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.batch_count) {
        return;
    }

    const uint plane_base = gid.z * params.plane_stride;
    const uint idx = plane_base + gid.y * params.width + gid.x;
    const float y0 = plane0[idx];
    const float y1 = plane1[idx];
    const float y2 = plane2[idx];
    float rgb0;
    float rgb1;
    float rgb2;

    if (params.transform == 0u) {
        const float i1 = y0 - floor((y2 + y1) * 0.25f);
        rgb0 = y2 + i1 + params.addends[0];
        rgb1 = i1 + params.addends[1];
        rgb2 = y1 + i1 + params.addends[2];
    } else {
        rgb0 = y2 * 1.402f + y0 + params.addends[0];
        rgb1 = y2 * -0.71414f + y1 * -0.34413f + y0 + params.addends[1];
        rgb2 = y1 * 1.772f + y0 + params.addends[2];
    }

    const uint out_idx = gid.z * params.output_stride + gid.y * params.out_stride + gid.x * 3u;
    out[out_idx] = scale_to_u8(rgb0, params.max_values[0], params.u8_scales[0]);
    out[out_idx + 1] = scale_to_u8(rgb1, params.max_values[1], params.u8_scales[1]);
    out[out_idx + 2] = scale_to_u8(rgb2, params.max_values[2], params.u8_scales[2]);
}

kernel void j2k_pack_rgb_opaque_rgba8(
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
    const uint out_idx = gid.y * params.out_stride + gid.x * 4u;
    out[out_idx] = scale_to_u8(plane0[idx], params.max_values[0], params.u8_scales[0]);
    out[out_idx + 1] = scale_to_u8(plane1[idx], params.max_values[1], params.u8_scales[1]);
    out[out_idx + 2] = scale_to_u8(plane2[idx], params.max_values[2], params.u8_scales[2]);
    out[out_idx + 3] = uchar(255);
}

kernel void j2k_pack_rgba8(
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
    const uint out_idx = gid.y * params.out_stride + gid.x * 4u;
    out[out_idx] = scale_to_u8(plane0[idx], params.max_values[0], params.u8_scales[0]);
    out[out_idx + 1] = scale_to_u8(plane1[idx], params.max_values[1], params.u8_scales[1]);
    out[out_idx + 2] = scale_to_u8(plane2[idx], params.max_values[2], params.u8_scales[2]);
    out[out_idx + 3] = scale_to_u8(plane3[idx], params.max_values[3], params.u8_scales[3]);
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
        out[out_idx] = pack_to_u16(plane0[idx], params.max_values[0], params.u16_scales[0]);
        return;
    }

    out[out_idx] = pack_to_u16(plane0[idx], params.max_values[0], params.u16_scales[0]);
    out[out_idx + 1] = pack_to_u16(plane1[idx], params.max_values[1], params.u16_scales[1]);
    out[out_idx + 2] = pack_to_u16(plane2[idx], params.max_values[2], params.u16_scales[2]);
}

kernel void j2k_pack_gray16(
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
    const uint out_idx = (gid.y * params.out_stride) / 2u + gid.x;
    out[out_idx] = pack_to_u16(plane0[idx], params.max_values[0], params.u16_scales[0]);
}

kernel void j2k_pack_rgb16(
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
    const uint out_idx = (gid.y * params.out_stride) / 2u + gid.x * 3u;
    out[out_idx] = pack_to_u16(plane0[idx], params.max_values[0], params.u16_scales[0]);
    out[out_idx + 1] = pack_to_u16(plane1[idx], params.max_values[1], params.u16_scales[1]);
    out[out_idx + 2] = pack_to_u16(plane2[idx], params.max_values[2], params.u16_scales[2]);
}

struct J2kRepeatedGrayPackParams {
    uint width;
    uint height;
    uint out_stride;
    uint batch_count;
    float max_value;
    float u8_scale;
    float u16_scale;
};

kernel void j2k_pack_u8_repeated_gray(
    device const float *plane0 [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    constant J2kRepeatedGrayPackParams &params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.batch_count) {
        return;
    }

    const uint plane_base = gid.z * params.width * params.height;
    const uint out_base = gid.z * params.out_stride * params.height;
    const uint plane_idx = plane_base + gid.y * params.width + gid.x;
    const uint out_idx = out_base + gid.y * params.out_stride + gid.x;
    out[out_idx] = scale_to_u8(plane0[plane_idx], params.max_value, params.u8_scale);
}

kernel void j2k_pack_u16_repeated_gray(
    device const float *plane0 [[buffer(0)]],
    device ushort *out [[buffer(1)]],
    constant J2kRepeatedGrayPackParams &params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.batch_count) {
        return;
    }

    const uint plane_base = gid.z * params.width * params.height;
    const uint out_base = (gid.z * params.out_stride * params.height) / 2u;
    const uint plane_idx = plane_base + gid.y * params.width + gid.x;
    const uint out_idx = out_base + gid.y * (params.out_stride / 2u) + gid.x;
    out[out_idx] = pack_to_u16(plane0[plane_idx], params.max_value, params.u16_scale);
}
"#,
    "\n",
    include_str!("classic.metal"),
    "\n",
    include_str!("encode_bitstream.metal"),
    "\n",
    include_str!("idwt.metal"),
    "\n",
    include_str!("fdwt.metal"),
    "\n",
    include_str!("mct.metal"),
    "\n",
    include_str!("store.metal"),
    "\n",
    include_str!("ht_cleanup.metal"),
);

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kValidateBytesParams {
    byte_len: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kValidateBytesStatus {
    code: u32,
    index: u32,
    expected: u32,
    actual: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kCopyInterleavedParams {
    src_width: u32,
    src_height: u32,
    src_stride: u32,
    dst_width: u32,
    dst_height: u32,
    dst_stride: u32,
    bytes_per_pixel: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kPackParams {
    width: u32,
    height: u32,
    out_stride: u32,
    output_channels: u32,
    opaque_alpha: u32,
    max_values: [f32; 4],
    u8_scales: [f32; 4],
    u16_scales: [f32; 4],
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kMctRgb8PackParams {
    width: u32,
    height: u32,
    out_stride: u32,
    transform: u32,
    addends: [f32; 3],
    max_values: [f32; 3],
    u8_scales: [f32; 3],
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kBatchedMctRgb8PackParams {
    width: u32,
    height: u32,
    out_stride: u32,
    transform: u32,
    batch_count: u32,
    plane_stride: u32,
    output_stride: u32,
    addends: [f32; 3],
    max_values: [f32; 3],
    u8_scales: [f32; 3],
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kRepeatedGrayPackParams {
    width: u32,
    height: u32,
    out_stride: u32,
    batch_count: u32,
    max_value: f32,
    u8_scale: f32,
    u16_scale: f32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct J2kScalarPackParams {
    max_value: f32,
    u8_scale: f32,
    u16_scale: f32,
}

#[cfg(target_os = "macos")]
fn j2k_scalar_pack_params(bit_depth: u32) -> J2kScalarPackParams {
    let clamped = bit_depth.min(16);
    let max_value_u16 = u16::try_from(((1u32 << clamped) - 1).max(1))
        .expect("clamped J2K bit depth max fits in u16");
    let max_value = f32::from(max_value_u16);
    let u8_scale = 255.0 / max_value;
    let u16_scale = if bit_depth <= 8 {
        65_535.0 / max_value
    } else {
        1.0
    };
    J2kScalarPackParams {
        max_value,
        u8_scale,
        u16_scale,
    }
}

#[cfg(target_os = "macos")]
fn j2k_pack_scale_arrays(bit_depths: [u32; 4]) -> ([f32; 4], [f32; 4], [f32; 4]) {
    let mut max_values = [1.0f32; 4];
    let mut u8_scales = [255.0f32; 4];
    let mut u16_scales = [65_535.0f32; 4];
    for (index, bit_depth) in bit_depths.into_iter().enumerate() {
        let params = j2k_scalar_pack_params(bit_depth);
        max_values[index] = params.max_value;
        u8_scales[index] = params.u8_scale;
        u16_scales[index] = params.u16_scale;
    }
    (max_values, u8_scales, u16_scales)
}

#[cfg(target_os = "macos")]
const J2K_CLASSIC_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STATUS_FAIL: u32 = 1;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STATUS_UNSUPPORTED: u32 = 2;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES: u32 = 1 << 0;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS: u32 = 1 << 1;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT: u32 = 1 << 2;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS: u32 = 1 << 3;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS: u32 = 1 << 4;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_MAX_WIDTH: u32 = 64;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_MAX_HEIGHT: u32 = 64;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_MAX_COEFF_COUNT: usize =
    (J2K_CLASSIC_MAX_WIDTH as usize + 2) * (J2K_CLASSIC_MAX_HEIGHT as usize + 2);

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kClassicCleanupBatchJob {
    coded_offset: u32,
    coded_len: u32,
    segment_offset: u32,
    segment_count: u32,
    width: u32,
    height: u32,
    output_stride: u32,
    output_offset: u32,
    missing_msbs: u32,
    total_bitplanes: u32,
    number_of_coding_passes: u32,
    sub_band_type: u32,
    style_flags: u32,
    strict: u32,
    dequantization_step: f32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kClassicSegment {
    data_offset: u32,
    data_length: u32,
    start_coding_pass: u32,
    end_coding_pass: u32,
    use_arithmetic: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kClassicStatus {
    code: u32,
    detail: u32,
    reserved0: u32,
    reserved1: u32,
}

#[cfg(target_os = "macos")]
const J2K_IDWT_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
const J2K_IDWT_STATUS_FAIL: u32 = 1;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kIdwtSingleDecompositionParams {
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
    ll_width: u32,
    ll_height: u32,
    hl_width: u32,
    hl_height: u32,
    lh_width: u32,
    lh_height: u32,
    hh_width: u32,
    hh_height: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kRepeatedIdwtSingleDecompositionParams {
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
    ll_width: u32,
    ll_height: u32,
    hl_width: u32,
    hl_height: u32,
    lh_width: u32,
    lh_height: u32,
    hh_width: u32,
    hh_height: u32,
    ll_instance_stride: u32,
    hl_instance_stride: u32,
    lh_instance_stride: u32,
    hh_instance_stride: u32,
    batch_count: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kIdwtStatus {
    code: u32,
    detail: u32,
    reserved0: u32,
    reserved1: u32,
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
const J2K_MCT_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
#[allow(dead_code)]
const J2K_MCT_STATUS_FAIL: u32 = 1;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct J2kInverseMctParams {
    len: u32,
    transform: u32,
    addend0: f32,
    addend1: f32,
    addend2: f32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct J2kForwardRctParams {
    len: u32,
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kForwardDwt53Params {
    full_width: u32,
    current_width: u32,
    current_height: u32,
    low_width: u32,
    low_height: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
#[allow(dead_code)]
struct J2kMctStatus {
    code: u32,
    detail: u32,
    reserved0: u32,
    reserved1: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kStoreParams {
    input_width: u32,
    source_x: u32,
    source_y: u32,
    copy_width: u32,
    copy_height: u32,
    output_width: u32,
    output_x: u32,
    output_y: u32,
    addend: f32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kRepeatedStoreParams {
    input_width: u32,
    input_height: u32,
    source_x: u32,
    source_y: u32,
    copy_width: u32,
    copy_height: u32,
    output_width: u32,
    output_height: u32,
    output_x: u32,
    output_y: u32,
    addend: f32,
    batch_count: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kRepeatedGrayStoreParams {
    input_width: u32,
    input_height: u32,
    source_x: u32,
    source_y: u32,
    copy_width: u32,
    copy_height: u32,
    output_width: u32,
    output_height: u32,
    output_x: u32,
    output_y: u32,
    addend: f32,
    batch_count: u32,
    max_value: f32,
    u8_scale: f32,
    u16_scale: f32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kGrayStoreParams {
    input_width: u32,
    source_x: u32,
    source_y: u32,
    copy_width: u32,
    copy_height: u32,
    output_width: u32,
    output_x: u32,
    output_y: u32,
    addend: f32,
    max_value: f32,
    u8_scale: f32,
    u16_scale: f32,
}

const J2K_HT_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
const J2K_HT_STATUS_FAIL: u32 = 1;
#[cfg(target_os = "macos")]
const J2K_HT_STATUS_UNSUPPORTED: u32 = 2;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtCleanupParams {
    width: u32,
    height: u32,
    coded_len: u32,
    cleanup_length: u32,
    refinement_length: u32,
    missing_msbs: u32,
    num_bitplanes: u32,
    number_of_coding_passes: u32,
    output_stride: u32,
    output_offset: u32,
    dequantization_step: f32,
    stripe_causal: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtCleanupBatchJob {
    coded_offset: u32,
    width: u32,
    height: u32,
    coded_len: u32,
    cleanup_length: u32,
    refinement_length: u32,
    missing_msbs: u32,
    num_bitplanes: u32,
    number_of_coding_passes: u32,
    output_stride: u32,
    output_offset: u32,
    dequantization_step: f32,
    stripe_causal: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtRepeatedBatchParams {
    job_count: u32,
    output_plane_len: u32,
    batch_count: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kClassicRepeatedBatchParams {
    job_count: u32,
    output_plane_len: u32,
    batch_count: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kHtStatus {
    code: u32,
    detail: u32,
    reserved0: u32,
    reserved1: u32,
}

#[cfg(target_os = "macos")]
const J2K_ENCODE_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
const J2K_ENCODE_STATUS_FAIL: u32 = 1;
#[cfg(target_os = "macos")]
const J2K_ENCODE_STATUS_UNSUPPORTED: u32 = 2;
#[cfg(target_os = "macos")]
const J2K_HT_ENCODE_MEL_SIZE: usize = 192;
#[cfg(target_os = "macos")]
const J2K_HT_ENCODE_VLC_SIZE: usize = 3072 - J2K_HT_ENCODE_MEL_SIZE;
#[cfg(target_os = "macos")]
const J2K_HT_ENCODE_MS_SIZE: usize = (16_384usize * 16).div_ceil(15);

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kClassicEncodeParams {
    width: u32,
    height: u32,
    sub_band_type: u32,
    total_bitplanes: u32,
    style_flags: u32,
    output_capacity: u32,
    segment_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kClassicEncodeBatchJob {
    coefficient_offset: u32,
    output_offset: u32,
    segment_offset: u32,
    width: u32,
    height: u32,
    sub_band_type: u32,
    total_bitplanes: u32,
    style_flags: u32,
    output_capacity: u32,
    segment_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kClassicEncodeStatus {
    code: u32,
    detail: u32,
    data_len: u32,
    number_of_coding_passes: u32,
    missing_bit_planes: u32,
    segment_count: u32,
    reserved0: u32,
    reserved1: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtEncodeParams {
    width: u32,
    height: u32,
    total_bitplanes: u32,
    output_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kHtEncodeBatchJob {
    coefficient_offset: u32,
    output_offset: u32,
    width: u32,
    height: u32,
    total_bitplanes: u32,
    output_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kHtEncodeStatus {
    code: u32,
    detail: u32,
    data_len: u32,
    num_coding_passes: u32,
    num_zero_bitplanes: u32,
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kPacketEncodeParams {
    resolution_count: u32,
    num_layers: u32,
    num_components: u32,
    code_block_count: u32,
    subband_count: u32,
    descriptor_count: u32,
    output_capacity: u32,
    header_capacity: u32,
    scratch_node_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kPacketDescriptor {
    packet_index: u32,
    state_index: u32,
    layer: u32,
    resolution: u32,
    component: u32,
    precinct_lo: u32,
    precinct_hi: u32,
    state_block_offset: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kPacketResolution {
    subband_offset: u32,
    subband_count: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kPacketSubband {
    block_offset: u32,
    block_count: u32,
    num_cbs_x: u32,
    num_cbs_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kPacketBlock {
    data_offset: u32,
    data_len: u32,
    num_coding_passes: u32,
    num_zero_bitplanes: u32,
    previously_included: u32,
    l_block: u32,
    block_coding_mode: u32,
    reserved0: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kPacketStateBlock {
    previously_included: u32,
    l_block: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kPacketEncodeStatus {
    code: u32,
    detail: u32,
    data_len: u32,
    reserved0: u32,
}

#[cfg(target_os = "macos")]
static METAL_RUNTIME: OnceLock<Result<Arc<MetalRuntime>, String>> = OnceLock::new();

#[cfg(target_os = "macos")]
thread_local! {
    static METAL_RUNTIME_OVERRIDE: RefCell<Option<Arc<MetalRuntime>>> = const { RefCell::new(None) };
}

#[cfg(target_os = "macos")]
struct MetalRuntime {
    device: Device,
    queue: CommandQueue,
    zero_u32_buffer: ComputePipelineState,
    validate_bytes_equal: ComputePipelineState,
    copy_interleaved_padded: ComputePipelineState,
    pack_gray8: ComputePipelineState,
    pack_rgb8: ComputePipelineState,
    pack_mct_rgb8: ComputePipelineState,
    pack_mct_rgb8_batched: ComputePipelineState,
    pack_rgb_opaque_rgba8: ComputePipelineState,
    pack_rgba8: ComputePipelineState,
    pack_gray16: ComputePipelineState,
    pack_rgb16: ComputePipelineState,
    pack_u8_repeated_gray: ComputePipelineState,
    pack_u16_repeated_gray: ComputePipelineState,
    classic_cleanup_plain_batched: ComputePipelineState,
    classic_cleanup_batched: ComputePipelineState,
    classic_cleanup_plain_repeated_batched: ComputePipelineState,
    classic_cleanup_plain_dev_repeated_batched: ComputePipelineState,
    classic_cleanup_repeated_batched: ComputePipelineState,
    classic_store_repeated_batched: ComputePipelineState,
    idwt_interleave: ComputePipelineState,
    idwt_reversible53_horizontal: ComputePipelineState,
    idwt_reversible53_vertical: ComputePipelineState,
    idwt_interleave_batched: ComputePipelineState,
    idwt_reversible53_horizontal_batched: ComputePipelineState,
    idwt_reversible53_vertical_batched: ComputePipelineState,
    idwt_irreversible97_single_decomposition: ComputePipelineState,
    fdwt53_horizontal: ComputePipelineState,
    fdwt53_vertical: ComputePipelineState,
    #[allow(dead_code)]
    inverse_mct: ComputePipelineState,
    forward_rct: ComputePipelineState,
    store_component: ComputePipelineState,
    store_component_repeated: ComputePipelineState,
    store_component_repeated_gray_u8: ComputePipelineState,
    store_component_repeated_gray_u16: ComputePipelineState,
    store_component_repeated_gray_u8_contiguous: ComputePipelineState,
    store_component_repeated_gray_u16_contiguous: ComputePipelineState,
    store_component_gray_u8: ComputePipelineState,
    store_component_gray_u16: ComputePipelineState,
    ht_cleanup: ComputePipelineState,
    ht_cleanup_batched: ComputePipelineState,
    ht_cleanup_repeated_batched: ComputePipelineState,
    classic_encode_code_block: ComputePipelineState,
    classic_encode_code_blocks: ComputePipelineState,
    ht_encode_code_block: ComputePipelineState,
    ht_encode_code_blocks: ComputePipelineState,
    packet_encode: ComputePipelineState,
    ht_vlc_table0: Buffer,
    ht_vlc_table1: Buffer,
    ht_uvlc_table0: Buffer,
    ht_uvlc_table1: Buffer,
    ht_vlc_encode_table0: Buffer,
    ht_vlc_encode_table1: Buffer,
    ht_uvlc_encode_table: Buffer,
    private_buffer_pool: Mutex<HashMap<usize, Vec<Buffer>>>,
}

#[cfg(target_os = "macos")]
impl MetalRuntime {
    fn new() -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or_else(|| "Metal is unavailable on this host".to_string())?;
        Self::new_with_device(&device)
    }

    fn new_with_device(device: &Device) -> Result<Self, String> {
        let options = CompileOptions::new();
        let library = device.new_library_with_source(SHADER_SOURCE, &options)?;
        let pipeline = |name: &str| {
            let function = library.get_function(name, None)?;
            device.new_compute_pipeline_state_with_function(&function)
        };
        let classic_cleanup_plain_batched_fn =
            library.get_function("j2k_decode_classic_cleanup_plain_batched", None)?;
        let classic_cleanup_batched_fn =
            library.get_function("j2k_decode_classic_cleanup_batched", None)?;
        let classic_cleanup_plain_repeated_batched_fn =
            library.get_function("j2k_decode_classic_cleanup_plain_repeated_batched", None)?;
        let classic_cleanup_plain_dev_repeated_batched_fn = library.get_function(
            "j2k_decode_classic_cleanup_plain_dev_repeated_batched",
            None,
        )?;
        let classic_cleanup_repeated_batched_fn =
            library.get_function("j2k_decode_classic_cleanup_repeated_batched", None)?;
        let classic_store_repeated_batched_fn =
            library.get_function("j2k_store_classic_repeated_batched", None)?;
        let idwt_interleave_fn = library.get_function("j2k_idwt_interleave", None)?;
        let idwt_interleave_batched_fn =
            library.get_function("j2k_idwt_interleave_batched", None)?;
        let idwt_reversible53_horizontal_fn =
            library.get_function("j2k_idwt_reversible53_horizontal_pass", None)?;
        let idwt_reversible53_horizontal_batched_fn =
            library.get_function("j2k_idwt_reversible53_horizontal_pass_batched", None)?;
        let idwt_reversible53_vertical_fn =
            library.get_function("j2k_idwt_reversible53_vertical_pass", None)?;
        let idwt_reversible53_vertical_batched_fn =
            library.get_function("j2k_idwt_reversible53_vertical_pass_batched", None)?;
        let idwt_irreversible97_single_decomposition_fn =
            library.get_function("j2k_idwt_irreversible97_single_decomposition", None)?;
        let fdwt53_horizontal_fn = library.get_function("j2k_forward_dwt53_horizontal", None)?;
        let fdwt53_vertical_fn = library.get_function("j2k_forward_dwt53_vertical", None)?;
        let inverse_mct_fn = library.get_function("j2k_inverse_mct", None)?;
        let forward_rct_fn = library.get_function("j2k_forward_rct", None)?;
        let store_component_fn = library.get_function("j2k_store_component", None)?;
        let store_component_repeated_fn =
            library.get_function("j2k_store_component_repeated", None)?;
        let store_component_repeated_gray_u8_fn =
            library.get_function("j2k_store_component_repeated_gray_u8", None)?;
        let store_component_repeated_gray_u16_fn =
            library.get_function("j2k_store_component_repeated_gray_u16", None)?;
        let store_component_repeated_gray_u8_contiguous_fn =
            library.get_function("j2k_store_component_repeated_gray_u8_contiguous", None)?;
        let store_component_repeated_gray_u16_contiguous_fn =
            library.get_function("j2k_store_component_repeated_gray_u16_contiguous", None)?;
        let store_component_gray_u8_fn =
            library.get_function("j2k_store_component_gray_u8", None)?;
        let store_component_gray_u16_fn =
            library.get_function("j2k_store_component_gray_u16", None)?;
        let ht_cleanup_fn = library.get_function("j2k_decode_ht_cleanup", None)?;
        let ht_cleanup_batched_fn = library.get_function("j2k_decode_ht_cleanup_batched", None)?;
        let ht_cleanup_repeated_batched_fn =
            library.get_function("j2k_decode_ht_cleanup_repeated_batched", None)?;
        let classic_cleanup_plain_batched =
            device.new_compute_pipeline_state_with_function(&classic_cleanup_plain_batched_fn)?;
        let classic_cleanup_batched =
            device.new_compute_pipeline_state_with_function(&classic_cleanup_batched_fn)?;
        let classic_cleanup_plain_repeated_batched = device
            .new_compute_pipeline_state_with_function(&classic_cleanup_plain_repeated_batched_fn)?;
        let classic_cleanup_plain_dev_repeated_batched = device
            .new_compute_pipeline_state_with_function(
                &classic_cleanup_plain_dev_repeated_batched_fn,
            )?;
        let classic_cleanup_repeated_batched = device
            .new_compute_pipeline_state_with_function(&classic_cleanup_repeated_batched_fn)?;
        let classic_store_repeated_batched =
            device.new_compute_pipeline_state_with_function(&classic_store_repeated_batched_fn)?;
        let idwt_interleave =
            device.new_compute_pipeline_state_with_function(&idwt_interleave_fn)?;
        let idwt_interleave_batched =
            device.new_compute_pipeline_state_with_function(&idwt_interleave_batched_fn)?;
        let idwt_reversible53_horizontal =
            device.new_compute_pipeline_state_with_function(&idwt_reversible53_horizontal_fn)?;
        let idwt_reversible53_horizontal_batched = device
            .new_compute_pipeline_state_with_function(&idwt_reversible53_horizontal_batched_fn)?;
        let idwt_reversible53_vertical =
            device.new_compute_pipeline_state_with_function(&idwt_reversible53_vertical_fn)?;
        let idwt_reversible53_vertical_batched = device
            .new_compute_pipeline_state_with_function(&idwt_reversible53_vertical_batched_fn)?;
        let idwt_irreversible97_single_decomposition = device
            .new_compute_pipeline_state_with_function(
                &idwt_irreversible97_single_decomposition_fn,
            )?;
        let fdwt53_horizontal =
            device.new_compute_pipeline_state_with_function(&fdwt53_horizontal_fn)?;
        let fdwt53_vertical =
            device.new_compute_pipeline_state_with_function(&fdwt53_vertical_fn)?;
        let inverse_mct = device.new_compute_pipeline_state_with_function(&inverse_mct_fn)?;
        let forward_rct = device.new_compute_pipeline_state_with_function(&forward_rct_fn)?;
        let store_component =
            device.new_compute_pipeline_state_with_function(&store_component_fn)?;
        let store_component_repeated =
            device.new_compute_pipeline_state_with_function(&store_component_repeated_fn)?;
        let store_component_repeated_gray_u8 = device
            .new_compute_pipeline_state_with_function(&store_component_repeated_gray_u8_fn)?;
        let store_component_repeated_gray_u16 = device
            .new_compute_pipeline_state_with_function(&store_component_repeated_gray_u16_fn)?;
        let store_component_repeated_gray_u8_contiguous = device
            .new_compute_pipeline_state_with_function(
                &store_component_repeated_gray_u8_contiguous_fn,
            )?;
        let store_component_repeated_gray_u16_contiguous = device
            .new_compute_pipeline_state_with_function(
                &store_component_repeated_gray_u16_contiguous_fn,
            )?;
        let store_component_gray_u8 =
            device.new_compute_pipeline_state_with_function(&store_component_gray_u8_fn)?;
        let store_component_gray_u16 =
            device.new_compute_pipeline_state_with_function(&store_component_gray_u16_fn)?;
        let ht_cleanup = device.new_compute_pipeline_state_with_function(&ht_cleanup_fn)?;
        let ht_cleanup_batched =
            device.new_compute_pipeline_state_with_function(&ht_cleanup_batched_fn)?;
        let ht_cleanup_repeated_batched =
            device.new_compute_pipeline_state_with_function(&ht_cleanup_repeated_batched_fn)?;
        let queue = new_command_queue(device)?;
        Ok(Self {
            device: device.clone(),
            queue,
            zero_u32_buffer: pipeline("j2k_zero_u32_buffer")?,
            validate_bytes_equal: pipeline("j2k_validate_bytes_equal")?,
            copy_interleaved_padded: pipeline("j2k_copy_interleaved_padded")?,
            pack_gray8: pipeline("j2k_pack_gray8")?,
            pack_rgb8: pipeline("j2k_pack_rgb8")?,
            pack_mct_rgb8: pipeline("j2k_pack_mct_rgb8")?,
            pack_mct_rgb8_batched: pipeline("j2k_pack_mct_rgb8_batched")?,
            pack_rgb_opaque_rgba8: pipeline("j2k_pack_rgb_opaque_rgba8")?,
            pack_rgba8: pipeline("j2k_pack_rgba8")?,
            pack_gray16: pipeline("j2k_pack_gray16")?,
            pack_rgb16: pipeline("j2k_pack_rgb16")?,
            pack_u8_repeated_gray: pipeline("j2k_pack_u8_repeated_gray")?,
            pack_u16_repeated_gray: pipeline("j2k_pack_u16_repeated_gray")?,
            classic_cleanup_plain_batched,
            classic_cleanup_batched,
            classic_cleanup_plain_repeated_batched,
            classic_cleanup_plain_dev_repeated_batched,
            classic_cleanup_repeated_batched,
            classic_store_repeated_batched,
            idwt_interleave,
            idwt_reversible53_horizontal,
            idwt_reversible53_vertical,
            idwt_interleave_batched,
            idwt_reversible53_horizontal_batched,
            idwt_reversible53_vertical_batched,
            idwt_irreversible97_single_decomposition,
            fdwt53_horizontal,
            fdwt53_vertical,
            inverse_mct,
            forward_rct,
            store_component,
            store_component_repeated,
            store_component_repeated_gray_u8,
            store_component_repeated_gray_u16,
            store_component_repeated_gray_u8_contiguous,
            store_component_repeated_gray_u16_contiguous,
            store_component_gray_u8,
            store_component_gray_u16,
            ht_cleanup,
            ht_cleanup_batched,
            ht_cleanup_repeated_batched,
            classic_encode_code_block: pipeline("j2k_encode_classic_code_block")?,
            classic_encode_code_blocks: pipeline("j2k_encode_classic_code_blocks")?,
            ht_encode_code_block: pipeline("j2k_encode_ht_code_block")?,
            ht_encode_code_blocks: pipeline("j2k_encode_ht_code_blocks")?,
            packet_encode: pipeline("j2k_encode_packetization")?,
            ht_vlc_table0: device.new_buffer_with_data(
                ht_vlc_table0().as_ptr().cast(),
                size_of_val(ht_vlc_table0()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_vlc_table1: device.new_buffer_with_data(
                ht_vlc_table1().as_ptr().cast(),
                size_of_val(ht_vlc_table1()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_uvlc_table0: device.new_buffer_with_data(
                ht_uvlc_table0().as_ptr().cast(),
                size_of_val(ht_uvlc_table0()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_uvlc_table1: device.new_buffer_with_data(
                ht_uvlc_table1().as_ptr().cast(),
                size_of_val(ht_uvlc_table1()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_vlc_encode_table0: device.new_buffer_with_data(
                ht_vlc_encode_table0().as_ptr().cast(),
                size_of_val(ht_vlc_encode_table0()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_vlc_encode_table1: device.new_buffer_with_data(
                ht_vlc_encode_table1().as_ptr().cast(),
                size_of_val(ht_vlc_encode_table1()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_uvlc_encode_table: device.new_buffer_with_data(
                ht_uvlc_encode_table().as_ptr().cast(),
                size_of_val(ht_uvlc_encode_table()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            private_buffer_pool: Mutex::new(HashMap::new()),
        })
    }

    fn take_private_buffer(&self, bytes: usize) -> Buffer {
        let bytes = bytes.max(1);
        let mut pool = self
            .private_buffer_pool
            .lock()
            .expect("private buffer pool lock not poisoned");
        if let Some(buffer) = pool.get_mut(&bytes).and_then(Vec::pop) {
            buffer
        } else {
            self.device
                .new_buffer(bytes as u64, MTLResourceOptions::StorageModePrivate)
        }
    }

    fn recycle_private_buffer(&self, bytes: usize, buffer: Buffer) {
        let bytes = bytes.max(1);
        self.private_buffer_pool
            .lock()
            .expect("private buffer pool lock not poisoned")
            .entry(bytes)
            .or_default()
            .push(buffer);
    }
}

#[cfg(target_os = "macos")]
fn new_command_queue(device: &Device) -> Result<CommandQueue, String> {
    let queue: *mut MTLCommandQueue = unsafe {
        device
            .as_ref()
            .send_message(Sel::register("newCommandQueue"), ())
            .map_err(|error| format!("Metal command queue creation failed: {error}"))?
    };
    if queue.is_null() {
        return Err("Metal command queue is unavailable on this host".to_string());
    }
    Ok(unsafe { CommandQueue::from_ptr(queue) })
}

#[cfg(target_os = "macos")]
fn with_runtime<R>(f: impl FnOnce(&MetalRuntime) -> Result<R, Error>) -> Result<R, Error> {
    let override_runtime = METAL_RUNTIME_OVERRIDE.with(|slot| slot.borrow().clone());
    if let Some(runtime) = override_runtime {
        return f(&runtime);
    }

    match METAL_RUNTIME.get_or_init(|| MetalRuntime::new().map(Arc::new)) {
        Ok(runtime) => f(runtime),
        Err(message) => Err(runtime_initialization_error(message)),
    }
}

#[cfg(target_os = "macos")]
fn runtime_initialization_error(message: &str) -> Error {
    match message {
        "Metal is unavailable on this host" | "Metal command queue is unavailable on this host" => {
            Error::MetalUnavailable
        }
        _ => Error::MetalKernel {
            message: message.to_string(),
        },
    }
}

#[cfg(target_os = "macos")]
struct RuntimeOverrideGuard {
    previous: Option<Arc<MetalRuntime>>,
}

#[cfg(target_os = "macos")]
impl Drop for RuntimeOverrideGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        METAL_RUNTIME_OVERRIDE.with(|slot| {
            slot.replace(previous);
        });
    }
}

#[cfg(target_os = "macos")]
fn with_runtime_for_device<R>(
    device: &Device,
    f: impl FnOnce(&MetalRuntime) -> Result<R, Error>,
) -> Result<R, Error> {
    let runtime = Arc::new(
        MetalRuntime::new_with_device(device)
            .map_err(|message| runtime_initialization_error(&message))?,
    );
    let previous = METAL_RUNTIME_OVERRIDE.with(|slot| slot.replace(Some(runtime.clone())));
    let _guard = RuntimeOverrideGuard { previous };
    f(&runtime)
}

#[cfg(target_os = "macos")]
pub(crate) fn validate_metal_buffer_matches_bytes(
    expected: &[u8],
    actual_buffer: &Buffer,
    actual_byte_offset: usize,
    session: &crate::MetalBackendSession,
) -> Result<(), Error> {
    if expected.is_empty() {
        return Ok(());
    }
    let byte_len = u32::try_from(expected.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal validation buffer exceeds u32 byte length".to_string(),
    })?;
    let actual_offset = u64::try_from(actual_byte_offset).map_err(|_| Error::MetalKernel {
        message: "J2K Metal validation buffer offset exceeds u64".to_string(),
    })?;

    with_runtime_for_device(&session.device, |runtime| {
        let expected_buffer = runtime.device.new_buffer_with_data(
            expected.as_ptr().cast(),
            expected.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status = J2kValidateBytesStatus::default();
        let status_buffer = runtime.device.new_buffer_with_data(
            (&raw const status).cast(),
            size_of::<J2kValidateBytesStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params = J2kValidateBytesParams { byte_len };

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.validate_bytes_equal);
        encoder.set_buffer(0, Some(actual_buffer), actual_offset);
        encoder.set_buffer(1, Some(&expected_buffer), 0);
        encoder.set_buffer(2, Some(&status_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kValidateBytesParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.dispatch_threads(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = unsafe {
            status_buffer
                .contents()
                .cast::<J2kValidateBytesStatus>()
                .read()
        };
        if status.code == 0 {
            return Ok(());
        }

        Err(Error::MetalKernel {
            message: format!(
                "J2K Metal validation mismatch at byte {}: expected {}, got {}",
                status.index, status.expected, status.actual
            ),
        })
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn copy_interleaved_padded_to_shared_buffer(
    src_buffer: &Buffer,
    src_byte_offset: usize,
    src_width: u32,
    src_height: u32,
    src_pitch_bytes: usize,
    dst_width: u32,
    dst_height: u32,
    bytes_per_pixel: usize,
    session: &crate::MetalBackendSession,
) -> Result<Buffer, Error> {
    if src_width > dst_width || src_height > dst_height {
        return Err(Error::MetalKernel {
            message: "J2K Metal input tile cannot be larger than encoded tile".to_string(),
        });
    }
    let src_stride = u32::try_from(src_pitch_bytes).map_err(|_| Error::MetalKernel {
        message: "J2K Metal input tile pitch exceeds u32".to_string(),
    })?;
    let dst_stride_usize = (dst_width as usize)
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal padded tile stride overflow".to_string(),
        })?;
    let dst_stride = u32::try_from(dst_stride_usize).map_err(|_| Error::MetalKernel {
        message: "J2K Metal padded tile stride exceeds u32".to_string(),
    })?;
    let dst_len = dst_stride_usize
        .checked_mul(dst_height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal padded tile byte length overflow".to_string(),
        })?;
    let bytes_per_pixel = u32::try_from(bytes_per_pixel).map_err(|_| Error::MetalKernel {
        message: "J2K Metal bytes-per-pixel exceeds u32".to_string(),
    })?;
    let src_offset = u64::try_from(src_byte_offset).map_err(|_| Error::MetalKernel {
        message: "J2K Metal input tile offset exceeds u64".to_string(),
    })?;

    with_runtime_for_device(&session.device, |runtime| {
        let dst_buffer = runtime
            .device
            .new_buffer(dst_len as u64, MTLResourceOptions::StorageModeShared);
        let params = J2kCopyInterleavedParams {
            src_width,
            src_height,
            src_stride,
            dst_width,
            dst_height,
            dst_stride,
            bytes_per_pixel,
        };
        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.copy_interleaved_padded);
        encoder.set_buffer(0, Some(src_buffer), src_offset);
        encoder.set_buffer(1, Some(&dst_buffer), 0);
        encoder.set_bytes(
            2,
            size_of::<J2kCopyInterleavedParams>() as u64,
            (&raw const params).cast(),
        );
        let width = runtime
            .copy_interleaved_padded
            .thread_execution_width()
            .max(1);
        let max_threads = runtime
            .copy_interleaved_padded
            .max_total_threads_per_threadgroup()
            .max(width);
        let height = (max_threads / width).max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(dst_width),
                height: u64::from(dst_height),
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
        Ok(dst_buffer)
    })
}

#[cfg(target_os = "macos")]
enum DirectStatusCheck {
    Classic { buffer: Buffer, len: usize },
    Ht { buffer: Buffer, len: usize },
    Idwt(Buffer),
    Mct(Buffer),
}

#[cfg(target_os = "macos")]
struct DirectScratchBuffer {
    bytes: usize,
    buffer: Buffer,
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
enum DirectHostRetention {
    Bytes(Vec<u8>),
    ClassicJobs(Vec<J2kClassicCleanupBatchJob>),
    ClassicSegments(Vec<J2kClassicSegment>),
    HtJobs(Vec<J2kHtCleanupBatchJob>),
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
pub(crate) struct PreparedDirectGrayscalePlan {
    dimensions: (u32, u32),
    bit_depth: u8,
    steps: Vec<PreparedDirectGrayscaleStep>,
    classic_groups: Vec<PreparedClassicSubBandGroup>,
    ht_groups: Vec<PreparedHtSubBandGroup>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
pub(crate) struct PreparedDirectColorPlan {
    dimensions: (u32, u32),
    bit_depths: [u8; 3],
    mct: bool,
    transform: J2kWaveletTransform,
    component_plans: Vec<PreparedDirectGrayscalePlan>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
enum PreparedDirectGrayscaleStep {
    ClassicSubBand(PreparedClassicSubBand),
    HtSubBand(PreparedHtSubBand),
    Idwt(J2kDirectIdwtStep),
    Store(J2kDirectStoreStep),
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedClassicSubBand {
    band_id: J2kDirectBandId,
    width: u32,
    height: u32,
    coded_data: Vec<u8>,
    coded_buffer: Buffer,
    jobs: Vec<J2kClassicCleanupBatchJob>,
    jobs_buffer: Buffer,
    segments: Vec<J2kClassicSegment>,
    segments_buffer: Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedClassicSubBandGroup {
    start_step: usize,
    end_step: usize,
    total_coefficients: usize,
    coded_data: Vec<u8>,
    coded_buffer: Buffer,
    jobs: Vec<J2kClassicCleanupBatchJob>,
    jobs_buffer: Buffer,
    segments: Vec<J2kClassicSegment>,
    segments_buffer: Buffer,
    members: Vec<PreparedClassicSubBandGroupMember>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedClassicSubBandGroupMember {
    band_id: J2kDirectBandId,
    offset_elements: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedHtSubBand {
    band_id: J2kDirectBandId,
    width: u32,
    height: u32,
    coded_data: Vec<u8>,
    coded_buffer: Buffer,
    jobs: Vec<J2kHtCleanupBatchJob>,
    jobs_buffer: Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedHtSubBandGroup {
    start_step: usize,
    end_step: usize,
    total_coefficients: usize,
    _coded_data: Vec<u8>,
    coded_buffer: Buffer,
    jobs: Vec<J2kHtCleanupBatchJob>,
    jobs_buffer: Buffer,
    members: Vec<PreparedHtSubBandGroupMember>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedHtSubBandGroupMember {
    band_id: J2kDirectBandId,
    offset_elements: usize,
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
    #[allow(dead_code)]
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

    #[allow(dead_code)]
    fn from_captured_planes(
        decoded: &NativeDecodedComponents<'_>,
        captured_planes: Vec<Buffer>,
    ) -> Option<Self> {
        let plane_count = decoded.planes().len();
        let supported_shape = matches!(
            (decoded.color_space(), decoded.has_alpha(), plane_count),
            (NativeColorSpace::Gray, false, 1) | (NativeColorSpace::RGB, false, 3)
        );
        if !supported_shape {
            return None;
        }
        if captured_planes.len() != plane_count || plane_count == 0 || plane_count > 4 {
            return None;
        }

        let mut bit_depths = [0u32; 4];
        let mut planes: [Option<Buffer>; 4] = [None, None, None, None];
        for (index, (plane, buffer)) in decoded
            .planes()
            .iter()
            .zip(captured_planes.into_iter())
            .enumerate()
        {
            bit_depths[index] = u32::from(plane.bit_depth());
            planes[index] = Some(buffer);
        }

        Some(Self {
            dims: decoded.dimensions(),
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
        let command_buffer = runtime.queue.new_command_buffer();
        let surface =
            encode_plane_stage_to_surface_in_command_buffer(runtime, command_buffer, &self, fmt)?;
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(surface)
    }
}

#[cfg(target_os = "macos")]
fn encode_plane_stage_to_surface_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    stage: &PlaneStage,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let pitch_bytes = stage.dims.0 as usize * fmt.bytes_per_pixel();
    let out_buffer = runtime.device.new_buffer(
        (pitch_bytes * stage.dims.1 as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let (output_channels, opaque_alpha, pipeline) = output_shape_for(
        &stage.color_space,
        stage.has_alpha,
        stage.plane_count,
        fmt,
        runtime,
    )?;
    let (max_values, u8_scales, u16_scales) = j2k_pack_scale_arrays(stage.bit_depths);

    let params = J2kPackParams {
        width: stage.dims.0,
        height: stage.dims.1,
        out_stride: u32::try_from(pitch_bytes).expect("J2K Metal output stride fits in u32"),
        output_channels,
        opaque_alpha,
        max_values,
        u8_scales,
        u16_scales,
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(
        0,
        stage.planes[0].as_ref().map(std::convert::AsRef::as_ref),
        0,
    );
    encoder.set_buffer(
        1,
        stage.planes[1].as_ref().map(std::convert::AsRef::as_ref),
        0,
    );
    encoder.set_buffer(
        2,
        stage.planes[2].as_ref().map(std::convert::AsRef::as_ref),
        0,
    );
    encoder.set_buffer(
        3,
        stage.planes[3].as_ref().map(std::convert::AsRef::as_ref),
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
            width: u64::from(stage.dims.0),
            height: u64::from(stage.dims.1),
            depth: 1,
        },
        MTLSize {
            width,
            height,
            depth: 1,
        },
    );
    encoder.end_encoding();

    Ok(Surface::from_metal_buffer(out_buffer, stage.dims, fmt))
}

#[cfg(target_os = "macos")]
fn encode_mct_rgb8_to_surface_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: [&Buffer; 3],
    dims: (u32, u32),
    bit_depths: [u8; 3],
    transform: J2kWaveletTransform,
) -> Surface {
    let pitch_bytes = dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let out_buffer = runtime.device.new_buffer(
        (pitch_bytes * dims.1 as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let (max_values, u8_scales, _) = j2k_pack_scale_arrays([
        u32::from(bit_depths[0]),
        u32::from(bit_depths[1]),
        u32::from(bit_depths[2]),
        0,
    ]);
    let params = J2kMctRgb8PackParams {
        width: dims.0,
        height: dims.1,
        out_stride: u32::try_from(pitch_bytes).expect("J2K Metal output stride fits in u32"),
        transform: mct_transform_code(transform),
        addends: [
            signed_sample_bias(bit_depths[0]),
            signed_sample_bias(bit_depths[1]),
            signed_sample_bias(bit_depths[2]),
        ],
        max_values: [max_values[0], max_values[1], max_values[2]],
        u8_scales: [u8_scales[0], u8_scales[1], u8_scales[2]],
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.pack_mct_rgb8);
    encoder.set_buffer(0, Some(planes[0]), 0);
    encoder.set_buffer(1, Some(planes[1]), 0);
    encoder.set_buffer(2, Some(planes[2]), 0);
    encoder.set_buffer(3, Some(&out_buffer), 0);
    encoder.set_bytes(
        4,
        size_of::<J2kMctRgb8PackParams>() as u64,
        (&raw const params).cast(),
    );

    let width = runtime.pack_mct_rgb8.thread_execution_width().max(1);
    let max_threads = runtime
        .pack_mct_rgb8
        .max_total_threads_per_threadgroup()
        .max(width);
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
    encoder.end_encoding();

    Surface::from_metal_buffer(out_buffer, dims, PixelFormat::Rgb8)
}

#[cfg(target_os = "macos")]
fn encode_batched_mct_rgb8_to_surfaces_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: [&Buffer; 3],
    dims: (u32, u32),
    count: usize,
    bit_depths: [u8; 3],
    transform: J2kWaveletTransform,
) -> Result<Vec<Surface>, Error> {
    let count_u32 = u32::try_from(count).map_err(|_| Error::MetalKernel {
        message: "J2K MetalDirect color batch count exceeds u32".to_string(),
    })?;
    let pitch_bytes = dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let surface_bytes =
        pitch_bytes
            .checked_mul(dims.1 as usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect color batch output size overflow".to_string(),
            })?;
    let total_bytes = surface_bytes
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect color batch output size overflow".to_string(),
        })?;
    let out_buffer = runtime
        .device
        .new_buffer(total_bytes as u64, MTLResourceOptions::StorageModeShared);
    let plane_stride = dims
        .0
        .checked_mul(dims.1)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect color batch plane stride overflow".to_string(),
        })?;
    let (max_values, u8_scales, _) = j2k_pack_scale_arrays([
        u32::from(bit_depths[0]),
        u32::from(bit_depths[1]),
        u32::from(bit_depths[2]),
        0,
    ]);
    let params = J2kBatchedMctRgb8PackParams {
        width: dims.0,
        height: dims.1,
        out_stride: u32::try_from(pitch_bytes).expect("J2K Metal output stride fits in u32"),
        transform: mct_transform_code(transform),
        batch_count: count_u32,
        plane_stride,
        output_stride: u32::try_from(surface_bytes).map_err(|_| Error::MetalKernel {
            message: "J2K MetalDirect color batch surface stride exceeds u32".to_string(),
        })?,
        addends: [
            signed_sample_bias(bit_depths[0]),
            signed_sample_bias(bit_depths[1]),
            signed_sample_bias(bit_depths[2]),
        ],
        max_values: [max_values[0], max_values[1], max_values[2]],
        u8_scales: [u8_scales[0], u8_scales[1], u8_scales[2]],
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.pack_mct_rgb8_batched);
    encoder.set_buffer(0, Some(planes[0]), 0);
    encoder.set_buffer(1, Some(planes[1]), 0);
    encoder.set_buffer(2, Some(planes[2]), 0);
    encoder.set_buffer(3, Some(&out_buffer), 0);
    encoder.set_bytes(
        4,
        size_of::<J2kBatchedMctRgb8PackParams>() as u64,
        (&raw const params).cast(),
    );

    let width = runtime
        .pack_mct_rgb8_batched
        .thread_execution_width()
        .max(1);
    let max_threads = runtime
        .pack_mct_rgb8_batched
        .max_total_threads_per_threadgroup()
        .max(width);
    let height = (max_threads / width).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(dims.0),
            height: u64::from(dims.1),
            depth: u64::from(count_u32),
        },
        MTLSize {
            width,
            height,
            depth: 1,
        },
    );
    encoder.end_encoding();

    Ok((0..count)
        .map(|index| {
            Surface::from_metal_buffer_with_offset(
                out_buffer.clone(),
                dims,
                PixelFormat::Rgb8,
                index * surface_bytes,
            )
        })
        .collect())
}

#[cfg(target_os = "macos")]
fn mct_transform_code(transform: J2kWaveletTransform) -> u32 {
    match transform {
        J2kWaveletTransform::Reversible53 => 0,
        J2kWaveletTransform::Irreversible97 => 1,
    }
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
pub(crate) fn pack_gray_plane_to_surface(
    plane: Buffer,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let mut bit_depths = [0u32; 4];
        bit_depths[0] = u32::from(bit_depth);
        PlaneStage {
            dims,
            plane_count: 1,
            color_space: NativeColorSpace::Gray,
            has_alpha: false,
            bit_depths,
            planes: [Some(plane), None, None, None],
        }
        .finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
fn prepare_classic_sub_band(
    job: &signinum_j2k_native::J2kOwnedSubBandPlan,
) -> Result<PreparedClassicSubBand, Error> {
    let mut jobs = Vec::with_capacity(job.jobs.len());
    let mut coded_data = Vec::new();
    let mut segments = Vec::new();

    for block in &job.jobs {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(&block.data);
        let segment_offset = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect segment table exceeds u32".to_string(),
        })?;
        for segment in &block.segments {
            let data_offset = coded_offset
                .checked_add(segment.data_offset)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect segment offset overflow".to_string(),
                })?;
            segments.push(J2kClassicSegment {
                data_offset,
                data_length: segment.data_length,
                start_coding_pass: u32::from(segment.start_coding_pass),
                end_coding_pass: u32::from(segment.end_coding_pass),
                use_arithmetic: u32::from(segment.use_arithmetic),
            });
        }
        jobs.push(J2kClassicCleanupBatchJob {
            coded_offset,
            coded_len: u32::try_from(block.data.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
            })?,
            segment_offset,
            segment_count: u32::try_from(block.segments.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K MetalDirect segment count exceeds u32".to_string(),
            })?,
            width: block.width,
            height: block.height,
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect output offset overflow".to_string(),
                })?,
            missing_msbs: u32::from(block.missing_bit_planes),
            total_bitplanes: u32::from(block.total_bitplanes),
            number_of_coding_passes: u32::from(block.number_of_coding_passes),
            sub_band_type: match block.sub_band_type {
                signinum_j2k_native::J2kSubBandType::LowLow => 0,
                signinum_j2k_native::J2kSubBandType::HighLow => 1,
                signinum_j2k_native::J2kSubBandType::LowHigh => 2,
                signinum_j2k_native::J2kSubBandType::HighHigh => 3,
            },
            style_flags: classic_style_flags(block.style),
            strict: u32::from(block.strict),
            dequantization_step: block.dequantization_step,
        });
    }

    with_runtime(|runtime| {
        let coded_buffer = borrow_slice_buffer(&runtime.device, &coded_data);
        let jobs_buffer = borrow_slice_buffer(&runtime.device, &jobs);
        let segments_buffer = borrow_slice_buffer(&runtime.device, &segments);
        Ok(PreparedClassicSubBand {
            band_id: job.band_id,
            width: job.width,
            height: job.height,
            coded_data,
            coded_buffer,
            jobs,
            jobs_buffer,
            segments,
            segments_buffer,
        })
    })
}

#[cfg(target_os = "macos")]
fn prepare_classic_sub_band_groups(
    steps: &[PreparedDirectGrayscaleStep],
) -> Result<Vec<PreparedClassicSubBandGroup>, Error> {
    let mut groups = Vec::new();
    let mut step_idx = 0;
    while step_idx < steps.len() {
        let start_step = step_idx;
        let mut sub_bands = Vec::new();
        while let Some(PreparedDirectGrayscaleStep::ClassicSubBand(sub_band)) = steps.get(step_idx)
        {
            sub_bands.push(sub_band);
            step_idx += 1;
        }
        if sub_bands.len() > 1 {
            groups.push(prepare_classic_sub_band_group(
                start_step, step_idx, &sub_bands,
            )?);
        }
        if step_idx == start_step {
            step_idx += 1;
        }
    }
    Ok(groups)
}

#[cfg(target_os = "macos")]
fn prepare_classic_sub_band_group(
    start_step: usize,
    end_step: usize,
    sub_bands: &[&PreparedClassicSubBand],
) -> Result<PreparedClassicSubBandGroup, Error> {
    let mut members = Vec::with_capacity(sub_bands.len());
    let mut jobs = Vec::new();
    let mut segments = Vec::new();
    let mut coded_data = Vec::new();
    let mut output_base = 0usize;

    for sub_band in sub_bands {
        members.push(PreparedClassicSubBandGroupMember {
            band_id: sub_band.band_id,
            offset_elements: output_base,
        });

        let coded_base = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect grouped coded payload exceeds u32".to_string(),
        })?;
        let segment_base = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect grouped segment table exceeds u32".to_string(),
        })?;
        let output_base_u32 = u32::try_from(output_base).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect grouped coefficient arena exceeds u32".to_string(),
        })?;

        for segment in &sub_band.segments {
            let mut grouped_segment = *segment;
            grouped_segment.data_offset =
                coded_base
                    .checked_add(segment.data_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect grouped segment offset overflow"
                            .to_string(),
                    })?;
            segments.push(grouped_segment);
        }

        for job in &sub_band.jobs {
            let mut grouped_job = *job;
            grouped_job.coded_offset =
                coded_base
                    .checked_add(job.coded_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect grouped job coded offset overflow"
                            .to_string(),
                    })?;
            grouped_job.segment_offset =
                segment_base
                    .checked_add(job.segment_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect grouped job segment offset overflow"
                            .to_string(),
                    })?;
            grouped_job.output_offset =
                output_base_u32
                    .checked_add(job.output_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect grouped output offset overflow"
                            .to_string(),
                    })?;
            jobs.push(grouped_job);
        }

        coded_data.extend_from_slice(&sub_band.coded_data);
        let sub_band_len =
            sub_band
                .width
                .checked_mul(sub_band.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect grouped sub-band size overflow".to_string(),
                })? as usize;
        output_base = output_base
            .checked_add(sub_band_len)
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K MetalDirect grouped coefficient arena overflow".to_string(),
            })?;
    }

    with_runtime(|runtime| {
        let coded_buffer = borrow_slice_buffer(&runtime.device, &coded_data);
        let jobs_buffer = borrow_slice_buffer(&runtime.device, &jobs);
        let segments_buffer = borrow_slice_buffer(&runtime.device, &segments);
        Ok(PreparedClassicSubBandGroup {
            start_step,
            end_step,
            total_coefficients: output_base,
            coded_data,
            coded_buffer,
            jobs,
            jobs_buffer,
            segments,
            segments_buffer,
            members,
        })
    })
}

#[cfg(target_os = "macos")]
fn prepare_ht_sub_band(
    job: &signinum_j2k_native::HtOwnedSubBandPlan,
) -> Result<PreparedHtSubBand, Error> {
    let mut jobs = Vec::with_capacity(job.jobs.len());
    let mut coded_data = Vec::new();
    for block in &job.jobs {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(&block.data);
        jobs.push(J2kHtCleanupBatchJob {
            coded_offset,
            width: block.width,
            height: block.height,
            coded_len: u32::try_from(block.data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
            })?,
            cleanup_length: block.cleanup_length,
            refinement_length: block.refinement_length,
            missing_msbs: u32::from(block.missing_bit_planes),
            num_bitplanes: u32::from(block.num_bitplanes),
            number_of_coding_passes: u32::from(block.number_of_coding_passes),
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K MetalDirect output offset overflow".to_string(),
                })?,
            dequantization_step: block.dequantization_step,
            stripe_causal: u32::from(block.stripe_causal),
        });
    }

    with_runtime(|runtime| {
        let coded_buffer = borrow_slice_buffer(&runtime.device, &coded_data);
        let jobs_buffer = borrow_slice_buffer(&runtime.device, &jobs);
        Ok(PreparedHtSubBand {
            band_id: job.band_id,
            width: job.width,
            height: job.height,
            coded_data,
            coded_buffer,
            jobs,
            jobs_buffer,
        })
    })
}

#[cfg(target_os = "macos")]
fn prepare_ht_sub_band_groups(
    steps: &[PreparedDirectGrayscaleStep],
) -> Result<Vec<PreparedHtSubBandGroup>, Error> {
    let mut groups = Vec::new();
    let mut step_idx = 0;
    while step_idx < steps.len() {
        let start_step = step_idx;
        let mut sub_bands = Vec::new();
        while let Some(PreparedDirectGrayscaleStep::HtSubBand(sub_band)) = steps.get(step_idx) {
            sub_bands.push(sub_band);
            step_idx += 1;
        }
        if sub_bands.len() > 1 {
            groups.push(prepare_ht_sub_band_group(start_step, step_idx, &sub_bands)?);
        }
        if step_idx == start_step {
            step_idx += 1;
        }
    }
    Ok(groups)
}

#[cfg(target_os = "macos")]
fn prepare_ht_sub_band_group(
    start_step: usize,
    end_step: usize,
    sub_bands: &[&PreparedHtSubBand],
) -> Result<PreparedHtSubBandGroup, Error> {
    let mut members = Vec::with_capacity(sub_bands.len());
    let mut jobs = Vec::new();
    let mut coded_data = Vec::new();
    let mut output_base = 0usize;

    for sub_band in sub_bands {
        members.push(PreparedHtSubBandGroupMember {
            band_id: sub_band.band_id,
            offset_elements: output_base,
        });

        let coded_base = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect grouped coded payload exceeds u32".to_string(),
        })?;
        let output_base_u32 = u32::try_from(output_base).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect grouped coefficient arena exceeds u32".to_string(),
        })?;
        for job in &sub_band.jobs {
            let mut grouped_job = *job;
            grouped_job.coded_offset =
                coded_base
                    .checked_add(job.coded_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect grouped coded offset overflow".to_string(),
                    })?;
            grouped_job.output_offset =
                output_base_u32
                    .checked_add(job.output_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect grouped output offset overflow".to_string(),
                    })?;
            jobs.push(grouped_job);
        }
        coded_data.extend_from_slice(&sub_band.coded_data);
        let sub_band_len =
            sub_band
                .width
                .checked_mul(sub_band.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K MetalDirect grouped sub-band size overflow".to_string(),
                })? as usize;
        output_base = output_base
            .checked_add(sub_band_len)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K MetalDirect grouped coefficient arena overflow".to_string(),
            })?;
    }

    with_runtime(|runtime| {
        let coded_buffer = borrow_slice_buffer(&runtime.device, &coded_data);
        let jobs_buffer = borrow_slice_buffer(&runtime.device, &jobs);
        Ok(PreparedHtSubBandGroup {
            start_step,
            end_step,
            total_coefficients: output_base,
            _coded_data: coded_data,
            coded_buffer,
            jobs,
            jobs_buffer,
            members,
        })
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_direct_grayscale_plan(
    plan: &J2kDirectGrayscalePlan,
) -> Result<PreparedDirectGrayscalePlan, Error> {
    let mut steps = Vec::with_capacity(plan.steps.len());
    for step in &plan.steps {
        match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                steps.push(PreparedDirectGrayscaleStep::ClassicSubBand(
                    prepare_classic_sub_band(sub_band)?,
                ));
            }
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                steps.push(PreparedDirectGrayscaleStep::HtSubBand(prepare_ht_sub_band(
                    sub_band,
                )?));
            }
            J2kDirectGrayscaleStep::Idwt(idwt) => {
                steps.push(PreparedDirectGrayscaleStep::Idwt(*idwt));
            }
            J2kDirectGrayscaleStep::Store(store) => {
                steps.push(PreparedDirectGrayscaleStep::Store(*store));
            }
        }
    }
    let classic_groups = prepare_classic_sub_band_groups(&steps)?;
    let ht_groups = prepare_ht_sub_band_groups(&steps)?;
    Ok(PreparedDirectGrayscalePlan {
        dimensions: plan.dimensions,
        bit_depth: plan.bit_depth,
        steps,
        classic_groups,
        ht_groups,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_direct_color_plan(
    plan: &J2kDirectColorPlan,
) -> Result<PreparedDirectColorPlan, Error> {
    let component_plans = plan
        .component_plans
        .iter()
        .map(prepare_direct_grayscale_plan)
        .collect::<Result<Vec<_>, _>>()?;
    if component_plans.len() != 3 {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K MetalDirect color plan expected 3 component plans, got {}",
                component_plans.len()
            ),
        });
    }
    Ok(PreparedDirectColorPlan {
        dimensions: plan.dimensions,
        bit_depths: plan.bit_depths,
        mct: plan.mct,
        transform: plan.transform,
        component_plans,
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_direct_grayscale_plan_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &J2kDirectGrayscalePlan,
    fmt: PixelFormat,
    retained_buffers: &mut Vec<Buffer>,
    retained_host_data: &mut Vec<DirectHostRetention>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Surface, Error> {
    let mut bands: Vec<(J2kDirectBandId, Buffer)> = Vec::new();
    let mut final_surface = None;

    for step in &plan.steps {
        match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                let output = take_f32_scratch_buffer(
                    runtime,
                    sub_band.width as usize * sub_band.height as usize,
                );
                let (host_data, buffers, status_check) =
                    encode_classic_sub_band_to_buffer_in_command_buffer(
                        runtime,
                        command_buffer,
                        sub_band,
                        &output.buffer,
                        scratch_buffers,
                    )?;
                retained_host_data.extend(host_data);
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                bands.push((sub_band.band_id, output.buffer.clone()));
                scratch_buffers.push(output);
            }
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                let output = take_f32_scratch_buffer(
                    runtime,
                    sub_band.width as usize * sub_band.height as usize,
                );
                let (host_data, buffers, status_check) =
                    encode_ht_sub_band_to_buffer_in_command_buffer(
                        runtime,
                        command_buffer,
                        sub_band,
                        &output.buffer,
                    )?;
                retained_host_data.extend(host_data);
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                bands.push((sub_band.band_id, output.buffer.clone()));
                scratch_buffers.push(output);
            }
            J2kDirectGrayscaleStep::Idwt(idwt) => {
                let ll = lookup_direct_band(&bands, idwt.ll_band_id, idwt.ll)?;
                let hl = lookup_direct_band(&bands, idwt.hl_band_id, idwt.hl)?;
                let lh = lookup_direct_band(&bands, idwt.lh_band_id, idwt.lh)?;
                let hh = lookup_direct_band(&bands, idwt.hh_band_id, idwt.hh)?;
                let params = J2kIdwtSingleDecompositionParams {
                    x0: idwt.rect.x0,
                    y0: idwt.rect.y0,
                    width: idwt.rect.width(),
                    height: idwt.rect.height(),
                    ll_width: idwt.ll.width(),
                    ll_height: idwt.ll.height(),
                    hl_width: idwt.hl.width(),
                    hl_height: idwt.hl.height(),
                    lh_width: idwt.lh.width(),
                    lh_height: idwt.lh.height(),
                    hh_width: idwt.hh.width(),
                    hh_height: idwt.hh.height(),
                };
                let output = take_f32_scratch_buffer(
                    runtime,
                    idwt.rect.width() as usize * idwt.rect.height() as usize,
                );
                match idwt.transform {
                    J2kWaveletTransform::Reversible53 => {
                        dispatch_reversible53_single_decomposition_buffers_in_command_buffer(
                            runtime,
                            command_buffer,
                            &ll,
                            &hl,
                            &lh,
                            &hh,
                            params,
                            &output.buffer,
                        );
                    }
                    J2kWaveletTransform::Irreversible97 => {
                        let status_check =
                            dispatch_irreversible97_single_decomposition_buffers_in_command_buffer(
                                runtime,
                                command_buffer,
                                &ll,
                                &hl,
                                &lh,
                                &hh,
                                params,
                                &output.buffer,
                            );
                        status_checks.push(status_check);
                    }
                }
                bands.push((idwt.output_band_id, output.buffer.clone()));
                scratch_buffers.push(output);
            }
            J2kDirectGrayscaleStep::Store(store) => {
                let input = lookup_direct_band(&bands, store.input_band_id, store.input_rect)?;
                if matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
                    let scale = j2k_scalar_pack_params(u32::from(plan.bit_depth));
                    final_surface = Some(encode_gray_store_to_surface_in_command_buffer(
                        runtime,
                        command_buffer,
                        &input,
                        0,
                        J2kGrayStoreParams {
                            input_width: store.input_rect.width(),
                            source_x: store.source_x,
                            source_y: store.source_y,
                            copy_width: store.copy_width,
                            copy_height: store.copy_height,
                            output_width: store.output_width,
                            output_x: store.output_x,
                            output_y: store.output_y,
                            addend: store.addend,
                            max_value: scale.max_value,
                            u8_scale: scale.u8_scale,
                            u16_scale: scale.u16_scale,
                        },
                        plan.dimensions,
                        fmt,
                    )?);
                } else {
                    let output = take_f32_scratch_buffer(
                        runtime,
                        store.output_width as usize * store.output_height as usize,
                    );
                    let params = J2kStoreParams {
                        input_width: store.input_rect.width(),
                        source_x: store.source_x,
                        source_y: store.source_y,
                        copy_width: store.copy_width,
                        copy_height: store.copy_height,
                        output_width: store.output_width,
                        output_x: store.output_x,
                        output_y: store.output_y,
                        addend: store.addend,
                    };
                    dispatch_store_component_buffer_in_command_buffer(
                        runtime,
                        command_buffer,
                        &input,
                        &output.buffer,
                        params,
                    );
                    retained_buffers.push(output.buffer.clone());
                    final_surface = Some(encode_gray_plane_to_surface_in_command_buffer(
                        runtime,
                        command_buffer,
                        &output.buffer,
                        plan.dimensions,
                        plan.bit_depth,
                        fmt,
                    )?);
                    scratch_buffers.push(output);
                }
            }
        }
    }

    let surface = final_surface.ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect grayscale plan did not produce a final stored plane".to_string(),
    })?;

    Ok(surface)
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_direct_grayscale_plan(
    plan: &J2kDirectGrayscalePlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut retained_host_data = Vec::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let surface = encode_direct_grayscale_plan_in_command_buffer(
            runtime,
            command_buffer,
            plan,
            fmt,
            &mut retained_buffers,
            &mut retained_host_data,
            &mut status_checks,
            &mut scratch_buffers,
        )?;
        command_buffer.commit();
        command_buffer.wait_until_completed();
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        drop(retained_host_data);
        recycle_scratch_buffers(runtime, scratch_buffers);
        Ok(surface)
    })
}

#[cfg(target_os = "macos")]
impl PreparedDirectGrayscalePlan {
    fn classic_group_starting_at(&self, step_idx: usize) -> Option<&PreparedClassicSubBandGroup> {
        self.classic_groups
            .iter()
            .find(|group| group.start_step == step_idx)
    }

    fn ht_group_starting_at(&self, step_idx: usize) -> Option<&PreparedHtSubBandGroup> {
        self.ht_groups
            .iter()
            .find(|group| group.start_step == step_idx)
    }
}

#[cfg(all(test, target_os = "macos"))]
fn prepared_direct_grayscale_plan_compute_encoder_count(
    plan: &PreparedDirectGrayscalePlan,
    _fmt: PixelFormat,
) -> usize {
    usize::from(!plan.steps.is_empty())
}

#[cfg(all(test, target_os = "macos"))]
fn prepared_repeated_direct_ht_cleanup_dispatch_count(plan: &PreparedDirectGrayscalePlan) -> usize {
    let mut dispatches = 0;
    let mut step_idx = 0;
    while step_idx < plan.steps.len() {
        if let Some(group) = plan.ht_group_starting_at(step_idx) {
            dispatches += 1;
            step_idx = group.end_step;
            continue;
        }
        if matches!(
            plan.steps[step_idx],
            PreparedDirectGrayscaleStep::HtSubBand(_)
        ) {
            dispatches += 1;
        }
        step_idx += 1;
    }
    dispatches
}

#[cfg(target_os = "macos")]
fn encode_prepared_direct_grayscale_plan_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    retained_buffers: &mut Vec<Buffer>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Surface, Error> {
    let encoder = command_buffer.new_compute_command_encoder();
    let result = (|| {
        let mut bands = Vec::<DirectBandSlice>::new();
        let mut final_surface = None;
        let mut step_idx = 0;

        while step_idx < plan.steps.len() {
            if let Some(group) = plan.classic_group_starting_at(step_idx) {
                let output = take_f32_scratch_buffer(runtime, group.total_coefficients);
                let (buffers, status_check) =
                    encode_prepared_classic_sub_band_group_to_buffer_in_encoder(
                        runtime,
                        encoder,
                        group,
                        &output.buffer,
                        scratch_buffers,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: member.offset_elements * size_of::<f32>(),
                    });
                }
                scratch_buffers.push(output);
                step_idx = group.end_step;
                continue;
            }

            if let Some(group) = plan.ht_group_starting_at(step_idx) {
                let output = take_f32_scratch_buffer(runtime, group.total_coefficients);
                let (buffers, status_check) =
                    encode_prepared_ht_sub_band_group_to_buffer_in_encoder(
                        runtime,
                        encoder,
                        group,
                        &output.buffer,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: member.offset_elements * size_of::<f32>(),
                    });
                }
                scratch_buffers.push(output);
                step_idx = group.end_step;
                continue;
            }

            let step = &plan.steps[step_idx];
            match step {
                PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                    let output = take_f32_scratch_buffer(
                        runtime,
                        sub_band.width as usize * sub_band.height as usize,
                    );
                    let (buffers, status_check) =
                        encode_prepared_classic_sub_band_to_buffer_in_encoder(
                            runtime,
                            encoder,
                            sub_band,
                            &output.buffer,
                            scratch_buffers,
                        )?;
                    retained_buffers.extend(buffers);
                    status_checks.push(status_check);
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                    let output = take_f32_scratch_buffer(
                        runtime,
                        sub_band.width as usize * sub_band.height as usize,
                    );
                    let (buffers, status_check) = encode_prepared_ht_sub_band_to_buffer_in_encoder(
                        runtime,
                        encoder,
                        sub_band,
                        &output.buffer,
                    )?;
                    retained_buffers.extend(buffers);
                    status_checks.push(status_check);
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::Idwt(idwt) => {
                    let (ll, ll_offset) =
                        lookup_direct_band_slice(&bands, idwt.ll_band_id, idwt.ll)?;
                    let (hl, hl_offset) =
                        lookup_direct_band_slice(&bands, idwt.hl_band_id, idwt.hl)?;
                    let (lh, lh_offset) =
                        lookup_direct_band_slice(&bands, idwt.lh_band_id, idwt.lh)?;
                    let (hh, hh_offset) =
                        lookup_direct_band_slice(&bands, idwt.hh_band_id, idwt.hh)?;
                    let params = J2kIdwtSingleDecompositionParams {
                        x0: idwt.rect.x0,
                        y0: idwt.rect.y0,
                        width: idwt.rect.width(),
                        height: idwt.rect.height(),
                        ll_width: idwt.ll.width(),
                        ll_height: idwt.ll.height(),
                        hl_width: idwt.hl.width(),
                        hl_height: idwt.hl.height(),
                        lh_width: idwt.lh.width(),
                        lh_height: idwt.lh.height(),
                        hh_width: idwt.hh.width(),
                        hh_height: idwt.hh.height(),
                    };
                    let output = take_f32_scratch_buffer(
                        runtime,
                        idwt.rect.width() as usize * idwt.rect.height() as usize,
                    );
                    match idwt.transform {
                        J2kWaveletTransform::Reversible53 => {
                            dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
                                runtime,
                                encoder,
                                &ll,
                                ll_offset,
                                &hl,
                                hl_offset,
                                &lh,
                                lh_offset,
                                &hh,
                                hh_offset,
                                params,
                                &output.buffer,
                                0,
                            );
                        }
                        J2kWaveletTransform::Irreversible97 => {
                            let status_check =
                                dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
                                    runtime,
                                    encoder,
                                    &ll,
                                    ll_offset,
                                    &hl,
                                    hl_offset,
                                    &lh,
                                    lh_offset,
                                    &hh,
                                    hh_offset,
                                    params,
                                    &output.buffer,
                                    0,
                                );
                            status_checks.push(status_check);
                        }
                    }
                    bands.push(DirectBandSlice {
                        band_id: idwt.output_band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::Store(store) => {
                    let (input, input_offset) =
                        lookup_direct_band_slice(&bands, store.input_band_id, store.input_rect)?;
                    if matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
                        let scale = j2k_scalar_pack_params(u32::from(plan.bit_depth));
                        final_surface = Some(encode_gray_store_to_surface_in_encoder(
                            runtime,
                            encoder,
                            &input,
                            input_offset,
                            J2kGrayStoreParams {
                                input_width: store.input_rect.width(),
                                source_x: store.source_x,
                                source_y: store.source_y,
                                copy_width: store.copy_width,
                                copy_height: store.copy_height,
                                output_width: store.output_width,
                                output_x: store.output_x,
                                output_y: store.output_y,
                                addend: store.addend,
                                max_value: scale.max_value,
                                u8_scale: scale.u8_scale,
                                u16_scale: scale.u16_scale,
                            },
                            plan.dimensions,
                            fmt,
                        )?);
                    } else {
                        let output = take_f32_scratch_buffer(
                            runtime,
                            store.output_width as usize * store.output_height as usize,
                        );
                        let params = J2kStoreParams {
                            input_width: store.input_rect.width(),
                            source_x: store.source_x,
                            source_y: store.source_y,
                            copy_width: store.copy_width,
                            copy_height: store.copy_height,
                            output_width: store.output_width,
                            output_x: store.output_x,
                            output_y: store.output_y,
                            addend: store.addend,
                        };
                        dispatch_store_component_buffer_in_encoder_with_offsets(
                            runtime,
                            encoder,
                            &input,
                            input_offset,
                            &output.buffer,
                            0,
                            params,
                        );
                        retained_buffers.push(output.buffer.clone());
                        final_surface = Some(encode_gray_plane_to_surface_in_encoder(
                            runtime,
                            encoder,
                            &output.buffer,
                            plan.dimensions,
                            plan.bit_depth,
                            fmt,
                        )?);
                        scratch_buffers.push(output);
                    }
                }
            }
            step_idx += 1;
        }

        final_surface.ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect prepared grayscale plan did not produce a final stored plane"
                .to_string(),
        })
    })();
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
fn encode_prepared_direct_component_plane_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &PreparedDirectGrayscalePlan,
    retained_buffers: &mut Vec<Buffer>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Buffer, Error> {
    let encoder = command_buffer.new_compute_command_encoder();
    let result = (|| {
        let mut bands = Vec::<DirectBandSlice>::new();
        let mut final_plane = None;
        let mut step_idx = 0;

        while step_idx < plan.steps.len() {
            if let Some(group) = plan.classic_group_starting_at(step_idx) {
                let output = take_f32_scratch_buffer(runtime, group.total_coefficients);
                let (buffers, status_check) =
                    encode_prepared_classic_sub_band_group_to_buffer_in_encoder(
                        runtime,
                        encoder,
                        group,
                        &output.buffer,
                        scratch_buffers,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: member.offset_elements * size_of::<f32>(),
                    });
                }
                scratch_buffers.push(output);
                step_idx = group.end_step;
                continue;
            }

            if let Some(group) = plan.ht_group_starting_at(step_idx) {
                let output = take_f32_scratch_buffer(runtime, group.total_coefficients);
                let (buffers, status_check) =
                    encode_prepared_ht_sub_band_group_to_buffer_in_encoder(
                        runtime,
                        encoder,
                        group,
                        &output.buffer,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: member.offset_elements * size_of::<f32>(),
                    });
                }
                scratch_buffers.push(output);
                step_idx = group.end_step;
                continue;
            }

            match &plan.steps[step_idx] {
                PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                    let output = take_f32_scratch_buffer(
                        runtime,
                        sub_band.width as usize * sub_band.height as usize,
                    );
                    let (buffers, status_check) =
                        encode_prepared_classic_sub_band_to_buffer_in_encoder(
                            runtime,
                            encoder,
                            sub_band,
                            &output.buffer,
                            scratch_buffers,
                        )?;
                    retained_buffers.extend(buffers);
                    status_checks.push(status_check);
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                    let output = take_f32_scratch_buffer(
                        runtime,
                        sub_band.width as usize * sub_band.height as usize,
                    );
                    let (buffers, status_check) = encode_prepared_ht_sub_band_to_buffer_in_encoder(
                        runtime,
                        encoder,
                        sub_band,
                        &output.buffer,
                    )?;
                    retained_buffers.extend(buffers);
                    status_checks.push(status_check);
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::Idwt(idwt) => {
                    let (ll, ll_offset) =
                        lookup_direct_band_slice(&bands, idwt.ll_band_id, idwt.ll)?;
                    let (hl, hl_offset) =
                        lookup_direct_band_slice(&bands, idwt.hl_band_id, idwt.hl)?;
                    let (lh, lh_offset) =
                        lookup_direct_band_slice(&bands, idwt.lh_band_id, idwt.lh)?;
                    let (hh, hh_offset) =
                        lookup_direct_band_slice(&bands, idwt.hh_band_id, idwt.hh)?;
                    let params = J2kIdwtSingleDecompositionParams {
                        x0: idwt.rect.x0,
                        y0: idwt.rect.y0,
                        width: idwt.rect.width(),
                        height: idwt.rect.height(),
                        ll_width: idwt.ll.width(),
                        ll_height: idwt.ll.height(),
                        hl_width: idwt.hl.width(),
                        hl_height: idwt.hl.height(),
                        lh_width: idwt.lh.width(),
                        lh_height: idwt.lh.height(),
                        hh_width: idwt.hh.width(),
                        hh_height: idwt.hh.height(),
                    };
                    let output = take_f32_scratch_buffer(
                        runtime,
                        idwt.rect.width() as usize * idwt.rect.height() as usize,
                    );
                    match idwt.transform {
                        J2kWaveletTransform::Reversible53 => {
                            dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
                                runtime,
                                encoder,
                                &ll,
                                ll_offset,
                                &hl,
                                hl_offset,
                                &lh,
                                lh_offset,
                                &hh,
                                hh_offset,
                                params,
                                &output.buffer,
                                0,
                            );
                        }
                        J2kWaveletTransform::Irreversible97 => {
                            status_checks.push(
                                dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
                                    runtime,
                                    encoder,
                                    &ll,
                                    ll_offset,
                                    &hl,
                                    hl_offset,
                                    &lh,
                                    lh_offset,
                                    &hh,
                                    hh_offset,
                                    params,
                                    &output.buffer,
                                    0,
                                ),
                            );
                        }
                    }
                    bands.push(DirectBandSlice {
                        band_id: idwt.output_band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::Store(store) => {
                    let (input, input_offset) =
                        lookup_direct_band_slice(&bands, store.input_band_id, store.input_rect)?;
                    let output = take_f32_scratch_buffer(
                        runtime,
                        store.output_width as usize * store.output_height as usize,
                    );
                    dispatch_store_component_buffer_in_encoder_with_offsets(
                        runtime,
                        encoder,
                        &input,
                        input_offset,
                        &output.buffer,
                        0,
                        J2kStoreParams {
                            input_width: store.input_rect.width(),
                            source_x: store.source_x,
                            source_y: store.source_y,
                            copy_width: store.copy_width,
                            copy_height: store.copy_height,
                            output_width: store.output_width,
                            output_x: store.output_x,
                            output_y: store.output_y,
                            addend: store.addend,
                        },
                    );
                    final_plane = Some(output.buffer.clone());
                    scratch_buffers.push(output);
                }
            }
            step_idx += 1;
        }

        final_plane.ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect component plan did not produce a stored plane".to_string(),
        })
    })();
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_repeated_prepared_direct_grayscale_plan(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    count: usize,
) -> Result<Vec<Surface>, Error> {
    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut retained_host_data = Vec::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let surfaces = encode_repeated_direct_grayscale_plan_in_command_buffer(
            runtime,
            command_buffer,
            plan,
            fmt,
            count,
            &mut retained_buffers,
            &mut retained_host_data,
            &mut status_checks,
            &mut scratch_buffers,
        )?;
        command_buffer.commit();
        command_buffer.wait_until_completed();
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        drop(retained_host_data);
        recycle_scratch_buffers(runtime, scratch_buffers);
        Ok(surfaces)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_grayscale_plan(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let surface = encode_prepared_direct_grayscale_plan_in_command_buffer(
            runtime,
            command_buffer,
            plan,
            fmt,
            &mut retained_buffers,
            &mut status_checks,
            &mut scratch_buffers,
        )?;
        command_buffer.commit();
        command_buffer.wait_until_completed();
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        recycle_scratch_buffers(runtime, scratch_buffers);
        Ok(surface)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_grayscale_plan_with_device(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        execute_prepared_direct_grayscale_plan(plan, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_grayscale_plan_batch(
    plans: &[Arc<PreparedDirectGrayscalePlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    if plans.is_empty() {
        return Ok(Vec::new());
    }

    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let mut surfaces = Vec::with_capacity(plans.len());

        for plan in plans {
            surfaces.push(encode_prepared_direct_grayscale_plan_in_command_buffer(
                runtime,
                command_buffer,
                plan,
                fmt,
                &mut retained_buffers,
                &mut status_checks,
                &mut scratch_buffers,
            )?);
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        recycle_scratch_buffers(runtime, scratch_buffers);
        Ok(surfaces)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let plans = [Arc::new(plan.clone())];
    let mut surfaces = execute_prepared_direct_color_plan_batch(&plans, fmt)?;
    surfaces.pop().ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect color plan produced no surface".to_string(),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan_with_device(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| execute_prepared_direct_color_plan(plan, fmt))
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan_batch(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    if plans.is_empty() {
        return Ok(Vec::new());
    }
    if plans
        .iter()
        .any(|plan| !prepared_direct_color_plan_supports_runtime(plan, fmt))
    {
        return Err(Error::MetalKernel {
            message: "unsupported classic kernel input in direct component plan".to_string(),
        });
    }

    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();

        if fmt == PixelFormat::Rgb8 {
            if let Some(surfaces) = try_encode_stacked_mct_rgb8_direct_color_batch(
                runtime,
                command_buffer,
                plans,
                &mut retained_buffers,
                &mut status_checks,
                &mut scratch_buffers,
            )? {
                command_buffer.commit();
                command_buffer.wait_until_completed();
                for status_check in status_checks {
                    validate_direct_status(status_check)?;
                }
                drop(retained_buffers);
                recycle_scratch_buffers(runtime, scratch_buffers);
                return Ok(surfaces);
            }
        }

        let mut surfaces = Vec::with_capacity(plans.len());

        for plan in plans {
            let surface = encode_prepared_direct_color_plan_in_command_buffer(
                runtime,
                command_buffer,
                plan,
                fmt,
                &mut retained_buffers,
                &mut status_checks,
                &mut scratch_buffers,
            )?;
            surfaces.push(surface);
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        recycle_scratch_buffers(runtime, scratch_buffers);
        Ok(surfaces)
    })
}

#[cfg(target_os = "macos")]
fn signed_sample_bias(bit_depth: u8) -> f32 {
    2.0_f32.powi(i32::from(bit_depth) - 1)
}

#[cfg(target_os = "macos")]
fn encode_prepared_direct_color_plan_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
    retained_buffers: &mut Vec<Buffer>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Surface, Error> {
    if plan.component_plans.len() != 3 {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K MetalDirect color execution expected 3 component plans, got {}",
                plan.component_plans.len()
            ),
        });
    }

    let mut planes = Vec::with_capacity(3);
    for component_plan in &plan.component_plans {
        planes.push(encode_prepared_direct_component_plane_in_command_buffer(
            runtime,
            command_buffer,
            component_plan,
            retained_buffers,
            status_checks,
            scratch_buffers,
        )?);
    }

    if plan.mct && fmt == PixelFormat::Rgb8 {
        return Ok(encode_mct_rgb8_to_surface_in_command_buffer(
            runtime,
            command_buffer,
            [&planes[0], &planes[1], &planes[2]],
            plan.dimensions,
            plan.bit_depths,
            plan.transform,
        ));
    }

    if plan.mct {
        let len = plan.dimensions.0 as usize * plan.dimensions.1 as usize;
        status_checks.push(dispatch_inverse_mct_buffers_in_command_buffer(
            runtime,
            command_buffer,
            [&planes[0], &planes[1], &planes[2]],
            len,
            plan.transform,
            [
                signed_sample_bias(plan.bit_depths[0]),
                signed_sample_bias(plan.bit_depths[1]),
                signed_sample_bias(plan.bit_depths[2]),
            ],
        )?);
    }

    let stage = PlaneStage {
        dims: plan.dimensions,
        plane_count: 3,
        color_space: NativeColorSpace::RGB,
        has_alpha: false,
        bit_depths: [
            u32::from(plan.bit_depths[0]),
            u32::from(plan.bit_depths[1]),
            u32::from(plan.bit_depths[2]),
            0,
        ],
        planes: [
            Some(planes[0].clone()),
            Some(planes[1].clone()),
            Some(planes[2].clone()),
            None,
        ],
    };
    encode_plane_stage_to_surface_in_command_buffer(runtime, command_buffer, &stage, fmt)
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct DirectBandSlice {
    band_id: J2kDirectBandId,
    buffer: Buffer,
    offset_bytes: usize,
}

#[cfg(target_os = "macos")]
fn lookup_direct_band_slice(
    bands: &[DirectBandSlice],
    band_id: J2kDirectBandId,
    rect: signinum_j2k_native::J2kRect,
) -> Result<(Buffer, usize), Error> {
    bands
        .iter()
        .find(|existing| existing.band_id == band_id)
        .map(|existing| (existing.buffer.clone(), existing.offset_bytes))
        .ok_or_else(|| Error::MetalKernel {
            message: format!(
                "missing J2K MetalDirect device band {} for rect ({}, {}, {}, {})",
                band_id, rect.x0, rect.y0, rect.x1, rect.y1
            ),
        })
}

#[cfg(target_os = "macos")]
fn lookup_repeated_direct_band_layout(
    band_sets: &[Vec<DirectBandSlice>],
    band_id: J2kDirectBandId,
    rect: signinum_j2k_native::J2kRect,
) -> Result<(Buffer, usize, u32), Error> {
    let first_bands = band_sets.first().ok_or_else(|| Error::MetalKernel {
        message: "missing J2K MetalDirect repeated band set".to_string(),
    })?;
    let (buffer, offset_bytes) = lookup_direct_band_slice(first_bands, band_id, rect)?;
    let stride_bytes = if let Some(second_bands) = band_sets.get(1) {
        let (_, next_offset_bytes) = lookup_direct_band_slice(second_bands, band_id, rect)?;
        next_offset_bytes
            .checked_sub(offset_bytes)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect repeated band offsets are not monotonic".to_string(),
            })?
    } else {
        rect.width() as usize * rect.height() as usize * size_of::<f32>()
    };
    if stride_bytes % size_of::<f32>() != 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect repeated band stride is not f32-aligned".to_string(),
        });
    }
    let stride_elements =
        u32::try_from(stride_bytes / size_of::<f32>()).map_err(|_| Error::MetalKernel {
            message: "J2K MetalDirect repeated band stride exceeds u32".to_string(),
        })?;
    Ok((buffer, offset_bytes, stride_elements))
}

#[cfg(target_os = "macos")]
struct StackedDirectComponentPlane {
    buffer: Buffer,
    dimensions: (u32, u32),
    count: usize,
}

#[cfg(target_os = "macos")]
fn try_encode_stacked_mct_rgb8_direct_color_batch(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plans: &[Arc<PreparedDirectColorPlan>],
    retained_buffers: &mut Vec<Buffer>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Option<Vec<Surface>>, Error> {
    let Some(first) = plans.first() else {
        return Ok(Some(Vec::new()));
    };
    if plans.len() <= 1
        || !first.mct
        || first.component_plans.len() != 3
        || !plans.iter().all(|plan| {
            plan.mct
                && plan.dimensions == first.dimensions
                && plan.bit_depths == first.bit_depths
                && plan.transform == first.transform
                && plan.component_plans.len() == 3
        })
    {
        return Ok(None);
    }

    let mut stacked_planes = Vec::with_capacity(3);
    for component_idx in 0..3 {
        let component_plan_refs = plans
            .iter()
            .map(|plan| &plan.component_plans[component_idx])
            .collect::<Vec<_>>();
        if !supports_stacked_direct_component_plane_batch(&component_plan_refs) {
            return Ok(None);
        }
        stacked_planes.push(encode_stacked_direct_component_plane_batch(
            runtime,
            command_buffer,
            &component_plan_refs,
            retained_buffers,
            status_checks,
            scratch_buffers,
        )?);
    }

    if !stacked_planes
        .iter()
        .all(|plane| plane.dimensions == first.dimensions && plane.count == plans.len())
    {
        return Ok(None);
    }

    let surfaces = encode_batched_mct_rgb8_to_surfaces_in_command_buffer(
        runtime,
        command_buffer,
        [
            &stacked_planes[0].buffer,
            &stacked_planes[1].buffer,
            &stacked_planes[2].buffer,
        ],
        first.dimensions,
        plans.len(),
        first.bit_depths,
        first.transform,
    )?;
    Ok(Some(surfaces))
}

#[cfg(target_os = "macos")]
fn supports_stacked_direct_component_plane_batch(plans: &[&PreparedDirectGrayscalePlan]) -> bool {
    let Some(first) = plans.first() else {
        return false;
    };
    if plans.iter().any(|plan| {
        plan.dimensions != first.dimensions
            || plan.bit_depth != first.bit_depth
            || plan.steps.len() != first.steps.len()
    }) {
        return false;
    }

    let mut step_idx = 0;
    while step_idx < first.steps.len() {
        if let Some(group) = first.classic_group_starting_at(step_idx) {
            if group.end_step <= step_idx
                || !plans.iter().all(|plan| {
                    plan.classic_group_starting_at(step_idx)
                        .is_some_and(|other| classic_group_shapes_match(group, other))
                })
            {
                return false;
            }
            step_idx = group.end_step;
            continue;
        }
        if first.ht_group_starting_at(step_idx).is_some() {
            return false;
        }

        match &first.steps[step_idx] {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                if !plans.iter().all(|plan| {
                    matches!(
                        &plan.steps[step_idx],
                        PreparedDirectGrayscaleStep::ClassicSubBand(other)
                            if classic_sub_band_shapes_match(sub_band, other)
                    )
                }) {
                    return false;
                }
            }
            PreparedDirectGrayscaleStep::HtSubBand(_) => return false,
            PreparedDirectGrayscaleStep::Idwt(idwt) => {
                if !plans.iter().all(|plan| {
                    matches!(
                        &plan.steps[step_idx],
                        PreparedDirectGrayscaleStep::Idwt(other)
                            if idwt_shapes_match(idwt, other)
                    )
                }) {
                    return false;
                }
            }
            PreparedDirectGrayscaleStep::Store(store) => {
                if !plans.iter().all(|plan| {
                    matches!(
                        &plan.steps[step_idx],
                        PreparedDirectGrayscaleStep::Store(other)
                            if store_shapes_match(store, other)
                    )
                }) {
                    return false;
                }
            }
        }
        step_idx += 1;
    }

    true
}

#[cfg(target_os = "macos")]
fn prepared_direct_color_plan_supports_runtime(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
) -> bool {
    matches!(
        fmt,
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
    ) && plan.component_plans.len() == 3
        && plan
            .component_plans
            .iter()
            .all(prepared_direct_component_plan_supports_runtime)
}

#[cfg(target_os = "macos")]
fn prepared_direct_component_plan_supports_runtime(plan: &PreparedDirectGrayscalePlan) -> bool {
    plan.steps.iter().all(|step| match step {
        PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => sub_band
            .jobs
            .iter()
            .all(|job| classic_prepared_job_supports_runtime(job, &sub_band.segments)),
        PreparedDirectGrayscaleStep::HtSubBand(_)
        | PreparedDirectGrayscaleStep::Idwt(_)
        | PreparedDirectGrayscaleStep::Store(_) => true,
    }) && plan.classic_groups.iter().all(|group| {
        group
            .jobs
            .iter()
            .all(|job| classic_prepared_job_supports_runtime(job, &group.segments))
    })
}

#[cfg(target_os = "macos")]
fn classic_prepared_job_supports_runtime(
    job: &J2kClassicCleanupBatchJob,
    segments: &[J2kClassicSegment],
) -> bool {
    if job.width == 0 || job.height == 0 {
        return true;
    }
    if job.width > J2K_CLASSIC_MAX_WIDTH || job.height > J2K_CLASSIC_MAX_HEIGHT {
        return false;
    }
    if job.output_stride < job.width {
        return false;
    }
    if job.total_bitplanes == 0 || job.total_bitplanes > 31 || job.missing_msbs >= 31 {
        return false;
    }
    let bitplanes = job.total_bitplanes.saturating_sub(job.missing_msbs);
    if bitplanes == 0 {
        return false;
    }
    let max_coding_passes = 1 + 3 * (bitplanes - 1);
    if job.number_of_coding_passes == 0 || job.number_of_coding_passes > max_coding_passes {
        return false;
    }

    let start = job.segment_offset as usize;
    let count = job.segment_count as usize;
    let Some(end) = start.checked_add(count) else {
        return false;
    };
    if end > segments.len() || count == 0 {
        return false;
    }

    let uses_bypass = (job.style_flags & J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) != 0;
    let mut expected_start = 0u32;
    let mut expected_offset = job.coded_offset;
    for segment in &segments[start..end] {
        if segment.start_coding_pass != expected_start
            || segment.start_coding_pass > segment.end_coding_pass
        {
            return false;
        }
        if uses_bypass {
            let expected_arithmetic =
                segment.start_coding_pass <= 9 || segment.start_coding_pass % 3 == 0;
            if (segment.use_arithmetic != 0) != expected_arithmetic {
                return false;
            }
            if segment.use_arithmetic == 0 {
                if segment.start_coding_pass % 3 != 1 {
                    return false;
                }
                if segment
                    .end_coding_pass
                    .saturating_sub(segment.start_coding_pass)
                    > 2
                {
                    return false;
                }
                if (segment.start_coding_pass..segment.end_coding_pass).any(|pass| pass % 3 == 0) {
                    return false;
                }
            }
        } else if segment.use_arithmetic == 0 {
            return false;
        }

        let Some(data_end) = segment.data_offset.checked_add(segment.data_length) else {
            return false;
        };
        if segment.data_offset != expected_offset
            || segment.data_offset < job.coded_offset
            || data_end > job.coded_offset.saturating_add(job.coded_len)
        {
            return false;
        }
        expected_offset = data_end;
        expected_start = segment.end_coding_pass;
    }

    expected_start == job.number_of_coding_passes
        && expected_offset == job.coded_offset.saturating_add(job.coded_len)
}

#[cfg(target_os = "macos")]
fn classic_group_shapes_match(
    first: &PreparedClassicSubBandGroup,
    other: &PreparedClassicSubBandGroup,
) -> bool {
    first.end_step == other.end_step
        && first.total_coefficients == other.total_coefficients
        && first.members.len() == other.members.len()
        && first
            .members
            .iter()
            .zip(&other.members)
            .all(|(left, right)| left.offset_elements == right.offset_elements)
}

#[cfg(target_os = "macos")]
fn classic_sub_band_shapes_match(
    first: &PreparedClassicSubBand,
    other: &PreparedClassicSubBand,
) -> bool {
    first.width == other.width && first.height == other.height
}

#[cfg(target_os = "macos")]
fn rect_shapes_match(
    first: signinum_j2k_native::J2kRect,
    other: signinum_j2k_native::J2kRect,
) -> bool {
    first.x0 == other.x0 && first.y0 == other.y0 && first.x1 == other.x1 && first.y1 == other.y1
}

#[cfg(target_os = "macos")]
fn idwt_shapes_match(first: &J2kDirectIdwtStep, other: &J2kDirectIdwtStep) -> bool {
    first.transform == other.transform
        && rect_shapes_match(first.rect, other.rect)
        && rect_shapes_match(first.ll, other.ll)
        && rect_shapes_match(first.hl, other.hl)
        && rect_shapes_match(first.lh, other.lh)
        && rect_shapes_match(first.hh, other.hh)
}

#[cfg(target_os = "macos")]
fn store_shapes_match(first: &J2kDirectStoreStep, other: &J2kDirectStoreStep) -> bool {
    rect_shapes_match(first.input_rect, other.input_rect)
        && first.source_x == other.source_x
        && first.source_y == other.source_y
        && first.copy_width == other.copy_width
        && first.copy_height == other.copy_height
        && first.output_width == other.output_width
        && first.output_height == other.output_height
        && first.output_x == other.output_x
        && first.output_y == other.output_y
        && first.addend.to_bits() == other.addend.to_bits()
}

#[cfg(target_os = "macos")]
fn encode_stacked_direct_component_plane_batch(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plans: &[&PreparedDirectGrayscalePlan],
    retained_buffers: &mut Vec<Buffer>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<StackedDirectComponentPlane, Error> {
    let Some(first) = plans.first() else {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect color batch has no component plans".to_string(),
        });
    };

    let count = plans.len();
    let mut band_sets = vec![Vec::<DirectBandSlice>::new(); count];
    let mut final_plane = None;
    let mut step_idx = 0;

    while step_idx < first.steps.len() {
        if let Some(group) = first.classic_group_starting_at(step_idx) {
            let groups = plans
                .iter()
                .map(|plan| {
                    plan.classic_group_starting_at(step_idx)
                        .expect("preflight validated classic group")
                })
                .collect::<Vec<_>>();
            let output = take_f32_scratch_buffer(runtime, group.total_coefficients * count);
            let (buffers, status_check) =
                encode_distinct_classic_sub_band_groups_to_buffer_in_command_buffer(
                    runtime,
                    command_buffer,
                    &groups,
                    &output.buffer,
                    scratch_buffers,
                )?;
            retained_buffers.extend(buffers);
            status_checks.push(status_check);
            let stride_bytes = group.total_coefficients * size_of::<f32>();
            for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                for member in &groups[instance_idx].members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes
                            + member.offset_elements * size_of::<f32>(),
                    });
                }
            }
            scratch_buffers.push(output);
            step_idx = group.end_step;
            continue;
        }

        match &first.steps[step_idx] {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                let sub_bands = plans
                    .iter()
                    .map(|plan| match &plan.steps[step_idx] {
                        PreparedDirectGrayscaleStep::ClassicSubBand(other) => other,
                        _ => unreachable!("preflight validated classic sub-band"),
                    })
                    .collect::<Vec<_>>();
                let per_instance_len = sub_band.width as usize * sub_band.height as usize;
                let output = take_f32_scratch_buffer(runtime, per_instance_len * count);
                let (buffers, status_check) =
                    encode_distinct_classic_sub_bands_to_buffer_in_command_buffer(
                        runtime,
                        command_buffer,
                        &sub_bands,
                        &output.buffer,
                        scratch_buffers,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    bands.push(DirectBandSlice {
                        band_id: sub_bands[instance_idx].band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes,
                    });
                }
                scratch_buffers.push(output);
            }
            PreparedDirectGrayscaleStep::HtSubBand(_) => unreachable!("preflight rejected HT"),
            PreparedDirectGrayscaleStep::Idwt(idwt) => {
                let per_instance_len = idwt.rect.width() as usize * idwt.rect.height() as usize;
                let output = take_f32_scratch_buffer(runtime, per_instance_len * count);
                match idwt.transform {
                    J2kWaveletTransform::Reversible53 => {
                        let (ll, ll_offset, low_low_stride) = lookup_repeated_direct_band_layout(
                            &band_sets,
                            idwt.ll_band_id,
                            idwt.ll,
                        )?;
                        let (hl, hl_offset, high_low_stride) = lookup_repeated_direct_band_layout(
                            &band_sets,
                            idwt.hl_band_id,
                            idwt.hl,
                        )?;
                        let (lh, lh_offset, low_high_stride) = lookup_repeated_direct_band_layout(
                            &band_sets,
                            idwt.lh_band_id,
                            idwt.lh,
                        )?;
                        let (hh, hh_offset, high_high_stride) = lookup_repeated_direct_band_layout(
                            &band_sets,
                            idwt.hh_band_id,
                            idwt.hh,
                        )?;
                        dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets(
                            runtime,
                            command_buffer,
                            &ll,
                            ll_offset,
                            &hl,
                            hl_offset,
                            &lh,
                            lh_offset,
                            &hh,
                            hh_offset,
                            J2kRepeatedIdwtSingleDecompositionParams {
                                x0: idwt.rect.x0,
                                y0: idwt.rect.y0,
                                width: idwt.rect.width(),
                                height: idwt.rect.height(),
                                ll_width: idwt.ll.width(),
                                ll_height: idwt.ll.height(),
                                hl_width: idwt.hl.width(),
                                hl_height: idwt.hl.height(),
                                lh_width: idwt.lh.width(),
                                lh_height: idwt.lh.height(),
                                hh_width: idwt.hh.width(),
                                hh_height: idwt.hh.height(),
                                ll_instance_stride: low_low_stride,
                                hl_instance_stride: high_low_stride,
                                lh_instance_stride: low_high_stride,
                                hh_instance_stride: high_high_stride,
                                batch_count: u32::try_from(count).map_err(|_| {
                                    Error::MetalKernel {
                                        message:
                                            "J2K MetalDirect color IDWT batch count exceeds u32"
                                                .to_string(),
                                    }
                                })?,
                            },
                            &output.buffer,
                        );
                    }
                    J2kWaveletTransform::Irreversible97 => {
                        for (instance_idx, bands) in band_sets.iter().enumerate() {
                            let PreparedDirectGrayscaleStep::Idwt(step) =
                                &plans[instance_idx].steps[step_idx]
                            else {
                                unreachable!("preflight validated IDWT")
                            };
                            let (ll, ll_offset) =
                                lookup_direct_band_slice(bands, step.ll_band_id, step.ll)?;
                            let (hl, hl_offset) =
                                lookup_direct_band_slice(bands, step.hl_band_id, step.hl)?;
                            let (lh, lh_offset) =
                                lookup_direct_band_slice(bands, step.lh_band_id, step.lh)?;
                            let (hh, hh_offset) =
                                lookup_direct_band_slice(bands, step.hh_band_id, step.hh)?;
                            status_checks.push(
                                dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
                                    runtime,
                                    command_buffer,
                                    &ll,
                                    ll_offset,
                                    &hl,
                                    hl_offset,
                                    &lh,
                                    lh_offset,
                                    &hh,
                                    hh_offset,
                                    J2kIdwtSingleDecompositionParams {
                                        x0: step.rect.x0,
                                        y0: step.rect.y0,
                                        width: step.rect.width(),
                                        height: step.rect.height(),
                                        ll_width: step.ll.width(),
                                        ll_height: step.ll.height(),
                                        hl_width: step.hl.width(),
                                        hl_height: step.hl.height(),
                                        lh_width: step.lh.width(),
                                        lh_height: step.lh.height(),
                                        hh_width: step.hh.width(),
                                        hh_height: step.hh.height(),
                                    },
                                    &output.buffer,
                                    instance_idx * per_instance_len * size_of::<f32>(),
                                ),
                            );
                        }
                    }
                }
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    let PreparedDirectGrayscaleStep::Idwt(step) =
                        &plans[instance_idx].steps[step_idx]
                    else {
                        unreachable!("preflight validated IDWT")
                    };
                    bands.push(DirectBandSlice {
                        band_id: step.output_band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes,
                    });
                }
                scratch_buffers.push(output);
            }
            PreparedDirectGrayscaleStep::Store(store) => {
                let (input, _) =
                    lookup_direct_band_slice(&band_sets[0], store.input_band_id, store.input_rect)?;
                let per_instance_len = store.output_width as usize * store.output_height as usize;
                let output = take_f32_scratch_buffer(runtime, per_instance_len * count);
                dispatch_store_component_repeated_in_command_buffer(
                    runtime,
                    command_buffer,
                    &input,
                    &output.buffer,
                    J2kRepeatedStoreParams {
                        input_width: store.input_rect.width(),
                        input_height: store.input_rect.height(),
                        source_x: store.source_x,
                        source_y: store.source_y,
                        copy_width: store.copy_width,
                        copy_height: store.copy_height,
                        output_width: store.output_width,
                        output_height: store.output_height,
                        output_x: store.output_x,
                        output_y: store.output_y,
                        addend: store.addend,
                        batch_count: u32::try_from(count).map_err(|_| Error::MetalKernel {
                            message: "J2K MetalDirect color store batch count exceeds u32"
                                .to_string(),
                        })?,
                    },
                );
                final_plane = Some(output.buffer.clone());
                scratch_buffers.push(output);
            }
        }
        step_idx += 1;
    }

    let buffer = final_plane.ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect color component batch did not produce a final plane".to_string(),
    })?;
    Ok(StackedDirectComponentPlane {
        buffer,
        dimensions: first.dimensions,
        count,
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_repeated_direct_grayscale_plan_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    count: usize,
    retained_buffers: &mut Vec<Buffer>,
    retained_host_data: &mut Vec<DirectHostRetention>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Vec<Surface>, Error> {
    let mut band_sets = vec![Vec::<DirectBandSlice>::new(); count];
    let mut surfaces = Vec::with_capacity(count);
    let mut stacked_outputs = true;
    let mut step_idx = 0;

    while step_idx < plan.steps.len() {
        if let Some(group) = plan.classic_group_starting_at(step_idx) {
            let per_instance_len = group.total_coefficients;
            let output = take_f32_scratch_buffer(runtime, per_instance_len * count);
            let (host_data, buffers, status_check) =
                encode_repeated_classic_sub_band_group_to_buffer_in_command_buffer(
                    runtime,
                    command_buffer,
                    group,
                    count,
                    &output.buffer,
                    scratch_buffers,
                )?;
            retained_host_data.extend(host_data);
            retained_buffers.extend(buffers);
            status_checks.push(status_check);
            let stride_bytes = per_instance_len * size_of::<f32>();
            for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes
                            + member.offset_elements * size_of::<f32>(),
                    });
                }
            }
            scratch_buffers.push(output);
            step_idx = group.end_step;
            continue;
        }

        if let Some(group) = plan.ht_group_starting_at(step_idx) {
            let per_instance_len = group.total_coefficients;
            let output = take_f32_scratch_buffer(runtime, per_instance_len * count);
            let (host_data, buffers, status_check) =
                encode_repeated_ht_sub_band_group_to_buffer_in_command_buffer(
                    runtime,
                    command_buffer,
                    group,
                    count,
                    &output.buffer,
                )?;
            retained_host_data.extend(host_data);
            retained_buffers.extend(buffers);
            status_checks.push(status_check);
            let stride_bytes = per_instance_len * size_of::<f32>();
            for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes
                            + member.offset_elements * size_of::<f32>(),
                    });
                }
            }
            scratch_buffers.push(output);
            step_idx = group.end_step;
            continue;
        }

        let step = &plan.steps[step_idx];
        match step {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                let per_instance_len = sub_band.width as usize * sub_band.height as usize;
                let output = take_f32_scratch_buffer(runtime, per_instance_len * count);
                let (host_data, buffers, status_check) =
                    encode_repeated_classic_sub_band_to_buffer_in_command_buffer(
                        runtime,
                        command_buffer,
                        sub_band,
                        count,
                        &output.buffer,
                        scratch_buffers,
                    )?;
                retained_host_data.extend(host_data);
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes,
                    });
                }
                scratch_buffers.push(output);
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                let per_instance_len = sub_band.width as usize * sub_band.height as usize;
                let output = take_f32_scratch_buffer(runtime, per_instance_len * count);
                let (host_data, buffers, status_check) =
                    encode_repeated_ht_sub_band_to_buffer_in_command_buffer(
                        runtime,
                        command_buffer,
                        sub_band,
                        count,
                        &output.buffer,
                    )?;
                retained_host_data.extend(host_data);
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes,
                    });
                }
                scratch_buffers.push(output);
            }
            PreparedDirectGrayscaleStep::Idwt(idwt) => {
                let params = J2kIdwtSingleDecompositionParams {
                    x0: idwt.rect.x0,
                    y0: idwt.rect.y0,
                    width: idwt.rect.width(),
                    height: idwt.rect.height(),
                    ll_width: idwt.ll.width(),
                    ll_height: idwt.ll.height(),
                    hl_width: idwt.hl.width(),
                    hl_height: idwt.hl.height(),
                    lh_width: idwt.lh.width(),
                    lh_height: idwt.lh.height(),
                    hh_width: idwt.hh.width(),
                    hh_height: idwt.hh.height(),
                };
                match idwt.transform {
                    J2kWaveletTransform::Reversible53 if stacked_outputs => {
                        let (ll, ll_offset, low_low_stride) = lookup_repeated_direct_band_layout(
                            &band_sets,
                            idwt.ll_band_id,
                            idwt.ll,
                        )?;
                        let (hl, hl_offset, high_low_stride) = lookup_repeated_direct_band_layout(
                            &band_sets,
                            idwt.hl_band_id,
                            idwt.hl,
                        )?;
                        let (lh, lh_offset, low_high_stride) = lookup_repeated_direct_band_layout(
                            &band_sets,
                            idwt.lh_band_id,
                            idwt.lh,
                        )?;
                        let (hh, hh_offset, high_high_stride) = lookup_repeated_direct_band_layout(
                            &band_sets,
                            idwt.hh_band_id,
                            idwt.hh,
                        )?;
                        let per_instance_len =
                            idwt.rect.width() as usize * idwt.rect.height() as usize;
                        let output = take_f32_scratch_buffer(runtime, per_instance_len * count);
                        dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets(
                            runtime,
                            command_buffer,
                            &ll,
                            ll_offset,
                            &hl,
                            hl_offset,
                            &lh,
                            lh_offset,
                            &hh,
                            hh_offset,
                            J2kRepeatedIdwtSingleDecompositionParams {
                                x0: params.x0,
                                y0: params.y0,
                                width: params.width,
                                height: params.height,
                                ll_width: params.ll_width,
                                ll_height: params.ll_height,
                                hl_width: params.hl_width,
                                hl_height: params.hl_height,
                                lh_width: params.lh_width,
                                lh_height: params.lh_height,
                                hh_width: params.hh_width,
                                hh_height: params.hh_height,
                                ll_instance_stride: low_low_stride,
                                hl_instance_stride: high_low_stride,
                                lh_instance_stride: low_high_stride,
                                hh_instance_stride: high_high_stride,
                                batch_count: u32::try_from(count).map_err(|_| {
                                    Error::MetalKernel {
                                        message:
                                            "J2K MetalDirect repeated IDWT batch count exceeds u32"
                                                .to_string(),
                                    }
                                })?,
                            },
                            &output.buffer,
                        );
                        let stride_bytes = per_instance_len * size_of::<f32>();
                        for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                            bands.push(DirectBandSlice {
                                band_id: idwt.output_band_id,
                                buffer: output.buffer.clone(),
                                offset_bytes: instance_idx * stride_bytes,
                            });
                        }
                        scratch_buffers.push(output);
                    }
                    _ => {
                        stacked_outputs = false;
                        for bands in &mut band_sets {
                            let (ll, ll_offset) =
                                lookup_direct_band_slice(bands, idwt.ll_band_id, idwt.ll)?;
                            let (hl, hl_offset) =
                                lookup_direct_band_slice(bands, idwt.hl_band_id, idwt.hl)?;
                            let (lh, lh_offset) =
                                lookup_direct_band_slice(bands, idwt.lh_band_id, idwt.lh)?;
                            let (hh, hh_offset) =
                                lookup_direct_band_slice(bands, idwt.hh_band_id, idwt.hh)?;
                            let output = take_f32_scratch_buffer(
                                runtime,
                                idwt.rect.width() as usize * idwt.rect.height() as usize,
                            );
                            match idwt.transform {
                                J2kWaveletTransform::Reversible53 => {
                                    dispatch_reversible53_single_decomposition_buffers_in_command_buffer_with_offsets(
                                        runtime,
                                        command_buffer,
                                        &ll,
                                        ll_offset,
                                        &hl,
                                        hl_offset,
                                        &lh,
                                        lh_offset,
                                        &hh,
                                        hh_offset,
                                        params,
                                        &output.buffer,
                                        0,
                                    );
                                }
                                J2kWaveletTransform::Irreversible97 => status_checks.push(
                                    dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
                                        runtime,
                                        command_buffer,
                                        &ll,
                                        ll_offset,
                                        &hl,
                                        hl_offset,
                                        &lh,
                                        lh_offset,
                                        &hh,
                                        hh_offset,
                                        params,
                                        &output.buffer,
                                        0,
                                    ),
                                ),
                            }
                            bands.push(DirectBandSlice {
                                band_id: idwt.output_band_id,
                                buffer: output.buffer.clone(),
                                offset_bytes: 0,
                            });
                            scratch_buffers.push(output);
                        }
                    }
                }
            }
            PreparedDirectGrayscaleStep::Store(store) => {
                if stacked_outputs {
                    let (input, _) = lookup_direct_band_slice(
                        &band_sets[0],
                        store.input_band_id,
                        store.input_rect,
                    )?;
                    let batch_count = u32::try_from(count).map_err(|_| Error::MetalKernel {
                        message: "J2K MetalDirect repeated store batch count exceeds u32"
                            .to_string(),
                    })?;
                    if matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
                        let scale = j2k_scalar_pack_params(u32::from(plan.bit_depth));
                        surfaces.extend(encode_repeated_gray_store_to_surfaces_in_command_buffer(
                            runtime,
                            command_buffer,
                            &input,
                            J2kRepeatedGrayStoreParams {
                                input_width: store.input_rect.width(),
                                input_height: store.input_rect.height(),
                                source_x: store.source_x,
                                source_y: store.source_y,
                                copy_width: store.copy_width,
                                copy_height: store.copy_height,
                                output_width: store.output_width,
                                output_height: store.output_height,
                                output_x: store.output_x,
                                output_y: store.output_y,
                                addend: store.addend,
                                batch_count,
                                max_value: scale.max_value,
                                u8_scale: scale.u8_scale,
                                u16_scale: scale.u16_scale,
                            },
                            plan.dimensions,
                            fmt,
                            count,
                        )?);
                    } else {
                        let per_instance_len =
                            store.output_width as usize * store.output_height as usize;
                        let output = take_f32_scratch_buffer(runtime, per_instance_len * count);
                        dispatch_store_component_repeated_in_command_buffer(
                            runtime,
                            command_buffer,
                            &input,
                            &output.buffer,
                            J2kRepeatedStoreParams {
                                input_width: store.input_rect.width(),
                                input_height: store.input_rect.height(),
                                source_x: store.source_x,
                                source_y: store.source_y,
                                copy_width: store.copy_width,
                                copy_height: store.copy_height,
                                output_width: store.output_width,
                                output_height: store.output_height,
                                output_x: store.output_x,
                                output_y: store.output_y,
                                addend: store.addend,
                                batch_count,
                            },
                        );
                        retained_buffers.push(output.buffer.clone());
                        surfaces.extend(encode_repeated_gray_plane_to_surfaces_in_command_buffer(
                            runtime,
                            command_buffer,
                            &output.buffer,
                            plan.dimensions,
                            plan.bit_depth,
                            fmt,
                            count,
                        )?);
                        scratch_buffers.push(output);
                    }
                } else {
                    for bands in &band_sets {
                        let (input, input_offset) =
                            lookup_direct_band_slice(bands, store.input_band_id, store.input_rect)?;
                        let output = take_f32_scratch_buffer(
                            runtime,
                            store.output_width as usize * store.output_height as usize,
                        );
                        let params = J2kStoreParams {
                            input_width: store.input_rect.width(),
                            source_x: store.source_x,
                            source_y: store.source_y,
                            copy_width: store.copy_width,
                            copy_height: store.copy_height,
                            output_width: store.output_width,
                            output_x: store.output_x,
                            output_y: store.output_y,
                            addend: store.addend,
                        };
                        dispatch_store_component_buffer_in_command_buffer_with_offsets(
                            runtime,
                            command_buffer,
                            &input,
                            input_offset,
                            &output.buffer,
                            0,
                            params,
                        );
                        retained_buffers.push(output.buffer.clone());
                        surfaces.push(encode_gray_plane_to_surface_in_command_buffer_with_offset(
                            runtime,
                            command_buffer,
                            &output.buffer,
                            0,
                            plan.dimensions,
                            plan.bit_depth,
                            fmt,
                        )?);
                        scratch_buffers.push(output);
                    }
                }
            }
        }
        step_idx += 1;
    }

    if surfaces.len() != count {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K MetalDirect repeated grayscale plan produced {} surfaces for count {}",
                surfaces.len(),
                count
            ),
        });
    }

    Ok(surfaces)
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
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
fn take_f32_scratch_buffer(runtime: &MetalRuntime, len: usize) -> DirectScratchBuffer {
    let bytes = len.max(1).saturating_mul(size_of::<f32>());
    DirectScratchBuffer {
        bytes,
        buffer: runtime.take_private_buffer(bytes),
    }
}

#[cfg(target_os = "macos")]
fn recycle_scratch_buffers(runtime: &MetalRuntime, scratch_buffers: Vec<DirectScratchBuffer>) {
    for scratch in scratch_buffers {
        runtime.recycle_private_buffer(scratch.bytes, scratch.buffer);
    }
}

#[cfg(target_os = "macos")]
fn validate_direct_status(status_check: DirectStatusCheck) -> Result<(), Error> {
    match status_check {
        DirectStatusCheck::Classic { buffer, len } => {
            let statuses = unsafe {
                core::slice::from_raw_parts(buffer.contents().cast::<J2kClassicStatus>(), len)
            };
            if let Some(status) = statuses
                .iter()
                .copied()
                .find(|status| status.code != J2K_CLASSIC_STATUS_OK)
            {
                return Err(decode_classic_status_error(status));
            }
        }
        DirectStatusCheck::Ht { buffer, len } => {
            let statuses = unsafe {
                core::slice::from_raw_parts(buffer.contents().cast::<J2kHtStatus>(), len)
            };
            if let Some(status) = statuses
                .iter()
                .copied()
                .find(|status| status.code != J2K_HT_STATUS_OK)
            {
                return Err(decode_ht_status_error(status));
            }
        }
        DirectStatusCheck::Idwt(buffer) => {
            let status = unsafe { buffer.contents().cast::<J2kIdwtStatus>().read() };
            if status.code != J2K_IDWT_STATUS_OK {
                return Err(decode_idwt_status_error(status));
            }
        }
        DirectStatusCheck::Mct(buffer) => {
            let status = unsafe { buffer.contents().cast::<J2kMctStatus>().read() };
            if status.code != J2K_MCT_STATUS_OK {
                return Err(decode_mct_status_error(status));
            }
        }
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn encode_gray_plane_to_surface_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plane: &Buffer,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let encoder = command_buffer.new_compute_command_encoder();
    let result =
        encode_gray_plane_to_surface_in_encoder(runtime, encoder, plane, dims, bit_depth, fmt);
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
fn encode_gray_plane_to_surface_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    plane: &Buffer,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    encode_gray_plane_to_surface_in_encoder_with_offset(
        runtime, encoder, plane, 0, dims, bit_depth, fmt,
    )
}

#[cfg(target_os = "macos")]
fn encode_gray_plane_to_surface_in_command_buffer_with_offset(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plane: &Buffer,
    plane_offset_bytes: usize,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let encoder = command_buffer.new_compute_command_encoder();
    let result = encode_gray_plane_to_surface_in_encoder_with_offset(
        runtime,
        encoder,
        plane,
        plane_offset_bytes,
        dims,
        bit_depth,
        fmt,
    );
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
fn encode_gray_plane_to_surface_in_encoder_with_offset(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    plane: &Buffer,
    plane_offset_bytes: usize,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let pitch_bytes = dims.0 as usize * fmt.bytes_per_pixel();
    let out_buffer = runtime.device.new_buffer(
        (pitch_bytes * dims.1 as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let (output_channels, opaque_alpha, pipeline) =
        output_shape_for(&NativeColorSpace::Gray, false, 1, fmt, runtime)?;
    let mut bit_depths = [0u32; 4];
    bit_depths[0] = u32::from(bit_depth);
    let (max_values, u8_scales, u16_scales) = j2k_pack_scale_arrays(bit_depths);
    let params = J2kPackParams {
        width: dims.0,
        height: dims.1,
        out_stride: u32::try_from(pitch_bytes).expect("J2K Metal output stride fits in u32"),
        output_channels,
        opaque_alpha,
        max_values,
        u8_scales,
        u16_scales,
    };

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(plane), plane_offset_bytes as u64);
    encoder.set_buffer(1, None, 0);
    encoder.set_buffer(2, None, 0);
    encoder.set_buffer(3, None, 0);
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

    Ok(Surface::from_metal_buffer(out_buffer, dims, fmt))
}

#[cfg(target_os = "macos")]
fn encode_repeated_gray_plane_to_surfaces_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plane: &Buffer,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
    count: usize,
) -> Result<Vec<Surface>, Error> {
    let count_u32 = u32::try_from(count).map_err(|_| Error::MetalKernel {
        message: "J2K Metal repeated grayscale surface count exceeds u32".to_string(),
    })?;
    let pitch_bytes = dims.0 as usize * fmt.bytes_per_pixel();
    let surface_bytes =
        pitch_bytes
            .checked_mul(dims.1 as usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal repeated grayscale surface size overflow".to_string(),
            })?;
    let total_bytes = surface_bytes
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal repeated grayscale output size overflow".to_string(),
        })?;
    let out_buffer = runtime
        .device
        .new_buffer(total_bytes as u64, MTLResourceOptions::StorageModeShared);
    let scale = j2k_scalar_pack_params(u32::from(bit_depth));
    let params = J2kRepeatedGrayPackParams {
        width: dims.0,
        height: dims.1,
        out_stride: u32::try_from(pitch_bytes).expect("J2K Metal output stride fits in u32"),
        batch_count: count_u32,
        max_value: scale.max_value,
        u8_scale: scale.u8_scale,
        u16_scale: scale.u16_scale,
    };
    let pipeline = match fmt {
        PixelFormat::Gray8 => &runtime.pack_u8_repeated_gray,
        PixelFormat::Gray16 => &runtime.pack_u16_repeated_gray,
        _ => {
            return Err(Error::MetalKernel {
                message: format!("J2K Metal repeated grayscale pack does not support {fmt:?}"),
            })
        }
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(plane), 0);
    encoder.set_buffer(1, Some(&out_buffer), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kRepeatedGrayPackParams>() as u64,
        (&raw const params).cast(),
    );
    let width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(width);
    let height = (max_threads / width).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(dims.0),
            height: u64::from(dims.1),
            depth: u64::from(count_u32),
        },
        MTLSize {
            width,
            height,
            depth: 1,
        },
    );
    encoder.end_encoding();

    let mut surfaces = Vec::with_capacity(count);
    for instance_idx in 0..count {
        surfaces.push(Surface::from_metal_buffer_with_offset(
            out_buffer.clone(),
            dims,
            fmt,
            instance_idx * surface_bytes,
        ));
    }
    Ok(surfaces)
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn owned_slice_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    let size = size_of_val(data).max(1);
    let buffer = device.new_buffer(size as u64, MTLResourceOptions::StorageModeShared);
    if !data.is_empty() {
        unsafe {
            core::ptr::copy_nonoverlapping(
                data.as_ptr().cast::<u8>(),
                buffer.contents().cast::<u8>(),
                size_of_val(data),
            );
        }
    }
    buffer
}

#[cfg(target_os = "macos")]
fn lookup_direct_band(
    bands: &[(J2kDirectBandId, Buffer)],
    band_id: J2kDirectBandId,
    rect: signinum_j2k_native::J2kRect,
) -> Result<Buffer, Error> {
    bands
        .iter()
        .find(|(existing, _)| *existing == band_id)
        .map(|(_, buffer)| buffer.to_owned())
        .ok_or_else(|| Error::MetalKernel {
            message: format!(
                "missing J2K MetalDirect device band {} for rect ({}, {}, {}, {})",
                band_id, rect.x0, rect.y0, rect.x1, rect.y1
            ),
        })
}

#[cfg(target_os = "macos")]
fn j2k_pack_kernel_name_for(
    color_space: &NativeColorSpace,
    has_alpha: bool,
    plane_count: usize,
    fmt: PixelFormat,
) -> Option<&'static str> {
    match (color_space, has_alpha, plane_count, fmt) {
        (NativeColorSpace::Gray, false, 1, PixelFormat::Gray8) => Some("j2k_pack_gray8"),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgb8)
        | (NativeColorSpace::RGB, true, 4, PixelFormat::Rgb8) => Some("j2k_pack_rgb8"),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgba8) => Some("j2k_pack_rgb_opaque_rgba8"),
        (NativeColorSpace::RGB, true, 4, PixelFormat::Rgba8) => Some("j2k_pack_rgba8"),
        (NativeColorSpace::Gray, false, 1, PixelFormat::Gray16) => Some("j2k_pack_gray16"),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgb16) => Some("j2k_pack_rgb16"),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
fn j2k_pack_pipeline_for<'a>(
    runtime: &'a MetalRuntime,
    kernel_name: &str,
) -> &'a ComputePipelineState {
    match kernel_name {
        "j2k_pack_gray8" => &runtime.pack_gray8,
        "j2k_pack_rgb8" => &runtime.pack_rgb8,
        "j2k_pack_rgb_opaque_rgba8" => &runtime.pack_rgb_opaque_rgba8,
        "j2k_pack_rgba8" => &runtime.pack_rgba8,
        "j2k_pack_gray16" => &runtime.pack_gray16,
        "j2k_pack_rgb16" => &runtime.pack_rgb16,
        _ => unreachable!("validated J2K pack kernel name"),
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
    let Some(kernel_name) = j2k_pack_kernel_name_for(color_space, has_alpha, plane_count, fmt)
    else {
        return Err(Error::MetalKernel {
            message: format!(
                "unsupported J2K Metal mapping for {color_space:?}, alpha={has_alpha}, planes={plane_count}, fmt={fmt:?}"
            ),
        });
    };
    let (output_channels, opaque_alpha) = match (color_space, has_alpha, plane_count, fmt) {
        (NativeColorSpace::Gray, false, 1, PixelFormat::Gray8 | PixelFormat::Gray16) => (1, 0),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgb8 | PixelFormat::Rgb16)
        | (NativeColorSpace::RGB, true, 4, PixelFormat::Rgb8) => (3, 0),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgba8) => (4, 1),
        (NativeColorSpace::RGB, true, 4, PixelFormat::Rgba8) => (4, 0),
        _ => unreachable!("validated J2K pack shape"),
    };
    Ok((
        output_channels,
        opaque_alpha,
        j2k_pack_pipeline_for(runtime, kernel_name),
    ))
}

#[cfg(target_os = "macos")]
fn required_classic_output_len(job: J2kCodeBlockDecodeJob<'_>) -> Result<usize, Error> {
    if job.height == 0 {
        return Ok(0);
    }

    job.output_stride
        .checked_mul(job.height as usize - 1)
        .and_then(|prefix| prefix.checked_add(job.width as usize))
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal output size overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
fn classic_style_flags(style: signinum_j2k_native::J2kCodeBlockStyle) -> u32 {
    let mut flags = 0u32;
    if style.reset_context_probabilities {
        flags |= J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES;
    }
    if style.termination_on_each_pass {
        flags |= J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS;
    }
    if style.vertically_causal_context {
        flags |= J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT;
    }
    if style.segmentation_symbols {
        flags |= J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS;
    }
    if style.selective_arithmetic_coding_bypass {
        flags |= J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS;
    }
    flags
}

#[cfg(target_os = "macos")]
fn decode_classic_status_error(status: J2kClassicStatus) -> Error {
    let kind = match status.code {
        J2K_CLASSIC_STATUS_FAIL => "decode failure",
        J2K_CLASSIC_STATUS_UNSUPPORTED => "unsupported classic kernel input",
        _ => "unexpected classic kernel status",
    };
    Error::MetalKernel {
        message: format!("classic J2K Metal kernel {kind} (detail={})", status.detail),
    }
}

#[cfg(target_os = "macos")]
fn decode_idwt_status_error(status: J2kIdwtStatus) -> Error {
    let kind = match status.code {
        J2K_IDWT_STATUS_FAIL => "decode failure",
        _ => "unexpected IDWT kernel status",
    };
    Error::MetalKernel {
        message: format!("J2K Metal IDWT kernel {kind} (detail={})", status.detail),
    }
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn decode_mct_status_error(status: J2kMctStatus) -> Error {
    let kind = match status.code {
        J2K_MCT_STATUS_FAIL => "decode failure",
        _ => "unexpected inverse MCT kernel status",
    };
    Error::MetalKernel {
        message: format!(
            "J2K Metal inverse MCT kernel {kind} (detail={})",
            status.detail
        ),
    }
}

fn wrap_f32_output_buffer(device: &Device, output: &mut [f32]) -> Buffer {
    if output.is_empty() {
        device.new_buffer(
            size_of::<f32>() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    } else {
        device.new_buffer_with_bytes_no_copy(
            output.as_mut_ptr().cast(),
            size_of_val(output) as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }
}

#[cfg(target_os = "macos")]
fn borrow_slice_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    if data.is_empty() {
        device.new_buffer(1, MTLResourceOptions::StorageModeShared)
    } else {
        device.new_buffer_with_bytes_no_copy(
            data.as_ptr().cast(),
            size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }
}

#[cfg(target_os = "macos")]
fn borrow_mut_slice_buffer<T>(device: &Device, data: &mut [T]) -> Buffer {
    if data.is_empty() {
        device.new_buffer(1, MTLResourceOptions::StorageModeShared)
    } else {
        device.new_buffer_with_bytes_no_copy(
            data.as_mut_ptr().cast(),
            size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }
}

#[cfg(target_os = "macos")]
fn copied_slice_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    if data.is_empty() {
        device.new_buffer(1, MTLResourceOptions::StorageModeShared)
    } else {
        device.new_buffer_with_data(
            data.as_ptr().cast(),
            size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
}

#[cfg(target_os = "macos")]
fn classic_coefficients_scratch_bytes(job_count: usize) -> Result<usize, Error> {
    job_count
        .max(1)
        .checked_mul(J2K_CLASSIC_MAX_COEFF_COUNT)
        .and_then(|count| count.checked_mul(size_of::<u32>()))
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K coefficient scratch size overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
fn take_classic_coefficients_scratch_buffer(
    runtime: &MetalRuntime,
    job_count: usize,
) -> Result<DirectScratchBuffer, Error> {
    let bytes = classic_coefficients_scratch_bytes(job_count)?;
    Ok(DirectScratchBuffer {
        bytes,
        buffer: runtime.take_private_buffer(bytes),
    })
}

#[cfg(target_os = "macos")]
fn classic_states_scratch_bytes(job_count: usize) -> Result<usize, Error> {
    job_count
        .max(1)
        .checked_mul(J2K_CLASSIC_MAX_COEFF_COUNT)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K MetalDirect states scratch overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
fn take_classic_states_scratch_buffer(
    runtime: &MetalRuntime,
    job_count: usize,
) -> Result<DirectScratchBuffer, Error> {
    let bytes = classic_states_scratch_bytes(job_count)?;
    Ok(DirectScratchBuffer {
        bytes,
        buffer: runtime.take_private_buffer(bytes),
    })
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
pub(crate) fn encode_forward_dwt53(
    samples: &[f32],
    width: u32,
    height: u32,
    num_levels: u8,
) -> Result<J2kForwardDwt53Output, Error> {
    if width == 0 || height == 0 {
        return Err(Error::MetalKernel {
            message: "J2K Metal forward DWT dimensions must be non-zero".to_string(),
        });
    }
    let expected_len = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal forward DWT dimensions overflow".to_string(),
        })?;
    if samples.len() != expected_len {
        return Err(Error::MetalKernel {
            message: "J2K Metal forward DWT sample length mismatch".to_string(),
        });
    }

    with_runtime(|runtime| {
        let bytes = size_of_val(samples);
        let buffer_a = copied_slice_buffer(&runtime.device, samples);
        let buffer_b = runtime
            .device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
        let command_buffer = runtime.queue.new_command_buffer();

        let mut current_width = width;
        let mut current_height = height;
        let mut shapes = Vec::new();
        let mut levels_run = 0u8;
        let mut active_is_a = true;

        while levels_run < num_levels && (current_width >= 2 || current_height >= 2) {
            let low_width = current_width.div_ceil(2);
            let low_height = current_height.div_ceil(2);
            let params = J2kForwardDwt53Params {
                full_width: width,
                current_width,
                current_height,
                low_width,
                low_height,
            };

            if current_height >= 2 {
                let (input, output) =
                    active_forward_dwt53_buffers(&buffer_a, &buffer_b, active_is_a);
                dispatch_forward_dwt53_pass(
                    &runtime.fdwt53_vertical,
                    command_buffer,
                    input,
                    output,
                    params,
                );
                active_is_a = !active_is_a;
            }
            if current_width >= 2 {
                let (input, output) =
                    active_forward_dwt53_buffers(&buffer_a, &buffer_b, active_is_a);
                dispatch_forward_dwt53_pass(
                    &runtime.fdwt53_horizontal,
                    command_buffer,
                    input,
                    output,
                    params,
                );
                active_is_a = !active_is_a;
            }

            shapes.push(J2kForwardDwt53Level {
                hl: Vec::new(),
                lh: Vec::new(),
                hh: Vec::new(),
                width: current_width,
                height: current_height,
                low_width,
                low_height,
                high_width: current_width / 2,
                high_height: current_height / 2,
            });
            current_width = low_width;
            current_height = low_height;
            levels_run = levels_run.saturating_add(1);
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let active_buffer = if active_is_a { &buffer_a } else { &buffer_b };
        let transformed = unsafe {
            core::slice::from_raw_parts(active_buffer.contents().cast::<f32>(), samples.len())
        };
        let output = extract_forward_dwt53_output(
            transformed,
            width,
            current_width,
            current_height,
            shapes,
        )?;
        Ok(output)
    })
}

#[cfg(target_os = "macos")]
fn active_forward_dwt53_buffers<'a>(
    buffer_a: &'a Buffer,
    buffer_b: &'a Buffer,
    active_is_a: bool,
) -> (&'a Buffer, &'a Buffer) {
    if active_is_a {
        (buffer_a, buffer_b)
    } else {
        (buffer_b, buffer_a)
    }
}

#[cfg(target_os = "macos")]
fn dispatch_forward_dwt53_pass(
    pipeline: &ComputePipelineState,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    output: &Buffer,
    params: J2kForwardDwt53Params,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kForwardDwt53Params>() as u64,
        (&raw const params).cast(),
    );
    let width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(width);
    let height = (max_threads / width).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.current_width),
            height: u64::from(params.current_height),
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

#[cfg(target_os = "macos")]
fn extract_forward_dwt53_output(
    transformed: &[f32],
    full_width: u32,
    ll_width: u32,
    ll_height: u32,
    mut shapes: Vec<J2kForwardDwt53Level>,
) -> Result<J2kForwardDwt53Output, Error> {
    let full_width_usize = full_width as usize;
    let mut ll = Vec::with_capacity((ll_width as usize) * (ll_height as usize));
    for y in 0..ll_height as usize {
        let row_start = y
            .checked_mul(full_width_usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal forward DWT LL row offset overflow".to_string(),
            })?;
        ll.extend_from_slice(&transformed[row_start..row_start + ll_width as usize]);
    }

    for shape in &mut shapes {
        shape.hl = extract_subband(
            transformed,
            full_width_usize,
            shape.low_width,
            0,
            shape.high_width,
            shape.low_height,
        )?;
        shape.lh = extract_subband(
            transformed,
            full_width_usize,
            0,
            shape.low_height,
            shape.low_width,
            shape.high_height,
        )?;
        shape.hh = extract_subband(
            transformed,
            full_width_usize,
            shape.low_width,
            shape.low_height,
            shape.high_width,
            shape.high_height,
        )?;
    }
    shapes.reverse();

    Ok(J2kForwardDwt53Output {
        ll,
        ll_width,
        ll_height,
        levels: shapes,
    })
}

#[cfg(target_os = "macos")]
fn extract_subband(
    transformed: &[f32],
    full_width: usize,
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
) -> Result<Vec<f32>, Error> {
    let mut out = Vec::with_capacity((width as usize) * (height as usize));
    for y in 0..height as usize {
        let row_start = (y0 as usize)
            .checked_add(y)
            .and_then(|row| row.checked_mul(full_width))
            .and_then(|row| row.checked_add(x0 as usize))
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal forward DWT subband offset overflow".to_string(),
            })?;
        out.extend_from_slice(&transformed[row_start..row_start + width as usize]);
    }
    Ok(out)
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
pub(crate) fn encode_forward_rct(
    plane0: &mut [f32],
    plane1: &mut [f32],
    plane2: &mut [f32],
) -> Result<(), Error> {
    with_runtime(|runtime| {
        let len = plane0.len();
        if len == 0 {
            return Ok(());
        }
        if plane1.len() != len || plane2.len() != len {
            return Err(Error::MetalKernel {
                message: "J2K Metal forward RCT plane lengths must match".to_string(),
            });
        }

        let params = J2kForwardRctParams {
            len: u32::try_from(len).map_err(|_| Error::MetalKernel {
                message: "J2K Metal forward RCT plane length exceeds u32".to_string(),
            })?,
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
        };
        let plane0_buffer = borrow_mut_slice_buffer(&runtime.device, plane0);
        let plane1_buffer = borrow_mut_slice_buffer(&runtime.device, plane1);
        let plane2_buffer = borrow_mut_slice_buffer(&runtime.device, plane2);
        let status = J2kMctStatus::default();
        let status_buffer = runtime.device.new_buffer_with_data(
            (&raw const status).cast(),
            size_of::<J2kMctStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.forward_rct);
        encoder.set_buffer(0, Some(&plane0_buffer), 0);
        encoder.set_buffer(1, Some(&plane1_buffer), 0);
        encoder.set_buffer(2, Some(&plane2_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kForwardRctParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(4, Some(&status_buffer), 0);
        let width = runtime
            .forward_rct
            .thread_execution_width()
            .max(1)
            .min(len as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: len as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = unsafe { status_buffer.contents().cast::<J2kMctStatus>().read() };
        if status.code != J2K_MCT_STATUS_OK {
            return Err(decode_mct_status_error(status));
        }

        Ok(())
    })
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
pub(crate) fn decode_inverse_mct(job: J2kInverseMctJob<'_>) -> Result<Vec<Buffer>, Error> {
    let J2kInverseMctJob {
        transform,
        plane0,
        plane1,
        plane2,
        addend0,
        addend1,
        addend2,
    } = job;
    with_runtime(|runtime| {
        let len = plane0.len();
        if len == 0 {
            return Ok(Vec::new());
        }
        if plane1.len() != len || plane2.len() != len {
            return Err(Error::MetalKernel {
                message: "J2K Metal inverse MCT plane lengths must match".to_string(),
            });
        }

        let transform = match transform {
            J2kWaveletTransform::Reversible53 => 0,
            J2kWaveletTransform::Irreversible97 => 1,
        };
        let params = J2kInverseMctParams {
            len: u32::try_from(len).map_err(|_| Error::MetalKernel {
                message: "J2K Metal inverse MCT plane length exceeds u32".to_string(),
            })?,
            transform,
            addend0,
            addend1,
            addend2,
        };
        let plane0_buffer = copied_slice_buffer(&runtime.device, plane0);
        let plane1_buffer = copied_slice_buffer(&runtime.device, plane1);
        let plane2_buffer = copied_slice_buffer(&runtime.device, plane2);
        let status = J2kMctStatus::default();
        let status_buffer = runtime.device.new_buffer_with_data(
            (&raw const status).cast(),
            size_of::<J2kMctStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.inverse_mct);
        encoder.set_buffer(0, Some(&plane0_buffer), 0);
        encoder.set_buffer(1, Some(&plane1_buffer), 0);
        encoder.set_buffer(2, Some(&plane2_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kInverseMctParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(4, Some(&status_buffer), 0);
        let width = runtime
            .inverse_mct
            .thread_execution_width()
            .max(1)
            .min(len as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: len as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = unsafe { status_buffer.contents().cast::<J2kMctStatus>().read() };
        if status.code != J2K_MCT_STATUS_OK {
            return Err(decode_mct_status_error(status));
        }

        let plane0_host =
            unsafe { core::slice::from_raw_parts(plane0_buffer.contents().cast::<f32>(), len) };
        let plane1_host =
            unsafe { core::slice::from_raw_parts(plane1_buffer.contents().cast::<f32>(), len) };
        let plane2_host =
            unsafe { core::slice::from_raw_parts(plane2_buffer.contents().cast::<f32>(), len) };
        for (dst, sample) in plane0.iter_mut().zip(plane0_host.iter().copied()) {
            *dst = sample - addend0;
        }
        for (dst, sample) in plane1.iter_mut().zip(plane1_host.iter().copied()) {
            *dst = sample - addend1;
        }
        for (dst, sample) in plane2.iter_mut().zip(plane2_host.iter().copied()) {
            *dst = sample - addend2;
        }
        Ok(vec![plane0_buffer, plane1_buffer, plane2_buffer])
    })
}

#[cfg(target_os = "macos")]
fn dispatch_inverse_mct_buffers_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: [&Buffer; 3],
    len: usize,
    transform: J2kWaveletTransform,
    addends: [f32; 3],
) -> Result<DirectStatusCheck, Error> {
    if len == 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect color MCT cannot run on an empty plane".to_string(),
        });
    }

    let transform = match transform {
        J2kWaveletTransform::Reversible53 => 0,
        J2kWaveletTransform::Irreversible97 => 1,
    };
    let params = J2kInverseMctParams {
        len: u32::try_from(len).map_err(|_| Error::MetalKernel {
            message: "J2K MetalDirect color MCT plane length exceeds u32".to_string(),
        })?,
        transform,
        addend0: addends[0],
        addend1: addends[1],
        addend2: addends[2],
    };
    let status = J2kMctStatus::default();
    let status_buffer = runtime.device.new_buffer_with_data(
        (&raw const status).cast(),
        size_of::<J2kMctStatus>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.inverse_mct);
    encoder.set_buffer(0, Some(planes[0]), 0);
    encoder.set_buffer(1, Some(planes[1]), 0);
    encoder.set_buffer(2, Some(planes[2]), 0);
    encoder.set_bytes(
        3,
        size_of::<J2kInverseMctParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(4, Some(&status_buffer), 0);
    let width = runtime
        .inverse_mct
        .thread_execution_width()
        .max(1)
        .min(len as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    Ok(DirectStatusCheck::Mct(status_buffer))
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_store_component_and_capture(
    job: J2kStoreComponentJob<'_>,
) -> Result<Buffer, Error> {
    let J2kStoreComponentJob {
        input,
        input_width,
        source_x,
        source_y,
        copy_width,
        copy_height,
        output,
        output_width,
        output_x,
        output_y,
        addend,
    } = job;
    with_runtime(|runtime| {
        if copy_width == 0 || copy_height == 0 {
            return Ok(wrap_f32_output_buffer(&runtime.device, output));
        }

        let required_input_height =
            source_y
                .checked_add(copy_height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal store source height overflow".to_string(),
                })?;
        let required_output_height =
            output_y
                .checked_add(copy_height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal store destination height overflow".to_string(),
                })?;
        if source_x
            .checked_add(copy_width)
            .is_none_or(|end| end > input_width)
            || output_x
                .checked_add(copy_width)
                .is_none_or(|end| end > output_width)
        {
            return Err(Error::MetalKernel {
                message: "J2K Metal store copy rectangle exceeds row bounds".to_string(),
            });
        }
        if input.len()
            < input_width as usize
                * usize::try_from(required_input_height).map_err(|_| Error::MetalKernel {
                    message: "J2K Metal store source height exceeds usize".to_string(),
                })?
            || output.len()
                < output_width as usize
                    * usize::try_from(required_output_height).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal store destination height exceeds usize".to_string(),
                    })?
        {
            return Err(Error::MetalKernel {
                message: "J2K Metal store buffers are smaller than required".to_string(),
            });
        }

        let params = J2kStoreParams {
            input_width,
            source_x,
            source_y,
            copy_width,
            copy_height,
            output_width,
            output_x,
            output_y,
            addend,
        };
        let input_buffer = borrow_slice_buffer(&runtime.device, input);
        let output_buffer = wrap_f32_output_buffer(&runtime.device, output);
        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.store_component);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            size_of::<J2kStoreParams>() as u64,
            (&raw const params).cast(),
        );
        let width = runtime.store_component.thread_execution_width().max(1);
        let max_threads = runtime
            .store_component
            .max_total_threads_per_threadgroup()
            .max(width);
        let height = (max_threads / width).max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(copy_width),
                height: u64::from(copy_height),
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
        Ok(output_buffer)
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::unnecessary_wraps)]
#[allow(dead_code)]
fn dispatch_store_component_buffer(
    runtime: &MetalRuntime,
    input: &Buffer,
    output: &Buffer,
    params: J2kStoreParams,
) -> Result<(), Error> {
    let command_buffer = runtime.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.store_component);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kStoreParams>() as u64,
        (&raw const params).cast(),
    );
    let width = runtime.store_component.thread_execution_width().max(1);
    let max_threads = runtime
        .store_component
        .max_total_threads_per_threadgroup()
        .max(width);
    let height = (max_threads / width).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.copy_width),
            height: u64::from(params.copy_height),
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
    Ok(())
}

#[cfg(target_os = "macos")]
fn dispatch_store_component_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    output: &Buffer,
    params: J2kStoreParams,
) {
    dispatch_store_component_buffer_in_command_buffer_with_offsets(
        runtime,
        command_buffer,
        input,
        0,
        output,
        0,
        params,
    );
}

#[cfg(target_os = "macos")]
fn dispatch_store_component_buffer_in_command_buffer_with_offsets(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    input_offset_bytes: usize,
    output: &Buffer,
    output_offset_bytes: usize,
    params: J2kStoreParams,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    dispatch_store_component_buffer_in_encoder_with_offsets(
        runtime,
        encoder,
        input,
        input_offset_bytes,
        output,
        output_offset_bytes,
        params,
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
fn dispatch_store_component_buffer_in_encoder_with_offsets(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    input: &Buffer,
    input_offset_bytes: usize,
    output: &Buffer,
    output_offset_bytes: usize,
    params: J2kStoreParams,
) {
    encoder.set_compute_pipeline_state(&runtime.store_component);
    encoder.set_buffer(0, Some(input), input_offset_bytes as u64);
    encoder.set_buffer(1, Some(output), output_offset_bytes as u64);
    encoder.set_bytes(
        2,
        size_of::<J2kStoreParams>() as u64,
        (&raw const params).cast(),
    );
    let width = runtime.store_component.thread_execution_width().max(1);
    let max_threads = runtime
        .store_component
        .max_total_threads_per_threadgroup()
        .max(width);
    let height = (max_threads / width).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.copy_width),
            height: u64::from(params.copy_height),
            depth: 1,
        },
        MTLSize {
            width,
            height,
            depth: 1,
        },
    );
}

fn dispatch_store_component_repeated_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    output: &Buffer,
    params: J2kRepeatedStoreParams,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.store_component_repeated);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kRepeatedStoreParams>() as u64,
        (&raw const params).cast(),
    );
    let width = runtime
        .store_component_repeated
        .thread_execution_width()
        .max(1);
    let max_threads = runtime
        .store_component_repeated
        .max_total_threads_per_threadgroup()
        .max(width);
    let height = (max_threads / width).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.copy_width),
            height: u64::from(params.copy_height),
            depth: u64::from(params.batch_count),
        },
        MTLSize {
            width,
            height,
            depth: 1,
        },
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
fn repeated_gray_store_is_contiguous_full_surface(params: J2kRepeatedGrayStoreParams) -> bool {
    params.source_x == 0
        && params.source_y == 0
        && params.output_x == 0
        && params.output_y == 0
        && params.copy_width == params.input_width
        && params.copy_height == params.input_height
        && params.copy_width == params.output_width
        && params.copy_height == params.output_height
}

#[cfg(target_os = "macos")]
fn encode_repeated_gray_store_to_surfaces_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    params: J2kRepeatedGrayStoreParams,
    dims: (u32, u32),
    fmt: PixelFormat,
    count: usize,
) -> Result<Vec<Surface>, Error> {
    let bytes_per_pixel = fmt.bytes_per_pixel();
    let pitch_bytes = dims.0 as usize * bytes_per_pixel;
    let surface_bytes =
        pitch_bytes
            .checked_mul(dims.1 as usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal repeated grayscale fused store size overflow".to_string(),
            })?;
    let total_bytes = surface_bytes
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal repeated grayscale fused store total size overflow".to_string(),
        })?;
    let out_buffer = runtime
        .device
        .new_buffer(total_bytes as u64, MTLResourceOptions::StorageModeShared);
    let contiguous_full_surface = repeated_gray_store_is_contiguous_full_surface(params);
    let pipeline = match (fmt, contiguous_full_surface) {
        (PixelFormat::Gray8, true) => &runtime.store_component_repeated_gray_u8_contiguous,
        (PixelFormat::Gray8, false) => &runtime.store_component_repeated_gray_u8,
        (PixelFormat::Gray16, true) => &runtime.store_component_repeated_gray_u16_contiguous,
        (PixelFormat::Gray16, false) => &runtime.store_component_repeated_gray_u16,
        _ => {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K Metal repeated grayscale fused store does not support {fmt:?}"
                ),
            })
        }
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(&out_buffer), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kRepeatedGrayStoreParams>() as u64,
        (&raw const params).cast(),
    );
    let width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(width);
    if contiguous_full_surface {
        let total_samples = u64::from(params.input_width)
            * u64::from(params.input_height)
            * u64::from(params.batch_count);
        encoder.dispatch_threads(
            MTLSize {
                width: total_samples,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: max_threads,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let height = (max_threads / width).max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(params.copy_width),
                height: u64::from(params.copy_height),
                depth: u64::from(params.batch_count),
            },
            MTLSize {
                width,
                height,
                depth: 1,
            },
        );
    }
    encoder.end_encoding();

    let mut surfaces = Vec::with_capacity(count);
    for instance_idx in 0..count {
        surfaces.push(Surface::from_metal_buffer_with_offset(
            out_buffer.clone(),
            dims,
            fmt,
            instance_idx * surface_bytes,
        ));
    }
    Ok(surfaces)
}

#[cfg(target_os = "macos")]
fn encode_gray_store_to_surface_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    input_offset_bytes: usize,
    params: J2kGrayStoreParams,
    dims: (u32, u32),
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let encoder = command_buffer.new_compute_command_encoder();
    let result = encode_gray_store_to_surface_in_encoder(
        runtime,
        encoder,
        input,
        input_offset_bytes,
        params,
        dims,
        fmt,
    );
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
fn encode_gray_store_to_surface_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    input: &Buffer,
    input_offset_bytes: usize,
    params: J2kGrayStoreParams,
    dims: (u32, u32),
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let pitch_bytes = dims.0 as usize * fmt.bytes_per_pixel();
    let out_buffer = runtime.device.new_buffer(
        (pitch_bytes * dims.1 as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let pipeline = match fmt {
        PixelFormat::Gray8 => &runtime.store_component_gray_u8,
        PixelFormat::Gray16 => &runtime.store_component_gray_u16,
        _ => {
            return Err(Error::MetalKernel {
                message: format!("J2K Metal grayscale fused store does not support {fmt:?}"),
            })
        }
    };

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), input_offset_bytes as u64);
    encoder.set_buffer(1, Some(&out_buffer), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kGrayStoreParams>() as u64,
        (&raw const params).cast(),
    );
    let width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(width);
    let height = (max_threads / width).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.copy_width),
            height: u64::from(params.copy_height),
            depth: 1,
        },
        MTLSize {
            width,
            height,
            depth: 1,
        },
    );

    Ok(Surface::from_metal_buffer(out_buffer, dims, fmt))
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_reversible53_single_decomposition_idwt(
    job: J2kSingleDecompositionIdwtJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    with_runtime(|runtime| {
        let required_len = job.rect.width() as usize * job.rect.height() as usize;
        if output.len() < required_len {
            return Err(Error::MetalKernel {
                message: "J2K Metal IDWT output slice is too small".to_string(),
            });
        }

        let params = J2kIdwtSingleDecompositionParams {
            x0: job.rect.x0,
            y0: job.rect.y0,
            width: job.rect.width(),
            height: job.rect.height(),
            ll_width: job.ll.rect.width(),
            ll_height: job.ll.rect.height(),
            hl_width: job.hl.rect.width(),
            hl_height: job.hl.rect.height(),
            lh_width: job.lh.rect.width(),
            lh_height: job.lh.rect.height(),
            hh_width: job.hh.rect.width(),
            hh_height: job.hh.rect.height(),
        };

        let ll = borrow_slice_buffer(&runtime.device, job.ll.coefficients);
        let hl = borrow_slice_buffer(&runtime.device, job.hl.coefficients);
        let lh = borrow_slice_buffer(&runtime.device, job.lh.coefficients);
        let hh = borrow_slice_buffer(&runtime.device, job.hh.coefficients);
        let decoded = wrap_f32_output_buffer(&runtime.device, output);

        let command_buffer = runtime.queue.new_command_buffer();

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.idwt_interleave);
        encoder.set_buffer(0, Some(&ll), 0);
        encoder.set_buffer(1, Some(&hl), 0);
        encoder.set_buffer(2, Some(&lh), 0);
        encoder.set_buffer(3, Some(&hh), 0);
        encoder.set_buffer(4, Some(&decoded), 0);
        encoder.set_bytes(
            5,
            size_of::<J2kIdwtSingleDecompositionParams>() as u64,
            (&raw const params).cast(),
        );
        let interleave_width = runtime.idwt_interleave.thread_execution_width().max(1);
        let interleave_height = (runtime
            .idwt_interleave
            .max_total_threads_per_threadgroup()
            .max(interleave_width)
            / interleave_width)
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(params.width),
                height: u64::from(params.height),
                depth: 1,
            },
            MTLSize {
                width: interleave_width,
                height: interleave_height,
                depth: 1,
            },
        );
        encoder.end_encoding();

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_horizontal);
        encoder.set_buffer(0, Some(&decoded), 0);
        encoder.set_bytes(
            1,
            size_of::<J2kIdwtSingleDecompositionParams>() as u64,
            (&raw const params).cast(),
        );
        let horizontal_width = runtime
            .idwt_reversible53_horizontal
            .thread_execution_width()
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(params.height),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: horizontal_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_vertical);
        encoder.set_buffer(0, Some(&decoded), 0);
        encoder.set_bytes(
            1,
            size_of::<J2kIdwtSingleDecompositionParams>() as u64,
            (&raw const params).cast(),
        );
        let vertical_width = runtime
            .idwt_reversible53_vertical
            .thread_execution_width()
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(params.width),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: vertical_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::unnecessary_wraps)]
#[allow(dead_code)]
fn dispatch_reversible53_single_decomposition_buffers(
    runtime: &MetalRuntime,
    ll: &Buffer,
    hl: &Buffer,
    lh: &Buffer,
    hh: &Buffer,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
) -> Result<(), Error> {
    let command_buffer = runtime.queue.new_command_buffer();

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.idwt_interleave);
    encoder.set_buffer(0, Some(ll), 0);
    encoder.set_buffer(1, Some(hl), 0);
    encoder.set_buffer(2, Some(lh), 0);
    encoder.set_buffer(3, Some(hh), 0);
    encoder.set_buffer(4, Some(decoded), 0);
    encoder.set_bytes(
        5,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let interleave_width = runtime.idwt_interleave.thread_execution_width().max(1);
    let interleave_height = (runtime
        .idwt_interleave
        .max_total_threads_per_threadgroup()
        .max(interleave_width)
        / interleave_width)
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.width),
            height: u64::from(params.height),
            depth: 1,
        },
        MTLSize {
            width: interleave_width,
            height: interleave_height,
            depth: 1,
        },
    );
    encoder.end_encoding();

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_horizontal);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_bytes(
        1,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let horizontal_width = runtime
        .idwt_reversible53_horizontal
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.height),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: horizontal_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_vertical);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_bytes(
        1,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let vertical_width = runtime
        .idwt_reversible53_vertical
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.width),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: vertical_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_reversible53_single_decomposition_buffers_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    ll: &Buffer,
    hl: &Buffer,
    lh: &Buffer,
    hh: &Buffer,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
) {
    dispatch_reversible53_single_decomposition_buffers_in_command_buffer_with_offsets(
        runtime,
        command_buffer,
        ll,
        0,
        hl,
        0,
        lh,
        0,
        hh,
        0,
        params,
        decoded,
        0,
    );
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_reversible53_single_decomposition_buffers_in_command_buffer_with_offsets(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    ll: &Buffer,
    ll_offset: usize,
    hl: &Buffer,
    hl_offset: usize,
    lh: &Buffer,
    lh_offset: usize,
    hh: &Buffer,
    hh_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
    decoded_offset: usize,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
        runtime,
        encoder,
        ll,
        ll_offset,
        hl,
        hl_offset,
        lh,
        lh_offset,
        hh,
        hh_offset,
        params,
        decoded,
        decoded_offset,
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    ll: &Buffer,
    ll_offset: usize,
    hl: &Buffer,
    hl_offset: usize,
    lh: &Buffer,
    lh_offset: usize,
    hh: &Buffer,
    hh_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
    decoded_offset: usize,
) {
    encoder.set_compute_pipeline_state(&runtime.idwt_interleave);
    encoder.set_buffer(0, Some(ll), ll_offset as u64);
    encoder.set_buffer(1, Some(hl), hl_offset as u64);
    encoder.set_buffer(2, Some(lh), lh_offset as u64);
    encoder.set_buffer(3, Some(hh), hh_offset as u64);
    encoder.set_buffer(4, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        5,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let interleave_width = runtime.idwt_interleave.thread_execution_width().max(1);
    let interleave_height = (runtime
        .idwt_interleave
        .max_total_threads_per_threadgroup()
        .max(interleave_width)
        / interleave_width)
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.width),
            height: u64::from(params.height),
            depth: 1,
        },
        MTLSize {
            width: interleave_width,
            height: interleave_height,
            depth: 1,
        },
    );

    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_horizontal);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        1,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let horizontal_width = runtime
        .idwt_reversible53_horizontal
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.height),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: horizontal_width,
            height: 1,
            depth: 1,
        },
    );

    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_vertical);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        1,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let vertical_width = runtime
        .idwt_reversible53_vertical
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.width),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: vertical_width,
            height: 1,
            depth: 1,
        },
    );
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn dispatch_reversible53_repeated_buffers_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    ll: &Buffer,
    hl: &Buffer,
    lh: &Buffer,
    hh: &Buffer,
    params: J2kRepeatedIdwtSingleDecompositionParams,
    decoded: &Buffer,
) {
    dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets(
        runtime,
        command_buffer,
        ll,
        0,
        hl,
        0,
        lh,
        0,
        hh,
        0,
        params,
        decoded,
    );
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    ll: &Buffer,
    ll_offset: usize,
    hl: &Buffer,
    hl_offset: usize,
    lh: &Buffer,
    lh_offset: usize,
    hh: &Buffer,
    hh_offset: usize,
    params: J2kRepeatedIdwtSingleDecompositionParams,
    decoded: &Buffer,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.idwt_interleave_batched);
    encoder.set_buffer(0, Some(ll), ll_offset as u64);
    encoder.set_buffer(1, Some(hl), hl_offset as u64);
    encoder.set_buffer(2, Some(lh), lh_offset as u64);
    encoder.set_buffer(3, Some(hh), hh_offset as u64);
    encoder.set_buffer(4, Some(decoded), 0);
    encoder.set_bytes(
        5,
        size_of::<J2kRepeatedIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let interleave_width = runtime
        .idwt_interleave_batched
        .thread_execution_width()
        .max(1);
    let interleave_height = (runtime
        .idwt_interleave_batched
        .max_total_threads_per_threadgroup()
        .max(interleave_width)
        / interleave_width)
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.width),
            height: u64::from(params.height),
            depth: u64::from(params.batch_count),
        },
        MTLSize {
            width: interleave_width,
            height: interleave_height,
            depth: 1,
        },
    );
    encoder.end_encoding();

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_horizontal_batched);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_bytes(
        1,
        size_of::<J2kRepeatedIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let horizontal_width = runtime
        .idwt_reversible53_horizontal_batched
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.height),
            height: u64::from(params.batch_count),
            depth: 1,
        },
        MTLSize {
            width: horizontal_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_vertical_batched);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_bytes(
        1,
        size_of::<J2kRepeatedIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let vertical_width = runtime
        .idwt_reversible53_vertical_batched
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.width),
            height: u64::from(params.batch_count),
            depth: 1,
        },
        MTLSize {
            width: vertical_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_irreversible97_single_decomposition_idwt(
    job: J2kSingleDecompositionIdwtJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    with_runtime(|runtime| {
        let required_len = job.rect.width() as usize * job.rect.height() as usize;
        if output.len() < required_len {
            return Err(Error::MetalKernel {
                message: "J2K Metal IDWT output slice is too small".to_string(),
            });
        }

        let params = J2kIdwtSingleDecompositionParams {
            x0: job.rect.x0,
            y0: job.rect.y0,
            width: job.rect.width(),
            height: job.rect.height(),
            ll_width: job.ll.rect.width(),
            ll_height: job.ll.rect.height(),
            hl_width: job.hl.rect.width(),
            hl_height: job.hl.rect.height(),
            lh_width: job.lh.rect.width(),
            lh_height: job.lh.rect.height(),
            hh_width: job.hh.rect.width(),
            hh_height: job.hh.rect.height(),
        };

        let ll = borrow_slice_buffer(&runtime.device, job.ll.coefficients);
        let hl = borrow_slice_buffer(&runtime.device, job.hl.coefficients);
        let lh = borrow_slice_buffer(&runtime.device, job.lh.coefficients);
        let hh = borrow_slice_buffer(&runtime.device, job.hh.coefficients);
        let decoded = wrap_f32_output_buffer(&runtime.device, output);
        let status_buffer = runtime.device.new_buffer(
            size_of::<J2kIdwtStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.idwt_irreversible97_single_decomposition);
        encoder.set_buffer(0, Some(&ll), 0);
        encoder.set_buffer(1, Some(&hl), 0);
        encoder.set_buffer(2, Some(&lh), 0);
        encoder.set_buffer(3, Some(&hh), 0);
        encoder.set_buffer(4, Some(&decoded), 0);
        encoder.set_bytes(
            5,
            size_of::<J2kIdwtSingleDecompositionParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(6, Some(&status_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = unsafe { status_buffer.contents().cast::<J2kIdwtStatus>().read() };
        if status.code != J2K_IDWT_STATUS_OK {
            return Err(decode_idwt_status_error(status));
        }
        Ok(())
    })
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn dispatch_irreversible97_single_decomposition_buffers(
    runtime: &MetalRuntime,
    ll: &Buffer,
    hl: &Buffer,
    lh: &Buffer,
    hh: &Buffer,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
) -> Result<(), Error> {
    let status_buffer = runtime.device.new_buffer(
        size_of::<J2kIdwtStatus>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = runtime.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.idwt_irreversible97_single_decomposition);
    encoder.set_buffer(0, Some(ll), 0);
    encoder.set_buffer(1, Some(hl), 0);
    encoder.set_buffer(2, Some(lh), 0);
    encoder.set_buffer(3, Some(hh), 0);
    encoder.set_buffer(4, Some(decoded), 0);
    encoder.set_bytes(
        5,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(6, Some(&status_buffer), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let status = unsafe { status_buffer.contents().cast::<J2kIdwtStatus>().read() };
    if status.code != J2K_IDWT_STATUS_OK {
        return Err(decode_idwt_status_error(status));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_irreversible97_single_decomposition_buffers_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    ll: &Buffer,
    hl: &Buffer,
    lh: &Buffer,
    hh: &Buffer,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
) -> DirectStatusCheck {
    dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
        runtime,
        command_buffer,
        ll,
        0,
        hl,
        0,
        lh,
        0,
        hh,
        0,
        params,
        decoded,
        0,
    )
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    ll: &Buffer,
    ll_offset: usize,
    hl: &Buffer,
    hl_offset: usize,
    lh: &Buffer,
    lh_offset: usize,
    hh: &Buffer,
    hh_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
    decoded_offset: usize,
) -> DirectStatusCheck {
    let status_buffer = runtime.device.new_buffer(
        size_of::<J2kIdwtStatus>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let encoder = command_buffer.new_compute_command_encoder();
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_status(
        runtime,
        encoder,
        ll,
        ll_offset,
        hl,
        hl_offset,
        lh,
        lh_offset,
        hh,
        hh_offset,
        params,
        decoded,
        decoded_offset,
        &status_buffer,
    );
    encoder.end_encoding();

    DirectStatusCheck::Idwt(status_buffer)
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    ll: &Buffer,
    ll_offset: usize,
    hl: &Buffer,
    hl_offset: usize,
    lh: &Buffer,
    lh_offset: usize,
    hh: &Buffer,
    hh_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
    decoded_offset: usize,
) -> DirectStatusCheck {
    let status_buffer = runtime.device.new_buffer(
        size_of::<J2kIdwtStatus>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_status(
        runtime,
        encoder,
        ll,
        ll_offset,
        hl,
        hl_offset,
        lh,
        lh_offset,
        hh,
        hh_offset,
        params,
        decoded,
        decoded_offset,
        &status_buffer,
    );

    DirectStatusCheck::Idwt(status_buffer)
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_status(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    ll: &Buffer,
    ll_offset: usize,
    hl: &Buffer,
    hl_offset: usize,
    lh: &Buffer,
    lh_offset: usize,
    hh: &Buffer,
    hh_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
    decoded_offset: usize,
    status_buffer: &Buffer,
) {
    encoder.set_compute_pipeline_state(&runtime.idwt_irreversible97_single_decomposition);
    encoder.set_buffer(0, Some(ll), ll_offset as u64);
    encoder.set_buffer(1, Some(hl), hl_offset as u64);
    encoder.set_buffer(2, Some(lh), lh_offset as u64);
    encoder.set_buffer(3, Some(hh), hh_offset as u64);
    encoder.set_buffer(4, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        5,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(6, Some(status_buffer), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
}

#[cfg(target_os = "macos")]
fn classic_batch_uses_plain_fast_path(
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
) -> bool {
    jobs.iter().all(|job| {
        if job.style_flags != 0
            || job.width > J2K_CLASSIC_MAX_WIDTH
            || job.height > J2K_CLASSIC_MAX_HEIGHT
        {
            return false;
        }
        let start = job.segment_offset as usize;
        let Some(end) = start.checked_add(job.segment_count as usize) else {
            return false;
        };
        segments.get(start..end).is_some_and(|job_segments| {
            job_segments
                .iter()
                .all(|segment| segment.use_arithmetic != 0)
        })
    })
}

#[cfg(target_os = "macos")]
fn classic_repeated_uses_plain_fast_path(
    count: usize,
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
) -> bool {
    let _ = (count, jobs, segments);
    // Batch-16 WSI benches are faster with device-state cleanup plus the separate parallel store.
    false
}

#[cfg(target_os = "macos")]
fn classic_batch_is_plain_arithmetic(
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
) -> bool {
    jobs.iter().all(|job| {
        job.style_flags == 0
            && segments[job.segment_offset as usize
                ..job.segment_offset as usize + job.segment_count as usize]
                .iter()
                .all(|segment| segment.use_arithmetic != 0)
    })
}

#[cfg(target_os = "macos")]
fn dispatch_classic_cleanup_batched(
    runtime: &MetalRuntime,
    coded_data: &[u8],
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
    decoded: &Buffer,
) -> Result<(), Error> {
    let input = borrow_slice_buffer(&runtime.device, coded_data);
    let jobs_buffer = borrow_slice_buffer(&runtime.device, jobs);
    let segments_buffer = borrow_slice_buffer(&runtime.device, segments);
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, jobs.len())?;
    let use_plain_fast_path = classic_batch_uses_plain_fast_path(jobs, segments)
        && runtime
            .classic_cleanup_plain_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    let pipeline = if use_plain_fast_path {
        &runtime.classic_cleanup_plain_batched
    } else {
        &runtime.classic_cleanup_batched
    };
    let status_buffer = runtime.device.new_buffer(
        (jobs.len().max(1) * size_of::<J2kClassicStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = runtime.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(&jobs_buffer), 0);
    encoder.set_buffer(3, Some(&segments_buffer), 0);
    encoder.set_buffer(4, Some(&status_buffer), 0);
    encoder.set_buffer(5, Some(&coefficients_scratch.buffer), 0);
    if use_plain_fast_path {
        encoder.dispatch_thread_groups(
            MTLSize {
                width: jobs.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let width = pipeline
            .thread_execution_width()
            .max(1)
            .min(jobs.len() as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: jobs.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
    }
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let statuses = unsafe {
        core::slice::from_raw_parts(
            status_buffer.contents().cast::<J2kClassicStatus>(),
            jobs.len(),
        )
    };
    let status = statuses
        .iter()
        .copied()
        .find(|status| status.code != J2K_CLASSIC_STATUS_OK);
    runtime.recycle_private_buffer(coefficients_scratch.bytes, coefficients_scratch.buffer);
    if let Some(status) = status {
        return Err(decode_classic_status_error(status));
    }

    Ok(())
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_classic_cleanup_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    use_plain_fast_path: bool,
    segments: &Buffer,
    decoded: &Buffer,
    coefficients_scratch: &Buffer,
) -> (DirectStatusCheck, Option<Buffer>) {
    let status_buffer = runtime.device.new_buffer(
        (job_count.max(1) * size_of::<J2kClassicStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let encoder = command_buffer.new_compute_command_encoder();
    dispatch_classic_cleanup_batched_in_encoder_with_status(
        runtime,
        encoder,
        coded_data,
        jobs,
        job_count,
        use_plain_fast_path,
        segments,
        decoded,
        coefficients_scratch,
        &status_buffer,
    );
    encoder.end_encoding();

    (
        DirectStatusCheck::Classic {
            buffer: status_buffer,
            len: job_count,
        },
        None,
    )
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_classic_cleanup_batched_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    use_plain_fast_path: bool,
    segments: &Buffer,
    decoded: &Buffer,
    coefficients_scratch: &Buffer,
) -> (DirectStatusCheck, Option<Buffer>) {
    let status_buffer = runtime.device.new_buffer(
        (job_count.max(1) * size_of::<J2kClassicStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    dispatch_classic_cleanup_batched_in_encoder_with_status(
        runtime,
        encoder,
        coded_data,
        jobs,
        job_count,
        use_plain_fast_path,
        segments,
        decoded,
        coefficients_scratch,
        &status_buffer,
    );

    (
        DirectStatusCheck::Classic {
            buffer: status_buffer,
            len: job_count,
        },
        None,
    )
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_classic_cleanup_batched_in_encoder_with_status(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    use_plain_fast_path: bool,
    segments: &Buffer,
    decoded: &Buffer,
    coefficients_scratch: &Buffer,
    status_buffer: &Buffer,
) {
    let pipeline = if use_plain_fast_path {
        &runtime.classic_cleanup_plain_batched
    } else {
        &runtime.classic_cleanup_batched
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(segments), 0);
    encoder.set_buffer(4, Some(status_buffer), 0);
    encoder.set_buffer(5, Some(coefficients_scratch), 0);
    if use_plain_fast_path {
        encoder.dispatch_thread_groups(
            MTLSize {
                width: job_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let width = pipeline
            .thread_execution_width()
            .max(1)
            .min(job_count as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: job_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_classic_cleanup_repeated_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    total_job_count: usize,
    output_plane_len: usize,
    use_plain_fast_path: bool,
    segments: &Buffer,
    decoded: &Buffer,
    coefficients_scratch: &Buffer,
) -> DirectStatusCheck {
    let pipeline = if use_plain_fast_path {
        &runtime.classic_cleanup_plain_repeated_batched
    } else {
        &runtime.classic_cleanup_repeated_batched
    };
    let status_buffer = runtime.device.new_buffer(
        (total_job_count.max(1) * size_of::<J2kClassicStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let repeated = J2kClassicRepeatedBatchParams {
        job_count: u32::try_from(job_count).expect("classic repeated base job count fits in u32"),
        output_plane_len: u32::try_from(output_plane_len)
            .expect("classic repeated output plane len fits in u32"),
        batch_count: u32::try_from(total_job_count / job_count.max(1))
            .expect("classic repeated batch count fits in u32"),
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(segments), 0);
    encoder.set_buffer(4, Some(&status_buffer), 0);
    encoder.set_buffer(5, Some(coefficients_scratch), 0);
    encoder.set_bytes(
        6,
        size_of::<J2kClassicRepeatedBatchParams>() as u64,
        (&raw const repeated).cast(),
    );
    if use_plain_fast_path {
        encoder.dispatch_thread_groups(
            MTLSize {
                width: job_count as u64,
                height: u64::from(repeated.batch_count),
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let width = pipeline
            .thread_execution_width()
            .max(1)
            .min(job_count as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: job_count as u64,
                height: u64::from(repeated.batch_count),
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
    }
    encoder.end_encoding();

    DirectStatusCheck::Classic {
        buffer: status_buffer,
        len: total_job_count,
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_classic_cleanup_plain_dev_repeated_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    total_job_count: usize,
    output_plane_len: usize,
    segments: &Buffer,
    decoded: &Buffer,
    coefficients_scratch: &Buffer,
    states_scratch: &Buffer,
) -> DirectStatusCheck {
    let status_buffer = runtime.device.new_buffer(
        (total_job_count.max(1) * size_of::<J2kClassicStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let repeated = J2kClassicRepeatedBatchParams {
        job_count: u32::try_from(job_count).expect("classic repeated base job count fits in u32"),
        output_plane_len: u32::try_from(output_plane_len)
            .expect("classic repeated output plane len fits in u32"),
        batch_count: u32::try_from(total_job_count / job_count.max(1))
            .expect("classic repeated batch count fits in u32"),
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.classic_cleanup_plain_dev_repeated_batched);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(segments), 0);
    encoder.set_buffer(4, Some(&status_buffer), 0);
    encoder.set_buffer(5, Some(coefficients_scratch), 0);
    encoder.set_buffer(6, Some(states_scratch), 0);
    encoder.set_bytes(
        7,
        size_of::<J2kClassicRepeatedBatchParams>() as u64,
        (&raw const repeated).cast(),
    );
    let width = runtime
        .classic_cleanup_plain_dev_repeated_batched
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: job_count as u64,
            height: u64::from(repeated.batch_count),
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    DirectStatusCheck::Classic {
        buffer: status_buffer,
        len: total_job_count,
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_classic_store_repeated_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    jobs: &Buffer,
    job_count: usize,
    total_job_count: usize,
    output_plane_len: usize,
    decoded: &Buffer,
    coefficients_scratch: &Buffer,
) {
    let repeated = J2kClassicRepeatedBatchParams {
        job_count: u32::try_from(job_count).expect("classic repeated base job count fits in u32"),
        output_plane_len: u32::try_from(output_plane_len)
            .expect("classic repeated output plane len fits in u32"),
        batch_count: u32::try_from(total_job_count / job_count.max(1))
            .expect("classic repeated batch count fits in u32"),
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.classic_store_repeated_batched);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_buffer(1, Some(jobs), 0);
    encoder.set_buffer(2, Some(coefficients_scratch), 0);
    encoder.set_bytes(
        3,
        size_of::<J2kClassicRepeatedBatchParams>() as u64,
        (&raw const repeated).cast(),
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: job_count as u64,
            height: u64::from(repeated.batch_count),
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn encode_classic_sub_band_to_buffer(
    runtime: &MetalRuntime,
    job: &signinum_j2k_native::J2kOwnedSubBandPlan,
    output: &Buffer,
) -> Result<(), Error> {
    if job.jobs.is_empty() {
        return Ok(());
    }

    let mut jobs = Vec::with_capacity(job.jobs.len());
    let mut coded_data = Vec::new();
    let mut segments = Vec::new();

    for block in &job.jobs {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(&block.data);
        let segment_offset = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect segment table exceeds u32".to_string(),
        })?;
        for segment in &block.segments {
            let data_offset = coded_offset
                .checked_add(segment.data_offset)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect segment offset overflow".to_string(),
                })?;
            segments.push(J2kClassicSegment {
                data_offset,
                data_length: segment.data_length,
                start_coding_pass: u32::from(segment.start_coding_pass),
                end_coding_pass: u32::from(segment.end_coding_pass),
                use_arithmetic: u32::from(segment.use_arithmetic),
            });
        }
        jobs.push(J2kClassicCleanupBatchJob {
            coded_offset,
            coded_len: u32::try_from(block.data.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
            })?,
            segment_offset,
            segment_count: u32::try_from(block.segments.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K MetalDirect segment count exceeds u32".to_string(),
            })?,
            width: block.width,
            height: block.height,
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect output offset overflow".to_string(),
                })?,
            missing_msbs: u32::from(block.missing_bit_planes),
            total_bitplanes: u32::from(block.total_bitplanes),
            number_of_coding_passes: u32::from(block.number_of_coding_passes),
            sub_band_type: match block.sub_band_type {
                signinum_j2k_native::J2kSubBandType::LowLow => 0,
                signinum_j2k_native::J2kSubBandType::HighLow => 1,
                signinum_j2k_native::J2kSubBandType::LowHigh => 2,
                signinum_j2k_native::J2kSubBandType::HighHigh => 3,
            },
            style_flags: classic_style_flags(block.style),
            strict: u32::from(block.strict),
            dequantization_step: block.dequantization_step,
        });
    }

    dispatch_classic_cleanup_batched(runtime, &coded_data, &jobs, &segments, output)
}

#[cfg(target_os = "macos")]
fn encode_classic_sub_band_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: &signinum_j2k_native::J2kOwnedSubBandPlan,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<DirectHostRetention>, Vec<Buffer>, DirectStatusCheck), Error> {
    if job.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            Vec::new(),
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let mut jobs = Vec::with_capacity(job.jobs.len());
    let mut coded_data = Vec::new();
    let mut segments = Vec::new();

    for block in &job.jobs {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(&block.data);
        let segment_offset = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect segment table exceeds u32".to_string(),
        })?;
        for segment in &block.segments {
            let data_offset = coded_offset
                .checked_add(segment.data_offset)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect segment offset overflow".to_string(),
                })?;
            segments.push(J2kClassicSegment {
                data_offset,
                data_length: segment.data_length,
                start_coding_pass: u32::from(segment.start_coding_pass),
                end_coding_pass: u32::from(segment.end_coding_pass),
                use_arithmetic: u32::from(segment.use_arithmetic),
            });
        }
        jobs.push(J2kClassicCleanupBatchJob {
            coded_offset,
            coded_len: u32::try_from(block.data.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
            })?,
            segment_offset,
            segment_count: u32::try_from(block.segments.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K MetalDirect segment count exceeds u32".to_string(),
            })?,
            width: block.width,
            height: block.height,
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect output offset overflow".to_string(),
                })?,
            missing_msbs: u32::from(block.missing_bit_planes),
            total_bitplanes: u32::from(block.total_bitplanes),
            number_of_coding_passes: u32::from(block.number_of_coding_passes),
            sub_band_type: match block.sub_band_type {
                signinum_j2k_native::J2kSubBandType::LowLow => 0,
                signinum_j2k_native::J2kSubBandType::HighLow => 1,
                signinum_j2k_native::J2kSubBandType::LowHigh => 2,
                signinum_j2k_native::J2kSubBandType::HighHigh => 3,
            },
            style_flags: classic_style_flags(block.style),
            strict: u32::from(block.strict),
            dequantization_step: block.dequantization_step,
        });
    }

    let coded_buffer = borrow_slice_buffer(&runtime.device, &coded_data);
    let jobs_buffer = borrow_slice_buffer(&runtime.device, &jobs);
    let segments_buffer = borrow_slice_buffer(&runtime.device, &segments);
    let use_plain_fast_path = classic_batch_uses_plain_fast_path(&jobs, &segments)
        && runtime
            .classic_cleanup_plain_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, jobs.len())?;
    let (status_check, states_scratch) = dispatch_classic_cleanup_batched_in_command_buffer(
        runtime,
        command_buffer,
        &coded_buffer,
        &jobs_buffer,
        jobs.len(),
        use_plain_fast_path,
        &segments_buffer,
        output,
        &coefficients_scratch.buffer,
    );
    let mut retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        retained_buffers.push(states_scratch);
    }
    Ok((
        vec![
            DirectHostRetention::Bytes(coded_data),
            DirectHostRetention::ClassicJobs(jobs),
            DirectHostRetention::ClassicSegments(segments),
        ],
        retained_buffers,
        status_check,
    ))
}

#[cfg(target_os = "macos")]
fn encode_distinct_classic_sub_bands_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    sub_bands: &[&PreparedClassicSubBand],
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = sub_bands.first() else {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    };
    let per_instance_len = first.width as usize * first.height as usize;
    encode_distinct_classic_batches_to_buffer_in_command_buffer(
        runtime,
        command_buffer,
        sub_bands.iter().map(|sub_band| DistinctClassicBatch {
            coded_data: &sub_band.coded_data,
            jobs: &sub_band.jobs,
            segments: &sub_band.segments,
            output_base: sub_bands
                .iter()
                .position(|candidate| core::ptr::eq(*candidate, *sub_band))
                .expect("sub-band exists")
                * per_instance_len,
        }),
        output,
        scratch_buffers,
    )
}

#[cfg(target_os = "macos")]
fn encode_distinct_classic_sub_band_groups_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    groups: &[&PreparedClassicSubBandGroup],
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = groups.first() else {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    };
    let per_instance_len = first.total_coefficients;
    encode_distinct_classic_batches_to_buffer_in_command_buffer(
        runtime,
        command_buffer,
        groups
            .iter()
            .enumerate()
            .map(|(index, group)| DistinctClassicBatch {
                coded_data: &group.coded_data,
                jobs: &group.jobs,
                segments: &group.segments,
                output_base: index * per_instance_len,
            }),
        output,
        scratch_buffers,
    )
}

#[cfg(target_os = "macos")]
struct DistinctClassicBatch<'a> {
    coded_data: &'a [u8],
    jobs: &'a [J2kClassicCleanupBatchJob],
    segments: &'a [J2kClassicSegment],
    output_base: usize,
}

#[cfg(target_os = "macos")]
fn encode_distinct_classic_batches_to_buffer_in_command_buffer<'a>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    batches: impl IntoIterator<Item = DistinctClassicBatch<'a>>,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let mut coded_data = Vec::new();
    let mut jobs = Vec::new();
    let mut segments = Vec::new();

    for batch in batches {
        let coded_base = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect distinct color coded payload exceeds u32".to_string(),
        })?;
        let segment_base = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect distinct color segment table exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(batch.coded_data);
        for segment in batch.segments {
            let mut adjusted = *segment;
            adjusted.data_offset =
                adjusted
                    .data_offset
                    .checked_add(coded_base)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect distinct color segment offset overflow"
                            .to_string(),
                    })?;
            segments.push(adjusted);
        }
        let output_base = u32::try_from(batch.output_base).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect distinct color output offset exceeds u32".to_string(),
        })?;
        for job in batch.jobs {
            let mut adjusted = *job;
            adjusted.coded_offset =
                adjusted
                    .coded_offset
                    .checked_add(coded_base)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect distinct color job coded offset overflow"
                            .to_string(),
                    })?;
            adjusted.segment_offset = adjusted
                .segment_offset
                .checked_add(segment_base)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect distinct color job segment offset overflow"
                        .to_string(),
                })?;
            adjusted.output_offset =
                adjusted
                    .output_offset
                    .checked_add(output_base)
                    .ok_or_else(|| Error::MetalKernel {
                        message:
                            "classic J2K MetalDirect distinct color job output offset overflow"
                                .to_string(),
                    })?;
            jobs.push(adjusted);
        }
    }

    if jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = owned_slice_buffer(&runtime.device, &coded_data);
    let jobs_buffer = owned_slice_buffer(&runtime.device, &jobs);
    let segments_buffer = owned_slice_buffer(&runtime.device, &segments);
    let use_plain_fast_path = classic_batch_uses_plain_fast_path(&jobs, &segments)
        && runtime
            .classic_cleanup_plain_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, jobs.len())?;
    let (status_check, states_scratch) = dispatch_classic_cleanup_batched_in_command_buffer(
        runtime,
        command_buffer,
        &coded_buffer,
        &jobs_buffer,
        jobs.len(),
        use_plain_fast_path,
        &segments_buffer,
        output,
        &coefficients_scratch.buffer,
    );
    let mut retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        retained_buffers.push(states_scratch);
    }
    Ok((retained_buffers, status_check))
}

#[cfg(target_os = "macos")]
fn encode_repeated_classic_sub_band_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: &PreparedClassicSubBand,
    count: usize,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<DirectHostRetention>, Vec<Buffer>, DirectStatusCheck), Error> {
    if count == 0 {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            Vec::new(),
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    if job.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            Vec::new(),
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let total_jobs = job
        .jobs
        .len()
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K MetalDirect repeated job count overflow".to_string(),
        })?;
    let coded_buffer = job.coded_buffer.clone();
    let jobs_buffer = job.jobs_buffer.clone();
    let segments_buffer = job.segments_buffer.clone();
    let use_plain_fast_path =
        classic_repeated_uses_plain_fast_path(count, &job.jobs, &job.segments)
            && runtime
                .classic_cleanup_plain_repeated_batched
                .max_total_threads_per_threadgroup()
                >= 32;
    let use_plain_dev_path = !use_plain_fast_path
        && count <= 16
        && classic_batch_is_plain_arithmetic(&job.jobs, &job.segments);
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, total_jobs)?;
    let states_scratch = if use_plain_dev_path {
        Some(take_classic_states_scratch_buffer(runtime, total_jobs)?)
    } else {
        None
    };
    let status_check = if use_plain_fast_path {
        dispatch_classic_cleanup_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &coded_buffer,
            &jobs_buffer,
            job.jobs.len(),
            total_jobs,
            job.width as usize * job.height as usize,
            true,
            &segments_buffer,
            output,
            &coefficients_scratch.buffer,
        )
    } else if let Some(states_scratch) = states_scratch.as_ref() {
        dispatch_classic_cleanup_plain_dev_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &coded_buffer,
            &jobs_buffer,
            job.jobs.len(),
            total_jobs,
            job.width as usize * job.height as usize,
            &segments_buffer,
            output,
            &coefficients_scratch.buffer,
            &states_scratch.buffer,
        )
    } else {
        dispatch_classic_cleanup_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &coded_buffer,
            &jobs_buffer,
            job.jobs.len(),
            total_jobs,
            job.width as usize * job.height as usize,
            use_plain_fast_path,
            &segments_buffer,
            output,
            &coefficients_scratch.buffer,
        )
    };
    if !use_plain_fast_path {
        dispatch_classic_store_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &jobs_buffer,
            job.jobs.len(),
            total_jobs,
            job.width as usize * job.height as usize,
            output,
            &coefficients_scratch.buffer,
        );
    }
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        scratch_buffers.push(states_scratch);
    }
    let retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    Ok((Vec::new(), retained_buffers, status_check))
}

#[cfg(target_os = "macos")]
fn encode_repeated_classic_sub_band_group_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    group: &PreparedClassicSubBandGroup,
    count: usize,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<DirectHostRetention>, Vec<Buffer>, DirectStatusCheck), Error> {
    if count == 0 || group.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            Vec::new(),
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let total_jobs = group
        .jobs
        .len()
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K MetalDirect repeated grouped job count overflow".to_string(),
        })?;
    let coded_buffer = group.coded_buffer.clone();
    let jobs_buffer = group.jobs_buffer.clone();
    let segments_buffer = group.segments_buffer.clone();
    let use_plain_fast_path =
        classic_repeated_uses_plain_fast_path(count, &group.jobs, &group.segments)
            && runtime
                .classic_cleanup_plain_repeated_batched
                .max_total_threads_per_threadgroup()
                >= 32;
    let use_plain_dev_path = !use_plain_fast_path
        && count <= 16
        && classic_batch_is_plain_arithmetic(&group.jobs, &group.segments);
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, total_jobs)?;
    let states_scratch = if use_plain_dev_path {
        Some(take_classic_states_scratch_buffer(runtime, total_jobs)?)
    } else {
        None
    };
    let status_check = if use_plain_fast_path {
        dispatch_classic_cleanup_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &coded_buffer,
            &jobs_buffer,
            group.jobs.len(),
            total_jobs,
            group.total_coefficients,
            true,
            &segments_buffer,
            output,
            &coefficients_scratch.buffer,
        )
    } else if let Some(states_scratch) = states_scratch.as_ref() {
        dispatch_classic_cleanup_plain_dev_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &coded_buffer,
            &jobs_buffer,
            group.jobs.len(),
            total_jobs,
            group.total_coefficients,
            &segments_buffer,
            output,
            &coefficients_scratch.buffer,
            &states_scratch.buffer,
        )
    } else {
        dispatch_classic_cleanup_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &coded_buffer,
            &jobs_buffer,
            group.jobs.len(),
            total_jobs,
            group.total_coefficients,
            use_plain_fast_path,
            &segments_buffer,
            output,
            &coefficients_scratch.buffer,
        )
    };
    if !use_plain_fast_path {
        dispatch_classic_store_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &jobs_buffer,
            group.jobs.len(),
            total_jobs,
            group.total_coefficients,
            output,
            &coefficients_scratch.buffer,
        );
    }
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        scratch_buffers.push(states_scratch);
    }
    let retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    Ok((Vec::new(), retained_buffers, status_check))
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn encode_prepared_classic_sub_band_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: &PreparedClassicSubBand,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let encoder = command_buffer.new_compute_command_encoder();
    let result = encode_prepared_classic_sub_band_to_buffer_in_encoder(
        runtime,
        encoder,
        job,
        output,
        scratch_buffers,
    );
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
fn encode_prepared_classic_sub_band_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    job: &PreparedClassicSubBand,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if job.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = job.coded_buffer.clone();
    let jobs_buffer = job.jobs_buffer.clone();
    let segments_buffer = job.segments_buffer.clone();
    let use_plain_fast_path = classic_batch_uses_plain_fast_path(&job.jobs, &job.segments)
        && runtime
            .classic_cleanup_plain_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, job.jobs.len())?;
    let (status_check, states_scratch) = dispatch_classic_cleanup_batched_in_encoder(
        runtime,
        encoder,
        &coded_buffer,
        &jobs_buffer,
        job.jobs.len(),
        use_plain_fast_path,
        &segments_buffer,
        output,
        &coefficients_scratch.buffer,
    );
    let mut retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        retained_buffers.push(states_scratch);
    }
    Ok((retained_buffers, status_check))
}

#[cfg(target_os = "macos")]
fn encode_prepared_classic_sub_band_group_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    group: &PreparedClassicSubBandGroup,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if group.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = group.coded_buffer.clone();
    let jobs_buffer = group.jobs_buffer.clone();
    let segments_buffer = group.segments_buffer.clone();
    let use_plain_fast_path = classic_batch_uses_plain_fast_path(&group.jobs, &group.segments)
        && runtime
            .classic_cleanup_plain_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, group.jobs.len())?;
    let (status_check, states_scratch) = dispatch_classic_cleanup_batched_in_encoder(
        runtime,
        encoder,
        &coded_buffer,
        &jobs_buffer,
        group.jobs.len(),
        use_plain_fast_path,
        &segments_buffer,
        output,
        &coefficients_scratch.buffer,
    );
    let mut retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        retained_buffers.push(states_scratch);
    }
    Ok((retained_buffers, status_check))
}

#[cfg(target_os = "macos")]
fn required_ht_output_len(job: HtCodeBlockDecodeJob<'_>) -> Result<usize, Error> {
    if job.height == 0 {
        return Ok(0);
    }

    job.output_stride
        .checked_mul(job.height as usize - 1)
        .and_then(|prefix| prefix.checked_add(job.width as usize))
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal output size overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn encode_ht_sub_band_to_buffer(
    runtime: &MetalRuntime,
    job: &signinum_j2k_native::HtOwnedSubBandPlan,
    output: &Buffer,
) -> Result<(), Error> {
    if job.jobs.is_empty() {
        return Ok(());
    }

    let mut jobs = Vec::with_capacity(job.jobs.len());
    let mut coded_data = Vec::new();
    for block in &job.jobs {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(&block.data);
        jobs.push(J2kHtCleanupBatchJob {
            coded_offset,
            width: block.width,
            height: block.height,
            coded_len: u32::try_from(block.data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
            })?,
            cleanup_length: block.cleanup_length,
            refinement_length: block.refinement_length,
            missing_msbs: u32::from(block.missing_bit_planes),
            num_bitplanes: u32::from(block.num_bitplanes),
            number_of_coding_passes: u32::from(block.number_of_coding_passes),
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K MetalDirect output offset overflow".to_string(),
                })?,
            dequantization_step: block.dequantization_step,
            stripe_causal: u32::from(block.stripe_causal),
        });
    }

    dispatch_ht_cleanup_batched(runtime, &coded_data, &jobs, output)
}

#[cfg(target_os = "macos")]
fn encode_ht_sub_band_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: &signinum_j2k_native::HtOwnedSubBandPlan,
    output: &Buffer,
) -> Result<(Vec<DirectHostRetention>, Vec<Buffer>, DirectStatusCheck), Error> {
    if job.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            Vec::new(),
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let mut jobs = Vec::with_capacity(job.jobs.len());
    let mut coded_data = Vec::new();
    for block in &job.jobs {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(&block.data);
        jobs.push(J2kHtCleanupBatchJob {
            coded_offset,
            width: block.width,
            height: block.height,
            coded_len: u32::try_from(block.data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
            })?,
            cleanup_length: block.cleanup_length,
            refinement_length: block.refinement_length,
            missing_msbs: u32::from(block.missing_bit_planes),
            num_bitplanes: u32::from(block.num_bitplanes),
            number_of_coding_passes: u32::from(block.number_of_coding_passes),
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K MetalDirect output offset overflow".to_string(),
                })?,
            dequantization_step: block.dequantization_step,
            stripe_causal: u32::from(block.stripe_causal),
        });
    }

    let coded_buffer = borrow_slice_buffer(&runtime.device, &coded_data);
    let jobs_buffer = borrow_slice_buffer(&runtime.device, &jobs);
    let status_check = dispatch_ht_cleanup_batched_in_command_buffer(
        runtime,
        command_buffer,
        &coded_buffer,
        &jobs_buffer,
        jobs.len(),
        output,
        ht_batch_output_word_count(&jobs)?,
    )?;
    Ok((
        vec![
            DirectHostRetention::Bytes(coded_data),
            DirectHostRetention::HtJobs(jobs),
        ],
        vec![coded_buffer, jobs_buffer],
        status_check,
    ))
}

#[cfg(target_os = "macos")]
fn encode_repeated_ht_sub_band_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: &PreparedHtSubBand,
    count: usize,
    output: &Buffer,
) -> Result<(Vec<DirectHostRetention>, Vec<Buffer>, DirectStatusCheck), Error> {
    if count == 0 || job.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            Vec::new(),
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let total_jobs = job
        .jobs
        .len()
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K MetalDirect repeated job count overflow".to_string(),
        })?;
    let coded_buffer = job.coded_buffer.clone();
    let jobs_buffer = job.jobs_buffer.clone();
    let status_check = dispatch_ht_cleanup_repeated_batched_in_command_buffer(
        runtime,
        command_buffer,
        &coded_buffer,
        &jobs_buffer,
        job.jobs.len(),
        total_jobs,
        job.width as usize * job.height as usize,
        output,
    )?;
    Ok((Vec::new(), vec![coded_buffer, jobs_buffer], status_check))
}

#[cfg(target_os = "macos")]
fn encode_repeated_ht_sub_band_group_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    group: &PreparedHtSubBandGroup,
    count: usize,
    output: &Buffer,
) -> Result<(Vec<DirectHostRetention>, Vec<Buffer>, DirectStatusCheck), Error> {
    if count == 0 || group.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            Vec::new(),
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let total_jobs = group
        .jobs
        .len()
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K MetalDirect repeated grouped job count overflow".to_string(),
        })?;
    let coded_buffer = group.coded_buffer.clone();
    let jobs_buffer = group.jobs_buffer.clone();
    let status_check = dispatch_ht_cleanup_repeated_batched_in_command_buffer(
        runtime,
        command_buffer,
        &coded_buffer,
        &jobs_buffer,
        group.jobs.len(),
        total_jobs,
        group.total_coefficients,
        output,
    )?;
    Ok((Vec::new(), vec![coded_buffer, jobs_buffer], status_check))
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn encode_prepared_ht_sub_band_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: &PreparedHtSubBand,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let encoder = command_buffer.new_compute_command_encoder();
    let result = encode_prepared_ht_sub_band_to_buffer_in_encoder(runtime, encoder, job, output);
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
fn encode_prepared_ht_sub_band_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    job: &PreparedHtSubBand,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if job.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = job.coded_buffer.clone();
    let jobs_buffer = job.jobs_buffer.clone();
    let status_check = dispatch_ht_cleanup_batched_in_encoder(
        runtime,
        encoder,
        &coded_buffer,
        &jobs_buffer,
        job.jobs.len(),
        output,
        job.width as usize * job.height as usize,
    )?;
    Ok((vec![coded_buffer, jobs_buffer], status_check))
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn encode_prepared_ht_sub_band_group_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    group: &PreparedHtSubBandGroup,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let encoder = command_buffer.new_compute_command_encoder();
    let result =
        encode_prepared_ht_sub_band_group_to_buffer_in_encoder(runtime, encoder, group, output);
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
fn encode_prepared_ht_sub_band_group_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    group: &PreparedHtSubBandGroup,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if group.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = group.coded_buffer.clone();
    let jobs_buffer = group.jobs_buffer.clone();
    let status_check = dispatch_ht_cleanup_batched_in_encoder(
        runtime,
        encoder,
        &coded_buffer,
        &jobs_buffer,
        group.jobs.len(),
        output,
        group.total_coefficients,
    )?;
    Ok((vec![coded_buffer, jobs_buffer], status_check))
}

#[cfg(target_os = "macos")]
fn decode_ht_status_error(status: J2kHtStatus) -> Error {
    let kind = match status.code {
        J2K_HT_STATUS_FAIL => "decode failure",
        J2K_HT_STATUS_UNSUPPORTED => "unsupported HT kernel input",
        _ => "unexpected HT kernel status",
    };
    Error::MetalKernel {
        message: format!("HTJ2K Metal kernel {kind} (detail={})", status.detail),
    }
}

#[cfg(target_os = "macos")]
fn ht_output_word_count(
    output_offset: u32,
    output_stride: u32,
    width: u32,
    height: u32,
) -> Result<usize, Error> {
    let end = if width == 0 || height == 0 {
        u64::from(output_offset)
    } else {
        u64::from(output_offset)
            .checked_add(u64::from(height - 1) * u64::from(output_stride))
            .and_then(|offset| offset.checked_add(u64::from(width)))
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal output span overflow".to_string(),
            })?
    };
    usize::try_from(end).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal output span exceeds usize".to_string(),
    })
}

#[cfg(target_os = "macos")]
fn ht_batch_output_word_count(jobs: &[J2kHtCleanupBatchJob]) -> Result<usize, Error> {
    let mut word_count = 0usize;
    for job in jobs {
        let job_word_count =
            ht_output_word_count(job.output_offset, job.output_stride, job.width, job.height)?;
        word_count = word_count.max(job_word_count);
    }
    Ok(word_count)
}

#[cfg(target_os = "macos")]
fn dispatch_zero_u32_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    buffer: &Buffer,
    word_count: usize,
) -> Result<(), Error> {
    let word_count = u32::try_from(word_count).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal zero-fill word count exceeds u32".to_string(),
    })?;
    if word_count == 0 {
        return Ok(());
    }

    encoder.set_compute_pipeline_state(&runtime.zero_u32_buffer);
    encoder.set_buffer(0, Some(buffer), 0);
    encoder.set_bytes(1, size_of::<u32>() as u64, (&raw const word_count).cast());
    let width = runtime.zero_u32_buffer.thread_execution_width().max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(word_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn encode_status_error(stage: &str, code: u32, detail: u32) -> Error {
    let kind = match code {
        J2K_ENCODE_STATUS_FAIL => "failure",
        J2K_ENCODE_STATUS_UNSUPPORTED => "unsupported input",
        _ => "unexpected status",
    };
    Error::MetalKernel {
        message: format!("{stage} Metal encode kernel {kind} (detail={detail})"),
    }
}

#[cfg(target_os = "macos")]
fn classic_encode_output_capacity(
    width: u32,
    height: u32,
    total_bitplanes: u8,
) -> Result<usize, Error> {
    let samples = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal encode block size overflow".to_string(),
        })?;
    samples
        .checked_mul(usize::from(total_bitplanes).max(1))
        .and_then(|bits| bits.checked_mul(8))
        .and_then(|bytes| bytes.checked_add(4096))
        .map(|bytes| bytes.max(4096) + 1)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal encode output capacity overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
fn classic_encode_sub_band_code(sub_band_type: signinum_j2k_native::J2kSubBandType) -> u32 {
    match sub_band_type {
        signinum_j2k_native::J2kSubBandType::LowLow => 0,
        signinum_j2k_native::J2kSubBandType::HighLow => 1,
        signinum_j2k_native::J2kSubBandType::LowHigh => 2,
        signinum_j2k_native::J2kSubBandType::HighHigh => 3,
    }
}

#[cfg(target_os = "macos")]
fn read_classic_encoded_code_block(
    status: J2kClassicEncodeStatus,
    output: &Buffer,
    output_offset: usize,
    output_capacity: usize,
    segment_buffer: &Buffer,
    segment_offset: usize,
    segment_capacity: usize,
) -> Result<EncodedJ2kCodeBlock, Error> {
    if status.code != J2K_ENCODE_STATUS_OK {
        return Err(encode_status_error(
            "classic Tier-1",
            status.code,
            status.detail,
        ));
    }
    let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
        message: "classic J2K Metal encode length exceeds usize".to_string(),
    })?;
    if data_len > output_capacity {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal encode length exceeds output buffer".to_string(),
        });
    }
    let data = if data_len == 0 {
        Vec::new()
    } else {
        unsafe {
            core::slice::from_raw_parts(output.contents().cast::<u8>().add(output_offset), data_len)
        }
        .to_vec()
    };
    let number_of_coding_passes =
        u8::try_from(status.number_of_coding_passes).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal encode pass count exceeds u8".to_string(),
        })?;
    let missing_bit_planes =
        u8::try_from(status.missing_bit_planes).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal encode missing bitplanes exceeds u8".to_string(),
        })?;
    let segment_count = usize::try_from(status.segment_count).map_err(|_| Error::MetalKernel {
        message: "classic J2K Metal encode segment count exceeds usize".to_string(),
    })?;
    if segment_count > segment_capacity {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal encode segment count exceeds buffer".to_string(),
        });
    }
    let raw_segments = if segment_count == 0 {
        &[][..]
    } else {
        unsafe {
            core::slice::from_raw_parts(
                segment_buffer
                    .contents()
                    .cast::<J2kClassicSegment>()
                    .add(segment_offset),
                segment_count,
            )
        }
    };
    let segments = raw_segments
        .iter()
        .map(|segment| {
            Ok(J2kCodeBlockSegment {
                data_offset: segment.data_offset,
                data_length: segment.data_length,
                start_coding_pass: u8::try_from(segment.start_coding_pass).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal encode segment start pass exceeds u8"
                            .to_string(),
                    }
                })?,
                end_coding_pass: u8::try_from(segment.end_coding_pass).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal encode segment end pass exceeds u8".to_string(),
                    }
                })?,
                use_arithmetic: segment.use_arithmetic != 0,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;

    Ok(EncodedJ2kCodeBlock {
        data,
        segments,
        number_of_coding_passes,
        missing_bit_planes,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_classic_tier1_code_blocks(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    with_runtime(|runtime| {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let mut coefficients = Vec::<i32>::new();
        let mut batch_jobs = Vec::<J2kClassicEncodeBatchJob>::with_capacity(jobs.len());
        let mut output_capacity_total = 0usize;
        let mut segment_capacity_total = 0usize;
        let segment_capacity_per_job = 256usize;

        for job in jobs {
            let expected_coefficients = usize::try_from(job.width)
                .ok()
                .and_then(|w| {
                    usize::try_from(job.height)
                        .ok()
                        .and_then(|h| w.checked_mul(h))
                })
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode coefficient count overflow".to_string(),
                })?;
            if job.coefficients.len() < expected_coefficients {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal encode coefficient slice is too small".to_string(),
                });
            }
            let coefficient_offset =
                u32::try_from(coefficients.len()).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode coefficient table exceeds u32".to_string(),
                })?;
            coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
            let output_capacity =
                classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode output table exceeds u32".to_string(),
                })?;
            let segment_offset =
                u32::try_from(segment_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode segment table exceeds u32".to_string(),
                })?;
            batch_jobs.push(J2kClassicEncodeBatchJob {
                coefficient_offset,
                output_offset,
                segment_offset,
                width: job.width,
                height: job.height,
                sub_band_type: classic_encode_sub_band_code(job.sub_band_type),
                total_bitplanes: u32::from(job.total_bitplanes),
                style_flags: classic_style_flags(job.style),
                output_capacity: u32::try_from(output_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal encode output capacity exceeds u32".to_string(),
                    }
                })?,
                segment_capacity: u32::try_from(segment_capacity_per_job).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal encode segment capacity exceeds u32"
                            .to_string(),
                    }
                })?,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode output buffer overflow".to_string(),
                })?;
            segment_capacity_total = segment_capacity_total
                .checked_add(segment_capacity_per_job)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode segment buffer overflow".to_string(),
                })?;
        }

        let coefficient_buffer = owned_slice_buffer(&runtime.device, &coefficients);
        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let output = runtime.device.new_buffer(
            output_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status_buffer = runtime.device.new_buffer(
            (jobs.len() * size_of::<J2kClassicEncodeStatus>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let segment_buffer = runtime.device.new_buffer(
            (segment_capacity_total * size_of::<J2kClassicSegment>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal encode job count exceeds u32".to_string(),
        })?;

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.classic_encode_code_blocks);
        encoder.set_buffer(0, Some(&coefficient_buffer), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_buffer(2, Some(&job_buffer), 0);
        encoder.set_buffer(3, Some(&status_buffer), 0);
        encoder.set_buffer(4, Some(&segment_buffer), 0);
        encoder.set_bytes(5, size_of::<u32>() as u64, (&raw const job_count).cast());
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(job_count),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: runtime
                    .classic_encode_code_blocks
                    .thread_execution_width()
                    .max(1),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let statuses = unsafe {
            core::slice::from_raw_parts(
                status_buffer.contents().cast::<J2kClassicEncodeStatus>(),
                jobs.len(),
            )
        };
        let mut results = Vec::with_capacity(jobs.len());
        for (idx, status) in statuses.iter().copied().enumerate() {
            let batch_job = batch_jobs[idx];
            results.push(read_classic_encoded_code_block(
                status,
                &output,
                usize::try_from(batch_job.output_offset).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode output offset exceeds usize".to_string(),
                })?,
                usize::try_from(batch_job.output_capacity).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode output capacity exceeds usize".to_string(),
                })?,
                &segment_buffer,
                usize::try_from(batch_job.segment_offset).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode segment offset exceeds usize".to_string(),
                })?,
                usize::try_from(batch_job.segment_capacity).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode segment capacity exceeds usize".to_string(),
                })?,
            )?);
        }

        Ok(results)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_classic_tier1_code_block(
    job: J2kTier1CodeBlockEncodeJob<'_>,
) -> Result<EncodedJ2kCodeBlock, Error> {
    with_runtime(|runtime| {
        let expected_coefficients = usize::try_from(job.width)
            .ok()
            .and_then(|w| {
                usize::try_from(job.height)
                    .ok()
                    .and_then(|h| w.checked_mul(h))
            })
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K Metal encode coefficient count overflow".to_string(),
            })?;
        if job.coefficients.len() < expected_coefficients {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal encode coefficient slice is too small".to_string(),
            });
        }

        let output_capacity =
            classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?;
        let output_capacity_u32 =
            u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode output capacity exceeds u32".to_string(),
            })?;
        let params = J2kClassicEncodeParams {
            width: job.width,
            height: job.height,
            sub_band_type: classic_encode_sub_band_code(job.sub_band_type),
            total_bitplanes: u32::from(job.total_bitplanes),
            style_flags: classic_style_flags(job.style),
            output_capacity: output_capacity_u32,
            segment_capacity: 256,
        };
        let coefficients =
            borrow_slice_buffer(&runtime.device, &job.coefficients[..expected_coefficients]);
        let output = runtime.device.new_buffer(
            output_capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status_buffer = runtime.device.new_buffer(
            size_of::<J2kClassicEncodeStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let segment_buffer = runtime.device.new_buffer(
            (usize::try_from(params.segment_capacity).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode segment capacity exceeds usize".to_string(),
            })? * size_of::<J2kClassicSegment>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.classic_encode_code_block);
        encoder.set_buffer(0, Some(&coefficients), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_bytes(
            2,
            size_of::<J2kClassicEncodeParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(3, Some(&status_buffer), 0);
        encoder.set_buffer(4, Some(&segment_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = unsafe {
            status_buffer
                .contents()
                .cast::<J2kClassicEncodeStatus>()
                .read()
        };
        if status.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                "classic Tier-1",
                status.code,
                status.detail,
            ));
        }
        let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal encode length exceeds usize".to_string(),
        })?;
        if data_len > output_capacity {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal encode length exceeds output buffer".to_string(),
            });
        }
        let data = if data_len == 0 {
            Vec::new()
        } else {
            unsafe { core::slice::from_raw_parts(output.contents().cast::<u8>(), data_len) }
                .to_vec()
        };
        let number_of_coding_passes =
            u8::try_from(status.number_of_coding_passes).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode pass count exceeds u8".to_string(),
            })?;
        let missing_bit_planes =
            u8::try_from(status.missing_bit_planes).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode missing bitplanes exceeds u8".to_string(),
            })?;
        let segment_count =
            usize::try_from(status.segment_count).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode segment count exceeds usize".to_string(),
            })?;
        let segment_capacity =
            usize::try_from(params.segment_capacity).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode segment capacity exceeds usize".to_string(),
            })?;
        if segment_count > segment_capacity {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal encode segment count exceeds buffer".to_string(),
            });
        }
        let raw_segments = if segment_count == 0 {
            &[][..]
        } else {
            unsafe {
                core::slice::from_raw_parts(
                    segment_buffer.contents().cast::<J2kClassicSegment>(),
                    segment_count,
                )
            }
        };
        let segments = raw_segments
            .iter()
            .map(|segment| {
                Ok(J2kCodeBlockSegment {
                    data_offset: segment.data_offset,
                    data_length: segment.data_length,
                    start_coding_pass: u8::try_from(segment.start_coding_pass).map_err(|_| {
                        Error::MetalKernel {
                            message: "classic J2K Metal encode segment start pass exceeds u8"
                                .to_string(),
                        }
                    })?,
                    end_coding_pass: u8::try_from(segment.end_coding_pass).map_err(|_| {
                        Error::MetalKernel {
                            message: "classic J2K Metal encode segment end pass exceeds u8"
                                .to_string(),
                        }
                    })?,
                    use_arithmetic: segment.use_arithmetic != 0,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;

        Ok(EncodedJ2kCodeBlock {
            data,
            segments,
            number_of_coding_passes,
            missing_bit_planes,
        })
    })
}

#[cfg(target_os = "macos")]
fn read_ht_encoded_code_block(
    status: J2kHtEncodeStatus,
    output: &Buffer,
    output_offset: usize,
    output_capacity: usize,
) -> Result<EncodedHtJ2kCodeBlock, Error> {
    if status.code != J2K_ENCODE_STATUS_OK {
        return Err(encode_status_error(
            "HTJ2K cleanup",
            status.code,
            status.detail,
        ));
    }
    let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal encode length exceeds usize".to_string(),
    })?;
    if data_len > output_capacity {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal encode length exceeds output buffer".to_string(),
        });
    }
    let data = if data_len == 0 {
        Vec::new()
    } else {
        unsafe {
            core::slice::from_raw_parts(output.contents().cast::<u8>().add(output_offset), data_len)
        }
        .to_vec()
    };
    Ok(EncodedHtJ2kCodeBlock {
        data,
        num_coding_passes: u8::try_from(status.num_coding_passes).map_err(|_| {
            Error::MetalKernel {
                message: "HTJ2K Metal encode pass count exceeds u8".to_string(),
            }
        })?,
        num_zero_bitplanes: u8::try_from(status.num_zero_bitplanes).map_err(|_| {
            Error::MetalKernel {
                message: "HTJ2K Metal encode zero bitplanes exceeds u8".to_string(),
            }
        })?,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_ht_cleanup_code_blocks(
    jobs: &[J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedHtJ2kCodeBlock>, Error> {
    with_runtime(|runtime| {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }

        let output_capacity =
            J2K_HT_ENCODE_MS_SIZE + J2K_HT_ENCODE_MEL_SIZE + J2K_HT_ENCODE_VLC_SIZE;
        let output_capacity_u32 =
            u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode output capacity exceeds u32".to_string(),
            })?;
        let mut coefficients = Vec::<i32>::new();
        let mut batch_jobs = Vec::<J2kHtEncodeBatchJob>::with_capacity(jobs.len());
        let mut output_capacity_total = 0usize;

        for job in jobs {
            let expected_coefficients = usize::try_from(job.width)
                .ok()
                .and_then(|w| {
                    usize::try_from(job.height)
                        .ok()
                        .and_then(|h| w.checked_mul(h))
                })
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal encode coefficient count overflow".to_string(),
                })?;
            if job.coefficients.len() < expected_coefficients {
                return Err(Error::MetalKernel {
                    message: "HTJ2K Metal encode coefficient slice is too small".to_string(),
                });
            }
            let coefficient_offset =
                u32::try_from(coefficients.len()).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal encode coefficient table exceeds u32".to_string(),
                })?;
            coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal encode output table exceeds u32".to_string(),
                })?;
            batch_jobs.push(J2kHtEncodeBatchJob {
                coefficient_offset,
                output_offset,
                width: job.width,
                height: job.height,
                total_bitplanes: u32::from(job.total_bitplanes),
                output_capacity: output_capacity_u32,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal encode output buffer overflow".to_string(),
                })?;
        }

        let coefficient_buffer = owned_slice_buffer(&runtime.device, &coefficients);
        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let output = runtime.device.new_buffer(
            output_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status_buffer = runtime.device.new_buffer(
            (jobs.len() * size_of::<J2kHtEncodeStatus>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K Metal encode job count exceeds u32".to_string(),
        })?;

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.ht_encode_code_blocks);
        encoder.set_buffer(0, Some(&coefficient_buffer), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_buffer(2, Some(&job_buffer), 0);
        encoder.set_buffer(3, Some(&runtime.ht_vlc_encode_table0), 0);
        encoder.set_buffer(4, Some(&runtime.ht_vlc_encode_table1), 0);
        encoder.set_buffer(5, Some(&runtime.ht_uvlc_encode_table), 0);
        encoder.set_buffer(6, Some(&status_buffer), 0);
        encoder.set_bytes(7, size_of::<u32>() as u64, (&raw const job_count).cast());
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(job_count),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: runtime
                    .ht_encode_code_blocks
                    .thread_execution_width()
                    .max(1),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let statuses = unsafe {
            core::slice::from_raw_parts(
                status_buffer.contents().cast::<J2kHtEncodeStatus>(),
                jobs.len(),
            )
        };
        let mut results = Vec::with_capacity(jobs.len());
        for (idx, status) in statuses.iter().copied().enumerate() {
            let batch_job = batch_jobs[idx];
            results.push(read_ht_encoded_code_block(
                status,
                &output,
                usize::try_from(batch_job.output_offset).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal encode output offset exceeds usize".to_string(),
                })?,
                usize::try_from(batch_job.output_capacity).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal encode output capacity exceeds usize".to_string(),
                })?,
            )?);
        }

        Ok(results)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_ht_cleanup_code_block(
    job: J2kHtCodeBlockEncodeJob<'_>,
) -> Result<EncodedHtJ2kCodeBlock, Error> {
    with_runtime(|runtime| {
        let expected_coefficients = usize::try_from(job.width)
            .ok()
            .and_then(|w| {
                usize::try_from(job.height)
                    .ok()
                    .and_then(|h| w.checked_mul(h))
            })
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal encode coefficient count overflow".to_string(),
            })?;
        if job.coefficients.len() < expected_coefficients {
            return Err(Error::MetalKernel {
                message: "HTJ2K Metal encode coefficient slice is too small".to_string(),
            });
        }
        let output_capacity =
            J2K_HT_ENCODE_MS_SIZE + J2K_HT_ENCODE_MEL_SIZE + J2K_HT_ENCODE_VLC_SIZE;
        let output_capacity_u32 =
            u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode output capacity exceeds u32".to_string(),
            })?;
        let params = J2kHtEncodeParams {
            width: job.width,
            height: job.height,
            total_bitplanes: u32::from(job.total_bitplanes),
            output_capacity: output_capacity_u32,
        };
        let coefficients =
            borrow_slice_buffer(&runtime.device, &job.coefficients[..expected_coefficients]);
        let output = runtime.device.new_buffer(
            output_capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status_buffer = runtime.device.new_buffer(
            size_of::<J2kHtEncodeStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.ht_encode_code_block);
        encoder.set_buffer(0, Some(&coefficients), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_bytes(
            2,
            size_of::<J2kHtEncodeParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(3, Some(&runtime.ht_vlc_encode_table0), 0);
        encoder.set_buffer(4, Some(&runtime.ht_vlc_encode_table1), 0);
        encoder.set_buffer(5, Some(&runtime.ht_uvlc_encode_table), 0);
        encoder.set_buffer(6, Some(&status_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = unsafe { status_buffer.contents().cast::<J2kHtEncodeStatus>().read() };
        if status.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                "HTJ2K cleanup",
                status.code,
                status.detail,
            ));
        }
        let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
            message: "HTJ2K Metal encode length exceeds usize".to_string(),
        })?;
        if data_len > output_capacity {
            return Err(Error::MetalKernel {
                message: "HTJ2K Metal encode length exceeds output buffer".to_string(),
            });
        }
        let data = if data_len == 0 {
            Vec::new()
        } else {
            unsafe { core::slice::from_raw_parts(output.contents().cast::<u8>(), data_len) }
                .to_vec()
        };
        Ok(EncodedHtJ2kCodeBlock {
            data,
            num_coding_passes: u8::try_from(status.num_coding_passes).map_err(|_| {
                Error::MetalKernel {
                    message: "HTJ2K Metal encode pass count exceeds u8".to_string(),
                }
            })?,
            num_zero_bitplanes: u8::try_from(status.num_zero_bitplanes).map_err(|_| {
                Error::MetalKernel {
                    message: "HTJ2K Metal encode zero bitplanes exceeds u8".to_string(),
                }
            })?,
        })
    })
}

#[cfg(target_os = "macos")]
fn packet_tree_node_count(width: u32, height: u32) -> Result<usize, Error> {
    if width == 0 || height == 0 {
        return Ok(0);
    }
    let mut total = 0usize;
    let mut w = width;
    let mut h = height;
    loop {
        total = total
            .checked_add(
                usize::try_from(w)
                    .ok()
                    .and_then(|wu| usize::try_from(h).ok().and_then(|hu| wu.checked_mul(hu)))
                    .ok_or_else(|| Error::MetalKernel {
                        message: "Tier-2 Metal packet tag-tree size overflow".to_string(),
                    })?,
            )
            .ok_or_else(|| Error::MetalKernel {
                message: "Tier-2 Metal packet tag-tree node count overflow".to_string(),
            })?;
        if w <= 1 && h <= 1 {
            break;
        }
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    Ok(total)
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_tier2_packetization(
    job: J2kPacketizationEncodeJob<'_>,
) -> Result<Vec<u8>, Error> {
    with_runtime(|runtime| {
        let mut resolutions = Vec::<J2kPacketResolution>::new();
        let mut subbands = Vec::<J2kPacketSubband>::new();
        let mut blocks = Vec::<J2kPacketBlock>::new();
        let mut payload = Vec::<u8>::new();
        let mut max_tree_nodes = 1usize;

        for resolution in job.resolutions {
            let subband_offset = u32::try_from(subbands.len()).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal packet subband table exceeds u32".to_string(),
            })?;
            for subband in &resolution.subbands {
                let block_offset = u32::try_from(blocks.len()).map_err(|_| Error::MetalKernel {
                    message: "Tier-2 Metal packet block table exceeds u32".to_string(),
                })?;
                max_tree_nodes = max_tree_nodes.max(packet_tree_node_count(
                    subband.num_cbs_x,
                    subband.num_cbs_y,
                )?);
                for code_block in &subband.code_blocks {
                    let data_offset =
                        u32::try_from(payload.len()).map_err(|_| Error::MetalKernel {
                            message: "Tier-2 Metal packet payload exceeds u32".to_string(),
                        })?;
                    let data_len =
                        u32::try_from(code_block.data.len()).map_err(|_| Error::MetalKernel {
                            message: "Tier-2 Metal packet code-block payload exceeds u32"
                                .to_string(),
                        })?;
                    payload.extend_from_slice(code_block.data);
                    blocks.push(J2kPacketBlock {
                        data_offset,
                        data_len,
                        num_coding_passes: u32::from(code_block.num_coding_passes),
                        num_zero_bitplanes: u32::from(code_block.num_zero_bitplanes),
                        previously_included: u32::from(code_block.previously_included),
                        l_block: code_block.l_block,
                        block_coding_mode: match code_block.block_coding_mode {
                            J2kPacketizationBlockCodingMode::Classic => 0,
                            J2kPacketizationBlockCodingMode::HighThroughput => 1,
                        },
                        reserved0: 0,
                    });
                }
                subbands.push(J2kPacketSubband {
                    block_offset,
                    block_count: u32::try_from(subband.code_blocks.len()).map_err(|_| {
                        Error::MetalKernel {
                            message: "Tier-2 Metal packet subband block count exceeds u32"
                                .to_string(),
                        }
                    })?,
                    num_cbs_x: subband.num_cbs_x,
                    num_cbs_y: subband.num_cbs_y,
                });
            }
            resolutions.push(J2kPacketResolution {
                subband_offset,
                subband_count: u32::try_from(resolution.subbands.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "Tier-2 Metal packet resolution subband count exceeds u32"
                            .to_string(),
                    }
                })?,
            });
        }

        let mut state_block_offsets = HashMap::<u32, (u32, usize)>::new();
        let mut state_blocks = Vec::<J2kPacketStateBlock>::new();
        let mut descriptors =
            Vec::<J2kPacketDescriptor>::with_capacity(job.packet_descriptors.len());
        for descriptor in job.packet_descriptors {
            let packet_index =
                usize::try_from(descriptor.packet_index).map_err(|_| Error::MetalKernel {
                    message: "Tier-2 Metal packet descriptor packet index exceeds usize"
                        .to_string(),
                })?;
            let resolution = resolutions
                .get(packet_index)
                .ok_or_else(|| Error::MetalKernel {
                    message: "Tier-2 Metal packet descriptor packet index out of range".to_string(),
                })?;
            let subband_start =
                usize::try_from(resolution.subband_offset).map_err(|_| Error::MetalKernel {
                    message: "Tier-2 Metal packet descriptor subband offset exceeds usize"
                        .to_string(),
                })?;
            let subband_count =
                usize::try_from(resolution.subband_count).map_err(|_| Error::MetalKernel {
                    message: "Tier-2 Metal packet descriptor subband count exceeds usize"
                        .to_string(),
                })?;
            let subband_end =
                subband_start
                    .checked_add(subband_count)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "Tier-2 Metal packet descriptor subband range overflow"
                            .to_string(),
                    })?;
            if subband_end > subbands.len() {
                return Err(Error::MetalKernel {
                    message: "Tier-2 Metal packet descriptor subband range out of bounds"
                        .to_string(),
                });
            }
            let mut packet_block_count = 0usize;
            for subband in &subbands[subband_start..subband_end] {
                packet_block_count = packet_block_count
                    .checked_add(usize::try_from(subband.block_count).map_err(|_| {
                        Error::MetalKernel {
                            message: "Tier-2 Metal packet descriptor block count exceeds usize"
                                .to_string(),
                        }
                    })?)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "Tier-2 Metal packet descriptor block count overflow".to_string(),
                    })?;
            }

            let (state_block_offset, existing_count) = if let Some(&(offset, count)) =
                state_block_offsets.get(&descriptor.state_index)
            {
                (offset, count)
            } else {
                let offset = u32::try_from(state_blocks.len()).map_err(|_| Error::MetalKernel {
                    message: "Tier-2 Metal packet state block offset exceeds u32".to_string(),
                })?;
                for subband in &subbands[subband_start..subband_end] {
                    let block_start =
                        usize::try_from(subband.block_offset).map_err(|_| Error::MetalKernel {
                            message: "Tier-2 Metal packet state block offset exceeds usize"
                                .to_string(),
                        })?;
                    let block_count =
                        usize::try_from(subband.block_count).map_err(|_| Error::MetalKernel {
                            message: "Tier-2 Metal packet state block count exceeds usize"
                                .to_string(),
                        })?;
                    let block_end =
                        block_start
                            .checked_add(block_count)
                            .ok_or_else(|| Error::MetalKernel {
                                message: "Tier-2 Metal packet state block range overflow"
                                    .to_string(),
                            })?;
                    if block_end > blocks.len() {
                        return Err(Error::MetalKernel {
                            message: "Tier-2 Metal packet state block range out of bounds"
                                .to_string(),
                        });
                    }
                    for block in &blocks[block_start..block_end] {
                        state_blocks.push(J2kPacketStateBlock {
                            previously_included: block.previously_included,
                            l_block: block.l_block,
                        });
                    }
                }
                state_block_offsets.insert(descriptor.state_index, (offset, packet_block_count));
                (offset, packet_block_count)
            };
            if existing_count != packet_block_count {
                return Err(Error::MetalKernel {
                    message: "Tier-2 Metal packet descriptor state layout mismatch".to_string(),
                });
            }

            descriptors.push(J2kPacketDescriptor {
                packet_index: descriptor.packet_index,
                state_index: descriptor.state_index,
                layer: u32::from(descriptor.layer),
                resolution: descriptor.resolution,
                component: u32::from(descriptor.component),
                precinct_lo: descriptor.precinct as u32,
                precinct_hi: (descriptor.precinct >> 32) as u32,
                state_block_offset,
            });
        }

        let header_capacity = blocks
            .len()
            .checked_mul(256)
            .and_then(|bytes| bytes.checked_add(4096))
            .map(|bytes| bytes.max(4096))
            .ok_or_else(|| Error::MetalKernel {
                message: "Tier-2 Metal packet header capacity overflow".to_string(),
            })?;
        let output_capacity = payload
            .len()
            .checked_add(
                header_capacity
                    .checked_mul(descriptors.len().max(resolutions.len()).max(1))
                    .ok_or_else(|| Error::MetalKernel {
                        message: "Tier-2 Metal packet output capacity overflow".to_string(),
                    })?,
            )
            .and_then(|bytes| bytes.checked_add(1024))
            .ok_or_else(|| Error::MetalKernel {
                message: "Tier-2 Metal packet output capacity overflow".to_string(),
            })?;

        let params = J2kPacketEncodeParams {
            resolution_count: u32::try_from(resolutions.len()).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal packet resolution count exceeds u32".to_string(),
            })?,
            num_layers: u32::from(job.num_layers),
            num_components: u32::from(job.num_components),
            code_block_count: u32::try_from(blocks.len()).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal packet code-block count exceeds u32".to_string(),
            })?,
            subband_count: u32::try_from(subbands.len()).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal packet subband count exceeds u32".to_string(),
            })?,
            descriptor_count: u32::try_from(descriptors.len()).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal packet descriptor count exceeds u32".to_string(),
            })?,
            output_capacity: u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal packet output capacity exceeds u32".to_string(),
            })?,
            header_capacity: u32::try_from(header_capacity).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal packet header capacity exceeds u32".to_string(),
            })?,
            scratch_node_capacity: u32::try_from(max_tree_nodes).map_err(|_| {
                Error::MetalKernel {
                    message: "Tier-2 Metal packet scratch node capacity exceeds u32".to_string(),
                }
            })?,
        };

        let resolution_buffer = copied_slice_buffer(&runtime.device, &resolutions);
        let subband_buffer = copied_slice_buffer(&runtime.device, &subbands);
        let block_buffer = copied_slice_buffer(&runtime.device, &blocks);
        let payload_buffer = copied_slice_buffer(&runtime.device, &payload);
        let descriptor_buffer = copied_slice_buffer(&runtime.device, &descriptors);
        let state_block_buffer = copied_slice_buffer(&runtime.device, &state_blocks);
        let output_buffer = runtime.device.new_buffer(
            output_capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let header_buffer = runtime.device.new_buffer(
            header_capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let scratch_words = max_tree_nodes
            .checked_mul(6)
            .ok_or_else(|| Error::MetalKernel {
                message: "Tier-2 Metal packet scratch size overflow".to_string(),
            })?;
        let scratch_buffer = runtime.device.new_buffer(
            (scratch_words * size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status_buffer = runtime.device.new_buffer(
            size_of::<J2kPacketEncodeStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.packet_encode);
        encoder.set_buffer(0, Some(&resolution_buffer), 0);
        encoder.set_buffer(1, Some(&subband_buffer), 0);
        encoder.set_buffer(2, Some(&block_buffer), 0);
        encoder.set_buffer(3, Some(&payload_buffer), 0);
        encoder.set_buffer(4, Some(&output_buffer), 0);
        encoder.set_buffer(5, Some(&header_buffer), 0);
        encoder.set_buffer(6, Some(&scratch_buffer), 0);
        encoder.set_bytes(
            7,
            size_of::<J2kPacketEncodeParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(8, Some(&status_buffer), 0);
        encoder.set_buffer(9, Some(&descriptor_buffer), 0);
        encoder.set_buffer(10, Some(&state_block_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = unsafe {
            status_buffer
                .contents()
                .cast::<J2kPacketEncodeStatus>()
                .read()
        };
        if status.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                "Tier-2 packetization",
                status.code,
                status.detail,
            ));
        }
        let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
            message: "Tier-2 Metal packet output length exceeds usize".to_string(),
        })?;
        if data_len > output_capacity {
            return Err(Error::MetalKernel {
                message: "Tier-2 Metal packet output length exceeds buffer".to_string(),
            });
        }
        Ok(if data_len == 0 {
            Vec::new()
        } else {
            unsafe { core::slice::from_raw_parts(output_buffer.contents().cast::<u8>(), data_len) }
                .to_vec()
        })
    })
}

#[cfg(target_os = "macos")]
fn dispatch_ht_cleanup(
    runtime: &MetalRuntime,
    coded_data: &[u8],
    params: J2kHtCleanupParams,
    decoded: &Buffer,
) -> Result<(), Error> {
    let input = borrow_slice_buffer(&runtime.device, coded_data);
    let status_buffer = runtime.device.new_buffer(
        size_of::<J2kHtStatus>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = runtime.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    dispatch_zero_u32_buffer_in_encoder(
        runtime,
        encoder,
        decoded,
        ht_output_word_count(
            params.output_offset,
            params.output_stride,
            params.width,
            params.height,
        )?,
    )?;
    encoder.set_compute_pipeline_state(&runtime.ht_cleanup);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kHtCleanupParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(3, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(5, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(7, Some(&status_buffer), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let status = unsafe { status_buffer.contents().cast::<J2kHtStatus>().read() };
    if status.code != J2K_HT_STATUS_OK {
        return Err(decode_ht_status_error(status));
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn dispatch_ht_cleanup_batched(
    runtime: &MetalRuntime,
    coded_data: &[u8],
    jobs: &[J2kHtCleanupBatchJob],
    decoded: &Buffer,
) -> Result<(), Error> {
    let input = borrow_slice_buffer(&runtime.device, coded_data);
    let jobs_buffer = borrow_slice_buffer(&runtime.device, jobs);
    let status_buffer = runtime.device.new_buffer(
        (jobs.len().max(1) * size_of::<J2kHtStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = runtime.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    dispatch_zero_u32_buffer_in_encoder(
        runtime,
        encoder,
        decoded,
        ht_batch_output_word_count(jobs)?,
    )?;
    encoder.set_compute_pipeline_state(&runtime.ht_cleanup_batched);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(&jobs_buffer), 0);
    encoder.set_buffer(3, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(5, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(7, Some(&status_buffer), 0);
    let width = runtime
        .ht_cleanup_batched
        .thread_execution_width()
        .max(1)
        .min(jobs.len() as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: jobs.len() as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let statuses = unsafe {
        core::slice::from_raw_parts(status_buffer.contents().cast::<J2kHtStatus>(), jobs.len())
    };
    if let Some(status) = statuses
        .iter()
        .copied()
        .find(|status| status.code != J2K_HT_STATUS_OK)
    {
        return Err(decode_ht_status_error(status));
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn dispatch_ht_cleanup_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    decoded: &Buffer,
    decoded_word_count: usize,
) -> Result<DirectStatusCheck, Error> {
    let status_buffer = runtime.device.new_buffer(
        (job_count.max(1) * size_of::<J2kHtStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let encoder = command_buffer.new_compute_command_encoder();
    dispatch_zero_u32_buffer_in_encoder(runtime, encoder, decoded, decoded_word_count)?;
    dispatch_ht_cleanup_batched_in_encoder_with_status(
        runtime,
        encoder,
        coded_data,
        jobs,
        job_count,
        decoded,
        &status_buffer,
    );
    encoder.end_encoding();

    Ok(DirectStatusCheck::Ht {
        buffer: status_buffer,
        len: job_count,
    })
}

#[cfg(target_os = "macos")]
fn dispatch_ht_cleanup_batched_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    decoded: &Buffer,
    decoded_word_count: usize,
) -> Result<DirectStatusCheck, Error> {
    let status_buffer = runtime.device.new_buffer(
        (job_count.max(1) * size_of::<J2kHtStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    dispatch_zero_u32_buffer_in_encoder(runtime, encoder, decoded, decoded_word_count)?;
    dispatch_ht_cleanup_batched_in_encoder_with_status(
        runtime,
        encoder,
        coded_data,
        jobs,
        job_count,
        decoded,
        &status_buffer,
    );

    Ok(DirectStatusCheck::Ht {
        buffer: status_buffer,
        len: job_count,
    })
}

#[cfg(target_os = "macos")]
fn dispatch_ht_cleanup_batched_in_encoder_with_status(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    decoded: &Buffer,
    status_buffer: &Buffer,
) {
    encoder.set_compute_pipeline_state(&runtime.ht_cleanup_batched);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(5, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(7, Some(status_buffer), 0);
    let width = runtime
        .ht_cleanup_batched
        .thread_execution_width()
        .max(1)
        .min(job_count as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: job_count as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_ht_cleanup_repeated_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    base_job_count: usize,
    total_job_count: usize,
    output_plane_len: usize,
    decoded: &Buffer,
) -> Result<DirectStatusCheck, Error> {
    let status_buffer = runtime.device.new_buffer(
        (total_job_count.max(1) * size_of::<J2kHtStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let batch_count =
        total_job_count
            .checked_div(base_job_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K MetalDirect repeated base job count is zero".to_string(),
            })?;
    let decoded_word_count =
        output_plane_len
            .checked_mul(batch_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K MetalDirect repeated output span overflow".to_string(),
            })?;
    let repeated = J2kHtRepeatedBatchParams {
        job_count: u32::try_from(base_job_count).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect repeated base job count exceeds u32".to_string(),
        })?,
        output_plane_len: u32::try_from(output_plane_len).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect repeated output plane length exceeds u32".to_string(),
        })?,
        batch_count: u32::try_from(batch_count).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect repeated batch count exceeds u32".to_string(),
        })?,
    };

    let encoder = command_buffer.new_compute_command_encoder();
    dispatch_zero_u32_buffer_in_encoder(runtime, encoder, decoded, decoded_word_count)?;
    encoder.set_compute_pipeline_state(&runtime.ht_cleanup_repeated_batched);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_bytes(
        3,
        size_of::<J2kHtRepeatedBatchParams>() as u64,
        (&raw const repeated).cast(),
    );
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(5, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(7, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(8, Some(&status_buffer), 0);
    let width = runtime
        .ht_cleanup_repeated_batched
        .thread_execution_width()
        .max(1)
        .min(base_job_count as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: base_job_count as u64,
            height: u64::from(repeated.batch_count),
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    Ok(DirectStatusCheck::Ht {
        buffer: status_buffer,
        len: total_job_count,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_classic_cleanup_code_block(
    job: J2kCodeBlockDecodeJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    let required_len = required_classic_output_len(job)?;
    if output.len() < required_len {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal output slice is too small".to_string(),
        });
    }

    if job.width == 0 || job.height == 0 {
        return Ok(());
    }

    with_runtime(|runtime| {
        let decoded = wrap_f32_output_buffer(&runtime.device, output);
        let batch_job = J2kClassicCleanupBatchJob {
            coded_offset: 0,
            coded_len: u32::try_from(job.data.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal coded payload exceeds u32".to_string(),
            })?,
            segment_offset: 0,
            segment_count: u32::try_from(job.segments.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal segment count exceeds u32".to_string(),
            })?,
            width: job.width,
            height: job.height,
            output_stride: u32::try_from(job.output_stride).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal output stride exceeds u32".to_string(),
            })?,
            output_offset: 0,
            missing_msbs: u32::from(job.missing_bit_planes),
            total_bitplanes: u32::from(job.total_bitplanes),
            number_of_coding_passes: u32::from(job.number_of_coding_passes),
            sub_band_type: match job.sub_band_type {
                signinum_j2k_native::J2kSubBandType::LowLow => 0,
                signinum_j2k_native::J2kSubBandType::HighLow => 1,
                signinum_j2k_native::J2kSubBandType::LowHigh => 2,
                signinum_j2k_native::J2kSubBandType::HighHigh => 3,
            },
            style_flags: classic_style_flags(job.style),
            strict: u32::from(job.strict),
            dequantization_step: job.dequantization_step,
        };
        let segments: Vec<_> = job
            .segments
            .iter()
            .map(|segment| J2kClassicSegment {
                data_offset: segment.data_offset,
                data_length: segment.data_length,
                start_coding_pass: u32::from(segment.start_coding_pass),
                end_coding_pass: u32::from(segment.end_coding_pass),
                use_arithmetic: u32::from(segment.use_arithmetic),
            })
            .collect();
        dispatch_classic_cleanup_batched(runtime, job.data, &[batch_job], &segments, &decoded)?;
        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_classic_cleanup_sub_band(
    job: J2kSubBandDecodeJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    let required_len = (job.width as usize)
        .checked_mul(job.height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal sub-band size overflow".to_string(),
        })?;
    if output.len() < required_len {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal sub-band output slice is too small".to_string(),
        });
    }
    if job.jobs.is_empty() {
        return Ok(());
    }

    with_runtime(|runtime| {
        let decoded = wrap_f32_output_buffer(&runtime.device, output);

        let mut jobs = Vec::with_capacity(job.jobs.len());
        let mut coded_data = Vec::new();
        let mut segments = Vec::new();

        for block in job.jobs {
            let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal batched coded payload exceeds u32".to_string(),
            })?;
            coded_data.extend_from_slice(block.code_block.data);
            let segment_offset = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal segment table exceeds u32".to_string(),
            })?;
            let end_x = block
                .output_x
                .checked_add(block.code_block.width)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal batched block width overflow".to_string(),
                })?;
            let end_y = block
                .output_y
                .checked_add(block.code_block.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal batched block height overflow".to_string(),
                })?;
            if end_x > job.width || end_y > job.height {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal batched block lies outside sub-band bounds"
                        .to_string(),
                });
            }
            for segment in block.code_block.segments {
                let data_offset =
                    coded_offset
                        .checked_add(segment.data_offset)
                        .ok_or_else(|| Error::MetalKernel {
                            message: "classic J2K Metal segment offset overflow".to_string(),
                        })?;
                segments.push(J2kClassicSegment {
                    data_offset,
                    data_length: segment.data_length,
                    start_coding_pass: u32::from(segment.start_coding_pass),
                    end_coding_pass: u32::from(segment.end_coding_pass),
                    use_arithmetic: u32::from(segment.use_arithmetic),
                });
            }
            jobs.push(J2kClassicCleanupBatchJob {
                coded_offset,
                coded_len: u32::try_from(block.code_block.data.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal coded payload exceeds u32".to_string(),
                    }
                })?,
                segment_offset,
                segment_count: u32::try_from(block.code_block.segments.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal segment count exceeds u32".to_string(),
                    }
                })?,
                width: block.code_block.width,
                height: block.code_block.height,
                output_stride: job.width,
                output_offset: block
                    .output_y
                    .checked_mul(job.width)
                    .and_then(|row| row.checked_add(block.output_x))
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K Metal output offset overflow".to_string(),
                    })?,
                missing_msbs: u32::from(block.code_block.missing_bit_planes),
                total_bitplanes: u32::from(block.code_block.total_bitplanes),
                number_of_coding_passes: u32::from(block.code_block.number_of_coding_passes),
                sub_band_type: match block.code_block.sub_band_type {
                    signinum_j2k_native::J2kSubBandType::LowLow => 0,
                    signinum_j2k_native::J2kSubBandType::HighLow => 1,
                    signinum_j2k_native::J2kSubBandType::LowHigh => 2,
                    signinum_j2k_native::J2kSubBandType::HighHigh => 3,
                },
                style_flags: classic_style_flags(block.code_block.style),
                strict: u32::from(block.code_block.strict),
                dequantization_step: block.code_block.dequantization_step,
            });
        }

        dispatch_classic_cleanup_batched(runtime, &coded_data, &jobs, &segments, &decoded)?;
        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_ht_cleanup_code_block(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    let required_len = required_ht_output_len(job)?;
    if output.len() < required_len {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal output slice is too small".to_string(),
        });
    }

    if job.width == 0 || job.height == 0 {
        return Ok(());
    }

    with_runtime(|runtime| {
        let params = J2kHtCleanupParams {
            width: job.width,
            height: job.height,
            coded_len: u32::try_from(job.data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal coded payload exceeds u32".to_string(),
            })?,
            cleanup_length: job.cleanup_length,
            refinement_length: job.refinement_length,
            missing_msbs: u32::from(job.missing_bit_planes),
            num_bitplanes: u32::from(job.num_bitplanes),
            number_of_coding_passes: u32::from(job.number_of_coding_passes),
            output_stride: u32::try_from(job.output_stride).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal output stride exceeds u32".to_string(),
            })?,
            output_offset: 0,
            dequantization_step: job.dequantization_step,
            stripe_causal: u32::from(job.stripe_causal),
        };
        let decoded = wrap_f32_output_buffer(&runtime.device, output);
        dispatch_ht_cleanup(runtime, job.data, params, &decoded)?;

        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_ht_cleanup_sub_band(
    job: HtSubBandDecodeJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    let required_len = (job.width as usize)
        .checked_mul(job.height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal sub-band size overflow".to_string(),
        })?;
    if output.len() < required_len {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal sub-band output slice is too small".to_string(),
        });
    }

    if job.jobs.is_empty() {
        return Ok(());
    }

    with_runtime(|runtime| {
        let decoded = wrap_f32_output_buffer(&runtime.device, output);

        let mut jobs = Vec::with_capacity(job.jobs.len());
        let mut coded_data = Vec::new();

        for block in job.jobs {
            let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal batched coded payload exceeds u32".to_string(),
            })?;
            coded_data.extend_from_slice(block.code_block.data);

            jobs.push(J2kHtCleanupBatchJob {
                coded_offset,
                width: block.code_block.width,
                height: block.code_block.height,
                coded_len: u32::try_from(block.code_block.data.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Metal coded payload exceeds u32".to_string(),
                    }
                })?,
                cleanup_length: block.code_block.cleanup_length,
                refinement_length: block.code_block.refinement_length,
                missing_msbs: u32::from(block.code_block.missing_bit_planes),
                num_bitplanes: u32::from(block.code_block.num_bitplanes),
                number_of_coding_passes: u32::from(block.code_block.number_of_coding_passes),
                output_stride: job.width,
                output_offset: block
                    .output_y
                    .checked_mul(job.width)
                    .and_then(|row| row.checked_add(block.output_x))
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Metal output offset overflow".to_string(),
                    })?,
                dequantization_step: block.code_block.dequantization_step,
                stripe_causal: u32::from(block.code_block.stripe_causal),
            });

            let end_x = block
                .output_x
                .checked_add(block.code_block.width)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batched block width overflow".to_string(),
                })?;
            let end_y = block
                .output_y
                .checked_add(block.code_block.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batched block height overflow".to_string(),
                })?;
            if end_x > job.width || end_y > job.height {
                return Err(Error::MetalKernel {
                    message: "HTJ2K Metal batched block lies outside sub-band bounds".to_string(),
                });
            }
        }

        dispatch_ht_cleanup_batched(runtime, &coded_data, &jobs, &decoded)?;
        Ok(())
    })
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::{
        classic_batch_uses_plain_fast_path, classic_repeated_uses_plain_fast_path,
        decode_scaled_to_surface, j2k_pack_kernel_name_for, j2k_pack_scale_arrays,
        output_shape_for, prepare_direct_grayscale_plan,
        prepared_direct_grayscale_plan_compute_encoder_count,
        prepared_repeated_direct_ht_cleanup_dispatch_count,
        repeated_gray_store_is_contiguous_full_surface, runtime_initialization_error,
        J2kClassicCleanupBatchJob, J2kClassicSegment, J2kRepeatedGrayStoreParams, MetalRuntime,
        PreparedDirectGrayscaleStep,
    };
    use signinum_core::PixelFormat;
    use signinum_j2k_native::{
        encode, encode_htj2k, ColorSpace as NativeColorSpace, DecodeSettings, DecoderContext,
        EncodeOptions, Image,
    };

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

    #[test]
    fn runtime_initialization_error_classifies_null_queue_as_unavailable() {
        assert!(matches!(
            runtime_initialization_error("Metal command queue is unavailable on this host"),
            crate::Error::MetalUnavailable
        ));
    }

    #[test]
    fn j2k_pack_selects_specialized_kernels_for_wsi_formats() {
        assert_eq!(
            j2k_pack_kernel_name_for(&NativeColorSpace::Gray, false, 1, PixelFormat::Gray8),
            Some("j2k_pack_gray8")
        );
        assert_eq!(
            j2k_pack_kernel_name_for(&NativeColorSpace::RGB, false, 3, PixelFormat::Rgb8),
            Some("j2k_pack_rgb8")
        );
        assert_eq!(
            j2k_pack_kernel_name_for(&NativeColorSpace::RGB, false, 3, PixelFormat::Rgba8),
            Some("j2k_pack_rgb_opaque_rgba8")
        );
        assert_eq!(
            j2k_pack_kernel_name_for(&NativeColorSpace::RGB, true, 4, PixelFormat::Rgba8),
            Some("j2k_pack_rgba8")
        );
        assert_eq!(
            j2k_pack_kernel_name_for(&NativeColorSpace::Gray, false, 1, PixelFormat::Gray16),
            Some("j2k_pack_gray16")
        );
        assert_eq!(
            j2k_pack_kernel_name_for(&NativeColorSpace::RGB, false, 3, PixelFormat::Rgb16),
            Some("j2k_pack_rgb16")
        );
        assert_eq!(
            j2k_pack_kernel_name_for(&NativeColorSpace::RGB, true, 4, PixelFormat::Rgb16),
            None,
            "RGBA input must not silently drop alpha when packing RGB16"
        );
    }

    #[test]
    fn j2k_pack_precomputes_scale_factors_on_cpu() {
        let (max_values, u8_scales, u16_scales) = j2k_pack_scale_arrays([8, 12, 16, 0]);

        assert_f32_near(max_values[0], 255.0);
        assert_f32_near(max_values[1], 4095.0);
        assert_f32_near(max_values[2], 65_535.0);
        assert_f32_near(max_values[3], 1.0);
        assert_f32_near(u8_scales[0], 1.0);
        assert_f32_near(u8_scales[1], 255.0 / 4095.0);
        assert_f32_near(u16_scales[0], 257.0);
        assert_f32_near(u16_scales[1], 1.0);
        assert_f32_near(u16_scales[2], 1.0);
        assert_f32_near(u16_scales[3], 65_535.0);
    }

    fn assert_f32_near(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= f32::EPSILON,
            "expected {actual} to be within f32 epsilon of {expected}"
        );
    }

    #[test]
    fn scaled_htj2k_decode_runs_through_metal_compute_path() {
        let pixels: Vec<u8> = (0..16).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let bytes = encode_htj2k(&pixels, 4, 4, 1, 8, false, &options).expect("encode ht gray8");

        let image = Image::new(
            &bytes,
            &DecodeSettings {
                target_resolution: Some((2, 2)),
                ..DecodeSettings::default()
            },
        )
        .expect("image");
        let host = image.decode().expect("host scaled decode");

        let surface = decode_scaled_to_surface(
            &bytes,
            (4, 4),
            PixelFormat::Gray8,
            signinum_core::Downscale::Half,
        )
        .expect("metal scaled decode");
        assert_eq!(surface.as_bytes(), host.as_slice());
    }

    #[test]
    fn prepared_ht_direct_plan_groups_cleanup_subbands_before_idwt() {
        let pixels: Vec<u8> = (0..64).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let bytes = encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode ht gray8");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("direct grayscale plan");
        let ht_subband_steps = plan
            .steps
            .iter()
            .filter(|step| {
                matches!(
                    step,
                    signinum_j2k_native::J2kDirectGrayscaleStep::HtSubBand(_)
                )
            })
            .count();
        assert!(
            ht_subband_steps > 1,
            "fixture must exercise multiple HT sub-band cleanup steps"
        );

        let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
        assert_eq!(
            prepared.ht_groups.len(),
            1,
            "single-tile HTJ2K direct decode should group adjacent HT sub-bands into one cleanup dispatch"
        );
        assert_eq!(prepared.ht_groups[0].members.len(), ht_subband_steps);
        assert!(matches!(
            prepared.steps[prepared.ht_groups[0].start_step],
            PreparedDirectGrayscaleStep::HtSubBand(_)
        ));
    }

    #[test]
    fn prepared_ht_direct_plan_encodes_full_decode_in_one_compute_encoder() {
        let pixels: Vec<u8> = (0..64).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let bytes = encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode ht gray8");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("direct grayscale plan");
        let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");

        assert_eq!(
            prepared_direct_grayscale_plan_compute_encoder_count(&prepared, PixelFormat::Gray8),
            1,
            "prepared single-tile direct decode should keep cleanup, IDWT, and grayscale store in one compute encoder"
        );
    }

    #[test]
    fn repeated_prepared_ht_direct_plan_groups_cleanup_subbands_before_idwt() {
        let pixels: Vec<u8> = (0..64).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let bytes = encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode ht gray8");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("direct grayscale plan");
        let ht_subband_steps = plan
            .steps
            .iter()
            .filter(|step| {
                matches!(
                    step,
                    signinum_j2k_native::J2kDirectGrayscaleStep::HtSubBand(_)
                )
            })
            .count();
        assert!(
            ht_subband_steps > 1,
            "fixture must exercise multiple HT sub-band cleanup steps"
        );

        let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
        assert_eq!(
            prepared_repeated_direct_ht_cleanup_dispatch_count(&prepared),
            1,
            "repeated HTJ2K WSI tile batches should group adjacent sub-band cleanups like the single-tile path"
        );
    }

    #[test]
    fn prepared_classic_direct_plan_groups_cleanup_subbands_before_idwt() {
        let pixels: Vec<u8> = (0..64).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let bytes = encode(&pixels, 8, 8, 1, 8, false, &options).expect("encode j2k gray8");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("direct grayscale plan");
        let classic_subband_steps = plan
            .steps
            .iter()
            .filter(|step| {
                matches!(
                    step,
                    signinum_j2k_native::J2kDirectGrayscaleStep::ClassicSubBand(_)
                )
            })
            .count();
        assert!(
            classic_subband_steps > 1,
            "fixture must exercise multiple classic sub-band cleanup steps"
        );

        let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
        assert_eq!(
            prepared.classic_groups.len(),
            1,
            "classic J2K direct decode should group adjacent sub-band cleanups before IDWT"
        );
        assert_eq!(
            prepared.classic_groups[0].members.len(),
            classic_subband_steps
        );
        assert!(matches!(
            prepared.steps[prepared.classic_groups[0].start_step],
            PreparedDirectGrayscaleStep::ClassicSubBand(_)
        ));
    }

    #[test]
    fn classic_plain_fast_path_accepts_style_zero_arithmetic_jobs() {
        let jobs = [J2kClassicCleanupBatchJob {
            coded_offset: 0,
            coded_len: 1,
            segment_offset: 0,
            segment_count: 1,
            width: 64,
            height: 64,
            output_stride: 64,
            output_offset: 0,
            missing_msbs: 0,
            total_bitplanes: 8,
            number_of_coding_passes: 1,
            sub_band_type: 0,
            style_flags: 0,
            strict: 1,
            dequantization_step: 1.0,
        }];
        let segments = [J2kClassicSegment {
            data_offset: 0,
            data_length: 1,
            start_coding_pass: 0,
            end_coding_pass: 1,
            use_arithmetic: 1,
        }];

        assert!(
            classic_batch_uses_plain_fast_path(&jobs, &segments),
            "style-0 arithmetic-only classic J2K jobs should use the fused plain cleanup/store kernel"
        );
    }

    #[test]
    fn classic_repeated_plain_fast_path_stays_off_for_wsi_batch_size() {
        let jobs = [J2kClassicCleanupBatchJob {
            coded_offset: 0,
            coded_len: 1,
            segment_offset: 0,
            segment_count: 1,
            width: 64,
            height: 64,
            output_stride: 64,
            output_offset: 0,
            missing_msbs: 0,
            total_bitplanes: 8,
            number_of_coding_passes: 1,
            sub_band_type: 0,
            style_flags: 0,
            strict: 1,
            dequantization_step: 1.0,
        }];
        let segments = [J2kClassicSegment {
            data_offset: 0,
            data_length: 1,
            start_coding_pass: 0,
            end_coding_pass: 1,
            use_arithmetic: 1,
        }];

        assert!(
            !classic_repeated_uses_plain_fast_path(16, &jobs, &segments),
            "batch-16 WSI classic J2K should keep the device-state cleanup plus separate store path"
        );
    }

    #[test]
    fn repeated_gray_store_detects_contiguous_full_wsi_tiles() {
        let full_tile = J2kRepeatedGrayStoreParams {
            input_width: 1024,
            input_height: 1024,
            source_x: 0,
            source_y: 0,
            copy_width: 1024,
            copy_height: 1024,
            output_width: 1024,
            output_height: 1024,
            output_x: 0,
            output_y: 0,
            addend: 0.0,
            batch_count: 16,
            max_value: 255.0,
            u8_scale: 1.0,
            u16_scale: 257.0,
        };
        assert!(
            repeated_gray_store_is_contiguous_full_surface(full_tile),
            "full repeated grayscale WSI stores should use the contiguous store kernel"
        );

        let windowed = J2kRepeatedGrayStoreParams {
            source_x: 1,
            copy_width: 1023,
            ..full_tile
        };
        assert!(
            !repeated_gray_store_is_contiguous_full_surface(windowed),
            "ROI/windowed repeated grayscale stores must stay on the generic store kernel"
        );
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_image_to_surface<'a>(
    image: &NativeImage<'a>,
    context: &mut NativeDecoderContext<'a>,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let mut code_block_decoder = MetalCodeBlockDecoder::default();
        let decoded = image
            .decode_components_with_ht_decoder(context, &mut code_block_decoder)
            .map_err(|error| Error::Decode(signinum_j2k::J2kError::Backend(error.to_string())))?;
        let stage = select_plane_stage(runtime, image, &decoded, &mut code_block_decoder)?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_image_to_surface_with_device<'a>(
    image: &NativeImage<'a>,
    context: &mut NativeDecoderContext<'a>,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| decode_image_to_surface(image, context, fmt))
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_image_region_to_surface<'a>(
    image: &NativeImage<'a>,
    context: &mut NativeDecoderContext<'a>,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let mut code_block_decoder = MetalCodeBlockDecoder::default();
        let decoded = image
            .decode_region_components_with_ht_decoder(
                context,
                (roi.x, roi.y, roi.w, roi.h),
                &mut code_block_decoder,
            )
            .map_err(|error| Error::Decode(signinum_j2k::J2kError::Backend(error.to_string())))?;
        let stage = select_plane_stage(runtime, image, &decoded, &mut code_block_decoder)?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_image_region_to_surface_with_device<'a>(
    image: &NativeImage<'a>,
    context: &mut NativeDecoderContext<'a>,
    fmt: PixelFormat,
    roi: Rect,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        decode_image_region_to_surface(image, context, fmt, roi)
    })
}

#[cfg(target_os = "macos")]
fn select_plane_stage(
    runtime: &MetalRuntime,
    image: &NativeImage<'_>,
    decoded: &NativeDecodedComponents<'_>,
    code_block_decoder: &mut MetalCodeBlockDecoder,
) -> Result<PlaneStage, Error> {
    if image.supports_direct_device_plane_reuse() {
        if matches!(decoded.color_space(), NativeColorSpace::RGB)
            && !decoded.has_alpha()
            && decoded.planes().len() == 3
        {
            if let Some(stage) = PlaneStage::from_captured_planes(
                decoded,
                code_block_decoder.mct.take_captured_planes(),
            ) {
                return Ok(stage);
            }
        }
        if matches!(decoded.color_space(), NativeColorSpace::Gray)
            && !decoded.has_alpha()
            && decoded.planes().len() == 1
        {
            if let Some(stage) = PlaneStage::from_captured_planes(
                decoded,
                code_block_decoder.store.take_captured_planes(),
            ) {
                return Ok(stage);
            }
        }
    }

    PlaneStage::from_planes(&runtime.device, decoded, None)
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_scaled_to_surface(
    bytes: &[u8],
    dims: (u32, u32),
    fmt: PixelFormat,
    scale: signinum_core::Downscale,
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
        .map_err(|error| Error::Decode(signinum_j2k::J2kError::Backend(error.to_string())))?;
    let mut context = NativeDecoderContext::default();
    decode_image_to_surface(&image, &mut context, fmt)
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_scaled_to_surface(
    bytes: &[u8],
    dims: (u32, u32),
    fmt: PixelFormat,
    roi: signinum_core::Rect,
    scale: signinum_core::Downscale,
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
        .map_err(|error| Error::Decode(signinum_j2k::J2kError::Backend(error.to_string())))?;
    let mut context = NativeDecoderContext::default();
    decode_image_region_to_surface(&image, &mut context, fmt, roi.scaled_covering(scale))
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_scaled_to_surface_with_device(
    bytes: &[u8],
    dims: (u32, u32),
    fmt: PixelFormat,
    scale: signinum_core::Downscale,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        decode_scaled_to_surface(bytes, dims, fmt, scale)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_scaled_to_surface_with_device(
    bytes: &[u8],
    dims: (u32, u32),
    fmt: PixelFormat,
    roi: signinum_core::Rect,
    scale: signinum_core::Downscale,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        decode_region_scaled_to_surface(bytes, dims, fmt, roi, scale)
    })
}
