// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

struct J2kStoreParams {
    uint input_width;
    uint source_x;
    uint source_y;
    uint copy_width;
    uint copy_height;
    uint output_width;
    uint output_x;
    uint output_y;
    float addend;
};

struct J2kRepeatedStoreParams {
    uint input_width;
    uint input_height;
    uint source_x;
    uint source_y;
    uint copy_width;
    uint copy_height;
    uint output_width;
    uint output_height;
    uint output_x;
    uint output_y;
    float addend;
    uint batch_count;
};

struct J2kRepeatedGrayStoreParams {
    uint input_width;
    uint input_height;
    uint source_x;
    uint source_y;
    uint copy_width;
    uint copy_height;
    uint output_width;
    uint output_height;
    uint output_x;
    uint output_y;
    float addend;
    uint batch_count;
    float max_value;
    float u8_scale;
    float u16_scale;
};

struct J2kGrayStoreParams {
    uint input_width;
    uint source_x;
    uint source_y;
    uint copy_width;
    uint copy_height;
    uint output_width;
    uint output_x;
    uint output_y;
    float addend;
    float max_value;
    float u8_scale;
    float u16_scale;
};

kernel void j2k_store_component(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant J2kStoreParams &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.copy_width || gid.y >= params.copy_height) {
        return;
    }

    const uint src_x = params.source_x + gid.x;
    const uint src_y = params.source_y + gid.y;
    const uint dst_x = params.output_x + gid.x;
    const uint dst_y = params.output_y + gid.y;

    const uint src_idx = src_y * params.input_width + src_x;
    const uint dst_idx = dst_y * params.output_width + dst_x;
    output[dst_idx] = input[src_idx] + params.addend;
}

kernel void j2k_store_component_repeated(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant J2kRepeatedStoreParams &params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.copy_width || gid.y >= params.copy_height || gid.z >= params.batch_count) {
        return;
    }

    const uint input_plane_len = params.input_width * params.input_height;
    const uint output_plane_len = params.output_width * params.output_height;
    const uint src_x = params.source_x + gid.x;
    const uint src_y = params.source_y + gid.y;
    const uint dst_x = params.output_x + gid.x;
    const uint dst_y = params.output_y + gid.y;

    const uint src_idx = gid.z * input_plane_len + src_y * params.input_width + src_x;
    const uint dst_idx = gid.z * output_plane_len + dst_y * params.output_width + dst_x;
    output[dst_idx] = input[src_idx] + params.addend;
}

kernel void j2k_store_component_repeated_gray_u8(
    device const float *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant J2kRepeatedGrayStoreParams &params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.copy_width || gid.y >= params.copy_height || gid.z >= params.batch_count) {
        return;
    }

    const uint input_plane_len = params.input_width * params.input_height;
    const uint output_plane_len = params.output_width * params.output_height;
    const uint src_x = params.source_x + gid.x;
    const uint src_y = params.source_y + gid.y;
    const uint dst_x = params.output_x + gid.x;
    const uint dst_y = params.output_y + gid.y;

    const uint src_idx = gid.z * input_plane_len + src_y * params.input_width + src_x;
    const uint dst_idx = gid.z * output_plane_len + dst_y * params.output_width + dst_x;
    output[dst_idx] = scale_to_u8(input[src_idx] + params.addend, params.max_value, params.u8_scale);
}

kernel void j2k_store_component_repeated_gray_u16(
    device const float *input [[buffer(0)]],
    device ushort *output [[buffer(1)]],
    constant J2kRepeatedGrayStoreParams &params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.copy_width || gid.y >= params.copy_height || gid.z >= params.batch_count) {
        return;
    }

    const uint input_plane_len = params.input_width * params.input_height;
    const uint output_plane_len = params.output_width * params.output_height;
    const uint src_x = params.source_x + gid.x;
    const uint src_y = params.source_y + gid.y;
    const uint dst_x = params.output_x + gid.x;
    const uint dst_y = params.output_y + gid.y;

    const uint src_idx = gid.z * input_plane_len + src_y * params.input_width + src_x;
    const uint dst_idx = gid.z * output_plane_len + dst_y * params.output_width + dst_x;
    output[dst_idx] = pack_to_u16(input[src_idx] + params.addend, params.max_value, params.u16_scale);
}

kernel void j2k_store_component_repeated_gray_u8_contiguous(
    device const float *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant J2kRepeatedGrayStoreParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint plane_len = params.input_width * params.input_height;
    const uint total_len = plane_len * params.batch_count;
    if (gid >= total_len) {
        return;
    }

    output[gid] = scale_to_u8(input[gid] + params.addend, params.max_value, params.u8_scale);
}

kernel void j2k_store_component_repeated_gray_u16_contiguous(
    device const float *input [[buffer(0)]],
    device ushort *output [[buffer(1)]],
    constant J2kRepeatedGrayStoreParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint plane_len = params.input_width * params.input_height;
    const uint total_len = plane_len * params.batch_count;
    if (gid >= total_len) {
        return;
    }

    output[gid] = pack_to_u16(input[gid] + params.addend, params.max_value, params.u16_scale);
}

kernel void j2k_store_component_gray_u8(
    device const float *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant J2kGrayStoreParams &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.copy_width || gid.y >= params.copy_height) {
        return;
    }

    const uint src_x = params.source_x + gid.x;
    const uint src_y = params.source_y + gid.y;
    const uint dst_x = params.output_x + gid.x;
    const uint dst_y = params.output_y + gid.y;

    const uint src_idx = src_y * params.input_width + src_x;
    const uint dst_idx = dst_y * params.output_width + dst_x;
    output[dst_idx] = scale_to_u8(input[src_idx] + params.addend, params.max_value, params.u8_scale);
}

kernel void j2k_store_component_gray_u16(
    device const float *input [[buffer(0)]],
    device ushort *output [[buffer(1)]],
    constant J2kGrayStoreParams &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.copy_width || gid.y >= params.copy_height) {
        return;
    }

    const uint src_x = params.source_x + gid.x;
    const uint src_y = params.source_y + gid.y;
    const uint dst_x = params.output_x + gid.x;
    const uint dst_y = params.output_y + gid.y;

    const uint src_idx = src_y * params.input_width + src_x;
    const uint dst_idx = dst_y * params.output_width + dst_x;
    output[dst_idx] = pack_to_u16(input[src_idx] + params.addend, params.max_value, params.u16_scale);
}
