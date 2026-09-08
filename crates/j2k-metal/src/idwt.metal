// SPDX-License-Identifier: MIT OR Apache-2.0

struct J2kIdwtSingleDecompositionParams {
    uint x0;
    uint y0;
    uint output_x;
    uint output_y;
    uint width;
    uint height;
    uint ll_x;
    uint ll_y;
    uint ll_width;
    uint ll_height;
    uint hl_x;
    uint hl_y;
    uint hl_width;
    uint hl_height;
    uint lh_x;
    uint lh_y;
    uint lh_width;
    uint lh_height;
    uint hh_x;
    uint hh_y;
    uint hh_width;
    uint hh_height;
};

struct J2kRepeatedIdwtSingleDecompositionParams {
    uint x0;
    uint y0;
    uint output_x;
    uint output_y;
    uint width;
    uint height;
    uint ll_x;
    uint ll_y;
    uint ll_width;
    uint ll_height;
    uint hl_x;
    uint hl_y;
    uint hl_width;
    uint hl_height;
    uint lh_x;
    uint lh_y;
    uint lh_width;
    uint lh_height;
    uint hh_x;
    uint hh_y;
    uint hh_width;
    uint hh_height;
    uint ll_instance_stride;
    uint hl_instance_stride;
    uint lh_instance_stride;
    uint hh_instance_stride;
    uint batch_count;
};

struct J2kIdwt97StepParams {
    float coefficient;
    uint parity;
    uint reserved0;
    uint reserved1;
};

inline uint ceil_div2_u32(uint value) {
    return (value + 1u) >> 1u;
}

inline uint low_index(uint coord, uint origin) {
    return ceil_div2_u32(coord) - ceil_div2_u32(origin);
}

inline uint high_index(uint coord, uint origin) {
    return (coord >> 1u) - (origin >> 1u);
}

inline uint periodic_symmetric_extension_left_u32(uint idx, uint offset) {
    return idx >= offset ? idx - offset : offset - idx;
}

inline uint periodic_symmetric_extension_right_u32(uint idx, uint offset, uint length) {
    const uint new_idx = idx + offset;
    if (new_idx >= length) {
        const uint overshoot = new_idx - length;
        return length - 2u - overshoot;
    }
    return new_idx;
}

inline float reversible53_predict(float s, float left, float right) {
    return s - floor((left + right) * 0.25f + 0.5f);
}

inline float reversible53_update(float s, float left, float right) {
    return s + floor((left + right) * 0.5f);
}

inline void irreversible97_horizontal_step(
    device float *row_ptr,
    uint width,
    uint first,
    float coefficient
) {
    if (first == 0u) {
        const uint left = periodic_symmetric_extension_left_u32(0u, 1u);
        const uint right = periodic_symmetric_extension_right_u32(0u, 1u, width);
        row_ptr[0] = fma(row_ptr[left] + row_ptr[right], coefficient, row_ptr[0]);
    }

    const uint middle_start = first == 0u ? 2u : 1u;
    for (uint x = middle_start; x + 1u < width; x += 2u) {
        row_ptr[x] = fma(row_ptr[x - 1u] + row_ptr[x + 1u], coefficient, row_ptr[x]);
    }

    if (width > 1u && ((width - 1u) & 1u) == first) {
        const uint x = width - 1u;
        const uint left = periodic_symmetric_extension_left_u32(x, 1u);
        const uint right = periodic_symmetric_extension_right_u32(x, 1u, width);
        row_ptr[x] = fma(row_ptr[left] + row_ptr[right], coefficient, row_ptr[x]);
    }
}

kernel void j2k_idwt_interleave(
    device const float *ll [[buffer(0)]],
    device const float *hl [[buffer(1)]],
    device const float *lh [[buffer(2)]],
    device const float *hh [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint global_x = params.x0 + params.output_x + gid.x;
    const uint global_y = params.y0 + params.output_y + gid.y;
    // JPEG 2000 assigns low-pass samples to even reference-grid
    // coordinates.  The decomposition origin only determines which band is
    // encountered first in this tile; it must not redefine that parity.
    const bool low_x = (global_x & 1u) == 0u;
    const bool low_y = (global_y & 1u) == 0u;
    const uint full_band_x = low_x ? low_index(global_x, params.x0) : high_index(global_x, params.x0);
    const uint full_band_y = low_y ? low_index(global_y, params.y0) : high_index(global_y, params.y0);
    const uint out_idx = gid.y * params.width + gid.x;

    if (low_y && low_x) {
        const uint band_x = full_band_x - params.ll_x;
        const uint band_y = full_band_y - params.ll_y;
        out[out_idx] = (band_x < params.ll_width && band_y < params.ll_height)
            ? ll[band_y * params.ll_width + band_x]
            : 0.0f;
    } else if (low_y) {
        const uint band_x = full_band_x - params.hl_x;
        const uint band_y = full_band_y - params.hl_y;
        out[out_idx] = (band_x < params.hl_width && band_y < params.hl_height)
            ? hl[band_y * params.hl_width + band_x]
            : 0.0f;
    } else if (low_x) {
        const uint band_x = full_band_x - params.lh_x;
        const uint band_y = full_band_y - params.lh_y;
        out[out_idx] = (band_x < params.lh_width && band_y < params.lh_height)
            ? lh[band_y * params.lh_width + band_x]
            : 0.0f;
    } else {
        const uint band_x = full_band_x - params.hh_x;
        const uint band_y = full_band_y - params.hh_y;
        out[out_idx] = (band_x < params.hh_width && band_y < params.hh_height)
            ? hh[band_y * params.hh_width + band_x]
            : 0.0f;
    }
}

kernel void j2k_idwt_interleave_batched(
    device const float *ll [[buffer(0)]],
    device const float *hl [[buffer(1)]],
    device const float *lh [[buffer(2)]],
    device const float *hh [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant J2kRepeatedIdwtSingleDecompositionParams &params [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.batch_count) {
        return;
    }

    const uint global_x = params.x0 + params.output_x + gid.x;
    const uint global_y = params.y0 + params.output_y + gid.y;
    // Low/high assignment is defined in the reference grid.  For an odd
    // tile origin, the first sample is therefore high-pass.
    const bool low_x = (global_x & 1u) == 0u;
    const bool low_y = (global_y & 1u) == 0u;
    const uint full_band_x = low_x ? low_index(global_x, params.x0) : high_index(global_x, params.x0);
    const uint full_band_y = low_y ? low_index(global_y, params.y0) : high_index(global_y, params.y0);
    const uint out_plane_len = params.width * params.height;
    const uint out_idx = gid.z * out_plane_len + gid.y * params.width + gid.x;

    if (low_y && low_x) {
        const uint band_x = full_band_x - params.ll_x;
        const uint band_y = full_band_y - params.ll_y;
        out[out_idx] = (band_x < params.ll_width && band_y < params.ll_height)
            ? ll[gid.z * params.ll_instance_stride + band_y * params.ll_width + band_x]
            : 0.0f;
    } else if (low_y) {
        const uint band_x = full_band_x - params.hl_x;
        const uint band_y = full_band_y - params.hl_y;
        out[out_idx] = (band_x < params.hl_width && band_y < params.hl_height)
            ? hl[gid.z * params.hl_instance_stride + band_y * params.hl_width + band_x]
            : 0.0f;
    } else if (low_x) {
        const uint band_x = full_band_x - params.lh_x;
        const uint band_y = full_band_y - params.lh_y;
        out[out_idx] = (band_x < params.lh_width && band_y < params.lh_height)
            ? lh[gid.z * params.lh_instance_stride + band_y * params.lh_width + band_x]
            : 0.0f;
    } else {
        const uint band_x = full_band_x - params.hh_x;
        const uint band_y = full_band_y - params.hh_y;
        out[out_idx] = (band_x < params.hh_width && band_y < params.hh_height)
            ? hh[gid.z * params.hh_instance_stride + band_y * params.hh_width + band_x]
            : 0.0f;
    }
}

kernel void j2k_idwt_irreversible97_interleave_horizontal_scale(
    device const float *ll [[buffer(0)]],
    device const float *hl [[buffer(1)]],
    device const float *lh [[buffer(2)]],
    device const float *hh [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(5)]],
    constant float &high_pass [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint global_x = params.x0 + params.output_x + gid.x;
    const uint global_y = params.y0 + params.output_y + gid.y;
    const bool low_x = (global_x & 1u) == 0u;
    const bool low_y = (global_y & 1u) == 0u;
    const uint full_band_x = low_x ? low_index(global_x, params.x0) : high_index(global_x, params.x0);
    const uint full_band_y = low_y ? low_index(global_y, params.y0) : high_index(global_y, params.y0);
    const uint out_idx = gid.y * params.width + gid.x;
    float sample;

    if (low_y && low_x) {
        const uint band_x = full_band_x - params.ll_x;
        const uint band_y = full_band_y - params.ll_y;
        sample = (band_x < params.ll_width && band_y < params.ll_height)
            ? ll[band_y * params.ll_width + band_x]
            : 0.0f;
    } else if (low_y) {
        const uint band_x = full_band_x - params.hl_x;
        const uint band_y = full_band_y - params.hl_y;
        sample = (band_x < params.hl_width && band_y < params.hl_height)
            ? hl[band_y * params.hl_width + band_x]
            : 0.0f;
    } else if (low_x) {
        const uint band_x = full_band_x - params.lh_x;
        const uint band_y = full_band_y - params.lh_y;
        sample = (band_x < params.lh_width && band_y < params.lh_height)
            ? lh[band_y * params.lh_width + band_x]
            : 0.0f;
    } else {
        const uint band_x = full_band_x - params.hh_x;
        const uint band_y = full_band_y - params.hh_y;
        sample = (band_x < params.hh_width && band_y < params.hh_height)
            ? hh[band_y * params.hh_width + band_x]
            : 0.0f;
    }

    if (params.width == 1u) {
        if (((params.x0 + params.output_x) & 1u) != 0u) {
            sample *= 0.5f;
        }
    } else {
        const uint first_even_x = (params.x0 + params.output_x) & 1u;
        const float KAPPA = CODEC_MATH_DWT97_KAPPA;
        sample *= (gid.x & 1u) == first_even_x ? KAPPA : high_pass;
    }
    out[out_idx] = sample;
}

kernel void j2k_idwt_irreversible97_interleave_horizontal_scale_batched(
    device const float *ll [[buffer(0)]],
    device const float *hl [[buffer(1)]],
    device const float *lh [[buffer(2)]],
    device const float *hh [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant J2kRepeatedIdwtSingleDecompositionParams &params [[buffer(5)]],
    constant float &high_pass [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.batch_count) {
        return;
    }

    const uint global_x = params.x0 + params.output_x + gid.x;
    const uint global_y = params.y0 + params.output_y + gid.y;
    const bool low_x = (global_x & 1u) == 0u;
    const bool low_y = (global_y & 1u) == 0u;
    const uint full_band_x = low_x ? low_index(global_x, params.x0) : high_index(global_x, params.x0);
    const uint full_band_y = low_y ? low_index(global_y, params.y0) : high_index(global_y, params.y0);
    const uint out_plane_len = params.width * params.height;
    const uint out_idx = gid.z * out_plane_len + gid.y * params.width + gid.x;
    float sample;

    if (low_y && low_x) {
        const uint band_x = full_band_x - params.ll_x;
        const uint band_y = full_band_y - params.ll_y;
        sample = (band_x < params.ll_width && band_y < params.ll_height)
            ? ll[gid.z * params.ll_instance_stride + band_y * params.ll_width + band_x]
            : 0.0f;
    } else if (low_y) {
        const uint band_x = full_band_x - params.hl_x;
        const uint band_y = full_band_y - params.hl_y;
        sample = (band_x < params.hl_width && band_y < params.hl_height)
            ? hl[gid.z * params.hl_instance_stride + band_y * params.hl_width + band_x]
            : 0.0f;
    } else if (low_x) {
        const uint band_x = full_band_x - params.lh_x;
        const uint band_y = full_band_y - params.lh_y;
        sample = (band_x < params.lh_width && band_y < params.lh_height)
            ? lh[gid.z * params.lh_instance_stride + band_y * params.lh_width + band_x]
            : 0.0f;
    } else {
        const uint band_x = full_band_x - params.hh_x;
        const uint band_y = full_band_y - params.hh_y;
        sample = (band_x < params.hh_width && band_y < params.hh_height)
            ? hh[gid.z * params.hh_instance_stride + band_y * params.hh_width + band_x]
            : 0.0f;
    }

    if (params.width == 1u) {
        if (((params.x0 + params.output_x) & 1u) != 0u) {
            sample *= 0.5f;
        }
    } else {
        const uint first_even_x = (params.x0 + params.output_x) & 1u;
        const float KAPPA = CODEC_MATH_DWT97_KAPPA;
        sample *= (gid.x & 1u) == first_even_x ? KAPPA : high_pass;
    }
    out[out_idx] = sample;
}

kernel void j2k_idwt_reversible53_horizontal_pass(
    device float *out [[buffer(0)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.height) {
        return;
    }

    device float *row_ptr = out + gid * params.width;

    if (params.width == 1u) {
        if (((params.x0 + params.output_x) & 1u) != 0u) {
            row_ptr[0] *= 0.5f;
        }
        return;
    }

    const uint first_even_x = (params.x0 + params.output_x) & 1u;
    const uint first_odd_x = 1u - first_even_x;

    if (first_even_x == 0u) {
        const uint left = periodic_symmetric_extension_left_u32(0u, 1u);
        const uint right = periodic_symmetric_extension_right_u32(0u, 1u, params.width);
        row_ptr[0] = reversible53_predict(row_ptr[0], row_ptr[left], row_ptr[right]);
    }

    const uint even_middle_start = first_even_x == 0u ? 2u : 1u;
    for (uint x = even_middle_start; x + 1u < params.width; x += 2u) {
        row_ptr[x] = reversible53_predict(row_ptr[x], row_ptr[x - 1u], row_ptr[x + 1u]);
    }

    if (((params.width - 1u) & 1u) == first_even_x) {
        const uint x = params.width - 1u;
        const uint left = periodic_symmetric_extension_left_u32(x, 1u);
        const uint right = periodic_symmetric_extension_right_u32(x, 1u, params.width);
        row_ptr[x] = reversible53_predict(row_ptr[x], row_ptr[left], row_ptr[right]);
    }

    if (first_odd_x == 0u) {
        const uint left = periodic_symmetric_extension_left_u32(0u, 1u);
        const uint right = periodic_symmetric_extension_right_u32(0u, 1u, params.width);
        row_ptr[0] = reversible53_update(row_ptr[0], row_ptr[left], row_ptr[right]);
    }

    const uint odd_middle_start = first_odd_x == 0u ? 2u : 1u;
    for (uint x = odd_middle_start; x + 1u < params.width; x += 2u) {
        row_ptr[x] = reversible53_update(row_ptr[x], row_ptr[x - 1u], row_ptr[x + 1u]);
    }

    if (((params.width - 1u) & 1u) == first_odd_x) {
        const uint x = params.width - 1u;
        const uint left = periodic_symmetric_extension_left_u32(x, 1u);
        const uint right = periodic_symmetric_extension_right_u32(x, 1u, params.width);
        row_ptr[x] = reversible53_update(row_ptr[x], row_ptr[left], row_ptr[right]);
    }
}

kernel void j2k_idwt_reversible53_horizontal_pass_batched(
    device float *out [[buffer(0)]],
    constant J2kRepeatedIdwtSingleDecompositionParams &params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.height || gid.y >= params.batch_count) {
        return;
    }

    const uint plane_len = params.width * params.height;
    device float *row_ptr = out + gid.y * plane_len + gid.x * params.width;

    if (params.width == 1u) {
        if (((params.x0 + params.output_x) & 1u) != 0u) {
            row_ptr[0] *= 0.5f;
        }
        return;
    }

    const uint first_even_x = (params.x0 + params.output_x) & 1u;
    const uint first_odd_x = 1u - first_even_x;

    if (first_even_x == 0u) {
        const uint left = periodic_symmetric_extension_left_u32(0u, 1u);
        const uint right = periodic_symmetric_extension_right_u32(0u, 1u, params.width);
        row_ptr[0] = reversible53_predict(row_ptr[0], row_ptr[left], row_ptr[right]);
    }

    const uint even_middle_start = first_even_x == 0u ? 2u : 1u;
    for (uint x = even_middle_start; x + 1u < params.width; x += 2u) {
        row_ptr[x] = reversible53_predict(row_ptr[x], row_ptr[x - 1u], row_ptr[x + 1u]);
    }

    if (((params.width - 1u) & 1u) == first_even_x) {
        const uint x = params.width - 1u;
        const uint left = periodic_symmetric_extension_left_u32(x, 1u);
        const uint right = periodic_symmetric_extension_right_u32(x, 1u, params.width);
        row_ptr[x] = reversible53_predict(row_ptr[x], row_ptr[left], row_ptr[right]);
    }

    if (first_odd_x == 0u) {
        const uint left = periodic_symmetric_extension_left_u32(0u, 1u);
        const uint right = periodic_symmetric_extension_right_u32(0u, 1u, params.width);
        row_ptr[0] = reversible53_update(row_ptr[0], row_ptr[left], row_ptr[right]);
    }

    const uint odd_middle_start = first_odd_x == 0u ? 2u : 1u;
    for (uint x = odd_middle_start; x + 1u < params.width; x += 2u) {
        row_ptr[x] = reversible53_update(row_ptr[x], row_ptr[x - 1u], row_ptr[x + 1u]);
    }

    if (((params.width - 1u) & 1u) == first_odd_x) {
        const uint x = params.width - 1u;
        const uint left = periodic_symmetric_extension_left_u32(x, 1u);
        const uint right = periodic_symmetric_extension_right_u32(x, 1u, params.width);
        row_ptr[x] = reversible53_update(row_ptr[x], row_ptr[left], row_ptr[right]);
    }
}

kernel void j2k_idwt_reversible53_vertical_pass(
    device float *out [[buffer(0)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.width) {
        return;
    }

    if (params.height == 1u) {
        if (((params.y0 + params.output_y) & 1u) != 0u) {
            out[gid] *= 0.5f;
        }
        return;
    }

    const uint first_even_y = (params.y0 + params.output_y) & 1u;
    const uint first_odd_y = 1u - first_even_y;

    for (uint row = first_even_y; row < params.height; row += 2u) {
        const uint row_above = periodic_symmetric_extension_left_u32(row, 1u);
        const uint row_below = periodic_symmetric_extension_right_u32(row, 1u, params.height);
        const uint idx = row * params.width + gid;
        out[idx] = reversible53_predict(
            out[idx],
            out[row_above * params.width + gid],
            out[row_below * params.width + gid]
        );
    }

    for (uint row = first_odd_y; row < params.height; row += 2u) {
        const uint row_above = periodic_symmetric_extension_left_u32(row, 1u);
        const uint row_below = periodic_symmetric_extension_right_u32(row, 1u, params.height);
        const uint idx = row * params.width + gid;
        out[idx] = reversible53_update(
            out[idx],
            out[row_above * params.width + gid],
            out[row_below * params.width + gid]
        );
    }
}

kernel void j2k_idwt_reversible53_vertical_pass_batched(
    device float *out [[buffer(0)]],
    constant J2kRepeatedIdwtSingleDecompositionParams &params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.batch_count) {
        return;
    }

    const uint plane_len = params.width * params.height;
    device float *plane = out + gid.y * plane_len;

    if (params.height == 1u) {
        if (((params.y0 + params.output_y) & 1u) != 0u) {
            plane[gid.x] *= 0.5f;
        }
        return;
    }

    const uint first_even_y = (params.y0 + params.output_y) & 1u;
    const uint first_odd_y = 1u - first_even_y;

    for (uint row = first_even_y; row < params.height; row += 2u) {
        const uint row_above = periodic_symmetric_extension_left_u32(row, 1u);
        const uint row_below = periodic_symmetric_extension_right_u32(row, 1u, params.height);
        const uint idx = row * params.width + gid.x;
        plane[idx] = reversible53_predict(
            plane[idx],
            plane[row_above * params.width + gid.x],
            plane[row_below * params.width + gid.x]
        );
    }

    for (uint row = first_odd_y; row < params.height; row += 2u) {
        const uint row_above = periodic_symmetric_extension_left_u32(row, 1u);
        const uint row_below = periodic_symmetric_extension_right_u32(row, 1u, params.height);
        const uint idx = row * params.width + gid.x;
        plane[idx] = reversible53_update(
            plane[idx],
            plane[row_above * params.width + gid.x],
            plane[row_below * params.width + gid.x]
        );
    }
}

kernel void j2k_idwt_irreversible97_horizontal_scale(
    device float *out [[buffer(0)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(1)]],
    constant float &high_pass [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    out += ulong(gid.z) * params.width * params.height;
    const float KAPPA = CODEC_MATH_DWT97_KAPPA;
    float sample = out[gid.y * params.width + gid.x];

    if (params.width == 1u) {
        if (((params.x0 + params.output_x) & 1u) != 0u) {
            sample *= 0.5f;
        }
    } else {
        const uint first_even_x = (params.x0 + params.output_x) & 1u;
        sample *= (gid.x & 1u) == first_even_x ? KAPPA : high_pass;
    }

    out[gid.y * params.width + gid.x] = sample;
}

kernel void j2k_idwt_irreversible97_vertical_scale(
    device float *out [[buffer(0)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(1)]],
    constant float &high_pass [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    out += ulong(gid.z) * params.width * params.height;
    const float KAPPA = CODEC_MATH_DWT97_KAPPA;
    float sample = out[gid.y * params.width + gid.x];

    if (params.height == 1u) {
        if (((params.y0 + params.output_y) & 1u) != 0u) {
            sample *= 0.5f;
        }
    } else {
        const uint first_even_y = (params.y0 + params.output_y) & 1u;
        sample *= (gid.y & 1u) == first_even_y ? KAPPA : high_pass;
    }

    out[gid.y * params.width + gid.x] = sample;
}

kernel void j2k_idwt_irreversible97_horizontal_step(
    device float *out [[buffer(0)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(1)]],
    constant J2kIdwt97StepParams &step [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint x = 2u * gid.x + step.parity;
    if (x >= params.width || gid.y >= params.height || params.width <= 1u) {
        return;
    }

    out += ulong(gid.z) * params.width * params.height;
    const uint left = periodic_symmetric_extension_left_u32(x, 1u);
    const uint right = periodic_symmetric_extension_right_u32(x, 1u, params.width);
    const uint idx = gid.y * params.width + x;
    out[idx] = fma(out[gid.y * params.width + left] + out[gid.y * params.width + right],
                   step.coefficient,
                   out[idx]);
}

kernel void j2k_idwt_irreversible97_vertical_step(
    device float *out [[buffer(0)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(1)]],
    constant J2kIdwt97StepParams &step [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint y = 2u * gid.y + step.parity;
    if (gid.x >= params.width || y >= params.height || params.height <= 1u) {
        return;
    }

    out += ulong(gid.z) * params.width * params.height;
    const uint above = periodic_symmetric_extension_left_u32(y, 1u);
    const uint below = periodic_symmetric_extension_right_u32(y, 1u, params.height);
    const uint idx = y * params.width + gid.x;
    out[idx] = fma(out[above * params.width + gid.x] + out[below * params.width + gid.x],
                   step.coefficient,
                   out[idx]);
}
