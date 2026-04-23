// SPDX-License-Identifier: Apache-2.0

struct J2kIdwtSingleDecompositionParams {
    uint x0;
    uint y0;
    uint width;
    uint height;
    uint ll_width;
    uint ll_height;
    uint hl_width;
    uint hl_height;
    uint lh_width;
    uint lh_height;
    uint hh_width;
    uint hh_height;
};

struct J2kRepeatedIdwtSingleDecompositionParams {
    uint x0;
    uint y0;
    uint width;
    uint height;
    uint ll_width;
    uint ll_height;
    uint hl_width;
    uint hl_height;
    uint lh_width;
    uint lh_height;
    uint hh_width;
    uint hh_height;
    uint batch_count;
};

struct J2kIdwtStatus {
    uint code;
    uint detail;
    uint reserved0;
    uint reserved1;
};

constant uint J2K_IDWT_STATUS_OK = 0;
constant uint J2K_IDWT_STATUS_FAIL = 1;

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

    const uint global_x = params.x0 + gid.x;
    const uint global_y = params.y0 + gid.y;
    const uint low_x_parity = params.x0 & 1u;
    const uint low_y_parity = params.y0 & 1u;
    const bool low_x = (global_x & 1u) == low_x_parity;
    const bool low_y = (global_y & 1u) == low_y_parity;
    const uint band_x = low_x ? low_index(global_x, params.x0) : high_index(global_x, params.x0);
    const uint band_y = low_y ? low_index(global_y, params.y0) : high_index(global_y, params.y0);
    const uint out_idx = gid.y * params.width + gid.x;

    if (low_y && low_x) {
        out[out_idx] = ll[band_y * params.ll_width + band_x];
    } else if (low_y) {
        out[out_idx] = hl[band_y * params.hl_width + band_x];
    } else if (low_x) {
        out[out_idx] = lh[band_y * params.lh_width + band_x];
    } else {
        out[out_idx] = hh[band_y * params.hh_width + band_x];
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

    const uint global_x = params.x0 + gid.x;
    const uint global_y = params.y0 + gid.y;
    const uint low_x_parity = params.x0 & 1u;
    const uint low_y_parity = params.y0 & 1u;
    const bool low_x = (global_x & 1u) == low_x_parity;
    const bool low_y = (global_y & 1u) == low_y_parity;
    const uint band_x = low_x ? low_index(global_x, params.x0) : high_index(global_x, params.x0);
    const uint band_y = low_y ? low_index(global_y, params.y0) : high_index(global_y, params.y0);
    const uint out_plane_len = params.width * params.height;
    const uint ll_plane_len = params.ll_width * params.ll_height;
    const uint hl_plane_len = params.hl_width * params.hl_height;
    const uint lh_plane_len = params.lh_width * params.lh_height;
    const uint hh_plane_len = params.hh_width * params.hh_height;
    const uint out_idx = gid.z * out_plane_len + gid.y * params.width + gid.x;

    if (low_y && low_x) {
        out[out_idx] = ll[gid.z * ll_plane_len + band_y * params.ll_width + band_x];
    } else if (low_y) {
        out[out_idx] = hl[gid.z * hl_plane_len + band_y * params.hl_width + band_x];
    } else if (low_x) {
        out[out_idx] = lh[gid.z * lh_plane_len + band_y * params.lh_width + band_x];
    } else {
        out[out_idx] = hh[gid.z * hh_plane_len + band_y * params.hh_width + band_x];
    }
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
        if ((params.x0 & 1u) != 0u) {
            row_ptr[0] *= 0.5f;
        }
        return;
    }

    const uint first_even_x = params.x0 & 1u;
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
        if ((params.x0 & 1u) != 0u) {
            row_ptr[0] *= 0.5f;
        }
        return;
    }

    const uint first_even_x = params.x0 & 1u;
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
        if ((params.y0 & 1u) != 0u) {
            out[gid] *= 0.5f;
        }
        return;
    }

    const uint first_even_y = params.y0 & 1u;
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
        if ((params.y0 & 1u) != 0u) {
            plane[gid.x] *= 0.5f;
        }
        return;
    }

    const uint first_even_y = params.y0 & 1u;
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

kernel void j2k_idwt_reversible53_single_decomposition(
    device const float *ll [[buffer(0)]],
    device const float *hl [[buffer(1)]],
    device const float *lh [[buffer(2)]],
    device const float *hh [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(5)]],
    device J2kIdwtStatus *status [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }

    status->code = J2K_IDWT_STATUS_OK;
    status->detail = 0u;
    status->reserved0 = 0u;
    status->reserved1 = 0u;

    if (params.width == 0u || params.height == 0u) {
        status->code = J2K_IDWT_STATUS_FAIL;
        status->detail = 1u;
        return;
    }

    const uint low_x_parity = params.x0 & 1u;
    const uint low_y_parity = params.y0 & 1u;

    for (uint local_y = 0u; local_y < params.height; ++local_y) {
        const uint global_y = params.y0 + local_y;
        const bool low_y = (global_y & 1u) == low_y_parity;
        const uint band_y = low_y ? low_index(global_y, params.y0) : high_index(global_y, params.y0);

        for (uint local_x = 0u; local_x < params.width; ++local_x) {
            const uint global_x = params.x0 + local_x;
            const bool low_x = (global_x & 1u) == low_x_parity;
            const uint band_x = low_x ? low_index(global_x, params.x0) : high_index(global_x, params.x0);
            const uint out_idx = local_y * params.width + local_x;

            if (low_y && low_x) {
                if (band_x >= params.ll_width || band_y >= params.ll_height) {
                    status->code = J2K_IDWT_STATUS_FAIL;
                    status->detail = 2u;
                    return;
                }
                out[out_idx] = ll[band_y * params.ll_width + band_x];
            } else if (low_y) {
                if (band_x >= params.hl_width || band_y >= params.hl_height) {
                    status->code = J2K_IDWT_STATUS_FAIL;
                    status->detail = 3u;
                    return;
                }
                out[out_idx] = hl[band_y * params.hl_width + band_x];
            } else if (low_x) {
                if (band_x >= params.lh_width || band_y >= params.lh_height) {
                    status->code = J2K_IDWT_STATUS_FAIL;
                    status->detail = 4u;
                    return;
                }
                out[out_idx] = lh[band_y * params.lh_width + band_x];
            } else {
                if (band_x >= params.hh_width || band_y >= params.hh_height) {
                    status->code = J2K_IDWT_STATUS_FAIL;
                    status->detail = 5u;
                    return;
                }
                out[out_idx] = hh[band_y * params.hh_width + band_x];
            }
        }
    }

    if (params.width == 1u) {
        if ((params.x0 & 1u) != 0u) {
            for (uint row = 0u; row < params.height; ++row) {
                out[row * params.width] *= 0.5f;
            }
        }
    } else {
        const uint first_even_x = params.x0 & 1u;
        const uint first_odd_x = 1u - first_even_x;

        for (uint row = 0u; row < params.height; ++row) {
            device float *row_ptr = out + row * params.width;

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
    }

    if (params.height == 1u) {
        if ((params.y0 & 1u) != 0u) {
            for (uint col = 0u; col < params.width; ++col) {
                out[col] *= 0.5f;
            }
        }
        return;
    }

    const uint first_even_y = params.y0 & 1u;
    const uint first_odd_y = 1u - first_even_y;

    for (uint row = first_even_y; row < params.height; row += 2u) {
        const uint row_above = periodic_symmetric_extension_left_u32(row, 1u);
        const uint row_below = periodic_symmetric_extension_right_u32(row, 1u, params.height);
        for (uint col = 0u; col < params.width; ++col) {
            const uint idx = row * params.width + col;
            out[idx] = reversible53_predict(
                out[idx],
                out[row_above * params.width + col],
                out[row_below * params.width + col]
            );
        }
    }

    for (uint row = first_odd_y; row < params.height; row += 2u) {
        const uint row_above = periodic_symmetric_extension_left_u32(row, 1u);
        const uint row_below = periodic_symmetric_extension_right_u32(row, 1u, params.height);
        for (uint col = 0u; col < params.width; ++col) {
            const uint idx = row * params.width + col;
            out[idx] = reversible53_update(
                out[idx],
                out[row_above * params.width + col],
                out[row_below * params.width + col]
            );
        }
    }
}

kernel void j2k_idwt_irreversible97_single_decomposition(
    device const float *ll [[buffer(0)]],
    device const float *hl [[buffer(1)]],
    device const float *lh [[buffer(2)]],
    device const float *hh [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(5)]],
    device J2kIdwtStatus *status [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }

    status->code = J2K_IDWT_STATUS_OK;
    status->detail = 0u;
    status->reserved0 = 0u;
    status->reserved1 = 0u;

    if (params.width == 0u || params.height == 0u) {
        status->code = J2K_IDWT_STATUS_FAIL;
        status->detail = 1u;
        return;
    }

    const float NEG_ALPHA = 1.5861343f;
    const float NEG_BETA = 0.052980117f;
    const float NEG_GAMMA = -0.8829111f;
    const float NEG_DELTA = -0.44350687f;
    const float KAPPA = 1.2301741f;
    const float INV_KAPPA = 1.0f / KAPPA;

    const uint low_x_parity = params.x0 & 1u;
    const uint low_y_parity = params.y0 & 1u;

    for (uint local_y = 0u; local_y < params.height; ++local_y) {
        const uint global_y = params.y0 + local_y;
        const bool low_y = (global_y & 1u) == low_y_parity;
        const uint band_y = low_y ? low_index(global_y, params.y0) : high_index(global_y, params.y0);

        for (uint local_x = 0u; local_x < params.width; ++local_x) {
            const uint global_x = params.x0 + local_x;
            const bool low_x = (global_x & 1u) == low_x_parity;
            const uint band_x = low_x ? low_index(global_x, params.x0) : high_index(global_x, params.x0);
            const uint out_idx = local_y * params.width + local_x;

            if (low_y && low_x) {
                if (band_x >= params.ll_width || band_y >= params.ll_height) {
                    status->code = J2K_IDWT_STATUS_FAIL;
                    status->detail = 2u;
                    return;
                }
                out[out_idx] = ll[band_y * params.ll_width + band_x];
            } else if (low_y) {
                if (band_x >= params.hl_width || band_y >= params.hl_height) {
                    status->code = J2K_IDWT_STATUS_FAIL;
                    status->detail = 3u;
                    return;
                }
                out[out_idx] = hl[band_y * params.hl_width + band_x];
            } else if (low_x) {
                if (band_x >= params.lh_width || band_y >= params.lh_height) {
                    status->code = J2K_IDWT_STATUS_FAIL;
                    status->detail = 4u;
                    return;
                }
                out[out_idx] = lh[band_y * params.lh_width + band_x];
            } else {
                if (band_x >= params.hh_width || band_y >= params.hh_height) {
                    status->code = J2K_IDWT_STATUS_FAIL;
                    status->detail = 5u;
                    return;
                }
                out[out_idx] = hh[band_y * params.hh_width + band_x];
            }
        }
    }

    if (params.width == 1u) {
        if ((params.x0 & 1u) != 0u) {
            for (uint row = 0u; row < params.height; ++row) {
                out[row * params.width] *= 0.5f;
            }
        }
    } else {
        const uint first_even_x = params.x0 & 1u;
        const uint first_odd_x = 1u - first_even_x;
        const float k0 = first_even_x == 0u ? KAPPA : INV_KAPPA;
        const float k1 = first_even_x == 0u ? INV_KAPPA : KAPPA;

        for (uint row = 0u; row < params.height; ++row) {
            device float *row_ptr = out + row * params.width;

            for (uint x = 0u; x + 1u < params.width; x += 2u) {
                row_ptr[x] *= k0;
                row_ptr[x + 1u] *= k1;
            }
            if ((params.width & 1u) != 0u) {
                row_ptr[params.width - 1u] *= k0;
            }

            irreversible97_horizontal_step(row_ptr, params.width, first_even_x, NEG_DELTA);
            irreversible97_horizontal_step(row_ptr, params.width, first_odd_x, NEG_GAMMA);
            irreversible97_horizontal_step(row_ptr, params.width, first_even_x, NEG_BETA);
            irreversible97_horizontal_step(row_ptr, params.width, first_odd_x, NEG_ALPHA);
        }
    }

    if (params.height == 1u) {
        if ((params.y0 & 1u) != 0u) {
            for (uint col = 0u; col < params.width; ++col) {
                out[col] *= 0.5f;
            }
        }
        return;
    }

    const uint first_even_y = params.y0 & 1u;
    const uint first_odd_y = 1u - first_even_y;
    const float k0 = first_even_y == 0u ? KAPPA : INV_KAPPA;
    const float k1 = first_even_y == 0u ? INV_KAPPA : KAPPA;

    for (uint row = 0u; row + 1u < params.height; row += 2u) {
        for (uint col = 0u; col < params.width; ++col) {
            out[row * params.width + col] *= k0;
            out[(row + 1u) * params.width + col] *= k1;
        }
    }
    if ((params.height & 1u) != 0u) {
        const uint row = params.height - 1u;
        for (uint col = 0u; col < params.width; ++col) {
            out[row * params.width + col] *= k0;
        }
    }

    for (uint row = first_even_y; row < params.height; row += 2u) {
        const uint row_above = periodic_symmetric_extension_left_u32(row, 1u);
        const uint row_below = periodic_symmetric_extension_right_u32(row, 1u, params.height);
        for (uint col = 0u; col < params.width; ++col) {
            const uint idx = row * params.width + col;
            out[idx] = fma(
                out[row_above * params.width + col] + out[row_below * params.width + col],
                NEG_DELTA,
                out[idx]
            );
        }
    }

    for (uint row = first_odd_y; row < params.height; row += 2u) {
        const uint row_above = periodic_symmetric_extension_left_u32(row, 1u);
        const uint row_below = periodic_symmetric_extension_right_u32(row, 1u, params.height);
        for (uint col = 0u; col < params.width; ++col) {
            const uint idx = row * params.width + col;
            out[idx] = fma(
                out[row_above * params.width + col] + out[row_below * params.width + col],
                NEG_GAMMA,
                out[idx]
            );
        }
    }

    for (uint row = first_even_y; row < params.height; row += 2u) {
        const uint row_above = periodic_symmetric_extension_left_u32(row, 1u);
        const uint row_below = periodic_symmetric_extension_right_u32(row, 1u, params.height);
        for (uint col = 0u; col < params.width; ++col) {
            const uint idx = row * params.width + col;
            out[idx] = fma(
                out[row_above * params.width + col] + out[row_below * params.width + col],
                NEG_BETA,
                out[idx]
            );
        }
    }

    for (uint row = first_odd_y; row < params.height; row += 2u) {
        const uint row_above = periodic_symmetric_extension_left_u32(row, 1u);
        const uint row_below = periodic_symmetric_extension_right_u32(row, 1u, params.height);
        for (uint col = 0u; col < params.width; ++col) {
            const uint idx = row * params.width + col;
            out[idx] = fma(
                out[row_above * params.width + col] + out[row_below * params.width + col],
                NEG_ALPHA,
                out[idx]
            );
        }
    }
}
