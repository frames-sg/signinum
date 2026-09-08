// SPDX-License-Identifier: MIT OR Apache-2.0

// Appended to the ordinary decode source only by the test-only experiment.
inline void audit_reversible53_line(
    device float *out, uint stride, uint length, uint origin,
    threadgroup float *line, uint lane, uint lanes
) {
    for (uint x = lane; x < length; x += lanes) {
        line[x] = out[ulong(x) * stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (length == 1u) {
        if (lane == 0u && (origin & 1u) != 0u) {
            line[0] *= 0.5f;
        }
    } else {
        const uint even = origin & 1u;
        for (uint x = even + 2u * lane; x < length; x += 2u * lanes) {
            const uint left = periodic_symmetric_extension_left_u32(x, 1u);
            const uint right = periodic_symmetric_extension_right_u32(x, 1u, length);
            line[x] = reversible53_predict(line[x], line[left], line[right]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint x = 1u - even + 2u * lane; x < length; x += 2u * lanes) {
            const uint left = periodic_symmetric_extension_left_u32(x, 1u);
            const uint right = periodic_symmetric_extension_right_u32(x, 1u, length);
            line[x] = reversible53_update(line[x], line[left], line[right]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint x = lane; x < length; x += lanes) {
        out[ulong(x) * stride] = line[x];
    }
}

kernel void audit_idwt53_horizontal_cooperative(
    device float *out [[buffer(0)]],
    constant J2kRepeatedIdwtSingleDecompositionParams &params [[buffer(1)]],
    threadgroup float *line [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 lanes [[threads_per_threadgroup]]
) {
    // The host launches exactly height by batch_count complete threadgroups.
    out += ulong(group.y) * params.width * params.height + ulong(group.x) * params.width;
    audit_reversible53_line(out, 1u, params.width, params.x0 + params.output_x,
                            line, lane, lanes.x);
}

kernel void audit_idwt53_vertical_cooperative(
    device float *out [[buffer(0)]],
    constant J2kRepeatedIdwtSingleDecompositionParams &params [[buffer(1)]],
    threadgroup float *line [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 lanes [[threads_per_threadgroup]]
) {
    out += ulong(group.y) * params.width * params.height + group.x;
    audit_reversible53_line(out, params.width, params.height, params.y0 + params.output_y,
                            line, lane, lanes.x);
}
