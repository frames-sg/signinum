// SPDX-License-Identifier: MIT OR Apache-2.0

// Full origin-zero component expansion, matching the native decoder's store.
// params: output width, output height, horizontal sampling, vertical sampling.
kernel void j2k_expand_sampled_plane(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint *params [[buffer(2)]],
    uint2 pos [[thread_position_in_grid]]
) {
    if (pos.x >= params[0] || pos.y >= params[1]) return;
    const uint input_width = params[0] / params[2] + (params[0] % params[2] != 0u);
    output[pos.y * params[0] + pos.x] =
        input[(pos.y / params[3]) * input_width + pos.x / params[2]];
}
