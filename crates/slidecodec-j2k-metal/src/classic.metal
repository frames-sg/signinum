#include <metal_stdlib>
using namespace metal;

struct J2kClassicCleanupBatchJob {
    uint coded_offset;
    uint coded_len;
    uint segment_offset;
    uint segment_count;
    uint width;
    uint height;
    uint output_stride;
    uint output_offset;
    uint missing_msbs;
    uint total_bitplanes;
    uint number_of_coding_passes;
    uint sub_band_type;
    uint style_flags;
    uint strict;
    float dequantization_step;
};

struct J2kClassicSegment {
    uint data_offset;
    uint data_length;
    uint start_coding_pass;
    uint end_coding_pass;
    uint use_arithmetic;
};

struct J2kClassicStatus {
    uint code;
    uint detail;
    uint reserved0;
    uint reserved1;
};

struct J2kQeData {
    uint qe;
    uchar nmps;
    uchar nlps;
    uchar switch_mps;
};

struct J2kArithmeticDecoder {
    device const uchar *data;
    uint data_len;
    uint c;
    uint a;
    uint base_pointer;
    uint shift_count;
};

struct J2kBypassDecoder {
    device const uchar *data;
    uint data_len;
    uint bit_pos;
    uint strict;
};

constant uint J2K_CLASSIC_STATUS_OK = 0u;
constant uint J2K_CLASSIC_STATUS_FAIL = 1u;
constant uint J2K_CLASSIC_STATUS_UNSUPPORTED = 2u;
constant uint J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES = 1u << 0;
constant uint J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS = 1u << 1;
constant uint J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT = 1u << 2;
constant uint J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS = 1u << 3;
constant uint J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS = 1u << 4;

constant uint J2K_CLASSIC_MAX_WIDTH = 64u;
constant uint J2K_CLASSIC_MAX_HEIGHT = 64u;
constant uint J2K_CLASSIC_PADDING = 1u;
constant uint J2K_CLASSIC_MAX_PADDED_WIDTH = J2K_CLASSIC_MAX_WIDTH + J2K_CLASSIC_PADDING * 2u;
constant uint J2K_CLASSIC_MAX_PADDED_HEIGHT = J2K_CLASSIC_MAX_HEIGHT + J2K_CLASSIC_PADDING * 2u;
constant uint J2K_CLASSIC_MAX_COEFF_COUNT = J2K_CLASSIC_MAX_PADDED_WIDTH * J2K_CLASSIC_MAX_PADDED_HEIGHT;
constant uchar J2K_SIG_SHIFT = 7u;
constant uchar J2K_MAG_REF_SHIFT = 6u;
constant uchar J2K_SIGN_SHIFT = 5u;
constant uchar J2K_STATE_MARKER_MASK = uchar(0x1Fu);

constant J2kQeData J2K_QE_TABLE[47] = {
    {0x5601u, 1u, 1u, 1u},
    {0x3401u, 2u, 6u, 0u},
    {0x1801u, 3u, 9u, 0u},
    {0x0AC1u, 4u, 12u, 0u},
    {0x0521u, 5u, 29u, 0u},
    {0x0221u, 38u, 33u, 0u},
    {0x5601u, 7u, 6u, 1u},
    {0x5401u, 8u, 14u, 0u},
    {0x4801u, 9u, 14u, 0u},
    {0x3801u, 10u, 14u, 0u},
    {0x3001u, 11u, 17u, 0u},
    {0x2401u, 12u, 18u, 0u},
    {0x1C01u, 13u, 20u, 0u},
    {0x1601u, 29u, 21u, 0u},
    {0x5601u, 15u, 14u, 1u},
    {0x5401u, 16u, 14u, 0u},
    {0x5101u, 17u, 15u, 0u},
    {0x4801u, 18u, 16u, 0u},
    {0x3801u, 19u, 17u, 0u},
    {0x3401u, 20u, 18u, 0u},
    {0x3001u, 21u, 19u, 0u},
    {0x2801u, 22u, 19u, 0u},
    {0x2401u, 23u, 20u, 0u},
    {0x2201u, 24u, 21u, 0u},
    {0x1C01u, 25u, 22u, 0u},
    {0x1801u, 26u, 23u, 0u},
    {0x1601u, 27u, 24u, 0u},
    {0x1401u, 28u, 25u, 0u},
    {0x1201u, 29u, 26u, 0u},
    {0x1101u, 30u, 27u, 0u},
    {0x0AC1u, 31u, 28u, 0u},
    {0x09C1u, 32u, 29u, 0u},
    {0x08A1u, 33u, 30u, 0u},
    {0x0521u, 34u, 31u, 0u},
    {0x0441u, 35u, 32u, 0u},
    {0x02A1u, 36u, 33u, 0u},
    {0x0221u, 37u, 34u, 0u},
    {0x0141u, 38u, 35u, 0u},
    {0x0111u, 39u, 36u, 0u},
    {0x0085u, 40u, 37u, 0u},
    {0x0049u, 41u, 38u, 0u},
    {0x0025u, 42u, 39u, 0u},
    {0x0015u, 43u, 40u, 0u},
    {0x0009u, 44u, 41u, 0u},
    {0x0005u, 45u, 42u, 0u},
    {0x0001u, 45u, 43u, 0u},
    {0x5601u, 46u, 46u, 0u},
};

constant uchar2 SIGN_CONTEXT_LOOKUP[256] = {
    uchar2(9,0), uchar2(10,0), uchar2(10,1), uchar2(0,0), uchar2(12,0), uchar2(13,0), uchar2(11,0), uchar2(0,0),
    uchar2(12,1), uchar2(11,1), uchar2(13,1), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
    uchar2(12,0), uchar2(13,0), uchar2(11,0), uchar2(0,0), uchar2(12,0), uchar2(13,0), uchar2(11,0), uchar2(0,0),
    uchar2(9,0), uchar2(10,0), uchar2(10,1), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
    uchar2(12,1), uchar2(11,1), uchar2(13,1), uchar2(0,0), uchar2(9,0), uchar2(10,0), uchar2(10,1), uchar2(0,0),
    uchar2(12,1), uchar2(11,1), uchar2(13,1), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
    uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
    uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
    uchar2(10,0), uchar2(10,0), uchar2(9,0), uchar2(0,0), uchar2(13,0), uchar2(13,0), uchar2(12,0), uchar2(0,0),
    uchar2(11,1), uchar2(11,1), uchar2(12,1), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
    uchar2(13,0), uchar2(13,0), uchar2(12,0), uchar2(0,0), uchar2(13,0), uchar2(13,0), uchar2(12,0), uchar2(0,0),
    uchar2(10,0), uchar2(10,0), uchar2(9,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
    uchar2(11,1), uchar2(11,1), uchar2(12,1), uchar2(0,0), uchar2(10,0), uchar2(10,0), uchar2(9,0), uchar2(0,0),
    uchar2(11,1), uchar2(11,1), uchar2(12,1), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
    uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
    uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
    uchar2(10,1), uchar2(9,0), uchar2(10,1), uchar2(0,0), uchar2(11,0), uchar2(12,0), uchar2(11,0), uchar2(0,0),
    uchar2(13,1), uchar2(12,1), uchar2(13,1), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
    uchar2(11,0), uchar2(12,0), uchar2(11,0), uchar2(0,0), uchar2(11,0), uchar2(12,0), uchar2(11,0), uchar2(0,0),
    uchar2(10,1), uchar2(9,0), uchar2(10,1), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
    uchar2(13,1), uchar2(12,1), uchar2(13,1), uchar2(0,0), uchar2(10,1), uchar2(9,0), uchar2(10,1), uchar2(0,0),
    uchar2(13,1), uchar2(12,1), uchar2(13,1), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
    uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
    uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
    uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
    uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
    uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
    uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
    uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
    uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0), uchar2(0,0),
};

constant uchar ZERO_CTX_LL_LH_LOOKUP[256] = {
    0,3,1,3,5,7,6,7,1,3,2,3,6,7,6,7,5,7,6,7,8,8,8,8,6,7,6,7,8,8,8,8,
    1,3,2,3,6,7,6,7,2,3,2,3,6,7,6,7,6,7,6,7,8,8,8,8,6,7,6,7,8,8,8,8,
    3,4,3,4,7,7,7,7,3,4,3,4,7,7,7,7,7,7,7,7,8,8,8,8,7,7,7,7,8,8,8,8,
    3,4,3,4,7,7,7,7,3,4,3,4,7,7,7,7,7,7,7,7,8,8,8,8,7,7,7,7,8,8,8,8,
    1,3,2,3,6,7,6,7,2,3,2,3,6,7,6,7,6,7,6,7,8,8,8,8,6,7,6,7,8,8,8,8,
    2,3,2,3,6,7,6,7,2,3,2,3,6,7,6,7,6,7,6,7,8,8,8,8,6,7,6,7,8,8,8,8,
    3,4,3,4,7,7,7,7,3,4,3,4,7,7,7,7,7,7,7,7,8,8,8,8,7,7,7,7,8,8,8,8,
    3,4,3,4,7,7,7,7,3,4,3,4,7,7,7,7,7,7,7,7,8,8,8,8,7,7,7,7,8,8,8,8,
};

constant uchar ZERO_CTX_HL_LOOKUP[256] = {
    0,5,1,6,3,7,3,7,1,6,2,6,3,7,3,7,3,7,3,7,4,7,4,7,3,7,3,7,4,7,4,7,
    1,6,2,6,3,7,3,7,2,6,2,6,3,7,3,7,3,7,3,7,4,7,4,7,3,7,3,7,4,7,4,7,
    5,8,6,8,7,8,7,8,6,8,6,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,
    6,8,6,8,7,8,7,8,6,8,6,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,
    1,6,2,6,3,7,3,7,2,6,2,6,3,7,3,7,3,7,3,7,4,7,4,7,3,7,3,7,4,7,4,7,
    2,6,2,6,3,7,3,7,2,6,2,6,3,7,3,7,3,7,3,7,4,7,4,7,3,7,3,7,4,7,4,7,
    6,8,6,8,7,8,7,8,6,8,6,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,
    6,8,6,8,7,8,7,8,6,8,6,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,
};

constant uchar ZERO_CTX_HH_LOOKUP[256] = {
    0,1,3,4,1,2,4,5,3,4,6,7,4,5,7,7,1,2,4,5,2,2,5,5,4,5,7,7,5,5,7,7,
    3,4,6,7,4,5,7,7,6,7,8,8,7,7,8,8,4,5,7,7,5,5,7,7,7,7,8,8,7,7,8,8,
    1,2,4,5,2,2,5,5,4,5,7,7,5,5,7,7,2,2,5,5,2,2,5,5,5,5,7,7,5,5,7,7,
    4,5,7,7,5,5,7,7,7,7,8,8,7,7,8,8,5,5,7,7,5,5,7,7,7,7,8,8,7,7,8,8,
    3,4,6,7,4,5,7,7,6,7,8,8,7,7,8,8,4,5,7,7,5,5,7,7,7,7,8,8,7,7,8,8,
    6,7,8,8,7,7,8,8,8,8,8,8,8,8,8,8,7,7,8,8,7,7,8,8,8,8,8,8,8,8,8,8,
    4,5,7,7,5,5,7,7,7,7,8,8,7,7,8,8,5,5,7,7,5,5,7,7,7,7,8,8,7,7,8,8,
    7,7,8,8,7,7,8,8,8,8,8,8,8,8,8,8,7,7,8,8,7,7,8,8,8,8,8,8,8,8,8,8,
};

inline uint coeff_index(uint padded_width, uint index_x, uint index_y) {
    return index_x + index_y * padded_width;
}

inline void set_classic_status(device J2kClassicStatus *status, uint code, uint detail) {
    status->code = code;
    status->detail = detail;
    status->reserved0 = 0u;
    status->reserved1 = 0u;
}

inline uchar state_bit(thread const uchar *states, uint idx, uchar shift) {
    return (states[idx] >> shift) & uchar(1u);
}

inline void set_state_bit(thread uchar *states, uint idx, uchar shift, uchar value) {
    states[idx] = uchar((states[idx] & uchar(~(1u << shift))) | ((value & 1u) << shift));
}

inline uchar state_bit_dev(device const uchar *states, uint idx, uchar shift) {
    return (states[idx] >> shift) & uchar(1u);
}

inline void set_state_bit_dev(device uchar *states, uint idx, uchar shift, uchar value) {
    states[idx] = uchar((states[idx] & uchar(~(1u << shift))) | ((value & 1u) << shift));
}

inline uchar state_bit_tg(threadgroup const uchar *states, uint idx, uchar shift) {
    return (states[idx] >> shift) & uchar(1u);
}

inline void set_state_bit_tg(threadgroup uchar *states, uint idx, uchar shift, uchar value) {
    states[idx] = uchar((states[idx] & uchar(~(1u << shift))) | ((value & 1u) << shift));
}

inline uint coeff_sign(thread const uchar *states, uint idx) {
    return uint(state_bit(states, idx, J2K_SIGN_SHIFT));
}

inline uint coeff_sign_dev(device const uchar *states, uint idx) {
    return uint(state_bit_dev(states, idx, J2K_SIGN_SHIFT));
}

inline uint coeff_sign_tg(threadgroup const uchar *states, uint idx) {
    return uint(state_bit_tg(states, idx, J2K_SIGN_SHIFT));
}

inline void coeff_push_bit(device uint *coefficients, uint idx, uint bit, uint position) {
    coefficients[idx] |= (bit << position);
}

inline void coeff_set_sign(thread uchar *states, uint idx, uint sign) {
    set_state_bit(states, idx, J2K_SIGN_SHIFT, uchar(sign));
}

inline void coeff_set_sign_dev(device uchar *states, uint idx, uint sign) {
    set_state_bit_dev(states, idx, J2K_SIGN_SHIFT, uchar(sign));
}

inline void coeff_set_sign_tg(threadgroup uchar *states, uint idx, uint sign) {
    set_state_bit_tg(states, idx, J2K_SIGN_SHIFT, uchar(sign));
}

inline uchar coeff_is_significant(thread const uchar *states, uint idx) {
    return state_bit(states, idx, J2K_SIG_SHIFT);
}

inline uchar coeff_is_significant_dev(device const uchar *states, uint idx) {
    return state_bit_dev(states, idx, J2K_SIG_SHIFT);
}

inline uchar coeff_is_significant_tg(threadgroup const uchar *states, uint idx) {
    return state_bit_tg(states, idx, J2K_SIG_SHIFT);
}

inline uchar coeff_zero_coded_marker(thread const uchar *states, uint idx) {
    return states[idx] & J2K_STATE_MARKER_MASK;
}

inline uchar coeff_zero_coded_marker_dev(device const uchar *states, uint idx) {
    return states[idx] & J2K_STATE_MARKER_MASK;
}

inline uchar coeff_zero_coded_marker_tg(threadgroup const uchar *states, uint idx) {
    return states[idx] & J2K_STATE_MARKER_MASK;
}

inline uchar coeff_is_zero_coded(thread const uchar *states, uint idx, uchar marker) {
    return uchar(marker != 0u && coeff_zero_coded_marker(states, idx) == marker);
}

inline uchar coeff_is_zero_coded_dev(device const uchar *states, uint idx, uchar marker) {
    return uchar(marker != 0u && coeff_zero_coded_marker_dev(states, idx) == marker);
}

inline uchar coeff_is_zero_coded_tg(threadgroup const uchar *states, uint idx, uchar marker) {
    return uchar(marker != 0u && coeff_zero_coded_marker_tg(states, idx) == marker);
}

inline void coeff_set_zero_coded_marker(thread uchar *states, uint idx, uchar marker) {
    states[idx] = uchar((states[idx] & uchar(0xE0u)) | (marker & J2K_STATE_MARKER_MASK));
}

inline void coeff_set_zero_coded_marker_dev(device uchar *states, uint idx, uchar marker) {
    states[idx] = uchar((states[idx] & uchar(0xE0u)) | (marker & J2K_STATE_MARKER_MASK));
}

inline void coeff_set_zero_coded_marker_tg(threadgroup uchar *states, uint idx, uchar marker) {
    states[idx] = uchar((states[idx] & uchar(0xE0u)) | (marker & J2K_STATE_MARKER_MASK));
}

inline uchar coeff_is_magnitude_refined(thread const uchar *states, uint idx) {
    return state_bit(states, idx, J2K_MAG_REF_SHIFT);
}

inline uchar coeff_is_magnitude_refined_dev(device const uchar *states, uint idx) {
    return state_bit_dev(states, idx, J2K_MAG_REF_SHIFT);
}

inline uchar coeff_is_magnitude_refined_tg(threadgroup const uchar *states, uint idx) {
    return state_bit_tg(states, idx, J2K_MAG_REF_SHIFT);
}

inline void coeff_set_magnitude_refined(thread uchar *states, uint idx) {
    set_state_bit(states, idx, J2K_MAG_REF_SHIFT, uchar(1u));
}

inline void coeff_set_magnitude_refined_dev(device uchar *states, uint idx) {
    set_state_bit_dev(states, idx, J2K_MAG_REF_SHIFT, uchar(1u));
}

inline void coeff_set_magnitude_refined_tg(threadgroup uchar *states, uint idx) {
    set_state_bit_tg(states, idx, J2K_MAG_REF_SHIFT, uchar(1u));
}

inline void reset_contexts(thread uchar *contexts) {
    for (uint idx = 0u; idx < 19u; ++idx) {
        contexts[idx] = uchar(0);
    }
    contexts[0] = uchar(4u);
    contexts[17] = uchar(3u);
    contexts[18] = uchar(46u);
}

inline uchar zero_context_label(uchar neighbors, uint sub_band_type) {
    if (sub_band_type == 1u) {
        return ZERO_CTX_HL_LOOKUP[neighbors];
    }
    if (sub_band_type == 3u) {
        return ZERO_CTX_HH_LOOKUP[neighbors];
    }
    return ZERO_CTX_LL_LH_LOOKUP[neighbors];
}

inline uchar neighborhood_states(thread const uchar *states, uint padded_width, uint index_x, uint index_y) {
    return uchar(
        (uint(coeff_is_significant(states, coeff_index(padded_width, index_x, index_y + 1u))) << 0u) |
        (uint(coeff_is_significant(states, coeff_index(padded_width, index_x + 1u, index_y + 1u))) << 1u) |
        (uint(coeff_is_significant(states, coeff_index(padded_width, index_x + 1u, index_y))) << 2u) |
        (uint(coeff_is_significant(states, coeff_index(padded_width, index_x - 1u, index_y + 1u))) << 3u) |
        (uint(coeff_is_significant(states, coeff_index(padded_width, index_x - 1u, index_y))) << 4u) |
        (uint(coeff_is_significant(states, coeff_index(padded_width, index_x + 1u, index_y - 1u))) << 5u) |
        (uint(coeff_is_significant(states, coeff_index(padded_width, index_x, index_y - 1u))) << 6u) |
        (uint(coeff_is_significant(states, coeff_index(padded_width, index_x - 1u, index_y - 1u))) << 7u)
    );
}

inline bool neighbor_in_next_stripe(uint index_y, uint height) {
    const uint real_y = index_y - J2K_CLASSIC_PADDING;
    return real_y + 1u < height && ((real_y + 1u) >> 2u) > (real_y >> 2u);
}

inline uchar effective_neighborhood_states(
    thread const uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y,
    uint height,
    uint style_flags
) {
    uchar states_mask = neighborhood_states(states, padded_width, index_x, index_y);
    if ((style_flags & J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT) != 0u &&
        neighbor_in_next_stripe(index_y, height)) {
        states_mask &= uchar(0b11110100);
    }
    return states_mask;
}

inline void set_significant(
    thread uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    const uint idx = coeff_index(padded_width, index_x, index_y);
    set_state_bit(states, idx, J2K_SIG_SHIFT, uchar(1u));
}

inline uchar neighborhood_states_plain_dev(
    device const uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    return uchar(
        (uint(coeff_is_significant_dev(states, coeff_index(padded_width, index_x, index_y + 1u))) << 0u) |
        (uint(coeff_is_significant_dev(states, coeff_index(padded_width, index_x + 1u, index_y + 1u))) << 1u) |
        (uint(coeff_is_significant_dev(states, coeff_index(padded_width, index_x + 1u, index_y))) << 2u) |
        (uint(coeff_is_significant_dev(states, coeff_index(padded_width, index_x - 1u, index_y + 1u))) << 3u) |
        (uint(coeff_is_significant_dev(states, coeff_index(padded_width, index_x - 1u, index_y))) << 4u) |
        (uint(coeff_is_significant_dev(states, coeff_index(padded_width, index_x + 1u, index_y - 1u))) << 5u) |
        (uint(coeff_is_significant_dev(states, coeff_index(padded_width, index_x, index_y - 1u))) << 6u) |
        (uint(coeff_is_significant_dev(states, coeff_index(padded_width, index_x - 1u, index_y - 1u))) << 7u)
    );
}

inline uchar neighborhood_states_plain_tg(
    threadgroup const uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    return uchar(
        (uint(coeff_is_significant_tg(states, coeff_index(padded_width, index_x, index_y + 1u))) << 0u) |
        (uint(coeff_is_significant_tg(states, coeff_index(padded_width, index_x + 1u, index_y + 1u))) << 1u) |
        (uint(coeff_is_significant_tg(states, coeff_index(padded_width, index_x + 1u, index_y))) << 2u) |
        (uint(coeff_is_significant_tg(states, coeff_index(padded_width, index_x - 1u, index_y + 1u))) << 3u) |
        (uint(coeff_is_significant_tg(states, coeff_index(padded_width, index_x - 1u, index_y))) << 4u) |
        (uint(coeff_is_significant_tg(states, coeff_index(padded_width, index_x + 1u, index_y - 1u))) << 5u) |
        (uint(coeff_is_significant_tg(states, coeff_index(padded_width, index_x, index_y - 1u))) << 6u) |
        (uint(coeff_is_significant_tg(states, coeff_index(padded_width, index_x - 1u, index_y - 1u))) << 7u)
    );
}

inline void set_significant_plain_dev(
    device uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    const uint idx = coeff_index(padded_width, index_x, index_y);
    set_state_bit_dev(states, idx, J2K_SIG_SHIFT, uchar(1u));
}

inline void set_significant_plain_tg(
    threadgroup uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    const uint idx = coeff_index(padded_width, index_x, index_y);
    set_state_bit_tg(states, idx, J2K_SIG_SHIFT, uchar(1u));
}

inline uchar magnitude_refinement_context(
    thread const uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y,
    uint height,
    uint style_flags
) {
    const uint idx = coeff_index(padded_width, index_x, index_y);
    const uchar m1 = coeff_is_magnitude_refined(states, idx) * uchar(16u);
    const uchar m2 = uchar(14u + min(uint(effective_neighborhood_states(
        states,
        padded_width,
        index_x,
        index_y,
        height,
        style_flags
    )), 1u));
    return max(m1, m2);
}

inline uchar2 sign_context(
    thread const uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y,
    uint height,
    uint style_flags
) {
    const uchar significances =
        effective_neighborhood_states(
            states,
            padded_width,
            index_x,
            index_y,
            height,
            style_flags
        ) & uchar(0b01010101);
    const uint left_sign = coeff_sign(states, coeff_index(padded_width, index_x - 1u, index_y));
    const uint right_sign = coeff_sign(states, coeff_index(padded_width, index_x + 1u, index_y));
    const uint top_sign = coeff_sign(states, coeff_index(padded_width, index_x, index_y - 1u));
    const uint bottom_sign =
        ((style_flags & J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT) != 0u &&
            neighbor_in_next_stripe(index_y, height))
        ? 0u
        : coeff_sign(states, coeff_index(padded_width, index_x, index_y + 1u));
    const uchar signs = uchar((top_sign << 6u) | (left_sign << 4u) | (right_sign << 2u) | bottom_sign);
    const uchar negative = significances & signs;
    const uchar positive = significances & uchar(~signs);
    return SIGN_CONTEXT_LOOKUP[uchar((negative << 1u) | positive)];
}

inline uchar2 sign_context_plain_dev(
    device const uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    const uchar significances =
        neighborhood_states_plain_dev(states, padded_width, index_x, index_y) & uchar(0b01010101);
    const uint left_sign = coeff_sign_dev(states, coeff_index(padded_width, index_x - 1u, index_y));
    const uint right_sign = coeff_sign_dev(states, coeff_index(padded_width, index_x + 1u, index_y));
    const uint top_sign = coeff_sign_dev(states, coeff_index(padded_width, index_x, index_y - 1u));
    const uint bottom_sign = coeff_sign_dev(states, coeff_index(padded_width, index_x, index_y + 1u));
    const uchar signs = uchar((top_sign << 6u) | (left_sign << 4u) | (right_sign << 2u) | bottom_sign);
    const uchar negative = significances & signs;
    const uchar positive = significances & uchar(~signs);
    return SIGN_CONTEXT_LOOKUP[uchar((negative << 1u) | positive)];
}

inline uchar2 sign_context_plain_tg(
    threadgroup const uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    const uchar significances =
        neighborhood_states_plain_tg(states, padded_width, index_x, index_y) & uchar(0b01010101);
    const uint left_sign = coeff_sign_tg(states, coeff_index(padded_width, index_x - 1u, index_y));
    const uint right_sign = coeff_sign_tg(states, coeff_index(padded_width, index_x + 1u, index_y));
    const uint top_sign = coeff_sign_tg(states, coeff_index(padded_width, index_x, index_y - 1u));
    const uint bottom_sign = coeff_sign_tg(states, coeff_index(padded_width, index_x, index_y + 1u));
    const uchar signs = uchar((top_sign << 6u) | (left_sign << 4u) | (right_sign << 2u) | bottom_sign);
    const uchar negative = significances & signs;
    const uchar positive = significances & uchar(~signs);
    return SIGN_CONTEXT_LOOKUP[uchar((negative << 1u) | positive)];
}

inline uchar current_byte(thread const J2kArithmeticDecoder &decoder) {
    return decoder.base_pointer < decoder.data_len ? decoder.data[decoder.base_pointer] : uchar(0xFF);
}

inline uchar next_byte(thread const J2kArithmeticDecoder &decoder) {
    return decoder.base_pointer + 1u < decoder.data_len ? decoder.data[decoder.base_pointer + 1u] : uchar(0xFF);
}

inline void arithmetic_read_byte(thread J2kArithmeticDecoder &decoder) {
    if (current_byte(decoder) == uchar(0xFF)) {
        const uchar b1 = next_byte(decoder);
        if (b1 > uchar(0x8F)) {
            decoder.shift_count = 8u;
        } else {
            decoder.base_pointer += 1u;
            decoder.c = decoder.c + 0xFE00u - (uint(current_byte(decoder)) << 9u);
            decoder.shift_count = 7u;
        }
    } else {
        decoder.base_pointer += 1u;
        decoder.c = decoder.c + 0xFF00u - (uint(current_byte(decoder)) << 8u);
        decoder.shift_count = 8u;
    }
}

inline bool raw_read_bit(thread J2kBypassDecoder &decoder, thread uint &bit) {
    const uint byte_pos = decoder.bit_pos / 8u;
    if (byte_pos >= decoder.data_len) {
        if (decoder.strict != 0u) {
            return false;
        }
        bit = 1u;
        decoder.bit_pos += 1u;
        return true;
    }

    const uint bit_pos = decoder.bit_pos % 8u;
    bit = (uint(decoder.data[byte_pos]) >> (7u - bit_pos)) & 1u;
    decoder.bit_pos += 1u;
    return true;
}

inline bool bypass_read_bit(thread J2kBypassDecoder &decoder, thread uint &bit) {
    const uint byte_pos = decoder.bit_pos / 8u;
    const uint bit_pos = decoder.bit_pos % 8u;
    if (!raw_read_bit(decoder, bit)) {
        return false;
    }
    if (bit_pos == 7u && byte_pos < decoder.data_len && decoder.data[byte_pos] == uchar(0xFFu)) {
        uint stuffed_bit = 0u;
        if (!raw_read_bit(decoder, stuffed_bit)) {
            return decoder.strict == 0u;
        }
        if (stuffed_bit != 0u && decoder.strict != 0u) {
            return false;
        }
    }
    return true;
}

inline void arithmetic_initialize(thread J2kArithmeticDecoder &decoder) {
    decoder.c = (uint(current_byte(decoder) ^ uchar(0xFF)) << 16u);
    arithmetic_read_byte(decoder);
    decoder.c <<= 7u;
    decoder.shift_count -= 7u;
    decoder.a = 0x8000u;
}

inline void arithmetic_renormalize(thread J2kArithmeticDecoder &decoder) {
    while ((decoder.a & 0x8000u) == 0u) {
        if (decoder.shift_count == 0u) {
            arithmetic_read_byte(decoder);
        }
        decoder.a <<= 1u;
        decoder.c <<= 1u;
        decoder.shift_count -= 1u;
    }
}

inline uint arithmetic_decode_bit(thread J2kArithmeticDecoder &decoder, thread uchar *contexts, uint ctx_label) {
    uchar ctx = contexts[ctx_label];
    const J2kQeData qe = J2K_QE_TABLE[ctx & uchar(0x7F)];
    decoder.a -= qe.qe;

    if ((decoder.c >> 16u) < decoder.a) {
        if ((decoder.a & 0x8000u) != 0u) {
            return uint(ctx >> 7u);
        }

        uint d;
        if (decoder.a < qe.qe) {
            d = uint((ctx >> 7u) ^ 1u);
            if (qe.switch_mps != 0u) {
                ctx ^= uchar(0x80);
            }
            ctx = uchar((ctx & 0x80u) | qe.nlps);
        } else {
            d = uint(ctx >> 7u);
            ctx = uchar((ctx & 0x80u) | qe.nmps);
        }
        contexts[ctx_label] = ctx;
        arithmetic_renormalize(decoder);
        return d;
    }

    decoder.c -= decoder.a << 16u;

    uint d;
    if (decoder.a < qe.qe) {
        decoder.a = qe.qe;
        d = uint(ctx >> 7u);
        ctx = uchar((ctx & 0x80u) | qe.nmps);
    } else {
        decoder.a = qe.qe;
        d = uint((ctx >> 7u) ^ 1u);
        if (qe.switch_mps != 0u) {
            ctx ^= uchar(0x80);
        }
        ctx = uchar((ctx & 0x80u) | qe.nlps);
    }
    contexts[ctx_label] = ctx;
    arithmetic_renormalize(decoder);
    return d;
}

inline void decode_sign_bit(
    thread J2kArithmeticDecoder &decoder,
    thread uchar *contexts,
    thread uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y,
    uint height,
    uint style_flags
) {
    const uchar2 sign_ctx = sign_context(
        states,
        padded_width,
        index_x,
        index_y,
        height,
        style_flags
    );
    const uint sign_bit = arithmetic_decode_bit(decoder, contexts, uint(sign_ctx.x)) ^ uint(sign_ctx.y);
    coeff_set_sign(states, coeff_index(padded_width, index_x, index_y), sign_bit);
    set_significant(states, padded_width, index_x, index_y);
}

inline void decode_sign_bit_plain_dev(
    thread J2kArithmeticDecoder &decoder,
    thread uchar *contexts,
    device uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    const uchar2 sign_ctx = sign_context_plain_dev(states, padded_width, index_x, index_y);
    const uint sign_bit = arithmetic_decode_bit(decoder, contexts, uint(sign_ctx.x)) ^ uint(sign_ctx.y);
    coeff_set_sign_dev(states, coeff_index(padded_width, index_x, index_y), sign_bit);
    set_significant_plain_dev(states, padded_width, index_x, index_y);
}

inline void decode_sign_bit_plain_tg(
    thread J2kArithmeticDecoder &decoder,
    thread uchar *contexts,
    threadgroup uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    const uchar2 sign_ctx = sign_context_plain_tg(states, padded_width, index_x, index_y);
    const uint sign_bit = arithmetic_decode_bit(decoder, contexts, uint(sign_ctx.x)) ^ uint(sign_ctx.y);
    coeff_set_sign_tg(states, coeff_index(padded_width, index_x, index_y), sign_bit);
    set_significant_plain_tg(states, padded_width, index_x, index_y);
}

inline uchar magnitude_refinement_context_plain_dev(
    device const uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    const uint idx = coeff_index(padded_width, index_x, index_y);
    const uchar m1 = coeff_is_magnitude_refined_dev(states, idx) * uchar(16u);
    const uchar m2 = uchar(14u + min(uint(neighborhood_states_plain_dev(
        states,
        padded_width,
        index_x,
        index_y
    )), 1u));
    return max(m1, m2);
}

inline uchar magnitude_refinement_context_plain_tg(
    threadgroup const uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    const uint idx = coeff_index(padded_width, index_x, index_y);
    const uchar m1 = coeff_is_magnitude_refined_tg(states, idx) * uchar(16u);
    const uchar m2 = uchar(14u + min(uint(neighborhood_states_plain_tg(
        states,
        padded_width,
        index_x,
        index_y
    )), 1u));
    return max(m1, m2);
}

inline bool decode_sign_bit_bypass(
    thread J2kBypassDecoder &decoder,
    thread uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    uint sign_bit = 0u;
    if (!bypass_read_bit(decoder, sign_bit)) {
        return false;
    }
    coeff_set_sign(states, coeff_index(padded_width, index_x, index_y), sign_bit);
    set_significant(states, padded_width, index_x, index_y);
    return true;
}

inline bool decode_classic_job(
    J2kClassicCleanupBatchJob job,
    device const uchar *coded_data,
    device const J2kClassicSegment *segments,
    device uint *coefficients_scratch,
    uint scratch_offset,
    device float *output,
    device J2kClassicStatus *status
) {
    if (job.width == 0u || job.height == 0u) {
        return true;
    }
    if (job.width > J2K_CLASSIC_MAX_WIDTH || job.height > J2K_CLASSIC_MAX_HEIGHT) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 0u);
        return false;
    }
    if (job.total_bitplanes == 0u || job.total_bitplanes > 31u || job.missing_msbs >= job.total_bitplanes) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 1u);
        return false;
    }

    const uint bitplanes = job.total_bitplanes - job.missing_msbs;
    const uint max_coding_passes = bitplanes == 0u ? 0u : 1u + 3u * (bitplanes - 1u);
    if (job.coded_len == 0u || max_coding_passes == 0u || job.number_of_coding_passes == 0u) {
        return true;
    }
    if (job.number_of_coding_passes > max_coding_passes) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 2u);
        return false;
    }
    const uint padded_width = job.width + J2K_CLASSIC_PADDING * 2u;
    const uint padded_height = job.height + J2K_CLASSIC_PADDING * 2u;
    const uint coeff_count = padded_width * padded_height;

    device uint *coefficients = coefficients_scratch + scratch_offset;
    thread uchar states[J2K_CLASSIC_MAX_COEFF_COUNT];
    for (uint idx = 0u; idx < coeff_count; ++idx) {
        coefficients[idx] = 0u;
        states[idx] = uchar(0);
    }

    thread uchar contexts[19];
    for (uint idx = 0u; idx < 19u; ++idx) {
        contexts[idx] = uchar(0);
    }
    contexts[0] = uchar(4u);
    contexts[17] = uchar(3u);
    contexts[18] = uchar(46u);

    if (job.segment_count == 0u) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 3u);
        return false;
    }

    const ulong coded_begin = ulong(job.coded_offset);
    const ulong coded_end = coded_begin + ulong(job.coded_len);
    uint expected_start = 0u;
    uint expected_offset = job.coded_offset;
    for (uint segment_idx = 0u; segment_idx < job.segment_count; ++segment_idx) {
        const J2kClassicSegment segment = segments[job.segment_offset + segment_idx];
        if (segment.start_coding_pass != expected_start || segment.start_coding_pass > segment.end_coding_pass) {
            set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 4u);
            return false;
        }
        if (segment.data_offset != expected_offset) {
            set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 6u);
            return false;
        }
        const ulong segment_end = ulong(segment.data_offset) + ulong(segment.data_length);
        if (ulong(segment.data_offset) < coded_begin || segment_end > coded_end) {
            set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 7u);
            return false;
        }
        expected_start = segment.end_coding_pass;
        expected_offset = segment.data_offset + segment.data_length;

        if (segment.start_coding_pass == segment.end_coding_pass) {
            continue;
        }

        J2kArithmeticDecoder decoder;
        J2kBypassDecoder bypass_decoder;
        const bool use_arithmetic = segment.use_arithmetic != 0u;
        if (use_arithmetic) {
            decoder.data = coded_data + segment.data_offset;
            decoder.data_len = segment.data_length;
            decoder.c = 0u;
            decoder.a = 0u;
            decoder.base_pointer = 0u;
            decoder.shift_count = 0u;
            arithmetic_initialize(decoder);
        } else {
            bypass_decoder.data = coded_data + segment.data_offset;
            bypass_decoder.data_len = segment.data_length;
            bypass_decoder.bit_pos = 0u;
            bypass_decoder.strict = job.strict;
        }

        uchar zero_coded_epoch = uchar((segment.start_coding_pass + 2u) / 3u);
        for (uint coding_pass = segment.start_coding_pass; coding_pass < segment.end_coding_pass; ++coding_pass) {
            const uint current_bitplane = (coding_pass + 2u) / 3u;
            const uint current_bit_position = bitplanes - 1u - current_bitplane;
            const uint pass_type = coding_pass % 3u;

            for (uint base_row = 0u; base_row < job.height; base_row += 4u) {
                const uint stripe_end = min(base_row + 4u, job.height);
                for (uint x = 0u; x < job.width; ++x) {
                    uint index_x = x + J2K_CLASSIC_PADDING;
                    uint index_y = base_row + J2K_CLASSIC_PADDING;
                    while (index_y < stripe_end + J2K_CLASSIC_PADDING) {
                        const uint idx = coeff_index(padded_width, index_x, index_y);
                        if (pass_type == 0u) {
                            if (!use_arithmetic) {
                                set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 5u);
                                return false;
                            }
                            if (coeff_is_significant(states, idx) == 0u &&
                                coeff_is_zero_coded(states, idx, zero_coded_epoch) == 0u) {
                                const bool use_rl =
                                    ((index_y - J2K_CLASSIC_PADDING) % 4u) == 0u &&
                                    (job.height - (index_y - J2K_CLASSIC_PADDING)) >= 4u &&
                                    effective_neighborhood_states(states, padded_width, index_x, index_y, job.height, job.style_flags) == 0u &&
                                    effective_neighborhood_states(states, padded_width, index_x, index_y + 1u, job.height, job.style_flags) == 0u &&
                                    effective_neighborhood_states(states, padded_width, index_x, index_y + 2u, job.height, job.style_flags) == 0u &&
                                    effective_neighborhood_states(states, padded_width, index_x, index_y + 3u, job.height, job.style_flags) == 0u;

                                uint bit = 0u;
                                if (use_rl) {
                                    bit = arithmetic_decode_bit(decoder, contexts, 17u);
                                    if (bit == 0u) {
                                        index_y += 4u;
                                        continue;
                                    }

                                    uint num_zeroes = arithmetic_decode_bit(decoder, contexts, 18u);
                                    num_zeroes = (num_zeroes << 1u) | arithmetic_decode_bit(decoder, contexts, 18u);
                                    index_y += num_zeroes;
                                } else {
                                    const uchar ctx_label = zero_context_label(
                                        effective_neighborhood_states(
                                            states,
                                            padded_width,
                                            index_x,
                                            index_y,
                                            job.height,
                                            job.style_flags
                                        ),
                                        job.sub_band_type
                                    );
                                    bit = arithmetic_decode_bit(decoder, contexts, uint(ctx_label));
                                }

                                if (bit == 1u) {
                                    coeff_push_bit(coefficients, coeff_index(padded_width, index_x, index_y), 1u, current_bit_position);
                                    decode_sign_bit(
                                        decoder,
                                        contexts,
                                        states,
                                        padded_width,
                                        index_x,
                                        index_y,
                                        job.height,
                                        job.style_flags
                                    );
                                }
                            }
                        } else if (pass_type == 1u) {
                            if (coeff_is_significant(states, idx) == 0u &&
                                effective_neighborhood_states(
                                    states,
                                    padded_width,
                                    index_x,
                                    index_y,
                                    job.height,
                                    job.style_flags
                                ) != 0u) {
                                const uchar ctx_label = zero_context_label(
                                    effective_neighborhood_states(
                                        states,
                                        padded_width,
                                        index_x,
                                        index_y,
                                        job.height,
                                        job.style_flags
                                    ),
                                    job.sub_band_type
                                );
                                uint bit = 0u;
                                if (use_arithmetic) {
                                    bit = arithmetic_decode_bit(decoder, contexts, uint(ctx_label));
                                } else if (!bypass_read_bit(bypass_decoder, bit)) {
                                    set_classic_status(status, J2K_CLASSIC_STATUS_FAIL, 11u);
                                    return false;
                                }
                                coeff_set_zero_coded_marker(states, idx, zero_coded_epoch);
                                if (bit == 1u) {
                                    coeff_push_bit(coefficients, idx, 1u, current_bit_position);
                                    if (use_arithmetic) {
                                        decode_sign_bit(
                                            decoder,
                                            contexts,
                                            states,
                                            padded_width,
                                            index_x,
                                            index_y,
                                            job.height,
                                            job.style_flags
                                        );
                                    } else if (!decode_sign_bit_bypass(
                                        bypass_decoder,
                                        states,
                                        padded_width,
                                        index_x,
                                        index_y
                                    )) {
                                        set_classic_status(status, J2K_CLASSIC_STATUS_FAIL, 12u);
                                        return false;
                                    }
                                }
                            }
                        } else {
                            if (coeff_is_significant(states, idx) != 0u &&
                                coeff_is_zero_coded(states, idx, zero_coded_epoch) == 0u) {
                                const uchar ctx_label = magnitude_refinement_context(
                                    states,
                                    padded_width,
                                    index_x,
                                    index_y,
                                    job.height,
                                    job.style_flags
                                );
                                uint bit = 0u;
                                if (use_arithmetic) {
                                    bit = arithmetic_decode_bit(decoder, contexts, uint(ctx_label));
                                } else if (!bypass_read_bit(bypass_decoder, bit)) {
                                    set_classic_status(status, J2K_CLASSIC_STATUS_FAIL, 13u);
                                    return false;
                                }
                                if (bit == 1u) {
                                    coeff_push_bit(coefficients, idx, 1u, current_bit_position);
                                }
                                coeff_set_magnitude_refined(states, idx);
                            }
                        }

                        index_y += 1u;
                    }
                }
            }

            if (pass_type == 0u) {
                if ((job.style_flags & J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS) != 0u) {
                    const uint b0 = arithmetic_decode_bit(decoder, contexts, 18u);
                    const uint b1 = arithmetic_decode_bit(decoder, contexts, 18u);
                    const uint b2 = arithmetic_decode_bit(decoder, contexts, 18u);
                    const uint b3 = arithmetic_decode_bit(decoder, contexts, 18u);
                    if ((b0 != 1u || b1 != 0u || b2 != 1u || b3 != 0u) && job.strict != 0u) {
                        set_classic_status(status, J2K_CLASSIC_STATUS_FAIL, 10u);
                        return false;
                    }
                }
                zero_coded_epoch = uchar(min(uint(zero_coded_epoch) + 1u, uint(J2K_STATE_MARKER_MASK)));
            }

            if ((job.style_flags & J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES) != 0u) {
                reset_contexts(contexts);
            }
        }
    }

    if (expected_start != job.number_of_coding_passes || expected_offset != job.coded_offset + job.coded_len) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 8u);
        return false;
    }

    for (uint y = 0u; y < job.height; ++y) {
        const uint output_row = job.output_offset + y * job.output_stride;
        for (uint x = 0u; x < job.width; ++x) {
            const uint coeff = coefficients[coeff_index(padded_width, x + J2K_CLASSIC_PADDING, y + J2K_CLASSIC_PADDING)];
            int magnitude = int(coeff & 0x7FFFFFFFu);
            if (coeff_sign(states, coeff_index(padded_width, x + J2K_CLASSIC_PADDING, y + J2K_CLASSIC_PADDING)) != 0u) {
                magnitude = -magnitude;
            }
            output[output_row + x] = float(magnitude) * job.dequantization_step;
        }
    }

    return true;
}

inline bool decode_classic_job_plain(
    J2kClassicCleanupBatchJob job,
    device const uchar *coded_data,
    device const J2kClassicSegment *segments,
    device uint *coefficients_scratch,
    uint scratch_offset,
    threadgroup uchar *states,
    device float *output,
    device J2kClassicStatus *status
) {
    if (job.width == 0u || job.height == 0u) {
        return true;
    }
    if (job.style_flags != 0u) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 12u);
        return false;
    }
    if (job.width > J2K_CLASSIC_MAX_WIDTH || job.height > J2K_CLASSIC_MAX_HEIGHT) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 0u);
        return false;
    }
    if (job.total_bitplanes == 0u || job.total_bitplanes > 31u || job.missing_msbs >= job.total_bitplanes) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 1u);
        return false;
    }

    const uint bitplanes = job.total_bitplanes - job.missing_msbs;
    const uint max_coding_passes = bitplanes == 0u ? 0u : 1u + 3u * (bitplanes - 1u);
    if (job.coded_len == 0u || max_coding_passes == 0u || job.number_of_coding_passes == 0u) {
        return true;
    }
    if (job.number_of_coding_passes > max_coding_passes) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 2u);
        return false;
    }

    const uint padded_width = job.width + J2K_CLASSIC_PADDING * 2u;
    device uint *coefficients = coefficients_scratch + scratch_offset;

    thread uchar contexts[19];
    reset_contexts(contexts);

    if (job.segment_count == 0u) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 3u);
        return false;
    }

    const ulong coded_begin = ulong(job.coded_offset);
    const ulong coded_end = coded_begin + ulong(job.coded_len);
    uint expected_start = 0u;
    uint expected_offset = job.coded_offset;
    for (uint segment_idx = 0u; segment_idx < job.segment_count; ++segment_idx) {
        const J2kClassicSegment segment = segments[job.segment_offset + segment_idx];
        if (segment.use_arithmetic == 0u) {
            set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 5u);
            return false;
        }
        if (segment.start_coding_pass != expected_start || segment.start_coding_pass > segment.end_coding_pass) {
            set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 4u);
            return false;
        }
        if (segment.data_offset != expected_offset) {
            set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 6u);
            return false;
        }
        const ulong segment_end = ulong(segment.data_offset) + ulong(segment.data_length);
        if (ulong(segment.data_offset) < coded_begin || segment_end > coded_end) {
            set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 7u);
            return false;
        }
        expected_start = segment.end_coding_pass;
        expected_offset = segment.data_offset + segment.data_length;

        if (segment.start_coding_pass == segment.end_coding_pass) {
            continue;
        }

        J2kArithmeticDecoder decoder;
        decoder.data = coded_data + segment.data_offset;
        decoder.data_len = segment.data_length;
        decoder.c = 0u;
        decoder.a = 0u;
        decoder.base_pointer = 0u;
        decoder.shift_count = 0u;
        arithmetic_initialize(decoder);

        uchar zero_coded_epoch = uchar((segment.start_coding_pass + 2u) / 3u);
        for (uint coding_pass = segment.start_coding_pass; coding_pass < segment.end_coding_pass; ++coding_pass) {
            const uint current_bitplane = (coding_pass + 2u) / 3u;
            const uint current_bit_position = bitplanes - 1u - current_bitplane;
            const uint pass_type = coding_pass % 3u;

            for (uint base_row = 0u; base_row < job.height; base_row += 4u) {
                const uint stripe_end = min(base_row + 4u, job.height);
                for (uint x = 0u; x < job.width; ++x) {
                    const uint index_x = x + J2K_CLASSIC_PADDING;
                    uint index_y = base_row + J2K_CLASSIC_PADDING;
                    while (index_y < stripe_end + J2K_CLASSIC_PADDING) {
                        const uint idx = coeff_index(padded_width, index_x, index_y);
                        if (pass_type == 0u) {
                            if (coeff_is_significant_tg(states, idx) == 0u &&
                                coeff_is_zero_coded_tg(states, idx, zero_coded_epoch) == 0u) {
                                const bool use_rl =
                                    ((index_y - J2K_CLASSIC_PADDING) % 4u) == 0u &&
                                    (job.height - (index_y - J2K_CLASSIC_PADDING)) >= 4u &&
                                    neighborhood_states_plain_tg(states, padded_width, index_x, index_y) == 0u &&
                                    neighborhood_states_plain_tg(states, padded_width, index_x, index_y + 1u) == 0u &&
                                    neighborhood_states_plain_tg(states, padded_width, index_x, index_y + 2u) == 0u &&
                                    neighborhood_states_plain_tg(states, padded_width, index_x, index_y + 3u) == 0u;

                                uint bit = 0u;
                                if (use_rl) {
                                    bit = arithmetic_decode_bit(decoder, contexts, 17u);
                                    if (bit == 0u) {
                                        index_y += 4u;
                                        continue;
                                    }

                                    uint num_zeroes = arithmetic_decode_bit(decoder, contexts, 18u);
                                    num_zeroes = (num_zeroes << 1u) | arithmetic_decode_bit(decoder, contexts, 18u);
                                    index_y += num_zeroes;
                                } else {
                                    const uchar ctx_label = zero_context_label(
                                        neighborhood_states_plain_tg(states, padded_width, index_x, index_y),
                                        job.sub_band_type
                                    );
                                    bit = arithmetic_decode_bit(decoder, contexts, uint(ctx_label));
                                }

                                if (bit == 1u) {
                                    coeff_push_bit(coefficients, coeff_index(padded_width, index_x, index_y), 1u, current_bit_position);
                                    decode_sign_bit_plain_tg(
                                        decoder,
                                        contexts,
                                        states,
                                        padded_width,
                                        index_x,
                                        index_y
                                    );
                                }
                            }
                        } else if (pass_type == 1u) {
                            if (coeff_is_significant_tg(states, idx) == 0u &&
                                neighborhood_states_plain_tg(states, padded_width, index_x, index_y) != 0u) {
                                const uchar ctx_label = zero_context_label(
                                    neighborhood_states_plain_tg(states, padded_width, index_x, index_y),
                                    job.sub_band_type
                                );
                                const uint bit = arithmetic_decode_bit(decoder, contexts, uint(ctx_label));
                                coeff_set_zero_coded_marker_tg(states, idx, zero_coded_epoch);
                                if (bit == 1u) {
                                    coeff_push_bit(coefficients, idx, 1u, current_bit_position);
                                    decode_sign_bit_plain_tg(
                                        decoder,
                                        contexts,
                                        states,
                                        padded_width,
                                        index_x,
                                        index_y
                                    );
                                }
                            }
                        } else {
                            if (coeff_is_significant_tg(states, idx) != 0u &&
                                coeff_is_zero_coded_tg(states, idx, zero_coded_epoch) == 0u) {
                                const uchar ctx_label = magnitude_refinement_context_plain_tg(
                                    states,
                                    padded_width,
                                    index_x,
                                    index_y
                                );
                                const uint bit = arithmetic_decode_bit(decoder, contexts, uint(ctx_label));
                                if (bit == 1u) {
                                    coeff_push_bit(coefficients, idx, 1u, current_bit_position);
                                }
                                coeff_set_magnitude_refined_tg(states, idx);
                            }
                        }

                        index_y += 1u;
                    }
                }
            }

            if (pass_type == 0u) {
                zero_coded_epoch = uchar(min(uint(zero_coded_epoch) + 1u, uint(J2K_STATE_MARKER_MASK)));
            }
        }
    }

    if (expected_start != job.number_of_coding_passes || expected_offset != job.coded_offset + job.coded_len) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 8u);
        return false;
    }

    for (uint y = 0u; y < job.height; ++y) {
        const uint output_row = job.output_offset + y * job.output_stride;
        for (uint x = 0u; x < job.width; ++x) {
            const uint coeff = coefficients[coeff_index(padded_width, x + J2K_CLASSIC_PADDING, y + J2K_CLASSIC_PADDING)];
            int magnitude = int(coeff & 0x7FFFFFFFu);
            if (coeff_sign_tg(states, coeff_index(padded_width, x + J2K_CLASSIC_PADDING, y + J2K_CLASSIC_PADDING)) != 0u) {
                magnitude = -magnitude;
            }
            output[output_row + x] = float(magnitude) * job.dequantization_step;
        }
    }

    return true;
}

kernel void j2k_decode_classic_cleanup_batched(
    device const uchar *coded_data [[buffer(0)]],
    device float *output [[buffer(1)]],
    device const J2kClassicCleanupBatchJob *jobs [[buffer(2)]],
    device const J2kClassicSegment *segments [[buffer(3)]],
    device J2kClassicStatus *statuses [[buffer(4)]],
    device uint *coefficients_scratch [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    device J2kClassicStatus *status = statuses + gid;
    set_classic_status(status, J2K_CLASSIC_STATUS_OK, 0u);
    if (!decode_classic_job(
            jobs[gid],
            coded_data,
            segments,
            coefficients_scratch,
            gid * J2K_CLASSIC_MAX_COEFF_COUNT,
            output,
            status
        ) &&
        status->code == J2K_CLASSIC_STATUS_OK) {
        set_classic_status(status, J2K_CLASSIC_STATUS_FAIL, 0u);
    }
}

kernel void j2k_decode_classic_cleanup_plain_batched(
    device const uchar *coded_data [[buffer(0)]],
    device float *output [[buffer(1)]],
    device const J2kClassicCleanupBatchJob *jobs [[buffer(2)]],
    device const J2kClassicSegment *segments [[buffer(3)]],
    device J2kClassicStatus *statuses [[buffer(4)]],
    device uint *coefficients_scratch [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    threadgroup uchar shared_states[J2K_CLASSIC_MAX_COEFF_COUNT];
    device J2kClassicStatus *status = statuses + gid;
    const J2kClassicCleanupBatchJob job = jobs[gid];
    const uint padded_width = job.width + J2K_CLASSIC_PADDING * 2u;
    const uint coeff_count = padded_width * (job.height + J2K_CLASSIC_PADDING * 2u);
    device uint *coefficients = coefficients_scratch + gid * J2K_CLASSIC_MAX_COEFF_COUNT;

    for (uint idx = lane; idx < coeff_count; idx += 32u) {
        coefficients[idx] = 0u;
        shared_states[idx] = uchar(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    set_classic_status(status, J2K_CLASSIC_STATUS_OK, 0u);
    if (lane == 0u) {
        if (!decode_classic_job_plain(
                job,
                coded_data,
                segments,
                coefficients_scratch,
                gid * J2K_CLASSIC_MAX_COEFF_COUNT,
                shared_states,
                output,
                status
            ) &&
            status->code == J2K_CLASSIC_STATUS_OK) {
            set_classic_status(status, J2K_CLASSIC_STATUS_FAIL, 0u);
        }
    }
}
