#include <metal_stdlib>
using namespace metal;

constant uint J2K_ENCODE_STATUS_OK = 0u;
constant uint J2K_ENCODE_STATUS_FAIL = 1u;
constant uint J2K_ENCODE_STATUS_UNSUPPORTED = 2u;

constant uchar J2K_ENCODE_SIGNIFICANT = uchar(1u << 7u);
constant uchar J2K_ENCODE_MAGNITUDE_REFINED = uchar(1u << 6u);
constant uchar J2K_ENCODE_SIGN = uchar(1u << 5u);

struct J2kClassicEncodeParams {
    uint width;
    uint height;
    uint sub_band_type;
    uint total_bitplanes;
    uint style_flags;
    uint output_capacity;
    uint segment_capacity;
};

struct J2kClassicEncodeStatus {
    uint code;
    uint detail;
    uint data_len;
    uint number_of_coding_passes;
    uint missing_bit_planes;
    uint segment_count;
    uint reserved0;
    uint reserved1;
};

struct J2kMqEncoder {
    device uchar *data;
    uint max_len;
    uint len;
    uint a;
    uint c;
    uint ct;
    uint failed;
};

struct J2kRawBitWriter {
    device uchar *data;
    uint max_len;
    uint len;
    uint buffer;
    uint bits_in_buffer;
    uint last_byte_was_ff;
    uint failed;
};

inline void j2k_set_encode_status(
    device J2kClassicEncodeStatus *status,
    uint code,
    uint detail,
    uint data_len,
    uint passes,
    uint missing,
    uint segments
) {
    status->code = code;
    status->detail = detail;
    status->data_len = data_len;
    status->number_of_coding_passes = passes;
    status->missing_bit_planes = missing;
    status->segment_count = segments;
    status->reserved0 = 0u;
    status->reserved1 = 0u;
}

inline void j2k_mq_init(thread J2kMqEncoder &encoder, device uchar *out, uint capacity) {
    encoder.data = out;
    encoder.max_len = capacity;
    encoder.len = 0u;
    encoder.a = 0x8000u;
    encoder.c = 0u;
    encoder.ct = 12u;
    encoder.failed = 0u;
    if (capacity == 0u) {
        encoder.failed = 1u;
        return;
    }
    encoder.data[0] = uchar(0);
    encoder.len = 1u;
}

inline void j2k_mq_push(thread J2kMqEncoder &encoder, uchar value) {
    if (encoder.len >= encoder.max_len) {
        encoder.failed = 1u;
        return;
    }
    encoder.data[encoder.len] = value;
    encoder.len += 1u;
}

inline void j2k_mq_byte_out(thread J2kMqEncoder &encoder) {
    if (encoder.failed != 0u || encoder.len == 0u) {
        encoder.failed = 1u;
        return;
    }

    uchar last_byte = encoder.data[encoder.len - 1u];
    if (last_byte == uchar(0xFFu)) {
        const uchar b = uchar(encoder.c >> 20u);
        j2k_mq_push(encoder, b);
        encoder.c &= 0xFFFFFu;
        encoder.ct = 7u;
    } else if ((encoder.c & 0x8000000u) == 0u) {
        const uchar b = uchar(encoder.c >> 19u);
        j2k_mq_push(encoder, b);
        encoder.c &= 0x7FFFFu;
        encoder.ct = 8u;
    } else {
        encoder.data[encoder.len - 1u] = uchar(encoder.data[encoder.len - 1u] + uchar(1u));
        encoder.c &= 0x7FFFFFFu;
        if (encoder.data[encoder.len - 1u] == uchar(0xFFu)) {
            const uchar b = uchar(encoder.c >> 20u);
            j2k_mq_push(encoder, b);
            encoder.c &= 0xFFFFFu;
            encoder.ct = 7u;
        } else {
            const uchar b = uchar(encoder.c >> 19u);
            j2k_mq_push(encoder, b);
            encoder.c &= 0x7FFFFu;
            encoder.ct = 8u;
        }
    }
}

inline void j2k_mq_renormalize(thread J2kMqEncoder &encoder) {
    do {
        encoder.a <<= 1u;
        encoder.c <<= 1u;
        encoder.ct -= 1u;
        if (encoder.ct == 0u) {
            j2k_mq_byte_out(encoder);
        }
    } while ((encoder.a & 0x8000u) == 0u && encoder.failed == 0u);
}

inline void j2k_mq_encode(
    thread J2kMqEncoder &encoder,
    thread uchar *contexts,
    uint ctx_label,
    uint bit
) {
    uchar ctx = contexts[ctx_label];
    const J2kQeData qe = J2K_QE_TABLE[ctx & uchar(0x7Fu)];
    const uint mps = uint(ctx >> 7u);
    encoder.a -= qe.qe;

    if (bit == mps) {
        if ((encoder.a & 0x8000u) != 0u) {
            encoder.c += qe.qe;
            return;
        }
        if (encoder.a < qe.qe) {
            encoder.a = qe.qe;
        } else {
            encoder.c += qe.qe;
        }
        ctx = uchar((ctx & uchar(0x80u)) | qe.nmps);
    } else {
        if (encoder.a < qe.qe) {
            encoder.c += qe.qe;
        } else {
            encoder.a = qe.qe;
        }
        if (qe.switch_mps != 0u) {
            ctx ^= uchar(0x80u);
        }
        ctx = uchar((ctx & uchar(0x80u)) | qe.nlps);
    }

    contexts[ctx_label] = ctx;
    j2k_mq_renormalize(encoder);
}

inline void j2k_mq_set_bits(thread J2kMqEncoder &encoder) {
    const uint temp = encoder.c + encoder.a;
    encoder.c |= 0xFFFFu;
    if (encoder.c >= temp) {
        encoder.c -= 0x8000u;
    }
}

inline void j2k_mq_finish(thread J2kMqEncoder &encoder) {
    j2k_mq_set_bits(encoder);
    encoder.c <<= encoder.ct;
    j2k_mq_byte_out(encoder);
    encoder.c <<= encoder.ct;
    j2k_mq_byte_out(encoder);
}

inline void j2k_raw_writer_init(thread J2kRawBitWriter &writer, device uchar *out, uint capacity) {
    writer.data = out;
    writer.max_len = capacity;
    writer.len = 0u;
    writer.buffer = 0u;
    writer.bits_in_buffer = 0u;
    writer.last_byte_was_ff = 0u;
    writer.failed = 0u;
}

inline void j2k_raw_writer_push(thread J2kRawBitWriter &writer, uchar value) {
    if (writer.len >= writer.max_len) {
        writer.failed = 1u;
        return;
    }
    writer.data[writer.len] = value;
    writer.len += 1u;
    writer.last_byte_was_ff = value == uchar(0xFFu) ? 1u : 0u;
}

inline void j2k_raw_writer_flush_byte(thread J2kRawBitWriter &writer) {
    const uint limit = writer.last_byte_was_ff != 0u ? 7u : 8u;
    const uchar byte = uchar(writer.buffer >> (writer.bits_in_buffer - limit));
    j2k_raw_writer_push(writer, byte);
    writer.bits_in_buffer -= limit;
    writer.buffer &= writer.bits_in_buffer == 0u ? 0u : ((1u << writer.bits_in_buffer) - 1u);
}

inline void j2k_raw_writer_write_bit(thread J2kRawBitWriter &writer, uint bit) {
    writer.buffer = (writer.buffer << 1u) | (bit & 1u);
    writer.bits_in_buffer += 1u;
    const uint limit = writer.last_byte_was_ff != 0u ? 7u : 8u;
    if (writer.bits_in_buffer >= limit) {
        j2k_raw_writer_flush_byte(writer);
    }
}

inline void j2k_raw_writer_finish(thread J2kRawBitWriter &writer) {
    if (writer.bits_in_buffer == 0u) {
        return;
    }
    const uint limit = writer.last_byte_was_ff != 0u ? 7u : 8u;
    const uint shift = limit - writer.bits_in_buffer;
    j2k_raw_writer_push(writer, uchar(writer.buffer << shift));
    writer.buffer = 0u;
    writer.bits_in_buffer = 0u;
}

inline uint j2k_classic_magnitude(int value) {
    return value < 0 ? uint(-value) : uint(value);
}

inline uchar j2k_classic_effective_neighbors(
    thread const uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y,
    uint height,
    uint style_flags
) {
    return effective_neighborhood_states(states, padded_width, index_x, index_y, height, style_flags);
}

inline void j2k_classic_encode_sign(
    uint idx,
    thread const uchar *states,
    thread J2kMqEncoder &encoder,
    thread uchar *contexts,
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
    const uint sign_bit = (uint(states[idx]) >> 5u) & 1u;
    j2k_mq_encode(encoder, contexts, uint(sign_ctx.x), sign_bit ^ uint(sign_ctx.y));
}

inline void j2k_classic_significance_pass(
    thread const uint *magnitudes,
    thread uchar *states,
    thread uchar *coded,
    thread J2kMqEncoder &encoder,
    thread uchar *contexts,
    uint width,
    uint height,
    uint padded_width,
    uint bit_mask,
    uint sub_band_type,
    uint style_flags
) {
    for (uint y_base = 0u; y_base < height; y_base += 4u) {
        for (uint x = 0u; x < width; ++x) {
            const uint y_end = min(y_base + 4u, height);
            for (uint y = y_base; y < y_end; ++y) {
                const uint ix = x + 1u;
                const uint iy = y + 1u;
                const uint idx = coeff_index(padded_width, ix, iy);
                const uchar neighbor_sig =
                    j2k_classic_effective_neighbors(states, padded_width, ix, iy, height, style_flags);
                if ((states[idx] & J2K_ENCODE_SIGNIFICANT) == 0u && neighbor_sig != 0u) {
                    const uint ctx_label = uint(zero_context_label(neighbor_sig, sub_band_type));
                    const uint bit = (magnitudes[idx] & bit_mask) != 0u ? 1u : 0u;
                    j2k_mq_encode(encoder, contexts, ctx_label, bit);
                    coded[idx] = uchar(1u);
                    if (bit != 0u) {
                        j2k_classic_encode_sign(
                            idx,
                            states,
                            encoder,
                            contexts,
                            padded_width,
                            ix,
                            iy,
                            height,
                            style_flags
                        );
                        set_significant(states, padded_width, ix, iy);
                    }
                }
            }
        }
    }
}

inline void j2k_classic_magnitude_refinement_pass(
    thread const uint *magnitudes,
    thread uchar *states,
    thread uchar *coded,
    thread J2kMqEncoder &encoder,
    thread uchar *contexts,
    uint width,
    uint height,
    uint padded_width,
    uint bit_mask,
    uint style_flags
) {
    for (uint y_base = 0u; y_base < height; y_base += 4u) {
        for (uint x = 0u; x < width; ++x) {
            const uint y_end = min(y_base + 4u, height);
            for (uint y = y_base; y < y_end; ++y) {
                const uint ix = x + 1u;
                const uint iy = y + 1u;
                const uint idx = coeff_index(padded_width, ix, iy);
                if ((states[idx] & J2K_ENCODE_SIGNIFICANT) != 0u && coded[idx] == 0u) {
                    const uint ctx_label =
                        uint(magnitude_refinement_context(states, padded_width, ix, iy, height, style_flags));
                    const uint bit = (magnitudes[idx] & bit_mask) != 0u ? 1u : 0u;
                    j2k_mq_encode(encoder, contexts, ctx_label, bit);
                    states[idx] |= J2K_ENCODE_MAGNITUDE_REFINED;
                }
            }
        }
    }
}

inline void j2k_classic_significance_pass_raw(
    thread const uint *magnitudes,
    thread uchar *states,
    thread uchar *coded,
    thread J2kRawBitWriter &writer,
    uint width,
    uint height,
    uint padded_width,
    uint bit_mask,
    uint style_flags
) {
    for (uint y_base = 0u; y_base < height; y_base += 4u) {
        for (uint x = 0u; x < width; ++x) {
            const uint y_end = min(y_base + 4u, height);
            for (uint y = y_base; y < y_end; ++y) {
                const uint ix = x + 1u;
                const uint iy = y + 1u;
                const uint idx = coeff_index(padded_width, ix, iy);
                const uchar neighbor_sig =
                    j2k_classic_effective_neighbors(states, padded_width, ix, iy, height, style_flags);
                if ((states[idx] & J2K_ENCODE_SIGNIFICANT) == 0u && neighbor_sig != 0u) {
                    const uint bit = (magnitudes[idx] & bit_mask) != 0u ? 1u : 0u;
                    j2k_raw_writer_write_bit(writer, bit);
                    coded[idx] = uchar(1u);
                    if (bit != 0u) {
                        j2k_raw_writer_write_bit(writer, (uint(states[idx]) >> 5u) & 1u);
                        set_significant(states, padded_width, ix, iy);
                    }
                }
            }
        }
    }
}

inline void j2k_classic_magnitude_refinement_pass_raw(
    thread const uint *magnitudes,
    thread uchar *states,
    thread uchar *coded,
    thread J2kRawBitWriter &writer,
    uint width,
    uint height,
    uint padded_width,
    uint bit_mask
) {
    for (uint y_base = 0u; y_base < height; y_base += 4u) {
        for (uint x = 0u; x < width; ++x) {
            const uint y_end = min(y_base + 4u, height);
            for (uint y = y_base; y < y_end; ++y) {
                const uint ix = x + 1u;
                const uint iy = y + 1u;
                const uint idx = coeff_index(padded_width, ix, iy);
                if ((states[idx] & J2K_ENCODE_SIGNIFICANT) != 0u && coded[idx] == 0u) {
                    const uint bit = (magnitudes[idx] & bit_mask) != 0u ? 1u : 0u;
                    j2k_raw_writer_write_bit(writer, bit);
                    states[idx] |= J2K_ENCODE_MAGNITUDE_REFINED;
                }
            }
        }
    }
}

inline void j2k_classic_cleanup_pass(
    thread const uint *magnitudes,
    thread uchar *states,
    thread uchar *coded,
    thread J2kMqEncoder &encoder,
    thread uchar *contexts,
    uint width,
    uint height,
    uint padded_width,
    uint bit_mask,
    uint sub_band_type,
    uint style_flags
) {
    for (uint y_base = 0u; y_base < height; y_base += 4u) {
        for (uint x = 0u; x < width; ++x) {
            const uint y_end = min(y_base + 4u, height);
            const uint stripe_height = y_end - y_base;

            if (stripe_height == 4u) {
                bool all_zero_uncoded = true;
                for (uint y = y_base; y < y_end; ++y) {
                    const uint ix = x + 1u;
                    const uint iy = y + 1u;
                    const uint idx = coeff_index(padded_width, ix, iy);
                    const uchar neighbor_sig =
                        j2k_classic_effective_neighbors(states, padded_width, ix, iy, height, style_flags);
                    if ((states[idx] & J2K_ENCODE_SIGNIFICANT) != 0u || coded[idx] != 0u || neighbor_sig != 0u) {
                        all_zero_uncoded = false;
                        break;
                    }
                }

                if (all_zero_uncoded) {
                    uint first_sig = 4u;
                    for (uint pos = 0u; pos < 4u; ++pos) {
                        const uint idx = coeff_index(padded_width, x + 1u, y_base + pos + 1u);
                        if ((magnitudes[idx] & bit_mask) != 0u) {
                            first_sig = pos;
                            break;
                        }
                    }

                    if (first_sig < 4u) {
                        j2k_mq_encode(encoder, contexts, 17u, 1u);
                        j2k_mq_encode(encoder, contexts, 18u, (first_sig >> 1u) & 1u);
                        j2k_mq_encode(encoder, contexts, 18u, first_sig & 1u);

                        const uint sig_y = y_base + first_sig;
                        const uint sig_idx = coeff_index(padded_width, x + 1u, sig_y + 1u);
                        j2k_classic_encode_sign(
                            sig_idx,
                            states,
                            encoder,
                            contexts,
                            padded_width,
                            x + 1u,
                            sig_y + 1u,
                            height,
                            style_flags
                        );
                        set_significant(states, padded_width, x + 1u, sig_y + 1u);

                        for (uint y = sig_y + 1u; y < y_end; ++y) {
                            const uint ix = x + 1u;
                            const uint iy = y + 1u;
                            const uint idx = coeff_index(padded_width, ix, iy);
                            if ((states[idx] & J2K_ENCODE_SIGNIFICANT) == 0u && coded[idx] == 0u) {
                                const uchar neighbor_sig =
                                    j2k_classic_effective_neighbors(
                                        states,
                                        padded_width,
                                        ix,
                                        iy,
                                        height,
                                        style_flags
                                    );
                                const uint ctx_label = uint(zero_context_label(neighbor_sig, sub_band_type));
                                const uint bit = (magnitudes[idx] & bit_mask) != 0u ? 1u : 0u;
                                j2k_mq_encode(encoder, contexts, ctx_label, bit);
                                if (bit != 0u) {
                                    j2k_classic_encode_sign(
                                        idx,
                                        states,
                                        encoder,
                                        contexts,
                                        padded_width,
                                        ix,
                                        iy,
                                        height,
                                        style_flags
                                    );
                                    set_significant(states, padded_width, ix, iy);
                                }
                            }
                        }
                        continue;
                    }

                    j2k_mq_encode(encoder, contexts, 17u, 0u);
                    continue;
                }
            }

            for (uint y = y_base; y < y_end; ++y) {
                const uint ix = x + 1u;
                const uint iy = y + 1u;
                const uint idx = coeff_index(padded_width, ix, iy);
                if ((states[idx] & J2K_ENCODE_SIGNIFICANT) == 0u && coded[idx] == 0u) {
                    const uchar neighbor_sig =
                        j2k_classic_effective_neighbors(states, padded_width, ix, iy, height, style_flags);
                    const uint ctx_label = uint(zero_context_label(neighbor_sig, sub_band_type));
                    const uint bit = (magnitudes[idx] & bit_mask) != 0u ? 1u : 0u;
                    j2k_mq_encode(encoder, contexts, ctx_label, bit);
                    if (bit != 0u) {
                        j2k_classic_encode_sign(
                            idx,
                            states,
                            encoder,
                            contexts,
                            padded_width,
                            ix,
                            iy,
                            height,
                            style_flags
                        );
                        set_significant(states, padded_width, ix, iy);
                    }
                }
            }
        }
    }
}

inline void j2k_classic_encode_segmentation_symbols(
    thread J2kMqEncoder &encoder,
    thread uchar *contexts
) {
    j2k_mq_encode(encoder, contexts, 18u, 1u);
    j2k_mq_encode(encoder, contexts, 18u, 0u);
    j2k_mq_encode(encoder, contexts, 18u, 1u);
    j2k_mq_encode(encoder, contexts, 18u, 0u);
}

inline uint j2k_classic_bypass_segment_idx(uint pass_idx) {
    if (pass_idx < 10u) {
        return 0u;
    }
    return 1u + (2u * ((pass_idx - 10u) / 3u)) + (((pass_idx - 10u) % 3u) == 2u ? 1u : 0u);
}

inline bool j2k_classic_push_segment(
    device J2kClassicSegment *segments,
    uint segment_capacity,
    thread uint &segment_count,
    uint data_offset,
    uint data_length,
    uint start_pass,
    uint end_pass,
    bool use_arithmetic
) {
    if (segment_count >= segment_capacity) {
        return false;
    }
    segments[segment_count].data_offset = data_offset;
    segments[segment_count].data_length = data_length;
    segments[segment_count].start_coding_pass = start_pass;
    segments[segment_count].end_coding_pass = end_pass;
    segments[segment_count].use_arithmetic = use_arithmetic ? 1u : 0u;
    segment_count += 1u;
    return true;
}

inline uint j2k_classic_finish_arithmetic_segment(thread J2kMqEncoder &encoder) {
    j2k_mq_finish(encoder);
    if (encoder.failed != 0u || encoder.len == 0u) {
        encoder.failed = 1u;
        return 0u;
    }
    const uint data_len = encoder.len - 1u;
    for (uint idx = 0u; idx < data_len; ++idx) {
        encoder.data[idx] = encoder.data[idx + 1u];
    }
    return data_len;
}

inline void j2k_encode_classic_code_block_impl(
    device const int *coefficients,
    device uchar *out,
    J2kClassicEncodeParams params,
    device J2kClassicEncodeStatus *status,
    device J2kClassicSegment *segments
) {
    j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 0u, 0u, 0u, 0u, 0u);

    if (params.width == 0u || params.height == 0u ||
        params.width > J2K_CLASSIC_MAX_WIDTH ||
        params.height > J2K_CLASSIC_MAX_HEIGHT ||
        params.total_bitplanes > 31u) {
        j2k_set_encode_status(status, J2K_ENCODE_STATUS_UNSUPPORTED, 1u, 0u, 0u, 0u, 0u);
        return;
    }

    const uint padded_width = params.width + 2u;
    const uint padded_height = params.height + 2u;
    const uint padded_count = padded_width * padded_height;

    thread uint magnitudes[J2K_CLASSIC_MAX_COEFF_COUNT];
    thread uchar states[J2K_CLASSIC_MAX_COEFF_COUNT];
    thread uchar coded[J2K_CLASSIC_MAX_COEFF_COUNT];

    for (uint idx = 0u; idx < padded_count; ++idx) {
        magnitudes[idx] = 0u;
        states[idx] = uchar(0u);
        coded[idx] = uchar(0u);
    }

    uint max_magnitude = 0u;
    for (uint y = 0u; y < params.height; ++y) {
        for (uint x = 0u; x < params.width; ++x) {
            const uint src_idx = y * params.width + x;
            const int value = coefficients[src_idx];
            const uint dst_idx = coeff_index(padded_width, x + 1u, y + 1u);
            const uint magnitude = j2k_classic_magnitude(value);
            magnitudes[dst_idx] = magnitude;
            if (value < 0) {
                states[dst_idx] |= J2K_ENCODE_SIGN;
            }
            max_magnitude = max(max_magnitude, magnitude);
        }
    }

    if (max_magnitude == 0u) {
        j2k_set_encode_status(
            status,
            J2K_ENCODE_STATUS_OK,
            0u,
            0u,
            0u,
            params.total_bitplanes,
            0u
        );
        return;
    }

    const uint num_bitplanes = 32u - clz(max_magnitude);
    if (num_bitplanes > params.total_bitplanes) {
        j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 3u, 0u, 0u, 0u, 0u);
        return;
    }
    const uint missing_bit_planes = params.total_bitplanes - num_bitplanes;

    thread uchar contexts[19];
    reset_contexts(contexts);

    if ((params.style_flags & (J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS |
                               J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS)) != 0u) {
        const uint total_passes = 1u + 3u * (num_bitplanes - 1u);
        uint data_cursor = 0u;
        uint segment_count = 0u;
        uint current_segment_idx = 0xFFFFFFFFu;
        uint current_segment_start_pass = 0u;
        bool current_use_arithmetic = true;
        bool have_segment = false;
        thread J2kMqEncoder arithmetic_encoder;
        thread J2kRawBitWriter raw_writer;

        for (uint coding_pass = 0u; coding_pass < total_passes; ++coding_pass) {
            const uint segment_idx =
                (params.style_flags & J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS) != 0u
                    ? coding_pass
                    : j2k_classic_bypass_segment_idx(coding_pass);
            const bool use_arithmetic =
                (params.style_flags & J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) == 0u ||
                coding_pass <= 9u ||
                (coding_pass % 3u) == 0u;

            if (!have_segment || current_segment_idx != segment_idx) {
                if (have_segment) {
                    uint segment_len = 0u;
                    if (current_use_arithmetic) {
                        segment_len = j2k_classic_finish_arithmetic_segment(arithmetic_encoder);
                        if (arithmetic_encoder.failed != 0u) {
                            j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 6u, 0u, 0u, 0u, 0u);
                            return;
                        }
                    } else {
                        j2k_raw_writer_finish(raw_writer);
                        if (raw_writer.failed != 0u) {
                            j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 7u, 0u, 0u, 0u, 0u);
                            return;
                        }
                        segment_len = raw_writer.len;
                    }
                    if (!j2k_classic_push_segment(
                            segments,
                            params.segment_capacity,
                            segment_count,
                            data_cursor,
                            segment_len,
                            current_segment_start_pass,
                            coding_pass,
                            current_use_arithmetic)) {
                        j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 8u, 0u, 0u, 0u, 0u);
                        return;
                    }
                    data_cursor += segment_len;
                }

                current_segment_idx = segment_idx;
                current_segment_start_pass = coding_pass;
                current_use_arithmetic = use_arithmetic;
                have_segment = true;
                if (data_cursor > params.output_capacity) {
                    j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 9u, 0u, 0u, 0u, 0u);
                    return;
                }
                const uint remaining_capacity = params.output_capacity - data_cursor;
                if (use_arithmetic) {
                    j2k_mq_init(arithmetic_encoder, out + data_cursor, remaining_capacity);
                } else {
                    j2k_raw_writer_init(raw_writer, out + data_cursor, remaining_capacity);
                }
            }

            const uint current_bitplane = (coding_pass + 2u) / 3u;
            const uint bit_mask = 1u << (num_bitplanes - 1u - current_bitplane);
            switch (coding_pass % 3u) {
                case 0u:
                    j2k_classic_cleanup_pass(
                        magnitudes,
                        states,
                        coded,
                        arithmetic_encoder,
                        contexts,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        params.sub_band_type,
                        params.style_flags
                    );
                    if ((params.style_flags & J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS) != 0u) {
                        j2k_classic_encode_segmentation_symbols(arithmetic_encoder, contexts);
                    }
                    for (uint idx = 0u; idx < padded_count; ++idx) {
                        coded[idx] = uchar(0u);
                    }
                    break;
                case 1u:
                    if (use_arithmetic) {
                        j2k_classic_significance_pass(
                            magnitudes,
                            states,
                            coded,
                            arithmetic_encoder,
                            contexts,
                            params.width,
                            params.height,
                            padded_width,
                            bit_mask,
                            params.sub_band_type,
                            params.style_flags
                        );
                    } else {
                        j2k_classic_significance_pass_raw(
                            magnitudes,
                            states,
                            coded,
                            raw_writer,
                            params.width,
                            params.height,
                            padded_width,
                            bit_mask,
                            params.style_flags
                        );
                    }
                    break;
                default:
                    if (use_arithmetic) {
                        j2k_classic_magnitude_refinement_pass(
                            magnitudes,
                            states,
                            coded,
                            arithmetic_encoder,
                            contexts,
                            params.width,
                            params.height,
                            padded_width,
                            bit_mask,
                            params.style_flags
                        );
                    } else {
                        j2k_classic_magnitude_refinement_pass_raw(
                            magnitudes,
                            states,
                            coded,
                            raw_writer,
                            params.width,
                            params.height,
                            padded_width,
                            bit_mask
                        );
                    }
                    break;
            }

            if ((params.style_flags & J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES) != 0u) {
                reset_contexts(contexts);
            }
            const bool current_failed = use_arithmetic
                ? arithmetic_encoder.failed != 0u
                : raw_writer.failed != 0u;
            if (current_failed) {
                j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 10u, 0u, 0u, 0u, 0u);
                return;
            }
        }

        if (have_segment) {
            uint segment_len = 0u;
            if (current_use_arithmetic) {
                segment_len = j2k_classic_finish_arithmetic_segment(arithmetic_encoder);
                if (arithmetic_encoder.failed != 0u) {
                    j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 11u, 0u, 0u, 0u, 0u);
                    return;
                }
            } else {
                j2k_raw_writer_finish(raw_writer);
                if (raw_writer.failed != 0u) {
                    j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 12u, 0u, 0u, 0u, 0u);
                    return;
                }
                segment_len = raw_writer.len;
            }
            if (!j2k_classic_push_segment(
                    segments,
                    params.segment_capacity,
                    segment_count,
                    data_cursor,
                    segment_len,
                    current_segment_start_pass,
                    total_passes,
                    current_use_arithmetic)) {
                j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 13u, 0u, 0u, 0u, 0u);
                return;
            }
            data_cursor += segment_len;
        }

        j2k_set_encode_status(
            status,
            J2K_ENCODE_STATUS_OK,
            0u,
            data_cursor,
            total_passes,
            missing_bit_planes,
            segment_count
        );
        return;
    }

    thread J2kMqEncoder encoder;
    j2k_mq_init(encoder, out, params.output_capacity);

    uint pass_count = 0u;
    for (int bp = int(num_bitplanes) - 1; bp >= 0; --bp) {
        const uint bit_mask = 1u << uint(bp);
        const bool first_bitplane = uint(bp) == num_bitplanes - 1u;

        if (first_bitplane) {
            j2k_classic_cleanup_pass(
                magnitudes,
                states,
                coded,
                encoder,
                contexts,
                params.width,
                params.height,
                padded_width,
                bit_mask,
                params.sub_band_type,
                params.style_flags
            );
            if ((params.style_flags & J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS) != 0u) {
                j2k_classic_encode_segmentation_symbols(encoder, contexts);
            }
            pass_count += 1u;
            if ((params.style_flags & J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES) != 0u) {
                reset_contexts(contexts);
            }
        } else {
            j2k_classic_significance_pass(
                magnitudes,
                states,
                coded,
                encoder,
                contexts,
                params.width,
                params.height,
                padded_width,
                bit_mask,
                params.sub_band_type,
                params.style_flags
            );
            pass_count += 1u;
            if ((params.style_flags & J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES) != 0u) {
                reset_contexts(contexts);
            }

            j2k_classic_magnitude_refinement_pass(
                magnitudes,
                states,
                coded,
                encoder,
                contexts,
                params.width,
                params.height,
                padded_width,
                bit_mask,
                params.style_flags
            );
            pass_count += 1u;
            if ((params.style_flags & J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES) != 0u) {
                reset_contexts(contexts);
            }

            j2k_classic_cleanup_pass(
                magnitudes,
                states,
                coded,
                encoder,
                contexts,
                params.width,
                params.height,
                padded_width,
                bit_mask,
                params.sub_band_type,
                params.style_flags
            );
            if ((params.style_flags & J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS) != 0u) {
                j2k_classic_encode_segmentation_symbols(encoder, contexts);
            }
            pass_count += 1u;
            if ((params.style_flags & J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES) != 0u) {
                reset_contexts(contexts);
            }
        }

        for (uint idx = 0u; idx < padded_count; ++idx) {
            coded[idx] = uchar(0u);
        }

        if (encoder.failed != 0u) {
            j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 4u, 0u, 0u, 0u, 0u);
            return;
        }
    }

    const uint data_len = j2k_classic_finish_arithmetic_segment(encoder);
    if (encoder.failed != 0u || encoder.len == 0u) {
        j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 5u, 0u, 0u, 0u, 0u);
        return;
    }
    uint segment_count = 0u;
    if (!j2k_classic_push_segment(
            segments,
            params.segment_capacity,
            segment_count,
            0u,
            data_len,
            0u,
            pass_count,
            true)) {
        j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 14u, 0u, 0u, 0u, 0u);
        return;
    }

    j2k_set_encode_status(
        status,
        J2K_ENCODE_STATUS_OK,
        0u,
        data_len,
        pass_count,
        missing_bit_planes,
        segment_count
    );
}

struct J2kClassicEncodeBatchJob {
    uint coefficient_offset;
    uint output_offset;
    uint segment_offset;
    uint width;
    uint height;
    uint sub_band_type;
    uint total_bitplanes;
    uint style_flags;
    uint output_capacity;
    uint segment_capacity;
};

kernel void j2k_encode_classic_code_block(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    constant J2kClassicEncodeParams &params [[buffer(2)]],
    device J2kClassicEncodeStatus *status [[buffer(3)]],
    device J2kClassicSegment *segments [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }
    j2k_encode_classic_code_block_impl(coefficients, out, params, status, segments);
}

kernel void j2k_encode_classic_code_blocks(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(2)]],
    device J2kClassicEncodeStatus *statuses [[buffer(3)]],
    device J2kClassicSegment *segments [[buffer(4)]],
    constant uint &job_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= job_count) {
        return;
    }
    const J2kClassicEncodeBatchJob job = jobs[gid];
    J2kClassicEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.sub_band_type = job.sub_band_type;
    params.total_bitplanes = job.total_bitplanes;
    params.style_flags = job.style_flags;
    params.output_capacity = job.output_capacity;
    params.segment_capacity = job.segment_capacity;
    j2k_encode_classic_code_block_impl(
        coefficients + job.coefficient_offset,
        out + job.output_offset,
        params,
        statuses + gid,
        segments + job.segment_offset
    );
}

constant uint J2K_HT_MAX_BITPLANES = 30u;
constant uint J2K_HT_MEL_SIZE = 192u;
constant uint J2K_HT_VLC_SIZE = 3072u - J2K_HT_MEL_SIZE;
constant uint J2K_HT_MS_SIZE = ((16384u * 16u) + 14u) / 15u;
constant uint J2K_HT_MEL_OFFSET = J2K_HT_MS_SIZE;
constant uint J2K_HT_VLC_OFFSET = J2K_HT_MS_SIZE + J2K_HT_MEL_SIZE;

struct J2kHtEncodeParams {
    uint width;
    uint height;
    uint total_bitplanes;
    uint output_capacity;
};

struct J2kHtEncodeStatus {
    uint code;
    uint detail;
    uint data_len;
    uint num_coding_passes;
    uint num_zero_bitplanes;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct J2kHtMelEncoder {
    uint pos;
    uint remaining_bits;
    uchar tmp;
    uint run;
    uint k;
    uint threshold;
    uint failed;
};

struct J2kHtVlcEncoder {
    uint pos;
    uint used_bits;
    uchar tmp;
    uint last_greater_than_8f;
    uint failed;
};

struct J2kHtMagSgnEncoder {
    uint pos;
    uint max_bits;
    uint used_bits;
    uint tmp;
    uint failed;
};

constant uint J2K_HT_MEL_EXP[13] = {
    0u, 0u, 0u, 1u, 1u, 1u, 2u, 2u, 2u, 3u, 3u, 4u, 5u
};

inline void j2k_set_ht_encode_status(
    device J2kHtEncodeStatus *status,
    uint code,
    uint detail,
    uint data_len,
    uint passes,
    uint zbp
) {
    status->code = code;
    status->detail = detail;
    status->data_len = data_len;
    status->num_coding_passes = passes;
    status->num_zero_bitplanes = zbp;
    status->reserved0 = 0u;
    status->reserved1 = 0u;
    status->reserved2 = 0u;
}

inline uint j2k_ht_aligned_sign_magnitude(int coefficient, uint total_bitplanes) {
    if (coefficient == 0) {
        return 0u;
    }
    const uint sign = coefficient < 0 ? 0x80000000u : 0u;
    const uint magnitude = (coefficient < 0 ? uint(-coefficient) : uint(coefficient))
        << (31u - total_bitplanes);
    return sign | magnitude;
}

inline void j2k_ht_mel_init(thread J2kHtMelEncoder &mel) {
    mel.pos = 0u;
    mel.remaining_bits = 8u;
    mel.tmp = uchar(0u);
    mel.run = 0u;
    mel.k = 0u;
    mel.threshold = 1u;
    mel.failed = 0u;
}

inline void j2k_ht_vlc_init(thread J2kHtVlcEncoder &vlc, device uchar *out) {
    vlc.pos = 1u;
    vlc.used_bits = 4u;
    vlc.tmp = uchar(0x0Fu);
    vlc.last_greater_than_8f = 1u;
    vlc.failed = 0u;
    out[J2K_HT_VLC_OFFSET + J2K_HT_VLC_SIZE - 1u] = uchar(0xFFu);
}

inline void j2k_ht_ms_init(thread J2kHtMagSgnEncoder &ms) {
    ms.pos = 0u;
    ms.max_bits = 8u;
    ms.used_bits = 0u;
    ms.tmp = 0u;
    ms.failed = 0u;
}

inline void j2k_ht_mel_emit_bit(thread J2kHtMelEncoder &mel, device uchar *out, bool bit) {
    mel.tmp = uchar((uint(mel.tmp) << 1u) | (bit ? 1u : 0u));
    mel.remaining_bits -= 1u;
    if (mel.remaining_bits == 0u) {
        if (mel.pos >= J2K_HT_MEL_SIZE) {
            mel.failed = 1u;
            return;
        }
        out[J2K_HT_MEL_OFFSET + mel.pos] = mel.tmp;
        mel.pos += 1u;
        mel.remaining_bits = mel.tmp == uchar(0xFFu) ? 7u : 8u;
        mel.tmp = uchar(0u);
    }
}

inline void j2k_ht_mel_encode(thread J2kHtMelEncoder &mel, device uchar *out, bool bit) {
    if (!bit) {
        mel.run += 1u;
        if (mel.run >= mel.threshold) {
            j2k_ht_mel_emit_bit(mel, out, true);
            mel.run = 0u;
            mel.k = min(mel.k + 1u, 12u);
            mel.threshold = 1u << J2K_HT_MEL_EXP[mel.k];
        }
    } else {
        j2k_ht_mel_emit_bit(mel, out, false);
        uint t = J2K_HT_MEL_EXP[mel.k];
        while (t > 0u) {
            t -= 1u;
            j2k_ht_mel_emit_bit(mel, out, ((mel.run >> t) & 1u) != 0u);
        }
        mel.run = 0u;
        mel.k = mel.k == 0u ? 0u : mel.k - 1u;
        mel.threshold = 1u << J2K_HT_MEL_EXP[mel.k];
    }
}

inline void j2k_ht_vlc_encode(
    thread J2kHtVlcEncoder &vlc,
    device uchar *out,
    uint codeword,
    uint codeword_len
) {
    while (codeword_len > 0u) {
        if (vlc.pos >= J2K_HT_VLC_SIZE) {
            vlc.failed = 1u;
            return;
        }

        uint available_bits = 8u - vlc.last_greater_than_8f - vlc.used_bits;
        const uint take = min(available_bits, codeword_len);
        const uint mask = take == 32u ? 0xFFFFFFFFu : ((1u << take) - 1u);
        vlc.tmp = uchar(uint(vlc.tmp) | ((codeword & mask) << vlc.used_bits));
        vlc.used_bits += take;
        available_bits -= take;
        codeword_len -= take;
        codeword >>= take;

        if (available_bits == 0u) {
            if (vlc.last_greater_than_8f != 0u && vlc.tmp != uchar(0x7Fu)) {
                vlc.last_greater_than_8f = 0u;
                continue;
            }

            const uint write_index = J2K_HT_VLC_SIZE - 1u - vlc.pos;
            out[J2K_HT_VLC_OFFSET + write_index] = vlc.tmp;
            vlc.pos += 1u;
            vlc.last_greater_than_8f = vlc.tmp > uchar(0x8Fu) ? 1u : 0u;
            vlc.tmp = uchar(0u);
            vlc.used_bits = 0u;
        }
    }
}

inline void j2k_ht_ms_encode(
    thread J2kHtMagSgnEncoder &ms,
    device uchar *out,
    uint codeword,
    uint codeword_len
) {
    while (codeword_len > 0u) {
        if (ms.pos >= J2K_HT_MS_SIZE) {
            ms.failed = 1u;
            return;
        }

        const uint take = min(ms.max_bits - ms.used_bits, codeword_len);
        const uint mask = take == 32u ? 0xFFFFFFFFu : ((1u << take) - 1u);
        ms.tmp |= (codeword & mask) << ms.used_bits;
        ms.used_bits += take;
        codeword >>= take;
        codeword_len -= take;

        if (ms.used_bits >= ms.max_bits) {
            out[ms.pos] = uchar(ms.tmp);
            ms.pos += 1u;
            ms.max_bits = ms.tmp == 0xFFu ? 7u : 8u;
            ms.tmp = 0u;
            ms.used_bits = 0u;
        }
    }
}

inline void j2k_ht_ms_terminate(thread J2kHtMagSgnEncoder &ms, device uchar *out) {
    if (ms.used_bits > 0u) {
        const uint unused = ms.max_bits - ms.used_bits;
        ms.tmp |= (0xFFu & ((1u << unused) - 1u)) << ms.used_bits;
        ms.used_bits += unused;
        if (ms.tmp != 0xFFu) {
            if (ms.pos >= J2K_HT_MS_SIZE) {
                ms.failed = 1u;
                return;
            }
            out[ms.pos] = uchar(ms.tmp);
            ms.pos += 1u;
        }
    } else if (ms.max_bits == 7u) {
        ms.pos = ms.pos == 0u ? 0u : ms.pos - 1u;
    }
}

inline void j2k_ht_process_sample(
    uint slot,
    uint value,
    uint p,
    thread int *rho_acc,
    thread int *e_q,
    thread int &e_qmax,
    thread uint *s
) {
    uint val = value + value;
    val >>= p;
    val &= ~1u;
    if (val != 0u) {
        rho_acc[0] |= int(1u << (slot & 0x3u));
        val -= 1u;
        e_q[slot] = int(32u - clz(val));
        e_qmax = max(e_qmax, e_q[slot]);
        val -= 1u;
        s[slot] = val + (value >> 31u);
    }
}

inline uchar j2k_ht_uvlc_byte(device const uchar *table, uint index, uint field) {
    return table[index * 6u + field];
}

inline void j2k_ht_encode_uvlc_pair(
    thread J2kHtVlcEncoder &vlc,
    device uchar *out,
    device const uchar *uvlc_table,
    uint first_index,
    uint second_index
) {
    const uchar first_pre = j2k_ht_uvlc_byte(uvlc_table, first_index, 0u);
    const uchar first_pre_len = j2k_ht_uvlc_byte(uvlc_table, first_index, 1u);
    const uchar first_suf = j2k_ht_uvlc_byte(uvlc_table, first_index, 2u);
    const uchar first_suf_len = j2k_ht_uvlc_byte(uvlc_table, first_index, 3u);
    const uchar second_pre = j2k_ht_uvlc_byte(uvlc_table, second_index, 0u);
    const uchar second_pre_len = j2k_ht_uvlc_byte(uvlc_table, second_index, 1u);
    const uchar second_suf = j2k_ht_uvlc_byte(uvlc_table, second_index, 2u);
    const uchar second_suf_len = j2k_ht_uvlc_byte(uvlc_table, second_index, 3u);
    j2k_ht_vlc_encode(vlc, out, uint(first_pre), uint(first_pre_len));
    j2k_ht_vlc_encode(vlc, out, uint(second_pre), uint(second_pre_len));
    j2k_ht_vlc_encode(vlc, out, uint(first_suf), uint(first_suf_len));
    j2k_ht_vlc_encode(vlc, out, uint(second_suf), uint(second_suf_len));
}

inline void j2k_ht_encode_uvlc(
    int u_q0,
    int u_q1,
    thread J2kHtVlcEncoder &vlc,
    device uchar *out,
    device const uchar *uvlc_table
) {
    if (u_q0 > 2 && u_q1 > 2) {
        j2k_ht_encode_uvlc_pair(vlc, out, uvlc_table, uint(u_q0 - 2), uint(u_q1 - 2));
    } else if (u_q0 > 2 && u_q1 > 0) {
        const uint first_index = uint(u_q0);
        const uchar first_pre = j2k_ht_uvlc_byte(uvlc_table, first_index, 0u);
        const uchar first_pre_len = j2k_ht_uvlc_byte(uvlc_table, first_index, 1u);
        const uchar first_suf = j2k_ht_uvlc_byte(uvlc_table, first_index, 2u);
        const uchar first_suf_len = j2k_ht_uvlc_byte(uvlc_table, first_index, 3u);
        j2k_ht_vlc_encode(vlc, out, uint(first_pre), uint(first_pre_len));
        j2k_ht_vlc_encode(vlc, out, uint(u_q1 - 1), 1u);
        j2k_ht_vlc_encode(vlc, out, uint(first_suf), uint(first_suf_len));
    } else {
        j2k_ht_encode_uvlc_pair(
            vlc,
            out,
            uvlc_table,
            uint(max(u_q0, 0)),
            uint(max(u_q1, 0))
        );
    }
}

inline void j2k_ht_encode_uvlc_non_initial(
    int u_q0,
    int u_q1,
    thread J2kHtVlcEncoder &vlc,
    device uchar *out,
    device const uchar *uvlc_table
) {
    j2k_ht_encode_uvlc_pair(
        vlc,
        out,
        uvlc_table,
        uint(max(u_q0, 0)),
        uint(max(u_q1, 0))
    );
}

inline void j2k_ht_encode_mag_signs(
    int rho,
    int u_q,
    ushort tuple,
    thread const uint *s,
    uint offset,
    thread J2kHtMagSgnEncoder &ms,
    device uchar *out
) {
    const uint e_k = uint(tuple & ushort(0xFu));
    for (uint bit = 0u; bit < 4u; ++bit) {
        const int sample_mask = int(1u << bit);
        if ((rho & sample_mask) == 0) {
            continue;
        }
        const int reduction = int((e_k >> bit) & 1u);
        const uint magnitude_bits = uint(u_q - reduction);
        const uint payload = magnitude_bits == 0u
            ? 0u
            : (s[offset + bit] & ((1u << magnitude_bits) - 1u));
        j2k_ht_ms_encode(ms, out, payload, magnitude_bits);
    }
}

inline int j2k_ht_encode_quad_initial_row(
    uint offset,
    uint c_q,
    int rho,
    int e_qmax,
    thread const int *e_q,
    thread const uint *s,
    uint lep,
    uint lcxp,
    thread uchar *e_val,
    thread uchar *cx_val,
    thread J2kHtMelEncoder &mel,
    thread J2kHtVlcEncoder &vlc,
    thread J2kHtMagSgnEncoder &ms,
    device uchar *out,
    device const ushort *vlc_table0
) {
    const int u_q = max(e_qmax, 1) - 1;
    uint eps = 0u;
    if (u_q > 0) {
        eps |= uint(e_q[offset] == e_qmax);
        eps |= uint(e_q[offset + 1u] == e_qmax) << 1u;
        eps |= uint(e_q[offset + 2u] == e_qmax) << 2u;
        eps |= uint(e_q[offset + 3u] == e_qmax) << 3u;
    }

    e_val[lep] = max(e_val[lep], uchar(e_q[offset + 1u]));
    e_val[lep + 1u] = uchar(e_q[offset + 3u]);
    cx_val[lcxp] = uchar(uint(cx_val[lcxp]) | uint((rho & 2) >> 1));
    cx_val[lcxp + 1u] = uchar((rho & 8) >> 3);

    const ushort tuple = vlc_table0[(c_q << 8u) | (uint(rho) << 4u) | eps];
    j2k_ht_vlc_encode(vlc, out, uint(tuple >> 8u), uint((tuple >> 4u) & ushort(0x7u)));
    if (c_q == 0u) {
        j2k_ht_mel_encode(mel, out, rho != 0);
    }
    j2k_ht_encode_mag_signs(rho, max(e_qmax, 1), tuple, s, offset, ms, out);
    return u_q;
}

inline int j2k_ht_encode_quad_non_initial_row(
    uint offset,
    uint c_q,
    int rho,
    int e_qmax,
    int max_e,
    thread const int *e_q,
    thread const uint *s,
    thread J2kHtMelEncoder &mel,
    thread J2kHtVlcEncoder &vlc,
    thread J2kHtMagSgnEncoder &ms,
    device uchar *out,
    device const ushort *vlc_table1
) {
    const int kappa = (rho & (rho - 1)) != 0 ? max(max_e, 1) : 1;
    const int u_q = max(e_qmax, kappa) - kappa;
    uint eps = 0u;
    if (u_q > 0) {
        eps |= uint(e_q[offset] == e_qmax);
        eps |= uint(e_q[offset + 1u] == e_qmax) << 1u;
        eps |= uint(e_q[offset + 2u] == e_qmax) << 2u;
        eps |= uint(e_q[offset + 3u] == e_qmax) << 3u;
    }

    const ushort tuple = vlc_table1[(c_q << 8u) | (uint(rho) << 4u) | eps];
    j2k_ht_vlc_encode(vlc, out, uint(tuple >> 8u), uint((tuple >> 4u) & ushort(0x7u)));
    if (c_q == 0u) {
        j2k_ht_mel_encode(mel, out, rho != 0);
    }
    j2k_ht_encode_mag_signs(rho, max(e_qmax, kappa), tuple, s, offset, ms, out);
    return u_q;
}

inline void j2k_ht_clear_quad_state(thread int *rho, thread int *e_q, thread int *e_qmax, thread uint *s) {
    rho[0] = 0;
    rho[1] = 0;
    for (uint idx = 0u; idx < 8u; ++idx) {
        e_q[idx] = 0;
        s[idx] = 0u;
    }
    e_qmax[0] = 0;
    e_qmax[1] = 0;
}

inline int j2k_ht_encode_first_quad_pair(
    device const int *coefficients,
    uint stride,
    uint height,
    uint total_bitplanes,
    uint p,
    thread uint &sp,
    uint x,
    thread uchar *e_val,
    thread uchar *cx_val,
    thread uint &c_q0,
    thread int *rho,
    thread int *e_q,
    thread int *e_qmax,
    thread uint *s,
    thread J2kHtMelEncoder &mel,
    thread J2kHtVlcEncoder &vlc,
    thread J2kHtMagSgnEncoder &ms,
    device uchar *out,
    device const ushort *vlc_table0,
    device const uchar *uvlc_table
) {
    const uint lep = x / 2u;
    const uint lcxp = x / 2u;

    j2k_ht_process_sample(0u, j2k_ht_aligned_sign_magnitude(coefficients[sp], total_bitplanes), p, &rho[0], e_q, e_qmax[0], s);
    j2k_ht_process_sample(
        1u,
        height > 1u ? j2k_ht_aligned_sign_magnitude(coefficients[sp + stride], total_bitplanes) : 0u,
        p,
        &rho[0],
        e_q,
        e_qmax[0],
        s
    );
    sp += 1u;

    if (x + 1u < stride) {
        j2k_ht_process_sample(2u, j2k_ht_aligned_sign_magnitude(coefficients[sp], total_bitplanes), p, &rho[0], e_q, e_qmax[0], s);
        j2k_ht_process_sample(
            3u,
            height > 1u ? j2k_ht_aligned_sign_magnitude(coefficients[sp + stride], total_bitplanes) : 0u,
            p,
            &rho[0],
            e_q,
            e_qmax[0],
            s
        );
        sp += 1u;
    }

    const int u_q0 = j2k_ht_encode_quad_initial_row(
        0u, c_q0, rho[0], e_qmax[0], e_q, s, lep, lcxp, e_val, cx_val, mel, vlc, ms, out, vlc_table0
    );

    if (x + 2u < stride) {
        j2k_ht_process_sample(4u, j2k_ht_aligned_sign_magnitude(coefficients[sp], total_bitplanes), p, &rho[1], e_q, e_qmax[1], s);
        j2k_ht_process_sample(
            5u,
            height > 1u ? j2k_ht_aligned_sign_magnitude(coefficients[sp + stride], total_bitplanes) : 0u,
            p,
            &rho[1],
            e_q,
            e_qmax[1],
            s
        );
        sp += 1u;

        if (x + 3u < stride) {
            j2k_ht_process_sample(6u, j2k_ht_aligned_sign_magnitude(coefficients[sp], total_bitplanes), p, &rho[1], e_q, e_qmax[1], s);
            j2k_ht_process_sample(
                7u,
                height > 1u ? j2k_ht_aligned_sign_magnitude(coefficients[sp + stride], total_bitplanes) : 0u,
                p,
                &rho[1],
                e_q,
                e_qmax[1],
                s
            );
            sp += 1u;
        }

        const uint c_q1 = uint((rho[0] >> 1) | (rho[0] & 1));
        const int u_q1 = j2k_ht_encode_quad_initial_row(
            4u, c_q1, rho[1], e_qmax[1], e_q, s, lep + 1u, lcxp + 1u, e_val, cx_val, mel, vlc, ms, out, vlc_table0
        );

        if (u_q0 > 0 && u_q1 > 0) {
            j2k_ht_mel_encode(mel, out, min(u_q0, u_q1) > 2);
        }
        j2k_ht_encode_uvlc(u_q0, u_q1, vlc, out, uvlc_table);
        c_q0 = uint((rho[1] >> 1) | (rho[1] & 1));
    } else {
        j2k_ht_encode_uvlc(u_q0, 0, vlc, out, uvlc_table);
        c_q0 = 0u;
    }

    j2k_ht_clear_quad_state(rho, e_q, e_qmax, s);
    return 0;
}

inline int j2k_ht_encode_non_initial_quad_pair(
    device const int *coefficients,
    uint stride,
    uint width,
    uint height,
    uint y,
    uint total_bitplanes,
    uint p,
    thread uint &sp,
    uint x,
    thread uchar *e_val,
    thread uchar *cx_val,
    thread uint &lep,
    thread uint &lcxp,
    thread int &max_e,
    thread uint &c_q0,
    thread int *rho,
    thread int *e_q,
    thread int *e_qmax,
    thread uint *s,
    thread J2kHtMelEncoder &mel,
    thread J2kHtVlcEncoder &vlc,
    thread J2kHtMagSgnEncoder &ms,
    device uchar *out,
    device const ushort *vlc_table1,
    device const uchar *uvlc_table
) {
    j2k_ht_process_sample(0u, j2k_ht_aligned_sign_magnitude(coefficients[sp], total_bitplanes), p, &rho[0], e_q, e_qmax[0], s);
    j2k_ht_process_sample(
        1u,
        y + 1u < height ? j2k_ht_aligned_sign_magnitude(coefficients[sp + stride], total_bitplanes) : 0u,
        p,
        &rho[0],
        e_q,
        e_qmax[0],
        s
    );
    sp += 1u;

    if (x + 1u < width) {
        j2k_ht_process_sample(2u, j2k_ht_aligned_sign_magnitude(coefficients[sp], total_bitplanes), p, &rho[0], e_q, e_qmax[0], s);
        j2k_ht_process_sample(
            3u,
            y + 1u < height ? j2k_ht_aligned_sign_magnitude(coefficients[sp + stride], total_bitplanes) : 0u,
            p,
            &rho[0],
            e_q,
            e_qmax[0],
            s
        );
        sp += 1u;
    }

    const int prev_max = max_e;
    const int u_q0 = j2k_ht_encode_quad_non_initial_row(
        0u, c_q0, rho[0], e_qmax[0], prev_max, e_q, s, mel, vlc, ms, out, vlc_table1
    );

    e_val[lep] = max(e_val[lep], uchar(e_q[1]));
    lep += 1u;
    max_e = int(max(e_val[lep], e_val[lep + 1u])) - 1;
    e_val[lep] = uchar(e_q[3]);
    cx_val[lcxp] = uchar(uint(cx_val[lcxp]) | uint((rho[0] & 2) >> 1));
    lcxp += 1u;
    uint c_q1 = uint(cx_val[lcxp]) + (uint(cx_val[lcxp + 1u]) << 2u);
    cx_val[lcxp] = uchar((rho[0] & 8) >> 3);

    int u_q1 = 0;
    if (x + 2u < width) {
        j2k_ht_process_sample(4u, j2k_ht_aligned_sign_magnitude(coefficients[sp], total_bitplanes), p, &rho[1], e_q, e_qmax[1], s);
        j2k_ht_process_sample(
            5u,
            y + 1u < height ? j2k_ht_aligned_sign_magnitude(coefficients[sp + stride], total_bitplanes) : 0u,
            p,
            &rho[1],
            e_q,
            e_qmax[1],
            s
        );
        sp += 1u;

        if (x + 3u < width) {
            j2k_ht_process_sample(6u, j2k_ht_aligned_sign_magnitude(coefficients[sp], total_bitplanes), p, &rho[1], e_q, e_qmax[1], s);
            j2k_ht_process_sample(
                7u,
                y + 1u < height ? j2k_ht_aligned_sign_magnitude(coefficients[sp + stride], total_bitplanes) : 0u,
                p,
                &rho[1],
                e_q,
                e_qmax[1],
                s
            );
            sp += 1u;
        }

        c_q1 |= uint((rho[0] & 4) >> 1);
        c_q1 |= uint((rho[0] & 8) >> 2);
        u_q1 = j2k_ht_encode_quad_non_initial_row(
            4u, c_q1, rho[1], e_qmax[1], max_e, e_q, s, mel, vlc, ms, out, vlc_table1
        );

        e_val[lep] = max(e_val[lep], uchar(e_q[5]));
        lep += 1u;
        max_e = int(max(e_val[lep], e_val[lep + 1u])) - 1;
        e_val[lep] = uchar(e_q[7]);
        cx_val[lcxp] = uchar(uint(cx_val[lcxp]) | uint((rho[1] & 2) >> 1));
        lcxp += 1u;
        c_q0 = uint(cx_val[lcxp]) + (uint(cx_val[lcxp + 1u]) << 2u);
        cx_val[lcxp] = uchar((rho[1] & 8) >> 3);
        c_q0 |= uint((rho[1] & 4) >> 1);
        c_q0 |= uint((rho[1] & 8) >> 2);
    } else {
        c_q0 = 0u;
    }

    j2k_ht_encode_uvlc_non_initial(u_q0, u_q1, vlc, out, uvlc_table);
    j2k_ht_clear_quad_state(rho, e_q, e_qmax, s);
    return 0;
}

inline void j2k_ht_terminate_mel_vlc(
    thread J2kHtMelEncoder &mel,
    thread J2kHtVlcEncoder &vlc,
    device uchar *out
) {
    if (mel.run > 0u) {
        j2k_ht_mel_emit_bit(mel, out, true);
    }

    mel.tmp = uchar(uint(mel.tmp) << mel.remaining_bits);
    const uchar mel_mask = uchar((0xFFu << mel.remaining_bits) & 0xFFu);
    const uchar vlc_mask = vlc.used_bits == 0u
        ? uchar(0u)
        : uchar((1u << vlc.used_bits) - 1u);

    if ((mel_mask | vlc_mask) == uchar(0u)) {
        return;
    }

    const uchar fused = mel.tmp | vlc.tmp;
    const bool fused_ok =
        ((((fused ^ mel.tmp) & mel_mask) | ((fused ^ vlc.tmp) & vlc_mask)) == uchar(0u)) &&
        fused != uchar(0xFFu);

    if (fused_ok && vlc.pos > 1u) {
        if (mel.pos >= J2K_HT_MEL_SIZE) {
            mel.failed = 1u;
            return;
        }
        out[J2K_HT_MEL_OFFSET + mel.pos] = fused;
        mel.pos += 1u;
    } else {
        if (mel.pos >= J2K_HT_MEL_SIZE || vlc.pos >= J2K_HT_VLC_SIZE) {
            mel.failed = 1u;
            vlc.failed = 1u;
            return;
        }
        out[J2K_HT_MEL_OFFSET + mel.pos] = mel.tmp;
        mel.pos += 1u;
        const uint write_index = J2K_HT_VLC_SIZE - 1u - vlc.pos;
        out[J2K_HT_VLC_OFFSET + write_index] = vlc.tmp;
        vlc.pos += 1u;
    }
}

inline void j2k_encode_ht_code_block_impl_with_max(
    device const int *coefficients,
    device uchar *out,
    J2kHtEncodeParams params,
    device const ushort *vlc_table0,
    device const ushort *vlc_table1,
    device const uchar *uvlc_table,
    device J2kHtEncodeStatus *status,
    uint max_magnitude
) {
    j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_FAIL, 0u, 0u, 0u, 0u);

    if (params.width == 0u || params.height == 0u ||
        params.total_bitplanes == 0u || params.total_bitplanes > J2K_HT_MAX_BITPLANES ||
        params.output_capacity < J2K_HT_MS_SIZE + J2K_HT_MEL_SIZE + J2K_HT_VLC_SIZE) {
        j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_UNSUPPORTED, 1u, 0u, 0u, 0u);
        return;
    }

    if (max_magnitude == 0u) {
        j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_OK, 0u, 0u, 0u, params.total_bitplanes);
        return;
    }

    const uint block_bitplanes = 32u - clz(max_magnitude);
    if (block_bitplanes > params.total_bitplanes) {
        j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_FAIL, 2u, 0u, 0u, 0u);
        return;
    }

    const uint missing_msbs = params.total_bitplanes - 1u;
    const uint p = 30u - missing_msbs;

    thread J2kHtMelEncoder mel;
    thread J2kHtVlcEncoder vlc;
    thread J2kHtMagSgnEncoder ms;
    j2k_ht_mel_init(mel);
    j2k_ht_vlc_init(vlc, out);
    j2k_ht_ms_init(ms);

    thread uchar e_val[513];
    thread uchar cx_val[513];
    for (uint idx = 0u; idx < 513u; ++idx) {
        e_val[idx] = uchar(0u);
        cx_val[idx] = uchar(0u);
    }

    thread int e_qmax[2];
    thread int e_q[8];
    thread int rho[2];
    thread uint s[8];
    j2k_ht_clear_quad_state(rho, e_q, e_qmax, s);

    uint c_q0 = 0u;
    uint sp = 0u;
    uint x = 0u;
    while (x < params.width) {
        j2k_ht_encode_first_quad_pair(
            coefficients,
            params.width,
            params.height,
            params.total_bitplanes,
            p,
            sp,
            x,
            e_val,
            cx_val,
            c_q0,
            rho,
            e_q,
            e_qmax,
            s,
            mel,
            vlc,
            ms,
            out,
            vlc_table0,
            uvlc_table
        );
        x += 4u;
    }

    const uint e_val_sentinel = (params.width + 1u) / 2u + 1u;
    if (e_val_sentinel < 513u) {
        e_val[e_val_sentinel] = uchar(0u);
    }

    uint y = 2u;
    while (y < params.height) {
        uint lep = 0u;
        int max_e = int(max(e_val[lep], e_val[lep + 1u])) - 1;
        e_val[lep] = uchar(0u);

        uint lcxp = 0u;
        c_q0 = uint(cx_val[lcxp]) + (uint(cx_val[lcxp + 1u]) << 2u);
        cx_val[lcxp] = uchar(0u);

        sp = y * params.width;
        x = 0u;
        while (x < params.width) {
            j2k_ht_encode_non_initial_quad_pair(
                coefficients,
                params.width,
                params.width,
                params.height,
                y,
                params.total_bitplanes,
                p,
                sp,
                x,
                e_val,
                cx_val,
                lep,
                lcxp,
                max_e,
                c_q0,
                rho,
                e_q,
                e_qmax,
                s,
                mel,
                vlc,
                ms,
                out,
                vlc_table1,
                uvlc_table
            );
            x += 4u;
        }

        y += 2u;
    }

    j2k_ht_terminate_mel_vlc(mel, vlc, out);
    j2k_ht_ms_terminate(ms, out);

    if (mel.failed != 0u || vlc.failed != 0u || ms.failed != 0u) {
        j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_FAIL, 3u, 0u, 0u, 0u);
        return;
    }

    const uint total_len = ms.pos + mel.pos + vlc.pos;
    if (total_len < 2u || total_len > params.output_capacity) {
        j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_FAIL, 4u, 0u, 0u, 0u);
        return;
    }

    for (uint idx = 0u; idx < mel.pos; ++idx) {
        out[ms.pos + idx] = out[J2K_HT_MEL_OFFSET + idx];
    }
    const uint vlc_start = J2K_HT_VLC_SIZE - vlc.pos;
    for (uint idx = 0u; idx < vlc.pos; ++idx) {
        out[ms.pos + mel.pos + idx] = out[J2K_HT_VLC_OFFSET + vlc_start + idx];
    }

    const uint last = total_len - 1u;
    const uint prev = total_len - 2u;
    const uint locator_bytes = mel.pos + vlc.pos;
    out[last] = uchar(locator_bytes >> 4u);
    out[prev] = uchar((out[prev] & uchar(0xF0u)) | uchar(locator_bytes & 0x0Fu));

    j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_OK, 0u, total_len, 1u, missing_msbs);
}

inline void j2k_encode_ht_code_block_impl(
    device const int *coefficients,
    device uchar *out,
    J2kHtEncodeParams params,
    device const ushort *vlc_table0,
    device const ushort *vlc_table1,
    device const uchar *uvlc_table,
    device J2kHtEncodeStatus *status
) {
    uint max_magnitude = 0u;
    for (uint y = 0u; y < params.height; ++y) {
        for (uint x = 0u; x < params.width; ++x) {
            max_magnitude = max(max_magnitude, j2k_classic_magnitude(coefficients[y * params.width + x]));
        }
    }
    j2k_encode_ht_code_block_impl_with_max(
        coefficients,
        out,
        params,
        vlc_table0,
        vlc_table1,
        uvlc_table,
        status,
        max_magnitude
    );
}

struct J2kHtEncodeBatchJob {
    uint coefficient_offset;
    uint output_offset;
    uint width;
    uint height;
    uint total_bitplanes;
    uint output_capacity;
};

kernel void j2k_encode_ht_code_block(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    constant J2kHtEncodeParams &params [[buffer(2)]],
    device const ushort *vlc_table0 [[buffer(3)]],
    device const ushort *vlc_table1 [[buffer(4)]],
    device const uchar *uvlc_table [[buffer(5)]],
    device J2kHtEncodeStatus *status [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }
    j2k_encode_ht_code_block_impl(
        coefficients,
        out,
        params,
        vlc_table0,
        vlc_table1,
        uvlc_table,
        status
    );
}

kernel void j2k_encode_ht_code_blocks(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    device const J2kHtEncodeBatchJob *jobs [[buffer(2)]],
    device const ushort *vlc_table0 [[buffer(3)]],
    device const ushort *vlc_table1 [[buffer(4)]],
    device const uchar *uvlc_table [[buffer(5)]],
    device J2kHtEncodeStatus *statuses [[buffer(6)]],
    constant uint &job_count [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= job_count) {
        return;
    }
    const J2kHtEncodeBatchJob job = jobs[gid];
    J2kHtEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.total_bitplanes = job.total_bitplanes;
    params.output_capacity = job.output_capacity;
    j2k_encode_ht_code_block_impl(
        coefficients + job.coefficient_offset,
        out + job.output_offset,
        params,
        vlc_table0,
        vlc_table1,
        uvlc_table,
        statuses + gid
    );
}

kernel void j2k_encode_ht_code_blocks_simd_prototype(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    device const J2kHtEncodeBatchJob *jobs [[buffer(2)]],
    device const ushort *vlc_table0 [[buffer(3)]],
    device const ushort *vlc_table1 [[buffer(4)]],
    device const uchar *uvlc_table [[buffer(5)]],
    device J2kHtEncodeStatus *statuses [[buffer(6)]],
    constant uint &job_count [[buffer(7)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (tg >= job_count) {
        return;
    }

    const J2kHtEncodeBatchJob job = jobs[tg];
    device const int *block = coefficients + job.coefficient_offset;
    const uint sample_count = job.width * job.height;
    uint local_max = 0u;
    for (uint idx = tid; idx < sample_count; idx += 32u) {
        local_max = max(local_max, j2k_classic_magnitude(block[idx]));
    }
    const uint block_max = simd_max(local_max);

    if (tid != 0u) {
        return;
    }

    J2kHtEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.total_bitplanes = job.total_bitplanes;
    params.output_capacity = job.output_capacity;
    j2k_encode_ht_code_block_impl_with_max(
        block,
        out + job.output_offset,
        params,
        vlc_table0,
        vlc_table1,
        uvlc_table,
        statuses + tg,
        block_max
    );
}

struct J2kPacketEncodeParams {
    uint resolution_count;
    uint num_layers;
    uint num_components;
    uint code_block_count;
    uint subband_count;
    uint descriptor_count;
    uint output_capacity;
    uint header_capacity;
    uint scratch_node_capacity;
};

struct J2kPacketDescriptor {
    uint packet_index;
    uint state_index;
    uint layer;
    uint resolution;
    uint component;
    uint precinct_lo;
    uint precinct_hi;
    uint state_block_offset;
};

struct J2kPacketResolution {
    uint subband_offset;
    uint subband_count;
};

struct J2kPacketSubband {
    uint block_offset;
    uint block_count;
    uint num_cbs_x;
    uint num_cbs_y;
};

struct J2kPacketBlock {
    uint data_offset;
    uint data_len;
    uint num_coding_passes;
    uint num_zero_bitplanes;
    uint previously_included;
    uint l_block;
    uint block_coding_mode;
    uint reserved0;
};

struct J2kResidentPacketBlock {
    uint tier1_job_index;
    uint previously_included;
    uint l_block;
    uint block_coding_mode;
};

struct J2kResidentPacketBlockParams {
    uint block_count;
    uint tier1_job_count;
};

struct J2kPacketStateBlock {
    uint previously_included;
    uint l_block;
};

struct J2kPacketEncodeStatus {
    uint code;
    uint detail;
    uint data_len;
    uint reserved0;
};

struct J2kLosslessCodestreamAssemblyParams {
    uint width;
    uint height;
    uint num_components;
    uint bit_depth;
    uint signed_samples;
    uint num_decomposition_levels;
    uint use_mct;
    uint guard_bits;
    uint progression_order;
    uint write_tlm;
    uint high_throughput;
    uint output_capacity;
};

struct J2kCodestreamAssemblyStatus {
    uint code;
    uint detail;
    uint data_len;
    uint reserved0;
};

struct J2kPacketBitWriter {
    device uchar *data;
    uint capacity;
    uint len;
    uint buffer;
    uint bits_in_buffer;
    uint last_byte_was_ff;
    uint failed;
};

inline void j2k_set_packet_status(device J2kPacketEncodeStatus *status, uint code, uint detail, uint len) {
    status->code = code;
    status->detail = detail;
    status->data_len = len;
    status->reserved0 = 0u;
}

inline void j2k_set_codestream_status(
    device J2kCodestreamAssemblyStatus *status,
    uint code,
    uint detail,
    uint len
) {
    status->code = code;
    status->detail = detail;
    status->data_len = len;
    status->reserved0 = 0u;
}

inline bool j2k_codestream_write_u8(
    device uchar *out,
    uint capacity,
    thread uint &cursor,
    uint value
) {
    if (cursor >= capacity) {
        return false;
    }
    out[cursor] = uchar(value & 0xFFu);
    cursor += 1u;
    return true;
}

inline bool j2k_codestream_write_u16(
    device uchar *out,
    uint capacity,
    thread uint &cursor,
    uint value
) {
    return j2k_codestream_write_u8(out, capacity, cursor, value >> 8u) &&
        j2k_codestream_write_u8(out, capacity, cursor, value);
}

inline bool j2k_codestream_write_u32(
    device uchar *out,
    uint capacity,
    thread uint &cursor,
    uint value
) {
    return j2k_codestream_write_u8(out, capacity, cursor, value >> 24u) &&
        j2k_codestream_write_u8(out, capacity, cursor, value >> 16u) &&
        j2k_codestream_write_u8(out, capacity, cursor, value >> 8u) &&
        j2k_codestream_write_u8(out, capacity, cursor, value);
}

inline bool j2k_codestream_write_marker(
    device uchar *out,
    uint capacity,
    thread uint &cursor,
    uint marker
) {
    return j2k_codestream_write_u8(out, capacity, cursor, 0xFFu) &&
        j2k_codestream_write_u8(out, capacity, cursor, marker);
}

kernel void j2k_assemble_lossless_classic_codestream(
    device const uchar *tile_data [[buffer(0)]],
    device const J2kPacketEncodeStatus *tile_status [[buffer(1)]],
    device uchar *out [[buffer(2)]],
    constant J2kLosslessCodestreamAssemblyParams &params [[buffer(3)]],
    device J2kCodestreamAssemblyStatus *status [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }

    j2k_set_codestream_status(status, J2K_ENCODE_STATUS_FAIL, 0u, 0u);
    const J2kPacketEncodeStatus packet_status = tile_status[0];
    if (packet_status.code != J2K_ENCODE_STATUS_OK) {
        j2k_set_codestream_status(status, J2K_ENCODE_STATUS_FAIL, packet_status.detail, 0u);
        return;
    }
    if (params.num_components == 0u || params.num_components > 255u ||
        params.bit_depth == 0u || params.bit_depth > 16u ||
        params.num_decomposition_levels > 31u) {
        j2k_set_codestream_status(status, J2K_ENCODE_STATUS_UNSUPPORTED, 1u, 0u);
        return;
    }

    const uint tile_len = packet_status.data_len;
    const uint tile_part_len = 14u + tile_len;
    uint cursor = 0u;
    bool ok = true;

    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x4Fu);

    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x51u);
    const uint siz_len = 38u + 3u * params.num_components;
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, siz_len);
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, params.width);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, params.height);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, params.width);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, params.height);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, params.num_components);
    const uint ssiz = (params.bit_depth - 1u) | (params.signed_samples != 0u ? 0x80u : 0u);
    for (uint comp = 0u; comp < params.num_components && ok; ++comp) {
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, ssiz);
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 1u);
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 1u);
    }

    if (params.high_throughput != 0u) {
        const uint magnitude_bits = params.bit_depth - 1u;
        const uint bp = magnitude_bits <= 8u ? 0u :
            (magnitude_bits < 28u ? magnitude_bits - 8u : 13u + (magnitude_bits >> 2u));
        ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x50u);
        ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 8u);
        ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, 0x00020000u);
        ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, bp);
    }

    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x52u);
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 12u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.progression_order);
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 1u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.use_mct != 0u ? 1u : 0u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.num_decomposition_levels);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 4u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 4u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.high_throughput != 0u ? 0x40u : 0u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 1u);

    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x5Cu);
    const uint qcd_steps = 1u + 3u * params.num_decomposition_levels;
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 3u + qcd_steps);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.guard_bits << 5u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.bit_depth << 3u);
    for (uint level = 0u; level < params.num_decomposition_levels && ok; ++level) {
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, (params.bit_depth + 1u) << 3u);
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, (params.bit_depth + 1u) << 3u);
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, (params.bit_depth + 2u) << 3u);
    }

    if (params.write_tlm != 0u) {
        ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x55u);
        ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 10u);
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 0u);
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 0x22u);
        ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 0u);
        ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, tile_part_len);
    }

    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x90u);
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 10u);
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, tile_part_len);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 1u);
    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x93u);

    if (!ok || cursor + tile_len + 2u > params.output_capacity) {
        j2k_set_codestream_status(status, J2K_ENCODE_STATUS_FAIL, 2u, cursor);
        return;
    }
    for (uint idx = 0u; idx < tile_len; ++idx) {
        out[cursor + idx] = tile_data[idx];
    }
    cursor += tile_len;
    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0xD9u);
    if (!ok) {
        j2k_set_codestream_status(status, J2K_ENCODE_STATUS_FAIL, 3u, cursor);
        return;
    }

    j2k_set_codestream_status(status, J2K_ENCODE_STATUS_OK, 0u, cursor);
}

struct J2kBatchedCodestreamAssemblyJob {
    uint tile_data_offset;
    uint codestream_offset;
    uint width;
    uint height;
    uint num_components;
    uint bit_depth;
    uint signed_samples;
    uint num_decomposition_levels;
    uint use_mct;
    uint guard_bits;
    uint progression_order;
    uint write_tlm;
    uint high_throughput;
    uint output_capacity;
};

kernel void j2k_assemble_lossless_codestream_batched(
    device const uchar *tile_data [[buffer(0)]],
    device const J2kPacketEncodeStatus *tile_status [[buffer(1)]],
    device uchar *out [[buffer(2)]],
    device const J2kBatchedCodestreamAssemblyJob *jobs [[buffer(3)]],
    device J2kCodestreamAssemblyStatus *status [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const J2kBatchedCodestreamAssemblyJob job = jobs[gid];
    device J2kCodestreamAssemblyStatus *tile_status_out = status + gid;
    device uchar *tile_out = out + job.codestream_offset;
    device const uchar *packet_data = tile_data + job.tile_data_offset;
    const J2kPacketEncodeStatus packet_status = tile_status[gid];

    j2k_set_codestream_status(tile_status_out, J2K_ENCODE_STATUS_FAIL, 0u, 0u);
    if (packet_status.code != J2K_ENCODE_STATUS_OK) {
        j2k_set_codestream_status(tile_status_out, J2K_ENCODE_STATUS_FAIL, packet_status.detail, 0u);
        return;
    }
    if (job.num_components == 0u || job.num_components > 255u ||
        job.bit_depth == 0u || job.bit_depth > 16u ||
        job.num_decomposition_levels > 31u) {
        j2k_set_codestream_status(tile_status_out, J2K_ENCODE_STATUS_UNSUPPORTED, 1u, 0u);
        return;
    }

    const uint tile_len = packet_status.data_len;
    const uint tile_part_len = 14u + tile_len;
    uint cursor = 0u;
    bool ok = true;

    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x4Fu);

    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x51u);
    const uint siz_len = 38u + 3u * job.num_components;
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, siz_len);
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, job.width);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, job.height);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, job.width);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, job.height);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, job.num_components);
    const uint ssiz = (job.bit_depth - 1u) | (job.signed_samples != 0u ? 0x80u : 0u);
    for (uint comp = 0u; comp < job.num_components && ok; ++comp) {
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, ssiz);
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 1u);
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 1u);
    }

    if (job.high_throughput != 0u) {
        const uint magnitude_bits = job.bit_depth - 1u;
        const uint bp = magnitude_bits <= 8u ? 0u :
            (magnitude_bits < 28u ? magnitude_bits - 8u : 13u + (magnitude_bits >> 2u));
        ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x50u);
        ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 8u);
        ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, 0x00020000u);
        ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, bp);
    }

    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x52u);
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 12u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.progression_order);
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 1u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.use_mct != 0u ? 1u : 0u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.num_decomposition_levels);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 4u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 4u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.high_throughput != 0u ? 0x40u : 0u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 1u);

    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x5Cu);
    const uint qcd_steps = 1u + 3u * job.num_decomposition_levels;
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 3u + qcd_steps);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.guard_bits << 5u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.bit_depth << 3u);
    for (uint level = 0u; level < job.num_decomposition_levels && ok; ++level) {
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, (job.bit_depth + 1u) << 3u);
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, (job.bit_depth + 1u) << 3u);
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, (job.bit_depth + 2u) << 3u);
    }

    if (job.write_tlm != 0u) {
        ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x55u);
        ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 10u);
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 0u);
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 0x22u);
        ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 0u);
        ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, tile_part_len);
    }

    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x90u);
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 10u);
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, tile_part_len);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 1u);
    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x93u);

    if (!ok || cursor + tile_len + 2u > job.output_capacity) {
        j2k_set_codestream_status(tile_status_out, J2K_ENCODE_STATUS_FAIL, 2u, cursor);
        return;
    }
    for (uint idx = 0u; idx < tile_len; ++idx) {
        tile_out[cursor + idx] = packet_data[idx];
    }
    cursor += tile_len;
    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0xD9u);
    if (!ok) {
        j2k_set_codestream_status(tile_status_out, J2K_ENCODE_STATUS_FAIL, 3u, cursor);
        return;
    }

    j2k_set_codestream_status(tile_status_out, J2K_ENCODE_STATUS_OK, 0u, cursor);
}

kernel void j2k_prepare_packet_blocks_from_classic_status(
    device const J2kResidentPacketBlock *resident_blocks [[buffer(0)]],
    device const J2kClassicEncodeBatchJob *tier1_jobs [[buffer(1)]],
    device const J2kClassicEncodeStatus *tier1_statuses [[buffer(2)]],
    device J2kPacketBlock *packet_blocks [[buffer(3)]],
    constant J2kResidentPacketBlockParams &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.block_count) {
        return;
    }

    const J2kResidentPacketBlock resident = resident_blocks[gid];
    J2kPacketBlock packet;
    packet.data_offset = 0u;
    packet.data_len = 0u;
    packet.num_coding_passes = 1u;
    packet.num_zero_bitplanes = 0u;
    packet.previously_included = resident.previously_included;
    packet.l_block = resident.l_block;
    packet.block_coding_mode = 0xFFFFFFFFu;
    packet.reserved0 = 0u;

    if (resident.tier1_job_index < params.tier1_job_count) {
        const J2kClassicEncodeBatchJob job = tier1_jobs[resident.tier1_job_index];
        const J2kClassicEncodeStatus tier1_status = tier1_statuses[resident.tier1_job_index];
        if (tier1_status.code == J2K_ENCODE_STATUS_OK &&
            tier1_status.data_len <= job.output_capacity &&
            tier1_status.segment_count <= job.segment_capacity) {
            packet.data_offset = job.output_offset;
            packet.data_len = tier1_status.data_len;
            packet.num_coding_passes = tier1_status.number_of_coding_passes;
            packet.num_zero_bitplanes = tier1_status.missing_bit_planes;
            packet.block_coding_mode = resident.block_coding_mode;
        } else {
            packet.reserved0 = tier1_status.detail;
        }
    }

    packet_blocks[gid] = packet;
}

kernel void j2k_prepare_packet_blocks_from_ht_status(
    device const J2kResidentPacketBlock *resident_blocks [[buffer(0)]],
    device const J2kHtEncodeBatchJob *tier1_jobs [[buffer(1)]],
    device const J2kHtEncodeStatus *tier1_statuses [[buffer(2)]],
    device J2kPacketBlock *packet_blocks [[buffer(3)]],
    constant J2kResidentPacketBlockParams &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.block_count) {
        return;
    }

    const J2kResidentPacketBlock resident = resident_blocks[gid];
    J2kPacketBlock packet;
    packet.data_offset = 0u;
    packet.data_len = 0u;
    packet.num_coding_passes = 1u;
    packet.num_zero_bitplanes = 0u;
    packet.previously_included = resident.previously_included;
    packet.l_block = resident.l_block;
    packet.block_coding_mode = 0xFFFFFFFFu;
    packet.reserved0 = 0u;

    if (resident.tier1_job_index < params.tier1_job_count) {
        const J2kHtEncodeBatchJob job = tier1_jobs[resident.tier1_job_index];
        const J2kHtEncodeStatus tier1_status = tier1_statuses[resident.tier1_job_index];
        if (tier1_status.code == J2K_ENCODE_STATUS_OK &&
            tier1_status.data_len <= job.output_capacity) {
            packet.data_offset = job.output_offset;
            packet.data_len = tier1_status.data_len;
            packet.num_coding_passes = tier1_status.num_coding_passes;
            packet.num_zero_bitplanes = tier1_status.num_zero_bitplanes;
            packet.block_coding_mode = resident.block_coding_mode;
        } else {
            packet.reserved0 = tier1_status.detail;
        }
    }

    packet_blocks[gid] = packet;
}

inline void j2k_packet_writer_init(thread J2kPacketBitWriter &writer, device uchar *data, uint capacity) {
    writer.data = data;
    writer.capacity = capacity;
    writer.len = 0u;
    writer.buffer = 0u;
    writer.bits_in_buffer = 0u;
    writer.last_byte_was_ff = 0u;
    writer.failed = 0u;
}

inline void j2k_packet_flush_byte(thread J2kPacketBitWriter &writer) {
    const uint limit = writer.last_byte_was_ff != 0u ? 7u : 8u;
    const uchar byte = uchar(writer.buffer >> (writer.bits_in_buffer - limit));
    if (writer.len >= writer.capacity) {
        writer.failed = 1u;
        return;
    }
    writer.data[writer.len] = byte;
    writer.len += 1u;
    writer.last_byte_was_ff = byte == uchar(0xFFu) ? 1u : 0u;
    writer.bits_in_buffer -= limit;
    writer.buffer &= writer.bits_in_buffer == 0u ? 0u : ((1u << writer.bits_in_buffer) - 1u);
}

inline void j2k_packet_write_bit(thread J2kPacketBitWriter &writer, uint bit) {
    writer.buffer = (writer.buffer << 1u) | (bit & 1u);
    writer.bits_in_buffer += 1u;
    const uint limit = writer.last_byte_was_ff != 0u ? 7u : 8u;
    if (writer.bits_in_buffer >= limit) {
        j2k_packet_flush_byte(writer);
    }
}

inline void j2k_packet_write_bits(thread J2kPacketBitWriter &writer, uint value, uint count) {
    for (int bit = int(count) - 1; bit >= 0; --bit) {
        j2k_packet_write_bit(writer, (value >> uint(bit)) & 1u);
    }
}

inline void j2k_packet_writer_finish(thread J2kPacketBitWriter &writer) {
    if (writer.bits_in_buffer > 0u) {
        const uint limit = writer.last_byte_was_ff != 0u ? 7u : 8u;
        const uint shift = limit - writer.bits_in_buffer;
        const uchar byte = uchar(writer.buffer << shift);
        if (writer.len >= writer.capacity) {
            writer.failed = 1u;
            return;
        }
        writer.data[writer.len] = byte;
        writer.len += 1u;
        writer.last_byte_was_ff = byte == uchar(0xFFu) ? 1u : 0u;
        writer.buffer = 0u;
        writer.bits_in_buffer = 0u;
    }
}

inline uint j2k_packet_ilog2(uint value) {
    return value == 0u ? 0u : 31u - clz(value);
}

inline bool j2k_packet_value_fits(uint value, uint bits) {
    return bits >= 32u || value < (1u << bits);
}

inline uint j2k_packet_bits_for_length(uint l_block, uint passes) {
    const uint log2_passes = passes <= 1u ? 0u : j2k_packet_ilog2(passes);
    return l_block + log2_passes;
}

inline uint j2k_packet_bits_for_ht_length(uint l_block, uint passes) {
    const uint placeholder_groups = (passes > 0u ? passes - 1u : 0u) / 3u;
    const uint placeholder_passes = placeholder_groups * 3u;
    return l_block + j2k_packet_ilog2(placeholder_passes + 1u);
}

inline void j2k_packet_encode_num_passes(uint passes, thread J2kPacketBitWriter &writer) {
    if (passes == 1u) {
        j2k_packet_write_bit(writer, 0u);
    } else if (passes == 2u) {
        j2k_packet_write_bits(writer, 0b10u, 2u);
    } else if (passes == 3u) {
        j2k_packet_write_bits(writer, 0b1100u, 4u);
    } else if (passes == 4u) {
        j2k_packet_write_bits(writer, 0b1101u, 4u);
    } else if (passes == 5u) {
        j2k_packet_write_bits(writer, 0b1110u, 4u);
    } else if (passes <= 36u) {
        j2k_packet_write_bits(writer, 0b1111u, 4u);
        j2k_packet_write_bits(writer, passes - 6u, 5u);
    } else {
        j2k_packet_write_bits(writer, 0x1FFu, 9u);
        j2k_packet_write_bits(writer, passes - 37u, 7u);
    }
}

inline void j2k_packet_encode_num_ht_passes(uint passes, thread J2kPacketBitWriter &writer) {
    if (passes == 1u) {
        j2k_packet_write_bit(writer, 0u);
    } else if (passes == 2u) {
        j2k_packet_write_bits(writer, 0b10u, 2u);
    } else if (passes <= 5u) {
        j2k_packet_write_bits(writer, 0b11u, 2u);
        j2k_packet_write_bits(writer, passes - 3u, 2u);
    } else if (passes <= 36u) {
        j2k_packet_write_bits(writer, 0b11u, 2u);
        j2k_packet_write_bits(writer, 0b11u, 2u);
        j2k_packet_write_bits(writer, passes - 6u, 5u);
    } else {
        j2k_packet_write_bits(writer, 0b11u, 2u);
        j2k_packet_write_bits(writer, 0b11u, 2u);
        j2k_packet_write_bits(writer, 31u, 5u);
        j2k_packet_write_bits(writer, passes - 37u, 7u);
    }
}

inline void j2k_packet_encode_length(
    uint length,
    thread uint &l_block,
    uint num_bits,
    thread J2kPacketBitWriter &writer
) {
    while (!j2k_packet_value_fits(length, num_bits)) {
        j2k_packet_write_bit(writer, 1u);
        l_block += 1u;
        num_bits += 1u;
    }
    j2k_packet_write_bit(writer, 0u);
    j2k_packet_write_bits(writer, length, num_bits);
}

inline uint j2k_packet_tree_offsets(
    uint width,
    uint height,
    thread uint *level_offsets,
    thread uint *level_widths,
    thread uint *level_heights,
    thread uint &levels
) {
    uint total = 0u;
    uint w = width;
    uint h = height;
    levels = 0u;
    while (true) {
        level_offsets[levels] = total;
        level_widths[levels] = w;
        level_heights[levels] = h;
        total += w * h;
        levels += 1u;
        if (w <= 1u && h <= 1u) {
            break;
        }
        w = (w + 1u) / 2u;
        h = (h + 1u) / 2u;
    }
    return total;
}

inline bool j2k_packet_prepare_tree(
    device const J2kPacketBlock *blocks,
    uint block_offset,
    uint block_count,
    uint num_cbs_x,
    uint num_cbs_y,
    bool zero_bitplanes,
    uint inclusion_layer,
    device uint *value,
    device uint *current,
    device uint *known,
    uint node_capacity,
    thread uint *level_offsets,
    thread uint *level_widths,
    thread uint *level_heights,
    thread uint &levels
) {
    if (num_cbs_x == 0u || num_cbs_y == 0u || num_cbs_x * num_cbs_y != block_count) {
        return false;
    }
    const uint node_count =
        j2k_packet_tree_offsets(num_cbs_x, num_cbs_y, level_offsets, level_widths, level_heights, levels);
    if (node_count > node_capacity || levels > 16u) {
        return false;
    }
    for (uint idx = 0u; idx < node_count; ++idx) {
        value[idx] = 0u;
        current[idx] = 0u;
        known[idx] = 0u;
    }
    for (uint idx = 0u; idx < block_count; ++idx) {
        const J2kPacketBlock block = blocks[block_offset + idx];
        value[idx] = zero_bitplanes
            ? block.num_zero_bitplanes
            : (block.num_coding_passes > 0u ? inclusion_layer : 0x7FFFFFFFu);
    }
    for (uint level = 1u; level < levels; ++level) {
        const uint prev_w = level_widths[level - 1u];
        const uint prev_h = level_heights[level - 1u];
        const uint cur_w = level_widths[level];
        const uint cur_h = level_heights[level];
        for (uint py = 0u; py < cur_h; ++py) {
            for (uint px = 0u; px < cur_w; ++px) {
                uint min_value = 0xFFFFFFFFu;
                for (uint dy = 0u; dy < 2u; ++dy) {
                    const uint cy = py * 2u + dy;
                    if (cy >= prev_h) {
                        continue;
                    }
                    for (uint dx = 0u; dx < 2u; ++dx) {
                        const uint cx = px * 2u + dx;
                        if (cx >= prev_w) {
                            continue;
                        }
                        const uint child = level_offsets[level - 1u] + cy * prev_w + cx;
                        min_value = min(min_value, value[child]);
                    }
                }
                value[level_offsets[level] + py * cur_w + px] = min_value;
            }
        }
    }
    return true;
}

inline void j2k_packet_tree_encode(
    uint x,
    uint y,
    uint threshold,
    device uint *value,
    device uint *current,
    device uint *known,
    thread uint *level_offsets,
    thread uint *level_widths,
    uint levels,
    thread J2kPacketBitWriter &writer
) {
    thread uint path[16];
    uint cx = x;
    uint cy = y;
    for (uint level = 0u; level < levels; ++level) {
        path[level] = level_offsets[level] + cy * level_widths[level] + cx;
        cx /= 2u;
        cy /= 2u;
    }

    uint parent_val = 0u;
    for (int level = int(levels) - 1; level >= 0; --level) {
        const uint node = path[uint(level)];
        const uint start = max(current[node], parent_val);
        if (known[node] == 0u) {
            const uint target = min(value[node], threshold);
            for (uint v = start; v < target; ++v) {
                j2k_packet_write_bit(writer, 0u);
            }
            if (value[node] < threshold) {
                j2k_packet_write_bit(writer, 1u);
                known[node] = 1u;
            }
            current[node] = target;
        }
        parent_val = current[node];
    }
}

kernel void j2k_encode_packetization(
    device const J2kPacketResolution *resolutions [[buffer(0)]],
    device const J2kPacketSubband *subbands [[buffer(1)]],
    device const J2kPacketBlock *blocks [[buffer(2)]],
    device const uchar *payload [[buffer(3)]],
    device uchar *out [[buffer(4)]],
    device uchar *header [[buffer(5)]],
    device uint *tree_scratch [[buffer(6)]],
    constant J2kPacketEncodeParams &params [[buffer(7)]],
    device J2kPacketEncodeStatus *status [[buffer(8)]],
    device const J2kPacketDescriptor *descriptors [[buffer(9)]],
    device J2kPacketStateBlock *state_blocks [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }

    j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 0u, 0u);

    const uint node_capacity = params.scratch_node_capacity;
    device uint *inc_value = tree_scratch;
    device uint *inc_current = tree_scratch + node_capacity;
    device uint *inc_known = tree_scratch + node_capacity * 2u;
    device uint *zbp_value = tree_scratch + node_capacity * 3u;
    device uint *zbp_current = tree_scratch + node_capacity * 4u;
    device uint *zbp_known = tree_scratch + node_capacity * 5u;

    uint out_len = 0u;
    const uint packet_count =
        params.descriptor_count > 0u ? params.descriptor_count : params.resolution_count;
    for (uint packet_order_idx = 0u; packet_order_idx < packet_count; ++packet_order_idx) {
        const bool has_descriptor = params.descriptor_count > 0u;
        const J2kPacketDescriptor descriptor = has_descriptor
            ? descriptors[packet_order_idx]
            : J2kPacketDescriptor{packet_order_idx, packet_order_idx, 0u, packet_order_idx, 0u, 0u, 0u, 0u};
        const uint packet_index = descriptor.packet_index;
        if (packet_index >= params.resolution_count) {
            j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 6u, 0u);
            return;
        }
        const J2kPacketResolution resolution = resolutions[packet_index];
        uint state_block_cursor = descriptor.state_block_offset;
        bool any_data = false;
        for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
            const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
            for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                if (blocks[subband.block_offset + block_idx].num_coding_passes > 0u) {
                    any_data = true;
                    break;
                }
            }
        }

        thread J2kPacketBitWriter writer;
        j2k_packet_writer_init(writer, header, params.header_capacity);
        if (!any_data) {
            j2k_packet_write_bit(writer, 0u);
            j2k_packet_writer_finish(writer);
        } else {
            j2k_packet_write_bit(writer, 1u);
            for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
                const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
                const uint subband_state_block_offset = state_block_cursor;
                state_block_cursor += subband.block_count;
                thread uint level_offsets[16];
                thread uint level_widths[16];
                thread uint level_heights[16];
                uint levels = 0u;
                if (!j2k_packet_prepare_tree(
                        blocks,
                        subband.block_offset,
                        subband.block_count,
                        subband.num_cbs_x,
                        subband.num_cbs_y,
                        false,
                        descriptor.layer,
                        inc_value,
                        inc_current,
                        inc_known,
                        node_capacity,
                        level_offsets,
                        level_widths,
                        level_heights,
                        levels)) {
                    j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 1u, 0u);
                    return;
                }
                thread uint z_level_offsets[16];
                thread uint z_level_widths[16];
                thread uint z_level_heights[16];
                uint z_levels = 0u;
                if (!j2k_packet_prepare_tree(
                        blocks,
                        subband.block_offset,
                        subband.block_count,
                        subband.num_cbs_x,
                        subband.num_cbs_y,
                        true,
                        descriptor.layer,
                        zbp_value,
                        zbp_current,
                        zbp_known,
                        node_capacity,
                        z_level_offsets,
                        z_level_widths,
                        z_level_heights,
                        z_levels)) {
                    j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 2u, 0u);
                    return;
                }

                for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                    const uint x = block_idx % subband.num_cbs_x;
                    const uint y = block_idx / subband.num_cbs_x;
                    const J2kPacketBlock block = blocks[subband.block_offset + block_idx];
                    const uint state_block_index = subband_state_block_offset + block_idx;
                    uint previously_included = block.previously_included;
                    uint local_l_block = block.l_block;
                    if (has_descriptor) {
                        previously_included = state_blocks[state_block_index].previously_included;
                        local_l_block = state_blocks[state_block_index].l_block;
                    }
                    if (previously_included == 0u) {
                        j2k_packet_tree_encode(
                            x,
                            y,
                            descriptor.layer + 1u,
                            inc_value,
                            inc_current,
                            inc_known,
                            level_offsets,
                            level_widths,
                            levels,
                            writer
                        );
                        if (block.num_coding_passes == 0u) {
                            continue;
                        }
                        j2k_packet_tree_encode(
                            x,
                            y,
                            block.num_zero_bitplanes + 1u,
                            zbp_value,
                            zbp_current,
                            zbp_known,
                            z_level_offsets,
                            z_level_widths,
                            z_levels,
                            writer
                        );
                    } else if (block.num_coding_passes > 0u) {
                        j2k_packet_write_bit(writer, 1u);
                    } else {
                        j2k_packet_write_bit(writer, 0u);
                        continue;
                    }

                    if (block.block_coding_mode == 0u) {
                        const uint num_bits =
                            j2k_packet_bits_for_length(local_l_block, block.num_coding_passes);
                        j2k_packet_encode_num_passes(block.num_coding_passes, writer);
                        j2k_packet_encode_length(block.data_len, local_l_block, num_bits, writer);
                    } else if (block.block_coding_mode == 1u) {
                        const uint num_bits =
                            j2k_packet_bits_for_ht_length(local_l_block, block.num_coding_passes);
                        j2k_packet_encode_num_ht_passes(block.num_coding_passes, writer);
                        j2k_packet_encode_length(block.data_len, local_l_block, num_bits, writer);
                    } else {
                        j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 7u, block.reserved0);
                        return;
                    }
                    if (has_descriptor) {
                        state_blocks[state_block_index].previously_included = 1u;
                        state_blocks[state_block_index].l_block = local_l_block;
                    }
                }
            }
            j2k_packet_writer_finish(writer);
        }

        if (writer.failed != 0u || out_len + writer.len > params.output_capacity) {
            j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 3u, 0u);
            return;
        }
        for (uint idx = 0u; idx < writer.len; ++idx) {
            out[out_len + idx] = header[idx];
        }
        out_len += writer.len;
        if (writer.len > 0u && header[writer.len - 1u] == uchar(0xFFu)) {
            if (out_len >= params.output_capacity) {
                j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 4u, 0u);
                return;
            }
            out[out_len] = uchar(0u);
            out_len += 1u;
        }

        if (any_data) {
            for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
                const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
                for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                    const J2kPacketBlock block = blocks[subband.block_offset + block_idx];
                    if (block.num_coding_passes == 0u) {
                        continue;
                    }
                    if (out_len + block.data_len > params.output_capacity) {
                        j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 5u, 0u);
                        return;
                    }
                    for (uint byte_idx = 0u; byte_idx < block.data_len; ++byte_idx) {
                        out[out_len + byte_idx] = payload[block.data_offset + byte_idx];
                    }
                    out_len += block.data_len;
                }
            }
        }
    }

    j2k_set_packet_status(status, J2K_ENCODE_STATUS_OK, 0u, out_len);
}

struct J2kBatchedPacketEncodeJob {
    uint resolution_offset;
    uint subband_offset;
    uint block_offset;
    uint descriptor_offset;
    uint state_block_offset;
    uint output_offset;
    uint header_offset;
    uint scratch_offset;
    uint resolution_count;
    uint num_layers;
    uint num_components;
    uint code_block_count;
    uint subband_count;
    uint descriptor_count;
    uint output_capacity;
    uint header_capacity;
    uint scratch_node_capacity;
};

kernel void j2k_encode_packetization_batched(
    device const J2kPacketResolution *all_resolutions [[buffer(0)]],
    device const J2kPacketSubband *all_subbands [[buffer(1)]],
    device const J2kPacketBlock *all_blocks [[buffer(2)]],
    device const uchar *payload [[buffer(3)]],
    device uchar *all_out [[buffer(4)]],
    device uchar *all_header [[buffer(5)]],
    device uint *all_tree_scratch [[buffer(6)]],
    device const J2kBatchedPacketEncodeJob *jobs [[buffer(7)]],
    device J2kPacketEncodeStatus *all_status [[buffer(8)]],
    device const J2kPacketDescriptor *all_descriptors [[buffer(9)]],
    device J2kPacketStateBlock *all_state_blocks [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    const J2kBatchedPacketEncodeJob job = jobs[gid];
    device const J2kPacketResolution *resolutions = all_resolutions + job.resolution_offset;
    device const J2kPacketSubband *subbands = all_subbands + job.subband_offset;
    device const J2kPacketBlock *blocks = all_blocks + job.block_offset;
    device uchar *out = all_out + job.output_offset;
    device uchar *header = all_header + job.header_offset;
    device uint *tree_scratch = all_tree_scratch + job.scratch_offset;
    device J2kPacketEncodeStatus *status = all_status + gid;
    device const J2kPacketDescriptor *descriptors = all_descriptors + job.descriptor_offset;
    device J2kPacketStateBlock *state_blocks = all_state_blocks + job.state_block_offset;

    J2kPacketEncodeParams params;
    params.resolution_count = job.resolution_count;
    params.num_layers = job.num_layers;
    params.num_components = job.num_components;
    params.code_block_count = job.code_block_count;
    params.subband_count = job.subband_count;
    params.descriptor_count = job.descriptor_count;
    params.output_capacity = job.output_capacity;
    params.header_capacity = job.header_capacity;
    params.scratch_node_capacity = job.scratch_node_capacity;

    j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 0u, 0u);

    const uint node_capacity = params.scratch_node_capacity;
    device uint *inc_value = tree_scratch;
    device uint *inc_current = tree_scratch + node_capacity;
    device uint *inc_known = tree_scratch + node_capacity * 2u;
    device uint *zbp_value = tree_scratch + node_capacity * 3u;
    device uint *zbp_current = tree_scratch + node_capacity * 4u;
    device uint *zbp_known = tree_scratch + node_capacity * 5u;

    uint out_len = 0u;
    const uint packet_count =
        params.descriptor_count > 0u ? params.descriptor_count : params.resolution_count;
    for (uint packet_order_idx = 0u; packet_order_idx < packet_count; ++packet_order_idx) {
        const bool has_descriptor = params.descriptor_count > 0u;
        const J2kPacketDescriptor descriptor = has_descriptor
            ? descriptors[packet_order_idx]
            : J2kPacketDescriptor{packet_order_idx, packet_order_idx, 0u, packet_order_idx, 0u, 0u, 0u, 0u};
        const uint packet_index = descriptor.packet_index;
        if (packet_index >= params.resolution_count) {
            j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 6u, 0u);
            return;
        }
        const J2kPacketResolution resolution = resolutions[packet_index];
        uint state_block_cursor = descriptor.state_block_offset;
        bool any_data = false;
        for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
            const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
            for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                if (blocks[subband.block_offset + block_idx].num_coding_passes > 0u) {
                    any_data = true;
                    break;
                }
            }
        }

        thread J2kPacketBitWriter writer;
        j2k_packet_writer_init(writer, header, params.header_capacity);
        if (!any_data) {
            j2k_packet_write_bit(writer, 0u);
            j2k_packet_writer_finish(writer);
        } else {
            j2k_packet_write_bit(writer, 1u);
            for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
                const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
                const uint subband_state_block_offset = state_block_cursor;
                state_block_cursor += subband.block_count;
                thread uint level_offsets[16];
                thread uint level_widths[16];
                thread uint level_heights[16];
                uint levels = 0u;
                if (!j2k_packet_prepare_tree(
                        blocks,
                        subband.block_offset,
                        subband.block_count,
                        subband.num_cbs_x,
                        subband.num_cbs_y,
                        false,
                        descriptor.layer,
                        inc_value,
                        inc_current,
                        inc_known,
                        node_capacity,
                        level_offsets,
                        level_widths,
                        level_heights,
                        levels)) {
                    j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 1u, 0u);
                    return;
                }
                thread uint z_level_offsets[16];
                thread uint z_level_widths[16];
                thread uint z_level_heights[16];
                uint z_levels = 0u;
                if (!j2k_packet_prepare_tree(
                        blocks,
                        subband.block_offset,
                        subband.block_count,
                        subband.num_cbs_x,
                        subband.num_cbs_y,
                        true,
                        descriptor.layer,
                        zbp_value,
                        zbp_current,
                        zbp_known,
                        node_capacity,
                        z_level_offsets,
                        z_level_widths,
                        z_level_heights,
                        z_levels)) {
                    j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 2u, 0u);
                    return;
                }

                for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                    const uint x = block_idx % subband.num_cbs_x;
                    const uint y = block_idx / subband.num_cbs_x;
                    const J2kPacketBlock block = blocks[subband.block_offset + block_idx];
                    const uint state_block_index = subband_state_block_offset + block_idx;
                    uint previously_included = block.previously_included;
                    uint local_l_block = block.l_block;
                    if (has_descriptor) {
                        previously_included = state_blocks[state_block_index].previously_included;
                        local_l_block = state_blocks[state_block_index].l_block;
                    }
                    if (previously_included == 0u) {
                        j2k_packet_tree_encode(
                            x,
                            y,
                            descriptor.layer + 1u,
                            inc_value,
                            inc_current,
                            inc_known,
                            level_offsets,
                            level_widths,
                            levels,
                            writer
                        );
                        if (block.num_coding_passes == 0u) {
                            continue;
                        }
                        j2k_packet_tree_encode(
                            x,
                            y,
                            block.num_zero_bitplanes + 1u,
                            zbp_value,
                            zbp_current,
                            zbp_known,
                            z_level_offsets,
                            z_level_widths,
                            z_levels,
                            writer
                        );
                    } else if (block.num_coding_passes > 0u) {
                        j2k_packet_write_bit(writer, 1u);
                    } else {
                        j2k_packet_write_bit(writer, 0u);
                        continue;
                    }

                    if (block.block_coding_mode == 0u) {
                        const uint num_bits =
                            j2k_packet_bits_for_length(local_l_block, block.num_coding_passes);
                        j2k_packet_encode_num_passes(block.num_coding_passes, writer);
                        j2k_packet_encode_length(block.data_len, local_l_block, num_bits, writer);
                    } else if (block.block_coding_mode == 1u) {
                        const uint num_bits =
                            j2k_packet_bits_for_ht_length(local_l_block, block.num_coding_passes);
                        j2k_packet_encode_num_ht_passes(block.num_coding_passes, writer);
                        j2k_packet_encode_length(block.data_len, local_l_block, num_bits, writer);
                    } else {
                        j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 7u, block.reserved0);
                        return;
                    }
                    if (has_descriptor) {
                        state_blocks[state_block_index].previously_included = 1u;
                        state_blocks[state_block_index].l_block = local_l_block;
                    }
                }
            }
            j2k_packet_writer_finish(writer);
        }

        if (writer.failed != 0u || out_len + writer.len > params.output_capacity) {
            j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 3u, 0u);
            return;
        }
        for (uint idx = 0u; idx < writer.len; ++idx) {
            out[out_len + idx] = header[idx];
        }
        out_len += writer.len;
        if (writer.len > 0u && header[writer.len - 1u] == uchar(0xFFu)) {
            if (out_len >= params.output_capacity) {
                j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 4u, 0u);
                return;
            }
            out[out_len] = uchar(0u);
            out_len += 1u;
        }

        if (any_data) {
            for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
                const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
                for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                    const J2kPacketBlock block = blocks[subband.block_offset + block_idx];
                    if (block.num_coding_passes == 0u) {
                        continue;
                    }
                    if (out_len + block.data_len > params.output_capacity) {
                        j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 5u, 0u);
                        return;
                    }
                    for (uint byte_idx = 0u; byte_idx < block.data_len; ++byte_idx) {
                        out[out_len + byte_idx] = payload[block.data_offset + byte_idx];
                    }
                    out_len += block.data_len;
                }
            }
        }
    }

    j2k_set_packet_status(status, J2K_ENCODE_STATUS_OK, 0u, out_len);
}
