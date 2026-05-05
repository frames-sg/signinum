#include <metal_stdlib>
using namespace metal;

struct J2kHtCleanupParams {
    uint width;
    uint height;
    uint coded_len;
    uint cleanup_length;
    uint refinement_length;
    uint missing_msbs;
    uint num_bitplanes;
    uint number_of_coding_passes;
    uint output_stride;
    uint output_offset;
    float dequantization_step;
    uint stripe_causal;
};

struct J2kHtCleanupBatchJob {
    uint coded_offset;
    uint width;
    uint height;
    uint coded_len;
    uint cleanup_length;
    uint refinement_length;
    uint missing_msbs;
    uint num_bitplanes;
    uint number_of_coding_passes;
    uint output_stride;
    uint output_offset;
    float dequantization_step;
    uint stripe_causal;
};

struct J2kHtRepeatedBatchParams {
    uint job_count;
    uint output_plane_len;
    uint batch_count;
};

struct J2kHtStatus {
    uint code;
    uint detail;
    uint reserved0;
    uint reserved1;
};

constant uint J2K_HT_STATUS_OK = 0u;
constant uint J2K_HT_STATUS_FAIL = 1u;
constant uint J2K_HT_STATUS_UNSUPPORTED = 2u;

constant uint J2K_HT_MAX_WIDTH = 256u;
constant uint J2K_HT_MAX_HEIGHT = 256u;
constant uint J2K_HT_MAX_COEFFICIENTS = 4096u;
constant uint J2K_HT_MAX_SSTR = 264u;
constant uint J2K_HT_MAX_SCRATCH = 3096u;
constant uint J2K_HT_MAX_VN = 130u;
constant uint J2K_HT_MAX_MSTR = 72u;
constant uint J2K_HT_MAX_SIGMA = 528u;
constant uint J2K_HT_MAX_PREV_ROW_SIG = 72u;

inline void set_ht_status(device J2kHtStatus *status, uint code, uint detail) {
    status->code = code;
    status->detail = detail;
    status->reserved0 = 0u;
    status->reserved1 = 0u;
}

struct MelDecoder {
    device const uchar *data;
    uint pos;
    uint remaining;
    bool unstuff;
    uchar current_byte;
    uchar bits_left;
    uint k;
    uint num_runs;
    ulong runs;
};

inline MelDecoder mel_decoder_new(device const uchar *data, uint lcup, uint scup) {
    MelDecoder decoder;
    decoder.data = data;
    decoder.pos = lcup - scup;
    decoder.remaining = scup - 1u;
    decoder.unstuff = false;
    decoder.current_byte = 0;
    decoder.bits_left = 0;
    decoder.k = 0u;
    decoder.num_runs = 0u;
    decoder.runs = 0u;
    return decoder;
}

inline bool mel_read_bit(thread MelDecoder &decoder, thread uint &bit) {
    if (decoder.bits_left == 0u) {
        uchar byte = decoder.remaining > 0u ? decoder.data[decoder.pos] : uchar(0xFF);
        if (decoder.remaining > 0u) {
            decoder.pos += 1u;
            decoder.remaining -= 1u;
        }
        if (decoder.remaining == 0u) {
            byte |= uchar(0x0F);
        }
        decoder.current_byte = byte;
        decoder.bits_left = uchar(8u - uint(decoder.unstuff));
        decoder.unstuff = byte == uchar(0xFF);
    }

    decoder.bits_left -= 1u;
    bit = uint((decoder.current_byte >> decoder.bits_left) & uchar(1));
    return true;
}

inline bool mel_read_bits(thread MelDecoder &decoder, uint count, thread uint &value) {
    value = 0u;
    for (uint idx = 0u; idx < count; ++idx) {
        uint bit = 0u;
        if (!mel_read_bit(decoder, bit)) {
            return false;
        }
        value = (value << 1u) | bit;
    }
    return true;
}

inline bool mel_decode_more_runs(thread MelDecoder &decoder) {
    constexpr uint MEL_EXP[13] = {0u, 0u, 0u, 1u, 1u, 1u, 2u, 2u, 2u, 3u, 3u, 4u, 5u};

    while (decoder.num_runs < 8u) {
        const uint eval = MEL_EXP[decoder.k];
        uint first = 0u;
        if (!mel_read_bit(decoder, first)) {
            return false;
        }

        uint run = 0u;
        if (first == 1u) {
            decoder.k = min(decoder.k + 1u, 12u);
            run = ((1u << eval) - 1u) << 1u;
        } else {
            decoder.k = decoder.k == 0u ? 0u : decoder.k - 1u;
            uint bits = 0u;
            if (!mel_read_bits(decoder, eval, bits)) {
                return false;
            }
            run = (bits << 1u) | 1u;
        }

        decoder.runs |= (ulong(run) << (decoder.num_runs * 7u));
        decoder.num_runs += 1u;

        if (eval == 5u && first == 0u && decoder.num_runs >= 8u) {
            break;
        }
    }

    return true;
}

inline bool mel_get_run(thread MelDecoder &decoder, thread int &run) {
    if (decoder.num_runs == 0u && !mel_decode_more_runs(decoder)) {
        return false;
    }

    run = int(decoder.runs & 0x7Ful);
    decoder.runs >>= 7u;
    decoder.num_runs -= 1u;
    return true;
}

struct ForwardBitReader {
    device const uchar *data;
    uint data_len;
    uint pos;
    ulong tmp;
    uint bits;
    bool unstuff;
    uchar pad;
};

inline ForwardBitReader forward_reader_new(device const uchar *data, uint data_len, uchar pad) {
    ForwardBitReader reader;
    reader.data = data;
    reader.data_len = data_len;
    reader.pos = 0u;
    reader.tmp = 0ul;
    reader.bits = 0u;
    reader.unstuff = false;
    reader.pad = pad;
    return reader;
}

inline void forward_reader_fill(thread ForwardBitReader &reader) {
    while (reader.bits <= 32u) {
        const uchar byte = reader.pos < reader.data_len ? reader.data[reader.pos++] : reader.pad;
        reader.tmp |= (ulong(byte) << reader.bits);
        reader.bits += 8u - uint(reader.unstuff);
        reader.unstuff = byte == uchar(0xFF);
    }
}

inline uint forward_reader_fetch(thread ForwardBitReader &reader) {
    if (reader.bits < 32u) {
        forward_reader_fill(reader);
    }
    return uint(reader.tmp);
}

inline void forward_reader_advance(thread ForwardBitReader &reader, uint count) {
    reader.tmp >>= count;
    reader.bits -= count;
}

struct ReverseBitReader {
    device const uchar *data;
    int pos;
    uint remaining;
    ulong tmp;
    uint bits;
    bool unstuff;
};

inline ReverseBitReader reverse_reader_new_vlc(
    device const uchar *data,
    uint lcup,
    uint scup
) {
    const uchar d = data[lcup - 2u];
    const ulong tmp = ulong(d >> 4);

    ReverseBitReader reader;
    reader.data = data;
    reader.pos = int(lcup) - 3;
    reader.remaining = scup - 2u;
    reader.tmp = tmp;
    reader.bits = 4u - uint((tmp & 0x7ul) == 0x7ul);
    reader.unstuff = (d | uchar(0x0F)) > uchar(0x8F);
    return reader;
}

inline ReverseBitReader reverse_reader_new_mrp(
    device const uchar *data,
    uint lcup,
    uint len2
) {
    ReverseBitReader reader;
    reader.data = data;
    reader.pos = int(lcup + len2) - 1;
    reader.remaining = len2;
    reader.tmp = 0ul;
    reader.bits = 0u;
    reader.unstuff = true;
    return reader;
}

inline void reverse_reader_fill(thread ReverseBitReader &reader) {
    while (reader.bits <= 32u) {
        const uchar byte = reader.remaining > 0u ? reader.data[reader.pos] : uchar(0u);
        if (reader.remaining > 0u) {
            reader.pos -= 1;
            reader.remaining -= 1u;
        }
        const uint d_bits = 8u - uint(reader.unstuff && (byte & uchar(0x7F)) == uchar(0x7F));
        reader.tmp |= (ulong(byte) << reader.bits);
        reader.bits += d_bits;
        reader.unstuff = byte > uchar(0x8F);
    }
}

inline uint reverse_reader_fetch(thread ReverseBitReader &reader) {
    if (reader.bits < 32u) {
        reverse_reader_fill(reader);
    }
    return uint(reader.tmp);
}

inline uint reverse_reader_advance(thread ReverseBitReader &reader, uint count) {
    reader.tmp >>= count;
    reader.bits -= count;
    return uint(reader.tmp);
}

inline uint read_u32_pair(thread const ushort *values, uint index) {
    return uint(values[index]) | (uint(values[index + 1u]) << 16u);
}

inline uint sample_mask(uint bit) {
    return 1u << (4u + bit);
}

inline int coefficient_to_i32(uint value, uint k_max) {
    const uint shift = 31u - k_max;
    const int magnitude = int((value & 0x7FFF'FFFFu) >> shift);
    return (value & 0x8000'0000u) != 0u ? -magnitude : magnitude;
}

inline float coefficient_to_float(uint value, uint k_max, float scale) {
    return float(coefficient_to_i32(value, k_max)) * scale;
}

inline uint coefficient_to_float_bits(uint value, uint k_max, float scale) {
    return as_type<uint>(coefficient_to_float(value, k_max, scale));
}

inline void decode_mag_sgn_sample_with_vn(
    thread ForwardBitReader &magsgn,
    uint inf,
    uint bit,
    uint uq,
    uint p,
    thread uint &value,
    thread uint &v_n
) {
    if ((inf & sample_mask(bit)) == 0u) {
        value = 0u;
        v_n = 0u;
        return;
    }

    const uint ms_val = forward_reader_fetch(magsgn);
    const uint m_n = uq - ((inf >> (12u + bit)) & 1u);
    forward_reader_advance(magsgn, m_n);

    value = ms_val << 31u;
    const uint mask = m_n == 0u ? 0u : (1u << m_n) - 1u;
    v_n = ms_val & mask;
    v_n |= ((inf >> (8u + bit)) & 1u) << m_n;
    v_n |= 1u;
    value |= (v_n + 2u) << (p - 1u);
}

inline void decode_ht_cleanup_impl(
    device const uchar *coded_data,
    device uint *decoded_data,
    J2kHtCleanupParams params,
    constant ushort *vlc_table0,
    constant ushort *vlc_table1,
    constant ushort *uvlc_table0,
    constant ushort *uvlc_table1,
    device J2kHtStatus *status
) {
    set_ht_status(status, J2K_HT_STATUS_OK, 0u);

    uint num_passes = params.number_of_coding_passes;
    if (num_passes > 1u && params.refinement_length == 0u) {
        num_passes = 1u;
    }

    if (params.width == 0u || params.height == 0u) {
        return;
    }
    if (params.width > J2K_HT_MAX_WIDTH || params.height > J2K_HT_MAX_HEIGHT ||
        params.width * params.height > J2K_HT_MAX_COEFFICIENTS) {
        set_ht_status(status, J2K_HT_STATUS_UNSUPPORTED, 1u);
        return;
    }
    if (params.num_bitplanes == 0u || params.num_bitplanes > 31u) {
        set_ht_status(status, J2K_HT_STATUS_FAIL, 2u);
        return;
    }
    if (num_passes > 3u || params.missing_msbs > 30u || params.missing_msbs == 30u) {
        set_ht_status(status, J2K_HT_STATUS_FAIL, 3u);
        return;
    }
    if (params.missing_msbs == 29u && num_passes > 1u) {
        num_passes = 1u;
    }

    const uint lcup = params.cleanup_length;
    if (lcup < 2u || params.coded_len < lcup + params.refinement_length) {
        set_ht_status(status, J2K_HT_STATUS_FAIL, 4u);
        return;
    }

    const uint scup = (uint(coded_data[lcup - 1u]) << 4u) + uint(coded_data[lcup - 2u] & uchar(0x0F));
    if (scup < 2u || scup > lcup || scup > 4079u) {
        set_ht_status(status, J2K_HT_STATUS_FAIL, 5u);
        return;
    }

    const uint width = params.width;
    const uint height = params.height;
    const uint stride = params.output_stride;
    const uint quad_rows = (height + 1u) / 2u;
    const uint sstr = (width + 9u) & ~7u;
    if (sstr > J2K_HT_MAX_SSTR || sstr * (quad_rows + 1u) > J2K_HT_MAX_SCRATCH) {
        set_ht_status(status, J2K_HT_STATUS_UNSUPPORTED, 6u);
        return;
    }

    thread ushort scratch[J2K_HT_MAX_SCRATCH];
    thread uint v_n_scratch[J2K_HT_MAX_VN];

    {
        thread MelDecoder mel = mel_decoder_new(coded_data, lcup, scup);
        thread ReverseBitReader vlc = reverse_reader_new_vlc(coded_data, lcup, scup);
        int run = 0;
        if (!mel_get_run(mel, run)) {
            set_ht_status(status, J2K_HT_STATUS_FAIL, 6u);
            return;
        }

        uint c_q = 0u;
        uint row_offset = 0u;
        uint x = 0u;

        while (x < width) {
            uint vlc_val = reverse_reader_fetch(vlc);
            uint t0 = uint(vlc_table0[c_q + (vlc_val & 0x7Fu)]);
            if (c_q == 0u) {
                run -= 2;
                t0 = run == -1 ? t0 : 0u;
                if (run < 0 && !mel_get_run(mel, run)) {
                    set_ht_status(status, J2K_HT_STATUS_FAIL, 7u);
                    return;
                }
            }
            scratch[row_offset] = ushort(t0);
            x += 2u;
            c_q = ((t0 & 0x10u) << 3u) | ((t0 & 0xE0u) << 2u);
            vlc_val = reverse_reader_advance(vlc, t0 & 0x7u);

            uint t1 = uint(vlc_table0[c_q + (vlc_val & 0x7Fu)]);
            if (c_q == 0u && x < width) {
                run -= 2;
                t1 = run == -1 ? t1 : 0u;
                if (run < 0 && !mel_get_run(mel, run)) {
                    set_ht_status(status, J2K_HT_STATUS_FAIL, 8u);
                    return;
                }
            }
            if (x >= width) {
                t1 = 0u;
            }
            scratch[row_offset + 2u] = ushort(t1);
            x += 2u;
            c_q = ((t1 & 0x10u) << 3u) | ((t1 & 0xE0u) << 2u);
            vlc_val = reverse_reader_advance(vlc, t1 & 0x7u);

            uint uvlc_mode = ((t0 & 0x8u) << 3u) | ((t1 & 0x8u) << 4u);
            if (uvlc_mode == 0xC0u) {
                run -= 2;
                if (run == -1) {
                    uvlc_mode += 0x40u;
                }
                if (run < 0 && !mel_get_run(mel, run)) {
                    set_ht_status(status, J2K_HT_STATUS_FAIL, 9u);
                    return;
                }
            }

            uint uvlc_entry = uint(uvlc_table0[uvlc_mode + (vlc_val & 0x3Fu)]);
            vlc_val = reverse_reader_advance(vlc, uvlc_entry & 0x7u);
            uvlc_entry >>= 3u;
            uint len = uvlc_entry & 0xFu;
            const uint tmp = vlc_val & ((1u << len) - 1u);
            vlc_val = reverse_reader_advance(vlc, len);
            uvlc_entry >>= 4u;
            len = uvlc_entry & 0x7u;
            uvlc_entry >>= 3u;
            scratch[row_offset + 1u] = ushort(1u + (uvlc_entry & 0x7u) + (tmp & ~(0xFFu << len)));
            scratch[row_offset + 3u] = ushort(1u + (uvlc_entry >> 3u) + (tmp >> len));

            row_offset += 4u;
        }
        scratch[row_offset] = 0u;
        scratch[row_offset + 1u] = 0u;

        for (uint y = 2u; y < height; y += 2u) {
            const uint row_base = (y >> 1u) * sstr;
            const uint prev_base = row_base - sstr;
            uint local_x = 0u;
            uint local_c_q = 0u;
            uint local_row_offset = row_base;

            while (local_x < width) {
                local_c_q |= (uint(scratch[prev_base + (local_row_offset - row_base)]) & 0xA0u) << 2u;
                local_c_q |= (uint(scratch[prev_base + (local_row_offset - row_base) + 2u]) & 0x20u) << 4u;

                uint vlc_val = reverse_reader_fetch(vlc);
                uint t0 = uint(vlc_table1[local_c_q + (vlc_val & 0x7Fu)]);
                if (local_c_q == 0u) {
                    run -= 2;
                    t0 = run == -1 ? t0 : 0u;
                    if (run < 0 && !mel_get_run(mel, run)) {
                        set_ht_status(status, J2K_HT_STATUS_FAIL, 10u);
                        return;
                    }
                }
                scratch[local_row_offset] = ushort(t0);
                local_x += 2u;

                local_c_q = ((t0 & 0x40u) << 2u) | ((t0 & 0x80u) << 1u);
                local_c_q |= uint(scratch[prev_base + (local_row_offset - row_base)]) & 0x80u;
                local_c_q |= (uint(scratch[prev_base + (local_row_offset - row_base) + 2u]) & 0xA0u) << 2u;
                local_c_q |= (uint(scratch[prev_base + (local_row_offset - row_base) + 4u]) & 0x20u) << 4u;
                vlc_val = reverse_reader_advance(vlc, t0 & 0x7u);

                uint t1 = uint(vlc_table1[local_c_q + (vlc_val & 0x7Fu)]);
                if (local_c_q == 0u && local_x < width) {
                    run -= 2;
                    t1 = run == -1 ? t1 : 0u;
                    if (run < 0 && !mel_get_run(mel, run)) {
                        set_ht_status(status, J2K_HT_STATUS_FAIL, 11u);
                        return;
                    }
                }
                if (local_x >= width) {
                    t1 = 0u;
                }
                scratch[local_row_offset + 2u] = ushort(t1);
                local_x += 2u;

                local_c_q = ((t1 & 0x40u) << 2u) | ((t1 & 0x80u) << 1u);
                local_c_q |= uint(scratch[prev_base + (local_row_offset - row_base) + 2u]) & 0x80u;
                vlc_val = reverse_reader_advance(vlc, t1 & 0x7u);

                const uint uvlc_mode = ((t0 & 0x8u) << 3u) | ((t1 & 0x8u) << 4u);
                uint uvlc_entry = uint(uvlc_table1[uvlc_mode + (vlc_val & 0x3Fu)]);
                vlc_val = reverse_reader_advance(vlc, uvlc_entry & 0x7u);
                uvlc_entry >>= 3u;
                uint len = uvlc_entry & 0xFu;
                const uint tmp = vlc_val & ((1u << len) - 1u);
                vlc_val = reverse_reader_advance(vlc, len);
                uvlc_entry >>= 4u;
                len = uvlc_entry & 0x7u;
                uvlc_entry >>= 3u;
                scratch[local_row_offset + 1u] =
                    ushort((uvlc_entry & 0x7u) + (tmp & ~(0xFFu << len)));
                scratch[local_row_offset + 3u] = ushort((uvlc_entry >> 3u) + (tmp >> len));

                local_row_offset += 4u;
            }

            scratch[local_row_offset] = 0u;
            scratch[local_row_offset + 1u] = 0u;
        }
    }

    const uint p = 30u - params.missing_msbs;

    {
        thread ForwardBitReader magsgn = forward_reader_new(coded_data, lcup - scup, uchar(0xFF));
        const uint v_n_width = ((width + 1u) / 2u) + 2u;
        if (v_n_width > J2K_HT_MAX_VN) {
            set_ht_status(status, J2K_HT_STATUS_UNSUPPORTED, 12u);
            return;
        }

        uint prev_v_n = 0u;
        uint x = 0u;
        uint sp = 0u;
        uint vp = 0u;
        uint dp = params.output_offset;
        const bool second_row_present = height > 1u;

        while (x < width) {
            const uint inf = uint(scratch[sp]);
            const uint uq = uint(scratch[sp + 1u]);
            if (uq > params.missing_msbs + 2u) {
                set_ht_status(status, J2K_HT_STATUS_FAIL, 13u);
                return;
            }

            uint value0 = 0u;
            uint ignored_vn = 0u;
            decode_mag_sgn_sample_with_vn(magsgn, inf, 0u, uq, p, value0, ignored_vn);
            decoded_data[dp] = value0;

            uint value1 = 0u;
            uint v_n1 = 0u;
            decode_mag_sgn_sample_with_vn(magsgn, inf, 1u, uq, p, value1, v_n1);
            if (second_row_present) {
                decoded_data[dp + stride] = value1;
            }
            v_n_scratch[vp] = prev_v_n | v_n1;
            prev_v_n = 0u;
            dp += 1u;
            x += 1u;

            if (x >= width) {
                vp += 1u;
                break;
            }

            uint value2 = 0u;
            decode_mag_sgn_sample_with_vn(magsgn, inf, 2u, uq, p, value2, ignored_vn);
            decoded_data[dp] = value2;

            uint value3 = 0u;
            uint v_n3 = 0u;
            decode_mag_sgn_sample_with_vn(magsgn, inf, 3u, uq, p, value3, v_n3);
            if (second_row_present) {
                decoded_data[dp + stride] = value3;
            }
            prev_v_n = v_n3;
            dp += 1u;
            x += 1u;
            sp += 2u;
            vp += 1u;
        }
        v_n_scratch[vp] = prev_v_n;

        for (uint y = 2u; y < height; y += 2u) {
            const uint row_base = (y >> 1u) * sstr;
            uint local_sp = row_base;
            uint local_vp = 0u;
            uint local_dp = params.output_offset + y * stride;
            uint local_prev_v_n = 0u;
            uint local_x = 0u;
            const bool local_second_row_present = y + 1u < height;

            while (local_x < width) {
                const uint inf = uint(scratch[local_sp]);
                const uint u_q = uint(scratch[local_sp + 1u]);
                uint gamma = inf & 0xF0u;
                gamma &= gamma - 0x10u;
                uint emax = v_n_scratch[local_vp] | v_n_scratch[local_vp + 1u];
                emax = 31u - clz(emax | 2u);
                const uint kappa = gamma != 0u ? emax : 1u;
                const uint uq = u_q + kappa;
                if (uq > params.missing_msbs + 2u) {
                    set_ht_status(status, J2K_HT_STATUS_FAIL, 14u);
                    return;
                }

                uint value0 = 0u;
                uint ignored_vn = 0u;
                decode_mag_sgn_sample_with_vn(magsgn, inf, 0u, uq, p, value0, ignored_vn);
                decoded_data[local_dp] = value0;

                uint value1 = 0u;
                uint v_n1 = 0u;
                decode_mag_sgn_sample_with_vn(magsgn, inf, 1u, uq, p, value1, v_n1);
                if (local_second_row_present) {
                    decoded_data[local_dp + stride] = value1;
                }
                v_n_scratch[local_vp] = local_prev_v_n | v_n1;
                local_prev_v_n = 0u;
                local_dp += 1u;
                local_x += 1u;

                if (local_x >= width) {
                    local_vp += 1u;
                    break;
                }

                uint value2 = 0u;
                decode_mag_sgn_sample_with_vn(magsgn, inf, 2u, uq, p, value2, ignored_vn);
                decoded_data[local_dp] = value2;

                uint value3 = 0u;
                uint v_n3 = 0u;
                decode_mag_sgn_sample_with_vn(magsgn, inf, 3u, uq, p, value3, v_n3);
                if (local_second_row_present) {
                    decoded_data[local_dp + stride] = value3;
                }
                local_prev_v_n = v_n3;
                local_dp += 1u;
                local_x += 1u;
                local_sp += 2u;
                local_vp += 1u;
            }

            v_n_scratch[local_vp] = local_prev_v_n;
        }
    }

    if (num_passes > 1u) {
        const uint sigma_rows = ((height + 3u) / 4u) + 1u;
        const uint mstr = ((((width + 3u) / 4u) + 2u + 7u) & ~7u);
        const uint prev_row_len = ((width + 3u) / 4u) + 8u;
        if (mstr > J2K_HT_MAX_MSTR || sigma_rows * mstr > J2K_HT_MAX_SIGMA) {
            set_ht_status(status, J2K_HT_STATUS_UNSUPPORTED, 15u);
            return;
        }
        if (prev_row_len > J2K_HT_MAX_PREV_ROW_SIG) {
            set_ht_status(status, J2K_HT_STATUS_UNSUPPORTED, 16u);
            return;
        }

        thread ushort sigma[J2K_HT_MAX_SIGMA];
        thread ushort prev_row_sig[J2K_HT_MAX_PREV_ROW_SIG];

        uint y = 0u;
        while (y < height) {
            uint sp_base = (y >> 1u) * sstr;
            uint dp_base = (y >> 2u) * mstr;
            uint local_x = 0u;
            uint sigma_sp = sp_base;
            uint sigma_dp = dp_base;
            while (local_x < width) {
                uint t0 = ((uint(scratch[sigma_sp]) & 0x30u) >> 4u)
                    | ((uint(scratch[sigma_sp]) & 0xC0u) >> 2u);
                t0 |= ((uint(scratch[sigma_sp + 2u]) & 0x30u) << 4u)
                    | ((uint(scratch[sigma_sp + 2u]) & 0xC0u) << 6u);
                uint t1 = ((uint(scratch[sigma_sp + sstr]) & 0x30u) >> 2u)
                    | (uint(scratch[sigma_sp + sstr]) & 0xC0u);
                t1 |= ((uint(scratch[sigma_sp + sstr + 2u]) & 0x30u) << 6u)
                    | ((uint(scratch[sigma_sp + sstr + 2u]) & 0xC0u) << 8u);
                sigma[sigma_dp] = ushort(t0 | t1);
                local_x += 4u;
                sigma_sp += 4u;
                sigma_dp += 1u;
            }
            sigma[sigma_dp] = 0u;
            y += 4u;
        }

        const uint sigma_tail = ((height + 3u) / 4u) * mstr;
        for (uint i = 0u; i <= (width + 3u) / 4u; ++i) {
            sigma[sigma_tail + i] = 0u;
        }

        for (uint i = 0u; i < prev_row_len; ++i) {
            prev_row_sig[i] = 0u;
        }

        thread ForwardBitReader sigprop =
            forward_reader_new(coded_data + lcup, params.refinement_length, uchar(0x00));

        for (y = 0u; y < height; y += 4u) {
            uint pattern = 0xFFFFu;
            if (height - y < 4u) {
                pattern = 0x7777u;
                if (height - y < 3u) {
                    pattern = 0x3333u;
                    if (height - y < 2u) {
                        pattern = 0x1111u;
                    }
                }
            }

            uint prev = 0u;
            const uint cur_row = (y >> 2u) * mstr;
            const uint next_row = cur_row + mstr;
            const uint dpp = params.output_offset + y * stride;

            for (uint x4 = 0u; x4 < width; x4 += 4u) {
                uint col_pattern = pattern;
                int s = int(x4) + 4 - int(width);
                s = max(s, 0);
                col_pattern >>= uint(s * 4);

                const uint idx = x4 >> 2u;
                const uint ps =
                    uint(prev_row_sig[idx]) | (uint(prev_row_sig[idx + 1u]) << 16u);
                const uint ns = read_u32_pair(sigma, next_row + idx);
                uint u = (ps & 0x8888'8888u) >> 3u;
                if (params.stripe_causal == 0u) {
                    u |= (ns & 0x1111'1111u) << 3u;
                }

                const uint cs = read_u32_pair(sigma, cur_row + idx);
                uint mbr = cs;
                mbr |= (cs & 0x7777'7777u) << 1u;
                mbr |= (cs & 0xEEEE'EEEEu) >> 1u;
                mbr |= u;
                const uint t = mbr;
                mbr |= t << 4u;
                mbr |= t >> 4u;
                mbr |= prev >> 12u;
                mbr &= col_pattern;
                mbr &= ~cs;

                uint new_sig = mbr;
                if (new_sig != 0u) {
                    uint cwd = forward_reader_fetch(sigprop);
                    uint cnt = 0u;
                    uint col_mask = 0xFu;
                    const uint inv_sig = ~cs & col_pattern;

                    for (uint i = 0u; i < 16u; i += 4u) {
                        if ((col_mask & new_sig) == 0u) {
                            col_mask <<= 4u;
                            continue;
                        }

                        uint sample_mask = 0x1111u & col_mask;
                        if ((new_sig & sample_mask) != 0u) {
                            new_sig &= ~sample_mask;
                            if ((cwd & 1u) != 0u) {
                                const uint t_bits = 0x33u << i;
                                new_sig |= t_bits & inv_sig;
                            }
                            cwd >>= 1u;
                            cnt += 1u;
                        }

                        sample_mask <<= 1u;
                        if ((new_sig & sample_mask) != 0u) {
                            new_sig &= ~sample_mask;
                            if ((cwd & 1u) != 0u) {
                                const uint t_bits = 0x76u << i;
                                new_sig |= t_bits & inv_sig;
                            }
                            cwd >>= 1u;
                            cnt += 1u;
                        }

                        sample_mask <<= 1u;
                        if ((new_sig & sample_mask) != 0u) {
                            new_sig &= ~sample_mask;
                            if ((cwd & 1u) != 0u) {
                                const uint t_bits = 0xECu << i;
                                new_sig |= t_bits & inv_sig;
                            }
                            cwd >>= 1u;
                            cnt += 1u;
                        }

                        sample_mask <<= 1u;
                        if ((new_sig & sample_mask) != 0u) {
                            new_sig &= ~sample_mask;
                            if ((cwd & 1u) != 0u) {
                                const uint t_bits = 0xC8u << i;
                                new_sig |= t_bits & inv_sig;
                            }
                            cwd >>= 1u;
                            cnt += 1u;
                        }

                        col_mask <<= 4u;
                    }

                    if (new_sig != 0u) {
                        uint sig_dp = dpp + x4;
                        const uint value = 3u << (p - 2u);
                        col_mask = 0xFu;

                        for (uint column = 0u; column < 4u; ++column) {
                            if ((col_mask & new_sig) == 0u) {
                                col_mask <<= 4u;
                                sig_dp += 1u;
                                continue;
                            }

                            uint sample_mask = 0x1111u & col_mask;
                            if ((new_sig & sample_mask) != 0u) {
                                decoded_data[sig_dp] = (cwd << 31u) | value;
                                cwd >>= 1u;
                                cnt += 1u;
                            }

                            sample_mask <<= 1u;
                            if ((new_sig & sample_mask) != 0u) {
                                decoded_data[sig_dp + stride] = (cwd << 31u) | value;
                                cwd >>= 1u;
                                cnt += 1u;
                            }

                            sample_mask <<= 1u;
                            if ((new_sig & sample_mask) != 0u) {
                                decoded_data[sig_dp + 2u * stride] = (cwd << 31u) | value;
                                cwd >>= 1u;
                                cnt += 1u;
                            }

                            sample_mask <<= 1u;
                            if ((new_sig & sample_mask) != 0u) {
                                decoded_data[sig_dp + 3u * stride] = (cwd << 31u) | value;
                                cwd >>= 1u;
                                cnt += 1u;
                            }

                            col_mask <<= 4u;
                            sig_dp += 1u;
                        }
                    }

                    forward_reader_advance(sigprop, cnt);
                }

                const uint combined_sig = new_sig | cs;
                prev_row_sig[idx] = ushort(combined_sig);
                if (idx + 1u < prev_row_len) {
                    prev_row_sig[idx + 1u] = ushort(combined_sig >> 16u);
                }

                const uint combined = combined_sig;
                uint next_prev = combined_sig;
                next_prev |= (combined & 0x7777u) << 1u;
                next_prev |= (combined & 0xEEEEu) >> 1u;
                prev = (next_prev | u) & 0xF000u;
            }
        }

        if (num_passes > 2u) {
            y = 0u;
            while (y < height) {
                uint sp_base = (y >> 1u) * sstr;
                uint dp_base = (y >> 2u) * mstr;
                uint local_x = 0u;
                uint sigma_sp = sp_base;
                uint sigma_dp = dp_base;
                while (local_x < width) {
                    uint t0 = ((uint(scratch[sigma_sp]) & 0x30u) >> 4u)
                        | ((uint(scratch[sigma_sp]) & 0xC0u) >> 2u);
                    t0 |= ((uint(scratch[sigma_sp + 2u]) & 0x30u) << 4u)
                        | ((uint(scratch[sigma_sp + 2u]) & 0xC0u) << 6u);
                    uint t1 = ((uint(scratch[sigma_sp + sstr]) & 0x30u) >> 2u)
                        | (uint(scratch[sigma_sp + sstr]) & 0xC0u);
                    t1 |= ((uint(scratch[sigma_sp + sstr + 2u]) & 0x30u) << 6u)
                        | ((uint(scratch[sigma_sp + sstr + 2u]) & 0xC0u) << 8u);
                    sigma[sigma_dp] = ushort(t0 | t1);
                    local_x += 4u;
                    sigma_sp += 4u;
                    sigma_dp += 1u;
                }
                sigma[sigma_dp] = 0u;
                y += 4u;
            }

            for (uint i = 0u; i <= (width + 3u) / 4u; ++i) {
                sigma[sigma_tail + i] = 0u;
            }

            thread ReverseBitReader magref =
                reverse_reader_new_mrp(coded_data, lcup, params.refinement_length);
            const uint half_value = 1u << (p - 2u);

            for (y = 0u; y < height; y += 4u) {
                uint cur_sig_idx = (y >> 2u) * mstr;
                const uint dpp = params.output_offset + y * stride;

                for (uint x8 = 0u; x8 < width; x8 += 8u) {
                    const uint cwd = reverse_reader_fetch(magref);
                    const uint sig = read_u32_pair(sigma, cur_sig_idx);
                    cur_sig_idx += 2u;
                    uint col_mask = 0xFu;
                    uint cwd_mut = cwd;

                    if (sig != 0u) {
                        for (uint column = 0u; column < 8u; ++column) {
                            if ((sig & col_mask) != 0u) {
                                uint mag_dp = dpp + x8 + column;
                                uint sample_mask = 0x1111'1111u & col_mask;

                                for (uint row = 0u; row < 4u; ++row) {
                                    if ((sig & sample_mask) != 0u) {
                                        uint sym = cwd_mut & 1u;
                                        sym = (1u - sym) << (p - 1u);
                                        sym |= half_value;
                                        decoded_data[mag_dp] ^= sym;
                                        cwd_mut >>= 1u;
                                    }
                                    sample_mask <<= 1u;
                                    mag_dp += stride;
                                }
                            }
                            col_mask <<= 4u;
                        }
                    }

                    reverse_reader_advance(magref, popcount(sig));
                }
            }
        }
    }

    for (uint y = 0u; y < height; ++y) {
        uint row_offset = params.output_offset + y * stride;
        for (uint x = 0u; x < width; ++x) {
            const uint idx = row_offset + x;
            decoded_data[idx] = coefficient_to_float_bits(
                decoded_data[idx],
                params.num_bitplanes,
                params.dequantization_step
            );
        }
    }
}

kernel void j2k_decode_ht_cleanup(
    device const uchar *coded_data [[buffer(0)]],
    device uint *decoded_data [[buffer(1)]],
    constant J2kHtCleanupParams &params [[buffer(2)]],
    constant ushort *vlc_table0 [[buffer(3)]],
    constant ushort *vlc_table1 [[buffer(4)]],
    constant ushort *uvlc_table0 [[buffer(5)]],
    constant ushort *uvlc_table1 [[buffer(6)]],
    device J2kHtStatus *status [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }

    decode_ht_cleanup_impl(
        coded_data,
        decoded_data,
        params,
        vlc_table0,
        vlc_table1,
        uvlc_table0,
        uvlc_table1,
        status
    );
}

kernel void j2k_decode_ht_cleanup_batched(
    device const uchar *coded_data [[buffer(0)]],
    device uint *decoded_data [[buffer(1)]],
    constant J2kHtCleanupBatchJob *jobs [[buffer(2)]],
    constant ushort *vlc_table0 [[buffer(3)]],
    constant ushort *vlc_table1 [[buffer(4)]],
    constant ushort *uvlc_table0 [[buffer(5)]],
    constant ushort *uvlc_table1 [[buffer(6)]],
    device J2kHtStatus *status [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    const constant J2kHtCleanupBatchJob &job = jobs[gid];

    J2kHtCleanupParams params;
    params.width = job.width;
    params.height = job.height;
    params.coded_len = job.coded_len;
    params.cleanup_length = job.cleanup_length;
    params.refinement_length = job.refinement_length;
    params.missing_msbs = job.missing_msbs;
    params.num_bitplanes = job.num_bitplanes;
    params.number_of_coding_passes = job.number_of_coding_passes;
    params.output_stride = job.output_stride;
    params.output_offset = job.output_offset;
    params.dequantization_step = job.dequantization_step;
    params.stripe_causal = job.stripe_causal;

    decode_ht_cleanup_impl(
        coded_data + job.coded_offset,
        decoded_data,
        params,
        vlc_table0,
        vlc_table1,
        uvlc_table0,
        uvlc_table1,
        status + gid
    );
}

kernel void j2k_decode_ht_cleanup_repeated_batched(
    device const uchar *coded_data [[buffer(0)]],
    device uint *decoded_data [[buffer(1)]],
    constant J2kHtCleanupBatchJob *jobs [[buffer(2)]],
    constant J2kHtRepeatedBatchParams &repeated [[buffer(3)]],
    constant ushort *vlc_table0 [[buffer(4)]],
    constant ushort *vlc_table1 [[buffer(5)]],
    constant ushort *uvlc_table0 [[buffer(6)]],
    constant ushort *uvlc_table1 [[buffer(7)]],
    device J2kHtStatus *status [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= repeated.job_count || gid.y >= repeated.batch_count) {
        return;
    }

    const constant J2kHtCleanupBatchJob &job = jobs[gid.x];

    J2kHtCleanupParams params;
    params.width = job.width;
    params.height = job.height;
    params.coded_len = job.coded_len;
    params.cleanup_length = job.cleanup_length;
    params.refinement_length = job.refinement_length;
    params.missing_msbs = job.missing_msbs;
    params.num_bitplanes = job.num_bitplanes;
    params.number_of_coding_passes = job.number_of_coding_passes;
    params.output_stride = job.output_stride;
    params.output_offset = job.output_offset + gid.y * repeated.output_plane_len;
    params.dequantization_step = job.dequantization_step;
    params.stripe_causal = job.stripe_causal;

    decode_ht_cleanup_impl(
        coded_data + job.coded_offset,
        decoded_data,
        params,
        vlc_table0,
        vlc_table1,
        uvlc_table0,
        uvlc_table1,
        status + gid.y * repeated.job_count + gid.x
    );
}
