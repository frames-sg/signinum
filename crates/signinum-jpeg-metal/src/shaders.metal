#include <metal_stdlib>
using namespace metal;

struct JpegPackParams {
    uint width;
    uint height;
    uint out_stride;
    uint alpha;
    uint mode;
    uint out_format;
};

struct JpegFast420Params {
    uint width;
    uint height;
    uint chroma_width;
    uint chroma_height;
    uint mcus_per_row;
    uint mcu_rows;
    uint restart_interval_mcus;
    uint restart_offset_count;
    uint restart_start_mcu;
    uint entropy_len;
    uint out_stride;
    uint alpha;
    uint out_format;
    uint origin_x;
    uint origin_y;
};

struct JpegFast420ScaledParams {
    uint scaled_width;
    uint scaled_height;
    uint chroma_width;
    uint chroma_height;
    uint mcus_per_row;
    uint mcu_rows;
    uint restart_interval_mcus;
    uint restart_offset_count;
    uint restart_start_mcu;
    uint entropy_len;
    uint scale_shift;
    uint origin_x;
    uint origin_y;
};

struct JpegFast444Params {
    uint width;
    uint height;
    uint mcus_per_row;
    uint mcu_rows;
    uint restart_interval_mcus;
    uint restart_offset_count;
    uint restart_start_mcu;
    uint entropy_len;
    uint origin_x;
    uint origin_y;
};

struct JpegFast444ScaledParams {
    uint scaled_width;
    uint scaled_height;
    uint mcus_per_row;
    uint mcu_rows;
    uint restart_interval_mcus;
    uint restart_offset_count;
    uint restart_start_mcu;
    uint entropy_len;
    uint scale_shift;
    uint origin_x;
    uint origin_y;
};

struct JpegFast420WindowedPackParams {
    uint src_width;
    uint src_height;
    uint chroma_width;
    uint chroma_height;
    uint src_x;
    uint src_y;
    uint width;
    uint height;
    uint out_stride;
    uint alpha;
    uint out_format;
};

struct JpegFast420BatchParams {
    uint width;
    uint height;
    uint chroma_width;
    uint chroma_height;
    uint mcus_per_row;
    uint mcu_rows;
    uint segment_count;
    uint tile_count;
    uint out_stride;
    uint alpha;
};

struct JpegFastRegionScaledBatchParams {
    uint scaled_width;
    uint scaled_height;
    uint chroma_width;
    uint chroma_height;
    uint mcus_per_row;
    uint mcu_rows;
    uint segment_count;
    uint tile_count;
    uint scale_shift;
    uint origin_x;
    uint origin_y;
};

struct JpegWindowedPackBatchParams {
    uint src_width;
    uint src_height;
    uint chroma_width;
    uint chroma_height;
    uint src_x;
    uint src_y;
    uint width;
    uint height;
    uint tile_count;
    uint out_stride;
    uint alpha;
    uint mode;
    uint out_format;
};

struct JpegDecodeStatus {
    uint code;
    uint detail;
    uint position;
    uint reserved;
};

struct JpegEntropyCheckpoint {
    uint mcu_index;
    uint entropy_pos;
    ulong bit_acc;
    uint bit_count;
    int y_prev_dc;
    int cb_prev_dc;
    int cr_prev_dc;
    uint reserved;
};

struct MetalHuffmanTable {
    uchar bits[16];
    ushort values_len;
    ushort reserved;
    uchar values[256];
};

struct PreparedHuffman {
    int min_code[17];
    int max_code[17];
    int val_offset[17];
    uchar values[256];
    uchar fast_symbol[512];
    uchar fast_len[512];
    ushort values_len;
};

struct BitReader {
    uint pos;
    ulong acc;
    uint bits;
};

constant uint MODE_GRAY = 0;
constant uint MODE_YCBCR = 1;
constant uint MODE_RGB = 2;

constant uint OUT_GRAY = 0;
constant uint OUT_RGB = 1;
constant uint OUT_RGBA = 2;

constant uint FAST420_STATUS_OK = 0;
constant uint FAST420_STATUS_TRUNCATED = 1;
constant uint FAST420_STATUS_HUFFMAN = 2;

constant ushort ZIGZAG[64] = {
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

constant int CONST_BITS = 13;
constant int PASS1_BITS = 2;

constant int FIX_0_211164243 = 1730;
constant int FIX_0_298631336 = 2446;
constant int FIX_0_390180644 = 3196;
constant int FIX_0_509795579 = 4176;
constant int FIX_0_541196100 = 4433;
constant int FIX_0_601344887 = 4926;
constant int FIX_0_720959822 = 5906;
constant int FIX_0_765366865 = 6270;
constant int FIX_0_850430095 = 6967;
constant int FIX_0_899976223 = 7373;
constant int FIX_1_061594337 = 8697;
constant int FIX_1_175875602 = 9633;
constant int FIX_1_272758580 = 10426;
constant int FIX_1_451774981 = 11893;
constant int FIX_1_501321110 = 12299;
constant int FIX_1_847759065 = 15137;
constant int FIX_1_961570560 = 16069;
constant int FIX_2_053119869 = 16819;
constant int FIX_2_172734803 = 17799;
constant int FIX_2_562915447 = 20995;
constant int FIX_3_072711026 = 25172;
constant int FIX_3_624509785 = 29692;

inline uchar clamp_u8(int value) {
    return uchar(clamp(value, 0, 255));
}

inline short clamp_i16(int value) {
    return short(clamp(value, int(short(-32768)), int(short(32767))));
}

inline bool refill_one_byte(
    thread BitReader &br,
    device const uchar *bytes,
    uint len
) {
    if (br.pos >= len) {
        return false;
    }
    const uint shift = 64u - 8u - br.bits;
    br.acc |= ulong(bytes[br.pos]) << shift;
    br.pos += 1;
    br.bits += 8;
    return true;
}

inline bool refill_four_bytes(
    thread BitReader &br,
    device const uchar *bytes,
    uint len
) {
    if (br.bits > 32u || br.pos + 4u > len) {
        return false;
    }
    const uint word = (uint(bytes[br.pos]) << 24)
        | (uint(bytes[br.pos + 1u]) << 16)
        | (uint(bytes[br.pos + 2u]) << 8)
        | uint(bytes[br.pos + 3u]);
    const uint shift = 64u - 32u - br.bits;
    br.acc |= ulong(word) << shift;
    br.pos += 4u;
    br.bits += 32u;
    return true;
}

inline bool refill_bits(
    thread BitReader &br,
    device const uchar *bytes,
    uint len
) {
    return refill_four_bytes(br, bytes, len) || refill_one_byte(br, bytes, len);
}

inline bool ensure_bits(
    thread BitReader &br,
    device const uchar *bytes,
    uint len,
    uint n,
    device JpegDecodeStatus *status
) {
    while (br.bits < n) {
        if (!refill_bits(br, bytes, len)) {
            status->code = FAST420_STATUS_TRUNCATED;
            status->position = br.pos;
            return false;
        }
    }
    return true;
}

inline void ensure_bits_padded(
    thread BitReader &br,
    device const uchar *bytes,
    uint len,
    uint n
) {
    while (br.bits < n) {
        if (!refill_bits(br, bytes, len)) {
            br.acc |= ulong(1) << (63u - br.bits);
            br.bits += 1;
        }
    }
}

inline uint peek_bits(thread const BitReader &br, uint n) {
    if (n == 0) {
        return 0;
    }
    return uint(br.acc >> (64u - n));
}

inline void consume_bits(thread BitReader &br, uint n) {
    br.acc <<= n;
    br.bits -= n;
}

inline int huff_extend(int value, uchar ssss) {
    if (ssss == 0) {
        return 0;
    }
    const int threshold = 1 << (ssss - 1);
    if (value < threshold) {
        return value + ((-1) << ssss) + 1;
    }
    return value;
}

inline bool receive_extend(
    thread BitReader &br,
    device const uchar *bytes,
    uint len,
    uchar ssss,
    device JpegDecodeStatus *status,
    thread int &value
) {
    if (ssss == 0) {
        value = 0;
        return true;
    }
    if (!ensure_bits(br, bytes, len, uint(ssss), status)) {
        return false;
    }
    value = huff_extend(int(peek_bits(br, uint(ssss))), ssss);
    consume_bits(br, uint(ssss));
    return true;
}

inline bool skip_receive_extend(
    thread BitReader &br,
    device const uchar *bytes,
    uint len,
    uchar ssss,
    device JpegDecodeStatus *status
) {
    if (ssss == 0) {
        return true;
    }
    if (!ensure_bits(br, bytes, len, uint(ssss), status)) {
        return false;
    }
    consume_bits(br, uint(ssss));
    return true;
}

inline bool configure_restart_thread(
    uint gid,
    uint total_mcus,
    uint restart_interval_mcus,
    uint restart_offset_count,
    uint restart_start_mcu,
    device const uint *restart_offsets,
    thread BitReader &br,
    thread uint &start_mcu,
    thread uint &end_mcu
) {
    br.pos = 0u;
    br.acc = 0u;
    br.bits = 0u;

    if (restart_interval_mcus == 0u) {
        if (gid != 0u) {
            return false;
        }
        start_mcu = 0u;
        end_mcu = total_mcus;
        return true;
    }

    if (gid >= restart_offset_count) {
        return false;
    }

    start_mcu = restart_start_mcu + gid * restart_interval_mcus;
    if (start_mcu >= total_mcus) {
        return false;
    }
    end_mcu = min(total_mcus, start_mcu + restart_interval_mcus);
    br.pos = restart_offsets[gid];
    return true;
}

inline bool configure_entropy_thread(
    uint gid,
    uint total_mcus,
    uint restart_interval_mcus,
    uint segment_count,
    uint restart_start_mcu,
    device const uint *restart_offsets,
    device const JpegEntropyCheckpoint *entropy_checkpoints,
    thread BitReader &br,
    thread uint &start_mcu,
    thread uint &end_mcu,
    thread int &y_prev_dc,
    thread int &cb_prev_dc,
    thread int &cr_prev_dc
) {
    y_prev_dc = 0;
    cb_prev_dc = 0;
    cr_prev_dc = 0;

    if (restart_interval_mcus != 0u) {
        return configure_restart_thread(
            gid,
            total_mcus,
            restart_interval_mcus,
            segment_count,
            restart_start_mcu,
            restart_offsets,
            br,
            start_mcu,
            end_mcu
        );
    }

    br.pos = 0u;
    br.acc = 0u;
    br.bits = 0u;

    if (gid >= segment_count) {
        return false;
    }

    const JpegEntropyCheckpoint checkpoint = entropy_checkpoints[gid];
    start_mcu = checkpoint.mcu_index;
    if (start_mcu >= total_mcus) {
        return false;
    }
    if (gid + 1u < segment_count) {
        end_mcu = min(total_mcus, entropy_checkpoints[gid + 1u].mcu_index);
    } else {
        end_mcu = total_mcus;
    }
    if (end_mcu <= start_mcu) {
        return false;
    }

    br.pos = checkpoint.entropy_pos;
    br.acc = checkpoint.bit_acc;
    br.bits = checkpoint.bit_count;
    y_prev_dc = checkpoint.y_prev_dc;
    cb_prev_dc = checkpoint.cb_prev_dc;
    cr_prev_dc = checkpoint.cr_prev_dc;
    return true;
}

inline bool configure_batch_entropy_thread(
    uint gid,
    uint total_mcus,
    uint segment_count,
    uint tile_count,
    device const uint *entropy_offsets,
    device const uint *entropy_lens,
    device const JpegEntropyCheckpoint *entropy_checkpoints,
    thread BitReader &br,
    thread uint &tile_index,
    thread uint &start_mcu,
    thread uint &end_mcu,
    thread uint &entropy_end,
    thread int &y_prev_dc,
    thread int &cb_prev_dc,
    thread int &cr_prev_dc
) {
    if (segment_count == 0u) {
        return false;
    }

    tile_index = gid / segment_count;
    const uint local_gid = gid - tile_index * segment_count;
    if (tile_index >= tile_count) {
        return false;
    }

    const uint checkpoint_base = tile_index * segment_count;
    const JpegEntropyCheckpoint checkpoint = entropy_checkpoints[checkpoint_base + local_gid];
    start_mcu = checkpoint.mcu_index;
    if (start_mcu >= total_mcus) {
        return false;
    }
    end_mcu = total_mcus;
    if (local_gid + 1u < segment_count) {
        end_mcu = min(total_mcus, entropy_checkpoints[checkpoint_base + local_gid + 1u].mcu_index);
    }
    if (end_mcu <= start_mcu) {
        return false;
    }

    const uint entropy_base = entropy_offsets[tile_index];
    entropy_end = entropy_base + entropy_lens[tile_index];
    br.pos = entropy_base + checkpoint.entropy_pos;
    br.acc = checkpoint.bit_acc;
    br.bits = checkpoint.bit_count;
    y_prev_dc = checkpoint.y_prev_dc;
    cb_prev_dc = checkpoint.cb_prev_dc;
    cr_prev_dc = checkpoint.cr_prev_dc;
    return true;
}

inline void prepare_huffman(
    constant MetalHuffmanTable &raw,
    thread PreparedHuffman &out
) {
    uchar huffsize[256];
    ushort huffcode[256];
    ushort huffsize_len = 0;
    for (uint i = 0; i < 17; ++i) {
        out.min_code[i] = 0x7fffffff;
        out.max_code[i] = -1;
        out.val_offset[i] = 0;
    }
    for (uint i = 0; i < raw.values_len; ++i) {
        out.values[i] = raw.values[i];
    }
    for (uint i = 0; i < 512; ++i) {
        out.fast_symbol[i] = 0;
        out.fast_len[i] = 0;
    }
    out.values_len = raw.values_len;
    for (uint len_minus_1 = 0; len_minus_1 < 16; ++len_minus_1) {
        const uchar len = uchar(len_minus_1 + 1);
        for (uchar count = 0; count < raw.bits[len_minus_1]; ++count) {
            huffsize[huffsize_len] = len;
            huffsize_len += 1;
        }
    }

    uint code = 0;
    uchar si = huffsize_len == 0 ? 0 : huffsize[0];
    for (ushort k = 0; k < huffsize_len; ++k) {
        const uchar s = huffsize[k];
        while (s != si) {
            code <<= 1;
            si += 1;
        }
        huffcode[k] = ushort(code);
        code += 1;
    }

    ushort k = 0;
    for (uint len_minus_1 = 0; len_minus_1 < 16; ++len_minus_1) {
        const uint len = len_minus_1 + 1;
        const ushort count = raw.bits[len_minus_1];
        if (count == 0) {
            continue;
        }
        out.min_code[len] = int(huffcode[k]);
        out.max_code[len] = int(huffcode[k + count - 1]);
        out.val_offset[len] = int(k) - out.min_code[len];
        k += count;
    }

    for (uint idx = 0; idx < huffsize_len; ++idx) {
        const uint len = uint(huffsize[idx]);
        if (len == 0u || len > 9u) {
            continue;
        }
        const uint prefix = uint(huffcode[idx]) << (9u - len);
        const uint fill = 1u << (9u - len);
        for (uint suffix = 0; suffix < fill; ++suffix) {
            out.fast_symbol[prefix | suffix] = raw.values[idx];
            out.fast_len[prefix | suffix] = huffsize[idx];
        }
    }
}

inline bool decode_symbol(
    thread BitReader &br,
    device const uchar *bytes,
    uint len,
    constant PreparedHuffman &table,
    device JpegDecodeStatus *status,
    thread uchar &symbol
) {
    ensure_bits_padded(br, bytes, len, 9);
    const uint fast_index = peek_bits(br, 9);
    const uchar len9 = table.fast_len[fast_index];
    if (len9 != 0) {
        consume_bits(br, uint(len9));
        symbol = table.fast_symbol[fast_index];
        return true;
    }

    ensure_bits_padded(br, bytes, len, 16);
    const int code16 = int(peek_bits(br, 16));
    for (uint length = 1; length <= 16; ++length) {
        const int code = code16 >> (16 - int(length));
        if (code <= table.max_code[length]) {
            if (code < table.min_code[length]) {
                continue;
            }
            const int idx = code + table.val_offset[length];
            if (idx < 0 || idx >= int(table.values_len)) {
                status->code = FAST420_STATUS_HUFFMAN;
                status->position = br.pos;
                return false;
            }
            consume_bits(br, length);
            symbol = table.values[idx];
            return true;
        }
    }
    status->code = FAST420_STATUS_HUFFMAN;
    status->position = br.pos;
    return false;
}

inline bool decode_block(
    thread BitReader &br,
    device const uchar *bytes,
    uint len,
    constant PreparedHuffman &dc_table,
    constant PreparedHuffman &ac_table,
    constant ushort *quant,
    thread int &prev_dc,
    device JpegDecodeStatus *status,
    thread short coeffs[64],
    thread bool &dc_only
) {
    for (uint i = 0; i < 64; ++i) {
        coeffs[i] = 0;
    }
    uchar ssss = 0;
    if (!decode_symbol(br, bytes, len, dc_table, status, ssss)) {
        return false;
    }
    if (ssss > 15) {
        status->code = FAST420_STATUS_HUFFMAN;
        status->position = br.pos;
        return false;
    }

    int diff = 0;
    if (!receive_extend(br, bytes, len, ssss, status, diff)) {
        return false;
    }
    prev_dc += diff;
    coeffs[0] = clamp_i16(prev_dc * int(quant[0]));

    dc_only = true;
    uint k = 1;
    while (k < 64) {
        uchar symbol = 0;
        if (!decode_symbol(br, bytes, len, ac_table, status, symbol)) {
            return false;
        }
        const uint run = uint(symbol >> 4);
        ssss = symbol & 0x0F;
        if (ssss == 0) {
            if (run == 15) {
                k += 16;
                continue;
            }
            break;
        }

        k += run;
        if (k >= 64) {
            status->code = FAST420_STATUS_HUFFMAN;
            status->position = br.pos;
            return false;
        }

        int value = 0;
        if (!receive_extend(br, bytes, len, ssss, status, value)) {
            return false;
        }
        coeffs[ZIGZAG[k]] = short(value * int(quant[k]));
        dc_only = false;
        k += 1;
    }
    return true;
}

inline bool decode_block_skip(
    thread BitReader &br,
    device const uchar *bytes,
    uint len,
    constant PreparedHuffman &dc_table,
    constant PreparedHuffman &ac_table,
    thread int &prev_dc,
    device JpegDecodeStatus *status
) {
    uchar ssss = 0;
    if (!decode_symbol(br, bytes, len, dc_table, status, ssss)) {
        return false;
    }
    if (ssss > 15) {
        status->code = FAST420_STATUS_HUFFMAN;
        status->position = br.pos;
        return false;
    }

    int diff = 0;
    if (!receive_extend(br, bytes, len, ssss, status, diff)) {
        return false;
    }
    prev_dc += diff;

    uint k = 1;
    while (k < 64) {
        uchar symbol = 0;
        if (!decode_symbol(br, bytes, len, ac_table, status, symbol)) {
            return false;
        }
        const uint run = uint(symbol >> 4);
        ssss = symbol & 0x0F;
        if (ssss == 0) {
            if (run == 15) {
                k += 16;
                continue;
            }
            break;
        }

        k += run;
        if (k >= 64) {
            status->code = FAST420_STATUS_HUFFMAN;
            status->position = br.pos;
            return false;
        }

        if (!skip_receive_extend(br, bytes, len, ssss, status)) {
            return false;
        }
        k += 1;
    }
    return true;
}

inline bool block_intersects_rect(
    uint block_x,
    uint block_y,
    uint block_width,
    uint block_height,
    uint rect_x,
    uint rect_y,
    uint rect_width,
    uint rect_height
) {
    const uint block_x1 = block_x + block_width;
    const uint block_y1 = block_y + block_height;
    const uint rect_x1 = rect_x + rect_width;
    const uint rect_y1 = rect_y + rect_height;
    return block_x < rect_x1 && rect_x < block_x1 && block_y < rect_y1 && rect_y < block_y1;
}

inline int descale(int value, int shift) {
    return value >> shift;
}

inline uchar descale_and_clamp(int value, int shift) {
    const int shifted = value >> shift;
    return clamp_u8(shifted + 128);
}

inline void idct_1d_column(
    thread const short input[64],
    thread int work[64],
    uint col
) {
    const int p0 = int(input[col]);
    const int p1 = int(input[col + 8]);
    const int p2 = int(input[col + 16]);
    const int p3 = int(input[col + 24]);
    const int p4 = int(input[col + 32]);
    const int p5 = int(input[col + 40]);
    const int p6 = int(input[col + 48]);
    const int p7 = int(input[col + 56]);

    if (p1 == 0 && p2 == 0 && p3 == 0 && p4 == 0 && p5 == 0 && p6 == 0 && p7 == 0) {
        const int dc = p0 << PASS1_BITS;
        work[col] = dc;
        work[col + 8] = dc;
        work[col + 16] = dc;
        work[col + 24] = dc;
        work[col + 32] = dc;
        work[col + 40] = dc;
        work[col + 48] = dc;
        work[col + 56] = dc;
        return;
    }

    const int z2 = p2;
    const int z3 = p6;
    const int z1 = (z2 + z3) * FIX_0_541196100;
    const int tmp2 = z1 - z3 * FIX_1_847759065;
    const int tmp3 = z1 + z2 * FIX_0_765366865;

    const int tmp0 = (p0 + p4) << CONST_BITS;
    const int tmp1 = (p0 - p4) << CONST_BITS;

    const int tmp10 = tmp0 + tmp3;
    const int tmp13 = tmp0 - tmp3;
    const int tmp11 = tmp1 + tmp2;
    const int tmp12 = tmp1 - tmp2;

    const int z1o = p7 + p1;
    const int z2o = p5 + p3;
    const int z3o = p7 + p3;
    const int z4o = p5 + p1;
    const int z5 = (z3o + z4o) * FIX_1_175875602;

    const int tmp0o = p7 * FIX_0_298631336;
    const int tmp1o = p5 * FIX_2_053119869;
    const int tmp2o = p3 * FIX_3_072711026;
    const int tmp3o = p1 * FIX_1_501321110;
    const int z1m = z1o * -FIX_0_899976223;
    const int z2m = z2o * -FIX_2_562915447;
    const int z3m = z3o * -FIX_1_961570560 + z5;
    const int z4m = z4o * -FIX_0_390180644 + z5;

    const int out0 = tmp0o + z1m + z3m;
    const int out1 = tmp1o + z2m + z4m;
    const int out2 = tmp2o + z2m + z3m;
    const int out3 = tmp3o + z1m + z4m;

    const int shift = CONST_BITS - PASS1_BITS;
    const int rounding = 1 << (shift - 1);
    work[col] = descale(tmp10 + out3 + rounding, shift);
    work[col + 56] = descale(tmp10 - out3 + rounding, shift);
    work[col + 8] = descale(tmp11 + out2 + rounding, shift);
    work[col + 48] = descale(tmp11 - out2 + rounding, shift);
    work[col + 16] = descale(tmp12 + out1 + rounding, shift);
    work[col + 40] = descale(tmp12 - out1 + rounding, shift);
    work[col + 24] = descale(tmp13 + out0 + rounding, shift);
    work[col + 32] = descale(tmp13 - out0 + rounding, shift);
}

inline void idct_1d_column_bottom_half_zero(
    thread const short input[64],
    thread int work[64],
    uint col
) {
    const int p0 = int(input[col]);
    const int p1 = int(input[col + 8]);
    const int p2 = int(input[col + 16]);
    const int p3 = int(input[col + 24]);

    if (p1 == 0 && p2 == 0 && p3 == 0) {
        const int dc = p0 << PASS1_BITS;
        work[col] = dc;
        work[col + 8] = dc;
        work[col + 16] = dc;
        work[col + 24] = dc;
        work[col + 32] = dc;
        work[col + 40] = dc;
        work[col + 48] = dc;
        work[col + 56] = dc;
        return;
    }

    const int z1 = p2 * FIX_0_541196100;
    const int tmp2 = z1;
    const int tmp3 = z1 + p2 * FIX_0_765366865;

    const int tmp0 = p0 << CONST_BITS;
    const int tmp1 = p0 << CONST_BITS;

    const int tmp10 = tmp0 + tmp3;
    const int tmp13 = tmp0 - tmp3;
    const int tmp11 = tmp1 + tmp2;
    const int tmp12 = tmp1 - tmp2;

    const int z5 = (p1 + p3) * FIX_1_175875602;
    const int z1m = p1 * -FIX_0_899976223;
    const int z2m = p3 * -FIX_2_562915447;
    const int z3m = p3 * -FIX_1_961570560 + z5;
    const int z4m = p1 * -FIX_0_390180644 + z5;

    const int out0 = z1m + z3m;
    const int out1 = z2m + z4m;
    const int out2 = p3 * FIX_3_072711026 + z2m + z3m;
    const int out3 = p1 * FIX_1_501321110 + z1m + z4m;

    const int shift = CONST_BITS - PASS1_BITS;
    const int rounding = 1 << (shift - 1);
    work[col] = descale(tmp10 + out3 + rounding, shift);
    work[col + 56] = descale(tmp10 - out3 + rounding, shift);
    work[col + 8] = descale(tmp11 + out2 + rounding, shift);
    work[col + 48] = descale(tmp11 - out2 + rounding, shift);
    work[col + 16] = descale(tmp12 + out1 + rounding, shift);
    work[col + 40] = descale(tmp12 - out1 + rounding, shift);
    work[col + 24] = descale(tmp13 + out0 + rounding, shift);
    work[col + 32] = descale(tmp13 - out0 + rounding, shift);
}

inline void idct_1d_row(
    thread const int work[64],
    thread uchar output[64],
    uint row
) {
    const uint base = row * 8;
    const int p0 = work[base];
    const int p1 = work[base + 1];
    const int p2 = work[base + 2];
    const int p3 = work[base + 3];
    const int p4 = work[base + 4];
    const int p5 = work[base + 5];
    const int p6 = work[base + 6];
    const int p7 = work[base + 7];

    const int shift = CONST_BITS + PASS1_BITS + 3;
    const int rounding = 1 << (shift - 1);

    if (p1 == 0 && p2 == 0 && p3 == 0 && p4 == 0 && p5 == 0 && p6 == 0 && p7 == 0) {
        const int dc_shift = PASS1_BITS + 3;
        const int dc_rounding = 1 << (dc_shift - 1);
        const uchar pixel = descale_and_clamp(p0 + dc_rounding, dc_shift);
        for (uint i = 0; i < 8; ++i) {
            output[base + i] = pixel;
        }
        return;
    }

    const int z2 = p2;
    const int z3 = p6;
    const int z1 = (z2 + z3) * FIX_0_541196100;
    const int tmp2 = z1 - z3 * FIX_1_847759065;
    const int tmp3 = z1 + z2 * FIX_0_765366865;

    const int tmp0 = (p0 + p4) << CONST_BITS;
    const int tmp1 = (p0 - p4) << CONST_BITS;

    const int tmp10 = tmp0 + tmp3;
    const int tmp13 = tmp0 - tmp3;
    const int tmp11 = tmp1 + tmp2;
    const int tmp12 = tmp1 - tmp2;

    const int z1o = p7 + p1;
    const int z2o = p5 + p3;
    const int z3o = p7 + p3;
    const int z4o = p5 + p1;
    const int z5 = (z3o + z4o) * FIX_1_175875602;

    const int tmp0o = p7 * FIX_0_298631336;
    const int tmp1o = p5 * FIX_2_053119869;
    const int tmp2o = p3 * FIX_3_072711026;
    const int tmp3o = p1 * FIX_1_501321110;
    const int z1m = z1o * -FIX_0_899976223;
    const int z2m = z2o * -FIX_2_562915447;
    const int z3m = z3o * -FIX_1_961570560 + z5;
    const int z4m = z4o * -FIX_0_390180644 + z5;

    const int out0 = tmp0o + z1m + z3m;
    const int out1 = tmp1o + z2m + z4m;
    const int out2 = tmp2o + z2m + z3m;
    const int out3 = tmp3o + z1m + z4m;

    output[base] = descale_and_clamp(tmp10 + out3 + rounding, shift);
    output[base + 7] = descale_and_clamp(tmp10 - out3 + rounding, shift);
    output[base + 1] = descale_and_clamp(tmp11 + out2 + rounding, shift);
    output[base + 6] = descale_and_clamp(tmp11 - out2 + rounding, shift);
    output[base + 2] = descale_and_clamp(tmp12 + out1 + rounding, shift);
    output[base + 5] = descale_and_clamp(tmp12 - out1 + rounding, shift);
    output[base + 3] = descale_and_clamp(tmp13 + out0 + rounding, shift);
    output[base + 4] = descale_and_clamp(tmp13 - out0 + rounding, shift);
}

inline void idct_islow(
    thread const short input[64],
    thread uchar output[64]
) {
    thread int work[64];
    bool upper_half_zero = true;
    for (uint i = 32; i < 64; ++i) {
        if (input[i] != 0) {
            upper_half_zero = false;
            break;
        }
    }
    for (uint col = 0; col < 8; ++col) {
        if (upper_half_zero) {
            idct_1d_column_bottom_half_zero(input, work, col);
        } else {
            idct_1d_column(input, work, col);
        }
    }
    for (uint row = 0; row < 8; ++row) {
        idct_1d_row(work, output, row);
    }
}

inline void idct_islow_dc_only(
    short dc_coeff,
    thread uchar output[64]
) {
    const uchar pixel = clamp_u8(((int(dc_coeff) + 4) >> 3) + 128);
    for (uint i = 0; i < 64; ++i) {
        output[i] = pixel;
    }
}

inline void deposit_block(
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint x,
    uint y,
    thread const uchar block[64]
) {
    if (x >= width || y >= height) {
        return;
    }
    const uint copy_width = min(8u, width - x);
    const uint copy_height = min(8u, height - y);
    if (copy_width == 8u && copy_height == 8u && (stride & 3u) == 0u) {
        for (uint by = 0; by < 8u; ++by) {
            const uint src = by * 8u;
            const uint dst = (y + by) * stride + x;
            *(device uchar4 *)(plane + dst) = uchar4(
                block[src],
                block[src + 1u],
                block[src + 2u],
                block[src + 3u]
            );
            *(device uchar4 *)(plane + dst + 4u) = uchar4(
                block[src + 4u],
                block[src + 5u],
                block[src + 6u],
                block[src + 7u]
            );
        }
        return;
    }
    for (uint by = 0; by < copy_height; ++by) {
        const uint dst = (y + by) * stride + x;
        for (uint bx = 0; bx < copy_width; ++bx) {
            plane[dst + bx] = block[by * 8 + bx];
        }
    }
}

inline void idct_4x4_column(
    thread const short input[64],
    thread int work[32],
    uint col
) {
    const int p0 = int(input[col]);
    const int p1 = int(input[col + 8]);
    const int p2 = int(input[col + 16]);
    const int p3 = int(input[col + 24]);
    const int p5 = int(input[col + 40]);
    const int p6 = int(input[col + 48]);
    const int p7 = int(input[col + 56]);

    if (p1 == 0 && p2 == 0 && p3 == 0 && p5 == 0 && p6 == 0 && p7 == 0) {
        const int dc = p0 << PASS1_BITS;
        work[col] = dc;
        work[8 + col] = dc;
        work[16 + col] = dc;
        work[24 + col] = dc;
        return;
    }

    const int tmp0_base = p0 << (CONST_BITS + 1);
    const int tmp2_even = p2 * FIX_1_847759065 + p6 * -FIX_0_765366865;
    const int tmp10 = tmp0_base + tmp2_even;
    const int tmp12 = tmp0_base - tmp2_even;

    const int tmp0 = p7 * -FIX_0_211164243
        + p5 * FIX_1_451774981
        + p3 * -FIX_2_172734803
        + p1 * FIX_1_061594337;
    const int tmp2 = p7 * -FIX_0_509795579
        + p5 * -FIX_0_601344887
        + p3 * FIX_0_899976223
        + p1 * FIX_2_562915447;

    const int shift = CONST_BITS - PASS1_BITS + 1;
    work[col] = descale(tmp10 + tmp2, shift);
    work[24 + col] = descale(tmp10 - tmp2, shift);
    work[8 + col] = descale(tmp12 + tmp0, shift);
    work[16 + col] = descale(tmp12 - tmp0, shift);
}

inline void idct_4x4_row(
    thread const int work[32],
    thread uchar output[16],
    uint row
) {
    const uint base = row * 8;
    const int p0 = work[base];
    const int p1 = work[base + 1];
    const int p2 = work[base + 2];
    const int p3 = work[base + 3];
    const int p5 = work[base + 5];
    const int p6 = work[base + 6];
    const int p7 = work[base + 7];

    const uint out = row * 4;
    if (p1 == 0 && p2 == 0 && p3 == 0 && p5 == 0 && p6 == 0 && p7 == 0) {
        const uchar dc = descale_and_clamp(p0, PASS1_BITS + 3);
        output[out] = dc;
        output[out + 1] = dc;
        output[out + 2] = dc;
        output[out + 3] = dc;
        return;
    }

    const int tmp0_base = p0 << (CONST_BITS + 1);
    const int tmp2_even = p2 * FIX_1_847759065 + p6 * -FIX_0_765366865;
    const int tmp10 = tmp0_base + tmp2_even;
    const int tmp12 = tmp0_base - tmp2_even;

    const int tmp0 = p7 * -FIX_0_211164243
        + p5 * FIX_1_451774981
        + p3 * -FIX_2_172734803
        + p1 * FIX_1_061594337;
    const int tmp2 = p7 * -FIX_0_509795579
        + p5 * -FIX_0_601344887
        + p3 * FIX_0_899976223
        + p1 * FIX_2_562915447;

    const int shift = CONST_BITS + PASS1_BITS + 3 + 1;
    output[out] = descale_and_clamp(tmp10 + tmp2, shift);
    output[out + 3] = descale_and_clamp(tmp10 - tmp2, shift);
    output[out + 1] = descale_and_clamp(tmp12 + tmp0, shift);
    output[out + 2] = descale_and_clamp(tmp12 - tmp0, shift);
}

inline void idct_islow_4x4(
    thread const short input[64],
    thread uchar output[16]
) {
    thread int work[32];
    for (uint col = 0; col < 8; ++col) {
        if (col == 4) {
            continue;
        }
        idct_4x4_column(input, work, col);
    }
    for (uint row = 0; row < 4; ++row) {
        idct_4x4_row(work, output, row);
    }
}

inline void idct_2x2_column(
    thread const short input[64],
    thread int work[16],
    uint col
) {
    const int p0 = int(input[col]);
    const int p1 = int(input[col + 8]);
    const int p3 = int(input[col + 24]);
    const int p5 = int(input[col + 40]);
    const int p7 = int(input[col + 56]);

    if (p1 == 0 && p3 == 0 && p5 == 0 && p7 == 0) {
        const int dc = p0 << PASS1_BITS;
        work[col] = dc;
        work[8 + col] = dc;
        return;
    }

    const int tmp10 = p0 << (CONST_BITS + 2);
    const int tmp0 = p7 * -FIX_0_720959822
        + p5 * FIX_0_850430095
        + p3 * -FIX_1_272758580
        + p1 * FIX_3_624509785;

    const int shift = CONST_BITS - PASS1_BITS + 2;
    work[col] = descale(tmp10 + tmp0, shift);
    work[8 + col] = descale(tmp10 - tmp0, shift);
}

inline void idct_2x2_row(
    thread const int work[16],
    thread uchar output[4],
    uint row
) {
    const uint base = row * 8;
    const int p0 = work[base];
    const int p1 = work[base + 1];
    const int p3 = work[base + 3];
    const int p5 = work[base + 5];
    const int p7 = work[base + 7];

    if (p1 == 0 && p3 == 0 && p5 == 0 && p7 == 0) {
        const uchar dc = descale_and_clamp(p0, PASS1_BITS + 3);
        const uint out = row * 2;
        output[out] = dc;
        output[out + 1] = dc;
        return;
    }

    const int tmp10 = p0 << (CONST_BITS + 2);
    const int tmp0 = p7 * -FIX_0_720959822
        + p5 * FIX_0_850430095
        + p3 * -FIX_1_272758580
        + p1 * FIX_3_624509785;

    const int shift = CONST_BITS + PASS1_BITS + 5;
    const uint out = row * 2;
    output[out] = descale_and_clamp(tmp10 + tmp0, shift);
    output[out + 1] = descale_and_clamp(tmp10 - tmp0, shift);
}

inline void idct_islow_2x2(
    thread const short input[64],
    thread uchar output[4]
) {
    thread int work[16];
    for (uint col = 0; col < 8; ++col) {
        if (col == 2 || col == 4 || col == 6) {
            continue;
        }
        idct_2x2_column(input, work, col);
    }
    for (uint row = 0; row < 2; ++row) {
        idct_2x2_row(work, output, row);
    }
}

inline uchar idct_islow_1x1(thread const short input[64]) {
    return descale_and_clamp(int(input[0]), 3);
}

inline void deposit_block_region(
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint origin_x,
    uint origin_y,
    uint block_x,
    uint block_y,
    thread const uchar pixels[64]
) {
    const int dst_x = int(block_x) - int(origin_x);
    const int dst_y = int(block_y) - int(origin_y);
    for (uint row = 0; row < 8; ++row) {
        const int out_y = dst_y + int(row);
        if (out_y < 0 || out_y >= int(height)) {
            continue;
        }
        for (uint col = 0; col < 8; ++col) {
            const int out_x = dst_x + int(col);
            if (out_x < 0 || out_x >= int(width)) {
                continue;
            }
            plane[uint(out_y) * stride + uint(out_x)] = pixels[row * 8u + col];
        }
    }
}

inline void deposit_block_4x4(
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint x,
    uint y,
    thread const uchar block[16]
) {
    if (x >= width || y >= height) {
        return;
    }
    const uint copy_width = min(4u, width - x);
    const uint copy_height = min(4u, height - y);
    for (uint by = 0; by < copy_height; ++by) {
        const uint dst = (y + by) * stride + x;
        for (uint bx = 0; bx < copy_width; ++bx) {
            plane[dst + bx] = block[by * 4 + bx];
        }
    }
}

inline void deposit_block_4x4_region(
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint origin_x,
    uint origin_y,
    uint block_x,
    uint block_y,
    thread const uchar pixels[16]
) {
    const int dst_x = int(block_x) - int(origin_x);
    const int dst_y = int(block_y) - int(origin_y);
    for (uint row = 0; row < 4; ++row) {
        const int out_y = dst_y + int(row);
        if (out_y < 0 || out_y >= int(height)) {
            continue;
        }
        for (uint col = 0; col < 4; ++col) {
            const int out_x = dst_x + int(col);
            if (out_x < 0 || out_x >= int(width)) {
                continue;
            }
            plane[uint(out_y) * stride + uint(out_x)] = pixels[row * 4u + col];
        }
    }
}

inline void deposit_block_2x2(
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint x,
    uint y,
    thread const uchar block[4]
) {
    if (x >= width || y >= height) {
        return;
    }
    const uint copy_width = min(2u, width - x);
    const uint copy_height = min(2u, height - y);
    for (uint by = 0; by < copy_height; ++by) {
        const uint dst = (y + by) * stride + x;
        for (uint bx = 0; bx < copy_width; ++bx) {
            plane[dst + bx] = block[by * 2 + bx];
        }
    }
}

inline void deposit_block_2x2_region(
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint origin_x,
    uint origin_y,
    uint block_x,
    uint block_y,
    thread const uchar pixels[4]
) {
    const int dst_x = int(block_x) - int(origin_x);
    const int dst_y = int(block_y) - int(origin_y);
    for (uint row = 0; row < 2; ++row) {
        const int out_y = dst_y + int(row);
        if (out_y < 0 || out_y >= int(height)) {
            continue;
        }
        for (uint col = 0; col < 2; ++col) {
            const int out_x = dst_x + int(col);
            if (out_x < 0 || out_x >= int(width)) {
                continue;
            }
            plane[uint(out_y) * stride + uint(out_x)] = pixels[row * 2u + col];
        }
    }
}

inline void deposit_scaled_block(
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint x,
    uint y,
    uint scale_shift,
    thread const short coeffs[64],
    bool dc_only
) {
    if (scale_shift == 1u) {
        thread uchar pixels4[16];
        if (dc_only) {
            const uchar pixel = idct_islow_1x1(coeffs);
            for (uint i = 0; i < 16; ++i) {
                pixels4[i] = pixel;
            }
        } else {
            idct_islow_4x4(coeffs, pixels4);
        }
        deposit_block_4x4(plane, stride, width, height, x, y, pixels4);
        return;
    }

    if (scale_shift == 2u) {
        thread uchar pixels2[4];
        if (dc_only) {
            const uchar pixel = idct_islow_1x1(coeffs);
            for (uint i = 0; i < 4; ++i) {
                pixels2[i] = pixel;
            }
        } else {
            idct_islow_2x2(coeffs, pixels2);
        }
        deposit_block_2x2(plane, stride, width, height, x, y, pixels2);
        return;
    }

    const uchar pixel = idct_islow_1x1(coeffs);
    if (x < width && y < height) {
        plane[y * stride + x] = pixel;
    }
}

inline void deposit_scaled_block_region(
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint origin_x,
    uint origin_y,
    uint x,
    uint y,
    uint scale_shift,
    thread const short coeffs[64],
    bool dc_only
) {
    if (scale_shift == 1u) {
        thread uchar pixels4[16];
        if (dc_only) {
            const uchar pixel = idct_islow_1x1(coeffs);
            for (uint i = 0; i < 16; ++i) {
                pixels4[i] = pixel;
            }
        } else {
            idct_islow_4x4(coeffs, pixels4);
        }
        deposit_block_4x4_region(plane, stride, width, height, origin_x, origin_y, x, y, pixels4);
    } else if (scale_shift == 2u) {
        thread uchar pixels2[4];
        if (dc_only) {
            const uchar pixel = idct_islow_1x1(coeffs);
            for (uint i = 0; i < 4; ++i) {
                pixels2[i] = pixel;
            }
        } else {
            idct_islow_2x2(coeffs, pixels2);
        }
        deposit_block_2x2_region(plane, stride, width, height, origin_x, origin_y, x, y, pixels2);
    } else {
        const int out_x = int(x) - int(origin_x);
        const int out_y = int(y) - int(origin_y);
        if (out_x >= 0 && out_x < int(width) && out_y >= 0 && out_y < int(height)) {
            const uchar pixel = idct_islow_1x1(coeffs);
            plane[uint(out_y) * stride + uint(out_x)] = pixel;
        }
    }
}

inline uchar h2v2_sample(
    device const uchar *near_row,
    device const uchar *curr_row,
    uint n,
    uint x
) {
    if (n == 0) {
        return 0;
    }
    const uint sample = min(x / 2, n - 1);
    const uint this_sum = 3u * uint(curr_row[sample]) + uint(near_row[sample]);
    if (n == 1) {
        return uchar((4u * this_sum + 8u) >> 4);
    }
    if (x == 0) {
        return uchar((this_sum * 4u + 8u) >> 4);
    }
    if (x == n * 2u - 1u) {
        return uchar((this_sum * 4u + 7u) >> 4);
    }
    if ((x & 1u) == 0u) {
        const uint last_sum = 3u * uint(curr_row[sample - 1]) + uint(near_row[sample - 1]);
        return uchar((this_sum * 3u + last_sum + 8u) >> 4);
    }
    const uint next_sum = 3u * uint(curr_row[sample + 1]) + uint(near_row[sample + 1]);
    return uchar((this_sum * 3u + next_sum + 7u) >> 4);
}

inline uchar h2v1_sample(
    device const uchar *row,
    uint n,
    uint x
) {
    if (n == 0) {
        return 0;
    }
    if (n == 1) {
        return row[0];
    }
    const uint sample = min(x / 2u, n - 1u);
    if (x == 0u) {
        return row[0];
    }
    if (x == n * 2u - 1u) {
        return row[n - 1u];
    }
    if ((x & 1u) == 0u) {
        const uint prev = uint(row[sample - 1u]);
        const uint curr = uint(row[sample]);
        return uchar((3u * curr + prev + 2u) >> 2);
    }
    const uint curr = uint(row[sample]);
    const uint next = uint(row[sample + 1u]);
    return uchar((3u * curr + next + 2u) >> 2);
}

kernel void jpeg_pack(
    device const uchar *plane0 [[buffer(0)]],
    device const uchar *plane1 [[buffer(1)]],
    device const uchar *plane2 [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegPackParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    uint out_idx = gid.y * params.out_stride;

    if (params.out_format == OUT_GRAY) {
        out_idx += gid.x;
        if (params.mode == MODE_GRAY || params.mode == MODE_YCBCR) {
            out[out_idx] = plane0[idx];
            return;
        }

        const uint r = plane0[idx];
        const uint g = plane1[idx];
        const uint b = plane2[idx];
        out[out_idx] = uchar((77u * r + 150u * g + 29u * b + 128u) >> 8);
        return;
    }

    out_idx += gid.x * (params.out_format == OUT_RGB ? 3u : 4u);

    if (params.mode == MODE_GRAY) {
        const uchar gray = plane0[idx];
        out[out_idx] = gray;
        out[out_idx + 1] = gray;
        out[out_idx + 2] = gray;
    } else if (params.mode == MODE_RGB) {
        out[out_idx] = plane0[idx];
        out[out_idx + 1] = plane1[idx];
        out[out_idx + 2] = plane2[idx];
    } else {
        const int y = int(plane0[idx]);
        const int cb = int(plane1[idx]) - 128;
        const int cr = int(plane2[idx]) - 128;
        out[out_idx] = clamp_u8(y + ((91881 * cr + (1 << 15)) >> 16));
        out[out_idx + 1] = clamp_u8(y - ((22554 * cb + 46802 * cr + (1 << 15)) >> 16));
        out[out_idx + 2] = clamp_u8(y + ((116130 * cb + (1 << 15)) >> 16));
    }

    if (params.out_format == OUT_RGBA) {
        out[out_idx + 3] = uchar(params.alpha);
    }
}

kernel void jpeg_pack_444_rgb_batch(
    device const uchar *plane0 [[buffer(0)]],
    device const uchar *plane1 [[buffer(1)]],
    device const uchar *plane2 [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegWindowedPackBatchParams &params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.tile_count) {
        return;
    }

    const uint plane_len = params.src_width * params.src_height;
    const uint plane_base = gid.z * plane_len;
    const uint src_x = gid.x + params.src_x;
    const uint src_y = gid.y + params.src_y;
    if (src_x >= params.src_width || src_y >= params.src_height) {
        return;
    }

    const uint idx = plane_base + src_y * params.src_width + src_x;
    const uint out_base = gid.z * params.out_stride * params.height;
    const uint out_idx = out_base + gid.y * params.out_stride + gid.x * 3u;

    if (params.mode == MODE_GRAY) {
        const uchar gray = plane0[idx];
        out[out_idx] = gray;
        out[out_idx + 1] = gray;
        out[out_idx + 2] = gray;
    } else if (params.mode == MODE_RGB) {
        out[out_idx] = plane0[idx];
        out[out_idx + 1] = plane1[idx];
        out[out_idx + 2] = plane2[idx];
    } else {
        const int y = int(plane0[idx]);
        const int cb = int(plane1[idx]) - 128;
        const int cr = int(plane2[idx]) - 128;
        out[out_idx] = clamp_u8(y + ((91881 * cr + (1 << 15)) >> 16));
        out[out_idx + 1] = clamp_u8(y - ((22554 * cb + 46802 * cr + (1 << 15)) >> 16));
        out[out_idx + 2] = clamp_u8(y + ((116130 * cb + (1 << 15)) >> 16));
    }
}

kernel void jpeg_decode_fast420(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast420Params &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *restart_offsets [[buffer(14)]],
    device JpegDecodeStatus *status [[buffer(15)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(16)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    thread BitReader br;
    uint start_mcu = 0u;
    uint end_mcu = 0u;
    int y_prev_dc = 0;
    int cb_prev_dc = 0;
    int cr_prev_dc = 0;
    if (!configure_entropy_thread(
        gid,
        total_mcus,
        params.restart_interval_mcus,
        params.restart_offset_count,
        params.restart_start_mcu,
        restart_offsets,
        entropy_checkpoints,
        br,
        start_mcu,
        end_mcu,
        y_prev_dc,
        cb_prev_dc,
        cr_prev_dc
    )) {
        return;
    }
    device JpegDecodeStatus *thread_status = status + gid;

    thread_status->code = FAST420_STATUS_OK;
    thread_status->detail = 0;
    thread_status->position = 0;
    thread_status->reserved = 0;

    thread short coeffs[64];
    thread uchar pixels[64];

    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
            const uint my = mcu_index / params.mcus_per_row;
            const uint mx = mcu_index % params.mcus_per_row;
            const uint y_x = mx * 16u;
            const uint y_y = my * 16u;
            const uint c_x = mx * 8u;
            const uint c_y = my * 8u;
            bool dc_only = false;

            if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            if (dc_only) {
                idct_islow_dc_only(coeffs[0], pixels);
            } else {
                idct_islow(coeffs, pixels);
            }
            deposit_block(y_plane, params.width, params.width, params.height, y_x, y_y, pixels);

            if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            if (dc_only) {
                idct_islow_dc_only(coeffs[0], pixels);
            } else {
                idct_islow(coeffs, pixels);
            }
            deposit_block(y_plane, params.width, params.width, params.height, y_x + 8u, y_y, pixels);

            if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            if (dc_only) {
                idct_islow_dc_only(coeffs[0], pixels);
            } else {
                idct_islow(coeffs, pixels);
            }
            deposit_block(y_plane, params.width, params.width, params.height, y_x, y_y + 8u, pixels);

            if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            if (dc_only) {
                idct_islow_dc_only(coeffs[0], pixels);
            } else {
                idct_islow(coeffs, pixels);
            }
            deposit_block(y_plane, params.width, params.width, params.height, y_x + 8u, y_y + 8u, pixels);

            if (!decode_block(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            if (dc_only) {
                idct_islow_dc_only(coeffs[0], pixels);
            } else {
                idct_islow(coeffs, pixels);
            }
            deposit_block(cb_plane, params.chroma_width, params.chroma_width, params.chroma_height, c_x, c_y, pixels);

            if (!decode_block(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            if (dc_only) {
                idct_islow_dc_only(coeffs[0], pixels);
            } else {
                idct_islow(coeffs, pixels);
            }
            deposit_block(cr_plane, params.chroma_width, params.chroma_width, params.chroma_height, c_x, c_y, pixels);
    }
}

kernel void jpeg_decode_fast420_batch(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast420BatchParams &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *entropy_offsets [[buffer(14)]],
    device const uint *entropy_lens [[buffer(15)]],
    device JpegDecodeStatus *status [[buffer(16)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(17)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint tile_index = gid / params.segment_count;
    const uint local_gid = gid - tile_index * params.segment_count;
    if (tile_index >= params.tile_count) {
        return;
    }

    device JpegDecodeStatus *thread_status = status + gid;
    thread_status->code = FAST420_STATUS_OK;
    thread_status->detail = 0;
    thread_status->position = 0;
    thread_status->reserved = 0;

    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    const uint checkpoint_base = tile_index * params.segment_count;
    const JpegEntropyCheckpoint checkpoint = entropy_checkpoints[checkpoint_base + local_gid];
    uint start_mcu = checkpoint.mcu_index;
    if (start_mcu >= total_mcus) {
        return;
    }
    uint end_mcu = total_mcus;
    if (local_gid + 1u < params.segment_count) {
        end_mcu = min(total_mcus, entropy_checkpoints[checkpoint_base + local_gid + 1u].mcu_index);
    }
    if (end_mcu <= start_mcu) {
        return;
    }

    const uint entropy_base = entropy_offsets[tile_index];
    const uint entropy_end = entropy_base + entropy_lens[tile_index];
    thread BitReader br;
    br.pos = entropy_base + checkpoint.entropy_pos;
    br.acc = checkpoint.bit_acc;
    br.bits = checkpoint.bit_count;

    int y_prev_dc = checkpoint.y_prev_dc;
    int cb_prev_dc = checkpoint.cb_prev_dc;
    int cr_prev_dc = checkpoint.cr_prev_dc;

    const uint y_plane_base = tile_index * params.width * params.height;
    const uint chroma_plane_base = tile_index * params.chroma_width * params.chroma_height;
    device uchar *tile_y_plane = y_plane + y_plane_base;
    device uchar *tile_cb_plane = cb_plane + chroma_plane_base;
    device uchar *tile_cr_plane = cr_plane + chroma_plane_base;

    thread short coeffs[64];
    thread uchar pixels[64];

    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint my = mcu_index / params.mcus_per_row;
        const uint mx = mcu_index % params.mcus_per_row;
        const uint y_x = mx * 16u;
        const uint y_y = my * 16u;
        const uint c_x = mx * 8u;
        const uint c_y = my * 8u;
        bool dc_only = false;

        if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        if (dc_only) {
            idct_islow_dc_only(coeffs[0], pixels);
        } else {
            idct_islow(coeffs, pixels);
        }
        deposit_block(tile_y_plane, params.width, params.width, params.height, y_x, y_y, pixels);

        if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        if (dc_only) {
            idct_islow_dc_only(coeffs[0], pixels);
        } else {
            idct_islow(coeffs, pixels);
        }
        deposit_block(tile_y_plane, params.width, params.width, params.height, y_x + 8u, y_y, pixels);

        if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        if (dc_only) {
            idct_islow_dc_only(coeffs[0], pixels);
        } else {
            idct_islow(coeffs, pixels);
        }
        deposit_block(tile_y_plane, params.width, params.width, params.height, y_x, y_y + 8u, pixels);

        if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        if (dc_only) {
            idct_islow_dc_only(coeffs[0], pixels);
        } else {
            idct_islow(coeffs, pixels);
        }
        deposit_block(tile_y_plane, params.width, params.width, params.height, y_x + 8u, y_y + 8u, pixels);

        if (!decode_block(br, entropy, entropy_end, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        if (dc_only) {
            idct_islow_dc_only(coeffs[0], pixels);
        } else {
            idct_islow(coeffs, pixels);
        }
        deposit_block(tile_cb_plane, params.chroma_width, params.chroma_width, params.chroma_height, c_x, c_y, pixels);

        if (!decode_block(br, entropy, entropy_end, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        if (dc_only) {
            idct_islow_dc_only(coeffs[0], pixels);
        } else {
            idct_islow(coeffs, pixels);
        }
        deposit_block(tile_cr_plane, params.chroma_width, params.chroma_width, params.chroma_height, c_x, c_y, pixels);
    }
}

kernel void jpeg_decode_fast422(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast420Params &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *restart_offsets [[buffer(14)]],
    device JpegDecodeStatus *status [[buffer(15)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(16)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    thread BitReader br;
    uint start_mcu = 0u;
    uint end_mcu = 0u;
    int y_prev_dc = 0;
    int cb_prev_dc = 0;
    int cr_prev_dc = 0;
    if (!configure_entropy_thread(
        gid,
        total_mcus,
        params.restart_interval_mcus,
        params.restart_offset_count,
        params.restart_start_mcu,
        restart_offsets,
        entropy_checkpoints,
        br,
        start_mcu,
        end_mcu,
        y_prev_dc,
        cb_prev_dc,
        cr_prev_dc
    )) {
        return;
    }
    device JpegDecodeStatus *thread_status = status + gid;

    thread_status->code = FAST420_STATUS_OK;
    thread_status->detail = 0;
    thread_status->position = 0;
    thread_status->reserved = 0;

    thread short coeffs[64];
    thread uchar pixels[64];

    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint my = mcu_index / params.mcus_per_row;
        const uint mx = mcu_index % params.mcus_per_row;
        const uint y_x = mx * 16u;
        const uint y_y = my * 8u;
        const uint c_x = mx * 8u;
        const uint c_y = my * 8u;
        bool dc_only = false;

        if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        if (dc_only) {
            idct_islow_dc_only(coeffs[0], pixels);
        } else {
            idct_islow(coeffs, pixels);
        }
        deposit_block(y_plane, params.width, params.width, params.height, y_x, y_y, pixels);

        if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        if (dc_only) {
            idct_islow_dc_only(coeffs[0], pixels);
        } else {
            idct_islow(coeffs, pixels);
        }
        deposit_block(y_plane, params.width, params.width, params.height, y_x + 8u, y_y, pixels);

        if (!decode_block(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        if (dc_only) {
            idct_islow_dc_only(coeffs[0], pixels);
        } else {
            idct_islow(coeffs, pixels);
        }
        deposit_block(cb_plane, params.chroma_width, params.chroma_width, params.chroma_height, c_x, c_y, pixels);

        if (!decode_block(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        if (dc_only) {
            idct_islow_dc_only(coeffs[0], pixels);
        } else {
            idct_islow(coeffs, pixels);
        }
        deposit_block(cr_plane, params.chroma_width, params.chroma_width, params.chroma_height, c_x, c_y, pixels);
    }
}

kernel void jpeg_decode_fast422_batch(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast420BatchParams &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *entropy_offsets [[buffer(14)]],
    device const uint *entropy_lens [[buffer(15)]],
    device JpegDecodeStatus *status [[buffer(16)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(17)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint tile_index = gid / params.segment_count;
    const uint local_gid = gid - tile_index * params.segment_count;
    if (tile_index >= params.tile_count) {
        return;
    }

    device JpegDecodeStatus *thread_status = status + gid;
    thread_status->code = FAST420_STATUS_OK;
    thread_status->detail = 0;
    thread_status->position = 0;
    thread_status->reserved = 0;

    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    const uint checkpoint_base = tile_index * params.segment_count;
    const JpegEntropyCheckpoint checkpoint = entropy_checkpoints[checkpoint_base + local_gid];
    uint start_mcu = checkpoint.mcu_index;
    if (start_mcu >= total_mcus) {
        return;
    }
    uint end_mcu = total_mcus;
    if (local_gid + 1u < params.segment_count) {
        end_mcu = min(total_mcus, entropy_checkpoints[checkpoint_base + local_gid + 1u].mcu_index);
    }
    if (end_mcu <= start_mcu) {
        return;
    }

    const uint entropy_base = entropy_offsets[tile_index];
    const uint entropy_end = entropy_base + entropy_lens[tile_index];
    thread BitReader br;
    br.pos = entropy_base + checkpoint.entropy_pos;
    br.acc = checkpoint.bit_acc;
    br.bits = checkpoint.bit_count;

    int y_prev_dc = checkpoint.y_prev_dc;
    int cb_prev_dc = checkpoint.cb_prev_dc;
    int cr_prev_dc = checkpoint.cr_prev_dc;

    const uint y_plane_base = tile_index * params.width * params.height;
    const uint chroma_plane_base = tile_index * params.chroma_width * params.chroma_height;
    device uchar *tile_y_plane = y_plane + y_plane_base;
    device uchar *tile_cb_plane = cb_plane + chroma_plane_base;
    device uchar *tile_cr_plane = cr_plane + chroma_plane_base;

    thread short coeffs[64];
    thread uchar pixels[64];

    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint my = mcu_index / params.mcus_per_row;
        const uint mx = mcu_index % params.mcus_per_row;
        const uint y_x = mx * 16u;
        const uint y_y = my * 8u;
        const uint c_x = mx * 8u;
        const uint c_y = my * 8u;
        bool dc_only = false;

        if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        if (dc_only) {
            idct_islow_dc_only(coeffs[0], pixels);
        } else {
            idct_islow(coeffs, pixels);
        }
        deposit_block(tile_y_plane, params.width, params.width, params.height, y_x, y_y, pixels);

        if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        if (dc_only) {
            idct_islow_dc_only(coeffs[0], pixels);
        } else {
            idct_islow(coeffs, pixels);
        }
        deposit_block(tile_y_plane, params.width, params.width, params.height, y_x + 8u, y_y, pixels);

        if (!decode_block(br, entropy, entropy_end, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        if (dc_only) {
            idct_islow_dc_only(coeffs[0], pixels);
        } else {
            idct_islow(coeffs, pixels);
        }
        deposit_block(tile_cb_plane, params.chroma_width, params.chroma_width, params.chroma_height, c_x, c_y, pixels);

        if (!decode_block(br, entropy, entropy_end, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        if (dc_only) {
            idct_islow_dc_only(coeffs[0], pixels);
        } else {
            idct_islow(coeffs, pixels);
        }
        deposit_block(tile_cr_plane, params.chroma_width, params.chroma_width, params.chroma_height, c_x, c_y, pixels);
    }
}

kernel void jpeg_decode_fast422_region(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast420Params &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *restart_offsets [[buffer(14)]],
    device JpegDecodeStatus *status [[buffer(15)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(16)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    thread BitReader br;
    uint start_mcu = 0u;
    uint end_mcu = 0u;
    int y_prev_dc = 0;
    int cb_prev_dc = 0;
    int cr_prev_dc = 0;
    if (!configure_entropy_thread(
        gid,
        total_mcus,
        params.restart_interval_mcus,
        params.restart_offset_count,
        params.restart_start_mcu,
        restart_offsets,
        entropy_checkpoints,
        br,
        start_mcu,
        end_mcu,
        y_prev_dc,
        cb_prev_dc,
        cr_prev_dc
    )) {
        return;
    }
    device JpegDecodeStatus *thread_status = status + gid;

    thread_status->code = FAST420_STATUS_OK;
    thread_status->detail = 0;
    thread_status->position = 0;
    thread_status->reserved = 0;

    thread short coeffs[64];
    thread uchar pixels[64];

    const uint chroma_origin_x = params.origin_x / 2u;
    const uint chroma_origin_y = params.origin_y;

    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint my = mcu_index / params.mcus_per_row;
        const uint mx = mcu_index % params.mcus_per_row;
        const uint y_x = mx * 16u;
        const uint y_y = my * 8u;
        const uint c_x = mx * 8u;
        const uint c_y = my * 8u;
        const bool mcu_intersects = block_intersects_rect(
            y_x,
            y_y,
            16u,
            8u,
            params.origin_x,
            params.origin_y,
            params.width,
            params.height
        );
        bool dc_only = false;

        if (mcu_intersects) {
            const bool y0_intersects = block_intersects_rect(
                y_x,
                y_y,
                8u,
                8u,
                params.origin_x,
                params.origin_y,
                params.width,
                params.height
            );
            if (y0_intersects) {
                if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                    return;
                }
                if (dc_only) {
                    idct_islow_dc_only(coeffs[0], pixels);
                } else {
                    idct_islow(coeffs, pixels);
                }
                deposit_block_region(
                    y_plane,
                    params.width,
                    params.width,
                    params.height,
                    params.origin_x,
                    params.origin_y,
                    y_x,
                    y_y,
                    pixels
                );
            } else if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                return;
            }

            const bool y1_intersects = block_intersects_rect(
                y_x + 8u,
                y_y,
                8u,
                8u,
                params.origin_x,
                params.origin_y,
                params.width,
                params.height
            );
            if (y1_intersects) {
                if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                    return;
                }
                if (dc_only) {
                    idct_islow_dc_only(coeffs[0], pixels);
                } else {
                    idct_islow(coeffs, pixels);
                }
                deposit_block_region(
                    y_plane,
                    params.width,
                    params.width,
                    params.height,
                    params.origin_x,
                    params.origin_y,
                    y_x + 8u,
                    y_y,
                    pixels
                );
            } else if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                return;
            }

            if (!decode_block(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            if (dc_only) {
                idct_islow_dc_only(coeffs[0], pixels);
            } else {
                idct_islow(coeffs, pixels);
            }
            deposit_block_region(
                cb_plane,
                params.chroma_width,
                params.chroma_width,
                params.chroma_height,
                chroma_origin_x,
                chroma_origin_y,
                c_x,
                c_y,
                pixels
            );

            if (!decode_block(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            if (dc_only) {
                idct_islow_dc_only(coeffs[0], pixels);
            } else {
                idct_islow(coeffs, pixels);
            }
            deposit_block_region(
                cr_plane,
                params.chroma_width,
                params.chroma_width,
                params.chroma_height,
                chroma_origin_x,
                chroma_origin_y,
                c_x,
                c_y,
                pixels
            );
        } else {
            if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                return;
            }
            if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                return;
            }
            if (!decode_block_skip(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_prev_dc, thread_status)) {
                return;
            }
            if (!decode_block_skip(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_prev_dc, thread_status)) {
                return;
            }
        }
    }
}

kernel void jpeg_decode_fast422_scaled(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast420ScaledParams &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *restart_offsets [[buffer(14)]],
    device JpegDecodeStatus *status [[buffer(15)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(16)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    thread BitReader br;
    uint start_mcu = 0u;
    uint end_mcu = 0u;
    int y_prev_dc = 0;
    int cb_prev_dc = 0;
    int cr_prev_dc = 0;
    if (!configure_entropy_thread(
        gid,
        total_mcus,
        params.restart_interval_mcus,
        params.restart_offset_count,
        params.restart_start_mcu,
        restart_offsets,
        entropy_checkpoints,
        br,
        start_mcu,
        end_mcu,
        y_prev_dc,
        cb_prev_dc,
        cr_prev_dc
    )) {
        return;
    }
    device JpegDecodeStatus *thread_status = status + gid;

    thread_status->code = FAST420_STATUS_OK;
    thread_status->detail = 0;
    thread_status->position = 0;
    thread_status->reserved = 0;

    thread short coeffs[64];

    const uint block_size = 8u >> params.scale_shift;
    const uint mcu_width = 16u >> params.scale_shift;
    const uint mcu_height = 8u >> params.scale_shift;

    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint my = mcu_index / params.mcus_per_row;
        const uint mx = mcu_index % params.mcus_per_row;
        const uint y_x = mx * mcu_width;
        const uint y_y = my * mcu_height;
        const uint c_x = mx * block_size;
        const uint c_y = my * block_size;
        bool dc_only = false;

        if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        deposit_scaled_block(
            y_plane,
            params.scaled_width,
            params.scaled_width,
            params.scaled_height,
            y_x,
            y_y,
            params.scale_shift,
            coeffs,
            dc_only
        );

        if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        deposit_scaled_block(
            y_plane,
            params.scaled_width,
            params.scaled_width,
            params.scaled_height,
            y_x + block_size,
            y_y,
            params.scale_shift,
            coeffs,
            dc_only
        );

        if (!decode_block(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        deposit_scaled_block(
            cb_plane,
            params.chroma_width,
            params.chroma_width,
            params.chroma_height,
            c_x,
            c_y,
            params.scale_shift,
            coeffs,
            dc_only
        );

        if (!decode_block(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        deposit_scaled_block(
            cr_plane,
            params.chroma_width,
            params.chroma_width,
            params.chroma_height,
            c_x,
            c_y,
            params.scale_shift,
            coeffs,
            dc_only
        );
    }
}

kernel void jpeg_decode_fast422_scaled_region(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast420ScaledParams &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *restart_offsets [[buffer(14)]],
    device JpegDecodeStatus *status [[buffer(15)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(16)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    thread BitReader br;
    uint start_mcu = 0u;
    uint end_mcu = 0u;
    int y_prev_dc = 0;
    int cb_prev_dc = 0;
    int cr_prev_dc = 0;
    if (!configure_entropy_thread(
        gid,
        total_mcus,
        params.restart_interval_mcus,
        params.restart_offset_count,
        params.restart_start_mcu,
        restart_offsets,
        entropy_checkpoints,
        br,
        start_mcu,
        end_mcu,
        y_prev_dc,
        cb_prev_dc,
        cr_prev_dc
    )) {
        return;
    }
    device JpegDecodeStatus *thread_status = status + gid;

    thread_status->code = FAST420_STATUS_OK;
    thread_status->detail = 0;
    thread_status->position = 0;
    thread_status->reserved = 0;

    thread short coeffs[64];

    const uint block_size = 8u >> params.scale_shift;
    const uint mcu_width = 16u >> params.scale_shift;
    const uint mcu_height = 8u >> params.scale_shift;
    const uint chroma_origin_x = params.origin_x / 2u;
    const uint chroma_origin_y = params.origin_y;

    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint my = mcu_index / params.mcus_per_row;
        const uint mx = mcu_index % params.mcus_per_row;
        const uint y_x = mx * mcu_width;
        const uint y_y = my * mcu_height;
        const uint c_x = mx * block_size;
        const uint c_y = my * block_size;
        const bool mcu_intersects = block_intersects_rect(
            y_x,
            y_y,
            mcu_width,
            mcu_height,
            params.origin_x,
            params.origin_y,
            params.scaled_width,
            params.scaled_height
        );
        bool dc_only = false;

        if (mcu_intersects) {
            const bool y0_intersects = block_intersects_rect(
                y_x,
                y_y,
                block_size,
                block_size,
                params.origin_x,
                params.origin_y,
                params.scaled_width,
                params.scaled_height
            );
            if (y0_intersects) {
                if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                    return;
                }
                deposit_scaled_block_region(
                    y_plane,
                    params.scaled_width,
                    params.scaled_width,
                    params.scaled_height,
                    params.origin_x,
                    params.origin_y,
                    y_x,
                    y_y,
                    params.scale_shift,
                    coeffs,
                    dc_only
                );
            } else if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                return;
            }

            const bool y1_intersects = block_intersects_rect(
                y_x + block_size,
                y_y,
                block_size,
                block_size,
                params.origin_x,
                params.origin_y,
                params.scaled_width,
                params.scaled_height
            );
            if (y1_intersects) {
                if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                    return;
                }
                deposit_scaled_block_region(
                    y_plane,
                    params.scaled_width,
                    params.scaled_width,
                    params.scaled_height,
                    params.origin_x,
                    params.origin_y,
                    y_x + block_size,
                    y_y,
                    params.scale_shift,
                    coeffs,
                    dc_only
                );
            } else if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                return;
            }

            if (!decode_block(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            deposit_scaled_block_region(
                cb_plane,
                params.chroma_width,
                params.chroma_width,
                params.chroma_height,
                chroma_origin_x,
                chroma_origin_y,
                c_x,
                c_y,
                params.scale_shift,
                coeffs,
                dc_only
            );

            if (!decode_block(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            deposit_scaled_block_region(
                cr_plane,
                params.chroma_width,
                params.chroma_width,
                params.chroma_height,
                chroma_origin_x,
                chroma_origin_y,
                c_x,
                c_y,
                params.scale_shift,
                coeffs,
                dc_only
            );
        } else {
            if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                return;
            }
            if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                return;
            }
            if (!decode_block_skip(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_prev_dc, thread_status)) {
                return;
            }
            if (!decode_block_skip(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_prev_dc, thread_status)) {
                return;
            }
        }
    }
}

kernel void jpeg_decode_fast422_scaled_region_batch(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFastRegionScaledBatchParams &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *entropy_offsets [[buffer(14)]],
    device const uint *entropy_lens [[buffer(15)]],
    device JpegDecodeStatus *status [[buffer(16)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(17)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    thread BitReader br;
    uint tile_index = 0u;
    uint start_mcu = 0u;
    uint end_mcu = 0u;
    uint entropy_end = 0u;
    int y_prev_dc = 0;
    int cb_prev_dc = 0;
    int cr_prev_dc = 0;
    if (!configure_batch_entropy_thread(
        gid,
        total_mcus,
        params.segment_count,
        params.tile_count,
        entropy_offsets,
        entropy_lens,
        entropy_checkpoints,
        br,
        tile_index,
        start_mcu,
        end_mcu,
        entropy_end,
        y_prev_dc,
        cb_prev_dc,
        cr_prev_dc
    )) {
        return;
    }
    device JpegDecodeStatus *thread_status = status + gid;

    thread_status->code = FAST420_STATUS_OK;
    thread_status->detail = 0;
    thread_status->position = 0;
    thread_status->reserved = 0;

    const uint y_plane_base = tile_index * params.scaled_width * params.scaled_height;
    const uint chroma_plane_base = tile_index * params.chroma_width * params.chroma_height;
    device uchar *tile_y_plane = y_plane + y_plane_base;
    device uchar *tile_cb_plane = cb_plane + chroma_plane_base;
    device uchar *tile_cr_plane = cr_plane + chroma_plane_base;

    thread short coeffs[64];

    const uint block_size = 8u >> params.scale_shift;
    const uint mcu_width = 16u >> params.scale_shift;
    const uint mcu_height = 8u >> params.scale_shift;
    const uint chroma_origin_x = params.origin_x / 2u;
    const uint chroma_origin_y = params.origin_y;

    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint my = mcu_index / params.mcus_per_row;
        const uint mx = mcu_index % params.mcus_per_row;
        const uint y_x = mx * mcu_width;
        const uint y_y = my * mcu_height;
        const uint c_x = mx * block_size;
        const uint c_y = my * block_size;
        const bool mcu_intersects = block_intersects_rect(
            y_x,
            y_y,
            mcu_width,
            mcu_height,
            params.origin_x,
            params.origin_y,
            params.scaled_width,
            params.scaled_height
        );
        bool dc_only = false;

        if (mcu_intersects) {
            const bool y0_intersects = block_intersects_rect(
                y_x,
                y_y,
                block_size,
                block_size,
                params.origin_x,
                params.origin_y,
                params.scaled_width,
                params.scaled_height
            );
            if (y0_intersects) {
                if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                    return;
                }
                deposit_scaled_block_region(
                    tile_y_plane,
                    params.scaled_width,
                    params.scaled_width,
                    params.scaled_height,
                    params.origin_x,
                    params.origin_y,
                    y_x,
                    y_y,
                    params.scale_shift,
                    coeffs,
                    dc_only
                );
            } else if (!decode_block_skip(br, entropy, entropy_end, y_dc, y_ac, y_prev_dc, thread_status)) {
                return;
            }

            const bool y1_intersects = block_intersects_rect(
                y_x + block_size,
                y_y,
                block_size,
                block_size,
                params.origin_x,
                params.origin_y,
                params.scaled_width,
                params.scaled_height
            );
            if (y1_intersects) {
                if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                    return;
                }
                deposit_scaled_block_region(
                    tile_y_plane,
                    params.scaled_width,
                    params.scaled_width,
                    params.scaled_height,
                    params.origin_x,
                    params.origin_y,
                    y_x + block_size,
                    y_y,
                    params.scale_shift,
                    coeffs,
                    dc_only
                );
            } else if (!decode_block_skip(br, entropy, entropy_end, y_dc, y_ac, y_prev_dc, thread_status)) {
                return;
            }

            if (!decode_block(br, entropy, entropy_end, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            deposit_scaled_block_region(
                tile_cb_plane,
                params.chroma_width,
                params.chroma_width,
                params.chroma_height,
                chroma_origin_x,
                chroma_origin_y,
                c_x,
                c_y,
                params.scale_shift,
                coeffs,
                dc_only
            );

            if (!decode_block(br, entropy, entropy_end, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            deposit_scaled_block_region(
                tile_cr_plane,
                params.chroma_width,
                params.chroma_width,
                params.chroma_height,
                chroma_origin_x,
                chroma_origin_y,
                c_x,
                c_y,
                params.scale_shift,
                coeffs,
                dc_only
            );
        } else {
            if (!decode_block_skip(br, entropy, entropy_end, y_dc, y_ac, y_prev_dc, thread_status)) {
                return;
            }
            if (!decode_block_skip(br, entropy, entropy_end, y_dc, y_ac, y_prev_dc, thread_status)) {
                return;
            }
            if (!decode_block_skip(br, entropy, entropy_end, cb_dc, cb_ac, cb_prev_dc, thread_status)) {
                return;
            }
            if (!decode_block_skip(br, entropy, entropy_end, cr_dc, cr_ac, cr_prev_dc, thread_status)) {
                return;
            }
        }
    }
}

kernel void jpeg_decode_fast420_region(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast420Params &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *restart_offsets [[buffer(14)]],
    device JpegDecodeStatus *status [[buffer(15)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(16)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    thread BitReader br;
    uint start_mcu = 0u;
    uint end_mcu = 0u;
    int y_prev_dc = 0;
    int cb_prev_dc = 0;
    int cr_prev_dc = 0;
    if (!configure_entropy_thread(
        gid,
        total_mcus,
        params.restart_interval_mcus,
        params.restart_offset_count,
        params.restart_start_mcu,
        restart_offsets,
        entropy_checkpoints,
        br,
        start_mcu,
        end_mcu,
        y_prev_dc,
        cb_prev_dc,
        cr_prev_dc
    )) {
        return;
    }
    device JpegDecodeStatus *thread_status = status + gid;

    thread_status->code = FAST420_STATUS_OK;
    thread_status->detail = 0;
    thread_status->position = 0;
    thread_status->reserved = 0;

    thread short coeffs[64];
    thread uchar pixels[64];

    const uint chroma_origin_x = params.origin_x / 2u;
    const uint chroma_origin_y = params.origin_y / 2u;

    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
            const uint my = mcu_index / params.mcus_per_row;
            const uint mx = mcu_index % params.mcus_per_row;
            const uint y_x = mx * 16u;
            const uint y_y = my * 16u;
            const uint c_x = mx * 8u;
            const uint c_y = my * 8u;
            const bool mcu_intersects = block_intersects_rect(
                y_x,
                y_y,
                16u,
                16u,
                params.origin_x,
                params.origin_y,
                params.width,
                params.height
            );
            bool dc_only = false;

            if (mcu_intersects) {
                const bool y0_intersects = block_intersects_rect(
                    y_x,
                    y_y,
                    8u,
                    8u,
                    params.origin_x,
                    params.origin_y,
                    params.width,
                    params.height
                );
                if (y0_intersects) {
                    if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                        return;
                    }
                    if (dc_only) {
                        idct_islow_dc_only(coeffs[0], pixels);
                    } else {
                        idct_islow(coeffs, pixels);
                    }
                    deposit_block_region(
                        y_plane,
                        params.width,
                        params.width,
                        params.height,
                        params.origin_x,
                        params.origin_y,
                        y_x,
                        y_y,
                        pixels
                    );
                } else if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }

                const bool y1_intersects = block_intersects_rect(
                    y_x + 8u,
                    y_y,
                    8u,
                    8u,
                    params.origin_x,
                    params.origin_y,
                    params.width,
                    params.height
                );
                if (y1_intersects) {
                    if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                        return;
                    }
                    if (dc_only) {
                        idct_islow_dc_only(coeffs[0], pixels);
                    } else {
                        idct_islow(coeffs, pixels);
                    }
                    deposit_block_region(
                        y_plane,
                        params.width,
                        params.width,
                        params.height,
                        params.origin_x,
                        params.origin_y,
                        y_x + 8u,
                        y_y,
                        pixels
                    );
                } else if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }

                const bool y2_intersects = block_intersects_rect(
                    y_x,
                    y_y + 8u,
                    8u,
                    8u,
                    params.origin_x,
                    params.origin_y,
                    params.width,
                    params.height
                );
                if (y2_intersects) {
                    if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                        return;
                    }
                    if (dc_only) {
                        idct_islow_dc_only(coeffs[0], pixels);
                    } else {
                        idct_islow(coeffs, pixels);
                    }
                    deposit_block_region(
                        y_plane,
                        params.width,
                        params.width,
                        params.height,
                        params.origin_x,
                        params.origin_y,
                        y_x,
                        y_y + 8u,
                        pixels
                    );
                } else if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }

                const bool y3_intersects = block_intersects_rect(
                    y_x + 8u,
                    y_y + 8u,
                    8u,
                    8u,
                    params.origin_x,
                    params.origin_y,
                    params.width,
                    params.height
                );
                if (y3_intersects) {
                    if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                        return;
                    }
                    if (dc_only) {
                        idct_islow_dc_only(coeffs[0], pixels);
                    } else {
                        idct_islow(coeffs, pixels);
                    }
                    deposit_block_region(
                        y_plane,
                        params.width,
                        params.width,
                        params.height,
                        params.origin_x,
                        params.origin_y,
                        y_x + 8u,
                        y_y + 8u,
                        pixels
                    );
                } else if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }

                if (!decode_block(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
                    return;
                }
                if (dc_only) {
                    idct_islow_dc_only(coeffs[0], pixels);
                } else {
                    idct_islow(coeffs, pixels);
                }
                deposit_block_region(
                    cb_plane,
                    params.chroma_width,
                    params.chroma_width,
                    params.chroma_height,
                    chroma_origin_x,
                    chroma_origin_y,
                    c_x,
                    c_y,
                    pixels
                );

                if (!decode_block(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
                    return;
                }
                if (dc_only) {
                    idct_islow_dc_only(coeffs[0], pixels);
                } else {
                    idct_islow(coeffs, pixels);
                }
                deposit_block_region(
                    cr_plane,
                    params.chroma_width,
                    params.chroma_width,
                    params.chroma_height,
                    chroma_origin_x,
                    chroma_origin_y,
                    c_x,
                    c_y,
                    pixels
                );
            } else {
                if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }
                if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }
                if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }
                if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }
                if (!decode_block_skip(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_prev_dc, thread_status)) {
                    return;
                }
                if (!decode_block_skip(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_prev_dc, thread_status)) {
                    return;
                }
            }
    }
}

kernel void jpeg_decode_fast420_scaled(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast420ScaledParams &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *restart_offsets [[buffer(14)]],
    device JpegDecodeStatus *status [[buffer(15)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(16)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    thread BitReader br;
    uint start_mcu = 0u;
    uint end_mcu = 0u;
    int y_prev_dc = 0;
    int cb_prev_dc = 0;
    int cr_prev_dc = 0;
    if (!configure_entropy_thread(
        gid,
        total_mcus,
        params.restart_interval_mcus,
        params.restart_offset_count,
        params.restart_start_mcu,
        restart_offsets,
        entropy_checkpoints,
        br,
        start_mcu,
        end_mcu,
        y_prev_dc,
        cb_prev_dc,
        cr_prev_dc
    )) {
        return;
    }
    device JpegDecodeStatus *thread_status = status + gid;

    thread_status->code = FAST420_STATUS_OK;
    thread_status->detail = 0;
    thread_status->position = 0;
    thread_status->reserved = 0;

    thread short coeffs[64];

    const uint y_block_size = 8u >> params.scale_shift;
    const uint c_block_size = 8u >> params.scale_shift;
    const uint y_mcu_size = 16u >> params.scale_shift;

    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
            const uint my = mcu_index / params.mcus_per_row;
            const uint mx = mcu_index % params.mcus_per_row;
            const uint y_x = mx * y_mcu_size;
            const uint y_y = my * y_mcu_size;
            const uint c_x = mx * c_block_size;
            const uint c_y = my * c_block_size;
            bool dc_only = false;

            if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            deposit_scaled_block(
                y_plane,
                params.scaled_width,
                params.scaled_width,
                params.scaled_height,
                y_x,
                y_y,
                params.scale_shift,
                coeffs,
                dc_only
            );

            if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            deposit_scaled_block(
                y_plane,
                params.scaled_width,
                params.scaled_width,
                params.scaled_height,
                y_x + y_block_size,
                y_y,
                params.scale_shift,
                coeffs,
                dc_only
            );

            if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            deposit_scaled_block(
                y_plane,
                params.scaled_width,
                params.scaled_width,
                params.scaled_height,
                y_x,
                y_y + y_block_size,
                params.scale_shift,
                coeffs,
                dc_only
            );

            if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            deposit_scaled_block(
                y_plane,
                params.scaled_width,
                params.scaled_width,
                params.scaled_height,
                y_x + y_block_size,
                y_y + y_block_size,
                params.scale_shift,
                coeffs,
                dc_only
            );

            if (!decode_block(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            deposit_scaled_block(
                cb_plane,
                params.chroma_width,
                params.chroma_width,
                params.chroma_height,
                c_x,
                c_y,
                params.scale_shift,
                coeffs,
                dc_only
            );

            if (!decode_block(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            deposit_scaled_block(
                cr_plane,
                params.chroma_width,
                params.chroma_width,
                params.chroma_height,
                c_x,
                c_y,
                params.scale_shift,
                coeffs,
                dc_only
            );
    }
}

kernel void jpeg_decode_fast420_scaled_region(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast420ScaledParams &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *restart_offsets [[buffer(14)]],
    device JpegDecodeStatus *status [[buffer(15)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(16)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    thread BitReader br;
    uint start_mcu = 0u;
    uint end_mcu = 0u;
    int y_prev_dc = 0;
    int cb_prev_dc = 0;
    int cr_prev_dc = 0;
    if (!configure_entropy_thread(
        gid,
        total_mcus,
        params.restart_interval_mcus,
        params.restart_offset_count,
        params.restart_start_mcu,
        restart_offsets,
        entropy_checkpoints,
        br,
        start_mcu,
        end_mcu,
        y_prev_dc,
        cb_prev_dc,
        cr_prev_dc
    )) {
        return;
    }
    device JpegDecodeStatus *thread_status = status + gid;

    thread_status->code = FAST420_STATUS_OK;
    thread_status->detail = 0;
    thread_status->position = 0;
    thread_status->reserved = 0;

    thread short coeffs[64];

    const uint y_block_size = 8u >> params.scale_shift;
    const uint c_block_size = 8u >> params.scale_shift;
    const uint y_mcu_size = 16u >> params.scale_shift;
    const uint chroma_origin_x = params.origin_x / 2u;
    const uint chroma_origin_y = params.origin_y / 2u;

    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
            const uint my = mcu_index / params.mcus_per_row;
            const uint mx = mcu_index % params.mcus_per_row;
            const uint y_x = mx * y_mcu_size;
            const uint y_y = my * y_mcu_size;
            const uint c_x = mx * c_block_size;
            const uint c_y = my * c_block_size;
            const bool mcu_intersects = block_intersects_rect(
                y_x,
                y_y,
                y_mcu_size,
                y_mcu_size,
                params.origin_x,
                params.origin_y,
                params.scaled_width,
                params.scaled_height
            );
            bool dc_only = false;

            if (mcu_intersects) {
                const bool y0_intersects = block_intersects_rect(
                    y_x,
                    y_y,
                    y_block_size,
                    y_block_size,
                    params.origin_x,
                    params.origin_y,
                    params.scaled_width,
                    params.scaled_height
                );
                if (y0_intersects) {
                    if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                        return;
                    }
                    deposit_scaled_block_region(
                        y_plane,
                        params.scaled_width,
                        params.scaled_width,
                        params.scaled_height,
                        params.origin_x,
                        params.origin_y,
                        y_x,
                        y_y,
                        params.scale_shift,
                        coeffs,
                        dc_only
                    );
                } else if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }

                const bool y1_intersects = block_intersects_rect(
                    y_x + y_block_size,
                    y_y,
                    y_block_size,
                    y_block_size,
                    params.origin_x,
                    params.origin_y,
                    params.scaled_width,
                    params.scaled_height
                );
                if (y1_intersects) {
                    if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                        return;
                    }
                    deposit_scaled_block_region(
                        y_plane,
                        params.scaled_width,
                        params.scaled_width,
                        params.scaled_height,
                        params.origin_x,
                        params.origin_y,
                        y_x + y_block_size,
                        y_y,
                        params.scale_shift,
                        coeffs,
                        dc_only
                    );
                } else if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }

                const bool y2_intersects = block_intersects_rect(
                    y_x,
                    y_y + y_block_size,
                    y_block_size,
                    y_block_size,
                    params.origin_x,
                    params.origin_y,
                    params.scaled_width,
                    params.scaled_height
                );
                if (y2_intersects) {
                    if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                        return;
                    }
                    deposit_scaled_block_region(
                        y_plane,
                        params.scaled_width,
                        params.scaled_width,
                        params.scaled_height,
                        params.origin_x,
                        params.origin_y,
                        y_x,
                        y_y + y_block_size,
                        params.scale_shift,
                        coeffs,
                        dc_only
                    );
                } else if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }

                const bool y3_intersects = block_intersects_rect(
                    y_x + y_block_size,
                    y_y + y_block_size,
                    y_block_size,
                    y_block_size,
                    params.origin_x,
                    params.origin_y,
                    params.scaled_width,
                    params.scaled_height
                );
                if (y3_intersects) {
                    if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                        return;
                    }
                    deposit_scaled_block_region(
                        y_plane,
                        params.scaled_width,
                        params.scaled_width,
                        params.scaled_height,
                        params.origin_x,
                        params.origin_y,
                        y_x + y_block_size,
                        y_y + y_block_size,
                        params.scale_shift,
                        coeffs,
                        dc_only
                    );
                } else if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }

                if (!decode_block(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
                    return;
                }
                deposit_scaled_block_region(
                    cb_plane,
                    params.chroma_width,
                    params.chroma_width,
                    params.chroma_height,
                    chroma_origin_x,
                    chroma_origin_y,
                    c_x,
                    c_y,
                    params.scale_shift,
                    coeffs,
                    dc_only
                );

                if (!decode_block(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
                    return;
                }
                deposit_scaled_block_region(
                    cr_plane,
                    params.chroma_width,
                    params.chroma_width,
                    params.chroma_height,
                    chroma_origin_x,
                    chroma_origin_y,
                    c_x,
                    c_y,
                    params.scale_shift,
                    coeffs,
                    dc_only
                );
            } else {
                if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }
                if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }
                if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }
                if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }
                if (!decode_block_skip(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_prev_dc, thread_status)) {
                    return;
                }
                if (!decode_block_skip(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_prev_dc, thread_status)) {
                    return;
                }
            }
    }
}

kernel void jpeg_decode_fast420_scaled_region_batch(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFastRegionScaledBatchParams &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *entropy_offsets [[buffer(14)]],
    device const uint *entropy_lens [[buffer(15)]],
    device JpegDecodeStatus *status [[buffer(16)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(17)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    thread BitReader br;
    uint tile_index = 0u;
    uint start_mcu = 0u;
    uint end_mcu = 0u;
    uint entropy_end = 0u;
    int y_prev_dc = 0;
    int cb_prev_dc = 0;
    int cr_prev_dc = 0;
    if (!configure_batch_entropy_thread(
        gid,
        total_mcus,
        params.segment_count,
        params.tile_count,
        entropy_offsets,
        entropy_lens,
        entropy_checkpoints,
        br,
        tile_index,
        start_mcu,
        end_mcu,
        entropy_end,
        y_prev_dc,
        cb_prev_dc,
        cr_prev_dc
    )) {
        return;
    }
    device JpegDecodeStatus *thread_status = status + gid;

    thread_status->code = FAST420_STATUS_OK;
    thread_status->detail = 0;
    thread_status->position = 0;
    thread_status->reserved = 0;

    const uint y_plane_base = tile_index * params.scaled_width * params.scaled_height;
    const uint chroma_plane_base = tile_index * params.chroma_width * params.chroma_height;
    device uchar *tile_y_plane = y_plane + y_plane_base;
    device uchar *tile_cb_plane = cb_plane + chroma_plane_base;
    device uchar *tile_cr_plane = cr_plane + chroma_plane_base;

    thread short coeffs[64];

    const uint y_block_size = 8u >> params.scale_shift;
    const uint c_block_size = 8u >> params.scale_shift;
    const uint y_mcu_size = 16u >> params.scale_shift;
    const uint chroma_origin_x = params.origin_x / 2u;
    const uint chroma_origin_y = params.origin_y / 2u;

    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
            const uint my = mcu_index / params.mcus_per_row;
            const uint mx = mcu_index % params.mcus_per_row;
            const uint y_x = mx * y_mcu_size;
            const uint y_y = my * y_mcu_size;
            const uint c_x = mx * c_block_size;
            const uint c_y = my * c_block_size;
            const bool mcu_intersects = block_intersects_rect(
                y_x,
                y_y,
                y_mcu_size,
                y_mcu_size,
                params.origin_x,
                params.origin_y,
                params.scaled_width,
                params.scaled_height
            );
            bool dc_only = false;

            if (mcu_intersects) {
                const bool y0_intersects = block_intersects_rect(
                    y_x,
                    y_y,
                    y_block_size,
                    y_block_size,
                    params.origin_x,
                    params.origin_y,
                    params.scaled_width,
                    params.scaled_height
                );
                if (y0_intersects) {
                    if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                        return;
                    }
                    deposit_scaled_block_region(
                        tile_y_plane,
                        params.scaled_width,
                        params.scaled_width,
                        params.scaled_height,
                        params.origin_x,
                        params.origin_y,
                        y_x,
                        y_y,
                        params.scale_shift,
                        coeffs,
                        dc_only
                    );
                } else if (!decode_block_skip(br, entropy, entropy_end, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }

                const bool y1_intersects = block_intersects_rect(
                    y_x + y_block_size,
                    y_y,
                    y_block_size,
                    y_block_size,
                    params.origin_x,
                    params.origin_y,
                    params.scaled_width,
                    params.scaled_height
                );
                if (y1_intersects) {
                    if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                        return;
                    }
                    deposit_scaled_block_region(
                        tile_y_plane,
                        params.scaled_width,
                        params.scaled_width,
                        params.scaled_height,
                        params.origin_x,
                        params.origin_y,
                        y_x + y_block_size,
                        y_y,
                        params.scale_shift,
                        coeffs,
                        dc_only
                    );
                } else if (!decode_block_skip(br, entropy, entropy_end, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }

                const bool y2_intersects = block_intersects_rect(
                    y_x,
                    y_y + y_block_size,
                    y_block_size,
                    y_block_size,
                    params.origin_x,
                    params.origin_y,
                    params.scaled_width,
                    params.scaled_height
                );
                if (y2_intersects) {
                    if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                        return;
                    }
                    deposit_scaled_block_region(
                        tile_y_plane,
                        params.scaled_width,
                        params.scaled_width,
                        params.scaled_height,
                        params.origin_x,
                        params.origin_y,
                        y_x,
                        y_y + y_block_size,
                        params.scale_shift,
                        coeffs,
                        dc_only
                    );
                } else if (!decode_block_skip(br, entropy, entropy_end, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }

                const bool y3_intersects = block_intersects_rect(
                    y_x + y_block_size,
                    y_y + y_block_size,
                    y_block_size,
                    y_block_size,
                    params.origin_x,
                    params.origin_y,
                    params.scaled_width,
                    params.scaled_height
                );
                if (y3_intersects) {
                    if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                        return;
                    }
                    deposit_scaled_block_region(
                        tile_y_plane,
                        params.scaled_width,
                        params.scaled_width,
                        params.scaled_height,
                        params.origin_x,
                        params.origin_y,
                        y_x + y_block_size,
                        y_y + y_block_size,
                        params.scale_shift,
                        coeffs,
                        dc_only
                    );
                } else if (!decode_block_skip(br, entropy, entropy_end, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }

                if (!decode_block(br, entropy, entropy_end, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
                    return;
                }
                deposit_scaled_block_region(
                    tile_cb_plane,
                    params.chroma_width,
                    params.chroma_width,
                    params.chroma_height,
                    chroma_origin_x,
                    chroma_origin_y,
                    c_x,
                    c_y,
                    params.scale_shift,
                    coeffs,
                    dc_only
                );

                if (!decode_block(br, entropy, entropy_end, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
                    return;
                }
                deposit_scaled_block_region(
                    tile_cr_plane,
                    params.chroma_width,
                    params.chroma_width,
                    params.chroma_height,
                    chroma_origin_x,
                    chroma_origin_y,
                    c_x,
                    c_y,
                    params.scale_shift,
                    coeffs,
                    dc_only
                );
            } else {
                if (!decode_block_skip(br, entropy, entropy_end, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }
                if (!decode_block_skip(br, entropy, entropy_end, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }
                if (!decode_block_skip(br, entropy, entropy_end, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }
                if (!decode_block_skip(br, entropy, entropy_end, y_dc, y_ac, y_prev_dc, thread_status)) {
                    return;
                }
                if (!decode_block_skip(br, entropy, entropy_end, cb_dc, cb_ac, cb_prev_dc, thread_status)) {
                    return;
                }
                if (!decode_block_skip(br, entropy, entropy_end, cr_dc, cr_ac, cr_prev_dc, thread_status)) {
                    return;
                }
            }
    }
}

kernel void jpeg_decode_fast444(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast444Params &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *restart_offsets [[buffer(14)]],
    device JpegDecodeStatus *status [[buffer(15)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(16)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    thread BitReader br;
    uint start_mcu = 0u;
    uint end_mcu = 0u;
    int y_prev_dc = 0;
    int cb_prev_dc = 0;
    int cr_prev_dc = 0;
    if (!configure_entropy_thread(
        gid,
        total_mcus,
        params.restart_interval_mcus,
        params.restart_offset_count,
        params.restart_start_mcu,
        restart_offsets,
        entropy_checkpoints,
        br,
        start_mcu,
        end_mcu,
        y_prev_dc,
        cb_prev_dc,
        cr_prev_dc
    )) {
        return;
    }
    device JpegDecodeStatus *thread_status = status + gid;
    thread_status->code = FAST420_STATUS_OK;
    thread_status->detail = 0;
    thread_status->position = 0;
    thread_status->reserved = 0;

    thread short coeffs[64];
    thread uchar pixels[64];
    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint my = mcu_index / params.mcus_per_row;
        const uint mx = mcu_index % params.mcus_per_row;
        const uint block_x = mx * 8u;
        const uint block_y = my * 8u;
        bool dc_only = false;

        if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        if (dc_only) {
            idct_islow_dc_only(coeffs[0], pixels);
        } else {
            idct_islow(coeffs, pixels);
        }
        deposit_block(y_plane, params.width, params.width, params.height, block_x, block_y, pixels);

        if (!decode_block(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        if (dc_only) {
            idct_islow_dc_only(coeffs[0], pixels);
        } else {
            idct_islow(coeffs, pixels);
        }
        deposit_block(cb_plane, params.width, params.width, params.height, block_x, block_y, pixels);

        if (!decode_block(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        if (dc_only) {
            idct_islow_dc_only(coeffs[0], pixels);
        } else {
            idct_islow(coeffs, pixels);
        }
        deposit_block(cr_plane, params.width, params.width, params.height, block_x, block_y, pixels);
    }
}

kernel void jpeg_decode_fast444_region(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast444Params &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *restart_offsets [[buffer(14)]],
    device JpegDecodeStatus *status [[buffer(15)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(16)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    thread BitReader br;
    uint start_mcu = 0u;
    uint end_mcu = 0u;
    int y_prev_dc = 0;
    int cb_prev_dc = 0;
    int cr_prev_dc = 0;
    if (!configure_entropy_thread(
        gid,
        total_mcus,
        params.restart_interval_mcus,
        params.restart_offset_count,
        params.restart_start_mcu,
        restart_offsets,
        entropy_checkpoints,
        br,
        start_mcu,
        end_mcu,
        y_prev_dc,
        cb_prev_dc,
        cr_prev_dc
    )) {
        return;
    }
    device JpegDecodeStatus *thread_status = status + gid;
    thread_status->code = FAST420_STATUS_OK;
    thread_status->detail = 0;
    thread_status->position = 0;
    thread_status->reserved = 0;

    thread short coeffs[64];
    thread uchar pixels[64];
    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint my = mcu_index / params.mcus_per_row;
        const uint mx = mcu_index % params.mcus_per_row;
        const uint block_x = mx * 8u;
        const uint block_y = my * 8u;
        const bool intersects = block_intersects_rect(
            block_x,
            block_y,
            8u,
            8u,
            params.origin_x,
            params.origin_y,
            params.width,
            params.height
        );
        bool dc_only = false;

        if (intersects) {
            if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            if (dc_only) {
                idct_islow_dc_only(coeffs[0], pixels);
            } else {
                idct_islow(coeffs, pixels);
            }
            deposit_block_region(
                y_plane,
                params.width,
                params.width,
                params.height,
                params.origin_x,
                params.origin_y,
                block_x,
                block_y,
                pixels
            );

            if (!decode_block(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            if (dc_only) {
                idct_islow_dc_only(coeffs[0], pixels);
            } else {
                idct_islow(coeffs, pixels);
            }
            deposit_block_region(
                cb_plane,
                params.width,
                params.width,
                params.height,
                params.origin_x,
                params.origin_y,
                block_x,
                block_y,
                pixels
            );

            if (!decode_block(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            if (dc_only) {
                idct_islow_dc_only(coeffs[0], pixels);
            } else {
                idct_islow(coeffs, pixels);
            }
            deposit_block_region(
                cr_plane,
                params.width,
                params.width,
                params.height,
                params.origin_x,
                params.origin_y,
                block_x,
                block_y,
                pixels
            );
        } else {
            if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                return;
            }
            if (!decode_block_skip(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_prev_dc, thread_status)) {
                return;
            }
            if (!decode_block_skip(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_prev_dc, thread_status)) {
                return;
            }
        }
    }
}

kernel void jpeg_decode_fast444_scaled(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast444ScaledParams &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *restart_offsets [[buffer(14)]],
    device JpegDecodeStatus *status [[buffer(15)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(16)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    thread BitReader br;
    uint start_mcu = 0u;
    uint end_mcu = 0u;
    int y_prev_dc = 0;
    int cb_prev_dc = 0;
    int cr_prev_dc = 0;
    if (!configure_entropy_thread(
        gid,
        total_mcus,
        params.restart_interval_mcus,
        params.restart_offset_count,
        params.restart_start_mcu,
        restart_offsets,
        entropy_checkpoints,
        br,
        start_mcu,
        end_mcu,
        y_prev_dc,
        cb_prev_dc,
        cr_prev_dc
    )) {
        return;
    }
    device JpegDecodeStatus *thread_status = status + gid;
    thread_status->code = FAST420_STATUS_OK;
    thread_status->detail = 0;
    thread_status->position = 0;
    thread_status->reserved = 0;

    thread short coeffs[64];
    const uint block_size = 8u >> params.scale_shift;

    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint my = mcu_index / params.mcus_per_row;
        const uint mx = mcu_index % params.mcus_per_row;
        const uint block_x = mx * block_size;
        const uint block_y = my * block_size;
        bool dc_only = false;

        if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        deposit_scaled_block(
            y_plane,
            params.scaled_width,
            params.scaled_width,
            params.scaled_height,
            block_x,
            block_y,
            params.scale_shift,
            coeffs,
            dc_only
        );

        if (!decode_block(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        deposit_scaled_block(
            cb_plane,
            params.scaled_width,
            params.scaled_width,
            params.scaled_height,
            block_x,
            block_y,
            params.scale_shift,
            coeffs,
            dc_only
        );

        if (!decode_block(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        deposit_scaled_block(
            cr_plane,
            params.scaled_width,
            params.scaled_width,
            params.scaled_height,
            block_x,
            block_y,
            params.scale_shift,
            coeffs,
            dc_only
        );
    }
}

kernel void jpeg_decode_fast444_scaled_region(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast444ScaledParams &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *restart_offsets [[buffer(14)]],
    device JpegDecodeStatus *status [[buffer(15)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(16)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    thread BitReader br;
    uint start_mcu = 0u;
    uint end_mcu = 0u;
    int y_prev_dc = 0;
    int cb_prev_dc = 0;
    int cr_prev_dc = 0;
    if (!configure_entropy_thread(
        gid,
        total_mcus,
        params.restart_interval_mcus,
        params.restart_offset_count,
        params.restart_start_mcu,
        restart_offsets,
        entropy_checkpoints,
        br,
        start_mcu,
        end_mcu,
        y_prev_dc,
        cb_prev_dc,
        cr_prev_dc
    )) {
        return;
    }
    device JpegDecodeStatus *thread_status = status + gid;
    thread_status->code = FAST420_STATUS_OK;
    thread_status->detail = 0;
    thread_status->position = 0;
    thread_status->reserved = 0;

    thread short coeffs[64];
    const uint block_size = 8u >> params.scale_shift;

    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint my = mcu_index / params.mcus_per_row;
        const uint mx = mcu_index % params.mcus_per_row;
        const uint block_x = mx * block_size;
        const uint block_y = my * block_size;
        const bool intersects = block_intersects_rect(
            block_x,
            block_y,
            block_size,
            block_size,
            params.origin_x,
            params.origin_y,
            params.scaled_width,
            params.scaled_height
        );
        bool dc_only = false;

        if (intersects) {
            if (!decode_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            deposit_scaled_block_region(
                y_plane,
                params.scaled_width,
                params.scaled_width,
                params.scaled_height,
                params.origin_x,
                params.origin_y,
                block_x,
                block_y,
                params.scale_shift,
                coeffs,
                dc_only
            );

            if (!decode_block(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            deposit_scaled_block_region(
                cb_plane,
                params.scaled_width,
                params.scaled_width,
                params.scaled_height,
                params.origin_x,
                params.origin_y,
                block_x,
                block_y,
                params.scale_shift,
                coeffs,
                dc_only
            );

            if (!decode_block(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            deposit_scaled_block_region(
                cr_plane,
                params.scaled_width,
                params.scaled_width,
                params.scaled_height,
                params.origin_x,
                params.origin_y,
                block_x,
                block_y,
                params.scale_shift,
                coeffs,
                dc_only
            );
        } else {
            if (!decode_block_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_prev_dc, thread_status)) {
                return;
            }
            if (!decode_block_skip(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_prev_dc, thread_status)) {
                return;
            }
            if (!decode_block_skip(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_prev_dc, thread_status)) {
                return;
            }
        }
    }
}

kernel void jpeg_decode_fast444_scaled_region_batch(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFastRegionScaledBatchParams &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *entropy_offsets [[buffer(14)]],
    device const uint *entropy_lens [[buffer(15)]],
    device JpegDecodeStatus *status [[buffer(16)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(17)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    thread BitReader br;
    uint tile_index = 0u;
    uint start_mcu = 0u;
    uint end_mcu = 0u;
    uint entropy_end = 0u;
    int y_prev_dc = 0;
    int cb_prev_dc = 0;
    int cr_prev_dc = 0;
    if (!configure_batch_entropy_thread(
        gid,
        total_mcus,
        params.segment_count,
        params.tile_count,
        entropy_offsets,
        entropy_lens,
        entropy_checkpoints,
        br,
        tile_index,
        start_mcu,
        end_mcu,
        entropy_end,
        y_prev_dc,
        cb_prev_dc,
        cr_prev_dc
    )) {
        return;
    }
    device JpegDecodeStatus *thread_status = status + gid;
    thread_status->code = FAST420_STATUS_OK;
    thread_status->detail = 0;
    thread_status->position = 0;
    thread_status->reserved = 0;

    const uint plane_base = tile_index * params.scaled_width * params.scaled_height;
    device uchar *tile_y_plane = y_plane + plane_base;
    device uchar *tile_cb_plane = cb_plane + plane_base;
    device uchar *tile_cr_plane = cr_plane + plane_base;

    thread short coeffs[64];
    const uint block_size = 8u >> params.scale_shift;

    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint my = mcu_index / params.mcus_per_row;
        const uint mx = mcu_index % params.mcus_per_row;
        const uint block_x = mx * block_size;
        const uint block_y = my * block_size;
        const bool intersects = block_intersects_rect(
            block_x,
            block_y,
            block_size,
            block_size,
            params.origin_x,
            params.origin_y,
            params.scaled_width,
            params.scaled_height
        );
        bool dc_only = false;

        if (intersects) {
            if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            deposit_scaled_block_region(
                tile_y_plane,
                params.scaled_width,
                params.scaled_width,
                params.scaled_height,
                params.origin_x,
                params.origin_y,
                block_x,
                block_y,
                params.scale_shift,
                coeffs,
                dc_only
            );

            if (!decode_block(br, entropy, entropy_end, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            deposit_scaled_block_region(
                tile_cb_plane,
                params.scaled_width,
                params.scaled_width,
                params.scaled_height,
                params.origin_x,
                params.origin_y,
                block_x,
                block_y,
                params.scale_shift,
                coeffs,
                dc_only
            );

            if (!decode_block(br, entropy, entropy_end, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
                return;
            }
            deposit_scaled_block_region(
                tile_cr_plane,
                params.scaled_width,
                params.scaled_width,
                params.scaled_height,
                params.origin_x,
                params.origin_y,
                block_x,
                block_y,
                params.scale_shift,
                coeffs,
                dc_only
            );
        } else {
            if (!decode_block_skip(br, entropy, entropy_end, y_dc, y_ac, y_prev_dc, thread_status)) {
                return;
            }
            if (!decode_block_skip(br, entropy, entropy_end, cb_dc, cb_ac, cb_prev_dc, thread_status)) {
                return;
            }
            if (!decode_block_skip(br, entropy, entropy_end, cr_dc, cr_ac, cr_prev_dc, thread_status)) {
                return;
            }
        }
    }
}

kernel void jpeg_pack_420(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420Params &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint y_idx = gid.y * params.width + gid.x;
    if (params.out_format == OUT_GRAY) {
        out[gid.y * params.out_stride + gid.x] = y_plane[y_idx];
        return;
    }

    const uint chroma_y = min(gid.y / 2u, params.chroma_height - 1u);
    const uint near_y = (gid.y & 1u) == 0u
        ? (chroma_y == 0u ? 0u : chroma_y - 1u)
        : min(chroma_y + 1u, params.chroma_height - 1u);
    device const uchar *curr_cb = cb_plane + chroma_y * params.chroma_width;
    device const uchar *near_cb = cb_plane + near_y * params.chroma_width;
    device const uchar *curr_cr = cr_plane + chroma_y * params.chroma_width;
    device const uchar *near_cr = cr_plane + near_y * params.chroma_width;

    const uchar cb = h2v2_sample(near_cb, curr_cb, params.chroma_width, gid.x);
    const uchar cr = h2v2_sample(near_cr, curr_cr, params.chroma_width, gid.x);
    const int y = int(y_plane[y_idx]);
    const int cb_centered = int(cb) - 128;
    const int cr_centered = int(cr) - 128;

    uint out_idx = gid.y * params.out_stride + gid.x * (params.out_format == OUT_RGB ? 3u : 4u);
    out[out_idx] = clamp_u8(y + ((91881 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 1] = clamp_u8(y - ((22554 * cb_centered + 46802 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 2] = clamp_u8(y + ((116130 * cb_centered + (1 << 15)) >> 16));
    if (params.out_format == OUT_RGBA) {
        out[out_idx + 3] = uchar(params.alpha);
    }
}

kernel void jpeg_pack_420_rgb(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420Params &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint y_idx = gid.y * params.width + gid.x;
    const uint chroma_y = min(gid.y / 2u, params.chroma_height - 1u);
    const uint near_y = (gid.y & 1u) == 0u
        ? (chroma_y == 0u ? 0u : chroma_y - 1u)
        : min(chroma_y + 1u, params.chroma_height - 1u);
    device const uchar *curr_cb = cb_plane + chroma_y * params.chroma_width;
    device const uchar *near_cb = cb_plane + near_y * params.chroma_width;
    device const uchar *curr_cr = cr_plane + chroma_y * params.chroma_width;
    device const uchar *near_cr = cr_plane + near_y * params.chroma_width;

    const uchar cb = h2v2_sample(near_cb, curr_cb, params.chroma_width, gid.x);
    const uchar cr = h2v2_sample(near_cr, curr_cr, params.chroma_width, gid.x);
    const int y = int(y_plane[y_idx]);
    const int cb_centered = int(cb) - 128;
    const int cr_centered = int(cr) - 128;

    const uint out_idx = gid.y * params.out_stride + gid.x * 3u;
    out[out_idx] = clamp_u8(y + ((91881 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 1] = clamp_u8(y - ((22554 * cb_centered + 46802 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 2] = clamp_u8(y + ((116130 * cb_centered + (1 << 15)) >> 16));
}

kernel void jpeg_pack_420_rgb_batch(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420BatchParams &params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.tile_count) {
        return;
    }

    const uint y_plane_base = gid.z * params.width * params.height;
    const uint chroma_plane_base = gid.z * params.chroma_width * params.chroma_height;
    device const uchar *tile_y_plane = y_plane + y_plane_base;
    device const uchar *tile_cb_plane = cb_plane + chroma_plane_base;
    device const uchar *tile_cr_plane = cr_plane + chroma_plane_base;

    const uint y_idx = gid.y * params.width + gid.x;
    const uint chroma_y = min(gid.y / 2u, params.chroma_height - 1u);
    const uint near_y = (gid.y & 1u) == 0u
        ? (chroma_y == 0u ? 0u : chroma_y - 1u)
        : min(chroma_y + 1u, params.chroma_height - 1u);
    device const uchar *curr_cb = tile_cb_plane + chroma_y * params.chroma_width;
    device const uchar *near_cb = tile_cb_plane + near_y * params.chroma_width;
    device const uchar *curr_cr = tile_cr_plane + chroma_y * params.chroma_width;
    device const uchar *near_cr = tile_cr_plane + near_y * params.chroma_width;

    const uchar cb = h2v2_sample(near_cb, curr_cb, params.chroma_width, gid.x);
    const uchar cr = h2v2_sample(near_cr, curr_cr, params.chroma_width, gid.x);
    const int y = int(tile_y_plane[y_idx]);
    const int cb_centered = int(cb) - 128;
    const int cr_centered = int(cr) - 128;

    const uint out_base = gid.z * params.out_stride * params.height;
    const uint out_idx = out_base + gid.y * params.out_stride + gid.x * 3u;
    out[out_idx] = clamp_u8(y + ((91881 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 1] = clamp_u8(y - ((22554 * cb_centered + 46802 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 2] = clamp_u8(y + ((116130 * cb_centered + (1 << 15)) >> 16));
}

kernel void jpeg_pack_420_rgba(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420Params &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint y_idx = gid.y * params.width + gid.x;
    const uint chroma_y = min(gid.y / 2u, params.chroma_height - 1u);
    const uint near_y = (gid.y & 1u) == 0u
        ? (chroma_y == 0u ? 0u : chroma_y - 1u)
        : min(chroma_y + 1u, params.chroma_height - 1u);
    device const uchar *curr_cb = cb_plane + chroma_y * params.chroma_width;
    device const uchar *near_cb = cb_plane + near_y * params.chroma_width;
    device const uchar *curr_cr = cr_plane + chroma_y * params.chroma_width;
    device const uchar *near_cr = cr_plane + near_y * params.chroma_width;

    const uchar cb = h2v2_sample(near_cb, curr_cb, params.chroma_width, gid.x);
    const uchar cr = h2v2_sample(near_cr, curr_cr, params.chroma_width, gid.x);
    const int y = int(y_plane[y_idx]);
    const int cb_centered = int(cb) - 128;
    const int cr_centered = int(cr) - 128;

    const uint out_idx = gid.y * params.out_stride + gid.x * 4u;
    out[out_idx] = clamp_u8(y + ((91881 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 1] = clamp_u8(y - ((22554 * cb_centered + 46802 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 2] = clamp_u8(y + ((116130 * cb_centered + (1 << 15)) >> 16));
    out[out_idx + 3] = uchar(params.alpha);
}

kernel void jpeg_pack_422_rgb(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420Params &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint y_idx = gid.y * params.width + gid.x;
    const uint chroma_y = min(gid.y, params.chroma_height - 1u);
    device const uchar *curr_cb = cb_plane + chroma_y * params.chroma_width;
    device const uchar *curr_cr = cr_plane + chroma_y * params.chroma_width;

    const uchar cb = h2v1_sample(curr_cb, params.chroma_width, gid.x);
    const uchar cr = h2v1_sample(curr_cr, params.chroma_width, gid.x);
    const int y = int(y_plane[y_idx]);
    const int cb_centered = int(cb) - 128;
    const int cr_centered = int(cr) - 128;

    const uint out_idx = gid.y * params.out_stride + gid.x * 3u;
    out[out_idx] = clamp_u8(y + ((91881 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 1] = clamp_u8(y - ((22554 * cb_centered + 46802 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 2] = clamp_u8(y + ((116130 * cb_centered + (1 << 15)) >> 16));
}

kernel void jpeg_pack_422_rgb_batch(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420BatchParams &params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.tile_count) {
        return;
    }

    const uint y_plane_base = gid.z * params.width * params.height;
    const uint chroma_plane_base = gid.z * params.chroma_width * params.chroma_height;
    device const uchar *tile_y_plane = y_plane + y_plane_base;
    device const uchar *tile_cb_plane = cb_plane + chroma_plane_base;
    device const uchar *tile_cr_plane = cr_plane + chroma_plane_base;

    const uint y_idx = gid.y * params.width + gid.x;
    const uint chroma_y = min(gid.y, params.chroma_height - 1u);
    device const uchar *curr_cb = tile_cb_plane + chroma_y * params.chroma_width;
    device const uchar *curr_cr = tile_cr_plane + chroma_y * params.chroma_width;

    const uchar cb = h2v1_sample(curr_cb, params.chroma_width, gid.x);
    const uchar cr = h2v1_sample(curr_cr, params.chroma_width, gid.x);
    const int y = int(tile_y_plane[y_idx]);
    const int cb_centered = int(cb) - 128;
    const int cr_centered = int(cr) - 128;

    const uint out_base = gid.z * params.out_stride * params.height;
    const uint out_idx = out_base + gid.y * params.out_stride + gid.x * 3u;
    out[out_idx] = clamp_u8(y + ((91881 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 1] = clamp_u8(y - ((22554 * cb_centered + 46802 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 2] = clamp_u8(y + ((116130 * cb_centered + (1 << 15)) >> 16));
}

kernel void jpeg_pack_422_rgba(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420Params &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint y_idx = gid.y * params.width + gid.x;
    const uint chroma_y = min(gid.y, params.chroma_height - 1u);
    device const uchar *curr_cb = cb_plane + chroma_y * params.chroma_width;
    device const uchar *curr_cr = cr_plane + chroma_y * params.chroma_width;

    const uchar cb = h2v1_sample(curr_cb, params.chroma_width, gid.x);
    const uchar cr = h2v1_sample(curr_cr, params.chroma_width, gid.x);
    const int y = int(y_plane[y_idx]);
    const int cb_centered = int(cb) - 128;
    const int cr_centered = int(cr) - 128;

    const uint out_idx = gid.y * params.out_stride + gid.x * 4u;
    out[out_idx] = clamp_u8(y + ((91881 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 1] = clamp_u8(y - ((22554 * cb_centered + 46802 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 2] = clamp_u8(y + ((116130 * cb_centered + (1 << 15)) >> 16));
    out[out_idx + 3] = uchar(params.alpha);
}

kernel void jpeg_pack_422_windowed(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420WindowedPackParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint src_x = gid.x + params.src_x;
    const uint src_y = gid.y + params.src_y;
    if (src_x >= params.src_width || src_y >= params.src_height) {
        return;
    }

    const uint y_idx = src_y * params.src_width + src_x;
    if (params.out_format == OUT_GRAY) {
        out[gid.y * params.out_stride + gid.x] = y_plane[y_idx];
        return;
    }

    const uint chroma_y = min(src_y, params.chroma_height - 1u);
    device const uchar *curr_cb = cb_plane + chroma_y * params.chroma_width;
    device const uchar *curr_cr = cr_plane + chroma_y * params.chroma_width;

    const uchar cb = h2v1_sample(curr_cb, params.chroma_width, src_x);
    const uchar cr = h2v1_sample(curr_cr, params.chroma_width, src_x);
    const int y = int(y_plane[y_idx]);
    const int cb_centered = int(cb) - 128;
    const int cr_centered = int(cr) - 128;

    uint out_idx = gid.y * params.out_stride + gid.x * (params.out_format == OUT_RGB ? 3u : 4u);
    out[out_idx] = clamp_u8(y + ((91881 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 1] = clamp_u8(y - ((22554 * cb_centered + 46802 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 2] = clamp_u8(y + ((116130 * cb_centered + (1 << 15)) >> 16));
    if (params.out_format == OUT_RGBA) {
        out[out_idx + 3] = uchar(params.alpha);
    }
}

kernel void jpeg_pack_422_windowed_rgb(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420WindowedPackParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint src_x = gid.x + params.src_x;
    const uint src_y = gid.y + params.src_y;
    if (src_x >= params.src_width || src_y >= params.src_height) {
        return;
    }

    const uint y_idx = src_y * params.src_width + src_x;
    const uint chroma_y = min(src_y, params.chroma_height - 1u);
    device const uchar *curr_cb = cb_plane + chroma_y * params.chroma_width;
    device const uchar *curr_cr = cr_plane + chroma_y * params.chroma_width;

    const uchar cb = h2v1_sample(curr_cb, params.chroma_width, src_x);
    const uchar cr = h2v1_sample(curr_cr, params.chroma_width, src_x);
    const int y = int(y_plane[y_idx]);
    const int cb_centered = int(cb) - 128;
    const int cr_centered = int(cr) - 128;

    const uint out_idx = gid.y * params.out_stride + gid.x * 3u;
    out[out_idx] = clamp_u8(y + ((91881 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 1] = clamp_u8(y - ((22554 * cb_centered + 46802 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 2] = clamp_u8(y + ((116130 * cb_centered + (1 << 15)) >> 16));
}

kernel void jpeg_pack_422_windowed_rgb_batch(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegWindowedPackBatchParams &params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.tile_count) {
        return;
    }

    const uint src_x = gid.x + params.src_x;
    const uint src_y = gid.y + params.src_y;
    if (src_x >= params.src_width || src_y >= params.src_height) {
        return;
    }

    const uint y_plane_base = gid.z * params.src_width * params.src_height;
    const uint chroma_plane_base = gid.z * params.chroma_width * params.chroma_height;
    device const uchar *tile_y_plane = y_plane + y_plane_base;
    device const uchar *tile_cb_plane = cb_plane + chroma_plane_base;
    device const uchar *tile_cr_plane = cr_plane + chroma_plane_base;

    const uint y_idx = src_y * params.src_width + src_x;
    const uint chroma_y = min(src_y, params.chroma_height - 1u);
    device const uchar *curr_cb = tile_cb_plane + chroma_y * params.chroma_width;
    device const uchar *curr_cr = tile_cr_plane + chroma_y * params.chroma_width;

    const uchar cb = h2v1_sample(curr_cb, params.chroma_width, src_x);
    const uchar cr = h2v1_sample(curr_cr, params.chroma_width, src_x);
    const int y = int(tile_y_plane[y_idx]);
    const int cb_centered = int(cb) - 128;
    const int cr_centered = int(cr) - 128;

    const uint out_base = gid.z * params.out_stride * params.height;
    const uint out_idx = out_base + gid.y * params.out_stride + gid.x * 3u;
    out[out_idx] = clamp_u8(y + ((91881 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 1] = clamp_u8(y - ((22554 * cb_centered + 46802 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 2] = clamp_u8(y + ((116130 * cb_centered + (1 << 15)) >> 16));
}

kernel void jpeg_pack_422_windowed_rgba(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420WindowedPackParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint src_x = gid.x + params.src_x;
    const uint src_y = gid.y + params.src_y;
    if (src_x >= params.src_width || src_y >= params.src_height) {
        return;
    }

    const uint y_idx = src_y * params.src_width + src_x;
    const uint chroma_y = min(src_y, params.chroma_height - 1u);
    device const uchar *curr_cb = cb_plane + chroma_y * params.chroma_width;
    device const uchar *curr_cr = cr_plane + chroma_y * params.chroma_width;

    const uchar cb = h2v1_sample(curr_cb, params.chroma_width, src_x);
    const uchar cr = h2v1_sample(curr_cr, params.chroma_width, src_x);
    const int y = int(y_plane[y_idx]);
    const int cb_centered = int(cb) - 128;
    const int cr_centered = int(cr) - 128;

    const uint out_idx = gid.y * params.out_stride + gid.x * 4u;
    out[out_idx] = clamp_u8(y + ((91881 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 1] = clamp_u8(y - ((22554 * cb_centered + 46802 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 2] = clamp_u8(y + ((116130 * cb_centered + (1 << 15)) >> 16));
    out[out_idx + 3] = uchar(params.alpha);
}

kernel void jpeg_pack_420_windowed(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420WindowedPackParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint src_x = gid.x + params.src_x;
    const uint src_y = gid.y + params.src_y;
    if (src_x >= params.src_width || src_y >= params.src_height) {
        return;
    }

    const uint y_idx = src_y * params.src_width + src_x;
    if (params.out_format == OUT_GRAY) {
        out[gid.y * params.out_stride + gid.x] = y_plane[y_idx];
        return;
    }

    const uint chroma_y = min(src_y / 2u, params.chroma_height - 1u);
    const uint near_y = (src_y & 1u) == 0u
        ? (chroma_y == 0u ? 0u : chroma_y - 1u)
        : min(chroma_y + 1u, params.chroma_height - 1u);
    device const uchar *curr_cb = cb_plane + chroma_y * params.chroma_width;
    device const uchar *near_cb = cb_plane + near_y * params.chroma_width;
    device const uchar *curr_cr = cr_plane + chroma_y * params.chroma_width;
    device const uchar *near_cr = cr_plane + near_y * params.chroma_width;

    const uchar cb = h2v2_sample(near_cb, curr_cb, params.chroma_width, src_x);
    const uchar cr = h2v2_sample(near_cr, curr_cr, params.chroma_width, src_x);
    const int y = int(y_plane[y_idx]);
    const int cb_centered = int(cb) - 128;
    const int cr_centered = int(cr) - 128;

    uint out_idx = gid.y * params.out_stride + gid.x * (params.out_format == OUT_RGB ? 3u : 4u);
    out[out_idx] = clamp_u8(y + ((91881 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 1] = clamp_u8(y - ((22554 * cb_centered + 46802 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 2] = clamp_u8(y + ((116130 * cb_centered + (1 << 15)) >> 16));
    if (params.out_format == OUT_RGBA) {
        out[out_idx + 3] = uchar(params.alpha);
    }
}

kernel void jpeg_pack_420_windowed_rgb(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420WindowedPackParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint src_x = gid.x + params.src_x;
    const uint src_y = gid.y + params.src_y;
    if (src_x >= params.src_width || src_y >= params.src_height) {
        return;
    }

    const uint y_idx = src_y * params.src_width + src_x;
    const uint chroma_y = min(src_y / 2u, params.chroma_height - 1u);
    const uint near_y = (src_y & 1u) == 0u
        ? (chroma_y == 0u ? 0u : chroma_y - 1u)
        : min(chroma_y + 1u, params.chroma_height - 1u);
    device const uchar *curr_cb = cb_plane + chroma_y * params.chroma_width;
    device const uchar *near_cb = cb_plane + near_y * params.chroma_width;
    device const uchar *curr_cr = cr_plane + chroma_y * params.chroma_width;
    device const uchar *near_cr = cr_plane + near_y * params.chroma_width;

    const uchar cb = h2v2_sample(near_cb, curr_cb, params.chroma_width, src_x);
    const uchar cr = h2v2_sample(near_cr, curr_cr, params.chroma_width, src_x);
    const int y = int(y_plane[y_idx]);
    const int cb_centered = int(cb) - 128;
    const int cr_centered = int(cr) - 128;

    const uint out_idx = gid.y * params.out_stride + gid.x * 3u;
    out[out_idx] = clamp_u8(y + ((91881 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 1] = clamp_u8(y - ((22554 * cb_centered + 46802 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 2] = clamp_u8(y + ((116130 * cb_centered + (1 << 15)) >> 16));
}

kernel void jpeg_pack_420_windowed_rgb_batch(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegWindowedPackBatchParams &params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.tile_count) {
        return;
    }

    const uint src_x = gid.x + params.src_x;
    const uint src_y = gid.y + params.src_y;
    if (src_x >= params.src_width || src_y >= params.src_height) {
        return;
    }

    const uint y_plane_base = gid.z * params.src_width * params.src_height;
    const uint chroma_plane_base = gid.z * params.chroma_width * params.chroma_height;
    device const uchar *tile_y_plane = y_plane + y_plane_base;
    device const uchar *tile_cb_plane = cb_plane + chroma_plane_base;
    device const uchar *tile_cr_plane = cr_plane + chroma_plane_base;

    const uint y_idx = src_y * params.src_width + src_x;
    const uint chroma_y = min(src_y / 2u, params.chroma_height - 1u);
    const uint near_y = (src_y & 1u) == 0u
        ? (chroma_y == 0u ? 0u : chroma_y - 1u)
        : min(chroma_y + 1u, params.chroma_height - 1u);
    device const uchar *curr_cb = tile_cb_plane + chroma_y * params.chroma_width;
    device const uchar *near_cb = tile_cb_plane + near_y * params.chroma_width;
    device const uchar *curr_cr = tile_cr_plane + chroma_y * params.chroma_width;
    device const uchar *near_cr = tile_cr_plane + near_y * params.chroma_width;

    const uchar cb = h2v2_sample(near_cb, curr_cb, params.chroma_width, src_x);
    const uchar cr = h2v2_sample(near_cr, curr_cr, params.chroma_width, src_x);
    const int y = int(tile_y_plane[y_idx]);
    const int cb_centered = int(cb) - 128;
    const int cr_centered = int(cr) - 128;

    const uint out_base = gid.z * params.out_stride * params.height;
    const uint out_idx = out_base + gid.y * params.out_stride + gid.x * 3u;
    out[out_idx] = clamp_u8(y + ((91881 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 1] = clamp_u8(y - ((22554 * cb_centered + 46802 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 2] = clamp_u8(y + ((116130 * cb_centered + (1 << 15)) >> 16));
}

kernel void jpeg_pack_420_windowed_rgba(
    device const uchar *y_plane [[buffer(0)]],
    device const uchar *cb_plane [[buffer(1)]],
    device const uchar *cr_plane [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant JpegFast420WindowedPackParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint src_x = gid.x + params.src_x;
    const uint src_y = gid.y + params.src_y;
    if (src_x >= params.src_width || src_y >= params.src_height) {
        return;
    }

    const uint y_idx = src_y * params.src_width + src_x;
    const uint chroma_y = min(src_y / 2u, params.chroma_height - 1u);
    const uint near_y = (src_y & 1u) == 0u
        ? (chroma_y == 0u ? 0u : chroma_y - 1u)
        : min(chroma_y + 1u, params.chroma_height - 1u);
    device const uchar *curr_cb = cb_plane + chroma_y * params.chroma_width;
    device const uchar *near_cb = cb_plane + near_y * params.chroma_width;
    device const uchar *curr_cr = cr_plane + chroma_y * params.chroma_width;
    device const uchar *near_cr = cr_plane + near_y * params.chroma_width;

    const uchar cb = h2v2_sample(near_cb, curr_cb, params.chroma_width, src_x);
    const uchar cr = h2v2_sample(near_cr, curr_cr, params.chroma_width, src_x);
    const int y = int(y_plane[y_idx]);
    const int cb_centered = int(cb) - 128;
    const int cr_centered = int(cr) - 128;

    const uint out_idx = gid.y * params.out_stride + gid.x * 4u;
    out[out_idx] = clamp_u8(y + ((91881 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 1] = clamp_u8(y - ((22554 * cb_centered + 46802 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 2] = clamp_u8(y + ((116130 * cb_centered + (1 << 15)) >> 16));
    out[out_idx + 3] = uchar(params.alpha);
}
