const SHADER_SOURCE: &str = include_str!("../src/shaders.metal");
const COMPUTE_SOURCE: &str = include_str!("../src/compute.rs");

#[test]
fn decode_loops_advance_mcu_coordinates_incrementally() {
    let marker = "inline void init_mcu_cursor(";
    let fast_decode_source = &SHADER_SOURCE[SHADER_SOURCE
        .find(marker)
        .expect("shader source should define init_mcu_cursor")..];
    assert!(
        !fast_decode_source.contains("mcu_index / params.mcus_per_row"),
        "decode hot loops must not divide every MCU to recover my"
    );
    assert!(
        !fast_decode_source.contains("mcu_index % params.mcus_per_row"),
        "decode hot loops must not modulo every MCU to recover mx"
    );
}

#[test]
fn batch_rgb_pack_kernels_process_pixel_groups() {
    let compact_compute = COMPUTE_SOURCE
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();

    assert!(
        SHADER_SOURCE.contains("const uint x0 = gid.x * 2u;"),
        "batch RGB pack kernels should group horizontal pixels per thread"
    );
    assert!(
        SHADER_SOURCE.contains("const uint y0 = gid.y * 2u;"),
        "420 batch RGB pack should group 2x2 output pixels per thread"
    );
    assert!(
        compact_compute
            .contains("(packed_pair_extent(width),packed_pair_extent(height),tile_count_u32,)"),
        "420 batch RGB pack dispatch should use grouped 2x2 grid dimensions"
    );
    assert!(
        compact_compute.contains("(packed_pair_extent(width),height,tile_count_u32)"),
        "422 batch RGB pack dispatch should use grouped horizontal grid dimensions"
    );
}

#[test]
fn fast420_batch_split_path_stays_wired() {
    assert!(
        SHADER_SOURCE.contains("kernel void jpeg_decode_fast420_batch_coeffs"),
        "split fast420 batch must keep the entropy-to-coefficients kernel"
    );
    assert!(
        SHADER_SOURCE.contains("kernel void jpeg_idct_deposit_fast420_batch"),
        "split fast420 batch must keep the IDCT/deposit kernel"
    );
    assert!(
        COMPUTE_SOURCE.contains("SIGNINUM_JPEG_METAL_SPLIT_FAST420_BATCH"),
        "split fast420 batch must stay opt-in until benchmarks promote it"
    );
}

#[test]
fn entropy_fast_paths_stay_wired() {
    assert!(
        SHADER_SOURCE.contains(
            "return refill_four_bytes(br, bytes, len) || refill_one_byte(br, bytes, len);"
        ),
        "bit refill must try a 4-byte load before falling back to byte refill"
    );
    assert!(
        SHADER_SOURCE.contains("const uchar len9 = table.fast_len[fast_index];")
            && SHADER_SOURCE.contains("symbol = table.fast_symbol[fast_index];"),
        "Huffman decode must keep the 9-bit fast table path"
    );
    assert!(
        COMPUTE_SOURCE.contains("fast_symbol: [u8; 512]")
            && COMPUTE_SOURCE.contains("fast_len: [u8; 512]"),
        "host PreparedHuffman layout must include the 9-bit fast table"
    );
}

#[test]
fn jpeg_encode_batch_entropy_path_stays_wired() {
    assert!(
        SHADER_SOURCE.contains("kernel void jpeg_encode_baseline_entropy_batch"),
        "JPEG encode should keep the batched entropy kernel"
    );
    assert!(
        COMPUTE_SOURCE.contains("jpeg_baseline_encode_batch_pipeline"),
        "host runtime should keep a compiled batch entropy pipeline"
    );
    assert!(
        COMPUTE_SOURCE.contains("encode_jpeg_baseline_entropy_batch_with_session"),
        "public batch encode path should dispatch through the Metal batch helper"
    );
}

#[test]
fn jpeg_encode_parallel_path_accepts_restart_segments() {
    assert!(
        SHADER_SOURCE.contains("jpeg_encode_push_restart_marker"),
        "JPEG encode shader should emit restart markers on GPU"
    );
    assert!(
        SHADER_SOURCE.contains("if (params.restart_interval_mcus != 0u && mcus_since_restart == params.restart_interval_mcus)"),
        "JPEG encode shader should handle restart intervals inside the entropy kernel"
    );
    assert!(
        !COMPUTE_SOURCE
            .contains("if first.restart_interval_mcus != 0 {\n        return Ok(None);\n    }"),
        "restart intervals must not force the JPEG batch encoder off the optimized Metal path"
    );
}
