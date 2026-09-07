// SPDX-License-Identifier: MIT OR Apache-2.0

//! Separate compiler inputs prevent optional encode/profile kernels from blocking decode.

const PRELUDE: &str = "#include <metal_stdlib>\nusing namespace metal;\n";

pub(super) fn decode_shader_source() -> String {
    [
        PRELUDE,
        include_str!("../pack.metal"),
        include_str!("../classic/abi.metal"),
        include_str!("../classic/constants.metal"),
        include_str!("../classic/qe_table.metal"),
        include_str!("../classic/context_tables.metal"),
        include_str!("../classic/support.metal"),
        include_str!("../classic/mq_decoder.metal"),
        include_str!("../classic/bypass_decoder.metal"),
        include_str!("../classic/pass_logic.metal"),
        include_str!("../classic/decode_kernels.metal"),
        j2k_codec_math::generated::DWT97_CONSTANTS_METAL,
        include_str!("../idwt.metal"),
        include_str!("../mct_abi.metal"),
        include_str!("../mct.metal"),
        include_str!("../store.metal"),
        include_str!("../sampled_plane.metal"),
        include_str!("../store_native_color_batch.metal"),
        include_str!("../ht_cleanup.metal"),
    ]
    .join("\n")
}

pub(super) fn encode_shader_source() -> String {
    [
        PRELUDE,
        include_str!("../encode_input.metal"),
        include_str!("../classic/abi.metal"),
        include_str!("../classic/constants.metal"),
        include_str!("../classic/qe_table.metal"),
        include_str!("../classic/context_tables.metal"),
        include_str!("../classic/support.metal"),
        include_str!("../encode_bitstream_shared.metal"),
        include_str!("../encode_bitstream_classic_core.metal"),
        include_str!("../encode_bitstream_classic_kernels.metal"),
        include_str!("../encode_bitstream_ht_refinement.metal"),
        include_str!("../encode_bitstream_ht.metal"),
        include_str!("../encode_bitstream_packetize.metal"),
        j2k_codec_math::generated::DWT97_CONSTANTS_METAL,
        include_str!("../fdwt.metal"),
        include_str!("../mct_abi.metal"),
        include_str!("../forward_mct.metal"),
        include_str!("../quantize.metal"),
    ]
    .join("\n")
}

pub(super) fn profile_shader_source() -> String {
    [
        PRELUDE,
        include_str!("../classic/abi.metal"),
        include_str!("../classic/constants.metal"),
        include_str!("../classic/qe_table.metal"),
        include_str!("../classic/context_tables.metal"),
        include_str!("../classic/support.metal"),
        include_str!("../encode_bitstream_shared.metal"),
        include_str!("../encode_bitstream_classic_core.metal"),
        include_str!("../encode_bitstream_classic_tokens.metal"),
        include_str!("../encode_bitstream_classic_symbol_plan.metal"),
        include_str!("../encode_bitstream_classic_profile_kernels.metal"),
    ]
    .join("\n")
}

pub(super) fn buffers_shader_source() -> String {
    [PRELUDE, include_str!("../buffer_ops.metal")].join("\n")
}

#[cfg(test)]
mod isolation_tests {
    #[test]
    fn decode_source_excludes_encoder_and_profiling_entrypoints() {
        let source = super::decode_shader_source();
        assert!(
            !source.contains("kernel void j2k_encode_classic_code_block"),
            "decoding must not compile encoder entrypoints"
        );
        assert!(
            !source.contains("kernel void j2k_profile_classic_tier1"),
            "decoding must not compile profiling entrypoints"
        );
        assert!(!source.contains("kernel void j2k_forward_"));
        assert!(source.contains("kernel void j2k_idwt_"));
    }

    #[test]
    fn encoder_and_buffer_operations_do_not_compile_profiling_entrypoints() {
        let encode = super::encode_shader_source();
        assert!(encode.contains("kernel void j2k_encode_classic_code_block"));
        assert!(!encode.contains("kernel void j2k_profile_classic_tier1"));
        assert!(!encode.contains("kernel void j2k_idwt_"));
        let buffers = super::buffers_shader_source();
        assert!(buffers.contains("kernel void j2k_validate_bytes_equal"));
        assert!(!buffers.contains("kernel void j2k_profile_classic_tier1"));
    }
}
