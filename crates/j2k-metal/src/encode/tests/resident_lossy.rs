// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::J2kEncodeStageAccelerator;
use j2k_native::{CpuOnlyJ2kEncodeStageAccelerator, DecodeSettings, EncodeOptions, Image};

#[test]
fn resident_lossy_reuses_private_buffers_after_warmup() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let device = j2k_metal_support::system_default_device().unwrap();
    crate::engine::with_isolated_runtime_for_device_for_test(&device, || {
        let pixels = j2k_test_support::patterned_rgb8(65, 49);
        let options = EncodeOptions {
            reversible: false,
            num_decomposition_levels: 3,
            guard_bits: 2,
            use_ht_block_coding: true,
            ..EncodeOptions::default()
        };
        let mut accelerator = crate::MetalEncodeStageAccelerator::default();
        let mut encode = || {
            j2k_native::encode_with_accelerator(
                &pixels,
                65,
                49,
                3,
                8,
                false,
                &options,
                &mut accelerator,
            )
            .unwrap()
        };
        let first = encode();
        crate::engine::reset_private_buffer_pool_misses_for_test();
        crate::engine::reset_shared_buffer_pool_misses_for_test();
        assert_eq!(encode(), first);
        assert_eq!(
            crate::engine::shared_buffer_pool_misses_for_test(),
            0,
            "a completed lossy encode must return shared staging and metadata for reuse"
        );
        assert_eq!(
            crate::engine::private_buffer_pool_misses_for_test(),
            0,
            "a completed lossy encode must return its private intermediates for reuse"
        );
        Ok(())
    })
    .unwrap();
}

#[test]
fn resident_lossy_submits_one_gpu_pipeline() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    crate::engine::reset_metal_command_buffers_for_test();
    let pixels = j2k_test_support::patterned_rgb8(65, 49);
    let options = EncodeOptions {
        reversible: false,
        num_decomposition_levels: 3,
        guard_bits: 2,
        use_ht_block_coding: true,
        ..EncodeOptions::default()
    };
    assert_resident_matches_scalar(&pixels, 65, 49, 3, &options);
    assert_eq!(crate::engine::metal_command_buffers_for_test(), 1,
        "transform, HT, packet headers and parallel payload copy must precede one completion boundary");
}

#[test]
fn resident_lossy_ht_matches_scalar_codestream_and_pixels() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    for (width, height, components, levels, block_size, scale) in [
        (64, 48, 1, 3, 64, 1.0),
        (65, 49, 1, 3, 64, 1.0),
        (64, 48, 3, 3, 64, 1.0),
        (65, 49, 3, 3, 64, 1.0),
        (1, 1, 1, 0, 64, 1.0),
        (1, 7, 1, 0, 64, 1.0),
        (7, 1, 3, 0, 64, 1.0),
        (17, 9, 1, 1, 32, 0.75),
        (65, 49, 3, 2, 32, 1.3),
        (64, 64, 3, 4, 64, 5.0),
    ] {
        let pixels = if components == 1 {
            j2k_test_support::patterned_gray8(width, height)
        } else {
            j2k_test_support::patterned_rgb8(width, height)
        };
        let options = EncodeOptions {
            reversible: false,
            num_decomposition_levels: levels,
            code_block_width_exp: if block_size == 32 { 3 } else { 4 },
            code_block_height_exp: if block_size == 32 { 3 } else { 4 },
            irreversible_quantization_scale: scale,
            irreversible_quantization_subband_scales: if block_size == 32 {
                j2k_native::IrreversibleQuantizationSubbandScales {
                    low_low: 0.75,
                    high_low: 1.25,
                    low_high: 2.5,
                    high_high: 5.0,
                }
            } else {
                j2k_native::IrreversibleQuantizationSubbandScales::default()
            },
            guard_bits: 2,
            use_mct: components == 3,
            use_ht_block_coding: true,
            ..EncodeOptions::default()
        };
        assert_resident_matches_scalar(&pixels, width, height, components, &options);
    }
}

fn assert_resident_matches_scalar(
    pixels: &[u8],
    width: u32,
    height: u32,
    components: u16,
    options: &EncodeOptions,
) {
    let expected = j2k_native::encode_with_accelerator(
        pixels,
        width,
        height,
        components,
        8,
        false,
        options,
        &mut CpuOnlyJ2kEncodeStageAccelerator,
    )
    .expect("scalar lossy encode");
    let mut accelerator = crate::MetalEncodeStageAccelerator::default();
    let actual = j2k_native::encode_with_accelerator(
        pixels,
        width,
        height,
        components,
        8,
        false,
        options,
        &mut accelerator,
    )
    .expect("resident lossy encode");
    assert!(
        accelerator.ht_tile_required_magnitude_bound().is_some(),
        "resident lossy tile hook must run"
    );
    assert_eq!(actual, expected, "{width}x{height} components {components}");
    let decode = |bytes: &[u8]| {
        Image::new(bytes, &DecodeSettings::default())
            .expect("parse lossy")
            .decode_native()
            .expect("decode lossy")
            .data
    };
    assert_eq!(decode(&actual), decode(&expected));
}

#[test]
fn resident_lossy_declines_unsupported_metadata_before_dispatch() {
    use j2k_native::{J2kHtj2kTileEncodeJob, J2kPacketizationProgressionOrder};
    let pixels = [0; 64];
    let steps = [(10, 0); 4];
    let job = J2kHtj2kTileEncodeJob {
        pixels: &pixels,
        width: 8,
        height: 8,
        num_components: 1,
        bit_depth: 8,
        signed: false,
        num_decomposition_levels: 1,
        reversible: false,
        use_mct: false,
        guard_bits: 2,
        code_block_width: 64,
        code_block_height: 64,
        progression_order: J2kPacketizationProgressionOrder::Lrcp,
        component_sampling: &[(1, 1)],
        quantization_steps: &steps,
    };
    for unsupported in [
        J2kHtj2kTileEncodeJob {
            reversible: true,
            ..job
        },
        J2kHtj2kTileEncodeJob {
            signed: true,
            ..job
        },
        J2kHtj2kTileEncodeJob {
            bit_depth: 16,
            ..job
        },
        J2kHtj2kTileEncodeJob {
            num_components: 4,
            ..job
        },
        J2kHtj2kTileEncodeJob { width: 0, ..job },
        J2kHtj2kTileEncodeJob { pixels: &[], ..job },
        J2kHtj2kTileEncodeJob {
            quantization_steps: &[],
            ..job
        },
        J2kHtj2kTileEncodeJob {
            component_sampling: &[(2, 1)],
            ..job
        },
        J2kHtj2kTileEncodeJob {
            code_block_width: 128,
            ..job
        },
        J2kHtj2kTileEncodeJob {
            code_block_height: 32,
            ..job
        },
        J2kHtj2kTileEncodeJob {
            quantization_steps: &[(32, 0); 4],
            ..job
        },
        J2kHtj2kTileEncodeJob {
            quantization_steps: &[(10, 2048); 4],
            ..job
        },
    ] {
        assert!(
            super::super::resident_lossy::encode_resident_lossy_ht_tile(unsupported)
                .expect("unsupported metadata declines without allocating GPU buffers")
                .is_none()
        );
    }
}

#[test]
fn resident_lossy_single_layer_byte_budget_keeps_native_allocation() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let pixels = j2k_test_support::patterned_rgb8(128, 128);
    let options = EncodeOptions {
        reversible: false,
        use_ht_block_coding: true,
        use_mct: true,
        num_decomposition_levels: 3,
        guard_bits: 2,
        num_layers: 1,
        quality_layer_byte_targets: vec![8000],
        ..EncodeOptions::default()
    };
    let expected = j2k_native::encode_with_accelerator(
        &pixels,
        128,
        128,
        3,
        8,
        false,
        &options,
        &mut CpuOnlyJ2kEncodeStageAccelerator,
    )
    .expect("scalar byte-budget encode");
    let mut accelerator = crate::MetalEncodeStageAccelerator::default();
    let actual = j2k_native::encode_with_accelerator(
        &pixels,
        128,
        128,
        3,
        8,
        false,
        &options,
        &mut accelerator,
    )
    .expect("Metal byte-budget encode");
    assert!(
        accelerator.ht_tile_required_magnitude_bound().is_none(),
        "whole-tile hook must decline budgets absent from its input contract"
    );
    assert_eq!(
        actual, expected,
        "byte-budget output must match scalar allocation"
    );
}

#[test]
fn resident_lossy_zero_coefficients_and_progressions_match_scalar() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    for progression_order in [
        j2k_native::EncodeProgressionOrder::Lrcp,
        j2k_native::EncodeProgressionOrder::Rlcp,
        j2k_native::EncodeProgressionOrder::Rpcl,
        j2k_native::EncodeProgressionOrder::Pcrl,
        j2k_native::EncodeProgressionOrder::Cprl,
    ] {
        for value in [0, 128, 255] {
            let pixels = vec![value; 33 * 17 * 3];
            let options = EncodeOptions {
                reversible: false,
                use_ht_block_coding: true,
                use_mct: true,
                guard_bits: 2,
                num_decomposition_levels: 3,
                progression_order,
                ..EncodeOptions::default()
            };
            assert_resident_matches_scalar(&pixels, 33, 17, 3, &options);
        }
    }
}
