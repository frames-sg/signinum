// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use signinum_core::{BackendKind, CodecError};
use signinum_j2k::{
    encode_j2k_lossless, encode_j2k_lossless_with_accelerator, j2k_lossless_decomposition_levels,
    j2k_lossless_decomposition_levels_for_progression, EncodeBackendPreference,
    EncodedHtJ2kCodeBlock, J2kBlockCodingMode, J2kEncodeDispatchReport, J2kEncodeStageAccelerator,
    J2kEncodeValidation, J2kHtCodeBlockEncodeJob, J2kLosslessEncodeOptions, J2kLosslessSamples,
    J2kPacketizationEncodeJob, J2kProgressionOrder, ReversibleTransform,
};
use signinum_j2k_native::{DecodeSettings, Image};

fn decode_native(codestream: &[u8]) -> signinum_j2k_native::RawBitmap {
    Image::new(codestream, &DecodeSettings::default())
        .expect("encoded codestream should parse")
        .decode_native()
        .expect("encoded codestream should decode")
}

#[test]
fn default_lossless_options_use_auto_cpu_safe_profile() {
    let options = J2kLosslessEncodeOptions::default();

    assert_eq!(options.backend, EncodeBackendPreference::Auto);
    assert_eq!(options.block_coding_mode, J2kBlockCodingMode::Classic);
    assert_eq!(options.progression, J2kProgressionOrder::Lrcp);
    assert_eq!(options.reversible_transform, ReversibleTransform::Rct53);
    assert_eq!(options.validation, J2kEncodeValidation::CpuRoundTrip);
}

#[test]
fn lossless_encode_can_skip_facade_cpu_validation_for_external_validation() {
    let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 17) & 0xFF) as u8).collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 3, 8, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::CpuOnly,
            validation: J2kEncodeValidation::External,
            ..J2kLosslessEncodeOptions::default()
        },
    )
    .expect("lossless encode without facade CPU validation");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(decode_native(&encoded.codestream).data, pixels);
}

#[test]
fn cpu_htj2k_lossless_round_trips_gray8() {
    let pixels: Vec<u8> = (0..64).map(|value| (value * 9) as u8).collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::CpuOnly,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            ..J2kLosslessEncodeOptions::default()
        },
    )
    .expect("HTJ2K lossless encode");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert!(encoded
        .codestream
        .windows(2)
        .any(|window| window == [0xFF, 0x50]));
    let cod_offset = marker_offset(&encoded.codestream, 0x52).expect("COD marker");
    assert_eq!(encoded.codestream[cod_offset + 12], 0x40);
    assert_eq!(decode_native(&encoded.codestream).data, pixels);
}

#[test]
fn cpu_htj2k_rpcl_writes_cod_rpcl_and_tlm() {
    let pixels: Vec<u8> = (0..64).map(|value| (value * 11) as u8).collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::CpuOnly,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            progression: J2kProgressionOrder::Rpcl,
            ..J2kLosslessEncodeOptions::default()
        },
    )
    .expect("HTJ2K RPCL lossless encode");

    let cod_offset = marker_offset(&encoded.codestream, 0x52).expect("COD marker");
    assert_eq!(encoded.codestream[cod_offset + 5], 0x02);
    assert!(marker_offset(&encoded.codestream, 0x55).is_some());
    assert_eq!(decode_native(&encoded.codestream).data, pixels);
}

#[test]
fn default_lossless_policy_enables_one_reversible_dwt_level_for_wsi_tiles() {
    let gray = vec![0; 64 * 64];
    let gray_samples = J2kLosslessSamples::new(&gray, 64, 64, 1, 8, false).unwrap();
    assert_eq!(j2k_lossless_decomposition_levels(gray_samples), 1);

    let rgb = vec![0; 512 * 512 * 3];
    let rgb_samples = J2kLosslessSamples::new(&rgb, 512, 512, 3, 8, false).unwrap();
    assert_eq!(j2k_lossless_decomposition_levels(rgb_samples), 1);
}

#[test]
fn default_lossless_policy_keeps_edge_tiles_undecomposed() {
    let gray = vec![0; 63 * 512];
    let samples = J2kLosslessSamples::new(&gray, 63, 512, 1, 8, false).unwrap();

    assert_eq!(j2k_lossless_decomposition_levels(samples), 0);
}

#[test]
fn rpcl_lossless_policy_reduces_base_resolution_to_64_or_less() {
    for (tile_size, expected_levels) in [(512usize, 3u8), (1024, 4), (2048, 5)] {
        let pixels = vec![0; tile_size * tile_size];
        let samples =
            J2kLosslessSamples::new(&pixels, tile_size as u32, tile_size as u32, 1, 8, false)
                .unwrap();

        assert_eq!(
            j2k_lossless_decomposition_levels_for_progression(samples, J2kProgressionOrder::Rpcl),
            expected_levels
        );
    }
}

#[test]
fn cpu_lossless_round_trips_gray8() {
    let pixels: Vec<u8> = (0..35).map(|v| (v * 7) as u8).collect();
    let samples = J2kLosslessSamples::new(&pixels, 7, 5, 1, 8, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::CpuOnly,
            ..J2kLosslessEncodeOptions::default()
        },
    )
    .expect("lossless encode");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(encoded.width, 7);
    assert_eq!(encoded.height, 5);
    assert_eq!(encoded.components, 1);
    assert_eq!(encoded.bit_depth, 8);
    assert!(encoded.codestream.starts_with(&[0xFF, 0x4F]));

    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 7);
    assert_eq!(decoded.height, 5);
    assert_eq!(decoded.num_components, 1);
    assert_eq!(decoded.bit_depth, 8);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn auto_lossless_round_trips_rgb16_odd_dimensions() {
    let mut pixels = Vec::new();
    for y in 0..3u16 {
        for x in 0..5u16 {
            for c in 0..3u16 {
                pixels.extend_from_slice(&(x * 101 + y * 307 + c * 997).to_le_bytes());
            }
        }
    }
    let samples = J2kLosslessSamples::new(&pixels, 5, 3, 3, 16, false).unwrap();

    let encoded = encode_j2k_lossless(samples, &J2kLosslessEncodeOptions::default())
        .expect("auto lossless encode");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(encoded.components, 3);
    assert_eq!(encoded.bit_depth, 16);

    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 5);
    assert_eq!(decoded.height, 3);
    assert_eq!(decoded.num_components, 3);
    assert_eq!(decoded.bit_depth, 16);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_round_trips_rgb8_high_variance_512() {
    let mut pixels = Vec::with_capacity(512 * 512 * 3);
    let mut state = 0x5eed_1234_u32;
    for _ in 0..512 * 512 * 3 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        pixels.push((state >> 24) as u8);
    }
    let samples = J2kLosslessSamples::new(&pixels, 512, 512, 3, 8, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::CpuOnly,
            ..J2kLosslessEncodeOptions::default()
        },
    )
    .expect("cpu lossless encode");

    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_round_trips_rgb8_constant_gray_512() {
    let pixels = vec![243u8; 512 * 512 * 3];
    let samples = J2kLosslessSamples::new(&pixels, 512, 512, 3, 8, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::CpuOnly,
            ..J2kLosslessEncodeOptions::default()
        },
    )
    .expect("cpu lossless encode");

    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_round_trips_rgb8_low_variance_slide_like_512() {
    let mut pixels = Vec::with_capacity(512 * 512 * 3);
    for y in 0..512u32 {
        for x in 0..512u32 {
            let base = 240u8.wrapping_add(((x / 19 + y / 23 + x * y / 4096) & 7) as u8);
            pixels.push(base);
            pixels.push(base.saturating_sub(2));
            pixels.push(base.saturating_add(2));
        }
    }
    let samples = J2kLosslessSamples::new(&pixels, 512, 512, 3, 8, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::CpuOnly,
            ..J2kLosslessEncodeOptions::default()
        },
    )
    .expect("cpu lossless encode");

    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_round_trips_rgb8_variable_chroma_512() {
    let mut pixels = Vec::with_capacity(512 * 512 * 3);
    for y in 0..512i32 {
        for x in 0..512i32 {
            let base = 238 + ((x / 17 + y / 29 + x * y / 8192) & 15);
            let red_delta = ((x * 3 + y * 5) & 31) - 15;
            let blue_delta = ((x * 7 - y * 3) & 31) - 15;
            pixels.push((base + red_delta).clamp(0, 255) as u8);
            pixels.push(base.clamp(0, 255) as u8);
            pixels.push((base + blue_delta).clamp(0, 255) as u8);
        }
    }
    let samples = J2kLosslessSamples::new(&pixels, 512, 512, 3, 8, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::CpuOnly,
            ..J2kLosslessEncodeOptions::default()
        },
    )
    .expect("cpu lossless encode");

    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.data, pixels);
}

#[test]
#[ignore = "requires SIGNINUM_J2K_APERIO_TILE_FIXTURE"]
fn cpu_lossless_round_trips_aperio_jp2k_problem_tile_512() {
    let Some(path) = std::env::var_os("SIGNINUM_J2K_APERIO_TILE_FIXTURE").map(PathBuf::from) else {
        return;
    };
    let pixels = std::fs::read(&path).expect("problem tile fixture");
    assert_eq!(pixels.len(), 512 * 512 * 3);
    let samples = J2kLosslessSamples::new(&pixels, 512, 512, 3, 8, false).unwrap();

    let codestream = signinum_j2k_native::encode(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &signinum_j2k_native::EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            ..signinum_j2k_native::EncodeOptions::default()
        },
    )
    .expect("cpu lossless encode");
    let decoded = decode_native(&codestream);
    let mismatch = decoded
        .data
        .iter()
        .zip(pixels.iter())
        .position(|(actual, expected)| actual != expected);
    assert!(
        mismatch.is_none(),
        "first mismatch at byte {:?}: expected {:?}, actual {:?}",
        mismatch,
        mismatch.map(|idx| pixels[idx]),
        mismatch.map(|idx| decoded.data[idx])
    );
}

#[test]
fn cpu_lossless_round_trips_rgb8_seed_130_64() {
    let mut pixels = Vec::with_capacity(64 * 64 * 3);
    let mut state = 0x0082_u32 ^ 0x9e37_79b9;
    for _ in 0..64 * 64 * 3 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        pixels.push((state >> 24) as u8);
    }
    let samples = J2kLosslessSamples::new(&pixels, 64, 64, 3, 8, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::CpuOnly,
            ..J2kLosslessEncodeOptions::default()
        },
    )
    .expect("cpu lossless encode");

    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_round_trips_gray8_seed_104_64() {
    let mut pixels = Vec::with_capacity(64 * 64);
    let mut state = 0x0068_u32 ^ 0x517c_c1b7;
    for _ in 0..64 * 64 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        pixels.push((state >> 24) as u8);
    }
    let samples = J2kLosslessSamples::new(&pixels, 64, 64, 1, 8, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::CpuOnly,
            ..J2kLosslessEncodeOptions::default()
        },
    )
    .expect("cpu lossless encode");

    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn prefer_device_falls_back_to_validated_cpu_until_device_encode_is_complete() {
    let pixels: Vec<u8> = (0..27).map(|v| (v * 3) as u8).collect();
    let samples = J2kLosslessSamples::new(&pixels, 3, 3, 3, 8, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::PreferDevice,
            ..J2kLosslessEncodeOptions::default()
        },
    )
    .expect("prefer-device lossless encode");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn require_device_errors_clearly_when_encode_backend_is_unavailable() {
    let pixels = vec![0u8; 4 * 4];
    let samples = J2kLosslessSamples::new(&pixels, 4, 4, 1, 8, false).unwrap();

    let err = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::RequireDevice,
            ..J2kLosslessEncodeOptions::default()
        },
    )
    .unwrap_err();

    assert!(err.is_unsupported());
    assert!(err.to_string().contains("device"));
    assert!(err.to_string().contains("encode"));
}

#[test]
fn accelerator_facade_prefer_device_falls_back_when_no_stage_dispatches() {
    #[derive(Default)]
    struct NoDispatchAccelerator;

    impl J2kEncodeStageAccelerator for NoDispatchAccelerator {}

    let pixels: Vec<u8> = (0..64).map(|value| (value * 5) as u8).collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).unwrap();
    let mut accelerator = NoDispatchAccelerator;

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::PreferDevice,
            ..J2kLosslessEncodeOptions::default()
        },
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("prefer-device encode should fall back to CPU without dispatch");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(decode_native(&encoded.codestream).data, pixels);
}

#[test]
fn accelerator_facade_require_device_errors_when_no_stage_dispatches() {
    #[derive(Default)]
    struct NoDispatchAccelerator;

    impl J2kEncodeStageAccelerator for NoDispatchAccelerator {}

    let pixels = vec![0u8; 8 * 8];
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).unwrap();
    let mut accelerator = NoDispatchAccelerator;

    let err = encode_j2k_lossless_with_accelerator(
        samples,
        &J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::RequireDevice,
            ..J2kLosslessEncodeOptions::default()
        },
        BackendKind::Metal,
        &mut accelerator,
    )
    .unwrap_err();

    assert!(err.is_unsupported());
    assert!(err.to_string().contains("did not dispatch"));
}

#[test]
fn accelerator_facade_require_device_errors_when_any_required_stage_is_missing() {
    #[derive(Default)]
    struct PacketizationDispatchAccelerator {
        packetization_dispatches: usize,
    }

    impl J2kEncodeStageAccelerator for PacketizationDispatchAccelerator {
        fn dispatch_report(&self) -> J2kEncodeDispatchReport {
            J2kEncodeDispatchReport {
                packetization: self.packetization_dispatches,
                ..J2kEncodeDispatchReport::default()
            }
        }

        fn encode_packetization(
            &mut self,
            _job: J2kPacketizationEncodeJob<'_>,
        ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
            self.packetization_dispatches = self.packetization_dispatches.saturating_add(1);
            Ok(None)
        }
    }

    let pixels: Vec<u8> = (0..64).map(|value| (value * 7) as u8).collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).unwrap();
    let mut accelerator = PacketizationDispatchAccelerator::default();

    let err = encode_j2k_lossless_with_accelerator(
        samples,
        &J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::RequireDevice,
            ..J2kLosslessEncodeOptions::default()
        },
        BackendKind::Metal,
        &mut accelerator,
    )
    .unwrap_err();

    assert!(err.is_unsupported());
    assert!(err.to_string().contains("tier1_code_block"));
}

#[test]
fn accelerator_facade_reports_requested_backend_after_all_required_stages_dispatch() {
    #[derive(Default)]
    struct FullClassicAccelerator {
        tier1_code_block_dispatches: usize,
        packetization_dispatches: usize,
    }

    impl J2kEncodeStageAccelerator for FullClassicAccelerator {
        fn dispatch_report(&self) -> J2kEncodeDispatchReport {
            J2kEncodeDispatchReport {
                tier1_code_block: self.tier1_code_block_dispatches,
                packetization: self.packetization_dispatches,
                ..J2kEncodeDispatchReport::default()
            }
        }

        fn encode_tier1_code_block(
            &mut self,
            job: signinum_j2k::J2kTier1CodeBlockEncodeJob<'_>,
        ) -> core::result::Result<Option<signinum_j2k::EncodedJ2kCodeBlock>, &'static str> {
            self.tier1_code_block_dispatches = self.tier1_code_block_dispatches.saturating_add(1);
            signinum_j2k_native::encode_j2k_code_block_scalar_with_style(
                job.coefficients,
                job.width,
                job.height,
                job.sub_band_type,
                job.total_bitplanes,
                job.style,
            )
            .map(Some)
        }

        fn encode_packetization(
            &mut self,
            _job: J2kPacketizationEncodeJob<'_>,
        ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
            self.packetization_dispatches = self.packetization_dispatches.saturating_add(1);
            Ok(None)
        }
    }

    let pixels: Vec<u8> = (0..64).map(|value| (value * 7) as u8).collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).unwrap();
    let mut accelerator = FullClassicAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::PreferDevice,
            ..J2kLosslessEncodeOptions::default()
        },
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("all required device stages should produce encoded codestream");

    assert_eq!(encoded.backend, BackendKind::Metal);
    assert_eq!(decode_native(&encoded.codestream).data, pixels);
}

#[test]
fn accelerator_facade_ht_require_device_checks_ht_code_block_stage() {
    #[derive(Default)]
    struct FullHtAccelerator {
        ht_code_block_dispatches: usize,
        packetization_dispatches: usize,
    }

    impl J2kEncodeStageAccelerator for FullHtAccelerator {
        fn dispatch_report(&self) -> J2kEncodeDispatchReport {
            J2kEncodeDispatchReport {
                ht_code_block: self.ht_code_block_dispatches,
                packetization: self.packetization_dispatches,
                ..J2kEncodeDispatchReport::default()
            }
        }

        fn encode_ht_code_block(
            &mut self,
            job: J2kHtCodeBlockEncodeJob<'_>,
        ) -> core::result::Result<Option<EncodedHtJ2kCodeBlock>, &'static str> {
            self.ht_code_block_dispatches = self.ht_code_block_dispatches.saturating_add(1);
            signinum_j2k_native::encode_ht_code_block_scalar(
                job.coefficients,
                job.width,
                job.height,
                job.total_bitplanes,
            )
            .map(|encoded| {
                Some(EncodedHtJ2kCodeBlock {
                    data: encoded.data,
                    num_coding_passes: encoded.num_coding_passes,
                    num_zero_bitplanes: encoded.num_zero_bitplanes,
                })
            })
        }

        fn encode_packetization(
            &mut self,
            _job: J2kPacketizationEncodeJob<'_>,
        ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
            self.packetization_dispatches = self.packetization_dispatches.saturating_add(1);
            Ok(None)
        }
    }

    let pixels: Vec<u8> = (0..64).map(|value| (value * 13) as u8).collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).unwrap();
    let mut accelerator = FullHtAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            ..J2kLosslessEncodeOptions::default()
        },
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("HT required stages should dispatch");

    assert_eq!(encoded.backend, BackendKind::Metal);
    assert_eq!(decode_native(&encoded.codestream).data, pixels);
}

fn marker_offset(codestream: &[u8], marker: u8) -> Option<usize> {
    codestream
        .windows(2)
        .position(|window| window == [0xFF, marker])
}

#[test]
fn sample_descriptor_rejects_short_pixel_buffers() {
    let pixels = vec![0u8; 5];

    let err = J2kLosslessSamples::new(&pixels, 2, 2, 3, 8, false).unwrap_err();

    assert!(err.to_string().contains("pixel data too short"));
}
