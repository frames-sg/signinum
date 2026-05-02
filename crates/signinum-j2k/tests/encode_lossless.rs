// SPDX-License-Identifier: Apache-2.0

use signinum_core::{BackendKind, CodecError};
use signinum_j2k::{
    encode_j2k_lossless, encode_j2k_lossless_with_accelerator, EncodeBackendPreference,
    J2kEncodeDispatchReport, J2kEncodeStageAccelerator, J2kLosslessEncodeOptions,
    J2kLosslessSamples, J2kPacketizationEncodeJob, J2kProgressionOrder, ReversibleTransform,
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
    assert_eq!(options.progression, J2kProgressionOrder::Lrcp);
    assert_eq!(options.reversible_transform, ReversibleTransform::Rct53);
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
fn accelerator_facade_reports_requested_backend_after_stage_dispatch() {
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
            _job: J2kPacketizationEncodeJob,
        ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
            self.packetization_dispatches = self.packetization_dispatches.saturating_add(1);
            Ok(None)
        }
    }

    let pixels: Vec<u8> = (0..64).map(|value| (value * 7) as u8).collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).unwrap();
    let mut accelerator = PacketizationDispatchAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::PreferDevice,
            ..J2kLosslessEncodeOptions::default()
        },
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("reported device dispatch should produce encoded codestream");

    assert_eq!(encoded.backend, BackendKind::Metal);
    assert_eq!(decode_native(&encoded.codestream).data, pixels);
}

#[test]
fn sample_descriptor_rejects_short_pixel_buffers() {
    let pixels = vec![0u8; 5];

    let err = J2kLosslessSamples::new(&pixels, 2, 2, 3, 8, false).unwrap_err();

    assert!(err.to_string().contains("pixel data too short"));
}
