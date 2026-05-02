// SPDX-License-Identifier: Apache-2.0

use signinum_core::{BackendKind, CodecError};
use signinum_j2k::{
    encode_j2k_lossless, EncodeBackendPreference, J2kLosslessEncodeOptions, J2kLosslessSamples,
    J2kProgressionOrder, ReversibleTransform,
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
fn sample_descriptor_rejects_short_pixel_buffers() {
    let pixels = vec![0u8; 5];

    let err = J2kLosslessSamples::new(&pixels, 2, 2, 3, 8, false).unwrap_err();

    assert!(err.to_string().contains("pixel data too short"));
}
