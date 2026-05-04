// SPDX-License-Identifier: Apache-2.0

use signinum::{
    j2k::{encode_j2k_lossless, J2kLosslessEncodeOptions, J2kLosslessSamples},
    tilecodec::UncompressedCodec,
    BackendKind, BackendRequest, TileDecompress,
};

#[test]
fn facade_default_features_enable_auto_device_adapters() {
    let manifest = std::fs::read_to_string(env!("CARGO_MANIFEST_DIR").to_owned() + "/Cargo.toml")
        .expect("read facade manifest");

    assert!(
        manifest.contains("default = [\"metal\"]"),
        "signinum should compile the portable Metal adapter by default so Auto is not documented or packaged as CPU-only"
    );
}

#[test]
fn facade_runtime_backend_default_is_auto() {
    assert_eq!(BackendRequest::default(), BackendRequest::Auto);
    assert_eq!(
        J2kLosslessEncodeOptions::default().backend,
        signinum::EncodeBackendPreference::Auto
    );
}

#[test]
fn facade_auto_j2k_lossless_encode_uses_device_when_available() {
    let pixels: Vec<u8> = (0..4 * 4 * 3)
        .map(|value| u8::try_from((value * 11) & 0xFF).expect("masked sample fits"))
        .collect();
    let samples = J2kLosslessSamples::new(&pixels, 4, 4, 3, 8, false).expect("valid samples");

    let encoded =
        encode_j2k_lossless(samples, &J2kLosslessEncodeOptions::default()).expect("encode");

    #[cfg(all(feature = "metal", target_os = "macos"))]
    match encoded.backend {
        BackendKind::Metal => {}
        BackendKind::Cpu => {
            let samples =
                J2kLosslessSamples::new(&pixels, 4, 4, 3, 8, false).expect("valid samples");
            let required = encode_j2k_lossless(
                samples,
                &J2kLosslessEncodeOptions {
                    backend: signinum::EncodeBackendPreference::RequireDevice,
                    ..J2kLosslessEncodeOptions::default()
                },
            );
            assert!(
                required.is_err(),
                "Auto fell back to CPU even though RequireDevice succeeded"
            );
        }
        BackendKind::Cuda => panic!("unexpected facade backend: Cuda"),
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert!(encoded.codestream.starts_with(&[0xFF, 0x4F]));
}

#[test]
fn facade_exports_tilecodec_contracts() {
    let input = [1, 2, 3, 4];
    let mut output = [0; 4];
    let mut pool = <UncompressedCodec as TileDecompress>::Pool::default();

    let written = UncompressedCodec::decompress_into(&mut pool, &input, &mut output)
        .expect("uncompressed tile copy");

    assert_eq!(written, input.len());
    assert_eq!(output, input);
}
