// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};
use signinum_core::{BackendRequest, ImageDecodeDevice, PixelFormat};
use signinum_j2k::J2kDecoder as CpuDecoder;
use signinum_j2k_metal::J2kDecoder as MetalDecoder;
use signinum_j2k_native::{encode, EncodeOptions};

fn fixture() -> Vec<u8> {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 2, 3, 8, false, &options).expect("encode")
}

fn bench_device_upload(c: &mut Criterion) {
    let bytes = fixture();
    let mut group = c.benchmark_group("j2k_metal_device");

    group.bench_function("cpu_decode_rgb8", |b| {
        let mut decoder = CpuDecoder::new(&bytes).expect("cpu decoder");
        b.iter(|| {
            let mut out = [0u8; 12];
            decoder
                .decode_into(&mut out, 6, PixelFormat::Rgb8)
                .expect("cpu decode")
        });
    });

    if metal_decode_available() {
        group.bench_function("metal_surface_rgb8", |b| {
            let mut decoder = MetalDecoder::new(&bytes).expect("metal decoder");
            b.iter(|| {
                decoder
                    .decode_to_device(PixelFormat::Rgb8, BackendRequest::Metal)
                    .expect("device decode")
            });
        });
    }

    group.finish();
}

fn metal_decode_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        metal::Device::system_default().is_some()
    }
    #[cfg(not(target_os = "macos"))]
    {
        assert!(
            std::env::var_os("SIGNINUM_REQUIRE_METAL_BENCH").is_none(),
            "SIGNINUM_REQUIRE_METAL_BENCH is set but this is not a Metal host"
        );
        false
    }
}

criterion_group!(benches, bench_device_upload);
criterion_main!(benches);
