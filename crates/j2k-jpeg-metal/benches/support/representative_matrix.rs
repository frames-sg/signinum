// SPDX-License-Identifier: MIT OR Apache-2.0

//! Quality and source-content controls with retained plans and resident output.
//! Source labels describe generated pixels, not measured coefficient density.

use super::{
    assert_metal_surface_pixels, fast_packet_family_label, fast_packet_plan, native_request_pixels,
};
use criterion::{Criterion, Throughput};
use j2k_core::PixelFormat;
use j2k_jpeg::DecodeRequest;
use j2k_jpeg_metal::{Decoder, MetalBackendSession};
use jpeg_encoder::{ColorType, Encoder, SamplingFactor};
use std::hint::black_box;

pub(super) fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("jpeg_metal_representative_matrix");
    for side in [256_u16, 1024] {
        group.throughput(Throughput::Elements(u64::from(side) * u64::from(side)));
        for quality in [35, 95] {
            for textured in [false, true] {
                let source = if textured {
                    "textured_source"
                } else {
                    "flat_source"
                };
                let rgb = if textured {
                    j2k_test_support::gpu_bench_rgb8(u32::from(side), u32::from(side))
                } else {
                    vec![96_u8; usize::from(side) * usize::from(side) * 3]
                };
                let mut bytes = Vec::new();
                let mut encoder = Encoder::new(&mut bytes, quality);
                encoder.set_sampling_factor(SamplingFactor::F_2_2);
                encoder.set_restart_interval(2);
                encoder
                    .encode(&rgb, side, side, ColorType::Rgb)
                    .expect("encode representative JPEG");
                assert_eq!(
                    fast_packet_family_label(fast_packet_plan(&bytes).expect("fast packet")),
                    "fast420",
                );
                let expected =
                    native_request_pixels(&bytes, DecodeRequest::full(PixelFormat::Rgb8));
                let session = MetalBackendSession::system_default().expect("Metal session");
                let mut decoder = Decoder::new(&bytes).expect("retained decoder");
                let probe = decoder
                    .decode_to_device_with_session(PixelFormat::Rgb8, &session)
                    .expect("representative resident probe");
                assert_metal_surface_pixels(&probe, &expected);
                drop(probe);
                group.bench_function(
                    format!("420_r2/{side}x{side}/quality{quality}/{source}/prepared/warm_session/resident/rgb8/single"),
                    |b| b.iter(|| {
                        black_box(decoder.decode_to_device_with_session(PixelFormat::Rgb8, &session)
                            .expect("representative resident decode"));
                    }),
                );
            }
        }
    }
    group.finish();
}
