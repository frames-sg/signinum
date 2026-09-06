// SPDX-License-Identifier: MIT OR Apache-2.0

//! Test-only comparisons of GPU scheduling with the same inputs and exact output.

#![cfg(test)]

use crate::{Codec, Decoder, MetalBackendSession, MetalBatchTextureOutput, MetalTextureTile};
use jpeg_encoder::{ColorType, Encoder, SamplingFactor};
use std::{cell::Cell, time::Instant};

thread_local! {
    static SETTINGS: Cell<(Option<u64>, Option<bool>)> = const { Cell::new((None, None)) };
}

pub(super) fn group_width() -> Option<u64> {
    SETTINGS.get().0
}
pub(super) fn component_planes() -> Option<bool> {
    SETTINGS.get().1
}

fn with_settings<T>(width: u64, planes: bool, run: impl FnOnce() -> T) -> T {
    struct Restore((Option<u64>, Option<bool>));
    impl Drop for Restore {
        fn drop(&mut self) {
            SETTINGS.set(self.0);
        }
    }
    let _restore = Restore(SETTINGS.replace((Some(width), Some(planes))));
    run()
}

fn input(
    width: u16,
    height: u16,
    sampling: SamplingFactor,
    restart: Option<u16>,
    variant: u8,
) -> Vec<u8> {
    let mut rgb = j2k_test_support::gpu_bench_rgb8(u32::from(width), u32::from(height));
    for pixel in rgb.chunks_exact_mut(3) {
        pixel[0] = pixel[0].wrapping_add(variant);
    }
    let mut bytes = Vec::new();
    let mut encoder = Encoder::new(&mut bytes, 90);
    encoder.set_sampling_factor(sampling);
    if let Some(interval) = restart {
        encoder.set_restart_interval(interval);
    }
    encoder
        .encode(&rgb, width, height, ColorType::Rgb)
        .expect("JPEG input");
    bytes
}

fn decode(
    decoders: &[&Decoder<'_>],
    output: &mut MetalBatchTextureOutput,
    session: &MetalBackendSession,
) -> Vec<MetalTextureTile> {
    Codec::decode_rgb8_decoder_batch_into_resizable_metal_textures_with_session(
        decoders, output, session,
    )
    .expect("batch decode")
    .into_iter()
    .map(|tile| tile.expect("valid tile"))
    .collect()
}

fn check_tiles(tiles: &[MetalTextureTile], expected: &[Vec<u8>], session: &MetalBackendSession) {
    for (tile, expected) in tiles.iter().zip(expected) {
        assert_eq!(
            crate::tests::download_rgba8_texture(
                session,
                tile.texture_trusted(),
                tile.dimensions()
            ),
            *expected
        );
    }
    assert_eq!(tiles.len(), expected.len());
}

fn expected(inputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    inputs
        .iter()
        .map(|bytes| {
            let (rgb, _) = j2k_jpeg::Decoder::new(bytes)
                .expect("CPU decoder")
                .decode_request(j2k_jpeg::DecodeRequest::full(j2k_core::PixelFormat::Rgb8))
                .expect("CPU decode");
            crate::tests::rgb_to_rgba_opaque(&rgb)
        })
        .collect()
}

#[test]
fn texture_scheduling_variants_preserve_distinct_edge_tiles() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let session = MetalBackendSession::system_default().expect("session");
    for (sampling, restart) in [
        (SamplingFactor::F_2_2, None),
        (SamplingFactor::F_2_2, Some(2)),
        (SamplingFactor::F_2_1, None),
        (SamplingFactor::F_1_1, None),
    ] {
        let inputs: Vec<_> = (0..2)
            .map(|v| input(65, 49, sampling, restart, v))
            .collect();
        let expected = expected(&inputs);
        let decoders: Vec<_> = inputs
            .iter()
            .map(|bytes| Decoder::new(bytes).expect("decoder"))
            .collect();
        let refs: Vec<_> = decoders.iter().collect();
        let mut output =
            MetalBatchTextureOutput::new_rgba8_tiles(&session, (65, 49), 2).expect("output");
        for planes in [false, true] {
            for width in [32, 64, 128, 256] {
                let tiles = with_settings(width, planes, || decode(&refs, &mut output, &session));
                check_tiles(&tiles, &expected, &session);
            }
        }
    }
}

#[test]
fn concurrent_texture_batches_keep_outputs_independent() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let session = MetalBackendSession::system_default().expect("session");
    let start = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        for variant in [0, 91] {
            let session = &session;
            let start = &start;
            scope.spawn(move || {
                let inputs = [input(129, 97, SamplingFactor::F_2_2, Some(2), variant)];
                let expected = expected(&inputs);
                let decoder = Decoder::new(&inputs[0]).expect("decoder");
                let mut output = MetalBatchTextureOutput::new_rgba8_tiles(session, (129, 97), 1)
                    .expect("output");
                start.wait();
                for _ in 0..4 {
                    let tiles = decode(&[&decoder], &mut output, session);
                    check_tiles(&tiles, &expected, session);
                }
            });
        }
    });
}

#[test]
#[ignore = "explicit GPU timing experiment; run serially with --nocapture"]
fn texture_scheduling_experiment() {
    assert!(j2k_test_support::metal_runtime_gate(module_path!()));
    let session = MetalBackendSession::system_default().expect("session");
    let cases = if std::env::var_os("J2K_JPEG_TEXTURE_SMALL_BATCHES").is_some() {
        vec![
            (16, 1, SamplingFactor::F_2_2, None),
            (64, 1, SamplingFactor::F_2_2, None),
            (256, 1, SamplingFactor::F_2_2, None),
            (256, 2, SamplingFactor::F_2_2, None),
            (256, 4, SamplingFactor::F_2_2, None),
            (256, 16, SamplingFactor::F_2_1, None),
            (256, 1, SamplingFactor::F_2_1, None),
            (512, 16, SamplingFactor::F_2_1, None),
        ]
    } else {
        vec![
            (256, 16, SamplingFactor::F_2_2, None),
            (256, 16, SamplingFactor::F_2_2, Some(2)),
            (512, 16, SamplingFactor::F_2_2, None),
            (512, 16, SamplingFactor::F_2_2, Some(4)),
        ]
    };
    for (dim, batch_size, sampling, restart) in cases {
        let inputs: Vec<_> = (0..batch_size)
            .map(|v| input(dim, dim, sampling, restart, v))
            .collect();
        let expected = expected(&inputs);
        let decoders: Vec<_> = inputs
            .iter()
            .map(|bytes| Decoder::new(bytes).expect("decoder"))
            .collect();
        let refs: Vec<_> = decoders.iter().collect();
        let dims = (u32::from(dim), u32::from(dim));
        let mut output =
            MetalBatchTextureOutput::new_rgba8_tiles(&session, dims, refs.len()).expect("output");
        for planes in [false, true] {
            for width in [256, 128, 64, 32] {
                with_settings(width, planes, || {
                    check_tiles(&decode(&refs, &mut output, &session), &expected, &session);
                    for _ in 0..5 {
                        std::hint::black_box(decode(&refs, &mut output, &session));
                    }
                    let mut samples = Vec::new();
                    for _ in 0..15 {
                        let start = Instant::now();
                        std::hint::black_box(decode(&refs, &mut output, &session));
                        samples.push(start.elapsed().as_micros());
                    }
                    samples.sort_unstable();
                    eprintln!("jpeg_texture_schedule dim={dim} sampling={sampling:?} restart={restart:?} tiles={batch_size} planes={planes} width={width} median_us={} min_us={} max_us={}", samples[7], samples[0], samples[14]);
                });
            }
        }
    }
}
