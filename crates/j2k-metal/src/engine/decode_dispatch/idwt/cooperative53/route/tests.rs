// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::*;
use crate::{MetalBatchDecodeResult, MetalBatchDecoder};
use j2k::{BatchDecodeOptions, CpuBatchDecoder, CpuBatchSamples, DecodeRequest, EncodedImage};
use j2k_core::{DeviceSurface, Downscale, Rect};

fn inputs(dimension: u32) -> Vec<EncodedImage> {
    (0..4u8)
        .map(|seed| {
            let pixels = (0..dimension * dimension)
                .map(|offset| u8::try_from(offset % 251).unwrap().wrapping_add(seed))
                .collect::<Vec<_>>();
            let bytes = j2k_native::encode_htj2k(
                &pixels,
                dimension,
                dimension,
                1,
                8,
                false,
                &j2k_native::EncodeOptions {
                    reversible: true,
                    num_decomposition_levels: 3,
                    ..j2k_native::EncodeOptions::default()
                },
            )
            .expect("encode distinct lossless input");
            EncodedImage::full(Arc::from(bytes))
        })
        .collect()
}

fn consume(result: &MetalBatchDecodeResult, host: bool) {
    assert!(result.errors().is_empty());
    assert!(result.group_errors().is_empty());
    let bytes = result
        .groups()
        .iter()
        .flat_map(crate::MetalBatchGroup::surfaces)
        .map(|surface| {
            if host {
                surface.as_bytes().expect("completed host pixels").len()
            } else {
                surface.byte_len()
            }
        })
        .sum::<usize>();
    std::hint::black_box(bytes);
}

fn assert_cpu_parity(actual: &MetalBatchDecodeResult, expected: &j2k::CpuBatchDecodeResult) {
    assert!(expected.errors().is_empty());
    consume(actual, true);
    assert_eq!(actual.groups().len(), expected.groups().len());
    for (actual, expected) in actual.groups().iter().zip(expected.groups()) {
        assert_eq!(actual.source_indices(), expected.source_indices());
        let CpuBatchSamples::U8(expected) = expected.samples() else {
            panic!("U8 oracle");
        };
        let actual = actual
            .surfaces()
            .iter()
            .flat_map(|surface| surface.as_bytes().unwrap().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(&actual, expected);
    }
}

#[test]
fn cooperative53_resident_graph_matches_cpu_pixels() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let options = BatchDecodeOptions::default();
    let mut decoder = MetalBatchDecoder::system_default_with_options(options).unwrap();
    let candidate = Rc::new(Cooperative53::new(decoder.backend_session().device()).unwrap());
    for dimension in [31, 128, 257] {
        let encoded = inputs(dimension);
        let roi = Rect {
            x: 3,
            y: 5,
            w: dimension - 7,
            h: dimension - 9,
        };
        for request in [
            DecodeRequest::Full,
            DecodeRequest::Region { roi },
            DecodeRequest::RegionReduced {
                roi,
                scale: Downscale::Half,
            },
        ] {
            let inputs = encoded
                .iter()
                .map(|image| EncodedImage::new(image.bytes.clone(), request))
                .collect::<Vec<_>>();
            let expected = CpuBatchDecoder::new(options)
                .decode(inputs.clone())
                .unwrap();
            let prepared = decoder.prepare(inputs).unwrap();
            assert_eq!(prepared.groups().len(), 1);
            let selected =
                SelectionGuard::new(candidate.clone(), decoder.backend_session().device());
            let actual = decoder.decode_prepared(&prepared).unwrap();
            if request == DecodeRequest::Full {
                assert!(
                    SelectionGuard::dispatches() > 0,
                    "full resident graph must reach cooperative kernels"
                );
            }
            eprintln!(
                "5/3 graph parity size={dimension} request={request:?} cooperative_dispatches={}",
                SelectionGuard::dispatches()
            );
            assert_cpu_parity(&actual, &expected);
            drop(selected);
            // A completed output remains readable after the test override is gone.
            consume(&actual, true);
        }
    }
}

#[test]
#[ignore = "manual paired full decode timing; candidate remains test-only"]
fn cooperative53_end_to_end_timing() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let options = BatchDecodeOptions::default();
    let mut decoder = MetalBatchDecoder::system_default_with_options(options).unwrap();
    let device = j2k_metal_support::system_default_device().unwrap();
    let candidate = Rc::new(Cooperative53::new(&device).unwrap());
    for dimension in [128, 512, 2048] {
        let inputs = inputs(dimension);
        let prepared = decoder.prepare(inputs.clone()).unwrap();
        let expected = CpuBatchDecoder::new(options)
            .decode(inputs.clone())
            .unwrap();
        for cooperative in [false, true] {
            let selected = cooperative.then(|| SelectionGuard::new(candidate.clone(), &device));
            let actual = decoder.decode_prepared(&prepared).unwrap();
            assert_cpu_parity(&actual, &expected);
            if selected.is_some() {
                assert!(
                    SelectionGuard::dispatches() > 0,
                    "parity probe must reach candidate"
                );
            }
        }
        for cold in [false, true] {
            for host in [false, true] {
                let mut run = || {
                    if cold {
                        let mut cold_decoder =
                            MetalBatchDecoder::system_default_with_options(options).unwrap();
                        consume(&cold_decoder.decode_batch(inputs.clone()).unwrap(), host);
                    } else {
                        consume(&decoder.decode_prepared(&prepared).unwrap(), host);
                    }
                };
                // Calibrate on the unchanged route, then use identical inner counts.
                let mut repetitions = 1u32;
                loop {
                    let start = std::time::Instant::now();
                    for _ in 0..repetitions {
                        run();
                    }
                    if start.elapsed() >= std::time::Duration::from_millis(20) || repetitions >= 64
                    {
                        break;
                    }
                    repetitions *= 2;
                }
                let mut measurements = [Vec::new(), Vec::new()];
                for iteration in 0..52 {
                    for index in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                        let selected =
                            (index == 1).then(|| SelectionGuard::new(candidate.clone(), &device));
                        let start = std::time::Instant::now();
                        for _ in 0..repetitions {
                            run();
                        }
                        let elapsed = start.elapsed().as_secs_f64() / f64::from(repetitions);
                        if selected.is_some() {
                            assert!(
                                SelectionGuard::dispatches() > 0,
                                "timed decode must reach candidate"
                            );
                        }
                        if iteration >= 2 {
                            measurements[index].push(elapsed);
                        }
                    }
                }
                for (route, samples) in ["old", "cooperative"].into_iter().zip(measurements) {
                    eprintln!("5/3 end_to_end size={dimension} batch=4 cold={cold} host={host} repeats={repetitions} route={route} seconds={samples:?}");
                }
            }
        }
    }
}
