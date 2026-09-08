// SPDX-License-Identifier: MIT OR Apache-2.0

//! Distinct compatible inputs, with preparation and host-copy costs separated.
//! Every row reuses its session and output allocation. `cold_plan` reparses raw
//! JPEGs per iteration; `prepared` retains decoders and their fast-packet plans.
//! `single_loop` uses the same buffer API with one item per completed call.

use super::{
    assert_metal_surface_pixels, fast_packet_family_label, fast_packet_plan,
    generated_rgb_jpeg_variant, native_request_pixels,
};
use criterion::{Criterion, Throughput};
use j2k_core::PixelFormat;
use j2k_jpeg::DecodeRequest;
use j2k_jpeg_metal::{
    Codec, Decoder, MetalBackendSession, MetalBatchOutputBuffer, MetalBufferBatchTarget,
    Rgb8MetalBatchOp, Rgb8MetalBatchRequest, Rgb8MetalBatchSource,
};
use jpeg_encoder::SamplingFactor;
use std::{hint::black_box, sync::Barrier, time::Instant};

const BATCH_SIZE: usize = 4;

pub(super) fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("jpeg_metal_distinct_batch");
    for side in [128_u16, 512] {
        let (inputs, expected) = distinct_inputs(side);
        let bytes = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let decoders = inputs
            .iter()
            .map(|bytes| Decoder::new(bytes).expect("prepared decoder"))
            .collect::<Vec<_>>();
        let decoder_refs = decoders.iter().collect::<Vec<_>>();
        let reversed_bytes = bytes.iter().copied().rev().collect::<Vec<_>>();
        let reversed_decoders = decoder_refs.iter().copied().rev().collect::<Vec<_>>();
        bench_two_streams(&mut group, side, &decoder_refs, &expected);
        group.throughput(Throughput::Elements(BATCH_SIZE as u64));
        for prepared in [false, true] {
            for host_copy in [false, true] {
                for batched in [false, true] {
                    let preparation = if prepared { "prepared" } else { "cold_plan" };
                    let destination = if host_copy { "host_copy" } else { "resident" };
                    let execution = if batched { "batch4" } else { "single_loop4" };
                    let session = MetalBackendSession::system_default().expect("Metal session");
                    let count = if batched { BATCH_SIZE } else { 1 };
                    let output = MetalBatchOutputBuffer::new_rgb8_tiles(
                        &session,
                        (u32::from(side), u32::from(side)),
                        count,
                    )
                    .expect("reused batch output");
                    let mut host = expected
                        .iter()
                        .map(|pixels| vec![0; pixels.len()])
                        .collect::<Vec<_>>();
                    let mut run = |verify: bool, reversed: bool| {
                        let bytes = if reversed { &reversed_bytes } else { &bytes };
                        let decoder_refs = if reversed {
                            &reversed_decoders
                        } else {
                            &decoder_refs
                        };
                        for start in (0..BATCH_SIZE).step_by(count) {
                            let end = start + count;
                            let source = if prepared {
                                Rgb8MetalBatchSource::Decoders(&decoder_refs[start..end])
                            } else {
                                Rgb8MetalBatchSource::Bytes(&bytes[start..end])
                            };
                            let surfaces = Codec::decode_rgb8_batch_into_buffer_with_session(
                                Rgb8MetalBatchRequest {
                                    source,
                                    op: Rgb8MetalBatchOp::Full,
                                },
                                MetalBufferBatchTarget::Reusable(&output),
                                &session,
                            )
                            .expect("distinct batch decode");
                            assert_eq!(surfaces.len(), count);
                            for (index, surface) in surfaces.into_iter().enumerate() {
                                let surface = surface.expect("distinct item decode");
                                if verify {
                                    let source_index = if reversed {
                                        BATCH_SIZE - 1 - start - index
                                    } else {
                                        start + index
                                    };
                                    assert_metal_surface_pixels(&surface, &expected[source_index]);
                                }
                                if host_copy {
                                    let pixels = surface.as_bytes().expect("host pixels");
                                    host[start + index].copy_from_slice(pixels.as_ref());
                                    black_box(&host[start + index]);
                                }
                                black_box(surface);
                            }
                        }
                    };
                    run(true, false);
                    run(true, true);
                    run(true, false);
                    group.bench_function(
                        format!("420_r2/{side}x{side}/distinct4/warm_session_reused_output/{preparation}/{destination}/{execution}"),
                        |b| b.iter(|| run(false, false)),
                    );
                }
            }
        }
    }
    group.finish();
}

fn distinct_inputs(side: u16) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let inputs = (0..BATCH_SIZE)
        .map(|variant| {
            generated_rgb_jpeg_variant(
                side,
                side,
                SamplingFactor::F_2_2,
                Some(2),
                u8::try_from(variant).expect("small batch variant"),
            )
        })
        .collect::<Vec<_>>();
    let expected = inputs
        .iter()
        .map(|bytes| {
            assert_eq!(
                fast_packet_family_label(fast_packet_plan(bytes).expect("fast packet")),
                "fast420"
            );
            native_request_pixels(bytes, DecodeRequest::full(PixelFormat::Rgb8))
        })
        .collect::<Vec<_>>();
    for first in 0..BATCH_SIZE {
        for second in first + 1..BATCH_SIZE {
            assert_ne!(inputs[first], inputs[second]);
            assert_ne!(expected[first], expected[second]);
        }
    }
    (inputs, expected)
}

// Each sample measures two streams of completed resident batches. Worker creation
// is excluded; concurrent timing includes the start barrier and worker joins.
// This measures throughput, not individual batch latency or proof of GPU overlap.
fn bench_two_streams(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    side: u16,
    decoders: &[&Decoder<'_>],
    expected: &[Vec<u8>],
) {
    group.throughput(Throughput::Elements((2 * BATCH_SIZE) as u64));
    for concurrent in [false, true] {
        let session = MetalBackendSession::system_default().expect("two-stream Metal session");
        let outputs = std::array::from_fn::<_, 2, _>(|_| {
            MetalBatchOutputBuffer::new_rgb8_tiles(
                &session,
                (u32::from(side), u32::from(side)),
                BATCH_SIZE,
            )
            .expect("independent stream output")
        });
        let decode = |output: &MetalBatchOutputBuffer, verify: bool| {
            let surfaces = Codec::decode_rgb8_batch_into_buffer_with_session(
                Rgb8MetalBatchRequest {
                    source: Rgb8MetalBatchSource::Decoders(decoders),
                    op: Rgb8MetalBatchOp::Full,
                },
                MetalBufferBatchTarget::Reusable(output),
                &session,
            )
            .expect("two-stream resident decode");
            assert_eq!(surfaces.len(), BATCH_SIZE);
            for (surface, expected) in surfaces.into_iter().zip(expected) {
                let surface = surface.expect("two-stream item");
                if verify {
                    assert_metal_surface_pixels(&surface, expected);
                }
                black_box(surface);
            }
        };
        // Exercise both destination owners concurrently before measuring, too.
        std::thread::scope(|scope| {
            let first = scope.spawn(|| decode(&outputs[0], true));
            let second = scope.spawn(|| decode(&outputs[1], true));
            first.join().expect("first probe worker");
            second.join().expect("second probe worker");
        });
        let execution = if concurrent {
            "concurrent_two"
        } else {
            "sequential_two"
        };
        group.bench_function(
            format!("420_r2/{side}x{side}/distinct4/warm_session_reused_output/prepared/resident/batch4/{execution}"),
            |b| b.iter_custom(|iters| {
                if concurrent {
                    let ready = Barrier::new(3);
                    let start = Barrier::new(3);
                    std::thread::scope(|scope| {
                        let workers = outputs.each_ref().map(|output| {
                            let ready = &ready;
                            let start = &start;
                            let decode = &decode;
                            scope.spawn(move || {
                                ready.wait();
                                start.wait();
                                for _ in 0..iters {
                                    decode(output, false);
                                }
                            })
                        });
                        ready.wait();
                        let began = Instant::now();
                        start.wait();
                        for worker in workers {
                            worker.join().expect("resident decode worker");
                        }
                        began.elapsed()
                    })
                } else {
                    let began = Instant::now();
                    for _ in 0..iters {
                        for output in &outputs {
                            decode(output, false);
                        }
                    }
                    began.elapsed()
                }
            }),
        );
    }
}
