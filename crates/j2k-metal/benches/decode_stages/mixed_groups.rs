// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    combined_dispatch_report, encode_htj2k, readback_bytes, require_success, resident_bytes, Arc,
    BatchDecodeOptions, Criterion, EncodeOptions, EncodedImage, MetalBatchDecodeResult,
    MetalBatchDecoder, Throughput,
};

fn consume(result: &MetalBatchDecodeResult, host: bool) {
    require_success(result);
    std::hint::black_box(if host {
        readback_bytes(result)
    } else {
        resident_bytes(result)
    });
}

fn fixtures() -> (Vec<EncodedImage>, Vec<Vec<u8>>) {
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for (index, (width, height)) in [(192, 128), (128, 128), (128, 160), (160, 128)]
        .into_iter()
        .enumerate()
    {
        let seed = u8::try_from(index).expect("small mixed batch");
        let pixels = (0..width * height)
            .map(|offset| {
                u8::try_from(offset % 251)
                    .expect("bounded pixel")
                    .wrapping_add(seed)
            })
            .collect::<Vec<_>>();
        let bytes: Arc<[u8]> = Arc::from(
            encode_htj2k(
                &pixels,
                width,
                height,
                1,
                8,
                false,
                &EncodeOptions {
                    reversible: true,
                    num_decomposition_levels: 3,
                    ..EncodeOptions::default()
                },
            )
            .expect("encode distinct mixed group fixture"),
        );
        inputs.push(EncodedImage::full(bytes));
        expected.push(pixels);
    }
    (inputs, expected)
}

pub(super) fn bench(criterion: &mut Criterion) {
    let options = BatchDecodeOptions::default();
    let (inputs, expected) = fixtures();
    let mut decoder = MetalBatchDecoder::system_default_with_options(options)
        .expect("mixed group benchmark decoder");
    let prepared = decoder
        .prepare(inputs.clone())
        .expect("prepare mixed batch");
    assert_eq!(prepared.groups().len(), 4);
    let singles = inputs
        .iter()
        .cloned()
        .map(|input| {
            decoder
                .prepare(vec![input])
                .expect("prepare single mixed input")
        })
        .collect::<Vec<_>>();
    let probe = decoder
        .decode_prepared(&prepared)
        .expect("mixed batch probe");
    require_success(&probe);
    for group in probe.groups() {
        assert_eq!(group.surfaces().len(), 1);
        let source = group.source_indices()[0];
        assert_eq!(
            group.surfaces()[0]
                .as_bytes()
                .expect("mixed probe pixels")
                .as_ref(),
            expected[source].as_slice()
        );
    }
    eprintln!(
        "mixed distinct groups=4 submissions={} dispatch_report={:?}",
        decoder.submissions().expect("mixed submissions"),
        combined_dispatch_report(&probe)
    );
    drop(probe);

    let mut group = criterion.benchmark_group("metal_decode_stages_mixed_distinct");
    group.throughput(Throughput::Elements(inputs.len() as u64));
    for host in [false, true] {
        let output = if host { "host" } else { "resident" };
        for serial in [false, true] {
            let route = if serial { "single_loop" } else { "batch" };
            group.bench_function(format!("prepared/{route}/{output}"), |bencher| {
                bencher.iter(|| {
                    if serial {
                        for single in &singles {
                            consume(
                                &decoder
                                    .decode_prepared(std::hint::black_box(single))
                                    .expect("prepared single decode"),
                                host,
                            );
                        }
                    } else {
                        consume(
                            &decoder
                                .decode_prepared(std::hint::black_box(&prepared))
                                .expect("prepared mixed batch decode"),
                            host,
                        );
                    }
                });
            });
            group.bench_function(format!("cold/{route}/{output}"), |bencher| {
                bencher.iter(|| {
                    let mut cold = MetalBatchDecoder::system_default_with_options(options)
                        .expect("cold mixed decoder");
                    if serial {
                        for input in &inputs {
                            consume(
                                &cold
                                    .decode_batch(vec![input.clone()])
                                    .expect("cold single decode"),
                                host,
                            );
                        }
                    } else {
                        consume(
                            &cold
                                .decode_batch(inputs.clone())
                                .expect("cold mixed batch decode"),
                            host,
                        );
                    }
                });
            });
        }
    }
    group.finish();
}
