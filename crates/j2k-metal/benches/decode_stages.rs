// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use std::sync::Arc;

#[cfg(target_os = "macos")]
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
#[cfg(target_os = "macos")]
use j2k::{BatchDecodeOptions, BatchLayout, DecodeRequest, EncodedImage, PreparedBatch};
#[cfg(target_os = "macos")]
use j2k_core::DeviceSurface;
#[cfg(target_os = "macos")]
use j2k_metal::{MetalBatchDecodeResult, MetalBatchDecoder, MetalDecodeDispatchReport};
#[cfg(target_os = "macos")]
use j2k_native::{encode_htj2k, EncodeOptions};

#[cfg(target_os = "macos")]
const DIMENSION: u32 = 512;
#[cfg(target_os = "macos")]
#[path = "decode_stages/geometry.rs"]
mod geometry;
#[cfg(target_os = "macos")]
#[path = "decode_stages/mixed_groups.rs"]
mod mixed_groups;
#[cfg(target_os = "macos")]
const BATCH_SIZE: usize = 16;
#[cfg(target_os = "macos")]
const STAGE_MARKERS: &[&str] = &[
    "entropy_tier1",
    "dequantization",
    "idwt",
    "inverse_mct",
    "final_store",
    "readback",
    "dispatch_report",
];

#[cfg(not(target_os = "macos"))]
fn main() {
    assert!(
        std::env::var_os("J2K_REQUIRE_METAL_BENCH").is_none(),
        "J2K Metal decode-stage benchmark requires macOS"
    );
    eprintln!("J2K Metal decode-stage benchmark skipped outside macOS");
}

#[cfg(target_os = "macos")]
fn fixture(reversible: bool) -> Arc<[u8]> {
    let pixels = j2k_test_support::patterned_rgb8(DIMENSION, DIMENSION);
    let options = EncodeOptions {
        reversible,
        num_decomposition_levels: 3,
        guard_bits: 2,
        ..EncodeOptions::default()
    };
    Arc::from(
        encode_htj2k(&pixels, DIMENSION, DIMENSION, 3, 8, false, &options)
            .expect("encode deterministic HTJ2K decode-stage fixture"),
    )
}

#[cfg(target_os = "macos")]
fn classic_fixture() -> Arc<[u8]> {
    let pixels = j2k_test_support::patterned_rgb8(DIMENSION, DIMENSION);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        guard_bits: 2,
        use_ht_block_coding: false,
        ..EncodeOptions::default()
    };
    Arc::from(
        j2k_native::encode(&pixels, DIMENSION, DIMENSION, 3, 8, false, &options)
            .expect("encode deterministic Classic J2K decode-stage fixture"),
    )
}

#[cfg(target_os = "macos")]
fn prepared_batch(decoder: &MetalBatchDecoder, reversible: bool) -> PreparedBatch {
    let bytes = fixture(reversible);
    let inputs = (0..BATCH_SIZE)
        .map(|_| EncodedImage {
            bytes: Arc::clone(&bytes),
            request: DecodeRequest::Full,
        })
        .collect();
    decoder
        .prepare(inputs)
        .expect("prepare repeated Metal decode-stage batch")
}

#[cfg(target_os = "macos")]
fn prepared_classic_batch(decoder: &MetalBatchDecoder) -> PreparedBatch {
    let bytes = classic_fixture();
    let inputs = (0..BATCH_SIZE)
        .map(|_| EncodedImage {
            bytes: Arc::clone(&bytes),
            request: DecodeRequest::Full,
        })
        .collect();
    decoder
        .prepare(inputs)
        .expect("prepare repeated Classic Metal decode-stage batch")
}

#[cfg(target_os = "macos")]
fn require_success(result: &MetalBatchDecodeResult) {
    assert!(
        result.errors().is_empty(),
        "decode-stage preparation failures: {:?}",
        result.errors()
    );
    assert!(
        result.group_errors().is_empty(),
        "decode-stage execution failures: {:?}",
        result.group_errors()
    );
    assert!(
        !result.groups().is_empty(),
        "decode-stage batch produced no groups"
    );
}

#[cfg(target_os = "macos")]
fn combined_dispatch_report(result: &MetalBatchDecodeResult) -> MetalDecodeDispatchReport {
    result
        .groups()
        .iter()
        .fold(MetalDecodeDispatchReport::new(), |mut total, group| {
            let report = group.dispatch_report();
            total.tier1 = total.tier1.saturating_add(report.tier1);
            total.ht_tier1 = total.ht_tier1.saturating_add(report.ht_tier1);
            total.ht_refinement = total.ht_refinement.saturating_add(report.ht_refinement);
            total.classic_tier1 = total.classic_tier1.saturating_add(report.classic_tier1);
            total.dequantization = total.dequantization.saturating_add(report.dequantization);
            total.idwt = total.idwt.saturating_add(report.idwt);
            total.mct = total.mct.saturating_add(report.mct);
            total.color_output = total.color_output.saturating_add(report.color_output);
            total.host_to_device = total.host_to_device.saturating_add(report.host_to_device);
            total
        })
}

#[cfg(target_os = "macos")]
fn resident_bytes(result: &MetalBatchDecodeResult) -> usize {
    result
        .groups()
        .iter()
        .flat_map(j2k_metal::MetalBatchGroup::surfaces)
        .map(j2k_metal::Surface::byte_len)
        .sum()
}

#[cfg(target_os = "macos")]
fn readback_bytes(result: &MetalBatchDecodeResult) -> usize {
    result
        .groups()
        .iter()
        .flat_map(j2k_metal::MetalBatchGroup::surfaces)
        .map(|surface| {
            surface
                .as_bytes()
                .expect("read back completed Metal decode-stage surface")
                .len()
        })
        .sum()
}

#[cfg(target_os = "macos")]
fn readback_hash(result: &MetalBatchDecodeResult) -> String {
    let bytes = result
        .groups()
        .iter()
        .flat_map(j2k_metal::MetalBatchGroup::surfaces)
        .flat_map(|surface| {
            surface
                .as_bytes()
                .expect("read back completed Metal decode-stage surface")
                .iter()
                .copied()
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    j2k_test_support::auto_routing_sha256(&bytes)
}

#[cfg(target_os = "macos")]
fn bench_decode_stages(criterion: &mut Criterion) {
    std::hint::black_box(STAGE_MARKERS);
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder = MetalBatchDecoder::system_default_with_options(options)
        .expect("create Metal decode-stage benchmark session");
    let prepared = prepared_batch(&decoder, true);

    let probe = decoder
        .decode_prepared(&prepared)
        .expect("run Metal decode-stage dispatch probe");
    require_success(&probe);
    let dispatch_report = combined_dispatch_report(&probe);
    eprintln!(
        "j2k_metal_decode_stage_dispatches output_sha256={} entropy_tier1={} dequantization={} idwt={} inverse_mct={} final_store={} host_to_device={}",
        readback_hash(&probe),
        dispatch_report.tier1,
        dispatch_report.dequantization,
        dispatch_report.idwt,
        dispatch_report.mct,
        dispatch_report.color_output,
        dispatch_report.host_to_device,
    );

    let mut group = criterion.benchmark_group("metal_decode_stages");
    group.sample_size(10);
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));
    group.bench_function("resident_end_to_end", |bencher| {
        bencher.iter(|| {
            let result = decoder
                .decode_prepared(std::hint::black_box(&prepared))
                .expect("Metal resident decode-stage benchmark succeeds");
            require_success(&result);
            std::hint::black_box(resident_bytes(&result));
        });
    });
    group.bench_function("readback_end_to_end", |bencher| {
        bencher.iter(|| {
            let result = decoder
                .decode_prepared(std::hint::black_box(&prepared))
                .expect("Metal readback decode-stage benchmark succeeds");
            require_success(&result);
            std::hint::black_box(readback_bytes(&result));
        });
    });
    group.finish();
}

#[cfg(target_os = "macos")]
fn bench_decode_stages_idwt97(criterion: &mut Criterion) {
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder = MetalBatchDecoder::system_default_with_options(options)
        .expect("create irreversible Metal decode-stage benchmark session");
    let prepared = prepared_batch(&decoder, false);
    let probe = decoder
        .decode_prepared(&prepared)
        .expect("run irreversible Metal decode-stage dispatch probe");
    require_success(&probe);
    eprintln!(
        "j2k_metal_decode_idwt97_probe output_sha256={} bytes={}",
        readback_hash(&probe),
        readback_bytes(&probe)
    );

    let mut group = criterion.benchmark_group("metal_decode_stages_idwt97");
    group.sample_size(10);
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));
    group.bench_function("resident_end_to_end", |bencher| {
        bencher.iter(|| {
            let result = decoder
                .decode_prepared(std::hint::black_box(&prepared))
                .expect("Metal irreversible resident decode-stage benchmark succeeds");
            require_success(&result);
            std::hint::black_box(resident_bytes(&result));
        });
    });
    group.bench_function("readback_end_to_end", |bencher| {
        bencher.iter(|| {
            let result = decoder
                .decode_prepared(std::hint::black_box(&prepared))
                .expect("Metal irreversible readback decode-stage benchmark succeeds");
            require_success(&result);
            std::hint::black_box(readback_bytes(&result));
        });
    });
    group.finish();
}

#[cfg(target_os = "macos")]
fn bench_decode_stages_classic(criterion: &mut Criterion) {
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder = MetalBatchDecoder::system_default_with_options(options)
        .expect("create Classic Metal decode-stage benchmark session");
    let prepared = prepared_classic_batch(&decoder);
    let probe = decoder
        .decode_prepared(&prepared)
        .expect("run Classic Metal decode-stage dispatch probe");
    require_success(&probe);
    let dispatch_report = combined_dispatch_report(&probe);
    assert!(dispatch_report.classic_tier1 > 0);
    eprintln!(
        "j2k_metal_classic_decode_probe output_sha256={} bytes={} classic_tier1_dispatches={}",
        readback_hash(&probe),
        readback_bytes(&probe),
        dispatch_report.classic_tier1,
    );

    let mut group = criterion.benchmark_group("metal_decode_stages_classic");
    group.sample_size(10);
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));
    group.bench_function("resident_end_to_end", |bencher| {
        bencher.iter(|| {
            let result = decoder
                .decode_prepared(std::hint::black_box(&prepared))
                .expect("Metal Classic resident decode-stage benchmark succeeds");
            require_success(&result);
            std::hint::black_box(resident_bytes(&result));
        });
    });
    group.finish();
}

#[cfg(target_os = "macos")]
criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_decode_stages, bench_decode_stages_idwt97, bench_decode_stages_classic, geometry::bench, mixed_groups::bench
}
#[cfg(target_os = "macos")]
criterion_main!(benches);
