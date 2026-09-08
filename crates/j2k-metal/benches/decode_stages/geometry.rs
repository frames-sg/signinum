// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    combined_dispatch_report, encode_htj2k, readback_bytes, readback_hash, require_success,
    resident_bytes, Arc, BatchDecodeOptions, BatchLayout, Criterion, EncodeOptions, EncodedImage,
    MetalBatchDecodeResult, MetalBatchDecoder, Throughput,
};

fn cold_input_decode(
    backend: &j2k_metal::MetalBackendSession,
    sources: &[Arc<[u8]>],
) -> MetalBatchDecodeResult {
    // Retain the initialized Metal device, queue, pipelines, and session caches.
    // A fresh decoder starts with empty prepared-image plan caches, so this row
    // measures input parsing/grouping and decoder preparation, not Metal startup.
    let mut decoder = MetalBatchDecoder::with_backend_session_and_options(
        backend.clone(),
        BatchDecodeOptions {
            layout: BatchLayout::Nhwc,
            ..BatchDecodeOptions::default()
        },
    );
    let inputs = sources
        .iter()
        .map(|bytes| EncodedImage::full(Arc::clone(bytes)))
        .collect();
    let prepared = decoder
        .prepare(inputs)
        .expect("cold-input geometry benchmark prepared batch");
    decoder
        .decode_prepared(&prepared)
        .expect("cold-input geometry benchmark decode")
}

pub(super) fn bench(criterion: &mut Criterion) {
    for (width, height, count) in [
        (128, 128, 16),
        (640, 480, 16),
        (1024, 1024, 16),
        (512, 512, 1),
    ] {
        let mut decoder = MetalBatchDecoder::system_default_with_options(BatchDecodeOptions {
            layout: BatchLayout::Nhwc,
            ..BatchDecodeOptions::default()
        })
        .expect("geometry benchmark Metal decoder");
        let options = EncodeOptions {
            reversible: false,
            num_decomposition_levels: 3,
            guard_bits: 2,
            ..EncodeOptions::default()
        };
        let mut inputs = Vec::new();
        let mut sources = Vec::new();
        let mut expected = Vec::new();
        let mut input_bytes = Vec::new();
        let mut input_hashes = std::collections::BTreeSet::new();
        for index in 0..count {
            let mut pixels = j2k_test_support::patterned_rgb8(width, height);
            pixels[0] = pixels[0].wrapping_add(u8::try_from(index).expect("small batch"));
            let bytes: Arc<[u8]> = Arc::from(
                encode_htj2k(&pixels, width, height, 3, 8, false, &options)
                    .expect("geometry benchmark fixture"),
            );
            assert!(
                input_hashes.insert(j2k_test_support::auto_routing_sha256(&bytes)),
                "each geometry fixture must have distinct codestream bytes"
            );
            expected.push(
                j2k_native::Image::new(&bytes, &j2k_native::DecodeSettings::default())
                    .expect("geometry oracle parses")
                    .decode_native()
                    .expect("geometry oracle decodes")
                    .data,
            );
            input_bytes.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            input_bytes.extend_from_slice(&bytes);
            inputs.push(EncodedImage::full(Arc::clone(&bytes)));
            sources.push(bytes);
        }
        let prepared = decoder
            .prepare(inputs)
            .expect("geometry benchmark prepared batch");
        let probe = decoder
            .decode_prepared(&prepared)
            .expect("geometry benchmark probe");
        require_success(&probe);
        assert_eq!(combined_dispatch_report(&probe).ht_tier1, 3);
        let surfaces = probe
            .groups()
            .iter()
            .flat_map(j2k_metal::MetalBatchGroup::surfaces)
            .collect::<Vec<_>>();
        assert_eq!(surfaces.len(), expected.len());
        for (surface, expected) in surfaces.iter().zip(&expected) {
            assert_eq!(
                surface.as_bytes().expect("geometry probe readback"),
                *expected
            );
        }
        let id = format!("metal_idwt97_geometry_distinct/{width}x{height}/batch-{count}");
        eprintln!(
            "{id} input_sha256={} output_sha256={}",
            j2k_test_support::auto_routing_sha256(&input_bytes),
            readback_hash(&probe)
        );
        let mut group = criterion.benchmark_group(id);
        group.throughput(Throughput::Elements(count));
        group.bench_function("resident", |bencher| {
            bencher.iter(|| {
                let result = decoder
                    .decode_prepared(std::hint::black_box(&prepared))
                    .expect("resident geometry decode");
                require_success(&result);
                std::hint::black_box(resident_bytes(&result));
            });
        });
        group.bench_function("readback", |bencher| {
            bencher.iter(|| {
                let result = decoder
                    .decode_prepared(std::hint::black_box(&prepared))
                    .expect("readback geometry decode");
                require_success(&result);
                std::hint::black_box(readback_bytes(&result));
            });
        });
        let backend = decoder.backend_session().clone();
        let cold_probe = cold_input_decode(&backend, &sources);
        require_success(&cold_probe);
        assert_eq!(combined_dispatch_report(&cold_probe).ht_tier1, 3);
        let cold_surfaces = cold_probe
            .groups()
            .iter()
            .flat_map(j2k_metal::MetalBatchGroup::surfaces)
            .collect::<Vec<_>>();
        assert_eq!(cold_surfaces.len(), expected.len());
        for (surface, expected) in cold_surfaces.iter().zip(&expected) {
            assert_eq!(
                surface.as_bytes().expect("cold-input probe readback"),
                *expected
            );
        }
        group.bench_function("cold_input_prepare_readback_warm_session", |bencher| {
            bencher.iter(|| {
                let result = cold_input_decode(&backend, &sources);
                require_success(&result);
                std::hint::black_box(readback_bytes(&result));
            });
        });
        group.finish();
    }
}
