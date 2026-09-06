// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::{Path, PathBuf};

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn cargo_toml() -> String {
    let path = manifest_dir().join("Cargo.toml");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("must read {}: {error}", path.display()))
}

fn bench_sources_under(path: &Path) -> Vec<PathBuf> {
    if !path.exists() {
        return Vec::new();
    }

    let mut sources = Vec::new();
    let mut pending = vec![path.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let entries = std::fs::read_dir(&dir)
            .unwrap_or_else(|error| panic!("must read {}: {error}", dir.display()));
        for entry in entries {
            let entry = entry.unwrap_or_else(|error| {
                panic!("must enumerate {}: {error}", dir.display());
            });
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                sources.push(path);
            }
        }
    }
    sources.sort();
    sources
}

#[test]
fn j2k_metal_declares_the_audited_routing_and_decode_stage_benches() {
    let cargo = cargo_toml();

    assert_eq!(
        cargo.matches("[[bench]]").count(),
        6,
        "j2k-metal must keep all audited benchmark targets"
    );
    assert!(
        cargo.contains("[[bench]]\nname = \"auto_routing\"\nharness = false\ntest = false"),
        "j2k-metal must keep the release-routing benchmark explicit"
    );
    assert!(
        cargo.contains("[[bench]]\nname = \"decode_stages\"\nharness = false\ntest = false"),
        "j2k-metal must keep the decode-stage benchmark explicit"
    );
    assert!(
        cargo.contains("[[bench]]\nname = \"transform_stages\"\nharness = false\ntest = false"),
        "j2k-metal must keep the transform-stage benchmark explicit"
    );
    assert!(
        cargo.contains(
            "[[bench]]\nname = \"resident_packetization\"\nharness = false\ntest = false"
        ),
        "j2k-metal must keep the resident packetization benchmark explicit"
    );
    assert!(
        cargo.contains("[[bench]]\nname = \"htj2k_candidates\"\nharness = false\ntest = false"),
        "j2k-metal must keep the end-to-end HTJ2K candidate benchmark explicit"
    );
    assert!(
        cargo.contains("[[bench]]\nname = \"readback\"\nharness = false\ntest = false"),
        "j2k-metal must keep the paired public readback benchmark explicit"
    );

    for target in ["device_upload", "compare", "encode_stages"] {
        assert!(
            !cargo.contains(&format!("name = \"{target}\"")),
            "legacy j2k-metal bench target must stay removed: {target}"
        );
    }
}

#[test]
fn j2k_metal_bench_dependencies_are_limited_to_the_audited_targets() {
    let cargo = cargo_toml();

    assert_eq!(cargo.matches("criterion =").count(), 1);
    assert!(
        !cargo.contains("j2k-compare ="),
        "legacy comparison dependency must stay removed"
    );
}

#[test]
fn j2k_metal_benches_directory_matches_the_audited_targets() {
    let sources = bench_sources_under(&manifest_dir().join("benches"));

    assert_eq!(
        sources,
        [
            manifest_dir().join("benches/auto_routing/decode.rs"),
            manifest_dir().join("benches/auto_routing/encode.rs"),
            manifest_dir().join("benches/auto_routing/runner.rs"),
            manifest_dir().join("benches/auto_routing.rs"),
            manifest_dir().join("benches/decode_stages/geometry.rs"),
            manifest_dir().join("benches/decode_stages.rs"),
            manifest_dir().join("benches/htj2k_candidates/case.rs"),
            manifest_dir().join("benches/htj2k_candidates/runner.rs"),
            manifest_dir().join("benches/htj2k_candidates.rs"),
            manifest_dir().join("benches/readback.rs"),
            manifest_dir().join("benches/resident_packetization/batch_compare.rs"),
            manifest_dir().join("benches/resident_packetization/classic_chunks.rs"),
            manifest_dir().join("benches/resident_packetization/packetization.rs"),
            manifest_dir().join("benches/resident_packetization/support.rs"),
            manifest_dir().join("benches/resident_packetization.rs"),
            manifest_dir().join("benches/transform_stages/resident_lossy.rs"),
            manifest_dir().join("benches/transform_stages.rs"),
        ],
        "j2k-metal benchmark sources must stay limited to audited evidence targets"
    );
}

#[test]
fn htj2k_candidate_bench_covers_geometry_modes_budgets_and_correctness() {
    let bench = [
        manifest_dir().join("benches/htj2k_candidates.rs"),
        manifest_dir().join("benches/htj2k_candidates/case.rs"),
        manifest_dir().join("benches/htj2k_candidates/runner.rs"),
    ]
    .into_iter()
    .map(|path| {
        std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("must read {}: {error}", path.display()))
    })
    .collect::<Vec<_>>()
    .join("\n");

    for expected in [
        "const SMALL_TILE_SIDE: u32 = 256",
        "const MEDIUM_TILE_SIDE: u32 = 512",
        "const LARGE_TILE_SIDE: u32 = 1024",
        "LosslessTwoLayers",
        "LosslessThreeLayers",
        "LossyTwoBudgets",
        "LossyThreeBudgets",
        "J2kRateTarget::BitsPerPixel",
        "candidate_set_dispatches",
        "verify_lossless_roundtrip",
        "verify_lossy_parity",
        "assert_output_parity",
    ] {
        assert!(
            bench.contains(expected),
            "HTJ2K candidate benchmark is missing `{expected}`"
        );
    }
}

#[test]
fn resident_packetization_bench_has_exact_output_probes() {
    let root = manifest_dir().join("benches/resident_packetization.rs");
    let mut paths = bench_sources_under(&manifest_dir().join("benches/resident_packetization"));
    paths.push(root);
    let source = paths
        .into_iter()
        .map(|path| {
            std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("must read {}: {error}", path.display()))
        })
        .collect::<Vec<_>>()
        .join("\n");

    for expected in [
        "const BATCH_SIZES: [usize; 3] = [1, 4, 16]",
        "J2kBlockCodingMode::Classic",
        "J2kBlockCodingMode::HighThroughput",
        "submit_lossless_batch_to_metal",
        "encode_lossless_batch_with_report",
        "encode_cpu_parallel",
        "gpu_encode_inflight_tiles: Some(batch_size)",
        "verify_cpu_metal_batch",
        "codestream_bytes()",
        "auto_routing_sha256",
    ] {
        assert!(
            source.contains(expected),
            "resident packetization benchmark is missing `{expected}`"
        );
    }
    assert!(
        !source.contains("env_flag_from_env") && !source.contains("route="),
        "the durable packetization benchmark must measure the production route"
    );
    assert!(
        !source.contains("packet_payload_copy_job_count_total")
            && !source.contains("packet_payload_copy_launched_stripe_count_total"),
        "the public batch outcome only publishes resident route counters when stage profiling is enabled"
    );
}

#[test]
fn metal_auto_routing_bench_uses_versioned_part15_workload_identity() {
    let bench = bench_sources_under(&manifest_dir().join("benches/auto_routing"))
        .into_iter()
        .map(|path| {
            std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("must read {}: {error}", path.display()))
        })
        .collect::<Vec<_>>()
        .join("\n");

    for expected in [
        "schema_version: workloads.manifest.schema_version",
        "validate_auto_routing_decode_identity",
        "case.codec.is_high_throughput()",
        "J2kBlockCodingMode::HighThroughput",
        "const BATCH_SIZE: usize = 16",
        "bench_batch_cell",
        "for_host_output_benchmark",
        "verify_lossless_output",
    ] {
        assert!(
            bench.contains(expected),
            "Metal Auto-routing benchmark is missing `{expected}`"
        );
    }

    for duplicate in [
        "AUTO_GRAY8_MIN_PIXELS",
        "AUTO_RGB8_BATCH_MIN_PIXELS",
        "AUTO_RGB8_LARGE_MIN_PIXELS",
        "expected_auto_decode_backend",
        "expected_auto_encode_dispatch",
    ] {
        assert!(
            !bench.contains(duplicate),
            "Metal benchmark must not duplicate production routing policy `{duplicate}`"
        );
    }
}

#[test]
fn metal_encode_auto_routing_accepts_external_staged_pnm_sources() {
    let path = manifest_dir().join("tests/encode_auto_routing_benchmark.rs");
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("must read {}: {error}", path.display()));

    for expected in [
        "J2K_METAL_ENCODE_INPUT_DIRS",
        "J2K_METAL_ENCODE_MANIFEST",
        "J2K_METAL_ENCODE_INCLUDE_GENERATED",
        "j2k_metal_encode_io_policy",
        "j2k_metal_encode_external_case_count",
        "j2k_metal_encode_external_input_format",
        "j2k_metal_encode_resident_bench",
        "j2k_metal_encode_resident_batch_sizes",
        "hybrid_cpu_packet_ms",
        "resident_host_ms",
        "resident_buffer_ms",
        "resident_input_storage=private",
        "resident_staging=already_padded_contiguous",
        "packetization_used",
        "codestream_assembly_used",
        "host_readback_ms",
        "staged-pnm-p5-p6",
        "lossless_external",
        "read_pnm_image",
        "validate_metal_encode_manifest_entry",
        "input_fnv1a64",
    ] {
        assert!(
            source.contains(expected),
            "Metal auto-routing benchmark is missing `{expected}`"
        );
    }
}

#[test]
fn metal_decode_benchmark_accepts_external_codestream_fixtures() {
    let path = manifest_dir().join("tests/metal_decode_benchmark.rs");
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("must read {}: {error}", path.display()));

    for expected in [
        "J2K_METAL_DECODE_INPUT_DIRS",
        "J2K_METAL_DECODE_MANIFEST",
        "J2K_METAL_DECODE_INCLUDE_GENERATED",
        "j2k_metal_decode_io_policy",
        "j2k_metal_decode_generated_case_count",
        "j2k_metal_decode_external_case_count",
        "j2k_metal_decode_bench",
        "metal_resident_ms",
        "metal_readback_ms",
        "input_fnv1a64",
        "raw-codestream",
        "wrapper_container_not_claimed_for_metal_decode",
    ] {
        assert!(
            source.contains(expected),
            "Metal decode benchmark is missing `{expected}`"
        );
    }
}
