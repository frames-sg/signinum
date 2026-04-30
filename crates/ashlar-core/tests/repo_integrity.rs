// SPDX-License-Identifier: Apache-2.0

use std::{fs, path::Path};

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
}

#[test]
fn conformance_manifest_lists_all_committed_jpeg_inputs() {
    let root = repo_root();
    let conformance = root.join("corpus/conformance");
    let manifest =
        fs::read_to_string(conformance.join("manifest.json")).expect("read conformance manifest");

    for entry in fs::read_dir(&conformance).expect("read conformance dir") {
        let entry = entry.expect("read conformance entry");
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("jpg") {
            continue;
        }
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("utf-8 fixture filename");
        assert!(
            manifest.contains(&format!("\"{filename}\"")),
            "conformance fixture {filename} is missing from manifest.json"
        );
    }
}

#[test]
fn corpus_readme_does_not_claim_committed_fixtures_are_absent() {
    let readme =
        fs::read_to_string(repo_root().join("corpus/README.md")).expect("read corpus README");

    assert!(
        !readme.contains("intentionally empty"),
        "corpus README still claims the committed fixture corpus is empty"
    );
}

#[test]
fn adapter_crates_do_not_import_codec_private_modules() {
    let root = repo_root();
    let adapter_crates = [
        "crates/ashlar-jpeg-metal",
        "crates/ashlar-jpeg-cuda",
        "crates/ashlar-j2k-metal",
        "crates/ashlar-j2k-cuda",
    ];

    for crate_dir in adapter_crates {
        for path in rust_sources(&root.join(crate_dir)) {
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
            assert!(
                !source.contains("::__private") && !source.contains(" __private::"),
                "adapter source {} imports a codec __private module; use the public adapter API",
                path.strip_prefix(root).unwrap_or(&path).display()
            );
        }
    }
}

#[test]
fn wsi_decode_api_guide_covers_public_surfaces() {
    let root = repo_root();
    let readme = fs::read_to_string(root.join("README.md")).expect("read README");
    let architecture =
        fs::read_to_string(root.join("docs/architecture.md")).expect("read architecture docs");
    let guide_path = root.join("docs/wsi-decode-api.md");
    let guide = fs::read_to_string(&guide_path).expect("read WSI decode API guide");

    assert!(
        readme.contains("docs/wsi-decode-api.md"),
        "README must link the WSI decode API guide"
    );
    assert!(
        architecture.contains("wsi-decode-api.md"),
        "architecture docs must link the WSI decode API guide"
    );

    for required in [
        "decode_region_scaled_into",
        "decode_rows",
        "TileBatchDecode",
        "BackendRequest::Auto",
        "BackendRequest::Metal",
        "BackendRequest::Cuda",
        "DeviceSurface",
        "ScratchPool",
        "DecoderContext",
    ] {
        assert!(
            guide.contains(required),
            "{} must document {required}",
            guide_path
                .strip_prefix(root)
                .unwrap_or(&guide_path)
                .display()
        );
    }
}

#[test]
fn ci_workflow_keeps_docs_and_benchmark_compile_gates() {
    let workflow =
        fs::read_to_string(repo_root().join(".github/workflows/ci.yml")).expect("read CI workflow");
    let xtask = fs::read_to_string(repo_root().join("xtask/src/main.rs")).expect("read xtask");

    for required in ["cargo xtask doc", "cargo xtask bench-build", "macos-13"] {
        assert!(
            workflow.contains(required),
            "CI workflow must contain `{required}`"
        );
    }

    for required in [
        "\"doc\"",
        "\"--workspace\"",
        "\"--all-features\"",
        "\"--no-deps\"",
        "\"ashlar-jpeg-metal\"",
        "\"ashlar-j2k-metal\"",
        "\"--no-run\"",
    ] {
        assert!(xtask.contains(required), "xtask must contain `{required}`");
    }
}

#[test]
fn gpu_validation_workflow_is_self_hosted_and_explicit() {
    let root = repo_root();
    let workflow_path = root.join(".github/workflows/gpu-validation.yml");
    let workflow = fs::read_to_string(&workflow_path).expect("read GPU validation workflow");

    for required in [
        "workflow_dispatch",
        "run-timed-benchmarks",
        "self-hosted",
        "metal",
        "cuda",
        "cargo test -p ashlar-jpeg-metal",
        "cargo test -p ashlar-j2k-metal",
        "cargo test -p ashlar-jpeg-cuda",
        "cargo test -p ashlar-j2k-cuda",
    ] {
        assert!(
            workflow.contains(required),
            "{} must contain `{required}`",
            workflow_path
                .strip_prefix(root)
                .unwrap_or(&workflow_path)
                .display()
        );
    }
}

fn rust_sources(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    collect_rust_sources(dir, &mut out);
    out
}

fn collect_rust_sources(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(dir).unwrap_or_else(|err| panic!("read {}: {err}", dir.display())) {
        let entry = entry.expect("read directory entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rust_sources(&path, out);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}
