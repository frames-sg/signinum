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
        "crates/signinum-jpeg-metal",
        "crates/signinum-jpeg-cuda",
        "crates/signinum-j2k-metal",
        "crates/signinum-j2k-cuda",
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
        "\"signinum-jpeg-metal\"",
        "\"signinum-j2k-metal\"",
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
        "cargo test -p signinum-jpeg-metal",
        "cargo test -p signinum-j2k-metal",
        "cargo test -p signinum-jpeg-cuda",
        "cargo test -p signinum-j2k-cuda",
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

#[test]
fn cuda_gpu_validation_job_stays_cuda_focused() {
    let root = repo_root();
    let workflow_path = root.join(".github/workflows/gpu-validation.yml");
    let workflow = fs::read_to_string(&workflow_path).expect("read GPU validation workflow");
    let cuda_job = workflow_job(&workflow, "cuda-x86_64-compatibility");

    for required in [
        "runs-on: [self-hosted, Linux, X64, cuda]",
        "SIGNINUM_REQUIRE_CUDA_RUNTIME",
        "uname -a",
        "rustc -Vv",
        "cargo -V",
        "nvidia-smi",
        "CUDA runtime validation requires a working CUDA driver",
        "cargo test -p signinum-jpeg-cuda --all-targets --features cuda-runtime",
        "cargo test -p signinum-j2k-cuda --all-targets --features cuda-runtime",
    ] {
        assert!(
            cuda_job.contains(required),
            "{} CUDA job must contain `{required}`",
            workflow_path
                .strip_prefix(root)
                .unwrap_or(&workflow_path)
                .display()
        );
    }

    for forbidden in [
        "cargo bench -p signinum-j2k-metal --bench compare --no-run",
        "cargo bench -p signinum-jpeg --no-run",
        "cargo test -p signinum-jpeg-metal",
        "cargo test -p signinum-j2k-metal",
    ] {
        assert!(
            !cuda_job.contains(forbidden),
            "{} CUDA job must not contain Metal validation command `{forbidden}`",
            workflow_path
                .strip_prefix(root)
                .unwrap_or(&workflow_path)
                .display()
        );
    }
}

#[test]
fn cpu_first_1_0_publish_policy_is_explicit() {
    let root = repo_root();
    let workspace = fs::read_to_string(root.join("Cargo.toml")).expect("read workspace manifest");
    let xtask = fs::read_to_string(root.join("xtask/src/main.rs")).expect("read xtask");
    let publishable = const_array_block(&xtask, "PUBLISHABLE_PACKAGES");
    let publish_workflow = fs::read_to_string(root.join(".github/workflows/publish.yml"))
        .expect("read publish workflow");

    assert!(
        workspace.contains("version      = \"1.0.0\""),
        "workspace package version must be the CPU-first 1.0 release version"
    );

    for package in [
        "signinum-core",
        "signinum-j2k-native",
        "signinum-tilecodec",
        "signinum-jpeg",
        "signinum-j2k",
        "signinum-cli",
    ] {
        assert!(
            publishable.contains(&format!("\"{package}\"")),
            "xtask package gate must include publishable CPU package {package}"
        );
        assert!(
            publish_workflow.contains(&format!("publish-{package}:")),
            "publish workflow must include CPU package {package}"
        );
    }

    for package in [
        "signinum-j2k-compare",
        "signinum-jpeg-metal",
        "signinum-jpeg-cuda",
        "signinum-j2k-metal",
        "signinum-j2k-cuda",
    ] {
        assert!(
            !publishable.contains(&format!("\"{package}\"")),
            "xtask package gate must not package pre-1.0 adapter/comparator package {package}"
        );
        assert!(
            !publish_workflow.contains(&format!("publish-{package}:")),
            "publish workflow must not publish pre-1.0 adapter/comparator package {package}"
        );
    }
}

fn const_array_block<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source
        .find(&format!("const {name}:"))
        .unwrap_or_else(|| panic!("missing const {name}"));
    let rest = &source[start..];
    let end = rest
        .find("];")
        .unwrap_or_else(|| panic!("unterminated const {name}"));
    &rest[..end]
}

#[test]
fn j2k_compare_stays_unpublished_and_out_of_j2k_package_deps() {
    let root = repo_root();
    let compare_manifest = fs::read_to_string(root.join("crates/signinum-j2k-compare/Cargo.toml"))
        .expect("read signinum-j2k-compare manifest");
    let j2k_manifest = fs::read_to_string(root.join("crates/signinum-j2k/Cargo.toml"))
        .expect("read signinum-j2k manifest");

    assert!(
        compare_manifest.contains("publish = false"),
        "signinum-j2k-compare must remain an unpublished local oracle helper"
    );
    assert!(
        !j2k_manifest.contains("signinum-j2k-compare"),
        "signinum-j2k must not package a dev-dependency on signinum-j2k-compare"
    );
}

#[test]
fn package_preflight_is_staged_dependency_aware() {
    let root = repo_root();
    let xtask = fs::read_to_string(root.join("xtask/src/main.rs")).expect("read xtask");
    let publish_script =
        fs::read_to_string(root.join("scripts/publish-crate.sh")).expect("read publish script");
    let release = fs::read_to_string(root.join("docs/release.md")).expect("read release docs");

    assert!(
        xtask.contains("STAGED_DEPENDENCY_PACKAGES"),
        "xtask package preflight must explicitly model crates blocked by unpublished staged dependencies"
    );
    assert!(
        xtask.contains("\"--list\"") && xtask.contains("unpublished workspace dependencies"),
        "xtask package preflight must validate package contents for staged downstream crates without hiding why strict packaging is skipped"
    );
    assert!(
        publish_script.contains("dry-run package list only")
            && publish_script.contains("signinum-cli")
            && publish_script.contains("cargo package -p \"$crate\" --list"),
        "publish workflow dry-run must not fail downstream crates only because staged dependency versions are not published yet"
    );
    assert!(
        release.contains("cargo package --list")
            && release.contains("cargo publish --dry-run")
            && release.contains("unpublished workspace dependencies"),
        "release docs must explain the pre-publish package validation limits"
    );
}

#[test]
fn public_docs_describe_cpu_first_1_0_and_cuda_runtime_surface_scope() {
    let root = repo_root();
    let readme = fs::read_to_string(root.join("README.md")).expect("read README");
    let architecture =
        fs::read_to_string(root.join("docs/architecture.md")).expect("read architecture docs");
    let release = fs::read_to_string(root.join("docs/release.md")).expect("read release docs");
    let wsi_api =
        fs::read_to_string(root.join("docs/wsi-decode-api.md")).expect("read WSI API docs");

    for (name, docs) in [
        ("README.md", readme.as_str()),
        ("docs/architecture.md", architecture.as_str()),
        ("docs/release.md", release.as_str()),
    ] {
        assert!(
            docs.contains("CPU-first 1.0"),
            "{name} must name the CPU-first 1.0 release posture"
        );
    }

    for (name, docs) in [
        ("README.md", readme.as_str()),
        ("docs/wsi-decode-api.md", wsi_api.as_str()),
        ("docs/release.md", release.as_str()),
    ] {
        assert!(
            docs.contains("cuda-runtime")
                && docs.contains("CUDA device memory")
                && docs.contains("no CUDA kernel decode")
                && docs.contains("NVIDIA performance"),
            "{name} must describe CUDA device-memory output without claiming CUDA kernel decode or NVIDIA performance"
        );
        assert!(
            !docs.contains("compatibility-only with no runtime CUDA decode"),
            "{name} must not describe CUDA as compatibility-only after runtime surface support exists"
        );
    }
}

fn workflow_job<'a>(workflow: &'a str, job_name: &str) -> &'a str {
    let marker = format!("  {job_name}:");
    let start = workflow
        .find(&marker)
        .unwrap_or_else(|| panic!("missing workflow job {job_name}"));
    let rest = &workflow[start..];
    let mut search_start = marker.len();
    let mut end = rest.len();
    while let Some(relative) = rest[search_start..].find("\n  ") {
        let candidate = search_start + relative + 1;
        if !rest[candidate..].starts_with("    ") {
            end = candidate;
            break;
        }
        search_start = candidate + 1;
    }
    &rest[..end]
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
