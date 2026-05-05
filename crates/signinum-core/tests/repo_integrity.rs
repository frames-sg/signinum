// SPDX-License-Identifier: Apache-2.0

use std::{
    ffi::OsStr,
    fs,
    path::{Component, Path, PathBuf},
};

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
fn workspace_contains_public_signinum_facade_crate() {
    let root = repo_root();
    let manifest_path = root.join("crates/signinum/Cargo.toml");
    let manifest = fs::read_to_string(&manifest_path).unwrap_or_else(|err| {
        panic!("read {}: {err}", manifest_path.display());
    });

    for required in [
        "name = \"signinum\"",
        "signinum-core",
        "signinum-jpeg",
        "signinum-j2k",
        "signinum-tilecodec",
    ] {
        assert!(
            manifest.contains(required),
            "{} must contain `{required}`",
            manifest_path
                .strip_prefix(root)
                .unwrap_or(&manifest_path)
                .display()
        );
    }

    let root_manifest =
        fs::read_to_string(root.join("Cargo.toml")).expect("read workspace manifest");
    assert!(
        root_manifest.contains("\"crates/signinum\""),
        "workspace members must include the public signinum facade crate"
    );
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
fn xtask_fuzz_build_checks_every_fuzz_manifest() {
    let root = repo_root();
    let xtask = fs::read_to_string(root.join("xtask/src/main.rs")).expect("read xtask");
    let mut manifests = Vec::new();

    for entry in fs::read_dir(root.join("crates")).expect("read crates dir") {
        let entry = entry.expect("read crate entry");
        let manifest = entry.path().join("fuzz/Cargo.toml");
        if manifest.exists() {
            manifests.push(manifest);
        }
    }
    manifests.sort();
    assert!(
        !manifests.is_empty(),
        "repository must keep fuzz targets under crates/*/fuzz"
    );

    for manifest in manifests {
        let relative = manifest
            .strip_prefix(root)
            .expect("fuzz manifest under repo root")
            .display()
            .to_string();
        assert!(
            xtask.contains(&relative),
            "xtask fuzz-build must check {relative}"
        );
    }
}

#[test]
fn ci_coverage_job_is_a_required_gate() {
    let workflow =
        fs::read_to_string(repo_root().join(".github/workflows/ci.yml")).expect("read CI workflow");
    let coverage_job = workflow_job(&workflow, "coverage");

    assert!(
        coverage_job.contains("taiki-e/install-action@cargo-llvm-cov")
            && coverage_job.contains("cargo xtask coverage"),
        "coverage job must install cargo-llvm-cov and run xtask coverage"
    );
    assert!(
        !coverage_job.contains("continue-on-error"),
        "coverage job must not be allowed to fail silently"
    );
}

#[test]
fn gpu_validation_workflow_is_self_hosted_and_explicit() {
    let root = repo_root();
    let workflow_path = root.join(".github/workflows/gpu-validation.yml");
    let workflow = fs::read_to_string(&workflow_path).expect("read GPU validation workflow");

    for required in [
        "workflow_dispatch",
        "run-timed-benchmarks",
        "run-metal-validation",
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
        "SIGNINUM_REQUIRE_CUDA_JPEG_HARDWARE_DECODE",
        "SIGNINUM_GPU_BENCH_DIM",
        "SIGNINUM_GPU_BENCH_BATCH",
        "SIGNINUM_GPU_BENCH_BATCH_DIM",
        "uname -a",
        "rustc -Vv",
        "cargo -V",
        "nvidia-smi",
        "ldconfig -p | grep -i nvjpeg",
        "CUDA runtime validation requires a working CUDA driver",
        "cargo test -p signinum-jpeg-cuda --all-targets --features cuda-runtime",
        "cargo test -p signinum-j2k-cuda --all-targets --features cuda-runtime",
        "cargo bench -p signinum-jpeg-cuda --bench device_decode --features cuda-runtime --no-run",
        "cargo bench -p signinum-jpeg-cuda --bench device_decode --features cuda-runtime -- --noplot",
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
fn crates_io_publish_policy_is_explicit() {
    let root = repo_root();
    let workspace = fs::read_to_string(root.join("Cargo.toml")).expect("read workspace manifest");
    let xtask = fs::read_to_string(root.join("xtask/src/main.rs")).expect("read xtask");
    let publishable = const_array_block(&xtask, "PUBLISHABLE_PACKAGES");
    let publish_workflow = fs::read_to_string(root.join(".github/workflows/publish.yml"))
        .expect("read publish workflow");

    assert!(
        workspace.contains("version      = \"1.0.1\""),
        "workspace package version must match the current staged release version"
    );

    for package in [
        "signinum-core",
        "signinum-cuda-runtime",
        "signinum-j2k-native",
        "signinum-tilecodec",
        "signinum-jpeg",
        "signinum-j2k",
        "signinum-jpeg-metal",
        "signinum-jpeg-cuda",
        "signinum-j2k-metal",
        "signinum-j2k-cuda",
        "signinum-cli",
    ] {
        assert!(
            publishable.contains(&format!("\"{package}\"")),
            "xtask package gate must include publishable package {package}"
        );
        assert!(
            publish_workflow.contains(&format!("publish-{package}:")),
            "publish workflow must include package {package}"
        );
    }

    let package = "signinum-j2k-compare";
    assert!(
        !publishable.contains(&format!("\"{package}\"")),
        "xtask package gate must not package local comparator package {package}"
    );
    assert!(
        !publish_workflow.contains(&format!("publish-{package}:")),
        "publish workflow must not publish local comparator package {package}"
    );
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
fn public_docs_describe_facade_auto_and_cuda_runtime_surface_scope() {
    let root = repo_root();
    let readme = fs::read_to_string(root.join("README.md")).expect("read README");
    let changelog = fs::read_to_string(root.join("CHANGELOG.md")).expect("read changelog");
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
            docs.contains("facade release")
                && docs.contains("Runtime backend selection defaults to `Auto`"),
            "{name} must name the facade release posture and Auto backend policy"
        );
    }

    for (name, docs) in [
        ("README.md", readme.as_str()),
        ("CHANGELOG.md", changelog.as_str()),
        ("docs/wsi-decode-api.md", wsi_api.as_str()),
        ("docs/release.md", release.as_str()),
    ] {
        assert!(
            docs.contains("cuda-runtime")
                && docs.contains("CUDA device memory")
                && docs.contains("nvJPEG")
                && docs.contains("NVIDIA performance"),
            "{name} must describe CUDA device-memory output and nvJPEG scope without overclaiming NVIDIA performance"
        );
        assert!(
            !docs.contains("compatibility-only with no runtime CUDA decode"),
            "{name} must not describe CUDA as compatibility-only after runtime surface support exists"
        );
    }
}

#[test]
fn public_docs_route_users_to_current_crates_after_rename() {
    let root = repo_root();
    let readme = fs::read_to_string(root.join("README.md")).expect("read README");
    let migration = fs::read_to_string(root.join("private-notes.md")).expect("read migration docs");

    for required in [
        "Which crate should I use?",
        "private-notes.md",
        "statumen",
        "signinum-jpeg",
        "signinum-j2k",
        "signinum-cli",
    ] {
        assert!(
            readme.contains(required),
            "README.md must route users to `{required}` after the rename"
        );
    }

    for mapping in [
        ("ashlar-core", "signinum-core"),
        ("ashlar-jpeg", "signinum-jpeg"),
        ("ashlar-j2k", "signinum-j2k"),
        ("ashlar-tilecodec", "signinum-tilecodec"),
        ("ashlar-cli", "signinum-cli"),
        ("ashlar-j2k-native", "signinum-j2k-native"),
        ("ashlar-jpeg-metal", "signinum-jpeg-metal"),
        ("ashlar-j2k-metal", "signinum-j2k-metal"),
        ("ashlar-jpeg-cuda", "signinum-jpeg-cuda"),
        ("ashlar-j2k-cuda", "signinum-j2k-cuda"),
    ] {
        assert!(
            migration.contains(mapping.0) && migration.contains(mapping.1),
            "private-notes.md must map `{}` to `{}`",
            mapping.0,
            mapping.1
        );
    }

    assert!(
        migration.contains("ziggurat") && migration.contains("statumen"),
        "private-notes.md must map the retired reader crate to statumen"
    );
}

#[test]
fn published_crates_have_crates_io_landing_readmes() {
    let root = repo_root();

    for crate_dir in [
        "crates/signinum-core",
        "crates/signinum-cuda-runtime",
        "crates/signinum-j2k-native",
        "crates/signinum-tilecodec",
        "crates/signinum-jpeg",
        "crates/signinum-j2k",
        "crates/signinum-jpeg-metal",
        "crates/signinum-jpeg-cuda",
        "crates/signinum-j2k-metal",
        "crates/signinum-j2k-cuda",
        "crates/signinum-cli",
    ] {
        let manifest_path = root.join(crate_dir).join("Cargo.toml");
        let manifest = fs::read_to_string(&manifest_path)
            .unwrap_or_else(|err| panic!("read {}: {err}", manifest_path.display()));
        let readme_path = root.join(crate_dir).join("README.md");

        assert!(
            manifest.contains("readme"),
            "{} must declare a readme for crates.io landing pages",
            manifest_path
                .strip_prefix(root)
                .unwrap_or(&manifest_path)
                .display()
        );
        assert!(
            readme_path.exists(),
            "{} must exist for crates.io landing pages",
            readme_path
                .strip_prefix(root)
                .unwrap_or(&readme_path)
                .display()
        );
    }
}

#[test]
fn public_text_does_not_embed_local_user_home_paths() {
    let root = repo_root();
    let mut offenders = Vec::new();

    for path in repo_text_files(root) {
        if is_archived_handoff(&path) {
            continue;
        }
        if path.ends_with("crates/signinum-core/tests/repo_integrity.rs") {
            continue;
        }
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        if source.contains("/Users/") || source.contains("C:\\Users\\") {
            offenders.push(
                path.strip_prefix(root)
                    .unwrap_or(&path)
                    .display()
                    .to_string(),
            );
        }
    }

    assert!(
        offenders.is_empty(),
        "public text must not embed local user-home paths; use env vars or repo-relative defaults: {offenders:?}"
    );
}

#[test]
fn referenced_shell_scripts_exist() {
    let root = repo_root();
    let mut missing = Vec::new();

    for path in repo_text_files(root) {
        if is_archived_handoff(&path) {
            continue;
        }
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        for script in referenced_shell_scripts(&source) {
            let root_relative = root.join(&script);
            let file_relative = path.parent().expect("text file has parent").join(&script);
            if !root_relative.exists() && !file_relative.exists() {
                missing.push(format!(
                    "{} references missing script {script}",
                    path.strip_prefix(root).unwrap_or(&path).display()
                ));
            }
        }
    }

    assert!(
        missing.is_empty(),
        "all referenced shell scripts must exist: {missing:?}"
    );
}

#[test]
fn public_narrative_docs_do_not_carry_stale_zeiss_claims() {
    let root = repo_root();
    let mut offenders = Vec::new();

    for relative in [
        "README.md",
        "docs/architecture.md",
        "docs/bench.md",
        "docs/parity.md",
        "docs/release.md",
        "docs/wsi-decode-api.md",
        "paper/paper.md",
        "paper/arxiv/main.tex",
    ] {
        let path = root.join(relative);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        if source.contains("Zeiss") {
            offenders.push(relative);
        }
    }

    assert!(
        offenders.is_empty(),
        "public narrative docs must not carry stale Zeiss integration claims: {offenders:?}"
    );
}

#[test]
fn packaged_rust_sources_do_not_include_files_outside_their_crate() {
    let root = repo_root();
    let workspace_crates = root.join("crates");
    let mut escaping = Vec::new();

    for source_path in rust_sources(&workspace_crates) {
        let Ok(relative_to_crates) = source_path.strip_prefix(&workspace_crates) else {
            continue;
        };
        let Some(crate_name) = relative_to_crates.components().next() else {
            continue;
        };
        let member_root = workspace_crates.join(crate_name.as_os_str());
        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|err| panic!("read {}: {err}", source_path.display()));

        for include_path in rust_include_paths(&source) {
            let resolved = normalize_path(
                &source_path
                    .parent()
                    .expect("source file has parent")
                    .join(&include_path),
            );
            if !resolved.starts_with(&member_root) {
                escaping.push(format!(
                    "{} includes {} outside package root",
                    source_path
                        .strip_prefix(root)
                        .unwrap_or(&source_path)
                        .display(),
                    include_path
                ));
            }
        }
    }

    assert!(
        escaping.is_empty(),
        "package source include paths must stay inside their crate so packaged tests/benches/examples are not dead: {escaping:?}"
    );
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

fn repo_text_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_repo_text_files(root, &mut out);
    out
}

fn collect_repo_text_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap_or_else(|err| panic!("read {}: {err}", dir.display())) {
        let entry = entry.expect("read directory entry");
        let path = entry.path();
        if path.is_dir() {
            if should_skip_repo_dir(&path) {
                continue;
            }
            collect_repo_text_files(&path, out);
            continue;
        }
        if is_repo_text_file(&path) {
            out.push(path);
        }
    }
}

fn should_skip_repo_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(OsStr::to_str)
        .is_some_and(|name| matches!(name, ".git" | ".venv" | "target"))
}

fn is_repo_text_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(OsStr::to_str),
        Some("bib" | "json" | "md" | "rs" | "sh" | "tex" | "toml" | "txt" | "yaml" | "yml")
    )
}

fn is_archived_handoff(path: &Path) -> bool {
    path.file_name()
        .and_then(OsStr::to_str)
        .is_some_and(|name| name.starts_with("HANDOFF-"))
}

fn referenced_shell_scripts(source: &str) -> Vec<String> {
    source
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_' | '/')))
        .filter(|token| {
            Path::new(token)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("sh"))
                && token.contains('/')
        })
        .filter(|token| !token.starts_with("http://") && !token.starts_with("https://"))
        .map(str::to_string)
        .collect()
}

fn rust_include_paths(source: &str) -> Vec<String> {
    let mut out = Vec::new();
    for marker in ["include_bytes!(\"", "include_str!(\""] {
        let mut rest = source;
        while let Some(start) = rest.find(marker) {
            let after_marker = &rest[start + marker.len()..];
            let Some(end) = after_marker.find('"') else {
                break;
            };
            out.push(after_marker[..end].to_string());
            rest = &after_marker[end + 1..];
        }
    }
    out
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                normalized.pop();
            }
            Component::CurDir => {}
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}
