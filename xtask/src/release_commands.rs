use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use crate::command_support::{
    command_output_os, ensure_clean_worktree, run_cargo, workspace_version,
};
use crate::process::cargo;

mod package_gate;
mod path_patches;
mod release_integrity_policy;
mod release_manifest;

use path_patches::workspace_path_patch_provenance_paths;
use release_integrity_policy::{
    validate_changelog_state, validate_patch_provenance, ReleaseIntegrityMode,
};
#[cfg(test)]
use release_manifest::parse_release_manifest_source;
use release_manifest::{
    crates_io_publishable, release_manifest_contract, validate_release_manifest_contract,
    ReleaseManifestContract,
};

const CPU_RELEASE_PACKAGES: &[&str] = &[
    "j2k-core",
    "j2k-codec-math",
    "j2k-jpeg",
    "j2k-types",
    "j2k-native",
    "j2k",
    "j2k-tilecodec",
    "j2k-cli",
];

pub(super) fn published_library_packages() -> Result<Vec<String>, String> {
    Ok(release_manifest_contract()?
        .library_packages()
        .map(ToString::to_string)
        .collect())
}

pub(super) fn release_integrity(args: impl Iterator<Item = String>) -> Result<(), String> {
    let mode = ReleaseIntegrityMode::parse(args)?;
    let metadata = cargo_metadata()?;
    let workspace_version = workspace_version()?;
    let release_manifest = release_manifest_contract()?;
    let publishable_set = release_manifest.ordered_crates().collect::<BTreeSet<_>>();
    let library_set = release_manifest.library_packages().collect::<BTreeSet<_>>();
    let mut errors = Vec::new();

    let workspace_packages = workspace_package_records(&metadata)?;
    validate_release_manifest_contract(&release_manifest, &workspace_packages, &mut errors)?;
    let mut workspace_names = BTreeSet::new();

    let unpublished_members = workspace_packages
        .iter()
        .copied()
        .map(|package| {
            let name = package_name(package)?;
            Ok((name, crates_io_publishable(package, name)?))
        })
        .filter_map(|result| match result {
            Ok((name, false)) => Some(Ok(name)),
            Ok((_, true)) => None,
            Err(error) => Some(Err(error)),
        })
        .collect::<Result<BTreeSet<_>, String>>()?;

    for package in workspace_packages {
        let name = package_name(package)?;
        workspace_names.insert(name.to_string());
        let listed_publishable = publishable_set.contains(name);
        let crates_io_eligible = crates_io_publishable(package, name)?;

        if listed_publishable && !crates_io_eligible {
            errors.push(format!(
                "`{name}` is listed in release-crates.json but is not eligible for crates.io publication"
            ));
            continue;
        }
        if !listed_publishable && crates_io_eligible {
            errors.push(format!(
                "`{name}` is eligible for crates.io publication but missing from release-crates.json"
            ));
            continue;
        }
        if !listed_publishable {
            continue;
        }

        validate_unpublished_dependencies(name, package, &unpublished_members, &mut errors)?;

        let version = package
            .get("version")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("cargo metadata package `{name}` has no string version"))?;
        if version != workspace_version {
            errors.push(format!(
                "`{name}` version {version} does not match workspace version {workspace_version}"
            ));
        }
        if package
            .get("readme")
            .and_then(serde_json::Value::as_str)
            .is_none()
        {
            errors.push(format!("`{name}` is publishable but has no package README"));
        }
        if !has_docs_rs_metadata(package) {
            errors.push(format!(
                "`{name}` is publishable but missing [package.metadata.docs.rs] with all-features and empty targets"
            ));
        }
        if has_lib_target(package) {
            if !library_set.contains(name) {
                errors.push(format!(
                    "`{name}` has a library target but release-crates.json does not assign a library API contract"
                ));
            }
        } else if name != "j2k-cli" {
            errors.push(format!(
                "`{name}` is publishable but has no library target and no explicit release-integrity exemption"
            ));
        }
    }

    for package in release_manifest.ordered_crates() {
        if !workspace_names.contains(package) {
            errors.push(format!(
                "`{package}` is listed in release-crates.json but is not a workspace member"
            ));
        }
    }

    validate_publish_workflow(&mut errors)?;
    validate_publish_script(&mut errors)?;
    validate_release_docs(&release_manifest, &mut errors)?;
    validate_release_metadata(&workspace_version, mode, &mut errors)?;

    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "release integrity violations:\n- {}",
            errors.join("\n- ")
        ))
    }
}

fn workspace_package_records(
    metadata: &serde_json::Value,
) -> Result<Vec<&serde_json::Value>, String> {
    let packages = metadata
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "cargo metadata did not contain a packages array".to_string())?;
    let mut packages_by_id = BTreeMap::new();
    for (index, package) in packages.iter().enumerate() {
        let id = package
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("cargo metadata packages[{index}] has no string id"))?;
        if packages_by_id.insert(id, package).is_some() {
            return Err(format!(
                "cargo metadata contains duplicate package id `{id}`"
            ));
        }
    }

    let members = metadata
        .get("workspace_members")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "cargo metadata did not contain a workspace_members array".to_string())?;
    let mut seen_members = BTreeSet::new();
    let mut workspace_packages = Vec::new();
    workspace_packages
        .try_reserve_exact(members.len())
        .map_err(|error| format!("reserve workspace package metadata: {error}"))?;
    for (index, member) in members.iter().enumerate() {
        let id = member
            .as_str()
            .ok_or_else(|| format!("cargo metadata workspace_members[{index}] is not a string"))?;
        if !seen_members.insert(id) {
            return Err(format!(
                "cargo metadata workspace_members contains duplicate id `{id}`"
            ));
        }
        let package = packages_by_id.get(id).copied().ok_or_else(|| {
            format!("cargo metadata workspace member `{id}` has no package record")
        })?;
        workspace_packages.push(package);
    }
    Ok(workspace_packages)
}

fn validate_release_metadata(
    workspace_version: &str,
    mode: ReleaseIntegrityMode,
    errors: &mut Vec<String>,
) -> Result<(), String> {
    let changelog_path = Path::new("CHANGELOG.md");
    let changelog = fs::read_to_string(changelog_path)
        .map_err(|err| format!("failed to read {}: {err}", changelog_path.display()))?;
    if let Err(error) = validate_changelog_state(&changelog, workspace_version, mode) {
        errors.push(format!("{}: {error}", changelog_path.display()));
    }

    if mode == ReleaseIntegrityMode::Publish {
        let manifest_path = Path::new("Cargo.toml");
        let manifest = fs::read_to_string(manifest_path)
            .map_err(|err| format!("failed to read {}: {err}", manifest_path.display()))?;
        for provenance_path in workspace_path_patch_provenance_paths(&manifest)? {
            let provenance = fs::read_to_string(&provenance_path)
                .map_err(|err| format!("failed to read {}: {err}", provenance_path.display()))?;
            if let Err(error) = validate_patch_provenance(&provenance) {
                errors.push(format!("{}: {error}", provenance_path.display()));
            }
        }
    }
    Ok(())
}

fn validate_unpublished_dependencies(
    name: &str,
    package: &serde_json::Value,
    unpublished_members: &BTreeSet<&str>,
    errors: &mut Vec<String>,
) -> Result<(), String> {
    let dependencies = package
        .get("dependencies")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("cargo metadata package `{name}` has no dependencies array"))?;
    for (index, dependency) in dependencies.iter().enumerate() {
        let dep_name = dependency
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                format!("cargo metadata package `{name}` dependency[{index}] has no string name")
            })?;
        if !unpublished_members.contains(dep_name) {
            continue;
        }
        let kind = match dependency.get("kind") {
            None | Some(serde_json::Value::Null) => "normal",
            Some(serde_json::Value::String(kind)) => kind,
            Some(_) => {
                return Err(format!(
                    "cargo metadata package `{name}` dependency `{dep_name}` has invalid kind"
                ));
            }
        };
        let req = dependency
            .get("req")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                format!(
                    "cargo metadata package `{name}` dependency `{dep_name}` has no string requirement"
                )
            })?;
        if kind != "dev" {
            errors.push(format!(
                "`{name}` has a {kind} dependency on unpublished crate `{dep_name}`; \
                 cargo publish cannot resolve it"
            ));
        } else if req != "*" {
            errors.push(format!(
                "`{name}` has a versioned dev-dependency `{dep_name} = \"{req}\"` on an \
                 unpublished crate; drop the version so cargo publish strips the path-only dev-dep"
            ));
        }
    }
    Ok(())
}

fn cargo_metadata() -> Result<serde_json::Value, String> {
    let data = command_output_os(
        cargo(),
        &["metadata", "--locked", "--no-deps", "--format-version", "1"],
    )?;
    serde_json::from_str(&data).map_err(|err| format!("failed to parse cargo metadata: {err}"))
}

fn package_name(package: &serde_json::Value) -> Result<&str, String> {
    package
        .get("name")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "cargo metadata package missing name".to_string())
}

fn has_lib_target(package: &serde_json::Value) -> bool {
    package
        .get("targets")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|targets| {
            targets.iter().any(|target| {
                target
                    .get("kind")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|kind| {
                        kind.iter()
                            .any(|entry| entry.as_str().is_some_and(|entry| entry == "lib"))
                    })
            })
        })
}

fn has_docs_rs_metadata(package: &serde_json::Value) -> bool {
    let Some(docs_rs) = package
        .get("metadata")
        .and_then(|metadata| metadata.get("docs"))
        .and_then(|docs| docs.get("rs"))
    else {
        return false;
    };

    docs_rs
        .get("all-features")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
        && docs_rs
            .get("targets")
            .and_then(serde_json::Value::as_array)
            .is_some_and(Vec::is_empty)
}

fn validate_publish_workflow(errors: &mut Vec<String>) -> Result<(), String> {
    let workflow_path = Path::new(".github/workflows/publish.yml");
    let workflow_source = fs::read_to_string(workflow_path)
        .map_err(|err| format!("failed to read {}: {err}", workflow_path.display()))?;
    validate_publish_workflow_source(&workflow_source, errors)
}

fn validate_publish_workflow_source(
    workflow_source: &str,
    errors: &mut Vec<String>,
) -> Result<(), String> {
    let workflow_path = Path::new(".github/workflows/publish.yml");
    for required in [
        "--origin-url \"${origin_url}\"",
        "--server-url \"${GITHUB_SERVER_URL}\"",
        "--ci-workflow full-validation.yml",
        "--cuda-job \"GPU / CUDA full release validation\"",
        "--metal-job \"GPU / Metal full release validation\"",
        "--t803-out-dir target/t803/release-evidence",
        "--t803-scope all",
        "cargo xtask t803 verify --scope all --candidate-sha",
        "j2k-t803-cpu-linux-x86_64-${candidate_sha}/cpu.json",
        "j2k-t803-cpu-macos-aarch64-${candidate_sha}/cpu.json",
        "j2k-t803-cpu-windows-x86_64-${candidate_sha}/cpu.json",
        "j2k-t803-cuda-linux-x86_64-${candidate_sha}/cuda.json",
        "j2k-t803-metal-macos-aarch64-${candidate_sha}/metal.json",
        "cargo xtask release-integrity --publish",
        "scripts/publish-crate.sh --preflight-all",
        "python3 scripts/publish_release.py preflight",
        "python3 scripts/publish_release.py publish",
        "environment: crates-io-publish",
        "CARGO_REGISTRY_TOKEN: ${{ secrets.CRATES_IO_API_TOKEN }}",
        "CARGO_UNSTABLE_PUBLISH_TIMEOUT: \"true\"",
        "CARGO_PUBLISH_TIMEOUT: \"600\"",
        "toolchain: nightly-",
    ] {
        if !workflow_source.contains(required) {
            errors.push(format!(
                "{} does not enforce publication preflight `{required}`",
                workflow_path.display()
            ));
        }
    }
    for forbidden in [
        "cargo publish --workspace",
        "CRATES_IO_INDEX_SETTLE_SECONDS",
        "sleep ",
    ] {
        if workflow_source.contains(forbidden) {
            errors.push(format!(
                "{} contains forbidden publishing behavior `{forbidden}`",
                workflow_path.display()
            ));
        }
    }
    let checkout_count = workflow_source.matches("uses: actions/checkout@").count();
    let explicit_ref_count = workflow_source.matches("ref: ${{ github.ref }}").count();
    let full_history_count = workflow_source.matches("fetch-depth: 0").count();
    if checkout_count == 0
        || explicit_ref_count != checkout_count
        || full_history_count != checkout_count
    {
        errors.push(format!(
            "{} must bind all {checkout_count} checkout steps to the triggering ref with full tag history; found {explicit_ref_count} explicit refs and {full_history_count} full-history fetches",
            workflow_path.display()
        ));
    }
    let workflow: serde_yaml_ng::Value = serde_yaml_ng::from_str(workflow_source)
        .map_err(|err| format!("failed to parse {}: {err}", workflow_path.display()))?;
    let workflow_mapping = workflow
        .as_mapping()
        .ok_or_else(|| format!("{} root must be a mapping", workflow_path.display()))?;
    let jobs = workflow_mapping
        .get(serde_yaml_ng::Value::String("jobs".to_string()))
        .and_then(serde_yaml_ng::Value::as_mapping)
        .ok_or_else(|| format!("{} jobs must be a mapping", workflow_path.display()))?;
    let actual_jobs = jobs
        .keys()
        .map(|key| {
            key.as_str()
                .map(ToString::to_string)
                .ok_or_else(|| format!("{} job names must be strings", workflow_path.display()))
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let expected_jobs = BTreeSet::from(["preflight".to_string(), "publish".to_string()]);
    if actual_jobs != expected_jobs {
        errors.push(format!(
            "{} must contain exactly preflight and publish jobs; found {actual_jobs:?}",
            workflow_path.display()
        ));
    }

    Ok(())
}

fn validate_publish_script(errors: &mut Vec<String>) -> Result<(), String> {
    let script_path = Path::new("scripts/publish-crate.sh");
    let script = fs::read_to_string(script_path)
        .map_err(|err| format!("failed to read {}: {err}", script_path.display()))?;
    validate_publish_script_source(&script, errors);

    let publisher_path = Path::new("scripts/publish_release.py");
    let publisher = fs::read_to_string(publisher_path)
        .map_err(|err| format!("failed to read {}: {err}", publisher_path.display()))?;
    for required in [
        "DEFAULT_MANIFEST = ROOT / \"release-crates.json\"",
        "hashlib.sha256",
        "validate_release_graph",
        "validate_registry_state",
        "RETRY_DELAYS_SECONDS = (5, 15, 30)",
        "[\"cargo\", \"publish\", \"--locked\", \"-p\", crate]",
        "CARGO_REGISTRY_TOKEN",
    ] {
        if !publisher.contains(required) {
            errors.push(format!(
                "{} does not enforce publisher check `{required}`",
                publisher_path.display()
            ));
        }
    }
    if publisher.contains("cargo publish --workspace") {
        errors.push(format!(
            "{} must preserve the ordered partial-release policy",
            publisher_path.display()
        ));
    }

    let manifest_validator_path = Path::new("scripts/release_manifest.py");
    let manifest_validator = fs::read_to_string(manifest_validator_path).map_err(|err| {
        format!(
            "failed to read {}: {err}",
            manifest_validator_path.display()
        )
    })?;
    validate_python_release_manifest_source(&manifest_validator, errors);
    Ok(())
}

fn validate_python_release_manifest_source(script: &str, errors: &mut Vec<String>) {
    let script_path = Path::new("scripts/release_manifest.py");
    for required in [
        "object_pairs_hook=_reject_duplicate_fields",
        "if type(schema) is not int or schema != 2",
        "\"workspace_members\"",
        "return \"crates-io\" in registries",
        "must be workspace/path sourced",
        "expected_requirement = f\"={versions_by_package[dependency_name]}\"",
        "if kind == \"dev\"",
        "if positions[dependency_name] >= positions[crate]",
        "registry_independent",
    ] {
        if !script.contains(required) {
            errors.push(format!(
                "{} does not enforce release-manifest check `{required}`",
                script_path.display()
            ));
        }
    }
}

fn validate_publish_script_source(script: &str, errors: &mut Vec<String>) {
    let script_path = Path::new("scripts/publish-crate.sh");
    for required in [
        "publish_release.py\" manifest",
        "--field ordered-crates",
        "--field registry-independent",
        "scripts/crates_io_version.py verify-set",
        "scripts/crates_io_version.py state",
        "cargo xtask release-integrity --publish",
        "workspace_repository",
        "normalize_repository_identity",
        "git config --get-all remote.origin.url",
        "git remote get-url --all origin",
        "git ls-remote --tags origin",
        "git show-ref --verify --quiet \"refs/tags/${expected_tag}\"",
        "git cat-file -t \"refs/tags/${expected_tag}\"",
        "refs/tags/${expected_tag}^{}",
        "refs/tags/${expected_tag}^{commit}",
        "HEAD^{commit}",
        "git status --porcelain=v1 --untracked-files=all",
        "cargo package -p \"$crate\" --no-verify",
        "cargo publish -p \"$crate\" --dry-run",
    ] {
        if !script.contains(required) {
            errors.push(format!(
                "{} does not enforce publish-script check `{required}`",
                script_path.display()
            ));
        }
    }
    if script.contains("cargo info") {
        errors.push(format!(
            "{} must not treat ambiguous cargo-info failures as version availability",
            script_path.display()
        ));
    }
}

fn validate_release_docs(
    manifest: &ReleaseManifestContract,
    errors: &mut Vec<String>,
) -> Result<(), String> {
    let release_doc_path = Path::new("docs/release.md");
    let release_doc = fs::read_to_string(release_doc_path)
        .map_err(|err| format!("failed to read {}: {err}", release_doc_path.display()))?;
    let packages = manifest.ordered_crates().collect::<Vec<_>>();
    validate_release_docs_source(&release_doc, &packages, errors);
    Ok(())
}

fn validate_release_docs_source(
    release_doc: &str,
    publishable_packages: &[&str],
    errors: &mut Vec<String>,
) {
    let release_doc_path = Path::new("docs/release.md");
    for package in publishable_packages {
        if !release_doc.contains(&format!("`{package}`")) {
            errors.push(format!(
                "{} does not document publishable crate `{package}`",
                release_doc_path.display()
            ));
        }
    }
    for required in [
        "cargo xtask release-integrity",
        "cargo xtask release-integrity --publish",
        "cargo package --no-verify",
        "already-published prefix",
        "Only an exact HTTP 404",
        "CRATES_IO_ALLOW_PUBLISHED_RERUN",
        "v<workspace.package.version>",
    ] {
        if !release_doc.contains(required) {
            errors.push(format!(
                "{} does not document `{required}`",
                release_doc_path.display()
            ));
        }
    }
}

pub(super) fn release_cpu() -> Result<(), String> {
    let mut args = vec!["test", "--release"];
    for package in CPU_RELEASE_PACKAGES {
        args.push("-p");
        args.push(package);
    }
    run_cargo(&args)
}

pub(super) fn package() -> Result<(), String> {
    ensure_clean_worktree()?;
    let metadata = cargo_metadata()?;
    let release_manifest = release_manifest_contract()?;
    for package in release_manifest.ordered_crates() {
        run_cargo(&["package", "-p", package, "--list"])?;
    }
    package_gate::run(&metadata, &release_manifest)
}

pub(super) fn package_consumer_smoke(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut target = "all".to_string();
    let mut cuda_runtime = false;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--target" => {
                target = args
                    .next()
                    .ok_or_else(|| "--target requires core, cuda, metal, or all".to_string())?;
            }
            "--cuda-runtime" => cuda_runtime = true,
            _ => return Err(package_consumer_usage()),
        }
    }
    let consumers = match target.as_str() {
        "core" => &["j2k"][..],
        "cuda" => &["j2k", "j2k-cuda"][..],
        "metal" => &["j2k", "j2k-metal", "j2k-mpsgraph"][..],
        "all" => &["j2k", "j2k-cuda", "j2k-metal", "j2k-mpsgraph"][..],
        _ => return Err(package_consumer_usage()),
    };
    if cuda_runtime && !consumers.contains(&"j2k-cuda") {
        return Err("--cuda-runtime requires --target cuda or all".to_string());
    }
    let metadata = cargo_metadata()?;
    let release_manifest = release_manifest_contract()?;
    package_gate::run_core_package_smoke(&metadata, &release_manifest, consumers, cuda_runtime)
}

fn package_consumer_usage() -> String {
    "usage: cargo xtask package-consumer-smoke [--target core|cuda|metal|all] [--cuda-runtime]"
        .to_string()
}

pub(super) fn j2k_ml_package_smoke() -> Result<(), String> {
    let metadata = cargo_metadata()?;
    let release_manifest = release_manifest_contract()?;
    package_gate::run_j2k_ml_package_smoke(&metadata, &release_manifest)
}

#[cfg(test)]
mod tests;
