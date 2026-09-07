// SPDX-License-Identifier: MIT OR Apache-2.0

//! Release-version and reviewed public-API compatibility gates.

mod compatibility;
mod review;

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    ffi::OsString,
    fmt::Write as _,
    fs,
    path::Path,
};

use crate::process::{self, cargo, CommandContext};
use crate::stable_api::{
    collect_package_apis, verify_cargo_public_api_version, CARGO_PUBLIC_API_VERSION,
    HIDDEN_API_SNAPSHOT, PUBLIC_API_SNAPSHOT, PUBLIC_API_TARGET, PUBLIC_API_TOOLCHAIN,
};
use compatibility::run_semver_checks;
#[cfg(test)]
use compatibility::{semver_check_args, semver_check_release_type};

const CARGO_SEMVER_CHECKS_VERSION: &str = "0.48.0";
const SEMVER_TOOLCHAIN: &str = "1.96";
const SEMVER_BASELINE_VERSION: &str = "0.11.0";
const SEMVER_BASELINE_TAG: &str = "v0.11.0";
const SEMVER_BASELINE_COMMIT: &str = "09d746a7b040258eb5dd505b44384eec0152a8b9";
const API_DIFF_REPORT: &str = "docs/release-evidence/public-api/reviewed-public-api-diff-0.11.1.md";
const API_REVIEW_CONFIG: &str = "docs/release-evidence/public-api/public-api-review-0.11.1.yml";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BaselineTransition<'a> {
    candidate_version: &'a str,
    required_next_baseline_version: &'a str,
    required_next_baseline_tag: &'a str,
}

const INTENTIONAL_BREAK_TRANSITION: Option<BaselineTransition<'static>> = None;

const SEMVER_BASELINE_PACKAGES: &[&str] = &[
    "j2k",
    "j2k-core",
    "j2k-codec-math",
    "j2k-jpeg",
    "j2k-tilecodec",
    "j2k-jpeg-metal",
    "j2k-metal",
    "j2k-jpeg-cuda",
    "j2k-cuda",
    "j2k-transcode",
    "j2k-transcode-cuda",
    "j2k-metal-support",
    "j2k-transcode-metal",
    "j2k-native",
    "j2k-types",
    "j2k-cuda-runtime",
    "j2k-profile",
    "j2k-ml",
    "j2k-cuda-build-support",
    "j2k-cuda-j2k-engine",
    "j2k-cuda-jpeg-engine",
    "j2k-cuda-transcode-engine",
    "j2k-mpsgraph",
    "j2k-mpsgraph-support",
];

const SEMVER_NEW_PACKAGES: &[&str] = &[];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct Version {
    major: u64,
    minor: u64,
    patch: u64,
}

impl Version {
    fn parse(value: &str) -> Result<Self, String> {
        if value.contains(['-', '+']) {
            return Err(format!(
                "release version `{value}` must not contain prerelease or build metadata"
            ));
        }
        let mut parts = value.split('.');
        let major = parse_version_component(parts.next(), value, "major")?;
        let minor = parse_version_component(parts.next(), value, "minor")?;
        let patch = parse_version_component(parts.next(), value, "patch")?;
        if parts.next().is_some() {
            return Err(format!(
                "release version `{value}` must have exactly three components"
            ));
        }
        Ok(Self {
            major,
            minor,
            patch,
        })
    }
}

fn parse_version_component(
    component: Option<&str>,
    version: &str,
    label: &str,
) -> Result<u64, String> {
    component
        .ok_or_else(|| format!("release version `{version}` is missing its {label} component"))?
        .parse::<u64>()
        .map_err(|err| {
            format!("release version `{version}` has an invalid {label} component: {err}")
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReleaseType {
    Major,
    Minor,
    Patch,
}

impl ReleaseType {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Major => "major",
            Self::Minor => "minor",
            Self::Patch => "patch",
        }
    }
}

fn computed_release_type(baseline: Version, candidate: Version) -> Result<ReleaseType, String> {
    if candidate <= baseline {
        return Err(format!(
            "candidate version {candidate:?} must be newer than baseline {baseline:?}"
        ));
    }

    if candidate.major != baseline.major {
        return Ok(ReleaseType::Major);
    }
    if candidate.minor != baseline.minor {
        return Ok(if candidate.major == 0 {
            ReleaseType::Major
        } else {
            ReleaseType::Minor
        });
    }
    Ok(match (candidate.major, candidate.minor) {
        (0, 0) => ReleaseType::Major,
        (0, _) => ReleaseType::Minor,
        (_, _) => ReleaseType::Patch,
    })
}

#[derive(Debug)]
struct PackageApiDiff {
    package: String,
    candidate_version: String,
    release_type: Option<ReleaseType>,
    baseline_count: usize,
    candidate_count: usize,
    added: BTreeSet<String>,
    removed: BTreeSet<String>,
    hidden: BTreeSet<String>,
}

impl PackageApiDiff {
    fn added_fingerprint(&self) -> String {
        fingerprint(&self.added)
    }

    fn removed_fingerprint(&self) -> String {
        fingerprint(&self.removed)
    }

    fn hidden_fingerprint(&self) -> String {
        fingerprint(&self.hidden)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Options {
    write_report: bool,
}

pub(crate) fn semver(
    args: impl Iterator<Item = String>,
    published_library_packages: &[&str],
    cargo_public_api_version: &str,
) -> Result<(), String> {
    let options = parse_options(args)?;
    require_macos()?;
    verify_tool_versions(cargo_public_api_version)?;
    verify_baseline_tag()?;
    validate_package_partition(published_library_packages)?;

    let versions = workspace_package_versions()?;
    let candidate_version = common_candidate_version(published_library_packages, &versions)?;
    validate_baseline_transition(
        SEMVER_BASELINE_VERSION,
        &candidate_version,
        INTENTIONAL_BREAK_TRANSITION,
    )?;
    let baseline = Version::parse(SEMVER_BASELINE_VERSION)?;
    let baseline_snapshot = baseline_api_snapshot(cargo_public_api_version)?;
    let baseline_apis = parse_api_snapshot(&baseline_snapshot)?;
    validate_snapshot_scope(
        "published baseline ordinary snapshot",
        SEMVER_BASELINE_PACKAGES,
        &baseline_apis,
    )?;
    let ordinary_snapshot = current_api_snapshot(
        PUBLIC_API_SNAPSHOT,
        SnapshotKind::Ordinary,
        cargo_public_api_version,
    )?;
    let hidden_snapshot = current_api_snapshot(
        HIDDEN_API_SNAPSHOT,
        SnapshotKind::Hidden,
        cargo_public_api_version,
    )?;
    let snapshot_apis = parse_api_snapshot(&ordinary_snapshot)?;
    let snapshot_hidden_apis = parse_api_snapshot(&hidden_snapshot)?;
    validate_snapshot_scope(
        "candidate ordinary snapshot",
        published_library_packages,
        &snapshot_apis,
    )?;
    validate_snapshot_scope(
        "candidate rustdoc-hidden snapshot",
        published_library_packages,
        &snapshot_hidden_apis,
    )?;
    let live_inventories = collect_package_apis(published_library_packages)?;
    let current_apis = live_inventories
        .iter()
        .map(|(package, inventory)| (package.clone(), inventory.ordinary.clone()))
        .collect::<BTreeMap<_, _>>();
    let current_hidden_apis = live_inventories
        .iter()
        .map(|(package, inventory)| (package.clone(), inventory.hidden.clone()))
        .collect::<BTreeMap<_, _>>();
    let stale_ordinary_packages =
        snapshot_drift(published_library_packages, &snapshot_apis, &current_apis);
    let stale_hidden_packages = snapshot_drift(
        published_library_packages,
        &snapshot_hidden_apis,
        &current_hidden_apis,
    );

    let diffs = build_package_diffs(
        published_library_packages,
        &versions,
        baseline,
        &baseline_apis,
        &current_apis,
        &current_hidden_apis,
    )?;

    if !stale_ordinary_packages.is_empty() || !stale_hidden_packages.is_empty() {
        return Err(format!(
            "committed published-library API snapshots are stale; ordinary packages: \
             {stale_ordinary_packages:?}; rustdoc-hidden packages: {stale_hidden_packages:?}; \
             run `cargo xtask stable-api --write` and review both inventory diffs before \
             regenerating the semver report"
        ));
    }

    let report = render_report(&candidate_version, &diffs, cargo_public_api_version);
    verify_or_write_report(options, &report)?;

    let review_config = review::load_review_config()?;
    review::validate_reviews(&review_config, &candidate_version, &diffs)?;
    run_semver_checks(&diffs)?;
    Ok(())
}

fn validate_baseline_transition(
    baseline_version: &str,
    candidate_version: &str,
    transition: Option<BaselineTransition<'_>>,
) -> Result<(), String> {
    let Some(transition) = transition else {
        return Ok(());
    };
    if transition.required_next_baseline_version != transition.candidate_version {
        return Err(format!(
            "intentional-break transition candidate {} must equal its required next baseline \
             version {}, so later candidate checks cannot keep using the older baseline",
            transition.candidate_version, transition.required_next_baseline_version
        ));
    }
    let expected_tag = format!("v{}", transition.required_next_baseline_version);
    if transition.required_next_baseline_tag != expected_tag {
        return Err(format!(
            "intentional-break transition requires next baseline tag `{expected_tag}`, found `{}`",
            transition.required_next_baseline_tag
        ));
    }
    let baseline = Version::parse(baseline_version)?;
    let transition_candidate = Version::parse(transition.candidate_version)?;
    computed_release_type(baseline, transition_candidate)?;
    if candidate_version != transition.candidate_version {
        return Err(format!(
            "the active intentional-break transition permits only candidate {} against baseline \
             {baseline_version}, found {candidate_version}; after publishing {}, rotate the \
             semver baseline to {} and disable INTENTIONAL_BREAK_TRANSITION before preparing a \
             later release",
            transition.candidate_version,
            transition.candidate_version,
            transition.required_next_baseline_tag
        ));
    }
    Ok(())
}

fn build_package_diffs(
    published_library_packages: &[&str],
    versions: &BTreeMap<String, String>,
    baseline: Version,
    baseline_apis: &BTreeMap<String, BTreeSet<String>>,
    current_apis: &BTreeMap<String, BTreeSet<String>>,
    current_hidden_apis: &BTreeMap<String, BTreeSet<String>>,
) -> Result<Vec<PackageApiDiff>, String> {
    let baseline_set = SEMVER_BASELINE_PACKAGES
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut diffs = Vec::with_capacity(published_library_packages.len());
    for package in published_library_packages {
        let current = current_apis.get(*package).cloned().ok_or_else(|| {
            format!("current public API snapshot is missing published library `{package}`")
        })?;
        let candidate = versions.get(*package).ok_or_else(|| {
            format!("workspace metadata is missing published library `{package}`")
        })?;
        let hidden = current_hidden_apis.get(*package).cloned().ok_or_else(|| {
            format!("current rustdoc-hidden API inventory is missing published library `{package}`")
        })?;
        if baseline_set.contains(package) {
            let baseline_api = baseline_apis.get(*package).ok_or_else(|| {
                format!(
                    "{SEMVER_BASELINE_TAG} public API snapshot is missing baseline package `{package}`"
                )
            })?;
            let release_type = computed_release_type(baseline, Version::parse(candidate)?)?;
            diffs.push(PackageApiDiff {
                package: (*package).to_string(),
                candidate_version: candidate.clone(),
                release_type: Some(release_type),
                baseline_count: baseline_api.len(),
                candidate_count: current.len(),
                added: current.difference(baseline_api).cloned().collect(),
                removed: baseline_api.difference(&current).cloned().collect(),
                hidden,
            });
            continue;
        }
        diffs.push(PackageApiDiff {
            package: (*package).to_string(),
            candidate_version: candidate.clone(),
            release_type: None,
            baseline_count: 0,
            candidate_count: current.len(),
            added: current,
            removed: BTreeSet::new(),
            hidden,
        });
    }
    Ok(diffs)
}

fn parse_options(args: impl Iterator<Item = String>) -> Result<Options, String> {
    let mut write_report = false;
    for arg in args {
        match arg.as_str() {
            "--write-report" if !write_report => write_report = true,
            "--write-report" => {
                return Err("duplicate semver argument `--write-report`".to_string())
            }
            other => return Err(format!("unknown semver argument `{other}`")),
        }
    }
    Ok(Options { write_report })
}

fn require_macos() -> Result<(), String> {
    if env::consts::OS == "macos" {
        Ok(())
    } else {
        Err(format!(
            "semver/API review must run on macOS so Metal public APIs are included; current host is {}",
            env::consts::OS
        ))
    }
}

fn verify_baseline_tag() -> Result<(), String> {
    let revision = capture_command(
        OsString::from("git"),
        &["rev-parse", &format!("{SEMVER_BASELINE_TAG}^{{commit}}")],
        &[],
        "resolve semver baseline tag",
    )?;
    validate_baseline_revision(revision.trim())
}

fn validate_baseline_revision(revision: &str) -> Result<(), String> {
    if revision == SEMVER_BASELINE_COMMIT {
        Ok(())
    } else {
        Err(format!(
            "{SEMVER_BASELINE_TAG} must peel to {SEMVER_BASELINE_COMMIT}, found `{revision}`"
        ))
    }
}

fn verify_tool_versions(cargo_public_api_version: &str) -> Result<(), String> {
    if cargo_public_api_version != CARGO_PUBLIC_API_VERSION {
        return Err(format!(
            "semver requested cargo-public-api {cargo_public_api_version}, but the stable API \
             collector is pinned to {CARGO_PUBLIC_API_VERSION}"
        ));
    }
    if env::var_os("J2K_SEMVER_TOOLCHAIN").is_some() {
        return Err(format!(
            "J2K_SEMVER_TOOLCHAIN overrides are not accepted; semver is pinned to Rust \
             {SEMVER_TOOLCHAIN}"
        ));
    }
    let args = semver_cargo_args(["semver-checks", "--version"]);
    let args = args.iter().map(String::as_str).collect::<Vec<_>>();
    let semver_version = capture_command(
        OsString::from("rustup"),
        &args,
        &[],
        "detect cargo-semver-checks",
    )?;
    require_version_token(
        &semver_version,
        "cargo-semver-checks",
        CARGO_SEMVER_CHECKS_VERSION,
    )?;
    verify_cargo_public_api_version()
}

fn semver_cargo_args<'a>(args: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    ["run", SEMVER_TOOLCHAIN, "cargo"]
        .into_iter()
        .chain(args)
        .map(str::to_string)
        .collect()
}

fn require_version_token(output: &str, tool: &str, expected: &str) -> Result<(), String> {
    let mut words = output.split_whitespace();
    let actual_tool = words.next().unwrap_or_default();
    let actual_version = words.next().unwrap_or_default();
    if actual_tool == tool && actual_version == expected {
        Ok(())
    } else {
        Err(format!(
            "{tool} version must be {expected}; found `{}`",
            output.trim()
        ))
    }
}

fn validate_package_partition(published_library_packages: &[&str]) -> Result<(), String> {
    let published = published_library_packages
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if published.len() != published_library_packages.len() {
        return Err("published library package set contains duplicates".to_string());
    }
    let baseline = SEMVER_BASELINE_PACKAGES
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let new = SEMVER_NEW_PACKAGES.iter().copied().collect::<BTreeSet<_>>();
    if baseline.len() != SEMVER_BASELINE_PACKAGES.len() {
        return Err("SEMVER_BASELINE_PACKAGES contains duplicates".to_string());
    }
    if new.len() != SEMVER_NEW_PACKAGES.len() {
        return Err("SEMVER_NEW_PACKAGES contains duplicates".to_string());
    }
    let overlap = baseline.intersection(&new).copied().collect::<Vec<_>>();
    if !overlap.is_empty() {
        return Err(format!(
            "semver baseline/new package lists overlap: {overlap:?}"
        ));
    }
    let configured = baseline.union(&new).copied().collect::<BTreeSet<_>>();
    if configured == published {
        Ok(())
    } else {
        let missing = published
            .difference(&configured)
            .copied()
            .collect::<Vec<_>>();
        let unexpected = configured
            .difference(&published)
            .copied()
            .collect::<Vec<_>>();
        Err(format!(
            "published-library semver baseline/new partition drifted; missing: {missing:?}; unexpected: {unexpected:?}"
        ))
    }
}

fn workspace_package_versions() -> Result<BTreeMap<String, String>, String> {
    let output = capture_command(
        cargo(),
        &["metadata", "--locked", "--no-deps", "--format-version=1"],
        &[],
        "read workspace package versions",
    )?;
    let metadata: serde_json::Value =
        serde_json::from_str(&output).map_err(|err| format!("parse cargo metadata JSON: {err}"))?;
    let packages = metadata
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "cargo metadata did not contain a packages array".to_string())?;
    Ok(packages
        .iter()
        .filter_map(|package| {
            Some((
                package.get("name")?.as_str()?.to_string(),
                package.get("version")?.as_str()?.to_string(),
            ))
        })
        .collect())
}

fn common_candidate_version(
    published_library_packages: &[&str],
    versions: &BTreeMap<String, String>,
) -> Result<String, String> {
    let mut candidates = published_library_packages
        .iter()
        .map(|package| {
            versions.get(*package).cloned().ok_or_else(|| {
                format!("workspace metadata is missing published library `{package}`")
            })
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    if candidates.len() != 1 {
        return Err(format!(
            "published library packages must share one candidate version, found {candidates:?}"
        ));
    }
    candidates
        .pop_first()
        .ok_or_else(|| "published library package list is empty".to_string())
}

fn baseline_api_snapshot(cargo_public_api_version: &str) -> Result<String, String> {
    let object = format!("{SEMVER_BASELINE_TAG}:docs/stable-api-1.0.public-api.txt");
    let snapshot = capture_command(
        OsString::from("git"),
        &["show", object.as_str()],
        &[],
        "read tagged public API snapshot",
    )?;
    if snapshot_uses_generator(&snapshot, cargo_public_api_version) {
        Ok(snapshot)
    } else {
        Err(format!(
            "{SEMVER_BASELINE_TAG} public API snapshot was not generated by cargo-public-api {cargo_public_api_version}"
        ))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SnapshotKind {
    Ordinary,
    Hidden,
}

impl SnapshotKind {
    const fn header(self) -> &'static str {
        match self {
            Self::Ordinary => "# J2K 1.0 Public API Snapshot",
            Self::Hidden => "# J2K 1.0 Rustdoc-Hidden Public API Snapshot",
        }
    }
}

fn current_api_snapshot(
    path: &str,
    kind: SnapshotKind,
    cargo_public_api_version: &str,
) -> Result<String, String> {
    let snapshot = fs::read_to_string(path).map_err(|err| format!("read {path}: {err}"))?;
    if current_snapshot_uses_generation_contract(&snapshot, kind, cargo_public_api_version) {
        Ok(snapshot)
    } else {
        Err(format!(
            "{path} does not record the pinned cargo-public-api {cargo_public_api_version}, \
             rustdoc toolchain {PUBLIC_API_TOOLCHAIN}, and target {PUBLIC_API_TARGET} contract"
        ))
    }
}

fn snapshot_uses_generator(snapshot: &str, cargo_public_api_version: &str) -> bool {
    let expected = format!("Generator: `cargo-public-api {cargo_public_api_version}`.");
    let mut lines = snapshot.lines();
    lines.next() == Some("# J2K 1.0 Public API Snapshot")
        && exact_metadata_line(&lines.take(9).collect::<Vec<_>>(), "Generator:", &expected)
}

fn current_snapshot_uses_generation_contract(
    snapshot: &str,
    kind: SnapshotKind,
    cargo_public_api_version: &str,
) -> bool {
    let generator = format!("Generator: `cargo-public-api {cargo_public_api_version}`.");
    let toolchain = format!("Rustdoc toolchain: `{PUBLIC_API_TOOLCHAIN}`.");
    let target = format!("Target: `{PUBLIC_API_TARGET}`.");
    let mut lines = snapshot.lines();
    if lines.next() != Some(kind.header()) {
        return false;
    }
    let metadata = lines.take(24).collect::<Vec<_>>();
    exact_metadata_line(&metadata, "Generator:", &generator)
        && exact_metadata_line(&metadata, "Rustdoc toolchain:", &toolchain)
        && exact_metadata_line(&metadata, "Target:", &target)
}

fn exact_metadata_line(lines: &[&str], prefix: &str, expected: &str) -> bool {
    let mut matching = lines
        .iter()
        .copied()
        .filter(|line| line.starts_with(prefix));
    matching.next() == Some(expected) && matching.next().is_none()
}

fn snapshot_drift(
    published_library_packages: &[&str],
    snapshot_apis: &BTreeMap<String, BTreeSet<String>>,
    current_apis: &BTreeMap<String, BTreeSet<String>>,
) -> Vec<String> {
    let expected = published_library_packages
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut drift = published_library_packages
        .iter()
        .copied()
        .filter(|package| snapshot_apis.get(*package) != current_apis.get(*package))
        .map(str::to_string)
        .collect::<Vec<_>>();
    drift.extend(
        snapshot_apis
            .keys()
            .filter(|package| !expected.contains(package.as_str()))
            .map(|package| format!("unexpected:{package}")),
    );
    drift
}

fn validate_snapshot_scope(
    label: &str,
    expected_packages: &[&str],
    snapshot_apis: &BTreeMap<String, BTreeSet<String>>,
) -> Result<(), String> {
    let expected = expected_packages.iter().copied().collect::<BTreeSet<_>>();
    let actual = snapshot_apis
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if expected == actual {
        return Ok(());
    }
    let missing = expected.difference(&actual).copied().collect::<Vec<_>>();
    let unexpected = actual.difference(&expected).copied().collect::<Vec<_>>();
    Err(format!(
        "{label} package scope drifted; missing: {missing:?}; unexpected: {unexpected:?}"
    ))
}

fn parse_api_snapshot(snapshot: &str) -> Result<BTreeMap<String, BTreeSet<String>>, String> {
    let mut sections = BTreeMap::<String, BTreeSet<String>>::new();
    let mut package = None::<String>;
    let mut in_api = false;
    for line in snapshot.lines() {
        if let Some(name) = line
            .strip_prefix("## `")
            .and_then(|rest| rest.strip_suffix('`'))
        {
            if in_api {
                return Err(
                    "public API snapshot starts a package heading inside a text fence".to_string(),
                );
            }
            package = Some(name.to_string());
            continue;
        }
        if line == "```text" {
            if in_api {
                return Err("public API snapshot contains nested text fences".to_string());
            }
            let name = package.as_ref().ok_or_else(|| {
                "public API snapshot contains a text fence before a package heading".to_string()
            })?;
            if sections.insert(name.clone(), BTreeSet::new()).is_some() {
                return Err(format!(
                    "public API snapshot contains duplicate API section for `{name}`"
                ));
            }
            in_api = true;
            continue;
        }
        if line == "```" {
            if !in_api {
                return Err("public API snapshot contains an unmatched closing fence".to_string());
            }
            in_api = false;
            continue;
        }
        if in_api && !line.is_empty() {
            let name = package
                .as_ref()
                .expect("an API fence cannot open without a package heading");
            sections
                .get_mut(name)
                .expect("an open API fence must have an initialized package section")
                .insert(line.to_string());
        }
    }
    if in_api {
        Err("public API snapshot ends inside a text fence".to_string())
    } else if sections.is_empty() {
        Err("public API snapshot did not contain package API sections".to_string())
    } else {
        Ok(sections)
    }
}

fn fingerprint(items: &BTreeSet<String>) -> String {
    if items.is_empty() {
        return "none".to_string();
    }
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for item in items {
        for byte in item.as_bytes().iter().copied().chain([b'\n']) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("fnv1a64:{hash:016x}")
}

#[expect(
    clippy::too_many_lines,
    reason = "the reviewed API report is one stable ordered Markdown schema"
)]
fn render_report(
    candidate_version: &str,
    diffs: &[PackageApiDiff],
    cargo_public_api_version: &str,
) -> String {
    let mut out = String::new();
    writeln!(
        &mut out,
        "# Reviewed public API diff for j2k {candidate_version}"
    )
    .unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "This report is generated by `cargo xtask semver --write-report`. Normal `cargo xtask semver` regenerates it in memory and fails if this committed file is stale. Every ordinary added/removed fingerprint and every full rustdoc-hidden candidate-inventory fingerprint requires an exact reviewed entry in `{API_REVIEW_CONFIG}`; report regeneration never updates that review config."
    )
    .unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "- Baseline registry version: `{SEMVER_BASELINE_VERSION}`"
    )
    .unwrap();
    writeln!(
        &mut out,
        "- Baseline source snapshot: `{SEMVER_BASELINE_TAG}` peeled to `{SEMVER_BASELINE_COMMIT}`"
    )
    .unwrap();
    writeln!(&mut out, "- Candidate version: `{candidate_version}`").unwrap();
    if let Some(transition) = INTENTIONAL_BREAK_TRANSITION {
        writeln!(
            &mut out,
            "- Active intentional-break transition: only `{}` may compare against \
             `{SEMVER_BASELINE_TAG}`",
            transition.candidate_version
        )
        .unwrap();
        writeln!(
            &mut out,
            "- Required next semver baseline: `{}` at version `{}` before any later candidate",
            transition.required_next_baseline_tag, transition.required_next_baseline_version
        )
        .unwrap();
    }
    writeln!(
        &mut out,
        "- Tool pins: Rust `{SEMVER_TOOLCHAIN}`, `cargo-semver-checks {CARGO_SEMVER_CHECKS_VERSION}`, `cargo-public-api {cargo_public_api_version}`, rustdoc `{PUBLIC_API_TOOLCHAIN}`, target `{PUBLIC_API_TARGET}`"
    )
    .unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "## Summary").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "| Package | Baseline | Candidate | Computed release type | Added | Removed/changed | Removed fingerprint | Added fingerprint | Rustdoc-hidden items | Hidden inventory fingerprint |"
    )
    .unwrap();
    writeln!(
        &mut out,
        "| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | --- |"
    )
    .unwrap();
    for diff in diffs {
        let baseline = if diff.release_type.is_some() {
            SEMVER_BASELINE_VERSION
        } else {
            "new/unpublished"
        };
        let release_type = diff.release_type.map_or("new", ReleaseType::as_str);
        writeln!(
            &mut out,
            "| `{}` | `{baseline}` | `{}` | `{release_type}` | {} | {} | `{}` | `{}` | {} | `{}` |",
            diff.package,
            diff.candidate_version,
            diff.added.len(),
            diff.removed.len(),
            diff.removed_fingerprint(),
            diff.added_fingerprint(),
            diff.hidden.len(),
            diff.hidden_fingerprint(),
        )
        .unwrap();
    }

    if diffs.iter().any(|diff| diff.release_type.is_none()) {
        writeln!(&mut out).unwrap();
        writeln!(
            &mut out,
            "## New packages without a {SEMVER_BASELINE_VERSION} registry baseline"
        )
        .unwrap();
        writeln!(&mut out).unwrap();
        for diff in diffs.iter().filter(|diff| diff.release_type.is_none()) {
            writeln!(
                &mut out,
                "- `{}` `{}`: {} ordinary public API items, fingerprint `{}`; {} rustdoc-hidden public API items, full-inventory fingerprint `{}`.",
                diff.package,
                diff.candidate_version,
                diff.candidate_count,
                diff.added_fingerprint(),
                diff.hidden.len(),
                diff.hidden_fingerprint()
            )
            .unwrap();
        }
    }

    writeln!(&mut out).unwrap();
    writeln!(&mut out, "## Published-package details").unwrap();
    for diff in diffs.iter().filter(|diff| diff.release_type.is_some()) {
        writeln!(&mut out).unwrap();
        writeln!(&mut out, "### `{}`", diff.package).unwrap();
        writeln!(&mut out).unwrap();
        writeln!(
            &mut out,
            "Baseline items: {}. Candidate items: {}. Computed release type: `{}`. Rustdoc-hidden candidate items: {}. Full hidden-inventory fingerprint: `{}`.",
            diff.baseline_count,
            diff.candidate_count,
            diff.release_type
                .expect("published diff release type")
                .as_str(),
            diff.hidden.len(),
            diff.hidden_fingerprint()
        )
        .unwrap();
        render_diff_items(
            &mut out,
            "Removed or changed baseline API items",
            &diff.removed,
        );
        render_diff_items(&mut out, "Added candidate API items", &diff.added);
    }
    out
}

fn render_diff_items(out: &mut String, heading: &str, items: &BTreeSet<String>) {
    writeln!(out).unwrap();
    writeln!(out, "#### {heading}").unwrap();
    writeln!(out).unwrap();
    if items.is_empty() {
        writeln!(out, "None.").unwrap();
        return;
    }
    writeln!(out, "```text").unwrap();
    for item in items {
        writeln!(out, "{item}").unwrap();
    }
    writeln!(out, "```").unwrap();
}

fn verify_or_write_report(options: Options, rendered: &str) -> Result<(), String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| "xtask manifest directory has no workspace parent".to_string())?
        .join(API_DIFF_REPORT);
    if options.write_report {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("create {}: {err}", parent.display()))?;
        }
        fs::write(&path, rendered).map_err(|err| format!("write {}: {err}", path.display()))?;
        eprintln!(
            "wrote {}; review its diff before editing {API_REVIEW_CONFIG}",
            path.display()
        );
        return Ok(());
    }

    let committed = fs::read_to_string(&path).map_err(|err| {
        format!(
            "read {}: {err}; run `cargo xtask semver --write-report` and review the generated diff",
            path.display()
        )
    })?;
    if committed == rendered {
        Ok(())
    } else {
        Err(format!(
            "{} is stale; run `cargo xtask semver --write-report`, review the diff, and update {API_REVIEW_CONFIG} only for intentional API changes",
            path.display()
        ))
    }
}

fn capture_command(
    program: impl AsRef<std::ffi::OsStr>,
    args: &[&str],
    envs: &[(&str, &str)],
    label: &str,
) -> Result<String, String> {
    let program = program.as_ref();
    eprintln!("+ {} {}", program.to_string_lossy(), args.join(" "));
    let output = process::command_output(program, args, CommandContext::new().envs(envs))?;
    if output.status.success() {
        return String::from_utf8(output.stdout)
            .map(|stdout| stdout.trim().to_string())
            .map_err(|error| {
                format!(
                    "{} emitted non-UTF-8 stdout while attempting to {label}: {error}",
                    program.to_string_lossy()
                )
            });
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(format!(
        "{} exited with {} while attempting to {label}:\n{}",
        program.to_string_lossy(),
        output.status,
        format!("{stdout}\n{stderr}").trim()
    ))
}

#[cfg(test)]
mod tests;
