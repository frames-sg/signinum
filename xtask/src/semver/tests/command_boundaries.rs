// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, ffi::OsString, fs, path::Path};

use super::super::review::load_review_config;
use super::super::{
    capture_command, current_api_snapshot, run_semver_checks, semver_check_args,
    validate_baseline_revision, verify_or_write_report, workspace_package_versions, Options,
    PackageApiDiff, ReleaseType, SnapshotKind, API_DIFF_REPORT, CARGO_PUBLIC_API_VERSION,
    HIDDEN_API_SNAPSHOT, PUBLIC_API_SNAPSHOT, SEMVER_BASELINE_COMMIT, SEMVER_BASELINE_PACKAGES,
    SEMVER_NEW_PACKAGES,
};
#[cfg(target_os = "macos")]
use super::super::{require_macos, semver};

fn workspace_path(relative: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest has workspace parent")
        .join(relative)
        .to_str()
        .expect("workspace path is UTF-8")
        .to_string()
}

#[test]
fn command_capture_preserves_success_failure_and_non_utf8_diagnostics() {
    assert_eq!(
        capture_command(
            OsString::from("/bin/sh"),
            &["-c", "printf '  success  \\n'"],
            &[],
            "capture success",
        ),
        Ok("success".to_string())
    );

    let error = capture_command(
        OsString::from("/bin/sh"),
        &["-c", "printf 'stdout'; printf 'stderr' >&2; exit 7"],
        &[],
        "capture failure",
    )
    .expect_err("nonzero command");
    assert!(error.contains("exit status: 7"));
    assert!(error.contains("stdout"));
    assert!(error.contains("stderr"));
    assert!(error.contains("capture failure"));

    let error = capture_command(
        OsString::from("/bin/sh"),
        &["-c", "printf '\\377'"],
        &[],
        "capture bytes",
    )
    .expect_err("non-UTF-8 stdout");
    assert!(error.contains("non-UTF-8 stdout"));
    assert!(error.contains("capture bytes"));
}

#[test]
fn committed_candidate_semver_inputs_match_the_pinned_workspace_contract() {
    let ordinary = current_api_snapshot(
        &workspace_path(PUBLIC_API_SNAPSHOT),
        SnapshotKind::Ordinary,
        CARGO_PUBLIC_API_VERSION,
    )
    .expect("ordinary API snapshot");
    let hidden = current_api_snapshot(
        &workspace_path(HIDDEN_API_SNAPSHOT),
        SnapshotKind::Hidden,
        CARGO_PUBLIC_API_VERSION,
    )
    .expect("hidden API snapshot");
    assert!(ordinary.starts_with("# J2K 1.0 Public API Snapshot"));
    assert!(hidden.starts_with("# J2K 1.0 Rustdoc-Hidden Public API Snapshot"));

    let versions = workspace_package_versions().expect("workspace package versions");
    assert_eq!(versions.get("j2k").map(String::as_str), Some("0.11.0"));
    assert!(versions.keys().collect::<BTreeSet<_>>().len() > 10);
}

#[test]
fn baseline_revision_validation_accepts_only_the_pinned_commit() {
    assert_eq!(validate_baseline_revision(SEMVER_BASELINE_COMMIT), Ok(()));
    let error = validate_baseline_revision("0000000000000000000000000000000000000000")
        .expect_err("mismatched baseline revision");
    assert!(error.contains(SEMVER_BASELINE_COMMIT));
}

#[test]
fn committed_review_config_covers_the_exact_release_scope() {
    let config = load_review_config().expect("load committed API review config");
    assert_eq!(config.candidate_version, "0.11.0");
    assert_eq!(config.break_ledger.len(), 1);
    assert_eq!(
        config
            .break_ledger
            .iter()
            .map(|entry| entry.removed_items.len())
            .sum::<usize>(),
        4
    );
    let expected_packages = SEMVER_BASELINE_PACKAGES
        .iter()
        .chain(SEMVER_NEW_PACKAGES)
        .map(|package| (*package).to_string())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        config.reviews.keys().cloned().collect::<BTreeSet<_>>(),
        expected_packages
    );
}

#[test]
fn report_verification_is_workspace_anchored_and_empty_checks_are_a_noop() {
    let committed =
        fs::read_to_string(workspace_path(API_DIFF_REPORT)).expect("read committed semver report");
    assert_eq!(
        verify_or_write_report(
            Options {
                write_report: false,
            },
            &committed,
        ),
        Ok(())
    );
    let error = verify_or_write_report(
        Options {
            write_report: false,
        },
        "stale report",
    )
    .expect_err("stale semver report");
    assert!(error.contains("is stale"));
    assert_eq!(run_semver_checks(&[]), Ok(()));
}

#[test]
fn semver_check_command_uses_the_computed_candidate_release_type() {
    let diff = PackageApiDiff {
        package: "j2k-core".to_string(),
        candidate_version: "0.11.0".to_string(),
        release_type: Some(ReleaseType::Major),
        baseline_count: 1,
        candidate_count: 0,
        added: BTreeSet::new(),
        removed: ["pub struct j2k_core::DecoderContext<C>".to_string()]
            .into_iter()
            .collect(),
        hidden: BTreeSet::new(),
    };

    assert_eq!(
        semver_check_args(&diff),
        [
            "run",
            "1.96",
            "cargo",
            "semver-checks",
            "check-release",
            "--package",
            "j2k-core",
            "--baseline-version",
            "0.10.0",
            "--release-type",
            "major",
            "--color",
            "never",
        ]
    );
}

#[cfg(target_os = "macos")]
#[test]
fn semver_top_level_rejects_a_mismatched_collector_pin_before_live_inventory() {
    require_macos().expect("macOS semver precondition");
    let error =
        semver(std::iter::empty(), &["j2k"], "0.0.0").expect_err("mismatched cargo-public-api pin");
    assert!(error.contains("requested cargo-public-api 0.0.0"));
    assert!(error.contains(CARGO_PUBLIC_API_VERSION));
}
