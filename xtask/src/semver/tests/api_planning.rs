// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use super::super::{
    build_package_diffs, common_candidate_version, render_report, snapshot_drift,
    validate_snapshot_scope, ReleaseType, Version,
};

fn items(values: &[&str]) -> BTreeSet<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

#[test]
fn common_candidate_version_rejects_empty_missing_and_mixed_sets() {
    let versions = BTreeMap::from([
        ("alpha".to_string(), "0.7.0".to_string()),
        ("beta".to_string(), "0.8.0".to_string()),
    ]);
    assert_eq!(
        common_candidate_version(&["alpha"], &versions),
        Ok("0.7.0".to_string())
    );
    assert!(common_candidate_version(&[], &versions)
        .expect_err("empty stable package set")
        .contains("share one candidate version"));
    assert!(common_candidate_version(&["missing"], &versions)
        .expect_err("missing version")
        .contains("missing published library"));
    assert!(common_candidate_version(&["alpha", "beta"], &versions)
        .expect_err("mixed versions")
        .contains("share one candidate version"));
}

#[test]
fn snapshot_scope_and_drift_report_missing_unexpected_and_changed_packages() {
    let snapshot = BTreeMap::from([
        ("alpha".to_string(), items(&["old"])),
        ("unexpected".to_string(), items(&["item"])),
    ]);
    let current = BTreeMap::from([
        ("alpha".to_string(), items(&["new"])),
        ("beta".to_string(), items(&["item"])),
    ]);

    assert_eq!(
        snapshot_drift(&["alpha", "beta"], &snapshot, &current),
        ["alpha", "beta", "unexpected:unexpected"]
    );
    let error = validate_snapshot_scope("candidate", &["alpha", "beta"], &snapshot)
        .expect_err("scope drift");
    assert!(error.contains("missing: [\"beta\"]"));
    assert!(error.contains("unexpected: [\"unexpected\"]"));
}

#[test]
fn package_diff_planning_distinguishes_published_and_new_packages() {
    let stable = ["j2k", "j2k-future"];
    let versions = BTreeMap::from([
        ("j2k".to_string(), "0.7.4".to_string()),
        ("j2k-future".to_string(), "0.7.4".to_string()),
    ]);
    let baseline = BTreeMap::from([("j2k".to_string(), items(&["keep", "removed"]))]);
    let current = BTreeMap::from([
        ("j2k".to_string(), items(&["keep", "added"])),
        ("j2k-future".to_string(), items(&["new api"])),
    ]);
    let hidden = BTreeMap::from([
        ("j2k".to_string(), items(&["hidden j2k"])),
        ("j2k-future".to_string(), items(&["hidden future"])),
    ]);

    let diffs = build_package_diffs(
        &stable,
        &versions,
        Version::parse("0.7.3").expect("baseline"),
        &baseline,
        &current,
        &hidden,
    )
    .expect("package diffs");

    assert_eq!(diffs.len(), 2);
    assert_eq!(diffs[0].release_type, Some(ReleaseType::Minor));
    assert_eq!(diffs[0].added, items(&["added"]));
    assert_eq!(diffs[0].removed, items(&["removed"]));
    assert_eq!(diffs[1].release_type, None);
    assert_eq!(diffs[1].added, items(&["new api"]));
    assert!(diffs[1].removed.is_empty());

    let report = render_report("0.7.4", &diffs, "0.52.0");
    assert!(report.contains("## New packages without a 0.11.0 registry baseline"));
    assert!(report.contains("- `j2k-future` `0.7.4`: 1 ordinary public API items"));
    assert!(report.contains("### `j2k`"));
    assert!(report.contains("```text\nremoved\n```"));
    assert!(report.contains("```text\nadded\n```"));
}

#[test]
fn package_diff_planning_reports_each_missing_inventory_boundary() {
    let versions = BTreeMap::from([("j2k".to_string(), "0.7.0".to_string())]);
    let baseline = BTreeMap::from([("j2k".to_string(), BTreeSet::new())]);
    let current = BTreeMap::from([("j2k".to_string(), BTreeSet::new())]);
    let hidden = BTreeMap::from([("j2k".to_string(), BTreeSet::new())]);
    let baseline_version = Version::parse("0.7.3").expect("baseline");

    for (versions, baseline, current, hidden, expected) in [
        (
            BTreeMap::new(),
            baseline.clone(),
            current.clone(),
            hidden.clone(),
            "workspace metadata is missing",
        ),
        (
            versions.clone(),
            BTreeMap::new(),
            current.clone(),
            hidden.clone(),
            "baseline package",
        ),
        (
            versions.clone(),
            baseline.clone(),
            BTreeMap::new(),
            hidden.clone(),
            "current public API snapshot is missing",
        ),
        (
            versions.clone(),
            baseline.clone(),
            current.clone(),
            BTreeMap::new(),
            "rustdoc-hidden API inventory is missing",
        ),
    ] {
        let error = build_package_diffs(
            &["j2k"],
            &versions,
            baseline_version,
            &baseline,
            &current,
            &hidden,
        )
        .expect_err("missing inventory boundary");
        assert!(error.contains(expected), "unexpected error: {error}");
    }
}
