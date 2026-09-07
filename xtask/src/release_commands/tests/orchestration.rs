// SPDX-License-Identifier: MIT OR Apache-2.0

use std::os::unix::fs::symlink;

use super::super::release_manifest::parse_release_manifest_source;
use super::{
    super::{package, release_integrity},
    integrity::{complete_publishable_metadata, metadata_program},
    package_fixture::packaged_metadata,
};
use crate::test_command::RecordingProgram;

const INTEGRITY_CHILD_ENV: &str = "XTASK_TEST_RELEASE_INTEGRITY_WORKSPACE_CHILD";
const PACKAGE_CHILD_ENV: &str = "XTASK_TEST_PACKAGE_WORKSPACE_CHILD";

fn run_test_in_dir(test_name: &str, recording: &RecordingProgram, current_dir: &std::path::Path) {
    let output = std::process::Command::new(std::env::current_exe().expect("current test binary"))
        .arg(test_name)
        .arg("--exact")
        .arg("--nocapture")
        .current_dir(current_dir)
        .env(INTEGRITY_CHILD_ENV, "1")
        .env("CARGO", recording.program())
        .output()
        .expect("run release command test in isolated directory");
    assert!(
        output.status.success(),
        "workspace child failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn release_integrity_publish_mode_accepts_hermetic_final_metadata() {
    let metadata = complete_publishable_metadata();
    let version = metadata["packages"][0]["version"]
        .as_str()
        .expect("publishable fixture version");
    let recording = metadata_program("release-integrity-publish-metadata", &metadata);
    if std::env::var_os(INTEGRITY_CHILD_ENV).is_some() {
        release_integrity(["--publish".to_string()].into_iter())
            .expect("finalized release metadata must pass publish mode");
        return;
    }

    let release_root = recording
        .program()
        .parent()
        .expect("recording program parent")
        .join("release-root");
    for directory in [
        release_root.join(".github/workflows"),
        release_root.join("docs"),
        release_root.join("scripts"),
    ] {
        std::fs::create_dir_all(directory).expect("create hermetic release directory");
    }
    for (path, source) in [
        (
            ".github/workflows/publish.yml",
            include_str!("../../../../.github/workflows/publish.yml"),
        ),
        (
            "docs/release.md",
            include_str!("../../../../docs/release.md"),
        ),
        (
            "scripts/publish-crate.sh",
            include_str!("../../../../scripts/publish-crate.sh"),
        ),
        (
            "scripts/publish_release.py",
            include_str!("../../../../scripts/publish_release.py"),
        ),
        (
            "scripts/release_manifest.py",
            include_str!("../../../../scripts/release_manifest.py"),
        ),
        (
            "release-crates.json",
            include_str!("../../../../release-crates.json"),
        ),
    ] {
        std::fs::write(release_root.join(path), source).expect("write release contract fixture");
    }
    std::fs::write(
        release_root.join("Cargo.toml"),
        format!("[workspace.package]\nversion = \"{version}\"\n"),
    )
    .expect("write workspace manifest fixture");
    std::fs::write(
        release_root.join("CHANGELOG.md"),
        format!("# Changelog\n\n## [{version}] - 2026-08-13\n"),
    )
    .expect("write finalized changelog fixture");

    run_test_in_dir(
        "release_commands::tests::orchestration::release_integrity_publish_mode_accepts_hermetic_final_metadata",
        &recording,
        &release_root,
    );
}

#[test]
fn package_command_executes_list_and_dependency_aware_gates_hermetically() {
    if std::env::var_os(PACKAGE_CHILD_ENV).is_some() {
        package().expect("hermetic package command");
        return;
    }

    let (metadata, package_target) = packaged_metadata();
    let cargo = metadata_program("release-package-cargo", &metadata);

    let git = RecordingProgram::new("release-package-git", "");
    let program_dir = git.program().parent().expect("fake git parent");
    symlink(git.program(), program_dir.join("git")).expect("fake git symlink");
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask workspace root");
    let output = std::process::Command::new(std::env::current_exe().expect("current test binary"))
        .arg("release_commands::tests::orchestration::package_command_executes_list_and_dependency_aware_gates_hermetically")
        .arg("--exact")
        .arg("--nocapture")
        .current_dir(workspace)
        .env(PACKAGE_CHILD_ENV, "1")
        .env("CARGO", cargo.program())
        .env("PATH", program_dir)
        .output()
        .expect("run package test from workspace root");
    std::fs::remove_dir_all(package_target).expect("remove package fixture directory");
    assert!(
        output.status.success(),
        "workspace child failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    assert!(git.log().starts_with("status --porcelain|"));
    let cargo_log = cargo.log();
    let commands = cargo_log.lines().collect::<Vec<_>>();
    let publishable_count =
        parse_release_manifest_source(include_str!("../../../../release-crates.json"))
            .expect("checked-in release manifest")
            .ordered_crates()
            .len();
    let consumer_commands = match std::env::consts::OS {
        "linux" | "macos" => 9,
        _ => 7,
    };
    let registry_independent = ["j2k-core", "j2k-profile", "j2k-types", "j2k-codec-math"];
    assert_eq!(
        commands.len(),
        1 + 2 * publishable_count + registry_independent.len() + consumer_commands
    );
    assert!(commands[0].starts_with("metadata --locked --no-deps --format-version 1|"));
    assert!(commands[1].starts_with("package -p j2k-core --list|"));
    for package in registry_independent {
        assert!(commands
            .iter()
            .any(|line| line.starts_with(&format!("publish -p {package} --dry-run|"))));
        assert!(commands
            .iter()
            .any(|line| line.starts_with(&format!("package -p {package} --no-verify|"))));
    }
    assert!(commands
        .iter()
        .any(|line| line.starts_with("package -p j2k-cli --no-verify")));
    assert!(commands
        .iter()
        .any(|line| line.contains("check --examples --no-default-features --features")));
    assert!(commands
        .iter()
        .any(|line| line.contains("doc --no-deps --no-default-features --features")));
}
