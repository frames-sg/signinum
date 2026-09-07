// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(unix)]

#[path = "command_orchestration/support.rs"]
mod support;

use std::fs;
use std::process::Output;

use support::{assert_success, Harness};

#[test]
fn release_status_executes_exact_sha_verification_without_exposing_tokens() {
    let harness = Harness::new();
    let explicit_sha = "A".repeat(40);
    let remote_sha = "B".repeat(40);

    assert_success(
        &harness.run_with_env(
            &[
                "release-status",
                "--sha",
                &explicit_sha,
                "--repository",
                "frames-sg/j2k",
                "--scope",
                "cpu",
            ],
            &[("GH_TOKEN", "present")],
        ),
        "release-status with explicit repository",
    );
    assert_success(
        &harness.run_with_env(
            &["release-status", "--sha", &remote_sha],
            &[("GITHUB_TOKEN", "present")],
        ),
        "release-status with remote-derived repository",
    );

    let help = harness.run(&["release-status", "--help"]);
    assert!(
        !help.status.success(),
        "help must preserve task error handling"
    );
    assert!(String::from_utf8_lossy(&help.stderr).contains(
        "usage: cargo xtask release-status --sha <40-hex-commit> [--repository owner/name] [--scope cpu|cuda|metal|all]"
    ));

    let log = harness.log();
    assert_eq!(
        log.lines()
            .filter(|line| line.starts_with("python3 "))
            .count(),
        2
    );
    assert!(log.contains("git config --get remote.origin.url"));
    assert!(log.contains(&format!(
        "--candidate-sha {}",
        explicit_sha.to_ascii_lowercase()
    )));
    assert!(log.contains(&format!(
        "--candidate-sha {}",
        remote_sha.to_ascii_lowercase()
    )));
    assert!(log.contains("--repository frames-sg/j2k"));
    assert!(log.contains("--token-env GH_TOKEN"));
    assert!(log.contains("--token-env GITHUB_TOKEN"));
    assert!(!log.contains("present"), "token values reached command log");
}

#[test]
fn release_critical_orchestrators_run_from_the_workspace_without_real_cargo() {
    let harness = Harness::new();

    assert_success(&harness.run(&["ci"]), "ci");
    assert_success(&harness.run(&["bench-build"]), "bench-build");
    assert_success(&harness.run(&["j2k-bench-signoff"]), "j2k-bench-signoff");
    assert_success(&harness.run(&["release-cpu"]), "release-cpu");
    assert_success(&harness.run(&["release-integrity"]), "release-integrity");
    assert_success(&harness.run(&["package"]), "package");

    for args in [
        &["release-integrity", "--unknown"][..],
        &["stable-api", "--unknown"][..],
        &["semver", "--unknown"][..],
    ] {
        let output = harness.run(args);
        assert!(!output.status.success(), "{args:?} must fail closed");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("xtask failed:"),
            "{args:?} must preserve the command error"
        );
    }
    let output = harness.run(&["stable-api"]);
    assert_stable_api_fails_at_expected_boundary(&output, "stable-api");
    let output = harness.run_with_env(
        &["stable-api"],
        &[
            ("RUSTFLAGS", "-D warnings"),
            ("RUSTDOCFLAGS", "-D warnings"),
        ],
    );
    assert_stable_api_fails_at_expected_boundary(
        &output,
        "stable-api under canonical CI warning flags",
    );
    let output = harness.run(&["semver"]);
    assert_semver_fails_at_expected_boundary(&output);

    let log = harness.log();
    assert!(log.contains("fmt --all -- --check|"));
    assert!(log.contains("bench -p j2k --bench public_api --no-run|"));
    assert!(log.contains("test -p j2k-compare --test in_process_parity -- --nocapture|"));
    assert!(log.contains("test --release -p j2k-core"));
    assert!(log.contains("package -p j2k-core --list|"));
    assert!(log.contains("publish -p j2k-core --dry-run|"));
    assert!(log.contains("package -p j2k-cli --no-verify"));
    #[cfg(target_os = "macos")]
    {
        assert!(log.contains("git rev-parse v0.11.0^{commit}"));
        assert!(log.contains("git show v0.11.0:docs/stable-api-1.0.public-api.txt"));
    }
}

fn assert_stable_api_fails_at_expected_boundary(output: &Output, context: &str) {
    assert!(!output.status.success(), "{context} must fail closed");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let expected_error = if cfg!(target_os = "macos") {
        "stable API snapshots are stale"
    } else {
        "stable-api snapshot must be generated on macOS so target-gated Metal APIs are included"
    };
    assert!(
        stderr.contains(expected_error),
        "{context} must reach its platform's stable-API boundary: {stderr}"
    );
}

fn assert_semver_fails_at_expected_boundary(output: &Output) {
    assert!(!output.status.success(), "synthetic API must fail semver");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let expected_error = if cfg!(target_os = "macos") {
        "committed published-library API snapshots are stale"
    } else {
        "semver/API review must run on macOS so Metal public APIs are included"
    };
    assert!(
        stderr.contains(expected_error),
        "semver must reach its platform's API-review boundary: {stderr}"
    );
}

#[test]
fn external_quality_commands_preserve_their_complete_fake_tool_plans() {
    let harness = Harness::new();

    for task in ["miri", "machete", "no-std"] {
        assert_success(&harness.run(&[task]), task);
    }
    for task in ["nextest", "typos", "deny", "downstream-smoke"] {
        let output = harness.run(&[task]);
        assert!(!output.status.success(), "{task} must remain retired");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("unknown task"),
            "{task} must fail at the dispatcher boundary"
        );
    }
    assert_success(
        &harness.run_with_env(
            &["fuzz-run"],
            &[
                ("J2K_FUZZ_TARGET", "aarch64-apple-darwin"),
                ("J2K_FUZZ_RUNS", "17"),
                ("J2K_FUZZ_MAX_TOTAL_TIME_SECONDS", "3"),
            ],
        ),
        "fuzz-run",
    );

    let log = harness.log();
    assert!(log.contains("cargo-machete --with-metadata\n"));
    assert!(log.contains("rustup run nightly cargo miri test -p j2k-core"));
    assert!(log.contains("rustup target add aarch64-unknown-none"));
    assert!(log.contains("check -p j2k-core --target wasm32-unknown-unknown|"));
    assert!(log.contains(
        "rustup run nightly cargo fuzz run --target aarch64-apple-darwin decode_fuzz -- -runs=17 -max_total_time=3"
    ));
    assert!(
        !log.contains("check -p -p"),
        "duplicate package flag: {log}"
    );
}

#[test]
fn codec_math_codegen_write_is_transactional_across_both_fragments() {
    let harness = Harness::new();
    let workspace = harness.path("codegen-workspace");
    let generated = workspace.join("crates/j2k-codec-math/generated");
    fs::create_dir_all(&generated).expect("create generated fragment directory");
    let metal = generated.join("dwt97_constants.metal");
    let rust = generated.join("dwt97_constants.rs");
    fs::write(&metal, "old metal\n").expect("seed Metal fragment");
    fs::create_dir(&rust).expect("make Rust fragment target unwritable");

    let output = harness.run_in(&workspace, &["codec-math-codegen", "--write"]);

    assert!(!output.status.success(), "unwritable target must fail");
    assert_eq!(
        fs::read_to_string(&metal).expect("read Metal fragment after failure"),
        "old metal\n",
        "failure updating the Rust fragment must not commit the Metal fragment"
    );
    let entries = fs::read_dir(&generated)
        .expect("read generated fragment directory")
        .collect::<Result<Vec<_>, _>>()
        .expect("read generated fragment entries");
    assert_eq!(entries.len(), 2, "failed transaction left sidecar files");
}
