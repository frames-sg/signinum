// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hosted compile and fail-closed hardware validation for the Metal adapters.

use std::{collections::BTreeSet, env};

use crate::gpu_validation::ValidationMode;
use crate::process::{self, cargo, CommandContext};

const METAL_GPU_TEST_SKIP_MARKER: &str = "J2K_GPU_TEST_SKIPPED gate=J2K_REQUIRE_METAL_RUNTIME";
const METAL_COMPILE_ENV: &[(&str, &str)] = &[("RUST_TEST_THREADS", "1")];
const METAL_RUNTIME_ENV: &[(&str, &str)] = &[
    ("J2K_REQUIRE_METAL_RUNTIME", "1"),
    ("RUST_TEST_THREADS", "1"),
];
const METAL_RUNTIME_TEST_TARGETS: &[&str] = &["--lib", "--bins", "--tests", "--examples"];

const METAL_COMPILE_PACKAGES: &[&str] = &[
    "j2k-metal-support",
    "j2k-jpeg-metal",
    "j2k-metal",
    "j2k-transcode-metal",
    "j2k-ml",
    "j2k-mpsgraph",
    "j2k-mpsgraph-support",
    "j2k",
];

const J2K_METAL_REQUIRED_IGNORED_TESTS: &[&str] = &[
    "engine::tests::classic::irreversible_hybrid_cpu_tier1_matches_native_decode_exactly",
    "engine::tests::classic::prepared_classic_direct_plan_groups_cleanup_subbands_before_idwt",
    "engine::tests::classic::prepared_classic_sub_band_decodes_on_cpu_for_hybrid_upload",
    "engine::tests::classic::prepared_irreversible_classic_sub_band_records_midpoint_reconstruction",
    "engine::tests::grouping::distinct_prepared_ht_direct_plans_support_stacked_component_batch",
    "engine::tests::grouping::grouped_ht_direct_plan_uses_one_group_coded_arena",
    "engine::tests::grouping::prepared_ht_direct_plan_encodes_full_decode_in_one_compute_encoder",
    "engine::tests::grouping::prepared_ht_direct_plan_groups_cleanup_subbands_before_idwt",
    "engine::tests::grouping::repeated_prepared_ht_direct_plan_groups_cleanup_subbands_before_idwt",
    "engine::tests::roi::cropped_region_ht_direct_plan_keeps_idwt_windows_bounded",
    "engine::tests::roi::cropped_region_scaled_ht_direct_plan_compacts_coded_payloads",
    "engine::tests::roi::cropped_region_scaled_ht_direct_plan_prunes_codeblocks_outside_output_roi",
    "engine::tests::roi::cropped_region_scaled_ht_direct_plan_reduces_idwt_output_work",
    "direct::tests::classic_direct_plan_idwt_inputs_match_native_backend_job",
    "direct::tests::classic_direct_plan_pre_store_band_is_not_all_zero",
    "direct::tests::classic_direct_plan_store_plane_matches_native_decode",
    "direct::tests::classic_direct_plan_sub_band_decode_produces_nonzero_coefficients",
    "direct::tests::ht_direct_plan_sub_band_decode_produces_nonzero_coefficients",
    "encode::tests::routing::auto_htj2k_padded_private_gray8_single_host_output_stays_cpu",
    "encode::tests::routing::auto_htj2k_padded_private_rgb8_single_host_output_stays_cpu",
    "idwt::tests::metal_irreversible_idwt_perf_guard",
];

// Capture needs an explicit output path; timing diagnostics are manual benchmarks.
// Sampled characterization additionally needs the caller-provided corpus.
// Correctness, sampled pixel parity, and kernel reuse have non-ignored tests.
// Inventory every diagnostic so ignored tests cannot silently escape review.
const METAL_OPTIONAL_IGNORED_TESTS: &[&str] = &[
    "decoder::tests::sampled::local_sampled_color_batch_characterization",
    "idwt::tests::metal_irreversible_idwt_gpu_capture",
    "engine::runtime::resource_profile_tests::benchmark_cold_and_repeated_session_kernel_initialization",
    "idwt::tests::metal_host_slice_idwt_decode_perf",
    "engine::decode_dispatch::idwt::irreversible::performance::metal_irreversible97_interleave_horizontal_scale_gpu_timing",
    "engine::decode_dispatch::idwt::irreversible::performance::metal_irreversible97_stage_gpu_timing",
];

struct MetalTestSuite {
    label: &'static str,
    package: &'static str,
    minimum_passed: usize,
    required_test: &'static str,
}

const METAL_TEST_SUITES: &[MetalTestSuite] = &[
    MetalTestSuite {
        label: "Metal support runtime",
        package: "j2k-metal-support",
        minimum_passed: 5,
        required_test: "tests::commit_and_wait_accepts_unlabeled_command_buffer",
    },
    MetalTestSuite {
        label: "JPEG Metal runtime",
        package: "j2k-jpeg-metal",
        minimum_passed: 100,
        required_test: "decode_to_metal_matches_cpu_decode_bytes",
    },
    MetalTestSuite {
        label: "J2K Metal and facade integration runtime",
        package: "j2k-metal",
        minimum_passed: 150,
        required_test:
            "encode::tests::stage_validation::metal_deinterleave_gray16_lossless_facade_dispatches_and_round_trips",
    },
    MetalTestSuite {
        label: "transcode Metal runtime",
        package: "j2k-transcode-metal",
        minimum_passed: 20,
        required_test:
            "ycbcr_420_jpeg_transcodes_to_htj2k_with_explicit_metal_97_and_native_sampling",
    },
    MetalTestSuite {
        label: "Burn J2K Metal tensor integration",
        package: "j2k-ml",
        minimum_passed: 4,
        required_test:
            "sessions::persistent_metal_burn_decoder_uploads_independent_ht_through_staging",
    },
    MetalTestSuite {
        label: "direct J2K MPSGraph integration",
        package: "j2k-mpsgraph",
        minimum_passed: 4,
        required_test:
            "prepared_group_submits_decode_and_identity_graph_without_a_cpu_wait",
    },
    MetalTestSuite {
        label: "J2K public facade",
        package: "j2k",
        minimum_passed: 1,
        required_test:
            "accelerator_facade_reports_requested_backend_after_all_required_stages_dispatch",
    },
];

/// Compiles every Metal-facing target and runs default/pure tests on hosted macOS.
pub(crate) fn metal_compile() -> Result<(), String> {
    require_macos("metal-compile")?;
    if env::var_os("J2K_REQUIRE_METAL_RUNTIME").is_some() {
        return Err(
            "metal-compile requires J2K_REQUIRE_METAL_RUNTIME to be unset; use `cargo xtask release-metal` for fail-closed hardware validation"
                .to_string(),
        );
    }
    run_metal_compile()
}

fn run_metal_compile() -> Result<(), String> {
    let mut library_clippy_args = vec!["clippy", "--lib", "--all-features"];
    append_packages(&mut library_clippy_args);
    library_clippy_args.extend_from_slice(&["--", "-D", "warnings"]);
    process::run_command(cargo(), &library_clippy_args, CommandContext::new())?;

    let mut non_library_clippy_args = vec![
        "clippy",
        "--bins",
        "--examples",
        "--tests",
        "--benches",
        "--all-features",
    ];
    append_packages(&mut non_library_clippy_args);
    append_non_library_clippy_flags(&mut non_library_clippy_args);
    process::run_command(cargo(), &non_library_clippy_args, CommandContext::new())?;

    let mut test_args = vec![
        "test",
        "--release",
        "--all-features",
        "--lib",
        "--bins",
        "--tests",
    ];
    append_packages(&mut test_args);
    process::run_command(
        cargo(),
        &test_args,
        CommandContext::new().envs(METAL_COMPILE_ENV),
    )?;

    let mut doc_args = vec!["test", "--release", "--all-features", "--doc"];
    append_packages(&mut doc_args);
    process::run_command(cargo(), &doc_args, CommandContext::new())
}

/// Runs every required Metal runtime suite and rejects all evidence of skipping.
pub(crate) fn release_metal(args: impl Iterator<Item = String>) -> Result<(), String> {
    require_macos("release-metal")?;
    run_release_metal(ValidationMode::parse(args)?)
}

fn run_release_metal(mode: ValidationMode) -> Result<(), String> {
    if mode == ValidationMode::Quick {
        for (args, label) in [
            (
                quick_clippy_args(false),
                "quick Metal production library Clippy",
            ),
            (
                quick_clippy_args(true),
                "quick Metal non-library target Clippy",
            ),
        ] {
            let output = run_cargo_captured(&args, METAL_RUNTIME_ENV, label)?;
            reject_skip_markers(&output, label)?;
        }
    }
    validate_required_ignored_inventory(mode)?;

    for suite in METAL_TEST_SUITES {
        let args = runtime_suite_args(suite.package, mode);
        let output = run_cargo_captured(&args, METAL_RUNTIME_ENV, suite.label)?;
        validate_test_run(
            &output,
            suite.label,
            suite.minimum_passed,
            &[suite.required_test],
        )?;
    }

    if mode == ValidationMode::Full {
        let output = run_cargo_captured(
            &[
                "test",
                "--release",
                "-p",
                "j2k-mpsgraph",
                "--test",
                "metal",
                "one_thousand_direct_submissions_reuse_one_session",
                "--",
                "--ignored",
                "--exact",
                "--show-output",
            ],
            METAL_RUNTIME_ENV,
            "direct MPSGraph 1,000-submission soak",
        )?;
        validate_test_run(
            &output,
            "direct MPSGraph 1,000-submission soak",
            1,
            &["one_thousand_direct_submissions_reuse_one_session"],
        )?;
    }

    let mut ignored_args = vec!["test"];
    ignored_args.extend_from_slice(mode.cargo_profile_args());
    ignored_args.extend_from_slice(&[
        "-p",
        "j2k-metal",
        "--lib",
        "--",
        "--ignored",
        "--show-output",
    ]);
    for optional_test in METAL_OPTIONAL_IGNORED_TESTS {
        ignored_args.extend_from_slice(&["--skip", optional_test]);
    }
    let output = run_cargo_captured(
        &ignored_args,
        METAL_RUNTIME_ENV,
        "required ignored J2K Metal runtime inventory",
    )?;
    validate_exact_ignored_run(&output)
}

fn quick_clippy_args(non_library_targets: bool) -> Vec<&'static str> {
    let mut args = vec!["clippy", "--profile", "gpu-quick"];
    if non_library_targets {
        args.extend_from_slice(&["--bins", "--examples", "--tests", "--benches"]);
    } else {
        args.push("--lib");
    }
    append_packages(&mut args);
    args.extend_from_slice(&["--features", "j2k-ml/metal"]);
    if non_library_targets {
        append_non_library_clippy_flags(&mut args);
    } else {
        args.extend_from_slice(&["--", "-D", "warnings"]);
    }
    args
}

fn append_non_library_clippy_flags(args: &mut Vec<&'static str>) {
    args.extend_from_slice(&[
        "--",
        "-D",
        "warnings",
        "-A",
        "clippy::disallowed_methods",
        "-A",
        "clippy::disallowed_macros",
    ]);
}

fn runtime_suite_args(package: &str, mode: ValidationMode) -> Vec<&str> {
    let mut args = vec!["test"];
    args.extend_from_slice(mode.cargo_profile_args());
    if mode == ValidationMode::Full {
        args.push("--all-features");
    }
    args.extend_from_slice(METAL_RUNTIME_TEST_TARGETS);
    args.extend_from_slice(&["-p", package]);
    if mode == ValidationMode::Quick && package == "j2k-ml" {
        args.extend_from_slice(&["--features", "metal"]);
    }
    args.extend_from_slice(&["--", "--show-output"]);
    args
}

fn require_macos(task: &str) -> Result<(), String> {
    if env::consts::OS == "macos" {
        Ok(())
    } else {
        Err(format!(
            "{task} requires macOS, but this host is {}; refusing to report Metal success without the required platform",
            env::consts::OS
        ))
    }
}

fn append_packages(args: &mut Vec<&str>) {
    for package in METAL_COMPILE_PACKAGES {
        args.extend_from_slice(&["-p", package]);
    }
}

fn validate_required_ignored_inventory(mode: ValidationMode) -> Result<(), String> {
    let mut args = vec!["test"];
    args.extend_from_slice(mode.cargo_profile_args());
    args.extend_from_slice(&["-p", "j2k-metal", "--lib", "--", "--ignored", "--list"]);
    let output = run_cargo_captured(
        &args,
        METAL_RUNTIME_ENV,
        "list required ignored J2K Metal runtime tests",
    )?;
    let actual = listed_rust_tests(&output);
    let expected = J2K_METAL_REQUIRED_IGNORED_TESTS
        .iter()
        .chain(METAL_OPTIONAL_IGNORED_TESTS)
        .map(|name| (*name).to_string())
        .collect::<BTreeSet<_>>();
    if actual == expected {
        return Ok(());
    }

    let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
    let unexpected = actual.difference(&expected).cloned().collect::<Vec<_>>();
    Err(format!(
        "ignored J2K Metal runtime inventory drifted; missing: {missing:?}; unexpected: {unexpected:?}"
    ))
}

fn run_cargo_captured(args: &[&str], envs: &[(&str, &str)], label: &str) -> Result<String, String> {
    let env_display = envs
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!(
        "+ {env_display} {} {}",
        cargo().to_string_lossy(),
        args.join(" ")
    );
    let output = process::command_output(cargo(), args, CommandContext::new().envs(envs))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    print!("{stdout}");
    eprint!("{stderr}");
    let combined = format!("{stdout}\n{stderr}");
    if output.status.success() {
        Ok(combined)
    } else {
        Err(format!(
            "`{}` exited with {} while running {label}",
            cargo().to_string_lossy(),
            output.status
        ))
    }
}

fn validate_test_run(
    output: &str,
    label: &str,
    minimum_passed: usize,
    required_tests: &[&str],
) -> Result<(), String> {
    reject_skip_markers(output, label)?;
    let passed = passed_test_count(output);
    if passed < minimum_passed {
        return Err(format!(
            "{label} executed {passed} tests, expected at least {minimum_passed}; the suite may not have exercised its runtime path"
        ));
    }

    let passed_names = passed_rust_tests(output);
    let missing = required_tests
        .iter()
        .filter(|name| !passed_names.contains(**name))
        .copied()
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{label} did not pass required named tests: {missing:?}"
        ))
    }
}

fn validate_exact_ignored_run(output: &str) -> Result<(), String> {
    let label = "required ignored J2K Metal runtime inventory";
    reject_skip_markers(output, label)?;
    let passed = passed_test_count(output);
    if passed != J2K_METAL_REQUIRED_IGNORED_TESTS.len() {
        return Err(format!(
            "{label} passed {passed} tests, expected exactly {}",
            J2K_METAL_REQUIRED_IGNORED_TESTS.len()
        ));
    }

    let actual = passed_rust_tests(output);
    let expected = J2K_METAL_REQUIRED_IGNORED_TESTS
        .iter()
        .map(|name| (*name).to_string())
        .collect::<BTreeSet<_>>();
    if actual == expected {
        Ok(())
    } else {
        let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
        let unexpected = actual.difference(&expected).cloned().collect::<Vec<_>>();
        Err(format!(
            "{label} name mismatch; missing: {missing:?}; unexpected: {unexpected:?}"
        ))
    }
}

fn reject_skip_markers(output: &str, label: &str) -> Result<(), String> {
    if output.contains(METAL_GPU_TEST_SKIP_MARKER) {
        Err(format!(
            "{label} emitted {METAL_GPU_TEST_SKIP_MARKER}; release Metal validation must fail rather than skip"
        ))
    } else {
        Ok(())
    }
}

fn passed_test_count(output: &str) -> usize {
    output
        .lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("test result: ok.")?;
            if !rest.contains(" passed") {
                return None;
            }
            rest.split_whitespace().next()?.parse::<usize>().ok()
        })
        .sum()
}

fn listed_rust_tests(output: &str) -> BTreeSet<String> {
    output
        .lines()
        .filter_map(|line| line.trim().strip_suffix(": test"))
        .map(str::to_string)
        .collect()
}

fn passed_rust_tests(output: &str) -> BTreeSet<String> {
    output
        .lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("test ")?;
            rest.strip_suffix(" ... ok").map(str::to_string)
        })
        .collect()
}

#[cfg(test)]
mod tests;
