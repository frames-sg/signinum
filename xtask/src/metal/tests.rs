// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::process::use_test_cargo_program;
use crate::test_command::RecordingProgram;

use super::{
    listed_rust_tests, metal_compile, passed_rust_tests, reject_skip_markers, release_metal,
    run_metal_compile, run_release_metal, runtime_suite_args, validate_exact_ignored_run,
    J2K_METAL_REQUIRED_IGNORED_TESTS, METAL_OPTIONAL_IGNORED_TESTS,
};
use crate::gpu_validation::ValidationMode;

fn recording_metal_cargo() -> RecordingProgram {
    let listed = J2K_METAL_REQUIRED_IGNORED_TESTS
        .iter()
        .chain(METAL_OPTIONAL_IGNORED_TESTS)
        .map(|name| format!("printf '%s\\n' '{name}: test'"))
        .collect::<Vec<_>>()
        .join("\n");
    let passed = J2K_METAL_REQUIRED_IGNORED_TESTS
        .iter()
        .map(|name| format!("printf '%s\\n' 'test {name} ... ok'"))
        .collect::<Vec<_>>()
        .join("\n");
    let passed_count = J2K_METAL_REQUIRED_IGNORED_TESTS.len();
    let script = format!(
        r#"case " $* " in
*" --list "*)
{listed}
;;
*"one_thousand_direct_submissions_reuse_one_session"*)
printf '%s\n' 'test one_thousand_direct_submissions_reuse_one_session ... ok'
printf '%s\n' 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 8 filtered out'
;;
*" --ignored "*)
{passed}
printf '%s\n' 'test result: ok. {passed_count} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out'
;;
*" -p j2k-metal-support "*)
printf '%s\n' 'test tests::commit_and_wait_accepts_unlabeled_command_buffer ... ok'
printf '%s\n' 'test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out'
;;
*" -p j2k-jpeg-metal "*)
printf '%s\n' 'test decode_to_metal_matches_cpu_decode_bytes ... ok'
printf '%s\n' 'test result: ok. 100 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out'
;;
*" -p j2k-transcode-metal "*)
printf '%s\n' 'test ycbcr_420_jpeg_transcodes_to_htj2k_with_explicit_metal_97_and_native_sampling ... ok'
printf '%s\n' 'test result: ok. 20 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out'
;;
*" -p j2k-metal "*)
printf '%s\n' 'test encode::tests::stage_validation::metal_deinterleave_gray16_lossless_facade_dispatches_and_round_trips ... ok'
printf '%s\n' 'test result: ok. 150 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out'
;;
*" -p j2k-ml "*)
printf '%s\n' 'test sessions::persistent_metal_burn_decoder_uploads_independent_ht_through_staging ... ok'
printf '%s\n' 'test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out'
;;
*" -p j2k-mpsgraph "*)
printf '%s\n' 'test prepared_group_submits_decode_and_identity_graph_without_a_cpu_wait ... ok'
printf '%s\n' 'test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out'
;;
*" -p j2k "*)
printf '%s\n' 'test accelerator_facade_reports_requested_backend_after_all_required_stages_dispatch ... ok'
printf '%s\n' 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out'
;;
esac"#
    );
    RecordingProgram::new("metal-command-test", &script)
}

#[test]
fn metal_commands_execute_complete_hermetic_compile_and_release_plans() {
    let recording = recording_metal_cargo();
    let _cargo = use_test_cargo_program(recording.program().as_os_str().to_owned());

    if cfg!(target_os = "macos") {
        metal_compile().expect("hermetic Metal compile plan");
        release_metal(std::iter::empty()).expect("hermetic Metal release plan");
    } else {
        run_metal_compile().expect("platform-independent Metal compile plan");
        run_release_metal(ValidationMode::Full).expect("platform-independent Metal release plan");
        assert!(metal_compile().is_err());
        assert!(release_metal(std::iter::empty()).is_err());
    }

    let log = recording.log();
    assert_eq!(log.lines().count(), 14);
    assert!(log.contains("clippy --lib --all-features"));
    assert!(log.contains("clippy --bins --examples --tests --benches --all-features"));
    assert!(log.contains("-A clippy::disallowed_methods -A clippy::disallowed_macros"));
    assert!(log.contains("test --release --all-features --lib --bins --tests"));
    assert!(log.contains("test --release --all-features --doc"));
    assert!(log.contains("--ignored --list"));
    assert!(log.contains("--ignored --show-output"));
    assert!(log.contains("--skip idwt::tests::metal_irreversible_idwt_gpu_capture"));
    assert!(log.contains("-p j2k-metal-support"));
    assert!(log.contains("-p j2k-jpeg-metal"));
    assert!(log.contains("-p j2k-transcode-metal"));
    assert!(log.contains("-p j2k-ml"));
    assert!(log.contains("-p j2k-mpsgraph"));
    assert!(log.contains("one_thousand_direct_submissions_reuse_one_session"));
}

#[test]
fn metal_quick_preserves_every_runtime_and_ignored_inventory_gate() {
    let recording = recording_metal_cargo();
    let _cargo = use_test_cargo_program(recording.program().as_os_str().to_owned());

    run_release_metal(ValidationMode::Quick).expect("quick Metal command plan");

    let log = recording.log();
    assert_eq!(log.lines().count(), 11);
    assert!(log.lines().all(|line| line.contains("--profile gpu-quick")));
    assert_eq!(log.matches("clippy --profile gpu-quick").count(), 2);
    assert!(log.contains("clippy --profile gpu-quick --lib"));
    assert!(log.contains("clippy --profile gpu-quick --bins --examples --tests --benches"));
    assert!(log.contains("-A clippy::disallowed_methods -A clippy::disallowed_macros"));
    assert!(!log.contains("--all-features"));
    assert!(!log.contains("cuda"));
    assert!(log.contains("--features j2k-ml/metal"));
    assert!(log.contains("-p j2k-ml --features metal"));
    assert!(log.contains("-p j2k-mpsgraph"));
    assert!(log.contains("--ignored --list"));
    assert!(log.contains("--ignored --show-output"));
    assert!(!log.contains("--release"));
}

#[test]
fn parses_listed_and_passed_rust_tests() {
    let listed = "alpha::works: test\nbeta::works: test\n";
    assert_eq!(
        listed_rust_tests(listed).into_iter().collect::<Vec<_>>(),
        ["alpha::works", "beta::works"]
    );

    let passed = "test alpha::works ... ok\ntest beta::works ... ok\n";
    assert_eq!(
        passed_rust_tests(passed).into_iter().collect::<Vec<_>>(),
        ["alpha::works", "beta::works"]
    );
}

#[test]
fn skip_marker_is_always_a_release_failure() {
    let err = reject_skip_markers(
        "J2K_GPU_TEST_SKIPPED gate=J2K_REQUIRE_METAL_RUNTIME",
        "Metal",
    )
    .expect_err("skip marker must fail");
    assert!(err.contains("must fail rather than skip"));
}

#[test]
fn unrelated_cuda_skip_marker_does_not_fail_metal_release_validation() {
    reject_skip_markers(
        "J2K_GPU_TEST_SKIPPED gate=J2K_REQUIRE_CUDA_RUNTIME context=CUDA-only test",
        "Burn J2K Metal tensor integration",
    )
    .expect("a CUDA-only skip must not masquerade as a skipped Metal runtime test");
}

#[test]
fn exact_ignored_validation_rejects_zero_tests() {
    let err = validate_exact_ignored_run(
        "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 18 filtered out",
    )
    .expect_err("zero selected tests must fail");
    assert!(err.contains("passed 0 tests"));
}

#[test]
fn ignored_inventory_is_unique_and_has_expected_size() {
    let required = J2K_METAL_REQUIRED_IGNORED_TESTS
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let optional = METAL_OPTIONAL_IGNORED_TESTS
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(required.len(), 21);
    assert_eq!(optional.len(), 5);
    assert!(optional.contains(
        "engine::decode_dispatch::idwt::cooperative53::route::tests::cooperative53_end_to_end_timing"
    ));
    assert!(optional.contains(
        "engine::decode_dispatch::idwt::cooperative53::tests::cooperative53_gpu_stage_timing"
    ));
    assert!(
        optional.contains("decoder::tests::sampled::local_sampled_color_batch_characterization")
    );
    assert!(optional.contains(
        "engine::runtime::resource_profile_tests::benchmark_cold_and_repeated_session_kernel_initialization"
    ));
    assert_eq!(required.len(), J2K_METAL_REQUIRED_IGNORED_TESTS.len());
    assert_eq!(optional.len(), METAL_OPTIONAL_IGNORED_TESTS.len());
    assert!(required.is_disjoint(&optional));
}

#[test]
fn ignored_engine_inventory_tracks_the_current_module_path() {
    let engine_tests = J2K_METAL_REQUIRED_IGNORED_TESTS
        .iter()
        .filter(|name| {
            name.starts_with("engine::tests::classic::")
                || name.starts_with("engine::tests::grouping::")
                || name.starts_with("engine::tests::roi::")
        })
        .count();
    assert_eq!(engine_tests, 13);
    assert!(J2K_METAL_REQUIRED_IGNORED_TESTS
        .iter()
        .all(|name| !name.starts_with("compute::tests::")));
}

#[test]
fn runtime_gate_excludes_benchmark_targets() {
    let args = runtime_suite_args("j2k-metal", ValidationMode::Full);
    assert!(args.contains(&"--lib"));
    assert!(args.contains(&"--bins"));
    assert!(args.contains(&"--tests"));
    assert!(args.contains(&"--examples"));
    assert!(!args.contains(&"--all-targets"));
    assert!(!args.contains(&"--benches"));
}

#[test]
fn quick_metal_runtime_uses_fast_profile_without_weakening_target_scope() {
    let args = runtime_suite_args("j2k-metal", ValidationMode::Quick);
    assert!(args
        .windows(2)
        .any(|pair| pair == ["--profile", "gpu-quick"]));
    assert!(!args.contains(&"--release"));
    assert!(!args.contains(&"--all-features"));
    assert!(args.contains(&"--lib"));
    assert!(args.contains(&"--bins"));
    assert!(args.contains(&"--tests"));
    assert!(args.contains(&"--examples"));
}
