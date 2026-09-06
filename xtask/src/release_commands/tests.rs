// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use super::release_manifest::{crates_io_publishable, release_manifest_contract};
use super::{
    has_docs_rs_metadata, has_lib_target, package_consumer_smoke, package_name, release_cpu,
    release_integrity, validate_publish_script_source, validate_publish_workflow_source,
    validate_release_docs_source, validate_unpublished_dependencies, workspace_package_records,
};

#[test]
fn package_consumer_smoke_rejects_invalid_routes_before_packaging() {
    for (args, expected) in [
        (
            vec!["--target", "unknown"],
            "usage: cargo xtask package-consumer-smoke",
        ),
        (
            vec!["--target", "metal", "--cuda-runtime"],
            "--cuda-runtime requires --target cuda or all",
        ),
    ] {
        let error = package_consumer_smoke(args.into_iter().map(str::to_string))
            .expect_err("invalid package consumer request");
        assert!(error.contains(expected), "unexpected error: {error}");
    }
}

#[test]
fn full_gpu_workflows_compile_the_packaged_adapter_archives() {
    let workflow = include_str!("../../../.github/workflows/gpu-validation-runner.yml");

    assert!(workflow.contains("cargo xtask package-consumer-smoke --target cuda --cuda-runtime"));
    assert!(workflow.contains("cargo xtask package-consumer-smoke --target metal"));
}

#[test]
fn gpu_dispatchers_call_immutable_reusable_workflows() {
    for (source, runner_file) in [
        (
            include_str!("../../../.github/workflows/gpu-validation.yml"),
            "gpu-validation-runner.yml",
        ),
        (
            include_str!("../../../.github/workflows/gpu-benchmarks.yml"),
            "gpu-benchmarks-runner.yml",
        ),
    ] {
        let workflow: serde_yaml_ng::Value =
            serde_yaml_ng::from_str(source).expect("workflow YAML");
        let jobs = workflow["jobs"].as_mapping().expect("dispatcher jobs");
        assert_eq!(
            jobs.len(),
            1,
            "dispatchers delegate all jobs to the trusted workflow"
        );
        let job = jobs.values().next().expect("dispatcher job");
        assert!(
            job["runs-on"].is_null(),
            "PR workflows cannot directly schedule self-hosted jobs"
        );
        let reference = job["uses"].as_str().expect("reusable workflow reference");
        let (path, revision) = reference
            .rsplit_once('@')
            .expect("pinned reusable workflow");
        assert_eq!(
            path,
            format!("frames-sg/j2k/.github/workflows/{runner_file}")
        );
        assert_eq!(revision.len(), 40);
        assert!(revision.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_eq!(workflow["on"].as_mapping().expect("triggers").len(), 1);
        assert!(workflow["on"]["workflow_dispatch"].is_mapping());
    }
}

#[test]
fn gpu_workflows_allow_only_owner_manual_runs_on_any_branch() {
    let mut gpu_jobs = 0;
    for source in [
        include_str!("../../../.github/workflows/gpu-validation-runner.yml"),
        include_str!("../../../.github/workflows/gpu-benchmarks-runner.yml"),
    ] {
        let workflow: serde_yaml_ng::Value =
            serde_yaml_ng::from_str(source).expect("valid GPU workflow YAML");
        let triggers = workflow["on"].as_mapping().expect("workflow triggers");
        assert_eq!(
            triggers.len(),
            1,
            "Runner workflows expose only the reusable entry point"
        );
        assert!(triggers.contains_key("workflow_call"));
        let jobs = workflow["jobs"].as_mapping().expect("workflow jobs");
        for (name, job) in jobs {
            let Some(runners) = job["runs-on"].as_sequence() else {
                continue;
            };
            if !runners
                .iter()
                .any(|runner| runner.as_str() == Some("self-hosted"))
            {
                continue;
            }
            gpu_jobs += 1;
            for step in job["steps"].as_sequence().expect("GPU job steps") {
                if step["uses"]
                    .as_str()
                    .is_some_and(|value| value.starts_with("actions/checkout@"))
                {
                    assert_eq!(step["with"]["ref"].as_str(), Some("${{ github.sha }}"));
                    assert_eq!(step["with"]["persist-credentials"].as_bool(), Some(false));
                }
            }
            let name = name.as_str().expect("job name");
            let selection = match name {
                "cuda-quick" => {
                    "inputs.mode == 'quick' && (inputs.target == 'cuda' || inputs.target == 'all')"
                }
                "metal-quick" => {
                    "inputs.mode == 'quick' && (inputs.target == 'metal' || inputs.target == 'all')"
                }
                "cuda-full" | "metal-full" => "inputs.mode == 'full' && inputs.target == 'all'",
                "cuda" => "inputs.lane == 'cuda'",
                "metal" => "inputs.lane == 'metal'",
                _ => panic!("unreviewed self-hosted GPU job: {name}"),
            };
            let expected = format!(
                "${{{{ github.repository == 'frames-sg/j2k' && github.event_name == 'workflow_dispatch' && github.actor == 'jcwal1516' && github.triggering_actor == 'jcwal1516' && ({selection}) }}}}"
            );
            assert_eq!(
                job["if"].as_str(),
                Some(expected.as_str()),
                "{name} must accept the owner's manual runs from any branch and reject other initiators or rerun actors"
            );
        }
    }
    assert_eq!(gpu_jobs, 6, "all GPU jobs retain the trusted owner guard");
}

#[cfg(unix)]
mod file_boundaries;
#[cfg(unix)]
mod integrity;
#[cfg(unix)]
mod orchestration;
#[cfg(unix)]
mod package_fixture;
#[cfg(unix)]
mod path_patches;
mod validation;

#[cfg(unix)]
use crate::{command_support::use_test_cargo_program, test_command::RecordingProgram};

#[test]
fn release_metadata_shape_helpers_fail_closed() {
    let valid = serde_json::json!({
        "name": "j2k-core",
        "publish": [],
        "targets": [{"kind": ["lib", "rlib"]}],
        "metadata": {"docs": {"rs": {"all-features": true, "targets": []}}},
    });
    assert_eq!(package_name(&valid), Ok("j2k-core"));
    assert!(!crates_io_publishable(&valid, "j2k-core").expect("valid publish field"));
    assert!(has_lib_target(&valid));
    assert!(has_docs_rs_metadata(&valid));

    assert!(package_name(&serde_json::json!({})).is_err());
    assert!(crates_io_publishable(&serde_json::json!({"publish": false}), "malformed").is_err());
    assert!(!has_lib_target(
        &serde_json::json!({"targets": [{"kind": "lib"}]})
    ));
    assert!(!has_docs_rs_metadata(&serde_json::json!({
        "metadata": {"docs": {"rs": {"all-features": false, "targets": []}}}
    })));
    assert!(!has_docs_rs_metadata(&serde_json::json!({
        "metadata": {"docs": {"rs": {"all-features": true, "targets": ["x86_64"]}}}
    })));
}

#[test]
fn workspace_package_records_preserve_member_order_and_reject_malformed_graphs() {
    let valid = serde_json::json!({
        "packages": [
            {"id": "a", "name": "alpha"},
            {"id": "b", "name": "beta"},
        ],
        "workspace_members": ["b", "a"],
    });
    let packages = workspace_package_records(&valid).expect("valid workspace metadata");
    assert_eq!(package_name(packages[0]), Ok("beta"));
    assert_eq!(package_name(packages[1]), Ok("alpha"));

    let malformed = [
        (
            serde_json::json!({"workspace_members": []}),
            "packages array",
        ),
        (
            serde_json::json!({"packages": [], "workspace_members": {}}),
            "workspace_members array",
        ),
        (
            serde_json::json!({"packages": [{}], "workspace_members": []}),
            "packages[0] has no string id",
        ),
        (
            serde_json::json!({
                "packages": [{"id": "a"}, {"id": "a"}],
                "workspace_members": ["a"],
            }),
            "duplicate package id `a`",
        ),
        (
            serde_json::json!({"packages": [], "workspace_members": [1]}),
            "workspace_members[0] is not a string",
        ),
        (
            serde_json::json!({"packages": [{"id": "a"}], "workspace_members": ["a", "a"]}),
            "workspace_members contains duplicate id `a`",
        ),
        (
            serde_json::json!({"packages": [], "workspace_members": ["missing"]}),
            "workspace member `missing` has no package record",
        ),
    ];
    for (metadata, expected) in malformed {
        let error = workspace_package_records(&metadata).expect_err("malformed metadata rejects");
        assert!(error.contains(expected), "unexpected error: {error}");
    }
}

#[test]
fn unpublished_dependency_policy_distinguishes_publishable_dev_and_runtime_edges() {
    let unpublished = BTreeSet::from(["internal"]);
    let package = serde_json::json!({
        "dependencies": [
            {"name": "published", "kind": null, "req": "1"},
            {"name": "internal", "kind": null, "req": "*"},
            {"name": "internal", "kind": "build", "req": "*"},
            {"name": "internal", "kind": "dev", "req": "1"},
            {"name": "internal", "kind": "dev", "req": "*"},
        ]
    });
    let mut errors = Vec::new();
    validate_unpublished_dependencies("consumer", &package, &unpublished, &mut errors)
        .expect("well-shaped dependencies");
    assert_eq!(errors.len(), 3);
    assert!(errors[0].contains("normal dependency on unpublished crate `internal`"));
    assert!(errors[1].contains("build dependency on unpublished crate `internal`"));
    assert!(errors[2].contains("versioned dev-dependency `internal = \"1\"`"));

    let malformed = [
        (serde_json::json!({}), "has no dependencies array"),
        (
            serde_json::json!({"dependencies": [{}]}),
            "dependency[0] has no string name",
        ),
        (
            serde_json::json!({"dependencies": [{"name": "internal", "kind": 4, "req": "*"}]}),
            "has invalid kind",
        ),
        (
            serde_json::json!({"dependencies": [{"name": "internal", "kind": "dev"}]}),
            "has no string requirement",
        ),
    ];
    for (package, expected) in malformed {
        let error =
            validate_unpublished_dependencies("consumer", &package, &unpublished, &mut Vec::new())
                .expect_err("malformed dependency metadata rejects");
        assert!(error.contains(expected), "unexpected error: {error}");
    }
}

#[test]
fn checked_in_publish_workflow_script_and_docs_agree_with_the_manifest() {
    let mut errors = Vec::new();
    validate_publish_workflow_source(
        include_str!("../../../.github/workflows/publish.yml"),
        &mut errors,
    )
    .expect("parse publish workflow");
    validate_publish_script_source(
        include_str!("../../../scripts/publish-crate.sh"),
        &mut errors,
    );
    let manifest = release_manifest_contract().expect("checked-in release manifest");
    let packages = manifest.ordered_crates().collect::<Vec<_>>();
    validate_release_docs_source(
        include_str!("../../../docs/release.md"),
        &packages,
        &mut errors,
    );

    assert!(errors.is_empty(), "release contract drift: {errors:#?}");
}

#[test]
fn publish_workflow_requires_all_exact_sha_t803_lanes() {
    let workflow = include_str!("../../../.github/workflows/publish.yml")
        .replace("--t803-scope all", "--t803-scope cpu")
        .replace("t803 verify --scope all", "t803 verify --scope cpu")
        .replace(
            "            --report \"target/t803/release-evidence/j2k-t803-cuda-linux-x86_64-${candidate_sha}/cuda.json\" \\\n",
            "",
        )
        .replace(
            "            --report \"target/t803/release-evidence/j2k-t803-metal-macos-aarch64-${candidate_sha}/metal.json\"\n",
            "",
        );
    let mut errors = Vec::new();
    validate_publish_workflow_source(&workflow, &mut errors).expect("parse publish workflow");

    assert!(
        errors
            .iter()
            .any(|error| error.contains("--t803-scope all")),
        "CPU-only publication evidence was accepted: {errors:#?}"
    );
    assert!(
        errors
            .iter()
            .any(|error| error.contains("j2k-t803-cuda-linux-x86_64")),
        "missing CUDA report was accepted: {errors:#?}"
    );
    assert!(
        errors
            .iter()
            .any(|error| error.contains("j2k-t803-metal-macos-aarch64")),
        "missing Metal report was accepted: {errors:#?}"
    );
}

#[test]
fn publish_workflow_rejects_checkout_that_can_peel_an_annotated_tag() {
    let workflow = include_str!("../../../.github/workflows/publish.yml").replacen(
        "          ref: ${{ github.ref }}\n",
        "",
        1,
    );
    let mut errors = Vec::new();

    validate_publish_workflow_source(&workflow, &mut errors).expect("parse publish workflow");

    assert!(errors
        .iter()
        .any(|error| error.contains("bind all 2 checkout steps to the triggering ref")));
}

#[cfg(unix)]
#[test]
fn release_cpu_executes_the_complete_fake_cargo_plan() {
    let recording = RecordingProgram::new("release-command-test", "");
    let _cargo = use_test_cargo_program(recording.program().as_os_str().to_owned());

    release_cpu().expect("release CPU plan");

    let log = recording.log();
    assert!(log.starts_with("test --release -p j2k-core -p j2k-codec-math"));
    assert!(log.contains("-p j2k-native -p j2k -p j2k-tilecodec -p j2k-cli|"));
    assert_eq!(log.lines().count(), 1);
}

#[test]
fn release_integrity_rejects_invalid_modes_before_external_work() {
    let error = release_integrity(["--unknown".to_string()].into_iter())
        .expect_err("unknown release-integrity argument");
    assert!(error.contains("unknown release-integrity argument"));
}
