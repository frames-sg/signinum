// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural GitHub Actions policy with workflow YAML as the source of truth.

mod github_expression;
mod yaml;

use std::collections::BTreeSet;
use std::path::{Component, PathBuf};

use serde_yaml_ng::Value;

use self::github_expression::{untrusted_event_run_tokens, value_contains_secret_reference};
use self::yaml::{
    display_key, is_numeric_zero, jobs, mapping_get, string_set, value_mapping, visit_mappings,
    workflow, workflows,
};
use super::repo_root;

#[test]
fn every_action_reference_is_pinned_to_a_forty_hex_sha() {
    let mut violations = Vec::new();
    for workflow in workflows() {
        visit_mappings(&workflow.document, &mut |mapping| {
            let Some(reference) = mapping_get(mapping, "uses").and_then(Value::as_str) else {
                return;
            };
            if reference.starts_with("./") || reference.starts_with("docker://") {
                return;
            }
            let pinned = reference.rsplit_once('@').is_some_and(|(_, revision)| {
                revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
            });
            if !pinned {
                violations.push(format!("{}: {reference}", workflow.file_name));
            }
        });
    }
    assert!(
        violations.is_empty(),
        "external actions must use a full forty-hex commit SHA:\n{}",
        violations.join("\n")
    );
}

#[test]
fn reusable_gpu_workflows_match_their_pinned_definitions() {
    for dispatcher in ["gpu-validation.yml", "gpu-benchmarks.yml"] {
        let caller = workflow(dispatcher);
        let reference = caller.document["jobs"]["run"]["uses"]
            .as_str()
            .expect("immutable GPU workflow call");
        let (path, revision) = reference
            .strip_prefix("frames-sg/j2k/")
            .and_then(|value| value.rsplit_once('@'))
            .expect("same-repository pinned workflow");
        let pinned = std::process::Command::new("git")
            .args(["show", &format!("{revision}:{path}")])
            .current_dir(repo_root())
            .output()
            .expect("read pinned workflow definition");
        assert!(pinned.status.success(), "workflow pin must resolve locally");
        let candidate =
            std::fs::read(repo_root().join(path)).expect("candidate workflow definition");
        assert_eq!(pinned.stdout, candidate,
            "update the dispatcher pin and runner-group approval when changing trusted GPU instructions");
    }
}

#[test]
fn every_workflow_declares_default_permissions_including_contents_read() {
    let mut violations = Vec::new();
    for workflow in workflows() {
        let root = value_mapping(&workflow.document, &workflow.file_name);
        let contents = mapping_get(root, "permissions")
            .and_then(Value::as_mapping)
            .and_then(|permissions| mapping_get(permissions, "contents"))
            .and_then(Value::as_str);
        if contents != Some("read") {
            violations.push(workflow.file_name.clone());
        }
    }
    assert!(
        violations.is_empty(),
        "workflows missing top-level contents: read permission: {violations:?}"
    );
}

#[test]
fn no_job_requests_a_write_permission_scope() {
    let mut violations = Vec::new();
    for workflow in workflows() {
        visit_mappings(&workflow.document, &mut |mapping| {
            let Some(permissions) = mapping_get(mapping, "permissions") else {
                return;
            };
            if permissions.as_str() == Some("write-all")
                || permissions.as_mapping().is_some_and(|scopes| {
                    scopes.values().any(|value| value.as_str() == Some("write"))
                })
            {
                violations.push(workflow.file_name.clone());
            }
        });
    }
    violations.sort();
    violations.dedup();
    assert!(
        violations.is_empty(),
        "workflow permission scopes must remain read-only: {violations:?}"
    );
}

#[test]
fn every_downloaded_release_asset_is_checksum_verified() {
    let mut violations = Vec::new();
    for workflow in workflows() {
        visit_mappings(&workflow.document, &mut |mapping| {
            let Some(script) = mapping_get(mapping, "run").and_then(Value::as_str) else {
                return;
            };
            if script.contains("/releases/download/") && !script.contains("sha256sum -c") {
                violations.push(workflow.file_name.clone());
            }
        });
    }
    assert!(
        violations.is_empty(),
        "downloaded release assets need same-step SHA-256 verification: {violations:?}"
    );
}

#[test]
fn the_secret_scan_redacts_findings_and_scans_full_history() {
    let secret_scan = workflow("secret-scan.yml");
    let mut full_checkout = false;
    let mut fail_closed_scan = false;
    visit_mappings(&secret_scan.document, &mut |mapping| {
        if mapping_get(mapping, "uses")
            .and_then(Value::as_str)
            .is_some_and(|reference| reference.starts_with("actions/checkout@"))
        {
            full_checkout = mapping_get(mapping, "with")
                .and_then(Value::as_mapping)
                .and_then(|with| mapping_get(with, "fetch-depth"))
                .is_some_and(is_numeric_zero);
        }
        if let Some(script) = mapping_get(mapping, "run").and_then(Value::as_str) {
            if script.contains("gitleaks detect") {
                fail_closed_scan = script.contains("--source .") && script.contains("--redact");
            }
        }
    });
    assert!(
        full_checkout,
        "secret scan checkout must fetch full history"
    );
    assert!(
        fail_closed_scan,
        "secret scan must redact findings and scan the repository"
    );
}

#[test]
fn stable_api_job_fetches_release_tags() {
    let full_validation = workflow("full-validation.yml");
    let root = value_mapping(&full_validation.document, &full_validation.file_name);
    let stable_api = jobs(root, &full_validation.file_name)
        .get(Value::String("stable-api".to_owned()))
        .and_then(Value::as_mapping)
        .expect("stable-api job");
    let steps = mapping_get(stable_api, "steps")
        .and_then(Value::as_sequence)
        .expect("stable-api steps");
    let checkout = steps
        .iter()
        .filter_map(Value::as_mapping)
        .find(|step| {
            mapping_get(step, "uses")
                .and_then(Value::as_str)
                .is_some_and(|reference| reference.starts_with("actions/checkout@"))
        })
        .expect("stable-api checkout step");
    let fetches_full_history = mapping_get(checkout, "with")
        .and_then(Value::as_mapping)
        .and_then(|with| mapping_get(with, "fetch-depth"))
        .is_some_and(is_numeric_zero);

    assert!(
        fetches_full_history,
        "stable-api checkout must fetch release tags used by repo-lint and stable-api"
    );
}

#[test]
fn rust_quality_job_fetches_release_tags() {
    let ci = workflow("ci.yml");
    let root = value_mapping(&ci.document, &ci.file_name);
    let rust_quality = jobs(root, &ci.file_name)
        .get(Value::String("rust-quality".to_owned()))
        .and_then(Value::as_mapping)
        .expect("rust-quality job");
    let steps = mapping_get(rust_quality, "steps")
        .and_then(Value::as_sequence)
        .expect("rust-quality steps");
    let checkout = steps
        .iter()
        .filter_map(Value::as_mapping)
        .find(|step| {
            mapping_get(step, "uses")
                .and_then(Value::as_str)
                .is_some_and(|reference| reference.starts_with("actions/checkout@"))
        })
        .expect("rust-quality checkout step");
    let fetches_full_history = mapping_get(checkout, "with")
        .and_then(Value::as_mapping)
        .and_then(|with| mapping_get(with, "fetch-depth"))
        .is_some_and(is_numeric_zero);

    assert!(
        fetches_full_history,
        "rust-quality checkout must fetch release tags used by repo-lint"
    );
}

#[test]
fn manual_publish_dispatch_is_dry_run_only() {
    let publish = workflow("publish.yml");
    let root = value_mapping(&publish.document, &publish.file_name);
    let triggers = mapping_get(root, "on")
        .and_then(Value::as_mapping)
        .expect("publish workflow trigger mapping");
    assert!(
        mapping_get(triggers, "workflow_dispatch").is_some(),
        "publish workflow must keep its manual dry-run entry point"
    );
    let dry_run = mapping_get(root, "env")
        .and_then(Value::as_mapping)
        .and_then(|environment| mapping_get(environment, "DRY_RUN_ONLY"))
        .and_then(Value::as_str)
        .expect("publish DRY_RUN_ONLY expression");
    assert!(
        dry_run.contains("github.event_name == 'workflow_dispatch'"),
        "manual publish dispatch must set DRY_RUN_ONLY"
    );

    let publish_job = jobs(root, &publish.file_name)
        .get(Value::String("publish".to_owned()))
        .and_then(Value::as_mapping)
        .expect("publish job");
    let condition = mapping_get(publish_job, "if")
        .and_then(Value::as_str)
        .expect("publish job event guard");
    assert!(
        condition.contains("github.event_name == 'push'"),
        "the credentialed publish job must not run for workflow_dispatch"
    );
}

#[test]
fn every_job_that_reads_a_secret_declares_a_protected_environment() {
    let mut violations = Vec::new();
    for workflow in workflows() {
        let root = value_mapping(&workflow.document, &workflow.file_name);
        for (job_name, job) in jobs(root, &workflow.file_name) {
            if value_contains_secret_reference(job)
                && job
                    .as_mapping()
                    .and_then(|mapping| mapping_get(mapping, "environment"))
                    .is_none_or(Value::is_null)
            {
                violations.push(format!("{}:{}", workflow.file_name, display_key(job_name)));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "jobs reading secrets need a protected environment:\n{}",
        violations.join("\n")
    );
}

#[test]
fn no_workflow_uses_pull_request_target_or_interpolates_untrusted_event_fields() {
    let mut violations = Vec::new();
    for workflow in workflows() {
        let root = value_mapping(&workflow.document, &workflow.file_name);
        let triggers = mapping_get(root, "on")
            .and_then(Value::as_mapping)
            .unwrap_or_else(|| panic!("{} trigger mapping", workflow.file_name));
        if mapping_get(triggers, "pull_request_target").is_some() {
            violations.push(format!("{}: pull_request_target", workflow.file_name));
        }
        visit_mappings(&workflow.document, &mut |mapping| {
            let Some(script) = mapping_get(mapping, "run").and_then(Value::as_str) else {
                return;
            };
            for token in untrusted_event_run_tokens(script) {
                violations.push(format!("{}: {token}", workflow.file_name));
            }
        });
    }
    assert!(
        violations.is_empty(),
        "untrusted event text must not reach shell bodies:\n{}",
        violations.join("\n")
    );
}

#[test]
fn every_required_pr_gate_is_listed_in_the_aggregate_job_needs() {
    let ci = workflow("ci.yml");
    let root = value_mapping(&ci.document, &ci.file_name);
    let jobs = jobs(root, &ci.file_name);
    let aggregate_name = Value::String("pr-required-checks".to_owned());
    let aggregate = jobs
        .get(&aggregate_name)
        .and_then(Value::as_mapping)
        .expect("PR aggregate job");

    let mut expected = jobs
        .keys()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    expected.remove("pr-required-checks");
    let actual = string_set(
        mapping_get(aggregate, "needs").expect("PR aggregate needs"),
        "PR aggregate needs",
    );
    assert_eq!(
        actual, expected,
        "PR aggregate needs must cover every sibling gate"
    );
}

#[test]
fn local_reusable_workflow_references_resolve_to_files_in_this_repo() {
    let mut violations = Vec::new();
    for workflow in workflows() {
        visit_mappings(&workflow.document, &mut |mapping| {
            let Some(reference) = mapping_get(mapping, "uses").and_then(Value::as_str) else {
                return;
            };
            if !reference.starts_with("./.github/workflows/") {
                return;
            }
            let relative = reference.trim_start_matches("./");
            let path = PathBuf::from(relative);
            if path
                .components()
                .any(|component| matches!(component, Component::ParentDir))
            {
                violations.push(format!("{}: path escape {reference}", workflow.file_name));
                return;
            }
            let target = repo_root().join(&path);
            if !target.is_file() {
                violations.push(format!("{}: missing {reference}", workflow.file_name));
                return;
            }
            let Some(target_name) = target.file_name().and_then(|name| name.to_str()) else {
                violations.push(format!("{}: non-UTF-8 {reference}", workflow.file_name));
                return;
            };
            let target_workflow = yaml::workflow(target_name);
            let target_root = value_mapping(&target_workflow.document, &target_workflow.file_name);
            let callable = mapping_get(target_root, "on")
                .and_then(Value::as_mapping)
                .is_some_and(|triggers| mapping_get(triggers, "workflow_call").is_some());
            if !callable {
                violations.push(format!(
                    "{}: target is not reusable {reference}",
                    workflow.file_name
                ));
            }
        });
    }
    assert!(
        violations.is_empty(),
        "local reusable workflow references must resolve in-repo:\n{}",
        violations.join("\n")
    );
}
