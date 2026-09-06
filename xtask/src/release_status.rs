// SPDX-License-Identifier: MIT OR Apache-2.0

use std::env;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

use crate::process::{self, CommandContext};

const CI_WORKFLOW: &str = "full-validation.yml";
const CI_BRANCH: &str = "main";
const RELEASE_CANDIDATE_JOB: &str = "Release candidate aggregate";
const GPU_WORKFLOW: &str = "gpu-validation.yml";
const CUDA_JOB: &str = "GPU / CUDA full release validation";
const METAL_JOB: &str = "GPU / Metal full release validation";

#[derive(Debug, Eq, PartialEq)]
struct Options {
    sha: String,
    repository: Option<String>,
    scope: String,
}

pub(crate) fn release_status(args: impl Iterator<Item = String>) -> Result<(), String> {
    let options = parse_options(args)?;
    let root = workspace_root()?;
    let env_repository = optional_unicode_env("GITHUB_REPOSITORY")?;
    let remote = if options.repository.is_none() && env_repository.is_none() {
        Some(origin_remote(&root)?)
    } else {
        None
    };
    let repository = resolve_repository(
        options.repository.as_deref(),
        env_repository.as_deref(),
        remote.as_deref(),
    )?;
    let token_env = select_token_env(
        env::var_os("GH_TOKEN").as_deref(),
        env::var_os("GITHUB_TOKEN").as_deref(),
    )?;

    let verifier = root.join("scripts/github_actions_verify.py");
    if !verifier.is_file() {
        return Err(format!(
            "GitHub Actions verifier is missing: {}",
            verifier.display()
        ));
    }

    let evidence_dir = root.join("target").join("t803").join(format!(
        "release-status-{}-{}",
        options.sha.clone(),
        std::process::id()
    ));
    let evidence_dir_text = evidence_dir.to_string_lossy().into_owned();

    let command_args = vec![
        "scripts/github_actions_verify.py".to_string(),
        "verify-candidate".to_string(),
        "--repository".to_string(),
        repository,
        "--candidate-sha".to_string(),
        options.sha.clone(),
        "--token-env".to_string(),
        token_env.to_string(),
        "--ci-workflow".to_string(),
        CI_WORKFLOW.to_string(),
        "--ci-branch".to_string(),
        CI_BRANCH.to_string(),
        "--aggregate-job".to_string(),
        RELEASE_CANDIDATE_JOB.to_string(),
        "--gpu-workflow".to_string(),
        GPU_WORKFLOW.to_string(),
        "--cuda-job".to_string(),
        CUDA_JOB.to_string(),
        "--metal-job".to_string(),
        METAL_JOB.to_string(),
        "--t803-scope".to_string(),
        options.scope.clone(),
        "--t803-out-dir".to_string(),
        evidence_dir_text,
    ];
    process::run_command_owned(
        OsString::from("python3"),
        &command_args,
        CommandContext::new().current_dir(&root),
    )?;
    verify_t803_reports(&root, &evidence_dir, &options.sha, &options.scope)
}

fn verify_t803_reports(
    root: &Path,
    evidence_dir: &Path,
    candidate_sha: &str,
    scope: &str,
) -> Result<(), String> {
    let mut args = [
        "run",
        "--quiet",
        "-p",
        "j2k-t803",
        "--features",
        "runner",
        "--bin",
        "j2k-t803-runner",
        "--",
        "verify",
        "--scope",
        scope,
        "--candidate-sha",
        candidate_sha,
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    for report in t803_report_paths(evidence_dir, candidate_sha, scope) {
        args.push("--report".to_string());
        args.push(report.to_string_lossy().into_owned());
    }
    process::run_command_owned(
        process::cargo(),
        &args,
        CommandContext::new().current_dir(root),
    )
}

fn t803_report_paths(evidence_dir: &Path, candidate_sha: &str, scope: &str) -> Vec<PathBuf> {
    let mut lanes = Vec::new();
    if matches!(scope, "cpu" | "all") {
        lanes.extend([
            ("cpu-linux-x86_64", "cpu"),
            ("cpu-macos-aarch64", "cpu"),
            ("cpu-windows-x86_64", "cpu"),
        ]);
    }
    if matches!(scope, "cuda" | "all") {
        lanes.push(("cuda-linux-x86_64", "cuda"));
    }
    if matches!(scope, "metal" | "all") {
        lanes.push(("metal-macos-aarch64", "metal"));
    }
    lanes
        .into_iter()
        .map(|(lane, stem)| {
            evidence_dir
                .join(format!("j2k-t803-{lane}-{candidate_sha}"))
                .join(format!("{stem}.json"))
        })
        .collect()
}

fn parse_options(args: impl Iterator<Item = String>) -> Result<Options, String> {
    let mut sha = None;
    let mut repository = None;
    let mut scope = None;
    let mut args = args;
    while let Some(arg) = args.next() {
        let slot = match arg.as_str() {
            "--sha" => &mut sha,
            "--repository" => &mut repository,
            "--scope" => &mut scope,
            "-h" | "--help" => return Err(usage()),
            other => {
                return Err(format!(
                    "unknown release-status argument `{other}`\n{}",
                    usage()
                ))
            }
        };
        if slot.is_some() {
            return Err(format!(
                "release-status argument `{arg}` was provided more than once"
            ));
        }
        *slot = Some(
            args.next()
                .ok_or_else(|| format!("release-status argument `{arg}` requires a value"))?,
        );
    }

    let sha = normalize_sha(&sha.ok_or_else(|| format!("--sha is required\n{}", usage()))?)?;
    let repository = repository.as_deref().map(validate_repository).transpose()?;
    let scope = validate_scope(scope.as_deref().unwrap_or("all"))?;
    Ok(Options {
        sha,
        repository,
        scope,
    })
}

fn validate_scope(value: &str) -> Result<String, String> {
    match value {
        "cpu" | "cuda" | "metal" | "all" => Ok(value.to_string()),
        _ => Err("--scope must be one of cpu, cuda, metal, or all".to_string()),
    }
}

fn normalize_sha(value: &str) -> Result<String, String> {
    if value.len() != 40 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("--sha must be exactly 40 hexadecimal characters".to_string());
    }
    Ok(value.to_ascii_lowercase())
}

fn workspace_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "xtask manifest directory has no workspace parent".to_string())
}

fn optional_unicode_env(name: &str) -> Result<Option<String>, String> {
    match env::var(name) {
        Ok(value) if value.is_empty() => Ok(None),
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(_)) => {
            Err(format!("{name} must contain valid Unicode owner/name text"))
        }
    }
}

fn origin_remote(root: &Path) -> Result<String, String> {
    let output = process::command_output(
        OsString::from("git"),
        &["config", "--get", "remote.origin.url"],
        CommandContext::new().current_dir(root),
    )
    .map_err(|err| format!("could not derive repository from remote.origin.url: {err}"))?;
    if !output.status.success() {
        return Err(
            "repository is required: pass --repository, set GITHUB_REPOSITORY, or configure remote.origin.url"
                .to_string(),
        );
    }
    let remote = String::from_utf8(output.stdout)
        .map_err(|_| "remote.origin.url must contain valid Unicode".to_string())?;
    let remote = remote.trim();
    if remote.is_empty() {
        return Err("remote.origin.url is empty".to_string());
    }
    Ok(remote.to_string())
}

fn resolve_repository(
    explicit: Option<&str>,
    environment: Option<&str>,
    remote: Option<&str>,
) -> Result<String, String> {
    if let Some(repository) = explicit.or(environment) {
        return validate_repository(repository);
    }
    repository_from_remote(remote.ok_or_else(|| {
        "repository is required: pass --repository, set GITHUB_REPOSITORY, or configure remote.origin.url"
            .to_string()
    })?)
}

fn repository_from_remote(remote: &str) -> Result<String, String> {
    let remote = remote.trim().trim_end_matches('/');
    let path = if let Some((_scheme, rest)) = remote.split_once("://") {
        rest.split_once('/')
            .map(|(_authority, path)| path)
            .ok_or_else(|| "remote.origin.url does not contain an owner/name path".to_string())?
    } else if let Some((authority, path)) = remote.split_once(':') {
        if authority.contains('@') {
            path
        } else {
            remote
        }
    } else {
        remote
    };
    let path = path.trim_matches('/');
    let path = path.strip_suffix(".git").unwrap_or(path);
    validate_repository(path)
}

fn validate_repository(value: &str) -> Result<String, String> {
    let mut components = value.split('/');
    let owner = components.next().unwrap_or_default();
    let repository = components.next().unwrap_or_default();
    if components.next().is_some()
        || !valid_repository_component(owner)
        || !valid_repository_component(repository)
    {
        return Err("repository must use non-empty owner/name form".to_string());
    }
    Ok(format!("{owner}/{repository}"))
}

fn valid_repository_component(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && !value.to_ascii_lowercase().ends_with(".git")
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn select_token_env(
    gh_token: Option<&OsStr>,
    github_token: Option<&OsStr>,
) -> Result<&'static str, String> {
    if gh_token.is_some_and(|value| !value.is_empty()) {
        Ok("GH_TOKEN")
    } else if github_token.is_some_and(|value| !value.is_empty()) {
        Ok("GITHUB_TOKEN")
    } else {
        Err("GitHub API authentication is required: set GH_TOKEN or GITHUB_TOKEN".to_string())
    }
}

fn usage() -> String {
    "usage: cargo xtask release-status --sha <40-hex-commit> [--repository owner/name] [--scope cpu|cuda|metal|all]".to_string()
}

#[cfg(test)]
mod tests;
