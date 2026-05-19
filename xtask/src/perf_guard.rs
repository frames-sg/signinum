use std::{
    collections::BTreeMap,
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BenchEstimate {
    pub(crate) id: String,
    pub(crate) median_ns: f64,
    pub(crate) median_lower_ns: f64,
    pub(crate) median_upper_ns: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegressionOutcome {
    pub(crate) id: String,
    pub(crate) baseline_ns: f64,
    pub(crate) current_ns: f64,
    pub(crate) delta_percent: f64,
    pub(crate) enforced: bool,
    pub(crate) threshold_exceeded: bool,
    pub(crate) regressed: bool,
}

#[derive(Debug, Clone)]
struct PerfGuardOptions {
    mode: PerfGuardMode,
    threshold_percent: f64,
    quick: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PerfGuardMode {
    GitRef { baseline_ref: String },
    RecordCurrent { name: String },
    CompareCurrent { name: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BenchCommand {
    package: &'static str,
    bench: &'static str,
    filter: Option<&'static str>,
    env: &'static [(&'static str, &'static str)],
}

const DEFAULT_BASELINE_REF: &str = "j2k-bench-original";
const DEFAULT_THRESHOLD_PERCENT: f64 = 10.0;
const MIN_ABSOLUTE_REGRESSION_NS: f64 = 100.0;
const BENCH_COMMANDS: &[BenchCommand] = &[
    BenchCommand {
        package: "signinum-j2k",
        bench: "public_api",
        filter: None,
        env: &[],
    },
    BenchCommand {
        package: "signinum-j2k-native",
        bench: "tier1_bitplane",
        filter: Some("htj2k_cleanup_decode/"),
        env: &[],
    },
    BenchCommand {
        package: "signinum-j2k-native",
        bench: "tier1_bitplane",
        filter: Some("htj2k_refinement_fixture_decode"),
        env: &[],
    },
    BenchCommand {
        package: "signinum-j2k-native",
        bench: "tier1_bitplane",
        filter: Some("htj2k_refinement_block_decode"),
        env: &[],
    },
    BenchCommand {
        package: "signinum-j2k-native",
        bench: "tier1_bitplane",
        filter: Some("htj2k_cleanup_encode/"),
        env: &[],
    },
    BenchCommand {
        package: "signinum-j2k-native",
        bench: "tier1_bitplane",
        filter: Some("htj2k_cleanup_encode_distribution"),
        env: &[],
    },
    BenchCommand {
        package: "signinum-j2k-native",
        bench: "htj2k_sigprop_phase",
        filter: None,
        env: &[],
    },
    BenchCommand {
        package: "signinum-j2k-metal",
        bench: "compare",
        filter: Some("wsi_tile_batch_region_scaled_rgb_q4"),
        env: &[("SIGNINUM_J2K_TILE_BATCH_SIZES", "16")],
    },
    BenchCommand {
        package: "signinum-j2k-metal",
        bench: "compare",
        filter: Some("htj2k_region_scaled_plan_build"),
        env: &[("SIGNINUM_J2K_TILE_BATCH_SIZES", "16")],
    },
    BenchCommand {
        package: "signinum-j2k-metal",
        bench: "compare",
        filter: Some("htj2k_feeder_coalesce"),
        env: &[("SIGNINUM_J2K_TILE_BATCH_SIZES", "16")],
    },
    BenchCommand {
        package: "signinum-j2k-metal",
        bench: "compare",
        filter: Some("htj2k_metal_route"),
        env: &[("SIGNINUM_J2K_TILE_BATCH_SIZES", "1,2,4,16")],
    },
];
const BENCH_SOURCE_FILES: &[&str] = &[
    "crates/signinum-j2k/benches/public_api.rs",
    "crates/signinum-j2k-metal/benches/common/mod.rs",
    "crates/signinum-j2k-metal/benches/compare.rs",
    "crates/signinum-j2k-native/benches/tier1_bitplane.rs",
    "crates/signinum-j2k-native/benches/htj2k_sigprop_phase.rs",
    "crates/signinum-j2k-native/fixtures/htj2k/openhtj2k_ds0_ht_09_b11.j2k",
];

pub(crate) fn j2k_perf_guard(args: impl Iterator<Item = String>) -> Result<(), String> {
    let options = PerfGuardOptions::parse(args)?;
    let root = repo_root()?;
    let perf_root = root.join("target").join("signinum-perf");
    fs::create_dir_all(&perf_root)
        .map_err(|err| format!("failed to create {}: {err}", perf_root.display()))?;

    let outcomes = match &options.mode {
        PerfGuardMode::GitRef { baseline_ref } => {
            let baseline_worktree = perf_root.join("baseline-worktree");
            recreate_baseline_worktree(&root, &baseline_worktree, baseline_ref)?;
            sync_benchmark_sources(&root, &baseline_worktree)?;

            let baseline_target = perf_root.join("baseline-target");
            let current_target = perf_root.join("current-target");
            reset_dir(&baseline_target)?;
            reset_dir(&current_target)?;

            run_benches(&baseline_worktree, &baseline_target, options.quick)?;
            run_benches(&root, &current_target, options.quick)?;

            let baseline = discover_estimates(&baseline_target.join("criterion"))?;
            let current = discover_estimates(&current_target.join("criterion"))?;
            compare_estimates(&baseline, &current, options.threshold_percent)?
        }
        PerfGuardMode::RecordCurrent { name } => {
            let target = perf_root.join("current-record-target");
            reset_dir(&target)?;
            run_benches(&root, &target, options.quick)?;
            let estimates = discover_estimates(&target.join("criterion"))?;
            let snapshot = current_snapshot_path(&perf_root, name)?;
            write_estimate_snapshot(&snapshot, &estimates)?;
            eprintln!(
                "Recorded J2K current-tree performance baseline `{name}` at {}",
                snapshot.display()
            );
            return Ok(());
        }
        PerfGuardMode::CompareCurrent { name } => {
            let snapshot = current_snapshot_path(&perf_root, name)?;
            let baseline = read_estimate_snapshot(&snapshot)?;
            let target = perf_root.join("current-compare-target");
            reset_dir(&target)?;
            run_benches(&root, &target, options.quick)?;
            let current = discover_estimates(&target.join("criterion"))?;
            compare_estimates(&baseline, &current, options.threshold_percent)?
        }
    };
    emit_report(&outcomes, options.threshold_percent);

    if outcomes.iter().any(|outcome| outcome.regressed) {
        Err("J2K performance guard found regressions".to_string())
    } else {
        Ok(())
    }
}

fn current_snapshot_path(perf_root: &Path, name: &str) -> Result<PathBuf, String> {
    validate_snapshot_name(name)?;
    Ok(perf_root
        .join("current-tree-baselines")
        .join(format!("{name}.json")))
}

fn validate_snapshot_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("current-tree baseline name must not be empty".to_string());
    }
    if name
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        Ok(())
    } else {
        Err(format!(
            "current-tree baseline name `{name}` may only contain ASCII letters, digits, '.', '-', and '_'"
        ))
    }
}

fn write_estimate_snapshot(path: &Path, estimates: &[BenchEstimate]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let mut sorted = estimates.to_vec();
    sorted.sort_by(|a, b| a.id.cmp(&b.id));
    let values = sorted
        .iter()
        .map(|estimate| {
            serde_json::json!({
                "id": estimate.id,
                "median_ns": estimate.median_ns,
                "median_lower_ns": estimate.median_lower_ns,
                "median_upper_ns": estimate.median_upper_ns,
            })
        })
        .collect::<Vec<_>>();
    let value = serde_json::json!({
        "version": 1,
        "estimates": values,
    });
    let data = serde_json::to_string_pretty(&value)
        .map_err(|err| format!("failed to serialize estimate snapshot: {err}"))?;
    fs::write(path, format!("{data}\n"))
        .map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn read_estimate_snapshot(path: &Path) -> Result<Vec<BenchEstimate>, String> {
    let data = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&data)
        .map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
    let version = value
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("{} is missing version", path.display()))?;
    if version != 1 {
        return Err(format!(
            "{} has unsupported estimate snapshot version {version}",
            path.display()
        ));
    }
    let raw_estimates = value
        .get("estimates")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{} is missing estimates", path.display()))?;
    let mut estimates = Vec::with_capacity(raw_estimates.len());
    for raw in raw_estimates {
        let id = raw
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("{} contains an estimate without id", path.display()))?
            .to_string();
        let median_ns = raw
            .get("median_ns")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| format!("{} estimate {id} is missing median_ns", path.display()))?;
        let median_lower_ns = raw
            .get("median_lower_ns")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| {
                format!(
                    "{} estimate {id} is missing median_lower_ns",
                    path.display()
                )
            })?;
        let median_upper_ns = raw
            .get("median_upper_ns")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| {
                format!(
                    "{} estimate {id} is missing median_upper_ns",
                    path.display()
                )
            })?;
        estimates.push(BenchEstimate {
            id,
            median_ns,
            median_lower_ns,
            median_upper_ns,
        });
    }
    estimates.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(estimates)
}

pub(crate) fn sync_benchmark_sources(source_root: &Path, target_root: &Path) -> Result<(), String> {
    for relative in BENCH_SOURCE_FILES {
        let source = source_root.join(relative);
        let target = target_root.join(relative);
        let parent = target.parent().ok_or_else(|| {
            format!(
                "benchmark source target has no parent directory: {}",
                target.display()
            )
        })?;
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
        fs::copy(&source, &target).map_err(|err| {
            format!(
                "failed to copy benchmark source {} to {}: {err}",
                source.display(),
                target.display()
            )
        })?;
    }

    Ok(())
}

pub(crate) fn compare_estimates(
    baseline: &[BenchEstimate],
    current: &[BenchEstimate],
    threshold_percent: f64,
) -> Result<Vec<RegressionOutcome>, String> {
    let current_by_id = current
        .iter()
        .map(|estimate| (estimate.id.as_str(), estimate))
        .collect::<BTreeMap<_, _>>();
    let mut outcomes = Vec::with_capacity(baseline.len());
    for base in baseline {
        let Some(now) = current_by_id.get(base.id.as_str()) else {
            if is_enforced_htj2k_perf_id(&base.id) {
                return Err(format!("missing current benchmark result for {}", base.id));
            }
            continue;
        };
        if base.median_ns <= 0.0 {
            return Err(format!(
                "baseline benchmark {} has non-positive median {}",
                base.id, base.median_ns
            ));
        }
        let delta_percent = ((now.median_ns - base.median_ns) / base.median_ns) * 100.0;
        let confident_absolute_delta_ns = now.median_lower_ns - base.median_upper_ns;
        let confident_delta_percent = (confident_absolute_delta_ns / base.median_upper_ns) * 100.0;
        let enforced = is_enforced_htj2k_perf_id(&base.id);
        let threshold_exceeded = confident_delta_percent > threshold_percent
            && confident_absolute_delta_ns > MIN_ABSOLUTE_REGRESSION_NS;
        outcomes.push(RegressionOutcome {
            id: base.id.clone(),
            baseline_ns: base.median_ns,
            current_ns: now.median_ns,
            delta_percent,
            enforced,
            threshold_exceeded,
            regressed: enforced && threshold_exceeded,
        });
    }
    Ok(outcomes)
}

fn is_enforced_htj2k_perf_id(id: &str) -> bool {
    matches!(
        id,
        stable_id if stable_id.starts_with("htj2k_cleanup_decode/")
            || stable_id.starts_with("htj2k_cleanup_encode/")
            || stable_id.starts_with("htj2k_cleanup_encode_distribution/")
            || stable_id.starts_with("htj2k_refinement_fixture_decode/")
            || stable_id.starts_with("htj2k_refinement_block_decode/")
            || stable_id.starts_with("htj2k_refinement_sigprop_phase/")
            || stable_id.starts_with("htj2k_cpuupload_decode_batch/")
            || stable_id.starts_with("htj2k_region_scaled_plan_build/")
            || stable_id.starts_with("htj2k_feeder_coalesce/")
            || stable_id.starts_with("j2k_public_decode/htj2k_")
            || stable_id.contains("_htj2k_")
            || stable_id.contains("/htj2k_")
    ) && !id.starts_with("htj2k_cleanup_encode_parallel_")
        && !id.starts_with("htj2k_metal_route/")
        && !id.starts_with("wsi_tile_batch_region_scaled_rgb_q4/signinum-cpu-staged-metal_")
}

pub(crate) fn discover_estimates(criterion_root: &Path) -> Result<Vec<BenchEstimate>, String> {
    let mut out = Vec::new();
    discover_estimates_inner(criterion_root, criterion_root, &mut out)?;
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

fn discover_estimates_inner(
    criterion_root: &Path,
    dir: &Path,
    out: &mut Vec<BenchEstimate>,
) -> Result<(), String> {
    if !dir.exists() {
        return Err(format!(
            "Criterion output directory does not exist: {}",
            criterion_root.display()
        ));
    }
    for entry in
        fs::read_dir(dir).map_err(|err| format!("failed to read {}: {err}", dir.display()))?
    {
        let entry =
            entry.map_err(|err| format!("failed to read {} entry: {err}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            discover_estimates_inner(criterion_root, &path, out)?;
            continue;
        }
        if path.file_name().and_then(|name| name.to_str()) != Some("estimates.json") {
            continue;
        }
        if path
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            != Some("new")
        {
            continue;
        }
        let id = estimate_id(criterion_root, &path)?;
        let (median_ns, median_lower_ns, median_upper_ns) = read_median_estimate(&path)?;
        out.push(BenchEstimate {
            id,
            median_ns,
            median_lower_ns,
            median_upper_ns,
        });
    }
    Ok(())
}

fn estimate_id(criterion_root: &Path, estimate_path: &Path) -> Result<String, String> {
    let bench_path = estimate_path
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| {
            format!(
                "invalid Criterion estimate path {}",
                estimate_path.display()
            )
        })?;
    let rel = bench_path.strip_prefix(criterion_root).map_err(|err| {
        format!(
            "failed to strip Criterion root {} from {}: {err}",
            criterion_root.display(),
            bench_path.display()
        )
    })?;
    let mut parts = Vec::new();
    for component in rel.components() {
        parts.push(component.as_os_str().to_string_lossy().into_owned());
    }
    Ok(parts.join("/"))
}

fn read_median_estimate(path: &Path) -> Result<(f64, f64, f64), String> {
    let data = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&data)
        .map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
    let median = value
        .get("median")
        .ok_or_else(|| format!("{} is missing median", path.display()))?;
    let point_estimate = median
        .get("point_estimate")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| format!("{} is missing median.point_estimate", path.display()))?;
    let confidence_interval = median
        .get("confidence_interval")
        .ok_or_else(|| format!("{} is missing median.confidence_interval", path.display()))?;
    let lower_bound = confidence_interval
        .get("lower_bound")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            format!(
                "{} is missing median.confidence_interval.lower_bound",
                path.display()
            )
        })?;
    let upper_bound = confidence_interval
        .get("upper_bound")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            format!(
                "{} is missing median.confidence_interval.upper_bound",
                path.display()
            )
        })?;
    Ok((point_estimate, lower_bound, upper_bound))
}

fn emit_report(outcomes: &[RegressionOutcome], threshold_percent: f64) {
    eprintln!("J2K performance guard threshold: +{threshold_percent:.2}% median");
    for outcome in outcomes {
        let status = if outcome.regressed {
            "REGRESSED"
        } else if outcome.threshold_exceeded && !outcome.enforced {
            "report"
        } else {
            "ok"
        };
        eprintln!(
            "{status:9} {:>9.2}% baseline={:.2}ns current={:.2}ns {}",
            outcome.delta_percent, outcome.baseline_ns, outcome.current_ns, outcome.id
        );
    }
}

fn run_benches(workdir: &Path, target_dir: &Path, quick: bool) -> Result<(), String> {
    for bench in BENCH_COMMANDS {
        let args = bench_args(*bench, quick);
        run_program_with_target(cargo(), &args, workdir, target_dir, bench.env)?;
    }
    Ok(())
}

fn bench_args(bench: BenchCommand, quick: bool) -> Vec<&'static str> {
    let mut args = vec!["bench", "-p", bench.package, "--bench", bench.bench];
    if bench.filter.is_some() || quick {
        args.push("--");
        if let Some(filter) = bench.filter {
            args.push(filter);
        }
        if quick {
            args.push("--quick");
        }
    }
    args
}

fn recreate_baseline_worktree(
    root: &Path,
    worktree: &Path,
    baseline_ref: &str,
) -> Result<(), String> {
    if worktree.exists() {
        run_program(
            OsString::from("git"),
            &["worktree", "remove", "--force", path_str(worktree)?],
            root,
        )?;
    }
    run_program(
        OsString::from("git"),
        &[
            "worktree",
            "add",
            "--detach",
            path_str(worktree)?,
            baseline_ref,
        ],
        root,
    )
}

fn reset_dir(path: &Path) -> Result<(), String> {
    if path.exists() {
        fs::remove_dir_all(path)
            .map_err(|err| format!("failed to remove {}: {err}", path.display()))?;
    }
    fs::create_dir_all(path).map_err(|err| format!("failed to create {}: {err}", path.display()))
}

fn repo_root() -> Result<PathBuf, String> {
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(|err| format!("failed to start `git rev-parse --show-toplevel`: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "`git rev-parse --show-toplevel` exited with {}",
            output.status
        ));
    }
    let path = String::from_utf8(output.stdout)
        .map_err(|err| format!("git root path was not UTF-8: {err}"))?;
    Ok(PathBuf::from(path.trim()))
}

fn run_program_with_target(
    program: OsString,
    args: &[&str],
    workdir: &Path,
    target_dir: &Path,
    extra_env: &[(&str, &str)],
) -> Result<(), String> {
    let display = program.to_string_lossy();
    let env_display = extra_env
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!(
        "+ CARGO_TARGET_DIR={}{}{} {} {}",
        target_dir.display(),
        if env_display.is_empty() { "" } else { " " },
        env_display,
        display,
        args.join(" ")
    );
    let mut command = Command::new(&program);
    command
        .args(args)
        .current_dir(workdir)
        .env("CARGO_TARGET_DIR", target_dir);
    for (key, value) in extra_env {
        command.env(key, value);
    }
    let status = command
        .status()
        .map_err(|err| format!("failed to start `{display}`: {err}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("`{display}` exited with {status}"))
    }
}

fn run_program(program: OsString, args: &[&str], workdir: &Path) -> Result<(), String> {
    let display = program.to_string_lossy();
    eprintln!("+ {} {}", display, args.join(" "));
    let status = Command::new(&program)
        .args(args)
        .current_dir(workdir)
        .status()
        .map_err(|err| format!("failed to start `{display}`: {err}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("`{display}` exited with {status}"))
    }
}

impl PerfGuardOptions {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut options = Self {
            mode: PerfGuardMode::GitRef {
                baseline_ref: DEFAULT_BASELINE_REF.to_string(),
            },
            threshold_percent: DEFAULT_THRESHOLD_PERCENT,
            quick: false,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--baseline-ref" => {
                    let baseline_ref = args
                        .next()
                        .ok_or_else(|| "--baseline-ref requires a value".to_string())?;
                    options.set_mode(PerfGuardMode::GitRef { baseline_ref })?;
                }
                "--record-current" => {
                    let name = args
                        .next()
                        .ok_or_else(|| "--record-current requires a value".to_string())?;
                    validate_snapshot_name(&name)?;
                    options.set_mode(PerfGuardMode::RecordCurrent { name })?;
                }
                "--compare-current" => {
                    let name = args
                        .next()
                        .ok_or_else(|| "--compare-current requires a value".to_string())?;
                    validate_snapshot_name(&name)?;
                    options.set_mode(PerfGuardMode::CompareCurrent { name })?;
                }
                "--threshold-percent" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "--threshold-percent requires a value".to_string())?;
                    options.threshold_percent = value
                        .parse::<f64>()
                        .map_err(|err| format!("invalid --threshold-percent `{value}`: {err}"))?;
                    if options.threshold_percent < 0.0 {
                        return Err("--threshold-percent must be non-negative".to_string());
                    }
                }
                "--quick" => options.quick = true,
                "--help" | "-h" => return Err(help_text()),
                other => {
                    return Err(format!(
                        "unknown j2k-perf-guard argument `{other}`\n{}",
                        help_text()
                    ))
                }
            }
        }
        Ok(options)
    }

    fn set_mode(&mut self, mode: PerfGuardMode) -> Result<(), String> {
        if self.mode
            != (PerfGuardMode::GitRef {
                baseline_ref: DEFAULT_BASELINE_REF.to_string(),
            })
        {
            return Err("choose only one baseline mode".to_string());
        }
        self.mode = mode;
        Ok(())
    }
}

fn help_text() -> String {
    "usage: cargo xtask j2k-perf-guard [--baseline-ref REF | --record-current NAME | --compare-current NAME] [--threshold-percent N] [--quick]".to_string()
}

fn path_str(path: &Path) -> Result<&str, String> {
    path.to_str()
        .ok_or_else(|| format!("path is not valid UTF-8: {}", path.display()))
}

fn cargo() -> OsString {
    env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

#[cfg(test)]
mod tests {
    use super::{
        bench_args, compare_estimates, is_enforced_htj2k_perf_id, read_estimate_snapshot,
        write_estimate_snapshot, BenchCommand, BenchEstimate, PerfGuardMode, PerfGuardOptions,
        BENCH_COMMANDS,
    };
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn perf_guard_parses_current_tree_record_mode() {
        let options = PerfGuardOptions::parse(
            ["--record-current", "htj2k-roi-baseline", "--quick"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap();

        assert_eq!(
            options.mode,
            PerfGuardMode::RecordCurrent {
                name: "htj2k-roi-baseline".to_string()
            }
        );
        assert!(options.quick);
    }

    #[test]
    fn perf_guard_parses_current_tree_compare_mode() {
        let options = PerfGuardOptions::parse(
            [
                "--compare-current",
                "htj2k-roi-baseline",
                "--threshold-percent",
                "7.5",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .unwrap();

        assert_eq!(
            options.mode,
            PerfGuardMode::CompareCurrent {
                name: "htj2k-roi-baseline".to_string()
            }
        );
        assert_eq!(options.threshold_percent, 7.5);
    }

    #[test]
    fn perf_guard_rejects_multiple_baseline_modes() {
        let error = PerfGuardOptions::parse(
            ["--record-current", "one", "--compare-current", "two"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap_err();

        assert!(
            error.contains("choose only one baseline mode"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn estimate_snapshot_round_trips_sorted_estimates() {
        let root = temp_dir("signinum-perf-snapshot-test");
        let path = root.join("baselines").join("htj2k.json");
        let estimates = vec![
            BenchEstimate {
                id: "z_group/z_case".to_string(),
                median_ns: 200.0,
                median_lower_ns: 190.0,
                median_upper_ns: 210.0,
            },
            BenchEstimate {
                id: "a_group/a_case".to_string(),
                median_ns: 100.0,
                median_lower_ns: 95.0,
                median_upper_ns: 105.0,
            },
        ];

        write_estimate_snapshot(&path, &estimates).unwrap();
        let round_trip = read_estimate_snapshot(&path).unwrap();

        assert_eq!(
            round_trip,
            vec![
                BenchEstimate {
                    id: "a_group/a_case".to_string(),
                    median_ns: 100.0,
                    median_lower_ns: 95.0,
                    median_upper_ns: 105.0,
                },
                BenchEstimate {
                    id: "z_group/z_case".to_string(),
                    median_ns: 200.0,
                    median_lower_ns: 190.0,
                    median_upper_ns: 210.0,
                },
            ]
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn perf_guard_runs_repeated_rgb_resident_metal_benchmark() {
        assert!(
            BENCH_COMMANDS.contains(&BenchCommand {
                package: "signinum-j2k-metal",
                bench: "compare",
                filter: Some("wsi_tile_batch_region_scaled_rgb_q4"),
                env: &[("SIGNINUM_J2K_TILE_BATCH_SIZES", "16")],
            }),
            "J2K perf guard must track the repeated RGB ROI+scaled resident Metal path"
        );
    }

    #[test]
    fn perf_guard_tracks_htj2k_maturation_benchmarks() {
        let expected = [
            BenchCommand {
                package: "signinum-j2k-native",
                bench: "tier1_bitplane",
                filter: Some("htj2k_cleanup_encode/"),
                env: &[],
            },
            BenchCommand {
                package: "signinum-j2k-native",
                bench: "tier1_bitplane",
                filter: Some("htj2k_cleanup_decode/"),
                env: &[],
            },
            BenchCommand {
                package: "signinum-j2k-native",
                bench: "htj2k_sigprop_phase",
                filter: None,
                env: &[],
            },
            BenchCommand {
                package: "signinum-j2k-metal",
                bench: "compare",
                filter: Some("htj2k_region_scaled_plan_build"),
                env: &[("SIGNINUM_J2K_TILE_BATCH_SIZES", "16")],
            },
            BenchCommand {
                package: "signinum-j2k-metal",
                bench: "compare",
                filter: Some("htj2k_feeder_coalesce"),
                env: &[("SIGNINUM_J2K_TILE_BATCH_SIZES", "16")],
            },
            BenchCommand {
                package: "signinum-j2k-metal",
                bench: "compare",
                filter: Some("htj2k_metal_route"),
                env: &[("SIGNINUM_J2K_TILE_BATCH_SIZES", "1,2,4,16")],
            },
        ];

        for command in expected {
            assert!(
                BENCH_COMMANDS.contains(&command),
                "J2K perf guard must track {command:?}"
            );
        }
    }

    #[test]
    fn perf_guard_errors_when_enforced_result_is_missing() {
        let error = compare_estimates(
            &[estimate(
                "htj2k_cleanup_encode/encode_64x64/2459041792",
                1_000.0,
            )],
            &[],
            10.0,
        )
        .unwrap_err();

        assert!(
            error.contains("missing current benchmark result"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn filtered_bench_command_passes_filter_before_quick_flag() {
        let command = BenchCommand {
            package: "signinum-j2k-metal",
            bench: "compare",
            filter: Some("wsi_tile_batch_region_scaled_rgb_q4"),
            env: &[("SIGNINUM_J2K_TILE_BATCH_SIZES", "16")],
        };

        assert_eq!(
            bench_args(command, true),
            vec![
                "bench",
                "-p",
                "signinum-j2k-metal",
                "--bench",
                "compare",
                "--",
                "wsi_tile_batch_region_scaled_rgb_q4",
                "--quick",
            ]
        );
    }

    #[test]
    fn perf_guard_enforces_stable_htj2k_rows_only() {
        assert!(is_enforced_htj2k_perf_id(
            "htj2k_cleanup_encode/encode_64x64/2459041792"
        ));
        assert!(is_enforced_htj2k_perf_id(
            "wsi_tile_batch_region_scaled_rgb_q4/signinum_htj2k_rgb_512_batch_16"
        ));
        assert!(is_enforced_htj2k_perf_id(
            "j2k_public_cpu_encode_matrix/rgb8_512_htj2k_external"
        ));
        assert!(!is_enforced_htj2k_perf_id(
            "htj2k_cleanup_encode_parallel_batch_size/rayon_par_iter_global_blocks/128"
        ));
        assert!(!is_enforced_htj2k_perf_id(
            "tier1_bitplane_encode/encode_64x64/default"
        ));
        assert!(!is_enforced_htj2k_perf_id(
            "wsi_tile_batch_region_scaled_rgb_q4/signinum-cpu-staged-metal_htj2k_rgb_512_batch_16"
        ));
        assert!(!is_enforced_htj2k_perf_id(
            "htj2k_metal_route/signinum-metal-resident_htj2k_rgb_512_batch_16"
        ));
    }

    #[test]
    fn perf_guard_reports_out_of_scope_regressions_without_failing() {
        let baseline = vec![
            estimate("htj2k_cleanup_encode/encode_64x64/2459041792", 1_000.0),
            estimate(
                "htj2k_cleanup_encode_parallel_batch_size/rayon_par_iter_global_blocks/128",
                1_000.0,
            ),
            estimate("tier1_bitplane_encode/encode_64x64/default", 1_000.0),
            estimate(
                "wsi_tile_batch_region_scaled_rgb_q4/signinum-cpu-staged-metal_htj2k_rgb_512_batch_16",
                1_000.0,
            ),
        ];
        let current = baseline
            .iter()
            .map(|estimate| BenchEstimate {
                id: estimate.id.clone(),
                median_ns: 1_300.0,
                median_lower_ns: 1_300.0,
                median_upper_ns: 1_310.0,
            })
            .collect::<Vec<_>>();

        let outcomes = compare_estimates(&baseline, &current, 10.0).unwrap();

        assert!(outcomes[0].enforced);
        assert!(outcomes[0].threshold_exceeded);
        assert!(outcomes[0].regressed);
        for outcome in &outcomes[1..] {
            assert!(!outcome.enforced, "{:?}", outcome);
            assert!(outcome.threshold_exceeded, "{:?}", outcome);
            assert!(!outcome.regressed, "{:?}", outcome);
        }
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let mut dir = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        dir.push(format!("{name}-{}-{nanos}", std::process::id()));
        dir
    }

    fn estimate(id: &str, median_ns: f64) -> BenchEstimate {
        BenchEstimate {
            id: id.to_string(),
            median_ns,
            median_lower_ns: median_ns,
            median_upper_ns: median_ns,
        }
    }
}
