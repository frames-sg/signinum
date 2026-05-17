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
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegressionOutcome {
    pub(crate) id: String,
    pub(crate) baseline_ns: f64,
    pub(crate) current_ns: f64,
    pub(crate) delta_percent: f64,
    pub(crate) regressed: bool,
}

#[derive(Debug, Clone)]
struct PerfGuardOptions {
    baseline_ref: String,
    threshold_percent: f64,
    quick: bool,
}

#[derive(Debug, Clone, Copy)]
struct BenchCommand {
    package: &'static str,
    bench: &'static str,
}

const DEFAULT_BASELINE_REF: &str = "j2k-bench-original";
const DEFAULT_THRESHOLD_PERCENT: f64 = 10.0;
const BENCH_COMMANDS: &[BenchCommand] = &[
    BenchCommand {
        package: "signinum-j2k",
        bench: "public_api",
    },
    BenchCommand {
        package: "signinum-j2k-native",
        bench: "tier1_bitplane",
    },
];

pub(crate) fn j2k_perf_guard(args: impl Iterator<Item = String>) -> Result<(), String> {
    let options = PerfGuardOptions::parse(args)?;
    let root = repo_root()?;
    let perf_root = root.join("target").join("signinum-perf");
    fs::create_dir_all(&perf_root)
        .map_err(|err| format!("failed to create {}: {err}", perf_root.display()))?;

    let baseline_worktree = perf_root.join("baseline-worktree");
    recreate_baseline_worktree(&root, &baseline_worktree, &options.baseline_ref)?;

    let baseline_target = perf_root.join("baseline-target");
    let current_target = perf_root.join("current-target");
    reset_dir(&baseline_target)?;
    reset_dir(&current_target)?;

    run_benches(&baseline_worktree, &baseline_target, options.quick)?;
    run_benches(&root, &current_target, options.quick)?;

    let baseline = discover_estimates(&baseline_target.join("criterion"))?;
    let current = discover_estimates(&current_target.join("criterion"))?;
    let outcomes = compare_estimates(&baseline, &current, options.threshold_percent)?;
    emit_report(&outcomes, options.threshold_percent);

    if outcomes.iter().any(|outcome| outcome.regressed) {
        Err("J2K performance guard found regressions".to_string())
    } else {
        Ok(())
    }
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
            return Err(format!("missing current benchmark result for {}", base.id));
        };
        if base.median_ns <= 0.0 {
            return Err(format!(
                "baseline benchmark {} has non-positive median {}",
                base.id, base.median_ns
            ));
        }
        let delta_percent = ((now.median_ns - base.median_ns) / base.median_ns) * 100.0;
        outcomes.push(RegressionOutcome {
            id: base.id.clone(),
            baseline_ns: base.median_ns,
            current_ns: now.median_ns,
            delta_percent,
            regressed: delta_percent > threshold_percent,
        });
    }
    Ok(outcomes)
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
        let median_ns = read_median_estimate(&path)?;
        out.push(BenchEstimate { id, median_ns });
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

fn read_median_estimate(path: &Path) -> Result<f64, String> {
    let data = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&data)
        .map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
    value
        .get("median")
        .and_then(|median| median.get("point_estimate"))
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| format!("{} is missing median.point_estimate", path.display()))
}

fn emit_report(outcomes: &[RegressionOutcome], threshold_percent: f64) {
    eprintln!("J2K performance guard threshold: +{threshold_percent:.2}% median");
    for outcome in outcomes {
        let status = if outcome.regressed { "REGRESSED" } else { "ok" };
        eprintln!(
            "{status:9} {:>9.2}% baseline={:.2}ns current={:.2}ns {}",
            outcome.delta_percent, outcome.baseline_ns, outcome.current_ns, outcome.id
        );
    }
}

fn run_benches(workdir: &Path, target_dir: &Path, quick: bool) -> Result<(), String> {
    for bench in BENCH_COMMANDS {
        let mut args = vec!["bench", "-p", bench.package, "--bench", bench.bench];
        if quick {
            args.extend_from_slice(&["--", "--quick"]);
        }
        run_program_with_target(cargo(), &args, workdir, target_dir)?;
    }
    Ok(())
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
) -> Result<(), String> {
    let display = program.to_string_lossy();
    eprintln!(
        "+ CARGO_TARGET_DIR={} {} {}",
        target_dir.display(),
        display,
        args.join(" ")
    );
    let status = Command::new(&program)
        .args(args)
        .current_dir(workdir)
        .env("CARGO_TARGET_DIR", target_dir)
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
            baseline_ref: DEFAULT_BASELINE_REF.to_string(),
            threshold_percent: DEFAULT_THRESHOLD_PERCENT,
            quick: false,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--baseline-ref" => {
                    options.baseline_ref = args
                        .next()
                        .ok_or_else(|| "--baseline-ref requires a value".to_string())?;
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
}

fn help_text() -> String {
    "usage: cargo xtask j2k-perf-guard [--baseline-ref REF] [--threshold-percent N] [--quick]"
        .to_string()
}

fn path_str(path: &Path) -> Result<&str, String> {
    path.to_str()
        .ok_or_else(|| format!("path is not valid UTF-8: {}", path.display()))
}

fn cargo() -> OsString {
    env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}
