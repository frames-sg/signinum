use std::env;
use std::ffi::OsString;
use std::process::{Command, ExitCode};

const PUBLISHABLE_PACKAGES: &[&str] = &[
    "signinum-core",
    "signinum-cuda-runtime",
    "signinum-j2k-native",
    "signinum-jpeg",
    "signinum-tilecodec",
    "signinum-j2k",
    "signinum-jpeg-metal",
    "signinum-j2k-metal",
    "signinum-jpeg-cuda",
    "signinum-j2k-cuda",
    "signinum-cli",
    "signinum",
];

const REGISTRY_INDEPENDENT_PACKAGES: &[&str] = &[
    "signinum-core",
    "signinum-cuda-runtime",
    "signinum-j2k-native",
];

const STAGED_DEPENDENCY_PACKAGES: &[&str] = &[
    "signinum-jpeg",
    "signinum-tilecodec",
    "signinum-j2k",
    "signinum-jpeg-metal",
    "signinum-j2k-metal",
    "signinum-jpeg-cuda",
    "signinum-j2k-cuda",
    "signinum-cli",
    "signinum",
];

const CPU_RELEASE_PACKAGES: &[&str] = &[
    "signinum-core",
    "signinum-jpeg",
    "signinum-j2k-native",
    "signinum-j2k",
    "signinum-tilecodec",
    "signinum-cli",
];

const NO_STD_TARGET: &str = "aarch64-unknown-none";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("xtask failed: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let task = env::args().nth(1).unwrap_or_else(|| "help".to_string());
    match task.as_str() {
        "fmt" => fmt(),
        "clippy" => clippy(),
        "test" => test(),
        "doc" | "docs" => doc(),
        "typos" => typos(),
        "bench-build" => bench_build(),
        "fuzz-build" => fuzz_build(),
        "deny" => deny(),
        "no-std" => no_std(),
        "release-cpu" => release_cpu(),
        "release-metal" => release_metal(),
        "coverage" => coverage(),
        "package" => package(),
        "ci" => ci(),
        "help" | "-h" | "--help" => {
            print_help();
            Ok(())
        }
        other => Err(format!("unknown task `{other}`")),
    }
}

fn ci() -> Result<(), String> {
    fmt()?;
    clippy()?;
    test()?;
    doc()
}

fn fmt() -> Result<(), String> {
    run_cargo(&["fmt", "--all", "--", "--check"])
}

fn clippy() -> Result<(), String> {
    run_cargo(&[
        "clippy",
        "--workspace",
        "--all-targets",
        "--all-features",
        "--",
        "-D",
        "warnings",
    ])
}

fn test() -> Result<(), String> {
    if env::consts::OS != "macos" {
        return test_workspace_without_benches(&[]);
    }

    test_workspace_without_benches(&["--exclude", "signinum-j2k-metal"])?;
    if skip_j2k_metal_runtime_on_hosted_github_macos() {
        eprintln!(
            "skipping signinum-j2k-metal runtime tests on GitHub-hosted macOS; \
             self-hosted gpu-validation runs the Metal runtime suite"
        );
        return test_package_without_benches("signinum-j2k-metal", true);
    }
    test_package_without_benches("signinum-j2k-metal", false)
}

fn test_workspace_without_benches(extra_args: &[&str]) -> Result<(), String> {
    let mut test_args = vec![
        "test",
        "--workspace",
        "--all-features",
        "--lib",
        "--bins",
        "--tests",
    ];
    test_args.extend_from_slice(extra_args);
    run_cargo(&test_args)?;

    let mut doc_args = vec!["test", "--workspace", "--all-features", "--doc"];
    doc_args.extend_from_slice(extra_args);
    run_cargo(&doc_args)
}

fn test_package_without_benches(package: &str, no_run: bool) -> Result<(), String> {
    let mut test_args = vec![
        "test",
        "-p",
        package,
        "--all-features",
        "--lib",
        "--bins",
        "--tests",
    ];
    if no_run {
        test_args.push("--no-run");
    }

    if no_run {
        return run_cargo(&test_args);
    }

    run_cargo_with_env(&test_args, &[("RUST_TEST_THREADS", "1")])?;
    run_cargo(&["test", "-p", package, "--all-features", "--doc"])
}

fn doc() -> Result<(), String> {
    run_cargo_with_env(
        &["doc", "--workspace", "--all-features", "--no-deps"],
        &[("RUSTDOCFLAGS", "-D warnings")],
    )
}

fn typos() -> Result<(), String> {
    run_program(OsString::from("typos"), &[], &[])
}

fn bench_build() -> Result<(), String> {
    run_cargo(&["bench", "-p", "signinum-jpeg", "--no-run"])?;
    run_cargo(&["bench", "-p", "signinum-jpeg-metal", "--no-run"])?;
    run_cargo(&[
        "bench",
        "-p",
        "signinum-jpeg-cuda",
        "--bench",
        "device_decode",
        "--features",
        "cuda-runtime",
        "--no-run",
    ])?;
    run_cargo(&["bench", "-p", "signinum-j2k-metal", "--no-run"])?;
    run_cargo(&[
        "bench",
        "-p",
        "signinum-tilecodec",
        "--bench",
        "compare",
        "--no-run",
    ])
}

fn fuzz_build() -> Result<(), String> {
    run_cargo(&[
        "check",
        "--manifest-path",
        "crates/signinum-j2k/fuzz/Cargo.toml",
    ])?;
    run_cargo(&[
        "check",
        "--manifest-path",
        "crates/signinum-jpeg/fuzz/Cargo.toml",
    ])?;
    run_cargo(&[
        "check",
        "--manifest-path",
        "crates/signinum-tilecodec/fuzz/Cargo.toml",
    ])
}

fn deny() -> Result<(), String> {
    run_cargo(&["deny", "check", "licenses", "advisories", "bans", "sources"])
}

fn no_std() -> Result<(), String> {
    run_program(
        OsString::from("rustup"),
        &["target", "add", NO_STD_TARGET],
        &[],
    )?;
    run_cargo(&["check", "-p", "signinum-core", "--target", NO_STD_TARGET])?;
    run_cargo(&[
        "check",
        "-p",
        "signinum-j2k-native",
        "--no-default-features",
        "--target",
        NO_STD_TARGET,
    ])
}

fn release_cpu() -> Result<(), String> {
    let mut args = vec!["test", "--release"];
    for package in CPU_RELEASE_PACKAGES {
        args.push("-p");
        args.push(package);
    }
    run_cargo(&args)
}

fn release_metal() -> Result<(), String> {
    if env::consts::OS != "macos" {
        eprintln!("skipping Metal release tests on {}", env::consts::OS);
        return Ok(());
    }
    if skip_j2k_metal_runtime_on_hosted_github_macos() {
        eprintln!(
            "skipping signinum-j2k-metal release runtime tests on GitHub-hosted macOS; \
             self-hosted gpu-validation runs the Metal runtime suite"
        );
        run_cargo_with_env(
            &["test", "--release", "-p", "signinum-jpeg-metal"],
            &[("RUST_TEST_THREADS", "1")],
        )?;
        return run_cargo(&["test", "--release", "-p", "signinum-j2k-metal", "--no-run"]);
    }
    run_cargo_with_env(
        &[
            "test",
            "--release",
            "-p",
            "signinum-jpeg-metal",
            "-p",
            "signinum-j2k-metal",
        ],
        &[("RUST_TEST_THREADS", "1")],
    )
}

fn skip_j2k_metal_runtime_on_hosted_github_macos() -> bool {
    env::consts::OS == "macos"
        && env::var_os("GITHUB_ACTIONS").is_some()
        && env::var_os("SIGNINUM_RUN_HOSTED_J2K_METAL_RUNTIME_TESTS").is_none()
}

fn coverage() -> Result<(), String> {
    run_cargo(&[
        "llvm-cov",
        "--workspace",
        "--all-features",
        "--lcov",
        "--output-path",
        "lcov.info",
    ])
}

fn package() -> Result<(), String> {
    ensure_clean_worktree()?;
    for package in PUBLISHABLE_PACKAGES {
        run_cargo(&["package", "-p", package, "--list"])?;
    }
    for package in REGISTRY_INDEPENDENT_PACKAGES {
        run_cargo(&["package", "-p", package, "--no-verify"])?;
    }
    for package in STAGED_DEPENDENCY_PACKAGES {
        eprintln!(
            "skipping strict package verification for {package}: unpublished workspace dependencies are staged for publication; `cargo package --list` validated package contents"
        );
    }
    Ok(())
}

fn ensure_clean_worktree() -> Result<(), String> {
    let output = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map_err(|err| format!("failed to start `git status --porcelain`: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "`git status --porcelain` exited with {}",
            output.status
        ));
    }

    let status = String::from_utf8_lossy(&output.stdout);
    if status.trim().is_empty() {
        Ok(())
    } else {
        Err(format!(
            "working tree must be clean before packaging:\n{status}"
        ))
    }
}

fn run_cargo(args: &[&str]) -> Result<(), String> {
    run_cargo_with_env(args, &[])
}

fn run_cargo_with_env(args: &[&str], envs: &[(&str, &str)]) -> Result<(), String> {
    run_program(cargo(), args, envs)
}

fn run_program(program: OsString, args: &[&str], envs: &[(&str, &str)]) -> Result<(), String> {
    let display = program.to_string_lossy();
    eprintln!("+ {} {}", display, args.join(" "));
    let mut command = Command::new(&program);
    command.args(args);
    for (key, value) in envs {
        command.env(key, value);
    }
    let status = command
        .status()
        .map_err(|err| format!("failed to start `{}`: {err}", display))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("`{}` exited with {status}", display))
    }
}

fn cargo() -> OsString {
    env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

fn print_help() {
    println!(
        "usage: cargo xtask <task>\n\n\
         tasks:\n\
           ci            fmt, clippy, test, and docs\n\
           fmt           check rustfmt\n\
           clippy        run clippy with warnings denied\n\
           test          run workspace tests\n\
           doc           build workspace docs with warnings denied\n\
           typos         run typos\n\
           bench-build   compile benchmark targets\n\
           fuzz-build    compile fuzz harnesses\n\
           deny          run cargo-deny\n\
           no-std        check no_std-compatible codec crates\n\
           release-cpu   run release-mode CPU codec tests\n\
           release-metal run release-mode Metal tests on macOS\n\
           coverage      generate lcov.info with cargo-llvm-cov\n\
           package       preflight CPU-first 1.0 packaging from a clean worktree; strict for registry-independent crates and list-only for staged dependencies"
    );
}
