use std::env;
use std::ffi::OsString;
use std::process::{Command, ExitCode};

const PUBLISHABLE_PACKAGES: &[&str] = &[
    "ashlar-core",
    "ashlar-j2k-native",
    "ashlar-j2k-compare",
    "ashlar-tilecodec",
    "ashlar-jpeg",
    "ashlar-jpeg-metal",
    "ashlar-jpeg-cuda",
    "ashlar-j2k",
    "ashlar-j2k-metal",
    "ashlar-j2k-cuda",
    "ashlar-cli",
];

const CPU_RELEASE_PACKAGES: &[&str] = &[
    "ashlar-core",
    "ashlar-jpeg",
    "ashlar-j2k-native",
    "ashlar-j2k",
    "ashlar-tilecodec",
    "ashlar-cli",
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
    run_cargo(&["test", "--workspace", "--all-targets", "--all-features"])
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
    run_cargo(&["bench", "-p", "ashlar-jpeg", "--no-run"])?;
    run_cargo(&["bench", "-p", "ashlar-jpeg-metal", "--no-run"])?;
    run_cargo(&["bench", "-p", "ashlar-j2k-metal", "--no-run"])?;
    run_cargo(&[
        "bench",
        "-p",
        "ashlar-tilecodec",
        "--bench",
        "compare",
        "--no-run",
    ])
}

fn fuzz_build() -> Result<(), String> {
    run_cargo(&[
        "check",
        "--manifest-path",
        "crates/ashlar-j2k/fuzz/Cargo.toml",
    ])?;
    run_cargo(&[
        "check",
        "--manifest-path",
        "crates/ashlar-tilecodec/fuzz/Cargo.toml",
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
    run_cargo(&["check", "-p", "ashlar-core", "--target", NO_STD_TARGET])?;
    run_cargo(&[
        "check",
        "-p",
        "ashlar-j2k-native",
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
    run_cargo(&[
        "test",
        "--release",
        "-p",
        "ashlar-jpeg-metal",
        "-p",
        "ashlar-j2k-metal",
    ])
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
    for package in PUBLISHABLE_PACKAGES {
        run_cargo(&["package", "-p", package, "--no-verify"])?;
    }
    Ok(())
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
           package       package publishable crates without verification"
    );
}
