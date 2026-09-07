// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fmt::Write as _,
    fs::{self, File},
    io::Cursor,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};

use flate2::{write::GzEncoder, Compression};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

pub(crate) struct Harness {
    root: PathBuf,
    cargo: PathBuf,
    log: PathBuf,
    path: String,
}

impl Harness {
    pub(crate) fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "j2k-xtask-orchestration-{}-{}",
            std::process::id(),
            NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).expect("create orchestration test directory");
        let cargo = root.join("cargo.sh");
        let log = root.join("cargo.log");
        let metadata = prepare_metadata_fixture(&root);
        fs::write(
            &cargo,
            format!(
                "#!/bin/sh\nprintf '%s|RUSTDOCFLAGS=%s|RUST_TEST_THREADS=%s\\n' \"$*\" \"${{RUSTDOCFLAGS-unset}}\" \"${{RUST_TEST_THREADS-unset}}\" >> '{}'\nif [ \"$1\" = metadata ]; then exec cat '{}'; fi\nif [ \"$1\" = clippy ]; then printf '%s\\n' '{{\"reason\":\"build-finished\",\"success\":true}}'; exit 0; fi\nif [ \"$1\" = test ]; then printf 'test result: ok. 100 passed; 0 failed;\\n'; exit 0; fi\nif [ \"$1\" = -V ]; then printf 'cargo 1.96.0\\n'; fi\n",
                log.display(),
                metadata.display()
            ),
        )
        .expect("write fake Cargo");
        make_executable(&cargo, "fake Cargo");
        let cargo_machete = root.join("cargo-machete");
        fs::write(
            &cargo_machete,
            format!(
                "#!/bin/sh\nprintf '%s %s\\n' \"$(basename \"$0\")\" \"$*\" >> '{}'\n",
                log.display()
            ),
        )
        .expect("write fake cargo-machete");
        make_executable(&cargo_machete, "fake cargo-machete");
        let git = root.join("git");
        let baseline_snapshot = root.join("stable-api-baseline.public-api.txt");
        fs::write(&baseline_snapshot, synthetic_baseline_snapshot())
            .expect("write synthetic baseline API snapshot");
        let real_git = find_program("git");
        fs::write(
            &git,
            format!(
                "#!/bin/sh\nprintf 'git %s\\n' \"$*\" >> '{}'\nif [ \"$1\" = status ]; then exit 0; fi\nif [ \"$1\" = config ] && [ \"$2\" = --get ] && [ \"$3\" = remote.origin.url ]; then printf '%s\\n' 'git@example.invalid:frames-sg/j2k.git'; exit 0; fi\nif [ \"$1\" = rev-parse ] && [ \"$2\" = 'v0.10.0^{{commit}}' ]; then printf '%s\\n' 'e19fccf26e7b12a3601eed40a2f2143d94b6c7bc'; exit 0; fi\nif [ \"$1\" = show ] && [ \"$2\" = 'v0.10.0:docs/stable-api-1.0.public-api.txt' ]; then exec cat '{}'; fi\nif [ \"$1\" = rev-parse ] && [ \"${{2#v}}\" != \"$2\" ]; then printf 'unexpected release revision: %s\\n' \"$2\" >&2; exit 97; fi\nif [ \"$1\" = show ] && [ \"${{2#v}}\" != \"$2\" ]; then printf 'unexpected release object: %s\\n' \"$2\" >&2; exit 97; fi\nexec \"{}\" \"$@\"\n",
                log.display(),
                baseline_snapshot.display(),
                real_git.display()
            ),
        )
        .expect("write fake git wrapper");
        make_executable(&git, "fake git");
        let rustup = root.join("rustup");
        fs::write(
            &rustup,
            format!(
                "#!/bin/sh\nprintf 'rustup %s\\n' \"$*\" >> '{}'\ncase \"$*\" in\n  *'public-api --version'*) printf 'cargo-public-api 0.52.0\\n' ;;\n  *' cargo public-api '*) case \"${{RUSTDOCFLAGS-}}\" in *document-hidden*) printf 'pub struct SyntheticHidden\\n' ;; *) printf 'pub struct Synthetic\\n' ;; esac ;;\n  *'semver-checks --version'*) printf 'cargo-semver-checks 0.48.0\\n' ;;\nesac\n",
                log.display()
            ),
        )
        .expect("write fake rustup");
        make_executable(&rustup, "fake rustup");
        let python = root.join("python3");
        fs::write(
            &python,
            format!(
                "#!/bin/sh\nprintf 'python3 %s\\n' \"$*\" >> '{}'\n",
                log.display()
            ),
        )
        .expect("write fake Python");
        make_executable(&python, "fake Python");
        let path = format!(
            "{}:{}",
            root.display(),
            std::env::var("PATH").unwrap_or_default()
        );
        Self {
            root,
            cargo,
            log,
            path,
        }
    }

    pub(crate) fn run(&self, args: &[&str]) -> Output {
        self.run_with_env(args, &[])
    }

    pub(crate) fn run_with_env(&self, args: &[&str], envs: &[(&str, &str)]) -> Output {
        self.run_in_with_env(workspace_root(), args, envs)
    }

    pub(crate) fn run_in(&self, directory: &Path, args: &[&str]) -> Output {
        self.run_in_with_env(directory, args, &[])
    }

    fn run_in_with_env(&self, directory: &Path, args: &[&str], envs: &[(&str, &str)]) -> Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
        command
            .args(args)
            .current_dir(directory)
            .env("CARGO", &self.cargo)
            .env("PATH", &self.path);
        for key in [
            "CARGO_BUILD_TARGET",
            "CARGO_ENCODED_RUSTDOCFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
            "DOCS_RS",
            "GH_TOKEN",
            "GITHUB_REPOSITORY",
            "GITHUB_TOKEN",
            "MACOSX_DEPLOYMENT_TARGET",
            "RUSTC",
            "RUSTC_BOOTSTRAP",
            "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
            "RUSTDOC",
            "RUSTDOCFLAGS",
            "RUSTFLAGS",
        ] {
            command.env_remove(key);
        }
        for (key, value) in envs {
            command.env(key, value);
        }
        command.output().expect("run xtask child process")
    }

    pub(crate) fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    pub(crate) fn log(&self) -> String {
        fs::read_to_string(&self.log).expect("read fake Cargo log")
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).expect("remove orchestration test directory");
    }
}

pub(crate) fn assert_success(output: &Output, task: &str) {
    assert!(
        output.status.success(),
        "{task} failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest has workspace parent")
}

fn find_program(name: &str) -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
        .map(|directory| directory.join(name))
        .find(|candidate| candidate.is_file())
        .unwrap_or_else(|| panic!("failed to find real {name} on PATH"))
}

fn make_executable(path: &Path, label: &str) {
    let mut permissions = fs::metadata(path).expect(label).permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(path, permissions).expect(label);
}

fn prepare_metadata_fixture(root: &Path) -> PathBuf {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let output = Command::new(cargo)
        .args(["metadata", "--format-version", "1", "--locked"])
        .current_dir(workspace_root())
        .output()
        .expect("capture workspace metadata for orchestration fixture");
    assert!(
        output.status.success(),
        "workspace metadata fixture failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut metadata =
        serde_json::from_slice::<serde_json::Value>(&output.stdout).expect("parse cargo metadata");
    let workspace_members = metadata
        .get("workspace_members")
        .and_then(serde_json::Value::as_array)
        .expect("metadata includes workspace members");
    let packaged_members = metadata
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .expect("metadata includes packages")
        .iter()
        .filter(|package| {
            package
                .get("id")
                .is_some_and(|id| workspace_members.contains(id))
        })
        .filter(|package| match package.get("publish") {
            None | Some(serde_json::Value::Null) => true,
            Some(serde_json::Value::Array(registries)) => !registries.is_empty(),
            Some(_) => panic!("workspace package has malformed publish metadata"),
        })
        .map(|package| {
            let name = package
                .get("name")
                .and_then(serde_json::Value::as_str)
                .expect("workspace package has a name");
            let version = package
                .get("version")
                .and_then(serde_json::Value::as_str)
                .expect("workspace package has a version");
            (name.to_string(), version.to_string())
        })
        .collect::<Vec<_>>();
    let target = root.join("target");
    metadata["target_directory"] = serde_json::Value::String(target.to_string_lossy().into_owned());
    let metadata_path = root.join("metadata.json");
    fs::write(
        &metadata_path,
        serde_json::to_vec(&metadata).expect("serialize metadata fixture"),
    )
    .expect("write metadata fixture");
    for (package, version) in packaged_members {
        write_packaged_fixture(
            &target
                .join("package")
                .join(format!("{package}-{version}.crate")),
            &package,
            &version,
        );
    }
    metadata_path
}

fn write_packaged_fixture(path: &Path, package: &str, version: &str) {
    fs::create_dir_all(path.parent().expect("package fixture has parent"))
        .expect("create package fixture directory");
    let encoder = GzEncoder::new(
        File::create(path).expect("create package fixture"),
        Compression::default(),
    );
    let mut archive = tar::Builder::new(encoder);
    let manifest = format!("[package]\nname = \"{package}\"\nversion = \"{version}\"\n");
    let mut header = tar::Header::new_gnu();
    header.set_mode(0o644);
    header.set_size(u64::try_from(manifest.len()).expect("package manifest fixture fits in u64"));
    header.set_cksum();
    archive
        .append_data(
            &mut header,
            format!("{package}-{version}/Cargo.toml"),
            Cursor::new(manifest),
        )
        .expect("append package manifest fixture");
    archive
        .into_inner()
        .expect("finish package archive")
        .finish()
        .expect("finish compressed package fixture");
}

fn synthetic_baseline_snapshot() -> String {
    let mut snapshot =
        String::from("# J2K 1.0 Public API Snapshot\n\nGenerator: `cargo-public-api 0.52.0`.\n\n");
    for package in [
        "j2k",
        "j2k-core",
        "j2k-codec-math",
        "j2k-jpeg",
        "j2k-tilecodec",
        "j2k-jpeg-metal",
        "j2k-metal",
        "j2k-jpeg-cuda",
        "j2k-cuda",
        "j2k-transcode",
        "j2k-transcode-cuda",
        "j2k-metal-support",
        "j2k-transcode-metal",
        "j2k-native",
        "j2k-types",
        "j2k-cuda-runtime",
        "j2k-profile",
        "j2k-ml",
        "j2k-cuda-build-support",
        "j2k-cuda-j2k-engine",
        "j2k-cuda-jpeg-engine",
        "j2k-cuda-transcode-engine",
        "j2k-mpsgraph",
    ] {
        writeln!(
            snapshot,
            "## `{package}`\n\n```text\npub struct SyntheticBaseline\n```\n"
        )
        .expect("writing to a String cannot fail");
    }
    snapshot
}
