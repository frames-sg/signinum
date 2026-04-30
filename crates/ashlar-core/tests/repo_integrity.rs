// SPDX-License-Identifier: Apache-2.0

use std::{fs, path::Path};

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
}

#[test]
fn conformance_manifest_lists_all_committed_jpeg_inputs() {
    let root = repo_root();
    let conformance = root.join("corpus/conformance");
    let manifest =
        fs::read_to_string(conformance.join("manifest.json")).expect("read conformance manifest");

    for entry in fs::read_dir(&conformance).expect("read conformance dir") {
        let entry = entry.expect("read conformance entry");
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("jpg") {
            continue;
        }
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("utf-8 fixture filename");
        assert!(
            manifest.contains(&format!("\"{filename}\"")),
            "conformance fixture {filename} is missing from manifest.json"
        );
    }
}

#[test]
fn corpus_readme_does_not_claim_committed_fixtures_are_absent() {
    let readme =
        fs::read_to_string(repo_root().join("corpus/README.md")).expect("read corpus README");

    assert!(
        !readme.contains("intentionally empty"),
        "corpus README still claims the committed fixture corpus is empty"
    );
}

#[test]
fn adapter_crates_do_not_import_codec_private_modules() {
    let root = repo_root();
    let adapter_crates = [
        "crates/ashlar-jpeg-metal",
        "crates/ashlar-jpeg-cuda",
        "crates/ashlar-j2k-metal",
        "crates/ashlar-j2k-cuda",
    ];

    for crate_dir in adapter_crates {
        for path in rust_sources(&root.join(crate_dir)) {
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
            assert!(
                !source.contains("::__private") && !source.contains(" __private::"),
                "adapter source {} imports a codec __private module; use the public adapter API",
                path.strip_prefix(root).unwrap_or(&path).display()
            );
        }
    }
}

fn rust_sources(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    collect_rust_sources(dir, &mut out);
    out
}

fn collect_rust_sources(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(dir).unwrap_or_else(|err| panic!("read {}: {err}", dir.display())) {
        let entry = entry.expect("read directory entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rust_sources(&path, out);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}
