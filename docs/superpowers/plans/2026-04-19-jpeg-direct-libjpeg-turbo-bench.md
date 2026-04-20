# JPEG Direct libjpeg-turbo Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a direct `libjpeg-turbo` comparator to the `slidecodec-jpeg` benchmark harness so full-frame, ROI, scaled, and tile-batch JPEG decode workloads can be measured against the local native decoder instead of treating `libjpeg-turbo` as a parity-only oracle.

**Architecture:** Keep the change bench-local. Add a handwritten FFI helper for TurboJPEG under `crates/slidecodec-jpeg/benches/common/`, detect/link the local `libjpeg-turbo` install from `build.rs`, and wire the reusable-handle comparator into the existing Criterion groups beside `slidecodec`, `jpeg-decoder`, and `zune-jpeg`. Use TurboJPEG for header decode, full-frame decode, scaled decode, and ROI-with-trim benchmarking so the comparator stays inside the library's fast path without touching production decoder code.

**Tech Stack:** Rust benches/tests, handwritten `extern "C"` FFI, local `pkg-config`, installed `libjpeg-turbo` 3.x, existing `criterion` compare bench, existing JPEG fixtures.

---

### Task 1: Add a failing direct-comparator regression test

**Files:**
- Create: `crates/slidecodec-jpeg/tests/libjpeg_turbo_compare.rs`
- Test: `cargo test -p slidecodec-jpeg --test libjpeg_turbo_compare`

- [ ] **Step 1: Add the parity-oriented integration test**

```rust
#[path = "../benches/common/libjpeg_turbo.rs"]
mod libjpeg_turbo;

use slidecodec_jpeg::{Decoder, Downscale, PixelFormat, Rect};

#[test]
fn turbojpeg_rgb_scaled_and_region_match_slidecodec_fixture() {
    if !libjpeg_turbo::is_available() {
        return;
    }

    let bytes = include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");
    let dec = Decoder::new(bytes).expect("slidecodec decoder");

    let (rgb, _) = dec.decode(PixelFormat::Rgb8).expect("slidecodec rgb");
    let turbo_rgb = libjpeg_turbo::decode_rgb(bytes).expect("turbojpeg rgb");
    assert_eq!(turbo_rgb, rgb);

    let (scaled, _) = dec
        .decode_scaled(PixelFormat::Rgb8, Downscale::Quarter)
        .expect("slidecodec scaled");
    let turbo_scaled =
        libjpeg_turbo::decode_scaled_rgb(bytes, Downscale::Quarter).expect("turbojpeg scaled");
    assert_eq!(turbo_scaled, scaled);

    let roi = Rect { x: 4, y: 4, w: 8, h: 8 };
    let (region, _) = dec
        .decode_region(PixelFormat::Rgb8, roi)
        .expect("slidecodec region");
    let turbo_region = libjpeg_turbo::decode_region_rgb(bytes, roi).expect("turbojpeg region");
    assert_eq!(turbo_region, region);
}
```

- [ ] **Step 2: Run the new test and confirm it fails before implementation**

Run: `cargo test -p slidecodec-jpeg --test libjpeg_turbo_compare`  
Expected: FAIL because `../benches/common/libjpeg_turbo.rs` and its symbols do not exist yet.

### Task 2: Add bench-local `libjpeg-turbo` bindings and linking

**Files:**
- Create: `crates/slidecodec-jpeg/build.rs`
- Create: `crates/slidecodec-jpeg/benches/common/libjpeg_turbo.rs`
- Modify: `crates/slidecodec-jpeg/Cargo.toml`
- Test: `cargo test -p slidecodec-jpeg --test libjpeg_turbo_compare`

- [ ] **Step 1: Add the build script hook**

```toml
[package]
build = "build.rs"
```

- [ ] **Step 2: Detect and link the local TurboJPEG install in `build.rs`**

```rust
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_libjpeg_turbo)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");

    let Ok(output) = Command::new("pkg-config")
        .args(["--libs", "libturbojpeg", "libjpeg"])
        .output()
    else {
        return;
    };
    if !output.status.success() {
        return;
    }

    println!("cargo:rustc-cfg=has_libjpeg_turbo");
    let flags = String::from_utf8_lossy(&output.stdout);
    for token in flags.split_whitespace() {
        if let Some(path) = token.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={path}");
        } else if let Some(lib) = token.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
}
```

- [ ] **Step 3: Add the FFI helper**

```rust
pub(crate) fn is_available() -> bool { cfg!(has_libjpeg_turbo) }

pub(crate) fn decode_rgb(bytes: &[u8]) -> Result<Vec<u8>, String> { /* ... */ }
pub(crate) fn decode_gray(bytes: &[u8]) -> Result<Vec<u8>, String> { /* ... */ }
pub(crate) fn decode_scaled_rgb(bytes: &[u8], factor: Downscale) -> Result<Vec<u8>, String> { /* ... */ }
pub(crate) fn decode_region_rgb(bytes: &[u8], roi: Rect) -> Result<Vec<u8>, String> { /* ... */ }
pub(crate) fn inspect(bytes: &[u8]) -> Result<(u32, u32), String> { /* ... */ }
```

Implementation requirements:
- reuse a `tjhandle` across repeated calls when the bench constructs a decoder once
- call `tj3DecompressHeader()` before scaled/ROI decode
- map `Downscale::{None,Half,Quarter,Eighth}` onto `tjscalingfactor`
- implement ROI by decoding an aligned crop region and trimming the excess columns in Rust when the requested `x` offset is not iMCU-aligned
- return explicit `Err(String)` with `tj3GetErrorStr()` output on failure
- compile to a no-op unavailable state when `pkg-config` cannot find `libjpeg-turbo`

- [ ] **Step 4: Re-run the direct comparator test**

Run: `cargo test -p slidecodec-jpeg --test libjpeg_turbo_compare`  
Expected: PASS on machines with `libjpeg-turbo`; SKIP-by-return on machines without it.

### Task 3: Wire the comparator into the Criterion bench

**Files:**
- Modify: `crates/slidecodec-jpeg/benches/common/mod.rs`
- Modify: `crates/slidecodec-jpeg/benches/compare.rs`
- Test: `cargo bench -p slidecodec-jpeg --bench compare --no-run`

- [ ] **Step 1: Expose reusable helper entry points in `benches/common/mod.rs`**

Add helper functions with the existing naming pattern:

```rust
pub(crate) fn libjpeg_turbo_inspect(bytes: &[u8]) { /* ... */ }
pub(crate) fn libjpeg_turbo_decode(bytes: &[u8], mode: DecodeMode) { /* ... */ }
pub(crate) fn libjpeg_turbo_decode_region(bytes: &[u8], side: u32) { /* ... */ }
pub(crate) fn libjpeg_turbo_decode_scaled(bytes: &[u8], factor: Downscale) { /* ... */ }
pub(crate) fn libjpeg_turbo_decode_region_scaled(bytes: &[u8], side: u32, factor: Downscale) { /* ... */ }
pub(crate) fn libjpeg_turbo_decode_batch_scaled(bytes: &[u8], batch: usize, factor: Downscale) { /* ... */ }
pub(crate) fn libjpeg_turbo_decode_batch_region_scaled(
    bytes: &[u8],
    batch: usize,
    side: u32,
    factor: Downscale,
) { /* ... */ }
```

Rules:
- RGB groups use TurboJPEG RGB decode
- grayscale groups use TurboJPEG gray decode
- tile-batch helpers reuse one TurboJPEG handle across the loop
- `decode_rows_rgb` remains slidecodec-only unless a true scanline comparator is added

- [ ] **Step 2: Add `libjpeg-turbo/...` rows to the existing compare groups**

Update:
- `inspect`
- `decode_rgb`
- `decode_gray`
- `wsi_region_rgb`
- `wsi_scaled_rgb_q4`
- `wsi_scaled_rgb_q8`
- `wsi_region_scaled_rgb_q4`
- `wsi_region_scaled_rgb_q8`
- `wsi_tile_batch_scaled_rgb_q4`
- `wsi_tile_batch_region_scaled_rgb_q4`

Pattern:

```rust
group.bench_function(format!("libjpeg-turbo/{}", input.name), |b| {
    b.iter(|| libjpeg_turbo_decode_scaled(&input.bytes, Downscale::Quarter));
});
```

- [ ] **Step 3: Build the compare bench**

Run: `cargo bench -p slidecodec-jpeg --bench compare --no-run`  
Expected: PASS

### Task 4: Update docs and verify the new benchmark surface

**Files:**
- Modify: `docs/bench.md`
- Modify: `README.md`
- Test: `cargo test -p slidecodec-jpeg --test libjpeg_turbo_compare`
- Test: `cargo clippy -p slidecodec-jpeg --all-targets -- -D warnings`
- Test: `cargo bench -p slidecodec-jpeg --bench compare --no-run`

- [ ] **Step 1: Update `docs/bench.md` comparator policy**

Document:
- `libjpeg-turbo` is now a direct speed comparator for JPEG benches
- TurboJPEG API is used for full-frame/scaled/tile-batch
- ROI decode uses TurboJPEG cropped decode with post-trim for non-iMCU-aligned left edges
- `decode_rows_rgb` remains slidecodec-only

- [ ] **Step 2: Update the top-level README benchmark description**

Replace the current vague “libjpeg-turbo-oriented workflows” wording with explicit direct benchmarking language.

- [ ] **Step 3: Run the narrow verification set**

Run:

```bash
cargo test -p slidecodec-jpeg --test libjpeg_turbo_compare
cargo clippy -p slidecodec-jpeg --all-targets -- -D warnings
cargo bench -p slidecodec-jpeg --bench compare --no-run
```

Expected:
- test passes
- clippy clean
- compare bench compiles with the direct comparator enabled on this machine

- [ ] **Step 4: Commit**

```bash
git add docs/bench.md README.md crates/slidecodec-jpeg/Cargo.toml crates/slidecodec-jpeg/build.rs \
  crates/slidecodec-jpeg/benches/common/mod.rs crates/slidecodec-jpeg/benches/common/libjpeg_turbo.rs \
  crates/slidecodec-jpeg/benches/compare.rs crates/slidecodec-jpeg/tests/libjpeg_turbo_compare.rs \
  docs/superpowers/plans/2026-04-19-jpeg-direct-libjpeg-turbo-bench.md
git commit -m "bench: add direct libjpeg-turbo JPEG comparator"
```
