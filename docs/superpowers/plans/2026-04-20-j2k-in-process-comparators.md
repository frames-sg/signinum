# J2K In-Process Comparator Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current CLI-based J2K benchmark baselines with in-process OpenJPEG and Grok library comparators, then run them on real SVS JP2K tile corpora with parity and codestream-parameter reporting.

**Architecture:** Keep production `slidecodec-j2k` decode code unchanged. Add dev-only benchmark/test support under `crates/slidecodec-j2k/` for two external-library wrappers: one around `libopenjp2`, one around Grok. Both wrappers must decode from memory buffers, force identical interleaved `Rgb8`/`Gray8` output, pin single-threaded execution, and expose the same full-frame / region / q4-scale / tile-batch operations that the existing `slidecodec` bench uses.

**Tech Stack:** Rust, handwritten FFI for `libopenjp2`, handwritten FFI or thin C shim for Grok, Criterion, system `tiffdump` for real SVS tile extraction, existing `slidecodec-j2k` test/bench harness.

---

### Task 1: Add the failing real-comparator tests

**Files:**
- Create: `crates/slidecodec-j2k/tests/common/in_process.rs`
- Create: `crates/slidecodec-j2k/tests/in_process_parity.rs`
- Modify: `crates/slidecodec-j2k/Cargo.toml`

- [ ] **Step 1: Write the failing shared test support**

```rust
// crates/slidecodec-j2k/tests/common/in_process.rs
pub struct ComparatorAvailability {
    pub openjpeg: bool,
    pub grok: bool,
}

pub fn comparator_availability() -> ComparatorAvailability {
    ComparatorAvailability {
        openjpeg: slidecodec_j2k::bench_support::openjpeg::is_available(),
        grok: slidecodec_j2k::bench_support::grok::is_available(),
    }
}
```

- [ ] **Step 2: Write failing parity tests against in-process comparators**

```rust
// crates/slidecodec-j2k/tests/in_process_parity.rs
#[test]
fn openjpeg_in_process_matches_slidecodec_rgb_fixture() {
    let input = bench_fixture_rgb();
    let ours = slidecodec_rgb(&input.bytes);
    let theirs = slidecodec_j2k::bench_support::openjpeg::decode_rgb(&input.bytes).unwrap();
    assert_eq!(ours, theirs);
}

#[test]
fn grok_in_process_matches_slidecodec_rgb_fixture() {
    let input = bench_fixture_rgb();
    let ours = slidecodec_rgb(&input.bytes);
    let theirs = slidecodec_j2k::bench_support::grok::decode_rgb(&input.bytes).unwrap();
    assert_eq!(ours, theirs);
}
```

- [ ] **Step 3: Run the tests to verify they fail for missing symbols/modules**

Run: `cargo test -p slidecodec-j2k --test in_process_parity -- --nocapture`

Expected: compile failure or test failure because `bench_support::{openjpeg,grok}` does not exist yet.

- [ ] **Step 4: Add the dev-only module hook**

```toml
# crates/slidecodec-j2k/Cargo.toml
[features]
default = []
bench-ffi = []
```

- [ ] **Step 5: Commit**

```bash
git add crates/slidecodec-j2k/Cargo.toml crates/slidecodec-j2k/tests/common/in_process.rs crates/slidecodec-j2k/tests/in_process_parity.rs
git commit -m "test: add failing in-process j2k comparator parity tests"
```

### Task 2: Implement in-process OpenJPEG bench support

**Files:**
- Create: `crates/slidecodec-j2k/src/bench_support/mod.rs`
- Create: `crates/slidecodec-j2k/src/bench_support/openjpeg.rs`
- Modify: `crates/slidecodec-j2k/src/lib.rs`
- Modify: `crates/slidecodec-j2k/Cargo.toml`
- Create: `crates/slidecodec-j2k/build.rs`

- [ ] **Step 1: Write the failing OpenJPEG-specific test**

```rust
#[test]
fn openjpeg_region_decode_matches_slidecodec_fixture() {
    let input = bench_fixture_rgb();
    let roi = Rect { x: 32, y: 32, w: 96, h: 96 };
    let ours = slidecodec_rgb_region(&input.bytes, roi);
    let theirs = slidecodec_j2k::bench_support::openjpeg::decode_rgb_region(&input.bytes, roi).unwrap();
    assert_eq!(ours, theirs);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p slidecodec-j2k --test in_process_parity openjpeg_region_decode_matches_slidecodec_fixture -- --nocapture`

Expected: unresolved OpenJPEG functions.

- [ ] **Step 3: Add build-time OpenJPEG detection**

```rust
// crates/slidecodec-j2k/build.rs
fn main() {
    if let Ok(lib) = pkg_config::Config::new().probe("libopenjp2") {
        println!("cargo:rustc-cfg=have_openjpeg");
        for path in lib.include_paths {
            println!("cargo:include={}", path.display());
        }
    }
}
```

- [ ] **Step 4: Implement the minimal OpenJPEG FFI wrapper**

```rust
// crates/slidecodec-j2k/src/bench_support/openjpeg.rs
pub fn decode_rgb(bytes: &[u8]) -> Result<Vec<u8>, String> { /* create memory stream, set default params, set threads=1, decode, interleave */ }
pub fn decode_rgb_region(bytes: &[u8], roi: Rect) -> Result<Vec<u8>, String> { /* opj_set_decode_area() before decode */ }
pub fn decode_rgb_scaled(bytes: &[u8], reduce: u32) -> Result<Vec<u8>, String> { /* opj_set_decoded_resolution_factor() before read_header/decode */ }
pub fn decode_rgb_tile_batch(bytes: &[u8], count: usize) -> Result<(), String> { /* loop decode_rgb with reused wrapper state where possible */ }
```

- [ ] **Step 5: Re-run the OpenJPEG parity tests**

Run: `cargo test -p slidecodec-j2k --test in_process_parity openjpeg -- --nocapture`

Expected: PASS for full-frame and region parity tests.

- [ ] **Step 6: Commit**

```bash
git add crates/slidecodec-j2k/build.rs crates/slidecodec-j2k/src/lib.rs crates/slidecodec-j2k/src/bench_support/mod.rs crates/slidecodec-j2k/src/bench_support/openjpeg.rs crates/slidecodec-j2k/Cargo.toml crates/slidecodec-j2k/tests/in_process_parity.rs
git commit -m "bench: add in-process openjpeg j2k comparator"
```

### Task 3: Implement in-process Grok bench support

**Files:**
- Create: `crates/slidecodec-j2k/src/bench_support/grok.rs`
- Modify: `crates/slidecodec-j2k/build.rs`
- Modify: `crates/slidecodec-j2k/src/bench_support/mod.rs`
- Modify: `crates/slidecodec-j2k/tests/in_process_parity.rs`

- [ ] **Step 1: Write the failing Grok-specific tests**

```rust
#[test]
fn grok_in_process_region_matches_slidecodec_fixture() {
    let input = bench_fixture_rgb();
    let roi = Rect { x: 32, y: 32, w: 96, h: 96 };
    let ours = slidecodec_rgb_region(&input.bytes, roi);
    let theirs = slidecodec_j2k::bench_support::grok::decode_rgb_region(&input.bytes, roi).unwrap();
    assert_eq!(ours, theirs);
}

#[test]
fn grok_in_process_scaled_matches_slidecodec_fixture_shape() {
    let input = bench_fixture_rgb();
    let theirs = slidecodec_j2k::bench_support::grok::decode_rgb_scaled(&input.bytes, 2).unwrap();
    assert_eq!(theirs.len(), 64 * 64 * 3);
}
```

- [ ] **Step 2: Run the Grok tests to verify they fail**

Run: `cargo test -p slidecodec-j2k --test in_process_parity grok -- --nocapture`

Expected: unresolved Grok module/functions.

- [ ] **Step 3: Extend build-time Grok detection**

```rust
// crates/slidecodec-j2k/build.rs
if let Some(root) = detect_grok_root() {
    println!("cargo:rustc-link-search=native={}", root.display());
    println!("cargo:rustc-link-lib=dylib=grokj2k");
    println!("cargo:rustc-cfg=have_grok");
}
```

- [ ] **Step 4: Implement the minimal Grok FFI wrapper**

```rust
// crates/slidecodec-j2k/src/bench_support/grok.rs
pub fn decode_rgb(bytes: &[u8]) -> Result<Vec<u8>, String> { /* grk_initialize(thread=1), memory stream, read header, force_rgb=true, decompress_fmt=PXM, full decode, copy image.interleaved_data */ }
pub fn decode_rgb_region(bytes: &[u8], roi: Rect) -> Result<Vec<u8>, String> { /* set dw_x0..dw_y1 in params before init/read_header */ }
pub fn decode_rgb_scaled(bytes: &[u8], reduce: u32) -> Result<Vec<u8>, String> { /* params.core.reduce = reduce */ }
pub fn decode_rgb_tile_batch(bytes: &[u8], count: usize) -> Result<(), String> { /* loop decode_rgb with the same parameter policy */ }
```

- [ ] **Step 5: Re-run the Grok parity tests**

Run: `SLIDECODEC_GROK_ROOT=/tmp/grok-slidecodec/build/bin cargo test -p slidecodec-j2k --test in_process_parity grok -- --nocapture`

Expected: PASS for full-frame and region parity tests.

- [ ] **Step 6: Commit**

```bash
git add crates/slidecodec-j2k/build.rs crates/slidecodec-j2k/src/bench_support/mod.rs crates/slidecodec-j2k/src/bench_support/grok.rs crates/slidecodec-j2k/tests/in_process_parity.rs
git commit -m "bench: add in-process grok j2k comparator"
```

### Task 4: Replace CLI comparators in the Criterion harness

**Files:**
- Modify: `crates/slidecodec-j2k/benches/common/mod.rs`
- Modify: `crates/slidecodec-j2k/benches/compare.rs`

- [ ] **Step 1: Write the failing benchmark compile expectation**

```rust
// benches/common/mod.rs
// Replace openjpeg_decode()/grok_decode() process wrappers with in-process wrapper calls.
```

- [ ] **Step 2: Run the bench compile to verify current shape still depends on CLI wrappers**

Run: `cargo bench -p slidecodec-j2k --bench compare --no-run`

Expected: still compiles against CLI wrappers before refactor.

- [ ] **Step 3: Switch benchmark operations to in-process wrappers**

```rust
pub(crate) fn openjpeg_decode(input: &BenchInput, reduce: Option<u32>, region: Option<Rect>, batch: usize) {
    for _ in 0..batch {
        match (reduce, region, input.mode) { /* call openjpeg::decode_rgb / region / scaled */ }
    }
}
```

- [ ] **Step 4: Pin output policy in the benchmark docs and code**

```rust
// all comparator wrappers must return interleaved RGB8 or Gray8 bytes
// all wrappers must set single-thread mode explicitly
```

- [ ] **Step 5: Re-run quick benchmark smoke**

Run: `SLIDECODEC_GROK_ROOT=/tmp/grok-slidecodec/build/bin cargo bench -p slidecodec-j2k --bench compare -- --quick`

Expected: benchmark groups run with in-process OpenJPEG and Grok rows.

- [ ] **Step 6: Commit**

```bash
git add crates/slidecodec-j2k/benches/common/mod.rs crates/slidecodec-j2k/benches/compare.rs
git commit -m "bench: switch j2k comparators to in-process libraries"
```

### Task 5: Add real SVS tile corpus extraction and actual-tile benchmark coverage

**Files:**
- Modify: `crates/slidecodec-j2k/benches/common/mod.rs`
- Create: `crates/slidecodec-j2k/tests/actual_wsi_tiles.rs`
- Modify: `docs/bench.md`

- [ ] **Step 1: Write the failing actual-tile extraction test**

```rust
#[test]
fn extracts_multiple_real_svs_j2k_tiles_from_openslide_testdata() {
    let inputs = actual_svs_j2k_tile_inputs().unwrap();
    assert!(inputs.iter().any(|input| input.name.contains(\"33003\")));
    assert!(inputs.iter().any(|input| input.name.contains(\"33005\")));
    assert!(inputs.len() >= 8);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p slidecodec-j2k --test actual_wsi_tiles -- --nocapture`

Expected: missing extraction helper.

- [ ] **Step 3: Implement TIFF-backed tile extraction for real SVS JP2K tiles**

```rust
pub(crate) fn actual_svs_j2k_tile_inputs() -> Result<Vec<BenchInput>, String> {
    // parse tiffdump output for TileOffsets/TileByteCounts on J2K-encoded directories,
    // extract a bounded sample across interior and edge tiles, not only tile 0
}
```

- [ ] **Step 4: Add actual-tile benchmark groups or inputs**

```rust
// include actual_svs_j2k_tile_inputs() when SLIDECODEC_BENCH_INPUTS points at SlideViewer corpus
```

- [ ] **Step 5: Document codestream parameters and corpus policy**

```markdown
- actual J2K SVS comparisons use extracted Aperio JP2K tiles
- report tile size, components, resolution levels, and tile-count sampling policy
- compare interleaved RGB8 outputs, single-threaded, in-process
```

- [ ] **Step 6: Commit**

```bash
git add crates/slidecodec-j2k/benches/common/mod.rs crates/slidecodec-j2k/tests/actual_wsi_tiles.rs docs/bench.md
git commit -m "bench: add actual svs jp2k tile corpus coverage"
```

### Task 6: Final verification and result capture

**Files:**
- Modify: `docs/bench.md`

- [ ] **Step 1: Run the narrow correctness suite**

Run: `SLIDECODEC_GROK_ROOT=/tmp/grok-slidecodec/build/bin cargo test -p slidecodec-j2k --test in_process_parity --test grok_parity --test openjpeg_parity --test actual_wsi_tiles -- --nocapture`

Expected: PASS

- [ ] **Step 2: Run lint and bench compile**

Run: `cargo clippy -p slidecodec-j2k --tests --benches -- -D warnings`

Expected: PASS

Run: `SLIDECODEC_GROK_ROOT=/tmp/grok-slidecodec/build/bin cargo bench -p slidecodec-j2k --bench compare --no-run`

Expected: PASS

- [ ] **Step 3: Run the actual-tile benchmark and record results**

Run: `SLIDECODEC_GROK_ROOT=/tmp/grok-slidecodec/build/bin cargo bench -p slidecodec-j2k --bench compare -- --quick`

Expected: in-process OpenJPEG and Grok rows for real SVS tile inputs with distributions, not single-shot CLI timings.

- [ ] **Step 4: Update docs with measurement context**

```markdown
- output format: interleaved RGB8 / Gray8
- thread count: 1 for slidecodec, OpenJPEG, and Grok
- warmup and Criterion distribution policy
- actual-slide sources and tile-sampling policy
```

- [ ] **Step 5: Commit**

```bash
git add docs/bench.md
git commit -m "docs: record in-process j2k comparator methodology"
```
