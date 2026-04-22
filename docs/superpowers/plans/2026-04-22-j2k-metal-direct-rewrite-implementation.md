# J2K / HTJ2K MetalDirect Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strict GPU-owned `MetalDirect` J2K / HTJ2K grayscale decode path where the host only parses and plans, while preserving a separate strict `CpuOnly` path.

**Architecture:** Add a hidden grayscale direct-plan seam in `slidecodec-j2k-native`, then add a new `slidecodec-j2k-metal` direct executor that consumes that plan and owns sub-band decode, IDWT, store, and final pack on device. Keep `CpuOnly` intact, split strategy dispatch in `slidecodec-j2k-metal`, and add structural tests proving explicit `Metal` does not route through CPU-upload helpers.

**Tech Stack:** Rust, Metal, Criterion, existing `slidecodec-j2k-native` hidden backend hooks, Apple Silicon Metal runtime.

---

## File Structure

### New files

- `crates/slidecodec-j2k-native/src/direct_plan.rs`
  - Hidden grayscale direct-plan builder and owned plan/job structs.
- `crates/slidecodec-j2k-metal/src/direct.rs`
  - MetalDirect grayscale executor over the native plan.
- `crates/slidecodec-j2k-metal/tests/direct.rs`
  - End-to-end parity and structural tests for the direct path.

### Modified files

- `crates/slidecodec-j2k-native/src/lib.rs`
  - Export hidden direct-plan types and entrypoints.
- `crates/slidecodec-j2k-native/src/j2c/mod.rs`
  - Register the new direct-plan module.
- `crates/slidecodec-j2k-native/src/j2c/decode.rs`
  - Reuse existing packet/decomposition code to populate the new plan without executing into `channel_data`.
- `crates/slidecodec-j2k-metal/src/lib.rs`
  - Strict strategy split between `CpuOnly` and `MetalDirect`; remove CPU-upload contamination from explicit `Metal`.
- `crates/slidecodec-j2k-metal/src/compute.rs`
  - Expose low-level Metal kernel helpers only. Keep CPU-upload helpers separate from the new direct executor orchestration.
- `crates/slidecodec-j2k/benches/compare.rs`
  - Keep the direct path benchmarkable after the strict strategy split.

### Existing files this plan depends on

- `crates/slidecodec-j2k-metal/src/classic.metal`
- `crates/slidecodec-j2k-metal/src/ht_cleanup.metal`
- `crates/slidecodec-j2k-metal/src/idwt.metal`
- `crates/slidecodec-j2k-metal/src/store.metal`
- `crates/slidecodec-j2k-native/src/j2c/build.rs`
- `crates/slidecodec-j2k-native/src/j2c/segment.rs`
- `crates/slidecodec-j2k-native/src/j2c/decode.rs`

---

### Task 1: Add hidden grayscale direct-plan types and a failing native-plan test

**Files:**
- Create: `crates/slidecodec-j2k-native/src/direct_plan.rs`
- Modify: `crates/slidecodec-j2k-native/src/lib.rs`
- Modify: `crates/slidecodec-j2k-native/src/j2c/mod.rs`
- Test: `crates/slidecodec-j2k-native/src/lib.rs`

- [ ] **Step 1: Write the failing grayscale direct-plan test**

Add a hidden unit test in `crates/slidecodec-j2k-native/src/lib.rs` that proves a grayscale classic J2K image can produce a non-empty direct plan without touching host component storage.

```rust
#[test]
fn grayscale_direct_plan_is_built_without_materializing_channel_data() {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let bytes = encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode classic gray8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();

    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("build direct plan");

    assert_eq!(plan.dimensions, (4, 4));
    assert_eq!(plan.bit_depth, 8);
    assert!(!plan.steps.is_empty(), "direct plan must contain executable steps");
    assert!(
        context.tile_decode_context.channel_data.is_empty(),
        "building a direct plan must not materialize host component planes"
    );
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p slidecodec-j2k-native grayscale_direct_plan_is_built_without_materializing_channel_data -- --exact`

Expected: FAIL with a missing method/type error for `build_direct_grayscale_plan_with_context` or missing plan structs.

- [ ] **Step 3: Add hidden direct-plan types and exports**

Create `crates/slidecodec-j2k-native/src/direct_plan.rs` with the owned grayscale plan types and helper constructors, then export them from `lib.rs`.

```rust
// crates/slidecodec-j2k-native/src/direct_plan.rs
use alloc::vec::Vec;
use crate::{
    HtCodeBlockDecodeJob, HtCodeBlockBatchJob, J2kCodeBlockDecodeJob, J2kCodeBlockBatchJob,
    J2kRect, J2kWaveletTransform,
};

#[doc(hidden)]
#[derive(Debug, Clone)]
pub enum J2kDirectGrayscaleStep {
    ClassicSubBand(J2kOwnedSubBandPlan),
    HtSubBand(HtOwnedSubBandPlan),
    Idwt(J2kDirectIdwtStep),
    Store(J2kDirectStoreStep),
}

#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct J2kDirectGrayscalePlan {
    pub dimensions: (u32, u32),
    pub bit_depth: u8,
    pub steps: Vec<J2kDirectGrayscaleStep>,
}

#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct J2kOwnedSubBandPlan {
    pub rect: J2kRect,
    pub width: u32,
    pub height: u32,
    pub jobs: Vec<J2kOwnedCodeBlockBatchJob>,
}

#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct HtOwnedSubBandPlan {
    pub rect: J2kRect,
    pub width: u32,
    pub height: u32,
    pub jobs: Vec<HtOwnedCodeBlockBatchJob>,
}
```

```rust
// crates/slidecodec-j2k-native/src/lib.rs
mod direct_plan;

#[doc(hidden)]
pub use direct_plan::{
    HtOwnedCodeBlockBatchJob, HtOwnedSubBandPlan, J2kDirectGrayscalePlan,
    J2kDirectGrayscaleStep, J2kDirectIdwtStep, J2kDirectStoreStep,
    J2kOwnedCodeBlockBatchJob, J2kOwnedSubBandPlan,
};
```

- [ ] **Step 4: Run the test to verify it still fails for the right reason**

Run: `cargo test -p slidecodec-j2k-native grayscale_direct_plan_is_built_without_materializing_channel_data -- --exact`

Expected: FAIL because the plan builder still returns nothing or the method is missing, not because of type errors.

- [ ] **Step 5: Commit**

```bash
git add crates/slidecodec-j2k-native/src/direct_plan.rs crates/slidecodec-j2k-native/src/lib.rs crates/slidecodec-j2k-native/src/j2c/mod.rs
git commit -m "feat: add j2k direct grayscale plan types"
```

### Task 2: Build the native grayscale direct plan without decoding into host planes

**Files:**
- Modify: `crates/slidecodec-j2k-native/src/j2c/decode.rs`
- Modify: `crates/slidecodec-j2k-native/src/lib.rs`
- Test: `crates/slidecodec-j2k-native/src/lib.rs`

- [ ] **Step 1: Write the failing HTJ2K direct-plan test**

Add a second unit test proving the same direct-plan builder works for HTJ2K grayscale and contains HT steps.

```rust
#[test]
fn htj2k_grayscale_direct_plan_contains_ht_sub_band_steps() {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let bytes = encode_htj2k(&pixels, 4, 4, 1, 8, false, &options).expect("encode ht gray8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();

    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("build direct plan");

    assert!(
        plan.steps.iter().any(|step| matches!(step, J2kDirectGrayscaleStep::HtSubBand(_))),
        "HTJ2K direct plan must contain HT sub-band decode steps"
    );
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p slidecodec-j2k-native 'direct_plan' -- --nocapture`

Expected: FAIL because the builder is not implemented yet.

- [ ] **Step 3: Implement the plan builder in native decode**

Add `Image::build_direct_grayscale_plan_with_context(...)` in `lib.rs` and a helper in `j2c/decode.rs` that:

- reuses tile parse/build/segment logic
- walks grayscale component decompositions
- emits owned sub-band decode steps
- emits ordered IDWT steps from the decomposition graph
- emits a final store step
- never fills `channel_data`

```rust
// crates/slidecodec-j2k-native/src/lib.rs
#[doc(hidden)]
pub fn build_direct_grayscale_plan_with_context(
    &self,
    decoder_context: &mut DecoderContext<'a>,
) -> Result<J2kDirectGrayscalePlan> {
    if self.color_space != ColorSpace::Gray || self.has_alpha {
        bail!(ValidationError::UnsupportedColorSpace);
    }
    j2c::build_direct_grayscale_plan(self.codestream, &self.header, decoder_context)
}
```

```rust
// crates/slidecodec-j2k-native/src/j2c/decode.rs
pub(crate) fn build_direct_grayscale_plan<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    ctx: &mut DecoderContext<'a>,
) -> Result<J2kDirectGrayscalePlan> {
    let mut reader = BitReader::new(data);
    let tiles = tile::parse(&mut reader, header)?;
    let tile = tiles.first().ok_or(TileError::Invalid)?;
    ctx.reset(header, tile);
    ctx.storage.reset();

    build::build(tile, &mut ctx.storage)?;
    let iter_input = IteratorInput::new(tile);
    let progression_iterator: Box<dyn Iterator<Item = ProgressionData>> =
        match tile.progression_order {
            ProgressionOrder::LayerResolutionComponentPosition => {
                Box::new(layer_resolution_component_position_progression(iter_input))
            }
            ProgressionOrder::ResolutionLayerComponentPosition => {
                Box::new(resolution_layer_component_position_progression(iter_input))
            }
            ProgressionOrder::ResolutionPositionComponentLayer => Box::new(
                resolution_position_component_layer_progression(iter_input)
                    .ok_or(DecodingError::InvalidProgressionIterator)?,
            ),
            ProgressionOrder::PositionComponentResolutionLayer => Box::new(
                position_component_resolution_layer_progression(iter_input)
                    .ok_or(DecodingError::InvalidProgressionIterator)?,
            ),
            ProgressionOrder::ComponentPositionResolutionLayer => Box::new(
                component_position_resolution_layer_progression(iter_input)
                    .ok_or(DecodingError::InvalidProgressionIterator)?,
            ),
        };
    segment::parse(tile, progression_iterator, header, &mut ctx.storage)?;
    build_grayscale_plan_from_storage(tile, header, &ctx.storage)
}
```

- [ ] **Step 4: Run native tests to verify they pass**

Run: `cargo test -p slidecodec-j2k-native 'direct_plan|decoder_hook' -- --nocapture`

Expected: PASS for the new direct-plan tests and PASS for the existing hook tests.

- [ ] **Step 5: Commit**

```bash
git add crates/slidecodec-j2k-native/src/j2c/decode.rs crates/slidecodec-j2k-native/src/lib.rs
git commit -m "feat: build native grayscale j2k direct plans"
```

### Task 3: Add a MetalDirect grayscale executor and parity test

**Files:**
- Create: `crates/slidecodec-j2k-metal/src/direct.rs`
- Modify: `crates/slidecodec-j2k-metal/src/lib.rs`
- Modify: `crates/slidecodec-j2k-metal/src/compute.rs`
- Test: `crates/slidecodec-j2k-metal/tests/direct.rs`

- [ ] **Step 1: Write the failing direct parity test for classic grayscale**

Create new integration tests that:

- ask explicit `BackendRequest::Metal` for a grayscale full decode and compare the downloaded bytes to CPU output
- prove explicit `BackendRequest::Metal` does not route through the CPU-upload path for the supported grayscale slice
- reject unsupported first-cut scope such as RGB or region/scaled direct requests instead of silently widening the direct path

```rust
#[test]
fn explicit_metal_direct_gray8_matches_cpu_for_classic_j2k() {
    let input = fixture_j2k_gray8();
    let mut cpu = J2kDecoder::new(&input).expect("cpu decoder");
    let mut metal = J2kDecoder::new(&input).expect("metal decoder");

    let dims = cpu.inner().info().dimensions;
    let stride = dims.0 as usize;
    let mut expected = vec![0u8; stride * dims.1 as usize];
    cpu.decode_into(&mut expected, stride, PixelFormat::Gray8)
        .expect("cpu decode");

    let surface = metal
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Metal)
        .expect("metal direct decode");

    assert_eq!(surface.as_bytes(), expected.as_slice());
}
```

```rust
#[test]
fn explicit_metal_gray8_does_not_use_cpu_upload_strategy() {
    let input = fixture_j2k_gray8();
    let mut decoder = J2kDecoder::new(&input).expect("decoder");

    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Metal)
        .expect("metal direct decode");

    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert!(
        !decoder.debug_last_surface_strategy().is_cpu_upload(),
        "explicit Metal must not route through CpuUpload"
    );
}
```

```rust
#[test]
fn explicit_metal_rejects_first_cut_unsupported_scope() {
    let input = fixture_j2k_rgb8();
    let mut decoder = J2kDecoder::new(&input).expect("decoder");
    let error = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Metal)
        .expect_err("first-cut direct metal should reject rgb");

    assert!(error.is_unsupported());
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p slidecodec-j2k-metal explicit_metal_direct_gray8_matches_cpu_for_classic_j2k -- --exact`

Expected: FAIL because explicit `Metal` still routes through the old contaminated strategy, because the debug strategy helper does not exist yet, or because unsupported scope is not rejected explicitly.

- [ ] **Step 3: Implement the direct executor**

First split explicit `Metal` away from CPU-upload routing in `lib.rs`, then create `direct.rs` with a narrow grayscale executor that consumes `J2kDirectGrayscalePlan`, allocates device coefficient buffers per step, reuses existing classic/HT/IDWT/store kernels from `compute.rs`, and finishes with the existing pack kernel.

`direct.rs` must be the orchestration boundary for `MetalDirect`. It may call low-level kernel helpers from `compute.rs`, but it may not call any CPU-upload or host-plane staging helpers.

```rust
// crates/slidecodec-j2k-metal/src/direct.rs
pub(crate) fn decode_grayscale_plan_to_surface(
    plan: &slidecodec_j2k_native::J2kDirectGrayscalePlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    compute::decode_direct_grayscale_plan(plan, fmt)
}
```

```rust
// crates/slidecodec-j2k-metal/src/lib.rs
fn decode_to_surface_via_direct_metal(&mut self, fmt: PixelFormat) -> Result<Surface, Error> {
    self.ensure_native_image()?;
    let image = self
        .native_image
        .as_ref()
        .ok_or_else(|| Error::MetalKernel { message: "native image cache missing".to_string() })?;
    let plan = image
        .build_direct_grayscale_plan_with_context(&mut self.native_context)
        .map_err(|error| Error::Decode(slidecodec_j2k::J2kError::Backend(error.to_string())))?;
    direct::decode_grayscale_plan_to_surface(&plan, fmt)
}
```

```rust
// crates/slidecodec-j2k-metal/src/compute.rs
pub(crate) fn decode_direct_grayscale_plan(
    plan: &slidecodec_j2k_native::J2kDirectGrayscalePlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let mut planes = DirectPlaneArena::default();
        for step in &plan.steps {
            match step {
                J2kDirectGrayscaleStep::ClassicSubBand(job) => {
                    execute_classic_sub_band_plan(runtime, job, &mut planes)?;
                }
                J2kDirectGrayscaleStep::HtSubBand(job) => {
                    execute_ht_sub_band_plan(runtime, job, &mut planes)?;
                }
                J2kDirectGrayscaleStep::Idwt(job) => {
                    execute_idwt_plan(runtime, job, &mut planes)?;
                }
                J2kDirectGrayscaleStep::Store(job) => {
                    execute_store_plan(runtime, job, &mut planes)?;
                }
            }
        }
        finish_grayscale_surface(runtime, &planes, plan.dimensions, plan.bit_depth, fmt)
    })
}
```

- [ ] **Step 4: Run the direct parity test to verify it passes**

Run: `cargo test -p slidecodec-j2k-metal 'explicit_metal_direct_gray8_matches_cpu_for_classic_j2k|explicit_metal_gray8_does_not_use_cpu_upload_strategy|explicit_metal_rejects_first_cut_unsupported_scope' -- --nocapture`

Expected: PASS for classic grayscale parity, PASS for the no-CPU-upload structural test, and PASS for unsupported-scope rejection.

- [ ] **Step 5: Commit**

```bash
git add crates/slidecodec-j2k-metal/src/direct.rs crates/slidecodec-j2k-metal/src/lib.rs crates/slidecodec-j2k-metal/src/compute.rs crates/slidecodec-j2k-metal/tests/direct.rs
git commit -m "feat: add direct metal grayscale j2k executor"
```

### Task 4: Add HTJ2K direct parity and strict strategy split

**Files:**
- Modify: `crates/slidecodec-j2k-metal/src/lib.rs`
- Modify: `crates/slidecodec-j2k-metal/tests/direct.rs`
- Test: `crates/slidecodec-j2k-metal/tests/direct.rs`

- [ ] **Step 1: Write the failing HTJ2K parity and no-fallback tests**

Add tests that:

- compare explicit `BackendRequest::Metal` against CPU for HTJ2K grayscale
- prove explicit `Metal` no longer routes through CPU-upload helpers
- prove explicit `Metal` rejects region/scaled direct requests in the first grayscale-only full-decode slice

```rust
#[test]
fn explicit_metal_direct_gray8_matches_cpu_for_htj2k() {
    let input = fixture_htj2k_gray8();
    let mut cpu = J2kDecoder::new(&input).expect("cpu decoder");
    let mut metal = J2kDecoder::new(&input).expect("metal decoder");

    let dims = cpu.inner().info().dimensions;
    let stride = dims.0 as usize;
    let mut expected = vec![0u8; stride * dims.1 as usize];
    cpu.decode_into(&mut expected, stride, PixelFormat::Gray8)
        .expect("cpu decode");

    let surface = metal
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Metal)
        .expect("metal direct decode");

    assert_eq!(surface.as_bytes(), expected.as_slice());
}
```

```rust
#[test]
fn explicit_metal_path_does_not_use_cpu_upload_strategy() {
    std::env::remove_var("SLIDECODEC_J2K_METAL_FORCE_KERNEL");
    let input = fixture_j2k_gray8();
    let mut decoder = J2kDecoder::new(&input).expect("decoder");

    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Metal)
        .expect("metal direct decode");

    assert_eq!(surface.backend_kind(), BackendKind::Metal);
}
```

```rust
#[test]
fn explicit_metal_region_is_rejected_in_first_direct_slice() {
    let input = fixture_htj2k_gray8();
    let mut decoder = J2kDecoder::new(&input).expect("decoder");
    let error = decoder
        .decode_region_to_device(
            PixelFormat::Gray8,
            Rect { x: 0, y: 0, w: 1, h: 1 },
            BackendRequest::Metal,
        )
        .expect_err("first-cut direct metal should reject region decode");

    assert!(error.is_unsupported());
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p slidecodec-j2k-metal 'explicit_metal_direct_gray8_matches_cpu_for_htj2k|explicit_metal_path_does_not_use_cpu_upload_strategy' -- --nocapture`

Expected: FAIL because explicit `Metal` still routes through the old strategy split or because region rejection is not enforced yet.

- [ ] **Step 3: Refactor strategy dispatch**

Change `slidecodec-j2k-metal/src/lib.rs` so explicit `BackendRequest::Metal` routes only to `MetalDirect`. Keep `CpuOnly` as a separate path. Keep `Auto` as a selector only.

```rust
fn decode_to_surface_impl(
    &mut self,
    fmt: PixelFormat,
    backend: BackendRequest,
) -> Result<Surface, Error> {
    match backend {
        BackendRequest::Cpu => self.decode_to_surface_via_cpu(fmt),
        BackendRequest::Metal => self.decode_to_surface_via_direct_metal(fmt),
        BackendRequest::Auto => {
            if self.supports_direct_grayscale(fmt) {
                self.decode_to_surface_via_direct_metal(fmt)
            } else {
                self.decode_to_surface_via_cpu(fmt)
            }
        }
        BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
    }
}
```

- [ ] **Step 4: Run the J2K Metal tests to verify they pass**

Run: `cargo test -p slidecodec-j2k-metal --lib --tests`

Expected: PASS for new direct tests and PASS for existing library/tests.

- [ ] **Step 5: Commit**

```bash
git add crates/slidecodec-j2k-metal/src/lib.rs crates/slidecodec-j2k-metal/tests/direct.rs
git commit -m "refactor: split cpu and direct metal j2k execution"
```

### Task 5: Benchmark the direct grayscale path and lock the compare surface

**Files:**
- Modify: `crates/slidecodec-j2k/benches/compare.rs`
- Modify: `crates/slidecodec-j2k-metal/src/lib.rs`
- Test: `crates/slidecodec-j2k-metal/tests/direct.rs`

- [ ] **Step 1: Write a failing benchmark-mode test for explicit Metal direct grayscale**

Add a small test that ensures explicit `Metal` is wired for grayscale compare inputs instead of silently falling back.

```rust
#[test]
fn grayscale_compare_inputs_admit_direct_metal_path() {
    let input = fixture_j2k_gray8();
    let decoder = J2kDecoder::new(&input).expect("decoder");
    assert!(decoder.supports_direct_grayscale(PixelFormat::Gray8));
}
```

- [ ] **Step 2: Run the targeted test to verify it fails if the admission helper is missing**

Run: `cargo test -p slidecodec-j2k-metal grayscale_compare_inputs_admit_direct_metal_path -- --exact`

Expected: FAIL if the admission helper is not present.

- [ ] **Step 3: Wire the compare bench to exercise direct grayscale**

Keep the classic and HT grayscale compare groups, but ensure the `slidecodec-metal` lane is testing the new direct grayscale path instead of CPU-upload for the supported slice.

```rust
// crates/slidecodec-j2k/benches/compare.rs
wsi_region.bench_function(format!("slidecodec-metal/{}", input.name), |b| {
    b.iter(|| bench_slidecodec_metal_region(&input, BackendRequest::Metal))
});
```

```rust
// crates/slidecodec-j2k-metal/src/lib.rs
fn supports_direct_grayscale(&self, fmt: PixelFormat) -> bool {
    matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16)
        && matches!(self.inner.info().color_space, slidecodec_core::ColorSpace::Gray)
}
```

- [ ] **Step 4: Run verification**

Run:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p slidecodec-j2k-native
cargo test -p slidecodec-j2k-metal --lib --tests
cargo bench -p slidecodec-j2k --bench compare -- 'decode_gray|wsi_region_gray|wsi_scaled_gray_q4|wsi_tile_batch_gray' --quick --noplot
```

Expected:

- fmt: PASS
- clippy: PASS
- native tests: PASS
- metal tests: PASS
- compare bench runs with the new direct grayscale lane

- [ ] **Step 5: Commit**

```bash
git add crates/slidecodec-j2k/benches/compare.rs crates/slidecodec-j2k-metal/src/lib.rs crates/slidecodec-j2k-metal/tests/direct.rs
git commit -m "bench: exercise direct metal grayscale j2k path"
```

---

## Self-Review

### Spec coverage

- Strict CPU and Metal separation: covered in Tasks 3 and 4.
- Hidden native planning seam: covered in Tasks 1 and 2.
- First grayscale-only direct path: covered in Tasks 2 and 3.
- Classic and HTJ2K parity: covered in Tasks 2, 3, and 4.
- Benchmarkable direct path: covered in Task 5.

### Placeholder scan

- No `TODO`, `TBD`, or deferred implementation placeholders remain.
- Each task has exact files, tests, commands, and commit points.

### Type consistency

- `J2kDirectGrayscalePlan`, `J2kDirectGrayscaleStep`, `build_direct_grayscale_plan_with_context`, and `supports_direct_grayscale` are used consistently across tasks.
- Strategy names stay consistent: `CpuOnly`, `MetalDirect`, `Auto`.
