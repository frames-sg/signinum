# JPEG 2000 Metal Encode Stages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add encode-stage boundaries for JPEG 2000 lossless acceleration and keep CPU output as the validation baseline.

**Architecture:** `signinum-j2k-native` will expose hidden stage hooks for forward RCT, forward 5/3 DWT, Tier-1 encode, and packetization. The existing CPU encoder will call these hooks and fall back per-stage when unsupported. `signinum-j2k-metal` will add encode-side modules/kernels incrementally without changing public `RequireDevice` behavior until full parity exists.

**Tech Stack:** Rust, `signinum-j2k-native`, `signinum-j2k`, `signinum-j2k-metal`, Apple Metal, Cargo tests, Clippy.

---

### Task 1: Native Encode Hook Interface

**Files:**
- Modify: `crates/signinum-j2k-native/src/lib.rs`
- Modify: `crates/signinum-j2k-native/src/j2c/encode.rs`
- Test: `crates/signinum-j2k-native/src/j2c/encode.rs`

- [ ] Add hidden job/result types for forward RCT, forward 5/3 DWT, Tier-1 code-block encode, and packetization.
- [ ] Add `J2kEncodeStageAccelerator` with default methods returning CPU fallback.
- [ ] Add `encode_with_accelerator(...)` beside the existing `encode(...)`.
- [ ] Write tests with a counting accelerator proving each stage hook is called for a small RGB lossless encode.
- [ ] Verify `cargo test -p signinum-j2k-native j2c::encode`.

### Task 2: Public Facade Keeps Strict Device Semantics

**Files:**
- Modify: `crates/signinum-j2k/src/encode.rs`
- Test: `crates/signinum-j2k/tests/encode_lossless.rs`

- [ ] Keep `Auto`, `CpuOnly`, and `PreferDevice` on the self-validating CPU path until a complete device encoder is available.
- [ ] Keep `RequireDevice` as a clear unsupported error.
- [ ] Add a test documenting that `PreferDevice` round-trips and reports CPU while device encode is incomplete.
- [ ] Verify `cargo test -p signinum-j2k --test encode_lossless`.

### Task 3: Metal Encode Stage Skeleton

**Files:**
- Modify: `crates/signinum-j2k-metal/src/lib.rs`
- Create: `crates/signinum-j2k-metal/src/encode.rs`
- Test: `crates/signinum-j2k-metal/src/encode.rs`

- [ ] Add a `MetalEncodeStageAccelerator` type with counters for stage attempts and dispatches.
- [ ] Implement `J2kEncodeStageAccelerator` for it, initially returning fallback for unsupported stages.
- [ ] Add tests proving the type can be used by `encode_with_accelerator` without changing CPU codestream validity.
- [ ] Verify `cargo test -p signinum-j2k-metal encode`.

### Task 4: Metal Forward RCT Kernel

**Files:**
- Modify: `crates/signinum-j2k-metal/src/compute.rs`
- Modify: `crates/signinum-j2k-metal/src/mct.metal`
- Modify: `crates/signinum-j2k-metal/src/encode.rs`

- [ ] Add a forward RCT Metal kernel operating on three f32 component planes.
- [ ] Add a macOS-only wrapper that uploads planes, dispatches the kernel, downloads planes, and compares with native CPU RCT.
- [ ] Enable only the forward RCT hook when the shape is RGB and non-empty.
- [ ] Verify `cargo test -p signinum-j2k-metal forward_rct`.

### Task 5: Verification

**Files:**
- Modified files from Tasks 1-4.

- [ ] Run `cargo fmt --all -- --check`.
- [ ] Run `cargo test -p signinum-j2k-native`.
- [ ] Run `cargo test -p signinum-j2k`.
- [ ] Run `cargo test -p signinum-j2k-metal`.
- [ ] Run `cargo clippy -p signinum-j2k-native --all-targets -- -D warnings`.
- [ ] Run `cargo clippy -p signinum-j2k --all-targets -- -D warnings`.
- [ ] Run `cargo clippy -p signinum-j2k-metal --all-targets -- -D warnings`.
