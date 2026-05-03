# Metal Fast420 Coefficient/IDCT Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an experimental full-frame fast420 batch path that splits entropy coefficient decode from IDCT/deposit.

**Architecture:** Keep the fused fast420 batch kernel as the default. Add an opt-in split path that dispatches entropy-to-coefficients, then block-level IDCT/deposit, then the existing pack kernel.

**Tech Stack:** Rust, Metal Shading Language, `metal` crate, Criterion, `signinum-jpeg-metal`.

---

### Task 1: Regression Guards

**Files:**
- Modify: `crates/signinum-jpeg-metal/tests/shader_integrity.rs`

- [x] Add guards for `jpeg_decode_fast420_batch_coeffs`, `jpeg_idct_deposit_fast420_batch`, and `SIGNINUM_JPEG_METAL_SPLIT_FAST420_BATCH`.
- [x] Run `cargo test -p signinum-jpeg-metal --test shader_integrity` and verify the new guards fail before implementation.
- [x] Add guards for the existing entropy fast paths: 4-byte refill and 9-bit Huffman lookup.

### Task 2: Split Kernels

**Files:**
- Modify: `crates/signinum-jpeg-metal/src/shaders.metal`

- [x] Add helpers that write decoded coefficient blocks and DC-only flags to scratch.
- [x] Add `jpeg_decode_fast420_batch_coeffs` using the same entropy scheduling as `jpeg_decode_fast420_batch`.
- [x] Add `jpeg_idct_deposit_fast420_batch` using one thread per block and the existing IDCT/deposit helpers.

### Task 3: Host Dispatch

**Files:**
- Modify: `crates/signinum-jpeg-metal/src/compute.rs`

- [x] Add split pipeline state fields and compile the two new kernels.
- [x] Add scratch sizing for fast420 batch coefficients and DC flags.
- [x] Add `SIGNINUM_JPEG_METAL_SPLIT_FAST420_BATCH=1` selection in the fast420 full-batch path.
- [x] Dispatch coefficient decode, IDCT/deposit, then the existing pack kernel for the split path.

### Task 4: Correctness And Benchmarks

**Files:**
- Modify: `crates/signinum-jpeg-metal/src/compute.rs`
- Modify: `crates/signinum-jpeg-metal/tests/shader_integrity.rs`

- [x] Add or reuse a Metal correctness test that forces the split path and compares output bytes to CPU decode.
- [x] Run `cargo test -p signinum-jpeg-metal --all-targets`.
- [x] Benchmark fused and split paths for generated batch64 at `512x512` and `1024x1024`.
- [x] Keep the split path opt-in unless benchmark evidence supports making it default.

## Benchmark Result

Generated 4:2:0 JPEG batch64, Apple M4 Pro:

- `512x512`: CPU mean `101.53 ms`; fused Metal mean `11.24 ms`; split Metal mean `12.32 ms`.
- `1024x1024`: CPU mean `447.26 ms`; fused Metal mean `37.23 ms`; split Metal mean `40.03 ms`.

The split path is correct but not faster in this scalar-IDCT form, so it remains
opt-in as a foundation for a future cooperative/threadgroup IDCT pass.
