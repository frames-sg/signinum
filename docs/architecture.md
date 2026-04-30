# Architecture

This document is the system map for `ashlar`. It is the first thing a new
contributor or coding agent should read before changing code, and it is the
single source of truth for how the workspace is shaped, where each
responsibility lives, and which dependency directions are legal. Anything that
is not described here, in a referenced design note, or in a crate-level README
should be treated as undocumented and verified before it is relied on.

The structure of this file follows the harness-engineering convention of
keeping a repository-local map, an explicit layering, and cross-links to
design notes that an agent can reach without leaving the repo.

## Companion documents

- [`README.md`](../README.md) — public surface, supported APIs, MSRV, examples.
- [`CHANGELOG.md`](../CHANGELOG.md) — release history.
- [`docs/bench.md`](bench.md) — benchmark methodology and comparator policy.
- [`docs/parity.md`](parity.md) — parity expectations against reference decoders.
- [`docs/release.md`](release.md) — release staging notes.
- [`HANDOFF-2026-04-23-adaptive-codec-runtime.md`](private-docs/HANDOFF-2026-04-23-adaptive-codec-runtime.md)
  — most recent in-flight design handoff (adaptive backend routing).
- Crate-level `README.md` files where present — crate-scoped contracts and
  feature notes.

## System map

The workspace is a single Cargo workspace defined in [`Cargo.toml`](../Cargo.toml).
All crates live under `crates/` and share `version = 0.1.0`, `edition = 2021`,
and `rust-version = 1.94`.

| Crate | Layer | Role |
|-------|-------|------|
| `ashlar-core` | foundation | Shared traits, pixel/sample types, backend capability metadata, device-surface contracts, scratch/context contracts. No image-format logic. |
| `ashlar-tilecodec` | codec | Tile decompression primitives: Deflate, Zstd, LZW, Uncompressed. Implements `TileDecompress` from `core`. |
| `ashlar-jpeg` | codec | Native pure-Rust JPEG decode for WSI tiles. CPU-first. Owns SIMD backends and fused entropy/IDCT/upsample paths. |
| `ashlar-j2k-native` | codec engine | Internal, unpublished pure-Rust JPEG 2000 / HTJ2K engine. Lives under `#![forbid(unsafe_code)]` and uses `fearless_simd`. |
| `ashlar-j2k` | codec | Public JPEG 2000 / HTJ2K crate. Wraps `j2k-native` with the ashlar-core trait surface (inspect, decode, ROI, scaled, row-bounded, tile-batch). |
| `ashlar-j2k-compare` | dev-only | OpenJPEG FFI bindings used as a reference decoder for conformance and parity testing. Unpublished. |
| `ashlar-jpeg-metal` | adapter | Apple Metal device-output adapter for `ashlar-jpeg`. Hosts compute kernels for color conversion, interleave/pack, and `MTLBuffer` production. |
| `ashlar-j2k-metal` | adapter | Apple Metal device-output adapter for `ashlar-j2k`. Same shape as the JPEG adapter. |
| `ashlar-jpeg-cuda` | adapter | CUDA-facing API adapter for JPEG. In `0.1.0` it validates fallback / unavailability behavior and exposes the device-output API surface; no runtime CUDA execution is shipped. |
| `ashlar-j2k-cuda` | adapter | CUDA-facing API adapter for J2K. Same shape and same `0.1.0` constraints as `jpeg-cuda`. |
| `ashlar-cli` | binary | `ashlar inspect <file>` entry point. Header parsing only, no decode. |

Out-of-tree but in-repo:

- `corpus/` — test data: `wsi-samples/`, `conformance/`, `regressions/`,
  `fuzz-seeds/`, each with a manifest describing source, license, and tolerance.
- `paper/` — research paper materials.
- `target/` — build output (gitignored).

## Layered architecture and dependency rules

ashlar is organized as four concentric layers. Dependencies must flow
inward only. An agent adding a dependency edge that points outward is changing
the architecture and should stop and update this document first.

```
foundation  →  codec engines  →  codecs  →  adapters  →  binary
```

| Layer | Members | May depend on |
|-------|---------|---------------|
| foundation | `ashlar-core` | `thiserror` only. No other workspace crate. `no_std + alloc` posture. Contains only the x86 CPUID/XGETBV unsafe required for CPU feature detection. |
| codec engines | `ashlar-j2k-native` | foundation. Internal only. Not re-exported. |
| codecs | `ashlar-jpeg`, `ashlar-j2k`, `ashlar-tilecodec` | foundation, codec engines. Must not depend on each other. Must not depend on adapters or `cli`. |
| adapters | `ashlar-jpeg-metal`, `ashlar-j2k-metal`, `ashlar-jpeg-cuda`, `ashlar-j2k-cuda` | foundation, exactly one matching codec, optional engine for the matching codec. Adapters in different format families must not depend on each other. |
| binary | `ashlar-cli` | foundation, codecs. Must not depend on adapters (kept host-neutral). |
| dev-only | `ashlar-j2k-compare` | foundation. Used as a reference comparator in tests/benches; never a runtime dependency. |

Hard rules enforced today (the goal is to mechanize these as the workspace
matures, mirroring harness-engineering structural tests):

1. `ashlar-core` is a leaf in the import graph. It does not import any
   other workspace crate.
2. Codec crates do not import each other. Cross-format work goes through
   `core` types or through caller code.
3. Adapter crates are additive. Removing all adapter crates must leave the
   codec stack fully functional on CPU.
4. Metal sources are gated by `cfg(target_os = "macos")`. Non-macOS hosts
   compile the adapter crate to a thin fallback that exercises the same
   public API but reports unavailability.
5. CUDA sources expose the same device-output surface but make no runtime
   performance claim in `0.1.0`.
6. `ashlar-jpeg` keeps its NEON and x86 intrinsics scoped per-backend
   in `crates/ashlar-jpeg/src/backend/`. `ashlar-j2k-native` keeps
   its SIMD behind `fearless_simd` so the engine can stay
   `#![forbid(unsafe_code)]`.
7. Adapter crates consume codec planning hooks through public `adapter`
   modules. Imports from codec `__private` modules are not allowed.

## Crate dependency graph

Workspace edges (excluding external crates and `dev-dependencies`):

```
ashlar-core         (leaf)

ashlar-tilecodec    -> ashlar-core

ashlar-jpeg         -> ashlar-core
ashlar-jpeg-metal   -> ashlar-jpeg, ashlar-core
ashlar-jpeg-cuda    -> ashlar-jpeg, ashlar-core

ashlar-j2k-native   -> ashlar-core
ashlar-j2k          -> ashlar-j2k-native, ashlar-core
ashlar-j2k-metal    -> ashlar-j2k, ashlar-j2k-native, ashlar-core
ashlar-j2k-cuda     -> ashlar-j2k, ashlar-core

ashlar-cli          -> ashlar-jpeg, ashlar-j2k, ashlar-core

ashlar-j2k-compare  -> ashlar-core (test/bench reference only)
```

## Core abstractions

These live in `ashlar-core` and are the contract every codec and adapter
implements. New extension points belong here.

### Codec entry traits

- `ImageCodec` — base trait. Associated types: `Error`, `Warning`, `Pool`.
- `ImageDecode<'a>` — CPU decode surface. Methods include `inspect`, `parse`,
  `decode_into`, `decode_into_with_scratch`, `decode_region_into`,
  `decode_scaled_into`.
- `ImageDecodeRows<'a, S: RowSink<_>>` — row-bounded decode for large tiles.
- `ImageDecodeDevice<'a>` — synchronous device decode; returns a
  `DeviceSurface`.
- `ImageDecodeSubmit<'a>` — asynchronous device decode; returns a
  `DeviceSubmission` whose `wait()` produces a `DeviceSurface`.
- `TileBatchDecode` — stateless tile decode with a caller-owned
  `DecoderContext` reused across tiles.
- `TileDecompress` — generic tile decompression (used by `tilecodec`).

### Backend and surface model

- `BackendKind` — `Cpu | Metal | Cuda`.
- `BackendRequest` — `Auto | Cpu | Metal | Cuda`. Callers state intent.
  `Auto` may resolve to a CPU-backed surface; explicit unsupported device
  requests return an error before decode work.
- `DeviceSurface` — trait describing decoded data sitting on a backend
  (CPU buffer, `MTLBuffer`, etc.). Queryable for backend kind, dimensions,
  pixel format, byte length.
- `DeviceSubmission` — async handle with `wait() -> DeviceSurface`.
- `ReadySubmission<T, E>` — synchronous submission used by CPU fallbacks.
- `CpuFeatures` — runtime detection of AVX2 / SSE4.1 / NEON, cached.

### Pixel and layout types

- `PixelFormat` — `Rgb8 | Rgba8 | Gray8 | Rgb16 | Rgba16 | Gray16`.
- `PixelLayout` — `Rgb | Rgba | Gray`.
- `Sample`, `SampleType`.
- `Downscale` — reduced-resolution decode specifier.
- `RowSink<S>` — caller-implemented sink for row streaming.
- `Rect` — ROI.
- `Info` — image metadata: dimensions, colorspace, components, bit depth,
  tile layout, MCU/coded-unit layout, restart interval, resolution levels.
- `Colorspace` — `Grayscale | YCbCr | Rgb | Cmyk | Ycck | SRgb | SGray |
  IccTagged | Rct | Ict`.
- `DecodeOutcome<W>` — decoded `Rect` plus `Vec<W>` warnings.

### Caller-owned state

- `ScratchPool` — caller-owned reusable allocations, reset per operation.
- `CodecContext` — codec-specific state (e.g., JPEG quantization tables).
- `DecoderContext<C>` — wrapper holding a `CodecContext`, persistent across
  tiles in a batch.

These three types encode a deliberate harness contract: the codec never
hides allocation, threading, or runtime state. WSI readers compose those
themselves.

## Decode pipeline

The shape is the same for both image codecs, modulo the engine inside.

```
compressed bytes
    │
    ▼
inspect(bytes)              → Info (no decode)
    │
    ▼
view = View::parse(bytes)   → borrowed parse (no copy)
    │
    ▼
decoder = Decoder::from_view(view)
    │
    ├─ CPU path ───────────────────────────────────────────────┐
    │   decode_into / decode_into_with_scratch                 │
    │   decode_region_into / decode_scaled_into / decode_rows  │
    │     entropy → IDCT or DWT → color conv → output buffer   │
    │     SIMD: AVX2 / SSE4.1 / NEON (jpeg)                    │
    │           fearless_simd (j2k-native)                     │
    │   returns DecodeOutcome<Warning>                         │
    │                                                           │
    └─ Device path (Metal today, CUDA API-only) ───────────────┘
        submit_to_device(session, fmt, BackendRequest::Metal)
            │
            ▼
        capability check
            │
            ├─ supported shape: prepare packet, upload to MTLBuffer,
            │   dispatch compute kernel (color conv, interleave/pack)
            │   → DeviceSubmission → wait() → Surface (with MTLBuffer)
            │
            └─ unsupported explicit backend: fail before decode;
                Auto/Cpu may wrap CPU output in a host-backed DeviceSurface
```

JPEG is fully fused on CPU: entropy decode, IDCT scheduling, upsampling, ROI,
and the fast 4:2:0 path live together in
`crates/ashlar-jpeg/src/entropy/sequential.rs` because splitting them
regresses WSI tile-batch performance. Splitting that module is planned but
gated on stable benchmark and parity coverage.

J2K parses boxes (COD, QCD, etc.) and the codestream, then drives
`ashlar-j2k-native` (DWT, context modeling, arithmetic decode) before
filling the caller-owned output buffer.

The Metal v1 path keeps parse and entropy decode on CPU and hands decoded
component rows or planes to compute kernels for color conversion, interleave,
clamp, and pack. ROI staging is currently CPU-side for J2K Metal.

## Backend story

There are three target backends. Selection is explicit in the public API.

- **CPU** — always available. Pure Rust, with SIMD where it is profitable.
  This is the baseline; every device path must have a working CPU fallback
  with equivalent results to within documented tolerance.
- **Metal** — Apple Silicon macOS. Compute kernels live next to the adapter
  crate. The adapter owns its `MTLDevice`/`MTLCommandQueue` session and
  produces `MTLBuffer`-backed `DeviceSurface`s so downstream GPU pipelines
  can consume the result without an extra download.
- **CUDA** — API-only in `0.1.0`. The adapter validates that explicit
  `BackendRequest::Cuda` returns the documented unavailable error before
  decode work. `Auto` and `Cpu` return CPU-backed host surfaces. No CUDA
  runtime or performance claim is made in this checkpoint.

`BackendRequest::Auto` stays conservative: small or low-yield decodes are
served from CPU; larger batches with supported shapes can be routed to a
device backend. The adaptive routing policy is iterating in
[`HANDOFF-2026-04-23-adaptive-codec-runtime.md`](private-docs/HANDOFF-2026-04-23-adaptive-codec-runtime.md).

## Lifecycle and ownership

ashlar deliberately externalizes anything that resembles a runtime.

- **Buffers** are caller-owned. Every `decode_*_into` writes into a slice the
  caller provides.
- **Scratch** is caller-owned via `ScratchPool` and reset per operation. WSI
  readers reuse pools across tiles to keep allocation off the hot path.
- **Decoder context** is caller-owned via `DecoderContext<C>` and reused
  across tiles in a batch.
- **Sessions** for device backends are owned by the adapter and held by the
  caller for the lifetime of a batch. They wrap `MTLDevice`/`MTLCommandQueue`
  state on Metal.
- **Threading and pyramid policy** are entirely the caller's responsibility.
  No crate spawns threads or owns I/O.

## Where to add things

| Change | Crate |
|--------|-------|
| New shared trait, pixel format, or backend kind | `ashlar-core` |
| New JPEG decode shape, marker, or CPU optimization | `ashlar-jpeg` |
| New JPEG GPU shape | `ashlar-jpeg-metal` (or `-cuda`) |
| New J2K codestream feature, ROI/scaled support | `ashlar-j2k-native`, surfaced through `ashlar-j2k` |
| New tile decompression codec (e.g. LZ4) | `ashlar-tilecodec` |
| New CLI subcommand | `ashlar-cli` |
| New conformance fixture | `corpus/conformance/` plus its manifest |
| New regression repro | `corpus/regressions/issue-NNN.<ext>` |

When in doubt, prefer adding to the lowest layer that the new behavior
genuinely requires, and never bypass `ashlar-core` to share types
between codec crates.

## Build and platform

- Rust edition `2021`, MSRV `1.94`, pinned by [`rust-toolchain.toml`](../rust-toolchain.toml).
- Supported decode hosts: `x86_64` and `aarch64` only. Other targets fail
  to build by design.
- Metal adapters compile and run on Apple Silicon macOS. On other hosts the
  adapter crate compiles to a fallback surface.
- CUDA adapter crates currently expose the API surface only. No CUDA runtime
  is required to build or test the workspace, and explicit CUDA requests fail
  as unavailable before decode validation.
- Release profile: `lto = "fat"`, `codegen-units = 1`, `strip = "symbols"`,
  `opt-level = 3`. `release-bench` inherits `release` but keeps debug info.
- Notable feature flags:
  - `ashlar-j2k-native`: `std` (default), `simd` (default, requires
    `std`), `logging`.
  - `ashlar-jpeg`: `scalar-only` retained for fuzzing and reference.
  - `ashlar-jpeg-cuda`, `ashlar-j2k-cuda`: `cuda-runtime` is
    declared but unused in `0.1.0`.

## Active areas

These are the surfaces under active change. Treat anything here as
provisional and check the most recent commits before relying on it.

- JPEG 2000 / HTJ2K ROI and reduced-resolution performance work in
  `ashlar-j2k` and `ashlar-j2k-native`.
- Metal adapter hardening: aligning the adapter session model with the
  wgpu-hal style, exposing the underlying `MTLBuffer` from `DeviceSurface`,
  and pushing more of the J2K full-tile path onto the GPU.
- Adaptive backend routing: deciding when `BackendRequest::Auto` should
  upgrade a batch to Metal vs. stay on CPU.
- Tightening the public WSI decode API documentation toward 1.0.
- Broadening release CI and adding x86_64 GPU benchmark coverage.

## Stability posture

`0.1.0` is a pre-1.0 source release. The CPU WSI decode surfaces in
`ashlar-jpeg`, `ashlar-j2k`, and `ashlar-tilecodec` are the
primary supported APIs. The Metal adapter APIs are hardening. The CUDA
adapter APIs are compatibility surfaces only. Breaking changes to any of
these surfaces should be reflected here and in [`CHANGELOG.md`](../CHANGELOG.md).
