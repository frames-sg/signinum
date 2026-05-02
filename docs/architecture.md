# Architecture

This document is the system map for `signinum`. It is the first thing a new
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
- [`docs/wsi-decode-api.md`](wsi-decode-api.md) — public WSI decode API guide.
- [`docs/superpowers/HANDOFF-2026-04-23-adaptive-codec-runtime.md`](superpowers/HANDOFF-2026-04-23-adaptive-codec-runtime.md)
  — most recent in-flight design handoff (adaptive backend routing).
- Crate-level `README.md` files where present — crate-scoped contracts and
  feature notes.

## System map

The workspace is a single Cargo workspace defined in [`Cargo.toml`](../Cargo.toml).
All crates live under `crates/` and share `edition = 2021` and
`rust-version = 1.94`. CPU-first 1.0 crates use the workspace package version;
implementation and adapter crates stay on explicit pre-1.0 versions.

| Crate | Layer | Role |
|-------|-------|------|
| `signinum-core` | foundation | Shared traits, pixel/sample types, backend capability metadata, device-surface contracts, scratch/context contracts. No image-format logic. |
| `signinum-tilecodec` | codec | Tile decompression primitives: Deflate, Zstd, LZW, Uncompressed. Implements `TileDecompress` from `core`. |
| `signinum-jpeg` | codec | Native pure-Rust JPEG decode for WSI tiles. CPU-first. Owns SIMD backends and fused entropy/IDCT/upsample paths. |
| `signinum-j2k-native` | codec engine | Published implementation dependency for `signinum-j2k`; not the stable user-facing API. Lives under `#![forbid(unsafe_code)]` and uses `fearless_simd`. |
| `signinum-j2k` | codec | Public JPEG 2000 / HTJ2K crate. Wraps `j2k-native` with the signinum-core trait surface (inspect, decode, ROI, scaled, row-bounded, tile-batch). |
| `signinum-j2k-compare` | dev-only | OpenJPEG FFI bindings used as a reference decoder for conformance and parity testing. Unpublished. |
| `signinum-jpeg-metal` | adapter | Apple Metal device-output adapter for `signinum-jpeg`. Hosts compute kernels for color conversion, interleave/pack, and `MTLBuffer` production. |
| `signinum-j2k-metal` | adapter | Apple Metal device-output adapter for `signinum-j2k`. Same shape as the JPEG adapter. |
| `signinum-jpeg-cuda` | adapter | CUDA-facing API adapter for JPEG. `Auto`/`Cpu` stay host-backed; explicit full-frame RGB8 CUDA requests use nvJPEG when `cuda-runtime`, a CUDA driver, and `libnvjpeg` are available, with CPU decode plus CUDA upload fallback for unsupported shapes. |
| `signinum-j2k-cuda` | adapter | CUDA-facing API adapter for J2K. Explicit CUDA requests upload CPU-decoded output into CUDA device memory when `cuda-runtime` and a CUDA driver are available. |
| `signinum-cli` | binary | `signinum inspect <file>` entry point. Header parsing only, no decode. |

Out-of-tree but in-repo:

- `corpus/` — test data: `wsi-samples/`, `conformance/`, `regressions/`,
  `fuzz-seeds/`, each with a manifest describing source, license, and tolerance.
- `paper/` — research paper materials.
- `target/` — build output (gitignored).

## Layered architecture and dependency rules

signinum is organized as four concentric layers. Dependencies must flow
inward only. An agent adding a dependency edge that points outward is changing
the architecture and should stop and update this document first.

```
foundation  →  codec engines  →  codecs  →  adapters  →  binary
```

| Layer | Members | May depend on |
|-------|---------|---------------|
| foundation | `signinum-core` | `thiserror` only. No other workspace crate. `no_std + alloc` posture. Contains only the x86 CPUID/XGETBV unsafe required for CPU feature detection. |
| codec engines | `signinum-j2k-native` | foundation. Internal only. Not re-exported. |
| codecs | `signinum-jpeg`, `signinum-j2k`, `signinum-tilecodec` | foundation, codec engines. Must not depend on each other. Must not depend on adapters or `cli`. |
| adapters | `signinum-jpeg-metal`, `signinum-j2k-metal`, `signinum-jpeg-cuda`, `signinum-j2k-cuda` | foundation, exactly one matching codec, optional engine for the matching codec. Adapters in different format families must not depend on each other. |
| binary | `signinum-cli` | foundation, codecs. Must not depend on adapters (kept host-neutral). |
| dev-only | `signinum-j2k-compare` | foundation. Used as a reference comparator in tests/benches; never a runtime dependency. |

Hard rules enforced today (the goal is to mechanize these as the workspace
matures, mirroring harness-engineering structural tests):

1. `signinum-core` is a leaf in the import graph. It does not import any
   other workspace crate.
2. Codec crates do not import each other. Cross-format work goes through
   `core` types or through caller code.
3. Adapter crates are additive. Removing all adapter crates must leave the
   codec stack fully functional on CPU.
4. Metal sources are gated by `cfg(target_os = "macos")`. Non-macOS hosts
   compile the adapter crate to a thin fallback that exercises the same
   public API but reports unavailability.
5. CUDA sources expose the same device-output surface. Explicit CUDA requests
   produce CUDA device memory when `cuda-runtime` and a CUDA driver are
   available. JPEG full-frame RGB8 requests may use nvJPEG; unsupported JPEG
   shapes and J2K CUDA requests use CPU decode plus CUDA upload.
6. `signinum-jpeg` keeps its NEON and x86 intrinsics scoped per-backend
   in `crates/signinum-jpeg/src/backend/`. `signinum-j2k-native` keeps
   its SIMD behind `fearless_simd` so the engine can stay
   `#![forbid(unsafe_code)]`.
7. Adapter crates consume codec planning hooks through public `adapter`
   modules. Imports from codec `__private` modules are not allowed.

## Crate dependency graph

Workspace edges (excluding external crates and `dev-dependencies`):

```
signinum-core         (leaf)

signinum-tilecodec    -> signinum-core

signinum-jpeg         -> signinum-core
signinum-jpeg-metal   -> signinum-jpeg, signinum-core
signinum-jpeg-cuda    -> signinum-jpeg, signinum-core

signinum-j2k-native   -> signinum-core
signinum-j2k          -> signinum-j2k-native, signinum-core
signinum-j2k-metal    -> signinum-j2k, signinum-j2k-native, signinum-core
signinum-j2k-cuda     -> signinum-j2k, signinum-core

signinum-cli          -> signinum-jpeg, signinum-j2k, signinum-core

signinum-j2k-compare  -> signinum-core (test/bench reference only)
```

## Core abstractions

These live in `signinum-core` and are the contract every codec and adapter
implements. New extension points belong here.

### Codec entry traits

- `ImageCodec` — base trait. Associated types: `Error`, `Warning`, `Pool`.
- `ImageDecode<'a>` — CPU decode surface. Methods include `inspect`, `parse`,
  `decode_into`, `decode_into_with_scratch`, `decode_region_into`,
  `decode_scaled_into`, and `decode_region_scaled_into`.
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
    │   decode_region_into / decode_scaled_into /              │
    │   decode_region_scaled_into / decode_rows                │
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
`crates/signinum-jpeg/src/entropy/sequential.rs` because splitting them
regresses WSI tile-batch performance. Splitting that module is planned but
gated on stable benchmark and parity coverage.

J2K parses boxes (COD, QCD, etc.) and the codestream, then drives
`signinum-j2k-native` (DWT, context modeling, arithmetic decode) before
filling the caller-owned output buffer. ROI and reduced-resolution requests
share the same core contract: the ROI is expressed in source coordinates, and
the returned decoded rectangle covers the floor-start/ceil-end projection onto
the requested reduced-resolution grid.

The Metal v1 path keeps parse and entropy decode on CPU and hands decoded
component rows or planes to compute kernels for color conversion, interleave,
clamp, and pack where a kernel path exists. J2K Metal exposes full, ROI,
scaled, and combined ROI+scaled device surfaces; explicit Metal requests return
Metal-backed surfaces on macOS, while unsupported host or platform shapes fail
through the adapter error path.

Metal adapter routing is explicit after the CPU-first 1.0 line.
`BackendRequest::Cpu` returns host-backed CPU surfaces. `BackendRequest::Auto`
may select Metal only for validated adapter-supported shapes; otherwise it
falls back to a host-backed CPU surface. `BackendRequest::Metal` is strict: it
returns a Metal-backed surface for supported shapes or a clear
unsupported/unavailable error. It does not silently return CPU output.

## Backend story

There are three target backends. Selection is explicit in the public API.

- **CPU** — always available. Pure Rust, with SIMD where it is profitable.
  This is the baseline; every device path must have a working CPU fallback
  with equivalent results to within documented tolerance.
- **Metal** — Apple Silicon macOS. Compute kernels live next to the adapter
  crate. The adapter owns its `MTLDevice`/`MTLCommandQueue` session and
  produces `MTLBuffer`-backed `DeviceSurface`s so downstream GPU pipelines
  can consume the result without an extra download.
- **CUDA** — explicit device-memory output. `Auto` and `Cpu` return CPU-backed
  host surfaces. `BackendRequest::Cuda` returns CUDA device memory when the
  `cuda-runtime` feature and a CUDA driver are available, and otherwise reports
  CUDA as unavailable. JPEG full-frame RGB8 can decode through nvJPEG when
  `libnvjpeg` is available; other CUDA shapes use CPU decode plus CUDA upload.

`BackendRequest::Auto` stays conservative: small or low-yield decodes are
served from CPU; larger batches with supported shapes can be routed to a
device backend. The adaptive routing policy is iterating in
[`docs/superpowers/HANDOFF-2026-04-23-adaptive-codec-runtime.md`](superpowers/HANDOFF-2026-04-23-adaptive-codec-runtime.md).

## Lifecycle and ownership

signinum deliberately externalizes anything that resembles a runtime.

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
| New shared trait, pixel format, or backend kind | `signinum-core` |
| New JPEG decode shape, marker, or CPU optimization | `signinum-jpeg` |
| New JPEG GPU shape | `signinum-jpeg-metal` (or `-cuda`) |
| New J2K codestream feature, ROI/scaled support | `signinum-j2k-native`, surfaced through `signinum-j2k` |
| New tile decompression codec (e.g. LZ4) | `signinum-tilecodec` |
| New CLI subcommand | `signinum-cli` |
| New conformance fixture | `corpus/conformance/` plus its manifest |
| New regression repro | `corpus/regressions/issue-NNN.<ext>` |

When in doubt, prefer adding to the lowest layer that the new behavior
genuinely requires, and never bypass `signinum-core` to share types
between codec crates.

## Build and platform

- Rust edition `2021`, MSRV `1.94`, pinned by [`rust-toolchain.toml`](../rust-toolchain.toml).
- Supported decode hosts: `x86_64` and `aarch64` only. Other targets fail
  to build by design.
- Metal adapters compile and run on Apple Silicon macOS. On other hosts the
  adapter crate compiles to a fallback surface.
- CUDA adapter crates expose runtime device-memory output for explicit CUDA
  requests when built with `cuda-runtime` on hosts with a CUDA driver. Hosts
  without CUDA return the documented unavailable error. JPEG full-frame RGB8
  CUDA decode additionally uses `libnvjpeg` when available.
- Release profile: `lto = "fat"`, `codegen-units = 1`, `strip = "symbols"`,
  `opt-level = 3`. `release-bench` inherits `release` but keeps debug info.
- Notable feature flags:
  - `signinum-j2k-native`: `std` (default), `simd` (default, requires
    `std`), `logging`.
  - `signinum-jpeg`: `scalar-only` retained for fuzzing and reference.
  - `signinum-jpeg-cuda`, `signinum-j2k-cuda`: `cuda-runtime` enables CUDA Driver
    API device allocation and explicit CUDA requests. `signinum-jpeg-cuda`
    also loads nvJPEG at runtime for full-frame RGB8 JPEG decode when present.

## Active areas

These are the surfaces under active change. Treat anything here as
provisional and check the most recent commits before relying on it.

- Metal adapter hardening: aligning the adapter session model with the
  wgpu-hal style, exposing the underlying `MTLBuffer` from `DeviceSurface`,
  and pushing more of the J2K full-tile path onto the GPU.
- Adaptive backend routing: deciding when `BackendRequest::Auto` should
  upgrade a batch to Metal vs. stay on CPU.
- Keeping the public WSI decode API guide aligned with the core trait surface.
- Broadening release CI and adding self-hosted x86_64 GPU benchmark coverage.

## Stability posture

CPU-first 1.0 covers `signinum-core`, `signinum-jpeg`, `signinum-j2k`,
`signinum-tilecodec`, and `signinum-cli`. `signinum-j2k-native` is published as an
implementation dependency, not as the primary stable API. The Metal adapter
APIs remain on the post-1.0 hardening track. The CUDA adapter APIs now expose
runtime CUDA device-memory output and nvJPEG JPEG decode, but remain pre-1.0
while broader CUDA decode and performance work harden. Breaking changes to any of these surfaces should be
reflected here and in [`CHANGELOG.md`](../CHANGELOG.md).
