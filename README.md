# J2K — Pure-Rust JPEG 2000 and HTJ2K Codec

[![crates.io](https://img.shields.io/crates/v/j2k.svg)](https://crates.io/crates/j2k)
[![docs.rs](https://img.shields.io/docsrs/j2k)](https://docs.rs/j2k)
[![CI](https://github.com/frames-sg/j2k/actions/workflows/ci.yml/badge.svg)](https://github.com/frames-sg/j2k/actions/workflows/ci.yml)
[![downloads](https://img.shields.io/crates/d/j2k.svg)](https://crates.io/crates/j2k)
[![license](https://img.shields.io/crates/l/j2k.svg)](#license)

**Docs & guides:** [Pure-Rust JPEG 2000 codec documentation](https://frames-sg.github.io/j2k/rust-jpeg2000-codec/)

**Release status:** `0.11.0` is published and security-supported. See the
[release notes](CHANGELOG.md), [release policy](docs/release.md), and
[security policy](SECURITY.md).

**A general-purpose, GPU-accelerated JPEG 2000 and HTJ2K codec with safe Rust
APIs, a portable CPU baseline, and production CUDA and Metal paths.**

J2K provides JPEG 2000 / HTJ2K decode, encode, recode, and
JPEG-to-HTJ2K coefficient-domain transcoding. Its public APIs cover whole-image,
region, reduced-resolution, tile, batch, host-output, and resident-device
workflows without coupling the codec to a particular application domain. The
workspace is dual-licensed under MIT/Apache-2.0.

Region and reduced-resolution decoding plus retained tiled batch plans avoid
whole-image work for slide-scale and other large-image readers; those are codec
capabilities, not domain-specific APIs.

Measured product GPU decode paths keep parsing on CPU while moving supported
Tier-1, dequantization, IDWT, MCT, output, and transfer work to CUDA or Metal.
On the current external HTJ2K/JPH routing matrix, fixed `Auto` policy qualifies
8 of 10 CUDA cells and 3 of 10 Metal cells; every promoted cell produced the
same bytes as CPU, exceeded the required 10% median speedup, and had a
non-overlapping 95% confidence interval. A **formal device-native** route would
require every reported stage to execute on the device; the current public
pipelines make no such claim.

Current-tree Metal host-output encode evidence additionally qualifies lossless
HTJ2K RGB8 at 1024 x 1024 and Gray8/RGB8 at 2048 x 2048. Those hybrid routes
run coefficient preparation and HT Tier-1 on Metal, then packetize on CPU.
Other measured 512 x 512 and Gray8 1024 x 1024 cells remain CPU-routed. The
performance scope is the measured Apple M4 Pro; exact results and qualifications
are in [docs/benchmark-evidence.md](docs/benchmark-evidence.md).

Release `0.11.0` formally claims ISO/IEC 15444-4:2024 / ITU-T T.803 v3
**Profile-1 Cclass-1**, **Profile-1 Cclass-1HF**, and **Annex G JP2 reader**
compliance for the CPU IUT. The same published release evidence covers these
selected HTJ2K Part 15 points:

- HTJ2K Part 15: **DS1-HM Cclass-1h, MMAGB 15**, including DS1-HT, DS0-HM,
  and DS0-HT subset evidence; **Cclass-1HFh, MMAGB 20**; and **Annex G JPH
  reader at MMAGB 15**.

The exact-SHA macOS arm64, Linux x86-64, and Windows x86-64 CPU reports each
pass all 160 selected cases—90 Part 1 and 70 Part 15—with zero skips. The CUDA
and Metal adapter-IUT reports pass the same 160 cases on real hardware with the
truthful combined headline **0/160 device-native, 81/160 hybrid, 79/160
CPU-routed** for each backend. The Part 15 split is 33 hybrid and 37 CPU-routed.
Production-owned dispatch counters, not capability-table predictions,
substantiate every stage label.

All five reports identify exact release SHA
`09d746a7b040258eb5dd505b44384eec0152a8b9` and are attached to the
[v0.11.0 release](https://github.com/frames-sg/j2k/releases/tag/v0.11.0). CPU
encoder evidence passes 56/56 cases; CUDA and Metal each pass 35/35. Encoder
results are informative Annex D/F evidence, not formal decoder conformance.
The exact scope and report rules are in
[docs/t803-conformance.md](docs/t803-conformance.md). T.803 does not establish
robustness, security, adoption, or performance.

Speed matters, but it is not the reason this project exists. The strategic
gap is a memory-safety-oriented Rust codec with a portable CPU baseline,
multi-vendor GPU adapters, explicit support boundaries, and reproducible
benchmark gates. The public crate release centers on `j2k`, with lower-level
crates for native codec internals, device adapters, JPEG input, and transcode
pipelines.

The codec support boundary is intentionally scoped and explicit: JPEG 2000
Part 1 still-image codestream features, JP2 wrapping, HTJ2K Part 15
codestreams, and JPH wrapping. JPX / JPEG 2000 Part 2 extensions are outside
this boundary unless a feature is required for standard JP2/JPH still-image
correctness. The implementation matrix is
[docs/public-support.md](docs/public-support.md).

The APIs expose codec operations rather than application-specific workflow
abstractions. Medical imaging, geospatial systems, digital preservation,
servers, desktop applications, and large tiled-image readers can use the same
decoder, encoder, and transcode surfaces. Domain containers, indexing,
application metadata, and workflow validation remain outside the codec layer.

## Why J2K exists

JPEG 2000 is still common in medical imaging, geospatial imagery, digital
preservation, and large tiled-image systems, but the implementation landscape
forces awkward tradeoffs:

| Option | Tradeoff J2K avoids |
| --- | --- |
| NVIDIA CUDA JPEG 2000 runtime | CUDA/NVIDIA GPU stacks are a good fit for NVIDIA-only deployments, but not for portable Rust applications that also need Metal or CPU-first operation. |
| [OpenJPEG](https://github.com/uclouvain/openjpeg) | Mature C implementation and useful comparator, but C codecs keep memory-safety risk on the adopter. |
| [Grok](https://github.com/GrokImageCompression/grok) | Capable C++ JPEG 2000 / HTJ2K implementation, but AGPL licensing is not usable for every commercial or embedded integration. |

J2K's intended position is different: a safe Rust public API, isolated
unsafe boundaries for FFI/GPU work, no active runtime dependency on NVIDIA's
JPEG 2000 runtime, strict errors for unsupported device routes, and dual
MIT/Apache-2.0 licensing.

## Memory Safety Posture

J2K is designed for safe Rust integration with untrusted image inputs. The
public codec API is safe Rust.
Unsafe code is isolated at audited FFI, GPU integration, architecture-specific
SIMD/intrinsic, allocation, and bounded pointer/buffer boundaries, where inputs
are validated and unsupported shapes fail with errors. The exhaustive inventory
is maintained in [docs/unsafe-audit.md](docs/unsafe-audit.md).

This is an engineering posture backed by an explicit unsafe inventory, tests,
fuzzing, and review—not a formal proof that all implementation defects are
impossible. It is also not a claim that every malformed codestream is accepted
or that every device path is faster than CPU. CPU remains the portable
correctness baseline; GPU acceleration is promoted only for measured paths.

## Quickstart

Use the public Rust API for application integration:

```bash
cargo add j2k
```

Run the command-line tool for quick inspection and JPEG-to-HTJ2K transcode
smoke tests:

```bash
cargo install j2k-cli
j2k inspect input.jp2
j2k transcode input.jpg output.j2k --htj2k --lossless-53
```

Runnable repository examples:

- `cargo run -p j2k --example decode_generated`
  ([crates/j2k/examples/decode_generated.rs](crates/j2k/examples/decode_generated.rs))
- `cargo run -p j2k-jpeg --example inspect`
  ([crates/j2k-jpeg/examples/inspect.rs](crates/j2k-jpeg/examples/inspect.rs))
- `cargo run -p j2k-transcode --example jpeg_to_htj2k`
  ([crates/j2k-transcode/examples/jpeg_to_htj2k.rs](crates/j2k-transcode/examples/jpeg_to_htj2k.rs))
- `cargo run -p j2k-transcode-metal --example jpeg_to_htj2k_route_report`
  ([crates/j2k-transcode-metal/examples/jpeg_to_htj2k_route_report.rs](crates/j2k-transcode-metal/examples/jpeg_to_htj2k_route_report.rs))
- `cargo run -p j2k-metal --example decode_route_report`
  ([crates/j2k-metal/examples/decode_route_report.rs](crates/j2k-metal/examples/decode_route_report.rs))
- `cargo run -p j2k-metal --example htj2k_encode_auto_report`
  ([crates/j2k-metal/examples/htj2k_encode_auto_report.rs](crates/j2k-metal/examples/htj2k_encode_auto_report.rs))
- `cargo run -p j2k-metal --example resident_encode_buffer`
  ([crates/j2k-metal/examples/resident_encode_buffer.rs](crates/j2k-metal/examples/resident_encode_buffer.rs))
- `cargo run -p j2k-ml --example training_batcher --features cpu`
  ([crates/j2k-ml/examples/training_batcher.rs](crates/j2k-ml/examples/training_batcher.rs))
- `cargo run -p j2k-ml --example cuda_upload --features cuda`
  ([crates/j2k-ml/examples/cuda_upload.rs](crates/j2k-ml/examples/cuda_upload.rs))
- `cargo run -p j2k-ml --example metal_upload --features metal`
  ([crates/j2k-ml/examples/metal_upload.rs](crates/j2k-ml/examples/metal_upload.rs))
- `cargo run -p j2k-mpsgraph --example resident_reference_graph`
  ([crates/j2k-mpsgraph/examples/resident_reference_graph.rs](crates/j2k-mpsgraph/examples/resident_reference_graph.rs))
- `cargo run -p j2k-tilecodec --example decompress`
  ([crates/j2k-tilecodec/examples/decompress.rs](crates/j2k-tilecodec/examples/decompress.rs))

JPEG 2000 callers that need reduced regions below the shared
`Downscale::Eighth` ceiling can use
`J2kDecoder::decode_region_scaled_pow2_into`. The level is an exact count of
power-of-two halvings; requests beyond any component's codestream resolution
ladder return an unsupported error instead of silently decoding at another
scale.

Runtime backend selection defaults to `Auto`: CPU remains the portable baseline,
and Metal or CUDA paths are selected only for supported shapes with validation
and benchmark evidence. Lossless HTJ2K host-output encode uses the qualified
Metal hybrid cells above and stays CPU for other measured shapes. Full-resident
Metal-buffer encode remains a separate batch API and evidence class. Explicit
device requests are strict. Unsupported device shapes return errors instead of
silently changing the requested backend. `Auto` is an optimization policy, not
a promise to use a device whenever one is available.

A new fixed hybrid threshold is eligible for `Auto` only when identical-output
external-corpus Criterion evidence shows a median at least 10% faster than CPU
and any supported strict-device route, with non-overlapping 95% confidence
intervals. The policy never calibrates at runtime. Explicit `Cuda` and `Metal`
requests remain strict, and an accelerator failure after `Auto` selects a device
is an error rather than a silent CPU retry.

CUDA paths use J2K-owned CUDA Oxide device kernels through `cuda-runtime`.
NVIDIA performance claims require self-hosted benchmark evidence; hosted CI is
not treated as NVIDIA performance evidence.

## High-throughput owned batches

The additive owned-batch API accepts `EncodedImage` values containing an
`Arc<[u8]>` and one of `Full`, `Region`, `Reduced`, or `RegionReduced`. It
prepares inputs concurrently, keeps unlike output shapes in separate groups
without padding, and returns source indices, indexed preparation failures, and
homogeneous group execution failures. Representable Gray, RGB, and RGBA groups
use exact native `U8`, `U16`, or `I16` storage in NCHW or NHWC order; float
conversion and normalization are deliberately not codec operations.

Preparation can retain either a `PreparedHtj2kPlan` or a
`PreparedClassicPlan` with per-tile packet, code-block, and destination
geometry. Both plans reference compressed payload ranges inside the original
`Arc<[u8]>`; neither duplicates the codestream. `CpuBatchDecoder` consumes
single- and multi-tile plans without reparsing. Inputs outside the retained-plan
boundary can remain metadata-only and use the broader CPU codec when that path
supports them.

`j2k-cuda` and `j2k-metal` own persistent accelerator sessions, resident
output, and validated caller-owned destinations. Their direct final stores
produce the requested native dtype and layout in the destination allocation,
so decoded pixels do not make a GPU-to-CPU-to-GPU round trip. The codec-wide
support boundary is maintained in
[docs/public-support.md](docs/public-support.md); the exact experimental Burn
adapter boundary is maintained in [docs/j2k-ml.md](docs/j2k-ml.md). Dated
hardware validation and performance results live only in
[docs/benchmark-evidence.md](docs/benchmark-evidence.md).

`j2k-ml` stages completed codec output through host memory and creates ordinary
Burn tensors with public APIs. Its `CudaUploadBurnDecoder` and
`MetalUploadBurnDecoder` still execute decoding on the named accelerator, but
they do not claim a direct Burn destination or zero-copy handoff. Readers such
as `wsi-rs` remain responsible for finding and supplying encoded image bytes.
Codec support and correctness do not by themselves constitute a speedup claim.

`j2k-mpsgraph` is the Apple Silicon direct path. It aliases completed resident
batches or queues decode and MPSGraph on one Metal command queue without an
application-level decoded-pixel GPU→CPU→GPU round trip. It does not claim
framework-internal zero-copy or a speedup; see
[docs/j2k-mpsgraph.md](docs/j2k-mpsgraph.md).

## Which crate should I use?

Use `cargo add j2k` for JPEG 2000 / HTJ2K application code. Lower-level
`j2k-*` crates remain public implementation and integration crates.

Use lower-level crates only when you need a specific integration point:

| Need | Crate |
| --- | --- |
| JPEG 2000 / HTJ2K inspect, decode, encode, and recode | `j2k` |
| Shared traits and backend types | `j2k-core` |
| Shared encode-stage contracts | `j2k-types` |
| Shared codec constants and pure helper algorithms | `j2k-codec-math` |
| JPEG inspect/decode and portable baseline encode | `j2k-jpeg` |
| Native JPEG 2000 and HTJ2K codec engine | `j2k-native` |
| JPEG-to-HTJ2K coefficient-domain transcode | `j2k-transcode` |
| CUDA adapters | `j2k-jpeg-cuda`, `j2k-cuda`, `j2k-transcode-cuda` |
| Metal adapters | `j2k-jpeg-metal`, `j2k-metal`, `j2k-transcode-metal` |
| Burn 0.21 native integer batch adapter | `j2k-ml` |
| Direct Apple Silicon MPSGraph batch adapter | `j2k-mpsgraph` |
| Tile compression codecs | `j2k-tilecodec` |
| Command-line inspection and JPEG-to-HTJ2K smoke transcode | `j2k-cli` |

## Support and evidence

The living codec support matrix is
[docs/public-support.md](docs/public-support.md). The Burn batch
adapter has a narrower, explicit boundary in
[docs/j2k-ml.md](docs/j2k-ml.md). Hardware measurements and their publication
qualifications are recorded separately in
[docs/benchmark-evidence.md](docs/benchmark-evidence.md). Release-scoped Part 1
and development Part 15 decoder conformance evidence is tracked separately in
[docs/t803-conformance.md](docs/t803-conformance.md).

The previous `j2k-ml 0.7.5` accelerator features were defective. That release
history is retained in the [release policy](docs/release.md); current CUDA and
Metal adapters use released dependency APIs and are validated as clean
packaged consumers before publication.

## Current backend posture

CPU is the correctness baseline. `BackendRequest::Auto` may return CPU-backed
outputs when a device path is unavailable, unsupported, or not benchmarked for
the requested shape.

GPU routing is intentionally selective. A Metal or CUDA path should be enabled
automatically only when the shape is supported, parity-covered, large or
regular enough to amortize dispatch and transfer costs, and backed by benchmark
evidence. Small tiles, irregular packet shapes, entropy-heavy stages, and
codestream assembly should remain CPU unless a measured resident path shows a
net win.

Metal adapters are macOS-only and experimental. Explicit Metal requests return
resident Metal surfaces or encode-stage dispatches only for supported adapter
paths. Metal encode support is not a blanket end-to-end guarantee for every
public encode route; unsupported explicit Metal shapes fail clearly.

CUDA adapters require a CUDA driver and adapter support. CUDA device memory
surfaces are available for supported paths; unsupported explicit CUDA requests
fail clearly. J2K-owned CUDA kernels are used for CUDA codec stages. NVIDIA
performance claims require recorded self-hosted benchmark output.

Lossy HTJ2K encoding can opt into the OpenHTJ2K-compatible visual Qfactor
profile with `J2kLossyEncodeOptions::with_qfactor(Some(quality))`, where
`quality` is `1..=100`. This profile is intentionally separate from byte,
bits-per-pixel, PSNR, quality-layer-target, and ROI controls.

## Public API and support policy

Stable APIs are `j2k`, `j2k-core` traits and value types, `j2k-jpeg`,
and `j2k-tilecodec`. Experimental APIs are the Metal adapters, CUDA adapters,
transcode crates, and backend encode-stage adapter SPI.

Codec contracts include `ImageDecode`, `decode_region_scaled_into`,
`decode_rows`, `TileBatchDecode`, `DeviceSurface`, `ScratchPool`, and
the concrete `J2kContext` and `j2k_jpeg::DecoderContext` types.
Bounded JPEG 2000/HTJ2K row decode through 24-bit component precision retains
one parsed tile graph for the operation and reuses it across stripes; stripe
output scratch remains bounded by `J2kRowDecodeOptions`. Higher-precision exact
integer output keeps the existing full-decode/crop compatibility path.
`BackendRequest::Auto` may return CPU output.
`BackendRequest::Metal` and `BackendRequest::Cuda` are strict and fail for
unsupported shapes.

Container and storage integrations should pass compatible compressed payloads
through when the payload kind, dimensions, component count, bit depth,
signedness, and color interpretation already match the destination
requirements. Decode and re-encode only when passthrough is invalid and the
source codec path is supported.

Unsupported input must fail explicitly. Error messages must avoid sensitive
internal details. Unsafe Rust inventory is tracked in
[docs/unsafe-audit.md](docs/unsafe-audit.md). Fuzzing and malformed-input tests
are part of release hardening. MSRV is declared in the root manifest.

Reference files:

- [docs/architecture.md](docs/architecture.md) - workspace layer rules and crate
  dependency graph
- [docs/benchmark-evidence.md](docs/benchmark-evidence.md) - reproducible
  benchmark commands and current CUDA/Metal evidence
- [docs/benchmark-corpora.md](docs/benchmark-corpora.md) - external corpus and
  adoption-benchmark manifest policy
- [docs/env-vars.md](docs/env-vars.md) - supported `J2K_*`
  environment variables
- [docs/public-support.md](docs/public-support.md) - exact J2K Part 1,
  HTJ2K Part 15, JP2/JPH, and out-of-scope support boundary
- [docs/t803-conformance.md](docs/t803-conformance.md) - release-scoped and
  development T.803 v3
  decoder claims, encoder procedure, blockers, and release evidence rules
- [docs/j2k-ml.md](docs/j2k-ml.md) - Burn native integer batch groups,
  prepared reuse, and explicit accelerator decode/upload adapters
- [docs/j2k-mpsgraph.md](docs/j2k-mpsgraph.md) - direct Apple Silicon
  completed-buffer, pipelined, and nonblocking MPSGraph integration
- [docs/release.md](docs/release.md) - release and package validation policy
- [docs/stable-api-1.0.md](docs/stable-api-1.0.md) - stable API snapshot policy
- [CHANGELOG.md](CHANGELOG.md) - current release notes

## Benchmark and parity policy

Benchmark publication requirements are maintained in
[docs/benchmark-corpora.md](docs/benchmark-corpora.md), with current run
evidence in [docs/benchmark-evidence.md](docs/benchmark-evidence.md).
Use `cargo run -p xtask --features adoption -- adoption-benchmark` for
publication bundles and
`cargo run -p xtask --features adoption -- adoption-report --run-dir <run-dir>`
for the guarded report.
OpenJPEG/Grok/CUDA/Metal/Kakadu/OpenJPH claims must use the required comparator
or hardware gates described in the benchmark docs; skipped rows and emulated
rows are diagnostic evidence only.

## Security

Report vulnerabilities according to [SECURITY.md](SECURITY.md). Codec errors
should be explicit, non-sensitive, and should not silently treat unsupported
input as successful decode.

## License

Dual-licensed under either [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option.
