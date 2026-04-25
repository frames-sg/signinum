# Architecture

Primary design references now live in-repo:

- `docs/superpowers/specs/2026-04-19-slidecodec-umbrella-design.md`
- `docs/superpowers/specs/2026-04-19-j2k-m1-scalar-baseline-design.md`
- `docs/superpowers/specs/2026-04-19-j2k-m3-ht-design.md`
- `docs/superpowers/specs/2026-04-19-m8-tilecodec-design.md`

Current implementation notes:

- `slidecodec-core` owns the shared codec traits, backend capability metadata,
  device-surface traits, and generic decoder context contracts.
- `slidecodec-jpeg` and `slidecodec-j2k` remain CPU-first crates with WSI
  decode-time ROI, reduced-resolution decode, scratch-pool reuse, and
  tile-batch APIs.
- SIMD implementation strategy is intentionally codec-specific today:
  `slidecodec-jpeg` uses tightly scoped architecture intrinsics in its NEON and
  x86 backends because those fused JPEG hot paths predate the J2K integration
  and are performance-sensitive; `slidecodec-j2k-native` uses `fearless_simd`
  so the HTJ2K/JPEG 2000 engine can remain under `#![forbid(unsafe_code)]`.
- `slidecodec-jpeg/src/entropy/sequential.rs` is still a large fused
  sequential decoder module. It keeps entropy decode, IDCT scheduling,
  upsample, ROI, and fast 4:2:0 tile paths close together to avoid regressing
  WSI tile-batch performance. Splitting it by subsampling/backend is planned
  after the current fast paths have stable benchmark and parity coverage.
- `slidecodec-jpeg-metal` and `slidecodec-j2k-metal` are additive Apple-host
  device-output adapters. Their current v1 path keeps parse/entropy and other
  codec-control-heavy stages on CPU, then hands decoded component rows or
  planes to Metal compute kernels for color conversion, interleave/pack, and
  final `MTLBuffer` production.
- `slidecodec-jpeg-cuda` and `slidecodec-j2k-cuda` mirror the device-output
  API surface and explicit backend selection semantics, but on non-CUDA hosts
  they validate fallback and unavailability behavior rather than runtime GPU
  execution. The `0.1.0` release makes no CUDA performance claim.
