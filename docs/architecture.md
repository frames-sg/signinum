# Architecture

Primary design references now live in-repo:

- `specs/2026-04-19-slidecodec-umbrella-design.md`
- `specs/2026-04-19-j2k-m1-scalar-baseline-design.md`
- `specs/2026-04-19-j2k-m3-ht-design.md`
- `specs/2026-04-19-m8-tilecodec-design.md`

Current implementation notes:

- `slidecodec-core` owns the shared codec traits, backend capability metadata,
  device-surface traits, and generic decoder context contracts.
- `slidecodec-jpeg` and `slidecodec-j2k` remain CPU-first crates with WSI
  decode-time ROI, reduced-resolution decode, scratch-pool reuse, and
  tile-batch APIs.
- `slidecodec-jpeg-metal` and `slidecodec-j2k-metal` are additive Apple-host
  device-output adapters. Their current v1 path keeps parse/entropy and other
  codec-control-heavy stages on CPU, then hands decoded component rows or
  planes to Metal compute kernels for color conversion, interleave/pack, and
  final `MTLBuffer` production.
- `slidecodec-jpeg-cuda` and `slidecodec-j2k-cuda` mirror the device-output
  API surface and explicit backend selection semantics, but on non-CUDA hosts
  they validate fallback and unavailability behavior rather than runtime GPU
  execution.
