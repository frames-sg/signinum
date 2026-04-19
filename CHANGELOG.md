# Changelog

All notable changes to this project will be documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Full scalar baseline decode pipeline (M1b):
  - `Decoder::new(input)` constructs a decoder ready for pixel decode.
  - `Decoder::decode_into(out, stride, fmt)` decodes SOF0 and SOF1 8-bit
    sequential JPEGs to `OutputFormat::{Rgb8, Rgba8, Gray8}`.
  - Chroma sampling: 4:4:4, 4:2:2, 4:2:0 with libjpeg-turbo fancy upsample.
  - Restart markers handled at MCU interval boundaries.
  - Ground-truth ISLOW integer IDCT in `idct::scalar`.
- Public `DecodeOutcome { decoded: Rect, warnings: Vec<Warning> }`.
- `JpegView::parse`, `Decoder::from_view`, and `Decoder::decode_rows` for the
  prepared-parse and row-streaming API slice.
- Internal prepared decode state now lives in an owned decode plan, and the
  baseline sequential path decodes through reusable MCU-row stripe buffers
  instead of full-image component planes.
- `JpegError::NotImplemented { sof }` for parseable SOFs that land in M3
  (Extended12, Progressive, Lossless) — transient variant removed in M3.
  `JpegError::is_not_implemented()` predicate for routing.
- First bit-exact libjpeg-turbo parity fixtures:
  `corpus/conformance/baseline_420_16x16.{jpg,rgb}` and
  `corpus/conformance/grayscale_8x8.{jpg,gray}`, with `manifest.json` recording
  the libjpeg-turbo version and regeneration via `generate.sh`.
- `cargo-fuzz` target `decode_fuzz` covering `Decoder::new + decode_into`.
- Native-only crate groundwork: supported targets are now `x86_64` and
  `aarch64`, and CI no longer carries wasm / `no-default-features` jobs for
  `slidecodec-jpeg`.
- Comparator benches now skip impractical multi-gigapixel full-frame decodes
  and add a `decode_rows_rgb` benchmark group for large extracted WSI JPEGs.

### Not yet (tracked)

- `OutputFormat::RawYCbCr8` — M2.
- `Decoder::decode_region_into`, `decode_downscaled_into`, `segments()`,
  `TableCache`, `DecoderBuilder` — M2.
- SOF1 12-bit, SOF2 progressive, SOF3 lossless — M3.
- SIMD IDCT and color convert (NEON / AVX2 / SSE4.1 / simd128) — M4.

### M1a (previously published under Unreleased)

- Workspace, CI (fmt, clippy, test × stable/beta/MSRV × linux/macos, wasm32,
  cargo-deny), licensing, and module skeleton (M0).
- Public API surface for header parsing: `Decoder::inspect(bytes) -> Info`
  (M1a). Supports SOF0 baseline, SOF1 extended 8/12-bit, SOF2 progressive
  (headers only), SOF3 lossless (headers only). Rejects arithmetic-coded
  and hierarchical variants with `JpegError::UnsupportedSof`.
- Typed error and warning enums: `JpegError`, `Warning`, `MarkerKind`,
  `UnsupportedReason`, `HuffmanFailure`, `BuilderConflictReason`, `TableKind`.
- Property-based test suite (`proptest`, 4096 cases) and `cargo-fuzz`
  `parse_fuzz` target covering `Decoder::inspect`.
- `slidecodec inspect <file>` CLI subcommand.
- `inspect` example in `examples/`.
