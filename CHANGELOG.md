# Changelog

All notable changes to this project will be documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-05-01

CPU-first 1.0 release posture.

### Changed

- Promoted `signinum-core`, `signinum-jpeg`, `signinum-j2k`, `signinum-tilecodec`,
  and `signinum-cli` to the stable CPU-first 1.0 release set.
- Kept `signinum-j2k-native` as a published pre-1.0 implementation dependency
  for `signinum-j2k`.
- Excluded Metal, CUDA, and comparator crates from the 1.0 publish workflow.
- Clarified that CUDA crates can use `cuda-runtime` to return CUDA device memory
  surfaces by uploading CPU-decoded bytes, with no CUDA kernel decode or
  NVIDIA performance claim.

## [0.1.0] - 2026-04-25

Initial public-source checkpoint. The workspace remains pre-1.0 while the
JPEG 2000 / HTJ2K ROI and GPU adapter APIs settle.

### Added

- `signinum-core` shared trait/type crate:
  - `ImageDecode`, `ImageDecodeRows`, `TileBatchDecode`, `TileDecompress`
  - `PixelFormat`, `Downscale`, `Info`, `Rect`, `DecodeOutcome`
  - `ScratchPool` and `DecoderContext` contracts
- `signinum-jpeg` as the WSI-oriented JPEG implementation with:
  - borrowed parse/decode surfaces
  - row-streaming decode
  - region and scaled decode
  - tile-batch/context/scratch reuse
  - external-corpus and parity coverage
- `signinum-j2k` with:
  - JP2 / raw codestream inspect
  - full-frame, region, scaled, row-streaming, and tile-batch decode
  - HTJ2K coverage
  - OpenJPEG differential tests and compare bench
- `signinum-tilecodec` with:
  - `DeflateCodec`
  - `ZstdCodec`
  - `LzwCodec`
  - `UncompressedCodec`
  - typed scratch pools and compare bench coverage
- `signinum-cli` inspect dispatch for JPEG and JPEG 2000 inputs
- workspace-level CI coverage for tests, clippy, bench build, fuzz-target
  build, and `cargo deny`

### Changed

- Workspace version promoted to `0.1.0`
- Benchmark documentation now covers JPEG, JPEG 2000, and tile decompression
- Top-level README now documents the full pathology codec stack instead of the
  original JPEG-only scope
