# Changelog

All notable changes to this project will be documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-04-19

### Added

- `slidecodec-core` shared trait/type crate:
  - `ImageDecode`, `ImageDecodeRows`, `TileBatchDecode`, `TileDecompress`
  - `PixelFormat`, `Downscale`, `Info`, `Rect`, `DecodeOutcome`
  - `ScratchPool` and `DecoderContext` contracts
- `slidecodec-jpeg` as the WSI-oriented JPEG implementation with:
  - borrowed parse/decode surfaces
  - row-streaming decode
  - region and scaled decode
  - tile-batch/context/scratch reuse
  - external-corpus and parity coverage
- `slidecodec-j2k` with:
  - JP2 / raw codestream inspect
  - full-frame, region, scaled, row-streaming, and tile-batch decode
  - HTJ2K coverage
  - OpenJPEG differential tests and compare bench
- `slidecodec-tilecodec` with:
  - `DeflateCodec`
  - `ZstdCodec`
  - `LzwCodec`
  - `UncompressedCodec`
  - typed scratch pools and compare bench coverage
- `slidecodec-cli` inspect dispatch for JPEG and JPEG 2000 inputs
- workspace-level CI coverage for tests, clippy, bench build, fuzz-target
  build, and `cargo deny`

### Changed

- Workspace version promoted to `1.0.0`
- Benchmark documentation now covers JPEG, JPEG 2000, and tile decompression
- Top-level README now documents the full pathology codec stack instead of the
  original JPEG-only scope
