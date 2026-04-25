---
title: 'slidecodec: A Rust JPEG decoder and memory-safe HTJ2K codec stack for whole-slide imaging with optional Apple Metal GPU acceleration'
tags:
  - Rust
  - whole-slide imaging
  - digital pathology
  - JPEG
  - JPEG 2000
  - HTJ2K
  - GPU
  - Apple Metal
authors:
  - name: <AUTHOR NAME>
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: <AFFILIATION>
    index: 1
date: 24 April 2026
bibliography: paper.bib
---

# Summary

`slidecodec` is a Rust codec workspace for whole-slide imaging (WSI)
in digital pathology. It provides a baseline JPEG decoder and a JPEG
2000 / HTJ2K encoder-decoder together with tile decompression primitives
(Deflate, Zstd, LZW, uncompressed), exposed through API surfaces shaped
for WSI access patterns: pixel-exact region-of-interest (ROI) decode,
decode-time downscale, tile-batch streaming, row streaming, and
caller-owned scratch and decoder contexts. Optional device-output
adapters expose decoded surfaces to downstream GPU pipelines: the Apple
Metal adapter implements inverse discrete wavelet transform (IDWT),
HTJ2K cleanup, multi-component transform (MCT), color conversion, and
output store as Metal compute kernels, while CUDA-facing crates currently
provide fallback-only API compatibility and explicit unavailable
semantics. The `slidecodec-j2k-native` codec is implemented under
`#![forbid(unsafe_code)]`, providing a fully memory-safe HTJ2K codec
suitable for ingesting untrusted slide bytes in clinical and research
contexts. The JPEG decoder uses tightly scoped `unsafe` only in
architecture-specific SIMD backends. `slidecodec` is the codec
foundation of `wsi-rs`
[@wsirs], an OpenSlide-compatible [@goode2013openslide] pure-Rust
whole-slide reader covering Aperio, Ventana, Trestle, and DICOM WSI
[@dicomwsi] containers.

# Statement of need

Whole-slide imaging viewers, segmentation pipelines, and analysis
frameworks decode JPEG and JPEG 2000 / HTJ2K [@iso15444_15; @taubman2019fbcot]
tiles continuously: a typical viewport pan decodes 64+ tiles at the
current pyramid level, often as ROI extracts at decode-time downscale,
and renders within an interactive latency budget. No existing
open-source codec covers this workload completely.

The HTJ2K codec landscape in particular has a structural gap. Grok
[@grok] is licensed under AGPL, restricting commercial pathology
integration. Kakadu [@kakadu] is proprietary and license-encumbered.
OpenJPEG [@openjpeg] is C with a documented history of memory-safety
CVEs and incomplete HTJ2K coverage. OpenHTJ2K [@openhtj2k] is
research-grade C++ without WSI-shaped APIs. There is no permissively
licensed, memory-safe HTJ2K encoder or decoder in the open-source
ecosystem prior to this work.

The JPEG side has high-quality decoders — libjpeg-turbo
[@libjpegturbo], `zune-jpeg` [@zunejpeg], and `jpeg-decoder`
[@jpegdecoder] — but none expose the API primitives WSI workloads
require: pixel-exact ROI decode (rather than MCU-aligned crop),
decode-time downscale composed with ROI, tile-batch submission with
shared scratch, row-streaming output for stripes and large tiles,
caller-owned decoder contexts free of global state, and borrowed
parse and decode surfaces. WSI viewers must currently choose between
a fast general-purpose decoder and a WSI-shaped wrapper that pays the
cost of full-image decode followed by host-side cropping.

`slidecodec` fills both gaps with a shared API surface across both
codec families. JPEG encoding is intentionally out of scope; downstream
callers can pair `slidecodec-jpeg` with existing Rust encoders when they
need JPEG write support. Optional Apple Metal device-output adapters extend the
same primitives with hybrid CPU-entropy / GPU-IDWT-and-color-convert
pipelines, exploiting Apple Silicon's unified memory model to keep
per-tile coefficient handoff at zero-copy cost — a design point not
practical on discrete GPUs that pay PCIe submission overhead per tile.

# Implementation

The workspace contains: `slidecodec-core` (shared traits, pixel and
sample types, scratch and context contracts); `slidecodec-jpeg` (Rust
JPEG decoder with NEON, x86, and scalar backends);
`slidecodec-j2k` and the internal `slidecodec-j2k-native` engine
(pure Rust JPEG 2000 / HTJ2K encoder and decoder under
`forbid(unsafe_code)`, using `fearless_simd` for portable SIMD);
`slidecodec-jpeg-metal` and `slidecodec-j2k-metal` (Apple Metal
device-output adapters with dedicated compute kernels);
`slidecodec-jpeg-cuda` and `slidecodec-j2k-cuda` (CUDA-facing crates
with fallback-only compatibility semantics in this release);
`slidecodec-tilecodec` (Deflate, Zstd, LZW, uncompressed tile
decompression); and `slidecodec-cli` (inspection entry point). A
session-level adaptive router selects CPU or GPU per request based on
image size, batch size, and ROI shape, with shared input interning
(`Arc<[u8]>`) and queued tile-batch submission.

The SIMD strategy differs between the codec families for historical and
technical reasons. The JPEG decoder's hand-written NEON and x86
intrinsics predate the JPEG 2000 engine integration and remain the
fastest validated path for its fused entropy, IDCT, upsample, and color
conversion loops. The JPEG 2000 / HTJ2K engine uses `fearless_simd` to
keep the codec crate under `forbid(unsafe_code)` while retaining portable
SIMD acceleration. Migrating the JPEG SIMD layer to a shared portable
abstraction is a post-stabilization maintenance goal, contingent on
matching the current architecture-specific performance.

A reproducible benchmark harness in `docs/bench.md` compares
`slidecodec` against libjpeg-turbo (system-linked via `pkg-config`),
`zune-jpeg`, `jpeg-decoder`, OpenJPEG, and Grok across full-image,
ROI, scaled, and tile-batch workloads. Performance gains are
workload-shaped: pixel-exact ROI and tile-batch decode on 4:2:0 JPEGs
achieve substantial speedup over libjpeg-turbo with native
MCU-aligned cropping on Apple Silicon, while full-image decode is
competitive but not faster. JPEG 2000 / HTJ2K Metal tile-batch decode
on distinct inputs achieves multi-fold speedup over pure-CPU decode
at 1024-class tile sizes. Single-tile decode below routing thresholds
is correctly directed to CPU.

# Limitations and roadmap

The CUDA crates are not runtime CUDA implementations in this release.
They keep the device-output API shape compilable for downstream callers,
exercise CPU fallback surfaces, and return explicit unavailable errors
for `BackendRequest::Cuda`. Full CUDA decode kernels and NVIDIA runtime
benchmarks are future work.

The JPEG crate is decode-only. JPEG encoding is outside the present
scope, while JPEG 2000 / HTJ2K encode and decode are included through
`slidecodec-j2k-native`. Additional roadmap items include AVX2 direct
emission for JPEG ROI paths, x86_64 GPU benchmark coverage, continued
tuning of adaptive CPU/GPU routing thresholds, and a maintainability
split of the large fused JPEG sequential entropy path once the current
performance-sensitive loop structure has stable regression coverage.

# Acknowledgments

The authors thank the maintainers of OpenJPEG, Grok, libjpeg-turbo,
`zune-jpeg`, and `jpeg-decoder` whose comparator implementations
shaped the benchmark harness, and the OpenSlide and DICOM WG-26
communities whose interoperability standards motivate this work.

# References
