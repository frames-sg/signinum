---
title: 'signinum: WSI-shaped JPEG and JPEG 2000 codecs for digital pathology'
tags:
  - Rust
  - whole-slide imaging
  - digital pathology
  - JPEG
  - JPEG 2000
  - HTJ2K
  - GPU
  - codec
authors:
  - name: Greg Furletti
    affiliation: 1
affiliations:
  - name: Independent researcher
    index: 1
date: 29 April 2026
bibliography: paper.bib
---

# Summary

`signinum` is a Rust codec workspace for whole-slide imaging (WSI), the
high-resolution microscopy format used in digital pathology. WSI applications
rarely decode one image once. Viewers, quality-control tools, and analysis
pipelines repeatedly decode many small tiles, regions, or reduced-resolution
views while users pan, zoom, and inspect tissue. `signinum` provides codec
primitives shaped for those workloads: JPEG inspection and decode, JPEG 2000 /
HTJ2K inspection and decode, restart-marker and coded-unit metadata, region of
interest (ROI) decode, decode-time downscale, row streaming, tile-batch decode,
caller-owned scratch buffers, tile-decompression codecs, and optional device
surface adapters. The current workspace includes CPU-first JPEG and JPEG 2000
crates, Apple Metal adapters validated on Apple Silicon, fallback-only CUDA API
adapters, Deflate/Zstd/LZW/Uncompressed tile decompression, a shared core crate,
and a CLI inspection entry point.

The workspace separates codec work from slide-container work. `signinum`
does not parse SVS, NDPI, DICOM, Mirax, Zeiss, or other WSI containers; that
responsibility belongs to readers such as `ziggurat` [@wsirs]. Instead,
`signinum` turns compressed tile bytes into CPU pixels or device-resident
surfaces and returns enough metadata for a reader to make correct tile and ROI
decisions.

# Statement of need

Digital pathology software depends on low-latency decoding of tiled JPEG,
JPEG 2000, and HTJ2K images [@iso15444_15; @taubman2019fbcot]. OpenSlide
[@goode2013openslide] established a widely used
vendor-neutral WSI reader, but its public interface is reader-oriented rather
than codec-oriented. Researchers building new Rust WSI readers, viewers, or
analysis pipelines still need reusable codec components that expose the
operations common in slide navigation: inspect without decode, decode only a
requested rectangle, downscale during decode, reuse scratch space across a
tile stream, and surface restart-marker or MCU geometry for formats such as
Hamamatsu NDPI.

General-purpose JPEG decoders such as libjpeg-turbo [@libjpegturbo],
`zune-jpeg` [@zunejpeg], and `jpeg-decoder` [@jpegdecoder] are mature and
valuable, but they do not provide a Rust API designed around WSI tile
scheduling and slide-reader ownership of caches, scratch buffers, and
coordinate systems. JPEG 2000 and HTJ2K add a different constraint: OpenJPEG
[@openjpeg], Grok [@grok], OpenHTJ2K [@openhtj2k], and Kakadu [@kakadu] cover
important parts of the ecosystem, but their licensing, language/runtime
assumptions, or API shapes are not always suitable for a permissively licensed
Rust WSI stack. `signinum` fills this gap by providing WSI-oriented codec
APIs with Apache-2.0 licensing and Rust-native integration.

# State of the field

`signinum` is not a replacement for OpenSlide; it is a lower-level codec
component that a reader can use to compete with or validate against OpenSlide.
It is also not intended to replace libjpeg-turbo for general JPEG decoding,
or Kakadu, Grok, OpenJPEG, and OpenHTJ2K for every JPEG 2000 deployment.
Those projects remain important comparators and, in some settings, better
choices.

The reason to build `signinum` rather than only contribute wrappers around
existing libraries is the combination of requirements: WSI-shaped ROI and
downscale APIs, restart-marker inspection, caller-owned state, Rust ownership
semantics, optional device-output adapters, and a small integration surface
for readers that already manage container parsing, cache policy, and viewport
prefetch. These choices let a Rust reader hand compressed tile bytes to the
codec without adopting a monolithic slide runtime or copying pixels through
intermediate image abstractions that are not needed by the caller.

# Software design

The workspace is layered around `signinum-core`, which defines shared pixel
formats, backend requests, rectangles, row sinks, scratch/context contracts,
and decode traits. Codec crates implement those traits for JPEG, JPEG 2000 /
HTJ2K, and tile-compression primitives. Adapter crates add platform-specific
device surfaces without forcing GPU dependencies into the CPU codecs. This
keeps reader integrations stable: `ziggurat` can submit compressed tile bytes and
choose CPU, automatic, or device-oriented output preferences without depending
on vendor-container details inside the codec crates.

This design makes two trade-offs explicit. First, the codec does not own
threading, slide pyramids, or caches. That keeps the API usable by WSI readers
with different I/O models, but it requires callers to be explicit about
scratch reuse and batch submission. Second, GPU support is additive rather
than mandatory. CPU decode is always available; Metal adapters are used only
for request shapes where device output is useful, while CUDA-facing crates
currently expose compatibility and fallback behavior rather than a runtime
CUDA implementation.

Correctness and maintainability are handled through parser-level inspection,
reference-comparator tests, fixture manifests, fuzz targets, and benchmark
groups documented in `docs/bench.md` and `docs/parity.md`. As of 29 April
2026, `cargo test --workspace --all-targets` passes across the CLI, core,
JPEG, JPEG Metal, JPEG CUDA, JPEG 2000, JPEG 2000 Metal, JPEG 2000 CUDA, and
tilecodec crates, including benchmark smoke targets. The same run includes
165 `signinum-jpeg` unit tests, 36 JPEG Metal tests, 75 native JPEG 2000 /
HTJ2K tests, 43 JPEG 2000 Metal tests, 10 tilecodec decompression tests, and
focused parity/regression tests against libjpeg-turbo, OpenJPEG, and Grok
where those comparator paths are available. The JPEG corpus report over the
committed conformance fixtures completed 11 rows with zero failures; those
fixtures are correctness smoke tests, not the basis for WSI-scale performance
claims.

The JPEG 2000 native engine is kept under `#![forbid(unsafe_code)]`;
unavoidable `unsafe` in the public workspace is isolated to CPU feature
detection and audited JPEG hot paths such as SIMD and low-level entropy-buffer
handling.

# Research impact statement

`signinum` is already integrated as the production codec layer for `ziggurat`,
a Rust WSI reader used by SlideViewer. That integration exercises the API on
real slide workloads: SVS, NDPI, DICOM WSI [@dicomwsi], Zeiss, Mirax,
Hamamatsu VMS, Leica, Ventana, and Philips TIFF readers resolve compressed
tile bytes and pass decode work to `signinum`. The companion SlideViewer
parity harness compares reader output against compatibility oracles, including
OpenSlide, while the `signinum` benchmark harness compares codec tasks
against libjpeg-turbo, `zune-jpeg`, `jpeg-decoder`, OpenJPEG, and Grok.

The current integration release gate passes on eight real local slides: two
NDPI slides, four JPEG-compressed SVS slides, and two JPEG 2000 SVS slides
from 94 MB to 2.5 GB. Representative signinum-backed reader medians include
NDPI B4 2k-region extraction at 45.8 ms versus 53.1 ms for OpenSlide and a
2.5 GB metastatic melanoma SVS 2k-region extraction at 20.3 ms versus 56.7 ms.
These are whole-reader measurements, so they are reported as integration
evidence rather than isolated codec microbenchmarks.

The near-term research use is reproducible WSI systems benchmarking:
measuring tile latency, ROI decode, reduced-resolution decode, and
device-output behavior separately from container parsing and viewer policy.
This separation is important for digital pathology methods papers, where the
codec contribution must be distinguishable from caching, I/O, and rendering
choices.

# AI usage disclosure

Generative AI assistance was used to draft and revise this paper text and to
check it against the JOSS paper-structure requirements. The software design,
implementation, tests, and benchmark claims are based on the repository
contents, local verification commands, and human review. The author is
responsible for final technical accuracy before submission.

# Acknowledgments

The author thanks the maintainers of OpenSlide, libjpeg-turbo, `zune-jpeg`,
`jpeg-decoder`, OpenJPEG, Grok, OpenHTJ2K, and Kakadu for the software and
documentation that make codec interoperability and benchmarking possible.

# References
