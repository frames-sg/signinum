---
title: 'slidecodec: WSI-shaped JPEG and JPEG 2000 codecs for digital pathology'
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
date: 27 April 2026
bibliography: paper.bib
---

# Summary

`slidecodec` is a Rust codec workspace for whole-slide imaging (WSI), the
high-resolution microscopy format used in digital pathology. WSI applications
rarely decode one image once. Viewers, quality-control tools, and analysis
pipelines repeatedly decode many small tiles, regions, or reduced-resolution
views while users pan, zoom, and inspect tissue. `slidecodec` provides codec
primitives shaped for those workloads: JPEG inspection and decode, JPEG 2000 /
HTJ2K inspection and decode, restart-marker and coded-unit metadata, region of
interest (ROI) decode, decode-time downscale, row streaming, tile-batch decode,
caller-owned scratch buffers, and optional Apple Metal device-output adapters.

The workspace separates codec work from slide-container work. `slidecodec`
does not parse SVS, NDPI, DICOM, Mirax, Zeiss, or other WSI containers; that
responsibility belongs to readers such as `wsi-rs` [@wsirs]. Instead,
`slidecodec` turns compressed tile bytes into CPU pixels or device-resident
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
Rust WSI stack. `slidecodec` fills this gap by providing WSI-oriented codec
APIs with Apache-2.0 licensing and Rust-native integration.

# State of the field

`slidecodec` is not a replacement for OpenSlide; it is a lower-level codec
component that a reader can use to compete with or validate against OpenSlide.
It is also not intended to replace libjpeg-turbo for general JPEG decoding,
or Kakadu, Grok, OpenJPEG, and OpenHTJ2K for every JPEG 2000 deployment.
Those projects remain important comparators and, in some settings, better
choices.

The reason to build `slidecodec` rather than only contribute wrappers around
existing libraries is the combination of requirements: WSI-shaped ROI and
downscale APIs, restart-marker inspection, caller-owned state, Rust ownership
semantics, optional device-output adapters, and a small integration surface
for readers that already manage container parsing, cache policy, and viewport
prefetch. These choices let a Rust reader hand compressed tile bytes to the
codec without adopting a monolithic slide runtime or copying pixels through
intermediate image abstractions that are not needed by the caller.

# Software design

The workspace is layered around `slidecodec-core`, which defines shared pixel
formats, backend requests, rectangles, row sinks, scratch/context contracts,
and decode traits. Codec crates implement those traits for JPEG, JPEG 2000 /
HTJ2K, and tile-compression primitives. Adapter crates add platform-specific
device surfaces without forcing GPU dependencies into the CPU codecs.

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
groups documented in `docs/bench.md` and `docs/parity.md`. The JPEG 2000
native engine is kept under `#![forbid(unsafe_code)]`; unavoidable `unsafe`
in the public workspace is isolated to CPU feature detection and audited
JPEG hot paths such as SIMD and low-level entropy-buffer handling.

# Research impact statement

`slidecodec` is already integrated as the production codec layer for `wsi-rs`,
a Rust WSI reader used by SlideViewer. That integration exercises the API on
real slide workloads: SVS, NDPI, DICOM WSI [@dicomwsi], Zeiss, Mirax,
Hamamatsu VMS, and Philips TIFF readers resolve compressed tile bytes and pass
decode work to `slidecodec`. The companion SlideViewer parity harness compares
reader output against compatibility oracles, including OpenSlide, while the
`slidecodec` benchmark harness compares codec tasks against libjpeg-turbo,
`zune-jpeg`, `jpeg-decoder`, OpenJPEG, and Grok.

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
