# Benchmarking methodology

`ashlar-jpeg` carries two Criterion benchmark targets:

- `compare` compares `ashlar-jpeg` against `jpeg-decoder`, `zune-jpeg`,
  and direct `libjpeg-turbo` decode paths on the same JPEG byte streams, and
  also carries a ashlar-only `decode_rows_rgb` group for large
  WSI-oriented inputs.
- `micro` measures ashlar-only hot paths that are useful when tuning
  regressions inside the crate: header inspect, Huffman symbol decode, and
  scalar IDCT.

The current benchmark contract is native-only: run and compare on `x86_64`
and `aarch64` hosts. wasm and `no_std` builds are no longer part of the
performance signoff path.

## Host setup

The in-tree correctness tests do not require system codec libraries. Comparator
benchmark rows are optional and are enabled only when their local dependency can
be discovered.

On macOS with Homebrew:

```sh
brew install pkg-config jpeg-turbo openjpeg
```

On Ubuntu/Debian:

```sh
sudo apt-get update
sudo apt-get install -y pkg-config libturbojpeg0-dev libjpeg-dev openjpeg-tools
```

JPEG comparator behavior:

- `libjpeg-turbo` is discovered with `pkg-config --libs libturbojpeg libjpeg`.
- If it is not available, the `libjpeg-turbo` benchmark rows are skipped.
- Set `ASHLAR_REQUIRE_LIBJPEG_TURBO=1` on signoff hosts to fail when the
  direct comparator is missing.

JPEG 2000 comparator behavior:

- OpenJPEG in-process comparator code is provided by the Rust `openjpeg-sys`
  dependency.
- The optional `opj_compress` binary is used only for generating one
  OpenJPEG-shaped fixture path when it is present; otherwise the bench falls
  back to the in-tree encoder for deterministic inputs.
- Grok rows are skipped unless `ASHLAR_GROK_SOURCE` and
  `ASHLAR_GROK_ROOT` point to a local Grok build with headers and shared
  libraries.

## Compared operations

- `inspect`
  - `ashlar-jpeg`: `Decoder::inspect`
  - `jpeg-decoder`: `Decoder::read_info`
  - `zune-jpeg`: `JpegDecoder::decode_headers`
  - `libjpeg-turbo`: reused TurboJPEG handle + `tj3DecompressHeader`
- `decode_rgb`
  - `ashlar-jpeg`: `Decoder::new` + `decode_into(PixelFormat::Rgb8)`
  - `jpeg-decoder`: `Decoder::decode`
  - `zune-jpeg`: `JpegDecoder::decode` with RGB output
  - `libjpeg-turbo`: reused TurboJPEG handle + `tj3Decompress8(..., TJPF_RGB)`
- `decode_gray`
  - `ashlar-jpeg`: `Decoder::new` + `decode_into(PixelFormat::Gray8)`
  - `jpeg-decoder`: `Decoder::decode`
  - `zune-jpeg`: `JpegDecoder::decode` with Luma output
  - `libjpeg-turbo`: reused TurboJPEG handle + `tj3Decompress8(..., TJPF_GRAY)`
- `decode_rows_rgb`
  - `ashlar-jpeg`: `Decoder::new` + `decode_rows` into a `RowSink<u8>`
  - no cross-crate comparator; TurboJPEG is a packed-output API, so this
    remains a ashlar-only streaming benchmark for very large WSI JPEGs
- `wsi_tile_batch_rgb`
  - `ashlar-jpeg`: repeated `Decoder::decode_tile` with a shared
    `DecoderContext` and `ScratchPool`
  - `libjpeg-turbo`: repeated `tj3Decompress8(..., TJPF_RGB)` with one reused
    TurboJPEG handle across the batch
- `wsi_region_rgb`
  - `ashlar-jpeg`: `Decoder::decode_region_into(..., PixelFormat::Rgb8, roi)`
  - `jpeg-decoder` / `zune-jpeg`: full RGB decode, then crop the centered
    `256×256` region in memory
  - `libjpeg-turbo`: TurboJPEG cropped decode, aligning the left crop boundary
    to the scaled iMCU width and trimming the over-read columns in Rust
- `wsi_scaled_rgb_q4`
  - `ashlar-jpeg`: `decode_scaled(PixelFormat::Rgb8, Downscale::Quarter)`
  - `jpeg-decoder` / `zune-jpeg`: full RGB decode, then spatially decimate by
    `4×` in memory
  - `libjpeg-turbo`: reused TurboJPEG handle + `tj3SetScalingFactor(1/4)`
- `wsi_scaled_rgb_q8`
  - `ashlar-jpeg`: `decode_scaled(PixelFormat::Rgb8, Downscale::Eighth)`
  - `jpeg-decoder` / `zune-jpeg`: full RGB decode, then spatially decimate by
    `8×` in memory
  - `libjpeg-turbo`: reused TurboJPEG handle + `tj3SetScalingFactor(1/8)`
- `wsi_region_scaled_rgb_q4`
  - `ashlar-jpeg`: `decode_region_scaled(PixelFormat::Rgb8, roi, Downscale::Quarter)`
  - `jpeg-decoder` / `zune-jpeg`: full RGB decode, crop the centered
    `256×256` region, then spatially decimate by `4×`
  - `libjpeg-turbo`: `tj3SetScalingFactor(1/4)` + cropped decode + left-edge
    trim when the ROI is not scaled-iMCU aligned
- `wsi_region_scaled_rgb_q8`
  - `ashlar-jpeg`: `decode_region_scaled(PixelFormat::Rgb8, roi, Downscale::Eighth)`
  - `jpeg-decoder` / `zune-jpeg`: full RGB decode, crop the centered
    `256×256` region, then spatially decimate by `8×`
  - `libjpeg-turbo`: `tj3SetScalingFactor(1/8)` + cropped decode + left-edge
    trim when the ROI is not scaled-iMCU aligned
- `wsi_tile_batch_scaled_rgb_q4`
  - `ashlar-jpeg`: repeated `decode_tile_scaled_into_in_context(..., PixelFormat::Rgb8, Downscale::Quarter)`
    with shared `DecoderContext`, shared `ScratchPool`, and one reused output
    buffer
  - `jpeg-decoder` / `zune-jpeg`: repeated fresh decode followed by in-memory
    `4×` decimation per tile
  - `libjpeg-turbo`: repeated scaled TurboJPEG decode with one reused handle
- `wsi_tile_batch_region_scaled_coalesced_rgb_q4`
  - `ashlar-jpeg`: repeated
    `decode_tile_region_scaled_into_in_context(..., PixelFormat::Rgb8, roi, Downscale::Quarter)`
    with shared `DecoderContext`, shared `ScratchPool`, and one reused output
    buffer; the Metal adapter queues 64 identical region+scaled requests, so
    the device path reports a `coalesce_hits_98p4pct` request hit rate by
    construction
- `wsi_tile_batch_region_scaled_distinct_rgb_q4`
  - `ashlar-jpeg`: the same region+scaled tile-batch CPU path over 64
    different RGB JPEG byte streams from one compatible input directory
  - `ashlar-jpeg-metal`: queued `BatchOp::RegionScaled` over those 64
    distinct byte streams; this is the cold-pan WSI viewport case and should
    report `coalesce_hits_0p0pct`
  - `jpeg-decoder` / `zune-jpeg`: repeated fresh decode, centered
    `256×256` crop, then `4×` in-memory decimation per tile
  - `libjpeg-turbo`: repeated scaled cropped decode with one reused handle
- `decode_reused_rgb` / `decode_reused_gray`
  - `ashlar-jpeg`: `Decoder::new` once per input, then `decode_into` into a
    pre-allocated buffer for every iteration — isolates pure decode cost from
    `Decoder::new` and output allocation.
  - No cross-crate comparator: neither `zune-jpeg` nor `jpeg-decoder` exposes a
    reusable decoder, so the fair comparison against them stays in
    `decode_rgb` / `decode_gray`. This group is the primary signal for the
    WSI tile-batch workload (Phase 3+ scratch-pool path).

`jpeg-decoder` is benchmarked with `default-features = false` so the comparison
stays single-threaded and does not fold Rayon scheduling into the baseline.
`libjpeg-turbo` is discovered through `pkg-config` at build time; if the local
machine does not expose both `libturbojpeg` and `libjpeg`, the `libjpeg-turbo`
rows are omitted from the compare bench. Set
`ASHLAR_REQUIRE_LIBJPEG_TURBO=1` when running the direct comparator test on
a signoff host to fail loudly instead of silently skipping the native path.

For WSI signoff, the primary performance surface is the reduced-output and
tile-batch groups:

- `wsi_scaled_*`
- `wsi_region_scaled_*`
- `wsi_tile_batch_*`

Those are the workloads that exploit ashlar's structural advantages:
DCT-domain downscale, decode-time crop, and shared decode-state reuse across a
tile stream. Fresh full-frame `decode_rgb` remains a useful generic JPEG
comparison, but it is not the decisive WSI-viewer workload.

## CPU-first JPEG proving policy

- Apple Silicon CPU (`aarch64/NEON`) is the first proving host for JPEG
  optimization work.
- The acceptance groups for CPU-first JPEG work are:
  - `decode_rows_rgb`
  - `wsi_region_rgb`
  - `wsi_scaled_rgb_q4`
  - `wsi_scaled_rgb_q8`
  - `wsi_region_scaled_rgb_q4`
  - `wsi_region_scaled_rgb_q8`
  - `wsi_tile_batch_rgb`
  - `wsi_tile_batch_scaled_rgb_q4`
  - `wsi_tile_batch_region_scaled_coalesced_rgb_q4`
  - `wsi_tile_batch_region_scaled_distinct_rgb_q4`
- Tiny committed fixtures remain useful for `micro_*` and correctness
  regression only; they are not valid evidence for WSI performance claims.

## Inputs

The benchmark harness always includes the committed conformance fixtures:

- `corpus/conformance/baseline_420_16x16.jpg`
- `corpus/conformance/grayscale_8x8.jpg`

Optional local inputs are discovered from `ASHLAR_BENCH_INPUTS`. The value
is parsed with the platform path separator, so it may contain one or more files
or directories:

```sh
ASHLAR_BENCH_INPUTS=/path/to/jpeg_dir:/path/to/extracted_wsi_tiles cargo bench -p ashlar-jpeg --bench compare
```

Discovery rules:

- recurse through directories
- accept only `.jpg` / `.jpeg`
- keep only files that `ashlar_jpeg::Decoder::new` can decode today
- classify grayscale vs RGB using ashlar header info so each file lands in
  the matching benchmark group
- classify each file by estimated full-frame output bytes:
  `width * height * bytes_per_pixel`
- `BoundedFullFrame` means the estimated full-frame output is `<= 512 MiB`
- `VeryLarge` means the estimate exceeds `512 MiB` or overflows the size math
- comparator `decode_rgb` / `decode_gray` benches include only
  `BoundedFullFrame` files
- `decode_rows_rgb` includes only `VeryLarge` RGB files; it does not duplicate
  the bounded full-frame cases

Whole-slide containers such as `.svs` or `.ndpi` are intentionally not decoded
directly by this harness. Extract JPEG tiles first, then point
`ASHLAR_BENCH_INPUTS` at the extracted tile directories.

The optional external corpus regression test uses the same `BoundedFullFrame` /
`VeryLarge` classification. It still routes every `VeryLarge` JPEG through
`Decoder::decode_rows`, including grayscale inputs, because that test is
checking practical local-corpus coverage rather than mirroring the benchmark
group names exactly.

The WSI-native groups (`wsi_region_*`, `wsi_scaled_*`, `wsi_tile_batch_*`)
intentionally compare complete viewer tasks rather than identical library APIs.
`ashlar-jpeg` performs crop/downscale during decode and can reuse shared
decode state across a tile batch; the comparator crates do the equivalent work
after a full decode because they do not expose ROI, DCT-domain reduced output,
or shared table/scratch reuse surfaces.

## Commands

Compile-only check:

```sh
cargo bench -p ashlar-jpeg --no-run
```

Run the comparator benches:

```sh
cargo bench -p ashlar-jpeg --bench compare
```

Run the ashlar-only microbenches:

```sh
cargo bench -p ashlar-jpeg --bench micro
```

Run against the local SlideViewer corpus:

```sh
ASHLAR_BENCH_INPUTS=/Users/user/Bench/SlideViewer/downloads/openslide-testdata-extracted/hamamatsu-vms/hamamatsu-vms-cmu1 \
  cargo bench -p ashlar-jpeg --bench compare
```

## `ashlar-j2k`

`ashlar-j2k` carries a dedicated Criterion comparator bench at
`crates/ashlar-j2k/benches/compare.rs`.

It uses deterministic runtime-generated codestreams so the bench is always
available without a checked-in J2K corpus:

- classic grayscale J2K
- classic RGB J2K
- HTJ2K grayscale

Bench groups:

- `inspect`
- `decode_gray`
- `decode_rgb`
- `wsi_region_gray`
- `wsi_scaled_gray_q4`
- `wsi_tile_batch_gray`

Comparator policy:

- `ashlar-j2k` is benchmarked through its public API
- OpenJPEG is benchmarked in-process through `openjpeg-sys`
- Grok is benchmarked in-process through the local `libgrokj2k` shared library
  plus a thin C shim compiled into `ashlar-j2k-compare`
- all three decoders produce packed `Gray8` or interleaved `Rgb8` output so
  output-layout work is included equally in the timing
- the OpenJPEG and Grok comparator paths are forced single-threaded
- classic J2K bench inputs are generated through the local `opj_compress`
  binary when available, so the RGB JP2 fixture path matches the OpenJPEG tool
  chain; otherwise the bench falls back to the in-tree encoder path
- `opj_compress` is discovered from `ASHLAR_OPENJPEG_COMPRESS_BIN`,
  otherwise `/opt/homebrew/bin/opj_compress` is used when present
- Grok library discovery is controlled by `ASHLAR_GROK_SOURCE` and
  `ASHLAR_GROK_ROOT`; by default the bench looks for a local Grok build at
  `/tmp/grok-ashlar` with shared libraries under `/tmp/grok-ashlar/build/bin`

Region and scale mapping:

- region decode uses the native OpenJPEG decode-area API and Grok region fields
- scaled decode uses native OpenJPEG reduction-factor decode and Grok reduction
  decode
- tile-batch decode is modeled as repeated decode invocations on the same tile

Compile the J2K compare bench:

```sh
cargo bench -p ashlar-j2k --bench compare --no-run
```

Run it locally against in-process OpenJPEG:

```sh
ASHLAR_OPENJPEG_COMPRESS_BIN=/opt/homebrew/bin/opj_compress \
  cargo bench -p ashlar-j2k --bench compare
```

Run it locally against OpenJPEG and Grok:

```sh
ASHLAR_OPENJPEG_COMPRESS_BIN=/opt/homebrew/bin/opj_compress \
ASHLAR_GROK_SOURCE=/tmp/grok-ashlar \
ASHLAR_GROK_ROOT=/tmp/grok-ashlar/build/bin \
  cargo bench -p ashlar-j2k --bench compare
```

## Device-output adapters

`ashlar-jpeg-metal` and `ashlar-j2k-metal` carry Apple-host device
benches that compare the CPU decode path against the corresponding
Metal-surface path.

Current v1 scope is explicit:

- JPEG: supported baseline WSI tile shapes can run Metal kernel paths for full,
  region, scaled, region+scaled, and batched RGB device-output decode;
  compatible queued region+scaled requests use a real `BatchOp::RegionScaled`
  path. The coalesced benchmark intentionally queues 64 identical requests and
  can collapse them to one immutable Metal surface; the distinct benchmark
  queues 64 different JPEG byte streams so it measures cold-pan batch
  throughput instead of duplicate-input reuse. Unsupported shapes fall back
  through CPU decode plus device-surface upload according to the requested
  backend
- J2K: CPU codestream/decode still reconstructs component planes, then Metal
  compute performs interleave/clamp/pack into the requested surface; ROI
  staging is still done on CPU for the current region path
- these benches measure complete codec-device tasks, including surface
  production; they do not include WSI container parsing, tile lookup, caching,
  or prefetch policy

`ashlar-jpeg-metal` compare bench names:

- `decode_rgb`
- `wsi_tile_batch_rgb`
- `wsi_region_rgb`
- `wsi_scaled_rgb_q4`
- `wsi_scaled_rgb_q8`
- `wsi_region_scaled_rgb_q4`
- `wsi_region_scaled_rgb_q8`
- `wsi_tile_batch_scaled_rgb_q4`
- `wsi_tile_batch_region_scaled_coalesced_rgb_q4`
- `wsi_tile_batch_region_scaled_distinct_rgb_q4`
- viewer/composite groups for contiguous and sparse viewport-shaped device
  output

Compile the Metal benches:

```sh
cargo bench -p ashlar-jpeg-metal --bench compare --no-run
cargo bench -p ashlar-jpeg-metal --bench device_upload --no-run
cargo bench -p ashlar-j2k-metal --bench device_upload --no-run
```

Run them on Apple Silicon macOS:

```sh
cargo bench -p ashlar-jpeg-metal --bench compare -- --noplot
cargo bench -p ashlar-jpeg-metal --bench device_upload -- --noplot
cargo bench -p ashlar-j2k-metal --bench device_upload -- --noplot
```

`ashlar-jpeg-cuda` and `ashlar-j2k-cuda` expose the same device-output
API surface, but on this Apple host they are verification stubs:

- `BackendRequest::Cpu` returns a host-backed surface
- `BackendRequest::Auto` falls back to the CPU surface
- `BackendRequest::Cuda` fails explicitly as unavailable

No CUDA runtime performance numbers are claimed from this machine.

## `ashlar-tilecodec`

`ashlar-tilecodec` carries a Criterion comparator bench at
`crates/ashlar-tilecodec/benches/compare.rs`.

It benchmarks four decompression paths:

- `DeflateCodec`
- `ZstdCodec`
- `LzwCodec`
- `UncompressedCodec`

Bench group:

- `decompress_into`

Comparator policy:

- `ashlar-tilecodec` is benchmarked through the public `TileDecompress`
  implementations with reusable typed pools
- Deflate is compared against direct `flate2` decode using the same zlib-backed
  implementation family
- Zstd is compared against direct `zstd` stream decode
- LZW is compared against direct `weezl` decode
- Uncompressed is compared against a plain `memcpy`

Compile the tilecodec compare bench:

```sh
cargo bench -p ashlar-tilecodec --bench compare --no-run
```

Run it locally:

```sh
cargo bench -p ashlar-tilecodec --bench compare
```

## Recorded baselines

All numbers on `aarch64-apple-darwin`, Criterion `--quick`, committed
fixtures only. Bigger inputs (local WSI tiles via `ASHLAR_BENCH_INPUTS`)
are not stored here — rerun locally to capture them per commit.

Pre-Phase-1 baseline (commit `9678d7d`, scalar-only decoder, audit snapshot):

| group | input | ashlar | jpeg-decoder | zune-jpeg |
|---|---|---|---|---|
| `decode_rgb` | `baseline_420_16x16` | 7.88 µs | 5.52 µs | 3.56 µs |
| `decode_gray` | `grayscale_8x8` | 2.89 µs | 1.68 µs | 1.26 µs |
| `decode_reused_rgb` | `baseline_420_16x16` | 1.19 µs | — | — |
| `decode_reused_gray` | `grayscale_8x8` | 0.22 µs | — | — |
| `micro/idct_reference_block` | — | 42 ns | — | — |
| `micro/upsample_h2v2_fancy_rows_128` | — | 473 ns | — | — |
| `micro/ycbcr_to_rgb_row_scalar_256` | — | 94 ns | — | — |
| `micro/huffman_luma_dc_zero_stream` (2048 syms) | — | 1.19 µs | — | — |

Ashlar is ~2.2× slower than zune on the fresh-mode groups at this
commit. The reused groups show the ceiling: per-tile `Decoder::new` and
output allocation eat 6–13× of the work on small fixtures, which Phase 3
eliminates entirely.

Post-Phase-1 snapshot (NEON SIMD ISLOW IDCT, same aarch64-apple-darwin):

| group | input | ashlar | Δ vs pre-Phase-1 |
|---|---|---|---|
| `decode_rgb` | `baseline_420_16x16` | 7.77 µs | -1.4% |
| `decode_gray` | `grayscale_8x8` | 2.92 µs | ~0 |
| `decode_reused_rgb` | `baseline_420_16x16` | 1.10 µs | -7.6% |
| `decode_reused_gray` | `grayscale_8x8` | 0.20 µs | -10.4% |
| `micro/idct_islow_scalar_block` | — | 38 ns | — |
| `micro/idct_islow_neon_block` | — | 18 ns | **-53%** (2.1× faster) |

Fresh-mode barely moves because `Decoder::new` dominates the 16×16 fixture
(IDCT is 8–10% of the work). On real WSI tiles (256×256+) where hundreds
of IDCTs run per decode, the 2.1× kernel win will compound proportionally.

Post-Phase-3 snapshot (ScratchPool + Phase 1 NEON IDCT):

| group | input | ashlar | zune-jpeg | ashlar advantage |
|---|---|---|---|---|
| `decode_scratch_rgb` | `baseline_420_16x16` | **920 ns** | 3614 ns (fresh-mode) | **3.9× faster** |
| `decode_scratch_gray` | `grayscale_8x8` | **75 ns** | 1253 ns (fresh-mode) | **16.7× faster** |
| `decode_reused_rgb` | `baseline_420_16x16` | 1235 ns | — | — |
| `decode_reused_gray` | `grayscale_8x8` | 260 ns | — | — |

The `decode_scratch_*` groups reuse both the `Decoder` and a
pre-allocated `ScratchPool` — the realistic WSI tile-batch shape. The
comparison against zune's fresh-mode decode isn't apples-to-apples on
allocation strategy (zune has no reusable decoder) but it does reflect
the real integration cost a WSI reader would pay. For a tile-batch
reader of 1000 tiles, we issue 1000 `decode_into_with_scratch` calls,
each at the scratch-group speed; zune's reader pays the fresh-mode
`JpegDecoder::new` + internal allocation on every tile.

Phase 4 partial (AVX2 ISLOW IDCT on x86_64):

AVX2 IDCT is wired in `src/idct/avx2.rs` using 128-bit SSE4.1 intrinsics
in the same 4-lane i32 structure as NEON. Coverage is validated by the
same proptest harness under `#[cfg(target_arch = "x86_64")]` plus
hand-picked edges. x86_64 runtime numbers land when CI (or a local Linux
host) runs `cargo bench -p ashlar-jpeg --bench micro`; expected ratio
is ≥2× over scalar, matching NEON.

## Policy

- Benchmark results are report-only for now; CI compiles the benches but does
  not fail on runtime performance deltas.
- libjpeg-turbo remains the primary JPEG parity oracle, and it is now also a
  direct speed comparator when available locally through `pkg-config`.
