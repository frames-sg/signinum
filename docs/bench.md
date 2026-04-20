# Benchmarking methodology

`slidecodec-jpeg` carries two Criterion benchmark targets:

- `compare` compares `slidecodec-jpeg` against `jpeg-decoder` and `zune-jpeg`
  on the same JPEG byte streams, and also carries a slidecodec-only
  `decode_rows_rgb` group for large WSI-oriented inputs.
- `micro` measures slidecodec-only hot paths that are useful when tuning
  regressions inside the crate: header inspect, Huffman symbol decode, and
  scalar IDCT.

The current benchmark contract is native-only: run and compare on `x86_64`
and `aarch64` hosts. wasm and `no_std` builds are no longer part of the
performance signoff path.

## Compared operations

- `inspect`
  - `slidecodec-jpeg`: `Decoder::inspect`
  - `jpeg-decoder`: `Decoder::read_info`
  - `zune-jpeg`: `JpegDecoder::decode_headers`
- `decode_rgb`
  - `slidecodec-jpeg`: `Decoder::new` + `decode_into(PixelFormat::Rgb8)`
  - `jpeg-decoder`: `Decoder::decode`
  - `zune-jpeg`: `JpegDecoder::decode` with RGB output
- `decode_gray`
  - `slidecodec-jpeg`: `Decoder::new` + `decode_into(PixelFormat::Gray8)`
  - `jpeg-decoder`: `Decoder::decode`
  - `zune-jpeg`: `JpegDecoder::decode` with Luma output
- `decode_rows_rgb`
  - `slidecodec-jpeg`: `Decoder::new` + `decode_rows` into a `RowSink<u8>`
  - no cross-crate comparator; this exists for very large WSI JPEGs where
    full-frame output buffers are not representative of the intended API
- `wsi_tile_batch_rgb`
  - `slidecodec-jpeg`: repeated `Decoder::decode_tile` with a shared
    `DecoderContext` and `ScratchPool`
  - no cross-crate comparator; this is the parse+decode tile-batch path that
    WSI readers actually use, including shared table-cache reuse
- `wsi_region_rgb`
  - `slidecodec-jpeg`: `Decoder::decode_region_into(..., PixelFormat::Rgb8, roi)`
  - `jpeg-decoder` / `zune-jpeg`: full RGB decode, then crop the centered
    `256×256` region in memory
- `wsi_scaled_rgb_q4`
  - `slidecodec-jpeg`: `decode_scaled(PixelFormat::Rgb8, Downscale::Quarter)`
  - `jpeg-decoder` / `zune-jpeg`: full RGB decode, then spatially decimate by
    `4×` in memory
- `wsi_scaled_rgb_q8`
  - `slidecodec-jpeg`: `decode_scaled(PixelFormat::Rgb8, Downscale::Eighth)`
  - `jpeg-decoder` / `zune-jpeg`: full RGB decode, then spatially decimate by
    `8×` in memory
- `wsi_region_scaled_rgb_q4`
  - `slidecodec-jpeg`: `decode_region_scaled(PixelFormat::Rgb8, roi, Downscale::Quarter)`
  - `jpeg-decoder` / `zune-jpeg`: full RGB decode, crop the centered
    `256×256` region, then spatially decimate by `4×`
- `wsi_region_scaled_rgb_q8`
  - `slidecodec-jpeg`: `decode_region_scaled(PixelFormat::Rgb8, roi, Downscale::Eighth)`
  - `jpeg-decoder` / `zune-jpeg`: full RGB decode, crop the centered
    `256×256` region, then spatially decimate by `8×`
- `wsi_tile_batch_scaled_rgb_q4`
  - `slidecodec-jpeg`: repeated `decode_tile_scaled_into_in_context(..., PixelFormat::Rgb8, Downscale::Quarter)`
    with shared `DecoderContext`, shared `ScratchPool`, and one reused output
    buffer
  - `jpeg-decoder` / `zune-jpeg`: repeated fresh decode followed by in-memory
    `4×` decimation per tile
- `wsi_tile_batch_region_scaled_rgb_q4`
  - `slidecodec-jpeg`: repeated
    `decode_tile_region_scaled_into_in_context(..., PixelFormat::Rgb8, roi, Downscale::Quarter)`
    with shared `DecoderContext`, shared `ScratchPool`, and one reused output
    buffer
  - `jpeg-decoder` / `zune-jpeg`: repeated fresh decode, centered
    `256×256` crop, then `4×` in-memory decimation per tile
- `decode_reused_rgb` / `decode_reused_gray`
  - `slidecodec-jpeg`: `Decoder::new` once per input, then `decode_into` into a
    pre-allocated buffer for every iteration — isolates pure decode cost from
    `Decoder::new` and output allocation.
  - No cross-crate comparator: neither `zune-jpeg` nor `jpeg-decoder` exposes a
    reusable decoder, so the fair comparison against them stays in
    `decode_rgb` / `decode_gray`. This group is the primary signal for the
    WSI tile-batch workload (Phase 3+ scratch-pool path).

`jpeg-decoder` is benchmarked with `default-features = false` so the comparison
stays single-threaded and does not fold Rayon scheduling into the baseline.

For WSI signoff, the primary performance surface is the reduced-output and
tile-batch groups:

- `wsi_scaled_*`
- `wsi_region_scaled_*`
- `wsi_tile_batch_*`

Those are the workloads that exploit slidecodec's structural advantages:
DCT-domain downscale, decode-time crop, and shared decode-state reuse across a
tile stream. Fresh full-frame `decode_rgb` remains a useful generic JPEG
comparison, but it is not the decisive WSI-viewer workload.

## Inputs

The benchmark harness always includes the committed conformance fixtures:

- `corpus/conformance/baseline_420_16x16.jpg`
- `corpus/conformance/grayscale_8x8.jpg`

Optional local inputs are discovered from `SLIDECODEC_BENCH_INPUTS`. The value
is parsed with the platform path separator, so it may contain one or more files
or directories:

```sh
SLIDECODEC_BENCH_INPUTS=/path/to/jpeg_dir:/path/to/extracted_wsi_tiles cargo bench -p slidecodec-jpeg --bench compare
```

Discovery rules:

- recurse through directories
- accept only `.jpg` / `.jpeg`
- keep only files that `slidecodec_jpeg::Decoder::new` can decode today
- classify grayscale vs RGB using slidecodec header info so each file lands in
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
`SLIDECODEC_BENCH_INPUTS` at the extracted tile directories.

The optional external corpus regression test uses the same `BoundedFullFrame` /
`VeryLarge` classification. It still routes every `VeryLarge` JPEG through
`Decoder::decode_rows`, including grayscale inputs, because that test is
checking practical local-corpus coverage rather than mirroring the benchmark
group names exactly.

The WSI-native groups (`wsi_region_*`, `wsi_scaled_*`, `wsi_tile_batch_*`)
intentionally compare complete viewer tasks rather than identical library APIs.
`slidecodec-jpeg` performs crop/downscale during decode and can reuse shared
decode state across a tile batch; the comparator crates do the equivalent work
after a full decode because they do not expose ROI, DCT-domain reduced output,
or shared table/scratch reuse surfaces.

## Commands

Compile-only check:

```sh
cargo bench -p slidecodec-jpeg --no-run
```

Run the comparator benches:

```sh
cargo bench -p slidecodec-jpeg --bench compare
```

Run the slidecodec-only microbenches:

```sh
cargo bench -p slidecodec-jpeg --bench micro
```

Run against the local SlideViewer corpus:

```sh
SLIDECODEC_BENCH_INPUTS=/Users/user/Bench/SlideViewer/downloads/openslide-testdata-extracted/hamamatsu-vms/hamamatsu-vms-cmu1 \
  cargo bench -p slidecodec-jpeg --bench compare
```

## `slidecodec-j2k`

`slidecodec-j2k` carries a dedicated Criterion comparator bench at
`crates/slidecodec-j2k/benches/compare.rs`.

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

- `slidecodec-j2k` is benchmarked through its public API
- OpenJPEG is benchmarked through the local `opj_decompress` binary
- classic J2K bench inputs are generated through the local `opj_compress`
  binary when available, so the OpenJPEG comparator always consumes its own
  valid JP2 output
- the binaries are discovered from `SLIDECODEC_OPENJPEG_BIN` and
  `SLIDECODEC_OPENJPEG_COMPRESS_BIN`, otherwise
  `/opt/homebrew/bin/opj_decompress` and `/opt/homebrew/bin/opj_compress`
  are used when present
- the OpenJPEG path is an end-to-end tool comparison, not an in-process FFI
  microbenchmark

Region and scale mapping:

- region decode uses `opj_decompress -d x0,y0,x1,y1`
- scaled decode uses `opj_decompress -r <reduce factor>`
- tile-batch decode is modeled as repeated decode invocations on the same tile

Compile the J2K compare bench:

```sh
cargo bench -p slidecodec-j2k --bench compare --no-run
```

Run it locally against OpenJPEG:

```sh
SLIDECODEC_OPENJPEG_BIN=/opt/homebrew/bin/opj_decompress \
SLIDECODEC_OPENJPEG_COMPRESS_BIN=/opt/homebrew/bin/opj_compress \
  cargo bench -p slidecodec-j2k --bench compare
```

## `slidecodec-tilecodec`

`slidecodec-tilecodec` carries a Criterion comparator bench at
`crates/slidecodec-tilecodec/benches/compare.rs`.

It benchmarks four decompression paths:

- `DeflateCodec`
- `ZstdCodec`
- `LzwCodec`
- `UncompressedCodec`

Bench group:

- `decompress_into`

Comparator policy:

- `slidecodec-tilecodec` is benchmarked through the public `TileDecompress`
  implementations with reusable typed pools
- Deflate is compared against direct `flate2` decode using the same zlib-backed
  implementation family
- Zstd is compared against direct `zstd` stream decode
- LZW is compared against direct `weezl` decode
- Uncompressed is compared against a plain `memcpy`

Compile the tilecodec compare bench:

```sh
cargo bench -p slidecodec-tilecodec --bench compare --no-run
```

Run it locally:

```sh
cargo bench -p slidecodec-tilecodec --bench compare
```

## Recorded baselines

All numbers on `aarch64-apple-darwin`, Criterion `--quick`, committed
fixtures only. Bigger inputs (local WSI tiles via `SLIDECODEC_BENCH_INPUTS`)
are not stored here — rerun locally to capture them per commit.

Pre-Phase-1 baseline (commit `9678d7d`, scalar-only decoder, audit snapshot):

| group | input | slidecodec | jpeg-decoder | zune-jpeg |
|---|---|---|---|---|
| `decode_rgb` | `baseline_420_16x16` | 7.88 µs | 5.52 µs | 3.56 µs |
| `decode_gray` | `grayscale_8x8` | 2.89 µs | 1.68 µs | 1.26 µs |
| `decode_reused_rgb` | `baseline_420_16x16` | 1.19 µs | — | — |
| `decode_reused_gray` | `grayscale_8x8` | 0.22 µs | — | — |
| `micro/idct_reference_block` | — | 42 ns | — | — |
| `micro/upsample_h2v2_fancy_rows_128` | — | 473 ns | — | — |
| `micro/ycbcr_to_rgb_row_scalar_256` | — | 94 ns | — | — |
| `micro/huffman_luma_dc_zero_stream` (2048 syms) | — | 1.19 µs | — | — |

Slidecodec is ~2.2× slower than zune on the fresh-mode groups at this
commit. The reused groups show the ceiling: per-tile `Decoder::new` and
output allocation eat 6–13× of the work on small fixtures, which Phase 3
eliminates entirely.

Post-Phase-1 snapshot (NEON SIMD ISLOW IDCT, same aarch64-apple-darwin):

| group | input | slidecodec | Δ vs pre-Phase-1 |
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

| group | input | slidecodec | zune-jpeg | slidecodec advantage |
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
host) runs `cargo bench -p slidecodec-jpeg --bench micro`; expected ratio
is ≥2× over scalar, matching NEON.

## Policy

- Benchmark results are report-only for now; CI compiles the benches but does
  not fail on runtime performance deltas.
- libjpeg-turbo remains the parity oracle. The comparison harness is for speed
  and coarse behavior comparisons against Rust decoders, not for declaring a
  new correctness oracle.
