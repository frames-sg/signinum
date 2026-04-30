# ashlar

Pathology codec stack for whole-slide imaging workloads.

## Status

`ashlar` is a native-first codec workspace for pathology and WSI software.
The current public-source version is `0.1.0`. The project is in active API
stabilization toward 1.0, with the WSI-shaped CPU decode APIs as the primary
supported surface and GPU adapter APIs still being hardened.
The core stack in this repository is:

- `ashlar-jpeg` — native JPEG decode for WSI tiles
- `ashlar-jpeg-metal` — Apple Metal JPEG tile decode and device-output
  adapter for batched WSI workloads
- `ashlar-jpeg-cuda` — CUDA-facing JPEG device-output API adapter; today it
  validates explicit CUDA-unavailable behavior and CPU-backed `Auto`/`Cpu`
  surfaces
- `ashlar-j2k` — native in-repo JPEG 2000 / HTJ2K inspect and decode;
  WSI-native ROI/context optimization milestones are still in progress, so the
  workspace remains pre-1.0
- `ashlar-j2k-metal` — Apple Metal device-output adapter for JPEG 2000 /
  HTJ2K tiles
- `ashlar-j2k-cuda` — CUDA-facing JPEG 2000 / HTJ2K device-output API
  adapter; today it validates explicit CUDA-unavailable behavior and CPU-backed
  `Auto`/`Cpu` surfaces
- `ashlar-tilecodec` — tile decompression primitives for Deflate, Zstd,
  LZW, and Uncompressed payloads
- `ashlar-core` — shared traits, pixel/sample types, scratch/context
  contracts
- `ashlar-cli` — CLI inspection entry point

Target decode hosts are native `x86_64` and `aarch64`.
Metal device-output adapters are validated on Apple Silicon macOS. CUDA crates
are source-published as fallback-only API compatibility adapters in this
checkpoint; no runtime CUDA implementation or NVIDIA performance claim is made
for `0.1.0`.

## Stabilization roadmap

Before 1.0, the project is focused on:

- completing JPEG 2000 / HTJ2K ROI and reduced-resolution performance work
- tightening public API documentation for the WSI decode surfaces
- promoting the GPU adapter APIs from compatibility surfaces to validated
  multi-host implementations
- adding x86_64 GPU benchmark coverage and broadening release CI

## What this is

`ashlar` is designed around WSI access patterns instead of generic
consumer-image decode:

- borrowed parse/decode surfaces
- caller-owned scratch pools and decoder contexts
- decode-time ROI and reduced-resolution output
- row-bounded output for large tiles and stripes
- tile-batch APIs for repeated viewer workloads
- additive device-output adapters for Metal and CUDA consumers
- explicit separation between image codecs and tile decompression codecs

The project is structured so WSI readers can compose their own threading,
vendor/container parsing, pyramid policy, caching, and prefetch around codec
primitives instead of paying for a monolithic runtime.

## Current scope

### `ashlar-jpeg`

- Baseline JPEG support already present in the crate
- ROI, scaled decode, row streaming, and tile-batch decode APIs
- WSI-focused benchmarking against `jpeg-decoder`, `zune-jpeg`, and direct
  `libjpeg-turbo` decode paths
- Metal and CUDA adapter crates keep the core JPEG decoder pure-Rust CPU while
  exposing device-output surfaces for downstream GPU pipelines; the Metal path
  has optimized kernel paths for supported baseline JPEG tile shapes, including
  batched 4:2:0 and 4:2:2 RGB WSI tile decode, while the CUDA crate is
  fallback-only in this checkpoint

### `ashlar-j2k`

- JP2 / raw codestream inspect
- full-frame, region, scaled, row-bounded, and tile-batch decode
- repo-local pure-Rust JPEG 2000 / HTJ2K decode engine
- native ROI/context/performance rewrite still in progress
- parity and benchmark coverage against Grok and OpenJPEG on CPU
- Metal and CUDA adapter crates expose device-output surfaces without moving
  the core decoder crate onto GPU-specific dependencies; the Metal path now
  runs compute kernels for component-plane interleave/clamp/pack after CPU
  decode, with ROI staging still performed on CPU today, while the CUDA crate
  is fallback-only in this checkpoint

### `ashlar-tilecodec`

- `DeflateCodec`
- `ZstdCodec`
- `LzwCodec`
- `UncompressedCodec`

These codecs expose the shared `TileDecompress` trait from `ashlar-core`.

## Quick start

JPEG inspect:

```rust
use ashlar_jpeg::{Decoder, JpegView};

let bytes = std::fs::read("tile.jpg")?;
let info = Decoder::inspect(&bytes)?;
println!(
    "{:?} {:?} mcu={:?} restart={:?}",
    info.dimensions,
    info.color_space,
    info.mcu_geometry,
    info.restart_interval
);
if let Some(index) = JpegView::parse(&bytes)?.restart_index()? {
    println!("restart segments={}", index.segments.len());
}
```

JPEG 2000 decode:

```rust
use ashlar_core::{Downscale, PixelFormat};
use ashlar_j2k::J2kDecoder;

let bytes = std::fs::read("tile.jp2")?;
let mut decoder = J2kDecoder::new(&bytes)?;
let (w, h) = decoder.info().dimensions;
let mut rgb = vec![0_u8; (w * h * 3) as usize];
decoder.decode_scaled_into(
    &mut ashlar_j2k::J2kScratchPool::new(),
    &mut rgb,
    (w * 3) as usize,
    PixelFormat::Rgb8,
    Downscale::None,
)?;
```

JPEG 2000 / HTJ2K Metal tile batch:

```rust
use ashlar_core::{BackendRequest, PixelFormat};
use ashlar_j2k_metal::MetalTileBatch;

let tile_bytes: Vec<Vec<u8>> = load_visible_j2k_tiles()?;
let mut batch = MetalTileBatch::with_capacity(tile_bytes.len());

for tile in &tile_bytes {
    batch.push_tile(tile, PixelFormat::Gray8, BackendRequest::Metal)?;
}

let surfaces = batch.decode_all()?;
```

WSI readers should own vendor parsing, pyramid levels, tile coordinates,
caching, prefetch, and viewport policy. The Metal adapters only batch codec
requests and return decoded surfaces in submission order. If a caller already
stores compressed tile payloads in `Arc<[u8]>`, the `push_shared_*` methods can
queue them without another tile-byte copy. Use explicit `BackendRequest::Metal`
when a batched caller wants Metal execution; `BackendRequest::Auto` remains
conservative for small or host-returned decodes.

Tile decompression:

```rust
use ashlar_core::TileDecompress;
use ashlar_tilecodec::{DeflateCodec, DeflatePool};

let compressed = std::fs::read("tile.deflate")?;
let mut pool = DeflatePool::new();
let mut out = vec![0_u8; 1 << 20];
let written = DeflateCodec::decompress_into(&mut pool, &compressed, &mut out)?;
println!("decoded {} bytes", written);
```

CLI inspect:

```sh
$ ashlar inspect tile.jp2
1024×1024 Srgb bit=8 comps=3 levels=6 tiles=Some(...)
```

Runnable crate examples are available under:

- `crates/ashlar-jpeg/examples`
- `crates/ashlar-j2k/examples`
- `crates/ashlar-tilecodec/examples`

## Benchmarks

Benchmark methodology and comparator policy live in [docs/bench.md](docs/bench.md).
The repo now carries dedicated compare benches for:

- `ashlar-jpeg`
- `ashlar-j2k`
- `ashlar-jpeg-metal`
- `ashlar-j2k-metal`
- `ashlar-tilecodec`

Release staging notes live in [docs/release.md](docs/release.md).

## License

Apache-2.0. See `LICENSE-APACHE`.

## MSRV

Rust 1.94.
