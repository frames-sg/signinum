# signinum

Pathology codec stack for whole-slide imaging workloads.

## Status

`signinum` is a native-first codec workspace for pathology and WSI software.
The current public-source release target is CPU-first 1.0: `signinum-core`,
`signinum-jpeg`, `signinum-j2k`, `signinum-tilecodec`, and `signinum-cli` are the
stable user-facing crates. GPU adapter crates remain pre-1.0 while their
runtime support and routing policies continue to harden.
The core stack in this repository is:

- `signinum-jpeg` — native JPEG decode for WSI tiles
- `signinum-jpeg-metal` — Apple Metal JPEG tile decode and device-output
  adapter for batched WSI workloads; pre-1.0
- `signinum-jpeg-cuda` — CUDA-facing JPEG device-output API adapter; explicit
  CUDA requests upload decoded output into CUDA device memory when the
  `cuda-runtime` feature and a CUDA driver are available; no CUDA kernel decode
  or NVIDIA performance claim is made yet
- `signinum-j2k` — native in-repo JPEG 2000 / HTJ2K inspect and decode;
  includes WSI-native ROI, reduced-resolution, and combined ROI+reduced-
  resolution decode surfaces
- `signinum-j2k-metal` — Apple Metal device-output adapter for JPEG 2000 /
  HTJ2K tiles; pre-1.0
- `signinum-j2k-cuda` — CUDA-facing JPEG 2000 / HTJ2K device-output API
  adapter; explicit CUDA requests upload decoded output into CUDA device memory
  when the `cuda-runtime` feature and a CUDA driver are available; no CUDA
  kernel decode or NVIDIA performance claim is made yet
- `signinum-tilecodec` — tile decompression primitives for Deflate, Zstd,
  LZW, and Uncompressed payloads
- `signinum-core` — shared traits, pixel/sample types, scratch/context
  contracts
- `signinum-cli` — CLI inspection entry point

Target decode hosts are native `x86_64` and `aarch64`.
CPU decode surfaces are the 1.0 compatibility promise. Metal device-output
adapters are validated on Apple Silicon macOS but stay on the post-1.0
hardening track. CUDA crates now provide explicit device-memory output through
the `cuda-runtime` feature when a CUDA driver is available, but decode bytes
are still CPU-produced and uploaded. There is no CUDA kernel decode and no
NVIDIA performance claim.

## Post-1.0 roadmap

After CPU-first 1.0, the project is focused on:

- hardening Metal adapter APIs and backend routing policies
- validating and recording Metal runtime benchmark baselines
- adding x86_64 GPU benchmark coverage
- hardening CUDA runtime surfaces toward kernel decode and benchmark coverage

## What this is

`signinum` is designed around WSI access patterns instead of generic
consumer-image decode:

- borrowed parse/decode surfaces
- caller-owned scratch pools and decoder contexts
- decode-time ROI and reduced-resolution output
- row-bounded output for large tiles and stripes
- tile-batch APIs for repeated viewer workloads
- additive device-output adapters for Metal and CUDA consumers
- explicit separation between image codecs and tile decompression codecs

The public WSI decode surface is documented in
[docs/wsi-decode-api.md](docs/wsi-decode-api.md).

The project is structured so WSI readers can compose their own threading,
vendor/container parsing, pyramid policy, caching, and prefetch around codec
primitives instead of paying for a monolithic runtime.

## Current scope

### `signinum-jpeg`

- Baseline JPEG support already present in the crate
- ROI, scaled decode, row streaming, and tile-batch decode APIs
- WSI-focused benchmarking against `jpeg-decoder`, `zune-jpeg`, and direct
  `libjpeg-turbo` decode paths
- Metal and CUDA adapter crates keep the core JPEG decoder pure-Rust CPU while
  exposing device-output surfaces for downstream GPU pipelines; the Metal path
  has optimized kernel paths for supported baseline JPEG tile shapes, including
  batched 4:2:0 and 4:2:2 RGB WSI tile decode, while explicit CUDA requests
  currently upload CPU-decoded bytes into CUDA device memory

### `signinum-j2k`

- JP2 / raw codestream inspect
- full-frame, region, scaled, combined region+scaled, row-bounded, and
  tile-batch decode
- repo-local pure-Rust JPEG 2000 / HTJ2K decode engine
- ROI+reduced-resolution performance coverage in the CPU and Metal benchmark
  harnesses
- parity and benchmark coverage against Grok and OpenJPEG on CPU
- Metal and CUDA adapter crates expose device-output surfaces without moving
  the core decoder crate onto GPU-specific dependencies; explicit Metal
  requests return Metal-backed full, ROI, scaled, and ROI+scaled surfaces on
  macOS, while explicit CUDA requests return CUDA-backed full, ROI, scaled, and
  ROI+scaled surfaces when `cuda-runtime` and a CUDA driver are available

### `signinum-tilecodec`

- `DeflateCodec`
- `ZstdCodec`
- `LzwCodec`
- `UncompressedCodec`

These codecs expose the shared `TileDecompress` trait from `signinum-core`.

## Quick start

JPEG inspect:

```rust
use signinum_jpeg::{Decoder, JpegView};

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
use signinum_core::{Downscale, PixelFormat};
use signinum_j2k::J2kDecoder;

let bytes = std::fs::read("tile.jp2")?;
let mut decoder = J2kDecoder::new(&bytes)?;
let (w, h) = decoder.info().dimensions;
let mut rgb = vec![0_u8; (w * h * 3) as usize];
decoder.decode_scaled_into(
    &mut signinum_j2k::J2kScratchPool::new(),
    &mut rgb,
    (w * 3) as usize,
    PixelFormat::Rgb8,
    Downscale::None,
)?;
```

JPEG 2000 / HTJ2K Metal tile batch:

```rust
use signinum_core::{BackendRequest, PixelFormat};
use signinum_j2k_metal::MetalTileBatch;

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
use signinum_core::TileDecompress;
use signinum_tilecodec::{DeflateCodec, DeflatePool};

let compressed = std::fs::read("tile.deflate")?;
let mut pool = DeflatePool::new();
let mut out = vec![0_u8; 1 << 20];
let written = DeflateCodec::decompress_into(&mut pool, &compressed, &mut out)?;
println!("decoded {} bytes", written);
```

CLI inspect:

```sh
$ signinum inspect tile.jp2
1024×1024 Srgb bit=8 comps=3 levels=6 tiles=Some(...)
```

Runnable crate examples are available under:

- `crates/signinum-jpeg/examples`
- `crates/signinum-j2k/examples`
- `crates/signinum-tilecodec/examples`

## Benchmarks

Benchmark methodology and comparator policy live in [docs/bench.md](docs/bench.md).
The repo now carries dedicated compare benches for:

- `signinum-jpeg`
- `signinum-j2k`
- `signinum-jpeg-metal`
- `signinum-j2k-metal`
- `signinum-tilecodec`

Release staging notes live in [docs/release.md](docs/release.md).

## License

Apache-2.0. See `LICENSE-APACHE`.

## MSRV

Rust 1.94.
