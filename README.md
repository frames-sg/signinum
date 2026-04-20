# slidecodec

Pathology codec stack for whole-slide imaging workloads.

## Status

`slidecodec` is a native-first codec workspace for pathology and WSI software.
The core stack in this repository is:

- `slidecodec-jpeg` — WSI-optimized JPEG decode
- `slidecodec-j2k` — JPEG 2000 / HTJ2K inspect and decode
- `slidecodec-tilecodec` — Deflate, Zstd, LZW, and Uncompressed tile
  decompression
- `slidecodec-core` — shared traits, pixel/sample types, scratch/context
  contracts
- `slidecodec-cli` — CLI inspection entry point

Target decode hosts are native `x86_64` and `aarch64`.

## What this is

`slidecodec` is designed around WSI access patterns instead of generic
consumer-image decode:

- borrowed parse/decode surfaces
- caller-owned scratch pools and decoder contexts
- decode-time ROI and reduced-resolution output
- row-streaming output for large tiles and stripes
- tile-batch APIs for repeated viewer workloads
- explicit separation between image codecs and tile decompression codecs

The project is structured so WSI readers can compose their own threading and
container parsing around codec primitives instead of paying for a monolithic
runtime.

## Current scope

### `slidecodec-jpeg`

- Baseline and extended/lossless JPEG support already present in the crate
- ROI, scaled decode, row streaming, and tile-batch decode APIs
- WSI-focused benchmarking against `jpeg-decoder`, `zune-jpeg`, and
  libjpeg-turbo-oriented workflows

### `slidecodec-j2k`

- JP2 / raw codestream inspect
- full-frame, region, scaled, row-streaming, and tile-batch decode
- HTJ2K handling through the current backend path
- OpenJPEG-oriented parity and benchmark coverage

### `slidecodec-tilecodec`

- `DeflateCodec`
- `ZstdCodec`
- `LzwCodec`
- `UncompressedCodec`

All expose the shared `TileDecompress` trait from `slidecodec-core`.

## Quick start

JPEG inspect:

```rust
use slidecodec_jpeg::Decoder;

let bytes = std::fs::read("tile.jpg")?;
let info = Decoder::inspect(&bytes)?;
println!("{:?} {:?}", info.dimensions, info.color_space);
```

JPEG 2000 decode:

```rust
use slidecodec_core::{Downscale, PixelFormat};
use slidecodec_j2k::J2kDecoder;

let bytes = std::fs::read("tile.jp2")?;
let mut decoder = J2kDecoder::new(&bytes)?;
let (w, h) = decoder.info().dimensions;
let mut rgb = vec![0_u8; (w * h * 3) as usize];
decoder.decode_scaled_into(
    &mut slidecodec_j2k::J2kScratchPool::new(),
    &mut rgb,
    (w * 3) as usize,
    PixelFormat::Rgb8,
    Downscale::None,
)?;
```

Tile decompression:

```rust
use slidecodec_core::TileDecompress;
use slidecodec_tilecodec::{DeflateCodec, DeflatePool};

let compressed = std::fs::read("tile.deflate")?;
let mut pool = DeflatePool::new();
let mut out = vec![0_u8; 1 << 20];
let written = DeflateCodec::decompress_into(&mut pool, &compressed, &mut out)?;
println!("decoded {} bytes", written);
```

CLI inspect:

```sh
$ slidecodec inspect tile.jp2
1024×1024 Srgb bit=8 comps=3 levels=6 tiles=Some(...)
```

## Benchmarks

Benchmark methodology and comparator policy live in [docs/bench.md](docs/bench.md).
The repo now carries dedicated compare benches for:

- `slidecodec-jpeg`
- `slidecodec-j2k`
- `slidecodec-tilecodec`

## License

Apache-2.0. See `LICENSE-APACHE`.

## MSRV

Rust 1.94.
