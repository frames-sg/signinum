# slidecodec

JPEG decoder optimized for whole-slide images (WSI).

## Status

Pre-0.1. Unstable. API changes without notice between minor versions. Do not use in production until 1.0.
Current decode targets are native `x86_64` and `aarch64` hosts.

## What this is

A pure-Rust JPEG decoder designed around the access patterns that whole-slide
image workloads actually need:

- **Partial/cropped decode** — skip MCUs outside the region of interest.
- **IDCT-level downscale** — 1/2, 1/4, 1/8 scale without full decode + resample.
- **Per-restart-segment parallelism** — caller-driven, no built-in thread pool.
- **Pre-parsed shared table cache** — amortize DQT/DHT parse across thousands of tiles.
- **Stride-aware direct-to-caller output** — zero-copy writes into tile caches.
- **Scan-fragment stitching** — first-class support for NDPI-style striped scans.

Scope: SOF0 (baseline), SOF1 (extended 8/12-bit), SOF2 (progressive), SOF3 (lossless Annex H Huffman).

## What this is NOT

If you are not decoding WSI files, use [`jpeg-decoder`] or [`zune-jpeg`]. Those
crates are battle-tested, general-purpose, and faster on typical web JPEGs.

This crate has:
- No support for arithmetic-coded JPEG (SOF9–11).
- No support for hierarchical JPEG (SOF5–7, 13–15).
- No color management / ICC profile application.
- No built-in async I/O or thread pool.

[`jpeg-decoder`]: https://crates.io/crates/jpeg-decoder
[`zune-jpeg`]: https://crates.io/crates/zune-jpeg

## License

Apache-2.0. See `LICENSE-APACHE`.

## MSRV

Rust 1.94. Bumps are minor-version events.

## Quick-start

```rust
use slidecodec_jpeg::{Decoder, OutputFormat};

let bytes = std::fs::read("tile.jpg")?;
let info = Decoder::inspect(&bytes)?;

let dec = Decoder::new(&bytes)?;
let (w, h) = dec.info().dimensions;
let mut rgb = vec![0u8; (w * h * 3) as usize];
dec.decode_into(&mut rgb, (w * 3) as usize, OutputFormat::Rgb8)?;
```

```rust
use slidecodec_jpeg::{Decoder, JpegView, JpegError, RgbRowSink};

struct Sink;

impl RgbRowSink for Sink {
    fn write_rgb_row(&mut self, _y: u32, _row: &[u8]) -> Result<(), JpegError> {
        Ok(())
    }
}

let bytes = std::fs::read("tile.jpg")?;
let view = JpegView::parse(&bytes)?;
let dec = Decoder::from_view(view)?;
dec.decode_rows(&mut Sink)?;
```

```sh
$ slidecodec inspect tile.jpg
1024×1024 Baseline8 YCbCr bit=8 samp=[(2, 2), (1, 1), (1, 1)] rst=Some(4) scans=1
```

## Status progression

- [x] M0 — Scaffolding, CI, licenses
- [x] M1a — Error taxonomy, header parser, `Decoder::inspect`
- [x] M1b — Bit reader, Huffman, IDCT, color convert, Decoder::decode_into
- [x] M1.x — `JpegView`, `Decoder::from_view`, `Decoder::decode_rows`
- [ ] M1.5 — Perf gate: prototype IDCT + Huffman benches
- [ ] M2 — WSI-specific APIs (partial decode, downscale, segments, table cache, stitch)
- [ ] M3 — Extended SOF support (SOF1/SOF2/SOF3)
- [ ] M4 — SIMD paths (NEON / AVX2 / SSE4.1 / simd128)
- [ ] M5 — Hardening (parity corpus, fuzz, miri)
- [ ] M6 — Release (0.1.0 to crates.io, SlideViewer integration)
