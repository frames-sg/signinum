# slidecodec

JPEG decoder optimized for whole-slide images (WSI).

## Status

Pre-0.1. Unstable. API changes without notice between minor versions. Do not use in production until 1.0.

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
