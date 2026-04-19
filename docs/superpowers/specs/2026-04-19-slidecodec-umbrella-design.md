# Slidecodec — Pathology Codec Stack Umbrella Design (rev 2)

**Status:** Umbrella design (pre-implementation).
**Release policy (revised):** no *public 1.0 release* until the core pathology codec stack is complete; internal crates evolve independently inside the workspace; milestones land on `main` but do not become public release promises.
**1.0 release gate (narrowed):** **JPEG + JPEG2000 (Part 1 + HT Part 15).** These two together are the coherent pathology decode promise.
**Concurrent track (may or may not land in 1.0):** `slidecodec-tilecodec` (LZW, Deflate, Zstd, Uncompressed). Needed for TIFF integration; thin wrappers around well-tuned system libraries. If ready at gate time, bundled in 1.0; otherwise shipped in 1.x.
**Post-1.0 roadmap (deferred, not gated):** `slidecodec-webp`, `slidecodec-jxl`. These are separate codecs with their own competitive landscape and should not turn this doc into a wishlist.
**Performance goals (revised to acceptance contracts, see §7):** per-codec, pinned-host, fixed-corpora, declared primary surface.

**This document is the umbrella.** Each milestone below gets its own spec → plan → implementation cycle.

---

## 1. Context

The `slidecodec` workspace currently contains one production crate: `slidecodec-jpeg`, a pure-Rust JPEG decoder on branch `perf/beat-both-wsi`. Current status (grounded in the tree, not aspirational):

- **Borrowed decoder architecture is live.** `JpegView<'a>` at `crates/slidecodec-jpeg/src/decoder.rs:49` and `Decoder<'a>` at `crates/slidecodec-jpeg/src/decoder.rs:82`. `Decoder::new(input: &'a [u8])` at `decoder.rs:112`, `Decoder::from_view`/`from_view_in_context` at `decoder.rs:118/124`.
- **WSI API surface is implemented.** `decode_rows` at `decoder.rs:340`, `decode_region_into` at `decoder.rs:369`, `decode_tile_into` at `decoder.rs:504`, plus `decode_into_with_scratch` and context-reuse variants. `DownscaleFactor` variants exist on `OutputFormat` (`info.rs:123,126`).
- **Performance baseline is competitive.** NEON IDCT lands a 2.1× block-level speedup; scratch-pool reuse and table cache deliver 3.9–16.7× over `zune-jpeg` on small-tile fresh-mode benches. AVX2 path is wired.
- **Correctness discipline is in place.** libjpeg-turbo byte-parity fixtures, `proptest` parser robustness (4096 cases), `cargo-fuzz` targets `parse_fuzz` and `decode_fuzz`, `#![deny(clippy::indexing_slicing, ...)]` in entropy/parse modules.

**JPEG is the reference implementation for the core WSI API model.** M0 extracts and generalizes that surface into `slidecodec-core`; it does not invent it.

The product vision is the **pathology decoder stack**: codecs that appear in WSI containers (SVS, OME-TIFF, iSyntax-adjacent), with WSI-specific API ergonomics (borrowed views, tile-batch context caching, scratch-pool reuse, ROI decode, resolution-level descent) and best-in-class performance per codec. JPEG is the beachhead. JPEG2000 is the next codec and is where the competitive gap vs. OpenJPEG is largest — especially HT-JPEG2000 (Part 15), where OpenJPEG has no first-party support (experimental via OpenHTJ2K/OpenJPH). A fast, correct, pure-Rust HT decoder is a genuine differentiator.

Because nothing is publicly released, we can reshape `slidecodec-jpeg`'s public surface freely during the core extraction. No backward-compat constraints.

---

## 2. Workspace layout

```
slidecodec/
├── crates/
│   ├── slidecodec-core/          [NEW] shared borrowed traits, sample types, scratch-pool trait, CPU detect
│   ├── slidecodec-jpeg/          [REFACTORED] implements ImageDecode<'a> + TileBatchDecode
│   ├── slidecodec-j2k/           [NEW] Part 1 baseline + HT (Part 15)
│   ├── slidecodec-tilecodec/     [NEW, parallel] LZW, Deflate, Zstd, Uncompressed (all impl TileDecompress)
│   └── slidecodec-cli/           [UPDATED] magic-byte dispatch across image codecs
├── corpus/
│   ├── conformance/              # always-cloned byte-parity gates (jpeg/, j2k/, tile/)
│   ├── wsi/                      # Git LFS; larger fixtures (aperio-svs/, tcia-samples/, philips/)
│   └── manifest.json             # reference tool versions, SHA256, regenerate scripts
├── docs/
│   ├── bench.md                  # per-codec perf acceptance contract + methodology
│   └── architecture.md           # link to this umbrella + per-codec specs
└── fuzz/                         # per-codec cargo-fuzz harnesses
```

Post-1.0: `slidecodec-webp` and `slidecodec-jxl` crates added without reshaping core. Core traits are designed today to admit them, but implementing them is not part of this umbrella's execution.

---

## 3. `slidecodec-core` trait surfaces

### Sample type is first-class

Because J2K (and later JXL) decode natively at 1–16-bit precisions and WSI sometimes carries 10/12/16-bit data, sample type must be in core on day one. `u8`-only row sinks are rejected.

```rust
// slidecodec-core::sample
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleType { U8, U16 }   // #[non_exhaustive]; future: U12 packed, F16, ...

pub trait Sample: Copy + Default + Send + Sync + 'static {
    const TYPE: SampleType;
    const BITS: u8;
}
impl Sample for u8  { const TYPE: SampleType = SampleType::U8;  const BITS: u8 = 8; }
impl Sample for u16 { const TYPE: SampleType = SampleType::U16; const BITS: u8 = 16; }
```

### `PixelFormat` — layout × sample, discriminated explicitly

```rust
#[non_exhaustive]
pub enum PixelLayout { Rgb, Rgba, Gray }

#[non_exhaustive]
pub enum PixelFormat {
    Rgb8, Rgba8, Gray8,
    Rgb16, Rgba16, Gray16,
}

impl PixelFormat {
    pub fn layout(self) -> PixelLayout { ... }
    pub fn sample(self) -> SampleType  { ... }
    pub fn bytes_per_pixel(self) -> usize { ... }
}
```

Rationale for discrete enum rather than `PixelFormat<S: Sample>`: keeps the core `ImageDecode` trait object-safe-shaped and avoids threading a generic `S` through every decode method. Codecs that only support one sample type reject others with `Unsupported`. 16-bit variants ship in M0 as part of the enum (non-breaking extension point preserved by `#[non_exhaustive]`).

### `Downscale` (separate from `PixelFormat`)

```rust
#[non_exhaustive]
pub enum Downscale { None, Half, Quarter, Eighth }
```

Decouples pixel layout from scale. Replaces the current draft `OutputFormat::Rgb8Scaled { factor }` in jpeg during the M0 refactor (pre-public; safe to reshape).

### `Info` — codec-agnostic; only shared concepts

```rust
pub struct Info {
    pub dimensions: (u32, u32),
    pub components: u8,
    pub colorspace: Colorspace,     // codec-agnostic enum, #[non_exhaustive]
    pub bit_depth: u8,              // per-component, 1..=16
    pub tile_layout: Option<TileLayout>,
    pub resolution_levels: u8,      // 1 for most JPEG cases; N for J2K
}
```

**Codec-specific metadata stays in codec crates, not in core `Info`.** JPEG's `scan_count`, `restart_interval`, `sampling_factors`, SOF kind, etc., move to a `JpegExtras` struct accessed via inherent methods on `JpegView<'a>` / `Decoder<'a>` (e.g., `view.extras() -> &JpegExtras`). J2K's tile-part table, progression order, precinct sizes, etc., live analogously on `J2kView<'a>` as `J2kExtras`. This preserves zero-cost typed access without polluting the core type or introducing `dyn Any` downcasts.

### `ImageCodec` — shared associated types (marker)

Lifts the three cross-cutting types (error, warning, pool) to a marker trait so the borrowed-decoder trait and the freestanding tile-batch trait can share them without coupling their lifetime stories.

```rust
pub trait ImageCodec {
    type Error:   CodecError;
    type Warning: core::fmt::Debug + core::fmt::Display + Send + Sync + 'static;
    type Pool:    ScratchPool;
}
```

### `ImageDecode<'a>` — borrowed decoder (matches current JPEG architecture)

```rust
pub trait ImageDecode<'a>: ImageCodec + Sized + 'a {
    type View: 'a;

    fn inspect(input: &'a [u8]) -> Result<Info, Self::Error>;
    fn parse(input: &'a [u8])   -> Result<Self::View, Self::Error>;
    fn from_view(view: Self::View) -> Result<Self, Self::Error>;

    fn decode_into(&mut self, out: &mut [u8], stride: usize, fmt: PixelFormat)
        -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    fn decode_into_with_scratch(&mut self, pool: &mut Self::Pool,
                                out: &mut [u8], stride: usize, fmt: PixelFormat)
        -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    fn decode_region_into(&mut self, pool: &mut Self::Pool,
                          out: &mut [u8], stride: usize, fmt: PixelFormat, roi: Rect)
        -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    fn decode_scaled_into(&mut self, pool: &mut Self::Pool,
                          out: &mut [u8], stride: usize, fmt: PixelFormat, scale: Downscale)
        -> Result<DecodeOutcome<Self::Warning>, Self::Error>;
}
```

Row streaming is parameterized by sample type:

```rust
pub trait RowSink<S: Sample> {
    type Error: core::error::Error + Send + Sync + 'static;
    fn write_row(&mut self, y: u32, row: &[S]) -> Result<(), Self::Error>;
}

pub enum DecodeRowsError<D, S> {
    Decode(D),
    Sink(S),
}
// Implements core::error::Error when D and S do; Display/Debug forwarding.

pub trait ImageDecodeRows<'a, S: Sample>: ImageDecode<'a> {
    fn decode_rows<R: RowSink<S>>(&mut self, sink: &mut R)
        -> Result<DecodeOutcome<Self::Warning>,
                  DecodeRowsError<Self::Error, R::Error>>;
}
```

Generic-sink signature (not `&mut dyn`) so the sink's typed error composes cleanly with the decoder's typed error via `DecodeRowsError`. Monomorphization cost is acceptable — row sinks are usually one per call site. JPEG implements `ImageDecodeRows<'a, u8>` only. J2K implements both `<'a, u8>` and `<'a, u16>`.

### `TileBatchDecode` — freestanding, per-tile borrow (no decoder-level `'a`)

Tile-batch methods are static/associated functions that take a fresh borrow per tile. Matches current jpeg `decode_tile_into` at `crates/slidecodec-jpeg/src/decoder.rs:504`. No parent `ImageDecode<'a>` constraint — the tile byte lifetime must not be forced to equal any decoder instance's lifetime.

```rust
pub trait TileBatchDecode: ImageCodec {
    type Context: CodecContext;

    fn decode_tile<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &'a [u8], out: &mut [u8], stride: usize, fmt: PixelFormat,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    fn decode_tile_region<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &'a [u8], out: &mut [u8], stride: usize,
        fmt: PixelFormat, roi: Rect,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    fn decode_tile_scaled<'a>(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &'a [u8], out: &mut [u8], stride: usize,
        fmt: PixelFormat, scale: Downscale,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;
}
```

### `TileDecompress` — byte→byte

```rust
pub trait TileDecompress {
    type Error: CodecError;
    type Pool: ScratchPool;

    fn expected_size(input: &[u8]) -> Result<Option<usize>, Self::Error>;
    fn decompress_into(pool: &mut Self::Pool, input: &[u8], out: &mut [u8])
        -> Result<usize, Self::Error>;
}
```

Each tile codec defines its own typed `Pool` (matching the `ImageDecode` pattern). Per-codec typed slots are where the actual scratch storage lives — window buffers for Deflate, dictionary buffers for LZW, decoder state for Zstd. The core `ScratchPool` trait still only exposes telemetry/reset; real byte arenas are codec-private fields behind typed accessors. For `Uncompressed` the pool is a unit `pub struct NoPool;` with trivial trait impl.

### Shared types

- `Rect { x: u32, y: u32, w: u32, h: u32 }`.
- `DecodeOutcome<W> { decoded: Rect, warnings: Vec<W> }`.
- `WarningKind` — `#[non_exhaustive]` enum `{ MinorCompliance, NonFatalTruncation, UnusualFeature, ... }`. Each codec's warning type exposes `kind(&self) -> WarningKind`.
- `DecoderContext<C: CodecContext>` — wrapper; per-worker owned; `!Sync` during decode via `&mut`.
- `CodecContext: Default + Send` — `{ clear(&mut self), cache_stats(&self) -> CacheStats }`. Not object-safe (used only as an associated type bound).
- `ScratchPool: Send` — `{ bytes_allocated(&self) -> usize, reset(&mut self) }`. Telemetry-only marker trait; the real byte-arena storage is codec-private, reached through each codec's own typed pool.
- `Colorspace` — `#[non_exhaustive]` enum covering JPEG (YCbCr, YCCK, Gray, Rgb, CMYK) + J2K (sRGB, sGray, ICC-tagged, RCT, ICT).
- `TileLayout` — optional tile grid description in `Info`.

---

## 4. Error taxonomy

Per-codec typed errors, composed from shared core sub-errors:

```rust
pub enum BufferError {
    OutputTooSmall    { required: usize, have: usize },
    StrideTooSmall    { row_bytes: usize, stride: usize },
    StrideNotAligned  { stride: usize, align: usize },
    SampleTypeMismatch { fmt: PixelFormat, pool_sample: SampleType },
}
pub enum InputError {
    TooShort     { need: usize, have: usize },
    TruncatedAt  { offset: usize, segment: &'static str },
}
pub struct NotImplemented { pub what: &'static str }
pub struct Unsupported    { pub what: &'static str }

pub trait CodecError: core::error::Error + Send + Sync + 'static {
    fn is_truncated(&self)       -> bool;
    fn is_not_implemented(&self) -> bool;
    fn is_unsupported(&self)     -> bool;
    fn is_buffer_error(&self)    -> bool;
}
```

Each codec composes: `J2kError::Buffer(BufferError)`, `J2kError::Input(InputError)`, `J2kError::Unsupported(Unsupported)`, plus codec-specific variants (`InvalidMarker`, `MqCoder`, `Ebcot`, `Ht`, `Dwt`, `Color`, …). Same shape preserved in `JpegError` during the M0 refactor.

**Panic-freeness is merge-blocking per codec milestone:**
- `#![deny(clippy::indexing_slicing, clippy::panic, clippy::unwrap_used, clippy::expect_used)]` at entropy/parse module scope; scoped exceptions documented inline.
- `cargo-fuzz` targets: `parse_fuzz` (inspect) and `decode_fuzz` (full decode). 1M iterations without a new panic before declaring any codec's milestone complete.
- `proptest` parser-robustness suite (4096 cases). Properties: inspect never panics; malformed returns `Err`; `decode_into` validates buffer sizes tightly.

`slidecodec-core` is `no_std`-compatible (`core::error::Error`, stabilized 1.81; no `std::io` in error types). Codec crates may use `std` freely.

---

## 5. WSI hot path — scratch pool, context, tile-batch, SIMD

### Scratch pools — per-codec, typed slots

Each codec defines its own pool matching jpeg's current shape (`src/internal/scratch.rs`): lazy allocation, grow monotonically, never shrink, typed slot access. `slidecodec-core`'s `ScratchPool` trait only carries telemetry/reset; the real scratch storage is per-codec typed fields. Both `ImageDecode::Pool` and `TileDecompress::Pool` use `type Pool: ScratchPool` (associated-type static dispatch) for typed-slot ergonomics in each codec's hot path.

No cross-codec arena sharing (WSI tile batches are single-codec; sharing is a phantom benefit). No `&mut dyn ScratchPool` anywhere in the public API — the trait exists purely as a bound.

### `DecoderContext<C>` — per-worker, `!Sync` mid-decode

Owned per worker thread. `Send` between decodes. Per-codec `C` owns caches (JPEG: DQT/DHT/decode-plan slots — current `crates/slidecodec-jpeg/src/context.rs`; J2K: parsed SIZ/COD/QCD tables + resolution-descent plans per tile shape; HT: FAST coding-pass tables).

### Tile-batch ownership contract (locked in for all codecs)

- Input bytes `&'a [u8]`, borrowed per tile; caller owns storage.
- Output buffer `&mut [u8]`, borrowed per tile; caller owns the pixel buffer.
- `DecoderContext<C>` owned per worker thread; `!Sync` during decode via `&mut` borrow.
- `ScratchPool` owned per worker thread; not shared.
- **No bundled thread pool.** WSI readers supply their own (rayon, tokio, …); slidecodec is a leaf.

Matches existing jpeg `tests/batch.rs` behavior; elevated to the trait-level contract.

### SIMD dispatch

`slidecodec-core::backend::CpuFeatures { avx2, sse41, neon }` detected once via `OnceLock`. Each codec owns its `backend/` directory (mirroring `slidecodec-jpeg/src/backend/{scalar,x86,neon}.rs`). Function-pointer dispatch selected at decoder construction; zero-cost in hot loops.

**Target matrix:** x86_64 (AVX2 + SSE4.1 baseline) and aarch64 (NEON). Other targets `compile_error!`. AVX-512 / SVE deferred until a measured bench win justifies the complexity.

---

## 6. Port-vs-design per codec layer

| Layer                              | Port from reference | Redesign for slidecodec |
|------------------------------------|---------------------|-------------------------|
| Marker/box parsers                 |                     | ✓ (mirror jpeg `parse/` shape) |
| MQ-coder (bit-exact state machine) | ✓ (OpenJPEG)        |                         |
| EBCOT Tier-1 coding passes         | ✓ (OpenJPEG)        | ✓ (SoA storage, SIMD-friendly) |
| EBCOT Tier-2 packet parsing        | ✓ (OpenJPEG)        | ✓ (precinct iteration) |
| DWT 9-7 & 5-3 lifting coefficients | ✓ (spec)            | ✓ (in-place SIMD, resolution-descent) |
| HT FAST block coder                | ✓ (OpenHTJ2K)       | ✓ (storage parallel to EBCOT) |
| Color transforms (ICT/RCT)         | ✓ (spec)            | ✓ (SIMD dispatch) |
| Tile/precinct iteration            |                     | ✓ clean slate |
| `ScratchPool`, `DecoderContext`    |                     | ✓ clean slate |

Reference licenses: OpenJPEG BSD-2, OpenHTJ2K MIT, OpenJPH BSD-2 — all Apache-2.0-compatible. Study for algorithm structure and use intermediate state as debug oracles during bring-up; do not copy code line-for-line.

---

## 7. Acceptance contract (per-codec, measurable)

Replaces the earlier "beat X on every operation" framing. Every codec milestone's perf sign-off must specify all four fields.

**Shape:**
- **Pinned hardware.** One x86_64 host, one aarch64 host, both documented in `docs/bench.md`: CPU model, core count, turbo policy, memory, OS, kernel. Bench runs use `release-bench` profile, `taskset`/`chrt`-pinned, host otherwise idle.
- **Fixed corpora.** Each bench group references a named corpus subset (by SHA256-rooted manifest). No ad-hoc inputs. Corpora: `corpus/conformance/*` (small, always-cloned) and `corpus/wsi/*` (LFS, opt-in).
- **Tie threshold.** slidecodec wins a bench group iff `median(ns_per_px) ≤ 0.95 × reference.median(ns_per_px)` with non-overlapping 99% Criterion confidence intervals. Within the tie band, we declare "competitive," not a win.
- **Declared primary surface per codec.** The set of bench groups where a win is mandatory before the milestone can be called complete. Non-primary groups must at minimum be "competitive."

**JPEG primary surface (for M0-refactor acceptance).** All comparators are configured to give libjpeg-turbo its fair best:

- **Comparator configuration for libjpeg-turbo (FFI):**
  - Full-frame + tile-batch: **TurboJPEG API** (`tjInit3Decompress` → `tj3Decompress8`), with one TurboJPEG handle reused across all iterations/tiles. Not `tjInitDecompress` per-tile; not the classic libjpeg `jpeg_create_decompress` loop (which allocates per call).
  - ROI: **classic libjpeg API** with `jpeg_crop_scanline` for column cropping + `jpeg_skip_scanlines` for row cropping — this is a true decode-time operation. Not decode-then-crop. Reusable `jpeg_decompress_struct` across tiles.
  - DCT-scaled (q4, q8): **TurboJPEG API** with `tj3Set(handle, TJPARAM_SCALINGFACTOR, ...)` using `TJSCALED_1_4` / `TJSCALED_1_8`. True decode-time DCT scaling. Not decode-then-decimate.
  - Runtime detection: SIMD ON (AVX2 on x86_64, NEON on aarch64, the libjpeg-turbo default).
- **Comparator configuration for `jpeg-decoder` and `zune-jpeg`:** fresh-per-tile construction (matches their idiomatic usage; neither has a reusable-handle API). Region / scale comparators decode full-frame then crop/decimate in memory (they have no native decode-time ROI or scale).
- **Primary bench groups (win mandatory):**
  - `wsi_tile_batch_rgb` — tile-batch parse+decode with `DecoderContext` reuse; **win** vs all three comparators.
  - `wsi_region_rgb` — ROI decode; **win** vs libjpeg-turbo's native `jpeg_crop_scanline`+`jpeg_skip_scanlines` path, and vs decode-then-crop for the other two.
  - `wsi_scaled_rgb_q4`, `wsi_scaled_rgb_q8` — DCT-scaled decode; **win** vs libjpeg-turbo's native TJSCALED path, and vs decode-then-decimate for the other two.
- **Secondary bench groups (competitive, must not regress):**
  - `decode_rgb`, `decode_gray` — generic full-frame; must stay within the 0.95× tie band vs. the pre-M0-refactor baseline and within the tie band vs. libjpeg-turbo's reusable TurboJPEG handle path.
  - `decode_rows_rgb` — **slidecodec-only, no cross-crate comparator.** This group exists for very large WSI JPEGs where full-frame output buffers are impractical (matches the existing `docs/bench.md` methodology note). Acceptance here is narrowed to two checks: (a) no regression vs. the pre-M0-refactor baseline on the same fixtures, and (b) throughput within the 0.95× tie band of `decode_rgb` on the same image when `decode_rgb` is actually feasible (i.e., small enough to fit a full-frame buffer) — which confirms that the row-streaming path pays no per-row overhead beyond the sink dispatch. Cross-crate comparison remains out of scope because neither `jpeg-decoder` nor `zune-jpeg` exposes a row-streaming decode API; comparing to decode-into-a-huge-buffer would be testing memory pressure, not codec throughput.

**J2K primary surface (for J2K-M2 acceptance; earlier milestones have correctness gates only):**
- `wsi_tile_batch_rgb` with `DecoderContext<J2kContext>` reuse: **win** vs OpenJPEG tile-by-tile.
- `wsi_region_rgb` (ROI via codestream navigation): **win** vs OpenJPEG decode-then-crop.
- `wsi_scaled_rgb_q{4,8,16}` (DWT resolution descent): **win** vs OpenJPEG decode-then-decimate — J2K's killer feature; largest expected margin.
- `decode_rgb` (full-frame): **competitive** with OpenJPEG. Not a primary win target.

**HT-J2K primary surface (for J2K-M3 acceptance):**
- Any HT workload: **win** vs OpenJPEG (no first-party HT). **Competitive** vs OpenHTJ2K reference.

**Tile codec primary surface (for M8 acceptance):**
- `decompress_into` throughput: **competitive** with system `libz` (Deflate) and `libzstd` (Zstd). LZW is compared against a well-known BSD-licensed reference implementation (the one used by `libtiff`). Uncompressed is a `memcpy` baseline. Not a primary win target — thin wrappers around tuned C libraries; we compete on API ergonomics, not raw throughput.

Bench is **not** run in CI; sign-off is manual, pinned-host, recorded in `docs/bench.md` per milestone.

---

## 8. Testing strategy

Four-layer correctness pyramid, applied per codec crate:

1. **ISO conformance gate** (merge-blocking where applicable).
   - J2K: ISO 15444-4 bitstreams, bit-exact decoded-output match in CI.
   - JPEG: libjpeg-turbo parity corpus (existing pattern at `corpus/conformance/baseline_420_16x16.*`).
2. **Reference-implementation parity oracle.**
   - JPEG ↔ libjpeg-turbo (existing `generate.sh` + `manifest.json` pattern).
   - J2K ↔ OpenJPEG (primary), OpenHTJ2K (HT).
   - Tile codecs ↔ system `zlib` (Deflate), `libzstd` (Zstd), `libtiff`'s LZW reference (LZW); no-op for Uncompressed.
   - Fixtures golden; regenerated only by explicit `./generate.sh`; manifest records tool version + SHA256.
3. **WSI corpus fixtures** — user-supplied, Git LFS. CI pulls LFS on conformance/bench jobs.
4. **`cargo-fuzz` + `proptest`** — per-codec `parse_fuzz` + `decode_fuzz`, 1M-iteration clean run before milestone ship; 4096-case `proptest` robustness suite.

CI matrix extends current jpeg matrix (`fmt × clippy × test × stable/beta/MSRV × linux/macos`) per codec crate, plus workspace-level `cargo deny`.

---

## 9. Milestone roadmap

Each milestone gets its own spec (`docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`) → plan → implementation.

| Milestone | Name | Exit criteria | Depends on |
|-----------|------|---------------|------------|
| **M0** | Core extraction + jpeg refactor | `slidecodec-core` exists with traits + sample types + `PixelFormat` (incl. 16-bit variants) + `Downscale`. `Decoder<'a>` implements `ImageDecode<'a> + ImageDecodeRows<'a, u8>`; unit `JpegCodec` implements `TileBatchDecode`. All existing tests green. JPEG acceptance contract (§7) holds (no perf regression on tile-batch/region/scaled; competitive on full-frame). Public API pre-1.0 churn is acceptable and audited. | — |
| **J2K-M0** | `slidecodec-j2k` skeleton + `inspect` | Crate exists; JP2 box parser + J2K codestream marker parser; `J2kDecoder::inspect() -> Info` populating precision (8/10/12/16 bit), components, resolution levels, tile layout; `parse_fuzz` + `proptest`; CLI `inspect` dispatches by magic bytes. | M0 |
| **J2K-M1** | Scalar baseline decode (9-7 irreversible, ICT) | MQ-coder + EBCOT T1 + T2 + DWT 9-7 + ICT + `decode_into(Rgb8/Rgba8/Gray8/Rgb16/Gray16)`. Full-frame only. Classic Part 1. ISO 15444-4 conformance (9-7 profile) + OpenJPEG parity + WSI corpus. `decode_fuzz` 1M clean. | J2K-M0 |
| **J2K-M1b** | Lossless (5-3 reversible, RCT) | Adds 5-3 DWT + RCT paths. Conformance + parity. | J2K-M1 |
| **J2K-M2** | WSI APIs + acceptance contract | `decode_region_into`, `decode_scaled_into` via DWT resolution descent (not post-decimation), `decode_tile` with `DecoderContext<J2kContext>`, `ImageDecodeRows<'a, u8>` + `<'a, u16>`. **J2K primary-surface acceptance contract (§7) met.** | J2K-M1b |
| **J2K-M3** | HT (Part 15) | HT FAST block coder, marker-detected per codestream. ISO HT test vectors + OpenHTJ2K parity + WSI HT corpus. `decode_fuzz` 1M clean on HT inputs. **HT acceptance contract met.** | J2K-M2 |
| **J2K-M4** | SIMD | NEON + AVX2 for DWT (9-7 + 5-3), T1 (EBCOT + HT), color. Updated bench pass vs acceptance contract on both archs. | J2K-M3 |
| **J2K-M5** | Hardening | Differential fuzzing vs OpenJPEG; memory-budget stress; cross-platform CI. | J2K-M4 |
| **M8** (parallel) | `slidecodec-tilecodec` | LZW, Deflate, Zstd, Uncompressed. `TileDecompress` impls. System-library parity. Tile-codec acceptance contract met. | M0 |
| **M9** | 1.0 release gate | Workspace `1.0.0`. JPEG and J2K milestones complete and sign-off'd on pinned hardware. `slidecodec-tilecodec` may ship in 1.0 if ready, otherwise 1.x. README rewrite. `docs/bench.md` final. | J2K-M5 + (M8 optional) |

Post-1.0 (separate umbrella in the future): WebP, JXL.

**Parallelizable:** M8 (tile codecs) runs in parallel with J2K-M{0..5}; no shared hot path beyond core.

---

## 10. What this plan does not do

- **Does not implement code.** This is the umbrella. Implementation begins with the M0 spec.
- **Does not commit WebP or JXL to 1.0.** Deferred to a later roadmap; core traits are designed to admit them without reshaping.
- **Does not pick specific reference-tool versions** (OpenJPEG, OpenHTJ2K). That choice lives in the J2K-M1 spec (captured in `manifest.json`).
- **Does not commit to AVX-512 / SVE.** Deferred until measured win.
- **Does not define LZMA support.** `slidecodec-tilecodec` covers Deflate / Zstd / LZW / Uncompressed only.

---

## 11. Critical files

Created in M0:
- `crates/slidecodec-core/Cargo.toml`
- `crates/slidecodec-core/src/lib.rs`
- `crates/slidecodec-core/src/sample.rs` — `SampleType`, `Sample` trait, `u8`/`u16` impls
- `crates/slidecodec-core/src/pixel.rs` — `PixelLayout`, `PixelFormat`
- `crates/slidecodec-core/src/scale.rs` — `Downscale`
- `crates/slidecodec-core/src/traits.rs` — `ImageCodec`, `ImageDecode<'a>`, `TileBatchDecode` (no lifetime), `ImageDecodeRows<'a, S>`, `TileDecompress`, `DecodeRowsError<D, S>`
- `crates/slidecodec-core/src/types.rs` — `Rect`, `Info`, `Colorspace`, `TileLayout`, `DecodeOutcome`, `WarningKind`
- `crates/slidecodec-core/src/row_sink.rs` — `RowSink<S: Sample>`
- `crates/slidecodec-core/src/scratch.rs` — `ScratchPool` trait
- `crates/slidecodec-core/src/context.rs` — `DecoderContext<C>`, `CodecContext`, `CacheStats`
- `crates/slidecodec-core/src/error.rs` — `BufferError`, `InputError`, `NotImplemented`, `Unsupported`, `CodecError` marker trait
- `crates/slidecodec-core/src/backend.rs` — `CpuFeatures` + `detect()`

Modified in M0:
- `Cargo.toml` (workspace) — add `slidecodec-core` member
- `crates/slidecodec-jpeg/Cargo.toml` — add `slidecodec-core` dep
- `crates/slidecodec-jpeg/src/lib.rs` — re-exports; trait impls wire-up
- `crates/slidecodec-jpeg/src/decoder.rs:49` (`JpegView<'a>`) — implement as `ImageDecode<'a>::View`
- `crates/slidecodec-jpeg/src/decoder.rs:82` (`Decoder<'a>`) — move inherent methods behind `impl<'a> ImageCodec for Decoder<'a>` + `impl<'a> ImageDecode<'a> for Decoder<'a>` + `impl<'a> ImageDecodeRows<'a, u8> for Decoder<'a>`
- Introduce a unit type `pub struct JpegCodec;` in `lib.rs` that carries the freestanding `TileBatchDecode` impl: `impl ImageCodec for JpegCodec` + `impl TileBatchDecode for JpegCodec`. The existing free functions `decode_tile_into` / `decode_tile_into_in_context` (`decoder.rs:504/519`) become the bodies of `JpegCodec::decode_tile` / `decode_tile_region`. The original top-level `decode_tile_into` stays as a convenience re-export so downstream code keeps working during the refactor.
- `crates/slidecodec-jpeg/src/decoder.rs:74` (`RgbRowSink`) — remove; callers migrate to core `RowSink<u8>` via a blanket impl shim during the refactor
- `crates/slidecodec-jpeg/src/info.rs:121-127` (`OutputFormat`) — drop the `Rgb8Scaled`/`Gray8Scaled` variants; callers move to `decode_scaled_into(..., PixelFormat::Rgb8, Downscale::Quarter)`. `RawYCbCr8` stays jpeg-specific (not in core `PixelFormat`), behind a jpeg-local extension trait.
- `crates/slidecodec-jpeg/src/error.rs` — compose with core sub-errors; implement `CodecError` marker
- `crates/slidecodec-jpeg/src/internal/scratch.rs` — implement core `ScratchPool` trait
- `crates/slidecodec-jpeg/src/context.rs` — implement core `CodecContext`
- `crates/slidecodec-jpeg/src/backend/mod.rs` — use core `CpuFeatures`
- `crates/slidecodec-cli/src/main.rs` — prepare magic-byte dispatch hook (wiring for j2k lands in J2K-M0)

Created in J2K-M0 (next spec after M0):
- `crates/slidecodec-j2k/` entire crate skeleton (parser + `inspect` only).

---

## 12. Verification (M0-specific)

- `cargo build --workspace` — clean.
- `cargo test --workspace` — all existing jpeg tests green; zero behavioral regressions in `tests/batch.rs`, `tests/view_and_rows.rs`, `tests/scratch_reuse.rs`, `tests/decode_into.rs`, `tests/external_wsi.rs`, `tests/regressions.rs`, `tests/idct_parity.rs`.
- `cargo clippy --workspace --all-targets -- -D warnings` — clean, pedantic level preserved.
- `cargo deny check` — clean.
- `cargo fuzz run parse_fuzz -- -runs=1000000` and `cargo fuzz run decode_fuzz -- -runs=1000000` — no new panics.
- `cargo bench -p slidecodec-jpeg --bench compare -- --quick` on both pinned hosts — acceptance contract (§7, JPEG primary surface) green: tile-batch/region/scaled wins hold; full-frame is within Criterion tie band of pre-refactor baseline.
- Manual audit of `slidecodec-jpeg` public API diff: every removed/renamed symbol is intentional, documented in CHANGELOG Unreleased section.

**Next step after this umbrella is approved:** write the M0 spec (core extraction + jpeg refactor), then the J2K-M0 spec (J2K skeleton + inspect) via a fresh brainstorm.
