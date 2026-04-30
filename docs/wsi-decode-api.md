# WSI Decode API

This guide describes the public decode surfaces intended for whole-slide
imaging readers. It covers the stable caller contract shared by JPEG,
JPEG 2000 / HTJ2K, tile decompression, and the device-output adapters.

## Ownership Model

ashlar does not own a viewer runtime. Callers own I/O, threading, tile
coordinates, pyramid selection, cache policy, and prefetch. Codec APIs only
parse compressed bytes and write decoded pixels into caller-provided storage.

Use caller-owned state for hot loops:

- `ScratchPool` reuses temporary allocations within one codec family.
- `DecoderContext` reuses codec tables and planning state across tile batches.
- `DeviceSubmission` lets adapter crates queue work and return a `DeviceSurface`
  after `wait()`.

The codec crates do not spawn worker threads, hold global decode queues, or
hide output allocation behind a runtime.

## CPU Decode Surfaces

Use `ImageDecode` when the caller has one compressed image or tile and wants
pixels in host memory.

Common shapes:

- `decode_into` decodes the full image.
- `decode_region_into` decodes a source-coordinate ROI.
- `decode_scaled_into` decodes the full image at a reduced resolution.
- `decode_region_scaled_into` decodes a source-coordinate ROI on a reduced
  resolution grid.

ROI coordinates are always expressed in source-image pixels. For
`decode_region_scaled_into`, the returned `DecodeOutcome::decoded` rectangle is
the floor-start / ceil-end projection of the source ROI into the scaled grid.
`Downscale::None` preserves the original ROI.

```rust
use ashlar_core::{Downscale, ImageDecode, PixelFormat, Rect};
use ashlar_j2k::{J2kDecoder, J2kScratchPool};

let bytes = std::fs::read("tile.jp2")?;
let mut decoder = J2kDecoder::new(&bytes)?;
let roi = Rect {
    x: 512,
    y: 512,
    w: 1024,
    h: 1024,
};
let scale = Downscale::Quarter;
let scaled = roi.scaled_covering(scale);
let stride = scaled.w as usize * PixelFormat::Gray8.bytes_per_pixel();
let mut out = vec![0_u8; stride * scaled.h as usize];

decoder.decode_region_scaled_into(
    &mut J2kScratchPool::new(),
    &mut out,
    stride,
    PixelFormat::Gray8,
    roi,
    scale,
)?;
```

## Row Streaming

Use `decode_rows` through `ImageDecodeRows` when a tile or image is too large
for one packed output buffer or when the caller wants to feed rows into a
streaming consumer. The caller implements `RowSink`, and ashlar forwards sink
errors without converting them into silent decode success.

Row streaming is a host-memory API. Device adapters return surfaces instead of
row callbacks.

## Tile Batches

Use `TileBatchDecode` when a WSI reader is decoding many independent tile
payloads with the same codec. The caller keeps one `DecoderContext` and one
`ScratchPool`, then calls the stateless tile helper repeatedly.

```rust
use ashlar_core::{DecoderContext, PixelFormat, TileBatchDecode};
use ashlar_jpeg::{JpegCodec, ScratchPool};

let mut ctx = DecoderContext::<ashlar_jpeg::DecoderContext>::new();
let mut pool = ScratchPool::new();

for tile in visible_tiles {
    JpegCodec::decode_tile(
        &mut ctx,
        &mut pool,
        tile,
        &mut output,
        stride,
        PixelFormat::Rgb8,
    )?;
}
```

Tile-batch helpers exist for full, ROI, scaled, and ROI+scaled decode. The
same source-coordinate ROI and reduced-grid coverage rules apply to tile-batch
ROI+scaled decode.

## Device Surfaces

Use `ImageDecodeDevice`, `ImageDecodeSubmit`, `TileBatchDecodeDevice`, or
`TileBatchDecodeSubmit` when a downstream pipeline wants a backend-tagged
surface. Completed operations return a `DeviceSurface`, which reports:

- backend kind
- dimensions
- pixel format
- byte length

Backend selection uses `BackendRequest`:

- `BackendRequest::Cpu` requires host-backed output.
- `BackendRequest::Auto` lets the adapter choose CPU or a device path. Auto is
  conservative and may return CPU-backed surfaces when benchmarks or shape
  support do not justify device execution.
- `BackendRequest::Metal` requires Metal execution or a Metal-backed upload on
  macOS. Unsupported explicit Metal requests return an error.
- `BackendRequest::Cuda` is a compatibility surface in `0.1.0`; explicit CUDA
  requests return unavailable before decode validation, while `Cpu` and `Auto`
  return CPU-backed host surfaces.

Callers should use explicit device requests only when they need that backend.
Use `Auto` for viewer paths where CPU fallback is acceptable.

## Error Contract

No decode path should fail silently. Unsupported formats, invalid regions,
too-small buffers, too-small strides, unavailable explicit backends, and row
sink aborts are returned as errors. Callers should handle `CodecError`
predicates for broad policy decisions and preserve detailed errors for logging.

## Current Validation Scope

Hosted CI validates CPU behavior, adapter fallback behavior, rustdoc, and
benchmark compilation. Runtime GPU validation is available through the manual
`.github/workflows/gpu-validation.yml` workflow on self-hosted runners:

- Apple Silicon runners labeled `self-hosted`, `macOS`, `ARM64`, `metal`
  validate Metal tests and optionally timed Metal benchmarks.
- x86_64 CUDA runners labeled `self-hosted`, `Linux`, `X64`, `cuda` validate
  CUDA adapter compatibility. They do not imply runtime CUDA decode support
  until the CUDA crates grow a real CUDA backend.
