# WSI/DICOM Passthrough Policy

This repository owns codec primitives and passthrough eligibility contracts,
not whole-slide container parsing or DICOM writing. WSI/DICOM conversion code
should use these primitives with a passthrough-first policy: preserve
compressed tile payloads when they are already legal for the requested
destination, and re-encode only when that is not possible.

## Decision Order

1. Inspect the source compressed tile bytes without decoding.
2. If the source codec, dimensions, sample layout, bit depth, color model,
   frame/tile order, and destination transfer syntax are compatible, copy the
   compressed tile bytes into the destination container unchanged.
3. If passthrough is not legal and diagnostic-preserving output is required,
   decode to pixels and encode a new lossless JPEG 2000 or HTJ2K codestream.
4. Use baseline JPEG encoding only for explicit non-diagnostic fallback cases:
   generated fixtures, preview/export derivatives, or a caller-requested legacy
   transfer syntax where lossy output is acceptable.

## Codec Responsibilities

- `signinum-core` provides `PassthroughCandidate`,
  `PassthroughRequirements`, compressed transfer syntax descriptors, payload
  kind descriptors, and typed rejection reasons. This is the shared contract
  container writers should use before choosing copy vs transcode.
- `signinum-jpeg` provides JPEG inspect/decode and a small baseline JPEG
  fallback encoder. `JpegView::passthrough_candidate()` exposes baseline and
  extended sequential JPEG interchange streams as borrowed copy candidates.
  JPEG encoding is not the WSI/DICOM hot path.
- `signinum-j2k` provides JPEG 2000 / HTJ2K inspect, decode, and lossless
  encode surfaces. `J2kView::passthrough_candidate()` classifies raw
  codestreams versus JP2 files and classic versus HT JPEG 2000 syntax when the
  native parser can prove that metadata. New diagnostic codestreams should use
  this path by default.
- Device adapters produce decoded device surfaces for viewer, QC, preprocessing,
  or GPU-resident downstream work. They do not make passthrough decisions.

## Passthrough Eligibility

A caller-owned container layer should require all of the following before
copying compressed bytes:

- destination transfer syntax matches the source codestream family and profile;
- frame dimensions, tile geometry, component count, bit depth, signedness, and
  planar/interleaved expectations are compatible;
- photometric interpretation and color transform expectations are compatible;
- the source payload is complete and independently inspectable by the matching
  codec parser;
- copying the payload preserves frame ordering and per-frame metadata expected
  by the destination container.

If any condition fails, the caller should choose an explicit transcode path and
record why passthrough was rejected.

```rust
use signinum_core::{
    CompressedPayloadKind, CompressedTransferSyntax, PassthroughRequirements,
};
use signinum_j2k::J2kView;

let view = J2kView::parse(tile_bytes)?;
let Some(candidate) = view.passthrough_candidate() else {
    // Parser could decode/inspect the payload but could not prove a copy-safe
    // transfer syntax. Decode and transcode instead.
    return Ok(CopyDecision::Transcode);
};
let requirements = PassthroughRequirements::new(
    CompressedTransferSyntax::HtJpeg2000Lossless,
    CompressedPayloadKind::Jpeg2000Codestream,
)
.with_dimensions(expected_dimensions)
.with_components(expected_components)
.with_bit_depth(expected_bit_depth);

match candidate.copy_bytes_if_eligible(&requirements) {
    Ok(bytes) => write_dicom_fragment(bytes)?,
    Err(reason) => transcode_and_record_reason(reason)?,
}
```

## Benchmark Signoff

JPEG decode and device-output benchmarks remain relevant for viewer and QC
paths. JPEG encode benchmarks are fallback diagnostics only. WSI/DICOM storage
signoff should prioritize passthrough eligibility coverage and J2K/HTJ2K
lossless encode throughput and parity.

## Local NDPI Regression Test

`signinum-jpeg` includes an optional local NDPI passthrough test. It does not
decode the whole-slide image and does not add NDPI container support to the
production crate. The test reads TIFF directories from a local NDPI file,
extracts JPEG-compressed payloads, and proves each eligible payload returns the
original borrowed bytes from either `JpegView::passthrough_candidate()` or the
container-level fallback for Hamamatsu JPEG strips whose embedded SOF
dimensions are zero and whose real dimensions live in the TIFF IFD.

```sh
SIGNINUM_NDPI_PATH=/path/to/slide.ndpi \
  cargo test -p signinum-jpeg --test ndpi_passthrough --profile release-bench -- --nocapture
```

Use `SIGNINUM_NDPI_TILE_LIMIT=0` for a full-container pass:

```sh
SIGNINUM_NDPI_PATH=/path/to/slide.ndpi \
SIGNINUM_NDPI_TILE_LIMIT=0 \
  cargo test -p signinum-jpeg --test ndpi_passthrough --profile release-bench -- --nocapture
```

Optional controls:

- `SIGNINUM_REQUIRE_NDPI=1` fails the test when `SIGNINUM_NDPI_PATH` is unset.
- `SIGNINUM_NDPI_TILE_LIMIT` defaults to `8`; `0` means all payloads.
- `SIGNINUM_NDPI_MAX_PAYLOAD_BYTES` defaults to `67108864` for bounded runs
  and no limit for full-container runs.
