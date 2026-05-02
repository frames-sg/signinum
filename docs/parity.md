# Parity Strategy

`signinum` keeps parity checks close to the codec surface instead of relying
on a single visual smoke test.

## JPEG

- Primary conformance fixtures live in `corpus/conformance/manifest.json` and
  compare decoded bytes against libjpeg-turbo-generated raw outputs.
- WSI-shaped fixtures and policy checks live in the `signinum-jpeg` test and
  bench suites.
- Tolerance is bit-exact for the committed baseline fixtures. Any future lossy
  tolerance must be recorded per fixture in the manifest.

## JPEG 2000 / HTJ2K

- CPU parity tests compare generated codestreams against the in-repo native
  engine and, where available, OpenJPEG/Grok comparator paths.
- ROI, scaled, combined ROI+scaled, row, and tile-batch surfaces are tested as
  API behavior, not only as full-frame decode.
- Metal and CUDA-named adapter crates must preserve CPU parity for fallback
  host surfaces. Metal crates must preserve decoded bytes for explicit
  Metal-backed ROI+scaled surfaces. CUDA upload-fallback surfaces must preserve
  decoded bytes after download. The nvJPEG full-frame RGB8 path in
  `signinum-jpeg-cuda` is checked against the CPU reference with a small
  vendor-IDCT tolerance and reports hardware decode separately from copy
  fallback stats.

## Maintenance Rules

- Every committed conformance input must be listed in the matching manifest.
- Fixture generation scripts are maintainer tools; CI reads committed fixtures
  and does not regenerate them.
- New codec behavior needs at least one focused parity or regression test
  before benchmark numbers are updated.
