# Parity Strategy

`slidecodec` keeps parity checks close to the codec surface instead of relying
on a single visual smoke test.

## JPEG

- Primary conformance fixtures live in `corpus/conformance/manifest.json` and
  compare decoded bytes against libjpeg-turbo-generated raw outputs.
- WSI-shaped fixtures and policy checks live in the `slidecodec-jpeg` test and
  bench suites.
- Tolerance is bit-exact for the committed baseline fixtures. Any future lossy
  tolerance must be recorded per fixture in the manifest.

## JPEG 2000 / HTJ2K

- CPU parity tests compare generated codestreams against the in-repo native
  engine and, where available, OpenJPEG/Grok comparator paths.
- ROI, scaled, row, and tile-batch surfaces are tested as API behavior, not
  only as full-frame decode.
- Metal and CUDA-named adapter crates must preserve CPU parity for fallback
  host surfaces. CUDA crates do not claim runtime CUDA execution in `0.1.0`.

## Maintenance Rules

- Every committed conformance input must be listed in the matching manifest.
- Fixture generation scripts are maintainer tools; CI reads committed fixtures
  and does not regenerate them.
- New codec behavior needs at least one focused parity or regression test
  before benchmark numbers are updated.
