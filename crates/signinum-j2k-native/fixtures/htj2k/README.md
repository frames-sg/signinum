HTJ2K fixtures for native decoder coverage.

These fixtures are from the OpenHTJ2K conformance data, copied from OpenHTJ2K
commit `ffe5acf9f1eedb87c36c3fd2134fdc1ddea5e75f`. They are tiny HTONLY
codestreams derived from the JPEG 2000 Part 4 / ITU-T T.803 HTJ2K conformance
set.

`openhtj2k_ds0_ht_12_b11.j2k` is copied from `ds0_ht_12_b11.j2k`, blob
`cf3fb0bc7e55898b4e6977f38ba0d38d91c359bf`. The native decoder sees 8 HT code
blocks, 2 non-empty refinement jobs, and up to 3 HT coding passes.

`openhtj2k_ds0_ht_09_b11.j2k` is copied from `ds0_ht_09_b11.j2k`, blob
`d4f2031359c32eb24825d00dde05a92cf3ae451e`. The native decoder sees 14 HT
code blocks, all with non-empty refinement jobs, and up to 3 HT coding passes.

It is intentionally checked-in test data: tests must not invoke an external
encoder or decoder at runtime.

The paired `.gray` file contains the expected 8-bit grayscale samples in
row-major order from decoding the checked-in codestream with OpenJPH.

The OpenHTJ2K source license is retained in `LICENSE.OpenHTJ2K`.
