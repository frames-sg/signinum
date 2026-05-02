# signinum-j2k-native

Implementation crate for `signinum-j2k`.

Most users should install `signinum-j2k` instead:

```sh
cargo add signinum-j2k
```

This crate contains the pure-Rust JPEG 2000 / HTJ2K engine used by the public
`signinum-j2k` CPU-first 1.0 API. It is published so `signinum-j2k` can be consumed
from crates.io, but downstream users should prefer the stable `signinum-j2k`
wrapper unless they intentionally need engine-level APIs.
