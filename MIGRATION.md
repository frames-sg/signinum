# Current Package Names

The public documentation now uses only the current Signinum and Statumen
package names. New projects should choose from these crates:

| Need | Package |
| --- | --- |
| Facade codec API | `signinum` |
| JPEG tile inspect/decode | `signinum-jpeg` |
| JPEG 2000 / HTJ2K inspect/decode/encode | `signinum-j2k` |
| Tile decompression primitives | `signinum-tilecodec` |
| Shared codec traits and pixel/backend types | `signinum-core` |
| Command-line inspection | `signinum-cli` |
| Whole-slide container reading | `statumen` |
| DICOM VL Whole Slide Microscopy export | `wsi-dicom` |

Prefer `signinum` when an application wants a single import surface for codec
primitives. Prefer the narrower crates when a downstream package wants a small
dependency graph or a specific codec API.

## Cargo Examples

```toml
[dependencies]
signinum = "1.2.3"
```

```toml
[dependencies]
signinum-jpeg = "1.1"
signinum-j2k = "1.2"
signinum-tilecodec = "1"
```

For CUDA device-memory output:

```toml
[dependencies]
signinum-jpeg-cuda = { version = "0.3", features = ["cuda-runtime"] }
signinum-j2k-cuda = { version = "0.3", features = ["cuda-runtime"] }
```

The CUDA adapters return CUDA device-memory surfaces when the runtime feature,
a CUDA driver, and the relevant runtime libraries are available. Unsupported
shapes fall back only where that backend's API explicitly documents fallback
behavior.
