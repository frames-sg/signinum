# signinum

Facade crate for the `signinum` pathology image codec workspace.

The default build exposes CPU-portable JPEG, JPEG 2000, shared core contracts,
tile decompression APIs, and the portable Metal adapter. Runtime backend
selection defaults to `Auto`: device paths are used for supported workloads when
compiled and available, with CPU as the fallback. CUDA remains available through
the explicit `cuda` or `gpu` features.
