# signinum-j2k-cuda

CUDA-facing device-output adapter for `signinum-j2k`.

Install this crate when a pipeline needs JPEG 2000 / HTJ2K output copied into
CUDA device memory:

```sh
cargo add signinum-j2k-cuda --features cuda-runtime
```

`BackendRequest::Cpu` and `BackendRequest::Auto` return host-backed CPU
surfaces. `BackendRequest::Cuda` requires the `cuda-runtime` feature and an
available CUDA driver; when both are present, the adapter uploads
CPU-decoded JPEG 2000 / HTJ2K bytes into CUDA device memory and returns a
CUDA-backed `DeviceSurface`.

This crate does not provide CUDA kernel JPEG 2000 / HTJ2K decode and makes no
NVIDIA performance claim. The stable CPU decode API lives in `signinum-j2k`.
