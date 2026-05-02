# signinum-jpeg-cuda

CUDA-facing device-output adapter for `signinum-jpeg`.

Install this crate when a pipeline needs JPEG tile output copied into CUDA
device memory:

```sh
cargo add signinum-jpeg-cuda --features cuda-runtime
```

`BackendRequest::Cpu` and `BackendRequest::Auto` return host-backed CPU
surfaces. `BackendRequest::Cuda` requires the `cuda-runtime` feature and an
available CUDA driver; when both are present, the adapter uploads
CPU-decoded JPEG bytes into CUDA device memory and returns a CUDA-backed
`DeviceSurface`.

This crate does not provide CUDA kernel JPEG decode and makes no NVIDIA
performance claim. The stable CPU decode API lives in `signinum-jpeg`.
