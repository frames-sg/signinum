# signinum-jpeg-cuda

CUDA-facing device-output adapter for `signinum-jpeg`.

Install this crate when a pipeline needs JPEG output in CUDA device memory:

```sh
cargo add signinum-jpeg-cuda --features cuda-runtime
```

`BackendRequest::Cpu` and `BackendRequest::Auto` return host-backed CPU
surfaces. `BackendRequest::Cuda` requires the `cuda-runtime` feature and an
available CUDA driver. For full-frame RGB8 JPEG decode, the adapter uses
NVIDIA nvJPEG when `libnvjpeg` is available and returns a CUDA-backed
`DeviceSurface` without first decoding to a host RGB buffer. Region, scaled,
non-RGB8, and nvJPEG-unsupported requests fall back to CPU decode plus CUDA
device-memory upload.

Use `cargo bench -p signinum-jpeg-cuda --bench device_decode --features
cuda-runtime` on an NVIDIA host to compare CPU decode, nvJPEG surface
production through a reused `CudaSession`, and decode-plus-download timing.
Set `SIGNINUM_GPU_BENCH_DIM=4096` for the generated large-tile benchmark, or
set `SIGNINUM_CUDA_BENCH_JPEG` to a large WSI-shaped JPEG tile. The same bench
also compares a CPU batch loop with nvJPEG batched RGB8 decode; tune it with
`SIGNINUM_GPU_BENCH_BATCH` and `SIGNINUM_GPU_BENCH_BATCH_DIM`.

The stable CPU decode API lives in `signinum-jpeg`.
