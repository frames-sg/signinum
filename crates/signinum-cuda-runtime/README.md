# signinum-cuda-runtime

CUDA Driver API runtime helpers for the `signinum` CUDA adapter crates.

Most downstream users should depend on `signinum-jpeg-cuda` or
`signinum-j2k-cuda` instead of using this crate directly. This crate owns the
small runtime layer used by those adapters to allocate CUDA device memory,
copy bytes between host and device, launch bundled CUDA kernels, call nvJPEG
when it is available, and report CUDA driver errors clearly.

The runtime currently exposes full-frame RGB8 JPEG decode through NVIDIA
nvJPEG for `signinum-jpeg-cuda`. JPEG 2000 / HTJ2K CUDA adapters still upload
CPU-decoded bytes into CUDA device memory and do not provide CUDA codestream
decode kernels.
