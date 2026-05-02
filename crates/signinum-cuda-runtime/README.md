# signinum-cuda-runtime

CUDA Driver API runtime helpers for the `signinum` CUDA adapter crates.

Most downstream users should depend on `signinum-jpeg-cuda` or
`signinum-j2k-cuda` instead of using this crate directly. This crate owns the
small runtime layer used by those adapters to allocate CUDA device memory,
copy bytes between host and device, and report CUDA driver errors clearly.

It does not decode JPEG or JPEG 2000 codestreams. The current CUDA adapter path
uploads CPU-decoded bytes into CUDA device memory and does not provide CUDA
kernel decode or NVIDIA performance claims.
