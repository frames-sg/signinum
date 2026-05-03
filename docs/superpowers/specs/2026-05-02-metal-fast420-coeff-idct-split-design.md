# Metal Fast420 Coefficient/IDCT Split Design

## Goal

Add an experimental split fast420 full-batch Metal decode path that separates
entropy decoding from IDCT/deposit so WSI tile batches can expose more GPU
parallelism than the current one-thread-per-entropy-segment fused kernel.

## Scope

- Target only full-frame fast420 RGB batch decode.
- Keep the existing fused fast420 batch path as the default control path until
  benchmarks show the split path should replace it.
- Do not change 422, 444, region, scaled, or single-tile decode paths in this
  slice.

## Architecture

The split path adds two kernels. The first kernel keeps the existing entropy
thread scheduling, but writes dequantized block coefficients and a `dc_only`
flag into scratch buffers. The second kernel dispatches one thread per output
block, runs the existing scalar IDCT implementation for that block, and deposits
pixels into the existing Y/Cb/Cr planes. The existing batch RGB pack kernel then
runs unchanged.

Coefficient scratch is laid out per tile as all Y blocks, then Cb blocks, then
Cr blocks. For 420, the per-tile block counts are:

- Y: `mcus_per_row * mcu_rows * 4`
- Cb: `mcus_per_row * mcu_rows`
- Cr: `mcus_per_row * mcu_rows`

Each block stores 64 signed 16-bit coefficients. A parallel byte flag buffer
stores whether the block can use the DC-only IDCT path.

## Selection

The host path keeps the current fused decode unless
`SIGNINUM_JPEG_METAL_SPLIT_FAST420_BATCH=1` is set. This gives direct A/B
benchmarking without changing public APIs.

## Validation

- Add shader integrity guards so the split kernels and opt-in host switch remain
  present.
- Add a split-path correctness test for fast420 full-batch decode against CPU
  bytes.
- Benchmark generated `512x512` and `1024x1024` batch64 with the existing fused
  path and the split opt-in path.
