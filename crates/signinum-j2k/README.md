# signinum-j2k

JPEG 2000 / HTJ2K inspect and CPU decode for whole-slide imaging workloads.

The CPU-first 1.0 surface covers borrowed inspect/parse, full-frame decode,
ROI decode, reduced-resolution decode, combined ROI+reduced-resolution decode,
row-bounded decode, and tile-batch decode through the shared `signinum-core`
traits.

GPU adapter crates are versioned separately and are not part of the CPU-first
1.0 stability promise.
