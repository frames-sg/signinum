# Adaptive Codec Runtime Handoff

## Branch

- `adaptive-codec-runtime`

## Current Goal State

The repo is no longer in the early `CPU decode + Metal pack` state for J2K/HTJ2K throughput work.

What is true now:

- `MetalDirect` is real for grayscale J2K/HTJ2K.
- `MetalDirect` wins the meaningful `1024x1024` throughput lanes.
- `Auto` / adaptive routing is evidence-driven and chooses CPU for CPU-owned lanes and Metal for GPU-owned lanes.

What is not true yet:

- `MetalDirect` does **not** win single-request full decode lanes.
- `region` / `scaled` J2K/HTJ2K MetalDirect are still not first-class GPU-native lanes.

## Latest Code Slice

This latest slice focused on reducing single-request MetalDirect fixed cost without regressing the already-winning throughput lanes.

Main files touched:

- [compute.rs](/Users/user/.config/superpowers/worktrees/signinum/adaptive-codec-runtime/crates/signinum-j2k-metal/src/compute.rs)
- [lib.rs](/Users/user/.config/superpowers/worktrees/signinum/adaptive-codec-runtime/crates/signinum-j2k-metal/src/lib.rs)
- [store.metal](/Users/user/.config/superpowers/worktrees/signinum/adaptive-codec-runtime/crates/signinum-j2k-metal/src/store.metal)

Key changes:

- `decode_direct_to_surface(...)` now uses the prepared direct grayscale plan path instead of the generic repeated path with `count=1`.
- single-request MetalDirect now has a true prepared executor instead of routing through repeated kernels by default.
- single grayscale tail is fused:
  - new single-store-to-Gray8 / Gray16 kernels
  - removed the extra single-request float-store-plus-pack tail
- prepared classic / HT sub-bands now carry reusable Metal buffers for coded payloads and job tables, so single decode does not rebuild those Metal buffer wrappers every call.
- throughput path remains intact:
  - repeated classic / HT batch wins were preserved
  - adaptive routing thresholds remain aligned to the measured matrix

## Verified Commands

- `cargo fmt --all`
- `cargo test -p signinum-j2k-metal --lib --tests`
- `cargo clippy -p signinum-j2k-metal -p signinum-j2k --all-targets -- -D warnings`
- `cargo bench -p signinum-j2k --bench compare -- 'decode_gray/(signinum|signinum-metal|signinum-adaptive)/(htj2k_gray_1024|j2k_gray_1024)' --quick --noplot`
- `cargo bench -p signinum-j2k --bench compare -- 'wsi_tile_batch_gray/(signinum|signinum-metal|signinum-adaptive)/(j2k_gray_1024|htj2k_gray_1024)' --quick --noplot`

## Latest Measured Frontier

### Single-request `decode_gray`

- classic `J2K 1024`
  - CPU: `15.973-16.398 ms`
  - adaptive: `15.394-15.966 ms`
  - MetalDirect: `250.31-250.35 ms`
- `HTJ2K 1024`
  - CPU: `2.0606-2.0660 ms`
  - adaptive: `2.0562-2.0923 ms`
  - MetalDirect: `6.9925-7.0246 ms`

### Throughput `wsi_tile_batch_gray`

- classic `J2K 1024`
  - CPU: `246.19-251.66 ms`
  - adaptive: `241.20-241.60 ms`
  - MetalDirect: `241.51-242.05 ms`
- `HTJ2K 1024`
  - CPU: `32.836-32.860 ms`
  - adaptive: `11.398-11.447 ms`
  - MetalDirect: `11.448-11.465 ms`

## What Was Cleaned Up

- removed the stale `docs/superpowers/specs`
- removed the stale `docs/superpowers/plans`

The intention is that this file is now the only current planning / handoff artifact under `docs/superpowers`.

## Actual Remaining Frontier

The batch-side MetalDirect work is in good shape for grayscale `1024` throughput. The remaining losses are now concentrated in **single-request fixed cost** and **non-batch request shapes**.

The main reason `HTJ2K decode_gray 1024` still loses is not kernel throughput anymore. The batch numbers prove the GPU kernels are strong when work is wide enough. The remaining loss is per-request structure:

- too many per-step encoders for a single tile
- per-sub-band cleanup stages still dispatched separately
- final synchronous command-buffer completion for a very small unit of work

## Next Best Targets

If continuing in a new chat, the next work should be:

1. Collapse single-request grayscale MetalDirect into fewer device stages.
   - The next real win target is `HTJ2K decode_gray 1024`.
   - The likely path is a more global single-tile executor:
     - one coefficient arena
     - one broader HT cleanup dispatch across sub-bands
     - fewer encoder boundaries before IDWT/store

2. Push GPU-native `region` / `scaled`.
   - Current explicit MetalDirect is still grayscale full decode first-cut.
   - The next logical extension is ROI / scaled request-local planning.

3. Consider output / status pooling only if single-request fixed cost still dominates after the broader executor rewrite.
   - The current private scratch pool is already in place.
   - Shared output/status pooling is only worth the complexity if the broader single executor still leaves a large fixed-cost gap.

4. Keep adaptive routing honest.
   - CPU should continue to own the small single-request lanes while MetalDirect owns the throughput lanes.
   - Do not force Metal into lanes where the matrix still says CPU is better.

## Current Recommendation For New Chat

Start from this branch and file, not from any old plan documents.

Immediate next target:

- `HTJ2K decode_gray 1024`

Immediate non-goal:

- do **not** spend time on old superpowers plan/spec archaeology
- do **not** rework JPEG here unless the new chat explicitly returns to JPEG
