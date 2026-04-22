# JPEG Metal Batch And Checkpoint Execution

Status: approved implementation spec derived from the current JPEG WSI and Metal work.

## Goal

Improve `slidecodec-jpeg-metal` for WSI-style JPEG workloads on Apple Silicon by:

- batching multiple tile decodes into one Metal submission
- adding planned non-restart entropy checkpoints so GPU workers can start from legal segment states
- keeping coefficient, reconstruction, and pack work on device for the whole batch
- avoiding fine-grained synchronous waits
- reusing packet planning, scratch buffers, and pipelines across repeated tile batches

The public API stays unchanged.

## Scope

In scope:

- private batch execution inside `slidecodec-jpeg-metal`
- private checkpoint planning for restart and non-restart baseline JPEG
- session-backed reuse of Metal pipelines, buffers, and packet storage
- faster repeated tile-batch execution for WSI-shaped JPEG workloads
- benchmark framing that treats CPU and Metal as separate first-class outcomes

Out of scope:

- `slidecodec-core` public API changes
- new public async or explicit batch submission APIs
- container or vendor-specific WSI extraction logic
- claims that Metal must beat CPU for every non-restart single-tile JPEG request
- progressive, arithmetic, or unsupported JPEG mode expansion

## Architecture

The implementation remains internal to `slidecodec-jpeg-metal`.

- `slidecodec-core` traits stay unchanged.
- `wsi-rs` or other callers continue to use the existing decode and submit APIs.
- `slidecodec-jpeg-metal` gains a private batch layer with three roles:
  - `BatchPlanner`: groups compatible requests into one internal batch
  - `CheckpointPlanner`: builds legal start states for restart-coded and non-restart scans
  - `BatchExecutor`: records and runs the Metal work for the full batch

Compatibility for first-cut batching is restricted to requests with the same execution shape:

- same sampling family
- same output family
- same scale mode
- same region/layout class

Requests that cannot safely share an execution plan still go through the same machinery as single-item batches.

## Execution Model

### 1. Internal batch submission

Repeated tile decode in `slidecodec-jpeg-metal` should stop behaving as N unrelated device calls.

- Add a private multi-request execution path under the existing tile APIs.
- Route repeated tile-batch work through one command buffer per internal batch.
- Public blocking APIs still block, but only once per batch instead of once per tile.

### 2. Restart and non-restart checkpoint planning

Restart-coded JPEG uses restart markers as the natural parallel boundary.

Non-restart JPEG needs synthetic checkpoints.

Each checkpoint record must capture:

- MCU start index
- byte offset into the entropy stream
- buffered bit-reader state
- DC predictor state per component
- restart expectations when applicable

The planner should emit checkpoints at a fixed MCU cadence for the first cut. GPU workers then decode their assigned segment from a legal precomputed state rather than from scan start.

### 3. Device-resident batch pipeline

The Metal path should keep the whole batch on device:

- entropy decode writes coefficient buffers
- reconstruction kernels consume those coefficients directly
- pack kernels write final output surfaces directly

There should be no host materialization of intermediate planes inside the batched Metal path.

### 4. Fewer waits

Avoid `commit()` + `wait_until_completed()` at overly fine granularity.

- one internal batch should produce one command buffer and one completion wait
- single-tile public calls may still execute as a batch of one
- the implementation should be structured so later overlap or async promotion is possible without redesigning the planner

### 5. Session-backed reuse

`MetalSession` becomes the reuse boundary, not just a submission counter.

It should cache:

- compute pipelines
- table uploads
- checkpoint packet storage
- reusable coefficient, reconstruction, and pack buffers
- high-water-mark scratch allocations

Repeated WSI batches should reuse these resources instead of rebuilding them on every decode.

## Correctness Constraints

This design is an execution rewrite, not a semantic rewrite.

- exact decode results must stay unchanged
- `BackendRequest::Metal` stays strict
- `BackendRequest::Auto` keeps existing fallback semantics
- checkpoint execution must preserve the same entropy state transitions as the CPU path
- unsupported batching shapes must fail back to the existing single-item internal execution path, not to hidden CPU execution

The design does not claim arbitrary byte-parallel Huffman decode for non-restart JPEG. Checkpoints create legal worker start states; they do not make baseline JPEG embarrassingly parallel.

## Performance And Evaluation

The benchmark and paper framing should remain honest.

- Apple Silicon CPU remains a first-class result, especially for small or non-restart single-tile latency.
- Metal is evaluated as an accelerator where enough parallel structure exists:
  - restart-coded JPEG
  - repeated tile batches
  - device-resident downstream work
  - JPEG2000 / HTJ2K, which remain the stronger GPU story

Success for this milestone is:

- measurable improvement on Metal JPEG WSI tile-batch benchmarks
- no regression in public behavior
- CPU path remains independently optimized and not de-emphasized
- paper claims describe Metal as workload-dependent acceleration, not universal JPEG dominance

## Verification

This milestone is complete when all of the following pass:

- `cargo test -p slidecodec-jpeg-metal`
- `cargo test -p slidecodec-jpeg`
- `cargo clippy -p slidecodec-jpeg -p slidecodec-jpeg-metal --all-targets -- -D warnings`
- targeted WSI-style JPEG Metal benches on Apple Silicon

Required benchmark checks:

- repeated tile-batch on restart-coded JPEG
- repeated tile-batch on non-restart JPEG
- region and scaled region on representative WSI-shaped inputs

The milestone exit criterion is not “Metal beats CPU everywhere.” The criterion is that the Metal path removes avoidable submission and staging losses, and measurably improves the batch-oriented WSI cases it is designed to accelerate.
