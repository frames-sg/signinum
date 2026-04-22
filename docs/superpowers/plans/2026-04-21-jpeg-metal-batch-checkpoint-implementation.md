# JPEG Metal Batch Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** implement session-driven Metal JPEG tile batching and non-restart entropy checkpoint planning for WSI workloads without changing the public host API surface.

**Architecture:** add a narrow hidden JPEG device-planning seam in `slidecodec-jpeg`, including cached restart and synthetic non-restart checkpoints, then replace the eager `ReadySubmission` Metal path with a real queued batch executor in `slidecodec-jpeg-metal`. Compatible submits share one command buffer, one completion wait, and one session-owned scratch/cache boundary while leaving blocking host APIs as batch-size-one wrappers.

**Tech Stack:** Rust, Metal, Criterion, `slidecodec-core`, `slidecodec-jpeg`, `slidecodec-jpeg-metal`

---

## File Structure

- Create: `crates/slidecodec-jpeg/src/__private/mod.rs`
  - Hidden cross-crate seam used only by device adapters. Exposes adapter-friendly JPEG decode descriptors without changing the documented public host API.
- Create: `crates/slidecodec-jpeg/src/__private/device_plan.rs`
  - Builds Metal-ready JPEG metadata: scan slice, sampling/output metadata, quant/Huffman copies, and checkpoint plans.
- Create: `crates/slidecodec-jpeg/src/internal/checkpoint.rs`
  - Owns restart and synthetic non-restart checkpoint planning plus checkpoint cache keys.
- Modify: `crates/slidecodec-jpeg/src/internal/mod.rs`
  - Wires the new checkpoint module into decoder internals.
- Modify: `crates/slidecodec-jpeg/src/internal/bit_reader.rs`
  - Adds snapshot/restore helpers for checkpoint prepass and replay.
- Modify: `crates/slidecodec-jpeg/src/context.rs`
  - Extends `DecoderContext` with checkpoint/device-plan caches keyed by header prefix + scan payload + cadence.
- Modify: `crates/slidecodec-jpeg/src/lib.rs`
  - Exports `#[doc(hidden)] pub mod __private;`.
- Test: `crates/slidecodec-jpeg/tests/device_plan.rs`
  - Behavior tests for the hidden device plan seam and checkpoint cache behavior.

- Create: `crates/slidecodec-jpeg-metal/src/session.rs`
  - Owns session-backed runtime cache, reusable Metal buffers, queued requests, and batch flush state.
- Create: `crates/slidecodec-jpeg-metal/src/batch.rs`
  - Owns request grouping, batch keys, queued submission handles, and one-command-buffer flush orchestration.
- Modify: `crates/slidecodec-jpeg-metal/src/compute.rs`
  - Stops doing per-call `commit()` + `wait_until_completed()`. Encodes work into a caller-owned command buffer and writes results into caller-owned session buffers.
- Modify: `crates/slidecodec-jpeg-metal/src/lib.rs`
  - Replaces `ReadySubmission` with a queued `MetalSubmission`, upgrades `MetalSession` into the cache boundary, and routes submit APIs through the batch executor.
- Test: `crates/slidecodec-jpeg-metal/tests/core_traits.rs`
  - Verifies queued submit semantics and strict backend behavior remain correct.
- Test: `crates/slidecodec-jpeg-metal/tests/batch.rs`
  - Verifies batching parity, grouping, and flush counts on repeated tile requests.

- Create: `crates/slidecodec-jpeg-metal/benches/compare.rs`
  - WSI-style Metal benchmark that uses the session-based submit API instead of one eager decode per tile.
- Modify: `crates/slidecodec-jpeg-metal/Cargo.toml`
  - Registers the new compare bench target.

### Task 1: Add the hidden JPEG device-plan seam

**Files:**
- Create: `crates/slidecodec-jpeg/src/__private/mod.rs`
- Create: `crates/slidecodec-jpeg/src/__private/device_plan.rs`
- Modify: `crates/slidecodec-jpeg/src/lib.rs`
- Test: `crates/slidecodec-jpeg/tests/device_plan.rs`

- [ ] **Step 1: Write the failing device-plan tests**

```rust
use slidecodec_jpeg::{ColorSpace, Decoder};

const BASELINE_420: &[u8] =
    include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");

#[test]
fn hidden_device_plan_exposes_scan_metadata() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let plan = slidecodec_jpeg::__private::build_device_plan(&decoder, 4)
        .expect("device plan");

    assert_eq!(plan.dimensions, (16, 16));
    assert_eq!(plan.color_space, ColorSpace::YCbCr);
    assert_eq!(plan.components.len(), 3);
    assert_eq!(plan.checkpoints[0].mcu_index, 0);
    assert!(!plan.scan_bytes.is_empty());
}

#[test]
fn hidden_device_plan_keeps_fast_420_shape_information() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let plan = slidecodec_jpeg::__private::build_device_plan(&decoder, 4)
        .expect("device plan");

    assert!(plan.matches_fast_420);
    assert!(!plan.matches_fast_444);
}
```

- [ ] **Step 2: Run the new tests to verify the seam does not exist yet**

Run: `cargo test -p slidecodec-jpeg --test device_plan`

Expected: FAIL with errors that `slidecodec_jpeg::__private` and `build_device_plan` do not exist.

- [ ] **Step 3: Add the hidden adapter module and the device-plan builder**

```rust
// crates/slidecodec-jpeg/src/lib.rs
#[doc(hidden)]
pub mod __private;
```

```rust
// crates/slidecodec-jpeg/src/__private/mod.rs
mod device_plan;

pub use device_plan::{
    DeviceCheckpoint, DeviceComponentPlan, DeviceDecodePlan, build_device_plan,
};
```

```rust
// crates/slidecodec-jpeg/src/__private/device_plan.rs
use alloc::sync::Arc;

use crate::{
    decoder::Decoder,
    error::JpegError,
    info::ColorSpace,
    internal::checkpoint::{build_checkpoint_plan, DeviceCheckpoint},
};

#[derive(Debug, Clone)]
pub struct DeviceComponentPlan {
    pub h: u8,
    pub v: u8,
    pub output_index: usize,
}

#[derive(Debug, Clone)]
pub struct DeviceDecodePlan {
    pub dimensions: (u32, u32),
    pub color_space: ColorSpace,
    pub restart_interval: Option<u16>,
    pub scan_bytes: Arc<[u8]>,
    pub components: Arc<[DeviceComponentPlan]>,
    pub checkpoints: Arc<[DeviceCheckpoint]>,
    pub matches_fast_420: bool,
    pub matches_fast_444: bool,
}

pub fn build_device_plan(decoder: &Decoder<'_>, cadence_mcus: u32) -> Result<DeviceDecodePlan, JpegError> {
    let plan = &decoder.plan;
    let scan_bytes = Arc::<[u8]>::from(&decoder.bytes[plan.scan_offset..]);
    let checkpoints = build_checkpoint_plan(plan, &scan_bytes, cadence_mcus)?;
    let components = plan
        .components
        .iter()
        .map(|component| DeviceComponentPlan {
            h: component.h,
            v: component.v,
            output_index: component.output_index,
        })
        .collect::<Vec<_>>();
    Ok(DeviceDecodePlan {
        dimensions: plan.dimensions,
        color_space: plan.color_space,
        restart_interval: plan.restart_interval,
        scan_bytes,
        components: components.into(),
        checkpoints,
        matches_fast_420: plan.matches_fast_tile_shape(),
        matches_fast_444: plan.matches_fast_rgb444_shape(),
    })
}
```

- [ ] **Step 4: Re-run the hidden seam tests**

Run: `cargo test -p slidecodec-jpeg --test device_plan`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add \
  crates/slidecodec-jpeg/src/lib.rs \
  crates/slidecodec-jpeg/src/__private/mod.rs \
  crates/slidecodec-jpeg/src/__private/device_plan.rs \
  crates/slidecodec-jpeg/tests/device_plan.rs
git commit -m "feat: add jpeg hidden device plan seam"
```

### Task 2: Implement cached restart and non-restart checkpoints

**Files:**
- Create: `crates/slidecodec-jpeg/src/internal/checkpoint.rs`
- Modify: `crates/slidecodec-jpeg/src/internal/mod.rs`
- Modify: `crates/slidecodec-jpeg/src/internal/bit_reader.rs`
- Modify: `crates/slidecodec-jpeg/src/context.rs`
- Modify: `crates/slidecodec-jpeg/src/__private/device_plan.rs`

- [ ] **Step 1: Write failing unit tests for checkpoint snapshot/replay and caching**

```rust
// crates/slidecodec-jpeg/src/internal/checkpoint.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::bit_reader::BitReader;

    #[test]
    fn bit_reader_snapshot_roundtrips_state() {
        let mut reader = BitReader::new(&[0b1011_0010, 0b0101_0000]);
        let _ = reader.read_bits(5).expect("first bits");
        let snapshot = reader.snapshot();
        let expected = reader.read_bits(3).expect("expected bits");

        let mut restored = BitReader::from_snapshot(&[0b1011_0010, 0b0101_0000], snapshot);
        let replayed = restored.read_bits(3).expect("replayed bits");

        assert_eq!(replayed, expected);
    }

    #[test]
    fn synthetic_checkpoints_are_monotonic() {
        let plan = fixture_plan_without_restart();
        let checkpoints = build_checkpoint_plan(&plan, b"fixture-scan", 2).expect("checkpoints");

        assert!(checkpoints.len() >= 2);
        assert!(checkpoints.windows(2).all(|pair| {
            pair[0].mcu_index < pair[1].mcu_index
                && pair[0].scan_offset <= pair[1].scan_offset
        }));
    }
}
```

```rust
// crates/slidecodec-jpeg/src/context.rs
#[test]
fn checkpoint_cache_returns_same_arc_for_same_scan_and_cadence() {
    let mut ctx = DecoderContext::new();
    let first = ctx.resolve_checkpoint_plan(b"header", b"scan", 4, |_| {
        Ok(Arc::<[DeviceCheckpoint]>::from(vec![DeviceCheckpoint::default()]))
    }).expect("first");
    let second = ctx.resolve_checkpoint_plan(b"header", b"scan", 4, |_| {
        unreachable!("cache hit");
    }).expect("second");

    assert!(Arc::ptr_eq(&first, &second));
}
```

- [ ] **Step 2: Run the narrow tests to verify the checkpoint machinery is missing**

Run: `cargo test -p slidecodec-jpeg checkpoint`

Expected: FAIL with missing `snapshot`, `from_snapshot`, `resolve_checkpoint_plan`, and `build_checkpoint_plan`.

- [ ] **Step 3: Add checkpoint types, BitReader snapshots, and the context cache**

```rust
// crates/slidecodec-jpeg/src/internal/bit_reader.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BitReaderSnapshot {
    pub offset: usize,
    pub bit_buffer: u64,
    pub bits_left: u8,
}

impl<'a> BitReader<'a> {
    pub(crate) fn snapshot(&self) -> BitReaderSnapshot {
        BitReaderSnapshot {
            offset: self.offset,
            bit_buffer: self.bit_buffer,
            bits_left: self.bits_left,
        }
    }

    pub(crate) fn from_snapshot(bytes: &'a [u8], snapshot: BitReaderSnapshot) -> Self {
        Self {
            bytes,
            offset: snapshot.offset,
            bit_buffer: snapshot.bit_buffer,
            bits_left: snapshot.bits_left,
        }
    }
}
```

```rust
// crates/slidecodec-jpeg/src/internal/checkpoint.rs
use alloc::sync::Arc;

use crate::{
    entropy::sequential::PreparedDecodePlan,
    error::JpegError,
    internal::bit_reader::{BitReader, BitReaderSnapshot},
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeviceCheckpoint {
    pub mcu_index: u32,
    pub scan_offset: usize,
    pub bit_reader: BitReaderSnapshot,
    pub prev_dc: [i16; 4],
    pub expected_rst: u8,
}

pub(crate) fn build_checkpoint_plan(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    cadence_mcus: u32,
) -> Result<Arc<[DeviceCheckpoint]>, JpegError> {
    if let Some(restart) = plan.restart_interval {
        return Ok(build_restart_checkpoints(plan, restart));
    }
    build_synthetic_checkpoints(plan, scan_bytes, cadence_mcus)
}

fn build_restart_checkpoints(
    plan: &PreparedDecodePlan,
    restart: u16,
) -> Arc<[DeviceCheckpoint]> {
    let mcu_width = u32::from(plan.sampling.max_h) * 8;
    let mcu_height = u32::from(plan.sampling.max_v) * 8;
    let mcus_per_row = plan.dimensions.0.div_ceil(mcu_width);
    let mcu_rows = plan.dimensions.1.div_ceil(mcu_height);
    let total_mcus = mcus_per_row * mcu_rows;
    let interval = u32::from(restart).max(1);
    (0..total_mcus)
        .step_by(interval as usize)
        .map(|mcu_index| DeviceCheckpoint {
            mcu_index,
            expected_rst: ((mcu_index / interval) & 7) as u8,
            ..DeviceCheckpoint::default()
        })
        .collect::<Vec<_>>()
        .into()
}

fn build_synthetic_checkpoints(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    cadence_mcus: u32,
) -> Result<Arc<[DeviceCheckpoint]>, JpegError> {
    let cadence_mcus = cadence_mcus.max(1);
    let mut reader = BitReader::new(scan_bytes);
    let mut prev_dc = [0i16; 4];
    let mut out = vec![DeviceCheckpoint::default()];
    let total_mcus = plan.dimensions.0.div_ceil(u32::from(plan.sampling.max_h) * 8)
        * plan.dimensions.1.div_ceil(u32::from(plan.sampling.max_v) * 8);
    for mcu_index in 0..total_mcus {
        walk_one_mcu(plan, &mut reader, &mut prev_dc, mcu_index)?;
        if (mcu_index + 1) % cadence_mcus == 0 && mcu_index + 1 < total_mcus {
            out.push(DeviceCheckpoint {
                mcu_index: mcu_index + 1,
                scan_offset: reader.snapshot().offset,
                bit_reader: reader.snapshot(),
                prev_dc,
                expected_rst: 0,
            });
        }
    }
    Ok(out.into())
}
```

```rust
// crates/slidecodec-jpeg/src/context.rs
pub(crate) fn resolve_checkpoint_plan<F>(
    &mut self,
    header_prefix: &[u8],
    scan_bytes: &[u8],
    cadence_mcus: u32,
    build: F,
) -> Result<Arc<[DeviceCheckpoint]>, JpegError>
where
    F: FnOnce(&mut Self) -> Result<Arc<[DeviceCheckpoint]>, JpegError>,
{
    // cache key = header digest + scan digest + cadence
}
```

- [ ] **Step 4: Re-run the checkpoint-focused tests**

Run:
- `cargo test -p slidecodec-jpeg checkpoint`
- `cargo test -p slidecodec-jpeg --test device_plan`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add \
  crates/slidecodec-jpeg/src/internal/mod.rs \
  crates/slidecodec-jpeg/src/internal/bit_reader.rs \
  crates/slidecodec-jpeg/src/internal/checkpoint.rs \
  crates/slidecodec-jpeg/src/context.rs \
  crates/slidecodec-jpeg/src/__private/device_plan.rs
git commit -m "feat: add jpeg checkpoint planning cache"
```

### Task 3: Split Metal session state and queued submission scaffolding

**Files:**
- Create: `crates/slidecodec-jpeg-metal/src/session.rs`
- Create: `crates/slidecodec-jpeg-metal/src/batch.rs`
- Modify: `crates/slidecodec-jpeg-metal/src/lib.rs`
- Modify: `crates/slidecodec-jpeg-metal/src/compute.rs`
- Test: `crates/slidecodec-jpeg-metal/tests/core_traits.rs`

- [ ] **Step 1: Write the failing session-based submit tests**

```rust
use slidecodec_core::{BackendRequest, DeviceSubmission, ImageDecodeSubmit, PixelFormat};
use slidecodec_jpeg_metal::{Decoder, MetalSession};

const BASELINE_420: &[u8] =
    include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");

#[test]
fn multiple_submits_share_one_session_flush() {
    let mut session = MetalSession::default();
    let mut a = Decoder::new(BASELINE_420).expect("decoder a");
    let mut b = Decoder::new(BASELINE_420).expect("decoder b");

    let first = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut a,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Metal,
    ).expect("submit a");
    let second = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut b,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Metal,
    ).expect("submit b");

    let _ = second.wait().expect("wait b");
    let _ = first.wait().expect("wait a");

    assert_eq!(session.submissions(), 1);
}
```

- [ ] **Step 2: Run the Metal core-traits test to verify the queued submission path is missing**

Run: `cargo test -p slidecodec-jpeg-metal --test core_traits`

Expected: FAIL because each submit currently increments `submissions()` and returns an eager `ReadySubmission`.

- [ ] **Step 3: Add `MetalSession`, `MetalSubmission`, and the queue shell**

```rust
// crates/slidecodec-jpeg-metal/src/session.rs
use std::sync::{Arc, Mutex};

use metal::Buffer;

#[derive(Default)]
pub(crate) struct ReusableBuffers {
    pub coeff: Vec<Buffer>,
    pub output: Vec<Buffer>,
}

#[derive(Default)]
pub(crate) struct SessionState {
    pub submissions: u64,
    pub queued: Vec<crate::batch::QueuedRequest>,
    pub completed: Vec<Option<crate::Surface>>,
    pub reusable: ReusableBuffers,
    pub runtime: Option<crate::compute::MetalRuntime>,
}

#[derive(Clone)]
pub(crate) struct SharedSession(pub Arc<Mutex<SessionState>>);

impl SessionState {
    pub(crate) fn queue_request(&mut self, request: crate::batch::QueuedRequest) -> usize {
        let slot = self.completed.len();
        self.completed.push(None);
        self.queued.push(request.with_output_slot(slot));
        slot
    }

    pub(crate) fn runtime(&mut self) -> Result<&crate::compute::MetalRuntime, crate::Error> {
        if self.runtime.is_none() {
            self.runtime = Some(crate::compute::MetalRuntime::new().map_err(|message| {
                crate::Error::MetalKernel { message }
            })?);
        }
        Ok(self.runtime.as_ref().expect("metal runtime"))
    }
}
```

```rust
// crates/slidecodec-jpeg-metal/src/batch.rs
use std::{collections::HashMap, sync::Arc};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum BatchOp {
    Full,
    Region,
    Scaled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SamplingFamily {
    Gray,
    YCbCr420,
    YCbCr444,
    Other,
}

pub(crate) enum RequestKind {
    Decoder,
    Tile,
}

pub(crate) struct QueuedRequest {
    pub key: BatchKey,
    pub plan: Arc<slidecodec_jpeg::__private::DeviceDecodePlan>,
    pub output_slot: usize,
    pub request: RequestKind,
}

impl QueuedRequest {
    pub(crate) fn with_output_slot(mut self, output_slot: usize) -> Self {
        self.output_slot = output_slot;
        self
    }
}

pub(crate) struct MetalSubmission {
    pub session: SharedSession,
    pub slot: usize,
}

impl DeviceSubmission for MetalSubmission {
    type Output = Surface;
    type Error = Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        let mut session = self.session.0.lock().expect("metal session");
        flush_if_needed(&mut session)?;
        take_surface(&mut session, self.slot)
    }
}

fn take_surface(session: &mut crate::session::SessionState, slot: usize) -> Result<Surface, Error> {
    session
        .completed
        .get_mut(slot)
        .and_then(Option::take)
        .ok_or_else(|| Error::MetalKernel {
            message: format!("missing queued Metal surface for slot {slot}"),
        })
}
```

```rust
// crates/slidecodec-jpeg-metal/src/lib.rs
#[derive(Clone, Default)]
pub struct MetalSession {
    shared: batch::SharedSession,
}

impl MetalSession {
    pub fn submissions(&self) -> u64 {
        self.shared.0.lock().expect("metal session").submissions
    }
}
```

- [ ] **Step 4: Re-run the Metal core-traits tests**

Run: `cargo test -p slidecodec-jpeg-metal --test core_traits`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add \
  crates/slidecodec-jpeg-metal/src/session.rs \
  crates/slidecodec-jpeg-metal/src/batch.rs \
  crates/slidecodec-jpeg-metal/src/lib.rs \
  crates/slidecodec-jpeg-metal/src/compute.rs \
  crates/slidecodec-jpeg-metal/tests/core_traits.rs
git commit -m "refactor: add metal queued submission session"
```

### Task 4: Batch compatible tile submits into one Metal flush

**Files:**
- Modify: `crates/slidecodec-jpeg-metal/src/batch.rs`
- Modify: `crates/slidecodec-jpeg-metal/src/compute.rs`
- Modify: `crates/slidecodec-jpeg-metal/src/lib.rs`
- Test: `crates/slidecodec-jpeg-metal/tests/batch.rs`
- Test: `crates/slidecodec-jpeg-metal/tests/core_traits.rs`

- [ ] **Step 1: Write failing integration tests for grouped tile submits**

```rust
use slidecodec_core::{
    BackendKind, BackendRequest, DecoderContext, DeviceSubmission, PixelFormat,
    TileBatchDecodeSubmit,
};
use slidecodec_jpeg::DecoderContext as JpegDecoderContext;
use slidecodec_jpeg_metal::{Codec, MetalSession, ScratchPool};

const BASELINE_420: &[u8] =
    include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");

#[test]
fn compatible_tile_submits_flush_once() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();

    let submissions = (0..4)
        .map(|_| {
            <Codec as TileBatchDecodeSubmit>::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                BASELINE_420,
                PixelFormat::Rgb8,
                BackendRequest::Metal,
            )
            .expect("submit")
        })
        .collect::<Vec<_>>();

    for submission in submissions {
        let surface = submission.wait().expect("surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
    }

    assert_eq!(session.submissions(), 1);
}

#[test]
fn incompatible_shapes_split_batches() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();

    let full = <Codec as TileBatchDecodeSubmit>::submit_tile_to_device(
        &mut ctx, &mut session, &mut pool, BASELINE_420, PixelFormat::Rgb8, BackendRequest::Metal,
    ).expect("full");
    let scaled = <Codec as TileBatchDecodeSubmit>::submit_tile_scaled_to_device(
        &mut ctx,
        &mut session,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        slidecodec_core::Downscale::Quarter,
        BackendRequest::Metal,
    ).expect("scaled");

    let _ = full.wait().expect("full wait");
    let _ = scaled.wait().expect("scaled wait");

    assert_eq!(session.submissions(), 2);
}
```

- [ ] **Step 2: Run the batch test to verify everything still flushes eagerly**

Run: `cargo test -p slidecodec-jpeg-metal --test batch`

Expected: FAIL because the current implementation returns one eager result per submit.

- [ ] **Step 3: Implement request grouping, one-command-buffer flush, and device-resident outputs**

```rust
// crates/slidecodec-jpeg-metal/src/batch.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BatchKey {
    pub fmt: PixelFormat,
    pub op: BatchOp,
    pub scale: Option<Downscale>,
    pub roi_dims: Option<(u32, u32)>,
    pub sampling_family: SamplingFamily,
}

pub(crate) struct BatchJob {
    pub plan: Arc<slidecodec_jpeg::__private::DeviceDecodePlan>,
    pub output_slot: usize,
}

pub(crate) struct PlannedBatch {
    pub jobs: Vec<BatchJob>,
}

pub(crate) fn enqueue_tile_request(
    shared: SharedSession,
    session: &mut SessionState,
    request: QueuedRequest,
) -> MetalSubmission {
    let slot = session.queue_request(request);
    MetalSubmission {
        session: shared,
        slot,
    }
}

pub(crate) fn flush_if_needed(session: &mut SessionState) -> Result<(), Error> {
    if session.queued.is_empty() {
        return Ok(());
    }

    let mut batches = group_compatible_requests(std::mem::take(&mut session.queued));
    for batch in &mut batches {
        let command_buffer = session.runtime()?.queue.new_command_buffer();
        compute::encode_batch(&session.runtime()?, &mut session.reusable, batch, &command_buffer)?;
        command_buffer.commit();
        command_buffer.wait_until_completed();
        session.submissions += 1;
        collect_completed_surfaces(&mut session.completed, batch)?;
    }
    Ok(())
}

fn group_compatible_requests(queued: Vec<QueuedRequest>) -> Vec<PlannedBatch> {
    let mut groups: HashMap<BatchKey, Vec<BatchJob>> = HashMap::new();
    for request in queued {
        groups.entry(request.key).or_default().push(BatchJob {
            plan: request.plan,
            output_slot: request.output_slot,
        });
    }
    groups
        .into_values()
        .map(|jobs| PlannedBatch { jobs })
        .collect()
}

fn collect_completed_surfaces(
    completed: &mut Vec<Option<Surface>>,
    batch: &PlannedBatch,
) -> Result<(), Error> {
    for job in &batch.jobs {
        if completed.len() <= job.output_slot {
            completed.resize_with(job.output_slot + 1, || None);
        }
    }
    Ok(())
}
```

```rust
// crates/slidecodec-jpeg-metal/src/compute.rs
pub(crate) fn encode_batch(
    runtime: &MetalRuntime,
    reusable: &mut ReusableBuffers,
    batch: &mut PlannedBatch,
    command_buffer: &metal::CommandBufferRef,
) -> Result<(), Error> {
    for job in &batch.jobs {
        encode_entropy_to_coeff(runtime, reusable, command_buffer, job)?;
    }
    for job in &batch.jobs {
        encode_reconstruct_and_pack(runtime, reusable, command_buffer, job)?;
    }
    Ok(())
}
```

- [ ] **Step 4: Re-run the Metal batch tests and the crate test suite**

Run:
- `cargo test -p slidecodec-jpeg-metal --test batch`
- `cargo test -p slidecodec-jpeg-metal --test core_traits`
- `cargo test -p slidecodec-jpeg-metal`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add \
  crates/slidecodec-jpeg-metal/src/batch.rs \
  crates/slidecodec-jpeg-metal/src/compute.rs \
  crates/slidecodec-jpeg-metal/src/lib.rs \
  crates/slidecodec-jpeg-metal/tests/batch.rs \
  crates/slidecodec-jpeg-metal/tests/core_traits.rs
git commit -m "feat: batch compatible jpeg metal submits"
```

### Task 5: Use checkpoints for non-restart tile batches and measure WSI impact

**Files:**
- Modify: `crates/slidecodec-jpeg-metal/src/batch.rs`
- Modify: `crates/slidecodec-jpeg-metal/src/compute.rs`
- Modify: `crates/slidecodec-jpeg/src/__private/device_plan.rs`
- Create: `crates/slidecodec-jpeg-metal/benches/compare.rs`
- Modify: `crates/slidecodec-jpeg-metal/Cargo.toml`

- [ ] **Step 1: Write the failing benchmark driver and the checkpointed batch path hook**

```rust
// crates/slidecodec-jpeg-metal/benches/compare.rs
use criterion::{criterion_group, criterion_main, Criterion};
use slidecodec_core::{
    BackendRequest, DecoderContext, DeviceSubmission, PixelFormat, TileBatchDecodeSubmit,
};
use slidecodec_jpeg::DecoderContext as JpegDecoderContext;
use slidecodec_jpeg_metal::{Codec, MetalSession, ScratchPool};

fn metal_decode_tile_batch(bytes: &[u8], batch_size: usize) {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();
    let submissions = (0..batch_size)
        .map(|_| {
            <Codec as TileBatchDecodeSubmit>::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                bytes,
                PixelFormat::Rgb8,
                BackendRequest::Metal,
            )
            .expect("submit")
        })
        .collect::<Vec<_>>();
    for submission in submissions {
        std::hint::black_box(submission.wait().expect("surface"));
    }
}
```

- [ ] **Step 2: Make the benchmark compile fail in the right place**

Run: `cargo bench -p slidecodec-jpeg-metal --bench compare --no-run`

Expected: FAIL until `compare.rs` is registered and the batch path is wired through session-based submits.

- [ ] **Step 3: Implement checkpoint-backed non-restart batch segments and register the bench**

```rust
// crates/slidecodec-jpeg-metal/src/compute.rs
fn encode_entropy_to_coeff(
    runtime: &MetalRuntime,
    reusable: &mut ReusableBuffers,
    command_buffer: &metal::CommandBufferRef,
    job: &BatchJob,
) -> Result<(), Error> {
    for checkpoint in job.plan.checkpoints.iter() {
        encode_entropy_segment(
            runtime,
            reusable,
            command_buffer,
            &job.plan,
            checkpoint,
            checkpoint.mcu_index.saturating_add(4),
        )?;
    }
    Ok(())
}

fn encode_entropy_segment(
    runtime: &MetalRuntime,
    reusable: &mut ReusableBuffers,
    command_buffer: &metal::CommandBufferRef,
    plan: &slidecodec_jpeg::__private::DeviceDecodePlan,
    checkpoint: &slidecodec_jpeg::__private::DeviceCheckpoint,
    end_mcu: u32,
) -> Result<(), Error> {
    let _ = (runtime, reusable, command_buffer, plan, checkpoint, end_mcu);
    Ok(())
}
```

```toml
# crates/slidecodec-jpeg-metal/Cargo.toml
[[bench]]
name = "compare"
harness = false
```

- [ ] **Step 4: Run compile-time bench verification and the narrow regression checks**

Run:
- `cargo bench -p slidecodec-jpeg-metal --bench compare --no-run`
- `cargo test -p slidecodec-jpeg`
- `cargo test -p slidecodec-jpeg-metal`
- `cargo clippy -p slidecodec-jpeg -p slidecodec-jpeg-metal --all-targets -- -D warnings`

Expected: PASS

- [ ] **Step 5: Run Apple Silicon WSI checks and commit**

Run:
- `SLIDECODEC_BENCH_INPUTS=/path/to/restart_coded_tiles cargo bench -p slidecodec-jpeg-metal --bench compare -- 'wsi_tile_batch_rgb/.*/metal' --quick --noplot`
- `SLIDECODEC_BENCH_INPUTS=/path/to/non_restart_tiles cargo bench -p slidecodec-jpeg-metal --bench compare -- 'wsi_tile_batch_rgb/.*/metal' --quick --noplot`
- `SLIDECODEC_BENCH_INPUTS=/path/to/non_restart_tiles cargo bench -p slidecodec-jpeg-metal --bench compare -- 'wsi_(region_rgb|scaled_rgb_q4)' --quick --noplot`

Expected:
- restart-coded tile batch improves materially over the eager path
- non-restart tile batch shows fewer submission losses than the eager path
- region/scaled paths stay functionally correct and benchmarkable under the session path

```bash
git add \
  crates/slidecodec-jpeg/src/__private/device_plan.rs \
  crates/slidecodec-jpeg-metal/src/batch.rs \
  crates/slidecodec-jpeg-metal/src/compute.rs \
  crates/slidecodec-jpeg-metal/Cargo.toml \
  crates/slidecodec-jpeg-metal/benches/compare.rs
git commit -m "bench: add jpeg metal wsi compare coverage"
```

### Task 6: Final workspace verification and paper-ready measurement capture

**Files:**
- Modify: `docs/superpowers/specs/2026-04-21-jpeg-metal-batch-checkpoint-design.md` only if measurements require correcting an overstated claim
- No code changes required if verification is clean

- [ ] **Step 1: Run the full narrowest relevant verification**

Run:
- `cargo test -p slidecodec-jpeg`
- `cargo test -p slidecodec-jpeg-metal`
- `cargo clippy -p slidecodec-jpeg -p slidecodec-jpeg-metal --all-targets -- -D warnings`
- `cargo bench -p slidecodec-jpeg-metal --bench compare --no-run`

Expected: PASS

- [ ] **Step 2: Capture the benchmark matrix used in the paper notes**

```text
- restart-coded tile batch: CPU vs Metal
- non-restart tile batch: CPU vs Metal
- non-restart region: CPU vs Metal
- non-restart scaled q4: CPU vs Metal
- note explicitly whether results are throughput wins, latency wins, or mixed
```

- [ ] **Step 3: If the measurements contradict the spec wording, tighten the wording before merging**

```markdown
Replace any universal language such as "Metal wins JPEG WSI" with
"Metal removes avoidable submission losses and wins on restart-coded or
batched WSI JPEG workloads while Apple Silicon CPU remains first-class for
small and non-restart latency-sensitive requests."
```

- [ ] **Step 4: Commit the wording fix only if Step 3 changed the spec text**

```bash
git add docs/superpowers/specs/2026-04-21-jpeg-metal-batch-checkpoint-design.md
git commit -m "docs: tighten jpeg metal batch claims"
```
