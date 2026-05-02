# Metal Routing Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit, tested Metal routing-policy helpers for `signinum-jpeg-metal` and `signinum-j2k-metal` so `Cpu`, `Auto`, and strict `Metal` requests behave predictably.

**Architecture:** Keep routing policy local to each Metal adapter crate. Add one focused `routing.rs` module per crate, keep kernel code behind `MetalKernel` decisions, and make CPU fallback call sites return host-backed `BackendKind::Cpu` surfaces. Preserve CPU-first 1.0 and CUDA compatibility-only posture.

**Tech Stack:** Rust 1.94, `signinum-core` backend traits, `signinum-jpeg-metal`, `signinum-j2k-metal`, existing Metal compute modules, Cargo test/clippy/doc gates.

---

## Preconditions

- The current worktree already contains unrelated CPU-first 1.0 release edits and user-owned `.gitignore` / `paper/arxiv/` changes. Do not stage, revert, or rewrite unrelated files.
- Use `git add <exact files>` in commit steps. Do not use broad `git add .`.
- The local macOS host is expected to have Metal runtime support. If running on non-macOS, execute the non-macOS unavailable tests and record that runtime Metal validation was not available.

## File Structure

- Create `crates/signinum-jpeg-metal/src/routing.rs`: JPEG Metal route decision enum, capability predicate, error conversion.
- Modify `crates/signinum-jpeg-metal/src/lib.rs`: import routing helper, add unsupported Metal error variant, replace scattered `AutoDevicePath` decisions, return CPU host surfaces for `Auto` CPU fallback.
- Modify `crates/signinum-jpeg-metal/tests/core_traits.rs`: direct decode routing contract tests.
- Modify `crates/signinum-jpeg-metal/tests/batch.rs`: tile-batch routing contract tests.
- Create `crates/signinum-j2k-metal/src/routing.rs`: J2K Metal route decision enum, format/request predicate, error conversion.
- Modify `crates/signinum-j2k-metal/src/lib.rs`: import routing helper, add unsupported Metal error variant, gate explicit Metal and session paths through routing.
- Modify `crates/signinum-j2k-metal/tests/device.rs`: direct decode and tile-batch routing contract tests.
- Modify `docs/architecture.md` and `docs/wsi-decode-api.md`: user-facing Metal routing contract.

---

### Task 1: Add JPEG Metal Routing Contract Tests

**Files:**
- Modify: `crates/signinum-jpeg-metal/tests/core_traits.rs`
- Modify: `crates/signinum-jpeg-metal/tests/batch.rs`

- [ ] **Step 1: Write failing direct decode tests**

In `crates/signinum-jpeg-metal/tests/core_traits.rs`, extend the imports:

```rust
use signinum_core::{
    BackendKind, BackendRequest, CodecError, DeviceSubmission, DeviceSurface, Downscale,
    ImageDecode, ImageDecodeDevice, ImageDecodeSubmit, PixelFormat, Rect,
};
use signinum_jpeg_metal::{Decoder, Error, MetalBackendSession, MetalSession, ScratchPool};
```

Add this fixture near the existing fixture constants:

```rust
const GRAYSCALE: &[u8] = include_bytes!("../../../corpus/conformance/grayscale_8x8.jpg");
```

Add these tests near `cpu_device_request_stays_host_backed`:

```rust
#[test]
fn auto_region_scaled_unsupported_metal_shape_returns_cpu_surface() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");

    let surface = decoder
        .decode_region_scaled_to_device(PixelFormat::Rgb8, roi, Downscale::Quarter, BackendRequest::Auto)
        .expect("auto region scaled surface");

    assert_eq!(surface.backend_kind(), BackendKind::Cpu);
    assert_eq!(surface.dimensions(), (3, 3));
    assert!(surface.metal_buffer().is_none());
}

#[test]
fn explicit_metal_unsupported_grayscale_shape_is_rejected() {
    let mut decoder = Decoder::new(GRAYSCALE).expect("decoder");

    let result = decoder.decode_to_device(PixelFormat::Gray8, BackendRequest::Metal);

    match result {
        Err(Error::UnsupportedMetalRequest { reason }) => {
            assert!(reason.contains("JPEG Metal"));
        }
        Err(other) => panic!("unexpected explicit Metal error: {other:?}"),
        Ok(surface) => panic!(
            "explicit Metal must not silently fall back; got {:?}",
            surface.backend_kind()
        ),
    }
}

#[test]
fn explicit_metal_unsupported_error_is_codec_unsupported() {
    let mut decoder = Decoder::new(GRAYSCALE).expect("decoder");
    let err = match decoder.decode_to_device(PixelFormat::Gray8, BackendRequest::Metal) {
        Err(err) => err,
        Ok(surface) => panic!(
            "explicit Metal must not silently fall back; got {:?}",
            surface.backend_kind()
        ),
    };

    assert!(err.is_unsupported());
}
```

- [ ] **Step 2: Write failing tile-batch tests**

In `crates/signinum-jpeg-metal/tests/batch.rs`, add `CodecError` to the core imports:

```rust
use signinum_core::{
    BackendKind, BackendRequest, CodecError, DecoderContext, DeviceSubmission, DeviceSurface,
    Downscale, PixelFormat, Rect, TileBatchDecodeDevice, TileBatchDecodeSubmit,
};
```

Add this fixture near `BASELINE_420`:

```rust
const GRAYSCALE: &[u8] = include_bytes!("../../../corpus/conformance/grayscale_8x8.jpg");
```

Add these tests after `tile_region_device_decode_has_expected_dimensions`:

```rust
#[test]
fn auto_tile_region_scaled_unsupported_metal_shape_returns_cpu_surface() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };

    let surface = Codec::decode_tile_region_scaled_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        roi,
        Downscale::Quarter,
        BackendRequest::Auto,
    )
    .expect("auto tile region scaled surface");

    assert_eq!(surface.backend_kind(), BackendKind::Cpu);
    assert_eq!(surface.dimensions(), (2, 2));
}

#[test]
fn explicit_metal_tile_unsupported_shape_is_rejected() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let result = Codec::decode_tile_to_device(
        &mut ctx,
        &mut pool,
        GRAYSCALE,
        PixelFormat::Gray8,
        BackendRequest::Metal,
    );

    match result {
        Err(signinum_jpeg_metal::Error::UnsupportedMetalRequest { reason }) => {
            assert!(reason.contains("JPEG Metal"));
        }
        Err(other) => panic!("unexpected explicit Metal tile error: {other:?}"),
        Ok(surface) => panic!(
            "explicit Metal tile request must not fall back; got {:?}",
            surface.backend_kind()
        ),
    }
}

#[test]
fn explicit_metal_tile_unsupported_error_is_codec_unsupported() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let err = match Codec::decode_tile_to_device(
        &mut ctx,
        &mut pool,
        GRAYSCALE,
        PixelFormat::Gray8,
        BackendRequest::Metal,
    ) {
        Err(err) => err,
        Ok(surface) => panic!(
            "explicit Metal tile request must not fall back; got {:?}",
            surface.backend_kind()
        ),
    };

    assert!(err.is_unsupported());
}
```

- [ ] **Step 3: Run focused tests and verify they fail**

Run:

```bash
cargo test -p signinum-jpeg-metal --test core_traits
cargo test -p signinum-jpeg-metal --test batch
```

Expected:

- The first test fails because `Auto` CPU fallback currently returns a Metal-backed upload surface in some JPEG paths.
- The explicit Metal tests fail to compile until `Error::UnsupportedMetalRequest` exists.

- [ ] **Step 4: Keep red tests uncommitted**

Do not commit this red state. The tests are committed with the implementation
after the focused JPEG Metal routing checks pass.

---

### Task 2: Implement JPEG Metal Routing Helper

**Files:**
- Create: `crates/signinum-jpeg-metal/src/routing.rs`
- Modify: `crates/signinum-jpeg-metal/src/lib.rs`

- [ ] **Step 1: Add the JPEG routing module**

Create `crates/signinum-jpeg-metal/src/routing.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0

use signinum_core::BackendRequest;
use signinum_jpeg::{
    adapter::{JpegMetalFast420PacketV1, JpegMetalFast422PacketV1, JpegMetalFast444PacketV1},
    Decoder as CpuDecoder,
};

use crate::{batch::BatchOp, Error};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouteDecision {
    CpuHost,
    MetalKernel,
    RejectExplicitMetal { reason: &'static str },
    MetalUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JpegMetalCapabilities {
    has_fast_packet: bool,
    auto_prefers_metal: bool,
}

impl JpegMetalCapabilities {
    pub(crate) fn for_request(
        decoder: &CpuDecoder<'_>,
        op: BatchOp,
        fast444_packet: Option<&JpegMetalFast444PacketV1>,
        fast422_packet: Option<&JpegMetalFast422PacketV1>,
        fast420_packet: Option<&JpegMetalFast420PacketV1>,
    ) -> Self {
        let has_fast_packet =
            fast444_packet.is_some() || fast422_packet.is_some() || fast420_packet.is_some();
        let auto_prefers_metal = !matches!(op, BatchOp::RegionScaled { .. })
            && decoder.info().restart_interval.is_some()
            && has_fast_packet;

        Self {
            has_fast_packet,
            auto_prefers_metal,
        }
    }
}

pub(crate) fn decide_route(
    backend: BackendRequest,
    capabilities: JpegMetalCapabilities,
) -> RouteDecision {
    match backend {
        BackendRequest::Cpu => RouteDecision::CpuHost,
        BackendRequest::Auto if capabilities.auto_prefers_metal => RouteDecision::MetalKernel,
        BackendRequest::Auto => RouteDecision::CpuHost,
        BackendRequest::Metal => {
            #[cfg(not(target_os = "macos"))]
            {
                RouteDecision::MetalUnavailable
            }
            #[cfg(target_os = "macos")]
            {
                if capabilities.has_fast_packet {
                    RouteDecision::MetalKernel
                } else {
                    RouteDecision::RejectExplicitMetal {
                        reason: "JPEG Metal supports explicit requests only for fast 4:2:0, 4:2:2, or 4:4:4 baseline packets",
                    }
                }
            }
        }
        BackendRequest::Cuda => RouteDecision::RejectExplicitMetal {
            reason: "CUDA request is not supported by signinum-jpeg-metal",
        },
    }
}

pub(crate) fn decision_error(decision: RouteDecision) -> Option<Error> {
    match decision {
        RouteDecision::RejectExplicitMetal { reason } => {
            Some(Error::UnsupportedMetalRequest { reason })
        }
        RouteDecision::MetalUnavailable => Some(Error::MetalUnavailable),
        RouteDecision::CpuHost | RouteDecision::MetalKernel => None,
    }
}
```

- [ ] **Step 2: Wire the module and error variant**

In `crates/signinum-jpeg-metal/src/lib.rs`, add the module next to the other local modules:

```rust
mod routing;
```

Add this error variant to `pub enum Error`:

```rust
    #[error("unsupported JPEG Metal request: {reason}")]
    UnsupportedMetalRequest { reason: &'static str },
```

Update `CodecError for Error::is_unsupported`:

```rust
        matches!(
            self,
            Self::UnsupportedBackend { .. }
                | Self::UnsupportedMetalRequest { .. }
                | Self::MetalUnavailable
                | Self::MetalKernel { .. }
        ) || matches!(self, Self::Decode(inner) if inner.is_unsupported())
```

- [ ] **Step 3: Replace the old auto-only decision type**

Remove this enum from `crates/signinum-jpeg-metal/src/lib.rs`:

```rust
#[cfg(any(target_os = "macos", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutoDevicePath {
    CpuUpload,
    MetalKernel,
}
```

Remove `choose_auto_device_path`.

Add this helper near the old `choose_auto_device_path` location:

```rust
fn choose_route(
    decoder: &CpuDecoder<'_>,
    op: batch::BatchOp,
    backend: BackendRequest,
    fast444_packet: Option<&JpegMetalFast444PacketV1>,
    fast422_packet: Option<&JpegMetalFast422PacketV1>,
    fast420_packet: Option<&JpegMetalFast420PacketV1>,
) -> routing::RouteDecision {
    let capabilities = routing::JpegMetalCapabilities::for_request(
        decoder,
        op,
        fast444_packet,
        fast422_packet,
        fast420_packet,
    );
    routing::decide_route(backend, capabilities)
}
```

- [ ] **Step 4: Make JPEG CPU fallback host-backed**

In each `Auto` CPU fallback branch inside `decode_surface_from_decoder`, call `upload_surface(..., BackendRequest::Cpu)` instead of passing through `backend`.

For full decode, the CPU branch should look like:

```rust
let dims = decoder.info().dimensions;
let stride = dims.0 as usize * fmt.bytes_per_pixel();
let mut out = vec![0u8; stride * dims.1 as usize];
decoder.decode_into_with_scratch(pool, &mut out, stride, fmt)?;
upload_surface(out, dims, fmt, BackendRequest::Cpu)
```

For region decode:

```rust
let dims = (roi.w, roi.h);
let stride = dims.0 as usize * fmt.bytes_per_pixel();
let mut out = vec![0u8; stride * dims.1 as usize];
decoder.decode_region_into_with_scratch(pool, &mut out, stride, fmt, to_jpeg_rect(roi))?;
upload_surface(out, dims, fmt, BackendRequest::Cpu)
```

For scaled decode:

```rust
let dims = scaled_dims(decoder.info().dimensions, scale);
let stride = dims.0 as usize * fmt.bytes_per_pixel();
let mut out = vec![0u8; stride * dims.1 as usize];
decoder.decode_scaled_into_with_scratch(pool, &mut out, stride, fmt, scale)?;
upload_surface(out, dims, fmt, BackendRequest::Cpu)
```

In `decode_region_scaled_cpu_upload`, pass `BackendRequest::Cpu` from `Auto` fallback call sites. Keep `BackendRequest::Cpu` call sites unchanged.

- [ ] **Step 5: Route JPEG `Auto` and explicit `Metal` through the helper**

For each operation in `decode_surface_from_decoder`, replace `BackendRequest::Auto` / `BackendRequest::Metal` branches with a route match. The full decode branch should follow this shape:

```rust
BackendRequest::Auto | BackendRequest::Metal => {
    let route = choose_route(
        decoder,
        op,
        backend,
        fast444_packet,
        fast422_packet,
        fast420_packet,
    );
    if let Some(error) = routing::decision_error(route) {
        return Err(error);
    }
    match route {
        routing::RouteDecision::CpuHost => {
            let dims = decoder.info().dimensions;
            let stride = dims.0 as usize * fmt.bytes_per_pixel();
            let mut out = vec![0u8; stride * dims.1 as usize];
            decoder.decode_into_with_scratch(pool, &mut out, stride, fmt)?;
            upload_surface(out, dims, fmt, BackendRequest::Cpu)
        }
        routing::RouteDecision::MetalKernel => compute::decode_to_surface(
            decoder,
            pool,
            fmt,
            fast444_packet,
            fast422_packet,
            fast420_packet,
        ),
        routing::RouteDecision::RejectExplicitMetal { .. }
        | routing::RouteDecision::MetalUnavailable => unreachable!("handled by decision_error"),
    }
}
```

Apply the same pattern for region, scaled, and region+scaled, using the existing compute functions for the `MetalKernel` arm and the existing CPU decode body for the `CpuHost` arm.

- [ ] **Step 6: Run JPEG routing tests**

Run:

```bash
cargo test -p signinum-jpeg-metal --test core_traits
cargo test -p signinum-jpeg-metal --test batch
```

Expected: all six tests pass.

- [ ] **Step 7: Run the full JPEG Metal test target**

Run:

```bash
cargo test -p signinum-jpeg-metal --all-targets
```

Expected: all tests pass. Existing explicit Metal supported-shape tests still return `BackendKind::Metal`.

- [ ] **Step 8: Commit JPEG routing implementation**

```bash
git add crates/signinum-jpeg-metal/src/routing.rs crates/signinum-jpeg-metal/src/lib.rs crates/signinum-jpeg-metal/tests/core_traits.rs crates/signinum-jpeg-metal/tests/batch.rs
git commit -m "feat: add jpeg metal routing policy"
```

---

### Task 3: Add J2K Metal Routing Contract Tests

**Files:**
- Modify: `crates/signinum-j2k-metal/tests/device.rs`

- [ ] **Step 1: Add codec error import**

In `crates/signinum-j2k-metal/tests/device.rs`, add `CodecError` to the `signinum_core` imports:

```rust
use signinum_core::{
    BackendKind, BackendRequest, CodecError, DeviceSubmission, DeviceSurface, Downscale,
    ImageDecode, ImageDecodeDevice, PixelFormat, Rect, TileBatchDecodeDevice,
    TileBatchDecodeSubmit,
};
```

- [ ] **Step 2: Add direct decode strict explicit Metal tests**

Add these tests near the existing explicit Metal full decode tests:

```rust
#[test]
fn explicit_metal_unsupported_rgba16_full_decode_is_rejected() {
    let bytes = fixture_rgb12();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");

    let result = decoder.decode_to_device(PixelFormat::Rgba16, BackendRequest::Metal);

    match result {
        Err(Error::UnsupportedMetalRequest { reason }) => {
            assert!(reason.contains("Rgba16"));
        }
        Err(other) => panic!("unexpected explicit Metal error: {other:?}"),
        Ok(surface) => panic!(
            "explicit Metal must not silently fall back; got {:?}",
            surface.backend_kind()
        ),
    }
}

#[test]
fn explicit_metal_unsupported_rgba16_error_is_codec_unsupported() {
    let bytes = fixture_rgb12();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let err = match decoder.decode_to_device(PixelFormat::Rgba16, BackendRequest::Metal) {
        Err(err) => err,
        Ok(surface) => panic!(
            "explicit Metal must not silently fall back; got {:?}",
            surface.backend_kind()
        ),
    };

    assert!(err.is_unsupported());
}
```

- [ ] **Step 3: Add Auto CPU host fallback assertion**

Extend `auto_region_and_scaled_fallback_to_cpu_surface_and_match_host_decode` by asserting that CPU fallback is host-backed:

```rust
assert!(region_surface.metal_buffer().is_none());
assert!(scaled_surface.metal_buffer().is_none());
```

Place the first assertion after `assert_eq!(region_surface.backend_kind(), BackendKind::Cpu);` and the second after `assert_eq!(scaled_surface.backend_kind(), BackendKind::Cpu);`.

- [ ] **Step 4: Add tile-batch strict explicit Metal test**

Add this test near the existing `MetalTileBatch` tests:

```rust
#[test]
fn explicit_metal_tile_unsupported_rgba16_is_rejected() {
    let bytes = fixture_rgb12();
    let mut ctx = signinum_core::DecoderContext::<J2kContext>::new();
    let mut pool = J2kScratchPool::new();

    let result = Codec::decode_tile_to_device(
        &mut ctx,
        &mut pool,
        &bytes,
        PixelFormat::Rgba16,
        BackendRequest::Metal,
    );

    match result {
        Err(Error::UnsupportedMetalRequest { reason }) => {
            assert!(reason.contains("Rgba16"));
        }
        Err(other) => panic!("unexpected explicit Metal tile error: {other:?}"),
        Ok(surface) => panic!(
            "explicit Metal tile request must not fall back; got {:?}",
            surface.backend_kind()
        ),
    }
}
```

- [ ] **Step 5: Run focused tests and verify they fail**

Run:

```bash
cargo test -p signinum-j2k-metal --test device explicit_metal_unsupported
cargo test -p signinum-j2k-metal --test device explicit_metal_tile_unsupported
```

Expected:

- Tests fail to compile until `Error::UnsupportedMetalRequest` exists.
- If the variant is added before these tests are run, they fail until explicit Metal preflight maps unsupported formats to that variant.

- [ ] **Step 6: Keep red tests uncommitted**

Do not commit this red state. The tests are committed with the implementation
after the focused J2K Metal routing checks pass.

---

### Task 4: Implement J2K Metal Routing Helper

**Files:**
- Create: `crates/signinum-j2k-metal/src/routing.rs`
- Modify: `crates/signinum-j2k-metal/src/lib.rs`

- [ ] **Step 1: Add the J2K routing module**

Create `crates/signinum-j2k-metal/src/routing.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0

use signinum_core::{BackendRequest, PixelFormat};

use crate::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouteDecision {
    CpuHost,
    MetalKernel,
    RejectExplicitMetal { reason: &'static str },
    MetalUnavailable,
}

pub(crate) fn supports_metal_format(fmt: PixelFormat) -> bool {
    !matches!(fmt, PixelFormat::Rgba16)
}

pub(crate) fn decide_route(backend: BackendRequest, fmt: PixelFormat) -> RouteDecision {
    match backend {
        BackendRequest::Cpu => RouteDecision::CpuHost,
        BackendRequest::Auto => RouteDecision::CpuHost,
        BackendRequest::Metal => {
            #[cfg(not(target_os = "macos"))]
            {
                RouteDecision::MetalUnavailable
            }
            #[cfg(target_os = "macos")]
            {
                if supports_metal_format(fmt) {
                    RouteDecision::MetalKernel
                } else {
                    RouteDecision::RejectExplicitMetal {
                        reason: "J2K Metal does not support PixelFormat::Rgba16",
                    }
                }
            }
        }
        BackendRequest::Cuda => RouteDecision::RejectExplicitMetal {
            reason: "CUDA request is not supported by signinum-j2k-metal",
        },
    }
}

pub(crate) fn decision_error(decision: RouteDecision) -> Option<Error> {
    match decision {
        RouteDecision::RejectExplicitMetal { reason } => {
            Some(Error::UnsupportedMetalRequest { reason })
        }
        RouteDecision::MetalUnavailable => Some(Error::MetalUnavailable),
        RouteDecision::CpuHost | RouteDecision::MetalKernel => None,
    }
}
```

- [ ] **Step 2: Wire the module and error variant**

In `crates/signinum-j2k-metal/src/lib.rs`, add the module near the other local modules:

```rust
mod routing;
```

Add this error variant to `pub enum Error`:

```rust
    #[error("unsupported J2K Metal request: {reason}")]
    UnsupportedMetalRequest { reason: &'static str },
```

Update `CodecError for Error::is_unsupported`:

```rust
        matches!(
            self,
            Self::UnsupportedBackend { .. }
                | Self::UnsupportedMetalRequest { .. }
                | Self::MetalUnavailable
                | Self::MetalKernel { .. }
        ) || matches!(self, Self::Decode(inner) if inner.is_unsupported())
```

- [ ] **Step 3: Route full decode**

Replace `decode_to_surface_impl`'s `match backend` with this structure:

```rust
match backend {
    BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
    _ => {
        let route = routing::decide_route(backend, fmt);
        if let Some(error) = routing::decision_error(route) {
            return Err(error);
        }
        match route {
            routing::RouteDecision::CpuHost => self.decode_to_cpu_surface(fmt),
            routing::RouteDecision::MetalKernel => {
                #[cfg(target_os = "macos")]
                {
                    if let Some(surface) = self.decode_direct_to_surface(fmt)? {
                        Ok(surface)
                    } else {
                        self.decode_full_to_metal_surface(fmt)
                    }
                }
                #[cfg(not(target_os = "macos"))]
                {
                    Err(Error::MetalUnavailable)
                }
            }
            routing::RouteDecision::RejectExplicitMetal { .. }
            | routing::RouteDecision::MetalUnavailable => unreachable!("handled by decision_error"),
        }
    }
}
```

- [ ] **Step 4: Route region, scaled, and region+scaled decode**

Apply the same route pattern to:

- `decode_region_to_surface_impl`
- `decode_scaled_to_surface_impl`
- `decode_region_scaled_to_surface_impl`

For `CpuHost`, call the existing CPU helper. For `MetalKernel`, call the existing Metal helper. For `Cuda`, keep `Err(Error::UnsupportedBackend { request: backend })`.

The region `MetalKernel` arm should call:

```rust
self.decode_region_to_metal_surface(fmt, plan)
```

The scaled `MetalKernel` arm should call:

```rust
self.decode_scaled_to_metal_surface(fmt, scale, plan)
```

The region+scaled `MetalKernel` arm should call:

```rust
self.decode_region_scaled_to_metal_surface(fmt, roi, scale, plan)
```

- [ ] **Step 5: Route explicit session methods**

At the start of each explicit session method, reject unsupported formats before building a plan:

- `decode_to_device_with_session`
- `decode_region_to_device_with_session`
- `decode_scaled_to_device_with_session`
- `decode_region_scaled_to_device_with_session`

Use this code shape:

```rust
if !routing::supports_metal_format(fmt) {
    return Err(Error::UnsupportedMetalRequest {
        reason: "J2K Metal does not support PixelFormat::Rgba16",
    });
}
```

Keep the existing non-macOS `Err(Error::MetalUnavailable)` branches.

- [ ] **Step 6: Run J2K routing tests**

Run:

```bash
cargo test -p signinum-j2k-metal --test device explicit_metal_unsupported
cargo test -p signinum-j2k-metal --test device explicit_metal_tile_unsupported
cargo test -p signinum-j2k-metal --test device auto_region_and_scaled_fallback_to_cpu_surface_and_match_host_decode
```

Expected: all targeted tests pass.

- [ ] **Step 7: Run the full J2K Metal test target**

Run:

```bash
RUST_TEST_THREADS=1 cargo test -p signinum-j2k-metal --all-targets
```

Expected: all tests pass. Existing explicit Metal supported-shape tests still return `BackendKind::Metal`.

- [ ] **Step 8: Commit J2K routing implementation**

```bash
git add crates/signinum-j2k-metal/src/routing.rs crates/signinum-j2k-metal/src/lib.rs crates/signinum-j2k-metal/tests/device.rs
git commit -m "feat: add j2k metal routing policy"
```

---

### Task 5: Document the Metal Routing Contract

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/wsi-decode-api.md`

- [ ] **Step 1: Update architecture docs**

In `docs/architecture.md`, add this paragraph to the Metal/GPU adapter section:

```markdown
Metal adapter routing is explicit after the CPU-first 1.0 line. `BackendRequest::Cpu`
returns host-backed CPU surfaces. `BackendRequest::Auto` may select Metal only
for validated adapter-supported shapes; otherwise it falls back to a host-backed
CPU surface. `BackendRequest::Metal` is strict: it returns a Metal-backed
surface for supported shapes or a clear unsupported/unavailable error. It does
not silently return CPU output.
```

- [ ] **Step 2: Update WSI decode API docs**

In `docs/wsi-decode-api.md`, add this paragraph near the backend request documentation:

```markdown
For Metal adapters, `BackendRequest::Auto` is a routing hint and may fall back
to host-backed CPU output when the request shape is not on the Metal-supported
path. `BackendRequest::Metal` is a strict request: supported shapes return
Metal-backed surfaces, unsupported shapes fail as unsupported, and hosts
without Metal fail as unavailable.
```

- [ ] **Step 3: Run docs check**

Run:

```bash
cargo xtask doc
```

Expected: rustdoc completes with warnings denied.

- [ ] **Step 4: Commit docs**

```bash
git add docs/architecture.md docs/wsi-decode-api.md
git commit -m "docs: describe metal routing policy"
```

---

### Task 6: Final Verification

**Files:**
- Verify only; no source edits expected.

- [ ] **Step 1: Format**

Run:

```bash
cargo fmt --all -- --check
```

Expected: exits 0.

- [ ] **Step 2: Full Metal tests**

Run:

```bash
cargo test -p signinum-jpeg-metal --all-targets
RUST_TEST_THREADS=1 cargo test -p signinum-j2k-metal --all-targets
```

Expected: both commands exit 0.

- [ ] **Step 3: Workspace clippy**

Run:

```bash
cargo xtask clippy
```

Expected: exits 0.

- [ ] **Step 4: Workspace docs**

Run:

```bash
cargo xtask doc
```

Expected: exits 0.

- [ ] **Step 5: Check for unintended release-file staging**

Run:

```bash
git status --short
git diff --cached --name-only
```

Expected:

- Only Metal routing files are staged after each Metal-routing commit.
- Existing CPU-first release edits, `.gitignore`, and `paper/arxiv/` remain untouched unless the user separately asks to stage or commit them.

- [ ] **Step 6: Final summary**

Report:

- Tests run and pass/fail status.
- Whether runtime Metal validation ran on macOS.
- Any unavailable-host limitation.
- Confirmation that CUDA runtime decode was not added.
