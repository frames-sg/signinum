# Metal Routing Policy Design

Date: 2026-05-01

## Goal

Harden the Metal adapter routing contract after the CPU-first 1.0 release
staging. This slice covers `signinum-jpeg-metal` and `signinum-j2k-metal`; it does
not add CUDA runtime decode and does not promote Metal crates to 1.0.

The output of this work is an explicit, tested policy for `BackendRequest::Cpu`,
`BackendRequest::Auto`, and `BackendRequest::Metal` across public device-output
entry points.

## Non-Goals

- No new Metal kernels.
- No performance tuning.
- No CUDA behavior changes.
- No public `signinum-core` routing abstraction.
- No claim that Metal crates are stable 1.0 artifacts.

## Routing Contract

`BackendRequest::Cpu` always returns a CPU host-backed surface when the CPU
decode path supports the request.

`BackendRequest::Auto` may choose Metal only for shapes the adapter recognizes
as supported and validated. If the request is unsupported for Metal, `Auto`
falls back to a CPU host-backed surface instead of failing because Metal was
not selected.

`BackendRequest::Metal` is strict. It must either return a Metal-backed surface
or fail with a clear error. It must not silently return a CPU host-backed
surface.

Metal-unavailable failures must remain distinct from unsupported-shape or
unsupported-request failures. Unsupported explicit Metal requests should be
classified as unsupported through the existing codec error surface.

## Architecture

Each Metal crate gets a small local routing-policy helper. The helper stays in
the adapter crate because JPEG and J2K have different capability predicates and
the shared `signinum-core` traits should not learn backend-specific policy.

The helper should normalize routing decisions into a small internal enum:

- `CpuHost`
- `MetalKernel`
- `RejectExplicitMetal { reason }`
- `MetalUnavailable`

The public decode paths should consult this helper before launching a kernel or
running a CPU fallback. Existing kernel code should remain behind the selected
`MetalKernel` path.

The helper should be easy to test without requiring benchmark-scale inputs. It
can expose `pub(crate)` decision functions or be tested through public decode
entry points, depending on which is clearer in each crate.

## JPEG Metal Scope

Cover the user-facing `signinum-jpeg-metal` entry points:

- full-frame device decode
- ROI device decode
- scaled device decode
- ROI+scaled device decode
- submit/session paths
- tile-batch paths

Supported explicit Metal paths should return `BackendKind::Metal`. Unsupported
explicit Metal paths should fail clearly instead of returning CPU output.
`Auto` should keep returning CPU output where the current policy considers CPU
the right path.

## J2K Metal Scope

Cover the user-facing `signinum-j2k-metal` entry points:

- full-frame device decode
- ROI device decode
- scaled device decode
- ROI+scaled device decode
- submit/session paths
- tile-batch paths

The policy must preserve current behavior where `Auto` uses CPU for small or
unsupported shapes and Metal for validated repeated/direct paths. Explicit
Metal must be strict: supported requests return `BackendKind::Metal`;
unsupported requests return an unsupported/unavailable error.

## Error Handling

Do not introduce silent fallbacks for explicit Metal. The error taxonomy should
remain simple:

- Metal device discovery failure: `MetalUnavailable`.
- Unsupported explicit Metal shape/request: existing unsupported error surface,
  with a reason string when useful.
- Kernel/runtime failure after selecting Metal: existing Metal kernel error.
- CPU decode or buffer validation failure: existing wrapped CPU/buffer errors.

If current code uses `MetalKernel` for unsupported preflight cases, prefer
moving those cases to a clearer unsupported-routing error where this can be
done without broad API churn.

## Tests

Add contract tests rather than implementation-coupled tests:

- `Cpu` returns CPU host-backed surfaces for supported CPU requests.
- `Auto` returns CPU host-backed surfaces for unsupported Metal shapes.
- `Auto` returns Metal-backed surfaces only for known supported Metal shapes.
- explicit `Metal` returns Metal-backed surfaces for supported shapes.
- explicit `Metal` rejects unsupported shapes instead of falling back to CPU.
- submit/session and tile-batch paths follow the same contract.

Tests should use existing committed fixtures and generated J2K fixtures. They
should avoid timing assertions.

## Documentation

Update Metal-facing docs to state the policy in user terms:

- CPU-first 1.0 remains the stable release posture.
- Metal is post-1.0 hardening work.
- `Auto` is allowed to fall back.
- explicit `Metal` is strict.
- CUDA remains compatibility-only with no runtime CUDA decode.

## Verification

The implementation plan should include at least:

- focused JPEG Metal routing tests
- focused J2K Metal routing tests
- `cargo test -p signinum-jpeg-metal --all-targets`
- `cargo test -p signinum-j2k-metal --all-targets`
- `cargo xtask clippy`
- `cargo xtask doc`

If the local host lacks Metal, the plan must still verify non-macOS/unavailable
behavior and clearly identify the missing runtime validation.
