# CUDA Runner Validation Design

## Context

The repository already contains `signinum-jpeg-cuda` and `signinum-j2k-cuda`
adapter crates. They are compatibility-only: `BackendRequest::Cpu` and
`BackendRequest::Auto` return CPU-backed host surfaces, while explicit
`BackendRequest::Cuda` returns an unavailable error before decode validation.
The `cuda-runtime` feature is declared but intentionally does not enable runtime
CUDA decode.

A self-hosted GitHub runner is now available for CUDA validation. The existing
manual `.github/workflows/gpu-validation.yml` workflow has a CUDA-labeled job,
but that job currently mixes CUDA adapter tests with a J2K Metal bench compile
command. Runner validation should make the CUDA job explicit, reproducible, and
hard to accidentally drift into Metal-only validation.

## Goals

- Validate that the self-hosted CUDA runner can build and test the CUDA adapter
  crates with `--features cuda-runtime`.
- Preserve the current compatibility-only CUDA behavior.
- Make the workflow output enough host context to diagnose runner setup issues.
- Keep validation separate from runtime CUDA decode or CUDA performance claims.
- Add repository integrity coverage so future workflow edits keep the CUDA job
  focused on CUDA adapter validation.

## Non-Goals

- No runtime CUDA decoder.
- No CUDA kernels, CUDA FFI, toolkit dependency, or device memory surface.
- No NVIDIA performance benchmark claims.
- No changes to Metal runtime behavior.
- No release of CUDA adapter crates as stable 1.0 artifacts.

## Recommended Approach

Tighten the existing manual GPU validation workflow instead of adding a new
workflow. The workflow already represents the intended split between Apple
Silicon Metal validation and x86_64 Linux CUDA validation, so the smallest
maintainable design is to make the CUDA job self-describing and crate-accurate.

The CUDA job should run on labels:

```yaml
[self-hosted, Linux, X64, cuda]
```

It should:

1. Check out the repository.
2. Install the stable Rust toolchain.
3. Print runner diagnostics such as kernel, architecture, Rust version, and
   optional `nvidia-smi` output.
4. Run:

```sh
cargo test -p signinum-jpeg-cuda --all-targets --features cuda-runtime
cargo test -p signinum-j2k-cuda --all-targets --features cuda-runtime
```

5. Compile benchmark targets that are relevant to CPU/J2K/JPEG validation
   without implying CUDA runtime timing. If a compile gate is kept in the CUDA
   job, it should be documented as compile coverage and should not be a Metal
   adapter bench unless there is an explicit reason.

## Workflow Contract

The CUDA runner job is a compatibility and build gate. Passing it means:

- the self-hosted runner can compile and run CUDA adapter tests;
- `cuda-runtime` remains build-compatible;
- explicit CUDA requests still fail clearly as unavailable;
- `Auto` and `Cpu` still produce CPU-backed surfaces.

Passing it does not mean:

- CUDA kernels were launched;
- a CUDA device surface was produced;
- CUDA decode performance was measured.

## Test Strategy

Update `crates/signinum-core/tests/repo_integrity.rs` to assert that the manual
GPU workflow remains explicit:

- it contains `workflow_dispatch`;
- it has self-hosted Metal and CUDA labels;
- the CUDA job runs both CUDA adapter test commands with
  `--features cuda-runtime`;
- the CUDA job does not depend on `signinum-j2k-metal --bench compare --no-run`
  as its J2K validation path.

Existing CUDA adapter behavior tests should remain in place. They are the
primary behavior coverage for the compatibility-only contract.

## Documentation

Keep public docs aligned with the workflow contract:

- `docs/wsi-decode-api.md` should say CUDA runner validation covers
  compatibility-only adapter behavior.
- `docs/release.md` should say CUDA runner success is not runtime CUDA decode
  validation.
- `docs/bench.md` should avoid presenting CUDA runner results as performance
  data.

## Risks

- A self-hosted runner may have a GPU label but no functional NVIDIA driver.
  Diagnostics should surface that clearly without turning compatibility-only
  tests into runtime CUDA requirements.
- The `cuda-runtime` feature is currently inert. Tests should prove the feature
  remains build-compatible, not pretend it exercises CUDA execution.
- Overloading the CUDA job with unrelated Metal compile gates can obscure
  failures and weaken the signal. Keep cross-backend compile gates in hosted CI
  or the Metal job unless there is a documented reason.

## Acceptance Criteria

- `.github/workflows/gpu-validation.yml` has a CUDA job focused on CUDA adapter
  compatibility validation.
- The CUDA job prints basic runner diagnostics.
- CUDA adapter tests run with `--features cuda-runtime`.
- Repository integrity tests protect the workflow contract.
- Docs continue to state that CUDA is compatibility-only with no runtime CUDA
  decode or CUDA performance claim.
