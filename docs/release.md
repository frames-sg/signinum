# Release Notes

## Current State

The repository is staged for CPU-first 1.0. The stable 1.0 release artifacts
are `signinum-core`, `signinum-jpeg`, `signinum-j2k`, `signinum-tilecodec`, and
`signinum-cli`. `signinum-j2k-native` is published as a `0.2.x` implementation
dependency so `signinum-j2k` can be installed from crates.io.

Metal and CUDA adapter crates remain pre-1.0 and are excluded from the
CPU-first crates.io publish workflow. CUDA explicit requests can produce CUDA
device-memory surfaces when built with `cuda-runtime` on a host with a CUDA
driver, but decode is still CPU-produced and uploaded. CUDA device memory is
validated separately from kernel work. There is no CUDA kernel decode and no
NVIDIA performance claim.

## Verification Gates

Hosted CI must pass before release staging:

1. `cargo fmt --all -- --check`
2. `cargo clippy --workspace --all-targets --all-features -- -D warnings`
3. `cargo test --workspace --all-targets --all-features` on Linux x86_64,
   Linux aarch64, macOS x86_64, and macOS aarch64 runners
4. `cargo doc --workspace --all-features --no-deps` with rustdoc warnings
   denied
5. Benchmark compile checks for JPEG, JPEG Metal, J2K Metal, and tilecodec

Runtime GPU validation is intentionally separate because hosted GitHub runners
do not provide the required devices. Run `.github/workflows/gpu-validation.yml`
on self-hosted runners before claiming Metal runtime validation:

1. Apple Silicon Metal runner labels: `self-hosted`, `macOS`, `ARM64`,
   `metal`
2. x86_64 CUDA runner labels: `self-hosted`, `Linux`, `X64`, `cuda`
3. Use the `run-timed-benchmarks` workflow input when a release needs measured
   GPU benchmark timing rather than compile-only coverage

Passing the CUDA self-hosted job validates `cuda-runtime` device-memory output
on a CUDA runner. It is not a CUDA kernel decode or NVIDIA performance claim.

## Crates.io

Crates.io publication is staged because workspace crates depend on each other.
Before publishing, run `cargo xtask package` from a clean worktree. The package
preflight runs `cargo package --list` for every CPU-first publishable crate,
then runs strict `cargo package --no-verify` only for crates that do not depend
on unpublished workspace versions. Downstream crates such as `signinum-jpeg`,
`signinum-tilecodec`, `signinum-j2k`, and `signinum-cli` cannot pass strict
pre-publish packaging until the prior staged crates exist on crates.io, because
Cargo resolves their versioned path dependencies against the registry during
packaging.

This is an unpublished workspace dependencies limit, not a package content
failure. The publish workflow's dry-run mode mirrors that limit: it uses
`cargo publish --dry-run` for registry-independent crates and
`cargo package --list` for crates blocked only by unpublished workspace
dependencies. Real publishes still run `cargo publish` in dependency order.

The CPU-first 1.0 publish order is:

1. `signinum-core` `1.0.0`
2. `signinum-j2k-native` `0.2.x`
3. `signinum-jpeg` `1.0.0`
4. `signinum-tilecodec` `1.0.0`
5. `signinum-j2k` `1.0.0`
6. `signinum-cli` `1.0.0`

`signinum-j2k-compare` remains `publish = false`; it is a local parity oracle
helper, not a released runtime dependency. Metal and CUDA crates are held for
the post-1.0 hardening track.
