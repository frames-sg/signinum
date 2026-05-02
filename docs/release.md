# Release Notes

## Current State

The repository is staged for the `signinum` facade release. The stable release
artifacts are `signinum`, `signinum-core`, `signinum-jpeg`, `signinum-j2k`,
`signinum-tilecodec`, and `signinum-cli`. `signinum-j2k-native` is published as
a `0.3.x` implementation dependency so `signinum-j2k` can be installed from
crates.io.

Metal and CUDA adapter crates are published as pre-1.0 `0.3.x` artifacts where
their APIs changed for the facade boundary. `signinum-jpeg-metal` remains
`0.2.0`.
CUDA explicit requests can produce CUDA device memory surfaces when built with
`cuda-runtime` on a host with a CUDA driver. `signinum-jpeg-cuda` can use
NVIDIA nvJPEG for full-frame RGB8 JPEG decode when `libnvjpeg` is installed;
unsupported JPEG shapes and the J2K CUDA adapter still use CPU decode plus
CUDA device memory upload. NVIDIA performance claims require self-hosted GPU
benchmark evidence.

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
and the opt-in nvJPEG JPEG decode path on a CUDA runner. Timed NVIDIA
performance claims require the `run-timed-benchmarks` workflow input and
recorded benchmark output.

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

The crates.io publish order is:

1. `signinum-cuda-runtime` `0.3.0`
2. `signinum-j2k-native` `0.3.0`
3. `signinum-j2k` `1.1.0`
4. `signinum-j2k-metal` `0.3.0`
5. `signinum-jpeg-cuda` `0.3.0`
6. `signinum-j2k-cuda` `0.3.0`
7. `signinum` `1.0.0`

Already-published crates reused by this release:

- `signinum-core` `1.0.0`
- `signinum-jpeg` `1.0.0`
- `signinum-tilecodec` `1.0.0`
- `signinum-jpeg-metal` `0.2.0`
- `signinum-cli` `1.0.0`

`signinum-j2k-compare` remains `publish = false`; it is a local parity oracle
helper, not a released runtime dependency.
