# Release Notes

## Current State

The workspace is versioned as `0.1.0` for the first public-source checkpoint.
It remains pre-1.0 while JPEG 2000 / HTJ2K ROI+reduced-resolution performance
thresholds and GPU adapter APIs settle.

The repository is ready for source publication once a real Git remote is
configured. Do not add placeholder `repository` or `homepage` metadata to
`Cargo.toml`; use the final public URL when it exists.

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
on self-hosted runners before claiming GPU runtime validation:

1. Apple Silicon Metal runner labels: `self-hosted`, `macOS`, `ARM64`,
   `metal`
2. x86_64 CUDA runner labels: `self-hosted`, `Linux`, `X64`, `cuda`
3. Use the `run-timed-benchmarks` workflow input when a release needs measured
   GPU benchmark timing rather than compile-only coverage

The CUDA crates remain compatibility adapters until a real CUDA backend lands.
Passing the CUDA self-hosted job validates API behavior and build compatibility;
it is not a CUDA decode performance claim.

## Crates.io

Crates.io publication is staged because workspace crates depend on each other.
The first publishable crate is:

1. `ashlar-core`

After `ashlar-core` is available in the registry, crates that depend only
on it can be published or dry-run verified:

1. `ashlar-jpeg`
2. `ashlar-tilecodec`

The JPEG Metal/CUDA adapter crates should be published only after
`ashlar-jpeg` is available.

The JPEG 2000 crates still need an explicit registry plan before publishing:
`ashlar-j2k` uses the repo-local native engine and its comparison benches
also exercise the Metal adapter. Keep those crates source-published until the
crate split and publish order are made acyclic for crates.io.
