# Release Notes

## Current State

The workspace is versioned as `0.1.0` for the first public-source checkpoint.
It remains pre-1.0 while JPEG 2000 / HTJ2K ROI behavior and GPU adapter APIs
settle.

The repository is ready for source publication once a real Git remote is
configured. Do not add placeholder `repository` or `homepage` metadata to
`Cargo.toml`; use the final public URL when it exists.

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
