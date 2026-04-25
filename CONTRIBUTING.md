# Contributing

Contributions should keep the workspace focused on WSI-shaped codec primitives:
safe parsing, predictable decode behavior, caller-owned scratch/context reuse,
and reproducible benchmarks.

## Development Setup

Use the Rust toolchain pinned by `rust-toolchain.toml`.

```sh
cargo test --workspace
cargo doc --workspace --no-deps
cargo clippy --workspace --all-targets -- -D warnings
```

Comparator benchmarks may need optional system libraries. See
`docs/bench.md` for setup and skip behavior.

## Pull Requests

- Keep changes scoped to one codec, adapter, or documentation topic when
  possible.
- Add or update behavior-focused tests for decode, API, or data-flow changes.
- Do not remove passing regression tests as cleanup.
- Avoid hardcoded secrets, credentials, or local machine paths.
- Surface unsupported inputs and backend failures explicitly; do not add silent
  fallback paths.
- Run the narrowest relevant tests before opening a PR, then run the workspace
  checks above before release-facing changes.

## Public API Changes

Public decode APIs are part of the WSI integration surface. Changes to ROI,
scaled decode, tile-batch, row-streaming, context, scratch-pool, or device
surface behavior should update:

- README quick-start or examples when user-facing behavior changes
- API docs for affected public items
- integration tests covering caller-visible behavior
- `docs/bench.md` when benchmark methodology changes
