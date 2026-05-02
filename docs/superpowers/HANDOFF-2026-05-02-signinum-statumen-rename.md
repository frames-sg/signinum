# Signinum / Statumen Rename Handoff

## Repositories

- Signinum local checkout: `/Users/user/Bench/ashlar`
- Signinum GitHub: `https://github.com/jcwal1516/signinum`
- Signinum latest pushed commit: `a646f55`
- Statumen local checkout: `/Users/user/Bench/ziggurat`
- Statumen GitHub: `https://github.com/jcwal1516/statumen`
- Statumen latest pushed commit: `c3a39fa`

Both local repos were clean and aligned with `origin/main` after the rename,
publish, and yank work.

## Registry State

Published replacement crates:

- `signinum-core 1.0.0`
- `signinum-jpeg 1.0.0`
- `signinum-j2k 1.0.0`
- `signinum-tilecodec 1.0.0`
- `signinum-cli 1.0.0`
- `signinum-j2k-native 0.2.0`
- `signinum-jpeg-metal 0.2.0`
- `signinum-j2k-metal 0.2.0`
- `signinum-jpeg-cuda 0.2.0`
- `signinum-j2k-cuda 0.2.0`
- `signinum-cuda-runtime 0.2.0`
- `statumen 0.1.0`

Yanked retired crates:

- `ashlar-core 0.1.0`
- `ashlar-jpeg 0.1.0`
- `ashlar-j2k 0.1.0`
- `ashlar-tilecodec 0.1.0`
- `ashlar-cli 0.1.0`
- `ashlar-j2k-native 0.1.0`
- `ashlar-jpeg-metal 0.1.0`
- `ashlar-j2k-metal 0.1.0`
- `ashlar-jpeg-cuda 0.1.0`
- `ashlar-j2k-cuda 0.1.0`
- `ashlar-j2k-compare 0.1.0`
- `ziggurat 0.1.0`

Cargo yanks do not support a yank message. Existing lockfiles can still build
with yanked versions, but new resolution will not select them by default.

## What Changed

Signinum:

- Renamed the workspace from the retired ashlar crate family to `signinum-*`.
- Renamed the CLI binary to `signinum`.
- Published CPU-first crates, Metal/CUDA adapter replacements, and the CUDA
  runtime helper.
- Kept `signinum-j2k-compare` unpublished as a local oracle/comparison helper.
- Updated release docs and publish policy to include pre-1.0 adapter
  replacements.

Statumen:

- Renamed the package from the retired ziggurat crate to `statumen`.
- Renamed GitHub repo to `jcwal1516/statumen`.
- Replaced temporary dependency aliases on retired codec crates with real
  `signinum-*` dependencies.
- Updated CI checkout paths, publish job names, and lockfile entries.
- Published `statumen 0.1.0`.

## Verification Already Run

Signinum:

- `cargo fmt --all -- --check`
- `cargo test --workspace --all-targets --all-features`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test -p signinum-core --test repo_integrity`
- `cargo package -p signinum-cuda-runtime --list --allow-dirty`
- crates.io API checks confirmed published replacement versions were visible.
- crates.io API checks confirmed retired versions were yanked.

Statumen:

- `cargo xtask fmt`
- `cargo xtask clippy`
- `cargo xtask test`
- `cargo xtask package`
- `DRY_RUN_ONLY=true scripts/publish-crate.sh`
- Source-name sweep after dependency flip returned no retired-name matches
  outside `.git` and build outputs.

## Known Shortcomings

1. The `signinum` bare crate is still only a reserved package.
   Users searching crates.io for the umbrella name may land on `signinum 0.0.1`
   rather than the real entry points.

2. Discoverability depends heavily on docs.
   The intended user routing is:
   - whole-slide reader: `statumen`
   - JPEG decoder: `signinum-jpeg`
   - JPEG 2000 / HTJ2K decoder: `signinum-j2k`
   - CLI: `signinum-cli`
   - device surfaces: `signinum-jpeg-metal`, `signinum-j2k-metal`,
     `signinum-jpeg-cuda`, `signinum-j2k-cuda`

3. The GPU story needs disciplined wording.
   Metal has real runtime validation. CUDA currently provides explicit
   device-memory surface/upload plumbing, not CUDA kernel decode.

4. `1.0.0` creates semver expectations.
   Public APIs in `signinum-core`, `signinum-jpeg`, `signinum-j2k`, and
   `signinum-tilecodec` should be treated as stable.

5. The local checkout directories still use the old folder names.
   This is not a GitHub or registry issue, but future agents may see:
   - `/Users/user/Bench/ashlar` for the `signinum` repo
   - `/Users/user/Bench/ziggurat` for the `statumen` repo

## Recommended Next Work

1. Decide what to do with the bare `signinum` crate.
   Best options:
   - publish a tiny facade/landing crate that routes users clearly, or
   - update the reserved crate metadata/README to make the entry points obvious.

2. Re-read every crates.io README and docs.rs landing page.
   The first screen should make the install target obvious.

3. Add explicit migration notes in both repos.
   A short `MIGRATION.md` or README section should map retired crate names to
   current crate names.

4. Audit docs for CUDA wording.
   Avoid language that implies NVIDIA decode acceleration exists today.

5. Consider whether `signinum-cli` should be installable as the primary
   quick-start artifact for users trying the codec family.

6. Rotate or revoke the crates.io token used during this publish/yank session.

## Do Not Repeat

- Do not publish a public oracle/comparison helper unless external users are
  supposed to depend on it.
- Do not yank a retired crate before its replacement is visible on crates.io.
- Do not describe yanks as deletes; yanked crates remain available to existing
  lockfiles.
- Do not force-push either renamed repository.
