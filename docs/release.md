# Release Policy

The current workspace version is `j2k` 0.11.0. Its publication requires the
candidate and tag gates below. The published 0.10.0 line remains
security-supported and carries the release-scoped Part 1 and selected
Part 15 T.803 decoder evidence described in
[`T.803 conformance`](t803-conformance.md).
Runtime backend selection defaults to `Auto`; CPU remains the portable baseline
while supported device paths are selected only with validation and benchmark
evidence.

## Release status

| Version | Distribution state | Security support |
| --- | --- | --- |
| `0.11.0` | Current release line. Distribution is recorded in the [GitHub release](https://github.com/frames-sg/j2k/releases/tag/v0.11.0) and [crate registry](https://crates.io/crates/j2k/0.11.0) after the required gates pass. | Security-supported. |
| `0.10.0` | Published on crates.io from annotated tag `v0.10.0`, with reviewed architecture-transition API evidence. | Supported. |
| `0.9.0` | Published on crates.io from annotated tag `v0.9.0`, with reviewed `objc2-metal` API-break evidence. | Supported. |
| `0.8.1` | Previous crates.io release from annotated tag `v0.8.1`. | Supported. |
| `0.8.0` | Previous crates.io release from annotated tag `v0.8.0`. | Supported. |
| `0.7.5` | Previous crates.io release. Its `j2k-ml` CPU feature works, but its CUDA and Metal features have the clean-consumer defect described below. | Supported, with the stated `j2k-ml` accelerator exception. |
| `0.7.3` | Previous published release line. | Supported. |
| `0.7.2` | Previous published release line. | Supported. |
| `0.7.1` | Previous published release line. | Supported. |
| `0.7.0` | Previous published release line. | Supported. |
| `0.6.x` | Previous published release line. | Supported for security fixes during the pre-1.0 transition. |
| `<0.6` | Historical releases. | Unsupported. |

Version `0.9.0` was published from annotated tag `v0.9.0`, which peels to
commit `b197f01ab4b9271f1cbc36921755a5b9d588bd5a`. The
[hosted validation](https://github.com/frames-sg/j2k/actions/runs/31427966052)
and [CUDA/Metal full validation](https://github.com/frames-sg/j2k/actions/runs/31427977279)
produced the exact-SHA evidence attached to the
[GitHub release](https://github.com/frames-sg/j2k/releases/tag/v0.9.0). The
[tag-triggered publish workflow](https://github.com/frames-sg/j2k/actions/runs/31434104062)
verified all five T.803 reports and published all 19 crates to crates.io.

Version `0.8.1` was published from annotated tag `v0.8.1`, which peels to
commit `f92646d0e6f0d0ef6c1e60b60beaad29da1afd3b`. The
[CPU validation](https://github.com/frames-sg/j2k/actions/runs/31140212203)
and [CUDA/Metal adapter validation](https://github.com/frames-sg/j2k/actions/runs/31141587695)
produced the exact-SHA evidence attached to the
[GitHub release](https://github.com/frames-sg/j2k/releases/tag/v0.8.1). The
tag-triggered workflow verified all five T.803 reports before publishing the
crates.

Version `0.8.0` was published from annotated tag `v0.8.0`, which peels to commit
`53e0ad3d4f75f492af55413e0dab5a5834bd09c6`. The
[tag-triggered publish workflow](https://github.com/frames-sg/j2k/actions/runs/30425822681)
validated all 19 registry targets and published the release to crates.io.
GitHub Pages is served directly from `main/docs`.

Version `0.7.5` is an explicitly reviewed source-compatibility exception to
the normal patch policy. Its wrapper-removal migrations are recorded under
the dated `0.7.5` heading in the [`CHANGELOG`](../CHANGELOG.md), and its reviewed
API evidence is compared directly with the published `v0.7.3` baseline.

The previous `j2k-ml 0.7.5` `cuda` and `metal` features do not compile for a
clean registry consumer because they reference CubeCL and wgpu interop methods
that are absent from the selected registry releases. The `cpu` feature is
unaffected. The published `0.8.0` adapters instead perform accelerator codec
decode, explicit decoded-pixel readback, and ordinary Burn tensor upload using
released public APIs. Do not recommend the defective 0.7.5 accelerator
features.

Version `0.8.0` was intentionally source- and behavior-incompatible
with `0.7.5`: decode entry points become strict by default, explicit leniency
is limited to the documented JP2/JPH metadata recoveries, warnings report
actual recovery rather than lenient configuration, and
`J2kDecodeWarning::LenientDecodeMode` becomes
`J2kDecodeWarning::LenientMetadataRecovery`. The [reviewed API
report][v0.8.0-api-report] records the release's generated signature diff, and
the adjacent [review configuration][v0.8.0-api-review] contains the exact
source- and behavior-break ledger with migrations.

Version `0.8.1` compares directly with published `v0.8.0`. Its [reviewed API
report](release-evidence/public-api/reviewed-public-api-diff-0.8.1.md) and
[review configuration](release-evidence/public-api/public-api-review-0.8.1.yml)
record an additive-only public surface: exact-resolution and sRGB/ICC decode
APIs, encode-stage context, and the shared irreversible midpoint calculation.
The one-time `0.7.5` to `0.8.0` transition allowance has been removed.

Version `0.9.0` compares directly with published `v0.8.1`. Its [reviewed API
report](release-evidence/public-api/reviewed-public-api-diff-0.9.0.md) and
[review configuration](release-evidence/public-api/public-api-review-0.9.0.yml)
record the intentional expert Metal API break: `metal-rs` device, queue,
buffer, texture, descriptor, size, and pixel-format types become retained or
borrowed `objc2-metal` objects and values. Callers construct texture descriptors
directly; the obsolete helper and unreachable raw-message-send errors are
removed. The break ledger enumerates every removed item in the four affected
Metal crates. The one-time transition was consumed by `0.9.0`.

The published `0.10.0` pre-1.0 minor release compares directly with published
`v0.9.0` at peeled commit
`b197f01ab4b9271f1cbc36921755a5b9d588bd5a`. Its
[reviewed API report](release-evidence/public-api/reviewed-public-api-diff-0.10.0.md)
and [review configuration](release-evidence/public-api/public-api-review-0.10.0.yml)
record the intentional architecture transition. Most generated removals are
canonical defining-path changes whose supported root re-exports remain. The
break ledger also records moving `transcode_kernels_built` from the low-level
CUDA runtime to the CUDA transcode engine and generalizing the Metal resident
codestream handoff to `DeviceCodestream`. This one-time transition applies only
to the `0.10.0` release and is now disabled.

Version `0.11.0` compares against published `v0.10.0`. Its
[API report](release-evidence/public-api/reviewed-public-api-diff-0.11.0.md) and
[review configuration](release-evidence/public-api/public-api-review-0.11.0.yml)
cover the new graph-submission support crate, JPEG classification and ICC APIs,
and lossy HT quality-factor options. The experimental MPSGraph adapter removes
four demonstration/reference helpers; applications construct their graphs with
`MpsGraphProgram::new` and keep reference calculations in their own test code.
The pre-1.0 minor-version increment reflects that source-compatibility change.
Conformance wording for this version requires its own exact-candidate evidence;
the published 0.10.0 reports are historical evidence, not a substitute.

Version `0.7.3` retained the API contract introduced by `0.7.1`, which
intentionally contracted parts of the published pre-1.0 `0.6.2` API. It does
not claim source compatibility with `0.6.x`. The
[`CHANGELOG`](../CHANGELOG.md) provides migration notes, and the [reviewed API
report][v0.7.3-api-report] records the additions, removals, and changed
signatures. That report was regenerated, independently reviewed, and verified
for the published tag. Any report prepared for a future release remains
provisional until it is regenerated and verified after that release's final
source freeze.

[v0.7.3-api-report]: https://github.com/frames-sg/j2k/blob/v0.7.3/engineering/reviewed-public-api-diff-0.7.3.md
[v0.8.0-api-report]: https://github.com/frames-sg/j2k/blob/v0.8.0/engineering/reviewed-public-api-diff-0.8.0.md
[v0.8.0-api-review]: https://github.com/frames-sg/j2k/blob/v0.8.0/engineering/public-api-review-0.8.0.yml

## Candidate freeze and exact-SHA evidence

Finish source, generated artifacts, documentation, changelog, and package
metadata before freezing a candidate. The freeze starts only from a clean
worktree:

```bash
test -z "$(git status --porcelain)"
RC_SHA=$(git rev-parse HEAD)
cargo xtask release-integrity --publish
cargo xtask package
```

Both offline candidate gates run from that clean commit. A failure or any
tracked correction invalidates `RC_SHA`; commit the correction, choose a new
candidate SHA, and rerun the local and exact-SHA evidence.

ISO/IEC 15444-4:2024 / ITU-T T.803 v3 claim eligibility is scoped independently.
CPU wording requires exact-SHA reports from Linux x86-64, macOS arm64, and
Windows x86-64. CUDA and Metal adapter wording each requires that adapter's own
exact-SHA real-hardware report. Every report in the selected scope must contain
all selected cases with no skips. An unavailable adapter blocks only its own
claim; it does not invalidate or suppress a complete CPU result. The optional
`--scope all` verifier is a coordinated-release convenience, not the definition
of CPU compliance. Current status is recorded in
[`docs/t803-conformance.md`](t803-conformance.md).

During release preparation, the changelog keeps a real `## [Unreleased]`
heading and a structured staged-version line. As the final release-preparation
edit before candidate freeze, replace that heading with
`## [<workspace-version>] - YYYY-MM-DD` using the actual intended tag date and
update every staged-document reference that still says the notes are under
`Unreleased`. Do not guess the date early. Any later date or note change creates
a new candidate and requires the exact-SHA gates again.

Move the intended protected `origin/main` tip to exactly `RC_SHA` through the
repository's normal reviewed push/merge workflow. Let `full-validation.yml`
finish for that push, then dispatch one `gpu-validation.yml` run with
`target=all` and `mode=full` for that exact commit. CUDA and Metal execute in
parallel within that run. Verify the evidence only after all three release jobs
have completed:

```bash
test "$(git rev-parse origin/main)" = "$RC_SHA"
cargo xtask release-status --sha "$RC_SHA" --scope all
```

Any tracked edit creates a new candidate: commit it, choose a new `RC_SHA`, and
rerun all exact-SHA evidence. Only after the verifier succeeds may the release
maintainer create an annotated `v<workspace-version>` tag that peels to
`RC_SHA`. Push that tag explicitly; do not use `--follow-tags`, move an existing
release tag, or treat a GitHub Pages deployment as release evidence.

Before final candidate freeze, complete both structured fields in every
`[patch.crates-io]` path override's `PATCH_PROVENANCE.md` record with the
actual reviewer identity and review date. The publish-integrity command
discovers these records from the workspace manifest and fails if any one is
missing or unapproved. The date must be a calendar-valid `YYYY-MM-DD`; never
infer either value from commit metadata. This generic validation remains in
force even when the current workspace has no repository-local path patches.
Also have a repository administrator enable GitHub private vulnerability
reporting under **Security** settings before exact-SHA candidate verification.
The authenticated candidate verifier reads that repository setting and fails
closed unless it reports enabled; the later tag verifier reuses the same
prerequisite.

## Versions and publish order

[`release-crates.json`](../release-crates.json) is the ordered release manifest
and source of truth for release-integrity, package construction, registry
recovery, API tiers, documentation coverage, semver scope, and publication.
Schema 2 records only ordered crate names and one of `stable`, `experimental`,
`implementation`, or `binary` as each crate's `api_contract`. Release scripts
must not publish from stale hard-coded crate/version pairs or maintain a second
authored list of registry-independent crates.

The manifest must contain every crates.io-eligible workspace member exactly
once; a member restricted to another registry is not crates.io eligible.
Library tiers require a library target, while the binary tier requires a binary
target and no library target. Every path dependency between release crates,
including dev-dependencies, must use the exact
`=<workspace.package.version>` requirement and resolve to that workspace
crate—not to a same-named registry or Git dependency. Normal and build
dependencies, including optional and target-specific edges, determine publish
ordering. Dev-dependencies do not.

Real publishes must run from tag `v<workspace.package.version>`. All
publishable crates must share that workspace version. If a crate version is
already on crates.io, the publish script fails by default; set
`CRATES_IO_ALLOW_PUBLISHED_RERUN=true` only for an intentional idempotent
rerun. A valid partial retry may contain only an already-published prefix of
the dependency-ordered list below, and every published `.crate` SHA-256 must
match the archive packaged locally from the exact tag. A published crate after
an available crate, or any checksum mismatch, is inconsistent state and fails
closed.

Publish in this order:

1. `j2k-core`
2. `j2k-profile`
3. `j2k-types`
4. `j2k-codec-math`
5. `j2k-cuda-build-support`
6. `j2k-cuda-runtime`
7. `j2k-cuda-j2k-engine`
8. `j2k-cuda-jpeg-engine`
9. `j2k-cuda-transcode-engine`
10. `j2k-metal-support`
11. `j2k-mpsgraph-support`
12. `j2k-native`
13. `j2k-jpeg`
14. `j2k-tilecodec`
15. `j2k`
16. `j2k-transcode`
17. `j2k-transcode-cuda`
18. `j2k-jpeg-metal`
19. `j2k-metal`
20. `j2k-transcode-metal`
21. `j2k-jpeg-cuda`
22. `j2k-cuda`
23. `j2k-ml`
24. `j2k-mpsgraph`
25. `j2k-cli`

Publish preflight must account for staged unpublished workspace dependencies.
Use the repo-owned package gate from a clean worktree:

```bash
cargo xtask package
```

That gate applies package listing and dry-run checks according to dependency
availability:

```bash
cargo package --list
cargo package --no-verify
cargo publish --dry-run
```

The gate lists all 24 package contents. It derives dependency closure and
registry independence from locked Cargo metadata, then constructs `.crate`
archives with `cargo package --no-verify` for the 20 staged packages whose
workspace dependencies are not yet available from crates.io. The four derived
registry-independent packages (`j2k-core`, `j2k-profile`, `j2k-types`, and
`j2k-codec-math`) run
`cargo publish --dry-run`, including Cargo's package verification build. Manual
publish-workflow runs remain dry-run-only: they validate the manifest and
construct every local archive without receiving the crates.io token.

After constructing `j2k-ml`, the gate creates a consumer outside the workspace
and compiles the packaged source against registry CubeCL and wgpu crates. Linux
checks `cpu`, `cuda`, and `cpu,cuda`; macOS checks `cpu`, `metal`, and
`cpu,metal`, and the combined host feature set also builds documentation. The
consumer imports and instantiates the corresponding public decoder APIs.
Temporary `[patch.crates-io]` entries cover only unpublished J2K workspace
crates; third-party overrides are forbidden. Run the focused form with:

```bash
cargo xtask j2k-ml-package-smoke
```

The Metal package-consumer lane also builds `j2k-mpsgraph` from its staged
crate archive and staged workspace dependencies:

```bash
cargo xtask package-consumer-smoke --target metal
```

An accelerator failure in this consumer is a distribution blocker even when
the same feature builds inside the repository through a workspace-root patch.

Before publication, the hosted preflight verifies that the checkout `origin` is
the exact workflow repository, no draft, prerelease, or published GitHub
Release exists for the tag, every target crate version has a determinate
crates.io state, and all archives package locally. Only an exact HTTP 404 means
a version is available; authentication errors, authorization failures,
malformed responses, and checksum mismatches stop publication. On an
intentional partial retry, `CRATES_IO_ALLOW_PUBLISHED_RERUN=true` permits only
the checksum-matched already-published prefix without moving the tag.

After `crates-io-publish` environment approval, one runner repeats the canonical
tag and prefix proof, packages all 23 archives, and publishes the remaining
manifest entries sequentially with `cargo publish --locked -p <crate>`. Cargo's
verification build stays enabled. There are no unconditional registry sleeps;
only retryable transport, HTTP 429, or server failures are retried with bounded
5, 15, and 30 second delays. The publisher re-queries and checksum-validates the
entire prefix before each retry. Authentication, authorization, package
verification, manifest, version, and checksum failures are never retried.

Run this before publishing:

```bash
cargo xtask codec-math-codegen
cargo xtask release-integrity
cargo xtask release-integrity --publish
cargo xtask public-support --final
```

The codec-math codegen gate verifies generated Rust and Metal fragments against
the Rust source of truth. The integrity gate parses lockfile-strict cargo
metadata with `cargo metadata --locked --no-deps`, `release-crates.json`,
manifests, `.github/workflows/publish.yml`, and this release document. It fails if a
crates.io-eligible workspace crate is missing from the dependency-ordered
manifest, docs.rs metadata, published-library semver/doc gates, or release
docs; if an API tier does not match its Cargo targets; if an internal
requirement is not exact; or if dependency order is invalid.

The ordinary integrity mode is an offline pre-candidate check. It accepts the
structured `Unreleased` state before a candidate, the frozen dated state used
while validating a release commit, and a fresh `Unreleased` section above the
dated release after publication. `--publish` remains offline but requires
exactly one dated heading for the workspace version, rejects the provisional
changelog markers, and requires completed patch-review approval fields. The tag
workflow separately uses the authenticated GitHub verifier to
confirm private vulnerability reporting, the annotated tag, and exact-SHA
hosted/GPU evidence. A direct real invocation of `scripts/publish-crate.sh`
independently requires the expected annotated Git tag to exist and peel exactly
to `HEAD`, treats `GITHUB_REF_NAME` only as an additional consistency check,
and rejects tracked or untracked worktree changes. It derives the canonical
repository identity from `[workspace.package].repository`, normalizes secure
HTTPS, scp-style SSH, and `ssh://` checkout URLs, and requires the checkout
`origin` to match that identity. It then queries `origin` directly and requires
the exact remote tag object and its peeled commit to match the verified local
annotated tag and `HEAD`. Any Git URL rewrite must still resolve to the same
canonical identity. Origin and remote-tag failures stop before Cargo or any
registry operation, and diagnostics do not print remote URLs or transport errors
that could contain credentials. Finally, the script reruns the strict offline
integrity mode so it cannot bypass those source and metadata checks.

The public-support gate verifies that the JPEG 2000 Part 1, JP2, HTJ2K Part 15,
JPH, known-limitation, and publication-gate rows remain synchronized with tests
and their support inventory. That implementation boundary does not substitute
for the T.803 exact-reference gate or authorize a conformance claim.

## Required gates

After the candidate is frozen and committed, hosted CI must pass for exactly
`RC_SHA` before release authorization:

- formatting
- tests
- clippy
- authoritative strict Clippy via `cargo xtask clippy-strict`
- panic-surface ratchet via `cargo xtask panic-surface`
- codec math fragment freshness via `cargo xtask codec-math-codegen`
- release integrity
- package validation
- semver checks for stable packages
- docs and stable API inventory
- benchmark target compilation
- unsafe audit
- bounded fuzz run
- coverage via `cargo xtask coverage`
- hosted macOS Metal compilation and pure tests via `cargo xtask metal-compile`
- exact-reference T.803 CPU reports on Linux x86-64, macOS arm64, and Windows
  x86-64 when the release declares CPU Profile/Cclass wording, with all selected
  cases present and passing
- an exact-reference CUDA or Metal adapter-IUT report from real hardware when
  the release declares wording for that adapter, with CPU/device/hybrid stages
  disclosed per case; compilation alone is never adapter conformance evidence
- packaged clean-consumer checks for `j2k`, `j2k-cuda`, and `j2k-metal`
- route-parity tests for every fixed `Auto` decision; any newly promoted hybrid
  threshold additionally requires verified external Criterion evidence and its
  artifact hash

Changed-line coverage records production Rust across CPU and accelerator
crates. The host lane enforces 80% across all changed production Rust and an
independent 80% release-critical gate. Accelerator lanes report raw host-Rust
coverage as audit evidence and enforce 80% for release-critical routing,
validation, allocation, ownership, public-API, parser, security, and error
boundaries. Broad accelerator implementation correctness is enforced by exact
CPU/backend output parity and fail-closed hardware suites, not by tests written
only to execute lines. GPU-heavy changes therefore require self-hosted
`gpu-validation` evidence. The Metal job delegates to
`cargo xtask release-metal`, which requires macOS, forces the strict runtime
gate, rejects GPU skip markers, checks named runtime sentinels and count floors,
and runs the exact declared ignored hardware-test inventory.

Benchmark compilation is a release build-health gate, not a performance
regression threshold. A release may claim performance only when the relevant
CPU, Metal, or CUDA benchmark artifacts are recorded in
[`docs/benchmark-evidence.md`](benchmark-evidence.md) or an attached run
bundle. `cargo xtask j2k-perf-guard --lane host` is available for explicit CPU Criterion
median regression signoff, but it is not part of the default release gate until
the release checklist supplies a baseline ref and artifact retention policy.
GPU performance signoff remains hardware-runner evidence, not hosted CI.

Hosted macOS runs `metal-compile` and does not claim hardware validation. A
release requires `release-metal` on a self-hosted Apple Silicon Metal runner;
missing devices, zero selected tests, skipped runtime paths, and inventory drift
are failures. These checks retain the per-backend minimum test count floors and
named runtime sentinels for every Metal-facing package. J2K Metal Criterion
bench signoff is reset until new narrow profiling benches are added.

Version 0.9.0 replaces the deprecated `metal-rs` host binding with the pinned
`objc2`, `objc2-foundation`, and `objc2-metal` stack. The workspace and
published Metal adapters no longer require a top-level crates.io patch or the
unmaintained `block 0.1.6` crate. Release review must keep the objc2 versions
unified and run the dependency-tree proof plus the normal Metal build and
runtime gates.

CUDA validation requires a self-hosted CUDA environment for runtime and NVIDIA performance evidence. CUDA paths use J2K-owned CUDA kernels, cuda-runtime integration, and CUDA device memory surfaces for supported shapes. NVIDIA performance claims require recorded self-hosted benchmark output.

Whole accelerator crates are not coverage exclusions. The changed-line
denominator covers executable production and required build-script Rust.
Syntax-level `#[cfg(test)]` code, Cargo test targets, and example/bench/fuzz
targets are reported as separate non-production source dispositions rather than
being mislabeled as uncovered production.

Reviewed non-host-instrumentable exclusions are exact and named: CUDA SIMT
device Rust, generated cuda-oxide host scaffolds, the shared SIMT prelude,
CUDA/NVTX FFI declaration spans, the embedded MSL string body, the generated
codec-math DWT fragment. Every generated line must match its named freshness,
integrity, or runtime-parity evidence. Metal and CUDA lanes publish separate
LCOV and summary artifacts and remain required before release.

Each coverage lane forces `CARGO_LLVM_COV_TARGET_DIR` and
`CARGO_LLVM_COV_BUILD_DIR` to the same unique empty directory and uses only
build-script outputs captured from that invocation for custom `cfg`
classification. This makes byte-identical build-script reruns valid current
evidence without admitting retained scopes from an earlier run. Every selected
package with a Cargo custom-build target must have current output; missing or
conflicting package evidence fails the gate. A custom cfg value not established
by current evidence remains unknown, so both it and its negation stay in the
changed-source denominator rather than disappearing as inactive.

## Published and unpublished crates

Published crates must declare package README files and docs.rs metadata.
Unpublished tooling and oracle helpers remain local even when versioned with the
workspace.

`j2k-test-support` is an unpublished dev helper. Comparator crates and
automation-only tooling are not runtime API.
