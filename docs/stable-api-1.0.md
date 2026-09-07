# Stable API Policy

The stable API inventory is generated. The human-maintained policy is small:
stable crates must preserve the public codec contracts, while experimental
adapters may evolve until promoted.

## Generated snapshot

The generated item-level companions are:

- `docs/stable-api-1.0.public-api.txt`
- `docs/stable-api-1.0.implementation-public-api.txt`

Regenerate or check them with:

```bash
cargo xtask stable-api
cargo xtask stable-api --write
```

This task must run on macOS with `cargo-public-api` `0.52.0` installed
(`cargo install cargo-public-api --version 0.52.0 --locked`) and the pinned
`nightly-2026-06-28` toolchain available. Both passes explicitly target
`aarch64-apple-darwin` so target-gated Metal APIs and rustdoc formatting do not
silently change with the runner host or floating nightly channel.

The ordinary snapshot uses `RUSTDOCFLAGS=-D warnings` so its comparison with
the published 0.9.0 snapshot keeps the same scope. A second pass adds
`--document-hidden-items` and records only the extra rustdoc-hidden items in
the implementation snapshot. Rustdoc can rewrite equivalent re-export paths
when hidden modules become visible, so the generator forms a conservative full
candidate inventory from the union of both passes and writes its lexically
sorted difference from the ordinary pass. This guarantees that the combined
inventory remains a superset of the ordinary contract while retaining rewritten
path variants for review rather than silently dropping reachable API. An empty
full cargo-public-api pass fails the gate; an empty per-package hidden-only
difference is recorded truthfully. The 0.9.0 baseline comparison continues to
use only the ordinary snapshot. Those
adapters are implementation-facing, but they are still reachable Rust API and
therefore remain in the reviewed inventory. Do not use `#[doc(hidden)]` as a
compatibility escape hatch.

The published 0.7.5 artifact recorded both ordinary and hidden-enabled passes
with the same generator, rustdoc, and target pins. The historical 0.8.0 semver
report compares its ordinary inventory with 0.7.5, and the 0.8.1 report
compares that release directly with published 0.8.0. The 0.9.0 report compares
the published release directly with published 0.8.1, the 0.10.0
report compares that release directly with published 0.9.0, and the 0.11.0
report compares against published 0.10.0. All reports also
record each package's complete hidden-inventory count and fingerprint.
Every semver invocation collects both live passes, compares both committed companions, and
requires exact ordinary added/removed fingerprints plus the hidden
count/fingerprint in
`docs/release-evidence/public-api/public-api-review-0.11.0.yml`.
Nonempty hidden inventories also require a package-specific hidden rationale.

The 0.8.0 review file contains the reviewed 0.7.5-to-0.8.0 break ledger. The
0.8.1 review file has no break-ledger entries because its generated diff is
additive only. The 0.9.0 review file deliberately enumerates every removed
`metal-rs` expert signature across `j2k-metal-support`, `j2k-jpeg-metal`,
`j2k-metal`, and `j2k-transcode-metal`, plus the obsolete texture-descriptor
helper and unreachable raw-message-send errors, with direct `objc2-metal`
migration guidance and no compatibility layer.
Source-break entries must enumerate every exact removed API item, package,
summary, and migration. Validation requires that inventory to equal the
generated removed-item set: an omitted, duplicate, unknown, or stale item fails
the gate. Behavior-break entries carry the same summary and migration requirements
but contain no removed items, because their observable changes
cannot be inferred from a Rust signature diff. Consequently, additions,
removals, hidden reachability, and declared behavior breaks are reviewed
together rather than treating a pre-1.0 version bump as sufficient evidence.

The two snapshot files are staged, synchronized, and committed as one
rollback-capable transaction. API generation rejects ambient compiler,
rustdoc, target, wrapper, deployment-target, bootstrap, and encoded flag
overrides that could silently change either pass. Toolchain selection is not
taken from the ambient Cargo process: both passes execute through the pinned
`rustup run` toolchain. `cargo xtask semver`
uses Rust `1.96` and does not accept the former `J2K_SEMVER_TOOLCHAIN` override.

The snapshots record the published workspace's public items and the CLI exit-code
contract expectations. Manual prose in this file must not duplicate that
inventory. The completed 0.7.5-to-0.8.0 comparison is in the generated
[`0.8.0` reviewed API report][v0.8.0-api-report].
That report became release evidence after source freeze and the exact-SHA
local, hosted, Metal, and CUDA gates completed. The completed 0.9.0 comparison
is in the generated
[`0.9.0` reviewed API report](release-evidence/public-api/reviewed-public-api-diff-0.9.0.md),
with its human review in
[`public-api-review-0.9.0.yml`](release-evidence/public-api/public-api-review-0.9.0.yml).
The completed 0.10.0 comparison is in the generated
[`0.10.0` reviewed API report](release-evidence/public-api/reviewed-public-api-diff-0.10.0.md),
with its human review in
[`public-api-review-0.10.0.yml`](release-evidence/public-api/public-api-review-0.10.0.yml).
The completed 0.11.0 comparison and its reviewed experimental MPSGraph removals
are in the generated [API report](release-evidence/public-api/reviewed-public-api-diff-0.11.0.md)
and [review configuration](release-evidence/public-api/public-api-review-0.11.0.yml).
Its exact-SHA release gates passed.

[v0.8.0-api-report]: https://github.com/frames-sg/j2k/blob/v0.8.0/engineering/reviewed-public-api-diff-0.8.0.md

The currently published stable contract is the `0.11.x` line. Version `0.8.0`
intentionally changed the strict-decoding behavior and one warning variant
under Cargo's pre-1.0 compatibility rules. It does not claim source or behavior
compatibility with `0.7.x`; its exact breaks and migrations are in the review
file. Version `0.8.1` adds exact-resolution and sRGB/ICC decode APIs plus
encode-stage context without removing or changing a 0.8.0 item. Version
`0.9.0` intentionally breaks the experimental and implementation-facing Metal
expert APIs by replacing `metal-rs` owners and `*Ref` wrappers with retained or
borrowed `objc2-metal` protocol objects. Texture descriptors are constructed
directly with objc2-metal, and only typed allocation and availability errors
remain. It does not claim Metal expert API source compatibility with `0.8.x`.
Version `0.7.0` similarly contracted parts
of the pre-1.0 `0.6.2` API and did not claim source compatibility with `0.6.x`.

## Stability tiers

`release-crates.json` assigns exactly one `api_contract` to every published
crate. The tier changes the support promise; it does not remove ordinary Rust
visibility or make a public item exempt from patch-release review.

All published libraries are built with missing-docs enforcement, recorded in
both the ordinary and rustdoc-hidden inventories, and checked for patch
compatibility. A public `#[doc(hidden)]` item remains callable Rust API and is
not private merely because rustdoc omits it. Use actual module or item privacy
for internals whenever possible.

- `stable`: `j2k`, `j2k-core`, `j2k-jpeg`, and `j2k-tilecodec`. These are the
  supported general-purpose consumer APIs. Their documented contracts are the
  long-term compatibility promise and should grow conservatively.
- `experimental`: `j2k-native`, `j2k-cuda`, `j2k-metal`, `j2k-jpeg-cuda`,
  `j2k-jpeg-metal`, `j2k-transcode`, `j2k-transcode-cuda`,
  `j2k-transcode-metal`, and `j2k-ml`. Patch releases preserve their public
  APIs. A pre-1.0 minor release may change them only with an explicit reviewed
  break ledger and migration guidance. Their runtime support remains limited
  by feature gates, hardware availability, and `docs/public-support.md`.
- `implementation`: `j2k-codec-math`, `j2k-profile`, `j2k-types`,
  `j2k-cuda-build-support`, `j2k-cuda-runtime`, `j2k-cuda-j2k-engine`,
  `j2k-cuda-jpeg-engine`, `j2k-cuda-transcode-engine`, and
  `j2k-metal-support`. These are published sibling-crate
  interfaces, not supported general-purpose extension APIs. They still receive
  documentation, inventory, and patch-compatibility checks because downstream
  Rust code can call them. A pre-1.0 minor release may revise them with the same
  reviewed break evidence.
- `binary`: `j2k-cli`. Its command, output, and exit-status behavior is governed
  by the CLI contract below rather than a Rust library inventory.

`j2k-codec-math` and `j2k-ml` are included in the published `0.7.5` semver
baseline. Test support, comparators, and xtask automation helpers remain
unpublished and receive no public API compatibility promise.

Patch releases normally preserve the active `0.x` public contract. Version
`0.7.5` is an explicit, maintainer-approved source-compatibility exception for
removing pass-through public wrappers. Its reviewed API diff must enumerate
every contracted item and its changelog must provide migration guidance. This
exception applied only to `0.7.5`.

The completed historical transition locks were intentionally narrow: `0.8.0`
was the only candidate permitted to compare against `v0.7.5`, and `0.9.0` was
the only candidate permitted to compare against `v0.8.1`, as intentional
pre-1.0 breaks. The currently configured semver baseline is published
`v0.9.0` at peeled commit
`b197f01ab4b9271f1cbc36921755a5b9d588bd5a`. The staged `0.10.0` pre-1.0
minor candidate compares directly against that baseline under a one-time
intentional-break transition. Its ledger records every generated removal and
direct migration; the allowance must be disabled after publication.

Before `1.0`, a minor release may intentionally change the contract only under
the same generated evidence, explicit break-ledger, and migration requirements.
Starting with `1.0`, stable crates follow the normal compatibility guarantees
for the declared major version.

## CLI contract

`j2k-cli` currently supports:

- `j2k inspect <file>`
- `j2k transcode <input.jpg> <output.j2k> --htj2k --lossless-53`

Recognized argument-validation failures return exit code `2`; these include an
unknown subcommand, a missing `inspect` file operand, and malformed or
unsupported `transcode` arguments. Runtime failures, including unreadable files
and unsupported codec inputs, return exit code `1`. Successful operational
commands return exit code `0` and write a single summary line to stdout. Help
and an invocation with no subcommand also return `0`, but print usage to stderr.

Additional arguments after the `inspect` file are currently ignored. This is a
known CLI limitation, not a stable contract: callers must not supply trailing
arguments or rely on them continuing to be accepted.
