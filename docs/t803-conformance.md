# ISO/IEC 15444-4 / ITU-T T.803 Conformance

Status: **Part 1 and selected Part 15 points published for 0.11.0 with
exact-clean-SHA release evidence**

Published formal decoder wording for release `0.11.0`:

- `j2k` CPU IUT:
  - JPEG 2000 Part 1: **Profile-1 Cclass-1 compliant; Profile-1 Cclass-1HF compliant; Annex G JP2 reader compliant.**
  - HTJ2K Part 15: **DS1-HM Cclass-1h, MMAGB 15**, including DS1-HT,
    DS0-HM, and DS0-HT subset evidence; **Cclass-1HFh, MMAGB 20**; and
    **Annex G JPH reader compliant at MMAGB 15**.
- `j2k-cuda` and `j2k-metal`: separate adapter-IUT results for the same Part 1
  and selected Part 15 points, with every CPU, device, hybrid, and transfer
  stage disclosed per case. Neither adapter is described as device-native.

The implemented harness targets ISO/IEC 15444-4:2024 / ITU-T T.803 v3. Part 4
defines JPEG 2000 conformance-testing procedures and reference comparisons; it
is not another codestream syntax or a performance benchmark. The wording above
is tied to the five exact-SHA reports attached to the
[v0.11.0 release](https://github.com/frames-sg/j2k/releases/tag/v0.11.0). All
reports identify the same immutable release commit and contain no
development-only feature evidence.

## Decoder evidence scope

| IUT | Evidence wording | Route boundary |
| --- | --- | --- |
| `j2k` CPU | Published `0.11.0` Part 1 Profile-1 Cclass-1, Profile-1 Cclass-1HF, and Annex G JP2 reader wording, plus selected Part 15 DS1-HM Cclass-1h at MMAGB 15, Cclass-1HFh at MMAGB 20, and Annex G JPH reader at MMAGB 15. | CPU implementation under test. |
| `j2k-cuda` | Published `0.11.0` adapter IUT evidence for the same Part 1 and selected Part 15 points. | Parsing, Tier-1, transforms, output, and transfers are reported per case as CPU, CUDA, or not used. |
| `j2k-metal` | Published `0.11.0` adapter IUT evidence for the same Part 1 and selected Part 15 points. | Parsing, Tier-1, transforms, output, and transfers are reported per case as CPU, Metal, or not used. |

CPU assistance is permitted for the adapter IUTs. Any such route is labelled
`hybrid`; it is not described as device-native. Annex G JP2 color and component
normalization currently runs through disclosed CPU stages for the GPU adapters.
JPX / Part 2 is outside this scope, except for JP2-compatible JPX input required
by Annex G. T.803 v3 provides no Cclass-2h ETS, so this project makes no formal
Cclass-2h claim; Cclass-2h-scale resource and boundary checks are extended
project evidence only.

The project does not use generic “full Part 1 compliant” or “full Part 15
compliant” labels. Every exact Profile/Cclass/MMAGB claim must be tied to
published reports for one immutable release SHA.

## Published 0.11.0 result

All five attached reports identify source commit
`09d746a7b040258eb5dd505b44384eec0152a8b9` and pass all 160 selected decoder
cases with zero skips. The CPU reports cover Linux x86-64, macOS arm64, and
Windows x86-64; the CUDA and Metal reports come from real hardware. The
release-status verifier checked all five report contents after the complete
hosted and GPU release workflows passed. The JSON reports and their Markdown
renderings are attached to [release 0.11.0](https://github.com/frames-sg/j2k/releases/tag/v0.11.0).
The historical 0.9.0 details below describe that earlier release only.

## Published 0.9.0 result

The macOS arm64, Linux x86-64, and native Windows x86-64/MSVC CPU reports
attached to [release 0.9.0](https://github.com/frames-sg/j2k/releases/tag/v0.9.0)
each pass all **160 selected cases with zero skips** at exact SHA
`b197f01ab4b9271f1cbc36921755a5b9d588bd5a`: 90 Part 1 decoder/JP2 cases and
70 Part 15 decoder/JPH cases. The Part 15 total comprises 60 formal codestream
comparisons plus all ten Annex G JPH families. For each BSET, the harness
selects the largest BMAGB not exceeding the claimed MMAGB.

Both real-hardware adapter reports also pass 160/160 and record the same honest
aggregate: **0/160 device-native, 81/160 hybrid, and 79/160 CPU-routed**. The
Part 1 split is 48 hybrid and 42 CPU-routed; the Part 15 split is 33 hybrid and
37 CPU-routed. CUDA ran on an NVIDIA GeForce RTX 4070 SUPER and Metal on an
Apple M4 Pro. Parsing ran on CPU in every case. Hybrid cases moved supported
Tier-1, dequantization, IDWT, MCT, color/output, and transfer work to the named
device. Although the aggregate splits coincide, the reports contain
backend-specific production dispatch counters—for example, CUDA uploaded byte
counts while Metal records host-input counts, and per-case dispatch counts
differ. The stage labels are observed execution, not capability-table
predictions.

The CPU Annex D/F encoder matrix passes 56/56 cases: 55 pass through the pinned
T.804 OpenJPEG decoder, while the mandatory HT+RGN capability case that
OpenJPEG 2.5.3 rejects passes through the independently pinned OpenHTJ2K decoder
as supplemental interoperability evidence. The CUDA and Metal matrices each
pass 35/35 through OpenJPEG; CUDA records 34 hybrid encoder routes and one
CPU-routed case, while Metal records 33 hybrid routes and two CPU-routed cases.
These encoder results are informative evidence, not the formal decoder claim.

All five reports identify the exact release SHA above. The release assets
include JSON and Markdown renderings covered by `SHA256SUMS`; the schema-7 JSON
digests are:

| IUT/platform | Report SHA-256 |
| --- | --- |
| CPU/macOS arm64 | `9638c41d41842e99f385ea71fd8c83791416d512233a5fed43081c9964d5092b` |
| CPU/Linux x86-64 | `509502eddf48ebe5d77d614234746693be30e1c537f1890af67edf4593de64eb` |
| CPU/Windows x86-64 MSVC | `57569214409450b27edb8ba1634e4e2fc6778f52370ef4b98a0ed94b093c3a9a` |
| CUDA/Linux x86-64, RTX 4070 SUPER | `84c52044254278bb45e48664d94765d8fd44bdd07fc6b03d610fa983f5944039` |
| Metal/macOS arm64, M4 Pro | `2206dd313ad5f10a16652ecc8b08e5c9f4a5f72eb0b31be361411a0462644d0c` |

These hashes are audit anchors for the published release result. T.803 does
not establish robustness, security, adoption, or performance.

The former `c1-c0p0-13` failure was an IUT harness defect. The codestream has
257 components and enables the reversible component transform. T.803 B.2.5
requires Cclass-0 comparison before inverse MCT, so its first-component
reference is 1; the Cclass-1 component-0 reference after inverse RCT is 0. The
harness had incorrectly inferred MCT use from a display colorspace, which is
unknown for this component count. It now reads the COD transform flag and
transform kind through the existing codestream inspector, reconstructs the
pre-MCT component for Cclass-0, and reports the MCT stage from the same
metadata. A 257-component regression test prevents the colorspace inference
from returning.

The report now independently decodes every selected codestream whose COD
enables MCT and whose SIZ declares more than four components. For `p0_13.j2k`,
the production decoder and vendored OpenJPEG 2.5.3 matched component metadata
and samples exactly for all 257 components before any T.803 normalization. Both
canonical native-output hashes are
`a01808e0cbf14288274188c8bebb5ef8c2aa46304eca964a2ac71bed1713c1fd`.
As a second manual check, OpenJPEG CLI 2.5.4 emitted 257 PGX components with
zero sample mismatches; the concatenated one-sample component payload SHA-256
was `54acfbfedc4d8da40f76f275e1a98f10af8ef1fb9fb39e5a67a00aabcbe6597c`.

The investigation independently confirmed byte-identical `p0_13.j2k`,
`c0p0_13.pgx`, and `c1p0_13-0.pgx` payloads in ITU's current attachment,
ISO's 2024 electronic insert, and ITU's 2002 suite. No corpus mapping, hash,
dimension, precision, signedness, reduction, crop, tolerance, or comparison
arithmetic was changed to obtain the passing result.

## Evidence commands

The official corpus is fetched only from the URL and archive digest pinned in
`corpus/j2k-conformance/t803-v3.toml`:

```bash
cargo xtask t803 fetch
cargo xtask t803 run --iut cpu --suite all
cargo xtask t803 run --iut cuda --suite all
cargo xtask t803 run --iut metal --suite all
```

`fetch` rejects unapproved redirects, archive or file hash drift, unsafe archive
entries, duplicate paths, unexpected required-case names, and resource-limit
violations. The copyrighted corpus stays under `target/t803/`. Only versioned
JSON/Markdown reports and hashes may be retained.

Add `--development` only while iterating on a dirty tree. Development runs use
the same corpus and comparisons, but their reports are intentionally ineligible
for release verification. The exact clean candidate runs omit that flag.

Release eligibility is scoped independently. CPU wording requires the three
CPU operating-system reports; each adapter wording requires only that
adapter's real-hardware report. An unavailable adapter blocks its own claim,
not the CPU claim:

```bash
cargo xtask t803 verify --scope cpu --candidate-sha "$RC_SHA" \
  --report path/to/cpu-linux.json \
  --report path/to/cpu-macos.json \
  --report path/to/cpu-windows.json
cargo xtask t803 verify --scope cuda --candidate-sha "$RC_SHA" \
  --report path/to/cuda.json
cargo xtask t803 verify --scope metal --candidate-sha "$RC_SHA" \
  --report path/to/metal.json
```

`--scope all` verifies all five reports together. A future release that carries
the Part 15 wording uses that coordinated scope in the tag-publish workflow;
independently, an unavailable adapter invalidates only its own adapter claim
and does not erase a complete CPU result.

All 160 selected Part 1/Part 15 decoder and JP2/JPH cases must be present with
no skips, every report must pass, source and corpus hashes must match, and the
IUT/platform/route identity must match the required lane. Reports are rejected
when a route labels CPU-assisted work as device-native.

## Encoder evidence

The CPU, CUDA, and Metal Annex F implementation compliance statements and the
stable pairwise/boundary matrix live in `corpus/j2k-conformance/`. The pinned
T.804 OpenJPEG reference implementation decodes 55/56 CPU cases and all 35
CUDA and Metal cases. OpenJPEG 2.5.3 rejects HT code-blocks carrying RGN before
decoding, so the retained CPU HT+RGN capability case uses the independently
pinned OpenHTJ2K decoder and is labelled supplemental interoperability evidence,
not T.804 evidence. Reference-decode success is the Annex D legality result;
lossless output must also match the source exactly. Lossy rate and PSNR checks
are separate project quality gates.

Encoder testing is informative under T.803 and is not the same formal claim as
decoder compliance. Accelerator dispatch and fallback stages are reported for
every encoder case.

T.803 does not establish robustness, security, adoption, or performance. Those
properties require their own fuzzing, security review, external workload, and
benchmark evidence.
