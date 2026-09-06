# Metal resource reuse and packetization — 2026-09-06

This change removes intermediate CPU payload copies from supported lossy HTJ2K
encoding, restores scratch-buffer reuse, and makes completed shared-memory
readback use direct CPU copies. Immutable pipelines can be shared across live
sessions on the same exact Metal device object.

## Design and compatibility

The lossy path now encodes input conversion, 9/7 transform, quantization, HT
coding, packet-block preparation, ordered packet headers, and parallel payload
copying into one command buffer. Compressed block payloads remain in private
Metal storage. Only HT status records and the completed packet bytes are read
on the CPU; native assembly still owns lossy codestream headers and CAP metadata.
The existing ordered-header and parallel-copy kernels are reused unchanged.
The packet topology planner now accepts borrowed metadata without requiring
transform-buffer ownership. Lossless callers retain their original assembly
offsets, and descriptor bounds still use the actual retained Tier-1 table size.

Private intermediates and shared input/metadata/output buffers are returned to
the session pool only after successful command completion. Validation errors
after completion still permit recycling; command errors discard uncertain
resources. A pool-state error takes precedence over a simultaneous result error
because it affects future session reuse.

`Surface::download_into` copies the checked completed shared range directly into
caller rows, preserving offsets, stride padding, and private-buffer rejection.
Packed readback validates all inputs before allocation or encoder creation,
copies shared ranges directly, and stages only private ranges on the retained
session queue. It preserves the previous aggregate limit of
`min(device.maxBufferLength, DEFAULT_MAX_HOST_ALLOCATION_BYTES)`. The result Vec
is reserved once and initialized by copying, without an initial zero-fill pass.
Host rejection now occurs before opening an encoder, avoiding the prior Metal
assertion when an unfinished encoder was released.

The kernel cache has eight weak metadata slots keyed by registry ID **and exact
device-object address**. OnceLock serializes each immutable group's initialization;
queues, pools, and execution caches remain session-owned. Cache poisoning uses
the existing typed state-error path. Kernels are released after their last owner
is dropped: this improves overlapping/persistent-session initialization, not
first-ever compilation or reinitialization after all sessions have been dropped.
Precompiled metallibs and strong process-lifetime retention were not introduced.

Public APIs, supported format predicates, Auto promotion rules, quantization,
progression ordering, and strict device contracts remain unchanged. The tests
compare complete codestreams with the CPU encoder and independently decode them.

## Measurements

Apple M4 Pro, 12 CPU cores, 48 GiB memory; macOS 26.5.2 build 25F84;
Rust 1.96.0; Apple Metal compiler 32023.883. Release-bench uses optimization
level 3, fat LTO, and one codegen unit. Criterion requested 20 samples, a
two-second warm-up, and a five-second measurement window; slow cells extended
the collection window.

The encoder baseline was captured from the working tree before these changes.
It already included the earlier P28 resident transform-through-HT optimization.
Inputs use the existing P28 synthetic gray/RGB patterns, 64x64 blocks, three
decomposition levels, and guard bits 2. The batch-16 case performs sixteen
sequential public encodes with distinct inputs; it is not a simultaneous GPU
batch. Timing includes host pixels through completed host codestreams.

Encoder values are mean wall time with 95% intervals, in milliseconds.

| Workload | Before | After |
| --- | ---: | ---: |
| RGB 128x128, one image | 5.671 [5.634, 5.710] | 2.750 [2.743, 2.758] |
| Gray 512x512, one image | 21.609 [21.544, 21.673] | 7.345 [5.409, 9.911] |
| RGB 512x512, one image | 46.126 [45.938, 46.377] | 15.869 [12.303, 20.528] |
| RGB 640x480, one image | 54.066 [53.944, 54.194] | 11.638 [11.333, 11.905] |
| RGB 1024x1024, one image | 171.675 [171.520, 171.844] | 21.291 [21.121, 21.470] |
| RGB 512x512, sixteen sequential images | 739.495 [738.114, 741.029] | 173.999 [169.085, 178.142] |

The final pass was noisier at 512x512 than the initial candidate pass (initial
RGB512 mean 8.091 ms, final 15.869 ms). All final intervals remain below their
pre-change baselines, but individual changes such as input pooling are not
isolated by this comparison. The stable large-image example is approximately
8.1x faster; these numbers must not be extrapolated to other GPUs or corpora.

The session probe measured 311.823 ms for first setup, then paired means of
2.786 ms for fresh uncached kernel groups and 0.043 ms for shared groups over
five alternating pairs. First setup is context, not the comparison denominator.
The probe asserts real decode/encode pipeline identity and keeps an anchor
session alive. Driver caches are warm; no first-after-boot or last-owner-dropped
startup improvement is claimed.

Readback compares the exact current public methods with their former algorithms
in the same executable. The sources are completed immutable decoded surfaces;
decode and fixture construction are outside timing. Outputs are compared exactly
before timing. Values are Criterion slope estimates with 95% intervals, in µs.

| Readback workload | Former path | Direct shared path |
| --- | ---: | ---: |
| RGB 512x512, download into caller | 18.642 [18.617, 18.669] | 9.200 [9.195, 9.205] |
| RGB 1024x1024, download into caller | 76.206 [76.094, 76.329] | 37.032 [36.823, 37.270] |
| RGB 512x512, packed batch 16 | 1934.595 [1684.561, 2340.648] | 185.200 [173.044, 204.985] |
| RGB 1024x1024, packed batch 16 | 6110.873 [6061.606, 6177.353] | 901.056 [877.301, 936.772] |

These are readback-only gains, not whole-decode speedups. Mixed/private behavior
has correctness coverage but was not timed. Repeated readback uses warm memory,
so its throughput is not a DRAM bandwidth measurement. The packed 512 baseline
was particularly noisy; its broad interval should accompany any comparison.

All measurements are from one machine with synthetic inputs. Before/after encoder
runs use separate builds and processes; readback variants run consecutively in
one process. OS/driver caches, other system activity, and DVFS can affect results.
No hardware occupancy, spill, cache-miss, or per-stage GPU-counter claims are made.

## Reproduction

Capture the encoder baseline before applying the change, then run the comparison:

```sh
J2K_REQUIRE_METAL_BENCH=1 cargo bench --profile release-bench -p j2k-metal --bench transform_stages -- metal_lossy_resident --sample-size 20 --warm-up-time 2 --measurement-time 5 --save-baseline metal-architecture-before
J2K_REQUIRE_METAL_BENCH=1 cargo bench --profile release-bench -p j2k-metal --bench transform_stages -- metal_lossy_resident --sample-size 20 --warm-up-time 2 --measurement-time 5 --baseline metal-architecture-before
J2K_REQUIRE_METAL_BENCH=1 cargo bench --profile release-bench -p j2k-metal --bench readback -- --sample-size 20 --warm-up-time 2 --measurement-time 5
J2K_REQUIRE_METAL_RUNTIME=1 cargo test --profile release-bench -p j2k-metal --lib engine::runtime::resource_profile_tests::benchmark_cold_and_repeated_session_kernel_initialization -- --exact --ignored --nocapture
```

The startup probe records first-session setup separately, then alternates five
pairs of uncached and shared sessions. Both initialize decode and encode groups;
an anchor session remains live. Pointer assertions verify actual kernel reuse.
It does not equate first-session compilation with warmed repeated-session work.

## Verification

Focused regressions first reproduced seven private-pool misses after warm-up,
two encoder command buffers, repeated same-device pipeline construction, and
unnecessary shared readback staging/copies. The final cases verify pool reuse,
one completion boundary, byte-exact scalar parity, all five progression orders,
odd/single-axis inputs, unsupported metadata and rate-budget fallback, strided
subranges, mixed storage, preflight errors, allocation limits, and cache lifetime
and concurrency. A planner regression protects actual Tier-1 table bounds.

The following checks passed in the development working tree, which also contained
unrelated sampled-decode changes:

```sh
# 538 passed, 28 ignored across unit/integration/example targets.
J2K_REQUIRE_METAL_RUNTIME=1 cargo test --profile gpu-quick -p j2k-metal --lib --tests --examples --all-features -- --test-threads=1
# 21 explicitly selected ignored Metal diagnostics passed.
J2K_REQUIRE_METAL_RUNTIME=1 cargo test --profile gpu-quick -p j2k-metal --lib --all-features -- --ignored --skip metal_irreversible_idwt_gpu_capture --skip benchmark_cold_and_repeated_session_kernel_initialization --skip local_sampled_color_batch_characterization --test-threads=1
cargo clippy --profile gpu-quick -p j2k-metal --lib --all-features -- -D warnings
cargo clippy --profile gpu-quick -p j2k-metal --bins --tests --benches --examples --all-features -- -D warnings -A clippy::disallowed_methods -A clippy::disallowed_macros
cargo check -p j2k-metal --target x86_64-unknown-linux-gnu --all-features
cargo fmt -p j2k-metal -- --check
cargo xtask unsafe-audit
git diff --check -- crates/j2k-metal docs/unsafe-audit.md
```

The startup diagnostic passed separately using the release-bench command above.
The manual GPU capture, local sampled-color characterization, and four ignored
integration cases were not run. Linux verification is compilation of the
non-Metal path, not runtime testing. No other Apple GPU, sanitizer run, or remote
release CI was exercised. Existing unrelated working-tree edits were preserved.

Before opening the PR, only this optimization's 17 files were selected onto
`main` and exported to an isolated source directory for validation. The same
unit/integration/example command passed 536 tests with 27 ignored; 21 additional
ignored Metal diagnostics passed (the sampled-color skip is unnecessary in this
snapshot). Both Clippy commands, formatting, Linux compilation, unsafe inventory,
and the staged diff check also passed. The lower test count reflects exclusion of
unrelated sampled-decode tests. Benchmark values above remain measurements of the
development working tree; they were not remeasured on this isolated snapshot.
