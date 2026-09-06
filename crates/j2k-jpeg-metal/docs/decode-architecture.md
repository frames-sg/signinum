# JPEG Metal decode architecture and measurements

Measured on an Apple M4 Pro, macOS 26.5.2, Rust 1.96.0, using the workspace
`release-bench` profile, on September 5–6, 2026. These changes concern baseline
JPEG; JPEG 2000 implementations are unchanged.

## Execution and ownership

Supported 4:2:0 and 4:2:2 full texture batches now decode entropy and perform IDCT
in one grid spanning the compatible tiles. The grid writes three reusable private
component planes. A subsequent GPU pass performs chroma reconstruction, color
conversion, and writes the caller's existing private RGBA textures. Coefficients
are not materialized in a full-image intermediate. The direct 4:4:4 texture path
is retained: it has no subsampled chroma repair and its plane decoder uses a
different ABI.

The prior direct subsampled texture shader ran a separate encoder per tile,
retained substantial per-thread pixel state, and required additional horizontal,
vertical, and corner repair passes. Avoiding component-plane storage did not
compensate for those costs in the measured workloads. The component-plane route
requires 1.5 MiB of private intermediate storage for sixteen 256×256 4:2:0 tiles,
reused on subsequent calls.

Packet entropy is copied directly into reusable shared Metal storage. Only the
offset, length, and checkpoint metadata is materialized in host vectors. Retained
packet owners and metadata remain budgeted; shared Metal allocations retain the
existing device and repository size limits. Shared storage is appropriate for
CPU-populated data, while GPU-only planes and textures stay private, following
[Apple's storage-mode guidance](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus).

Each runtime has two independently locked scratch slots. Concurrent callers can
stage separate batches without overwriting each other's inputs or statuses;
further callers block instead of allocating additional slots. Each lease lasts
through command-buffer completion **and CPU status consumption**. An availability
gate wakes callers when either slot is returned, including when an owner
unwinds; poisoned state remains a typed error. The public
synchronous and deferred-submit contracts remain unchanged; this does not add an
asynchronous API. Reusing the same output still requires its existing access gate.
Two concurrent callers can retain two sets of scratch capacity.

Successful immutable pipeline registries are retained per Metal device registry
ID for the process lifetime. Independently created sessions share those pipelines
but own separate queues and mutable runtime state. Initialization is serialized
per device, failures are not published, and a poisoned cache or failed metadata
reservation falls back to an uncached load while preserving loader errors.

## Preparation and compatibility

Checkpoint traversal now uses the existing entropy-only `skip_block` operation
instead of constructing dequantized coefficient blocks it discards. Differential
tests preserve bit-reader snapshots, DC prediction, and exact errors.

Full entropy validation remains eager. Deferring packet construction or deriving
restart work only from marker offsets would move malformed-Huffman errors out of
the current constructor and is not part of this change. Small/one-shot `Auto`
routing and explicit Metal rejection behavior are unchanged.

## Controlled implementation comparison

The ignored `texture_scheduling_experiment` test compares the direct and
component-plane implementations in the same optimized executable, with identical
distinct JPEG tiles, reusable sessions/textures, CPU output parity checks for
every tile, five warm iterations, and fifteen measured iterations per variant.
The table shows median batch wall time using the existing 256-thread cap.

| 4:2:0 workload, 16 distinct tiles | Direct textures | Component planes |
|---|---:|---:|
| 256×256, no restart | 6.723 ms | 0.517 ms |
| 256×256, restart interval 2 | 12.127 ms | 0.755 ms |
| 512×512, no restart | 6.993 ms | 1.200 ms |
| 512×512, restart interval 4 | 26.683 ms | 1.644 ms |

The public-API Criterion benchmark also confirms the improvement. The original
run used ten samples, one second of warmup and two seconds of measurement. The
final run used ten samples, two seconds of warmup and five seconds of measurement:

| Warm 256×256 RGBA texture batch, 16 tiles | Original estimate | Final estimate (95% interval) |
|---|---:|---:|
| No restart markers | 6.635 ms | 0.677 ms (0.625–0.730 ms) |
| Restart interval 2 | 12.029 ms | 0.914 ms (0.907–0.918 ms) |

An intervening run was substantially noisier, including a slower CPU-only
restart control. Repeating that control without source changes measured
0.314 ms (0.312–0.317 ms), close to the original 0.311 ms; the apparent CPU
regression was not reproduced. These are throughput comparisons for warm
resident batches, not latency claims for all JPEG requests.

Additional comparisons covered batch sizes 1, 2, and 4, 16×16 and 64×64 inputs,
and 4:2:2 at 256×256 and 512×512. Component planes also won those comparisons,
although the small-workload measurements were noisier. Separate correctness
coverage includes distinct odd-sized edge tiles, restart streams, mixed tables,
reused output, and concurrent calls sharing a backend session.

Threadgroup widths of 32, 64, 128, and 256 were compared. Smaller groups modestly
helped the direct shader, but there was no consistent additional benefit once
using component planes. Production therefore retains the existing threadgroup
policy. This follows Apple's advice to measure work distribution rather than
assuming a particular group size is universally best:
[Scale compute workloads across Apple GPUs](https://developer.apple.com/videos/play/wwdc2022/10159/).

The earlier P19 experiment tested a different design: GPU entropy decoding into
full-image coefficient scratch followed by a separate IDCT pass. Its rejection
does not apply to this component-plane intermediate.

The existing fast-packet planning benchmark measured about 301 µs without
restarts and 312 µs with restart interval 2 after the entropy-only traversal
change. Neither comparison showed a statistically supported improvement; no
planning speedup is claimed. No isolated speedup is claimed for direct shared
staging or the two-slot pool either.

These measurements are local diagnostics, not a cross-device performance
guarantee. Variant order was fixed, the desktop was not isolated, and no GPU
occupancy or spill counters were captured. Warm measurements exclude packet and
pipeline initialization. The original unmodified warm baseline varied roughly
6.6–7.2 ms across separate runs.

## Reproduction and checks

```sh
# Same-input implementation and threadgroup comparison (also asserts parity).
J2K_REQUIRE_METAL_RUNTIME=1 cargo test --locked -p j2k-jpeg-metal --lib --profile release-bench texture_scheduling_experiment -- --ignored --test-threads=1 --nocapture

# Small batches and 4:2:2 controls.
J2K_REQUIRE_METAL_RUNTIME=1 J2K_JPEG_TEXTURE_SMALL_BATCHES=1 cargo test --locked -p j2k-jpeg-metal --lib --profile release-bench texture_scheduling_experiment -- --ignored --test-threads=1 --nocapture

# Public API benchmark: fresh CPU/Metal RGB and warm resident batch-16 textures.
J2K_REQUIRE_METAL_BENCH=1 cargo bench --locked -p j2k-jpeg-metal --bench compare --profile release-bench -- '^(decode_rgb/generated/fast420(_restart2)?_256x256/(cpu|metal)|wsi_tile_batch_rgba_textures/batch16/warm_session_reused_textures/generated/fast420(_restart2)?_256x256)$' --sample-size 10 --warm-up-time 1 --measurement-time 2 --noplot

# Final resident-output run with longer sampling.
J2K_REQUIRE_METAL_BENCH=1 cargo bench --locked -p j2k-jpeg-metal --bench compare --profile release-bench -- '^wsi_tile_batch_rgba_textures/batch16/warm_session_reused_textures/generated/fast420(_restart2)?_256x256$' --sample-size 10 --warm-up-time 2 --measurement-time 5 --noplot

# Runtime tests require real Metal instead of accepting an unavailable device.
J2K_REQUIRE_METAL_RUNTIME=1 cargo test --locked -p j2k-jpeg-metal -- --test-threads=1
cargo test -p j2k-jpeg --lib
cargo fmt --check -p j2k-jpeg-metal -p j2k-jpeg
cargo clippy --locked -p j2k-jpeg-metal --lib -- -D warnings
cargo clippy --locked -p j2k-jpeg-metal --tests --benches --bins -- -D warnings -A clippy::disallowed_methods -A clippy::disallowed_macros
```

The non-library Clippy allowances match the repository's existing Metal check
policy. They do not apply to production library checks.
