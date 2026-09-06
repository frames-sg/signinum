# Changelog

This changelog tracks the current release line. Historical phase notes
and stale roadmap entries have been removed from the public documentation set.

## [0.11.0] - 2026-09-06

- Batches baseline JPEG 4:2:0 and 4:2:2 Metal texture decode through reusable component planes, with bounded scratch leases held through status consumption.
- Keeps lossy HT Metal encode payloads resident through packetization, reuses allocation capacity, and shares immutable pipelines per device.
- Adds the separately published `j2k-mpsgraph-support` ownership boundary for codec-independent graph submission.
- Adds lossy HT quality-factor options, reusable HT block-encoding workspaces, prepared region decoding, and candidate-set accelerator integration.
- Corrects the Windows OpenHTJ2K reference runtime configuration used by release validation.

- Corrects TIFF `JPEGTables` assembly for abbreviated JPEG tiles, including
  SOI/EOI-bearing tiles with external DQT/DHT definitions, missing tile or
  length-delimited `JPEGTables` EOI markers, and TIFF-defined DRI/DAC handling.
- Adds progressive, general lossless, and DICOM lossless-SV1 JPEG passthrough
  syntax classification plus ordered APP2 ICC profile extraction, insertion,
  and fail-closed replacement helpers.
- Hardens `j2k-mpsgraph` completed-buffer handoff with same-device validation,
  adds automated F32 graph parity against a higher-precision CPU oracle, and
  makes direct-handoff benchmark paths prepared-input-equivalent, warmed,
  interleaved, and correctness-checked before any speed qualification. Its
  runnable example now constructs a caller-owned graph through the public
  `MpsGraphProgram::new` boundary. Breaking: removes the repository-specific
  identity, RGB8 reference-graph, and CPU-oracle convenience APIs from the
  experimental production surface; equivalent fixtures remain dev-only.

## [0.10.0] - 2026-08-25

- Adds the experimental `j2k-mpsgraph` crate for Apple Silicon macOS 11+.
  Completed resident batches and pipelined direct decode feed static rank-four
  MPSGraph programs without decoded-pixel readback/re-upload; blocking,
  nonblocking, reference-oracle, benchmark, package, and release validation
  paths are included. No zero-copy or speed claim is made.

- Breaking: narrows `j2k-cuda-runtime` to codec-neutral CUDA allocation,
  launch, completion, and diagnostics. CUDA JPEG 2000, JPEG, and transcode
  ownership now lives in the published implementation crates
  `j2k-cuda-j2k-engine`, `j2k-cuda-jpeg-engine`, and
  `j2k-cuda-transcode-engine`; callers of
  `j2k_cuda_runtime::transcode_kernels_built` must use
  `j2k_cuda_transcode_engine::transcode_kernels_built`.
- Breaking: generalizes
  `j2k_transcode_metal::resident_codestream_buffer_from_metal_encoded_j2k`
  from `&j2k_metal::MetalEncodedJ2k` to any `&impl DeviceCodestream`. Existing
  `MetalEncodedJ2k` calls continue to type-check, while custom device
  codestreams implement `j2k_core::DeviceCodestream`; this removes the
  transcode adapter's dependency on the full public Metal adapter.
- Moves stable codec contracts, prepared-plan geometry, capability rejection,
  host allocation phases, packing, and encode geometry into single typed
  owners. Native produces typed plans consumed by CPU, CUDA, and Metal; the
  adapters no longer downcast erased plans or duplicate resource policy.
- Decomposes the native color, Metal JPEG/JPEG 2000, CUDA, and transcode
  orchestration roots by pipeline ownership, with repository policies enforcing
  dependency direction, source boundaries, unsafe inventory, clone ceilings,
  benchmark registration, and generated routing evidence.
- Adds bounded HTJ2K lossy candidate generation with two consecutive HT sets,
  exact cleanup/SigProp/MagRef byte boundaries, per-pass distortion scoring,
  and dependency-aware slope allocation. CUDA and Metal can produce the exact
  candidate sets on device; CUDA entropy writing remains serial within each
  code block. Each selected HT set is emitted atomically in the layer of its
  final selected pass so external decoders retain an unambiguous cleanup
  boundary.
- Makes bounded classic JPEG 2000 and HTJ2K row decode through 24-bit component
  precision parse the tile graph once per row operation and reuse it across
  stripes while retaining the parsed metadata in the aggregate allocation
  baseline. Higher-precision exact-integer rows keep their compatibility path.
- Promotes measured staged CUDA JPEG encoding and adaptive CUDA JPEG checkpoint
  launch geometry. The staged encoder improves the representative 512 x 512
  batch-8 cells by about 95%, while the checkpoint policy uses one-thread
  launches below 128 checkpoints and 128-thread blocks at or above that
  threshold.
- Records and removes rejected Metal and CUDA candidates whose product
  intervals did not qualify, including terminal IDWT/store specialization,
  wide-axis and FDWT staging, RGB input fusion, packetization, terminal
  column/quantization fusion, and JPEG coefficient/IDCT defusion. No rejected
  selector, fallback, or benchmark-only production branch remains.
- Repairs ties-to-even irreversible color rounding in Metal and CUDA, the JPEG
  restart-marker bit-reader boundary, CUDA restart-coded capability routing,
  clean packaged CUDA-Oxide sibling resolution, and lane-specific changed-line
  coverage orchestration.

- Refactors the CPU JPEG AVX2 and NEON paths around unforgeable capability
  tokens and safe kernel entry points. SIMD dispatch, IDCT/color arithmetic,
  tests, and benchmark wrappers no longer require unsafe calls; the remaining
  raw loads/stores and exact-AVX2 bridge are isolated behind fixed-size safe
  interfaces. Ten production SIMD unsafe blocks remain, with no `unsafe fn`,
  under an AST-enforced 24-block cap.
- Uses `fearless_simd 0.7.0` for AArch64 NEON kernels while preserving the
  existing AVX2-only x86 acceleration envelope with a private exact-AVX2 token;
  `fearless_simd::Avx2` is intentionally not used because its 0.7 contract is
  the broader x86-64-v3 feature set.
- Adds deterministic CPU JPEG decode and expanded SIMD microbenchmarks covering
  grayscale, 4:4:4, 4:2:2, full/odd/cropped 4:2:0, row streaming, specialized
  IDCT, tail widths, and unaligned source rows.

## [0.9.0] - 2026-08-10

- Breaking: all expert Metal APIs in `j2k-metal-support`, `j2k-metal`,
  `j2k-jpeg-metal`, and `j2k-transcode-metal` now use objc2 ownership directly:
  owned objects are `Retained<ProtocolObject<dyn MTL...>>`, borrowed objects are
  `&ProtocolObject<dyn MTL...>`, and Metal value/descriptor types come from
  `objc2-metal`. Callers must replace `metal::Device`, `*Ref`, `MTLSize`,
  texture, queue, command-buffer, and buffer spellings with their objc2
  equivalents; there is no `metal-rs` compatibility feature or adapter.
- Replaces the workspace's deprecated `metal-rs 0.33` dependency with the
  pinned `objc2 0.6.4`, `objc2-foundation 0.3.2`, and `objc2-metal 0.3.2`
  stack. This removes the transitive `block 0.1.6` dependency, its local patch,
  and all associated lint, coverage, packaging, and release exceptions while
  preserving the existing Metal kernels and codec behavior.
- Centralizes Metal object creation, retained command submission, checked
  buffer access, shared-event creation, pipeline loading, and resident-image
  ownership in `j2k-metal-support`. Typed objc2 APIs now cover queue/resource
  identity, GPU timing, capture, textures, and event ordering; the remaining
  pointer and immediate-byte bindings have explicit safety proofs.
- Shader compilation failures retain their typed Metal diagnostics. The objc2
  binding does not expose warning-only `NSError` values returned alongside a
  successful library, so successful compilations no longer log those warnings.
- Removes `checked_texture_descriptor` and the unreachable raw-message-send
  `MetalSupportError` variants. Expert callers construct
  `MTLTextureDescriptor` directly and handle the remaining typed allocation
  and availability errors.

- Extends the shared ISO/IEC 15444-4:2024 / ITU-T T.803 v3 framework with
  unversioned HTJ2K Part 15 development evidence. The CPU IUT passes all 160
  selected Part 1 and Part 15 cases with zero skips on macOS arm64, Linux
  x86-64, and native Windows x86-64/MSVC, including DS1-HM Cclass-1h at MMAGB
  15, Cclass-1HFh at MMAGB 20, and Annex G JPH at MMAGB 15.
- Extends the CUDA and Metal adapter-IUT reports to the same selected Part 15
  points. Each combined Part 1/Part 15 headline is 0/160 device-native, 81/160
  hybrid, and 79/160 CPU-routed; production-owned counters disclose execution
  per stage. Exact-clean-SHA Part 15 release evidence remains pending.
- Fixes HTJ2K decoder defects in irreversible midpoint reconstruction, ROI,
  packet/header parsing, HT cleanup/refinement, progression, tile parts, and
  JPH validation without vector-specific exceptions or relaxed tolerances.
- Versions the Annex D/F encoder ICS matrix for classic and HT modes. CPU passes
  55/56 cases through the pinned T.804 OpenJPEG decoder; OpenJPEG 2.5.3 rejects
  HT+RGN, so the mandatory 56th capability case passes through the independently
  pinned OpenHTJ2K decoder as explicitly supplemental interoperability evidence.
  The CUDA and Metal matrices pass 35/35 through OpenJPEG. All encoder results
  are informative evidence, not formal decoder conformance.
- Adds HTJ2K/JPH routing workloads and promotes only cells with identical
  output, at least 10% median improvement, and non-overlapping Criterion 95%
  confidence intervals. The production Metal HTJ2K host-output route now uses
  Metal coefficient preparation and HT Tier-1 with CPU packetization for the
  verified RGB8 1,024 x 1,024 and Gray8/RGB8 2,048 x 2,048 cells; measured
  smaller cells stay on CPU. Explicit CUDA and Metal requests remain strict.

## [0.8.1] - 2026-08-06

- Adds release-scoped ISO/IEC 15444-4:2024 / ITU-T T.803 v3 decoder evidence.
  The CPU IUT is Profile-1 Cclass-1 compliant, Profile-1 Cclass-1HF compliant,
  and Annex G JP2 reader compliant across all 90 selected cases with zero skips
  on macOS arm64, Linux x86-64, and Windows x86-64.
- Publishes separate CUDA and Metal adapter-IUT results for the same selected
  codestream classes. Each headline is 0/90 device-native, 48/90 hybrid, and
  42/90 CPU-routed; reports disclose parsing, Tier-1, dequantization, IDWT,
  MCT, color/output, and transfer execution per case.
- Adds exact codestream-resolution decoding through
  `decode_native_components_at_reduction` and explicit Annex G normalization
  through `decode_srgb8`, including Gray/RGB/RGBA layouts, component mapping,
  palettes, CDEF ordering, subsampling, enumerated color spaces, and restricted
  ICC conversion.
- Fixes Part 1 decoder conformance defects in irreversible midpoint
  reconstruction, ROI handling, packet/header parsing, progression and
  tile-part handling, component transforms, and JP2 validation without
  vector-specific exceptions or relaxed tolerances.
- Adds deterministic Annex D/F encoder ICS matrices. The CPU matrix passes
  28/28 cases and the CUDA and Metal matrices pass 25/25 through the pinned
  T.804 OpenJPEG decoder; this is informative encoder evidence, not the formal
  decoder claim.
- Independently verifies `p0_13.j2k` before harness normalization: the
  production decoder and OpenJPEG match all 257 native components exactly.
- Promotes only benchmark-qualified fixed `Auto` hybrid cells after identical
  output, at least 10% median improvement, and non-overlapping Criterion 95%
  confidence intervals. Explicit CUDA and Metal requests remain strict.
- Makes tag publication verify all three CPU reports plus the CUDA and Metal
  reports for the exact release SHA, and rotates the public API compatibility
  baseline to the published `v0.8.0` release.

## [0.8.0] - 2026-07-29

- Breaking: decoding is strict by default in `j2k` and `j2k-native`. Callers
  that intentionally accept the documented JP2/JPH optional-metadata
  recoveries must select `DecodeSettings::lenient()` per job; raw codestream,
  entropy, bounds, overflow, allocation, and resource-safety checks remain
  strict in both modes.
- Breaking: lenient decoding now warns only when a permitted recovery was
  actually used. `J2kDecodeWarning::LenientDecodeMode` is renamed to
  `J2kDecodeWarning::LenientMetadataRecovery`. New
  `J2kView::parse_with_settings` and `J2kDecoder::new_with_settings`
  constructors keep this policy local to one input.
- Adds `CudaLosslessEncoder`, a reusable, exclusively borrowed lossless-encode
  façade with explicit `Auto`, CPU-only, and strict-CUDA routing. Its opaque
  result reports requested and actual execution plus any unavailable-device or
  incomplete-route fallback; execution errors are returned without a hidden
  full CPU retry, and the encoder can be reused after failure.
- Routes 25–38-bit `Auto` lossless encode requests to the supported CPU path
  while preserving `RequireDevice` failure semantics.
- Defines stable, experimental, implementation, and binary API tiers in the
  ordered release manifest, and makes the `0.7.5`→`0.8.0` semver transition
  fail closed with an exact source- and behavior-break ledger. A later `0.8.x`
  candidate must use a published `v0.8.0` baseline.
- Fixes the published `j2k-ml 0.7.5` `cuda` and `metal` clean-consumer
  failures. The adapters now use accelerator codec decode, explicit
  decoded-pixel readback, and ordinary Burn tensor upload through released
  dependency APIs.
- Adds `CudaUploadBurnDecoder` and `MetalUploadBurnDecoder` as explicit aliases
  for staged host-memory upload behavior while preserving `CudaBurnDecoder`,
  `MetalBurnDecoder`, and their inherent method paths unchanged.
- Removes the private CubeCL and wgpu patch sources from the release graph.
  This is not a direct-destination or zero-copy accelerator contract.

## [0.7.5] - 2026-07-22

- Known release defect: the published `j2k-ml 0.7.5` `cuda` and `metal`
  features depend on accelerator interop methods that are not present in the
  registry CubeCL and wgpu releases selected by clean consumers. The `cpu`
  feature is unaffected. Version `0.8.0` replaces those routes with explicitly
  staged accelerator decode, decoded-pixel readback, and ordinary Burn tensor
  upload using released public APIs.
- Compatibility exception: this release is an explicitly source-incompatible
  `0.7.x` patch against the published `0.7.3` API. The reviewed compatibility
  evidence and migration notes cover the complete contracted surface.
- Adds persistent, high-throughput owned batch decoding for JPEG 2000 and
  HTJ2K, including prepared-input reuse, homogeneous exact-integer output
  groups, CPU decode, and resident/external-destination CUDA and Metal paths.
- Publishes `j2k-ml`, the thin Burn 0.21 adapter for ordinary rank-4 `U8`,
  `U16`, and `I16` tensors. The codec remains responsible for parsing,
  grouping, and decoding; the adapter owns only tensor allocation and guarded
  CPU/CUDA/Metal interop.
- Fixes Metal stacked multi-tile batch output and removes redundant same-queue
  synchronization while preserving strict completion and status validation.
- Removes the pass-through `j2k_core::DecoderContext<C>` type and its
  `j2k::DecoderContext` re-export. Construct and pass the concrete context
  directly: `J2kContext::new()` (or `Default`) for `j2k`, and
  `j2k_jpeg::DecoderContext::new()` (or `Default`) for JPEG. Batch trait
  implementations now receive `&mut Self::Context`; call `CodecContext::clear`
  and `CodecContext::cache_stats` on that concrete value.
- Removes `try_collect_ordered_batch_results`. Migrate to
  `try_collect_ordered_batch_results_with_limits`, passing the previous
  retained-byte and maximum-retained-byte pair for both its aggregate and
  collection limit pairs.
- Removes the five forwarding accessors on
  `J2kResidentHtj2kTileEncodeJob`. Read the validated input directly through
  `job.input.width()`, `height()`, `num_components()`, `bit_depth()`, and
  `signed()`.
- Internal: removes pass-through decode, encode, allocation, crop, recode,
  coverage-normalization, fixture, cache-key, progression, and GPU-routing
  layers while retaining codec-specific scratch-pool types.

## [0.7.3] - 2026-07-15

- Fixes invalid TLM `Stlm` descriptors in CPU and Metal codestream writers so
  HTJ2K RPCL output is accepted by conforming external decoders.

## [0.7.2] - 2026-07-15

- Fixes strict CUDA Oxide builds from crates.io by resolving the packaged
  `j2k-codec-math` source through Cargo build metadata instead of assuming the
  unpublished workspace directory layout.

## [0.7.1] - 2026-07-14

- Adds CUDA classic JPEG 2000 Tier-1 code-block decoding with shared host/device
  MQ and context tables, batched resident submission, bounded host accounting,
  and explicit CUDA ABI and unsafe-boundary inventories.
- Adds an opaque, zero-copy Metal resident-image contract with checked layout
  and device identity, submission-owned input and output lifetimes, safe
  decoder-to-encoder handoff, and explicit unsafe raw-buffer interop.
- Improves Metal decode residency with dynamic pool ceilings and enforced
  inflight limits, fixes irreversible inverse-MCT addends, fuses native inverse
  MCT output handling, and uses a staged irreversible 9/7 IDWT implementation.
- Adds hardened CUDA and Metal profiling/capture controls and documents their
  benchmark-only environment variables.
- Adds the unpublished experimental `j2k-ml` workspace crate with CPU and Metal
  validation paths and an optional CUDA-Oxide kernel route, including checked
  adoption and stream-order completion for CubeCL memory-pool allocations.

## [0.7.0] - 2026-07-12

- Moves the test-only `corpus_validation`, `dct53_1d`, and `dct53_multilevel`
  modules out of the shipped `j2k-transcode` library into test support files.
- Internal: consolidates duplicated test fixtures/helpers into
  `j2k-test-support` and xtask-shared modules, macro-generates the CUDA
  GPU-ABI byte-view wrappers, and collapses repeated classic Tier-1
  profiling-label chains in `j2k-metal`.
- Internal: replaces the duplicated generic/component-row and interleaved-RGB
  JPEG sequential scan drivers with one typed, monomorphized geometry,
  restart-seek, rolling-stripe, and finish-order owner. Output emitters remain
  explicit and the hot MCU-row kernel remains a focused module.
- Internal: makes release-integrity Cargo-metadata parsing fail closed on
  malformed, duplicate, or unmatched workspace package records and malformed
  unpublished-dependency fields instead of silently dropping them.
- Internal: removes the last blanket string-to-error conversions. CUDA HTJ2K
  packetization now marks every invalid plan explicitly for CPU fallback while
  keeping host-allocation failure a hard typed stage error.
- Replaces static-string failures on all 14 fallible
  `J2kEncodeStageAccelerator` hooks with the non-exhaustive,
  `no_std`-compatible `J2kEncodeStageError` taxonomy. Native and facade encode
  failures now retain the failed operation, typed stage category, and concrete
  CUDA/Metal runtime source; `Ok(None)`/`Ok(false)` is reserved for an ordinary
  capability decline.
- Makes `TranscodeStageError` source-preserving and move-only. Accepted CUDA
  and Metal execution failures now retain stable backend/operation context and
  their concrete adapter error through `Error::source`; device cap and device
  allocation failures remain distinct from host resource failures and ordinary
  unsupported/unavailable declines. Validation metrics now return the typed
  `MetricsError`, including cap and allocator categories, and
  `JpegToHtj2kError::Metrics` exposes that source directly.
- Gives public HTJ2K segment-length math and shared 9/7 code-block option
  validation non-exhaustive typed errors with allocation-free `reason()` text.
  The cleanup-distribution helper now returns the existing typed native
  `EncodeError`, preserving invalid-input and arithmetic categories instead of
  collapsing them to static strings.
- Makes the four doc-hidden public scalar classic/HT code-block encode and
  classic token-pack helpers return `EncodeResult`. Invalid input, checked-cap,
  allocator, arithmetic, and invariant failures now reach adapter callers as
  their original native category.
- JPEG Metal batch failures now preserve their exact decode, encode, buffer,
  routing, runtime, and kernel error variants when one group failure is
  reported to multiple output slots instead of rendering most failures into a
  generic kernel-error string. `JpegEncodeError` and `j2k_jpeg_metal::Error`
  now implement `Clone` to support that typed replication.
- J2K Metal resident-batch preparation now propagates its original typed
  backend error directly instead of rendering it into a second generic
  `MetalKernel` message at the submit coordinator.
- J2K and JPEG Metal runtime, command-completion, buffer-access, and readback
  failures now retain `MetalSupportError` as their source while preserving
  operation-specific diagnostics and existing unavailable/unsupported routing.
  Prepared-plan cache allocation and invariant failures are separate typed J2K
  Metal errors; readback allocation failures no longer invent a saturated byte
  count from an element count.
- Makes `j2k-metal::Surface::as_bytes` and
  `j2k-jpeg-metal::Surface::as_bytes` fallible. Host-backed views remain
  borrowed, while Metal synchronization, poisoned access-gate, readback, and
  range failures now return typed adapter errors instead of panicking.
- JPEG Metal now classifies sampling once and builds, shares, and caches only
  the matching 4:2:0, 4:2:2, or 4:4:4 fast-packet family. Ordinary capability
  mismatch remains an optional fallback; malformed input, allocation caps,
  allocator failures, and internal invariants retain `FastPacketError` and its
  nested JPEG source instead of being erased by `.ok()` probes.
- `j2k-jpeg::adapter` now provides one doc-hidden backend-neutral cached-plan
  owner and byte-bounded flat LRU for JPEG accelerators. Complete input and the
  single matching 4:2:0, 4:2:2, or 4:4:4 packet are shared without deep hit
  clones, charged by actual nested `Vec` capacities, and admitted under
  fallible 8-entry/64-MiB defaults with typed errors and high-water diagnostics.
  One `resolve` boundary owns the full hit, fallible copy, inspect-once build,
  optional admission, and current-request return sequence for both backends.
- `j2k-metal-support` now owns every raw Metal buffer, texture, shader,
  pipeline, queue, command-buffer, and encoder constructor. Nil Objective-C
  results are rejected before forming foreign handles, non-owning command
  objects are retained and returned as owned Rust handles across autorelease
  pools, and the borrow-erasing no-copy buffer API is removed in favor of
  Metal-owned uploads. All J2K/JPEG/transcode Metal production, test, example,
  and benchmark callers use the checked boundary.
- J2K and JPEG Metal batch request, entropy/checkpoint, packet-plan, grouped
  result, resident-encode, surface, and texture metadata now reserves through
  checked aggregate 512 MiB budgets and reports allocator/cap failures as
  typed `BatchInfrastructureError` sources instead of aborting or presenting
  them as tile, encode, buffer, or Metal-kernel failures. JPEG Metal adds the
  typed `Error::BatchInfrastructure` variant; doc-hidden J2K benchmark
  grouping now returns `Result` so allocation failure is observable. Resident
  J2K code-block and nested packet metadata now move between plan, prepare,
  submit, and compute stages instead of deep-cloning their vectors.
- Tile-codec malformed and operational decoder I/O failures now retain their
  original `std::io::Error` sources, kinds, and operation context instead of
  reducing backend errors to display strings.
- Hardens safe GPU ABI byte views so every implementation must prove a fully
  initialized, padding-free representation at compile time. Five CUDA records
  and the JPEG Metal entropy checkpoint now use explicit initialized tail
  fields while preserving their established device sizes and offsets.
- Replaces approximate CUDA test-count floors and duplicated workflow logic
  with repository-owned CPU, CUDA, Metal, coverage, package, and exact-SHA
  release gates. Publication preflight rejects the wrong origin or tag, an
  existing GitHub Release, indeterminate crates.io state, and an invalid
  dependency-ordered retry before the first registry mutation.
- Splits the largest native single-tile, Metal resident-codestream,
  direct-stacked batch, and adoption-report orchestrators into focused modules
  while preserving byte output, dispatch order, resource retention, and
  regression coverage.
- Splits facade decode orchestration from 8-bit component/layout conversion and
  16-bit native channel conversion. Warning policy and backend dispatch remain
  in the small root; explicit child modules retain RGB/gray/alpha behavior and
  are protected by structural ratchets.
- Splits the oversized CUDA runtime HTJ2K encode/decode hosts and native JP2
  container root into focused API, planning, launch, completion, ABI/resource,
  metadata, parsing, and validation modules while preserving their public
  paths, GPU layouts, diagnostics, and allocation contracts.
- Makes the doc-hidden public native J2K direct-plan, owned-subband, and owned
  code-block graph move-only and exposes fallible retained-capacity accounting.
  J2K Metal decoder and session caches now share native and prepared plans with
  `Arc`, so cache hits and single-plan color execution no longer deep-clone
  entropy payloads, job vectors, or prepared Metal owners.
- Replaces J2K Metal's digest-bucket prepared-plan maps with a randomized,
  full-key-validating flat LRU. Each cache has explicit 64 MiB host and 256 MiB
  device ceilings covering allocator-returned key/entry capacities, nested
  native and prepared host owners, and reported Metal-buffer lengths. Metadata
  growth is fallible, replacement reuses the owned key, deterministic eviction
  happens before commit, and disabled or individually oversized admission is a
  non-error; allocation, lock, and invariant failures retain their typed source
  contracts.
- Makes the public AVX2 benchmark IDCT wrapper perform runtime feature
  detection and fall back to the scalar implementation on CPUs without AVX2;
  calling the safe wrapper no longer relies on an undocumented caller-side
  feature check.
- Reworks the doc-hidden native precomputed 9/7 encoders to borrow DWT
  coefficients directly, move owned preencoded payloads, and account
  prequantized/preencoded preparation, Tier-1, packetization, marker/tile-part
  output, and aggregate batch codestream ownership under one checked cap. The
  consuming `encode_precomputed_htj2k_97_batch_owned_with_accelerator` adapter
  lets JPEG-to-HTJ2K release source coefficient images before shared Tier-1 and
  final output growth.
- Reworks native J2K/HTJ2K encode planning, transform/Tier-1 ownership,
  packet metadata, tile-part assembly, and post-encode HT validation around
  fallible phase-wide allocation accounting. Standard, typed high-bit,
  multi-tile, precomputed, preencoded, and batch routes reconcile actual
  allocator capacities and preserve typed resource and accelerator failures.
- Makes native J2K tile and tile-part metadata use one transactional
  actual-capacity ledger for inherited components, marker overrides, packed
  headers, packet lengths, and retained tile parts. Temporary PPT/PLT owners
  roll back on parse failure, the final owner graph is checked exactly, and
  multi-tile PPM decoding now advances packed headers per packet instead of
  incorrectly assigning one header per tile part.
- Replaces the per-decode deep clone of native tile-part packet readers and
  length metadata with an allocation-free borrowing cursor. Parsed tile state
  remains immutable across repeated normal decode, direct-plan construction,
  and coefficient recode; large native ROI, SIMD/component, marker, coding-
  style, quantization, encode-parameter, and component-ROI owners are move-only.
- Makes lightweight native codestream inspection reserve its public SIZ
  component metadata fallibly. The non-exhaustive
  `J2kCodestreamHeaderError` adds `HostAllocationFailed { bytes }` instead of
  allowing a valid maximum-component header to abort on allocation failure.
- Makes reusable J2K row-decode scratch reconcile the allocator-reported
  packed-byte capacity before allocating the simultaneously live u16 row, so
  allocator overcapacity cannot cause a second allocation beyond the shared
  cap. Stale scratch owners are released transactionally and cap, arithmetic,
  and allocator failures remain typed before any row-sink callback.
- Keeps the cached native `Image` metadata inside the facade component-handoff
  peak for full and region, borrowed and owned decodes. Plane payloads and ICC
  profiles still move without copying; Metal plane staging reconstructs only
  heap-free Gray/RGB color variants and rejects ICC/unsupported variants before
  ownership transfer.
- Makes unused public J2K decoded-output, JP2 metadata, recode, coefficient,
  transform, encoded-block, and precomputed/prequantized/preencoded owner
  graphs move-only. These owners can retain image-sized vectors near the host
  cap; callers that need shared ownership should move them into `Arc` instead
  of relying on an infallible deep clone.
- Counts generated codestream capacity throughout facade round-trip validation,
  preserves structured native decode resource errors in the validation-phase
  `NativeValidation` category through the facade-owned opaque
  `NativeBackendError`, and compares sampled or high-bit components without
  allocating a second reference-grid image. The wrapper keeps the concrete
  native cause as the next `Error::source` without exposing `j2k-native` in the
  facade's public signatures. Lossy byte-target and PSNR searches keep only
  scalar candidate state while probing, revalidate the final encode, and report
  accelerator dispatch from that returned attempt only.
- Enforces `JpegTilePrepareOptions::duplicate_table_policy` across every DQT
  and DHT definition in multi-table `JPEGTables` markers. `AllowIdentical`
  coalesces byte-identical definitions, the default `RejectConflicting`
  preserves them for source-byte parity, both reject conflicts, and malformed
  later definitions now fail before abbreviated-tile assembly. The primary DQT
  parser also validates the precision selector before deriving its payload
  length, so invalid precision cannot be misreported as truncation.
- Reconciles JPEG warning-vector ownership under the same actual-capacity
  ledger used by decode planning. Parsed warnings, scan warnings, and the
  merged public result now constrain each subsequent reservation, and batch
  planning imports the same maximum-result formula. `PreparedJpeg::Owned` is
  move-only so TIFF assembly cannot be duplicated through an infallible clone.
- Makes the remaining large public JPEG result and backend-packet owners
  move-only. `EncodedJpeg`, `JpegDctImage`, `JpegDctComponent`, `RestartIndex`,
  `DeviceDecodePlan`, and the four doc-hidden fast-packet types can each retain
  caller-derived vectors near the 512 MiB host cap; supported backend sharing
  now uses `Arc` instead of duplicating those payloads. CUDA host download,
  diagnostic, direct-decode, and packetization-plan owners are move-only for
  the same reason.
- Gives public coefficient-domain baseline JPEG re-emission a typed,
  non-exhaustive `JpegDctImageError` contract. Caller-supplied dimensions,
  component order, every sampling factor, the ten-block MCU limit, checked
  block grids/counts, baseline quantization values, and the shared-chroma-table
  limitation are rejected before capacity planning or entropy encoding.
  Baseline DC-difference and AC magnitude-category limits are also enforced,
  preventing oversized caller coefficients from aliasing entropy symbols,
  being reported as internal encoding failures, or producing invalid streams.

### Breaking API Changes

Version `0.7.0` intentionally contracts the published pre-1.0 `0.6.2` API. It
does not claim source compatibility with `0.6.x`; the frozen-candidate reviewed
API diff must enumerate every removed or changed item, while the migration notes
below identify supported replacements and changes with no compatibility shim.

- Changes every fallible `J2kEncodeStageAccelerator` method to return
  `J2kEncodeStageResult<T>`. `j2k_native::EncodeError::Accelerator` now stores
  `source: J2kEncodeStageError` instead of `detail: &'static str`, and
  `EncodeError`, `ResidentHtj2kEncodeError`, and facade `J2kError` are no
  longer `Clone`/`Copy`. Backend implementers must return a typed hard failure
  after accepting work and use `Ok(None)` or `Ok(false)` only to decline it.
- Changes `j2k_native::packet_math::ht_segment_lengths` to return
  `HtSegmentLengthError`, changes
  `j2k_native::collect_ht_cleanup_encode_distribution` to return
  `EncodeResult<_>`, and changes
  `j2k_transcode::validate_htj2k97_codeblock_options` to return
  `Htj2k97CodeBlockOptionsError`. Callers should match typed variants for
  policy and use `reason()` only for presentation.
- Changes `j2k_native::{encode_j2k_code_block_scalar_with_style,
  pack_j2k_code_block_scalar_from_tier1_tokens,
  encode_ht_code_block_scalar, encode_ht_code_block_scalar_with_passes}` from
  static-string results to `EncodeResult<_>`. Adapter crossings must preserve
  the typed `EncodeError`; no string compatibility conversion is provided.
- Changes `TranscodeStageError::Backend(String)` to the source-preserving
  `Backend { backend, operation, source }` form, adds
  `DeviceMemoryCapExceeded` and `DeviceAllocationFailed`, and removes
  `Clone`/equality implementations from the error. Changes validation metric
  construction to return `MetricsError` and stores that concrete type in
  `JpegToHtj2kError::Metrics` instead of rendered text.

- Marks `j2k_tilecodec::TileCodecError` non-exhaustive, removes the unused
  `Backend(String)` variant, and adds `Io { context, source }`. The existing
  `Malformed` category now exposes `source: std::io::Error` instead of a
  rendered `message: String`; downstream matches must include a wildcard and
  can inspect the typed source and its `ErrorKind` directly.
- Adds `j2k_metal::Error::{MetalSupport, PreparedPlanCacheAllocation,
  PreparedPlanCacheInvariant}` and `j2k_jpeg_metal::Error::MetalSupport` so
  shared Metal and prepared-cache sources remain inspectable. These public
  pre-1.0 enum additions require exhaustive downstream matches to handle the
  new variants; `j2k_jpeg_metal::Error` remains `Clone`.
- Adds `JpegEncodeError::InvalidDctImage { reason }` and the public
  non-exhaustive `j2k_jpeg::transcode::JpegDctImageError` reason taxonomy.
  Exhaustive `JpegEncodeError` matches must handle the new variant. Removes
  the unused `JpegEncodeError::Internal(String)` variant after a workspace-wide
  owner audit found no production constructor; impossible encoder states use
  allocation-free `InternalInvariant { reason }`, while invalid coefficient
  input uses `InvalidDctImage { reason }`.
- Removes `Clone` from `j2k_jpeg::PreparedJpeg`; owned normalized TIFF/WSI JPEG
  payloads can approach the codec allocation cap and must not expose an
  infallible full-payload duplication contract. Use `PreparedJpeg::try_clone`
  when a duplicate is required; borrowed values remain allocation-free and
  owned values preserve typed cap versus allocator failures.
- Removes `Clone` from `j2k_jpeg::{EncodedJpeg, RestartIndex}`,
  `j2k_jpeg::transcode::{JpegDctImage, JpegDctComponent}`, and the doc-hidden
  `DeviceDecodePlan` and
  `JpegFast{420,422,444}PacketV1`/`JpegGrayPacketV1` adapter owners. Removes
  deep `Clone` from CUDA host-band, diagnostic, direct-decode, and HTJ2K
  packetization owner graphs. No production caller duplicated these near-cap
  owners; downstream code that needs shared access should move them into `Arc`
  and clone the `Arc`.
- Removes `Clone` from the large J2K owner families exposed by `j2k`,
  `j2k-native`, `j2k-types`, and `j2k-transcode`. This includes facade decoded
  component/color outputs, support/file/palette/COLR metadata, native
  `ColorSpace` and JP2 metadata/container outputs, `ReencodedHtj2k`,
  `Reversible53CoefficientImage`, encoded classic/HT code blocks, packetization
  metadata, forward DWT outputs, precomputed/prequantized/preencoded 5/3 and
  9/7 graphs (including compact and batch forms),
  `ReversibleDwt53FirstLevel`, and `Dwt{53,97}TwoDimensional`. No production
  caller required payload duplication; downstream sharing should use `Arc` or
  an application-owned fallible copy policy.
- Removes the doc-hidden `j2k_native::deinterleave_reference` compatibility
  wrapper, which panicked on invalid caller geometry. Backend parity and
  diagnostic callers must use the typed `try_deinterleave_reference` entry
  point.

- Removes pre-1.0 public surfaces that are no longer part of the supported
  contract: `j2k-core` `DecodeRequest` (with the four `*_request` trait default
  methods), `BackendFailureKind`, and `WarningKind`; `j2k-profile`
  `MetricUnit` (with the `unit` parameter of
  `ProfileField::metric`/`metric_with_summary`), `emit_gpu_route_profile`,
  `flush_profile_summary_to`, `ProfileSummary::flush_to`, and
  `ProfileField::raw`; and `j2k-metal-support` `MetalDeviceSession` plus the
  borrowed `buffer_contents_slice`/`buffer_contents_slice_mut` wrappers. Metal
  callers must use the checked owned-value read, read-vector, write, or fill
  operations and uphold their documented unsafe synchronization contract.
- Makes `j2k-profile` owned diagnostics explicitly bounded and fallible.
  `ProfileField` and `SummaryLabel` constructors, `ProfileSummary`
  construction/record/format/take operations, and public profile parsing and
  formatting now return `ProfileResult` with the non-exhaustive
  `ProfileError`; callers that customize limits can use `ProfileLimits`.
  `parse_profile_line` now returns `Result<Option<_>, _>` and malformed field
  tokens are rejected instead of silently discarded. `ProfileSummary`,
  `ParsedProfileFields`, `SummaryLabel`, `ProfileField`, and
  `TranscodeBatchProfileRow` are move-only; `BatchTranscodeReport::profile_row`
  is likewise fallible. Optional emission reports failures as explicit
  `j2k_profile_error` diagnostics without changing codec success results.
  Removes `duration_us_string`; duration values now use the same typed,
  bounded `ProfileField` construction as every other numeric metric. JPEG CPU,
  Metal, and CUDA profile callers no longer preformat owned field strings and
  report field-construction failures through `emit_profile_error`.
  The remaining removed rustdoc-hidden helpers migrate as follows:
  `GPU_ROUTE_PROFILE_ENV` is no longer exported; use the documented
  `J2K_GPU_ROUTE_PROFILE` key with `profile_stage_mode_from_env` or
  `StageModeCache::mode_from_env`. Replace `env_flag_from_value` with a
  `ProfileStageMode::Rows` match on `profile_stage_mode_from_value`,
  `gpu_route_profile_stage_mode_from_value` with
  `profile_stage_mode_from_value`, `gpu_route_profile_mode_enabled` with an
  explicit comparison to `ProfileStageMode::Disabled`, and
  `gpu_route_profile_stage_mode` with the same environment-key APIs. Replace
  `gpu_route_summary_labels` and `gpu_route_profile_summary` with caller-owned,
  fallible `SummaryLabel` values and `ProfileSummary::counts_only`; there is no
  specialized public GPU-route summary constructor. `format_profile_row` has
  no generic string-row replacement: use `emit_profile_row_now` or typed
  `ProfileField` emission, and use `format_profile_key_value_fields` only when
  serializing fields without the row prefix. Replace
  `emit_profile_row_u128_now` with fallible `format_profile_row_u128` followed
  by `emit_profile_line`. Replace `record_timing_summary_str` with
  `emit_profile_row_with_timing_summary` for thread-local summaries, or filter
  the caller-owned fields before fallible `ProfileSummary::record_str`.
- Removes the redundant `cuda-oxide-*` passthrough features from `j2k-cuda`
  and `j2k-transcode-cuda`; enable `cuda-runtime` instead, which already
  activates the underlying kernel families.
- Marks GPU diagnostic profile structs `#[non_exhaustive]`; downstream literal
  construction must switch to the provided constructors.
- Changes both `j2k-metal::Surface::as_bytes` and
  `j2k-jpeg-metal::Surface::as_bytes` from borrowed `&[u8]` to
  `Result<Cow<'_, [u8]>, Error>`. Host-backed surfaces can still return a
  borrowed view, while GPU-backed surfaces may perform an owned readback;
  callers must now handle synchronization, access-gate, and range failures.
- Changes `j2k-metal::MetalEncodedJ2k::codestream_bytes` from
  `Result<&[u8], _>` to `Result<Vec<u8>, _>`, makes its resource fields private
  behind metadata getters, and makes raw construction and consuming buffer
  handoff unsafe. Callers that only need encoded bytes should use the owned
  readback or `to_encoded_j2k`.
- Makes borrowed Metal `Buffer`/`Texture` accessors and external input-tile
  constructors unsafe when they can bypass safe synchronization. This includes
  raw surface/batch/texture access, `MetalLosslessEncodeTile::from_buffer`, and
  `JpegBaselineMetalEncodeTile::new`. Callers must keep ranges initialized and
  live, use the matching device, and exclude overlapping CPU/GPU access through
  actual command completion. Reusable JPEG outputs now serialize safe access
  across clones, subsets, derived surfaces, and viewport-cache reuse.
- Collapses the 26 `j2k-metal` lossless encode/submit permutations
  (`{encode,submit}_lossless_from_{padded_,}metal_buffer{,s}` x
  to-metal/with-report/with-config/to-metal-batch) into two supported
  request-based entry points: `submit_lossless_batch` (host codestream bytes),
  `submit_lossless_batch_to_metal` (Metal-backed codestreams with batch
  stats), plus the doc-hidden diagnostic helper
  `encode_lossless_batch_with_report` (host-byte timing reports). All take the
  new `MetalLosslessEncodeBatchRequest` (`tiles` + now-public
  `MetalEncodeInputStaging` + `MetalLosslessEncodeConfig`). The
  single-tile submit wrapper `SubmittedJ2kLosslessMetalEncode` is removed
  (submit a one-tile batch and take the first result). Single-tile
  `_with_report` callers now route through the batch report entry, which
  may use the resident batch path where the removed wrapper always used
  the per-tile staged pipeline; the report's stage-residency fields can
  differ accordingly.
- Collapses the 24 `j2k-jpeg-metal` `Codec::decode_rgb8_*_with_session`
  batch permutations (full/scaled/region-scaled x bytes/decoders x
  reusable/resizable buffer/textures) into two request-based entry points:
  `decode_rgb8_batch_into_buffer_with_session` and
  `decode_rgb8_batch_into_textures_with_session`, taking the new
  `Rgb8MetalBatchRequest` (`Rgb8MetalBatchSource` + `Rgb8MetalBatchOp`) and
  `MetalBufferBatchTarget`/`MetalTextureBatchTarget` enums. The three hottest
  permutations remain as convenience wrappers with unchanged signatures:
  `decode_rgb8_decoder_batch_into_resizable_metal_textures_with_session` and
  the two region-scaled resizable forms. Unsupported-batch error reasons no
  longer distinguish byte vs decoder sources.
- Introduces the `j2k-types` contract crate: the 49 encode-stage
  job/output/report types shared by `j2k` and `j2k-native`
  are defined once there (both crates re-export them at their existing
  paths), and the `j2k` adapter's `*_from_native`/`*_to_native`
  encode-stage converter functions are removed since both sides now use the
  same types.
- Widens transcode accelerator errors: `DctToWaveletStageAccelerator` methods
  now return `TranscodeStageError`
  (`Unsupported`/`Backend`/`MemoryCapExceeded`/`HostAllocationFailed`/
  `DeviceUnavailable`) instead of `&'static str`,
  `JpegToHtj2kError::Accelerator` carries the new type, and the Metal/CUDA
  transcode error `as_static_str` funnels are removed so backend failures keep
  their diagnostic detail. The blanket `From<&'static str>` conversion for
  `TranscodeStageError` is removed; downstream accelerators must choose an
  explicit error category. `JpegToHtj2kError` adds `InternalInvariant` for
  missing, duplicate, or out-of-range batch worker results, so exhaustive
  matches must handle the new variant.
- Changes `j2k-transcode-cuda::CudaTranscodeError` from `Copy` to an owned,
  cloneable error and adds typed host-cap, host-allocation, and runtime variants.
  `CudaRuntimeFailure` exposes the failed operation, complete rendered CUDA
  diagnostic, and unavailability classification; Auto fallback remains limited
  to unavailable or unsupported work.
- Changes the public `j2k-transcode` 5/3 and 9/7 DCT transform functions,
  including the `dev-support` scratch adapters, to return the new
  non-exhaustive `DctTransformError` instead of using `DctGridError` for both
  grid validation and execution failures. `DctGridError` remains the typed
  invalid-grid payload; transform callers must also handle invalid sample
  planes, aggregate memory-cap rejection, and host-allocation failure.
  `JpegToHtj2kError::Grid(String)` and `Grid97(String)` are replaced by typed
  `Dct53(DctTransformError)` and `Dct97(DctTransformError)` variants, with
  cap/allocation failures lifted to the existing top-level typed variants.
- Adds `j2k-jpeg::JpegError::HostAllocationFailed { bytes }`; callers that
  exhaustively match the pre-1.0 decoder error must handle bounded progressive
  coefficient and extraction allocation failure explicitly.
- Changes `j2k::TileBatchError` to the shared
  `j2k_core::BatchDecodeError<J2kError>`. Batch failures now distinguish a
  per-tile `Tile { index, source }` error from a typed
  `Infrastructure(BatchInfrastructureError)` scheduling, allocation, channel,
  or worker failure. Exhaustive callers must handle both branches; use
  `tile_error()` and `infrastructure_error()` when variant-independent access
  is preferable.
- Removes the unused doc-hidden
  `j2k_core::collect_indexed_batch_results` helper, whose signature could only
  report tile errors and therefore panicked on malformed worker indices or
  missing results while allocating infallibly. Internal adapter authors must
  use `try_collect_indexed_batch_results` or the ordered-slot collectors and
  handle `BatchDecodeError::Infrastructure`.
- Adds `j2k_metal::Error::BatchInfrastructure`, preserving non-resource CPU
  batch scheduling and result-collection failures as a typed source instead of
  flattening them into a Metal-kernel string. Exhaustive matches on the
  pre-1.0 error enum must handle the new variant.
- Adds `j2k_metal::Error::MetalStateInvariant` for contradictions in checked
  Metal ownership and accounting ledgers. Exhaustive matches on the pre-1.0
  error enum must handle the new variant.
- Adds `AllocationTooLarge` and `HostAllocationFailed` variants to
  `j2k_native::DecodeError`; exhaustive pre-1.0 matches must handle native
  codec-cap rejection separately from allocator failure. The crate also adds
  non-exhaustive `j2k_native::EncodeError` and `EncodeResult<T>` for native
  resource diagnostics and changes the public native pixel, ROI,
  component-plane, precomputed, preencoded, compact, and batch encode entry
  points from `Result<_, &'static str>` to `EncodeResult<_>`. The `j2k` facade
  adds `J2kError::NativeEncode { context, source }` and
  `J2kError::NativeValidation { context, source }`; their `source` fields use
  the opaque facade-owned `NativeBackendError`, retain the concrete native
  error as the next source link, and keep generated-output failures distinct
  from truncated caller input or an unsupported caller request. The Metal and
  CUDA adapter error enums use equivalent adapter-owned opaque wrappers, so no
  public facade or adapter signature names `j2k-native`. Meanwhile,
  `JpegToHtj2kError::Encode` now carries `Htj2kEncodeError`, with typed
  `Htj2kEncodeErrorKind` classification and the concrete native encode error as
  its next source; callers can inspect allocation, accelerator, validation,
  unsupported-request, and invariant categories without parsing display
  strings.
- Adds `J2kError::NativeResidentEncode { context, source }` as a typed fallback
  for future variants of the non-exhaustive native resident-encode boundary.
  Existing resident error categories keep their narrower mappings. Recode
  decoded-sample mismatch now uses `BackendErrorKind::Validation`, and the
  private generic facade string-to-backend constructors are removed.
- Renames backend capability detection to compile-time defaults and makes
  facade Metal/CUDA gating symmetric with those compile-time features.
- Renames the JPEG fast packet adapter surface from Metal-specific names to
  backend-neutral `JpegFast*` packet/table/error names.
- Contracts the published pre-1.0 `j2k` facade by removing experimental
  adaptive-routing and internal view/context surfaces plus legacy preference
  aliases. There is no compatibility shim for those implementation-facing
  items; use the supported facade codec methods and concrete backend requests.
- Replaces broad facade glob reexports with explicit export lists and adds the
  missing root `TileBatchDecodeDevice` and `TileBatchDecodeSubmit` traits.
- Removes blanket `From<String>` and `From<&str>` conversions for
  `j2k::BackendError`; downstream adapters must construct
  `BackendError::new(BackendErrorKind, message)` or use a typed convenience
  constructor so failures cannot silently collapse to `Other`.
- Removes implementation-detail parser, entropy, tile-helper, kernel, planning,
  and resident-handoff surfaces from `j2k-core`, `j2k-jpeg`, `j2k-tilecodec`,
  `j2k-jpeg-cuda`, `j2k-cuda`, `j2k-transcode-cuda`,
  `j2k-transcode-metal`, `j2k-native`, and `j2k-cuda-runtime`. Use the supported
  codec, adapter, and transcode entry points; removed low-level helpers have no
  general compatibility replacement.
- Narrows CUDA runtime root exports to explicit public modules and types. Its
  doc-hidden asynchronous HTJ2K cleanup and IDWT enqueue methods are now unsafe,
  as is early pool-release after completion. Direct runtime callers must retain
  every target, resource, and pool on the owning context until actual CUDA
  completion; the safe `j2k-cuda` facade owns that lifecycle for normal decode.
- Makes the doc-hidden CUDA baseline-JPEG Huffman representation opaque.
  Runtime-integrator code must construct it with
  `CudaJpegHuffmanTable::from_jpeg_bits_values`; arbitrary canonical-table
  field literals are no longer accepted at the safe decode boundary.
- Removes the redundant doc-hidden
  `j2k-cuda::CudaLosslessBufferEncodeOutcome::host_outcome` owner. Its scalar
  residency/timing fields are now flattened onto the buffer outcome.
  `CudaEncodedJ2k::encoded` and `encoded()` are replaced by
  `CudaEncodedJ2kMetadata` plus `metadata()`; the returned value retains only
  metadata and the CUDA-resident codestream buffer. Final assembly still uses
  host bytes transiently, but those bytes are copied directly to CUDA and
  released before the resident result is returned.
- Changes the `j2k-cuda` and `j2k-jpeg-cuda` `Error::CudaRuntime` payload from
  a flattened `message: String` to a typed `source: CudaError`. Direct runtime
  host-allocation failures map to each adapter's `HostAllocationFailed`
  variant, while kernel, completion, and resource-release error trees remain
  inspectable through the source chain.
- Adds `JpegEncodeError::MemoryCapExceeded` and `HostAllocationFailed` so
  callers exhaustively matching the pre-1.0 encoder error enum must handle
  bounded geometry plus fallible plane, entropy, and frame allocation
  explicitly. CPU restart encoding now uses a fixed upper bound of 64 ordered
  parallel chunks instead of allocating one vector per restart segment.
- Changes `J2kResidentEncodeInput::new` from `Result<_, &'static str>` to the
  typed, non-exhaustive `J2kResidentEncodeInputError`; adapters that still use
  the shared string-based stage SPI can call its stable `reason()` accessor.
- Makes `ProfileSummary` drop emission explicit and removes its deep `Clone`;
  retained summary graphs must be moved instead of duplicated infallibly.
- Makes JPEG sampling-factor construction fallible instead of panicking on
  caller input, and applies the shared default host allocation cap to the
  public JPEG output-buffer constructors and resize operations.

### Additive API Changes

- Adds `j2k_metal::MetalBufferPoolDiagnostics`,
  `MetalBufferPoolsDiagnostics`, and
  `MetalBackendSession::buffer_pool_diagnostics` on macOS. Long-lived sessions
  can inspect bounded private/shared scratch retention, allocator-reported
  metadata capacity, stable high-water marks, eviction/rejection counts, and
  accounting failures without exposing pooled Metal resources.
- Hardens CUDA JPEG decode and encode safe-plan validation to reject the
  JPEG-reserved all-ones code before driver work; generic canonical derivation
  remains decoder-compatible with complete prefix tables.
- Adds the no-std const
  `j2k-codec-math::dwt::max_decomposition_levels` geometry policy; native,
  facade, and CUDA encoders now share the same shorter-axis decomposition
  ceiling.
- Adds the allocation-free `j2k-codec-math::dwt::linearized_dwt53_row`
  primitive and fixed-capacity row/tap types. CPU and Metal transcode paths use
  the same constant-work symbolic 5/3 definition instead of maintaining
  separate dense-basis derivations.
- Adds an implementation-facing, no-std resident-input HTJ2K encode contract
  shared by `j2k-types`, `j2k-native`, and the `j2k` facade. CUDA-resident
  encode now carries validated geometry and sample format without allocating a
  fake host image, and a declined whole-tile hook fails explicitly instead of
  entering the CPU sample pipeline. Native resident contract violations,
  accelerator failures, and codestream-finalization failures retain distinct
  typed classifications.
- Adds the implementation-facing
  `Reversible53CoefficientImage::encode_htj2k` handoff used by the facade's
  coefficient recoder. It accounts the complete retained coefficient tree in
  the native encode session instead of exposing a caller-assembled allocation
  token.
- Binds CUDA-resident encode resources to `CudaSession`: compatible single and
  batch requests reuse one uploaded HT table set, while a tile from another
  CUDA context is rejected before resource upload or launch.
- Adds the non-exhaustive `j2k-native::DecodeErrorClass` and
  `DecodeError::classify` so facade and adapter code can classify backend,
  unsupported, short-input, and truncated-input failures without matching
  native implementation-detail variants.
- Adds `j2k-cuda-runtime::CudaError::CompletionFailed`, retaining both the
  original CUDA operation error and a failed context-wide completion check.
- Adds `j2k-cuda-runtime::CudaError::ResourceReleaseFailed` and
  `j2k-cuda::Error::CudaCleanupFailed` so a primary operation failure and a
  later retained-resource cleanup failure remain separately diagnosable.
- Adds the doc-hidden unsafe
  `j2k-cuda-runtime::CudaContext::submit_default_stream_named` helper for the
  few typed queued paths that intentionally return after submission. Existing
  safe default-stream timing helpers now establish completion even when timing
  collection is disabled.

### Maturity And Fixes

- Fixes confirmed stale-cache, malformed-input, tile-grid, GPU validator, FFI
  cleanup, unsafe-deposit, and test-helper drift findings from the release
  audit.
- Fixes paletted JP2/JPH pixel fallback so palette mappings are materialized as
  direct component planes before palette and component-mapping boxes are
  dropped. Round-trip validation now catches index-only output that would lose
  the resolved colors.
- Enforces shared host allocation caps for codec-owned output scratch, aligns
  lightweight J2K inspect SIZ validation with full decode, and caps bounded J2K
  row-decode stripes by bytes as well as rows.
- Bounds CUDA coefficient-transcode staging, readback widening, quantized band
  reslicing, and resident output metadata with checked phase-wide budgets and
  fallible allocation. Grouped resident assembly validates output identity and
  no longer rebuilds caller-sized results with infallible collection.
- Bounds CPU JPEG-to-HTJ2K coefficient extraction, reference transforms,
  Rayon batches, component grouping, validation output, and direct DCT 5/3 and
  9/7 scratch under aggregate simultaneously-live host budgets with typed
  allocation errors. Native encode receives only the remaining host budget
  after reusable scratch, reports, metrics, and prior batch codestreams are
  measured from allocator capacities; CPU tile encode is serialized instead
  of multiplying independent process caps. Validation histograms are now
  move-only sorted vectors with fallible one-shot reservation and typed
  length/cap/allocation failures. Progressive extraction consumes decoded
  `i32` planes incrementally and `DctExtractOptions::dequantized_only()` no
  longer retains the unused quantized `i16` plane. The former sparse 5/3 row
  builder's input-driven quadratic basis sweep is replaced by constant work per
  row.
- Bounds Metal coefficient-transcode host/device staging, readback, weight
  rows, code-block output, and component/resolution/subband metadata under
  checked aggregate budgets with fallible allocation and complete runtime
  diagnostics. The mixed crate root and transform owners are split into
  focused route, accelerator, reversible, irreversible, geometry, buffer,
  code-block-output, and weight modules without raising their structure caps.
- Bounds baseline JPEG encoder sample geometry before multiplication or plane
  allocation, reserves RGB planes fallibly, and borrows grayscale input
  directly instead of cloning the entire source image.
- Rejects sparse or unrepresentable packet-state indices before native or CUDA
  packetization allocates state arrays, including the `u32::MAX` safe-input
  boundary on 32-bit targets.
- Removes the legacy `nvcc`/C++ kernel probe and checked-in PTX fallback path;
  strict GPU validation now requires every enabled Rust CUDA-Oxide project to
  build successfully for the configured device architecture.
- Prevents pooled CUDA allocations from being recycled or destroyed while
  asynchronous cleanup or IDWT work can still reference them. Queued work now
  holds pool reuse, validates context ownership, synchronizes before release on
  ordinary errors, and poisons the context resource lifetime when completion
  cannot be established instead of risking reuse or deallocation. Failed CUDA
  resource creation, destruction, and ownership transfers quarantine the
  context even after successful synchronization because completion alone
  cannot prove whether the state transition committed.
- Rejects CUDA J2K store jobs whose source or destination arithmetic can leave
  allocation bounds or the device `u32` indexing ABI, and deterministically
  zero-initializes every unwritten byte in partial and zero-copy outputs.
- Validates every CUDA IDWT band/output extent and allocation before driver
  work, rejects overlapping queued targets, and makes unbounded sequence
  aggregation fallible. Forward CUDA DWT now rejects excessive levels,
  degenerate later-level geometry, and transforms whose live device index
  exceeds `u32`.
- Validates CUDA baseline JPEG encode input/entropy ranges, sampling and MCU
  geometry, kernel arithmetic, quantization tables, and canonical prefix-free
  Huffman codes before driver work. Batch entropy regions must be disjoint and
  single-tile input/output offsets are honored exactly.
- Validates CUDA baseline JPEG decode sampling grids, checkpoint coverage and
  bit state, pitched `u32` output addressing, quantizers, and canonical
  role-correct Huffman tables before driver work. Owned device output is
  initialized defensively, device-side validation reports malformed metadata
  instead of returning an OK status with unwritten pixels, and all three
  4:2:0/4:2:2/4:4:4 routes share the same coverage contract. High-level
  capability routing rejects RGB8 images beyond the device byte-address domain
  before constructing CUDA packets or checkpoints.
- Makes high-level JPEG entropy/checkpoint preparation fallible and bounded:
  checkpoint capacity follows actual cadence/restart semantics, already
  EOI-terminated entropy is borrowed, destuffing stops at the first EOI, and
  required staging uses the shared host-allocation cap rather than infallible
  caller-sized vectors.
- Makes tag publication fail closed on a real dated changelog heading,
  structured patch-review approval, an enabled private vulnerability-reporting
  setting, an annotated tag that peels to the exact candidate, and a clean
  publish worktree.
- Strengthens unsafe, fuzz, Miri, and dependency-advisory governance in CI.
- Moves repository policy checks into `cargo xtask repo-lint` and pins public
  API, release-integrity, environment-variable, workflow, and packaging
  invariants there.
- Documents supported `J2K_*` environment variables and removes the
  experiment-only JPEG Metal fast420 split selector from the runtime surface.
- Routes env-gated Metal timing output through `j2k-profile`.
- Adds stricter J2K component-plane validation before output writes and removes
  stale generated-table dead-code suppression.
- Adds source-aware changed-path coverage for production and build-script Rust,
  including accelerator-host routing. The gate reports non-production source
  roles separately and rejects missing executable coverage, changed functions
  without covered bodies, opaque changed macros, unreachable or untracked Rust,
  and stale or conflicting build-script cfg evidence.

## [0.6.2] - 2026-06-28

- Defines the public codec claim as JPEG 2000 Part 1 codestream support, JP2
  still-image wrapping, HTJ2K Part 15 support, and JPH wrapping, with
  JPX / JPEG 2000 Part 2 extensions explicitly out of scope.
- Adds the public support matrix and `cargo xtask public-support` gate covering
  signed/high-bit samples, wide component counts, progression/tile-part cases,
  packet-marker combinations, ROI maxshift, JP2/JPH metadata, HT refinement
  decode/encode, J2K-to-HTJ2K recode, and publication gates.
- Expands the public `j2k` facade with component-plane decode/encode APIs for
  arbitrary component counts, signedness, sampling, mixed bit depths, and
  native high-bit sample access while keeping Gray/RGB/RGBA convenience paths.
- Adds JP2/JPH wrapper helpers and validation for still-image metadata,
  codestream branding, color boxes, palettes/component mapping, channel
  definitions, BPCC, and ICC preservation paths.
- Completes repo-local HTJ2K cleanup/refinement, multi-layer/rate, JPH, and
  recode self-checks while keeping external OpenJPH/Kakadu evidence as a
  publication gate.
- Removes the cuda-oxide transcode IDCT per-thread local-memory table
  materialization, reducing the self-hosted RTX 4070 SUPER cuda-oxide
  JPEG-to-HTJ2K transcode profile from `40.813 MP/s` before the fix to
  `380.411 MP/s` in the `v0.6.2` validation run.
- Keeps cuda-oxide transcode opt-in and records the CUDA C vs cuda-oxide
  benchmark evidence in the public benchmark documentation.
- Adds `j2k transcode <input.jpg> <output.j2k> --htj2k --lossless-53` to the
  CLI as the first conservative JPEG-to-HTJ2K smoke-test command.
- Refreshes adoption-facing docs with a shorter quickstart, support matrix, and
  current `0.6.x` security/environment-variable line.

## [0.6.1] - 2026-06-22

- Refreshes public package metadata, docs.rs landing text, and README search
  signals for the `j2k` rename, Rust programming language discovery, and
  CUDA/Metal GPU JPEG 2000 / HTJ2K queries.

## [0.6.0] - 2026-06-20

- Stages the `j2k` facade release.
- Keeps CPU decode as the portable correctness baseline.
- Treats Runtime backend selection defaults to `Auto` as the public backend
  policy.
- Adds resident Metal and CUDA device memory surfaces for supported adapter
  paths through cuda-runtime integration.
- Uses J2K-owned CUDA kernels for supported CUDA codec stages.
- Requires recorded benchmark evidence before NVIDIA performance claims.
- Consolidates shared J2K encode-stage, CUDA submit, Metal runtime, tilecodec,
  JPEG output, and test-support helpers.
