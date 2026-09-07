# Environment Variables

This is the supported `J2K_*` environment-variable surface for the
workspace. Variables not listed here are internal symbols, generated metadata,
or test-only implementation details and must not be treated as user controls.

Stability values:

- Stable: supported for the published v0.11.x contract.
- Experimental: accepted for diagnostics or adapter tuning, but may change
  before 1.0.
- Test/CI: supported only for repository tests, CI, and release validation.
- Benchmark: supported only for benchmark harnesses and benchmark signoff.
- Generated: emitted by a build script for version reporting; do not set by
  hand unless reproducing the build-script contract.

The xtask identifier `J2K_METAL_REQUIRED_IGNORED_TESTS` is not an environment
variable. It is the compile-time inventory of ignored Metal runtime tests that
`cargo xtask release-metal` must execute exactly; operators cannot set or
override it.

| Internal identifier | Meaning | Operator default | Stability |
| --- | --- | --- | --- |
| `J2K_METAL_REQUIRED_IGNORED_TESTS` | Compile-time xtask inventory used to reject missing, unexpected, skipped, or partially executed ignored Metal runtime tests. This is not read from the environment. | Fixed in xtask source | Internal validation |

## Library Runtime And Profiling

| Variable | Effect | Default | Stability |
| --- | --- | --- | --- |
| `J2K_GPU_ROUTE_PROFILE` | Emits facade/adapter GPU route decisions. Use `1` for rows or `summary` for aggregate rows. | Disabled | Experimental |
| `J2K_JPEG_PROFILE_STAGES` | Emits JPEG CPU profiling rows. Use `1` for rows or `summary` where supported. | Disabled | Experimental |
| `J2K_PROFILE_STAGES` | Emits native/CUDA J2K profiling rows. Use `1` for rows or `summary` where supported. | Disabled | Experimental |
| `J2K_CUDA_TRACE` | Writes CUDA HTJ2K profile trace JSON to an operator-supplied path. Existing files are not overwritten, and parent directories are not created. | No trace file | Experimental |
| `J2K_CUDA_IDWT_TRACE` | Enables CUDA IDWT trace/profile output. | Disabled | Experimental |
| `J2K_CUDA_IDWT_WIDE_TILED` | Historical P14 switch; it has no effect after the tiled cooperative candidate was rejected. | No effect | Historical |
| `J2K_CUDA_DISABLE_SHARED_FDWT97` | Historical P15 split-process benchmark switch; it has no effect after the shared-staging candidate was rejected. | No effect | Historical |
| `J2K_CUDA_FDWT97_TRACE` | Historical P15 launch-mode trace control; it has no effect after the shared-staging candidate was rejected. | No effect | Historical |
| `J2K_CUDA_P17_PROFILE` | Restricts the CUDA HTJ2K decode benchmark to the P17 pre-prototype final-IDWT/store attribution matrix. It does not select a production route. | Disabled | Benchmark |
| `J2K_CUDA_DISABLE_FUSED_INPUT_MCT` | Historical P16 split-process benchmark switch; it has no effect after the input-fusion candidate was rejected. | No effect | Historical |
| `J2K_CUDA_JPEG_DISABLE_STAGED_BASELINE_ENCODE` | Historical P18 split-process A/B control; it has no effect after staged baseline encode promotion and serial fallback removal. | No effect | Historical |
| `J2K_CUDA_JPEG_DISABLE_PACKED_CHECKPOINTS` | Historical P19 split-process A/B control; it has no effect after adaptive checkpoint launch promotion and removal of the all-serial fallback branch. | No effect | Historical |
| `J2K_CUDA_JPEG_DISABLE_COEFFICIENT_IDCT_SPLIT` | Historical P19 split-process A/B control; it has no effect after the coefficient/IDCT split candidate was rejected and removed. | No effect | Historical |
| `J2K_CUDA_HT_ENCODE_BLOCK_THREADS` | Historical P26 split-process control for 1/32/128-thread HT encode blocks. Removed after the smaller-block candidates were rejected. | No effect | Historical |
| `J2K_CUDA_DISABLE_STAGE_TIMINGS` | Disables CUDA stage timing collection for benchmark runs. | Timings enabled | Experimental |
| `J2K_CUDA_DISABLE_COMPACT_PREENCODED` | Forces the CUDA transcode adapter to decline compact preencoded resident HT encode support. | Compact resident support enabled when supported | Experimental |
| `J2K_JPEG_METAL_FAST420_BATCH_TIMING` | Emits JPEG Metal fast 4:2:0 batch timing profiles. Use `1` for rows or `summary` for aggregate rows. | Disabled | Experimental |
| `J2K_METAL_PROFILE_STAGES` | Enables J2K Metal stage profile rows for truthy values such as `1`; `summary` / `aggregate` emits aggregate rows where the profile path supports summaries. | Disabled | Experimental |
| `J2K_METAL_PROFILE_SIGNPOSTS` | Enables J2K Metal OS signposts when set to `1` and stage profiling is enabled. | Disabled | Experimental |
| `J2K_METAL_PROFILE_DECODE_LABEL` | Labels J2K Metal decode profile rows. Non-alphanumeric characters are sanitized. | `unlabeled` | Experimental |
| `J2K_METAL_PROFILE_DECODE_SPLIT_COMMANDS` | Adds split-command decode timing rows when J2K Metal stage profiling is enabled. | Disabled | Experimental |
| `J2K_METAL_PROFILE_COEFFICIENT_PREP_SPLIT_COMMANDS` | Adds split-command coefficient-prep timing rows when J2K Metal stage profiling is enabled. | Disabled | Experimental |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_DENSITY` | Emits classic Tier-1 density profiling when J2K Metal stage profiling is enabled. | Disabled | Experimental |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_RAW_PACK` | Emits classic Tier-1 raw-pack profiling when J2K Metal stage profiling is enabled. | Disabled | Experimental |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_ARITHMETIC_PACK` | Emits classic Tier-1 arithmetic-pack profiling when J2K Metal stage profiling is enabled. | Disabled | Experimental |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_SYMBOL_PLAN` | Emits classic Tier-1 symbol-plan profiling when J2K Metal stage profiling is enabled. | Disabled | Experimental |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_PASS_PLAN` | Emits classic Tier-1 pass-plan profiling and also enables symbol-plan profiling. | Disabled | Experimental |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_TOKEN_EMIT` | Emits classic Tier-1 token-emission profiling when J2K Metal stage profiling is enabled. | Disabled | Experimental |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_SPLIT_TOKEN_EMIT` | Emits classic Tier-1 split-token-emission profiling when J2K Metal stage profiling is enabled. | Disabled | Experimental |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_TOKEN_PACK` | Emits classic Tier-1 token-pack profiling and also enables token-emission profiling. | Disabled | Experimental |
| `J2K_METAL_DISABLE_FUSED_IDWT97` | Historical P2 same-process benchmark switch recorded in the validated rejection evidence; no production route currently reads it. | No effect after rejected prototype cleanup | Benchmark record |
| `J2K_METAL_DISABLE_FUSED_IDWT53_INTERLEAVE_HORIZONTAL` | Historical P1 same-process benchmark switch recorded in the validated rejection evidence; no production route currently reads it. | No effect after rejected prototype cleanup | Benchmark record |
| `J2K_METAL_DISABLE_FUSED_FDWT97` | Historical P3 same-process benchmark switch recorded in validated rejection evidence; no production route currently reads it. | No effect after rejected prototype cleanup | Benchmark record |
| `J2K_METAL_DISABLE_COOPERATIVE_PACKETIZATION` | Historical P11 same-process benchmark switch recorded in validated rejection evidence; no production route currently reads it. | No effect after rejected prototype cleanup | Benchmark record |
| `J2K_TRANSCODE_METAL_DISABLE_FUSED_COLUMN_QUANTIZE` | Historical P12 same-host benchmark switch recorded in the validated rejection evidence; no production route currently reads it. | No effect after rejected prototype cleanup | Benchmark record |
| `J2K_CUDA_DISABLE_DWT97_FUSED_COLUMN_QUANTIZE` | Historical P13 split-process benchmark switch recorded in the validated rejection evidence; no production route currently reads it. | No effect after rejected prototype cleanup | Benchmark record |
| `J2K_JPEG_METAL_DISABLE_SPLIT_COEFF_IDCT` | Historical P19 same-host benchmark switch recorded in validated rejection evidence; no production route currently reads it. | No effect after rejected prototype cleanup | Benchmark record |
| `J2K_JPEG_METAL_DISABLE_STAGED_BASELINE_ENCODE` | Historical P18 same-host A/B switch recorded in validated promotion evidence; the promoted staged route has no runtime switch and production code no longer reads this variable. | No effect | Historical experiment control |
| `J2K_METAL_DISABLE_FUSED_INPUT_MCT` | Historical P5 A/B switch retained in experiment records. The promoted route is now limited to the measured unsigned RGB8 512x512 geometry, and production code no longer reads this variable. | No effect | Historical experiment control |
| `J2K_METAL_DISABLE_BATCHED_IDWT97` | Historical P20 A/B control for per-image versus bounded batched 9/7 reconstruction. Removed after promotion; production code no longer reads it. | No effect | Historical experiment control |
| `J2K_METAL_DISABLE_IDWT97_CHUNKS` | Historical P27 A/B control for 16 MiB chunks above the existing batched reconstruction limit. Removed with the rejected candidate. | No effect | Historical experiment control |
| `J2K_METAL_DISABLE_RESIDENT_LOSSY` | Historical P28 A/B control for staged versus resident transform-through-HT lossy encoding. Removed after promotion. | No effect | Historical experiment control |
| `J2K_METAL_FORCE_SPLIT_ENCODE` | Historical P21 A/B control for HT command-buffer coalescing. Removed with the rejected candidate. | No effect | Historical experiment control |
| `J2K_METAL_DISABLE_SMALL_HT_STATE` | Historical P21/P22 A/B control for width-specialized HT context arrays. Removed with the rejected candidate. | No effect | Historical experiment control |
| `J2K_METAL_IDWT97_SMALL_GROUPS` | Historical P25 A/B control for 128-thread inverse-transform groups. Removed with the rejected candidate. | No effect | Historical experiment control |
| `J2K_TRANSCODE_METAL_PROFILE_STAGES` | Enables transcode Metal profiling in the DCT 5/3 and 9/7 benchmark harness. Use `1` for rows or `summary` for aggregate rows. | Disabled | Benchmark |

## Experimental Backend Routing

| Variable | Effect | Default | Stability |
| --- | --- | --- | --- |
| `J2K_METAL_HT_PACKET_CAPACITY` | Overrides J2K Metal HT packet buffer capacity. Invalid values are ignored by the fallback parser. | Built-in capacity | Experimental |
| `J2K_METAL_CLASSIC_SELECTIVE_BYPASS` | Set to `0` to disable classic selective arithmetic coding bypass in the Metal resident style flags. | Selective bypass enabled | Experimental |
| `J2K_METAL_CLASSIC_TIER1_GPU_TOKEN_PACK` | Requests the classic Tier-1 GPU token-pack route when supported. | Disabled | Experimental |
| `J2K_METAL_CLASSIC_TIER1_SPLIT_GPU_TOKEN_PACK` | Requests the split classic Tier-1 GPU token-pack route. | Disabled | Experimental |
| `J2K_METAL_CLASSIC_TIER1_SPLIT_MQ_BYTE_GPU_TOKEN_PACK` | Set to `1` to request, or `0` to disable, the split MQ-byte GPU token-pack route. | Auto | Experimental |
| `J2K_HYBRID_FLAT_CPU_TIER1` | Forces flat CPU Tier-1 batching for J2K Metal hybrid decode when set to a truthy value accepted by the adapter. | Adapter default | Experimental |
| `J2K_CUDA_OXIDE_ARCH` | Overrides the cuda-oxide build target when a `j2k-cuda-runtime/cuda-oxide-*` feature is enabled. | `sm_80` | Experimental |
| `J2K_REQUIRE_CUDA_OXIDE_BUILD` | Requires every enabled cuda-oxide PTX project to build successfully. Use this on Linux/NVIDIA validation and benchmark hosts. | Disabled | Test/CI |

## Test And CI Gates

| Variable | Effect | Default | Stability |
| --- | --- | --- | --- |
| `J2K_ML_BATCH_INPUT_MODE` | Selects content-distinct generated inputs (`distinct`) or one repeated `Arc` owner (`repeated`) for the `j2k-ml` batch benchmarks. One process uses one mode. | `distinct` | Benchmark |
| `J2K_MPSGRAPH_BENCH_ITERATIONS` | Number of repeated samples per size/batch/path cell in the direct MPSGraph benchmark. | `5` | Benchmark |
| `J2K_JPEG_TEXTURE_SMALL_BATCHES` | When set to any value, selects the small-batch and tile-size control matrix for the ignored JPEG Metal `texture_scheduling_experiment` test. | Regular 16-tile timing matrix | Benchmark |
| `J2K_ML_BATCH_PROCESS_MODE` | Selects uninstrumented Criterion measurement (`criterion`) or the separate low-batch telemetry/profile process (`profile`) for CUDA and Metal batch benchmarks. | `criterion` | Benchmark |
| `J2K_REQUIRE_OPENJPEG` | Makes OpenJPEG parity tests and benchmark comparator runs fail instead of skip when OpenJPEG tools are unavailable. | Skip unavailable comparator paths | Benchmark |
| `J2K_REQUIRE_GROK` | Makes Grok parity tests and benchmark comparator runs fail instead of skip when Grok tools or libraries are unavailable. | Skip unavailable comparator paths | Benchmark |
| `J2K_REQUIRE_OPENJPH` | Makes optional OpenJPH fixture comparator rows fail instead of skip when `ojph_expand` is unavailable. Intended only for HTJ2K/JPH-compatible CLI context rows. | Skip unavailable OpenJPH path unless explicitly included | Benchmark |
| `J2K_REQUIRE_KAKADU` | Makes optional Kakadu fixture/encoder comparator rows fail instead of skip when `kdu_expand` or `kdu_compress` is unavailable. Intended only for proprietary CLI/file-output context rows. | Skip unavailable Kakadu path unless explicitly included | Benchmark |
| `J2K_REQUIRE_LIBJPEG_TURBO` | Makes libjpeg-turbo comparison tests fail instead of skip when the bench feature/tooling is unavailable. | Skip unavailable comparator path | Test/CI |
| `J2K_REQUIRE_CUDA_RUNTIME` | Makes CUDA tests and benchmarks require a usable CUDA runtime instead of skipping. | Skip runtime-only CUDA paths | Test/CI |
| `J2K_CUDA_LOSSY_PARITY_PNM` | Path to an external RGB PNM fixture used by required CUDA irreversible color-transform, DWT 9/7, and lossy encode parity tests. | Optional generated fixture unless the release runner supplies a file | Test/CI |
| `J2K_REQUIRE_CUDA_JPEG_HARDWARE_DECODE` | Requires CUDA JPEG hardware decode coverage in relevant CUDA tests/benches. | Hardware decode may skip | Test/CI |
| `J2K_REQUIRE_METAL_RUNTIME` | Runs runtime-only Metal tests and makes them require a usable Metal runtime instead of default-skipping. | Skip runtime-only Metal paths | Test/CI |
| `J2K_REQUIRE_CUDA_BENCH` | Makes CUDA benchmark probes fail instead of skip when CUDA is unavailable or does not dispatch. | Skip unavailable CUDA benchmark paths | Benchmark |
| `J2K_REQUIRE_METAL_BENCH` | Makes Metal benchmark probes fail instead of skip when Metal is unavailable or does not dispatch. | Skip unavailable Metal benchmark paths | Benchmark |
| `J2K_GPU_CACHE_ENV` | Internal CI fingerprint of the accelerator driver/runtime and compiler environment, included in the GPU dependency-cache key. The GPU validation workflow computes this value; operators should not set it manually. | Computed by GPU CI | Test/CI |
| `J2K_REQUIRE_WSI_ROOT` | Makes external JPEG WSI tests fail if `J2K_WSI_ROOT` is missing or empty. | Skip external WSI tests | Test/CI |
| `J2K_REQUIRE_TRANSCODE_WSI_ROOT` | Makes transcode corpus validation require `J2K_TRANSCODE_WSI_ROOT`. | Skip external transcode WSI corpus | Test/CI |
| `J2K_REQUIRE_NDPI` | Makes NDPI passthrough tests fail if `J2K_NDPI_PATH` is missing. | Skip NDPI passthrough test | Test/CI |

## External Data And Comparator Paths

| Variable | Effect | Default | Stability |
| --- | --- | --- | --- |
| `J2K_WSI_ROOT` | Path list root for external JPEG WSI integration tests. | Not set | Test/CI |
| `J2K_WSI_SVS_PATH` | Path to an external Aperio SVS fixture used by J2K Metal IDWT and in-process parity tests. | Not set | Test/CI |
| `J2K_TRANSCODE_WSI_ROOT` | Path list root for external JPEG-to-HTJ2K transcode corpus validation. | Not set | Test/CI |
| `J2K_TRANSCODE_WSI_TILE_LIMIT` | Maximum number of external WSI tiles used by transcode corpus validation. | Built-in validation default | Test/CI |
| `J2K_TRANSCODE_WSI_MAX_PAYLOAD_BYTES` | Maximum external WSI tile payload accepted by transcode corpus validation. | Built-in validation default | Test/CI |
| `J2K_NDPI_PATH` | Path to an NDPI slide for passthrough tests. | Not set | Test/CI |
| `J2K_NDPI_TILE_LIMIT` | Maximum NDPI tiles inspected by the passthrough test; `0` means no explicit tile-count cap. | Test default | Test/CI |
| `J2K_NDPI_MAX_PAYLOAD_BYTES` | Maximum NDPI tile payload accepted by the passthrough test. | Test default | Test/CI |
| `J2K_ISO_CONFORMANCE_DIR` | Path to ISO J2K conformance vectors for ignored/external tests. | Not set | Test/CI |
| `J2K_APERIO_TILE_FIXTURE` | Path to an Aperio J2K tile fixture for the ignored lossless encode test. | Not set | Test/CI |
| `J2K_PARITY_CORPUS_MANIFEST` | Manifest path used by `scripts/parity-corpus-fetch.sh`. | `corpus/wsi-samples/manifest.json` or first script argument | Test/CI |
| `J2K_PARITY_CORPUS_DIR` | Output directory used by `scripts/parity-corpus-fetch.sh`. | `corpus/wsi-samples` or second script argument | Test/CI |
| `J2K_PARITY_CORPUS_MAX_BYTES` | Maximum accepted byte size for each downloaded parity-corpus fixture. | `536870912` | Test/CI |
| `J2K_OPENJPEG_BIN` | Override for OpenJPEG `opj_decompress` in J2K parity tests. | `opj_decompress` on `PATH` | Test/CI |
| `J2K_OPENJPEG_DECOMPRESS_BIN` | Override for OpenJPEG `opj_decompress` in benchmark reports. | `opj_decompress` on `PATH` | Benchmark |
| `J2K_OPENJPEG_COMPRESS_BIN` | Override for OpenJPEG `opj_compress` in J2K parity tests and encoder benchmark reports. | `opj_compress` on `PATH` | Benchmark |
| `J2K_OPENHTJ2K_DEC_BIN` | Absolute path to the pinned `open_htj2k_dec` executable used by the T.803 CPU encoder matrix's HT+RGN interoperability case. `scripts/prepare-openhtj2k-reference.sh` sets it in CI. | Required when the selected matrix contains the OpenHTJ2K case | Test/CI |
| `J2K_OPENHTJ2K_SOURCE_DIR` | Absolute path to the clean official OpenHTJ2K 0.19.0 source checkout at the pinned commit. The runner verifies its origin, tag, commit, and tracked-file status before executing the decoder. | Required with `J2K_OPENHTJ2K_DEC_BIN` | Test/CI |
| `J2K_OPENHTJ2K_LIB_DIR` | Directory containing the pinned OpenHTJ2K 0.19.0 static library used by the in-process comparator. `scripts/prepare-openhtj2k-reference.sh` emits it with the source and CLI paths. | Not linked unless both source and library artifacts are available | Test/benchmark |
| `J2K_OPENHTJ2K_VERSION` | Build-script metadata for the exact linked OpenHTJ2K tag; consumers should not set it directly. | Exact source tag, or `unknown` | Build metadata |
| `J2K_OPENJPH_EXPAND_BIN` | Override for OpenJPH `ojph_expand` in optional fixture comparator rows. | `ojph_expand` on `PATH`, `/opt/homebrew/bin/ojph_expand`, or `/usr/local/bin/ojph_expand` | Benchmark |
| `J2K_OPENJPH_COMPRESS_BIN` | Override for the pinned OpenJPH `ojph_compress` used by `jp2k_encode_compare --openjph-matrix`. `scripts/prepare-openjph-reference.sh` builds and emits both OpenJPH CLI paths. | Pinned `target/reference/openjph-0.31.0` build, then `ojph_compress` on `PATH` | Benchmark |
| `J2K_OPENJPH_SOURCE_DIR` | Absolute path to the clean official OpenJPH 0.31.0 source checkout at the pinned commit used by the in-process comparator. `scripts/prepare-openjph-reference.sh` creates and validates it. | Not linked unless set | Test/benchmark |
| `J2K_OPENJPH_LIB_DIR` | Directory containing the pinned OpenJPH 0.31.0 static library. `scripts/prepare-openjph-reference.sh` emits it with the source and CLI paths. | Not linked unless both source and library artifacts are available | Test/benchmark |
| `J2K_OPENJPH_VERSION` | Build-script metadata for the exact linked OpenJPH tag; consumers should not set it directly. | Exact source tag, or `unknown` | Build metadata |
| `J2K_KDU_EXPAND_BIN` | Override for Kakadu `kdu_expand` in optional fixture comparator rows. | `kdu_expand` on `PATH`, `/opt/homebrew/bin/kdu_expand`, or `/usr/local/bin/kdu_expand` | Benchmark |
| `J2K_KDU_COMPRESS_BIN` | Override for Kakadu `kdu_compress` in optional encoder comparator rows. | `kdu_compress` on `PATH`, `/opt/homebrew/bin/kdu_compress`, or `/usr/local/bin/kdu_compress` | Benchmark |
| `J2K_GROK_BIN` | Override for Grok `grk_decompress` in J2K parity tests. | `grk_decompress` on `PATH` | Test/CI |
| `J2K_GROK_COMPRESS_BIN` | Override for Grok `grk_compress` in J2K parity tests and encoder benchmark reports. | `grk_compress` on `PATH` | Benchmark |
| `J2K_GROK_ROOT` | Path to Grok installation/library root for in-process comparator builds and benchmark reports. | Not set | Test/CI |
| `J2K_GROK_SOURCE` | Path to Grok source used by the in-process comparator build script. | Not set | Test/CI |
| `J2K_GROK_VERSION` | Build-script emitted Grok version metadata consumed by the comparator crate. | `unavailable` if not emitted | Generated |
| `J2K_GROK_LIB_DIR` | Build-script emitted Grok library path metadata consumed by the comparator crate. | `unavailable` if not emitted | Generated |

## Benchmark Harnesses

| Variable | Effect | Default | Stability |
| --- | --- | --- | --- |
| `J2K_BENCH_INPUTS` | Path-list of JPEG inputs for JPEG and Metal benchmark/report harnesses. | Harness-specific generated fixtures | Benchmark |
| `J2K_REPORT_ITERS` | Iteration count for custom report-style JPEG benchmarks. | Harness default | Benchmark |
| `J2K_ALLOC_REPORT` | Enables allocation report output in the CPU JPEG encode benchmark. | Disabled | Benchmark |
| `J2K_FORCE_FULL_FRAME` | Forces benchmark classification to full-frame mode. | Auto classification | Benchmark |
| `J2K_JPEG_TILE_BATCH_SIZE` | Overrides JPEG tile batch size in benchmark comparison runs. | Harness default | Benchmark |
| `J2K_JPEG_ENCODE_BENCH_DIM` | Image dimensions for the JPEG Metal baseline encode benchmark. | Harness default | Benchmark |
| `J2K_JPEG_ENCODE_BENCH_BATCH` | Batch size for the JPEG Metal baseline encode benchmark. | Harness default | Benchmark |
| `J2K_JPEG_ENCODE_BENCH_QUALITY` | Quality for the JPEG Metal baseline encode benchmark. | Harness default | Benchmark |
| `J2K_GPU_BENCH_JPEG` | JPEG fixture path for GPU JPEG upload/decode benchmarks. | Generated fixture unless strict bench requires a file | Benchmark |
| `J2K_CUDA_BENCH_JPEG` | CUDA-specific JPEG fixture path; falls back to `J2K_GPU_BENCH_JPEG` in CUDA JPEG benches. | Not set | Benchmark |
| `J2K_GPU_BENCH_SMALL_FIXTURE` | Allows GPU JPEG benches to use small generated fixtures. | Disabled | Benchmark |
| `J2K_GPU_BENCH_DIM` | Generated GPU benchmark dimensions, usually `N` or `WIDTHxHEIGHT`. | Harness default | Benchmark |
| `J2K_GPU_BENCH_BATCH` | Generated GPU benchmark batch count. | Harness default | Benchmark |
| `J2K_GPU_BENCH_BATCH_DIM` | Generated GPU benchmark batch tile dimensions. | Harness default | Benchmark |
| `J2K_GPU_BENCH_RESTART_INTERVAL` | Restart interval for generated GPU JPEG upload benchmark fixtures. | Harness default | Benchmark |
| `J2K_GPU_BENCH_SUBSAMPLING` | Generated GPU JPEG benchmark subsampling, accepted values depend on the harness. | Harness default | Benchmark |
| `J2K_CUDA_BENCH_SUBSAMPLING` | CUDA-specific generated JPEG benchmark subsampling; falls back with `J2K_GPU_BENCH_SUBSAMPLING`. | Harness default | Benchmark |
| `J2K_FIXTURE_COMPARE_REPEATS` | Repeat count for `jp2k_fixture_compare`. | Tool default | Benchmark |
| `J2K_FIXTURE_COMPARE_MODE` | Benchmark mode for `jp2k_fixture_compare`: `portable-native` for native comparable rows, `portable-emulated` for task-equivalent rows with `decode_method` labels, or `capability` for feature coverage with explicit skips. | `portable-native` | Benchmark |
| `J2K_FIXTURE_COMPARE_BATCH_SIZES` | Backward-compatible comma-separated batch sizes for `jp2k_fixture_compare`; when set, it applies to both per-fixture rows and mixed external rows. Prefer the case/mixed split below for full adoption runs. | Not set | Benchmark |
| `J2K_FIXTURE_COMPARE_BATCH_SIZE` | Backward-compatible single batch size for `jp2k_fixture_compare` when `J2K_FIXTURE_COMPARE_BATCH_SIZES` is unset. | Tool default | Benchmark |
| `J2K_FIXTURE_COMPARE_CASE_BATCH_SIZES` | Comma-separated per-fixture detail batch sizes for `jp2k_fixture_compare`. Publication defaults keep every fixture covered at batch `1`; large throughput batches are measured by mixed external rows. | `1` | Benchmark |
| `J2K_FIXTURE_COMPARE_MIXED_BATCH_SIZES` | Comma-separated mixed-external throughput batch sizes for `jp2k_fixture_compare`. Publication defaults include the huge-batch decode matrix without multiplying every fixture row by every large batch size. | `1,16,256,1024` | Benchmark |
| `J2K_FIXTURE_COMPARE_THREADS` | Worker count for `jp2k_fixture_compare`. | Tool default | Benchmark |
| `J2K_FIXTURE_COMPARE_INPUT_DIRS` | Optional path-list of directories recursively scanned for external `.j2k`, `.j2c`, `.jp2`, `.jph`, or `.jhc` fixtures included in `jp2k_fixture_compare`. | Not set | Benchmark |
| `J2K_FIXTURE_COMPARE_INPUT_DIR` | Backward-compatible single external fixture directory recursively scanned when `J2K_FIXTURE_COMPARE_INPUT_DIRS` is unset. | Not set | Benchmark |
| `J2K_FIXTURE_COMPARE_MANIFEST` | Optional TSV manifest for external fixtures. Requires `path` and `corpus_category`; supports `corpus_name`, `license_status`, `encode_command`, `input_fnv1a64`, `source_fnv1a64`, `codec`, and `container` (`raw-codestream`, `j2k`, `j2c`, `jp2`, `jph`, or `jhc`). Publication runs require fixture hash, codec, and container pins; materialized raw/boxed variants should include `source_fnv1a64` so source diversity is not inflated by container wrappers. For native compressed corpora, `source_fnv1a64` normally equals `input_fnv1a64`; publishable decode claims require independent native compressed classic J2K and HTJ2K coverage, not only repo-materialized codestreams. | Not set | Benchmark |
| `J2K_FIXTURE_COMPARE_INCLUDE_GENERATED` | Set to `0`, `false`, `no`, or `off` to omit generated smoke fixtures from `jp2k_fixture_compare` external-corpus publication runs. | Generated fixtures included | Benchmark |
| `J2K_INCLUDE_OPENJPH` | Adds optional OpenJPH `ojph_expand` rows to `jp2k_fixture_compare`. Rows are HTJ2K/JPH-compatible full/scaled CLI/file-output context rows labeled `decode_method=openjph-cli-process-output-pnm`; unsupported fixtures are skipped explicitly. | Disabled | Benchmark |
| `J2K_INCLUDE_KAKADU` | Adds optional Kakadu `kdu_expand` fixture rows and `kdu_compress` encoder rows. Rows are proprietary CLI/file-output context rows labeled separately from the default J2K/OpenJPEG/Grok matrix. | Disabled | Benchmark |
| `J2K_ENCODE_COMPARE_REPEATS` | Repeat count for `jp2k_encode_compare`. | Tool default | Benchmark |
| `J2K_ENCODE_COMPARE_BATCH_SIZES` | Backward-compatible comma-separated batch sizes for `jp2k_encode_compare`; when set, it applies to both per-source rows and mixed external rows. Prefer the case/mixed split below for full adoption runs. | Not set | Benchmark |
| `J2K_ENCODE_COMPARE_CASE_BATCH_SIZES` | Comma-separated per-source detail batch sizes for `jp2k_encode_compare`. Publication defaults keep every source image covered at batch `1`. | `1` | Benchmark |
| `J2K_ENCODE_COMPARE_MIXED_BATCH_SIZES` | Comma-separated mixed-source throughput batch sizes for `jp2k_encode_compare`; rows encode identical staged PNM bytes through J2K, OpenJPEG, and Grok CLI processes with rotating encoder sample order. | `1,16,256` | Benchmark |
| `J2K_ENCODE_COMPARE_INPUT_DIRS` | Optional path-list of directories recursively scanned for external 8-bit gray/RGB `.pgm`, `.ppm`, `.pnm`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, or `.bmp` source images included in `jp2k_encode_compare`. Non-PNM images are decoded to canonical PNM outside the timed loop. | Not set | Benchmark |
| `J2K_ENCODE_COMPARE_MANIFEST` | Optional TSV manifest for external encode source images. Requires `path` and `corpus_category`; supports `corpus_name`, `license_status`, `source_command`, and `input_fnv1a64`. Publication runs require the decoded-pixel hash pin. | Not set | Benchmark |
| `J2K_ENCODE_COMPARE_INCLUDE_GENERATED` | Set to `0`, `false`, `no`, or `off` to omit generated smoke source images from `jp2k_encode_compare` external-corpus publication runs. | Generated source images included | Benchmark |
| `J2K_ENCODE_COMPARE_ENCODERS` | Comma-separated encoder filter for local smoke checks, with values `j2k`, `openjpeg`, `grok`, and optional `kakadu`/`kdu`. Any filter blocks publication eligibility. | All default encoders | Benchmark |
| `J2K_OPENJPH_MATRIX_REPEATS` | Positive encode repeat count per producer/profile/format cell in `jp2k_encode_compare --openjph-matrix`; the median is reported. Parity is checked outside the timed encode operation. | `3` | Benchmark |
| `J2K_BATCH_COMPARE_REPEATS` | Repeat count for `jp2k_batch_compare`. | Tool default | Benchmark |
| `J2K_BATCH_COMPARE_THREADS` | Outer worker count shared by J2K and the in-process reference rows in `jp2k_batch_compare`. Reference decoders remain single-threaded per tile to avoid nested oversubscription. | Tool default | Benchmark |
| `J2K_BATCH_COMPARE_MAX_ABS_DIFF` | Maximum allowed per-byte difference between J2K and each linked OpenHTJ2K/OpenJPH in-process preflight. The benchmark exits before timing when the gate fails. | `1` | Benchmark |
| `J2K_ROI_COMPARE_REPEATS` | Repeat count for `jp2k_roi_batch_compare`. | Tool default | Benchmark |
| `J2K_ROI_COMPARE_THREADS` | Worker count for `jp2k_roi_batch_compare`. | Tool default | Benchmark |
| `J2K_CUDA_DECODE_FORMATS` | Comma-separated CUDA J2K decode benchmark output formats such as `gray8,rgb8,rgba8`. | Harness default | Benchmark |
| `J2K_CUDA_DECODE_BATCH_SIZES` | Comma-separated CUDA J2K decode mixed-external batch sizes. | Harness default | Benchmark |
| `J2K_CUDA_DECODE_CASE_BATCH_SIZES` | Comma-separated CUDA J2K decode per-fixture batch sizes. Keep this bounded for external adoption runs; use `J2K_CUDA_DECODE_BATCH_SIZES` for huge mixed batches. | Harness default | Benchmark |
| `J2K_CUDA_DECODE_SAMPLE_SIZE` | Criterion sample size for CUDA J2K decode benchmark rows. Must be at least 10. | 10 | Benchmark |
| `J2K_CUDA_DECODE_MEASUREMENT_SECONDS` | Positive measurement duration in seconds for each CUDA J2K decode Criterion row. | 1 second | Benchmark |
| `J2K_CUDA_DECODE_INPUT_DIRS` | Optional path-list of external HTJ2K `.j2k`, `.j2c`, `.jp2`, `.jph`, or `.jhc` fixtures included in the CUDA decode Criterion benchmark. | Not set | Benchmark |
| `J2K_CUDA_DECODE_MANIFEST` | Optional TSV manifest for CUDA decode external fixtures. Uses the same pinned `path`, `input_fnv1a64`, `codec`, and `container` fields as `J2K_FIXTURE_COMPARE_MANIFEST`. | Not set | Benchmark |
| `J2K_CUDA_DECODE_INCLUDE_GENERATED` | Set to `0`, `false`, `no`, or `off` to omit generated CUDA decode fixtures when external fixtures are provided. | Generated CUDA decode fixtures included | Benchmark |
| `J2K_CUDA_ENCODE_INPUT_DIRS` | Optional path-list of staged external `.pgm`, `.ppm`, or `.pnm` source images included in the CUDA HTJ2K encode Criterion benchmark. Use the same canonical PNM source assets as `J2K_ENCODE_COMPARE_INPUT_DIRS` after staging. | Not set | Benchmark |
| `J2K_CUDA_ENCODE_MANIFEST` | Optional TSV manifest for CUDA encode staged PNM sources. Uses `path` and pinned `input_fnv1a64` from `J2K_ENCODE_COMPARE_MANIFEST`. | Not set | Benchmark |
| `J2K_CUDA_ENCODE_INCLUDE_GENERATED` | Set to `0`, `false`, `no`, or `off` to omit the generated CUDA host-input encode row when external staged PNM sources are provided. Code-block/device-input microbenchmarks remain generated component rows. | Generated CUDA host-input row included | Benchmark |
| `J2K_CUDA_ENCODE_SAMPLE_SIZE` | Criterion sample size for CUDA HTJ2K encode benchmark rows. Must be at least 10. | 10 | Benchmark |
| `J2K_SAMPLED_CORPUS` | Directory containing `level0-tile0.j2k` through `level2-tile7.j2k` for the opt-in sampled-color Metal characterization test. Requires an optimized build and Metal hardware. | Not set | Benchmark |
| `J2K_METAL_DECODE_INPUT_DIRS` | Optional path-list of external `.j2k`, `.j2c`, `.jp2`, `.jph`, or `.jhc` fixtures included in the Metal decode benchmark. Wrapper containers are emitted as structured skips until wrapper-specific strict Metal parity is claimed. | Not set | Benchmark |
| `J2K_METAL_DECODE_MANIFEST` | Optional TSV manifest for Metal decode external fixtures. Uses pinned `path` and `input_fnv1a64`; optional `codec` and `container` labels are recorded in benchmark rows. | Not set | Benchmark |
| `J2K_METAL_DECODE_INCLUDE_GENERATED` | Set to `0`, `false`, `no`, or `off` to omit generated Metal decode smoke fixtures when external fixtures are provided. Publication gates require this to be false for Metal decode speed claims. | Generated Metal decode fixtures included | Benchmark |
| `J2K_METAL_CAPTURE_PATH` | Absolute, nonexistent `.gputrace` output path for the ignored Metal GPU-capture diagnostic. Requires `MTL_CAPTURE_ENABLED=1`; captured timings are diagnostic only. | Not set | Benchmark |
| `J2K_METAL_ENCODE_INPUT_DIRS` | Optional path-list of staged external `.pgm`, `.ppm`, or `.pnm` source images included in the Metal auto-routing encode benchmark. Use the same canonical PNM source assets as `J2K_ENCODE_COMPARE_INPUT_DIRS` after staging. | Not set | Benchmark |
| `J2K_METAL_ENCODE_MANIFEST` | Optional TSV manifest for Metal encode staged PNM sources. Uses `path` and pinned `input_fnv1a64` from `J2K_ENCODE_COMPARE_MANIFEST`. | Not set | Benchmark |
| `J2K_METAL_ENCODE_INCLUDE_GENERATED` | Set to `0`, `false`, `no`, or `off` to omit generated Metal host-input auto-routing rows when external staged PNM sources are provided. Stage microbenchmarks remain generated component rows. | Generated Metal host-input rows included | Benchmark |
| `J2K_METAL_ENCODE_RESIDENT_MAX_ESTIMATED_OUTPUT_BYTES` | Maximum raw-byte estimate allowed for a Metal resident host-output encode benchmark row before the row is emitted as a structured memory-budget skip. This protects huge batches from materializing multi-gigabyte host codestream outputs in one benchmark process. | `2147483648` | Benchmark |
| `J2K_ADOPTION_FIXTURES` | Required repository variable or environment override for the manual GPU benchmark workflow decode fixture directory path-list passed to `cargo run -p xtask --features adoption -- adoption-benchmark --fixtures`. | None; adoption runs fail closed when unset | Benchmark/CI |
| `J2K_ADOPTION_MANIFEST` | Required repository variable or environment override for the manual GPU benchmark workflow decode fixture manifest passed to `--manifest`. | None; adoption runs fail closed when unset | Benchmark/CI |
| `J2K_ADOPTION_ENCODE_FIXTURES` | Required repository variable or environment override for the manual GPU benchmark workflow staged PNM encode fixture directory path-list passed to `--encode-fixtures`. | None; adoption runs fail closed when unset | Benchmark/CI |
| `J2K_ADOPTION_ENCODE_MANIFEST` | Required repository variable or environment override for the manual GPU benchmark workflow staged PNM encode manifest passed to `--encode-manifest`. | None; adoption runs fail closed when unset | Benchmark/CI |
| `J2K_AUTO_ROUTING_MANIFEST` | Path to the hash-pinned schema-v1 JSON workload manifest used by the CUDA and Metal Auto-routing Criterion benches. | None; routing runs fail closed when unset | Benchmark/CI |
| `J2K_AUTO_ROUTING_ROOT` | Corpus root against which every relative path in `J2K_AUTO_ROUTING_MANIFEST` is resolved and hash-checked. | None; routing runs fail closed when unset | Benchmark/CI |
| `J2K_AUTO_ROUTING_EVIDENCE` | Output path for raw route-parity evidence written beside Criterion estimates. | None; routing runs fail closed when unset | Benchmark/CI |
| `J2K_AUTO_ROUTING_CANDIDATE_SHA` | Exact lowercase 40-hex commit identity recorded in route evidence. | None; routing runs fail closed when unset | Benchmark/CI |
| `J2K_AUTO_ROUTING_HARDWARE` | Non-empty accelerator hardware identity recorded in route evidence. | None; routing runs fail closed when unset | Benchmark/CI |
| `J2K_AUTO_ROUTING_DRIVER` | Non-empty driver/toolchain identity recorded in route evidence. | None; routing runs fail closed when unset | Benchmark/CI |
| `J2K_CUDA_PROFILE_BATCH_SIZE` | Batch size for the CUDA HTJ2K decode profile example. | Example default | Benchmark |
| `J2K_CUDA_PROFILE_ITERATIONS` | Iteration count for the CUDA HTJ2K decode profile example. | Example default | Benchmark |

## Xtask And Release Tooling

| Variable | Effect | Default | Stability |
| --- | --- | --- | --- |
| `J2K_COVERAGE_BASE` | Git revision used by `cargo xtask coverage` to compute changed executable Rust lines. CI pins this to the pull-request base SHA, the pre-push SHA, or the v0.6.2 release baseline for accelerator release evidence. | `HEAD^` only for local or scheduled fallback runs | Test/CI |
| `J2K_FUZZ_RUNS` | Number of runs passed to each `cargo xtask fuzz-run` target. | `1000` | Test/CI |
| `J2K_FUZZ_MAX_TOTAL_TIME_SECONDS` | Optional libFuzzer max total time for `cargo xtask fuzz-run`. | Not passed | Test/CI |
| `J2K_FUZZ_TARGET` | Target triple passed to `cargo fuzz run --target` by `cargo xtask fuzz-run`. | Nightly host target from `rustc -vV` | Test/CI |
| `J2K_SEMVER_TOOLCHAIN` | Rejected by `cargo xtask semver`; Rust `1.96` is pinned in source and CI. | Must not be set | Test/CI |

CI overrides the fuzz defaults to 512 runs / 60 seconds for pull requests and
20,000 runs / 900 seconds for the scheduled long fuzz job.
