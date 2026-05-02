# JPEG 2000 Metal Encode Stages Design

## Goal
Add encode-side acceleration boundaries for JPEG 2000 lossless output so Signinum can use separate forward RCT, forward 5/3 DWT, Tier-1, and packetization work instead of reusing decode-oriented Metal kernels.

## Architecture
The pure-Rust encoder remains the correctness baseline. `signinum-j2k-native` exposes hidden encode-stage hooks that can accelerate forward RCT, forward 5/3 DWT, Tier-1 code-block encoding, and packetization while preserving CPU fallback for every stage. `signinum-j2k-metal` implements device-side stages behind those hooks and only becomes selectable by higher-level APIs after parity and round-trip validation pass.

## Components
- `signinum-j2k-native`: owns the JPEG 2000 encode graph, stage traits, CPU fallback, and codestream writer.
- `signinum-j2k-metal`: owns Metal buffers, kernels, and host wrappers for supported encode stages.
- `signinum-j2k`: keeps the public lossless encode API and treats CPU output as the validation oracle.

## Data Flow
Interleaved samples are deinterleaved into component planes. RGB images may run a forward RCT stage. Each component then runs a reversible 5/3 DWT stage. Quantized subbands are split into code-blocks and sent through Tier-1 encode. Packetization remains host-side first because LRCP packet headers, tag trees, and byte lengths are serial and inexpensive relative to transform and Tier-1 work.

## Error Handling
Unsupported device shapes return `Ok(false)` from stage hooks and fall back to CPU. Strict device requests only succeed when all required encode stages are device-backed and validated. Kernel errors are surfaced as explicit unsupported or kernel errors; no silent CPU fallback is used for `RequireDevice`.

## Testing
Tests cover hook dispatch, CPU fallback equivalence, lossless round-trip for RGB/grayscale 8/16-bit, odd dimensions, and macOS-only Metal parity where kernels are present. Public `RequireDevice` remains unsupported until the complete device encode path is available.
