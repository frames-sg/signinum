// SPDX-License-Identifier: MIT OR Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};
use j2k_core::{
    BackendRequest, DeviceSubmission, Downscale, ImageDecodeSubmit, PixelFormat, Rect,
    TileBatchDecodeSubmit,
};
use j2k_jpeg::{
    adapter::{
        build_fast420_packet, build_fast422_packet, build_fast444_packet, decoder_bytes,
        summarize_device_batch,
    },
    decode_tile_region_scaled_into_in_context, decode_tile_scaled_into_in_context, DecodeRequest,
    Decoder as CpuDecoder, DecoderContext as JpegDecoderContext, ScratchPool as CpuScratchPool,
};
use j2k_jpeg_metal::{
    decode_viewport_to_surface, suggest_viewport_workload, Codec, Decoder, MetalDecodeRequest,
    MetalSession, ScratchPool, ViewportTile, ViewportWorkload,
};
#[cfg(target_os = "macos")]
use j2k_jpeg_metal::{
    MetalBackendSession, MetalBatchOutputBuffer, MetalBatchTextureOutput, MetalBufferBatchTarget,
    Rgb8MetalBatchOp, Rgb8MetalBatchRequest, Rgb8MetalBatchSource, SurfaceResidency,
};
use jpeg_encoder::{ColorType, Encoder, SamplingFactor};
use std::collections::HashSet;

#[path = "support/bench_inputs.rs"]
mod bench_inputs;
use bench_inputs::{BenchInput, CorpusInputClass, DecodeMode};

#[cfg(target_os = "macos")]
const RESIDENT_TEXTURE_BATCH_SIZES: [usize; 3] = [16, 64, 256];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DeviceBatchKey {
    dimensions: (u32, u32),
    restart_interval: Option<u16>,
    checkpoint_count: usize,
    matches_fast_420: bool,
    matches_fast_422: bool,
    matches_fast_444: bool,
}

struct DistinctTileBatch<'a> {
    name: String,
    coalesce_hit_rate: String,
    tiles: Vec<&'a BenchInput>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FastPacketPlan {
    matches_fast_420: bool,
    matches_fast_422: bool,
    matches_fast_444: bool,
}

impl FastPacketPlan {
    fn has_fast_packet(self) -> bool {
        self.matches_fast_420 || self.matches_fast_422 || self.matches_fast_444
    }
}

fn load_bench_inputs() -> Vec<BenchInput> {
    let inputs = vec![
        BenchInput {
            name: "repo/baseline_420_16x16".to_string(),
            bytes: include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg").to_vec(),
            dimensions: (16, 16),
            mode: DecodeMode::Rgb,
            input_class: CorpusInputClass::BoundedFullFrame,
        },
        BenchInput {
            name: "repo/baseline_422_16x8".to_string(),
            bytes: include_bytes!("../fixtures/jpeg/baseline_422_16x8.jpg").to_vec(),
            dimensions: (16, 8),
            mode: DecodeMode::Rgb,
            input_class: CorpusInputClass::BoundedFullFrame,
        },
        BenchInput {
            name: "repo/baseline_444_8x8".to_string(),
            bytes: include_bytes!("../fixtures/jpeg/baseline_444_8x8.jpg").to_vec(),
            dimensions: (8, 8),
            mode: DecodeMode::Rgb,
            input_class: CorpusInputClass::BoundedFullFrame,
        },
        BenchInput {
            name: "repo/grayscale_8x8".to_string(),
            bytes: include_bytes!("../fixtures/jpeg/grayscale_8x8.jpg").to_vec(),
            dimensions: (8, 8),
            mode: DecodeMode::Gray,
            input_class: CorpusInputClass::BoundedFullFrame,
        },
        BenchInput {
            name: "generated/fast420_256x256".to_string(),
            bytes: generated_rgb_jpeg(256, 256, SamplingFactor::F_2_2, None),
            dimensions: (256, 256),
            mode: DecodeMode::Rgb,
            input_class: CorpusInputClass::BoundedFullFrame,
        },
        BenchInput {
            name: "generated/fast420_restart2_256x256".to_string(),
            bytes: generated_rgb_jpeg(256, 256, SamplingFactor::F_2_2, Some(2)),
            dimensions: (256, 256),
            mode: DecodeMode::Rgb,
            input_class: CorpusInputClass::BoundedFullFrame,
        },
        BenchInput {
            name: "generated/fast422_256x256".to_string(),
            bytes: generated_rgb_jpeg(256, 256, SamplingFactor::F_2_1, None),
            dimensions: (256, 256),
            mode: DecodeMode::Rgb,
            input_class: CorpusInputClass::BoundedFullFrame,
        },
        BenchInput {
            name: "generated/fast444_256x256".to_string(),
            bytes: generated_rgb_jpeg(256, 256, SamplingFactor::F_1_1, None),
            dimensions: (256, 256),
            mode: DecodeMode::Rgb,
            input_class: CorpusInputClass::BoundedFullFrame,
        },
    ];
    bench_inputs::load_bench_inputs(inputs)
}

fn generated_rgb_jpeg(
    width: u16,
    height: u16,
    sampling: SamplingFactor,
    restart_interval: Option<u16>,
) -> Vec<u8> {
    generated_rgb_jpeg_variant(width, height, sampling, restart_interval, 0)
}

fn generated_rgb_jpeg_variant(
    width: u16,
    height: u16,
    sampling: SamplingFactor,
    restart_interval: Option<u16>,
    variant: u8,
) -> Vec<u8> {
    let mut rgb = j2k_test_support::gpu_bench_rgb8(u32::from(width), u32::from(height));
    if variant != 0 {
        for (index, sample) in rgb.iter_mut().enumerate() {
            let position = u8::try_from(index % 251).expect("generated JPEG position fits u8");
            *sample = sample.wrapping_add(position.wrapping_mul(variant));
        }
    }

    let mut jpeg = Vec::new();
    let mut encoder = Encoder::new(&mut jpeg, 90);
    encoder.set_sampling_factor(sampling);
    if let Some(interval) = restart_interval {
        encoder.set_restart_interval(interval);
    }
    encoder
        .encode(&rgb, width, height, ColorType::Rgb)
        .expect("encode generated benchmark JPEG");
    jpeg
}

fn parent_name(name: &str) -> &str {
    name.rsplit_once('/').map_or("repo", |(parent, _)| parent)
}

fn display_parent_name(parent: &str) -> &str {
    parent
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or(parent)
}

fn device_batch_key(input: &BenchInput) -> Option<DeviceBatchKey> {
    let decoder = CpuDecoder::new(&input.bytes).ok()?;
    let summary = summarize_device_batch(&decoder, 4);
    Some(DeviceBatchKey {
        dimensions: input.dimensions,
        restart_interval: summary.restart_interval,
        checkpoint_count: summary.checkpoint_count,
        matches_fast_420: summary.matches_fast_420,
        matches_fast_422: summary.matches_fast_422,
        matches_fast_444: summary.matches_fast_444,
    })
}

fn fast_packet_plan(bytes: &[u8]) -> Option<FastPacketPlan> {
    let decoder = CpuDecoder::new(bytes).ok()?;
    let bytes = decoder_bytes(&decoder);
    Some(FastPacketPlan {
        matches_fast_444: build_fast444_packet(bytes).is_ok(),
        matches_fast_422: build_fast422_packet(bytes).is_ok(),
        matches_fast_420: build_fast420_packet(bytes).is_ok(),
    })
}

fn fast_packet_family_label(plan: FastPacketPlan) -> &'static str {
    if plan.matches_fast_420 {
        "fast420"
    } else if plan.matches_fast_422 {
        "fast422"
    } else if plan.matches_fast_444 {
        "fast444"
    } else {
        "no_fast_packet"
    }
}

fn checkpoint_label(key: DeviceBatchKey) -> &'static str {
    if key.restart_interval.is_some() {
        "restart"
    } else if key.checkpoint_count > 0 {
        "checkpointed"
    } else {
        "unchunked"
    }
}

fn fast_packet_scope_label(input: &BenchInput) -> String {
    let Some(plan) = fast_packet_plan(&input.bytes) else {
        return "reject/not_jpeg".to_string();
    };
    let checkpoint = device_batch_key(input).map_or("unknown", checkpoint_label);

    let family = fast_packet_family_label(plan);
    let disposition = if plan.has_fast_packet() {
        "accept"
    } else {
        "reject"
    };
    format!("{disposition}/{family}/{checkpoint}")
}

fn bench_fast_packet_planning(c: &mut Criterion, inputs: &[BenchInput]) {
    let mut group = c.benchmark_group("jpeg_metal_fast_packet_planning");
    for input in inputs
        .iter()
        .filter(|input| input.input_class == CorpusInputClass::BoundedFullFrame)
    {
        let scope = fast_packet_scope_label(input);
        group.bench_function(format!("{scope}/{}", input.name), |b| {
            b.iter(|| {
                std::hint::black_box(
                    fast_packet_plan(std::hint::black_box(input.bytes.as_slice()))
                        .expect("fast packet plan"),
                );
            });
        });
    }
    group.finish();
}

fn digest_bytes(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
    let mut hash = FNV_OFFSET;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn coalesce_hit_rate_label(hit_count: usize, total_count: usize) -> String {
    let tenths = hit_count
        .saturating_mul(1000)
        .checked_div(total_count)
        .unwrap_or(0);
    format!("coalesce_hits_{}p{}pct", tenths / 10, tenths % 10)
}

fn duplicate_hit_count(tiles: &[&BenchInput]) -> usize {
    let mut seen = HashSet::with_capacity(tiles.len());
    tiles
        .iter()
        .filter(|tile| !seen.insert((tile.bytes.len(), digest_bytes(&tile.bytes))))
        .count()
}

fn distinct_region_scaled_batches<'a>(
    inputs: &'a [BenchInput],
    batch_size: usize,
    side: u32,
) -> Vec<DistinctTileBatch<'a>> {
    let mut groups: Vec<(String, DeviceBatchKey, Vec<&'a BenchInput>)> = Vec::new();
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.dimensions.0 >= side && input.dimensions.1 >= side
    }) {
        let Some(key) = device_batch_key(input) else {
            continue;
        };
        let parent = parent_name(&input.name).to_string();
        if let Some((_, _, tiles)) = groups
            .iter_mut()
            .find(|(group_parent, group_key, _)| *group_parent == parent && *group_key == key)
        {
            tiles.push(input);
        } else {
            groups.push((parent, key, vec![input]));
        }
    }

    groups
        .into_iter()
        .filter_map(|(parent, key, tiles)| {
            if tiles.len() < batch_size {
                return None;
            }
            let tiles = tiles.into_iter().take(batch_size).collect::<Vec<_>>();
            Some(DistinctTileBatch {
                name: format!(
                    "{}/{}x{}/distinct_{}_of_{}",
                    display_parent_name(&parent),
                    key.dimensions.0,
                    key.dimensions.1,
                    batch_size,
                    batch_size
                ),
                coalesce_hit_rate: coalesce_hit_rate_label(
                    duplicate_hit_count(&tiles),
                    tiles.len(),
                ),
                tiles,
            })
        })
        .collect()
}

fn centered_roi((width, height): (u32, u32), side: u32) -> Rect {
    let w = side.min(width);
    let h = side.min(height);
    Rect {
        x: (width - w) / 2,
        y: (height - h) / 2,
        w,
        h,
    }
}

fn to_jpeg_rect(rect: Rect) -> j2k_jpeg::Rect {
    j2k_jpeg::Rect {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: rect.h,
    }
}

fn cpu_decode_tile_batch(bytes: &[u8], batch_size: usize) {
    let mut ctx = JpegDecoderContext::new();
    let mut pool = CpuScratchPool::new();
    let mut out = Vec::new();
    for _ in 0..batch_size {
        let decoder = CpuDecoder::from_view_in_context(
            j2k_jpeg::JpegView::parse(bytes).expect("view"),
            &mut ctx,
        )
        .expect("decoder");
        let dims = decoder.info().dimensions;
        let stride = dims.0 as usize * 3;
        out.resize(stride * dims.1 as usize, 0);
        decoder
            .decode_into_with_scratch(&mut pool, &mut out, stride, PixelFormat::Rgb8)
            .expect("cpu tile batch decode");
    }
    std::hint::black_box(out);
}

fn scaled_rect(rect: Rect, scale: Downscale) -> Rect {
    let denom = scale.denominator();
    let x_end = rect.x + rect.w;
    let y_end = rect.y + rect.h;
    let x0 = rect.x / denom;
    let y0 = rect.y / denom;
    let x1 = x_end.div_ceil(denom);
    let y1 = y_end.div_ceil(denom);
    Rect {
        x: x0,
        y: y0,
        w: x1.saturating_sub(x0),
        h: y1.saturating_sub(y0),
    }
}

fn cpu_decode_full(bytes: &[u8]) {
    let decoder = CpuDecoder::new(bytes).expect("cpu decoder");
    let dims = decoder.info().dimensions;
    let stride = dims.0 as usize * 3;
    let mut out = vec![0u8; stride * dims.1 as usize];
    decoder
        .decode_into_with_scratch(
            &mut CpuScratchPool::new(),
            &mut out,
            stride,
            PixelFormat::Rgb8,
        )
        .expect("cpu full decode");
    std::hint::black_box(out);
}

fn metal_decode_full(bytes: &[u8]) {
    let mut decoder = Decoder::new(bytes).expect("metal decoder");
    let mut session = MetalSession::default();
    let submission = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut decoder,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Metal,
    )
    .expect("full submit");
    std::hint::black_box(submission.wait().expect("surface"));
}

fn cpu_decode_region(bytes: &[u8], side: u32) {
    let decoder = CpuDecoder::new(bytes).expect("cpu decoder");
    let roi = centered_roi(decoder.info().dimensions, side);
    let (out, _) = decoder
        .decode_request(DecodeRequest::region(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
        ))
        .expect("cpu region decode");
    std::hint::black_box(out);
}

fn cpu_decode_region_scaled(bytes: &[u8], side: u32, factor: Downscale) {
    let decoder = CpuDecoder::new(bytes).expect("cpu decoder");
    let roi = centered_roi(decoder.info().dimensions, side);
    let (out, _) = decoder
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            to_jpeg_rect(roi),
            factor,
        ))
        .expect("cpu region scaled decode");
    std::hint::black_box(out);
}

fn cpu_decode_scaled(bytes: &[u8], factor: Downscale) {
    let decoder = CpuDecoder::new(bytes).expect("cpu decoder");
    let (out, _) = decoder
        .decode_request(DecodeRequest::scaled(PixelFormat::Rgb8, factor))
        .expect("cpu scaled decode");
    std::hint::black_box(out);
}

fn cpu_decode_tile_batch_scaled(bytes: &[u8], batch_size: usize, factor: Downscale) {
    let decoder = CpuDecoder::new(bytes).expect("cpu decoder");
    let dims = decoder.info().dimensions;
    let out_width = dims.0.div_ceil(factor.denominator());
    let out_height = dims.1.div_ceil(factor.denominator());
    let stride = out_width as usize * 3;
    let mut out = vec![0u8; stride * out_height as usize];
    let mut ctx = JpegDecoderContext::new();
    let mut pool = CpuScratchPool::new();
    for _ in 0..batch_size {
        decode_tile_scaled_into_in_context(
            bytes,
            &mut ctx,
            &mut pool,
            j2k_jpeg::TileDecodeOutput {
                out: &mut out,
                stride,
                fmt: PixelFormat::Rgb8,
            },
            factor,
        )
        .expect("cpu scaled tile batch");
    }
    std::hint::black_box(out);
}

fn cpu_decode_tile_batch_region_scaled(
    bytes: &[u8],
    batch_size: usize,
    side: u32,
    factor: Downscale,
) {
    let decoder = CpuDecoder::new(bytes).expect("cpu decoder");
    let roi = centered_roi(decoder.info().dimensions, side);
    let scaled = scaled_rect(roi, factor);
    let stride = scaled.w as usize * 3;
    let mut out = vec![0u8; stride * scaled.h as usize];
    let mut ctx = JpegDecoderContext::new();
    let mut pool = CpuScratchPool::new();
    for _ in 0..batch_size {
        decode_tile_region_scaled_into_in_context(
            bytes,
            &mut ctx,
            &mut pool,
            j2k_jpeg::TileDecodeOutput {
                out: &mut out,
                stride,
                fmt: PixelFormat::Rgb8,
            },
            to_jpeg_rect(roi),
            factor,
        )
        .expect("cpu region scaled tile batch");
    }
    std::hint::black_box(out);
}

fn cpu_decode_distinct_tile_batch_region_scaled(
    tiles: &[&BenchInput],
    side: u32,
    factor: Downscale,
) {
    let mut ctx = JpegDecoderContext::new();
    let mut pool = CpuScratchPool::new();
    let mut out = Vec::new();
    for tile in tiles {
        let roi = centered_roi(tile.dimensions, side);
        let scaled = scaled_rect(roi, factor);
        let stride = scaled.w as usize * 3;
        out.resize(stride * scaled.h as usize, 0);
        decode_tile_region_scaled_into_in_context(
            &tile.bytes,
            &mut ctx,
            &mut pool,
            j2k_jpeg::TileDecodeOutput {
                out: &mut out,
                stride,
                fmt: PixelFormat::Rgb8,
            },
            to_jpeg_rect(roi),
            factor,
        )
        .expect("cpu distinct region scaled tile batch");
        std::hint::black_box(out.as_slice());
    }
    std::hint::black_box(out);
}

fn metal_decode_tile_batch(bytes: &[u8], batch_size: usize) {
    device_decode_tile_batch(bytes, batch_size, BackendRequest::Metal);
}

#[cfg(target_os = "macos")]
fn supports_resident_texture_batch(input: &BenchInput) -> bool {
    device_batch_key(input)
        .is_some_and(|key| key.matches_fast_420 || key.matches_fast_422 || key.matches_fast_444)
}

#[cfg(target_os = "macos")]
fn bench_resident_texture_batches(c: &mut Criterion, inputs: &[BenchInput], has_metal: bool) {
    if !has_metal {
        return;
    }

    let mut group = c.benchmark_group("wsi_tile_batch_rgba_textures");
    for &batch_size in &RESIDENT_TEXTURE_BATCH_SIZES {
        for input in inputs
            .iter()
            .filter(|input| input.mode == DecodeMode::Rgb && supports_resident_texture_batch(input))
        {
            let session = MetalBackendSession::system_default().expect("Metal backend session");
            let mut output =
                MetalBatchTextureOutput::new_rgba8_tiles(&session, input.dimensions, batch_size)
                    .expect("texture output");
            let decoders = (0..batch_size)
                .map(|_| Decoder::new(input.bytes.as_slice()).expect("metal decoder"))
                .collect::<Vec<_>>();
            group.bench_function(
                format!(
                    "batch{batch_size}/warm_session_reused_textures/{}",
                    input.name
                ),
                move |b| {
                    let decoder_refs = decoders.iter().collect::<Vec<_>>();
                    b.iter(|| {
                        let tiles = Codec::decode_rgb8_decoder_batch_into_resizable_metal_textures_with_session(
                            &decoder_refs,
                            &mut output,
                            &session,
                        )
                        .expect("resident texture batch decode");
                        std::hint::black_box(tiles);
                    });
                },
            );
        }
    }
    group.finish();
}

#[cfg(not(target_os = "macos"))]
fn bench_resident_texture_batches(_c: &mut Criterion, _inputs: &[BenchInput], _has_metal: bool) {}

#[cfg(target_os = "macos")]
fn bench_resident_viewport_outputs(_c: &mut Criterion, _inputs: &[BenchInput], _has_metal: bool) {}

#[cfg(not(target_os = "macos"))]
fn bench_resident_viewport_outputs(_c: &mut Criterion, _inputs: &[BenchInput], _has_metal: bool) {}

fn auto_decode_tile_batch(bytes: &[u8], batch_size: usize) {
    device_decode_tile_batch(bytes, batch_size, BackendRequest::Auto);
}

fn device_decode_tile_batch(bytes: &[u8], batch_size: usize, backend: BackendRequest) {
    let mut ctx = JpegDecoderContext::default();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();
    let submissions = (0..batch_size)
        .map(|_| {
            <Codec as TileBatchDecodeSubmit>::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                bytes,
                PixelFormat::Rgb8,
                backend,
            )
            .expect("submit")
        })
        .collect::<Vec<_>>();
    for submission in submissions {
        std::hint::black_box(submission.wait().expect("surface"));
    }
}

fn metal_decode_tile_batch_scaled(bytes: &[u8], batch_size: usize, factor: Downscale) {
    device_decode_tile_batch_scaled(bytes, batch_size, factor, BackendRequest::Metal);
}

fn auto_decode_tile_batch_scaled(bytes: &[u8], batch_size: usize, factor: Downscale) {
    device_decode_tile_batch_scaled(bytes, batch_size, factor, BackendRequest::Auto);
}

fn device_decode_tile_batch_scaled(
    bytes: &[u8],
    batch_size: usize,
    factor: Downscale,
    backend: BackendRequest,
) {
    let mut ctx = JpegDecoderContext::default();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();
    let submissions = (0..batch_size)
        .map(|_| {
            <Codec as TileBatchDecodeSubmit>::submit_tile_scaled_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                bytes,
                PixelFormat::Rgb8,
                factor,
                backend,
            )
            .expect("scaled submit")
        })
        .collect::<Vec<_>>();
    for submission in submissions {
        std::hint::black_box(submission.wait().expect("surface"));
    }
}

fn metal_decode_tile_batch_region_scaled(
    bytes: &[u8],
    batch_size: usize,
    side: u32,
    factor: Downscale,
) {
    let cpu = CpuDecoder::new(bytes).expect("cpu decoder");
    let roi = centered_roi(cpu.info().dimensions, side);
    let mut ctx = JpegDecoderContext::default();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();
    let submissions = (0..batch_size)
        .map(|_| {
            Codec::submit_tile_request_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                bytes,
                MetalDecodeRequest::region_scaled(
                    PixelFormat::Rgb8,
                    roi,
                    factor,
                    BackendRequest::Metal,
                ),
            )
            .expect("region scaled submit")
        })
        .collect::<Vec<_>>();
    for submission in submissions {
        std::hint::black_box(submission.wait().expect("surface"));
    }
    assert_eq!(
        session.submissions().expect("session submissions"),
        1,
        "coalesced region+scaled tile batch should flush once"
    );
    std::hint::black_box(session.submissions().expect("session submissions"));
}

fn metal_decode_distinct_tile_batch_region_scaled(
    tiles: &[&BenchInput],
    side: u32,
    factor: Downscale,
) {
    let mut ctx = JpegDecoderContext::default();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();
    let submissions = tiles
        .iter()
        .map(|tile| {
            let roi = centered_roi(tile.dimensions, side);
            Codec::submit_tile_request_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                &tile.bytes,
                MetalDecodeRequest::region_scaled(
                    PixelFormat::Rgb8,
                    roi,
                    factor,
                    BackendRequest::Metal,
                ),
            )
            .expect("distinct region scaled submit")
        })
        .collect::<Vec<_>>();
    for submission in submissions {
        std::hint::black_box(submission.wait().expect("surface"));
    }
    std::hint::black_box(session.submissions().expect("session submissions"));
}

fn metal_decode_region(bytes: &[u8], side: u32) {
    let cpu = CpuDecoder::new(bytes).expect("cpu decoder");
    let roi = centered_roi(cpu.info().dimensions, side);
    let mut decoder = Decoder::new(bytes).expect("metal decoder");
    let mut session = MetalSession::default();
    let submission = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_region_to_device(
        &mut decoder,
        &mut session,
        PixelFormat::Rgb8,
        roi,
        BackendRequest::Metal,
    )
    .expect("region submit");
    std::hint::black_box(submission.wait().expect("surface"));
}

fn metal_decode_region_scaled(bytes: &[u8], side: u32, factor: Downscale) {
    let cpu = CpuDecoder::new(bytes).expect("cpu decoder");
    let roi = centered_roi(cpu.info().dimensions, side);
    let mut decoder = Decoder::new(bytes).expect("metal decoder");
    let surface = decoder
        .decode_request_to_device(MetalDecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            roi,
            factor,
            BackendRequest::Metal,
        ))
        .expect("region scaled surface");
    std::hint::black_box(surface);
}

fn metal_decode_scaled(bytes: &[u8], factor: Downscale) {
    let mut decoder = Decoder::new(bytes).expect("metal decoder");
    let mut session = MetalSession::default();
    let submission = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_scaled_to_device(
        &mut decoder,
        &mut session,
        PixelFormat::Rgb8,
        factor,
        BackendRequest::Metal,
    )
    .expect("scaled submit");
    std::hint::black_box(submission.wait().expect("surface"));
}

fn cpu_viewport_composite(bytes: &[u8], dimensions: (u32, u32)) {
    public_viewport_surface(bytes, dimensions, BackendRequest::Cpu);
}

fn hybrid_viewport_composite(bytes: &[u8], dimensions: (u32, u32)) {
    public_viewport_surface(bytes, dimensions, BackendRequest::Auto);
}

fn cpu_viewport_composite_device(bytes: &[u8], dimensions: (u32, u32)) {
    public_viewport_surface(bytes, dimensions, BackendRequest::Cpu);
}

fn hybrid_viewport_composite_device(bytes: &[u8], dimensions: (u32, u32)) {
    public_viewport_surface(bytes, dimensions, BackendRequest::Auto);
}

fn public_viewport_surface(bytes: &[u8], dimensions: (u32, u32), backend: BackendRequest) {
    let Some(workload) = suggest_viewport_workload(dimensions) else {
        return;
    };
    let decoder = CpuDecoder::new(bytes).expect("cpu decoder");
    let mut pool = CpuScratchPool::new();
    let surface =
        decode_viewport_to_surface(&decoder, &mut pool, &workload, backend).expect("viewport");
    std::hint::black_box(surface);
}

fn scheduled_viewport_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    workload: &ViewportWorkload,
    backend: BackendRequest,
) {
    let surface = decode_viewport_to_surface(decoder, pool, workload, backend).expect("viewport");
    std::hint::black_box(surface);
}

fn sparse_viewport_workload(workload: &ViewportWorkload) -> Option<ViewportWorkload> {
    let first = *workload.tiles.first()?;
    let last = *workload.tiles.last()?;
    Some(ViewportWorkload {
        scale: workload.scale,
        viewport_dims: workload.viewport_dims,
        tiles: vec![
            ViewportTile {
                source_roi: first.source_roi,
                dest: first.dest,
            },
            ViewportTile {
                source_roi: last.source_roi,
                dest: last.dest,
            },
        ],
    })
}

fn metal_available() -> bool {
    let required = std::env::var_os("J2K_REQUIRE_METAL_BENCH").is_some();
    #[cfg(target_os = "macos")]
    {
        let available = j2k_metal_support::system_default_device().is_ok();
        assert!(
            !required || available,
            "J2K_REQUIRE_METAL_BENCH is set but no default Metal device is available"
        );
        available
    }
    #[cfg(not(target_os = "macos"))]
    {
        assert!(
            !required,
            "J2K_REQUIRE_METAL_BENCH is set but this is not a Metal host"
        );
        false
    }
}

fn bench_full_and_tile_decode_groups(c: &mut Criterion, inputs: &[BenchInput], has_metal: bool) {
    let mut decode_rgb = c.benchmark_group("decode_rgb");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        decode_rgb.bench_function(format!("{}/cpu", input.name), |b| {
            b.iter(|| cpu_decode_full(&input.bytes));
        });
        if has_metal {
            decode_rgb.bench_function(format!("{}/metal", input.name), |b| {
                b.iter(|| metal_decode_full(&input.bytes));
            });
        }
    }
    decode_rgb.finish();

    let mut wsi_tile_batch_rgb = c.benchmark_group("wsi_tile_batch_rgb");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb) {
        wsi_tile_batch_rgb.bench_function(format!("{}/cpu", input.name), |b| {
            b.iter(|| cpu_decode_tile_batch(&input.bytes, 64));
        });
        if has_metal {
            wsi_tile_batch_rgb.bench_function(format!("{}/metal", input.name), |b| {
                b.iter(|| metal_decode_tile_batch(&input.bytes, 64));
            });
        }
        wsi_tile_batch_rgb.bench_function(format!("{}/auto", input.name), |b| {
            b.iter(|| auto_decode_tile_batch(&input.bytes, 64));
        });
    }
    wsi_tile_batch_rgb.finish();
}

#[cfg(target_os = "macos")]
fn pixel_format_label(fmt: PixelFormat) -> &'static str {
    match fmt {
        PixelFormat::Gray8 => "gray8",
        PixelFormat::Rgb8 => "rgb8",
        PixelFormat::Rgba8 => "rgba8",
        _ => "other",
    }
}

#[cfg(target_os = "macos")]
fn native_request_pixels(bytes: &[u8], request: DecodeRequest) -> Vec<u8> {
    CpuDecoder::new(bytes)
        .expect("native retained-session benchmark decoder")
        .decode_request(request)
        .expect("native retained-session benchmark decode")
        .0
}

#[cfg(target_os = "macos")]
fn assert_metal_surface_pixels(surface: &j2k_jpeg_metal::Surface, expected: &[u8]) {
    assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
    let actual = surface.as_bytes().expect("Metal benchmark surface bytes");
    let actual = actual.as_ref();
    if actual != expected {
        let first_mismatch = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual != expected);
        let mismatch_count = actual
            .iter()
            .zip(expected)
            .filter(|(actual, expected)| actual != expected)
            .count()
            + actual.len().abs_diff(expected.len());
        let max_delta = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| actual.abs_diff(*expected))
            .max()
            .unwrap_or(0);
        panic!(
            "Metal benchmark pixels differ: actual_len={} expected_len={} first_mismatch={first_mismatch:?} mismatch_count={mismatch_count} max_delta={max_delta}",
            actual.len(),
            expected.len(),
        );
    }
}

#[cfg(target_os = "macos")]
fn bench_retained_full_rows(c: &mut Criterion) {
    let fixtures = [
        (
            "420_nr",
            generated_rgb_jpeg(256, 256, SamplingFactor::F_2_2, None),
            "fast420",
        ),
        (
            "420_r2",
            generated_rgb_jpeg(256, 256, SamplingFactor::F_2_2, Some(2)),
            "fast420",
        ),
        (
            "422_nr",
            generated_rgb_jpeg(256, 256, SamplingFactor::F_2_1, None),
            "fast422",
        ),
        (
            "444_nr",
            generated_rgb_jpeg(256, 256, SamplingFactor::F_1_1, None),
            "fast444",
        ),
    ];
    let mut group = c.benchmark_group("jpeg_metal_single_retained");
    for (fixture_name, bytes, expected_family) in &fixtures {
        let packet = fast_packet_plan(bytes).expect("retained-session fast packet");
        assert_eq!(fast_packet_family_label(packet), *expected_family);
        for fmt in [PixelFormat::Gray8, PixelFormat::Rgb8, PixelFormat::Rgba8] {
            let expected = native_request_pixels(bytes, DecodeRequest::full(fmt));
            let session = MetalBackendSession::system_default().expect("Metal backend session");
            let mut decoder = Decoder::new(bytes).expect("retained Metal decoder");
            let probe = decoder
                .decode_to_device_with_session(fmt, &session)
                .expect("retained-session Metal probe");
            assert_metal_surface_pixels(&probe, &expected);
            drop(probe);

            group.bench_function(
                format!("{fixture_name}/full/{}", pixel_format_label(fmt)),
                |b| {
                    b.iter(|| {
                        let surface = decoder
                            .decode_to_device_with_session(fmt, &session)
                            .expect("retained-session Metal decode");
                        std::hint::black_box(surface);
                    });
                },
            );
        }
    }
    group.finish();
}

#[cfg(target_os = "macos")]
fn bench_single_request_controls(c: &mut Criterion) {
    let scaled_bytes = generated_rgb_jpeg(256, 256, SamplingFactor::F_2_2, Some(2));
    let region_bytes = generated_rgb_jpeg(256, 256, SamplingFactor::F_1_1, None);
    let controls = [
        (
            "420_r2",
            "scaled_q4",
            scaled_bytes.as_slice(),
            MetalDecodeRequest::scaled(
                PixelFormat::Rgb8,
                Downscale::Quarter,
                BackendRequest::Metal,
            ),
            DecodeRequest::scaled(PixelFormat::Rgb8, Downscale::Quarter),
        ),
        (
            "444_nr",
            "region192_q4",
            region_bytes.as_slice(),
            MetalDecodeRequest::region_scaled(
                PixelFormat::Rgb8,
                centered_roi((256, 256), 192),
                Downscale::Quarter,
                BackendRequest::Metal,
            ),
            DecodeRequest::region_scaled(
                PixelFormat::Rgb8,
                to_jpeg_rect(centered_roi((256, 256), 192)),
                Downscale::Quarter,
            ),
        ),
    ];
    let mut group = c.benchmark_group("jpeg_metal_single_request_controls");
    for (fixture_name, name, bytes, metal_request, native_request) in controls {
        let expected = native_request_pixels(bytes, native_request);
        let mut decoder = Decoder::new(bytes).expect("retained request-control decoder");
        let probe = decoder
            .decode_request_to_device(metal_request)
            .expect("global-runtime request control probe");
        assert_metal_surface_pixels(&probe, &expected);
        drop(probe);
        group.bench_function(
            format!("{fixture_name}/global_runtime_control/{name}"),
            |b| {
                b.iter(|| {
                    let surface = decoder
                        .decode_request_to_device(metal_request)
                        .expect("global-runtime request control");
                    std::hint::black_box(surface);
                });
            },
        );
    }
    group.finish();
}

#[cfg(target_os = "macos")]
fn bench_retained_distinct_batch(c: &mut Criterion) {
    let distinct_bytes = (1..=4)
        .map(|variant| {
            generated_rgb_jpeg_variant(256, 256, SamplingFactor::F_2_2, Some(2), variant)
        })
        .collect::<Vec<_>>();
    for bytes in &distinct_bytes {
        let packet = fast_packet_plan(bytes).expect("distinct batch fast packet");
        assert_eq!(fast_packet_family_label(packet), "fast420");
    }
    let expected = distinct_bytes
        .iter()
        .map(|bytes| native_request_pixels(bytes, DecodeRequest::full(PixelFormat::Rgb8)))
        .collect::<Vec<_>>();
    for first in 0..distinct_bytes.len() {
        for second in first + 1..distinct_bytes.len() {
            assert_ne!(
                distinct_bytes[first], distinct_bytes[second],
                "distinct batch JPEG inputs {first} and {second}"
            );
            assert_ne!(
                expected[first], expected[second],
                "distinct batch pixels {first} and {second}"
            );
        }
    }
    let decoders = distinct_bytes
        .iter()
        .map(|bytes| Decoder::new(bytes).expect("distinct retained Metal decoder"))
        .collect::<Vec<_>>();
    let decoder_refs = decoders.iter().collect::<Vec<_>>();
    let session = MetalBackendSession::system_default().expect("distinct batch Metal session");
    let output = MetalBatchOutputBuffer::new_rgb8_tiles(&session, (256, 256), decoder_refs.len())
        .expect("distinct batch output");
    let decode_batch = || {
        Codec::decode_rgb8_batch_into_buffer_with_session(
            Rgb8MetalBatchRequest {
                source: Rgb8MetalBatchSource::Decoders(&decoder_refs),
                op: Rgb8MetalBatchOp::Full,
            },
            MetalBufferBatchTarget::Reusable(&output),
            &session,
        )
        .expect("distinct retained-session batch")
    };
    let probe = decode_batch();
    assert_eq!(probe.len(), expected.len());
    for (surface, expected) in probe.into_iter().zip(&expected) {
        assert_metal_surface_pixels(&surface.expect("distinct batch surface"), expected);
    }
    let mut batch_group = c.benchmark_group("jpeg_metal_single_retained_batch_control");
    batch_group.bench_function("420_r2/distinct4/rgb8", |b| {
        b.iter(|| std::hint::black_box(decode_batch()));
    });
    batch_group.finish();
}

#[cfg(target_os = "macos")]
fn bench_retained_single_decode(c: &mut Criterion, has_metal: bool) {
    if !has_metal {
        return;
    }
    bench_retained_full_rows(c);
    bench_single_request_controls(c);
    bench_retained_distinct_batch(c);
}

#[cfg(not(target_os = "macos"))]
fn bench_retained_single_decode(_c: &mut Criterion, _has_metal: bool) {}

fn bench_region_and_scaled_decode_groups(
    c: &mut Criterion,
    inputs: &[BenchInput],
    has_metal: bool,
) {
    let mut wsi_region_rgb = c.benchmark_group("wsi_region_rgb");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        wsi_region_rgb.bench_function(format!("{}/cpu", input.name), |b| {
            b.iter(|| cpu_decode_region(&input.bytes, 256));
        });
        if has_metal {
            wsi_region_rgb.bench_function(format!("{}/metal", input.name), |b| {
                b.iter(|| metal_decode_region(&input.bytes, 256));
            });
        }
    }
    wsi_region_rgb.finish();

    let mut wsi_scaled_rgb_q4 = c.benchmark_group("wsi_scaled_rgb_q4");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        wsi_scaled_rgb_q4.bench_function(format!("{}/cpu", input.name), |b| {
            b.iter(|| cpu_decode_scaled(&input.bytes, Downscale::Quarter));
        });
        if has_metal {
            wsi_scaled_rgb_q4.bench_function(format!("{}/metal", input.name), |b| {
                b.iter(|| metal_decode_scaled(&input.bytes, Downscale::Quarter));
            });
        }
    }
    wsi_scaled_rgb_q4.finish();

    let mut wsi_scaled_rgb_q8 = c.benchmark_group("wsi_scaled_rgb_q8");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        wsi_scaled_rgb_q8.bench_function(format!("{}/cpu", input.name), |b| {
            b.iter(|| cpu_decode_scaled(&input.bytes, Downscale::Eighth));
        });
        if has_metal {
            wsi_scaled_rgb_q8.bench_function(format!("{}/metal", input.name), |b| {
                b.iter(|| metal_decode_scaled(&input.bytes, Downscale::Eighth));
            });
        }
    }
    wsi_scaled_rgb_q8.finish();

    let mut wsi_region_scaled_rgb_q4 = c.benchmark_group("wsi_region_scaled_rgb_q4");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        wsi_region_scaled_rgb_q4.bench_function(format!("{}/cpu", input.name), |b| {
            b.iter(|| cpu_decode_region_scaled(&input.bytes, 256, Downscale::Quarter));
        });
        if has_metal {
            wsi_region_scaled_rgb_q4.bench_function(format!("{}/metal", input.name), |b| {
                b.iter(|| metal_decode_region_scaled(&input.bytes, 256, Downscale::Quarter));
            });
        }
    }
    wsi_region_scaled_rgb_q4.finish();

    let mut wsi_region_scaled_rgb_q8 = c.benchmark_group("wsi_region_scaled_rgb_q8");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        wsi_region_scaled_rgb_q8.bench_function(format!("{}/cpu", input.name), |b| {
            b.iter(|| cpu_decode_region_scaled(&input.bytes, 256, Downscale::Eighth));
        });
        if has_metal {
            wsi_region_scaled_rgb_q8.bench_function(format!("{}/metal", input.name), |b| {
                b.iter(|| metal_decode_region_scaled(&input.bytes, 256, Downscale::Eighth));
            });
        }
    }
    wsi_region_scaled_rgb_q8.finish();
}

fn bench_scaled_tile_batch_groups(
    c: &mut Criterion,
    inputs: &[BenchInput],
    distinct_batches: &[DistinctTileBatch<'_>],
    coalesced_hit_rate: &str,
    has_metal: bool,
) {
    let mut wsi_tile_batch_scaled_rgb_q4 = c.benchmark_group("wsi_tile_batch_scaled_rgb_q4");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb) {
        wsi_tile_batch_scaled_rgb_q4.bench_function(format!("{}/cpu", input.name), |b| {
            b.iter(|| cpu_decode_tile_batch_scaled(&input.bytes, 64, Downscale::Quarter));
        });
        if has_metal {
            wsi_tile_batch_scaled_rgb_q4.bench_function(format!("{}/metal", input.name), |b| {
                b.iter(|| metal_decode_tile_batch_scaled(&input.bytes, 64, Downscale::Quarter));
            });
        }
        wsi_tile_batch_scaled_rgb_q4.bench_function(format!("{}/auto", input.name), |b| {
            b.iter(|| auto_decode_tile_batch_scaled(&input.bytes, 64, Downscale::Quarter));
        });
    }
    wsi_tile_batch_scaled_rgb_q4.finish();

    let mut wsi_tile_batch_region_scaled_coalesced_rgb_q4 =
        c.benchmark_group("wsi_tile_batch_region_scaled_coalesced_rgb_q4");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb) {
        wsi_tile_batch_region_scaled_coalesced_rgb_q4.bench_function(
            format!("coalesce_all/{coalesced_hit_rate}/cpu/{}", input.name),
            |b| {
                b.iter(|| {
                    cpu_decode_tile_batch_region_scaled(&input.bytes, 64, 256, Downscale::Quarter);
                });
            },
        );
        if has_metal {
            wsi_tile_batch_region_scaled_coalesced_rgb_q4.bench_function(
                format!("coalesce_all/{coalesced_hit_rate}/metal/{}", input.name),
                |b| {
                    b.iter(|| {
                        metal_decode_tile_batch_region_scaled(
                            &input.bytes,
                            64,
                            256,
                            Downscale::Quarter,
                        );
                    });
                },
            );
        }
    }
    wsi_tile_batch_region_scaled_coalesced_rgb_q4.finish();

    let mut wsi_tile_batch_region_scaled_distinct_rgb_q4 =
        c.benchmark_group("wsi_tile_batch_region_scaled_distinct_rgb_q4");
    for batch in distinct_batches {
        wsi_tile_batch_region_scaled_distinct_rgb_q4.bench_function(
            format!(
                "coalesce_none/{}/cpu/{}",
                batch.coalesce_hit_rate, batch.name
            ),
            |b| {
                b.iter(|| {
                    cpu_decode_distinct_tile_batch_region_scaled(
                        &batch.tiles,
                        256,
                        Downscale::Quarter,
                    );
                });
            },
        );
        if has_metal {
            wsi_tile_batch_region_scaled_distinct_rgb_q4.bench_function(
                format!(
                    "coalesce_none/{}/metal/{}",
                    batch.coalesce_hit_rate, batch.name
                ),
                |b| {
                    b.iter(|| {
                        metal_decode_distinct_tile_batch_region_scaled(
                            &batch.tiles,
                            256,
                            Downscale::Quarter,
                        );
                    });
                },
            );
        }
    }
    wsi_tile_batch_region_scaled_distinct_rgb_q4.finish();
}

fn bench_cold_viewport_groups(c: &mut Criterion, inputs: &[BenchInput], has_metal: bool) {
    let mut viewer_region_scaled_composite_rgb =
        c.benchmark_group("viewer_region_scaled_composite_rgb");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb
            && input.input_class == CorpusInputClass::BoundedFullFrame
            && suggest_viewport_workload(input.dimensions).is_some()
    }) {
        viewer_region_scaled_composite_rgb.bench_function(format!("{}/cpu", input.name), |b| {
            b.iter(|| cpu_viewport_composite(&input.bytes, input.dimensions));
        });
        if has_metal {
            viewer_region_scaled_composite_rgb.bench_function(
                format!("{}/hybrid", input.name),
                |b| {
                    b.iter(|| hybrid_viewport_composite(&input.bytes, input.dimensions));
                },
            );
        }
    }
    viewer_region_scaled_composite_rgb.finish();

    let mut viewer_region_scaled_composite_rgb_device =
        c.benchmark_group("viewer_region_scaled_composite_rgb_device");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb
            && input.input_class == CorpusInputClass::BoundedFullFrame
            && suggest_viewport_workload(input.dimensions).is_some()
    }) {
        viewer_region_scaled_composite_rgb_device.bench_function(
            format!("{}/cpu", input.name),
            |b| {
                b.iter(|| cpu_viewport_composite_device(&input.bytes, input.dimensions));
            },
        );
        if has_metal {
            viewer_region_scaled_composite_rgb_device.bench_function(
                format!("{}/hybrid", input.name),
                |b| {
                    b.iter(|| hybrid_viewport_composite_device(&input.bytes, input.dimensions));
                },
            );
        }
    }
    viewer_region_scaled_composite_rgb_device.finish();
}

fn bench_warm_viewport_group(c: &mut Criterion, inputs: &[BenchInput], has_metal: bool) {
    let mut viewer_region_scaled_composite_rgb_warm =
        c.benchmark_group("viewer_region_scaled_composite_rgb_warm");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb
            && input.input_class == CorpusInputClass::BoundedFullFrame
            && suggest_viewport_workload(input.dimensions).is_some()
    }) {
        let workload = suggest_viewport_workload(input.dimensions).expect("warm workload");
        let cpu_bytes = input.bytes.clone();
        viewer_region_scaled_composite_rgb_warm.bench_function(
            format!("{}/cpu", input.name),
            move |b| {
                let decoder = CpuDecoder::new(&cpu_bytes).expect("cpu decoder");
                let mut pool = CpuScratchPool::new();
                b.iter(|| {
                    let surface = decode_viewport_to_surface(
                        &decoder,
                        &mut pool,
                        &workload,
                        BackendRequest::Cpu,
                    )
                    .expect("cpu warm viewport");
                    std::hint::black_box(surface);
                });
            },
        );

        if has_metal {
            let workload = suggest_viewport_workload(input.dimensions).expect("warm workload");
            let hybrid_bytes = input.bytes.clone();
            viewer_region_scaled_composite_rgb_warm.bench_function(
                format!("{}/hybrid", input.name),
                move |b| {
                    let decoder = CpuDecoder::new(&hybrid_bytes).expect("cpu decoder");
                    let mut pool = CpuScratchPool::new();
                    b.iter(|| {
                        let surface = decode_viewport_to_surface(
                            &decoder,
                            &mut pool,
                            &workload,
                            BackendRequest::Auto,
                        )
                        .expect("hybrid warm viewport");
                        std::hint::black_box(surface);
                    });
                },
            );
        }
    }
    viewer_region_scaled_composite_rgb_warm.finish();
}

fn bench_warm_viewport_device_group(c: &mut Criterion, inputs: &[BenchInput], has_metal: bool) {
    let mut viewer_region_scaled_composite_rgb_device_warm =
        c.benchmark_group("viewer_region_scaled_composite_rgb_device_warm");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb
            && input.input_class == CorpusInputClass::BoundedFullFrame
            && suggest_viewport_workload(input.dimensions).is_some()
    }) {
        let workload = suggest_viewport_workload(input.dimensions).expect("warm workload");
        let cpu_bytes = input.bytes.clone();
        viewer_region_scaled_composite_rgb_device_warm.bench_function(
            format!("{}/cpu", input.name),
            move |b| {
                let decoder = CpuDecoder::new(&cpu_bytes).expect("cpu decoder");
                let mut pool = CpuScratchPool::new();
                b.iter(|| {
                    let surface = decode_viewport_to_surface(
                        &decoder,
                        &mut pool,
                        &workload,
                        BackendRequest::Cpu,
                    )
                    .expect("cpu warm viewport surface");
                    std::hint::black_box(surface);
                });
            },
        );

        if has_metal {
            let workload = suggest_viewport_workload(input.dimensions).expect("warm workload");
            let hybrid_bytes = input.bytes.clone();
            viewer_region_scaled_composite_rgb_device_warm.bench_function(
                format!("{}/hybrid", input.name),
                move |b| {
                    let decoder = CpuDecoder::new(&hybrid_bytes).expect("cpu decoder");
                    let mut pool = CpuScratchPool::new();
                    b.iter(|| {
                        let surface = decode_viewport_to_surface(
                            &decoder,
                            &mut pool,
                            &workload,
                            BackendRequest::Auto,
                        )
                        .expect("hybrid warm viewport surface");
                        std::hint::black_box(surface);
                    });
                },
            );
        }
    }
    viewer_region_scaled_composite_rgb_device_warm.finish();
}

fn bench_contiguous_viewport_groups(c: &mut Criterion, inputs: &[BenchInput], has_metal: bool) {
    let mut viewer_contiguous_region_scaled_rgb =
        c.benchmark_group("viewer_contiguous_region_scaled_rgb");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb
            && input.input_class == CorpusInputClass::BoundedFullFrame
            && suggest_viewport_workload(input.dimensions).is_some()
    }) {
        let workload = suggest_viewport_workload(input.dimensions).expect("viewport workload");
        viewer_contiguous_region_scaled_rgb.bench_function(format!("{}/cpu", input.name), |b| {
            let decoder = CpuDecoder::new(&input.bytes).expect("cpu decoder");
            let mut pool = CpuScratchPool::new();
            b.iter(|| {
                let surface =
                    decode_viewport_to_surface(&decoder, &mut pool, &workload, BackendRequest::Cpu)
                        .expect("cpu contiguous viewport");
                std::hint::black_box(surface);
            });
        });
        if has_metal {
            viewer_contiguous_region_scaled_rgb.bench_function(
                format!("{}/hybrid", input.name),
                |b| {
                    let decoder = CpuDecoder::new(&input.bytes).expect("cpu decoder");
                    let mut pool = CpuScratchPool::new();
                    b.iter(|| {
                        let surface = decode_viewport_to_surface(
                            &decoder,
                            &mut pool,
                            &workload,
                            BackendRequest::Auto,
                        )
                        .expect("hybrid contiguous viewport");
                        std::hint::black_box(surface);
                    });
                },
            );
        }
    }
    viewer_contiguous_region_scaled_rgb.finish();

    let mut viewer_contiguous_region_scaled_rgb_device =
        c.benchmark_group("viewer_contiguous_region_scaled_rgb_device");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb
            && input.input_class == CorpusInputClass::BoundedFullFrame
            && suggest_viewport_workload(input.dimensions).is_some()
    }) {
        let workload = suggest_viewport_workload(input.dimensions).expect("viewport workload");
        viewer_contiguous_region_scaled_rgb_device.bench_function(
            format!("{}/cpu", input.name),
            |b| {
                let decoder = CpuDecoder::new(&input.bytes).expect("cpu decoder");
                let mut pool = CpuScratchPool::new();
                b.iter(|| {
                    let surface = decode_viewport_to_surface(
                        &decoder,
                        &mut pool,
                        &workload,
                        BackendRequest::Cpu,
                    )
                    .expect("cpu contiguous upload");
                    std::hint::black_box(surface);
                });
            },
        );
        if has_metal {
            viewer_contiguous_region_scaled_rgb_device.bench_function(
                format!("{}/hybrid", input.name),
                |b| {
                    let decoder = CpuDecoder::new(&input.bytes).expect("cpu decoder");
                    let mut pool = CpuScratchPool::new();
                    b.iter(|| {
                        let surface = decode_viewport_to_surface(
                            &decoder,
                            &mut pool,
                            &workload,
                            BackendRequest::Auto,
                        )
                        .expect("hybrid contiguous viewport");
                        std::hint::black_box(surface);
                    });
                },
            );
        }
    }
    viewer_contiguous_region_scaled_rgb_device.finish();
}

fn bench_adaptive_viewport_groups(c: &mut Criterion, inputs: &[BenchInput]) {
    let mut viewer_best_region_scaled_rgb_device =
        c.benchmark_group("viewer_best_region_scaled_rgb_device");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb
            && input.input_class == CorpusInputClass::BoundedFullFrame
            && suggest_viewport_workload(input.dimensions).is_some()
    }) {
        let workload = suggest_viewport_workload(input.dimensions).expect("viewport workload");
        viewer_best_region_scaled_rgb_device.bench_function(
            format!("{}/cpu_only", input.name),
            |b| {
                let decoder = CpuDecoder::new(&input.bytes).expect("cpu decoder");
                let mut pool = CpuScratchPool::new();
                b.iter(|| {
                    scheduled_viewport_surface(&decoder, &mut pool, &workload, BackendRequest::Cpu);
                });
            },
        );
        let workload = suggest_viewport_workload(input.dimensions).expect("viewport workload");
        viewer_best_region_scaled_rgb_device.bench_function(
            format!("{}/adaptive", input.name),
            |b| {
                let decoder = CpuDecoder::new(&input.bytes).expect("cpu decoder");
                let mut pool = CpuScratchPool::new();
                b.iter(|| {
                    scheduled_viewport_surface(
                        &decoder,
                        &mut pool,
                        &workload,
                        BackendRequest::Auto,
                    );
                });
            },
        );
    }
    viewer_best_region_scaled_rgb_device.finish();

    let mut viewer_best_region_scaled_composite_rgb_device =
        c.benchmark_group("viewer_best_region_scaled_composite_rgb_device");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb
            && input.input_class == CorpusInputClass::BoundedFullFrame
            && suggest_viewport_workload(input.dimensions)
                .and_then(|workload| sparse_viewport_workload(&workload))
                .is_some()
    }) {
        let workload = sparse_viewport_workload(
            &suggest_viewport_workload(input.dimensions).expect("viewport workload"),
        )
        .expect("sparse workload");
        viewer_best_region_scaled_composite_rgb_device.bench_function(
            format!("{}/cpu_only", input.name),
            |b| {
                let decoder = CpuDecoder::new(&input.bytes).expect("cpu decoder");
                let mut pool = CpuScratchPool::new();
                b.iter(|| {
                    scheduled_viewport_surface(&decoder, &mut pool, &workload, BackendRequest::Cpu);
                });
            },
        );
        let workload = sparse_viewport_workload(
            &suggest_viewport_workload(input.dimensions).expect("viewport workload"),
        )
        .expect("sparse workload");
        viewer_best_region_scaled_composite_rgb_device.bench_function(
            format!("{}/adaptive", input.name),
            |b| {
                let decoder = CpuDecoder::new(&input.bytes).expect("cpu decoder");
                let mut pool = CpuScratchPool::new();
                b.iter(|| {
                    scheduled_viewport_surface(
                        &decoder,
                        &mut pool,
                        &workload,
                        BackendRequest::Auto,
                    );
                });
            },
        );
    }
    viewer_best_region_scaled_composite_rgb_device.finish();
}

fn bench_compare(c: &mut Criterion) {
    let inputs = load_bench_inputs();
    let distinct_batches = distinct_region_scaled_batches(&inputs, 64, 256);
    let coalesced_hit_rate = coalesce_hit_rate_label(63, 64);
    let has_metal = metal_available();

    bench_fast_packet_planning(c, &inputs);
    bench_full_and_tile_decode_groups(c, &inputs, has_metal);
    bench_retained_single_decode(c, has_metal);
    bench_resident_texture_batches(c, &inputs, has_metal);
    bench_resident_viewport_outputs(c, &inputs, has_metal);
    bench_region_and_scaled_decode_groups(c, &inputs, has_metal);
    bench_scaled_tile_batch_groups(
        c,
        &inputs,
        &distinct_batches,
        &coalesced_hit_rate,
        has_metal,
    );
    bench_cold_viewport_groups(c, &inputs, has_metal);
    bench_warm_viewport_group(c, &inputs, has_metal);
    bench_warm_viewport_device_group(c, &inputs, has_metal);
    bench_contiguous_viewport_groups(c, &inputs, has_metal);
    bench_adaptive_viewport_groups(c, &inputs);
}

criterion_group!(benches, bench_compare);
criterion_main!(benches);
