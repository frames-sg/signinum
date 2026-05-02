// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};
use signinum_core::{
    BackendRequest, DecoderContext, DeviceSubmission, Downscale, ImageDecodeSubmit, PixelFormat,
    Rect, TileBatchDecodeSubmit,
};
use signinum_jpeg::{
    adapter::summarize_device_batch, decode_tile_region_scaled_into_in_context,
    decode_tile_scaled_into_in_context, Decoder as CpuDecoder,
    DecoderContext as JpegDecoderContext, ScratchPool as CpuScratchPool,
};
use signinum_jpeg_metal::viewport::{
    compose_viewport_cpu, compose_viewport_cpu_to_surface, compose_viewport_hybrid,
    decode_viewport_region_cpu, decode_viewport_region_cpu_to_surface,
    decode_viewport_region_hybrid, decode_viewport_to_surface, suggest_viewport_workload,
    ViewportTile, ViewportWorkload,
};
use signinum_jpeg_metal::{Codec, Decoder, MetalSession, ScratchPool};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

const FULL_FRAME_MAX_OUTPUT_BYTES: usize = 512 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodeMode {
    Gray,
    Rgb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CorpusInputClass {
    BoundedFullFrame,
    VeryLarge,
}

#[derive(Clone)]
struct BenchInput {
    name: String,
    bytes: Vec<u8>,
    dimensions: (u32, u32),
    mode: DecodeMode,
    input_class: CorpusInputClass,
}

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

fn load_bench_inputs() -> Vec<BenchInput> {
    let mut inputs = vec![
        BenchInput {
            name: "repo/baseline_420_16x16".to_string(),
            bytes: include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg").to_vec(),
            dimensions: (16, 16),
            mode: DecodeMode::Rgb,
            input_class: CorpusInputClass::BoundedFullFrame,
        },
        BenchInput {
            name: "repo/grayscale_8x8".to_string(),
            bytes: include_bytes!("../../../corpus/conformance/grayscale_8x8.jpg").to_vec(),
            dimensions: (8, 8),
            mode: DecodeMode::Gray,
            input_class: CorpusInputClass::BoundedFullFrame,
        },
    ];

    let mut seen = inputs
        .iter()
        .map(|input| input.name.clone())
        .collect::<Vec<_>>();
    for path in
        std::env::split_paths(&std::env::var_os("SIGNINUM_BENCH_INPUTS").unwrap_or_default())
    {
        collect_jpegs(&path, &mut inputs, &mut seen);
    }
    inputs.sort_by(|a, b| a.name.cmp(&b.name));
    inputs
}

fn collect_jpegs(path: &Path, inputs: &mut Vec<BenchInput>, seen: &mut Vec<String>) {
    if path.is_file() {
        push_jpeg(path, inputs, seen);
        return;
    }
    if !path.is_dir() {
        return;
    }

    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let child = entry.path();
            if child.is_dir() {
                stack.push(child);
            } else {
                push_jpeg(&child, inputs, seen);
            }
        }
    }
}

fn push_jpeg(path: &Path, inputs: &mut Vec<BenchInput>, seen: &mut Vec<String>) {
    if !is_jpeg(path) {
        return;
    }
    let Ok(bytes) = fs::read(path) else {
        return;
    };
    let Ok(decoder) = CpuDecoder::new(&bytes) else {
        return;
    };
    let Some(mode) = color_space_mode(decoder.info().color_space) else {
        return;
    };
    let name = relative_name(path);
    if seen.contains(&name) {
        return;
    }

    seen.push(name.clone());
    let dimensions = decoder.info().dimensions;
    inputs.push(BenchInput {
        name,
        bytes,
        dimensions,
        mode,
        input_class: classify_corpus_input(dimensions, mode),
    });
}

fn is_jpeg(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg"))
}

fn relative_name(path: &Path) -> String {
    let absolute = path.canonicalize().unwrap_or_else(|_| PathBuf::from(path));
    if let Some(prefix) = std::env::var_os("HOME") {
        let prefix = PathBuf::from(prefix);
        if let Ok(stripped) = absolute.strip_prefix(prefix) {
            return stripped.display().to_string();
        }
    }
    absolute.display().to_string()
}

fn color_space_mode(color_space: signinum_jpeg::ColorSpace) -> Option<DecodeMode> {
    match color_space {
        signinum_jpeg::ColorSpace::Grayscale => Some(DecodeMode::Gray),
        signinum_jpeg::ColorSpace::YCbCr | signinum_jpeg::ColorSpace::Rgb => Some(DecodeMode::Rgb),
        signinum_jpeg::ColorSpace::Cmyk | signinum_jpeg::ColorSpace::Ycck => None,
    }
}

fn classify_corpus_input(dimensions: (u32, u32), mode: DecodeMode) -> CorpusInputClass {
    let bpp = match mode {
        DecodeMode::Gray => 1usize,
        DecodeMode::Rgb => 3usize,
    };
    let bytes = usize::try_from(dimensions.0)
        .ok()
        .and_then(|width| {
            usize::try_from(dimensions.1)
                .ok()
                .map(|height| (width, height))
        })
        .and_then(|(width, height)| width.checked_mul(height))
        .and_then(|pixels| pixels.checked_mul(bpp));
    match bytes {
        Some(bytes) if bytes <= FULL_FRAME_MAX_OUTPUT_BYTES => CorpusInputClass::BoundedFullFrame,
        _ => CorpusInputClass::VeryLarge,
    }
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
    let tenths = if total_count == 0 {
        0
    } else {
        hit_count.saturating_mul(1000) / total_count
    };
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

fn to_jpeg_rect(rect: Rect) -> signinum_jpeg::Rect {
    signinum_jpeg::Rect {
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
            signinum_jpeg::JpegView::parse(bytes).expect("view"),
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
        .decode_region(
            PixelFormat::Rgb8,
            signinum_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
        )
        .expect("cpu region decode");
    std::hint::black_box(out);
}

fn cpu_decode_region_scaled(bytes: &[u8], side: u32, factor: Downscale) {
    let decoder = CpuDecoder::new(bytes).expect("cpu decoder");
    let roi = centered_roi(decoder.info().dimensions, side);
    let (out, _) = decoder
        .decode_region_scaled(PixelFormat::Rgb8, to_jpeg_rect(roi), factor)
        .expect("cpu region scaled decode");
    std::hint::black_box(out);
}

fn cpu_decode_scaled(bytes: &[u8], factor: Downscale) {
    let decoder = CpuDecoder::new(bytes).expect("cpu decoder");
    let (out, _) = decoder
        .decode_scaled(PixelFormat::Rgb8, factor)
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
            &mut out,
            stride,
            PixelFormat::Rgb8,
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
            &mut out,
            stride,
            PixelFormat::Rgb8,
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
            &mut out,
            stride,
            PixelFormat::Rgb8,
            to_jpeg_rect(roi),
            factor,
        )
        .expect("cpu distinct region scaled tile batch");
        std::hint::black_box(out.as_slice());
    }
    std::hint::black_box(out);
}

fn metal_decode_tile_batch(bytes: &[u8], batch_size: usize) {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
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
                BackendRequest::Metal,
            )
            .expect("submit")
        })
        .collect::<Vec<_>>();
    for submission in submissions {
        std::hint::black_box(submission.wait().expect("surface"));
    }
}

fn metal_decode_tile_batch_scaled(bytes: &[u8], batch_size: usize, factor: Downscale) {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
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
                BackendRequest::Metal,
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
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();
    let submissions = (0..batch_size)
        .map(|_| {
            Codec::submit_tile_region_scaled_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                bytes,
                PixelFormat::Rgb8,
                roi,
                factor,
                BackendRequest::Metal,
            )
            .expect("region scaled submit")
        })
        .collect::<Vec<_>>();
    for submission in submissions {
        std::hint::black_box(submission.wait().expect("surface"));
    }
    assert_eq!(
        session.submissions(),
        1,
        "coalesced region+scaled tile batch should flush once"
    );
    std::hint::black_box(session.submissions());
}

fn metal_decode_distinct_tile_batch_region_scaled(
    tiles: &[&BenchInput],
    side: u32,
    factor: Downscale,
) {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();
    let submissions = tiles
        .iter()
        .map(|tile| {
            let roi = centered_roi(tile.dimensions, side);
            Codec::submit_tile_region_scaled_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                &tile.bytes,
                PixelFormat::Rgb8,
                roi,
                factor,
                BackendRequest::Metal,
            )
            .expect("distinct region scaled submit")
        })
        .collect::<Vec<_>>();
    for submission in submissions {
        std::hint::black_box(submission.wait().expect("surface"));
    }
    std::hint::black_box(session.submissions());
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
        .decode_region_scaled_to_device(PixelFormat::Rgb8, roi, factor, BackendRequest::Metal)
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
    let Some(workload) = suggest_viewport_workload(dimensions) else {
        return;
    };
    let decoder = CpuDecoder::new(bytes).expect("cpu decoder");
    let mut pool = CpuScratchPool::new();
    let out = compose_viewport_cpu(
        &decoder,
        &mut pool,
        PixelFormat::Rgb8,
        workload.scale,
        workload.viewport_dims,
        &workload.tiles,
    )
    .expect("cpu viewport");
    std::hint::black_box(out);
}

fn hybrid_viewport_composite(bytes: &[u8], dimensions: (u32, u32)) {
    let Some(workload) = suggest_viewport_workload(dimensions) else {
        return;
    };
    let decoder = CpuDecoder::new(bytes).expect("cpu decoder");
    let mut pool = CpuScratchPool::new();
    let surface = compose_viewport_hybrid(
        &decoder,
        &mut pool,
        workload.scale,
        workload.viewport_dims,
        &workload.tiles,
    )
    .expect("hybrid viewport");
    let stride = workload.viewport_dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut out = vec![0u8; stride * workload.viewport_dims.1 as usize];
    surface
        .download_into(&mut out, stride)
        .expect("hybrid viewport download");
    std::hint::black_box(out);
}

fn cpu_viewport_composite_device(bytes: &[u8], dimensions: (u32, u32)) {
    let Some(workload) = suggest_viewport_workload(dimensions) else {
        return;
    };
    let decoder = CpuDecoder::new(bytes).expect("cpu decoder");
    let mut pool = CpuScratchPool::new();
    let surface = compose_viewport_cpu_to_surface(
        &decoder,
        &mut pool,
        workload.scale,
        workload.viewport_dims,
        &workload.tiles,
    )
    .expect("cpu viewport surface");
    std::hint::black_box(surface);
}

fn hybrid_viewport_composite_device(bytes: &[u8], dimensions: (u32, u32)) {
    let Some(workload) = suggest_viewport_workload(dimensions) else {
        return;
    };
    let decoder = CpuDecoder::new(bytes).expect("cpu decoder");
    let mut pool = CpuScratchPool::new();
    let surface = compose_viewport_hybrid(
        &decoder,
        &mut pool,
        workload.scale,
        workload.viewport_dims,
        &workload.tiles,
    )
    .expect("hybrid viewport surface");
    std::hint::black_box(surface);
}

fn scheduled_viewport_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    workload: &signinum_jpeg_metal::viewport::ViewportWorkload,
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

fn bench_compare(c: &mut Criterion) {
    let inputs = load_bench_inputs();
    let distinct_batches = distinct_region_scaled_batches(&inputs, 64, 256);
    let coalesced_hit_rate = coalesce_hit_rate_label(63, 64);

    let mut decode_rgb = c.benchmark_group("decode_rgb");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        decode_rgb.bench_function(format!("{}/cpu", input.name), |b| {
            b.iter(|| cpu_decode_full(&input.bytes));
        });
        decode_rgb.bench_function(format!("{}/metal", input.name), |b| {
            b.iter(|| metal_decode_full(&input.bytes));
        });
    }
    decode_rgb.finish();

    let mut wsi_tile_batch_rgb = c.benchmark_group("wsi_tile_batch_rgb");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb) {
        wsi_tile_batch_rgb.bench_function(format!("{}/cpu", input.name), |b| {
            b.iter(|| cpu_decode_tile_batch(&input.bytes, 64));
        });
        wsi_tile_batch_rgb.bench_function(format!("{}/metal", input.name), |b| {
            b.iter(|| metal_decode_tile_batch(&input.bytes, 64));
        });
    }
    wsi_tile_batch_rgb.finish();

    let mut wsi_region_rgb = c.benchmark_group("wsi_region_rgb");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        wsi_region_rgb.bench_function(format!("{}/cpu", input.name), |b| {
            b.iter(|| cpu_decode_region(&input.bytes, 256));
        });
        wsi_region_rgb.bench_function(format!("{}/metal", input.name), |b| {
            b.iter(|| metal_decode_region(&input.bytes, 256));
        });
    }
    wsi_region_rgb.finish();

    let mut wsi_scaled_rgb_q4 = c.benchmark_group("wsi_scaled_rgb_q4");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        wsi_scaled_rgb_q4.bench_function(format!("{}/cpu", input.name), |b| {
            b.iter(|| cpu_decode_scaled(&input.bytes, Downscale::Quarter));
        });
        wsi_scaled_rgb_q4.bench_function(format!("{}/metal", input.name), |b| {
            b.iter(|| metal_decode_scaled(&input.bytes, Downscale::Quarter));
        });
    }
    wsi_scaled_rgb_q4.finish();

    let mut wsi_scaled_rgb_q8 = c.benchmark_group("wsi_scaled_rgb_q8");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        wsi_scaled_rgb_q8.bench_function(format!("{}/cpu", input.name), |b| {
            b.iter(|| cpu_decode_scaled(&input.bytes, Downscale::Eighth));
        });
        wsi_scaled_rgb_q8.bench_function(format!("{}/metal", input.name), |b| {
            b.iter(|| metal_decode_scaled(&input.bytes, Downscale::Eighth));
        });
    }
    wsi_scaled_rgb_q8.finish();

    let mut wsi_region_scaled_rgb_q4 = c.benchmark_group("wsi_region_scaled_rgb_q4");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        wsi_region_scaled_rgb_q4.bench_function(format!("{}/cpu", input.name), |b| {
            b.iter(|| cpu_decode_region_scaled(&input.bytes, 256, Downscale::Quarter));
        });
        wsi_region_scaled_rgb_q4.bench_function(format!("{}/metal", input.name), |b| {
            b.iter(|| metal_decode_region_scaled(&input.bytes, 256, Downscale::Quarter));
        });
    }
    wsi_region_scaled_rgb_q4.finish();

    let mut wsi_region_scaled_rgb_q8 = c.benchmark_group("wsi_region_scaled_rgb_q8");
    for input in inputs.iter().filter(|input| {
        input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
    }) {
        wsi_region_scaled_rgb_q8.bench_function(format!("{}/cpu", input.name), |b| {
            b.iter(|| cpu_decode_region_scaled(&input.bytes, 256, Downscale::Eighth));
        });
        wsi_region_scaled_rgb_q8.bench_function(format!("{}/metal", input.name), |b| {
            b.iter(|| metal_decode_region_scaled(&input.bytes, 256, Downscale::Eighth));
        });
    }
    wsi_region_scaled_rgb_q8.finish();

    let mut wsi_tile_batch_scaled_rgb_q4 = c.benchmark_group("wsi_tile_batch_scaled_rgb_q4");
    for input in inputs.iter().filter(|input| input.mode == DecodeMode::Rgb) {
        wsi_tile_batch_scaled_rgb_q4.bench_function(format!("{}/cpu", input.name), |b| {
            b.iter(|| cpu_decode_tile_batch_scaled(&input.bytes, 64, Downscale::Quarter));
        });
        wsi_tile_batch_scaled_rgb_q4.bench_function(format!("{}/metal", input.name), |b| {
            b.iter(|| metal_decode_tile_batch_scaled(&input.bytes, 64, Downscale::Quarter));
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
    wsi_tile_batch_region_scaled_coalesced_rgb_q4.finish();

    let mut wsi_tile_batch_region_scaled_distinct_rgb_q4 =
        c.benchmark_group("wsi_tile_batch_region_scaled_distinct_rgb_q4");
    for batch in &distinct_batches {
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
    wsi_tile_batch_region_scaled_distinct_rgb_q4.finish();

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
        viewer_region_scaled_composite_rgb.bench_function(format!("{}/hybrid", input.name), |b| {
            b.iter(|| hybrid_viewport_composite(&input.bytes, input.dimensions));
        });
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
        viewer_region_scaled_composite_rgb_device.bench_function(
            format!("{}/hybrid", input.name),
            |b| {
                b.iter(|| hybrid_viewport_composite_device(&input.bytes, input.dimensions));
            },
        );
    }
    viewer_region_scaled_composite_rgb_device.finish();

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
                    let out = compose_viewport_cpu(
                        &decoder,
                        &mut pool,
                        PixelFormat::Rgb8,
                        workload.scale,
                        workload.viewport_dims,
                        &workload.tiles,
                    )
                    .expect("cpu warm viewport");
                    std::hint::black_box(out);
                });
            },
        );

        let workload = suggest_viewport_workload(input.dimensions).expect("warm workload");
        let hybrid_bytes = input.bytes.clone();
        viewer_region_scaled_composite_rgb_warm.bench_function(
            format!("{}/hybrid", input.name),
            move |b| {
                let decoder = CpuDecoder::new(&hybrid_bytes).expect("cpu decoder");
                let mut pool = CpuScratchPool::new();
                let stride =
                    workload.viewport_dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
                let mut out = vec![0u8; stride * workload.viewport_dims.1 as usize];
                b.iter(|| {
                    let surface = compose_viewport_hybrid(
                        &decoder,
                        &mut pool,
                        workload.scale,
                        workload.viewport_dims,
                        &workload.tiles,
                    )
                    .expect("hybrid warm viewport");
                    surface
                        .download_into(&mut out, stride)
                        .expect("hybrid warm download");
                    std::hint::black_box(&out);
                });
            },
        );
    }
    viewer_region_scaled_composite_rgb_warm.finish();

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
                    let surface = compose_viewport_cpu_to_surface(
                        &decoder,
                        &mut pool,
                        workload.scale,
                        workload.viewport_dims,
                        &workload.tiles,
                    )
                    .expect("cpu warm viewport surface");
                    std::hint::black_box(surface);
                });
            },
        );

        let workload = suggest_viewport_workload(input.dimensions).expect("warm workload");
        let hybrid_bytes = input.bytes.clone();
        viewer_region_scaled_composite_rgb_device_warm.bench_function(
            format!("{}/hybrid", input.name),
            move |b| {
                let decoder = CpuDecoder::new(&hybrid_bytes).expect("cpu decoder");
                let mut pool = CpuScratchPool::new();
                b.iter(|| {
                    let surface = compose_viewport_hybrid(
                        &decoder,
                        &mut pool,
                        workload.scale,
                        workload.viewport_dims,
                        &workload.tiles,
                    )
                    .expect("hybrid warm viewport surface");
                    std::hint::black_box(surface);
                });
            },
        );
    }
    viewer_region_scaled_composite_rgb_device_warm.finish();

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
                let out =
                    decode_viewport_region_cpu(&decoder, &mut pool, PixelFormat::Rgb8, &workload)
                        .expect("cpu contiguous viewport");
                std::hint::black_box(out);
            });
        });
        viewer_contiguous_region_scaled_rgb.bench_function(format!("{}/hybrid", input.name), |b| {
            let decoder = CpuDecoder::new(&input.bytes).expect("cpu decoder");
            let mut pool = CpuScratchPool::new();
            let stride = workload.viewport_dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
            let mut out = vec![0u8; stride * workload.viewport_dims.1 as usize];
            b.iter(|| {
                let surface = decode_viewport_region_hybrid(&decoder, &mut pool, &workload)
                    .expect("hybrid contiguous viewport");
                surface
                    .download_into(&mut out, stride)
                    .expect("hybrid contiguous download");
                std::hint::black_box(&out);
            });
        });
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
                    let surface =
                        decode_viewport_region_cpu_to_surface(&decoder, &mut pool, &workload)
                            .expect("cpu contiguous upload");
                    std::hint::black_box(surface);
                });
            },
        );
        viewer_contiguous_region_scaled_rgb_device.bench_function(
            format!("{}/hybrid", input.name),
            |b| {
                let decoder = CpuDecoder::new(&input.bytes).expect("cpu decoder");
                let mut pool = CpuScratchPool::new();
                b.iter(|| {
                    let surface = decode_viewport_region_hybrid(&decoder, &mut pool, &workload)
                        .expect("hybrid contiguous viewport");
                    std::hint::black_box(surface);
                });
            },
        );
    }
    viewer_contiguous_region_scaled_rgb_device.finish();

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

criterion_group!(benches, bench_compare);
criterion_main!(benches);
