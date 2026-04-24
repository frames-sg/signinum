// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use slidecodec_core::BackendRequest;
use slidecodec_jpeg::{ColorSpace, Decoder as CpuDecoder, ScratchPool};
use slidecodec_jpeg_metal::viewport::{decode_viewport_to_surface, suggest_viewport_workload};

const FULL_FRAME_MAX_OUTPUT_BYTES: usize = 512 * 1024 * 1024;
const DEFAULT_WARMUP_ITERS: usize = 3;
const DEFAULT_SAMPLE_ITERS: usize = 10;

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

#[derive(Clone, Copy)]
struct Summary {
    min: Duration,
    median: Duration,
    max: Duration,
}

fn main() {
    let sample_iters = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_SAMPLE_ITERS);

    let mut inputs = load_bench_inputs();
    inputs.retain(|input| {
        input.mode == DecodeMode::Rgb
            && input.input_class == CorpusInputClass::BoundedFullFrame
            && suggest_viewport_workload(input.dimensions).is_some()
    });

    if inputs.is_empty() {
        eprintln!(
            "viewport_report: no eligible JPEG inputs found; set SLIDECODEC_BENCH_INPUTS to extracted JPEG tiles or levels"
        );
        std::process::exit(2);
    }

    println!("| input | cpu median | adaptive median | speedup | cpu min/max | adaptive min/max |");
    println!("|---|---:|---:|---:|---:|---:|");

    for input in inputs {
        let cpu = summarize_input(&input, BackendRequest::Cpu, sample_iters);
        let adaptive = summarize_input(&input, BackendRequest::Auto, sample_iters);
        let speedup = ratio(cpu.median, adaptive.median);
        println!(
            "| {} | {} | {} | {:.3}x | {} / {} | {} / {} |",
            input.name,
            format_duration(cpu.median),
            format_duration(adaptive.median),
            speedup,
            format_duration(cpu.min),
            format_duration(cpu.max),
            format_duration(adaptive.min),
            format_duration(adaptive.max),
        );
    }
}

fn summarize_input(input: &BenchInput, backend: BackendRequest, sample_iters: usize) -> Summary {
    let decoder = CpuDecoder::new(&input.bytes).expect("cpu decoder");
    let workload = suggest_viewport_workload(input.dimensions).expect("viewport workload");
    let mut pool = ScratchPool::new();

    for _ in 0..DEFAULT_WARMUP_ITERS {
        let surface =
            decode_viewport_to_surface(&decoder, &mut pool, &workload, backend).expect("surface");
        std::hint::black_box(surface);
    }

    let mut samples = Vec::with_capacity(sample_iters);
    for _ in 0..sample_iters {
        let start = Instant::now();
        let surface =
            decode_viewport_to_surface(&decoder, &mut pool, &workload, backend).expect("surface");
        std::hint::black_box(surface);
        samples.push(start.elapsed());
    }
    summarize(&mut samples)
}

fn summarize(samples: &mut [Duration]) -> Summary {
    samples.sort_unstable();
    Summary {
        min: samples[0],
        median: samples[samples.len() / 2],
        max: samples[samples.len() - 1],
    }
}

fn ratio(cpu: Duration, adaptive: Duration) -> f64 {
    cpu.as_secs_f64() / adaptive.as_secs_f64()
}

fn format_duration(duration: Duration) -> String {
    let nanos = duration.as_nanos();
    if nanos >= 1_000_000 {
        format!("{:.3} ms", duration.as_secs_f64() * 1_000.0)
    } else if nanos >= 1_000 {
        format!("{:.3} µs", duration.as_secs_f64() * 1_000_000.0)
    } else {
        format!("{nanos} ns")
    }
}

fn load_bench_inputs() -> Vec<BenchInput> {
    let mut inputs = vec![
        BenchInput {
            name: "repo/baseline_420_16x16".to_string(),
            bytes: include_bytes!("../../../../corpus/conformance/baseline_420_16x16.jpg").to_vec(),
            dimensions: (16, 16),
            mode: DecodeMode::Rgb,
            input_class: CorpusInputClass::BoundedFullFrame,
        },
        BenchInput {
            name: "repo/grayscale_8x8".to_string(),
            bytes: include_bytes!("../../../../corpus/conformance/grayscale_8x8.jpg").to_vec(),
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
        std::env::split_paths(&std::env::var_os("SLIDECODEC_BENCH_INPUTS").unwrap_or_default())
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

fn color_space_mode(color_space: ColorSpace) -> Option<DecodeMode> {
    match color_space {
        ColorSpace::Grayscale => Some(DecodeMode::Gray),
        ColorSpace::YCbCr | ColorSpace::Rgb => Some(DecodeMode::Rgb),
        ColorSpace::Cmyk | ColorSpace::Ycck => None,
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
