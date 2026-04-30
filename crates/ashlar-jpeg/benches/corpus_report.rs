// SPDX-License-Identifier: Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::format_push_string,
    clippy::manual_clamp,
    clippy::semicolon_if_nothing_returned,
    clippy::unnested_or_patterns,
    dead_code
)]

mod common;

use ashlar_jpeg::{Downscale, Rect};
use common::{
    ashlar_decode, ashlar_decode_region, ashlar_decode_region_scaled, ashlar_decode_rows,
    ashlar_decode_scaled, ashlar_decode_tile_batch_region_scaled, ashlar_decode_tile_batch_scaled,
    ashlar_inspect, centered_roi,
    classification::{should_compare_full_frame, CorpusInputClass},
    jpeg_decoder_decode, jpeg_decoder_decode_batch_region_scaled, jpeg_decoder_decode_batch_scaled,
    jpeg_decoder_decode_region, jpeg_decoder_decode_region_scaled, jpeg_decoder_decode_scaled,
    jpeg_decoder_inspect, load_bench_inputs, scaled_rect, zune_decode,
    zune_decode_batch_region_scaled, zune_decode_batch_scaled, zune_decode_region,
    zune_decode_region_scaled, zune_decode_scaled, zune_inspect, BenchInput, DecodeMode,
};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const ROI_SIDE: u32 = 256;
const TILE_BATCH: usize = 64;
const DEFAULT_ITERS: usize = 3;
const TIE_THRESHOLD: f64 = 0.01;

fn main() {
    let mut inputs = load_bench_inputs();
    if std::env::var_os("ASHLAR_BENCH_INPUTS").is_some() {
        inputs.retain(|input| !input.name.starts_with("repo/"));
    }
    inputs.sort_by(|lhs, rhs| {
        lhs.input_class
            .cmp(&rhs.input_class)
            .then_with(|| lhs.mode.cmp(&rhs.mode))
            .then_with(|| lhs.name.cmp(&rhs.name))
    });
    let iterations = std::env::var("ASHLAR_REPORT_ITERS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&iters| iters > 0)
        .unwrap_or(DEFAULT_ITERS);

    let mut rows = Vec::new();
    for input in &inputs {
        rows.extend(run_input(input, iterations));
    }

    let report_dir = PathBuf::from("target/bench-reports");
    fs::create_dir_all(&report_dir).expect("create target/bench-reports");
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after unix epoch")
        .as_secs();
    let csv_path = report_dir.join(format!("corpus-report-{timestamp}.csv"));
    let md_path = report_dir.join(format!("corpus-report-{timestamp}.md"));
    let latest_csv = report_dir.join("corpus-report-latest.csv");
    let latest_md = report_dir.join("corpus-report-latest.md");

    let csv = render_csv(&rows);
    let markdown = render_markdown(&rows, iterations);

    fs::write(&csv_path, &csv).expect("write CSV report");
    fs::write(&md_path, &markdown).expect("write Markdown report");
    fs::write(&latest_csv, &csv).expect("write latest CSV report");
    fs::write(&latest_md, &markdown).expect("write latest Markdown report");

    println!("Wrote {}", csv_path.display());
    println!("Wrote {}", md_path.display());
    println!();
    println!("{markdown}");
}

#[derive(Clone, Copy)]
enum Operation {
    Inspect,
    DecodeRgb,
    DecodeGray,
    DecodeRowsRgb,
    WsiRegionRgb,
    WsiScaledRgbQ4,
    WsiScaledRgbQ8,
    WsiRegionScaledRgbQ4,
    WsiRegionScaledRgbQ8,
    WsiTileBatchScaledRgbQ4,
    WsiTileBatchRegionScaledRgbQ4,
}

impl Operation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Inspect => "inspect",
            Self::DecodeRgb => "decode_rgb",
            Self::DecodeGray => "decode_gray",
            Self::DecodeRowsRgb => "decode_rows_rgb",
            Self::WsiRegionRgb => "wsi_region_rgb",
            Self::WsiScaledRgbQ4 => "wsi_scaled_rgb_q4",
            Self::WsiScaledRgbQ8 => "wsi_scaled_rgb_q8",
            Self::WsiRegionScaledRgbQ4 => "wsi_region_scaled_rgb_q4",
            Self::WsiRegionScaledRgbQ8 => "wsi_region_scaled_rgb_q8",
            Self::WsiTileBatchScaledRgbQ4 => "wsi_tile_batch_scaled_rgb_q4",
            Self::WsiTileBatchRegionScaledRgbQ4 => "wsi_tile_batch_region_scaled_rgb_q4",
        }
    }
}

#[derive(Clone, Copy)]
enum Library {
    Ashlar,
    JpegDecoder,
    Zune,
}

impl Library {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ashlar => "ashlar",
            Self::JpegDecoder => "jpeg-decoder",
            Self::Zune => "zune-jpeg",
        }
    }
}

#[derive(Clone)]
struct Measurement {
    ns: Option<u128>,
    error: Option<String>,
}

impl Measurement {
    fn success(ns: u128) -> Self {
        Self {
            ns: Some(ns),
            error: None,
        }
    }

    fn skipped(reason: &str) -> Self {
        Self {
            ns: None,
            error: Some(reason.to_string()),
        }
    }

    fn failure(message: String) -> Self {
        Self {
            ns: None,
            error: Some(message),
        }
    }
}

struct ReportRow {
    input_name: String,
    mode: DecodeMode,
    input_class: CorpusInputClass,
    operation: Operation,
    ashlar: Measurement,
    jpeg_decoder: Measurement,
    zune: Measurement,
}

fn run_input(input: &BenchInput, iterations: usize) -> Vec<ReportRow> {
    let mut rows = vec![run_compare_row(
        input,
        Operation::Inspect,
        iterations_for(input, Operation::Inspect, iterations),
    )];
    match (input.mode, input.input_class) {
        (DecodeMode::Gray, _) if should_compare_full_frame(input.mode, input.input_class) => {
            rows.push(run_compare_row(
                input,
                Operation::DecodeGray,
                iterations_for(input, Operation::DecodeGray, iterations),
            ));
        }
        (DecodeMode::Rgb, CorpusInputClass::BoundedFullFrame) => {
            rows.push(run_compare_row(
                input,
                Operation::DecodeRgb,
                iterations_for(input, Operation::DecodeRgb, iterations),
            ));
            rows.push(run_compare_row(
                input,
                Operation::WsiRegionRgb,
                iterations_for(input, Operation::WsiRegionRgb, iterations),
            ));
            rows.push(run_compare_row(
                input,
                Operation::WsiScaledRgbQ4,
                iterations_for(input, Operation::WsiScaledRgbQ4, iterations),
            ));
            rows.push(run_compare_row(
                input,
                Operation::WsiScaledRgbQ8,
                iterations_for(input, Operation::WsiScaledRgbQ8, iterations),
            ));
            rows.push(run_compare_row(
                input,
                Operation::WsiRegionScaledRgbQ4,
                iterations_for(input, Operation::WsiRegionScaledRgbQ4, iterations),
            ));
            rows.push(run_compare_row(
                input,
                Operation::WsiRegionScaledRgbQ8,
                iterations_for(input, Operation::WsiRegionScaledRgbQ8, iterations),
            ));
            rows.push(run_compare_row(
                input,
                Operation::WsiTileBatchScaledRgbQ4,
                iterations_for(input, Operation::WsiTileBatchScaledRgbQ4, iterations),
            ));
            rows.push(run_compare_row(
                input,
                Operation::WsiTileBatchRegionScaledRgbQ4,
                iterations_for(input, Operation::WsiTileBatchRegionScaledRgbQ4, iterations),
            ));
        }
        (DecodeMode::Rgb, CorpusInputClass::VeryLarge)
            if should_compare_full_frame(input.mode, input.input_class) =>
        {
            rows.push(run_compare_row(
                input,
                Operation::DecodeRgb,
                iterations_for(input, Operation::DecodeRgb, iterations),
            ));
            rows.push(run_ashlar_only_row(
                input,
                Operation::DecodeRowsRgb,
                iterations_for(input, Operation::DecodeRowsRgb, iterations),
                "comparator skipped for very large RGB input; report uses ashlar decode_rows",
            ));
        }
        (DecodeMode::Rgb, CorpusInputClass::VeryLarge) => {
            rows.push(run_ashlar_only_row(
                input,
                Operation::DecodeRowsRgb,
                iterations_for(input, Operation::DecodeRowsRgb, iterations),
                "comparator skipped for very large RGB input; report uses ashlar decode_rows",
            ));
        }
        (DecodeMode::Gray, CorpusInputClass::BoundedFullFrame) => {
            unreachable!("bounded grayscale inputs are always compared full-frame")
        }
        (DecodeMode::Gray, CorpusInputClass::VeryLarge) => {}
    }
    rows
}

fn iterations_for(input: &BenchInput, operation: Operation, default_iters: usize) -> usize {
    if input.input_class == CorpusInputClass::VeryLarge {
        return match operation {
            Operation::Inspect => default_iters,
            Operation::DecodeRowsRgb => default_iters.min(2).max(1),
            Operation::DecodeRgb | Operation::DecodeGray => 1,
            _ => 1,
        };
    }
    default_iters
}

fn inner_loops_for(input: &BenchInput, operation: Operation) -> usize {
    if matches!(
        operation,
        Operation::DecodeRowsRgb
            | Operation::WsiTileBatchScaledRgbQ4
            | Operation::WsiTileBatchRegionScaledRgbQ4
    ) {
        return 1;
    }

    if matches!(
        operation,
        Operation::WsiScaledRgbQ4
            | Operation::WsiScaledRgbQ8
            | Operation::WsiRegionScaledRgbQ4
            | Operation::WsiRegionScaledRgbQ8
    ) {
        let Some(output_bytes) = estimated_output_bytes(input, operation) else {
            return 1;
        };
        return if output_bytes <= 512 * 1024 { 8 } else { 1 };
    }

    let Some(output_bytes) = estimated_output_bytes(input, operation) else {
        return match operation {
            Operation::Inspect => 64,
            Operation::DecodeRowsRgb => 1,
            _ => 1,
        };
    };

    if output_bytes <= 512 * 1024 {
        64
    } else if output_bytes <= 2 * 1024 * 1024 {
        16
    } else if output_bytes <= 8 * 1024 * 1024 {
        8
    } else if output_bytes <= 64 * 1024 * 1024 {
        2
    } else {
        1
    }
}

fn estimated_output_bytes(input: &BenchInput, operation: Operation) -> Option<usize> {
    let bpp = match operation {
        Operation::DecodeGray => 1usize,
        Operation::DecodeRowsRgb
        | Operation::DecodeRgb
        | Operation::WsiRegionRgb
        | Operation::WsiScaledRgbQ4
        | Operation::WsiScaledRgbQ8
        | Operation::WsiRegionScaledRgbQ4
        | Operation::WsiRegionScaledRgbQ8
        | Operation::WsiTileBatchScaledRgbQ4
        | Operation::WsiTileBatchRegionScaledRgbQ4 => 3usize,
        Operation::Inspect => return None,
    };

    let dims = match operation {
        Operation::DecodeRgb | Operation::DecodeGray | Operation::DecodeRowsRgb => input.dimensions,
        Operation::WsiRegionRgb => rect_dims(centered_roi(input.dimensions, ROI_SIDE)),
        Operation::WsiScaledRgbQ4 => rect_dims(scaled_rect(
            Rect::full(input.dimensions),
            Downscale::Quarter,
        )),
        Operation::WsiScaledRgbQ8 => {
            rect_dims(scaled_rect(Rect::full(input.dimensions), Downscale::Eighth))
        }
        Operation::WsiRegionScaledRgbQ4 => rect_dims(scaled_rect(
            centered_roi(input.dimensions, ROI_SIDE),
            Downscale::Quarter,
        )),
        Operation::WsiRegionScaledRgbQ8 => rect_dims(scaled_rect(
            centered_roi(input.dimensions, ROI_SIDE),
            Downscale::Eighth,
        )),
        Operation::WsiTileBatchScaledRgbQ4 => rect_dims(scaled_rect(
            Rect::full(input.dimensions),
            Downscale::Quarter,
        )),
        Operation::WsiTileBatchRegionScaledRgbQ4 => rect_dims(scaled_rect(
            centered_roi(input.dimensions, ROI_SIDE),
            Downscale::Quarter,
        )),
        Operation::Inspect => return None,
    };

    usize::try_from(dims.0)
        .ok()
        .and_then(|width| usize::try_from(dims.1).ok().map(|height| (width, height)))
        .and_then(|(width, height)| width.checked_mul(height))
        .and_then(|pixels| pixels.checked_mul(bpp))
}

fn rect_dims(rect: Rect) -> (u32, u32) {
    (rect.w, rect.h)
}

fn run_compare_row(input: &BenchInput, operation: Operation, iterations: usize) -> ReportRow {
    let (ashlar, jpeg_decoder, zune) = run_compare_measurements(operation, input, iterations);
    ReportRow {
        input_name: input.name.clone(),
        mode: input.mode,
        input_class: input.input_class,
        operation,
        ashlar,
        jpeg_decoder,
        zune,
    }
}

fn run_ashlar_only_row(
    input: &BenchInput,
    operation: Operation,
    iterations: usize,
    skip_reason: &str,
) -> ReportRow {
    ReportRow {
        input_name: input.name.clone(),
        mode: input.mode,
        input_class: input.input_class,
        operation,
        ashlar: run_measurement(Library::Ashlar, operation, input, iterations),
        jpeg_decoder: Measurement::skipped(skip_reason),
        zune: Measurement::skipped(skip_reason),
    }
}

fn run_measurement(
    library: Library,
    operation: Operation,
    input: &BenchInput,
    iterations: usize,
) -> Measurement {
    if !is_supported(library, operation, input) {
        return Measurement::skipped("unsupported for this library/input combination");
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut samples = Vec::with_capacity(iterations);
        run_operation(library, operation, input);
        for _ in 0..iterations {
            let start = Instant::now();
            run_operation(library, operation, input);
            samples.push(start.elapsed().as_nanos());
        }
        samples.sort_unstable();
        samples[samples.len() / 2]
    }));

    match result {
        Ok(ns) => Measurement::success(ns),
        Err(payload) => Measurement::failure(panic_message(payload)),
    }
}

fn is_supported(library: Library, operation: Operation, input: &BenchInput) -> bool {
    matches!(
        (library, operation, input.mode, input.input_class),
        (Library::Ashlar, Operation::Inspect, _, _)
            | (Library::JpegDecoder, Operation::Inspect, _, _)
            | (Library::Zune, Operation::Inspect, _, _)
            | (
                _,
                Operation::DecodeRgb,
                DecodeMode::Rgb,
                CorpusInputClass::BoundedFullFrame
            )
            | (
                _,
                Operation::DecodeRgb,
                DecodeMode::Rgb,
                CorpusInputClass::VeryLarge
            )
            | (
                _,
                Operation::DecodeGray,
                DecodeMode::Gray,
                CorpusInputClass::BoundedFullFrame
            )
            | (
                _,
                Operation::DecodeGray,
                DecodeMode::Gray,
                CorpusInputClass::VeryLarge
            )
            | (
                _,
                Operation::WsiRegionRgb,
                DecodeMode::Rgb,
                CorpusInputClass::BoundedFullFrame
            )
            | (
                _,
                Operation::WsiScaledRgbQ4,
                DecodeMode::Rgb,
                CorpusInputClass::BoundedFullFrame
            )
            | (_, Operation::WsiScaledRgbQ8, DecodeMode::Rgb, _)
            | (
                _,
                Operation::WsiRegionScaledRgbQ4,
                DecodeMode::Rgb,
                CorpusInputClass::BoundedFullFrame
            )
            | (_, Operation::WsiRegionScaledRgbQ8, DecodeMode::Rgb, _)
            | (_, Operation::WsiTileBatchScaledRgbQ4, DecodeMode::Rgb, _)
            | (
                _,
                Operation::WsiTileBatchRegionScaledRgbQ4,
                DecodeMode::Rgb,
                _
            )
            | (
                Library::Ashlar,
                Operation::DecodeRowsRgb,
                DecodeMode::Rgb,
                CorpusInputClass::VeryLarge
            )
    )
}

fn run_compare_measurements(
    operation: Operation,
    input: &BenchInput,
    iterations: usize,
) -> (Measurement, Measurement, Measurement) {
    let mut ashlar = MeasurementState::new(
        Library::Ashlar,
        is_supported(Library::Ashlar, operation, input),
        iterations,
    );
    let mut jpeg_decoder = MeasurementState::new(
        Library::JpegDecoder,
        is_supported(Library::JpegDecoder, operation, input),
        iterations,
    );
    let mut zune = MeasurementState::new(
        Library::Zune,
        is_supported(Library::Zune, operation, input),
        iterations,
    );
    let mut states = [&mut ashlar, &mut jpeg_decoder, &mut zune];
    let inner_loops = inner_loops_for(input, operation);

    for state in &mut states {
        state.warm(operation, input);
    }

    for iteration in 0..iterations {
        for step in 0..states.len() {
            let idx = (iteration + step) % states.len();
            states[idx].measure(operation, input, inner_loops);
        }
    }

    (ashlar.finish(), jpeg_decoder.finish(), zune.finish())
}

fn time_operation(
    library: Library,
    operation: Operation,
    input: &BenchInput,
    inner_loops: usize,
) -> Measurement {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let start = Instant::now();
        for _ in 0..inner_loops {
            run_operation(library, operation, input);
        }
        start.elapsed().as_nanos() / inner_loops as u128
    }));

    match result {
        Ok(ns) => Measurement::success(ns),
        Err(payload) => Measurement::failure(panic_message(payload)),
    }
}

fn run_operation(library: Library, operation: Operation, input: &BenchInput) {
    match (library, operation) {
        (Library::Ashlar, Operation::Inspect) => ashlar_inspect(&input.bytes),
        (Library::JpegDecoder, Operation::Inspect) => jpeg_decoder_inspect(&input.bytes),
        (Library::Zune, Operation::Inspect) => zune_inspect(&input.bytes),
        (Library::Ashlar, Operation::DecodeRgb) => ashlar_decode(&input.bytes, DecodeMode::Rgb),
        (Library::JpegDecoder, Operation::DecodeRgb) => jpeg_decoder_decode(&input.bytes),
        (Library::Zune, Operation::DecodeRgb) => zune_decode(&input.bytes, DecodeMode::Rgb),
        (Library::Ashlar, Operation::DecodeGray) => ashlar_decode(&input.bytes, DecodeMode::Gray),
        (Library::JpegDecoder, Operation::DecodeGray) => jpeg_decoder_decode(&input.bytes),
        (Library::Zune, Operation::DecodeGray) => zune_decode(&input.bytes, DecodeMode::Gray),
        (Library::Ashlar, Operation::DecodeRowsRgb) => ashlar_decode_rows(&input.bytes),
        (Library::Ashlar, Operation::WsiRegionRgb) => ashlar_decode_region(&input.bytes, ROI_SIDE),
        (Library::JpegDecoder, Operation::WsiRegionRgb) => {
            jpeg_decoder_decode_region(&input.bytes, ROI_SIDE)
        }
        (Library::Zune, Operation::WsiRegionRgb) => zune_decode_region(&input.bytes, ROI_SIDE),
        (Library::Ashlar, Operation::WsiScaledRgbQ4) => {
            ashlar_decode_scaled(&input.bytes, Downscale::Quarter);
        }
        (Library::JpegDecoder, Operation::WsiScaledRgbQ4) => {
            jpeg_decoder_decode_scaled(&input.bytes, Downscale::Quarter);
        }
        (Library::Zune, Operation::WsiScaledRgbQ4) => {
            zune_decode_scaled(&input.bytes, Downscale::Quarter);
        }
        (Library::Ashlar, Operation::WsiScaledRgbQ8) => {
            ashlar_decode_scaled(&input.bytes, Downscale::Eighth);
        }
        (Library::JpegDecoder, Operation::WsiScaledRgbQ8) => {
            jpeg_decoder_decode_scaled(&input.bytes, Downscale::Eighth);
        }
        (Library::Zune, Operation::WsiScaledRgbQ8) => {
            zune_decode_scaled(&input.bytes, Downscale::Eighth);
        }
        (Library::Ashlar, Operation::WsiRegionScaledRgbQ4) => {
            ashlar_decode_region_scaled(&input.bytes, ROI_SIDE, Downscale::Quarter);
        }
        (Library::JpegDecoder, Operation::WsiRegionScaledRgbQ4) => {
            jpeg_decoder_decode_region_scaled(&input.bytes, ROI_SIDE, Downscale::Quarter);
        }
        (Library::Zune, Operation::WsiRegionScaledRgbQ4) => {
            zune_decode_region_scaled(&input.bytes, ROI_SIDE, Downscale::Quarter);
        }
        (Library::Ashlar, Operation::WsiRegionScaledRgbQ8) => {
            ashlar_decode_region_scaled(&input.bytes, ROI_SIDE, Downscale::Eighth);
        }
        (Library::JpegDecoder, Operation::WsiRegionScaledRgbQ8) => {
            jpeg_decoder_decode_region_scaled(&input.bytes, ROI_SIDE, Downscale::Eighth);
        }
        (Library::Zune, Operation::WsiRegionScaledRgbQ8) => {
            zune_decode_region_scaled(&input.bytes, ROI_SIDE, Downscale::Eighth);
        }
        (Library::Ashlar, Operation::WsiTileBatchScaledRgbQ4) => {
            ashlar_decode_tile_batch_scaled(&input.bytes, TILE_BATCH, Downscale::Quarter);
        }
        (Library::JpegDecoder, Operation::WsiTileBatchScaledRgbQ4) => {
            jpeg_decoder_decode_batch_scaled(&input.bytes, TILE_BATCH, Downscale::Quarter);
        }
        (Library::Zune, Operation::WsiTileBatchScaledRgbQ4) => {
            zune_decode_batch_scaled(&input.bytes, TILE_BATCH, Downscale::Quarter);
        }
        (Library::Ashlar, Operation::WsiTileBatchRegionScaledRgbQ4) => {
            ashlar_decode_tile_batch_region_scaled(
                &input.bytes,
                TILE_BATCH,
                ROI_SIDE,
                Downscale::Quarter,
            );
        }
        (Library::JpegDecoder, Operation::WsiTileBatchRegionScaledRgbQ4) => {
            jpeg_decoder_decode_batch_region_scaled(
                &input.bytes,
                TILE_BATCH,
                ROI_SIDE,
                Downscale::Quarter,
            );
        }
        (Library::Zune, Operation::WsiTileBatchRegionScaledRgbQ4) => {
            zune_decode_batch_region_scaled(&input.bytes, TILE_BATCH, ROI_SIDE, Downscale::Quarter);
        }
        _ => unreachable!("unsupported operation dispatched after validation"),
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        return (*message).to_string();
    }
    "panic without string payload".to_string()
}

fn render_csv(rows: &[ReportRow]) -> String {
    let mut csv = String::from(
        "input,mode,class,operation,ashlar_ns,jpeg_decoder_ns,zune_ns,ashlar_error,jpeg_decoder_error,zune_error,fastest\n",
    );
    for row in rows {
        let fastest = fastest_label(row).unwrap_or("n/a");
        csv.push_str(&format!(
            "\"{}\",{},{},{},{},{},{},\"{}\",\"{}\",\"{}\",{}\n",
            escape_csv(&row.input_name),
            row.mode.as_str(),
            row.input_class.as_str(),
            row.operation.as_str(),
            render_ns(&row.ashlar),
            render_ns(&row.jpeg_decoder),
            render_ns(&row.zune),
            escape_csv(row.ashlar.error.as_deref().unwrap_or("")),
            escape_csv(row.jpeg_decoder.error.as_deref().unwrap_or("")),
            escape_csv(row.zune.error.as_deref().unwrap_or("")),
            fastest
        ));
    }
    csv
}

fn render_markdown(rows: &[ReportRow], iterations: usize) -> String {
    let mut summary = BTreeMap::<&'static str, Summary>::new();
    for row in rows {
        summary
            .entry(row.operation.as_str())
            .or_default()
            .accumulate(row);
    }

    let mut md = String::new();
    md.push_str("# Ashlar JPEG corpus report\n\n");
    md.push_str(&format!(
        "- inputs: {}\n",
        rows.iter()
            .map(|row| &row.input_name)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    ));
    md.push_str(&format!("- rows: {}\n", rows.len()));
    md.push_str(&format!("- iterations per measurement: {iterations}\n"));
    md.push_str(&format!(
        "- tie threshold: {:.0}%\n\n",
        TIE_THRESHOLD * 100.0
    ));
    md.push_str("## Summary by operation\n\n");
    md.push_str("| operation | ashlar fastest | vs jpeg wins | vs jpeg losses | vs zune wins | vs zune losses | failures |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|---:|\n");
    for (operation, stats) in &summary {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n",
            operation,
            stats.ashlar_fastest,
            stats.vs_jpeg_wins,
            stats.vs_jpeg_losses,
            stats.vs_zune_wins,
            stats.vs_zune_losses,
            stats.failures,
        ));
    }
    md.push_str("\n## Rows where ashlar is not fastest\n\n");
    md.push_str("| input | operation | ashlar | jpeg-decoder | zune-jpeg | fastest |\n");
    md.push_str("|---|---|---:|---:|---:|---|\n");
    let mut any_slower = false;
    for row in rows {
        let fastest = fastest_label(row);
        if fastest == Some("ashlar") || fastest.is_none() {
            continue;
        }
        any_slower = true;
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            row.input_name,
            row.operation.as_str(),
            format_measurement(&row.ashlar),
            format_measurement(&row.jpeg_decoder),
            format_measurement(&row.zune),
            fastest.unwrap_or("n/a"),
        ));
    }
    if !any_slower {
        md.push_str("| none | — | — | — | — | — |\n");
    }

    md.push_str("\n## Failures / skips\n\n");
    md.push_str("| input | operation | ashlar | jpeg-decoder | zune-jpeg |\n");
    md.push_str("|---|---|---|---|---|\n");
    let mut any_failures = false;
    for row in rows {
        if row.ashlar.error.is_none()
            && row.jpeg_decoder.error.is_none()
            && row.zune.error.is_none()
        {
            continue;
        }
        any_failures = true;
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            row.input_name,
            row.operation.as_str(),
            row.ashlar.error.as_deref().unwrap_or("ok"),
            row.jpeg_decoder.error.as_deref().unwrap_or("ok"),
            row.zune.error.as_deref().unwrap_or("ok"),
        ));
    }
    if !any_failures {
        md.push_str("| none | — | — | — | — |\n");
    }

    md
}

#[derive(Default)]
struct Summary {
    ashlar_fastest: usize,
    vs_jpeg_wins: usize,
    vs_jpeg_losses: usize,
    vs_zune_wins: usize,
    vs_zune_losses: usize,
    failures: usize,
}

struct MeasurementState {
    library: Library,
    samples: Vec<u128>,
    error: Option<String>,
    supported: bool,
}

impl MeasurementState {
    fn new(library: Library, supported: bool, iterations: usize) -> Self {
        Self {
            library,
            samples: Vec::with_capacity(iterations),
            error: None,
            supported,
        }
    }

    fn warm(&mut self, operation: Operation, input: &BenchInput) {
        if !self.supported || self.error.is_some() {
            return;
        }
        let measurement = time_operation(self.library, operation, input, 1);
        if let Some(message) = measurement.error {
            self.error = Some(message);
        }
    }

    fn measure(&mut self, operation: Operation, input: &BenchInput, inner_loops: usize) {
        if !self.supported || self.error.is_some() {
            return;
        }
        match time_operation(self.library, operation, input, inner_loops) {
            Measurement {
                ns: Some(ns),
                error: None,
            } => self.samples.push(ns),
            Measurement {
                ns: None,
                error: Some(message),
            } => self.error = Some(message),
            Measurement {
                ns: None,
                error: None,
            } => self.error = Some("measurement without result".to_string()),
            Measurement {
                ns: Some(_),
                error: Some(message),
            } => self.error = Some(message),
        }
    }

    fn finish(mut self) -> Measurement {
        if !self.supported {
            return Measurement::skipped("unsupported for this library/input combination");
        }
        if let Some(message) = self.error {
            return Measurement::failure(message);
        }
        self.samples.sort_unstable();
        Measurement::success(self.samples[self.samples.len() / 2])
    }
}

impl Summary {
    fn accumulate(&mut self, row: &ReportRow) {
        if fastest_label(row) == Some("ashlar") {
            self.ashlar_fastest += 1;
        }
        match compare_measurements(&row.ashlar, &row.jpeg_decoder) {
            Some(Ordering::Less) => self.vs_jpeg_wins += 1,
            Some(Ordering::Greater) => self.vs_jpeg_losses += 1,
            _ => {}
        }
        match compare_measurements(&row.ashlar, &row.zune) {
            Some(Ordering::Less) => self.vs_zune_wins += 1,
            Some(Ordering::Greater) => self.vs_zune_losses += 1,
            _ => {}
        }
        if row.ashlar.error.is_some()
            || row.jpeg_decoder.error.is_some()
            || row.zune.error.is_some()
        {
            self.failures += 1;
        }
    }
}

fn fastest_label(row: &ReportRow) -> Option<&'static str> {
    let mut best_ns: Option<u128> = None;
    for measurement in [&row.ashlar, &row.jpeg_decoder, &row.zune] {
        let Some(ns) = measurement.ns else {
            continue;
        };
        best_ns = Some(best_ns.map_or(ns, |best| best.min(ns)));
    }
    let best_ns = best_ns?;
    if row.ashlar.ns.is_some_and(|ns| {
        let max_ns = ns.max(best_ns) as f64;
        max_ns > 0.0 && ((ns as f64 - best_ns as f64).abs() / max_ns) <= TIE_THRESHOLD
    }) {
        return Some("ashlar");
    }
    if row.jpeg_decoder.ns.is_some_and(|ns| ns == best_ns) {
        return Some("jpeg-decoder");
    }
    if row.zune.ns.is_some_and(|ns| ns == best_ns) {
        return Some("zune-jpeg");
    }
    None
}

fn compare_measurements(lhs: &Measurement, rhs: &Measurement) -> Option<Ordering> {
    let (Some(lhs_ns), Some(rhs_ns)) = (lhs.ns, rhs.ns) else {
        return None;
    };
    let max_ns = lhs_ns.max(rhs_ns) as f64;
    if max_ns > 0.0 && ((lhs_ns as f64 - rhs_ns as f64).abs() / max_ns) <= TIE_THRESHOLD {
        return Some(Ordering::Equal);
    }
    Some(lhs_ns.cmp(&rhs_ns))
}

fn format_measurement(measurement: &Measurement) -> String {
    if let Some(ns) = measurement.ns {
        format_ns(ns)
    } else {
        measurement
            .error
            .clone()
            .unwrap_or_else(|| "n/a".to_string())
    }
}

fn render_ns(measurement: &Measurement) -> String {
    measurement.ns.map_or_else(String::new, |ns| ns.to_string())
}

fn format_ns(ns: u128) -> String {
    if ns >= 1_000_000 {
        format!("{:.3} ms", ns as f64 / 1_000_000.0)
    } else if ns >= 1_000 {
        format!("{:.3} µs", ns as f64 / 1_000.0)
    } else {
        format!("{ns} ns")
    }
}

fn escape_csv(raw: &str) -> String {
    raw.replace('"', "\"\"")
}
