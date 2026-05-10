// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Instant;

use signinum_j2k::{decode_tiles_into, J2kDecoder, PixelFormat, TileBatchOptions, TileDecodeJob};
use signinum_j2k_compare::{grok, openjpeg};

const DEFAULT_BATCH_SIZES: &[usize] = &[1, 16, 512, 1024];
const DEFAULT_REPEATS: usize = 3;

#[derive(Clone)]
struct TileInput {
    path: PathBuf,
    bytes: Vec<u8>,
    dimensions: (u32, u32),
    format: PixelFormat,
}

struct Measurement {
    decoder: &'static str,
    batch_size: usize,
    repeats: usize,
    sample_ms: Vec<f64>,
    median_ms: f64,
    mean_ms: f64,
    tiles_per_second_median: f64,
    decoded_bytes_per_repeat: usize,
}

#[derive(Clone, Copy)]
enum ExternalDecoder {
    OpenJpeg,
    Grok,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        return Err("usage: jp2k_batch_compare <raw-tile-dir> [batch-size ...]".to_string());
    }

    let tile_dir = PathBuf::from(&args[0]);
    if !tile_dir.is_dir() {
        return Err(format!(
            "raw tile path is not a directory: {}",
            tile_dir.display()
        ));
    }
    let batch_sizes = if args.len() > 1 {
        args[1..]
            .iter()
            .map(|value| parse_positive_usize(value, "batch size"))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        DEFAULT_BATCH_SIZES.to_vec()
    };
    let max_batch_size = batch_sizes
        .iter()
        .copied()
        .max()
        .ok_or_else(|| "no batch sizes requested".to_string())?;
    let repeats = std::env::var("SIGNINUM_J2K_BATCH_COMPARE_REPEATS")
        .ok()
        .map(|value| parse_positive_usize(&value, "SIGNINUM_J2K_BATCH_COMPARE_REPEATS"))
        .transpose()?
        .unwrap_or(DEFAULT_REPEATS);
    let workers = std::env::var("SIGNINUM_J2K_BATCH_COMPARE_THREADS")
        .ok()
        .map(|value| parse_positive_usize(&value, "SIGNINUM_J2K_BATCH_COMPARE_THREADS"))
        .transpose()?
        .map(|value| NonZeroUsize::new(value).expect("positive value was validated"));

    let (tiles, skipped) = load_tiles(&tile_dir, max_batch_size)?;
    if tiles.len() < max_batch_size {
        return Err(format!(
            "only loaded {} supported tiles from {}; need {max_batch_size}; skipped {skipped}",
            tiles.len(),
            tile_dir.display()
        ));
    }
    let format = tiles[0].format;
    if !tiles.iter().all(|tile| tile.format == format) {
        return Err("selected tiles do not share one output pixel format".to_string());
    }

    println!(
        "tile_dir\t{}\nloaded_tiles\t{}\nskipped_unsupported\t{}\nformat\t{:?}\nworkers\t{}\nopenjpeg_available\t{}\ngrok_available\t{}",
        tile_dir.display(),
        tiles.len(),
        skipped,
        format,
        workers.map_or_else(|| "auto".to_string(), |value| value.get().to_string()),
        openjpeg::is_available(),
        grok::is_available()
    );
    println!(
        "decoder\tbatch_size\trepeats\tmedian_ms\tmean_ms\ttiles_per_second_median\tdecoded_bytes_per_repeat\tsamples_ms"
    );

    for batch_size in batch_sizes {
        emit_measurement(measure_signinum(&tiles[..batch_size], repeats, workers)?);
        emit_measurement(measure_external(
            &tiles[..batch_size],
            repeats,
            workers,
            ExternalDecoder::OpenJpeg,
        )?);
        if grok::is_available() {
            emit_measurement(measure_external(
                &tiles[..batch_size],
                repeats,
                workers,
                ExternalDecoder::Grok,
            )?);
        } else {
            println!("grok\t{batch_size}\t{repeats}\tNA\tNA\tNA\tNA\tunavailable");
        }
    }

    Ok(())
}

fn emit_measurement(row: Measurement) {
    let Measurement {
        decoder,
        batch_size,
        repeats,
        sample_ms,
        median_ms,
        mean_ms,
        tiles_per_second_median,
        decoded_bytes_per_repeat,
    } = row;
    let samples = sample_ms
        .iter()
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "{decoder}\t{batch_size}\t{repeats}\t{median_ms:.6}\t{mean_ms:.6}\t{tiles_per_second_median:.3}\t{decoded_bytes_per_repeat}\t{samples}"
    );
}

fn load_tiles(dir: &Path, limit: usize) -> Result<(Vec<TileInput>, usize), String> {
    let mut paths = std::fs::read_dir(dir)
        .map_err(|err| format!("read tile dir {}: {err}", dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("read tile dir entry: {err}"))?;
    paths.retain(|path| {
        path.extension()
            .and_then(|value| value.to_str())
            .is_some_and(|extension| {
                matches!(
                    extension.to_ascii_lowercase().as_str(),
                    "j2k" | "j2c" | "jp2" | "jph" | "jhc"
                )
            })
    });
    paths.sort();

    let mut tiles = Vec::with_capacity(limit);
    let mut skipped = 0usize;
    for path in paths {
        if tiles.len() == limit {
            break;
        }
        let bytes =
            std::fs::read(&path).map_err(|err| format!("read {}: {err}", path.display()))?;
        let Ok(info) = J2kDecoder::inspect(&bytes) else {
            skipped += 1;
            continue;
        };
        let Some(format) = pixel_format(info.components, info.bit_depth) else {
            skipped += 1;
            continue;
        };
        tiles.push(TileInput {
            path,
            bytes,
            dimensions: info.dimensions,
            format,
        });
    }
    Ok((tiles, skipped))
}

fn pixel_format(components: u8, bit_depth: u8) -> Option<PixelFormat> {
    match (components, bit_depth) {
        (1, 8) => Some(PixelFormat::Gray8),
        (3, 8) => Some(PixelFormat::Rgb8),
        _ => None,
    }
}

fn measure_signinum(
    tiles: &[TileInput],
    repeats: usize,
    workers: Option<NonZeroUsize>,
) -> Result<Measurement, String> {
    let format = tiles[0].format;
    let mut samples = Vec::with_capacity(repeats);
    let mut decoded_bytes_per_repeat = decode_signinum_once(tiles, format, workers)?;
    std::hint::black_box(decoded_bytes_per_repeat);
    for _ in 0..repeats {
        let started = Instant::now();
        decoded_bytes_per_repeat = decode_signinum_once(tiles, format, workers)?;
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(decoded_bytes_per_repeat);
    }
    Ok(measurement(
        "signinum",
        tiles.len(),
        repeats,
        samples,
        decoded_bytes_per_repeat,
    ))
}

fn decode_signinum_once(
    tiles: &[TileInput],
    format: PixelFormat,
    workers: Option<NonZeroUsize>,
) -> Result<usize, String> {
    let mut outputs = tiles
        .iter()
        .map(|tile| vec![0_u8; output_len(tile, format)])
        .collect::<Vec<_>>();
    let mut jobs = tiles
        .iter()
        .zip(outputs.iter_mut())
        .map(|(tile, out)| TileDecodeJob {
            input: tile.bytes.as_slice(),
            out: out.as_mut_slice(),
            stride: stride(tile, format),
        })
        .collect::<Vec<_>>();
    decode_tiles_into(&mut jobs, format, TileBatchOptions { workers })
        .map_err(|err| format!("signinum batch decode failed: {err}"))?;
    Ok(outputs.iter().map(Vec::len).sum())
}

fn measure_external(
    tiles: &[TileInput],
    repeats: usize,
    workers: Option<NonZeroUsize>,
    decoder: ExternalDecoder,
) -> Result<Measurement, String> {
    let decoder_name = match decoder {
        ExternalDecoder::OpenJpeg => "openjpeg",
        ExternalDecoder::Grok => "grok",
    };
    let mut samples = Vec::with_capacity(repeats);
    let mut decoded_bytes_per_repeat = decode_external_once(tiles, workers, decoder)?;
    std::hint::black_box(decoded_bytes_per_repeat);
    for _ in 0..repeats {
        let started = Instant::now();
        decoded_bytes_per_repeat = decode_external_once(tiles, workers, decoder)?;
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(decoded_bytes_per_repeat);
    }
    Ok(measurement(
        decoder_name,
        tiles.len(),
        repeats,
        samples,
        decoded_bytes_per_repeat,
    ))
}

fn decode_external_once(
    tiles: &[TileInput],
    workers: Option<NonZeroUsize>,
    decoder: ExternalDecoder,
) -> Result<usize, String> {
    let worker_count = workers
        .map_or_else(
            || std::thread::available_parallelism().map_or(1, NonZeroUsize::get),
            NonZeroUsize::get,
        )
        .max(1)
        .min(tiles.len().max(1));
    let chunk_size = tiles.len().div_ceil(worker_count);
    let total_decoded = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for chunk in tiles.chunks(chunk_size) {
            handles.push(scope.spawn(move || {
                chunk
                    .iter()
                    .map(|tile| decode_external_tile(tile, decoder))
                    .try_fold(0usize, |acc, decoded| decoded.map(|data| acc + data.len()))
            }));
        }
        let mut decoded_bytes = 0usize;
        for handle in handles {
            match handle.join() {
                Ok(Ok(bytes)) => decoded_bytes += bytes,
                Ok(Err(err)) => return Err(err),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        Ok(decoded_bytes)
    })?;
    Ok(total_decoded)
}

fn decode_external_tile(tile: &TileInput, decoder: ExternalDecoder) -> Result<Vec<u8>, String> {
    let result = match (decoder, tile.format) {
        (ExternalDecoder::OpenJpeg, PixelFormat::Gray8) => openjpeg::decode_gray(&tile.bytes),
        (ExternalDecoder::OpenJpeg, PixelFormat::Rgb8) => openjpeg::decode_rgb(&tile.bytes),
        (ExternalDecoder::Grok, PixelFormat::Gray8) => grok::decode_gray(&tile.bytes),
        (ExternalDecoder::Grok, PixelFormat::Rgb8) => grok::decode_rgb(&tile.bytes),
        (_, other) => Err(format!(
            "{other:?} is not implemented for external comparator"
        )),
    };
    result.map_err(|err| format!("{}: {err}", tile.path.display()))
}

fn measurement(
    decoder: &'static str,
    batch_size: usize,
    repeats: usize,
    samples: Vec<f64>,
    decoded_bytes_per_repeat: usize,
) -> Measurement {
    let median_ms = median(samples.clone());
    let mean_ms = samples.iter().sum::<f64>() / usize_to_f64(samples.len());
    Measurement {
        decoder,
        batch_size,
        repeats,
        sample_ms: samples,
        median_ms,
        mean_ms,
        tiles_per_second_median: usize_to_f64(batch_size) / (median_ms / 1000.0),
        decoded_bytes_per_repeat,
    }
}

fn stride(tile: &TileInput, format: PixelFormat) -> usize {
    tile.dimensions.0 as usize * format.bytes_per_pixel()
}

fn output_len(tile: &TileInput, format: PixelFormat) -> usize {
    stride(tile, format) * tile.dimensions.1 as usize
}

fn parse_positive_usize(value: &str, label: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|err| format!("invalid {label} {value:?}: {err}"))?;
    if parsed == 0 {
        return Err(format!("{label} must be > 0"));
    }
    Ok(parsed)
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}
