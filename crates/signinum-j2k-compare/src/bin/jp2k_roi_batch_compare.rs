// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroUsize;
use std::time::Instant;

use signinum_core::{tile_batch_worker_count, Downscale, PixelFormat, Rect};
use signinum_j2k::{
    decode_tiles_region_scaled_into, encode_j2k_lossless, EncodeBackendPreference,
    J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples,
    TileBatchOptions, TileRegionScaledDecodeJob,
};
use signinum_j2k_compare::grok;

const DEFAULT_REPEATS: usize = 9;
const DEFAULT_BATCH_SIZE: usize = 16;

struct CompareCase {
    name: &'static str,
    bytes: Vec<u8>,
    roi: Rect,
    scale: Downscale,
    batch_size: usize,
}

struct Measurement {
    decoder: &'static str,
    case_name: &'static str,
    repeats: usize,
    batch_size: usize,
    median_us: f64,
    mean_us: f64,
    tiles_per_second_median: f64,
    decoded_bytes_per_repeat: usize,
    samples_us: Vec<f64>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    if !grok::is_available() {
        return Err(
            "in-process Grok is unavailable; install libgrokj2k or set SIGNINUM_GROK_SOURCE/SIGNINUM_GROK_ROOT"
                .to_string(),
        );
    }

    let repeats = std::env::var("SIGNINUM_J2K_ROI_COMPARE_REPEATS")
        .ok()
        .map(|value| parse_positive_usize(&value, "SIGNINUM_J2K_ROI_COMPARE_REPEATS"))
        .transpose()?
        .unwrap_or(DEFAULT_REPEATS);
    let workers = std::env::var("SIGNINUM_J2K_ROI_COMPARE_THREADS")
        .ok()
        .map(|value| parse_positive_usize(&value, "SIGNINUM_J2K_ROI_COMPARE_THREADS"))
        .transpose()?
        .map(|value| NonZeroUsize::new(value).expect("positive value was validated"));

    let cases = compare_cases()?;
    println!(
        "repeats\t{repeats}\nworkers\t{}\ngrok_available\t{}",
        workers.map_or_else(|| "auto".to_string(), |value| value.get().to_string()),
        grok::is_available()
    );
    println!(
        "decoder\tcase\tbatch_size\trepeats\tmedian_us\tmean_us\ttiles_per_second_median\tdecoded_bytes_per_repeat\tsamples_us"
    );

    for case in &cases {
        validate_case(case)?;
        emit_measurement(&measure_signinum(case, repeats, workers)?);
        emit_measurement(&measure_grok(case, repeats, workers)?);
    }
    Ok(())
}

fn compare_cases() -> Result<Vec<CompareCase>, String> {
    let raw_512 = encode_htj2k_rgb_codestream(512, 512)?;
    let jp2_512 = wrap_codestream_jp2(&raw_512, 512, 512, 3, 8, 16);
    let raw_256 = encode_htj2k_rgb_codestream(256, 256)?;
    let jp2_256 = wrap_codestream_jp2(&raw_256, 256, 256, 3, 8, 16);
    Ok(vec![
        CompareCase {
            name: "htj2k_raw_rgb8_512_roi256_q4_repeated_batch16",
            bytes: raw_512,
            roi: Rect {
                x: 128,
                y: 128,
                w: 256,
                h: 256,
            },
            scale: Downscale::Quarter,
            batch_size: DEFAULT_BATCH_SIZE,
        },
        CompareCase {
            name: "htj2k_jp2_rgb8_512_roi256_q4_repeated_batch16",
            bytes: jp2_512,
            roi: Rect {
                x: 128,
                y: 128,
                w: 256,
                h: 256,
            },
            scale: Downscale::Quarter,
            batch_size: DEFAULT_BATCH_SIZE,
        },
        CompareCase {
            name: "htj2k_jp2_rgb8_256_roi128_q4_repeated_batch16",
            bytes: jp2_256,
            roi: Rect {
                x: 64,
                y: 64,
                w: 128,
                h: 128,
            },
            scale: Downscale::Quarter,
            batch_size: DEFAULT_BATCH_SIZE,
        },
    ])
}

fn encode_htj2k_rgb_codestream(width: u32, height: u32) -> Result<Vec<u8>, String> {
    let pixels = patterned_rgb8(width, height);
    let samples = J2kLosslessSamples::new(&pixels, width, height, 3, 8, false)
        .map_err(|error| error.to_string())?;
    let options = J2kLosslessEncodeOptions {
        backend: EncodeBackendPreference::CpuOnly,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        max_decomposition_levels: Some(2),
        validation: J2kEncodeValidation::External,
        ..J2kLosslessEncodeOptions::default()
    };
    Ok(encode_j2k_lossless(samples, &options)
        .map_err(|error| error.to_string())?
        .codestream)
}

fn patterned_rgb8(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 3 + y * 5) & 0xff) as u8);
            pixels.push(((x * 7 + y * 11 + 17) & 0xff) as u8);
            pixels.push(((x * 13 + y * 19 + 31) & 0xff) as u8);
        }
    }
    pixels
}

fn validate_case(case: &CompareCase) -> Result<(), String> {
    let ours = decode_signinum_once(case)?;
    let theirs = decode_grok_once(case, None)?;
    if ours != theirs {
        return Err(format!(
            "{}: signinum/Grok ROI+scale output mismatch: {} vs {} bytes",
            case.name,
            ours.len(),
            theirs.len()
        ));
    }
    Ok(())
}

fn measure_signinum(
    case: &CompareCase,
    repeats: usize,
    workers: Option<NonZeroUsize>,
) -> Result<Measurement, String> {
    let mut samples = Vec::with_capacity(repeats);
    let mut decoded = decode_signinum_once(case)?;
    std::hint::black_box(&decoded);
    for _ in 0..repeats {
        let started = Instant::now();
        decoded = decode_signinum_once_with_workers(case, workers)?;
        samples.push(started.elapsed().as_secs_f64() * 1_000_000.0);
        std::hint::black_box(&decoded);
    }
    Ok(measurement(
        "signinum",
        case,
        repeats,
        samples,
        decoded.len(),
    ))
}

fn decode_signinum_once(case: &CompareCase) -> Result<Vec<u8>, String> {
    decode_signinum_once_with_workers(case, None)
}

fn decode_signinum_once_with_workers(
    case: &CompareCase,
    workers: Option<NonZeroUsize>,
) -> Result<Vec<u8>, String> {
    let output_len = output_len(case);
    let stride = output_stride(case);
    let mut outputs = vec![vec![0_u8; output_len]; case.batch_size];
    let mut jobs = outputs
        .iter_mut()
        .map(|out| TileRegionScaledDecodeJob {
            input: case.bytes.as_slice(),
            out: out.as_mut_slice(),
            stride,
            roi: case.roi,
            scale: case.scale,
        })
        .collect::<Vec<_>>();
    decode_tiles_region_scaled_into(&mut jobs, PixelFormat::Rgb8, TileBatchOptions { workers })
        .map_err(|error| error.to_string())?;
    Ok(outputs.into_iter().flatten().collect())
}

fn measure_grok(
    case: &CompareCase,
    repeats: usize,
    workers: Option<NonZeroUsize>,
) -> Result<Measurement, String> {
    let mut samples = Vec::with_capacity(repeats);
    let mut decoded = decode_grok_once(case, workers)?;
    std::hint::black_box(&decoded);
    for _ in 0..repeats {
        let started = Instant::now();
        decoded = decode_grok_once(case, workers)?;
        samples.push(started.elapsed().as_secs_f64() * 1_000_000.0);
        std::hint::black_box(&decoded);
    }
    Ok(measurement("grok", case, repeats, samples, decoded.len()))
}

fn decode_grok_once(case: &CompareCase, workers: Option<NonZeroUsize>) -> Result<Vec<u8>, String> {
    let worker_count = tile_batch_worker_count(
        case.batch_size,
        TileBatchOptions { workers },
        std::thread::available_parallelism().map_or(1, NonZeroUsize::get),
    );
    let chunk_size = case.batch_size.div_ceil(worker_count);
    let reduce = reduce_factor(case.scale)?;
    let chunks = (0..case.batch_size)
        .collect::<Vec<_>>()
        .chunks(chunk_size)
        .map(<[_]>::to_vec)
        .collect::<Vec<_>>();

    let outputs = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            handles.push(scope.spawn(move || {
                chunk
                    .iter()
                    .map(|_| grok::decode_rgb_region_scaled(&case.bytes, case.roi, reduce))
                    .collect::<Result<Vec<_>, _>>()
            }));
        }

        let mut outputs = Vec::with_capacity(case.batch_size);
        for handle in handles {
            match handle.join() {
                Ok(Ok(mut chunk_outputs)) => outputs.append(&mut chunk_outputs),
                Ok(Err(error)) => return Err(error),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        Ok(outputs)
    })?;
    Ok(outputs.into_iter().flatten().collect())
}

fn measurement(
    decoder: &'static str,
    case: &CompareCase,
    repeats: usize,
    samples: Vec<f64>,
    decoded_bytes_per_repeat: usize,
) -> Measurement {
    let median_us = median(samples.clone());
    let mean_us = samples.iter().sum::<f64>() / usize_to_f64(samples.len());
    Measurement {
        decoder,
        case_name: case.name,
        repeats,
        batch_size: case.batch_size,
        median_us,
        mean_us,
        tiles_per_second_median: usize_to_f64(case.batch_size) / (median_us / 1_000_000.0),
        decoded_bytes_per_repeat,
        samples_us: samples,
    }
}

fn emit_measurement(row: &Measurement) {
    let samples = row
        .samples_us
        .iter()
        .map(|value| format!("{value:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}",
        row.decoder,
        row.case_name,
        row.batch_size,
        row.repeats,
        row.median_us,
        row.mean_us,
        row.tiles_per_second_median,
        row.decoded_bytes_per_repeat,
        samples
    );
}

fn output_stride(case: &CompareCase) -> usize {
    case.roi.scaled_covering(case.scale).w as usize * PixelFormat::Rgb8.bytes_per_pixel()
}

fn output_len(case: &CompareCase) -> usize {
    let scaled = case.roi.scaled_covering(case.scale);
    output_stride(case) * scaled.h as usize
}

fn reduce_factor(scale: Downscale) -> Result<u32, String> {
    match scale {
        Downscale::None => Ok(0),
        Downscale::Half => Ok(1),
        Downscale::Quarter => Ok(2),
        Downscale::Eighth => Ok(3),
        _ => Err(format!("unsupported downscale for Grok compare: {scale:?}")),
    }
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn parse_positive_usize(value: &str, label: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {label} {value:?}: {error}"))?;
    if parsed == 0 {
        return Err(format!("{label} must be > 0"));
    }
    Ok(parsed)
}

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

fn wrap_codestream_jp2(
    codestream: &[u8],
    width: u32,
    height: u32,
    components: u16,
    bit_depth: u8,
    colorspace_enum: u32,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0D, 0x0A, 0x87, 0x0A]);
    bytes.extend_from_slice(&[
        0, 0, 0, 20, b'f', b't', b'y', b'p', b'j', b'p', b'2', b' ', 0, 0, 0, 0, b'j', b'p', b'2',
        b' ',
    ]);

    let bpc = bit_depth.saturating_sub(1);
    bytes.extend_from_slice(&[
        0, 0, 0, 45, b'j', b'p', b'2', b'h', 0, 0, 0, 22, b'i', b'h', b'd', b'r',
    ]);
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&components.to_be_bytes());
    bytes.extend_from_slice(&[bpc, 7, 0, 0]);
    bytes.extend_from_slice(&[0, 0, 0, 15, b'c', b'o', b'l', b'r', 1, 0, 0]);
    bytes.extend_from_slice(&colorspace_enum.to_be_bytes());

    let len = (8 + codestream.len()) as u32;
    bytes.extend_from_slice(&len.to_be_bytes());
    bytes.extend_from_slice(b"jp2c");
    bytes.extend_from_slice(codestream);
    bytes
}
