// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

use criterion::black_box;
use rayon::prelude::*;
use signinum_core::{
    BackendRequest, DeviceSubmission, ImageDecodeDevice, TileBatchDecodeDevice,
    TileBatchDecodeSubmit,
};
use signinum_j2k::{
    decode_tiles_into, decode_tiles_region_scaled_into, CompressedTransferSyntax, DecoderContext,
    Downscale, J2kContext, J2kDecoder, J2kScratchPool, PixelFormat, Rect, TileBatchOptions,
    TileDecodeJob, TileRegionScaledDecodeJob,
};
use signinum_j2k_compare::{grok, openjpeg};
use signinum_j2k_metal::{
    extract_dicom_encapsulated_frames_with_limit, Codec as MetalJ2kCodec,
    J2kDecoder as MetalJ2kDecoder, J2kScratchPool as MetalJ2kScratchPool, MetalSession,
};
use signinum_j2k_native::{encode, encode_htj2k, EncodeOptions};
use std::{
    collections::BTreeSet,
    env, fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecodeMode {
    Gray8,
    Gray16,
    Rgb8,
}

#[derive(Clone, Debug)]
pub(crate) struct BenchInput {
    pub name: &'static str,
    pub bytes: Vec<u8>,
    pub dimensions: (u32, u32),
    pub mode: DecodeMode,
    pub is_ht: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct ExternalTileBatch {
    pub name: String,
    pub inputs: Vec<Vec<u8>>,
    pub dimensions: Vec<(u32, u32)>,
    pub mode: DecodeMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExternalCodecFamily {
    J2k,
    Htj2k,
    Unknown,
}

const AUTO_REPEATED_GRAYSCALE_MIN_DIM: u32 = 512;
const AUTO_REPEATED_GRAYSCALE_MIN_COUNT: usize = 16;
const EXTERNAL_WSI_TILE_DIR_ENV: &str = "SIGNINUM_J2K_METAL_WSI_TILE_DIR";
const J2K_TILE_BATCH_SIZES_ENV: &str = "SIGNINUM_J2K_TILE_BATCH_SIZES";
const DEFAULT_J2K_TILE_BATCH_SIZES: &[usize] = &[16, 32, 64, 128];

pub(crate) fn j2k_tile_batch_sizes() -> Vec<usize> {
    env::var(J2K_TILE_BATCH_SIZES_ENV)
        .ok()
        .map_or_else(default_j2k_tile_batch_sizes, |raw| {
            let parsed = raw
                .split(',')
                .filter_map(|value| value.trim().parse::<usize>().ok())
                .filter(|&value| value > 0)
                .collect::<Vec<_>>();
            if parsed.is_empty() {
                default_j2k_tile_batch_sizes()
            } else {
                parsed
            }
        })
}

fn default_j2k_tile_batch_sizes() -> Vec<usize> {
    DEFAULT_J2K_TILE_BATCH_SIZES.to_vec()
}

pub(crate) fn bench_inputs() -> Vec<BenchInput> {
    let mut inputs = vec![
        BenchInput {
            name: "j2k_gray_1024",
            bytes: classic_bench_bytes(
                "j2k_gray_1024",
                &gradient_u8(1024, 1024, 1),
                1024,
                1024,
                DecodeMode::Gray8,
            ),
            dimensions: (1024, 1024),
            mode: DecodeMode::Gray8,
            is_ht: false,
        },
        BenchInput {
            name: "j2k_gray_512",
            bytes: classic_bench_bytes(
                "j2k_gray_512",
                &gradient_u8(512, 512, 1),
                512,
                512,
                DecodeMode::Gray8,
            ),
            dimensions: (512, 512),
            mode: DecodeMode::Gray8,
            is_ht: false,
        },
        BenchInput {
            name: "j2k_rgb_1024",
            bytes: classic_bench_bytes(
                "j2k_rgb_1024",
                &gradient_u8(1024, 1024, 3),
                1024,
                1024,
                DecodeMode::Rgb8,
            ),
            dimensions: (1024, 1024),
            mode: DecodeMode::Rgb8,
            is_ht: false,
        },
        BenchInput {
            name: "j2k_rgb_256",
            bytes: classic_bench_bytes(
                "j2k_rgb_256",
                &gradient_u8(256, 256, 3),
                256,
                256,
                DecodeMode::Rgb8,
            ),
            dimensions: (256, 256),
            mode: DecodeMode::Rgb8,
            is_ht: false,
        },
    ];

    inputs.extend(ht_bench_inputs());

    inputs
}

pub(crate) fn external_wsi_tile_batches(max_count: usize) -> Vec<ExternalTileBatch> {
    let Some(root) = external_tile_root() else {
        eprintln!(
            "skipping external WSI J2K tile benchmarks: set {EXTERNAL_WSI_TILE_DIR_ENV} to a directory of JP2/J2K/JPH/JHC tiles or DICOM WSI files"
        );
        return Vec::new();
    };

    let mut paths = collect_external_tile_paths(&root);
    if paths.is_empty() {
        eprintln!(
            "skipping external WSI J2K tile benchmarks: no JP2/J2K/JPH/JHC tiles or DICOM WSI files found under {}",
            root.display()
        );
        return Vec::new();
    }
    paths.sort();

    let mut j2k_gray8 = Vec::new();
    let mut j2k_gray16 = Vec::new();
    let mut j2k_rgb8 = Vec::new();
    let mut htj2k_gray8 = Vec::new();
    let mut htj2k_gray16 = Vec::new();
    let mut htj2k_rgb8 = Vec::new();
    let mut unknown_gray8 = Vec::new();
    let mut unknown_gray16 = Vec::new();
    let mut unknown_rgb8 = Vec::new();
    for path in paths {
        let Ok(source_bytes) = fs::read(&path) else {
            eprintln!(
                "skipping unreadable external WSI source: {}",
                path.display()
            );
            continue;
        };
        let frames = external_source_frames(&path, source_bytes, max_count);
        for bytes in frames {
            let Ok(info) = J2kDecoder::inspect(&bytes) else {
                eprintln!(
                    "skipping unsupported external J2K tile/frame: {}",
                    path.display()
                );
                continue;
            };
            let Some(mode) = external_decode_mode(info.components, info.bit_depth) else {
                continue;
            };
            let family = external_codec_family(&bytes);
            let entry = (bytes, info.dimensions);
            match (family, mode) {
                (ExternalCodecFamily::J2k, DecodeMode::Gray8) => j2k_gray8.push(entry),
                (ExternalCodecFamily::J2k, DecodeMode::Gray16) => j2k_gray16.push(entry),
                (ExternalCodecFamily::J2k, DecodeMode::Rgb8) => j2k_rgb8.push(entry),
                (ExternalCodecFamily::Htj2k, DecodeMode::Gray8) => htj2k_gray8.push(entry),
                (ExternalCodecFamily::Htj2k, DecodeMode::Gray16) => htj2k_gray16.push(entry),
                (ExternalCodecFamily::Htj2k, DecodeMode::Rgb8) => htj2k_rgb8.push(entry),
                (ExternalCodecFamily::Unknown, DecodeMode::Gray8) => unknown_gray8.push(entry),
                (ExternalCodecFamily::Unknown, DecodeMode::Gray16) => unknown_gray16.push(entry),
                (ExternalCodecFamily::Unknown, DecodeMode::Rgb8) => unknown_rgb8.push(entry),
            }
        }
    }

    [
        external_tile_batch(&root, "j2k_gray8", DecodeMode::Gray8, j2k_gray8, max_count),
        external_tile_batch(
            &root,
            "j2k_gray16",
            DecodeMode::Gray16,
            j2k_gray16,
            max_count,
        ),
        external_tile_batch(&root, "j2k_rgb8", DecodeMode::Rgb8, j2k_rgb8, max_count),
        external_tile_batch(
            &root,
            "htj2k_gray8",
            DecodeMode::Gray8,
            htj2k_gray8,
            max_count,
        ),
        external_tile_batch(
            &root,
            "htj2k_gray16",
            DecodeMode::Gray16,
            htj2k_gray16,
            max_count,
        ),
        external_tile_batch(&root, "htj2k_rgb8", DecodeMode::Rgb8, htj2k_rgb8, max_count),
        external_tile_batch(
            &root,
            "j2k_unknown_gray8",
            DecodeMode::Gray8,
            unknown_gray8,
            max_count,
        ),
        external_tile_batch(
            &root,
            "j2k_unknown_gray16",
            DecodeMode::Gray16,
            unknown_gray16,
            max_count,
        ),
        external_tile_batch(
            &root,
            "j2k_unknown_rgb8",
            DecodeMode::Rgb8,
            unknown_rgb8,
            max_count,
        ),
    ]
    .into_iter()
    .flatten()
    .collect()
}

fn external_tile_root() -> Option<PathBuf> {
    let raw = env::var_os(EXTERNAL_WSI_TILE_DIR_ENV)?;
    if raw.is_empty() {
        return None;
    }
    let root = PathBuf::from(raw);
    if root.is_dir() {
        Some(root)
    } else {
        eprintln!(
            "skipping external WSI J2K tile benchmarks: {} is not a directory",
            root.display()
        );
        None
    }
}

fn collect_external_tile_paths(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            eprintln!(
                "skipping unreadable external tile directory: {}",
                dir.display()
            );
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if is_external_wsi_source_path(&path) {
                out.push(path);
            }
        }
    }
    out
}

fn is_external_wsi_source_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "jp2" | "j2k" | "j2c" | "jpc" | "jph" | "jhc" | "dcm" | "dicom"
            )
        })
}

fn external_source_frames(path: &Path, bytes: Vec<u8>, max_count: usize) -> Vec<Vec<u8>> {
    if is_dicom_path(path) {
        match extract_dicom_encapsulated_frames_with_limit(&bytes, max_count) {
            Ok(frames) => frames,
            Err(error) => {
                eprintln!(
                    "skipping external DICOM WSI source {}: {error}",
                    path.display()
                );
                Vec::new()
            }
        }
    } else {
        vec![bytes]
    }
}

fn is_dicom_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "dcm" | "dicom"))
}

fn external_decode_mode(components: u8, bit_depth: u8) -> Option<DecodeMode> {
    match (components, bit_depth) {
        (1, 1..=8) => Some(DecodeMode::Gray8),
        (1, 9..=16) => Some(DecodeMode::Gray16),
        (3, 1..=8) => Some(DecodeMode::Rgb8),
        _ => None,
    }
}

fn external_codec_family(bytes: &[u8]) -> ExternalCodecFamily {
    let Ok(decoder) = J2kDecoder::new(bytes) else {
        return ExternalCodecFamily::Unknown;
    };
    let Some(candidate) = decoder.passthrough_candidate() else {
        return ExternalCodecFamily::Unknown;
    };
    match candidate.transfer_syntax() {
        CompressedTransferSyntax::Jpeg2000Lossless | CompressedTransferSyntax::Jpeg2000Lossy => {
            ExternalCodecFamily::J2k
        }
        CompressedTransferSyntax::HtJpeg2000Lossless
        | CompressedTransferSyntax::HtJpeg2000Lossy => ExternalCodecFamily::Htj2k,
        _ => ExternalCodecFamily::Unknown,
    }
}

fn external_tile_batch(
    root: &Path,
    label: &str,
    mode: DecodeMode,
    entries: Vec<(Vec<u8>, (u32, u32))>,
    max_count: usize,
) -> Option<ExternalTileBatch> {
    if entries.is_empty() || max_count == 0 {
        return None;
    }

    let count = entries.len().min(max_count);
    let (inputs, dimensions): (Vec<_>, Vec<_>) = entries.into_iter().take(count).unzip();
    let distinct_dims = dimensions.iter().copied().collect::<BTreeSet<_>>().len();
    let root_name = root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("external_wsi");
    Some(ExternalTileBatch {
        name: format!("{root_name}_{label}_{count}tiles_{distinct_dims}dims"),
        inputs,
        dimensions,
        mode,
    })
}

fn ht_bench_inputs() -> Vec<BenchInput> {
    let candidates = [
        ("htj2k_gray_1024", 1024_u32, 1024_u32),
        ("htj2k_gray_512", 512_u32, 512_u32),
    ];

    let mut inputs = Vec::with_capacity(candidates.len());
    let mut errors = Vec::new();
    for (name, width, height) in candidates {
        let pixels = ht_bench_pixels(width, height, 1);
        match try_encode_ht(&pixels, width, height, 1, 8) {
            Ok(codestream) => inputs.push(BenchInput {
                name,
                bytes: wrap_codestream_jp2(&codestream, width, height, 1, 8, 17),
                dimensions: (width, height),
                mode: DecodeMode::Gray8,
                is_ht: true,
            }),
            Err(error) => errors.push(format!("{name}: {error}")),
        }
    }

    if inputs.is_empty() {
        eprintln!(
            "skipping HTJ2K bench inputs: {}",
            errors
                .last()
                .map_or("no HTJ2K benchmark candidate succeeded", String::as_str)
        );
    }
    inputs
}

fn ht_bench_pixels(width: u32, height: u32, channels: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(width as usize * height as usize * channels);
    let width_denom = width.saturating_sub(1).max(1);
    let height_denom = height.saturating_sub(1).max(1);
    for y in 0..height {
        let y_base = (y * 29) / height_denom;
        for x in 0..width {
            let x_base = (x * 31) / width_denom;
            for c in 0..channels {
                out.push((x_base + y_base + c as u32 * 17) as u8);
            }
        }
    }
    out
}

pub(crate) fn signinum_inspect(bytes: &[u8]) {
    black_box(J2kDecoder::inspect(bytes).expect("signinum inspect"));
}

pub(crate) fn signinum_decode(bytes: &[u8], mode: DecodeMode) {
    let mut decoder = J2kDecoder::new(bytes).expect("signinum decoder");
    let info = decoder.info().dimensions;
    let (fmt, stride) = mode_geometry(mode, info);
    let mut out = vec![0_u8; stride * info.1 as usize];
    decoder
        .decode_into(&mut out, stride, fmt)
        .expect("signinum decode");
    black_box(out);
}

pub(crate) fn signinum_decode_region(bytes: &[u8], mode: DecodeMode, edge: u32) {
    let mut decoder = J2kDecoder::new(bytes).expect("signinum decoder");
    let roi = centered_roi(decoder.info().dimensions, edge);
    let fmt = mode_format(mode);
    let stride = roi.w as usize * fmt.bytes_per_pixel();
    let mut pool = J2kScratchPool::new();
    let mut out = vec![0_u8; stride * roi.h as usize];
    decoder
        .decode_region_into(&mut pool, &mut out, stride, fmt, roi)
        .expect("signinum region decode");
    black_box(out);
}

pub(crate) fn signinum_decode_scaled(bytes: &[u8], mode: DecodeMode, scale: Downscale) {
    let mut decoder = J2kDecoder::new(bytes).expect("signinum decoder");
    let dims = scaled_dims(decoder.info().dimensions, scale);
    let fmt = mode_format(mode);
    let stride = dims.0 as usize * fmt.bytes_per_pixel();
    let mut pool = J2kScratchPool::new();
    let mut out = vec![0_u8; stride * dims.1 as usize];
    decoder
        .decode_scaled_into(&mut pool, &mut out, stride, fmt, scale)
        .expect("signinum scaled decode");
    black_box(out);
}

pub(crate) fn signinum_decode_region_scaled(
    bytes: &[u8],
    mode: DecodeMode,
    edge: u32,
    scale: Downscale,
) {
    let mut decoder = J2kDecoder::new(bytes).expect("signinum decoder");
    let roi = centered_roi(decoder.info().dimensions, edge);
    let scaled = roi.scaled_covering(scale);
    let fmt = mode_format(mode);
    let stride = scaled.w as usize * fmt.bytes_per_pixel();
    let mut pool = J2kScratchPool::new();
    let mut out = vec![0_u8; stride * scaled.h as usize];
    decoder
        .decode_region_scaled_into(&mut pool, &mut out, stride, fmt, roi, scale)
        .expect("signinum region scaled decode");
    black_box(out);
}

pub(crate) fn signinum_decode_tile_batch(bytes: &[u8], mode: DecodeMode, count: usize) {
    let decoder = J2kDecoder::new(bytes).expect("signinum decoder");
    let dims = decoder.info().dimensions;
    let (fmt, stride) = mode_geometry(mode, dims);
    let mut outputs = (0..count)
        .map(|_| vec![0_u8; stride * dims.1 as usize])
        .collect::<Vec<_>>();
    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileDecodeJob {
                input: bytes,
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        decode_tiles_into(&mut jobs, fmt, TileBatchOptions::default()).expect("tile decode")
    };
    black_box((outputs, outcomes));
}

pub(crate) fn signinum_decode_tile_batch_region_scaled(
    bytes: &[u8],
    mode: DecodeMode,
    edge: u32,
    scale: Downscale,
    count: usize,
) {
    let decoder = J2kDecoder::new(bytes).expect("signinum decoder");
    let roi = centered_roi(decoder.info().dimensions, edge);
    let scaled = roi.scaled_covering(scale);
    let fmt = mode_format(mode);
    let stride = scaled.w as usize * fmt.bytes_per_pixel();
    let mut outputs = (0..count)
        .map(|_| vec![0_u8; stride * scaled.h as usize])
        .collect::<Vec<_>>();
    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileRegionScaledDecodeJob {
                input: bytes,
                out: out.as_mut_slice(),
                stride,
                roi,
                scale,
            })
            .collect::<Vec<_>>();
        decode_tiles_region_scaled_into(&mut jobs, fmt, TileBatchOptions::default())
            .expect("tile region scaled decode")
    };
    black_box((outputs, outcomes));
}

pub(crate) fn distinct_rgb_tile_batch_inputs(input: &BenchInput, count: usize) -> Vec<Vec<u8>> {
    assert_eq!(input.mode, DecodeMode::Rgb8);
    (0..count)
        .map(|index| {
            let name = format!("{}_distinct_{index}", input.name);
            classic_bench_bytes(
                &name,
                &gradient_variant_u8(input.dimensions.0, input.dimensions.1, 3, index as u32),
                input.dimensions.0,
                input.dimensions.1,
                input.mode,
            )
        })
        .collect()
}

pub(crate) fn distinct_gray_tile_batch_inputs(input: &BenchInput, count: usize) -> Vec<Vec<u8>> {
    assert_eq!(input.mode, DecodeMode::Gray8);
    (0..count)
        .map(|index| {
            let pixels =
                gradient_variant_u8(input.dimensions.0, input.dimensions.1, 1, index as u32);
            if input.is_ht {
                wrap_codestream_jp2(
                    &try_encode_ht(&pixels, input.dimensions.0, input.dimensions.1, 1, 8)
                        .expect("encode distinct HTJ2K grayscale benchmark tile"),
                    input.dimensions.0,
                    input.dimensions.1,
                    1,
                    8,
                    17,
                )
            } else {
                let name = format!("{}_distinct_{index}", input.name);
                classic_bench_bytes(
                    &name,
                    &pixels,
                    input.dimensions.0,
                    input.dimensions.1,
                    input.mode,
                )
            }
        })
        .collect()
}

pub(crate) fn signinum_decode_tile_batch_distinct(inputs: &[Vec<u8>], mode: DecodeMode) {
    let Some(first) = inputs.first() else {
        return;
    };
    let decoder = J2kDecoder::new(first).expect("signinum decoder");
    let dims = decoder.info().dimensions;
    let (fmt, stride) = mode_geometry(mode, dims);
    let mut outputs = inputs
        .iter()
        .map(|_| vec![0_u8; stride * dims.1 as usize])
        .collect::<Vec<_>>();
    let outcomes = {
        let mut jobs = inputs
            .iter()
            .zip(outputs.iter_mut())
            .map(|(bytes, out)| TileDecodeJob {
                input: bytes,
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        decode_tiles_into(&mut jobs, fmt, TileBatchOptions::default()).expect("tile decode")
    };
    black_box((outputs, outcomes));
}

pub(crate) fn signinum_decode_tile_batch_region_scaled_distinct(
    inputs: &[Vec<u8>],
    mode: DecodeMode,
    edge: u32,
    scale: Downscale,
) {
    let Some(first) = inputs.first() else {
        return;
    };
    let decoder = J2kDecoder::new(first).expect("signinum decoder");
    let roi = centered_roi(decoder.info().dimensions, edge);
    let scaled = roi.scaled_covering(scale);
    let fmt = mode_format(mode);
    let stride = scaled.w as usize * fmt.bytes_per_pixel();
    let mut outputs = inputs
        .iter()
        .map(|_| vec![0_u8; stride * scaled.h as usize])
        .collect::<Vec<_>>();
    let outcomes = {
        let mut jobs = inputs
            .iter()
            .zip(outputs.iter_mut())
            .map(|(bytes, out)| TileRegionScaledDecodeJob {
                input: bytes,
                out: out.as_mut_slice(),
                stride,
                roi,
                scale,
            })
            .collect::<Vec<_>>();
        decode_tiles_region_scaled_into(&mut jobs, fmt, TileBatchOptions::default())
            .expect("tile region scaled decode")
    };
    black_box((outputs, outcomes));
}

pub(crate) fn signinum_decode_external_tile_batch_region_scaled(
    batch: &ExternalTileBatch,
    count: usize,
    edge: u32,
    scale: Downscale,
) {
    let count = count.min(batch.inputs.len());
    if count == 0 {
        return;
    }

    let fmt = mode_format(batch.mode);
    let (rois, stride, height) = external_batch_output_geometry(batch, count, edge, scale, fmt);
    let mut outputs = (0..count)
        .map(|_| vec![0_u8; stride * height as usize])
        .collect::<Vec<_>>();
    let outcomes = {
        let mut jobs = batch
            .inputs
            .iter()
            .zip(rois.iter())
            .take(count)
            .zip(outputs.iter_mut())
            .map(|((bytes, roi), out)| TileRegionScaledDecodeJob {
                input: bytes,
                out: out.as_mut_slice(),
                stride,
                roi: *roi,
                scale,
            })
            .collect::<Vec<_>>();
        decode_tiles_region_scaled_into(&mut jobs, fmt, TileBatchOptions::default())
            .expect("external tile region scaled decode")
    };
    black_box((outputs, outcomes));
}

pub(crate) fn openjpeg_decode_external_tile_batch_region_scaled(
    batch: &ExternalTileBatch,
    count: usize,
    edge: u32,
    scale: Downscale,
) {
    let count = count.min(batch.inputs.len());
    if count == 0 {
        return;
    }

    let reduce = downscale_reduction_factor(scale);
    let (rois, _, _) =
        external_batch_output_geometry(batch, count, edge, scale, mode_format(batch.mode));
    let outputs = batch
        .inputs
        .par_iter()
        .zip(rois.par_iter())
        .take(count)
        .map(|(bytes, roi)| match batch.mode {
            DecodeMode::Gray8 => openjpeg::decode_gray_region_scaled(bytes, *roi, reduce),
            DecodeMode::Rgb8 => openjpeg::decode_rgb_region_scaled(bytes, *roi, reduce),
            DecodeMode::Gray16 => {
                Err("openjpeg: Gray16 benchmark output is not implemented".to_string())
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("OpenJPEG external tile region scaled decode");
    black_box(outputs);
}

pub(crate) fn grok_decode_external_tile_batch_region_scaled(
    batch: &ExternalTileBatch,
    count: usize,
    edge: u32,
    scale: Downscale,
) {
    let count = count.min(batch.inputs.len());
    if count == 0 {
        return;
    }

    let reduce = downscale_reduction_factor(scale);
    let (rois, _, _) =
        external_batch_output_geometry(batch, count, edge, scale, mode_format(batch.mode));
    let outputs = batch
        .inputs
        .par_iter()
        .zip(rois.par_iter())
        .take(count)
        .map(|(bytes, roi)| match batch.mode {
            DecodeMode::Gray8 => grok::decode_gray_region_scaled(bytes, *roi, reduce),
            DecodeMode::Rgb8 => grok::decode_rgb_region_scaled(bytes, *roi, reduce),
            DecodeMode::Gray16 => {
                Err("grok: Gray16 benchmark output is not implemented".to_string())
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("Grok external tile region scaled decode");
    black_box(outputs);
}

pub(crate) fn metal_available() -> bool {
    cfg!(target_os = "macos")
}

pub(crate) fn signinum_metal_decode(bytes: &[u8], mode: DecodeMode) {
    let mut decoder = MetalJ2kDecoder::new(bytes).expect("signinum metal decoder");
    let surface = decoder
        .decode_to_device(mode_format(mode), BackendRequest::Metal)
        .expect("signinum metal decode");
    black_box(surface);
}

pub(crate) fn signinum_adaptive_decode(bytes: &[u8], mode: DecodeMode) {
    signinum_decode(bytes, mode);
}

pub(crate) fn signinum_metal_supports_decode(bytes: &[u8], mode: DecodeMode) -> bool {
    let mut decoder = MetalJ2kDecoder::new(bytes).expect("signinum metal decoder");
    decoder
        .decode_to_device(mode_format(mode), BackendRequest::Metal)
        .is_ok()
}

pub(crate) fn signinum_metal_decode_region(bytes: &[u8], mode: DecodeMode, edge: u32) {
    let cpu_decoder = J2kDecoder::new(bytes).expect("signinum decoder");
    let roi = centered_roi(cpu_decoder.info().dimensions, edge);
    let mut decoder = MetalJ2kDecoder::new(bytes).expect("signinum metal decoder");
    let surface = decoder
        .decode_region_to_device(mode_format(mode), roi, BackendRequest::Metal)
        .expect("signinum metal region decode");
    black_box(surface);
}

pub(crate) fn signinum_adaptive_decode_region(bytes: &[u8], mode: DecodeMode, edge: u32) {
    signinum_decode_region(bytes, mode, edge);
}

pub(crate) fn signinum_metal_supports_region(bytes: &[u8], mode: DecodeMode, edge: u32) -> bool {
    let cpu_decoder = J2kDecoder::new(bytes).expect("signinum decoder");
    let roi = centered_roi(cpu_decoder.info().dimensions, edge);
    let mut decoder = MetalJ2kDecoder::new(bytes).expect("signinum metal decoder");
    decoder
        .decode_region_to_device(mode_format(mode), roi, BackendRequest::Metal)
        .is_ok()
}

pub(crate) fn signinum_metal_decode_scaled(bytes: &[u8], mode: DecodeMode, scale: Downscale) {
    let mut decoder = MetalJ2kDecoder::new(bytes).expect("signinum metal decoder");
    let surface = decoder
        .decode_scaled_to_device(mode_format(mode), scale, BackendRequest::Metal)
        .expect("signinum metal scaled decode");
    black_box(surface);
}

pub(crate) fn signinum_adaptive_decode_scaled(bytes: &[u8], mode: DecodeMode, scale: Downscale) {
    signinum_decode_scaled(bytes, mode, scale);
}

pub(crate) fn signinum_metal_decode_region_scaled(
    bytes: &[u8],
    mode: DecodeMode,
    edge: u32,
    scale: Downscale,
) {
    let cpu_decoder = J2kDecoder::new(bytes).expect("signinum decoder");
    let roi = centered_roi(cpu_decoder.info().dimensions, edge);
    let mut decoder = MetalJ2kDecoder::new(bytes).expect("signinum metal decoder");
    let surface = decoder
        .decode_region_scaled_to_device(mode_format(mode), roi, scale, BackendRequest::Metal)
        .expect("signinum metal region scaled decode");
    black_box(surface);
}

pub(crate) fn signinum_adaptive_decode_region_scaled(
    bytes: &[u8],
    mode: DecodeMode,
    edge: u32,
    scale: Downscale,
) {
    signinum_decode_region_scaled(bytes, mode, edge, scale);
}

pub(crate) fn signinum_metal_supports_scaled(
    bytes: &[u8],
    mode: DecodeMode,
    scale: Downscale,
) -> bool {
    let mut decoder = MetalJ2kDecoder::new(bytes).expect("signinum metal decoder");
    decoder
        .decode_scaled_to_device(mode_format(mode), scale, BackendRequest::Metal)
        .is_ok()
}

pub(crate) fn signinum_metal_supports_region_scaled(
    bytes: &[u8],
    mode: DecodeMode,
    edge: u32,
    scale: Downscale,
) -> bool {
    if !supports_metal_region_scaled_mode(mode) {
        return false;
    }
    let cpu_decoder = J2kDecoder::new(bytes).expect("signinum decoder");
    let roi = centered_roi(cpu_decoder.info().dimensions, edge);
    let mut decoder = MetalJ2kDecoder::new(bytes).expect("signinum metal decoder");
    decoder
        .decode_region_scaled_to_device(mode_format(mode), roi, scale, BackendRequest::Metal)
        .is_ok()
}

pub(crate) fn signinum_metal_decode_tile_batch(bytes: &[u8], mode: DecodeMode, count: usize) {
    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut session = MetalSession::default();
    let mut pool = MetalJ2kScratchPool::new();
    let submissions = (0..count)
        .map(|_| {
            MetalJ2kCodec::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                bytes,
                mode_format(mode),
                BackendRequest::Metal,
            )
            .expect("signinum metal tile submit")
        })
        .collect::<Vec<_>>();
    let surfaces = submissions
        .into_iter()
        .map(|submission| submission.wait().expect("signinum metal tile decode"))
        .collect::<Vec<_>>();
    black_box(surfaces);
}

pub(crate) fn signinum_metal_decode_tile_batch_region_scaled(
    bytes: &[u8],
    mode: DecodeMode,
    edge: u32,
    scale: Downscale,
    count: usize,
) {
    let cpu_decoder = J2kDecoder::new(bytes).expect("signinum decoder");
    let roi = centered_roi(cpu_decoder.info().dimensions, edge);
    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut session = MetalSession::default();
    let mut pool = MetalJ2kScratchPool::new();
    let submissions = (0..count)
        .map(|_| {
            MetalJ2kCodec::submit_tile_region_scaled_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                bytes,
                mode_format(mode),
                roi,
                scale,
                BackendRequest::Metal,
            )
            .expect("signinum metal tile region scaled submit")
        })
        .collect::<Vec<_>>();
    let surfaces = submissions
        .into_iter()
        .map(|submission| {
            submission
                .wait()
                .expect("signinum metal tile region scaled decode")
        })
        .collect::<Vec<_>>();
    black_box(surfaces);
}

pub(crate) fn signinum_metal_decode_tile_batch_distinct(inputs: &[Vec<u8>], mode: DecodeMode) {
    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut session = MetalSession::default();
    let mut pool = MetalJ2kScratchPool::new();
    let submissions = inputs
        .iter()
        .map(|bytes| {
            MetalJ2kCodec::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                bytes,
                mode_format(mode),
                BackendRequest::Metal,
            )
            .expect("signinum metal tile submit")
        })
        .collect::<Vec<_>>();
    let surfaces = submissions
        .into_iter()
        .map(|submission| submission.wait().expect("signinum metal tile decode"))
        .collect::<Vec<_>>();
    black_box(surfaces);
}

pub(crate) fn signinum_metal_decode_tile_batch_region_scaled_distinct(
    inputs: &[Vec<u8>],
    mode: DecodeMode,
    edge: u32,
    scale: Downscale,
) {
    let Some(first) = inputs.first() else {
        return;
    };
    let cpu_decoder = J2kDecoder::new(first).expect("signinum decoder");
    let roi = centered_roi(cpu_decoder.info().dimensions, edge);
    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut session = MetalSession::default();
    let mut pool = MetalJ2kScratchPool::new();
    let submissions = inputs
        .iter()
        .map(|bytes| {
            MetalJ2kCodec::submit_tile_region_scaled_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                bytes,
                mode_format(mode),
                roi,
                scale,
                BackendRequest::Metal,
            )
            .expect("signinum metal tile region scaled submit")
        })
        .collect::<Vec<_>>();
    let surfaces = submissions
        .into_iter()
        .map(|submission| {
            submission
                .wait()
                .expect("signinum metal tile region scaled decode")
        })
        .collect::<Vec<_>>();
    black_box(surfaces);
}

pub(crate) fn signinum_metal_decode_external_tile_batch_region_scaled(
    batch: &ExternalTileBatch,
    count: usize,
    edge: u32,
    scale: Downscale,
) {
    let count = count.min(batch.inputs.len());
    if count == 0 {
        return;
    }

    let fmt = mode_format(batch.mode);
    let (rois, _, _) = external_batch_output_geometry(batch, count, edge, scale, fmt);
    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut session = MetalSession::default();
    let mut pool = MetalJ2kScratchPool::new();
    let submissions = batch
        .inputs
        .iter()
        .zip(rois.iter())
        .take(count)
        .map(|(bytes, roi)| {
            MetalJ2kCodec::submit_tile_region_scaled_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                bytes,
                fmt,
                *roi,
                scale,
                BackendRequest::Metal,
            )
            .expect("signinum metal external tile region scaled submit")
        })
        .collect::<Vec<_>>();
    let surfaces = submissions
        .into_iter()
        .map(|submission| {
            submission
                .wait()
                .expect("signinum metal external tile region scaled decode")
        })
        .collect::<Vec<_>>();
    black_box(surfaces);
}

fn signinum_adaptive_decode_tile_batch_to_device(input: &BenchInput, count: usize) {
    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut session = MetalSession::default();
    let mut pool = MetalJ2kScratchPool::new();
    let submissions = (0..count)
        .map(|_| {
            MetalJ2kCodec::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                &input.bytes,
                mode_format(input.mode),
                BackendRequest::Auto,
            )
            .expect("signinum auto tile submit")
        })
        .collect::<Vec<_>>();
    let surfaces = submissions
        .into_iter()
        .map(|submission| submission.wait().expect("signinum auto tile decode"))
        .collect::<Vec<_>>();
    black_box(surfaces);
}

fn signinum_adaptive_decode_tile_batch_region_scaled_to_device(
    input: &BenchInput,
    edge: u32,
    scale: Downscale,
    count: usize,
) {
    let decoder = J2kDecoder::new(&input.bytes).expect("signinum decoder");
    let roi = centered_roi(decoder.info().dimensions, edge);
    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut session = MetalSession::default();
    let mut pool = MetalJ2kScratchPool::new();
    let submissions = (0..count)
        .map(|_| {
            MetalJ2kCodec::submit_tile_region_scaled_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                &input.bytes,
                mode_format(input.mode),
                roi,
                scale,
                BackendRequest::Auto,
            )
            .expect("signinum auto tile region scaled submit")
        })
        .collect::<Vec<_>>();
    let surfaces = submissions
        .into_iter()
        .map(|submission| {
            submission
                .wait()
                .expect("signinum auto tile region scaled decode")
        })
        .collect::<Vec<_>>();
    black_box(surfaces);
}

pub(crate) fn signinum_adaptive_decode_tile_batch(input: &BenchInput, count: usize) {
    #[cfg(target_os = "macos")]
    if should_auto_use_direct_grayscale_input(input, count) {
        signinum_adaptive_decode_tile_batch_to_device(input, count);
        return;
    }

    signinum_decode_tile_batch(&input.bytes, input.mode, count);
}

pub(crate) fn signinum_adaptive_decode_tile_batch_region_scaled(
    input: &BenchInput,
    edge: u32,
    scale: Downscale,
    count: usize,
) {
    signinum_adaptive_decode_tile_batch_region_scaled_to_device(input, edge, scale, count);
}

pub(crate) fn signinum_adaptive_decode_tile_batch_region_scaled_distinct(
    inputs: &[Vec<u8>],
    mode: DecodeMode,
    edge: u32,
    scale: Downscale,
) {
    let Some(first) = inputs.first() else {
        return;
    };
    let decoder = J2kDecoder::new(first).expect("signinum decoder");
    let roi = centered_roi(decoder.info().dimensions, edge);
    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut session = MetalSession::default();
    let mut pool = MetalJ2kScratchPool::new();
    let submissions = inputs
        .iter()
        .map(|bytes| {
            MetalJ2kCodec::submit_tile_region_scaled_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                bytes,
                mode_format(mode),
                roi,
                scale,
                BackendRequest::Auto,
            )
            .expect("signinum auto distinct tile region scaled submit")
        })
        .collect::<Vec<_>>();
    let surfaces = submissions
        .into_iter()
        .map(|submission| {
            submission
                .wait()
                .expect("signinum auto distinct tile region scaled decode")
        })
        .collect::<Vec<_>>();
    black_box(surfaces);
}

pub(crate) fn signinum_adaptive_decode_external_tile_batch_region_scaled(
    batch: &ExternalTileBatch,
    count: usize,
    edge: u32,
    scale: Downscale,
) {
    let count = count.min(batch.inputs.len());
    if count == 0 {
        return;
    }

    let fmt = mode_format(batch.mode);
    let (rois, _, _) = external_batch_output_geometry(batch, count, edge, scale, fmt);
    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut session = MetalSession::default();
    let mut pool = MetalJ2kScratchPool::new();
    let submissions = batch
        .inputs
        .iter()
        .zip(rois.iter())
        .take(count)
        .map(|(bytes, roi)| {
            MetalJ2kCodec::submit_tile_region_scaled_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                bytes,
                fmt,
                *roi,
                scale,
                BackendRequest::Auto,
            )
            .expect("signinum auto external tile region scaled submit")
        })
        .collect::<Vec<_>>();
    let surfaces = submissions
        .into_iter()
        .map(|submission| {
            submission
                .wait()
                .expect("signinum auto external tile region scaled decode")
        })
        .collect::<Vec<_>>();
    black_box(surfaces);
}

fn should_auto_use_direct_grayscale_input(input: &BenchInput, count: usize) -> bool {
    if !matches!(input.mode, DecodeMode::Gray8 | DecodeMode::Gray16) || count == 0 {
        return false;
    }
    if input.dimensions.0.max(input.dimensions.1) < AUTO_REPEATED_GRAYSCALE_MIN_DIM {
        return false;
    }
    count >= AUTO_REPEATED_GRAYSCALE_MIN_COUNT
}

pub(crate) fn signinum_metal_supports_tile_batch(bytes: &[u8], mode: DecodeMode) -> bool {
    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut pool = MetalJ2kScratchPool::new();
    MetalJ2kCodec::decode_tile_to_device(
        &mut ctx,
        &mut pool,
        bytes,
        mode_format(mode),
        BackendRequest::Metal,
    )
    .is_ok()
}

pub(crate) fn signinum_metal_supports_tile_batch_region_scaled(
    bytes: &[u8],
    mode: DecodeMode,
    edge: u32,
    scale: Downscale,
) -> bool {
    let cpu_decoder = J2kDecoder::new(bytes).expect("signinum decoder");
    let roi = centered_roi(cpu_decoder.info().dimensions, edge);
    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut pool = MetalJ2kScratchPool::new();
    MetalJ2kCodec::decode_tile_region_scaled_to_device(
        &mut ctx,
        &mut pool,
        bytes,
        mode_format(mode),
        roi,
        scale,
        BackendRequest::Metal,
    )
    .is_ok()
}

pub(crate) fn signinum_metal_supports_tile_batch_distinct(
    inputs: &[Vec<u8>],
    mode: DecodeMode,
) -> bool {
    inputs
        .iter()
        .all(|bytes| signinum_metal_supports_tile_batch(bytes, mode))
}

pub(crate) fn signinum_metal_supports_tile_batch_region_scaled_distinct(
    inputs: &[Vec<u8>],
    mode: DecodeMode,
    edge: u32,
    scale: Downscale,
) -> bool {
    if !supports_metal_region_scaled_mode(mode) {
        return false;
    }
    let Some(first) = inputs.first() else {
        return true;
    };
    let cpu_decoder = J2kDecoder::new(first).expect("signinum decoder");
    let roi = centered_roi(cpu_decoder.info().dimensions, edge);
    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut pool = MetalJ2kScratchPool::new();
    inputs.iter().all(|bytes| {
        MetalJ2kCodec::decode_tile_region_scaled_to_device(
            &mut ctx,
            &mut pool,
            bytes,
            mode_format(mode),
            roi,
            scale,
            BackendRequest::Metal,
        )
        .is_ok()
    })
}

pub(crate) fn signinum_metal_supports_external_tile_batch_region_scaled(
    batch: &ExternalTileBatch,
    count: usize,
    edge: u32,
    scale: Downscale,
) -> bool {
    if !supports_metal_region_scaled_mode(batch.mode) {
        return false;
    }
    let count = count.min(batch.inputs.len());
    if count == 0 {
        return true;
    }

    let fmt = mode_format(batch.mode);
    let (rois, _, _) = external_batch_output_geometry(batch, count, edge, scale, fmt);
    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut pool = MetalJ2kScratchPool::new();
    batch
        .inputs
        .iter()
        .zip(rois.iter())
        .take(count)
        .all(|(bytes, roi)| {
            MetalJ2kCodec::decode_tile_region_scaled_to_device(
                &mut ctx,
                &mut pool,
                bytes,
                fmt,
                *roi,
                scale,
                BackendRequest::Metal,
            )
            .is_ok()
        })
}

pub(crate) fn openjpeg_supports_external_tile_batch_region_scaled(
    batch: &ExternalTileBatch,
    count: usize,
) -> bool {
    matches!(batch.mode, DecodeMode::Gray8 | DecodeMode::Rgb8)
        && batch
            .inputs
            .iter()
            .take(count.min(batch.inputs.len()))
            .all(|bytes| matches!(external_codec_family(bytes), ExternalCodecFamily::J2k))
}

pub(crate) fn grok_supports_external_tile_batch_region_scaled(
    batch: &ExternalTileBatch,
    count: usize,
) -> bool {
    grok::is_available()
        && matches!(batch.mode, DecodeMode::Gray8 | DecodeMode::Rgb8)
        && batch
            .inputs
            .iter()
            .take(count.min(batch.inputs.len()))
            .all(|bytes| {
                matches!(
                    external_codec_family(bytes),
                    ExternalCodecFamily::J2k | ExternalCodecFamily::Htj2k
                )
            })
}

fn encode_j2k(pixels: &[u8], width: u32, height: u32, components: u8, bit_depth: u8) -> Vec<u8> {
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        guard_bits: 2,
        ..EncodeOptions::default()
    };
    encode(
        pixels, width, height, components, bit_depth, false, &options,
    )
    .expect("encode")
}

fn try_encode_ht(
    pixels: &[u8],
    width: u32,
    height: u32,
    components: u8,
    bit_depth: u8,
) -> Result<Vec<u8>, String> {
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        guard_bits: 2,
        ..EncodeOptions::default()
    };
    encode_htj2k(
        pixels, width, height, components, bit_depth, false, &options,
    )
    .map_err(std::string::ToString::to_string)
}

fn classic_bench_bytes(
    _name: &str,
    pixels: &[u8],
    width: u32,
    height: u32,
    mode: DecodeMode,
) -> Vec<u8> {
    let (components, colorspace) = match mode {
        DecodeMode::Gray8 | DecodeMode::Gray16 => (1_u16, 17_u32),
        DecodeMode::Rgb8 => (3_u16, 16_u32),
    };
    wrap_codestream_jp2(
        &encode_j2k(pixels, width, height, components as u8, 8),
        width,
        height,
        components,
        8,
        colorspace,
    )
}

fn gradient_u8(width: u32, height: u32, channels: usize) -> Vec<u8> {
    gradient_variant_u8(width, height, channels, 0)
}

fn gradient_variant_u8(width: u32, height: u32, channels: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(width as usize * height as usize * channels);
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                out.push(((x + y + seed * 13 + (c as u32 * 17)) & 0xFF) as u8);
            }
        }
    }
    out
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

pub(crate) fn centered_roi(dims: (u32, u32), edge: u32) -> Rect {
    let w = edge.min(dims.0);
    let h = edge.min(dims.1);
    Rect {
        x: (dims.0 - w) / 2,
        y: (dims.1 - h) / 2,
        w,
        h,
    }
}

fn external_batch_output_geometry(
    batch: &ExternalTileBatch,
    count: usize,
    edge: u32,
    scale: Downscale,
    fmt: PixelFormat,
) -> (Vec<Rect>, usize, u32) {
    let rois = batch
        .dimensions
        .iter()
        .take(count)
        .map(|&dims| centered_roi(dims, edge))
        .collect::<Vec<_>>();
    let (max_width, max_height) = rois
        .iter()
        .map(|roi| {
            let scaled = roi.scaled_covering(scale);
            (scaled.w, scaled.h)
        })
        .fold((0_u32, 0_u32), |(max_w, max_h), (w, h)| {
            (max_w.max(w), max_h.max(h))
        });
    (rois, max_width as usize * fmt.bytes_per_pixel(), max_height)
}

fn downscale_reduction_factor(scale: Downscale) -> u32 {
    scale.denominator().trailing_zeros()
}

fn mode_format(mode: DecodeMode) -> PixelFormat {
    match mode {
        DecodeMode::Gray8 => PixelFormat::Gray8,
        DecodeMode::Gray16 => PixelFormat::Gray16,
        DecodeMode::Rgb8 => PixelFormat::Rgb8,
    }
}

fn supports_metal_region_scaled_mode(mode: DecodeMode) -> bool {
    matches!(mode, DecodeMode::Gray8 | DecodeMode::Gray16)
}

fn mode_geometry(mode: DecodeMode, dims: (u32, u32)) -> (PixelFormat, usize) {
    let fmt = mode_format(mode);
    (fmt, dims.0 as usize * fmt.bytes_per_pixel())
}

fn scaled_dims(dims: (u32, u32), scale: Downscale) -> (u32, u32) {
    let denom = scale.denominator();
    (dims.0.div_ceil(denom), dims.1.div_ceil(denom))
}
