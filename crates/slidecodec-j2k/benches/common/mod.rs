// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

use criterion::black_box;
use slidecodec_core::{
    BackendRequest, DeviceSubmission, ImageDecodeDevice, TileBatchDecodeDevice,
    TileBatchDecodeSubmit,
};
use slidecodec_j2k::{
    DecoderContext, Downscale, J2kCodec, J2kContext, J2kDecoder, J2kScratchPool, PixelFormat, Rect,
    TileBatchDecode,
};
use slidecodec_j2k_compare::{grok as grok_compare, openjpeg as openjpeg_compare};
use slidecodec_j2k_metal::{
    Codec as MetalJ2kCodec, J2kDecoder as MetalJ2kDecoder, J2kScratchPool as MetalJ2kScratchPool,
    MetalSession,
};
use slidecodec_j2k_native::{encode, encode_htj2k, EncodeOptions};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecodeMode {
    Gray8,
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

    match ht_bench_input() {
        Ok(input) => inputs.push(input),
        Err(error) => eprintln!("skipping HTJ2K bench input: {error}"),
    }

    inputs
}

fn ht_bench_input() -> Result<BenchInput, String> {
    let candidates = [
        ("htj2k_gray_1024", 1024_u32, 1024_u32),
        ("htj2k_gray_512", 512_u32, 512_u32),
        ("htj2k_gray_256", 256_u32, 256_u32),
        ("htj2k_gray_128", 128_u32, 128_u32),
        ("htj2k_gray_64", 64_u32, 64_u32),
        ("htj2k_gray_8", 8_u32, 8_u32),
    ];

    let mut last_error = None;
    for (name, width, height) in candidates {
        let pixels = ht_bench_pixels(width, height, 1);
        match try_encode_ht(&pixels, width, height, 1, 8) {
            Ok(codestream) => {
                return Ok(BenchInput {
                    name,
                    bytes: wrap_codestream_jp2(&codestream, width, height, 1, 8, 17),
                    dimensions: (width, height),
                    mode: DecodeMode::Gray8,
                    is_ht: true,
                })
            }
            Err(error) => last_error = Some(format!("{name}: {error}")),
        }
    }

    Err(last_error.unwrap_or_else(|| "no HTJ2K benchmark candidate succeeded".to_string()))
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

pub(crate) fn slidecodec_inspect(bytes: &[u8]) {
    black_box(J2kDecoder::inspect(bytes).expect("slidecodec inspect"));
}

pub(crate) fn slidecodec_decode(bytes: &[u8], mode: DecodeMode) {
    let mut decoder = J2kDecoder::new(bytes).expect("slidecodec decoder");
    let info = decoder.info().dimensions;
    let (fmt, stride) = mode_geometry(mode, info);
    let mut out = vec![0_u8; stride * info.1 as usize];
    decoder
        .decode_into(&mut out, stride, fmt)
        .expect("slidecodec decode");
    black_box(out);
}

pub(crate) fn slidecodec_decode_region(bytes: &[u8], mode: DecodeMode, edge: u32) {
    let mut decoder = J2kDecoder::new(bytes).expect("slidecodec decoder");
    let roi = centered_roi(decoder.info().dimensions, edge);
    let fmt = mode_format(mode);
    let stride = roi.w as usize * fmt.bytes_per_pixel();
    let mut pool = J2kScratchPool::new();
    let mut out = vec![0_u8; stride * roi.h as usize];
    decoder
        .decode_region_into(&mut pool, &mut out, stride, fmt, roi)
        .expect("slidecodec region decode");
    black_box(out);
}

pub(crate) fn slidecodec_decode_scaled(bytes: &[u8], mode: DecodeMode, scale: Downscale) {
    let mut decoder = J2kDecoder::new(bytes).expect("slidecodec decoder");
    let dims = scaled_dims(decoder.info().dimensions, scale);
    let fmt = mode_format(mode);
    let stride = dims.0 as usize * fmt.bytes_per_pixel();
    let mut pool = J2kScratchPool::new();
    let mut out = vec![0_u8; stride * dims.1 as usize];
    decoder
        .decode_scaled_into(&mut pool, &mut out, stride, fmt, scale)
        .expect("slidecodec scaled decode");
    black_box(out);
}

pub(crate) fn slidecodec_decode_tile_batch(bytes: &[u8], mode: DecodeMode, count: usize) {
    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut pool = J2kScratchPool::new();
    let decoder = J2kDecoder::new(bytes).expect("slidecodec decoder");
    let dims = decoder.info().dimensions;
    let (fmt, stride) = mode_geometry(mode, dims);
    let mut out = vec![0_u8; stride * dims.1 as usize];
    for _ in 0..count {
        J2kCodec::decode_tile(&mut ctx, &mut pool, bytes, &mut out, stride, fmt)
            .expect("tile decode");
    }
    black_box(out);
}

pub(crate) fn metal_available() -> bool {
    cfg!(target_os = "macos")
}

pub(crate) fn slidecodec_metal_decode(bytes: &[u8], mode: DecodeMode) {
    let mut decoder = MetalJ2kDecoder::new(bytes).expect("slidecodec metal decoder");
    let surface = decoder
        .decode_to_device(mode_format(mode), BackendRequest::Metal)
        .expect("slidecodec metal decode");
    black_box(surface);
}

pub(crate) fn slidecodec_adaptive_decode(bytes: &[u8], mode: DecodeMode) {
    slidecodec_decode(bytes, mode);
}

pub(crate) fn slidecodec_metal_supports_decode(bytes: &[u8], mode: DecodeMode) -> bool {
    let mut decoder = MetalJ2kDecoder::new(bytes).expect("slidecodec metal decoder");
    decoder
        .decode_to_device(mode_format(mode), BackendRequest::Metal)
        .is_ok()
}

pub(crate) fn slidecodec_metal_decode_region(bytes: &[u8], mode: DecodeMode, edge: u32) {
    let cpu_decoder = J2kDecoder::new(bytes).expect("slidecodec decoder");
    let roi = centered_roi(cpu_decoder.info().dimensions, edge);
    let mut decoder = MetalJ2kDecoder::new(bytes).expect("slidecodec metal decoder");
    let surface = decoder
        .decode_region_to_device(mode_format(mode), roi, BackendRequest::Metal)
        .expect("slidecodec metal region decode");
    black_box(surface);
}

pub(crate) fn slidecodec_adaptive_decode_region(bytes: &[u8], mode: DecodeMode, edge: u32) {
    slidecodec_decode_region(bytes, mode, edge);
}

pub(crate) fn slidecodec_metal_supports_region(bytes: &[u8], mode: DecodeMode, edge: u32) -> bool {
    let cpu_decoder = J2kDecoder::new(bytes).expect("slidecodec decoder");
    let roi = centered_roi(cpu_decoder.info().dimensions, edge);
    let mut decoder = MetalJ2kDecoder::new(bytes).expect("slidecodec metal decoder");
    decoder
        .decode_region_to_device(mode_format(mode), roi, BackendRequest::Metal)
        .is_ok()
}

pub(crate) fn slidecodec_metal_decode_scaled(bytes: &[u8], mode: DecodeMode, scale: Downscale) {
    let mut decoder = MetalJ2kDecoder::new(bytes).expect("slidecodec metal decoder");
    let surface = decoder
        .decode_scaled_to_device(mode_format(mode), scale, BackendRequest::Metal)
        .expect("slidecodec metal scaled decode");
    black_box(surface);
}

pub(crate) fn slidecodec_adaptive_decode_scaled(bytes: &[u8], mode: DecodeMode, scale: Downscale) {
    slidecodec_decode_scaled(bytes, mode, scale);
}

pub(crate) fn slidecodec_metal_supports_scaled(
    bytes: &[u8],
    mode: DecodeMode,
    scale: Downscale,
) -> bool {
    let mut decoder = MetalJ2kDecoder::new(bytes).expect("slidecodec metal decoder");
    decoder
        .decode_scaled_to_device(mode_format(mode), scale, BackendRequest::Metal)
        .is_ok()
}

pub(crate) fn slidecodec_metal_decode_tile_batch(bytes: &[u8], mode: DecodeMode, count: usize) {
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
            .expect("slidecodec metal tile submit")
        })
        .collect::<Vec<_>>();
    let surfaces = submissions
        .into_iter()
        .map(|submission| submission.wait().expect("slidecodec metal tile decode"))
        .collect::<Vec<_>>();
    black_box(surfaces);
}

fn slidecodec_adaptive_decode_tile_batch_to_device(input: &BenchInput, count: usize) {
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
            .expect("slidecodec auto tile submit")
        })
        .collect::<Vec<_>>();
    let surfaces = submissions
        .into_iter()
        .map(|submission| submission.wait().expect("slidecodec auto tile decode"))
        .collect::<Vec<_>>();
    black_box(surfaces);
}

pub(crate) fn slidecodec_adaptive_decode_tile_batch(input: &BenchInput, count: usize) {
    #[cfg(target_os = "macos")]
    if should_auto_use_direct_grayscale_input(input, count) {
        slidecodec_adaptive_decode_tile_batch_to_device(input, count);
        return;
    }

    slidecodec_decode_tile_batch(&input.bytes, input.mode, count);
}

fn should_auto_use_direct_grayscale_input(input: &BenchInput, count: usize) -> bool {
    if input.mode != DecodeMode::Gray8 || count == 0 {
        return false;
    }
    if input.dimensions.0.max(input.dimensions.1) < 1024 {
        return false;
    }
    count >= 16
}

pub(crate) fn slidecodec_metal_supports_tile_batch(bytes: &[u8], mode: DecodeMode) -> bool {
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

pub(crate) fn openjpeg_available() -> bool {
    openjpeg_compare::is_available()
}

pub(crate) fn grok_available() -> bool {
    grok_compare::is_available()
}

pub(crate) fn openjpeg_decode(
    input: &BenchInput,
    reduce: Option<u32>,
    region: Option<Rect>,
    batch: usize,
) {
    for _ in 0..batch {
        let out = match (input.mode, reduce, region) {
            (DecodeMode::Gray8, None, None) => openjpeg_compare::decode_gray(&input.bytes),
            (DecodeMode::Gray8, Some(reduce), None) => {
                openjpeg_compare::decode_gray_scaled(&input.bytes, reduce)
            }
            (DecodeMode::Gray8, None, Some(region)) => {
                openjpeg_compare::decode_gray_region(&input.bytes, region)
            }
            (DecodeMode::Rgb8, None, None) => openjpeg_compare::decode_rgb(&input.bytes),
            (DecodeMode::Rgb8, Some(reduce), None) => {
                openjpeg_compare::decode_rgb_scaled(&input.bytes, reduce)
            }
            (DecodeMode::Rgb8, None, Some(region)) => {
                openjpeg_compare::decode_rgb_region(&input.bytes, region)
            }
            _ => panic!("unsupported OpenJPEG bench shape"),
        }
        .expect("openjpeg decode");
        black_box(out);
    }
}

pub(crate) fn grok_decode(
    input: &BenchInput,
    reduce: Option<u32>,
    region: Option<Rect>,
    batch: usize,
) {
    for _ in 0..batch {
        let out = match (input.mode, reduce, region) {
            (DecodeMode::Gray8, None, None) => grok_compare::decode_gray(&input.bytes),
            (DecodeMode::Gray8, Some(reduce), None) => {
                grok_compare::decode_gray_scaled(&input.bytes, reduce)
            }
            (DecodeMode::Gray8, None, Some(region)) => {
                grok_compare::decode_gray_region(&input.bytes, region)
            }
            (DecodeMode::Rgb8, None, None) => grok_compare::decode_rgb(&input.bytes),
            (DecodeMode::Rgb8, Some(reduce), None) => {
                grok_compare::decode_rgb_scaled(&input.bytes, reduce)
            }
            (DecodeMode::Rgb8, None, Some(region)) => {
                grok_compare::decode_rgb_region(&input.bytes, region)
            }
            _ => panic!("unsupported Grok bench shape"),
        }
        .expect("grok decode");
        black_box(out);
    }
}

fn encode_j2k(pixels: &[u8], width: u32, height: u32, components: u8, bit_depth: u8) -> Vec<u8> {
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
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
        ..EncodeOptions::default()
    };
    encode_htj2k(
        pixels, width, height, components, bit_depth, false, &options,
    )
    .map_err(std::string::ToString::to_string)
}

fn classic_bench_bytes(
    name: &str,
    pixels: &[u8],
    width: u32,
    height: u32,
    mode: DecodeMode,
) -> Vec<u8> {
    if let Some(bytes) = openjpeg_encode_jp2(name, pixels, width, height, mode) {
        return bytes;
    }
    let (components, colorspace) = match mode {
        DecodeMode::Gray8 => (1_u16, 17_u32),
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
    let mut out = Vec::with_capacity(width as usize * height as usize * channels);
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                out.push(((x + y + (c as u32 * 17)) & 0xFF) as u8);
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

fn mode_format(mode: DecodeMode) -> PixelFormat {
    match mode {
        DecodeMode::Gray8 => PixelFormat::Gray8,
        DecodeMode::Rgb8 => PixelFormat::Rgb8,
    }
}

fn mode_geometry(mode: DecodeMode, dims: (u32, u32)) -> (PixelFormat, usize) {
    let fmt = mode_format(mode);
    (fmt, dims.0 as usize * fmt.bytes_per_pixel())
}

fn scaled_dims(dims: (u32, u32), scale: Downscale) -> (u32, u32) {
    let denom = scale.denominator();
    (dims.0.div_ceil(denom), dims.1.div_ceil(denom))
}

fn openjpeg_compress_bin() -> Option<PathBuf> {
    static OPENJPEG_COMPRESS: OnceLock<Option<PathBuf>> = OnceLock::new();
    OPENJPEG_COMPRESS
        .get_or_init(|| {
            if let Some(path) = std::env::var_os("SLIDECODEC_OPENJPEG_COMPRESS_BIN") {
                return Some(PathBuf::from(path));
            }
            let default = PathBuf::from("/opt/homebrew/bin/opj_compress");
            if default.exists() {
                return Some(default);
            }
            None
        })
        .clone()
}

fn openjpeg_encode_jp2(
    name: &str,
    pixels: &[u8],
    width: u32,
    height: u32,
    mode: DecodeMode,
) -> Option<Vec<u8>> {
    let bin = openjpeg_compress_bin()?;
    let dir = openjpeg_temp_dir();
    let src_path = dir.join(match mode {
        DecodeMode::Gray8 => format!("{name}.pgm"),
        DecodeMode::Rgb8 => format!("{name}.ppm"),
    });
    let out_path = dir.join(format!("{name}.jp2"));
    write_pnm(&src_path, pixels, width, height, mode).ok()?;
    let status = Command::new(bin)
        .arg("-i")
        .arg(&src_path)
        .arg("-o")
        .arg(&out_path)
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    fs::read(out_path).ok()
}

fn openjpeg_temp_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("slidecodec-j2k-bench-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("create OpenJPEG temp dir");
        dir
    })
}

fn write_pnm(
    path: &Path,
    pixels: &[u8],
    width: u32,
    height: u32,
    mode: DecodeMode,
) -> std::io::Result<()> {
    let mut bytes = Vec::new();
    match mode {
        DecodeMode::Gray8 => {
            bytes.extend_from_slice(format!("P5\n{width} {height}\n255\n").as_bytes());
        }
        DecodeMode::Rgb8 => {
            bytes.extend_from_slice(format!("P6\n{width} {height}\n255\n").as_bytes());
        }
    }
    bytes.extend_from_slice(pixels);
    fs::write(path, bytes)
}
