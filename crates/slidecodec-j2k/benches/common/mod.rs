// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

use criterion::black_box;
use dicom_toolkit_jpeg2000::{encode, encode_htj2k, EncodeOptions};
use slidecodec_j2k::{
    DecoderContext, Downscale, J2kCodec, J2kContext, J2kDecoder, J2kScratchPool, PixelFormat, Rect,
    TileBatchDecode,
};
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
    vec![
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
        BenchInput {
            name: "htj2k_gray_512",
            bytes: wrap_codestream_jp2(
                &encode_ht(&gradient_u8(512, 512, 1), 512, 512, 1, 8),
                512,
                512,
                1,
                8,
                17,
            ),
            dimensions: (512, 512),
            mode: DecodeMode::Gray8,
            is_ht: true,
        },
    ]
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

pub(crate) fn openjpeg_available() -> bool {
    openjpeg_bin().is_some() && openjpeg_compress_bin().is_some()
}

pub(crate) fn openjpeg_decode(
    input: &BenchInput,
    reduce: Option<u32>,
    region: Option<Rect>,
    batch: usize,
) {
    let harness = OpenJpegHarness::for_input(input);
    for _ in 0..batch {
        harness.run(reduce, region);
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

fn encode_ht(pixels: &[u8], width: u32, height: u32, components: u8, bit_depth: u8) -> Vec<u8> {
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        ..EncodeOptions::default()
    };
    encode_htj2k(
        pixels, width, height, components, bit_depth, false, &options,
    )
    .expect("encode ht")
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

fn openjpeg_bin() -> Option<PathBuf> {
    static OPENJPEG: OnceLock<Option<PathBuf>> = OnceLock::new();
    OPENJPEG
        .get_or_init(|| {
            if let Some(path) = std::env::var_os("SLIDECODEC_OPENJPEG_BIN") {
                return Some(PathBuf::from(path));
            }
            let default = PathBuf::from("/opt/homebrew/bin/opj_decompress");
            if default.exists() {
                return Some(default);
            }
            None
        })
        .clone()
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

struct OpenJpegHarness {
    bin: PathBuf,
    input_path: PathBuf,
    output_path: PathBuf,
    mode: DecodeMode,
}

impl OpenJpegHarness {
    fn for_input(input: &BenchInput) -> Self {
        let bin = openjpeg_bin().expect("OpenJPEG binary");
        let dir = openjpeg_temp_dir();
        let input_path = dir.join(format!("{}.jp2", input.name));
        let output_path = dir.join(match input.mode {
            DecodeMode::Gray8 => format!("{}.pgm", input.name),
            DecodeMode::Rgb8 => format!("{}.ppm", input.name),
        });
        fs::write(&input_path, &input.bytes).expect("write benchmark input");
        Self {
            bin,
            input_path,
            output_path,
            mode: input.mode,
        }
    }

    fn run(&self, reduce: Option<u32>, region: Option<Rect>) {
        let mut command = Command::new(&self.bin);
        command.arg("-i").arg(&self.input_path);
        command.arg("-o").arg(&self.output_path);
        command.arg("-quiet");
        if let Some(reduce) = reduce {
            command.arg("-r").arg(reduce.to_string());
        }
        if let Some(region) = region {
            command.arg("-d").arg(format!(
                "{},{},{},{}",
                region.x,
                region.y,
                region.x + region.w,
                region.y + region.h
            ));
        }
        if matches!(self.mode, DecodeMode::Rgb8) {
            command.arg("-force-rgb");
        }
        let status = command.status().expect("run openjpeg");
        assert!(status.success(), "OpenJPEG decode failed");
        black_box(
            fs::metadata(&self.output_path)
                .expect("openjpeg output metadata")
                .len(),
        );
    }
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
