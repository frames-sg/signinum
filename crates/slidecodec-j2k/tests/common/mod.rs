// SPDX-License-Identifier: Apache-2.0

use slidecodec_core::{Downscale, PixelFormat, Rect};
use slidecodec_j2k::J2kDecoder;
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::{
        atomic::{AtomicUsize, Ordering},
        OnceLock,
    },
};

pub(crate) mod in_process;

pub(crate) fn bench_fixture_rgb() -> Option<Vec<u8>> {
    let pixels = gradient_u8(128, 128, 3);
    openjpeg_encode_jp2("in_process_parity_rgb", &pixels, 128, 128)
}

pub(crate) fn slidecodec_rgb(bytes: &[u8]) -> Vec<u8> {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    let dims = decoder.info().dimensions;
    let mut out = vec![0_u8; dims.0 as usize * dims.1 as usize * 3];
    decoder
        .decode_into(&mut out, dims.0 as usize * 3, PixelFormat::Rgb8)
        .expect("decode");
    out
}

pub(crate) fn slidecodec_rgb_region(bytes: &[u8], roi: Rect) -> Vec<u8> {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    let mut out = vec![0_u8; roi.w as usize * roi.h as usize * 3];
    decoder
        .decode_region_into(
            &mut slidecodec_j2k::J2kScratchPool::new(),
            &mut out,
            roi.w as usize * 3,
            PixelFormat::Rgb8,
            roi,
        )
        .expect("region decode");
    out
}

pub(crate) fn slidecodec_rgb_scaled_q4(bytes: &[u8]) -> Vec<u8> {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    let dims = decoder.info().dimensions;
    let scaled = (dims.0.div_ceil(4), dims.1.div_ceil(4));
    let mut out = vec![0_u8; scaled.0 as usize * scaled.1 as usize * 3];
    decoder
        .decode_scaled_into(
            &mut slidecodec_j2k::J2kScratchPool::new(),
            &mut out,
            scaled.0 as usize * 3,
            PixelFormat::Rgb8,
            Downscale::Quarter,
        )
        .expect("scaled decode");
    out
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

fn openjpeg_encode_jp2(name: &str, pixels: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    let bin = openjpeg_compress_bin()?;
    let dir = openjpeg_temp_dir();
    let unique = next_temp_suffix();
    let src_path = dir.join(format!("{name}-{unique}.ppm"));
    let out_path = dir.join(format!("{name}-{unique}.jp2"));
    write_ppm(&src_path, pixels, width, height).ok()?;
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

fn next_temp_suffix() -> usize {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

fn openjpeg_compress_bin() -> Option<PathBuf> {
    static OPENJPEG_COMPRESS: OnceLock<Option<PathBuf>> = OnceLock::new();
    OPENJPEG_COMPRESS
        .get_or_init(|| {
            if let Some(path) = std::env::var_os("SLIDECODEC_OPENJPEG_COMPRESS_BIN") {
                let path = PathBuf::from(path);
                if path.exists() {
                    return Some(path);
                }
            }
            let default = PathBuf::from("/opt/homebrew/bin/opj_compress");
            default.exists().then_some(default)
        })
        .clone()
}

fn openjpeg_temp_dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join("slidecodec-j2k-openjpeg-tests");
        fs::create_dir_all(&dir).expect("create openjpeg temp dir");
        dir
    })
    .clone()
}

fn write_ppm(
    path: &std::path::Path,
    pixels: &[u8],
    width: u32,
    height: u32,
) -> std::io::Result<()> {
    let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
    bytes.extend_from_slice(pixels);
    fs::write(path, bytes)
}
