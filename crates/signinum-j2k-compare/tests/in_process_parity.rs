// SPDX-License-Identifier: Apache-2.0

use signinum_core::{Downscale, PixelFormat, Rect};
use signinum_j2k::J2kDecoder;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        atomic::{AtomicUsize, Ordering},
        OnceLock,
    },
};

#[test]
fn openjpeg_in_process_matches_signinum_rgb_fixture() {
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let ours = signinum_rgb(&input);
    let theirs = signinum_j2k_compare::openjpeg::decode_rgb(&input).expect("openjpeg");
    assert_eq!(ours, theirs);
}

#[test]
fn openjpeg_in_process_region_matches_signinum_rgb_fixture() {
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let roi = Rect {
        x: 16,
        y: 24,
        w: 64,
        h: 64,
    };
    let ours = signinum_rgb_region(&input, roi);
    let theirs = signinum_j2k_compare::openjpeg::decode_rgb_region(&input, roi).expect("openjpeg");
    assert_eq!(ours, theirs);
}

#[test]
fn grok_in_process_matches_signinum_rgb_fixture() {
    if !signinum_j2k_compare::grok::is_available() {
        assert!(
            !require_grok(),
            "SIGNINUM_REQUIRE_GROK is set but in-process Grok is unavailable"
        );
        return;
    }
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let ours = signinum_rgb(&input);
    let theirs = signinum_j2k_compare::grok::decode_rgb(&input).expect("grok");
    assert_eq!(ours, theirs);
}

#[test]
fn grok_in_process_scaled_matches_signinum_rgb_fixture() {
    if !signinum_j2k_compare::grok::is_available() {
        assert!(
            !require_grok(),
            "SIGNINUM_REQUIRE_GROK is set but in-process Grok is unavailable"
        );
        return;
    }
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let ours = signinum_rgb_scaled_q4(&input);
    let theirs = signinum_j2k_compare::grok::decode_rgb_scaled(&input, 2).expect("grok");
    assert_eq!(ours, theirs);
}

fn bench_fixture_rgb() -> Option<Vec<u8>> {
    let pixels = gradient_u8(128, 128, 3);
    openjpeg_encode_jp2("in_process_parity_rgb", &pixels, 128, 128)
}

fn signinum_rgb(bytes: &[u8]) -> Vec<u8> {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    let dims = decoder.info().dimensions;
    let mut out = vec![0_u8; dims.0 as usize * dims.1 as usize * 3];
    decoder
        .decode_into(&mut out, dims.0 as usize * 3, PixelFormat::Rgb8)
        .expect("decode");
    out
}

fn signinum_rgb_region(bytes: &[u8], roi: Rect) -> Vec<u8> {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    let mut out = vec![0_u8; roi.w as usize * roi.h as usize * 3];
    decoder
        .decode_region_into(
            &mut signinum_j2k::J2kScratchPool::new(),
            &mut out,
            roi.w as usize * 3,
            PixelFormat::Rgb8,
            roi,
        )
        .expect("region decode");
    out
}

fn signinum_rgb_scaled_q4(bytes: &[u8]) -> Vec<u8> {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    let dims = decoder.info().dimensions;
    let scaled = (dims.0.div_ceil(4), dims.1.div_ceil(4));
    let mut out = vec![0_u8; scaled.0 as usize * scaled.1 as usize * 3];
    decoder
        .decode_scaled_into(
            &mut signinum_j2k::J2kScratchPool::new(),
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
    let Some(bin) = openjpeg_compress_bin() else {
        assert!(
            !require_openjpeg(),
            "SIGNINUM_REQUIRE_OPENJPEG is set but opj_compress was not found"
        );
        return None;
    };
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
        assert!(
            !require_openjpeg(),
            "SIGNINUM_REQUIRE_OPENJPEG is set but opj_compress failed"
        );
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
            if let Some(path) = std::env::var_os("SIGNINUM_OPENJPEG_COMPRESS_BIN") {
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

fn require_openjpeg() -> bool {
    std::env::var_os("SIGNINUM_REQUIRE_OPENJPEG").is_some()
}

fn require_grok() -> bool {
    std::env::var_os("SIGNINUM_REQUIRE_GROK").is_some()
}

fn openjpeg_temp_dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join("signinum-j2k-openjpeg-tests");
        fs::create_dir_all(&dir).expect("create openjpeg temp dir");
        dir
    })
    .clone()
}

fn write_ppm(path: &Path, pixels: &[u8], width: u32, height: u32) -> std::io::Result<()> {
    let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
    bytes.extend_from_slice(pixels);
    fs::write(path, bytes)
}
