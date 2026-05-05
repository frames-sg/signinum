// SPDX-License-Identifier: Apache-2.0

use signinum_core::{Downscale, PixelFormat, Rect};
use signinum_j2k::{
    encode_j2k_lossless, EncodeBackendPreference, J2kDecoder, J2kLosslessEncodeOptions,
    J2kLosslessSamples,
};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
};

#[test]
fn classic_gray_full_decode_matches_openjpeg() {
    let Some(paths) = OpenJpegPaths::discover() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 1);
    let jp2 = encode_with_openjpeg(&paths, "parity_full_gray", &pixels, 128, 128, 1);

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = vec![0_u8; 128 * 128];
    decoder
        .decode_into(&mut out, 128, PixelFormat::Gray8)
        .expect("signinum decode");

    let expected = decode_with_openjpeg(&paths, "parity_full_gray", &jp2, ".pgm", &[]);
    assert_eq!(out, expected);
}

#[test]
fn classic_gray_region_decode_matches_openjpeg_area_decode() {
    let Some(paths) = OpenJpegPaths::discover() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 1);
    let jp2 = encode_with_openjpeg(&paths, "parity_region_gray", &pixels, 128, 128, 1);
    let roi = Rect {
        x: 16,
        y: 24,
        w: 48,
        h: 48,
    };

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = vec![0_u8; roi.w as usize * roi.h as usize];
    decoder
        .decode_region_into(
            &mut signinum_j2k::J2kScratchPool::new(),
            &mut out,
            roi.w as usize,
            PixelFormat::Gray8,
            roi,
        )
        .expect("signinum region decode");

    let expected = decode_with_openjpeg(
        &paths,
        "parity_region_gray",
        &jp2,
        ".pgm",
        &[
            "-d",
            &format!("{},{},{},{}", roi.x, roi.y, roi.x + roi.w, roi.y + roi.h),
        ],
    );
    assert_eq!(out, expected);
}

#[test]
fn classic_gray_scaled_decode_matches_openjpeg_reduce() {
    let Some(paths) = OpenJpegPaths::discover() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 1);
    let jp2 = encode_with_openjpeg(&paths, "parity_scaled_gray", &pixels, 128, 128, 1);

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = vec![0_u8; 32 * 32];
    decoder
        .decode_scaled_into(
            &mut signinum_j2k::J2kScratchPool::new(),
            &mut out,
            32,
            PixelFormat::Gray8,
            Downscale::Quarter,
        )
        .expect("signinum scaled decode");

    let expected = decode_with_openjpeg(&paths, "parity_scaled_gray", &jp2, ".pgm", &["-r", "2"]);
    assert_eq!(out, expected);
}

#[test]
fn classic_lossless_encode_decodes_with_openjpeg() {
    let Some(paths) = OpenJpegPaths::discover() else {
        return;
    };
    let pixels = gradient_u8(64, 64, 3);
    let samples = J2kLosslessSamples::new(&pixels, 64, 64, 3, 8, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions {
            backend: EncodeBackendPreference::CpuOnly,
            ..J2kLosslessEncodeOptions::default()
        },
    )
    .expect("signinum encode");

    let decoded = decode_j2k_with_openjpeg(&paths, "signinum_encode_rgb", &encoded.codestream);
    assert_eq!(decoded, pixels);
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

#[derive(Clone)]
struct OpenJpegPaths {
    compress: PathBuf,
    decompress: PathBuf,
}

impl OpenJpegPaths {
    fn discover() -> Option<Self> {
        let paths = discover_openjpeg_paths();
        assert!(
            paths.is_some() || !require_openjpeg(),
            "SIGNINUM_REQUIRE_OPENJPEG is set but opj_compress/opj_decompress were not found"
        );
        paths
    }
}

fn discover_openjpeg_paths() -> Option<OpenJpegPaths> {
    let compress = std::env::var_os("SIGNINUM_OPENJPEG_COMPRESS_BIN")
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from("/opt/homebrew/bin/opj_compress")))
        .filter(|path| path.exists())?;
    let decompress = std::env::var_os("SIGNINUM_OPENJPEG_BIN")
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from("/opt/homebrew/bin/opj_decompress")))
        .filter(|path| path.exists())?;
    Some(OpenJpegPaths {
        compress,
        decompress,
    })
}

fn require_openjpeg() -> bool {
    std::env::var_os("SIGNINUM_REQUIRE_OPENJPEG").is_some()
}

fn encode_with_openjpeg(
    paths: &OpenJpegPaths,
    stem: &str,
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: usize,
) -> Vec<u8> {
    let dir = temp_dir();
    let src_path = dir.join(if channels == 1 {
        format!("{stem}.pgm")
    } else {
        format!("{stem}.ppm")
    });
    let out_path = dir.join(format!("{stem}.jp2"));
    write_pnm(&src_path, pixels, width, height, channels).expect("write pnm");
    let status = Command::new(&paths.compress)
        .arg("-i")
        .arg(&src_path)
        .arg("-o")
        .arg(&out_path)
        .status()
        .expect("run opj_compress");
    assert!(status.success(), "opj_compress failed");
    fs::read(out_path).expect("read jp2")
}

fn decode_with_openjpeg(
    paths: &OpenJpegPaths,
    stem: &str,
    jp2: &[u8],
    output_ext: &str,
    extra_args: &[&str],
) -> Vec<u8> {
    let dir = temp_dir();
    let input_path = dir.join(format!("{stem}.jp2"));
    let output_path = dir.join(format!("{stem}{output_ext}"));
    fs::write(&input_path, jp2).expect("write jp2");
    let mut command = Command::new(&paths.decompress);
    command.arg("-i").arg(&input_path);
    command.arg("-o").arg(&output_path);
    command.arg("-quiet");
    command.args(extra_args);
    let status = command.status().expect("run opj_decompress");
    assert!(status.success(), "opj_decompress failed");
    read_pnm_pixels(&output_path)
}

fn decode_j2k_with_openjpeg(paths: &OpenJpegPaths, stem: &str, codestream: &[u8]) -> Vec<u8> {
    let dir = temp_dir();
    let input_path = dir.join(format!("{stem}.j2k"));
    let output_path = dir.join(format!("{stem}.ppm"));
    fs::write(&input_path, codestream).expect("write j2k");
    let status = Command::new(&paths.decompress)
        .arg("-i")
        .arg(&input_path)
        .arg("-o")
        .arg(&output_path)
        .arg("-quiet")
        .status()
        .expect("run opj_decompress");
    assert!(status.success(), "opj_decompress failed");
    read_pnm_pixels(&output_path)
}

fn write_pnm(
    path: &Path,
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: usize,
) -> std::io::Result<()> {
    let mut bytes = Vec::new();
    if channels == 1 {
        bytes.extend_from_slice(format!("P5\n{width} {height}\n255\n").as_bytes());
    } else {
        bytes.extend_from_slice(format!("P6\n{width} {height}\n255\n").as_bytes());
    }
    bytes.extend_from_slice(pixels);
    fs::write(path, bytes)
}

fn read_pnm_pixels(path: &Path) -> Vec<u8> {
    let bytes = fs::read(path).expect("read pnm");
    let mut cursor = 0;

    let magic = read_pnm_token(&bytes, &mut cursor).expect("pnm magic");
    assert!(magic == b"P5" || magic == b"P6", "unexpected pnm magic");

    let _width = read_pnm_token(&bytes, &mut cursor).expect("pnm width");
    let _height = read_pnm_token(&bytes, &mut cursor).expect("pnm height");
    let _maxval = read_pnm_token(&bytes, &mut cursor).expect("pnm maxval");

    while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
        cursor += 1;
    }

    bytes[cursor..].to_vec()
}

fn read_pnm_token<'a>(bytes: &'a [u8], cursor: &mut usize) -> Option<&'a [u8]> {
    skip_pnm_separators(bytes, cursor);
    if *cursor >= bytes.len() {
        return None;
    }

    let start = *cursor;
    while *cursor < bytes.len() {
        let byte = bytes[*cursor];
        if byte.is_ascii_whitespace() || byte == b'#' {
            break;
        }
        *cursor += 1;
    }

    if start == *cursor {
        None
    } else {
        Some(&bytes[start..*cursor])
    }
}

fn skip_pnm_separators(bytes: &[u8], cursor: &mut usize) {
    while *cursor < bytes.len() {
        let byte = bytes[*cursor];
        if byte.is_ascii_whitespace() {
            *cursor += 1;
            continue;
        }
        if byte == b'#' {
            *cursor += 1;
            while *cursor < bytes.len() && bytes[*cursor] != b'\n' {
                *cursor += 1;
            }
            continue;
        }
        break;
    }
}

fn temp_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("signinum-j2k-parity-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    })
}
