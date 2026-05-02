// SPDX-License-Identifier: Apache-2.0

use signinum_core::{Downscale, PixelFormat, Rect};
use signinum_j2k::J2kDecoder;
use signinum_j2k_native::{encode_htj2k, EncodeOptions};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
};

#[test]
fn classic_gray_full_decode_matches_grok() {
    let Some(path) = grok_decompress_bin() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 1);
    let jp2 = classic_jp2(&pixels, 128, 128, 1).expect("classic jp2");

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = vec![0_u8; 128 * 128];
    decoder
        .decode_into(&mut out, 128, PixelFormat::Gray8)
        .expect("signinum decode");

    let expected = decode_with_grok(&path, "grok_full_gray", &jp2, ".pgm", &[]);
    assert_eq!(out, expected);
}

#[test]
fn classic_gray_region_decode_matches_grok_area_decode() {
    let Some(path) = grok_decompress_bin() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 1);
    let jp2 = classic_jp2(&pixels, 128, 128, 1).expect("classic jp2");
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

    let expected = decode_with_grok(
        &path,
        "grok_region_gray",
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
fn classic_gray_scaled_decode_matches_grok_reduce() {
    let Some(path) = grok_decompress_bin() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 1);
    let jp2 = classic_jp2(&pixels, 128, 128, 1).expect("classic jp2");

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

    let expected = decode_with_grok(&path, "grok_scaled_gray", &jp2, ".pgm", &["-r", "2"]);
    assert_eq!(out, expected);
}

#[test]
fn ht_gray_full_decode_matches_grok() {
    let Some(path) = grok_decompress_bin() else {
        return;
    };
    let pixels = gradient_u8(128, 128, 1);
    let jp2 = ht_jp2(&pixels, 128, 128, 1);

    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = vec![0_u8; 128 * 128];
    decoder
        .decode_into(&mut out, 128, PixelFormat::Gray8)
        .expect("signinum decode");

    let expected = decode_with_grok(&path, "grok_full_ht_gray", &jp2, ".pgm", &[]);
    assert_eq!(out, expected);
}

fn classic_jp2(pixels: &[u8], width: u32, height: u32, components: u8) -> Option<Vec<u8>> {
    let bin = grok_compress_bin()?;
    let dir = temp_dir();
    let src_path = dir.join(if components == 1 {
        "grok_classic_input.pgm"
    } else {
        "grok_classic_input.ppm"
    });
    let out_path = dir.join("grok_classic_output.jp2");
    write_pnm(&src_path, pixels, width, height, components).ok()?;
    let status = Command::new(bin)
        .arg("-i")
        .arg(&src_path)
        .arg("-o")
        .arg(&out_path)
        .arg("-n")
        .arg("4")
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    fs::read(out_path).ok()
}

fn ht_jp2(pixels: &[u8], width: u32, height: u32, components: u8) -> Vec<u8> {
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        ..EncodeOptions::default()
    };
    let codestream =
        encode_htj2k(pixels, width, height, components, 8, false, &options).expect("encode ht");
    wrap_codestream_jp2(&codestream, width, height, u16::from(components), 8, 17)
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

fn grok_decompress_bin() -> Option<PathBuf> {
    static GROK: OnceLock<Option<PathBuf>> = OnceLock::new();
    GROK.get_or_init(discover_grok_decompress_bin).clone()
}

fn grok_compress_bin() -> Option<PathBuf> {
    static GROK: OnceLock<Option<PathBuf>> = OnceLock::new();
    GROK.get_or_init(discover_grok_compress_bin).clone()
}

fn discover_grok_decompress_bin() -> Option<PathBuf> {
    std::env::var_os("SIGNINUM_GROK_BIN")
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from("/opt/homebrew/bin/grk_decompress")))
        .or_else(|| Some(PathBuf::from("/usr/local/bin/grk_decompress")))
        .filter(|path| path.exists())
}

fn discover_grok_compress_bin() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("SIGNINUM_GROK_COMPRESS_BIN")
        .map(PathBuf::from)
        .filter(|path| path.exists())
    {
        return Some(path);
    }
    if let Some(path) = std::env::var_os("SIGNINUM_GROK_BIN")
        .map(PathBuf::from)
        .filter(|path| path.exists())
    {
        let sibling = path.with_file_name("grk_compress");
        if sibling.exists() {
            return Some(sibling);
        }
    }
    [
        "/opt/homebrew/bin/grk_compress",
        "/usr/local/bin/grk_compress",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|path| path.exists())
}

fn decode_with_grok(
    bin: &Path,
    stem: &str,
    jp2: &[u8],
    output_ext: &str,
    extra_args: &[&str],
) -> Vec<u8> {
    let dir = temp_dir();
    let input_path = dir.join(format!("{stem}.jp2"));
    let output_path = dir.join(format!("{stem}{output_ext}"));
    fs::write(&input_path, jp2).expect("write jp2");
    let mut command = Command::new(bin);
    command.arg("-i").arg(&input_path);
    command.arg("-o").arg(&output_path);
    command.args(extra_args);
    let status = command.status().expect("run grk_decompress");
    assert!(status.success(), "grk_decompress failed");
    read_pnm_pixels(&output_path)
}

fn write_pnm(
    path: &Path,
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: u8,
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
        let dir =
            std::env::temp_dir().join(format!("signinum-j2k-grok-parity-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    })
}
