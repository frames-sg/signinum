// SPDX-License-Identifier: Apache-2.0

use std::io::{self, Read};
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let subcommand = args.next();
    match subcommand.as_deref() {
        Some("inspect") => {
            let path = match args.next() {
                Some(p) => p,
                None => {
                    eprintln!("usage: slidecodec inspect <file>");
                    return ExitCode::from(2);
                }
            };
            inspect(Path::new(&path))
        }
        Some("--help") | Some("-h") | Some("help") | None => {
            eprintln!("slidecodec {}", env!("CARGO_PKG_VERSION"));
            eprintln!("Usage:");
            eprintln!(
                "  slidecodec inspect <file>    Parse JPEG or JPEG 2000 headers and print Info"
            );
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("unknown subcommand: {other}");
            ExitCode::from(2)
        }
    }
}

fn inspect(path: &Path) -> ExitCode {
    let bytes = match read_file(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error reading {}: {e}", path.display());
            return ExitCode::from(1);
        }
    };
    match inspect_bytes(&bytes) {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(1)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InspectFormat {
    Jpeg,
    J2k,
}

fn detect_inspect_format(bytes: &[u8]) -> InspectFormat {
    if bytes.starts_with(&[0, 0, 0, 12, b'j', b'P', b' ', b' ']) || bytes.starts_with(&[0xFF, 0x4F])
    {
        InspectFormat::J2k
    } else {
        InspectFormat::Jpeg
    }
}

fn inspect_bytes(bytes: &[u8]) -> Result<String, String> {
    match detect_inspect_format(bytes) {
        InspectFormat::Jpeg => match slidecodec_jpeg::Decoder::inspect(bytes) {
            Ok(info) => Ok(format!(
                "{}×{} {:?} {:?} bit={} samp={:?} rst={:?} scans={}",
                info.dimensions.0,
                info.dimensions.1,
                info.sof_kind,
                info.color_space,
                info.bit_depth,
                info.sampling.components(),
                info.restart_interval,
                info.scan_count,
            )),
            Err(e) => {
                let mut message = format!("error: {e}");
                if e.is_unsupported() {
                    message.push_str(
                        "\nhint: this file is not supported by slidecodec; try jpeg-decoder or openjpeg",
                    );
                }
                Err(message)
            }
        },
        InspectFormat::J2k => match slidecodec_j2k::J2kDecoder::inspect(bytes) {
            Ok(info) => Ok(format!(
                "{}×{} {:?} bit={} comps={} levels={} tiles={:?}",
                info.dimensions.0,
                info.dimensions.1,
                info.colorspace,
                info.bit_depth,
                info.components,
                info.resolution_levels,
                info.tile_layout,
            )),
            Err(e) => Err(format!("error: {e}")),
        },
    }
}

fn read_file(path: &Path) -> io::Result<Vec<u8>> {
    let mut file = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::{detect_inspect_format, inspect_bytes, InspectFormat};

    fn minimal_j2k_codestream() -> Vec<u8> {
        let mut bytes = vec![0xFF, 0x4F];
        let mut siz = Vec::new();
        push_u16(&mut siz, 0);
        push_u32(&mut siz, 128);
        push_u32(&mut siz, 64);
        push_u32(&mut siz, 0);
        push_u32(&mut siz, 0);
        push_u32(&mut siz, 64);
        push_u32(&mut siz, 64);
        push_u32(&mut siz, 0);
        push_u32(&mut siz, 0);
        push_u16(&mut siz, 3);
        for _ in 0..3 {
            siz.extend_from_slice(&[0x07, 0x01, 0x01]);
        }
        bytes.extend_from_slice(&[0xFF, 0x51]);
        push_u16(&mut bytes, (siz.len() + 2) as u16);
        bytes.extend_from_slice(&siz);

        let cod = [0x00, 0x00, 0x00, 0x01, 0x01, 0x05, 0x04, 0x04, 0x00, 0x01];
        bytes.extend_from_slice(&[0xFF, 0x52]);
        push_u16(&mut bytes, (cod.len() + 2) as u16);
        bytes.extend_from_slice(&cod);
        bytes.extend_from_slice(&[0xFF, 0x90, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        bytes
    }

    fn minimal_jp2() -> Vec<u8> {
        let codestream = minimal_j2k_codestream();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&[0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0D, 0x0A, 0x87, 0x0A]);
        bytes.extend_from_slice(&[
            0, 0, 0, 20, b'f', b't', b'y', b'p', b'j', b'p', b'2', b' ', 0, 0, 0, 0, b'j', b'p',
            b'2', b' ',
        ]);
        bytes.extend_from_slice(&[
            0, 0, 0, 45, b'j', b'p', b'2', b'h', 0, 0, 0, 22, b'i', b'h', b'd', b'r', 0, 0, 0, 64,
            0, 0, 0, 128, 0, 3, 7, 7, 0, 0, 0, 0, 0, 15, b'c', b'o', b'l', b'r', 1, 0, 0, 0, 0, 0,
            16,
        ]);
        let len = (8 + codestream.len()) as u32;
        bytes.extend_from_slice(&len.to_be_bytes());
        bytes.extend_from_slice(b"jp2c");
        bytes.extend_from_slice(&codestream);
        bytes
    }

    fn push_u16(out: &mut Vec<u8>, value: u16) {
        out.extend_from_slice(&value.to_be_bytes());
    }

    fn push_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_be_bytes());
    }

    #[test]
    fn detects_j2k_codestream_magic() {
        assert_eq!(
            detect_inspect_format(&minimal_j2k_codestream()),
            InspectFormat::J2k
        );
    }

    #[test]
    fn detects_jp2_magic() {
        assert_eq!(detect_inspect_format(&minimal_jp2()), InspectFormat::J2k);
    }

    #[test]
    fn inspect_bytes_dispatches_to_j2k() {
        let line = inspect_bytes(&minimal_jp2()).expect("jp2 inspect");
        assert!(line.contains("128×64"));
        assert!(line.contains("levels=6"));
    }
}
