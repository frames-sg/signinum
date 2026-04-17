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
            eprintln!("slidecodec 0.0.0");
            eprintln!("Usage:");
            eprintln!("  slidecodec inspect <file>    Parse headers and print Info");
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
    match slidecodec_jpeg::Decoder::inspect(&bytes) {
        Ok(info) => {
            println!("{}×{} {:?} {:?} bit={} samp={:?} rst={:?} scans={}",
                info.dimensions.0, info.dimensions.1,
                info.sof_kind, info.color_space,
                info.bit_depth, info.sampling.components,
                info.restart_interval, info.scan_count,
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            if e.is_unsupported() {
                eprintln!("hint: this file is not supported by slidecodec; try jpeg-decoder or openjpeg");
            }
            ExitCode::from(1)
        }
    }
}

fn read_file(path: &Path) -> io::Result<Vec<u8>> {
    let mut file = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    Ok(buf)
}
