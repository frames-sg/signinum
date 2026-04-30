// SPDX-License-Identifier: Apache-2.0

//! Minimal example: parse headers of a JPEG file passed on the command line.
//!
//! `cargo run --example inspect -- path/to/file.jpg`

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let path: PathBuf = if let Some(p) = std::env::args_os().nth(1) {
        p.into()
    } else {
        eprintln!("usage: inspect <jpeg-file>");
        return ExitCode::from(2);
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("reading {}: {e}", path.display());
            return ExitCode::from(1);
        }
    };
    match ashlar_jpeg::Decoder::inspect(&bytes) {
        Ok(info) => {
            println!("{info:#?}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}
