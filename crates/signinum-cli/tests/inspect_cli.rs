// SPDX-License-Identifier: Apache-2.0

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

fn signinum_bin() -> &'static str {
    env!("CARGO_BIN_EXE_signinum")
}

fn run_signinum(args: &[&str]) -> Output {
    Command::new(signinum_bin())
        .args(args)
        .output()
        .expect("run signinum CLI")
}

fn write_temp_file(name: &str, bytes: &[u8]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("signinum-cli-tests-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("create CLI test temp dir");
    let path = dir.join(name);
    fs::write(&path, bytes).expect("write CLI test input");
    path
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn inspect_cli_reports_jpeg_info() {
    let jpeg = minimal_jpeg();
    let input = write_temp_file("grayscale_8x8.jpg", &jpeg);

    let output = run_signinum(&["inspect", path_str(&input)]);

    assert!(output.status.success(), "stderr: {}", stderr(&output));
    let stdout = stdout(&output);
    assert!(stdout.contains("8"));
    assert!(stdout.contains("Grayscale"));
    assert!(stdout.contains("bit=8"));
}

#[test]
fn inspect_cli_reports_jp2_info() {
    let input = write_temp_file("minimal.jp2", &minimal_jp2());

    let output = run_signinum(&["inspect", path_str(&input)]);

    assert!(output.status.success(), "stderr: {}", stderr(&output));
    let stdout = stdout(&output);
    assert!(stdout.contains("128"));
    assert!(stdout.contains("64"));
    assert!(stdout.contains("levels=6"));
}

#[test]
fn inspect_cli_rejects_unknown_subcommand() {
    let output = run_signinum(&["unknown"]);

    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("unknown subcommand: unknown"));
}

#[test]
fn inspect_cli_reports_missing_file() {
    let missing = std::env::temp_dir()
        .join(format!("signinum-cli-tests-{}", std::process::id()))
        .join("missing.jpg");

    let output = run_signinum(&["inspect", path_str(&missing)]);

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("error reading"));
}

#[test]
fn inspect_cli_reports_invalid_jpeg() {
    let input = write_temp_file("invalid.jpg", b"not a jpeg");

    let output = run_signinum(&["inspect", path_str(&input)]);

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("error:"));
}

fn path_str(path: &Path) -> &str {
    path.to_str().expect("test path is UTF-8")
}

fn minimal_jpeg() -> Vec<u8> {
    let gray = (0..64).map(|value| (value * 3) as u8).collect::<Vec<_>>();
    signinum_jpeg::encode_jpeg_baseline(
        signinum_jpeg::JpegSamples::Gray8 {
            data: &gray,
            width: 8,
            height: 8,
        },
        signinum_jpeg::JpegEncodeOptions {
            quality: 90,
            subsampling: signinum_jpeg::JpegSubsampling::Gray,
            restart_interval: None,
            backend: signinum_jpeg::JpegBackend::Cpu,
        },
    )
    .expect("encode CLI test JPEG")
    .data
}

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
        0, 0, 0, 20, b'f', b't', b'y', b'p', b'j', b'p', b'2', b' ', 0, 0, 0, 0, b'j', b'p', b'2',
        b' ',
    ]);
    bytes.extend_from_slice(&[
        0, 0, 0, 45, b'j', b'p', b'2', b'h', 0, 0, 0, 22, b'i', b'h', b'd', b'r', 0, 0, 0, 64, 0,
        0, 0, 128, 0, 3, 7, 7, 0, 0, 0, 0, 0, 15, b'c', b'o', b'l', b'r', 1, 0, 0, 0, 0, 0, 16,
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
