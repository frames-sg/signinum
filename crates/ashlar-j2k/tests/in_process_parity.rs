// SPDX-License-Identifier: Apache-2.0

mod common;

use ashlar_core::Rect;
use common::{ashlar_rgb, ashlar_rgb_region, ashlar_rgb_scaled_q4, bench_fixture_rgb, in_process};

#[test]
fn openjpeg_in_process_matches_ashlar_rgb_fixture() {
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let ours = ashlar_rgb(&input);
    let theirs = in_process::openjpeg::decode_rgb(&input).expect("openjpeg");
    assert_eq!(ours, theirs);
}

#[test]
fn openjpeg_in_process_region_matches_ashlar_rgb_fixture() {
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let roi = Rect {
        x: 16,
        y: 24,
        w: 64,
        h: 64,
    };
    let ours = ashlar_rgb_region(&input, roi);
    let theirs = in_process::openjpeg::decode_rgb_region(&input, roi).expect("openjpeg");
    assert_eq!(ours, theirs);
}

#[test]
fn grok_in_process_matches_ashlar_rgb_fixture() {
    if !in_process::grok::is_available() {
        return;
    }
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let ours = ashlar_rgb(&input);
    let theirs = in_process::grok::decode_rgb(&input).expect("grok");
    assert_eq!(ours, theirs);
}

#[test]
fn grok_in_process_scaled_matches_ashlar_rgb_fixture() {
    if !in_process::grok::is_available() {
        return;
    }
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let ours = ashlar_rgb_scaled_q4(&input);
    let theirs = in_process::grok::decode_rgb_scaled(&input, 2).expect("grok");
    assert_eq!(ours, theirs);
}
