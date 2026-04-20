// SPDX-License-Identifier: Apache-2.0

mod common;

use common::{
    bench_fixture_rgb, slidecodec_rgb, slidecodec_rgb_region, slidecodec_rgb_scaled_q4, in_process,
};
use slidecodec_core::Rect;

#[test]
fn openjpeg_in_process_matches_slidecodec_rgb_fixture() {
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let ours = slidecodec_rgb(&input);
    let theirs = in_process::openjpeg::decode_rgb(&input).expect("openjpeg");
    assert_eq!(ours, theirs);
}

#[test]
fn openjpeg_in_process_region_matches_slidecodec_rgb_fixture() {
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let roi = Rect {
        x: 16,
        y: 24,
        w: 64,
        h: 64,
    };
    let ours = slidecodec_rgb_region(&input, roi);
    let theirs = in_process::openjpeg::decode_rgb_region(&input, roi).expect("openjpeg");
    assert_eq!(ours, theirs);
}

#[test]
fn grok_in_process_matches_slidecodec_rgb_fixture() {
    if !in_process::grok::is_available() {
        return;
    }
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let ours = slidecodec_rgb(&input);
    let theirs = in_process::grok::decode_rgb(&input).expect("grok");
    assert_eq!(ours, theirs);
}

#[test]
fn grok_in_process_scaled_matches_slidecodec_rgb_fixture() {
    if !in_process::grok::is_available() {
        return;
    }
    let Some(input) = bench_fixture_rgb() else {
        return;
    };
    let ours = slidecodec_rgb_scaled_q4(&input);
    let theirs = in_process::grok::decode_rgb_scaled(&input, 2).expect("grok");
    assert_eq!(ours, theirs);
}
