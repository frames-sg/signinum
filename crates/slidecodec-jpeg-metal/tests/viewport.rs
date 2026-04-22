// SPDX-License-Identifier: Apache-2.0

use slidecodec_core::{Downscale, PixelFormat, Rect};
use slidecodec_jpeg::{Decoder, ScratchPool};
use slidecodec_jpeg_metal::viewport::{
    compose_viewport_cpu, compose_viewport_hybrid, ViewportTile,
};

const BASELINE_420: &[u8] = include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");

fn quadrant_tiles() -> [ViewportTile; 4] {
    [
        ViewportTile {
            source_roi: Rect {
                x: 0,
                y: 0,
                w: 8,
                h: 8,
            },
            dest: Rect {
                x: 0,
                y: 0,
                w: 8,
                h: 8,
            },
        },
        ViewportTile {
            source_roi: Rect {
                x: 8,
                y: 0,
                w: 8,
                h: 8,
            },
            dest: Rect {
                x: 8,
                y: 0,
                w: 8,
                h: 8,
            },
        },
        ViewportTile {
            source_roi: Rect {
                x: 0,
                y: 8,
                w: 8,
                h: 8,
            },
            dest: Rect {
                x: 0,
                y: 8,
                w: 8,
                h: 8,
            },
        },
        ViewportTile {
            source_roi: Rect {
                x: 8,
                y: 8,
                w: 8,
                h: 8,
            },
            dest: Rect {
                x: 8,
                y: 8,
                w: 8,
                h: 8,
            },
        },
    ]
}

#[test]
fn cpu_viewport_quadrants_match_full_decode() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut pool = ScratchPool::new();

    let actual = compose_viewport_cpu(
        &decoder,
        &mut pool,
        PixelFormat::Rgb8,
        Downscale::None,
        (16, 16),
        &quadrant_tiles(),
    )
    .expect("viewport");
    let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("full decode");

    assert_eq!(actual, expected);
}

#[cfg(target_os = "macos")]
#[test]
fn hybrid_viewport_quadrants_match_cpu_viewport() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut cpu_pool = ScratchPool::new();
    let mut hybrid_pool = ScratchPool::new();

    let expected = compose_viewport_cpu(
        &decoder,
        &mut cpu_pool,
        PixelFormat::Rgb8,
        Downscale::None,
        (16, 16),
        &quadrant_tiles(),
    )
    .expect("cpu viewport");
    let actual = compose_viewport_hybrid(
        &decoder,
        &mut hybrid_pool,
        Downscale::None,
        (16, 16),
        &quadrant_tiles(),
    )
    .expect("hybrid viewport");

    assert_eq!(actual.as_bytes(), expected.as_slice());
}
