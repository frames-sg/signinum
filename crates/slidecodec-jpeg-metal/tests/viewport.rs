// SPDX-License-Identifier: Apache-2.0

use slidecodec_core::{Downscale, PixelFormat, Rect};
use slidecodec_jpeg::{Decoder, ScratchPool};
use slidecodec_jpeg_metal::viewport::{
    compose_viewport_cpu, compose_viewport_hybrid, suggest_viewport_workload, ViewportTile,
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

#[test]
fn suggested_viewport_workload_is_fixed_for_macro_like_input() {
    let workload = suggest_viewport_workload((1_191, 408)).expect("workload");

    assert_eq!(workload.scale, Downscale::Half);
    assert_eq!(workload.viewport_dims, (576, 192));
    assert_eq!(workload.tiles.len(), 12);
    assert_eq!(
        workload.tiles.first(),
        Some(&ViewportTile {
            source_roi: Rect {
                x: 18,
                y: 12,
                w: 192,
                h: 192,
            },
            dest: Rect {
                x: 0,
                y: 0,
                w: 96,
                h: 96,
            },
        })
    );
    assert_eq!(
        workload.tiles.last(),
        Some(&ViewportTile {
            source_roi: Rect {
                x: 978,
                y: 204,
                w: 192,
                h: 192,
            },
            dest: Rect {
                x: 480,
                y: 96,
                w: 96,
                h: 96,
            },
        })
    );
}

#[test]
fn cpu_viewport_misaligned_scaled_tile_matches_direct_decode() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut cpu_pool = ScratchPool::new();
    let roi = Rect {
        x: 1,
        y: 1,
        w: 10,
        h: 10,
    };
    let tiles = [ViewportTile {
        source_roi: roi,
        dest: Rect {
            x: 0,
            y: 0,
            w: 6,
            h: 6,
        },
    }];

    let viewport = compose_viewport_cpu(
        &decoder,
        &mut cpu_pool,
        PixelFormat::Rgb8,
        Downscale::Half,
        (6, 6),
        &tiles,
    )
    .expect("cpu viewport");
    let (expected, _outcome) = decoder
        .decode_region_scaled(
            PixelFormat::Rgb8,
            slidecodec_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            Downscale::Half,
        )
        .expect("direct decode");

    assert_eq!(expected.len(), 6 * 6 * 3);
    assert_eq!(viewport, expected);
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

#[cfg(target_os = "macos")]
#[test]
fn hybrid_viewport_misaligned_scaled_tile_matches_cpu_viewport() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut cpu_pool = ScratchPool::new();
    let mut hybrid_pool = ScratchPool::new();
    let tiles = [ViewportTile {
        source_roi: Rect {
            x: 1,
            y: 1,
            w: 10,
            h: 10,
        },
        dest: Rect {
            x: 0,
            y: 0,
            w: 6,
            h: 6,
        },
    }];

    let expected = compose_viewport_cpu(
        &decoder,
        &mut cpu_pool,
        PixelFormat::Rgb8,
        Downscale::Half,
        (6, 6),
        &tiles,
    )
    .expect("cpu viewport");
    let actual =
        compose_viewport_hybrid(&decoder, &mut hybrid_pool, Downscale::Half, (6, 6), &tiles)
            .expect("hybrid viewport");

    assert_eq!(actual.as_bytes(), expected.as_slice());
}
