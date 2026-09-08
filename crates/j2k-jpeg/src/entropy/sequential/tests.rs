// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::entropy::huffman::{HuffmanTable, PreparedHuffmanTables};
use crate::info::{ColorSpace, SamplingFactors};
use crate::output::Rgb8Writer;
use crate::parse::tables::{HuffmanTableRole, HuffmanValues, RawHuffmanTable};
use crate::Decoder;
use alloc::vec;
use j2k_test_support::JPEG_BASELINE_420_16X16;

#[test]
fn sequential_decode_modules_stay_focused_and_fragment_free() {
    const ROOT: &str = include_str!("../sequential.rs");
    const GENERIC: &str = include_str!("generic.rs");
    const GENERIC_DRIVER: &str = include_str!("generic/driver.rs");
    const GENERIC_ROW: &str = include_str!("generic/row.rs");
    const DCT: &str = include_str!("dct.rs");
    const DCT_ALLOCATION: &str = include_str!("dct/allocation.rs");
    const OUTPUT_SCRATCH: &str = include_str!("output_scratch.rs");
    const PLAN: &str = include_str!("plan.rs");
    const PLAN_RESOLVED: &str = include_str!("plan/resolved.rs");
    const RGB444: &str = include_str!("rgb444.rs");
    const STRIPE: &str = include_str!("stripe.rs");
    const FAST420: &str = include_str!("fast420/mod.rs");
    const FAST420_ROWS: &str = include_str!("fast420/rows.rs");

    let modules = [
        ("sequential.rs", ROOT, 240usize),
        ("sequential/generic.rs", GENERIC, 180),
        ("sequential/generic/driver.rs", GENERIC_DRIVER, 260),
        ("sequential/generic/row.rs", GENERIC_ROW, 230),
        ("sequential/dct.rs", DCT, 200),
        ("sequential/dct/allocation.rs", DCT_ALLOCATION, 320),
        ("sequential/output_scratch.rs", OUTPUT_SCRATCH, 60),
        ("sequential/plan.rs", PLAN, 120),
        ("sequential/plan/resolved.rs", PLAN_RESOLVED, 80),
        ("sequential/rgb444.rs", RGB444, 300),
        ("sequential/stripe.rs", STRIPE, 240),
        ("sequential/fast420/mod.rs", FAST420, 650),
        ("sequential/fast420/rows.rs", FAST420_ROWS, 700),
    ];

    for (path, source, max_lines) in modules {
        let line_count = source.lines().count();
        assert!(
            line_count <= max_lines,
            "{path} grew to {line_count} lines; split it before exceeding {max_lines}"
        );
        assert!(
            !source.contains("include!(") && !source.contains("#[path"),
            "{path} must remain a real Rust module, not a textual source fragment"
        );
    }

    for declaration in [
        "mod dct;",
        "mod fast420;",
        "mod generic;",
        "mod output_scratch;",
        "mod plan;",
        "mod rgb444;",
        "mod stripe;",
    ] {
        assert!(
            ROOT.contains(declaration),
            "sequential facade lost required module boundary {declaration}"
        );
    }
    assert!(
        DCT.contains("mod allocation;"),
        "sequential DCT execution lost its allocation boundary"
    );
    for declaration in ["mod driver;", "mod row;"] {
        assert!(
            GENERIC.contains(declaration),
            "generic sequential owner lost required boundary {declaration}"
        );
    }
    for shared_owner in [
        "struct ScanSetup",
        "struct ScanBuffers",
        "trait StripeEmitter",
        "fn decode_scan_rows",
    ] {
        assert!(
            GENERIC_DRIVER.contains(shared_owner),
            "generic scan driver lost typed shared owner {shared_owner}"
        );
    }
    assert!(!GENERIC.contains("clippy::too_many_lines"));
    assert!(!GENERIC_DRIVER.contains("macro_rules!"));
}

#[test]
fn fast_tile_rgb_matches_generic_baseline_decode() {
    let dec = Decoder::new(JPEG_BASELINE_420_16X16).expect("fixture must parse");
    assert!(dec.plan.matches_fast_tile_shape());

    let mut generic = vec![0u8; (dec.info.dimensions.0 * dec.info.dimensions.1 * 3) as usize];
    let mut fast = vec![0u8; generic.len()];
    let mut generic_writer = Rgb8Writer::new(
        &mut generic,
        dec.info.dimensions.0 as usize * 3,
        dec.info.dimensions.0,
    );
    let mut fast_writer = Rgb8Writer::new(
        &mut fast,
        dec.info.dimensions.0 as usize * 3,
        dec.info.dimensions.0,
    );
    let mut generic_pool = ScratchPool::new();
    let mut fast_pool = ScratchPool::new();
    let scan_bytes = &dec.bytes[dec.plan.scan_offset..];

    decode_scan_baseline_rgb(
        &dec.plan,
        dec.backend,
        scan_bytes,
        &mut generic_pool,
        &mut generic_writer,
        DownscaleFactor::Full,
        Rect::full(dec.info.dimensions),
    )
    .expect("generic path must decode");
    decode_scan_fast_tile_rgb(
        &dec.plan,
        dec.backend,
        scan_bytes,
        &mut fast_pool,
        &mut fast_writer,
    )
    .expect("fast path must decode");

    assert_eq!(fast, generic);
}

#[cfg(feature = "bench-internals")]
#[test]
fn fast_tile_profiled_rgb_matches_unprofiled_decode() {
    let dec = Decoder::new(JPEG_BASELINE_420_16X16).expect("fixture must parse");
    assert!(dec.plan.matches_fast_tile_shape());

    let mut unprofiled = vec![0u8; (dec.info.dimensions.0 * dec.info.dimensions.1 * 3) as usize];
    let mut profiled = vec![0u8; unprofiled.len()];
    let mut unprofiled_writer = Rgb8Writer::new(
        &mut unprofiled,
        dec.info.dimensions.0 as usize * 3,
        dec.info.dimensions.0,
    );
    let mut profiled_writer = Rgb8Writer::new(
        &mut profiled,
        dec.info.dimensions.0 as usize * 3,
        dec.info.dimensions.0,
    );
    let mut unprofiled_pool = ScratchPool::new();
    let mut profiled_pool = ScratchPool::new();
    let mut profile = BenchFast420Profile::default();
    let scan_bytes = &dec.bytes[dec.plan.scan_offset..];

    decode_scan_fast_tile_rgb(
        &dec.plan,
        dec.backend,
        scan_bytes,
        &mut unprofiled_pool,
        &mut unprofiled_writer,
    )
    .expect("unprofiled fast path must decode");
    decode_scan_fast_tile_rgb_profiled(
        &dec.plan,
        dec.backend,
        scan_bytes,
        &mut profiled_pool,
        &mut profiled_writer,
        &mut profile,
    )
    .expect("profiled fast path must decode");

    assert_eq!(profiled, unprofiled);
    assert!(profile.block_activity_counts().total_blocks() > 0);
}

#[test]
fn fast_tile_row_decoder_uses_stripe_local_y_offsets() {
    let mut huffman_tables =
        PreparedHuffmanTables::try_with_capacity(2).expect("bounded test arena");
    let dc = huffman_tables
        .push(trivial_dc_table())
        .expect("reserved DC slot");
    let ac = huffman_tables
        .push(eob_ac_table())
        .expect("reserved AC slot");
    let y_comp = PreparedComponentPlan {
        h: 2,
        v: 2,
        output_index: 0,
        quant: [1u16; 64],
        dc_table: Some(dc),
        ac_table: Some(ac),
    };
    let cb_comp = PreparedComponentPlan {
        h: 1,
        v: 1,
        output_index: 1,
        quant: [1u16; 64],
        dc_table: Some(dc),
        ac_table: Some(ac),
    };
    let cr_comp = PreparedComponentPlan {
        h: 1,
        v: 1,
        output_index: 2,
        quant: [1u16; 64],
        dc_table: Some(dc),
        ac_table: Some(ac),
    };
    let plan = PreparedDecodePlan {
        components: vec![y_comp, cb_comp, cr_comp],
        huffman_tables,
        sampling: SamplingFactors::from_validated_components(&[(2, 2), (1, 1), (1, 1)]),
        color_space: ColorSpace::YCbCr,
        restart_interval: None,
        dimensions: (16, 16),
        scan_offset: 0,
        scratch_bytes: 0,
    };
    let components = FastTile420Components {
        y: plan.resolved_component(0).expect("Y component"),
        cb: plan.resolved_component(1).expect("Cb component"),
        cr: plan.resolved_component(2).expect("Cr component"),
    };
    let mut stripe = StripeBuffer {
        planes: vec![vec![0u8; 16 * 16], vec![0u8; 8 * 8], vec![0u8; 8 * 8]],
        plane_strides: vec![16, 8, 8],
        plane_rows: vec![16, 8, 8],
    };
    let mut br = BitReader::new(&[0u8; 16]);
    let mut coeff = CoefficientBlock::default();
    let mut pixels = [0u8; 64];
    let mut y_dc = 0;
    let mut cb_dc = 0;
    let mut cr_dc = 0;
    let mut profiler = NoopFast420Profiler;

    decode_mcu_row_fast_tile_420(
        components,
        Backend::detect(),
        &mut FastTile420EntropyState {
            br: &mut br,
            dc: FastTile420DcState {
                y: &mut y_dc,
                cb: &mut cb_dc,
                cr: &mut cr_dc,
            },
            coeff: &mut coeff,
        },
        &mut pixels,
        FastTile420Window {
            mcus_per_row: 1,
            stripe_mcu_start: 0,
            stripe_mcus_per_row: 1,
        },
        &mut stripe,
        &mut profiler,
    )
    .expect("second stripe row must still decode within stripe-local buffers");

    assert_eq!(stripe.planes[0][0], 128);
    assert_eq!(stripe.planes[0][8], 128);
    assert_eq!(stripe.planes[0][16 * 8], 128);
    assert_eq!(stripe.planes[0][16 * 8 + 8], 128);
    assert_eq!(stripe.planes[1][0], 128);
    assert_eq!(stripe.planes[2][0], 128);
}

#[test]
fn fast_tile_region_layout_shrinks_horizontal_stripe_span() {
    let roi = Rect {
        x: 17,
        y: 3,
        w: 9,
        h: 8,
    };

    let layout = Fast420RegionLayout::new(64, roi);

    assert_eq!(layout.stripe_mcu_start, 0);
    assert_eq!(layout.stripe_mcus_per_row, 2);
    assert_eq!(layout.row_width(), 32);
    assert_eq!(layout.chroma_width(), 16);
    assert_eq!(layout.crop_start, 17);
    assert_eq!(layout.crop_end, 26);
    assert!(layout.y_decode_start <= roi.x as usize);
    assert!(layout.y_decode_end >= (roi.x + roi.w) as usize);
}

#[test]
fn direct_420_crop_policy_includes_scaled_regions_when_backend_supports_it() {
    let backend = Backend::detect();
    assert!(backend.prefers_cropped_420_region(64, 9));
    assert!(should_use_direct_420_crop(
        backend,
        DownscaleFactor::Quarter,
        64,
        9
    ));
    assert!(should_use_direct_420_crop(
        backend,
        DownscaleFactor::Eighth,
        64,
        9
    ));
}

#[test]
fn fast420_vertical_context_only_keeps_neighbor_stripes_when_needed() {
    let middle_roi = Rect {
        x: 0,
        y: 76,
        w: 256,
        h: 256,
    };
    assert_eq!(fast420_first_decode_mcu_row(middle_roi, 16), 4);
    assert_eq!(fast420_decode_mcu_row_end(middle_roi, 16, 26), 21);

    let top_edge_roi = Rect {
        x: 0,
        y: 64,
        w: 32,
        h: 16,
    };
    assert_eq!(fast420_first_decode_mcu_row(top_edge_roi, 16), 3);

    let bottom_edge_roi = Rect {
        x: 0,
        y: 78,
        w: 32,
        h: 18,
    };
    assert_eq!(fast420_decode_mcu_row_end(bottom_edge_roi, 16, 26), 7);
}

#[test]
fn deposit_block_writes_expected_rows_at_offset() {
    let mut plane = vec![0xA5u8; 16 * 16];
    let mut block = [0u8; 64];
    for (i, byte) in block.iter_mut().enumerate() {
        *byte = u8::try_from(i).expect("8x8 block index fits in u8");
    }

    deposit_block(&mut plane, 16, 3, 2, &block);

    for row in 0..8usize {
        let dst_start = (2 + row) * 16 + 3;
        assert_eq!(
            &plane[dst_start..dst_start + 8],
            &block[row * 8..row * 8 + 8]
        );
        assert_eq!(plane[(2 + row) * 16 + 2], 0xA5);
        assert_eq!(plane[(2 + row) * 16 + 11], 0xA5);
    }
    assert_eq!(plane[0], 0xA5);
    assert_eq!(plane[plane.len() - 1], 0xA5);
}

#[test]
fn deposit_block_writes_expected_rows_at_bottom_right_edge() {
    let mut plane = vec![0x5Au8; 16 * 16];
    let mut block = [0u8; 64];
    for (i, byte) in block.iter_mut().enumerate() {
        *byte = 255u8.wrapping_sub(u8::try_from(i).expect("8x8 block index fits in u8"));
    }

    deposit_block(&mut plane, 16, 8, 8, &block);

    for row in 0..8usize {
        let dst_start = (8 + row) * 16 + 8;
        assert_eq!(
            &plane[dst_start..dst_start + 8],
            &block[row * 8..row * 8 + 8]
        );
        assert_eq!(plane[(8 + row) * 16 + 7], 0x5A);
    }
    assert_eq!(plane[plane.len() - 1], block[63]);
}

#[test]
fn deposit_dc_block_writes_uniform_rows_without_temp_block() {
    let mut plane = vec![0u8; 16 * 16];
    deposit_dc_block(&mut plane, 16, 4, 5, 217);

    for row in 5..13 {
        assert_eq!(&plane[row * 16 + 4..row * 16 + 12], &[217; 8]);
    }
    assert!(plane[..5 * 16].iter().all(|&value| value == 0));
    assert!(plane[13 * 16..].iter().all(|&value| value == 0));
}

#[test]
fn component_row_triplet_uses_neighbor_stripes_and_clamps_edges() {
    let prev = StripeBuffer {
        planes: vec![vec![], vec![10, 11, 12, 13, 14, 15], vec![]],
        plane_strides: vec![0, 2, 0],
        plane_rows: vec![0, 3, 0],
    };
    let curr = StripeBuffer {
        planes: vec![vec![], vec![20, 21, 22, 23, 24, 25], vec![]],
        plane_strides: vec![0, 2, 0],
        plane_rows: vec![0, 3, 0],
    };
    let next = StripeBuffer {
        planes: vec![vec![], vec![30, 31, 32, 33, 34, 35], vec![]],
        plane_strides: vec![0, 2, 0],
        plane_rows: vec![0, 3, 0],
    };

    let prev_plane = Some(prev.plane(1));
    let curr_plane = curr.plane(1);
    let next_plane = Some(next.plane(1));

    let (top_prev, top_curr, top_next) =
        component_row_triplet(prev_plane, curr_plane, next_plane, 0);
    assert_eq!(top_prev, &[14, 15]);
    assert_eq!(top_curr, &[20, 21]);
    assert_eq!(top_next, &[22, 23]);

    let (mid_prev, mid_curr, mid_next) =
        component_row_triplet(prev_plane, curr_plane, next_plane, 1);
    assert_eq!(mid_prev, &[20, 21]);
    assert_eq!(mid_curr, &[22, 23]);
    assert_eq!(mid_next, &[24, 25]);

    let (bot_prev, bot_curr, bot_next) =
        component_row_triplet(prev_plane, curr_plane, next_plane, 2);
    assert_eq!(bot_prev, &[22, 23]);
    assert_eq!(bot_curr, &[24, 25]);
    assert_eq!(bot_next, &[30, 31]);

    let (clamp_prev, clamp_curr, clamp_next) = component_row_triplet(None, curr_plane, None, 0);
    assert_eq!(clamp_prev, &[20, 21]);
    assert_eq!(clamp_curr, &[20, 21]);
    assert_eq!(clamp_next, &[22, 23]);

    let (tail_prev, tail_curr, tail_next) = component_row_triplet(None, curr_plane, None, 2);
    assert_eq!(tail_prev, &[22, 23]);
    assert_eq!(tail_curr, &[24, 25]);
    assert_eq!(tail_next, &[24, 25]);
}

#[test]
fn emit_stripe_rgb_444_matches_direct_ycbcr_conversion_with_trailing_row() {
    let width = 17usize;
    let height = 7u32;
    let mut stripe = StripeBuffer {
        planes: vec![
            vec![0u8; width * 8],
            vec![0u8; width * 8],
            vec![0u8; width * 8],
        ],
        plane_strides: vec![width, width, width],
        plane_rows: vec![8, 8, 8],
    };
    for row in 0..8usize {
        for col in 0..width {
            stripe.planes[0][row * width + col] =
                u8::try_from((row * 31 + col * 7 + 11) & 0xFF).expect("fixture is byte-masked");
            stripe.planes[1][row * width + col] =
                u8::try_from((row * 17 + col * 13 + 97) & 0xFF).expect("fixture is byte-masked");
            stripe.planes[2][row * width + col] =
                u8::try_from((row * 23 + col * 19 + 53) & 0xFF).expect("fixture is byte-masked");
        }
    }

    let plan = PreparedDecodePlan {
        components: vec![],
        huffman_tables: PreparedHuffmanTables::try_with_capacity(0).expect("empty test arena"),
        sampling: SamplingFactors::from_validated_components(&[(1, 1), (1, 1), (1, 1)]),
        color_space: ColorSpace::YCbCr,
        restart_interval: None,
        dimensions: (
            u32::try_from(width).expect("fixture width fits in u32"),
            height,
        ),
        scan_offset: 0,
        scratch_bytes: 0,
    };
    let mut actual = vec![0u8; width * height as usize * 3];
    let mut writer = Rgb8Writer::new(
        &mut actual,
        width * 3,
        u32::try_from(width).expect("fixture width fits in u32"),
    );

    emit_stripe_rgb_444(&plan, Backend::detect(), &stripe, 0, &mut writer)
        .expect("emit stripe must succeed");

    let mut expected = vec![0u8; actual.len()];
    for row in 0..height as usize {
        for col in 0..width {
            let y = stripe.planes[0][row * width + col];
            let cb = stripe.planes[1][row * width + col];
            let cr = stripe.planes[2][row * width + col];
            let (r, g, b) = crate::color::ycbcr::ycbcr_to_rgb(y, cb, cr);
            let dst = (row * width + col) * 3;
            expected[dst] = r;
            expected[dst + 1] = g;
            expected[dst + 2] = b;
        }
    }

    assert_eq!(actual, expected);
}

fn trivial_dc_table() -> HuffmanTable {
    HuffmanTable::from_raw(
        &RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0]),
        },
        HuffmanTableRole::Dc,
    )
    .expect("trivial DC table must be valid")
}

fn eob_ac_table() -> HuffmanTable {
    HuffmanTable::from_raw(
        &RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0x00]),
        },
        HuffmanTableRole::Ac,
    )
    .expect("trivial AC table must be valid")
}
