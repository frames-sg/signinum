// SPDX-License-Identifier: Apache-2.0

use signinum_j2k::{
    decode_tiles_into, decode_tiles_region_scaled_into, Downscale, J2kDecoder, PixelFormat, Rect,
    TileBatchOptions, TileDecodeJob, TileRegionScaledDecodeJob,
};
use signinum_j2k_native::{encode, EncodeOptions};
use std::num::NonZeroUsize;

fn encode_codestream(
    pixels: &[u8],
    width: u32,
    height: u32,
    components: u8,
    bit_depth: u8,
) -> Vec<u8> {
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(
        pixels, width, height, components, bit_depth, false, &options,
    )
    .expect("encode")
}

fn rgb_fixture() -> Vec<u8> {
    let pixels = (0_u8..48).collect::<Vec<_>>();
    encode_codestream(&pixels, 4, 4, 3, 8)
}

fn decode_rgb8_reference(bytes: &[u8]) -> (Vec<u8>, usize) {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    let (width, height) = decoder.info().dimensions;
    let stride = width as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut out = vec![0_u8; stride * height as usize];
    decoder
        .decode_into(&mut out, stride, PixelFormat::Rgb8)
        .expect("decode reference");
    (out, stride)
}

#[test]
fn production_batch_decode_empty_input_succeeds() {
    let mut jobs: Vec<TileDecodeJob<'_, '_>> = Vec::new();

    let outcomes = decode_tiles_into(&mut jobs, PixelFormat::Rgb8, TileBatchOptions::default())
        .expect("empty batch succeeds");

    assert!(outcomes.is_empty());
}

#[test]
fn production_batch_decode_worker_one_matches_single_tile_decode() {
    let codestream = rgb_fixture();
    let (expected, stride) = decode_rgb8_reference(&codestream);
    let mut actual = vec![0_u8; expected.len()];
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(1),
    };

    let outcomes = {
        let mut jobs = vec![TileDecodeJob {
            input: &codestream,
            out: actual.as_mut_slice(),
            stride,
        }];
        decode_tiles_into(&mut jobs, PixelFormat::Rgb8, options).expect("batch decode")
    };

    assert_eq!(outcomes.len(), 1);
    assert_eq!(actual, expected);
}

#[test]
fn production_batch_decode_parallel_preserves_order_and_output() {
    const JOBS: usize = 16;
    let codestream = rgb_fixture();
    let (expected, stride) = decode_rgb8_reference(&codestream);
    let mut outputs = (0..JOBS)
        .map(|_| vec![0_u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(4),
    };

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileDecodeJob {
                input: codestream.as_slice(),
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        decode_tiles_into(&mut jobs, PixelFormat::Rgb8, options).expect("batch decode")
    };

    assert_eq!(outcomes.len(), JOBS);
    for (index, out) in outputs.iter().enumerate() {
        assert_eq!(out, &expected, "tile {index} output diverged");
    }
}

#[test]
fn production_batch_decode_matches_repeated_single_tile_decodes() {
    let inputs = [
        rgb_fixture(),
        encode_codestream(&(48_u8..96).collect::<Vec<_>>(), 4, 4, 3, 8),
        encode_codestream(&(96_u8..144).collect::<Vec<_>>(), 4, 4, 3, 8),
    ];
    let expected = inputs
        .iter()
        .map(|input| decode_rgb8_reference(input).0)
        .collect::<Vec<_>>();
    let stride = 4 * PixelFormat::Rgb8.bytes_per_pixel();
    let mut outputs = expected
        .iter()
        .map(|tile| vec![0_u8; tile.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(2),
    };

    let outcomes = {
        let mut jobs = inputs
            .iter()
            .zip(outputs.iter_mut())
            .map(|(input, out)| TileDecodeJob {
                input: input.as_slice(),
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        decode_tiles_into(&mut jobs, PixelFormat::Rgb8, options).expect("batch decode")
    };

    assert_eq!(outcomes.len(), inputs.len());
    assert_eq!(outputs, expected);
}

#[test]
fn production_batch_region_scaled_decode_parallel_preserves_order_and_output() {
    const JOBS: usize = 12;
    let codestream = rgb_fixture();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let scale = Downscale::Half;
    let scaled_roi = roi.scaled_covering(scale);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel();

    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut pool = signinum_j2k::J2kScratchPool::new();
    let mut expected = vec![0_u8; stride * scaled_roi.h as usize];
    decoder
        .decode_region_scaled_into(
            &mut pool,
            &mut expected,
            stride,
            PixelFormat::Rgb8,
            roi,
            scale,
        )
        .expect("decode reference");

    let mut outputs = (0..JOBS)
        .map(|_| vec![0_u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(3),
    };

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileRegionScaledDecodeJob {
                input: codestream.as_slice(),
                out: out.as_mut_slice(),
                stride,
                roi,
                scale,
            })
            .collect::<Vec<_>>();
        decode_tiles_region_scaled_into(&mut jobs, PixelFormat::Rgb8, options)
            .expect("batch decode")
    };

    assert_eq!(outcomes.len(), JOBS);
    for outcome in &outcomes {
        assert_eq!(outcome.decoded, scaled_roi);
    }
    for (index, out) in outputs.iter().enumerate() {
        assert_eq!(out, &expected, "tile {index} output diverged");
    }
}

#[test]
fn production_batch_decode_reports_first_failing_tile_index() {
    let codestream = rgb_fixture();
    let (expected, stride) = decode_rgb8_reference(&codestream);
    let mut outputs = (0..3)
        .map(|_| vec![0_u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(2),
    };

    let err = {
        let inputs: [&[u8]; 3] = [codestream.as_slice(), b"not j2k", codestream.as_slice()];
        let mut jobs = inputs
            .into_iter()
            .zip(outputs.iter_mut())
            .map(|(input, out)| TileDecodeJob {
                input,
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        decode_tiles_into(&mut jobs, PixelFormat::Rgb8, options).expect_err("bad tile fails")
    };

    assert_eq!(err.index, 1);
}
