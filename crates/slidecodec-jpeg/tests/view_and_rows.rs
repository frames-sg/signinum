// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the parsed-view API and row-streaming decode surface.

use slidecodec_jpeg::{Decoder, JpegError, JpegView, OutputFormat, RowSink};

mod fixtures;
use fixtures::{grayscale_8x8_jpeg, minimal_baseline_420_jpeg, rgb_app14_8x8_jpeg};

#[derive(Default)]
struct CollectRows {
    rows: Vec<(u32, Vec<u8>)>,
}

impl RowSink<u8> for CollectRows {
    type Error = JpegError;

    fn write_row(&mut self, y: u32, row: &[u8]) -> Result<(), JpegError> {
        self.rows.push((y, row.to_vec()));
        Ok(())
    }
}

#[test]
fn jpeg_view_parse_matches_decoder_inspect() {
    let bytes = minimal_baseline_420_jpeg();
    let view = JpegView::parse(&bytes).expect("parsed view must construct");
    let info = Decoder::inspect(&bytes).expect("inspect must succeed");
    assert_eq!(view.info(), &info);
}

#[test]
fn decoder_from_view_matches_decoder_new_rgb_output() {
    let bytes = rgb_app14_8x8_jpeg();
    let dec_from_new = Decoder::new(&bytes).expect("decoder::new must succeed");
    let dec_from_view = Decoder::from_view(JpegView::parse(&bytes).unwrap())
        .expect("decoder::from_view must succeed");

    let (w, h) = dec_from_new.info().dimensions;
    let stride = (w * 3) as usize;
    let mut new_out = vec![0u8; stride * h as usize];
    let mut view_out = vec![0u8; stride * h as usize];

    dec_from_new
        .decode_into(&mut new_out, stride, OutputFormat::Rgb8)
        .unwrap();
    dec_from_view
        .decode_into(&mut view_out, stride, OutputFormat::Rgb8)
        .unwrap();

    assert_eq!(view_out, new_out);
}

#[test]
fn decode_rows_matches_decode_into_rgb8() {
    let bytes = minimal_baseline_420_jpeg();
    let dec = Decoder::new(&bytes).expect("decoder::new must succeed");
    let (w, h) = dec.info().dimensions;
    let stride = (w * 3) as usize;

    let mut expected = vec![0u8; stride * h as usize];
    dec.decode_into(&mut expected, stride, OutputFormat::Rgb8)
        .unwrap();

    let mut sink = CollectRows::default();
    dec.decode_rows(&mut sink)
        .expect("decode_rows must succeed");

    assert_eq!(sink.rows.len(), h as usize);
    for (row_idx, (y, row)) in sink.rows.iter().enumerate() {
        assert_eq!(*y as usize, row_idx);
        assert_eq!(row.len(), stride);
        assert_eq!(
            row.as_slice(),
            &expected[row_idx * stride..(row_idx + 1) * stride]
        );
    }
}

#[test]
fn decode_rows_matches_decode_into_rgb8_for_grayscale_input() {
    let bytes = grayscale_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("decoder::new must succeed");
    let (w, h) = dec.info().dimensions;
    let stride = (w * 3) as usize;

    let mut expected = vec![0u8; stride * h as usize];
    dec.decode_into(&mut expected, stride, OutputFormat::Rgb8)
        .unwrap();

    let mut sink = CollectRows::default();
    dec.decode_rows(&mut sink)
        .expect("decode_rows must succeed");

    assert_eq!(sink.rows.len(), h as usize);
    for (row_idx, (y, row)) in sink.rows.iter().enumerate() {
        assert_eq!(*y as usize, row_idx);
        assert_eq!(
            row.as_slice(),
            &expected[row_idx * stride..(row_idx + 1) * stride]
        );
    }
}
