// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the parsed-view API and row-streaming decode surface.

use signinum_jpeg::{
    ComponentRowWriter, Decoder, Downscale, JpegError, JpegView, PixelFormat, Rect, RowSink,
    ScratchPool,
};

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

#[derive(Default)]
struct CollectGrayComponentRows {
    rows: Vec<(u32, Vec<u8>)>,
}

impl ComponentRowWriter for CollectGrayComponentRows {
    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), JpegError> {
        self.rows.push((y, gray_row.to_vec()));
        Ok(())
    }

    fn write_ycbcr_row(
        &mut self,
        _y: u32,
        _y_row: &[u8],
        _cb_row: &[u8],
        _cr_row: &[u8],
    ) -> Result<(), JpegError> {
        unreachable!("grayscale test writer should not receive ycbcr rows");
    }

    fn write_rgb_row(
        &mut self,
        _y: u32,
        _r_row: &[u8],
        _g_row: &[u8],
        _b_row: &[u8],
    ) -> Result<(), JpegError> {
        unreachable!("grayscale test writer should not receive rgb rows");
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
        .decode_scaled_into(&mut new_out, stride, PixelFormat::Rgb8, Downscale::None)
        .unwrap();
    dec_from_view
        .decode_scaled_into(&mut view_out, stride, PixelFormat::Rgb8, Downscale::None)
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
    dec.decode_scaled_into(&mut expected, stride, PixelFormat::Rgb8, Downscale::None)
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
    dec.decode_scaled_into(&mut expected, stride, PixelFormat::Rgb8, Downscale::None)
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

#[test]
fn decode_rows_matches_decode_into_rgb8_for_restart_coded_grayscale_wsi_shape() {
    let bytes = restart_coded_grayscale_jpeg(24, 24);
    let dec = Decoder::new(&bytes).expect("restart-coded grayscale fixture must parse");
    let (w, h) = dec.info().dimensions;
    let stride = (w * 3) as usize;

    let mut expected = vec![0u8; stride * h as usize];
    dec.decode_scaled_into(&mut expected, stride, PixelFormat::Rgb8, Downscale::None)
        .expect("full decode must succeed");

    let mut sink = CollectRows::default();
    dec.decode_rows(&mut sink)
        .expect("decode_rows must succeed on restart-coded input");

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
fn region_component_rows_scaled_matches_gray_region_decode_for_restart_fixture() {
    let bytes = restart_coded_grayscale_jpeg(24, 24);
    let dec = Decoder::new(&bytes).expect("restart-coded grayscale fixture must parse");
    let roi = Rect {
        x: 5,
        y: 6,
        w: 11,
        h: 10,
    };

    let mut pool = ScratchPool::new();
    let mut sink = CollectGrayComponentRows::default();
    dec.decode_region_component_rows_with_scratch(&mut pool, &mut sink, roi, Downscale::Half)
        .expect("scaled region component rows must decode");

    let expected = dec
        .decode_region_scaled(PixelFormat::Gray8, roi, Downscale::Half)
        .expect("scaled region decode must succeed")
        .0;

    let mut collected = Vec::new();
    for (row_idx, (y, row)) in sink.rows.iter().enumerate() {
        assert_eq!(*y as usize, row_idx);
        collected.extend_from_slice(row);
    }

    assert_eq!(collected, expected);
}

fn restart_coded_grayscale_jpeg(width: u16, height: u16) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff,
        0xc0,
        0x00,
        11,
        8,
        (height >> 8) as u8,
        height as u8,
        (width >> 8) as u8,
        width as u8,
        1,
        1,
        0x11,
        0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xdd, 0x00, 0x04, 0x00, 0x01]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, 0, 63, 0]);

    let mcu_cols = u32::from(width).div_ceil(8);
    let mcu_rows = u32::from(height).div_ceil(8);
    let mcu_count = (mcu_cols * mcu_rows) as usize;
    for mcu in 0..mcu_count {
        bytes.push(0x00);
        if mcu + 1 != mcu_count {
            bytes.extend_from_slice(&[0xff, 0xd0 | ((mcu as u8) & 0x07)]);
        }
    }

    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}
