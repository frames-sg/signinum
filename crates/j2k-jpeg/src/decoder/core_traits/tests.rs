// SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;
use j2k_core::{DecodeRowsError, ImageDecode, ImageDecodeRows, TileBatchDecode};

use super::{
    core_outcome, CroppedWriter, Decoder, DecoderContext, Downscale, DownscaleFactor,
    InterleavedRgbWriter, JpegCodec, JpegError, OutputWriter, PixelFormat,
    ProgressiveDownscaleWriter, Rect, RowSink, ScratchPool, TileRegionScaledDecodeJob, Warning,
};

const JPEG: &[u8] = j2k_test_support::JPEG_BASELINE_420_16X16;

type RecordedComponentRow = (u32, Vec<u8>, Vec<u8>, Vec<u8>);

#[derive(Default)]
struct RecordedRows {
    rgb: Vec<RecordedComponentRow>,
    ycbcr: Vec<RecordedComponentRow>,
    gray: Vec<(u32, Vec<u8>)>,
    interleaved: Vec<(u32, Vec<u8>, Option<Vec<u8>>)>,
    interleaved_row_len: usize,
}

impl OutputWriter for RecordedRows {
    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), JpegError> {
        self.rgb
            .push((y, r_row.to_vec(), g_row.to_vec(), b_row.to_vec()));
        Ok(())
    }

    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
    ) -> Result<(), JpegError> {
        self.ycbcr
            .push((y, y_row.to_vec(), cb_row.to_vec(), cr_row.to_vec()));
        Ok(())
    }

    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), JpegError> {
        self.gray.push((y, gray_row.to_vec()));
        Ok(())
    }
}

impl InterleavedRgbWriter for RecordedRows {
    fn with_rgb_rows<R, F>(&mut self, y: u32, row_count: usize, fill: F) -> Result<R, JpegError>
    where
        F: FnOnce(&mut [u8], Option<&mut [u8]>) -> Result<R, JpegError>,
    {
        let mut top = vec![0; self.interleaved_row_len];
        let mut bottom = (row_count == 2).then(|| vec![0; self.interleaved_row_len]);
        let result = fill(&mut top, bottom.as_deref_mut())?;
        self.interleaved.push((y, top, bottom));
        Ok(result)
    }
}

fn decode_direct(
    roi: Option<Rect>,
    scale: Downscale,
) -> (Vec<u8>, j2k_core::DecodeOutcome<Warning>) {
    let decoder = Decoder::new(JPEG).expect("fixture decoder");
    let source = roi.unwrap_or_else(|| Rect::full((16, 16)));
    let output_width = source.w.div_ceil(scale.denominator());
    let output_height = source.h.div_ceil(scale.denominator());
    let mut out = vec![0; output_width as usize * output_height as usize * 3];
    let outcome = decoder
        .decode_region_scaled_into(
            &mut out,
            output_width as usize * 3,
            PixelFormat::Rgb8,
            source,
            scale,
        )
        .expect("direct reference decode");
    (out, core_outcome(outcome))
}

#[test]
fn core_image_decode_adapter_preserves_full_region_and_scaled_results() {
    let info = <Decoder<'_> as ImageDecode<'_>>::inspect(JPEG).expect("core inspect");
    assert_eq!(info.dimensions, (16, 16));

    let view = <Decoder<'_> as ImageDecode<'_>>::parse(JPEG).expect("core parse");
    let mut decoder =
        <Decoder<'_> as ImageDecode<'_>>::from_view(view).expect("core decoder from view");

    let (expected_full, expected_full_outcome) = decode_direct(None, Downscale::None);
    let mut full = vec![0; expected_full.len()];
    let full_outcome = <Decoder<'_> as ImageDecode<'_>>::decode_into(
        &mut decoder,
        &mut full,
        16 * 3,
        PixelFormat::Rgb8,
    )
    .expect("core full decode");
    assert_eq!((full, full_outcome), (expected_full, expected_full_outcome));

    let mut pool = ScratchPool::new();
    let (expected_scratch, expected_scratch_outcome) = decode_direct(None, Downscale::None);
    let mut scratch = vec![0; expected_scratch.len()];
    let scratch_outcome = <Decoder<'_> as ImageDecode<'_>>::decode_into_with_scratch(
        &mut decoder,
        &mut pool,
        &mut scratch,
        16 * 3,
        PixelFormat::Rgb8,
    )
    .expect("core full decode with scratch");
    assert_eq!(
        (scratch, scratch_outcome),
        (expected_scratch, expected_scratch_outcome)
    );

    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let (expected_region, expected_region_outcome) = decode_direct(Some(roi), Downscale::None);
    let mut region = vec![0; expected_region.len()];
    let region_outcome = <Decoder<'_> as ImageDecode<'_>>::decode_region_into(
        &mut decoder,
        &mut pool,
        &mut region,
        8 * 3,
        PixelFormat::Rgb8,
        roi.into(),
    )
    .expect("core region decode");
    assert_eq!(
        (region, region_outcome),
        (expected_region, expected_region_outcome)
    );

    let (expected_scaled, expected_scaled_outcome) = decode_direct(None, Downscale::Half);
    let mut scaled = vec![0; expected_scaled.len()];
    let scaled_outcome = <Decoder<'_> as ImageDecode<'_>>::decode_scaled_into(
        &mut decoder,
        &mut pool,
        &mut scaled,
        8 * 3,
        PixelFormat::Rgb8,
        Downscale::Half,
    )
    .expect("core scaled decode");
    assert_eq!(
        (scaled, scaled_outcome),
        (expected_scaled, expected_scaled_outcome)
    );

    let (expected_region_scaled, expected_region_scaled_outcome) =
        decode_direct(Some(roi), Downscale::Half);
    let mut region_scaled = vec![0; expected_region_scaled.len()];
    let region_scaled_outcome = <Decoder<'_> as ImageDecode<'_>>::decode_region_scaled_into(
        &mut decoder,
        &mut pool,
        &mut region_scaled,
        4 * 3,
        PixelFormat::Rgb8,
        roi.into(),
        Downscale::Half,
    )
    .expect("core region-scaled decode");
    assert_eq!(
        (region_scaled, region_scaled_outcome),
        (expected_region_scaled, expected_region_scaled_outcome)
    );
}

#[test]
fn core_tile_adapter_preserves_full_region_and_scaled_results() {
    let mut context = DecoderContext::new();
    let mut pool = ScratchPool::new();

    let (expected_full, expected_full_outcome) = decode_direct(None, Downscale::None);
    let mut full = vec![0; expected_full.len()];
    let full_outcome = <JpegCodec as TileBatchDecode>::decode_tile(
        &mut context,
        &mut pool,
        JPEG,
        &mut full,
        16 * 3,
        PixelFormat::Rgb8,
    )
    .expect("core tile decode");
    assert_eq!((full, full_outcome), (expected_full, expected_full_outcome));

    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let (expected_region, expected_region_outcome) = decode_direct(Some(roi), Downscale::None);
    let mut region = vec![0; expected_region.len()];
    let region_outcome = <JpegCodec as TileBatchDecode>::decode_tile_region(
        &mut context,
        &mut pool,
        JPEG,
        &mut region,
        8 * 3,
        PixelFormat::Rgb8,
        roi.into(),
    )
    .expect("core tile region decode");
    assert_eq!(
        (region, region_outcome),
        (expected_region, expected_region_outcome)
    );

    let (expected_scaled, expected_scaled_outcome) = decode_direct(None, Downscale::Half);
    let mut scaled = vec![0; expected_scaled.len()];
    let scaled_outcome = <JpegCodec as TileBatchDecode>::decode_tile_scaled(
        &mut context,
        &mut pool,
        JPEG,
        &mut scaled,
        8 * 3,
        PixelFormat::Rgb8,
        Downscale::Half,
    )
    .expect("core scaled tile decode");
    assert_eq!(
        (scaled, scaled_outcome),
        (expected_scaled, expected_scaled_outcome)
    );

    let (expected_region_scaled, expected_region_scaled_outcome) =
        decode_direct(Some(roi), Downscale::Half);
    let mut region_scaled = vec![0; expected_region_scaled.len()];
    let region_scaled_outcome = <JpegCodec as TileBatchDecode>::decode_tile_region_scaled(
        &mut context,
        &mut pool,
        PixelFormat::Rgb8,
        TileRegionScaledDecodeJob {
            input: JPEG,
            out: &mut region_scaled,
            stride: 4 * 3,
            roi: roi.into(),
            scale: Downscale::Half,
        },
    )
    .expect("core region-scaled tile decode");
    assert_eq!(
        (region_scaled, region_scaled_outcome),
        (expected_region_scaled, expected_region_scaled_outcome)
    );
}

#[derive(Debug, Eq, PartialEq)]
struct SinkStopped;

impl fmt::Display for SinkStopped {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("sink stopped")
    }
}

impl std::error::Error for SinkStopped {}

struct RejectFirstRow;

impl RowSink<u8> for RejectFirstRow {
    type Error = SinkStopped;

    fn write_row(&mut self, _y: u32, _row: &[u8]) -> Result<(), Self::Error> {
        Err(SinkStopped)
    }
}

#[test]
fn core_row_adapter_preserves_the_original_sink_error_type() {
    let mut decoder = Decoder::new(JPEG).expect("fixture decoder");
    let error =
        <Decoder<'_> as ImageDecodeRows<'_, u8>>::decode_rows(&mut decoder, &mut RejectFirstRow)
            .expect_err("sink rejection must stop row decode");

    assert!(matches!(error, DecodeRowsError::Sink(SinkStopped)));
}

#[test]
fn progressive_downscale_writer_samples_each_output_mode_and_skips_intermediate_rows() {
    let mut rows = RecordedRows::default();
    {
        let mut writer = ProgressiveDownscaleWriter::new(&mut rows, DownscaleFactor::Half, (5, 4))
            .expect("bounded progressive row scratch");
        assert!(writer.capacity_bytes().expect("row capacity") >= 9);

        writer
            .write_rgb_row(1, &[1; 5], &[2; 5], &[3; 5])
            .expect("skipped RGB row");
        writer
            .write_rgb_row(
                2,
                &[1, 2, 3, 4, 5],
                &[6, 7, 8, 9, 10],
                &[11, 12, 13, 14, 15],
            )
            .expect("sampled RGB row");
        writer
            .write_ycbcr_row(0, &[21, 22, 23, 24, 25], &[31; 5], &[41; 5])
            .expect("sampled YCbCr row");
        writer
            .write_gray_row(1, &[51; 5])
            .expect("skipped grayscale row");
        writer
            .write_gray_row(2, &[51, 52, 53, 54, 55])
            .expect("sampled grayscale row");
    }

    assert_eq!(
        rows.rgb,
        vec![(1, vec![1, 3, 5], vec![6, 8, 10], vec![11, 13, 15])]
    );
    assert_eq!(
        rows.ycbcr,
        vec![(0, vec![21, 23, 25], vec![31; 3], vec![41; 3])]
    );
    assert_eq!(rows.gray, vec![(1, vec![51, 53, 55])]);
}

#[test]
fn cropped_interleaved_writer_emits_only_rows_inside_the_source_window() {
    let inner = RecordedRows {
        interleaved_row_len: 2 * 3,
        ..RecordedRows::default()
    };
    let mut writer = CroppedWriter::new(
        inner,
        Rect {
            x: 1,
            y: 1,
            w: 2,
            h: 2,
        },
        0,
        4,
    )
    .expect("bounded crop writer");
    let top = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let bottom = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31];

    let result = writer
        .with_rgb_rows(0, 2, |dst_top, dst_bottom| {
            dst_top.copy_from_slice(&top);
            dst_bottom
                .expect("second source row")
                .copy_from_slice(&bottom);
            Ok(7)
        })
        .expect("bottom-only crop");
    assert_eq!(result, 7);
    writer
        .with_rgb_rows(1, 2, |dst_top, dst_bottom| {
            dst_top.copy_from_slice(&top);
            dst_bottom
                .expect("second source row")
                .copy_from_slice(&bottom);
            Ok(())
        })
        .expect("two-row crop");
    writer
        .with_rgb_rows(2, 1, |dst_top, dst_bottom| {
            assert!(dst_bottom.is_none());
            dst_top.copy_from_slice(&top);
            Ok(())
        })
        .expect("top-only crop");
    writer
        .with_rgb_rows(3, 1, |dst_top, _| {
            dst_top.copy_from_slice(&top);
            Ok(())
        })
        .expect("fully skipped crop");

    assert_eq!(
        writer.inner.interleaved,
        vec![
            (0, bottom[3..9].to_vec(), None),
            (0, top[3..9].to_vec(), Some(bottom[3..9].to_vec())),
            (1, top[3..9].to_vec(), None),
        ]
    );
}

#[test]
fn cropped_interleaved_writer_failed_row_reservation_is_transactional() {
    let mut writer = CroppedWriter::new(
        RecordedRows::default(),
        Rect {
            x: 0,
            y: 0,
            w: 1,
            h: 1,
        },
        0,
        1,
    )
    .expect("initial crop geometry");
    writer.rgb_row_len = usize::MAX;
    writer.rgb_rows_bytes = usize::MAX;
    writer.top_row.push(1);
    writer.bottom_row.push(2);

    let error = writer
        .with_rgb_rows(0, 1, |_, _| -> Result<(), JpegError> {
            unreachable!("failed reservation must not invoke the row fill")
        })
        .expect_err("impossible row reservation must fail");

    assert!(matches!(
        error,
        JpegError::HostAllocationFailed { .. } | JpegError::MemoryCapExceeded { .. }
    ));
    assert!(writer.top_row.is_empty());
    assert!(writer.bottom_row.is_empty());
    assert!(writer.inner.interleaved.is_empty());
}

#[test]
fn core_tile_context_route_reuses_cached_plan_without_cloning() {
    let (expected, _) = decode_direct(None, Downscale::None);
    let stride = 16 * PixelFormat::Rgb8.bytes_per_pixel();
    let mut output = vec![0u8; expected.len()];
    let mut context = DecoderContext::new();
    let mut pool = ScratchPool::new();

    <JpegCodec as TileBatchDecode>::decode_tile(
        &mut context,
        &mut pool,
        JPEG,
        &mut output,
        stride,
        PixelFormat::Rgb8,
    )
    .expect("warm core tile context");
    let clones = context.decode_plan_clone_count();
    <JpegCodec as TileBatchDecode>::decode_tile(
        &mut context,
        &mut pool,
        JPEG,
        &mut output,
        stride,
        PixelFormat::Rgb8,
    )
    .expect("reuse core tile context");

    assert_eq!(output, expected);
    assert_eq!(context.decode_plan_clone_count(), clones);
}
