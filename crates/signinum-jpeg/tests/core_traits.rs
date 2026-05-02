use signinum_core::{
    CodedUnitLayout, DecoderContext as CoreDecoderContext, Downscale, ImageDecode, ImageDecodeRows,
    PixelFormat, RowSink, TileBatchDecode,
};
use signinum_jpeg::{Decoder, DecoderContext, JpegCodec, JpegError, ScratchPool};

struct CollectRows {
    rows: Vec<Vec<u8>>,
}

impl RowSink<u8> for CollectRows {
    type Error = JpegError;

    fn write_row(&mut self, _y: u32, row: &[u8]) -> Result<(), Self::Error> {
        self.rows.push(row.to_vec());
        Ok(())
    }
}

#[test]
fn decoder_implements_core_traits_for_rgb_decode() {
    let bytes = include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");
    let mut dec = <Decoder<'_> as ImageDecode<'_>>::from_view(
        <Decoder<'_> as ImageDecode<'_>>::parse(bytes).expect("parse"),
    )
    .expect("decoder");
    let info = <Decoder<'_> as ImageDecode<'_>>::inspect(bytes).expect("inspect");
    assert_eq!(info.dimensions, (16, 16));
    assert_eq!(
        info.coded_unit_layout,
        Some(CodedUnitLayout {
            unit_width: 16,
            unit_height: 16,
            units_x: 1,
            units_y: 1,
        })
    );
    assert_eq!(info.restart_interval, None);

    let mut out = vec![0u8; 16 * 16 * 3];
    let outcome = <Decoder<'_> as ImageDecode<'_>>::decode_into(
        &mut dec,
        &mut out,
        16 * 3,
        PixelFormat::Rgb8,
    )
    .expect("decode");
    assert_eq!(outcome.decoded.w, 16);
}

#[test]
fn core_inspect_exposes_restart_interval_and_coded_units_for_wsi_planning() {
    let bytes = include_bytes!("../../../corpus/conformance/baseline_420_restart_32x16.jpg");
    let info = <Decoder<'_> as ImageDecode<'_>>::inspect(bytes).expect("inspect");

    assert_eq!(info.dimensions, (32, 16));
    assert_eq!(
        info.coded_unit_layout,
        Some(CodedUnitLayout {
            unit_width: 16,
            unit_height: 16,
            units_x: 2,
            units_y: 1,
        })
    );
    assert_eq!(info.restart_interval, Some(2));
}

#[test]
fn row_and_tile_core_traits_are_callable() {
    let bytes = include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");
    let mut dec = Decoder::new(bytes).expect("decoder");
    let mut sink = CollectRows { rows: Vec::new() };
    <Decoder<'_> as ImageDecodeRows<'_, u8>>::decode_rows(&mut dec, &mut sink)
        .expect("decode_rows");
    assert!(!sink.rows.is_empty());

    let mut out = vec![0u8; 16 * 16 * 3];
    let mut pool = ScratchPool::new();
    let mut ctx = CoreDecoderContext::<DecoderContext>::new();
    JpegCodec::decode_tile_scaled(
        &mut ctx,
        &mut pool,
        bytes,
        &mut out,
        16 * 3,
        PixelFormat::Rgb8,
        Downscale::None,
    )
    .expect("tile decode");
}
