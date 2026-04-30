use ashlar_core::Downscale;
use ashlar_jpeg::{ComponentRowWriter, Decoder, Rect, ScratchPool};

mod fixtures;

#[derive(Default)]
struct CollectRows {
    gray: Vec<Vec<u8>>,
    ycbcr: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>,
    rgb: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>,
}

impl ComponentRowWriter for CollectRows {
    fn write_gray_row(&mut self, _y: u32, gray_row: &[u8]) -> Result<(), ashlar_jpeg::JpegError> {
        self.gray.push(gray_row.to_vec());
        Ok(())
    }

    fn write_ycbcr_row(
        &mut self,
        _y: u32,
        y_row: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
    ) -> Result<(), ashlar_jpeg::JpegError> {
        self.ycbcr
            .push((y_row.to_vec(), cb_row.to_vec(), cr_row.to_vec()));
        Ok(())
    }

    fn write_rgb_row(
        &mut self,
        _y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), ashlar_jpeg::JpegError> {
        self.rgb
            .push((r_row.to_vec(), g_row.to_vec(), b_row.to_vec()));
        Ok(())
    }
}

#[test]
fn component_rows_expose_ycbcr_rows_for_full_decode() {
    let mut pool = ScratchPool::new();
    let mut rows = CollectRows::default();
    let jpeg = fixtures::minimal_baseline_420_jpeg();
    let decoder = Decoder::new(&jpeg).expect("decoder");

    decoder
        .decode_component_rows_with_scratch(&mut pool, &mut rows)
        .expect("decode component rows");

    assert!(rows.gray.is_empty());
    assert!(rows.rgb.is_empty());
    assert_eq!(rows.ycbcr.len(), 16);
    assert!(rows
        .ycbcr
        .iter()
        .all(|(y, cb, cr)| y.len() == 16 && cb.len() == 16 && cr.len() == 16));
}

#[test]
fn component_rows_region_scaled_rebases_output_rows() {
    let mut pool = ScratchPool::new();
    let mut rows = CollectRows::default();
    let jpeg = fixtures::minimal_baseline_420_jpeg();
    let decoder = Decoder::new(&jpeg).expect("decoder");
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };

    decoder
        .decode_region_component_rows_with_scratch(&mut pool, &mut rows, roi, Downscale::Half)
        .expect("decode region component rows");

    assert_eq!(rows.ycbcr.len(), 4);
    assert!(rows
        .ycbcr
        .iter()
        .all(|(y, cb, cr)| y.len() == 4 && cb.len() == 4 && cr.len() == 4));
}
