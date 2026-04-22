use slidecodec_jpeg::{ColorSpace, Decoder};

const BASELINE_420: &[u8] = include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");

#[test]
fn hidden_device_plan_exposes_scan_metadata() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let plan = slidecodec_jpeg::__private::build_device_plan(&decoder, 4).expect("device plan");

    assert_eq!(plan.dimensions, (16, 16));
    assert_eq!(plan.color_space, ColorSpace::YCbCr);
    assert_eq!(plan.components.len(), 3);
    assert_eq!(plan.checkpoints[0].mcu_index, 0);
    assert!(!plan.scan_bytes.is_empty());
}

#[test]
fn hidden_device_plan_keeps_fast_420_shape_information() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let plan = slidecodec_jpeg::__private::build_device_plan(&decoder, 4).expect("device plan");

    assert!(plan.matches_fast_420);
    assert!(!plan.matches_fast_444);
}

#[test]
fn hidden_device_plan_scan_bytes_stop_before_terminal_marker() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let plan = slidecodec_jpeg::__private::build_device_plan(&decoder, 4).expect("device plan");

    assert!(!plan.scan_bytes.ends_with(&[0xff, 0xd9]));
}

#[test]
fn hidden_device_plan_checkpoint_cadence_handles_multi_mcu_inputs() {
    let bytes = restart_coded_grayscale_jpeg(24, 24);
    let decoder = Decoder::new(&bytes).expect("restart-coded decoder");

    let cadence_zero =
        slidecodec_jpeg::__private::build_device_plan(&decoder, 0).expect("zero-cadence plan");
    let cadence_two =
        slidecodec_jpeg::__private::build_device_plan(&decoder, 2).expect("cadence-two plan");

    assert_eq!(
        cadence_zero
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.mcu_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8]
    );
    assert_eq!(
        cadence_two
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.mcu_index)
            .collect::<Vec<_>>(),
        vec![0, 2, 4, 6, 8]
    );
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
