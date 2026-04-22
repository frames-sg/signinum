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
