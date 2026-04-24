use slidecodec_jpeg::{ColorSpace, Decoder, Warning};

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
fn hidden_device_plan_scan_bytes_keep_terminal_eoi() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let plan = slidecodec_jpeg::__private::build_device_plan(&decoder, 4).expect("device plan");

    assert!(plan.scan_bytes.ends_with(&[0xff, 0xd9]));
}

#[test]
fn hidden_device_plan_checkpoint_cadence_handles_multi_mcu_inputs() {
    let bytes = grayscale_jpeg(24, 24);
    let decoder = Decoder::new(&bytes).expect("grayscale decoder");

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
    let zero_offsets = cadence_zero
        .checkpoints
        .iter()
        .map(|checkpoint| checkpoint.scan_offset)
        .collect::<Vec<_>>();
    assert_eq!(zero_offsets.first(), Some(&0));
    assert!(zero_offsets.windows(2).all(|pair| pair[0] <= pair[1]));
    assert_eq!(
        cadence_two
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.mcu_index)
            .collect::<Vec<_>>(),
        vec![0, 2, 4, 6, 8]
    );
    let cadence_two_offsets = cadence_two
        .checkpoints
        .iter()
        .map(|checkpoint| checkpoint.scan_offset)
        .collect::<Vec<_>>();
    assert_eq!(cadence_two_offsets.first(), Some(&0));
    assert!(cadence_two_offsets
        .windows(2)
        .all(|pair| pair[0] <= pair[1]));
    assert!(cadence_two
        .checkpoints
        .iter()
        .all(|checkpoint| checkpoint.bits_buffered <= 64 && checkpoint.expected_rst == 0));
}

#[test]
fn hidden_device_plan_restart_checkpoints_capture_resume_state() {
    let bytes = restart_coded_grayscale_jpeg(24, 24);
    let decoder = Decoder::new(&bytes).expect("restart-coded decoder");
    let plan = slidecodec_jpeg::__private::build_device_plan(&decoder, 2).expect("device plan");

    assert_eq!(
        plan.checkpoints
            .iter()
            .map(|checkpoint| checkpoint.mcu_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8]
    );
    assert_eq!(
        plan.checkpoints
            .iter()
            .map(|checkpoint| checkpoint.scan_offset)
            .collect::<Vec<_>>(),
        vec![0, 3, 6, 9, 12, 15, 18, 21, 24]
    );
    assert_eq!(
        plan.checkpoints
            .iter()
            .map(|checkpoint| checkpoint.expected_rst)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 0]
    );
    assert!(plan
        .checkpoints
        .iter()
        .all(|checkpoint| checkpoint.bits_buffered == 0 && checkpoint.prev_dc == [0; 4]));
}

#[test]
fn hidden_device_plan_treats_dri_zero_as_non_restart_fast_path() {
    let bytes = insert_restart_interval(BASELINE_420.to_vec(), 0);
    let decoder = Decoder::new(&bytes).expect("decoder");
    let plan = slidecodec_jpeg::__private::build_device_plan(&decoder, 2).expect("device plan");

    assert_eq!(plan.restart_interval, None);
    assert!(plan.matches_fast_420);
    assert_eq!(
        plan.checkpoints
            .iter()
            .map(|checkpoint| checkpoint.expected_rst)
            .collect::<Vec<_>>(),
        vec![0; plan.checkpoints.len()]
    );
}

#[test]
fn hidden_device_plan_handles_restart_after_partial_entropy_byte() {
    let bytes = grayscale_restart_jpeg();
    let decoder = Decoder::new(&bytes).expect("restart-coded decoder");
    let plan = slidecodec_jpeg::__private::build_device_plan(&decoder, 2).expect("device plan");

    assert_eq!(plan.checkpoints.len(), 2);
    assert_eq!(plan.checkpoints[1].mcu_index, 1);
    assert_eq!(plan.checkpoints[1].scan_offset, 3);
    assert_eq!(plan.checkpoints[1].expected_rst, 1);
}

#[test]
fn hidden_device_plan_surfaces_missing_eoi_warning() {
    let mut bytes = grayscale_jpeg(24, 24);
    bytes.truncate(bytes.len() - 2);

    let decoder = Decoder::new(&bytes).expect("decoder");
    let plan = slidecodec_jpeg::__private::build_device_plan(&decoder, 2)
        .expect("missing EOI should remain decodable");

    assert!(plan.warnings.contains(&Warning::MissingEoi));
}

#[test]
fn hidden_device_plan_treats_trailing_ff_as_missing_eoi() {
    let mut bytes = grayscale_jpeg(24, 24);
    bytes.truncate(bytes.len() - 1);

    let decoder = Decoder::new(&bytes).expect("decoder");
    let plan = slidecodec_jpeg::__private::build_device_plan(&decoder, 2)
        .expect("trailing FF should remain decodable");

    assert!(plan.warnings.contains(&Warning::MissingEoi));
    assert_eq!(plan.scan_bytes.last(), Some(&0xff));
}

#[test]
fn hidden_device_plan_rejects_non_eoi_marker_after_entropy() {
    let mut bytes = restart_coded_grayscale_jpeg(24, 24);
    let marker = bytes
        .windows(2)
        .position(|window| matches!(window, [0xff, 0xd0..=0xd7]))
        .expect("restart marker");
    bytes[marker + 1] = 0xe0;

    let decoder = Decoder::new(&bytes).expect("restart-coded decoder");
    let err = slidecodec_jpeg::__private::build_device_plan(&decoder, 2)
        .expect_err("unexpected marker should fail");

    assert!(matches!(
        err,
        slidecodec_jpeg::JpegError::UnexpectedMarker {
            expected: slidecodec_jpeg::MarkerKind::Eoi,
            found: 0xe0,
            ..
        }
    ));
}

#[test]
fn hidden_device_plan_rejects_restart_marker_without_dri() {
    let bytes = insert_entropy_marker(BASELINE_420.to_vec(), 0xd0);
    let decoder = Decoder::new(&bytes).expect("decoder");
    let err = slidecodec_jpeg::__private::build_device_plan(&decoder, 2)
        .expect_err("restart marker without DRI must fail");

    assert!(matches!(
        err,
        slidecodec_jpeg::JpegError::UnexpectedMarker {
            expected: slidecodec_jpeg::MarkerKind::Eoi,
            found: 0xd0,
            ..
        }
    ));
}

#[test]
fn hidden_device_plan_rejects_doubled_ff_before_terminal_eoi() {
    let mut bytes = grayscale_jpeg(24, 24);
    bytes.insert(bytes.len() - 1, 0xff);

    let decoder = Decoder::new(&bytes).expect("decoder");
    let err = slidecodec_jpeg::__private::build_device_plan(&decoder, 2)
        .expect_err("double-FF terminal marker should fail");

    assert!(matches!(
        err,
        slidecodec_jpeg::JpegError::UnexpectedMarker {
            expected: slidecodec_jpeg::MarkerKind::Eoi,
            found: 0xff,
            ..
        }
    ));
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

fn grayscale_jpeg(width: u16, height: u16) -> Vec<u8> {
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
    bytes.extend(std::iter::repeat_n(0x00, mcu_count));

    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn grayscale_restart_jpeg() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(std::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[0xff, 0xc0, 0x00, 11, 8, 0, 8, 0, 16, 1, 1, 0x11, 0]);
    bytes.extend_from_slice(&[0xff, 0xdd, 0x00, 0x04, 0x00, 0x01]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, 0, 63, 0]);
    bytes.extend_from_slice(&[0x00, 0xff, 0xd0, 0x00, 0xff, 0xd9]);
    bytes
}

fn insert_restart_interval(mut bytes: Vec<u8>, interval: u16) -> Vec<u8> {
    let sos = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xda])
        .expect("SOS marker");
    bytes.splice(
        sos..sos,
        [
            0xff,
            0xdd,
            0x00,
            0x04,
            (interval >> 8) as u8,
            interval as u8,
        ],
    );
    bytes
}

fn insert_entropy_marker(mut bytes: Vec<u8>, marker: u8) -> Vec<u8> {
    bytes.splice(bytes.len() - 2..bytes.len() - 2, [0xff, marker]);
    bytes
}
