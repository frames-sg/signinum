mod fixtures;

use slidecodec_jpeg::adapter::metal_fast420::{
    build_metal_fast420_packet, build_metal_fast422_packet, build_metal_fast444_packet,
    build_metal_gray_packet, MetalFast420PacketError,
};

fn rewrite_three_component_ids(mut bytes: Vec<u8>, component_ids: [u8; 3]) -> Vec<u8> {
    assert_eq!(&bytes[..2], &[0xff, 0xd8], "fixture must start with SOI");
    let mut pos = 2usize;
    while pos + 4 <= bytes.len() {
        assert_eq!(bytes[pos], 0xff, "marker alignment");
        let marker = bytes[pos + 1];
        pos += 2;
        if marker == 0xd9 {
            break;
        }
        let len = u16::from_be_bytes([bytes[pos], bytes[pos + 1]]) as usize;
        let payload_start = pos + 2;
        match marker {
            0xc0..=0xc2 => {
                let component_count = bytes[payload_start + 5] as usize;
                assert_eq!(component_count, 3, "expected three-component SOF");
                for (index, component_id) in component_ids.into_iter().enumerate() {
                    bytes[payload_start + 6 + index * 3] = component_id;
                }
            }
            0xda => {
                let component_count = bytes[payload_start] as usize;
                assert_eq!(component_count, 3, "expected three-component SOS");
                for (index, component_id) in component_ids.into_iter().enumerate() {
                    bytes[payload_start + 1 + index * 2] = component_id;
                }
                break;
            }
            _ => {}
        }
        pos += len;
    }
    bytes
}

fn strip_dri_and_restart_markers(bytes: &[u8]) -> Vec<u8> {
    assert_eq!(&bytes[..2], &[0xff, 0xd8], "fixture must start with SOI");
    let mut out = Vec::with_capacity(bytes.len());
    out.extend_from_slice(&bytes[..2]);

    let mut pos = 2usize;
    while pos + 4 <= bytes.len() {
        assert_eq!(bytes[pos], 0xff, "marker alignment");
        let marker = bytes[pos + 1];
        pos += 2;
        if marker == 0xd9 {
            out.extend_from_slice(&[0xff, 0xd9]);
            break;
        }
        let len = u16::from_be_bytes([bytes[pos], bytes[pos + 1]]) as usize;
        let segment_end = pos + len;
        match marker {
            0xdd => {}
            0xda => {
                out.extend_from_slice(&[0xff, marker]);
                out.extend_from_slice(&bytes[pos..segment_end]);
                let mut entropy_pos = segment_end;
                while entropy_pos + 1 < bytes.len() {
                    let byte = bytes[entropy_pos];
                    if byte != 0xff {
                        out.push(byte);
                        entropy_pos += 1;
                        continue;
                    }
                    let next = bytes[entropy_pos + 1];
                    match next {
                        0x00 => {
                            out.extend_from_slice(&[0xff, 0x00]);
                            entropy_pos += 2;
                        }
                        0xd0..=0xd7 => {
                            entropy_pos += 2;
                        }
                        0xd9 => {
                            out.extend_from_slice(&[0xff, 0xd9]);
                            return out;
                        }
                        _ => panic!("unexpected marker 0xff{next:02x} in entropy"),
                    }
                }
                panic!("fixture entropy must terminate with EOI");
            }
            _ => {
                out.extend_from_slice(&[0xff, marker]);
                out.extend_from_slice(&bytes[pos..segment_end]);
            }
        }
        pos = segment_end;
    }

    out
}

#[test]
fn baseline_420_fixture_builds_fast420_packet() {
    let bytes = fixtures::minimal_baseline_420_jpeg();
    let packet = build_metal_fast420_packet(&bytes).expect("fast420 packet");

    assert_eq!(packet.dimensions, (16, 16));
    assert_eq!(packet.mcus_per_row, 1);
    assert_eq!(packet.mcu_rows, 1);
    assert!(
        !packet.entropy_bytes.is_empty(),
        "entropy payload must be present"
    );
    assert!(packet.y_dc_table.values_len > 0);
    assert!(packet.y_ac_table.values_len > 0);
    assert!(packet.cb_dc_table.values_len > 0);
    assert!(packet.cb_ac_table.values_len > 0);
    assert!(packet.cr_dc_table.values_len > 0);
    assert!(packet.cr_ac_table.values_len > 0);
}

#[test]
fn baseline_420_restart_fixture_builds_fast420_packet() {
    let bytes = fixtures::baseline_420_restart_32x16_jpeg();
    let packet = build_metal_fast420_packet(&bytes).expect("restart fast420 packet");

    assert_eq!(packet.dimensions, (32, 16));
    assert_eq!(packet.mcus_per_row, 2);
    assert_eq!(packet.mcu_rows, 1);
    assert!(packet.restart_interval_mcus > 0);
    assert_eq!(packet.restart_offsets.first(), Some(&0));
    assert!(!packet.restart_offsets.is_empty());
    assert_eq!(
        packet.entropy_checkpoints.len(),
        packet.restart_offsets.len()
    );
    assert_eq!(packet.entropy_checkpoints[0].mcu_index, 0);
    assert_eq!(packet.entropy_checkpoints[0].entropy_pos, 0);
    assert_eq!(packet.entropy_checkpoints[0].bit_acc, 0);
    assert_eq!(packet.entropy_checkpoints[0].bit_count, 0);
}

#[test]
fn stripped_restart_fixture_builds_nonrestart_entropy_checkpoints() {
    let bytes = strip_dri_and_restart_markers(&fixtures::baseline_420_restart_32x16_jpeg());
    let packet = build_metal_fast420_packet(&bytes).expect("nonrestart fast420 packet");

    assert_eq!(packet.restart_interval_mcus, 0);
    assert_eq!(packet.restart_offsets, vec![0]);
    assert_eq!(packet.entropy_checkpoints.len(), 2);
    assert_eq!(packet.entropy_checkpoints[0].mcu_index, 0);
    assert_eq!(packet.entropy_checkpoints[0].entropy_pos, 0);
    assert_eq!(packet.entropy_checkpoints[1].mcu_index, 1);
    assert!(packet.entropy_checkpoints[1].entropy_pos > packet.entropy_checkpoints[0].entropy_pos);
}

#[test]
fn baseline_420_packet_accepts_zero_based_component_ids() {
    let bytes = rewrite_three_component_ids(fixtures::minimal_baseline_420_jpeg(), [0, 1, 2]);
    let packet = build_metal_fast420_packet(&bytes).expect("fast420 packet");

    assert_eq!(packet.dimensions, (16, 16));
    assert_eq!(packet.mcus_per_row, 1);
    assert_eq!(packet.mcu_rows, 1);
}

#[test]
fn baseline_444_fixture_builds_fast444_packet() {
    let bytes = fixtures::baseline_444_8x8_jpeg();
    let packet = build_metal_fast444_packet(&bytes).expect("fast444 packet");

    assert_eq!(packet.dimensions, (8, 8));
    assert_eq!(packet.mcus_per_row, 1);
    assert_eq!(packet.mcu_rows, 1);
    assert!(
        !packet.entropy_bytes.is_empty(),
        "entropy payload must be present"
    );
    assert_eq!(packet.entropy_checkpoints.len(), 1);
    assert!(packet.y_dc_table.values_len > 0);
    assert!(packet.y_ac_table.values_len > 0);
    assert!(packet.cb_dc_table.values_len > 0);
    assert!(packet.cb_ac_table.values_len > 0);
    assert!(packet.cr_dc_table.values_len > 0);
    assert!(packet.cr_ac_table.values_len > 0);
}

#[test]
fn baseline_444_packet_accepts_zero_based_component_ids() {
    let bytes = rewrite_three_component_ids(fixtures::baseline_444_8x8_jpeg(), [0, 1, 2]);
    let packet = build_metal_fast444_packet(&bytes).expect("fast444 packet");

    assert_eq!(packet.dimensions, (8, 8));
    assert_eq!(packet.mcus_per_row, 1);
    assert_eq!(packet.mcu_rows, 1);
}

#[test]
fn baseline_422_fixture_builds_fast422_packet() {
    let bytes = fixtures::baseline_422_16x8_jpeg();
    let packet = build_metal_fast422_packet(&bytes).expect("fast422 packet");

    assert_eq!(packet.dimensions, (16, 8));
    assert_eq!(packet.mcus_per_row, 1);
    assert_eq!(packet.mcu_rows, 1);
    assert!(
        !packet.entropy_bytes.is_empty(),
        "entropy payload must be present"
    );
    assert!(packet.y_dc_table.values_len > 0);
    assert!(packet.y_ac_table.values_len > 0);
    assert!(packet.cb_dc_table.values_len > 0);
    assert!(packet.cb_ac_table.values_len > 0);
    assert!(packet.cr_dc_table.values_len > 0);
    assert!(packet.cr_ac_table.values_len > 0);
}

#[test]
fn baseline_422_packet_accepts_zero_based_component_ids() {
    let bytes = rewrite_three_component_ids(fixtures::baseline_422_16x8_jpeg(), [0, 1, 2]);
    let packet = build_metal_fast422_packet(&bytes).expect("fast422 packet");

    assert_eq!(packet.dimensions, (16, 8));
    assert_eq!(packet.mcus_per_row, 1);
    assert_eq!(packet.mcu_rows, 1);
}

#[test]
fn grayscale_fixture_is_rejected_for_fast420_subset() {
    let bytes = fixtures::grayscale_8x8_jpeg();
    let error = build_metal_fast420_packet(&bytes).expect_err("grayscale must be rejected");

    assert!(matches!(
        error,
        MetalFast420PacketError::UnsupportedColorSpace(_)
            | MetalFast420PacketError::UnsupportedSampling
    ));
}

#[test]
fn grayscale_fixture_builds_gray_packet() {
    let bytes = fixtures::grayscale_8x8_jpeg();
    let packet = build_metal_gray_packet(&bytes).expect("gray packet");

    assert_eq!(packet.dimensions, (8, 8));
    assert_eq!(packet.mcus_per_row, 1);
    assert_eq!(packet.mcu_rows, 1);
    assert_eq!(packet.restart_offsets, vec![0]);
    assert!(
        !packet.entropy_bytes.is_empty(),
        "entropy payload must be present"
    );
    assert!(packet.y_dc_table.values_len > 0);
    assert!(packet.y_ac_table.values_len > 0);
}

#[test]
fn progressive_fixture_is_rejected_for_fast420_subset() {
    let bytes = fixtures::progressive_8x8_jpeg();
    let error = build_metal_fast420_packet(&bytes).expect_err("progressive must be rejected");

    assert!(matches!(error, MetalFast420PacketError::UnsupportedSof(_)));
}
