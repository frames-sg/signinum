// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    parse_dht, parse_dqt, HuffmanTableRole, HuffmanTables, HuffmanValues, ProgressiveTableState,
    QuantTables, RawHuffmanTable,
};
use crate::error::JpegError;
use crate::parse::allocation::ParsedMetadataBudget;

fn ones_64() -> [u16; 64] {
    [1; 64]
}

#[test]
fn parses_single_8bit_quant_table() {
    let mut payload = alloc::vec![0u8];
    payload.extend(core::iter::repeat_n(1u8, 64));
    let mut tables = QuantTables::default();
    parse_dqt(&payload, 0, &mut tables, &mut ParsedMetadataBudget::new()).unwrap();
    assert_eq!(tables.entries[0].unwrap(), ones_64());
    assert_eq!(tables.versions.len(), 1);
}

#[test]
fn dqt_redefinition_creates_one_version_and_snapshot_tracks_it() {
    let mut tables = QuantTables::default();
    let mut budget = ParsedMetadataBudget::new();
    for value in [1u8, 2] {
        let mut payload = alloc::vec![0u8];
        payload.extend(core::iter::repeat_n(value, 64));
        parse_dqt(&payload, 0, &mut tables, &mut budget).unwrap();
    }
    let state = ProgressiveTableState::capture(&HuffmanTables::default(), &tables);
    assert_eq!(tables.versions.len(), 2);
    assert_eq!(tables.resolve(&state, 0), Some(&[2; 64]));
}

#[test]
fn parses_single_dc_huffman_table() {
    let mut payload = alloc::vec![0u8, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
    payload.extend_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    let mut tables = HuffmanTables::default();
    parse_dht(&payload, 0, &mut tables, &mut ParsedMetadataBudget::new()).unwrap();
    let table = tables.dc[0].as_ref().unwrap();
    assert_eq!(
        table.values.as_slice(),
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    );
    assert_eq!(tables.versions.len(), 1);
}

#[test]
fn dht_versions_preserve_redefinitions_and_identical_cross_role_tables() {
    let mut tables = HuffmanTables::default();
    let mut budget = ParsedMetadataBudget::new();
    let first_dc = [0u8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xaa];
    parse_dht(&first_dc, 0, &mut tables, &mut budget).unwrap();
    let first_state = ProgressiveTableState::capture(&tables, &QuantTables::default());
    let second_dc = [0u8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xbb];
    parse_dht(&second_dc, 0, &mut tables, &mut budget).unwrap();
    let identical_ac = [0x10u8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xbb];
    parse_dht(&identical_ac, 0, &mut tables, &mut budget).unwrap();
    let active_state = ProgressiveTableState::capture(&tables, &QuantTables::default());

    assert_eq!(tables.versions.len(), 3);
    assert_eq!(tables.versions[0].role, HuffmanTableRole::Dc);
    assert_eq!(tables.versions[1].role, HuffmanTableRole::Dc);
    assert_eq!(tables.versions[2].role, HuffmanTableRole::Ac);
    assert_eq!(
        tables.retained_allocation_bytes().unwrap(),
        tables.versions.capacity() * core::mem::size_of_val(&tables.versions[0])
    );
    assert_eq!(
        tables
            .resolve_dc(&first_state, 0)
            .unwrap()
            .values
            .as_slice(),
        &[0xaa]
    );
    assert_eq!(
        tables
            .resolve_dc(&active_state, 0)
            .unwrap()
            .values
            .as_slice(),
        &[0xbb]
    );
    assert_eq!(
        tables
            .resolve_ac(&active_state, 0)
            .unwrap()
            .values
            .as_slice(),
        &[0xbb]
    );
}

#[test]
fn rejects_truncated_dqt() {
    let payload = alloc::vec![0u8, 1, 2, 3];
    let error = parse_dqt(
        &payload,
        0,
        &mut QuantTables::default(),
        &mut ParsedMetadataBudget::new(),
    )
    .unwrap_err();
    assert!(matches!(error, JpegError::Truncated { .. }));
}

#[test]
fn rejects_invalid_dqt_precision_before_length_interpretation() {
    let mut tables = QuantTables::default();
    let error = parse_dqt(&[0x20], 7, &mut tables, &mut ParsedMetadataBudget::new())
        .expect_err("DQT precision is restricted to 8 or 16 bits");

    assert_eq!(error, JpegError::UnsupportedBitDepth { depth: 2 });
    assert!(tables.versions.is_empty());
}

#[test]
fn rejects_zero_quantizers_without_mutating_table_state() {
    for (precision, zero_offset, payload) in [
        (0_u8, 1_usize, {
            let mut payload = alloc::vec![0_u8];
            payload.extend(core::iter::repeat_n(1_u8, 64));
            payload[1] = 0;
            payload
        }),
        (1_u8, 1_usize, {
            let mut payload = alloc::vec![0x10_u8];
            for _ in 0..64 {
                payload.extend_from_slice(&1_u16.to_be_bytes());
            }
            payload[1] = 0;
            payload[2] = 0;
            payload
        }),
    ] {
        let mut tables = QuantTables::default();
        let error = parse_dqt(&payload, 11, &mut tables, &mut ParsedMetadataBudget::new())
            .expect_err("zero DQT entries are forbidden");
        assert_eq!(
            error,
            JpegError::InvalidQuantizationValue {
                offset: 11 + zero_offset,
                table: 0,
                coefficient: 0,
            },
            "precision {precision}"
        );
        assert!(tables.versions.is_empty());
    }
}

#[test]
fn rejects_huffman_with_more_than_256_values() {
    let mut payload = alloc::vec![0u8];
    payload.extend(core::iter::repeat_n(17u8, 16));
    payload.push(0);
    let error = parse_dht(
        &payload,
        0,
        &mut HuffmanTables::default(),
        &mut ParsedMetadataBudget::new(),
    )
    .unwrap_err();
    assert!(matches!(error, JpegError::InvalidSegmentLength { .. }));
}

#[test]
fn table_state_rejects_an_unvalidated_huffman_class_without_panicking() {
    let mut tables = HuffmanTables::default();
    let error = tables
        .define(
            2,
            0,
            RawHuffmanTable {
                bits: [0; 16],
                values: HuffmanValues::default(),
            },
            &mut ParsedMetadataBudget::new(),
        )
        .expect_err("only DC and AC Huffman classes are valid");

    assert_eq!(
        error,
        JpegError::InternalInvariant {
            reason: "unvalidated DHT class reached table definition",
        }
    );
    assert!(tables.versions.is_empty());

    let error = tables
        .define(
            0,
            4,
            RawHuffmanTable {
                bits: [0; 16],
                values: HuffmanValues::default(),
            },
            &mut ParsedMetadataBudget::new(),
        )
        .expect_err("Huffman slots are limited to four entries");
    assert_eq!(
        error,
        JpegError::InternalInvariant {
            reason: "unvalidated DHT slot reached table definition",
        }
    );
    assert!(tables.versions.is_empty());
}

#[test]
fn table_state_rejects_an_unvalidated_quant_slot_without_panicking() {
    let mut tables = QuantTables::default();
    let error = tables
        .define(4, [1; 64], &mut ParsedMetadataBudget::new())
        .expect_err("quantization slots are limited to four entries");

    assert_eq!(
        error,
        JpegError::InternalInvariant {
            reason: "unvalidated DQT slot reached table definition",
        }
    );
    assert!(tables.versions.is_empty());
}
