// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compare bulk refill against the original byte-wise state transition.

use super::{BitReader, BitReaderSnapshot, ACC_BITS, REFILL_THRESHOLD};
use crate::error::JpegError;
use alloc::vec::Vec;

fn scalar_refill(reader: &mut BitReader<'_>) {
    while reader.bits < REFILL_THRESHOLD && reader.refill_one_byte() {}
}

fn assert_same_state(actual: &BitReader<'_>, expected: &BitReader<'_>) {
    assert_eq!(actual.snapshot(), expected.snapshot());
    assert_eq!(actual.acc, expected.acc);
    assert_eq!(actual.bits, expected.bits);
    assert_eq!(actual.synthetic_bits, expected.synthetic_bits);
    assert_eq!(actual.marker, expected.marker);
    assert_eq!(actual.marker_end, expected.marker_end);
    assert_eq!(actual.allow_eof_padding, expected.allow_eof_padding);
}

fn compare_refill(bytes: &[u8], pos: usize, bits: u8) {
    let acc = if bits == 0 {
        0
    } else {
        0xa55a_5aa5_a55a_5aa5 & (u64::MAX << (ACC_BITS - bits))
    };
    let snapshot = BitReaderSnapshot { pos, acc, bits };
    let mut actual = BitReader::from_snapshot(bytes, snapshot);
    let mut expected = BitReader::from_snapshot(bytes, snapshot);
    actual.refill_to_threshold();
    scalar_refill(&mut expected);
    assert_same_state(&actual, &expected);
}

#[test]
fn bulk_refill_matches_scalar_for_all_buffer_sizes_and_short_tails() {
    let bytes = [
        0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd,
        0xef,
    ];
    for len in 0..=bytes.len() {
        for pos in 0..=len {
            for bits in 0..=ACC_BITS {
                compare_refill(&bytes[..len], pos, bits);
            }
        }
    }
}

#[test]
fn bulk_refill_matches_scalar_at_stuffing_fill_and_marker_boundaries() {
    let suffixes: &[&[u8]] = &[
        &[0xff],
        &[0xff, 0xff],
        &[0xff, 0x00, 0x12, 0x34, 0x56, 0x78],
        &[0xff, 0xff, 0x00, 0x12, 0x34, 0x56, 0x78],
        &[0xff, 0xd0, 0x12, 0x34, 0x56, 0x78],
        &[0xff, 0xd7, 0x12, 0x34, 0x56, 0x78],
        &[0xff, 0xd9, 0x12, 0x34, 0x56, 0x78],
        &[0xff, 0xff, 0xd9, 0x12, 0x34, 0x56, 0x78],
    ];
    for prefix_len in 0..=8 {
        for suffix in suffixes {
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(prefix_len + suffix.len())
                .expect("reserve marker-boundary comparison input");
            bytes.resize(prefix_len, 0x5a);
            bytes.extend_from_slice(suffix);
            for pos in 0..=prefix_len {
                for bits in 0..=ACC_BITS {
                    compare_refill(&bytes, pos, bits);
                }
            }
        }
    }
}

fn scalar_read_bits(reader: &mut BitReader<'_>, count: u8) -> Result<u32, JpegError> {
    reader.ensure_bits(count)?;
    let value = reader.peek_bits(count);
    reader.consume_bits(count);
    scalar_refill(reader);
    Ok(value)
}

#[test]
fn bulk_refill_preserves_stream_reads_and_resumed_snapshots() {
    let mut bytes = Vec::new();
    for byte in (0u8..=u8::MAX).cycle().take(1_024) {
        bytes.push(byte);
        if byte == 0xff {
            bytes.push(0x00);
        }
    }
    bytes.extend_from_slice(&[0xff, 0xd9]);
    let mut actual = BitReader::new(&bytes);
    let mut expected = BitReader::new(&bytes);
    let mut reached_end = false;
    for (step, count) in [0, 1, 7, 12, 16, 31, 32]
        .into_iter()
        .cycle()
        .take(1_024)
        .enumerate()
    {
        let result = actual.read_bits(count);
        assert_eq!(result, scalar_read_bits(&mut expected, count));
        assert_same_state(&actual, &expected);
        if result.is_err() {
            reached_end = true;
            break;
        }
        if step.is_multiple_of(17) && actual.marker.is_none() {
            actual = BitReader::from_snapshot(&bytes, actual.snapshot());
            expected = BitReader::from_snapshot(&bytes, expected.snapshot());
        }
    }
    assert!(
        reached_end,
        "the comparison must exercise the terminal marker"
    );
    assert_eq!(actual.take_marker(), Some(0xd9));
    assert_eq!(expected.take_marker(), Some(0xd9));
}

#[test]
fn bulk_refill_preserves_padded_huffman_prefetch() {
    let bytes = [0x5a; 512];
    let mut actual = BitReader::new(&bytes);
    let mut expected = BitReader::new(&bytes);
    for count in [12, 16, 1, 7].into_iter().cycle().take(256) {
        let needs_refill = expected.bits < count;
        expected.ensure_bits(count).unwrap();
        if needs_refill {
            scalar_refill(&mut expected);
        }
        actual.ensure_bits_padded(count).unwrap();
        assert_same_state(&actual, &expected);
        actual.consume_bits(count);
        expected.consume_bits(count);
    }
}

#[test]
fn bulk_refill_preserves_synthetic_terminal_bits() {
    for bytes in [&[0x5a][..], &[0x5a, 0xff, 0xd9][..]] {
        let mut actual = BitReader::new_with_eof_padding(bytes, true);
        let mut expected = BitReader::new_with_eof_padding(bytes, true);
        actual.ensure_bits_padded(16).unwrap();
        expected.ensure_bits_padded(16).unwrap();
        assert_eq!(actual.synthetic_bits, 8);
        actual.refill_to_threshold();
        scalar_refill(&mut expected);
        assert_same_state(&actual, &expected);
        assert_eq!(actual.snapshot().bits, 8);
    }
}
