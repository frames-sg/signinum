// SPDX-License-Identifier: Apache-2.0

//! Bit-level reader over an entropy-coded JPEG scan. Presents three core
//! operations: `peek_bits(n)` to examine the next up-to-32 bits, `consume_bits(n)`
//! to advance past them, and `read_bits(n)` which combines both. Internally
//! refills a 64-bit accumulator from the scan bytes, inlines `0xFF 0x00`
//! unstuffing (T.81 §F.1.2.3), and stops refilling when it hits a marker.
//!
//! When the reader hits a marker mid-refill, it sets `marker()` to `Some(code)`
//! and leaves the input cursor pointing at the marker's leading `0xFF`. The
//! MCU loop calls [`BitReader::take_marker`] at segment boundaries to observe
//! and consume RST markers; observing a non-RST marker (e.g. EOI) is how the
//! MCU loop detects end-of-scan without a separate length cursor.

#![allow(dead_code)] // MCU loop in Task 14 wires these up.

use crate::error::{HuffmanFailure, JpegError};

/// Maximum bits the accumulator can hold. Kept at 64 so a single `u64` is
/// enough; refill replenishes up to 56 bits at a time, leaving 8 bits of head
/// room so a peek of up to 8 bits never needs a refill.
const ACC_BITS: u8 = 64;

/// Refill threshold. Every call that consumes bits ensures `bits >= 56`
/// before returning so the next 16-bit peek can always succeed without a
/// mid-decode refill. 56 matches the spec's "refill when `bits < 56`, 4 bytes
/// at a time" guidance (spec §5 hot-path discipline).
const REFILL_THRESHOLD: u8 = 56;

pub(crate) struct BitReader<'a> {
    bytes: &'a [u8],
    /// Cursor into `bytes`. Always either (a) past the last consumed byte, or
    /// (b) pointing at the leading `0xFF` of a marker the refill paused at.
    pos: usize,
    /// MSB-first bit accumulator. The `bits` most-significant bits contain
    /// the next coded bits; lower (`64 - bits`) bits are zero.
    acc: u64,
    /// Number of valid bits in `acc`, 0..=64.
    bits: u8,
    /// Set when refill stopped at a marker. Cleared by [`Self::take_marker`].
    marker: Option<u8>,
}

impl<'a> BitReader<'a> {
    /// Build a reader over an entropy-coded scan. `bytes` must start at the
    /// first entropy byte — i.e. the byte *after* an SOS payload, what
    /// `ParsedHeader.sos_offset` points at.
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            acc: 0,
            bits: 0,
            marker: None,
        }
    }

    /// Ensure at least `n` bits are in the accumulator, refilling as needed.
    /// Returns `HuffmanDecode { TableExhausted }` if the scan is truncated.
    pub(crate) fn ensure_bits(&mut self, n: u8) -> Result<(), JpegError> {
        while self.bits < n {
            if !self.refill_one_byte() {
                if self.bits >= n {
                    return Ok(());
                }
                return Err(JpegError::HuffmanDecode {
                    mcu: 0,
                    reason: HuffmanFailure::TableExhausted,
                });
            }
        }
        Ok(())
    }

    /// Refill one byte of data into the accumulator. Returns `true` if a
    /// byte was added, `false` if the refill paused at a marker or ran out
    /// of input.
    fn refill_one_byte(&mut self) -> bool {
        if self.marker.is_some() || self.pos >= self.bytes.len() {
            return false;
        }
        let b = self.bytes[self.pos];
        if b == 0xFF {
            if self.pos + 1 >= self.bytes.len() {
                return false;
            }
            let next = self.bytes[self.pos + 1];
            if next == 0x00 {
                self.push_byte(0xFF);
                self.pos += 2;
                true
            } else {
                self.marker = Some(next);
                false
            }
        } else {
            self.push_byte(b);
            self.pos += 1;
            true
        }
    }

    fn push_byte(&mut self, b: u8) {
        let shift = ACC_BITS - 8 - self.bits;
        self.acc |= u64::from(b) << shift;
        self.bits += 8;
    }

    /// Return the next `n` bits (MSB-first) without advancing. Caller must
    /// have ensured enough bits via `ensure_bits`. `n <= 16` on the hot path
    /// (Huffman codes up to 16 bits).
    pub(crate) fn peek_bits(&self, n: u8) -> u32 {
        debug_assert!(n <= 32, "peek_bits({n}) exceeds u32");
        debug_assert!(
            n <= self.bits,
            "peek_bits({n}) with only {} buffered",
            self.bits
        );
        if n == 0 {
            0
        } else {
            (self.acc >> (ACC_BITS - n)) as u32
        }
    }

    /// Advance past `n` bits previously examined with `peek_bits`.
    pub(crate) fn consume_bits(&mut self, n: u8) {
        debug_assert!(
            n <= self.bits,
            "consume_bits({n}) with only {} buffered",
            self.bits
        );
        self.acc <<= n;
        self.bits -= n;
    }

    /// Combined peek + consume. Refills as needed.
    pub(crate) fn read_bits(&mut self, n: u8) -> Result<u32, JpegError> {
        self.ensure_bits(n)?;
        let v = self.peek_bits(n);
        self.consume_bits(n);
        self.refill_to_threshold();
        Ok(v)
    }

    /// After consuming bits, top up the accumulator so the next Huffman peek
    /// can always examine 16 bits without further refill.
    fn refill_to_threshold(&mut self) {
        while self.bits < REFILL_THRESHOLD && self.refill_one_byte() {}
    }

    /// Signed-value extension per T.81 §F.2.2.1 ("EXTEND" procedure). `ssss`
    /// is the category — a non-zero value in `1..=15` — and the return is the
    /// signed coefficient value.
    pub(crate) fn receive_extend(&mut self, ssss: u8) -> Result<i32, JpegError> {
        if ssss == 0 {
            return Ok(0);
        }
        let v = self.read_bits(ssss)? as i32;
        let threshold = 1i32 << (ssss - 1);
        Ok(if v < threshold {
            v + ((-1i32) << ssss) + 1
        } else {
            v
        })
    }

    /// Consume and return the marker that paused the last refill. Returns
    /// `None` if no marker has been observed. The MCU loop calls this at
    /// restart-interval boundaries to observe `RST0..=RST7` and resume.
    pub(crate) fn take_marker(&mut self) -> Option<u8> {
        let m = self.marker.take()?;
        self.pos += 2;
        Some(m)
    }

    /// Current cursor into the input. Used only by diagnostics; not part of
    /// hot-path APIs.
    pub(crate) fn position(&self) -> usize {
        self.pos
    }

    /// Reset the bit accumulator at a restart interval boundary. Called by
    /// the MCU loop after observing an RST marker.
    pub(crate) fn reset_at_restart(&mut self) {
        self.acc = 0;
        self.bits = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_bits_in_msb_first_order() {
        let data = [0b1011_0010u8, 0b0110_0100];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_bits(1).unwrap(), 0b1);
        assert_eq!(br.read_bits(3).unwrap(), 0b011);
        assert_eq!(br.read_bits(8).unwrap(), 0b0010_0110);
        assert_eq!(br.read_bits(2).unwrap(), 0b01);
        assert_eq!(br.read_bits(2).unwrap(), 0b00);
    }

    #[test]
    fn unstuffs_ff00_sequence_as_single_ff_data_byte() {
        let data = [0xFFu8, 0x00, 0x55];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_bits(8).unwrap(), 0xFF);
        assert_eq!(br.read_bits(8).unwrap(), 0x55);
    }

    #[test]
    fn stops_at_rst_marker_and_exposes_code() {
        let data = [0x42u8, 0xFF, 0xD3, 0x99];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_bits(8).unwrap(), 0x42);
        let err = br.read_bits(8).unwrap_err();
        assert!(matches!(err, JpegError::HuffmanDecode { .. }));
        assert_eq!(br.take_marker(), Some(0xD3));
    }

    #[test]
    fn stops_at_eoi_marker() {
        let data = [0x11u8, 0x22, 0xFF, 0xD9];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_bits(8).unwrap(), 0x11);
        assert_eq!(br.read_bits(8).unwrap(), 0x22);
        let err = br.read_bits(8).unwrap_err();
        assert!(matches!(err, JpegError::HuffmanDecode { .. }));
        assert_eq!(br.take_marker(), Some(0xD9));
    }

    #[test]
    fn peek_does_not_advance_cursor() {
        let data = [0xAB, 0xCD];
        let mut br = BitReader::new(&data);
        br.ensure_bits(16).unwrap();
        assert_eq!(br.peek_bits(8), 0xAB);
        assert_eq!(br.peek_bits(8), 0xAB);
        br.consume_bits(4);
        assert_eq!(br.peek_bits(8), 0xBC);
    }

    #[test]
    fn receive_extend_matches_t81_f_2_2_1() {
        for (raw, ssss, expected) in [
            (0b010u16, 3u8, -5i32),
            (0b000u16, 3u8, -7i32),
            (0b111u16, 3u8, 7i32),
            (0b100u16, 3u8, 4i32),
            (0b0u16, 1u8, -1i32),
            (0b1u16, 1u8, 1i32),
        ] {
            let data = [(raw << (8 - ssss)) as u8];
            let mut br = BitReader::new(&data);
            let got = br.receive_extend(ssss).unwrap();
            assert_eq!(got, expected, "ssss={ssss} raw={raw:b}");
        }
    }

    #[test]
    fn refills_across_many_bytes_without_losing_bits() {
        let data = [0xAAu8; 12];
        let mut br = BitReader::new(&data);
        for i in 0..96 {
            let bit = br.read_bits(1).unwrap();
            let expected = if i % 2 == 0 { 1 } else { 0 };
            assert_eq!(bit, expected, "bit {i}");
        }
    }

    #[test]
    fn reports_huffman_failure_on_truncated_scan() {
        let data = [0x55u8];
        let mut br = BitReader::new(&data);
        let _ = br.read_bits(8).unwrap();
        let err = br.read_bits(1).unwrap_err();
        assert!(matches!(
            err,
            JpegError::HuffmanDecode {
                reason: HuffmanFailure::TableExhausted,
                ..
            }
        ));
    }
}
