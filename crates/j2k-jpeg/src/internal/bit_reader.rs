// SPDX-License-Identifier: MIT OR Apache-2.0

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

use crate::error::{HuffmanFailure, JpegError};

#[cfg(test)]
mod refill_tests;
mod terminal;

/// Maximum bits the accumulator can hold. Kept at 64 so a single `u64` is
/// enough; refill replenishes up to 56 bits at a time, leaving 8 bits of head
/// room so a peek of up to 8 bits never needs a refill.
const ACC_BITS: u8 = 64;

/// Refill threshold. Every call that consumes bits ensures `bits >= 56`
/// before returning so the next 16-bit peek can always succeed without a
/// mid-decode refill. 56 matches the spec's "refill when `bits < 56`, 4 bytes
/// at a time" guidance (spec §5 hot-path discipline).
const REFILL_THRESHOLD: u8 = 56;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BitReaderSnapshot {
    pub(crate) pos: usize,
    pub(crate) acc: u64,
    pub(crate) bits: u8,
}

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
    /// Synthetic trailing one bits appended only for terminal Huffman lookahead.
    synthetic_bits: u8,
    /// Set when refill stopped at a marker. Cleared by [`Self::take_marker`].
    marker: Option<u8>,
    /// Cursor immediately after the observed marker code, including FF fill.
    marker_end: usize,
    /// Permit libjpeg-compatible synthetic lookahead at physical EOF.
    allow_eof_padding: bool,
}

impl<'a> BitReader<'a> {
    /// Build a reader over an entropy-coded scan. `bytes` must start at the
    /// first entropy byte — i.e. the byte *after* an SOS payload, what
    /// `ParsedHeader.sos_offset` points at.
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self::new_with_eof_padding(bytes, false)
    }

    pub(crate) fn new_with_eof_padding(bytes: &'a [u8], allow_eof_padding: bool) -> Self {
        Self {
            bytes,
            pos: 0,
            acc: 0,
            bits: 0,
            synthetic_bits: 0,
            marker: None,
            marker_end: 0,
            allow_eof_padding,
        }
    }

    /// Ensure at least `n` bits are in the accumulator, refilling as needed.
    /// Returns `HuffmanDecode { TableExhausted }` if the scan is truncated.
    #[expect(
        clippy::inline_always,
        reason = "measured bit-buffer hot path requires cross-helper inlining"
    )]
    #[inline(always)]
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

    /// Fill the accumulator to at least `n` bits, padding with `1` bits when
    /// refill pauses at a marker (end of entropy segment). Matches
    /// libjpeg-turbo's `FILL_BIT_BUFFER_SLOW` policy so a trailing short
    /// Huffman code immediately before EOI still decodes; bits over-read past
    /// the true end of the stream are guaranteed to be `1` — any such
    /// over-read yields invalid Huffman codes rather than spurious short-code
    /// matches.
    ///
    /// Physical EOF normally returns `TableExhausted`. The progressive parser
    /// can explicitly permit the same one-bit lookahead policy when it has
    /// recorded EOF as the scan boundary and retained `Warning::MissingEoi`.
    #[expect(
        clippy::inline_always,
        reason = "measured bit-buffer hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    pub(crate) fn ensure_bits_padded(&mut self, n: u8) -> Result<(), JpegError> {
        let mut refilled = false;
        while self.bits < n {
            if !self.refill_one_byte() {
                if self.marker.is_none()
                    && !(self.allow_eof_padding && self.pos == self.bytes.len())
                {
                    return Err(JpegError::HuffmanDecode {
                        mcu: 0,
                        reason: HuffmanFailure::TableExhausted,
                    });
                }
                while self.bits < n {
                    self.acc |= 1u64 << (ACC_BITS - 1 - self.bits);
                    self.bits += 1;
                    self.synthetic_bits += 1;
                }
                return Ok(());
            }
            refilled = true;
        }
        if refilled {
            self.refill_to_threshold();
        }
        Ok(())
    }

    /// Refill one byte of data into the accumulator. Returns `true` if a
    /// byte was added, `false` if the refill paused at a marker or ran out
    /// of input.
    #[expect(
        clippy::inline_always,
        reason = "measured bit-buffer hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    fn refill_one_byte(&mut self) -> bool {
        if self.marker.is_some() || self.pos >= self.bytes.len() {
            return false;
        }
        let b = self.bytes[self.pos];
        if b == 0xFF {
            let mut code_pos = self.pos + 1;
            while code_pos < self.bytes.len() && self.bytes[code_pos] == 0xff {
                code_pos += 1;
            }
            if code_pos >= self.bytes.len() {
                return false;
            }
            let next = self.bytes[code_pos];
            if next == 0x00 {
                self.push_byte(0xFF);
                self.pos = code_pos + 1;
                true
            } else {
                self.pos = code_pos - 1;
                self.marker = Some(next);
                self.marker_end = code_pos + 1;
                false
            }
        } else {
            self.push_byte(b);
            self.pos += 1;
            true
        }
    }

    #[expect(
        clippy::inline_always,
        reason = "measured bit-buffer hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    fn push_byte(&mut self, b: u8) {
        let shift = ACC_BITS - 8 - self.bits;
        self.acc |= u64::from(b) << shift;
        self.bits += 8;
    }

    /// Return the next `n` bits (MSB-first) without advancing. Caller must
    /// have ensured enough bits via `ensure_bits`. `n <= 16` on the hot path
    /// (Huffman codes up to 16 bits).
    #[expect(
        clippy::inline_always,
        reason = "measured bit-buffer hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the shifted accumulator is masked to the requested at-most-32-bit field"
    )]
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
    #[expect(
        clippy::inline_always,
        reason = "measured bit-buffer hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    pub(crate) fn consume_bits(&mut self, n: u8) {
        debug_assert!(
            n <= self.bits,
            "consume_bits({n}) with only {} buffered",
            self.bits
        );
        let real_bits = self.bits - self.synthetic_bits;
        if n > real_bits {
            self.synthetic_bits -= n - real_bits;
        }
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
        // Below 32 buffered bits, reaching the 56-bit threshold requires at
        // least four more bytes. Load them together only when none needs JPEG
        // unstuffing. The strict bound preserves the scalar reader's exact
        // cursor and reservoir size, including checkpoint snapshots.
        if self.bits < 32 && self.marker.is_none() && self.synthetic_bits == 0 {
            let chunk = self
                .bytes
                .get(self.pos..)
                .and_then(|remaining| remaining.first_chunk::<4>())
                .filter(|chunk| !chunk.contains(&0xff));
            if let Some(chunk) = chunk {
                let word = u64::from(u32::from_be_bytes(*chunk));
                self.acc |= word << (ACC_BITS - 32 - self.bits);
                self.bits += 32;
                self.pos += 4;
            }
        }
        // Keep short tails, stuffing, markers, and terminal padding on the
        // existing byte-wise path. Do not bulk-prefetch in restart probes.
        while self.bits < REFILL_THRESHOLD && self.refill_one_byte() {}
    }

    /// Signed-value extension per T.81 §F.2.2.1 ("EXTEND" procedure). `ssss`
    /// is the category — a non-zero value in `1..=15` — and the return is the
    /// signed coefficient value.
    #[expect(
        clippy::inline_always,
        reason = "measured bit-buffer hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    #[expect(
        clippy::cast_possible_wrap,
        reason = "JPEG receive-extend reads at most 15 bits before converting to signed arithmetic"
    )]
    pub(crate) fn receive_extend(&mut self, ssss: u8) -> Result<i32, JpegError> {
        if ssss == 0 {
            return Ok(0);
        }
        self.ensure_bits(ssss)?;
        let v = self.peek_bits(ssss) as i32;
        self.consume_bits(ssss);
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
        self.pos = self.marker_end;
        self.marker_end = 0;
        Some(m)
    }

    /// Consume the next restart marker and return the following RST index.
    ///
    /// Entropy decoders normally prefetch far enough to observe the marker.
    /// When at most one byte's legal alignment padding remains buffered, one
    /// refill attempt probes the next byte without discarding an `ensure_bits`
    /// error. A missing marker is a scan-position failure rather than a
    /// Huffman-symbol failure, so it retains the caller's MCU coordinates.
    pub(crate) fn consume_restart_marker(
        &mut self,
        expected_rst: u8,
        mcu_at: u32,
        mcu_total: u32,
    ) -> Result<u8, JpegError> {
        if self.marker.is_none() && self.bits <= 7 {
            self.refill_one_byte();
        }
        let marker = self
            .take_marker()
            .ok_or(JpegError::UnexpectedEoi { mcu_at, mcu_total })?;
        let expected = 0xd0 | expected_rst;
        if marker != expected {
            return Err(JpegError::RestartMismatch {
                offset: self.position(),
                expected: expected_rst,
                found: marker,
            });
        }

        self.reset_at_restart();
        Ok((expected_rst + 1) & 0x07)
    }

    /// Current cursor into the input. Used only by diagnostics; not part of
    /// hot-path APIs.
    pub(crate) fn position(&self) -> usize {
        self.pos
    }

    pub(crate) fn snapshot(&self) -> BitReaderSnapshot {
        let real_bits = self.bits.saturating_sub(self.synthetic_bits);
        let acc = if real_bits == 0 {
            0
        } else {
            self.acc & (u64::MAX << (ACC_BITS - real_bits))
        };
        BitReaderSnapshot {
            pos: self.pos,
            acc,
            bits: real_bits,
        }
    }

    /// Reset the bit accumulator at a restart interval boundary. Called by
    /// the MCU loop after observing an RST marker.
    pub(crate) fn reset_at_restart(&mut self) {
        self.acc = 0;
        self.bits = 0;
        self.synthetic_bits = 0;
    }

    pub(crate) fn from_snapshot(bytes: &'a [u8], snapshot: BitReaderSnapshot) -> Self {
        Self {
            bytes,
            pos: snapshot.pos,
            acc: snapshot.acc,
            bits: snapshot.bits,
            synthetic_bits: 0,
            marker: None,
            marker_end: 0,
            allow_eof_padding: false,
        }
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
    fn padded_huffman_lookahead_prefetches_reservoir_without_consuming_bits() {
        let data = [0x12u8, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        let mut br = BitReader::new(&data);

        br.ensure_bits_padded(12).unwrap();

        let snapshot = br.snapshot();
        assert!(
            snapshot.bits >= 56,
            "expected full hot-path reservoir, got {} bits",
            snapshot.bits
        );
        assert_eq!(br.peek_bits(16), 0x1234);
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
            let data =
                [u8::try_from(raw << (8 - ssss))
                    .expect("test magnitude is aligned within one byte")];
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
            let expected = u32::from(i % 2 == 0);
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

    #[test]
    fn snapshot_roundtrips_reader_state() {
        let data = [0xABu8, 0xCD, 0xEF];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_bits(5).unwrap(), 0b10101);
        let snapshot = br.snapshot();
        let expected = br.read_bits(7).unwrap();

        let mut restored = BitReader::from_snapshot(&data, snapshot);
        assert_eq!(restored.read_bits(7).unwrap(), expected);
    }

    #[test]
    fn snapshot_excludes_synthetic_terminal_lookahead() {
        let data = [0xabu8];
        let mut br = BitReader::new_with_eof_padding(&data, true);
        br.ensure_bits_padded(16).unwrap();

        let snapshot = br.snapshot();
        assert_eq!(snapshot.bits, 8);
        assert_eq!(snapshot.acc, 0xab_u64 << 56);

        let mut restored = BitReader::from_snapshot(&data, snapshot);
        assert_eq!(restored.read_bits(8).unwrap(), 0xab);
        assert!(restored.read_bits(1).is_err());
    }

    #[test]
    fn restart_markers_consume_fill_and_advance_the_expected_sequence() {
        let data = [0xff, 0xff, 0xd0, 0xff, 0xd1];
        let mut br = BitReader::new(&data);

        let next = br.consume_restart_marker(0, 1, 3).unwrap();
        assert_eq!(next, 1);
        assert_eq!(br.position(), 3);

        let next = br.consume_restart_marker(next, 2, 3).unwrap();
        assert_eq!(next, 2);
        assert_eq!(br.position(), data.len());
    }

    #[test]
    fn restart_marker_probe_accepts_unprefetched_marker_after_buffered_padding() {
        let data = [0xaf, 0xff, 0xd0];
        let mut br = BitReader::from_snapshot(
            &data,
            BitReaderSnapshot {
                pos: 1,
                acc: 0x0f_u64 << 60,
                bits: 4,
            },
        );

        assert_eq!(br.consume_restart_marker(0, 16, 2_048).unwrap(), 1);
        assert_eq!(br.position(), data.len());
        assert_eq!(br.snapshot().bits, 0);
    }

    #[test]
    fn wrong_restart_marker_preserves_offset_and_buffered_padding() {
        let data = [0xa0, 0xff, 0xd1];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_bits(4).unwrap(), 0x0a);
        assert_eq!(br.snapshot().bits, 4);

        let error = br.consume_restart_marker(0, 4, 9).unwrap_err();
        assert_eq!(
            error,
            JpegError::RestartMismatch {
                offset: data.len(),
                expected: 0,
                found: 0xd1,
            }
        );
        assert_eq!(br.snapshot().bits, 4);
    }

    #[test]
    fn missing_or_truncated_restart_marker_keeps_mcu_coordinates() {
        for data in [&[][..], &[0xff][..]] {
            let mut br = BitReader::new(data);
            assert_eq!(
                br.consume_restart_marker(0, 7, 11).unwrap_err(),
                JpegError::UnexpectedEoi {
                    mcu_at: 7,
                    mcu_total: 11,
                }
            );
        }
    }

    #[test]
    fn stuffed_ff_is_entropy_data_not_a_restart_marker() {
        let data = [0xff, 0x00, 0xff, 0xd0];
        let mut br = BitReader::new(&data);

        assert_eq!(
            br.consume_restart_marker(0, 2, 5).unwrap_err(),
            JpegError::UnexpectedEoi {
                mcu_at: 2,
                mcu_total: 5,
            }
        );
        assert_eq!(br.snapshot().bits, 8);
    }

    #[test]
    fn validated_restart_discards_only_then_buffered_padding() {
        let data = [0xa0, 0xff, 0xd0];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_bits(4).unwrap(), 0x0a);
        assert_eq!(br.snapshot().bits, 4);

        assert_eq!(br.consume_restart_marker(0, 1, 2).unwrap(), 1);
        assert_eq!(br.snapshot().bits, 0);
    }
}
