// SPDX-License-Identifier: Apache-2.0

//! Public [`Decoder`] entry points.

use crate::entropy::huffman::HuffmanTable;
use crate::entropy::sequential::{decode_scan_baseline, ComponentCtx, DecodeContext};
use crate::error::{JpegError, MarkerKind, Warning};
use crate::info::{Info, OutputFormat, Rect, SofKind};
use crate::output::{validate_buffer, Gray8Writer, Rgb8Writer, Rgba8Writer};
use crate::parse::header::{parse_header, ParsedHeader};
use alloc::vec::Vec;

/// Non-fatal outcome of a successful decode. See spec Section 2.
///
/// `DecodeOutcome` lives on `decoder.rs` rather than `info.rs` because it
/// carries `Warning` values from `error.rs`, and moving it into `info` would
/// create a `info → error` cycle (see `info.rs` header note).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeOutcome {
    /// The rectangle actually written to the output buffer. For `decode_into`
    /// this is always `Rect::full(info.dimensions)`; later milestones add
    /// `decode_region_into` which can return a narrower rect.
    pub decoded: Rect,
    /// Warnings emitted during parse or decode. Empty when the stream is
    /// syntactically clean and every capability was exercised without fallback.
    pub warnings: Vec<Warning>,
}

/// A borrowed view of a JPEG stream ready to decode. Constructed via
/// [`Decoder::new`]. `Decoder<'a>: Send + Sync`.
#[derive(Debug)]
pub struct Decoder<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) header: ParsedHeader,
    pub(crate) info: Info,
    /// Huffman tables pre-built at construction. Indexed by slot id 0..=3.
    pub(crate) dc_tables: [Option<HuffmanTable>; 4],
    pub(crate) ac_tables: [Option<HuffmanTable>; 4],
}

impl<'a> Decoder<'a> {
    /// Parse the headers without decoding pixels. O(header size).
    ///
    /// # Errors
    /// Returns any structural, unsupported-SOF, or sanity-check error
    /// encountered before the Start-of-Scan marker. See [`JpegError`].
    pub fn inspect(input: &'a [u8]) -> Result<Info, JpegError> {
        parse_header(input).map(|h| h.info())
    }

    /// Build a decoder ready for `decode_into`. Parses the full header, pre-
    /// builds every referenced Huffman table, and validates that the stream is
    /// one of the SOFs this release implements.
    ///
    /// # Errors
    /// - Any parse error encountered before SOS (see [`Self::inspect`]).
    /// - [`JpegError::NotImplemented`] for SOFs that parse but are not yet
    ///   decodable (Extended12, Progressive, Lossless — all land in M3).
    /// - [`JpegError::MissingHuffmanTable`] if the scan references a DC/AC
    ///   table slot that was never defined by a DHT segment.
    pub fn new(input: &'a [u8]) -> Result<Self, JpegError> {
        let header = parse_header(input)?;
        let info = header.info();
        match info.sof_kind {
            SofKind::Baseline8 | SofKind::Extended8 => {}
            other => return Err(JpegError::NotImplemented { sof: other }),
        }

        let mut dc_tables: [Option<HuffmanTable>; 4] = Default::default();
        let mut ac_tables: [Option<HuffmanTable>; 4] = Default::default();
        let scan = header.scan.as_ref().ok_or(JpegError::MissingMarker {
            marker: MarkerKind::Sos,
        })?;
        for comp in &scan.components {
            let di = comp.dc_table as usize;
            let ai = comp.ac_table as usize;
            if dc_tables[di].is_none() {
                let raw = header.huffman_tables.dc[di].as_ref().ok_or(
                    JpegError::MissingHuffmanTable {
                        component: comp.id,
                        class: 0,
                        id: comp.dc_table,
                    },
                )?;
                dc_tables[di] = Some(HuffmanTable::from_raw(raw)?);
            }
            if ac_tables[ai].is_none() {
                let raw = header.huffman_tables.ac[ai].as_ref().ok_or(
                    JpegError::MissingHuffmanTable {
                        component: comp.id,
                        class: 1,
                        id: comp.ac_table,
                    },
                )?;
                ac_tables[ai] = Some(HuffmanTable::from_raw(raw)?);
            }
        }

        Ok(Self {
            bytes: input,
            header,
            info,
            dc_tables,
            ac_tables,
        })
    }

    /// The parsed header as a public [`Info`].
    pub fn info(&self) -> &Info {
        &self.info
    }

    /// Decode the full image into the caller's buffer.
    ///
    /// # Errors
    /// - [`JpegError::OutputBufferTooSmall`] or [`JpegError::InvalidStride`]
    ///   if the provided buffer/stride cannot hold the image at `fmt`.
    /// - [`JpegError::NotImplemented`] if `fmt` requests a raw output the
    ///   current release does not emit (e.g. `RawYCbCr8`).
    /// - Any entropy- or structural-decode error from the scan walker.
    pub fn decode_into(
        &self,
        out: &mut [u8],
        stride: usize,
        fmt: OutputFormat,
    ) -> Result<DecodeOutcome, JpegError> {
        let (w, h) = self.info.dimensions;
        let bpp = fmt.bytes_per_pixel();
        validate_buffer(out, stride, w, h, bpp)?;

        let ctx = self.build_decode_context()?;
        let sos_offset = self.header.sos_offset.ok_or(JpegError::MissingMarker {
            marker: MarkerKind::Sos,
        })?;
        let scan_bytes = &self.bytes[sos_offset..];

        let warnings = match fmt {
            OutputFormat::Rgb8 => {
                let mut writer = Rgb8Writer::new(out, stride, w);
                decode_scan_baseline(&ctx, scan_bytes, &mut writer)?
            }
            OutputFormat::Rgba8 { alpha } => {
                let mut writer = Rgba8Writer::new(out, stride, w, alpha);
                decode_scan_baseline(&ctx, scan_bytes, &mut writer)?
            }
            OutputFormat::Gray8 => {
                let mut writer = Gray8Writer::new(out, stride, w);
                decode_scan_baseline(&ctx, scan_bytes, &mut writer)?
            }
            OutputFormat::RawYCbCr8 => {
                return Err(JpegError::NotImplemented {
                    sof: self.info.sof_kind,
                });
            }
        };

        Ok(DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings,
        })
    }

    fn build_decode_context(&self) -> Result<DecodeContext<'_>, JpegError> {
        let header = &self.header;
        let scan = header.scan.as_ref().ok_or(JpegError::MissingMarker {
            marker: MarkerKind::Sos,
        })?;
        let mut components = Vec::with_capacity(scan.components.len());
        for (i, scan_comp) in scan.components.iter().enumerate() {
            let (h, v) = header.sampling.components[i];
            let quant_id = header.quant_table_ids[i] as usize;
            let quant = header.quant_tables.entries[quant_id].as_ref().ok_or(
                JpegError::MissingQuantTable {
                    component: scan_comp.id,
                    table_id: quant_id as u8,
                },
            )?;
            let dc_table = self.dc_tables[scan_comp.dc_table as usize].as_ref().ok_or(
                JpegError::MissingHuffmanTable {
                    component: scan_comp.id,
                    class: 0,
                    id: scan_comp.dc_table,
                },
            )?;
            let ac_table = self.ac_tables[scan_comp.ac_table as usize].as_ref().ok_or(
                JpegError::MissingHuffmanTable {
                    component: scan_comp.id,
                    class: 1,
                    id: scan_comp.ac_table,
                },
            )?;
            components.push(ComponentCtx {
                h,
                v,
                quant,
                dc_table,
                ac_table,
            });
        }
        Ok(DecodeContext {
            components,
            sampling: &header.sampling,
            color_space: self.info.color_space,
            restart_interval: header.restart_interval,
            dimensions: self.info.dimensions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Warning;
    use alloc::vec;
    use alloc::vec::Vec;

    fn minimal_baseline_jpeg() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&[0xFF, 0xD8]);
        v.extend_from_slice(&[0xFF, 0xDB, 0x00, 67, 0x00]);
        v.extend(core::iter::repeat(1u8).take(64));
        v.extend_from_slice(&[
            0xFF,
            0xC0,
            0x00,
            17,
            8,
            0,
            16,
            0,
            16,
            3,
            1,
            (2 << 4) | 2,
            0,
            2,
            (1 << 4) | 1,
            0,
            3,
            (1 << 4) | 1,
            0,
        ]);
        v.extend_from_slice(&[
            0xFF, 0xC4, 0x00, 20, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xAA,
        ]);
        v.extend_from_slice(&[
            0xFF, 0xC4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xBB,
        ]);
        v.extend_from_slice(&[0xFF, 0xDA, 0x00, 12, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0]);
        v.extend_from_slice(&[0x00, 0xFF, 0xD9]);
        v
    }

    #[test]
    fn decoder_new_succeeds_on_baseline_stream() {
        let bytes = minimal_baseline_jpeg();
        let dec = Decoder::new(&bytes).unwrap();
        assert_eq!(dec.info().dimensions, (16, 16));
    }

    #[test]
    fn decoder_new_rejects_progressive_with_not_implemented() {
        let mut bytes = minimal_baseline_jpeg();
        let p = bytes.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        bytes[p + 1] = 0xC2;
        let err = Decoder::new(&bytes).unwrap_err();
        assert!(err.is_not_implemented());
    }

    #[test]
    fn decoder_new_rejects_arithmetic_as_unsupported() {
        let mut bytes = minimal_baseline_jpeg();
        let p = bytes.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        bytes[p + 1] = 0xC9;
        let err = Decoder::new(&bytes).unwrap_err();
        assert!(err.is_unsupported());
    }

    #[test]
    fn decode_outcome_carries_rect_and_warnings() {
        let outcome = DecodeOutcome {
            decoded: Rect {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
            },
            warnings: vec![Warning::MissingEoi],
        };
        assert_eq!(outcome.decoded.w, 16);
        assert_eq!(outcome.warnings.len(), 1);
    }

    #[test]
    fn decode_into_rejects_undersized_buffer() {
        let bytes = minimal_baseline_jpeg();
        let dec = Decoder::new(&bytes).unwrap();
        let mut buf = vec![0u8; 4];
        let err = dec
            .decode_into(&mut buf, 48, OutputFormat::Rgb8)
            .unwrap_err();
        assert!(matches!(err, JpegError::OutputBufferTooSmall { .. }));
    }

    #[test]
    fn decode_into_rejects_invalid_stride() {
        let bytes = minimal_baseline_jpeg();
        let dec = Decoder::new(&bytes).unwrap();
        let mut buf = vec![0u8; 16 * 16 * 3];
        let err = dec
            .decode_into(&mut buf, 10, OutputFormat::Rgb8)
            .unwrap_err();
        assert!(matches!(err, JpegError::InvalidStride { .. }));
    }

    #[test]
    fn decode_into_raw_ycbcr_returns_not_implemented() {
        let bytes = minimal_baseline_jpeg();
        let dec = Decoder::new(&bytes).unwrap();
        let mut buf = vec![0u8; 16 * 16 * 3];
        let err = dec
            .decode_into(&mut buf, 48, OutputFormat::RawYCbCr8)
            .unwrap_err();
        assert!(err.is_not_implemented());
    }
}
