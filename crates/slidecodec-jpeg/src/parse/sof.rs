// SPDX-License-Identifier: Apache-2.0

//! Parse a Start-of-Frame segment payload into header-level facts.
//!
//! SOF marker codes we accept:
//! - `FFC0` — SOF0 baseline sequential, 8-bit
//! - `FFC1` — SOF1 extended sequential, 8 or 12 bit
//! - `FFC2` — SOF2 progressive, 8 or 12 bit
//! - `FFC3` — SOF3 lossless (Annex H predictor)
//!
//! Anything else maps to `JpegError::UnsupportedSof` with a typed reason.

use crate::error::{JpegError, UnsupportedReason};
use crate::info::{SamplingFactors, SofKind};
use alloc::vec::Vec;

#[derive(Debug)]
pub(crate) struct ParsedSof {
    pub(crate) sof_kind: SofKind,
    pub(crate) bit_depth: u8,
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) sampling: SamplingFactors,
    pub(crate) component_ids: Vec<u8>,
    /// Quantization-table id for each component (SOF3 ignores this; kept
    /// for baseline/extended/progressive).
    pub(crate) quant_table_ids: Vec<u8>,
}

pub(crate) fn parse_sof(
    marker_code: u8,
    payload: &[u8],
    payload_offset: usize,
) -> Result<ParsedSof, JpegError> {
    // Structural floor: precision(1) + height(2) + width(2) + Nf(1) + at least one component triple(3)
    if payload.len() < 8 {
        return Err(JpegError::Truncated {
            offset: payload_offset + payload.len(),
            expected: 8 - payload.len(),
        });
    }

    let precision = payload[0];
    let height = u16::from_be_bytes([payload[1], payload[2]]);
    let width = u16::from_be_bytes([payload[3], payload[4]]);
    let nf = payload[5];

    let expected_len = 6 + (nf as usize) * 3;
    if payload.len() < expected_len {
        return Err(JpegError::Truncated {
            offset: payload_offset + payload.len(),
            expected: expected_len - payload.len(),
        });
    }

    let sof_kind = match (marker_code, precision) {
        (0xC0, 8) => SofKind::Baseline8,
        (0xC1, 8) => SofKind::Extended8,
        (0xC1, 12) => SofKind::Extended12,
        (0xC2, 8) => SofKind::Progressive8,
        (0xC2, 12) => SofKind::Progressive12,
        (0xC3, 2..=16) => SofKind::Lossless,
        // Differential / hierarchical
        (0xC5 | 0xC6 | 0xC7, _) => {
            return Err(JpegError::UnsupportedSof {
                marker: marker_code,
                reason: UnsupportedReason::Hierarchical,
            });
        }
        // Arithmetic
        (0xC9 | 0xCA | 0xCB, _) => {
            return Err(JpegError::UnsupportedSof {
                marker: marker_code,
                reason: UnsupportedReason::ArithmeticCoding,
            });
        }
        // Differential + arithmetic
        (0xCD | 0xCE | 0xCF, _) => {
            return Err(JpegError::UnsupportedSof {
                marker: marker_code,
                reason: UnsupportedReason::ArithmeticAndHierarchical,
            });
        }
        (_, bad_precision) => {
            return Err(JpegError::UnsupportedBitDepth {
                depth: bad_precision,
            });
        }
    };

    if width == 0 || height == 0 {
        return Err(JpegError::ZeroDimension { width, height });
    }
    if width > 65_500 || height > 65_500 {
        return Err(JpegError::DimensionOverflow {
            width: u32::from(width),
            height: u32::from(height),
        });
    }

    if !matches!(nf, 1 | 3 | 4) {
        return Err(JpegError::UnsupportedComponentCount { count: nf });
    }

    let mut components = Vec::with_capacity(nf as usize);
    let mut component_ids = Vec::with_capacity(nf as usize);
    let mut quant_table_ids = Vec::with_capacity(nf as usize);

    for i in 0..nf as usize {
        let base = 6 + i * 3;
        let component_id = payload[base];
        let sampling_byte = payload[base + 1];
        let tq = payload[base + 2];
        let h = sampling_byte >> 4;
        let v = sampling_byte & 0x0F;
        if !(1..=4).contains(&h) || !(1..=4).contains(&v) {
            return Err(JpegError::InvalidSampling {
                component: i as u8,
                h,
                v,
            });
        }
        components.push((h, v));
        component_ids.push(component_id);
        quant_table_ids.push(tq);
    }

    Ok(ParsedSof {
        sof_kind,
        bit_depth: precision,
        width,
        height,
        sampling: SamplingFactors::from_components(&components),
        component_ids,
        quant_table_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn sof0_420_payload() -> Vec<u8> {
        // precision=8, height=16, width=16, Nf=3,
        // [Y: id=1 H=2 V=2 Tq=0][Cb: id=2 H=1 V=1 Tq=1][Cr: id=3 H=1 V=1 Tq=1]
        vec![
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
            1,
            3,
            (1 << 4) | 1,
            1,
        ]
    }

    #[test]
    fn parses_sof0_baseline_420() {
        let p = parse_sof(0xC0, &sof0_420_payload(), 0).unwrap();
        assert_eq!(p.sof_kind, SofKind::Baseline8);
        assert_eq!(p.width, 16);
        assert_eq!(p.height, 16);
        assert_eq!(p.bit_depth, 8);
        assert_eq!(p.sampling.components(), &[(2, 2), (1, 1), (1, 1)]);
        assert_eq!(p.sampling.max_h, 2);
        assert_eq!(p.sampling.max_v, 2);
        assert_eq!(p.component_ids, vec![1, 2, 3]);
        assert_eq!(p.quant_table_ids, vec![0, 1, 1]);
    }

    #[test]
    fn parses_sof1_extended_12_bit() {
        let mut payload = sof0_420_payload();
        payload[0] = 12;
        let p = parse_sof(0xC1, &payload, 0).unwrap();
        assert_eq!(p.sof_kind, SofKind::Extended12);
        assert_eq!(p.bit_depth, 12);
    }

    #[test]
    fn parses_sof3_lossless() {
        let mut payload = sof0_420_payload();
        payload[0] = 16; // 16-bit lossless
        let p = parse_sof(0xC3, &payload, 0).unwrap();
        assert_eq!(p.sof_kind, SofKind::Lossless);
        assert_eq!(p.bit_depth, 16);
    }

    #[test]
    fn rejects_sof9_arithmetic() {
        let err = parse_sof(0xC9, &sof0_420_payload(), 0).unwrap_err();
        assert!(matches!(
            err,
            JpegError::UnsupportedSof {
                reason: UnsupportedReason::ArithmeticCoding,
                ..
            }
        ));
    }

    #[test]
    fn rejects_sof5_hierarchical() {
        let err = parse_sof(0xC5, &sof0_420_payload(), 0).unwrap_err();
        assert!(matches!(
            err,
            JpegError::UnsupportedSof {
                reason: UnsupportedReason::Hierarchical,
                ..
            }
        ));
    }

    #[test]
    fn rejects_zero_dimension() {
        let mut payload = sof0_420_payload();
        payload[1] = 0;
        payload[2] = 0;
        let err = parse_sof(0xC0, &payload, 0).unwrap_err();
        assert!(matches!(err, JpegError::ZeroDimension { .. }));
    }

    #[test]
    fn rejects_two_component_image() {
        let payload = vec![8, 0, 16, 0, 16, 2, 1, 0x11, 0, 2, 0x11, 0];
        let err = parse_sof(0xC0, &payload, 0).unwrap_err();
        assert!(matches!(
            err,
            JpegError::UnsupportedComponentCount { count: 2 }
        ));
    }

    #[test]
    fn rejects_invalid_sampling_factor() {
        let mut payload = sof0_420_payload();
        payload[7] = 0x05; // H=0, V=5 — H is zero which is invalid
        let err = parse_sof(0xC0, &payload, 0).unwrap_err();
        assert!(matches!(err, JpegError::InvalidSampling { .. }));
    }

    #[test]
    fn rejects_truncated_payload() {
        let err = parse_sof(0xC0, &[8, 0, 16, 0, 16], 0).unwrap_err();
        assert!(matches!(err, JpegError::Truncated { .. }));
    }
}
