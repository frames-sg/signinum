// SPDX-License-Identifier: MIT OR Apache-2.0

use super::MetalIdwtDecoder;
use crate::engine::{
    idwt_host_transfer_counters_for_test, reset_idwt_host_transfer_counters_for_test,
};
use j2k_native::{
    HtCodeBlockDecoder, J2kIdwtBand, J2kIdwtNormalization, J2kRect, J2kSingleDecompositionIdwtJob,
    J2kWaveletTransform,
};

struct Fixture {
    rect: J2kRect,
    sizes: [(u32, u32); 4],
    coefficients: [Vec<f32>; 4],
}

impl Fixture {
    fn new(width: u32, height: u32, x0: u32, y0: u32) -> Self {
        let low_width = (x0 + width).div_ceil(2) - x0.div_ceil(2);
        let low_height = (y0 + height).div_ceil(2) - y0.div_ceil(2);
        let sizes = [
            (low_width, low_height),
            (width - low_width, low_height),
            (low_width, height - low_height),
            (width - low_width, height - low_height),
        ];
        Self {
            rect: J2kRect {
                x0,
                y0,
                x1: x0 + width,
                y1: y0 + height,
            },
            sizes,
            coefficients: sizes.map(|(width, height)| {
                (0..width * height)
                    .map(|i| f32::from(i16::try_from(i % 31).unwrap()) - 15.0)
                    .collect()
            }),
        }
    }

    fn job(&self, transform: J2kWaveletTransform) -> J2kSingleDecompositionIdwtJob<'_> {
        let band = |index: usize| J2kIdwtBand {
            rect: J2kRect {
                x0: 0,
                y0: 0,
                x1: self.sizes[index].0,
                y1: self.sizes[index].1,
            },
            coefficients: &self.coefficients[index],
        };
        J2kSingleDecompositionIdwtJob {
            rect: self.rect,
            transform,
            ll: band(0),
            hl: band(1),
            lh: band(2),
            hh: band(3),
        }
    }
}

#[test]
fn host_slice_idwt_preserves_capacity_and_tail_contract() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let mut decoder = MetalIdwtDecoder::default();
    for (width, height, x0, y0) in [
        (1, 1, 0, 0),
        (1, 7, 1, 0),
        (9, 1, 0, 1),
        (3, 2, 1, 1),
        (9, 7, 0, 1),
        (2, 3, 1, 0),
    ] {
        let fixture = Fixture::new(width, height, x0, y0);
        let len = width as usize * height as usize;
        for transform in [
            J2kWaveletTransform::Reversible53,
            J2kWaveletTransform::Irreversible97,
        ] {
            for normalization in [
                J2kIdwtNormalization::Standard,
                J2kIdwtNormalization::OpenJpegCodestream,
            ] {
                let mut expected = vec![0.0_f32; len];
                assert!(decoder
                    .decode_single_decomposition_idwt_with_normalization(
                        fixture.job(transform),
                        normalization,
                        &mut expected
                    )
                    .unwrap());
                let mut oversized = vec![123.25_f32; len + 5];
                assert!(decoder
                    .decode_single_decomposition_idwt_with_normalization(
                        fixture.job(transform),
                        normalization,
                        &mut oversized
                    )
                    .unwrap());
                for (actual, expected) in oversized[..len].iter().zip(&expected) {
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
                assert!(oversized[len..]
                    .iter()
                    .all(|v| v.to_bits() == 123.25_f32.to_bits()));
                let mut undersized = vec![123.25_f32; len - 1];
                reset_idwt_host_transfer_counters_for_test();
                assert!(decoder
                    .decode_single_decomposition_idwt_with_normalization(
                        fixture.job(transform),
                        normalization,
                        &mut undersized
                    )
                    .is_err());
                assert!(undersized
                    .iter()
                    .all(|v| v.to_bits() == 123.25_f32.to_bits()));
                assert_eq!(idwt_host_transfer_counters_for_test(), (0, 0, 0));
            }
        }
    }
}

#[test]
fn host_slice_idwt_output_transfer_accounting() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let fixture = Fixture::new(9, 7, 1, 1);
    let mut decoder = MetalIdwtDecoder::default();
    for transform in [
        J2kWaveletTransform::Reversible53,
        J2kWaveletTransform::Irreversible97,
    ] {
        let mut output = vec![5.0_f32; 9 * 7 + 4];
        reset_idwt_host_transfer_counters_for_test();
        assert!(decoder
            .decode_single_decomposition_idwt(fixture.job(transform), &mut output)
            .unwrap());
        assert_eq!(
            idwt_host_transfer_counters_for_test(),
            (
                output.len() * size_of::<f32>(),
                1,
                output.len() * size_of::<f32>()
            )
        );
    }
}

#[test]
fn empty_host_slice_job_is_not_dispatched() {
    let fixture = Fixture::new(0, 7, 0, 0);
    let mut decoder = MetalIdwtDecoder::default();
    for transform in [
        J2kWaveletTransform::Reversible53,
        J2kWaveletTransform::Irreversible97,
    ] {
        let mut output = [99.0_f32; 3];
        reset_idwt_host_transfer_counters_for_test();
        assert!(!decoder
            .decode_single_decomposition_idwt(fixture.job(transform), &mut output)
            .unwrap());
        assert_eq!(output.map(f32::to_bits), [99.0_f32.to_bits(); 3]);
        assert_eq!(idwt_host_transfer_counters_for_test(), (0, 0, 0));
    }
}

#[test]
fn oversized_host_slice_geometry_rejects_before_transfers() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let mut fixture = Fixture::new(1, 1, 0, 0);
    fixture.rect.x1 = u32::MAX;
    fixture.rect.y1 = u32::MAX;
    let mut decoder = MetalIdwtDecoder::default();
    for transform in [
        J2kWaveletTransform::Reversible53,
        J2kWaveletTransform::Irreversible97,
    ] {
        let mut output = [99.0_f32; 1];
        reset_idwt_host_transfer_counters_for_test();
        assert!(decoder
            .decode_single_decomposition_idwt(fixture.job(transform), &mut output)
            .is_err());
        assert_eq!(output.map(f32::to_bits), [99.0_f32.to_bits()]);
        assert_eq!(idwt_host_transfer_counters_for_test(), (0, 0, 0));
    }
}
