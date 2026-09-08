// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{copy_decoded_ht_blocks_to_sub_band, decode_ht_sub_band_blocks_parallel};
use super::*;
use crate::j2c::{build::SubBandType, ht_block_decode::CombinedCodeBlockData, rect::IntRect};
use rayon::ThreadPoolBuilder;

fn fixture() -> (Vec<PendingHtBlock>, SubBand, DecompositionStorage<'static>) {
    let mut pending = Vec::new();
    for (y, height) in [(0, 3), (3, 3), (6, 2)] {
        for x in [0, 4] {
            let values = (0..4 * height)
                .map(|i| {
                    if i % 3 == 0 {
                        0
                    } else {
                        i32::try_from(i).unwrap() - 7
                    }
                })
                .collect::<Vec<_>>();
            let encoded =
                crate::j2c::ht_block_encode::encode_code_block(&values, 4, height, 8).unwrap();
            pending.push(PendingHtBlock {
                combined: CombinedCodeBlockData {
                    data: encoded.data,
                    cleanup_length: encoded.ht_cleanup_length,
                    refinement_length: encoded.ht_refinement_length,
                },
                output_x: x,
                output_y: y,
                width: 4,
                height,
                missing_bit_planes: encoded.num_zero_bitplanes,
                number_of_coding_passes: encoded.num_coding_passes,
            });
        }
    }
    let sub_band = SubBand {
        sub_band_type: SubBandType::LowLow,
        rect: IntRect::from_xywh(0, 0, 8, 8),
        precincts: 0..0,
        coefficients: 2..66,
    };
    let mut storage = DecompositionStorage {
        coefficients: vec![19.0; 68],
        ..Default::default()
    };
    storage.coefficients[sub_band.coefficients.clone()].fill(0.0);
    storage.structural_workspace_bytes = storage.coefficients.capacity() * size_of::<f32>();
    (pending, sub_band, storage)
}

fn parameters() -> HtParallelParameters {
    HtParallelParameters {
        strict: true,
        num_bitplanes: 8,
        roi_shift: 0,
        stripe_causal: false,
        dequantization_step: 1.0,
        irreversible_midpoint: false,
    }
}

fn staged(
    pending: &[PendingHtBlock],
    sub_band: &SubBand,
    tile: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'_>,
    budget: &mut DecodeAllocationBudget,
    parameters: HtParallelParameters,
) -> Result<()> {
    let decoded = decode_ht_sub_band_blocks_parallel(
        pending,
        sub_band,
        parameters,
        &mut tile.ht_task_workspaces,
        &mut tile.parallel_coefficients,
        &mut storage.structural_workspace_bytes,
        &mut tile.debug_counters.ht_parallel_tasks,
        &mut tile.debug_counters.ht_task_workspace_growths,
        &mut tile.debug_counters.parallel_coefficients,
        budget,
    )?;
    copy_decoded_ht_blocks_to_sub_band(
        &decoded,
        sub_band,
        storage,
        &mut tile.debug_counters.parallel_coefficients,
    )
}

#[test]
fn direct_stripes_preserve_publication_on_entropy_failure_and_unwind() {
    ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| {
        for (irreversible_midpoint, dequantization_step) in [(false, 1.0), (true, 0.375)] {
        let parameters = HtParallelParameters { dequantization_step, irreversible_midpoint, ..parameters() };
        let (mut pending, sub_band, mut storage) = fixture();
        let mut expected_tile = TileDecodeContext::default();
        let (_, _, mut expected) = fixture();
        let mut expected_budget = DecodeAllocationBudget::for_storage(&expected).unwrap();
        staged(&pending, &sub_band, &mut expected_tile, &mut expected, &mut expected_budget, parameters).unwrap();
        assert!(expected.coefficients[2..66].iter().any(|value| *value != 0.0));
        let original = storage.coefficients.clone();
        pending[3].combined.cleanup_length += 1;
        let mut reference = TileDecodeContext::default();
        let (_, _, mut reference_storage) = fixture();
        let mut reference_budget = DecodeAllocationBudget::for_storage(&reference_storage).unwrap();
        let expected_error = staged(&pending, &sub_band, &mut reference, &mut reference_storage, &mut reference_budget, parameters).unwrap_err();
        let mut tile = TileDecodeContext::default();
        let mut budget = DecodeAllocationBudget::for_storage(&storage).unwrap();
        let error = try_decode_ht_stripes(&pending, &sub_band, parameters, &mut tile, &mut storage, &mut budget).unwrap_err();
        assert_eq!(error, expected_error);
        assert_eq!(storage.coefficients, original, "failed middle block must publish nothing");
        assert_eq!(tile.debug_counters.parallel_coefficients.direct_bytes, 0);
        assert_eq!(budget.live_bytes(), storage.structural_workspace_bytes);
        pending[3].combined.cleanup_length -= 1;
        assert!(try_decode_ht_stripes(&pending, &sub_band, parameters, &mut tile, &mut storage, &mut budget).unwrap());
        assert!(storage.coefficients.iter().map(|value| value.to_bits())
            .eq(expected.coefficients.iter().map(|value| value.to_bits())),
            "odd stripes and sparse blocks must match staged coefficient bits, including sentinels");
        assert_eq!(tile.debug_counters.parallel_coefficients.scatter_bytes, 0);
        assert_eq!(tile.debug_counters.parallel_coefficients.direct_bytes, 64 * size_of::<f32>());
        storage.coefficients[2..66].fill(0.0);
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let band = UnpublishedBand { output: &mut storage.coefficients[2..66], committed: false };
            band.output[..16].fill(42.0);
            panic!("unwind before publication");
        }));
        assert!(unwind.is_err());
        assert_eq!(storage.coefficients, original, "unwind must restore the unpublished band only");
        }
    });
}

#[test]
fn irregular_nonzero_and_padded_bands_remain_staged() {
    for case in 0..3 {
        let (mut pending, mut sub_band, mut storage) = fixture();
        match case {
            0 => {
                pending.remove(2);
            }
            1 => storage.coefficients[12] = -0.0,
            _ => sub_band.coefficients.end += 1,
        }
        let before = storage
            .coefficients
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let mut tile = TileDecodeContext::default();
        let mut budget = DecodeAllocationBudget::for_storage(&storage).unwrap();
        assert!(!try_decode_ht_stripes(
            &pending,
            &sub_band,
            parameters(),
            &mut tile,
            &mut storage,
            &mut budget
        )
        .unwrap());
        assert_eq!(
            storage
                .coefficients
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            before
        );
        assert_eq!(tile.debug_counters.parallel_coefficients.direct_bytes, 0);
    }
}

#[test]
fn direct_admission_preserves_the_staged_tight_cap_with_a_retained_slab() {
    ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
        .install(|| {
            let (pending, sub_band, mut expected) = fixture();
            let mut cold = TileDecodeContext::default();
            let mut baseline_budget = DecodeAllocationBudget::for_storage(&expected).unwrap();
            staged(
                &pending,
                &sub_band,
                &mut cold,
                &mut expected,
                &mut baseline_budget,
                parameters(),
            )
            .unwrap();
            let cap = baseline_budget.live_bytes();
            let (_, _, mut storage) = fixture();
            let base = storage.structural_workspace_bytes;
            let mut tile = TileDecodeContext {
                parallel_coefficients: vec![7.0; (cap - base) / size_of::<f32>()],
                ..Default::default()
            };
            storage.structural_workspace_bytes +=
                tile.parallel_coefficients.capacity() * size_of::<f32>();
            let initial = storage.structural_workspace_bytes;
            let mut budget =
                DecodeAllocationBudget::from_live_bytes_with_cap(initial, cap).unwrap();
            assert!(
                !try_decode_ht_stripes(
                    &pending,
                    &sub_band,
                    parameters(),
                    &mut tile,
                    &mut storage,
                    &mut budget
                )
                .unwrap(),
                "optional stripe metadata cannot fit until the staged path releases its slab"
            );
            assert_eq!(
                budget.live_bytes(),
                initial,
                "failed admission must discard transient charges and preserve the cap"
            );
            staged(
                &pending,
                &sub_band,
                &mut tile,
                &mut storage,
                &mut budget,
                parameters(),
            )
            .unwrap();
            assert_eq!(storage.coefficients, expected.coefficients);
            assert!(budget.live_bytes() <= cap);
            assert!(tile.debug_counters.parallel_coefficients.scatter_bytes > 0);
        });
}
