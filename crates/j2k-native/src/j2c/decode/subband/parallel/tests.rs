// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    balanced_task_plan, decode_ht_sub_band_blocks_parallel, install_decoded_blocks,
    prepare_coefficient_slab, prepare_ht_outputs, prepare_ht_task_workspaces,
    prepare_ht_task_workspaces_with_coefficients, try_prepare_ht_task_workspaces,
    validate_pending_blocks, HtParallelParameters,
};
use crate::error::{DecodeError, ValidationError};
use crate::j2c::build::{SubBand, SubBandType};
use crate::j2c::decode::subband::pending::PendingHtBlock;
use crate::j2c::decode::DecodeAllocationBudget;
use crate::j2c::ht_block_decode::CombinedCodeBlockData;
use crate::j2c::rect::IntRect;
use crate::{try_reserve_decode_elements, try_resize_decode_elements};
use alloc::vec::Vec;
use core::mem::size_of;
use rayon::ThreadPoolBuilder;

fn ht_pending(dimensions: &[(u32, u32)]) -> Vec<PendingHtBlock> {
    dimensions
        .iter()
        .map(|&(width, height)| PendingHtBlock {
            combined: CombinedCodeBlockData {
                data: Vec::new(),
                cleanup_length: 0,
                refinement_length: 0,
            },
            output_x: 0,
            output_y: 0,
            width,
            height,
            missing_bit_planes: 0,
            number_of_coding_passes: 0,
        })
        .collect()
}

fn test_sub_band(width: u32, height: u32) -> SubBand {
    SubBand {
        sub_band_type: SubBandType::LowLow,
        rect: IntRect::from_xywh(0, 0, width, height),
        precincts: 0..0,
        coefficients: 0..(width as usize * height as usize),
    }
}

#[test]
fn balanced_task_plans_preserve_exact_task_counts_and_input_order() {
    for (job_count, requested_tasks, expected_lengths) in [
        (16, 12, &[2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1][..]),
        (13, 12, &[2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1][..]),
        (12, 4, &[3, 3, 3, 3][..]),
        (3, 12, &[1, 1, 1][..]),
        (0, 12, &[][..]),
    ] {
        let values: Vec<_> = (0..job_count).collect();
        let plan = balanced_task_plan(job_count, requested_tasks).unwrap();
        let chunks: Vec<_> = plan.chunks(&values).collect();

        assert_eq!(plan.active_workspaces, expected_lengths.len());
        assert_eq!(
            chunks.iter().map(|chunk| chunk.len()).collect::<Vec<_>>(),
            expected_lengths
        );
        assert_eq!(
            chunks.into_iter().flatten().copied().collect::<Vec<_>>(),
            values
        );
    }
}

#[test]
fn ht_parallel_workspace_count_is_bounded_by_workers_and_jobs() {
    let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
    pool.install(|| {
        for (job_count, expected) in [(8, 4), (2, 2)] {
            let pending = ht_pending(&vec![(32, 32); job_count]);
            let mut budget = DecodeAllocationBudget::from_live_bytes(0).unwrap();
            let mut workspaces = Vec::new();
            let mut structural = 0;
            let (plan, _) =
                prepare_ht_task_workspaces(&pending, &mut workspaces, &mut structural, &mut budget)
                    .unwrap();

            assert_eq!(plan.active_workspaces, expected);
            assert_eq!(workspaces.len(), expected);
        }
    });
}

#[test]
fn ht_capacity_retry_uses_actual_capacity_and_an_unpoisoned_budget() {
    let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
    pool.install(|| {
        let pending = ht_pending(&[(32, 32); 8]);
        let mut workspace = crate::HtCodeBlockDecodeWorkspace::default();
        workspace.reserve(32, 32).unwrap();
        let workspace_bytes = workspace.allocated_bytes().unwrap();
        let mut metadata = Vec::<super::HtTaskWorkspace>::new();
        try_reserve_decode_elements(&mut metadata, 1).unwrap();
        let one_task_cap =
            metadata.capacity() * core::mem::size_of::<super::HtTaskWorkspace>() + workspace_bytes;
        let mut budget = DecodeAllocationBudget::from_live_bytes_with_cap(0, one_task_cap).unwrap();
        let mut workspaces = Vec::new();
        let mut structural = 0;

        let (plan, growths) =
            prepare_ht_task_workspaces(&pending, &mut workspaces, &mut structural, &mut budget)
                .expect("one HT task fits after larger task plans exceed the cap");

        assert_eq!(plan.active_workspaces, 1);
        assert_eq!(growths, 1);
        assert_eq!(workspaces.len(), 1);
        assert_eq!(
            super::ht_task_workspace_bytes(&workspaces).unwrap(),
            structural
        );
        assert_eq!(budget.live_bytes(), structural);
        assert_eq!(structural, one_task_cap);
    });
}

#[test]
fn ht_workspace_pressure_shrinks_coefficients_before_reducing_task_count() {
    let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
    pool.install(|| {
        let pending = ht_pending(&[(32, 32); 8]);
        let total_coefficients = 8 * 32 * 32;
        let mut fresh_coefficients = vec![0.0; total_coefficients];
        let fresh_coefficient_bytes = fresh_coefficients.capacity() * size_of::<f32>();
        let mut fresh_workspaces = Vec::new();
        let mut fresh_structural = fresh_coefficient_bytes;
        let mut fresh_stats = super::super::super::ParallelCoefficientStats::default();
        let mut fresh_budget = DecodeAllocationBudget::from_live_bytes(fresh_structural).unwrap();
        let (fresh_plan, _) = prepare_ht_task_workspaces_with_coefficients(
            &pending,
            &mut fresh_workspaces,
            &mut fresh_coefficients,
            total_coefficients,
            &mut fresh_structural,
            &mut fresh_stats,
            &mut fresh_budget,
        )
        .unwrap();
        assert_eq!(fresh_plan.active_workspaces, 4);
        let cap = fresh_budget.live_bytes();
        let workspace_bytes = cap - fresh_coefficient_bytes;

        let oversized_len = total_coefficients + workspace_bytes / (2 * size_of::<f32>());
        let mut coefficients = vec![7.0; oversized_len];
        let old_coefficient_bytes = coefficients.capacity() * size_of::<f32>();
        assert!(old_coefficient_bytes < cap);
        assert!(old_coefficient_bytes + workspace_bytes > cap);
        let mut workspaces = Vec::new();
        let mut structural = old_coefficient_bytes;
        let mut stats = super::super::super::ParallelCoefficientStats::default();
        let mut budget = DecodeAllocationBudget::from_live_bytes_with_cap(structural, cap).unwrap();

        let (plan, _) = prepare_ht_task_workspaces_with_coefficients(
            &pending,
            &mut workspaces,
            &mut coefficients,
            total_coefficients,
            &mut structural,
            &mut stats,
            &mut budget,
        )
        .expect("shrinking optional coefficients preserves the fitting initial task plan");

        assert_eq!(plan.active_workspaces, fresh_plan.active_workspaces);
        assert_eq!(workspaces.len(), fresh_workspaces.len());
        assert_eq!(coefficients.len(), total_coefficients);
        assert!(budget.live_bytes() <= cap);
    });
}

#[test]
fn ht_mixed_shapes_grow_componentwise_then_reuse_warm_reservations() {
    let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| {
        let mut workspaces = Vec::new();
        let mut structural = 0;
        for (dimensions, expect_growth) in [
            ([(64, 32); 4], true),
            ([(32, 64); 4], true),
            ([(64, 32); 4], false),
        ] {
            let mut budget = DecodeAllocationBudget::from_live_bytes(structural).unwrap();
            let (_, growths) = prepare_ht_task_workspaces(
                &ht_pending(&dimensions),
                &mut workspaces,
                &mut structural,
                &mut budget,
            )
            .unwrap();
            assert_eq!(growths != 0, expect_growth);
            let bytes_before_decode = super::ht_task_workspace_bytes(&workspaces).unwrap();
            for slot in &mut workspaces {
                for &(width, height) in &dimensions {
                    slot.workspace.prepare(width, height).unwrap();
                }
            }
            assert_eq!(
                super::ht_task_workspace_bytes(&workspaces).unwrap(),
                bytes_before_decode
            );
        }
        assert!(workspaces
            .iter()
            .all(|slot| { (slot.prepared_width, slot.prepared_height) == (64, 64) }));
    });
}

#[test]
fn ht_failed_growth_rolls_back_then_retry_evicts_and_rebuilds_exactly() {
    let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| {
        let mut workspaces = Vec::new();
        let mut structural = 0;
        let mut initial_budget = DecodeAllocationBudget::from_live_bytes(0).unwrap();
        prepare_ht_task_workspaces(
            &ht_pending(&[(64, 16); 4]),
            &mut workspaces,
            &mut structural,
            &mut initial_budget,
        )
        .unwrap();
        let old_bytes = structural;
        let old_dimensions = workspaces
            .iter()
            .map(|slot| (slot.prepared_width, slot.prepared_height))
            .collect::<Vec<_>>();
        let plan = balanced_task_plan(4, 1).unwrap();
        let mut rollback_budget =
            DecodeAllocationBudget::from_live_bytes_with_cap(old_bytes, old_bytes).unwrap();

        try_prepare_ht_task_workspaces(
            &ht_pending(&[(16, 64); 4]),
            plan,
            &mut workspaces,
            &mut structural,
            &mut rollback_budget,
        )
        .expect_err("replacement peak exceeds the retained-bank cap");
        assert_eq!(structural, old_bytes);
        assert_eq!(rollback_budget.live_bytes(), old_bytes);
        assert_eq!(
            workspaces
                .iter()
                .map(|slot| (slot.prepared_width, slot.prepared_height))
                .collect::<Vec<_>>(),
            old_dimensions
        );

        let mut fresh_workspace = crate::HtCodeBlockDecodeWorkspace::default();
        fresh_workspace.reserve(16, 64).unwrap();
        let mut fresh_metadata = Vec::<super::HtTaskWorkspace>::new();
        try_reserve_decode_elements(&mut fresh_metadata, 2).unwrap();
        let fresh_cap = fresh_metadata.capacity() * core::mem::size_of::<super::HtTaskWorkspace>()
            + 2 * fresh_workspace.allocated_bytes().unwrap();
        assert!(old_bytes <= fresh_cap);
        let mut retry_budget =
            DecodeAllocationBudget::from_live_bytes_with_cap(old_bytes, fresh_cap).unwrap();
        let (plan, growths) = prepare_ht_task_workspaces(
            &ht_pending(&[(16, 64); 4]),
            &mut workspaces,
            &mut structural,
            &mut retry_budget,
        )
        .expect("evicting the retained bank lets the fresh shape fit");

        assert_eq!(plan.active_workspaces, 2);
        assert_eq!(growths, 2);
        assert_eq!(workspaces.len(), 2);
        assert!(workspaces
            .iter()
            .all(|slot| { (slot.prepared_width, slot.prepared_height) == (16, 64) }));
        assert_eq!(
            super::ht_task_workspace_bytes(&workspaces).unwrap(),
            fresh_cap
        );
        assert_eq!(structural, fresh_cap);
        assert_eq!(retry_budget.live_bytes(), fresh_cap);
    });
}

#[test]
fn ht_entropy_error_leaves_workspace_bank_reusable_for_valid_blocks() {
    let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| {
        let parameters = HtParallelParameters {
            strict: true,
            num_bitplanes: 1,
            roi_shift: 0,
            stripe_causal: false,
            dequantization_step: 1.0,
            irreversible_midpoint: false,
        };
        let mut malformed = ht_pending(&[(16, 16); 4]);
        malformed[2].combined.cleanup_length = 1;
        malformed[2].number_of_coding_passes = 1;
        let sub_band = test_sub_band(16, 16);
        let mut workspaces = Vec::new();
        let mut coefficient_slab = Vec::new();
        let mut structural = 0;
        let mut maximum_tasks = 0;
        let mut workspace_growths = 0;
        let mut coefficient_stats = super::super::super::ParallelCoefficientStats::default();
        let mut malformed_budget = DecodeAllocationBudget::from_live_bytes(0).unwrap();

        let malformed_result = decode_ht_sub_band_blocks_parallel(
            &malformed,
            &sub_band,
            parameters,
            &mut workspaces,
            &mut coefficient_slab,
            &mut structural,
            &mut maximum_tasks,
            &mut workspace_growths,
            &mut coefficient_stats,
            &mut malformed_budget,
        );
        assert!(
            malformed_result.is_err(),
            "truncated cleanup payload must fail after workspace activation"
        );
        let retained = super::ht_task_workspace_bytes(&workspaces).unwrap();
        let retained_coefficients = coefficient_slab.capacity() * size_of::<f32>();
        assert_eq!(retained + retained_coefficients, structural);
        let retained_initialized_coefficients = coefficient_slab.len();
        coefficient_slab.fill(7.0);

        let mut valid_budget = DecodeAllocationBudget::from_live_bytes(structural).unwrap();
        workspace_growths = 0;
        let decoded = decode_ht_sub_band_blocks_parallel(
            &ht_pending(&[(8, 8); 4]),
            &sub_band,
            parameters,
            &mut workspaces,
            &mut coefficient_slab,
            &mut structural,
            &mut maximum_tasks,
            &mut workspace_growths,
            &mut coefficient_stats,
            &mut valid_budget,
        )
        .expect("zero-pass blocks decode after the entropy error");

        assert_eq!(workspace_growths, 0);
        assert_eq!(
            super::ht_task_workspace_bytes(&workspaces).unwrap(),
            retained
        );
        assert!(decoded
            .iter()
            .all(|block| block.coefficients.iter().all(|&value| value == 0.0)));
        drop(decoded);
        assert_eq!(coefficient_slab.len(), retained_initialized_coefficients);
        assert!(coefficient_slab[4 * 8 * 8..]
            .iter()
            .all(|&value| value.to_bits() == 7.0_f32.to_bits()));
    });
}

#[test]
fn coefficient_total_rejects_overflow_before_output_allocation() {
    let sub_band = SubBand {
        sub_band_type: SubBandType::LowLow,
        rect: IntRect::from_xywh(0, 0, u32::MAX, u32::MAX),
        precincts: 0..0,
        coefficients: 0..0,
    };
    let error = validate_pending_blocks(
        [(0, 0, u32::MAX, u32::MAX), (0, 0, u32::MAX, u32::MAX)].into_iter(),
        &sub_band,
    )
    .expect_err("aggregate coefficient count must reject overflow");
    assert!(matches!(
        error,
        DecodeError::Validation(ValidationError::ImageTooLarge)
    ));
}

#[test]
fn descriptor_preflight_rejects_a_late_invalid_block_before_touching_retained_slab() {
    let mut pending = ht_pending(&[(2, 2), (2, 2)]);
    pending[1].output_x = 4;
    let sub_band = test_sub_band(5, 3);
    let mut coefficients = vec![7.0; 16];
    let retained_capacity = coefficients.capacity();
    let retained_bytes = retained_capacity * size_of::<f32>();
    let mut structural = retained_bytes;
    let mut stats = super::super::super::ParallelCoefficientStats::default();
    let mut budget = DecodeAllocationBudget::from_live_bytes(retained_bytes).unwrap();

    let result = prepare_ht_outputs(
        &pending,
        &sub_band,
        &mut coefficients,
        &mut structural,
        &mut stats,
        &mut budget,
    );

    assert!(matches!(
        result,
        Err(DecodeError::Decoding(
            crate::DecodingError::CodeBlockDecodeFailure
        ))
    ));
    assert_eq!(coefficients.capacity(), retained_capacity);
    assert!(coefficients
        .iter()
        .all(|&value| value.to_bits() == 7.0_f32.to_bits()));
    assert_eq!(structural, retained_bytes);
    assert_eq!(budget.live_bytes(), retained_bytes);
    assert_eq!(
        stats,
        super::super::super::ParallelCoefficientStats::default()
    );
}

#[test]
fn descriptor_reservation_releases_an_oversized_retained_slab_under_a_tight_cap() {
    let pending = ht_pending(&[(1, 1); 4]);
    let sub_band = test_sub_band(1, 1);
    let mut coefficients = vec![7.0; 16];
    let old_bytes = coefficients.capacity() * size_of::<f32>();
    let mut descriptor_probe: Vec<super::DecodedBlock<'static>> = Vec::new();
    try_reserve_decode_elements(&mut descriptor_probe, pending.len()).unwrap();
    let descriptor_bytes = descriptor_probe.capacity() * size_of::<super::DecodedBlock<'_>>();
    let mut slab_probe = Vec::new();
    try_resize_decode_elements(&mut slab_probe, pending.len(), 0.0).unwrap();
    let fresh_slab_bytes = slab_probe.capacity() * size_of::<f32>();
    let cap = descriptor_bytes + fresh_slab_bytes;
    assert!(old_bytes + descriptor_bytes > cap);
    let mut structural = old_bytes;
    let mut stats = super::super::super::ParallelCoefficientStats::default();
    let mut budget = DecodeAllocationBudget::from_live_bytes_with_cap(old_bytes, cap).unwrap();

    let (mut decoded, _) = prepare_ht_outputs(
        &pending,
        &sub_band,
        &mut coefficients,
        &mut structural,
        &mut stats,
        &mut budget,
    )
    .expect("fresh descriptors and smaller slab fit after retained slab eviction");
    install_decoded_blocks(
        &mut decoded,
        &mut coefficients,
        pending.len(),
        pending.iter().map(|pending| {
            (
                pending.output_x,
                pending.output_y,
                pending.width,
                pending.height,
            )
        }),
    )
    .unwrap();

    assert_eq!(decoded.len(), pending.len());
    assert!(decoded
        .iter()
        .all(|block| block.coefficients.iter().all(|&value| value == 0.0)));
    drop(decoded);
    assert_eq!(structural, fresh_slab_bytes);
    assert_eq!(budget.live_bytes(), descriptor_bytes + structural);
    assert_eq!(stats.allocations, 1);
    assert_eq!(stats.peak_live_bytes, old_bytes.max(fresh_slab_bytes));
}

#[test]
fn coefficient_slab_growth_releases_old_capacity_before_a_tight_cap_allocation() {
    let mut coefficients = vec![1.0; 16];
    let old_bytes = coefficients.capacity() * size_of::<f32>();
    let mut fresh = Vec::new();
    try_resize_decode_elements(&mut fresh, 64, 0.0).unwrap();
    let fresh_bytes = fresh.capacity() * size_of::<f32>();
    let mut structural = old_bytes;
    let mut stats = super::super::super::ParallelCoefficientStats::default();
    let mut budget = DecodeAllocationBudget::from_live_bytes_with_cap(old_bytes, fresh_bytes)
        .expect("retained slab fits the fresh allocation cap");

    prepare_coefficient_slab(
        &mut coefficients,
        64,
        &mut structural,
        &mut stats,
        &mut budget,
    )
    .expect("old optional scratch is released before allocating the larger slab");

    assert_eq!(structural, coefficients.capacity() * size_of::<f32>());
    assert_eq!(budget.live_bytes(), structural);
    assert!(coefficients.iter().all(|&value| value == 0.0));
    assert_eq!(stats.allocations, 1);
    assert_eq!(stats.peak_live_bytes, old_bytes.max(structural));

    let error = prepare_coefficient_slab(
        &mut coefficients,
        128,
        &mut structural,
        &mut stats,
        &mut budget,
    )
    .expect_err("a slab exceeding the cap must fail after releasing optional scratch");
    assert!(matches!(
        error,
        DecodeError::Validation(ValidationError::ImageTooLarge)
    ));
    assert_eq!(coefficients.capacity(), 0);
    assert_eq!((structural, budget.live_bytes()), (0, 0));
    assert_eq!(stats.allocations, 1);

    prepare_coefficient_slab(
        &mut coefficients,
        64,
        &mut structural,
        &mut stats,
        &mut budget,
    )
    .expect("a fitting slab can be prepared after capacity failure");
    assert_eq!(budget.live_bytes(), structural);
    assert!(coefficients.iter().all(|&value| value == 0.0));
}

#[test]
fn classic_workspace_cap_can_release_an_oversized_coefficient_slab() {
    let pending = (0..4)
        .map(|index| super::PendingClassicBlock {
            combined_data: Vec::new(),
            segments: Vec::new(),
            output_x: index * 32,
            output_y: 0,
            width: 32,
            height: 32,
            missing_bit_planes: 0,
            number_of_coding_passes: 0,
        })
        .collect::<Vec<_>>();
    let sub_band = test_sub_band(128, 32);
    let parameters = super::ClassicParallelParameters {
        sub_band_type: crate::J2kSubBandType::LowLow,
        style: crate::J2kCodeBlockStyle {
            selective_arithmetic_coding_bypass: false,
            reset_context_probabilities: false,
            termination_on_each_pass: false,
            vertically_causal_context: false,
            segmentation_symbols: false,
        },
        strict: true,
        total_bitplanes: 1,
        roi_shift: 0,
        dequantization_step: 1.0,
        irreversible_midpoint: false,
    };
    let mut fresh_slab = Vec::new();
    let mut fresh_structural = 0;
    let mut fresh_stats = super::super::super::ParallelCoefficientStats::default();
    let mut fresh_budget = DecodeAllocationBudget::from_live_bytes(0).unwrap();
    let fresh = super::decode_classic_sub_band_blocks_parallel(
        &pending,
        &sub_band,
        parameters,
        &mut fresh_slab,
        &mut fresh_structural,
        &mut fresh_stats,
        &mut fresh_budget,
    )
    .expect("fresh coefficient and entropy buffers fit");
    assert!(fresh
        .iter()
        .all(|block| block.coefficients.iter().all(|&v| v == 0.0)));
    let cap = fresh_budget.live_bytes();
    drop(fresh);

    let mut retained_slab = vec![7.0; fresh_slab.capacity() * 2];
    let mut structural = retained_slab.capacity() * size_of::<f32>();
    assert!(structural < cap);
    let mut budget = DecodeAllocationBudget::from_live_bytes_with_cap(structural, cap).unwrap();
    let mut stats = super::super::super::ParallelCoefficientStats::default();
    let decoded = super::decode_classic_sub_band_blocks_parallel(
        &pending,
        &sub_band,
        parameters,
        &mut retained_slab,
        &mut structural,
        &mut stats,
        &mut budget,
    )
    .expect("oversized optional coefficients must not crowd out fitting entropy workspaces");
    assert!(decoded
        .iter()
        .all(|block| block.coefficients.iter().all(|&v| v == 0.0)));
    assert!(budget.live_bytes() <= cap);
}
