// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use j2k_transcode::accelerator::PrequantizedHtj2k97Component;
use j2k_transcode::accelerator::TranscodeStageError;
use j2k_transcode::accelerator::{
    DctGridToDwt97Job, DctGridToHtj2k97CodeBlockJob, DctToWaveletStageAccelerator,
    Htj2k97CodeBlockOptions, IrreversibleQuantizationSubbandScales,
};
#[cfg(target_os = "macos")]
use j2k_transcode::{dct8x8_blocks_then_dwt97_float, Dwt97TwoDimensional};
use j2k_transcode_metal::weights::{Dwt97WeightRows, SparseDwt97WeightRows};
use j2k_transcode_metal::MetalDctToWaveletStageAccelerator;
#[cfg(target_os = "macos")]
use j2k_transcode_test_support::prequantized_component_from_dwt97;

#[cfg(target_os = "macos")]
fn should_run_metal_runtime() -> bool {
    j2k_test_support::metal_runtime_gate(module_path!())
}

#[test]
fn explicit_metal_reports_unavailable_on_non_macos() {
    #[cfg(target_os = "macos")]
    if !should_run_metal_runtime() {
        return;
    }

    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
    let blocks = vec![[[0.0; 8]; 8]];
    let result = accelerator.dct_grid_to_dwt97(DctGridToDwt97Job {
        blocks: &blocks,
        block_cols: 1,
        block_rows: 1,
        width: 8,
        height: 8,
    });

    #[cfg(not(target_os = "macos"))]
    assert!(matches!(
        result.expect_err("explicit Metal is unavailable off macOS"),
        TranscodeStageError::DeviceUnavailable
    ));

    #[cfg(target_os = "macos")]
    let _ = result;
}

#[test]
fn auto_metal_falls_back_for_tiny_jobs() {
    let mut accelerator = MetalDctToWaveletStageAccelerator::for_auto();
    let blocks = vec![[[0.0; 8]; 8]];
    let output = accelerator
        .dct_grid_to_dwt97(DctGridToDwt97Job {
            blocks: &blocks,
            block_cols: 1,
            block_rows: 1,
            width: 8,
            height: 8,
        })
        .expect("auto accelerator can decline tiny job");

    assert!(output.is_none());
    assert_eq!(accelerator.dwt97_attempts(), 1);
    assert_eq!(accelerator.dwt97_dispatches(), 0);
}

#[cfg(target_os = "macos")]
#[test]
fn auto_metal_uses_cpu_for_97_jobs_by_default() {
    let blocks = structured_blocks(64, 64);
    let mut accelerator = MetalDctToWaveletStageAccelerator::for_auto();

    match accelerator.dct_grid_to_dwt97(DctGridToDwt97Job {
        blocks: &blocks,
        block_cols: 64,
        block_rows: 64,
        width: 512,
        height: 512,
    }) {
        Ok(None) => {}
        Err(TranscodeStageError::DeviceUnavailable) => {
            j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        }
        Ok(Some(_)) => panic!("auto Metal should leave 9/7 jobs on the optimized CPU path"),
        Err(message) => panic!("auto Metal 9/7 accelerator failed: {message}"),
    }

    assert_eq!(accelerator.dwt97_attempts(), 1);
    assert_eq!(accelerator.dwt97_dispatches(), 0);
}

#[cfg(target_os = "macos")]
#[test]
fn explicit_metal_dct97_matches_scalar_for_structured_cases() {
    if !should_run_metal_runtime() {
        return;
    }

    let blocks = structured_blocks(2, 2);
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();

    for (width, height) in [(8, 8), (13, 11), (16, 16)] {
        let actual = match accelerator.dct_grid_to_dwt97(DctGridToDwt97Job {
            blocks: &blocks,
            block_cols: 2,
            block_rows: 2,
            width,
            height,
        }) {
            Ok(Some(output)) => output,
            Ok(None) => panic!("explicit Metal accelerator must not silently fall back"),
            Err(TranscodeStageError::DeviceUnavailable) => {
                j2k_test_support::metal_device_unavailable_is_skip(module_path!());
                return;
            }
            Err(message) => panic!("explicit Metal accelerator failed: {message}"),
        };
        let expected = dct8x8_blocks_then_dwt97_float(&blocks, 2, 2, width, height)
            .expect("scalar 9/7 IDCT path accepts covered grid");

        let max_diff = max_abs_diff(&actual, &expected);
        assert!(
            max_diff <= 2.0e-2,
            "Metal 9/7 DCT transform diverged for {width}x{height}: {max_diff}"
        );
    }

    assert_eq!(accelerator.dwt97_dispatches(), 3);
}

#[cfg(target_os = "macos")]
#[test]
fn explicit_metal_dct97_batch_matches_scalar_for_structured_cases() {
    if !should_run_metal_runtime() {
        return;
    }

    let first = structured_blocks(2, 2);
    let second = structured_blocks_with_offset(2, 2, 97.0);
    let jobs = [
        DctGridToDwt97Job {
            blocks: &first,
            block_cols: 2,
            block_rows: 2,
            width: 13,
            height: 11,
        },
        DctGridToDwt97Job {
            blocks: &second,
            block_cols: 2,
            block_rows: 2,
            width: 13,
            height: 11,
        },
    ];
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();

    let actual = match accelerator.dct_grid_to_dwt97_batch(&jobs) {
        Ok(Some(output)) => output,
        Ok(None) => panic!("explicit Metal batch accelerator must not silently fall back"),
        Err(TranscodeStageError::DeviceUnavailable) => {
            j2k_test_support::metal_device_unavailable_is_skip(module_path!());
            return;
        }
        Err(message) => panic!("explicit Metal batch accelerator failed: {message}"),
    };

    assert_eq!(actual.len(), jobs.len());
    for (actual, job) in actual.iter().zip(jobs.iter()) {
        let expected = dct8x8_blocks_then_dwt97_float(
            job.blocks,
            job.block_cols,
            job.block_rows,
            job.width,
            job.height,
        )
        .expect("scalar 9/7 IDCT path accepts covered grid");

        let max_diff = max_abs_diff(actual, &expected);
        assert!(
            max_diff <= 2.0e-2,
            "Metal 9/7 batch transform diverged: {max_diff}"
        );
    }

    assert_eq!(accelerator.dwt97_batch_dispatches(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn explicit_metal_dct97_batch_reports_idct_row_and_column_stage_timings() {
    if !should_run_metal_runtime() {
        return;
    }

    let batch_blocks = [
        structured_blocks_with_offset(4, 4, 0.0),
        structured_blocks_with_offset(4, 4, 3.0),
    ];
    let jobs: Vec<_> = batch_blocks
        .iter()
        .map(|blocks| DctGridToDwt97Job {
            blocks,
            block_cols: 4,
            block_rows: 4,
            width: 29,
            height: 31,
        })
        .collect();
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();

    match accelerator.dct_grid_to_dwt97_batch(&jobs) {
        Ok(Some(_)) => {}
        Ok(None) => panic!("explicit Metal batch accelerator must not silently fall back"),
        Err(TranscodeStageError::DeviceUnavailable) => {
            j2k_test_support::metal_device_unavailable_is_skip(module_path!());
            return;
        }
        Err(message) => panic!("explicit Metal batch accelerator failed: {message}"),
    }

    let timings = accelerator
        .last_dwt97_batch_stage_timings()
        .expect("Metal 9/7 batch records backend stage timings");
    assert!(timings.pack_upload_us > 0);
    assert!(timings.idct_row_lift_us > 0);
    assert!(timings.column_lift_us > 0);
    assert_eq!(timings.quantize_codeblock_us, 0);
    assert!(timings.readback_us > 0);
}

#[cfg(target_os = "macos")]
#[test]
fn explicit_metal_dct97_codeblock_batch_matches_scalar_quantized_layout() {
    if !should_run_metal_runtime() {
        return;
    }

    let batch_blocks = [
        structured_blocks_with_offset(4, 4, 0.0),
        structured_blocks_with_offset(4, 4, 37.0),
    ];
    let jobs: Vec<_> = batch_blocks
        .iter()
        .map(|blocks| DctGridToHtj2k97CodeBlockJob {
            blocks,
            block_cols: 4,
            block_rows: 4,
            width: 29,
            height: 31,
            x_rsiz: 1,
            y_rsiz: 1,
        })
        .collect();
    let options = Htj2k97CodeBlockOptions {
        bit_depth: 8,
        guard_bits: 2,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        irreversible_quantization_scale: 2.5,
        irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales {
            low_low: 0.9,
            high_low: 1.1,
            low_high: 1.2,
            high_high: 1.5,
        },
    };
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();

    let call_started = std::time::Instant::now();
    let actual = match accelerator.dct_grid_to_htj2k97_codeblock_batch(&jobs, options) {
        Ok(Some(output)) => output,
        Ok(None) => {
            panic!("explicit Metal code-block batch accelerator must not silently fall back")
        }
        Err(TranscodeStageError::DeviceUnavailable) => {
            j2k_test_support::metal_device_unavailable_is_skip(module_path!());
            return;
        }
        Err(message) => panic!("explicit Metal code-block batch accelerator failed: {message}"),
    };
    let call_us = call_started.elapsed().as_micros();

    assert_eq!(actual.len(), jobs.len());
    for (actual, job) in actual.iter().zip(jobs.iter()) {
        let dwt = dct8x8_blocks_then_dwt97_float(
            job.blocks,
            job.block_cols,
            job.block_rows,
            job.width,
            job.height,
        )
        .expect("scalar 9/7 IDCT path accepts covered grid");
        let expected = prequantized_component_from_dwt97(&dwt, options, job.x_rsiz, job.y_rsiz);

        assert_prequantized_component_layout_eq(actual, &expected);
        assert_prequantized_component_coefficients_close(actual, &expected, 1);
    }

    assert_eq!(accelerator.dwt97_batch_dispatches(), 1);
    assert_eq!(accelerator.htj2k97_codeblock_batch_attempts(), 1);
    assert_eq!(accelerator.htj2k97_codeblock_batch_dispatches(), 1);
    let timings = accelerator
        .last_dwt97_batch_stage_timings()
        .expect("Metal code-block batch records backend stage timings");
    assert!(timings.pack_upload_us > 0);
    assert!(timings.idct_row_lift_us > 0);
    // This measures CPU command encoding, which can round down to zero
    // microseconds. Dispatch and coefficient checks above prove execution.
    assert!(timings.column_lift_us <= call_us);
    // The retained staged P12 route materializes four resident subbands per job.
    assert_eq!(timings.resident_dwt_handoff_count, jobs.len() * 4);
    assert!(timings.quantize_codeblock_us > 0);
    assert!(timings.readback_us > 0);
}

#[cfg(target_os = "macos")]
#[test]
fn explicit_metal_dct97_codeblock_batch_accepts_zero_guard_bits_and_matches_scalar() {
    if !should_run_metal_runtime() {
        return;
    }

    // The old Metal-only validator rejected guard_bits == 0; the shared
    // validator accepts it (CUDA and the native encoder always did). This
    // pins the Metal kernel against the CPU oracle for the newly reachable
    // option so the widening cannot silently produce divergent code-blocks.
    let blocks = structured_blocks_with_offset(4, 4, 19.0);
    let jobs = [DctGridToHtj2k97CodeBlockJob {
        blocks: &blocks,
        block_cols: 4,
        block_rows: 4,
        width: 29,
        height: 31,
        x_rsiz: 1,
        y_rsiz: 1,
    }];
    let options = Htj2k97CodeBlockOptions {
        bit_depth: 8,
        guard_bits: 0,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        irreversible_quantization_scale: 2.5,
        irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales {
            low_low: 0.9,
            high_low: 1.1,
            low_high: 1.2,
            high_high: 1.5,
        },
    };
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();

    let actual = match accelerator.dct_grid_to_htj2k97_codeblock_batch(&jobs, options) {
        Ok(Some(output)) => output,
        Ok(None) => {
            panic!("explicit Metal code-block batch accelerator must not silently fall back")
        }
        Err(TranscodeStageError::DeviceUnavailable) => {
            j2k_test_support::metal_device_unavailable_is_skip(module_path!());
            return;
        }
        Err(message) => panic!("explicit Metal rejected guard_bits == 0: {message}"),
    };

    assert_eq!(actual.len(), jobs.len());
    let job = &jobs[0];
    let dwt = dct8x8_blocks_then_dwt97_float(
        job.blocks,
        job.block_cols,
        job.block_rows,
        job.width,
        job.height,
    )
    .expect("scalar 9/7 IDCT path accepts covered grid");
    let expected = prequantized_component_from_dwt97(&dwt, options, job.x_rsiz, job.y_rsiz);

    assert_prequantized_component_layout_eq(&actual[0], &expected);
    assert_prequantized_component_coefficients_close(&actual[0], &expected, 1);
}

#[test]
fn auto_metal_dct97_codeblock_batch_declines_above_staged_axis_for_cpu_fallback() {
    let blocks = structured_blocks_with_offset(129, 1, 0.0);
    let jobs = [DctGridToHtj2k97CodeBlockJob {
        blocks: &blocks,
        block_cols: 129,
        block_rows: 1,
        width: 1025,
        height: 8,
        x_rsiz: 1,
        y_rsiz: 1,
    }];
    let options = Htj2k97CodeBlockOptions {
        bit_depth: 8,
        guard_bits: 2,
        code_block_width_exp: 4,
        code_block_height_exp: 4,
        irreversible_quantization_scale: 1.0,
        irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales::default(),
    };
    let mut accelerator = MetalDctToWaveletStageAccelerator::for_auto();

    let output = accelerator
        .dct_grid_to_htj2k97_codeblock_batch(&jobs, options)
        .expect("Auto oversized-axis decision is not a backend failure");

    assert!(
        output.is_none(),
        "oversized staged axes must use CPU fallback"
    );
    assert_eq!(accelerator.dwt97_batch_attempts(), 1);
    assert_eq!(accelerator.dwt97_batch_dispatches(), 0);
    assert_eq!(accelerator.htj2k97_codeblock_batch_attempts(), 1);
    assert_eq!(accelerator.htj2k97_codeblock_batch_dispatches(), 0);
}

#[test]
fn weight_rows_match_expected_geometry_for_supported_lengths() {
    for sample_len in [8_usize, 13, 16] {
        let rows = Dwt97WeightRows::for_len(sample_len).expect("bounded dense 9/7 weight rows");

        assert_eq!(rows.low.len(), sample_len.div_ceil(2));
        assert_eq!(rows.high.len(), sample_len / 2);
        assert!(rows.low.iter().all(|row| row.len() == sample_len));
        assert!(rows.high.iter().all(|row| row.len() == sample_len));
        assert!(rows
            .low
            .iter()
            .all(|row| row.iter().any(|&value| value.to_bits() != 0)));
        assert!(rows
            .high
            .iter()
            .all(|row| row.iter().any(|&value| value.to_bits() != 0)));
    }
}

#[test]
fn weight_rows_are_deterministic() {
    let first = Dwt97WeightRows::for_len(13).expect("bounded dense 9/7 weight rows");
    let second = Dwt97WeightRows::for_len(13).expect("bounded dense 9/7 weight rows");

    assert_eq!(f32_rows_to_bits(&first.low), f32_rows_to_bits(&second.low));
    assert_eq!(
        f32_rows_to_bits(&first.high),
        f32_rows_to_bits(&second.high)
    );
}

#[test]
fn sparse_weight_rows_reconstruct_dense_rows_for_wsi_lengths() {
    for sample_len in [8_usize, 13, 16, 512, 1024, 2048] {
        let dense = Dwt97WeightRows::for_len(sample_len).expect("bounded dense 9/7 weight rows");
        let sparse =
            SparseDwt97WeightRows::for_len(sample_len).expect("bounded sparse 9/7 weight rows");

        assert!(sparse.max_taps_per_row() <= 16);
        assert_eq!(sparse.low.len(), dense.low.len());
        assert_eq!(sparse.high.len(), dense.high.len());
        assert_eq!(reconstruct_sparse_rows(&sparse.low, sample_len), dense.low);
        assert_eq!(
            reconstruct_sparse_rows(&sparse.high, sample_len),
            dense.high
        );
    }
}

fn f32_rows_to_bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|row| row.iter().map(|value| value.to_bits()).collect())
        .collect()
}

fn reconstruct_sparse_rows(
    rows: &[j2k_transcode_metal::weights::SparseWeightRow],
    sample_len: usize,
) -> Vec<Vec<f32>> {
    rows.iter()
        .map(|row| {
            let mut dense = vec![0.0; sample_len];
            for tap in &row.taps {
                dense[tap.sample_idx] = tap.weight;
            }
            dense
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn max_abs_diff(actual: &Dwt97TwoDimensional<f64>, expected: &Dwt97TwoDimensional<f64>) -> f64 {
    assert_eq!(actual.low_width, expected.low_width);
    assert_eq!(actual.low_height, expected.low_height);
    assert_eq!(actual.high_width, expected.high_width);
    assert_eq!(actual.high_height, expected.high_height);

    actual
        .ll
        .iter()
        .zip(expected.ll.iter())
        .chain(actual.hl.iter().zip(expected.hl.iter()))
        .chain(actual.lh.iter().zip(expected.lh.iter()))
        .chain(actual.hh.iter().zip(expected.hh.iter()))
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

#[cfg(target_os = "macos")]
fn assert_prequantized_component_layout_eq(
    actual: &PrequantizedHtj2k97Component,
    expected: &PrequantizedHtj2k97Component,
) {
    assert_eq!(actual.x_rsiz, expected.x_rsiz);
    assert_eq!(actual.y_rsiz, expected.y_rsiz);
    assert_eq!(actual.resolutions.len(), expected.resolutions.len());
    for (actual_resolution, expected_resolution) in
        actual.resolutions.iter().zip(expected.resolutions.iter())
    {
        assert_eq!(
            actual_resolution.subbands.len(),
            expected_resolution.subbands.len()
        );
        for (actual_subband, expected_subband) in actual_resolution
            .subbands
            .iter()
            .zip(expected_resolution.subbands.iter())
        {
            assert_eq!(actual_subband.sub_band_type, expected_subband.sub_band_type);
            assert_eq!(actual_subband.num_cbs_x, expected_subband.num_cbs_x);
            assert_eq!(actual_subband.num_cbs_y, expected_subband.num_cbs_y);
            assert_eq!(
                actual_subband.total_bitplanes,
                expected_subband.total_bitplanes
            );
            assert_eq!(
                actual_subband.code_blocks.len(),
                expected_subband.code_blocks.len()
            );
            for (actual_block, expected_block) in actual_subband
                .code_blocks
                .iter()
                .zip(expected_subband.code_blocks.iter())
            {
                assert_eq!(actual_block.width, expected_block.width);
                assert_eq!(actual_block.height, expected_block.height);
                assert_eq!(
                    actual_block.coefficients.len(),
                    expected_block.coefficients.len()
                );
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn assert_prequantized_component_coefficients_close(
    actual: &PrequantizedHtj2k97Component,
    expected: &PrequantizedHtj2k97Component,
    max_abs_error: i32,
) {
    for (actual_resolution, expected_resolution) in
        actual.resolutions.iter().zip(expected.resolutions.iter())
    {
        for (actual_subband, expected_subband) in actual_resolution
            .subbands
            .iter()
            .zip(expected_resolution.subbands.iter())
        {
            for (actual_block, expected_block) in actual_subband
                .code_blocks
                .iter()
                .zip(expected_subband.code_blocks.iter())
            {
                for (&actual_coefficient, &expected_coefficient) in actual_block
                    .coefficients
                    .iter()
                    .zip(expected_block.coefficients.iter())
                {
                    assert!(
                        (actual_coefficient - expected_coefficient).abs() <= max_abs_error,
                        "quantized coefficient diverged: actual {actual_coefficient}, expected {expected_coefficient}"
                    );
                }
            }
        }
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "test grid indices are bounded by allocated fixture dimensions"
)]
fn structured_blocks(block_cols: usize, block_rows: usize) -> Vec<[[f64; 8]; 8]> {
    let mut blocks = Vec::with_capacity(block_cols * block_rows);
    for block_y in 0..block_rows {
        for block_x in 0..block_cols {
            let mut block = [[0.0; 8]; 8];
            block[0][0] = 384.0 + (block_x * 19 + block_y * 23) as f64;
            block[0][1] = -17.0 + block_x as f64;
            block[1][0] = 11.0 - block_y as f64;
            block[2][3] = 7.0;
            block[4][4] = -3.0;
            block[7][7] = 2.0;
            blocks.push(block);
        }
    }
    blocks
}

fn structured_blocks_with_offset(
    block_cols: usize,
    block_rows: usize,
    offset: f64,
) -> Vec<[[f64; 8]; 8]> {
    let mut blocks = structured_blocks(block_cols, block_rows);
    for block in &mut blocks {
        block[0][0] += offset;
        block[3][2] -= offset / 7.0;
    }
    blocks
}
