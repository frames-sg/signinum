// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{dispatch_irreversible97_interleave_horizontal_scale, dispatch_irreversible97_stages};
use crate::engine::decode_dispatch::idwt::batched_irreversible::dispatch_irreversible97_repeated_interleave_horizontal_scale;
use crate::engine::decode_dispatch::idwt::{
    IdwtSubBandBuffers, RepeatedIdwtDispatch, SingleIdwtDispatch,
};
use crate::engine::decode_dispatch::{
    J2kIdwtSingleDecompositionParams, J2kRepeatedIdwtSingleDecompositionParams,
};
use crate::engine::{
    checked_buffer_slice, commit_and_wait_metal, completed_command_buffer_gpu_duration,
    copied_slice_buffer, new_command_buffer, new_compute_command_encoder, with_runtime, Buffer,
    MetalRuntime,
};
use crate::metal_types::prelude::*;
use j2k_codec_math::dwt;
use std::time::{Duration, Instant};

const CASES: [(u32, u32, u32); 4] = [
    (128, 128, 16),
    (640, 480, 16),
    (1024, 1024, 16),
    (512, 512, 1),
];
const SAMPLE_COUNT: usize = 50;
const WARM_DURATION: Duration = Duration::from_secs(3);
const TARGET_BATCH_DURATION: Duration = Duration::from_millis(200);

struct InterleaveTimingFixture {
    ll: Buffer,
    hl: Buffer,
    lh: Buffer,
    hh: Buffer,
    single_params: J2kIdwtSingleDecompositionParams,
    repeated_params: J2kRepeatedIdwtSingleDecompositionParams,
    output_seed: Vec<f32>,
}

fn seeded_coefficients(width: u32, height: u32, batch_count: u32) -> Vec<f32> {
    let len = width as usize * height as usize * batch_count as usize;
    (0..len)
        .map(|index| {
            let primary =
                f32::from(u16::try_from(index % 257).expect("seed remainder fits u16")) - 128.0;
            let secondary = f32::from(
                u8::try_from((index / 257) % 17).expect("secondary seed remainder fits u8"),
            );
            primary.mul_add(0.001, secondary * 0.0001)
        })
        .collect()
}

fn stage_params(width: u32, height: u32) -> J2kIdwtSingleDecompositionParams {
    J2kIdwtSingleDecompositionParams {
        x0: 1,
        y0: 1,
        output_x: 0,
        output_y: 0,
        width,
        height,
        ll_x: 0,
        ll_y: 0,
        ll_width: 0,
        ll_height: 0,
        hl_x: 0,
        hl_y: 0,
        hl_width: 0,
        hl_height: 0,
        lh_x: 0,
        lh_y: 0,
        lh_width: 0,
        lh_height: 0,
        hh_x: 0,
        hh_y: 0,
        hh_width: 0,
        hh_height: 0,
    }
}

fn interleave_timing_fixture(
    runtime: &MetalRuntime,
    width: u32,
    height: u32,
    batch_count: u32,
) -> InterleaveTimingFixture {
    let low_width = width.div_ceil(2);
    let low_height = height.div_ceil(2);
    let high_width = width / 2;
    let high_height = height / 2;
    let approximation_stride = low_width * low_height;
    let horizontal_detail_stride = high_width * low_height;
    let vertical_detail_stride = low_width * high_height;
    let diagonal_detail_stride = high_width * high_height;
    let band = |stride: u32, seed: f32| {
        let stride = stride as usize;
        let values = (0..stride * batch_count as usize)
            .map(|index| {
                let position =
                    f32::from(u16::try_from(index % 257).expect("band seed position fits u16"));
                let instance = f32::from(
                    u8::try_from((index / stride) % 31).expect("band seed instance fits u8"),
                );
                position.mul_add(0.001, instance.mul_add(0.0001, seed))
            })
            .collect::<Vec<_>>();
        copied_slice_buffer(&runtime.device, &values).expect("IDWT97 interleave timing band")
    };
    let single_params = J2kIdwtSingleDecompositionParams {
        x0: 0,
        y0: 0,
        output_x: 0,
        output_y: 0,
        width,
        height,
        ll_x: 0,
        ll_y: 0,
        ll_width: low_width,
        ll_height: low_height,
        hl_x: 0,
        hl_y: 0,
        hl_width: high_width,
        hl_height: low_height,
        lh_x: 0,
        lh_y: 0,
        lh_width: low_width,
        lh_height: high_height,
        hh_x: 0,
        hh_y: 0,
        hh_width: high_width,
        hh_height: high_height,
    };
    InterleaveTimingFixture {
        ll: band(approximation_stride, 0.125),
        hl: band(horizontal_detail_stride, -0.25),
        lh: band(vertical_detail_stride, 0.375),
        hh: band(diagonal_detail_stride, -0.5),
        repeated_params: J2kRepeatedIdwtSingleDecompositionParams {
            x0: single_params.x0,
            y0: single_params.y0,
            output_x: single_params.output_x,
            output_y: single_params.output_y,
            width,
            height,
            ll_x: single_params.ll_x,
            ll_y: single_params.ll_y,
            ll_width: single_params.ll_width,
            ll_height: single_params.ll_height,
            hl_x: single_params.hl_x,
            hl_y: single_params.hl_y,
            hl_width: single_params.hl_width,
            hl_height: single_params.hl_height,
            lh_x: single_params.lh_x,
            lh_y: single_params.lh_y,
            lh_width: single_params.lh_width,
            lh_height: single_params.lh_height,
            hh_x: single_params.hh_x,
            hh_y: single_params.hh_y,
            hh_width: single_params.hh_width,
            hh_height: single_params.hh_height,
            ll_instance_stride: approximation_stride,
            hl_instance_stride: horizontal_detail_stride,
            lh_instance_stride: vertical_detail_stride,
            hh_instance_stride: diagonal_detail_stride,
            batch_count,
        },
        single_params,
        output_seed: vec![0.0; width as usize * height as usize * batch_count as usize],
    }
}

fn run_interleave_horizontal_scale_command(
    runtime: &MetalRuntime,
    fixture: &InterleaveTimingFixture,
    batch_count: u32,
    validate_output: bool,
) -> (Duration, usize, usize) {
    let decoded = copied_slice_buffer(&runtime.device, &fixture.output_seed)
        .expect("IDWT97 interleave timing output");
    let sub_bands = IdwtSubBandBuffers {
        ll: &fixture.ll,
        ll_offset: 0,
        hl: &fixture.hl,
        hl_offset: 0,
        lh: &fixture.lh,
        lh_offset: 0,
        hh: &fixture.hh,
        hh_offset: 0,
    };
    let command_buffer = new_command_buffer(&runtime.queue).expect("IDWT97 interleave command");
    let encoder = new_compute_command_encoder(&command_buffer).expect("IDWT97 interleave encoder");
    let high_pass = dwt::IDWT97_OPENJPEG_TWO_INV_KAPPA_F32 * 0.5;
    crate::engine::test_counters::reset_idwt97_logical_dispatches_for_test();
    if batch_count == 1 {
        dispatch_irreversible97_interleave_horizontal_scale(
            &encoder,
            SingleIdwtDispatch {
                kernels: runtime.decode().expect("IDWT97 decode kernels"),
                sub_bands,
                params: fixture.single_params,
                decoded: &decoded,
                decoded_offset: 0,
            },
            high_pass,
        );
    } else {
        dispatch_irreversible97_repeated_interleave_horizontal_scale(
            &encoder,
            RepeatedIdwtDispatch {
                kernels: runtime.decode().expect("IDWT97 decode kernels"),
                sub_bands,
                params: fixture.repeated_params,
                decoded: &decoded,
            },
            high_pass,
        );
    }
    encoder.endEncoding();
    commit_and_wait_metal(&command_buffer).expect("complete IDWT97 interleave and scale");
    let gpu_duration = completed_command_buffer_gpu_duration(&command_buffer)
        .expect("completed IDWT97 interleave command must expose GPU timestamps");
    if validate_output {
        let output = checked_buffer_slice::<f32>(
            &decoded,
            fixture.output_seed.len(),
            "IDWT97 interleave timing probe",
        )
        .expect("read IDWT97 interleave timing probe");
        assert!(output.iter().all(|value| value.is_finite()));
    }
    let (logical_positions, dispatches) =
        crate::engine::test_counters::idwt97_logical_dispatches_for_test();
    (gpu_duration, logical_positions, dispatches)
}

fn run_interleave_horizontal_scale_batch(
    runtime: &MetalRuntime,
    fixture: &InterleaveTimingFixture,
    batch_count: u32,
    iterations: usize,
) -> (Duration, Duration, usize, usize) {
    let started = Instant::now();
    let mut gpu_duration = Duration::ZERO;
    let mut logical_positions: usize = 0;
    let mut dispatches: usize = 0;
    for _ in 0..iterations {
        let (command_gpu, command_positions, command_dispatches) =
            run_interleave_horizontal_scale_command(runtime, fixture, batch_count, false);
        gpu_duration = gpu_duration.saturating_add(command_gpu);
        logical_positions = logical_positions.saturating_add(command_positions);
        dispatches = dispatches.saturating_add(command_dispatches);
    }
    (
        gpu_duration,
        started.elapsed(),
        logical_positions,
        dispatches,
    )
}

fn run_stage_command(
    runtime: &MetalRuntime,
    width: u32,
    height: u32,
    batch_count: u32,
    coefficients: &[f32],
    validate_output: bool,
) -> Duration {
    let decoded = copied_slice_buffer(&runtime.device, coefficients).expect("seed IDWT97 buffer");
    let command_buffer = new_command_buffer(&runtime.queue).expect("IDWT97 command buffer");
    let encoder = new_compute_command_encoder(&command_buffer).expect("IDWT97 compute encoder");
    dispatch_irreversible97_stages(
        &encoder,
        runtime.decode().expect("IDWT97 decode kernels"),
        &decoded,
        0,
        stage_params(width, height),
        dwt::DWT97_INV_KAPPA_F32,
        batch_count,
    );
    encoder.endEncoding();
    commit_and_wait_metal(&command_buffer).expect("complete IDWT97 stages");
    let gpu_duration = completed_command_buffer_gpu_duration(&command_buffer)
        .expect("completed IDWT97 command buffer must expose GPU timestamps");
    if validate_output {
        let output =
            checked_buffer_slice::<f32>(&decoded, coefficients.len(), "IDWT97 timing probe")
                .expect("read IDWT97 timing probe");
        assert!(output.iter().all(|value| value.is_finite()));
    }
    gpu_duration
}

fn run_stage_batch(
    runtime: &MetalRuntime,
    width: u32,
    height: u32,
    batch_count: u32,
    coefficients: &[f32],
    iterations: usize,
) -> (Duration, Duration, usize, usize) {
    crate::engine::test_counters::reset_idwt97_logical_dispatches_for_test();
    let started = Instant::now();
    let mut gpu_duration = Duration::ZERO;
    for _ in 0..iterations {
        gpu_duration = gpu_duration.saturating_add(run_stage_command(
            runtime,
            width,
            height,
            batch_count,
            coefficients,
            false,
        ));
    }
    let wall_duration = started.elapsed();
    let (logical_positions, dispatches) =
        crate::engine::test_counters::idwt97_logical_dispatches_for_test();
    (gpu_duration, wall_duration, logical_positions, dispatches)
}

fn calibrated_iterations(wall_duration: Duration) -> usize {
    let wall_ns = wall_duration.as_nanos().max(1);
    let target_ns = TARGET_BATCH_DURATION.as_nanos();
    let iterations = target_ns.div_ceil(wall_ns);
    usize::try_from(iterations)
        .unwrap_or(10_000)
        .clamp(1, 10_000)
}

#[test]
#[ignore = "GPU timing harness; run explicitly with --ignored --nocapture"]
fn metal_irreversible97_interleave_horizontal_scale_gpu_timing() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    with_runtime(|runtime| {
        for (width, height, batch_count) in CASES {
            let fixture = interleave_timing_fixture(runtime, width, height, batch_count);
            let (_, probe_logical_positions, probe_dispatches) =
                run_interleave_horizontal_scale_command(runtime, &fixture, batch_count, true);

            let warm_started = Instant::now();
            while warm_started.elapsed() < WARM_DURATION {
                run_interleave_horizontal_scale_command(runtime, &fixture, batch_count, false);
            }

            let (_, calibration_wall, _, _) =
                run_interleave_horizontal_scale_batch(runtime, &fixture, batch_count, 1);
            let iterations = calibrated_iterations(calibration_wall);
            let route = if batch_count == 1 { "single" } else { "batch" };
            for sample in 0..SAMPLE_COUNT {
                let (gpu, wall, logical_positions, dispatches) =
                    run_interleave_horizontal_scale_batch(
                        runtime,
                        &fixture,
                        batch_count,
                        iterations,
                    );
                assert_eq!(dispatches, probe_dispatches * iterations);
                assert_eq!(
                    logical_positions,
                    probe_logical_positions * iterations
                );
                println!(
                    "idwt97_gpu_interleave_horizontal_scale fixture=synthetic route={route} width={width} height={height} batch={batch_count} x0=0 y0=0 high_pass={} sample={sample} iterations={iterations} gpu_ns={} wall_ns={} logical_requested_positions={logical_positions} dispatches={dispatches}",
                    dwt::IDWT97_OPENJPEG_TWO_INV_KAPPA_F32 * 0.5,
                    gpu.as_nanos(),
                    wall.as_nanos(),
                );
            }
        }
        Ok(())
    })
    .expect("run irreversible97 interleave and horizontal scale GPU timing harness");
}

#[test]
#[ignore = "GPU timing harness; run explicitly with --ignored --nocapture"]
fn metal_irreversible97_stage_gpu_timing() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    with_runtime(|runtime| {
        for (width, height, batch_count) in CASES {
            let coefficients = seeded_coefficients(width, height, batch_count);
            crate::engine::test_counters::reset_idwt97_logical_dispatches_for_test();
            run_stage_command(
                runtime,
                width,
                height,
                batch_count,
                &coefficients,
                true,
            );
            let (probe_logical_positions, probe_dispatches) =
                crate::engine::test_counters::idwt97_logical_dispatches_for_test();
            assert_eq!(probe_dispatches, 10);

            let warm_started = Instant::now();
            while warm_started.elapsed() < WARM_DURATION {
                run_stage_command(
                    runtime,
                    width,
                    height,
                    batch_count,
                    &coefficients,
                    false,
                );
            }

            let (_, calibration_wall, _, _) = run_stage_batch(
                runtime,
                width,
                height,
                batch_count,
                &coefficients,
                1,
            );
            let iterations = calibrated_iterations(calibration_wall);
            for sample in 0..SAMPLE_COUNT {
                let (gpu, wall, logical_positions, dispatches) = run_stage_batch(
                    runtime,
                    width,
                    height,
                    batch_count,
                    &coefficients,
                    iterations,
                );
                assert_eq!(dispatches, probe_dispatches * iterations);
                assert_eq!(
                    logical_positions,
                    probe_logical_positions * iterations
                );
                println!(
                    "idwt97_gpu_stages width={width} height={height} batch={batch_count} x0=1 y0=1 high_pass={} sample={sample} iterations={iterations} gpu_ns={} wall_ns={} logical_requested_positions={logical_positions} dispatches={dispatches}",
                    dwt::DWT97_INV_KAPPA_F32,
                    gpu.as_nanos(),
                    wall.as_nanos(),
                );
            }
        }
        Ok(())
    })
    .expect("run irreversible97 GPU stage timing harness");
}
