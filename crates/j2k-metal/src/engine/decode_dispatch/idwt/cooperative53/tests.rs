// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn params(
    geometry: (u32, u32, u32, u32, u32, u32),
    batch: u32,
) -> J2kRepeatedIdwtSingleDecompositionParams {
    let (width, height, x0, y0, output_x, output_y) = geometry;
    J2kRepeatedIdwtSingleDecompositionParams {
        x0,
        y0,
        output_x,
        output_y,
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
        ll_instance_stride: 0,
        hl_instance_stride: 0,
        lh_instance_stride: 0,
        hh_instance_stride: 0,
        batch_count: batch,
    }
}

fn compare(
    runtime: &MetalRuntime,
    candidate: &Cooperative53,
    params: &J2kRepeatedIdwtSingleDecompositionParams,
) -> [bool; 2] {
    const PREFIX: usize = 4;
    let count = params.width as usize * params.height as usize * params.batch_count as usize;
    let mut seed = vec![7.0f32; count + 2 * PREFIX];
    // Deterministic signed coefficients exercise negative floor operations.
    let mut state = 0x1234_5678_u32;
    for value in &mut seed[PREFIX..PREFIX + count] {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *value = f32::from(i16::try_from(state % 8192).unwrap() - 4096);
    }
    let buffers = [
        copied_slice_buffer(&runtime.device, &seed).unwrap(),
        copied_slice_buffer(&runtime.device, &seed).unwrap(),
    ];
    let mut selected = [false; 2];
    for (index, axis) in [Axis::Horizontal, Axis::Vertical].into_iter().enumerate() {
        let command = new_command_buffer(&runtime.queue).unwrap();
        let encoder = new_compute_command_encoder(&command).unwrap();
        for (cooperative, buffer) in [false, true].into_iter().zip(&buffers) {
            let used = dispatch_axis(
                runtime,
                candidate,
                &encoder,
                buffer,
                PREFIX * size_of::<f32>(),
                params,
                axis,
                cooperative,
            );
            if cooperative {
                selected[index] = used;
            }
        }
        encoder.endEncoding();
        commit_and_wait_metal(&command).unwrap();
        let reference = crate::engine::checked_buffer_slice::<f32>(
            &buffers[0],
            seed.len(),
            "old 5/3 coefficients",
        )
        .unwrap();
        let actual = crate::engine::checked_buffer_slice::<f32>(
            &buffers[1],
            seed.len(),
            "cooperative 5/3 coefficients",
        )
        .unwrap();
        for (index, (actual, expected)) in actual.iter().zip(&reference).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{axis:?} coefficient {index}"
            );
        }
        for value in actual[..PREFIX].iter().chain(&actual[PREFIX + count..]) {
            assert_eq!(value.to_bits(), 7.0f32.to_bits(), "offset guard changed");
        }
    }
    selected
}

#[test]
fn cooperative53_matches_old_intermediate_coefficients_and_falls_back() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    with_runtime(|runtime| {
        let candidate = Cooperative53::new(&runtime.device).expect("candidate pipelines");
        for (width, height) in [
            (1, 1),
            (1, 9),
            (9, 1),
            (2, 3),
            (3, 2),
            (31, 33),
            (32, 32),
            (127, 129),
            (129, 127),
        ] {
            for x0 in [0, 1] {
                for y0 in [0, 1] {
                    for (output_x, output_y) in [(0, 0), (3, 2), (2, 3)] {
                        for batch in [1, 3] {
                            assert_eq!(
                                compare(
                                    runtime,
                                    &candidate,
                                    &params((width, height, x0, y0, output_x, output_y), batch)
                                ),
                                [true, true],
                                "small lines must execute the candidate"
                            );
                        }
                    }
                }
            }
        }
        // Exercise real device memory boundaries, not an invented fixed cap.
        for axis in [Axis::Horizontal, Axis::Vertical] {
            let available = candidate
                .max_memory
                .checked_sub(candidate.pipeline(axis).staticThreadgroupMemoryLength())
                .unwrap();
            let largest = u32::try_from((available & !15) / size_of::<f32>()).unwrap();
            assert!(largest > 1);
            assert!(candidate.layout(axis, largest).is_some());
            assert!(candidate.layout(axis, largest + 1).is_none());
            for length in [largest - 1, largest, largest + 1] {
                let geometry = match axis {
                    Axis::Horizontal => (length, 3, 1, 0, 2, 3),
                    Axis::Vertical => (3, length, 0, 1, 3, 2),
                };
                let selected = compare(runtime, &candidate, &params(geometry, 3));
                let index = match axis {
                    Axis::Horizontal => 0,
                    Axis::Vertical => 1,
                };
                assert_eq!(selected[index], length <= largest);
            }
        }
        Ok(())
    })
    .expect("cooperative 5/3 comparison");
}

#[test]
#[ignore = "manual old/new GPU stage timing; not an end-to-end acceptance benchmark"]
fn cooperative53_gpu_stage_timing() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    with_runtime(|runtime| {
        let candidate = Cooperative53::new(&runtime.device).expect("candidate pipelines");
        for dimension in [128, 512, 2048] {
            let params = params((dimension, dimension, 1, 1, 2, 3), 3);
            assert_eq!(compare(runtime, &candidate, &params), [true, true]);
            let seed = vec![0.0f32; dimension as usize * dimension as usize * 3];
            let mut measurements = [Vec::new(), Vec::new()];
            for iteration in 0..22 {
                // Alternate order to avoid always favoring the second route.
                for index in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let output = copied_slice_buffer(&runtime.device, &seed).unwrap();
                    let command = new_command_buffer(&runtime.queue).unwrap();
                    let encoder = new_compute_command_encoder(&command).unwrap();
                    for axis in [Axis::Horizontal, Axis::Vertical] {
                        let selected = dispatch_axis(
                            runtime,
                            &candidate,
                            &encoder,
                            &output,
                            0,
                            &params,
                            axis,
                            index == 1,
                        );
                        assert_eq!(selected, index == 1);
                        encoder.memory_barrier_with_resources(&[&output]);
                    }
                    encoder.endEncoding();
                    commit_and_wait_metal(&command).unwrap();
                    let elapsed = command.GPUEndTime() - command.GPUStartTime();
                    assert!(
                        elapsed.is_finite() && elapsed > 0.0,
                        "GPU timestamps unavailable"
                    );
                    if iteration >= 2 {
                        measurements[index].push(elapsed);
                    }
                }
            }
            for (route, samples) in ["old", "cooperative"].into_iter().zip(measurements) {
                eprintln!(
                    "5/3 stage size={dimension} batch=3 route={route} gpu_seconds={samples:?}"
                );
            }
        }
        Ok(())
    })
    .expect("manual 5/3 stage timing");
}
