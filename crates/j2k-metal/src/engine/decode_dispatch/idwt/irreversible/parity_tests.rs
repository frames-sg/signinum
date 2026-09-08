// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::engine::runtime::MetalRuntime;
use crate::metal_types::ComputePipelineState;
use j2k_metal_support::MetalPipelineLoader;

// Original full-grid lifting kernels, kept only as an independent test oracle.
const REFERENCE_STEPS: &str = r"kernel void audit_idwt97_horizontal_reference(
    device float *out [[buffer(0)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(1)]],
    constant J2kIdwt97StepParams &step [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || params.width <= 1u
        || (gid.x & 1u) != step.parity) {
        return;
    }

    out += ulong(gid.z) * params.width * params.height;
    const uint left = periodic_symmetric_extension_left_u32(gid.x, 1u);
    const uint right = periodic_symmetric_extension_right_u32(gid.x, 1u, params.width);
    const uint idx = gid.y * params.width + gid.x;
    out[idx] = fma(out[gid.y * params.width + left] + out[gid.y * params.width + right],
                   step.coefficient,
                   out[idx]);
}

kernel void audit_idwt97_vertical_reference(
    device float *out [[buffer(0)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(1)]],
    constant J2kIdwt97StepParams &step [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || params.height <= 1u
        || (gid.y & 1u) != step.parity) {
        return;
    }

    out += ulong(gid.z) * params.width * params.height;
    const uint above = periodic_symmetric_extension_left_u32(gid.y, 1u);
    const uint below = periodic_symmetric_extension_right_u32(gid.y, 1u, params.height);
    const uint idx = gid.y * params.width + gid.x;
    out[idx] = fma(out[above * params.width + gid.x] + out[below * params.width + gid.x],
                   step.coefficient,
                   out[idx]);
}
";

#[derive(Clone, Copy)]
enum Stage {
    Scale(f32),
    Lift(J2kIdwt97StepParams),
}

fn compare_stage(
    runtime: &MetalRuntime,
    buffers: &[Buffer; 2],
    pipelines: [&ComputePipelineState; 2],
    params: &J2kIdwtSingleDecompositionParams,
    batch: u32,
    stage: Stage,
) {
    const PREFIX: usize = 4;
    let command = new_command_buffer(&runtime.queue).expect("comparison command");
    let encoder = new_compute_command_encoder(&command).expect("comparison encoder");
    for (buffer, pipeline) in buffers.iter().zip(pipelines) {
        encoder.setComputePipelineState(pipeline);
        encoder.set_buffer(0, Some(buffer), (PREFIX * size_of::<f32>()) as u64);
        encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(1, params);
        match stage {
            Stage::Scale(high_pass) => encoder.set_bytes::<f32>(2, &high_pass),
            Stage::Lift(step) => encoder.set_bytes::<J2kIdwt97StepParams>(2, &step),
        }
        // Full original bounds remain a legal upper bound for a compact
        // kernel's guards. Production grid coverage is checked separately.
        dispatch_3d_pipeline(&encoder, pipeline, (params.width, params.height, batch));
    }
    encoder.endEncoding();
    commit_and_wait_metal(&command).expect("completed comparison");
    let len = PREFIX + params.width as usize * params.height as usize * batch as usize + PREFIX;
    let expected = checked_buffer_slice::<f32>(&buffers[0], len, "reference stages").unwrap();
    let actual = checked_buffer_slice::<f32>(&buffers[1], len, "production stages").unwrap();
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(actual.is_finite());
        assert_eq!(actual.to_bits(), expected.to_bits(), "coefficient {index}");
    }
    for value in actual[..PREFIX].iter().chain(&actual[len - PREFIX..]) {
        assert_eq!(value.to_bits(), 7.0_f32.to_bits(), "offset guard changed");
    }
}

fn geometry_params(geometry: (u32, u32, u32, u32, u32, u32)) -> J2kIdwtSingleDecompositionParams {
    let (width, height, x0, y0, output_x, output_y) = geometry;
    J2kIdwtSingleDecompositionParams {
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
    }
}

fn compare_geometry(
    runtime: &MetalRuntime,
    references: &[ComputePipelineState; 2],
    geometry: (u32, u32, u32, u32, u32, u32),
    batch: u32,
    high_pass: f32,
) {
    let (width, height, x0, y0, output_x, output_y) = geometry;
    let params = geometry_params(geometry);
    let count = width as usize * height as usize * batch as usize;
    let mut seed = vec![7.0; count + 8];
    for (index, value) in seed[4..count + 4].iter_mut().enumerate() {
        *value = f32::from(i16::try_from(index % 257).unwrap() - 128) * 0.03125;
    }
    let buffers = [
        copied_slice_buffer(&runtime.device, &seed).unwrap(),
        copied_slice_buffer(&runtime.device, &seed).unwrap(),
    ];
    let kernels = runtime.decode().expect("production kernels");
    for (axis, origin, offset, scale, lift) in [
        (
            0,
            x0,
            output_x,
            &kernels.idwt_irreversible97_horizontal_scale,
            &kernels.idwt_irreversible97_horizontal_step,
        ),
        (
            1,
            y0,
            output_y,
            &kernels.idwt_irreversible97_vertical_scale,
            &kernels.idwt_irreversible97_vertical_step,
        ),
    ] {
        compare_stage(
            runtime,
            &buffers,
            [scale, scale],
            &params,
            batch,
            Stage::Scale(high_pass),
        );
        let even = (origin + offset) & 1;
        for (coefficient, parity) in [
            (dwt::IDWT97_NEG_DELTA_F32, even),
            (dwt::IDWT97_NEG_GAMMA_F32, 1 - even),
            (dwt::IDWT97_NEG_BETA_F32, even),
            (dwt::IDWT97_NEG_ALPHA_F32, 1 - even),
        ] {
            compare_stage(
                runtime,
                &buffers,
                [&references[axis], lift],
                &params,
                batch,
                Stage::Lift(J2kIdwt97StepParams {
                    coefficient,
                    parity,
                    _reserved0: 0,
                    _reserved1: 0,
                }),
            );
        }
    }
}

#[test]
fn irreversible97_lifting_intermediates_match_full_grid_reference_bits() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    with_runtime(|runtime| {
        let source = format!(
            "{}\n{REFERENCE_STEPS}",
            crate::engine::shader_source::decode_shader_source()
        );
        let loader = MetalPipelineLoader::new(&runtime.device, &source).expect("reference library");
        let references = [
            loader
                .pipeline("audit_idwt97_horizontal_reference")
                .unwrap(),
            loader.pipeline("audit_idwt97_vertical_reference").unwrap(),
        ];
        for geometry in [
            (1, 1, 0, 0, 0, 0),
            (1, 1, 1, 1, 0, 0),
            (1, 7, 1, 0, 2, 3),
            (9, 1, 0, 1, 3, 2),
            (2, 3, 0, 1, 0, 0),
            (3, 2, 1, 0, 0, 0),
            (5, 7, 0, 0, 1, 1),
            (18, 31, 1, 1, 2, 3),
            (129, 7, 0, 1, 3, 2),
        ] {
            for batch in [1, 3, 16] {
                for high_pass in [
                    dwt::DWT97_INV_KAPPA_F32,
                    dwt::IDWT97_OPENJPEG_TWO_INV_KAPPA_F32 * 0.5,
                ] {
                    compare_geometry(runtime, &references, geometry, batch, high_pass);
                }
            }
        }
        Ok(())
    })
    .expect("intermediate bitwise comparison");
}

#[test]
fn irreversible97_stage_grids_remove_inactive_parity_positions() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    with_runtime(|runtime| {
        for (width, height, batch) in [(5, 7, 3), (1, 1, 1), (1, 7, 3), (9, 1, 16), (128, 64, 16)] {
            let params = geometry_params((width, height, 1, 0, 2, 3));
            let samples = vec![0.25_f32; width as usize * height as usize * batch as usize];
            let buffer = copied_slice_buffer(&runtime.device, &samples).unwrap();
            let command = new_command_buffer(&runtime.queue).unwrap();
            let encoder = new_compute_command_encoder(&command).unwrap();
            crate::engine::test_counters::reset_idwt97_logical_dispatches_for_test();
            dispatch_irreversible97_stages(
                &encoder,
                runtime.decode().unwrap(),
                &buffer,
                0,
                params,
                dwt::DWT97_INV_KAPPA_F32,
                batch,
            );
            encoder.endEncoding();
            commit_and_wait_metal(&command)?;
            let (positions, dispatches) =
                crate::engine::test_counters::idwt97_logical_dispatches_for_test();
            assert_eq!(
                positions,
                6 * samples.len(),
                "two scales plus eight half-parity lifts"
            );
            assert_eq!(
                dispatches,
                2 + if width == 1 { 2 } else { 4 } + if height == 1 { 2 } else { 4 },
                "zero-count parity dispatches must be skipped"
            );
        }
        Ok(())
    })
    .expect("production parity grids");
}

#[test]
fn compact_parity_axis_enumerates_original_active_coordinates() {
    for length in 0..=257 {
        for odd in [false, true] {
            let parity = u32::from(odd);
            let original = (0..length)
                .filter(|index| index & 1 == parity)
                .collect::<Vec<_>>();
            let compact = (0..parity_axis_len(length, odd))
                .map(|index| index * 2 + parity)
                .collect::<Vec<_>>();
            assert_eq!(compact, original, "length={length}, odd={odd}");
        }
    }
    assert_eq!(parity_axis_len(u32::MAX, false), 1_u32 << 31);
    assert_eq!(parity_axis_len(u32::MAX, true), (1_u32 << 31) - 1);
    for (odd, expected_last) in [(false, u32::MAX - 1), (true, u32::MAX - 2)] {
        let last = (parity_axis_len(u32::MAX, odd) - 1)
            .checked_mul(2)
            .and_then(|index| index.checked_add(u32::from(odd)));
        assert_eq!(last, Some(expected_last));
    }
}

#[path = "interleave_tests.rs"]
mod interleave_tests;
