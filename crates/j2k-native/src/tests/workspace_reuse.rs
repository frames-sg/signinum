// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{fixture_gray, fixture_ht_gray};
use crate::{
    encode, encode_typed_component_planes_53, DecodeSettings, DecoderContext, EncodeOptions,
    EncodeTypedComponentPlane, Image,
};

#[test]
fn decoder_workspace_reuses_component_owners_across_distinct_input_lifetimes() {
    let mut workspace = crate::DecoderWorkspace::default();
    let first_pixels = {
        let bytes = fixture_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("first image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("first workspace decode");
        let pixels = decoded.data.clone();
        drop(decoded);
        workspace = context.into_workspace();
        pixels
    };

    let second_pixels = {
        let bytes = fixture_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("second image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("second workspace decode");
        let pixels = decoded.data.clone();
        drop(decoded);
        workspace = context.into_workspace();
        pixels
    };

    assert_eq!(second_pixels, first_pixels);
    assert_eq!(workspace.stats().decode_calls(), 2);
    assert_eq!(workspace.stats().component_owner_reuses(), 1);
    assert!(workspace.stats().retained_component_bytes() > 0);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScratchCapacitySnapshot {
    classic_coefficients: (*const crate::j2c::bitplane::Coefficient, usize),
    classic_payload: (*const u8, usize),
    ht_coefficients: (*const u32, usize),
    idwt_output: (*const f32, usize),
    idwt_scratch: (*const f32, usize),
    idwt_output_i64: (*const i64, usize),
    idwt_scratch_i64: (*const i64, usize),
}

fn scratch_capacity_snapshot(context: &DecoderContext<'_>) -> ScratchCapacitySnapshot {
    let tile = &context.tile_decode_context;
    ScratchCapacitySnapshot {
        classic_coefficients: (
            tile.bit_plane_decode_context.coefficient_ptr_for_test(),
            tile.bit_plane_decode_context
                .coefficient_capacity_for_test(),
        ),
        classic_payload: tile
            .bit_plane_decode_buffers
            .combined_layers_owner_for_test(),
        ht_coefficients: tile.ht_block_decode_context.coefficient_owner_for_test(),
        idwt_output: (
            tile.idwt_output.coefficients.as_ptr(),
            tile.idwt_output.coefficients.capacity(),
        ),
        idwt_scratch: (
            tile.idwt_scratch_buffer.as_ptr(),
            tile.idwt_scratch_buffer.capacity(),
        ),
        idwt_output_i64: (
            tile.idwt_output.coefficients_i64.as_ptr(),
            tile.idwt_output.coefficients_i64.capacity(),
        ),
        idwt_scratch_i64: (
            tile.idwt_scratch_buffer_i64.as_ptr(),
            tile.idwt_scratch_buffer_i64.capacity(),
        ),
    }
}

#[test]
fn decoder_workspace_reuses_classic_tier1_and_idwt_owners_across_input_lifetimes() {
    let mut workspace = crate::DecoderWorkspace::default();
    let first = {
        let bytes = fixture_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("first image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("first classic decode");
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };
    let second = {
        let bytes = fixture_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("second image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("second classic decode");
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };

    assert!(first.classic_coefficients.1 > 0);
    assert!(first.classic_payload.1 > 0);
    assert!(first.idwt_output.1 > 0);
    assert_eq!(second, first);
    assert_eq!(workspace.stats().tier1_owner_reuses(), 1);
    assert_eq!(workspace.stats().idwt_owner_reuses(), 1);
    assert!(workspace.stats().retained_tier1_bytes() > 0);
    assert!(workspace.stats().retained_idwt_bytes() > 0);
}

#[test]
fn decoder_workspace_reuses_ht_tier1_owner_across_input_lifetimes() {
    let mut workspace = crate::DecoderWorkspace::default();
    let first = {
        let bytes = fixture_ht_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("first HT image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("first HT decode");
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };
    let second = {
        let bytes = fixture_ht_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("second HT image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("second HT decode");
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };

    assert!(first.ht_coefficients.1 > 0);
    assert!(first.idwt_output.1 > 0);
    assert_eq!(second, first);
    assert_eq!(workspace.stats().tier1_owner_reuses(), 1);
    assert_eq!(workspace.stats().idwt_owner_reuses(), 1);
}

#[test]
fn decoder_workspace_reuses_scratch_across_alternating_shapes_and_precision() {
    let large_pixels = (0..16_u16 * 16)
        .map(|sample| (sample & 0xff) as u8)
        .collect::<Vec<_>>();
    let large_bytes = encode(
        &large_pixels,
        16,
        16,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        },
    )
    .expect("encode large classic fixture");
    let exact_samples = [0_u32, 1, (1_u32 << 28) + 7, (1_u32 << 29) - 1];
    let exact_pixels = exact_samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let exact_planes = [EncodeTypedComponentPlane {
        data: &exact_pixels,
        x_rsiz: 1,
        y_rsiz: 1,
        bit_depth: 29,
        signed: false,
    }];
    let exact_bytes = encode_typed_component_planes_53(
        &exact_planes,
        2,
        2,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            use_mct: false,
            ..EncodeOptions::default()
        },
    )
    .expect("encode exact fixture");

    let mut workspace = crate::DecoderWorkspace::default();
    let large_scratch = {
        let image = Image::new(&large_bytes, &DecodeSettings::default()).expect("large image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_native_with_context(&mut context)
            .expect("large decode");
        assert_eq!(decoded.data, large_pixels);
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };
    let exact_scratch = {
        let image = Image::new(&exact_bytes, &DecodeSettings::default()).expect("exact image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_native_with_context(&mut context)
            .expect("exact decode");
        assert_eq!(decoded.data, exact_pixels);
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };
    let final_scratch = {
        let bytes = fixture_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("small image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("small decode");
        assert_eq!(decoded.data, (0_u8..16).collect::<Vec<_>>());
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };

    assert_eq!(
        final_scratch.classic_coefficients,
        large_scratch.classic_coefficients
    );
    assert_eq!(final_scratch.classic_payload, large_scratch.classic_payload);
    assert_eq!(final_scratch.idwt_output, large_scratch.idwt_output);
    assert_eq!(final_scratch.idwt_scratch, large_scratch.idwt_scratch);
    assert!(exact_scratch.idwt_output_i64.1 > 0);
    assert_eq!(final_scratch.idwt_output_i64, exact_scratch.idwt_output_i64);
    assert_eq!(workspace.stats().decode_calls(), 3);
    assert_eq!(workspace.stats().tier1_owner_reuses(), 2);
    assert_eq!(workspace.stats().idwt_owner_reuses(), 2);
    assert_eq!(workspace.stats().scratch_capacity_retries(), 0);
}

#[cfg(feature = "parallel")]
fn parallel_gray_fixture(width: u32, height: u32, ht: bool, reversible: bool) -> Vec<u8> {
    let pixels = super::gradient_pixels(width, height, 1);
    encode(
        &pixels,
        width,
        height,
        1,
        8,
        false,
        &EncodeOptions {
            reversible,
            guard_bits: if reversible { 1 } else { 2 },
            num_decomposition_levels: 3,
            use_ht_block_coding: ht,
            ..EncodeOptions::default()
        },
    )
    .expect("encode parallel workspace fixture")
}

#[cfg(feature = "parallel")]
fn parallel_workspace_measurements(context: &DecoderContext<'_>) -> (usize, usize, usize, usize) {
    let tile = &context.tile_decode_context;
    let counters = tile.debug_counters;
    (
        tile.ht_parallel_workspace_count(),
        tile.ht_parallel_workspace_bytes().expect("HT bank bytes"),
        counters.ht_parallel_tasks,
        counters.ht_task_workspace_growths,
    )
}

#[cfg(feature = "parallel")]
#[test]
fn ht_task_workspaces_match_serial_and_reuse_capacity() {
    for reversible in [false, true] {
        let bytes = parallel_gray_fixture(257, 263, true, reversible);
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("odd image");
        let mut serial = DecoderContext::default();
        serial.set_cpu_decode_parallelism(crate::CpuDecodeParallelism::Serial);
        let expected = image
            .decode_with_context(&mut serial)
            .expect("serial baseline");
        if reversible {
            assert_eq!(expected.data, super::gradient_pixels(257, 263, 1));
        }
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("test pool");
            pool.install(|| {
                let mut context = DecoderContext::default();
                let first = image
                    .decode_with_context(&mut context)
                    .expect("first decode");
                assert_eq!(first.data, expected.data, "reversible={reversible}");
                assert_eq!((first.width, first.height), (257, 263));
                let (slots, bytes, tasks, growths) = parallel_workspace_measurements(&context);
                assert!(
                    tasks > 0 && tasks <= threads.saturating_mul(2),
                    "tasks={tasks}, pool={threads}"
                );
                assert!(slots > 0 && slots <= tasks);
                assert!(
                    slots
                        < context
                            .tile_decode_context
                            .debug_counters
                            .decoded_code_blocks
                );
                assert!(bytes > 0 && growths > 0);

                let second = image
                    .decode_with_context(&mut context)
                    .expect("warm decode");
                assert_eq!(second.data, expected.data);
                let (warm_slots, warm_bytes, warm_tasks, warm_growths) =
                    parallel_workspace_measurements(&context);
                assert_eq!((warm_slots, warm_bytes, warm_tasks), (slots, bytes, tasks));
                assert_eq!(warm_growths, 0, "unchanged workspaces must not grow");
            });
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn ht_task_workspaces_survive_input_lifetimes_and_region_errors() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("test pool");
    pool.install(|| {
        let (workspace, retained_bank_bytes) = {
            let bytes = parallel_gray_fixture(257, 263, true, true);
            let image = Image::new(&bytes, &DecodeSettings::default()).expect("first image");
            let mut context = DecoderContext::default();
            let decoded = image
                .decode_with_context(&mut context)
                .expect("first decode");
            assert_eq!(decoded.data, super::gradient_pixels(257, 263, 1));
            let (_, bank_bytes, tasks, _) = parallel_workspace_measurements(&context);
            assert!(tasks > 0);
            (context.into_workspace(), bank_bytes)
        };
        // The original codestream and all borrowed parsing state are gone.
        let bytes = parallel_gray_fixture(241, 127, true, true);
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("second image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("second decode");
        assert_eq!(decoded.data, super::gradient_pixels(241, 127, 1));
        let (_, bank_bytes, _, growths) = parallel_workspace_measurements(&context);
        assert!(bank_bytes <= retained_bank_bytes);
        assert_eq!(growths, 0);

        let roi = (7, 9, 53, 47);
        let region = image
            .decode_region_with_context(roi, &mut context)
            .expect("ROI");
        assert_eq!(
            region.data,
            super::crop_interleaved(&decoded.data, 241, 1, roi)
        );
        assert!(image
            .decode_region_with_context((242, 0, 1, 1), &mut context)
            .is_err());
        let recovered = image
            .decode_with_context(&mut context)
            .expect("reuse after error");
        assert_eq!(recovered.data, decoded.data);

        let reduced = Image::new(
            &bytes,
            &DecodeSettings {
                target_resolution: Some((121, 64)),
                ..DecodeSettings::default()
            },
        )
        .expect("reduced image");
        let mut serial = DecoderContext::default();
        serial.set_cpu_decode_parallelism(crate::CpuDecodeParallelism::Serial);
        let expected = reduced
            .decode_with_context(&mut serial)
            .expect("serial reduced");
        let actual = reduced
            .decode_with_context(&mut context)
            .expect("reused reduced");
        assert_eq!(actual.data, expected.data);
        assert_eq!(
            (actual.width, actual.height),
            (expected.width, expected.height)
        );
        assert_eq!(parallel_workspace_measurements(&context).3, 0);
    });
}

#[cfg(feature = "parallel")]
#[test]
fn ht_task_workspaces_remain_reusable_across_classic_decode() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("test pool");
    for reversible in [true, false] {
        let ht_bytes = parallel_gray_fixture(257, 263, true, reversible);
        let classic_bytes = parallel_gray_fixture(241, 127, false, reversible);
        let ht = Image::new(&ht_bytes, &DecodeSettings::default()).expect("HT image");
        let classic =
            Image::new(&classic_bytes, &DecodeSettings::default()).expect("classic image");
        pool.install(|| {
            let mut context = DecoderContext::default();
            let first = ht
                .decode_with_context(&mut context)
                .expect("first HT decode");
            let (slots, bank_bytes, _, growths) = parallel_workspace_measurements(&context);
            assert!(slots > 0 && bank_bytes > 0 && growths > 0);

            let mut serial = DecoderContext::default();
            serial.set_cpu_decode_parallelism(crate::CpuDecodeParallelism::Serial);
            let expected = classic
                .decode_with_context(&mut serial)
                .expect("classic oracle");
            let actual = classic
                .decode_with_context(&mut context)
                .expect("classic with HT scratch");
            assert_eq!(actual.data, expected.data);
            assert_eq!(
                (actual.width, actual.height),
                (expected.width, expected.height)
            );
            let (classic_slots, classic_bank_bytes, tasks, growths) =
                parallel_workspace_measurements(&context);
            assert_eq!((classic_slots, classic_bank_bytes), (slots, bank_bytes));
            assert_eq!((tasks, growths), (0, 0));

            let repeated = ht
                .decode_with_context(&mut context)
                .expect("HT after classic");
            assert_eq!(repeated.data, first.data);
            let (warm_slots, warm_bytes, tasks, growths) =
                parallel_workspace_measurements(&context);
            assert_eq!((warm_slots, warm_bytes), (slots, bank_bytes));
            assert!(tasks > 0);
            assert_eq!(growths, 0);
            assert_eq!(context.workspace_stats().scratch_capacity_retries(), 0);
        });
    }
}

#[cfg(feature = "parallel")]
#[test]
#[ignore = "coefficient allocation measurement; run explicitly with --ignored --nocapture"]
fn parallel_coefficient_allocation_report() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("test pool");
    for ht in [false, true] {
        for reversible in [false, true] {
            let bytes = parallel_gray_fixture(257, 263, ht, reversible);
            let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
            pool.install(|| {
                let mut context = DecoderContext::default();
                let cold = image.decode_with_context(&mut context).expect("cold decode");
                let cold_stats = context.tile_decode_context.debug_counters.parallel_coefficients;
                let cold_retained = context.tile_decode_context.tier1_capacity_bytes().expect("retained bytes");
                let warm = image.decode_with_context(&mut context).expect("warm decode");
                let warm_stats = context.tile_decode_context.debug_counters.parallel_coefficients;
                let warm_retained = context.tile_decode_context.tier1_capacity_bytes().expect("retained bytes");
                assert_eq!(warm.data, cold.data);
                if ht {
                    assert!(cold_stats.direct_bytes > 0);
                    assert_eq!(cold_stats.allocations, 0);
                    assert_eq!(cold_stats.scatter_bytes, 0);
                } else {
                    assert!(cold_stats.allocations > 0);
                    assert!(cold_stats.scatter_bytes > 0);
                }
                assert_eq!(warm_stats.direct_bytes, cold_stats.direct_bytes);
                assert_eq!(warm_stats.scatter_bytes, cold_stats.scatter_bytes);
                println!("parallel_coefficients ht={ht} reversible={reversible} width=257 height=263 pool=2 cold={cold_stats:?} warm={warm_stats:?} cold_retained_tier1_bytes={cold_retained} warm_retained_tier1_bytes={warm_retained}");
            });
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_coefficient_buffers_reuse_capacity_across_fitting_images() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("test pool");
    for ht in [false, true] {
        for reversible in [false, true] {
            let first_bytes = parallel_gray_fixture(257, 263, ht, reversible);
            let second_bytes = parallel_gray_fixture(241, 127, ht, reversible);
            let first = Image::new(&first_bytes, &DecodeSettings::default()).expect("first image");
            let second =
                Image::new(&second_bytes, &DecodeSettings::default()).expect("second image");
            pool.install(|| {
                let mut context = DecoderContext::default();
                first.decode_with_context(&mut context).expect("reserve coefficient buffers");
                let first_stats = context.tile_decode_context.debug_counters.parallel_coefficients;
                if ht {
                    assert!(first_stats.direct_bytes > 0);
                    assert_eq!(first_stats.allocations, 0);
                } else {
                    assert!(first_stats.allocations > 0);
                }
                for image in [&first, &second, &first] {
                    let mut serial = DecoderContext::default();
                    serial.set_cpu_decode_parallelism(crate::CpuDecodeParallelism::Serial);
                    let expected = image.decode_with_context(&mut serial).expect("serial oracle");
                    let actual = image.decode_with_context(&mut context).expect("reused decode");
                    assert_eq!(actual.data, expected.data, "ht={ht}, reversible={reversible}");
                    assert_eq!((actual.width, actual.height), (expected.width, expected.height));
                    assert_eq!(context.tile_decode_context.debug_counters.parallel_coefficients.allocations, 0,
                        "fitting coefficient buffers must not allocate again: ht={ht}, reversible={reversible}");
                }
            });
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_ht_complete_stripes_write_destination_without_scatter() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for reversible in [false, true] {
        let bytes = parallel_gray_fixture(257, 263, true, reversible);
        let image = Image::new(&bytes, &DecodeSettings::default()).unwrap();
        pool.install(|| {
            let mut serial = DecoderContext::default();
            serial.set_cpu_decode_parallelism(crate::CpuDecodeParallelism::Serial);
            let expected = image.decode_with_context(&mut serial).unwrap();
            let mut parallel = DecoderContext::default();
            for _ in 0..2 {
                let actual = image.decode_with_context(&mut parallel).unwrap();
                assert_eq!(actual.data, expected.data);
                assert_eq!(
                    (actual.width, actual.height),
                    (expected.width, expected.height)
                );
                let stats = parallel
                    .tile_decode_context
                    .debug_counters
                    .parallel_coefficients;
                assert!(
                    stats.direct_bytes > 0,
                    "complete HT stripes must reach direct destination writes: {stats:?}"
                );
                assert_eq!(
                    stats.scatter_bytes, 0,
                    "eligible complete HT bands must not scatter"
                );
                assert_eq!(
                    stats.allocations, 0,
                    "direct HT bands must not allocate a coefficient slab"
                );
            }
        });
    }
}
