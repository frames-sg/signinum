// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{checked_buffer_copy_into, copied_slice_buffer};
use std::time::{Duration, Instant};

const CASES: [(u32, u32); 2] = [(512, 512), (509, 383)];

fn seeded_values(width: u32, height: u32) -> Vec<f32> {
    let len = width as usize * height as usize;
    (0..len)
        .map(|index| {
            let value = f32::from(u16::try_from(index % 257).expect("seed value fits u16"));
            (value - 128.0) * 0.03125
        })
        .collect()
}

#[test]
#[ignore = "host copy performance harness; run explicitly with --ignored --nocapture"]
fn metal_checked_buffer_copy_into_perf() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    crate::engine::with_runtime(|runtime| {
        for (width, height) in CASES {
            let expected = seeded_values(width, height);
            let buffer = copied_slice_buffer(&runtime.device, &expected).expect("copy source");
            let mut output = vec![0.0_f32; expected.len()];
            checked_buffer_copy_into(&buffer, 0, &mut output, "copy timing probe")?;
            assert!(output
                .iter()
                .zip(&expected)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));

            let warm_started = Instant::now();
            let mut warm_iterations = 0_u64;
            while warm_started.elapsed() < Duration::from_secs(3) {
                checked_buffer_copy_into(&buffer, 0, &mut output, "copy timing warmup")?;
                std::hint::black_box(&output);
                warm_iterations += 1;
            }
            let iterations = (warm_iterations / 15).max(1);
            for sample in 0..50 {
                crate::engine::reset_idwt_host_transfer_counters_for_test();
                let started = Instant::now();
                for _ in 0..iterations {
                    checked_buffer_copy_into(&buffer, 0, &mut output, "copy timing sample")?;
                    std::hint::black_box(&output);
                }
                let elapsed = started.elapsed();
                let (upload_bytes, readback_allocations, readback_bytes) =
                    crate::engine::idwt_host_transfer_counters_for_test();
                println!(
                    "metal_checked_buffer_copy_into width={width} height={height} sample={sample} iterations={iterations} elapsed_ns={} overwritten_output_upload_bytes={upload_bytes} temporary_readback_vec_allocations={readback_allocations} temporary_readback_vec_bytes={readback_bytes}",
                    elapsed.as_nanos(),
                );
            }
        }
        Ok(())
    })
    .expect("run checked Metal buffer copy timing harness");
}
