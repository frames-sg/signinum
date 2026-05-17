use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use signinum_j2k_native::{
    decode_ht_code_block_scalar, decode_j2k_code_block_scalar, encode_ht_code_block_scalar,
    encode_j2k_code_block_scalar_with_style, HtCodeBlockDecodeJob, J2kCodeBlockDecodeJob,
    J2kCodeBlockStyle, J2kSubBandType,
};

fn generated_coefficients(width: u32, height: u32, seed: u32) -> Vec<i32> {
    let mut state = seed ^ 0xa24b_aed4;
    let mut coefficients = Vec::with_capacity(width as usize * height as usize);
    for idx in 0..width * height {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let value = ((state >> 16) & 0x01ff) as i32 - 255;
        coefficients.push(if (idx + seed).is_multiple_of(13) {
            0
        } else {
            value
        });
    }
    coefficients
}

fn default_style() -> J2kCodeBlockStyle {
    J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: false,
        reset_context_probabilities: false,
        termination_on_each_pass: false,
        vertically_causal_context: false,
        segmentation_symbols: false,
    }
}

fn bench_tier1_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier1_bitplane_decode");
    let cases = [
        ("default", default_style(), J2kSubBandType::LowLow),
        (
            "bypass",
            J2kCodeBlockStyle {
                selective_arithmetic_coding_bypass: true,
                ..default_style()
            },
            J2kSubBandType::HighLow,
        ),
        (
            "segmented",
            J2kCodeBlockStyle {
                termination_on_each_pass: true,
                reset_context_probabilities: true,
                segmentation_symbols: true,
                ..default_style()
            },
            J2kSubBandType::HighHigh,
        ),
        (
            "vertically_causal",
            J2kCodeBlockStyle {
                vertically_causal_context: true,
                ..default_style()
            },
            J2kSubBandType::LowHigh,
        ),
    ];

    for (name, style, sub_band_type) in cases {
        let width = 64;
        let height = 64;
        let total_bitplanes = 10;
        let coefficients = generated_coefficients(width, height, 0x5151_0000);
        let encoded = encode_j2k_code_block_scalar_with_style(
            &coefficients,
            width,
            height,
            sub_band_type,
            total_bitplanes,
            style,
        )
        .expect("encode code block");
        let job = J2kCodeBlockDecodeJob {
            data: &encoded.data,
            segments: &encoded.segments,
            width,
            height,
            output_stride: width as usize,
            missing_bit_planes: encoded.missing_bit_planes,
            number_of_coding_passes: encoded.number_of_coding_passes,
            total_bitplanes,
            sub_band_type,
            style,
            strict: true,
            dequantization_step: 1.0,
        };
        let mut output = vec![0.0; width as usize * height as usize];

        group.bench_with_input(BenchmarkId::new("decode_64x64", name), &job, |b, job| {
            b.iter(|| {
                decode_j2k_code_block_scalar(*job, &mut output).expect("decode code block");
                black_box(&output);
            });
        });
    }

    group.finish();
}

fn bench_tier1_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier1_bitplane_encode");
    let cases = [
        ("default", default_style(), J2kSubBandType::LowLow),
        (
            "bypass",
            J2kCodeBlockStyle {
                selective_arithmetic_coding_bypass: true,
                ..default_style()
            },
            J2kSubBandType::HighLow,
        ),
        (
            "segmented",
            J2kCodeBlockStyle {
                termination_on_each_pass: true,
                reset_context_probabilities: true,
                segmentation_symbols: true,
                ..default_style()
            },
            J2kSubBandType::HighHigh,
        ),
        (
            "vertically_causal",
            J2kCodeBlockStyle {
                vertically_causal_context: true,
                ..default_style()
            },
            J2kSubBandType::LowHigh,
        ),
    ];

    for (name, style, sub_band_type) in cases {
        let width = 64;
        let height = 64;
        let total_bitplanes = 10;
        let coefficients = generated_coefficients(width, height, 0x7171_0000);

        group.bench_with_input(
            BenchmarkId::new("encode_64x64", name),
            &coefficients,
            |b, coefficients| {
                b.iter(|| {
                    let encoded = encode_j2k_code_block_scalar_with_style(
                        black_box(coefficients),
                        width,
                        height,
                        sub_band_type,
                        total_bitplanes,
                        style,
                    )
                    .expect("encode code block");
                    black_box(encoded);
                });
            },
        );
    }

    group.finish();
}

fn bench_htj2k_cleanup_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("htj2k_cleanup_decode");
    for &seed in &[0x9191_0000, 0x9191_0001] {
        let width = 64;
        let height = 64;
        let total_bitplanes = 10;
        let coefficients = generated_coefficients(width, height, seed);
        let encoded = encode_ht_code_block_scalar(&coefficients, width, height, total_bitplanes)
            .expect("encode HTJ2K code block");
        let job = HtCodeBlockDecodeJob {
            data: &encoded.data,
            cleanup_length: encoded.data.len() as u32,
            refinement_length: 0,
            width,
            height,
            output_stride: width as usize,
            missing_bit_planes: encoded.num_zero_bitplanes,
            number_of_coding_passes: encoded.num_coding_passes,
            num_bitplanes: total_bitplanes,
            stripe_causal: false,
            strict: true,
            dequantization_step: 1.0,
        };
        let mut output = vec![0.0; width as usize * height as usize];

        group.bench_with_input(BenchmarkId::new("decode_64x64", seed), &job, |b, job| {
            b.iter(|| {
                decode_ht_code_block_scalar(*job, &mut output).expect("decode HTJ2K code block");
                black_box(&output);
            });
        });
    }
    group.finish();
}

fn bench_htj2k_cleanup_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("htj2k_cleanup_encode");
    for &seed in &[0x9292_0000, 0x9292_0001] {
        let width = 64;
        let height = 64;
        let total_bitplanes = 10;
        let coefficients = generated_coefficients(width, height, seed);

        group.bench_with_input(
            BenchmarkId::new("encode_64x64", seed),
            &coefficients,
            |b, coefficients| {
                b.iter(|| {
                    let encoded = encode_ht_code_block_scalar(
                        black_box(coefficients),
                        width,
                        height,
                        total_bitplanes,
                    )
                    .expect("encode HTJ2K code block");
                    black_box(encoded);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_tier1_decode,
    bench_tier1_encode,
    bench_htj2k_cleanup_decode,
    bench_htj2k_cleanup_encode
);
criterion_main!(benches);
