use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use signinum_j2k_native::{
    collect_ht_cleanup_encode_distribution, decode_ht_code_block_scalar,
    decode_j2k_code_block_scalar, encode_ht_code_block_scalar,
    encode_j2k_code_block_scalar_with_style, DecodeSettings, DecoderContext, HtCodeBlockDecodeJob,
    HtCodeBlockDecoder, Image, J2kCodeBlockDecodeJob, J2kCodeBlockStyle, J2kSubBandType, Result,
};

const HTJ2K_REFINEMENT_FIXTURE: &[u8] =
    include_bytes!("../fixtures/htj2k/openhtj2k_ds0_ht_09_b11.j2k");

#[derive(Clone)]
struct OwnedHtCodeBlockDecodeJob {
    data: Vec<u8>,
    cleanup_length: u32,
    refinement_length: u32,
    width: u32,
    height: u32,
    missing_bit_planes: u8,
    number_of_coding_passes: u8,
    num_bitplanes: u8,
    stripe_causal: bool,
    strict: bool,
    dequantization_step: f32,
}

impl OwnedHtCodeBlockDecodeJob {
    fn from_job(job: HtCodeBlockDecodeJob<'_>) -> Self {
        Self {
            data: job.data.to_vec(),
            cleanup_length: job.cleanup_length,
            refinement_length: job.refinement_length,
            width: job.width,
            height: job.height,
            missing_bit_planes: job.missing_bit_planes,
            number_of_coding_passes: job.number_of_coding_passes,
            num_bitplanes: job.num_bitplanes,
            stripe_causal: job.stripe_causal,
            strict: job.strict,
            dequantization_step: job.dequantization_step,
        }
    }

    fn borrowed_with_passes(
        &self,
        number_of_coding_passes: u8,
        refinement_length: u32,
    ) -> HtCodeBlockDecodeJob<'_> {
        HtCodeBlockDecodeJob {
            data: &self.data,
            cleanup_length: self.cleanup_length,
            refinement_length,
            width: self.width,
            height: self.height,
            output_stride: self.width as usize,
            missing_bit_planes: self.missing_bit_planes,
            number_of_coding_passes,
            num_bitplanes: self.num_bitplanes,
            stripe_causal: self.stripe_causal,
            strict: self.strict,
            dequantization_step: self.dequantization_step,
        }
    }

    fn output_len(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

#[derive(Default)]
struct CollectingHtDecoder {
    jobs: Vec<OwnedHtCodeBlockDecodeJob>,
}

impl HtCodeBlockDecoder for CollectingHtDecoder {
    fn decode_code_block(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> Result<()> {
        if job.refinement_length > 0 {
            self.jobs.push(OwnedHtCodeBlockDecodeJob::from_job(job));
        }

        decode_ht_code_block_scalar(job, output)
    }
}

fn collect_refinement_fixture_jobs() -> Vec<OwnedHtCodeBlockDecodeJob> {
    let image =
        Image::new(HTJ2K_REFINEMENT_FIXTURE, &DecodeSettings::default()).expect("fixture image");
    let mut context = DecoderContext::default();
    let mut decoder = CollectingHtDecoder::default();
    image
        .decode_components_with_ht_decoder(&mut context, &mut decoder)
        .expect("decode fixture while collecting HTJ2K jobs");
    assert_eq!(
        decoder.jobs.len(),
        14,
        "expected every ds0_ht_09_b11 HT block to carry refinement data"
    );
    decoder.jobs
}

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

fn generated_htj2k_encode_batch(blocks: usize, width: u32, height: u32) -> Vec<Vec<i32>> {
    (0..blocks)
        .map(|index| generated_coefficients(width, height, 0xA5A5_0000 ^ index as u32))
        .collect()
}

fn encoded_ht_block_len(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
) -> usize {
    let encoded = encode_ht_code_block_scalar(coefficients, width, height, total_bitplanes)
        .expect("encode HTJ2K code block");
    encoded.data.len()
        + usize::from(encoded.num_coding_passes)
        + usize::from(encoded.num_zero_bitplanes)
}

fn encode_ht_batch_serial(
    blocks: &[Vec<i32>],
    width: u32,
    height: u32,
    total_bitplanes: u8,
) -> usize {
    blocks
        .iter()
        .map(|coefficients| encoded_ht_block_len(coefficients, width, height, total_bitplanes))
        .sum()
}

fn encode_ht_batch_rayon(
    blocks: &[Vec<i32>],
    width: u32,
    height: u32,
    total_bitplanes: u8,
) -> usize {
    blocks
        .par_iter()
        .map(|coefficients| encoded_ht_block_len(coefficients, width, height, total_bitplanes))
        .sum()
}

fn encode_ht_batch_rayon_chunks(
    blocks: &[Vec<i32>],
    chunk_size: usize,
    width: u32,
    height: u32,
    total_bitplanes: u8,
) -> usize {
    blocks
        .par_chunks(chunk_size)
        .map(|chunk| encode_ht_batch_serial(chunk, width, height, total_bitplanes))
        .sum()
}

fn encoded_j2k_block_len(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
) -> usize {
    let encoded = encode_j2k_code_block_scalar_with_style(
        coefficients,
        width,
        height,
        J2kSubBandType::LowLow,
        total_bitplanes,
        default_style(),
    )
    .expect("encode J2K code block");
    encoded.data.len()
        + encoded.segments.len()
        + usize::from(encoded.number_of_coding_passes)
        + usize::from(encoded.missing_bit_planes)
}

fn encode_j2k_batch_serial(
    blocks: &[Vec<i32>],
    width: u32,
    height: u32,
    total_bitplanes: u8,
) -> usize {
    blocks
        .iter()
        .map(|coefficients| encoded_j2k_block_len(coefficients, width, height, total_bitplanes))
        .sum()
}

fn encode_j2k_batch_rayon(
    blocks: &[Vec<i32>],
    width: u32,
    height: u32,
    total_bitplanes: u8,
) -> usize {
    blocks
        .par_iter()
        .map(|coefficients| encoded_j2k_block_len(coefficients, width, height, total_bitplanes))
        .sum()
}

fn benchmark_thread_counts() -> Vec<usize> {
    let available = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let mut counts = vec![1, available.min(2), available.min(4), available];
    counts.sort_unstable();
    counts.dedup();
    counts
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

fn bench_htj2k_refinement_fixture_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("htj2k_refinement_fixture_decode");
    let image =
        Image::new(HTJ2K_REFINEMENT_FIXTURE, &DecodeSettings::default()).expect("fixture image");

    group.bench_function("ds0_ht_09_b11_full", |b| {
        b.iter(|| {
            let mut context = DecoderContext::default();
            let components = image
                .decode_components_with_context(&mut context)
                .expect("decode HTJ2K refinement fixture");
            black_box(components.planes()[0].samples());
        });
    });

    group.finish();
}

fn bench_htj2k_refinement_block_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("htj2k_refinement_block_decode");
    let jobs = collect_refinement_fixture_jobs();
    let job = jobs
        .first()
        .expect("fixture must contain a refinement HTJ2K block");
    let cleanup_job = job.borrowed_with_passes(1, 0);
    let sigprop_job = job.borrowed_with_passes(2, job.refinement_length);
    let magref_job = job.borrowed_with_passes(job.number_of_coding_passes, job.refinement_length);

    let mut cleanup_output = vec![0.0; job.output_len()];
    group.bench_function("ds0_ht_09_b11_cleanup", |b| {
        b.iter(|| {
            cleanup_output.fill(0.0);
            decode_ht_code_block_scalar(black_box(cleanup_job), &mut cleanup_output)
                .expect("cleanup-limited HTJ2K decode");
            black_box(&cleanup_output);
        });
    });

    let mut sigprop_output = vec![0.0; job.output_len()];
    group.bench_function("ds0_ht_09_b11_sigprop", |b| {
        b.iter(|| {
            sigprop_output.fill(0.0);
            decode_ht_code_block_scalar(black_box(sigprop_job), &mut sigprop_output)
                .expect("significance-propagation-limited HTJ2K decode");
            black_box(&sigprop_output);
        });
    });

    let mut magref_output = vec![0.0; job.output_len()];
    group.bench_function("ds0_ht_09_b11_magref_full", |b| {
        b.iter(|| {
            magref_output.fill(0.0);
            decode_ht_code_block_scalar(black_box(magref_job), &mut magref_output)
                .expect("full HTJ2K decode through magnitude refinement");
            black_box(&magref_output);
        });
    });

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

fn bench_htj2k_cleanup_encode_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("htj2k_cleanup_encode_distribution");
    for &seed in &[0x9292_0000, 0x9292_0001] {
        let width = 64;
        let height = 64;
        let total_bitplanes = 10;
        let coefficients = generated_coefficients(width, height, seed);

        group.bench_with_input(
            BenchmarkId::new("rho_eq_uq_64x64", seed),
            &coefficients,
            |b, coefficients| {
                b.iter(|| {
                    let distribution = collect_ht_cleanup_encode_distribution(
                        black_box(coefficients),
                        width,
                        height,
                        total_bitplanes,
                    )
                    .expect("collect HTJ2K cleanup encode distribution");
                    black_box(distribution);
                });
            },
        );
    }
    group.finish();
}

fn bench_htj2k_cleanup_encode_parallel_granularity(c: &mut Criterion) {
    let mut group = c.benchmark_group("htj2k_cleanup_encode_parallel_granularity");
    let width = 64;
    let height = 64;
    let total_bitplanes = 10;
    let blocks = generated_htj2k_encode_batch(128, width, height);

    group.bench_function("serial_128_blocks", |b| {
        b.iter(|| {
            let total_len =
                encode_ht_batch_serial(black_box(&blocks), width, height, total_bitplanes);
            black_box(total_len);
        });
    });

    group.bench_function("rayon_par_iter_global_128_blocks", |b| {
        b.iter(|| {
            let total_len =
                encode_ht_batch_rayon(black_box(&blocks), width, height, total_bitplanes);
            black_box(total_len);
        });
    });

    for threads in benchmark_thread_counts() {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build benchmark Rayon pool");
        group.bench_with_input(
            BenchmarkId::new("rayon_par_iter_threads", threads),
            &threads,
            |b, _| {
                b.iter(|| {
                    let total_len = pool.install(|| {
                        encode_ht_batch_rayon(black_box(&blocks), width, height, total_bitplanes)
                    });
                    black_box(total_len);
                });
            },
        );
    }

    let available = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let chunk_pool = ThreadPoolBuilder::new()
        .num_threads(available)
        .build()
        .expect("build benchmark Rayon chunk pool");
    for chunk_size in [4usize, 8, 16, 32, 64] {
        group.bench_with_input(
            BenchmarkId::new("rayon_par_chunks_128_blocks", chunk_size),
            &chunk_size,
            |b, &chunk_size| {
                b.iter(|| {
                    let total_len = chunk_pool.install(|| {
                        encode_ht_batch_rayon_chunks(
                            black_box(&blocks),
                            chunk_size,
                            width,
                            height,
                            total_bitplanes,
                        )
                    });
                    black_box(total_len);
                });
            },
        );
    }

    group.finish();
}

fn bench_htj2k_cleanup_encode_parallel_batch_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("htj2k_cleanup_encode_parallel_batch_size");
    let width = 64;
    let height = 64;
    let total_bitplanes = 10;

    for block_count in [1usize, 2, 4, 8, 16, 32, 128] {
        let blocks = generated_htj2k_encode_batch(block_count, width, height);
        group.bench_with_input(
            BenchmarkId::new("serial_blocks", block_count),
            &blocks,
            |b, blocks| {
                b.iter(|| {
                    let total_len =
                        encode_ht_batch_serial(black_box(blocks), width, height, total_bitplanes);
                    black_box(total_len);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("rayon_par_iter_global_blocks", block_count),
            &blocks,
            |b, blocks| {
                b.iter(|| {
                    let total_len =
                        encode_ht_batch_rayon(black_box(blocks), width, height, total_bitplanes);
                    black_box(total_len);
                });
            },
        );
    }

    group.finish();
}

fn bench_j2k_tier1_encode_parallel_batch_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("j2k_tier1_encode_parallel_batch_size");
    let width = 64;
    let height = 64;
    let total_bitplanes = 10;

    for block_count in [1usize, 2, 4, 8, 16, 32, 128] {
        let blocks = generated_htj2k_encode_batch(block_count, width, height);
        group.bench_with_input(
            BenchmarkId::new("serial_blocks", block_count),
            &blocks,
            |b, blocks| {
                b.iter(|| {
                    let total_len =
                        encode_j2k_batch_serial(black_box(blocks), width, height, total_bitplanes);
                    black_box(total_len);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("rayon_par_iter_global_blocks", block_count),
            &blocks,
            |b, blocks| {
                b.iter(|| {
                    let total_len =
                        encode_j2k_batch_rayon(black_box(blocks), width, height, total_bitplanes);
                    black_box(total_len);
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
    bench_htj2k_refinement_fixture_decode,
    bench_htj2k_refinement_block_decode,
    bench_htj2k_cleanup_encode,
    bench_htj2k_cleanup_encode_distribution,
    bench_htj2k_cleanup_encode_parallel_granularity,
    bench_htj2k_cleanup_encode_parallel_batch_size,
    bench_j2k_tier1_encode_parallel_batch_size
);
criterion_main!(benches);
