use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use signinum_j2k_native::{
    decode_ht_code_block_scalar, decode_ht_sigprop_benchmark_state,
    prepare_ht_sigprop_benchmark_state, DecodeSettings, DecoderContext, HtCodeBlockDecodeJob,
    HtCodeBlockDecoder, Image, Result,
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
}

fn decode_scalar_ht_batch(
    jobs: &[OwnedHtCodeBlockDecodeJob],
    outputs: &mut [Vec<f32>],
) -> Result<()> {
    for (job, output) in jobs.iter().zip(outputs.iter_mut()) {
        decode_ht_code_block_scalar(
            job.borrowed_with_passes(job.number_of_coding_passes, job.refinement_length),
            output,
        )?;
    }
    Ok(())
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

fn bench_htj2k_refinement_sigprop_phase(c: &mut Criterion) {
    let mut group = c.benchmark_group("htj2k_refinement_sigprop_phase");
    let jobs = collect_refinement_fixture_jobs();
    let job = jobs
        .first()
        .expect("fixture must contain a refinement HTJ2K block");
    let sigprop_job = job.borrowed_with_passes(2, job.refinement_length);
    let mut state =
        prepare_ht_sigprop_benchmark_state(sigprop_job).expect("prepare SigProp benchmark state");
    let mut output = vec![0u32; state.output_len()];

    group.bench_function("ds0_ht_09_b11_sigprop_only", |b| {
        b.iter(|| {
            decode_ht_sigprop_benchmark_state(black_box(&mut state), &mut output)
                .expect("decode HTJ2K SigProp phase");
            black_box(&output);
        });
    });

    group.finish();
}

fn bench_htj2k_cpuupload_decode_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("htj2k_cpuupload_decode_batch");
    let fixture_jobs = collect_refinement_fixture_jobs();

    for count in [1_usize, 2, 4, fixture_jobs.len()] {
        let jobs = fixture_jobs
            .iter()
            .cloned()
            .cycle()
            .take(count)
            .collect::<Vec<_>>();
        let mut outputs = jobs
            .iter()
            .map(|job| vec![0.0_f32; job.width as usize * job.height as usize])
            .collect::<Vec<_>>();

        group.bench_with_input(
            BenchmarkId::new("ds0_ht_09_b11_scalar_batch", count),
            &count,
            |b, _| {
                b.iter(|| {
                    decode_scalar_ht_batch(black_box(&jobs), black_box(&mut outputs))
                        .expect("decode HTJ2K scalar batch");
                    black_box(&outputs);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_htj2k_refinement_sigprop_phase,
    bench_htj2k_cpuupload_decode_batch
);
criterion_main!(benches);
