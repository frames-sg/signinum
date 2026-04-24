// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use crate::compute;
use slidecodec_j2k_native::{
    decode_ht_code_block_scalar, DecodingError, HtCodeBlockDecodeJob, HtCodeBlockDecoder,
    HtSubBandDecodeJob, Result,
};

#[allow(dead_code)]
#[derive(Default)]
pub(crate) struct MetalHtBlockDecoder {
    blocks_decoded: usize,
    kernel_dispatches: usize,
    sub_band_batches: usize,
    batched_kernel_dispatches: usize,
}

#[allow(dead_code)]
impl MetalHtBlockDecoder {
    #[cfg(test)]
    pub(crate) fn blocks_decoded(&self) -> usize {
        self.blocks_decoded
    }

    #[cfg(test)]
    pub(crate) fn kernel_dispatches(&self) -> usize {
        self.kernel_dispatches
    }

    #[cfg(test)]
    pub(crate) fn sub_band_batches(&self) -> usize {
        self.sub_band_batches
    }

    #[cfg(test)]
    pub(crate) fn batched_kernel_dispatches(&self) -> usize {
        self.batched_kernel_dispatches
    }
}

impl HtCodeBlockDecoder for MetalHtBlockDecoder {
    fn decode_sub_band(&mut self, job: HtSubBandDecodeJob<'_>, output: &mut [f32]) -> Result<bool> {
        #[cfg(target_os = "macos")]
        if job.jobs.len() > 1
            && job
                .jobs
                .iter()
                .all(|job| supports_metal_ht_kernel(&job.code_block))
        {
            compute::decode_ht_cleanup_sub_band(job, output)
                .map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
            self.sub_band_batches = self.sub_band_batches.saturating_add(1);
            self.batched_kernel_dispatches = self.batched_kernel_dispatches.saturating_add(1);
            return Ok(true);
        }

        Ok(false)
    }

    fn decode_code_block(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> Result<()> {
        self.blocks_decoded = self.blocks_decoded.saturating_add(1);
        #[cfg(target_os = "macos")]
        if supports_metal_ht_kernel(&job) {
            compute::decode_ht_cleanup_code_block(job, output)
                .map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
            self.kernel_dispatches = self.kernel_dispatches.saturating_add(1);
            return Ok(());
        }
        decode_ht_code_block_scalar(job, output)
    }
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn supports_metal_ht_kernel(job: &HtCodeBlockDecodeJob<'_>) -> bool {
    if job.width == 0 || job.height == 0 {
        return false;
    }
    if job.width > 64 || job.height > 64 {
        return false;
    }
    if job.num_bitplanes == 0 || job.num_bitplanes > 31 || job.missing_bit_planes >= 30 {
        return false;
    }
    if job.number_of_coding_passes == 0 || job.number_of_coding_passes > 3 {
        return false;
    }
    let Ok(cleanup_len) = usize::try_from(job.cleanup_length) else {
        return false;
    };
    let Ok(refinement_len) = usize::try_from(job.refinement_length) else {
        return false;
    };
    if cleanup_len
        .checked_add(refinement_len)
        .is_none_or(|len| len != job.data.len())
    {
        return false;
    }
    if job.output_stride < job.width as usize {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::MetalHtBlockDecoder;
    #[cfg(target_os = "macos")]
    use crate::compute;
    use slidecodec_j2k_native::{
        decode_ht_code_block_scalar, encode_htj2k, ColorSpace, DecodeSettings, DecoderContext,
        EncodeOptions, HtCodeBlockDecodeJob, HtCodeBlockDecoder, Image,
    };

    #[derive(Clone)]
    struct OwnedHtJob {
        data: Vec<u8>,
        cleanup_length: u32,
        refinement_length: u32,
        width: u32,
        height: u32,
        output_stride: usize,
        missing_bit_planes: u8,
        number_of_coding_passes: u8,
        num_bitplanes: u8,
        stripe_causal: bool,
        strict: bool,
        dequantization_step: f32,
    }

    impl OwnedHtJob {
        fn as_job(&self) -> HtCodeBlockDecodeJob<'_> {
            HtCodeBlockDecodeJob {
                data: &self.data,
                cleanup_length: self.cleanup_length,
                refinement_length: self.refinement_length,
                width: self.width,
                height: self.height,
                output_stride: self.output_stride,
                missing_bit_planes: self.missing_bit_planes,
                number_of_coding_passes: self.number_of_coding_passes,
                num_bitplanes: self.num_bitplanes,
                stripe_causal: self.stripe_causal,
                strict: self.strict,
                dequantization_step: self.dequantization_step,
            }
        }

        fn output_len(&self) -> usize {
            self.output_stride * self.height as usize
        }
    }

    #[derive(Default)]
    struct CaptureFirstHtJob {
        first: Option<OwnedHtJob>,
    }

    impl HtCodeBlockDecoder for CaptureFirstHtJob {
        fn decode_code_block(
            &mut self,
            job: HtCodeBlockDecodeJob<'_>,
            output: &mut [f32],
        ) -> slidecodec_j2k_native::Result<()> {
            if self.first.is_none() {
                self.first = Some(OwnedHtJob {
                    data: job.data.to_vec(),
                    cleanup_length: job.cleanup_length,
                    refinement_length: job.refinement_length,
                    width: job.width,
                    height: job.height,
                    output_stride: job.output_stride,
                    missing_bit_planes: job.missing_bit_planes,
                    number_of_coding_passes: job.number_of_coding_passes,
                    num_bitplanes: job.num_bitplanes,
                    stripe_causal: job.stripe_causal,
                    strict: job.strict,
                    dequantization_step: job.dequantization_step,
                });
            }

            decode_ht_code_block_scalar(job, output)
        }
    }

    fn fixture_ht_gray8() -> Vec<u8> {
        let pixels: Vec<u8> = (0..16).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        encode_htj2k(&pixels, 4, 4, 1, 8, false, &options).expect("encode ht gray8")
    }

    fn fixture_ht_gray8_multi_block() -> Vec<u8> {
        let pixels: Vec<u8> = (0..64).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            code_block_width_exp: 0,
            code_block_height_exp: 0,
            ..EncodeOptions::default()
        };
        encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode multi-block ht gray8")
    }

    fn synthetic_refinement_job(number_of_coding_passes: u8) -> OwnedHtJob {
        let bytes = fixture_ht_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut capture = CaptureFirstHtJob::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut capture)
            .expect("capture cleanup job");

        let mut job = capture.first.expect("captured cleanup job");
        job.data.push(0);
        job.refinement_length = 1;
        job.number_of_coding_passes = number_of_coding_passes;
        job
    }

    #[test]
    fn metal_ht_decoder_matches_native_decode() {
        let bytes = fixture_ht_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");

        let mut baseline_context = DecoderContext::default();
        let baseline = image
            .decode_components_with_context(&mut baseline_context)
            .expect("baseline decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalHtBlockDecoder::default();
        let hooked = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        assert!(
            decoder.blocks_decoded() > 0,
            "HT codeblock hook must be used"
        );
        #[cfg(target_os = "macos")]
        assert!(
            decoder.kernel_dispatches() > 0,
            "cleanup-only HTJ2K fixture must execute the Metal kernel"
        );
        assert_eq!(hooked.dimensions(), baseline.dimensions());
        assert!(matches!(hooked.color_space(), ColorSpace::Gray));
        assert_eq!(
            core::mem::discriminant(hooked.color_space()),
            core::mem::discriminant(baseline.color_space())
        );
        assert_eq!(hooked.has_alpha(), baseline.has_alpha());
        assert_eq!(hooked.planes().len(), baseline.planes().len());

        for (hooked_plane, baseline_plane) in hooked.planes().iter().zip(baseline.planes()) {
            assert_eq!(hooked_plane.bit_depth(), baseline_plane.bit_depth());
            assert_eq!(hooked_plane.samples(), baseline_plane.samples());
        }
    }

    #[test]
    fn metal_ht_decoder_batches_multi_block_subbands() {
        let bytes = fixture_ht_gray8_multi_block();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");

        let mut baseline_context = DecoderContext::default();
        let baseline = image
            .decode_components_with_context(&mut baseline_context)
            .expect("baseline decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalHtBlockDecoder::default();
        let hooked = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        #[cfg(target_os = "macos")]
        assert!(
            decoder.sub_band_batches() > 0,
            "multi-block HTJ2K fixture must exercise the batched Metal path"
        );
        #[cfg(target_os = "macos")]
        assert_eq!(
            decoder.batched_kernel_dispatches(),
            1,
            "multi-block HTJ2K fixture must complete in one batched Metal dispatch"
        );
        assert_eq!(hooked.dimensions(), baseline.dimensions());
        assert_eq!(hooked.planes().len(), baseline.planes().len());
        for (hooked_plane, baseline_plane) in hooked.planes().iter().zip(baseline.planes()) {
            assert_eq!(hooked_plane.samples(), baseline_plane.samples());
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_ht_kernel_matches_scalar_for_synthetic_refinement_job() {
        let job = synthetic_refinement_job(3);
        assert!(
            super::supports_metal_ht_kernel(&job.as_job()),
            "synthetic refinement job must stay on the Metal kernel path"
        );

        let mut expected = vec![0.0f32; job.output_len()];
        decode_ht_code_block_scalar(job.as_job(), &mut expected).expect("scalar decode");

        let mut actual = vec![0.0f32; job.output_len()];
        compute::decode_ht_cleanup_code_block(job.as_job(), &mut actual).expect("metal decode");

        assert_eq!(actual, expected);
    }
}
