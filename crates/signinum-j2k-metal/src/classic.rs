// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use crate::compute;
#[cfg(target_os = "macos")]
use signinum_j2k_native::DecodingError;
use signinum_j2k_native::{
    decode_ht_code_block_scalar, decode_j2k_code_block_scalar, decode_j2k_sub_band_scalar,
    HtCodeBlockDecodeJob, HtCodeBlockDecoder, J2kCodeBlockDecodeJob, J2kSubBandDecodeJob, Result,
};

#[derive(Default)]
pub(crate) struct MetalClassicBlockDecoder {
    blocks_decoded: usize,
    #[cfg(target_os = "macos")]
    kernel_dispatches: usize,
    sub_band_batches: usize,
    #[cfg(target_os = "macos")]
    batched_kernel_dispatches: usize,
}

impl MetalClassicBlockDecoder {
    #[cfg(test)]
    pub(crate) fn blocks_decoded(&self) -> usize {
        self.blocks_decoded
    }

    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn kernel_dispatches(&self) -> usize {
        self.kernel_dispatches
    }

    #[cfg(test)]
    pub(crate) fn sub_band_batches(&self) -> usize {
        self.sub_band_batches
    }

    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn batched_kernel_dispatches(&self) -> usize {
        self.batched_kernel_dispatches
    }
}

impl HtCodeBlockDecoder for MetalClassicBlockDecoder {
    fn decode_j2k_sub_band(
        &mut self,
        job: J2kSubBandDecodeJob<'_>,
        output: &mut [f32],
    ) -> Result<bool> {
        if job.jobs.len() <= 1 {
            return Ok(false);
        }

        self.sub_band_batches = self.sub_band_batches.saturating_add(1);
        #[cfg(target_os = "macos")]
        if job
            .jobs
            .iter()
            .all(|batch_job| supports_metal_classic_kernel(&batch_job.code_block))
        {
            compute::decode_classic_cleanup_sub_band(job, output)
                .map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
            self.batched_kernel_dispatches = self.batched_kernel_dispatches.saturating_add(1);
            return Ok(true);
        }

        decode_j2k_sub_band_scalar(job, output)?;
        Ok(true)
    }

    fn decode_j2k_code_block(
        &mut self,
        job: J2kCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> Result<bool> {
        self.blocks_decoded = self.blocks_decoded.saturating_add(1);
        #[cfg(target_os = "macos")]
        if supports_metal_classic_kernel(&job) {
            compute::decode_classic_cleanup_code_block(job, output)
                .map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
            self.kernel_dispatches = self.kernel_dispatches.saturating_add(1);
            return Ok(true);
        }

        decode_j2k_code_block_scalar(job, output)?;
        Ok(true)
    }

    fn decode_code_block(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> Result<()> {
        decode_ht_code_block_scalar(job, output)
    }
}

#[cfg(target_os = "macos")]
fn supports_metal_classic_kernel(job: &J2kCodeBlockDecodeJob<'_>) -> bool {
    if job.width == 0 || job.height == 0 {
        return false;
    }
    if job.width > 64 || job.height > 64 {
        return false;
    }
    if job.output_stride < job.width as usize {
        return false;
    }
    if job.number_of_coding_passes == 0 {
        return false;
    }
    if job.data.is_empty() {
        return false;
    }
    if job.total_bitplanes == 0 || job.total_bitplanes > 31 || job.missing_bit_planes >= 31 {
        return false;
    }
    let bitplanes = job.total_bitplanes.saturating_sub(job.missing_bit_planes);
    if bitplanes == 0 {
        return false;
    }
    let max_coding_passes = 1 + 3 * (bitplanes - 1);
    if job.number_of_coding_passes > max_coding_passes {
        return false;
    }
    if job.segments.is_empty() {
        return false;
    }

    let uses_bypass = job.style.selective_arithmetic_coding_bypass;
    let mut expected_start = 0u8;
    let mut expected_offset = 0usize;
    for segment in job.segments {
        if segment.start_coding_pass != expected_start
            || segment.start_coding_pass > segment.end_coding_pass
        {
            return false;
        }
        if uses_bypass {
            let expected_arithmetic =
                segment.start_coding_pass <= 9 || segment.start_coding_pass % 3 == 0;
            if segment.use_arithmetic != expected_arithmetic {
                return false;
            }
            if !segment.use_arithmetic {
                if segment.start_coding_pass % 3 != 1 {
                    return false;
                }
                if segment
                    .end_coding_pass
                    .saturating_sub(segment.start_coding_pass)
                    > 2
                {
                    return false;
                }
                if (segment.start_coding_pass..segment.end_coding_pass).any(|pass| pass % 3 == 0) {
                    return false;
                }
            }
        } else if !segment.use_arithmetic {
            return false;
        }
        let Some(data_start) = usize::try_from(segment.data_offset).ok() else {
            return false;
        };
        let Some(data_len) = usize::try_from(segment.data_length).ok() else {
            return false;
        };
        let Some(data_end) = data_start.checked_add(data_len) else {
            return false;
        };
        if data_start != expected_offset || data_end > job.data.len() {
            return false;
        }
        expected_offset = data_end;
        expected_start = segment.end_coding_pass;
    }
    expected_start == job.number_of_coding_passes && expected_offset == job.data.len()
}

#[cfg(test)]
mod tests {
    #![cfg_attr(not(target_os = "macos"), allow(dead_code))]

    use super::MetalClassicBlockDecoder;
    #[cfg(target_os = "macos")]
    use crate::compute;
    use signinum_j2k_native::{
        decode_j2k_code_block_scalar, encode, ColorSpace, DecodeSettings, DecoderContext,
        EncodeOptions, HtCodeBlockDecodeJob, HtCodeBlockDecoder, Image, J2kCodeBlockDecodeJob,
        J2kCodeBlockSegment,
    };

    #[derive(Clone)]
    struct OwnedClassicJob {
        data: Vec<u8>,
        segments: Vec<signinum_j2k_native::J2kCodeBlockSegment>,
        width: u32,
        height: u32,
        output_stride: usize,
        missing_bit_planes: u8,
        number_of_coding_passes: u8,
        total_bitplanes: u8,
        sub_band_type: signinum_j2k_native::J2kSubBandType,
        style: signinum_j2k_native::J2kCodeBlockStyle,
        strict: bool,
        dequantization_step: f32,
    }

    impl OwnedClassicJob {
        fn as_job(&self) -> J2kCodeBlockDecodeJob<'_> {
            J2kCodeBlockDecodeJob {
                data: &self.data,
                segments: &self.segments,
                width: self.width,
                height: self.height,
                output_stride: self.output_stride,
                missing_bit_planes: self.missing_bit_planes,
                number_of_coding_passes: self.number_of_coding_passes,
                total_bitplanes: self.total_bitplanes,
                sub_band_type: self.sub_band_type,
                style: self.style,
                strict: self.strict,
                dequantization_step: self.dequantization_step,
            }
        }

        fn output_len(&self) -> usize {
            self.output_stride * self.height as usize
        }
    }

    fn split_job_into_two_valid_segments(job: &OwnedClassicJob) -> OwnedClassicJob {
        let mut baseline = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut baseline).expect("baseline decode");

        for split_pass in 1..job.number_of_coding_passes {
            for split_offset in 1..job.data.len() {
                let mut candidate = job.clone();
                let split_offset_u32 =
                    u32::try_from(split_offset).expect("split offset fits in u32");
                let remaining_len_u32 = u32::try_from(job.data.len() - split_offset)
                    .expect("remaining segment length fits in u32");
                candidate.segments = vec![
                    J2kCodeBlockSegment {
                        data_offset: 0,
                        data_length: split_offset_u32,
                        start_coding_pass: 0,
                        end_coding_pass: split_pass,
                        use_arithmetic: true,
                    },
                    J2kCodeBlockSegment {
                        data_offset: split_offset_u32,
                        data_length: remaining_len_u32,
                        start_coding_pass: split_pass,
                        end_coding_pass: job.number_of_coding_passes,
                        use_arithmetic: true,
                    },
                ];
                let mut candidate_output = vec![0.0f32; candidate.output_len()];
                if decode_j2k_code_block_scalar(candidate.as_job(), &mut candidate_output).is_ok()
                    && candidate_output == baseline
                {
                    return candidate;
                }
            }
        }

        panic!("expected to find a valid two-segment classic codeblock split");
    }

    #[derive(Default)]
    struct CaptureFirstClassicJob {
        first: Option<OwnedClassicJob>,
    }

    impl HtCodeBlockDecoder for CaptureFirstClassicJob {
        fn decode_j2k_code_block(
            &mut self,
            job: J2kCodeBlockDecodeJob<'_>,
            output: &mut [f32],
        ) -> signinum_j2k_native::Result<bool> {
            if self.first.is_none() {
                self.first = Some(OwnedClassicJob {
                    data: job.data.to_vec(),
                    segments: job.segments.to_vec(),
                    width: job.width,
                    height: job.height,
                    output_stride: job.output_stride,
                    missing_bit_planes: job.missing_bit_planes,
                    number_of_coding_passes: job.number_of_coding_passes,
                    total_bitplanes: job.total_bitplanes,
                    sub_band_type: job.sub_band_type,
                    style: job.style,
                    strict: job.strict,
                    dequantization_step: job.dequantization_step,
                });
            }
            decode_j2k_code_block_scalar(job, output)?;
            Ok(true)
        }

        fn decode_code_block(
            &mut self,
            job: HtCodeBlockDecodeJob<'_>,
            output: &mut [f32],
        ) -> signinum_j2k_native::Result<()> {
            signinum_j2k_native::decode_ht_code_block_scalar(job, output)
        }
    }

    fn fixture_j2k_gray8() -> Vec<u8> {
        let pixels: Vec<u8> = (0..16).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode classic gray8")
    }

    fn fixture_j2k_gray1_cleanup_only() -> Vec<u8> {
        let pixels = vec![0u8, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0];
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            ..EncodeOptions::default()
        };
        encode(&pixels, 4, 4, 1, 1, false, &options).expect("encode classic gray1")
    }

    fn fixture_j2k_gray1_cleanup_only_multi_block() -> Vec<u8> {
        let pixels = vec![
            0u8, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0,
            1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1,
            0, 1, 0, 1, 0, 1, 0,
        ];
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            code_block_width_exp: 0,
            code_block_height_exp: 0,
            ..EncodeOptions::default()
        };
        encode(&pixels, 8, 8, 1, 1, false, &options).expect("encode classic gray1 multi-block")
    }

    fn fixture_j2k_gray8_tall() -> Vec<u8> {
        let pixels: Vec<u8> = (0u16..32u16)
            .map(|value| u8::try_from((value * 17) & 0xFF).expect("fixture sample fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            ..EncodeOptions::default()
        };
        encode(&pixels, 4, 8, 1, 8, false, &options).expect("encode classic tall gray8")
    }

    fn fixture_j2k_gray16_bypass() -> Vec<u8> {
        let pixels: Vec<u8> = (0u16..16u16)
            .flat_map(|index| ((index * 0x111) | 0x123).to_le_bytes())
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            ..EncodeOptions::default()
        };
        encode(&pixels, 4, 4, 1, 16, false, &options).expect("encode classic gray16 bypass")
    }

    fn integral_coefficients(values: &[f32]) -> Vec<i32> {
        values
            .iter()
            .map(|value| {
                assert!(value.is_finite(), "coefficient must be finite");
                let rounded = value.round();
                assert!(
                    (rounded - *value).abs() <= f32::EPSILON,
                    "reversible classic coefficients must be integral"
                );
                format!("{rounded:.0}")
                    .parse::<i32>()
                    .expect("integral coefficient fits in i32")
            })
            .collect()
    }

    #[test]
    fn metal_classic_decoder_matches_native_decode() {
        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");

        let mut baseline_context = DecoderContext::default();
        let baseline = image
            .decode_components_with_context(&mut baseline_context)
            .expect("baseline decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalClassicBlockDecoder::default();
        let hooked = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        assert!(
            decoder.blocks_decoded() > 0,
            "classic J2K hook must be used"
        );
        #[cfg(target_os = "macos")]
        assert!(decoder.kernel_dispatches() > 0);
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
    fn metal_classic_decoder_batches_multi_block_subbands() {
        let bytes = fixture_j2k_gray1_cleanup_only_multi_block();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");

        let mut baseline_context = DecoderContext::default();
        let baseline = image
            .decode_components_with_context(&mut baseline_context)
            .expect("baseline decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalClassicBlockDecoder::default();
        let hooked = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        assert!(
            decoder.sub_band_batches() > 0,
            "multi-block classic fixture must exercise the batched classic path"
        );
        #[cfg(target_os = "macos")]
        assert!(decoder.batched_kernel_dispatches() > 0);
        assert_eq!(hooked.dimensions(), baseline.dimensions());
        assert_eq!(hooked.planes().len(), baseline.planes().len());

        for (hooked_plane, baseline_plane) in hooked.planes().iter().zip(baseline.planes()) {
            assert_eq!(hooked_plane.bit_depth(), baseline_plane.bit_depth());
            assert_eq!(hooked_plane.samples(), baseline_plane.samples());
        }
    }

    #[test]
    fn metal_classic_decoder_matches_native_decode_for_cleanup_only_fixture() {
        let bytes = fixture_j2k_gray1_cleanup_only();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");

        let mut baseline_context = DecoderContext::default();
        let baseline = image
            .decode_components_with_context(&mut baseline_context)
            .expect("baseline decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalClassicBlockDecoder::default();
        let hooked = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        assert!(decoder.blocks_decoded() > 0);
        #[cfg(target_os = "macos")]
        assert!(decoder.kernel_dispatches() > 0);

        for (hooked_plane, baseline_plane) in hooked.planes().iter().zip(baseline.planes()) {
            assert_eq!(hooked_plane.samples(), baseline_plane.samples());
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_kernel_matches_scalar_for_captured_cleanup_job() {
        let bytes = fixture_j2k_gray1_cleanup_only();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut capture = CaptureFirstClassicJob::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut capture)
            .expect("capture classic job");

        let job = capture.first.expect("captured classic job");
        assert_eq!(job.number_of_coding_passes, 1);
        assert!(super::supports_metal_classic_kernel(&job.as_job()));

        let mut expected = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut expected).expect("scalar decode");
        let mut actual = vec![0.0f32; job.output_len()];
        compute::decode_classic_cleanup_code_block(job.as_job(), &mut actual)
            .expect("metal decode");
        assert_eq!(actual, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_kernel_matches_scalar_for_cleanup_job_with_segmentation_symbols() {
        let bytes = fixture_j2k_gray1_cleanup_only();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut capture = CaptureFirstClassicJob::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut capture)
            .expect("capture classic job");

        let mut job = capture.first.expect("captured classic job");
        assert_eq!(job.number_of_coding_passes, 1);
        let mut original_coefficients = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut original_coefficients)
            .expect("decode original cleanup job");
        let original_coefficients = integral_coefficients(&original_coefficients);
        let encoded = signinum_j2k_native::encode_j2k_code_block_scalar_with_style(
            &original_coefficients,
            job.width,
            job.height,
            job.sub_band_type,
            job.total_bitplanes,
            signinum_j2k_native::J2kCodeBlockStyle {
                segmentation_symbols: true,
                ..job.style
            },
        )
        .expect("encode segmentation-symbol cleanup job");
        job.style.segmentation_symbols = true;
        job.data = encoded.data;
        job.segments = encoded.segments;
        job.number_of_coding_passes = encoded.number_of_coding_passes;
        job.missing_bit_planes = encoded.missing_bit_planes;

        assert!(super::supports_metal_classic_kernel(&job.as_job()));

        let mut expected = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut expected).expect("scalar decode");
        let mut actual = vec![0.0f32; job.output_len()];
        compute::decode_classic_cleanup_code_block(job.as_job(), &mut actual)
            .expect("metal decode");
        assert_eq!(actual, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_kernel_matches_scalar_for_captured_multi_pass_job() {
        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut capture = CaptureFirstClassicJob::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut capture)
            .expect("capture classic job");

        let job = capture.first.expect("captured classic job");
        assert!(job.number_of_coding_passes > 1);
        assert!(super::supports_metal_classic_kernel(&job.as_job()));

        let mut expected = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut expected).expect("scalar decode");
        let mut actual = vec![0.0f32; job.output_len()];
        compute::decode_classic_cleanup_code_block(job.as_job(), &mut actual)
            .expect("metal decode");
        assert_eq!(actual, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_kernel_matches_scalar_for_multi_pass_job_with_reset_contexts() {
        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut capture = CaptureFirstClassicJob::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut capture)
            .expect("capture classic job");

        let mut job = capture.first.expect("captured classic job");
        assert!(job.number_of_coding_passes > 1);
        let mut original_coefficients = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut original_coefficients)
            .expect("decode original multi-pass job");
        let original_coefficients = integral_coefficients(&original_coefficients);
        let encoded = signinum_j2k_native::encode_j2k_code_block_scalar_with_style(
            &original_coefficients,
            job.width,
            job.height,
            job.sub_band_type,
            job.total_bitplanes,
            signinum_j2k_native::J2kCodeBlockStyle {
                reset_context_probabilities: true,
                ..job.style
            },
        )
        .expect("encode reset-context classic job");
        job.style.reset_context_probabilities = true;
        job.data = encoded.data;
        job.segments = encoded.segments;
        job.number_of_coding_passes = encoded.number_of_coding_passes;
        job.missing_bit_planes = encoded.missing_bit_planes;

        assert!(super::supports_metal_classic_kernel(&job.as_job()));

        let mut expected = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut expected).expect("scalar decode");
        let mut actual = vec![0.0f32; job.output_len()];
        compute::decode_classic_cleanup_code_block(job.as_job(), &mut actual)
            .expect("metal decode");
        assert_eq!(actual, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_kernel_matches_scalar_for_tall_job_with_vertically_causal_context() {
        let bytes = fixture_j2k_gray8_tall();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut capture = CaptureFirstClassicJob::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut capture)
            .expect("capture classic job");

        let mut job = capture.first.expect("captured classic job");
        assert!(job.height > 4, "fixture must cross a stripe boundary");
        let mut original_coefficients = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut original_coefficients)
            .expect("decode original tall job");
        let original_coefficients = integral_coefficients(&original_coefficients);
        let encoded = signinum_j2k_native::encode_j2k_code_block_scalar_with_style(
            &original_coefficients,
            job.width,
            job.height,
            job.sub_band_type,
            job.total_bitplanes,
            signinum_j2k_native::J2kCodeBlockStyle {
                vertically_causal_context: true,
                ..job.style
            },
        )
        .expect("encode vertically-causal classic job");
        job.style.vertically_causal_context = true;
        job.data = encoded.data;
        job.segments = encoded.segments;
        job.number_of_coding_passes = encoded.number_of_coding_passes;
        job.missing_bit_planes = encoded.missing_bit_planes;

        assert!(super::supports_metal_classic_kernel(&job.as_job()));

        let mut expected = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut expected).expect("scalar decode");
        let mut actual = vec![0.0f32; job.output_len()];
        compute::decode_classic_cleanup_code_block(job.as_job(), &mut actual)
            .expect("metal decode");
        assert_eq!(actual, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_kernel_matches_scalar_for_multi_pass_job_with_termination_on_each_pass() {
        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut capture = CaptureFirstClassicJob::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut capture)
            .expect("capture classic job");

        let mut job = capture.first.expect("captured classic job");
        assert!(job.number_of_coding_passes > 1);
        let mut original_coefficients = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut original_coefficients)
            .expect("decode original multi-pass job");
        let original_coefficients = integral_coefficients(&original_coefficients);
        let encoded = signinum_j2k_native::encode_j2k_code_block_scalar_with_style(
            &original_coefficients,
            job.width,
            job.height,
            job.sub_band_type,
            job.total_bitplanes,
            signinum_j2k_native::J2kCodeBlockStyle {
                termination_on_each_pass: true,
                ..job.style
            },
        )
        .expect("encode terminated classic job");
        job.style.termination_on_each_pass = true;
        job.data = encoded.data;
        job.segments = encoded.segments;
        job.number_of_coding_passes = encoded.number_of_coding_passes;
        job.missing_bit_planes = encoded.missing_bit_planes;

        assert!(
            job.segments.len() > 1,
            "termination must produce multiple segments"
        );
        assert!(super::supports_metal_classic_kernel(&job.as_job()));

        let mut expected = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut expected).expect("scalar decode");
        let mut actual = vec![0.0f32; job.output_len()];
        compute::decode_classic_cleanup_code_block(job.as_job(), &mut actual)
            .expect("metal decode");
        assert_eq!(actual, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_kernel_matches_scalar_for_bypass_job() {
        let bytes = fixture_j2k_gray16_bypass();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut capture = CaptureFirstClassicJob::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut capture)
            .expect("capture classic job");

        let mut job = capture.first.expect("captured classic job");
        assert!(
            job.number_of_coding_passes > 10,
            "fixture must reach bypass-eligible coding passes"
        );
        let mut original_coefficients = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut original_coefficients)
            .expect("decode original bypass job");
        let original_coefficients = integral_coefficients(&original_coefficients);
        let encoded = signinum_j2k_native::encode_j2k_code_block_scalar_with_style(
            &original_coefficients,
            job.width,
            job.height,
            job.sub_band_type,
            job.total_bitplanes,
            signinum_j2k_native::J2kCodeBlockStyle {
                selective_arithmetic_coding_bypass: true,
                ..job.style
            },
        )
        .expect("encode bypass classic job");
        job.style.selective_arithmetic_coding_bypass = true;
        job.data = encoded.data;
        job.segments = encoded.segments;
        job.number_of_coding_passes = encoded.number_of_coding_passes;
        job.missing_bit_planes = encoded.missing_bit_planes;

        assert!(
            job.segments.iter().any(|segment| !segment.use_arithmetic),
            "bypass job must contain non-arithmetic segments"
        );
        assert!(super::supports_metal_classic_kernel(&job.as_job()));

        let mut expected = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut expected).expect("scalar decode");
        let mut actual = vec![0.0f32; job.output_len()];
        compute::decode_classic_cleanup_code_block(job.as_job(), &mut actual)
            .expect("metal decode");
        assert_eq!(actual, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_kernel_accepts_zero_length_prefix_segment() {
        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut capture = CaptureFirstClassicJob::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut capture)
            .expect("capture classic job");

        let mut job = capture.first.expect("captured classic job");
        let real = job.segments[0];
        job.segments = vec![
            J2kCodeBlockSegment {
                data_offset: 0,
                data_length: 0,
                start_coding_pass: 0,
                end_coding_pass: 0,
                use_arithmetic: true,
            },
            real,
        ];

        assert!(super::supports_metal_classic_kernel(&job.as_job()));

        let mut expected = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut expected).expect("scalar decode");
        let mut actual = vec![0.0f32; job.output_len()];
        compute::decode_classic_cleanup_code_block(job.as_job(), &mut actual)
            .expect("metal decode");
        assert_eq!(actual, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_kernel_matches_scalar_for_valid_two_segment_job() {
        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut capture = CaptureFirstClassicJob::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut capture)
            .expect("capture classic job");

        let captured = capture.first.expect("captured classic job");
        let job = split_job_into_two_valid_segments(&captured);
        assert!(job.segments.len() > 1);
        assert!(super::supports_metal_classic_kernel(&job.as_job()));

        let mut expected = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut expected).expect("scalar decode");
        let mut actual = vec![0.0f32; job.output_len()];
        compute::decode_classic_cleanup_code_block(job.as_job(), &mut actual)
            .expect("metal decode");
        assert_eq!(actual, expected);
    }
}
