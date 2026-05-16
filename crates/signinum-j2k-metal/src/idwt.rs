// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use crate::compute;
#[cfg(target_os = "macos")]
use signinum_j2k_native::J2kWaveletTransform;
use signinum_j2k_native::{
    decode_ht_code_block_scalar, HtCodeBlockDecodeJob, HtCodeBlockDecoder,
    J2kSingleDecompositionIdwtJob, Result,
};

#[derive(Default)]
pub(crate) struct MetalIdwtDecoder {
    #[cfg(target_os = "macos")]
    kernel_dispatches: usize,
}

impl MetalIdwtDecoder {
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn kernel_dispatches(&self) -> usize {
        self.kernel_dispatches
    }
}

impl HtCodeBlockDecoder for MetalIdwtDecoder {
    fn decode_single_decomposition_idwt(
        &mut self,
        job: J2kSingleDecompositionIdwtJob<'_>,
        output: &mut [f32],
    ) -> Result<bool> {
        #[cfg(target_os = "macos")]
        if supports_metal_idwt(&job) {
            match job.transform {
                J2kWaveletTransform::Reversible53 => {
                    compute::decode_reversible53_single_decomposition_idwt(job, output)
                }
                J2kWaveletTransform::Irreversible97 => {
                    compute::decode_irreversible97_single_decomposition_idwt(job, output)
                }
            }
            .map_err(|_| signinum_j2k_native::DecodingError::CodeBlockDecodeFailure)?;
            self.kernel_dispatches = self.kernel_dispatches.saturating_add(1);
            return Ok(true);
        }
        #[cfg(not(target_os = "macos"))]
        let _ = (job, output);

        Ok(false)
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
fn supports_metal_idwt(job: &J2kSingleDecompositionIdwtJob<'_>) -> bool {
    if !matches!(
        job.transform,
        J2kWaveletTransform::Reversible53 | J2kWaveletTransform::Irreversible97
    ) {
        return false;
    }
    let width = job.rect.width();
    let height = job.rect.height();
    if width == 0 || height == 0 {
        return false;
    }

    let expected_output = width as usize * height as usize;
    let expected_band_lengths = [
        job.ll.rect.width() as usize * job.ll.rect.height() as usize,
        job.hl.rect.width() as usize * job.hl.rect.height() as usize,
        job.lh.rect.width() as usize * job.lh.rect.height() as usize,
        job.hh.rect.width() as usize * job.hh.rect.height() as usize,
    ];

    expected_output > 0
        && job.ll.coefficients.len() == expected_band_lengths[0]
        && job.hl.coefficients.len() == expected_band_lengths[1]
        && job.lh.coefficients.len() == expected_band_lengths[2]
        && job.hh.coefficients.len() == expected_band_lengths[3]
}

#[cfg(test)]
mod tests {
    use super::MetalIdwtDecoder;
    use signinum_j2k_native::{
        encode, DecodeSettings, DecoderContext, EncodeOptions, HtCodeBlockDecodeJob,
        HtCodeBlockDecoder, Image,
    };

    fn fixture_j2k_gray8() -> Vec<u8> {
        let pixels: Vec<u8> = (0..16).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode classic gray8")
    }

    fn fixture_j2k_gray8_two_levels() -> Vec<u8> {
        let pixels: Vec<u8> = (0..64).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        };
        encode(&pixels, 8, 8, 1, 8, false, &options).expect("encode classic gray8 two levels")
    }

    fn fixture_j2k_gray8_irreversible() -> Vec<u8> {
        let pixels: Vec<u8> = (0..16).collect();
        let options = EncodeOptions {
            reversible: false,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode classic gray8 irreversible")
    }

    #[test]
    fn metal_idwt_decoder_matches_native_decode() {
        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut expected_context = DecoderContext::default();
        let expected = image
            .decode_components_with_context(&mut expected_context)
            .expect("native decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalIdwtDecoder::default();
        let actual = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        assert_eq!(actual.dimensions(), expected.dimensions());
        assert_eq!(actual.planes().len(), expected.planes().len());
        assert_eq!(
            actual.planes()[0].samples(),
            expected.planes()[0].samples(),
            "Metal IDWT output must match native decode"
        );
        #[cfg(target_os = "macos")]
        assert!(
            decoder.kernel_dispatches() > 0,
            "single-decomposition grayscale fixture must exercise the Metal IDWT kernel"
        );
    }

    #[test]
    fn metal_idwt_decoder_matches_native_decode_for_two_levels() {
        let bytes = fixture_j2k_gray8_two_levels();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut expected_context = DecoderContext::default();
        let expected = image
            .decode_components_with_context(&mut expected_context)
            .expect("native decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalIdwtDecoder::default();
        let actual = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        assert_eq!(actual.dimensions(), expected.dimensions());
        assert_eq!(
            actual.planes()[0].samples(),
            expected.planes()[0].samples(),
            "Metal IDWT output must match native decode for multi-level reversible images"
        );
        #[cfg(target_os = "macos")]
        assert!(
            decoder.kernel_dispatches() >= 2,
            "two-level grayscale fixture must dispatch the Metal IDWT kernel for each level"
        );
    }

    #[test]
    fn metal_idwt_decoder_matches_native_decode_for_irreversible_image() {
        let bytes = fixture_j2k_gray8_irreversible();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut expected_context = DecoderContext::default();
        let expected = image
            .decode_components_with_context(&mut expected_context)
            .expect("native decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalIdwtDecoder::default();
        let actual = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        assert_eq!(actual.dimensions(), expected.dimensions());
        assert_eq!(
            actual.planes()[0].samples(),
            expected.planes()[0].samples(),
            "Metal IDWT output must match native decode for irreversible images"
        );
        #[cfg(target_os = "macos")]
        assert!(
            decoder.kernel_dispatches() > 0,
            "irreversible grayscale fixture must exercise the Metal IDWT kernel"
        );
    }

    struct CpuOnlyCodeBlockDecoder;

    impl HtCodeBlockDecoder for CpuOnlyCodeBlockDecoder {
        fn decode_code_block(
            &mut self,
            job: HtCodeBlockDecodeJob<'_>,
            output: &mut [f32],
        ) -> signinum_j2k_native::Result<()> {
            signinum_j2k_native::decode_ht_code_block_scalar(job, output)
        }
    }

    #[test]
    fn default_decoder_without_idwt_kernel_still_decodes() {
        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut decoder = CpuOnlyCodeBlockDecoder;
        let image_components = image
            .decode_components_with_ht_decoder(&mut context, &mut decoder)
            .expect("decode without idwt override");
        assert_eq!(image_components.dimensions(), (4, 4));
    }
}
