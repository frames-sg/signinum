// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use crate::compute;
#[cfg(target_os = "macos")]
use metal::Buffer;
use signinum_j2k_native::{
    decode_ht_code_block_scalar, HtCodeBlockDecodeJob, HtCodeBlockDecoder, J2kInverseMctJob, Result,
};

#[allow(dead_code)]
#[derive(Default)]
pub(crate) struct MetalMctDecoder {
    kernel_dispatches: usize,
    #[cfg(target_os = "macos")]
    captured_planes: Vec<Buffer>,
}

#[allow(dead_code)]
impl MetalMctDecoder {
    #[cfg(test)]
    pub(crate) fn kernel_dispatches(&self) -> usize {
        self.kernel_dispatches
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn take_captured_planes(&mut self) -> Vec<Buffer> {
        core::mem::take(&mut self.captured_planes)
    }
}

impl HtCodeBlockDecoder for MetalMctDecoder {
    fn decode_inverse_mct(&mut self, job: J2kInverseMctJob<'_>) -> Result<bool> {
        #[cfg(target_os = "macos")]
        if supports_metal_inverse_mct(&job) {
            self.captured_planes = compute::decode_inverse_mct(job)
                .map_err(|_| signinum_j2k_native::DecodingError::CodeBlockDecodeFailure)?;
            self.kernel_dispatches = self.kernel_dispatches.saturating_add(1);
            return Ok(true);
        }
        #[cfg(not(target_os = "macos"))]
        let _ = job;

        Ok(false)
    }

    fn decode_code_block(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> signinum_j2k_native::Result<()> {
        decode_ht_code_block_scalar(job, output)
    }
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn supports_metal_inverse_mct(job: &J2kInverseMctJob<'_>) -> bool {
    let len = job.plane0.len();
    len > 0 && job.plane1.len() == len && job.plane2.len() == len
}

#[cfg(test)]
mod tests {
    use super::MetalMctDecoder;
    use signinum_j2k_native::{
        encode, DecodeSettings, DecoderContext, EncodeOptions, HtCodeBlockDecodeJob,
        HtCodeBlockDecoder, Image,
    };

    fn fixture_j2k_rgb8() -> Vec<u8> {
        let pixels = [10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        encode(&pixels, 2, 2, 3, 8, false, &options).expect("encode classic rgb8")
    }

    fn fixture_j2k_rgb8_irreversible() -> Vec<u8> {
        let pixels = [10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let options = EncodeOptions {
            reversible: false,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        encode(&pixels, 2, 2, 3, 8, false, &options).expect("encode irreversible rgb8")
    }

    #[test]
    fn metal_mct_decoder_matches_native_decode() {
        let bytes = fixture_j2k_rgb8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut expected_context = DecoderContext::default();
        let expected = image
            .decode_components_with_context(&mut expected_context)
            .expect("native decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalMctDecoder::default();
        let actual = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        assert_eq!(actual.dimensions(), expected.dimensions());
        assert_eq!(actual.planes().len(), expected.planes().len());
        for (actual_plane, expected_plane) in actual.planes().iter().zip(expected.planes().iter()) {
            assert_eq!(
                actual_plane.samples(),
                expected_plane.samples(),
                "Metal MCT output must match native decode"
            );
        }
        #[cfg(target_os = "macos")]
        assert!(
            decoder.kernel_dispatches() > 0,
            "RGB fixture must exercise the Metal MCT kernel"
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
    fn default_decoder_without_mct_kernel_still_decodes() {
        let bytes = fixture_j2k_rgb8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut decoder = CpuOnlyCodeBlockDecoder;
        let image_components = image
            .decode_components_with_ht_decoder(&mut context, &mut decoder)
            .expect("decode without mct override");
        assert_eq!(image_components.dimensions(), (2, 2));
    }

    #[test]
    fn metal_mct_decoder_matches_native_decode_for_irreversible_rgb() {
        let bytes = fixture_j2k_rgb8_irreversible();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut expected_context = DecoderContext::default();
        let expected = image
            .decode_components_with_context(&mut expected_context)
            .expect("native decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalMctDecoder::default();
        let actual = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        assert_eq!(actual.dimensions(), expected.dimensions());
        for (actual_plane, expected_plane) in actual.planes().iter().zip(expected.planes().iter()) {
            assert_eq!(
                actual_plane.samples(),
                expected_plane.samples(),
                "Metal MCT output must match native decode for irreversible RGB images"
            );
        }
        #[cfg(target_os = "macos")]
        assert!(
            decoder.kernel_dispatches() > 0,
            "irreversible RGB fixture must exercise the Metal MCT kernel"
        );
    }

    #[test]
    fn metal_mct_decoder_captures_final_rgb_planes_matching_host_output() {
        let bytes = fixture_j2k_rgb8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut decoder = MetalMctDecoder::default();
        let components = image
            .decode_components_with_ht_decoder(&mut context, &mut decoder)
            .expect("hooked decode");
        #[cfg(not(target_os = "macos"))]
        let _ = components;

        #[cfg(target_os = "macos")]
        {
            let captured = decoder.take_captured_planes();
            assert_eq!(captured.len(), components.planes().len());
            for (plane, buffer) in components.planes().iter().zip(captured.iter()) {
                let captured = unsafe {
                    core::slice::from_raw_parts(
                        buffer.contents().cast::<f32>(),
                        plane.samples().len(),
                    )
                };
                assert_eq!(
                    captured,
                    plane.samples(),
                    "captured Metal MCT planes must match final decoded RGB planes"
                );
            }
        }
    }
}
