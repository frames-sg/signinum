// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use crate::compute;
#[cfg(target_os = "macos")]
use metal::Buffer;
use slidecodec_j2k_native::{
    decode_ht_code_block_scalar, HtCodeBlockDecodeJob, HtCodeBlockDecoder, J2kStoreComponentJob,
    Result,
};

#[allow(dead_code)]
#[derive(Default)]
pub(crate) struct MetalStoreDecoder {
    kernel_dispatches: usize,
    #[cfg(target_os = "macos")]
    captured_planes: Vec<Buffer>,
}

#[allow(dead_code)]
impl MetalStoreDecoder {
    #[cfg(test)]
    pub(crate) fn kernel_dispatches(&self) -> usize {
        self.kernel_dispatches
    }

    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn captured_plane_count(&self) -> usize {
        self.captured_planes.len()
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn take_captured_planes(&mut self) -> Vec<Buffer> {
        core::mem::take(&mut self.captured_planes)
    }
}

impl HtCodeBlockDecoder for MetalStoreDecoder {
    fn decode_store_component(&mut self, job: J2kStoreComponentJob<'_>) -> Result<bool> {
        #[cfg(target_os = "macos")]
        if supports_metal_store(&job) {
            let captured = compute::decode_store_component_and_capture(job)
                .map_err(|_| slidecodec_j2k_native::DecodingError::CodeBlockDecodeFailure)?;
            self.captured_planes.push(captured);
            self.kernel_dispatches = self.kernel_dispatches.saturating_add(1);
            return Ok(true);
        }

        Ok(false)
    }

    fn decode_code_block(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> slidecodec_j2k_native::Result<()> {
        decode_ht_code_block_scalar(job, output)
    }
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn supports_metal_store(job: &J2kStoreComponentJob<'_>) -> bool {
    job.copy_width > 0
        && job.copy_height > 0
        && job.input_width > 0
        && job.output_width > 0
        && job.input.len() >= job.input_width as usize
        && job.output.len() >= job.output_width as usize
}

#[cfg(test)]
mod tests {
    use super::MetalStoreDecoder;
    use slidecodec_j2k_native::{
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

    #[test]
    fn metal_store_decoder_matches_native_decode() {
        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut expected_context = DecoderContext::default();
        let expected = image
            .decode_components_with_context(&mut expected_context)
            .expect("native decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalStoreDecoder::default();
        let actual = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        assert_eq!(actual.dimensions(), expected.dimensions());
        assert_eq!(
            actual.planes()[0].samples(),
            expected.planes()[0].samples(),
            "Metal store output must match native decode"
        );
        #[cfg(target_os = "macos")]
        assert!(
            decoder.kernel_dispatches() > 0,
            "grayscale fixture must exercise the Metal store kernel"
        );
    }

    struct CpuOnlyCodeBlockDecoder;

    impl HtCodeBlockDecoder for CpuOnlyCodeBlockDecoder {
        fn decode_code_block(
            &mut self,
            job: HtCodeBlockDecodeJob<'_>,
            output: &mut [f32],
        ) -> slidecodec_j2k_native::Result<()> {
            slidecodec_j2k_native::decode_ht_code_block_scalar(job, output)
        }
    }

    #[test]
    fn default_decoder_without_store_kernel_still_decodes() {
        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut decoder = CpuOnlyCodeBlockDecoder;
        let image_components = image
            .decode_components_with_ht_decoder(&mut context, &mut decoder)
            .expect("decode without store override");
        assert_eq!(image_components.dimensions(), (4, 4));
    }

    #[test]
    fn metal_store_decoder_matches_native_region_decode() {
        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let roi = (1, 1, 2, 2);
        let mut expected_context = DecoderContext::default();
        let expected = image
            .decode_region_components_with_ht_decoder(
                &mut expected_context,
                roi,
                &mut CpuOnlyCodeBlockDecoder,
            )
            .expect("native region decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalStoreDecoder::default();
        let actual = image
            .decode_region_components_with_ht_decoder(&mut hooked_context, roi, &mut decoder)
            .expect("hooked region decode");

        assert_eq!(actual.dimensions(), expected.dimensions());
        assert_eq!(
            actual.planes()[0].samples(),
            expected.planes()[0].samples(),
            "Metal region store output must match native region decode"
        );
        #[cfg(target_os = "macos")]
        assert!(
            decoder.kernel_dispatches() > 0,
            "region fixture must exercise the Metal store kernel"
        );
    }

    #[test]
    fn metal_store_decoder_captures_device_plane_for_full_decode() {
        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut decoder = MetalStoreDecoder::default();
        let _decoded = image
            .decode_components_with_ht_decoder(&mut context, &mut decoder)
            .expect("hooked decode");
        #[cfg(target_os = "macos")]
        assert_eq!(
            decoder.captured_plane_count(),
            1,
            "full grayscale decode should capture one Metal-backed plane"
        );
    }
}
