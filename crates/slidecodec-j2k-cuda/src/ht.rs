// SPDX-License-Identifier: Apache-2.0

use slidecodec_j2k_native::{
    decode_ht_code_block_scalar, decode_j2k_code_block_scalar, decode_j2k_sub_band_scalar,
    HtCodeBlockDecodeJob, HtCodeBlockDecoder, J2kCodeBlockDecodeJob, J2kSubBandDecodeJob, Result,
};

#[derive(Default)]
pub(crate) struct CudaHtBlockDecoder {
    blocks_decoded: usize,
    sub_band_batches: usize,
}

impl CudaHtBlockDecoder {
    #[cfg(test)]
    pub(crate) fn blocks_decoded(&self) -> usize {
        self.blocks_decoded
    }

    #[cfg(test)]
    pub(crate) fn sub_band_batches(&self) -> usize {
        self.sub_band_batches
    }
}

impl HtCodeBlockDecoder for CudaHtBlockDecoder {
    fn decode_j2k_sub_band(
        &mut self,
        job: J2kSubBandDecodeJob<'_>,
        output: &mut [f32],
    ) -> Result<bool> {
        if job.jobs.len() <= 1 {
            return Ok(false);
        }

        self.sub_band_batches = self.sub_band_batches.saturating_add(1);
        decode_j2k_sub_band_scalar(job, output)?;
        Ok(true)
    }

    fn decode_j2k_code_block(
        &mut self,
        job: J2kCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> Result<bool> {
        self.blocks_decoded = self.blocks_decoded.saturating_add(1);
        decode_j2k_code_block_scalar(job, output)?;
        Ok(true)
    }

    fn decode_code_block(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> Result<()> {
        self.blocks_decoded = self.blocks_decoded.saturating_add(1);
        decode_ht_code_block_scalar(job, output)
    }
}

#[cfg(test)]
mod tests {
    use super::CudaHtBlockDecoder;
    use slidecodec_j2k_native::{
        encode, encode_htj2k, DecodeSettings, DecoderContext, EncodeOptions, Image,
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

    fn fixture_j2k_gray8_multi_block() -> Vec<u8> {
        let pixels: Vec<u8> = (0..64).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            code_block_width_exp: 0,
            code_block_height_exp: 0,
            ..EncodeOptions::default()
        };
        encode(&pixels, 8, 8, 1, 8, false, &options).expect("encode multi-block classic gray8")
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

    #[test]
    fn cuda_ht_decoder_matches_native_decode() {
        let bytes = fixture_ht_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");

        let mut baseline_context = DecoderContext::default();
        let baseline = image
            .decode_components_with_context(&mut baseline_context)
            .expect("baseline decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = CudaHtBlockDecoder::default();
        let hooked = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        assert!(
            decoder.blocks_decoded() > 0,
            "HT codeblock hook must be used"
        );
        assert_eq!(hooked.dimensions(), baseline.dimensions());
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
    fn cuda_classic_decoder_matches_native_decode() {
        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");

        let mut baseline_context = DecoderContext::default();
        let baseline = image
            .decode_components_with_context(&mut baseline_context)
            .expect("baseline decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = CudaHtBlockDecoder::default();
        let hooked = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        assert!(
            decoder.blocks_decoded() > 0,
            "classic codeblock hook must be used"
        );
        assert_eq!(hooked.dimensions(), baseline.dimensions());
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
    fn cuda_classic_decoder_batches_multi_block_subbands() {
        let bytes = fixture_j2k_gray8_multi_block();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");

        let mut baseline_context = DecoderContext::default();
        let baseline = image
            .decode_components_with_context(&mut baseline_context)
            .expect("baseline decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = CudaHtBlockDecoder::default();
        let hooked = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        assert!(
            decoder.sub_band_batches() > 0,
            "multi-block classic fixture must exercise the batched classic path"
        );
        assert_eq!(hooked.dimensions(), baseline.dimensions());
        assert_eq!(hooked.planes().len(), baseline.planes().len());

        for (hooked_plane, baseline_plane) in hooked.planes().iter().zip(baseline.planes()) {
            assert_eq!(hooked_plane.bit_depth(), baseline_plane.bit_depth());
            assert_eq!(hooked_plane.samples(), baseline_plane.samples());
        }
    }
}
