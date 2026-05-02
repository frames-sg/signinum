// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use crate::compute;
use signinum_j2k_native::{
    EncodedJ2kCodeBlock, J2kEncodeStageAccelerator, J2kForwardDwt53Job, J2kForwardDwt53Output,
    J2kForwardRctJob, J2kPacketizationEncodeJob, J2kTier1CodeBlockEncodeJob,
};

/// Encode-stage accelerator for JPEG 2000 Metal work.
///
/// The type is wired into the native encoder hook interface. Stages return
/// fallback until each Metal encode kernel is complete and parity validated.
#[derive(Debug, Default, Clone)]
pub struct MetalEncodeStageAccelerator {
    forward_rct_attempts: usize,
    forward_dwt53_attempts: usize,
    tier1_code_block_attempts: usize,
    packetization_attempts: usize,
    forward_rct_dispatches: usize,
    forward_dwt53_dispatches: usize,
    tier1_code_block_dispatches: usize,
    packetization_dispatches: usize,
}

impl MetalEncodeStageAccelerator {
    pub fn forward_rct_attempts(&self) -> usize {
        self.forward_rct_attempts
    }

    pub fn forward_dwt53_attempts(&self) -> usize {
        self.forward_dwt53_attempts
    }

    pub fn tier1_code_block_attempts(&self) -> usize {
        self.tier1_code_block_attempts
    }

    pub fn packetization_attempts(&self) -> usize {
        self.packetization_attempts
    }

    pub fn forward_rct_dispatches(&self) -> usize {
        self.forward_rct_dispatches
    }

    pub fn forward_dwt53_dispatches(&self) -> usize {
        self.forward_dwt53_dispatches
    }

    pub fn tier1_code_block_dispatches(&self) -> usize {
        self.tier1_code_block_dispatches
    }

    pub fn packetization_dispatches(&self) -> usize {
        self.packetization_dispatches
    }
}

impl J2kEncodeStageAccelerator for MetalEncodeStageAccelerator {
    fn encode_forward_rct(
        &mut self,
        job: J2kForwardRctJob<'_>,
    ) -> core::result::Result<bool, &'static str> {
        self.forward_rct_attempts = self.forward_rct_attempts.saturating_add(1);
        #[cfg(target_os = "macos")]
        {
            compute::encode_forward_rct(job.plane0, job.plane1, job.plane2)
                .map_err(|_| "Metal forward RCT encode kernel failed")?;
            self.forward_rct_dispatches = self.forward_rct_dispatches.saturating_add(1);
            Ok(true)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            Ok(false)
        }
    }

    fn encode_forward_dwt53(
        &mut self,
        job: J2kForwardDwt53Job<'_>,
    ) -> core::result::Result<Option<J2kForwardDwt53Output>, &'static str> {
        self.forward_dwt53_attempts = self.forward_dwt53_attempts.saturating_add(1);
        if job.num_levels == 0 {
            return Ok(None);
        }
        #[cfg(target_os = "macos")]
        {
            let output =
                compute::encode_forward_dwt53(job.samples, job.width, job.height, job.num_levels)
                    .map_err(|_| "Metal forward 5/3 DWT encode kernel failed")?;
            self.forward_dwt53_dispatches = self.forward_dwt53_dispatches.saturating_add(1);
            Ok(Some(output))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            Ok(None)
        }
    }

    fn encode_tier1_code_block(
        &mut self,
        _job: J2kTier1CodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<EncodedJ2kCodeBlock>, &'static str> {
        self.tier1_code_block_attempts = self.tier1_code_block_attempts.saturating_add(1);
        Ok(None)
    }

    fn encode_packetization(
        &mut self,
        _job: J2kPacketizationEncodeJob,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        self.packetization_attempts = self.packetization_attempts.saturating_add(1);
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::MetalEncodeStageAccelerator;
    use signinum_j2k_native::{
        encode_with_accelerator, DecodeSettings, EncodeOptions, Image, J2kEncodeStageAccelerator,
        J2kForwardDwt53Job,
    };

    #[test]
    fn metal_encode_stage_accelerator_preserves_cpu_codestream_validity() {
        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| (i & 0xFF) as u8).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 8, 8, 3, 8, false, &options, &mut accelerator)
                .expect("encode with metal stage accelerator");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.num_components, 3);
        assert_eq!(decoded.bit_depth, 8);
        assert_eq!(accelerator.forward_rct_attempts(), 1);
        assert_eq!(accelerator.forward_dwt53_attempts(), 3);
        assert!(accelerator.tier1_code_block_attempts() > 0);
        assert_eq!(accelerator.packetization_attempts(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_forward_rct_dispatch_round_trips_rgb8_lossless_tile() {
        let pixels: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 17) & 0xFF) as u8).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            ..EncodeOptions::default()
        };
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 7, 5, 3, 8, false, &options, &mut accelerator)
                .expect("encode with metal forward RCT");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(accelerator.forward_rct_attempts(), 1);
        assert_eq!(accelerator.forward_rct_dispatches(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_forward_dwt53_dispatch_round_trips_gray8_lossless_tile() {
        let pixels: Vec<u8> = (0..8 * 8).map(|i| ((i * 5) & 0xFF) as u8).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 8, 8, 1, 8, false, &options, &mut accelerator)
                .expect("encode with metal forward DWT 5/3");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(accelerator.forward_dwt53_attempts(), 1);
        assert_eq!(accelerator.forward_dwt53_dispatches(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_forward_dwt53_handles_single_sample_edge_dimensions() {
        for (width, height) in [(1, 8), (8, 1)] {
            let samples: Vec<f32> = (0..width * height)
                .map(|i| {
                    f32::from(
                        u8::try_from((i * 11 + width * 3 + height * 5) & 0xFF)
                            .expect("masked sample fits in u8"),
                    ) - 128.0
                })
                .collect();
            let mut accelerator = MetalEncodeStageAccelerator::default();

            let output = accelerator
                .encode_forward_dwt53(J2kForwardDwt53Job {
                    samples: &samples,
                    width,
                    height,
                    num_levels: 1,
                })
                .expect("metal DWT 5/3 stage")
                .expect("metal DWT 5/3 dispatch");

            assert_eq!(output.ll_width, width.div_ceil(2));
            assert_eq!(output.ll_height, height.div_ceil(2));
            assert_eq!(output.levels.len(), 1);
            assert_eq!(accelerator.forward_dwt53_attempts(), 1);
            assert_eq!(accelerator.forward_dwt53_dispatches(), 1);
        }
    }
}
