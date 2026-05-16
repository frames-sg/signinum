// SPDX-License-Identifier: Apache-2.0

#[cfg(all(target_os = "macos", test))]
use signinum_j2k_native::{
    HtCodeBlockBatchJob, HtCodeBlockDecodeJob, HtOwnedCodeBlockBatchJob, HtOwnedSubBandPlan,
    J2kCodeBlockBatchJob, J2kCodeBlockDecodeJob, J2kDirectBandId, J2kDirectGrayscalePlan,
    J2kDirectGrayscaleStep, J2kIdwtBand, J2kOwnedCodeBlockBatchJob, J2kOwnedSubBandPlan, J2kRect,
    J2kSingleDecompositionIdwtJob,
};

#[cfg(all(target_os = "macos", test))]
use crate::compute;
#[cfg(all(target_os = "macos", test))]
use crate::Error;

#[cfg(target_os = "macos")]
#[cfg(test)]
#[derive(Default)]
struct DirectExecutionState {
    bands: Vec<(J2kDirectBandId, Vec<f32>)>,
    captured_plane_storage: Option<Vec<f32>>,
}

#[cfg(target_os = "macos")]
#[cfg(test)]
struct ExecutedGrayscalePlane {
    storage: Vec<f32>,
}

#[cfg(target_os = "macos")]
#[cfg(test)]
impl DirectExecutionState {
    fn insert_band(&mut self, band_id: J2kDirectBandId, coefficients: Vec<f32>) {
        self.bands.retain(|(existing, _)| *existing != band_id);
        self.bands.push((band_id, coefficients));
    }

    fn band(&self, band_id: J2kDirectBandId, rect: J2kRect) -> Result<&[f32], Error> {
        self.bands
            .iter()
            .find(|(existing, _)| *existing == band_id)
            .map(|(_, coefficients)| coefficients.as_slice())
            .ok_or_else(|| Error::MetalKernel {
                message: format!(
                    "missing J2K MetalDirect coefficients for band {} rect ({}, {}, {}, {})",
                    band_id, rect.x0, rect.y0, rect.x1, rect.y1
                ),
            })
    }
}

#[cfg(target_os = "macos")]
#[cfg(test)]
fn execute_grayscale_plan_to_plane(
    plan: &J2kDirectGrayscalePlan,
) -> Result<ExecutedGrayscalePlane, Error> {
    let mut state = DirectExecutionState::default();

    for step in &plan.steps {
        match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                let mut output = vec![0.0f32; sub_band.width as usize * sub_band.height as usize];
                decode_classic_sub_band(sub_band, &mut output)?;
                state.insert_band(sub_band.band_id, output);
            }
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                let mut output = vec![0.0f32; sub_band.width as usize * sub_band.height as usize];
                decode_ht_sub_band(sub_band, &mut output)?;
                state.insert_band(sub_band.band_id, output);
            }
            J2kDirectGrayscaleStep::Idwt(idwt) => {
                let ll = state.band(idwt.ll_band_id, idwt.ll)?;
                let hl = state.band(idwt.hl_band_id, idwt.hl)?;
                let lh = state.band(idwt.lh_band_id, idwt.lh)?;
                let hh = state.band(idwt.hh_band_id, idwt.hh)?;
                let mut output =
                    vec![0.0f32; idwt.rect.width() as usize * idwt.rect.height() as usize];
                let job = J2kSingleDecompositionIdwtJob {
                    rect: idwt.rect,
                    transform: idwt.transform,
                    ll: J2kIdwtBand {
                        rect: idwt.ll,
                        coefficients: ll,
                    },
                    hl: J2kIdwtBand {
                        rect: idwt.hl,
                        coefficients: hl,
                    },
                    lh: J2kIdwtBand {
                        rect: idwt.lh,
                        coefficients: lh,
                    },
                    hh: J2kIdwtBand {
                        rect: idwt.hh,
                        coefficients: hh,
                    },
                };
                match idwt.transform {
                    signinum_j2k_native::J2kWaveletTransform::Reversible53 => {
                        compute::decode_reversible53_single_decomposition_idwt(job, &mut output)?;
                    }
                    signinum_j2k_native::J2kWaveletTransform::Irreversible97 => {
                        compute::decode_irreversible97_single_decomposition_idwt(job, &mut output)?;
                    }
                }
                state.insert_band(idwt.output_band_id, output);
            }
            J2kDirectGrayscaleStep::Store(store) => {
                let input = state.band(store.input_band_id, store.input_rect)?;
                let mut output =
                    vec![0.0f32; store.output_width as usize * store.output_height as usize];
                let job = signinum_j2k_native::J2kStoreComponentJob {
                    input,
                    input_width: store.input_rect.width(),
                    source_x: store.source_x,
                    source_y: store.source_y,
                    copy_width: store.copy_width,
                    copy_height: store.copy_height,
                    output: &mut output,
                    output_width: store.output_width,
                    output_x: store.output_x,
                    output_y: store.output_y,
                    addend: store.addend,
                };
                let _captured_plane = compute::decode_store_component_and_capture(job)?;
                state.captured_plane_storage = Some(output);
            }
        }
    }

    let storage = state
        .captured_plane_storage
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect grayscale plan did not retain host plane storage".to_string(),
        })?;
    Ok(ExecutedGrayscalePlane { storage })
}

#[cfg(target_os = "macos")]
#[cfg(test)]
fn decode_classic_sub_band(plan: &J2kOwnedSubBandPlan, output: &mut [f32]) -> Result<(), Error> {
    if let [block] = plan.jobs.as_slice() {
        let start = block.output_y as usize * plan.width as usize + block.output_x as usize;
        return compute::decode_classic_cleanup_code_block(
            classic_job(block),
            &mut output[start..],
        );
    }

    let jobs: Vec<_> = plan
        .jobs
        .iter()
        .map(|owned| J2kCodeBlockBatchJob {
            output_x: owned.output_x,
            output_y: owned.output_y,
            code_block: classic_job(owned),
        })
        .collect();
    compute::decode_classic_cleanup_sub_band(
        signinum_j2k_native::J2kSubBandDecodeJob {
            width: plan.width,
            height: plan.height,
            jobs: &jobs,
        },
        output,
    )
}

#[cfg(target_os = "macos")]
#[cfg(test)]
fn decode_ht_sub_band(plan: &HtOwnedSubBandPlan, output: &mut [f32]) -> Result<(), Error> {
    if let [block] = plan.jobs.as_slice() {
        let start = block.output_y as usize * plan.width as usize + block.output_x as usize;
        return compute::decode_ht_cleanup_code_block(ht_job(block), &mut output[start..]);
    }

    let jobs: Vec<_> = plan
        .jobs
        .iter()
        .map(|owned| HtCodeBlockBatchJob {
            output_x: owned.output_x,
            output_y: owned.output_y,
            code_block: ht_job(owned),
        })
        .collect();
    compute::decode_ht_cleanup_sub_band(
        signinum_j2k_native::HtSubBandDecodeJob {
            width: plan.width,
            height: plan.height,
            jobs: &jobs,
        },
        output,
    )
}

#[cfg(target_os = "macos")]
#[cfg(test)]
fn classic_job(owned: &J2kOwnedCodeBlockBatchJob) -> J2kCodeBlockDecodeJob<'_> {
    J2kCodeBlockDecodeJob {
        data: &owned.data,
        segments: &owned.segments,
        width: owned.width,
        height: owned.height,
        output_stride: owned.output_stride,
        missing_bit_planes: owned.missing_bit_planes,
        number_of_coding_passes: owned.number_of_coding_passes,
        total_bitplanes: owned.total_bitplanes,
        sub_band_type: owned.sub_band_type,
        style: owned.style,
        strict: owned.strict,
        dequantization_step: owned.dequantization_step,
    }
}

#[cfg(target_os = "macos")]
#[cfg(test)]
fn ht_job(owned: &HtOwnedCodeBlockBatchJob) -> HtCodeBlockDecodeJob<'_> {
    HtCodeBlockDecodeJob {
        data: &owned.data,
        cleanup_length: owned.cleanup_length,
        refinement_length: owned.refinement_length,
        width: owned.width,
        height: owned.height,
        output_stride: owned.output_stride,
        missing_bit_planes: owned.missing_bit_planes,
        number_of_coding_passes: owned.number_of_coding_passes,
        num_bitplanes: owned.num_bitplanes,
        stripe_causal: owned.stripe_causal,
        strict: owned.strict,
        dequantization_step: owned.dequantization_step,
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn is_unsupported_direct_plan_error(message: &str) -> bool {
    message.contains("direct grayscale plan only supports")
        || message.contains("direct color plan only supports")
        || message.contains("direct component plan only supports")
        || message.contains("UnsupportedColorSpace")
        || message.contains("Unsupported color space")
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::{
        decode_classic_sub_band, decode_ht_sub_band, execute_grayscale_plan_to_plane,
        DirectExecutionState,
    };
    use signinum_j2k_native::{
        encode, encode_htj2k, DecodeSettings, DecoderContext, EncodeOptions, HtCodeBlockDecoder,
        Image, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kSingleDecompositionIdwtJob,
    };

    fn classic_plan() -> signinum_j2k_native::J2kDirectGrayscalePlan {
        let pixels: Vec<u8> = (0..16).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let bytes = encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode classic gray8");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("direct plan")
    }

    fn ht_plan() -> signinum_j2k_native::J2kDirectGrayscalePlan {
        let pixels: Vec<u8> = (0..16).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let bytes = encode_htj2k(&pixels, 4, 4, 1, 8, false, &options).expect("encode ht gray8");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("direct plan")
    }

    fn execute_to_pre_store_band(plan: &J2kDirectGrayscalePlan) -> Vec<f32> {
        let mut state = DirectExecutionState::default();
        let mut target_rect = None;
        for step in &plan.steps {
            match step {
                J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                    let mut output =
                        vec![0.0f32; sub_band.width as usize * sub_band.height as usize];
                    decode_classic_sub_band(sub_band, &mut output).expect("decode classic");
                    state.insert_band(sub_band.band_id, output);
                }
                J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                    let mut output =
                        vec![0.0f32; sub_band.width as usize * sub_band.height as usize];
                    decode_ht_sub_band(sub_band, &mut output).expect("decode ht");
                    state.insert_band(sub_band.band_id, output);
                }
                J2kDirectGrayscaleStep::Idwt(idwt) => {
                    let ll = state.band(idwt.ll_band_id, idwt.ll).expect("ll");
                    let hl = state.band(idwt.hl_band_id, idwt.hl).expect("hl");
                    let lh = state.band(idwt.lh_band_id, idwt.lh).expect("lh");
                    let hh = state.band(idwt.hh_band_id, idwt.hh).expect("hh");
                    let mut output =
                        vec![0.0f32; idwt.rect.width() as usize * idwt.rect.height() as usize];
                    let job = signinum_j2k_native::J2kSingleDecompositionIdwtJob {
                        rect: idwt.rect,
                        transform: idwt.transform,
                        ll: signinum_j2k_native::J2kIdwtBand {
                            rect: idwt.ll,
                            coefficients: ll,
                        },
                        hl: signinum_j2k_native::J2kIdwtBand {
                            rect: idwt.hl,
                            coefficients: hl,
                        },
                        lh: signinum_j2k_native::J2kIdwtBand {
                            rect: idwt.lh,
                            coefficients: lh,
                        },
                        hh: signinum_j2k_native::J2kIdwtBand {
                            rect: idwt.hh,
                            coefficients: hh,
                        },
                    };
                    match idwt.transform {
                        signinum_j2k_native::J2kWaveletTransform::Reversible53 => {
                            crate::compute::decode_reversible53_single_decomposition_idwt(
                                job,
                                &mut output,
                            )
                            .expect("53 idwt");
                        }
                        signinum_j2k_native::J2kWaveletTransform::Irreversible97 => {
                            crate::compute::decode_irreversible97_single_decomposition_idwt(
                                job,
                                &mut output,
                            )
                            .expect("97 idwt");
                        }
                    }
                    state.insert_band(idwt.output_band_id, output);
                }
                J2kDirectGrayscaleStep::Store(store) => {
                    target_rect = Some((store.input_band_id, store.input_rect));
                    break;
                }
            }
        }
        let (band_id, rect) = target_rect.expect("store rect");
        state.band(band_id, rect).expect("pre-store band").to_vec()
    }

    fn execute_to_first_idwt_inputs(
        plan: &J2kDirectGrayscalePlan,
    ) -> (
        signinum_j2k_native::J2kDirectIdwtStep,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    ) {
        let mut state = DirectExecutionState::default();
        for step in &plan.steps {
            match step {
                J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                    let mut output =
                        vec![0.0f32; sub_band.width as usize * sub_band.height as usize];
                    decode_classic_sub_band(sub_band, &mut output).expect("decode classic");
                    state.insert_band(sub_band.band_id, output);
                }
                J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                    let mut output =
                        vec![0.0f32; sub_band.width as usize * sub_band.height as usize];
                    decode_ht_sub_band(sub_band, &mut output).expect("decode ht");
                    state.insert_band(sub_band.band_id, output);
                }
                J2kDirectGrayscaleStep::Idwt(idwt) => {
                    return (
                        *idwt,
                        state.band(idwt.ll_band_id, idwt.ll).expect("ll").to_vec(),
                        state.band(idwt.hl_band_id, idwt.hl).expect("hl").to_vec(),
                        state.band(idwt.lh_band_id, idwt.lh).expect("lh").to_vec(),
                        state.band(idwt.hh_band_id, idwt.hh).expect("hh").to_vec(),
                    );
                }
                J2kDirectGrayscaleStep::Store(_) => break,
            }
        }
        panic!("plan did not contain an IDWT step")
    }

    #[derive(Default)]
    struct CaptureIdwtJob {
        rect: Option<signinum_j2k_native::J2kRect>,
        ll_rect: Option<signinum_j2k_native::J2kRect>,
        hl_rect: Option<signinum_j2k_native::J2kRect>,
        lh_rect: Option<signinum_j2k_native::J2kRect>,
        hh_rect: Option<signinum_j2k_native::J2kRect>,
        ll: Vec<f32>,
        hl: Vec<f32>,
        lh: Vec<f32>,
        hh: Vec<f32>,
    }

    impl HtCodeBlockDecoder for CaptureIdwtJob {
        fn decode_code_block(
            &mut self,
            job: signinum_j2k_native::HtCodeBlockDecodeJob<'_>,
            output: &mut [f32],
        ) -> signinum_j2k_native::Result<()> {
            signinum_j2k_native::decode_ht_code_block_scalar(job, output)
        }

        fn decode_single_decomposition_idwt(
            &mut self,
            job: J2kSingleDecompositionIdwtJob<'_>,
            _output: &mut [f32],
        ) -> signinum_j2k_native::Result<bool> {
            if self.rect.is_none() {
                self.rect = Some(job.rect);
                self.ll_rect = Some(job.ll.rect);
                self.hl_rect = Some(job.hl.rect);
                self.lh_rect = Some(job.lh.rect);
                self.hh_rect = Some(job.hh.rect);
                self.ll = job.ll.coefficients.to_vec();
                self.hl = job.hl.coefficients.to_vec();
                self.lh = job.lh.coefficients.to_vec();
                self.hh = job.hh.coefficients.to_vec();
            }
            Ok(false)
        }
    }

    #[test]
    fn classic_direct_plan_sub_band_decode_produces_nonzero_coefficients() {
        let plan = classic_plan();
        let sub_band = plan
            .steps
            .iter()
            .find_map(|step| match step {
                J2kDirectGrayscaleStep::ClassicSubBand(plan) => Some(plan.clone()),
                _ => None,
            })
            .expect("classic sub-band step");
        let mut output = vec![0.0f32; sub_band.width as usize * sub_band.height as usize];
        decode_classic_sub_band(&sub_band, &mut output).expect("decode classic sub-band");
        assert!(
            output.iter().any(|sample| *sample != 0.0),
            "classic direct sub-band decode must produce nonzero coefficients for the fixture"
        );
    }

    #[test]
    fn ht_direct_plan_sub_band_decode_produces_nonzero_coefficients() {
        let plan = ht_plan();
        let sub_band = plan
            .steps
            .iter()
            .find_map(|step| match step {
                J2kDirectGrayscaleStep::HtSubBand(plan) => Some(plan.clone()),
                _ => None,
            })
            .expect("ht sub-band step");
        let mut output = vec![0.0f32; sub_band.width as usize * sub_band.height as usize];
        decode_ht_sub_band(&sub_band, &mut output).expect("decode ht sub-band");
        assert!(
            output.iter().any(|sample| *sample != 0.0),
            "HT direct sub-band decode must produce nonzero coefficients for the fixture"
        );
    }

    #[test]
    fn classic_direct_plan_store_plane_matches_native_decode() {
        let plan = classic_plan();
        let executed = execute_grayscale_plan_to_plane(&plan).expect("execute direct plan");

        let pixels: Vec<u8> = (0..16).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let bytes = encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode classic gray8");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let decoded = image
            .decode_components_with_context(&mut context)
            .expect("native decode");

        assert_eq!(decoded.planes().len(), 1);
        assert_eq!(
            executed.storage,
            decoded.planes()[0].samples(),
            "direct grayscale host plane must match native grayscale decode before final pack"
        );
    }

    #[test]
    fn classic_direct_plan_pre_store_band_is_not_all_zero() {
        let plan = classic_plan();
        let band = execute_to_pre_store_band(&plan);
        assert!(
            band.iter().any(|sample| *sample != 0.0),
            "direct grayscale pre-store band must not collapse to all zeros"
        );
    }

    #[test]
    fn classic_direct_plan_idwt_inputs_match_native_backend_job() {
        let plan = classic_plan();
        let (idwt, ll, hl, lh, hh) = execute_to_first_idwt_inputs(&plan);

        let pixels: Vec<u8> = (0..16).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let bytes = encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode classic gray8");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut capture = CaptureIdwtJob::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut capture)
            .expect("capture native idwt job");

        assert_eq!(capture.rect, Some(idwt.rect));
        assert_eq!(capture.ll_rect, Some(idwt.ll));
        assert_eq!(capture.hl_rect, Some(idwt.hl));
        assert_eq!(capture.lh_rect, Some(idwt.lh));
        assert_eq!(capture.hh_rect, Some(idwt.hh));
        assert_eq!(capture.ll, ll);
        assert_eq!(capture.hl, hl);
        assert_eq!(capture.lh, lh);
        assert_eq!(capture.hh, hh);
    }
}
