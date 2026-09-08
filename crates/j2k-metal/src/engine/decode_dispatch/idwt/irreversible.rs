// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use super::super::{
    checked_buffer_copy_into, commit_and_wait_metal, copied_slice_buffer, dispatch_2d_pipeline,
    dispatch_3d_pipeline, hybrid_stage_signpost, label_compute_encoder, new_command_buffer,
    new_compute_command_encoder, new_shared_buffer, with_runtime, Buffer, CommandBufferRef,
    ComputeCommandEncoderRef, Error, J2kIdwt97StepParams, J2kIdwtSingleDecompositionParams,
    J2kSingleDecompositionIdwtJob, SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE,
};
use super::{checked_host_output_layout, IdwtSubBandBuffers, SingleIdwtDispatch};
use j2k_codec_math::dwt;

#[cfg(test)]
use super::super::checked_buffer_slice;

pub(crate) fn decode_irreversible97_single_decomposition_idwt(
    job: J2kSingleDecompositionIdwtJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    decode_irreversible97_staged_single_decomposition_idwt(job, output)
}

pub(crate) fn decode_openjpeg_irreversible97_single_decomposition_idwt(
    job: J2kSingleDecompositionIdwtJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    decode_irreversible97_staged_single_decomposition_idwt_with_high_pass(
        job,
        output,
        dwt::IDWT97_OPENJPEG_TWO_INV_KAPPA_F32 * 0.5,
    )
}

pub(crate) fn decode_irreversible97_staged_single_decomposition_idwt(
    job: J2kSingleDecompositionIdwtJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    decode_irreversible97_staged_single_decomposition_idwt_with_high_pass(
        job,
        output,
        dwt::DWT97_INV_KAPPA_F32,
    )
}

fn decode_irreversible97_staged_single_decomposition_idwt_with_high_pass(
    job: J2kSingleDecompositionIdwtJob<'_>,
    output: &mut [f32],
    high_pass: f32,
) -> Result<(), Error> {
    with_runtime(|runtime| {
        let (required_len, required_bytes) =
            checked_host_output_layout(job.rect.width(), job.rect.height(), output.len())?;

        let params = J2kIdwtSingleDecompositionParams {
            x0: job.rect.x0,
            y0: job.rect.y0,
            output_x: 0,
            output_y: 0,
            width: job.rect.width(),
            height: job.rect.height(),
            ll_x: 0,
            ll_y: 0,
            ll_width: job.ll.rect.width(),
            ll_height: job.ll.rect.height(),
            hl_x: 0,
            hl_y: 0,
            hl_width: job.hl.rect.width(),
            hl_height: job.hl.rect.height(),
            lh_x: 0,
            lh_y: 0,
            lh_width: job.lh.rect.width(),
            lh_height: job.lh.rect.height(),
            hh_x: 0,
            hh_y: 0,
            hh_width: job.hh.rect.width(),
            hh_height: job.hh.rect.height(),
        };

        let decoded = new_shared_buffer(&runtime.device, required_bytes)?;
        let ll = copied_slice_buffer(&runtime.device, job.ll.coefficients)?;
        let hl = copied_slice_buffer(&runtime.device, job.hl.coefficients)?;
        let lh = copied_slice_buffer(&runtime.device, job.lh.coefficients)?;
        let hh = copied_slice_buffer(&runtime.device, job.hh.coefficients)?;
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command_buffer)?;
        dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_high_pass(
            &encoder,
            SingleIdwtDispatch {
                kernels: runtime.decode()?,
                sub_bands: IdwtSubBandBuffers {
                    ll: &ll,
                    ll_offset: 0,
                    hl: &hl,
                    hl_offset: 0,
                    lh: &lh,
                    lh_offset: 0,
                    hh: &hh,
                    hh_offset: 0,
                },
                params,
                decoded: &decoded,
                decoded_offset: 0,
            },
            high_pass,
        );
        encoder.endEncoding();
        commit_and_wait_metal(&command_buffer)?;

        checked_buffer_copy_into(&decoded, 0, &mut output[..required_len], "IDWT output")?;
        Ok(())
    })
}

pub(in crate::engine) fn dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
    command_buffer: &CommandBufferRef,
    dispatch: SingleIdwtDispatch<'_>,
) -> Result<(), Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE);
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K decode hybrid irreversible97 IDWT");
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
        &encoder, dispatch,
    );
    encoder.endEncoding();
    Ok(())
}

pub(in crate::engine) fn dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
    encoder: &ComputeCommandEncoderRef,
    dispatch: SingleIdwtDispatch<'_>,
) {
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_high_pass(
        encoder,
        dispatch,
        dwt::IDWT97_OPENJPEG_TWO_INV_KAPPA_F32 * 0.5,
    );
}

fn dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_high_pass(
    encoder: &ComputeCommandEncoderRef,
    dispatch: SingleIdwtDispatch<'_>,
    high_pass: f32,
) {
    let SingleIdwtDispatch {
        kernels,
        sub_bands,
        params,
        decoded,
        decoded_offset,
    } = dispatch;
    let IdwtSubBandBuffers {
        ll,
        ll_offset,
        hl,
        hl_offset,
        lh,
        lh_offset,
        hh,
        hh_offset,
    } = sub_bands;
    encoder.setComputePipelineState(&kernels.idwt_interleave);
    encoder.set_buffer(0, Some(ll), ll_offset as u64);
    encoder.set_buffer(1, Some(hl), hl_offset as u64);
    encoder.set_buffer(2, Some(lh), lh_offset as u64);
    encoder.set_buffer(3, Some(hh), hh_offset as u64);
    encoder.set_buffer(4, Some(decoded), decoded_offset as u64);
    encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(5, &params);
    dispatch_2d_pipeline(
        encoder,
        &kernels.idwt_interleave,
        (params.width, params.height),
    );
    encoder.memory_barrier_with_resources(&[decoded]);

    dispatch_irreversible97_stages(
        encoder,
        kernels,
        decoded,
        decoded_offset,
        params,
        high_pass,
        1,
    );
}

pub(super) fn dispatch_irreversible97_stages(
    encoder: &ComputeCommandEncoderRef,
    kernels: &crate::engine::runtime::DecodeKernels,
    decoded: &Buffer,
    decoded_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
    high_pass: f32,
    batch_count: u32,
) {
    #[cfg(test)]
    crate::engine::test_counters::record_idwt97_stage_sequence();

    encoder.setComputePipelineState(&kernels.idwt_irreversible97_horizontal_scale);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(1, &params);
    encoder.set_bytes::<f32>(2, &high_pass);
    let horizontal_scale_grid = (params.width, params.height, batch_count);
    #[cfg(test)]
    crate::engine::test_counters::record_idwt97_logical_dispatch(horizontal_scale_grid);
    dispatch_3d_pipeline(
        encoder,
        &kernels.idwt_irreversible97_horizontal_scale,
        horizontal_scale_grid,
    );
    encoder.memory_barrier_with_resources(&[decoded]);

    let first_even_x = (params.x0 + params.output_x) & 1;
    let first_odd_x = 1 - first_even_x;
    encoder.setComputePipelineState(&kernels.idwt_irreversible97_horizontal_step);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(1, &params);
    for (coefficient, parity) in [
        (dwt::IDWT97_NEG_DELTA_F32, first_even_x),
        (dwt::IDWT97_NEG_GAMMA_F32, first_odd_x),
        (dwt::IDWT97_NEG_BETA_F32, first_even_x),
        (dwt::IDWT97_NEG_ALPHA_F32, first_odd_x),
    ] {
        let step = J2kIdwt97StepParams {
            coefficient,
            parity,
            _reserved0: 0,
            _reserved1: 0,
        };
        encoder.set_bytes::<J2kIdwt97StepParams>(2, &step);
        let horizontal_step_grid = (params.width, params.height, batch_count);
        #[cfg(test)]
        crate::engine::test_counters::record_idwt97_logical_dispatch(horizontal_step_grid);
        dispatch_3d_pipeline(
            encoder,
            &kernels.idwt_irreversible97_horizontal_step,
            horizontal_step_grid,
        );
        encoder.memory_barrier_with_resources(&[decoded]);
    }
    encoder.setComputePipelineState(&kernels.idwt_irreversible97_vertical_scale);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(1, &params);
    encoder.set_bytes::<f32>(2, &high_pass);
    let vertical_scale_grid = (params.width, params.height, batch_count);
    #[cfg(test)]
    crate::engine::test_counters::record_idwt97_logical_dispatch(vertical_scale_grid);
    dispatch_3d_pipeline(
        encoder,
        &kernels.idwt_irreversible97_vertical_scale,
        vertical_scale_grid,
    );
    encoder.memory_barrier_with_resources(&[decoded]);

    let first_even_y = (params.y0 + params.output_y) & 1;
    let first_odd_y = 1 - first_even_y;
    encoder.setComputePipelineState(&kernels.idwt_irreversible97_vertical_step);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(1, &params);
    for (coefficient, parity) in [
        (dwt::IDWT97_NEG_DELTA_F32, first_even_y),
        (dwt::IDWT97_NEG_GAMMA_F32, first_odd_y),
        (dwt::IDWT97_NEG_BETA_F32, first_even_y),
        (dwt::IDWT97_NEG_ALPHA_F32, first_odd_y),
    ] {
        let step = J2kIdwt97StepParams {
            coefficient,
            parity,
            _reserved0: 0,
            _reserved1: 0,
        };
        encoder.set_bytes::<J2kIdwt97StepParams>(2, &step);
        let vertical_step_grid = (params.width, params.height, batch_count);
        #[cfg(test)]
        crate::engine::test_counters::record_idwt97_logical_dispatch(vertical_step_grid);
        dispatch_3d_pipeline(
            encoder,
            &kernels.idwt_irreversible97_vertical_step,
            vertical_step_grid,
        );
        encoder.memory_barrier_with_resources(&[decoded]);
    }
}

#[cfg(test)]
mod parity_tests;
#[cfg(test)]
mod performance;
