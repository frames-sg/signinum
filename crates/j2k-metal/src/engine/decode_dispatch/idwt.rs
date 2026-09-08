// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use super::{
    checked_buffer_slice, commit_and_wait_metal, copied_slice_buffer, dispatch_2d_pipeline,
    dispatch_3d_pipeline, hybrid_stage_signpost, label_compute_encoder, new_command_buffer,
    new_compute_command_encoder, with_runtime, Buffer, CommandBufferRef, ComputeCommandEncoderRef,
    DirectIdwtCommandBuffers, Error, J2kIdwtSingleDecompositionParams,
    J2kRepeatedIdwtSingleDecompositionParams, J2kSingleDecompositionIdwtJob,
    SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE,
};
#[cfg(target_os = "macos")]
mod batched_irreversible;
#[cfg(all(test, target_os = "macos"))]
mod cooperative53;
#[cfg(target_os = "macos")]
mod irreversible;
#[cfg(target_os = "macos")]
pub(in crate::engine) use batched_irreversible::{
    dispatch_irreversible97_repeated_buffers_in_command_buffer_with_offsets,
    dispatch_irreversible97_repeated_buffers_in_encoder_with_offsets,
};
#[cfg(all(target_os = "macos", test))]
pub(crate) use irreversible::decode_irreversible97_staged_single_decomposition_idwt;
#[cfg(target_os = "macos")]
pub(crate) use irreversible::{
    decode_irreversible97_single_decomposition_idwt,
    decode_openjpeg_irreversible97_single_decomposition_idwt,
};
#[cfg(target_os = "macos")]
pub(in crate::engine) use irreversible::{
    dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets,
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets,
};

#[cfg(target_os = "macos")]
pub(crate) fn decode_reversible53_single_decomposition_idwt(
    job: J2kSingleDecompositionIdwtJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    with_runtime(|runtime| {
        let required_len = job.rect.width() as usize * job.rect.height() as usize;
        if output.len() < required_len {
            return Err(Error::MetalKernel {
                message: "J2K Metal IDWT output slice is too small".to_string(),
            });
        }

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

        let ll = copied_slice_buffer(&runtime.device, job.ll.coefficients)?;
        let hl = copied_slice_buffer(&runtime.device, job.hl.coefficients)?;
        let lh = copied_slice_buffer(&runtime.device, job.lh.coefficients)?;
        let hh = copied_slice_buffer(&runtime.device, job.hh.coefficients)?;
        let decoded = copied_slice_buffer(&runtime.device, output)?;

        let command_buffer = new_command_buffer(&runtime.queue)?;

        let encoder = new_compute_command_encoder(&command_buffer)?;
        encoder.setComputePipelineState(&runtime.decode()?.idwt_interleave);
        encoder.set_buffer(0, Some(&ll), 0);
        encoder.set_buffer(1, Some(&hl), 0);
        encoder.set_buffer(2, Some(&lh), 0);
        encoder.set_buffer(3, Some(&hh), 0);
        encoder.set_buffer(4, Some(&decoded), 0);
        encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(5, &params);
        dispatch_2d_pipeline(
            &encoder,
            &runtime.decode()?.idwt_interleave,
            (params.width, params.height),
        );
        encoder.endEncoding();

        let encoder = new_compute_command_encoder(&command_buffer)?;
        encoder.setComputePipelineState(&runtime.decode()?.idwt_reversible53_horizontal);
        encoder.set_buffer(0, Some(&decoded), 0);
        encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(1, &params);
        let horizontal_width = runtime
            .decode()?
            .idwt_reversible53_horizontal
            .threadExecutionWidth()
            .max(1);
        encoder.dispatchThreads_threadsPerThreadgroup(
            j2k_metal_support::mtl_size(u64::from(params.height), 1, 1),
            j2k_metal_support::mtl_size(horizontal_width as u64, 1, 1),
        );
        encoder.endEncoding();

        let encoder = new_compute_command_encoder(&command_buffer)?;
        encoder.setComputePipelineState(&runtime.decode()?.idwt_reversible53_vertical);
        encoder.set_buffer(0, Some(&decoded), 0);
        encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(1, &params);
        let vertical_width = runtime
            .decode()?
            .idwt_reversible53_vertical
            .threadExecutionWidth()
            .max(1);
        encoder.dispatchThreads_threadsPerThreadgroup(
            j2k_metal_support::mtl_size(u64::from(params.width), 1, 1),
            j2k_metal_support::mtl_size(vertical_width as u64, 1, 1),
        );
        encoder.endEncoding();
        commit_and_wait_metal(&command_buffer)?;
        let decoded_host = checked_buffer_slice::<f32>(&decoded, output.len(), "IDWT output")?;
        output.copy_from_slice(&decoded_host);
        Ok(())
    })
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::engine) struct IdwtSubBandBuffers<'a> {
    pub(in crate::engine) ll: &'a Buffer,
    pub(in crate::engine) ll_offset: usize,
    pub(in crate::engine) hl: &'a Buffer,
    pub(in crate::engine) hl_offset: usize,
    pub(in crate::engine) lh: &'a Buffer,
    pub(in crate::engine) lh_offset: usize,
    pub(in crate::engine) hh: &'a Buffer,
    pub(in crate::engine) hh_offset: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::engine) struct SingleIdwtDispatch<'a> {
    pub(in crate::engine) kernels: &'a crate::engine::runtime::DecodeKernels,
    pub(in crate::engine) sub_bands: IdwtSubBandBuffers<'a>,
    pub(in crate::engine) params: J2kIdwtSingleDecompositionParams,
    pub(in crate::engine) decoded: &'a Buffer,
    pub(in crate::engine) decoded_offset: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::engine) struct RepeatedIdwtDispatch<'a> {
    pub(in crate::engine) kernels: &'a crate::engine::runtime::DecodeKernels,
    pub(in crate::engine) sub_bands: IdwtSubBandBuffers<'a>,
    pub(in crate::engine) params: J2kRepeatedIdwtSingleDecompositionParams,
    pub(in crate::engine) decoded: &'a Buffer,
}

#[cfg(target_os = "macos")]
pub(in crate::engine) fn dispatch_reversible53_single_decomposition_buffers_in_command_buffer_with_offsets(
    command_buffer: &CommandBufferRef,
    dispatch: SingleIdwtDispatch<'_>,
) -> Result<(), Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE);
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K decode hybrid reversible53 IDWT");
    dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(&encoder, dispatch);
    encoder.endEncoding();
    Ok(())
}

#[cfg(target_os = "macos")]
pub(in crate::engine) fn dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
    encoder: &ComputeCommandEncoderRef,
    dispatch: SingleIdwtDispatch<'_>,
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

    encoder.setComputePipelineState(&kernels.idwt_reversible53_horizontal);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(1, &params);
    let horizontal_width = kernels
        .idwt_reversible53_horizontal
        .threadExecutionWidth()
        .max(1);
    encoder.dispatchThreads_threadsPerThreadgroup(
        j2k_metal_support::mtl_size(u64::from(params.height), 1, 1),
        j2k_metal_support::mtl_size(horizontal_width as u64, 1, 1),
    );
    encoder.memory_barrier_with_resources(&[decoded]);

    encoder.setComputePipelineState(&kernels.idwt_reversible53_vertical);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(1, &params);
    let vertical_width = kernels
        .idwt_reversible53_vertical
        .threadExecutionWidth()
        .max(1);
    encoder.dispatchThreads_threadsPerThreadgroup(
        j2k_metal_support::mtl_size(u64::from(params.width), 1, 1),
        j2k_metal_support::mtl_size(vertical_width as u64, 1, 1),
    );
}

#[cfg(target_os = "macos")]
pub(in crate::engine) fn dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets(
    command_buffers: DirectIdwtCommandBuffers<'_>,
    dispatch: RepeatedIdwtDispatch<'_>,
) -> Result<(), Error> {
    let RepeatedIdwtDispatch {
        kernels,
        sub_bands,
        params,
        decoded,
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
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE);
    let encoder = new_compute_command_encoder(command_buffers.interleave)?;
    label_compute_encoder(&encoder, "J2K decode hybrid repeated IDWT interleave");
    encoder.setComputePipelineState(&kernels.idwt_interleave_batched);
    encoder.set_buffer(0, Some(ll), ll_offset as u64);
    encoder.set_buffer(1, Some(hl), hl_offset as u64);
    encoder.set_buffer(2, Some(lh), lh_offset as u64);
    encoder.set_buffer(3, Some(hh), hh_offset as u64);
    encoder.set_buffer(4, Some(decoded), 0);
    encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(5, &params);
    dispatch_3d_pipeline(
        &encoder,
        &kernels.idwt_interleave_batched,
        (params.width, params.height, params.batch_count),
    );
    encoder.endEncoding();

    let encoder = new_compute_command_encoder(command_buffers.horizontal)?;
    label_compute_encoder(&encoder, "J2K decode hybrid repeated IDWT horizontal");
    encoder.setComputePipelineState(&kernels.idwt_reversible53_horizontal_batched);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(1, &params);
    let horizontal_width = kernels
        .idwt_reversible53_horizontal_batched
        .threadExecutionWidth()
        .max(1);
    encoder.dispatchThreads_threadsPerThreadgroup(
        j2k_metal_support::mtl_size(u64::from(params.height), u64::from(params.batch_count), 1),
        j2k_metal_support::mtl_size(horizontal_width as u64, 1, 1),
    );
    encoder.endEncoding();

    let encoder = new_compute_command_encoder(command_buffers.vertical)?;
    label_compute_encoder(&encoder, "J2K decode hybrid repeated IDWT vertical");
    encoder.setComputePipelineState(&kernels.idwt_reversible53_vertical_batched);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(1, &params);
    let vertical_width = kernels
        .idwt_reversible53_vertical_batched
        .threadExecutionWidth()
        .max(1);
    encoder.dispatchThreads_threadsPerThreadgroup(
        j2k_metal_support::mtl_size(u64::from(params.width), u64::from(params.batch_count), 1),
        j2k_metal_support::mtl_size(vertical_width as u64, 1, 1),
    );
    encoder.endEncoding();
    Ok(())
}

#[cfg(target_os = "macos")]
pub(in crate::engine) fn dispatch_reversible53_repeated_buffers_in_encoder_with_offsets(
    encoder: &ComputeCommandEncoderRef,
    dispatch: RepeatedIdwtDispatch<'_>,
) {
    let RepeatedIdwtDispatch {
        kernels,
        sub_bands,
        params,
        decoded,
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
    encoder.setComputePipelineState(&kernels.idwt_interleave_batched);
    encoder.set_buffer(0, Some(ll), ll_offset as u64);
    encoder.set_buffer(1, Some(hl), hl_offset as u64);
    encoder.set_buffer(2, Some(lh), lh_offset as u64);
    encoder.set_buffer(3, Some(hh), hh_offset as u64);
    encoder.set_buffer(4, Some(decoded), 0);
    encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(5, &params);
    dispatch_3d_pipeline(
        encoder,
        &kernels.idwt_interleave_batched,
        (params.width, params.height, params.batch_count),
    );
    encoder.memory_barrier_with_resources(&[decoded]);
    #[cfg(test)]
    if cooperative53::route::try_dispatch(encoder, decoded, &params) {
        return;
    }

    encoder.setComputePipelineState(&kernels.idwt_reversible53_horizontal_batched);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(1, &params);
    let horizontal_width = kernels
        .idwt_reversible53_horizontal_batched
        .threadExecutionWidth()
        .max(1);
    encoder.dispatchThreads_threadsPerThreadgroup(
        j2k_metal_support::mtl_size(u64::from(params.height), u64::from(params.batch_count), 1),
        j2k_metal_support::mtl_size(horizontal_width as u64, 1, 1),
    );
    encoder.memory_barrier_with_resources(&[decoded]);

    encoder.setComputePipelineState(&kernels.idwt_reversible53_vertical_batched);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(1, &params);
    let vertical_width = kernels
        .idwt_reversible53_vertical_batched
        .threadExecutionWidth()
        .max(1);
    encoder.dispatchThreads_threadsPerThreadgroup(
        j2k_metal_support::mtl_size(u64::from(params.width), u64::from(params.batch_count), 1),
        j2k_metal_support::mtl_size(vertical_width as u64, 1, 1),
    );
}
