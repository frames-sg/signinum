// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::metal_types::prelude::*;

use super::irreversible::{
    dispatch_irreversible97_horizontal_scale, dispatch_irreversible97_stages_after_horizontal_scale,
};
use super::{
    dispatch_3d_pipeline, label_compute_encoder, new_compute_command_encoder, CommandBufferRef,
    ComputeCommandEncoderRef, Error, J2kIdwtSingleDecompositionParams,
    J2kRepeatedIdwtSingleDecompositionParams, RepeatedIdwtDispatch,
};

pub(in crate::engine) fn dispatch_irreversible97_repeated_buffers_in_command_buffer_with_offsets(
    command_buffer: &CommandBufferRef,
    dispatch: RepeatedIdwtDispatch<'_>,
) -> Result<(), Error> {
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K decode batched irreversible97 IDWT");
    dispatch_irreversible97_repeated_buffers_in_encoder_with_offsets(&encoder, dispatch);
    encoder.endEncoding();
    Ok(())
}

pub(in crate::engine) fn dispatch_irreversible97_repeated_buffers_in_encoder_with_offsets(
    encoder: &ComputeCommandEncoderRef,
    dispatch: RepeatedIdwtDispatch<'_>,
) {
    let high_pass = j2k_codec_math::dwt::IDWT97_OPENJPEG_TWO_INV_KAPPA_F32 * 0.5;
    dispatch_irreversible97_repeated_interleave_horizontal_scale(encoder, dispatch, high_pass);
    dispatch_irreversible97_stages_after_horizontal_scale(
        encoder,
        dispatch.kernels,
        dispatch.decoded,
        0,
        single_params(dispatch.params),
        high_pass,
        dispatch.params.batch_count,
    );
}

pub(super) fn dispatch_irreversible97_repeated_interleave_horizontal_scale(
    encoder: &ComputeCommandEncoderRef,
    dispatch: RepeatedIdwtDispatch<'_>,
    high_pass: f32,
) {
    let RepeatedIdwtDispatch {
        kernels,
        sub_bands,
        params,
        decoded,
    } = dispatch;
    encoder.setComputePipelineState(&kernels.idwt_interleave_batched);
    for (index, buffer, offset) in [
        (0, sub_bands.ll, sub_bands.ll_offset),
        (1, sub_bands.hl, sub_bands.hl_offset),
        (2, sub_bands.lh, sub_bands.lh_offset),
        (3, sub_bands.hh, sub_bands.hh_offset),
    ] {
        encoder.set_buffer(index, Some(buffer), offset as u64);
    }
    encoder.set_buffer(4, Some(decoded), 0);
    encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(5, &params);
    dispatch_3d_pipeline(
        encoder,
        &kernels.idwt_interleave_batched,
        (params.width, params.height, params.batch_count),
    );
    #[cfg(test)]
    crate::engine::test_counters::record_idwt97_logical_dispatch((
        params.width,
        params.height,
        params.batch_count,
    ));
    encoder.memory_barrier_with_resources(&[decoded]);

    // The stacked-plan preflight guarantees identical geometry and origin
    // parity. Only the plane offset varies along the third grid dimension.
    dispatch_irreversible97_horizontal_scale(
        encoder,
        kernels,
        decoded,
        0,
        single_params(params),
        high_pass,
        params.batch_count,
    );
}

fn single_params(
    params: J2kRepeatedIdwtSingleDecompositionParams,
) -> J2kIdwtSingleDecompositionParams {
    J2kIdwtSingleDecompositionParams {
        x0: params.x0,
        y0: params.y0,
        output_x: params.output_x,
        output_y: params.output_y,
        width: params.width,
        height: params.height,
        ll_x: params.ll_x,
        ll_y: params.ll_y,
        ll_width: params.ll_width,
        ll_height: params.ll_height,
        hl_x: params.hl_x,
        hl_y: params.hl_y,
        hl_width: params.hl_width,
        hl_height: params.hl_height,
        lh_x: params.lh_x,
        lh_y: params.lh_y,
        lh_width: params.lh_width,
        lh_height: params.lh_height,
        hh_x: params.hh_x,
        hh_y: params.hh_y,
        hh_width: params.hh_width,
        hh_height: params.hh_height,
    }
}
