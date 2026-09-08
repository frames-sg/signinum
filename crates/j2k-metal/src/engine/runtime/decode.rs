// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::metal_types::{Buffer, ComputePipelineState, Device};
use j2k_metal_support::{checked_shared_buffer_with_slice, MetalPipelineLoader, MetalSupportError};
use j2k_native::{ht_uvlc_table0, ht_uvlc_table1, ht_vlc_table0, ht_vlc_table1};

pub(in crate::engine) struct DecodeKernels {
    pub(in crate::engine) expand_sampled_plane: ComputePipelineState,
    pub(in crate::engine) pack_gray8: ComputePipelineState,
    pub(in crate::engine) pack_rgb8: ComputePipelineState,
    pub(in crate::engine) pack_mct_rgb8: ComputePipelineState,
    pub(in crate::engine) pack_mct_rgb8_batched: ComputePipelineState,
    pub(in crate::engine) pack_rgb_opaque_rgba8: ComputePipelineState,
    pub(in crate::engine) pack_rgba8: ComputePipelineState,
    pub(in crate::engine) pack_gray16: ComputePipelineState,
    pub(in crate::engine) pack_rgb16: ComputePipelineState,
    pub(in crate::engine) pack_u8_repeated_gray: ComputePipelineState,
    pub(in crate::engine) pack_u16_repeated_gray: ComputePipelineState,
    pub(in crate::engine) classic_cleanup_plain_batched: ComputePipelineState,
    pub(in crate::engine) classic_cleanup_batched: ComputePipelineState,
    pub(in crate::engine) classic_cleanup_plain_repeated_batched: ComputePipelineState,
    pub(in crate::engine) classic_cleanup_plain_dev_repeated_batched: ComputePipelineState,
    pub(in crate::engine) classic_cleanup_repeated_batched: ComputePipelineState,
    pub(in crate::engine) classic_store_repeated_batched: ComputePipelineState,
    pub(in crate::engine) idwt_interleave: ComputePipelineState,
    pub(in crate::engine) idwt_reversible53_horizontal: ComputePipelineState,
    pub(in crate::engine) idwt_reversible53_vertical: ComputePipelineState,
    pub(in crate::engine) idwt_interleave_batched: ComputePipelineState,
    pub(in crate::engine) idwt_irreversible97_interleave_horizontal_scale: ComputePipelineState,
    pub(in crate::engine) idwt_irreversible97_interleave_horizontal_scale_batched:
        ComputePipelineState,
    pub(in crate::engine) idwt_reversible53_horizontal_batched: ComputePipelineState,
    pub(in crate::engine) idwt_reversible53_vertical_batched: ComputePipelineState,
    #[cfg(test)]
    pub(in crate::engine) idwt_irreversible97_horizontal_scale: ComputePipelineState,
    pub(in crate::engine) idwt_irreversible97_vertical_scale: ComputePipelineState,
    pub(in crate::engine) idwt_irreversible97_horizontal_step: ComputePipelineState,
    pub(in crate::engine) idwt_irreversible97_vertical_step: ComputePipelineState,
    pub(in crate::engine) inverse_mct: ComputePipelineState,
    pub(in crate::engine) store_component: ComputePipelineState,
    pub(in crate::engine) store_component_repeated: ComputePipelineState,
    pub(in crate::engine) store_component_repeated_gray_u8: ComputePipelineState,
    pub(in crate::engine) store_component_repeated_gray_u16: ComputePipelineState,
    pub(in crate::engine) store_component_repeated_gray_i16: ComputePipelineState,
    pub(in crate::engine) store_component_repeated_gray_u8_contiguous: ComputePipelineState,
    pub(in crate::engine) store_component_repeated_gray_u16_contiguous: ComputePipelineState,
    pub(in crate::engine) store_component_gray_u8: ComputePipelineState,
    pub(in crate::engine) store_component_gray_u16: ComputePipelineState,
    pub(in crate::engine) store_component_gray_i16: ComputePipelineState,
    pub(in crate::engine) store_native_rgb_batch_u8: ComputePipelineState,
    pub(in crate::engine) store_native_rgb_batch_u16: ComputePipelineState,
    pub(in crate::engine) store_native_rgb_batch_i16: ComputePipelineState,
    pub(in crate::engine) store_native_rgba_batch_u8: ComputePipelineState,
    pub(in crate::engine) store_native_rgba_batch_u16: ComputePipelineState,
    pub(in crate::engine) store_native_rgba_batch_i16: ComputePipelineState,
    pub(in crate::engine) ht_cleanup: ComputePipelineState,
    pub(in crate::engine) ht_cleanup_batched: ComputePipelineState,
    pub(in crate::engine) ht_cleanup_batched_cleanup_only: ComputePipelineState,
    pub(in crate::engine) ht_cleanup_batched_sigprop: ComputePipelineState,
    pub(in crate::engine) ht_cleanup_batched_magref: ComputePipelineState,
    pub(in crate::engine) ht_cleanup_repeated_batched_cleanup_only: ComputePipelineState,
    pub(in crate::engine) ht_cleanup_repeated_batched_sigprop: ComputePipelineState,
    pub(in crate::engine) ht_cleanup_repeated_batched_magref: ComputePipelineState,
    pub(in crate::engine) ht_vlc_table0: Buffer,
    pub(in crate::engine) ht_vlc_table1: Buffer,
    pub(in crate::engine) ht_uvlc_table0: Buffer,
    pub(in crate::engine) ht_uvlc_table1: Buffer,
}

impl DecodeKernels {
    pub(super) fn new(device: &Device) -> Result<Self, MetalSupportError> {
        let source = super::super::shader_source::decode_shader_source();
        let loader = MetalPipelineLoader::new(device, &source)?;
        Ok(Self {
            expand_sampled_plane: loader.pipeline("j2k_expand_sampled_plane")?,
            pack_gray8: loader.pipeline("j2k_pack_gray8")?,
            pack_rgb8: loader.pipeline("j2k_pack_rgb8")?,
            pack_mct_rgb8: loader.pipeline("j2k_pack_mct_rgb8")?,
            pack_mct_rgb8_batched: loader.pipeline("j2k_pack_mct_rgb8_batched")?,
            pack_rgb_opaque_rgba8: loader.pipeline("j2k_pack_rgb_opaque_rgba8")?,
            pack_rgba8: loader.pipeline("j2k_pack_rgba8")?,
            pack_gray16: loader.pipeline("j2k_pack_gray16")?,
            pack_rgb16: loader.pipeline("j2k_pack_rgb16")?,
            pack_u8_repeated_gray: loader.pipeline("j2k_pack_u8_repeated_gray")?,
            pack_u16_repeated_gray: loader.pipeline("j2k_pack_u16_repeated_gray")?,
            classic_cleanup_plain_batched: loader
                .pipeline("j2k_decode_classic_cleanup_plain_batched")?,
            classic_cleanup_batched: loader.pipeline("j2k_decode_classic_cleanup_batched")?,
            classic_cleanup_plain_repeated_batched: loader
                .pipeline("j2k_decode_classic_cleanup_plain_repeated_batched")?,
            classic_cleanup_plain_dev_repeated_batched: loader
                .pipeline("j2k_decode_classic_cleanup_plain_dev_repeated_batched")?,
            classic_cleanup_repeated_batched: loader
                .pipeline("j2k_decode_classic_cleanup_repeated_batched")?,
            classic_store_repeated_batched: loader
                .pipeline("j2k_store_classic_repeated_batched")?,
            idwt_interleave: loader.pipeline("j2k_idwt_interleave")?,
            idwt_reversible53_horizontal: loader
                .pipeline("j2k_idwt_reversible53_horizontal_pass")?,
            idwt_reversible53_vertical: loader.pipeline("j2k_idwt_reversible53_vertical_pass")?,
            idwt_interleave_batched: loader.pipeline("j2k_idwt_interleave_batched")?,
            idwt_irreversible97_interleave_horizontal_scale: loader
                .pipeline("j2k_idwt_irreversible97_interleave_horizontal_scale")?,
            idwt_irreversible97_interleave_horizontal_scale_batched: loader
                .pipeline("j2k_idwt_irreversible97_interleave_horizontal_scale_batched")?,
            idwt_reversible53_horizontal_batched: loader
                .pipeline("j2k_idwt_reversible53_horizontal_pass_batched")?,
            idwt_reversible53_vertical_batched: loader
                .pipeline("j2k_idwt_reversible53_vertical_pass_batched")?,
            #[cfg(test)]
            idwt_irreversible97_horizontal_scale: loader
                .pipeline("j2k_idwt_irreversible97_horizontal_scale")?,
            idwt_irreversible97_vertical_scale: loader
                .pipeline("j2k_idwt_irreversible97_vertical_scale")?,
            idwt_irreversible97_horizontal_step: loader
                .pipeline("j2k_idwt_irreversible97_horizontal_step")?,
            idwt_irreversible97_vertical_step: loader
                .pipeline("j2k_idwt_irreversible97_vertical_step")?,
            inverse_mct: loader.pipeline("j2k_inverse_mct")?,
            store_component: loader.pipeline("j2k_store_component")?,
            store_component_repeated: loader.pipeline("j2k_store_component_repeated")?,
            store_component_repeated_gray_u8: loader
                .pipeline("j2k_store_component_repeated_gray_u8")?,
            store_component_repeated_gray_u16: loader
                .pipeline("j2k_store_component_repeated_gray_u16")?,
            store_component_repeated_gray_i16: loader
                .pipeline("j2k_store_component_repeated_gray_i16")?,
            store_component_repeated_gray_u8_contiguous: loader
                .pipeline("j2k_store_component_repeated_gray_u8_contiguous")?,
            store_component_repeated_gray_u16_contiguous: loader
                .pipeline("j2k_store_component_repeated_gray_u16_contiguous")?,
            store_component_gray_u8: loader.pipeline("j2k_store_component_gray_u8")?,
            store_component_gray_u16: loader.pipeline("j2k_store_component_gray_u16")?,
            store_component_gray_i16: loader.pipeline("j2k_store_component_gray_i16")?,
            store_native_rgb_batch_u8: loader.pipeline("j2k_store_native_rgb_batch_u8")?,
            store_native_rgb_batch_u16: loader.pipeline("j2k_store_native_rgb_batch_u16")?,
            store_native_rgb_batch_i16: loader.pipeline("j2k_store_native_rgb_batch_i16")?,
            store_native_rgba_batch_u8: loader.pipeline("j2k_store_native_rgba_batch_u8")?,
            store_native_rgba_batch_u16: loader.pipeline("j2k_store_native_rgba_batch_u16")?,
            store_native_rgba_batch_i16: loader.pipeline("j2k_store_native_rgba_batch_i16")?,
            ht_cleanup: loader.pipeline("j2k_decode_ht_cleanup")?,
            ht_cleanup_batched: loader.pipeline("j2k_decode_ht_cleanup_batched")?,
            ht_cleanup_batched_cleanup_only: loader
                .pipeline("j2k_decode_ht_cleanup_batched_cleanup_only")?,
            ht_cleanup_batched_sigprop: loader.pipeline("j2k_decode_ht_cleanup_batched_sigprop")?,
            ht_cleanup_batched_magref: loader.pipeline("j2k_decode_ht_cleanup_batched_magref")?,
            ht_cleanup_repeated_batched_cleanup_only: loader
                .pipeline("j2k_decode_ht_cleanup_repeated_batched_cleanup_only")?,
            ht_cleanup_repeated_batched_sigprop: loader
                .pipeline("j2k_decode_ht_cleanup_repeated_batched_sigprop")?,
            ht_cleanup_repeated_batched_magref: loader
                .pipeline("j2k_decode_ht_cleanup_repeated_batched_magref")?,
            ht_vlc_table0: checked_shared_buffer_with_slice(device, ht_vlc_table0())?,
            ht_vlc_table1: checked_shared_buffer_with_slice(device, ht_vlc_table1())?,
            ht_uvlc_table0: checked_shared_buffer_with_slice(device, ht_uvlc_table0())?,
            ht_uvlc_table1: checked_shared_buffer_with_slice(device, ht_uvlc_table1())?,
        })
    }
}
