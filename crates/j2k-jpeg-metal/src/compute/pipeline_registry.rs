// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG Metal shader source and immutable compute-pipeline registry.

#[cfg(test)]
mod tests;

use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex},
};

use crate::metal_types::{ComputePipelineState, Device};
use j2k_core::PixelFormat;
use j2k_metal_support::{MetalPipelineLoader, MetalSupportError};
use objc2_metal::MTLDevice as _;

// Registry IDs come from actual Metal devices, so this retains at most one
// immutable pipeline set per device observed by the process. Keeping successful
// entries resident avoids recompilation across short-lived backend sessions.
static PIPELINE_REGISTRIES: LazyLock<PipelineRegistryCache> =
    LazyLock::new(PipelineRegistryCache::default);

struct PipelineRegistryCache {
    slots: Mutex<HashMap<u64, Arc<PipelineRegistrySlot>>>,
}

impl Default for PipelineRegistryCache {
    fn default() -> Self {
        Self {
            slots: Mutex::new(HashMap::new()),
        }
    }
}

struct PipelineRegistrySlot {
    registry: Mutex<Option<Arc<JpegPipelineRegistry>>>,
}

impl Default for PipelineRegistrySlot {
    fn default() -> Self {
        Self {
            registry: Mutex::new(None),
        }
    }
}

impl PipelineRegistryCache {
    fn get_or_try_init(
        &self,
        device_registry_id: u64,
        load: impl FnOnce() -> Result<JpegPipelineRegistry, MetalSupportError>,
    ) -> Result<Arc<JpegPipelineRegistry>, MetalSupportError> {
        let slot = {
            let Ok(mut slots) = self.slots.lock() else {
                return load().map(Arc::new);
            };
            if let Some(slot) = slots.get(&device_registry_id) {
                Arc::clone(slot)
            } else {
                // Pipeline reuse is an optimization. If cache metadata cannot
                // grow, preserve decode availability through an uncached load.
                if slots.try_reserve(1).is_err() {
                    drop(slots);
                    return load().map(Arc::new);
                }
                let slot = Arc::new(PipelineRegistrySlot::default());
                slots.insert(device_registry_id, Arc::clone(&slot));
                slot
            }
        };
        let Ok(mut cached) = slot.registry.lock() else {
            return load().map(Arc::new);
        };
        if let Some(registry) = cached.as_ref() {
            return Ok(Arc::clone(registry));
        }

        // Construction is transactional: `load` builds every pipeline in a
        // local value, and the cache publishes it only after complete success.
        // An error or panic cannot expose a partially initialized registry. A
        // panic poisons this slot, so later callers safely bypass the cache.
        let registry = Arc::new(load()?);
        *cached = Some(Arc::clone(&registry));
        Ok(registry)
    }
}

pub(in crate::compute) const SHADER_SOURCE: &str = concat!(
    include_str!("../shaders_shared.metal"),
    include_str!("../shaders_encode.metal"),
    include_str!("../shaders_encode_staged.metal"),
    include_str!("../shaders_decode_helpers.metal"),
    include_str!("../shaders_pack_444.metal"),
    include_str!("../shaders_decode_fast420.metal"),
    include_str!("../shaders_decode_fast422_regions.metal"),
    include_str!("../shaders_decode_fast444.metal"),
    include_str!("../shaders_pack_subsampled.metal"),
);

pub(in crate::compute) struct JpegPipelineRegistry {
    pub(in crate::compute) pack: ComputePipelineState,
    pub(in crate::compute) jpeg_baseline_encode_precompute_batch: ComputePipelineState,
    pub(in crate::compute) jpeg_baseline_encode_entropy_from_coeffs_batch: ComputePipelineState,
    pub(in crate::compute) pack_420: ComputePipelineState,
    pub(in crate::compute) pack_420_rgb: ComputePipelineState,
    pub(in crate::compute) pack_420_rgba: ComputePipelineState,
    pub(in crate::compute) pack_420_rgb_batch: ComputePipelineState,
    pub(in crate::compute) pack_420_rgba_texture: ComputePipelineState,
    pub(in crate::compute) pack_420_windowed_rgb_batch: ComputePipelineState,
    pub(in crate::compute) pack_420_windowed_rgba_texture: ComputePipelineState,
    pub(in crate::compute) pack_422_rgb: ComputePipelineState,
    pub(in crate::compute) pack_422_rgba: ComputePipelineState,
    pub(in crate::compute) pack_422_rgb_batch: ComputePipelineState,
    pub(in crate::compute) pack_422_rgba_texture: ComputePipelineState,
    pub(in crate::compute) pack_422_windowed_rgb_batch: ComputePipelineState,
    pub(in crate::compute) pack_422_windowed_rgba_texture: ComputePipelineState,
    pub(in crate::compute) pack_444_rgb_batch: ComputePipelineState,
    pub(in crate::compute) pack_444_rgba_texture: ComputePipelineState,
    pub(in crate::compute) pack_422_windowed: ComputePipelineState,
    pub(in crate::compute) pack_422_windowed_rgb: ComputePipelineState,
    pub(in crate::compute) pack_422_windowed_rgba: ComputePipelineState,
    pub(in crate::compute) pack_420_windowed: ComputePipelineState,
    pub(in crate::compute) pack_420_windowed_rgb: ComputePipelineState,
    pub(in crate::compute) pack_420_windowed_rgba: ComputePipelineState,
    pub(in crate::compute) fast420_decode: ComputePipelineState,
    pub(in crate::compute) fast420_batch_decode: ComputePipelineState,
    #[cfg(test)]
    pub(in crate::compute) fast420_batch_coeffs_decode: ComputePipelineState,
    #[cfg(test)]
    pub(in crate::compute) fast420_batch_idct_deposit: ComputePipelineState,
    pub(in crate::compute) fast420_scaled_region_batch_decode: ComputePipelineState,
    pub(in crate::compute) fast420_rgba_texture_batch_decode: ComputePipelineState,
    pub(in crate::compute) fast420_rgba_texture_boundary: ComputePipelineState,
    pub(in crate::compute) fast420_rgba_texture_vertical_boundary: ComputePipelineState,
    pub(in crate::compute) fast420_rgba_texture_corner: ComputePipelineState,
    pub(in crate::compute) fast422_decode: ComputePipelineState,
    pub(in crate::compute) fast422_batch_decode: ComputePipelineState,
    pub(in crate::compute) fast422_scaled_region_batch_decode: ComputePipelineState,
    pub(in crate::compute) fast422_rgba_texture_batch_decode: ComputePipelineState,
    pub(in crate::compute) fast422_rgba_texture_boundary: ComputePipelineState,
    pub(in crate::compute) fast422_region_decode: ComputePipelineState,
    pub(in crate::compute) fast422_scaled_decode: ComputePipelineState,
    pub(in crate::compute) fast422_scaled_region_decode: ComputePipelineState,
    pub(in crate::compute) fast420_region_decode: ComputePipelineState,
    pub(in crate::compute) fast420_scaled_decode: ComputePipelineState,
    pub(in crate::compute) fast420_scaled_region_decode: ComputePipelineState,
    pub(in crate::compute) fast444_decode: ComputePipelineState,
    pub(in crate::compute) fast444_region_decode: ComputePipelineState,
    pub(in crate::compute) fast444_scaled_decode: ComputePipelineState,
    pub(in crate::compute) fast444_scaled_region_decode: ComputePipelineState,
    pub(in crate::compute) fast444_scaled_region_batch_decode: ComputePipelineState,
    pub(in crate::compute) fast444_rgba_texture_batch_decode: ComputePipelineState,
    pub(in crate::compute) rgb8_to_rgba_texture: ComputePipelineState,
}

// SAFETY: Metal compute pipeline states are immutable after construction and
// Metal permits their concurrent use across command queues and threads.
unsafe impl Send for JpegPipelineRegistry {}
// SAFETY: Shared references expose only immutable retained pipeline states.
unsafe impl Sync for JpegPipelineRegistry {}

impl JpegPipelineRegistry {
    pub(in crate::compute) fn shared(device: &Device) -> Result<Arc<Self>, MetalSupportError> {
        PIPELINE_REGISTRIES.get_or_try_init(device.registryID(), || Self::load(device))
    }

    pub(in crate::compute) fn load(device: &Device) -> Result<Self, MetalSupportError> {
        let loader = MetalPipelineLoader::new(device, SHADER_SOURCE)?;
        let pipeline = |name: &str| loader.pipeline(name);
        Ok(Self {
            pack: pipeline("jpeg_pack")?,
            jpeg_baseline_encode_precompute_batch: pipeline(
                "jpeg_encode_baseline_precompute_batch",
            )?,
            jpeg_baseline_encode_entropy_from_coeffs_batch: pipeline(
                "jpeg_encode_baseline_entropy_from_coeffs_batch",
            )?,
            pack_420: pipeline("jpeg_pack_420")?,
            pack_420_rgb: pipeline("jpeg_pack_420_rgb")?,
            pack_420_rgba: pipeline("jpeg_pack_420_rgba")?,
            pack_420_rgb_batch: pipeline("jpeg_pack_420_rgb_batch")?,
            pack_420_rgba_texture: pipeline("jpeg_pack_420_rgba_texture")?,
            pack_420_windowed_rgb_batch: pipeline("jpeg_pack_420_windowed_rgb_batch")?,
            pack_420_windowed_rgba_texture: pipeline("jpeg_pack_420_windowed_rgba_texture")?,
            pack_422_rgb: pipeline("jpeg_pack_422_rgb")?,
            pack_422_rgba: pipeline("jpeg_pack_422_rgba")?,
            pack_422_rgb_batch: pipeline("jpeg_pack_422_rgb_batch")?,
            pack_422_rgba_texture: pipeline("jpeg_pack_422_rgba_texture")?,
            pack_422_windowed_rgb_batch: pipeline("jpeg_pack_422_windowed_rgb_batch")?,
            pack_422_windowed_rgba_texture: pipeline("jpeg_pack_422_windowed_rgba_texture")?,
            pack_444_rgb_batch: pipeline("jpeg_pack_444_rgb_batch")?,
            pack_444_rgba_texture: pipeline("jpeg_pack_444_rgba_texture")?,
            pack_422_windowed: pipeline("jpeg_pack_422_windowed")?,
            pack_422_windowed_rgb: pipeline("jpeg_pack_422_windowed_rgb")?,
            pack_422_windowed_rgba: pipeline("jpeg_pack_422_windowed_rgba")?,
            pack_420_windowed: pipeline("jpeg_pack_420_windowed")?,
            pack_420_windowed_rgb: pipeline("jpeg_pack_420_windowed_rgb")?,
            pack_420_windowed_rgba: pipeline("jpeg_pack_420_windowed_rgba")?,
            fast420_decode: pipeline("jpeg_decode_fast420")?,
            fast420_batch_decode: pipeline("jpeg_decode_fast420_batch")?,
            #[cfg(test)]
            fast420_batch_coeffs_decode: pipeline("jpeg_decode_fast420_batch_coeffs")?,
            #[cfg(test)]
            fast420_batch_idct_deposit: pipeline("jpeg_idct_deposit_fast420_batch")?,
            fast420_scaled_region_batch_decode: pipeline(
                "jpeg_decode_fast420_scaled_region_batch",
            )?,
            fast420_rgba_texture_batch_decode: pipeline("jpeg_decode_fast420_rgba_texture_batch")?,
            fast420_rgba_texture_boundary: pipeline(
                "jpeg_resolve_fast420_rgba_texture_boundaries",
            )?,
            fast420_rgba_texture_vertical_boundary: pipeline(
                "jpeg_resolve_fast420_rgba_texture_vertical_boundaries",
            )?,
            fast420_rgba_texture_corner: pipeline("jpeg_resolve_fast420_rgba_texture_corners")?,
            fast422_decode: pipeline("jpeg_decode_fast422")?,
            fast422_batch_decode: pipeline("jpeg_decode_fast422_batch")?,
            fast422_scaled_region_batch_decode: pipeline(
                "jpeg_decode_fast422_scaled_region_batch",
            )?,
            fast422_rgba_texture_batch_decode: pipeline("jpeg_decode_fast422_rgba_texture_batch")?,
            fast422_rgba_texture_boundary: pipeline(
                "jpeg_resolve_fast422_rgba_texture_boundaries",
            )?,
            fast422_region_decode: pipeline("jpeg_decode_fast422_region")?,
            fast422_scaled_decode: pipeline("jpeg_decode_fast422_scaled")?,
            fast422_scaled_region_decode: pipeline("jpeg_decode_fast422_scaled_region")?,
            fast420_region_decode: pipeline("jpeg_decode_fast420_region")?,
            fast420_scaled_decode: pipeline("jpeg_decode_fast420_scaled")?,
            fast420_scaled_region_decode: pipeline("jpeg_decode_fast420_scaled_region")?,
            fast444_decode: pipeline("jpeg_decode_fast444")?,
            fast444_region_decode: pipeline("jpeg_decode_fast444_region")?,
            fast444_scaled_decode: pipeline("jpeg_decode_fast444_scaled")?,
            fast444_scaled_region_decode: pipeline("jpeg_decode_fast444_scaled_region")?,
            fast444_scaled_region_batch_decode: pipeline(
                "jpeg_decode_fast444_scaled_region_batch",
            )?,
            fast444_rgba_texture_batch_decode: pipeline("jpeg_decode_fast444_rgba_texture_batch")?,
            rgb8_to_rgba_texture: pipeline("jpeg_copy_rgb8_to_rgba_texture")?,
        })
    }

    pub(in crate::compute) fn pack_420_for_format(
        &self,
        fmt: PixelFormat,
    ) -> &ComputePipelineState {
        match fmt {
            PixelFormat::Rgb8 => &self.pack_420_rgb,
            PixelFormat::Rgba8 => &self.pack_420_rgba,
            _ => &self.pack_420,
        }
    }

    pub(in crate::compute) fn pack_420_windowed_for_format(
        &self,
        fmt: PixelFormat,
    ) -> &ComputePipelineState {
        match fmt {
            PixelFormat::Rgb8 => &self.pack_420_windowed_rgb,
            PixelFormat::Rgba8 => &self.pack_420_windowed_rgba,
            _ => &self.pack_420_windowed,
        }
    }

    pub(in crate::compute) fn pack_422_for_format(
        &self,
        fmt: PixelFormat,
    ) -> Option<&ComputePipelineState> {
        match fmt {
            PixelFormat::Rgb8 => Some(&self.pack_422_rgb),
            PixelFormat::Rgba8 => Some(&self.pack_422_rgba),
            _ => None,
        }
    }

    pub(in crate::compute) fn pack_422_windowed_for_format(
        &self,
        fmt: PixelFormat,
    ) -> &ComputePipelineState {
        match fmt {
            PixelFormat::Rgb8 => &self.pack_422_windowed_rgb,
            PixelFormat::Rgba8 => &self.pack_422_windowed_rgba,
            _ => &self.pack_422_windowed,
        }
    }
}
