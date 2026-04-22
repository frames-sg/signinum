// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use std::{
    mem::{size_of, size_of_val},
    sync::{Arc, OnceLock},
};

#[cfg(target_os = "macos")]
use metal::{
    Buffer, CommandBufferRef, CommandQueue, CompileOptions, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize,
};
use slidecodec_core::{PixelFormat, Rect};
#[cfg(test)]
use slidecodec_j2k_native::HtCodeBlockDecoder;
use slidecodec_j2k_native::{
    ht_uvlc_table0, ht_uvlc_table1, ht_vlc_table0, ht_vlc_table1, ColorSpace as NativeColorSpace,
    DecodedComponents as NativeDecodedComponents, HtCodeBlockDecodeJob, HtSubBandDecodeJob,
    J2kCodeBlockDecodeJob, J2kDirectBandId, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep,
    J2kInverseMctJob, J2kSingleDecompositionIdwtJob, J2kStoreComponentJob, J2kSubBandDecodeJob,
    J2kWaveletTransform,
};
#[cfg(test)]
use slidecodec_j2k_native::{
    DecodeSettings as NativeDecodeSettings, DecoderContext as NativeDecoderContext,
    Image as NativeImage,
};

#[cfg(test)]
use crate::{
    classic::MetalClassicBlockDecoder, ht::MetalHtBlockDecoder, idwt::MetalIdwtDecoder,
    mct::MetalMctDecoder, store::MetalStoreDecoder,
};
use crate::{Error, Surface};

#[cfg(test)]
#[derive(Default)]
struct MetalCodeBlockDecoder {
    classic: MetalClassicBlockDecoder,
    ht: MetalHtBlockDecoder,
    idwt: MetalIdwtDecoder,
    mct: MetalMctDecoder,
    store: MetalStoreDecoder,
}

#[cfg(test)]
impl HtCodeBlockDecoder for MetalCodeBlockDecoder {
    fn decode_j2k_sub_band(
        &mut self,
        job: J2kSubBandDecodeJob<'_>,
        output: &mut [f32],
    ) -> slidecodec_j2k_native::Result<bool> {
        self.classic.decode_j2k_sub_band(job, output)
    }

    fn decode_j2k_code_block(
        &mut self,
        job: slidecodec_j2k_native::J2kCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> slidecodec_j2k_native::Result<bool> {
        self.classic.decode_j2k_code_block(job, output)
    }

    fn decode_sub_band(
        &mut self,
        job: HtSubBandDecodeJob<'_>,
        output: &mut [f32],
    ) -> slidecodec_j2k_native::Result<bool> {
        self.ht.decode_sub_band(job, output)
    }

    fn decode_code_block(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> slidecodec_j2k_native::Result<()> {
        self.ht.decode_code_block(job, output)
    }

    fn decode_single_decomposition_idwt(
        &mut self,
        job: J2kSingleDecompositionIdwtJob<'_>,
        output: &mut [f32],
    ) -> slidecodec_j2k_native::Result<bool> {
        self.idwt.decode_single_decomposition_idwt(job, output)
    }

    fn decode_inverse_mct(
        &mut self,
        job: J2kInverseMctJob<'_>,
    ) -> slidecodec_j2k_native::Result<bool> {
        self.mct.decode_inverse_mct(job)
    }

    fn decode_store_component(
        &mut self,
        job: J2kStoreComponentJob<'_>,
    ) -> slidecodec_j2k_native::Result<bool> {
        self.store.decode_store_component(job)
    }
}

#[cfg(target_os = "macos")]
const SHADER_SOURCE: &str = concat!(
    r#"
#include <metal_stdlib>
using namespace metal;

struct J2kPackParams {
    uint width;
    uint height;
    uint out_stride;
    uint plane_count;
    uint output_channels;
    uint opaque_alpha;
    uint bit_depths[4];
};

inline uint max_value_for_bit_depth(uint bit_depth) {
    const uint clamped = min(bit_depth, 16u);
    const uint max_value = (1u << clamped) - 1u;
    return max(max_value, 1u);
}

inline uchar scale_to_u8(float sample, uint bit_depth) {
    const float max_value = float(max_value_for_bit_depth(bit_depth));
    const float clamped = clamp(sample, 0.0f, max_value);
    return uchar(min(floor((clamped * 255.0f) / max_value + 0.5f), 255.0f));
}

inline ushort pack_to_u16(float sample, uint bit_depth) {
    const float max_value = float(max_value_for_bit_depth(bit_depth));
    const float clamped = clamp(sample, 0.0f, max_value);
    if (bit_depth <= 8u) {
        return ushort(min(floor((clamped * 65535.0f) / max_value + 0.5f), 65535.0f));
    }
    return ushort(min(floor(clamped + 0.5f), 65535.0f));
}

kernel void j2k_pack_u8(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device const float *plane3 [[buffer(3)]],
    device uchar *out [[buffer(4)]],
    constant J2kPackParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    uint out_idx = gid.y * params.out_stride + gid.x * params.output_channels;

    if (params.output_channels == 1) {
        out[out_idx] = scale_to_u8(plane0[idx], params.bit_depths[0]);
        return;
    }

    out[out_idx] = scale_to_u8(plane0[idx], params.bit_depths[0]);
    out[out_idx + 1] = scale_to_u8(plane1[idx], params.bit_depths[1]);
    out[out_idx + 2] = scale_to_u8(plane2[idx], params.bit_depths[2]);

    if (params.output_channels == 4) {
        out[out_idx + 3] = params.opaque_alpha != 0
            ? uchar(255)
            : scale_to_u8(plane3[idx], params.bit_depths[3]);
    }
}

kernel void j2k_pack_u16(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device const float *plane3 [[buffer(3)]],
    device ushort *out [[buffer(4)]],
    constant J2kPackParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    uint out_idx = (gid.y * params.out_stride) / 2u + gid.x * params.output_channels;

    if (params.output_channels == 1) {
        out[out_idx] = pack_to_u16(plane0[idx], params.bit_depths[0]);
        return;
    }

    out[out_idx] = pack_to_u16(plane0[idx], params.bit_depths[0]);
    out[out_idx + 1] = pack_to_u16(plane1[idx], params.bit_depths[1]);
    out[out_idx + 2] = pack_to_u16(plane2[idx], params.bit_depths[2]);
}
"#,
    "\n",
    include_str!("classic.metal"),
    "\n",
    include_str!("idwt.metal"),
    "\n",
    include_str!("mct.metal"),
    "\n",
    include_str!("store.metal"),
    "\n",
    include_str!("ht_cleanup.metal"),
);

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kPackParams {
    width: u32,
    height: u32,
    out_stride: u32,
    plane_count: u32,
    output_channels: u32,
    opaque_alpha: u32,
    bit_depths: [u32; 4],
}

#[cfg(target_os = "macos")]
const J2K_CLASSIC_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STATUS_FAIL: u32 = 1;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STATUS_UNSUPPORTED: u32 = 2;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES: u32 = 1 << 0;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS: u32 = 1 << 1;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT: u32 = 1 << 2;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS: u32 = 1 << 3;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS: u32 = 1 << 4;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kClassicCleanupBatchJob {
    coded_offset: u32,
    coded_len: u32,
    segment_offset: u32,
    segment_count: u32,
    width: u32,
    height: u32,
    output_stride: u32,
    output_offset: u32,
    missing_msbs: u32,
    total_bitplanes: u32,
    number_of_coding_passes: u32,
    sub_band_type: u32,
    style_flags: u32,
    strict: u32,
    dequantization_step: f32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kClassicSegment {
    data_offset: u32,
    data_length: u32,
    start_coding_pass: u32,
    end_coding_pass: u32,
    use_arithmetic: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kClassicStatus {
    code: u32,
    detail: u32,
    reserved0: u32,
    reserved1: u32,
}

#[cfg(target_os = "macos")]
const J2K_IDWT_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
const J2K_IDWT_STATUS_FAIL: u32 = 1;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kIdwtSingleDecompositionParams {
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
    ll_width: u32,
    ll_height: u32,
    hl_width: u32,
    hl_height: u32,
    lh_width: u32,
    lh_height: u32,
    hh_width: u32,
    hh_height: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kIdwtStatus {
    code: u32,
    detail: u32,
    reserved0: u32,
    reserved1: u32,
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
const J2K_MCT_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
#[allow(dead_code)]
const J2K_MCT_STATUS_FAIL: u32 = 1;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct J2kInverseMctParams {
    len: u32,
    transform: u32,
    addend0: f32,
    addend1: f32,
    addend2: f32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
#[allow(dead_code)]
struct J2kMctStatus {
    code: u32,
    detail: u32,
    reserved0: u32,
    reserved1: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kStoreParams {
    input_width: u32,
    source_x: u32,
    source_y: u32,
    copy_width: u32,
    copy_height: u32,
    output_width: u32,
    output_x: u32,
    output_y: u32,
    addend: f32,
}

const J2K_HT_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
const J2K_HT_STATUS_FAIL: u32 = 1;
#[cfg(target_os = "macos")]
const J2K_HT_STATUS_UNSUPPORTED: u32 = 2;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtCleanupParams {
    width: u32,
    height: u32,
    coded_len: u32,
    cleanup_length: u32,
    refinement_length: u32,
    missing_msbs: u32,
    num_bitplanes: u32,
    number_of_coding_passes: u32,
    output_stride: u32,
    output_offset: u32,
    dequantization_step: f32,
    stripe_causal: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtCleanupBatchJob {
    coded_offset: u32,
    width: u32,
    height: u32,
    coded_len: u32,
    cleanup_length: u32,
    refinement_length: u32,
    missing_msbs: u32,
    num_bitplanes: u32,
    number_of_coding_passes: u32,
    output_stride: u32,
    output_offset: u32,
    dequantization_step: f32,
    stripe_causal: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kHtStatus {
    code: u32,
    detail: u32,
    reserved0: u32,
    reserved1: u32,
}

#[cfg(target_os = "macos")]
static METAL_RUNTIME: OnceLock<Result<Arc<MetalRuntime>, String>> = OnceLock::new();

#[cfg(target_os = "macos")]
struct MetalRuntime {
    device: Device,
    queue: CommandQueue,
    pack_u8: ComputePipelineState,
    pack_u16: ComputePipelineState,
    classic_cleanup_batched: ComputePipelineState,
    idwt_interleave: ComputePipelineState,
    idwt_reversible53_horizontal: ComputePipelineState,
    idwt_reversible53_vertical: ComputePipelineState,
    idwt_irreversible97_single_decomposition: ComputePipelineState,
    #[allow(dead_code)]
    inverse_mct: ComputePipelineState,
    store_component: ComputePipelineState,
    ht_cleanup: ComputePipelineState,
    ht_cleanup_batched: ComputePipelineState,
    ht_vlc_table0: Buffer,
    ht_vlc_table1: Buffer,
    ht_uvlc_table0: Buffer,
    ht_uvlc_table1: Buffer,
}

#[cfg(target_os = "macos")]
impl MetalRuntime {
    fn new() -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or_else(|| "Metal is unavailable on this host".to_string())?;
        let options = CompileOptions::new();
        let library = device.new_library_with_source(SHADER_SOURCE, &options)?;
        let pack_u8_fn = library.get_function("j2k_pack_u8", None)?;
        let pack_u16_fn = library.get_function("j2k_pack_u16", None)?;
        let classic_cleanup_batched_fn =
            library.get_function("j2k_decode_classic_cleanup_batched", None)?;
        let idwt_interleave_fn = library.get_function("j2k_idwt_interleave", None)?;
        let idwt_reversible53_horizontal_fn =
            library.get_function("j2k_idwt_reversible53_horizontal_pass", None)?;
        let idwt_reversible53_vertical_fn =
            library.get_function("j2k_idwt_reversible53_vertical_pass", None)?;
        let idwt_irreversible97_single_decomposition_fn =
            library.get_function("j2k_idwt_irreversible97_single_decomposition", None)?;
        let inverse_mct_fn = library.get_function("j2k_inverse_mct", None)?;
        let store_component_fn = library.get_function("j2k_store_component", None)?;
        let ht_cleanup_fn = library.get_function("j2k_decode_ht_cleanup", None)?;
        let ht_cleanup_batched_fn = library.get_function("j2k_decode_ht_cleanup_batched", None)?;
        let pack_u8 = device.new_compute_pipeline_state_with_function(&pack_u8_fn)?;
        let pack_u16 = device.new_compute_pipeline_state_with_function(&pack_u16_fn)?;
        let classic_cleanup_batched =
            device.new_compute_pipeline_state_with_function(&classic_cleanup_batched_fn)?;
        let idwt_interleave =
            device.new_compute_pipeline_state_with_function(&idwt_interleave_fn)?;
        let idwt_reversible53_horizontal =
            device.new_compute_pipeline_state_with_function(&idwt_reversible53_horizontal_fn)?;
        let idwt_reversible53_vertical =
            device.new_compute_pipeline_state_with_function(&idwt_reversible53_vertical_fn)?;
        let idwt_irreversible97_single_decomposition = device
            .new_compute_pipeline_state_with_function(
                &idwt_irreversible97_single_decomposition_fn,
            )?;
        let inverse_mct = device.new_compute_pipeline_state_with_function(&inverse_mct_fn)?;
        let store_component =
            device.new_compute_pipeline_state_with_function(&store_component_fn)?;
        let ht_cleanup = device.new_compute_pipeline_state_with_function(&ht_cleanup_fn)?;
        let ht_cleanup_batched =
            device.new_compute_pipeline_state_with_function(&ht_cleanup_batched_fn)?;
        let queue = device.new_command_queue();
        Ok(Self {
            device: device.clone(),
            queue,
            pack_u8,
            pack_u16,
            classic_cleanup_batched,
            idwt_interleave,
            idwt_reversible53_horizontal,
            idwt_reversible53_vertical,
            idwt_irreversible97_single_decomposition,
            inverse_mct,
            store_component,
            ht_cleanup,
            ht_cleanup_batched,
            ht_vlc_table0: device.new_buffer_with_data(
                ht_vlc_table0().as_ptr().cast(),
                size_of_val(ht_vlc_table0()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_vlc_table1: device.new_buffer_with_data(
                ht_vlc_table1().as_ptr().cast(),
                size_of_val(ht_vlc_table1()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_uvlc_table0: device.new_buffer_with_data(
                ht_uvlc_table0().as_ptr().cast(),
                size_of_val(ht_uvlc_table0()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_uvlc_table1: device.new_buffer_with_data(
                ht_uvlc_table1().as_ptr().cast(),
                size_of_val(ht_uvlc_table1()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
        })
    }
}

#[cfg(target_os = "macos")]
fn with_runtime<R>(f: impl FnOnce(&MetalRuntime) -> Result<R, Error>) -> Result<R, Error> {
    match METAL_RUNTIME.get_or_init(|| MetalRuntime::new().map(Arc::new)) {
        Ok(runtime) => f(runtime),
        Err(message) => Err(Error::MetalKernel {
            message: message.clone(),
        }),
    }
}

#[cfg(target_os = "macos")]
enum DirectStatusCheck {
    Classic { buffer: Buffer, len: usize },
    Ht { buffer: Buffer, len: usize },
    Idwt(Buffer),
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
enum DirectHostRetention {
    Bytes(Vec<u8>),
    ClassicJobs(Vec<J2kClassicCleanupBatchJob>),
    ClassicSegments(Vec<J2kClassicSegment>),
    HtJobs(Vec<J2kHtCleanupBatchJob>),
}

#[cfg(target_os = "macos")]
struct PlaneStage {
    dims: (u32, u32),
    plane_count: usize,
    color_space: NativeColorSpace,
    has_alpha: bool,
    bit_depths: [u32; 4],
    planes: [Option<Buffer>; 4],
}

#[cfg(target_os = "macos")]
impl PlaneStage {
    #[allow(dead_code)]
    fn from_planes(
        device: &Device,
        decoded: &NativeDecodedComponents<'_>,
        roi: Option<Rect>,
    ) -> Result<Self, Error> {
        let full_dims = decoded.dimensions();
        let roi = roi.unwrap_or(Rect {
            x: 0,
            y: 0,
            w: full_dims.0,
            h: full_dims.1,
        });
        let dims = (roi.w, roi.h);
        let plane_count = decoded.planes().len();
        if plane_count == 0 || plane_count > 4 {
            return Err(Error::MetalKernel {
                message: format!("unsupported J2K plane count {plane_count}"),
            });
        }

        let mut bit_depths = [0u32; 4];
        let mut planes: [Option<Buffer>; 4] = [None, None, None, None];
        for (index, plane) in decoded.planes().iter().enumerate() {
            bit_depths[index] = u32::from(plane.bit_depth());
            let len = dims.0 as usize * dims.1 as usize;
            let buffer = device.new_buffer(
                (len * size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            copy_plane_samples(&buffer, plane.samples(), full_dims.0 as usize, roi);
            planes[index] = Some(buffer);
        }

        Ok(Self {
            dims,
            plane_count,
            color_space: decoded.color_space().clone(),
            has_alpha: decoded.has_alpha(),
            bit_depths,
            planes,
        })
    }

    #[allow(dead_code)]
    fn from_captured_planes(
        decoded: &NativeDecodedComponents<'_>,
        captured_planes: Vec<Buffer>,
    ) -> Option<Self> {
        let plane_count = decoded.planes().len();
        let supported_shape = matches!(
            (decoded.color_space(), decoded.has_alpha(), plane_count),
            (NativeColorSpace::Gray, false, 1) | (NativeColorSpace::RGB, false, 3)
        );
        if !supported_shape {
            return None;
        }
        if captured_planes.len() != plane_count || plane_count == 0 || plane_count > 4 {
            return None;
        }

        let mut bit_depths = [0u32; 4];
        let mut planes: [Option<Buffer>; 4] = [None, None, None, None];
        for (index, (plane, buffer)) in decoded
            .planes()
            .iter()
            .zip(captured_planes.into_iter())
            .enumerate()
        {
            bit_depths[index] = u32::from(plane.bit_depth());
            planes[index] = Some(buffer);
        }

        Some(Self {
            dims: decoded.dimensions(),
            plane_count,
            color_space: decoded.color_space().clone(),
            has_alpha: decoded.has_alpha(),
            bit_depths,
            planes,
        })
    }

    fn finish_with_runtime(
        self,
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> Result<Surface, Error> {
        let pitch_bytes = self.dims.0 as usize * fmt.bytes_per_pixel();
        let out_buffer = runtime.device.new_buffer(
            (pitch_bytes * self.dims.1 as usize) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let (output_channels, opaque_alpha, pipeline) = output_shape_for(
            &self.color_space,
            self.has_alpha,
            self.plane_count,
            fmt,
            runtime,
        )?;

        let params = J2kPackParams {
            width: self.dims.0,
            height: self.dims.1,
            out_stride: u32::try_from(pitch_bytes).expect("J2K Metal output stride fits in u32"),
            plane_count: u32::try_from(self.plane_count).expect("J2K plane count fits in u32"),
            output_channels,
            opaque_alpha,
            bit_depths: self.bit_depths,
        };

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(
            0,
            self.planes[0].as_ref().map(std::convert::AsRef::as_ref),
            0,
        );
        encoder.set_buffer(
            1,
            self.planes[1].as_ref().map(std::convert::AsRef::as_ref),
            0,
        );
        encoder.set_buffer(
            2,
            self.planes[2].as_ref().map(std::convert::AsRef::as_ref),
            0,
        );
        encoder.set_buffer(
            3,
            self.planes[3].as_ref().map(std::convert::AsRef::as_ref),
            0,
        );
        encoder.set_buffer(4, Some(&out_buffer), 0);
        encoder.set_bytes(
            5,
            size_of::<J2kPackParams>() as u64,
            (&raw const params).cast(),
        );

        let width = pipeline.thread_execution_width().max(1);
        let max_threads = pipeline.max_total_threads_per_threadgroup().max(width);
        let height = (max_threads / width).max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(self.dims.0),
                height: u64::from(self.dims.1),
                depth: 1,
            },
            MTLSize {
                width,
                height,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(Surface::from_metal_buffer(out_buffer, self.dims, fmt))
    }
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
pub(crate) fn pack_gray_plane_to_surface(
    plane: Buffer,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let mut bit_depths = [0u32; 4];
        bit_depths[0] = u32::from(bit_depth);
        PlaneStage {
            dims,
            plane_count: 1,
            color_space: NativeColorSpace::Gray,
            has_alpha: false,
            bit_depths,
            planes: [Some(plane), None, None, None],
        }
        .finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
fn encode_direct_grayscale_plan_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &J2kDirectGrayscalePlan,
    fmt: PixelFormat,
    retained_buffers: &mut Vec<Buffer>,
    retained_host_data: &mut Vec<DirectHostRetention>,
    status_checks: &mut Vec<DirectStatusCheck>,
) -> Result<Surface, Error> {
    let mut bands: Vec<(J2kDirectBandId, Buffer)> = Vec::new();
    let mut final_surface = None;

    for step in &plan.steps {
        match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                let output = alloc_f32_buffer(
                    &runtime.device,
                    sub_band.width as usize * sub_band.height as usize,
                );
                let (host_data, buffers, status_check) =
                    encode_classic_sub_band_to_buffer_in_command_buffer(
                        runtime,
                        command_buffer,
                        sub_band,
                        &output,
                    )?;
                retained_host_data.extend(host_data);
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                bands.push((sub_band.band_id, output));
            }
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                let output = alloc_f32_buffer(
                    &runtime.device,
                    sub_band.width as usize * sub_band.height as usize,
                );
                let (host_data, buffers, status_check) =
                    encode_ht_sub_band_to_buffer_in_command_buffer(
                        runtime,
                        command_buffer,
                        sub_band,
                        &output,
                    )?;
                retained_host_data.extend(host_data);
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                bands.push((sub_band.band_id, output));
            }
            J2kDirectGrayscaleStep::Idwt(idwt) => {
                let ll = lookup_direct_band(&bands, idwt.ll_band_id, idwt.ll)?;
                let hl = lookup_direct_band(&bands, idwt.hl_band_id, idwt.hl)?;
                let lh = lookup_direct_band(&bands, idwt.lh_band_id, idwt.lh)?;
                let hh = lookup_direct_band(&bands, idwt.hh_band_id, idwt.hh)?;
                let params = J2kIdwtSingleDecompositionParams {
                    x0: idwt.rect.x0,
                    y0: idwt.rect.y0,
                    width: idwt.rect.width(),
                    height: idwt.rect.height(),
                    ll_width: idwt.ll.width(),
                    ll_height: idwt.ll.height(),
                    hl_width: idwt.hl.width(),
                    hl_height: idwt.hl.height(),
                    lh_width: idwt.lh.width(),
                    lh_height: idwt.lh.height(),
                    hh_width: idwt.hh.width(),
                    hh_height: idwt.hh.height(),
                };
                let output = alloc_f32_buffer(
                    &runtime.device,
                    idwt.rect.width() as usize * idwt.rect.height() as usize,
                );
                match idwt.transform {
                    J2kWaveletTransform::Reversible53 => {
                        dispatch_reversible53_single_decomposition_buffers_in_command_buffer(
                            runtime,
                            command_buffer,
                            &ll,
                            &hl,
                            &lh,
                            &hh,
                            params,
                            &output,
                        );
                    }
                    J2kWaveletTransform::Irreversible97 => {
                        let status_check =
                            dispatch_irreversible97_single_decomposition_buffers_in_command_buffer(
                                runtime,
                                command_buffer,
                                &ll,
                                &hl,
                                &lh,
                                &hh,
                                params,
                                &output,
                            );
                        status_checks.push(status_check);
                    }
                }
                bands.push((idwt.output_band_id, output));
            }
            J2kDirectGrayscaleStep::Store(store) => {
                let input = lookup_direct_band(&bands, store.input_band_id, store.input_rect)?;
                let output = alloc_f32_buffer(
                    &runtime.device,
                    store.output_width as usize * store.output_height as usize,
                );
                let params = J2kStoreParams {
                    input_width: store.input_rect.width(),
                    source_x: store.source_x,
                    source_y: store.source_y,
                    copy_width: store.copy_width,
                    copy_height: store.copy_height,
                    output_width: store.output_width,
                    output_x: store.output_x,
                    output_y: store.output_y,
                    addend: store.addend,
                };
                dispatch_store_component_buffer_in_command_buffer(
                    runtime,
                    command_buffer,
                    &input,
                    &output,
                    params,
                );
                retained_buffers.push(output.clone());

                final_surface = Some(encode_gray_plane_to_surface_in_command_buffer(
                    runtime,
                    command_buffer,
                    &output,
                    plan.dimensions,
                    plan.bit_depth,
                    fmt,
                )?);
            }
        }
    }

    let surface = final_surface.ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect grayscale plan did not produce a final stored plane".to_string(),
    })?;

    Ok(surface)
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_direct_grayscale_plan(
    plan: &J2kDirectGrayscalePlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut retained_host_data = Vec::new();
        let mut status_checks = Vec::new();
        let surface = encode_direct_grayscale_plan_in_command_buffer(
            runtime,
            command_buffer,
            plan,
            fmt,
            &mut retained_buffers,
            &mut retained_host_data,
            &mut status_checks,
        )?;
        command_buffer.commit();
        command_buffer.wait_until_completed();
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        drop(retained_host_data);
        Ok(surface)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_repeated_direct_grayscale_plan(
    plan: &J2kDirectGrayscalePlan,
    fmt: PixelFormat,
    count: usize,
) -> Result<Vec<Surface>, Error> {
    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut retained_host_data = Vec::new();
        let mut status_checks = Vec::new();
        let mut surfaces = Vec::with_capacity(count);
        for _ in 0..count {
            surfaces.push(encode_direct_grayscale_plan_in_command_buffer(
                runtime,
                command_buffer,
                plan,
                fmt,
                &mut retained_buffers,
                &mut retained_host_data,
                &mut status_checks,
            )?);
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        drop(retained_host_data);
        Ok(surfaces)
    })
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn copy_plane_samples(buffer: &Buffer, samples: &[f32], image_width: usize, roi: Rect) {
    let row_width = roi.w as usize;
    let dst = unsafe {
        core::slice::from_raw_parts_mut(buffer.contents().cast::<f32>(), row_width * roi.h as usize)
    };

    for row in 0..roi.h as usize {
        let src_start = (roi.y as usize + row) * image_width + roi.x as usize;
        let src_end = src_start + row_width;
        let dst_start = row * row_width;
        dst[dst_start..dst_start + row_width].copy_from_slice(&samples[src_start..src_end]);
    }
}

#[cfg(target_os = "macos")]
fn alloc_f32_buffer(device: &Device, len: usize) -> Buffer {
    let bytes = len.max(1).saturating_mul(size_of::<f32>());
    device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
}

#[cfg(target_os = "macos")]
fn validate_direct_status(status_check: DirectStatusCheck) -> Result<(), Error> {
    match status_check {
        DirectStatusCheck::Classic { buffer, len } => {
            let statuses = unsafe {
                core::slice::from_raw_parts(buffer.contents().cast::<J2kClassicStatus>(), len)
            };
            if let Some(status) = statuses
                .iter()
                .copied()
                .find(|status| status.code != J2K_CLASSIC_STATUS_OK)
            {
                return Err(decode_classic_status_error(status));
            }
        }
        DirectStatusCheck::Ht { buffer, len } => {
            let statuses = unsafe {
                core::slice::from_raw_parts(buffer.contents().cast::<J2kHtStatus>(), len)
            };
            if let Some(status) = statuses
                .iter()
                .copied()
                .find(|status| status.code != J2K_HT_STATUS_OK)
            {
                return Err(decode_ht_status_error(status));
            }
        }
        DirectStatusCheck::Idwt(buffer) => {
            let status = unsafe { buffer.contents().cast::<J2kIdwtStatus>().read() };
            if status.code != J2K_IDWT_STATUS_OK {
                return Err(decode_idwt_status_error(status));
            }
        }
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn encode_gray_plane_to_surface_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plane: &Buffer,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let pitch_bytes = dims.0 as usize * fmt.bytes_per_pixel();
    let out_buffer = runtime.device.new_buffer(
        (pitch_bytes * dims.1 as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let (output_channels, opaque_alpha, pipeline) =
        output_shape_for(&NativeColorSpace::Gray, false, 1, fmt, runtime)?;
    let mut bit_depths = [0u32; 4];
    bit_depths[0] = u32::from(bit_depth);
    let params = J2kPackParams {
        width: dims.0,
        height: dims.1,
        out_stride: u32::try_from(pitch_bytes).expect("J2K Metal output stride fits in u32"),
        plane_count: 1,
        output_channels,
        opaque_alpha,
        bit_depths,
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(plane), 0);
    encoder.set_buffer(1, None, 0);
    encoder.set_buffer(2, None, 0);
    encoder.set_buffer(3, None, 0);
    encoder.set_buffer(4, Some(&out_buffer), 0);
    encoder.set_bytes(
        5,
        size_of::<J2kPackParams>() as u64,
        (&raw const params).cast(),
    );
    let width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(width);
    let height = (max_threads / width).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(dims.0),
            height: u64::from(dims.1),
            depth: 1,
        },
        MTLSize {
            width,
            height,
            depth: 1,
        },
    );
    encoder.end_encoding();

    Ok(Surface::from_metal_buffer(out_buffer, dims, fmt))
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn owned_slice_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    let size = size_of_val(data).max(1);
    let buffer = device.new_buffer(size as u64, MTLResourceOptions::StorageModeShared);
    if !data.is_empty() {
        unsafe {
            core::ptr::copy_nonoverlapping(
                data.as_ptr().cast::<u8>(),
                buffer.contents().cast::<u8>(),
                size_of_val(data),
            );
        }
    }
    buffer
}

#[cfg(target_os = "macos")]
fn lookup_direct_band(
    bands: &[(J2kDirectBandId, Buffer)],
    band_id: J2kDirectBandId,
    rect: slidecodec_j2k_native::J2kRect,
) -> Result<Buffer, Error> {
    bands
        .iter()
        .find(|(existing, _)| *existing == band_id)
        .map(|(_, buffer)| buffer.to_owned())
        .ok_or_else(|| Error::MetalKernel {
            message: format!(
                "missing J2K MetalDirect device band {} for rect ({}, {}, {}, {})",
                band_id, rect.x0, rect.y0, rect.x1, rect.y1
            ),
        })
}

#[cfg(target_os = "macos")]
fn output_shape_for<'a>(
    color_space: &NativeColorSpace,
    has_alpha: bool,
    plane_count: usize,
    fmt: PixelFormat,
    runtime: &'a MetalRuntime,
) -> Result<(u32, u32, &'a ComputePipelineState), Error> {
    match (color_space, has_alpha, plane_count, fmt) {
        (NativeColorSpace::Gray, false, 1, PixelFormat::Gray8) => Ok((1, 0, &runtime.pack_u8)),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgb8)
        | (NativeColorSpace::RGB, true, 4, PixelFormat::Rgb8) => Ok((3, 0, &runtime.pack_u8)),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgba8) => Ok((4, 1, &runtime.pack_u8)),
        (NativeColorSpace::RGB, true, 4, PixelFormat::Rgba8) => Ok((4, 0, &runtime.pack_u8)),
        (NativeColorSpace::Gray, false, 1, PixelFormat::Gray16) => Ok((1, 0, &runtime.pack_u16)),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgb16) => Ok((3, 0, &runtime.pack_u16)),
        _ => Err(Error::MetalKernel {
            message: format!(
                "unsupported J2K Metal mapping for {color_space:?}, alpha={has_alpha}, planes={plane_count}, fmt={fmt:?}"
            ),
        }),
    }
}

#[cfg(target_os = "macos")]
fn required_classic_output_len(job: J2kCodeBlockDecodeJob<'_>) -> Result<usize, Error> {
    if job.height == 0 {
        return Ok(0);
    }

    job.output_stride
        .checked_mul(job.height as usize - 1)
        .and_then(|prefix| prefix.checked_add(job.width as usize))
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal output size overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
fn classic_style_flags(style: slidecodec_j2k_native::J2kCodeBlockStyle) -> u32 {
    let mut flags = 0u32;
    if style.reset_context_probabilities {
        flags |= J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES;
    }
    if style.termination_on_each_pass {
        flags |= J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS;
    }
    if style.vertically_causal_context {
        flags |= J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT;
    }
    if style.segmentation_symbols {
        flags |= J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS;
    }
    if style.selective_arithmetic_coding_bypass {
        flags |= J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS;
    }
    flags
}

#[cfg(target_os = "macos")]
fn decode_classic_status_error(status: J2kClassicStatus) -> Error {
    let kind = match status.code {
        J2K_CLASSIC_STATUS_FAIL => "decode failure",
        J2K_CLASSIC_STATUS_UNSUPPORTED => "unsupported classic kernel input",
        _ => "unexpected classic kernel status",
    };
    Error::MetalKernel {
        message: format!("classic J2K Metal kernel {kind} (detail={})", status.detail),
    }
}

#[cfg(target_os = "macos")]
fn decode_idwt_status_error(status: J2kIdwtStatus) -> Error {
    let kind = match status.code {
        J2K_IDWT_STATUS_FAIL => "decode failure",
        _ => "unexpected IDWT kernel status",
    };
    Error::MetalKernel {
        message: format!("J2K Metal IDWT kernel {kind} (detail={})", status.detail),
    }
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn decode_mct_status_error(status: J2kMctStatus) -> Error {
    let kind = match status.code {
        J2K_MCT_STATUS_FAIL => "decode failure",
        _ => "unexpected inverse MCT kernel status",
    };
    Error::MetalKernel {
        message: format!(
            "J2K Metal inverse MCT kernel {kind} (detail={})",
            status.detail
        ),
    }
}

fn wrap_f32_output_buffer(device: &Device, output: &mut [f32]) -> Buffer {
    if output.is_empty() {
        device.new_buffer(
            size_of::<f32>() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    } else {
        device.new_buffer_with_bytes_no_copy(
            output.as_mut_ptr().cast(),
            size_of_val(output) as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }
}

#[cfg(target_os = "macos")]
fn borrow_slice_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    if data.is_empty() {
        device.new_buffer(1, MTLResourceOptions::StorageModeShared)
    } else {
        device.new_buffer_with_bytes_no_copy(
            data.as_ptr().cast(),
            size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
pub(crate) fn decode_inverse_mct(job: J2kInverseMctJob<'_>) -> Result<Vec<Buffer>, Error> {
    let J2kInverseMctJob {
        transform,
        plane0,
        plane1,
        plane2,
        addend0,
        addend1,
        addend2,
    } = job;
    with_runtime(|runtime| {
        let len = plane0.len();
        if len == 0 {
            return Ok(Vec::new());
        }
        if plane1.len() != len || plane2.len() != len {
            return Err(Error::MetalKernel {
                message: "J2K Metal inverse MCT plane lengths must match".to_string(),
            });
        }

        let transform = match transform {
            J2kWaveletTransform::Reversible53 => 0,
            J2kWaveletTransform::Irreversible97 => 1,
        };
        let params = J2kInverseMctParams {
            len: u32::try_from(len).map_err(|_| Error::MetalKernel {
                message: "J2K Metal inverse MCT plane length exceeds u32".to_string(),
            })?,
            transform,
            addend0,
            addend1,
            addend2,
        };
        let plane0_buffer = borrow_slice_buffer(&runtime.device, plane0);
        let plane1_buffer = borrow_slice_buffer(&runtime.device, plane1);
        let plane2_buffer = borrow_slice_buffer(&runtime.device, plane2);
        let status = J2kMctStatus::default();
        let status_buffer = runtime.device.new_buffer_with_data(
            (&raw const status).cast(),
            size_of::<J2kMctStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.inverse_mct);
        encoder.set_buffer(0, Some(&plane0_buffer), 0);
        encoder.set_buffer(1, Some(&plane1_buffer), 0);
        encoder.set_buffer(2, Some(&plane2_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kInverseMctParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(4, Some(&status_buffer), 0);
        let width = runtime.inverse_mct.thread_execution_width().max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: len as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = unsafe { status_buffer.contents().cast::<J2kMctStatus>().read() };
        if status.code != J2K_MCT_STATUS_OK {
            return Err(decode_mct_status_error(status));
        }

        let plane0_host =
            unsafe { core::slice::from_raw_parts(plane0_buffer.contents().cast::<f32>(), len) };
        let plane1_host =
            unsafe { core::slice::from_raw_parts(plane1_buffer.contents().cast::<f32>(), len) };
        let plane2_host =
            unsafe { core::slice::from_raw_parts(plane2_buffer.contents().cast::<f32>(), len) };
        for (dst, sample) in plane0.iter_mut().zip(plane0_host.iter().copied()) {
            *dst = sample - addend0;
        }
        for (dst, sample) in plane1.iter_mut().zip(plane1_host.iter().copied()) {
            *dst = sample - addend1;
        }
        for (dst, sample) in plane2.iter_mut().zip(plane2_host.iter().copied()) {
            *dst = sample - addend2;
        }
        Ok(vec![plane0_buffer, plane1_buffer, plane2_buffer])
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_store_component_and_capture(
    job: J2kStoreComponentJob<'_>,
) -> Result<Buffer, Error> {
    let J2kStoreComponentJob {
        input,
        input_width,
        source_x,
        source_y,
        copy_width,
        copy_height,
        output,
        output_width,
        output_x,
        output_y,
        addend,
    } = job;
    with_runtime(|runtime| {
        if copy_width == 0 || copy_height == 0 {
            return Ok(wrap_f32_output_buffer(&runtime.device, output));
        }

        let required_input_height =
            source_y
                .checked_add(copy_height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal store source height overflow".to_string(),
                })?;
        let required_output_height =
            output_y
                .checked_add(copy_height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal store destination height overflow".to_string(),
                })?;
        if source_x
            .checked_add(copy_width)
            .is_none_or(|end| end > input_width)
            || output_x
                .checked_add(copy_width)
                .is_none_or(|end| end > output_width)
        {
            return Err(Error::MetalKernel {
                message: "J2K Metal store copy rectangle exceeds row bounds".to_string(),
            });
        }
        if input.len()
            < input_width as usize
                * usize::try_from(required_input_height).map_err(|_| Error::MetalKernel {
                    message: "J2K Metal store source height exceeds usize".to_string(),
                })?
            || output.len()
                < output_width as usize
                    * usize::try_from(required_output_height).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal store destination height exceeds usize".to_string(),
                    })?
        {
            return Err(Error::MetalKernel {
                message: "J2K Metal store buffers are smaller than required".to_string(),
            });
        }

        let params = J2kStoreParams {
            input_width,
            source_x,
            source_y,
            copy_width,
            copy_height,
            output_width,
            output_x,
            output_y,
            addend,
        };
        let input_buffer = borrow_slice_buffer(&runtime.device, input);
        let output_buffer = wrap_f32_output_buffer(&runtime.device, output);
        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.store_component);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            size_of::<J2kStoreParams>() as u64,
            (&raw const params).cast(),
        );
        let width = runtime.store_component.thread_execution_width().max(1);
        let max_threads = runtime
            .store_component
            .max_total_threads_per_threadgroup()
            .max(width);
        let height = (max_threads / width).max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(copy_width),
                height: u64::from(copy_height),
                depth: 1,
            },
            MTLSize {
                width,
                height,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(output_buffer)
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::unnecessary_wraps)]
#[allow(dead_code)]
fn dispatch_store_component_buffer(
    runtime: &MetalRuntime,
    input: &Buffer,
    output: &Buffer,
    params: J2kStoreParams,
) -> Result<(), Error> {
    let command_buffer = runtime.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.store_component);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kStoreParams>() as u64,
        (&raw const params).cast(),
    );
    let width = runtime.store_component.thread_execution_width().max(1);
    let max_threads = runtime
        .store_component
        .max_total_threads_per_threadgroup()
        .max(width);
    let height = (max_threads / width).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.copy_width),
            height: u64::from(params.copy_height),
            depth: 1,
        },
        MTLSize {
            width,
            height,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}

#[cfg(target_os = "macos")]
fn dispatch_store_component_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    output: &Buffer,
    params: J2kStoreParams,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.store_component);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kStoreParams>() as u64,
        (&raw const params).cast(),
    );
    let width = runtime.store_component.thread_execution_width().max(1);
    let max_threads = runtime
        .store_component
        .max_total_threads_per_threadgroup()
        .max(width);
    let height = (max_threads / width).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.copy_width),
            height: u64::from(params.copy_height),
            depth: 1,
        },
        MTLSize {
            width,
            height,
            depth: 1,
        },
    );
    encoder.end_encoding();
}

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
            width: job.rect.width(),
            height: job.rect.height(),
            ll_width: job.ll.rect.width(),
            ll_height: job.ll.rect.height(),
            hl_width: job.hl.rect.width(),
            hl_height: job.hl.rect.height(),
            lh_width: job.lh.rect.width(),
            lh_height: job.lh.rect.height(),
            hh_width: job.hh.rect.width(),
            hh_height: job.hh.rect.height(),
        };

        let ll = borrow_slice_buffer(&runtime.device, job.ll.coefficients);
        let hl = borrow_slice_buffer(&runtime.device, job.hl.coefficients);
        let lh = borrow_slice_buffer(&runtime.device, job.lh.coefficients);
        let hh = borrow_slice_buffer(&runtime.device, job.hh.coefficients);
        let decoded = wrap_f32_output_buffer(&runtime.device, output);

        let command_buffer = runtime.queue.new_command_buffer();

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.idwt_interleave);
        encoder.set_buffer(0, Some(&ll), 0);
        encoder.set_buffer(1, Some(&hl), 0);
        encoder.set_buffer(2, Some(&lh), 0);
        encoder.set_buffer(3, Some(&hh), 0);
        encoder.set_buffer(4, Some(&decoded), 0);
        encoder.set_bytes(
            5,
            size_of::<J2kIdwtSingleDecompositionParams>() as u64,
            (&raw const params).cast(),
        );
        let interleave_width = runtime.idwt_interleave.thread_execution_width().max(1);
        let interleave_height = (runtime
            .idwt_interleave
            .max_total_threads_per_threadgroup()
            .max(interleave_width)
            / interleave_width)
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(params.width),
                height: u64::from(params.height),
                depth: 1,
            },
            MTLSize {
                width: interleave_width,
                height: interleave_height,
                depth: 1,
            },
        );
        encoder.end_encoding();

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_horizontal);
        encoder.set_buffer(0, Some(&decoded), 0);
        encoder.set_bytes(
            1,
            size_of::<J2kIdwtSingleDecompositionParams>() as u64,
            (&raw const params).cast(),
        );
        let horizontal_width = runtime
            .idwt_reversible53_horizontal
            .thread_execution_width()
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(params.height),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: horizontal_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_vertical);
        encoder.set_buffer(0, Some(&decoded), 0);
        encoder.set_bytes(
            1,
            size_of::<J2kIdwtSingleDecompositionParams>() as u64,
            (&raw const params).cast(),
        );
        let vertical_width = runtime
            .idwt_reversible53_vertical
            .thread_execution_width()
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(params.width),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: vertical_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::unnecessary_wraps)]
#[allow(dead_code)]
fn dispatch_reversible53_single_decomposition_buffers(
    runtime: &MetalRuntime,
    ll: &Buffer,
    hl: &Buffer,
    lh: &Buffer,
    hh: &Buffer,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
) -> Result<(), Error> {
    let command_buffer = runtime.queue.new_command_buffer();

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.idwt_interleave);
    encoder.set_buffer(0, Some(ll), 0);
    encoder.set_buffer(1, Some(hl), 0);
    encoder.set_buffer(2, Some(lh), 0);
    encoder.set_buffer(3, Some(hh), 0);
    encoder.set_buffer(4, Some(decoded), 0);
    encoder.set_bytes(
        5,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let interleave_width = runtime.idwt_interleave.thread_execution_width().max(1);
    let interleave_height = (runtime
        .idwt_interleave
        .max_total_threads_per_threadgroup()
        .max(interleave_width)
        / interleave_width)
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.width),
            height: u64::from(params.height),
            depth: 1,
        },
        MTLSize {
            width: interleave_width,
            height: interleave_height,
            depth: 1,
        },
    );
    encoder.end_encoding();

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_horizontal);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_bytes(
        1,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let horizontal_width = runtime
        .idwt_reversible53_horizontal
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.height),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: horizontal_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_vertical);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_bytes(
        1,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let vertical_width = runtime
        .idwt_reversible53_vertical
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.width),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: vertical_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_reversible53_single_decomposition_buffers_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    ll: &Buffer,
    hl: &Buffer,
    lh: &Buffer,
    hh: &Buffer,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.idwt_interleave);
    encoder.set_buffer(0, Some(ll), 0);
    encoder.set_buffer(1, Some(hl), 0);
    encoder.set_buffer(2, Some(lh), 0);
    encoder.set_buffer(3, Some(hh), 0);
    encoder.set_buffer(4, Some(decoded), 0);
    encoder.set_bytes(
        5,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let interleave_width = runtime.idwt_interleave.thread_execution_width().max(1);
    let interleave_height = (runtime
        .idwt_interleave
        .max_total_threads_per_threadgroup()
        .max(interleave_width)
        / interleave_width)
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.width),
            height: u64::from(params.height),
            depth: 1,
        },
        MTLSize {
            width: interleave_width,
            height: interleave_height,
            depth: 1,
        },
    );
    encoder.end_encoding();

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_horizontal);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_bytes(
        1,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let horizontal_width = runtime
        .idwt_reversible53_horizontal
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.height),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: horizontal_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_vertical);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_bytes(
        1,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let vertical_width = runtime
        .idwt_reversible53_vertical
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.width),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: vertical_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_irreversible97_single_decomposition_idwt(
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
            width: job.rect.width(),
            height: job.rect.height(),
            ll_width: job.ll.rect.width(),
            ll_height: job.ll.rect.height(),
            hl_width: job.hl.rect.width(),
            hl_height: job.hl.rect.height(),
            lh_width: job.lh.rect.width(),
            lh_height: job.lh.rect.height(),
            hh_width: job.hh.rect.width(),
            hh_height: job.hh.rect.height(),
        };

        let ll = borrow_slice_buffer(&runtime.device, job.ll.coefficients);
        let hl = borrow_slice_buffer(&runtime.device, job.hl.coefficients);
        let lh = borrow_slice_buffer(&runtime.device, job.lh.coefficients);
        let hh = borrow_slice_buffer(&runtime.device, job.hh.coefficients);
        let decoded = wrap_f32_output_buffer(&runtime.device, output);
        let status_buffer = runtime.device.new_buffer(
            size_of::<J2kIdwtStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.idwt_irreversible97_single_decomposition);
        encoder.set_buffer(0, Some(&ll), 0);
        encoder.set_buffer(1, Some(&hl), 0);
        encoder.set_buffer(2, Some(&lh), 0);
        encoder.set_buffer(3, Some(&hh), 0);
        encoder.set_buffer(4, Some(&decoded), 0);
        encoder.set_bytes(
            5,
            size_of::<J2kIdwtSingleDecompositionParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(6, Some(&status_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = unsafe { status_buffer.contents().cast::<J2kIdwtStatus>().read() };
        if status.code != J2K_IDWT_STATUS_OK {
            return Err(decode_idwt_status_error(status));
        }
        Ok(())
    })
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn dispatch_irreversible97_single_decomposition_buffers(
    runtime: &MetalRuntime,
    ll: &Buffer,
    hl: &Buffer,
    lh: &Buffer,
    hh: &Buffer,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
) -> Result<(), Error> {
    let status_buffer = runtime.device.new_buffer(
        size_of::<J2kIdwtStatus>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = runtime.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.idwt_irreversible97_single_decomposition);
    encoder.set_buffer(0, Some(ll), 0);
    encoder.set_buffer(1, Some(hl), 0);
    encoder.set_buffer(2, Some(lh), 0);
    encoder.set_buffer(3, Some(hh), 0);
    encoder.set_buffer(4, Some(decoded), 0);
    encoder.set_bytes(
        5,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(6, Some(&status_buffer), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let status = unsafe { status_buffer.contents().cast::<J2kIdwtStatus>().read() };
    if status.code != J2K_IDWT_STATUS_OK {
        return Err(decode_idwt_status_error(status));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_irreversible97_single_decomposition_buffers_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    ll: &Buffer,
    hl: &Buffer,
    lh: &Buffer,
    hh: &Buffer,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
) -> DirectStatusCheck {
    let status_buffer = runtime.device.new_buffer(
        size_of::<J2kIdwtStatus>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.idwt_irreversible97_single_decomposition);
    encoder.set_buffer(0, Some(ll), 0);
    encoder.set_buffer(1, Some(hl), 0);
    encoder.set_buffer(2, Some(lh), 0);
    encoder.set_buffer(3, Some(hh), 0);
    encoder.set_buffer(4, Some(decoded), 0);
    encoder.set_bytes(
        5,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(6, Some(&status_buffer), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    DirectStatusCheck::Idwt(status_buffer)
}

#[cfg(target_os = "macos")]
fn dispatch_classic_cleanup_batched(
    runtime: &MetalRuntime,
    coded_data: &[u8],
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
    decoded: &Buffer,
) -> Result<(), Error> {
    let input = borrow_slice_buffer(&runtime.device, coded_data);
    let jobs_buffer = borrow_slice_buffer(&runtime.device, jobs);
    let segments_buffer = borrow_slice_buffer(&runtime.device, segments);
    let status_buffer = runtime.device.new_buffer(
        (jobs.len().max(1) * size_of::<J2kClassicStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = runtime.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.classic_cleanup_batched);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(&jobs_buffer), 0);
    encoder.set_buffer(3, Some(&segments_buffer), 0);
    encoder.set_buffer(4, Some(&status_buffer), 0);
    let width = runtime
        .classic_cleanup_batched
        .thread_execution_width()
        .max(1)
        .min(jobs.len() as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: jobs.len() as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let statuses = unsafe {
        core::slice::from_raw_parts(
            status_buffer.contents().cast::<J2kClassicStatus>(),
            jobs.len(),
        )
    };
    if let Some(status) = statuses
        .iter()
        .copied()
        .find(|status| status.code != J2K_CLASSIC_STATUS_OK)
    {
        return Err(decode_classic_status_error(status));
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn dispatch_classic_cleanup_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    segments: &Buffer,
    decoded: &Buffer,
) -> DirectStatusCheck {
    let status_buffer = runtime.device.new_buffer(
        (job_count.max(1) * size_of::<J2kClassicStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.classic_cleanup_batched);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(segments), 0);
    encoder.set_buffer(4, Some(&status_buffer), 0);
    let width = runtime
        .classic_cleanup_batched
        .thread_execution_width()
        .max(1)
        .min(job_count as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: job_count as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    DirectStatusCheck::Classic {
        buffer: status_buffer,
        len: job_count,
    }
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn encode_classic_sub_band_to_buffer(
    runtime: &MetalRuntime,
    job: &slidecodec_j2k_native::J2kOwnedSubBandPlan,
    output: &Buffer,
) -> Result<(), Error> {
    if job.jobs.is_empty() {
        return Ok(());
    }

    let mut jobs = Vec::with_capacity(job.jobs.len());
    let mut coded_data = Vec::new();
    let mut segments = Vec::new();

    for block in &job.jobs {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(&block.data);
        let segment_offset = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect segment table exceeds u32".to_string(),
        })?;
        for segment in &block.segments {
            let data_offset = coded_offset
                .checked_add(segment.data_offset)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect segment offset overflow".to_string(),
                })?;
            segments.push(J2kClassicSegment {
                data_offset,
                data_length: segment.data_length,
                start_coding_pass: u32::from(segment.start_coding_pass),
                end_coding_pass: u32::from(segment.end_coding_pass),
                use_arithmetic: u32::from(segment.use_arithmetic),
            });
        }
        jobs.push(J2kClassicCleanupBatchJob {
            coded_offset,
            coded_len: u32::try_from(block.data.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
            })?,
            segment_offset,
            segment_count: u32::try_from(block.segments.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K MetalDirect segment count exceeds u32".to_string(),
            })?,
            width: block.width,
            height: block.height,
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect output offset overflow".to_string(),
                })?,
            missing_msbs: u32::from(block.missing_bit_planes),
            total_bitplanes: u32::from(block.total_bitplanes),
            number_of_coding_passes: u32::from(block.number_of_coding_passes),
            sub_band_type: match block.sub_band_type {
                slidecodec_j2k_native::J2kSubBandType::LowLow => 0,
                slidecodec_j2k_native::J2kSubBandType::HighLow => 1,
                slidecodec_j2k_native::J2kSubBandType::LowHigh => 2,
                slidecodec_j2k_native::J2kSubBandType::HighHigh => 3,
            },
            style_flags: classic_style_flags(block.style),
            strict: u32::from(block.strict),
            dequantization_step: block.dequantization_step,
        });
    }

    dispatch_classic_cleanup_batched(runtime, &coded_data, &jobs, &segments, output)
}

#[cfg(target_os = "macos")]
fn encode_classic_sub_band_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: &slidecodec_j2k_native::J2kOwnedSubBandPlan,
    output: &Buffer,
) -> Result<(Vec<DirectHostRetention>, Vec<Buffer>, DirectStatusCheck), Error> {
    if job.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            Vec::new(),
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let mut jobs = Vec::with_capacity(job.jobs.len());
    let mut coded_data = Vec::new();
    let mut segments = Vec::new();

    for block in &job.jobs {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(&block.data);
        let segment_offset = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect segment table exceeds u32".to_string(),
        })?;
        for segment in &block.segments {
            let data_offset = coded_offset
                .checked_add(segment.data_offset)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect segment offset overflow".to_string(),
                })?;
            segments.push(J2kClassicSegment {
                data_offset,
                data_length: segment.data_length,
                start_coding_pass: u32::from(segment.start_coding_pass),
                end_coding_pass: u32::from(segment.end_coding_pass),
                use_arithmetic: u32::from(segment.use_arithmetic),
            });
        }
        jobs.push(J2kClassicCleanupBatchJob {
            coded_offset,
            coded_len: u32::try_from(block.data.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
            })?,
            segment_offset,
            segment_count: u32::try_from(block.segments.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K MetalDirect segment count exceeds u32".to_string(),
            })?,
            width: block.width,
            height: block.height,
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect output offset overflow".to_string(),
                })?,
            missing_msbs: u32::from(block.missing_bit_planes),
            total_bitplanes: u32::from(block.total_bitplanes),
            number_of_coding_passes: u32::from(block.number_of_coding_passes),
            sub_band_type: match block.sub_band_type {
                slidecodec_j2k_native::J2kSubBandType::LowLow => 0,
                slidecodec_j2k_native::J2kSubBandType::HighLow => 1,
                slidecodec_j2k_native::J2kSubBandType::LowHigh => 2,
                slidecodec_j2k_native::J2kSubBandType::HighHigh => 3,
            },
            style_flags: classic_style_flags(block.style),
            strict: u32::from(block.strict),
            dequantization_step: block.dequantization_step,
        });
    }

    let coded_buffer = borrow_slice_buffer(&runtime.device, &coded_data);
    let jobs_buffer = borrow_slice_buffer(&runtime.device, &jobs);
    let segments_buffer = borrow_slice_buffer(&runtime.device, &segments);
    let status_check = dispatch_classic_cleanup_batched_in_command_buffer(
        runtime,
        command_buffer,
        &coded_buffer,
        &jobs_buffer,
        jobs.len(),
        &segments_buffer,
        output,
    );
    Ok((
        vec![
            DirectHostRetention::Bytes(coded_data),
            DirectHostRetention::ClassicJobs(jobs),
            DirectHostRetention::ClassicSegments(segments),
        ],
        vec![coded_buffer, jobs_buffer, segments_buffer],
        status_check,
    ))
}

#[cfg(target_os = "macos")]
fn required_ht_output_len(job: HtCodeBlockDecodeJob<'_>) -> Result<usize, Error> {
    if job.height == 0 {
        return Ok(0);
    }

    job.output_stride
        .checked_mul(job.height as usize - 1)
        .and_then(|prefix| prefix.checked_add(job.width as usize))
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal output size overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn encode_ht_sub_band_to_buffer(
    runtime: &MetalRuntime,
    job: &slidecodec_j2k_native::HtOwnedSubBandPlan,
    output: &Buffer,
) -> Result<(), Error> {
    if job.jobs.is_empty() {
        return Ok(());
    }

    let mut jobs = Vec::with_capacity(job.jobs.len());
    let mut coded_data = Vec::new();
    for block in &job.jobs {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(&block.data);
        jobs.push(J2kHtCleanupBatchJob {
            coded_offset,
            width: block.width,
            height: block.height,
            coded_len: u32::try_from(block.data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
            })?,
            cleanup_length: block.cleanup_length,
            refinement_length: block.refinement_length,
            missing_msbs: u32::from(block.missing_bit_planes),
            num_bitplanes: u32::from(block.num_bitplanes),
            number_of_coding_passes: u32::from(block.number_of_coding_passes),
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K MetalDirect output offset overflow".to_string(),
                })?,
            dequantization_step: block.dequantization_step,
            stripe_causal: u32::from(block.stripe_causal),
        });
    }

    dispatch_ht_cleanup_batched(runtime, &coded_data, &jobs, output)
}

#[cfg(target_os = "macos")]
fn encode_ht_sub_band_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: &slidecodec_j2k_native::HtOwnedSubBandPlan,
    output: &Buffer,
) -> Result<(Vec<DirectHostRetention>, Vec<Buffer>, DirectStatusCheck), Error> {
    if job.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            Vec::new(),
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let mut jobs = Vec::with_capacity(job.jobs.len());
    let mut coded_data = Vec::new();
    for block in &job.jobs {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(&block.data);
        jobs.push(J2kHtCleanupBatchJob {
            coded_offset,
            width: block.width,
            height: block.height,
            coded_len: u32::try_from(block.data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
            })?,
            cleanup_length: block.cleanup_length,
            refinement_length: block.refinement_length,
            missing_msbs: u32::from(block.missing_bit_planes),
            num_bitplanes: u32::from(block.num_bitplanes),
            number_of_coding_passes: u32::from(block.number_of_coding_passes),
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K MetalDirect output offset overflow".to_string(),
                })?,
            dequantization_step: block.dequantization_step,
            stripe_causal: u32::from(block.stripe_causal),
        });
    }

    let coded_buffer = borrow_slice_buffer(&runtime.device, &coded_data);
    let jobs_buffer = borrow_slice_buffer(&runtime.device, &jobs);
    let status_check = dispatch_ht_cleanup_batched_in_command_buffer(
        runtime,
        command_buffer,
        &coded_buffer,
        &jobs_buffer,
        jobs.len(),
        output,
    );
    Ok((
        vec![
            DirectHostRetention::Bytes(coded_data),
            DirectHostRetention::HtJobs(jobs),
        ],
        vec![coded_buffer, jobs_buffer],
        status_check,
    ))
}

#[cfg(target_os = "macos")]
fn decode_ht_status_error(status: J2kHtStatus) -> Error {
    let kind = match status.code {
        J2K_HT_STATUS_FAIL => "decode failure",
        J2K_HT_STATUS_UNSUPPORTED => "unsupported HT kernel input",
        _ => "unexpected HT kernel status",
    };
    Error::MetalKernel {
        message: format!("HTJ2K Metal kernel {kind} (detail={})", status.detail),
    }
}

#[cfg(target_os = "macos")]
fn dispatch_ht_cleanup(
    runtime: &MetalRuntime,
    coded_data: &[u8],
    params: J2kHtCleanupParams,
    decoded: &Buffer,
) -> Result<(), Error> {
    let input = borrow_slice_buffer(&runtime.device, coded_data);
    let status_buffer = runtime.device.new_buffer(
        size_of::<J2kHtStatus>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = runtime.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.ht_cleanup);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kHtCleanupParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(3, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(5, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(7, Some(&status_buffer), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let status = unsafe { status_buffer.contents().cast::<J2kHtStatus>().read() };
    if status.code != J2K_HT_STATUS_OK {
        return Err(decode_ht_status_error(status));
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn dispatch_ht_cleanup_batched(
    runtime: &MetalRuntime,
    coded_data: &[u8],
    jobs: &[J2kHtCleanupBatchJob],
    decoded: &Buffer,
) -> Result<(), Error> {
    let input = borrow_slice_buffer(&runtime.device, coded_data);
    let jobs_buffer = borrow_slice_buffer(&runtime.device, jobs);
    let status_buffer = runtime.device.new_buffer(
        (jobs.len().max(1) * size_of::<J2kHtStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = runtime.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.ht_cleanup_batched);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(&jobs_buffer), 0);
    encoder.set_buffer(3, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(5, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(7, Some(&status_buffer), 0);
    let width = runtime
        .ht_cleanup_batched
        .thread_execution_width()
        .max(1)
        .min(jobs.len() as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: jobs.len() as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let statuses = unsafe {
        core::slice::from_raw_parts(status_buffer.contents().cast::<J2kHtStatus>(), jobs.len())
    };
    if let Some(status) = statuses
        .iter()
        .copied()
        .find(|status| status.code != J2K_HT_STATUS_OK)
    {
        return Err(decode_ht_status_error(status));
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn dispatch_ht_cleanup_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    decoded: &Buffer,
) -> DirectStatusCheck {
    let status_buffer = runtime.device.new_buffer(
        (job_count.max(1) * size_of::<J2kHtStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.ht_cleanup_batched);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(5, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(7, Some(&status_buffer), 0);
    let width = runtime
        .ht_cleanup_batched
        .thread_execution_width()
        .max(1)
        .min(job_count as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: job_count as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    DirectStatusCheck::Ht {
        buffer: status_buffer,
        len: job_count,
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_classic_cleanup_code_block(
    job: J2kCodeBlockDecodeJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    let required_len = required_classic_output_len(job)?;
    if output.len() < required_len {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal output slice is too small".to_string(),
        });
    }

    if job.width == 0 || job.height == 0 {
        return Ok(());
    }

    with_runtime(|runtime| {
        let decoded = wrap_f32_output_buffer(&runtime.device, output);
        let batch_job = J2kClassicCleanupBatchJob {
            coded_offset: 0,
            coded_len: u32::try_from(job.data.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal coded payload exceeds u32".to_string(),
            })?,
            segment_offset: 0,
            segment_count: u32::try_from(job.segments.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal segment count exceeds u32".to_string(),
            })?,
            width: job.width,
            height: job.height,
            output_stride: u32::try_from(job.output_stride).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal output stride exceeds u32".to_string(),
            })?,
            output_offset: 0,
            missing_msbs: u32::from(job.missing_bit_planes),
            total_bitplanes: u32::from(job.total_bitplanes),
            number_of_coding_passes: u32::from(job.number_of_coding_passes),
            sub_band_type: match job.sub_band_type {
                slidecodec_j2k_native::J2kSubBandType::LowLow => 0,
                slidecodec_j2k_native::J2kSubBandType::HighLow => 1,
                slidecodec_j2k_native::J2kSubBandType::LowHigh => 2,
                slidecodec_j2k_native::J2kSubBandType::HighHigh => 3,
            },
            style_flags: classic_style_flags(job.style),
            strict: u32::from(job.strict),
            dequantization_step: job.dequantization_step,
        };
        let segments: Vec<_> = job
            .segments
            .iter()
            .map(|segment| J2kClassicSegment {
                data_offset: segment.data_offset,
                data_length: segment.data_length,
                start_coding_pass: u32::from(segment.start_coding_pass),
                end_coding_pass: u32::from(segment.end_coding_pass),
                use_arithmetic: u32::from(segment.use_arithmetic),
            })
            .collect();
        dispatch_classic_cleanup_batched(runtime, job.data, &[batch_job], &segments, &decoded)?;
        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_classic_cleanup_sub_band(
    job: J2kSubBandDecodeJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    let required_len = (job.width as usize)
        .checked_mul(job.height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal sub-band size overflow".to_string(),
        })?;
    if output.len() < required_len {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal sub-band output slice is too small".to_string(),
        });
    }
    if job.jobs.is_empty() {
        return Ok(());
    }

    with_runtime(|runtime| {
        let decoded = wrap_f32_output_buffer(&runtime.device, output);

        let mut jobs = Vec::with_capacity(job.jobs.len());
        let mut coded_data = Vec::new();
        let mut segments = Vec::new();

        for block in job.jobs {
            let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal batched coded payload exceeds u32".to_string(),
            })?;
            coded_data.extend_from_slice(block.code_block.data);
            let segment_offset = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal segment table exceeds u32".to_string(),
            })?;
            let end_x = block
                .output_x
                .checked_add(block.code_block.width)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal batched block width overflow".to_string(),
                })?;
            let end_y = block
                .output_y
                .checked_add(block.code_block.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal batched block height overflow".to_string(),
                })?;
            if end_x > job.width || end_y > job.height {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal batched block lies outside sub-band bounds"
                        .to_string(),
                });
            }
            for segment in block.code_block.segments {
                let data_offset =
                    coded_offset
                        .checked_add(segment.data_offset)
                        .ok_or_else(|| Error::MetalKernel {
                            message: "classic J2K Metal segment offset overflow".to_string(),
                        })?;
                segments.push(J2kClassicSegment {
                    data_offset,
                    data_length: segment.data_length,
                    start_coding_pass: u32::from(segment.start_coding_pass),
                    end_coding_pass: u32::from(segment.end_coding_pass),
                    use_arithmetic: u32::from(segment.use_arithmetic),
                });
            }
            jobs.push(J2kClassicCleanupBatchJob {
                coded_offset,
                coded_len: u32::try_from(block.code_block.data.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal coded payload exceeds u32".to_string(),
                    }
                })?,
                segment_offset,
                segment_count: u32::try_from(block.code_block.segments.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal segment count exceeds u32".to_string(),
                    }
                })?,
                width: block.code_block.width,
                height: block.code_block.height,
                output_stride: job.width,
                output_offset: block
                    .output_y
                    .checked_mul(job.width)
                    .and_then(|row| row.checked_add(block.output_x))
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K Metal output offset overflow".to_string(),
                    })?,
                missing_msbs: u32::from(block.code_block.missing_bit_planes),
                total_bitplanes: u32::from(block.code_block.total_bitplanes),
                number_of_coding_passes: u32::from(block.code_block.number_of_coding_passes),
                sub_band_type: match block.code_block.sub_band_type {
                    slidecodec_j2k_native::J2kSubBandType::LowLow => 0,
                    slidecodec_j2k_native::J2kSubBandType::HighLow => 1,
                    slidecodec_j2k_native::J2kSubBandType::LowHigh => 2,
                    slidecodec_j2k_native::J2kSubBandType::HighHigh => 3,
                },
                style_flags: classic_style_flags(block.code_block.style),
                strict: u32::from(block.code_block.strict),
                dequantization_step: block.code_block.dequantization_step,
            });
        }

        dispatch_classic_cleanup_batched(runtime, &coded_data, &jobs, &segments, &decoded)?;
        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_ht_cleanup_code_block(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    let required_len = required_ht_output_len(job)?;
    if output.len() < required_len {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal output slice is too small".to_string(),
        });
    }

    if job.width == 0 || job.height == 0 {
        return Ok(());
    }

    with_runtime(|runtime| {
        let params = J2kHtCleanupParams {
            width: job.width,
            height: job.height,
            coded_len: u32::try_from(job.data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal coded payload exceeds u32".to_string(),
            })?,
            cleanup_length: job.cleanup_length,
            refinement_length: job.refinement_length,
            missing_msbs: u32::from(job.missing_bit_planes),
            num_bitplanes: u32::from(job.num_bitplanes),
            number_of_coding_passes: u32::from(job.number_of_coding_passes),
            output_stride: u32::try_from(job.output_stride).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal output stride exceeds u32".to_string(),
            })?,
            output_offset: 0,
            dequantization_step: job.dequantization_step,
            stripe_causal: u32::from(job.stripe_causal),
        };
        let decoded = wrap_f32_output_buffer(&runtime.device, output);
        dispatch_ht_cleanup(runtime, job.data, params, &decoded)?;

        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_ht_cleanup_sub_band(
    job: HtSubBandDecodeJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    let required_len = (job.width as usize)
        .checked_mul(job.height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal sub-band size overflow".to_string(),
        })?;
    if output.len() < required_len {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal sub-band output slice is too small".to_string(),
        });
    }

    if job.jobs.is_empty() {
        return Ok(());
    }

    with_runtime(|runtime| {
        let decoded = wrap_f32_output_buffer(&runtime.device, output);

        let mut jobs = Vec::with_capacity(job.jobs.len());
        let mut coded_data = Vec::new();

        for block in job.jobs {
            let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal batched coded payload exceeds u32".to_string(),
            })?;
            coded_data.extend_from_slice(block.code_block.data);

            jobs.push(J2kHtCleanupBatchJob {
                coded_offset,
                width: block.code_block.width,
                height: block.code_block.height,
                coded_len: u32::try_from(block.code_block.data.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Metal coded payload exceeds u32".to_string(),
                    }
                })?,
                cleanup_length: block.code_block.cleanup_length,
                refinement_length: block.code_block.refinement_length,
                missing_msbs: u32::from(block.code_block.missing_bit_planes),
                num_bitplanes: u32::from(block.code_block.num_bitplanes),
                number_of_coding_passes: u32::from(block.code_block.number_of_coding_passes),
                output_stride: job.width,
                output_offset: block
                    .output_y
                    .checked_mul(job.width)
                    .and_then(|row| row.checked_add(block.output_x))
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Metal output offset overflow".to_string(),
                    })?,
                dequantization_step: block.code_block.dequantization_step,
                stripe_causal: u32::from(block.code_block.stripe_causal),
            });

            let end_x = block
                .output_x
                .checked_add(block.code_block.width)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batched block width overflow".to_string(),
                })?;
            let end_y = block
                .output_y
                .checked_add(block.code_block.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batched block height overflow".to_string(),
                })?;
            if end_x > job.width || end_y > job.height {
                return Err(Error::MetalKernel {
                    message: "HTJ2K Metal batched block lies outside sub-band bounds".to_string(),
                });
            }
        }

        dispatch_ht_cleanup_batched(runtime, &coded_data, &jobs, &decoded)?;
        Ok(())
    })
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::{decode_scaled_to_surface, output_shape_for, MetalRuntime};
    use slidecodec_core::PixelFormat;
    use slidecodec_j2k_native::{
        encode_htj2k, ColorSpace as NativeColorSpace, DecodeSettings, EncodeOptions, Image,
    };

    #[test]
    fn rgb16_with_alpha_is_rejected() {
        let runtime = MetalRuntime::new().expect("Metal runtime");
        let result = output_shape_for(
            &NativeColorSpace::RGB,
            true,
            4,
            PixelFormat::Rgb16,
            &runtime,
        );
        assert!(result.is_err(), "RGBA input must not silently map to Rgb16");
    }

    #[test]
    fn scaled_htj2k_decode_runs_through_metal_compute_path() {
        let pixels: Vec<u8> = (0..16).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let bytes = encode_htj2k(&pixels, 4, 4, 1, 8, false, &options).expect("encode ht gray8");

        let image = Image::new(
            &bytes,
            &DecodeSettings {
                target_resolution: Some((2, 2)),
                ..DecodeSettings::default()
            },
        )
        .expect("image");
        let host = image.decode().expect("host scaled decode");

        let surface = decode_scaled_to_surface(
            &bytes,
            (4, 4),
            PixelFormat::Gray8,
            slidecodec_core::Downscale::Half,
        )
        .expect("metal scaled decode");
        assert_eq!(surface.as_bytes(), host.as_slice());
    }
}

#[cfg(target_os = "macos")]
#[cfg(test)]
pub(crate) fn decode_image_to_surface<'a>(
    image: &NativeImage<'a>,
    context: &mut NativeDecoderContext<'a>,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let mut code_block_decoder = MetalCodeBlockDecoder::default();
        let decoded = image
            .decode_components_with_ht_decoder(context, &mut code_block_decoder)
            .map_err(|error| Error::Decode(slidecodec_j2k::J2kError::Backend(error.to_string())))?;
        let stage = select_plane_stage(runtime, image, &decoded, &mut code_block_decoder)?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn decode_image_region_to_surface<'a>(
    image: &NativeImage<'a>,
    context: &mut NativeDecoderContext<'a>,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let mut code_block_decoder = MetalCodeBlockDecoder::default();
        let decoded = image
            .decode_region_components_with_ht_decoder(
                context,
                (roi.x, roi.y, roi.w, roi.h),
                &mut code_block_decoder,
            )
            .map_err(|error| Error::Decode(slidecodec_j2k::J2kError::Backend(error.to_string())))?;
        let stage = select_plane_stage(runtime, image, &decoded, &mut code_block_decoder)?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
#[cfg(test)]
fn select_plane_stage(
    runtime: &MetalRuntime,
    image: &NativeImage<'_>,
    decoded: &NativeDecodedComponents<'_>,
    code_block_decoder: &mut MetalCodeBlockDecoder,
) -> Result<PlaneStage, Error> {
    if image.supports_direct_device_plane_reuse() {
        if matches!(decoded.color_space(), NativeColorSpace::RGB)
            && !decoded.has_alpha()
            && decoded.planes().len() == 3
        {
            if let Some(stage) = PlaneStage::from_captured_planes(
                decoded,
                code_block_decoder.mct.take_captured_planes(),
            ) {
                return Ok(stage);
            }
        }
        if matches!(decoded.color_space(), NativeColorSpace::Gray)
            && !decoded.has_alpha()
            && decoded.planes().len() == 1
        {
            if let Some(stage) = PlaneStage::from_captured_planes(
                decoded,
                code_block_decoder.store.take_captured_planes(),
            ) {
                return Ok(stage);
            }
        }
    }

    PlaneStage::from_planes(&runtime.device, decoded, None)
}

#[cfg(target_os = "macos")]
#[cfg(test)]
pub(crate) fn decode_scaled_to_surface(
    bytes: &[u8],
    dims: (u32, u32),
    fmt: PixelFormat,
    scale: slidecodec_core::Downscale,
) -> Result<Surface, Error> {
    let target_dims = (
        dims.0.div_ceil(scale.denominator()),
        dims.1.div_ceil(scale.denominator()),
    );
    let settings = NativeDecodeSettings {
        target_resolution: Some(target_dims),
        ..NativeDecodeSettings::default()
    };
    let image = NativeImage::new(bytes, &settings)
        .map_err(|error| Error::Decode(slidecodec_j2k::J2kError::Backend(error.to_string())))?;
    let mut context = NativeDecoderContext::default();
    decode_image_to_surface(&image, &mut context, fmt)
}
