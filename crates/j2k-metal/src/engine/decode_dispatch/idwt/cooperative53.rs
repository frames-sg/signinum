// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unselected whole-line 5/3 experiment with a test-only decode-graph hook.

use super::*;
use crate::engine::runtime::MetalRuntime;
use crate::metal_types::{ComputePipelineState, DeviceRef};
use j2k_metal_support::{MetalPipelineLoader, MetalSupportError};

struct Cooperative53 {
    horizontal: ComputePipelineState,
    vertical: ComputePipelineState,
    max_memory: usize,
}

#[derive(Clone, Copy, Debug)]
enum Axis {
    Horizontal,
    Vertical,
}

impl Cooperative53 {
    fn new(device: &DeviceRef) -> Result<Self, MetalSupportError> {
        let source = format!(
            "{}\n{}",
            crate::engine::shader_source::decode_shader_source(),
            include_str!("cooperative53.metal")
        );
        let loader = MetalPipelineLoader::new(device, &source)?;
        Ok(Self {
            horizontal: loader.pipeline("audit_idwt53_horizontal_cooperative")?,
            vertical: loader.pipeline("audit_idwt53_vertical_cooperative")?,
            max_memory: device.maxThreadgroupMemoryLength(),
        })
    }

    fn pipeline(&self, axis: Axis) -> &ComputePipelineState {
        match axis {
            Axis::Horizontal => &self.horizontal,
            Axis::Vertical => &self.vertical,
        }
    }

    fn layout(&self, axis: Axis, length: u32) -> Option<(usize, usize)> {
        let pipeline = self.pipeline(axis);
        let bytes = usize::try_from(length)
            .ok()?
            .checked_mul(size_of::<f32>())?;
        // Metal dynamic threadgroup allocations are rounded to 16 bytes.
        let dynamic_bytes = bytes.checked_add(15)? & !15;
        if bytes == 0
            || dynamic_bytes.checked_add(pipeline.staticThreadgroupMemoryLength())?
                > self.max_memory
        {
            return None;
        }
        let simd = pipeline.threadExecutionWidth();
        let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
        if simd == 0 || max_threads < simd {
            return None;
        }
        let threads = (max_threads / simd).min(4).checked_mul(simd)?;
        (threads > 0).then_some((dynamic_bytes, threads))
    }
}

// Explicit test-only opt-in. Each axis falls back independently to its original
// device kernel if its complete line cannot fit the reported resource limits.
#[expect(
    clippy::too_many_arguments,
    reason = "test-only stage comparison names its complete dispatch inputs"
)]
fn dispatch_axis(
    runtime: &MetalRuntime,
    candidate: &Cooperative53,
    encoder: &ComputeCommandEncoderRef,
    decoded: &Buffer,
    byte_offset: usize,
    params: &J2kRepeatedIdwtSingleDecompositionParams,
    axis: Axis,
    cooperative: bool,
) -> bool {
    let (length, lines) = match axis {
        Axis::Horizontal => (params.width, params.height),
        Axis::Vertical => (params.height, params.width),
    };
    assert!(lines > 0 && length > 0 && params.batch_count > 0);
    let required = usize::try_from(params.width)
        .unwrap()
        .checked_mul(usize::try_from(params.height).unwrap())
        .and_then(|count| count.checked_mul(usize::try_from(params.batch_count).unwrap()))
        .expect("test image element count fits usize");
    // The existing reference kernels use uint plane/index arithmetic.
    assert!(u32::try_from(required).is_ok());
    let end = required
        .checked_mul(size_of::<f32>())
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .expect("test output range");
    assert!(byte_offset.is_multiple_of(size_of::<f32>()) && end <= decoded.length());
    let layout = cooperative
        .then(|| candidate.layout(axis, length))
        .flatten();
    let kernels = runtime.decode().expect("reference decode kernels");
    let pipeline = if layout.is_some() {
        candidate.pipeline(axis)
    } else {
        match axis {
            Axis::Horizontal => &kernels.idwt_reversible53_horizontal_batched,
            Axis::Vertical => &kernels.idwt_reversible53_vertical_batched,
        }
    };
    encoder.setComputePipelineState(pipeline);
    encoder.set_buffer(0, Some(decoded), byte_offset as u64);
    encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(1, params);
    if let Some((bytes, threads)) = layout {
        encoder.set_idwt_threadgroup_memory(bytes);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            j2k_metal_support::mtl_size(u64::from(lines), u64::from(params.batch_count), 1),
            j2k_metal_support::mtl_size(threads as u64, 1, 1),
        );
    } else {
        encoder.dispatchThreads_threadsPerThreadgroup(
            j2k_metal_support::mtl_size(u64::from(lines), u64::from(params.batch_count), 1),
            j2k_metal_support::mtl_size(pipeline.threadExecutionWidth().max(1) as u64, 1, 1),
        );
    }
    layout.is_some()
}

pub(super) mod route;
mod tests;
