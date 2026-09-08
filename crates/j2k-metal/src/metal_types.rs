// SPDX-License-Identifier: MIT OR Apache-2.0

//! Crate-private spellings and checked operations for objc2 Metal objects.
//!
//! Public expert APIs spell out the corresponding objc2 protocol-object types
//! directly. These aliases and traits are only the host implementation's
//! fixed shader/resource vocabulary.

use core::{ffi::c_void, ptr::NonNull};

use j2k_core::accelerator::GpuAbi;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLEvent, MTLResource, MTLSharedEvent,
};

/// Protocol traits and checked crate-private operations needed for method
/// dispatch on objc2 protocol objects.
pub(crate) mod prelude {
    pub(crate) use super::{J2kBlitEncoderExt, J2kComputeEncoderExt};
    pub(crate) use objc2_foundation::NSString;
    pub(crate) use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLComputePipelineState, MTLDevice, MTLResource,
    };
}

pub(crate) type BlitCommandEncoder = Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>;
pub(crate) type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub(crate) type BufferRef = ProtocolObject<dyn MTLBuffer>;
pub(crate) type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
pub(crate) type CommandBufferRef = ProtocolObject<dyn MTLCommandBuffer>;
pub(crate) type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub(crate) type CommandQueueRef = ProtocolObject<dyn MTLCommandQueue>;
pub(crate) type ComputeCommandEncoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;
pub(crate) type ComputeCommandEncoderRef = ProtocolObject<dyn MTLComputeCommandEncoder>;
pub(crate) type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub(crate) type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub(crate) type DeviceRef = ProtocolObject<dyn MTLDevice>;
pub(crate) type Event = Retained<ProtocolObject<dyn MTLEvent>>;
pub(crate) type SharedEvent = Retained<ProtocolObject<dyn MTLSharedEvent>>;

/// Checked binding vocabulary for J2K's private, statically matched shaders.
///
/// objc2 marks resource and byte binding unsafe because arbitrary offsets,
/// indices, pointers, lifetimes, and shader layouts are not generally valid.
/// J2K creates retaining command buffers through `j2k-metal-support`; this
/// trait additionally validates indices and offsets and accepts only `GpuAbi`
/// values for immediate byte copies.
pub(crate) trait J2kComputeEncoderExt {
    fn set_buffer(&self, index: u64, buffer: Option<&ProtocolObject<dyn MTLBuffer>>, offset: u64);

    fn set_bytes<T: GpuAbi>(&self, index: u64, value: &T);

    // Fixed dynamic slot used only by the unselected IDWT experiment.
    #[cfg(test)]
    fn set_idwt_threadgroup_memory(&self, length: usize);

    fn memory_barrier_with_resources(&self, resources: &[&ProtocolObject<dyn MTLBuffer>]);
}

impl J2kComputeEncoderExt for ProtocolObject<dyn MTLComputeCommandEncoder> {
    fn set_buffer(&self, index: u64, buffer: Option<&ProtocolObject<dyn MTLBuffer>>, offset: u64) {
        let index = usize::try_from(index).expect("Metal buffer index fits usize");
        (index < 31)
            .then_some(())
            .expect("Metal buffer index exceeds the API binding table");
        let offset = usize::try_from(offset).expect("Metal buffer offset fits usize");
        if let Some(buffer) = buffer {
            (offset <= buffer.length())
                .then_some(())
                .expect("Metal buffer offset is out of bounds");
        }
        // SAFETY: The slot is within Metal's 31-entry buffer table and the
        // offset is within the allocation. Every J2K encoder comes from a
        // support-created retaining command buffer, so the bound resource is
        // retained through completion. Private call sites define the matching
        // shader ABI and submission ordering.
        unsafe { self.setBuffer_offset_atIndex(buffer, offset, index) };
    }

    fn set_bytes<T: GpuAbi>(&self, index: u64, value: &T) {
        let index = usize::try_from(index).expect("Metal byte-binding index fits usize");
        (index < 31)
            .then_some(())
            .expect("Metal byte-binding index exceeds the API binding table");
        let bytes = T::as_bytes(value);
        (!bytes.is_empty())
            .then_some(())
            .expect("Metal byte binding requires a nonempty ABI value");
        (bytes.len() == core::mem::size_of::<T>())
            .then_some(())
            .expect("Metal byte-binding length must match its ABI value");
        let pointer = NonNull::from(bytes).cast::<c_void>();
        // SAFETY: `GpuAbi::as_bytes` provides exactly `bytes.len()` initialized,
        // padding-free bytes, Metal copies them synchronously, and the binding
        // index was checked above.
        unsafe { self.setBytes_length_atIndex(pointer, bytes.len(), index) };
    }

    #[cfg(test)]
    fn set_idwt_threadgroup_memory(&self, length: usize) {
        assert!(length > 0 && length.is_multiple_of(16));
        // SAFETY: Slot zero matches the experiment's sole threadgroup argument.
        // Its checked dispatch planner enforces device/pipeline memory limits;
        // the positive 16-byte-aligned allocation contains the complete line.
        unsafe { self.setThreadgroupMemoryLength_atIndex(length, 0) };
    }

    fn memory_barrier_with_resources(&self, resources: &[&ProtocolObject<dyn MTLBuffer>]) {
        (!resources.is_empty())
            .then_some(())
            .expect("Metal resource barrier requires a resource");
        let mut resource_pointers: Vec<NonNull<ProtocolObject<dyn MTLResource>>> = resources
            .iter()
            .map(|resource| {
                let resource: &ProtocolObject<dyn MTLResource> =
                    ProtocolObject::from_ref(*resource);
                NonNull::from(resource)
            })
            .collect();
        let pointer = NonNull::new(resource_pointers.as_mut_ptr())
            .expect("a nonempty Metal resource pointer array is non-null");
        // SAFETY: `pointer` addresses exactly `resource_pointers.len()` valid
        // protocol-object pointers for this synchronous encoding call. Each
        // buffer is retained by the J2K submission and its retaining command
        // buffer until GPU completion, and every pointer is upcast through the
        // declared `MTLBuffer: MTLResource` protocol relationship.
        unsafe { self.memoryBarrierWithResources_count(pointer, resource_pointers.len()) };
    }
}

/// Checked buffer-copy vocabulary for support-created retaining command
/// buffers.
pub(crate) trait J2kBlitEncoderExt {
    fn copy_from_buffer(
        &self,
        source: &ProtocolObject<dyn MTLBuffer>,
        source_offset: u64,
        destination: &ProtocolObject<dyn MTLBuffer>,
        destination_offset: u64,
        size: u64,
    ) -> Result<(), crate::Error>;
}

impl J2kBlitEncoderExt for ProtocolObject<dyn MTLBlitCommandEncoder> {
    fn copy_from_buffer(
        &self,
        source: &ProtocolObject<dyn MTLBuffer>,
        source_offset: u64,
        destination: &ProtocolObject<dyn MTLBuffer>,
        destination_offset: u64,
        size: u64,
    ) -> Result<(), crate::Error> {
        let source_offset =
            usize::try_from(source_offset).map_err(|_| crate::Error::MetalKernel {
                message: "Metal source copy offset exceeds usize".to_string(),
            })?;
        let destination_offset =
            usize::try_from(destination_offset).map_err(|_| crate::Error::MetalKernel {
                message: "Metal destination copy offset exceeds usize".to_string(),
            })?;
        let size = usize::try_from(size).map_err(|_| crate::Error::MetalKernel {
            message: "Metal copy size exceeds usize".to_string(),
        })?;
        let source_in_bounds = source_offset
            .checked_add(size)
            .is_some_and(|end| end <= source.length());
        if !source_in_bounds {
            return Err(crate::Error::MetalKernel {
                message: "Metal source copy range is out of bounds".to_string(),
            });
        }
        let destination_in_bounds = destination_offset
            .checked_add(size)
            .is_some_and(|end| end <= destination.length());
        if !destination_in_bounds {
            return Err(crate::Error::MetalKernel {
                message: "Metal destination copy range is out of bounds".to_string(),
            });
        }
        // SAFETY: Both byte ranges were checked against their allocations.
        // The encoder belongs to a support-created retaining command buffer,
        // which retains both resources until completion; private call sites
        // establish synchronization and compatible byte representations.
        unsafe {
            self.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                source,
                source_offset,
                destination,
                destination_offset,
                size,
            );
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        any::Any,
        panic::{catch_unwind, AssertUnwindSafe},
    };

    use j2k_core::accelerator::GpuAbi;
    use j2k_metal_support::{
        checked_blit_command_encoder, checked_command_buffer, checked_command_queue,
        checked_compute_command_encoder, checked_shared_buffer, system_default_device,
    };
    use objc2_metal::{MTLBuffer as _, MTLCommandEncoder as _};

    use super::{J2kBlitEncoderExt as _, J2kComputeEncoderExt as _};

    #[derive(Clone, Copy)]
    struct ZeroSizedAbi;

    // SAFETY: This deliberately invalid zero-sized implementation is confined
    // to tests that verify the binding boundary rejects zero-sized ABI values.
    unsafe impl GpuAbi for ZeroSizedAbi {
        const NAME: &'static str = "ZeroSizedAbi";
    }

    fn panic_message(payload: &(dyn Any + Send)) -> String {
        if let Some(message) = payload.downcast_ref::<&str>() {
            (*message).to_owned()
        } else if let Some(message) = payload.downcast_ref::<String>() {
            message.clone()
        } else {
            "non-string panic payload".to_owned()
        }
    }

    fn assert_panics_with(f: impl FnOnce(), expected: &str) {
        let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("operation must panic");
        assert_eq!(panic_message(payload.as_ref()), expected);
    }

    #[test]
    fn compute_bindings_preserve_slot_offset_and_abi_validation() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        let Ok(device) = system_default_device() else {
            j2k_test_support::metal_device_unavailable_is_skip(module_path!());
            return;
        };
        let queue = checked_command_queue(&device).expect("Metal command queue");
        let command_buffer = checked_command_buffer(&queue).expect("Metal command buffer");
        let encoder = checked_compute_command_encoder(&command_buffer)
            .expect("Metal compute command encoder");
        let buffer = checked_shared_buffer(&device, 4).expect("Metal test buffer");

        encoder.set_buffer(30, None, 0);
        encoder.set_buffer(
            0,
            Some(&buffer),
            u64::try_from(buffer.length()).expect("buffer length fits u64"),
        );
        assert_panics_with(
            || encoder.set_buffer(31, None, 0),
            "Metal buffer index exceeds the API binding table",
        );
        assert_panics_with(
            || {
                encoder.set_buffer(
                    0,
                    Some(&buffer),
                    u64::try_from(buffer.length() + 1).expect("buffer length fits u64"),
                );
            },
            "Metal buffer offset is out of bounds",
        );
        assert_panics_with(
            || encoder.set_bytes(0, &ZeroSizedAbi),
            "Metal byte binding requires a nonempty ABI value",
        );
        encoder.endEncoding();
    }

    #[test]
    fn blit_bindings_reject_overflowing_copy_ranges() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        let Ok(device) = system_default_device() else {
            j2k_test_support::metal_device_unavailable_is_skip(module_path!());
            return;
        };
        let queue = checked_command_queue(&device).expect("Metal command queue");
        let command_buffer = checked_command_buffer(&queue).expect("Metal command buffer");
        let encoder =
            checked_blit_command_encoder(&command_buffer).expect("Metal blit command encoder");
        let buffer = checked_shared_buffer(&device, 4).expect("Metal test buffer");

        let error = encoder
            .copy_from_buffer(&buffer, u64::MAX, &buffer, 0, 1)
            .expect_err("overflowing source range must fail");
        assert!(error
            .to_string()
            .contains("Metal source copy range is out of bounds"));
        encoder.endEncoding();
    }
}
