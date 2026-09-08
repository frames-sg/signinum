// SPDX-License-Identifier: MIT OR Apache-2.0

//! Test-thread selection of the candidate inside the ordinary resident graph.

use std::{cell::RefCell, rc::Rc};

use super::*;

struct Selection {
    candidate: Rc<Cooperative53>,
    device_id: u64,
    dispatches: usize,
}

thread_local! {
    static SELECTED: RefCell<Option<Selection>> = const { RefCell::new(None) };
}

struct SelectionGuard(Option<Selection>);

impl SelectionGuard {
    fn new(candidate: Rc<Cooperative53>, device: &DeviceRef) -> Self {
        Self(SELECTED.with(|selected| {
            selected.replace(Some(Selection {
                candidate,
                device_id: device.registryID(),
                dispatches: 0,
            }))
        }))
    }

    fn dispatches() -> usize {
        SELECTED.with(|selected| {
            selected
                .borrow()
                .as_ref()
                .expect("selected test route")
                .dispatches
        })
    }
}

impl Drop for SelectionGuard {
    fn drop(&mut self) {
        SELECTED.with(|selected| {
            selected.replace(self.0.take());
        });
    }
}

pub(in crate::engine::decode_dispatch::idwt) fn try_dispatch(
    encoder: &ComputeCommandEncoderRef,
    decoded: &Buffer,
    params: &J2kRepeatedIdwtSingleDecompositionParams,
) -> bool {
    SELECTED.with(|selected| {
        let mut selected = selected.borrow_mut();
        let Some(selected) = selected.as_mut() else {
            return false;
        };
        if encoder.device().registryID() != selected.device_id {
            return false;
        }
        let candidate = &selected.candidate;
        let Some(horizontal) = candidate.layout(Axis::Horizontal, params.width) else {
            return false;
        };
        let Some(vertical) = candidate.layout(Axis::Vertical, params.height) else {
            return false;
        };
        // This hook receives the same validated buffer/ABI as the old sequence.
        // Both axes must fit before any candidate work is encoded.
        for (axis, lines, (bytes, threads)) in [
            (Axis::Horizontal, params.height, horizontal),
            (Axis::Vertical, params.width, vertical),
        ] {
            encoder.setComputePipelineState(candidate.pipeline(axis));
            encoder.set_buffer(0, Some(decoded), 0);
            encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(1, params);
            encoder.set_idwt_threadgroup_memory(bytes);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                j2k_metal_support::mtl_size(u64::from(lines), u64::from(params.batch_count), 1),
                j2k_metal_support::mtl_size(threads as u64, 1, 1),
            );
            selected.dispatches += 1;
            if matches!(axis, Axis::Horizontal) {
                encoder.memory_barrier_with_resources(&[decoded]);
            }
        }
        true
    })
}

mod tests;
