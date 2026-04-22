// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use slidecodec_core::{BackendRequest, DeviceSubmission, Downscale, PixelFormat, Rect};

use crate::{session::SharedSession, Error, Surface};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchOp {
    Full,
    Region(Rect),
    Scaled(Downscale),
}

#[derive(Clone)]
pub(crate) struct QueuedRequest {
    pub(crate) input: Arc<[u8]>,
    pub(crate) fmt: PixelFormat,
    pub(crate) backend: BackendRequest,
    pub(crate) op: BatchOp,
    pub(crate) output_slot: usize,
}

impl QueuedRequest {
    pub(crate) fn new(
        input: Arc<[u8]>,
        fmt: PixelFormat,
        backend: BackendRequest,
        op: BatchOp,
    ) -> Self {
        Self {
            input,
            fmt,
            backend,
            op,
            output_slot: usize::MAX,
        }
    }

    pub(crate) fn with_output_slot(mut self, output_slot: usize) -> Self {
        self.output_slot = output_slot;
        self
    }
}

pub struct MetalSubmission {
    pub(crate) session: SharedSession,
    pub(crate) slot: usize,
}

impl DeviceSubmission for MetalSubmission {
    type Output = Surface;
    type Error = Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        let mut session = self.session.0.lock().expect("metal session");
        flush_if_needed(&mut session);
        take_surface(&mut session, self.slot)
    }
}

pub(crate) fn flush_if_needed(session: &mut crate::session::SessionState) {
    if session.queued.is_empty() {
        return;
    }

    let queued = std::mem::take(&mut session.queued);
    session.submissions = session.submissions.saturating_add(1);
    for request in queued {
        let result = crate::decode_surface_from_bytes(
            request.input.as_ref(),
            request.fmt,
            request.backend,
            request.op,
        );
        session.completed[request.output_slot] = Some(result);
    }
}

fn take_surface(session: &mut crate::session::SessionState, slot: usize) -> Result<Surface, Error> {
    session
        .completed
        .get_mut(slot)
        .and_then(Option::take)
        .ok_or_else(|| Error::MetalKernel {
            message: format!("missing queued Metal surface for slot {slot}"),
        })?
}
