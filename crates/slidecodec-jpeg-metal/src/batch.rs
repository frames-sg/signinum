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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BatchKey {
    fmt: PixelFormat,
    backend: BackendRequest,
    kind: BatchKind,
    shape: BatchShape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchKind {
    Full,
    Region { dims: (u32, u32) },
    Scaled { scale: Downscale },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SamplingFamily {
    Unknown,
    Fast420,
    Fast444,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BatchShape {
    pub(crate) restart_interval: Option<u16>,
    pub(crate) checkpoint_count: usize,
    pub(crate) sampling_family: SamplingFamily,
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

    pub(crate) fn key(
        &self,
        session: &mut crate::session::SessionState,
    ) -> Result<BatchKey, Error> {
        Ok(BatchKey {
            fmt: self.fmt,
            backend: self.backend,
            kind: match self.op {
                BatchOp::Full => BatchKind::Full,
                BatchOp::Region(roi) => BatchKind::Region {
                    dims: (roi.w, roi.h),
                },
                BatchOp::Scaled(scale) => BatchKind::Scaled { scale },
            },
            shape: session.resolve_batch_shape(&self.input, self.backend)?,
        })
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

    let batches = group_compatible_requests(std::mem::take(&mut session.queued), session);
    for batch in batches {
        session.submissions = session.submissions.saturating_add(1);
        for request in batch {
            let result = crate::decode_surface_from_bytes(
                request.input.as_ref(),
                request.fmt,
                request.backend,
                request.op,
            );
            session.completed[request.output_slot] = Some(result);
        }
    }
}

fn group_compatible_requests(
    queued: Vec<QueuedRequest>,
    session: &mut crate::session::SessionState,
) -> Vec<Vec<QueuedRequest>> {
    let mut batches: Vec<(BatchKey, Vec<QueuedRequest>)> = Vec::new();
    for request in queued {
        let key = match request.key(session) {
            Ok(key) => key,
            Err(err) => {
                session.completed[request.output_slot] = Some(Err(err));
                continue;
            }
        };
        if let Some((_, batch)) = batches.iter_mut().find(|(batch_key, _)| *batch_key == key) {
            batch.push(request);
        } else {
            batches.push((key, vec![request]));
        }
    }
    batches.into_iter().map(|(_, batch)| batch).collect()
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
