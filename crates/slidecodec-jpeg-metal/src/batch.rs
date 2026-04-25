// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use slidecodec_core::{BackendRequest, DeviceSubmission, Downscale, PixelFormat, Rect};
use slidecodec_jpeg::__private::{
    JpegMetalFast420PacketV1, JpegMetalFast422PacketV1, JpegMetalFast444PacketV1,
};

use crate::{session::SharedSession, Error, Surface};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchOp {
    Full,
    Region(Rect),
    Scaled(Downscale),
    RegionScaled { roi: Rect, scale: Downscale },
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
    RegionScaled { dims: (u32, u32), scale: Downscale },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SamplingFamily {
    Unknown,
    Fast420,
    Fast422,
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
    pub(crate) fast444_packet: Option<Arc<JpegMetalFast444PacketV1>>,
    pub(crate) fast422_packet: Option<Arc<JpegMetalFast422PacketV1>>,
    pub(crate) fast420_packet: Option<Arc<JpegMetalFast420PacketV1>>,
    pub(crate) output_slot: usize,
}

impl QueuedRequest {
    #[cfg(test)]
    pub(crate) fn new(
        input: Arc<[u8]>,
        fmt: PixelFormat,
        backend: BackendRequest,
        op: BatchOp,
        fast444_packet: Option<JpegMetalFast444PacketV1>,
        fast422_packet: Option<JpegMetalFast422PacketV1>,
        fast420_packet: Option<JpegMetalFast420PacketV1>,
    ) -> Self {
        Self {
            input,
            fmt,
            backend,
            op,
            fast444_packet: fast444_packet.map(Arc::new),
            fast422_packet: fast422_packet.map(Arc::new),
            fast420_packet: fast420_packet.map(Arc::new),
            output_slot: usize::MAX,
        }
    }

    pub(crate) fn new_shared(
        input: Arc<[u8]>,
        fmt: PixelFormat,
        backend: BackendRequest,
        op: BatchOp,
        fast444_packet: Option<Arc<JpegMetalFast444PacketV1>>,
        fast422_packet: Option<Arc<JpegMetalFast422PacketV1>>,
        fast420_packet: Option<Arc<JpegMetalFast420PacketV1>>,
    ) -> Self {
        Self {
            input,
            fmt,
            backend,
            op,
            fast444_packet,
            fast422_packet,
            fast420_packet,
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
                BatchOp::RegionScaled { roi, scale } => {
                    let scaled = scaled_rect_covering(roi, scale);
                    BatchKind::RegionScaled {
                        dims: (scaled.w, scaled.h),
                        scale,
                    }
                }
            },
            shape: session.resolve_batch_shape(&self.input, self.backend)?,
        })
    }
}

fn scaled_rect_covering(rect: Rect, scale: Downscale) -> Rect {
    let denom = scale.denominator();
    let x_end = rect.x + rect.w;
    let y_end = rect.y + rect.h;
    let x0 = rect.x / denom;
    let y0 = rect.y / denom;
    let x1 = x_end.div_ceil(denom);
    let y1 = y_end.div_ceil(denom);
    Rect {
        x: x0,
        y: y0,
        w: x1.saturating_sub(x0),
        h: y1.saturating_sub(y0),
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
        match crate::decode_compatible_batch(&batch) {
            Ok(Some(results)) => {
                for (request, result) in batch.into_iter().zip(results) {
                    session.completed[request.output_slot] = Some(result);
                }
            }
            Ok(None) => {
                for request in batch {
                    let result = crate::decode_surface_from_bytes(
                        request.input.as_ref(),
                        request.fmt,
                        request.backend,
                        request.op,
                        request.fast444_packet,
                        request.fast422_packet,
                        request.fast420_packet,
                    );
                    session.completed[request.output_slot] = Some(result);
                }
            }
            Err(err) => {
                for request in batch {
                    session.completed[request.output_slot] = Some(Err(Error::MetalKernel {
                        message: format!("batched JPEG Metal decode failed: {err}"),
                    }));
                }
            }
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
