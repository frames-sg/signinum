// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

use signinum_core::{BackendRequest, DeviceSubmission, Downscale, PixelFormat, Rect};

use crate::{Error, J2kDecoder, MetalSession, Surface};

const AUTO_REGION_SCALED_GRAYSCALE_BATCH64_MIN_DIM: u32 = 512;
const AUTO_REGION_SCALED_GRAYSCALE_BATCH64_MIN_COUNT: usize = 64;
const AUTO_REGION_SCALED_GRAYSCALE_BATCH16_MIN_DIM: u32 = 1024;
const AUTO_REGION_SCALED_GRAYSCALE_BATCH16_MIN_COUNT: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchOp {
    Full,
    Region(Rect),
    Scaled(Downscale),
    RegionScaled { roi: Rect, scale: Downscale },
}

#[derive(Clone)]
struct QueuedRequest {
    input: Arc<[u8]>,
    fmt: PixelFormat,
    backend: BackendRequest,
    op: BatchOp,
    output_slot: usize,
}

impl QueuedRequest {
    fn max_image_dim(&self) -> Option<u32> {
        let decoder = J2kDecoder::new(self.input.as_ref()).ok()?;
        let dims = decoder.inner.info().dimensions;
        Some(dims.0.max(dims.1))
    }
}

#[derive(Default)]
pub(crate) struct SessionState {
    pub(crate) submissions: u64,
    queued: Vec<QueuedRequest>,
    completed: Vec<Option<Result<Surface, Error>>>,
}

#[derive(Clone, Default)]
pub(crate) struct SharedSession(pub(crate) Arc<Mutex<SessionState>>);

pub struct MetalSubmission {
    session: SharedSession,
    slot: usize,
}

impl DeviceSubmission for MetalSubmission {
    type Output = Surface;
    type Error = Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        let mut session = self.session.0.lock().expect("J2K Metal session");
        flush_if_needed(&mut session);
        take_surface(&mut session, self.slot)
    }
}

pub(crate) fn queue_tile_request(
    session: &mut MetalSession,
    input: &[u8],
    fmt: PixelFormat,
    backend: BackendRequest,
    op: BatchOp,
) -> MetalSubmission {
    queue_tile_request_shared(session, Arc::<[u8]>::from(input), fmt, backend, op)
}

pub(crate) fn queue_tile_request_shared(
    session: &mut MetalSession,
    input: Arc<[u8]>,
    fmt: PixelFormat,
    backend: BackendRequest,
    op: BatchOp,
) -> MetalSubmission {
    let mut state = session.shared.0.lock().expect("J2K Metal session");
    let slot = state.completed.len();
    state.completed.push(None);
    state.queued.push(QueuedRequest {
        input,
        fmt,
        backend,
        op,
        output_slot: slot,
    });
    MetalSubmission {
        session: session.shared.clone(),
        slot,
    }
}

fn flush_if_needed(session: &mut SessionState) {
    if session.queued.is_empty() {
        return;
    }

    for batch in group_metal_requests(std::mem::take(&mut session.queued)) {
        process_batch(session, batch);
    }
}

fn group_metal_requests(queued: Vec<QueuedRequest>) -> Vec<Vec<QueuedRequest>> {
    coalesce_distinct_region_scaled_grayscale_metal_requests(
        coalesce_distinct_full_color_metal_requests(
            coalesce_distinct_full_grayscale_metal_requests(group_repeated_full_metal_requests(
                queued,
            )),
        ),
    )
}

fn group_repeated_full_metal_requests(queued: Vec<QueuedRequest>) -> Vec<Vec<QueuedRequest>> {
    let mut batches: Vec<Vec<QueuedRequest>> = Vec::new();
    for request in queued {
        if let Some(batch) = batches
            .iter_mut()
            .find(|batch| can_decode_as_repeated_full_metal_batch(&batch[0], &request))
        {
            batch.push(request);
        } else {
            batches.push(vec![request]);
        }
    }
    batches
}

fn coalesce_distinct_full_grayscale_metal_requests(
    repeated_batches: Vec<Vec<QueuedRequest>>,
) -> Vec<Vec<QueuedRequest>> {
    let mut batches = Vec::new();
    let mut gray8 = Vec::new();
    let mut gray16 = Vec::new();

    for batch in repeated_batches {
        if batch.len() == 1 && is_distinct_full_grayscale_metal_candidate(&batch[0]) {
            let request = batch
                .into_iter()
                .next()
                .expect("single-entry batch has request");
            match request.fmt {
                PixelFormat::Gray8 => gray8.push(request),
                PixelFormat::Gray16 => gray16.push(request),
                _ => unreachable!("candidate pixel format is restricted above"),
            }
        } else {
            batches.push(batch);
        }
    }

    push_coalesced_or_single(&mut batches, gray8);
    push_coalesced_or_single(&mut batches, gray16);
    batches
}

fn coalesce_distinct_region_scaled_grayscale_metal_requests(
    repeated_batches: Vec<Vec<QueuedRequest>>,
) -> Vec<Vec<QueuedRequest>> {
    let mut batches = Vec::new();
    let mut metal_gray8 = Vec::new();
    let mut metal_gray16 = Vec::new();
    let mut auto_gray8 = Vec::new();
    let mut auto_gray16 = Vec::new();

    for batch in repeated_batches {
        if batch.len() == 1 && is_region_scaled_grayscale_batch_candidate(&batch[0]) {
            let request = batch
                .into_iter()
                .next()
                .expect("single-entry batch has request");
            match (request.backend, request.fmt) {
                (BackendRequest::Metal, PixelFormat::Gray8) => metal_gray8.push(request),
                (BackendRequest::Metal, PixelFormat::Gray16) => metal_gray16.push(request),
                (BackendRequest::Auto, PixelFormat::Gray8) => auto_gray8.push(request),
                (BackendRequest::Auto, PixelFormat::Gray16) => auto_gray16.push(request),
                _ => unreachable!("candidate backend and pixel format are restricted above"),
            }
        } else {
            batches.push(batch);
        }
    }

    push_coalesced_or_single(&mut batches, metal_gray8);
    push_coalesced_or_single(&mut batches, metal_gray16);
    push_auto_region_scaled_grayscale_batches(&mut batches, auto_gray8);
    push_auto_region_scaled_grayscale_batches(&mut batches, auto_gray16);
    batches
}

fn push_coalesced_or_single(batches: &mut Vec<Vec<QueuedRequest>>, requests: Vec<QueuedRequest>) {
    if requests.is_empty() {
        return;
    }
    if requests.len() == 1 {
        batches.extend(requests.into_iter().map(|request| vec![request]));
    } else {
        batches.push(requests);
    }
}

fn push_auto_region_scaled_grayscale_batches(
    batches: &mut Vec<Vec<QueuedRequest>>,
    requests: Vec<QueuedRequest>,
) {
    let Some(min_dim) = auto_region_scaled_grayscale_metal_min_dim(&requests) else {
        push_coalesced_or_single(batches, requests);
        return;
    };

    let mut metal_requests = Vec::new();
    let mut cpu_requests = Vec::new();
    for request in requests {
        if request
            .max_image_dim()
            .is_some_and(|max_dim| max_dim >= min_dim)
        {
            metal_requests.push(request);
        } else {
            cpu_requests.push(request);
        }
    }
    push_coalesced_or_single(batches, metal_requests);
    push_coalesced_or_single(batches, cpu_requests);
}

#[allow(clippy::similar_names)]
fn coalesce_distinct_full_color_metal_requests(
    repeated_batches: Vec<Vec<QueuedRequest>>,
) -> Vec<Vec<QueuedRequest>> {
    let mut batches = Vec::new();
    let mut rgb8 = Vec::new();
    let mut rgba8 = Vec::new();
    let mut rgb16 = Vec::new();

    for batch in repeated_batches {
        if batch.len() == 1 && is_distinct_full_color_metal_candidate(&batch[0]) {
            let request = batch
                .into_iter()
                .next()
                .expect("single-entry batch has request");
            match request.fmt {
                PixelFormat::Rgb8 => rgb8.push(request),
                PixelFormat::Rgba8 => rgba8.push(request),
                PixelFormat::Rgb16 => rgb16.push(request),
                _ => unreachable!("candidate pixel format is restricted above"),
            }
        } else {
            batches.push(batch);
        }
    }

    push_coalesced_or_single(&mut batches, rgb8);
    push_coalesced_or_single(&mut batches, rgba8);
    push_coalesced_or_single(&mut batches, rgb16);
    batches
}

fn can_decode_as_repeated_full_grayscale_batch(
    first: &QueuedRequest,
    next: &QueuedRequest,
) -> bool {
    is_repeated_full_grayscale_candidate(first)
        && is_repeated_full_grayscale_candidate(next)
        && first.fmt == next.fmt
        && first.backend == next.backend
        && first.input.as_ref() == next.input.as_ref()
}

fn can_decode_as_repeated_full_color_batch(first: &QueuedRequest, next: &QueuedRequest) -> bool {
    is_repeated_full_color_candidate(first)
        && is_repeated_full_color_candidate(next)
        && first.fmt == next.fmt
        && first.backend == next.backend
        && first.input.as_ref() == next.input.as_ref()
}

fn can_decode_as_repeated_full_metal_batch(first: &QueuedRequest, next: &QueuedRequest) -> bool {
    can_decode_as_repeated_full_grayscale_batch(first, next)
        || can_decode_as_repeated_full_color_batch(first, next)
}

fn is_repeated_full_grayscale_candidate(request: &QueuedRequest) -> bool {
    matches!(request.op, BatchOp::Full)
        && matches!(request.fmt, PixelFormat::Gray8 | PixelFormat::Gray16)
        && matches!(
            request.backend,
            BackendRequest::Auto | BackendRequest::Metal
        )
}

fn is_repeated_full_color_candidate(request: &QueuedRequest) -> bool {
    matches!(request.op, BatchOp::Full)
        && matches!(
            request.fmt,
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
        )
        && request.backend == BackendRequest::Metal
}

fn is_distinct_full_grayscale_metal_candidate(request: &QueuedRequest) -> bool {
    matches!(request.op, BatchOp::Full)
        && matches!(request.fmt, PixelFormat::Gray8 | PixelFormat::Gray16)
        && request.backend == BackendRequest::Metal
}

fn is_distinct_full_color_metal_candidate(request: &QueuedRequest) -> bool {
    matches!(request.op, BatchOp::Full)
        && matches!(
            request.fmt,
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
        )
        && request.backend == BackendRequest::Metal
}

fn is_region_scaled_grayscale_batch_candidate(request: &QueuedRequest) -> bool {
    matches!(request.op, BatchOp::RegionScaled { .. })
        && matches!(request.fmt, PixelFormat::Gray8 | PixelFormat::Gray16)
        && matches!(
            request.backend,
            BackendRequest::Auto | BackendRequest::Metal
        )
}

fn should_auto_use_metal_for_region_scaled_grayscale_batch(requests: &[QueuedRequest]) -> bool {
    auto_region_scaled_grayscale_metal_min_dim(requests).is_some()
}

fn auto_region_scaled_grayscale_metal_min_dim(requests: &[QueuedRequest]) -> Option<u32> {
    let mut count_512_class = 0usize;
    let mut count_1024_class = 0usize;
    for request in requests {
        let Some(max_dim) = request.max_image_dim() else {
            continue;
        };
        if max_dim >= AUTO_REGION_SCALED_GRAYSCALE_BATCH64_MIN_DIM {
            count_512_class += 1;
        }
        if max_dim >= AUTO_REGION_SCALED_GRAYSCALE_BATCH16_MIN_DIM {
            count_1024_class += 1;
        }
    }

    if count_512_class >= AUTO_REGION_SCALED_GRAYSCALE_BATCH64_MIN_COUNT {
        Some(AUTO_REGION_SCALED_GRAYSCALE_BATCH64_MIN_DIM)
    } else if count_1024_class >= AUTO_REGION_SCALED_GRAYSCALE_BATCH16_MIN_COUNT {
        Some(AUTO_REGION_SCALED_GRAYSCALE_BATCH16_MIN_DIM)
    } else {
        None
    }
}

fn can_decode_requests_as_repeated_full_grayscale_batch(requests: &[QueuedRequest]) -> bool {
    let Some((first, rest)) = requests.split_first() else {
        return false;
    };
    !rest.is_empty()
        && rest
            .iter()
            .all(|request| can_decode_as_repeated_full_grayscale_batch(first, request))
}

fn can_decode_requests_as_repeated_full_color_batch(requests: &[QueuedRequest]) -> bool {
    let Some((first, rest)) = requests.split_first() else {
        return false;
    };
    !rest.is_empty()
        && rest
            .iter()
            .all(|request| can_decode_as_repeated_full_color_batch(first, request))
}

fn process_batch(session: &mut SessionState, requests: Vec<QueuedRequest>) {
    if can_decode_requests_as_repeated_full_grayscale_batch(&requests) {
        if let Some(Ok(surfaces)) = decode_repeated_full_grayscale(&requests[0], requests.len()) {
            if surfaces.len() == requests.len() {
                session.submissions = session.submissions.saturating_add(1);
                for (request, surface) in requests.into_iter().zip(surfaces) {
                    session.completed[request.output_slot] = Some(Ok(surface));
                }
                return;
            }
        }
    }

    if can_decode_requests_as_repeated_full_color_batch(&requests) {
        if let Some(Ok(surfaces)) = decode_repeated_full_color(&requests[0], requests.len()) {
            if surfaces.len() == requests.len() {
                session.submissions = session.submissions.saturating_add(1);
                for (request, surface) in requests.into_iter().zip(surfaces) {
                    session.completed[request.output_slot] = Some(Ok(surface));
                }
                return;
            }
        }
    }

    if requests.len() > 1 {
        if let Some(Ok(surfaces)) = decode_distinct_full_grayscale_batch(&requests) {
            if surfaces.len() == requests.len() {
                session.submissions = session.submissions.saturating_add(1);
                for (request, surface) in requests.into_iter().zip(surfaces) {
                    session.completed[request.output_slot] = Some(Ok(surface));
                }
                return;
            }
        }
    }

    if requests.len() > 1 {
        if let Some(result) = decode_distinct_full_color_batch(&requests) {
            match result {
                Ok(surfaces) if surfaces.len() == requests.len() => {
                    session.submissions = session.submissions.saturating_add(1);
                    for (request, surface) in requests.into_iter().zip(surfaces) {
                        session.completed[request.output_slot] = Some(Ok(surface));
                    }
                    return;
                }
                Ok(_) | Err(_) => {}
            }
        }
    }

    if requests.len() > 1 {
        if let Some(Ok(surfaces)) = decode_distinct_region_scaled_grayscale_batch(&requests) {
            if surfaces.len() == requests.len() {
                session.submissions = session.submissions.saturating_add(1);
                for (request, surface) in requests.into_iter().zip(surfaces) {
                    session.completed[request.output_slot] = Some(Ok(surface));
                }
                return;
            }
        }
    }

    for request in requests {
        session.submissions = session.submissions.saturating_add(1);
        session.completed[request.output_slot] = Some(decode_individual(&request));
    }
}

fn decode_repeated_full_grayscale(
    request: &QueuedRequest,
    count: usize,
) -> Option<Result<Vec<Surface>, Error>> {
    if !is_repeated_full_grayscale_candidate(request) || count <= 1 {
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        let result =
            J2kDecoder::new(request.input.as_ref()).and_then(|mut decoder| match request.backend {
                BackendRequest::Auto => {
                    decoder.decode_repeated_grayscale_auto_to_device(request.fmt, count)
                }
                BackendRequest::Metal => {
                    decoder.decode_repeated_grayscale_direct_to_device(request.fmt, count)
                }
                _ => unreachable!("candidate backend is restricted above"),
            });
        Some(result)
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

fn decode_repeated_full_color(
    request: &QueuedRequest,
    count: usize,
) -> Option<Result<Vec<Surface>, Error>> {
    if !is_repeated_full_color_candidate(request) || count <= 1 {
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        let result = J2kDecoder::new(request.input.as_ref()).and_then(|mut decoder| {
            decoder.decode_repeated_color_direct_to_device(request.fmt, count)
        });
        Some(result)
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

fn decode_distinct_full_grayscale_batch(
    requests: &[QueuedRequest],
) -> Option<Result<Vec<Surface>, Error>> {
    let first = requests.first()?;
    if requests.len() <= 1
        || !requests.iter().all(|request| {
            is_distinct_full_grayscale_metal_candidate(request) && request.fmt == first.fmt
        })
    {
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        let inputs = requests
            .iter()
            .map(|request| request.input.clone())
            .collect::<Vec<_>>();
        Some(crate::decode_full_grayscale_batch_direct_to_device(
            &inputs, first.fmt,
        ))
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

fn decode_distinct_full_color_batch(
    requests: &[QueuedRequest],
) -> Option<Result<Vec<Surface>, Error>> {
    let first = requests.first()?;
    if requests.len() <= 1
        || !requests.iter().all(|request| {
            is_distinct_full_color_metal_candidate(request) && request.fmt == first.fmt
        })
    {
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        let inputs = requests
            .iter()
            .map(|request| request.input.clone())
            .collect::<Vec<_>>();
        Some(crate::decode_full_color_batch_direct_to_device(
            &inputs, first.fmt,
        ))
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

fn decode_distinct_region_scaled_grayscale_batch(
    requests: &[QueuedRequest],
) -> Option<Result<Vec<Surface>, Error>> {
    let first = requests.first()?;
    if requests.len() <= 1
        || !requests.iter().all(|request| {
            is_region_scaled_grayscale_batch_candidate(request)
                && request.fmt == first.fmt
                && request.backend == first.backend
        })
    {
        return None;
    }
    if first.backend == BackendRequest::Auto
        && !should_auto_use_metal_for_region_scaled_grayscale_batch(requests)
    {
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        let request_specs = requests
            .iter()
            .map(|request| match request.op {
                BatchOp::RegionScaled { roi, scale } => (request.input.clone(), roi, scale),
                _ => unreachable!("candidate op is restricted above"),
            })
            .collect::<Vec<_>>();
        Some(
            crate::decode_region_scaled_grayscale_batch_direct_to_device(&request_specs, first.fmt),
        )
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

fn decode_individual(request: &QueuedRequest) -> Result<Surface, Error> {
    let mut decoder = J2kDecoder::new(request.input.as_ref())?;
    match request.op {
        BatchOp::Full => decoder.decode_to_surface_impl(request.fmt, request.backend),
        BatchOp::Region(roi) => {
            decoder.decode_region_to_surface_impl(request.fmt, roi, request.backend)
        }
        BatchOp::Scaled(scale) => {
            decoder.decode_scaled_to_surface_impl(request.fmt, scale, request.backend)
        }
        BatchOp::RegionScaled { roi, scale } => {
            decoder.decode_region_scaled_to_surface_impl(request.fmt, roi, scale, request.backend)
        }
    }
}

fn take_surface(session: &mut SessionState, slot: usize) -> Result<Surface, Error> {
    session
        .completed
        .get_mut(slot)
        .and_then(Option::take)
        .ok_or_else(|| Error::MetalKernel {
            message: format!("missing queued J2K Metal surface for slot {slot}"),
        })?
}
