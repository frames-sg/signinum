// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use slidecodec_core::BackendRequest;
use slidecodec_jpeg::__private::{
    build_metal_fast420_packet, build_metal_fast422_packet, build_metal_fast444_packet,
    JpegMetalFast420PacketV1, JpegMetalFast422PacketV1, JpegMetalFast444PacketV1,
};

use crate::{batch, Error};

const BATCH_SHAPE_CACHE_SLOTS: usize = 8;
const FAST_PACKET_CACHE_SLOTS: usize = 8;
const INPUT_ALIAS_CACHE_SLOTS: usize = 8;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

pub(crate) type SharedFastPackets = (
    Option<Arc<JpegMetalFast444PacketV1>>,
    Option<Arc<JpegMetalFast422PacketV1>>,
    Option<Arc<JpegMetalFast420PacketV1>>,
);

#[derive(Clone)]
pub(crate) struct CachedBatchShape {
    digest: u64,
    input: Arc<[u8]>,
    shape: batch::BatchShape,
}

#[derive(Clone)]
pub(crate) struct CachedFastPackets {
    digest: u64,
    input: Arc<[u8]>,
    fast444_packet: Option<Arc<JpegMetalFast444PacketV1>>,
    fast422_packet: Option<Arc<JpegMetalFast422PacketV1>>,
    fast420_packet: Option<Arc<JpegMetalFast420PacketV1>>,
}

#[derive(Clone)]
struct CachedInputAlias {
    source_ptr: usize,
    source_len: usize,
    input: Arc<[u8]>,
}

#[derive(Default)]
pub(crate) struct SessionState {
    pub(crate) submissions: u64,
    pub(crate) queued: Vec<crate::batch::QueuedRequest>,
    pub(crate) completed: Vec<Option<Result<crate::Surface, crate::Error>>>,
    batch_shapes: VecDeque<CachedBatchShape>,
    fast_packets: VecDeque<CachedFastPackets>,
    input_aliases: VecDeque<CachedInputAlias>,
}

impl SessionState {
    pub(crate) fn queue_request(&mut self, request: crate::batch::QueuedRequest) -> usize {
        let slot = self.completed.len();
        self.completed.push(None);
        self.queued.push(request.with_output_slot(slot));
        slot
    }

    pub(crate) fn intern_input_slice(&mut self, input: &[u8]) -> Arc<[u8]> {
        let source_ptr = input.as_ptr() as usize;
        let source_len = input.len();
        if let Some(entry) = self
            .input_aliases
            .iter()
            .find(|entry| entry.source_ptr == source_ptr && entry.source_len == source_len)
        {
            return Arc::clone(&entry.input);
        }

        let input = Arc::<[u8]>::from(input);
        if self.input_aliases.len() == INPUT_ALIAS_CACHE_SLOTS {
            self.input_aliases.pop_front();
        }
        self.input_aliases.push_back(CachedInputAlias {
            source_ptr,
            source_len,
            input: Arc::clone(&input),
        });
        input
    }

    pub(crate) fn resolve_batch_shape(
        &mut self,
        input: &Arc<[u8]>,
        backend: BackendRequest,
    ) -> Result<batch::BatchShape, Error> {
        #[cfg(not(target_os = "macos"))]
        {
            if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
                return Ok(batch::BatchShape {
                    restart_interval: None,
                    checkpoint_count: 0,
                    sampling_family: batch::SamplingFamily::Unknown,
                });
            }
        }

        match backend {
            BackendRequest::Auto | BackendRequest::Metal => {}
            BackendRequest::Cpu | BackendRequest::Cuda => {
                return Ok(batch::BatchShape {
                    restart_interval: None,
                    checkpoint_count: 0,
                    sampling_family: batch::SamplingFamily::Unknown,
                });
            }
        }

        if let Some(entry) = self
            .batch_shapes
            .iter()
            .find(|entry| Arc::ptr_eq(&entry.input, input))
        {
            return Ok(entry.shape);
        }

        let digest = digest_bytes(input.as_ref());
        if let Some(entry) = self
            .batch_shapes
            .iter()
            .find(|entry| entry.digest == digest && entry.input.as_ref() == input.as_ref())
        {
            return Ok(entry.shape);
        }

        let decoder = slidecodec_jpeg::Decoder::new(input.as_ref())?;
        let summary = slidecodec_jpeg::__private::summarize_device_batch(&decoder, 4);
        let shape = batch::BatchShape {
            restart_interval: summary.restart_interval,
            checkpoint_count: summary.checkpoint_count,
            sampling_family: if summary.matches_fast_420 {
                batch::SamplingFamily::Fast420
            } else if summary.matches_fast_422 {
                batch::SamplingFamily::Fast422
            } else if summary.matches_fast_444 {
                batch::SamplingFamily::Fast444
            } else {
                batch::SamplingFamily::Other
            },
        };

        if self.batch_shapes.len() == BATCH_SHAPE_CACHE_SLOTS {
            self.batch_shapes.pop_front();
        }
        self.batch_shapes.push_back(CachedBatchShape {
            digest,
            input: Arc::clone(input),
            shape,
        });

        Ok(shape)
    }

    pub(crate) fn resolve_fast_packets(
        &mut self,
        input: &Arc<[u8]>,
        backend: BackendRequest,
    ) -> SharedFastPackets {
        if !matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            return (None, None, None);
        }

        if let Some(entry) = self
            .fast_packets
            .iter()
            .find(|entry| Arc::ptr_eq(&entry.input, input))
        {
            return (
                entry.fast444_packet.clone(),
                entry.fast422_packet.clone(),
                entry.fast420_packet.clone(),
            );
        }

        let digest = digest_bytes(input.as_ref());
        if let Some(entry) = self
            .fast_packets
            .iter()
            .find(|entry| entry.digest == digest && entry.input.as_ref() == input.as_ref())
        {
            return (
                entry.fast444_packet.clone(),
                entry.fast422_packet.clone(),
                entry.fast420_packet.clone(),
            );
        }

        let fast444_packet = build_metal_fast444_packet(input.as_ref())
            .ok()
            .map(Arc::new);
        let fast422_packet = build_metal_fast422_packet(input.as_ref())
            .ok()
            .map(Arc::new);
        let fast420_packet = build_metal_fast420_packet(input.as_ref())
            .ok()
            .map(Arc::new);
        if self.fast_packets.len() == FAST_PACKET_CACHE_SLOTS {
            self.fast_packets.pop_front();
        }
        self.fast_packets.push_back(CachedFastPackets {
            digest,
            input: Arc::clone(input),
            fast444_packet: fast444_packet.clone(),
            fast422_packet: fast422_packet.clone(),
            fast420_packet: fast420_packet.clone(),
        });

        (fast444_packet, fast422_packet, fast420_packet)
    }
}

#[derive(Clone, Default)]
pub(crate) struct SharedSession(pub(crate) Arc<Mutex<SessionState>>);

fn digest_bytes(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_shape_cache_hits_for_repeated_input() {
        let mut session = SessionState::default();
        let input = Arc::<[u8]>::from(
            include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg").as_slice(),
        );

        let first = session
            .resolve_batch_shape(&input, BackendRequest::Metal)
            .expect("first shape");
        let second = session
            .resolve_batch_shape(&input, BackendRequest::Metal)
            .expect("second shape");

        assert_eq!(first, second);
        assert_eq!(session.batch_shapes.len(), 1);
    }

    #[test]
    fn fast_packet_cache_hits_for_repeated_input() {
        let mut session = SessionState::default();
        let input = Arc::<[u8]>::from(
            include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg").as_slice(),
        );

        let first = session.resolve_fast_packets(&input, BackendRequest::Metal);
        let second = session.resolve_fast_packets(&input, BackendRequest::Metal);

        assert!(first.2.is_some());
        assert_eq!(first, second);
        assert_eq!(session.fast_packets.len(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn batch_shape_tracks_fast422_sampling_family() {
        let mut session = SessionState::default();
        let input = Arc::<[u8]>::from(
            include_bytes!("../../../corpus/conformance/baseline_422_16x8.jpg").as_slice(),
        );

        let shape = session
            .resolve_batch_shape(&input, BackendRequest::Metal)
            .expect("fast422 shape");

        assert_eq!(shape.sampling_family, batch::SamplingFamily::Fast422);
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn non_macos_auto_and_metal_shape_resolution_stays_unparsed() {
        let mut session = SessionState::default();
        let invalid = Arc::<[u8]>::from(&b"not a jpeg"[..]);

        let auto = session
            .resolve_batch_shape(&invalid, BackendRequest::Auto)
            .expect("auto shape");
        let metal = session
            .resolve_batch_shape(&invalid, BackendRequest::Metal)
            .expect("metal shape");

        assert_eq!(auto.sampling_family, batch::SamplingFamily::Unknown);
        assert_eq!(metal.sampling_family, batch::SamplingFamily::Unknown);
        assert!(session.batch_shapes.is_empty());
    }
}
