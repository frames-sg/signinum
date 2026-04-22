// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

#[derive(Default)]
pub(crate) struct SessionState {
    pub(crate) submissions: u64,
    pub(crate) queued: Vec<crate::batch::QueuedRequest>,
    pub(crate) completed: Vec<Option<Result<crate::Surface, crate::Error>>>,
}

impl SessionState {
    pub(crate) fn queue_request(&mut self, request: crate::batch::QueuedRequest) -> usize {
        let slot = self.completed.len();
        self.completed.push(None);
        self.queued.push(request.with_output_slot(slot));
        slot
    }
}

#[derive(Clone, Default)]
pub(crate) struct SharedSession(pub(crate) Arc<Mutex<SessionState>>);
