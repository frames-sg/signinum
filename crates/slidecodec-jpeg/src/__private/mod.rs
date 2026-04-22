// SPDX-License-Identifier: Apache-2.0

#[doc(hidden)]
mod device_plan;

use crate::Decoder;

pub use crate::internal::checkpoint::DeviceCheckpoint;
pub use device_plan::{
    build_device_plan, summarize_device_batch, DeviceBatchSummary, DeviceComponentPlan,
    DeviceDecodePlan,
};

pub fn decoder_bytes<'a>(decoder: &'a Decoder<'a>) -> &'a [u8] {
    decoder.bytes
}
