// SPDX-License-Identifier: Apache-2.0

//! JPEG marker-level parser. Walks the byte stream until the end of the
//! headers (the SOS marker) and populates [`crate::info::Info`] plus the
//! parsed DQT / DHT / DRI / APP14 state. See spec Section 3 phase 1.

pub(crate) mod markers;
pub(crate) mod sof;
pub(crate) mod tables;
pub(crate) mod adobe_app14;
