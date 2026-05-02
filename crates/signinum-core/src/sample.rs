// SPDX-License-Identifier: Apache-2.0

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SampleType {
    U8,
    U16,
}

pub trait Sample: Copy + Default + Send + Sync + 'static {
    const TYPE: SampleType;
    const BITS: u8;
}

impl Sample for u8 {
    const TYPE: SampleType = SampleType::U8;
    const BITS: u8 = 8;
}

impl Sample for u16 {
    const TYPE: SampleType = SampleType::U16;
    const BITS: u8 = 16;
}
