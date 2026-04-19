// SPDX-License-Identifier: Apache-2.0

use crate::sample::Sample;

pub trait RowSink<S: Sample> {
    type Error: core::error::Error + Send + Sync + 'static;

    fn write_row(&mut self, y: u32, row: &[S]) -> Result<(), Self::Error>;
}
