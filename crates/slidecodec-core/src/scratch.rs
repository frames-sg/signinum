// SPDX-License-Identifier: Apache-2.0

pub trait ScratchPool: Send {
    fn bytes_allocated(&self) -> usize;
    fn reset(&mut self);
}
