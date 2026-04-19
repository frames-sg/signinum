// SPDX-License-Identifier: Apache-2.0

//! Reusable scratch buffers for the decode path.
//!
//! A [`ScratchPool`] owns every `Vec` that the sequential scan decoder
//! would otherwise allocate on each call: the three rolling MCU stripe
//! buffers, the per-component DC predictor, the chroma upsample rows, and
//! the RGB row buffers used by [`crate::decoder::RgbRowSink`] drivers.
//!
//! Use [`Decoder::decode_into_with_scratch`](crate::Decoder::decode_into_with_scratch)
//! / [`decode_rows_with_scratch`](crate::Decoder::decode_rows_with_scratch)
//! with a single long-lived pool to pay the allocation cost once across a
//! tile batch. The pool grows monotonically; it never shrinks.

use crate::entropy::sequential::{PreparedDecodePlan, StripeBuffer};
use alloc::vec::Vec;
use slidecodec_core::ScratchPool as CoreScratchPool;

#[derive(Debug, Default)]
pub(crate) struct YCbCr420Rows {
    pub(crate) cb_top: Vec<u8>,
    pub(crate) cb_bot: Vec<u8>,
    pub(crate) cr_top: Vec<u8>,
    pub(crate) cr_bot: Vec<u8>,
}

impl YCbCr420Rows {
    fn resize_width(&mut self, width: usize) {
        self.cb_top.resize(width, 0);
        self.cb_bot.resize(width, 0);
        self.cr_top.resize(width, 0);
        self.cr_bot.resize(width, 0);
    }
}

#[derive(Debug, Default)]
pub(crate) struct YCbCrGenericRows {
    pub(crate) cb_up: Vec<u8>,
    pub(crate) cr_up: Vec<u8>,
}

impl YCbCrGenericRows {
    fn resize_width(&mut self, width: usize) {
        self.cb_up.resize(width, 0);
        self.cr_up.resize(width, 0);
    }
}

#[derive(Debug, Default)]
pub(crate) struct RgbGenericRows {
    pub(crate) r: Vec<u8>,
    pub(crate) g: Vec<u8>,
    pub(crate) b: Vec<u8>,
}

impl RgbGenericRows {
    fn resize_width(&mut self, width: usize) {
        self.r.resize(width, 0);
        self.g.resize(width, 0);
        self.b.resize(width, 0);
    }
}

#[derive(Debug, Default)]
pub(crate) struct SinkRows {
    pub(crate) top_row: Vec<u8>,
    pub(crate) bottom_row: Vec<u8>,
}

impl SinkRows {
    fn resize_width(&mut self, width: usize) {
        let rgb_len = width.saturating_mul(3);
        self.top_row.resize(rgb_len, 0);
        self.bottom_row.resize(rgb_len, 0);
    }
}

/// Pool of decoder-internal scratch buffers, reusable across many
/// [`Decoder::decode_into_with_scratch`](crate::Decoder::decode_into_with_scratch)
/// / [`decode_rows_with_scratch`](crate::Decoder::decode_rows_with_scratch)
/// calls.
#[derive(Debug, Default)]
pub struct ScratchPool {
    pub(crate) prev_dc: Vec<i32>,
    pub(crate) stripe_a: StripeBuffer,
    pub(crate) stripe_b: StripeBuffer,
    pub(crate) stripe_c: StripeBuffer,
    pub(crate) ycbcr_420_rows: YCbCr420Rows,
    pub(crate) ycbcr_generic_rows: YCbCrGenericRows,
    pub(crate) rgb_generic_rows: RgbGenericRows,
    sink_rows: SinkRows,
}

impl ScratchPool {
    /// Create an empty pool. The first decode that uses it pays the full
    /// allocation cost; subsequent decodes at the same-or-smaller shape
    /// reuse the underlying `Vec`s with zero allocations.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Grow every internal scratch buffer to the shape required by `plan`
    /// and zero the predictor so each decode starts clean.
    pub(crate) fn prepare_for(
        &mut self,
        plan: &PreparedDecodePlan,
        mcus_per_row: u32,
        block_size: u32,
    ) {
        let n = plan.sampling.len();
        if self.prev_dc.len() < n {
            self.prev_dc.resize(n, 0);
        }
        for dc in &mut self.prev_dc[..n] {
            *dc = 0;
        }
        self.stripe_a.resize_for(plan, mcus_per_row, block_size);
        self.stripe_b.resize_for(plan, mcus_per_row, block_size);
        self.stripe_c.resize_for(plan, mcus_per_row, block_size);
        let denominator = 8 / block_size.max(1);
        let width = plan.dimensions.0.div_ceil(denominator) as usize;
        self.ycbcr_420_rows.resize_width(width);
        self.ycbcr_generic_rows.resize_width(width);
        self.rgb_generic_rows.resize_width(width);
        self.sink_rows.resize_width(width);
    }

    pub(crate) fn take_sink_rows(&mut self, width: usize) -> SinkRows {
        let mut rows = core::mem::take(&mut self.sink_rows);
        rows.resize_width(width);
        rows
    }

    pub(crate) fn restore_sink_rows(&mut self, rows: SinkRows) {
        self.sink_rows = rows;
    }
}

impl CoreScratchPool for ScratchPool {
    fn bytes_allocated(&self) -> usize {
        fn vec_bytes<T>(vec: &Vec<T>) -> usize {
            vec.capacity().saturating_mul(core::mem::size_of::<T>())
        }

        fn stripe_bytes(stripe: &StripeBuffer) -> usize {
            let mut total = 0usize;
            for plane in &stripe.planes {
                total = total.saturating_add(vec_bytes(plane));
            }
            total = total
                .saturating_add(vec_bytes(&stripe.plane_strides))
                .saturating_add(vec_bytes(&stripe.plane_rows));
            total
        }

        let mut total = vec_bytes(&self.prev_dc);
        total = total
            .saturating_add(stripe_bytes(&self.stripe_a))
            .saturating_add(stripe_bytes(&self.stripe_b))
            .saturating_add(stripe_bytes(&self.stripe_c))
            .saturating_add(vec_bytes(&self.ycbcr_420_rows.cb_top))
            .saturating_add(vec_bytes(&self.ycbcr_420_rows.cb_bot))
            .saturating_add(vec_bytes(&self.ycbcr_420_rows.cr_top))
            .saturating_add(vec_bytes(&self.ycbcr_420_rows.cr_bot))
            .saturating_add(vec_bytes(&self.ycbcr_generic_rows.cb_up))
            .saturating_add(vec_bytes(&self.ycbcr_generic_rows.cr_up))
            .saturating_add(vec_bytes(&self.rgb_generic_rows.r))
            .saturating_add(vec_bytes(&self.rgb_generic_rows.g))
            .saturating_add(vec_bytes(&self.rgb_generic_rows.b))
            .saturating_add(vec_bytes(&self.sink_rows.top_row))
            .saturating_add(vec_bytes(&self.sink_rows.bottom_row));
        total
    }

    fn reset(&mut self) {
        fn clear_stripe(stripe: &mut StripeBuffer) {
            for plane in &mut stripe.planes {
                plane.clear();
            }
            stripe.plane_strides.clear();
            stripe.plane_rows.clear();
        }

        self.prev_dc.clear();
        clear_stripe(&mut self.stripe_a);
        clear_stripe(&mut self.stripe_b);
        clear_stripe(&mut self.stripe_c);
        self.ycbcr_420_rows.cb_top.clear();
        self.ycbcr_420_rows.cb_bot.clear();
        self.ycbcr_420_rows.cr_top.clear();
        self.ycbcr_420_rows.cr_bot.clear();
        self.ycbcr_generic_rows.cb_up.clear();
        self.ycbcr_generic_rows.cr_up.clear();
        self.rgb_generic_rows.r.clear();
        self.rgb_generic_rows.g.clear();
        self.rgb_generic_rows.b.clear();
        self.sink_rows.top_row.clear();
        self.sink_rows.bottom_row.clear();
    }
}
