// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    ComponentRowWriter, CompressedTransferSyntax, CoreDecodeOutcome, DecodeOutcome,
    DecodeRowsError, Decoder, DecoderContext, Downscale, DownscaleFactor, ImageCodec, ImageDecode,
    ImageDecodeRows, InterleavedRgbWriter, JpegCodec, JpegError, JpegView, OutputWriter,
    PixelFormat, Rect, RowSink, ScratchPool, SofKind, TileBatchDecode, Vec, Warning,
};
use crate::allocation::{
    checked_add_allocation_bytes, checked_allocation_bytes, checked_allocation_len,
    try_reserve_for_len_with_live_budget, try_resize_filled,
};
use j2k_core::TileRegionScaledDecodeJob;

#[cfg(test)]
mod tests;

pub(super) fn jpeg_passthrough_syntax(view: &JpegView<'_>) -> Option<CompressedTransferSyntax> {
    let info = view.info();
    match info.sof_kind {
        SofKind::Baseline8 if info.bit_depth == 8 => Some(CompressedTransferSyntax::JpegBaseline8),
        SofKind::Extended8 | SofKind::Extended12 => {
            Some(CompressedTransferSyntax::JpegExtendedSequential)
        }
        SofKind::Progressive8 | SofKind::Progressive12 => {
            Some(CompressedTransferSyntax::JpegProgressive)
        }
        SofKind::Lossless if view.lossless_predictor() == Some(1) => {
            Some(CompressedTransferSyntax::JpegLosslessSv1)
        }
        SofKind::Lossless => Some(CompressedTransferSyntax::JpegLossless),
        SofKind::Baseline8 => None,
    }
}

pub(super) fn core_outcome(outcome: DecodeOutcome) -> CoreDecodeOutcome<Warning> {
    outcome.into()
}

#[doc(hidden)]
impl ImageCodec for JpegCodec {
    type Error = JpegError;
    type Warning = Warning;
    type Pool = ScratchPool;
}

#[doc(hidden)]
impl ImageCodec for Decoder<'_> {
    type Error = JpegError;
    type Warning = Warning;
    type Pool = ScratchPool;
}

#[doc(hidden)]
impl<'a> ImageDecode<'a> for Decoder<'a> {
    type View = JpegView<'a>;

    fn inspect(input: &'a [u8]) -> Result<j2k_core::Info, Self::Error> {
        Ok(Decoder::inspect(input)?.to_core_info())
    }

    fn parse(input: &'a [u8]) -> Result<Self::View, Self::Error> {
        JpegView::parse(input)
    }

    fn from_view(view: Self::View) -> Result<Self, Self::Error> {
        Decoder::from_view(view)
    }

    fn decode_into(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<CoreDecodeOutcome<Self::Warning>, Self::Error> {
        Decoder::decode_into(self, out, stride, fmt).map(core_outcome)
    }

    fn decode_into_with_scratch(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<CoreDecodeOutcome<Self::Warning>, Self::Error> {
        Decoder::decode_into_with_scratch(self, pool, out, stride, fmt).map(core_outcome)
    }

    fn decode_region_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: j2k_core::Rect,
    ) -> Result<CoreDecodeOutcome<Self::Warning>, Self::Error> {
        Decoder::decode_region_into_with_scratch(self, pool, out, stride, fmt, roi.into())
            .map(core_outcome)
    }

    fn decode_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<CoreDecodeOutcome<Self::Warning>, Self::Error> {
        Decoder::decode_scaled_into_with_scratch(self, pool, out, stride, fmt, scale)
            .map(core_outcome)
    }

    fn decode_region_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: j2k_core::Rect,
        scale: Downscale,
    ) -> Result<CoreDecodeOutcome<Self::Warning>, Self::Error> {
        Decoder::decode_region_scaled_into_with_scratch(
            self,
            pool,
            out,
            stride,
            fmt,
            roi.into(),
            scale,
        )
        .map(core_outcome)
    }
}

pub(super) struct CoreRowSinkAdapter<'a, R: RowSink<u8>> {
    pub(super) sink: &'a mut R,
    pub(super) sink_error: Option<R::Error>,
}

impl<R: RowSink<u8>> RowSink<u8> for CoreRowSinkAdapter<'_, R> {
    type Error = JpegError;

    fn write_row(&mut self, y: u32, row: &[u8]) -> Result<(), JpegError> {
        match self.sink.write_row(y, row) {
            Ok(()) => Ok(()),
            Err(err) => {
                self.sink_error = Some(err);
                Err(JpegError::RowSinkAborted)
            }
        }
    }
}

#[doc(hidden)]
impl<'a> ImageDecodeRows<'a, u8> for Decoder<'a> {
    fn decode_rows<R: RowSink<u8>>(
        &mut self,
        sink: &mut R,
    ) -> Result<CoreDecodeOutcome<Self::Warning>, DecodeRowsError<Self::Error, R::Error>> {
        let mut adapter = CoreRowSinkAdapter {
            sink,
            sink_error: None,
        };
        match Decoder::decode_rows(self, &mut adapter) {
            Ok(outcome) => Ok(core_outcome(outcome)),
            Err(JpegError::RowSinkAborted) => match adapter.sink_error {
                Some(err) => Err(DecodeRowsError::Sink(err)),
                None => Err(DecodeRowsError::Decode(JpegError::InternalInvariant {
                    reason: "row sink abort stores the original sink error",
                })),
            },
            Err(err) => Err(DecodeRowsError::Decode(err)),
        }
    }
}

#[doc(hidden)]
impl TileBatchDecode for JpegCodec {
    type Context = DecoderContext;

    fn decode_tile(
        ctx: &mut Self::Context,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<CoreDecodeOutcome<Self::Warning>, Self::Error> {
        let view = JpegView::parse(input)?;
        Decoder::with_view_in_context(view, ctx, |decoder| {
            decoder
                .decode_into_with_scratch(pool, out, stride, fmt)
                .map(core_outcome)
        })
    }

    fn decode_tile_region(
        ctx: &mut Self::Context,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: j2k_core::Rect,
    ) -> Result<CoreDecodeOutcome<Self::Warning>, Self::Error> {
        let view = JpegView::parse(input)?;
        Decoder::with_view_in_context(view, ctx, |decoder| {
            decoder
                .decode_region_into_with_scratch(pool, out, stride, fmt, roi.into())
                .map(core_outcome)
        })
    }

    fn decode_tile_scaled(
        ctx: &mut Self::Context,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<CoreDecodeOutcome<Self::Warning>, Self::Error> {
        let view = JpegView::parse(input)?;
        Decoder::with_view_in_context(view, ctx, |decoder| {
            decoder
                .decode_scaled_into_with_scratch(pool, out, stride, fmt, scale)
                .map(core_outcome)
        })
    }

    fn decode_tile_region_scaled(
        ctx: &mut Self::Context,
        pool: &mut Self::Pool,
        fmt: PixelFormat,
        job: TileRegionScaledDecodeJob<'_, '_>,
    ) -> Result<CoreDecodeOutcome<Self::Warning>, Self::Error> {
        let TileRegionScaledDecodeJob {
            input,
            out,
            stride,
            roi,
            scale,
        } = job;
        let view = JpegView::parse(input)?;
        Decoder::with_view_in_context(view, ctx, |decoder| {
            decoder
                .decode_region_scaled_into_with_scratch(pool, out, stride, fmt, roi.into(), scale)
                .map(core_outcome)
        })
    }
}

pub(super) struct CroppedWriter<W> {
    pub(super) inner: W,
    pub(super) rect: Rect,
    pub(super) source_x0: u32,
    pub(super) rgb_row_len: usize,
    pub(super) rgb_rows_bytes: usize,
    pub(super) top_row: Vec<u8>,
    pub(super) bottom_row: Vec<u8>,
}

pub(super) struct ProgressiveDownscaleWriter<'a, W> {
    pub(super) inner: &'a mut W,
    pub(super) denom: u32,
    pub(super) scaled_width: usize,
    pub(super) r: Vec<u8>,
    pub(super) g: Vec<u8>,
    pub(super) b: Vec<u8>,
}

impl<'a, W> ProgressiveDownscaleWriter<'a, W> {
    pub(super) fn new(
        inner: &'a mut W,
        downscale: DownscaleFactor,
        dimensions: (u32, u32),
    ) -> Result<Self, JpegError> {
        let denom = downscale.denominator();
        let scaled_width = dimensions.0.div_ceil(denom) as usize;
        let row_bytes = checked_allocation_len::<u8>(scaled_width, 3)?;
        let mut live_bytes = 0;
        let mut r = Vec::new();
        try_reserve_for_len_with_live_budget(&mut r, scaled_width, &mut live_bytes, row_bytes)?;
        r.resize(scaled_width, 0);
        let mut g = Vec::new();
        try_reserve_for_len_with_live_budget(&mut g, scaled_width, &mut live_bytes, row_bytes)?;
        g.resize(scaled_width, 0);
        let mut b = Vec::new();
        try_reserve_for_len_with_live_budget(&mut b, scaled_width, &mut live_bytes, row_bytes)?;
        b.resize(scaled_width, 0);
        Ok(Self {
            inner,
            denom,
            scaled_width,
            r,
            g,
            b,
        })
    }

    pub(super) fn capacity_bytes(&self) -> Result<usize, JpegError> {
        let rg = checked_add_allocation_bytes(
            checked_allocation_bytes::<u8>(self.r.capacity())?,
            checked_allocation_bytes::<u8>(self.g.capacity())?,
        )?;
        checked_add_allocation_bytes(rg, checked_allocation_bytes::<u8>(self.b.capacity())?)
    }

    fn should_emit(&self, y: u32) -> bool {
        y.is_multiple_of(self.denom)
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "validated JPEG output widths originate as u32 dimensions before slice indexing"
    )]
    fn sample_row(
        src: &[u8],
        denom: u32,
        width: usize,
        dst: &mut Vec<u8>,
    ) -> Result<(), JpegError> {
        try_resize_filled(dst, width, 0)?;
        for (x, out) in dst.iter_mut().enumerate() {
            let src_x = (x as u32)
                .saturating_mul(denom)
                .min(src.len().saturating_sub(1) as u32);
            *out = src[src_x as usize];
        }
        Ok(())
    }
}

impl<W: OutputWriter> OutputWriter for ProgressiveDownscaleWriter<'_, W> {
    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), JpegError> {
        if !self.should_emit(y) {
            return Ok(());
        }
        Self::sample_row(r_row, self.denom, self.scaled_width, &mut self.r)?;
        Self::sample_row(g_row, self.denom, self.scaled_width, &mut self.g)?;
        Self::sample_row(b_row, self.denom, self.scaled_width, &mut self.b)?;
        self.inner
            .write_rgb_row(y / self.denom, &self.r, &self.g, &self.b)
    }

    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
    ) -> Result<(), JpegError> {
        if !self.should_emit(y) {
            return Ok(());
        }
        Self::sample_row(y_row, self.denom, self.scaled_width, &mut self.r)?;
        Self::sample_row(cb_row, self.denom, self.scaled_width, &mut self.g)?;
        Self::sample_row(cr_row, self.denom, self.scaled_width, &mut self.b)?;
        self.inner
            .write_ycbcr_row(y / self.denom, &self.r, &self.g, &self.b)
    }

    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), JpegError> {
        if !self.should_emit(y) {
            return Ok(());
        }
        Self::sample_row(gray_row, self.denom, self.scaled_width, &mut self.r)?;
        self.inner.write_gray_row(y / self.denom, &self.r)
    }
}

impl<W: ComponentRowWriter + ?Sized> OutputWriter for &mut W {
    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), JpegError> {
        ComponentRowWriter::write_rgb_row(*self, y, r_row, g_row, b_row)
    }

    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
    ) -> Result<(), JpegError> {
        ComponentRowWriter::write_ycbcr_row(*self, y, y_row, cb_row, cr_row)
    }

    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), JpegError> {
        ComponentRowWriter::write_gray_row(*self, y, gray_row)
    }
}

impl<W> CroppedWriter<W> {
    pub(super) fn new(
        inner: W,
        rect: Rect,
        source_x0: u32,
        source_width: u32,
    ) -> Result<Self, JpegError> {
        let rgb_row_len = checked_allocation_len::<u8>(source_width as usize, 3)?;
        let rgb_rows_bytes = checked_allocation_len::<u8>(rgb_row_len, 2)?;
        Ok(Self {
            inner,
            rect,
            source_x0,
            rgb_row_len,
            rgb_rows_bytes,
            top_row: Vec::new(),
            bottom_row: Vec::new(),
        })
    }

    fn crop_range(&self, bytes_per_pixel: usize) -> Result<core::ops::Range<usize>, JpegError> {
        let x0 = self
            .rect
            .x
            .checked_sub(self.source_x0)
            .ok_or(JpegError::InternalInvariant {
                reason: "crop window must cover requested rect",
            })? as usize;
        let start = x0
            .checked_mul(bytes_per_pixel)
            .ok_or(JpegError::InternalInvariant {
                reason: "crop window byte offset overflow",
            })?;
        let width = (self.rect.w as usize).checked_mul(bytes_per_pixel).ok_or(
            JpegError::InternalInvariant {
                reason: "crop window byte width overflow",
            },
        )?;
        let end = start
            .checked_add(width)
            .ok_or(JpegError::InternalInvariant {
                reason: "crop window byte range overflow",
            })?;
        Ok(start..end)
    }

    fn crop_row(row: &[u8], range: core::ops::Range<usize>) -> Result<&[u8], JpegError> {
        row.get(range).ok_or(JpegError::InternalInvariant {
            reason: "crop window exceeds source row",
        })
    }

    fn prepare_rgb_rows(&mut self) -> Result<(), JpegError> {
        if self.top_row.len() == self.rgb_row_len && self.bottom_row.len() == self.rgb_row_len {
            return Ok(());
        }

        self.top_row = Vec::new();
        self.bottom_row = Vec::new();
        let mut live_bytes = 0;
        let reserve_result = (|| {
            try_reserve_for_len_with_live_budget(
                &mut self.top_row,
                self.rgb_row_len,
                &mut live_bytes,
                self.rgb_rows_bytes,
            )?;
            try_reserve_for_len_with_live_budget(
                &mut self.bottom_row,
                self.rgb_row_len,
                &mut live_bytes,
                self.rgb_rows_bytes,
            )
        })();
        if let Err(error) = reserve_result {
            self.top_row = Vec::new();
            self.bottom_row = Vec::new();
            return Err(error);
        }
        self.top_row.resize(self.rgb_row_len, 0);
        self.bottom_row.resize(self.rgb_row_len, 0);
        Ok(())
    }
}

impl<W: OutputWriter> OutputWriter for CroppedWriter<W> {
    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), JpegError> {
        if y < self.rect.y || y >= self.rect.y + self.rect.h {
            return Ok(());
        }
        let range = self.crop_range(1)?;
        self.inner.write_rgb_row(
            y - self.rect.y,
            Self::crop_row(r_row, range.clone())?,
            Self::crop_row(g_row, range.clone())?,
            Self::crop_row(b_row, range)?,
        )
    }

    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
    ) -> Result<(), JpegError> {
        if y < self.rect.y || y >= self.rect.y + self.rect.h {
            return Ok(());
        }
        let range = self.crop_range(1)?;
        self.inner.write_ycbcr_row(
            y - self.rect.y,
            Self::crop_row(y_row, range.clone())?,
            Self::crop_row(cb_row, range.clone())?,
            Self::crop_row(cr_row, range)?,
        )
    }

    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), JpegError> {
        if y < self.rect.y || y >= self.rect.y + self.rect.h {
            return Ok(());
        }
        let range = self.crop_range(1)?;
        self.inner
            .write_gray_row(y - self.rect.y, Self::crop_row(gray_row, range)?)
    }
}

impl<W: InterleavedRgbWriter> InterleavedRgbWriter for CroppedWriter<W> {
    fn with_rgb_rows<R, F>(&mut self, y: u32, row_count: usize, fill: F) -> Result<R, JpegError>
    where
        F: FnOnce(&mut [u8], Option<&mut [u8]>) -> Result<R, JpegError>,
    {
        self.prepare_rgb_rows()?;

        let result = match row_count {
            1 => fill(&mut self.top_row, None)?,
            2 => fill(&mut self.top_row, Some(&mut self.bottom_row))?,
            _ => unreachable!("CroppedWriter only supports one or two rows"),
        };

        let top_in = y >= self.rect.y && y < self.rect.y + self.rect.h;
        let bottom_y = y + 1;
        let bottom_in =
            row_count == 2 && bottom_y >= self.rect.y && bottom_y < self.rect.y + self.rect.h;
        let range = self.crop_range(3)?;

        match (top_in, bottom_in) {
            (false, false) => {}
            (true, false) => {
                self.inner.with_rgb_rows(y - self.rect.y, 1, |dst, _| {
                    dst.copy_from_slice(Self::crop_row(&self.top_row, range.clone())?);
                    Ok(())
                })?;
            }
            (false, true) => {
                self.inner
                    .with_rgb_rows(bottom_y - self.rect.y, 1, |dst, _| {
                        dst.copy_from_slice(Self::crop_row(&self.bottom_row, range.clone())?);
                        Ok(())
                    })?;
            }
            (true, true) => {
                self.inner
                    .with_rgb_rows(y - self.rect.y, 2, |dst_top, dst_bottom| {
                        dst_top.copy_from_slice(Self::crop_row(&self.top_row, range.clone())?);
                        let dst_bottom = dst_bottom.ok_or(JpegError::InternalInvariant {
                            reason: "row_count=2 supplies bottom row",
                        })?;
                        dst_bottom.copy_from_slice(Self::crop_row(&self.bottom_row, range)?);
                        Ok(())
                    })?;
            }
        }

        Ok(result)
    }
}
