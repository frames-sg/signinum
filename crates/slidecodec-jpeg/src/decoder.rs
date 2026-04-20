// SPDX-License-Identifier: Apache-2.0

//! Public [`Decoder`] entry points.

use crate::backend::Backend;
use crate::context::DecoderContext;
use crate::entropy::huffman::HuffmanTable;
use crate::entropy::sequential::{
    decode_scan_baseline, decode_scan_baseline_rgb, decode_scan_fast_rgb_444,
    decode_scan_fast_tile_rgb, decode_scan_fast_tile_rgb_region, PreparedComponentPlan,
    PreparedDecodePlan,
};
use crate::error::{JpegError, MarkerKind, Warning};
use crate::info::{ColorSpace, DownscaleFactor, Info, OutputFormat, Rect, SofKind};
use crate::internal::scratch::{ScratchPool, SinkRows};
use crate::output::{
    validate_buffer, Gray8Writer, InterleavedRgbWriter, OutputWriter, Rgb8Writer, Rgba8Writer,
};
use crate::parse::header::{parse_header, parse_info, ParsedHeader};
use crate::JpegCodec;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::RefCell;
use slidecodec_core::{
    Colorspace as CoreColorspace, DecodeOutcome as CoreDecodeOutcome, DecodeRowsError,
    DecoderContext as CoreDecoderContext, Downscale, ImageCodec, ImageDecode, ImageDecodeRows,
    PixelFormat, RowSink, TileBatchDecode,
};

const DEFAULT_MAX_DECODE_BYTES: usize = 512 * 1024 * 1024;

std::thread_local! {
    static DEFAULT_SCRATCH: RefCell<ScratchPool> = RefCell::new(ScratchPool::new());
    static DEFAULT_CONTEXT: RefCell<DecoderContext> = RefCell::new(DecoderContext::new());
}

/// Non-fatal outcome of a successful decode. See spec Section 2.
///
/// `DecodeOutcome` lives on `decoder.rs` rather than `info.rs` because it
/// carries `Warning` values from `error.rs`, and moving it into `info` would
/// create a `info → error` cycle (see `info.rs` header note).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeOutcome {
    /// The rectangle actually written to the output buffer. For `decode_into`
    /// this is always `Rect::full(info.dimensions)`; later milestones add
    /// `decode_region_into` which can return a narrower rect.
    pub decoded: Rect,
    /// Warnings emitted during parse or decode. Empty when the stream is
    /// syntactically clean and every capability was exercised without fallback.
    pub warnings: Vec<Warning>,
}

/// Receives decoded component rows before they are packed into the final
/// interleaved pixel format.
pub trait ComponentRowWriter {
    /// Receive one grayscale row.
    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), JpegError>;

    /// Receive one full-width Y/Cb/Cr row.
    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
    ) -> Result<(), JpegError>;

    /// Receive one full-width planar RGB row.
    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), JpegError>;
}

/// A parsed borrowed view of a JPEG stream.
#[derive(Debug)]
pub struct JpegView<'a> {
    bytes: &'a [u8],
    header: ParsedHeader,
    info: Info,
}

impl<'a> JpegView<'a> {
    /// Parse the stream into a borrowed view that can later build a decoder.
    pub fn parse(input: &'a [u8]) -> Result<Self, JpegError> {
        let header = parse_header(input)?;
        let info = header.info();
        Ok(Self {
            bytes: input,
            header,
            info,
        })
    }

    /// Header-derived metadata for the parsed stream.
    pub fn info(&self) -> &Info {
        &self.info
    }
}

/// A borrowed view of a JPEG stream ready to decode. Constructed via
/// [`Decoder::new`] or [`Decoder::from_view`]. `Decoder<'a>: Send + Sync`.
#[derive(Debug)]
pub struct Decoder<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) info: Info,
    pub(crate) warnings: Arc<[Warning]>,
    pub(crate) backend: Backend,
    pub(crate) plan: PreparedDecodePlan,
}

impl<'a> Decoder<'a> {
    /// Parse the headers without decoding pixels. The parser walks headers up
    /// to the first SOS and then performs a lightweight marker scan so
    /// `Info::scan_count` reflects all scans in the file.
    ///
    /// # Errors
    /// Returns any structural, unsupported-SOF, or sanity-check error
    /// encountered before the Start-of-Scan marker. See [`JpegError`].
    pub fn inspect(input: &'a [u8]) -> Result<Info, JpegError> {
        parse_info(input)
    }

    /// Build a decoder ready for `decode_into`. Parses the full header, pre-
    /// builds every referenced Huffman table, and validates that the stream is
    /// one of the SOFs this release implements.
    ///
    /// # Errors
    /// - Any parse error encountered before SOS (see [`Self::inspect`]).
    /// - [`JpegError::NotImplemented`] for SOFs that parse but are not yet
    ///   decodable (Extended12, Progressive, Lossless — all land in M3).
    /// - [`JpegError::MissingHuffmanTable`] if the scan references a DC/AC
    ///   table slot that was never defined by a DHT segment.
    pub fn new(input: &'a [u8]) -> Result<Self, JpegError> {
        let view = JpegView::parse(input)?;
        DEFAULT_CONTEXT.with(|ctx| Self::from_view_in_context(view, &mut ctx.borrow_mut()))
    }

    /// Build a decoder from a previously parsed [`JpegView`].
    pub fn from_view(view: JpegView<'a>) -> Result<Self, JpegError> {
        DEFAULT_CONTEXT.with(|ctx| Self::from_view_in_context(view, &mut ctx.borrow_mut()))
    }

    /// Build a decoder from a previously parsed [`JpegView`], reusing shared
    /// compiled DHT/DQT state from `ctx` when table contents repeat.
    pub fn from_view_in_context(
        view: JpegView<'a>,
        ctx: &mut DecoderContext,
    ) -> Result<Self, JpegError> {
        let JpegView {
            bytes,
            header,
            info,
        } = view;
        let backend = Backend::detect();
        let (info, warnings, plan) = if let Some(scan_offset) = header.sos_offset {
            let header_prefix = &bytes[..scan_offset];
            ctx.resolve_decode_plan(header_prefix, |ctx| {
                let plan = Self::build_prepared_plan(&header, &info, ctx)?;
                Ok((
                    info.clone(),
                    Arc::<[Warning]>::from(header.warnings.as_slice()),
                    plan,
                ))
            })?
        } else {
            let plan = Self::build_prepared_plan(&header, &info, ctx)?;
            (
                info,
                Arc::<[Warning]>::from(header.warnings.as_slice()),
                plan,
            )
        };
        Ok(Self {
            bytes,
            info,
            warnings,
            backend,
            plan,
        })
    }

    fn build_prepared_plan(
        header: &ParsedHeader,
        info: &Info,
        ctx: &mut DecoderContext,
    ) -> Result<PreparedDecodePlan, JpegError> {
        match info.sof_kind {
            SofKind::Baseline8 | SofKind::Extended8 => {}
            other => return Err(JpegError::NotImplemented { sof: other }),
        }
        match info.color_space {
            ColorSpace::Grayscale | ColorSpace::YCbCr | ColorSpace::Rgb => {}
            color_space => return Err(JpegError::UnsupportedColorSpace { color_space }),
        }

        let mut dc_tables: [Option<Arc<HuffmanTable>>; 4] = Default::default();
        let mut ac_tables: [Option<Arc<HuffmanTable>>; 4] = Default::default();
        let scan = header.scan.as_ref().ok_or(JpegError::MissingMarker {
            marker: MarkerKind::Sos,
        })?;
        if header.scan_count != 1 {
            return Err(JpegError::InvalidSequentialScanCount {
                sof: info.sof_kind,
                count: header.scan_count,
            });
        }
        // M1b requires the first component (Y for 3-component, single for
        // grayscale) to be the maximally-sampled component. Non-luma-leading
        // layouts are pathological; real baselines always satisfy this.
        if let Some((h, v)) = header.sampling.component(0) {
            if h != header.sampling.max_h || v != header.sampling.max_v {
                return Err(JpegError::NotImplemented { sof: info.sof_kind });
            }
        }
        // Every component must declare H,V in 1..=4 per T.81 §B.2.2, and max_h
        // must actually divide every component's H (same for V). Malformed
        // streams can set H=0 (div-by-zero in upsample ratio), non-divisors
        // (arbitrary ratios M2 handles), or ratios that don't produce planes
        // that cover the image width.
        for (h, v) in header.sampling.iter() {
            if h == 0 || v == 0 || h > 4 || v > 4 {
                return Err(JpegError::NotImplemented { sof: info.sof_kind });
            }
            if !header.sampling.max_h.is_multiple_of(h) || !header.sampling.max_v.is_multiple_of(v)
            {
                return Err(JpegError::NotImplemented { sof: info.sof_kind });
            }
        }
        for comp in &scan.components {
            let di = comp.dc_table as usize;
            let ai = comp.ac_table as usize;
            if dc_tables[di].is_none() {
                let raw = header.huffman_tables.dc[di].as_ref().ok_or(
                    JpegError::MissingHuffmanTable {
                        component: comp.id,
                        class: 0,
                        id: comp.dc_table,
                    },
                )?;
                dc_tables[di] = Some(ctx.resolve_huffman_table(raw)?);
            }
            if ac_tables[ai].is_none() {
                let raw = header.huffman_tables.ac[ai].as_ref().ok_or(
                    JpegError::MissingHuffmanTable {
                        component: comp.id,
                        class: 1,
                        id: comp.ac_table,
                    },
                )?;
                ac_tables[ai] = Some(ctx.resolve_huffman_table(raw)?);
            }
        }

        build_decode_plan(header, info, &dc_tables, &ac_tables, ctx)
    }

    /// The parsed header as a public [`Info`].
    pub fn info(&self) -> &Info {
        &self.info
    }

    /// Decode the full image into the caller's buffer.
    ///
    /// # Errors
    /// - [`JpegError::OutputBufferTooSmall`] or [`JpegError::InvalidStride`]
    ///   if the provided buffer/stride cannot hold the image at `fmt`.
    /// - [`JpegError::NotImplemented`] if `fmt` requests a raw output the
    ///   current release does not emit (e.g. `RawYCbCr8`).
    /// - Any entropy- or structural-decode error from the scan walker.
    pub fn decode_into(
        &self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome, JpegError> {
        DEFAULT_SCRATCH
            .with(|pool| self.decode_into_with_scratch(&mut pool.borrow_mut(), out, stride, fmt))
    }

    /// Decode the full image into a freshly allocated tightly-packed buffer.
    ///
    /// This is the owned-output counterpart to [`Self::decode_into`]. It
    /// avoids pre-zeroing the destination buffer, which matters on WSI-sized
    /// RGB outputs where the allocation itself can otherwise dominate the
    /// benchmark.
    pub fn decode(&self, fmt: PixelFormat) -> Result<(Vec<u8>, DecodeOutcome), JpegError> {
        DEFAULT_SCRATCH.with(|pool| self.decode_with_scratch(&mut pool.borrow_mut(), fmt))
    }

    /// Decode the full image into the caller's buffer using the core
    /// `PixelFormat` + `Downscale` contract.
    pub fn decode_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<DecodeOutcome, JpegError> {
        DEFAULT_SCRATCH.with(|pool| {
            self.decode_scaled_into_with_scratch(&mut pool.borrow_mut(), out, stride, fmt, scale)
        })
    }

    /// Decode the full image into a freshly allocated tightly-packed buffer
    /// using the core `PixelFormat` + `Downscale` contract.
    pub fn decode_scaled(
        &self,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<(Vec<u8>, DecodeOutcome), JpegError> {
        DEFAULT_SCRATCH
            .with(|pool| self.decode_scaled_with_scratch(&mut pool.borrow_mut(), fmt, scale))
    }

    /// [`Self::decode`] with caller-owned scratch.
    pub fn decode_with_scratch(
        &self,
        pool: &mut ScratchPool,
        fmt: PixelFormat,
    ) -> Result<(Vec<u8>, DecodeOutcome), JpegError> {
        self.decode_scaled_with_scratch(pool, fmt, Downscale::None)
    }

    /// [`Self::decode_scaled`] with caller-owned scratch.
    pub fn decode_scaled_with_scratch(
        &self,
        pool: &mut ScratchPool,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<(Vec<u8>, DecodeOutcome), JpegError> {
        let legacy = output_format_from_parts(self.info.sof_kind, fmt, scale)?;
        let downscale = legacy.downscale();
        let (width, height) = scaled_dimensions(self.info.dimensions, downscale);
        let stride = width as usize * legacy.bytes_per_pixel();
        let len = stride * height as usize;
        let mut out = allocate_output_buffer(len);
        let outcome = self.decode_scaled_into_with_scratch(pool, &mut out, stride, fmt, scale)?;
        Ok((out, outcome))
    }

    /// Decode the full image into the caller's buffer, reusing the supplied
    /// [`ScratchPool`]. On a long-running tile batch this eliminates the
    /// per-tile allocation of stripe buffers, the DC predictor, and the
    /// chroma upsample rows — the realistic WSI reader shape. The first
    /// call against a fresh pool does the allocation; subsequent calls at
    /// the same-or-smaller shape reuse the underlying `Vec`s.
    ///
    /// # Errors
    /// Identical to [`Self::decode_into`].
    pub fn decode_into_with_scratch(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_scaled_into_with_scratch(pool, out, stride, fmt, Downscale::None)
    }

    fn decode_into_output_format_with_scratch(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: OutputFormat,
    ) -> Result<DecodeOutcome, JpegError> {
        let downscale = fmt.downscale();
        let (w, h) = scaled_dimensions(self.info.dimensions, downscale);
        let _ = self.decode_scratch_bytes(DEFAULT_MAX_DECODE_BYTES)?;
        let bpp = fmt.bytes_per_pixel();
        validate_buffer(out, stride, w, h, bpp)?;
        match fmt {
            OutputFormat::Rgb8 | OutputFormat::Rgb8Scaled { .. } => {
                let mut writer = Rgb8Writer::new(out, stride, w);
                self.decode_rgb_with_writer(
                    pool,
                    &mut writer,
                    downscale,
                    Rect::full(self.info.dimensions),
                )
            }
            OutputFormat::Rgba8 { alpha } => {
                let mut writer = Rgba8Writer::new(out, stride, w, alpha);
                self.decode_with_writer(
                    pool,
                    &mut writer,
                    downscale,
                    Rect::full(self.info.dimensions),
                )
            }
            OutputFormat::Gray8 | OutputFormat::Gray8Scaled { .. } => {
                let mut writer = Gray8Writer::new(out, stride, w);
                self.decode_with_writer(
                    pool,
                    &mut writer,
                    downscale,
                    Rect::full(self.info.dimensions),
                )
            }
        }
    }

    /// [`Self::decode_scaled_into`] with caller-owned scratch.
    pub fn decode_scaled_into_with_scratch(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_into_output_format_with_scratch(
            pool,
            out,
            stride,
            output_format_from_parts(self.info.sof_kind, fmt, scale)?,
        )
    }

    /// Decode the full image into interleaved RGB rows delivered to `sink`.
    pub fn decode_rows<S>(&self, sink: &mut S) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        DEFAULT_SCRATCH.with(|pool| self.decode_rows_with_scratch(&mut pool.borrow_mut(), sink))
    }

    /// [`Self::decode_rows`] with caller-owned scratch. See
    /// [`Self::decode_into_with_scratch`] for the reuse contract.
    pub fn decode_rows_with_scratch<S>(
        &self,
        pool: &mut ScratchPool,
        sink: &mut S,
    ) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        let width = self.info.dimensions.0 as usize;
        let rows = pool.take_sink_rows(width);
        let mut writer = SinkWriter::new(sink, rows, self.backend);
        let result = self.decode_rgb_with_writer(
            pool,
            &mut writer,
            DownscaleFactor::Full,
            Rect::full(self.info.dimensions),
        );
        pool.restore_sink_rows(writer.into_rows());
        result
    }

    /// Decode the full image into component rows.
    pub fn decode_component_rows_with_scratch<W>(
        &self,
        pool: &mut ScratchPool,
        writer: &mut W,
    ) -> Result<DecodeOutcome, JpegError>
    where
        W: ComponentRowWriter,
    {
        self.decode_region_component_rows_with_scratch(
            pool,
            writer,
            Rect::full(self.info.dimensions),
            Downscale::None,
        )
    }

    /// Decode `roi` into component rows, optionally at a reduced scale.
    pub fn decode_region_component_rows_with_scratch<W>(
        &self,
        pool: &mut ScratchPool,
        writer: &mut W,
        roi: Rect,
        scale: Downscale,
    ) -> Result<DecodeOutcome, JpegError>
    where
        W: ComponentRowWriter,
    {
        if !roi.is_within(self.info.dimensions) {
            return Err(JpegError::RectOutOfBounds {
                rect: roi,
                width: self.info.dimensions.0,
                height: self.info.dimensions.1,
            });
        }

        let downscale = jpeg_downscale(scale);
        let scaled_roi = scaled_rect_covering(roi, downscale)?;
        let mut adapter = ComponentWriterAdapter { inner: writer };

        if roi == Rect::full(self.info.dimensions) {
            self.decode_with_writer(pool, &mut adapter, downscale, roi)
        } else {
            let (scaled_width, _) = scaled_dimensions(self.info.dimensions, downscale);
            let mut cropped = CroppedWriter::new(adapter, scaled_roi, scaled_width);
            self.decode_with_writer(pool, &mut cropped, downscale, roi)
        }
    }

    /// Decode a rectangular region of the image into the caller's buffer.
    ///
    /// `roi` is expressed in source-image coordinates. If `fmt` requests a
    /// downscaled output, the written pixels cover the corresponding bounding
    /// box in the scaled image grid.
    pub fn decode_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        DEFAULT_SCRATCH.with(|pool| {
            self.decode_region_into_with_scratch(&mut pool.borrow_mut(), out, stride, fmt, roi)
        })
    }

    /// Decode `roi` into a freshly allocated tightly-packed buffer.
    pub fn decode_region(
        &self,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<(Vec<u8>, DecodeOutcome), JpegError> {
        DEFAULT_SCRATCH
            .with(|pool| self.decode_region_with_scratch(&mut pool.borrow_mut(), fmt, roi))
    }

    /// Decode `roi` into a freshly allocated tightly-packed buffer using the
    /// core `PixelFormat` + `Downscale` contract.
    pub fn decode_region_scaled(
        &self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<(Vec<u8>, DecodeOutcome), JpegError> {
        DEFAULT_SCRATCH.with(|pool| {
            self.decode_region_scaled_with_scratch(&mut pool.borrow_mut(), fmt, roi, scale)
        })
    }

    /// [`Self::decode_region`] with caller-owned scratch.
    pub fn decode_region_with_scratch(
        &self,
        pool: &mut ScratchPool,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<(Vec<u8>, DecodeOutcome), JpegError> {
        self.decode_region_scaled_with_scratch(pool, fmt, roi, Downscale::None)
    }

    /// [`Self::decode_region_scaled`] with caller-owned scratch.
    pub fn decode_region_scaled_with_scratch(
        &self,
        pool: &mut ScratchPool,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<(Vec<u8>, DecodeOutcome), JpegError> {
        let legacy = output_format_from_parts(self.info.sof_kind, fmt, scale)?;
        let scaled_roi = scaled_rect_covering(roi, legacy.downscale())?;
        let stride = scaled_roi.w as usize * legacy.bytes_per_pixel();
        let len = stride * scaled_roi.h as usize;
        let mut out = allocate_output_buffer(len);
        let outcome =
            self.decode_region_scaled_into_with_scratch(pool, &mut out, stride, fmt, roi, scale)?;
        Ok((out, outcome))
    }

    /// [`Self::decode_region_into`] with caller-owned scratch.
    pub fn decode_region_into_with_scratch(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_region_scaled_into_with_scratch(pool, out, stride, fmt, roi, Downscale::None)
    }

    fn decode_region_into_output_format_with_scratch(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: OutputFormat,
        roi: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        if !roi.is_within(self.info.dimensions) {
            return Err(JpegError::RectOutOfBounds {
                rect: roi,
                width: self.info.dimensions.0,
                height: self.info.dimensions.1,
            });
        }

        let downscale = fmt.downscale();
        let scaled_roi = scaled_rect_covering(roi, downscale)?;
        let _ = self.decode_scratch_bytes(DEFAULT_MAX_DECODE_BYTES)?;
        validate_buffer(
            out,
            stride,
            scaled_roi.w,
            scaled_roi.h,
            fmt.bytes_per_pixel(),
        )?;

        match fmt {
            OutputFormat::Rgb8 | OutputFormat::Rgb8Scaled { .. } => {
                if fmt == OutputFormat::Rgb8
                    && downscale == DownscaleFactor::Full
                    && self.plan.matches_fast_tile_shape()
                {
                    let mut writer = Rgb8Writer::new(out, stride, scaled_roi.w);
                    let scan_bytes = &self.bytes[self.plan.scan_offset..];
                    let scan_warnings = decode_scan_fast_tile_rgb_region(
                        &self.plan,
                        self.backend,
                        scan_bytes,
                        pool,
                        &mut writer,
                        roi,
                    )?;
                    Ok(DecodeOutcome {
                        decoded: roi,
                        warnings: merged_warnings(&self.warnings, scan_warnings),
                    })
                } else {
                    let base = Rgb8Writer::new(out, stride, scaled_roi.w);
                    let (scaled_width, _) = scaled_dimensions(self.info.dimensions, downscale);
                    let mut writer = CroppedWriter::new(base, scaled_roi, scaled_width);
                    self.decode_rgb_with_writer(pool, &mut writer, downscale, roi)
                }
            }
            OutputFormat::Rgba8 { alpha } => {
                let base = Rgba8Writer::new(out, stride, scaled_roi.w, alpha);
                let (scaled_width, _) = scaled_dimensions(self.info.dimensions, downscale);
                let mut writer = CroppedWriter::new(base, scaled_roi, scaled_width);
                self.decode_with_writer(pool, &mut writer, downscale, roi)
            }
            OutputFormat::Gray8 | OutputFormat::Gray8Scaled { .. } => {
                let base = Gray8Writer::new(out, stride, scaled_roi.w);
                let (scaled_width, _) = scaled_dimensions(self.info.dimensions, downscale);
                let mut writer = CroppedWriter::new(base, scaled_roi, scaled_width);
                self.decode_with_writer(pool, &mut writer, downscale, roi)
            }
        }
    }

    /// Decode `roi` into the caller's buffer using the core `PixelFormat` +
    /// `Downscale` contract.
    pub fn decode_region_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<DecodeOutcome, JpegError> {
        DEFAULT_SCRATCH.with(|pool| {
            self.decode_region_scaled_into_with_scratch(
                &mut pool.borrow_mut(),
                out,
                stride,
                fmt,
                roi,
                scale,
            )
        })
    }

    /// [`Self::decode_region_scaled_into`] with caller-owned scratch.
    pub fn decode_region_scaled_into_with_scratch(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_region_into_output_format_with_scratch(
            pool,
            out,
            stride,
            output_format_from_parts(self.info.sof_kind, fmt, scale)?,
            roi,
        )
    }

    /// Decode the full image into RGBA with a caller-chosen alpha byte.
    pub fn decode_rgba8_into_with_alpha(
        &self,
        out: &mut [u8],
        stride: usize,
        alpha: u8,
    ) -> Result<DecodeOutcome, JpegError> {
        DEFAULT_SCRATCH.with(|pool| {
            self.decode_rgba8_into_with_alpha_with_scratch(
                &mut pool.borrow_mut(),
                out,
                stride,
                alpha,
            )
        })
    }

    /// [`Self::decode_rgba8_into_with_alpha`] with caller-owned scratch.
    pub fn decode_rgba8_into_with_alpha_with_scratch(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        alpha: u8,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_into_output_format_with_scratch(
            pool,
            out,
            stride,
            OutputFormat::Rgba8 { alpha },
        )
    }

    /// Decode a region into RGBA with a caller-chosen alpha byte.
    pub fn decode_region_rgba8_into_with_alpha(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        alpha: u8,
    ) -> Result<DecodeOutcome, JpegError> {
        DEFAULT_SCRATCH.with(|pool| {
            self.decode_region_rgba8_into_with_alpha_with_scratch(
                &mut pool.borrow_mut(),
                out,
                stride,
                roi,
                alpha,
            )
        })
    }

    /// [`Self::decode_region_rgba8_into_with_alpha`] with caller-owned scratch.
    pub fn decode_region_rgba8_into_with_alpha_with_scratch(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        alpha: u8,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_region_into_output_format_with_scratch(
            pool,
            out,
            stride,
            OutputFormat::Rgba8 { alpha },
            roi,
        )
    }
}

/// One-shot parse-plus-decode of an independent JPEG tile into the caller's
/// buffer, reusing a pre-allocated [`ScratchPool`]. This is the primitive
/// WSI tile-batch readers want: one function call per tile, with all
/// heap state external.
///
/// Parallelism is the caller's responsibility. The idiomatic shape is
/// [`std::thread::scope`] with one `ScratchPool` per worker thread —
/// no crate dependency on `rayon`.
///
/// # Example
///
/// ```no_run
/// use slidecodec_jpeg::{decode_tile_into, PixelFormat, ScratchPool};
///
/// let bytes: &[u8] = todo!("read tile bytes");
/// let mut out = vec![0u8; 256 * 256 * 3];
/// let mut pool = ScratchPool::new();
/// decode_tile_into(bytes, &mut pool, &mut out, 256 * 3, PixelFormat::Rgb8)?;
/// # Ok::<(), slidecodec_jpeg::JpegError>(())
/// ```
///
/// # Errors
/// Forwarded from [`Decoder::new`] (parse) and
/// [`Decoder::decode_into_with_scratch`] (decode).
pub fn decode_tile_into(
    bytes: &[u8],
    pool: &mut ScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<DecodeOutcome, JpegError> {
    DEFAULT_CONTEXT.with(|ctx| {
        decode_tile_into_in_context(bytes, &mut ctx.borrow_mut(), pool, out, stride, fmt)
    })
}

/// One-shot parse-plus-decode of an independent JPEG tile into the caller's
/// buffer, reusing both caller-owned [`DecoderContext`] and caller-owned
/// [`ScratchPool`].
pub fn decode_tile_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<DecodeOutcome, JpegError> {
    let dec = Decoder::from_view_in_context(JpegView::parse(bytes)?, ctx)?;
    dec.decode_into_with_scratch(pool, out, stride, fmt)
}

/// One-shot parse-plus-region-decode of an independent JPEG tile into the
/// caller's buffer, reusing both caller-owned [`DecoderContext`] and
/// caller-owned [`ScratchPool`].
pub fn decode_tile_region_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<DecodeOutcome, JpegError> {
    let dec = Decoder::from_view_in_context(JpegView::parse(bytes)?, ctx)?;
    dec.decode_region_into_with_scratch(pool, out, stride, fmt, roi)
}

/// One-shot parse-plus-scaled-decode of an independent JPEG tile into the
/// caller's buffer, reusing both caller-owned [`DecoderContext`] and
/// caller-owned [`ScratchPool`].
pub fn decode_tile_scaled_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    scale: Downscale,
) -> Result<DecodeOutcome, JpegError> {
    let dec = Decoder::from_view_in_context(JpegView::parse(bytes)?, ctx)?;
    dec.decode_scaled_into_with_scratch(pool, out, stride, fmt, scale)
}

/// One-shot parse-plus-region-scaled-decode of an independent JPEG tile into
/// the caller's buffer, reusing both caller-owned [`DecoderContext`] and
/// caller-owned [`ScratchPool`].
#[allow(clippy::too_many_arguments)]
pub fn decode_tile_region_scaled_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
) -> Result<DecodeOutcome, JpegError> {
    let dec = Decoder::from_view_in_context(JpegView::parse(bytes)?, ctx)?;
    dec.decode_region_scaled_into_with_scratch(pool, out, stride, fmt, roi, scale)
}

impl Decoder<'_> {
    /// One-shot parse-plus-row-decode of a JPEG tile using caller-owned shared
    /// table context and caller-owned scratch.
    pub fn decode_tile<S>(
        bytes: &[u8],
        ctx: &mut DecoderContext,
        pool: &mut ScratchPool,
        sink: &mut S,
    ) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        let dec = Decoder::from_view_in_context(JpegView::parse(bytes)?, ctx)?;
        dec.decode_rows_with_scratch(pool, sink)
    }
}

impl Decoder<'_> {
    fn decode_scratch_bytes(&self, cap: usize) -> Result<usize, JpegError> {
        if self.plan.scratch_bytes > cap {
            return Err(JpegError::MemoryCapExceeded {
                requested: self.plan.scratch_bytes,
                cap,
            });
        }
        Ok(self.plan.scratch_bytes)
    }

    fn decode_with_writer<W: OutputWriter>(
        &self,
        pool: &mut ScratchPool,
        writer: &mut W,
        downscale: DownscaleFactor,
        decoded: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        let _ = self.decode_scratch_bytes(DEFAULT_MAX_DECODE_BYTES)?;
        let output_rect = scaled_rect_covering(decoded, downscale)?;
        let scan_bytes = &self.bytes[self.plan.scan_offset..];
        let scan_warnings = decode_scan_baseline(
            &self.plan,
            self.backend,
            scan_bytes,
            pool,
            writer,
            downscale,
            output_rect,
        )?;
        Ok(DecodeOutcome {
            decoded,
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }

    fn decode_rgb_with_writer<W: OutputWriter + InterleavedRgbWriter>(
        &self,
        pool: &mut ScratchPool,
        writer: &mut W,
        downscale: DownscaleFactor,
        decoded: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        let _ = self.decode_scratch_bytes(DEFAULT_MAX_DECODE_BYTES)?;
        let output_rect = scaled_rect_covering(decoded, downscale)?;
        let scan_bytes = &self.bytes[self.plan.scan_offset..];
        let scan_warnings =
            if downscale == DownscaleFactor::Full && self.plan.matches_fast_tile_shape() {
                decode_scan_fast_tile_rgb(&self.plan, self.backend, scan_bytes, pool, writer)?
            } else if downscale == DownscaleFactor::Full
                && decoded == Rect::full(self.info.dimensions)
                && self.plan.matches_fast_rgb444_shape()
            {
                decode_scan_fast_rgb_444(&self.plan, self.backend, scan_bytes, pool, writer)?
            } else {
                decode_scan_baseline_rgb(
                    &self.plan,
                    self.backend,
                    scan_bytes,
                    pool,
                    writer,
                    downscale,
                    output_rect,
                )?
            };
        Ok(DecodeOutcome {
            decoded,
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }
}

fn merged_warnings(header_warnings: &[Warning], scan_warnings: Vec<Warning>) -> Vec<Warning> {
    if header_warnings.is_empty() {
        return scan_warnings;
    }
    if scan_warnings.is_empty() {
        return header_warnings.to_vec();
    }
    let mut warnings = Vec::with_capacity(header_warnings.len() + scan_warnings.len());
    warnings.extend_from_slice(header_warnings);
    warnings.extend(scan_warnings);
    warnings
}

fn core_colorspace(color_space: ColorSpace) -> CoreColorspace {
    match color_space {
        ColorSpace::Grayscale => CoreColorspace::Grayscale,
        ColorSpace::YCbCr => CoreColorspace::YCbCr,
        ColorSpace::Rgb => CoreColorspace::Rgb,
        ColorSpace::Cmyk => CoreColorspace::Cmyk,
        ColorSpace::Ycck => CoreColorspace::Ycck,
    }
}

fn core_info(info: &Info) -> slidecodec_core::Info {
    slidecodec_core::Info {
        dimensions: info.dimensions,
        components: info.sampling.len() as u8,
        colorspace: core_colorspace(info.color_space),
        bit_depth: info.bit_depth,
        tile_layout: None,
        resolution_levels: 1,
    }
}

fn core_rect(rect: Rect) -> slidecodec_core::Rect {
    slidecodec_core::Rect {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: rect.h,
    }
}

fn jpeg_rect(rect: slidecodec_core::Rect) -> Rect {
    Rect {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: rect.h,
    }
}

fn core_outcome(outcome: DecodeOutcome) -> CoreDecodeOutcome<Warning> {
    CoreDecodeOutcome {
        decoded: core_rect(outcome.decoded),
        warnings: outcome.warnings,
    }
}

fn jpeg_downscale(scale: Downscale) -> DownscaleFactor {
    match scale {
        Downscale::None => DownscaleFactor::Full,
        Downscale::Half => DownscaleFactor::Half,
        Downscale::Quarter => DownscaleFactor::Quarter,
        Downscale::Eighth => DownscaleFactor::Eighth,
        _ => unreachable!("unsupported Downscale variant"),
    }
}

fn output_format_from_parts(
    sof_kind: SofKind,
    fmt: PixelFormat,
    scale: Downscale,
) -> Result<OutputFormat, JpegError> {
    match (fmt, scale) {
        (PixelFormat::Rgb8, Downscale::None) => Ok(OutputFormat::Rgb8),
        (PixelFormat::Rgb8, scale) => Ok(OutputFormat::Rgb8Scaled {
            factor: jpeg_downscale(scale),
        }),
        (PixelFormat::Gray8, Downscale::None) => Ok(OutputFormat::Gray8),
        (PixelFormat::Gray8, scale) => Ok(OutputFormat::Gray8Scaled {
            factor: jpeg_downscale(scale),
        }),
        (PixelFormat::Rgba8, Downscale::None) => Ok(OutputFormat::Rgba8 { alpha: 255 }),
        (PixelFormat::Rgba8, _) => Err(JpegError::DownscaleUnsupported { sof: sof_kind }),
        (PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16, _) => {
            Err(JpegError::UnsupportedBitDepth { depth: 16 })
        }
        _ => Err(JpegError::DownscaleUnsupported { sof: sof_kind }),
    }
}

impl ImageCodec for JpegCodec {
    type Error = JpegError;
    type Warning = Warning;
    type Pool = ScratchPool;
}

impl ImageCodec for Decoder<'_> {
    type Error = JpegError;
    type Warning = Warning;
    type Pool = ScratchPool;
}

impl<'a> ImageDecode<'a> for Decoder<'a> {
    type View = JpegView<'a>;

    fn inspect(input: &'a [u8]) -> Result<slidecodec_core::Info, Self::Error> {
        Ok(core_info(&Decoder::inspect(input)?))
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
        roi: slidecodec_core::Rect,
    ) -> Result<CoreDecodeOutcome<Self::Warning>, Self::Error> {
        Decoder::decode_region_into_with_scratch(self, pool, out, stride, fmt, jpeg_rect(roi))
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
}

struct CoreRowSinkAdapter<'a, R: RowSink<u8>> {
    sink: &'a mut R,
    sink_error: Option<R::Error>,
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
            Err(JpegError::RowSinkAborted) => Err(DecodeRowsError::Sink(
                adapter
                    .sink_error
                    .expect("row sink abort stores the original sink error"),
            )),
            Err(err) => Err(DecodeRowsError::Decode(err)),
        }
    }
}

impl TileBatchDecode for JpegCodec {
    type Context = DecoderContext;

    fn decode_tile(
        ctx: &mut CoreDecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<CoreDecodeOutcome<Self::Warning>, Self::Error> {
        let dec = Decoder::from_view_in_context(JpegView::parse(input)?, ctx.codec_mut())?;
        dec.decode_into_with_scratch(pool, out, stride, fmt)
            .map(core_outcome)
    }

    fn decode_tile_region(
        ctx: &mut CoreDecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: slidecodec_core::Rect,
    ) -> Result<CoreDecodeOutcome<Self::Warning>, Self::Error> {
        let dec = Decoder::from_view_in_context(JpegView::parse(input)?, ctx.codec_mut())?;
        dec.decode_region_into_with_scratch(pool, out, stride, fmt, jpeg_rect(roi))
            .map(core_outcome)
    }

    fn decode_tile_scaled(
        ctx: &mut CoreDecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<CoreDecodeOutcome<Self::Warning>, Self::Error> {
        let dec = Decoder::from_view_in_context(JpegView::parse(input)?, ctx.codec_mut())?;
        dec.decode_scaled_into_with_scratch(pool, out, stride, fmt, scale)
            .map(core_outcome)
    }
}

#[allow(clippy::uninit_vec)]
fn allocate_output_buffer(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    // Safety: all owned-output entrypoints use tight row strides, and the
    // decode writers fully initialize every byte in the destination on success.
    // If decode returns an error, dropping a Vec<u8> with uninitialized bytes is
    // still sound because `u8` has no drop glue.
    unsafe {
        out.set_len(len);
    }
    out
}

fn scaled_dimensions(dims: (u32, u32), factor: DownscaleFactor) -> (u32, u32) {
    let denom = factor.denominator();
    (dims.0.div_ceil(denom), dims.1.div_ceil(denom))
}

fn scaled_rect_covering(rect: Rect, factor: DownscaleFactor) -> Result<Rect, JpegError> {
    let denom = factor.denominator();
    let x_end = rect
        .x
        .checked_add(rect.w)
        .ok_or(JpegError::RectOutOfBounds {
            rect,
            width: u32::MAX,
            height: u32::MAX,
        })?;
    let y_end = rect
        .y
        .checked_add(rect.h)
        .ok_or(JpegError::RectOutOfBounds {
            rect,
            width: u32::MAX,
            height: u32::MAX,
        })?;
    let x0 = rect.x / denom;
    let y0 = rect.y / denom;
    let x1 = x_end.div_ceil(denom);
    let y1 = y_end.div_ceil(denom);
    Ok(Rect {
        x: x0,
        y: y0,
        w: x1.saturating_sub(x0),
        h: y1.saturating_sub(y0),
    })
}

struct CroppedWriter<W> {
    inner: W,
    rect: Rect,
    source_width: u32,
    top_row: Vec<u8>,
    bottom_row: Vec<u8>,
}

struct ComponentWriterAdapter<'a, W> {
    inner: &'a mut W,
}

impl<W: ComponentRowWriter> OutputWriter for ComponentWriterAdapter<'_, W> {
    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), JpegError> {
        self.inner.write_rgb_row(y, r_row, g_row, b_row)
    }

    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
    ) -> Result<(), JpegError> {
        self.inner.write_ycbcr_row(y, y_row, cb_row, cr_row)
    }

    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), JpegError> {
        self.inner.write_gray_row(y, gray_row)
    }
}

impl<W> CroppedWriter<W> {
    fn new(inner: W, rect: Rect, source_width: u32) -> Self {
        let row_len = source_width as usize * 3;
        Self {
            inner,
            rect,
            source_width,
            top_row: vec![0; row_len],
            bottom_row: vec![0; row_len],
        }
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
        let x0 = self.rect.x as usize;
        let x1 = x0 + self.rect.w as usize;
        self.inner.write_rgb_row(
            y - self.rect.y,
            &r_row[x0..x1],
            &g_row[x0..x1],
            &b_row[x0..x1],
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
        let x0 = self.rect.x as usize;
        let x1 = x0 + self.rect.w as usize;
        self.inner.write_ycbcr_row(
            y - self.rect.y,
            &y_row[x0..x1],
            &cb_row[x0..x1],
            &cr_row[x0..x1],
        )
    }

    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), JpegError> {
        if y < self.rect.y || y >= self.rect.y + self.rect.h {
            return Ok(());
        }
        let x0 = self.rect.x as usize;
        let x1 = x0 + self.rect.w as usize;
        self.inner
            .write_gray_row(y - self.rect.y, &gray_row[x0..x1])
    }
}

impl<W: InterleavedRgbWriter> InterleavedRgbWriter for CroppedWriter<W> {
    fn with_rgb_rows<R, F>(&mut self, y: u32, row_count: usize, fill: F) -> Result<R, JpegError>
    where
        F: FnOnce(&mut [u8], Option<&mut [u8]>) -> Result<R, JpegError>,
    {
        let row_len = self.source_width as usize * 3;
        if self.top_row.len() != row_len {
            self.top_row.resize(row_len, 0);
            self.bottom_row.resize(row_len, 0);
        }

        let result = match row_count {
            1 => fill(&mut self.top_row, None)?,
            2 => fill(&mut self.top_row, Some(&mut self.bottom_row))?,
            _ => unreachable!("CroppedWriter only supports one or two rows"),
        };

        let top_in = y >= self.rect.y && y < self.rect.y + self.rect.h;
        let bottom_y = y + 1;
        let bottom_in =
            row_count == 2 && bottom_y >= self.rect.y && bottom_y < self.rect.y + self.rect.h;
        let x0 = self.rect.x as usize * 3;
        let x1 = x0 + self.rect.w as usize * 3;

        match (top_in, bottom_in) {
            (false, false) => {}
            (true, false) => {
                self.inner.with_rgb_rows(y - self.rect.y, 1, |dst, _| {
                    dst.copy_from_slice(&self.top_row[x0..x1]);
                    Ok(())
                })?;
            }
            (false, true) => {
                self.inner
                    .with_rgb_rows(bottom_y - self.rect.y, 1, |dst, _| {
                        dst.copy_from_slice(&self.bottom_row[x0..x1]);
                        Ok(())
                    })?;
            }
            (true, true) => {
                self.inner
                    .with_rgb_rows(y - self.rect.y, 2, |dst_top, dst_bottom| {
                        dst_top.copy_from_slice(&self.top_row[x0..x1]);
                        dst_bottom
                            .expect("row_count=2 supplies bottom row")
                            .copy_from_slice(&self.bottom_row[x0..x1]);
                        Ok(())
                    })?;
            }
        }

        Ok(result)
    }
}

fn build_decode_plan(
    header: &ParsedHeader,
    info: &Info,
    dc_tables: &[Option<Arc<HuffmanTable>>; 4],
    ac_tables: &[Option<Arc<HuffmanTable>>; 4],
    ctx: &mut DecoderContext,
) -> Result<PreparedDecodePlan, JpegError> {
    let scan = header.scan.as_ref().ok_or(JpegError::MissingMarker {
        marker: MarkerKind::Sos,
    })?;
    let scan_offset = header.sos_offset.ok_or(JpegError::MissingMarker {
        marker: MarkerKind::Sos,
    })?;

    let mut components = Vec::with_capacity(scan.components.len());
    for scan_comp in scan.components.iter().copied() {
        let output_index = find_component_index(&header.component_ids, scan_comp.id).ok_or(
            JpegError::UnknownScanComponent {
                offset: scan_offset,
                component: scan_comp.id,
            },
        )?;
        let (h, v) = header
            .sampling
            .component(output_index)
            .ok_or(JpegError::MissingMarker {
                marker: MarkerKind::Sof,
            })?;
        let quant_id =
            *header
                .quant_table_ids
                .get(output_index)
                .ok_or(JpegError::MissingMarker {
                    marker: MarkerKind::Sof,
                })? as usize;
        let quant = *header
            .quant_tables
            .entries
            .get(quant_id)
            .and_then(|q| q.as_ref())
            .ok_or(JpegError::MissingQuantTable {
                component: scan_comp.id,
                table_id: quant_id as u8,
            })?;
        let dc_table = dc_tables[scan_comp.dc_table as usize].as_ref().ok_or(
            JpegError::MissingHuffmanTable {
                component: scan_comp.id,
                class: 0,
                id: scan_comp.dc_table,
            },
        )?;
        let ac_table = ac_tables[scan_comp.ac_table as usize].as_ref().ok_or(
            JpegError::MissingHuffmanTable {
                component: scan_comp.id,
                class: 1,
                id: scan_comp.ac_table,
            },
        )?;
        components.push(PreparedComponentPlan {
            h,
            v,
            output_index,
            quant: ctx.resolve_quant_table(quant),
            dc_table: Arc::clone(dc_table),
            ac_table: Arc::clone(ac_table),
        });
    }

    let scratch_bytes =
        compute_decode_scratch_bytes(info.dimensions, info.sampling, DEFAULT_MAX_DECODE_BYTES)?;

    Ok(PreparedDecodePlan {
        components,
        sampling: info.sampling,
        color_space: info.color_space,
        restart_interval: header.restart_interval,
        dimensions: info.dimensions,
        scan_offset,
        scratch_bytes,
    })
}

struct SinkWriter<'a, S> {
    sink: &'a mut S,
    rows: SinkRows,
    backend: Backend,
}

impl<'a, S> SinkWriter<'a, S> {
    fn new(sink: &'a mut S, rows: SinkRows, backend: Backend) -> Self {
        debug_assert_eq!(rows.top_row.len(), rows.bottom_row.len());
        Self {
            sink,
            rows,
            backend,
        }
    }

    fn into_rows(self) -> SinkRows {
        self.rows
    }
}

impl<S> InterleavedRgbWriter for SinkWriter<'_, S>
where
    S: RowSink<u8, Error = JpegError>,
{
    fn with_rgb_rows<R, F>(&mut self, y: u32, row_count: usize, fill: F) -> Result<R, JpegError>
    where
        F: FnOnce(&mut [u8], Option<&mut [u8]>) -> Result<R, JpegError>,
    {
        let result = match row_count {
            1 => fill(&mut self.rows.top_row, None),
            2 => fill(&mut self.rows.top_row, Some(&mut self.rows.bottom_row)),
            _ => unreachable!("SinkWriter only supports one or two rows"),
        }?;
        self.sink.write_row(y, &self.rows.top_row)?;
        if row_count == 2 {
            self.sink.write_row(y + 1, &self.rows.bottom_row)?;
        }
        Ok(result)
    }
}

impl<S> OutputWriter for SinkWriter<'_, S>
where
    S: RowSink<u8, Error = JpegError>,
{
    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), JpegError> {
        self.backend
            .fill_rgb_row_from_rgb(r_row, g_row, b_row, &mut self.rows.top_row);
        self.sink.write_row(y, &self.rows.top_row)
    }

    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
    ) -> Result<(), JpegError> {
        self.backend
            .fill_rgb_row_from_ycbcr(y_row, cb_row, cr_row, &mut self.rows.top_row);
        self.sink.write_row(y, &self.rows.top_row)
    }

    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), JpegError> {
        self.backend
            .fill_rgb_row_from_gray(gray_row, &mut self.rows.top_row);
        self.sink.write_row(y, &self.rows.top_row)
    }
}

fn find_component_index(component_ids: &[u8], id: u8) -> Option<usize> {
    component_ids
        .iter()
        .position(|&component_id| component_id == id)
}

fn compute_decode_scratch_bytes(
    (width, height): (u32, u32),
    sampling: crate::info::SamplingFactors,
    cap: usize,
) -> Result<usize, JpegError> {
    let max_h = u32::from(sampling.max_h);
    let max_v = u32::from(sampling.max_v);
    let mcu_width = 8u32
        .checked_mul(max_h)
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        })?;
    let mcu_height = 8u32
        .checked_mul(max_v)
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        })?;
    let mcus_per_row = width.div_ceil(mcu_width);
    let _mcu_rows = height.div_ceil(mcu_height);

    let mut stripe_total = 0usize;
    for (h, v) in sampling.iter() {
        let cols = checked_usize_product(&[mcus_per_row as usize, usize::from(h), 8usize], cap)?;
        let rows = checked_usize_product(&[usize::from(v), 8usize], cap)?;
        let plane = cols.checked_mul(rows).ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        })?;
        stripe_total = stripe_total
            .checked_add(plane)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;
        if stripe_total > cap {
            return Err(JpegError::MemoryCapExceeded {
                requested: stripe_total,
                cap,
            });
        }
    }

    let stripe_buffers = checked_usize_product(&[stripe_total, 3], cap)?;
    let row_scratch = checked_usize_product(&[width as usize, 7], cap)?;
    let total = stripe_buffers
        .checked_add(row_scratch)
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        })?;
    if total > cap {
        return Err(JpegError::MemoryCapExceeded {
            requested: total,
            cap,
        });
    }

    Ok(total)
}

fn checked_usize_product(factors: &[usize], cap: usize) -> Result<usize, JpegError> {
    let mut value = 1usize;
    for factor in factors {
        value = value
            .checked_mul(*factor)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Warning;
    use alloc::vec;
    use alloc::vec::Vec;

    fn minimal_baseline_jpeg() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&[0xFF, 0xD8]);
        v.extend_from_slice(&[0xFF, 0xDB, 0x00, 67, 0x00]);
        v.extend(core::iter::repeat_n(1u8, 64));
        v.extend_from_slice(&[
            0xFF,
            0xC0,
            0x00,
            17,
            8,
            0,
            16,
            0,
            16,
            3,
            1,
            (2 << 4) | 2,
            0,
            2,
            (1 << 4) | 1,
            0,
            3,
            (1 << 4) | 1,
            0,
        ]);
        v.extend_from_slice(&[
            0xFF, 0xC4, 0x00, 20, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xAA,
        ]);
        v.extend_from_slice(&[
            0xFF, 0xC4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xBB,
        ]);
        v.extend_from_slice(&[0xFF, 0xDA, 0x00, 12, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0]);
        v.extend_from_slice(&[0x00, 0xFF, 0xD9]);
        v
    }

    #[test]
    fn decoder_new_succeeds_on_baseline_stream() {
        let bytes = minimal_baseline_jpeg();
        let dec = Decoder::new(&bytes).unwrap();
        assert_eq!(dec.info().dimensions, (16, 16));
    }

    #[test]
    fn decoder_new_rejects_progressive_with_not_implemented() {
        let mut bytes = minimal_baseline_jpeg();
        let p = bytes.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        bytes[p + 1] = 0xC2;
        let err = Decoder::new(&bytes).unwrap_err();
        assert!(err.is_not_implemented());
    }

    #[test]
    fn decoder_new_rejects_arithmetic_as_unsupported() {
        let mut bytes = minimal_baseline_jpeg();
        let p = bytes.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        bytes[p + 1] = 0xC9;
        let err = Decoder::new(&bytes).unwrap_err();
        assert!(err.is_unsupported());
    }

    #[test]
    fn decode_outcome_carries_rect_and_warnings() {
        let outcome = DecodeOutcome {
            decoded: Rect {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
            },
            warnings: vec![Warning::MissingEoi],
        };
        assert_eq!(outcome.decoded.w, 16);
        assert_eq!(outcome.warnings.len(), 1);
    }

    #[test]
    fn decode_into_rejects_undersized_buffer() {
        let bytes = minimal_baseline_jpeg();
        let dec = Decoder::new(&bytes).unwrap();
        let mut buf = vec![0u8; 4];
        let err = dec
            .decode_into(&mut buf, 48, PixelFormat::Rgb8)
            .unwrap_err();
        assert!(matches!(err, JpegError::OutputBufferTooSmall { .. }));
    }

    #[test]
    fn decode_into_rejects_invalid_stride() {
        let bytes = minimal_baseline_jpeg();
        let dec = Decoder::new(&bytes).unwrap();
        let mut buf = vec![0u8; 16 * 16 * 3];
        let err = dec
            .decode_into(&mut buf, 10, PixelFormat::Rgb8)
            .unwrap_err();
        assert!(matches!(err, JpegError::InvalidStride { .. }));
    }
}
