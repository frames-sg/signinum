// SPDX-License-Identifier: Apache-2.0

//! Public [`Decoder`] entry points.

use crate::backend::Backend;
use crate::context::DecoderContext;
use crate::entropy::huffman::HuffmanTable;
use crate::entropy::progressive::{
    decode_progressive, PreparedProgressiveComponentPlan, PreparedProgressivePlan,
    PreparedProgressiveScan, PreparedProgressiveScanComponent,
};
use crate::entropy::sequential::{
    decode_scan_baseline, decode_scan_baseline_rgb, decode_scan_fast_rgb_444,
    decode_scan_fast_tile_rgb, decode_scan_fast_tile_rgb_region,
    decode_scan_fast_tile_rgb_region_scaled, fast_tile_region_first_decode_mcu,
    stripe_region_layout, PreparedComponentPlan, PreparedDecodePlan,
};
use crate::error::{JpegError, MarkerKind, Warning};
use crate::info::{
    ColorSpace, DecodeOptions, DownscaleFactor, Info, OutputFormat, Rect, RestartIndex,
    RestartSegment, SofKind,
};
use crate::internal::checkpoint::{checkpoint_before_mcu, CpuCheckpointCache, DeviceCheckpoint};
use crate::internal::scratch::{ScratchPool, SinkRows};
use crate::output::{
    validate_buffer, Gray8Writer, InterleavedRgbWriter, OutputWriter, Rgb8Writer, Rgba8Writer,
};
use crate::parse::header::{parse_header, parse_info, ParsedHeader};
use crate::parse::tables::{HuffmanValues, RawHuffmanTable};
use crate::profile::{duration_us_string, emit_jpeg_profile_row, jpeg_profile_stages_enabled};
use crate::JpegCodec;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::num::NonZeroUsize;
use signinum_core::{
    CompressedPayloadKind, CompressedTransferSyntax, DecodeOutcome as CoreDecodeOutcome,
    DecodeRowsError, DecoderContext as CoreDecoderContext, Downscale, ImageCodec, ImageDecode,
    ImageDecodeRows, PassthroughCandidate, PixelFormat, RowSink, TileBatchDecode,
};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const DEFAULT_MAX_DECODE_BYTES: usize = 512 * 1024 * 1024;
const CPU_ROI_CHECKPOINT_CADENCE_MCUS: u32 = 1024;
const CPU_ROI_CHECKPOINT_MIN_TARGET_MCUS: u32 = 4096;

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

/// One tile decode request for [`decode_tiles_into`].
pub struct TileDecodeJob<'i, 'o> {
    /// Compressed JPEG tile bytes.
    pub input: &'i [u8],
    /// Caller-owned output buffer for this tile.
    pub out: &'o mut [u8],
    /// Distance in bytes between output rows.
    pub stride: usize,
}

/// One scaled tile decode request for [`decode_tiles_scaled_into`].
pub struct TileScaledDecodeJob<'i, 'o> {
    /// Compressed JPEG tile bytes.
    pub input: &'i [u8],
    /// Caller-owned output buffer for this tile.
    pub out: &'o mut [u8],
    /// Distance in bytes between output rows.
    pub stride: usize,
    /// Downscale factor applied to the full-tile decode.
    pub scale: Downscale,
}

/// One ROI+scaled tile decode request for
/// [`decode_tiles_region_scaled_into`].
pub struct TileRegionScaledDecodeJob<'i, 'o> {
    /// Compressed JPEG tile bytes.
    pub input: &'i [u8],
    /// Caller-owned output buffer for this tile.
    pub out: &'o mut [u8],
    /// Distance in bytes between output rows.
    pub stride: usize,
    /// Region of interest in source-image coordinates.
    pub roi: Rect,
    /// Downscale factor applied to the region decode.
    pub scale: Downscale,
}

/// Worker configuration for [`decode_tiles_into`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TileBatchOptions {
    /// Worker count. `None` uses [`std::thread::available_parallelism`].
    pub workers: Option<NonZeroUsize>,
}

/// Error returned by [`decode_tiles_into`], annotated with the failing tile
/// index from the caller's input order.
#[derive(Debug)]
pub struct TileBatchError {
    /// Index of the first failing tile in input order.
    pub index: usize,
    /// Decode error reported for that tile.
    pub source: JpegError,
}

impl core::fmt::Display for TileBatchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "tile {} decode failed: {}", self.index, self.source)
    }
}

impl std::error::Error for TileBatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
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
    options: DecodeOptions,
}

impl<'a> JpegView<'a> {
    /// Parse the stream into a borrowed view that can later build a decoder.
    pub fn parse(input: &'a [u8]) -> Result<Self, JpegError> {
        Self::parse_with_options(input, DecodeOptions::default())
    }

    /// Parse the stream with explicit decode options.
    pub fn parse_with_options(input: &'a [u8], options: DecodeOptions) -> Result<Self, JpegError> {
        let header = parse_header(input)?;
        let mut info = header.info();
        options.apply_to_info(&mut info);
        Ok(Self {
            bytes: input,
            header,
            info,
            options,
        })
    }

    /// Header-derived metadata for the parsed stream.
    pub fn info(&self) -> &Info {
        &self.info
    }

    /// Original compressed bytes backing this view.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Return a byte-preserving passthrough candidate for active DICOM/WSI
    /// transfer syntaxes.
    ///
    /// Progressive JPEG is intentionally not exposed here because the active
    /// conversion path should transcode it rather than introduce a retired or
    /// unsupported destination syntax.
    pub fn passthrough_candidate(&self) -> Option<PassthroughCandidate<'a>> {
        jpeg_passthrough_syntax(&self.info).map(|transfer_syntax| {
            PassthroughCandidate::new(
                self.bytes,
                transfer_syntax,
                CompressedPayloadKind::JpegInterchange,
                self.info.to_core_info(),
            )
        })
    }

    /// Build a restart-marker byte-offset index for the first scan.
    ///
    /// Offsets are absolute byte positions in the original JPEG byte slice.
    /// Returns `Ok(None)` when the stream has no non-zero DRI marker.
    pub fn restart_index(&self) -> Result<Option<RestartIndex>, JpegError> {
        restart_index_for_stream(
            self.bytes,
            self.header.sos_offset,
            &self.info,
            self.info.restart_interval,
        )
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
    pub(crate) progressive_plan: Option<PreparedProgressivePlan>,
    pub(crate) cpu_entropy_checkpoints: Mutex<CpuCheckpointCache>,
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
        Self::inspect_with_options(input, DecodeOptions::default())
    }

    /// Parse headers with explicit decode options, without decoding pixels.
    ///
    /// The options are applied to the returned [`Info`] exactly as they would
    /// be for [`Self::new_with_options`].
    pub fn inspect_with_options(
        input: &'a [u8],
        options: DecodeOptions,
    ) -> Result<Info, JpegError> {
        let mut info = parse_info(input)?;
        options.apply_to_info(&mut info);
        Ok(info)
    }

    /// Build a decoder with explicit decode options.
    pub fn new_with_options(input: &'a [u8], options: DecodeOptions) -> Result<Self, JpegError> {
        let view = JpegView::parse_with_options(input, options)?;
        DEFAULT_CONTEXT.with(|ctx| Self::from_view_in_context(view, &mut ctx.borrow_mut()))
    }

    /// Build a decoder ready for `decode_into`. Parses the full header, pre-
    /// builds every referenced Huffman table, and validates that the stream is
    /// one of the SOFs this release implements.
    ///
    /// # Errors
    /// - Any parse error encountered before SOS (see [`Self::inspect`]).
    /// - [`JpegError::NotImplemented`] for SOFs that parse but are not yet
    ///   decodable (Extended12, Progressive12, Lossless — all land in M3).
    /// - [`JpegError::MissingHuffmanTable`] if the scan references a DC/AC
    ///   table slot that was never defined by a DHT segment.
    pub fn new(input: &'a [u8]) -> Result<Self, JpegError> {
        Self::new_with_options(input, DecodeOptions::default())
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
            options,
        } = view;
        let backend = Backend::detect();
        let (info, warnings, plan, progressive_plan) = if info.sof_kind == SofKind::Progressive8 {
            let progressive_plan = Self::build_progressive_plan(&header, &info, ctx)?;
            let plan = Self::build_progressive_placeholder_plan(&header, &info, ctx)?;
            (
                info,
                Arc::<[Warning]>::from(header.warnings.as_slice()),
                plan,
                Some(progressive_plan),
            )
        } else if options == DecodeOptions::default() {
            if let Some(scan_offset) = header.sos_offset {
                let header_prefix = &bytes[..scan_offset];
                let (info, warnings, plan) = ctx.resolve_decode_plan(header_prefix, |ctx| {
                    let plan = Self::build_prepared_plan(&header, &info, ctx)?;
                    Ok((
                        info.clone(),
                        Arc::<[Warning]>::from(header.warnings.as_slice()),
                        plan,
                    ))
                })?;
                (info, warnings, plan, None)
            } else {
                let plan = Self::build_prepared_plan(&header, &info, ctx)?;
                (
                    info,
                    Arc::<[Warning]>::from(header.warnings.as_slice()),
                    plan,
                    None,
                )
            }
        } else {
            let plan = Self::build_prepared_plan(&header, &info, ctx)?;
            (
                info,
                Arc::<[Warning]>::from(header.warnings.as_slice()),
                plan,
                None,
            )
        };
        Ok(Self {
            bytes,
            info,
            warnings,
            backend,
            plan,
            progressive_plan,
            cpu_entropy_checkpoints: Mutex::new(CpuCheckpointCache::default()),
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

    fn build_progressive_plan(
        header: &ParsedHeader,
        info: &Info,
        ctx: &mut DecoderContext,
    ) -> Result<PreparedProgressivePlan, JpegError> {
        if info.sof_kind != SofKind::Progressive8 {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        match info.color_space {
            ColorSpace::Grayscale | ColorSpace::YCbCr | ColorSpace::Rgb => {}
            color_space => return Err(JpegError::UnsupportedColorSpace { color_space }),
        }
        validate_sampling_factors(header, info)?;
        if header.progressive_scans.is_empty() {
            return Err(JpegError::MissingMarker {
                marker: MarkerKind::Sos,
            });
        }

        let max_h = u32::from(header.sampling.max_h);
        let max_v = u32::from(header.sampling.max_v);
        let mcu_cols = info.dimensions.0.div_ceil(8 * max_h);
        let mcu_rows = info.dimensions.1.div_ceil(8 * max_v);
        let mut components = Vec::with_capacity(header.component_ids.len());
        for (output_index, &id) in header.component_ids.iter().enumerate() {
            let (h, v) =
                header
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
                    component: id,
                    table_id: quant_id as u8,
                })?;
            components.push(PreparedProgressiveComponentPlan {
                h,
                v,
                output_index,
                quant: ctx.resolve_quant_table(quant),
                block_cols: mcu_cols * u32::from(h),
                block_rows: mcu_rows * u32::from(v),
                sample_width: info
                    .dimensions
                    .0
                    .saturating_mul(u32::from(h))
                    .div_ceil(max_h),
                sample_height: info
                    .dimensions
                    .1
                    .saturating_mul(u32::from(v))
                    .div_ceil(max_v),
            });
        }

        let mut scans = Vec::with_capacity(header.progressive_scans.len());
        for parsed in &header.progressive_scans {
            let mut scan_components = Vec::with_capacity(parsed.scan.components.len());
            for component in &parsed.scan.components {
                let component_index = find_component_index(&header.component_ids, component.id)
                    .ok_or(JpegError::UnknownScanComponent {
                        offset: parsed.entropy_offset,
                        component: component.id,
                    })?;
                let quant_id = *header.quant_table_ids.get(component_index).ok_or(
                    JpegError::MissingMarker {
                        marker: MarkerKind::Sof,
                    },
                )?;
                let _ = parsed
                    .quant_tables
                    .entries
                    .get(quant_id as usize)
                    .and_then(|q| q.as_ref())
                    .ok_or(JpegError::MissingQuantTable {
                        component: component.id,
                        table_id: quant_id,
                    })?;
                let dc_table = if parsed.scan.ss == 0 {
                    Some(resolve_progressive_huffman(
                        ctx,
                        &parsed.huffman_tables.dc,
                        component.id,
                        0,
                        component.dc_table,
                    )?)
                } else {
                    None
                };
                let ac_table = if parsed.scan.ss > 0 {
                    Some(resolve_progressive_huffman(
                        ctx,
                        &parsed.huffman_tables.ac,
                        component.id,
                        1,
                        component.ac_table,
                    )?)
                } else {
                    None
                };
                scan_components.push(PreparedProgressiveScanComponent {
                    component_index,
                    dc_table,
                    ac_table,
                });
            }
            scans.push(PreparedProgressiveScan {
                components: scan_components,
                ss: parsed.scan.ss,
                se: parsed.scan.se,
                ah: parsed.scan.ah,
                al: parsed.scan.al,
                entropy_offset: parsed.entropy_offset,
                restart_interval: parsed.restart_interval,
            });
        }

        let scratch_bytes =
            compute_progressive_scratch_bytes(&components, info.dimensions.0 as usize)?;
        Ok(PreparedProgressivePlan {
            components,
            scans,
            sampling: info.sampling,
            color_space: info.color_space,
            dimensions: info.dimensions,
            mcu_cols,
            mcu_rows,
            scratch_bytes,
        })
    }

    fn build_progressive_placeholder_plan(
        header: &ParsedHeader,
        info: &Info,
        ctx: &mut DecoderContext,
    ) -> Result<PreparedDecodePlan, JpegError> {
        let empty_raw = RawHuffmanTable {
            bits: [0; 16],
            values: HuffmanValues::default(),
        };
        let empty_huffman = ctx.resolve_huffman_table(&empty_raw)?;
        let mut components = Vec::with_capacity(header.component_ids.len());
        for (output_index, &id) in header.component_ids.iter().enumerate() {
            let (h, v) =
                header
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
                    component: id,
                    table_id: quant_id as u8,
                })?;
            components.push(PreparedComponentPlan {
                h,
                v,
                output_index,
                quant: ctx.resolve_quant_table(quant),
                dc_table: Arc::clone(&empty_huffman),
                ac_table: Arc::clone(&empty_huffman),
            });
        }
        Ok(PreparedDecodePlan {
            components,
            sampling: info.sampling,
            color_space: info.color_space,
            restart_interval: header.restart_interval,
            dimensions: info.dimensions,
            scan_offset: header.sos_offset.ok_or(JpegError::MissingMarker {
                marker: MarkerKind::Sos,
            })?,
            scratch_bytes: compute_decode_scratch_bytes(
                info.dimensions,
                info.sampling,
                DEFAULT_MAX_DECODE_BYTES,
            )?,
        })
    }

    /// The parsed header as a public [`Info`].
    pub fn info(&self) -> &Info {
        &self.info
    }

    /// Return a byte-preserving passthrough candidate for this decoded stream.
    pub fn passthrough_candidate(&self) -> Option<PassthroughCandidate<'a>> {
        jpeg_passthrough_syntax(&self.info).map(|transfer_syntax| {
            PassthroughCandidate::new(
                self.bytes,
                transfer_syntax,
                CompressedPayloadKind::JpegInterchange,
                self.info.to_core_info(),
            )
        })
    }

    /// Build a restart-marker byte-offset index for the first scan.
    ///
    /// Offsets are absolute byte positions in the original JPEG byte slice.
    /// Returns `Ok(None)` when the stream has no non-zero DRI marker.
    pub fn restart_index(&self) -> Result<Option<RestartIndex>, JpegError> {
        restart_index_for_stream(
            self.bytes,
            Some(self.plan.scan_offset),
            &self.info,
            self.plan.restart_interval,
        )
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
        let profile_enabled = jpeg_profile_stages_enabled();
        let total_start = profile_enabled.then(Instant::now);
        let downscale = fmt.downscale();
        let (w, h) = scaled_dimensions(self.info.dimensions, downscale);
        let scratch_bytes = self.decode_scratch_bytes(DEFAULT_MAX_DECODE_BYTES)?;
        let bpp = fmt.bytes_per_pixel();
        validate_buffer(out, stride, w, h, bpp)?;
        let decode_start = profile_enabled.then(Instant::now);
        let result = match fmt {
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
        };
        if let (Some(total_start), Some(decode_start), Ok(outcome)) =
            (total_start, decode_start, &result)
        {
            let source_width_s = self.info.dimensions.0.to_string();
            let source_height_s = self.info.dimensions.1.to_string();
            let output_width_s = w.to_string();
            let output_height_s = h.to_string();
            let stride_s = stride.to_string();
            let bpp_s = bpp.to_string();
            let output_bytes_s = stride.saturating_mul(h as usize).to_string();
            let scratch_bytes_s = scratch_bytes.to_string();
            let warning_count_s = outcome.warnings.len().to_string();
            let decode_us = duration_us_string(decode_start.elapsed());
            let total_us = duration_us_string(total_start.elapsed());
            emit_jpeg_profile_row(
                "decode",
                "cpu",
                &[
                    ("mode", "full"),
                    ("fmt", output_format_profile_name(fmt)),
                    ("downscale", downscale_profile_name(downscale)),
                    ("source_width", source_width_s.as_str()),
                    ("source_height", source_height_s.as_str()),
                    ("output_width", output_width_s.as_str()),
                    ("output_height", output_height_s.as_str()),
                    ("stride", stride_s.as_str()),
                    ("bpp", bpp_s.as_str()),
                    ("scratch_bytes", scratch_bytes_s.as_str()),
                    ("output_bytes", output_bytes_s.as_str()),
                    ("decode_us", decode_us.as_str()),
                    ("total_us", total_us.as_str()),
                    ("warnings", warning_count_s.as_str()),
                ],
            );
        }
        result
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
            let (source_x0, source_width) =
                self.source_window_for_output_rect(downscale, scaled_roi);
            let mut cropped = CroppedWriter::new(adapter, scaled_roi, source_x0, source_width);
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
        let profile_enabled = jpeg_profile_stages_enabled();
        let total_start = profile_enabled.then(Instant::now);
        if !roi.is_within(self.info.dimensions) {
            return Err(JpegError::RectOutOfBounds {
                rect: roi,
                width: self.info.dimensions.0,
                height: self.info.dimensions.1,
            });
        }

        if roi == Rect::full(self.info.dimensions) {
            return self.decode_into_output_format_with_scratch(pool, out, stride, fmt);
        }

        let downscale = fmt.downscale();
        let scaled_roi = scaled_rect_covering(roi, downscale)?;
        let scratch_bytes = self.decode_scratch_bytes(DEFAULT_MAX_DECODE_BYTES)?;
        validate_buffer(
            out,
            stride,
            scaled_roi.w,
            scaled_roi.h,
            fmt.bytes_per_pixel(),
        )?;

        let decode_start = profile_enabled.then(Instant::now);
        let result = match fmt {
            OutputFormat::Rgb8 | OutputFormat::Rgb8Scaled { .. } => {
                if fmt == OutputFormat::Rgb8
                    && downscale == DownscaleFactor::Full
                    && self.plan.matches_fast_tile_shape()
                {
                    let mut writer = Rgb8Writer::new(out, stride, scaled_roi.w);
                    let scan_bytes = &self.bytes[self.plan.scan_offset..];
                    let checkpoint = self.checkpoint_for_mcu(
                        scan_bytes,
                        fast_tile_region_first_decode_mcu(&self.plan, roi, DownscaleFactor::Full),
                    )?;
                    let scan_warnings = decode_scan_fast_tile_rgb_region(
                        &self.plan,
                        self.backend,
                        scan_bytes,
                        pool,
                        &mut writer,
                        roi,
                        checkpoint.as_ref(),
                    )?;
                    Ok(DecodeOutcome {
                        decoded: roi,
                        warnings: merged_warnings(&self.warnings, scan_warnings),
                    })
                } else if matches!(fmt, OutputFormat::Rgb8Scaled { .. })
                    && self.plan.matches_fast_tile_shape()
                {
                    let mut writer = Rgb8Writer::new(out, stride, scaled_roi.w);
                    let scan_bytes = &self.bytes[self.plan.scan_offset..];
                    let checkpoint = self.checkpoint_for_mcu(
                        scan_bytes,
                        fast_tile_region_first_decode_mcu(&self.plan, scaled_roi, downscale),
                    )?;
                    let scan_warnings = decode_scan_fast_tile_rgb_region_scaled(
                        &self.plan,
                        self.backend,
                        scan_bytes,
                        pool,
                        &mut writer,
                        scaled_roi,
                        downscale,
                        checkpoint.as_ref(),
                    )?;
                    Ok(DecodeOutcome {
                        decoded: scaled_roi,
                        warnings: merged_warnings(&self.warnings, scan_warnings),
                    })
                } else {
                    let base = Rgb8Writer::new(out, stride, scaled_roi.w);
                    let (source_x0, source_width) =
                        self.source_window_for_output_rect(downscale, scaled_roi);
                    let mut writer = CroppedWriter::new(base, scaled_roi, source_x0, source_width);
                    self.decode_rgb_with_writer(pool, &mut writer, downscale, roi)
                }
            }
            OutputFormat::Rgba8 { alpha } => {
                let base = Rgba8Writer::new(out, stride, scaled_roi.w, alpha);
                let (source_x0, source_width) =
                    self.source_window_for_output_rect(downscale, scaled_roi);
                let mut writer = CroppedWriter::new(base, scaled_roi, source_x0, source_width);
                self.decode_with_writer(pool, &mut writer, downscale, roi)
            }
            OutputFormat::Gray8 | OutputFormat::Gray8Scaled { .. } => {
                let base = Gray8Writer::new(out, stride, scaled_roi.w);
                let (source_x0, source_width) =
                    self.source_window_for_output_rect(downscale, scaled_roi);
                let mut writer = CroppedWriter::new(base, scaled_roi, source_x0, source_width);
                self.decode_with_writer(pool, &mut writer, downscale, roi)
            }
        };
        if let (Some(total_start), Some(decode_start), Ok(outcome)) =
            (total_start, decode_start, &result)
        {
            let source_width_s = self.info.dimensions.0.to_string();
            let source_height_s = self.info.dimensions.1.to_string();
            let roi_x_s = roi.x.to_string();
            let roi_y_s = roi.y.to_string();
            let roi_w_s = roi.w.to_string();
            let roi_h_s = roi.h.to_string();
            let output_width_s = scaled_roi.w.to_string();
            let output_height_s = scaled_roi.h.to_string();
            let stride_s = stride.to_string();
            let bpp_s = fmt.bytes_per_pixel().to_string();
            let output_bytes_s = stride.saturating_mul(scaled_roi.h as usize).to_string();
            let scratch_bytes_s = scratch_bytes.to_string();
            let warning_count_s = outcome.warnings.len().to_string();
            let decode_us = duration_us_string(decode_start.elapsed());
            let total_us = duration_us_string(total_start.elapsed());
            let mode = if downscale == DownscaleFactor::Full {
                "region"
            } else {
                "region_scaled"
            };
            emit_jpeg_profile_row(
                "decode",
                "cpu",
                &[
                    ("mode", mode),
                    ("fmt", output_format_profile_name(fmt)),
                    ("downscale", downscale_profile_name(downscale)),
                    ("source_width", source_width_s.as_str()),
                    ("source_height", source_height_s.as_str()),
                    ("roi_x", roi_x_s.as_str()),
                    ("roi_y", roi_y_s.as_str()),
                    ("roi_w", roi_w_s.as_str()),
                    ("roi_h", roi_h_s.as_str()),
                    ("output_width", output_width_s.as_str()),
                    ("output_height", output_height_s.as_str()),
                    ("stride", stride_s.as_str()),
                    ("bpp", bpp_s.as_str()),
                    ("scratch_bytes", scratch_bytes_s.as_str()),
                    ("output_bytes", output_bytes_s.as_str()),
                    ("decode_us", decode_us.as_str()),
                    ("total_us", total_us.as_str()),
                    ("warnings", warning_count_s.as_str()),
                ],
            );
        }
        result
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
/// Parallelism is the caller's responsibility for this primitive. For
/// production batch decode, use [`decode_tiles_into`].
///
/// # Example
///
/// ```no_run
/// use signinum_jpeg::{decode_tile_into, PixelFormat, ScratchPool};
///
/// let bytes: &[u8] = todo!("read tile bytes");
/// let mut out = vec![0u8; 256 * 256 * 3];
/// let mut pool = ScratchPool::new();
/// decode_tile_into(bytes, &mut pool, &mut out, 256 * 3, PixelFormat::Rgb8)?;
/// # Ok::<(), signinum_jpeg::JpegError>(())
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
    decode_tile_into_in_context_with_options(
        bytes,
        ctx,
        pool,
        out,
        stride,
        fmt,
        DecodeOptions::default(),
    )
}

/// One-shot parse-plus-decode of an independent JPEG tile into the caller's
/// buffer, reusing both caller-owned [`DecoderContext`] and caller-owned
/// [`ScratchPool`], with explicit JPEG decode options.
pub fn decode_tile_into_in_context_with_options(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    options: DecodeOptions,
) -> Result<DecodeOutcome, JpegError> {
    let dec = Decoder::from_view_in_context(JpegView::parse_with_options(bytes, options)?, ctx)?;
    dec.decode_into_with_scratch(pool, out, stride, fmt)
}

/// Decode independent JPEG tiles into caller-owned output buffers using a
/// scoped CPU worker pool.
///
/// Each worker owns one [`DecoderContext`] and one [`ScratchPool`], so repeated
/// tiles reuse parsed table state and heap scratch within that worker without
/// sharing mutable decoder state across threads. Returned outcomes preserve
/// the caller's input order.
///
/// # Errors
/// Returns [`TileBatchError`] with the first failing tile index in input order.
pub fn decode_tiles_into(
    jobs: &mut [TileDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    decode_tiles_into_with_options(jobs, fmt, DecodeOptions::default(), options)
}

/// Decode independent JPEG tiles into caller-owned output buffers using a
/// scoped CPU worker pool and explicit JPEG decode options.
///
/// Use this variant when container metadata has already resolved ambiguous
/// three-component JPEG data to RGB or YCbCr via [`DecodeOptions`].
///
/// # Errors
/// Returns [`TileBatchError`] with the first failing tile index in input order.
pub fn decode_tiles_into_with_options(
    jobs: &mut [TileDecodeJob<'_, '_>],
    fmt: PixelFormat,
    decode_options: DecodeOptions,
    options: TileBatchOptions,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }

    let job_count = jobs.len();
    let worker_count = tile_batch_worker_count(job_count, options);
    let chunk_size = job_count.div_ceil(worker_count);
    let results = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for (chunk_index, chunk) in jobs.chunks_mut(chunk_size).enumerate() {
            let start_index = chunk_index * chunk_size;
            handles.push(
                scope.spawn(move || decode_tile_job_chunk(start_index, chunk, fmt, decode_options)),
            );
        }

        let mut results = Vec::with_capacity(job_count);
        for handle in handles {
            match handle.join() {
                Ok(chunk_results) => results.extend(chunk_results),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        results
    });

    collect_tile_batch_results(job_count, results)
}

/// Decode independent JPEG tiles at reduced resolution into caller-owned
/// output buffers using a scoped CPU worker pool.
///
/// Each worker owns one [`DecoderContext`] and one [`ScratchPool`], so repeated
/// tiles reuse parsed table state and heap scratch within that worker without
/// sharing mutable decoder state across threads. Returned outcomes preserve
/// the caller's input order.
///
/// # Errors
/// Returns [`TileBatchError`] with the first failing tile index in input order.
pub fn decode_tiles_scaled_into(
    jobs: &mut [TileScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    decode_tiles_scaled_into_with_options(jobs, fmt, DecodeOptions::default(), options)
}

/// Decode independent JPEG tiles at reduced resolution into caller-owned
/// output buffers using a scoped CPU worker pool and explicit JPEG decode
/// options.
///
/// # Errors
/// Returns [`TileBatchError`] with the first failing tile index in input order.
pub fn decode_tiles_scaled_into_with_options(
    jobs: &mut [TileScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    decode_options: DecodeOptions,
    options: TileBatchOptions,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }

    let job_count = jobs.len();
    let worker_count = tile_batch_worker_count(job_count, options);
    let chunk_size = job_count.div_ceil(worker_count);
    let results = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for (chunk_index, chunk) in jobs.chunks_mut(chunk_size).enumerate() {
            let start_index = chunk_index * chunk_size;
            handles.push(scope.spawn(move || {
                decode_tile_scaled_job_chunk(start_index, chunk, fmt, decode_options)
            }));
        }

        let mut results = Vec::with_capacity(job_count);
        for handle in handles {
            match handle.join() {
                Ok(chunk_results) => results.extend(chunk_results),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        results
    });

    collect_tile_batch_results(job_count, results)
}

/// Decode independent JPEG tile regions at reduced resolution into
/// caller-owned output buffers using a scoped CPU worker pool.
///
/// Each worker owns one [`DecoderContext`] and one [`ScratchPool`], so repeated
/// tiles reuse parsed table state and heap scratch within that worker without
/// sharing mutable decoder state across threads. Returned outcomes preserve
/// the caller's input order.
///
/// # Errors
/// Returns [`TileBatchError`] with the first failing tile index in input order.
pub fn decode_tiles_region_scaled_into(
    jobs: &mut [TileRegionScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    decode_tiles_region_scaled_into_with_options(jobs, fmt, DecodeOptions::default(), options)
}

/// Decode independent JPEG tile regions at reduced resolution into
/// caller-owned output buffers using a scoped CPU worker pool and explicit JPEG
/// decode options.
///
/// # Errors
/// Returns [`TileBatchError`] with the first failing tile index in input order.
pub fn decode_tiles_region_scaled_into_with_options(
    jobs: &mut [TileRegionScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    decode_options: DecodeOptions,
    options: TileBatchOptions,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }

    let job_count = jobs.len();
    let worker_count = tile_batch_worker_count(job_count, options);
    let chunk_size = job_count.div_ceil(worker_count);
    let results = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for (chunk_index, chunk) in jobs.chunks_mut(chunk_size).enumerate() {
            let start_index = chunk_index * chunk_size;
            handles.push(scope.spawn(move || {
                decode_tile_region_scaled_job_chunk(start_index, chunk, fmt, decode_options)
            }));
        }

        let mut results = Vec::with_capacity(job_count);
        for handle in handles {
            match handle.join() {
                Ok(chunk_results) => results.extend(chunk_results),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        results
    });

    collect_tile_batch_results(job_count, results)
}

fn collect_tile_batch_results(
    job_count: usize,
    results: Vec<(usize, Result<DecodeOutcome, JpegError>)>,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    let mut outcomes = Vec::with_capacity(job_count);
    outcomes.resize_with(job_count, || None);
    let mut first_error = None::<TileBatchError>;
    for (index, result) in results {
        match result {
            Ok(outcome) => outcomes[index] = Some(outcome),
            Err(source) => {
                if first_error
                    .as_ref()
                    .is_none_or(|current| index < current.index)
                {
                    first_error = Some(TileBatchError { index, source });
                }
            }
        }
    }

    if let Some(err) = first_error {
        return Err(err);
    }

    Ok(outcomes
        .into_iter()
        .map(|outcome| outcome.expect("successful batch stores one outcome per tile"))
        .collect())
}

fn tile_batch_worker_count(batch_size: usize, options: TileBatchOptions) -> usize {
    if batch_size <= 1 {
        return 1;
    }
    let workers = options.workers.map_or_else(
        || std::thread::available_parallelism().map_or(1, NonZeroUsize::get),
        NonZeroUsize::get,
    );
    workers.max(1).min(batch_size)
}

fn decode_tile_job_chunk(
    start_index: usize,
    jobs: &mut [TileDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: DecodeOptions,
) -> Vec<(usize, Result<DecodeOutcome, JpegError>)> {
    let mut ctx = DecoderContext::new();
    let mut pool = ScratchPool::new();
    let mut results = Vec::with_capacity(jobs.len());
    for (local_index, job) in jobs.iter_mut().enumerate() {
        let outcome = decode_tile_into_in_context_with_options(
            job.input, &mut ctx, &mut pool, job.out, job.stride, fmt, options,
        );
        results.push((start_index + local_index, outcome));
    }
    results
}

fn decode_tile_scaled_job_chunk(
    start_index: usize,
    jobs: &mut [TileScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: DecodeOptions,
) -> Vec<(usize, Result<DecodeOutcome, JpegError>)> {
    let mut ctx = DecoderContext::new();
    let mut pool = ScratchPool::new();
    let mut results = Vec::with_capacity(jobs.len());
    for (local_index, job) in jobs.iter_mut().enumerate() {
        let outcome = decode_tile_scaled_into_in_context_with_options(
            job.input, &mut ctx, &mut pool, job.out, job.stride, fmt, job.scale, options,
        );
        results.push((start_index + local_index, outcome));
    }
    results
}

fn decode_tile_region_scaled_job_chunk(
    start_index: usize,
    jobs: &mut [TileRegionScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: DecodeOptions,
) -> Vec<(usize, Result<DecodeOutcome, JpegError>)> {
    let mut ctx = DecoderContext::new();
    let mut pool = ScratchPool::new();
    let mut results = Vec::with_capacity(jobs.len());
    for (local_index, job) in jobs.iter_mut().enumerate() {
        let outcome = decode_tile_region_scaled_into_in_context_with_options(
            job.input, &mut ctx, &mut pool, job.out, job.stride, fmt, job.roi, job.scale, options,
        );
        results.push((start_index + local_index, outcome));
    }
    results
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
    decode_tile_region_into_in_context_with_options(
        bytes,
        ctx,
        pool,
        out,
        stride,
        fmt,
        roi,
        DecodeOptions::default(),
    )
}

/// One-shot parse-plus-region-decode of an independent JPEG tile into the
/// caller's buffer, reusing caller-owned state and explicit JPEG decode
/// options.
#[allow(clippy::too_many_arguments)]
pub fn decode_tile_region_into_in_context_with_options(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    roi: Rect,
    options: DecodeOptions,
) -> Result<DecodeOutcome, JpegError> {
    let dec = Decoder::from_view_in_context(JpegView::parse_with_options(bytes, options)?, ctx)?;
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
    decode_tile_scaled_into_in_context_with_options(
        bytes,
        ctx,
        pool,
        out,
        stride,
        fmt,
        scale,
        DecodeOptions::default(),
    )
}

/// One-shot parse-plus-scaled-decode of an independent JPEG tile into the
/// caller's buffer, reusing caller-owned state and explicit JPEG decode
/// options.
#[allow(clippy::too_many_arguments)]
pub fn decode_tile_scaled_into_in_context_with_options(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    scale: Downscale,
    options: DecodeOptions,
) -> Result<DecodeOutcome, JpegError> {
    let dec = Decoder::from_view_in_context(JpegView::parse_with_options(bytes, options)?, ctx)?;
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
    decode_tile_region_scaled_into_in_context_with_options(
        bytes,
        ctx,
        pool,
        out,
        stride,
        fmt,
        roi,
        scale,
        DecodeOptions::default(),
    )
}

/// One-shot parse-plus-region-scaled-decode of an independent JPEG tile into
/// the caller's buffer, reusing caller-owned state and explicit JPEG decode
/// options.
#[allow(clippy::too_many_arguments)]
pub fn decode_tile_region_scaled_into_in_context_with_options(
    bytes: &[u8],
    ctx: &mut DecoderContext,
    pool: &mut ScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
    options: DecodeOptions,
) -> Result<DecodeOutcome, JpegError> {
    let dec = Decoder::from_view_in_context(JpegView::parse_with_options(bytes, options)?, ctx)?;
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
        let scratch_bytes = self
            .progressive_plan
            .as_ref()
            .map_or(self.plan.scratch_bytes, |plan| plan.scratch_bytes);
        if scratch_bytes > cap {
            return Err(JpegError::MemoryCapExceeded {
                requested: scratch_bytes,
                cap,
            });
        }
        Ok(scratch_bytes)
    }

    fn checkpoint_for_mcu(
        &self,
        scan_bytes: &[u8],
        target_mcu: u32,
    ) -> Result<Option<DeviceCheckpoint>, JpegError> {
        if self.plan.restart_interval.is_some() || target_mcu < CPU_ROI_CHECKPOINT_MIN_TARGET_MCUS {
            return Ok(None);
        }

        let mut cache = self
            .cpu_entropy_checkpoints
            .lock()
            .expect("CPU entropy checkpoint cache mutex poisoned");
        checkpoint_before_mcu(
            &self.plan,
            scan_bytes,
            CPU_ROI_CHECKPOINT_CADENCE_MCUS,
            target_mcu,
            &mut cache,
        )
    }

    fn source_window_for_output_rect(
        &self,
        downscale: DownscaleFactor,
        output_rect: Rect,
    ) -> (u32, u32) {
        let layout = stripe_region_layout(&self.plan, downscale, output_rect);
        (layout.source_x0, layout.source_width)
    }

    fn decode_with_writer<W: OutputWriter>(
        &self,
        pool: &mut ScratchPool,
        writer: &mut W,
        downscale: DownscaleFactor,
        decoded: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        let _ = self.decode_scratch_bytes(DEFAULT_MAX_DECODE_BYTES)?;
        let profile_enabled = jpeg_profile_stages_enabled();
        if let Some(plan) = &self.progressive_plan {
            if downscale != DownscaleFactor::Full || decoded != Rect::full(self.info.dimensions) {
                return Err(JpegError::NotImplemented {
                    sof: self.info.sof_kind,
                });
            }
            let scan_start = profile_enabled.then(Instant::now);
            let scan_warnings = decode_progressive(plan, self.backend, self.bytes, writer)?;
            if let Some(start) = scan_start {
                emit_decode_scan_profile(
                    "progressive",
                    self.info.dimensions,
                    decoded,
                    downscale,
                    start.elapsed(),
                );
            }
            return Ok(DecodeOutcome {
                decoded,
                warnings: merged_warnings(&self.warnings, scan_warnings),
            });
        }
        let output_rect = scaled_rect_covering(decoded, downscale)?;
        let scan_bytes = &self.bytes[self.plan.scan_offset..];
        let scan_start = profile_enabled.then(Instant::now);
        let scan_warnings = decode_scan_baseline(
            &self.plan,
            self.backend,
            scan_bytes,
            pool,
            writer,
            downscale,
            output_rect,
        )?;
        if let Some(start) = scan_start {
            emit_decode_scan_profile(
                "baseline",
                self.info.dimensions,
                decoded,
                downscale,
                start.elapsed(),
            );
        }
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
        let profile_enabled = jpeg_profile_stages_enabled();
        if let Some(plan) = &self.progressive_plan {
            if downscale != DownscaleFactor::Full || decoded != Rect::full(self.info.dimensions) {
                return Err(JpegError::NotImplemented {
                    sof: self.info.sof_kind,
                });
            }
            let scan_start = profile_enabled.then(Instant::now);
            let scan_warnings = decode_progressive(plan, self.backend, self.bytes, writer)?;
            if let Some(start) = scan_start {
                emit_decode_scan_profile(
                    "progressive_rgb",
                    self.info.dimensions,
                    decoded,
                    downscale,
                    start.elapsed(),
                );
            }
            return Ok(DecodeOutcome {
                decoded,
                warnings: merged_warnings(&self.warnings, scan_warnings),
            });
        }
        let output_rect = scaled_rect_covering(decoded, downscale)?;
        let scan_bytes = &self.bytes[self.plan.scan_offset..];
        let scan_start = profile_enabled.then(Instant::now);
        let (scan_path, scan_warnings) =
            if downscale == DownscaleFactor::Full && self.plan.matches_fast_tile_shape() {
                (
                    "fast420_rgb",
                    decode_scan_fast_tile_rgb(&self.plan, self.backend, scan_bytes, pool, writer)?,
                )
            } else if downscale == DownscaleFactor::Full
                && decoded == Rect::full(self.info.dimensions)
                && self.plan.matches_fast_rgb444_shape()
            {
                (
                    "fast444_rgb",
                    decode_scan_fast_rgb_444(&self.plan, self.backend, scan_bytes, pool, writer)?,
                )
            } else {
                (
                    "baseline_rgb",
                    decode_scan_baseline_rgb(
                        &self.plan,
                        self.backend,
                        scan_bytes,
                        pool,
                        writer,
                        downscale,
                        output_rect,
                    )?,
                )
            };
        if let Some(start) = scan_start {
            emit_decode_scan_profile(
                scan_path,
                self.info.dimensions,
                decoded,
                downscale,
                start.elapsed(),
            );
        }
        Ok(DecodeOutcome {
            decoded,
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }
}

fn restart_index_for_stream(
    bytes: &[u8],
    scan_data_offset: Option<usize>,
    info: &Info,
    restart_interval: Option<u16>,
) -> Result<Option<RestartIndex>, JpegError> {
    let Some(interval_mcus) = restart_interval
        .filter(|&interval| interval > 0)
        .map(u32::from)
    else {
        return Ok(None);
    };
    let scan_data_offset = scan_data_offset.ok_or(JpegError::MissingMarker {
        marker: MarkerKind::Sos,
    })?;
    if !matches!(info.sof_kind, SofKind::Baseline8 | SofKind::Extended8) || info.scan_count != 1 {
        return Err(JpegError::NotImplemented { sof: info.sof_kind });
    }
    let total_mcus = info.mcu_geometry.count;
    let expected_restarts = total_mcus.saturating_sub(1) / interval_mcus;
    let mut segments = Vec::new();
    segments.push(RestartSegment {
        start_mcu: 0,
        entropy_offset: scan_data_offset,
        marker_offset: None,
        marker: None,
    });

    let mut found_restarts = 0u32;
    let mut expected_rst = 0xd0u8;
    let mut pos = scan_data_offset;
    while pos < bytes.len() {
        if bytes[pos] != 0xff {
            pos += 1;
            continue;
        }

        let mut marker_code_pos = pos + 1;
        while marker_code_pos < bytes.len() && bytes[marker_code_pos] == 0xff {
            marker_code_pos += 1;
        }
        if marker_code_pos >= bytes.len() {
            return Err(JpegError::Truncated {
                offset: pos,
                expected: 1,
            });
        }

        let marker = bytes[marker_code_pos];
        let marker_offset = marker_code_pos - 1;
        match marker {
            0x00 => pos = marker_code_pos + 1,
            0xd0..=0xd7 => {
                if found_restarts >= expected_restarts {
                    return Err(JpegError::UnexpectedMarker {
                        offset: marker_offset,
                        expected: MarkerKind::Eoi,
                        found: marker,
                    });
                }
                if marker != expected_rst {
                    return Err(JpegError::RestartMismatch {
                        offset: marker_offset,
                        expected: expected_rst & 0x07,
                        found: marker,
                    });
                }
                found_restarts += 1;
                segments.push(RestartSegment {
                    start_mcu: found_restarts.saturating_mul(interval_mcus),
                    entropy_offset: marker_code_pos + 1,
                    marker_offset: Some(marker_offset),
                    marker: Some(marker),
                });
                expected_rst = if expected_rst == 0xd7 {
                    0xd0
                } else {
                    expected_rst + 1
                };
                pos = marker_code_pos + 1;
            }
            0xd9 => {
                if found_restarts != expected_restarts {
                    return Err(JpegError::UnexpectedEoi {
                        mcu_at: found_restarts
                            .saturating_add(1)
                            .saturating_mul(interval_mcus),
                        mcu_total: total_mcus,
                    });
                }
                return Ok(Some(RestartIndex {
                    scan_data_offset,
                    interval_mcus,
                    segments,
                }));
            }
            found => {
                return Err(JpegError::UnexpectedMarker {
                    offset: marker_offset,
                    expected: MarkerKind::Eoi,
                    found,
                });
            }
        }
    }

    Err(JpegError::MissingMarker {
        marker: MarkerKind::Eoi,
    })
}

fn output_format_profile_name(fmt: OutputFormat) -> &'static str {
    match fmt {
        OutputFormat::Rgb8 | OutputFormat::Rgb8Scaled { .. } => "Rgb8",
        OutputFormat::Rgba8 { .. } => "Rgba8",
        OutputFormat::Gray8 | OutputFormat::Gray8Scaled { .. } => "Gray8",
    }
}

fn downscale_profile_name(downscale: DownscaleFactor) -> &'static str {
    match downscale {
        DownscaleFactor::Full => "full",
        DownscaleFactor::Half => "half",
        DownscaleFactor::Quarter => "quarter",
        DownscaleFactor::Eighth => "eighth",
    }
}

fn emit_decode_scan_profile(
    scan_path: &str,
    dimensions: (u32, u32),
    decoded: Rect,
    downscale: DownscaleFactor,
    elapsed: Duration,
) {
    let source_width_s = dimensions.0.to_string();
    let source_height_s = dimensions.1.to_string();
    let decoded_x_s = decoded.x.to_string();
    let decoded_y_s = decoded.y.to_string();
    let decoded_w_s = decoded.w.to_string();
    let decoded_h_s = decoded.h.to_string();
    let scan_us = duration_us_string(elapsed);
    emit_jpeg_profile_row(
        "decode_scan",
        "cpu",
        &[
            ("scan_path", scan_path),
            ("downscale", downscale_profile_name(downscale)),
            ("source_width", source_width_s.as_str()),
            ("source_height", source_height_s.as_str()),
            ("decoded_x", decoded_x_s.as_str()),
            ("decoded_y", decoded_y_s.as_str()),
            ("decoded_w", decoded_w_s.as_str()),
            ("decoded_h", decoded_h_s.as_str()),
            ("scan_us", scan_us.as_str()),
        ],
    );
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

fn jpeg_passthrough_syntax(info: &Info) -> Option<CompressedTransferSyntax> {
    match info.sof_kind {
        SofKind::Baseline8 if info.bit_depth == 8 => Some(CompressedTransferSyntax::JpegBaseline8),
        SofKind::Extended8 | SofKind::Extended12 => {
            Some(CompressedTransferSyntax::JpegExtendedSequential)
        }
        SofKind::Baseline8 | SofKind::Progressive8 | SofKind::Progressive12 | SofKind::Lossless => {
            None
        }
    }
}

fn core_rect(rect: Rect) -> signinum_core::Rect {
    signinum_core::Rect {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: rect.h,
    }
}

fn jpeg_rect(rect: signinum_core::Rect) -> Rect {
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

    fn inspect(input: &'a [u8]) -> Result<signinum_core::Info, Self::Error> {
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
        roi: signinum_core::Rect,
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

    fn decode_region_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: signinum_core::Rect,
        scale: Downscale,
    ) -> Result<CoreDecodeOutcome<Self::Warning>, Self::Error> {
        Decoder::decode_region_scaled_into_with_scratch(
            self,
            pool,
            out,
            stride,
            fmt,
            jpeg_rect(roi),
            scale,
        )
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
        roi: signinum_core::Rect,
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

    fn decode_tile_region_scaled(
        ctx: &mut CoreDecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: signinum_core::Rect,
        scale: Downscale,
    ) -> Result<CoreDecodeOutcome<Self::Warning>, Self::Error> {
        let dec = Decoder::from_view_in_context(JpegView::parse(input)?, ctx.codec_mut())?;
        dec.decode_region_scaled_into_with_scratch(pool, out, stride, fmt, jpeg_rect(roi), scale)
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
    source_x0: u32,
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
    fn new(inner: W, rect: Rect, source_x0: u32, source_width: u32) -> Self {
        let row_len = source_width as usize * 3;
        Self {
            inner,
            rect,
            source_x0,
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
        let x0 = self
            .rect
            .x
            .checked_sub(self.source_x0)
            .expect("crop window must cover requested rect") as usize;
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
        let x0 = self
            .rect
            .x
            .checked_sub(self.source_x0)
            .expect("crop window must cover requested rect") as usize;
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
        let x0 = self
            .rect
            .x
            .checked_sub(self.source_x0)
            .expect("crop window must cover requested rect") as usize;
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
        let x0 = self
            .rect
            .x
            .checked_sub(self.source_x0)
            .expect("crop window must cover requested rect") as usize
            * 3;
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

fn validate_sampling_factors(header: &ParsedHeader, info: &Info) -> Result<(), JpegError> {
    if let Some((h, v)) = header.sampling.component(0) {
        if h != header.sampling.max_h || v != header.sampling.max_v {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
    }
    for (h, v) in header.sampling.iter() {
        if h == 0 || v == 0 || h > 4 || v > 4 {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        if !header.sampling.max_h.is_multiple_of(h) || !header.sampling.max_v.is_multiple_of(v) {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
    }
    Ok(())
}

fn resolve_progressive_huffman(
    ctx: &mut DecoderContext,
    tables: &[Option<RawHuffmanTable>; 4],
    component: u8,
    class: u8,
    id: u8,
) -> Result<Arc<HuffmanTable>, JpegError> {
    let raw = tables
        .get(id as usize)
        .and_then(|table| table.as_ref())
        .ok_or(JpegError::MissingHuffmanTable {
            component,
            class,
            id,
        })?;
    ctx.resolve_huffman_table(raw)
}

fn compute_progressive_scratch_bytes(
    components: &[PreparedProgressiveComponentPlan],
    output_width: usize,
) -> Result<usize, JpegError> {
    let cap = DEFAULT_MAX_DECODE_BYTES;
    let mut total = 0usize;
    for component in components {
        let blocks = checked_usize_product(
            &[component.block_cols as usize, component.block_rows as usize],
            cap,
        )?;
        let coeffs = checked_usize_product(&[blocks, 64, core::mem::size_of::<i32>()], cap)?;
        total = total
            .checked_add(coeffs)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;

        let plane = checked_usize_product(
            &[
                component.block_cols as usize,
                component.block_rows as usize,
                64,
            ],
            cap,
        )?;
        total = total
            .checked_add(plane)
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
    }
    total =
        total
            .checked_add(output_width.saturating_mul(3))
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
    use crate::output::OutputWriter;
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

    fn dc_only_420_jpeg(width: u16, height: u16) -> Vec<u8> {
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
            (height >> 8) as u8,
            height as u8,
            (width >> 8) as u8,
            width as u8,
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
            0xFF, 0xC4, 0x00, 20, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        v.extend_from_slice(&[
            0xFF, 0xC4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        v.extend_from_slice(&[0xFF, 0xDA, 0x00, 12, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0]);

        let mcus_per_row = u32::from(width).div_ceil(16);
        let mcu_rows = u32::from(height).div_ceil(16);
        let entropy_bits = mcus_per_row * mcu_rows * 12;
        let entropy_bytes = (entropy_bits as usize).div_ceil(8) + 8;
        v.extend(core::iter::repeat_n(0u8, entropy_bytes));
        v.extend_from_slice(&[0xFF, 0xD9]);
        v
    }

    #[test]
    fn decoder_new_succeeds_on_baseline_stream() {
        let bytes = minimal_baseline_jpeg();
        let dec = Decoder::new(&bytes).unwrap();
        assert_eq!(dec.info().dimensions, (16, 16));
    }

    #[test]
    fn decoder_new_rejects_extended12_with_not_implemented() {
        let mut bytes = minimal_baseline_jpeg();
        let p = bytes.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        bytes[p + 1] = 0xC1;
        bytes[p + 4] = 12;
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

    #[test]
    fn large_fast_420_region_decode_populates_cpu_entropy_checkpoints() {
        let bytes = dc_only_420_jpeg(1024, 2048);
        let dec = Decoder::new(&bytes).expect("decoder");
        assert!(dec.plan.matches_fast_tile_shape());

        let roi = Rect {
            x: 64,
            y: 1536,
            w: 64,
            h: 64,
        };
        let mut out = vec![0u8; roi.w as usize * roi.h as usize * 3];
        let mut pool = ScratchPool::new();
        dec.decode_region_into_with_scratch(
            &mut pool,
            &mut out,
            roi.w as usize * 3,
            PixelFormat::Rgb8,
            roi,
        )
        .expect("deep ROI decode");

        let cache = dec
            .cpu_entropy_checkpoints
            .lock()
            .expect("checkpoint cache mutex");
        assert!(cache
            .checkpoints
            .iter()
            .any(|checkpoint| checkpoint.mcu_index >= CPU_ROI_CHECKPOINT_MIN_TARGET_MCUS));
    }

    #[derive(Default)]
    struct GrayRows {
        rows: Vec<(u32, Vec<u8>)>,
    }

    impl OutputWriter for GrayRows {
        fn write_rgb_row(
            &mut self,
            _y: u32,
            _r_row: &[u8],
            _g_row: &[u8],
            _b_row: &[u8],
        ) -> Result<(), JpegError> {
            unreachable!("gray test writer should not receive rgb rows");
        }

        fn write_ycbcr_row(
            &mut self,
            _y: u32,
            _y_row: &[u8],
            _cb_row: &[u8],
            _cr_row: &[u8],
        ) -> Result<(), JpegError> {
            unreachable!("gray test writer should not receive ycbcr rows");
        }

        fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), JpegError> {
            self.rows.push((y, gray_row.to_vec()));
            Ok(())
        }
    }

    #[test]
    fn cropped_writer_honors_source_window_origin() {
        let inner = GrayRows::default();
        let rect = Rect {
            x: 6,
            y: 1,
            w: 2,
            h: 1,
        };
        let mut writer = CroppedWriter::new(inner, rect, 4, 4);

        writer
            .write_gray_row(1, &[10, 20, 30, 40])
            .expect("crop write must succeed");

        assert_eq!(writer.inner.rows, vec![(0, vec![30, 40])]);
    }
}
