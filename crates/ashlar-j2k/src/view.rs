// SPDX-License-Identifier: Apache-2.0

use crate::{
    backend::{
        image as backend_image, inspect_info, inspect_info_from_image, DecodeSettings, Image,
    },
    context::J2kContext,
    decode::{
        decode_image_into_with_native_context, decode_image_region_into_with_native_context,
        decode_region_scaled_from_info, decode_scaled_from_info, validate_buffer, validate_region,
        validate_supported_format, J2kDecodeOutcome,
    },
    parse::parse_info,
    scratch::J2kScratchPool,
    J2kError,
};
use alloc::vec::Vec;
use ashlar_core::{
    DecodeRowsError, DecoderContext, Downscale, ImageCodec, ImageDecode, ImageDecodeRows, Info,
    PixelFormat, Rect, RowSink, TileBatchDecode,
};
use core::convert::Infallible;

/// Borrowed parse result for a JP2 or raw JPEG 2000 / HTJ2K codestream.
///
/// Use this when a caller wants to inspect metadata once and build a decoder
/// later without copying compressed tile bytes.
pub struct J2kView<'a> {
    bytes: &'a [u8],
    info: Info,
    image: Option<Image<'a>>,
}

impl<'a> J2kView<'a> {
    /// Parse container/codestream metadata into a borrowed view.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the input is not a supported JP2/J2C/HTJ2K
    /// stream or when backend inspection rejects the codestream.
    pub fn parse(input: &'a [u8]) -> Result<Self, J2kError> {
        let info = match parse_info(input) {
            Ok(info) => info,
            Err(error) if should_retry_with_backend(&error) => inspect_info(input)?,
            Err(error) => return Err(error),
        };
        let image = Some(backend_image(input, DecodeSettings::default())?);
        Ok(Self {
            bytes: input,
            info,
            image,
        })
    }

    /// Header-derived image metadata.
    pub fn info(&self) -> &Info {
        &self.info
    }

    /// Original compressed bytes backing this view.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

/// JPEG 2000 / HTJ2K decoder with WSI-shaped full-frame, ROI, and scaled
/// output methods.
///
/// The decoder borrows compressed tile bytes and owns reusable native decode
/// context so repeated operations can avoid reparsing backend state.
pub struct J2kDecoder<'a> {
    bytes: &'a [u8],
    info: Info,
    image: Option<Image<'a>>,
    native_context: ashlar_j2k_native::DecoderContext<'a>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
/// Marker type used by generic tile-batch decode traits.
pub struct J2kCodec;

impl<'a> J2kDecoder<'a> {
    /// Inspect JP2/J2C/HTJ2K metadata without decoding pixels.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the input cannot be parsed or inspected as a
    /// supported JPEG 2000 / HTJ2K image.
    pub fn inspect(input: &'a [u8]) -> Result<Info, J2kError> {
        match parse_info(input) {
            Ok(info) => Ok(info),
            Err(error) if should_retry_with_backend(&error) => inspect_info(input),
            Err(error) => Err(error),
        }
    }

    /// Create a decoder from compressed bytes.
    ///
    /// # Errors
    /// Returns [`J2kError`] for unsupported or malformed input.
    pub fn new(input: &'a [u8]) -> Result<Self, J2kError> {
        Self::from_view(J2kView::parse(input)?)
    }

    /// Create a decoder from a previously parsed [`J2kView`].
    ///
    /// # Errors
    /// Returns [`J2kError`] if the parsed view cannot be promoted to a decoder.
    pub fn from_view(view: J2kView<'a>) -> Result<Self, J2kError> {
        Ok(Self {
            bytes: view.bytes,
            info: view.info,
            image: view.image,
            native_context: ashlar_j2k_native::DecoderContext::default(),
        })
    }

    /// Header-derived image metadata.
    pub fn info(&self) -> &Info {
        &self.info
    }

    /// Original compressed bytes backing this decoder.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Decode the full image into `out` using `stride` bytes per output row.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the format is unsupported, the output buffer
    /// is too small, or the codestream fails during decode.
    pub fn decode_into(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        self.decode_into_cached(out, stride, fmt)
    }

    /// Decode the full image with caller-owned scratch.
    ///
    /// The current native full-frame path writes directly into the caller's
    /// output buffer; the pool is accepted to satisfy the shared codec trait
    /// and is used by reduced-resolution and row-bounded paths.
    ///
    /// # Errors
    /// Same as [`Self::decode_into`].
    pub fn decode_into_with_scratch(
        &mut self,
        _pool: &mut J2kScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        self.decode_into_cached(out, stride, fmt)
    }

    fn decode_into_cached(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        validate_supported_format(fmt)?;
        validate_buffer(self.info.dimensions, out.len(), stride, fmt)?;
        self.ensure_image()?;
        let (Some(image), native_context) = (self.image.as_ref(), &mut self.native_context) else {
            return Err(J2kError::Backend(
                "internal image cache missing".to_string(),
            ));
        };
        decode_image_into_with_native_context(image, native_context, out, stride, fmt)?;
        Ok(ashlar_core::DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings: Vec::new(),
        })
    }

    /// Decode a source-coordinate region into `out`.
    ///
    /// `roi` is expressed in full-resolution source pixels. The output buffer
    /// must hold `roi.w * roi.h * fmt.bytes_per_pixel()` bytes with the
    /// provided row stride.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the region is out of bounds, the output buffer
    /// is too small, the format is unsupported, or decode fails.
    pub fn decode_region_into(
        &mut self,
        _pool: &mut J2kScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        self.decode_region_into_cached(out, stride, fmt, roi)
    }

    fn decode_region_into_cached(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        validate_supported_format(fmt)?;
        validate_region(roi, self.info.dimensions)?;
        validate_buffer((roi.w, roi.h), out.len(), stride, fmt)?;
        self.ensure_image()?;
        let (Some(image), native_context) = (self.image.as_ref(), &mut self.native_context) else {
            return Err(J2kError::Backend(
                "internal image cache missing".to_string(),
            ));
        };
        decode_image_region_into_with_native_context(image, native_context, out, stride, fmt, roi)?;
        Ok(ashlar_core::DecodeOutcome {
            decoded: roi,
            warnings: Vec::new(),
        })
    }

    /// Decode the full image at a reduced resolution.
    ///
    /// `scale` uses the shared [`Downscale`] contract; `Downscale::None`
    /// delegates to full-resolution decode.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the format or scale request is unsupported,
    /// the output buffer is too small, or decode fails.
    pub fn decode_scaled_into(
        &mut self,
        pool: &mut J2kScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        if scale == Downscale::None {
            return self.decode_into_with_scratch(pool, out, stride, fmt);
        }
        decode_scaled_from_info(
            self.bytes,
            self.info.dimensions,
            pool,
            out,
            stride,
            fmt,
            scale,
        )
    }

    pub fn decode_region_scaled_into(
        &mut self,
        pool: &mut J2kScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        if scale == Downscale::None {
            return self.decode_region_into(pool, out, stride, fmt, roi);
        }
        decode_region_scaled_from_info(
            self.bytes,
            self.info.dimensions,
            pool,
            out,
            stride,
            fmt,
            roi,
            scale,
        )
    }

    fn ensure_image(&mut self) -> Result<(), J2kError> {
        if self.image.is_none() {
            self.image = Some(backend_image(self.bytes, DecodeSettings::default())?);
            if self.info.tile_layout.is_none() {
                self.info = inspect_info_from_image(self.cached_image()?);
            }
        }
        Ok(())
    }

    fn cached_image(&self) -> Result<&Image<'a>, J2kError> {
        self.image
            .as_ref()
            .ok_or_else(|| J2kError::Backend("internal image cache missing".to_string()))
    }
}

impl ImageCodec for J2kDecoder<'_> {
    type Error = J2kError;
    type Warning = Infallible;
    type Pool = J2kScratchPool;
}

impl<'a> ImageDecode<'a> for J2kDecoder<'a> {
    type View = J2kView<'a>;

    fn inspect(input: &'a [u8]) -> Result<Info, Self::Error> {
        Self::inspect(input)
    }

    fn parse(input: &'a [u8]) -> Result<Self::View, Self::Error> {
        J2kView::parse(input)
    }

    fn from_view(view: Self::View) -> Result<Self, Self::Error> {
        Self::from_view(view)
    }

    fn decode_into(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<ashlar_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_into(self, out, stride, fmt)
    }

    fn decode_into_with_scratch(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<ashlar_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_into_with_scratch(self, pool, out, stride, fmt)
    }

    fn decode_region_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<ashlar_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_region_into(self, pool, out, stride, fmt, roi)
    }

    fn decode_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<ashlar_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_scaled_into(self, pool, out, stride, fmt, scale)
    }

    fn decode_region_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<ashlar_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_region_scaled_into(self, pool, out, stride, fmt, roi, scale)
    }
}

impl<'a> ImageDecodeRows<'a, u8> for J2kDecoder<'a> {
    fn decode_rows<R: RowSink<u8>>(
        &mut self,
        sink: &mut R,
    ) -> Result<ashlar_core::DecodeOutcome<Self::Warning>, DecodeRowsError<Self::Error, R::Error>>
    {
        let fmt = row_format_u8(self.info()).map_err(DecodeRowsError::Decode)?;
        let row_bytes = row_bytes_for(self.info(), fmt).map_err(DecodeRowsError::Decode)?;
        let mut pool = J2kScratchPool::new();
        let row = pool.packed_bytes(row_bytes);
        for y in 0..self.info.dimensions.1 {
            self.decode_region_into_cached(
                row,
                row_bytes,
                fmt,
                Rect {
                    x: 0,
                    y,
                    w: self.info.dimensions.0,
                    h: 1,
                },
            )
            .map_err(DecodeRowsError::Decode)?;
            sink.write_row(y, row).map_err(DecodeRowsError::Sink)?;
        }
        Ok(ashlar_core::DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings: Vec::new(),
        })
    }
}

impl<'a> ImageDecodeRows<'a, u16> for J2kDecoder<'a> {
    fn decode_rows<R: RowSink<u16>>(
        &mut self,
        sink: &mut R,
    ) -> Result<ashlar_core::DecodeOutcome<Self::Warning>, DecodeRowsError<Self::Error, R::Error>>
    {
        let fmt = row_format_u16(self.info()).map_err(DecodeRowsError::Decode)?;
        let row_bytes = row_bytes_for(self.info(), fmt).map_err(DecodeRowsError::Decode)?;
        let samples_per_row = row_samples_for(self.info(), fmt).map_err(DecodeRowsError::Decode)?;
        let mut pool = J2kScratchPool::new();
        let (packed, row) = pool.packed_bytes_and_row_u16(row_bytes, samples_per_row);
        for y in 0..self.info.dimensions.1 {
            self.decode_region_into_cached(
                packed,
                row_bytes,
                fmt,
                Rect {
                    x: 0,
                    y,
                    w: self.info.dimensions.0,
                    h: 1,
                },
            )
            .map_err(DecodeRowsError::Decode)?;
            for (dst, src) in row.iter_mut().zip(packed.chunks_exact(2)) {
                *dst = u16::from_le_bytes([src[0], src[1]]);
            }
            sink.write_row(y, row).map_err(DecodeRowsError::Sink)?;
        }
        Ok(ashlar_core::DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings: Vec::new(),
        })
    }
}

impl ImageCodec for J2kCodec {
    type Error = J2kError;
    type Warning = Infallible;
    type Pool = J2kScratchPool;
}

impl TileBatchDecode for J2kCodec {
    type Context = J2kContext;

    fn decode_tile(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<ashlar_core::DecodeOutcome<Self::Warning>, Self::Error> {
        ctx.codec_mut().record_tile_decode();
        let mut decoder = J2kDecoder::new(input)?;
        decoder.decode_into_with_scratch(pool, out, stride, fmt)
    }

    fn decode_tile_region(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<ashlar_core::DecodeOutcome<Self::Warning>, Self::Error> {
        ctx.codec_mut().record_tile_decode();
        let mut decoder = J2kDecoder::new(input)?;
        decoder.decode_region_into(pool, out, stride, fmt, roi)
    }

    fn decode_tile_scaled(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<ashlar_core::DecodeOutcome<Self::Warning>, Self::Error> {
        ctx.codec_mut().record_tile_decode();
        let mut decoder = J2kDecoder::new(input)?;
        decoder.decode_scaled_into(pool, out, stride, fmt, scale)
    }

    fn decode_tile_region_scaled(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<ashlar_core::DecodeOutcome<Self::Warning>, Self::Error> {
        ctx.codec_mut().record_tile_decode();
        let mut decoder = J2kDecoder::new(input)?;
        decoder.decode_region_scaled_into(pool, out, stride, fmt, roi, scale)
    }
}

fn row_format_u8(info: &Info) -> Result<PixelFormat, J2kError> {
    match info.components {
        1 => Ok(PixelFormat::Gray8),
        3 => Ok(PixelFormat::Rgb8),
        4 => Ok(PixelFormat::Rgba8),
        _ => Err(ashlar_core::Unsupported {
            what: "row decode only supports Gray/RGB/RGBA images in J2K-M2",
        }
        .into()),
    }
}

fn row_format_u16(info: &Info) -> Result<PixelFormat, J2kError> {
    match info.components {
        1 => Ok(PixelFormat::Gray16),
        3 => Ok(PixelFormat::Rgb16),
        4 => Err(ashlar_core::Unsupported {
            what: "Rgba16 row decode is not supported by ashlar-j2k",
        }
        .into()),
        _ => Err(ashlar_core::Unsupported {
            what: "row decode only supports Gray/RGB images in J2K-M2",
        }
        .into()),
    }
}

fn row_bytes_for(info: &Info, fmt: PixelFormat) -> Result<usize, J2kError> {
    (info.dimensions.0 as usize)
        .checked_mul(fmt.bytes_per_pixel())
        .ok_or(J2kError::DimensionOverflow {
            width: info.dimensions.0,
            height: info.dimensions.1,
        })
}

fn row_samples_for(info: &Info, fmt: PixelFormat) -> Result<usize, J2kError> {
    (info.dimensions.0 as usize)
        .checked_mul(fmt.channels())
        .ok_or(J2kError::DimensionOverflow {
            width: info.dimensions.0,
            height: info.dimensions.1,
        })
}

fn should_retry_with_backend(error: &J2kError) -> bool {
    matches!(
        error,
        J2kError::InvalidMarker {
            marker: 0x50
                | 0x53
                | 0x55
                | 0x57
                | 0x58
                | 0x59
                | 0x5C
                | 0x5D
                | 0x5E
                | 0x5F
                | 0x60
                | 0x61
                | 0x63
                | 0x64
                | 0x91
                | 0x92,
            ..
        }
    )
}
