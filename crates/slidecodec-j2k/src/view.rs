// SPDX-License-Identifier: Apache-2.0

use crate::{
    backend::{
        image as backend_image, inspect_info, inspect_info_from_image, DecodeSettings, Image,
    },
    context::J2kContext,
    decode::{
        decode_image_into_with_native_context, decode_image_region_into_with_native_context,
        decode_scaled_from_info, validate_buffer, validate_region, validate_supported_format,
        J2kDecodeOutcome,
    },
    parse::parse_info,
    scratch::J2kScratchPool,
    J2kError,
};
use alloc::vec::Vec;
use core::convert::Infallible;
use slidecodec_core::{
    DecodeRowsError, DecoderContext, Downscale, ImageCodec, ImageDecode, ImageDecodeRows, Info,
    PixelFormat, Rect, RowSink, TileBatchDecode,
};

pub struct J2kView<'a> {
    bytes: &'a [u8],
    info: Info,
    image: Option<Image<'a>>,
}

impl<'a> J2kView<'a> {
    pub fn parse(input: &'a [u8]) -> Result<Self, J2kError> {
        let info = match parse_info(input) {
            Ok(info) => info,
            Err(error) if should_retry_with_backend(&error) => inspect_info(input)?,
            Err(error) => return Err(error),
        };
        let image = backend_image(input, DecodeSettings::default()).ok();
        Ok(Self {
            bytes: input,
            info,
            image,
        })
    }

    pub fn info(&self) -> &Info {
        &self.info
    }

    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

pub struct J2kDecoder<'a> {
    bytes: &'a [u8],
    info: Info,
    image: Option<Image<'a>>,
    native_context: slidecodec_j2k_native::DecoderContext<'a>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct J2kCodec;

impl<'a> J2kDecoder<'a> {
    pub fn inspect(input: &'a [u8]) -> Result<Info, J2kError> {
        match parse_info(input) {
            Ok(info) => Ok(info),
            Err(error) if should_retry_with_backend(&error) => inspect_info(input),
            Err(error) => Err(error),
        }
    }

    pub fn new(input: &'a [u8]) -> Result<Self, J2kError> {
        Self::from_view(J2kView::parse(input)?)
    }

    pub fn from_view(view: J2kView<'a>) -> Result<Self, J2kError> {
        Ok(Self {
            bytes: view.bytes,
            info: view.info,
            image: view.image,
            native_context: slidecodec_j2k_native::DecoderContext::default(),
        })
    }

    pub fn info(&self) -> &Info {
        &self.info
    }

    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    pub fn decode_into(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        self.decode_into_with_scratch(&mut J2kScratchPool::new(), out, stride, fmt)
    }

    pub fn decode_into_with_scratch(
        &mut self,
        _pool: &mut J2kScratchPool,
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
        Ok(slidecodec_core::DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings: Vec::new(),
        })
    }

    pub fn decode_region_into(
        &mut self,
        pool: &mut J2kScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        let _ = pool;
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
        Ok(slidecodec_core::DecodeOutcome {
            decoded: roi,
            warnings: Vec::new(),
        })
    }

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
    ) -> Result<slidecodec_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_into(self, out, stride, fmt)
    }

    fn decode_into_with_scratch(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<slidecodec_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_into_with_scratch(self, pool, out, stride, fmt)
    }

    fn decode_region_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<slidecodec_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_region_into(self, pool, out, stride, fmt, roi)
    }

    fn decode_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<slidecodec_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_scaled_into(self, pool, out, stride, fmt, scale)
    }
}

impl<'a> ImageDecodeRows<'a, u8> for J2kDecoder<'a> {
    fn decode_rows<R: RowSink<u8>>(
        &mut self,
        sink: &mut R,
    ) -> Result<slidecodec_core::DecodeOutcome<Self::Warning>, DecodeRowsError<Self::Error, R::Error>>
    {
        let fmt = row_format_u8(self.info()).map_err(DecodeRowsError::Decode)?;
        let row_bytes = row_bytes_for(self.info(), fmt).map_err(DecodeRowsError::Decode)?;
        let total_len =
            total_output_bytes(self.info(), row_bytes).map_err(DecodeRowsError::Decode)?;
        let mut pool = J2kScratchPool::new();
        let packed = pool.packed_bytes(total_len);
        self.decode_into(packed, row_bytes, fmt)
            .map_err(DecodeRowsError::Decode)?;
        for (y, row) in packed.chunks_exact(row_bytes).enumerate() {
            sink.write_row(y as u32, row)
                .map_err(DecodeRowsError::Sink)?;
        }
        Ok(slidecodec_core::DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings: Vec::new(),
        })
    }
}

impl<'a> ImageDecodeRows<'a, u16> for J2kDecoder<'a> {
    fn decode_rows<R: RowSink<u16>>(
        &mut self,
        sink: &mut R,
    ) -> Result<slidecodec_core::DecodeOutcome<Self::Warning>, DecodeRowsError<Self::Error, R::Error>>
    {
        let fmt = row_format_u16(self.info()).map_err(DecodeRowsError::Decode)?;
        let row_bytes = row_bytes_for(self.info(), fmt).map_err(DecodeRowsError::Decode)?;
        let samples_per_row = row_samples_for(self.info(), fmt).map_err(DecodeRowsError::Decode)?;
        let total_len =
            total_output_bytes(self.info(), row_bytes).map_err(DecodeRowsError::Decode)?;
        let mut pool = J2kScratchPool::new();
        let (packed, row) = pool.packed_bytes_and_row_u16(total_len, samples_per_row);
        self.decode_into(packed, row_bytes, fmt)
            .map_err(DecodeRowsError::Decode)?;
        for (y, row_bytes) in packed.chunks_exact(row_bytes).enumerate() {
            for (dst, src) in row.iter_mut().zip(row_bytes.chunks_exact(2)) {
                *dst = u16::from_le_bytes([src[0], src[1]]);
            }
            sink.write_row(y as u32, row)
                .map_err(DecodeRowsError::Sink)?;
        }
        Ok(slidecodec_core::DecodeOutcome {
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
        _ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<slidecodec_core::DecodeOutcome<Self::Warning>, Self::Error> {
        let mut decoder = J2kDecoder::new(input)?;
        decoder.decode_into_with_scratch(pool, out, stride, fmt)
    }

    fn decode_tile_region(
        _ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<slidecodec_core::DecodeOutcome<Self::Warning>, Self::Error> {
        let mut decoder = J2kDecoder::new(input)?;
        decoder.decode_region_into(pool, out, stride, fmt, roi)
    }

    fn decode_tile_scaled(
        _ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<slidecodec_core::DecodeOutcome<Self::Warning>, Self::Error> {
        let mut decoder = J2kDecoder::new(input)?;
        decoder.decode_scaled_into(pool, out, stride, fmt, scale)
    }
}

fn row_format_u8(info: &Info) -> Result<PixelFormat, J2kError> {
    match info.components {
        1 => Ok(PixelFormat::Gray8),
        3 => Ok(PixelFormat::Rgb8),
        4 => Ok(PixelFormat::Rgba8),
        _ => Err(slidecodec_core::Unsupported {
            what: "row decode only supports Gray/RGB/RGBA images in J2K-M2",
        }
        .into()),
    }
}

fn row_format_u16(info: &Info) -> Result<PixelFormat, J2kError> {
    match info.components {
        1 => Ok(PixelFormat::Gray16),
        3 => Ok(PixelFormat::Rgb16),
        4 => Err(slidecodec_core::Unsupported {
            what: "Rgba16 row decode is not supported by slidecodec-j2k",
        }
        .into()),
        _ => Err(slidecodec_core::Unsupported {
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

fn total_output_bytes(info: &Info, row_bytes: usize) -> Result<usize, J2kError> {
    row_bytes
        .checked_mul(info.dimensions.1 as usize)
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
