mod arithmetic_decoder;
pub(crate) mod arithmetic_encoder;
pub(crate) mod bitplane;
pub(crate) mod bitplane_encode;
pub(crate) mod build;
pub(crate) mod capabilities;
pub(crate) mod codestream;
pub(crate) mod codestream_write;
pub(crate) mod coefficient_view;
mod decode;
pub(crate) mod encode;
pub(crate) mod fdwt;
pub(crate) mod forward_mct;
pub(crate) mod ht_block_decode;
pub(crate) mod ht_block_encode;
pub(crate) mod ht_encode_tables;
pub(crate) mod ht_tables;
pub(crate) mod idwt;
mod mct;
mod mq;
pub(crate) mod packet_encode;
mod progression;
pub(crate) mod quantize;
pub(crate) mod recode;
mod rect;
mod roi;
mod segment;
mod tag_tree;
pub(crate) mod tag_tree_encode;
mod tile;

#[cfg(test)]
pub(crate) use tile::{reset_tile_parse_calls, tile_parse_calls};

use alloc::vec::Vec;

use super::jp2::colr::{ColorSpace, ColorSpecificationBox, EnumeratedColorspace};
use super::jp2::ImageBoxes;
use crate::error::{bail, FormatError, MarkerError, Result};
use crate::image::{ImageProperties, ImageSource};
use crate::j2c::codestream::markers;
use crate::reader::BitReader;
use crate::{resolve_alpha_and_color_space, DecodeSettings, Image};

use crate::math::{SimdBuffer, SIMD_WIDTH};
pub(crate) use codestream::Header;
#[cfg(test)]
pub(crate) use decode::should_decode_classic_sub_band_in_parallel;
pub(crate) use decode::{
    build_component_grid_color_plan, build_direct_color_plan, build_direct_grayscale_plan,
    build_referenced_classic_color_plan, build_referenced_classic_grayscale_plan,
    build_referenced_classic_rgba_plan, build_referenced_htj2k_color_plan,
    build_referenced_htj2k_grayscale_plan, build_referenced_htj2k_rgba_plan,
    decode_preparsed_with_capacity_retry as decode_preparsed, decode_with_capacity_retry as decode,
    prepare_region_tiles,
};
pub use decode::{CpuDecodeParallelism, DecoderContext, DecoderWorkspace, DecoderWorkspaceStats};
pub use recode::Reversible53CoefficientImage;
pub(crate) use segment::MAX_BITPLANE_COUNT;
pub(crate) use tile::ParsedTiles;

pub(crate) struct ParsedCodestream<'a> {
    pub(crate) header: Header<'a>,
    pub(crate) data: &'a [u8],
}

#[derive(Debug)]
pub(crate) struct ComponentData {
    pub(crate) container: SimdBuffer<{ SIMD_WIDTH }>,
    pub(crate) integer_container: Option<Vec<i64>>,
    pub(crate) bit_depth: u8,
    pub(crate) signed: bool,
}

crate::move_only::assert_move_only!(ComponentData);

pub(crate) fn parse<'a>(
    stream: &'a [u8],
    settings: &DecodeSettings,
    exact_reduction_levels: Option<u8>,
) -> Result<Image<'a>> {
    parse_with_retained_baseline(stream, settings, 0, exact_reduction_levels)
}

pub(crate) fn parse_with_retained_baseline<'a>(
    stream: &'a [u8],
    settings: &DecodeSettings,
    retained_baseline_bytes: usize,
    exact_reduction_levels: Option<u8>,
) -> Result<Image<'a>> {
    let mut strict_settings = *settings;
    strict_settings.strict = true;
    let parsed_codestream = parse_raw_with_retained_baseline(
        stream,
        &strict_settings,
        retained_baseline_bytes,
        exact_reduction_levels,
    )?;
    let header = &parsed_codestream.header;
    // Raw codestreams do not carry JP2 channel definitions. Keep the
    // conventional grayscale/RGB assumptions for 1- and 3-component images,
    // but preserve two-component data as independent channels instead of
    // forcing it through grayscale validation.
    let (cs, enumerated_value) = match header.component_infos.len() {
        1 => (
            ColorSpace::Enumerated(EnumeratedColorspace::Greyscale),
            Some(17),
        ),
        2 => (ColorSpace::Unknown, None),
        _ => (ColorSpace::Enumerated(EnumeratedColorspace::Srgb), Some(16)),
    };

    let color_specification = ColorSpecificationBox {
        method: u8::from(enumerated_value.is_some()),
        enumerated_value,
        color_space: cs,
    };
    let boxes = ImageBoxes::try_with_synthetic_color_specification(
        header,
        color_specification,
        retained_baseline_bytes,
    )?;

    let (color_space, has_alpha, _) = resolve_alpha_and_color_space(
        &boxes,
        &parsed_codestream.header,
        &strict_settings,
        retained_baseline_bytes,
    )?;
    let properties = ImageProperties::new(boxes, *settings, color_space, has_alpha, false);
    if retained_baseline_bytes == 0 {
        Image::from_parsed_parts(
            ImageSource::new(stream, parsed_codestream.data),
            parsed_codestream.header,
            properties,
        )
    } else {
        Image::from_parsed_parts_with_retained_baseline(
            ImageSource::new(stream, parsed_codestream.data),
            parsed_codestream.header,
            properties,
            retained_baseline_bytes,
        )
    }
}

#[cfg(test)]
pub(crate) fn parse_raw<'a>(
    stream: &'a [u8],
    settings: &DecodeSettings,
) -> Result<ParsedCodestream<'a>> {
    parse_raw_with_retained_baseline(stream, settings, 0, None)
}

pub(crate) fn parse_raw_with_retained_baseline<'a>(
    stream: &'a [u8],
    settings: &DecodeSettings,
    retained_baseline_bytes: usize,
    exact_reduction_levels: Option<u8>,
) -> Result<ParsedCodestream<'a>> {
    let mut reader = BitReader::new(stream);

    let marker = reader.read_marker()?;
    if marker != markers::SOC {
        bail!(MarkerError::Expected("SOC"));
    }

    let header = codestream::read_header(
        &mut reader,
        settings,
        retained_baseline_bytes,
        exact_reduction_levels,
    )?;
    let code_stream_data = reader.tail().ok_or(FormatError::MissingCodestream)?;

    Ok(ParsedCodestream {
        header,
        data: code_stream_data,
    })
}
