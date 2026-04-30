// SPDX-License-Identifier: Apache-2.0

use crate::J2kError;
use alloc::string::ToString;
use ashlar_core::{Colorspace, Info};

pub(crate) use ashlar_j2k_native::{ColorSpace, DecodeSettings, Image, RawBitmap};

pub(crate) fn image(bytes: &[u8], settings: DecodeSettings) -> Result<Image<'_>, J2kError> {
    Image::new(bytes, &settings).map_err(|err| J2kError::Backend(err.to_string()))
}

pub(crate) fn inspect_info(bytes: &[u8]) -> Result<Info, J2kError> {
    let image = image(bytes, DecodeSettings::default())?;
    Ok(inspect_info_from_image(&image))
}

pub(crate) fn inspect_info_from_image(image: &Image<'_>) -> Info {
    let components = image.color_space().num_channels() + u8::from(image.has_alpha());
    Info {
        dimensions: (image.width(), image.height()),
        components,
        colorspace: map_colorspace(image.color_space()),
        bit_depth: image.original_bit_depth(),
        tile_layout: None,
        coded_unit_layout: None,
        restart_interval: None,
        resolution_levels: 1,
    }
}

pub(crate) fn map_colorspace(color_space: &ColorSpace) -> Colorspace {
    match color_space {
        ColorSpace::Gray => Colorspace::SGray,
        ColorSpace::RGB => Colorspace::Rgb,
        ColorSpace::CMYK => Colorspace::Cmyk,
        ColorSpace::Unknown { .. } | ColorSpace::Icc { .. } => Colorspace::IccTagged,
    }
}
