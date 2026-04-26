// SPDX-License-Identifier: Apache-2.0

mod boxes;
mod codestream;

use self::boxes::parse_jp2;
use self::codestream::{parse_codestream, CodestreamInfo};
use crate::J2kError;
use slidecodec_core::{Colorspace, Info, TileLayout, Unsupported};

pub(crate) fn parse_info(input: &[u8]) -> Result<Info, J2kError> {
    if boxes::looks_like_jp2(input) {
        return parse_jp2(input);
    }
    if codestream::looks_like_codestream(input) {
        let parsed = parse_codestream(input)?;
        return Ok(parsed.into_info(None));
    }
    Err(J2kError::Unsupported(Unsupported {
        what: "input is not a JP2 container or raw JPEG 2000 codestream",
    }))
}

fn infer_colorspace(components: u8, has_mct: bool, reversible: bool) -> Colorspace {
    match (components, has_mct, reversible) {
        (1, _, _) => Colorspace::SGray,
        (3, false, _) => Colorspace::Rgb,
        (3, true, false) => Colorspace::Ict,
        (3, true, true) => Colorspace::Rct,
        _ => Colorspace::IccTagged,
    }
}

#[derive(Debug, Clone, Copy)]
struct ParsedSiz {
    dimensions: (u32, u32),
    components: u8,
    bit_depth: u8,
    tile_layout: TileLayout,
}

#[derive(Debug, Clone, Copy)]
struct ParsedCod {
    resolution_levels: u8,
    has_mct: bool,
    reversible: bool,
}

impl CodestreamInfo {
    fn into_info(self, colorspace: Option<Colorspace>) -> Info {
        Info {
            dimensions: self.siz.dimensions,
            components: self.siz.components,
            colorspace: colorspace.unwrap_or_else(|| {
                infer_colorspace(self.siz.components, self.cod.has_mct, self.cod.reversible)
            }),
            bit_depth: self.siz.bit_depth,
            tile_layout: Some(self.siz.tile_layout),
            coded_unit_layout: None,
            restart_interval: None,
            resolution_levels: self.cod.resolution_levels,
        }
    }
}
