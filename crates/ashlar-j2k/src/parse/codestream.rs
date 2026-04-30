// SPDX-License-Identifier: Apache-2.0

use super::{ParsedCod, ParsedSiz};
use crate::J2kError;
use ashlar_core::{InputError, TileLayout};

const MARKER_SOC: u8 = 0x4F;
const MARKER_SIZ: u8 = 0x51;
const MARKER_COD: u8 = 0x52;
const MARKER_SOT: u8 = 0x90;
const MARKER_SOD: u8 = 0x93;
const MARKER_EOC: u8 = 0xD9;

#[derive(Debug, Clone, Copy)]
pub(crate) struct CodestreamInfo {
    pub(crate) siz: ParsedSiz,
    pub(crate) cod: ParsedCod,
}

pub(crate) fn looks_like_codestream(input: &[u8]) -> bool {
    input.len() >= 2 && input[0] == 0xFF && input[1] == MARKER_SOC
}

pub(crate) fn parse_codestream(input: &[u8]) -> Result<CodestreamInfo, J2kError> {
    if input.len() < 2 {
        return Err(InputError::TooShort {
            need: 2,
            have: input.len(),
        }
        .into());
    }
    if !looks_like_codestream(input) {
        return Err(J2kError::InvalidMarker {
            offset: 0,
            marker: input[1],
        });
    }

    let mut offset = 2usize;
    let mut siz = None;
    let mut cod = None;
    let mut terminated = false;

    while offset < input.len() {
        let marker = read_marker(input, &mut offset)?;
        match marker {
            MARKER_SOT | MARKER_SOD | MARKER_EOC => {
                terminated = true;
                break;
            }
            MARKER_SIZ => {
                let payload = read_segment_payload(input, &mut offset, "SIZ")?;
                siz = Some(parse_siz(payload)?);
            }
            MARKER_COD => {
                let payload = read_segment_payload(input, &mut offset, "COD")?;
                cod = Some(parse_cod(payload)?);
            }
            _ => {
                let _ = read_segment_payload(input, &mut offset, "segment")?;
            }
        }
    }

    if !terminated {
        return Err(InputError::TruncatedAt {
            offset,
            segment: "main header terminator",
        }
        .into());
    }

    Ok(CodestreamInfo {
        siz: siz.ok_or(J2kError::MissingRequiredMarker { marker: "SIZ" })?,
        cod: cod.ok_or(J2kError::MissingRequiredMarker { marker: "COD" })?,
    })
}

fn read_marker(input: &[u8], offset: &mut usize) -> Result<u8, J2kError> {
    if *offset + 2 > input.len() {
        return Err(InputError::TruncatedAt {
            offset: *offset,
            segment: "marker",
        }
        .into());
    }
    if input[*offset] != 0xFF {
        return Err(J2kError::InvalidMarker {
            offset: *offset,
            marker: input[*offset],
        });
    }
    let marker = input[*offset + 1];
    *offset += 2;
    Ok(marker)
}

fn read_segment_payload<'a>(
    input: &'a [u8],
    offset: &mut usize,
    segment: &'static str,
) -> Result<&'a [u8], J2kError> {
    if *offset + 2 > input.len() {
        return Err(InputError::TruncatedAt {
            offset: *offset,
            segment,
        }
        .into());
    }
    let length = u16::from_be_bytes([input[*offset], input[*offset + 1]]) as usize;
    if length < 2 {
        return Err(J2kError::InvalidBox {
            offset: *offset,
            what: "segment length smaller than header",
        });
    }
    let start = *offset + 2;
    let end = *offset + length;
    if end > input.len() {
        return Err(InputError::TruncatedAt {
            offset: *offset,
            segment,
        }
        .into());
    }
    *offset = end;
    Ok(&input[start..end])
}

#[allow(clippy::similar_names)]
fn parse_siz(payload: &[u8]) -> Result<ParsedSiz, J2kError> {
    if payload.len() < 36 {
        return Err(J2kError::InvalidSiz {
            what: "payload shorter than fixed SIZ header",
        });
    }
    let x_size = read_u32(payload, 2);
    let y_size = read_u32(payload, 6);
    let x_origin = read_u32(payload, 10);
    let y_origin = read_u32(payload, 14);
    let tile_width = read_u32(payload, 18);
    let tile_height = read_u32(payload, 22);
    let tile_x_origin = read_u32(payload, 26);
    let tile_y_origin = read_u32(payload, 30);
    let component_count = read_u16(payload, 34);

    let component_bytes = usize::from(component_count) * 3;
    if payload.len() < 36 + component_bytes {
        return Err(J2kError::InvalidSiz {
            what: "component descriptors truncated",
        });
    }
    if component_count == 0 {
        return Err(J2kError::InvalidSiz {
            what: "component count must be non-zero",
        });
    }
    if component_count > u16::from(u8::MAX) {
        return Err(J2kError::Unsupported(ashlar_core::Unsupported {
            what: "component count > 255",
        }));
    }
    if x_size <= x_origin || y_size <= y_origin {
        return Err(J2kError::InvalidSiz {
            what: "image origin must be smaller than image size",
        });
    }
    if tile_width == 0 || tile_height == 0 {
        return Err(J2kError::InvalidSiz {
            what: "tile size must be non-zero",
        });
    }
    if tile_x_origin > x_size || tile_y_origin > y_size {
        return Err(J2kError::InvalidSiz {
            what: "tile origin must be within image bounds",
        });
    }

    let width = x_size - x_origin;
    let height = y_size - y_origin;
    let tiles_x = (x_size - tile_x_origin).div_ceil(tile_width);
    let tiles_y = (y_size - tile_y_origin).div_ceil(tile_height);
    let mut bit_depth = 0u8;
    for idx in 0..usize::from(component_count) {
        let ssiz = payload[36 + idx * 3];
        bit_depth = bit_depth.max((ssiz & 0x7F) + 1);
    }

    Ok(ParsedSiz {
        dimensions: (width, height),
        components: component_count as u8,
        bit_depth,
        tile_layout: TileLayout {
            tile_width,
            tile_height,
            tiles_x,
            tiles_y,
        },
    })
}

fn parse_cod(payload: &[u8]) -> Result<ParsedCod, J2kError> {
    if payload.len() < 10 {
        return Err(J2kError::InvalidCod {
            what: "payload shorter than fixed COD header",
        });
    }
    Ok(ParsedCod {
        resolution_levels: payload[5].saturating_add(1),
        has_mct: payload[4] != 0,
        reversible: payload[9] == 1,
    })
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_be_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}
