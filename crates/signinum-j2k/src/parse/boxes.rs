// SPDX-License-Identifier: Apache-2.0

use super::codestream::parse_codestream;
use crate::J2kError;
use signinum_core::{Colorspace, InputError};

const JP2_SIGNATURE: [u8; 12] = [0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0D, 0x0A, 0x87, 0x0A];
const JP2_SIGNATURE_PREFIX: [u8; 8] = [0, 0, 0, 12, b'j', b'P', b' ', b' '];

pub(crate) fn looks_like_jp2(input: &[u8]) -> bool {
    input.starts_with(&JP2_SIGNATURE_PREFIX)
}

pub(crate) fn parse_jp2(input: &[u8]) -> Result<signinum_core::Info, J2kError> {
    if input.len() < JP2_SIGNATURE.len() {
        return Err(InputError::TooShort {
            need: JP2_SIGNATURE.len(),
            have: input.len(),
        }
        .into());
    }

    let mut offset = 0usize;
    let mut saw_signature = false;
    let mut saw_ftyp = false;
    let mut saw_jp2h = false;
    let mut saw_ihdr = false;
    let mut colorspace = None;
    let mut codestream = None;

    while offset < input.len() {
        let header = read_box_header(input, offset)?;
        if header.end > input.len() {
            return Err(InputError::TruncatedAt {
                offset,
                segment: "box payload",
            }
            .into());
        }
        let payload = &input[header.payload_start..header.end];
        match &header.box_type {
            b"jP  " => {
                if saw_signature || offset != 0 {
                    return Err(J2kError::InvalidBox {
                        offset,
                        what: "signature box must appear exactly once at the start of the file",
                    });
                }
                if payload != &JP2_SIGNATURE[8..] {
                    return Err(J2kError::InvalidBox {
                        offset,
                        what: "invalid JP2 signature payload",
                    });
                }
                saw_signature = true;
            }
            b"ftyp" => {
                if !saw_signature || saw_ftyp || saw_jp2h || codestream.is_some() {
                    return Err(J2kError::InvalidBox {
                        offset,
                        what: "file type box must appear exactly once before jp2h and jp2c",
                    });
                }
                if payload.len() < 8 {
                    return Err(J2kError::InvalidBox {
                        offset,
                        what: "ftyp payload shorter than 8 bytes",
                    });
                }
                saw_ftyp = true;
            }
            b"jp2h" => {
                if !saw_ftyp || saw_jp2h || codestream.is_some() {
                    return Err(J2kError::InvalidBox {
                        offset,
                        what: "jp2h must appear exactly once after ftyp and before jp2c",
                    });
                }
                let (ihdr, colr) = parse_jp2h(payload, header.payload_start)?;
                saw_jp2h = true;
                saw_ihdr = ihdr;
                colorspace = colr.or(colorspace);
            }
            b"jp2c" => {
                if !saw_jp2h || codestream.is_some() {
                    return Err(J2kError::InvalidBox {
                        offset,
                        what: "jp2c must appear exactly once after jp2h",
                    });
                }
                codestream = Some(payload);
            }
            _ => {}
        }
        offset = header.end;
    }

    if !saw_signature {
        return Err(J2kError::MissingRequiredBox { box_type: "jP  " });
    }
    if !saw_ftyp {
        return Err(J2kError::MissingRequiredBox { box_type: "ftyp" });
    }
    if !saw_jp2h {
        return Err(J2kError::MissingRequiredBox { box_type: "jp2h" });
    }
    if !saw_ihdr {
        return Err(J2kError::MissingRequiredBox { box_type: "ihdr" });
    }
    let codestream = codestream.ok_or(J2kError::MissingRequiredBox { box_type: "jp2c" })?;
    Ok(parse_codestream(codestream)?.into_info(colorspace))
}

fn parse_jp2h(payload: &[u8], base_offset: usize) -> Result<(bool, Option<Colorspace>), J2kError> {
    let mut offset = 0usize;
    let mut saw_ihdr = false;
    let mut colorspace = None;

    while offset < payload.len() {
        let header = read_box_header(payload, offset)?;
        let inner = &payload[header.payload_start..header.end];
        match &header.box_type {
            b"ihdr" => {
                if inner.len() < 14 {
                    return Err(J2kError::InvalidBox {
                        offset: base_offset + offset,
                        what: "ihdr payload shorter than 14 bytes",
                    });
                }
                saw_ihdr = true;
            }
            b"colr" => {
                colorspace = parse_colr(inner).or(colorspace);
            }
            _ => {}
        }
        offset = header.end;
    }

    Ok((saw_ihdr, colorspace))
}

fn parse_colr(payload: &[u8]) -> Option<Colorspace> {
    if payload.len() < 3 {
        return None;
    }
    match payload[0] {
        1 if payload.len() >= 7 => {
            match u32::from_be_bytes([payload[3], payload[4], payload[5], payload[6]]) {
                16 => Some(Colorspace::SRgb),
                17 => Some(Colorspace::SGray),
                18 => Some(Colorspace::YCbCr),
                _ => Some(Colorspace::IccTagged),
            }
        }
        2 => Some(Colorspace::IccTagged),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
struct BoxHeader {
    box_type: [u8; 4],
    payload_start: usize,
    end: usize,
}

fn read_box_header(input: &[u8], offset: usize) -> Result<BoxHeader, J2kError> {
    if offset + 8 > input.len() {
        return Err(InputError::TruncatedAt {
            offset,
            segment: "box header",
        }
        .into());
    }
    let lbox = u32::from_be_bytes([
        input[offset],
        input[offset + 1],
        input[offset + 2],
        input[offset + 3],
    ]);
    let box_type = [
        input[offset + 4],
        input[offset + 5],
        input[offset + 6],
        input[offset + 7],
    ];

    let (payload_start, end) = match lbox {
        0 => (offset + 8, input.len()),
        1 => {
            if offset + 16 > input.len() {
                return Err(InputError::TruncatedAt {
                    offset,
                    segment: "extended box header",
                }
                .into());
            }
            let xlbox = u64::from_be_bytes([
                input[offset + 8],
                input[offset + 9],
                input[offset + 10],
                input[offset + 11],
                input[offset + 12],
                input[offset + 13],
                input[offset + 14],
                input[offset + 15],
            ]);
            if xlbox < 16 {
                return Err(J2kError::InvalidBox {
                    offset,
                    what: "extended box length smaller than header",
                });
            }
            let end = offset
                .checked_add(xlbox as usize)
                .ok_or(J2kError::InvalidBox {
                    offset,
                    what: "extended box length overflow",
                })?;
            (offset + 16, end)
        }
        length if length < 8 => {
            return Err(J2kError::InvalidBox {
                offset,
                what: "box length smaller than header",
            })
        }
        length => {
            let end = offset
                .checked_add(length as usize)
                .ok_or(J2kError::InvalidBox {
                    offset,
                    what: "box length overflow",
                })?;
            (offset + 8, end)
        }
    };

    Ok(BoxHeader {
        box_type,
        payload_start,
        end,
    })
}
