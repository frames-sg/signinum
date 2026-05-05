// SPDX-License-Identifier: Apache-2.0

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DicomFrameExtractError {
    #[error("DICOM Pixel Data element was not found")]
    PixelDataNotFound,
    #[error("DICOM Pixel Data is not encapsulated")]
    PixelDataNotEncapsulated,
    #[error("DICOM Pixel Data is truncated while reading {what}")]
    Truncated { what: &'static str },
    #[error("DICOM Pixel Data is malformed: {what}")]
    Malformed { what: &'static str },
}

const PIXEL_DATA_TAG: [u8; 4] = [0xE0, 0x7F, 0x10, 0x00];
const EXTENDED_OFFSET_TABLE_TAG: [u8; 4] = [0xE0, 0x7F, 0x01, 0x00];
const EXTENDED_OFFSET_TABLE_LENGTHS_TAG: [u8; 4] = [0xE0, 0x7F, 0x02, 0x00];
const ITEM_TAG: [u8; 4] = [0xFE, 0xFF, 0x00, 0xE0];
const SEQUENCE_DELIMITATION_TAG: [u8; 4] = [0xFE, 0xFF, 0xDD, 0xE0];
const UNDEFINED_LENGTH: u32 = u32::MAX;

pub fn extract_dicom_encapsulated_frames(
    input: &[u8],
) -> Result<Vec<Vec<u8>>, DicomFrameExtractError> {
    extract_dicom_encapsulated_frames_with_limit(input, usize::MAX)
}

pub fn extract_dicom_encapsulated_frames_with_limit(
    input: &[u8],
    limit: usize,
) -> Result<Vec<Vec<u8>>, DicomFrameExtractError> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let pixel_data = find_encapsulated_pixel_data_value(input)?;
    let extended_offsets = extended_offset_table_entries(input, pixel_data.tag_offset)?;
    let extended_lengths = extended_offset_table_length_entries(input, pixel_data.tag_offset)?;
    if let (Some(offsets), Some(lengths)) = (&extended_offsets, &extended_lengths) {
        if offsets.len() != lengths.len() {
            return Err(DicomFrameExtractError::Malformed {
                what: "Extended Offset Table and Lengths have different entry counts",
            });
        }
    }

    let (bot, fragment_cursor) = read_basic_offset_table(input, pixel_data.value_offset)?;
    let offsets = basic_offset_table_entries(bot)?;
    let extended_offsets = offsets
        .is_empty()
        .then_some(extended_offsets)
        .flatten()
        .unwrap_or_default();
    let stop_offset = offsets
        .get(limit)
        .map(|offset| u64::from(*offset))
        .or_else(|| extended_offsets.get(limit).copied());
    let fragments = read_pixel_data_fragments(
        input,
        fragment_cursor,
        stop_offset,
        (offsets.is_empty() && extended_offsets.is_empty()).then_some(limit),
    )?;
    if fragments.is_empty() {
        return Ok(Vec::new());
    }

    if offsets.is_empty() && extended_offsets.is_empty() {
        return Ok(fragments
            .into_iter()
            .map(|fragment| fragment.payload.to_vec())
            .collect());
    }

    if !offsets.is_empty() {
        let frame_count = offsets.len().min(limit);
        let offsets = offsets[..frame_count]
            .iter()
            .copied()
            .map(u64::from)
            .collect::<Vec<_>>();
        return frames_from_offset_table(&fragments, &offsets, None);
    }

    let frame_count = extended_offsets.len().min(limit);
    let lengths = extended_lengths
        .as_ref()
        .map(|lengths| &lengths[..frame_count]);
    frames_from_offset_table(&fragments, &extended_offsets[..frame_count], lengths)
}

#[derive(Clone, Copy)]
struct PixelDataLocation {
    tag_offset: usize,
    value_offset: usize,
}

fn find_encapsulated_pixel_data_value(
    input: &[u8],
) -> Result<PixelDataLocation, DicomFrameExtractError> {
    let Some(tag_offset) = input
        .windows(PIXEL_DATA_TAG.len())
        .position(|window| window == PIXEL_DATA_TAG)
    else {
        return Err(DicomFrameExtractError::PixelDataNotFound);
    };
    let header = input
        .get(tag_offset..)
        .ok_or(DicomFrameExtractError::Truncated {
            what: "Pixel Data header",
        })?;
    if header.len() < 8 {
        return Err(DicomFrameExtractError::Truncated {
            what: "Pixel Data header",
        });
    }

    let after_tag = tag_offset + 4;
    let vr = input
        .get(after_tag..after_tag + 2)
        .ok_or(DicomFrameExtractError::Truncated {
            what: "Pixel Data VR",
        })?;
    let (value_offset, length) = if is_explicit_vr(vr) {
        if is_long_explicit_vr(vr) {
            let length_offset = after_tag + 4;
            (
                after_tag + 8,
                read_u32(input, length_offset, "Pixel Data length")?,
            )
        } else {
            let length = u32::from(read_u16(input, after_tag + 2, "Pixel Data length")?);
            (after_tag + 4, length)
        }
    } else {
        (
            after_tag + 4,
            read_u32(input, after_tag, "Pixel Data length")?,
        )
    };

    if length != UNDEFINED_LENGTH {
        return Err(DicomFrameExtractError::PixelDataNotEncapsulated);
    }
    if value_offset > input.len() {
        return Err(DicomFrameExtractError::Truncated {
            what: "Pixel Data value",
        });
    }
    Ok(PixelDataLocation {
        tag_offset,
        value_offset,
    })
}

#[derive(Clone, Copy)]
struct DicomFragment<'a> {
    item_offset: u64,
    payload: &'a [u8],
}

fn read_basic_offset_table(
    input: &[u8],
    cursor: usize,
) -> Result<(&[u8], usize), DicomFrameExtractError> {
    let tag = input
        .get(cursor..cursor + 4)
        .ok_or(DicomFrameExtractError::Truncated {
            what: "Basic Offset Table item tag",
        })?;
    if tag != ITEM_TAG {
        return Err(DicomFrameExtractError::Malformed {
            what: "encapsulated Pixel Data does not start with a Basic Offset Table item",
        });
    }
    let length = usize::try_from(read_u32(input, cursor + 4, "Basic Offset Table length")?)
        .map_err(|_| DicomFrameExtractError::Malformed {
            what: "Basic Offset Table length exceeds usize",
        })?;
    let payload_start = cursor + 8;
    let payload_end =
        payload_start
            .checked_add(length)
            .ok_or(DicomFrameExtractError::Malformed {
                what: "Basic Offset Table length overflow",
            })?;
    let payload =
        input
            .get(payload_start..payload_end)
            .ok_or(DicomFrameExtractError::Truncated {
                what: "Basic Offset Table payload",
            })?;
    Ok((payload, payload_end))
}

fn read_pixel_data_fragments(
    input: &[u8],
    mut cursor: usize,
    stop_item_offset: Option<u64>,
    max_fragments: Option<usize>,
) -> Result<Vec<DicomFragment<'_>>, DicomFrameExtractError> {
    let mut fragments = Vec::new();
    let mut first_fragment_item_offset = None;
    loop {
        let tag = input
            .get(cursor..cursor + 4)
            .ok_or(DicomFrameExtractError::Truncated {
                what: "Pixel Data item tag",
            })?;
        let length = read_u32(input, cursor + 4, "Pixel Data item length")?;
        cursor += 8;
        if tag == SEQUENCE_DELIMITATION_TAG {
            if length != 0 {
                return Err(DicomFrameExtractError::Malformed {
                    what: "non-zero sequence delimitation length",
                });
            }
            break;
        }
        if tag != ITEM_TAG {
            return Err(DicomFrameExtractError::Malformed {
                what: "expected item tag in encapsulated Pixel Data",
            });
        }
        if length == UNDEFINED_LENGTH {
            return Err(DicomFrameExtractError::Malformed {
                what: "undefined-length Pixel Data item",
            });
        }
        let length = usize::try_from(length).map_err(|_| DicomFrameExtractError::Malformed {
            what: "Pixel Data item length exceeds usize",
        })?;
        let payload_end = cursor
            .checked_add(length)
            .ok_or(DicomFrameExtractError::Malformed {
                what: "Pixel Data item length overflow",
            })?;
        let payload = input
            .get(cursor..payload_end)
            .ok_or(DicomFrameExtractError::Truncated {
                what: "Pixel Data item payload",
            })?;

        let item_tag_offset = cursor - 8;
        let first = *first_fragment_item_offset.get_or_insert(item_tag_offset);
        let item_offset = u64::try_from(item_tag_offset - first).map_err(|_| {
            DicomFrameExtractError::Malformed {
                what: "fragment item offset exceeds u64",
            }
        })?;
        if stop_item_offset.is_some_and(|stop| item_offset >= stop) {
            break;
        }
        fragments.push(DicomFragment {
            item_offset,
            payload,
        });
        if max_fragments.is_some_and(|max| fragments.len() >= max) {
            break;
        }
        cursor = payload_end;
    }

    Ok(fragments)
}

fn extended_offset_table_entries(
    input: &[u8],
    before: usize,
) -> Result<Option<Vec<u64>>, DicomFrameExtractError> {
    let Some(payload) = read_element_value_before(
        input,
        EXTENDED_OFFSET_TABLE_TAG,
        before,
        "Extended Offset Table",
    )?
    else {
        return Ok(None);
    };
    Ok(Some(u64_entries(payload, "Extended Offset Table")?))
}

fn extended_offset_table_length_entries(
    input: &[u8],
    before: usize,
) -> Result<Option<Vec<u64>>, DicomFrameExtractError> {
    let Some(payload) = read_element_value_before(
        input,
        EXTENDED_OFFSET_TABLE_LENGTHS_TAG,
        before,
        "Extended Offset Table Lengths",
    )?
    else {
        return Ok(None);
    };
    Ok(Some(u64_entries(payload, "Extended Offset Table Lengths")?))
}

fn read_element_value_before<'a>(
    input: &'a [u8],
    tag: [u8; 4],
    before: usize,
    name: &'static str,
) -> Result<Option<&'a [u8]>, DicomFrameExtractError> {
    let search = input
        .get(..before)
        .ok_or(DicomFrameExtractError::Truncated {
            what: "DICOM element search range",
        })?;
    let Some(tag_offset) = search.windows(tag.len()).position(|window| window == tag) else {
        return Ok(None);
    };
    read_element_value_at(input, tag_offset, name).map(Some)
}

fn read_element_value_at<'a>(
    input: &'a [u8],
    tag_offset: usize,
    name: &'static str,
) -> Result<&'a [u8], DicomFrameExtractError> {
    let after_tag = tag_offset + 4;
    let vr = input
        .get(after_tag..after_tag + 2)
        .ok_or(DicomFrameExtractError::Truncated { what: name })?;
    let (value_offset, length) = if is_explicit_vr(vr) {
        if is_long_explicit_vr(vr) {
            (
                after_tag + 8,
                read_u32(input, after_tag + 4, "DICOM element length")?,
            )
        } else {
            (
                after_tag + 4,
                u32::from(read_u16(input, after_tag + 2, "DICOM element length")?),
            )
        }
    } else {
        (
            after_tag + 4,
            read_u32(input, after_tag, "DICOM element length")?,
        )
    };
    if length == UNDEFINED_LENGTH {
        return Err(DicomFrameExtractError::Malformed {
            what: "Extended Offset Table element has undefined length",
        });
    }
    let length = usize::try_from(length).map_err(|_| DicomFrameExtractError::Malformed {
        what: "DICOM element length exceeds usize",
    })?;
    let value_end = value_offset
        .checked_add(length)
        .ok_or(DicomFrameExtractError::Malformed {
            what: "DICOM element length overflow",
        })?;
    input
        .get(value_offset..value_end)
        .ok_or(DicomFrameExtractError::Truncated { what: name })
}

fn u64_entries(payload: &[u8], name: &'static str) -> Result<Vec<u64>, DicomFrameExtractError> {
    if !payload.len().is_multiple_of(8) {
        return Err(DicomFrameExtractError::Malformed { what: name });
    }
    Ok(payload
        .chunks_exact(8)
        .map(|chunk| {
            u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect())
}

fn basic_offset_table_entries(bot: &[u8]) -> Result<Vec<u32>, DicomFrameExtractError> {
    if bot.is_empty() {
        return Ok(Vec::new());
    }
    if !bot.len().is_multiple_of(4) {
        return Err(DicomFrameExtractError::Malformed {
            what: "Basic Offset Table length is not a multiple of four",
        });
    }
    Ok(bot
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn frames_from_offset_table(
    fragments: &[DicomFragment<'_>],
    offsets: &[u64],
    lengths: Option<&[u64]>,
) -> Result<Vec<Vec<u8>>, DicomFrameExtractError> {
    let mut frames = Vec::with_capacity(offsets.len());
    for (index, start) in offsets.iter().copied().enumerate() {
        let end = offsets.get(index + 1).copied().unwrap_or(u64::MAX);
        if end <= start {
            return Err(DicomFrameExtractError::Malformed {
                what: "offset table offsets are not strictly increasing",
            });
        }
        let mut frame = Vec::new();
        let max_payload_len = lengths
            .and_then(|entries| entries.get(index))
            .and_then(|length| usize::try_from(*length).ok());
        for fragment in fragments
            .iter()
            .filter(|fragment| fragment.item_offset >= start && fragment.item_offset < end)
        {
            if let Some(max_payload_len) = max_payload_len {
                let remaining = max_payload_len.saturating_sub(frame.len());
                frame.extend_from_slice(&fragment.payload[..fragment.payload.len().min(remaining)]);
                if frame.len() >= max_payload_len {
                    break;
                }
            } else {
                frame.extend_from_slice(fragment.payload);
            }
        }
        if frame.is_empty() {
            return Err(DicomFrameExtractError::Malformed {
                what: "offset table frame has no fragments",
            });
        }
        frames.push(frame);
    }
    Ok(frames)
}

fn is_explicit_vr(vr: &[u8]) -> bool {
    vr.len() == 2 && vr.iter().all(u8::is_ascii_uppercase)
}

fn is_long_explicit_vr(vr: &[u8]) -> bool {
    matches!(
        vr,
        b"OB" | b"OD" | b"OF" | b"OL" | b"OV" | b"OW" | b"SQ" | b"UC" | b"UR" | b"UT" | b"UN"
    )
}

fn read_u16(
    input: &[u8],
    offset: usize,
    what: &'static str,
) -> Result<u16, DicomFrameExtractError> {
    let bytes = input
        .get(offset..offset + 2)
        .ok_or(DicomFrameExtractError::Truncated { what })?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32(
    input: &[u8],
    offset: usize,
    what: &'static str,
) -> Result<u32, DicomFrameExtractError> {
    let bytes = input
        .get(offset..offset + 4)
        .ok_or(DicomFrameExtractError::Truncated { what })?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_item(bytes: &mut Vec<u8>, payload: &[u8]) {
        bytes.extend_from_slice(&[0xFE, 0xFF, 0x00, 0xE0]);
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(payload);
    }

    fn explicit_ob_pixel_data(bot: &[u8], fragments: &[&[u8]]) -> Vec<u8> {
        let mut bytes = vec![0; 128];
        bytes.extend_from_slice(b"DICM");
        bytes.extend_from_slice(&[0xE0, 0x7F, 0x10, 0x00]);
        bytes.extend_from_slice(b"OB");
        bytes.extend_from_slice(&[0, 0]);
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        push_item(&mut bytes, bot);
        for fragment in fragments {
            push_item(&mut bytes, fragment);
        }
        bytes.extend_from_slice(&[0xFE, 0xFF, 0xDD, 0xE0]);
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes
    }

    fn push_explicit_ov_element(bytes: &mut Vec<u8>, tag: [u8; 4], payload: &[u8]) {
        bytes.extend_from_slice(&tag);
        bytes.extend_from_slice(b"OV");
        bytes.extend_from_slice(&[0, 0]);
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(payload);
    }

    fn explicit_ob_pixel_data_with_extended_offsets(
        offsets: &[u64],
        lengths: &[u64],
        bot: &[u8],
        fragments: &[&[u8]],
    ) -> Vec<u8> {
        let mut bytes = vec![0; 128];
        bytes.extend_from_slice(b"DICM");
        let mut offset_payload = Vec::new();
        for offset in offsets {
            offset_payload.extend_from_slice(&offset.to_le_bytes());
        }
        push_explicit_ov_element(&mut bytes, [0xE0, 0x7F, 0x01, 0x00], &offset_payload);
        let mut length_payload = Vec::new();
        for length in lengths {
            length_payload.extend_from_slice(&length.to_le_bytes());
        }
        push_explicit_ov_element(&mut bytes, [0xE0, 0x7F, 0x02, 0x00], &length_payload);
        bytes.extend_from_slice(&[0xE0, 0x7F, 0x10, 0x00]);
        bytes.extend_from_slice(b"OB");
        bytes.extend_from_slice(&[0, 0]);
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        push_item(&mut bytes, bot);
        for fragment in fragments {
            push_item(&mut bytes, fragment);
        }
        bytes.extend_from_slice(&[0xFE, 0xFF, 0xDD, 0xE0]);
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes
    }

    #[test]
    fn dicom_empty_basic_offset_table_extracts_each_fragment_as_frame() {
        let bytes = explicit_ob_pixel_data(&[], &[b"first", b"second"]);

        let frames = extract_dicom_encapsulated_frames(&bytes).expect("extract frames");

        assert_eq!(frames, vec![b"first".to_vec(), b"second".to_vec()]);
    }

    #[test]
    fn dicom_basic_offset_table_groups_fragments_into_frames() {
        let first = b"aa";
        let second = b"bb";
        let third = b"cc";
        let second_frame_offset = 8 + first.len() as u32 + 8 + second.len() as u32;
        let mut bot = Vec::new();
        bot.extend_from_slice(&0_u32.to_le_bytes());
        bot.extend_from_slice(&second_frame_offset.to_le_bytes());
        let bytes = explicit_ob_pixel_data(&bot, &[first, second, third]);

        let frames = extract_dicom_encapsulated_frames(&bytes).expect("extract frames");

        assert_eq!(frames, vec![b"aabb".to_vec(), b"cc".to_vec()]);
    }

    #[test]
    fn dicom_extended_offset_table_lengths_trim_padded_frame_payloads() {
        let first_with_padding = b"abc\0";
        let second = b"de";
        let second_frame_offset = 8 + first_with_padding.len() as u64;
        let bytes = explicit_ob_pixel_data_with_extended_offsets(
            &[0, second_frame_offset],
            &[3, second.len() as u64],
            &[],
            &[first_with_padding, second],
        );

        let frames = extract_dicom_encapsulated_frames(&bytes).expect("extract frames");

        assert_eq!(frames, vec![b"abc".to_vec(), b"de".to_vec()]);
    }

    #[test]
    fn dicom_extract_limit_stops_after_requested_empty_bot_frames() {
        let bytes = explicit_ob_pixel_data(&[], &[b"first", b"second", b"third"]);

        let frames =
            extract_dicom_encapsulated_frames_with_limit(&bytes, 2).expect("extract frames");

        assert_eq!(frames, vec![b"first".to_vec(), b"second".to_vec()]);
    }

    #[test]
    fn dicom_extract_limit_stops_after_requested_basic_offset_frames() {
        let first = b"aa";
        let second = b"bb";
        let third = b"cc";
        let fourth = b"dd";
        let second_frame_offset = 8 + first.len() as u32 + 8 + second.len() as u32;
        let third_frame_offset = second_frame_offset + 8 + third.len() as u32;
        let mut bot = Vec::new();
        bot.extend_from_slice(&0_u32.to_le_bytes());
        bot.extend_from_slice(&second_frame_offset.to_le_bytes());
        bot.extend_from_slice(&third_frame_offset.to_le_bytes());
        let bytes = explicit_ob_pixel_data(&bot, &[first, second, third, fourth]);

        let frames =
            extract_dicom_encapsulated_frames_with_limit(&bytes, 2).expect("extract frames");

        assert_eq!(frames, vec![b"aabb".to_vec(), b"cc".to_vec()]);
    }
}
