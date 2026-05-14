// SPDX-License-Identifier: Apache-2.0

use crate::{error::BufferError, pixel::PixelFormat};

/// Copy tightly packed pixel rows into a caller-provided strided output buffer.
///
/// `src` must contain at least `width * height * fmt.bytes_per_pixel()` bytes.
/// The destination may have row padding, expressed by `stride`.
pub fn copy_tight_pixels_to_strided_output(
    src: &[u8],
    dimensions: (u32, u32),
    fmt: PixelFormat,
    out: &mut [u8],
    stride: usize,
) -> Result<(), BufferError> {
    if dimensions.0 == 0 || dimensions.1 == 0 {
        return Ok(());
    }

    let row_bytes = (dimensions.0 as usize)
        .checked_mul(fmt.bytes_per_pixel())
        .ok_or(BufferError::SizeOverflow {
            what: "row byte count",
        })?;
    if stride < row_bytes {
        return Err(BufferError::StrideTooSmall { row_bytes, stride });
    }
    let height = dimensions.1 as usize;
    let required_src = row_bytes
        .checked_mul(height)
        .ok_or(BufferError::SizeOverflow {
            what: "tight source size",
        })?;
    if src.len() < required_src {
        return Err(BufferError::InputTooSmall {
            required: required_src,
            have: src.len(),
        });
    }
    let required = stride
        .checked_mul(height - 1)
        .and_then(|prefix| prefix.checked_add(row_bytes))
        .ok_or(BufferError::SizeOverflow {
            what: "strided output size",
        })?;
    if out.len() < required {
        return Err(BufferError::OutputTooSmall {
            required,
            have: out.len(),
        });
    }

    for y in 0..dimensions.1 as usize {
        let src_row = &src[y * row_bytes..(y + 1) * row_bytes];
        let dst_start = y * stride;
        out[dst_start..dst_start + row_bytes].copy_from_slice(src_row);
    }

    Ok(())
}
