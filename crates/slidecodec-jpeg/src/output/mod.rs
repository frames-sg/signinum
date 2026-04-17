// SPDX-License-Identifier: Apache-2.0

//! Stride-aware output writers. One implementor per [`crate::OutputFormat`];
//! the decode loop is generic over `<W: OutputWriter>` and monomorphized at
//! each call site so there is no dynamic dispatch on the per-pixel hot path.

use crate::error::JpegError;

pub(crate) mod gray8;
pub(crate) mod rgb8;
pub(crate) mod rgba8;

pub(crate) use gray8::Gray8Writer;
pub(crate) use rgb8::Rgb8Writer;
pub(crate) use rgba8::Rgba8Writer;

/// A destination for decoded pixel rows. Each writer carries a mutable slice
/// of the caller's output buffer and the stride in bytes between rows.
pub(crate) trait OutputWriter {
    /// Write one full-width row of YCbCr data at output row `y`.
    fn write_ycbcr_row(&mut self, y: u32, y_row: &[u8], cb_row: &[u8], cr_row: &[u8]);

    /// Write one full-width row of grayscale data.
    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]);
}

/// Validate that the caller's `out`/`stride` pair is large enough to hold an
/// `image_width × image_height` image at `bytes_per_pixel`.
pub(crate) fn validate_buffer(
    out: &[u8],
    stride: usize,
    image_width: u32,
    image_height: u32,
    bytes_per_pixel: usize,
) -> Result<(), JpegError> {
    let row_bytes = (image_width as usize).checked_mul(bytes_per_pixel).ok_or(
        JpegError::OutputBufferTooSmall {
            required: usize::MAX,
            provided: out.len(),
        },
    )?;
    if stride < row_bytes {
        return Err(JpegError::InvalidStride {
            stride,
            row: row_bytes,
        });
    }
    let last_row_start = (image_height as usize)
        .saturating_sub(1)
        .checked_mul(stride)
        .ok_or(JpegError::OutputBufferTooSmall {
            required: usize::MAX,
            provided: out.len(),
        })?;
    let required = last_row_start + row_bytes;
    if out.len() < required {
        return Err(JpegError::OutputBufferTooSmall {
            required,
            provided: out.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_buffer_accepts_tight_fit() {
        let out = alloc::vec![0u8; 16 * 16 * 3];
        validate_buffer(&out, 16 * 3, 16, 16, 3).unwrap();
    }

    #[test]
    fn validates_buffer_accepts_padded_stride() {
        let out = alloc::vec![0u8; 16 * 64];
        validate_buffer(&out, 64, 16, 16, 3).unwrap();
    }

    #[test]
    fn validates_buffer_rejects_stride_less_than_row_width() {
        let out = alloc::vec![0u8; 16 * 16 * 3];
        let err = validate_buffer(&out, 16, 16, 16, 3).unwrap_err();
        assert!(matches!(err, JpegError::InvalidStride { .. }));
    }

    #[test]
    fn validates_buffer_rejects_undersized_output() {
        let out = alloc::vec![0u8; 10];
        let err = validate_buffer(&out, 16 * 3, 16, 16, 3).unwrap_err();
        assert!(matches!(err, JpegError::OutputBufferTooSmall { .. }));
    }
}
