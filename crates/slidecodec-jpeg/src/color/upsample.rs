// SPDX-License-Identifier: Apache-2.0

//! Chroma upsamplers. Three shapes this milestone supports:
//!
//! - **`upsample_1x1`** (4:4:4): identity copy; no resampling.
//! - **`upsample_h2v1_fancy`** (4:2:2): 2× horizontal triangle filter.
//! - **`upsample_h2v2_fancy`** (4:2:0): 2× horizontal + 2× vertical triangle
//!   filter (the libjpeg-turbo default for 4:2:0).
//!
//! The "fancy" name is libjpeg-turbo's; the filter weights are `(3, 1)` for
//! the two nearest chroma samples. At image edges the far sample is clamped
//! to the nearest (replicate) so the filter always has valid taps.

#![allow(dead_code)]

/// Identity upsample: one output row is the input row unchanged. Output width
/// equals input width. Used for 4:4:4 where no upsample is needed.
pub(crate) fn upsample_1x1(input: &[u8], output: &mut [u8]) {
    let n = input.len().min(output.len());
    output[..n].copy_from_slice(&input[..n]);
}

/// Horizontal fancy upsample (4:2:2). `input_row` has length `input_cols`;
/// `output_row` must have length `2 * input_cols`.
pub(crate) fn upsample_h2v1_fancy(input_row: &[u8], output_row: &mut [u8]) {
    let n = input_row.len();
    assert_eq!(output_row.len(), n * 2, "output row must be 2× input width");
    if n == 0 {
        return;
    }
    if n == 1 {
        output_row[0] = input_row[0];
        output_row[1] = input_row[0];
        return;
    }
    output_row[0] = input_row[0];
    output_row[1] = ((3 * input_row[0] as u32 + input_row[1] as u32 + 2) / 4) as u8;
    for i in 1..n - 1 {
        let prev = input_row[i - 1] as u32;
        let curr = input_row[i] as u32;
        let next = input_row[i + 1] as u32;
        output_row[2 * i] = ((3 * curr + prev + 2) / 4) as u8;
        output_row[2 * i + 1] = ((3 * curr + next + 2) / 4) as u8;
    }
    let last = input_row[n - 1] as u32;
    let before = input_row[n - 2] as u32;
    output_row[2 * n - 2] = ((3 * last + before + 2) / 4) as u8;
    output_row[2 * n - 1] = input_row[n - 1];
}

/// Produce two output rows for 4:2:0 vertical+horizontal fancy upsample.
///
/// Matches libjpeg-turbo's `h2v2_fancy_upsample` in `jdsample.c` bit-for-bit:
/// alternating `+8`/`+7` rounding across the two output columns of each
/// chroma sample, with distinct formulas at the first and last column so
/// boundary pixels stay consistent with interior pixels under the same
/// 3:1 / 1:3 blend.
pub(crate) fn upsample_h2v2_fancy(
    prev: &[u8],
    curr: &[u8],
    next: &[u8],
    out_top: &mut [u8],
    out_bot: &mut [u8],
) {
    let n = curr.len();
    assert_eq!(prev.len(), n);
    assert_eq!(next.len(), n);
    assert_eq!(out_top.len(), 2 * n);
    assert_eq!(out_bot.len(), 2 * n);
    if n == 0 {
        return;
    }

    emit_h2v2_row(prev, curr, out_top);
    emit_h2v2_row(next, curr, out_bot);
}

/// Blend one "near" chroma row against `curr` using libjpeg-turbo's 3:1
/// vertical weighting, then run the 3:1 horizontal blend and emit one
/// upsampled luma row. `near` = chroma row above (for the top output) or
/// below (for the bottom output) the current chroma row.
fn emit_h2v2_row(near: &[u8], curr: &[u8], out: &mut [u8]) {
    let n = curr.len();
    // Column sums: `colsum[i] = 3 * curr[i] + near[i]`. libjpeg-turbo streams
    // these as `this/next/last` without materializing the whole array.
    let colsum = |i: usize| 3 * curr[i] as u32 + near[i] as u32;

    if n == 1 {
        // Degenerate edge: just replicate.
        let v = (4 * colsum(0) + 8) >> 4;
        out[0] = v as u8;
        out[1] = v as u8;
        return;
    }

    // First column: both output taps use only `colsum(0)` and `colsum(1)`.
    let mut this = colsum(0);
    let mut next = colsum(1);
    out[0] = ((this * 4 + 8) >> 4) as u8;
    out[1] = ((this * 3 + next + 7) >> 4) as u8;

    // General interior columns.
    for i in 1..n - 1 {
        let last = this;
        this = next;
        next = colsum(i + 1);
        out[2 * i] = ((this * 3 + last + 8) >> 4) as u8;
        out[2 * i + 1] = ((this * 3 + next + 7) >> 4) as u8;
    }

    // Last column: `this` currently holds `colsum(n-2)`, `next` holds `colsum(n-1)`.
    let last = this;
    this = next;
    out[2 * (n - 1)] = ((this * 3 + last + 8) >> 4) as u8;
    out[2 * (n - 1) + 1] = ((this * 4 + 7) >> 4) as u8;
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn upsample_1x1_is_memcpy() {
        let input = vec![1u8, 2, 3, 4];
        let mut output = vec![0u8; 4];
        upsample_1x1(&input, &mut output);
        assert_eq!(output, input);
    }

    #[test]
    fn h2v1_fancy_replicates_edges_and_interpolates_middle() {
        let input = vec![10u8, 20, 30, 40];
        let mut output = vec![0u8; 8];
        upsample_h2v1_fancy(&input, &mut output);
        assert_eq!(output[0], 10);
        assert_eq!(output[1], 13);
        assert_eq!(output[2], 18);
        assert_eq!(output[3], 23);
        assert_eq!(output[4], 28);
        assert_eq!(output[5], 33);
        assert_eq!(output[6], 38);
        assert_eq!(output[7], 40);
    }

    #[test]
    fn h2v1_fancy_handles_single_sample_row() {
        let input = vec![42u8];
        let mut output = vec![0u8; 2];
        upsample_h2v1_fancy(&input, &mut output);
        assert_eq!(output, vec![42, 42]);
    }

    #[test]
    fn h2v2_fancy_produces_uniform_output_for_uniform_input() {
        let row = vec![100u8; 4];
        let mut top = vec![0u8; 8];
        let mut bot = vec![0u8; 8];
        upsample_h2v2_fancy(&row, &row, &row, &mut top, &mut bot);
        assert!(top.iter().all(|&v| v == 100));
        assert!(bot.iter().all(|&v| v == 100));
    }

    #[test]
    fn h2v2_fancy_blends_toward_adjacent_row_asymmetrically() {
        let prev = vec![0u8; 2];
        let curr = vec![200u8; 2];
        let next = vec![200u8; 2];
        let mut top = vec![0u8; 4];
        let mut bot = vec![0u8; 4];
        upsample_h2v2_fancy(&prev, &curr, &next, &mut top, &mut bot);
        assert_eq!(top[0], 150);
        assert_eq!(bot[0], 200);
    }
}
