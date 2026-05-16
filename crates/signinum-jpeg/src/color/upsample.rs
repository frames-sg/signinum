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

/// Identity upsample: one output row is the input row unchanged. Output width
/// equals input width. Used for 4:4:4 where no upsample is needed.
pub(crate) fn upsample_1x1(input: &[u8], output: &mut [u8]) {
    let n = input.len().min(output.len());
    output[..n].copy_from_slice(&input[..n]);
}

/// Horizontal fancy upsample (4:2:2). `input_row` has length `input_cols`;
/// `output_row` must have length `2 * input_cols`.
#[cfg(test)]
pub(crate) fn upsample_h2v1_fancy(input_row: &[u8], output_row: &mut [u8]) {
    let n = input_row.len();
    assert_eq!(output_row.len(), n * 2, "output row must be 2× input width");
    upsample_h2v1_fancy_row(input_row, output_row.len(), output_row);
}

/// Horizontal fancy upsample that emits only the visible output width.
pub(crate) fn upsample_h2v1_fancy_row(
    input_row: &[u8],
    output_width: usize,
    output_row: &mut [u8],
) {
    let n = input_row.len();
    assert!(
        output_width <= output_row.len(),
        "output width must fit in the destination slice"
    );
    assert!(
        output_width <= n * 2,
        "visible width cannot exceed the full upsampled row"
    );
    if output_width == 0 || n == 0 {
        return;
    }
    if n == 1 {
        output_row[..output_width].fill(input_row[0]);
        return;
    }

    for (x, slot) in output_row.iter_mut().enumerate().take(output_width) {
        let sample = x / 2;
        *slot = match x {
            0 => input_row[0],
            _ if x == n * 2 - 1 => input_row[n - 1],
            _ if x.is_multiple_of(2) => {
                let prev = input_row[sample - 1] as u32;
                let curr = input_row[sample] as u32;
                ((3 * curr + prev + 2) / 4) as u8
            }
            _ => {
                let curr = input_row[sample] as u32;
                let next = input_row[sample + 1] as u32;
                ((3 * curr + next + 2) / 4) as u8
            }
        };
    }
}

/// Produce two output rows for 4:2:0 vertical+horizontal fancy upsample.
///
/// Matches libjpeg-turbo's `h2v2_fancy_upsample` in `jdsample.c` bit-for-bit:
/// alternating `+8`/`+7` rounding across the two output columns of each
/// chroma sample, with distinct formulas at the first and last column so
/// boundary pixels stay consistent with interior pixels under the same
/// 3:1 / 1:3 blend.
#[cfg(test)]
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

    upsample_h2v2_fancy_rows(prev, curr, next, 2 * n, out_top, out_bot);
}

/// Vertical+horizontal fancy upsample that emits only the visible output width.
pub(crate) fn upsample_h2v2_fancy_rows(
    prev: &[u8],
    curr: &[u8],
    next: &[u8],
    output_width: usize,
    out_top: &mut [u8],
    out_bot: &mut [u8],
) {
    let n = curr.len();
    assert_eq!(prev.len(), n);
    assert_eq!(next.len(), n);
    assert!(
        output_width <= out_top.len() && output_width <= out_bot.len(),
        "output width must fit in both destination rows"
    );
    assert!(
        output_width <= n * 2,
        "visible width cannot exceed the full upsampled rows"
    );
    if output_width == 0 || n == 0 {
        return;
    }

    emit_h2v2_row(prev, curr, output_width, out_top);
    emit_h2v2_row(next, curr, output_width, out_bot);
}

/// Emit one visible output row from a 4:2:0 fancy-upsampled chroma triple.
pub(crate) fn upsample_h2v2_fancy_row(
    prev: &[u8],
    curr: &[u8],
    next: &[u8],
    output_width: usize,
    output_is_bottom: bool,
    out: &mut [u8],
) {
    let near = if output_is_bottom { next } else { prev };
    emit_h2v2_row(near, curr, output_width, out);
}

/// Blend one "near" chroma row against `curr` using libjpeg-turbo's 3:1
/// vertical weighting, then run the 3:1 horizontal blend and emit one
/// upsampled luma row. `near` = chroma row above (for the top output) or
/// below (for the bottom output) the current chroma row.
fn emit_h2v2_row(near: &[u8], curr: &[u8], output_width: usize, out: &mut [u8]) {
    let n = curr.len();
    // Column sums: `colsum[i] = 3 * curr[i] + near[i]`. libjpeg-turbo streams
    // these as `this/next/last` without materializing the whole array.
    let colsum = |i: usize| 3 * curr[i] as u32 + near[i] as u32;

    if n == 1 {
        // Degenerate edge: just replicate.
        let v = (4 * colsum(0) + 8) >> 4;
        out[..output_width].fill(v as u8);
        return;
    }

    for (x, slot) in out.iter_mut().enumerate().take(output_width) {
        let sample = x / 2;
        let this = colsum(sample);
        *slot = match x {
            0 => ((this * 4 + 8) >> 4) as u8,
            _ if x == n * 2 - 1 => ((this * 4 + 7) >> 4) as u8,
            _ if x.is_multiple_of(2) => {
                let last = colsum(sample - 1);
                ((this * 3 + last + 8) >> 4) as u8
            }
            _ => {
                let next = colsum(sample + 1);
                ((this * 3 + next + 7) >> 4) as u8
            }
        };
    }
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

    #[test]
    fn h2v1_fancy_row_matches_truncated_full_output() {
        let input = vec![10u8, 20, 30, 40];
        let mut full = vec![0u8; 8];
        let mut truncated = vec![0u8; 7];
        upsample_h2v1_fancy(&input, &mut full);
        upsample_h2v1_fancy_row(&input, truncated.len(), &mut truncated);
        assert_eq!(truncated, full[..truncated.len()]);
    }

    #[test]
    fn h2v2_fancy_rows_match_truncated_full_output() {
        let prev = vec![0u8, 10, 20, 30];
        let curr = vec![40u8, 50, 60, 70];
        let next = vec![80u8, 90, 100, 110];
        let mut full_top = vec![0u8; 8];
        let mut full_bot = vec![0u8; 8];
        let mut truncated_top = vec![0u8; 7];
        let mut truncated_bot = vec![0u8; 7];

        upsample_h2v2_fancy(&prev, &curr, &next, &mut full_top, &mut full_bot);
        upsample_h2v2_fancy_rows(
            &prev,
            &curr,
            &next,
            truncated_top.len(),
            &mut truncated_top,
            &mut truncated_bot,
        );

        assert_eq!(truncated_top, full_top[..truncated_top.len()]);
        assert_eq!(truncated_bot, full_bot[..truncated_bot.len()]);
    }
}
