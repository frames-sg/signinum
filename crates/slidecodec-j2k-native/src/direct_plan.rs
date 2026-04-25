use alloc::vec::Vec;

use crate::{J2kRect, J2kWaveletTransform};

/// Hidden identifier for one device-owned grayscale coefficient band.
#[doc(hidden)]
pub type J2kDirectBandId = u32;

/// Hidden grayscale-only direct device-plan step for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub enum J2kDirectGrayscaleStep {
    /// Decode one classic J2K sub-band into a device-owned coefficient buffer.
    ClassicSubBand(J2kOwnedSubBandPlan),
    /// Decode one HTJ2K sub-band into a device-owned coefficient buffer.
    HtSubBand(HtOwnedSubBandPlan),
    /// Apply one single-decomposition IDWT level on device-owned buffers.
    Idwt(J2kDirectIdwtStep),
    /// Store the final component plane into an output plane buffer.
    Store(J2kDirectStoreStep),
}

/// Hidden grayscale-only direct device plan for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct J2kDirectGrayscalePlan {
    /// Final output dimensions.
    pub dimensions: (u32, u32),
    /// Final output bit depth.
    pub bit_depth: u8,
    /// Ordered execution steps for the direct device pipeline.
    pub steps: Vec<J2kDirectGrayscaleStep>,
}

/// Hidden RGB direct device plan for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct J2kDirectColorPlan {
    /// Final output dimensions.
    pub dimensions: (u32, u32),
    /// Final output bit depths for the first three color components.
    pub bit_depths: [u8; 3],
    /// Whether inverse MCT must be applied after component stores.
    pub mct: bool,
    /// Wavelet transform used by the codestream's color transform.
    pub transform: J2kWaveletTransform,
    /// Per-component direct plans. RGB plans currently contain exactly three components.
    pub component_plans: Vec<J2kDirectGrayscalePlan>,
}

/// Hidden owned classic J2K sub-band decode job.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct J2kOwnedSubBandPlan {
    /// Stable identifier for the decoded coefficient band produced by this step.
    pub band_id: J2kDirectBandId,
    /// Absolute sub-band rect in component coordinates.
    pub rect: J2kRect,
    /// Sub-band width in samples.
    pub width: u32,
    /// Sub-band height in samples.
    pub height: u32,
    /// Owned code-block jobs for this sub-band.
    pub jobs: Vec<J2kOwnedCodeBlockBatchJob>,
}

/// Hidden owned HTJ2K sub-band decode job.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct HtOwnedSubBandPlan {
    /// Stable identifier for the decoded coefficient band produced by this step.
    pub band_id: J2kDirectBandId,
    /// Absolute sub-band rect in component coordinates.
    pub rect: J2kRect,
    /// Sub-band width in samples.
    pub width: u32,
    /// Sub-band height in samples.
    pub height: u32,
    /// Owned code-block jobs for this sub-band.
    pub jobs: Vec<HtOwnedCodeBlockBatchJob>,
}

/// Hidden owned classic J2K batched code-block decode job.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct J2kOwnedCodeBlockBatchJob {
    /// X offset within the target sub-band coefficient buffer.
    pub output_x: u32,
    /// Y offset within the target sub-band coefficient buffer.
    pub output_y: u32,
    /// Combined payload bytes for all coded segments in this code block.
    pub data: Vec<u8>,
    /// Coded segments for the code block.
    pub segments: Vec<crate::J2kCodeBlockSegment>,
    /// Code-block width in samples.
    pub width: u32,
    /// Code-block height in samples.
    pub height: u32,
    /// Output row stride, in samples, for the target sub-band storage.
    pub output_stride: usize,
    /// Missing most-significant bit planes for this code block.
    pub missing_bit_planes: u8,
    /// Number of coding passes present for this code block.
    pub number_of_coding_passes: u8,
    /// Total coded bitplanes for the parent sub-band.
    pub total_bitplanes: u8,
    /// The sub-band type containing this code block.
    pub sub_band_type: crate::J2kSubBandType,
    /// The code-block style flags.
    pub style: crate::J2kCodeBlockStyle,
    /// Whether strict decode validation is enabled for the parent image.
    pub strict: bool,
    /// Dequantization step to apply to decoded coefficients.
    pub dequantization_step: f32,
}

/// Hidden owned HTJ2K batched code-block decode job.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct HtOwnedCodeBlockBatchJob {
    /// X offset within the target sub-band coefficient buffer.
    pub output_x: u32,
    /// Y offset within the target sub-band coefficient buffer.
    pub output_y: u32,
    /// Combined cleanup/refinement bytes for the code block.
    pub data: Vec<u8>,
    /// Cleanup segment length in bytes.
    pub cleanup_length: u32,
    /// Refinement segment length in bytes.
    pub refinement_length: u32,
    /// Code-block width in samples.
    pub width: u32,
    /// Code-block height in samples.
    pub height: u32,
    /// Output row stride, in samples, for the target sub-band storage.
    pub output_stride: usize,
    /// Missing most-significant bit planes for this code block.
    pub missing_bit_planes: u8,
    /// Number of coding passes present for this code block.
    pub number_of_coding_passes: u8,
    /// Total coded bitplanes for the parent sub-band.
    pub num_bitplanes: u8,
    /// Whether vertically causal context was enabled.
    pub stripe_causal: bool,
    /// Whether strict decode validation is enabled for the parent image.
    pub strict: bool,
    /// Dequantization step to apply to decoded coefficients.
    pub dequantization_step: f32,
}

/// Hidden single grayscale IDWT step for a direct device plan.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct J2kDirectIdwtStep {
    /// Stable identifier of the output coefficient band produced by this step.
    pub output_band_id: J2kDirectBandId,
    /// Output rect of this decomposition level.
    pub rect: J2kRect,
    /// Transform to apply.
    pub transform: J2kWaveletTransform,
    /// Stable identifier of the LL input band.
    pub ll_band_id: J2kDirectBandId,
    /// LL band rect.
    pub ll: J2kRect,
    /// Stable identifier of the HL input band.
    pub hl_band_id: J2kDirectBandId,
    /// HL band rect.
    pub hl: J2kRect,
    /// Stable identifier of the LH input band.
    pub lh_band_id: J2kDirectBandId,
    /// LH band rect.
    pub lh: J2kRect,
    /// Stable identifier of the HH input band.
    pub hh_band_id: J2kDirectBandId,
    /// HH band rect.
    pub hh: J2kRect,
}

/// Hidden grayscale store step for a direct device plan.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct J2kDirectStoreStep {
    /// Stable identifier of the input coefficient band.
    pub input_band_id: J2kDirectBandId,
    /// Source rect of the input plane.
    pub input_rect: J2kRect,
    /// Source x offset to begin copying from.
    pub source_x: u32,
    /// Source y offset to begin copying from.
    pub source_y: u32,
    /// Number of samples to copy per row.
    pub copy_width: u32,
    /// Number of rows to copy.
    pub copy_height: u32,
    /// Destination row width.
    pub output_width: u32,
    /// Destination height.
    pub output_height: u32,
    /// Destination x offset to begin writing at.
    pub output_x: u32,
    /// Destination y offset to begin writing at.
    pub output_y: u32,
    /// Constant value added to every copied sample.
    pub addend: f32,
}
