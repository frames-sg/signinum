/*!
Internal pure-Rust JPEG 2000 codec engine for `signinum`.

This module tree was imported from the `dicom-toolkit-jpeg2000` 0.5.0 crate
and adapted in-repo so `signinum-j2k` no longer depends on an external
production decoder crate.

`dicom-toolkit-jpeg2000` is the JPEG 2000 engine used by `dicom-toolkit-rs`.
It is a maintained fork of the original `hayro-jpeg2000` project with
DICOM-focused extensions, including native-bit-depth decode for 8/12/16-bit
images and pure-Rust JPEG 2000 encoding.

The crate can decode both raw JPEG 2000 codestreams (`.j2c`) and images wrapped
inside the JP2 container format. The decoder supports the vast majority of features
defined in the JPEG 2000 core coding system (ISO/IEC 15444-1) as well as some color
spaces from the extensions (ISO/IEC 15444-2). There are still some missing pieces
for some "obscure" features (for example support for progression order
changes in tile-parts), but the features that commonly appear in real-world
images are supported.

The crate offers both a high-level 8-bit decode path for general image use and
a native-bit-depth decode path for integrations such as DICOM, plus encoder APIs
for emitting raw JPEG 2000 and HTJ2K codestreams.

# Example
```rust,no_run
use signinum_j2k_native::{DecodeSettings, Image};

let data = std::fs::read("image.jp2").unwrap();
let image = Image::new(&data, &DecodeSettings::default()).unwrap();

println!(
    "{}x{} image in {:?} with alpha={}",
    image.width(),
    image.height(),
    image.color_space(),
    image.has_alpha(),
);

let bitmap = image.decode().unwrap();
```

If you want to see a more comprehensive example, please take a look
at the example in [GitHub](https://github.com/knopkem/dicom-toolkit-rs/blob/main/crates/dicom-toolkit-jpeg2000/examples/png.rs),
which shows the main steps needed to convert a JPEG 2000 image into PNG.

# Testing
The decoder has been tested against 20.000+ images scraped from random PDFs
on the internet and also passes a large part of the `OpenJPEG` test suite. So you
can expect the crate to perform decently in terms of decoding correctness.

# Performance
A decent amount of effort has already been put into optimizing this crate
(both in terms of raw performance but also memory allocations). However, there
are some more important optimizations that have not been implemented yet, so
there is definitely still room for improvement (and I am planning on implementing
them eventually).

Overall, you should expect this crate to have worse performance than `OpenJPEG`,
but the difference gap should not be too large.

# Safety
By default, the crate has the `simd` feature enabled, which uses the
[`fearless_simd`](https://github.com/linebender/fearless_simd) crate to accelerate
important parts of the pipeline. If you want to eliminate any usage of unsafe
in this crate as well as its dependencies, you can simply disable this
feature, at the cost of worse decoding performance. Unsafe code is forbidden
via a crate-level attribute.

The crate is `no_std` compatible but requires an allocator to be available.
*/

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![forbid(missing_docs)]
#![allow(clippy::too_many_arguments)]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::error::{bail, err};
use crate::j2c::{ComponentData, Header};
use crate::jp2::cdef::{ChannelAssociation, ChannelType};
use crate::jp2::cmap::ComponentMappingType;
use crate::jp2::colr::{CieLab, EnumeratedColorspace};
use crate::jp2::icc::ICCMetadata;
use crate::jp2::{DecodedImage, ImageBoxes};

pub mod error;
#[macro_use]
pub(crate) mod log;
mod direct_plan;
pub(crate) mod math;
pub(crate) mod writer;

use crate::math::{dispatch, f32x8, Level, Simd, SIMD_WIDTH};
#[doc(hidden)]
pub use direct_plan::{
    HtOwnedCodeBlockBatchJob, HtOwnedSubBandPlan, J2kDirectBandId, J2kDirectColorPlan,
    J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kDirectIdwtStep, J2kDirectStoreStep,
    J2kOwnedCodeBlockBatchJob, J2kOwnedSubBandPlan,
};
pub use error::{
    ColorError, DecodeError, DecodingError, FormatError, MarkerError, Result, TileError,
    ValidationError,
};
pub use j2c::encode::{
    encode, encode_htj2k, encode_with_accelerator, EncodeOptions, EncodeProgressionOrder,
};
pub use j2c::{CpuDecodeParallelism, DecoderContext};

mod j2c;
mod jp2;
pub(crate) mod reader;
#[doc(hidden)]
pub use j2c::ht_encode_tables::HtUvlcTableEntry;

/// Hidden HTJ2K code-block job description for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct HtCodeBlockDecodeJob<'a> {
    /// Combined cleanup/refinement bytes for the code block.
    pub data: &'a [u8],
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

/// Hidden HTJ2K batched code-block decode job for one sub-band.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct HtCodeBlockBatchJob<'a> {
    /// X offset within the target sub-band coefficient buffer.
    pub output_x: u32,
    /// Y offset within the target sub-band coefficient buffer.
    pub output_y: u32,
    /// The actual code-block decode parameters.
    pub code_block: HtCodeBlockDecodeJob<'a>,
}

/// Hidden HTJ2K batched sub-band decode request for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct HtSubBandDecodeJob<'a> {
    /// Sub-band width in samples.
    pub width: u32,
    /// Sub-band height in samples.
    pub height: u32,
    /// Code blocks to decode into this sub-band.
    pub jobs: &'a [HtCodeBlockBatchJob<'a>],
}

/// Hidden classic J2K sub-band kind for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum J2kSubBandType {
    /// Low-low sub-band.
    LowLow,
    /// High-low sub-band.
    HighLow,
    /// Low-high sub-band.
    LowHigh,
    /// High-high sub-band.
    HighHigh,
}

/// Hidden classic J2K code-block style for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct J2kCodeBlockStyle {
    /// Selective arithmetic coding bypass was enabled.
    pub selective_arithmetic_coding_bypass: bool,
    /// Context probabilities reset after each pass.
    pub reset_context_probabilities: bool,
    /// Coding terminated after each pass.
    pub termination_on_each_pass: bool,
    /// Vertically causal context was enabled.
    pub vertically_causal_context: bool,
    /// Segmentation symbols were enabled.
    pub segmentation_symbols: bool,
}

/// Hidden classic J2K coded segment for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct J2kCodeBlockSegment {
    /// Byte offset of this segment within the combined payload.
    pub data_offset: u32,
    /// Segment payload length in bytes.
    pub data_length: u32,
    /// First coding pass covered by this segment.
    pub start_coding_pass: u8,
    /// One-past-last coding pass covered by this segment.
    pub end_coding_pass: u8,
    /// Whether this segment is decoded through the arithmetic path.
    pub use_arithmetic: bool,
}

/// Hidden classic J2K code-block job description for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct J2kCodeBlockDecodeJob<'a> {
    /// Combined payload bytes for all coded segments in this code block.
    pub data: &'a [u8],
    /// Coded segments for the code block.
    pub segments: &'a [J2kCodeBlockSegment],
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
    pub sub_band_type: J2kSubBandType,
    /// The code-block style flags.
    pub style: J2kCodeBlockStyle,
    /// Whether strict decode validation is enabled for the parent image.
    pub strict: bool,
    /// Dequantization step to apply to decoded coefficients.
    pub dequantization_step: f32,
}

/// Hidden encoded classic J2K code-block payload for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct EncodedJ2kCodeBlock {
    /// Combined payload bytes for all coded segments in this code block.
    pub data: Vec<u8>,
    /// Coded segments for the code block.
    pub segments: Vec<J2kCodeBlockSegment>,
    /// Number of coding passes present for this code block.
    pub number_of_coding_passes: u8,
    /// Missing most-significant bit planes for this code block.
    pub missing_bit_planes: u8,
}

/// Hidden encoded HTJ2K cleanup code-block payload for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct EncodedHtJ2kCodeBlock {
    /// Combined cleanup/refinement bytes for this code block.
    pub data: Vec<u8>,
    /// Number of coding passes present for this code block.
    pub num_coding_passes: u8,
    /// Number of zero most-significant bitplanes before first inclusion.
    pub num_zero_bitplanes: u8,
}

/// Hidden forward RCT job for backend experimentation.
#[doc(hidden)]
#[derive(Debug)]
pub struct J2kForwardRctJob<'a> {
    /// First component plane, updated in place.
    pub plane0: &'a mut [f32],
    /// Second component plane, updated in place.
    pub plane1: &'a mut [f32],
    /// Third component plane, updated in place.
    pub plane2: &'a mut [f32],
}

/// Hidden forward 5/3 DWT job for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct J2kForwardDwt53Job<'a> {
    /// Source samples in row-major order.
    pub samples: &'a [f32],
    /// Source width in samples.
    pub width: u32,
    /// Source height in samples.
    pub height: u32,
    /// Number of decomposition levels requested.
    pub num_levels: u8,
}

/// Hidden forward 5/3 DWT output for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct J2kForwardDwt53Output {
    /// LL subband coefficients from the lowest decomposition level.
    pub ll: Vec<f32>,
    /// LL subband width.
    pub ll_width: u32,
    /// LL subband height.
    pub ll_height: u32,
    /// Higher resolution detail levels, ordered from lowest to highest.
    pub levels: Vec<J2kForwardDwt53Level>,
}

/// Hidden forward 5/3 DWT detail level for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct J2kForwardDwt53Level {
    /// HL subband coefficients.
    pub hl: Vec<f32>,
    /// LH subband coefficients.
    pub lh: Vec<f32>,
    /// HH subband coefficients.
    pub hh: Vec<f32>,
    /// Full-resolution width represented by this level.
    pub width: u32,
    /// Full-resolution height represented by this level.
    pub height: u32,
    /// Low-pass width at this level.
    pub low_width: u32,
    /// Low-pass height at this level.
    pub low_height: u32,
    /// High-pass width at this level.
    pub high_width: u32,
    /// High-pass height at this level.
    pub high_height: u32,
}

/// Hidden Tier-1 classic J2K code-block encode job for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct J2kTier1CodeBlockEncodeJob<'a> {
    /// Quantized coefficients in row-major order.
    pub coefficients: &'a [i32],
    /// Code-block width in samples.
    pub width: u32,
    /// Code-block height in samples.
    pub height: u32,
    /// Subband kind containing this code-block.
    pub sub_band_type: J2kSubBandType,
    /// Total bitplanes for this subband/code-block.
    pub total_bitplanes: u8,
    /// Classic J2K code-block style flags.
    pub style: J2kCodeBlockStyle,
}

/// Hidden HTJ2K cleanup-only code-block encode job for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct J2kHtCodeBlockEncodeJob<'a> {
    /// Quantized coefficients in row-major order.
    pub coefficients: &'a [i32],
    /// Code-block width in samples.
    pub width: u32,
    /// Code-block height in samples.
    pub height: u32,
    /// Total bitplanes for this subband/code-block.
    pub total_bitplanes: u8,
}

/// Hidden LRCP packetization code-block contribution for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kPacketizationCodeBlock<'a> {
    /// Encoded Tier-1 bitstream bytes for this packet contribution.
    pub data: &'a [u8],
    /// Number of coding passes in this contribution.
    pub num_coding_passes: u8,
    /// Number of zero most-significant bitplanes before first inclusion.
    pub num_zero_bitplanes: u8,
    /// Whether this code-block was included in a previous packet.
    pub previously_included: bool,
    /// L-block value used for segment length coding.
    pub l_block: u32,
    /// Block coder used for this contribution.
    pub block_coding_mode: J2kPacketizationBlockCodingMode,
}

/// Hidden packetization block coding mode for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum J2kPacketizationBlockCodingMode {
    /// Classic JPEG 2000 Part 1 EBCOT block coding.
    Classic,
    /// High-throughput JPEG 2000 Part 15 block coding.
    HighThroughput,
}

/// Hidden packet progression order for backend packetization experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum J2kPacketizationProgressionOrder {
    /// Layer-resolution-component-position progression.
    Lrcp,
    /// Resolution-position-component-layer progression.
    Rpcl,
}

/// Hidden LRCP packetization subband precinct for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct J2kPacketizationSubband<'a> {
    /// Code-block contributions in row-major order.
    pub code_blocks: Vec<J2kPacketizationCodeBlock<'a>>,
    /// Number of code-blocks in the x direction.
    pub num_cbs_x: u32,
    /// Number of code-blocks in the y direction.
    pub num_cbs_y: u32,
}

/// Hidden LRCP packetization resolution packet for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct J2kPacketizationResolution<'a> {
    /// Subbands in packet order: LL for resolution 0, then HL/LH/HH.
    pub subbands: Vec<J2kPacketizationSubband<'a>>,
}

/// Hidden explicit packet descriptor for backend packetization experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kPacketizationPacketDescriptor {
    /// Index into the packet contribution array.
    pub packet_index: u32,
    /// Persistent packet-state index for repeated layer/precinct packets.
    pub state_index: u32,
    /// Quality layer for inclusion tag-tree thresholds.
    pub layer: u8,
    /// Resolution index in the output progression.
    pub resolution: u32,
    /// Component index in the output progression.
    pub component: u8,
    /// Precinct index in the output progression.
    pub precinct: u64,
}

/// Hidden LRCP packetization job for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kPacketizationEncodeJob<'a> {
    /// Number of resolution packets prepared for packetization.
    pub resolution_count: u32,
    /// Number of layers to write.
    pub num_layers: u8,
    /// Number of image components.
    pub num_components: u8,
    /// Total number of code-block contributions.
    pub code_block_count: u32,
    /// Packet progression order to emit.
    pub progression_order: J2kPacketizationProgressionOrder,
    /// Explicit packet descriptors in output progression order.
    pub packet_descriptors: &'a [J2kPacketizationPacketDescriptor],
    /// Packet payload prepared by Tier-1, in LRCP packet order.
    pub resolutions: &'a [J2kPacketizationResolution<'a>],
}

/// Hidden encode-stage dispatch counters for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct J2kEncodeDispatchReport {
    /// Forward RCT kernel dispatch count.
    pub forward_rct: usize,
    /// Forward reversible 5/3 DWT kernel dispatch count.
    pub forward_dwt53: usize,
    /// Tier-1 code-block encode dispatch count.
    pub tier1_code_block: usize,
    /// HTJ2K cleanup-only code-block encode dispatch count.
    pub ht_code_block: usize,
    /// Packetization dispatch count.
    pub packetization: usize,
}

impl J2kEncodeDispatchReport {
    /// Return the saturating per-stage delta from `before` to `self`.
    #[must_use]
    pub fn saturating_delta(self, before: Self) -> Self {
        Self {
            forward_rct: self.forward_rct.saturating_sub(before.forward_rct),
            forward_dwt53: self.forward_dwt53.saturating_sub(before.forward_dwt53),
            tier1_code_block: self
                .tier1_code_block
                .saturating_sub(before.tier1_code_block),
            ht_code_block: self.ht_code_block.saturating_sub(before.ht_code_block),
            packetization: self.packetization.saturating_sub(before.packetization),
        }
    }

    /// Return total dispatches across all encode stages.
    #[must_use]
    pub fn total(self) -> usize {
        self.forward_rct
            .saturating_add(self.forward_dwt53)
            .saturating_add(self.tier1_code_block)
            .saturating_add(self.ht_code_block)
            .saturating_add(self.packetization)
    }

    /// Return whether at least one encode stage dispatched.
    #[must_use]
    pub fn any(self) -> bool {
        self.total() > 0
    }
}

/// Hidden JPEG 2000 encode-stage accelerator for backend experimentation.
#[doc(hidden)]
pub trait J2kEncodeStageAccelerator {
    /// Report cumulative backend dispatches completed by this accelerator.
    fn dispatch_report(&self) -> J2kEncodeDispatchReport {
        J2kEncodeDispatchReport::default()
    }

    /// Optionally apply forward RCT in place.
    ///
    /// Return `Ok(true)` after writing transformed planes. Return `Ok(false)`
    /// to use the CPU fallback.
    fn encode_forward_rct(
        &mut self,
        _job: J2kForwardRctJob<'_>,
    ) -> core::result::Result<bool, &'static str> {
        Ok(false)
    }

    /// Optionally run a forward reversible 5/3 DWT.
    ///
    /// Return `Ok(Some(output))` with all subbands populated. Return
    /// `Ok(None)` to use the CPU fallback.
    fn encode_forward_dwt53(
        &mut self,
        _job: J2kForwardDwt53Job<'_>,
    ) -> core::result::Result<Option<J2kForwardDwt53Output>, &'static str> {
        Ok(None)
    }

    /// Optionally encode one classic Tier-1 code-block.
    ///
    /// Return `Ok(Some(output))` with encoded bytes and pass metadata. Return
    /// `Ok(None)` to use the CPU fallback.
    fn encode_tier1_code_block(
        &mut self,
        _job: J2kTier1CodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<EncodedJ2kCodeBlock>, &'static str> {
        Ok(None)
    }

    /// Optionally encode multiple classic Tier-1 code-blocks in one backend dispatch.
    ///
    /// Return `Ok(Some(outputs))` with one encoded output per input job. Return
    /// `Ok(None)` to use the per-block hook or CPU fallback.
    fn encode_tier1_code_blocks(
        &mut self,
        _jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
    ) -> core::result::Result<Option<Vec<EncodedJ2kCodeBlock>>, &'static str> {
        Ok(None)
    }

    /// Optionally encode one HTJ2K cleanup-only code-block.
    ///
    /// Return `Ok(Some(output))` with encoded bytes and pass metadata. Return
    /// `Ok(None)` to use the CPU fallback.
    fn encode_ht_code_block(
        &mut self,
        _job: J2kHtCodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<EncodedHtJ2kCodeBlock>, &'static str> {
        Ok(None)
    }

    /// Optionally encode multiple HTJ2K cleanup-only code-blocks in one backend dispatch.
    ///
    /// Return `Ok(Some(outputs))` with one encoded output per input job. Return
    /// `Ok(None)` to use the per-block hook or CPU fallback.
    fn encode_ht_code_blocks(
        &mut self,
        _jobs: &[J2kHtCodeBlockEncodeJob<'_>],
    ) -> core::result::Result<Option<Vec<EncodedHtJ2kCodeBlock>>, &'static str> {
        Ok(None)
    }

    /// Return whether native CPU code-block fallback should use internal rayon parallelism.
    ///
    /// External accelerators keep serial per-block fallback so their hooks still
    /// observe every fallback block after a declined batch hook.
    #[doc(hidden)]
    fn prefer_parallel_cpu_code_block_fallback(&self) -> bool {
        false
    }

    /// Optionally packetize prepared packet contributions.
    ///
    /// Return `Ok(Some(bytes))` with the complete tile bitstream. Return
    /// `Ok(None)` to use the CPU fallback.
    fn encode_packetization(
        &mut self,
        _job: J2kPacketizationEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        Ok(None)
    }
}

/// Hidden CPU-only encode accelerator that always falls back to native stages.
#[doc(hidden)]
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuOnlyJ2kEncodeStageAccelerator;

impl J2kEncodeStageAccelerator for CpuOnlyJ2kEncodeStageAccelerator {
    fn prefer_parallel_cpu_code_block_fallback(&self) -> bool {
        true
    }
}

/// Hidden classic J2K batched code-block decode job for one sub-band.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct J2kCodeBlockBatchJob<'a> {
    /// X offset within the target sub-band coefficient buffer.
    pub output_x: u32,
    /// Y offset within the target sub-band coefficient buffer.
    pub output_y: u32,
    /// The actual code-block decode parameters.
    pub code_block: J2kCodeBlockDecodeJob<'a>,
}

/// Hidden classic J2K batched sub-band decode request for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct J2kSubBandDecodeJob<'a> {
    /// Sub-band width in samples.
    pub width: u32,
    /// Sub-band height in samples.
    pub height: u32,
    /// Code blocks to decode into this sub-band.
    pub jobs: &'a [J2kCodeBlockBatchJob<'a>],
}

/// Hidden integer rectangle for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kRect {
    /// Inclusive minimum x coordinate.
    pub x0: u32,
    /// Inclusive minimum y coordinate.
    pub y0: u32,
    /// Exclusive maximum x coordinate.
    pub x1: u32,
    /// Exclusive maximum y coordinate.
    pub y1: u32,
}

impl J2kRect {
    /// Rectangle width in samples.
    pub fn width(self) -> u32 {
        self.x1.saturating_sub(self.x0)
    }

    /// Rectangle height in samples.
    pub fn height(self) -> u32 {
        self.y1.saturating_sub(self.y0)
    }
}

/// Hidden wavelet transform selector for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum J2kWaveletTransform {
    /// Reversible 5/3 transform.
    Reversible53,
    /// Irreversible 9/7 transform.
    Irreversible97,
}

/// Hidden single sub-band payload for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct J2kIdwtBand<'a> {
    /// Rect covered by this band.
    pub rect: J2kRect,
    /// Band coefficients in row-major order.
    pub coefficients: &'a [f32],
}

/// Hidden single-decomposition IDWT job for backend experimentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct J2kSingleDecompositionIdwtJob<'a> {
    /// Output rect of the decomposition level.
    pub rect: J2kRect,
    /// Transform to apply.
    pub transform: J2kWaveletTransform,
    /// LL band input.
    pub ll: J2kIdwtBand<'a>,
    /// HL band input.
    pub hl: J2kIdwtBand<'a>,
    /// LH band input.
    pub lh: J2kIdwtBand<'a>,
    /// HH band input.
    pub hh: J2kIdwtBand<'a>,
}

/// Hidden inverse MCT job for backend experimentation.
#[doc(hidden)]
#[derive(Debug)]
pub struct J2kInverseMctJob<'a> {
    /// Transform to apply.
    pub transform: J2kWaveletTransform,
    /// First component plane, updated in place.
    pub plane0: &'a mut [f32],
    /// Second component plane, updated in place.
    pub plane1: &'a mut [f32],
    /// Third component plane, updated in place.
    pub plane2: &'a mut [f32],
    /// Constant sign-shift addend applied to the first plane after inverse MCT.
    pub addend0: f32,
    /// Constant sign-shift addend applied to the second plane after inverse MCT.
    pub addend1: f32,
    /// Constant sign-shift addend applied to the third plane after inverse MCT.
    pub addend2: f32,
}

/// Hidden component-store job for backend experimentation.
#[doc(hidden)]
#[derive(Debug)]
pub struct J2kStoreComponentJob<'a> {
    /// Source IDWT coefficients in row-major order.
    pub input: &'a [f32],
    /// Source row width.
    pub input_width: u32,
    /// Source x offset to begin copying from.
    pub source_x: u32,
    /// Source y offset to begin copying from.
    pub source_y: u32,
    /// Number of samples to copy per row.
    pub copy_width: u32,
    /// Number of rows to copy.
    pub copy_height: u32,
    /// Destination component plane in row-major order.
    pub output: &'a mut [f32],
    /// Destination row width.
    pub output_width: u32,
    /// Destination x offset to begin writing at.
    pub output_x: u32,
    /// Destination y offset to begin writing at.
    pub output_y: u32,
    /// Constant value added to every copied sample.
    pub addend: f32,
}

/// Hidden HTJ2K code-block decode hook for backend experimentation.
#[doc(hidden)]
pub trait HtCodeBlockDecoder {
    /// Optionally decode a full classic J2K sub-band in one batch.
    ///
    /// Implementations should return `Ok(true)` if they handled the request and
    /// wrote the decoded coefficients into `output`. Returning `Ok(false)`
    /// falls back to per-code-block decode via `decode_j2k_code_block`.
    fn decode_j2k_sub_band(
        &mut self,
        _job: J2kSubBandDecodeJob<'_>,
        _output: &mut [f32],
    ) -> Result<bool> {
        Ok(false)
    }

    /// Optionally decode one classic J2K code block.
    ///
    /// Implementations should return `Ok(true)` if they handled the request
    /// and wrote the decoded coefficients into `output`. Returning `Ok(false)`
    /// falls back to the scalar bitplane decoder.
    fn decode_j2k_code_block(
        &mut self,
        _job: J2kCodeBlockDecodeJob<'_>,
        _output: &mut [f32],
    ) -> Result<bool> {
        Ok(false)
    }

    /// Optionally decode a full HTJ2K sub-band in one batch.
    ///
    /// Implementations should return `Ok(true)` if they handled the request and
    /// wrote the decoded coefficients into `output`. Returning `Ok(false)`
    /// falls back to per-code-block decode via `decode_code_block`.
    fn decode_sub_band(
        &mut self,
        _job: HtSubBandDecodeJob<'_>,
        _output: &mut [f32],
    ) -> Result<bool> {
        Ok(false)
    }

    /// Optionally decode one single-decomposition IDWT level on a backend.
    ///
    /// Implementations should return `Ok(true)` if they handled the request
    /// and wrote the transformed coefficients into `output`. Returning
    /// `Ok(false)` falls back to the scalar/SIMD CPU IDWT path.
    fn decode_single_decomposition_idwt(
        &mut self,
        _job: J2kSingleDecompositionIdwtJob<'_>,
        _output: &mut [f32],
    ) -> Result<bool> {
        Ok(false)
    }

    /// Optionally apply inverse MCT on a backend.
    ///
    /// Implementations should return `Ok(true)` if they handled the request
    /// and updated the component planes in place. Returning `Ok(false)` falls
    /// back to the scalar/SIMD CPU MCT path.
    fn decode_inverse_mct(&mut self, _job: J2kInverseMctJob<'_>) -> Result<bool> {
        Ok(false)
    }

    /// Optionally store one component plane on a backend.
    ///
    /// Implementations should return `Ok(true)` if they handled the request
    /// and updated the destination plane in place. Returning `Ok(false)` falls
    /// back to the CPU store path.
    fn decode_store_component(&mut self, _job: J2kStoreComponentJob<'_>) -> Result<bool> {
        Ok(false)
    }

    /// Decode one HTJ2K code block into `output`, writing `job.width` samples per row.
    fn decode_code_block(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> Result<()>;
}

fn internal_j2k_sub_band_type(sub_band_type: J2kSubBandType) -> j2c::build::SubBandType {
    match sub_band_type {
        J2kSubBandType::LowLow => j2c::build::SubBandType::LowLow,
        J2kSubBandType::HighLow => j2c::build::SubBandType::HighLow,
        J2kSubBandType::LowHigh => j2c::build::SubBandType::LowHigh,
        J2kSubBandType::HighHigh => j2c::build::SubBandType::HighHigh,
    }
}

fn internal_j2k_code_block_style(style: J2kCodeBlockStyle) -> j2c::codestream::CodeBlockStyle {
    j2c::codestream::CodeBlockStyle {
        selective_arithmetic_coding_bypass: style.selective_arithmetic_coding_bypass,
        reset_context_probabilities: style.reset_context_probabilities,
        termination_on_each_pass: style.termination_on_each_pass,
        vertically_causal_context: style.vertically_causal_context,
        segmentation_symbols: style.segmentation_symbols,
        high_throughput_block_coding: false,
    }
}

/// Hidden scalar classic J2K encoder helper for backend experimentation.
#[doc(hidden)]
pub fn encode_j2k_code_block_scalar_with_style(
    coefficients: &[i32],
    width: u32,
    height: u32,
    sub_band_type: J2kSubBandType,
    total_bitplanes: u8,
    style: J2kCodeBlockStyle,
) -> core::result::Result<EncodedJ2kCodeBlock, &'static str> {
    let encoded = j2c::bitplane_encode::encode_code_block_segments_with_style(
        coefficients,
        width,
        height,
        internal_j2k_sub_band_type(sub_band_type),
        total_bitplanes,
        &internal_j2k_code_block_style(style),
    );
    let segments = encoded
        .segments
        .into_iter()
        .map(|segment| J2kCodeBlockSegment {
            data_offset: segment.data_offset,
            data_length: segment.data_length,
            start_coding_pass: segment.start_coding_pass,
            end_coding_pass: segment.end_coding_pass,
            use_arithmetic: segment.use_arithmetic,
        })
        .collect();

    Ok(EncodedJ2kCodeBlock {
        data: encoded.data,
        segments,
        number_of_coding_passes: encoded.num_coding_passes,
        missing_bit_planes: encoded.num_zero_bitplanes,
    })
}

/// Hidden scalar HTJ2K cleanup-only encoder helper for backend experimentation.
#[doc(hidden)]
pub fn encode_ht_code_block_scalar(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
) -> core::result::Result<EncodedHtJ2kCodeBlock, &'static str> {
    let encoded =
        j2c::ht_block_encode::encode_code_block(coefficients, width, height, total_bitplanes)?;
    Ok(EncodedHtJ2kCodeBlock {
        data: encoded.data,
        num_coding_passes: encoded.num_coding_passes,
        num_zero_bitplanes: encoded.num_zero_bitplanes,
    })
}

/// Hidden scalar Tier-2 packetization helper for backend experimentation.
#[doc(hidden)]
pub fn encode_j2k_packetization_scalar(
    job: J2kPacketizationEncodeJob<'_>,
) -> core::result::Result<Vec<u8>, &'static str> {
    let mut resolutions = job
        .resolutions
        .iter()
        .map(|resolution| j2c::packet_encode::ResolutionPacket {
            subbands: resolution
                .subbands
                .iter()
                .map(|subband| j2c::packet_encode::SubbandPrecinct {
                    code_blocks: subband
                        .code_blocks
                        .iter()
                        .map(|code_block| j2c::packet_encode::CodeBlockPacketData {
                            data: code_block.data.to_vec(),
                            num_coding_passes: code_block.num_coding_passes,
                            num_zero_bitplanes: code_block.num_zero_bitplanes,
                            previously_included: code_block.previously_included,
                            l_block: code_block.l_block,
                            block_coding_mode: match code_block.block_coding_mode {
                                J2kPacketizationBlockCodingMode::Classic => {
                                    j2c::codestream_write::BlockCodingMode::Classic
                                }
                                J2kPacketizationBlockCodingMode::HighThroughput => {
                                    j2c::codestream_write::BlockCodingMode::HighThroughput
                                }
                            },
                        })
                        .collect(),
                    num_cbs_x: subband.num_cbs_x,
                    num_cbs_y: subband.num_cbs_y,
                })
                .collect(),
        })
        .collect::<Vec<_>>();

    let descriptors = job
        .packet_descriptors
        .iter()
        .map(|descriptor| j2c::packet_encode::PacketDescriptor {
            packet_index: descriptor.packet_index,
            state_index: descriptor.state_index,
            layer: descriptor.layer,
            resolution: descriptor.resolution,
            component: descriptor.component,
            precinct: descriptor.precinct,
        })
        .collect::<Vec<_>>();

    if descriptors.is_empty() {
        Ok(j2c::packet_encode::form_tile_bitstream_for_progression(
            &mut resolutions,
            job.num_layers,
            job.num_components,
            job.progression_order,
        ))
    } else {
        j2c::packet_encode::form_tile_bitstream_with_descriptors(&mut resolutions, &descriptors)
    }
}

/// Hidden scalar classic J2K decoder helper for backend experimentation.
#[doc(hidden)]
pub fn decode_j2k_code_block_scalar(
    job: J2kCodeBlockDecodeJob<'_>,
    output: &mut [f32],
) -> Result<()> {
    let required_len = if job.height == 0 {
        0
    } else {
        job.output_stride
            .checked_mul(job.height as usize - 1)
            .and_then(|prefix| prefix.checked_add(job.width as usize))
            .ok_or(DecodingError::CodeBlockDecodeFailure)?
    };
    if output.len() < required_len {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }

    let style = internal_j2k_code_block_style(job.style);
    let sub_band_type = internal_j2k_sub_band_type(job.sub_band_type);
    let code_block_stride =
        usize::try_from(job.width).map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
    let mut bit_plane_decode_context = j2c::bitplane::BitPlaneDecodeContext::default();

    j2c::bitplane::decode_code_block_segments_validated(
        job.data,
        job.segments,
        job.width,
        job.height,
        job.missing_bit_planes,
        job.number_of_coding_passes,
        job.total_bitplanes,
        sub_band_type,
        &style,
        job.strict,
        &mut bit_plane_decode_context,
    )?;

    for (row_idx, coeff_row) in bit_plane_decode_context
        .coefficient_rows()
        .enumerate()
        .take(job.height as usize)
    {
        let row_start = row_idx * job.output_stride;
        let output_row = &mut output[row_start..row_start + code_block_stride];
        for (coefficient, sample) in coeff_row.iter().zip(output_row.iter_mut()) {
            *sample = coefficient.get() as f32 * job.dequantization_step;
        }
    }

    Ok(())
}

/// Hidden scalar classic J2K batched decoder helper for backend experimentation.
#[doc(hidden)]
pub fn decode_j2k_sub_band_scalar(job: J2kSubBandDecodeJob<'_>, output: &mut [f32]) -> Result<()> {
    let required_len = if job.height == 0 {
        0
    } else {
        usize::try_from(job.width)
            .ok()
            .and_then(|width| width.checked_mul(job.height as usize))
            .ok_or(DecodingError::CodeBlockDecodeFailure)?
    };
    if output.len() < required_len {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }

    let sub_band_width =
        usize::try_from(job.width).map_err(|_| DecodingError::CodeBlockDecodeFailure)?;

    for batch_job in job.jobs {
        let code_block = batch_job.code_block;
        if code_block.output_stride != sub_band_width {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
        if batch_job
            .output_x
            .checked_add(code_block.width)
            .is_none_or(|x1| x1 > job.width)
            || batch_job
                .output_y
                .checked_add(code_block.height)
                .is_none_or(|y1| y1 > job.height)
        {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }

        let base_idx = usize::try_from(batch_job.output_y)
            .ok()
            .and_then(|y| y.checked_mul(sub_band_width))
            .and_then(|row| row.checked_add(batch_job.output_x as usize))
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let block_output_len = if code_block.height == 0 {
            0
        } else {
            code_block
                .output_stride
                .checked_mul(code_block.height as usize - 1)
                .and_then(|prefix| prefix.checked_add(code_block.width as usize))
                .ok_or(DecodingError::CodeBlockDecodeFailure)?
        };
        let end_idx = base_idx
            .checked_add(block_output_len)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        if end_idx > output.len() {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }

        decode_j2k_code_block_scalar(code_block, &mut output[base_idx..end_idx])?;
    }

    Ok(())
}

/// Hidden scalar HTJ2K decoder helper for backend experimentation.
#[doc(hidden)]
pub fn decode_ht_code_block_scalar(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
) -> Result<()> {
    let required_len = if job.height == 0 {
        0
    } else {
        job.output_stride
            .checked_mul(job.height as usize - 1)
            .and_then(|prefix| prefix.checked_add(job.width as usize))
            .ok_or(DecodingError::CodeBlockDecodeFailure)?
    };
    if output.len() < required_len {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    let code_block_stride =
        usize::try_from(job.width).map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
    let code_block_len = code_block_stride
        .checked_mul(job.height as usize)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;

    let combined = j2c::ht_block_decode::CombinedCodeBlockData {
        data: job.data.to_vec(),
        cleanup_length: job.cleanup_length,
        refinement_length: job.refinement_length,
    };
    let mut coefficients = vec![0u32; code_block_len];
    j2c::ht_block_decode::decode_combined_validated(
        &combined,
        job.missing_bit_planes,
        job.num_bitplanes,
        job.number_of_coding_passes,
        job.stripe_causal,
        job.strict,
        &mut coefficients,
        job.width,
        job.height,
        job.width,
    )?;

    for (row_idx, coeff_row) in coefficients
        .chunks_exact(code_block_stride)
        .enumerate()
        .take(job.height as usize)
    {
        let row_start = row_idx * job.output_stride;
        let output_row = &mut output[row_start..row_start + code_block_stride];
        for (coefficient, sample) in coeff_row.iter().copied().zip(output_row.iter_mut()) {
            *sample = j2c::ht_block_decode::coefficient_to_i32(coefficient, job.num_bitplanes)
                as f32
                * job.dequantization_step;
        }
    }

    Ok(())
}

/// Hidden HTJ2K VLC table 0 for backend experimentation.
#[doc(hidden)]
pub fn ht_vlc_table0() -> &'static [u16; 1024] {
    &j2c::ht_tables::VLC_TABLE0
}

/// Hidden HTJ2K VLC table 1 for backend experimentation.
#[doc(hidden)]
pub fn ht_vlc_table1() -> &'static [u16; 1024] {
    &j2c::ht_tables::VLC_TABLE1
}

/// Hidden HTJ2K UVLC table 0 for backend experimentation.
#[doc(hidden)]
pub fn ht_uvlc_table0() -> &'static [u16; 320] {
    &j2c::ht_tables::UVLC_TABLE0
}

/// Hidden HTJ2K UVLC table 1 for backend experimentation.
#[doc(hidden)]
pub fn ht_uvlc_table1() -> &'static [u16; 256] {
    &j2c::ht_tables::UVLC_TABLE1
}

/// Hidden HTJ2K cleanup encoder VLC table 0 for backend experimentation.
#[doc(hidden)]
pub fn ht_vlc_encode_table0() -> &'static [u16; 2048] {
    &j2c::ht_encode_tables::HT_VLC_ENCODE_TABLE0
}

/// Hidden HTJ2K cleanup encoder VLC table 1 for backend experimentation.
#[doc(hidden)]
pub fn ht_vlc_encode_table1() -> &'static [u16; 2048] {
    &j2c::ht_encode_tables::HT_VLC_ENCODE_TABLE1
}

/// Hidden HTJ2K cleanup encoder UVLC table for backend experimentation.
#[doc(hidden)]
pub fn ht_uvlc_encode_table() -> &'static [HtUvlcTableEntry; 75] {
    &j2c::ht_encode_tables::HT_UVLC_ENCODE_TABLE
}

/// JP2 signature box: 00 00 00 0C 6A 50 20 20
pub(crate) const JP2_MAGIC: &[u8] = b"\x00\x00\x00\x0C\x6A\x50\x20\x20";
/// Codestream signature: FF 4F FF 51 (SOC + SIZ markers)
pub(crate) const CODESTREAM_MAGIC: &[u8] = b"\xFF\x4F\xFF\x51";

/// Settings to apply during decoding.
#[derive(Debug, Copy, Clone)]
pub struct DecodeSettings {
    /// Whether palette indices should be resolved.
    ///
    /// JPEG2000 images can be stored in two different ways. First, by storing
    /// RGB values (depending on the color space) for each pixel. Secondly, by
    /// only storing a single index for each channel, and then resolving the
    /// actual color using the index.
    ///
    /// If you disable this option, in case you have an image with palette
    /// indices, they will not be resolved, but instead a grayscale image
    /// will be returned, with each pixel value corresponding to the palette
    /// index of the location.
    pub resolve_palette_indices: bool,
    /// Whether strict mode should be enabled when decoding.
    ///
    /// It is recommended to leave this flag disabled, unless you have a
    /// specific reason not to.
    pub strict: bool,
    /// A hint for the target resolution that the image should be decoded at.
    pub target_resolution: Option<(u32, u32)>,
}

impl Default for DecodeSettings {
    fn default() -> Self {
        Self {
            resolve_palette_indices: true,
            strict: false,
            target_resolution: None,
        }
    }
}

/// A JPEG2000 image or codestream.
pub struct Image<'a> {
    /// The tile-part payload used by the legacy JPEG 2000 decoder.
    pub(crate) codestream: &'a [u8],
    /// The header of the J2C codestream.
    pub(crate) header: Header<'a>,
    /// The JP2 boxes of the image. In the case of a raw codestream, we
    /// will synthesize the necessary boxes.
    pub(crate) boxes: ImageBoxes,
    /// Settings that should be applied during decoding.
    pub(crate) settings: DecodeSettings,
    /// Whether the image has an alpha channel.
    pub(crate) has_alpha: bool,
    /// The color space of the image.
    pub(crate) color_space: ColorSpace,
}

impl<'a> Image<'a> {
    /// Try to create a new JPEG2000 image from the given data.
    pub fn new(data: &'a [u8], settings: &DecodeSettings) -> Result<Self> {
        if data.starts_with(JP2_MAGIC) {
            jp2::parse(data, *settings)
        } else if data.starts_with(CODESTREAM_MAGIC) {
            j2c::parse(data, settings)
        } else {
            err!(FormatError::InvalidSignature)
        }
    }

    /// Whether the image has an alpha channel.
    pub fn has_alpha(&self) -> bool {
        self.has_alpha
    }

    /// The color space of the image.
    pub fn color_space(&self) -> &ColorSpace {
        &self.color_space
    }

    /// The width of the image.
    pub fn width(&self) -> u32 {
        self.header.size_data.image_width()
    }

    /// The height of the image.
    pub fn height(&self) -> u32 {
        self.header.size_data.image_height()
    }

    /// The original bit depth of the image. You usually don't need to do anything
    /// with this parameter, it just exists for informational purposes.
    pub fn original_bit_depth(&self) -> u8 {
        // Note that this only works if all components have the same precision.
        self.header.component_infos[0].size_info.precision
    }

    /// Whether decode finishes with additional host-side component mutation or reordering.
    #[doc(hidden)]
    pub fn supports_direct_device_plane_reuse(&self) -> bool {
        if self.settings.resolve_palette_indices && self.boxes.palette.is_some() {
            return false;
        }
        if self.boxes.channel_definition.is_some() {
            return false;
        }
        !matches!(
            self.boxes
                .color_specification
                .as_ref()
                .map(|spec| &spec.color_space),
            Some(jp2::colr::ColorSpace::Enumerated(
                EnumeratedColorspace::Sycc | EnumeratedColorspace::CieLab(_)
            ))
        )
    }

    /// Decode the image and return its decoded result as a `Vec<u8>`, with each
    /// channel interleaved.
    pub fn decode(&self) -> Result<Vec<u8>> {
        let bitmap = self.decode_with_context(&mut DecoderContext::default())?;
        Ok(bitmap.data)
    }

    /// Decode the image and return its decoded result using a caller-provided
    /// decoder context so allocations can be reused across repeated decodes.
    pub fn decode_with_context(&self, decoder_context: &mut DecoderContext<'a>) -> Result<Bitmap> {
        let buffer_size = self.width() as usize
            * self.height() as usize
            * (self.color_space.num_channels() as usize + if self.has_alpha { 1 } else { 0 });
        let mut buf = vec![0; buffer_size];
        self.decode_into(&mut buf, decoder_context)?;

        Ok(Bitmap {
            color_space: self.color_space.clone(),
            data: buf,
            has_alpha: self.has_alpha,
            width: self.width(),
            height: self.height(),
            original_bit_depth: self.original_bit_depth(),
        })
    }

    /// Decode the image into borrowed component planes using a caller-provided
    /// decoder context so allocations can be reused across repeated decodes.
    pub fn decode_components_with_context<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
    ) -> Result<DecodedComponents<'ctx>> {
        let decoded_image = self.prepare_decoded_image(decoder_context)?;
        let planes = decoded_image
            .decoded_components
            .iter()
            .map(|component| ComponentPlane {
                samples: component.container.truncated(),
                bit_depth: component.bit_depth,
            })
            .collect();

        Ok(DecodedComponents {
            dimensions: (self.width(), self.height()),
            color_space: self.color_space.clone(),
            has_alpha: self.has_alpha,
            planes,
        })
    }

    /// Build a hidden grayscale direct device plan without materializing host component planes.
    #[doc(hidden)]
    pub fn build_direct_grayscale_plan_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<J2kDirectGrayscalePlan> {
        if !matches!(self.color_space, ColorSpace::Gray) || self.has_alpha {
            bail!(DecodingError::UnsupportedFeature(
                "direct grayscale plan only supports grayscale images without alpha"
            ));
        }

        j2c::build_direct_grayscale_plan(self.codestream, &self.header, decoder_context)
    }

    /// Build a hidden RGB direct device plan without materializing host component planes.
    #[doc(hidden)]
    pub fn build_direct_color_plan_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<J2kDirectColorPlan> {
        if !matches!(self.color_space, ColorSpace::RGB) || self.has_alpha {
            bail!(DecodingError::UnsupportedFeature(
                "direct color plan only supports RGB images without alpha"
            ));
        }

        j2c::build_direct_color_plan(self.codestream, &self.header, decoder_context)
    }

    /// Decode borrowed component planes while delegating HTJ2K code-block decode.
    #[doc(hidden)]
    pub fn decode_components_with_ht_decoder<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
        ht_decoder: &mut dyn HtCodeBlockDecoder,
    ) -> Result<DecodedComponents<'ctx>> {
        let decoded_image =
            self.prepare_decoded_image_with_ht_decoder(decoder_context, ht_decoder)?;
        let planes = decoded_image
            .decoded_components
            .iter()
            .map(|component| ComponentPlane {
                samples: component.container.truncated(),
                bit_depth: component.bit_depth,
            })
            .collect();

        Ok(DecodedComponents {
            dimensions: (self.width(), self.height()),
            color_space: self.color_space.clone(),
            has_alpha: self.has_alpha,
            planes,
        })
    }

    /// Decode borrowed component planes for a requested region while
    /// delegating code-block/transform stages through the hidden backend hook.
    #[doc(hidden)]
    pub fn decode_region_components_with_ht_decoder<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
        roi: (u32, u32, u32, u32),
        ht_decoder: &mut dyn HtCodeBlockDecoder,
    ) -> Result<DecodedComponents<'ctx>> {
        validate_roi((self.width(), self.height()), roi)?;
        let (_x, _y, width, height) = roi;
        let decoded_image = self.prepare_decoded_image_with_region_and_ht_decoder(
            decoder_context,
            Some(roi),
            Some(ht_decoder),
        )?;
        let planes = decoded_image
            .decoded_components
            .iter()
            .map(|component| ComponentPlane {
                samples: component.container.truncated(),
                bit_depth: component.bit_depth,
            })
            .collect();

        Ok(DecodedComponents {
            dimensions: (width, height),
            color_space: self.color_space.clone(),
            has_alpha: self.has_alpha,
            planes,
        })
    }

    /// Decode a region of the image and return it as an 8-bit interleaved bitmap.
    pub fn decode_region(&self, roi: (u32, u32, u32, u32)) -> Result<Bitmap> {
        self.decode_region_with_context(roi, &mut DecoderContext::default())
    }

    /// Decode a region of the image and return it as an 8-bit interleaved bitmap
    /// using a caller-provided decoder context.
    pub fn decode_region_with_context(
        &self,
        roi: (u32, u32, u32, u32),
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<Bitmap> {
        validate_roi((self.width(), self.height()), roi)?;
        let mut decoded_image =
            self.prepare_decoded_image_with_region(decoder_context, Some(roi))?;
        let (_x, _y, width, height) = roi;
        let channels =
            self.color_space.num_channels() as usize + if self.has_alpha { 1 } else { 0 };
        let mut data = vec![0; width as usize * height as usize * channels];
        interleave_and_convert_region(
            &mut decoded_image,
            width as usize,
            (0, 0, width, height),
            &mut data,
        );
        Ok(Bitmap {
            color_space: self.color_space.clone(),
            data,
            has_alpha: self.has_alpha,
            width,
            height,
            original_bit_depth: self.original_bit_depth(),
        })
    }

    /// Decode the image at native bit depth without scaling to 8-bit.
    ///
    /// For images with bit depth ≤ 8, returns pixel data as `Vec<u8>`.
    /// For images with bit depth > 8 (e.g., 12-bit or 16-bit), returns
    /// pixel data as little-endian `u16` values packed into `Vec<u8>`.
    ///
    /// This is essential for medical imaging (DICOM) where 12-bit and 16-bit
    /// images must preserve their full dynamic range.
    pub fn decode_native(&self) -> Result<RawBitmap> {
        let mut decoder_context = DecoderContext::default();
        self.decode_native_with_context(&mut decoder_context)
    }

    /// Decode a region of the image at native bit depth.
    pub fn decode_native_region(&self, roi: (u32, u32, u32, u32)) -> Result<RawBitmap> {
        self.decode_native_region_with_context(roi, &mut DecoderContext::default())
    }

    /// Decode the image at native bit depth using a caller-provided decoder
    /// context so allocations can be reused across repeated decodes.
    pub fn decode_native_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<RawBitmap> {
        self.decode_with_output_region(decoder_context, None)?;

        let components = &decoder_context.tile_decode_context.channel_data;
        let bit_depth = self.original_bit_depth();
        let num_components = components.len() as u8;
        let width = self.width();
        let height = self.height();
        let pixel_count = width as usize * height as usize;

        if bit_depth <= 8 {
            let max_val = ((1u32 << bit_depth) - 1) as f32;
            let mut data = Vec::with_capacity(pixel_count * num_components as usize);
            for i in 0..pixel_count {
                for component in components.iter() {
                    let v = math::round_f32(component.container.truncated()[i]);
                    let clamped = if v < 0.0 {
                        0.0
                    } else if v > max_val {
                        max_val
                    } else {
                        v
                    };
                    data.push(clamped as u8);
                }
            }
            Ok(RawBitmap {
                data,
                width,
                height,
                bit_depth,
                num_components,
                bytes_per_sample: 1,
            })
        } else {
            let max_val = ((1u32 << bit_depth) - 1) as f32;
            let mut data = Vec::with_capacity(pixel_count * num_components as usize * 2);
            for i in 0..pixel_count {
                for component in components.iter() {
                    let v = math::round_f32(component.container.truncated()[i]);
                    let clamped = if v < 0.0 {
                        0.0
                    } else if v > max_val {
                        max_val
                    } else {
                        v
                    };
                    let val = clamped as u16;
                    data.extend_from_slice(&val.to_le_bytes());
                }
            }
            Ok(RawBitmap {
                data,
                width,
                height,
                bit_depth,
                num_components,
                bytes_per_sample: 2,
            })
        }
    }

    /// Decode a region of the image at native bit depth using a caller-provided
    /// decoder context.
    pub fn decode_native_region_with_context(
        &self,
        roi: (u32, u32, u32, u32),
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<RawBitmap> {
        validate_roi((self.width(), self.height()), roi)?;
        self.decode_with_output_region(decoder_context, Some(roi))?;

        let components = &decoder_context.tile_decode_context.channel_data;
        let bit_depth = self.original_bit_depth();
        let num_components = components.len() as u8;
        let bytes_per_sample = if bit_depth <= 8 { 1 } else { 2 };
        let (_x, _y, width, height) = roi;
        let mut data = Vec::with_capacity(
            width as usize * height as usize * num_components as usize * bytes_per_sample,
        );
        let max_val = ((1u32 << bit_depth) - 1) as f32;

        for row in 0..height as usize {
            for col in 0..width as usize {
                let idx = row * width as usize + col;
                for component in components {
                    let v = math::round_f32(component.container.truncated()[idx]);
                    let clamped = if v < 0.0 {
                        0.0
                    } else if v > max_val {
                        max_val
                    } else {
                        v
                    };
                    if bit_depth <= 8 {
                        data.push(clamped as u8);
                    } else {
                        data.extend_from_slice(&(clamped as u16).to_le_bytes());
                    }
                }
            }
        }

        Ok(RawBitmap {
            data,
            width,
            height,
            bit_depth,
            num_components,
            bytes_per_sample: bytes_per_sample as u8,
        })
    }

    /// Decode the image into the given buffer.
    ///
    /// This method does the same as [`Image::decode`], but you can provide
    /// a custom buffer for the output, as well as a decoder context. Doing
    /// so allows the internal decode engine to reuse memory allocations, so
    /// this is especially recommended if you plan on converting multiple
    /// images in the same session.
    ///
    /// The buffer must have the correct size.
    pub fn decode_into(
        &self,
        buf: &mut [u8],
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<()> {
        let mut decoded_image = self.prepare_decoded_image(decoder_context)?;
        interleave_and_convert(&mut decoded_image, buf);

        Ok(())
    }

    fn prepare_decoded_image<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
    ) -> Result<DecodedImage<'ctx>> {
        self.prepare_decoded_image_with_region(decoder_context, None)
    }

    fn prepare_decoded_image_with_ht_decoder<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
        ht_decoder: &mut dyn HtCodeBlockDecoder,
    ) -> Result<DecodedImage<'ctx>> {
        self.prepare_decoded_image_with_region_and_ht_decoder(
            decoder_context,
            None,
            Some(ht_decoder),
        )
    }

    fn prepare_decoded_image_with_region<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
        output_region: Option<(u32, u32, u32, u32)>,
    ) -> Result<DecodedImage<'ctx>> {
        self.prepare_decoded_image_with_region_and_ht_decoder(decoder_context, output_region, None)
    }

    fn prepare_decoded_image_with_region_and_ht_decoder<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
        output_region: Option<(u32, u32, u32, u32)>,
        ht_decoder: Option<&mut dyn HtCodeBlockDecoder>,
    ) -> Result<DecodedImage<'ctx>> {
        let settings = &self.settings;
        self.decode_with_output_region_and_ht_decoder(decoder_context, output_region, ht_decoder)?;
        let mut decoded_image = DecodedImage {
            decoded_components: &mut decoder_context.tile_decode_context.channel_data,
            boxes: self.boxes.clone(),
        };

        if settings.resolve_palette_indices {
            let components = core::mem::take(decoded_image.decoded_components);
            *decoded_image.decoded_components =
                resolve_palette_indices(components, &decoded_image.boxes)?;
        }

        if let Some(cdef) = &decoded_image.boxes.channel_definition {
            let mut components = decoded_image
                .decoded_components
                .iter()
                .cloned()
                .zip(
                    cdef.channel_definitions
                        .iter()
                        .map(|c| match c._association {
                            ChannelAssociation::WholeImage => u16::MAX,
                            ChannelAssociation::Colour(c) => c,
                        }),
                )
                .collect::<Vec<_>>();
            components.sort_by_key(|component| component.1);
            *decoded_image.decoded_components = components.into_iter().map(|c| c.0).collect();
        }

        let bit_depth = decoded_image.decoded_components[0].bit_depth;
        convert_color_space(&mut decoded_image, bit_depth)?;
        Ok(decoded_image)
    }

    fn decode_with_output_region(
        &self,
        decoder_context: &mut DecoderContext<'a>,
        output_region: Option<(u32, u32, u32, u32)>,
    ) -> Result<()> {
        self.decode_with_output_region_and_ht_decoder(decoder_context, output_region, None)
    }

    fn decode_with_output_region_and_ht_decoder(
        &self,
        decoder_context: &mut DecoderContext<'a>,
        output_region: Option<(u32, u32, u32, u32)>,
        mut ht_decoder: Option<&mut dyn HtCodeBlockDecoder>,
    ) -> Result<()> {
        decoder_context.set_output_region(output_region);
        let decode_result = j2c::decode(
            self.codestream,
            &self.header,
            decoder_context,
            &mut ht_decoder,
        );
        decoder_context.set_output_region(None);
        decode_result
    }
}

pub(crate) fn resolve_alpha_and_color_space(
    boxes: &ImageBoxes,
    header: &Header<'_>,
    settings: &DecodeSettings,
) -> Result<(ColorSpace, bool)> {
    let mut num_components = header.component_infos.len();

    // Override number of components with what is actually in the palette box
    // in case we resolve them.
    if settings.resolve_palette_indices {
        if let Some(palette_box) = &boxes.palette {
            num_components = palette_box.columns.len();
        }
    }

    let mut has_alpha = false;

    if let Some(cdef) = &boxes.channel_definition {
        let last = cdef.channel_definitions.last().unwrap();
        has_alpha = last.channel_type == ChannelType::Opacity;
    }

    let mut color_space = get_color_space(boxes, num_components)?;

    // If we didn't resolve palette indices, we need to assume grayscale image.
    if !settings.resolve_palette_indices && boxes.palette.is_some() {
        has_alpha = false;
        color_space = ColorSpace::Gray;
    }

    let actual_num_components = header.component_infos.len();

    // Validate the number of channels.
    if boxes.palette.is_none()
        && actual_num_components
            != (color_space.num_channels() + if has_alpha { 1 } else { 0 }) as usize
    {
        if !settings.strict
            && actual_num_components == color_space.num_channels() as usize + 1
            && !has_alpha
        {
            // See OPENJPEG test case orb-blue10-lin-j2k. Assume that we have an
            // alpha channel in this case.
            has_alpha = true;
        } else {
            // Color space is invalid, attempt to repair.
            if actual_num_components == 1 || (actual_num_components == 2 && has_alpha) {
                color_space = ColorSpace::Gray;
            } else if actual_num_components == 3 {
                color_space = ColorSpace::RGB;
            } else if actual_num_components == 4 {
                if has_alpha {
                    color_space = ColorSpace::RGB;
                } else {
                    color_space = ColorSpace::CMYK;
                }
            } else {
                bail!(ValidationError::TooManyChannels);
            }
        }
    }

    Ok((color_space, has_alpha))
}

/// The color space of the image.
#[derive(Debug, Clone)]
pub enum ColorSpace {
    /// A grayscale image.
    Gray,
    /// An RGB image.
    RGB,
    /// A CMYK image.
    CMYK,
    /// An unknown color space.
    Unknown {
        /// The number of channels of the color space.
        num_channels: u8,
    },
    /// An image based on an ICC profile.
    Icc {
        /// The raw data of the ICC profile.
        profile: Vec<u8>,
        /// The number of channels used by the ICC profile.
        num_channels: u8,
    },
}

impl ColorSpace {
    /// Return the number of expected channels for the color space.
    pub fn num_channels(&self) -> u8 {
        match self {
            Self::Gray => 1,
            Self::RGB => 3,
            Self::CMYK => 4,
            Self::Unknown { num_channels } => *num_channels,
            Self::Icc {
                num_channels: num_components,
                ..
            } => *num_components,
        }
    }
}

/// A bitmap storing the decoded result of the image.
pub struct Bitmap {
    /// The color space of the image.
    pub color_space: ColorSpace,
    /// The raw pixel data of the image. The result will always be in
    /// 8-bit (in case the original image had a different bit-depth, this
    /// decode path scales it to 8-bit).
    ///
    /// The size is guaranteed to equal
    /// `width * height * (num_channels + (if has_alpha { 1 } else { 0 })`.
    /// Pixels are interleaved on a per-channel basis, the alpha channel always
    /// appearing as the last channel, if available.
    pub data: Vec<u8>,
    /// Whether the image has an alpha channel.
    pub has_alpha: bool,
    /// The width of the image.
    pub width: u32,
    /// The height of the image.
    pub height: u32,
    /// The original bit depth of the image. You usually don't need to do anything
    /// with this parameter, it just exists for informational purposes.
    pub original_bit_depth: u8,
}

/// Raw decoded pixel data at native bit depth (no 8-bit scaling).
///
/// For bit depths ≤ 8, `data` contains one byte per sample.
/// For bit depths > 8 (e.g., 12 or 16), `data` contains two bytes per sample
/// in little-endian byte order (`u16` LE).
///
/// Samples are interleaved: for a 3-component image, the layout is
/// `[R0, G0, B0, R1, G1, B1, ...]`.
pub struct RawBitmap {
    /// The raw pixel data at native bit depth.
    pub data: Vec<u8>,
    /// The width of the image in pixels.
    pub width: u32,
    /// The height of the image in pixels.
    pub height: u32,
    /// The original bit depth per sample (e.g., 8, 12, 16).
    pub bit_depth: u8,
    /// The number of components (e.g., 1 for grayscale, 3 for RGB).
    pub num_components: u8,
    /// Bytes per sample: 1 for bit_depth ≤ 8, 2 for bit_depth > 8.
    pub bytes_per_sample: u8,
}

/// A borrowed decoded component plane.
pub struct ComponentPlane<'a> {
    samples: &'a [f32],
    bit_depth: u8,
}

impl<'a> ComponentPlane<'a> {
    /// Component samples in row-major order.
    pub fn samples(&self) -> &'a [f32] {
        self.samples
    }

    /// Bit depth of this component plane.
    pub fn bit_depth(&self) -> u8 {
        self.bit_depth
    }
}

/// Borrowed decoded component planes for an image.
pub struct DecodedComponents<'a> {
    dimensions: (u32, u32),
    color_space: ColorSpace,
    has_alpha: bool,
    planes: Vec<ComponentPlane<'a>>,
}

impl<'a> DecodedComponents<'a> {
    /// Dimensions of the decoded image represented by these planes.
    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    /// Color space after JPEG 2000 color conversion has been applied.
    pub fn color_space(&self) -> &ColorSpace {
        &self.color_space
    }

    /// Whether the decoded image has an alpha channel.
    pub fn has_alpha(&self) -> bool {
        self.has_alpha
    }

    /// Borrowed decoded component planes in display order.
    pub fn planes(&self) -> &[ComponentPlane<'a>] {
        &self.planes
    }
}

fn interleave_and_convert(image: &mut DecodedImage<'_>, buf: &mut [u8]) {
    let components = &mut *image.decoded_components;
    let num_components = components.len();

    let mut all_same_bit_depth = Some(components[0].bit_depth);

    for component in components.iter().skip(1) {
        if Some(component.bit_depth) != all_same_bit_depth {
            all_same_bit_depth = None;
        }
    }

    let max_len = components[0].container.truncated().len();

    let mut output_iter = buf.iter_mut();

    if all_same_bit_depth == Some(8) && num_components <= 4 {
        // Fast path for the common case.
        match num_components {
            // Gray-scale.
            1 => {
                for (output, input) in output_iter.zip(
                    components[0]
                        .container
                        .iter()
                        .map(|v| math::round_f32(*v) as u8),
                ) {
                    *output = input;
                }
            }
            // Gray-scale with alpha.
            2 => {
                let c0 = &components[0];
                let c1 = &components[1];

                let c0 = &c0.container[..max_len];
                let c1 = &c1.container[..max_len];

                for i in 0..max_len {
                    *output_iter.next().unwrap() = math::round_f32(c0[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c1[i]) as u8;
                }
            }
            // RGB
            3 => {
                let c0 = &components[0];
                let c1 = &components[1];
                let c2 = &components[2];

                let c0 = &c0.container[..max_len];
                let c1 = &c1.container[..max_len];
                let c2 = &c2.container[..max_len];

                for i in 0..max_len {
                    *output_iter.next().unwrap() = math::round_f32(c0[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c1[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c2[i]) as u8;
                }
            }
            // RGBA or CMYK.
            4 => {
                let c0 = &components[0];
                let c1 = &components[1];
                let c2 = &components[2];
                let c3 = &components[3];

                let c0 = &c0.container[..max_len];
                let c1 = &c1.container[..max_len];
                let c2 = &c2.container[..max_len];
                let c3 = &c3.container[..max_len];

                for i in 0..max_len {
                    *output_iter.next().unwrap() = math::round_f32(c0[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c1[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c2[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c3[i]) as u8;
                }
            }
            _ => unreachable!(),
        }
    } else {
        // Slow path that also requires us to scale to 8 bit.
        let mul_factor = ((1 << 8) - 1) as f32;

        for sample in 0..max_len {
            for channel in components.iter() {
                *output_iter.next().unwrap() = math::round_f32(
                    (channel.container[sample] / ((1_u32 << channel.bit_depth) - 1) as f32)
                        * mul_factor,
                ) as u8;
            }
        }
    }
}

fn interleave_and_convert_region(
    image: &mut DecodedImage<'_>,
    image_width: usize,
    roi: (u32, u32, u32, u32),
    buf: &mut [u8],
) {
    let components = &mut *image.decoded_components;
    let num_components = components.len();
    let (x, y, width, height) = roi;
    let mut output_iter = buf.iter_mut();

    let mut all_same_bit_depth = Some(components[0].bit_depth);
    for component in components.iter().skip(1) {
        if Some(component.bit_depth) != all_same_bit_depth {
            all_same_bit_depth = None;
        }
    }

    if all_same_bit_depth == Some(8) && num_components <= 4 {
        for row in y as usize..(y + height) as usize {
            let row_base = row * image_width;
            for col in x as usize..(x + width) as usize {
                let idx = row_base + col;
                for component in components.iter() {
                    *output_iter.next().unwrap() = math::round_f32(component.container[idx]) as u8;
                }
            }
        }
    } else {
        let mul_factor = ((1 << 8) - 1) as f32;
        for row in y as usize..(y + height) as usize {
            let row_base = row * image_width;
            for col in x as usize..(x + width) as usize {
                let idx = row_base + col;
                for component in components.iter() {
                    *output_iter.next().unwrap() = math::round_f32(
                        (component.container[idx] / ((1_u32 << component.bit_depth) - 1) as f32)
                            * mul_factor,
                    ) as u8;
                }
            }
        }
    }
}

fn validate_roi(dims: (u32, u32), roi: (u32, u32, u32, u32)) -> Result<()> {
    let (image_width, image_height) = dims;
    let (x, y, width, height) = roi;
    let x_end = x
        .checked_add(width)
        .ok_or(ValidationError::InvalidDimensions)?;
    let y_end = y
        .checked_add(height)
        .ok_or(ValidationError::InvalidDimensions)?;
    if x_end > image_width || y_end > image_height {
        return Err(ValidationError::InvalidDimensions.into());
    }
    Ok(())
}

fn convert_color_space(image: &mut DecodedImage<'_>, bit_depth: u8) -> Result<()> {
    if let Some(jp2::colr::ColorSpace::Enumerated(e)) = &image
        .boxes
        .color_specification
        .as_ref()
        .map(|i| &i.color_space)
    {
        match e {
            EnumeratedColorspace::Sycc => {
                dispatch!(Level::new(), simd => {
                    sycc_to_rgb(simd, image.decoded_components, bit_depth)
                })?;
            }
            EnumeratedColorspace::CieLab(cielab) => {
                dispatch!(Level::new(), simd => {
                    cielab_to_rgb(simd, image.decoded_components, bit_depth, cielab)
                })?;
            }
            _ => {}
        }
    }

    Ok(())
}

fn get_color_space(boxes: &ImageBoxes, num_components: usize) -> Result<ColorSpace> {
    let cs = match boxes
        .color_specification
        .as_ref()
        .map(|c| &c.color_space)
        .unwrap_or(&jp2::colr::ColorSpace::Unknown)
    {
        jp2::colr::ColorSpace::Enumerated(e) => {
            match e {
                EnumeratedColorspace::Cmyk => ColorSpace::CMYK,
                EnumeratedColorspace::Srgb => ColorSpace::RGB,
                EnumeratedColorspace::RommRgb => {
                    // Use an ICC profile to process the RommRGB color space.
                    ColorSpace::Icc {
                        profile: include_bytes!("../assets/ProPhoto-v2-micro.icc").to_vec(),
                        num_channels: 3,
                    }
                }
                EnumeratedColorspace::EsRgb => ColorSpace::RGB,
                EnumeratedColorspace::Greyscale => ColorSpace::Gray,
                EnumeratedColorspace::Sycc => ColorSpace::RGB,
                EnumeratedColorspace::CieLab(_) => ColorSpace::Icc {
                    profile: include_bytes!("../assets/LAB.icc").to_vec(),
                    num_channels: 3,
                },
                _ => bail!(FormatError::Unsupported),
            }
        }
        jp2::colr::ColorSpace::Icc(icc) => {
            if let Some(metadata) = ICCMetadata::from_data(icc) {
                ColorSpace::Icc {
                    profile: icc.clone(),
                    num_channels: metadata.color_space.num_components(),
                }
            } else {
                // See OPENJPEG test orb-blue10-lin-jp2.jp2. They seem to
                // assume RGB in this case (even though the image has 4
                // components with no opacity channel, they assume RGBA instead
                // of CMYK).
                ColorSpace::RGB
            }
        }
        jp2::colr::ColorSpace::Unknown => match num_components {
            1 => ColorSpace::Gray,
            3 => ColorSpace::RGB,
            4 => ColorSpace::CMYK,
            _ => ColorSpace::Unknown {
                num_channels: num_components as u8,
            },
        },
    };

    Ok(cs)
}

fn resolve_palette_indices(
    components: Vec<ComponentData>,
    boxes: &ImageBoxes,
) -> Result<Vec<ComponentData>> {
    let Some(palette) = boxes.palette.as_ref() else {
        // Nothing to resolve.
        return Ok(components);
    };

    let mapping = boxes.component_mapping.as_ref().unwrap();
    let mut resolved = Vec::with_capacity(mapping.entries.len());

    for entry in &mapping.entries {
        let component_idx = entry.component_index as usize;
        let component = components
            .get(component_idx)
            .ok_or(ColorError::PaletteResolutionFailed)?;

        match entry.mapping_type {
            ComponentMappingType::Direct => resolved.push(component.clone()),
            ComponentMappingType::Palette { column } => {
                let column_idx = column as usize;
                let column_info = palette
                    .columns
                    .get(column_idx)
                    .ok_or(ColorError::PaletteResolutionFailed)?;

                let mut mapped =
                    Vec::with_capacity(component.container.truncated().len() + SIMD_WIDTH);

                for &sample in component.container.truncated() {
                    let index = math::round_f32(sample) as i64;
                    let value = palette
                        .map(index as usize, column_idx)
                        .ok_or(ColorError::PaletteResolutionFailed)?;
                    mapped.push(value as f32);
                }

                resolved.push(ComponentData {
                    container: math::SimdBuffer::new(mapped),
                    bit_depth: column_info.bit_depth,
                });
            }
        }
    }

    Ok(resolved)
}

#[inline(always)]
fn cielab_to_rgb<S: Simd>(
    simd: S,
    components: &mut [ComponentData],
    bit_depth: u8,
    lab: &CieLab,
) -> Result<()> {
    let (head, _) = components
        .split_at_mut_checked(3)
        .ok_or(ColorError::LabConversionFailed)?;

    let [l, a, b] = head else {
        unreachable!();
    };

    let prec0 = l.bit_depth;
    let prec1 = a.bit_depth;
    let prec2 = b.bit_depth;

    // Prevent underflows/divisions by zero further below.
    if prec0 < 4 || prec1 < 4 || prec2 < 4 {
        bail!(ColorError::LabConversionFailed);
    }

    let rl = lab.rl.unwrap_or(100);
    let ra = lab.ra.unwrap_or(170);
    let rb = lab.ra.unwrap_or(200);
    let ol = lab.ol.unwrap_or(0);
    let oa = lab.oa.unwrap_or(1 << (bit_depth - 1));
    let ob = lab
        .ob
        .unwrap_or((1 << (bit_depth - 2)) + (1 << (bit_depth - 3)));

    // Copied from OpenJPEG.
    let min_l = -(rl as f32 * ol as f32) / ((1 << prec0) - 1) as f32;
    let max_l = min_l + rl as f32;
    let min_a = -(ra as f32 * oa as f32) / ((1 << prec1) - 1) as f32;
    let max_a = min_a + ra as f32;
    let min_b = -(rb as f32 * ob as f32) / ((1 << prec2) - 1) as f32;
    let max_b = min_b + rb as f32;

    let bit_max = (1_u32 << bit_depth) - 1;

    // Note that we are not doing the actual conversion with the ICC profile yet,
    // just decoding the raw LAB values.
    // We leave applying the ICC profile to the user.
    let divisor_l = ((1 << prec0) - 1) as f32;
    let divisor_a = ((1 << prec1) - 1) as f32;
    let divisor_b = ((1 << prec2) - 1) as f32;

    let scale_l_final = bit_max as f32 / 100.0;
    let scale_ab_final = bit_max as f32 / 255.0;

    let l_offset = min_l * scale_l_final;
    let l_scale = (max_l - min_l) / divisor_l * scale_l_final;
    let a_offset = (min_a + 128.0) * scale_ab_final;
    let a_scale = (max_a - min_a) / divisor_a * scale_ab_final;
    let b_offset = (min_b + 128.0) * scale_ab_final;
    let b_scale = (max_b - min_b) / divisor_b * scale_ab_final;

    let l_offset_v = f32x8::splat(simd, l_offset);
    let l_scale_v = f32x8::splat(simd, l_scale);
    let a_offset_v = f32x8::splat(simd, a_offset);
    let a_scale_v = f32x8::splat(simd, a_scale);
    let b_offset_v = f32x8::splat(simd, b_offset);
    let b_scale_v = f32x8::splat(simd, b_scale);

    // Note that we are not doing the actual conversion with the ICC profile yet,
    // just decoding the raw LAB values.
    // We leave applying the ICC profile to the user.
    for ((l_chunk, a_chunk), b_chunk) in l
        .container
        .chunks_exact_mut(SIMD_WIDTH)
        .zip(a.container.chunks_exact_mut(SIMD_WIDTH))
        .zip(b.container.chunks_exact_mut(SIMD_WIDTH))
    {
        let l_v = f32x8::from_slice(simd, l_chunk);
        let a_v = f32x8::from_slice(simd, a_chunk);
        let b_v = f32x8::from_slice(simd, b_chunk);

        l_v.mul_add(l_scale_v, l_offset_v).store(l_chunk);
        a_v.mul_add(a_scale_v, a_offset_v).store(a_chunk);
        b_v.mul_add(b_scale_v, b_offset_v).store(b_chunk);
    }

    Ok(())
}

#[inline(always)]
fn sycc_to_rgb<S: Simd>(simd: S, components: &mut [ComponentData], bit_depth: u8) -> Result<()> {
    let offset = (1_u32 << (bit_depth as u32 - 1)) as f32;
    let max_value = ((1_u32 << bit_depth as u32) - 1) as f32;

    let (head, _) = components
        .split_at_mut_checked(3)
        .ok_or(ColorError::SyccConversionFailed)?;

    let [y, cb, cr] = head else {
        unreachable!();
    };

    let offset_v = f32x8::splat(simd, offset);
    let max_v = f32x8::splat(simd, max_value);
    let zero_v = f32x8::splat(simd, 0.0);
    let cr_to_r = f32x8::splat(simd, 1.402);
    let cb_to_g = f32x8::splat(simd, -0.344136);
    let cr_to_g = f32x8::splat(simd, -0.714136);
    let cb_to_b = f32x8::splat(simd, 1.772);

    for ((y_chunk, cb_chunk), cr_chunk) in y
        .container
        .chunks_exact_mut(SIMD_WIDTH)
        .zip(cb.container.chunks_exact_mut(SIMD_WIDTH))
        .zip(cr.container.chunks_exact_mut(SIMD_WIDTH))
    {
        let y_v = f32x8::from_slice(simd, y_chunk);
        let cb_v = f32x8::from_slice(simd, cb_chunk) - offset_v;
        let cr_v = f32x8::from_slice(simd, cr_chunk) - offset_v;

        // r = y + 1.402 * cr
        let r = cr_v.mul_add(cr_to_r, y_v);
        // g = y - 0.344136 * cb - 0.714136 * cr
        let g = cr_v.mul_add(cr_to_g, cb_v.mul_add(cb_to_g, y_v));
        // b = y + 1.772 * cb
        let b = cb_v.mul_add(cb_to_b, y_v);

        r.min(max_v).max(zero_v).store(y_chunk);
        g.min(max_v).max(zero_v).store(cb_chunk);
        b.min(max_v).max(zero_v).store(cr_chunk);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct DecodeWorkCounter {
        classic_code_blocks: usize,
        ht_code_blocks: usize,
        idwt_output_samples: usize,
    }

    impl DecodeWorkCounter {
        fn code_blocks(&self) -> usize {
            self.classic_code_blocks + self.ht_code_blocks
        }
    }

    struct FailingHtDecoder {
        called: bool,
    }

    impl HtCodeBlockDecoder for FailingHtDecoder {
        fn decode_code_block(
            &mut self,
            _job: HtCodeBlockDecodeJob<'_>,
            _output: &mut [f32],
        ) -> Result<()> {
            self.called = true;
            Err(DecodingError::CodeBlockDecodeFailure.into())
        }
    }

    struct FailingClassicDecoder {
        called: bool,
    }

    impl HtCodeBlockDecoder for FailingClassicDecoder {
        fn decode_code_block(
            &mut self,
            _job: HtCodeBlockDecodeJob<'_>,
            _output: &mut [f32],
        ) -> Result<()> {
            panic!("HT hook must not be used for classic J2K test")
        }

        fn decode_j2k_code_block(
            &mut self,
            _job: J2kCodeBlockDecodeJob<'_>,
            _output: &mut [f32],
        ) -> Result<bool> {
            self.called = true;
            Err(DecodingError::CodeBlockDecodeFailure.into())
        }
    }

    struct FailingClassicBatchDecoder {
        called: bool,
    }

    impl HtCodeBlockDecoder for FailingClassicBatchDecoder {
        fn decode_code_block(
            &mut self,
            _job: HtCodeBlockDecodeJob<'_>,
            _output: &mut [f32],
        ) -> Result<()> {
            panic!("HT hook must not be used for classic J2K batch test")
        }

        fn decode_j2k_code_block(
            &mut self,
            _job: J2kCodeBlockDecodeJob<'_>,
            _output: &mut [f32],
        ) -> Result<bool> {
            panic!(
                "per-block classic hook must not be used when the batch hook handles the sub-band"
            )
        }

        fn decode_j2k_sub_band(
            &mut self,
            _job: J2kSubBandDecodeJob<'_>,
            _output: &mut [f32],
        ) -> Result<bool> {
            self.called = true;
            Err(DecodingError::CodeBlockDecodeFailure.into())
        }
    }

    fn fixture() -> Vec<u8> {
        let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        encode(&pixels, 2, 2, 3, 8, false, &options).expect("encode")
    }

    fn fixture_multi_block() -> Vec<u8> {
        let pixels: Vec<u8> = (0..64).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            code_block_width_exp: 0,
            code_block_height_exp: 0,
            ..EncodeOptions::default()
        };
        encode(&pixels, 8, 8, 1, 8, false, &options).expect("encode multi-block classic")
    }

    fn fixture_gray() -> Vec<u8> {
        let pixels: Vec<u8> = (0..16).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode classic gray8")
    }

    fn fixture_ht_gray() -> Vec<u8> {
        let pixels: Vec<u8> = (0..16).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        encode_htj2k(&pixels, 4, 4, 1, 8, false, &options).expect("encode ht gray8")
    }

    fn gradient_pixels(width: u32, height: u32, components: u8) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * components as usize);
        for y in 0..height {
            for x in 0..width {
                for component in 0..components {
                    pixels.push(((x * 3 + y * 5 + u32::from(component) * 41) & 0xff) as u8);
                }
            }
        }
        pixels
    }

    fn roi_fixture(classic: bool, components: u8) -> Vec<u8> {
        let width = 64;
        let height = 64;
        let pixels = gradient_pixels(width, height, components);
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            code_block_width_exp: 0,
            code_block_height_exp: 0,
            ..EncodeOptions::default()
        };
        if classic {
            encode(&pixels, width, height, components, 8, false, &options)
                .expect("encode ROI classic fixture")
        } else {
            encode_htj2k(&pixels, width, height, components, 8, false, &options)
                .expect("encode ROI HT fixture")
        }
    }

    fn crop_interleaved(
        full: &[u8],
        full_width: u32,
        channels: usize,
        roi: (u32, u32, u32, u32),
    ) -> Vec<u8> {
        let (x, y, width, height) = roi;
        let mut out = Vec::with_capacity(width as usize * height as usize * channels);
        let row_bytes = full_width as usize * channels;
        let roi_row_bytes = width as usize * channels;
        for row in y as usize..(y + height) as usize {
            let start = row * row_bytes + x as usize * channels;
            out.extend_from_slice(&full[start..start + roi_row_bytes]);
        }
        out
    }

    fn count_decode_work(bytes: &[u8], roi: Option<(u32, u32, u32, u32)>) -> DecodeWorkCounter {
        let image = Image::new(bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        match roi {
            Some(roi) => {
                image
                    .decode_region_with_context(roi, &mut context)
                    .expect("region decode with counter");
            }
            None => {
                image
                    .decode_with_context(&mut context)
                    .expect("full decode with counter");
            }
        }
        let counters = context.tile_decode_context.debug_counters;
        DecodeWorkCounter {
            classic_code_blocks: counters.decoded_code_blocks,
            ht_code_blocks: 0,
            idwt_output_samples: counters.idwt_output_samples,
        }
    }

    #[test]
    fn roi_decode_matches_full_crop_for_classic_and_htj2k_gray_and_rgb() {
        let cases = [
            (true, 1_u8, true, false),
            (true, 3_u8, false, false),
            (false, 1_u8, true, false),
            (false, 3_u8, false, false),
        ];
        let rois = [
            (20, 18, 17, 19),
            (0, 0, 9, 11),
            (63, 63, 1, 1),
            (7, 5, 13, 9),
            (0, 0, 64, 64),
        ];

        for (classic, components, expect_gray, has_alpha) in cases {
            let bytes = roi_fixture(classic, components);
            let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
            let full = image.decode().expect("full decode");
            let channels = components as usize;
            for roi in rois {
                let region = image.decode_region(roi).expect("region decode");
                assert_eq!(matches!(region.color_space, ColorSpace::Gray), expect_gray);
                assert_eq!(region.has_alpha, has_alpha);
                assert_eq!(
                    region.data,
                    crop_interleaved(&full, 64, channels, roi),
                    "classic={classic} components={components} roi={roi:?}"
                );
            }
        }
    }

    #[test]
    fn roi_decode_prunes_code_blocks_and_idwt_work_for_classic_and_htj2k() {
        let roi = (48, 48, 16, 16);
        for classic in [true, false] {
            let bytes = {
                let pixels = gradient_pixels(128, 128, 1);
                let options = EncodeOptions {
                    reversible: true,
                    num_decomposition_levels: 3,
                    code_block_width_exp: 0,
                    code_block_height_exp: 0,
                    ..EncodeOptions::default()
                };
                if classic {
                    encode(&pixels, 128, 128, 1, 8, false, &options)
                        .expect("encode classic work fixture")
                } else {
                    encode_htj2k(&pixels, 128, 128, 1, 8, false, &options)
                        .expect("encode ht work fixture")
                }
            };
            let full = count_decode_work(&bytes, None);
            let region = count_decode_work(&bytes, Some(roi));

            assert!(
                region.code_blocks() > 0 && region.code_blocks() < full.code_blocks(),
                "ROI should decode fewer code-blocks for classic={classic}; full={}, region={}",
                full.code_blocks(),
                region.code_blocks()
            );
            assert!(
                region.idwt_output_samples > 0
                    && region.idwt_output_samples < full.idwt_output_samples,
                "ROI should produce fewer IDWT output samples for classic={classic}; full={}, region={}",
                full.idwt_output_samples,
                region.idwt_output_samples
            );
        }
    }

    #[test]
    fn region_decode_reuses_region_sized_component_storage() {
        let bytes = fixture();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();

        let bitmap = image
            .decode_region_with_context((1, 0, 1, 2), &mut context)
            .expect("region decode");

        assert_eq!((bitmap.width, bitmap.height), (1, 2));
        assert!(context
            .tile_decode_context
            .channel_data
            .iter()
            .all(|component| component.container.truncated().len() == 2));
    }

    #[test]
    fn native_region_decode_reuses_region_sized_component_storage() {
        let bytes = fixture();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();

        let bitmap = image
            .decode_native_region_with_context((1, 0, 1, 2), &mut context)
            .expect("native region decode");

        assert_eq!((bitmap.width, bitmap.height), (1, 2));
        assert!(context
            .tile_decode_context
            .channel_data
            .iter()
            .all(|component| component.container.truncated().len() == 2));
    }

    #[test]
    fn decoder_context_defaults_to_auto_cpu_parallelism() {
        let context = DecoderContext::default();

        assert_eq!(context.cpu_decode_parallelism(), CpuDecodeParallelism::Auto);
    }

    #[test]
    fn classic_j2k_auto_and_serial_cpu_parallelism_match_pixels() {
        let bytes = fixture_multi_block();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut auto_context = DecoderContext::default();
        let mut serial_context = DecoderContext::default();
        serial_context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);

        let auto = image
            .decode_with_context(&mut auto_context)
            .expect("auto decode");
        let serial = image
            .decode_with_context(&mut serial_context)
            .expect("serial decode");

        assert_eq!(auto.data, serial.data);
    }

    #[test]
    fn serial_cpu_parallelism_disables_classic_sub_band_parallel_branch() {
        assert!(!j2c::should_decode_classic_sub_band_in_parallel(
            CpuDecodeParallelism::Serial,
            16
        ));
    }

    #[test]
    fn grayscale_direct_plan_is_built_without_materializing_channel_data() {
        let bytes = fixture_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();

        let plan = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("build direct plan");

        assert_eq!(plan.dimensions, (4, 4));
        assert_eq!(plan.bit_depth, 8);
        assert!(
            !plan.steps.is_empty(),
            "direct plan must contain executable steps"
        );
        assert!(
            plan.steps.iter().any(|step| matches!(
                step,
                J2kDirectGrayscaleStep::ClassicSubBand(plan) if !plan.jobs.is_empty()
            )),
            "classic J2K direct plan must contain at least one non-empty classic sub-band job"
        );
        assert!(
            context.tile_decode_context.channel_data.is_empty(),
            "building a direct plan must not materialize host component planes"
        );
    }

    #[test]
    fn grayscale_direct_plan_honors_target_resolution() {
        let bytes = fixture_ht_gray();
        let image = Image::new(
            &bytes,
            &DecodeSettings {
                target_resolution: Some((2, 2)),
                ..DecodeSettings::default()
            },
        )
        .expect("scaled image");
        let mut context = DecoderContext::default();

        let plan = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("build scaled direct plan");

        assert_eq!(plan.dimensions, (2, 2));
        assert!(plan.steps.iter().any(|step| matches!(
            step,
            J2kDirectGrayscaleStep::HtSubBand(plan) if !plan.jobs.is_empty()
        )));
        assert!(plan.steps.iter().any(|step| matches!(
            step,
            J2kDirectGrayscaleStep::Store(store)
                if store.output_width == 2 && store.output_height == 2
        )));
        assert!(
            context.tile_decode_context.channel_data.is_empty(),
            "building a scaled direct plan must not materialize host component planes"
        );
    }

    #[test]
    fn htj2k_grayscale_direct_plan_contains_ht_sub_band_steps() {
        let bytes = fixture_ht_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();

        let plan = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("build direct plan");

        assert!(
            plan.steps.iter().any(|step| matches!(
                step,
                J2kDirectGrayscaleStep::HtSubBand(plan) if !plan.jobs.is_empty()
            )),
            "HTJ2K direct plan must contain at least one non-empty HT sub-band decode step"
        );
    }

    #[test]
    fn ht_decoder_hook_is_used_for_htj2k_codeblocks() {
        let pixels: Vec<u8> = (0..16).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let bytes = encode_htj2k(&pixels, 4, 4, 1, 8, false, &options).expect("encode ht");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut hooked_context = DecoderContext::default();
        let mut hook = FailingHtDecoder { called: false };
        let error = match image.decode_components_with_ht_decoder(&mut hooked_context, &mut hook) {
            Ok(_) => panic!("hooked decode must use external HT decoder"),
            Err(error) => error,
        };

        assert!(hook.called, "HT decoder hook must be invoked");
        assert_eq!(
            error,
            DecodeError::Decoding(DecodingError::CodeBlockDecodeFailure)
        );
    }

    #[test]
    fn classic_decoder_hook_is_used_for_j2k_codeblocks() {
        let bytes = fixture();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut hooked_context = DecoderContext::default();
        let mut hook = FailingClassicDecoder { called: false };
        let error = match image.decode_components_with_ht_decoder(&mut hooked_context, &mut hook) {
            Ok(_) => panic!("hooked decode must use external classic decoder"),
            Err(error) => error,
        };

        assert!(hook.called, "classic decoder hook must be invoked");
        assert_eq!(
            error,
            DecodeError::Decoding(DecodingError::CodeBlockDecodeFailure)
        );
    }

    #[test]
    fn classic_sub_band_decoder_hook_is_used_for_j2k_codeblocks() {
        let bytes = fixture_multi_block();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut hooked_context = DecoderContext::default();
        let mut hook = FailingClassicBatchDecoder { called: false };
        let error = match image.decode_components_with_ht_decoder(&mut hooked_context, &mut hook) {
            Ok(_) => panic!("hooked decode must use external classic batch decoder"),
            Err(error) => error,
        };

        assert!(hook.called, "classic sub-band decoder hook must be invoked");
        assert_eq!(
            error,
            DecodeError::Decoding(DecodingError::CodeBlockDecodeFailure)
        );
    }
}
