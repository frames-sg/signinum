//! Top-level JPEG 2000 encode orchestration.
//!
//! Coordinates the full encoding pipeline:
//!   pixels → MCT → DWT → quantize → EBCOT T1 → T2 → codestream
//!
//! Supports both lossless (5-3 reversible) and lossy (9-7 irreversible) encoding.

use alloc::vec;
use alloc::vec::Vec;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::bitplane_encode;
use super::build::SubBandType;
use super::codestream_write::{self, BlockCodingMode, EncodeParams};
use super::fdwt::{self, DwtDecomposition};
use super::forward_mct;
use super::ht_block_encode;
use super::packet_encode::{self, CodeBlockPacketData, ResolutionPacket, SubbandPrecinct};
use super::quantize::{self, QuantStepSize};
use crate::math::{floor_f32, log2_f32};
use crate::{
    CpuOnlyJ2kEncodeStageAccelerator, EncodedJ2kCodeBlock, J2kEncodeStageAccelerator,
    J2kForwardDwt53Job, J2kForwardDwt53Level, J2kForwardDwt53Output, J2kForwardRctJob,
    J2kPacketizationBlockCodingMode, J2kPacketizationCodeBlock, J2kPacketizationEncodeJob,
    J2kPacketizationPacketDescriptor, J2kPacketizationResolution, J2kPacketizationSubband,
    J2kSubBandType, J2kTier1CodeBlockEncodeJob,
};
use crate::{DecodeSettings, Image};

/// Encoding options for JPEG 2000.
#[derive(Debug, Clone)]
pub struct EncodeOptions {
    /// Number of decomposition levels (default: 5).
    pub num_decomposition_levels: u8,
    /// Use reversible (lossless) transform (default: true).
    pub reversible: bool,
    /// Code-block width exponent minus 2 (default: 4, meaning 2^6=64).
    pub code_block_width_exp: u8,
    /// Code-block height exponent minus 2 (default: 4, meaning 2^6=64).
    pub code_block_height_exp: u8,
    /// Number of guard bits (default: 1 for reversible, 2 for irreversible).
    pub guard_bits: u8,
    /// Encode using HT block coding (HTJ2K / Part 15) instead of classic EBCOT.
    pub use_ht_block_coding: bool,
    /// Packet progression order to write in COD and use for packetization.
    pub progression_order: EncodeProgressionOrder,
    /// Write a TLM marker segment for the single tile-part.
    pub write_tlm: bool,
    /// Apply the JPEG 2000 multi-component color transform for 3+ component inputs.
    pub use_mct: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            num_decomposition_levels: 5,
            reversible: true,
            code_block_width_exp: 4,
            code_block_height_exp: 4,
            guard_bits: 1,
            use_ht_block_coding: false,
            progression_order: EncodeProgressionOrder::Lrcp,
            write_tlm: false,
            use_mct: true,
        }
    }
}

/// JPEG 2000 packet progression orders supported by the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EncodeProgressionOrder {
    /// Layer-resolution-component-position progression.
    #[default]
    Lrcp,
    /// Resolution-position-component-layer progression.
    Rpcl,
}

/// Encode pixel data into a JPEG 2000 codestream.
///
/// # Arguments
/// * `pixels` — Raw pixel data. For 8-bit: one byte per sample. For >8-bit: two bytes per sample (little-endian u16).
/// * `width` — Image width in pixels.
/// * `height` — Image height in pixels.
/// * `num_components` — Number of components (1 for grayscale, 3 for RGB).
/// * `bit_depth` — Bits per sample (e.g., 8, 12, 16).
/// * `signed` — Whether samples are signed.
/// * `options` — Encoding parameters.
///
/// # Returns
/// The encoded JPEG 2000 codestream bytes (`.j2c` format).
pub fn encode(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u8,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_with_accelerator(
        pixels,
        width,
        height,
        num_components,
        bit_depth,
        signed,
        options,
        &mut accelerator,
    )
}

/// Encode pixel data into a JPEG 2000 codestream using optional encode-stage hooks.
///
/// Stage hooks may accelerate forward RCT, forward 5/3 DWT, Tier-1 code-block
/// encode, and packetization. Returning fallback from a hook preserves the CPU
/// baseline for that stage.
pub fn encode_with_accelerator(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u8,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    let block_coding_mode = block_coding_mode(options);
    let codestream = encode_impl(
        pixels,
        width,
        height,
        num_components,
        bit_depth,
        signed,
        options,
        block_coding_mode,
        accelerator,
    )?;

    if block_coding_mode == BlockCodingMode::HighThroughput {
        validate_htj2k_codestream(
            &codestream,
            pixels,
            width,
            height,
            num_components,
            bit_depth,
            signed,
            options.reversible,
        )?;
    }

    Ok(codestream)
}

/// Encode pixel data into an HTJ2K codestream.
///
/// Lossless HTJ2K output is self-validated before it is returned.
pub fn encode_htj2k(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u8,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
) -> Result<Vec<u8>, &'static str> {
    let mut options = options.clone();
    options.use_ht_block_coding = true;
    encode(
        pixels,
        width,
        height,
        num_components,
        bit_depth,
        signed,
        &options,
    )
}

fn block_coding_mode(options: &EncodeOptions) -> BlockCodingMode {
    if options.use_ht_block_coding {
        BlockCodingMode::HighThroughput
    } else {
        BlockCodingMode::Classic
    }
}

fn validate_htj2k_codestream(
    codestream: &[u8],
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u8,
    bit_depth: u8,
    signed: bool,
    reversible: bool,
) -> Result<(), &'static str> {
    let image = Image::new(codestream, &DecodeSettings::default())
        .map_err(|_| "generated HTJ2K codestream failed self-validation")?;
    let decoded = image
        .decode_native()
        .map_err(|_| "generated HTJ2K codestream failed self-validation")?;

    if decoded.width != width
        || decoded.height != height
        || decoded.bit_depth != bit_depth
        || decoded.num_components != num_components
    {
        return Err("generated HTJ2K codestream failed self-validation");
    }

    if reversible && !native_samples_equal(pixels, &decoded.data, bit_depth, signed) {
        return Err("generated HTJ2K codestream did not roundtrip");
    }

    Ok(())
}

fn native_samples_equal(expected: &[u8], actual: &[u8], bit_depth: u8, signed: bool) -> bool {
    if expected.len() != actual.len() {
        return false;
    }

    let bytes_per_sample = if bit_depth <= 8 { 1 } else { 2 };
    let sample_count = expected.len() / bytes_per_sample;
    (0..sample_count).all(|sample_index| {
        decode_native_sample(expected, sample_index, bit_depth, signed)
            == decode_native_sample(actual, sample_index, bit_depth, signed)
    })
}

fn decode_native_sample(bytes: &[u8], sample_index: usize, bit_depth: u8, signed: bool) -> i32 {
    let byte_offset = sample_index * if bit_depth <= 8 { 1 } else { 2 };
    let mask = (1u32 << u32::from(bit_depth)) - 1;
    let raw = if bit_depth <= 8 {
        u32::from(bytes[byte_offset])
    } else {
        u32::from(u16::from_le_bytes([
            bytes[byte_offset],
            bytes[byte_offset + 1],
        ]))
    } & mask;

    if signed {
        let shift = 32 - u32::from(bit_depth);
        ((raw << shift) as i32) >> shift
    } else {
        raw as i32
    }
}

fn encode_impl(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u8,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    block_coding_mode: BlockCodingMode,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    if width == 0 || height == 0 {
        return Err("invalid dimensions");
    }
    if num_components == 0 || num_components > 4 {
        return Err("unsupported component count");
    }
    if bit_depth == 0 || bit_depth > 16 {
        return Err("unsupported bit depth");
    }

    let num_pixels = (width * height) as usize;
    let bytes_per_sample = if bit_depth <= 8 { 1 } else { 2 };
    let expected_len = num_pixels * num_components as usize * bytes_per_sample;
    if pixels.len() < expected_len {
        return Err("pixel data too short");
    }

    // Step 1: Convert pixel bytes to f32 component arrays
    let mut components = deinterleave_to_f32(pixels, num_pixels, num_components, bit_depth, signed);

    // Step 2: Apply forward MCT if RGB with 3+ components
    let use_mct = options.use_mct && num_components >= 3;
    if use_mct {
        if options.reversible {
            if !try_encode_forward_rct(&mut components, accelerator)? {
                forward_mct::forward_rct(&mut components);
            }
        } else {
            forward_mct::forward_ict(&mut components);
        }
    }

    // Step 3: Apply forward DWT to each component
    let num_levels = options.num_decomposition_levels.min(
        // Don't decompose more than the image supports
        max_decomposition_levels(width, height),
    );

    let decompositions: Vec<DwtDecomposition> = components
        .iter()
        .map(|comp| {
            encode_forward_dwt(
                comp,
                width,
                height,
                num_levels,
                options.reversible,
                accelerator,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Step 4: Compute quantization step sizes
    let guard_bits = if options.reversible {
        if use_mct {
            options.guard_bits.max(2)
        } else {
            options.guard_bits
        }
    } else {
        options.guard_bits.max(2)
    };

    let step_sizes =
        quantize::compute_step_sizes(bit_depth, num_levels, options.reversible, guard_bits);

    // Step 5: Quantize and encode code-blocks for each component
    let cb_width = 1u32 << (options.code_block_width_exp + 2);
    let cb_height = 1u32 << (options.code_block_height_exp + 2);

    let mut component_resolution_packets: Vec<Vec<PreparedResolutionPacket>> =
        Vec::with_capacity(num_components as usize);

    for decomp in decompositions.iter().take(num_components as usize) {
        let mut packets = Vec::with_capacity(num_levels as usize + 1);

        // LL subband (resolution 0)
        let ll_subband = prepare_subband(
            &decomp.ll,
            decomp.ll_width,
            decomp.ll_height,
            &step_sizes[0],
            guard_bits,
            options.reversible,
            block_coding_mode,
            cb_width,
            cb_height,
            SubBandType::LowLow,
        )?;
        packets.push(PreparedResolutionPacket {
            subbands: vec![ll_subband],
        });

        // Higher resolution levels
        for (level_idx, level) in decomp.levels.iter().enumerate() {
            let step_base = 1 + level_idx * 3;

            // HL subband
            let hl_subband = prepare_subband(
                &level.hl,
                level.high_width,
                level.low_height,
                &step_sizes[step_base],
                guard_bits,
                options.reversible,
                block_coding_mode,
                cb_width,
                cb_height,
                SubBandType::HighLow,
            )?;

            // LH subband
            let lh_subband = prepare_subband(
                &level.lh,
                level.low_width,
                level.high_height,
                &step_sizes[step_base + 1],
                guard_bits,
                options.reversible,
                block_coding_mode,
                cb_width,
                cb_height,
                SubBandType::LowHigh,
            )?;

            // HH subband
            let hh_subband = prepare_subband(
                &level.hh,
                level.high_width,
                level.high_height,
                &step_sizes[step_base + 2],
                guard_bits,
                options.reversible,
                block_coding_mode,
                cb_width,
                cb_height,
                SubBandType::HighHigh,
            )?;

            packets.push(PreparedResolutionPacket {
                subbands: vec![hl_subband, lh_subband, hh_subband],
            });
        }

        component_resolution_packets.push(packets);
    }

    let prepared_resolution_packets =
        ordered_prepared_resolution_packets(component_resolution_packets, options)?;
    let resolution_packets =
        encode_prepared_resolution_packets(prepared_resolution_packets, accelerator)?;

    // Step 6: Form tile bitstream (T2)
    let mut resolution_packets = resolution_packets;
    let packetization_resolutions = public_packetization_resolutions(&resolution_packets);
    let packet_descriptors =
        packet_descriptors_for_order(resolution_packets.len(), 1, num_components)?;
    let packetization_job = J2kPacketizationEncodeJob {
        resolution_count: resolution_packets.len() as u32,
        num_layers: 1,
        num_components,
        code_block_count: count_code_blocks(&resolution_packets)?,
        progression_order: public_packetization_progression_order(options.progression_order),
        packet_descriptors: &packet_descriptors,
        resolutions: &packetization_resolutions,
    };
    let tile_data = accelerator
        .encode_packetization(packetization_job)?
        .unwrap_or_else(|| {
            packet_encode::form_tile_bitstream(&mut resolution_packets, 1, num_components)
        });

    // Step 7: Write codestream
    let quant_params: Vec<(u16, u16)> = step_sizes
        .iter()
        .map(|s| (s.exponent, s.mantissa))
        .collect();

    let params = EncodeParams {
        width,
        height,
        num_components,
        bit_depth,
        signed,
        num_decomposition_levels: num_levels,
        reversible: options.reversible,
        code_block_width_exp: options.code_block_width_exp,
        code_block_height_exp: options.code_block_height_exp,
        num_layers: 1,
        use_mct,
        guard_bits,
        block_coding_mode,
        progression_order: options.progression_order,
        write_tlm: options.write_tlm,
    };

    Ok(codestream_write::write_codestream(
        &params,
        &tile_data,
        &quant_params,
    ))
}

fn try_encode_forward_rct(
    components: &mut [Vec<f32>],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<bool, &'static str> {
    debug_assert!(components.len() >= 3);
    let (plane0, rest) = components.split_at_mut(1);
    let (plane1, plane2) = rest.split_at_mut(1);
    accelerator.encode_forward_rct(J2kForwardRctJob {
        plane0: &mut plane0[0],
        plane1: &mut plane1[0],
        plane2: &mut plane2[0],
    })
}

fn encode_forward_dwt(
    component: &[f32],
    width: u32,
    height: u32,
    num_levels: u8,
    reversible: bool,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<DwtDecomposition, &'static str> {
    if reversible {
        if let Some(output) = accelerator.encode_forward_dwt53(J2kForwardDwt53Job {
            samples: component,
            width,
            height,
            num_levels,
        })? {
            return convert_forward_dwt53_output(output);
        }
    }

    Ok(fdwt::forward_dwt(
        component, width, height, num_levels, reversible,
    ))
}

fn convert_forward_dwt53_output(
    output: J2kForwardDwt53Output,
) -> Result<DwtDecomposition, &'static str> {
    validate_band_len(output.ll.len(), output.ll_width, output.ll_height)?;
    let mut levels = Vec::with_capacity(output.levels.len());
    for level in output.levels {
        validate_dwt53_level(&level)?;
        levels.push(fdwt::DwtLevel {
            hl: level.hl,
            lh: level.lh,
            hh: level.hh,
            width: level.width,
            height: level.height,
            low_width: level.low_width,
            low_height: level.low_height,
            high_width: level.high_width,
            high_height: level.high_height,
        });
    }
    Ok(DwtDecomposition {
        ll: output.ll,
        ll_width: output.ll_width,
        ll_height: output.ll_height,
        levels,
    })
}

fn validate_dwt53_level(level: &J2kForwardDwt53Level) -> Result<(), &'static str> {
    validate_band_len(level.hl.len(), level.high_width, level.low_height)?;
    validate_band_len(level.lh.len(), level.low_width, level.high_height)?;
    validate_band_len(level.hh.len(), level.high_width, level.high_height)?;
    Ok(())
}

fn validate_band_len(actual: usize, width: u32, height: u32) -> Result<(), &'static str> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .ok_or("accelerated DWT output dimensions overflow")?;
    if actual != expected {
        return Err("accelerated DWT output length mismatch");
    }
    Ok(())
}

fn count_code_blocks(resolution_packets: &[ResolutionPacket]) -> Result<u32, &'static str> {
    let count = resolution_packets
        .iter()
        .flat_map(|resolution| resolution.subbands.iter())
        .try_fold(0usize, |acc, subband| {
            acc.checked_add(subband.code_blocks.len())
                .ok_or("packetization code-block count overflow")
        })?;
    u32::try_from(count).map_err(|_| "packetization code-block count exceeds u32")
}

fn packet_descriptors_for_order(
    packet_count: usize,
    num_layers: u8,
    num_components: u8,
) -> Result<Vec<J2kPacketizationPacketDescriptor>, &'static str> {
    if num_layers != 1 {
        return Err("encode currently prepares one packet contribution layer");
    }
    let component_count = usize::from(num_components).max(1);
    (0..packet_count)
        .map(|packet_index| {
            Ok(J2kPacketizationPacketDescriptor {
                packet_index: u32::try_from(packet_index)
                    .map_err(|_| "packet descriptor index exceeds u32")?,
                state_index: u32::try_from(packet_index)
                    .map_err(|_| "packet descriptor state index exceeds u32")?,
                layer: 0,
                resolution: u32::try_from(packet_index / component_count)
                    .map_err(|_| "packet descriptor resolution exceeds u32")?,
                component: u8::try_from(packet_index % component_count)
                    .map_err(|_| "packet descriptor component exceeds u8")?,
                precinct: 0,
            })
        })
        .collect()
}

fn ordered_prepared_resolution_packets(
    component_resolution_packets: Vec<Vec<PreparedResolutionPacket>>,
    options: &EncodeOptions,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    match options.progression_order {
        EncodeProgressionOrder::Lrcp | EncodeProgressionOrder::Rpcl => {
            lrcp_ordered_prepared_resolution_packets(component_resolution_packets)
        }
    }
}

fn lrcp_ordered_prepared_resolution_packets(
    component_resolution_packets: Vec<Vec<PreparedResolutionPacket>>,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    let resolution_count = component_resolution_packets
        .first()
        .map_or(0usize, alloc::vec::Vec::len);
    let mut component_iters: Vec<_> = component_resolution_packets
        .into_iter()
        .map(alloc::vec::Vec::into_iter)
        .collect();
    let mut resolution_packets =
        Vec::with_capacity(resolution_count.saturating_mul(component_iters.len()));

    for _resolution in 0..resolution_count {
        for component in &mut component_iters {
            resolution_packets.push(
                component
                    .next()
                    .ok_or("component packet resolution count mismatch")?,
            );
        }
    }

    if component_iters
        .iter_mut()
        .any(|component| component.next().is_some())
    {
        return Err("component packet resolution count mismatch");
    }

    Ok(resolution_packets)
}

fn public_packetization_progression_order(
    progression_order: EncodeProgressionOrder,
) -> crate::J2kPacketizationProgressionOrder {
    match progression_order {
        EncodeProgressionOrder::Lrcp => crate::J2kPacketizationProgressionOrder::Lrcp,
        EncodeProgressionOrder::Rpcl => crate::J2kPacketizationProgressionOrder::Rpcl,
    }
}

fn public_packetization_resolutions(
    resolution_packets: &[ResolutionPacket],
) -> Vec<J2kPacketizationResolution<'_>> {
    resolution_packets
        .iter()
        .map(|resolution| J2kPacketizationResolution {
            subbands: resolution
                .subbands
                .iter()
                .map(|subband| J2kPacketizationSubband {
                    code_blocks: subband
                        .code_blocks
                        .iter()
                        .map(|code_block| J2kPacketizationCodeBlock {
                            data: &code_block.data,
                            num_coding_passes: code_block.num_coding_passes,
                            num_zero_bitplanes: code_block.num_zero_bitplanes,
                            previously_included: code_block.previously_included,
                            l_block: code_block.l_block,
                            block_coding_mode: public_packetization_block_coding_mode(
                                code_block.block_coding_mode,
                            ),
                        })
                        .collect(),
                    num_cbs_x: subband.num_cbs_x,
                    num_cbs_y: subband.num_cbs_y,
                })
                .collect(),
        })
        .collect()
}

fn public_packetization_block_coding_mode(
    block_coding_mode: BlockCodingMode,
) -> J2kPacketizationBlockCodingMode {
    match block_coding_mode {
        BlockCodingMode::Classic => J2kPacketizationBlockCodingMode::Classic,
        BlockCodingMode::HighThroughput => J2kPacketizationBlockCodingMode::HighThroughput,
    }
}

struct PreparedEncodeCodeBlock {
    coefficients: Vec<i32>,
    width: u32,
    height: u32,
}

struct PreparedEncodeSubband {
    code_blocks: Vec<PreparedEncodeCodeBlock>,
    num_cbs_x: u32,
    num_cbs_y: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
    block_coding_mode: BlockCodingMode,
}

struct PreparedResolutionPacket {
    subbands: Vec<PreparedEncodeSubband>,
}

fn prepare_subband(
    coefficients: &[f32],
    width: u32,
    height: u32,
    step_size: &QuantStepSize,
    guard_bits: u8,
    reversible: bool,
    block_coding_mode: BlockCodingMode,
    cb_width: u32,
    cb_height: u32,
    sub_band_type: SubBandType,
) -> Result<PreparedEncodeSubband, &'static str> {
    if width == 0 || height == 0 {
        return Ok(PreparedEncodeSubband {
            code_blocks: Vec::new(),
            num_cbs_x: 0,
            num_cbs_y: 0,
            sub_band_type,
            total_bitplanes: 0,
            block_coding_mode,
        });
    }

    // Quantize
    let quantized = quantize::quantize_subband(coefficients, step_size, guard_bits, reversible);
    debug_assert!(step_size.exponent <= u16::from(u8::MAX));
    let total_bitplanes = guard_bits
        .saturating_add(step_size.exponent as u8)
        .saturating_sub(1);

    // Split into code-blocks
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);
    let mut code_blocks = Vec::with_capacity((num_cbs_x * num_cbs_y) as usize);

    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let x0 = cbx * cb_width;
            let y0 = cby * cb_height;
            let x1 = (x0 + cb_width).min(width);
            let y1 = (y0 + cb_height).min(height);
            let cbw = x1 - x0;
            let cbh = y1 - y0;

            // Extract code-block coefficients
            let mut cb_coeffs = vec![0i32; (cbw * cbh) as usize];
            for y in 0..cbh {
                for x in 0..cbw {
                    cb_coeffs[(y * cbw + x) as usize] =
                        quantized[((y0 + y) * width + (x0 + x)) as usize];
                }
            }

            code_blocks.push(PreparedEncodeCodeBlock {
                coefficients: cb_coeffs,
                width: cbw,
                height: cbh,
            });
        }
    }

    Ok(PreparedEncodeSubband {
        code_blocks,
        num_cbs_x,
        num_cbs_y,
        sub_band_type,
        total_bitplanes,
        block_coding_mode,
    })
}

fn encode_prepared_resolution_packets(
    prepared_packets: Vec<PreparedResolutionPacket>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<ResolutionPacket>, &'static str> {
    let subband_counts: Vec<_> = prepared_packets
        .iter()
        .map(|packet| packet.subbands.len())
        .collect();
    let prepared_subbands: Vec<_> = prepared_packets
        .into_iter()
        .flat_map(|packet| packet.subbands)
        .collect();
    let mut encoded_subbands =
        encode_prepared_subbands(prepared_subbands, accelerator)?.into_iter();

    subband_counts
        .into_iter()
        .map(|subband_count| {
            let mut subbands = Vec::with_capacity(subband_count);
            for _ in 0..subband_count {
                subbands.push(
                    encoded_subbands
                        .next()
                        .ok_or("encoded subband count mismatch")?,
                );
            }
            Ok(ResolutionPacket { subbands })
        })
        .collect()
}

fn encode_prepared_subbands(
    prepared_subbands: Vec<PreparedEncodeSubband>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<SubbandPrecinct>, &'static str> {
    let block_coding_mode = prepared_subbands
        .iter()
        .find(|subband| !subband.code_blocks.is_empty())
        .map(|subband| subband.block_coding_mode);
    let encoded_blocks = match block_coding_mode {
        Some(BlockCodingMode::HighThroughput) => {
            encode_all_ht_code_blocks(&prepared_subbands, accelerator)?
        }
        Some(BlockCodingMode::Classic) => {
            encode_all_tier1_code_blocks(&prepared_subbands, accelerator)?
        }
        None => Vec::new(),
    };

    let mut encoded_iter = encoded_blocks.into_iter();
    let mut precincts = Vec::with_capacity(prepared_subbands.len());
    for subband in prepared_subbands {
        let mut code_blocks = Vec::with_capacity(subband.code_blocks.len());
        for _ in 0..subband.code_blocks.len() {
            let encoded = encoded_iter
                .next()
                .ok_or("encoded code-block count mismatch")?;
            code_blocks.push(CodeBlockPacketData {
                data: encoded.data,
                num_coding_passes: encoded.num_coding_passes,
                num_zero_bitplanes: encoded.num_zero_bitplanes,
                previously_included: false,
                l_block: 3,
                block_coding_mode: subband.block_coding_mode,
            });
        }
        precincts.push(SubbandPrecinct {
            code_blocks,
            num_cbs_x: subband.num_cbs_x,
            num_cbs_y: subband.num_cbs_y,
        });
    }
    if encoded_iter.next().is_some() {
        return Err("encoded code-block count mismatch");
    }

    Ok(precincts)
}

fn encode_all_ht_code_blocks(
    prepared_subbands: &[PreparedEncodeSubband],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    let jobs: Vec<_> = prepared_subbands
        .iter()
        .flat_map(|subband| {
            subband
                .code_blocks
                .iter()
                .map(move |block| crate::J2kHtCodeBlockEncodeJob {
                    coefficients: &block.coefficients,
                    width: block.width,
                    height: block.height,
                    total_bitplanes: subband.total_bitplanes,
                })
        })
        .collect();

    if let Some(encoded) = accelerator.encode_ht_code_blocks(&jobs)? {
        if encoded.len() != jobs.len() {
            return Err("accelerated HT code-block batch length mismatch");
        }
        return Ok(encoded
            .into_iter()
            .map(ht_encoded_code_block_from_accelerator)
            .collect());
    }

    if accelerator.prefer_parallel_cpu_code_block_fallback() {
        return encode_all_ht_code_blocks_parallel(&jobs);
    }

    jobs.iter()
        .map(|job| {
            encode_ht_code_block(
                job.coefficients,
                job.width,
                job.height,
                job.total_bitplanes,
                accelerator,
            )
        })
        .collect()
}

fn encode_all_tier1_code_blocks(
    prepared_subbands: &[PreparedEncodeSubband],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    let style = default_public_code_block_style();
    let jobs: Vec<_> = prepared_subbands
        .iter()
        .flat_map(|subband| {
            let public_sub_band_type = public_sub_band_type(subband.sub_band_type);
            subband
                .code_blocks
                .iter()
                .map(move |block| J2kTier1CodeBlockEncodeJob {
                    coefficients: &block.coefficients,
                    width: block.width,
                    height: block.height,
                    sub_band_type: public_sub_band_type,
                    total_bitplanes: subband.total_bitplanes,
                    style,
                })
        })
        .collect();

    if let Some(encoded) = accelerator.encode_tier1_code_blocks(&jobs)? {
        if encoded.len() != jobs.len() {
            return Err("accelerated classic code-block batch length mismatch");
        }
        return Ok(encoded
            .into_iter()
            .map(encoded_code_block_from_accelerator)
            .collect());
    }

    if accelerator.prefer_parallel_cpu_code_block_fallback() {
        return encode_all_tier1_code_blocks_parallel(&jobs);
    }

    let mut encoded = Vec::with_capacity(jobs.len());
    for subband in prepared_subbands {
        for block in &subband.code_blocks {
            encoded.push(encode_tier1_code_block(
                &block.coefficients,
                block.width,
                block.height,
                subband.sub_band_type,
                subband.total_bitplanes,
                accelerator,
            )?);
        }
    }
    Ok(encoded)
}

#[cfg(feature = "parallel")]
fn encode_all_ht_code_blocks_parallel(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    jobs.par_iter()
        .map(|job| {
            ht_block_encode::encode_code_block(
                job.coefficients,
                job.width,
                job.height,
                job.total_bitplanes,
            )
        })
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn encode_all_ht_code_blocks_parallel(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    jobs.iter()
        .map(|job| {
            ht_block_encode::encode_code_block(
                job.coefficients,
                job.width,
                job.height,
                job.total_bitplanes,
            )
        })
        .collect()
}

#[cfg(feature = "parallel")]
fn encode_all_tier1_code_blocks_parallel(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    jobs.par_iter()
        .map(|job| {
            Ok(bitplane_encode::encode_code_block(
                job.coefficients,
                job.width,
                job.height,
                internal_sub_band_type(job.sub_band_type),
                job.total_bitplanes,
            ))
        })
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn encode_all_tier1_code_blocks_parallel(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    jobs.iter()
        .map(|job| {
            Ok(bitplane_encode::encode_code_block(
                job.coefficients,
                job.width,
                job.height,
                internal_sub_band_type(job.sub_band_type),
                job.total_bitplanes,
            ))
        })
        .collect()
}

fn encode_ht_code_block(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<bitplane_encode::EncodedCodeBlock, &'static str> {
    if let Some(encoded) = accelerator.encode_ht_code_block(crate::J2kHtCodeBlockEncodeJob {
        coefficients,
        width,
        height,
        total_bitplanes,
    })? {
        return Ok(ht_encoded_code_block_from_accelerator(encoded));
    }

    ht_block_encode::encode_code_block(coefficients, width, height, total_bitplanes)
}

fn ht_encoded_code_block_from_accelerator(
    encoded: crate::EncodedHtJ2kCodeBlock,
) -> bitplane_encode::EncodedCodeBlock {
    bitplane_encode::EncodedCodeBlock {
        data: encoded.data,
        num_coding_passes: encoded.num_coding_passes,
        num_zero_bitplanes: encoded.num_zero_bitplanes,
    }
}

fn encode_tier1_code_block(
    coefficients: &[i32],
    width: u32,
    height: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<bitplane_encode::EncodedCodeBlock, &'static str> {
    if let Some(encoded) = accelerator.encode_tier1_code_block(J2kTier1CodeBlockEncodeJob {
        coefficients,
        width,
        height,
        sub_band_type: public_sub_band_type(sub_band_type),
        total_bitplanes,
        style: default_public_code_block_style(),
    })? {
        return Ok(encoded_code_block_from_accelerator(encoded));
    }

    Ok(bitplane_encode::encode_code_block(
        coefficients,
        width,
        height,
        sub_band_type,
        total_bitplanes,
    ))
}

fn encoded_code_block_from_accelerator(
    encoded: EncodedJ2kCodeBlock,
) -> bitplane_encode::EncodedCodeBlock {
    bitplane_encode::EncodedCodeBlock {
        data: encoded.data,
        num_coding_passes: encoded.number_of_coding_passes,
        num_zero_bitplanes: encoded.missing_bit_planes,
    }
}

fn public_sub_band_type(sub_band_type: SubBandType) -> J2kSubBandType {
    match sub_band_type {
        SubBandType::LowLow => J2kSubBandType::LowLow,
        SubBandType::HighLow => J2kSubBandType::HighLow,
        SubBandType::LowHigh => J2kSubBandType::LowHigh,
        SubBandType::HighHigh => J2kSubBandType::HighHigh,
    }
}

fn internal_sub_band_type(sub_band_type: J2kSubBandType) -> SubBandType {
    match sub_band_type {
        J2kSubBandType::LowLow => SubBandType::LowLow,
        J2kSubBandType::HighLow => SubBandType::HighLow,
        J2kSubBandType::LowHigh => SubBandType::LowHigh,
        J2kSubBandType::HighHigh => SubBandType::HighHigh,
    }
}

fn default_public_code_block_style() -> crate::J2kCodeBlockStyle {
    crate::J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: false,
        reset_context_probabilities: false,
        termination_on_each_pass: false,
        vertically_causal_context: false,
        segmentation_symbols: false,
    }
}

/// Convert interleaved pixel bytes to per-component f32 arrays.
fn deinterleave_to_f32(
    pixels: &[u8],
    num_pixels: usize,
    num_components: u8,
    bit_depth: u8,
    signed: bool,
) -> Vec<Vec<f32>> {
    let nc = num_components as usize;
    let mut components = vec![vec![0.0f32; num_pixels]; nc];
    let unsigned_offset = if signed {
        0.0
    } else {
        (1u32 << (bit_depth as u32 - 1)) as f32
    };

    if bit_depth <= 8 {
        for (i, pixel) in pixels.chunks_exact(nc).take(num_pixels).enumerate() {
            for (c, component) in components.iter_mut().enumerate().take(nc) {
                let val = pixel[c];
                component[i] = if signed {
                    (val as i8) as f32
                } else {
                    val as f32 - unsigned_offset
                };
            }
        }
    } else {
        // 16-bit samples (little-endian)
        for (i, pixel) in pixels.chunks_exact(nc * 2).take(num_pixels).enumerate() {
            for (c, component) in components.iter_mut().enumerate().take(nc) {
                let offset = c * 2;
                let val = u16::from_le_bytes([pixel[offset], pixel[offset + 1]]);
                component[i] = if signed {
                    (val as i16) as f32
                } else {
                    val as f32 - unsigned_offset
                };
            }
        }
    }

    components
}

/// Calculate the maximum number of decomposition levels for given dimensions.
fn max_decomposition_levels(width: u32, height: u32) -> u8 {
    let min_dim = width.min(height);
    if min_dim <= 1 {
        return 0;
    }
    floor_f32(log2_f32(min_dim as f32)) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_8bit_gray() {
        let width = 8u32;
        let height = 8u32;
        let pixels: Vec<u8> = (0..64).collect();

        let result = encode(
            &pixels,
            width,
            height,
            1,
            8,
            false,
            &EncodeOptions {
                num_decomposition_levels: 2,
                ..Default::default()
            },
        );

        assert!(result.is_ok());
        let codestream = result.unwrap();
        // Verify SOC marker
        assert_eq!(codestream[0], 0xFF);
        assert_eq!(codestream[1], 0x4F);
        // Verify EOC marker
        let len = codestream.len();
        assert_eq!(codestream[len - 2], 0xFF);
        assert_eq!(codestream[len - 1], 0xD9);
    }

    #[test]
    fn test_encode_16bit_gray() {
        let width = 8u32;
        let height = 8u32;
        let mut pixels = Vec::with_capacity(128);
        for i in 0..64u16 {
            let val = i * 100;
            pixels.extend_from_slice(&val.to_le_bytes());
        }

        let result = encode(
            &pixels,
            width,
            height,
            1,
            16,
            false,
            &EncodeOptions {
                num_decomposition_levels: 2,
                ..Default::default()
            },
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_encode_rgb() {
        let width = 16u32;
        let height = 16u32;
        let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i & 0xFF) as u8).collect();

        let result = encode(
            &pixels,
            width,
            height,
            3,
            8,
            false,
            &EncodeOptions {
                num_decomposition_levels: 3,
                ..Default::default()
            },
        );

        assert!(result.is_ok(), "RGB encode failed: {:?}", result.err());
    }

    #[test]
    fn encode_with_accelerator_calls_lossless_stage_hooks() {
        #[derive(Default)]
        struct CountingAccelerator {
            forward_rct: usize,
            forward_dwt53: usize,
            tier1_code_blocks: usize,
            tier1_code_block_batches: usize,
            tier1_batched_jobs: usize,
            packetization: usize,
            packetization_resolution_count: u32,
            packetization_code_block_count: u32,
            packetization_saw_payload: bool,
        }

        impl crate::J2kEncodeStageAccelerator for CountingAccelerator {
            fn encode_forward_rct(
                &mut self,
                _job: crate::J2kForwardRctJob<'_>,
            ) -> core::result::Result<bool, &'static str> {
                self.forward_rct += 1;
                Ok(false)
            }

            fn encode_forward_dwt53(
                &mut self,
                _job: crate::J2kForwardDwt53Job<'_>,
            ) -> core::result::Result<Option<crate::J2kForwardDwt53Output>, &'static str>
            {
                self.forward_dwt53 += 1;
                Ok(None)
            }

            fn encode_tier1_code_block(
                &mut self,
                _job: crate::J2kTier1CodeBlockEncodeJob<'_>,
            ) -> core::result::Result<Option<crate::EncodedJ2kCodeBlock>, &'static str>
            {
                self.tier1_code_blocks += 1;
                Ok(None)
            }

            fn encode_tier1_code_blocks(
                &mut self,
                jobs: &[crate::J2kTier1CodeBlockEncodeJob<'_>],
            ) -> core::result::Result<Option<Vec<crate::EncodedJ2kCodeBlock>>, &'static str>
            {
                self.tier1_code_block_batches += 1;
                self.tier1_batched_jobs += jobs.len();
                Ok(None)
            }

            fn encode_packetization(
                &mut self,
                job: crate::J2kPacketizationEncodeJob<'_>,
            ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
                self.packetization += 1;
                self.packetization_resolution_count = job.resolution_count;
                self.packetization_code_block_count = job.code_block_count;
                self.packetization_saw_payload = job
                    .resolutions
                    .iter()
                    .flat_map(|resolution| resolution.subbands.iter())
                    .flat_map(|subband| subband.code_blocks.iter())
                    .any(|code_block| !code_block.data.is_empty());
                Ok(None)
            }
        }

        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| (i & 0xFF) as u8).collect();
        let options = EncodeOptions {
            num_decomposition_levels: 1,
            reversible: true,
            ..EncodeOptions::default()
        };
        let mut accelerator = CountingAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 8, 8, 3, 8, false, &options, &mut accelerator)
                .expect("encode with accelerator hooks");

        assert!(codestream.starts_with(&[0xFF, 0x4F]));
        assert_eq!(accelerator.forward_rct, 1);
        assert_eq!(accelerator.forward_dwt53, 3);
        assert!(accelerator.tier1_code_block_batches > 0);
        assert_eq!(
            accelerator.tier1_code_blocks,
            accelerator.tier1_batched_jobs
        );
        assert_eq!(accelerator.packetization, 1);
        assert_eq!(accelerator.packetization_resolution_count, 6);
        assert_eq!(
            accelerator.packetization_code_block_count,
            u32::try_from(accelerator.tier1_code_blocks).expect("test code-block count fits u32")
        );
        assert!(accelerator.packetization_saw_payload);
    }

    #[test]
    fn cpu_only_accelerator_opts_into_parallel_block_fallback_only_for_native_cpu() {
        #[derive(Default)]
        struct ExternalAccelerator;

        impl crate::J2kEncodeStageAccelerator for ExternalAccelerator {}

        let cpu = crate::CpuOnlyJ2kEncodeStageAccelerator;
        let external = ExternalAccelerator;

        assert!(cpu.prefer_parallel_cpu_code_block_fallback());
        assert!(!external.prefer_parallel_cpu_code_block_fallback());
    }

    #[test]
    fn cpu_parallel_block_fallback_matches_serial_classic_and_htj2k_output() {
        #[derive(Default)]
        struct SerialCpuFallbackAccelerator;

        impl crate::J2kEncodeStageAccelerator for SerialCpuFallbackAccelerator {}

        let pixels = gradient_u8(96, 80);
        for use_ht_block_coding in [false, true] {
            let options = EncodeOptions {
                num_decomposition_levels: 1,
                code_block_width_exp: 2,
                code_block_height_exp: 2,
                use_ht_block_coding,
                ..EncodeOptions::default()
            };
            let parallel = encode(&pixels, 96, 80, 1, 8, false, &options)
                .expect("parallel CPU fallback encode");
            let mut serial_accelerator = SerialCpuFallbackAccelerator;
            let serial = encode_with_accelerator(
                &pixels,
                96,
                80,
                1,
                8,
                false,
                &options,
                &mut serial_accelerator,
            )
            .expect("serial CPU fallback encode");

            assert_eq!(parallel, serial);
        }
    }

    #[test]
    fn test_encode_lossy() {
        let pixels: Vec<u8> = (0..64).collect();

        let result = encode(
            &pixels,
            8,
            8,
            1,
            8,
            false,
            &EncodeOptions {
                num_decomposition_levels: 2,
                reversible: false,
                guard_bits: 2,
                ..Default::default()
            },
        );

        assert!(result.is_ok());
    }

    fn assert_htj2k_lossless_roundtrip(
        pixels: &[u8],
        width: u32,
        height: u32,
        bit_depth: u8,
        num_decomposition_levels: u8,
    ) {
        let codestream = encode_htj2k(
            pixels,
            width,
            height,
            1,
            bit_depth,
            false,
            &EncodeOptions {
                num_decomposition_levels,
                ..Default::default()
            },
        )
        .expect("HTJ2K encode");

        assert!(codestream.windows(2).any(|window| window == [0xFF, 0x50]));
        let cod_offset = codestream
            .windows(2)
            .position(|window| window == [0xFF, 0x52])
            .expect("COD marker");
        assert_eq!(codestream[cod_offset + 12], 0x40);

        let image = Image::new(
            &codestream,
            &DecodeSettings {
                resolve_palette_indices: true,
                strict: true,
                target_resolution: None,
            },
        )
        .expect("parse HT codestream");
        let decoded = image.decode_native().expect("decode HT codestream");

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.bit_depth, bit_depth);
        assert_eq!(decoded.data, pixels);
    }

    fn gradient_u8(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 17 + y * 31) % 256) as u8);
            }
        }
        pixels
    }

    #[test]
    fn test_encode_high_throughput_zero_image_roundtrip() {
        let width = 4u32;
        let height = 4u32;
        let sample = 2048u16.to_le_bytes();
        let mut pixels = Vec::with_capacity((width * height * 2) as usize);
        for _ in 0..(width * height) {
            pixels.extend_from_slice(&sample);
        }

        let codestream = encode(
            &pixels,
            width,
            height,
            1,
            12,
            false,
            &EncodeOptions {
                num_decomposition_levels: 2,
                use_ht_block_coding: true,
                ..Default::default()
            },
        )
        .expect("HT all-zero encode");

        assert!(codestream.windows(2).any(|window| window == [0xFF, 0x50]));
        let cod_offset = codestream
            .windows(2)
            .position(|window| window == [0xFF, 0x52])
            .expect("COD marker");
        assert_eq!(codestream[cod_offset + 12], 0x40);

        let image =
            Image::new(&codestream, &DecodeSettings::default()).expect("parse HT codestream");
        let decoded = image.decode_native().expect("decode HT codestream");

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.bit_depth, 12);
        assert_eq!(decoded.data, pixels);
    }

    #[test]
    fn test_encode_high_throughput_nonzero_roundtrip() {
        let width = 1u32;
        let height = 1u32;
        let pixels = 2049u16.to_le_bytes().to_vec();

        let codestream = encode_htj2k(
            &pixels,
            width,
            height,
            1,
            12,
            false,
            &EncodeOptions {
                num_decomposition_levels: 0,
                ..Default::default()
            },
        )
        .expect("HT non-zero encode");

        assert!(codestream.windows(2).any(|window| window == [0xFF, 0x50]));
        let image =
            Image::new(&codestream, &DecodeSettings::default()).expect("parse HT codestream");
        let decoded = image.decode_native().expect("decode HT codestream");

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.bit_depth, 12);
        assert_eq!(decoded.data, pixels);
    }

    #[test]
    fn test_encode_high_throughput_varied_12bit_roundtrip() {
        let mut pixels = Vec::with_capacity(32);
        for i in 0u16..16 {
            pixels.extend_from_slice(&((i * 257) & 0x0FFF).to_le_bytes());
        }

        let codestream = encode_htj2k(
            &pixels,
            4,
            4,
            1,
            12,
            false,
            &EncodeOptions {
                num_decomposition_levels: 1,
                ..Default::default()
            },
        )
        .expect("HT varied encode");

        let image =
            Image::new(&codestream, &DecodeSettings::default()).expect("parse HT codestream");
        let decoded = image.decode_native().expect("decode HT codestream");

        assert_eq!(decoded.width, 4);
        assert_eq!(decoded.height, 4);
        assert_eq!(decoded.bit_depth, 12);
        assert_eq!(decoded.data, pixels);
    }

    #[test]
    fn test_encode_high_throughput_gradient_8bit_roundtrip() {
        let pixels: Vec<u8> = (0..64).collect();

        let codestream = encode_htj2k(
            &pixels,
            8,
            8,
            1,
            8,
            false,
            &EncodeOptions {
                num_decomposition_levels: 3,
                ..Default::default()
            },
        )
        .expect("HT gradient encode");

        let image =
            Image::new(&codestream, &DecodeSettings::default()).expect("parse HT codestream");
        let decoded = image.decode_native().expect("decode HT codestream");

        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.bit_depth, 8);
        assert_eq!(decoded.data, pixels);
    }

    #[test]
    fn test_encode_high_throughput_varied_12bit_large_roundtrip() {
        let width = 16u32;
        let height = 8u32;
        let mut pixels = Vec::with_capacity((width * height * 2) as usize);
        for y in 0u16..height as u16 {
            for x in 0u16..width as u16 {
                let value = (x * 257 + y * 17) & 0x0FFF;
                pixels.extend_from_slice(&value.to_le_bytes());
            }
        }

        assert_htj2k_lossless_roundtrip(&pixels, width, height, 12, 4);
    }

    #[test]
    fn test_encode_high_throughput_ramp_16bit_roundtrip() {
        let width = 48u32;
        let height = 24u32;
        let mut pixels = Vec::with_capacity((width * height * 2) as usize);
        for y in 0u16..height as u16 {
            for x in 0u16..width as u16 {
                let value = x * 521 + y * 997;
                pixels.extend_from_slice(&value.to_le_bytes());
            }
        }

        assert_htj2k_lossless_roundtrip(&pixels, width, height, 16, 4);
    }

    #[test]
    fn test_encode_high_throughput_lossy_large_gradient_is_parseable() {
        let pixels = gradient_u8(128, 128);

        let codestream = encode_htj2k(
            &pixels,
            128,
            128,
            1,
            8,
            false,
            &EncodeOptions {
                num_decomposition_levels: 5,
                reversible: false,
                guard_bits: 2,
                ..Default::default()
            },
        )
        .expect("lossy HT encode");

        assert!(codestream.windows(2).any(|window| window == [0xFF, 0x50]));
        assert!(!codestream.is_empty());

        let image = Image::new(
            &codestream,
            &DecodeSettings {
                resolve_palette_indices: true,
                strict: true,
                target_resolution: None,
            },
        )
        .expect("parse lossy HT codestream");
        let decoded = image.decode_native().expect("decode lossy HT codestream");

        assert_eq!(decoded.width, 128);
        assert_eq!(decoded.height, 128);
        assert_eq!(decoded.bit_depth, 8);
    }

    #[test]
    fn test_encode_invalid_dimensions() {
        let result = encode(&[], 0, 0, 1, 8, false, &EncodeOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_too_short() {
        let pixels = vec![0u8; 10]; // Way too short for 8x8
        let result = encode(&pixels, 8, 8, 1, 8, false, &EncodeOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_deinterleave_rgb() {
        let pixels = vec![
            10u8, 20, 30, // pixel 0: R=10, G=20, B=30
            40, 50, 60, // pixel 1: R=40, G=50, B=60
        ];
        let comps = deinterleave_to_f32(&pixels, 2, 3, 8, false);
        assert_eq!(comps[0], vec![-118.0, -88.0]); // R
        assert_eq!(comps[1], vec![-108.0, -78.0]); // G
        assert_eq!(comps[2], vec![-98.0, -68.0]); // B
    }

    #[test]
    fn test_encode_decode_roundtrip_gray_8bit() {
        use crate::{DecodeSettings, Image};

        // Constant image: all pixels = 42 — simplest possible test
        let original: Vec<u8> = vec![42u8; 64]; // 8x8, all same value
        let encoded = encode(
            &original,
            8,
            8,
            1,
            8,
            false,
            &EncodeOptions {
                num_decomposition_levels: 0,
                reversible: true,
                ..Default::default()
            },
        )
        .expect("encode failed");

        let settings = DecodeSettings {
            resolve_palette_indices: false,
            strict: false,
            target_resolution: None,
        };
        let image = Image::new(&encoded, &settings).expect("parse failed");
        let decoded = image.decode_native().expect("decode failed");

        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.data, original, "round-trip mismatch");
    }

    #[test]
    fn test_encode_decode_roundtrip_gray_8bit_single_dwt_level() {
        use crate::{DecodeSettings, Image};

        let original: Vec<u8> = (0..64 * 64)
            .map(|value| ((value * 37 + value / 7) & 0xFF) as u8)
            .collect();
        let encoded = encode(
            &original,
            64,
            64,
            1,
            8,
            false,
            &EncodeOptions {
                num_decomposition_levels: 1,
                reversible: true,
                ..Default::default()
            },
        )
        .expect("encode failed");

        let image = Image::new(&encoded, &DecodeSettings::default()).expect("parse failed");
        let decoded = image.decode_native().expect("decode failed");

        assert_eq!(decoded.width, 64);
        assert_eq!(decoded.height, 64);
        assert_eq!(decoded.data, original, "round-trip mismatch");
    }
}
