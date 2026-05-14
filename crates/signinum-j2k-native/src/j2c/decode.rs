//! Decoding JPEG2000 code streams.
//!
//! This is the "core" module of the crate that orchestrates all
//! stages in such a way that a given codestream is decoded into its
//! component channels.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use super::bitplane::{BitPlaneDecodeBuffers, BitPlaneDecodeContext};
use super::build::{CodeBlock, Decomposition, Layer, Precinct, Segment, SubBand, SubBandType};
use super::codestream::{ComponentInfo, Header, ProgressionOrder, QuantizationStyle};
use super::ht_block_decode::{self, HtBlockDecodeContext};
use super::idwt::IDWTOutput;
use super::progression::{
    component_position_resolution_layer_progression,
    layer_resolution_component_position_progression,
    position_component_resolution_layer_progression,
    resolution_layer_component_position_progression,
    resolution_position_component_layer_progression, IteratorInput, ProgressionData,
};
use super::roi::RoiPlan;
use super::tag_tree::TagNode;
use super::tile::{ComponentTile, ResolutionTile, Tile};
use super::{bitplane, build, idwt, mct, segment, tile, ComponentData};
use crate::error::{bail, ColorError, DecodingError, Result, TileError};
use crate::j2c::segment::MAX_BITPLANE_COUNT;
use crate::math::SimdBuffer;
use crate::profile;
use crate::reader::BitReader;
use crate::{
    decode_j2k_code_block_scalar, HtCodeBlockBatchJob, HtCodeBlockDecodeJob, HtCodeBlockDecoder,
    HtOwnedCodeBlockBatchJob, HtOwnedSubBandPlan, HtSubBandDecodeJob, J2kCodeBlockBatchJob,
    J2kCodeBlockDecodeJob, J2kCodeBlockSegment, J2kCodeBlockStyle, J2kDirectBandId,
    J2kDirectColorPlan, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kDirectIdwtStep,
    J2kDirectStoreStep, J2kOwnedCodeBlockBatchJob, J2kOwnedSubBandPlan, J2kRect,
    J2kStoreComponentJob, J2kSubBandDecodeJob, J2kSubBandType, J2kWaveletTransform,
};
use core::ops::{DerefMut, Range};

pub(crate) fn decode<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    ctx: &mut DecoderContext<'a>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
) -> Result<()> {
    let mut reader = BitReader::new(data);
    let profile_enabled = profile::profile_stages_enabled();
    let total_start = profile::profile_now(profile_enabled);
    let mut profile_timings = DecodeProfileTimings::default();
    let stage_start = profile::profile_now(profile_enabled);
    let tiles = tile::parse(&mut reader, header)?;
    profile_timings.parse_tiles_us += profile::elapsed_us(stage_start);

    if tiles.is_empty() {
        bail!(TileError::Invalid);
    }

    ctx.reset(header, &tiles[0]);
    let cpu_decode_parallelism = ctx.cpu_decode_parallelism;
    let (tile_ctx, storage) = (&mut ctx.tile_decode_context, &mut ctx.storage);

    for tile in &tiles {
        ltrace!(
            "tile {} rect [{},{} {}x{}]",
            tile.idx,
            tile.rect.x0,
            tile.rect.y0,
            tile.rect.width(),
            tile.rect.height(),
        );

        let iter_input = IteratorInput::new(tile);

        let progression_iterator: Box<dyn Iterator<Item = ProgressionData>> =
            match tile.progression_order {
                ProgressionOrder::LayerResolutionComponentPosition => {
                    Box::new(layer_resolution_component_position_progression(iter_input))
                }
                ProgressionOrder::ResolutionLayerComponentPosition => {
                    Box::new(resolution_layer_component_position_progression(iter_input))
                }
                ProgressionOrder::ResolutionPositionComponentLayer => Box::new(
                    resolution_position_component_layer_progression(iter_input)
                        .ok_or(DecodingError::InvalidProgressionIterator)?,
                ),
                ProgressionOrder::PositionComponentResolutionLayer => Box::new(
                    position_component_resolution_layer_progression(iter_input)
                        .ok_or(DecodingError::InvalidProgressionIterator)?,
                ),
                ProgressionOrder::ComponentPositionResolutionLayer => Box::new(
                    component_position_resolution_layer_progression(iter_input)
                        .ok_or(DecodingError::InvalidProgressionIterator)?,
                ),
            };

        decode_tile(
            tile,
            header,
            progression_iterator,
            tile_ctx,
            storage,
            ht_decoder,
            cpu_decode_parallelism,
            profile_enabled,
            &mut profile_timings,
        )?;
    }

    // Note that this assumes that either all tiles have MCT or none of them.
    // In theory, only some could have it... But hopefully no such cursed
    // images exist!
    if tiles[0].mct {
        let stage_start = profile::profile_now(profile_enabled);
        mct::apply_inverse(tile_ctx, &tiles[0].component_infos, header, ht_decoder)?;
        apply_sign_shift(tile_ctx, &header.component_infos);
        profile_timings.mct_us += profile::elapsed_us(stage_start);
    }

    if profile_enabled {
        profile::emit_profile_row(
            "decode",
            "cpu",
            &[
                ("parse_tiles_us", profile_timings.parse_tiles_us),
                ("build_us", profile_timings.build_us),
                ("segment_us", profile_timings.segment_us),
                ("codeblock_us", profile_timings.codeblock_us),
                ("idwt_us", profile_timings.idwt_us),
                ("store_us", profile_timings.store_us),
                ("mct_us", profile_timings.mct_us),
                ("total_us", profile::elapsed_us(total_start)),
            ],
        );
    }

    Ok(())
}

pub(crate) fn build_direct_grayscale_plan<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    ctx: &mut DecoderContext<'a>,
) -> Result<J2kDirectGrayscalePlan> {
    let mut reader = BitReader::new(data);
    let tiles = tile::parse(&mut reader, header)?;

    if tiles.len() != 1 {
        bail!(DecodingError::UnsupportedFeature(
            "direct grayscale plan only supports single-tile codestreams"
        ));
    }

    let tile = &tiles[0];
    if tile.component_infos.len() != 1 {
        bail!(DecodingError::UnsupportedFeature(
            "direct grayscale plan only supports single-component codestreams"
        ));
    }
    ctx.tile_decode_context.channel_data.clear();
    ctx.tile_decode_context.output_region = None;
    ctx.storage.reset();

    build::build(tile, &mut ctx.storage)?;

    let iter_input = IteratorInput::new(tile);
    let progression_iterator: Box<dyn Iterator<Item = ProgressionData>> =
        match tile.progression_order {
            ProgressionOrder::LayerResolutionComponentPosition => {
                Box::new(layer_resolution_component_position_progression(iter_input))
            }
            ProgressionOrder::ResolutionLayerComponentPosition => {
                Box::new(resolution_layer_component_position_progression(iter_input))
            }
            ProgressionOrder::ResolutionPositionComponentLayer => Box::new(
                resolution_position_component_layer_progression(iter_input)
                    .ok_or(DecodingError::InvalidProgressionIterator)?,
            ),
            ProgressionOrder::PositionComponentResolutionLayer => Box::new(
                position_component_resolution_layer_progression(iter_input)
                    .ok_or(DecodingError::InvalidProgressionIterator)?,
            ),
            ProgressionOrder::ComponentPositionResolutionLayer => Box::new(
                component_position_resolution_layer_progression(iter_input)
                    .ok_or(DecodingError::InvalidProgressionIterator)?,
            ),
        };
    segment::parse(tile, progression_iterator, header, &mut ctx.storage)?;

    let component_info = &tile.component_infos[0];
    build_component_plan_from_storage(
        tile,
        header,
        &ctx.storage,
        0,
        (1_u32 << (component_info.size_info.precision - 1)) as f32,
    )
}

pub(crate) fn build_direct_color_plan<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    ctx: &mut DecoderContext<'a>,
) -> Result<J2kDirectColorPlan> {
    let mut reader = BitReader::new(data);
    let tiles = tile::parse(&mut reader, header)?;

    if tiles.len() != 1 {
        bail!(DecodingError::UnsupportedFeature(
            "direct color plan only supports single-tile codestreams"
        ));
    }

    let tile = &tiles[0];
    if tile.component_infos.len() != 3 {
        bail!(DecodingError::UnsupportedFeature(
            "direct color plan only supports three-component RGB codestreams"
        ));
    }
    if header.skipped_resolution_levels != 0 {
        bail!(DecodingError::UnsupportedFeature(
            "direct color plan only supports full-resolution decode"
        ));
    }

    let transform = tile.component_infos[0].wavelet_transform();
    if tile.mct
        && (transform != tile.component_infos[1].wavelet_transform()
            || transform != tile.component_infos[2].wavelet_transform())
    {
        bail!(ColorError::Mct);
    }

    ctx.tile_decode_context.channel_data.clear();
    ctx.tile_decode_context.output_region = None;
    ctx.storage.reset();

    build::build(tile, &mut ctx.storage)?;

    let iter_input = IteratorInput::new(tile);
    let progression_iterator: Box<dyn Iterator<Item = ProgressionData>> =
        match tile.progression_order {
            ProgressionOrder::LayerResolutionComponentPosition => {
                Box::new(layer_resolution_component_position_progression(iter_input))
            }
            ProgressionOrder::ResolutionLayerComponentPosition => {
                Box::new(resolution_layer_component_position_progression(iter_input))
            }
            ProgressionOrder::ResolutionPositionComponentLayer => Box::new(
                resolution_position_component_layer_progression(iter_input)
                    .ok_or(DecodingError::InvalidProgressionIterator)?,
            ),
            ProgressionOrder::PositionComponentResolutionLayer => Box::new(
                position_component_resolution_layer_progression(iter_input)
                    .ok_or(DecodingError::InvalidProgressionIterator)?,
            ),
            ProgressionOrder::ComponentPositionResolutionLayer => Box::new(
                component_position_resolution_layer_progression(iter_input)
                    .ok_or(DecodingError::InvalidProgressionIterator)?,
            ),
        };
    segment::parse(tile, progression_iterator, header, &mut ctx.storage)?;

    let mut bit_depths = [0_u8; 3];
    let mut component_plans = Vec::with_capacity(3);
    for (component_idx, bit_depth) in bit_depths.iter_mut().enumerate() {
        let component_info = &tile.component_infos[component_idx];
        *bit_depth = component_info.size_info.precision;
        let addend = if tile.mct {
            0.0
        } else {
            (1_u32 << (component_info.size_info.precision - 1)) as f32
        };
        component_plans.push(build_component_plan_from_storage(
            tile,
            header,
            &ctx.storage,
            component_idx,
            addend,
        )?);
    }

    Ok(J2kDirectColorPlan {
        dimensions: (
            header.size_data.image_width(),
            header.size_data.image_height(),
        ),
        bit_depths,
        mct: tile.mct,
        transform: J2kWaveletTransform::from(transform),
        component_plans,
    })
}

fn build_component_plan_from_storage(
    tile: &Tile<'_>,
    header: &Header<'_>,
    storage: &DecompositionStorage<'_>,
    component_idx: usize,
    store_addend: f32,
) -> Result<J2kDirectGrayscalePlan> {
    let component_info =
        tile.component_infos
            .get(component_idx)
            .ok_or(DecodingError::UnsupportedFeature(
                "direct component plan index is out of range",
            ))?;
    if component_info.size_info.horizontal_resolution != 1
        || component_info.size_info.vertical_resolution != 1
    {
        bail!(DecodingError::UnsupportedFeature(
            "direct component plan only supports unit-sampled components"
        ));
    }

    let tile_decompositions =
        storage
            .tile_decompositions
            .get(component_idx)
            .ok_or(DecodingError::UnsupportedFeature(
                "direct component decomposition index is out of range",
            ))?;
    let mut steps = Vec::new();
    let mut next_band_id: J2kDirectBandId = 0;
    let mut sub_band_ids = vec![None; storage.sub_bands.len()];

    for resolution in 0..component_info.num_resolution_levels() - header.skipped_resolution_levels {
        let sub_band_iter = tile_decompositions.sub_band_iter(resolution, &storage.decompositions);
        for sub_band_idx in sub_band_iter {
            if let Some(step) = build_grayscale_sub_band_step(
                &storage.sub_bands[sub_band_idx],
                next_band_id,
                resolution,
                component_info,
                storage,
                header,
            )? {
                sub_band_ids[sub_band_idx] = Some(next_band_id);
                next_band_id = next_band_id
                    .checked_add(1)
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                steps.push(step);
            }
        }
    }

    let mut current_ll_rect = storage.sub_bands[tile_decompositions.first_ll_sub_band].rect;
    let mut current_ll_band_id = sub_band_ids[tile_decompositions.first_ll_sub_band]
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let decompositions = &storage.decompositions[tile_decompositions.decompositions.clone()];
    let decompositions = &decompositions[..decompositions
        .len()
        .saturating_sub(header.skipped_resolution_levels as usize)];
    for decomposition in decompositions {
        let hl = &storage.sub_bands[decomposition.sub_bands[0]];
        let lh = &storage.sub_bands[decomposition.sub_bands[1]];
        let hh = &storage.sub_bands[decomposition.sub_bands[2]];
        let output_band_id = next_band_id;
        next_band_id = next_band_id
            .checked_add(1)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        steps.push(J2kDirectGrayscaleStep::Idwt(J2kDirectIdwtStep {
            output_band_id,
            rect: J2kRect::from(decomposition.rect),
            transform: J2kWaveletTransform::from(component_info.wavelet_transform()),
            ll_band_id: current_ll_band_id,
            ll: J2kRect::from(current_ll_rect),
            hl_band_id: sub_band_ids[decomposition.sub_bands[0]]
                .ok_or(DecodingError::CodeBlockDecodeFailure)?,
            hl: J2kRect::from(hl.rect),
            lh_band_id: sub_band_ids[decomposition.sub_bands[1]]
                .ok_or(DecodingError::CodeBlockDecodeFailure)?,
            lh: J2kRect::from(lh.rect),
            hh_band_id: sub_band_ids[decomposition.sub_bands[2]]
                .ok_or(DecodingError::CodeBlockDecodeFailure)?,
            hh: J2kRect::from(hh.rect),
        }));
        current_ll_rect = decomposition.rect;
        current_ll_band_id = output_band_id;
    }

    let component_tile = ComponentTile::new(tile, component_info);
    let resolution_tile = ResolutionTile::new(
        component_tile,
        component_info.num_resolution_levels() - 1 - header.skipped_resolution_levels,
    );
    let image_x_offset = header.size_data.image_area_x_offset;
    let image_y_offset = header.size_data.image_area_y_offset;
    let source_x = image_x_offset.saturating_sub(current_ll_rect.x0);
    let source_y = image_y_offset.saturating_sub(current_ll_rect.y0);
    let copy_width = resolution_tile
        .rect
        .width()
        .min(current_ll_rect.width().saturating_sub(source_x));
    let copy_height = resolution_tile
        .rect
        .height()
        .min(current_ll_rect.height().saturating_sub(source_y));
    let output_x = resolution_tile.rect.x0.saturating_sub(image_x_offset);
    let output_y = resolution_tile.rect.y0.saturating_sub(image_y_offset);
    steps.push(J2kDirectGrayscaleStep::Store(J2kDirectStoreStep {
        input_band_id: current_ll_band_id,
        input_rect: J2kRect::from(current_ll_rect),
        source_x,
        source_y,
        copy_width,
        copy_height,
        output_width: header.size_data.image_width(),
        output_height: header.size_data.image_height(),
        output_x,
        output_y,
        addend: store_addend,
    }));

    Ok(J2kDirectGrayscalePlan {
        dimensions: (
            header.size_data.image_width(),
            header.size_data.image_height(),
        ),
        bit_depth: component_info.size_info.precision,
        steps,
    })
}

fn build_grayscale_sub_band_step(
    sub_band: &SubBand,
    band_id: J2kDirectBandId,
    resolution: u8,
    component_info: &ComponentInfo,
    storage: &DecompositionStorage<'_>,
    header: &Header<'_>,
) -> Result<Option<J2kDirectGrayscaleStep>> {
    let dequantization_step = {
        if component_info.quantization_info.quantization_style == QuantizationStyle::NoQuantization
        {
            1.0
        } else {
            let (exponent, mantissa) =
                component_info.exponent_mantissa(sub_band.sub_band_type, resolution)?;

            let r_b = {
                let log_gain = match sub_band.sub_band_type {
                    SubBandType::LowLow => 0,
                    SubBandType::LowHigh => 1,
                    SubBandType::HighLow => 1,
                    SubBandType::HighHigh => 2,
                };

                component_info.size_info.precision as u16 + log_gain
            };

            crate::math::pow2i(r_b as i32 - exponent as i32) * (1.0 + (mantissa as f32) / 2048.0)
        }
    };

    let num_bitplanes = {
        let (exponent, _) = component_info.exponent_mantissa(sub_band.sub_band_type, resolution)?;
        let num_bitplanes = (component_info.quantization_info.guard_bits as u16)
            .checked_add(exponent)
            .and_then(|x| x.checked_sub(1))
            .ok_or(DecodingError::InvalidBitplaneCount)?;

        if num_bitplanes > MAX_BITPLANE_COUNT as u16 {
            bail!(DecodingError::TooManyBitplanes);
        }

        num_bitplanes as u8
    };

    if component_info
        .coding_style
        .parameters
        .code_block_style
        .uses_high_throughput_block_coding()
    {
        let stripe_causal = component_info
            .coding_style
            .parameters
            .code_block_style
            .vertically_causal_context;
        let mut jobs = Vec::new();
        for precinct in sub_band
            .precincts
            .clone()
            .map(|idx| &storage.precincts[idx])
        {
            for code_block in precinct
                .code_blocks
                .clone()
                .map(|idx| &storage.code_blocks[idx])
            {
                let actual_bitplanes = if header.strict {
                    num_bitplanes
                        .checked_sub(code_block.missing_bit_planes)
                        .ok_or(DecodingError::InvalidBitplaneCount)?
                } else {
                    num_bitplanes.saturating_sub(code_block.missing_bit_planes)
                };
                let max_coding_passes = if actual_bitplanes == 0 {
                    0
                } else {
                    1 + 3 * (actual_bitplanes - 1)
                };
                if code_block.number_of_coding_passes > max_coding_passes && header.strict {
                    bail!(DecodingError::TooManyCodingPasses);
                }
                if code_block.number_of_coding_passes == 0 || actual_bitplanes == 0 {
                    continue;
                }

                let combined = ht_block_decode::collect_code_block_data(code_block, storage)?;
                jobs.push(HtOwnedCodeBlockBatchJob {
                    output_x: code_block.rect.x0 - sub_band.rect.x0,
                    output_y: code_block.rect.y0 - sub_band.rect.y0,
                    data: combined.data,
                    cleanup_length: combined.cleanup_length,
                    refinement_length: combined.refinement_length,
                    width: code_block.rect.width(),
                    height: code_block.rect.height(),
                    output_stride: sub_band.rect.width() as usize,
                    missing_bit_planes: code_block.missing_bit_planes,
                    number_of_coding_passes: code_block.number_of_coding_passes,
                    num_bitplanes,
                    stripe_causal,
                    strict: header.strict,
                    dequantization_step,
                });
            }
        }

        return Ok(Some(J2kDirectGrayscaleStep::HtSubBand(
            HtOwnedSubBandPlan {
                band_id,
                rect: J2kRect::from(sub_band.rect),
                width: sub_band.rect.width(),
                height: sub_band.rect.height(),
                jobs,
            },
        )));
    }

    let classic_job_sub_band_type = match sub_band.sub_band_type {
        SubBandType::LowLow => J2kSubBandType::LowLow,
        SubBandType::HighLow => J2kSubBandType::HighLow,
        SubBandType::LowHigh => J2kSubBandType::LowHigh,
        SubBandType::HighHigh => J2kSubBandType::HighHigh,
    };
    let classic_job_style = J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: component_info
            .coding_style
            .parameters
            .code_block_style
            .selective_arithmetic_coding_bypass,
        reset_context_probabilities: component_info
            .coding_style
            .parameters
            .code_block_style
            .reset_context_probabilities,
        termination_on_each_pass: component_info
            .coding_style
            .parameters
            .code_block_style
            .termination_on_each_pass,
        vertically_causal_context: component_info
            .coding_style
            .parameters
            .code_block_style
            .vertically_causal_context,
        segmentation_symbols: component_info
            .coding_style
            .parameters
            .code_block_style
            .segmentation_symbols,
    };

    let mut jobs = Vec::new();
    for precinct in sub_band
        .precincts
        .clone()
        .map(|idx| &storage.precincts[idx])
    {
        for code_block in precinct
            .code_blocks
            .clone()
            .map(|idx| &storage.code_blocks[idx])
        {
            let (combined_data, segments) = collect_classic_code_block_data(
                code_block,
                &component_info.coding_style.parameters.code_block_style,
                storage,
            )?;
            jobs.push(J2kOwnedCodeBlockBatchJob {
                output_x: code_block.rect.x0 - sub_band.rect.x0,
                output_y: code_block.rect.y0 - sub_band.rect.y0,
                data: combined_data,
                segments,
                width: code_block.rect.width(),
                height: code_block.rect.height(),
                output_stride: sub_band.rect.width() as usize,
                missing_bit_planes: code_block.missing_bit_planes,
                number_of_coding_passes: code_block.number_of_coding_passes,
                total_bitplanes: num_bitplanes,
                sub_band_type: classic_job_sub_band_type,
                style: classic_job_style,
                strict: header.strict,
                dequantization_step,
            });
        }
    }

    Ok(Some(J2kDirectGrayscaleStep::ClassicSubBand(
        J2kOwnedSubBandPlan {
            band_id,
            rect: J2kRect::from(sub_band.rect),
            width: sub_band.rect.width(),
            height: sub_band.rect.height(),
            jobs,
        },
    )))
}

fn collect_classic_code_block_data(
    code_block: &CodeBlock,
    style: &super::codestream::CodeBlockStyle,
    storage: &DecompositionStorage<'_>,
) -> Result<(Vec<u8>, Vec<J2kCodeBlockSegment>)> {
    let mut combined_data = Vec::new();
    let mut collected_segments = Vec::new();
    let mut last_segment_idx = 0u8;
    let mut segment_start_offset = 0usize;
    let mut segment_start_coding_pass = 0u8;
    let mut coding_passes = 0u8;
    let is_normal_mode =
        !style.selective_arithmetic_coding_bypass && !style.termination_on_each_pass;

    for layer in &storage.layers[code_block.layers.start..code_block.layers.end] {
        let Some(range) = layer.segments.clone() else {
            continue;
        };

        for segment in &storage.segments[range] {
            if segment.idx != last_segment_idx {
                if segment.idx != last_segment_idx + 1 {
                    bail!(DecodingError::CodeBlockDecodeFailure);
                }
                if coding_passes > segment_start_coding_pass
                    || combined_data.len() > segment_start_offset
                {
                    let data_offset = u32::try_from(segment_start_offset)
                        .map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
                    let data_length = u32::try_from(combined_data.len() - segment_start_offset)
                        .map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
                    let use_arithmetic = if style.selective_arithmetic_coding_bypass {
                        if segment_start_coding_pass <= 9 {
                            true
                        } else {
                            segment_start_coding_pass.is_multiple_of(3)
                        }
                    } else {
                        true
                    };
                    collected_segments.push(J2kCodeBlockSegment {
                        data_offset,
                        data_length,
                        start_coding_pass: segment_start_coding_pass,
                        end_coding_pass: coding_passes,
                        use_arithmetic,
                    });
                }
                segment_start_offset = combined_data.len();
                segment_start_coding_pass = coding_passes;
                last_segment_idx += 1;
            }

            combined_data.extend_from_slice(segment.data);
            coding_passes = coding_passes.saturating_add(segment.coding_pases);
        }
    }

    if coding_passes > segment_start_coding_pass || combined_data.len() > segment_start_offset {
        let data_offset = u32::try_from(segment_start_offset)
            .map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
        let data_length = u32::try_from(combined_data.len().saturating_sub(segment_start_offset))
            .map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
        let use_arithmetic = if style.selective_arithmetic_coding_bypass {
            if segment_start_coding_pass <= 9 {
                true
            } else {
                segment_start_coding_pass.is_multiple_of(3)
            }
        } else {
            true
        };
        collected_segments.push(J2kCodeBlockSegment {
            data_offset,
            data_length,
            start_coding_pass: segment_start_coding_pass,
            end_coding_pass: coding_passes,
            use_arithmetic,
        });
    }

    if is_normal_mode {
        collected_segments.clear();
        collected_segments.push(J2kCodeBlockSegment {
            data_offset: 0,
            data_length: u32::try_from(combined_data.len())
                .map_err(|_| DecodingError::CodeBlockDecodeFailure)?,
            start_coding_pass: 0,
            end_coding_pass: coding_passes,
            use_arithmetic: true,
        });
    }

    if coding_passes != code_block.number_of_coding_passes {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }

    Ok((combined_data, collected_segments))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OutputRegion {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl OutputRegion {
    pub(crate) fn from_tuple(region: (u32, u32, u32, u32)) -> Self {
        let (x, y, width, height) = region;
        Self {
            x,
            y,
            width,
            height,
        }
    }

    fn dimensions(self) -> (u32, u32) {
        (self.width, self.height)
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecodeDebugCounters {
    pub(crate) decoded_code_blocks: usize,
    pub(crate) skipped_code_blocks: usize,
    pub(crate) idwt_output_samples: usize,
}

/// CPU parallelism policy for native JPEG 2000 decode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CpuDecodeParallelism {
    /// Allow a single tile decode to use internal code-block parallelism.
    #[default]
    Auto,
    /// Keep code-block decode serial for callers that already parallelize tiles.
    Serial,
}

/// A decoder context for decoding JPEG2000 images.
pub struct DecoderContext<'a> {
    pub(crate) tile_decode_context: TileDecodeContext,
    storage: DecompositionStorage<'a>,
    cpu_decode_parallelism: CpuDecodeParallelism,
}

impl Default for DecoderContext<'_> {
    fn default() -> Self {
        Self {
            tile_decode_context: TileDecodeContext::default(),
            storage: DecompositionStorage::default(),
            cpu_decode_parallelism: CpuDecodeParallelism::Auto,
        }
    }
}

impl DecoderContext<'_> {
    fn reset(&mut self, header: &Header<'_>, initial_tile: &Tile<'_>) {
        self.tile_decode_context.reset(header, initial_tile);
        self.storage.reset();
    }

    pub(crate) fn set_output_region(&mut self, output_region: Option<(u32, u32, u32, u32)>) {
        self.tile_decode_context.output_region = output_region.map(OutputRegion::from_tuple);
    }

    /// Return the native CPU decode parallelism policy.
    pub fn cpu_decode_parallelism(&self) -> CpuDecodeParallelism {
        self.cpu_decode_parallelism
    }

    /// Set the native CPU decode parallelism policy.
    pub fn set_cpu_decode_parallelism(&mut self, parallelism: CpuDecodeParallelism) {
        self.cpu_decode_parallelism = parallelism;
    }
}

fn decode_tile<'a, 'b>(
    tile: &'b Tile<'a>,
    header: &Header<'_>,
    progression_iterator: Box<dyn Iterator<Item = ProgressionData> + '_>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    cpu_decode_parallelism: CpuDecodeParallelism,
    profile_enabled: bool,
    profile_timings: &mut DecodeProfileTimings,
) -> Result<()> {
    storage.reset();

    // This is the method that orchestrates all steps.

    // First, we build the decompositions, including their sub-bands, precincts
    // and code blocks.
    let stage_start = profile::profile_now(profile_enabled);
    build::build(tile, storage)?;
    if let Some(output_region) = tile_ctx.output_region {
        storage.roi_plan = RoiPlan::build(tile, header, storage, output_region);
        if storage.roi_plan.is_some() {
            storage.coefficients.fill(0.0);
        }
    }
    profile_timings.build_us += profile::elapsed_us(stage_start);
    // Next, we parse the layers/segments for each code block.
    let stage_start = profile::profile_now(profile_enabled);
    segment::parse(tile, progression_iterator, header, storage)?;
    profile_timings.segment_us += profile::elapsed_us(stage_start);
    // We then decode the bitplanes of each code block, yielding the
    // (possibly dequantized) coefficients of each code block.
    let stage_start = profile::profile_now(profile_enabled);
    decode_component_tile_bit_planes(
        tile,
        tile_ctx,
        storage,
        header,
        ht_decoder,
        cpu_decode_parallelism,
    )?;
    profile_timings.codeblock_us += profile::elapsed_us(stage_start);

    // Unlike before, we interleave the apply_idwt and store stages
    // for each component tile so we can reuse allocations better.
    for (idx, component_info) in header.component_infos.iter().enumerate() {
        // Next, we apply the inverse discrete wavelet transform.
        let stage_start = profile::profile_now(profile_enabled);
        idwt::apply(
            storage,
            tile_ctx,
            idx,
            header,
            component_info.wavelet_transform(),
            ht_decoder,
        )?;
        profile_timings.idwt_us += profile::elapsed_us(stage_start);
        // Finally, we store the raw samples for the tile area in the correct
        // location. Note that in case we have MCT, we are not applying it yet.
        // It will be applied in the very end once all tiles have been processed.
        // The reason we do this is that applying MCT requires access to the
        // data from _all_ components. If we didn't defer this until the end
        // we would have to collect the IDWT outputs of all components before
        // applying it. By not applying MCT here, we can get away with doing
        // IDWT and store on a per-component basis. Thus, we only need to
        // store one IDWT output at a time, allowing for better reuse of
        // allocations.
        let stage_start = profile::profile_now(profile_enabled);
        store(tile, header, tile_ctx, component_info, idx, ht_decoder)?;
        profile_timings.store_us += profile::elapsed_us(stage_start);
    }

    Ok(())
}

#[derive(Default)]
struct DecodeProfileTimings {
    parse_tiles_us: u128,
    build_us: u128,
    segment_us: u128,
    codeblock_us: u128,
    idwt_us: u128,
    store_us: u128,
    mct_us: u128,
}

/// All decompositions for a single tile.
#[derive(Clone)]
pub(crate) struct TileDecompositions {
    pub(crate) first_ll_sub_band: usize,
    pub(crate) decompositions: Range<usize>,
}

impl TileDecompositions {
    pub(crate) fn sub_band_iter(
        &self,
        resolution: u8,
        decompositions: &[Decomposition],
    ) -> SubBandIter {
        let indices = if resolution == 0 {
            [
                self.first_ll_sub_band,
                self.first_ll_sub_band,
                self.first_ll_sub_band,
            ]
        } else {
            decompositions[self.decompositions.clone()][resolution as usize - 1].sub_bands
        };

        SubBandIter {
            next_idx: 0,
            indices,
            resolution,
        }
    }
}

#[derive(Clone)]
pub(crate) struct SubBandIter {
    resolution: u8,
    next_idx: usize,
    indices: [usize; 3],
}

impl Iterator for SubBandIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let value = if self.resolution == 0 {
            if self.next_idx > 0 {
                None
            } else {
                Some(self.indices[0])
            }
        } else if self.next_idx >= self.indices.len() {
            None
        } else {
            Some(self.indices[self.next_idx])
        };

        self.next_idx += 1;

        value
    }
}

/// A buffer so that we can reuse allocations for layers/code blocks/etc.
/// across different tiles.
#[derive(Default)]
pub(crate) struct DecompositionStorage<'a> {
    pub(crate) segments: Vec<Segment<'a>>,
    pub(crate) layers: Vec<Layer>,
    pub(crate) code_blocks: Vec<CodeBlock>,
    pub(crate) precincts: Vec<Precinct>,
    pub(crate) tag_tree_nodes: Vec<TagNode>,
    pub(crate) coefficients: Vec<f32>,
    pub(crate) sub_bands: Vec<SubBand>,
    pub(crate) decompositions: Vec<Decomposition>,
    pub(crate) tile_decompositions: Vec<TileDecompositions>,
    pub(crate) roi_plan: Option<RoiPlan>,
}

impl DecompositionStorage<'_> {
    fn reset(&mut self) {
        self.segments.clear();
        self.layers.clear();
        self.code_blocks.clear();
        // No need to clear the coefficients, as they will be resized
        // and then overridden.
        // self.coefficients.clear();
        self.precincts.clear();
        self.sub_bands.clear();
        self.decompositions.clear();
        self.tile_decompositions.clear();
        self.tag_tree_nodes.clear();
        self.roi_plan = None;
    }
}

/// A reusable context used during the decoding of a single tile.
///
/// Some of the fields are temporary in nature and reset after moving on to the
/// next tile, some contain global state.
#[derive(Default)]
pub(crate) struct TileDecodeContext {
    /// A reusable buffer for the IDWT output.
    pub(crate) idwt_output: IDWTOutput,
    /// A scratch buffer used during IDWT.
    pub(crate) idwt_scratch_buffer: Vec<f32>,
    /// A reusable context for decoding code blocks.
    pub(crate) bit_plane_decode_context: BitPlaneDecodeContext,
    /// Reusable buffers for decoding bitplanes.
    pub(crate) bit_plane_decode_buffers: BitPlaneDecodeBuffers,
    /// A reusable context for decoding HTJ2K code blocks.
    pub(crate) ht_block_decode_context: HtBlockDecodeContext,
    /// The raw, decoded samples for each channel.
    pub(crate) channel_data: Vec<ComponentData>,
    /// Optional output window for region-local decode storage.
    pub(crate) output_region: Option<OutputRegion>,
    /// Debug counters for tests and ROI instrumentation.
    pub(crate) debug_counters: DecodeDebugCounters,
}

impl TileDecodeContext {
    /// Reset the context for processing a new image.
    fn reset(&mut self, header: &Header<'_>, initial_tile: &Tile<'_>) {
        // Bitplane decode context and buffers will be reset in the
        // corresponding methods. IDWT output and scratch buffer will be
        // overridden on demand, so those don't need to be reset either.
        self.channel_data.clear();
        self.debug_counters = DecodeDebugCounters::default();

        let (output_width, output_height) =
            self.output_region.map(OutputRegion::dimensions).unwrap_or((
                header.size_data.image_width(),
                header.size_data.image_height(),
            ));

        // TODO: SIMD Buffers should be reused across runs!
        for info in &initial_tile.component_infos {
            self.channel_data.push(ComponentData {
                container: SimdBuffer::zeros(output_width as usize * output_height as usize),
                bit_depth: info.size_info.precision,
            });
        }
    }
}

fn decode_component_tile_bit_planes<'a>(
    tile: &Tile<'a>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    header: &Header<'_>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    cpu_decode_parallelism: CpuDecodeParallelism,
) -> Result<()> {
    for (tile_decompositions_idx, component_info) in tile.component_infos.iter().enumerate() {
        // Only decode the resolution levels we actually care about.
        for resolution in
            0..component_info.num_resolution_levels() - header.skipped_resolution_levels
        {
            let tile_composition = &storage.tile_decompositions[tile_decompositions_idx];
            let sub_band_iter = tile_composition.sub_band_iter(resolution, &storage.decompositions);

            for sub_band_idx in sub_band_iter {
                decode_sub_band_bitplanes(
                    sub_band_idx,
                    resolution,
                    component_info,
                    tile_ctx,
                    storage,
                    header,
                    ht_decoder,
                    cpu_decode_parallelism,
                )?;
            }
        }
    }

    Ok(())
}

fn decode_sub_band_bitplanes(
    sub_band_idx: usize,
    resolution: u8,
    component_info: &ComponentInfo,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'_>,
    header: &Header<'_>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    cpu_decode_parallelism: CpuDecodeParallelism,
) -> Result<()> {
    let sub_band = storage.sub_bands[sub_band_idx].clone();

    let dequantization_step = {
        if component_info.quantization_info.quantization_style == QuantizationStyle::NoQuantization
        {
            1.0
        } else {
            let (exponent, mantissa) =
                component_info.exponent_mantissa(sub_band.sub_band_type, resolution)?;

            let r_b = {
                let log_gain = match sub_band.sub_band_type {
                    SubBandType::LowLow => 0,
                    SubBandType::LowHigh => 1,
                    SubBandType::HighLow => 1,
                    SubBandType::HighHigh => 2,
                };

                component_info.size_info.precision as u16 + log_gain
            };

            crate::math::pow2i(r_b as i32 - exponent as i32) * (1.0 + (mantissa as f32) / 2048.0)
        }
    };

    let num_bitplanes = {
        let (exponent, _) = component_info.exponent_mantissa(sub_band.sub_band_type, resolution)?;
        // Equation (E-2)
        let num_bitplanes = (component_info.quantization_info.guard_bits as u16)
            .checked_add(exponent)
            .and_then(|x| x.checked_sub(1))
            .ok_or(DecodingError::InvalidBitplaneCount)?;

        if num_bitplanes > MAX_BITPLANE_COUNT as u16 {
            bail!(DecodingError::TooManyBitplanes);
        }

        num_bitplanes as u8
    };

    if component_info
        .coding_style
        .parameters
        .code_block_style
        .uses_high_throughput_block_coding()
    {
        decode_sub_band_ht_blocks(
            sub_band_idx,
            &sub_band,
            component_info,
            tile_ctx,
            storage,
            header,
            ht_decoder,
            num_bitplanes,
            dequantization_step,
        )?;
        return Ok(());
    }

    let classic_job_sub_band_type = match sub_band.sub_band_type {
        SubBandType::LowLow => J2kSubBandType::LowLow,
        SubBandType::HighLow => J2kSubBandType::HighLow,
        SubBandType::LowHigh => J2kSubBandType::LowHigh,
        SubBandType::HighHigh => J2kSubBandType::HighHigh,
    };
    let classic_job_style = J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: component_info
            .coding_style
            .parameters
            .code_block_style
            .selective_arithmetic_coding_bypass,
        reset_context_probabilities: component_info
            .coding_style
            .parameters
            .code_block_style
            .reset_context_probabilities,
        termination_on_each_pass: component_info
            .coding_style
            .parameters
            .code_block_style
            .termination_on_each_pass,
        vertically_causal_context: component_info
            .coding_style
            .parameters
            .code_block_style
            .vertically_causal_context,
        segmentation_symbols: component_info
            .coding_style
            .parameters
            .code_block_style
            .segmentation_symbols,
    };

    if let Some(ht_decoder) = ht_decoder.as_deref_mut() {
        let pending_blocks =
            collect_pending_classic_blocks(sub_band_idx, &sub_band, component_info, storage)?;

        let batch_jobs: Vec<_> = pending_blocks
            .iter()
            .map(|pending| J2kCodeBlockBatchJob {
                output_x: pending.output_x,
                output_y: pending.output_y,
                code_block: J2kCodeBlockDecodeJob {
                    data: &pending.combined_data,
                    segments: &pending.segments,
                    width: pending.width,
                    height: pending.height,
                    output_stride: sub_band.rect.width() as usize,
                    missing_bit_planes: pending.missing_bit_planes,
                    number_of_coding_passes: pending.number_of_coding_passes,
                    total_bitplanes: num_bitplanes,
                    sub_band_type: classic_job_sub_band_type,
                    style: classic_job_style,
                    strict: header.strict,
                    dequantization_step,
                },
            })
            .collect();

        let base_store = &mut storage.coefficients[sub_band.coefficients.clone()];
        if ht_decoder.decode_j2k_sub_band(
            J2kSubBandDecodeJob {
                width: sub_band.rect.width(),
                height: sub_band.rect.height(),
                jobs: &batch_jobs,
            },
            base_store,
        )? {
            tile_ctx.debug_counters.decoded_code_blocks += batch_jobs.len();
            return Ok(());
        }

        let output_stride = sub_band.rect.width() as usize;
        for job in batch_jobs {
            tile_ctx.debug_counters.decoded_code_blocks += 1;
            let base_idx = (job.output_y * sub_band.rect.width()) as usize + job.output_x as usize;
            let output_len = if job.code_block.height == 0 {
                0
            } else {
                output_stride
                    .checked_mul(job.code_block.height as usize - 1)
                    .and_then(|prefix| prefix.checked_add(job.code_block.width as usize))
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?
            };
            let output_slice = &mut base_store[base_idx..base_idx + output_len];
            if ht_decoder.decode_j2k_code_block(job.code_block, output_slice)? {
                continue;
            }
            decode_j2k_code_block_scalar(job.code_block, output_slice)?;
        }

        return Ok(());
    }

    let code_block_count = count_classic_code_blocks(sub_band_idx, &sub_band, storage);
    if should_decode_classic_sub_band_in_parallel(cpu_decode_parallelism, code_block_count) {
        #[cfg(feature = "parallel")]
        {
            let pending_blocks =
                collect_pending_classic_blocks(sub_band_idx, &sub_band, component_info, storage)?;
            let decoded_blocks = decode_classic_sub_band_blocks_parallel(
                &pending_blocks,
                classic_job_sub_band_type,
                classic_job_style,
                header.strict,
                num_bitplanes,
                dequantization_step,
            )?;
            tile_ctx.debug_counters.decoded_code_blocks += decoded_blocks.len();
            copy_decoded_classic_blocks_to_sub_band(&decoded_blocks, &sub_band, storage)?;
            return Ok(());
        }
    }

    for precinct in sub_band
        .precincts
        .clone()
        .map(|idx| &storage.precincts[idx])
    {
        for code_block in precinct
            .code_blocks
            .clone()
            .map(|idx| &storage.code_blocks[idx])
        {
            if !code_block_required_by_index(storage, sub_band_idx, code_block) {
                tile_ctx.debug_counters.skipped_code_blocks += 1;
                continue;
            }
            tile_ctx.debug_counters.decoded_code_blocks += 1;
            let x_offset = code_block.rect.x0 - sub_band.rect.x0;
            let y_offset = code_block.rect.y0 - sub_band.rect.y0;
            let output_stride = sub_band.rect.width() as usize;
            let base_idx = (y_offset * sub_band.rect.width()) as usize + x_offset as usize;

            bitplane::decode(
                code_block,
                sub_band.sub_band_type,
                num_bitplanes,
                &component_info.coding_style.parameters.code_block_style,
                tile_ctx,
                storage,
                header.strict,
            )?;

            let base_store = &mut storage.coefficients[sub_band.coefficients.clone()];
            let mut base_idx = base_idx;

            for coefficients in tile_ctx.bit_plane_decode_context.coefficient_rows() {
                let out_row = &mut base_store[base_idx..];

                for (output, coefficient) in out_row.iter_mut().zip(coefficients.iter().copied()) {
                    *output = coefficient.get() as f32;
                    *output *= dequantization_step;
                }

                base_idx += output_stride;
            }
        }
    }

    Ok(())
}

struct PendingHtBlock {
    combined: ht_block_decode::CombinedCodeBlockData,
    output_x: u32,
    output_y: u32,
    width: u32,
    height: u32,
    missing_bit_planes: u8,
    number_of_coding_passes: u8,
}

struct PendingClassicBlock {
    combined_data: Vec<u8>,
    segments: Vec<J2kCodeBlockSegment>,
    output_x: u32,
    output_y: u32,
    width: u32,
    height: u32,
    missing_bit_planes: u8,
    number_of_coding_passes: u8,
}

#[cfg(feature = "parallel")]
struct DecodedClassicBlock {
    output_x: u32,
    output_y: u32,
    width: u32,
    height: u32,
    coefficients: Vec<f32>,
}

fn count_classic_code_blocks(
    sub_band_idx: usize,
    sub_band: &SubBand,
    storage: &DecompositionStorage<'_>,
) -> usize {
    sub_band
        .precincts
        .clone()
        .map(|idx| &storage.precincts[idx])
        .map(|precinct| {
            precinct
                .code_blocks
                .clone()
                .filter(|idx| {
                    let code_block = &storage.code_blocks[*idx];
                    code_block_required_by_index(storage, sub_band_idx, code_block)
                })
                .count()
        })
        .sum()
}

fn code_block_required_by_index(
    storage: &DecompositionStorage<'_>,
    sub_band_idx: usize,
    code_block: &CodeBlock,
) -> bool {
    storage
        .roi_plan
        .as_ref()
        .is_none_or(|plan| plan.code_block_required(sub_band_idx, code_block.rect))
}

fn collect_pending_classic_blocks(
    sub_band_idx: usize,
    sub_band: &SubBand,
    component_info: &ComponentInfo,
    storage: &DecompositionStorage<'_>,
) -> Result<Vec<PendingClassicBlock>> {
    let mut pending_blocks =
        Vec::with_capacity(count_classic_code_blocks(sub_band_idx, sub_band, storage));
    for precinct in sub_band
        .precincts
        .clone()
        .map(|idx| &storage.precincts[idx])
    {
        for code_block in precinct
            .code_blocks
            .clone()
            .map(|idx| &storage.code_blocks[idx])
        {
            if !code_block_required_by_index(storage, sub_band_idx, code_block) {
                continue;
            }
            let (combined_data, segments) = collect_classic_code_block_data(
                code_block,
                &component_info.coding_style.parameters.code_block_style,
                storage,
            )?;
            pending_blocks.push(PendingClassicBlock {
                combined_data,
                segments,
                output_x: code_block.rect.x0 - sub_band.rect.x0,
                output_y: code_block.rect.y0 - sub_band.rect.y0,
                width: code_block.rect.width(),
                height: code_block.rect.height(),
                missing_bit_planes: code_block.missing_bit_planes,
                number_of_coding_passes: code_block.number_of_coding_passes,
            });
        }
    }
    Ok(pending_blocks)
}

pub(crate) fn should_decode_classic_sub_band_in_parallel(
    parallelism: CpuDecodeParallelism,
    code_block_count: usize,
) -> bool {
    cfg!(feature = "parallel") && parallelism == CpuDecodeParallelism::Auto && code_block_count >= 4
}

#[cfg(feature = "parallel")]
fn decode_classic_sub_band_blocks_parallel(
    pending_blocks: &[PendingClassicBlock],
    sub_band_type: J2kSubBandType,
    style: J2kCodeBlockStyle,
    strict: bool,
    total_bitplanes: u8,
    dequantization_step: f32,
) -> Result<Vec<DecodedClassicBlock>> {
    use rayon::prelude::*;

    pending_blocks
        .par_iter()
        .map(|pending| {
            let output_stride = pending.width as usize;
            let output_len = output_stride
                .checked_mul(pending.height as usize)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let mut coefficients = vec![0.0; output_len];
            decode_j2k_code_block_scalar(
                J2kCodeBlockDecodeJob {
                    data: &pending.combined_data,
                    segments: &pending.segments,
                    width: pending.width,
                    height: pending.height,
                    output_stride,
                    missing_bit_planes: pending.missing_bit_planes,
                    number_of_coding_passes: pending.number_of_coding_passes,
                    total_bitplanes,
                    sub_band_type,
                    style,
                    strict,
                    dequantization_step,
                },
                &mut coefficients,
            )?;
            Ok(DecodedClassicBlock {
                output_x: pending.output_x,
                output_y: pending.output_y,
                width: pending.width,
                height: pending.height,
                coefficients,
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .collect()
}

#[cfg(feature = "parallel")]
fn copy_decoded_classic_blocks_to_sub_band(
    decoded_blocks: &[DecodedClassicBlock],
    sub_band: &SubBand,
    storage: &mut DecompositionStorage<'_>,
) -> Result<()> {
    let sub_band_width = sub_band.rect.width() as usize;
    let base_store = &mut storage.coefficients[sub_band.coefficients.clone()];
    for block in decoded_blocks {
        if block
            .output_x
            .checked_add(block.width)
            .is_none_or(|x1| x1 > sub_band.rect.width())
            || block
                .output_y
                .checked_add(block.height)
                .is_none_or(|y1| y1 > sub_band.rect.height())
        {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
        let block_width = block.width as usize;
        for row in 0..block.height as usize {
            let dst_start = (block.output_y as usize + row)
                .checked_mul(sub_band_width)
                .and_then(|offset| offset.checked_add(block.output_x as usize))
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let dst_end = dst_start
                .checked_add(block_width)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let src_start = row
                .checked_mul(block_width)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let src_end = src_start
                .checked_add(block_width)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            base_store[dst_start..dst_end].copy_from_slice(&block.coefficients[src_start..src_end]);
        }
    }
    Ok(())
}

fn decode_sub_band_ht_blocks(
    sub_band_idx: usize,
    sub_band: &SubBand,
    component_info: &ComponentInfo,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'_>,
    header: &Header<'_>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    num_bitplanes: u8,
    dequantization_step: f32,
) -> Result<()> {
    let stripe_causal = component_info
        .coding_style
        .parameters
        .code_block_style
        .vertically_causal_context;

    if let Some(ht_decoder) = ht_decoder.as_deref_mut() {
        let mut pending_blocks = Vec::new();
        for precinct in sub_band
            .precincts
            .clone()
            .map(|idx| &storage.precincts[idx])
        {
            for code_block in precinct
                .code_blocks
                .clone()
                .map(|idx| &storage.code_blocks[idx])
            {
                if !code_block_required_by_index(storage, sub_band_idx, code_block) {
                    continue;
                }
                let actual_bitplanes = if header.strict {
                    num_bitplanes
                        .checked_sub(code_block.missing_bit_planes)
                        .ok_or(DecodingError::InvalidBitplaneCount)?
                } else {
                    num_bitplanes.saturating_sub(code_block.missing_bit_planes)
                };
                let max_coding_passes = if actual_bitplanes == 0 {
                    0
                } else {
                    1 + 3 * (actual_bitplanes - 1)
                };
                if code_block.number_of_coding_passes > max_coding_passes && header.strict {
                    bail!(DecodingError::TooManyCodingPasses);
                }
                if code_block.number_of_coding_passes == 0 || actual_bitplanes == 0 {
                    continue;
                }

                pending_blocks.push(PendingHtBlock {
                    combined: ht_block_decode::collect_code_block_data(code_block, storage)?,
                    output_x: code_block.rect.x0 - sub_band.rect.x0,
                    output_y: code_block.rect.y0 - sub_band.rect.y0,
                    width: code_block.rect.width(),
                    height: code_block.rect.height(),
                    missing_bit_planes: code_block.missing_bit_planes,
                    number_of_coding_passes: code_block.number_of_coding_passes,
                });
            }
        }

        let batch_jobs: Vec<_> = pending_blocks
            .iter()
            .map(|pending| HtCodeBlockBatchJob {
                output_x: pending.output_x,
                output_y: pending.output_y,
                code_block: HtCodeBlockDecodeJob {
                    data: &pending.combined.data,
                    cleanup_length: pending.combined.cleanup_length,
                    refinement_length: pending.combined.refinement_length,
                    width: pending.width,
                    height: pending.height,
                    output_stride: sub_band.rect.width() as usize,
                    missing_bit_planes: pending.missing_bit_planes,
                    number_of_coding_passes: pending.number_of_coding_passes,
                    num_bitplanes,
                    stripe_causal,
                    strict: header.strict,
                    dequantization_step,
                },
            })
            .collect();

        let base_store = &mut storage.coefficients[sub_band.coefficients.clone()];
        if ht_decoder.decode_sub_band(
            HtSubBandDecodeJob {
                width: sub_band.rect.width(),
                height: sub_band.rect.height(),
                jobs: &batch_jobs,
            },
            base_store,
        )? {
            tile_ctx.debug_counters.decoded_code_blocks += batch_jobs.len();
            return Ok(());
        }

        let output_stride = sub_band.rect.width() as usize;
        for job in batch_jobs {
            tile_ctx.debug_counters.decoded_code_blocks += 1;
            let base_idx = (job.output_y * sub_band.rect.width()) as usize + job.output_x as usize;
            let output_len = if job.code_block.height == 0 {
                0
            } else {
                output_stride * (job.code_block.height as usize - 1) + job.code_block.width as usize
            };
            ht_decoder.decode_code_block(
                job.code_block,
                &mut base_store[base_idx..base_idx + output_len],
            )?;
        }

        return Ok(());
    }

    for precinct in sub_band
        .precincts
        .clone()
        .map(|idx| &storage.precincts[idx])
    {
        for code_block in precinct
            .code_blocks
            .clone()
            .map(|idx| &storage.code_blocks[idx])
        {
            if !code_block_required_by_index(storage, sub_band_idx, code_block) {
                tile_ctx.debug_counters.skipped_code_blocks += 1;
                continue;
            }
            tile_ctx.debug_counters.decoded_code_blocks += 1;
            ht_block_decode::decode(
                code_block,
                num_bitplanes,
                stripe_causal,
                &mut tile_ctx.ht_block_decode_context,
                storage,
                header.strict,
            )?;

            let x_offset = code_block.rect.x0 - sub_band.rect.x0;
            let y_offset = code_block.rect.y0 - sub_band.rect.y0;
            let base_store = &mut storage.coefficients[sub_band.coefficients.clone()];
            let mut base_idx = (y_offset * sub_band.rect.width()) as usize + x_offset as usize;
            let output_stride = sub_band.rect.width() as usize;

            for coefficients in tile_ctx.ht_block_decode_context.coefficient_rows() {
                let out_row = &mut base_store[base_idx..];

                for (output, coefficient) in out_row.iter_mut().zip(coefficients.iter().copied()) {
                    *output =
                        ht_block_decode::coefficient_to_i32(coefficient, num_bitplanes) as f32;
                    *output *= dequantization_step;
                }

                base_idx += output_stride;
            }
        }
    }

    Ok(())
}

fn apply_sign_shift(tile_ctx: &mut TileDecodeContext, component_infos: &[ComponentInfo]) {
    for (channel_data, component_info) in
        tile_ctx.channel_data.iter_mut().zip(component_infos.iter())
    {
        let addend = (1_u32 << (component_info.size_info.precision - 1)) as f32;
        for sample in channel_data.container.deref_mut() {
            *sample += addend;
        }
    }
}

fn store<'a>(
    tile: &'a Tile<'a>,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    component_info: &ComponentInfo,
    component_idx: usize,
    backend: &mut Option<&mut dyn HtCodeBlockDecoder>,
) -> Result<()> {
    let channel_data = &mut tile_ctx.channel_data[component_idx];
    let idwt_output = &mut tile_ctx.idwt_output;

    let component_tile = ComponentTile::new(tile, component_info);
    let resolution_tile = ResolutionTile::new(
        component_tile,
        component_info.num_resolution_levels() - 1 - header.skipped_resolution_levels,
    );

    let sign_shift = if tile.mct {
        0.0
    } else {
        (1_u32 << (component_info.size_info.precision - 1)) as f32
    };

    let (scale_x, scale_y) = (
        component_info.size_info.horizontal_resolution,
        component_info.size_info.vertical_resolution,
    );

    let (image_x_offset, image_y_offset) = (
        header.size_data.image_area_x_offset,
        header.size_data.image_area_y_offset,
    );

    if let Some(output_region) = tile_ctx.output_region {
        store_region(
            tile,
            header,
            tile_ctx,
            component_info,
            component_idx,
            output_region,
            backend,
            sign_shift,
        )?;
        return Ok(());
    }

    if scale_x == 1 && scale_y == 1 {
        let source_x = image_x_offset.saturating_sub(idwt_output.rect.x0);
        let source_y = image_y_offset.saturating_sub(idwt_output.rect.y0);
        let copy_width = resolution_tile
            .rect
            .width()
            .min(idwt_output.rect.width().saturating_sub(source_x));
        let copy_height = resolution_tile
            .rect
            .height()
            .min(idwt_output.rect.height().saturating_sub(source_y));
        let output_x = resolution_tile.rect.x0.saturating_sub(image_x_offset);
        let output_y = resolution_tile.rect.y0.saturating_sub(image_y_offset);

        let handled = if let Some(backend) = backend.as_deref_mut() {
            copy_width > 0
                && copy_height > 0
                && backend.decode_store_component(J2kStoreComponentJob {
                    input: &idwt_output.coefficients,
                    input_width: idwt_output.rect.width(),
                    source_x,
                    source_y,
                    copy_width,
                    copy_height,
                    output: &mut channel_data.container,
                    output_width: header.size_data.image_width(),
                    output_x,
                    output_y,
                    addend: sign_shift,
                })?
        } else {
            false
        };

        if handled {
            return Ok(());
        }

        // If no sub-sampling, use a fast path where we copy rows of coefficients
        // at once.

        // The rect of the IDWT output corresponds to the rect of the highest
        // decomposition level of the tile, which is usually not 1:1 aligned
        // with the actual tile rectangle. We also need to account for the
        // offset of the reference grid.

        let skip_x = image_x_offset.saturating_sub(idwt_output.rect.x0);
        let skip_y = image_y_offset.saturating_sub(idwt_output.rect.y0);

        if sign_shift != 0.0 {
            for sample in idwt_output.coefficients.iter_mut() {
                *sample += sign_shift;
            }
        }

        let input_row_iter = idwt_output
            .coefficients
            .chunks_exact(idwt_output.rect.width() as usize)
            .skip(skip_y as usize)
            .take(idwt_output.rect.height() as usize);

        let output_row_iter = channel_data
            .container
            .chunks_exact_mut(header.size_data.image_width() as usize)
            .skip(resolution_tile.rect.y0.saturating_sub(image_y_offset) as usize);

        for (input_row, output_row) in input_row_iter.zip(output_row_iter) {
            let input_row = &input_row[skip_x as usize..];
            let output_row = &mut output_row
                [resolution_tile.rect.x0.saturating_sub(image_x_offset) as usize..]
                [..input_row.len()];

            output_row.copy_from_slice(input_row);
        }
    } else {
        if sign_shift != 0.0 {
            for sample in idwt_output.coefficients.iter_mut() {
                *sample += sign_shift;
            }
        }
        let image_width = header.size_data.image_width();
        let image_height = header.size_data.image_height();

        let x_shrink_factor = header.size_data.x_shrink_factor;
        let y_shrink_factor = header.size_data.y_shrink_factor;

        let x_offset = header
            .size_data
            .image_area_x_offset
            .div_ceil(x_shrink_factor);
        let y_offset = header
            .size_data
            .image_area_y_offset
            .div_ceil(y_shrink_factor);

        // Otherwise, copy sample by sample.
        for y in resolution_tile.rect.y0..resolution_tile.rect.y1 {
            let relative_y = (y - component_tile.rect.y0) as usize;
            let reference_grid_y = (scale_y as u32 * y) / y_shrink_factor;

            for x in resolution_tile.rect.x0..resolution_tile.rect.x1 {
                let relative_x = (x - component_tile.rect.x0) as usize;
                let reference_grid_x = (scale_x as u32 * x) / x_shrink_factor;

                let sample = idwt_output.coefficients
                    [relative_y * idwt_output.rect.width() as usize + relative_x];

                for x_position in u32::max(reference_grid_x, x_offset)
                    ..u32::min(reference_grid_x + scale_x as u32, image_width + x_offset)
                {
                    for y_position in u32::max(reference_grid_y, y_offset)
                        ..u32::min(reference_grid_y + scale_y as u32, image_height + y_offset)
                    {
                        let pos = (y_position - y_offset) as usize * image_width as usize
                            + (x_position - x_offset) as usize;

                        channel_data.container[pos] = sample;
                    }
                }
            }
        }
    }

    Ok(())
}

fn store_region<'a>(
    tile: &'a Tile<'a>,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    component_info: &ComponentInfo,
    component_idx: usize,
    output_region: OutputRegion,
    backend: &mut Option<&mut dyn HtCodeBlockDecoder>,
    sign_shift: f32,
) -> Result<()> {
    let channel_data = &mut tile_ctx.channel_data[component_idx];
    let idwt_output = &mut tile_ctx.idwt_output;

    let component_tile = ComponentTile::new(tile, component_info);
    let resolution_tile = ResolutionTile::new(
        component_tile,
        component_info.num_resolution_levels() - 1 - header.skipped_resolution_levels,
    );

    let (scale_x, scale_y) = (
        component_info.size_info.horizontal_resolution,
        component_info.size_info.vertical_resolution,
    );
    let image_width = header.size_data.image_width();
    let image_height = header.size_data.image_height();
    let x_shrink_factor = header.size_data.x_shrink_factor;
    let y_shrink_factor = header.size_data.y_shrink_factor;
    let x_offset = header
        .size_data
        .image_area_x_offset
        .div_ceil(x_shrink_factor);
    let y_offset = header
        .size_data
        .image_area_y_offset
        .div_ceil(y_shrink_factor);
    let region_x1 = output_region.x + output_region.width;
    let region_y1 = output_region.y + output_region.height;
    let output_width = output_region.width as usize;

    if scale_x == 1 && scale_y == 1 {
        let region_rect_x0 = output_region.x + x_offset;
        let region_rect_y0 = output_region.y + y_offset;
        let region_rect_x1 = region_x1 + x_offset;
        let region_rect_y1 = region_y1 + y_offset;
        let copy_x0 = idwt_output
            .rect
            .x0
            .max(resolution_tile.rect.x0)
            .max(region_rect_x0);
        let copy_y0 = idwt_output
            .rect
            .y0
            .max(resolution_tile.rect.y0)
            .max(region_rect_y0);
        let copy_x1 = idwt_output
            .rect
            .x1
            .min(resolution_tile.rect.x1)
            .min(region_rect_x1);
        let copy_y1 = idwt_output
            .rect
            .y1
            .min(resolution_tile.rect.y1)
            .min(region_rect_y1);

        let handled = if let Some(backend) = backend.as_deref_mut() {
            copy_x0 < copy_x1
                && copy_y0 < copy_y1
                && backend.decode_store_component(J2kStoreComponentJob {
                    input: &idwt_output.coefficients,
                    input_width: idwt_output.rect.width(),
                    source_x: copy_x0 - idwt_output.rect.x0,
                    source_y: copy_y0 - idwt_output.rect.y0,
                    copy_width: copy_x1 - copy_x0,
                    copy_height: copy_y1 - copy_y0,
                    output: &mut channel_data.container,
                    output_width: output_region.width,
                    output_x: copy_x0 - region_rect_x0,
                    output_y: copy_y0 - region_rect_y0,
                    addend: sign_shift,
                })?
        } else {
            false
        };

        if handled {
            return Ok(());
        }

        if sign_shift != 0.0 {
            for sample in idwt_output.coefficients.iter_mut() {
                *sample += sign_shift;
            }
        }

        if copy_x0 < copy_x1 && copy_y0 < copy_y1 {
            let input_width = idwt_output.rect.width() as usize;
            let copy_width = (copy_x1 - copy_x0) as usize;
            for y in copy_y0..copy_y1 {
                let src_start = (y - idwt_output.rect.y0) as usize * input_width
                    + (copy_x0 - idwt_output.rect.x0) as usize;
                let dst_start = (y - region_rect_y0) as usize * output_width
                    + (copy_x0 - region_rect_x0) as usize;
                channel_data.container[dst_start..dst_start + copy_width]
                    .copy_from_slice(&idwt_output.coefficients[src_start..src_start + copy_width]);
            }
        }

        return Ok(());
    }

    if sign_shift != 0.0 {
        for sample in idwt_output.coefficients.iter_mut() {
            *sample += sign_shift;
        }
    }

    for y in resolution_tile.rect.y0..resolution_tile.rect.y1 {
        let relative_y = (y - component_tile.rect.y0) as usize;
        let reference_grid_y = (scale_y as u32 * y) / y_shrink_factor;

        for x in resolution_tile.rect.x0..resolution_tile.rect.x1 {
            let relative_x = (x - component_tile.rect.x0) as usize;
            let reference_grid_x = (scale_x as u32 * x) / x_shrink_factor;

            let sample = idwt_output.coefficients
                [relative_y * idwt_output.rect.width() as usize + relative_x];

            for x_position in u32::max(reference_grid_x, x_offset)
                ..u32::min(reference_grid_x + scale_x as u32, image_width + x_offset)
            {
                let image_x = x_position - x_offset;
                if image_x < output_region.x || image_x >= region_x1 {
                    continue;
                }

                for y_position in u32::max(reference_grid_y, y_offset)
                    ..u32::min(reference_grid_y + scale_y as u32, image_height + y_offset)
                {
                    let image_y = y_position - y_offset;
                    if image_y < output_region.y || image_y >= region_y1 {
                        continue;
                    }

                    let pos = (image_y - output_region.y) as usize * output_width
                        + (image_x - output_region.x) as usize;
                    channel_data.container[pos] = sample;
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{collect_classic_code_block_data, CodeBlock, DecompositionStorage, Layer, Segment};
    use crate::error::DecodingError;
    use crate::j2c::codestream::CodeBlockStyle;
    use crate::j2c::rect::IntRect;

    fn classic_test_style() -> CodeBlockStyle {
        CodeBlockStyle {
            selective_arithmetic_coding_bypass: false,
            reset_context_probabilities: false,
            termination_on_each_pass: true,
            vertically_causal_context: false,
            segmentation_symbols: false,
            high_throughput_block_coding: false,
        }
    }

    fn classic_test_code_block() -> CodeBlock {
        CodeBlock {
            rect: IntRect::from_xywh(0, 0, 1, 1),
            x_idx: 0,
            y_idx: 0,
            layers: 0..1,
            has_been_included: true,
            missing_bit_planes: 0,
            number_of_coding_passes: 3,
            l_block: 3,
            non_empty_layer_count: 1,
        }
    }

    #[test]
    fn collect_classic_code_block_data_preserves_zero_length_segments() {
        let mut storage = DecompositionStorage::default();
        storage.layers.push(Layer {
            segments: Some(0..3),
        });
        storage.segments.push(Segment {
            idx: 0,
            coding_pases: 1,
            data_length: 1,
            data: &[0xAA],
        });
        storage.segments.push(Segment {
            idx: 1,
            coding_pases: 1,
            data_length: 0,
            data: &[],
        });
        storage.segments.push(Segment {
            idx: 2,
            coding_pases: 1,
            data_length: 1,
            data: &[0xBB],
        });

        let (combined_data, segments) = collect_classic_code_block_data(
            &classic_test_code_block(),
            &classic_test_style(),
            &storage,
        )
        .expect("collect classic segments");

        assert_eq!(combined_data, vec![0xAA, 0xBB]);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].data_offset, 0);
        assert_eq!(segments[0].data_length, 1);
        assert_eq!(segments[0].start_coding_pass, 0);
        assert_eq!(segments[0].end_coding_pass, 1);
        assert_eq!(segments[1].data_offset, 1);
        assert_eq!(segments[1].data_length, 0);
        assert_eq!(segments[1].start_coding_pass, 1);
        assert_eq!(segments[1].end_coding_pass, 2);
        assert_eq!(segments[2].data_offset, 1);
        assert_eq!(segments[2].data_length, 1);
        assert_eq!(segments[2].start_coding_pass, 2);
        assert_eq!(segments[2].end_coding_pass, 3);
    }

    #[test]
    fn collect_classic_code_block_data_rejects_non_contiguous_segment_indices() {
        let mut storage = DecompositionStorage::default();
        storage.layers.push(Layer {
            segments: Some(0..2),
        });
        storage.segments.push(Segment {
            idx: 0,
            coding_pases: 1,
            data_length: 1,
            data: &[0xAA],
        });
        storage.segments.push(Segment {
            idx: 2,
            coding_pases: 2,
            data_length: 1,
            data: &[0xBB],
        });

        let error = collect_classic_code_block_data(
            &classic_test_code_block(),
            &classic_test_style(),
            &storage,
        )
        .expect_err("non-contiguous segment indices must fail");

        assert_eq!(error, DecodingError::CodeBlockDecodeFailure.into());
    }
}
