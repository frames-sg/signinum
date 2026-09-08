// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prepared decode-plan construction and validation.

mod construction;
mod progressive_quant;

#[cfg(test)]
mod tests;

use construction::PreparedConstructionBudget;
use progressive_quant::latch_progressive_quant_tables;

use super::allocation::{
    compute_progressive_scratch_bytes, decode_workspace_cap, ensure_prepared_construction_fits,
    prepared_decode_plan_allocation_bytes, progressive_prepared_allocation_bytes,
};
use super::{
    compute_decode_scratch_bytes, compute_extended12_planes_scratch_bytes,
    compute_lossless_scratch_bytes, lossless_color_sampling, ColorSpace, DecodeOptions, Decoder,
    DecoderContext, Info, JpegError, MarkerKind, ParsedHeader, PreparedComponentPlan,
    PreparedDecodePlan, PreparedDecoderMetadata, PreparedHuffmanTableId, PreparedHuffmanTables,
    PreparedLosslessPlan, PreparedProgressiveComponentPlan, PreparedProgressivePlan,
    PreparedProgressiveScan, PreparedProgressiveScanComponent, SofKind, DEFAULT_MAX_DECODE_BYTES,
};

impl Decoder<'_> {
    pub(super) fn validate_cached_plan_lease(
        header: &ParsedHeader,
        ctx: &DecoderContext,
        plan: &PreparedDecodePlan,
        external_live_bytes: usize,
    ) -> Result<(), JpegError> {
        let retained_parsed_bytes = header.retained_allocation_bytes()?;
        PreparedConstructionBudget::with_external_live(
            external_live_bytes,
            retained_parsed_bytes,
            ctx.retained_allocation_bytes(),
        )?;
        ensure_prepared_construction_fits(header, plan.retained_allocation_bytes()?)
    }

    pub(super) fn prepare_header_with_external_live(
        header: ParsedHeader,
        info: Info,
        options: DecodeOptions,
        bytes: &[u8],
        ctx: &mut DecoderContext,
        external_live_bytes: usize,
    ) -> Result<PreparedDecoderMetadata, JpegError> {
        let retained_parsed_bytes = header.retained_allocation_bytes()?;
        let mut construction = PreparedConstructionBudget::with_external_live(
            external_live_bytes,
            retained_parsed_bytes,
            ctx.retained_allocation_bytes(),
        )?;
        let (plan, progressive_plan, lossless_plan) = if matches!(
            info.sof_kind,
            SofKind::Progressive8 | SofKind::Progressive12
        ) {
            let progressive_plan =
                Self::build_progressive_plan(&header, &info, ctx, &mut construction)?;
            let plan = Self::build_progressive_host_output_plan(
                &header,
                &info,
                &progressive_plan,
                &mut construction,
            )?;
            (plan, Some(progressive_plan), None)
        } else if info.sof_kind == SofKind::Lossless {
            let (plan, lossless_plan) =
                Self::build_lossless_plan(&header, &info, ctx, &mut construction)?;
            (plan, None, Some(lossless_plan))
        } else if options == DecodeOptions::default() {
            if let Some(scan_offset) = header.sos_offset {
                let header_prefix =
                    bytes
                        .get(..scan_offset)
                        .ok_or(JpegError::InternalInvariant {
                            reason: "parsed SOS offset is outside the JPEG input",
                        })?;
                let plan =
                    ctx.resolve_decode_plan(header_prefix, retained_parsed_bytes, |ctx| {
                        Self::build_prepared_plan(&header, &info, ctx, &mut construction)
                    })?;
                construction.rebase_after_plan_cache(
                    ctx.retained_allocation_bytes(),
                    plan.retained_allocation_bytes()?,
                )?;
                (plan, None, None)
            } else {
                (
                    Self::build_prepared_plan(&header, &info, ctx, &mut construction)?,
                    None,
                    None,
                )
            }
        } else {
            (
                Self::build_prepared_plan(&header, &info, ctx, &mut construction)?,
                None,
                None,
            )
        };
        let mut prepared_bytes = plan.retained_allocation_bytes()?;
        if let Some(progressive) = &progressive_plan {
            prepared_bytes = crate::allocation::checked_add_allocation_bytes(
                prepared_bytes,
                progressive.retained_allocation_bytes()?,
            )?;
        }
        construction.verify_retained(ctx.retained_allocation_bytes(), prepared_bytes)?;
        ensure_prepared_construction_fits(&header, prepared_bytes)?;
        Ok(PreparedDecoderMetadata {
            info,
            warnings: header.warnings,
            plan,
            progressive_plan,
            lossless_plan,
        })
    }

    pub(super) fn build_prepared_plan(
        header: &ParsedHeader,
        info: &Info,
        ctx: &mut DecoderContext,
        construction: &mut PreparedConstructionBudget,
    ) -> Result<PreparedDecodePlan, JpegError> {
        match info.sof_kind {
            SofKind::Baseline8 | SofKind::Extended8 | SofKind::Extended12 => {}
            other => return Err(JpegError::NotImplemented { sof: other }),
        }
        if info.sof_kind == SofKind::Extended12
            && !matches!(
                info.color_space,
                ColorSpace::Grayscale
                    | ColorSpace::YCbCr
                    | ColorSpace::Rgb
                    | ColorSpace::Cmyk
                    | ColorSpace::Ycck
            )
        {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        match info.color_space {
            ColorSpace::Grayscale
            | ColorSpace::YCbCr
            | ColorSpace::Rgb
            | ColorSpace::Cmyk
            | ColorSpace::Ycck => {}
        }

        let scan = header.scan.as_ref().ok_or(JpegError::MissingMarker {
            marker: MarkerKind::Sos,
        })?;
        if header.scan_count != 1 {
            return Err(JpegError::InvalidSequentialScanCount {
                sof: info.sof_kind,
                count: header.scan_count,
            });
        }
        validate_leading_component_sampling(header, info)?;
        // Every component must declare H,V in 1..=4 per T.81 §B.2.2, and max_h
        // must actually divide every component's H (same for V). Malformed
        // streams can set H=0 (div-by-zero in upsample ratio), non-divisors
        // (arbitrary ratios M2 handles), or ratios that don't produce planes
        // that cover the image width.
        for (h, v) in header.sampling.iter() {
            if h == 0 || v == 0 || h > 4 || v > 4 {
                return Err(JpegError::NotImplemented { sof: info.sof_kind });
            }
            if !header.sampling.max_h.is_multiple_of(h) || !header.sampling.max_v.is_multiple_of(v)
            {
                return Err(JpegError::NotImplemented { sof: info.sof_kind });
            }
        }
        let prepared_bytes = prepared_decode_plan_allocation_bytes(
            scan.components.len(),
            header.huffman_tables.versions.len(),
        )?;
        ensure_prepared_construction_fits(header, prepared_bytes)?;
        let huffman_tables = compile_huffman_versions(header, ctx, construction)?;
        let prepared_bytes = prepared_decode_plan_allocation_bytes(
            scan.components.len(),
            huffman_tables.capacity(),
        )?;
        ensure_prepared_construction_fits(header, prepared_bytes)?;
        let workspace_cap = decode_workspace_cap(header, prepared_bytes)?;

        build_decode_plan(
            header,
            info,
            huffman_tables,
            workspace_cap,
            ctx,
            construction,
        )
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "JPEG component counts are validated against the eight-bit SOF field"
    )]
    pub(super) fn build_lossless_plan(
        header: &ParsedHeader,
        info: &Info,
        ctx: &mut DecoderContext,
        construction: &mut PreparedConstructionBudget,
    ) -> Result<(PreparedDecodePlan, PreparedLosslessPlan), JpegError> {
        if info.sof_kind != SofKind::Lossless {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        if header.scan_count != 1 {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        let scan = header.scan.as_ref().ok_or(JpegError::MissingMarker {
            marker: MarkerKind::Sos,
        })?;
        if !(1..=7).contains(&scan.ss) {
            return Err(JpegError::UnsupportedPredictor { predictor: scan.ss });
        }
        if scan.se != 0 || scan.ah != 0 || scan.al != 0 {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        let expected_components = match (info.color_space, info.bit_depth) {
            (ColorSpace::Grayscale, 8 | 16) => 1,
            (ColorSpace::Rgb | ColorSpace::YCbCr, 8 | 16) => 3,
            _ => return Err(JpegError::NotImplemented { sof: info.sof_kind }),
        };
        if scan.components.len() != expected_components {
            return Err(JpegError::UnsupportedComponentCount {
                count: scan.components.len() as u8,
            });
        }
        let prepared_bytes = prepared_decode_plan_allocation_bytes(
            scan.components.len(),
            header.huffman_tables.versions.len(),
        )?;
        ensure_prepared_construction_fits(header, prepared_bytes)?;
        let huffman_tables = compile_huffman_versions(header, ctx, construction)?;
        let prepared_bytes = prepared_decode_plan_allocation_bytes(
            scan.components.len(),
            huffman_tables.capacity(),
        )?;
        ensure_prepared_construction_fits(header, prepared_bytes)?;
        let workspace_cap = decode_workspace_cap(header, prepared_bytes)?;
        let mut components = construction.try_vec(scan.components.len())?;
        let mut first_dc_table = None;
        for scan_component in scan.components.iter().copied() {
            let component_index = find_component_index(&header.component_ids, scan_component.id)
                .ok_or(JpegError::UnknownScanComponent {
                    offset: header.sos_offset.unwrap_or_default(),
                    component: scan_component.id,
                })?;
            let (h, v) =
                header
                    .sampling
                    .component(component_index)
                    .ok_or(JpegError::MissingMarker {
                        marker: MarkerKind::Sof,
                    })?;
            let dc_table = prepared_active_huffman_id(
                &header.huffman_tables,
                &huffman_tables,
                0,
                scan_component.dc_table,
            )
            .ok_or(JpegError::MissingHuffmanTable {
                component: scan_component.id,
                class: 0,
                id: scan_component.dc_table,
            })?;
            first_dc_table.get_or_insert(dc_table);
            components.push(PreparedComponentPlan {
                h,
                v,
                output_index: component_index,
                quant: ctx.resolve_quant_table([1; 64]),
                dc_table: Some(dc_table),
                ac_table: None,
            });
        }
        if matches!(info.color_space, ColorSpace::Rgb | ColorSpace::YCbCr)
            && lossless_color_sampling(info).is_none()
        {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        let plan = PreparedDecodePlan {
            components,
            huffman_tables,
            sampling: info.sampling,
            color_space: info.color_space,
            restart_interval: header.restart_interval,
            dimensions: info.dimensions,
            scan_offset: header.sos_offset.ok_or(JpegError::MissingMarker {
                marker: MarkerKind::Sos,
            })?,
            scratch_bytes: compute_lossless_scratch_bytes(info, workspace_cap)?,
        };
        let lossless = PreparedLosslessPlan {
            predictor: scan.ss,
            bit_depth: info.bit_depth,
            dc_table: first_dc_table.ok_or(JpegError::MissingMarker {
                marker: MarkerKind::Sos,
            })?,
            dimensions: info.dimensions,
            scan_offset: plan.scan_offset,
        };
        Ok((plan, lossless))
    }

    #[expect(
        clippy::too_many_lines,
        reason = "progressive planning validates the ordered scan script while compiling shared table state"
    )]
    pub(super) fn build_progressive_plan(
        header: &ParsedHeader,
        info: &Info,
        ctx: &mut DecoderContext,
        construction: &mut PreparedConstructionBudget,
    ) -> Result<PreparedProgressivePlan, JpegError> {
        if !matches!(
            info.sof_kind,
            SofKind::Progressive8 | SofKind::Progressive12
        ) {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        match (info.sof_kind, info.color_space) {
            (
                SofKind::Progressive8 | SofKind::Progressive12,
                ColorSpace::Grayscale | ColorSpace::YCbCr | ColorSpace::Rgb,
            )
            | (SofKind::Progressive12, ColorSpace::Cmyk | ColorSpace::Ycck) => {}
            (_, color_space) => return Err(JpegError::UnsupportedColorSpace { color_space }),
        }
        validate_sampling_factors(header, info)?;
        if header.progressive_scans.is_empty() {
            return Err(JpegError::MissingMarker {
                marker: MarkerKind::Sos,
            });
        }
        let latched_quant_tables = latch_progressive_quant_tables(header)?;

        let max_h = u32::from(header.sampling.max_h);
        let max_v = u32::from(header.sampling.max_v);
        let mcu_cols = info.dimensions.0.div_ceil(8 * max_h);
        let mcu_rows = info.dimensions.1.div_ceil(8 * max_v);
        let total_scan_components =
            header
                .progressive_scans
                .iter()
                .try_fold(0usize, |total, parsed| {
                    total.checked_add(parsed.scan.components.len()).ok_or(
                        JpegError::MemoryCapExceeded {
                            requested: usize::MAX,
                            cap: DEFAULT_MAX_DECODE_BYTES,
                        },
                    )
                })?;
        let prepared_bytes = progressive_prepared_allocation_bytes(
            header.component_ids.len(),
            header.progressive_scans.len(),
            total_scan_components,
            header.huffman_tables.versions.len(),
        )?;
        ensure_prepared_construction_fits(header, prepared_bytes)?;
        let huffman_tables = compile_huffman_versions(header, ctx, construction)?;
        let prepared_bytes = progressive_prepared_allocation_bytes(
            header.component_ids.len(),
            header.progressive_scans.len(),
            total_scan_components,
            huffman_tables.capacity(),
        )?;
        ensure_prepared_construction_fits(header, prepared_bytes)?;
        let workspace_cap = decode_workspace_cap(header, prepared_bytes)?;

        let mut components = construction.try_vec(header.component_ids.len())?;
        for (output_index, &id) in header.component_ids.iter().enumerate() {
            let (h, v) =
                header
                    .sampling
                    .component(output_index)
                    .ok_or(JpegError::MissingMarker {
                        marker: MarkerKind::Sof,
                    })?;
            let quant_id =
                *header
                    .quant_table_ids
                    .get(output_index)
                    .ok_or(JpegError::MissingMarker {
                        marker: MarkerKind::Sof,
                    })?;
            let quant =
                latched_quant_tables
                    .table(output_index)
                    .ok_or(JpegError::MissingQuantTable {
                        component: id,
                        table_id: quant_id,
                    })?;
            components.push(PreparedProgressiveComponentPlan {
                h,
                v,
                output_index,
                quant: ctx.resolve_quant_table(quant),
                block_cols: mcu_cols * u32::from(h),
                block_rows: mcu_rows * u32::from(v),
                sample_width: info
                    .dimensions
                    .0
                    .saturating_mul(u32::from(h))
                    .div_ceil(max_h),
                sample_height: info
                    .dimensions
                    .1
                    .saturating_mul(u32::from(v))
                    .div_ceil(max_v),
            });
        }

        let mut scan_components = construction.try_vec(total_scan_components)?;
        let mut scans = construction.try_vec(header.progressive_scans.len())?;
        for parsed in &header.progressive_scans {
            let component_start = scan_components.len();
            for component in &parsed.scan.components {
                let component_index = find_component_index(&header.component_ids, component.id)
                    .ok_or(JpegError::UnknownScanComponent {
                        offset: parsed.entropy_offset,
                        component: component.id,
                    })?;
                let dc_table = if parsed.scan.ss == 0 {
                    Some(resolve_progressive_huffman(
                        &parsed.table_state,
                        &huffman_tables,
                        component.id,
                        0,
                        component.dc_table,
                    )?)
                } else {
                    None
                };
                let ac_table = if parsed.scan.ss > 0 {
                    Some(resolve_progressive_huffman(
                        &parsed.table_state,
                        &huffman_tables,
                        component.id,
                        1,
                        component.ac_table,
                    )?)
                } else {
                    None
                };
                scan_components.push(PreparedProgressiveScanComponent {
                    component_index,
                    dc_table,
                    ac_table,
                });
            }
            scans.push(PreparedProgressiveScan {
                component_start,
                component_len: parsed.scan.components.len(),
                ss: parsed.scan.ss,
                se: parsed.scan.se,
                ah: parsed.scan.ah,
                al: parsed.scan.al,
                entropy_offset: parsed.entropy_offset,
                terminal_offset: parsed.terminal_offset,
                terminal_code: parsed.terminal_code,
                restart_interval: parsed.restart_interval,
            });
        }

        let scratch_bytes = compute_progressive_scratch_bytes(
            &components,
            info.dimensions.0 as usize,
            info.sof_kind,
            workspace_cap,
        )?;
        Ok(PreparedProgressivePlan {
            components,
            scan_components,
            scans,
            huffman_tables,
            sampling: info.sampling,
            color_space: info.color_space,
            dimensions: info.dimensions,
            mcu_cols,
            mcu_rows,
            scratch_bytes,
        })
    }

    pub(super) fn build_progressive_host_output_plan(
        header: &ParsedHeader,
        info: &Info,
        progressive: &PreparedProgressivePlan,
        construction: &mut PreparedConstructionBudget,
    ) -> Result<PreparedDecodePlan, JpegError> {
        let mut components = construction.try_vec(progressive.components.len())?;
        for component in &progressive.components {
            components.push(PreparedComponentPlan {
                h: component.h,
                v: component.v,
                output_index: component.output_index,
                quant: component.quant,
                dc_table: None,
                ac_table: None,
            });
        }
        Ok(PreparedDecodePlan {
            components,
            huffman_tables: construction.try_huffman_tables(0)?,
            sampling: info.sampling,
            color_space: info.color_space,
            restart_interval: header.restart_interval,
            dimensions: info.dimensions,
            scan_offset: header.sos_offset.ok_or(JpegError::MissingMarker {
                marker: MarkerKind::Sos,
            })?,
            // Progressive decoding uses the aggregate coefficient/render
            // workspace recorded on `PreparedProgressivePlan`; this companion
            // plan only supplies shared output geometry and table handles.
            scratch_bytes: 0,
        })
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "JPEG scan component counts are validated against the eight-bit SOS field"
)]
fn build_decode_plan(
    header: &ParsedHeader,
    info: &Info,
    huffman_tables: PreparedHuffmanTables,
    workspace_cap: usize,
    ctx: &mut DecoderContext,
    construction: &mut PreparedConstructionBudget,
) -> Result<PreparedDecodePlan, JpegError> {
    let scan = header.scan.as_ref().ok_or(JpegError::MissingMarker {
        marker: MarkerKind::Sos,
    })?;
    let scan_offset = header.sos_offset.ok_or(JpegError::MissingMarker {
        marker: MarkerKind::Sos,
    })?;

    let mut components = construction.try_vec(scan.components.len())?;
    for scan_comp in scan.components.iter().copied() {
        let output_index = find_component_index(&header.component_ids, scan_comp.id).ok_or(
            JpegError::UnknownScanComponent {
                offset: scan_offset,
                component: scan_comp.id,
            },
        )?;
        let (h, v) = header
            .sampling
            .component(output_index)
            .ok_or(JpegError::MissingMarker {
                marker: MarkerKind::Sof,
            })?;
        let quant_id =
            *header
                .quant_table_ids
                .get(output_index)
                .ok_or(JpegError::MissingMarker {
                    marker: MarkerKind::Sof,
                })? as usize;
        let quant = *header
            .quant_tables
            .entries
            .get(quant_id)
            .and_then(|q| q.as_ref())
            .ok_or(JpegError::MissingQuantTable {
                component: scan_comp.id,
                table_id: quant_id as u8,
            })?;
        let dc_table = prepared_active_huffman_id(
            &header.huffman_tables,
            &huffman_tables,
            0,
            scan_comp.dc_table,
        )
        .ok_or(JpegError::MissingHuffmanTable {
            component: scan_comp.id,
            class: 0,
            id: scan_comp.dc_table,
        })?;
        let ac_table = prepared_active_huffman_id(
            &header.huffman_tables,
            &huffman_tables,
            1,
            scan_comp.ac_table,
        )
        .ok_or(JpegError::MissingHuffmanTable {
            component: scan_comp.id,
            class: 1,
            id: scan_comp.ac_table,
        })?;
        components.push(PreparedComponentPlan {
            h,
            v,
            output_index,
            quant: ctx.resolve_quant_table(quant),
            dc_table: Some(dc_table),
            ac_table: Some(ac_table),
        });
    }

    let mut scratch_bytes =
        compute_decode_scratch_bytes(info.dimensions, info.sampling, workspace_cap)?;
    if info.sof_kind == SofKind::Extended12 {
        // The sequential 12-bit paths render through full-frame u16 component
        // planes, which dwarf the stripe-based estimate above.
        scratch_bytes = scratch_bytes.max(compute_extended12_planes_scratch_bytes(
            &components,
            info.dimensions,
            info.sampling,
            workspace_cap,
        )?);
    }

    Ok(PreparedDecodePlan {
        components,
        huffman_tables,
        sampling: info.sampling,
        color_space: info.color_space,
        restart_interval: header.restart_interval,
        dimensions: info.dimensions,
        scan_offset,
        scratch_bytes,
    })
}

fn validate_sampling_factors(header: &ParsedHeader, info: &Info) -> Result<(), JpegError> {
    validate_leading_component_sampling(header, info)?;
    for (h, v) in header.sampling.iter() {
        if h == 0 || v == 0 || h > 4 || v > 4 {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        if !header.sampling.max_h.is_multiple_of(h) || !header.sampling.max_v.is_multiple_of(v) {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
    }
    Ok(())
}

fn validate_leading_component_sampling(
    header: &ParsedHeader,
    info: &Info,
) -> Result<(), JpegError> {
    if !matches!(info.color_space, ColorSpace::YCbCr) {
        return Ok(());
    }
    if let Some((h, v)) = header.sampling.component(0) {
        if h != header.sampling.max_h || v != header.sampling.max_v {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
    }
    Ok(())
}

fn resolve_progressive_huffman(
    state: &crate::parse::tables::ProgressiveTableState,
    prepared: &PreparedHuffmanTables,
    component: u8,
    class: u8,
    id: u8,
) -> Result<PreparedHuffmanTableId, JpegError> {
    let version_index = match class {
        0 => state.dc_version_index(id),
        1 => state.ac_version_index(id),
        _ => None,
    }
    .ok_or(JpegError::MissingHuffmanTable {
        component,
        class,
        id,
    })?;
    prepared
        .id_at(version_index)
        .ok_or(JpegError::InternalInvariant {
            reason: "raw Huffman version is absent from the prepared arena",
        })
}

fn compile_huffman_versions(
    header: &ParsedHeader,
    ctx: &mut DecoderContext,
    construction: &mut PreparedConstructionBudget,
) -> Result<PreparedHuffmanTables, JpegError> {
    let mut prepared = construction.try_huffman_tables(header.huffman_tables.versions.len())?;
    for version in &header.huffman_tables.versions {
        prepared.push(construction.resolve_huffman_table(ctx, &version.raw, version.role)?)?;
    }
    Ok(prepared)
}

fn prepared_active_huffman_id(
    tables: &crate::parse::tables::HuffmanTables,
    prepared: &PreparedHuffmanTables,
    class: u8,
    slot: u8,
) -> Option<PreparedHuffmanTableId> {
    let version_index = match class {
        0 => tables.active_dc_version_index(slot),
        1 => tables.active_ac_version_index(slot),
        _ => None,
    }?;
    prepared.id_at(version_index)
}

pub(super) fn find_component_index(component_ids: &[u8], id: u8) -> Option<usize> {
    component_ids
        .iter()
        .position(|&component_id| component_id == id)
}
