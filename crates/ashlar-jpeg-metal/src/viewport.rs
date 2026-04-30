// SPDX-License-Identifier: Apache-2.0

use ashlar_core::{BackendRequest, Downscale, PixelFormat, Rect};
#[cfg(target_os = "macos")]
use ashlar_jpeg::adapter::{
    build_metal_fast420_packet_for_decoder, build_metal_fast422_packet_for_decoder,
    build_metal_fast444_packet_for_decoder,
};
use ashlar_jpeg::{Decoder as CpuDecoder, Rect as JpegRect, ScratchPool};

use crate::{Error, Surface};

const VIEWPORT_TILE_EDGE: u32 = 96;
const VIEWPORT_TILE_COLS: u32 = 6;
const VIEWPORT_TILE_ROWS: u32 = 2;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewportTile {
    pub source_roi: Rect,
    pub dest: Rect,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewportWorkload {
    pub scale: Downscale,
    pub viewport_dims: (u32, u32),
    pub tiles: Vec<ViewportTile>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewportSurfaceStrategy {
    CpuComposite,
    CpuContiguous,
    HybridComposite,
    HybridContiguous,
}

pub fn viewport_source_bounds(workload: &ViewportWorkload) -> Rect {
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    for tile in &workload.tiles {
        min_x = min_x.min(tile.source_roi.x);
        min_y = min_y.min(tile.source_roi.y);
        max_x = max_x.max(tile.source_roi.x.saturating_add(tile.source_roi.w));
        max_y = max_y.max(tile.source_roi.y.saturating_add(tile.source_roi.h));
    }

    Rect {
        x: min_x,
        y: min_y,
        w: max_x.saturating_sub(min_x),
        h: max_y.saturating_sub(min_y),
    }
}

pub fn is_contiguous_viewport_workload(workload: &ViewportWorkload) -> bool {
    if workload.tiles.is_empty() {
        return false;
    }

    let source = viewport_source_bounds(workload);
    let scaled_source = scaled_rect_covering(source, workload.scale);
    if (scaled_source.w, scaled_source.h) != workload.viewport_dims {
        return false;
    }

    let viewport_area = u64::from(workload.viewport_dims.0) * u64::from(workload.viewport_dims.1);
    let mut area_sum = 0u64;

    for tile in &workload.tiles {
        let scaled_tile = scaled_rect_covering(tile.source_roi, workload.scale);
        let expected = Rect {
            x: scaled_tile.x.saturating_sub(scaled_source.x),
            y: scaled_tile.y.saturating_sub(scaled_source.y),
            w: scaled_tile.w,
            h: scaled_tile.h,
        };
        if tile.dest != expected {
            return false;
        }
        if tile.dest.x.saturating_add(tile.dest.w) > workload.viewport_dims.0
            || tile.dest.y.saturating_add(tile.dest.h) > workload.viewport_dims.1
        {
            return false;
        }

        area_sum = area_sum.saturating_add(u64::from(tile.dest.w) * u64::from(tile.dest.h));
    }

    for (idx, tile) in workload.tiles.iter().enumerate() {
        let tile_right = tile.dest.x.saturating_add(tile.dest.w);
        let tile_bottom = tile.dest.y.saturating_add(tile.dest.h);
        for other in &workload.tiles[idx + 1..] {
            let other_right = other.dest.x.saturating_add(other.dest.w);
            let other_bottom = other.dest.y.saturating_add(other.dest.h);
            let separated = tile_right <= other.dest.x
                || other_right <= tile.dest.x
                || tile_bottom <= other.dest.y
                || other_bottom <= tile.dest.y;
            if !separated {
                return false;
            }
        }
    }

    area_sum == viewport_area
}

pub fn choose_viewport_surface_strategy(
    workload: &ViewportWorkload,
    backend: BackendRequest,
) -> Result<ViewportSurfaceStrategy, Error> {
    let contiguous = is_contiguous_viewport_workload(workload);
    match backend {
        BackendRequest::Cpu => Ok(if contiguous {
            ViewportSurfaceStrategy::CpuContiguous
        } else {
            ViewportSurfaceStrategy::CpuComposite
        }),
        BackendRequest::Auto | BackendRequest::Metal => {
            #[cfg(target_os = "macos")]
            {
                Ok(if contiguous {
                    ViewportSurfaceStrategy::HybridContiguous
                } else {
                    ViewportSurfaceStrategy::HybridComposite
                })
            }
            #[cfg(not(target_os = "macos"))]
            {
                if matches!(backend, BackendRequest::Metal) {
                    Err(Error::MetalUnavailable)
                } else if contiguous {
                    Ok(ViewportSurfaceStrategy::CpuContiguous)
                } else {
                    Ok(ViewportSurfaceStrategy::CpuComposite)
                }
            }
        }
        BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
    }
}

fn choose_viewport_surface_strategy_for_decoder(
    decoder: &CpuDecoder<'_>,
    workload: &ViewportWorkload,
    backend: BackendRequest,
) -> Result<ViewportSurfaceStrategy, Error> {
    if !matches!(backend, BackendRequest::Auto) {
        return choose_viewport_surface_strategy(workload, backend);
    }

    #[cfg(not(target_os = "macos"))]
    let _ = decoder;

    #[cfg(target_os = "macos")]
    {
        let contiguous = is_contiguous_viewport_workload(workload);
        if !contiguous {
            return Ok(ViewportSurfaceStrategy::CpuComposite);
        }

        let restart_coded = decoder.info().restart_interval.is_some();
        if !restart_coded {
            return Ok(ViewportSurfaceStrategy::CpuContiguous);
        }

        let has_direct_packet = build_metal_fast444_packet_for_decoder(decoder).is_ok()
            || build_metal_fast420_packet_for_decoder(decoder).is_ok();
        Ok(if has_direct_packet {
            ViewportSurfaceStrategy::HybridContiguous
        } else {
            ViewportSurfaceStrategy::CpuContiguous
        })
    }

    #[cfg(not(target_os = "macos"))]
    {
        choose_viewport_surface_strategy(workload, backend)
    }
}

pub fn suggest_viewport_workload(dimensions: (u32, u32)) -> Option<ViewportWorkload> {
    let scales = [
        Downscale::Eighth,
        Downscale::Quarter,
        Downscale::Half,
        Downscale::None,
    ];
    let viewport_dims = (
        VIEWPORT_TILE_EDGE * VIEWPORT_TILE_COLS,
        VIEWPORT_TILE_EDGE * VIEWPORT_TILE_ROWS,
    );
    for scale in scales {
        let denom = scale.denominator();
        let Some(x) = viewport_origin(dimensions.0, viewport_dims.0.saturating_mul(denom), denom)
        else {
            continue;
        };
        let Some(y) = viewport_origin(dimensions.1, viewport_dims.1.saturating_mul(denom), denom)
        else {
            continue;
        };
        let source_viewport = Rect {
            x,
            y,
            w: viewport_dims.0.saturating_mul(denom),
            h: viewport_dims.1.saturating_mul(denom),
        };
        let scaled_source = scaled_rect_covering(source_viewport, scale);
        if (scaled_source.w, scaled_source.h) != viewport_dims {
            continue;
        }
        let source_tile = VIEWPORT_TILE_EDGE.saturating_mul(denom);
        let mut tiles = Vec::with_capacity((VIEWPORT_TILE_COLS * VIEWPORT_TILE_ROWS) as usize);
        for row in 0..VIEWPORT_TILE_ROWS {
            for col in 0..VIEWPORT_TILE_COLS {
                tiles.push(ViewportTile {
                    source_roi: Rect {
                        x: source_viewport.x + col * source_tile,
                        y: source_viewport.y + row * source_tile,
                        w: source_tile,
                        h: source_tile,
                    },
                    dest: Rect {
                        x: col * VIEWPORT_TILE_EDGE,
                        y: row * VIEWPORT_TILE_EDGE,
                        w: VIEWPORT_TILE_EDGE,
                        h: VIEWPORT_TILE_EDGE,
                    },
                });
            }
        }

        return Some(ViewportWorkload {
            scale,
            viewport_dims,
            tiles,
        });
    }

    None
}

pub fn compose_viewport_cpu(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    fmt: PixelFormat,
    scale: Downscale,
    viewport_dims: (u32, u32),
    tiles: &[ViewportTile],
) -> Result<Vec<u8>, Error> {
    let bpp = fmt.bytes_per_pixel();
    let viewport_stride = viewport_dims.0 as usize * bpp;
    let mut viewport = vec![0u8; viewport_stride * viewport_dims.1 as usize];

    for tile in tiles {
        let scaled = scaled_rect_covering(tile.source_roi, scale);
        let tile_dims = (scaled.w, scaled.h);
        if tile_dims != (tile.dest.w, tile.dest.h) {
            return Err(Error::MetalKernel {
                message: format!(
                    "viewport tile dims {:?} do not match destination rect {:?}",
                    tile_dims, tile.dest
                ),
            });
        }
        let tile_stride = tile_dims.0 as usize * bpp;
        let mut tile_bytes = vec![0u8; tile_stride * tile_dims.1 as usize];
        decoder.decode_region_scaled_into_with_scratch(
            pool,
            &mut tile_bytes,
            tile_stride,
            fmt,
            to_jpeg_rect(tile.source_roi),
            scale,
        )?;
        blit_into_viewport(
            &tile_bytes,
            tile_dims,
            fmt,
            &mut viewport,
            viewport_dims,
            tile.dest,
        )?;
    }

    Ok(viewport)
}

pub fn decode_viewport_region_cpu(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    fmt: PixelFormat,
    workload: &ViewportWorkload,
) -> Result<Vec<u8>, Error> {
    let source = viewport_source_bounds(workload);
    let stride = workload.viewport_dims.0 as usize * fmt.bytes_per_pixel();
    let mut viewport = vec![0u8; stride * workload.viewport_dims.1 as usize];
    decoder.decode_region_scaled_into_with_scratch(
        pool,
        &mut viewport,
        stride,
        fmt,
        to_jpeg_rect(source),
        workload.scale,
    )?;
    Ok(viewport)
}

pub fn decode_viewport_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
    backend: BackendRequest,
) -> Result<Surface, Error> {
    match choose_viewport_surface_strategy_for_decoder(decoder, workload, backend)? {
        ViewportSurfaceStrategy::CpuComposite => compose_viewport_cpu_to_surface(
            decoder,
            pool,
            workload.scale,
            workload.viewport_dims,
            &workload.tiles,
        ),
        ViewportSurfaceStrategy::CpuContiguous => {
            decode_viewport_region_cpu_to_surface(decoder, pool, workload)
        }
        ViewportSurfaceStrategy::HybridComposite => compose_viewport_hybrid(
            decoder,
            pool,
            workload.scale,
            workload.viewport_dims,
            &workload.tiles,
        ),
        ViewportSurfaceStrategy::HybridContiguous => {
            decode_viewport_region_hybrid(decoder, pool, workload)
        }
    }
}

#[cfg(target_os = "macos")]
pub fn decode_viewport_region_cpu_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
) -> Result<Surface, Error> {
    let bytes = decode_viewport_region_cpu(decoder, pool, PixelFormat::Rgb8, workload)?;
    crate::upload_surface(
        bytes,
        workload.viewport_dims,
        PixelFormat::Rgb8,
        ashlar_core::BackendRequest::Metal,
    )
}

#[cfg(not(target_os = "macos"))]
pub fn decode_viewport_region_cpu_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
) -> Result<Surface, Error> {
    let bytes = decode_viewport_region_cpu(decoder, pool, PixelFormat::Rgb8, workload)?;
    crate::upload_surface(
        bytes,
        workload.viewport_dims,
        PixelFormat::Rgb8,
        ashlar_core::BackendRequest::Cpu,
    )
}

#[cfg(target_os = "macos")]
pub fn compose_viewport_cpu_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    scale: Downscale,
    viewport_dims: (u32, u32),
    tiles: &[ViewportTile],
) -> Result<Surface, Error> {
    let bytes = compose_viewport_cpu(
        decoder,
        pool,
        PixelFormat::Rgb8,
        scale,
        viewport_dims,
        tiles,
    )?;
    crate::upload_surface(
        bytes,
        viewport_dims,
        PixelFormat::Rgb8,
        ashlar_core::BackendRequest::Metal,
    )
}

#[cfg(not(target_os = "macos"))]
pub fn compose_viewport_cpu_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    scale: Downscale,
    viewport_dims: (u32, u32),
    tiles: &[ViewportTile],
) -> Result<Surface, Error> {
    let bytes = compose_viewport_cpu(
        decoder,
        pool,
        PixelFormat::Rgb8,
        scale,
        viewport_dims,
        tiles,
    )?;
    crate::upload_surface(
        bytes,
        viewport_dims,
        PixelFormat::Rgb8,
        ashlar_core::BackendRequest::Cpu,
    )
}

#[cfg(target_os = "macos")]
pub fn compose_viewport_hybrid(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    scale: Downscale,
    viewport_dims: (u32, u32),
    tiles: &[ViewportTile],
) -> Result<Surface, Error> {
    crate::compute::compose_rgb_viewport_from_regions(decoder, pool, scale, viewport_dims, tiles)
}

#[cfg(target_os = "macos")]
pub fn decode_viewport_region_hybrid(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
) -> Result<Surface, Error> {
    let use_direct_kernel = decoder.info().restart_interval.is_some();
    let fast444_packet = use_direct_kernel
        .then(|| build_metal_fast444_packet_for_decoder(decoder).ok())
        .flatten();
    let fast422_packet = use_direct_kernel
        .then(|| build_metal_fast422_packet_for_decoder(decoder).ok())
        .flatten();
    let fast420_packet = use_direct_kernel
        .then(|| build_metal_fast420_packet_for_decoder(decoder).ok())
        .flatten();
    crate::compute::decode_region_scaled_to_surface(
        decoder,
        pool,
        PixelFormat::Rgb8,
        to_jpeg_rect(viewport_source_bounds(workload)),
        workload.scale,
        fast444_packet.as_ref(),
        fast422_packet.as_ref(),
        fast420_packet.as_ref(),
    )
}

#[cfg(not(target_os = "macos"))]
pub fn decode_viewport_region_hybrid(
    _decoder: &CpuDecoder<'_>,
    _pool: &mut ScratchPool,
    _workload: &ViewportWorkload,
) -> Result<Surface, Error> {
    Err(Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
pub fn compose_viewport_hybrid(
    _decoder: &CpuDecoder<'_>,
    _pool: &mut ScratchPool,
    _scale: Downscale,
    _viewport_dims: (u32, u32),
    _tiles: &[ViewportTile],
) -> Result<Surface, Error> {
    Err(Error::MetalUnavailable)
}

fn viewport_origin(full_extent: u32, viewport_extent: u32, align: u32) -> Option<u32> {
    if viewport_extent > full_extent || align == 0 {
        return None;
    }

    let centered = (full_extent - viewport_extent) / 2;
    Some(centered - centered % align)
}

fn scaled_rect_covering(rect: Rect, scale: Downscale) -> Rect {
    let denom = scale.denominator();
    let x_end = rect.x + rect.w;
    let y_end = rect.y + rect.h;
    let x0 = rect.x / denom;
    let y0 = rect.y / denom;
    let x1 = x_end.div_ceil(denom);
    let y1 = y_end.div_ceil(denom);
    Rect {
        x: x0,
        y: y0,
        w: x1.saturating_sub(x0),
        h: y1.saturating_sub(y0),
    }
}

fn to_jpeg_rect(rect: Rect) -> JpegRect {
    JpegRect {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: rect.h,
    }
}

fn blit_into_viewport(
    tile: &[u8],
    tile_dims: (u32, u32),
    fmt: PixelFormat,
    viewport: &mut [u8],
    viewport_dims: (u32, u32),
    dest: Rect,
) -> Result<(), Error> {
    if dest.x.saturating_add(dest.w) > viewport_dims.0
        || dest.y.saturating_add(dest.h) > viewport_dims.1
    {
        return Err(Error::MetalKernel {
            message: format!("viewport destination {dest:?} exceeds viewport {viewport_dims:?}"),
        });
    }

    let bpp = fmt.bytes_per_pixel();
    let tile_stride = tile_dims.0 as usize * bpp;
    let viewport_stride = viewport_dims.0 as usize * bpp;
    for row in 0..tile_dims.1 as usize {
        let src_start = row * tile_stride;
        let src_end = src_start + tile_stride;
        let dst_start = (dest.y as usize + row) * viewport_stride + dest.x as usize * bpp;
        let dst_end = dst_start + tile_stride;
        viewport[dst_start..dst_end].copy_from_slice(&tile[src_start..src_end]);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASELINE_420: &[u8] =
        include_bytes!("../../../corpus/conformance/baseline_420_16x16.jpg");
    const BASELINE_420_RESTART: &[u8] =
        include_bytes!("../../../corpus/conformance/baseline_420_restart_32x16.jpg");

    fn sparse_workload_from(workload: &ViewportWorkload) -> ViewportWorkload {
        ViewportWorkload {
            scale: workload.scale,
            viewport_dims: workload.viewport_dims,
            tiles: vec![
                *workload.tiles.first().expect("viewport tile"),
                *workload.tiles.last().expect("viewport tile"),
            ],
        }
    }

    #[test]
    fn auto_strategy_keeps_large_contiguous_nonrestart_workloads_on_cpu_contiguous() {
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let workload = suggest_viewport_workload((2_048, 1_024)).expect("contiguous workload");

        assert_eq!(
            choose_viewport_surface_strategy_for_decoder(&decoder, &workload, BackendRequest::Auto)
                .expect("strategy"),
            ViewportSurfaceStrategy::CpuContiguous
        );
    }

    #[test]
    fn auto_strategy_prefers_hybrid_for_restart_coded_contiguous_workloads() {
        let decoder = CpuDecoder::new(BASELINE_420_RESTART).expect("decoder");
        let workload = ViewportWorkload {
            scale: Downscale::None,
            viewport_dims: (32, 16),
            tiles: vec![ViewportTile {
                source_roi: Rect {
                    x: 0,
                    y: 0,
                    w: 32,
                    h: 16,
                },
                dest: Rect {
                    x: 0,
                    y: 0,
                    w: 32,
                    h: 16,
                },
            }],
        };

        assert_eq!(
            choose_viewport_surface_strategy_for_decoder(&decoder, &workload, BackendRequest::Auto)
                .expect("strategy"),
            ViewportSurfaceStrategy::HybridContiguous
        );
    }

    #[test]
    fn auto_strategy_keeps_large_sparse_nonrestart_workloads_on_cpu_composite() {
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let contiguous = suggest_viewport_workload((2_048, 1_024)).expect("contiguous workload");
        let workload = sparse_workload_from(&contiguous);

        assert_eq!(
            choose_viewport_surface_strategy_for_decoder(&decoder, &workload, BackendRequest::Auto)
                .expect("strategy"),
            ViewportSurfaceStrategy::CpuComposite
        );
    }

    #[test]
    fn auto_strategy_keeps_restart_coded_sparse_workloads_on_cpu_composite() {
        let decoder = CpuDecoder::new(BASELINE_420_RESTART).expect("decoder");
        let contiguous = suggest_viewport_workload((8_192, 2_048)).expect("contiguous workload");
        let workload = sparse_workload_from(&contiguous);

        assert_eq!(
            choose_viewport_surface_strategy_for_decoder(&decoder, &workload, BackendRequest::Auto)
                .expect("strategy"),
            ViewportSurfaceStrategy::CpuComposite
        );
    }
}
