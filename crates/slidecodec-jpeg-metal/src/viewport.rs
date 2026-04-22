// SPDX-License-Identifier: Apache-2.0

use slidecodec_core::{Downscale, PixelFormat, Rect};
use slidecodec_jpeg::{Decoder as CpuDecoder, Rect as JpegRect, ScratchPool};

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
        slidecodec_core::BackendRequest::Metal,
    )
}

#[cfg(not(target_os = "macos"))]
pub fn compose_viewport_cpu_to_surface(
    _decoder: &CpuDecoder<'_>,
    _pool: &mut ScratchPool,
    _scale: Downscale,
    _viewport_dims: (u32, u32),
    _tiles: &[ViewportTile],
) -> Result<Surface, Error> {
    Err(Error::MetalUnavailable)
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
