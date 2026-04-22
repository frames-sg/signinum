// SPDX-License-Identifier: Apache-2.0

use slidecodec_core::{Downscale, PixelFormat, Rect};
use slidecodec_jpeg::{Decoder as CpuDecoder, Rect as JpegRect, ScratchPool};

use crate::{Error, Surface};

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
    let tile_edges = [256u32, 128u32, 64u32];

    for scale in scales {
        let denom = scale.denominator();
        for tile_edge in tile_edges {
            let src_tile = tile_edge.saturating_mul(denom);
            if src_tile == 0 {
                continue;
            }

            let cols = (dimensions.0 / src_tile).min(4);
            let rows = (dimensions.1 / src_tile).min(4);
            if cols.saturating_mul(rows) < 4 {
                continue;
            }

            let mut tiles = Vec::with_capacity((cols * rows) as usize);
            for row in 0..rows {
                for col in 0..cols {
                    tiles.push(ViewportTile {
                        source_roi: Rect {
                            x: col * src_tile,
                            y: row * src_tile,
                            w: src_tile,
                            h: src_tile,
                        },
                        dest: Rect {
                            x: col * tile_edge,
                            y: row * tile_edge,
                            w: tile_edge,
                            h: tile_edge,
                        },
                    });
                }
            }

            return Some(ViewportWorkload {
                scale,
                viewport_dims: (cols * tile_edge, rows * tile_edge),
                tiles,
            });
        }
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
        let tile_dims = scaled_dims((tile.source_roi.w, tile.source_roi.h), scale);
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
    let mut stages = Vec::with_capacity(tiles.len());
    for tile in tiles {
        stages.push(crate::compute::decode_region_scaled_to_viewport_stage(
            decoder,
            pool,
            to_jpeg_rect(tile.source_roi),
            scale,
            tile.dest,
        )?);
    }
    crate::compute::compose_rgb_viewport(&stages, viewport_dims)
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

fn scaled_dims(full: (u32, u32), scale: Downscale) -> (u32, u32) {
    (
        full.0.div_ceil(scale.denominator()),
        full.1.div_ceil(scale.denominator()),
    )
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
