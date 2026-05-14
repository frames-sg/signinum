//! The irreversible multi-component transformation, as specified in
//! Annex G.2 and G.3.

use super::codestream::{Header, WaveletTransform};
use super::decode::TileDecodeContext;
use crate::error::{bail, err, ColorError, Result};
use crate::math::{dispatch, f32x8, floor_f32, Level, Simd};
use crate::{HtCodeBlockDecoder, J2kInverseMctJob, J2kWaveletTransform};

/// Apply the inverse multi-component transform, as specified in G.2 and G.3.
pub(crate) fn apply_inverse(
    tile_ctx: &mut TileDecodeContext,
    component_infos: &[super::codestream::ComponentInfo],
    header: &Header<'_>,
    backend: &mut Option<&mut dyn HtCodeBlockDecoder>,
) -> Result<()> {
    if tile_ctx.channel_data.len() < 3 {
        return if header.strict {
            err!(ColorError::Mct)
        } else {
            Ok(())
        };
    }

    let (s, _) = tile_ctx.channel_data.split_at_mut(3);
    let [s0, s1, s2] = s else { unreachable!() };

    let transform = component_infos[0].wavelet_transform();

    if transform != component_infos[1].wavelet_transform()
        || component_infos[1].wavelet_transform() != component_infos[2].wavelet_transform()
    {
        bail!(ColorError::Mct);
    }

    if s0.container.len() != s1.container.len() || s1.container.len() != s2.container.len() {
        bail!(ColorError::Mct);
    }

    let handled = if let Some(backend) = backend.as_deref_mut() {
        backend.decode_inverse_mct(J2kInverseMctJob {
            transform: J2kWaveletTransform::from(transform),
            plane0: &mut s0.container,
            plane1: &mut s1.container,
            plane2: &mut s2.container,
            addend0: (1_u32 << (component_infos[0].size_info.precision - 1)) as f32,
            addend1: (1_u32 << (component_infos[1].size_info.precision - 1)) as f32,
            addend2: (1_u32 << (component_infos[2].size_info.precision - 1)) as f32,
        })?
    } else {
        false
    };

    if !handled {
        apply_inner(
            transform,
            &mut s0.container,
            &mut s1.container,
            &mut s2.container,
        );
    }

    Ok(())
}

fn apply_inner(transform: WaveletTransform, s0: &mut [f32], s1: &mut [f32], s2: &mut [f32]) {
    dispatch!(Level::new(), simd => apply_inner_impl(simd, transform, s0, s1, s2));
}

#[inline(always)]
fn apply_inner_impl<S: Simd>(
    simd: S,
    transform: WaveletTransform,
    s0: &mut [f32],
    s1: &mut [f32],
    s2: &mut [f32],
) {
    match transform {
        // Irreversible MCT, specified in G.3.
        WaveletTransform::Irreversible97 => {
            let mut s0_chunks = s0.chunks_exact_mut(8);
            let mut s1_chunks = s1.chunks_exact_mut(8);
            let mut s2_chunks = s2.chunks_exact_mut(8);
            for ((y0, y1), y2) in s0_chunks
                .by_ref()
                .zip(s1_chunks.by_ref())
                .zip(s2_chunks.by_ref())
            {
                let y_0 = f32x8::from_slice(simd, y0);
                let y_1 = f32x8::from_slice(simd, y1);
                let y_2 = f32x8::from_slice(simd, y2);

                let i0 = y_2.mul_add(f32x8::splat(simd, 1.402), y_0);
                let i1 = y_2.mul_add(
                    f32x8::splat(simd, -0.71414),
                    y_1.mul_add(f32x8::splat(simd, -0.34413), y_0),
                );
                let i2 = y_1.mul_add(f32x8::splat(simd, 1.772), y_0);

                i0.store(y0);
                i1.store(y1);
                i2.store(y2);
            }
            for ((y0, y1), y2) in s0_chunks
                .into_remainder()
                .iter_mut()
                .zip(s1_chunks.into_remainder().iter_mut())
                .zip(s2_chunks.into_remainder().iter_mut())
            {
                let src0 = *y0;
                let src1 = *y1;
                let src2 = *y2;
                *y0 = src0 + 1.402 * src2;
                *y1 = src0 - 0.34413 * src1 - 0.71414 * src2;
                *y2 = src0 + 1.772 * src1;
            }
        }
        // Reversible MCT, specified in G.2.
        WaveletTransform::Reversible53 => {
            let mut s0_chunks = s0.chunks_exact_mut(8);
            let mut s1_chunks = s1.chunks_exact_mut(8);
            let mut s2_chunks = s2.chunks_exact_mut(8);
            for ((y0, y1), y2) in s0_chunks
                .by_ref()
                .zip(s1_chunks.by_ref())
                .zip(s2_chunks.by_ref())
            {
                let y_0 = f32x8::from_slice(simd, y0);
                let y_1 = f32x8::from_slice(simd, y1);
                let y_2 = f32x8::from_slice(simd, y2);

                let i1 = y_0 - ((y_2 + y_1) * 0.25).floor();
                let i0 = y_2 + i1;
                let i2 = y_1 + i1;

                i0.store(y0);
                i1.store(y1);
                i2.store(y2);
            }
            for ((y0, y1), y2) in s0_chunks
                .into_remainder()
                .iter_mut()
                .zip(s1_chunks.into_remainder().iter_mut())
                .zip(s2_chunks.into_remainder().iter_mut())
            {
                let src0 = *y0;
                let src1 = *y1;
                let src2 = *y2;
                let i1 = src0 - floor_f32((src2 + src1) * 0.25);
                *y0 = src2 + i1;
                *y1 = i1;
                *y2 = src1 + i1;
            }
        }
    }
}
