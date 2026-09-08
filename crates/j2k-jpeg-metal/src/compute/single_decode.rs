// SPDX-License-Identifier: MIT OR Apache-2.0

mod fast444;
mod routing;
pub(in crate::compute) mod scratch;
mod subsampled;

pub(super) use fast444::try_decode_fast444_scaled_region_to_surface_with_mode_and_status;
#[cfg(test)]
pub(super) use fast444::{
    try_decode_fast444_region_to_surface, try_decode_fast444_scaled_region_to_surface,
    try_decode_fast444_scaled_to_surface, try_decode_fast444_to_surface,
};
pub(crate) use routing::{
    decode_private_rgb8_tile_with_session, decode_region_scaled_to_surface,
    decode_region_to_surface, decode_scaled_to_surface, decode_to_surface,
    decode_to_surface_with_session,
};
#[cfg(test)]
pub(super) use subsampled::{
    try_decode_fast420_region_to_surface, try_decode_fast420_scaled_region_to_surface,
    try_decode_fast420_scaled_to_surface, try_decode_fast422_region_to_surface,
    try_decode_fast422_scaled_to_surface, try_decode_fast422_to_surface,
};
pub(super) use subsampled::{
    try_decode_fast420_scaled_region_to_surface_with_status,
    try_decode_fast422_scaled_region_to_surface,
};
