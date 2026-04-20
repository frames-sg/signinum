// SPDX-License-Identifier: Apache-2.0

pub(crate) mod openjpeg {
    pub(crate) use slidecodec_j2k_compare::openjpeg::{decode_rgb, decode_rgb_region};
}

pub(crate) mod grok {
    pub(crate) use slidecodec_j2k_compare::grok::{decode_rgb, decode_rgb_scaled, is_available};
}
