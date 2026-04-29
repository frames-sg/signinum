// SPDX-License-Identifier: Apache-2.0

use crate::error::J2kError;
use slidecodec_core::{Downscale, Rect};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceDecodeRequest {
    Full,
    Region { roi: Rect },
    Scaled { scale: Downscale },
    RegionScaled { roi: Rect, scale: Downscale },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceDecodePlan {
    source_dims: (u32, u32),
    source_rect: Rect,
    scale: Downscale,
    output_rect: Rect,
}

impl DeviceDecodePlan {
    pub fn for_image(
        source_dims: (u32, u32),
        request: DeviceDecodeRequest,
    ) -> Result<Self, J2kError> {
        let (source_rect, scale) = match request {
            DeviceDecodeRequest::Full => (Rect::full(source_dims), Downscale::None),
            DeviceDecodeRequest::Region { roi } => (roi, Downscale::None),
            DeviceDecodeRequest::Scaled { scale } => (Rect::full(source_dims), scale),
            DeviceDecodeRequest::RegionScaled { roi, scale } => (roi, scale),
        };

        if !source_rect.is_within(source_dims) {
            return Err(J2kError::InvalidRegion {
                x: source_rect.x,
                y: source_rect.y,
                w: source_rect.w,
                h: source_rect.h,
                image_w: source_dims.0,
                image_h: source_dims.1,
            });
        }

        Ok(Self {
            source_dims,
            source_rect,
            scale,
            output_rect: scaled_rect_covering(source_rect, scale),
        })
    }

    pub fn source_dims(self) -> (u32, u32) {
        self.source_dims
    }

    pub fn source_rect(self) -> Rect {
        self.source_rect
    }

    pub fn scale(self) -> Downscale {
        self.scale
    }

    pub fn output_rect(self) -> Rect {
        self.output_rect
    }

    pub fn output_dims(self) -> (u32, u32) {
        (self.output_rect.w, self.output_rect.h)
    }

    pub fn target_resolution(self) -> Option<(u32, u32)> {
        (self.scale != Downscale::None).then_some(self.output_dims())
    }

    pub fn is_full_frame(self) -> bool {
        self.source_rect == Rect::full(self.source_dims) && self.scale == Downscale::None
    }
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
