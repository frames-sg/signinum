// SPDX-License-Identifier: MIT OR Apache-2.0

//! Direct device-plan construction without host component materialization.

use crate::error::bail;
use crate::j2c;
use crate::{
    ColorSpace, DecoderContext, DecodingError, DirectPlanUnsupportedReason, J2kDirectColorPlan,
    J2kDirectGrayscalePlan, Result,
};

use super::Image;

mod referenced;

impl<'a> Image<'a> {
    /// Build full-image RGB component plans on their native sampling grids.
    ///
    /// Unlike the full-resolution color plan, each component's output dimensions
    /// are its sampled dimensions. The returned sampling factors map those
    /// planes to the image grid by sample replication. The caller must perform
    /// that expansion before interleaving RGB output.
    ///
    /// Supports a single unsigned origin-zero tile without MCT, alpha, or resolution
    /// reduction. Other geometry returns a direct-plan capability error.
    ///
    /// # Errors
    ///
    /// Returns a direct-plan capability error for unrepresented geometry, or
    /// the underlying packet-validation/allocation error while constructing the
    /// coefficient graph. Planning retains compressed payloads and metadata;
    /// it does not decode coefficients or allocate expanded image planes.
    #[doc(hidden)]
    pub fn build_component_grid_color_plan_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<(J2kDirectColorPlan, [(u8, u8); 3])> {
        if !matches!(self.color_space, ColorSpace::RGB) || self.has_alpha {
            bail!(DecodingError::DirectPlanUnsupported(
                DirectPlanUnsupportedReason::ColorRgbImageWithoutAlpha
            ));
        }
        j2c::build_component_grid_color_plan(
            self.codestream,
            &self.header,
            self.retained_metadata_bytes()?,
            decoder_context,
        )
    }

    /// Build an adapter grayscale direct device plan without materializing host component planes.
    #[doc(hidden)]
    pub fn build_direct_grayscale_plan_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<J2kDirectGrayscalePlan> {
        if !matches!(self.color_space, ColorSpace::Gray) || self.has_alpha {
            bail!(DecodingError::DirectPlanUnsupported(
                DirectPlanUnsupportedReason::GrayscaleImageWithoutAlpha
            ));
        }

        j2c::build_direct_grayscale_plan(
            self.codestream,
            &self.header,
            self.retained_metadata_bytes()?,
            decoder_context,
        )
    }

    /// Build an adapter grayscale direct device plan for an output-space region.
    #[doc(hidden)]
    pub fn build_direct_grayscale_plan_region_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
        output_region: (u32, u32, u32, u32),
    ) -> Result<J2kDirectGrayscalePlan> {
        if !matches!(self.color_space, ColorSpace::Gray) || self.has_alpha {
            bail!(DecodingError::DirectPlanUnsupported(
                DirectPlanUnsupportedReason::GrayscaleImageWithoutAlpha
            ));
        }

        let retained_metadata_bytes = self.retained_metadata_bytes()?;
        decoder_context.set_output_region(Some(output_region));
        let result = j2c::build_direct_grayscale_plan(
            self.codestream,
            &self.header,
            retained_metadata_bytes,
            decoder_context,
        );
        decoder_context.set_output_region(None);
        result
    }

    /// Build an adapter RGB direct device plan without materializing host component planes.
    #[doc(hidden)]
    pub fn build_direct_color_plan_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<J2kDirectColorPlan> {
        if !matches!(self.color_space, ColorSpace::RGB) || self.has_alpha {
            bail!(DecodingError::DirectPlanUnsupported(
                DirectPlanUnsupportedReason::ColorRgbImageWithoutAlpha
            ));
        }

        j2c::build_direct_color_plan(
            self.codestream,
            &self.header,
            self.retained_metadata_bytes()?,
            decoder_context,
        )
    }

    /// Build an adapter RGB direct device plan for an output-space region.
    #[doc(hidden)]
    pub fn build_direct_color_plan_region_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
        output_region: (u32, u32, u32, u32),
    ) -> Result<J2kDirectColorPlan> {
        if !matches!(self.color_space, ColorSpace::RGB) || self.has_alpha {
            bail!(DecodingError::DirectPlanUnsupported(
                DirectPlanUnsupportedReason::ColorRgbImageWithoutAlpha
            ));
        }

        let retained_metadata_bytes = self.retained_metadata_bytes()?;
        decoder_context.set_output_region(Some(output_region));
        let result = j2c::build_direct_color_plan(
            self.codestream,
            &self.header,
            retained_metadata_bytes,
            decoder_context,
        );
        decoder_context.set_output_region(None);
        result
    }
}
