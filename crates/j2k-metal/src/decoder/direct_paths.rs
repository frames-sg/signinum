// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use objc2_metal::MTLDevice as _;

#[cfg(target_os = "macos")]
use std::sync::Arc;

#[cfg(target_os = "macos")]
use crate::metal_types::Device;
#[cfg(target_os = "macos")]
use j2k_core::{BackendRequest, PixelFormat};
#[cfg(target_os = "macos")]
use j2k_native::{DecodeSettings as NativeDecodeSettings, Image as NativeImage};

#[cfg(target_os = "macos")]
use super::surface::CPU_STAGED_METAL_REQUIRES_EXPLICIT_API;
use super::J2kDecoder;
#[cfg(target_os = "macos")]
use crate::direct;
#[cfg(target_os = "macos")]
use crate::error::{adapter_backend_error, native_decode_error};
#[cfg(target_os = "macos")]
use crate::session::direct_plan_cache::{
    cached_session_direct_color_plan, cached_session_direct_gray_plan, direct_gray_plan_cache_key,
    direct_plan_cache_key, store_session_direct_color_plan, store_session_direct_gray_plan,
};
#[cfg(target_os = "macos")]
use crate::{Error, MetalBackendSession, MetalDirectFallbackReason, Surface};

macro_rules! define_ensure_prepared_direct_plan {
    (
        with_session: $with_session:ident,
        plain: $plain:ident,
        prepare_fresh: $prepare_fresh:ident,
        plan_field: $plan_field:ident,
        prepared_field: $prepared_field:ident,
        prepared_device_field: $prepared_device_field:ident,
        prepared_ty: $prepared_ty:path,
        cache_key: $cache_key:ident,
        cached: $cached:ident,
        store: $store:ident,
        build: $build:ident,
        prepare: $prepare:path
    ) => {
        #[cfg(target_os = "macos")]
        pub(super) fn $with_session(
            &mut self,
            fmt: PixelFormat,
            session: &MetalBackendSession,
        ) -> Result<Option<Arc<$prepared_ty>>, Error> {
            let device_registry_id = session.device().registryID();
            if self.$prepared_field.is_some()
                && self.$prepared_device_field != Some(device_registry_id)
            {
                self.$prepared_field = None;
                self.$prepared_device_field = None;
            }
            if self.$prepared_field.is_none() {
                let cache_key = $cache_key(self.bytes, fmt);
                if let Some((plan, prepared)) = $cached(session, cache_key)? {
                    self.$plan_field = Some(plan);
                    self.$prepared_field = Some(prepared);
                    self.$prepared_device_field = Some(device_registry_id);
                }
            }
            self.$prepare_fresh(Some((session, fmt)), device_registry_id)
        }

        #[cfg(target_os = "macos")]
        fn $plain(&mut self) -> Result<Option<Arc<$prepared_ty>>, Error> {
            let device_registry_id = crate::engine::current_runtime_device_registry_id()?;
            if self.$prepared_field.is_some()
                && self.$prepared_device_field != Some(device_registry_id)
            {
                self.$prepared_field = None;
                self.$prepared_device_field = None;
            }
            self.$prepare_fresh(None, device_registry_id)
        }

        #[cfg(target_os = "macos")]
        fn $prepare_fresh(
            &mut self,
            session_cache: Option<(&MetalBackendSession, PixelFormat)>,
            device_registry_id: u64,
        ) -> Result<Option<Arc<$prepared_ty>>, Error> {
            if self.$prepared_field.is_none() {
                self.ensure_native_image()?;
                let (Some(image), native_context) =
                    (self.native_image.as_ref(), &mut self.native_context)
                else {
                    return Err(Error::Decode(adapter_backend_error(
                        "native image cache missing".to_string(),
                    )));
                };
                let plan = match image.$build(native_context) {
                    Ok(plan) => Arc::new(plan),
                    Err(error) if direct::is_unsupported_direct_plan_error(&error) => {
                        return Ok(None);
                    }
                    Err(error) => return Err(native_decode_error(error)),
                };
                let prepared = if let Some((session, _)) = &session_cache {
                    Arc::new(crate::engine::with_runtime_for_session(session, |_| {
                        $prepare(plan.as_ref())
                    })?)
                } else {
                    Arc::new($prepare(plan.as_ref())?)
                };
                if let Some((session, fmt)) = session_cache {
                    let cache_key = $cache_key(self.bytes, fmt);
                    $store(session, cache_key, plan.clone(), prepared.clone())?;
                }
                self.$plan_field = Some(plan);
                self.$prepared_field = Some(prepared);
                self.$prepared_device_field = Some(device_registry_id);
            }
            Ok(self.$prepared_field.clone())
        }
    };
}

impl J2kDecoder<'_> {
    #[cfg(target_os = "macos")]
    pub(super) fn ensure_native_image(&mut self) -> Result<(), Error> {
        if self.native_image.is_none() {
            self.native_image = Some(
                NativeImage::new(self.bytes, &NativeDecodeSettings::default())
                    .map_err(native_decode_error)?,
            );
        }
        Ok(())
    }

    define_ensure_prepared_direct_plan! {
        with_session: ensure_prepared_direct_gray_plan_with_session,
        plain: ensure_prepared_direct_gray_plan,
        prepare_fresh: prepare_fresh_direct_gray_plan,
        plan_field: native_direct_gray_plan,
        prepared_field: native_prepared_direct_gray_plan,
        prepared_device_field: native_prepared_direct_gray_device_registry_id,
        prepared_ty: crate::engine::PreparedDirectGrayscalePlan,
        cache_key: direct_gray_plan_cache_key,
        cached: cached_session_direct_gray_plan,
        store: store_session_direct_gray_plan,
        build: build_direct_grayscale_plan_with_context,
        prepare: crate::engine::prepare_direct_grayscale_plan
    }

    define_ensure_prepared_direct_plan! {
        with_session: ensure_prepared_direct_color_plan_with_session,
        plain: ensure_prepared_direct_color_plan,
        prepare_fresh: prepare_fresh_direct_color_plan,
        plan_field: native_direct_color_plan,
        prepared_field: native_prepared_direct_color_plan,
        prepared_device_field: native_prepared_direct_color_device_registry_id,
        prepared_ty: crate::engine::PreparedDirectColorPlan,
        cache_key: direct_plan_cache_key,
        cached: cached_session_direct_color_plan,
        store: store_session_direct_color_plan,
        build: build_direct_color_plan_with_context,
        prepare: crate::engine::prepare_direct_color_plan
    }

    #[cfg(target_os = "macos")]
    pub(super) fn decode_direct_to_surface(
        &mut self,
        fmt: PixelFormat,
    ) -> Result<Option<Surface>, Error> {
        if matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
            let Some(plan) = self.ensure_prepared_direct_gray_plan()? else {
                return Ok(None);
            };
            return Ok(Some(crate::engine::execute_prepared_direct_grayscale_plan(
                &plan, fmt,
            )?));
        }

        if matches!(
            fmt,
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
        ) {
            let Some(plan) = self.ensure_prepared_direct_color_plan()? else {
                return Ok(None);
            };
            return match crate::engine::execute_prepared_direct_color_plan(plan, fmt) {
                Ok(surface) => Ok(Some(surface)),
                Err(error) if is_direct_runtime_fallback_error(&error) => Ok(None),
                Err(error) => Err(error),
            };
        }

        Ok(None)
    }

    #[cfg(target_os = "macos")]
    pub(super) fn decode_direct_to_surface_with_session(
        &mut self,
        fmt: PixelFormat,
        session: &MetalBackendSession,
    ) -> Result<Option<Surface>, Error> {
        if matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
            let Some(plan) = self.ensure_prepared_direct_gray_plan_with_session(fmt, session)?
            else {
                return Ok(None);
            };
            return Ok(Some(
                crate::engine::execute_prepared_direct_grayscale_plan_with_device(
                    &plan,
                    fmt,
                    session.device_handle(),
                )?,
            ));
        }

        if matches!(
            fmt,
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
        ) {
            let Some(plan) = self.ensure_prepared_direct_color_plan_with_session(fmt, session)?
            else {
                return Ok(None);
            };
            return match crate::engine::execute_prepared_direct_color_plan_with_device(
                plan,
                fmt,
                session.device_handle(),
            ) {
                Ok(surface) => Ok(Some(surface)),
                Err(error) if is_direct_runtime_fallback_error(&error) => Ok(None),
                Err(error) => Err(error),
            };
        }

        Ok(None)
    }

    #[cfg(target_os = "macos")]
    pub(super) fn decode_full_to_metal_surface(
        &mut self,
        fmt: PixelFormat,
    ) -> Result<Surface, Error> {
        let inputs = [self.bytes];
        match crate::engine::decode_component_grid_color_batch(&inputs, fmt, None) {
            Ok(mut surfaces) => return surfaces.pop().ok_or_else(adapter_missing_sampled_surface),
            Err(Error::MetalDirectFallback {
                reason: MetalDirectFallbackReason::UnsupportedPlan,
                ..
            }) => {}
            Err(error) => return Err(error),
        }
        self.ensure_native_image()?;
        let (Some(image), native_context) = (self.native_image.as_ref(), &mut self.native_context)
        else {
            return Err(Error::Decode(adapter_backend_error(
                "native image cache missing".to_string(),
            )));
        };
        crate::engine::decode_image_to_surface(image, native_context, fmt)
    }

    #[cfg(target_os = "macos")]
    pub(super) fn decode_full_to_metal_surface_with_device(
        &mut self,
        fmt: PixelFormat,
        device: &Device,
    ) -> Result<Surface, Error> {
        let inputs = [self.bytes];
        match crate::engine::decode_component_grid_color_batch(&inputs, fmt, None) {
            Ok(mut surfaces) => return surfaces.pop().ok_or_else(adapter_missing_sampled_surface),
            Err(Error::MetalDirectFallback {
                reason: MetalDirectFallbackReason::UnsupportedPlan,
                ..
            }) => {}
            Err(error) => return Err(error),
        }
        self.ensure_native_image()?;
        let (Some(image), native_context) = (self.native_image.as_ref(), &mut self.native_context)
        else {
            return Err(Error::Decode(adapter_backend_error(
                "native image cache missing".to_string(),
            )));
        };
        crate::engine::decode_image_to_surface_with_device(image, native_context, fmt, device)
    }

    #[cfg(target_os = "macos")]
    fn decode_repeated_cpu_to_surfaces(
        &mut self,
        fmt: PixelFormat,
        count: usize,
    ) -> Result<Vec<Surface>, Error> {
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal repeated CPU surface collection",
        );
        let mut surfaces = budget.try_vec(count, "J2K Metal repeated CPU surfaces")?;
        for _ in 0..count {
            surfaces.push(self.decode_to_cpu_surface(fmt)?);
        }
        Ok(surfaces)
    }

    #[cfg(target_os = "macos")]
    #[doc(hidden)]
    pub fn decode_repeated_grayscale_direct_to_device(
        &mut self,
        fmt: PixelFormat,
        count: usize,
    ) -> Result<Vec<Surface>, Error> {
        self.decode_repeated_grayscale_direct_to_device_routed(fmt, count, None)
    }

    #[cfg(target_os = "macos")]
    #[doc(hidden)]
    pub fn decode_repeated_grayscale_direct_to_device_with_session(
        &mut self,
        fmt: PixelFormat,
        count: usize,
        session: &MetalBackendSession,
    ) -> Result<Vec<Surface>, Error> {
        self.decode_repeated_grayscale_direct_to_device_routed(fmt, count, Some(session))
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn decode_repeated_grayscale_direct_to_device_routed(
        &mut self,
        fmt: PixelFormat,
        count: usize,
        session: Option<&MetalBackendSession>,
    ) -> Result<Vec<Surface>, Error> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let plan = match session {
            Some(session) => self.ensure_prepared_direct_gray_plan_with_session(fmt, session)?,
            None => self.ensure_prepared_direct_gray_plan()?,
        };
        let Some(plan) = plan else {
            return Err(Error::MetalDirectFallback {
                message: format!(
                    "explicit J2K MetalDirect repeated batch does not support {fmt:?}"
                ),
                reason: MetalDirectFallbackReason::UnsupportedPlan,
            });
        };
        match session {
            Some(session) => crate::engine::with_runtime_for_session(session, |_| {
                crate::engine::execute_repeated_prepared_direct_grayscale_plan(&plan, fmt, count)
            }),
            None => {
                crate::engine::execute_repeated_prepared_direct_grayscale_plan(&plan, fmt, count)
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[doc(hidden)]
    pub fn decode_repeated_color_direct_to_device(
        &mut self,
        fmt: PixelFormat,
        count: usize,
    ) -> Result<Vec<Surface>, Error> {
        self.decode_repeated_color_direct_to_device_routed(fmt, count, None)
    }

    #[cfg(target_os = "macos")]
    #[doc(hidden)]
    pub fn decode_repeated_color_direct_to_device_with_session(
        &mut self,
        fmt: PixelFormat,
        count: usize,
        session: &MetalBackendSession,
    ) -> Result<Vec<Surface>, Error> {
        self.decode_repeated_color_direct_to_device_routed(fmt, count, Some(session))
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn decode_repeated_color_direct_to_device_routed(
        &mut self,
        fmt: PixelFormat,
        count: usize,
        session: Option<&MetalBackendSession>,
    ) -> Result<Vec<Surface>, Error> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let surface = match session {
            Some(session) => self.decode_request_to_device_with_session(
                crate::MetalDecodeRequest::full(fmt, BackendRequest::Metal),
                session,
            )?,
            None => self.decode_op_to_surface_impl(super::MetalDecodeRequest::full(
                fmt,
                BackendRequest::Metal,
            ))?,
        };
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal repeated color surface collection",
        );
        Ok(budget.try_filled(count, surface, "J2K Metal repeated color surfaces")?)
    }

    #[cfg(target_os = "macos")]
    #[doc(hidden)]
    pub fn decode_repeated_grayscale_auto_to_device(
        &mut self,
        fmt: PixelFormat,
        count: usize,
    ) -> Result<Vec<Surface>, Error> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let dims = self.inner.info().dimensions;
        let Some(support) = j2k::J2kDecoder::inspect_support(self.bytes).ok() else {
            return self.decode_repeated_cpu_to_surfaces(fmt, count);
        };
        if !crate::routing::auto_repeated_decode_uses_metal(
            dims,
            fmt,
            count,
            support.transfer_syntax,
            support.payload_kind,
        ) {
            return self.decode_repeated_cpu_to_surfaces(fmt, count);
        }
        let device_registry_id = crate::engine::current_runtime_device_registry_id()?;
        if self.native_prepared_direct_gray_plan.is_some()
            && self.native_prepared_direct_gray_device_registry_id != Some(device_registry_id)
        {
            self.native_prepared_direct_gray_plan = None;
            self.native_prepared_direct_gray_device_registry_id = None;
        }
        if self.native_prepared_direct_gray_plan.is_none() {
            self.ensure_native_image()?;
            let (Some(image), native_context) =
                (self.native_image.as_ref(), &mut self.native_context)
            else {
                return Err(Error::Decode(adapter_backend_error(
                    "native image cache missing".to_string(),
                )));
            };
            let Ok(plan) = image.build_direct_grayscale_plan_with_context(native_context) else {
                return self.decode_repeated_cpu_to_surfaces(fmt, count);
            };
            let plan = Arc::new(plan);
            let prepared = Arc::new(crate::engine::prepare_direct_grayscale_plan(plan.as_ref())?);
            self.native_direct_gray_plan = Some(plan);
            self.native_prepared_direct_gray_plan = Some(prepared);
            self.native_prepared_direct_gray_device_registry_id = Some(device_registry_id);
        }
        let Some(prepared) = self.native_prepared_direct_gray_plan.as_ref() else {
            return self.decode_repeated_cpu_to_surfaces(fmt, count);
        };
        crate::engine::execute_repeated_prepared_direct_grayscale_plan(prepared, fmt, count)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn decode_repeated_color_auto_to_device_routed(
        &mut self,
        fmt: PixelFormat,
        count: usize,
        session: Option<&MetalBackendSession>,
    ) -> Result<Vec<Surface>, Error> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let Some(support) = j2k::J2kDecoder::inspect_support(self.bytes).ok() else {
            return self.decode_repeated_cpu_to_surfaces(fmt, count);
        };
        if !crate::routing::auto_repeated_decode_uses_metal(
            self.inner.info().dimensions,
            fmt,
            count,
            support.transfer_syntax,
            support.payload_kind,
        ) {
            return self.decode_repeated_cpu_to_surfaces(fmt, count);
        }
        match self.decode_repeated_color_direct_to_device_routed(fmt, count, session) {
            Err(error) if error.is_direct_fallback() => {
                self.decode_repeated_cpu_to_surfaces(fmt, count)
            }
            result => result,
        }
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn is_direct_runtime_fallback_error(error: &Error) -> bool {
    error.is_direct_fallback()
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_full_grayscale_batch_direct_to_device_routed(
    inputs: &[Arc<[u8]>],
    fmt: PixelFormat,
    session: Option<&MetalBackendSession>,
) -> Result<Vec<Surface>, Error> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    if !matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
        return Err(Error::MetalKernel {
            message: format!("J2K MetalDirect full grayscale batch does not support {fmt:?}"),
        });
    }

    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal direct grayscale batch plan");
    let mut plans = budget.try_vec(inputs.len(), "J2K Metal direct grayscale plans")?;
    for input in inputs {
        let mut decoder = J2kDecoder::new(input.as_ref())?;
        let plan = match session {
            Some(session) => decoder.ensure_prepared_direct_gray_plan_with_session(fmt, session)?,
            None => decoder.ensure_prepared_direct_gray_plan()?,
        };
        let Some(plan) = plan else {
            return Err(Error::MetalDirectFallback {
                message: format!(
                    "explicit J2K MetalDirect batch currently supports full grayscale Gray8/Gray16 only; fmt={fmt:?}"
                ),
                reason: MetalDirectFallbackReason::UnsupportedPlan,
            });
        };
        plans.push(plan);
    }
    match session {
        Some(session) => crate::engine::with_runtime_for_session(session, |_| {
            crate::engine::execute_prepared_direct_grayscale_plan_batch(&plans, fmt)
        }),
        None => crate::engine::execute_prepared_direct_grayscale_plan_batch(&plans, fmt),
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_full_color_batch_direct_to_device_routed(
    inputs: &[Arc<[u8]>],
    fmt: PixelFormat,
    session: Option<&MetalBackendSession>,
) -> Result<Vec<Surface>, Error> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    if !matches!(
        fmt,
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
    ) {
        return Err(Error::MetalKernel {
            message: format!("J2K MetalDirect full color batch does not support {fmt:?}"),
        });
    }

    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal direct color batch plan");
    let mut plans = budget.try_vec(inputs.len(), "J2K Metal direct color plans")?;
    for input in inputs {
        let mut decoder = J2kDecoder::new(input.as_ref())?;
        let plan = match session {
            Some(session) => {
                decoder.ensure_prepared_direct_color_plan_with_session(fmt, session)?
            }
            None => decoder.ensure_prepared_direct_color_plan()?,
        };
        let Some(plan) = plan else {
            return crate::engine::decode_component_grid_color_batch(inputs, fmt, session);
        };
        plans.push(plan);
    }
    let result = match session {
        Some(session) => crate::engine::with_runtime_for_session(session, |_| {
            crate::engine::execute_prepared_direct_color_plan_batch(&plans, fmt)
        }),
        None => crate::engine::execute_prepared_direct_color_plan_batch(&plans, fmt),
    };
    match result {
        Ok(surfaces) => Ok(surfaces),
        Err(error) if is_direct_runtime_fallback_error(&error) => Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::unsupported_operation(
                CPU_STAGED_METAL_REQUIRES_EXPLICIT_API,
            ),
        )),
        Err(error) => Err(error),
    }
}

#[cfg(target_os = "macos")]
fn adapter_missing_sampled_surface() -> Error {
    Error::MetalStateInvariant {
        state: "sampled full-image decode",
        reason: "one input produced no surface",
    }
}
