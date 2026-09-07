// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

#[cfg(target_os = "macos")]
use std::sync::Arc;

use j2k::DeviceDecodeRequest;
use j2k_core::{
    BackendKind, BackendRequest, CodecError, DeviceSurface, Downscale, PixelFormat, Rect,
};

#[cfg(target_os = "macos")]
use super::is_direct_runtime_fallback_error;
use super::surface::upload_surface;
#[cfg(target_os = "macos")]
use super::J2kDecoder;
use super::{DecodeOperation, MetalDecodeRequest};
use crate::{batch, Error, Storage, Surface, SurfaceResidency};
#[cfg(target_os = "macos")]
use crate::{hybrid, MetalBackendSession, MetalDirectFallbackReason};

#[cfg(target_os = "macos")]
mod region_scaled_routing;

#[cfg(target_os = "macos")]
fn should_run_metal_runtime() -> bool {
    j2k_test_support::metal_runtime_gate(module_path!())
}

#[cfg(target_os = "macos")]
#[test]
fn direct_runtime_fallback_classification_uses_structured_variant() {
    let fallback = Error::MetalDirectFallback {
        message: "arbitrary fallback text".to_string(),
        reason: MetalDirectFallbackReason::UnsupportedRuntimeInput,
    };
    assert!(is_direct_runtime_fallback_error(&fallback));

    let message_only = Error::MetalKernel {
        message: "unsupported classic kernel input in direct component plan".to_string(),
    };
    assert!(!is_direct_runtime_fallback_error(&message_only));
}

#[test]
fn metal_decode_request_maps_geometry_to_report_and_batch_ops() {
    let roi = Rect {
        x: 1,
        y: 2,
        w: 3,
        h: 4,
    };
    let requests = [
        (
            MetalDecodeRequest::full(PixelFormat::Gray8, BackendRequest::Auto),
            DecodeOperation::Full,
            batch::BatchOp::Full,
        ),
        (
            MetalDecodeRequest::region(PixelFormat::Gray8, roi, BackendRequest::Auto),
            DecodeOperation::Region,
            batch::BatchOp::Region(roi),
        ),
        (
            MetalDecodeRequest::scaled(PixelFormat::Gray8, Downscale::Half, BackendRequest::Auto),
            DecodeOperation::Scaled,
            batch::BatchOp::Scaled(Downscale::Half),
        ),
        (
            MetalDecodeRequest::region_scaled(
                PixelFormat::Gray8,
                roi,
                Downscale::Quarter,
                BackendRequest::Auto,
            ),
            DecodeOperation::RegionScaled,
            batch::BatchOp::RegionScaled {
                roi,
                scale: Downscale::Quarter,
            },
        ),
    ];

    for (request, report_operation, batch_op) in requests {
        assert_eq!(request.op.report_operation(), report_operation);
        assert_eq!(request.op.batch_op(), batch_op);
    }
}

#[test]
fn metal_decode_operation_lowers_once_to_shared_device_geometry() {
    let roi = Rect {
        x: 1,
        y: 2,
        w: 3,
        h: 4,
    };
    for (operation, expected) in [
        (super::MetalDecodeOp::Full, DeviceDecodeRequest::Full),
        (
            super::MetalDecodeOp::Region(roi),
            DeviceDecodeRequest::Region { roi },
        ),
        (
            super::MetalDecodeOp::Scaled(Downscale::Half),
            DeviceDecodeRequest::Scaled {
                scale: Downscale::Half,
            },
        ),
        (
            super::MetalDecodeOp::RegionScaled {
                roi,
                scale: Downscale::Quarter,
            },
            DeviceDecodeRequest::RegionScaled {
                roi,
                scale: Downscale::Quarter,
            },
        ),
    ] {
        assert_eq!(operation.device_request(), expected);
    }
}

#[test]
fn metal_runtime_failures_are_not_unsupported_errors() {
    for err in [
        Error::MetalRuntime {
            message: "runtime".to_string(),
        },
        Error::MetalKernel {
            message: "kernel".to_string(),
        },
        Error::MetalStatePoisoned {
            state: "J2K Metal session",
        },
    ] {
        assert!(!err.is_unsupported(), "{err:?}");
    }
}

#[test]
fn cpu_uploaded_surface_reports_host_residency() {
    let surface = upload_surface(
        vec![1, 2, 3],
        (1, 1),
        PixelFormat::Rgb8,
        BackendRequest::Cpu,
    )
    .expect("create CPU surface");

    assert_eq!(surface.backend_kind(), BackendKind::Cpu);
    assert_eq!(surface.residency(), SurfaceResidency::Host);
    #[cfg(target_os = "macos")]
    assert!(surface.metal_buffer_trusted().is_none());
}

#[test]
fn download_into_reports_inconsistent_surface_storage_range() {
    let surface = Surface {
        backend: BackendKind::Cpu,
        residency: SurfaceResidency::Host,
        dimensions: (2, 1),
        fmt: PixelFormat::Gray8,
        pitch_bytes: 2,
        byte_offset: 0,
        storage: Storage::from_host(vec![7]),
    };
    let mut out = [0_u8; 2];

    let err = surface
        .download_into(&mut out, 2)
        .expect_err("inconsistent surface storage should be reported");

    assert!(matches!(
        err,
        Error::MetalKernel { message }
            if message == "J2K Metal surface byte range 0..2 exceeds storage length 1"
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn metal_backend_sessions_own_distinct_direct_plan_caches() {
    if !should_run_metal_runtime() {
        return;
    }

    let Ok(device) = j2k_metal_support::system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };

    let first = MetalBackendSession::new(device.clone());
    let second = MetalBackendSession::new(device);

    assert_ne!(
        crate::session::direct_plan_cache::direct_cache_ids_for_test(&first),
        crate::session::direct_plan_cache::direct_cache_ids_for_test(&second)
    );
}

#[cfg(target_os = "macos")]
#[test]
fn fresh_direct_plan_preparation_uses_the_explicit_session_runtime() {
    if !should_run_metal_runtime() {
        return;
    }

    let Ok(device) = j2k_metal_support::system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let pixels = j2k_test_support::gradient_u8(32, 32, 1);
    let bytes = j2k_native::encode(
        &pixels,
        32,
        32,
        1,
        8,
        false,
        &j2k_native::EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..j2k_native::EncodeOptions::default()
        },
    )
    .expect("encode classic grayscale session-runtime fixture");
    let session = MetalBackendSession::new(device.clone());
    let session_runtime = session.runtime().expect("explicit session runtime");

    crate::engine::reset_direct_tier1_input_buffer_prepares_for_test();
    crate::engine::with_isolated_runtime_for_device_for_test(&device, || {
        let mut decoder = J2kDecoder::new(&bytes)?;
        let prepared =
            decoder.ensure_prepared_direct_gray_plan_with_session(PixelFormat::Gray8, &session)?;
        assert!(prepared.is_some());
        Ok(())
    })
    .expect("prepare direct plan with explicit session");

    assert!(
        crate::engine::direct_tier1_input_buffer_prepares_for_test() > 0,
        "fixture must allocate classic Tier-1 input buffers"
    );
    assert_eq!(
        crate::engine::direct_tier1_input_buffer_runtime_for_test(),
        Arc::as_ptr(&session_runtime).addr(),
        "fresh cached buffers must be prepared by the explicit session runtime"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn decoder_local_prepared_binding_is_revalidated_for_session_device() {
    if !should_run_metal_runtime() {
        return;
    }

    let Ok(device) = j2k_metal_support::system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let pixels = j2k_test_support::gradient_u8(32, 32, 1);
    let bytes = j2k_native::encode(
        &pixels,
        32,
        32,
        1,
        8,
        false,
        &j2k_native::EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..j2k_native::EncodeOptions::default()
        },
    )
    .expect("encode classic grayscale device-identity fixture");
    let session = MetalBackendSession::new(device);
    let expected_device = session.device().registryID();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let first = decoder
        .ensure_prepared_direct_gray_plan_with_session(PixelFormat::Gray8, &session)
        .expect("first prepared plan")
        .expect("supported first prepared plan");

    decoder.native_prepared_direct_gray_device_registry_id = Some(u64::MAX);
    let second = decoder
        .ensure_prepared_direct_gray_plan_with_session(PixelFormat::Gray8, &session)
        .expect("device-scoped prepared plan")
        .expect("supported device-scoped prepared plan");

    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(
        decoder.native_prepared_direct_gray_device_registry_id,
        Some(expected_device)
    );
}

#[cfg(target_os = "macos")]
#[test]
fn repeated_session_hits_share_native_and_prepared_plan_owners() {
    if !should_run_metal_runtime() {
        return;
    }

    let Ok(device) = j2k_metal_support::system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let pixels = j2k_test_support::gradient_u8(32, 32, 3);
    let bytes = j2k_native::encode(
        &pixels,
        32,
        32,
        3,
        8,
        false,
        &j2k_native::EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..j2k_native::EncodeOptions::default()
        },
    )
    .expect("encode RGB cache fixture");
    let session = MetalBackendSession::new(device);
    let request = MetalDecodeRequest::full(PixelFormat::Rgb8, BackendRequest::Metal);

    let mut first = J2kDecoder::new(&bytes).expect("first decoder");
    first
        .decode_request_to_device_with_session(request, &session)
        .expect("first session decode");
    let first_native = first
        .native_direct_color_plan
        .as_ref()
        .expect("first native plan")
        .clone();
    let first_prepared = first
        .native_prepared_direct_color_plan
        .as_ref()
        .expect("first prepared plan")
        .clone();

    let mut second = J2kDecoder::new(&bytes).expect("second decoder");
    second
        .decode_request_to_device_with_session(request, &session)
        .expect("cached session decode");
    let second_native = second
        .native_direct_color_plan
        .as_ref()
        .expect("cached native plan");
    let second_prepared = second
        .native_prepared_direct_color_plan
        .as_ref()
        .expect("cached prepared plan");

    assert!(Arc::ptr_eq(&first_native, second_native));
    assert!(Arc::ptr_eq(&first_prepared, second_prepared));
}

#[cfg(target_os = "macos")]
#[test]
fn explicit_metal_request_does_not_stage_cpu_pixels() {
    if !should_run_metal_runtime() {
        return;
    }

    if j2k_metal_support::system_default_device().is_err() {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    }

    let result = upload_surface(
        vec![1, 2, 3],
        (1, 1),
        PixelFormat::Rgb8,
        BackendRequest::Metal,
    );

    assert!(matches!(
        result,
        Err(Error::UnsupportedMetalRequest { reason })
            if reason.contains("CPU-staged")
                && reason.contains("explicit")
                && reason.contains("Metal")
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn repeated_region_scaled_color_batch_reuses_prepared_plan() {
    if !should_run_metal_runtime() {
        return;
    }

    if j2k_metal_support::system_default_device().is_err() {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    }

    let pixels = j2k_test_support::gradient_u8(64, 64, 3);
    let options = j2k_native::EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..j2k_native::EncodeOptions::default()
    };
    let input = Arc::<[u8]>::from(
        j2k_native::encode(&pixels, 64, 64, 3, 8, false, &options).expect("encode rgb8"),
    );
    let roi = Rect {
        x: 8,
        y: 8,
        w: 32,
        h: 32,
    };
    let scale = Downscale::Quarter;
    let requests = vec![(input.clone(), roi, scale); 4];
    let _guard = hybrid::region_scaled_color_plan_test_lock_for_test();
    hybrid::reset_region_scaled_color_plan_builds_for_test();

    let surfaces =
        hybrid::decode_region_scaled_color_batch_direct_to_device(&requests, PixelFormat::Rgb8)
            .expect("repeated RGB region-scaled batch");

    assert_eq!(surfaces.len(), requests.len());
    assert_eq!(
        hybrid::region_scaled_color_plan_builds_for_test(),
        1,
        "repeated RGB ROI+scaled batches should build and crop one prepared direct color plan"
    );
}

#[test]
fn decoder_modules_remain_explicit_without_suppression_shortcuts() {
    use std::collections::BTreeSet;
    use syn::{Item, UseTree};

    fn use_tree_has_glob(tree: &UseTree) -> bool {
        match tree {
            UseTree::Glob(_) => true,
            UseTree::Group(group) => group.items.iter().any(use_tree_has_glob),
            UseTree::Path(path) => use_tree_has_glob(&path.tree),
            UseTree::Name(_) | UseTree::Rename(_) => false,
        }
    }

    fn assert_explicit_items(name: &str, items: &[Item]) {
        for item in items {
            match item {
                Item::Macro(item) => assert!(
                    !item.mac.path.is_ident("include"),
                    "{name} must be a real parsed module"
                ),
                Item::Mod(module) => {
                    if let Some((_, nested)) = &module.content {
                        assert_explicit_items(name, nested);
                    }
                }
                Item::Use(item) => assert!(
                    !use_tree_has_glob(&item.tree),
                    "{name} must use explicit imports"
                ),
                _ => {}
            }
        }
    }

    fn parse_explicit_module(name: &str, source: &str) -> syn::File {
        let file = syn::parse_file(source)
            .unwrap_or_else(|error| panic!("parse decoder module {name}: {error}"));
        assert!(
            !file.attrs.iter().any(|attr| attr.path().is_ident("allow")),
            "{name} must not use module-wide lint suppression"
        );
        assert_explicit_items(name, &file.items);
        file
    }

    const MODULES: [(&str, &str); 6] = [
        ("adapters.rs", include_str!("adapters.rs")),
        ("core.rs", include_str!("core.rs")),
        ("direct_paths.rs", include_str!("direct_paths.rs")),
        ("request.rs", include_str!("request.rs")),
        ("routes.rs", include_str!("routes.rs")),
        ("surface.rs", include_str!("surface.rs")),
    ];

    let root = parse_explicit_module("decoder.rs", include_str!("../decoder.rs"));
    let external_modules = root
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Mod(module) if module.content.is_none() => Some(module.ident.to_string()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let expected_modules = [
        "adapters",
        "core",
        "direct_paths",
        "request",
        "routes",
        "surface",
        "tests",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<BTreeSet<_>>();
    assert_eq!(external_modules, expected_modules);

    for (name, source) in MODULES {
        parse_explicit_module(name, source);
    }
}

#[cfg(target_os = "macos")]
mod sampled;
