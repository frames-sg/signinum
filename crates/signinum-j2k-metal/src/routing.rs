// SPDX-License-Identifier: Apache-2.0

use signinum_core::{BackendRequest, PixelFormat};

use crate::{profile, Error};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouteDecision {
    CpuHost,
    #[cfg(target_os = "macos")]
    MetalKernel,
    RejectExplicitMetal {
        reason: &'static str,
    },
    RejectUnsupportedBackend {
        request: BackendRequest,
    },
    #[cfg(not(target_os = "macos"))]
    MetalUnavailable,
}

pub(crate) fn supports_metal_format(fmt: PixelFormat) -> bool {
    matches!(
        fmt,
        PixelFormat::Gray8
            | PixelFormat::Rgb8
            | PixelFormat::Rgba8
            | PixelFormat::Gray16
            | PixelFormat::Rgb16
    )
}

pub(crate) fn decide_route(backend: BackendRequest, fmt: PixelFormat) -> RouteDecision {
    let decision = match backend {
        BackendRequest::Cpu | BackendRequest::Auto => RouteDecision::CpuHost,
        BackendRequest::Metal => {
            if supports_metal_format(fmt) {
                #[cfg(not(target_os = "macos"))]
                {
                    RouteDecision::MetalUnavailable
                }
                #[cfg(target_os = "macos")]
                {
                    RouteDecision::MetalKernel
                }
            } else {
                RouteDecision::RejectExplicitMetal {
                    reason: unsupported_metal_format_reason(fmt),
                }
            }
        }
        BackendRequest::Cuda => RouteDecision::RejectUnsupportedBackend {
            request: BackendRequest::Cuda,
        },
    };
    if profile::gpu_route_profile_enabled() {
        let request_s = format!("{backend:?}");
        let fmt_s = format!("{fmt:?}");
        let (decision_s, reason_s) = j2k_route_decision_profile(decision);
        profile::emit_gpu_route_profile(
            "j2k",
            "gpu_route",
            "metal",
            &[
                ("request", request_s.as_str()),
                ("fmt", fmt_s.as_str()),
                ("decision", decision_s),
                ("reason", reason_s),
            ],
        );
    }
    decision
}

pub(crate) fn decision_error(decision: RouteDecision) -> Option<Error> {
    match decision {
        RouteDecision::RejectExplicitMetal { reason } => {
            Some(Error::UnsupportedMetalRequest { reason })
        }
        RouteDecision::RejectUnsupportedBackend { request } => {
            Some(Error::UnsupportedBackend { request })
        }
        #[cfg(not(target_os = "macos"))]
        RouteDecision::MetalUnavailable => Some(Error::MetalUnavailable),
        #[cfg(target_os = "macos")]
        RouteDecision::CpuHost | RouteDecision::MetalKernel => None,
        #[cfg(not(target_os = "macos"))]
        RouteDecision::CpuHost => None,
    }
}

fn unsupported_metal_format_reason(fmt: PixelFormat) -> &'static str {
    match fmt {
        PixelFormat::Rgba16 => "J2K Metal does not support PixelFormat::Rgba16",
        _ => "J2K Metal does not support the requested PixelFormat",
    }
}

fn j2k_route_decision_profile(decision: RouteDecision) -> (&'static str, &'static str) {
    match decision {
        RouteDecision::CpuHost => ("cpu_host", "none"),
        #[cfg(target_os = "macos")]
        RouteDecision::MetalKernel => ("metal_kernel", "none"),
        RouteDecision::RejectExplicitMetal { .. } => {
            ("reject_explicit_metal", "unsupported_format")
        }
        RouteDecision::RejectUnsupportedBackend { .. } => {
            ("reject_unsupported_backend", "unsupported_backend")
        }
        #[cfg(not(target_os = "macos"))]
        RouteDecision::MetalUnavailable => ("metal_unavailable", "metal_unavailable"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_route_reports_unsupported_backend() {
        assert_eq!(
            decide_route(BackendRequest::Cuda, PixelFormat::Rgba16),
            RouteDecision::RejectUnsupportedBackend {
                request: BackendRequest::Cuda
            }
        );
        assert!(matches!(
            decision_error(decide_route(BackendRequest::Cuda, PixelFormat::Rgba16)),
            Some(Error::UnsupportedBackend {
                request: BackendRequest::Cuda
            })
        ));
    }

    #[test]
    fn explicit_metal_unsupported_format_is_rejected_before_launch() {
        assert!(matches!(
            decide_route(BackendRequest::Metal, PixelFormat::Rgba16),
            RouteDecision::RejectExplicitMetal { reason } if reason.contains("Rgba16")
        ));
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn explicit_metal_unsupported_format_is_rejected_before_host_unavailability() {
        assert!(matches!(
            decide_route(BackendRequest::Metal, PixelFormat::Rgba16),
            RouteDecision::RejectExplicitMetal { reason } if reason.contains("Rgba16")
        ));
        assert!(matches!(
            decide_route(BackendRequest::Metal, PixelFormat::Rgb8),
            RouteDecision::MetalUnavailable
        ));
    }
}
