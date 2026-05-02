// SPDX-License-Identifier: Apache-2.0

use signinum_core::{BackendRequest, PixelFormat};

use crate::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouteDecision {
    CpuHost,
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    MetalKernel,
    RejectExplicitMetal {
        reason: &'static str,
    },
    RejectUnsupportedBackend {
        request: BackendRequest,
    },
    #[cfg_attr(target_os = "macos", allow(dead_code))]
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
    match backend {
        BackendRequest::Cpu | BackendRequest::Auto => RouteDecision::CpuHost,
        BackendRequest::Metal => {
            if !supports_metal_format(fmt) {
                return RouteDecision::RejectExplicitMetal {
                    reason: unsupported_metal_format_reason(fmt),
                };
            }

            #[cfg(not(target_os = "macos"))]
            {
                RouteDecision::MetalUnavailable
            }
            #[cfg(target_os = "macos")]
            {
                RouteDecision::MetalKernel
            }
        }
        BackendRequest::Cuda => RouteDecision::RejectUnsupportedBackend {
            request: BackendRequest::Cuda,
        },
    }
}

pub(crate) fn decision_error(decision: RouteDecision) -> Option<Error> {
    match decision {
        RouteDecision::RejectExplicitMetal { reason } => {
            Some(Error::UnsupportedMetalRequest { reason })
        }
        RouteDecision::RejectUnsupportedBackend { request } => {
            Some(Error::UnsupportedBackend { request })
        }
        RouteDecision::MetalUnavailable => Some(Error::MetalUnavailable),
        RouteDecision::CpuHost | RouteDecision::MetalKernel => None,
    }
}

fn unsupported_metal_format_reason(fmt: PixelFormat) -> &'static str {
    match fmt {
        PixelFormat::Rgba16 => "J2K Metal does not support PixelFormat::Rgba16",
        _ => "J2K Metal does not support the requested PixelFormat",
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
