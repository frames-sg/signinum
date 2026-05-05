// SPDX-License-Identifier: Apache-2.0

use signinum_core::{BackendRequest, PixelFormat};
use signinum_jpeg::{
    adapter::{JpegMetalFast420PacketV1, JpegMetalFast422PacketV1, JpegMetalFast444PacketV1},
    Decoder as CpuDecoder,
};

use crate::{batch::BatchOp, Error};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JpegMetalCapabilities {
    has_fast_packet: bool,
    supports_output_format: bool,
}

impl JpegMetalCapabilities {
    pub(crate) fn for_request(
        _decoder: &CpuDecoder<'_>,
        fmt: PixelFormat,
        _op: BatchOp,
        fast444_packet: Option<&JpegMetalFast444PacketV1>,
        fast422_packet: Option<&JpegMetalFast422PacketV1>,
        fast420_packet: Option<&JpegMetalFast420PacketV1>,
    ) -> Self {
        let has_fast_packet =
            fast444_packet.is_some() || fast422_packet.is_some() || fast420_packet.is_some();
        let supports_output_format = supports_metal_output_format(fmt);

        Self {
            has_fast_packet,
            supports_output_format,
        }
    }
}

pub(crate) fn decide_route(
    backend: BackendRequest,
    capabilities: JpegMetalCapabilities,
) -> RouteDecision {
    match backend {
        BackendRequest::Cpu => RouteDecision::CpuHost,
        BackendRequest::Auto => {
            let _ = capabilities;
            RouteDecision::CpuHost
        }
        BackendRequest::Metal => {
            if !capabilities.has_fast_packet {
                return RouteDecision::RejectExplicitMetal {
                    reason: "JPEG Metal supports explicit requests only for fast 4:2:0, 4:2:2, or 4:4:4 baseline packets",
                };
            }
            if !capabilities.supports_output_format {
                return RouteDecision::RejectExplicitMetal {
                    reason: "JPEG Metal supports explicit requests only for Gray8, Rgb8, or Rgba8 output formats",
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

fn supports_metal_output_format(fmt: PixelFormat) -> bool {
    matches!(
        fmt,
        PixelFormat::Gray8 | PixelFormat::Rgb8 | PixelFormat::Rgba8
    )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_route_reports_unsupported_backend() {
        let capabilities = JpegMetalCapabilities {
            has_fast_packet: true,
            supports_output_format: true,
        };

        assert_eq!(
            decide_route(BackendRequest::Cuda, capabilities),
            RouteDecision::RejectUnsupportedBackend {
                request: BackendRequest::Cuda
            }
        );
        assert!(matches!(
            decision_error(decide_route(BackendRequest::Cuda, capabilities)),
            Some(Error::UnsupportedBackend {
                request: BackendRequest::Cuda
            })
        ));
    }

    #[test]
    fn explicit_metal_unsupported_output_format_is_rejected_before_launch() {
        let capabilities = JpegMetalCapabilities {
            has_fast_packet: true,
            supports_output_format: false,
        };

        assert!(matches!(
            decide_route(BackendRequest::Metal, capabilities),
            RouteDecision::RejectExplicitMetal { reason }
                if reason.contains("Gray8, Rgb8, or Rgba8")
        ));
    }

    #[test]
    fn auto_routes_single_request_to_cpu_even_when_metal_capabilities_match() {
        let capabilities = JpegMetalCapabilities {
            has_fast_packet: true,
            supports_output_format: true,
        };

        assert_eq!(
            decide_route(BackendRequest::Auto, capabilities),
            RouteDecision::CpuHost
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn auto_routes_to_cpu_host_on_non_macos_even_when_metal_would_be_preferred() {
        let capabilities = JpegMetalCapabilities {
            has_fast_packet: true,
            supports_output_format: true,
        };

        assert_eq!(
            decide_route(BackendRequest::Auto, capabilities),
            RouteDecision::CpuHost
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn explicit_metal_unsupported_shape_is_rejected_before_host_unavailability() {
        let capabilities = JpegMetalCapabilities {
            has_fast_packet: false,
            supports_output_format: true,
        };

        assert!(matches!(
            decide_route(BackendRequest::Metal, capabilities),
            RouteDecision::RejectExplicitMetal { reason }
                if reason.contains("JPEG Metal")
        ));
    }
}
