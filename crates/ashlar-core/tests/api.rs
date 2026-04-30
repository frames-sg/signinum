use ashlar_core::{
    BackendCapabilities, BackendKind, BackendRequest, CpuFeatures, DeviceSubmission, DeviceSurface,
    Downscale, PixelFormat, PixelLayout, ReadySubmission, Rect, SampleType,
};

#[test]
fn pixel_format_reports_layout_and_sample_type() {
    assert_eq!(PixelFormat::Rgb8.layout(), PixelLayout::Rgb);
    assert_eq!(PixelFormat::Rgb8.sample(), SampleType::U8);
    assert_eq!(PixelFormat::Rgb16.sample(), SampleType::U16);
}

#[test]
fn downscale_reports_expected_denominators() {
    assert_eq!(Downscale::None.denominator(), 1);
    assert_eq!(Downscale::Half.denominator(), 2);
    assert_eq!(Downscale::Quarter.denominator(), 4);
    assert_eq!(Downscale::Eighth.denominator(), 8);
}

#[test]
fn rect_scaled_covering_uses_floor_start_and_ceil_end() {
    let roi = Rect {
        x: 3,
        y: 5,
        w: 10,
        h: 11,
    };

    assert_eq!(
        roi.scaled_covering(Downscale::Quarter),
        Rect {
            x: 0,
            y: 1,
            w: 4,
            h: 3,
        }
    );
    assert_eq!(roi.scaled_covering(Downscale::None), roi);
}

#[test]
fn rect_full_and_is_within_match_existing_jpeg_behavior() {
    let full = Rect::full((640, 480));
    assert_eq!(
        full,
        Rect {
            x: 0,
            y: 0,
            w: 640,
            h: 480,
        }
    );
    assert!(Rect {
        x: 10,
        y: 10,
        w: 100,
        h: 100,
    }
    .is_within((640, 480)));
}

#[test]
fn backend_capabilities_resolve_auto_and_explicit_requests() {
    let caps = BackendCapabilities {
        cpu: CpuFeatures::default(),
        metal: true,
        cuda: false,
    };
    assert_eq!(caps.resolve(BackendRequest::Auto), Some(BackendKind::Metal));
    assert_eq!(caps.resolve(BackendRequest::Cpu), Some(BackendKind::Cpu));
    assert_eq!(
        caps.resolve(BackendRequest::Metal),
        Some(BackendKind::Metal)
    );
    assert_eq!(caps.resolve(BackendRequest::Cuda), None);
    assert!(caps.supports(BackendRequest::Metal));
    assert!(!caps.supports(BackendRequest::Cuda));
}

#[derive(Debug, Clone, Copy)]
struct DummySurface {
    backend: BackendKind,
    dims: (u32, u32),
    fmt: PixelFormat,
    len: usize,
}

impl DeviceSurface for DummySurface {
    fn backend_kind(&self) -> BackendKind {
        self.backend
    }

    fn dimensions(&self) -> (u32, u32) {
        self.dims
    }

    fn pixel_format(&self) -> PixelFormat {
        self.fmt
    }

    fn byte_len(&self) -> usize {
        self.len
    }
}

#[test]
fn device_surface_contract_reports_metadata() {
    let surface = DummySurface {
        backend: BackendKind::Metal,
        dims: (32, 16),
        fmt: PixelFormat::Rgb8,
        len: 32 * 16 * 3,
    };
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (32, 16));
    assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
    assert_eq!(surface.byte_len(), 32 * 16 * 3);
}

#[test]
fn ready_submission_waits_immediate_success() {
    let submission = ReadySubmission::<u32, &'static str>::from_result(Ok(7));
    assert_eq!(submission.wait().expect("success"), 7);
}

#[test]
fn ready_submission_waits_immediate_error() {
    let submission = ReadySubmission::<u32, &'static str>::from_result(Err("nope"));
    assert_eq!(submission.wait().expect_err("error"), "nope");
}
