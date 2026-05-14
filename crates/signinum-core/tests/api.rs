use signinum_core::{
    copy_tight_pixels_to_strided_output, BackendCapabilities, BackendKind, BackendRequest,
    BufferError, CodecContext, CodecError, CpuFeatures, DecoderContext, DeviceSubmission,
    DeviceSurface, Downscale, ImageCodec, PassthroughCandidate, PassthroughDecision,
    PassthroughRejectReason, PassthroughRequirements, PixelFormat, PixelLayout, ReadySubmission,
    Rect, SampleType, ScratchPool, TileBatchDecodeManyDevice,
};
use signinum_core::{
    CodedUnitLayout, Colorspace, CompressedPayloadKind, CompressedTransferSyntax, Info, TileLayout,
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
fn copy_tight_pixels_to_strided_output_copies_exact_rows() {
    let src = [1, 2, 3, 4, 5, 6];
    let mut out = [0; 6];

    copy_tight_pixels_to_strided_output(&src, (2, 1), PixelFormat::Rgb8, &mut out, 6)
        .expect("copy exact rows");

    assert_eq!(out, src);
}

#[test]
fn copy_tight_pixels_to_strided_output_preserves_row_padding() {
    let src = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut out = [0xee; 12];

    copy_tight_pixels_to_strided_output(&src, (2, 2), PixelFormat::Gray16, &mut out, 6)
        .expect("copy padded rows");

    assert_eq!(out, [1, 2, 3, 4, 0xee, 0xee, 5, 6, 7, 8, 0xee, 0xee]);
}

#[test]
fn copy_tight_pixels_to_strided_output_accepts_empty_height() {
    let mut out = [0xee; 2];

    copy_tight_pixels_to_strided_output(&[], (3, 0), PixelFormat::Rgba8, &mut out, 0)
        .expect("copy zero rows");

    assert_eq!(out, [0xee, 0xee]);
}

#[test]
fn copy_tight_pixels_to_strided_output_accepts_empty_width() {
    let mut out = [0xee; 2];

    copy_tight_pixels_to_strided_output(&[], (0, 3), PixelFormat::Rgba8, &mut out, 0)
        .expect("copy zero-width rows");

    assert_eq!(out, [0xee, 0xee]);
}

#[test]
fn copy_tight_pixels_to_strided_output_rejects_short_source() {
    let mut out = [0; 6];

    let err = copy_tight_pixels_to_strided_output(
        &[1, 2, 3, 4, 5],
        (2, 1),
        PixelFormat::Rgb8,
        &mut out,
        6,
    )
    .expect_err("source too small");

    assert_eq!(
        err,
        BufferError::InputTooSmall {
            required: 6,
            have: 5,
        }
    );
}

#[test]
fn copy_tight_pixels_to_strided_output_rejects_small_stride() {
    let mut out = [0; 6];

    let err = copy_tight_pixels_to_strided_output(
        &[1, 2, 3, 4, 5, 6],
        (2, 1),
        PixelFormat::Rgb8,
        &mut out,
        5,
    )
    .expect_err("stride too small");

    assert_eq!(
        err,
        BufferError::StrideTooSmall {
            row_bytes: 6,
            stride: 5,
        }
    );
}

#[test]
fn copy_tight_pixels_to_strided_output_rejects_small_output() {
    let src = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut out = [0; 9];

    let err = copy_tight_pixels_to_strided_output(&src, (2, 2), PixelFormat::Gray16, &mut out, 6)
        .expect_err("output too small");

    assert_eq!(
        err,
        BufferError::OutputTooSmall {
            required: 10,
            have: 9,
        }
    );
}

#[test]
fn copy_tight_pixels_to_strided_output_rejects_strided_output_overflow() {
    let mut out = [];

    let err = copy_tight_pixels_to_strided_output(
        &[1, 2],
        (1, 2),
        PixelFormat::Gray8,
        &mut out,
        usize::MAX,
    )
    .expect_err("strided output overflows");

    assert_eq!(
        err,
        BufferError::SizeOverflow {
            what: "strided output size",
        }
    );
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

#[derive(Default)]
struct DummyPool;

impl ScratchPool for DummyPool {
    fn bytes_allocated(&self) -> usize {
        0
    }

    fn reset(&mut self) {}
}

#[derive(Debug, thiserror::Error)]
#[error("dummy decode error")]
struct DummyError;

impl CodecError for DummyError {
    fn is_truncated(&self) -> bool {
        false
    }

    fn is_not_implemented(&self) -> bool {
        false
    }

    fn is_unsupported(&self) -> bool {
        false
    }

    fn is_buffer_error(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy)]
struct DummyCodec;

#[derive(Default)]
struct DummyContext;

impl CodecContext for DummyContext {
    fn clear(&mut self) {}
}

impl ImageCodec for DummyCodec {
    type Error = DummyError;
    type Warning = core::convert::Infallible;
    type Pool = DummyPool;
}

impl TileBatchDecodeManyDevice for DummyCodec {
    type Context = DummyContext;
    type DeviceSurface = DummySurface;

    fn decode_tiles_to_device(
        _ctx: &mut DecoderContext<Self::Context>,
        _pool: &mut Self::Pool,
        inputs: &[&[u8]],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Vec<Self::DeviceSurface>, Self::Error> {
        Ok(inputs
            .iter()
            .map(|input| DummySurface {
                backend: match backend {
                    BackendRequest::Cuda => BackendKind::Cuda,
                    BackendRequest::Metal => BackendKind::Metal,
                    BackendRequest::Auto | BackendRequest::Cpu => BackendKind::Cpu,
                },
                dims: (
                    u32::try_from(input.len()).expect("dummy input length fits in u32"),
                    1,
                ),
                fmt,
                len: input.len() * fmt.bytes_per_pixel(),
            })
            .collect())
    }
}

#[test]
fn tile_batch_decode_many_device_returns_ordered_surfaces() {
    let mut ctx = DecoderContext::<DummyContext>::new();
    let mut pool = DummyPool;
    let inputs: [&[u8]; 2] = [b"abc".as_slice(), b"abcdef".as_slice()];

    let surfaces = DummyCodec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Rgb8,
        BackendRequest::Cuda,
    )
    .expect("batch surfaces");

    assert_eq!(surfaces.len(), 2);
    assert_eq!(surfaces[0].backend_kind(), BackendKind::Cuda);
    assert_eq!(surfaces[0].dimensions(), (3, 1));
    assert_eq!(surfaces[1].dimensions(), (6, 1));
    assert_eq!(surfaces[1].byte_len(), 18);
}

fn passthrough_info() -> Info {
    Info {
        dimensions: (512, 512),
        components: 3,
        colorspace: Colorspace::SRgb,
        bit_depth: 8,
        tile_layout: Some(TileLayout {
            tile_width: 512,
            tile_height: 512,
            tiles_x: 1,
            tiles_y: 1,
        }),
        coded_unit_layout: Some(CodedUnitLayout {
            unit_width: 512,
            unit_height: 512,
            units_x: 1,
            units_y: 1,
        }),
        restart_interval: None,
        resolution_levels: 1,
    }
}

#[test]
fn passthrough_candidate_copies_when_syntax_payload_and_metadata_match() {
    let bytes = [0xff, 0x4f, 0xff, 0xd9];
    let candidate = PassthroughCandidate::new(
        &bytes,
        CompressedTransferSyntax::Jpeg2000Lossless,
        CompressedPayloadKind::Jpeg2000Codestream,
        passthrough_info(),
    );
    let requirements = PassthroughRequirements::new(
        CompressedTransferSyntax::Jpeg2000Lossless,
        CompressedPayloadKind::Jpeg2000Codestream,
    )
    .with_dimensions((512, 512))
    .with_components(3)
    .with_bit_depth(8)
    .with_colorspace(Colorspace::SRgb);

    assert_eq!(
        candidate.evaluate(&requirements),
        PassthroughDecision::Copy { bytes: &bytes }
    );
    assert!(core::ptr::eq(
        candidate
            .copy_bytes_if_eligible(&requirements)
            .expect("eligible bytes")
            .as_ptr(),
        bytes.as_ptr()
    ));
}

#[test]
fn passthrough_candidate_rejects_transfer_syntax_mismatch_before_metadata() {
    let bytes = [0xff, 0x4f, 0xff, 0xd9];
    let candidate = PassthroughCandidate::new(
        &bytes,
        CompressedTransferSyntax::Jpeg2000Lossless,
        CompressedPayloadKind::Jpeg2000Codestream,
        passthrough_info(),
    );
    let requirements = PassthroughRequirements::new(
        CompressedTransferSyntax::HtJpeg2000Lossless,
        CompressedPayloadKind::Jpeg2000Codestream,
    )
    .with_dimensions((256, 256));

    assert_eq!(
        candidate.evaluate(&requirements),
        PassthroughDecision::Transcode {
            reason: PassthroughRejectReason::TransferSyntaxMismatch {
                source: CompressedTransferSyntax::Jpeg2000Lossless,
                destination: CompressedTransferSyntax::HtJpeg2000Lossless,
            }
        }
    );
}

#[test]
fn passthrough_candidate_rejects_jp2_container_for_dicom_codestream_payload() {
    let bytes = [0, 0, 0, 12, b'j', b'P', b' ', b' '];
    let candidate = PassthroughCandidate::new(
        &bytes,
        CompressedTransferSyntax::Jpeg2000Lossless,
        CompressedPayloadKind::Jp2File,
        passthrough_info(),
    );
    let requirements = PassthroughRequirements::new(
        CompressedTransferSyntax::Jpeg2000Lossless,
        CompressedPayloadKind::Jpeg2000Codestream,
    );

    assert_eq!(
        candidate.copy_bytes_if_eligible(&requirements),
        Err(PassthroughRejectReason::PayloadKindMismatch {
            source: CompressedPayloadKind::Jp2File,
            destination: CompressedPayloadKind::Jpeg2000Codestream,
        })
    );
}
