use signinum_core::{
    BackendRequest, CodecError, DecoderContext, DeviceSubmission, DeviceSurface, Downscale,
    ImageDecode, ImageDecodeDevice, ImageDecodeSubmit, PixelFormat, Rect, TileBatchDecodeDevice,
    TileBatchDecodeManyDevice,
};
use signinum_jpeg_cuda::{Codec, CudaSession, Decoder, Error};

const BASELINE_420: &[u8] = include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg");
const NVJPEG_RGB8_MAX_CHANNEL_DELTA: u8 = 16;

#[test]
fn auto_falls_back_to_cpu_surface() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), signinum_core::BackendKind::Cpu);
    assert!(surface.as_host_bytes().is_some());
}

#[test]
fn explicit_cuda_request_returns_cuda_surface_or_clear_unavailable_error() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    match decoder.decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda) {
        Ok(surface) => {
            assert_eq!(surface.backend_kind(), signinum_core::BackendKind::Cuda);
            assert_eq!(surface.as_host_bytes(), None);
            #[cfg(feature = "cuda-runtime")]
            assert_ne!(
                surface.cuda_surface().expect("cuda surface").device_ptr(),
                0
            );
        }
        Err(error) => assert!(error.is_unsupported()),
    }
}

#[test]
fn explicit_cuda_request_validates_decode_before_upload() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");

    let error = decoder
        .decode_to_device(PixelFormat::Rgba16, BackendRequest::Cuda)
        .expect_err("unsupported decode");
    assert!(error.is_unsupported());
    assert!(!matches!(error, Error::CudaUnavailable));
}

#[test]
fn explicit_cuda_request_returns_cuda_surface_when_runtime_required() {
    if !runtime_required() {
        return;
    }

    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda)
        .expect("cuda surface");
    assert_eq!(surface.backend_kind(), signinum_core::BackendKind::Cuda);
    assert_eq!(surface.as_host_bytes(), None);
    assert_cuda_surface(&surface);
    assert_eq!(surface.dimensions(), (16, 16));

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download cuda surface");

    let (expected, _) = signinum_jpeg::Decoder::new(BASELINE_420)
        .expect("host decoder")
        .decode(PixelFormat::Rgb8)
        .expect("host decode");
    assert_surface_bytes_match_or_are_close(&surface, &downloaded, &expected);
}

#[test]
fn explicit_cuda_region_scaled_surface_matches_host_when_runtime_required() {
    if !runtime_required() {
        return;
    }

    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);

    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let surface = decoder
        .decode_region_scaled_to_device(PixelFormat::Rgb8, roi, scale, BackendRequest::Cuda)
        .expect("cuda region+scaled surface");
    assert_eq!(surface.backend_kind(), signinum_core::BackendKind::Cuda);
    assert_eq!(surface.as_host_bytes(), None);
    assert_cuda_surface(&surface);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download cuda surface");

    let (expected, _) = signinum_jpeg::Decoder::new(BASELINE_420)
        .expect("host decoder")
        .decode_region_scaled(
            PixelFormat::Rgb8,
            signinum_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        )
        .expect("host decode");
    assert_eq!(downloaded, expected);
}

#[test]
fn explicit_cuda_download_respects_padded_stride_when_runtime_required() {
    if !runtime_required() {
        return;
    }

    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda)
        .expect("cuda surface");
    assert_cuda_surface(&surface);
    let row_bytes = surface.pitch_bytes();
    let stride = row_bytes + 5;
    let mut downloaded = vec![0xCD; stride * surface.dimensions().1 as usize];
    surface
        .download_into(&mut downloaded, stride)
        .expect("download cuda surface");

    let (expected, _) = signinum_jpeg::Decoder::new(BASELINE_420)
        .expect("host decoder")
        .decode(PixelFormat::Rgb8)
        .expect("host decode");
    for (row, expected_row) in expected.chunks(row_bytes).enumerate() {
        let start = row * stride;
        assert_surface_bytes_match_or_are_close(
            &surface,
            &downloaded[start..start + row_bytes],
            expected_row,
        );
        assert_eq!(&downloaded[start + row_bytes..start + stride], &[0xCD; 5]);
    }
}

#[test]
fn explicit_cuda_full_frame_uses_hardware_decode_when_required() {
    if !hardware_decode_required() {
        return;
    }

    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda)
        .expect("cuda surface");
    let cuda = surface.cuda_surface().expect("cuda surface");
    let stats = cuda.stats();
    assert!(
        stats.used_hardware_decode(),
        "explicit full-frame RGB8 CUDA decode must use a CUDA JPEG decode path when required"
    );
    assert!(
        stats.decode_kernel_dispatches() > 0,
        "hardware decode path must report decode kernel dispatches"
    );
    assert_eq!(
        stats.copy_kernel_dispatches(),
        0,
        "hardware decode path should not be reported as the CPU decode plus copy fallback"
    );
}

fn runtime_required() -> bool {
    std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_some()
}

fn hardware_decode_required() -> bool {
    std::env::var_os("SIGNINUM_REQUIRE_CUDA_JPEG_HARDWARE_DECODE").is_some()
}

fn assert_cuda_surface(surface: &signinum_jpeg_cuda::Surface) {
    let cuda = surface.cuda_surface().expect("cuda surface");
    assert_ne!(cuda.device_ptr(), 0);
    assert!(cuda.stats().kernel_dispatches() > 0);
}

fn assert_surface_bytes_match_or_are_close(
    surface: &signinum_jpeg_cuda::Surface,
    actual: &[u8],
    expected: &[u8],
) {
    assert_eq!(actual.len(), expected.len());
    let stats = surface.cuda_surface().expect("cuda surface").stats();
    if !stats.used_hardware_decode() {
        assert_eq!(actual, expected);
        return;
    }

    let max_delta = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| actual.abs_diff(*expected))
        .max()
        .unwrap_or(0);
    assert!(
        max_delta <= NVJPEG_RGB8_MAX_CHANNEL_DELTA,
        "nvJPEG decode differed from the CPU reference by max channel delta {max_delta}"
    );
}

#[test]
fn submit_to_device_auto_falls_back_to_cpu_surface() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut session = CudaSession::default();
    let surface = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut decoder,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Auto,
    )
    .expect("submission")
    .wait()
    .expect("surface");
    assert_eq!(surface.backend_kind(), signinum_core::BackendKind::Cpu);
    assert!(surface.as_host_bytes().is_some());
    assert!(session.submissions() >= 1);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn submit_to_device_auto_does_not_initialize_cuda_runtime() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut session = CudaSession::default();
    let surface = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut decoder,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Auto,
    )
    .expect("submission")
    .wait()
    .expect("surface");
    assert_eq!(surface.backend_kind(), signinum_core::BackendKind::Cpu);
    assert_eq!(session.submissions(), 1);
    assert!(!session.is_runtime_initialized());
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn explicit_cuda_submissions_reuse_session_runtime_when_required() {
    if !runtime_required() {
        return;
    }

    let mut session = CudaSession::default();
    assert!(!session.is_runtime_initialized());

    let mut first = Decoder::new(BASELINE_420).expect("decoder");
    let first_surface = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut first,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Cuda,
    )
    .expect("first submission")
    .wait()
    .expect("first surface");
    assert_eq!(
        first_surface.backend_kind(),
        signinum_core::BackendKind::Cuda
    );
    assert_cuda_surface(&first_surface);
    assert!(session.is_runtime_initialized());

    let mut second = Decoder::new(BASELINE_420).expect("decoder");
    let second_surface = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut second,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Cuda,
    )
    .expect("second submission")
    .wait()
    .expect("second surface");
    assert_eq!(
        second_surface.backend_kind(),
        signinum_core::BackendKind::Cuda
    );
    assert_cuda_surface(&second_surface);
    assert_eq!(session.submissions(), 2);
    assert!(session.is_runtime_initialized());
}

#[test]
fn auto_region_scaled_surface_matches_host_decode() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);

    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let surface = decoder
        .decode_region_scaled_to_device(PixelFormat::Rgb8, roi, scale, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), signinum_core::BackendKind::Cpu);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));

    let mut host_decoder = Decoder::new(BASELINE_420).expect("host decoder");
    let mut host = vec![0u8; scaled.w as usize * scaled.h as usize * 3];
    host_decoder
        .decode_region_scaled_into(
            &mut signinum_jpeg::ScratchPool::new(),
            &mut host,
            scaled.w as usize * 3,
            PixelFormat::Rgb8,
            roi,
            scale,
        )
        .expect("host decode");
    assert_eq!(surface.as_host_bytes(), Some(host.as_slice()));
}

#[test]
fn tile_batch_region_scaled_auto_surface_matches_host_decode() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let mut ctx = DecoderContext::<signinum_jpeg::DecoderContext>::new();
    let mut pool = signinum_jpeg::ScratchPool::new();
    let surface = Codec::decode_tile_region_scaled_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        roi,
        scale,
        BackendRequest::Auto,
    )
    .expect("surface");
    assert_eq!(surface.backend_kind(), signinum_core::BackendKind::Cpu);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));

    let (expected, _) = signinum_jpeg::Decoder::new(BASELINE_420)
        .expect("host decoder")
        .decode_region_scaled(
            PixelFormat::Rgb8,
            signinum_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        )
        .expect("host decode");
    assert_eq!(surface.as_host_bytes(), Some(expected.as_slice()));
}

#[test]
fn tile_batch_region_scaled_cuda_surface_matches_host_when_runtime_required() {
    if !runtime_required() {
        return;
    }

    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let mut ctx = DecoderContext::<signinum_jpeg::DecoderContext>::new();
    let mut pool = signinum_jpeg::ScratchPool::new();
    let surface = Codec::decode_tile_region_scaled_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        roi,
        scale,
        BackendRequest::Cuda,
    )
    .expect("cuda tile batch surface");
    assert_eq!(surface.backend_kind(), signinum_core::BackendKind::Cuda);
    assert_eq!(surface.as_host_bytes(), None);
    assert_cuda_surface(&surface);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download cuda surface");

    let (expected, _) = signinum_jpeg::Decoder::new(BASELINE_420)
        .expect("host decoder")
        .decode_region_scaled(
            PixelFormat::Rgb8,
            signinum_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        )
        .expect("host decode");
    assert_eq!(downloaded, expected);
}

#[test]
fn decode_tiles_to_device_auto_preserves_order_and_matches_host_bytes() {
    let mut ctx = DecoderContext::<signinum_jpeg::DecoderContext>::new();
    let mut pool = signinum_jpeg::ScratchPool::new();
    let inputs = [BASELINE_420, BASELINE_420];

    let surfaces = Codec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Rgb8,
        BackendRequest::Auto,
    )
    .expect("batch surfaces");

    assert_eq!(surfaces.len(), inputs.len());
    let (expected, _) = signinum_jpeg::Decoder::new(BASELINE_420)
        .expect("host decoder")
        .decode(PixelFormat::Rgb8)
        .expect("host decode");
    for surface in surfaces {
        assert_eq!(surface.dimensions(), (16, 16));
        match surface.backend_kind() {
            signinum_core::BackendKind::Cpu => {
                assert_eq!(surface.as_host_bytes(), Some(expected.as_slice()));
            }
            signinum_core::BackendKind::Cuda => {
                let mut downloaded = vec![0u8; surface.byte_len()];
                surface
                    .download_into(&mut downloaded, surface.pitch_bytes())
                    .expect("download cuda surface");
                assert_surface_bytes_match_or_are_close(&surface, &downloaded, &expected);
            }
            signinum_core::BackendKind::Metal => panic!("JPEG CUDA batch returned Metal surface"),
        }
    }
}

#[test]
fn decode_tiles_to_device_explicit_cuda_returns_cuda_surfaces_or_clear_unavailable_error() {
    let mut ctx = DecoderContext::<signinum_jpeg::DecoderContext>::new();
    let mut pool = signinum_jpeg::ScratchPool::new();
    let inputs = [BASELINE_420, BASELINE_420];

    match Codec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Rgb8,
        BackendRequest::Cuda,
    ) {
        Ok(surfaces) => {
            assert_eq!(surfaces.len(), inputs.len());
            for surface in surfaces {
                assert_eq!(surface.backend_kind(), signinum_core::BackendKind::Cuda);
                assert_eq!(surface.as_host_bytes(), None);
                assert_cuda_surface(&surface);
            }
        }
        Err(error) => assert!(error.is_unsupported()),
    }
}

#[test]
fn decode_tiles_to_device_explicit_cuda_uses_hardware_decode_when_required() {
    if !hardware_decode_required() {
        return;
    }

    let mut ctx = DecoderContext::<signinum_jpeg::DecoderContext>::new();
    let mut pool = signinum_jpeg::ScratchPool::new();
    let inputs = [BASELINE_420, BASELINE_420];

    let surfaces = Codec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Rgb8,
        BackendRequest::Cuda,
    )
    .expect("cuda batch surfaces");

    assert_eq!(surfaces.len(), inputs.len());
    for surface in surfaces {
        let stats = surface.cuda_surface().expect("cuda surface").stats();
        assert!(
            stats.used_hardware_decode(),
            "explicit full-tile RGB8 CUDA batch decode must use nvJPEG when required"
        );
        assert!(
            stats.decode_kernel_dispatches() > 0,
            "hardware batch decode path must report decode dispatches"
        );
        assert_eq!(
            stats.copy_kernel_dispatches(),
            0,
            "hardware batch decode path should not be reported as CPU decode plus copy"
        );
    }
}
