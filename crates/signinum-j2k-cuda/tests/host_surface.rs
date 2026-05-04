use signinum_core::{
    BackendRequest, CodecError, DecoderContext, DeviceSubmission, DeviceSurface, Downscale,
    ImageDecode, ImageDecodeDevice, ImageDecodeSubmit, PixelFormat, Rect, TileBatchDecodeDevice,
    TileBatchDecodeManyDevice,
};
use signinum_j2k_cuda::{Codec, CudaSession, Error, J2kDecoder};
use signinum_j2k_native::{encode, encode_htj2k, EncodeOptions};

fn fixture() -> Vec<u8> {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 2, 3, 8, false, &options).expect("encode")
}

fn fixture_ht_gray8() -> Vec<u8> {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, 4, 4, 1, 8, false, &options).expect("encode ht gray8")
}

#[test]
fn auto_falls_back_to_cpu_surface() {
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), signinum_core::BackendKind::Cpu);
    assert!(surface.as_host_bytes().is_some());
}

#[test]
fn explicit_cuda_request_returns_cuda_surface_or_clear_unavailable_error() {
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
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
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");

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

    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda)
        .expect("cuda surface");
    assert_eq!(surface.backend_kind(), signinum_core::BackendKind::Cuda);
    assert_eq!(surface.as_host_bytes(), None);
    assert_cuda_surface(&surface);
    assert_eq!(surface.dimensions(), (2, 2));

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download cuda surface");

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut expected = [0u8; 12];
    host_decoder
        .decode_into(&mut expected, 6, PixelFormat::Rgb8)
        .expect("host decode");
    assert_eq!(downloaded, expected);
}

#[test]
fn explicit_cuda_region_scaled_surface_matches_host_when_runtime_required() {
    if !runtime_required() {
        return;
    }

    let bytes = fixture_ht_gray8();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_region_scaled_to_device(PixelFormat::Gray8, roi, scale, BackendRequest::Cuda)
        .expect("cuda region+scaled surface");
    assert_eq!(surface.backend_kind(), signinum_core::BackendKind::Cuda);
    assert_eq!(surface.as_host_bytes(), None);
    assert_cuda_surface(&surface);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download cuda surface");

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut expected = vec![0u8; scaled.w as usize * scaled.h as usize];
    host_decoder
        .decode_region_scaled_into(
            &mut signinum_j2k_cuda::J2kScratchPool::new(),
            &mut expected,
            scaled.w as usize,
            PixelFormat::Gray8,
            roi,
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

    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
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

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut expected = [0u8; 12];
    host_decoder
        .decode_into(&mut expected, row_bytes, PixelFormat::Rgb8)
        .expect("host decode");
    for (row, expected_row) in expected.chunks(row_bytes).enumerate() {
        let start = row * stride;
        assert_eq!(&downloaded[start..start + row_bytes], expected_row);
        assert_eq!(&downloaded[start + row_bytes..start + stride], &[0xCD; 5]);
    }
}

fn runtime_required() -> bool {
    std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_some()
}

fn assert_cuda_surface(surface: &signinum_j2k_cuda::Surface) {
    let cuda = surface.cuda_surface().expect("cuda surface");
    assert_ne!(cuda.device_ptr(), 0);
    assert!(cuda.stats().kernel_dispatches() > 0);
}

#[test]
fn submit_to_device_auto_falls_back_to_cpu_surface() {
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut session = CudaSession::default();
    let surface = <J2kDecoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
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
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut session = CudaSession::default();
    let surface = <J2kDecoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
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

    let bytes = fixture();
    let mut session = CudaSession::default();
    assert!(!session.is_runtime_initialized());

    let mut first = J2kDecoder::new(&bytes).expect("decoder");
    let first_surface = <J2kDecoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
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

    let mut second = J2kDecoder::new(&bytes).expect("decoder");
    let second_surface = <J2kDecoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
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
fn auto_classic_full_frame_surface_matches_host_decode() {
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), signinum_core::BackendKind::Cpu);

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 12];
    host_decoder
        .decode_into(&mut host, 6, PixelFormat::Rgb8)
        .expect("host decode");
    assert_eq!(surface.as_host_bytes(), Some(host.as_slice()));
}

#[test]
fn auto_htj2k_full_frame_surface_matches_host_decode() {
    let bytes = fixture_ht_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), signinum_core::BackendKind::Cpu);

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 16];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray8)
        .expect("host decode");
    assert_eq!(surface.as_host_bytes(), Some(host.as_slice()));
}

#[test]
fn auto_region_scaled_surface_matches_host_decode() {
    let bytes = fixture_ht_gray8();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_region_scaled_to_device(PixelFormat::Gray8, roi, scale, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), signinum_core::BackendKind::Cpu);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = vec![0u8; scaled.w as usize * scaled.h as usize];
    host_decoder
        .decode_region_scaled_into(
            &mut signinum_j2k_cuda::J2kScratchPool::new(),
            &mut host,
            scaled.w as usize,
            PixelFormat::Gray8,
            roi,
            scale,
        )
        .expect("host decode");
    assert_eq!(surface.as_host_bytes(), Some(host.as_slice()));
}

#[test]
fn tile_batch_region_scaled_cuda_surface_matches_host_when_runtime_required() {
    if !runtime_required() {
        return;
    }

    let bytes = fixture_ht_gray8();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let mut ctx = DecoderContext::<signinum_j2k_cuda::J2kContext>::new();
    let mut pool = signinum_j2k_cuda::J2kScratchPool::new();
    let surface = Codec::decode_tile_region_scaled_to_device(
        &mut ctx,
        &mut pool,
        &bytes,
        PixelFormat::Gray8,
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

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut expected = vec![0u8; scaled.w as usize * scaled.h as usize];
    host_decoder
        .decode_region_scaled_into(
            &mut signinum_j2k_cuda::J2kScratchPool::new(),
            &mut expected,
            scaled.w as usize,
            PixelFormat::Gray8,
            roi,
            scale,
        )
        .expect("host decode");
    assert_eq!(downloaded, expected);
}

#[test]
fn decode_tiles_to_device_auto_preserves_order_and_matches_host_bytes() {
    let bytes = fixture_ht_gray8();
    let mut ctx = DecoderContext::<signinum_j2k_cuda::J2kContext>::new();
    let mut pool = signinum_j2k_cuda::J2kScratchPool::new();
    let inputs = [bytes.as_slice(), bytes.as_slice()];

    let surfaces = Codec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Gray8,
        BackendRequest::Auto,
    )
    .expect("batch surfaces");

    assert_eq!(surfaces.len(), inputs.len());
    let mut expected = [0u8; 16];
    J2kDecoder::new(&bytes)
        .expect("host decoder")
        .decode_into(&mut expected, 4, PixelFormat::Gray8)
        .expect("host decode");
    for surface in surfaces {
        assert_eq!(surface.dimensions(), (4, 4));
        match surface.backend_kind() {
            signinum_core::BackendKind::Cpu => {
                assert_eq!(surface.as_host_bytes(), Some(expected.as_slice()));
            }
            signinum_core::BackendKind::Cuda => {
                let mut downloaded = vec![0u8; surface.byte_len()];
                surface
                    .download_into(&mut downloaded, surface.pitch_bytes())
                    .expect("download cuda surface");
                assert_eq!(downloaded, expected);
            }
            signinum_core::BackendKind::Metal => panic!("J2K CUDA batch returned Metal surface"),
        }
    }
}

#[test]
fn decode_tiles_to_device_explicit_cuda_returns_cuda_surfaces_or_clear_unavailable_error() {
    let bytes = fixture_ht_gray8();
    let mut ctx = DecoderContext::<signinum_j2k_cuda::J2kContext>::new();
    let mut pool = signinum_j2k_cuda::J2kScratchPool::new();
    let inputs = [bytes.as_slice(), bytes.as_slice()];

    match Codec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Gray8,
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
