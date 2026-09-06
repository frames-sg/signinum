use super::*;
use j2k_core::CodecError;
#[cfg(target_os = "macos")]
use j2k_core::{DeviceSurface, ImageDecode, ImageDecodeDevice};
#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBlitCommandEncoder, MTLCommandEncoder, MTLOrigin, MTLSize, MTLTexture};

#[cfg(target_os = "macos")]
fn patterned_index_byte(index: usize) -> u8 {
    u8::try_from(index % 256).expect("modulo 256 fits in u8")
}

#[cfg(target_os = "macos")]
pub(crate) fn rgb_to_rgba_opaque(rgb: &[u8]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(rgb.len() / 3 * 4);
    for pixel in rgb.chunks_exact(3) {
        rgba.extend_from_slice(pixel);
        rgba.push(u8::MAX);
    }
    rgba
}

#[cfg(target_os = "macos")]
pub(crate) fn download_rgba8_texture(
    session: &MetalBackendSession,
    texture: &ProtocolObject<dyn MTLTexture>,
    dimensions: (u32, u32),
) -> Vec<u8> {
    let row_bytes = dimensions.0 as usize * PixelFormat::Rgba8.bytes_per_pixel();
    let byte_len = row_bytes * dimensions.1 as usize;
    let buffer = j2k_metal_support::checked_shared_buffer(session.device(), byte_len)
        .expect("texture readback buffer");
    let queue =
        j2k_metal_support::checked_command_queue(session.device()).expect("texture readback queue");
    let command_buffer =
        j2k_metal_support::checked_command_buffer(&queue).expect("texture readback command buffer");
    let blit = j2k_metal_support::checked_blit_command_encoder(&command_buffer)
        .expect("texture readback blit encoder");
    // SAFETY: The source region matches the validated texture dimensions, the
    // destination allocation covers `byte_len`, and both resources remain
    // retained until the command buffer completes below.
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            texture,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize {
                width: dimensions.0 as usize,
                height: dimensions.1 as usize,
                depth: 1,
            },
            &buffer,
            0,
            row_bytes,
            byte_len,
        );
    }
    blit.endEncoding();
    j2k_metal_support::commit_and_wait(&command_buffer).expect("texture readback blit");

    crate::buffers::checked_buffer_slice::<u8>(&buffer, byte_len, "texture test readback")
        .expect("texture readback buffer must be CPU-visible and bounded")
}

#[cfg(target_os = "macos")]
fn should_run_metal_runtime() -> bool {
    j2k_test_support::metal_runtime_gate(module_path!())
}

#[cfg(target_os = "macos")]
mod reusable_output;
#[cfg(target_os = "macos")]
mod textures;

#[cfg(target_os = "macos")]
fn decode_rgb8_buffer_batch_with_session(
    source: Rgb8MetalBatchSource<'_, '_>,
    op: Rgb8MetalBatchOp,
    target: MetalBufferBatchTarget<'_>,
    session: &MetalBackendSession,
) -> Result<Vec<Result<Surface, Error>>, Error> {
    Codec::decode_rgb8_batch_into_buffer_with_session(
        Rgb8MetalBatchRequest { source, op },
        target,
        session,
    )
}

#[cfg(target_os = "macos")]
fn decode_rgb8_texture_batch_with_session(
    source: Rgb8MetalBatchSource<'_, '_>,
    op: Rgb8MetalBatchOp,
    target: MetalTextureBatchTarget<'_>,
    session: &MetalBackendSession,
) -> Result<Vec<Result<MetalTextureTile, Error>>, Error> {
    Codec::decode_rgb8_batch_into_textures_with_session(
        Rgb8MetalBatchRequest { source, op },
        target,
        session,
    )
}

#[cfg(target_os = "macos")]
fn assert_reusable_rgba_texture_tiles(
    session: &MetalBackendSession,
    output: &MetalBatchTextureOutput,
    tiles: Vec<Result<MetalTextureTile, Error>>,
    dimensions: (u32, u32),
    expected_tiles: &[&[u8]],
) {
    assert_eq!(tiles.len(), expected_tiles.len());
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), dimensions);
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        let actual_rgba =
            download_rgba8_texture(session, tile.texture_trusted(), tile.dimensions());
        assert_eq!(actual_rgba.as_slice(), expected_tiles[index]);
    }
}

use j2k_jpeg::adapter::{build_fast420_packet, build_fast444_packet};
#[cfg(target_os = "macos")]
use j2k_jpeg::adapter::{build_fast422_packet, build_gray_packet, JpegHuffmanTable};
#[cfg(target_os = "macos")]
use j2k_jpeg::{
    encode_jpeg_baseline, DecodeRequest, JpegBackend, JpegEncodeOptions, JpegSamples,
    JpegSubsampling,
};

const BASELINE_420: &[u8] = include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg");
const BASELINE_420_RESTART: &[u8] =
    include_bytes!("../fixtures/jpeg/baseline_420_restart_32x16.jpg");
#[cfg(target_os = "macos")]
const BASELINE_422: &[u8] = include_bytes!("../fixtures/jpeg/baseline_422_16x8.jpg");
const BASELINE_444: &[u8] = include_bytes!("../fixtures/jpeg/baseline_444_8x8.jpg");
#[cfg(target_os = "macos")]
const GRAYSCALE: &[u8] = include_bytes!("../fixtures/jpeg/grayscale_8x8.jpg");

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
            state: "JPEG Metal session",
        },
    ] {
        assert!(!err.is_unsupported(), "{err:?}");
    }
}

#[test]
fn auto_route_prefers_cpu_host_for_nonrestart_packets() {
    let decoder_420 = CpuDecoder::new(BASELINE_420).expect("420 decoder");
    let packet_420 = build_fast420_packet(BASELINE_420).expect("420 packet");
    assert_eq!(
        choose_route(
            &decoder_420,
            BackendRequest::Auto,
            PixelFormat::Rgb8,
            batch::BatchOp::Full,
            JpegFastPackets::new(None, None, Some(&packet_420)),
        ),
        routing::RouteDecision::CpuHost
    );

    let decoder_444 = CpuDecoder::new(BASELINE_444).expect("444 decoder");
    let packet_444 = build_fast444_packet(BASELINE_444).expect("444 packet");
    assert_eq!(
        choose_route(
            &decoder_444,
            BackendRequest::Auto,
            PixelFormat::Rgb8,
            batch::BatchOp::Scaled(Downscale::Quarter),
            JpegFastPackets::new(Some(&packet_444), None, None),
        ),
        routing::RouteDecision::CpuHost
    );
}

#[test]
fn auto_route_keeps_small_single_restart_packets_on_cpu_host() {
    let decoder = CpuDecoder::new(BASELINE_420_RESTART).expect("restart decoder");
    let packet = build_fast420_packet(BASELINE_420_RESTART).expect("restart packet");

    assert_eq!(
        choose_route(
            &decoder,
            BackendRequest::Auto,
            PixelFormat::Rgb8,
            batch::BatchOp::Full,
            JpegFastPackets::new(None, None, Some(&packet))
        ),
        routing::RouteDecision::CpuHost
    );
    assert_eq!(
        choose_route(
            &decoder,
            BackendRequest::Auto,
            PixelFormat::Rgb8,
            batch::BatchOp::Region(Rect {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
            }),
            JpegFastPackets::new(None, None, Some(&packet)),
        ),
        routing::RouteDecision::CpuHost
    );
}

#[cfg(target_os = "macos")]
#[test]
fn prepared_huffman_host_matches_shared_canonical_derivation_for_fixture_packets() {
    let packet_420 = build_fast420_packet(BASELINE_420).expect("420 packet");
    assert_color_packet_huffman_matches_shared(
        "420",
        [
            ("y dc", &packet_420.y_dc_table),
            ("y ac", &packet_420.y_ac_table),
            ("cb dc", &packet_420.cb_dc_table),
            ("cb ac", &packet_420.cb_ac_table),
            ("cr dc", &packet_420.cr_dc_table),
            ("cr ac", &packet_420.cr_ac_table),
        ],
    );

    let packet_420_restart =
        build_fast420_packet(BASELINE_420_RESTART).expect("420 restart packet");
    assert_color_packet_huffman_matches_shared(
        "420 restart",
        [
            ("y dc", &packet_420_restart.y_dc_table),
            ("y ac", &packet_420_restart.y_ac_table),
            ("cb dc", &packet_420_restart.cb_dc_table),
            ("cb ac", &packet_420_restart.cb_ac_table),
            ("cr dc", &packet_420_restart.cr_dc_table),
            ("cr ac", &packet_420_restart.cr_ac_table),
        ],
    );

    let packet_422 = build_fast422_packet(BASELINE_422).expect("422 packet");
    assert_color_packet_huffman_matches_shared(
        "422",
        [
            ("y dc", &packet_422.y_dc_table),
            ("y ac", &packet_422.y_ac_table),
            ("cb dc", &packet_422.cb_dc_table),
            ("cb ac", &packet_422.cb_ac_table),
            ("cr dc", &packet_422.cr_dc_table),
            ("cr ac", &packet_422.cr_ac_table),
        ],
    );

    let packet_444 = build_fast444_packet(BASELINE_444).expect("444 packet");
    assert_color_packet_huffman_matches_shared(
        "444",
        [
            ("y dc", &packet_444.y_dc_table),
            ("y ac", &packet_444.y_ac_table),
            ("cb dc", &packet_444.cb_dc_table),
            ("cb ac", &packet_444.cb_ac_table),
            ("cr dc", &packet_444.cr_dc_table),
            ("cr ac", &packet_444.cr_ac_table),
        ],
    );

    let packet_gray = build_gray_packet(GRAYSCALE).expect("gray packet");
    assert_prepared_huffman_matches_shared("gray y dc", &packet_gray.y_dc_table);
    assert_prepared_huffman_matches_shared("gray y ac", &packet_gray.y_ac_table);
}

#[cfg(target_os = "macos")]
fn assert_color_packet_huffman_matches_shared(label: &str, tables: [(&str, &JpegHuffmanTable); 6]) {
    for (table_label, table) in tables {
        assert_prepared_huffman_matches_shared(&format!("{label} {table_label}"), table);
    }
}

#[cfg(target_os = "macos")]
fn assert_prepared_huffman_matches_shared(label: &str, table: &JpegHuffmanTable) {
    let canonical = table
        .derive_canonical()
        .unwrap_or_else(|error| panic!("{label}: shared canonical derivation failed: {error}"));
    let prepared = crate::abi::PreparedHuffmanHost::from(table);
    let values_len = usize::from(table.values_len);

    assert_eq!(prepared.min_code, canonical.min_code, "{label} min_code");
    assert_eq!(prepared.max_code, canonical.max_code, "{label} max_code");
    assert_eq!(
        prepared.val_offset, canonical.val_offset,
        "{label} val_offset"
    );
    assert_eq!(prepared.values_len, table.values_len, "{label} values_len");
    assert_eq!(
        &prepared.values[..values_len],
        &table.values[..values_len],
        "{label} values"
    );

    let mut fast_symbol = [0u8; 512];
    let mut fast_len = [0u8; 512];
    for idx in 0..canonical.huffsize_len {
        let len = usize::from(canonical.huffsize[idx]);
        if len == 0 || len > 9 {
            continue;
        }
        let code = usize::from(canonical.huffcode[idx]);
        let prefix = code << (9 - len);
        let fill = 1usize << (9 - len);
        for suffix in 0..fill {
            fast_symbol[prefix | suffix] = table.values[idx];
            fast_len[prefix | suffix] = canonical.huffsize[idx];
        }
    }

    assert_eq!(prepared.fast_symbol, fast_symbol, "{label} fast_symbol");
    assert_eq!(prepared.fast_len, fast_len, "{label} fast_len");
}

#[cfg(target_os = "macos")]
#[test]
fn metal_backend_session_reuses_compiled_runtime() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    assert!(!session.runtime_initialized_for_test());

    let mut first = Decoder::new(BASELINE_420).expect("first decoder");
    let first_surface = first
        .decode_to_device_with_session(PixelFormat::Rgb8, &session)
        .expect("first session decode");
    assert_eq!(
        first_surface.residency(),
        SurfaceResidency::MetalResidentDecode
    );
    let first_runtime = session
        .runtime_ptr_for_test()
        .expect("session runtime after first decode");

    let mut second = Decoder::new(BASELINE_420).expect("second decoder");
    second
        .decode_to_device_with_session(PixelFormat::Rgb8, &session)
        .expect("second session decode");
    let second_runtime = session
        .runtime_ptr_for_test()
        .expect("session runtime after second decode");

    assert_eq!(first_runtime, second_runtime);
}

#[cfg(target_os = "macos")]
#[test]
fn jpeg_rgb8_batch_decode_uses_backend_session_runtime() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    assert!(!session.runtime_initialized_for_test());

    let inputs = [BASELINE_420, BASELINE_420];
    let results = decode_rgb8_batch_to_device_with_session(&inputs, &session)
        .expect("session batch decode")
        .expect("baseline JPEG batch should use Metal batch path");

    assert_eq!(results.len(), 2);
    assert!(session.runtime_initialized_for_test());
    for result in results {
        let surface = result.expect("surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (16, 16));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn queued_jpeg_batch_decode_uses_metal_session_runtime() {
    use j2k_core::DeviceSubmission as _;

    if !should_run_metal_runtime() {
        return;
    }

    let backend_session = MetalBackendSession::system_default().expect("Metal backend session");
    assert!(!backend_session.runtime_initialized_for_test());
    let mut session = MetalSession::with_backend_session(backend_session.clone());
    let mut ctx = j2k_jpeg::DecoderContext::default();
    let mut pool = ScratchPool::new();

    let submissions = (0..2)
        .map(|_| {
            <Codec as j2k_core::TileBatchDecodeSubmit>::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                BASELINE_420,
                PixelFormat::Rgb8,
                BackendRequest::Metal,
            )
            .expect("queued Metal tile submit")
        })
        .collect::<Vec<_>>();

    for submission in submissions {
        let surface = submission.wait().expect("queued Metal surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (16, 16));
    }

    assert_eq!(session.submissions().expect("session submissions"), 1);
    assert!(
        backend_session.runtime_initialized_for_test(),
        "queued MetalSession batch decode should reuse its backend runtime"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn default_queued_jpeg_batch_decode_lazily_initializes_backend_session() {
    use j2k_core::DeviceSubmission as _;

    if !should_run_metal_runtime() {
        return;
    }

    let mut session = MetalSession::default();
    assert!(session
        .shared
        .0
        .lock()
        .expect("metal session")
        .backend_session
        .is_none());
    let mut ctx = j2k_jpeg::DecoderContext::default();
    let mut pool = ScratchPool::new();

    let submissions = (0..2)
        .map(|_| {
            <Codec as j2k_core::TileBatchDecodeSubmit>::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                BASELINE_420,
                PixelFormat::Rgb8,
                BackendRequest::Metal,
            )
            .expect("queued Metal tile submit")
        })
        .collect::<Vec<_>>();

    for submission in submissions {
        let surface = submission.wait().expect("queued Metal surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
    }

    let runtime_initialized = session
        .shared
        .0
        .lock()
        .expect("metal session")
        .backend_session
        .as_ref()
        .is_some_and(MetalBackendSession::runtime_initialized_for_test);
    assert!(runtime_initialized);
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_batch_decode_can_write_into_reusable_metal_output_buffer() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (16, 16), 2).expect("output buffer");
    let inputs = [BASELINE_420, BASELINE_420];
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("cpu decode");

    let surfaces = decode_rgb8_buffer_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalBufferBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode into reusable output");

    assert_eq!(surfaces.len(), 2);
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(
        output.tile_stride_bytes(),
        16 * 16 * PixelFormat::Rgb8.bytes_per_pixel()
    );
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (16, 16));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer_trusted().expect("metal buffer");
        assert!(std::ptr::eq(buffer, output.buffer_trusted()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            expected.as_slice()
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_resizes_reusable_metal_output_buffer() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("cpu decode");

    let surfaces = decode_rgb8_buffer_batch_with_session(
        Rgb8MetalBatchSource::Decoders(&decoders),
        Rgb8MetalBatchOp::Full,
        MetalBufferBatchTarget::Resizable(&mut output),
        &session,
    )
    .expect("decode cached decoder batch into resizable reusable output");

    assert_eq!(output.dimensions(), (16, 16));
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(surfaces.len(), 2);
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (16, 16));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer_trusted().expect("metal buffer");
        assert!(std::ptr::eq(buffer, output.buffer_trusted()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            expected.as_slice()
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_can_write_into_fixed_metal_output_buffer() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (16, 16), 2).expect("output buffer");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("cpu decode");

    let surfaces = decode_rgb8_buffer_batch_with_session(
        Rgb8MetalBatchSource::Decoders(&decoders),
        Rgb8MetalBatchOp::Full,
        MetalBufferBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode cached decoder batch into fixed reusable output");

    assert_eq!(surfaces.len(), 2);
    assert_eq!(output.dimensions(), (16, 16));
    assert_eq!(output.tile_capacity(), 2);
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (16, 16));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer_trusted().expect("metal buffer");
        assert!(std::ptr::eq(buffer, output.buffer_trusted()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            expected.as_slice()
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_rejects_mixed_output_dimensions_without_resizing_buffer() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_444).expect("second decoder");
    let decoders = [&first, &second];

    let Err(err) = decode_rgb8_buffer_batch_with_session(
        Rgb8MetalBatchSource::Decoders(&decoders),
        Rgb8MetalBatchOp::Full,
        MetalBufferBatchTarget::Resizable(&mut output),
        &session,
    ) else {
        panic!("mixed output dimensions should be rejected");
    };

    assert!(matches!(err, Error::UnsupportedMetalRequest { .. }));
    assert_eq!(output.dimensions(), (1, 1));
    assert_eq!(output.tile_capacity(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_rejects_mixed_sampling_without_resizing_buffer() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let rgb = j2k_test_support::patterned_rgb8(16, 16);
    let fast420 = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: 16,
            height: 16,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode fast420 jpeg");
    let fast444 = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: 16,
            height: 16,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode fast444 jpeg");
    let first = Decoder::new(&fast420.data).expect("first decoder");
    let second = Decoder::new(&fast444.data).expect("second decoder");
    let decoders = [&first, &second];

    let Err(err) = decode_rgb8_buffer_batch_with_session(
        Rgb8MetalBatchSource::Decoders(&decoders),
        Rgb8MetalBatchOp::Full,
        MetalBufferBatchTarget::Resizable(&mut output),
        &session,
    ) else {
        panic!("mixed sampling should be rejected");
    };

    assert!(matches!(
        err,
        Error::UnsupportedMetalRequest { reason }
            if reason.contains("same fast-packet sampling family")
    ));
    assert_eq!(output.dimensions(), (1, 1));
    assert_eq!(output.tile_capacity(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_metal_report_exposes_required_output_shape() {
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];

    let full =
        Codec::inspect_rgb8_decoder_batch_metal_output(&decoders, j2k_jpeg::JpegDecodeOp::Full);
    let scaled = Codec::inspect_rgb8_decoder_batch_metal_output(
        &decoders,
        j2k_jpeg::JpegDecodeOp::Scaled(Downscale::Quarter),
    );
    let roi = Rect {
        x: 1,
        y: 2,
        w: 10,
        h: 9,
    };
    let region_scaled = Codec::inspect_rgb8_decoder_batch_metal_output(
        &decoders,
        j2k_jpeg::JpegDecodeOp::RegionScaled {
            roi: j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale: Downscale::Quarter,
        },
    );

    assert!(full.eligibility.eligible);
    assert_eq!(full.tile_count, 2);
    assert_eq!(full.output_dimensions, Some((16, 16)));
    assert_eq!(full.required_tile_capacity(), 2);

    assert!(scaled.eligibility.eligible);
    assert_eq!(scaled.output_dimensions, Some((4, 4)));

    assert!(region_scaled.eligibility.eligible);
    let expected = roi.scaled_covering(Downscale::Quarter);
    assert_eq!(
        region_scaled.output_dimensions,
        Some((expected.w, expected.h))
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_metal_report_rejects_incompatible_batches_without_launch() {
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_444).expect("second decoder");
    let decoders = [&first, &second];

    let mixed =
        Codec::inspect_rgb8_decoder_batch_metal_output(&decoders, j2k_jpeg::JpegDecodeOp::Full);
    let region = Codec::inspect_rgb8_decoder_batch_metal_output(
        &[&first],
        j2k_jpeg::JpegDecodeOp::Region(j2k_jpeg::Rect {
            x: 0,
            y: 0,
            w: 8,
            h: 8,
        }),
    );

    assert!(!mixed.eligibility.eligible);
    assert_eq!(mixed.output_dimensions, None);
    assert!(mixed
        .eligibility
        .reason
        .expect("mixed rejection")
        .contains("matching output dimensions"));

    assert!(!region.eligibility.eligible);
    assert!(region
        .eligibility
        .reason
        .expect("region rejection")
        .contains("full, scaled, or region-scaled"));
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_metal_report_rejects_mixed_sampling_family() {
    let rgb = j2k_test_support::patterned_rgb8(16, 16);
    let fast420 = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: 16,
            height: 16,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode fast420 jpeg");
    let fast444 = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: 16,
            height: 16,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode fast444 jpeg");
    let first = Decoder::new(&fast420.data).expect("first decoder");
    let second = Decoder::new(&fast444.data).expect("second decoder");
    let decoders = [&first, &second];

    let report =
        Codec::inspect_rgb8_decoder_batch_metal_output(&decoders, j2k_jpeg::JpegDecodeOp::Full);

    assert!(!report.eligibility.eligible);
    assert_eq!(report.output_dimensions, None);
    assert!(report
        .eligibility
        .reason
        .expect("mixed sampling rejection")
        .contains("same fast-packet sampling family"));
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_metal_report_rejects_restart_fast422_full_tiles() {
    let rgb = j2k_test_support::patterned_rgb8(64, 32);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: 64,
            height: 32,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: Some(4),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode restart fast422 jpeg");
    let packet = build_fast422_packet(&jpeg.data).expect("restart fast422 packet");
    assert_ne!(packet.restart_interval_mcus, 0);
    let first = Decoder::new(&jpeg.data).expect("first decoder");
    let second = Decoder::new(&jpeg.data).expect("second decoder");
    let decoders = [&first, &second];

    let full =
        Codec::inspect_rgb8_decoder_batch_metal_output(&decoders, j2k_jpeg::JpegDecodeOp::Full);
    let scaled = Codec::inspect_rgb8_decoder_batch_metal_output(
        &decoders,
        j2k_jpeg::JpegDecodeOp::Scaled(Downscale::Half),
    );

    assert!(!full.eligibility.eligible);
    assert_eq!(full.output_dimensions, None);
    assert!(full
        .eligibility
        .reason
        .expect("restart fast422 full rejection")
        .contains("restart-coded full-tile 4:2:2 or 4:4:4"));

    assert!(scaled.eligibility.eligible);
    assert_eq!(scaled.output_dimensions, Some((32, 16)));
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_fast444_batch_decode_can_write_into_reusable_metal_output_buffer() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (8, 8), 2).expect("output buffer");
    let inputs = [BASELINE_444, BASELINE_444];
    let (expected, _) = CpuDecoder::new(BASELINE_444)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("cpu decode");

    let surfaces = decode_rgb8_buffer_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalBufferBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode into reusable output");

    assert_eq!(surfaces.len(), 2);
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (8, 8));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer_trusted().expect("metal buffer");
        assert!(std::ptr::eq(buffer, output.buffer_trusted()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            expected.as_slice()
        );
    }
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "this end-to-end helper keeps mixed-table grouping, resident-buffer identity, dispatch counts, and CPU parity in one assertion"
)]
fn assert_table_mixed_full_buffer_groups_resident(
    subsampling: JpegSubsampling,
    dimensions: (u32, u32),
    first_quality: u8,
    second_quality: u8,
) {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let rgb_a = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_b = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_c = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    for (index, pixel) in rgb_b.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(43)
            .wrapping_add(17);
        pixel[0] ^= delta.rotate_left(1);
        pixel[1] = pixel[1].wrapping_sub(delta);
        pixel[2] = pixel[2].wrapping_add(delta.rotate_right(2));
    }
    for (index, pixel) in rgb_c.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(47)
            .wrapping_add(23);
        pixel[0] = pixel[0].wrapping_add(delta.rotate_left(2));
        pixel[1] ^= delta.rotate_right(1);
        pixel[2] = pixel[2].wrapping_sub(delta);
    }

    let jpeg_a = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_a,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: first_quality,
            subsampling,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode first table-mixed full buffer jpeg");
    let jpeg_b = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_b,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: second_quality,
            subsampling,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode second table-mixed full buffer jpeg");
    let jpeg_c = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_c,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: first_quality,
            subsampling,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode third table-mixed full buffer jpeg");

    match subsampling {
        JpegSubsampling::Ybr420 => {
            let packet_a = build_fast420_packet(&jpeg_a.data).expect("first packet");
            let packet_b = build_fast420_packet(&jpeg_b.data).expect("second packet");
            let packet_c = build_fast420_packet(&jpeg_c.data).expect("third packet");
            assert_eq!(packet_a.y_quant, packet_c.y_quant);
            assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
            assert_eq!(
                packet_a.entropy_checkpoints.len(),
                packet_c.entropy_checkpoints.len()
            );
            assert_ne!(packet_a.y_quant, packet_b.y_quant);
        }
        JpegSubsampling::Ybr422 => {
            let packet_a = build_fast422_packet(&jpeg_a.data).expect("first packet");
            let packet_b = build_fast422_packet(&jpeg_b.data).expect("second packet");
            let packet_c = build_fast422_packet(&jpeg_c.data).expect("third packet");
            assert_eq!(packet_a.y_quant, packet_c.y_quant);
            assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
            assert_eq!(
                packet_a.entropy_checkpoints.len(),
                packet_c.entropy_checkpoints.len()
            );
            assert_ne!(packet_a.y_quant, packet_b.y_quant);
        }
        JpegSubsampling::Ybr444 => {
            let packet_a = build_fast444_packet(&jpeg_a.data).expect("first packet");
            let packet_b = build_fast444_packet(&jpeg_b.data).expect("second packet");
            let packet_c = build_fast444_packet(&jpeg_c.data).expect("third packet");
            assert_eq!(packet_a.y_quant, packet_c.y_quant);
            assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
            assert_eq!(
                packet_a.entropy_checkpoints.len(),
                packet_c.entropy_checkpoints.len()
            );
            assert_ne!(packet_a.y_quant, packet_b.y_quant);
        }
        JpegSubsampling::Gray => panic!("table-mixed buffer helper expects YCbCr sampling"),
    }

    let output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, dimensions, 3).expect("output buffer");
    let inputs = [
        jpeg_a.data.as_slice(),
        jpeg_b.data.as_slice(),
        jpeg_c.data.as_slice(),
    ];
    let expected_tiles = inputs
        .iter()
        .map(|input| {
            CpuDecoder::new(input)
                .expect("cpu decoder")
                .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
                .expect("cpu decode")
                .0
        })
        .collect::<Vec<_>>();
    assert_ne!(expected_tiles[0], expected_tiles[1]);
    assert_ne!(expected_tiles[0], expected_tiles[2]);
    assert_ne!(expected_tiles[1], expected_tiles[2]);

    let surfaces = decode_rgb8_buffer_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalBufferBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode table-mixed full tiles into reusable output buffer");

    assert_eq!(surfaces.len(), 3);
    for (index, surface) in surfaces.into_iter().enumerate() {
        let surface = surface.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), dimensions);
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer_trusted().expect("metal buffer");
        assert!(std::ptr::eq(buffer, output.buffer_trusted()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            expected_tiles[index].as_slice()
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_table_mixed_fast420_buffer_batch_groups_resident_dispatches() {
    assert_table_mixed_full_buffer_groups_resident(JpegSubsampling::Ybr420, (128, 96), 90, 72);
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_table_mixed_fast422_buffer_batch_groups_resident_dispatches() {
    assert_table_mixed_full_buffer_groups_resident(JpegSubsampling::Ybr422, (128, 96), 91, 73);
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_table_mixed_fast444_buffer_batch_groups_resident_dispatches() {
    assert_table_mixed_full_buffer_groups_resident(JpegSubsampling::Ybr444, (96, 96), 92, 74);
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_scaled_batch_decode_can_write_into_reusable_metal_output_buffer() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let scale = Downscale::Quarter;
    let output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (4, 4), 2).expect("output buffer");
    let inputs = [BASELINE_420, BASELINE_420];
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::scaled(PixelFormat::Rgb8, scale))
        .expect("cpu scaled decode");

    let surfaces = decode_rgb8_buffer_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Scaled(scale),
        MetalBufferBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode scaled into reusable output");

    assert_eq!(surfaces.len(), 2);
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (4, 4));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer_trusted().expect("metal buffer");
        assert!(std::ptr::eq(buffer, output.buffer_trusted()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            expected.as_slice()
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_scaled_batch_resizes_reusable_metal_output_buffer() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let scale = Downscale::Quarter;
    let mut output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::scaled(PixelFormat::Rgb8, scale))
        .expect("cpu scaled decode");

    let surfaces = decode_rgb8_buffer_batch_with_session(
        Rgb8MetalBatchSource::Decoders(&decoders),
        Rgb8MetalBatchOp::Scaled(scale),
        MetalBufferBatchTarget::Resizable(&mut output),
        &session,
    )
    .expect("decode cached decoder scaled batch into resizable reusable output");

    assert_eq!(output.dimensions(), (4, 4));
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(surfaces.len(), 2);
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (4, 4));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer_trusted().expect("metal buffer");
        assert!(std::ptr::eq(buffer, output.buffer_trusted()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            expected.as_slice()
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_scaled_batch_can_write_into_fixed_metal_output_buffer() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let scale = Downscale::Quarter;
    let output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (4, 4), 2).expect("output buffer");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::scaled(PixelFormat::Rgb8, scale))
        .expect("cpu scaled decode");

    let surfaces = decode_rgb8_buffer_batch_with_session(
        Rgb8MetalBatchSource::Decoders(&decoders),
        Rgb8MetalBatchOp::Scaled(scale),
        MetalBufferBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode cached decoder scaled batch into fixed reusable output");

    assert_eq!(surfaces.len(), 2);
    assert_eq!(output.dimensions(), (4, 4));
    assert_eq!(output.tile_capacity(), 2);
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (4, 4));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer_trusted().expect("metal buffer");
        assert!(std::ptr::eq(buffer, output.buffer_trusted()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            expected.as_slice()
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_region_scaled_batch_decode_can_write_into_reusable_metal_output_buffer() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 5,
        h: 4,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let output = MetalBatchOutputBuffer::new_rgb8_tiles(&session, (scaled.w, scaled.h), 2)
        .expect("output buffer");
    let inputs = [BASELINE_444, BASELINE_444];
    let (expected, _) = CpuDecoder::new(BASELINE_444)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("cpu region scaled decode");

    let surfaces = decode_rgb8_buffer_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::RegionScaled { roi, scale },
        MetalBufferBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode region scaled into reusable output");

    assert_eq!(surfaces.len(), 2);
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer_trusted().expect("metal buffer");
        assert!(std::ptr::eq(buffer, output.buffer_trusted()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            expected.as_slice()
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_region_scaled_batch_decode_resizes_reusable_metal_output_buffer() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 5,
        h: 4,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let mut output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let inputs = [BASELINE_444, BASELINE_444];
    let (expected, _) = CpuDecoder::new(BASELINE_444)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("cpu region scaled decode");

    let surfaces = Codec::decode_rgb8_region_scaled_batch_into_resizable_metal_buffer_with_session(
        &inputs,
        roi,
        scale,
        &mut output,
        &session,
    )
    .expect("decode region scaled into resizable reusable output");

    assert_eq!(output.dimensions(), (scaled.w, scaled.h));
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(surfaces.len(), 2);
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer_trusted().expect("metal buffer");
        assert!(std::ptr::eq(buffer, output.buffer_trusted()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            expected.as_slice()
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_region_scaled_batch_resizes_reusable_metal_output_buffer() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 10,
        h: 9,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let mut output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("cpu region scaled decode");

    let surfaces = decode_rgb8_buffer_batch_with_session(
        Rgb8MetalBatchSource::Decoders(&decoders),
        Rgb8MetalBatchOp::RegionScaled { roi, scale },
        MetalBufferBatchTarget::Resizable(&mut output),
        &session,
    )
    .expect("decode cached decoder batch into resizable reusable output");

    assert_eq!(output.dimensions(), (scaled.w, scaled.h));
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(surfaces.len(), 2);
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer_trusted().expect("metal buffer");
        assert!(std::ptr::eq(buffer, output.buffer_trusted()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            expected.as_slice()
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_region_scaled_batch_can_write_into_fixed_metal_output_buffer() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 10,
        h: 9,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let output = MetalBatchOutputBuffer::new_rgb8_tiles(&session, (scaled.w, scaled.h), 2)
        .expect("output buffer");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("cpu region scaled decode");

    let surfaces = decode_rgb8_buffer_batch_with_session(
        Rgb8MetalBatchSource::Decoders(&decoders),
        Rgb8MetalBatchOp::RegionScaled { roi, scale },
        MetalBufferBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode cached decoder region-scaled batch into fixed reusable output");

    assert_eq!(surfaces.len(), 2);
    assert_eq!(output.dimensions(), (scaled.w, scaled.h));
    assert_eq!(output.tile_capacity(), 2);
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer_trusted().expect("metal buffer");
        assert!(std::ptr::eq(buffer, output.buffer_trusted()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            expected.as_slice()
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_restart_fast420_region_scaled_batch_decode_writes_reusable_metal_output_buffer() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let dimensions = (128, 128);
    let roi = Rect {
        x: 9,
        y: 11,
        w: 73,
        h: 67,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let rgb = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: Some(4),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode restart-coded fast420 region-scaled jpeg");
    let packet = build_fast420_packet(&jpeg.data).expect("restart fast420 packet");
    assert_ne!(packet.restart_interval_mcus, 0);
    assert!(!packet.restart_offsets.is_empty());

    let output = MetalBatchOutputBuffer::new_rgb8_tiles(&session, (scaled.w, scaled.h), 2)
        .expect("output buffer");
    let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
    let (expected, _) = CpuDecoder::new(&jpeg.data)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("cpu region-scaled decode");

    let surfaces = decode_rgb8_buffer_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::RegionScaled { roi, scale },
        MetalBufferBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode restart-coded region-scaled tiles into reusable output buffer");

    assert_eq!(surfaces.len(), 2);
    for (index, surface) in surfaces.into_iter().enumerate() {
        let surface = surface.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer_trusted().expect("metal buffer");
        assert!(std::ptr::eq(buffer, output.buffer_trusted()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            expected.as_slice()
        );
    }
}

#[cfg(target_os = "macos")]
fn assert_restart_region_scaled_buffer_batch_writes_reusable_metal_output(
    subsampling: JpegSubsampling,
    dimensions: (u32, u32),
) {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 0,
        y: 0,
        w: dimensions.0,
        h: dimensions.1,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let rgb = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling,
            restart_interval: Some(256),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode restart-coded region-scaled jpeg");
    match subsampling {
        JpegSubsampling::Ybr422 => {
            let packet = build_fast422_packet(&jpeg.data).expect("restart fast422 packet");
            assert_ne!(packet.restart_interval_mcus, 0);
            assert!(!packet.restart_offsets.is_empty());
        }
        JpegSubsampling::Ybr444 => {
            let packet = build_fast444_packet(&jpeg.data).expect("restart fast444 packet");
            assert_ne!(packet.restart_interval_mcus, 0);
            assert!(!packet.restart_offsets.is_empty());
        }
        _ => panic!("restart region-scaled buffer helper expects fast422 or fast444"),
    }

    let output = MetalBatchOutputBuffer::new_rgb8_tiles(&session, (scaled.w, scaled.h), 2)
        .expect("output buffer");
    let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
    let (expected, _) = CpuDecoder::new(&jpeg.data)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("cpu region-scaled decode");

    let surfaces = decode_rgb8_buffer_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::RegionScaled { roi, scale },
        MetalBufferBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode restart-coded region-scaled tiles into reusable output buffer");

    assert_eq!(surfaces.len(), 2);
    for (index, surface) in surfaces.into_iter().enumerate() {
        let surface = surface.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer_trusted().expect("metal buffer");
        assert!(std::ptr::eq(buffer, output.buffer_trusted()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            expected.as_slice()
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_restart_fast422_region_scaled_batch_decode_writes_reusable_metal_output_buffer() {
    assert_restart_region_scaled_buffer_batch_writes_reusable_metal_output(
        JpegSubsampling::Ybr422,
        (128, 96),
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_restart_fast444_region_scaled_batch_decode_writes_reusable_metal_output_buffer() {
    assert_restart_region_scaled_buffer_batch_writes_reusable_metal_output(
        JpegSubsampling::Ybr444,
        (96, 96),
    );
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "this end-to-end helper keeps region scaling, mixed-table grouping, resident-buffer identity, dispatch counts, and CPU parity in one assertion"
)]
fn assert_table_mixed_region_scaled_buffer_groups_resident(
    subsampling: JpegSubsampling,
    dimensions: (u32, u32),
    first_quality: u8,
    second_quality: u8,
) {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 0,
        y: 0,
        w: dimensions.0,
        h: dimensions.1,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let rgb_a = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_b = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_c = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    for (index, pixel) in rgb_b.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(37)
            .wrapping_add(19);
        pixel[0] = pixel[0].wrapping_add(delta.rotate_left(1));
        pixel[1] ^= delta;
        pixel[2] = pixel[2].wrapping_sub(delta.rotate_right(2));
    }
    for (index, pixel) in rgb_c.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(53)
            .wrapping_add(11);
        pixel[0] ^= delta.rotate_right(1);
        pixel[1] = pixel[1].wrapping_sub(delta.rotate_left(2));
        pixel[2] = pixel[2].wrapping_add(delta);
    }

    let jpeg_a = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_a,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: first_quality,
            subsampling,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode first table-mixed region-scaled buffer jpeg");
    let jpeg_b = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_b,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: second_quality,
            subsampling,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode second table-mixed region-scaled buffer jpeg");
    let jpeg_c = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_c,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: first_quality,
            subsampling,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode third table-mixed region-scaled buffer jpeg");

    match subsampling {
        JpegSubsampling::Ybr420 => {
            let packet_a = build_fast420_packet(&jpeg_a.data).expect("first packet");
            let packet_b = build_fast420_packet(&jpeg_b.data).expect("second packet");
            let packet_c = build_fast420_packet(&jpeg_c.data).expect("third packet");
            assert_eq!(packet_a.y_quant, packet_c.y_quant);
            assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
            assert_eq!(
                packet_a.entropy_checkpoints.len(),
                packet_c.entropy_checkpoints.len()
            );
            assert_ne!(packet_a.y_quant, packet_b.y_quant);
        }
        JpegSubsampling::Ybr422 => {
            let packet_a = build_fast422_packet(&jpeg_a.data).expect("first packet");
            let packet_b = build_fast422_packet(&jpeg_b.data).expect("second packet");
            let packet_c = build_fast422_packet(&jpeg_c.data).expect("third packet");
            assert_eq!(packet_a.y_quant, packet_c.y_quant);
            assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
            assert_eq!(
                packet_a.entropy_checkpoints.len(),
                packet_c.entropy_checkpoints.len()
            );
            assert_ne!(packet_a.y_quant, packet_b.y_quant);
        }
        JpegSubsampling::Ybr444 => {
            let packet_a = build_fast444_packet(&jpeg_a.data).expect("first packet");
            let packet_b = build_fast444_packet(&jpeg_b.data).expect("second packet");
            let packet_c = build_fast444_packet(&jpeg_c.data).expect("third packet");
            assert_eq!(packet_a.y_quant, packet_c.y_quant);
            assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
            assert_eq!(
                packet_a.entropy_checkpoints.len(),
                packet_c.entropy_checkpoints.len()
            );
            assert_ne!(packet_a.y_quant, packet_b.y_quant);
        }
        JpegSubsampling::Gray => panic!("table-mixed buffer helper expects YCbCr sampling"),
    }

    let output = MetalBatchOutputBuffer::new_rgb8_tiles(&session, (scaled.w, scaled.h), 3)
        .expect("output buffer");
    let inputs = [
        jpeg_a.data.as_slice(),
        jpeg_b.data.as_slice(),
        jpeg_c.data.as_slice(),
    ];
    let expected_tiles = inputs
        .iter()
        .map(|input| {
            CpuDecoder::new(input)
                .expect("cpu decoder")
                .decode_request(DecodeRequest::region_scaled(
                    PixelFormat::Rgb8,
                    j2k_jpeg::Rect {
                        x: roi.x,
                        y: roi.y,
                        w: roi.w,
                        h: roi.h,
                    },
                    scale,
                ))
                .expect("cpu region-scaled decode")
                .0
        })
        .collect::<Vec<_>>();
    assert_ne!(expected_tiles[0], expected_tiles[1]);
    assert_ne!(expected_tiles[0], expected_tiles[2]);
    assert_ne!(expected_tiles[1], expected_tiles[2]);

    let surfaces = decode_rgb8_buffer_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::RegionScaled { roi, scale },
        MetalBufferBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode table-mixed region-scaled tiles into reusable output buffer");

    assert_eq!(surfaces.len(), 3);
    for (index, surface) in surfaces.into_iter().enumerate() {
        let surface = surface.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer_trusted().expect("metal buffer");
        assert!(std::ptr::eq(buffer, output.buffer_trusted()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            expected_tiles[index].as_slice()
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_table_mixed_fast420_region_scaled_buffer_batch_groups_resident_dispatches() {
    assert_table_mixed_region_scaled_buffer_groups_resident(
        JpegSubsampling::Ybr420,
        (128, 96),
        90,
        72,
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_table_mixed_fast422_region_scaled_buffer_batch_groups_resident_dispatches() {
    assert_table_mixed_region_scaled_buffer_groups_resident(
        JpegSubsampling::Ybr422,
        (128, 96),
        91,
        73,
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_table_mixed_fast444_region_scaled_buffer_batch_groups_resident_dispatches() {
    assert_table_mixed_region_scaled_buffer_groups_resident(
        JpegSubsampling::Ybr444,
        (96, 96),
        92,
        74,
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_fast444_region_scaled_batch_decode_can_write_into_reusable_metal_textures() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 5,
        h: 4,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 2)
        .expect("texture output");
    let inputs = [BASELINE_444, BASELINE_444];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_444)
        .expect("cpu decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("cpu region scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_texture_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::RegionScaled { roi, scale },
        MetalTextureBatchTarget::Reusable(&output),
        &session,
    )
    .expect("decode region scaled into reusable textures");

    assert_eq!(tiles.len(), 2);
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(output.dimensions(), (scaled.w, scaled.h));
    assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture_trusted(),
            output.texture_trusted(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture_trusted(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_batch_output_buffer_ensure_region_scaled_tiles_uses_scaled_roi_shape() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let scaled = roi.scaled_covering(Downscale::Quarter);
    let mut output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");

    output
        .ensure_rgb8_region_scaled_tiles(&session, roi, Downscale::Quarter, 2)
        .expect("ensure region-scaled output");

    assert_eq!(output.dimensions(), (scaled.w, scaled.h));
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(
        output.tile_stride_bytes(),
        scaled.w as usize * scaled.h as usize * PixelFormat::Rgb8.bytes_per_pixel()
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_batch_texture_output_ensure_scaled_tiles_uses_scaled_full_shape() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");

    output
        .ensure_rgba8_scaled_tiles(&session, (16, 16), Downscale::Quarter, 2)
        .expect("ensure scaled texture output");

    assert_eq!(output.dimensions(), (4, 4));
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_batch_outputs_can_ensure_from_resident_batch_report() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let report = Codec::inspect_rgb8_decoder_batch_metal_output(
        &decoders,
        j2k_jpeg::JpegDecodeOp::Scaled(Downscale::Quarter),
    );
    assert!(report.eligibility.eligible);

    let mut buffer =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let mut textures =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");

    buffer
        .ensure_rgb8_batch_report(&session, &report)
        .expect("ensure buffer from report");
    textures
        .ensure_rgba8_batch_report(&session, &report)
        .expect("ensure textures from report");

    assert_eq!(buffer.dimensions(), (4, 4));
    assert_eq!(buffer.tile_capacity(), 2);
    assert_eq!(
        buffer.tile_stride_bytes(),
        4 * 4 * PixelFormat::Rgb8.bytes_per_pixel()
    );
    assert_eq!(textures.dimensions(), (4, 4));
    assert_eq!(textures.tile_capacity(), 2);
    assert_eq!(textures.pixel_format(), PixelFormat::Rgba8);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_batch_outputs_reject_ineligible_report_without_resizing() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_444).expect("second decoder");
    let decoders = [&first, &second];
    let report =
        Codec::inspect_rgb8_decoder_batch_metal_output(&decoders, j2k_jpeg::JpegDecodeOp::Full);
    assert!(!report.eligibility.eligible);

    let mut buffer =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let mut textures =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");

    let buffer_err = buffer
        .ensure_rgb8_batch_report(&session, &report)
        .expect_err("ineligible report should reject buffer ensure");
    let texture_err = textures
        .ensure_rgba8_batch_report(&session, &report)
        .expect_err("ineligible report should reject texture ensure");

    assert!(matches!(
        buffer_err,
        Error::UnsupportedMetalRequest { reason }
            if reason.contains("matching output dimensions")
    ));
    assert!(matches!(
        texture_err,
        Error::UnsupportedMetalRequest { reason }
            if reason.contains("matching output dimensions")
    ));
    assert_eq!(buffer.dimensions(), (1, 1));
    assert_eq!(buffer.tile_capacity(), 1);
    assert_eq!(textures.dimensions(), (1, 1));
    assert_eq!(textures.tile_capacity(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn warm_session_reuses_private_intermediate_buffers_for_reusable_output_batches() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (16, 16), 2).expect("output buffer");
    let inputs = [BASELINE_420, BASELINE_420];

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let first = decode_rgb8_buffer_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalBufferBatchTarget::Reusable(&output),
        &session,
    )
    .expect("first decode");
    for surface in first {
        assert_eq!(
            surface.expect("surface").residency(),
            SurfaceResidency::MetalResidentDecode
        );
    }
    let allocations_after_first = compute::jpeg_private_buffer_allocations_for_test();

    let second = decode_rgb8_buffer_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalBufferBatchTarget::Reusable(&output),
        &session,
    )
    .expect("second decode");
    for surface in second {
        assert_eq!(
            surface.expect("surface").residency(),
            SurfaceResidency::MetalResidentDecode
        );
    }

    assert!(
        allocations_after_first > 0,
        "first batch should allocate private intermediate buffers"
    );
    assert_eq!(
        compute::jpeg_private_buffer_allocations_for_test(),
        allocations_after_first,
        "warm session batch should reuse private intermediate buffers"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn warm_session_reuses_shared_upload_buffers_for_reusable_output_batches() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (16, 16), 2).expect("output buffer");
    let inputs = [BASELINE_420, BASELINE_420];

    compute::reset_jpeg_shared_buffer_allocations_for_test();
    decode_rgb8_buffer_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalBufferBatchTarget::Reusable(&output),
        &session,
    )
    .expect("first decode");
    let allocations_after_first = compute::jpeg_shared_buffer_allocations_for_test();

    decode_rgb8_buffer_batch_with_session(
        Rgb8MetalBatchSource::Bytes(&inputs),
        Rgb8MetalBatchOp::Full,
        MetalBufferBatchTarget::Reusable(&output),
        &session,
    )
    .expect("second decode");

    assert!(
        allocations_after_first > 0,
        "first batch should allocate shared upload/status buffers"
    );
    assert_eq!(
        compute::jpeg_shared_buffer_allocations_for_test(),
        allocations_after_first,
        "warm session batch should reuse shared upload/status buffers"
    );
}
