// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(not(target_os = "macos"))]
fn main() {
    assert!(
        std::env::var_os("J2K_REQUIRE_METAL_BENCH").is_none(),
        "J2K Metal readback benchmark requires macOS"
    );
    eprintln!("J2K Metal readback benchmark skipped outside macOS");
}

#[cfg(target_os = "macos")]
mod macos {
    use std::{sync::Arc, time::Duration};

    use criterion::{Criterion, Throughput};
    use j2k::{BatchDecodeOptions, BatchLayout, EncodedImage};
    use j2k_core::{copy_tight_pixels_to_strided_output, DeviceSurface, PixelFormat};
    use j2k_metal::{
        download_surfaces_packed, MetalBackendSession, MetalBatchDecodeResult, MetalBatchDecoder,
        Surface,
    };
    use j2k_native::{encode_htj2k, DecodeSettings, EncodeOptions, Image};
    use objc2_metal::{MTLBlitCommandEncoder, MTLCommandEncoder};

    struct SharedFixture {
        session: MetalBackendSession,
        result: MetalBatchDecodeResult,
    }

    impl SharedFixture {
        fn new(width: u32, count: usize) -> Self {
            let pixels = j2k_test_support::patterned_rgb8(width, width);
            let encoded: Arc<[u8]> = Arc::from(
                encode_htj2k(
                    &pixels,
                    width,
                    width,
                    3,
                    8,
                    false,
                    &EncodeOptions {
                        reversible: false,
                        num_decomposition_levels: 3,
                        guard_bits: 2,
                        ..EncodeOptions::default()
                    },
                )
                .expect("encode readback benchmark fixture"),
            );
            let expected = Image::new(&encoded, &DecodeSettings::default())
                .expect("parse readback benchmark fixture")
                .decode_native()
                .expect("decode readback benchmark oracle")
                .data;
            let inputs = (0..count)
                .map(|_| EncodedImage::full(Arc::clone(&encoded)))
                .collect();
            let session = MetalBackendSession::system_default().expect("Metal benchmark session");
            let mut decoder = MetalBatchDecoder::with_backend_session_and_options(
                session.clone(),
                BatchDecodeOptions {
                    layout: BatchLayout::Nhwc,
                    ..BatchDecodeOptions::default()
                },
            );
            let prepared = decoder.prepare(inputs).expect("prepare readback fixture");
            let result = decoder
                .decode_prepared(&prepared)
                .expect("decode readback fixture");
            assert!(result.errors().is_empty());
            assert!(result.group_errors().is_empty());
            let surfaces = result
                .groups()
                .iter()
                .flat_map(j2k_metal::MetalBatchGroup::surfaces)
                .collect::<Vec<_>>();
            assert_eq!(surfaces.len(), count);
            for surface in surfaces {
                assert_eq!(
                    surface.as_bytes().expect("validate readback surface"),
                    expected
                );
            }
            Self { session, result }
        }

        fn surfaces(&self) -> Vec<&Surface> {
            self.result
                .groups()
                .iter()
                .flat_map(j2k_metal::MetalBatchGroup::surfaces)
                .collect()
        }
    }

    fn legacy_download_into(surface: &Surface, output: &mut [u8], stride: usize) {
        // SAFETY: The completed decode result remains immutable throughout the
        // benchmark and no GPU work overlaps these reads.
        let (buffer, byte_offset) =
            unsafe { surface.metal_buffer() }.expect("benchmark Metal surface");
        // SAFETY: The completed surface range remains immutable during this copy.
        let temporary = unsafe {
            j2k_metal_support::checked_buffer_read_vec::<u8>(
                buffer,
                byte_offset,
                surface.byte_len(),
            )
        }
        .expect("legacy surface readback");
        copy_tight_pixels_to_strided_output(
            &temporary,
            surface.dimensions(),
            surface.pixel_format(),
            output,
            stride,
        )
        .expect("legacy strided surface copy");
    }

    fn legacy_download_surfaces_packed(
        session: &MetalBackendSession,
        surfaces: &[&Surface],
    ) -> Vec<u8> {
        use j2k_metal_support::{
            checked_blit_command_encoder, checked_buffer_read_vec, checked_command_buffer,
            checked_command_queue, checked_shared_buffer, commit_and_wait,
        };

        let total = surfaces.iter().map(|surface| surface.byte_len()).sum();
        let staging = checked_shared_buffer(session.device(), total).expect("legacy staging");
        let queue = checked_command_queue(session.device()).expect("legacy readback queue");
        let command = checked_command_buffer(&queue).expect("legacy readback command");
        let blit = checked_blit_command_encoder(&command).expect("legacy readback blit");
        let mut destination_offset = 0usize;
        for surface in surfaces {
            // SAFETY: Completed immutable benchmark surfaces remain retained.
            let (buffer, source_offset) =
                unsafe { surface.metal_buffer() }.expect("legacy benchmark Metal surface");
            // SAFETY: The summed staging allocation covers every exact surface
            // range and retains all resources through completion.
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    buffer,
                    source_offset,
                    &staging,
                    destination_offset,
                    surface.byte_len(),
                );
            }
            destination_offset += surface.byte_len();
        }
        blit.endEncoding();
        commit_and_wait(&command).expect("legacy packed readback completion");
        // SAFETY: The staging blit completed and no writer overlaps this copy.
        unsafe { checked_buffer_read_vec::<u8>(&staging, 0, total) }
            .expect("legacy packed staging copy")
    }

    fn bench_download_into(criterion: &mut Criterion) {
        for width in [512, 1024] {
            let fixture = SharedFixture::new(width, 1);
            let surface = fixture.surfaces()[0];
            let stride = width as usize * PixelFormat::Rgb8.bytes_per_pixel();
            let mut legacy_output = vec![0u8; surface.byte_len()];
            let mut direct_output = vec![0u8; surface.byte_len()];
            legacy_download_into(surface, &mut legacy_output, stride);
            surface
                .download_into(&mut direct_output, stride)
                .expect("direct benchmark probe");
            assert_eq!(legacy_output, direct_output);

            let mut group = criterion.benchmark_group(format!(
                "metal_surface_download_into/{width}x{width}/rgb8/batch-1"
            ));
            group.throughput(Throughput::Bytes(surface.byte_len() as u64));
            group.bench_function("legacy_temporary_vec", |bencher| {
                bencher.iter(|| {
                    legacy_download_into(
                        std::hint::black_box(surface),
                        std::hint::black_box(&mut legacy_output),
                        stride,
                    );
                });
            });
            group.bench_function("direct_shared", |bencher| {
                bencher.iter(|| {
                    std::hint::black_box(surface)
                        .download_into(std::hint::black_box(&mut direct_output), stride)
                        .expect("direct shared benchmark readback");
                });
            });
            group.finish();
        }
    }

    fn bench_packed_shared(criterion: &mut Criterion) {
        for (width, count) in [(512, 16), (1024, 16)] {
            let fixture = SharedFixture::new(width, count);
            let surfaces = fixture.surfaces();
            let legacy = legacy_download_surfaces_packed(&fixture.session, &surfaces);
            let direct = download_surfaces_packed(&fixture.session, &surfaces)
                .expect("direct packed benchmark probe");
            assert_eq!(legacy, direct);

            let mut group = criterion.benchmark_group(format!(
                "metal_surfaces_packed/{width}x{width}/rgb8/batch-{count}"
            ));
            group.throughput(Throughput::Bytes(direct.len() as u64));
            group.bench_function("legacy_stage_all_new_queue", |bencher| {
                bencher.iter(|| {
                    std::hint::black_box(legacy_download_surfaces_packed(
                        &fixture.session,
                        std::hint::black_box(&surfaces),
                    ));
                });
            });
            group.bench_function("direct_shared", |bencher| {
                bencher.iter(|| {
                    std::hint::black_box(
                        download_surfaces_packed(&fixture.session, std::hint::black_box(&surfaces))
                            .expect("direct packed shared benchmark"),
                    );
                });
            });
            group.finish();
        }
    }

    pub(super) fn run() {
        let mut criterion = Criterion::default()
            .sample_size(10)
            .warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(3))
            .configure_from_args();
        bench_download_into(&mut criterion);
        bench_packed_shared(&mut criterion);
        criterion.final_summary();
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run();
}
