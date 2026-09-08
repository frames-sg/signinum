// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::engine as compute;
#[cfg(target_os = "macos")]
use j2k_native::J2kWaveletTransform;
use j2k_native::{HtCodeBlockDecoder, J2kIdwtNormalization, J2kSingleDecompositionIdwtJob, Result};

#[derive(Default)]
pub(crate) struct MetalIdwtDecoder {
    #[cfg(target_os = "macos")]
    kernel_dispatches: usize,
}

impl MetalIdwtDecoder {
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn kernel_dispatches(&self) -> usize {
        self.kernel_dispatches
    }

    #[cfg_attr(
        not(target_os = "macos"),
        expect(
            clippy::unnecessary_wraps,
            clippy::unused_self,
            reason = "the non-Metal stub preserves the shared fallible decoder contract"
        )
    )]
    fn decode_with_normalization(
        &mut self,
        job: J2kSingleDecompositionIdwtJob<'_>,
        normalization: J2kIdwtNormalization,
        output: &mut [f32],
    ) -> Result<bool> {
        #[cfg(target_os = "macos")]
        if supports_metal_idwt(&job) {
            match job.transform {
                J2kWaveletTransform::Reversible53 => {
                    compute::decode_reversible53_single_decomposition_idwt(job, output)
                }
                J2kWaveletTransform::Irreversible97
                    if normalization == J2kIdwtNormalization::OpenJpegCodestream =>
                {
                    compute::decode_openjpeg_irreversible97_single_decomposition_idwt(job, output)
                }
                J2kWaveletTransform::Irreversible97 => {
                    compute::decode_irreversible97_single_decomposition_idwt(job, output)
                }
            }
            .map_err(|_| j2k_native::DecodingError::CodeBlockDecodeFailure)?;
            self.kernel_dispatches = self.kernel_dispatches.saturating_add(1);
            return Ok(true);
        }
        #[cfg(not(target_os = "macos"))]
        let _ = (job, normalization, output);

        Ok(false)
    }
}

impl HtCodeBlockDecoder for MetalIdwtDecoder {
    fn decode_single_decomposition_idwt(
        &mut self,
        job: J2kSingleDecompositionIdwtJob<'_>,
        output: &mut [f32],
    ) -> Result<bool> {
        self.decode_with_normalization(job, J2kIdwtNormalization::Standard, output)
    }

    fn decode_single_decomposition_idwt_with_normalization(
        &mut self,
        job: J2kSingleDecompositionIdwtJob<'_>,
        normalization: J2kIdwtNormalization,
        output: &mut [f32],
    ) -> Result<bool> {
        self.decode_with_normalization(job, normalization, output)
    }
}

#[cfg(target_os = "macos")]
fn supports_metal_idwt(job: &J2kSingleDecompositionIdwtJob<'_>) -> bool {
    if !matches!(
        job.transform,
        J2kWaveletTransform::Reversible53 | J2kWaveletTransform::Irreversible97
    ) {
        return false;
    }
    let width = job.rect.width();
    let height = job.rect.height();
    if width == 0 || height == 0 {
        return false;
    }

    let expected_output = width as usize * height as usize;
    let expected_band_lengths = [
        job.ll.rect.width() as usize * job.ll.rect.height() as usize,
        job.hl.rect.width() as usize * job.hl.rect.height() as usize,
        job.lh.rect.width() as usize * job.lh.rect.height() as usize,
        job.hh.rect.width() as usize * job.hh.rect.height() as usize,
    ];

    expected_output > 0
        && job.ll.coefficients.len() == expected_band_lengths[0]
        && job.hl.coefficients.len() == expected_band_lengths[1]
        && job.lh.coefficients.len() == expected_band_lengths[2]
        && job.hh.coefficients.len() == expected_band_lengths[3]
}

#[cfg(test)]
mod tests {
    use super::MetalIdwtDecoder;
    use j2k_native::{
        encode, DecodeSettings, DecoderContext, EncodeOptions, HtCodeBlockDecoder, Image,
    };
    use std::io::{Read, Seek, SeekFrom};

    const APERIO_TILE_OFFSET: u64 = 12_696_074;
    const APERIO_TILE_BYTES: usize = 23_080;

    #[cfg(target_os = "macos")]
    fn should_run_metal_runtime() -> bool {
        j2k_test_support::metal_runtime_gate(module_path!())
    }

    fn fixture_j2k_gray8() -> Vec<u8> {
        let pixels: Vec<u8> = (0..16).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode classic gray8")
    }

    fn fixture_j2k_gray8_two_levels() -> Vec<u8> {
        let pixels: Vec<u8> = (0..64).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        };
        encode(&pixels, 8, 8, 1, 8, false, &options).expect("encode classic gray8 two levels")
    }

    fn fixture_j2k_gray8_irreversible() -> Vec<u8> {
        let pixels: Vec<u8> = (0..16).collect();
        let options = EncodeOptions {
            reversible: false,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode classic gray8 irreversible")
    }

    #[test]
    fn metal_idwt_decoder_matches_native_decode() {
        #[cfg(target_os = "macos")]
        if !should_run_metal_runtime() {
            return;
        }

        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut expected_context = DecoderContext::default();
        let expected = image
            .decode_components_with_context(&mut expected_context)
            .expect("native decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalIdwtDecoder::default();
        let actual = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        assert_eq!(actual.dimensions(), expected.dimensions());
        assert_eq!(actual.planes().len(), expected.planes().len());
        assert_eq!(
            actual.planes()[0].samples(),
            expected.planes()[0].samples(),
            "Metal IDWT output must match native decode"
        );
        #[cfg(target_os = "macos")]
        assert!(
            decoder.kernel_dispatches() > 0,
            "single-decomposition grayscale fixture must exercise the Metal IDWT kernel"
        );
    }

    #[test]
    fn metal_idwt_decoder_matches_native_decode_for_two_levels() {
        #[cfg(target_os = "macos")]
        if !should_run_metal_runtime() {
            return;
        }

        let bytes = fixture_j2k_gray8_two_levels();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut expected_context = DecoderContext::default();
        let expected = image
            .decode_components_with_context(&mut expected_context)
            .expect("native decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalIdwtDecoder::default();
        let actual = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        assert_eq!(actual.dimensions(), expected.dimensions());
        assert_eq!(
            actual.planes()[0].samples(),
            expected.planes()[0].samples(),
            "Metal IDWT output must match native decode for multi-level reversible images"
        );
        #[cfg(target_os = "macos")]
        assert!(
            decoder.kernel_dispatches() >= 2,
            "two-level grayscale fixture must dispatch the Metal IDWT kernel for each level"
        );
    }

    #[test]
    fn metal_idwt_decoder_matches_native_decode_for_irreversible_image() {
        #[cfg(target_os = "macos")]
        if !should_run_metal_runtime() {
            return;
        }

        let bytes = fixture_j2k_gray8_irreversible();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut expected_context = DecoderContext::default();
        let expected = image
            .decode_components_with_context(&mut expected_context)
            .expect("native decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalIdwtDecoder::default();
        let actual = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        assert_eq!(actual.dimensions(), expected.dimensions());
        assert_eq!(
            actual.planes()[0].samples(),
            expected.planes()[0].samples(),
            "Metal IDWT output must match native decode for irreversible images"
        );
        #[cfg(target_os = "macos")]
        assert!(
            decoder.kernel_dispatches() > 0,
            "irreversible grayscale fixture must exercise the Metal IDWT kernel"
        );
    }

    #[test]
    fn metal_codestream_normalization_matches_cpu_for_aperio_lossy_tile() {
        #[cfg(target_os = "macos")]
        if !should_run_metal_runtime() {
            return;
        }
        let Some(path) = std::env::var_os("J2K_WSI_SVS_PATH") else {
            eprintln!("J2K_WSI_SVS_PATH is unset; skipping external Aperio Metal fixture");
            return;
        };
        let mut file = std::fs::File::open(path).expect("open Aperio JP2K slide");
        file.seek(SeekFrom::Start(APERIO_TILE_OFFSET))
            .expect("seek to level-zero tile (14, 16)");
        let mut codestream = vec![0_u8; APERIO_TILE_BYTES];
        file.read_exact(&mut codestream)
            .expect("read level-zero tile (14, 16)");
        assert_eq!(
            j2k_test_support::fnv1a64_hex(&codestream),
            "36dc1f181d439cd5"
        );

        let image = Image::new(&codestream, &DecodeSettings::default()).expect("image");
        let mut expected_context = DecoderContext::default();
        let expected = image
            .decode_components_with_context(&mut expected_context)
            .expect("CPU decode");
        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalIdwtDecoder::default();
        let actual = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("Metal IDWT decode");

        assert_eq!(actual.dimensions(), expected.dimensions());
        assert_eq!(actual.planes().len(), expected.planes().len());
        for (component, (actual, expected)) in
            actual.planes().iter().zip(expected.planes()).enumerate()
        {
            assert_eq!(
                actual.samples(),
                expected.samples(),
                "Metal component {component} differs from CPU"
            );
        }
        #[cfg(target_os = "macos")]
        assert!(
            decoder.kernel_dispatches() > 0,
            "Aperio lossy tile must exercise Metal codestream-normalized IDWT"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn staged_irreversible_idwt_matches_native_odd_geometry() {
        const EXPECTED_BITS: [u32; 15] = [
            3_243_516_307,
            1_088_535_889,
            1_086_781_832,
            3_237_068_116,
            1_072_899_983,
            1_090_811_482,
            3_205_449_560,
            3_240_528_043,
            1_057_965_919,
            1_095_801_595,
            3_240_069_383,
            1_090_703_406,
            1_072_530_023,
            3_238_711_343,
            1_091_758_981,
        ];
        if !should_run_metal_runtime() {
            return;
        }
        let rect = j2k_native::J2kRect {
            x0: 0,
            y0: 0,
            x1: 5,
            y1: 3,
        };
        let ll = [0.5, -1.25, 2.0, 3.5, -4.25, 5.75];
        let hl = [6.5, -7.0, 8.25, -9.5];
        let lh = [10.0, -11.5, 12.75];
        let hh = [-13.0, 14.25];
        let job = j2k_native::J2kSingleDecompositionIdwtJob {
            rect,
            transform: j2k_native::J2kWaveletTransform::Irreversible97,
            ll: j2k_native::J2kIdwtBand {
                rect: j2k_native::J2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 3,
                    y1: 2,
                },
                coefficients: &ll,
            },
            hl: j2k_native::J2kIdwtBand {
                rect: j2k_native::J2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 2,
                    y1: 2,
                },
                coefficients: &hl,
            },
            lh: j2k_native::J2kIdwtBand {
                rect: j2k_native::J2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 3,
                    y1: 1,
                },
                coefficients: &lh,
            },
            hh: j2k_native::J2kIdwtBand {
                rect: j2k_native::J2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 2,
                    y1: 1,
                },
                coefficients: &hh,
            },
        };
        let mut actual = vec![0.0; EXPECTED_BITS.len()];

        crate::engine::decode_irreversible97_staged_single_decomposition_idwt(job, &mut actual)
            .expect("staged irreversible Metal IDWT");

        for (index, (actual, expected_bits)) in actual.iter().zip(EXPECTED_BITS).enumerate() {
            let expected = f32::from_bits(expected_bits);
            assert!(
                (actual - expected).abs() <= 2.0e-5,
                "sample {index}: expected {expected}, got {actual}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn staged_irreversible_idwt_preserves_degenerate_origin_scaling() {
        if !should_run_metal_runtime() {
            return;
        }

        let coefficient = [8.0];
        for (x0, y0, expected) in [(0, 0, 8.0_f32), (1, 0, 4.0), (0, 1, 4.0), (1, 1, 2.0)] {
            let rect = j2k_native::J2kRect {
                x0,
                y0,
                x1: x0 + 1,
                y1: y0 + 1,
            };
            let band = j2k_native::J2kIdwtBand {
                rect: j2k_native::J2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                coefficients: &coefficient,
            };
            let job = j2k_native::J2kSingleDecompositionIdwtJob {
                rect,
                transform: j2k_native::J2kWaveletTransform::Irreversible97,
                ll: band,
                hl: band,
                lh: band,
                hh: band,
            };
            let mut actual = [0.0];

            crate::engine::decode_irreversible97_staged_single_decomposition_idwt(job, &mut actual)
                .expect("degenerate staged irreversible Metal IDWT");

            assert_eq!(
                actual[0].to_bits(),
                expected.to_bits(),
                "origin ({x0}, {y0})"
            );
        }
    }

    #[cfg(target_os = "macos")]
    const IRREVERSIBLE_IDWT_PERF_WIDTH: u32 = 1023;
    #[cfg(target_os = "macos")]
    const IRREVERSIBLE_IDWT_PERF_HEIGHT: u32 = 767;

    #[cfg(target_os = "macos")]
    struct IrreversibleIdwtPerfFixture {
        rect: j2k_native::J2kRect,
        ll: Vec<f32>,
        hl: Vec<f32>,
        lh: Vec<f32>,
        hh: Vec<f32>,
    }

    #[cfg(target_os = "macos")]
    impl IrreversibleIdwtPerfFixture {
        fn new() -> Self {
            let low_width = IRREVERSIBLE_IDWT_PERF_WIDTH.div_ceil(2);
            let low_height = IRREVERSIBLE_IDWT_PERF_HEIGHT.div_ceil(2);
            let high_width = IRREVERSIBLE_IDWT_PERF_WIDTH / 2;
            let high_height = IRREVERSIBLE_IDWT_PERF_HEIGHT / 2;
            let make_band = |width: u32, height: u32, seed: u32| {
                (0..width * height)
                    .map(|index| {
                        let value = index.wrapping_mul(37).wrapping_add(seed * 101) % 4093;
                        (f32::from(u16::try_from(value).expect("pattern value fits u16")) - 2046.0)
                            * 0.03125
                    })
                    .collect::<Vec<_>>()
            };
            Self {
                rect: j2k_native::J2kRect {
                    x0: 0,
                    y0: 0,
                    x1: IRREVERSIBLE_IDWT_PERF_WIDTH,
                    y1: IRREVERSIBLE_IDWT_PERF_HEIGHT,
                },
                ll: make_band(low_width, low_height, 1),
                hl: make_band(high_width, low_height, 2),
                lh: make_band(low_width, high_height, 3),
                hh: make_band(high_width, high_height, 4),
            }
        }

        fn job(&self) -> j2k_native::J2kSingleDecompositionIdwtJob<'_> {
            let low_width = IRREVERSIBLE_IDWT_PERF_WIDTH.div_ceil(2);
            let low_height = IRREVERSIBLE_IDWT_PERF_HEIGHT.div_ceil(2);
            let high_width = IRREVERSIBLE_IDWT_PERF_WIDTH / 2;
            let high_height = IRREVERSIBLE_IDWT_PERF_HEIGHT / 2;
            let band = |x1, y1, coefficients| j2k_native::J2kIdwtBand {
                rect: j2k_native::J2kRect {
                    x0: 0,
                    y0: 0,
                    x1,
                    y1,
                },
                coefficients,
            };
            j2k_native::J2kSingleDecompositionIdwtJob {
                rect: self.rect,
                transform: j2k_native::J2kWaveletTransform::Irreversible97,
                ll: band(low_width, low_height, &self.ll),
                hl: band(high_width, low_height, &self.hl),
                lh: band(low_width, high_height, &self.lh),
                hh: band(high_width, high_height, &self.hh),
            }
        }

        fn output() -> Vec<f32> {
            vec![
                0.0;
                IRREVERSIBLE_IDWT_PERF_WIDTH as usize * IRREVERSIBLE_IDWT_PERF_HEIGHT as usize
            ]
        }
    }

    #[cfg(target_os = "macos")]
    struct MetalCaptureGuard {
        manager: objc2::rc::Retained<objc2_metal::MTLCaptureManager>,
    }

    #[cfg(target_os = "macos")]
    impl Drop for MetalCaptureGuard {
        fn drop(&mut self) {
            if self.manager.isCapturing() {
                self.manager.stopCapture();
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "GPU capture harness; run explicitly with --ignored --nocapture"]
    fn metal_irreversible_idwt_gpu_capture() {
        use objc2::runtime::{AnyObject, ProtocolObject};
        use objc2_foundation::{NSString, NSURL};
        use objc2_metal::{
            MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager, MTLDevice,
        };

        if !should_run_metal_runtime() {
            return;
        }
        if std::env::var_os("CARGO_LLVM_COV_TARGET_DIR").is_some() {
            println!("skipping GPU capture while the Metal coverage lane is instrumented");
            return;
        }
        assert_eq!(
            std::env::var("MTL_CAPTURE_ENABLED").as_deref(),
            Ok("1"),
            "set MTL_CAPTURE_ENABLED=1 to enable the Metal capture API"
        );
        let trace_path = std::path::PathBuf::from(
            std::env::var_os("J2K_METAL_CAPTURE_PATH")
                .expect("set J2K_METAL_CAPTURE_PATH to an absolute .gputrace output path"),
        );
        assert!(
            trace_path.is_absolute(),
            "J2K_METAL_CAPTURE_PATH must be absolute"
        );
        assert_eq!(
            trace_path.extension().and_then(std::ffi::OsStr::to_str),
            Some("gputrace"),
            "J2K_METAL_CAPTURE_PATH must end in .gputrace"
        );
        assert!(
            !trace_path.exists(),
            "refusing to overwrite existing GPU trace {}",
            trace_path.display()
        );

        let fixture = IrreversibleIdwtPerfFixture::new();
        let mut output = IrreversibleIdwtPerfFixture::output();
        let device = j2k_metal_support::system_default_device().expect("Metal capture device");
        crate::engine::with_isolated_runtime_for_device_for_test(&device, || {
            crate::engine::decode_irreversible97_staged_single_decomposition_idwt(
                fixture.job(),
                &mut output,
            )?;

            // SAFETY: `sharedCaptureManager` returns Metal's process-wide
            // retained singleton. This ignored macOS-only harness calls it
            // after the runtime/device gate and keeps the returned owner alive
            // until capture is stopped by `MetalCaptureGuard`.
            let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
            assert!(
                manager.supportsDestination(MTLCaptureDestination::GPUTraceDocument),
                "Metal GPU trace documents are unavailable on this host"
            );
            let descriptor = MTLCaptureDescriptor::new();
            let device_object: &ProtocolObject<dyn MTLDevice> = &device;
            let capture_object: &AnyObject = device_object.as_ref();
            // SAFETY: Metal capture descriptors accept exactly an MTLDevice,
            // MTLCommandQueue, or MTLCaptureScope. `capture_object` is the
            // retained `MTLDevice` created above and outlives the capture.
            unsafe { descriptor.setCaptureObject(Some(capture_object)) };
            descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
            let trace_path = NSString::from_str(&trace_path.to_string_lossy());
            let trace_url = NSURL::fileURLWithPath(&trace_path);
            descriptor.setOutputURL(Some(&trace_url));
            manager
                .startCaptureWithDescriptor_error(&descriptor)
                .map_err(|error| crate::Error::MetalRuntime {
                    message: error.to_string(),
                })?;
            let capture = MetalCaptureGuard { manager };
            let result = crate::engine::decode_irreversible97_staged_single_decomposition_idwt(
                fixture.job(),
                &mut output,
            );
            drop(capture);
            result
        })
        .expect("captured irreversible Metal IDWT");

        assert!(
            trace_path.is_dir(),
            "Metal capture did not create GPU trace package {}",
            trace_path.display()
        );
        println!("j2k_metal_idwt97_capture path={}", trace_path.display());
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "performance guard harness; run explicitly with --ignored --nocapture"]
    fn metal_irreversible_idwt_perf_guard() {
        const ITERS: usize = 11;
        if !should_run_metal_runtime() {
            return;
        }
        let fixture = IrreversibleIdwtPerfFixture::new();
        let mut output = IrreversibleIdwtPerfFixture::output();
        crate::engine::decode_irreversible97_staged_single_decomposition_idwt(
            fixture.job(),
            &mut output,
        )
        .expect("warm irreversible Metal IDWT");
        let mut samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let started = std::time::Instant::now();
            crate::engine::decode_irreversible97_staged_single_decomposition_idwt(
                fixture.job(),
                &mut output,
            )
            .expect("measured irreversible Metal IDWT");
            samples.push(started.elapsed());
        }
        samples.sort_unstable();
        let median = samples[ITERS / 2];
        let p25 = samples[ITERS / 4];
        let p75 = samples[ITERS * 3 / 4];
        println!(
            "j2k_metal_idwt97_perf mode=staged size={}x{} iterations={ITERS} median_ms={:.3} p25_ms={:.3} p75_ms={:.3} iqr_ms={:.3}",
            IRREVERSIBLE_IDWT_PERF_WIDTH,
            IRREVERSIBLE_IDWT_PERF_HEIGHT,
            median.as_secs_f64() * 1_000.0,
            p25.as_secs_f64() * 1_000.0,
            p75.as_secs_f64() * 1_000.0,
            p75.saturating_sub(p25).as_secs_f64() * 1_000.0
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "host-slice IDWT performance harness; run explicitly with --ignored --nocapture"]
    fn metal_host_slice_idwt_decode_perf() {
        use std::time::{Duration, Instant};

        if !should_run_metal_runtime() {
            return;
        }
        for reversible in [true, false] {
            for (width, height) in [(512, 512), (509, 383)] {
                let pixels = j2k_test_support::patterned_gray8(width, height);
                let bytes = encode(
                    &pixels,
                    width,
                    height,
                    1,
                    8,
                    false,
                    &EncodeOptions {
                        reversible,
                        num_decomposition_levels: 3,
                        ..EncodeOptions::default()
                    },
                )
                .expect("host-slice IDWT fixture");
                let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
                let mut reference_context = DecoderContext::default();
                let expected = image
                    .decode_components_with_context(&mut reference_context)
                    .expect("native component oracle");
                let mut context = DecoderContext::default();
                let mut decoder = MetalIdwtDecoder::default();
                let actual = image
                    .decode_components_with_ht_decoder(&mut context, &mut decoder)
                    .expect("host-slice IDWT probe");
                assert_eq!(actual.dimensions(), expected.dimensions());
                assert_eq!(actual.planes().len(), 1);
                assert_eq!(
                    actual.planes()[0].samples().len(),
                    expected.planes()[0].samples().len()
                );
                for (actual, expected) in actual.planes()[0]
                    .samples()
                    .iter()
                    .zip(expected.planes()[0].samples())
                {
                    assert!((actual - expected).abs() <= 1.0e-3);
                }
                assert!(decoder.kernel_dispatches() >= 3);
                drop(actual);

                // The prepared image and CPU workspace are retained. Each
                // iteration still executes the actual synchronous host-slice
                // callbacks, including their Metal transfers and completion.
                let mut decode = || {
                    let output = image
                        .decode_components_with_ht_decoder(&mut context, &mut decoder)
                        .expect("measured host-slice IDWT decode");
                    std::hint::black_box(output.planes()[0].samples());
                };
                let warm_started = Instant::now();
                let mut warm_iterations = 0_u64;
                while warm_started.elapsed() < Duration::from_secs(3) {
                    decode();
                    warm_iterations += 1;
                }
                // Fifty samples of roughly 200 ms, based on the three-second
                // warmup. Raw batch durations permit external CI analysis.
                let iterations = (warm_iterations / 15).max(1);
                for sample in 0..50 {
                    let started = Instant::now();
                    for _ in 0..iterations {
                        decode();
                    }
                    let elapsed = started.elapsed();
                    println!(
                        "metal_host_slice_idwt_decode reversible={reversible} width={width} height={height} sample={sample} iterations={iterations} elapsed_ns={}",
                        elapsed.as_nanos()
                    );
                }
            }
        }
    }

    struct CpuOnlyCodeBlockDecoder;

    impl HtCodeBlockDecoder for CpuOnlyCodeBlockDecoder {}

    #[test]
    fn default_decoder_without_idwt_kernel_still_decodes() {
        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut decoder = CpuOnlyCodeBlockDecoder;
        let image_components = image
            .decode_components_with_ht_decoder(&mut context, &mut decoder)
            .expect("decode without idwt override");
        assert_eq!(image_components.dimensions(), (4, 4));
    }
}
