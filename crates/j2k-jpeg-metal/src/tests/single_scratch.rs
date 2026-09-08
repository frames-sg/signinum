// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
#[ignore = "allocation diagnostic; run explicitly with --ignored --nocapture"]
fn metal_single_session_buffer_allocation_report() {
    if !should_run_metal_runtime() {
        return;
    }
    for (sampling, input) in [
        ("420", BASELINE_420),
        ("420-restart", BASELINE_420_RESTART),
        ("422", BASELINE_422),
        ("444", BASELINE_444),
    ] {
        for fmt in [PixelFormat::Gray8, PixelFormat::Rgb8, PixelFormat::Rgba8] {
            let session = MetalBackendSession::system_default().expect("session");
            // Exclude one-time runtime resources from per-decode allocation counts.
            session.runtime_result().as_ref().expect("runtime");
            let mut decoder = Decoder::new(input).expect("decoder");
            let (expected, _) = CpuDecoder::new(input)
                .expect("native decoder")
                .decode_request(DecodeRequest::full(fmt))
                .expect("native pixels");
            for iteration in 0..3 {
                compute::reset_jpeg_private_buffer_allocations_for_test();
                compute::reset_jpeg_shared_buffer_allocations_for_test();
                let surface = decoder
                    .decode_to_device_with_session(fmt, &session)
                    .expect("single decode");
                let private = compute::jpeg_private_buffer_allocations_for_test();
                let shared = compute::jpeg_shared_buffer_allocations_for_test();
                assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
                let mut actual = vec![0; expected.len()];
                let stride = surface.dimensions().0 as usize * fmt.bytes_per_pixel();
                surface.download_into(&mut actual, stride).expect("pixels");
                assert_eq!(actual, expected);
                println!("metal_single_session_allocations sampling={sampling} format={fmt:?} iteration={iteration} private_buffers={private} shared_buffers={shared} returned_surfaces=1");
            }
        }
    }
}
