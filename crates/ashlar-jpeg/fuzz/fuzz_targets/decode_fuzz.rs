// SPDX-License-Identifier: Apache-2.0
#![no_main]

use libfuzzer_sys::fuzz_target;
use ashlar_jpeg::{Decoder, PixelFormat};

/// The invariant: on any arbitrary byte input, `Decoder::new` + `decode_into`
/// either succeeds with a filled buffer or returns a typed `JpegError` — and
/// NEVER panics. libfuzzer aborts on panic; that is the detection mechanism.
///
/// We cap the output buffer at 1 MiB so a fuzzer-crafted SOF declaring
/// enormous dimensions doesn't OOM the host. A real `MemoryCapExceeded`
/// handshake lands in M2 (DecoderBuilder::max_decode_bytes, capability O).
const MAX_OUTPUT_BYTES: usize = 1 << 20;

fuzz_target!(|data: &[u8]| {
    let dec = match Decoder::new(data) {
        Ok(d) => d,
        Err(_) => return,
    };
    let (w, h) = dec.info().dimensions;
    let bpp = 3;
    let needed = (w as usize).saturating_mul(h as usize).saturating_mul(bpp);
    if needed == 0 || needed > MAX_OUTPUT_BYTES {
        return;
    }
    let stride = (w as usize).saturating_mul(bpp);
    let mut out = vec![0u8; needed];
    let _ = dec.decode_into(&mut out, stride, PixelFormat::Rgb8);
});
