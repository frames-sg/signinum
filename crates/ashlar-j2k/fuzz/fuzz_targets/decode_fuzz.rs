#![no_main]

use libfuzzer_sys::fuzz_target;
use ashlar_j2k::{J2kDecoder, PixelFormat};

const MAX_PIXELS: u32 = 1 << 20;

fuzz_target!(|data: &[u8]| {
    let Ok(mut decoder) = J2kDecoder::new(data) else {
        return;
    };

    let dims = decoder.info().dimensions;
    let Some(pixels) = dims.0.checked_mul(dims.1) else {
        return;
    };
    if pixels == 0 || pixels > MAX_PIXELS {
        return;
    }

    let fmt = if decoder.info().components == 1 {
        PixelFormat::Gray8
    } else {
        PixelFormat::Rgb8
    };
    let stride = dims.0 as usize * fmt.bytes_per_pixel();
    let mut out = vec![0_u8; stride * dims.1 as usize];
    let _ = decoder.decode_into(&mut out, stride, fmt);
});
