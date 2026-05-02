#![no_main]

use libfuzzer_sys::fuzz_target;
use signinum_j2k::J2kDecoder;

fuzz_target!(|data: &[u8]| {
    let _ = J2kDecoder::inspect(data);
});
