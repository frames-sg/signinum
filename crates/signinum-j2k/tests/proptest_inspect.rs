// SPDX-License-Identifier: Apache-2.0

use proptest::prelude::*;
use signinum_j2k::J2kDecoder;

proptest! {
    #[test]
    fn inspect_never_panics_on_arbitrary_bytes(data in proptest::collection::vec(any::<u8>(), 0..512)) {
        let _ = J2kDecoder::inspect(&data);
    }

    #[test]
    fn inspect_never_panics_when_prefixed_with_codestream_marker(mut data in proptest::collection::vec(any::<u8>(), 0..512)) {
        data.splice(0..0, [0xFF, 0x4F]);
        let _ = J2kDecoder::inspect(&data);
    }
}
