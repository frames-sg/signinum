// SPDX-License-Identifier: Apache-2.0

//! Property-based tests: the parser must never panic on arbitrary bytes,
//! must always return a typed `JpegError`, and must not consume unbounded
//! memory for crafted headers.

use proptest::prelude::*;
use signinum_jpeg::Decoder;

proptest! {
    #![proptest_config(ProptestConfig { cases: 4096, .. ProptestConfig::default() })]

    #[test]
    fn inspect_never_panics_on_arbitrary_bytes(bytes in proptest::collection::vec(any::<u8>(), 0..4096)) {
        // The contract is "never panic, always a Result". We don't care if it
        // is Ok or Err — only that it returns.
        let _ = Decoder::inspect(&bytes);
    }

    #[test]
    fn inspect_never_panics_when_prefixed_with_soi(bytes in proptest::collection::vec(any::<u8>(), 0..4096)) {
        let mut input = vec![0xFF, 0xD8];
        input.extend_from_slice(&bytes);
        let _ = Decoder::inspect(&input);
    }

    #[test]
    fn inspect_error_always_returns_io_misuse_false_for_parse_errors(
        bytes in proptest::collection::vec(any::<u8>(), 0..256)
    ) {
        if let Err(e) = Decoder::inspect(&bytes) {
            // Parse errors on arbitrary bytes are never API-misuse errors —
            // `is_api_misuse` variants require caller-visible structures that
            // parsing alone cannot construct.
            prop_assert!(!e.is_api_misuse());
        }
    }
}
