// SPDX-License-Identifier: Apache-2.0

//! Fixture JPEGs for decode integration tests. Inputs are committed under
//! `corpus/conformance/` and embedded via `include_bytes!` so tests remain
//! hermetic (no filesystem dependency at run time).

/// A 16×16 baseline JPEG with 4:2:0 sampling.
pub fn minimal_baseline_420_jpeg() -> Vec<u8> {
    include_bytes!("../../../../corpus/conformance/baseline_420_16x16.jpg").to_vec()
}

/// An 8×8 grayscale (single-component) baseline JPEG.
pub fn grayscale_8x8_jpeg() -> Vec<u8> {
    include_bytes!("../../../../corpus/conformance/grayscale_8x8.jpg").to_vec()
}
