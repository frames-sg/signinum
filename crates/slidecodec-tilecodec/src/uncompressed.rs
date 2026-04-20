// SPDX-License-Identifier: Apache-2.0

use crate::{pool::NoPool, TileCodecError};
use slidecodec_core::{BufferError, TileDecompress};

pub struct UncompressedCodec;

impl TileDecompress for UncompressedCodec {
    type Error = TileCodecError;
    type Pool = NoPool;

    fn expected_size(input: &[u8]) -> Result<Option<usize>, Self::Error> {
        Ok(Some(input.len()))
    }

    fn decompress_into(
        _pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
    ) -> Result<usize, Self::Error> {
        if out.len() < input.len() {
            return Err(TileCodecError::Buffer(BufferError::OutputTooSmall {
                required: input.len(),
                have: out.len(),
            }));
        }
        out[..input.len()].copy_from_slice(input);
        Ok(input.len())
    }
}
