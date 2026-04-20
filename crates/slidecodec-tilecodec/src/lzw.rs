// SPDX-License-Identifier: Apache-2.0

use crate::{pool::LzwPool, TileCodecError};
use slidecodec_core::{BufferError, TileDecompress};
use weezl::{decode::Decoder, BitOrder};

pub struct LzwCodec;

impl TileDecompress for LzwCodec {
    type Error = TileCodecError;
    type Pool = LzwPool;

    fn expected_size(_input: &[u8]) -> Result<Option<usize>, Self::Error> {
        Ok(None)
    }

    fn decompress_into(
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
    ) -> Result<usize, Self::Error> {
        let mut decoder = Decoder::new(BitOrder::Msb, 8);
        pool.scratch = decoder
            .decode(input)
            .map_err(|error| TileCodecError::Backend(format!("lzw decode failed: {error:?}")))?;

        if out.len() < pool.scratch.len() {
            return Err(TileCodecError::Buffer(BufferError::OutputTooSmall {
                required: pool.scratch.len(),
                have: out.len(),
            }));
        }
        out[..pool.scratch.len()].copy_from_slice(&pool.scratch);
        Ok(pool.scratch.len())
    }
}
