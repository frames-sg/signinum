// SPDX-License-Identifier: Apache-2.0

use crate::{pool::ZstdPool, TileCodecError};
use slidecodec_core::{BufferError, TileDecompress};
use std::io::Read;

pub struct ZstdCodec;

impl TileDecompress for ZstdCodec {
    type Error = TileCodecError;
    type Pool = ZstdPool;

    fn expected_size(_input: &[u8]) -> Result<Option<usize>, Self::Error> {
        Ok(None)
    }

    fn decompress_into(
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
    ) -> Result<usize, Self::Error> {
        pool.scratch.clear();
        let mut decoder = zstd::stream::read::Decoder::new(input).map_err(|error| {
            TileCodecError::Backend(format!("zstd decoder init failed: {error}"))
        })?;
        decoder
            .read_to_end(&mut pool.scratch)
            .map_err(|error| TileCodecError::Backend(format!("zstd decode failed: {error}")))?;

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
