// SPDX-License-Identifier: Apache-2.0

use crate::{
    bounded::{copy_scratch_to_output, read_to_scratch_bounded, BoundedReadError},
    pool::ZstdPool,
    TileCodecError,
};
use ashlar_core::TileDecompress;

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
        let written = match read_to_scratch_bounded(&mut decoder, &mut pool.scratch, out.len()) {
            Ok(written) => written,
            Err(BoundedReadError::OutputTooSmall(error)) => return Err(error.into()),
            Err(BoundedReadError::Io(error)) => {
                return Err(TileCodecError::Backend(format!(
                    "zstd decode failed: {error}"
                )));
            }
        };

        copy_scratch_to_output(&pool.scratch, out);
        Ok(written)
    }
}
