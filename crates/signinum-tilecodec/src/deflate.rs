// SPDX-License-Identifier: Apache-2.0

use crate::{
    bounded::{copy_scratch_to_output, read_to_scratch_bounded, BoundedReadError},
    pool::DeflatePool,
    TileCodecError,
};
use flate2::read::{DeflateDecoder, ZlibDecoder};
use signinum_core::TileDecompress;

pub struct DeflateCodec;

impl TileDecompress for DeflateCodec {
    type Error = TileCodecError;
    type Pool = DeflatePool;

    fn expected_size(_input: &[u8]) -> Result<Option<usize>, Self::Error> {
        Ok(None)
    }

    fn decompress_into(
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
    ) -> Result<usize, Self::Error> {
        match read_to_scratch_bounded(ZlibDecoder::new(input), &mut pool.scratch, out.len()) {
            Ok(written) => {
                copy_scratch_to_output(&pool.scratch, out);
                Ok(written)
            }
            Err(BoundedReadError::OutputTooSmall(error)) => Err(error.into()),
            Err(BoundedReadError::Io(zlib_error)) => {
                pool.scratch.clear();
                match read_to_scratch_bounded(
                    DeflateDecoder::new(input),
                    &mut pool.scratch,
                    out.len(),
                ) {
                    Ok(written) => {
                        copy_scratch_to_output(&pool.scratch, out);
                        Ok(written)
                    }
                    Err(BoundedReadError::OutputTooSmall(error)) => Err(error.into()),
                    Err(BoundedReadError::Io(raw_error)) => Err(TileCodecError::Backend(format!(
                        "deflate decode failed (zlib: {zlib_error}; raw: {raw_error})"
                    ))),
                }
            }
        }
    }
}
