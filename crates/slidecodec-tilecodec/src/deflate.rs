// SPDX-License-Identifier: Apache-2.0

use crate::{pool::DeflatePool, TileCodecError};
use flate2::read::{DeflateDecoder, ZlibDecoder};
use slidecodec_core::{BufferError, TileDecompress};
use std::io::Read;

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
        pool.scratch.clear();
        match read_all(ZlibDecoder::new(input), &mut pool.scratch) {
            Ok(()) => copy_to_output(&pool.scratch, out),
            Err(zlib_error) => {
                pool.scratch.clear();
                read_all(DeflateDecoder::new(input), &mut pool.scratch).map_err(|raw_error| {
                    TileCodecError::Backend(format!(
                        "deflate decode failed (zlib: {zlib_error}; raw: {raw_error})"
                    ))
                })?;
                copy_to_output(&pool.scratch, out)
            }
        }
    }
}

fn read_all<R: Read>(mut reader: R, scratch: &mut Vec<u8>) -> std::io::Result<()> {
    reader.read_to_end(scratch)?;
    Ok(())
}

fn copy_to_output(decoded: &[u8], out: &mut [u8]) -> Result<usize, TileCodecError> {
    if out.len() < decoded.len() {
        return Err(TileCodecError::Buffer(BufferError::OutputTooSmall {
            required: decoded.len(),
            have: out.len(),
        }));
    }
    out[..decoded.len()].copy_from_slice(decoded);
    Ok(decoded.len())
}
