// SPDX-License-Identifier: Apache-2.0

use crate::{
    bounded::{copy_scratch_to_output, observed_too_small},
    pool::LzwPool,
    TileCodecError,
};
use signinum_core::TileDecompress;
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
        let limit = out.len().saturating_add(1);
        if pool.scratch.len() != limit {
            pool.scratch.resize(limit, 0);
        }

        let mut input_offset = 0usize;
        let mut output_offset = 0usize;
        loop {
            if output_offset == pool.scratch.len() {
                return Err(TileCodecError::Buffer(observed_too_small(
                    output_offset,
                    out.len(),
                )));
            }

            let result =
                decoder.decode_bytes(&input[input_offset..], &mut pool.scratch[output_offset..]);
            input_offset += result.consumed_in;
            output_offset += result.consumed_out;

            if output_offset > out.len() {
                return Err(TileCodecError::Buffer(observed_too_small(
                    output_offset,
                    out.len(),
                )));
            }

            match result.status {
                Ok(weezl::LzwStatus::Done) => {
                    pool.scratch.truncate(output_offset);
                    let written = copy_scratch_to_output(&pool.scratch, out);
                    return Ok(written);
                }
                Ok(weezl::LzwStatus::Ok) => {}
                Ok(weezl::LzwStatus::NoProgress) => {
                    return Err(TileCodecError::Backend(
                        "lzw decode failed: no progress before end marker".to_string(),
                    ));
                }
                Err(error) => {
                    return Err(TileCodecError::Backend(format!(
                        "lzw decode failed: {error:?}"
                    )));
                }
            }

            if input_offset == input.len() {
                return Err(TileCodecError::Backend(
                    "lzw decode failed: missing end marker".to_string(),
                ));
            }
        }
    }
}
