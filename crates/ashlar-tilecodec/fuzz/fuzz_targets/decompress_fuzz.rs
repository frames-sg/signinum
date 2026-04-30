#![no_main]

use libfuzzer_sys::fuzz_target;
use ashlar_tilecodec::TileDecompress;
use ashlar_tilecodec::{
    DeflateCodec, DeflatePool, LzwCodec, LzwPool, NoPool, UncompressedCodec, ZstdCodec, ZstdPool,
};

fuzz_target!(|data: &[u8]| {
    let mut out = vec![0_u8; data.len().saturating_mul(8).saturating_add(1024)];

    let mut deflate_pool = DeflatePool::new();
    let _ = DeflateCodec::decompress_into(&mut deflate_pool, data, &mut out);

    let mut zstd_pool = ZstdPool::new();
    let _ = ZstdCodec::decompress_into(&mut zstd_pool, data, &mut out);

    let mut lzw_pool = LzwPool::new();
    let _ = LzwCodec::decompress_into(&mut lzw_pool, data, &mut out);

    let mut no_pool = NoPool;
    let _ = UncompressedCodec::decompress_into(&mut no_pool, data, &mut out);
});
