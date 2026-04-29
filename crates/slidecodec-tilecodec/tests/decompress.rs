// SPDX-License-Identifier: Apache-2.0

use flate2::{
    write::{DeflateEncoder, ZlibEncoder},
    Compression,
};
use slidecodec_core::{ScratchPool, TileDecompress};
use slidecodec_tilecodec::{
    DeflateCodec, DeflatePool, LzwCodec, LzwPool, NoPool, TileCodecError, UncompressedCodec,
    ZstdCodec, ZstdPool,
};
use std::io::Write;
use weezl::{encode::Encoder, BitOrder};

fn sample_bytes() -> Vec<u8> {
    (0..=255).cycle().take(8192).collect()
}

#[test]
fn deflate_codec_decodes_zlib_wrapped_payload() {
    let source = sample_bytes();
    let mut compressor = ZlibEncoder::new(Vec::new(), Compression::default());
    compressor.write_all(&source).expect("write zlib");
    let encoded_bytes = compressor.finish().expect("finish zlib");

    let mut pool = DeflatePool::new();
    let mut out = vec![0_u8; source.len()];
    let written =
        DeflateCodec::decompress_into(&mut pool, &encoded_bytes, &mut out).expect("deflate");

    assert_eq!(written, source.len());
    assert_eq!(&out[..written], source.as_slice());
    assert!(pool.bytes_allocated() >= source.len());
}

#[test]
fn deflate_codec_decodes_raw_deflate_payload() {
    let source = sample_bytes();
    let mut compressor = DeflateEncoder::new(Vec::new(), Compression::default());
    compressor.write_all(&source).expect("write deflate");
    let encoded_bytes = compressor.finish().expect("finish deflate");

    let mut pool = DeflatePool::new();
    let mut out = vec![0_u8; source.len()];
    let written =
        DeflateCodec::decompress_into(&mut pool, &encoded_bytes, &mut out).expect("deflate");

    assert_eq!(written, source.len());
    assert_eq!(&out[..written], source.as_slice());
}

#[test]
fn zstd_codec_roundtrips_payload() {
    let source = sample_bytes();
    let encoded = zstd::stream::encode_all(std::io::Cursor::new(&source), 1).expect("zstd encode");

    let mut pool = ZstdPool::new();
    let mut out = vec![0_u8; source.len()];
    let written = ZstdCodec::decompress_into(&mut pool, &encoded, &mut out).expect("zstd");

    assert_eq!(written, source.len());
    assert_eq!(&out[..written], source.as_slice());
}

#[test]
fn lzw_codec_roundtrips_payload() {
    let source = sample_bytes();
    let mut compressor = Encoder::new(BitOrder::Msb, 8);
    let encoded_bytes = compressor.encode(&source).expect("lzw encode");

    let mut pool = LzwPool::new();
    let mut out = vec![0_u8; source.len()];
    let written = LzwCodec::decompress_into(&mut pool, &encoded_bytes, &mut out).expect("lzw");

    assert_eq!(written, source.len());
    assert_eq!(&out[..written], source.as_slice());
}

#[test]
fn uncompressed_codec_copies_input_verbatim() {
    let source = sample_bytes();
    let mut pool = NoPool;
    let mut out = vec![0_u8; source.len()];
    let written = UncompressedCodec::decompress_into(&mut pool, &source, &mut out).expect("copy");

    assert_eq!(written, source.len());
    assert_eq!(out, source);
    assert_eq!(
        UncompressedCodec::expected_size(&source).expect("size"),
        Some(source.len())
    );
}

#[test]
fn codecs_reject_undersized_output() {
    let source = sample_bytes();

    let mut zlib = ZlibEncoder::new(Vec::new(), Compression::default());
    zlib.write_all(&source).expect("write zlib");
    let deflate = zlib.finish().expect("finish zlib");

    let zstd = zstd::stream::encode_all(std::io::Cursor::new(&source), 1).expect("zstd encode");
    let mut lzw_encoder = Encoder::new(BitOrder::Msb, 8);
    let lzw = lzw_encoder.encode(&source).expect("lzw encode");

    let mut deflate_pool = DeflatePool::new();
    let mut zstd_pool = ZstdPool::new();
    let mut lzw_pool = LzwPool::new();
    let mut no_pool = NoPool;
    let mut tiny = vec![0_u8; source.len() / 2];

    assert!(matches!(
        DeflateCodec::decompress_into(&mut deflate_pool, &deflate, &mut tiny),
        Err(TileCodecError::Buffer(_))
    ));
    assert!(matches!(
        ZstdCodec::decompress_into(&mut zstd_pool, &zstd, &mut tiny),
        Err(TileCodecError::Buffer(_))
    ));
    assert!(matches!(
        LzwCodec::decompress_into(&mut lzw_pool, &lzw, &mut tiny),
        Err(TileCodecError::Buffer(_))
    ));
    assert!(matches!(
        UncompressedCodec::decompress_into(&mut no_pool, &source, &mut tiny),
        Err(TileCodecError::Buffer(_))
    ));
}

#[test]
fn deflate_codec_rejects_oversized_zlib_without_full_scratch_allocation() {
    let source = vec![0xA5; 1 << 20];
    let mut compressor = ZlibEncoder::new(Vec::new(), Compression::best());
    compressor.write_all(&source).expect("write zlib");
    let encoded_bytes = compressor.finish().expect("finish zlib");

    let mut pool = DeflatePool::new();
    let mut tiny = vec![0_u8; 128];
    let err = DeflateCodec::decompress_into(&mut pool, &encoded_bytes, &mut tiny).unwrap_err();

    assert!(matches!(
        err,
        TileCodecError::Buffer(slidecodec_core::BufferError::OutputTooSmall {
            required,
            have: 128
        }) if required == 129
    ));
    assert!(
        pool.bytes_allocated() < source.len() / 16,
        "deflate scratch grew to {} bytes for {} decoded bytes",
        pool.bytes_allocated(),
        source.len()
    );
}

#[test]
fn zstd_codec_rejects_oversized_payload_without_full_scratch_allocation() {
    let source = vec![0x5A; 1 << 20];
    let encoded = zstd::stream::encode_all(std::io::Cursor::new(&source), 19).expect("zstd encode");

    let mut pool = ZstdPool::new();
    let mut tiny = vec![0_u8; 128];
    let err = ZstdCodec::decompress_into(&mut pool, &encoded, &mut tiny).unwrap_err();

    assert!(matches!(
        err,
        TileCodecError::Buffer(slidecodec_core::BufferError::OutputTooSmall {
            required,
            have: 128
        }) if required == 129
    ));
    assert!(
        pool.bytes_allocated() < source.len() / 16,
        "zstd scratch grew to {} bytes for {} decoded bytes",
        pool.bytes_allocated(),
        source.len()
    );
}

#[test]
fn lzw_codec_rejects_oversized_payload_without_full_scratch_allocation() {
    let source = vec![0x33; 1 << 20];
    let mut compressor = Encoder::new(BitOrder::Msb, 8);
    let encoded_bytes = compressor.encode(&source).expect("lzw encode");

    let mut pool = LzwPool::new();
    let mut tiny = vec![0_u8; 128];
    let err = LzwCodec::decompress_into(&mut pool, &encoded_bytes, &mut tiny).unwrap_err();

    assert!(matches!(
        err,
        TileCodecError::Buffer(slidecodec_core::BufferError::OutputTooSmall {
            required,
            have: 128
        }) if required == 129
    ));
    assert!(
        pool.bytes_allocated() < source.len() / 16,
        "lzw scratch grew to {} bytes for {} decoded bytes",
        pool.bytes_allocated(),
        source.len()
    );
}

#[test]
fn pools_can_be_reused_across_calls() {
    let source = sample_bytes();
    let encoded = zstd::stream::encode_all(std::io::Cursor::new(&source), 1).expect("zstd encode");
    let mut pool = ZstdPool::new();
    let mut out = vec![0_u8; source.len()];

    let first = ZstdCodec::decompress_into(&mut pool, &encoded, &mut out).expect("first decode");
    let second = ZstdCodec::decompress_into(&mut pool, &encoded, &mut out).expect("second decode");

    assert_eq!(first, source.len());
    assert_eq!(second, source.len());
    assert_eq!(&out[..second], source.as_slice());
    let allocated_before_reset = pool.bytes_allocated();
    assert!(allocated_before_reset >= source.len());

    pool.reset();
    assert_eq!(pool.bytes_allocated(), allocated_before_reset);
}
