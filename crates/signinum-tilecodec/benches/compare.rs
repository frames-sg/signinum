// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use flate2::{write::ZlibEncoder, Compression};
use signinum_core::TileDecompress;
use signinum_tilecodec::{
    DeflateCodec, DeflatePool, LzwCodec, LzwPool, NoPool, UncompressedCodec, ZstdCodec, ZstdPool,
};
use std::io::{Read, Write};
use weezl::{encode::Encoder, BitOrder};

fn sample_bytes() -> Vec<u8> {
    (0..=255).cycle().take(64 * 1024).collect()
}

fn bench_compare(c: &mut Criterion) {
    let source = sample_bytes();

    let mut zlib = ZlibEncoder::new(Vec::new(), Compression::default());
    zlib.write_all(&source).expect("write zlib");
    let deflate = zlib.finish().expect("finish zlib");

    let zstd = zstd::stream::encode_all(std::io::Cursor::new(&source), 1).expect("zstd encode");
    let mut lzw_encoder = Encoder::new(BitOrder::Msb, 8);
    let lzw = lzw_encoder.encode(&source).expect("lzw encode");

    let mut group = c.benchmark_group("decompress_into");
    group.throughput(Throughput::Bytes(source.len() as u64));

    let mut deflate_pool = DeflatePool::new();
    let mut zstd_pool = ZstdPool::new();
    let mut lzw_pool = LzwPool::new();
    let mut no_pool = NoPool;

    group.bench_function(BenchmarkId::new("signinum", "deflate"), |b| {
        let mut out = vec![0_u8; source.len()];
        b.iter(|| {
            DeflateCodec::decompress_into(&mut deflate_pool, &deflate, &mut out)
                .expect("signinum deflate")
        });
    });

    group.bench_function(BenchmarkId::new("reference", "deflate"), |b| {
        let mut scratch = Vec::new();
        b.iter(|| {
            scratch.clear();
            flate2::read::ZlibDecoder::new(deflate.as_slice())
                .read_to_end(&mut scratch)
                .expect("reference deflate");
        });
    });

    group.bench_function(BenchmarkId::new("signinum", "zstd"), |b| {
        let mut out = vec![0_u8; source.len()];
        b.iter(|| {
            ZstdCodec::decompress_into(&mut zstd_pool, &zstd, &mut out).expect("signinum zstd")
        });
    });

    group.bench_function(BenchmarkId::new("reference", "zstd"), |b| {
        let mut scratch = Vec::new();
        b.iter(|| {
            scratch.clear();
            zstd::stream::read::Decoder::new(zstd.as_slice())
                .expect("reference zstd init")
                .read_to_end(&mut scratch)
                .expect("reference zstd");
        });
    });

    group.bench_function(BenchmarkId::new("signinum", "lzw"), |b| {
        let mut out = vec![0_u8; source.len()];
        b.iter(|| LzwCodec::decompress_into(&mut lzw_pool, &lzw, &mut out).expect("signinum lzw"));
    });

    group.bench_function(BenchmarkId::new("reference", "lzw"), |b| {
        let mut decoder = weezl::decode::Decoder::new(BitOrder::Msb, 8);
        b.iter(|| decoder.decode(&lzw).expect("reference lzw"));
    });

    group.bench_function(BenchmarkId::new("signinum", "uncompressed"), |b| {
        let mut out = vec![0_u8; source.len()];
        b.iter(|| {
            UncompressedCodec::decompress_into(&mut no_pool, &source, &mut out)
                .expect("signinum raw")
        });
    });

    group.bench_function(BenchmarkId::new("reference", "uncompressed"), |b| {
        let mut out = vec![0_u8; source.len()];
        b.iter(|| out.copy_from_slice(&source));
    });

    group.finish();
}

criterion_group!(benches, bench_compare);
criterion_main!(benches);
