// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use j2k_jpeg::{
    adapter::{
        build_fast420_packet, build_fast422_packet, build_fast444_packet, build_gray_packet,
    },
    encode_jpeg_baseline, Decoder, DecoderContext, JpegBackend, JpegEncodeOptions, JpegError,
    JpegSamples, JpegSubsampling, JpegView, PixelFormat, RowSink, ScratchPool,
};
use j2k_test_support::{patterned_gray8, patterned_rgb8};

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn zeroed_bytes(len: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .expect("reserve deterministic benchmark output");
    bytes.resize(len, 0);
    bytes
}

#[derive(Clone, Copy)]
enum DecodeMode {
    Buffer(PixelFormat),
    Rows,
}

#[derive(Clone, Copy)]
enum FastPacketKind {
    Gray,
    Ybr420,
    Ybr422,
    Ybr444,
}

struct DecodeCase {
    name: &'static str,
    width: u32,
    height: u32,
    bytes: Vec<u8>,
    mode: DecodeMode,
    fast_packet: FastPacketKind,
    expected_output: Option<Vec<u8>>,
    expected_checksum: u64,
}

impl DecodeCase {
    fn new(
        name: &'static str,
        width: u32,
        height: u32,
        bytes: Vec<u8>,
        mode: DecodeMode,
        fast_packet: FastPacketKind,
    ) -> Self {
        let decoder = Decoder::new(&bytes).expect("generated benchmark JPEG must parse");
        assert_eq!(decoder.info().dimensions, (width, height));
        let expected_output = match mode {
            DecodeMode::Buffer(format) => Some(decode_buffer_output(&decoder, format)),
            DecodeMode::Rows => None,
        };
        let expected_checksum = expected_output.as_ref().map_or_else(
            || decode_rows_checksum(&decoder),
            |output| fnv1a_update(FNV_OFFSET_BASIS, output),
        );
        assert_ne!(expected_checksum, FNV_OFFSET_BASIS);
        Self {
            name,
            width,
            height,
            bytes,
            mode,
            fast_packet,
            expected_output,
            expected_checksum,
        }
    }

    fn pixels(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

fn fnv1a_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash = (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME);
    }
    hash
}

fn bytes_per_pixel(format: PixelFormat) -> usize {
    match format {
        PixelFormat::Gray8 => 1,
        PixelFormat::Rgb8 => 3,
        _ => panic!("decode benchmark only supports Gray8 and Rgb8"),
    }
}

fn decode_buffer_output(decoder: &Decoder<'_>, format: PixelFormat) -> Vec<u8> {
    let (width, height) = decoder.info().dimensions;
    let stride = width as usize * bytes_per_pixel(format);
    let mut output = zeroed_bytes(stride * height as usize);
    let outcome = decoder
        .decode_into(&mut output, stride, format)
        .expect("generated benchmark JPEG must decode");
    assert_eq!((outcome.decoded.w, outcome.decoded.h), (width, height));
    output
}

struct ChecksumSink {
    hash: u64,
    expected_y: u32,
    row_bytes: usize,
}

impl ChecksumSink {
    fn new(row_bytes: usize) -> Self {
        Self {
            hash: FNV_OFFSET_BASIS,
            expected_y: 0,
            row_bytes,
        }
    }
}

impl RowSink<u8> for ChecksumSink {
    type Error = JpegError;

    fn write_row(&mut self, y: u32, row: &[u8]) -> Result<(), Self::Error> {
        assert_eq!(y, self.expected_y);
        assert_eq!(row.len(), self.row_bytes);
        self.hash = fnv1a_update(self.hash, row);
        self.expected_y += 1;
        Ok(())
    }
}

fn decode_rows_checksum(decoder: &Decoder<'_>) -> u64 {
    let (width, height) = decoder.info().dimensions;
    let mut sink = ChecksumSink::new(width as usize * 3);
    let outcome = decoder
        .decode_rows(&mut sink)
        .expect("generated benchmark JPEG row decode must succeed");
    assert_eq!((outcome.decoded.w, outcome.decoded.h), (width, height));
    assert_eq!(sink.expected_y, height);
    sink.hash
}

fn encode_gray(width: u32, height: u32) -> Vec<u8> {
    let pixels = patterned_gray8(width, height);
    encode_jpeg_baseline(
        JpegSamples::Gray8 {
            data: &pixels,
            width,
            height,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Gray,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode deterministic grayscale benchmark JPEG")
    .data
}

fn encode_rgb(width: u32, height: u32, subsampling: JpegSubsampling) -> Vec<u8> {
    encode_rgb_with_restart(width, height, subsampling, None)
}

fn encode_rgb_with_restart(
    width: u32,
    height: u32,
    subsampling: JpegSubsampling,
    restart_interval: Option<u16>,
) -> Vec<u8> {
    let pixels = patterned_rgb8(width, height);
    encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &pixels,
            width,
            height,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling,
            restart_interval,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode deterministic RGB benchmark JPEG")
    .data
}

fn decode_cases() -> Vec<DecodeCase> {
    let rgb_420 = encode_rgb(512, 512, JpegSubsampling::Ybr420);
    let mut cases = Vec::new();
    cases
        .try_reserve_exact(7)
        .expect("reserve deterministic benchmark cases");
    cases.push(DecodeCase::new(
        "gray8_512",
        512,
        512,
        encode_gray(512, 512),
        DecodeMode::Buffer(PixelFormat::Gray8),
        FastPacketKind::Gray,
    ));
    cases.push(DecodeCase::new(
        "rgb8_512_444",
        512,
        512,
        encode_rgb(512, 512, JpegSubsampling::Ybr444),
        DecodeMode::Buffer(PixelFormat::Rgb8),
        FastPacketKind::Ybr444,
    ));
    cases.push(DecodeCase::new(
        "rgb8_512_422",
        512,
        512,
        encode_rgb(512, 512, JpegSubsampling::Ybr422),
        DecodeMode::Buffer(PixelFormat::Rgb8),
        FastPacketKind::Ybr422,
    ));
    cases.push(DecodeCase::new(
        "rgb8_512_420",
        512,
        512,
        rgb_420.clone(),
        DecodeMode::Buffer(PixelFormat::Rgb8),
        FastPacketKind::Ybr420,
    ));
    cases.push(DecodeCase::new(
        "rgb8_512_420_restart7",
        512,
        512,
        encode_rgb_with_restart(512, 512, JpegSubsampling::Ybr420, Some(7)),
        DecodeMode::Buffer(PixelFormat::Rgb8),
        FastPacketKind::Ybr420,
    ));
    cases.push(DecodeCase::new(
        "rgb8_257x263_420",
        257,
        263,
        encode_rgb(257, 263, JpegSubsampling::Ybr420),
        DecodeMode::Buffer(PixelFormat::Rgb8),
        FastPacketKind::Ybr420,
    ));
    cases.push(DecodeCase::new(
        "rgb8_512_420_rows",
        512,
        512,
        rgb_420,
        DecodeMode::Rows,
        FastPacketKind::Ybr420,
    ));
    cases
}

fn bench_comparable_decode_routes(c: &mut Criterion) {
    let cases = decode_cases();

    let mut prepared = c.benchmark_group("jpeg_cpu_decode_prepared_context_scratch_output_reused");
    for case in &cases {
        let DecodeMode::Buffer(format) = case.mode else {
            continue;
        };
        let decoder = Decoder::new(&case.bytes).expect("benchmark JPEG must parse");
        let stride = case.width as usize * bytes_per_pixel(format);
        let mut output = zeroed_bytes(stride * case.height as usize);
        let mut scratch = ScratchPool::new();
        decoder
            .decode_into_with_scratch(&mut scratch, &mut output, stride, format)
            .expect("prepared benchmark validation decode must succeed");
        assert_eq!(Some(output.as_slice()), case.expected_output.as_deref());
        prepared.throughput(Throughput::Elements(case.pixels()));
        prepared.bench_function(case.name, |b| {
            b.iter(|| {
                let outcome = decoder
                    .decode_into_with_scratch(&mut scratch, &mut output, stride, format)
                    .expect("prepared benchmark decode must succeed");
                std::hint::black_box((&output, outcome));
            });
        });
    }
    prepared.finish();

    let mut cold = c.benchmark_group("jpeg_cpu_decode_cold_context_scratch_output_reused");
    for case in &cases {
        let DecodeMode::Buffer(format) = case.mode else {
            continue;
        };
        let stride = case.width as usize * bytes_per_pixel(format);
        let mut output = zeroed_bytes(stride * case.height as usize);
        let mut validation_context = DecoderContext::new();
        let validation_view =
            JpegView::parse(case.bytes.as_slice()).expect("cold benchmark JPEG must parse");
        let validation_decoder =
            Decoder::from_view_in_context(validation_view, &mut validation_context)
                .expect("cold benchmark JPEG must prepare");
        let mut validation_scratch = ScratchPool::new();
        validation_decoder
            .decode_into_with_scratch(&mut validation_scratch, &mut output, stride, format)
            .expect("cold benchmark validation decode must succeed");
        assert_eq!(Some(output.as_slice()), case.expected_output.as_deref());
        cold.throughput(Throughput::Elements(case.pixels()));
        cold.bench_function(case.name, |b| {
            b.iter(|| {
                let mut context = DecoderContext::new();
                let view = JpegView::parse(std::hint::black_box(case.bytes.as_slice()))
                    .expect("cold benchmark JPEG must parse");
                let decoder = Decoder::from_view_in_context(view, &mut context)
                    .expect("cold benchmark JPEG must prepare");
                let mut scratch = ScratchPool::new();
                let outcome = decoder
                    .decode_into_with_scratch(&mut scratch, &mut output, stride, format)
                    .expect("cold benchmark decode must succeed");
                std::hint::black_box((&output, outcome));
            });
        });
    }
    cold.finish();
}

fn bench_fast_packet_prepare_with_implicit_context(c: &mut Criterion) {
    let cases = decode_cases();
    let mut group = c.benchmark_group("jpeg_fast_packet_prepare_implicit_context");
    for case in cases
        .iter()
        .filter(|case| matches!(case.mode, DecodeMode::Buffer(_)))
    {
        group.throughput(Throughput::Bytes(case.bytes.len() as u64));
        group.bench_function(case.name, |b| {
            b.iter(|| match case.fast_packet {
                FastPacketKind::Gray => {
                    std::hint::black_box(
                        build_gray_packet(std::hint::black_box(case.bytes.as_slice()))
                            .expect("benchmark grayscale packet preparation must succeed"),
                    );
                }
                FastPacketKind::Ybr420 => {
                    std::hint::black_box(
                        build_fast420_packet(std::hint::black_box(case.bytes.as_slice()))
                            .expect("benchmark 4:2:0 packet preparation must succeed"),
                    );
                }
                FastPacketKind::Ybr422 => {
                    std::hint::black_box(
                        build_fast422_packet(std::hint::black_box(case.bytes.as_slice()))
                            .expect("benchmark 4:2:2 packet preparation must succeed"),
                    );
                }
                FastPacketKind::Ybr444 => {
                    std::hint::black_box(
                        build_fast444_packet(std::hint::black_box(case.bytes.as_slice()))
                            .expect("benchmark 4:4:4 packet preparation must succeed"),
                    );
                }
            });
        });
    }
    group.finish();
}

fn bench_decode_cpu(c: &mut Criterion) {
    let cases = decode_cases();
    let mut group = c.benchmark_group("jpeg_cpu_decode_runtime");
    for case in &cases {
        let decoder = Decoder::new(&case.bytes).expect("benchmark JPEG must parse");
        let expected_checksum = case.expected_checksum;
        group.throughput(Throughput::Elements(case.pixels()));
        match case.mode {
            DecodeMode::Buffer(format) => {
                let stride = case.width as usize * bytes_per_pixel(format);
                let mut output = zeroed_bytes(stride * case.height as usize);
                group.bench_function(case.name, |b| {
                    b.iter(|| {
                        let outcome = decoder
                            .decode_into(&mut output, stride, format)
                            .expect("benchmark decode must succeed");
                        std::hint::black_box(outcome);
                        let checksum = fnv1a_update(FNV_OFFSET_BASIS, &output);
                        debug_assert_eq!(checksum, expected_checksum);
                        std::hint::black_box(checksum);
                    });
                });
            }
            DecodeMode::Rows => {
                group.bench_function(case.name, |b| {
                    b.iter(|| {
                        let mut sink = ChecksumSink::new(case.width as usize * 3);
                        let outcome = decoder
                            .decode_rows(&mut sink)
                            .expect("benchmark row decode must succeed");
                        debug_assert_eq!(sink.hash, expected_checksum);
                        std::hint::black_box((outcome, sink.hash));
                    });
                });
            }
        }
    }
    group.finish();
}

criterion_group! {
    name = decode_cpu_benches;
    config = Criterion::default()
        .confidence_level(0.95)
        .sample_size(50)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(10));
    targets = bench_decode_cpu, bench_comparable_decode_routes, bench_fast_packet_prepare_with_implicit_context
}
criterion_main!(decode_cpu_benches);
