// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

pub(crate) mod classification;
mod libjpeg_turbo;

pub(crate) use self::classification::DecodeMode;
use self::classification::{classify_corpus_input, color_space_mode, CorpusInputClass};
pub(crate) use self::libjpeg_turbo::TurboJpegDecoder;
use signinum_jpeg::{
    decode_tile_region_scaled_into_in_context, decode_tile_scaled_into_in_context, Decoder,
    DecoderContext, Downscale, JpegError, PixelFormat, Rect, RowSink, ScratchPool,
};
use std::fs;
use std::path::{Path, PathBuf};
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace as ZuneColorSpace;
use zune_core::options::DecoderOptions;

const ZUNE_DIMENSION_LIMIT: usize = 1 << 20;

#[derive(Clone)]
pub(crate) struct BenchInput {
    pub(crate) name: String,
    pub(crate) bytes: Vec<u8>,
    pub(crate) dimensions: (u32, u32),
    pub(crate) mode: DecodeMode,
    pub(crate) input_class: CorpusInputClass,
}

pub(crate) fn load_bench_inputs() -> Vec<BenchInput> {
    let mut inputs = vec![
        BenchInput {
            name: "repo/baseline_420_16x16".to_string(),
            bytes: include_bytes!("../../../../corpus/conformance/baseline_420_16x16.jpg").to_vec(),
            dimensions: (16, 16),
            mode: DecodeMode::Rgb,
            input_class: CorpusInputClass::BoundedFullFrame,
        },
        BenchInput {
            name: "repo/grayscale_8x8".to_string(),
            bytes: include_bytes!("../../../../corpus/conformance/grayscale_8x8.jpg").to_vec(),
            dimensions: (8, 8),
            mode: DecodeMode::Gray,
            input_class: CorpusInputClass::BoundedFullFrame,
        },
    ];

    let mut seen = inputs
        .iter()
        .map(|input| input.name.clone())
        .collect::<Vec<_>>();
    for path in
        std::env::split_paths(&std::env::var_os("SIGNINUM_BENCH_INPUTS").unwrap_or_default())
    {
        collect_jpegs(&path, &mut inputs, &mut seen);
    }

    inputs.sort_by(|a, b| a.name.cmp(&b.name));
    inputs
}

fn collect_jpegs(path: &Path, inputs: &mut Vec<BenchInput>, seen: &mut Vec<String>) {
    if path.is_file() {
        push_jpeg(path, inputs, seen);
        return;
    }
    if !path.is_dir() {
        return;
    }

    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let child = entry.path();
            if child.is_dir() {
                stack.push(child);
            } else {
                push_jpeg(&child, inputs, seen);
            }
        }
    }
}

fn push_jpeg(path: &Path, inputs: &mut Vec<BenchInput>, seen: &mut Vec<String>) {
    if !is_jpeg(path) {
        return;
    }
    let Ok(bytes) = fs::read(path) else {
        return;
    };
    let Ok(dec) = Decoder::new(&bytes) else {
        return;
    };
    let Some(mode) = color_space_mode(dec.info().color_space) else {
        return;
    };
    let dimensions = dec.info().dimensions;
    let input_class = classify_corpus_input(dimensions, mode);

    let name = relative_name(path);
    if seen.contains(&name) {
        return;
    }
    seen.push(name.clone());
    inputs.push(BenchInput {
        name,
        bytes,
        dimensions,
        mode,
        input_class,
    });
}

fn relative_name(path: &Path) -> String {
    let absolute = path.canonicalize().unwrap_or_else(|_| PathBuf::from(path));
    if let Some(prefix) = std::env::var_os("HOME") {
        let prefix = PathBuf::from(prefix);
        if let Ok(stripped) = absolute.strip_prefix(prefix) {
            return stripped.display().to_string();
        }
    }
    absolute.display().to_string()
}

fn is_jpeg(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg"))
}

pub(crate) fn signinum_inspect(bytes: &[u8]) {
    let info = Decoder::inspect(bytes).expect("signinum inspect");
    std::hint::black_box(info);
}

pub(crate) fn libjpeg_turbo_available() -> bool {
    libjpeg_turbo::is_available()
}

pub(crate) fn libjpeg_turbo_inspect(decoder: &mut TurboJpegDecoder, bytes: &[u8]) {
    let info = decoder.inspect(bytes).expect("libjpeg-turbo inspect");
    std::hint::black_box(info);
}

pub(crate) fn jpeg_decoder_inspect(bytes: &[u8]) {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    decoder.read_info().expect("jpeg-decoder read_info");
    std::hint::black_box(decoder.info());
}

pub(crate) fn zune_inspect(bytes: &[u8]) {
    let options = DecoderOptions::new_fast()
        .set_max_width(ZUNE_DIMENSION_LIMIT)
        .set_max_height(ZUNE_DIMENSION_LIMIT);
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    decoder.decode_headers().expect("zune-jpeg decode_headers");
    std::hint::black_box(decoder.info());
}

pub(crate) fn signinum_decode(bytes: &[u8], mode: DecodeMode) {
    let dec = Decoder::new(bytes).expect("signinum decoder");
    let fmt = match mode {
        DecodeMode::Gray => PixelFormat::Gray8,
        DecodeMode::Rgb => PixelFormat::Rgb8,
    };
    let (out, _) = dec.decode(fmt).expect("signinum decode");
    std::hint::black_box(out);
}

pub(crate) fn libjpeg_turbo_decode(decoder: &mut TurboJpegDecoder, bytes: &[u8], mode: DecodeMode) {
    let out = match mode {
        DecodeMode::Gray => decoder.decode_gray(bytes),
        DecodeMode::Rgb => decoder.decode_rgb(bytes),
    }
    .expect("libjpeg-turbo decode");
    std::hint::black_box(out);
}

/// Output dimensions (`stride`, `total length`, `PixelFormat`) for `mode`.
pub(crate) fn output_geometry(dec: &Decoder<'_>, mode: DecodeMode) -> (PixelFormat, usize, usize) {
    let (width, height) = dec.info().dimensions;
    match mode {
        DecodeMode::Gray => {
            let len = (width as usize) * (height as usize);
            (PixelFormat::Gray8, width as usize, len)
        }
        DecodeMode::Rgb => {
            let stride = (width as usize) * 3;
            let len = stride * (height as usize);
            (PixelFormat::Rgb8, stride, len)
        }
    }
}

/// Reused-decoder driver: a pre-built `Decoder` decodes into a pre-allocated
/// buffer. Isolates pure decode cost from `Decoder::new` + output allocation —
/// the realistic WSI tile-batch shape. Competitor crates are not called from
/// this helper because neither `zune-jpeg` nor `jpeg-decoder` expose a reusable
/// decoder; fairness is preserved by keeping this group signinum-only.
pub(crate) fn signinum_decode_reused(
    dec: &Decoder<'_>,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) {
    dec.decode_into(out, stride, fmt)
        .expect("signinum decode (reused)");
    std::hint::black_box(&*out);
}

/// Scratch-reuse driver: reuses both the pre-built `Decoder` and a
/// pre-allocated `ScratchPool`. The pool amortizes stripe-buffer and
/// DC-predictor allocations across many tiles — the shape every Phase 3
/// WSI benchmark is trying to surface.
pub(crate) fn signinum_decode_with_scratch(
    dec: &Decoder<'_>,
    pool: &mut ScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) {
    dec.decode_into_with_scratch(pool, out, stride, fmt)
        .expect("signinum decode (scratch)");
    std::hint::black_box(&*out);
}

#[derive(Default)]
struct NullSink;

impl RowSink<u8> for NullSink {
    type Error = JpegError;

    fn write_row(&mut self, _y: u32, _row: &[u8]) -> Result<(), JpegError> {
        Ok(())
    }
}

pub(crate) fn signinum_decode_rows(bytes: &[u8]) {
    let dec = Decoder::new(bytes).expect("signinum decoder");
    let mut sink = NullSink;
    dec.decode_rows(&mut sink).expect("signinum decode_rows");
}

pub(crate) fn signinum_decode_tile_batch(bytes: &[u8], batch_size: usize) {
    let mut ctx = DecoderContext::new();
    let mut pool = ScratchPool::new();
    let mut sink = NullSink;
    for _ in 0..batch_size {
        Decoder::decode_tile(bytes, &mut ctx, &mut pool, &mut sink)
            .expect("signinum decode_tile batch");
    }
}

pub(crate) fn libjpeg_turbo_decode_batch(
    decoder: &mut TurboJpegDecoder,
    bytes: &[u8],
    batch_size: usize,
) {
    for _ in 0..batch_size {
        let out = decoder.decode_rgb(bytes).expect("libjpeg-turbo decode");
        std::hint::black_box(out);
    }
}

pub(crate) fn signinum_decode_tile_batch_scaled(
    bytes: &[u8],
    batch_size: usize,
    factor: Downscale,
) {
    let info = Decoder::inspect(bytes).expect("signinum inspect");
    let out_width = info.dimensions.0.div_ceil(scale_denominator(factor));
    let out_height = info.dimensions.1.div_ceil(scale_denominator(factor));
    let stride = out_width as usize * 3;
    let mut out = vec![0u8; stride * out_height as usize];
    let mut ctx = DecoderContext::new();
    let mut pool = ScratchPool::new();
    for _ in 0..batch_size {
        decode_tile_scaled_into_in_context(
            bytes,
            &mut ctx,
            &mut pool,
            &mut out,
            stride,
            PixelFormat::Rgb8,
            factor,
        )
        .expect("signinum scaled tile batch");
    }
    std::hint::black_box(out);
}

pub(crate) fn signinum_decode_tile_batch_region_scaled(
    bytes: &[u8],
    batch_size: usize,
    side: u32,
    factor: Downscale,
) {
    let info = Decoder::inspect(bytes).expect("signinum inspect");
    let roi = centered_roi(info.dimensions, side);
    let scaled = scaled_rect(roi, factor);
    let stride = scaled.w as usize * 3;
    let mut out = vec![0u8; stride * scaled.h as usize];
    let mut ctx = DecoderContext::new();
    let mut pool = ScratchPool::new();
    for _ in 0..batch_size {
        decode_tile_region_scaled_into_in_context(
            bytes,
            &mut ctx,
            &mut pool,
            &mut out,
            stride,
            PixelFormat::Rgb8,
            roi,
            factor,
        )
        .expect("signinum region-scaled tile batch");
    }
    std::hint::black_box(out);
}

pub(crate) fn signinum_decode_region(bytes: &[u8], side: u32) {
    let dec = Decoder::new(bytes).expect("signinum decoder");
    let roi = centered_roi(dec.info().dimensions, side);
    let (out, _) = dec
        .decode_region(PixelFormat::Rgb8, roi)
        .expect("signinum region decode");
    std::hint::black_box(out);
}

pub(crate) fn libjpeg_turbo_decode_region(decoder: &mut TurboJpegDecoder, bytes: &[u8], roi: Rect) {
    let out = decoder
        .decode_region_rgb(bytes, roi)
        .expect("libjpeg-turbo region decode");
    std::hint::black_box(out);
}

pub(crate) fn signinum_decode_scaled(bytes: &[u8], factor: Downscale) {
    let dec = Decoder::new(bytes).expect("signinum decoder");
    let (out, _) = dec
        .decode_scaled(PixelFormat::Rgb8, factor)
        .expect("signinum scaled decode");
    std::hint::black_box(out);
}

pub(crate) fn libjpeg_turbo_decode_scaled(
    decoder: &mut TurboJpegDecoder,
    bytes: &[u8],
    factor: Downscale,
) {
    let out = decoder
        .decode_scaled_rgb(bytes, factor)
        .expect("libjpeg-turbo scaled decode");
    std::hint::black_box(out);
}

pub(crate) fn signinum_decode_region_scaled(bytes: &[u8], side: u32, factor: Downscale) {
    let dec = Decoder::new(bytes).expect("signinum decoder");
    let roi = centered_roi(dec.info().dimensions, side);
    let (out, _) = dec
        .decode_region_scaled(PixelFormat::Rgb8, roi, factor)
        .expect("signinum scaled region decode");
    std::hint::black_box(out);
}

pub(crate) fn libjpeg_turbo_decode_region_scaled(
    decoder: &mut TurboJpegDecoder,
    bytes: &[u8],
    roi: Rect,
    factor: Downscale,
) {
    let out = decoder
        .decode_region_scaled_rgb(bytes, roi, factor)
        .expect("libjpeg-turbo scaled region decode");
    std::hint::black_box(out);
}

pub(crate) fn jpeg_decoder_decode(bytes: &[u8]) {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    let out = decoder.decode().expect("jpeg-decoder decode");
    std::hint::black_box(out);
}

pub(crate) fn jpeg_decoder_decode_region(bytes: &[u8], side: u32) {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    let out = decoder.decode().expect("jpeg-decoder decode");
    let info = decoder.info().expect("jpeg-decoder info");
    let roi = centered_roi((info.width.into(), info.height.into()), side);
    let cropped = crop_rgb(&out, info.width as usize, roi);
    std::hint::black_box(cropped);
}

pub(crate) fn jpeg_decoder_decode_scaled(bytes: &[u8], factor: Downscale) {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    let out = decoder.decode().expect("jpeg-decoder decode");
    let info = decoder.info().expect("jpeg-decoder info");
    let scaled = decimate_rgb(
        &out,
        info.width as usize,
        info.height as usize,
        scale_denominator(factor) as usize,
    );
    std::hint::black_box(scaled);
}

pub(crate) fn jpeg_decoder_decode_region_scaled(bytes: &[u8], side: u32, factor: Downscale) {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    let out = decoder.decode().expect("jpeg-decoder decode");
    let info = decoder.info().expect("jpeg-decoder info");
    let roi = centered_roi((info.width.into(), info.height.into()), side);
    let cropped = crop_rgb(&out, info.width as usize, roi);
    let scaled = decimate_rgb(
        &cropped,
        roi.w as usize,
        roi.h as usize,
        scale_denominator(factor) as usize,
    );
    std::hint::black_box(scaled);
}

pub(crate) fn zune_decode(bytes: &[u8], mode: DecodeMode) {
    let colorspace = match mode {
        DecodeMode::Gray => ZuneColorSpace::Luma,
        DecodeMode::Rgb => ZuneColorSpace::RGB,
    };
    let options = DecoderOptions::new_fast()
        .set_max_width(ZUNE_DIMENSION_LIMIT)
        .set_max_height(ZUNE_DIMENSION_LIMIT)
        .jpeg_set_out_colorspace(colorspace);
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    let out = decoder.decode().expect("zune-jpeg decode");
    std::hint::black_box(out);
}

pub(crate) fn zune_decode_region(bytes: &[u8], side: u32) {
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(
        ZCursor::new(bytes),
        DecoderOptions::new_fast()
            .set_max_width(ZUNE_DIMENSION_LIMIT)
            .set_max_height(ZUNE_DIMENSION_LIMIT)
            .jpeg_set_out_colorspace(ZuneColorSpace::RGB),
    );
    let out = decoder.decode().expect("zune-jpeg decode");
    let info = decoder.info().expect("zune-jpeg info");
    let roi = centered_roi((info.width as u32, info.height as u32), side);
    let cropped = crop_rgb(&out, info.width.into(), roi);
    std::hint::black_box(cropped);
}

pub(crate) fn zune_decode_scaled(bytes: &[u8], factor: Downscale) {
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(
        ZCursor::new(bytes),
        DecoderOptions::new_fast()
            .set_max_width(ZUNE_DIMENSION_LIMIT)
            .set_max_height(ZUNE_DIMENSION_LIMIT)
            .jpeg_set_out_colorspace(ZuneColorSpace::RGB),
    );
    let out = decoder.decode().expect("zune-jpeg decode");
    let info = decoder.info().expect("zune-jpeg info");
    let scaled = decimate_rgb(
        &out,
        info.width.into(),
        info.height.into(),
        scale_denominator(factor) as usize,
    );
    std::hint::black_box(scaled);
}

pub(crate) fn zune_decode_region_scaled(bytes: &[u8], side: u32, factor: Downscale) {
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(
        ZCursor::new(bytes),
        DecoderOptions::new_fast()
            .set_max_width(ZUNE_DIMENSION_LIMIT)
            .set_max_height(ZUNE_DIMENSION_LIMIT)
            .jpeg_set_out_colorspace(ZuneColorSpace::RGB),
    );
    let out = decoder.decode().expect("zune-jpeg decode");
    let info = decoder.info().expect("zune-jpeg info");
    let roi = centered_roi((info.width as u32, info.height as u32), side);
    let cropped = crop_rgb(&out, info.width.into(), roi);
    let scaled = decimate_rgb(
        &cropped,
        roi.w as usize,
        roi.h as usize,
        scale_denominator(factor) as usize,
    );
    std::hint::black_box(scaled);
}

pub(crate) fn jpeg_decoder_decode_batch_scaled(bytes: &[u8], batch_size: usize, factor: Downscale) {
    for _ in 0..batch_size {
        jpeg_decoder_decode_scaled(bytes, factor);
    }
}

pub(crate) fn libjpeg_turbo_decode_batch_scaled(
    decoder: &mut TurboJpegDecoder,
    bytes: &[u8],
    batch_size: usize,
    factor: Downscale,
) {
    for _ in 0..batch_size {
        let out = decoder
            .decode_scaled_rgb(bytes, factor)
            .expect("libjpeg-turbo scaled decode");
        std::hint::black_box(out);
    }
}

pub(crate) fn libjpeg_turbo_decode_batch_region_scaled(
    decoder: &mut TurboJpegDecoder,
    bytes: &[u8],
    batch_size: usize,
    roi: Rect,
    factor: Downscale,
) {
    for _ in 0..batch_size {
        let out = decoder
            .decode_region_scaled_rgb(bytes, roi, factor)
            .expect("libjpeg-turbo scaled region decode");
        std::hint::black_box(out);
    }
}

pub(crate) fn jpeg_decoder_decode_batch_region_scaled(
    bytes: &[u8],
    batch_size: usize,
    side: u32,
    factor: Downscale,
) {
    for _ in 0..batch_size {
        jpeg_decoder_decode_region_scaled(bytes, side, factor);
    }
}

pub(crate) fn zune_decode_batch_scaled(bytes: &[u8], batch_size: usize, factor: Downscale) {
    for _ in 0..batch_size {
        zune_decode_scaled(bytes, factor);
    }
}

pub(crate) fn zune_decode_batch_region_scaled(
    bytes: &[u8],
    batch_size: usize,
    side: u32,
    factor: Downscale,
) {
    for _ in 0..batch_size {
        zune_decode_region_scaled(bytes, side, factor);
    }
}

pub(crate) fn centered_roi((width, height): (u32, u32), side: u32) -> Rect {
    let w = side.min(width);
    let h = side.min(height);
    Rect {
        x: (width - w) / 2,
        y: (height - h) / 2,
        w,
        h,
    }
}

pub(crate) fn scaled_rect(rect: Rect, factor: Downscale) -> Rect {
    let denom = scale_denominator(factor);
    let x_end = rect.x + rect.w;
    let y_end = rect.y + rect.h;
    Rect {
        x: rect.x / denom,
        y: rect.y / denom,
        w: x_end.div_ceil(denom) - rect.x / denom,
        h: y_end.div_ceil(denom) - rect.y / denom,
    }
}

fn scale_denominator(factor: Downscale) -> u32 {
    match factor {
        Downscale::None => 1,
        Downscale::Half => 2,
        Downscale::Quarter => 4,
        Downscale::Eighth => 8,
        _ => unreachable!("unsupported Downscale variant"),
    }
}

fn crop_rgb(full: &[u8], width: usize, roi: Rect) -> Vec<u8> {
    let stride = width * 3;
    let mut out = vec![0u8; roi.w as usize * roi.h as usize * 3];
    for row in 0..roi.h as usize {
        let src_start = (roi.y as usize + row) * stride + roi.x as usize * 3;
        let src_end = src_start + roi.w as usize * 3;
        let dst_start = row * roi.w as usize * 3;
        out[dst_start..dst_start + roi.w as usize * 3].copy_from_slice(&full[src_start..src_end]);
    }
    out
}

fn decimate_rgb(full: &[u8], width: usize, height: usize, denom: usize) -> Vec<u8> {
    let out_width = width.div_ceil(denom);
    let out_height = height.div_ceil(denom);
    let mut out = vec![0u8; out_width * out_height * 3];
    for y in 0..out_height {
        let src_y = (y * denom).min(height.saturating_sub(1));
        for x in 0..out_width {
            let src_x = (x * denom).min(width.saturating_sub(1));
            let src = (src_y * width + src_x) * 3;
            let dst = (y * out_width + x) * 3;
            out[dst..dst + 3].copy_from_slice(&full[src..src + 3]);
        }
    }
    out
}
