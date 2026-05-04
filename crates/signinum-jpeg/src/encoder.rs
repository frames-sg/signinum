// SPDX-License-Identifier: Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    clippy::many_single_char_names
)]

use alloc::string::String;
use alloc::vec::Vec;
use core::f64::consts::PI;

use thiserror::Error;

use crate::entropy::ZIGZAG;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegBackend {
    Auto,
    Cpu,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegSubsampling {
    Gray,
    Ybr444,
    Ybr422,
    Ybr420,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegEncodeOptions {
    pub quality: u8,
    pub subsampling: JpegSubsampling,
    pub restart_interval: Option<u16>,
    pub backend: JpegBackend,
}

impl Default for JpegEncodeOptions {
    fn default() -> Self {
        Self {
            quality: 90,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: None,
            backend: JpegBackend::Auto,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum JpegSamples<'a> {
    Gray8 {
        data: &'a [u8],
        width: u32,
        height: u32,
    },
    Rgb8 {
        data: &'a [u8],
        width: u32,
        height: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedJpeg {
    pub data: Vec<u8>,
    pub backend: JpegBackend,
}

#[derive(Debug, Error)]
pub enum JpegEncodeError {
    #[error("JPEG encode requires nonzero dimensions")]
    EmptyDimensions,
    #[error("JPEG baseline dimensions must fit in u16, got {width}x{height}")]
    DimensionsTooLarge { width: u32, height: u32 },
    #[error("JPEG sample buffer length mismatch: expected {expected}, got {actual}")]
    SampleLength { expected: usize, actual: usize },
    #[error("JPEG subsampling {subsampling:?} is incompatible with {samples}")]
    IncompatibleSubsampling {
        subsampling: JpegSubsampling,
        samples: &'static str,
    },
    #[error("JPEG restart interval must be nonzero when provided")]
    InvalidRestartInterval,
    #[error("JPEG encode backend {backend:?} is unavailable in signinum-jpeg CPU crate")]
    UnsupportedBackend { backend: JpegBackend },
    #[error("JPEG encoded marker segment is too large: {name}")]
    SegmentTooLarge { name: &'static str },
    #[error("JPEG entropy symbol has no Huffman code: {symbol}")]
    MissingHuffmanCode { symbol: u8 },
    #[error("JPEG encode failed: {0}")]
    Internal(String),
}

#[derive(Clone, Copy)]
struct Sampling {
    components: u8,
    h: [u8; 3],
    v: [u8; 3],
    max_h: u8,
    max_v: u8,
}

#[derive(Clone, Copy)]
struct HuffmanCode {
    code: u16,
    len: u8,
}

struct HuffmanEncoder {
    codes: [Option<HuffmanCode>; 256],
}

struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    used: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current: 0,
            used: 0,
        }
    }

    fn write_bits(&mut self, code: u16, len: u8) {
        for bit_idx in (0..len).rev() {
            let bit = ((code >> bit_idx) & 1) as u8;
            self.current = (self.current << 1) | bit;
            self.used += 1;
            if self.used == 8 {
                self.push_byte(self.current);
                self.current = 0;
                self.used = 0;
            }
        }
    }

    fn align_with_ones(&mut self) {
        if self.used == 0 {
            return;
        }
        let remaining = 8 - self.used;
        self.current <<= remaining;
        self.current |= (1u8 << remaining) - 1;
        self.push_byte(self.current);
        self.current = 0;
        self.used = 0;
    }

    fn push_restart_marker(&mut self, rst: u8) {
        self.align_with_ones();
        self.bytes.push(0xFF);
        self.bytes.push(0xD0 + (rst & 0x07));
    }

    fn into_bytes(mut self) -> Vec<u8> {
        self.align_with_ones();
        self.bytes
    }

    fn push_byte(&mut self, byte: u8) {
        self.bytes.push(byte);
        if byte == 0xFF {
            self.bytes.push(0x00);
        }
    }
}

pub fn encode_jpeg_baseline(
    samples: JpegSamples<'_>,
    options: JpegEncodeOptions,
) -> Result<EncodedJpeg, JpegEncodeError> {
    match options.backend {
        JpegBackend::Auto | JpegBackend::Cpu => encode_jpeg_baseline_cpu(samples, options),
        JpegBackend::Metal => Err(JpegEncodeError::UnsupportedBackend {
            backend: options.backend,
        }),
    }
}

fn encode_jpeg_baseline_cpu(
    samples: JpegSamples<'_>,
    options: JpegEncodeOptions,
) -> Result<EncodedJpeg, JpegEncodeError> {
    if options.restart_interval == Some(0) {
        return Err(JpegEncodeError::InvalidRestartInterval);
    }
    let (width, height) = samples.dimensions();
    validate_dimensions(width, height)?;
    samples.validate(options.subsampling)?;

    let sampling = sampling_for(options.subsampling);
    let q_luma = scaled_quant_table(&STD_LUMA_Q, options.quality);
    let q_chroma = scaled_quant_table(&STD_CHROMA_Q, options.quality);
    let huff_dc_luma = HuffmanEncoder::new(&STD_LUMA_DC_BITS, &STD_LUMA_DC_VALUES)?;
    let huff_ac_luma = HuffmanEncoder::new(&STD_LUMA_AC_BITS, &STD_LUMA_AC_VALUES)?;
    let huff_dc_chroma = HuffmanEncoder::new(&STD_CHROMA_DC_BITS, &STD_CHROMA_DC_VALUES)?;
    let huff_ac_chroma = HuffmanEncoder::new(&STD_CHROMA_AC_BITS, &STD_CHROMA_AC_VALUES)?;
    let cosine = cosine_table();
    let planes = component_planes(samples, options.subsampling)?;

    let mut out = Vec::new();
    write_marker(&mut out, 0xD8);
    write_dqt(&mut out, 0, &q_luma)?;
    if sampling.components == 3 {
        write_dqt(&mut out, 1, &q_chroma)?;
    }
    if let Some(restart_interval) = options.restart_interval {
        write_dri(&mut out, restart_interval)?;
    }
    write_sof0(&mut out, width, height, sampling)?;
    write_dht(&mut out, 0, 0, &STD_LUMA_DC_BITS, &STD_LUMA_DC_VALUES)?;
    write_dht(&mut out, 1, 0, &STD_LUMA_AC_BITS, &STD_LUMA_AC_VALUES)?;
    if sampling.components == 3 {
        write_dht(&mut out, 0, 1, &STD_CHROMA_DC_BITS, &STD_CHROMA_DC_VALUES)?;
        write_dht(&mut out, 1, 1, &STD_CHROMA_AC_BITS, &STD_CHROMA_AC_VALUES)?;
    }
    write_sos(&mut out, sampling.components)?;

    let entropy = encode_entropy(
        &planes,
        width,
        height,
        sampling,
        &q_luma,
        &q_chroma,
        [&huff_dc_luma, &huff_dc_chroma],
        [&huff_ac_luma, &huff_ac_chroma],
        &cosine,
        options.restart_interval,
    )?;
    out.extend_from_slice(&entropy);
    write_marker(&mut out, 0xD9);

    Ok(EncodedJpeg {
        data: out,
        backend: JpegBackend::Cpu,
    })
}

impl JpegSamples<'_> {
    fn dimensions(self) -> (u32, u32) {
        match self {
            Self::Gray8 { width, height, .. } | Self::Rgb8 { width, height, .. } => (width, height),
        }
    }

    fn validate(self, subsampling: JpegSubsampling) -> Result<(), JpegEncodeError> {
        let (data, width, height, components, name) = match self {
            Self::Gray8 {
                data,
                width,
                height,
            } => (data, width, height, 1usize, "Gray8"),
            Self::Rgb8 {
                data,
                width,
                height,
            } => (data, width, height, 3usize, "Rgb8"),
        };
        let expected = width as usize * height as usize * components;
        if data.len() != expected {
            return Err(JpegEncodeError::SampleLength {
                expected,
                actual: data.len(),
            });
        }
        match (name, subsampling) {
            ("Gray8", JpegSubsampling::Gray) => Ok(()),
            (
                "Rgb8",
                JpegSubsampling::Ybr444 | JpegSubsampling::Ybr422 | JpegSubsampling::Ybr420,
            ) => Ok(()),
            _ => Err(JpegEncodeError::IncompatibleSubsampling {
                subsampling,
                samples: name,
            }),
        }
    }
}

fn validate_dimensions(width: u32, height: u32) -> Result<(), JpegEncodeError> {
    if width == 0 || height == 0 {
        return Err(JpegEncodeError::EmptyDimensions);
    }
    if width > u16::MAX as u32 || height > u16::MAX as u32 {
        return Err(JpegEncodeError::DimensionsTooLarge { width, height });
    }
    Ok(())
}

fn sampling_for(subsampling: JpegSubsampling) -> Sampling {
    match subsampling {
        JpegSubsampling::Gray => Sampling {
            components: 1,
            h: [1, 0, 0],
            v: [1, 0, 0],
            max_h: 1,
            max_v: 1,
        },
        JpegSubsampling::Ybr444 => Sampling {
            components: 3,
            h: [1, 1, 1],
            v: [1, 1, 1],
            max_h: 1,
            max_v: 1,
        },
        JpegSubsampling::Ybr422 => Sampling {
            components: 3,
            h: [2, 1, 1],
            v: [1, 1, 1],
            max_h: 2,
            max_v: 1,
        },
        JpegSubsampling::Ybr420 => Sampling {
            components: 3,
            h: [2, 1, 1],
            v: [2, 1, 1],
            max_h: 2,
            max_v: 2,
        },
    }
}

fn component_planes(
    samples: JpegSamples<'_>,
    subsampling: JpegSubsampling,
) -> Result<Vec<Vec<u8>>, JpegEncodeError> {
    match samples {
        JpegSamples::Gray8 { data, .. } => Ok(vec![data.to_vec()]),
        JpegSamples::Rgb8 {
            data,
            width,
            height,
        } => {
            if subsampling == JpegSubsampling::Gray {
                return Err(JpegEncodeError::IncompatibleSubsampling {
                    subsampling,
                    samples: "Rgb8",
                });
            }
            let pixels = width as usize * height as usize;
            let mut y_plane = Vec::with_capacity(pixels);
            let mut cb_plane = Vec::with_capacity(pixels);
            let mut cr_plane = Vec::with_capacity(pixels);
            for rgb in data.chunks_exact(3) {
                let (y, cb, cr) = rgb_to_ycbcr(rgb[0], rgb[1], rgb[2]);
                y_plane.push(y);
                cb_plane.push(cb);
                cr_plane.push(cr);
            }
            Ok(vec![y_plane, cb_plane, cr_plane])
        }
    }
}

fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = i32::from(r);
    let g = i32::from(g);
    let b = i32::from(b);
    let y = (19_595 * r + 38_470 * g + 7_471 * b + 32_768) >> 16;
    let cb = (-11_059 * r - 21_709 * g + 32_768 * b + 8_421_376) >> 16;
    let cr = (32_768 * r - 27_439 * g - 5_329 * b + 8_421_376) >> 16;
    (clamp_u8(y), clamp_u8(cb), clamp_u8(cr))
}

fn clamp_u8(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

#[allow(clippy::too_many_arguments)]
fn encode_entropy(
    planes: &[Vec<u8>],
    width: u32,
    height: u32,
    sampling: Sampling,
    q_luma: &[u8; 64],
    q_chroma: &[u8; 64],
    dc_tables: [&HuffmanEncoder; 2],
    ac_tables: [&HuffmanEncoder; 2],
    cosine: &[[f64; 8]; 8],
    restart_interval: Option<u16>,
) -> Result<Vec<u8>, JpegEncodeError> {
    let mcu_width = u32::from(sampling.max_h) * 8;
    let mcu_height = u32::from(sampling.max_v) * 8;
    let mcus_per_row = width.div_ceil(mcu_width);
    let mcu_rows = height.div_ceil(mcu_height);
    let mut writer = BitWriter::new();
    let mut prev_dc = [0i32; 3];
    let mut mcus_since_restart = 0u16;
    let mut rst = 0u8;

    for mcu_y in 0..mcu_rows {
        for mcu_x in 0..mcus_per_row {
            if let Some(interval) = restart_interval {
                if mcus_since_restart == interval {
                    writer.push_restart_marker(rst);
                    rst = (rst + 1) & 7;
                    prev_dc = [0; 3];
                    mcus_since_restart = 0;
                }
            }
            for component in 0..sampling.components as usize {
                let quant = if component == 0 { q_luma } else { q_chroma };
                let dc_table = if component == 0 {
                    dc_tables[0]
                } else {
                    dc_tables[1]
                };
                let ac_table = if component == 0 {
                    ac_tables[0]
                } else {
                    ac_tables[1]
                };
                for block_y in 0..sampling.v[component] {
                    for block_x in 0..sampling.h[component] {
                        let block = sample_block(
                            planes, width, height, sampling, component, mcu_x, mcu_y, block_x,
                            block_y,
                        );
                        let coeffs = fdct_quantize(&block, quant, cosine);
                        encode_block(
                            &coeffs,
                            &mut prev_dc[component],
                            dc_table,
                            ac_table,
                            &mut writer,
                        )?;
                    }
                }
            }
            mcus_since_restart = mcus_since_restart.saturating_add(1);
        }
    }

    Ok(writer.into_bytes())
}

#[allow(clippy::too_many_arguments)]
fn sample_block(
    planes: &[Vec<u8>],
    width: u32,
    height: u32,
    sampling: Sampling,
    component: usize,
    mcu_x: u32,
    mcu_y: u32,
    block_x: u8,
    block_y: u8,
) -> [u8; 64] {
    let mut out = [0u8; 64];
    let max_h = u32::from(sampling.max_h);
    let max_v = u32::from(sampling.max_v);
    let comp_h = u32::from(sampling.h[component]);
    let comp_v = u32::from(sampling.v[component]);
    let x_scale = max_h / comp_h;
    let y_scale = max_v / comp_v;
    let mcu_origin_x = mcu_x * max_h * 8;
    let mcu_origin_y = mcu_y * max_v * 8;
    for y in 0..8u32 {
        for x in 0..8u32 {
            let value = if component == 0 {
                let sx = (mcu_origin_x + u32::from(block_x) * 8 + x).min(width - 1);
                let sy = (mcu_origin_y + u32::from(block_y) * 8 + y).min(height - 1);
                planes[component][(sy as usize * width as usize) + sx as usize]
            } else {
                let mut sum = 0u32;
                for dy in 0..y_scale {
                    for dx in 0..x_scale {
                        let sx = (mcu_origin_x + (u32::from(block_x) * 8 + x) * x_scale + dx)
                            .min(width - 1);
                        let sy = (mcu_origin_y + (u32::from(block_y) * 8 + y) * y_scale + dy)
                            .min(height - 1);
                        sum += u32::from(
                            planes[component][sy as usize * width as usize + sx as usize],
                        );
                    }
                }
                (sum / (x_scale * y_scale)) as u8
            };
            out[(y * 8 + x) as usize] = value;
        }
    }
    out
}

fn fdct_quantize(block: &[u8; 64], quant: &[u8; 64], cosine: &[[f64; 8]; 8]) -> [i32; 64] {
    let mut coeffs = [0i32; 64];
    for v in 0..8 {
        for u in 0..8 {
            let mut sum = 0.0;
            for y in 0..8 {
                for x in 0..8 {
                    let sample = f64::from(block[y * 8 + x]) - 128.0;
                    sum += sample * cosine[u][x] * cosine[v][y];
                }
            }
            let cu = if u == 0 {
                core::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let cv = if v == 0 {
                core::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let natural = v * 8 + u;
            let transformed = 0.25 * cu * cv * sum;
            coeffs[natural] = (transformed / f64::from(quant[natural])).round() as i32;
        }
    }
    coeffs
}

fn encode_block(
    coeffs: &[i32; 64],
    prev_dc: &mut i32,
    dc_table: &HuffmanEncoder,
    ac_table: &HuffmanEncoder,
    writer: &mut BitWriter,
) -> Result<(), JpegEncodeError> {
    let diff = coeffs[0] - *prev_dc;
    *prev_dc = coeffs[0];
    let dc_size = magnitude_category(diff);
    dc_table.write_symbol(dc_size, writer)?;
    if dc_size > 0 {
        writer.write_bits(magnitude_bits(diff, dc_size), dc_size);
    }

    let mut zero_run = 0u8;
    for k in 1..64 {
        let coeff = coeffs[ZIGZAG[k] as usize];
        if coeff == 0 {
            zero_run = zero_run.saturating_add(1);
            continue;
        }
        while zero_run >= 16 {
            ac_table.write_symbol(0xF0, writer)?;
            zero_run -= 16;
        }
        let size = magnitude_category(coeff);
        let symbol = (zero_run << 4) | size;
        ac_table.write_symbol(symbol, writer)?;
        writer.write_bits(magnitude_bits(coeff, size), size);
        zero_run = 0;
    }
    if zero_run > 0 {
        ac_table.write_symbol(0, writer)?;
    }
    Ok(())
}

fn magnitude_category(value: i32) -> u8 {
    if value == 0 {
        return 0;
    }
    let mut abs = value.unsigned_abs();
    let mut size = 0u8;
    while abs > 0 {
        size += 1;
        abs >>= 1;
    }
    size
}

fn magnitude_bits(value: i32, size: u8) -> u16 {
    if size == 0 {
        return 0;
    }
    if value >= 0 {
        value as u16
    } else {
        (value + ((1i32 << size) - 1)) as u16
    }
}

fn cosine_table() -> [[f64; 8]; 8] {
    let mut table = [[0.0; 8]; 8];
    for u in 0..8 {
        for x in 0..8 {
            table[u][x] = (((2 * x + 1) as f64 * u as f64 * PI) / 16.0).cos();
        }
    }
    table
}

impl HuffmanEncoder {
    fn new(bits: &[u8; 16], values: &[u8]) -> Result<Self, JpegEncodeError> {
        let mut codes = [None; 256];
        let mut code = 0u16;
        let mut idx = 0usize;
        for (len_minus_1, count) in bits.iter().copied().enumerate() {
            let len = (len_minus_1 + 1) as u8;
            for _ in 0..count {
                let symbol = *values.get(idx).ok_or_else(|| {
                    JpegEncodeError::Internal("Huffman table count exceeds values".into())
                })?;
                codes[symbol as usize] = Some(HuffmanCode { code, len });
                code = code
                    .checked_add(1)
                    .ok_or_else(|| JpegEncodeError::Internal("Huffman code overflow".into()))?;
                idx += 1;
            }
            code <<= 1;
        }
        if idx != values.len() {
            return Err(JpegEncodeError::Internal(
                "Huffman values exceed table counts".into(),
            ));
        }
        Ok(Self { codes })
    }

    fn write_symbol(&self, symbol: u8, writer: &mut BitWriter) -> Result<(), JpegEncodeError> {
        let code =
            self.codes[symbol as usize].ok_or(JpegEncodeError::MissingHuffmanCode { symbol })?;
        writer.write_bits(code.code, code.len);
        Ok(())
    }
}

fn scaled_quant_table(base: &[u8; 64], quality: u8) -> [u8; 64] {
    let quality = quality.clamp(1, 100);
    let scale = if quality < 50 {
        5000 / u32::from(quality)
    } else {
        200 - u32::from(quality) * 2
    };
    let mut out = [0u8; 64];
    for (idx, value) in base.iter().copied().enumerate() {
        let scaled = (u32::from(value) * scale + 50) / 100;
        out[idx] = scaled.clamp(1, 255) as u8;
    }
    out
}

fn write_marker(out: &mut Vec<u8>, marker: u8) {
    out.push(0xFF);
    out.push(marker);
}

fn write_segment(
    out: &mut Vec<u8>,
    marker: u8,
    payload: &[u8],
    name: &'static str,
) -> Result<(), JpegEncodeError> {
    let len = payload
        .len()
        .checked_add(2)
        .and_then(|value| u16::try_from(value).ok())
        .ok_or(JpegEncodeError::SegmentTooLarge { name })?;
    write_marker(out, marker);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(payload);
    Ok(())
}

fn write_dqt(out: &mut Vec<u8>, table_id: u8, quant: &[u8; 64]) -> Result<(), JpegEncodeError> {
    let mut payload = Vec::with_capacity(65);
    payload.push(table_id);
    for &natural_idx in &ZIGZAG {
        payload.push(quant[natural_idx as usize]);
    }
    write_segment(out, 0xDB, &payload, "DQT")
}

fn write_dri(out: &mut Vec<u8>, restart_interval: u16) -> Result<(), JpegEncodeError> {
    write_segment(out, 0xDD, &restart_interval.to_be_bytes(), "DRI")
}

fn write_sof0(
    out: &mut Vec<u8>,
    width: u32,
    height: u32,
    sampling: Sampling,
) -> Result<(), JpegEncodeError> {
    let mut payload = Vec::with_capacity(6 + sampling.components as usize * 3);
    payload.push(8);
    payload.extend_from_slice(&(height as u16).to_be_bytes());
    payload.extend_from_slice(&(width as u16).to_be_bytes());
    payload.push(sampling.components);
    for component in 0..sampling.components as usize {
        payload.push((component + 1) as u8);
        payload.push((sampling.h[component] << 4) | sampling.v[component]);
        payload.push(u8::from(component != 0));
    }
    write_segment(out, 0xC0, &payload, "SOF0")
}

fn write_dht(
    out: &mut Vec<u8>,
    class: u8,
    table_id: u8,
    bits: &[u8; 16],
    values: &[u8],
) -> Result<(), JpegEncodeError> {
    let mut payload = Vec::with_capacity(17 + values.len());
    payload.push((class << 4) | table_id);
    payload.extend_from_slice(bits);
    payload.extend_from_slice(values);
    write_segment(out, 0xC4, &payload, "DHT")
}

fn write_sos(out: &mut Vec<u8>, components: u8) -> Result<(), JpegEncodeError> {
    let mut payload = Vec::with_capacity(4 + components as usize * 2);
    payload.push(components);
    for component in 0..components {
        payload.push(component + 1);
        payload.push(if component == 0 { 0x00 } else { 0x11 });
    }
    payload.push(0);
    payload.push(63);
    payload.push(0);
    write_segment(out, 0xDA, &payload, "SOS")
}

const STD_LUMA_Q: [u8; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

const STD_CHROMA_Q: [u8; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

const STD_LUMA_DC_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const STD_LUMA_DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const STD_CHROMA_DC_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const STD_CHROMA_DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

const STD_LUMA_AC_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D];
const STD_LUMA_AC_VALUES: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
];

const STD_CHROMA_AC_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
const STD_CHROMA_AC_VALUES: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
];
