// SPDX-License-Identifier: Apache-2.0

use openjpeg_sys::{
    opj_codec_set_threads, opj_create_decompress, opj_dparameters_t, opj_decode,
    opj_destroy_codec, opj_end_decompress, opj_image, opj_image_destroy, opj_image_t,
    opj_read_header, opj_set_decode_area, opj_set_decoded_resolution_factor,
    opj_set_default_decoder_parameters, opj_setup_decoder, opj_stream_create,
    opj_stream_destroy, opj_stream_set_read_function, opj_stream_set_seek_function,
    opj_stream_set_skip_function, opj_stream_set_user_data, opj_stream_set_user_data_length,
    opj_stream_t, OPJ_BOOL, OPJ_CODEC_FORMAT, OPJ_FALSE, OPJ_OFF_T, OPJ_SIZE_T,
    OPJ_STREAM_READ, OPJ_TRUE,
};
use slidecodec_core::Rect;
use std::{ffi::c_void, ptr, slice};

pub fn is_available() -> bool {
    true
}

pub fn decode_rgb(bytes: &[u8]) -> Result<Vec<u8>, String> {
    decode(bytes, 3, None, None)
}

pub fn decode_gray(bytes: &[u8]) -> Result<Vec<u8>, String> {
    decode(bytes, 1, None, None)
}

pub fn decode_rgb_region(bytes: &[u8], roi: Rect) -> Result<Vec<u8>, String> {
    decode(bytes, 3, None, Some(roi))
}

pub fn decode_gray_region(bytes: &[u8], roi: Rect) -> Result<Vec<u8>, String> {
    decode(bytes, 1, None, Some(roi))
}

pub fn decode_rgb_scaled(bytes: &[u8], reduce: u32) -> Result<Vec<u8>, String> {
    decode(bytes, 3, Some(reduce), None)
}

pub fn decode_gray_scaled(bytes: &[u8], reduce: u32) -> Result<Vec<u8>, String> {
    decode(bytes, 1, Some(reduce), None)
}

fn decode(
    bytes: &[u8],
    channels: usize,
    reduce: Option<u32>,
    region: Option<Rect>,
) -> Result<Vec<u8>, String> {
    let mut image = ptr::null_mut();
    let codec_format = codec_format(bytes)?;
    let stream = create_stream(bytes)?;
    let codec = create_codec(codec_format)?;
    let result = unsafe {
        if opj_read_header(stream, codec, &raw mut image) == bool_false() {
            Err("openjpeg: failed to read header".to_string())
        } else {
            if let Some(reduce) = reduce {
                if opj_set_decoded_resolution_factor(codec, reduce) == bool_false() {
                    return Err("openjpeg: failed to set reduction factor".to_string());
                }
            }
            if let Some(roi) = region {
                if opj_set_decode_area(
                    codec,
                    image,
                    roi.x as i32,
                    roi.y as i32,
                    (roi.x + roi.w) as i32,
                    (roi.y + roi.h) as i32,
                ) == bool_false()
                {
                    return Err("openjpeg: failed to set decode area".to_string());
                }
            }
            if opj_decode(codec, stream, image) == bool_false() {
                Err("openjpeg: decode failed".to_string())
            } else {
                let packed = pack_image(image, channels)?;
                if opj_end_decompress(codec, stream) == bool_false() {
                    Err("openjpeg: end_decompress failed".to_string())
                } else {
                    Ok(packed)
                }
            }
        }
    };
    unsafe {
        if !image.is_null() {
            opj_image_destroy(image);
        }
        opj_destroy_codec(codec);
        opj_stream_destroy(stream);
    }
    result
}

fn codec_format(bytes: &[u8]) -> Result<OPJ_CODEC_FORMAT, String> {
    if bytes.starts_with(&[0, 0, 0, 12, b'j', b'P', b' ', b' ']) {
        Ok(OPJ_CODEC_FORMAT::OPJ_CODEC_JP2)
    } else if bytes.starts_with(&[0xFF, 0x4F]) {
        Ok(OPJ_CODEC_FORMAT::OPJ_CODEC_J2K)
    } else {
        Err("openjpeg: unsupported container, expected JP2 or raw J2K".to_string())
    }
}

fn create_stream(bytes: &[u8]) -> Result<*mut opj_stream_t, String> {
    let stream = unsafe { opj_stream_create(64 * 1024, OPJ_STREAM_READ as OPJ_BOOL) };
    if stream.is_null() {
        return Err("openjpeg: failed to create stream".to_string());
    }
    let user = Box::into_raw(Box::new(MemoryStream::new(bytes)));
    unsafe {
        opj_stream_set_user_data(stream, user.cast(), Some(drop_memory_stream));
        opj_stream_set_user_data_length(stream, bytes.len() as u64);
        opj_stream_set_read_function(stream, Some(read_memory));
        opj_stream_set_skip_function(stream, Some(skip_memory));
        opj_stream_set_seek_function(stream, Some(seek_memory));
    }
    Ok(stream)
}

fn create_codec(codec_format: OPJ_CODEC_FORMAT) -> Result<*mut openjpeg_sys::opj_codec_t, String> {
    let mut params = unsafe { std::mem::zeroed::<opj_dparameters_t>() };
    unsafe { opj_set_default_decoder_parameters(&raw mut params) };
    let codec = unsafe { opj_create_decompress(codec_format) };
    if codec.is_null() {
        return Err("openjpeg: failed to create codec".to_string());
    }
    let setup_ok = unsafe { opj_setup_decoder(codec, &raw mut params) };
    if setup_ok == bool_false() {
        unsafe { opj_destroy_codec(codec) };
        return Err("openjpeg: setup_decoder failed".to_string());
    }
    let threading_ok = unsafe { opj_codec_set_threads(codec, 1) };
    if threading_ok == bool_false() {
        unsafe { opj_destroy_codec(codec) };
        return Err("openjpeg: codec_set_threads failed".to_string());
    }
    Ok(codec)
}

fn pack_image(image: *mut opj_image_t, channels: usize) -> Result<Vec<u8>, String> {
    if image.is_null() {
        return Err("openjpeg: null image".to_string());
    }
    let image_ref = unsafe { &*image };
    if image_ref.numcomps == 0 || image_ref.comps.is_null() {
        return Err("openjpeg: image has no components".to_string());
    }
    let comp0 = unsafe { &*image_ref.comps };
    let width = comp0.w as usize;
    let height = comp0.h as usize;
    let mut out = vec![0_u8; width * height * channels];
    for row in 0..height {
        for col in 0..width {
            let dst = (row * width + col) * channels;
            let sample0 = read_component(image_ref, 0, row, col)?;
            if channels == 1 {
                out[dst] = sample0;
                continue;
            }
            if image_ref.numcomps == 1 {
                out[dst] = sample0;
                out[dst + 1] = sample0;
                out[dst + 2] = sample0;
                continue;
            }
            out[dst] = sample0;
            out[dst + 1] = read_component(image_ref, 1, row, col)?;
            out[dst + 2] = read_component(image_ref, 2, row, col)?;
        }
    }
    Ok(out)
}

fn read_component(image: &opj_image, index: usize, row: usize, col: usize) -> Result<u8, String> {
    let comp = unsafe {
        image
            .comps
            .add(index)
            .as_ref()
            .ok_or_else(|| "openjpeg: component missing".to_string())?
    };
    if comp.data.is_null() {
        return Err("openjpeg: component data missing".to_string());
    }
    let stride = comp.w as usize;
    let data = unsafe { slice::from_raw_parts(comp.data, stride * comp.h as usize) };
    let value = data[row * stride + col];
    Ok(scale_to_u8(value, comp.prec, comp.sgnd != 0))
}

fn scale_to_u8(value: i32, precision: u32, signed: bool) -> u8 {
    let adjusted = if signed {
        value.saturating_add(1_i32 << precision.saturating_sub(1))
    } else {
        value
    };
    if precision <= 8 {
        adjusted.clamp(0, 255) as u8
    } else {
        let max = i64::from((1_u32 << precision.min(31)) - 1);
        let scaled = (i64::from(adjusted.max(0)) * 255 + max / 2) / max.max(1);
        scaled.clamp(0, 255) as u8
    }
}

struct MemoryStream {
    ptr: *const u8,
    len: usize,
    offset: usize,
}

impl MemoryStream {
    fn new(bytes: &[u8]) -> Self {
        Self {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
            offset: 0,
        }
    }
}

unsafe extern "C" fn read_memory(
    buffer: *mut c_void,
    bytes: OPJ_SIZE_T,
    user_data: *mut c_void,
) -> OPJ_SIZE_T {
    let state = &mut *user_data.cast::<MemoryStream>();
    let remaining = state.len.saturating_sub(state.offset);
    if remaining == 0 {
        return usize::MAX;
    }
    let count = remaining.min(bytes);
    ptr::copy_nonoverlapping(state.ptr.add(state.offset), buffer.cast::<u8>(), count);
    state.offset += count;
    count
}

unsafe extern "C" fn skip_memory(bytes: OPJ_OFF_T, user_data: *mut c_void) -> OPJ_OFF_T {
    let state = &mut *user_data.cast::<MemoryStream>();
    if bytes < 0 {
        return -1;
    }
    let next = state.offset.saturating_add(bytes as usize);
    if next > state.len {
        return -1;
    }
    state.offset = next;
    bytes
}

unsafe extern "C" fn seek_memory(bytes: OPJ_OFF_T, user_data: *mut c_void) -> OPJ_BOOL {
    let state = &mut *user_data.cast::<MemoryStream>();
    if bytes < 0 || bytes as usize > state.len {
        return bool_false();
    }
    state.offset = bytes as usize;
    bool_true()
}

unsafe extern "C" fn drop_memory_stream(user_data: *mut c_void) {
    drop(Box::from_raw(user_data.cast::<MemoryStream>()));
}

const fn bool_false() -> OPJ_BOOL {
    OPJ_FALSE as OPJ_BOOL
}

const fn bool_true() -> OPJ_BOOL {
    OPJ_TRUE as OPJ_BOOL
}
