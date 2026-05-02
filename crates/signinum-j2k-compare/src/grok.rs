// SPDX-License-Identifier: Apache-2.0

use signinum_core::Rect;
#[cfg(have_grok)]
use std::{ffi::c_void, ptr};

pub fn is_available() -> bool {
    cfg!(have_grok)
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
    channels: u32,
    reduce: Option<u32>,
    region: Option<Rect>,
) -> Result<Vec<u8>, String> {
    #[cfg(have_grok)]
    unsafe {
        let mut out = ptr::null_mut();
        let mut out_len = 0_usize;
        let mut out_width = 0_u32;
        let mut out_height = 0_u32;
        let (has_region, x0, y0, x1, y1) = match region {
            Some(roi) => (1, roi.x, roi.y, roi.x + roi.w, roi.y + roi.h),
            None => (0, 0, 0, 0, 0),
        };
        let ok = signinum_grok_decode_u8(
            bytes.as_ptr(),
            bytes.len(),
            reduce.unwrap_or(0),
            has_region,
            x0,
            y0,
            x1,
            y1,
            channels,
            &raw mut out,
            &raw mut out_len,
            &raw mut out_width,
            &raw mut out_height,
        );
        if ok == 0 || out.is_null() {
            return Err("grok: decode failed".to_string());
        }
        let packed = std::slice::from_raw_parts(out, out_len).to_vec();
        signinum_grok_free(out.cast());
        let expected = out_width as usize * out_height as usize * channels as usize;
        if packed.len() != expected {
            return Err(format!(
                "grok: unexpected output length {} != {}",
                packed.len(),
                expected
            ));
        }
        Ok(packed)
    }

    #[cfg(not(have_grok))]
    {
        let _ = (bytes, channels, reduce, region);
        Err("grok: local library not available".to_string())
    }
}

#[cfg(have_grok)]
unsafe extern "C" {
    fn signinum_grok_decode_u8(
        bytes: *const u8,
        len: usize,
        reduce: u32,
        has_region: i32,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
        channels: u32,
        out_data: *mut *mut u8,
        out_len: *mut usize,
        out_width: *mut u32,
        out_height: *mut u32,
    ) -> i32;
    fn signinum_grok_free(ptr: *mut c_void);
}
