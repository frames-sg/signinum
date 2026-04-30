// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

use ashlar_jpeg::{Downscale, Rect};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InspectInfo {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) subsamp: i32,
}

#[cfg(has_libjpeg_turbo)]
mod imp {
    use super::{Downscale, InspectInfo, Rect};
    use std::ffi::{c_char, c_int, c_void, CStr};
    use std::ptr::NonNull;

    const TJINIT_DECOMPRESS: c_int = 1;
    const TJPF_RGB: c_int = 0;
    const TJPF_GRAY: c_int = 6;
    const TJPARAM_SUBSAMP: c_int = 4;
    const TJPARAM_JPEGWIDTH: c_int = 5;
    const TJPARAM_JPEGHEIGHT: c_int = 6;
    const TJUNSCALED: TjScalingFactor = TjScalingFactor { num: 1, denom: 1 };
    const TJMCU_WIDTH: [u32; 7] = [8, 16, 16, 8, 8, 32, 8];
    const TJUNCROPPED: TjRegion = TjRegion {
        x: 0,
        y: 0,
        w: 0,
        h: 0,
    };

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct TjScalingFactor {
        num: c_int,
        denom: c_int,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct TjRegion {
        x: c_int,
        y: c_int,
        w: c_int,
        h: c_int,
    }

    type TjHandle = *mut c_void;

    unsafe extern "C" {
        fn tj3Init(init_type: c_int) -> TjHandle;
        fn tj3Destroy(handle: TjHandle);
        fn tj3GetErrorStr(handle: TjHandle) -> *mut c_char;
        fn tj3Get(handle: TjHandle, param: c_int) -> c_int;
        fn tj3DecompressHeader(handle: TjHandle, jpeg_buf: *const u8, jpeg_size: usize) -> c_int;
        fn tj3SetScalingFactor(handle: TjHandle, scaling_factor: TjScalingFactor) -> c_int;
        fn tj3SetCroppingRegion(handle: TjHandle, cropping_region: TjRegion) -> c_int;
        fn tj3Decompress8(
            handle: TjHandle,
            jpeg_buf: *const u8,
            jpeg_size: usize,
            dst_buf: *mut u8,
            pitch: c_int,
            pixel_format: c_int,
        ) -> c_int;
    }

    pub(crate) struct TurboJpegDecoder {
        handle: NonNull<c_void>,
    }

    impl TurboJpegDecoder {
        pub(crate) fn new() -> Result<Self, String> {
            let handle = unsafe { tj3Init(TJINIT_DECOMPRESS) };
            let Some(handle) = NonNull::new(handle) else {
                return Err("tj3Init returned null".to_string());
            };
            Ok(Self { handle })
        }

        pub(crate) fn inspect(&mut self, bytes: &[u8]) -> Result<InspectInfo, String> {
            self.read_header(bytes)
        }

        pub(crate) fn decode_rgb(&mut self, bytes: &[u8]) -> Result<Vec<u8>, String> {
            self.decode(bytes, TJPF_RGB, None, Downscale::None)
        }

        pub(crate) fn decode_gray(&mut self, bytes: &[u8]) -> Result<Vec<u8>, String> {
            self.decode(bytes, TJPF_GRAY, None, Downscale::None)
        }

        pub(crate) fn decode_scaled_rgb(
            &mut self,
            bytes: &[u8],
            factor: Downscale,
        ) -> Result<Vec<u8>, String> {
            self.decode(bytes, TJPF_RGB, None, factor)
        }

        pub(crate) fn decode_region_rgb(
            &mut self,
            bytes: &[u8],
            roi: Rect,
        ) -> Result<Vec<u8>, String> {
            self.decode(bytes, TJPF_RGB, Some(roi), Downscale::None)
        }

        pub(crate) fn decode_region_scaled_rgb(
            &mut self,
            bytes: &[u8],
            roi: Rect,
            factor: Downscale,
        ) -> Result<Vec<u8>, String> {
            self.decode(bytes, TJPF_RGB, Some(roi), factor)
        }

        fn decode(
            &mut self,
            bytes: &[u8],
            pixel_format: c_int,
            roi: Option<Rect>,
            factor: Downscale,
        ) -> Result<Vec<u8>, String> {
            let header = self.read_header(bytes)?;
            let scale = scaling_factor(factor);
            self.set_scaling(scale)?;
            let bytes_per_pixel = bytes_per_pixel(pixel_format);

            if let Some(roi) = roi {
                let scaled_roi = scaled_rect(roi, factor);
                let scaled_mcu = scaled_mcu_width(header.subsamp, scale);
                let aligned_x = scaled_roi.x - scaled_roi.x % scaled_mcu;
                let trim_left = scaled_roi.x - aligned_x;
                let crop_width = trim_left + scaled_roi.w;
                self.set_crop(TjRegion {
                    x: to_c_int(aligned_x)?,
                    y: to_c_int(scaled_roi.y)?,
                    w: to_c_int(crop_width)?,
                    h: to_c_int(scaled_roi.h)?,
                })?;

                let pitch = crop_width as usize * bytes_per_pixel;
                let mut out = vec![0u8; pitch * scaled_roi.h as usize];
                self.decompress(bytes, &mut out, pitch, pixel_format)?;
                if trim_left == 0 {
                    return Ok(out);
                }
                return Ok(trim_packed_rows(
                    &out,
                    crop_width as usize,
                    scaled_roi.w as usize,
                    scaled_roi.h as usize,
                    trim_left as usize,
                    bytes_per_pixel,
                ));
            }

            self.set_crop(TJUNCROPPED)?;
            let out_width = scaled_dimension(header.width, scale);
            let out_height = scaled_dimension(header.height, scale);
            let pitch = out_width as usize * bytes_per_pixel;
            let mut out = vec![0u8; pitch * out_height as usize];
            self.decompress(bytes, &mut out, pitch, pixel_format)?;
            Ok(out)
        }

        fn read_header(&mut self, bytes: &[u8]) -> Result<InspectInfo, String> {
            let rc =
                unsafe { tj3DecompressHeader(self.handle.as_ptr(), bytes.as_ptr(), bytes.len()) };
            if rc != 0 {
                return Err(self.error_string());
            }

            let width = unsafe { tj3Get(self.handle.as_ptr(), TJPARAM_JPEGWIDTH) };
            let height = unsafe { tj3Get(self.handle.as_ptr(), TJPARAM_JPEGHEIGHT) };
            let subsamp = unsafe { tj3Get(self.handle.as_ptr(), TJPARAM_SUBSAMP) };
            if width < 0 || height < 0 || subsamp < 0 {
                return Err("tj3Get returned incomplete header parameters".to_string());
            }

            Ok(InspectInfo {
                width: width as u32,
                height: height as u32,
                subsamp,
            })
        }

        fn set_scaling(&mut self, scale: TjScalingFactor) -> Result<(), String> {
            let rc = unsafe { tj3SetScalingFactor(self.handle.as_ptr(), scale) };
            if rc != 0 {
                return Err(self.error_string());
            }
            Ok(())
        }

        fn set_crop(&mut self, region: TjRegion) -> Result<(), String> {
            let rc = unsafe { tj3SetCroppingRegion(self.handle.as_ptr(), region) };
            if rc != 0 {
                return Err(self.error_string());
            }
            Ok(())
        }

        fn decompress(
            &mut self,
            bytes: &[u8],
            out: &mut [u8],
            pitch: usize,
            pixel_format: c_int,
        ) -> Result<(), String> {
            let rc = unsafe {
                tj3Decompress8(
                    self.handle.as_ptr(),
                    bytes.as_ptr(),
                    bytes.len(),
                    out.as_mut_ptr(),
                    to_c_int(pitch as u32)?,
                    pixel_format,
                )
            };
            if rc != 0 {
                return Err(self.error_string());
            }
            Ok(())
        }

        fn error_string(&self) -> String {
            let ptr = unsafe { tj3GetErrorStr(self.handle.as_ptr()) };
            if ptr.is_null() {
                return "libjpeg-turbo error".to_string();
            }
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        }
    }

    impl Drop for TurboJpegDecoder {
        fn drop(&mut self) {
            unsafe { tj3Destroy(self.handle.as_ptr()) };
        }
    }

    pub(crate) fn is_available() -> bool {
        true
    }

    pub(crate) fn decode_rgb(bytes: &[u8]) -> Result<Vec<u8>, String> {
        let mut decoder = TurboJpegDecoder::new()?;
        decoder.decode_rgb(bytes)
    }

    pub(crate) fn decode_scaled_rgb(bytes: &[u8], factor: Downscale) -> Result<Vec<u8>, String> {
        let mut decoder = TurboJpegDecoder::new()?;
        decoder.decode_scaled_rgb(bytes, factor)
    }

    pub(crate) fn decode_region_rgb(bytes: &[u8], roi: Rect) -> Result<Vec<u8>, String> {
        let mut decoder = TurboJpegDecoder::new()?;
        decoder.decode_region_rgb(bytes, roi)
    }

    fn scaling_factor(factor: Downscale) -> TjScalingFactor {
        match factor {
            Downscale::None => TJUNSCALED,
            Downscale::Half => TjScalingFactor { num: 1, denom: 2 },
            Downscale::Quarter => TjScalingFactor { num: 1, denom: 4 },
            Downscale::Eighth => TjScalingFactor { num: 1, denom: 8 },
            _ => unreachable!("unsupported Downscale variant"),
        }
    }

    fn bytes_per_pixel(pixel_format: c_int) -> usize {
        match pixel_format {
            TJPF_RGB => 3,
            TJPF_GRAY => 1,
            _ => unreachable!("unsupported pixel format"),
        }
    }

    fn scaled_dimension(dimension: u32, scale: TjScalingFactor) -> u32 {
        (dimension * scale.num as u32).div_ceil(scale.denom as u32)
    }

    fn scaled_rect(rect: Rect, factor: Downscale) -> Rect {
        let denom = match factor {
            Downscale::None => 1,
            Downscale::Half => 2,
            Downscale::Quarter => 4,
            Downscale::Eighth => 8,
            _ => unreachable!("unsupported Downscale variant"),
        };
        let x_end = rect.x + rect.w;
        let y_end = rect.y + rect.h;
        Rect {
            x: rect.x / denom,
            y: rect.y / denom,
            w: x_end.div_ceil(denom) - rect.x / denom,
            h: y_end.div_ceil(denom) - rect.y / denom,
        }
    }

    fn scaled_mcu_width(subsamp: i32, scale: TjScalingFactor) -> u32 {
        let mcu = TJMCU_WIDTH.get(subsamp as usize).copied().unwrap_or(8);
        scaled_dimension(mcu, scale).max(1)
    }

    fn trim_packed_rows(
        full: &[u8],
        source_width: usize,
        target_width: usize,
        height: usize,
        trim_left: usize,
        bytes_per_pixel: usize,
    ) -> Vec<u8> {
        let source_stride = source_width * bytes_per_pixel;
        let target_stride = target_width * bytes_per_pixel;
        let mut out = vec![0u8; target_stride * height];
        for row in 0..height {
            let src_start = row * source_stride + trim_left * bytes_per_pixel;
            let src_end = src_start + target_stride;
            let dst_start = row * target_stride;
            out[dst_start..dst_start + target_stride].copy_from_slice(&full[src_start..src_end]);
        }
        out
    }

    fn to_c_int(value: u32) -> Result<c_int, String> {
        c_int::try_from(value).map_err(|_| format!("value {value} does not fit into c_int"))
    }
}

#[cfg(not(has_libjpeg_turbo))]
mod imp {
    use super::{Downscale, InspectInfo, Rect};

    pub(crate) struct TurboJpegDecoder;

    impl TurboJpegDecoder {
        pub(crate) fn new() -> Result<Self, String> {
            Err("libjpeg-turbo not available".to_string())
        }

        pub(crate) fn inspect(&mut self, _bytes: &[u8]) -> Result<InspectInfo, String> {
            let _ = self;
            Err("libjpeg-turbo not available".to_string())
        }

        pub(crate) fn decode_rgb(&mut self, _bytes: &[u8]) -> Result<Vec<u8>, String> {
            let _ = self;
            Err("libjpeg-turbo not available".to_string())
        }

        pub(crate) fn decode_gray(&mut self, _bytes: &[u8]) -> Result<Vec<u8>, String> {
            let _ = self;
            Err("libjpeg-turbo not available".to_string())
        }

        pub(crate) fn decode_scaled_rgb(
            &mut self,
            _bytes: &[u8],
            _factor: Downscale,
        ) -> Result<Vec<u8>, String> {
            let _ = self;
            Err("libjpeg-turbo not available".to_string())
        }

        pub(crate) fn decode_region_rgb(
            &mut self,
            _bytes: &[u8],
            _roi: Rect,
        ) -> Result<Vec<u8>, String> {
            let _ = self;
            Err("libjpeg-turbo not available".to_string())
        }

        pub(crate) fn decode_region_scaled_rgb(
            &mut self,
            _bytes: &[u8],
            _roi: Rect,
            _factor: Downscale,
        ) -> Result<Vec<u8>, String> {
            let _ = self;
            Err("libjpeg-turbo not available".to_string())
        }
    }

    pub(crate) fn is_available() -> bool {
        false
    }

    pub(crate) fn decode_rgb(_bytes: &[u8]) -> Result<Vec<u8>, String> {
        Err("libjpeg-turbo not available".to_string())
    }

    pub(crate) fn decode_scaled_rgb(_bytes: &[u8], _factor: Downscale) -> Result<Vec<u8>, String> {
        Err("libjpeg-turbo not available".to_string())
    }

    pub(crate) fn decode_region_rgb(_bytes: &[u8], _roi: Rect) -> Result<Vec<u8>, String> {
        Err("libjpeg-turbo not available".to_string())
    }
}

#[allow(unused_imports)]
pub(crate) use imp::{
    decode_region_rgb, decode_rgb, decode_scaled_rgb, is_available, TurboJpegDecoder,
};
