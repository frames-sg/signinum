// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) fn dense_gray8(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 17 + y * 31 + (x ^ y) * seed + seed * 13) & 0xff) as u8);
        }
    }
    pixels
}

#[cfg(feature = "parallel")]
pub(crate) fn sparse_gray8(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut pixels = vec![128; width as usize * height as usize];
    if width == 0 || height == 0 {
        return pixels;
    }

    let x_offset = seed.wrapping_mul(17) % width.min(32);
    let y_offset = seed.wrapping_mul(29) % height.min(32);
    let impulse = (seed.wrapping_mul(37) & 0xff) as u8;
    let impulse = if impulse == 128 { 127 } else { impulse };
    for y in (y_offset..height).step_by(32) {
        for x in (x_offset..width).step_by(32) {
            let index = y as usize * width as usize + x as usize;
            pixels[index] = impulse;
        }
    }
    pixels
}
