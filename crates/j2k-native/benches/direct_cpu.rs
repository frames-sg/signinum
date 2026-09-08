use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(feature = "parallel")]
use j2k_native::encode;
use j2k_native::{
    encode_htj2k, execute_direct_color_plan_rgb8_into, execute_direct_color_plan_rgba8_into,
    CpuDecodeParallelism, DecodeSettings, DecoderContext, EncodeOptions, Image,
    J2kDirectCpuScratch, J2kRect,
};
#[cfg(feature = "parallel")]
use rayon::ThreadPoolBuilder;

const TILE_SIDE: u32 = 512;

fn patterned_rgb8(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 3 + y * 5) & 0xff) as u8);
            pixels.push(((x * 7 + y * 11 + 17) & 0xff) as u8);
            pixels.push(((x * 13 + y * 19 + 31) & 0xff) as u8);
        }
    }
    pixels
}

fn patterned_gray8(width: u32, height: u32) -> Vec<u8> {
    patterned_gray8_with_seed(width, height, 0)
}

fn patterned_gray8_with_seed(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 17 + y * 31 + (x ^ y) * seed + seed * 13) & 0xff) as u8);
        }
    }
    pixels
}

#[cfg(feature = "parallel")]
fn gray53_codestream(
    width: u32,
    height: u32,
    seed: u32,
    use_ht_block_coding: bool,
) -> (Vec<u8>, Vec<u8>) {
    let pixels = patterned_gray8_with_seed(width, height, seed);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 5,
        use_ht_block_coding,
        ..EncodeOptions::default()
    };
    let codestream = encode(&pixels, width, height, 1, 8, false, &options)
        .expect("encode reversible gray 5/3 codestream");
    (codestream, pixels)
}

#[cfg(feature = "parallel")]
fn validate_gray_decode(image: &Image<'_>, expected: &[u8], dimensions: (u32, u32)) {
    let decoded = image
        .decode_with_context(&mut DecoderContext::default())
        .expect("validate reversible gray 5/3 decode");
    assert_eq!((decoded.width, decoded.height), dimensions);
    assert_eq!(decoded.data, expected);
}

fn htj2k_gray_codestream(width: u32, height: u32, reversible: bool) -> Vec<u8> {
    let pixels = patterned_gray8(width, height);
    let options = EncodeOptions {
        reversible,
        guard_bits: if reversible { 1 } else { 2 },
        num_decomposition_levels: 5,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, width, height, 1, 8, false, &options).expect("encode HTJ2K gray")
}

fn htj2k_rgb_codestream(width: u32, height: u32) -> Vec<u8> {
    let pixels = patterned_rgb8(width, height);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, width, height, 3, 8, false, &options).expect("encode HTJ2K RGB")
}

fn htj2k_rgb97_codestream(width: u32, height: u32) -> Vec<u8> {
    let pixels = patterned_rgb8(width, height);
    let options = EncodeOptions {
        reversible: false,
        guard_bits: 2,
        num_decomposition_levels: 5,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, width, height, 3, 8, false, &options).expect("encode HTJ2K 9/7 RGB")
}

fn direct_roi_plan(bytes: &[u8]) -> (j2k_native::J2kDirectColorPlan, J2kRect) {
    let image = Image::new(
        bytes,
        &DecodeSettings {
            target_resolution: Some((TILE_SIDE / 4, TILE_SIDE / 4)),
            ..DecodeSettings::default()
        },
    )
    .expect("scaled HTJ2K image");
    let output_region = J2kRect {
        x0: 32,
        y0: 32,
        x1: 96,
        y1: 96,
    };
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_color_plan_region_with_context(
            &mut context,
            (
                output_region.x0,
                output_region.y0,
                output_region.width(),
                output_region.height(),
            ),
        )
        .expect("direct RGB region plan");
    (plan, output_region)
}

fn bench_direct_color_plan_97(c: &mut Criterion) {
    let codestream = htj2k_rgb97_codestream(TILE_SIDE, TILE_SIDE);
    let (plan, output_region) = direct_roi_plan(&codestream);
    let rgb_stride = output_region.width() as usize * 3;
    let rgb_len = rgb_stride * output_region.height() as usize;

    let mut group = c.benchmark_group("j2k_native_direct_cpu_color_plan");
    group.bench_function("htj2k_rgb8_97_roi256_q4_reuse_scratch", |b| {
        let mut scratch = J2kDirectCpuScratch::new();
        let mut out = vec![0_u8; rgb_len];
        b.iter(|| {
            execute_direct_color_plan_rgb8_into(
                std::hint::black_box(&plan),
                output_region,
                &mut scratch,
                &mut out,
                rgb_stride,
            )
            .expect("execute RGB direct 9/7 plan");
            std::hint::black_box(&out);
        });
    });
    group.finish();
}

fn bench_full_decode(c: &mut Criterion) {
    let gray_codestream = htj2k_gray_codestream(TILE_SIDE, TILE_SIDE, false);
    let gray_image =
        Image::new(&gray_codestream, &DecodeSettings::default()).expect("HTJ2K 9/7 gray image");
    let rgb_codestream = htj2k_rgb97_codestream(TILE_SIDE, TILE_SIDE);
    let rgb_image =
        Image::new(&rgb_codestream, &DecodeSettings::default()).expect("HTJ2K 9/7 RGB image");

    let mut group = c.benchmark_group("j2k_native_full_decode");
    group.bench_function("htj2k_gray8_512x512_97_reuse_context", |b| {
        let mut context = DecoderContext::default();
        b.iter(|| {
            let decoded = gray_image
                .decode_with_context(&mut context)
                .expect("decode HTJ2K 9/7 gray");
            std::hint::black_box(decoded.data.len());
        });
    });
    group.bench_function("htj2k_rgb8_512x512_97_reuse_context", |b| {
        let mut context = DecoderContext::default();
        b.iter(|| {
            let decoded = rgb_image
                .decode_with_context(&mut context)
                .expect("decode HTJ2K 9/7 RGB");
            std::hint::black_box(decoded.data.len());
        });
    });
    group.bench_function("htj2k_rgb8_512x512_97_serial_context", |b| {
        let mut context = DecoderContext::default();
        context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
        b.iter(|| {
            let decoded = rgb_image
                .decode_with_context(&mut context)
                .expect("decode HTJ2K 9/7 RGB");
            std::hint::black_box(decoded.data.len());
        });
    });
    group.finish();
}

#[cfg(feature = "parallel")]
fn bench_generic_public_workspace(c: &mut Criterion) {
    const SECONDARY_DIMENSIONS: (u32, u32) = (509, 383);
    let mut group = c.benchmark_group("j2k_native_generic_public_workspace");

    for (codec, use_ht_block_coding) in [("classic53", false), ("ht53", true)] {
        let (primary_codestream, primary_pixels) =
            gray53_codestream(TILE_SIDE, TILE_SIDE, 3, use_ht_block_coding);
        let (secondary_codestream, secondary_pixels) = gray53_codestream(
            SECONDARY_DIMENSIONS.0,
            SECONDARY_DIMENSIONS.1,
            11,
            use_ht_block_coding,
        );
        assert_ne!(primary_codestream, secondary_codestream);
        assert_ne!(primary_pixels, secondary_pixels);

        let primary_image = Image::new(&primary_codestream, &DecodeSettings::default())
            .expect("prepare primary reversible gray 5/3 image");
        let secondary_image = Image::new(&secondary_codestream, &DecodeSettings::default())
            .expect("prepare secondary reversible gray 5/3 image");
        validate_gray_decode(&primary_image, &primary_pixels, (TILE_SIDE, TILE_SIDE));
        validate_gray_decode(&secondary_image, &secondary_pixels, SECONDARY_DIMENSIONS);

        group.bench_function(
            format!("{codec}_gray8_512x512_fresh_context_default_pool"),
            |b| {
                b.iter(|| {
                    let mut context = DecoderContext::default();
                    let decoded = primary_image
                        .decode_with_context(&mut context)
                        .expect("decode prepared image with fresh context");
                    std::hint::black_box(decoded.data);
                });
            },
        );
        group.bench_function(
            format!("{codec}_gray8_512x512_reuse_context_default_pool"),
            |b| {
                let mut context = DecoderContext::default();
                let warmed = primary_image
                    .decode_with_context(&mut context)
                    .expect("warm reusable context");
                assert_eq!(warmed.data, primary_pixels);
                drop(warmed);
                b.iter(|| {
                    let decoded = primary_image
                        .decode_with_context(&mut context)
                        .expect("decode prepared image with reused context");
                    std::hint::black_box(decoded.data);
                });
            },
        );
        group.bench_function(
            format!("{codec}_gray8_two_input_512x512_509x383_reuse_context_default_pool"),
            |b| {
                let mut context = DecoderContext::default();
                let first = primary_image
                    .decode_with_context(&mut context)
                    .expect("warm primary image");
                assert_eq!(first.data, primary_pixels);
                drop(first);
                let second = secondary_image
                    .decode_with_context(&mut context)
                    .expect("warm secondary image");
                assert_eq!(second.data, secondary_pixels);
                drop(second);
                let mut use_secondary = false;
                b.iter(|| {
                    use_secondary = !use_secondary;
                    let decoded = if use_secondary {
                        secondary_image.decode_with_context(&mut context)
                    } else {
                        primary_image.decode_with_context(&mut context)
                    }
                    .expect("decode alternating prepared image with reused context");
                    std::hint::black_box(decoded.data);
                });
            },
        );

        for threads in [1usize, 2, 4] {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("build benchmark Rayon pool");
            group.bench_function(
                format!("{codec}_gray8_512x512_reuse_context_pool_threads_{threads}"),
                |b| {
                    let mut context = DecoderContext::default();
                    let warmed = pool
                        .install(|| primary_image.decode_with_context(&mut context))
                        .expect("warm context in benchmark Rayon pool");
                    assert_eq!(warmed.data, primary_pixels);
                    drop(warmed);
                    b.iter(|| {
                        pool.install(|| {
                            let decoded = primary_image
                                .decode_with_context(&mut context)
                                .expect("decode prepared image in benchmark Rayon pool");
                            std::hint::black_box(decoded.data);
                        });
                    });
                },
            );
        }
    }
    group.finish();
}

#[cfg(not(feature = "parallel"))]
fn bench_generic_public_workspace(_c: &mut Criterion) {}

fn bench_direct_color_plan(c: &mut Criterion) {
    let codestream = htj2k_rgb_codestream(TILE_SIDE, TILE_SIDE);
    let (plan, output_region) = direct_roi_plan(&codestream);
    let three_channel_stride = output_region.width() as usize * 3;
    let four_channel_stride = output_region.width() as usize * 4;
    let three_channel_output_len = three_channel_stride * output_region.height() as usize;
    let four_channel_output_len = four_channel_stride * output_region.height() as usize;

    let mut group = c.benchmark_group("j2k_native_direct_cpu_color_plan");
    group.bench_function("htj2k_rgb8_roi256_q4_fresh_scratch", |b| {
        b.iter(|| {
            let mut scratch = J2kDirectCpuScratch::new();
            let mut out = vec![0_u8; three_channel_output_len];
            execute_direct_color_plan_rgb8_into(
                std::hint::black_box(&plan),
                output_region,
                &mut scratch,
                &mut out,
                three_channel_stride,
            )
            .expect("execute RGB direct plan");
            std::hint::black_box(out);
        });
    });
    group.bench_function("htj2k_rgb8_roi256_q4_reuse_scratch", |b| {
        let mut scratch = J2kDirectCpuScratch::new();
        let mut out = vec![0_u8; three_channel_output_len];
        b.iter(|| {
            execute_direct_color_plan_rgb8_into(
                std::hint::black_box(&plan),
                output_region,
                &mut scratch,
                &mut out,
                three_channel_stride,
            )
            .expect("execute RGB direct plan");
            std::hint::black_box(&out);
        });
    });
    group.bench_function("htj2k_rgba8_roi256_q4_reuse_scratch", |b| {
        let mut scratch = J2kDirectCpuScratch::new();
        let mut out = vec![0_u8; four_channel_output_len];
        b.iter(|| {
            execute_direct_color_plan_rgba8_into(
                std::hint::black_box(&plan),
                output_region,
                &mut scratch,
                &mut out,
                four_channel_stride,
            )
            .expect("execute RGBA direct plan");
            std::hint::black_box(&out);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_full_decode,
    bench_generic_public_workspace,
    bench_direct_color_plan,
    bench_direct_color_plan_97
);
criterion_main!(benches);
