use criterion::{black_box, criterion_group, criterion_main, Criterion};
use signinum_j2k_native::{
    encode_htj2k, execute_direct_color_plan_rgb8_into, execute_direct_color_plan_rgba8_into,
    DecodeSettings, DecoderContext, EncodeOptions, Image, J2kDirectCpuScratch, J2kRect,
};

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

fn htj2k_rgb_codestream(width: u32, height: u32) -> Vec<u8> {
    let pixels = patterned_rgb8(width, height);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, width, height, 3, 8, false, &options).expect("encode HTJ2K RGB")
}

fn direct_roi_plan(bytes: &[u8]) -> (signinum_j2k_native::J2kDirectColorPlan, J2kRect) {
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

fn bench_direct_color_plan(c: &mut Criterion) {
    let codestream = htj2k_rgb_codestream(TILE_SIDE, TILE_SIDE);
    let (plan, output_region) = direct_roi_plan(&codestream);
    let rgb_stride = output_region.width() as usize * 3;
    let rgba_stride = output_region.width() as usize * 4;
    let rgb_len = rgb_stride * output_region.height() as usize;
    let rgba_len = rgba_stride * output_region.height() as usize;

    let mut group = c.benchmark_group("j2k_native_direct_cpu_color_plan");
    group.bench_function("htj2k_rgb8_roi256_q4_fresh_scratch", |b| {
        b.iter(|| {
            let mut scratch = J2kDirectCpuScratch::new();
            let mut out = vec![0_u8; rgb_len];
            execute_direct_color_plan_rgb8_into(
                black_box(&plan),
                output_region,
                &mut scratch,
                &mut out,
                rgb_stride,
            )
            .expect("execute RGB direct plan");
            black_box(out);
        });
    });
    group.bench_function("htj2k_rgb8_roi256_q4_reuse_scratch", |b| {
        let mut scratch = J2kDirectCpuScratch::new();
        let mut out = vec![0_u8; rgb_len];
        b.iter(|| {
            execute_direct_color_plan_rgb8_into(
                black_box(&plan),
                output_region,
                &mut scratch,
                &mut out,
                rgb_stride,
            )
            .expect("execute RGB direct plan");
            black_box(&out);
        });
    });
    group.bench_function("htj2k_rgba8_roi256_q4_reuse_scratch", |b| {
        let mut scratch = J2kDirectCpuScratch::new();
        let mut out = vec![0_u8; rgba_len];
        b.iter(|| {
            execute_direct_color_plan_rgba8_into(
                black_box(&plan),
                output_region,
                &mut scratch,
                &mut out,
                rgba_stride,
            )
            .expect("execute RGBA direct plan");
            black_box(&out);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_direct_color_plan);
criterion_main!(benches);
