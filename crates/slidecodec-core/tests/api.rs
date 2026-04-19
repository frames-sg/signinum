use slidecodec_core::{Downscale, PixelFormat, PixelLayout, Rect, SampleType};

#[test]
fn pixel_format_reports_layout_and_sample_type() {
    assert_eq!(PixelFormat::Rgb8.layout(), PixelLayout::Rgb);
    assert_eq!(PixelFormat::Rgb8.sample(), SampleType::U8);
    assert_eq!(PixelFormat::Rgb16.sample(), SampleType::U16);
}

#[test]
fn downscale_reports_expected_denominators() {
    assert_eq!(Downscale::None.denominator(), 1);
    assert_eq!(Downscale::Half.denominator(), 2);
    assert_eq!(Downscale::Quarter.denominator(), 4);
    assert_eq!(Downscale::Eighth.denominator(), 8);
}

#[test]
fn rect_full_and_is_within_match_existing_jpeg_behavior() {
    let full = Rect::full((640, 480));
    assert_eq!(
        full,
        Rect {
            x: 0,
            y: 0,
            w: 640,
            h: 480,
        }
    );
    assert!(Rect {
        x: 10,
        y: 10,
        w: 100,
        h: 100,
    }
    .is_within((640, 480)));
}
