use ashlar_core::{Downscale, Rect};
use ashlar_j2k::adapter::device_plan::{DeviceDecodePlan, DeviceDecodeRequest};

#[test]
fn full_request_plan_uses_full_source_rect() {
    let plan = DeviceDecodePlan::for_image((320, 200), DeviceDecodeRequest::Full)
        .expect("full request plan");

    assert_eq!(plan.source_rect(), Rect::full((320, 200)));
    assert_eq!(plan.output_rect(), Rect::full((320, 200)));
    assert_eq!(plan.output_dims(), (320, 200));
    assert_eq!(plan.target_resolution(), None);
    assert!(plan.is_full_frame());
}

#[test]
fn scaled_request_plan_reports_target_resolution() {
    let plan = DeviceDecodePlan::for_image(
        (320, 200),
        DeviceDecodeRequest::Scaled {
            scale: Downscale::Quarter,
        },
    )
    .expect("scaled request plan");

    assert_eq!(plan.output_rect(), Rect::full((80, 50)));
    assert_eq!(plan.output_dims(), (80, 50));
    assert_eq!(plan.target_resolution(), Some((80, 50)));
}

#[test]
fn region_scaled_request_plan_uses_covering_scaled_rect() {
    let roi = Rect {
        x: 7,
        y: 9,
        w: 11,
        h: 13,
    };
    let plan = DeviceDecodePlan::for_image(
        (64, 64),
        DeviceDecodeRequest::RegionScaled {
            roi,
            scale: Downscale::Half,
        },
    )
    .expect("region scaled request plan");

    assert_eq!(plan.source_rect(), roi);
    assert_eq!(
        plan.output_rect(),
        Rect {
            x: 3,
            y: 4,
            w: 6,
            h: 7,
        }
    );
    assert_eq!(plan.output_dims(), (6, 7));
    assert_eq!(plan.target_resolution(), Some((6, 7)));
    assert!(!plan.is_full_frame());
}

#[test]
fn invalid_region_is_rejected() {
    let roi = Rect {
        x: 60,
        y: 60,
        w: 8,
        h: 8,
    };
    let result = DeviceDecodePlan::for_image((64, 64), DeviceDecodeRequest::Region { roi });

    assert!(result.is_err(), "out-of-bounds ROI must be rejected");
}
