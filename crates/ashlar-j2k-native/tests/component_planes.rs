use ashlar_j2k_native::{encode, DecodeSettings, DecoderContext, EncodeOptions, Image};

fn fixture() -> Vec<u8> {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 2, 3, 8, false, &options).expect("encode")
}

#[test]
fn decoded_components_expose_component_planes() {
    let bytes = fixture();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let bitmap = image
        .decode_with_context(&mut DecoderContext::default())
        .expect("bitmap");
    let planes = image
        .decode_components_with_context(&mut context)
        .expect("component decode");

    assert_eq!(planes.dimensions(), (2, 2));
    assert_eq!(planes.planes().len(), 3);
    assert_eq!(planes.planes()[0].bit_depth(), 8);
    assert!(planes
        .planes()
        .iter()
        .all(|plane| plane.samples().len() == 4));

    let mut interleaved = Vec::with_capacity(12);
    for idx in 0..4 {
        for plane in planes.planes() {
            interleaved.push(plane.samples()[idx].round() as u8);
        }
    }
    assert_eq!(interleaved, bitmap.data);
}
