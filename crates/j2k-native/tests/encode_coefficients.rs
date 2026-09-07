use j2k_native::{
    encode, encode_component_planes_53, encode_precomputed_htj2k_53, encode_precomputed_htj2k_97,
    encode_precomputed_j2k_53, DecodeSettings, DecoderContext, EncodeComponentPlane, EncodeError,
    EncodeOptions, Image, J2kForwardDwt53Level, J2kForwardDwt53Output, J2kForwardDwt97Level,
    J2kForwardDwt97Output, PrecomputedHtj2k53Component, PrecomputedHtj2k53Image,
    PrecomputedHtj2k97Component, PrecomputedHtj2k97Image,
};

#[test]
fn native_encode_rejects_dimension_overflow_before_length_check() {
    let err = encode(
        &[0; 16],
        u32::MAX,
        u32::MAX,
        4,
        16,
        false,
        &EncodeOptions::default(),
    )
    .expect_err("dimension overflow should be rejected");

    assert_eq!(
        err,
        EncodeError::ArithmeticOverflow {
            what: "raw encode byte length"
        }
    );
}

#[test]
fn native_decode_rejects_zero_target_resolution() {
    let bytes = encode(
        &[10, 20, 30, 40],
        2,
        2,
        1,
        8,
        false,
        &EncodeOptions::default(),
    )
    .expect("encode fixture");
    let settings = DecodeSettings {
        target_resolution: Some((0, 1)),
        ..DecodeSettings::default()
    };

    assert!(Image::new(&bytes, &settings).is_err());
}

#[test]
fn jp2_decode_rejects_short_channel_definition_box() {
    let codestream = encode(
        &[10, 20, 30, 40, 50, 60],
        2,
        1,
        3,
        8,
        false,
        &EncodeOptions::default(),
    )
    .expect("encode RGB fixture");
    let jp2 = wrap_codestream_jp2_with_short_cdef(&codestream, 2, 1, 3, 8);

    let image = Image::new(&jp2, &DecodeSettings::default()).expect("parse JP2 fixture");
    let err = image
        .decode()
        .expect_err("short cdef must not silently drop a component");

    assert_eq!(err.to_string(), "invalid channel definition");
}

#[test]
fn precomputed_zero_grayscale_53_coefficients_decode_with_native_decoder() {
    let image = PrecomputedHtj2k53Image {
        width: 8,
        height: 8,
        bit_depth: 8,
        signed: false,
        components: vec![PrecomputedHtj2k53Component {
            x_rsiz: 1,
            y_rsiz: 1,
            dwt: zero_dwt53(8, 8),
        }],
    };
    let bytes = encode_precomputed_htj2k_53(&image, &precomputed_options())
        .expect("encode precomputed HTJ2K coefficients");
    let decoded = Image::new(&bytes, &DecodeSettings::default())
        .expect("native parser accepts codestream")
        .decode_native()
        .expect("native decoder accepts codestream");

    assert_eq!((decoded.width, decoded.height), (8, 8));
    assert_eq!(decoded.num_components, 1);
    assert!(decoded.data.iter().all(|&sample| sample == 128));
}

#[test]
fn precomputed_53_encode_roundtrips_more_than_four_components() {
    let image = PrecomputedHtj2k53Image {
        width: 8,
        height: 8,
        bit_depth: 8,
        signed: false,
        components: (0..5)
            .map(|_| PrecomputedHtj2k53Component {
                x_rsiz: 1,
                y_rsiz: 1,
                dwt: zero_dwt53(8, 8),
            })
            .collect(),
    };
    let bytes = encode_precomputed_htj2k_53(&image, &precomputed_options())
        .expect("encode five-component precomputed 5/3 HTJ2K coefficients");
    let decoded = Image::new(&bytes, &DecodeSettings::default())
        .expect("native parser accepts five-component codestream")
        .decode_native()
        .expect("native decoder accepts five-component codestream");

    assert_eq!((decoded.width, decoded.height), (8, 8));
    assert_eq!(decoded.num_components, 5);
    assert_eq!(decoded.data.len(), 8 * 8 * 5);
    assert!(decoded.data.iter().all(|&sample| sample == 128));
}

#[test]
fn precomputed_encode_writes_component_sampling_in_siz() {
    let image = PrecomputedHtj2k53Image {
        width: 16,
        height: 16,
        bit_depth: 8,
        signed: false,
        components: vec![
            PrecomputedHtj2k53Component {
                x_rsiz: 1,
                y_rsiz: 1,
                dwt: zero_dwt53(16, 16),
            },
            PrecomputedHtj2k53Component {
                x_rsiz: 2,
                y_rsiz: 2,
                dwt: zero_dwt53(8, 8),
            },
            PrecomputedHtj2k53Component {
                x_rsiz: 2,
                y_rsiz: 2,
                dwt: zero_dwt53(8, 8),
            },
        ],
    };
    let bytes = encode_precomputed_htj2k_53(&image, &precomputed_options())
        .expect("encode precomputed subsampled HTJ2K coefficients");
    let siz = find_marker(&bytes, 0x51).expect("SIZ marker");
    let component_info = siz + 40;

    assert_eq!(bytes[component_info + 1], 1);
    assert_eq!(bytes[component_info + 2], 1);
    assert_eq!(bytes[component_info + 4], 2);
    assert_eq!(bytes[component_info + 5], 2);
    assert_eq!(bytes[component_info + 7], 2);
    assert_eq!(bytes[component_info + 8], 2);

    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse sampled codestream");
    let (plan, sampling) = image
        .build_component_grid_color_plan_with_context(&mut DecoderContext::default())
        .expect("sampled component-grid plan");
    assert_eq!(sampling, [(1, 1), (2, 2), (2, 2)]);
    assert!(
        image
            .build_direct_color_plan_with_context(&mut DecoderContext::default())
            .is_err(),
        "full-resolution plan contract must remain separate from component-grid output"
    );
    assert_eq!(plan.dimensions, (16, 16));
    assert_eq!(
        plan.component_plans
            .iter()
            .map(|p| p.dimensions)
            .collect::<Vec<_>>(),
        [(16, 16), (8, 8), (8, 8)]
    );
    let mut context = DecoderContext::default();
    let components = image
        .decode_components_with_context(&mut context)
        .expect("decode sampled component planes");
    let borrowed_sampling = components
        .planes()
        .iter()
        .map(j2k_native::ComponentPlane::sampling)
        .collect::<Vec<_>>();
    assert_eq!(borrowed_sampling, [(1, 1), (2, 2), (2, 2)]);

    let owned_components = image
        .decode_native_components()
        .expect("decode owned native sampled component planes");
    let owned_sampling = owned_components
        .planes()
        .iter()
        .map(j2k_native::NativeComponentPlane::sampling)
        .collect::<Vec<_>>();
    assert_eq!(owned_sampling, borrowed_sampling);
}

#[test]
fn precomputed_classic_53_encode_preserves_component_sampling_in_siz() {
    let image = PrecomputedHtj2k53Image {
        width: 16,
        height: 16,
        bit_depth: 8,
        signed: false,
        components: vec![
            PrecomputedHtj2k53Component {
                x_rsiz: 1,
                y_rsiz: 1,
                dwt: zero_dwt53(16, 16),
            },
            PrecomputedHtj2k53Component {
                x_rsiz: 2,
                y_rsiz: 2,
                dwt: zero_dwt53(8, 8),
            },
            PrecomputedHtj2k53Component {
                x_rsiz: 2,
                y_rsiz: 2,
                dwt: zero_dwt53(8, 8),
            },
        ],
    };

    let bytes = encode_precomputed_j2k_53(&image, &precomputed_options())
        .expect("encode precomputed sampled classic J2K coefficients");
    let decoded = Image::new(&bytes, &DecodeSettings::default())
        .expect("native parser accepts sampled classic codestream")
        .decode_native()
        .expect("native decoder accepts sampled classic codestream");

    assert_eq!((decoded.width, decoded.height), (16, 16));
    assert_eq!(decoded.num_components, 3);

    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse sampled codestream");
    let mut context = DecoderContext::default();
    let components = image
        .decode_components_with_context(&mut context)
        .expect("decode sampled classic component planes");
    let sampling = components
        .planes()
        .iter()
        .map(j2k_native::ComponentPlane::sampling)
        .collect::<Vec<_>>();
    assert_eq!(sampling, [(1, 1), (2, 2), (2, 2)]);
}

#[test]
fn component_plane_53_encode_preserves_sampling_for_classic_and_htj2k() {
    let luma = vec![128_u8; 16 * 16];
    let chroma_blue = vec![96_u8; 8 * 8];
    let chroma_red = vec![160_u8; 8 * 8];
    let planes = [
        EncodeComponentPlane {
            data: &luma,
            x_rsiz: 1,
            y_rsiz: 1,
        },
        EncodeComponentPlane {
            data: &chroma_blue,
            x_rsiz: 2,
            y_rsiz: 2,
        },
        EncodeComponentPlane {
            data: &chroma_red,
            x_rsiz: 2,
            y_rsiz: 2,
        },
    ];

    for use_ht_block_coding in [false, true] {
        let options = EncodeOptions {
            reversible: true,
            use_ht_block_coding,
            num_decomposition_levels: 1,
            use_mct: false,
            validate_high_throughput_codestream: false,
            ..EncodeOptions::default()
        };
        let bytes = encode_component_planes_53(&planes, 16, 16, 8, false, &options)
            .expect("component-plane encode");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse codestream");
        let mut context = DecoderContext::default();
        let components = image
            .decode_components_with_context(&mut context)
            .expect("decode component planes");
        let sampling = components
            .planes()
            .iter()
            .map(j2k_native::ComponentPlane::sampling)
            .collect::<Vec<_>>();

        assert_eq!(sampling, [(1, 1), (2, 2), (2, 2)]);
    }
}

#[test]
fn raw_pixel_encode_rejects_component_sampling_without_component_sized_dwt() {
    let pixels = vec![0; 16 * 16 * 3];
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        use_mct: false,
        component_sampling: Some(vec![(1, 1), (2, 2), (2, 2)]),
        ..EncodeOptions::default()
    };

    let err = encode(&pixels, 16, 16, 3, 8, false, &options)
        .expect_err("raw pixel encode cannot infer sampled component grids");

    assert_eq!(
        err,
        EncodeError::InternalInvariant {
            what: "component sampling requires component-sized DWT geometry"
        }
    );
}

#[test]
fn component_grid_plan_rejects_unrepresented_geometry() {
    for (mct, signed, reduced, offset) in [
        (true, false, false, false),
        (false, true, false, false),
        (false, false, true, false),
        (false, false, false, true),
    ] {
        let mut bytes = encode(
            &[100; 32 * 32 * 3],
            32,
            32,
            3,
            8,
            signed,
            &EncodeOptions {
                use_mct: mct,
                num_decomposition_levels: 2,
                ..EncodeOptions::default()
            },
        )
        .unwrap();
        if offset {
            let siz = find_marker(&bytes, 0x51).unwrap();
            // Shift the reference grid, image origin, and tile origin together.
            bytes[siz + 6..siz + 10].copy_from_slice(&33_u32.to_be_bytes());
            bytes[siz + 14..siz + 18].copy_from_slice(&1_u32.to_be_bytes());
            bytes[siz + 30..siz + 34].copy_from_slice(&1_u32.to_be_bytes());
        }
        let image = Image::new(
            &bytes,
            &DecodeSettings {
                target_resolution: reduced.then_some((16, 16)),
                ..DecodeSettings::default()
            },
        )
        .unwrap();
        assert!(
            matches!(
                image.build_component_grid_color_plan_with_context(&mut DecoderContext::default()),
                Err(j2k_native::DecodeError::Decoding(
                    j2k_native::DecodingError::DirectPlanUnsupported(
                        j2k_native::DirectPlanUnsupportedReason::ComponentGridFullImage
                    )
                ))
            ),
            "unrepresented geometry: {mct}/{signed}/{reduced}/{offset}"
        );
    }
}

#[test]
fn precomputed_encode_rejects_component_sampling_geometry_mismatch() {
    let image = PrecomputedHtj2k53Image {
        width: 16,
        height: 16,
        bit_depth: 8,
        signed: false,
        components: vec![PrecomputedHtj2k53Component {
            x_rsiz: 2,
            y_rsiz: 2,
            dwt: zero_dwt53(16, 16),
        }],
    };

    let err = encode_precomputed_htj2k_53(&image, &precomputed_options())
        .expect_err("component DWT geometry must match SIZ sampling");

    assert_eq!(
        err,
        EncodeError::InvalidInput {
            what: "precomputed DWT component dimensions mismatch",
        }
    );
}

#[test]
fn precomputed_encode_rejects_recursive_level_geometry_mismatch() {
    let mut dwt = zero_dwt53(8, 8);
    dwt.levels[0].low_width = 3;
    let image = PrecomputedHtj2k53Image {
        width: 8,
        height: 8,
        bit_depth: 8,
        signed: false,
        components: vec![PrecomputedHtj2k53Component {
            x_rsiz: 1,
            y_rsiz: 1,
            dwt,
        }],
    };

    let err = encode_precomputed_htj2k_53(&image, &precomputed_options())
        .expect_err("level geometry must match recursive 5/3 expectations");

    assert_eq!(
        err,
        EncodeError::InvalidInput {
            what: "precomputed DWT recursive geometry mismatch",
        }
    );
}

#[test]
fn precomputed_zero_grayscale_97_coefficients_decode_with_native_decoder() {
    let image = PrecomputedHtj2k97Image {
        width: 8,
        height: 8,
        bit_depth: 8,
        signed: false,
        components: vec![PrecomputedHtj2k97Component {
            x_rsiz: 1,
            y_rsiz: 1,
            dwt: zero_dwt97(8, 8),
        }],
    };
    let bytes = encode_precomputed_htj2k_97(&image, &precomputed_lossy_options())
        .expect("encode precomputed 9/7 HTJ2K coefficients");
    let decoded = Image::new(&bytes, &DecodeSettings::default())
        .expect("native parser accepts 9/7 codestream")
        .decode_native()
        .expect("native decoder accepts 9/7 codestream");

    assert_eq!((decoded.width, decoded.height), (8, 8));
    assert_eq!(decoded.num_components, 1);
    assert!(decoded.data.iter().all(|&sample| sample == 128));
}

#[test]
fn precomputed_97_encode_roundtrips_more_than_four_components() {
    let image = PrecomputedHtj2k97Image {
        width: 8,
        height: 8,
        bit_depth: 8,
        signed: false,
        components: (0..5)
            .map(|_| PrecomputedHtj2k97Component {
                x_rsiz: 1,
                y_rsiz: 1,
                dwt: zero_dwt97(8, 8),
            })
            .collect(),
    };
    let bytes = encode_precomputed_htj2k_97(&image, &precomputed_lossy_options())
        .expect("encode five-component precomputed 9/7 HTJ2K coefficients");
    let decoded = Image::new(&bytes, &DecodeSettings::default())
        .expect("native parser accepts five-component 9/7 codestream")
        .decode_native()
        .expect("native decoder accepts five-component 9/7 codestream");

    assert_eq!((decoded.width, decoded.height), (8, 8));
    assert_eq!(decoded.num_components, 5);
    assert_eq!(decoded.data.len(), 8 * 8 * 5);
    assert!(decoded.data.iter().all(|&sample| sample == 128));
}

#[test]
fn precomputed_97_encode_rejects_component_sampling_geometry_mismatch() {
    let image = PrecomputedHtj2k97Image {
        width: 16,
        height: 16,
        bit_depth: 8,
        signed: false,
        components: vec![PrecomputedHtj2k97Component {
            x_rsiz: 2,
            y_rsiz: 2,
            dwt: zero_dwt97(16, 16),
        }],
    };

    let err = encode_precomputed_htj2k_97(&image, &precomputed_lossy_options())
        .expect_err("component DWT geometry must match SIZ sampling");

    assert_eq!(
        err,
        EncodeError::InvalidInput {
            what: "precomputed DWT component dimensions mismatch",
        }
    );
}

fn precomputed_options() -> EncodeOptions {
    EncodeOptions {
        num_decomposition_levels: 1,
        reversible: true,
        use_ht_block_coding: true,
        use_mct: false,
        validate_high_throughput_codestream: false,
        ..EncodeOptions::default()
    }
}

fn precomputed_lossy_options() -> EncodeOptions {
    EncodeOptions {
        num_decomposition_levels: 1,
        reversible: false,
        use_ht_block_coding: true,
        use_mct: false,
        validate_high_throughput_codestream: false,
        ..EncodeOptions::default()
    }
}

fn zero_dwt53(width: u32, height: u32) -> J2kForwardDwt53Output {
    let low_width = width.div_ceil(2);
    let low_height = height.div_ceil(2);
    let high_width = width / 2;
    let high_height = height / 2;

    J2kForwardDwt53Output {
        ll: vec![0.0; (low_width * low_height) as usize],
        ll_width: low_width,
        ll_height: low_height,
        levels: vec![J2kForwardDwt53Level {
            hl: vec![0.0; (high_width * low_height) as usize],
            lh: vec![0.0; (low_width * high_height) as usize],
            hh: vec![0.0; (high_width * high_height) as usize],
            width,
            height,
            low_width,
            low_height,
            high_width,
            high_height,
        }],
    }
}

fn zero_dwt97(width: u32, height: u32) -> J2kForwardDwt97Output {
    let low_width = width.div_ceil(2);
    let low_height = height.div_ceil(2);
    let high_width = width / 2;
    let high_height = height / 2;

    J2kForwardDwt97Output {
        ll: vec![0.0; (low_width * low_height) as usize],
        ll_width: low_width,
        ll_height: low_height,
        levels: vec![J2kForwardDwt97Level {
            hl: vec![0.0; (high_width * low_height) as usize],
            lh: vec![0.0; (low_width * high_height) as usize],
            hh: vec![0.0; (high_width * high_height) as usize],
            width,
            height,
            low_width,
            low_height,
            high_width,
            high_height,
        }],
    }
}

fn find_marker(codestream: &[u8], marker: u8) -> Option<usize> {
    codestream
        .windows(2)
        .position(|window| window == [0xff, marker])
}

fn wrap_codestream_jp2_with_short_cdef(
    codestream: &[u8],
    width: u32,
    height: u32,
    components: u16,
    bit_depth: u8,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    push_box(&mut bytes, *b"jP  ", &[0x0D, 0x0A, 0x87, 0x0A]);
    push_box(
        &mut bytes,
        *b"ftyp",
        &[b'j', b'p', b'2', b' ', 0, 0, 0, 0, b'j', b'p', b'2', b' '],
    );

    let mut jp2h = Vec::new();
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&components.to_be_bytes());
    ihdr.extend_from_slice(&[bit_depth.saturating_sub(1), 7, 0, 0]);
    push_box(&mut jp2h, *b"ihdr", &ihdr);

    let mut colr = vec![1, 0, 0];
    colr.extend_from_slice(&16_u32.to_be_bytes());
    push_box(&mut jp2h, *b"colr", &colr);

    let mut cdef = Vec::new();
    cdef.extend_from_slice(&2_u16.to_be_bytes());
    for (channel_index, association) in [(0_u16, 1_u16), (1, 2)] {
        cdef.extend_from_slice(&channel_index.to_be_bytes());
        cdef.extend_from_slice(&0_u16.to_be_bytes());
        cdef.extend_from_slice(&association.to_be_bytes());
    }
    push_box(&mut jp2h, *b"cdef", &cdef);

    push_box(&mut bytes, *b"jp2h", &jp2h);
    push_box(&mut bytes, *b"jp2c", codestream);
    bytes
}

fn push_box(bytes: &mut Vec<u8>, box_type: [u8; 4], payload: &[u8]) {
    let len = u32::try_from(payload.len())
        .expect("test box payload length fits u32")
        .checked_add(8)
        .expect("test box length fits u32");
    bytes.extend_from_slice(&len.to_be_bytes());
    bytes.extend_from_slice(&box_type);
    bytes.extend_from_slice(payload);
}
