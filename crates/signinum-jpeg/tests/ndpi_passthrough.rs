// SPDX-License-Identifier: Apache-2.0

//! Optional local NDPI passthrough coverage.
//!
//! Set `SIGNINUM_NDPI_PATH=/path/to/slide.ndpi` to run this test against a
//! local slide. The test reads TIFF directories and compressed JPEG payloads;
//! it does not decode the whole-slide image. Set `SIGNINUM_NDPI_TILE_LIMIT=0`
//! to validate every JPEG payload in the container.

use signinum_core::{CodedUnitLayout, Colorspace, Info as CoreInfo};
use signinum_jpeg::{
    CompressedPayloadKind, CompressedTransferSyntax, JpegError, JpegView, PassthroughCandidate,
    PassthroughDecision, PassthroughRequirements,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    time::Instant,
};

const NDPI_PATH_ENV: &str = "SIGNINUM_NDPI_PATH";
const REQUIRE_NDPI_ENV: &str = "SIGNINUM_REQUIRE_NDPI";
const TILE_LIMIT_ENV: &str = "SIGNINUM_NDPI_TILE_LIMIT";
const MAX_PAYLOAD_BYTES_ENV: &str = "SIGNINUM_NDPI_MAX_PAYLOAD_BYTES";
const DEFAULT_TILE_LIMIT: usize = 8;
const DEFAULT_BOUNDED_MAX_PAYLOAD_BYTES: u64 = 64 * 1024 * 1024;

const TAG_IMAGE_WIDTH: u16 = 256;
const TAG_IMAGE_LENGTH: u16 = 257;
const TAG_BITS_PER_SAMPLE: u16 = 258;
const TAG_COMPRESSION: u16 = 259;
const TAG_PHOTOMETRIC: u16 = 262;
const TAG_STRIP_OFFSETS: u16 = 273;
const TAG_SAMPLES_PER_PIXEL: u16 = 277;
const TAG_STRIP_BYTE_COUNTS: u16 = 279;
const TAG_TILE_OFFSETS: u16 = 324;
const TAG_TILE_BYTE_COUNTS: u16 = 325;
const TIFF_COMPRESSION_JPEG: u64 = 7;

#[test]
fn ndpi_jpeg_payloads_are_passthrough_candidates_when_local_slide_is_available() {
    let Some(raw_path) = env::var_os(NDPI_PATH_ENV) else {
        assert!(
            env::var_os(REQUIRE_NDPI_ENV).is_none(),
            "{REQUIRE_NDPI_ENV} is set but {NDPI_PATH_ENV} is not configured"
        );
        return;
    };
    let path = PathBuf::from(raw_path);
    let tile_limit = ScanLimit::from_env();
    let default_max_payload_bytes = match tile_limit {
        ScanLimit::All => u64::MAX,
        ScanLimit::Payloads(_) => DEFAULT_BOUNDED_MAX_PAYLOAD_BYTES,
    };
    let max_payload_bytes = env_u64(MAX_PAYLOAD_BYTES_ENV, default_max_payload_bytes);

    let mut file = File::open(&path)
        .unwrap_or_else(|error| panic!("open NDPI slide {}: {error}", path.display()));
    let started = Instant::now();
    let summary = validate_jpeg_payloads(&mut file, tile_limit, max_payload_bytes, &path)
        .unwrap_or_else(|error| panic!("parse NDPI slide {}: {error}", path.display()));

    assert!(
        summary.validated_payloads > 0,
        "no JPEG compressed NDPI payloads under {max_payload_bytes} bytes were found in {}",
        path.display()
    );

    eprintln!(
        "validated {} of {} NDPI JPEG payload(s) from {} across {} dimension set(s) and {} IFD(s) in {:.3}s",
        summary.validated_payloads,
        summary.jpeg_payloads_seen,
        path.display(),
        summary.distinct_dimensions.len(),
        summary.ifd_payload_counts.len(),
        started.elapsed().as_secs_f64()
    );
    eprintln!(
        "payload bytes validated={} ndpi_zero_dimension_payloads={} skipped_too_large={} skipped_non_jpeg={}",
        summary.bytes_validated,
        summary.ndpi_zero_dimension_payloads,
        summary.skipped_too_large,
        summary.skipped_non_jpeg
    );
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[derive(Debug)]
struct NdpiPassthroughSummary {
    jpeg_payloads_seen: usize,
    validated_payloads: usize,
    bytes_validated: u64,
    skipped_too_large: usize,
    skipped_non_jpeg: usize,
    ndpi_zero_dimension_payloads: usize,
    distinct_dimensions: BTreeSet<(u32, u32)>,
    ifd_payload_counts: BTreeMap<usize, usize>,
}

impl NdpiPassthroughSummary {
    fn new() -> Self {
        Self {
            jpeg_payloads_seen: 0,
            validated_payloads: 0,
            bytes_validated: 0,
            skipped_too_large: 0,
            skipped_non_jpeg: 0,
            ndpi_zero_dimension_payloads: 0,
            distinct_dimensions: BTreeSet::new(),
            ifd_payload_counts: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ScanLimit {
    All,
    Payloads(usize),
}

impl ScanLimit {
    fn from_env() -> Self {
        match env_usize(TILE_LIMIT_ENV, DEFAULT_TILE_LIMIT) {
            0 => Self::All,
            limit => Self::Payloads(limit),
        }
    }

    fn remaining(self, seen: usize) -> usize {
        match self {
            Self::All => usize::MAX,
            Self::Payloads(limit) => limit.saturating_sub(seen),
        }
    }

    fn reached(self, seen: usize) -> bool {
        matches!(self, Self::Payloads(limit) if seen >= limit)
    }
}

fn validate_jpeg_payloads(
    file: &mut File,
    limit: ScanLimit,
    max_payload_bytes: u64,
    path: &Path,
) -> Result<NdpiPassthroughSummary, String> {
    let layout = TiffLayout::read(file).map_err(|error| format!("read TIFF header: {error}"))?;
    let mut summary = NdpiPassthroughSummary::new();
    let mut seen_ifds = BTreeSet::new();
    let mut ifd_offset = layout.first_ifd_offset;
    let mut ifd_index = 0usize;

    while ifd_offset != 0 && !limit.reached(summary.jpeg_payloads_seen) {
        if !seen_ifds.insert(ifd_offset) {
            return Err(format!("IFD loop at offset {ifd_offset}"));
        }
        if ifd_index > 256 {
            return Err("too many TIFF directories while scanning NDPI".to_string());
        }

        let ifd = read_ifd(file, layout, ifd_offset)
            .map_err(|error| format!("read IFD {ifd_index} at offset {ifd_offset}: {error}"))?;
        if first_numeric(file, layout, &ifd, TAG_COMPRESSION)? == Some(TIFF_COMPRESSION_JPEG) {
            let scan = PayloadScanContext {
                layout,
                ifd_index,
                limit,
                max_payload_bytes,
                path,
            };
            validate_payloads_from_ifd(file, &ifd, scan, &mut summary)?;
        }

        ifd_offset = ifd.next_ifd_offset;
        ifd_index += 1;
    }

    Ok(summary)
}

#[derive(Debug, Clone, Copy)]
struct PayloadScanContext<'a> {
    layout: TiffLayout,
    ifd_index: usize,
    limit: ScanLimit,
    max_payload_bytes: u64,
    path: &'a Path,
}

fn validate_payloads_from_ifd(
    file: &mut File,
    ifd: &Ifd,
    scan: PayloadScanContext<'_>,
    summary: &mut NdpiPassthroughSummary,
) -> Result<(), String> {
    let (offset_tag, count_tag) = if ifd.entry(TAG_TILE_OFFSETS).is_some() {
        (TAG_TILE_OFFSETS, TAG_TILE_BYTE_COUNTS)
    } else {
        (TAG_STRIP_OFFSETS, TAG_STRIP_BYTE_COUNTS)
    };
    let metadata = ifd_image_metadata(file, scan.layout, ifd)?;
    let remaining = scan.limit.remaining(summary.jpeg_payloads_seen);
    let offsets = numeric_values(file, scan.layout, ifd, offset_tag, remaining)?;
    let counts = numeric_values(file, scan.layout, ifd, count_tag, remaining)?;

    for (offset, byte_count) in offsets.into_iter().zip(counts).take(remaining) {
        summary.jpeg_payloads_seen += 1;
        if byte_count == 0 {
            summary.skipped_non_jpeg += 1;
            continue;
        }
        if byte_count > scan.max_payload_bytes {
            summary.skipped_too_large += 1;
            continue;
        }
        let bytes = read_exact_at(file, offset, byte_count as usize)
            .map_err(|error| format!("read JPEG payload at offset {offset}: {error}"))?;
        if !bytes.starts_with(&[0xff, 0xd8]) {
            summary.skipped_non_jpeg += 1;
            continue;
        }
        validate_payload_passthrough(&bytes, metadata, scan.ifd_index, offset, scan.path, summary);
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct IfdImageMetadata {
    dimensions: (u32, u32),
    components: u8,
    bit_depth: u8,
    colorspace: Colorspace,
}

fn ifd_image_metadata(
    file: &mut File,
    layout: TiffLayout,
    ifd: &Ifd,
) -> Result<IfdImageMetadata, String> {
    let width = first_numeric(file, layout, ifd, TAG_IMAGE_WIDTH)?
        .ok_or_else(|| "JPEG-compressed IFD missing ImageWidth".to_string())?;
    let height = first_numeric(file, layout, ifd, TAG_IMAGE_LENGTH)?
        .ok_or_else(|| "JPEG-compressed IFD missing ImageLength".to_string())?;
    let components = first_numeric(file, layout, ifd, TAG_SAMPLES_PER_PIXEL)?.unwrap_or(1);
    let bit_depth = first_numeric(file, layout, ifd, TAG_BITS_PER_SAMPLE)?.unwrap_or(8);
    let photometric = first_numeric(file, layout, ifd, TAG_PHOTOMETRIC)?;

    Ok(IfdImageMetadata {
        dimensions: (
            u32::try_from(width).map_err(|_| format!("ImageWidth {width} exceeds u32"))?,
            u32::try_from(height).map_err(|_| format!("ImageLength {height} exceeds u32"))?,
        ),
        components: u8::try_from(components)
            .map_err(|_| format!("SamplesPerPixel {components} exceeds u8"))?,
        bit_depth: u8::try_from(bit_depth)
            .map_err(|_| format!("BitsPerSample {bit_depth} exceeds u8"))?,
        colorspace: match (photometric, components) {
            (Some(6), 3) => Colorspace::YCbCr,
            (Some(2), 3) => Colorspace::Rgb,
            (Some(0 | 1), 1) => Colorspace::Grayscale,
            (_, 1) => Colorspace::Grayscale,
            (_, 3) => Colorspace::YCbCr,
            _ => Colorspace::IccTagged,
        },
    })
}

fn validate_payload_passthrough(
    bytes: &[u8],
    ifd_metadata: IfdImageMetadata,
    ifd_index: usize,
    offset: u64,
    path: &Path,
    summary: &mut NdpiPassthroughSummary,
) {
    let parsed = ndpi_passthrough_candidate(bytes, ifd_metadata).unwrap_or_else(|error| {
        panic!(
            "JPEG passthrough classification failed for IFD {ifd_index} payload at offset {offset} ({} bytes) in {}: {error}",
            bytes.len(),
            path.display()
        )
    });
    if parsed.used_ifd_dimensions {
        summary.ndpi_zero_dimension_payloads += 1;
    }
    let candidate = parsed.candidate;
    let requirements = PassthroughRequirements::new(
        candidate.transfer_syntax(),
        CompressedPayloadKind::JpegInterchange,
    )
    .with_dimensions(parsed.dimensions)
    .with_components(parsed.components)
    .with_bit_depth(parsed.bit_depth);

    assert_eq!(
        candidate.evaluate(&requirements),
        PassthroughDecision::Copy { bytes }
    );
    let passthrough_bytes = candidate
        .copy_bytes_if_eligible(&requirements)
        .expect("NDPI JPEG payload should be eligible for byte-preserving copy");
    assert_eq!(passthrough_bytes, bytes);
    assert!(
        std::ptr::eq(passthrough_bytes.as_ptr(), bytes.as_ptr()),
        "passthrough must return the original borrowed payload bytes"
    );

    summary.validated_payloads += 1;
    summary.bytes_validated += bytes.len() as u64;
    summary.distinct_dimensions.insert(parsed.dimensions);
    *summary.ifd_payload_counts.entry(ifd_index).or_insert(0) += 1;
}

#[derive(Debug)]
struct ParsedNdpiCandidate<'a> {
    candidate: PassthroughCandidate<'a>,
    dimensions: (u32, u32),
    components: u8,
    bit_depth: u8,
    used_ifd_dimensions: bool,
}

fn ndpi_passthrough_candidate(
    bytes: &[u8],
    ifd_metadata: IfdImageMetadata,
) -> Result<ParsedNdpiCandidate<'_>, String> {
    match JpegView::parse(bytes) {
        Ok(view) => {
            let candidate = view
                .passthrough_candidate()
                .ok_or_else(|| "JPEG payload is not an active passthrough candidate".to_string())?;
            Ok(ParsedNdpiCandidate {
                candidate,
                dimensions: view.info().dimensions,
                components: view.info().sampling.len() as u8,
                bit_depth: view.info().bit_depth,
                used_ifd_dimensions: false,
            })
        }
        Err(JpegError::ZeroDimension { .. }) => {
            let syntax = scan_zero_dimension_jpeg_syntax(bytes)?;
            let info = CoreInfo {
                dimensions: ifd_metadata.dimensions,
                components: ifd_metadata.components,
                colorspace: ifd_metadata.colorspace,
                bit_depth: ifd_metadata.bit_depth,
                tile_layout: None,
                coded_unit_layout: Some(CodedUnitLayout {
                    unit_width: 8,
                    unit_height: 8,
                    units_x: ifd_metadata.dimensions.0.div_ceil(8),
                    units_y: ifd_metadata.dimensions.1.div_ceil(8),
                }),
                restart_interval: None,
                resolution_levels: 1,
            };
            Ok(ParsedNdpiCandidate {
                candidate: PassthroughCandidate::new(
                    bytes,
                    syntax,
                    CompressedPayloadKind::JpegInterchange,
                    info,
                ),
                dimensions: ifd_metadata.dimensions,
                components: ifd_metadata.components,
                bit_depth: ifd_metadata.bit_depth,
                used_ifd_dimensions: true,
            })
        }
        Err(error) => Err(format!("{error:?}")),
    }
}

fn scan_zero_dimension_jpeg_syntax(bytes: &[u8]) -> Result<CompressedTransferSyntax, String> {
    if !bytes.starts_with(&[0xff, 0xd8]) {
        return Err("payload does not start with JPEG SOI".to_string());
    }
    let mut cursor = 2usize;
    while cursor + 4 <= bytes.len() {
        while cursor < bytes.len() && bytes[cursor] != 0xff {
            cursor += 1;
        }
        while cursor < bytes.len() && bytes[cursor] == 0xff {
            cursor += 1;
        }
        if cursor >= bytes.len() {
            break;
        }
        let marker = bytes[cursor];
        cursor += 1;
        if matches!(marker, 0xd8 | 0xd9 | 0x01 | 0xd0..=0xd7) {
            continue;
        }
        if cursor + 2 > bytes.len() {
            return Err("truncated JPEG segment length".to_string());
        }
        let length = u16::from_be_bytes([bytes[cursor], bytes[cursor + 1]]) as usize;
        if length < 2 || cursor + length > bytes.len() {
            return Err("invalid JPEG segment length".to_string());
        }
        let payload = &bytes[cursor + 2..cursor + length];
        match marker {
            0xc0 if payload.first() == Some(&8) => {
                return Ok(CompressedTransferSyntax::JpegBaseline8);
            }
            0xc1 if matches!(payload.first(), Some(8 | 12)) => {
                return Ok(CompressedTransferSyntax::JpegExtendedSequential);
            }
            0xc2 => return Err("progressive JPEG is not an active passthrough target".to_string()),
            0xc3 => return Err("lossless JPEG is not an active passthrough target".to_string()),
            0xda => return Err("JPEG SOS reached before SOF".to_string()),
            _ => {}
        }
        cursor += length;
    }

    Err("JPEG SOF marker not found".to_string())
}

#[derive(Debug, Clone, Copy)]
struct TiffLayout {
    endian: Endian,
    flavor: TiffFlavor,
    first_ifd_offset: u64,
}

impl TiffLayout {
    fn read(file: &mut File) -> io::Result<Self> {
        let header = read_exact_at(file, 0, 16)?;
        let endian = match &header[..2] {
            b"II" => Endian::Little,
            b"MM" => Endian::Big,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "not a TIFF byte-order header",
                ))
            }
        };
        match endian.u16(&header[2..4]) {
            42 => Ok(Self {
                endian,
                flavor: TiffFlavor::Classic,
                first_ifd_offset: u64::from(endian.u32(&header[4..8])),
            }),
            43 => {
                let offset_size = endian.u16(&header[4..6]);
                let zero = endian.u16(&header[6..8]);
                if offset_size != 8 || zero != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unsupported BigTIFF header",
                    ));
                }
                Ok(Self {
                    endian,
                    flavor: TiffFlavor::Big,
                    first_ifd_offset: endian.u64(&header[8..16]),
                })
            }
            magic => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported TIFF magic {magic}"),
            )),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Endian {
    Little,
    Big,
}

impl Endian {
    fn u16(self, bytes: &[u8]) -> u16 {
        let value = [bytes[0], bytes[1]];
        match self {
            Self::Little => u16::from_le_bytes(value),
            Self::Big => u16::from_be_bytes(value),
        }
    }

    fn u32(self, bytes: &[u8]) -> u32 {
        let value = [bytes[0], bytes[1], bytes[2], bytes[3]];
        match self {
            Self::Little => u32::from_le_bytes(value),
            Self::Big => u32::from_be_bytes(value),
        }
    }

    fn u64(self, bytes: &[u8]) -> u64 {
        let value = [
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ];
        match self {
            Self::Little => u64::from_le_bytes(value),
            Self::Big => u64::from_be_bytes(value),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum TiffFlavor {
    Classic,
    Big,
}

#[derive(Debug)]
struct Ifd {
    entries: Vec<IfdEntry>,
    next_ifd_offset: u64,
}

impl Ifd {
    fn entry(&self, tag: u16) -> Option<&IfdEntry> {
        self.entries.iter().find(|entry| entry.tag == tag)
    }
}

#[derive(Debug, Clone, Copy)]
struct IfdEntry {
    tag: u16,
    field_type: u16,
    count: u64,
    value_or_offset: u64,
    inline_value: [u8; 8],
}

fn read_ifd(file: &mut File, layout: TiffLayout, offset: u64) -> io::Result<Ifd> {
    match layout.flavor {
        TiffFlavor::Classic => read_classic_ifd(file, layout.endian, offset),
        TiffFlavor::Big => read_big_ifd(file, layout.endian, offset),
    }
}

fn read_classic_ifd(file: &mut File, endian: Endian, offset: u64) -> io::Result<Ifd> {
    let count_bytes = read_exact_at(file, offset, 2)?;
    let entry_count = usize::from(endian.u16(&count_bytes));
    let bytes = read_exact_at(file, offset + 2, entry_count * 12 + 4)?;
    let mut entries = Vec::with_capacity(entry_count);
    for chunk in bytes[..entry_count * 12].chunks_exact(12) {
        let mut inline_value = [0u8; 8];
        inline_value[..4].copy_from_slice(&chunk[8..12]);
        entries.push(IfdEntry {
            tag: endian.u16(&chunk[0..2]),
            field_type: endian.u16(&chunk[2..4]),
            count: u64::from(endian.u32(&chunk[4..8])),
            value_or_offset: u64::from(endian.u32(&chunk[8..12])),
            inline_value,
        });
    }
    let next_ifd_offset = u64::from(endian.u32(&bytes[entry_count * 12..entry_count * 12 + 4]));
    Ok(Ifd {
        entries,
        next_ifd_offset,
    })
}

fn read_big_ifd(file: &mut File, endian: Endian, offset: u64) -> io::Result<Ifd> {
    let count_bytes = read_exact_at(file, offset, 8)?;
    let entry_count = usize::try_from(endian.u64(&count_bytes)).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "BigTIFF directory entry count does not fit in usize",
        )
    })?;
    if entry_count > 4096 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unreasonably large BigTIFF directory",
        ));
    }
    let bytes = read_exact_at(file, offset + 8, entry_count * 20 + 8)?;
    let mut entries = Vec::with_capacity(entry_count);
    for chunk in bytes[..entry_count * 20].chunks_exact(20) {
        let mut inline_value = [0u8; 8];
        inline_value.copy_from_slice(&chunk[12..20]);
        entries.push(IfdEntry {
            tag: endian.u16(&chunk[0..2]),
            field_type: endian.u16(&chunk[2..4]),
            count: endian.u64(&chunk[4..12]),
            value_or_offset: endian.u64(&chunk[12..20]),
            inline_value,
        });
    }
    let next_ifd_offset = endian.u64(&bytes[entry_count * 20..entry_count * 20 + 8]);
    Ok(Ifd {
        entries,
        next_ifd_offset,
    })
}

fn first_numeric(
    file: &mut File,
    layout: TiffLayout,
    ifd: &Ifd,
    tag: u16,
) -> Result<Option<u64>, String> {
    Ok(numeric_values(file, layout, ifd, tag, 1)?
        .into_iter()
        .next())
}

fn numeric_values(
    file: &mut File,
    layout: TiffLayout,
    ifd: &Ifd,
    tag: u16,
    max_values: usize,
) -> Result<Vec<u64>, String> {
    let Some(entry) = ifd.entry(tag) else {
        return Ok(Vec::new());
    };
    let Some(type_size) = tiff_type_size(entry.field_type) else {
        return Ok(Vec::new());
    };
    let value_count = usize::try_from(entry.count)
        .unwrap_or(usize::MAX)
        .min(max_values);
    let read_len = value_count
        .checked_mul(type_size)
        .ok_or_else(|| format!("tag {tag} value byte count overflow"))?;
    let total_len = entry.count.saturating_mul(type_size as u64);
    let data = if total_len <= inline_slot_len(layout.flavor) as u64 {
        entry.inline_value[..read_len].to_vec()
    } else {
        read_exact_at(file, entry.value_or_offset, read_len).map_err(|error| {
            format!(
                "read tag {tag} values at offset {}: {error}",
                entry.value_or_offset
            )
        })?
    };

    Ok(data
        .chunks_exact(type_size)
        .map(|chunk| numeric_from_tiff_value(layout.endian, entry.field_type, chunk))
        .collect())
}

fn tiff_type_size(field_type: u16) -> Option<usize> {
    match field_type {
        1 => Some(1),
        3 => Some(2),
        4 | 9 => Some(4),
        16 | 17 | 18 => Some(8),
        _ => None,
    }
}

fn numeric_from_tiff_value(endian: Endian, field_type: u16, bytes: &[u8]) -> u64 {
    match field_type {
        1 => u64::from(bytes[0]),
        3 => u64::from(endian.u16(bytes)),
        4 => u64::from(endian.u32(bytes)),
        9 => i64::from(endian.u32(bytes) as i32) as u64,
        16 | 18 => endian.u64(bytes),
        17 => endian.u64(bytes) as i64 as u64,
        _ => 0,
    }
}

fn inline_slot_len(flavor: TiffFlavor) -> usize {
    match flavor {
        TiffFlavor::Classic => 4,
        TiffFlavor::Big => 8,
    }
}

fn read_exact_at(file: &mut File, offset: u64, len: usize) -> io::Result<Vec<u8>> {
    let mut bytes = vec![0u8; len];
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}
