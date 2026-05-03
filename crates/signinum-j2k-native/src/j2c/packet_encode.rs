//! Tier-2 packet formation for JPEG 2000 encoding.
//!
//! Organizes encoded code-block bitstreams into packets according to the
//! LRCP progression order. Each packet contains code-block data for a
//! single (layer, resolution, component, precinct) tuple.
//!
//! A packet at resolution 0 has one subband (LL).
//! A packet at resolution r > 0 has three subbands (HL, LH, HH).
//! Each subband has its own tag trees for inclusion and zero bitplanes.
//!
//! See Annex B of ITU-T T.800.

use alloc::vec::Vec;

use super::codestream_write::BlockCodingMode;
use super::tag_tree_encode::TagTreeEncoder;
use crate::writer::BitWriter;

/// A code-block's contribution to a packet.
#[derive(Debug)]
pub(crate) struct CodeBlockPacketData {
    /// Encoded bitstream data.
    pub(crate) data: Vec<u8>,
    /// Number of coding passes in this contribution.
    pub(crate) num_coding_passes: u8,
    /// Number of zero bitplanes (only relevant for first inclusion).
    pub(crate) num_zero_bitplanes: u8,
    /// Whether this code-block has been included in a previous packet.
    pub(crate) previously_included: bool,
    /// L-block value (for segment length encoding, starts at 3).
    pub(crate) l_block: u32,
    /// Block coder used for this contribution.
    pub(crate) block_coding_mode: BlockCodingMode,
}

/// Information about a single subband's precinct.
#[derive(Debug)]
pub(crate) struct SubbandPrecinct {
    /// Code-blocks in this subband's precinct (row-major order).
    pub(crate) code_blocks: Vec<CodeBlockPacketData>,
    /// Number of code-blocks in the x direction.
    pub(crate) num_cbs_x: u32,
    /// Number of code-blocks in the y direction.
    pub(crate) num_cbs_y: u32,
}

/// A resolution-level packet containing one or more subband precincts.
///
/// Resolution 0 has 1 subband (LL).
/// Resolution r>0 has 3 subbands (HL, LH, HH).
#[derive(Debug)]
pub(crate) struct ResolutionPacket {
    /// Subbands in this resolution's precinct.
    pub(crate) subbands: Vec<SubbandPrecinct>,
}

/// Form a packet from a resolution-level packet (possibly multiple subbands).
///
/// Returns the packet bytes (header + body).
pub(crate) fn form_packet(resolution: &mut ResolutionPacket) -> Vec<u8> {
    let mut header_writer = BitWriter::new();
    let mut body = Vec::new();

    // Check if any code-block across all subbands has data
    let any_data = resolution
        .subbands
        .iter()
        .any(|sb| sb.code_blocks.iter().any(|cb| cb.num_coding_passes > 0));

    if !any_data {
        // Empty packet: just write 0 bit
        header_writer.write_bit(0);
        return header_writer.finish();
    }

    // Non-empty packet indicator
    header_writer.write_bit(1);

    // Process each subband in order (LL for res 0; HL, LH, HH for res > 0)
    for subband in resolution.subbands.iter_mut() {
        // Create tag trees for this subband's code-block inclusion and zero bitplanes
        let mut inclusion_tree = TagTreeEncoder::new(subband.num_cbs_x, subband.num_cbs_y);
        let mut zbp_tree = TagTreeEncoder::new(subband.num_cbs_x, subband.num_cbs_y);

        // Set up tag tree values
        for (i, cb) in subband.code_blocks.iter().enumerate() {
            let x = i as u32 % subband.num_cbs_x;
            let y = i as u32 / subband.num_cbs_x;

            let inclusion_val = if cb.num_coding_passes > 0 {
                0
            } else {
                u32::MAX / 2
            };
            inclusion_tree.set_value(x, y, inclusion_val);
            zbp_tree.set_value(x, y, cb.num_zero_bitplanes as u32);
        }

        // Encode each code-block's packet contribution
        for (i, cb) in subband.code_blocks.iter_mut().enumerate() {
            let x = i as u32 % subband.num_cbs_x;
            let y = i as u32 / subband.num_cbs_x;

            if !cb.previously_included {
                // First inclusion: use tag tree
                inclusion_tree.encode(x, y, 1, &mut header_writer);

                if cb.num_coding_passes == 0 {
                    continue;
                }

                // Zero bitplanes: use tag tree
                zbp_tree.encode(x, y, cb.num_zero_bitplanes as u32 + 1, &mut header_writer);
            } else if cb.num_coding_passes > 0 {
                header_writer.write_bit(1);
            } else {
                header_writer.write_bit(0);
                continue;
            }

            if cb.num_coding_passes == 0 {
                continue;
            }

            let data_len = cb.data.len() as u32;
            match cb.block_coding_mode {
                BlockCodingMode::Classic => {
                    let num_bits = bits_for_length(cb.l_block, cb.num_coding_passes);
                    encode_num_coding_passes(cb.num_coding_passes, &mut header_writer);
                    encode_length(data_len, &mut cb.l_block, num_bits, &mut header_writer);
                }
                BlockCodingMode::HighThroughput => {
                    debug_assert!(
                        cb.num_coding_passes <= 1,
                        "current HT packet writer only supports cleanup-only contributions"
                    );
                    let num_bits = bits_for_ht_cleanup_length(cb.l_block, cb.num_coding_passes);
                    encode_num_ht_coding_passes(cb.num_coding_passes, &mut header_writer);
                    encode_length(data_len, &mut cb.l_block, num_bits, &mut header_writer);
                }
            }

            // Append code-block data to body
            body.extend_from_slice(&cb.data);
            cb.previously_included = true;
        }
    }

    // Assemble: header (byte-aligned) + body. Packet headers use JPEG 2000
    // bit stuffing; if the final header byte is 0xff, the following byte must
    // carry the stuffed zero bit before any packet body bytes.
    let mut packet = header_writer.finish();
    if packet.last().copied() == Some(0xff) {
        packet.push(0x00);
    }
    packet.extend_from_slice(&body);
    packet
}

/// Encode the number of coding passes using the variable-length code from Table B.4.
fn encode_num_coding_passes(num_passes: u8, writer: &mut BitWriter) {
    match num_passes {
        1 => writer.write_bit(0),
        2 => writer.write_bits(0b10, 2),
        3 => writer.write_bits(0b1100, 4),
        4 => writer.write_bits(0b1101, 4),
        5 => writer.write_bits(0b1110, 4),
        6..=36 => {
            writer.write_bits(0b1111, 4);
            writer.write_bits((num_passes - 6) as u32, 5);
        }
        37..=164 => {
            writer.write_bits(0b1_1111_1111, 9);
            writer.write_bits((num_passes - 37) as u32, 7);
        }
        _ => unreachable!("JPEG 2000 supports 1..=164 coding passes per contribution"),
    }
}

fn encode_num_ht_coding_passes(num_passes: u8, writer: &mut BitWriter) {
    match num_passes {
        1 => writer.write_bit(0),
        2 => writer.write_bits(0b10, 2),
        3..=5 => {
            writer.write_bits(0b11, 2);
            writer.write_bits(u32::from(num_passes - 3), 2);
        }
        6..=36 => {
            writer.write_bits(0b11, 2);
            writer.write_bits(0b11, 2);
            writer.write_bits(u32::from(num_passes - 6), 5);
        }
        37..=164 => {
            writer.write_bits(0b11, 2);
            writer.write_bits(0b11, 2);
            writer.write_bits(31, 5);
            writer.write_bits(u32::from(num_passes - 37), 7);
        }
        _ => unreachable!("JPEG 2000 supports 1..=164 coding passes per contribution"),
    }
}

fn encode_length(length: u32, l_block: &mut u32, mut num_bits: u32, writer: &mut BitWriter) {
    while !value_fits_in_bits(length, num_bits) {
        writer.write_bit(1);
        *l_block += 1;
        num_bits += 1;
    }
    writer.write_bit(0);
    writer.write_bits(length, num_bits as u8);
}

fn value_fits_in_bits(value: u32, bits: u32) -> bool {
    bits >= u32::BITS || value < (1u32 << bits)
}

/// Calculate number of bits needed to encode a segment length.
fn bits_for_length(l_block: u32, num_coding_passes: u8) -> u32 {
    let log2_passes = if num_coding_passes <= 1 {
        0
    } else {
        (num_coding_passes as u32).ilog2()
    };
    l_block + log2_passes
}

fn bits_for_ht_cleanup_length(l_block: u32, raw_num_passes: u8) -> u32 {
    let placeholder_groups = u32::from(raw_num_passes.saturating_sub(1)) / 3;
    let placeholder_passes = placeholder_groups * 3;
    l_block + (placeholder_passes + 1).ilog2()
}

/// Form tile bitstream from resolution packets in LRCP order.
///
/// `resolution_packets` contains one `ResolutionPacket` per resolution level:
/// - Index 0: LL band (resolution 0)
/// - Index 1..N: higher resolutions (each with HL, LH, HH subbands)
pub(crate) fn form_tile_bitstream(
    resolution_packets: &mut [ResolutionPacket],
    _num_layers: u8,
    _num_components: u8,
) -> Vec<u8> {
    let mut tile_data = Vec::new();

    // LRCP: Layer → Resolution → Component → Position
    // For single layer, single component, this is just resolution order
    for resolution in resolution_packets.iter_mut() {
        let packet = form_packet(resolution);
        tile_data.extend_from_slice(&packet);
    }

    tile_data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::j2c::tag_tree::{TagNode, TagTree};
    use crate::reader::BitReader;

    fn decode_num_ht_coding_passes_for_test(data: &[u8]) -> Option<u8> {
        let mut reader = BitReader::new(data);
        let mut num_passes = 1u32;

        if reader.read_bits_with_stuffing(1)? == 1 {
            num_passes = 2;

            if reader.read_bits_with_stuffing(1)? == 1 {
                let extension = reader.read_bits_with_stuffing(2)?;
                num_passes = 3 + extension;

                if extension == 3 {
                    let extension = reader.read_bits_with_stuffing(5)?;
                    num_passes = 6 + extension;

                    if extension == 31 {
                        num_passes = 37 + reader.read_bits_with_stuffing(7)?;
                    }
                }
            }
        }

        Some(num_passes as u8)
    }

    fn decode_num_coding_passes_for_test(data: &[u8]) -> Option<u8> {
        let mut reader = BitReader::new(data);
        decode_num_coding_passes_from_reader_for_test(&mut reader)
    }

    fn decode_num_coding_passes_from_reader_for_test(reader: &mut BitReader<'_>) -> Option<u8> {
        let passes = if reader.peak_bits_with_stuffing(9) == Some(0x1ff) {
            reader.read_bits_with_stuffing(9)?;
            reader.read_bits_with_stuffing(7)? + 37
        } else if reader.peak_bits_with_stuffing(4) == Some(0x0f) {
            reader.read_bits_with_stuffing(4)?;
            reader.read_bits_with_stuffing(5)? + 6
        } else if reader.peak_bits_with_stuffing(4) == Some(0b1110) {
            reader.read_bits_with_stuffing(4)?;
            5
        } else if reader.peak_bits_with_stuffing(4) == Some(0b1101) {
            reader.read_bits_with_stuffing(4)?;
            4
        } else if reader.peak_bits_with_stuffing(4) == Some(0b1100) {
            reader.read_bits_with_stuffing(4)?;
            3
        } else if reader.peak_bits_with_stuffing(2) == Some(0b10) {
            reader.read_bits_with_stuffing(2)?;
            2
        } else if reader.peak_bits_with_stuffing(1) == Some(0) {
            reader.read_bits_with_stuffing(1)?;
            1
        } else {
            return None;
        };
        Some(passes as u8)
    }

    #[test]
    fn test_empty_packet() {
        let mut resolution = ResolutionPacket {
            subbands: vec![SubbandPrecinct {
                code_blocks: vec![CodeBlockPacketData {
                    data: Vec::new(),
                    num_coding_passes: 0,
                    num_zero_bitplanes: 31,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode: BlockCodingMode::Classic,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };

        let packet = form_packet(&mut resolution);
        assert!(!packet.is_empty());
    }

    #[test]
    fn test_non_empty_packet() {
        let mut resolution = ResolutionPacket {
            subbands: vec![SubbandPrecinct {
                code_blocks: vec![CodeBlockPacketData {
                    data: vec![0x12, 0x34, 0x56],
                    num_coding_passes: 1,
                    num_zero_bitplanes: 20,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode: BlockCodingMode::Classic,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };

        let packet = form_packet(&mut resolution);
        assert!(packet.len() >= 3);
    }

    #[test]
    fn packet_header_round_trips_varied_8x8_codeblock_lengths() {
        let zero_bitplanes = [
            2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 3, 2, 1, 1, 1, 1, 2, 3, 2, 2, 1,
            1, 1, 1, 2, 3, 2, 2, 1, 1, 1, 1, 2, 2, 2, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2,
            2, 2, 2, 1, 1, 1,
        ];
        let lengths = [
            1901, 2062, 1895, 2329, 2860, 2842, 2852, 2836, 2174, 2121, 1878, 2197, 2877, 2870,
            2854, 2862, 2097, 2143, 1906, 2059, 2724, 2879, 2860, 2847, 1928, 1967, 2105, 2318,
            2605, 2911, 2892, 2860, 1998, 1995, 2073, 2075, 2339, 2935, 2896, 2897, 1877, 1938,
            1841, 2000, 2271, 2877, 2826, 2828, 2098, 1899, 1953, 2061, 2135, 2886, 2869, 2909,
            2168, 1921, 1966, 2048, 2159, 2792, 2853, 2815,
        ];
        let mut resolution = ResolutionPacket {
            subbands: vec![SubbandPrecinct {
                code_blocks: zero_bitplanes
                    .iter()
                    .copied()
                    .zip(lengths.iter().copied())
                    .map(|(num_zero_bitplanes, len)| CodeBlockPacketData {
                        data: vec![0; len],
                        num_coding_passes: 1 + 3 * (8 - num_zero_bitplanes) - 2,
                        num_zero_bitplanes,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: BlockCodingMode::Classic,
                    })
                    .collect(),
                num_cbs_x: 8,
                num_cbs_y: 8,
            }],
        };

        let packet = form_packet(&mut resolution);
        let body_len: usize = lengths.iter().sum();
        let header_len = packet.len() - body_len;
        let mut reader = BitReader::new(&packet[..header_len]);
        assert_eq!(reader.read_bits_with_stuffing(1), Some(1));

        let mut inclusion_nodes = Vec::<TagNode>::new();
        let mut inclusion_tree = TagTree::new(8, 8, &mut inclusion_nodes);
        let mut zbp_nodes = Vec::<TagNode>::new();
        let mut zbp_tree = TagTree::new(8, 8, &mut zbp_nodes);

        for (idx, (&expected_zbp, &expected_len)) in
            zero_bitplanes.iter().zip(lengths.iter()).enumerate()
        {
            let x = idx as u32 % 8;
            let y = idx as u32 / 8;
            let included = inclusion_tree
                .read(x, y, &mut reader, 1, &mut inclusion_nodes)
                .expect("inclusion tag")
                == 0;
            assert!(included, "inclusion at index {idx}");

            let actual_zbp = zbp_tree
                .read(x, y, &mut reader, u32::MAX, &mut zbp_nodes)
                .expect("zero bitplane tag");
            assert_eq!(actual_zbp, u32::from(expected_zbp), "zbp at index {idx}");

            let passes = decode_num_coding_passes_from_reader_for_test(&mut reader)
                .expect("number of coding passes");
            let mut l_block = 3u32;
            while reader.read_bits_with_stuffing(1).expect("lblock increment") == 1 {
                l_block += 1;
            }
            let length_bits = l_block + u32::from(passes).ilog2();
            let actual_len = reader
                .read_bits_with_stuffing(length_bits as u8)
                .expect("code-block length");
            assert_eq!(actual_len, expected_len as u32, "length at index {idx}");
        }
    }

    #[test]
    fn packet_header_trailing_ff_stuffs_zero_before_body() {
        for len in 1..4096 {
            let mut resolution = ResolutionPacket {
                subbands: vec![SubbandPrecinct {
                    code_blocks: vec![CodeBlockPacketData {
                        data: vec![0x80; len],
                        num_coding_passes: 1,
                        num_zero_bitplanes: 0,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: BlockCodingMode::Classic,
                    }],
                    num_cbs_x: 1,
                    num_cbs_y: 1,
                }],
            };

            let packet = form_packet(&mut resolution);
            let header_len = packet.len() - len;
            let has_boundary_ff = packet[header_len - 1] == 0xff
                || (header_len >= 2
                    && packet[header_len - 2] == 0xff
                    && packet[header_len - 1] == 0x00);

            if !has_boundary_ff {
                continue;
            }

            let mut reader = BitReader::new(&packet);
            assert_eq!(reader.read_bits_with_stuffing(1), Some(1));

            let mut inclusion_nodes = Vec::<TagNode>::new();
            let mut inclusion_tree = TagTree::new(1, 1, &mut inclusion_nodes);
            let included = inclusion_tree
                .read(0, 0, &mut reader, 1, &mut inclusion_nodes)
                .expect("inclusion tag")
                == 0;
            assert!(included);

            let mut zbp_nodes = Vec::<TagNode>::new();
            let mut zbp_tree = TagTree::new(1, 1, &mut zbp_nodes);
            assert_eq!(
                zbp_tree
                    .read(0, 0, &mut reader, u32::MAX, &mut zbp_nodes)
                    .expect("zero bitplane tag"),
                0
            );

            let passes = decode_num_coding_passes_from_reader_for_test(&mut reader)
                .expect("number of coding passes");
            assert_eq!(passes, 1);

            let mut l_block = 3u32;
            while reader.read_bits_with_stuffing(1).expect("lblock increment") == 1 {
                l_block += 1;
            }
            let actual_len = reader
                .read_bits_with_stuffing(l_block as u8)
                .expect("code-block length");
            assert_eq!(actual_len, len as u32);

            reader.align();
            let expected_body = vec![0x80; len];
            assert_eq!(reader.offset(), header_len);
            assert_eq!(reader.read_bytes(len), Some(expected_body.as_slice()));
            return;
        }

        panic!("did not find a packet header ending in 0xff");
    }

    #[test]
    fn test_multi_subband_packet() {
        let mut resolution = ResolutionPacket {
            subbands: vec![
                SubbandPrecinct {
                    code_blocks: vec![CodeBlockPacketData {
                        data: vec![0x10, 0x20],
                        num_coding_passes: 1,
                        num_zero_bitplanes: 20,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: BlockCodingMode::Classic,
                    }],
                    num_cbs_x: 1,
                    num_cbs_y: 1,
                },
                SubbandPrecinct {
                    code_blocks: vec![CodeBlockPacketData {
                        data: vec![0x30, 0x40],
                        num_coding_passes: 1,
                        num_zero_bitplanes: 22,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: BlockCodingMode::Classic,
                    }],
                    num_cbs_x: 1,
                    num_cbs_y: 1,
                },
                SubbandPrecinct {
                    code_blocks: vec![CodeBlockPacketData {
                        data: vec![0x50],
                        num_coding_passes: 1,
                        num_zero_bitplanes: 24,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: BlockCodingMode::Classic,
                    }],
                    num_cbs_x: 1,
                    num_cbs_y: 1,
                },
            ],
        };

        let packet = form_packet(&mut resolution);
        // Should contain all 5 bytes of code-block data
        assert!(packet.len() >= 5);
    }

    #[test]
    fn test_encode_num_passes() {
        let mut w = BitWriter::new();
        encode_num_coding_passes(1, &mut w);
        let d = w.finish();
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn test_encode_num_passes_round_trip() {
        for num_passes in [1u8, 2, 3, 4, 5, 6, 19, 37, 38, 100, 164] {
            let mut w = BitWriter::new();
            encode_num_coding_passes(num_passes, &mut w);
            let data = w.finish();
            assert_eq!(decode_num_coding_passes_for_test(&data), Some(num_passes));
        }
    }

    #[test]
    fn test_encode_num_ht_passes_round_trip() {
        for num_passes in [1u8, 2, 3, 4, 5, 6, 19, 37, 38, 100, 164] {
            let mut w = BitWriter::new();
            encode_num_ht_coding_passes(num_passes, &mut w);
            let data = w.finish();
            assert_eq!(
                decode_num_ht_coding_passes_for_test(&data),
                Some(num_passes)
            );
        }
    }

    #[test]
    fn test_non_empty_ht_packet() {
        let mut resolution = ResolutionPacket {
            subbands: vec![SubbandPrecinct {
                code_blocks: vec![CodeBlockPacketData {
                    data: vec![0x12, 0x34, 0x56],
                    num_coding_passes: 1,
                    num_zero_bitplanes: 20,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode: BlockCodingMode::HighThroughput,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };

        let packet = form_packet(&mut resolution);
        assert!(packet.len() >= 3);
    }
}
