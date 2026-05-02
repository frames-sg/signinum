# signinum-core

Shared CPU-first 1.0 decode contracts for the `signinum` workspace.

This crate contains the stable value types and traits used by the CPU codec
crates:

- pixel/sample formats
- ROI and downscale geometry
- caller-owned scratch and decoder context traits
- row streaming and tile-batch decode traits
- backend request and device-surface contracts

It contains no image-format parser or decoder.
