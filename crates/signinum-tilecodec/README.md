# signinum-tilecodec

Tile decompression primitives for pathology image containers.

The CPU-first 1.0 API provides `TileDecompress` implementations for Deflate,
Zstd, LZW, and Uncompressed payloads, with caller-owned scratch pools where a
codec benefits from reuse.
