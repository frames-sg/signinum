# signinum-jpeg

JPEG tile inspect and CPU decode for whole-slide imaging workloads.
The baseline JPEG encoder is kept for generated fixtures and explicit fallback
or derived-output use. WSI/DICOM storage conversion should prefer compressed
tile passthrough, or lossless JPEG 2000 / HTJ2K encode when a new diagnostic
codestream is required.

Install:

```sh
cargo add signinum-jpeg
```

Use this crate when you need codec primitives directly. Use
[`statumen`](https://github.com/frames-sg/statumen) when you need a whole-slide
reader/container layer.

```rust
use signinum_jpeg::{Decoder, JpegError, JpegView, RowSink};

let info = Decoder::inspect(bytes)?;
println!(
    "{}×{} {:?} mcu={:?} restart={:?}",
    info.dimensions.0,
    info.dimensions.1,
    info.sof_kind,
    info.mcu_geometry,
    info.restart_interval
);

let view = JpegView::parse(bytes)?;
if let Some(candidate) = view.passthrough_candidate() {
    println!(
        "passthrough syntax={:?} payload={:?}",
        candidate.transfer_syntax(),
        candidate.payload_kind()
    );
}
if let Some(index) = view.restart_index()? {
    println!("restart segments={}", index.segments.len());
}
let decoder = Decoder::from_view(view)?;

struct Sink;

impl RowSink<u8> for Sink {
    type Error = JpegError;

    fn write_row(&mut self, _y: u32, _row: &[u8]) -> Result<(), JpegError> {
        Ok(())
    }
}

decoder.decode_rows(&mut Sink)?;
```

For WSI viewers decoding many independent JPEG tiles per frame, use the
production batch APIs. `TileBatchOptions::default()` uses host parallelism and
one reusable decoder context plus scratch pool per worker. The `_with_options`
variants are the intended path when TIFF or DICOM metadata has already resolved
ambiguous three-component JPEG data to RGB or YCbCr.

```rust
use signinum_jpeg::{
    decode_tiles_into_with_options, ColorTransform, DecodeOptions, PixelFormat,
    TileBatchOptions, TileDecodeJob,
};

let decode_options = DecodeOptions::default().with_color_transform(ColorTransform::ForceYCbCr);
let stride = tile_width as usize * 3;
let mut outputs = compressed_tiles
    .iter()
    .map(|_| vec![0_u8; stride * tile_height as usize])
    .collect::<Vec<_>>();
let mut jobs = compressed_tiles
    .iter()
    .zip(outputs.iter_mut())
    .map(|(input, out)| TileDecodeJob {
        input: input.as_ref(),
        out: out.as_mut_slice(),
        stride,
    })
    .collect::<Vec<_>>();

decode_tiles_into_with_options(
    &mut jobs,
    PixelFormat::Rgb8,
    decode_options,
    TileBatchOptions::default(),
)?;
```

Current decode targets are native `x86_64` and `aarch64` hosts.
