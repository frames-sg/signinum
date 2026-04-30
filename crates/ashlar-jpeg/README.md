# ashlar-jpeg

Core library crate for `ashlar`. See the top-level [README](../../README.md)
for project positioning and MSRV.

```rust
use ashlar_jpeg::{Decoder, JpegError, JpegView, RowSink};

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

Current decode targets are native `x86_64` and `aarch64` hosts.
