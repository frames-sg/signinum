# slidecodec-jpeg

Core library crate for `slidecodec`. See the top-level [README](../../README.md)
for project positioning and MSRV.

```rust
use slidecodec_jpeg::{Decoder, JpegView, JpegError, RgbRowSink};

let info = Decoder::inspect(bytes)?;
println!("{}×{} {:?}", info.dimensions.0, info.dimensions.1, info.sof_kind);

let view = JpegView::parse(bytes)?;
let decoder = Decoder::from_view(view)?;

struct Sink;

impl RgbRowSink for Sink {
    fn write_rgb_row(&mut self, _y: u32, _row: &[u8]) -> Result<(), JpegError> {
        Ok(())
    }
}

decoder.decode_rows(&mut Sink)?;
```

Current decode targets are native `x86_64` and `aarch64` hosts.
