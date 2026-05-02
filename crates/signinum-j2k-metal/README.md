# signinum-j2k-metal

Apple Metal device-output adapter for `signinum-j2k`.

Install this crate when a macOS pipeline needs JPEG 2000 / HTJ2K output as a
Metal-backed `DeviceSurface`:

```sh
cargo add signinum-j2k-metal
```

The adapter exposes full, ROI, reduced-resolution, and combined
ROI+reduced-resolution device surfaces. `BackendRequest::Auto` may choose a
validated Metal path for supported shapes and otherwise returns host-backed CPU
output. `BackendRequest::Metal` is strict and reports unsupported or
unavailable Metal requests as errors.

The stable CPU decode API lives in `signinum-j2k`. This adapter remains
pre-1.0 while runtime validation and routing policies continue to harden.
