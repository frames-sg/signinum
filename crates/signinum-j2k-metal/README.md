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
output. `BackendRequest::Metal` is strict: it returns resident Metal decode
surfaces only, and reports unsupported or unavailable Metal requests as errors.
Use the explicit `decode_*_cpu_staged_metal_surface_with_session` APIs when
CPU-decoded bytes need to be uploaded into a Metal buffer.

The stable CPU decode API lives in `signinum-j2k`. This adapter remains
pre-1.0 while runtime validation and routing policies continue to harden.
