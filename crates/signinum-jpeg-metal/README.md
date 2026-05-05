# signinum-jpeg-metal

Apple Metal device-output adapter for `signinum-jpeg`.

Install this crate when a macOS pipeline needs JPEG tile output as a
Metal-backed `DeviceSurface`:

```sh
cargo add signinum-jpeg-metal
```

`BackendRequest::Auto` may choose a validated Metal path for supported JPEG
tile shapes and otherwise returns host-backed CPU output. `BackendRequest::Metal`
is strict: supported requests return resident Metal decode surfaces on macOS,
while unsupported shapes, CPU-staged upload fallbacks, or hosts without Metal
return an error. Check `Surface::residency()` when a caller needs to distinguish
`MetalResidentDecode` from explicit CPU-staged upload buffers.

The stable CPU decode API lives in `signinum-jpeg`. This adapter remains
pre-1.0 while runtime validation and routing policies continue to harden.
