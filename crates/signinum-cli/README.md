# signinum-cli

Command-line inspection utility for `signinum`.

Install:

```sh
cargo install signinum-cli
```

The CPU-first 1.0 CLI provides:

```sh
signinum inspect <file>
```

It parses JPEG and JPEG 2000 headers and prints decoded metadata. It does not
own WSI container parsing, caching, prefetch, or image decode workflows.
