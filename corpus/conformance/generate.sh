#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Regenerate libjpeg-turbo reference fixtures. Run once, commit the output.
# CI does NOT run this — it only reads the committed files.
#
# Requires: libjpeg-turbo installed with `cjpeg` and `djpeg` on $PATH.
#   macOS:  brew install jpeg-turbo
#   Linux:  apt-get install libjpeg-turbo-progs
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v cjpeg >/dev/null || ! command -v djpeg >/dev/null; then
    echo "error: cjpeg/djpeg not on PATH; install libjpeg-turbo" >&2
    exit 1
fi
LJT_VERSION=$(cjpeg -version 2>&1 | head -1 || true)
echo "Using: $LJT_VERSION"

# --- Fixture 1: baseline 4:2:0, 16x16 ---------------------------------------
python3 - <<'PY' > gradient_16x16.ppm
import sys
header = b"P6\n16 16\n255\n"
body = bytearray()
for y in range(16):
    for x in range(16):
        body.extend([
            (x * 16) & 0xFF,
            (y * 16) & 0xFF,
            ((x + y) * 8) & 0xFF,
        ])
sys.stdout.buffer.write(header + bytes(body))
PY

cjpeg -quality 90 -sample 2x2,1x1,1x1 -baseline -optimize \
    -outfile baseline_420_16x16.jpg gradient_16x16.ppm
# djpeg writes PPM (P6 header + raw RGB). Strip the header to get raw bytes.
djpeg -rgb -outfile baseline_420_16x16.ppm baseline_420_16x16.jpg
python3 - <<'PY'
with open("baseline_420_16x16.ppm", "rb") as f:
    data = f.read()
# PPM P6 header: magic\nW H\nmaxval\n  (three newline-terminated lines)
i = 0
for _ in range(3):
    i = data.index(b"\n", i) + 1
with open("baseline_420_16x16.rgb", "wb") as f:
    f.write(data[i:])
PY
rm -f baseline_420_16x16.ppm
test $(wc -c < baseline_420_16x16.rgb) -eq 768

# --- Fixture 2: grayscale 8x8 -----------------------------------------------
python3 - <<'PY' > gray_8x8.pgm
import sys
header = b"P5\n8 8\n255\n"
body = bytes((x * 32) & 0xFF for _ in range(8) for x in range(8))
sys.stdout.buffer.write(header + body)
PY
cjpeg -quality 90 -grayscale -baseline -optimize \
    -outfile grayscale_8x8.jpg gray_8x8.pgm
# djpeg writes PGM (P5 header + raw 8-bit). Strip the header.
djpeg -grayscale -outfile grayscale_8x8.pgm grayscale_8x8.jpg
python3 - <<'PY'
with open("grayscale_8x8.pgm", "rb") as f:
    data = f.read()
i = 0
for _ in range(3):
    i = data.index(b"\n", i) + 1
with open("grayscale_8x8.gray", "wb") as f:
    f.write(data[i:])
PY
rm -f grayscale_8x8.pgm
test $(wc -c < grayscale_8x8.gray) -eq 64

cat > manifest.json <<EOF
{
  "libjpeg_turbo_version": "$LJT_VERSION",
  "generated_on": "$(date -u +%FT%TZ)",
  "fixtures": [
    {
      "input":     "baseline_420_16x16.jpg",
      "reference": "baseline_420_16x16.rgb",
      "format":    "Rgb8",
      "width":     16,
      "height":    16,
      "tolerance": "bit_exact",
      "sampling":  "4:2:0"
    },
    {
      "input":     "grayscale_8x8.jpg",
      "reference": "grayscale_8x8.gray",
      "format":    "Gray8",
      "width":     8,
      "height":    8,
      "tolerance": "bit_exact",
      "sampling":  "grayscale"
    }
  ]
}
EOF

rm -f gradient_16x16.ppm gray_8x8.pgm

echo "Regenerated fixtures:"
ls -la *.jpg *.rgb *.gray manifest.json
