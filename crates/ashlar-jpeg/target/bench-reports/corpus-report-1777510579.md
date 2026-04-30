# Ashlar JPEG corpus report

- inputs: 2
- rows: 11
- iterations per measurement: 3
- tie threshold: 1%

## Summary by operation

| operation | ashlar fastest | vs jpeg wins | vs jpeg losses | vs zune wins | vs zune losses | failures |
|---|---:|---:|---:|---:|---:|---:|
| decode_gray | 1 | 1 | 0 | 1 | 0 | 0 |
| decode_rgb | 1 | 1 | 0 | 1 | 0 | 0 |
| inspect | 2 | 2 | 0 | 2 | 0 | 0 |
| wsi_region_rgb | 1 | 1 | 0 | 1 | 0 | 0 |
| wsi_region_scaled_rgb_q4 | 1 | 1 | 0 | 1 | 0 | 0 |
| wsi_region_scaled_rgb_q8 | 1 | 1 | 0 | 1 | 0 | 0 |
| wsi_scaled_rgb_q4 | 1 | 1 | 0 | 1 | 0 | 0 |
| wsi_scaled_rgb_q8 | 1 | 1 | 0 | 1 | 0 | 0 |
| wsi_tile_batch_region_scaled_rgb_q4 | 1 | 1 | 0 | 1 | 0 | 0 |
| wsi_tile_batch_scaled_rgb_q4 | 1 | 1 | 0 | 1 | 0 | 0 |

## Rows where ashlar is not fastest

| input | operation | ashlar | jpeg-decoder | zune-jpeg | fastest |
|---|---|---:|---:|---:|---|
| none | — | — | — | — | — |

## Failures / skips

| input | operation | ashlar | jpeg-decoder | zune-jpeg |
|---|---|---|---|---|
| none | — | — | — | — |
