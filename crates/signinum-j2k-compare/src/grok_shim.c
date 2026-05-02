#include "grok.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void signinum_grok_init_once(void) {
  static int initialized = 0;
  if (!initialized) {
    grk_initialize(NULL, 1, NULL);
    initialized = 1;
  }
}

static uint8_t signinum_clamp_u8(int32_t value) {
  if (value < 0) {
    return 0;
  }
  if (value > 255) {
    return 255;
  }
  return (uint8_t)value;
}

int signinum_grok_decode_u8(const uint8_t *bytes, size_t len, uint32_t reduce,
                              int has_region, uint32_t x0, uint32_t y0,
                              uint32_t x1, uint32_t y1, uint32_t channels,
                              uint8_t **out_data, size_t *out_len,
                              uint32_t *out_width, uint32_t *out_height) {
  grk_object *codec = NULL;
  grk_image *image = NULL;
  grk_stream_params stream_params;
  grk_decompress_parameters params;
  grk_header_info header_info;
  uint8_t *packed = NULL;

  if (!bytes || !out_data || !out_len || !out_width || !out_height) {
    return 0;
  }

  *out_data = NULL;
  *out_len = 0;
  *out_width = 0;
  *out_height = 0;

  signinum_grok_init_once();
  memset(&stream_params, 0, sizeof(stream_params));
  memset(&params, 0, sizeof(params));
  memset(&header_info, 0, sizeof(header_info));

  stream_params.buf = (uint8_t *)bytes;
  stream_params.buf_len = len;
  stream_params.stream_len = len;
  stream_params.is_read_stream = true;

  params.core.reduce = (uint8_t)reduce;
  params.force_rgb = channels == 3;
  params.upsample = channels == 3;
  params.num_threads = 1;
  if (has_region) {
    params.dw_x0 = x0;
    params.dw_y0 = y0;
    params.dw_x1 = x1;
    params.dw_y1 = y1;
  }

  header_info.color_space = channels == 3 ? GRK_CLRSPC_SRGB : GRK_CLRSPC_GRAY;
  header_info.decompress_fmt = GRK_FMT_PXM;
  header_info.force_rgb = channels == 3;
  header_info.upsample = channels == 3;

  codec = grk_decompress_init(&stream_params, &params);
  if (!codec) {
    return 0;
  }
  if (!grk_decompress_read_header(codec, &header_info)) {
    grk_object_unref(codec);
    return 0;
  }
  if (!grk_decompress(codec, NULL)) {
    grk_object_unref(codec);
    return 0;
  }

  image = grk_decompress_get_image(codec);
  if (!image || image->numcomps == 0 || !image->comps) {
    grk_object_unref(codec);
    return 0;
  }

  uint32_t width = image->decompress_width ? image->decompress_width : image->comps[0].w;
  uint32_t height = image->decompress_height ? image->decompress_height : image->comps[0].h;
  size_t total = (size_t)width * (size_t)height * (size_t)channels;
  packed = (uint8_t *)malloc(total);
  if (!packed) {
    grk_object_unref(codec);
    return 0;
  }

  for (uint32_t row = 0; row < height; ++row) {
    for (uint32_t col = 0; col < width; ++col) {
      size_t dst = ((size_t)row * width + col) * channels;
      grk_image_comp *c0 = &image->comps[0];
      int32_t v0 = ((int32_t *)c0->data)[(size_t)row * c0->stride + col];
      if (channels == 1) {
        packed[dst] = signinum_clamp_u8(v0);
        continue;
      }
      if (image->numcomps == 1) {
        uint8_t gray = signinum_clamp_u8(v0);
        packed[dst] = gray;
        packed[dst + 1] = gray;
        packed[dst + 2] = gray;
        continue;
      }
      grk_image_comp *c1 = &image->comps[1];
      grk_image_comp *c2 = &image->comps[2];
      int32_t v1 = ((int32_t *)c1->data)[(size_t)row * c1->stride + col];
      int32_t v2 = ((int32_t *)c2->data)[(size_t)row * c2->stride + col];
      packed[dst] = signinum_clamp_u8(v0);
      packed[dst + 1] = signinum_clamp_u8(v1);
      packed[dst + 2] = signinum_clamp_u8(v2);
    }
  }

  *out_data = packed;
  *out_len = total;
  *out_width = width;
  *out_height = height;
  grk_object_unref(codec);
  return 1;
}

void signinum_grok_free(void *ptr) { free(ptr); }
