extern "C" __global__ void signinum_j2k_forward_rct(
    float *plane0,
    float *plane1,
    float *plane2,
    unsigned long long len
) {
    const unsigned long long idx =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }

    const float r = plane0[idx];
    const float g = plane1[idx];
    const float b = plane2[idx];
    plane0[idx] = floorf((r + 2.0f * g + b) * 0.25f);
    plane1[idx] = b - g;
    plane2[idx] = r - g;
}

__device__ float signinum_j2k_fdwt53_predict_row(
    const float *src,
    unsigned int row_base,
    unsigned int width,
    unsigned int high_index
) {
    const unsigned int odd = high_index * 2u + 1u;
    const unsigned int last_even = (width % 2u == 0u) ? width - 2u : width - 1u;
    const float left = src[row_base + odd - 1u];
    const float right = (odd + 1u < width) ? src[row_base + odd + 1u] : src[row_base + last_even];
    return src[row_base + odd] - floorf((left + right) * 0.5f);
}

__device__ float signinum_j2k_fdwt53_predict_col(
    const float *src,
    unsigned int x,
    unsigned int full_width,
    unsigned int height,
    unsigned int high_index
) {
    const unsigned int odd = high_index * 2u + 1u;
    const unsigned int last_even = (height % 2u == 0u) ? height - 2u : height - 1u;
    const float top = src[(odd - 1u) * full_width + x];
    const float bottom = (odd + 1u < height)
        ? src[(odd + 1u) * full_width + x]
        : src[last_even * full_width + x];
    return src[odd * full_width + x] - floorf((top + bottom) * 0.5f);
}

extern "C" __global__ void signinum_j2k_forward_dwt53_horizontal(
    const float *src,
    float *dst,
    unsigned int full_width,
    unsigned int current_width,
    unsigned int current_height,
    unsigned int low_width
) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= current_width || y >= current_height) {
        return;
    }

    const unsigned int row_base = y * full_width;
    if (x < low_width) {
        const unsigned int even = x * 2u;
        const float left = x > 0u
            ? signinum_j2k_fdwt53_predict_row(src, row_base, current_width, x - 1u)
            : signinum_j2k_fdwt53_predict_row(src, row_base, current_width, 0u);
        const float right = even + 1u < current_width
            ? signinum_j2k_fdwt53_predict_row(src, row_base, current_width, x)
            : left;
        dst[row_base + x] = src[row_base + even] + floorf((left + right) * 0.25f + 0.5f);
        return;
    }

    dst[row_base + x] = signinum_j2k_fdwt53_predict_row(
        src,
        row_base,
        current_width,
        x - low_width
    );
}

extern "C" __global__ void signinum_j2k_forward_dwt53_vertical(
    const float *src,
    float *dst,
    unsigned int full_width,
    unsigned int current_width,
    unsigned int current_height,
    unsigned int low_height
) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= current_width || y >= current_height) {
        return;
    }

    if (y < low_height) {
        const unsigned int even = y * 2u;
        const float top = y > 0u
            ? signinum_j2k_fdwt53_predict_col(src, x, full_width, current_height, y - 1u)
            : signinum_j2k_fdwt53_predict_col(src, x, full_width, current_height, 0u);
        const float bottom = even + 1u < current_height
            ? signinum_j2k_fdwt53_predict_col(src, x, full_width, current_height, y)
            : top;
        dst[y * full_width + x] =
            src[even * full_width + x] + floorf((top + bottom) * 0.25f + 0.5f);
        return;
    }

    dst[y * full_width + x] = signinum_j2k_fdwt53_predict_col(
        src,
        x,
        full_width,
        current_height,
        y - low_height
    );
}
