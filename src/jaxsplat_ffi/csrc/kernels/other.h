#pragma once

#include <cuda_runtime.h>

namespace kernels {

__global__ void
tiled_memset(float *dst, size_t n_dst, const float *src, size_t n_src);

__global__ void fill_value(float *dst, size_t n, float value);

__global__ void fill_background(
    float3 *img, float *final_Ts, size_t pixels_per_image, size_t num_images,
    const float3 *background
);

__global__ void compute_cov2d_bounds(
    const unsigned num_points,
    const float *__restrict__ covs2d,
    float *__restrict__ conics,
    float *__restrict__ radii
);

} // namespace kernels
