#pragma once

#include <cuda_runtime.h>

namespace kernels {

__global__ void
tiled_memset(float *dst, size_t n_dst, const float *src, size_t n_src);

__global__ void compute_cov2d_bounds(
    const unsigned num_points,
    const float *__restrict__ covs2d,
    float *__restrict__ conics,
    float *__restrict__ radii
);

} // namespace kernels
