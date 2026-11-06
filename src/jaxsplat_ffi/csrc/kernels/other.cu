#include "helpers.h"
#include "other.h"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

__global__ void kernels::tiled_memset(
    float *dst,
    size_t n_dst,
    const float *src,
    size_t n_src
) {
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n_dst;
         tid += blockDim.x * gridDim.x) {
        dst[tid] = src[tid % n_src];
    }
}

__global__ void kernels::compute_cov2d_bounds(
    const unsigned num_points,
    const float *__restrict__ covs2d,
    float *__restrict__ conics,
    float *__restrict__ radii
) {
    unsigned row = cg::this_grid().thread_rank();
    if (row >= num_points) {
        return;
    }
    int index = row * 3;
    float3 conic;
    float radius;
    float3 cov2d{
        (float)covs2d[index],
        (float)covs2d[index + 1],
        (float)covs2d[index + 2]
    };
    helpers::compute_cov2d_bounds(cov2d, conic, radius);
    conics[index] = conic.x;
    conics[index + 1] = conic.y;
    conics[index + 2] = conic.z;
    radii[row] = radius;
}

__global__ void kernels::fill_value(float *dst, size_t n, float value) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n;
         tid += blockDim.x * gridDim.x) {
        dst[tid] = value;
    }
}

// Nothing visible: image = per-image background, transmittance = 1.
__global__ void kernels::fill_background(
    float3 *img,
    float *final_Ts,
    size_t pixels_per_image,
    size_t num_images,
    const float3 *background
) {
    const size_t n = pixels_per_image * num_images;
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n;
         tid += blockDim.x * gridDim.x) {
        img[tid] = background[tid / pixels_per_image];
        final_Ts[tid] = 1.f;
    }
}
