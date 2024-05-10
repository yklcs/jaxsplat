#pragma once

#include "render.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace impls {

void project_fwd(
    cudaStream_t stream,
    const render::FwdDescriptor &d,
    const render::FwdArgs &args
);

void project_bwd(
    cudaStream_t stream,
    const render::BwdDescriptor &d,
    const render::BwdArgs &args
);

void rasterize_fwd(
    cudaStream_t stream,
    const render::FwdDescriptor &d,
    const render::FwdArgs &args
);

void rasterize_bwd(
    cudaStream_t stream,
    const render::BwdDescriptor &d,
    const render::BwdArgs &args
);

void compute_cumulative_intersects(
    cudaStream_t stream,
    const int num_points,
    const int32_t *num_tiles_hit,
    int32_t &num_intersects,
    int32_t *cum_tiles_hit
);

void bin_and_sort_gaussians(
    cudaStream_t stream,
    const int num_points,
    const int num_intersects,
    const float2 *xys,
    const float *depths,
    const int *radii,
    const int32_t *cum_tiles_hit,
    const dim3 tile_bounds,
    const unsigned block_width,
    std::int32_t *gaussian_ids_sorted,
    int2 *tile_bins
);

} // namespace impls
