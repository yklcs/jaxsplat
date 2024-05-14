#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace ops {

struct Descriptor {
    unsigned num_points;
    unsigned num_intersects;
    dim3 img_shape;

    float4 intrins;
    float glob_scale;
    float clip_thresh;

    unsigned block_width;
    unsigned grid_dim_1d;
    unsigned block_dim_1d;
    dim3 grid_dim_2d;
    dim3 block_dim_2d;
};

void cumsum(
    cudaStream_t stream,
    const int32_t *input,
    int32_t *output,
    const int num_items
);

void sort_and_bin(
    cudaStream_t stream,
    const Descriptor &d,
    const float2 *xys,
    const float *depths,
    const int *radii,
    const int *cum_tiles_hit,
    int *gaussian_ids_sorted,
    int2 *tile_bins
);

namespace project::fwd {

void xla(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
);
struct Tensors;
Tensors
unpack_tensors(cudaStream_t stream, const Descriptor &d, void **buffers);

} // namespace project::fwd

namespace project::bwd {

void xla(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
);
struct Tensors;
Tensors
unpack_tensors(cudaStream_t stream, const Descriptor &d, void **buffers);

} // namespace project::bwd

namespace rasterize::fwd {

void xla(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
);
struct Tensors;
Tensors
unpack_tensors(cudaStream_t stream, const Descriptor &d, void **buffers);

} // namespace rasterize::fwd

namespace rasterize::bwd {

void xla(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
);
struct Tensors;
Tensors
unpack_tensors(cudaStream_t stream, const Descriptor &d, void **buffers);

} // namespace rasterize::bwd

struct ops::project::fwd::Tensors {
    struct In {
        // geometry
        const float3 *mean3ds;
        const float3 *scales;
        const float4 *quats;
        const float *viewmat;
    } in;
    struct Out {
        // projection
        float *cov3ds;
        float2 *xys;
        float *depths;
        int *radii;
        float3 *conics;
        float *compensation;
        int *num_tiles_hit;
        int *cum_tiles_hit;
    } out;
};

struct project::bwd::Tensors {
    struct In {
        // geometry
        const float3 *mean3ds;
        const float3 *scales;
        const float4 *quats;
        const float *viewmat;

        // projection output
        const float *cov3ds;
        const float2 *xys;
        const int *radii;
        const float3 *conics;
        const float *compensation;

        const float *v_compensation;
        const float2 *v_xy;
        const float *v_depth;
        const float3 *v_conic;
    } in;
    struct Out {
        // geometry
        float3 *v_mean3d;
        float3 *v_scale;
        float4 *v_quat;

        // projection
        float3 *v_cov2d;
        float *v_cov3d;
    } out;
};

struct rasterize::fwd::Tensors {
    struct In {
        // appearance
        const float3 *colors;
        const float *opacities;
        const float3 *background;

        // projection output
        const float2 *xys;
        const float *depths;
        const int *radii;
        const float3 *conics;
        const int *cum_tiles_hit;
    } in;
    struct Out {
        // sort and bin
        int *gaussian_ids_sorted;
        int2 *tile_bins;

        // rasterization output
        float *final_Ts;
        int *final_idx;
        float3 *out_img;
    } out;
};

struct rasterize::bwd::Tensors {
    struct In {
        // appearance
        const float3 *colors;
        const float *opacities;
        const float3 *background;

        // projection output
        const float2 *xys;
        const float3 *conics;

        // sorting and binning
        const int *gaussian_ids_sorted;
        const int2 *tile_bins;

        // rasterization output
        const float *final_Ts;
        const int *final_idx;

        // vjps
        const float3 *v_out_img;
        const float *v_out_img_alpha;
    } in;
    struct Out {
        // appearance
        float3 *v_colors;
        float *v_opacity;

        // projection
        float2 *v_xy;
        float2 *v_xy_abs;
        float3 *v_conic;
    } out;
};

} // namespace ops
