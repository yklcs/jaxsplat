#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace render {

void fwd(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
);

struct FwdArgs {
    struct In {
        // geometry
        const float3 *mean3ds;
        const float3 *scales;
        const float4 *quats;
        const float *viewmat;

        // appearance
        const float3 *colors;
        const float *opacities;
        const float3 *background;
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

        // sorting and binning
        int *gaussian_ids_sorted;
        int2 *tile_bins;

        // rasterization
        float3 *out_img;
        float *final_Ts;
        int *final_idx;
    } out;
};

FwdArgs unpack_fwd_args(void **buffers);

struct FwdDescriptor {
    unsigned num_points;
    float glob_scale;
    float2 f;
    float2 c;
    dim3 img_shape;
    unsigned block_width;
    float clip_thresh;
};

void bwd(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
);

struct BwdArgs {
    struct In {
        // geometry
        const float3 *mean3ds;
        const float3 *scales;
        const float4 *quats;
        const float *viewmat;

        // appearance
        const float3 *colors;
        const float *opacities;
        const float3 *background;

        // projection output
        const float *cov3ds;
        const float2 *xys;
        const int *radii;
        const float3 *conics;
        const float *compensation;

        // sorting and binning
        const int *gaussian_ids_sorted;
        const int2 *tile_bins;

        // rasterization output
        const float *final_Ts;
        const int *final_idx;

        // vjps
        const float3 *v_out_img;
        const float *v_out_img_alpha;
        const float *v_compensation;
    } in;
    struct Out {
        // geometry
        float3 *v_mean3d;
        float3 *v_scale;
        float4 *v_quat;

        // appearance
        float3 *v_colors;
        float *v_opacity;

        // projection
        float3 *v_cov2d;
        float2 *v_xy;
        float2 *v_xy_abs;
        float *v_depth;
        float3 *v_conic;
        float *v_cov3d;
    } out;
};

BwdArgs unpack_bwd_args(void **buffers);

struct BwdDescriptor {
    unsigned num_points;
    float glob_scale;
    float2 f;
    float2 c;
    dim3 img_shape;
    unsigned block_width;
};

} // namespace rasterize
