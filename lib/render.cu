#include "common.h"
#include "ffi.h"
#include "impls.h"
#include "render.h"

#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

void render::fwd(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
) {
    cudaError_t cuda_err;

    const auto &d =
        *unpack_descriptor<render::FwdDescriptor>(opaque, opaque_len);
    const auto args = render::unpack_fwd_args(buffers);

    const dim3 grid_dim_2d = {
        (d.img_shape.x + d.block_width - 1) / d.block_width,
        (d.img_shape.x + d.block_width - 1) / d.block_width,
        1
    };
    const int num_tiles = grid_dim_2d.x * grid_dim_2d.y;

    impls::project_fwd(stream, d, args);

    int num_intersects;
    int *cum_tiles_hit;
    cuda_err =
        cudaMalloc(&cum_tiles_hit, sizeof(*cum_tiles_hit) * d.num_points);
    throw_if_cuda_error(cuda_err);

    impls::compute_cumulative_intersects(
        stream,
        d.num_points,
        args.out.num_tiles_hit,
        num_intersects,
        cum_tiles_hit
    );

    std::cout << num_intersects << std::endl;

    impls::bin_and_sort_gaussians(
        stream,
        d.num_points,
        num_intersects,
        args.out.xys,
        args.out.depths,
        args.out.radii,
        cum_tiles_hit,
        grid_dim_2d,
        d.block_width,
        args.out.gaussian_ids_sorted,
        args.out.tile_bins
    );

    impls::rasterize_fwd(stream, d, args);

    cuda_err = cudaFree(cum_tiles_hit);
    throw_if_cuda_error(cuda_err);
}

void render::bwd(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
) {
    const auto &d =
        *unpack_descriptor<render::BwdDescriptor>(opaque, opaque_len);
    const auto &args = render::unpack_bwd_args(buffers);

    impls::rasterize_bwd(stream, d, args);
    impls::project_bwd(stream, d, args);
}

render::FwdArgs render::unpack_fwd_args(void **buffers) {
    FwdArgs args;
    std::size_t idx = 0;

    args.in.mean3ds = static_cast<float3 *>(buffers[idx++]);
    args.in.scales = static_cast<float3 *>(buffers[idx++]);
    args.in.quats = static_cast<float4 *>(buffers[idx++]);
    args.in.viewmat = static_cast<float *>(buffers[idx++]);
    args.in.colors = static_cast<float3 *>(buffers[idx++]);
    args.in.opacities = static_cast<float *>(buffers[idx++]);
    args.in.background = static_cast<float3 *>(buffers[idx++]);

    args.out.cov3ds = static_cast<float *>(buffers[idx++]);
    args.out.xys = static_cast<float2 *>(buffers[idx++]);
    args.out.depths = static_cast<float *>(buffers[idx++]);
    args.out.radii = static_cast<int *>(buffers[idx++]);
    args.out.conics = static_cast<float3 *>(buffers[idx++]);
    args.out.compensation = static_cast<float *>(buffers[idx++]);
    args.out.num_tiles_hit = static_cast<int *>(buffers[idx++]);
    args.out.gaussian_ids_sorted = static_cast<int *>(buffers[idx++]);
    args.out.tile_bins = static_cast<int2 *>(buffers[idx++]);
    args.out.out_img = static_cast<float3 *>(buffers[idx++]);
    args.out.final_Ts = static_cast<float *>(buffers[idx++]);
    args.out.final_idx = static_cast<int *>(buffers[idx++]);

    return args;
}

render::BwdArgs render::unpack_bwd_args(void **buffers) {
    BwdArgs args;
    std::size_t idx = 0;

    args.in.mean3ds = static_cast<float3 *>(buffers[idx++]);
    args.in.scales = static_cast<float3 *>(buffers[idx++]);
    args.in.quats = static_cast<float4 *>(buffers[idx++]);
    args.in.viewmat = static_cast<float *>(buffers[idx++]);
    args.in.colors = static_cast<float3 *>(buffers[idx++]);
    args.in.opacities = static_cast<float *>(buffers[idx++]);
    args.in.background = static_cast<float3 *>(buffers[idx++]);
    args.in.cov3ds = static_cast<float *>(buffers[idx++]);
    args.in.xys = static_cast<float2 *>(buffers[idx++]);
    args.in.radii = static_cast<int *>(buffers[idx++]);
    args.in.conics = static_cast<float3 *>(buffers[idx++]);
    args.in.compensation = static_cast<float *>(buffers[idx++]);
    args.in.gaussian_ids_sorted = static_cast<int *>(buffers[idx++]);
    args.in.tile_bins = static_cast<int2 *>(buffers[idx++]);
    args.in.final_Ts = static_cast<float *>(buffers[idx++]);
    args.in.final_idx = static_cast<int *>(buffers[idx++]);
    args.in.v_out_img = static_cast<float3 *>(buffers[idx++]);
    args.in.v_out_img_alpha = static_cast<float *>(buffers[idx++]);
    args.in.v_compensation = static_cast<float *>(buffers[idx++]);

    args.out.v_mean3d = static_cast<float3 *>(buffers[idx++]);
    args.out.v_scale = static_cast<float3 *>(buffers[idx++]);
    args.out.v_quat = static_cast<float4 *>(buffers[idx++]);
    args.out.v_colors = static_cast<float3 *>(buffers[idx++]);
    args.out.v_opacity = static_cast<float *>(buffers[idx++]);
    args.out.v_cov2d = static_cast<float3 *>(buffers[idx++]);
    args.out.v_xy = static_cast<float2 *>(buffers[idx++]);
    args.out.v_xy_abs = static_cast<float2 *>(buffers[idx++]);
    args.out.v_depth = static_cast<float *>(buffers[idx++]);
    args.out.v_conic = static_cast<float3 *>(buffers[idx++]);
    args.out.v_cov3d = static_cast<float *>(buffers[idx++]);

    return args;
}
