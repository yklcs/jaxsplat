#include "impls.h"
#include "kernels/kernels.h"
#include "render.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

void impls::project_fwd(
    cudaStream_t stream,
    const render::FwdDescriptor &d,
    const render::FwdArgs &args
) {
    dim3 grid_dim_2d = {
        (d.img_shape.x + d.block_width - 1) / d.block_width,
        (d.img_shape.y + d.block_width - 1) / d.block_width,
        1
    };

    float4 intrins = {d.f.x, d.f.y, d.c.x, d.c.y};

    const unsigned block_dim_1d = d.block_width * d.block_width;
    const unsigned grid_dim_1d =
        (d.num_points + block_dim_1d - 1) / block_dim_1d;
    kernels::project_gaussians_fwd<<<grid_dim_1d, block_dim_1d, 0, stream>>>(
        // in
        d.num_points,
        args.in.mean3ds,
        args.in.scales,
        d.glob_scale,
        args.in.quats,
        args.in.viewmat,
        intrins,
        d.img_shape,
        grid_dim_2d,
        d.block_width,
        d.clip_thresh,

        // out
        args.out.cov3ds,
        args.out.xys,
        args.out.depths,
        args.out.radii,
        args.out.conics,
        args.out.compensation,
        args.out.num_tiles_hit
    );
}

void impls::project_bwd(
    cudaStream_t stream,
    const render::BwdDescriptor &d,
    const render::BwdArgs &args
) {
    float4 intrins = {d.f.x, d.f.y, d.c.x, d.c.y};
    const unsigned num_cov3d = d.num_points * 6;

    const unsigned block_dim_1d = d.block_width * d.block_width;
    const unsigned grid_dim_1d =
        (d.num_points + block_dim_1d - 1) / block_dim_1d;
    kernels::project_gaussians_bwd<<<grid_dim_1d, block_dim_1d, 0, stream>>>(
        // in
        d.num_points,
        args.in.mean3ds,
        args.in.scales,
        d.glob_scale,
        args.in.quats,
        args.in.viewmat,
        intrins,
        d.img_shape,
        args.in.cov3ds,
        args.in.radii,
        args.in.conics,
        args.in.compensation,
        args.out.v_xy,
        args.out.v_depth,
        args.out.v_conic,
        args.in.v_compensation,

        // out
        args.out.v_cov2d,
        args.out.v_cov3d,
        args.out.v_mean3d,
        args.out.v_scale,
        args.out.v_quat
    );
}

void impls::rasterize_fwd(
    cudaStream_t stream,
    const render::FwdDescriptor &d,
    const render::FwdArgs &args
) {
    dim3 grid_dim_2d = {
        (d.img_shape.x + d.block_width - 1) / d.block_width,
        (d.img_shape.y + d.block_width - 1) / d.block_width,
        1
    };
    dim3 block_dim_2d = {d.block_width, d.block_width, 1};
    kernels::rasterize_fwd<<<grid_dim_2d, block_dim_2d, 0, stream>>>(
        // in
        grid_dim_2d,
        d.img_shape,
        args.out.gaussian_ids_sorted,
        args.out.tile_bins,
        args.out.xys,
        args.out.conics,
        args.in.colors,
        args.in.opacities,

        // out
        args.out.final_Ts,
        args.out.final_idx,
        args.out.out_img,
        *args.in.background
    );
}

void impls::rasterize_bwd(
    cudaStream_t stream,
    const render::BwdDescriptor &d,
    const render::BwdArgs &args
) {
    dim3 grid_dim_2d = {
        (d.img_shape.x + d.block_width - 1) / d.block_width,
        (d.img_shape.y + d.block_width - 1) / d.block_width,
        1
    };
    dim3 block_dim_2d = {d.block_width, d.block_width, 1};

    kernels::rasterize_bwd<<<grid_dim_2d, block_dim_2d, 0, stream>>>(
        // in
        grid_dim_2d,
        d.img_shape,
        args.in.gaussian_ids_sorted,
        args.in.tile_bins,
        args.in.xys,
        args.in.conics,
        args.in.colors,
        args.in.opacities,
        *args.in.background,
        args.in.final_Ts,
        args.in.final_idx,
        args.in.v_out_img,
        args.in.v_out_img_alpha,

        // out
        args.out.v_xy,
        args.out.v_xy_abs,
        args.out.v_conic,
        args.out.v_colors,
        args.out.v_opacity
    );
}

void impls::compute_cumulative_intersects(
    cudaStream_t stream,
    const int num_points,
    const int32_t *num_tiles_hit,
    int32_t &num_intersects,
    int32_t *cum_tiles_hit
) {
    void *sum_ws = nullptr;
    size_t sum_ws_bytes;

    cub::DeviceScan::InclusiveSum(
        sum_ws,
        sum_ws_bytes,
        num_tiles_hit,
        cum_tiles_hit,
        num_points,
        stream
    );
    cudaMalloc(&sum_ws, sum_ws_bytes);

    cub::DeviceScan::InclusiveSum(
        sum_ws,
        sum_ws_bytes,
        num_tiles_hit,
        cum_tiles_hit,
        num_points,
        stream
    );

    cudaMemcpy(
        &num_intersects,
        &(cum_tiles_hit[num_points - 1]),
        sizeof(int32_t),
        cudaMemcpyDeviceToHost
    );
    cudaFree(sum_ws);
}

void impls::bin_and_sort_gaussians(
    cudaStream_t stream,
    const int num_points,
    const int num_intersects,
    const float2 *xys,
    const float *depths,
    const int *radii,
    const int *cum_tiles_hit,
    const dim3 tile_bounds,
    const unsigned block_width,
    std::int32_t *gaussian_ids_sorted,
    int2 *tile_bins
) {
    std::int32_t *gaussian_ids_unsorted;
    std::int64_t *isect_ids_unsorted;
    std::int64_t *isect_ids_sorted;
    cudaMalloc(
        &gaussian_ids_unsorted,
        num_intersects * sizeof(*gaussian_ids_unsorted)
    );
    cudaMalloc(
        &isect_ids_unsorted,
        num_intersects * sizeof(*isect_ids_unsorted)
    );
    cudaMalloc(&isect_ids_sorted, num_intersects * sizeof(*isect_ids_sorted));

    const unsigned block_dim_1d = block_width * block_width;
    const unsigned grid_dim_1d = (num_points + block_dim_1d - 1) / block_dim_1d;
    kernels::
        map_gaussian_to_intersects<<<grid_dim_1d, block_dim_1d, 0, stream>>>(
            num_points,
            xys,
            depths,
            radii,
            cum_tiles_hit,
            tile_bounds,
            block_width,
            isect_ids_unsorted,
            gaussian_ids_unsorted
        );

    // sort intersections by ascending tile ID and depth with RadixSort
    int32_t max_tile_id = (int32_t)(tile_bounds.x * tile_bounds.y);
    int msb = 32 - __builtin_clz(max_tile_id) + 1;
    // allocate workspace memory
    void *sort_ws = nullptr;
    size_t sort_ws_bytes;
    cub::DeviceRadixSort::SortPairs(
        sort_ws,
        sort_ws_bytes,
        isect_ids_unsorted,
        isect_ids_sorted,
        gaussian_ids_unsorted,
        gaussian_ids_sorted,
        num_intersects,
        0,
        32 + msb,
        stream
    );
    cudaMalloc(&sort_ws, sort_ws_bytes);
    cub::DeviceRadixSort::SortPairs(
        sort_ws,
        sort_ws_bytes,
        isect_ids_unsorted,
        isect_ids_sorted,
        gaussian_ids_unsorted,
        gaussian_ids_sorted,
        num_intersects,
        0,
        32 + msb,
        stream
    );
    cudaFree(sort_ws);

    kernels::get_tile_bin_edges<<<grid_dim_1d, block_dim_1d, 0, stream>>>(
        num_intersects,
        isect_ids_sorted,
        tile_bins
    );

    // free intermediate work spaces
    cudaFree(isect_ids_unsorted);
    cudaFree(isect_ids_sorted);
    cudaFree(gaussian_ids_unsorted);
}
