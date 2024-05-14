#include "common.h"
#include "ffi.h"
#include "kernels/kernels.h"
#include "ops.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>

void ops::project::fwd::xla(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
) {
    cudaError_t cuda_err;
    const auto &d = *unpack_descriptor<Descriptor>(opaque, opaque_len);

    const auto tensors = unpack_tensors(stream, d, buffers);
    cudaStreamSynchronize(stream);

    kernels::
        project_gaussians_fwd<<<d.grid_dim_1d, d.block_dim_1d, 0, stream>>>(
            // in
            d.num_points,
            tensors.in.mean3ds,
            tensors.in.scales,
            d.glob_scale,
            tensors.in.quats,
            tensors.in.viewmat,
            d.intrins,
            d.img_shape,
            d.grid_dim_2d,
            d.block_width,
            d.clip_thresh,

            // out
            tensors.out.cov3ds,
            tensors.out.xys,
            tensors.out.depths,
            tensors.out.radii,
            tensors.out.conics,
            tensors.out.compensation,
            tensors.out.num_tiles_hit
        );
    cuda_err = cudaGetLastError();
    CUDA_THROW_IF_ERR(cuda_err);
    cudaStreamSynchronize(stream);

    cumsum(
        stream,
        tensors.out.num_tiles_hit,
        tensors.out.cum_tiles_hit,
        d.num_points
    );
    cuda_err = cudaGetLastError();
    CUDA_THROW_IF_ERR(cuda_err);
    cudaStreamSynchronize(stream);
}

void ops::project::bwd::xla(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
) {
    cudaError_t cuda_err;
    const auto &d = *unpack_descriptor<Descriptor>(opaque, opaque_len);

    const auto &tensors = unpack_tensors(stream, d, buffers);
    cudaStreamSynchronize(stream);

    kernels::
        project_gaussians_bwd<<<d.grid_dim_1d, d.block_dim_1d, 0, stream>>>(
            // in
            d.num_points,
            tensors.in.mean3ds,
            tensors.in.scales,
            d.glob_scale,
            tensors.in.quats,
            tensors.in.viewmat,
            d.intrins,
            d.img_shape,
            tensors.in.cov3ds,
            tensors.in.radii,
            tensors.in.conics,
            tensors.in.compensation,
            tensors.in.v_xy,
            tensors.in.v_depth,
            tensors.in.v_conic,
            tensors.in.v_compensation,

            // out
            tensors.out.v_cov2d,
            tensors.out.v_cov3d,
            tensors.out.v_mean3d,
            tensors.out.v_scale,
            tensors.out.v_quat
        );
    cuda_err = cudaGetLastError();
    CUDA_THROW_IF_ERR(cuda_err);
    cudaStreamSynchronize(stream);
}

void ops::rasterize::fwd::xla(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
) {
    cudaError_t cuda_err;
    const auto &d = *unpack_descriptor<Descriptor>(opaque, opaque_len);

    const auto tensors = unpack_tensors(stream, d, buffers);
    cudaStreamSynchronize(stream);

    sort_and_bin(
        stream,
        d,
        tensors.in.xys,
        tensors.in.depths,
        tensors.in.radii,
        tensors.in.cum_tiles_hit,
        tensors.out.gaussian_ids_sorted,
        tensors.out.tile_bins
    );
    cuda_err = cudaGetLastError();
    CUDA_THROW_IF_ERR(cuda_err);
    cudaStreamSynchronize(stream);

    kernels::rasterize_fwd<<<d.grid_dim_2d, d.block_dim_2d, 0, stream>>>(
        // in
        d.grid_dim_2d,
        d.img_shape,
        tensors.out.gaussian_ids_sorted,
        tensors.out.tile_bins,
        tensors.in.xys,
        tensors.in.conics,
        tensors.in.colors,
        tensors.in.opacities,

        // out
        tensors.out.final_Ts,
        tensors.out.final_idx,
        tensors.out.out_img,
        *tensors.in.background
    );
    cuda_err = cudaGetLastError();
    CUDA_THROW_IF_ERR(cuda_err);
    cudaStreamSynchronize(stream);
};

void ops::rasterize::bwd::xla(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
) {
    cudaError_t cuda_err;
    const auto &d = *unpack_descriptor<Descriptor>(opaque, opaque_len);

    const auto tensors = unpack_tensors(stream, d, buffers);
    cudaStreamSynchronize(stream);

    kernels::rasterize_bwd<<<d.grid_dim_2d, d.block_dim_2d, 0, stream>>>(
        // in
        d.grid_dim_2d,
        d.img_shape,
        tensors.in.gaussian_ids_sorted,
        tensors.in.tile_bins,
        tensors.in.xys,
        tensors.in.conics,
        tensors.in.colors,
        tensors.in.opacities,
        *tensors.in.background,
        tensors.in.final_Ts,
        tensors.in.final_idx,
        tensors.in.v_out_img,
        tensors.in.v_out_img_alpha,

        // out
        tensors.out.v_xy,
        tensors.out.v_xy_abs,
        tensors.out.v_conic,
        tensors.out.v_colors,
        tensors.out.v_opacity
    );
    cuda_err = cudaGetLastError();
    CUDA_THROW_IF_ERR(cuda_err);
    cudaStreamSynchronize(stream);
};

ops::project::fwd::Tensors ops::project::fwd::unpack_tensors(
    cudaStream_t stream,
    const Descriptor &d,
    void **buffers
) {
    Tensors tensors;
    cudaError_t cuda_err;
    std::size_t idx = 0;

    tensors.in.mean3ds = static_cast<float3 *>(buffers[idx++]);
    tensors.in.scales = static_cast<float3 *>(buffers[idx++]);
    tensors.in.quats = static_cast<float4 *>(buffers[idx++]);
    tensors.in.viewmat = static_cast<float *>(buffers[idx++]);

    tensors.out.cov3ds = static_cast<float *>(buffers[idx++]);
    tensors.out.xys = static_cast<float2 *>(buffers[idx++]);
    tensors.out.depths = static_cast<float *>(buffers[idx++]);
    tensors.out.radii = static_cast<int *>(buffers[idx++]);
    tensors.out.conics = static_cast<float3 *>(buffers[idx++]);
    tensors.out.compensation = static_cast<float *>(buffers[idx++]);
    tensors.out.num_tiles_hit = static_cast<int *>(buffers[idx++]);
    tensors.out.cum_tiles_hit = static_cast<int *>(buffers[idx++]);

    cuda_err = cudaMemsetAsync(
        tensors.out.cov3ds,
        0,
        sizeof(*tensors.out.cov3ds) * d.num_points * 6,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        tensors.out.xys,
        0,
        sizeof(*tensors.out.xys) * d.num_points,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        tensors.out.depths,
        0,
        sizeof(*tensors.out.depths) * d.num_points,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        tensors.out.radii,
        0,
        sizeof(*tensors.out.radii) * d.num_points,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        tensors.out.conics,
        0,
        sizeof(*tensors.out.conics) * d.num_points,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        tensors.out.compensation,
        0,
        sizeof(*tensors.out.compensation) * d.num_points,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        tensors.out.num_tiles_hit,
        0,
        sizeof(*tensors.out.num_tiles_hit) * d.num_points,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        tensors.out.cum_tiles_hit,
        0,
        sizeof(*tensors.out.cum_tiles_hit) * d.num_points,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    return tensors;
}

ops::project::bwd::Tensors ops::project::bwd::unpack_tensors(
    cudaStream_t stream,
    const Descriptor &d,
    void **buffers
) {
    Tensors tensors;
    cudaError_t cuda_err;
    std::size_t idx = 0;

    tensors.in.mean3ds = static_cast<float3 *>(buffers[idx++]);
    tensors.in.scales = static_cast<float3 *>(buffers[idx++]);
    tensors.in.quats = static_cast<float4 *>(buffers[idx++]);
    tensors.in.viewmat = static_cast<float *>(buffers[idx++]);
    tensors.in.cov3ds = static_cast<float *>(buffers[idx++]);
    tensors.in.xys = static_cast<float2 *>(buffers[idx++]);
    tensors.in.radii = static_cast<int *>(buffers[idx++]);
    tensors.in.conics = static_cast<float3 *>(buffers[idx++]);
    tensors.in.compensation = static_cast<float *>(buffers[idx++]);
    tensors.in.v_compensation = static_cast<float *>(buffers[idx++]);
    tensors.in.v_xy = static_cast<float2 *>(buffers[idx++]);
    tensors.in.v_depth = static_cast<float *>(buffers[idx++]);
    tensors.in.v_conic = static_cast<float3 *>(buffers[idx++]);

    tensors.out.v_mean3d = static_cast<float3 *>(buffers[idx++]);
    tensors.out.v_scale = static_cast<float3 *>(buffers[idx++]);
    tensors.out.v_quat = static_cast<float4 *>(buffers[idx++]);
    tensors.out.v_cov2d = static_cast<float3 *>(buffers[idx++]);
    tensors.out.v_cov3d = static_cast<float *>(buffers[idx++]);

    cuda_err = cudaMemsetAsync(
        tensors.out.v_cov2d,
        0,
        sizeof(*tensors.out.v_cov2d) * d.num_points,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        tensors.out.v_cov3d,
        0,
        sizeof(*tensors.out.v_cov3d) * d.num_points * 6,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        tensors.out.v_mean3d,
        0,
        sizeof(*tensors.out.v_mean3d) * d.num_points,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        tensors.out.v_scale,
        0,
        sizeof(*tensors.out.v_scale) * d.num_points,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        tensors.out.v_quat,
        0,
        sizeof(*tensors.out.v_quat) * d.num_points,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    return tensors;
}

ops::rasterize::fwd::Tensors ops::rasterize::fwd::unpack_tensors(
    cudaStream_t stream,
    const Descriptor &d,
    void **buffers
) {
    Tensors tensors;
    cudaError_t cuda_err;
    std::size_t idx = 0;

    tensors.in.colors = static_cast<float3 *>(buffers[idx++]);
    tensors.in.opacities = static_cast<float *>(buffers[idx++]);
    tensors.in.background = static_cast<float3 *>(buffers[idx++]);
    tensors.in.xys = static_cast<float2 *>(buffers[idx++]);
    tensors.in.depths = static_cast<float *>(buffers[idx++]);
    tensors.in.radii = static_cast<int *>(buffers[idx++]);
    tensors.in.conics = static_cast<float3 *>(buffers[idx++]);
    tensors.in.cum_tiles_hit = static_cast<int *>(buffers[idx++]);

    tensors.out.gaussian_ids_sorted = static_cast<int *>(buffers[idx++]);
    tensors.out.tile_bins = static_cast<int2 *>(buffers[idx++]);
    tensors.out.final_Ts = static_cast<float *>(buffers[idx++]);
    tensors.out.final_idx = static_cast<int *>(buffers[idx++]);
    tensors.out.out_img = static_cast<float3 *>(buffers[idx++]);

    const auto img_size = d.img_shape.x * d.img_shape.y;

    cuda_err = cudaMemsetAsync(
        tensors.out.gaussian_ids_sorted,
        0,
        sizeof(*tensors.out.gaussian_ids_sorted) * d.num_intersects,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        tensors.out.tile_bins,
        0,
        sizeof(*tensors.out.tile_bins) * d.grid_dim_2d.x * d.grid_dim_2d.y,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        tensors.out.final_Ts,
        0,
        sizeof(*tensors.out.final_Ts) * img_size,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        tensors.out.final_idx,
        0,
        sizeof(*tensors.out.final_idx) * img_size,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        tensors.out.out_img,
        0,
        sizeof(*tensors.out.out_img) * img_size,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    return tensors;
}

ops::rasterize::bwd::Tensors ops::rasterize::bwd::unpack_tensors(
    cudaStream_t stream,
    const Descriptor &d,
    void **buffers
) {
    Tensors tensors;
    cudaError_t cuda_err;
    std::size_t idx = 0;

    tensors.in.colors = static_cast<float3 *>(buffers[idx++]);
    tensors.in.opacities = static_cast<float *>(buffers[idx++]);
    tensors.in.background = static_cast<float3 *>(buffers[idx++]);
    tensors.in.xys = static_cast<float2 *>(buffers[idx++]);
    tensors.in.conics = static_cast<float3 *>(buffers[idx++]);
    tensors.in.gaussian_ids_sorted = static_cast<int *>(buffers[idx++]);
    tensors.in.tile_bins = static_cast<int2 *>(buffers[idx++]);
    tensors.in.final_Ts = static_cast<float *>(buffers[idx++]);
    tensors.in.final_idx = static_cast<int *>(buffers[idx++]);
    tensors.in.v_out_img = static_cast<float3 *>(buffers[idx++]);
    tensors.in.v_out_img_alpha = static_cast<float *>(buffers[idx++]);

    tensors.out.v_colors = static_cast<float3 *>(buffers[idx++]);
    tensors.out.v_opacity = static_cast<float *>(buffers[idx++]);
    tensors.out.v_xy = static_cast<float2 *>(buffers[idx++]);
    tensors.out.v_xy_abs = static_cast<float2 *>(buffers[idx++]);
    tensors.out.v_conic = static_cast<float3 *>(buffers[idx++]);

    cuda_err = cudaMemsetAsync(
        tensors.out.v_xy,
        0,
        sizeof(*tensors.out.v_xy) * d.num_points,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        tensors.out.v_xy_abs,
        0,
        sizeof(*tensors.out.v_xy_abs) * d.num_points,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        tensors.out.v_conic,
        0,
        sizeof(*tensors.out.v_conic) * d.num_points,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        tensors.out.v_colors,
        0,
        sizeof(*tensors.out.v_colors) * d.num_points,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        tensors.out.v_opacity,
        0,
        sizeof(*tensors.out.v_opacity) * d.num_points,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    return tensors;
}

void ops::cumsum(
    cudaStream_t stream,
    const int32_t *input,
    int32_t *output,
    const int num_items
) {
    cudaError_t cuda_err;

    void *sum_ws = nullptr;
    size_t sum_ws_bytes;

    cuda_err = cub::DeviceScan::InclusiveSum(
        sum_ws,
        sum_ws_bytes,
        input,
        output,
        num_items,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMalloc(&sum_ws, sum_ws_bytes);
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cub::DeviceScan::InclusiveSum(
        sum_ws,
        sum_ws_bytes,
        input,
        output,
        num_items,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaFree(sum_ws);
    CUDA_THROW_IF_ERR(cuda_err);
}

void ops::sort_and_bin(
    cudaStream_t stream,
    const Descriptor &d,
    const float2 *xys,
    const float *depths,
    const int *radii,
    const int *cum_tiles_hit,
    int *gaussian_ids_sorted,
    int2 *tile_bins
) {
    cudaError_t cuda_err;

    std::int32_t *gaussian_ids_unsorted;
    std::int64_t *isect_ids_unsorted;
    std::int64_t *isect_ids_sorted;

    cuda_err = cudaMalloc(
        &gaussian_ids_unsorted,
        d.num_intersects * sizeof(*gaussian_ids_unsorted)
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        gaussian_ids_unsorted,
        0,
        d.num_intersects * sizeof(*gaussian_ids_unsorted),
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMalloc(
        &isect_ids_unsorted,
        d.num_intersects * sizeof(*isect_ids_unsorted)
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        isect_ids_unsorted,
        0,
        d.num_intersects * sizeof(*isect_ids_unsorted),
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMalloc(
        &isect_ids_sorted,
        d.num_intersects * sizeof(*isect_ids_sorted)
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMemsetAsync(
        isect_ids_sorted,
        0,
        d.num_intersects * sizeof(*isect_ids_sorted),
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    kernels::map_gaussian_to_intersects<<<
        d.grid_dim_1d,
        d.block_dim_1d,
        0,
        stream>>>(
        d.num_points,
        xys,
        depths,
        radii,
        cum_tiles_hit,
        d.grid_dim_2d,
        d.block_width,
        isect_ids_unsorted,
        gaussian_ids_unsorted
    );
    cuda_err = cudaGetLastError();
    CUDA_THROW_IF_ERR(cuda_err);

    // sort intersections by ascending tile ID and depth with RadixSort
    int32_t max_tile_id = (int32_t)(d.grid_dim_2d.x * d.grid_dim_2d.y);
    int msb = 32 - __builtin_clz(max_tile_id) + 1;
    // allocate workspace memory
    void *sort_ws = nullptr;
    size_t sort_ws_bytes;
    cuda_err = cub::DeviceRadixSort::SortPairs(
        sort_ws,
        sort_ws_bytes,
        isect_ids_unsorted,
        isect_ids_sorted,
        gaussian_ids_unsorted,
        gaussian_ids_sorted,
        d.num_intersects,
        0,
        32 + msb,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaMalloc(&sort_ws, sort_ws_bytes);
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cub::DeviceRadixSort::SortPairs(
        sort_ws,
        sort_ws_bytes,
        isect_ids_unsorted,
        isect_ids_sorted,
        gaussian_ids_unsorted,
        gaussian_ids_sorted,
        d.num_intersects,
        0,
        32 + msb,
        stream
    );
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaFree(sort_ws);
    CUDA_THROW_IF_ERR(cuda_err);

    // printf(
    //     "%d %d\n",
    //     (d.num_intersects + d.block_dim_1d - 1) / d.block_dim_1d,
    //     d.block_dim_1d
    // );
    kernels::get_tile_bin_edges<<<
        (d.num_intersects + d.block_dim_1d - 1) / d.block_dim_1d,
        d.block_dim_1d,
        0,
        stream>>>(d.num_intersects, isect_ids_sorted, tile_bins);
    cuda_err = cudaGetLastError();
    CUDA_THROW_IF_ERR(cuda_err);

    // int test[4];
    // cudaMemcpy(test, tile_bins, sizeof(int) * 4, cudaMemcpyDefault);
    // printf("tile_bins_test %d %d %d %d\n", test[0], test[1], test[2],
    // test[3]);

    cudaStreamSynchronize(stream);

    // free intermediate work spaces
    cuda_err = cudaFree(isect_ids_unsorted);
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaFree(isect_ids_sorted);
    CUDA_THROW_IF_ERR(cuda_err);

    cuda_err = cudaFree(gaussian_ids_unsorted);
    CUDA_THROW_IF_ERR(cuda_err);

    cudaStreamSynchronize(stream);
}
