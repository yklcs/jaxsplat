#include "common.h"
#include "ffi.h"
#include "kernels/kernels.h"
#include "ops.h"

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

void compute_cov2d_bounds(cudaStream_t stream, void **buffers,
                          const char *opaque, std::size_t opaque_len) {
  const Cov2DBoundsDescriptor &d =
      *unpack_descriptor<Cov2DBoundsDescriptor>(opaque, opaque_len);

  const float *__restrict__ covs2d = static_cast<float *>(buffers[0]);

  float *__restrict__ conics = static_cast<float *>(buffers[1]);
  float *__restrict__ radii = static_cast<float *>(buffers[2]);

  constexpr unsigned block_dim = 256;
  const unsigned grid_dim =
      std::min((d.num_points + block_dim - 1) / block_dim, MAX_GRID_DIM);

  kernels::compute_cov2d_bounds<<<grid_dim, block_dim, 0, stream>>>(
      d.num_points, covs2d, conics, radii);

  throw_if_error(cudaGetLastError());
}

void project_gaussians_fwd(cudaStream_t stream, void **buffers,
                           const char *opaque, std::size_t opaque_len) {

  const ProjectGaussiansFwdDescriptor &d =
      *unpack_descriptor<ProjectGaussiansFwdDescriptor>(opaque, opaque_len);

  const float *__restrict__ means3d = static_cast<float *>(buffers[0]);
  const float *__restrict__ scales = static_cast<float *>(buffers[1]);
  const float *__restrict__ quats = static_cast<float *>(buffers[2]);
  const float *__restrict__ viewmat = static_cast<float *>(buffers[3]);

  float *__restrict__ covs3d_d = static_cast<float *>(buffers[4]);
  float *__restrict__ xys_d = static_cast<float *>(buffers[5]);
  float *__restrict__ depths_d = static_cast<float *>(buffers[6]);
  int *__restrict__ radii_d = static_cast<int *>(buffers[7]);
  float *__restrict__ conics_d = static_cast<float *>(buffers[8]);
  float *__restrict__ compensation_d = static_cast<float *>(buffers[9]);
  std::int32_t *__restrict__ num_tiles_hit_d =
      static_cast<std::int32_t *>(buffers[10]);

  dim3 img_size;
  img_size.x = d.img_shape.first;
  img_size.y = d.img_shape.second;

  dim3 tile_bounds_dim3;
  tile_bounds_dim3.x = int((img_size.x + d.block_width - 1) / d.block_width);
  tile_bounds_dim3.y = int((img_size.y + d.block_width - 1) / d.block_width);
  tile_bounds_dim3.z = 1;

  float4 intrins = {d.f.first, d.f.second, d.c.first, d.c.second};

  constexpr unsigned block_dim = 256;
  const unsigned grid_dim =
      std::min((d.num_points + block_dim - 1) / block_dim, MAX_GRID_DIM);

  kernels::project_gaussians_fwd<<<grid_dim, block_dim, 0, stream>>>(
      d.num_points, (float3 *)means3d, (float3 *)scales, d.glob_scale,
      (float4 *)quats, viewmat, intrins, img_size, tile_bounds_dim3,
      d.block_width, d.clip_thresh,
      // Outputs.
      covs3d_d, (float2 *)xys_d, depths_d, radii_d, (float3 *)conics_d,
      compensation_d, num_tiles_hit_d);
}

void project_gaussians_bwd(cudaStream_t stream, void **buffers,
                           const char *opaque, std::size_t opaque_len) {

  const ProjectGaussiansBwdDescriptor &d =
      *unpack_descriptor<ProjectGaussiansBwdDescriptor>(opaque, opaque_len);

  const float *__restrict__ means3d = static_cast<float *>(buffers[0]);
  const float *__restrict__ scales = static_cast<float *>(buffers[1]);
  const float *__restrict__ quats = static_cast<float *>(buffers[2]);
  const float *__restrict__ viewmat = static_cast<float *>(buffers[3]);
  const float *__restrict__ cov3d = static_cast<float *>(buffers[4]);
  const int *__restrict__ radii = static_cast<int *>(buffers[5]);
  const float *__restrict__ conics = static_cast<float *>(buffers[6]);
  const float *__restrict__ compensation = static_cast<float *>(buffers[7]);
  const float *__restrict__ v_xy = static_cast<float *>(buffers[8]);
  const float *__restrict__ v_depth = static_cast<float *>(buffers[9]);
  const float *__restrict__ v_conic = static_cast<float *>(buffers[10]);
  const float *__restrict__ v_compensation = static_cast<float *>(buffers[11]);

  float *__restrict__ v_cov2d = static_cast<float *>(buffers[12]);
  float *__restrict__ v_cov3d = static_cast<float *>(buffers[13]);
  float *__restrict__ v_mean3d = static_cast<float *>(buffers[14]);
  float *__restrict__ v_scale = static_cast<float *>(buffers[15]);
  float *__restrict__ v_quat = static_cast<float *>(buffers[16]);

  dim3 img_size = {d.img_shape.first, d.img_shape.second};
  float4 intrins = {d.f.first, d.f.second, d.c.first, d.c.second};
  const unsigned num_cov3d = d.num_points * 6;

  constexpr unsigned block_dim = 256;
  const unsigned grid_dim =
      std::min((d.num_points + block_dim - 1) / block_dim, MAX_GRID_DIM);

  kernels::project_gaussians_bwd<<<grid_dim, block_dim, 0, stream>>>(
      d.num_points, (float3 *)means3d, (float3 *)scales, d.glob_scale,
      (float4 *)quats, viewmat, intrins, img_size, cov3d, radii,
      (float3 *)conics, compensation, (float2 *)v_xy, v_depth,
      (float3 *)v_conic, v_compensation, (float3 *)v_cov2d, v_cov3d,
      (float3 *)v_mean3d, (float3 *)v_scale, (float4 *)v_quat);
}

void rasterize_gaussians_fwd(cudaStream_t stream, void **buffers,
                             const char *opaque, std::size_t opaque_len) {

  const RasterizeGaussiansFwdDescriptor &d =
      *unpack_descriptor<RasterizeGaussiansFwdDescriptor>(opaque, opaque_len);

  const int *__restrict__ gaussian_ids_sorted = static_cast<int *>(buffers[0]);
  const int *__restrict__ tile_bins = static_cast<int *>(buffers[1]);
  const float *__restrict__ xys = static_cast<float *>(buffers[2]);
  const float *__restrict__ conics = static_cast<float *>(buffers[3]);
  const float *__restrict__ colors = static_cast<float *>(buffers[4]);
  const float *__restrict__ opacities = static_cast<float *>(buffers[5]);
  const float *__restrict__ background = static_cast<float *>(buffers[6]);

  float *__restrict__ out_img = static_cast<float *>(buffers[7]);
  float *__restrict__ final_Ts = static_cast<float *>(buffers[8]);
  int *__restrict__ final_idx = static_cast<int *>(buffers[9]);

  dim3 grid_dim = {std::get<0>(d.grid_dim), std::get<1>(d.grid_dim),
                   std::get<2>(d.grid_dim)};

  dim3 block_dim = {std::get<0>(d.block_dim), std::get<1>(d.block_dim),
                    std::get<2>(d.block_dim)};

  dim3 img_shape = {std::get<0>(d.img_shape), std::get<1>(d.img_shape),
                    std::get<2>(d.img_shape)};

  kernels::rasterize_fwd<<<grid_dim, block_dim, 0, stream>>>(
      grid_dim, img_shape, gaussian_ids_sorted, (int2 *)tile_bins,
      (float2 *)xys, (float3 *)conics, (float3 *)colors, opacities, final_Ts,
      final_idx, (float3 *)out_img, *(float3 *)background);
}
