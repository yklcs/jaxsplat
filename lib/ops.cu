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
