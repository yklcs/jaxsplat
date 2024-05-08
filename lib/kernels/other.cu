#include "helpers.h"
#include "other.h"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

__global__ void kernels::compute_cov2d_bounds(const unsigned num_points,
                                              const float *__restrict__ covs2d,
                                              float *__restrict__ conics,
                                              float *__restrict__ radii) {
  unsigned row = cg::this_grid().thread_rank();
  if (row >= num_points) {
    return;
  }
  int index = row * 3;
  float3 conic;
  float radius;
  float3 cov2d{(float)covs2d[index], (float)covs2d[index + 1],
               (float)covs2d[index + 2]};
  helpers::compute_cov2d_bounds(cov2d, conic, radius);
  conics[index] = conic.x;
  conics[index + 1] = conic.y;
  conics[index + 2] = conic.z;
  radii[row] = radius;
}
