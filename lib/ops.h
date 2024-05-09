#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <tuple>
#include <utility>

// helpers

void compute_cov2d_bounds(cudaStream_t stream, void **buffers,
                          const char *opaque, std::size_t opaque_len);

struct Cov2DBoundsDescriptor {
  unsigned num_points;
};

// projection

void project_gaussians_fwd(cudaStream_t stream, void **buffers,
                           const char *opaque, std::size_t opaque_len);

struct ProjectGaussiansFwdDescriptor {
  unsigned num_points;
  float glob_scale;
  std::pair<float, float> f;
  std::pair<float, float> c;
  std::pair<unsigned, unsigned> img_shape;
  unsigned block_width;
  float clip_thresh;
};

void project_gaussians_bwd(cudaStream_t stream, void **buffers,
                           const char *opaque, std::size_t opaque_len);

struct ProjectGaussiansBwdDescriptor {
  unsigned num_points;
  float glob_scale;
  std::pair<float, float> f;
  std::pair<float, float> c;
  std::pair<unsigned, unsigned> img_shape;
};

// rasterization

void rasterize_gaussians_fwd(cudaStream_t stream, void **buffers,
                             const char *opaque, std::size_t opaque_len);

struct RasterizeGaussiansFwdDescriptor {
  std::tuple<unsigned, unsigned, unsigned> grid_dim;
  std::tuple<unsigned, unsigned, unsigned> block_dim;
  std::tuple<unsigned, unsigned, unsigned> img_shape;
};
