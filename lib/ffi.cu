#include "ffi.h"
#include "ops.h"

#include <utility>

py::dict registrations() {
  py::dict dict;
  dict["compute_cov2d_bounds"] = encapsulate_function(compute_cov2d_bounds);
  dict["project_gaussians_fwd"] = encapsulate_function(project_gaussians_fwd);
  dict["project_gaussians_bwd"] = encapsulate_function(project_gaussians_bwd);
  return dict;
}

PYBIND11_MODULE(_jaxsplat, m) {
  m.def("registrations", &registrations);
  m.def("make_cov2d_bounds_descriptor", [](unsigned num_points) {
    return pack_descriptor(Cov2DBoundsDescriptor{num_points});
  });
  m.def(
      "make_project_gaussians_fwd_descriptor",
      [](unsigned num_points, float glob_scale, std::pair<float, float> f,
         std::pair<float, float> c, std::pair<unsigned, unsigned> img_shape,
         unsigned block_width, float clip_thresh) {
        return pack_descriptor(ProjectGaussiansFwdDescriptor{
            num_points, glob_scale, f, c, img_shape, block_width, clip_thresh});
      },
      py::arg("num_points"), py::arg("glob_scale"), py::arg("f"), py::arg("c"),
      py::arg("img_shape"), py::arg("block_width"), py::arg("clip_thresh"));
  m.def(
      "make_project_gaussians_bwd_descriptor",
      // [](unsigned num_points, float glob_scale, float fx, float fy, float cx,
      //    float cy, unsigned img_width, unsigned img_height) {
      [](unsigned num_points, float glob_scale, std::pair<float, float> f,
         std::pair<float, float> c, std::pair<unsigned, unsigned> img_shape) {
        return pack_descriptor(ProjectGaussiansBwdDescriptor{
            // num_points, glob_scale, fx, fy, cx, cy, img_width, img_height});
            num_points, glob_scale, f, c, img_shape});
      },
      py::arg("num_points"), py::arg("glob_scale"), py::arg("f"), py::arg("c"),
      py::arg("img_shape"));
}
