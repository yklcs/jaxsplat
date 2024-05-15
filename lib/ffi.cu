#include "ffi.h"
#include "ops.h"

py::dict registrations() {
    py::dict dict;

    dict["project_fwd"] = encapsulate_function(ops::project::fwd::xla);
    dict["project_bwd"] = encapsulate_function(ops::project::bwd::xla);
    dict["rasterize_fwd"] = encapsulate_function(ops::rasterize::fwd::xla);
    dict["rasterize_bwd"] = encapsulate_function(ops::rasterize::bwd::xla);

    return dict;
}

py::bytes make_descriptor(
    unsigned num_points,
    std::pair<unsigned, unsigned> img_shape,
    std::pair<float, float> f,
    std::pair<float, float> c,
    float glob_scale,
    float clip_thresh,
    unsigned block_width
) {
    float4 intrins = {f.first, f.second, c.first, c.second};

    // img_shape is in (H,W)
    dim3 img_shape_dim3 = {img_shape.second, img_shape.first, 1};

    const unsigned block_dim_1d = block_width * block_width;
    const unsigned grid_dim_1d = (num_points + block_dim_1d - 1) / block_dim_1d;
    dim3 block_dim_2d = {block_width, block_width, 1};
    dim3 grid_dim_2d = {
        (img_shape_dim3.x + block_width - 1) / block_width,
        (img_shape_dim3.y + block_width - 1) / block_width,
        1
    };

    ops::Descriptor desc = {
        num_points,
        img_shape_dim3,
        intrins,
        glob_scale,
        clip_thresh,
        block_width,
        grid_dim_1d,
        block_dim_1d,
        grid_dim_2d,
        block_dim_2d
    };

    return pack_descriptor(desc);
}

PYBIND11_MODULE(_jaxsplat, m) {
    m.def("registrations", &registrations);

    m.def(
        "make_descriptor",
        make_descriptor,
        py::arg("num_points"),
        py::arg("img_shape"),
        py::arg("f"),
        py::arg("c"),
        py::arg("glob_scale"),
        py::arg("clip_thresh"),
        py::arg("block_width")
    );
}
