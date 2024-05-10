#include "ffi.h"
#include "render.h"

py::dict registrations() {
    py::dict dict;

    dict["render_fwd"] = encapsulate_function(render::fwd);
    dict["render_bwd"] = encapsulate_function(render::bwd);

    return dict;
}

py::bytes make_render_fwd_descriptor(
    unsigned num_points,
    float glob_scale,
    std::pair<float, float> f,
    std::pair<float, float> c,
    std::pair<unsigned, unsigned> img_shape,
    unsigned block_width,
    float clip_thresh
) {
    float2 f_float2 = {f.first, f.second};
    float2 c_float2 = {c.first, c.second};
    dim3 img_shape_dim3 = {img_shape.first, img_shape.second, 1};

    return pack_descriptor(render::FwdDescriptor{
        num_points,
        glob_scale,
        f_float2,
        c_float2,
        img_shape_dim3,
        block_width,
        clip_thresh
    });
}

py::bytes make_render_bwd_descriptor(
    unsigned num_points,
    float glob_scale,
    std::pair<float, float> f,
    std::pair<float, float> c,
    std::pair<unsigned, unsigned> img_shape,
    unsigned block_width
) {
    float2 f_float2 = {f.first, f.second};
    float2 c_float2 = {c.first, c.second};
    dim3 img_shape_dim3 = {img_shape.first, img_shape.second, 1};

    return pack_descriptor(render::BwdDescriptor{
        num_points,
        glob_scale,
        f_float2,
        c_float2,
        img_shape_dim3,
        block_width
    });
}

PYBIND11_MODULE(_jaxsplat, m) {
    m.def("registrations", &registrations);

    m.def(
        "make_render_fwd_descriptor",
        make_render_fwd_descriptor,
        py::arg("num_points"),
        py::arg("glob_scale"),
        py::arg("f"),
        py::arg("c"),
        py::arg("img_shape"),
        py::arg("block_width"),
        py::arg("clip_thresh")
    );
    m.def(
        "make_render_bwd_descriptor",
        make_render_bwd_descriptor,
        py::arg("num_points"),
        py::arg("glob_scale"),
        py::arg("f"),
        py::arg("c"),
        py::arg("img_shape"),
        py::arg("block_width")
    );
}
