import jax

from jaxsplat import impl


def project_gaussians_fwd(
    # input
    means3d: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    # desc
    glob_scale: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_width: int,
    img_height: int,
    block_width: int,
    clip_thresh: float,
):
    num_points = means3d.shape[0]

    return impl._project_gaussians_fwd_p.bind(
        means3d,
        scales,
        quats,
        viewmat,
        # desc
        num_points=num_points,
        glob_scale=glob_scale,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        img_width=img_width,
        img_height=img_height,
        block_width=block_width,
        clip_thresh=clip_thresh,
    )


# for _name, _value in jaxsplatlib.registrations().items():
#     xla_client.register_custom_call_target(_name, _value, platform="gpu")


# def compute_cov2d_bounds(covs2d):
#     conics, radii = _compute_cov2d_bounds_p.bind(covs2d)
#     return conics, radii


# def _compute_cov2d_bounds_lowering(ctx, covs2d):
#     type_in = mlir.ir.RankedTensorType(covs2d.type)
#     num_points = type_in.shape[0]

#     type_conics = mlir.ir.RankedTensorType.get((num_points, 3), mlir.ir.F32Type.get())
#     type_radii = mlir.ir.RankedTensorType.get((num_points, 1), mlir.ir.F32Type.get())

#     layout = tuple(range(len(type_in.shape) - 1, -1, -1))

#     opaque = jaxsplatlib.make_cov2d_bounds_descriptor(num_points)

#     return custom_call(
#         "compute_cov2d_bounds",
#         result_types=[type_conics, type_radii],
#         operands=[covs2d],
#         operand_layouts=[layout],
#         result_layouts=[layout, layout],
#         backend_config=opaque,
#     ).results


# def _compute_cov2d_bounds_abstract(covs2d):
#     shape_conics = (covs2d.shape[0], 3)
#     shape_radii = (covs2d.shape[0], 1)
#     dtype = dtypes.canonicalize_dtype(covs2d.dtype)
#     return (ShapedArray(shape_conics, dtype), ShapedArray(shape_radii, dtype))


# _compute_cov2d_bounds_p = core.Primitive("compute_cov2d_bounds")
# _compute_cov2d_bounds_p.multiple_results = True
# _compute_cov2d_bounds_p.def_impl(partial(xla.apply_primitive, _compute_cov2d_bounds_p))
# _compute_cov2d_bounds_p.def_abstract_eval(_compute_cov2d_bounds_abstract)

# mlir.register_lowering(
#     _compute_cov2d_bounds_p,
#     _compute_cov2d_bounds_lowering,
#     platform="gpu",
# )
