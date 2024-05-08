from . import jaxsplatlib

from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call

from dataclasses import dataclass


def layout(shape):
    return tuple(range(len(shape) - 1, -1, -1))


class Type:
    shape: tuple[int, ...]
    dtype: ir.Type
    tensor_type: ir.RankedTensorType

    def __init__(self, shape: tuple[int, ...], dtype: ir.Type):
        self.shape = shape
        self.type = dtype
        self.layout = layout(shape)
        self.tensor_type = ir.RankedTensorType.get(shape, self.type)


def _project_gaussians_fwd_lowering(
    ctx: mlir.LoweringRuleContext,
    # input
    means3d,
    scales,
    quats,
    viewmat,
    # desc
    num_points,
    glob_scale,
    fx,
    fy,
    cx,
    cy,
    img_width,
    img_height,
    block_width,
    clip_thresh,
):
    opaque = jaxsplatlib.make_project_gaussians_fwd_descriptor(
        num_points,
        glob_scale,
        fx,
        fy,
        cx,
        cy,
        img_width,
        img_height,
        block_width,
        clip_thresh,
    )

    @dataclass
    class Types:
        in_means3d = Type((num_points, 3), ir.F32Type.get())
        in_scales = Type((num_points, 3), ir.F32Type.get())
        in_quats = Type((num_points, 4), ir.F32Type.get())
        in_viewmat = Type((4, 4), ir.F32Type.get())

        out_covs3d = Type((num_points, 3), ir.F32Type.get())
        out_xys = Type((num_points, 2), ir.F32Type.get())
        out_depths = Type((num_points, 1), ir.F32Type.get())
        out_radii = Type((num_points, 1), ir.IntegerType.get_signless(32))
        out_conics = Type((num_points, 3), ir.F32Type.get())
        out_compensation = Type((num_points, 1), ir.F32Type.get())
        out_num_tiles_hit = Type((num_points, 1), ir.IntegerType.get_unsigned(32))

    t = Types()

    return custom_call(
        "project_gaussians_fwd",
        operands=[means3d, scales, quats, viewmat],
        operand_layouts=[
            t.in_means3d.layout,
            t.in_scales.layout,
            t.in_quats.layout,
            t.in_viewmat.layout,
        ],
        result_types=[
            t.out_covs3d.tensor_type,
            t.out_xys.tensor_type,
            t.out_depths.tensor_type,
            t.out_radii.tensor_type,
            t.out_conics.tensor_type,
            t.out_compensation.tensor_type,
            t.out_num_tiles_hit.tensor_type,
        ],
        result_layouts=[
            t.out_covs3d.layout,
            t.out_xys.layout,
            t.out_depths.layout,
            t.out_radii.layout,
            t.out_conics.layout,
            t.out_compensation.layout,
            t.out_num_tiles_hit.layout,
        ],
        backend_config=opaque,
    ).results
