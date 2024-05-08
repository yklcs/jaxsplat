from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call

from jaxsplat import _jaxsplat
from jaxsplat._projection.abstract import ProjectGaussiansFwdTypes


def _project_gaussians_fwd_lowering(
    ctx: mlir.LoweringRuleContext,
    # input
    means3d: ir.Value,
    scales: ir.Value,
    quats: ir.Value,
    viewmat: ir.Value,
    # desc
    num_points: ir.Value,
    glob_scale: ir.Value,
    f: ir.Value,
    c: ir.Value,
    img_shape: ir.Value,
    block_width: ir.Value,
    clip_thresh: ir.Value,
):
    opaque = _jaxsplat.make_project_gaussians_fwd_descriptor(
        num_points,
        glob_scale,
        f,
        c,
        img_shape,
        block_width,
        clip_thresh,
    )

    t = ProjectGaussiansFwdTypes(num_points)

    return custom_call(
        "project_gaussians_fwd",
        operands=[means3d, scales, quats, viewmat],
        operand_layouts=[
            t.in_means3d.layout(),
            t.in_scales.layout(),
            t.in_quats.layout(),
            t.in_viewmat.layout(),
        ],
        result_types=[
            t.out_covs3d.ir_tensor_type(),
            t.out_xys.ir_tensor_type(),
            t.out_depths.ir_tensor_type(),
            t.out_radii.ir_tensor_type(),
            t.out_conics.ir_tensor_type(),
            t.out_compensation.ir_tensor_type(),
            t.out_num_tiles_hit.ir_tensor_type(),
        ],
        result_layouts=[
            t.out_covs3d.layout(),
            t.out_xys.layout(),
            t.out_depths.layout(),
            t.out_radii.layout(),
            t.out_conics.layout(),
            t.out_compensation.layout(),
            t.out_num_tiles_hit.layout(),
        ],
        backend_config=opaque,
    ).results
