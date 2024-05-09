from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call
import jax

from jaxsplat import _jaxsplat
from jaxsplat._projection.abstract import (
    ProjectGaussiansFwdTypes,
    ProjectGaussiansBwdTypes,
)


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


def _project_gaussians_bwd_lowering(
    ctx: mlir.LoweringRuleContext,
    means3d: ir.Value,
    scales: ir.Value,
    quats: ir.Value,
    viewmat: ir.Value,
    cov3d: ir.Value,
    radii: ir.Value,
    conics: ir.Value,
    compensation: ir.Value,
    v_xy: ir.Value,
    v_depth: ir.Value,
    v_conic: ir.Value,
    v_compensation: ir.Value,
    num_points: int,
    glob_scale: float,
    f: tuple[float, float],
    c: tuple[float, float],
    img_shape: tuple[float, float],
):
    opaque = _jaxsplat.make_project_gaussians_bwd_descriptor(
        num_points=num_points, glob_scale=glob_scale, f=f, c=c, img_shape=img_shape
    )

    t = ProjectGaussiansBwdTypes(num_points)

    return custom_call(
        "project_gaussians_bwd",
        operands=[
            means3d,
            scales,
            quats,
            viewmat,
            cov3d,
            radii,
            conics,
            compensation,
            v_xy,
            v_depth,
            v_conic,
            v_compensation,
        ],
        operand_layouts=[
            t.in_means3d.layout(),
            t.in_scales.layout(),
            t.in_quats.layout(),
            t.in_viewmat.layout(),
            t.in_cov3d.layout(),
            t.in_radii.layout(),
            t.in_conics.layout(),
            t.in_compensation.layout(),
            t.in_v_xy.layout(),
            t.in_v_depth.layout(),
            t.in_v_conic.layout(),
            t.in_v_compensation.layout(),
        ],
        result_types=[
            t.out_v_cov2d.ir_tensor_type(),
            t.out_v_cov3d.ir_tensor_type(),
            t.out_v_mean3d.ir_tensor_type(),
            t.out_v_scale.ir_tensor_type(),
            t.out_v_quat.ir_tensor_type(),
        ],
        result_layouts=[
            t.out_v_cov2d.layout(),
            t.out_v_cov3d.layout(),
            t.out_v_mean3d.layout(),
            t.out_v_scale.layout(),
            t.out_v_quat.layout(),
        ],
        backend_config=opaque,
    ).results
