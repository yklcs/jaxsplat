from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call

from jaxsplat import _jaxsplat
from jaxsplat._project.abstract import (
    ProjectFwdTypes,
    ProjectBwdTypes,
)


def _project_fwd_rule(
    ctx: mlir.LoweringRuleContext,
    #
    mean3ds: ir.Value,
    scales: ir.Value,
    quats: ir.Value,
    viewmat: ir.Value,
    #
    num_points: int,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    block_width: int,
):
    opaque = _jaxsplat.make_descriptor(
        num_points=num_points,
        img_shape=img_shape,
        f=f,
        c=c,
        glob_scale=glob_scale,
        clip_thresh=clip_thresh,
        block_width=block_width,
    )

    t = ProjectFwdTypes(num_points)

    return custom_call(
        "project_fwd",
        operands=[
            mean3ds,
            scales,
            quats,
            viewmat,
        ],
        operand_layouts=[
            t.in_mean3ds.layout(),
            t.in_scales.layout(),
            t.in_quats.layout(),
            t.in_viewmat.layout(),
        ],
        result_types=[
            t.out_cov3ds.ir_tensor_type(),
            t.out_xys.ir_tensor_type(),
            t.out_depths.ir_tensor_type(),
            t.out_radii.ir_tensor_type(),
            t.out_conics.ir_tensor_type(),
            t.out_compensation.ir_tensor_type(),
            t.out_num_tiles_hit.ir_tensor_type(),
            t.out_cum_tiles_hit.ir_tensor_type(),
        ],
        result_layouts=[
            t.out_cov3ds.layout(),
            t.out_xys.layout(),
            t.out_depths.layout(),
            t.out_radii.layout(),
            t.out_conics.layout(),
            t.out_compensation.layout(),
            t.out_num_tiles_hit.layout(),
            t.out_cum_tiles_hit.layout(),
        ],
        backend_config=opaque,
    ).results


def _project_bwd_rule(
    ctx: mlir.LoweringRuleContext,
    #
    mean3ds: ir.Value,
    scales: ir.Value,
    quats: ir.Value,
    viewmat: ir.Value,
    cov3ds: ir.Value,
    xys: ir.Value,
    radii: ir.Value,
    conics: ir.Value,
    compensation: ir.Value,
    v_compensation: ir.Value,
    v_xy: ir.Value,
    v_depth: ir.Value,
    v_conic: ir.Value,
    #
    num_points: int,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    block_width: int,
):
    opaque = _jaxsplat.make_descriptor(
        num_points=num_points,
        img_shape=img_shape,
        f=f,
        c=c,
        glob_scale=glob_scale,
        clip_thresh=clip_thresh,
        block_width=block_width,
    )

    t = ProjectBwdTypes(num_points)

    return custom_call(
        "project_bwd",
        operands=[
            mean3ds,
            scales,
            quats,
            viewmat,
            cov3ds,
            xys,
            radii,
            conics,
            compensation,
            v_compensation,
            v_xy,
            v_depth,
            v_conic,
        ],
        operand_layouts=[
            t.in_mean3ds.layout(),
            t.in_scales.layout(),
            t.in_quats.layout(),
            t.in_viewmat.layout(),
            t.in_cov3ds.layout(),
            t.in_xys.layout(),
            t.in_radii.layout(),
            t.in_conics.layout(),
            t.in_compensation.layout(),
            t.in_v_compensation.layout(),
            t.in_v_xy.layout(),
            t.in_v_depth.layout(),
            t.in_v_conic.layout(),
        ],
        result_types=[
            t.out_v_mean3d.ir_tensor_type(),
            t.out_v_scale.ir_tensor_type(),
            t.out_v_quat.ir_tensor_type(),
            t.out_v_cov2d.ir_tensor_type(),
            t.out_v_cov3d.ir_tensor_type(),
        ],
        result_layouts=[
            t.out_v_mean3d.layout(),
            t.out_v_scale.layout(),
            t.out_v_quat.layout(),
            t.out_v_cov2d.layout(),
            t.out_v_cov3d.layout(),
        ],
        backend_config=opaque,
    ).results
