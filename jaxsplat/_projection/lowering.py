from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call
import jax.numpy as jnp

from dataclasses import dataclass

from jaxsplat import jaxsplatlib
from jaxsplat._types import Type


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
    fx: ir.Value,
    fy: ir.Value,
    cx: ir.Value,
    cy: ir.Value,
    img_width: ir.Value,
    img_height: ir.Value,
    block_width: ir.Value,
    clip_thresh: ir.Value,
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
        in_means3d = Type((num_points, 3), jnp.float32)
        in_scales = Type((num_points, 3), jnp.float32)
        in_quats = Type((num_points, 4), jnp.float32)
        in_viewmat = Type((4, 4), jnp.float32)

        out_covs3d = Type((num_points, 3), jnp.float32)
        out_xys = Type((num_points, 2), jnp.float32)
        out_depths = Type((num_points, 1), jnp.float32)
        out_radii = Type((num_points, 1), jnp.int32)
        out_conics = Type((num_points, 3), jnp.float32)
        out_compensation = Type((num_points, 1), jnp.float32)
        out_num_tiles_hit = Type((num_points, 1), jnp.uint32)

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
            t.out_covs3d.ir_tensor_type,
            t.out_xys.ir_tensor_type,
            t.out_depths.ir_tensor_type,
            t.out_radii.ir_tensor_type,
            t.out_conics.ir_tensor_type,
            t.out_compensation.ir_tensor_type,
            t.out_num_tiles_hit.ir_tensor_type,
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
