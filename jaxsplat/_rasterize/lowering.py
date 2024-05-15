from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call

from jaxsplat import _jaxsplat
from jaxsplat._rasterize.abstract import (
    RasterizeFwdTypes,
    RasterizeBwdTypes,
)


def _rasterize_fwd_rule(
    ctx: mlir.LoweringRuleContext,
    #
    colors: ir.Value,
    opacities: ir.Value,
    background: ir.Value,
    xys: ir.Value,
    depths: ir.Value,
    radii: ir.Value,
    conics: ir.Value,
    cum_tiles_hit: ir.Value,
    #
    num_points: int,
    img_shape: tuple[int, int],
    block_width: int,
):
    opaque = _jaxsplat.make_descriptor(
        num_points=num_points,
        img_shape=img_shape,
        f=(0.0, 0.0),
        c=(0.0, 0.0),
        glob_scale=0.0,
        clip_thresh=0.0,
        block_width=block_width,
    )

    t = RasterizeFwdTypes(num_points, img_shape)

    return custom_call(
        "rasterize_fwd",
        operands=[
            colors,
            opacities,
            background,
            xys,
            depths,
            radii,
            conics,
            cum_tiles_hit,
        ],
        operand_layouts=[
            t.in_colors.layout(),
            t.in_opacities.layout(),
            t.in_background.layout(),
            t.in_xys.layout(),
            t.in_depths.layout(),
            t.in_radii.layout(),
            t.in_conics.layout(),
            t.in_cum_tiles_hit.layout(),
        ],
        result_types=[
            t.out_final_Ts.ir_tensor_type(),
            t.out_final_idx.ir_tensor_type(),
            t.out_img.ir_tensor_type(),
        ],
        result_layouts=[
            t.out_final_Ts.layout(),
            t.out_final_idx.layout(),
            t.out_img.layout(),
        ],
        backend_config=opaque,
    ).results


def _rasterize_bwd_rule(
    ctx: mlir.LoweringRuleContext,
    #
    colors: ir.Value,
    opacities: ir.Value,
    background: ir.Value,
    xys: ir.Value,
    depths: ir.Value,
    radii: ir.Value,
    conics: ir.Value,
    cum_tiles_hit: ir.Value,
    final_Ts: ir.Value,
    final_idx: ir.Value,
    v_img: ir.Value,
    v_img_alpha: ir.Value,
    #
    num_points: int,
    img_shape: tuple[int, int],
    block_width: int,
):
    opaque = _jaxsplat.make_descriptor(
        num_points=num_points,
        img_shape=img_shape,
        f=(0.0, 0.0),
        c=(0.0, 0.0),
        glob_scale=0.0,
        clip_thresh=0.0,
        block_width=block_width,
    )

    t = RasterizeBwdTypes(num_points, img_shape)

    return custom_call(
        "rasterize_bwd",
        operands=[
            colors,
            opacities,
            background,
            xys,
            depths,
            radii,
            conics,
            cum_tiles_hit,
            final_Ts,
            final_idx,
            v_img,
            v_img_alpha,
        ],
        operand_layouts=[
            t.in_colors.layout(),
            t.in_opacities.layout(),
            t.in_background.layout(),
            t.in_xys.layout(),
            t.in_depths.layout(),
            t.in_radii.layout(),
            t.in_conics.layout(),
            t.in_cum_tiles_hit.layout(),
            t.in_final_Ts.layout(),
            t.in_final_idx.layout(),
            t.in_v_img.layout(),
            t.in_v_img_alpha.layout(),
        ],
        result_types=[
            t.out_v_color.ir_tensor_type(),
            t.out_v_opacity.ir_tensor_type(),
            t.out_v_xy.ir_tensor_type(),
            t.out_v_xy_abs.ir_tensor_type(),
            t.out_v_conic.ir_tensor_type(),
        ],
        result_layouts=[
            t.out_v_color.layout(),
            t.out_v_opacity.layout(),
            t.out_v_xy.layout(),
            t.out_v_xy_abs.layout(),
            t.out_v_conic.layout(),
        ],
        backend_config=opaque,
    ).results
